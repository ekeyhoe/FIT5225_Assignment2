"""
The ML Pipeline file contains the ML Processing logic.
This could be used by other Lambda_handlers as well, if needed. 

This file has:
- WildlifePipelineLambda class: 
Contains methods for processing images and videos, 
including detection, classification, and most important tagging.
- calculate_checksum function:
Utility function to calculate file checksum for deduplication.


Instead of using config.yaml, we are using the environment variables for the model paths and other configurations.
Because if we want to change paths or parameters we don't have to build different Docker Image. 

"""
import sys
print(sys.executable)

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from megadetector.detection import run_detector_batch
from labels_map import get_common_name


# we use the environment variables instead of config.yaml for lambda to 
# avoid the need to create another docker image when a new model is added. ---> if any clarification required, ask Mohit. 

s3= boto3.client('s3')

def get_env_config() -> Dict:
    """Load config from environment variables."""
    return {
        'DEVICE': 'cpu',
        'CONF_THRESH': float(os.getenv('CONF_THRESH', '0.05')),
        'SNIP_SIZE': int(os.getenv('SNIP_SIZE', '600')),
        'MODEL_BUCKET': os.getenv('MODEL_BUCKET', 'ecolens-models'),
        'SPECIES_MODEL_KEY': os.getenv('SPECIES_MODEL_KEY', 'model.pt'),
        'MD_MODEL_KEY': os.getenv('MD_MODEL_KEY', 'mdv5a.pt'),
        'TEMP_DIR': '/tmp/ecolens',
    }

# Most of the the Functions below are same from Tejeshvi's code. 

def calculate_checksum(file_path: str) -> str:           #MD5 checksum of file for deduplication.
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_model_from_s3(bucket: str, key: str, local_path: str) -> str:
    """Download model from S3 to /tmp/. Returns local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)
    return local_path

class WildlifePipelineLambda:
    """Lambda-adapted ML pipeline: S3 models, /tmp paths, returns dict."""
    
    def __init__(self):
        self.config = get_env_config()
        self.temp_dir = Path(self.config['TEMP_DIR'])
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Device selection
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        # Species class list (from batch.py)
        self.classes = [
            'Alectura_lathami', 'Antechinus_agilis', 'Bos_taurus',
            'Burhinus_grallarius', 'Canis_familiaris', 'Chalcophaps_longirostris',
            'Colluricincla_harmonica', 'Corcorax_melanorhamphos',
            'Dacelo_novaeguineae', 'Dama_dama', 'Eopsaltria_australis',
            'Felis_catus', 'Geopelia_humeralis', 'Gymnorhina_tibicen',
            'Homo_sapiens', 'Isoodon_macrourus', 'Lepus_europaeus',
            'Macropus_giganteus', 'Menura_novaehollandiae', 'Mus_musculus',
            'Oryctolagus_cuniculus', 'Perameles_nasuta', 'Pitta_versicolor',
            'Rattus', 'Rattus_fuscipes', 'Rattus_rattus', 'Strepera_graculina',
            'Sus_scrofa', 'Tachyglossus_aculeatus', 'Thylogale_stigmatica',
            'Trichosurus_caninus', 'Trichosurus_cunninghami',
            'Trichosurus_vulpecula', 'Varanus_varius', 'Vombatus_ursinus',
            'Vulpes_vulpes', 'Wallabia_bicolor', 'Canis_dingo', 'Capra_hircus',
            'Casuarius_casuarius', 'Heteromyias_cinereifrons',
            'Hypsiprymnodon_moschatus', 'Megapodius_reinwardt',
            'Notamacropus_rufogriseus', 'Orthonyx_spaldingii',
            'Uromys_caudimaculatus'
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
        ])
        
        # Download and load species model
        model_local = str(self.temp_dir / 'model.pt')
        if not Path(model_local).exists():
            download_model_from_s3(
                self.config['MODEL_BUCKET'],
                self.config['SPECIES_MODEL_KEY'],
                model_local
            )
        
        self.species_model = torch.load(model_local, map_location=self.device, weights_only=False)
        self.species_model.eval()
        self.species_model.to(self.device)
    
    def run_megadetector(self, image_path: str) -> Dict:
        """Run MegaDetector on single image. Returns detection data."""
        md_model_local = str(self.temp_dir / 'mdv5a.pt')
        if not Path(md_model_local).exists():
            download_model_from_s3(
                self.config['MODEL_BUCKET'],
                self.config['MD_MODEL_KEY'],
                md_model_local
            )
        
        data = run_detector_batch.load_and_run_detector_batch(
            image_file_names=[image_path],
            model_file=md_model_local
        )
        return data[0] if data else {}
    
    def crop_animals(self, md_data: Dict, original_image_path: str) -> List[str]:
        """Extract animal crops from detections."""
        cropped_paths = []
        
        if not md_data or 'detections' not in md_data:
            return cropped_paths
        
        detections = md_data.get('detections', [])
        img = Image.open(original_image_path).convert('RGB')
        width, height = img.size
        
        crop_num = 0
        for detection in detections:
            conf = detection.get('conf', 0)
            
            if detection.get('category') != '1':
                continue
            
            if conf < self.config['CONF_THRESH']:
                continue
            
            bbox = detection.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            x, y, w, h = bbox
            left = int(x * width)
            top = int(y * height)
            right = int((x + w) * width)
            bottom = int((y + h) * height)
            
            crop = img.crop((left, top, right, bottom))
            resized = crop.resize((self.config['SNIP_SIZE'], self.config['SNIP_SIZE']), Image.BILINEAR)
            
            crop_path = self.temp_dir / f"crop_{crop_num}.jpg"
            resized.save(str(crop_path))
            cropped_paths.append(str(crop_path))
            crop_num += 1
        
        return cropped_paths
    
    @torch.no_grad()
    def classify_crop(self, image_path: str) -> Tuple[str, float]:
        """Classify animal crop. Returns (species_latin, confidence)."""
        img = Image.open(image_path).convert('RGB')
        
        img = self.transform(img)
        img = img.unsqueeze(0)
        img = img.permute(0, 2, 3, 1)
        img = img.to(self.device)
        
        logits = self.species_model(img)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        best_idx = int(np.argmax(probs))
        species_latin = self.classes[best_idx]
        confidence = float(probs[best_idx])
        
        return species_latin, confidence
    
    def map_to_common_name(self, species_latin: str) -> str:
        """Map Latin name to common name for display in detections.
        Uses lightweight labels_map module (no ML dependencies).
        """
        return get_common_name(species_latin)
    
    def process_image(self, image_path: str) -> Dict:
        """Process image: detect → classify → aggregate tags.
        Tags stored as scientific names (Genus_species).
        Returns {
            'file_type': 'image',
            'tags': {'Canis_familiaris': count, ...},
            'detections': [{'latin': ..., 'common': ..., 'confidence': ...}, ...],
        }
        """
        # MegaDetector
        md_data = self.run_megadetector(image_path)
        cropped_paths = self.crop_animals(md_data, image_path)
        
        detections = []
        predictions = []
        
        for crop_path in cropped_paths:
            species_latin, confidence = self.classify_crop(crop_path)
            species_common = self.map_to_common_name(species_latin)
            
            # Store scientific name in predictions (for tag aggregation)
            predictions.append(species_latin)
            
            # But include both in detections (for display)
            detections.append({
                'latin': species_latin,
                'common': species_common,
                'confidence': confidence
            })
        
        # Aggregate by SCIENTIFIC name
        tag_counts = Counter(predictions)
        
        return {
            'file_type': 'image',
            'tags': dict(tag_counts),
            'detections': detections,
        }
    
    def process_video(self, video_path: str) -> Dict:
        """Process video: extract 1 frame/sec, detect/classify per frame.
        Tags stored as scientific names (Genus_species).
        Returns same as process_image but for video.
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_interval = fps  # 1 frame per second
        
        predictions = []
        detections = []
        frame_count = 0
        extracted = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Save frame as temp image
                frame_path = self.temp_dir / f"frame_{extracted}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                # Detect + classify
                md_data = self.run_megadetector(str(frame_path))
                cropped_paths = self.crop_animals(md_data, str(frame_path))
                
                for crop_path in cropped_paths:
                    species_latin, confidence = self.classify_crop(crop_path)
                    species_common = self.map_to_common_name(species_latin)
                    
                    # Store scientific name in predictions
                    predictions.append(species_latin)
                    
                    detections.append({
                        'frame': extracted,
                        'latin': species_latin,
                        'common': species_common,
                        'confidence': confidence
                    })
                
                extracted += 1
            
            frame_count += 1
        
        cap.release()
        
        # Aggregate by SCIENTIFIC name
        tag_counts = Counter(predictions)
        
        return {
            'file_type': 'video',
            'tags': dict(tag_counts),
            'detections': detections,
        }
    
    def extract_first_frame(self, video_path: str) -> str:
        """Extract first frame from video, save to /tmp/, return path.
        Lightweight operation - just frame extraction, no ML.
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Save frame as temp image
        frame_path = self.temp_dir / "first_frame.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        return str(frame_path)