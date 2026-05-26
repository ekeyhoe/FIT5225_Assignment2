import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from megadetector.detection import run_detector_batch


def read_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def calculate_checksum(file_path: str) -> str:
    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def create_thumbnail(image_path: str, output_dir: str = "./thumbnails") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((300, 300))

    thumbnail_file = output_path / f"{Path(image_path).stem}_thumb.jpg"
    img.save(thumbnail_file, "JPEG", quality=80)

    return str(thumbnail_file)


class WildlifePipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = read_yaml(config_path)

        self.md_model_path = self.config["MEGADETECTOR_MODEL"]
        self.species_model_path = self.config["SPECIES_MODEL"]
        self.snip_dir = Path(self.config["SNIP_DIR"])
        self.snip_dir.mkdir(parents=True, exist_ok=True)

        self.conf_thresh = float(self.config["LOWER_CONF"])
        self.snip_size = int(self.config["SNIP_SIZE"])

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

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

        self.species_model = torch.load(
            self.species_model_path,
            map_location=self.device,
            weights_only=False
        )

        self.species_model.eval()
        self.species_model.to(self.device)

    def run_megadetector(self, image_paths: List[str]) -> List[Dict]:
        data = run_detector_batch.load_and_run_detector_batch(
            image_file_names=image_paths,
            model_file=self.md_model_path
        )
        return data

    def crop_animals(self, md_data: List[Dict]) -> List[str]:
        cropped_paths = []

        for entry in md_data:
            img_path = entry["file"]

            if not Path(img_path).exists():
                continue

            detections = entry.get("detections", [])
            img = Image.open(img_path).convert("RGB")
            width, height = img.size

            crop_num = 0

            for detection in detections:
                conf = detection["conf"]

                if detection["category"] != "1":
                    continue

                if conf < self.conf_thresh:
                    continue

                x, y, w, h = detection["bbox"]

                left = int(x * width)
                top = int(y * height)
                right = int((x + w) * width)
                bottom = int((y + h) * height)

                crop = img.crop((left, top, right, bottom))
                resized = crop.resize((self.snip_size, self.snip_size), Image.BILINEAR)

                out_name = f"{Path(img_path).stem}-{crop_num}{Path(img_path).suffix}"
                out_path = self.snip_dir / out_name

                resized.save(out_path)
                cropped_paths.append(str(out_path))

                crop_num += 1

        return cropped_paths

    @torch.no_grad()
    def classify_crop(self, image_path: str) -> Dict:
        img = Image.open(image_path).convert("RGB")

        img = self.transform(img)
        img = img.unsqueeze(0)
        img = img.permute(0, 2, 3, 1)
        img = img.to(self.device)

        logits = self.species_model(img)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        best_idx = int(np.argmax(probs))

        return {
            "species": self.classes[best_idx],
            "confidence": float(probs[best_idx])
        }

    def process_image(self, image_path: str) -> Dict:
        md_data = self.run_megadetector([image_path])
        cropped_paths = self.crop_animals(md_data)

        predictions = []

        for crop_path in cropped_paths:
            prediction = self.classify_crop(crop_path)
            predictions.append(prediction)

        tag_counts = Counter([p["species"] for p in predictions])

        thumbnail_path = create_thumbnail(image_path)

        return {
            "file_type": "image",
            "thumbnail_path": thumbnail_path,
            "tags": dict(tag_counts),
            "detections": predictions,
            "cropped_paths": cropped_paths
        }