"""
Label mapping: Scientific name ↔ Common name
Lightweight module for query handlers (no ML dependencies)
"""

from pathlib import Path
from typing import Dict


def load_label_map() -> Dict[str, str]:
    """Load Latin → common name mapping from labels.txt.
    Format: UUID;class;order;family;genus;species;common_name
    Returns {Genus_species: common_name}
    
    Used by query handlers for display/search without loading ML models.
    """
    label_map = {}
    labels_path = Path(__file__).parent / 'labels.txt'
    
    if not labels_path.exists():
        return {}
    
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 7:
                genus = parts[4]
                species = parts[5]
                common = parts[6]
                
                # Key: scientific name
                # Value: common name (for display)
                latin_key = f"{genus.capitalize()}_{species}"
                label_map[latin_key] = common.lower()
    
    return label_map


def get_common_name(scientific_name: str) -> str:
    """Get common name for a scientific species name.
    Falls back to lowercase scientific name if not found.
    """
    label_map = load_label_map()
    return label_map.get(scientific_name, scientific_name.lower())


def map_tags_to_common(tags: Dict[str, int]) -> Dict[str, int]:
    """Convert tags from scientific names to common names.
    Input: {'Canis_familiaris': 3, 'Phascolarctos_cinereus': 1}
    Output: {'dingo': 3, 'koala': 1}
    """
    label_map = load_label_map()
    return {
        label_map.get(scientific, scientific.lower()): count
        for scientific, count in tags.items()
    }