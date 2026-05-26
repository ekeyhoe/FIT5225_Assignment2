import json
from pathlib import Path
from typing import Dict, List, Optional


class JsonDB:
    def __init__(self, db_file: str):
        self.db_file = Path(db_file)
        if not self.db_file.exists():
            self.db_file.write_text("[]")

    def _read(self) -> List[Dict]:
        with open(self.db_file, "r") as f:
            return json.load(f)

    def _write(self, data: List[Dict]) -> None:
        with open(self.db_file, "w") as f:
            json.dump(data, f, indent=2)

    def all(self) -> List[Dict]:
        return self._read()

    def insert(self, record: Dict) -> None:
        data = self._read()
        data.append(record)
        self._write(data)

    def find_by_checksum(self, checksum: str) -> Optional[Dict]:
        for record in self._read():
            if record["checksum"] == checksum:
                return record
        return None

    def find_by_tags(self, tags: Dict[str, int]) -> List[Dict]:
        results = []

        for record in self._read():
            record_tags = record.get("tags", {})
            match = True

            for tag, required_count in tags.items():
                if record_tags.get(tag, 0) < required_count:
                    match = False
                    break

            if match:
                results.append(record)

        return results

    def find_by_species(self, species: str) -> List[Dict]:
        return [
            record for record in self._read()
            if record.get("tags", {}).get(species, 0) >= 1
        ]

    def find_by_thumbnail(self, thumbnail_path: str) -> Optional[Dict]:
        for record in self._read():
            if record.get("thumbnail_path") == thumbnail_path:
                return record
        return None

    def update_tags(self, paths: List[str], tags: List[str], operation: int) -> List[Dict]:
        data = self._read()
        updated = []

        for record in data:
            if record["original_path"] in paths or record.get("thumbnail_path") in paths:
                record_tags = record.setdefault("tags", {})

                for tag in tags:
                    if operation == 1:
                        record_tags[tag] = record_tags.get(tag, 0) + 1
                    elif operation == 0:
                        record_tags.pop(tag, None)

                updated.append(record)

        self._write(data)
        return updated

    def delete_by_paths(self, paths: List[str]) -> List[Dict]:
        data = self._read()
        deleted = []
        remaining = []

        for record in data:
            if record["original_path"] in paths or record.get("thumbnail_path") in paths:
                deleted.append(record)
            else:
                remaining.append(record)

        self._write(remaining)
        return deleted