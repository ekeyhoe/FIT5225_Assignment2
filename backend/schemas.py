from pydantic import BaseModel
from typing import Dict, List, Optional


class MediaRecord(BaseModel):
    id: str
    filename: str
    original_path: str
    thumbnail_path: Optional[str] = None
    file_type: str
    checksum: str
    tags: Dict[str, int]


class TagQuery(BaseModel):
    tags: Dict[str, int]


class SpeciesQuery(BaseModel):
    species: str


class ThumbnailQuery(BaseModel):
    thumbnail_path: str


class DeleteRequest(BaseModel):
    paths: List[str]


class ManualTagUpdate(BaseModel):
    paths: List[str]
    tags: List[str]
    operation: int