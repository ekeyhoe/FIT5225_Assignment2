import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles

from json_db import JsonDB
from ml_pipeline import WildlifePipeline, calculate_checksum
from schemas import (
    TagQuery,
    SpeciesQuery,
    ThumbnailQuery,
    DeleteRequest,
    ManualTagUpdate,
)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Aussie EcoLens Local API")

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")
app.mount("/cropped_images", StaticFiles(directory="cropped_images"), name="cropped_images")

db = JsonDB("./db.json")
pipeline = WildlifePipeline("config.yaml")


@app.get("/")
def root():
    return {
        "message": "Aussie EcoLens Local API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/records")
def get_records():
    return db.all()


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    saved_path = UPLOAD_DIR / f"{file_id}{file_ext}"

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    checksum = calculate_checksum(str(saved_path))
    existing = db.find_by_checksum(checksum)

    if existing:
        saved_path.unlink(missing_ok=True)
        return {
            "message": "Duplicate file detected. Existing record returned.",
            "record": existing
        }

    try:
        result = pipeline.process_image(str(saved_path))
    except Exception as e:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

    record = {
        "id": file_id,
        "filename": file.filename,
        "original_path": str(saved_path),
        "thumbnail_path": result["thumbnail_path"],
        "file_type": result["file_type"],
        "checksum": checksum,
        "tags": result["tags"],
        "detections": result["detections"],
        "cropped_paths": result["cropped_paths"]
    }

    db.insert(record)

    return {
        "message": "File uploaded and processed successfully",
        "record": record
    }


@app.post("/query/tags")
def query_by_tags(query: TagQuery):
    results = db.find_by_tags(query.tags)

    return {
        "query": query.tags,
        "count": len(results),
        "results": results
    }


@app.post("/query/species")
def query_by_species(query: SpeciesQuery):
    results = db.find_by_species(query.species)

    return {
        "species": query.species,
        "count": len(results),
        "results": results
    }


@app.post("/query/thumbnail")
def query_by_thumbnail(query: ThumbnailQuery):
    result = db.find_by_thumbnail(query.thumbnail_path)

    if not result:
        raise HTTPException(status_code=404, detail="No file found for this thumbnail")

    return {
        "thumbnail_path": query.thumbnail_path,
        "original_path": result["original_path"],
        "record": result
    }


@app.post("/query/file-tags")
def query_by_uploaded_file_tags(file: UploadFile = File(...)):
    temp_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    temp_path = UPLOAD_DIR / f"temp_query_{temp_id}{file_ext}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = pipeline.process_image(str(temp_path))
        tags = result["tags"]
        matches = db.find_by_tags(tags)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "detected_tags": tags,
        "count": len(matches),
        "results": matches,
        "note": "Query file was processed temporarily and not added to the database."
    }


@app.post("/tags/update")
def update_tags(update: ManualTagUpdate):
    if update.operation not in [0, 1]:
        raise HTTPException(status_code=400, detail="operation must be 1 for add or 0 for remove")

    updated = db.update_tags(update.paths, update.tags, update.operation)

    return {
        "message": "Tags updated",
        "count": len(updated),
        "updated_records": updated
    }


@app.post("/delete")
def delete_files(request: DeleteRequest):
    deleted = db.delete_by_paths(request.paths)

    for record in deleted:
        Path(record["original_path"]).unlink(missing_ok=True)

        if record.get("thumbnail_path"):
            Path(record["thumbnail_path"]).unlink(missing_ok=True)

        for crop_path in record.get("cropped_paths", []):
            Path(crop_path).unlink(missing_ok=True)

    return {
        "message": "Files deleted",
        "count": len(deleted),
        "deleted_records": deleted
    }