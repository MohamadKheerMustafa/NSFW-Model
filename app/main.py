from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from typing import List, Dict
import shutil
import os
import uuid
from pathlib import Path
from .model import classify_image, retrain_model
from .utils import save_uploaded_image
import logging

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

@app.post("/classify")
async def classify(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> Dict:
    """Classify an image and add to training dataset"""
    try:
        image_path = ""
        try:
            # Save and process image
            image_path = save_uploaded_image(file)
            result = await run_in_threadpool(classify_image, image_path)
            
            # Move to training dataset folder
            dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
            target_folder = "nsfw" if result["nsfw"] else "sfw"
            target_dir = dataset_dir / "train" / target_folder
            target_dir.mkdir(parents=True, exist_ok=True)
            new_path = target_dir / f"{uuid.uuid4()}.jpg"
            shutil.move(image_path, str(new_path))
            
            # Trigger background retraining
            background_tasks.add_task(retrain_model)
            
            return {
                "classification": result,
                "dataset_added": target_folder,
                "file_size": os.path.getsize(new_path)
            }
        except Exception as e:
            if Path(image_path).exists():
                os.remove(image_path)
            raise
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(500, detail=str(e))

@app.post("/classify-multiple")
async def classify_multiple(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
) -> Dict:
    """Batch classification endpoint"""
    results = []
    success_count = 0
    failed_files = []
    
    for file in files:
        try:
            image_path = save_uploaded_image(file)
            result = await run_in_threadpool(classify_image, image_path)
            
            # Move to training dataset
            dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
            target_folder = "nsfw" if result["nsfw"] else "sfw"
            target_dir = dataset_dir / "train" / target_folder
            target_dir.mkdir(parents=True, exist_ok=True)
            new_path = target_dir / f"{uuid.uuid4()}.jpg"
            shutil.move(image_path, str(new_path))
            
            results.append({
                "filename": file.filename,
                **result,
                "saved_path": str(new_path)
            })
            success_count += 1
        except Exception as e:
            failed_files.append({
                "filename": file.filename,
                "error": str(e)
            })
            if 'image_path' in locals() and Path(image_path).exists():
                os.remove(image_path)
    
    if success_count > 0:
        background_tasks.add_task(retrain_model)
    
    return {
        "processed": results,
        "success_count": success_count,
        "failed_files": failed_files
    }

@app.post("/retrain")
async def manual_retrain(background_tasks: BackgroundTasks):
    """Manual retraining trigger"""
    background_tasks.add_task(retrain_model)
    return {"status": "Retraining started"}

# @app.get("/model-info")
# async def model_info() -> Dict:
#     """Get current model information"""
#     return {
#         "model_path": str(MODEL_PATH),
#         "input_size": IMG_SIZE,
#         "last_modified": os.path.getmtime(MODEL_PATH)
#     }

@app.get("/health")
async def health_check() -> Dict:
    """Health check endpoint"""
    return {"status": "healthy"}