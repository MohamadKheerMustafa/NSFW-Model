from pathlib import Path
from fastapi import UploadFile, HTTPException
import uuid
from typing import Optional
from PIL import Image, ImageOps
import io
import logging

logger = logging.getLogger("uvicorn.error")

def validate_and_convert_image(content: bytes) -> bytes:
    """Convert any image to verified RGB JPEG with enhanced brightness/contrast normalization"""
    try:
        # First verification pass
        img = Image.open(io.BytesIO(content))
        img.verify()
        
        # Second processing pass with enhanced normalization
        with Image.open(io.BytesIO(content)) as img:
            # Apply EXIF orientation correction
            img = ImageOps.exif_transpose(img)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Apply adaptive auto-contrast for better brightness/contrast normalization
            img = ImageOps.autocontrast(img, cutoff=2)
            
            # Enhance image details
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)  # Slightly enhance sharpness
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=95, subsampling=0)
            return output.getvalue()
            
    except Exception as e:
        logger.error(f"Image conversion failed: {str(e)}")
        return None

def save_uploaded_image(file: UploadFile, label: Optional[int] = None) -> str:
    """Save uploaded file with enhanced validation"""
    base_dir = Path(__file__).resolve().parent.parent
    file_content = file.file.read()
    
    if not file_content:
        raise HTTPException(400, "Empty file content")
    
    converted = validate_and_convert_image(file_content)
    if not converted:
        raise HTTPException(400, "Invalid image content")
    
    # Determine target directory
    folder = "nsfw" if label == 1 else "sfw" if label == 0 else "temp"
    target_dir = base_dir / "dataset" / folder if label is not None else base_dir / "temp"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JPEG
    filename = target_dir / f"{uuid.uuid4()}.jpg"
    with open(filename, "wb") as f:
        f.write(converted)
    
    return str(filename)