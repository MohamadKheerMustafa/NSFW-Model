#!/usr/bin/env python3
import os
import sys
import requests
import concurrent.futures
import time
import uuid
from pathlib import Path
from PIL import Image, ImageOps
import io
import logging
from tqdm import tqdm
import argparse
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).resolve().parent
DATASET_URL_DIR = BASE_DIR / "dataset_url"
DATASET_DIR = BASE_DIR / "dataset"  # Main dataset directory
TIMEOUT = 5  # Timeout for requests in seconds
MAX_WORKERS = 5  # Number of concurrent downloads
RETRIES = 2  # Number of retries for failed downloads
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

# Define file categorization rules
NSFW_FILES = ["urls_hentai.txt", "urls_porn.txt"]
SFW_FILES = ["urls_drawings.txt", "urls_neutral.txt", "urls_sexy.txt"]

def get_file_hash(content: bytes) -> str:
    """Generate a hash for the file content to prevent duplicates"""
    return hashlib.md5(content).hexdigest()

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
        logger.debug(f"Image conversion failed: {str(e)}")
        return None

def is_valid_image(file_path: Path) -> bool:
    """Check if a file is a valid image."""
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify file integrity
            if img.format.lower() not in ['jpeg', 'png', 'gif', 'bmp']:
                return False
            return True
    except (IOError, SyntaxError, ValueError):
        return False

def download_image(url: str, category: str, file_hashes=None, retry_count=0) -> bool:
    """Download an image from a URL and save it to the appropriate category folder in dataset"""
    try:
        # Create a unique filename
        filename = f"{uuid.uuid4()}.jpg"
        target_dir = DATASET_DIR / category
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        
        # Download the image with timeout
        try:
            response = requests.get(url, timeout=TIMEOUT, stream=True)
            response.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            if retry_count < RETRIES:
                # Exponential backoff
                time.sleep(1 * (2 ** retry_count))
                return download_image(url, category, file_hashes, retry_count + 1)
            else:
                logger.debug(f"Failed to download {url} after {RETRIES} retries: {str(e)}")
                return False
        
        # Validate and convert the image
        converted = validate_and_convert_image(response.content)
        if not converted:
            logger.debug(f"Failed to validate/convert image from {url}")
            return False
        
        # Check for duplicates using content hash
        if file_hashes is not None:
            content_hash = get_file_hash(converted)
            if content_hash in file_hashes:
                logger.debug(f"Duplicate image detected from {url}")
                return False
            file_hashes.add(content_hash)
        
        # Save the image
        with open(target_path, "wb") as f:
            f.write(converted)
        
        return True
    except Exception as e:
        logger.debug(f"Error processing {url}: {str(e)}")
        return False

def process_url_file(file_path: Path, category: str, file_hashes=None, limit=None, progress_bar=None):
    """Process a file containing URLs and download the images"""
    success_count = 0
    failure_count = 0
    
    try:
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        if limit:
            urls = urls[:limit]
        
        logger.info(f"Processing {len(urls)} URLs from {file_path}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(download_image, url, category, file_hashes): url for url in urls}
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1
                except Exception as e:
                    logger.debug(f"Error processing {url}: {str(e)}")
                    failure_count += 1
                
                if progress_bar:
                    progress_bar.update(1)
        
        return success_count, failure_count
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return 0, len(urls) if 'urls' in locals() else 0

def scan_existing_images(directory: Path):
    """Scan existing images and create a set of their content hashes"""
    file_hashes = set()
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in VALID_EXTENSIONS and is_valid_image(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        file_hashes.add(get_file_hash(content))
                except Exception as e:
                    logger.debug(f"Error reading {file_path}: {str(e)}")
    return file_hashes

def main():
    """Main function to download images from URL files"""
    global MAX_WORKERS, TIMEOUT
    
    parser = argparse.ArgumentParser(description='Download images from URL files to dataset folder')
    parser.add_argument('--limit', type=int, help='Limit the number of images to download per file')
    parser.add_argument('--category', type=str, choices=['nsfw', 'sfw', 'all'], default='all',
                        help='Specify which category to download (nsfw, sfw, or all)')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help=f'Number of concurrent downloads (default: {MAX_WORKERS})')
    parser.add_argument('--timeout', type=int, default=TIMEOUT,
                        help=f'Timeout for requests in seconds (default: {TIMEOUT})')
    parser.add_argument('--no-duplicate-check', action='store_true',
                        help='Skip duplicate image checking (faster but may download duplicates)')
    args = parser.parse_args()
    
    # Update global constants based on arguments
    MAX_WORKERS = args.workers
    TIMEOUT = args.timeout
    
    start_time = time.time()
    
    # Create dataset_url and dataset directories if they don't exist
    DATASET_URL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Scan existing images to prevent duplicates
    file_hashes = None
    if not args.no_duplicate_check:
        logger.info("Scanning existing images to prevent duplicates...")
        file_hashes = scan_existing_images(DATASET_DIR)
        logger.info(f"Found {len(file_hashes)} existing images")
    
    # Find all URL files
    url_files = []
    
    # Process all text files in the dataset_url directory
    for file_path in DATASET_URL_DIR.glob("*.txt"):
        file_name = file_path.name
        
        # Determine category based on file name
        if file_name in NSFW_FILES and args.category in ['nsfw', 'all']:
            url_files.append((file_path, "nsfw"))
        elif file_name in SFW_FILES and args.category in ['sfw', 'all']:
            url_files.append((file_path, "sfw"))
        # Process files in specific folders
        elif file_path.parent.name == "nsfw" and args.category in ['nsfw', 'all']:
            url_files.append((file_path, "nsfw"))
        elif file_path.parent.name == "sfw" and args.category in ['sfw', 'all']:
            url_files.append((file_path, "sfw"))
        # Process root directory files that don't match the predefined lists
        elif file_path.parent == DATASET_URL_DIR:
            if "nsfw" in file_path.name.lower() and args.category in ['nsfw', 'all']:
                url_files.append((file_path, "nsfw"))
            elif args.category in ['sfw', 'all'] and file_name not in NSFW_FILES:
                url_files.append((file_path, "sfw"))
    
    if not url_files:
        logger.error(f"No URL files found in {DATASET_URL_DIR} for category '{args.category}'")
        return
    
    logger.info(f"Found {len(url_files)} URL files to process")
    
    total_success = 0
    total_failure = 0
    
    # Count total URLs for progress bar
    total_urls = 0
    for file_path, _ in url_files:
        try:
            with open(file_path, 'r') as f:
                file_urls = [line.strip() for line in f if line.strip()]
                if args.limit:
                    file_urls = file_urls[:args.limit]
                total_urls += len(file_urls)
        except Exception as e:
            logger.error(f"Error counting URLs in {file_path}: {str(e)}")
    
    logger.info(f"Total URLs to process: {total_urls}")
    
    try:
        # Process each URL file
        with tqdm(total=total_urls, desc="Downloading images") as progress_bar:
            for file_path, category in url_files:
                logger.info(f"Processing {file_path} for category {category}")
                success, failure = process_url_file(file_path, category, file_hashes, args.limit, progress_bar)
                total_success += success
                total_failure += failure
                logger.info(f"Completed {file_path}: {success} successful, {failure} failed")
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
    finally:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"\nDownload process ran for {duration:.2f} seconds")
        logger.info(f"Total images processed: {total_success + total_failure}")
        logger.info(f"Successfully downloaded: {total_success}")
        logger.info(f"Failed: {total_failure}")

if __name__ == "__main__":
    main()