# Media NAS Model

A machine learning system for classifying images as Safe for Work (SFW) or Not Safe for Work (NSFW).

## Features

- Image classification using a MobileNetV2-based neural network with custom attention mechanisms
- FastAPI web service for easy integration
- Automatic model retraining with new data
- Comprehensive image preprocessing and validation

## Project Structure

- `app/main.py`: FastAPI application for handling image classification requests
- `app/model.py`: Machine learning model and training functions
- `app/utils.py`: Utility functions for image processing
- `reorganize_dataset.py`: Script to organize dataset into train/val/test splits
- `download_to_dataset_url.py`: Script to download images from URLs
- `test_retrain.py`: Script to test model retraining

## Installation

```bash
# Clone the repository
git clone https://github.com/MohamadKheerMustafa/NSFW-Model.git
cd media-nas-model

# Install dependencies
pip install -r requirements.txt