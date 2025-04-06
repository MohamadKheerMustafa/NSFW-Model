#!/usr/bin/env python3

import logging
from app.model import retrain_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("Starting model retraining test...")
    try:
        retrain_model()
        print("Retraining completed successfully!")
    except Exception as e:
        print(f"Retraining failed with error: {str(e)}")