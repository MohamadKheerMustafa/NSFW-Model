#!/usr/bin/env python3
import os
import time
import logging
import tensorflow as tf
from pathlib import Path
from datetime import datetime, timedelta
from app.model import (
    model, MODEL_PATH, BATCH_SIZE, INITIAL_LR, FINE_TUNE_LR, FINE_TUNE_AT,
    create_data_pipeline, get_class_weights, delete_non_image_files
)
from tensorflow.keras import optimizers, losses, metrics, callbacks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Training duration in minutes
TRAINING_DURATION_MINUTES = 30
# Checkpoint save interval in minutes
CHECKPOINT_INTERVAL_MINUTES = 5
# Maximum epochs per phase (will be limited by time)
MAX_EPOCHS_PHASE1 = 200
MAX_EPOCHS_PHASE2 = 100
# Time allocation for phases (as a fraction of total time)
PHASE1_TIME_FRACTION = 0.6  # 60% for feature extraction
PHASE2_TIME_FRACTION = 0.4  # 40% for fine-tuning
# Progress update interval in seconds
PROGRESS_UPDATE_INTERVAL = 60

def train_for_duration():
    """Train the model for a fixed duration with time-based checkpoints"""
    try:
        start_time = time.time()
        end_time = start_time + (TRAINING_DURATION_MINUTES * 60)
        next_checkpoint_time = start_time + (CHECKPOINT_INTERVAL_MINUTES * 60)
        
        logger.info(f"Starting timed training for {TRAINING_DURATION_MINUTES} minutes")
        logger.info(f"Training will end at: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
        
        # Prepare dataset
        dataset_dir = Path(__file__).resolve().parent / "dataset"
        delete_non_image_files(dataset_dir)
        
        try:
            train_ds, val_ds, test_ds, steps_per_epoch = create_data_pipeline(dataset_dir)
        except ValueError as e:
            logger.error(f"Dataset error: {str(e)}")
            return
        
        class_weights = get_class_weights(dataset_dir)
        
        # Create a timestamped directory for checkpoints
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(__file__).resolve().parent / "checkpoints" / timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom callback to stop training after the specified duration with enhanced progress reporting
        class TimeLimitCallback(callbacks.Callback):
            def __init__(self, end_time, next_checkpoint_time, checkpoint_dir, phase_name="training"):
                super().__init__()
                self.end_time = end_time
                self.next_checkpoint_time = next_checkpoint_time
                self.checkpoint_dir = checkpoint_dir
                self.best_val_loss = float('inf')
                self.best_weights = None
                self.start_time = time.time()
                self.last_progress_time = self.start_time
                self.phase_name = phase_name
                self.epoch_times = []
                logger.info(f"Phase '{phase_name}' will run until: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
            
            def on_epoch_end(self, epoch, logs=None):
                current_time = time.time()
                elapsed = current_time - self.start_time
                remaining = max(0, self.end_time - current_time)
                
                # Track epoch time for better estimates
                epoch_time = current_time - self.epoch_start_time
                self.epoch_times.append(epoch_time)
                avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
                estimated_epochs_remaining = int(remaining / avg_epoch_time) if avg_epoch_time > 0 else 0
                
                # Progress reporting at regular intervals
                if current_time - self.last_progress_time >= PROGRESS_UPDATE_INTERVAL or current_time >= self.end_time:
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()]) if logs else "No metrics"
                    logger.info(
                        f"{self.phase_name} progress: {elapsed/60:.1f}min elapsed, {remaining/60:.1f}min remaining, "
                        f"~{estimated_epochs_remaining} epochs remaining, {metrics_str}"
                    )
                    self.last_progress_time = current_time
                
                # Save checkpoint at regular intervals
                if current_time >= self.next_checkpoint_time:
                    checkpoint_path = self.checkpoint_dir / f"{self.phase_name}_checkpoint_epoch{epoch}.keras"
                    self.model.save(checkpoint_path)
                    logger.info(f"Saved {self.phase_name} checkpoint at epoch {epoch}")
                    self.next_checkpoint_time = current_time + (CHECKPOINT_INTERVAL_MINUTES * 60)
                
                # Track best model
                if logs and logs.get('val_loss', float('inf')) < self.best_val_loss:
                    self.best_val_loss = logs.get('val_loss')
                    self.best_weights = self.model.get_weights()
                    logger.info(f"New best model with val_loss: {self.best_val_loss:.4f}")
                
                # Stop if time limit reached
                if current_time >= self.end_time:
                    logger.info(f"{self.phase_name} time limit reached after {epoch + 1} epochs")
                    self.model.stop_training = True
            
            def on_train_end(self, logs=None):
                # Save the best model
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    best_model_path = self.checkpoint_dir / f"best_{self.phase_name}_model.keras"
                    self.model.save(best_model_path)
                    logger.info(f"Saved best {self.phase_name} model with val_loss: {self.best_val_loss:.4f}")
                    
                    # Also save to the main model path if this is the final phase
                    if self.phase_name == "fine_tuning":
                        self.model.save(MODEL_PATH)
                        logger.info(f"Saved final model to {MODEL_PATH}")
                
                phase_duration = time.time() - self.start_time
                logger.info(f"{self.phase_name} completed in {phase_duration/60:.2f} minutes")
        
        # Phase 1: Feature extraction
        logger.info("Phase 1: Feature extraction training")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
            loss=losses.BinaryFocalCrossentropy(gamma=2.0),
            metrics=[
                metrics.BinaryAccuracy(name='acc'),
                metrics.AUC(name='auc'),
                metrics.Precision(name='precision'),
                metrics.Recall(name='recall')
            ]
        )
        
        # Calculate time allocation for each phase
        phase1_time_limit = start_time + (TRAINING_DURATION_MINUTES * 60 * PHASE1_TIME_FRACTION)
        phase2_time_limit = end_time  # Use all remaining time for phase 2
        
        # Estimate epochs based on available time
        estimated_time_per_epoch = 60  # Initial estimate: 60 seconds per epoch
        phase1_epochs = min(MAX_EPOCHS_PHASE1, int((phase1_time_limit - start_time) / estimated_time_per_epoch))
        
        # Callbacks for phase 1
        time_limit_callback = TimeLimitCallback(
            end_time=phase1_time_limit,
            next_checkpoint_time=next_checkpoint_time,
            checkpoint_dir=checkpoint_dir,
            phase_name="feature_extraction"
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train phase 1
        phase1_start_time = time.time()
        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=phase1_epochs,
            validation_data=val_ds,
            validation_steps=steps_per_epoch // 4,
            class_weight=class_weights,
            callbacks=[time_limit_callback, reduce_lr, early_stopping],
            verbose=1
        )
        phase1_end_time = time.time()
        
        # Calculate actual time spent in phase 1
        phase1_duration = phase1_end_time - phase1_start_time
        logger.info(f"Phase 1 completed in {phase1_duration/60:.2f} minutes")
        
        # Check if we still have time for phase 2
        remaining_time = end_time - phase1_end_time
        if remaining_time <= 60:  # At least 1 minute needed for phase 2
            logger.info("Insufficient time remaining for phase 2, skipping fine-tuning")
            # Save the best model from phase 1 as the final model
            if time_limit_callback.best_weights is not None:
                model.set_weights(time_limit_callback.best_weights)
                model.save(MODEL_PATH)
                logger.info(f"Saved best feature extraction model as final model")
            return
        
        # Phase 2: Fine-tuning
        logger.info(f"Phase 2: Fine-tuning with {remaining_time/60:.1f} minutes remaining")
        
        # Unfreeze layers for fine-tuning
        for layer in model.layers[1].layers[FINE_TUNE_AT:]:
            layer.trainable = True
        
        model.compile(
            optimizer=optimizers.Adam(
                learning_rate=FINE_TUNE_LR,
                clipnorm=1.0
            ),
            loss=losses.BinaryFocalCrossentropy(gamma=2.0),
            metrics=[
                metrics.BinaryAccuracy(name='acc'),
                metrics.AUC(name='auc'),
                metrics.Precision(name='precision'),
                metrics.Recall(name='recall')
            ]
        )
        
        # Estimate epochs for phase 2 based on phase 1 performance
        if len(time_limit_callback.epoch_times) > 0:
            avg_epoch_time = sum(time_limit_callback.epoch_times) / len(time_limit_callback.epoch_times)
            estimated_phase2_epochs = min(MAX_EPOCHS_PHASE2, int(remaining_time / avg_epoch_time))
            logger.info(f"Estimated {estimated_phase2_epochs} epochs for phase 2 (avg epoch: {avg_epoch_time:.1f}s)")
        else:
            estimated_phase2_epochs = MAX_EPOCHS_PHASE2
        
        # Callbacks for phase 2
        time_limit_callback = TimeLimitCallback(
            end_time=end_time,
            next_checkpoint_time=next_checkpoint_time,
            checkpoint_dir=checkpoint_dir,
            phase_name="fine_tuning"
        )
        
        # Adjust learning rate schedule for phase 2
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,  # More aggressive LR reduction for fine-tuning
            min_lr=1e-8,
            verbose=1
        )
        
        # Train phase 2
        phase2_start_time = time.time()
        model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=estimated_phase2_epochs,  # Will be limited by time callback
            validation_data=val_ds,
            validation_steps=steps_per_epoch // 4,
            class_weight=class_weights,
            callbacks=[time_limit_callback, reduce_lr, early_stopping],
            verbose=1
        )
        phase2_end_time = time.time()
        
        # Calculate actual time spent in phase 2
        phase2_duration = phase2_end_time - phase2_start_time
        logger.info(f"Phase 2 completed in {phase2_duration/60:.2f} minutes")
        
        # Final evaluation
        logger.info("Evaluating final model on test set")
        test_results = model.evaluate(test_ds, verbose=1)
        metrics_names = model.metrics_names
        
        # Print detailed evaluation results
        logger.info("===== Final Model Evaluation Results =====")
        for name, value in zip(metrics_names, test_results):
            logger.info(f"Test {name}: {value:.4f}")
        
        # Calculate and report total training time
        total_duration = time.time() - start_time
        actual_minutes = total_duration / 60
        logger.info(f"===== Training Summary =====")
        logger.info(f"Total training completed in {actual_minutes:.2f} minutes")
        
        # Report if we achieved the target duration
        if abs(actual_minutes - TRAINING_DURATION_MINUTES) <= 1.0:
            logger.info(f"Successfully completed full {TRAINING_DURATION_MINUTES} minute training session")
        elif actual_minutes < TRAINING_DURATION_MINUTES - 1.0:
            logger.info(f"Training completed early by {TRAINING_DURATION_MINUTES - actual_minutes:.2f} minutes")
        else:
            logger.info(f"Training exceeded target by {actual_minutes - TRAINING_DURATION_MINUTES:.2f} minutes")
        
        # Report saved model locations
        logger.info(f"Final model saved to: {MODEL_PATH}")
        logger.info(f"Training checkpoints saved to: {checkpoint_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train_for_duration()