import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import threading
import logging
from tensorflow.keras import layers, applications, optimizers, losses, metrics, callbacks
from PIL import Image, ImageOps, ImageEnhance
from typing import Dict

# Configuration
MODEL_DIR = Path(__file__).resolve().parent.parent / "trained_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "media_nas_model.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
model_lock = threading.Lock()
logger = logging.getLogger("uvicorn.error")

# Model parameters
DROPOUT_RATE = 0.4  # Increased dropout for better generalization
INITIAL_LR = 5e-5  # Reduced initial learning rate for more stable training
FINE_TUNE_LR = 1e-6  # Reduced fine-tuning learning rate
AUGMENTATION = True
FINE_TUNE_AT = -30  # Increased fine-tuning layers for better adaptation
MIXUP_ALPHA = 0.2  # Mixup augmentation alpha parameter
CLASS_THRESHOLD = 0.6  # Increased threshold for more conservative NSFW classification
ENSEMBLE_PREDICTIONS = 5  # Number of predictions to ensemble with TTA

def attention_block(x, filters):
    """Enhanced attention mechanism with spatial and channel attention"""
    # Spatial attention with improved feature extraction
    spatial = layers.Conv2D(filters // 4, 1, padding='same')(x)
    spatial = layers.BatchNormalization()(spatial)
    spatial = layers.Activation('relu')(spatial)
    spatial = layers.Conv2D(1, 3, padding='same')(spatial)
    spatial = layers.BatchNormalization()(spatial)
    spatial = layers.Activation('sigmoid')(spatial)
    
    # Channel attention with better feature weighting
    channel = layers.GlobalAveragePooling2D()(x)
    channel = layers.Reshape((1, 1, filters))(channel)
    channel = layers.Dense(filters // 4, activation='relu')(channel)
    channel = layers.BatchNormalization()(channel)
    channel = layers.Dense(filters, activation='sigmoid')(channel)
    
    # Combine attentions with residual connection
    attended = layers.Multiply()([x, spatial])
    attended = layers.Multiply()([attended, channel])
    return layers.Add()([attended, x])

def build_model() -> tf.keras.Model:
    """Build enhanced classification model with multi-scale attention and advanced regularization"""
    base_model = applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet',
        pooling=None
    )
    
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = applications.mobilenet_v2.preprocess_input(inputs)
    
    # Extract features using base model
    x = base_model(x)
    
    # Apply attention mechanism
    attended_features = attention_block(x, x.shape[-1])
    x = attended_features
    
    # Add spatial processing
    x = layers.Conv2D(512, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    # Enhanced dense layers with skip connections and strong regularization
    dense1 = layers.Dense(512)(x)
    dense1 = layers.LayerNormalization()(dense1)
    dense1 = layers.Activation('relu')(dense1)
    dense1 = layers.Dropout(DROPOUT_RATE)(dense1)
    
    dense2 = layers.Dense(256)(dense1)
    dense2 = layers.LayerNormalization()(dense2)
    dense2 = layers.Activation('relu')(dense2)
    dense2 = layers.Dropout(DROPOUT_RATE)(dense2)
    
    # Skip connection with projection
    skip = layers.Dense(256)(dense1)
    x = layers.Add()([dense2, skip])
    
    # Final classification layers
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    activity_regularizer=tf.keras.regularizers.l1(0.01))(x)
    x = layers.Dropout(DROPOUT_RATE/2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    base_model.trainable = False
    
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
    return model

# Initialize model
if MODEL_PATH.exists():
    logger.info("Loading existing model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    logger.info("Creating new model...")
    model = build_model()
    model.save(MODEL_PATH)

def delete_non_image_files(dataset_dir: Path):
    """Clean dataset directory from invalid files with enhanced verification"""
    invalid_count = 0
    converted_count = 0
    total_count = 0
    min_size = 16  # Minimum dimension in pixels
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                continue  # Skip non-image files
                
            total_count += 1
            try:
                # First verification pass
                with Image.open(file_path) as img:
                    img.verify()
                
                # Second processing pass with enhanced verification
                with Image.open(file_path) as img:
                    # Check for corrupt EXIF data
                    try:
                        img = ImageOps.exif_transpose(img)
                    except Exception:
                        # If EXIF processing fails, just use the image as is
                        pass
                    
                    # Check for minimum size
                    width, height = img.size
                    if width < min_size or height < min_size:
                        logger.warning(f"Removing too small image: {file_path} ({width}x{height})")
                        file_path.unlink()
                        invalid_count += 1
                        continue
                    
                    # Check for monochrome or grayscale images
                    if img.mode not in ['RGB', 'RGBA']:
                        img = img.convert('RGB')
                        converted_count += 1
                    elif img.mode == 'RGBA':
                        # Remove alpha channel
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                        img = background
                        converted_count += 1
                    
                    # Check if image is mostly solid color (likely corrupt or useless)
                    img_array = np.array(img.resize((32, 32)))  # Resize for faster processing
                    std_val = np.std(img_array)
                    if std_val < 10:  # Very low standard deviation indicates mostly solid color
                        logger.warning(f"Removing low-variance image: {file_path} (std: {std_val:.2f})")
                        file_path.unlink()
                        invalid_count += 1
                        continue
                    
                    # Convert to jpg if needed
                    if file_path.suffix.lower() != '.jpg':
                        jpg_path = file_path.with_suffix('.jpg')
                        img.save(jpg_path, 'JPEG', quality=95, subsampling=0)
                        file_path.unlink()
                        converted_count += 1
                    elif img.mode != 'RGB':  # Save again if we converted the mode
                        img.save(file_path, 'JPEG', quality=95, subsampling=0)
                        converted_count += 1
            except Exception as e:
                logger.warning(f"Removing invalid file: {file_path} - {str(e)}")
                if file_path.exists():
                    file_path.unlink()
                invalid_count += 1
    
    logger.info(f"Dataset cleaning complete: {total_count} files processed, {invalid_count} invalid files removed, {converted_count} files converted")
    return total_count - invalid_count  # Return number of valid files

def reorganize_dataset(dataset_dir: Path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) -> None:
    """Reorganize dataset into train/val/test splits while maintaining class distribution"""
    import random
    import shutil

    # Create split directories
    splits = ['train', 'val', 'test']
    classes = ['nsfw', 'sfw']
    
    for split in splits:
        for cls in classes:
            (dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for cls in classes:
        source_dir = dataset_dir / cls
        if not source_dir.exists():
            continue
            
        # Get all images
        images = list(source_dir.glob('*.jpg'))
        random.shuffle(images)
        
        # Calculate split sizes
        total = len(images)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        # Split images
        train_files = images[:train_size]
        val_files = images[train_size:train_size + val_size]
        test_files = images[train_size + val_size:]
        
        # Move files
        for files, split in zip([train_files, val_files, test_files], splits):
            target_dir = dataset_dir / split / cls
            for file in files:
                shutil.move(str(file), str(target_dir / file.name))

def create_data_pipeline(dataset_dir: Path) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int]:
    """Create optimized data pipeline with train/validation/test splits and enhanced augmentation"""
    # Verify split directories exist
    splits = ['train', 'val', 'test']
    for split in splits:
        if not (dataset_dir / split).exists():
            raise ValueError(f"Dataset split directory '{split}' must exist")
    
    # Configure enhanced dataset augmentation with improved brightness/contrast normalization
    augmentation = tf.keras.Sequential([
        # Geometric transformations
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.3),  # Increased rotation
        layers.RandomZoom(0.3),      # Increased zoom
        layers.RandomTranslation(0.1, 0.1),  # Added translation
        
        # Enhanced brightness/contrast normalization
        layers.RandomContrast(0.4),  # Increased contrast variation
        layers.RandomBrightness(0.4),  # Increased brightness variation
        
        # Noise for robustness
        layers.GaussianNoise(0.08)   # Increased noise
    ])
    
    # Custom function for additional color augmentation
    def color_augmentation(x, y):
        # Apply random saturation and hue adjustments using tf.image operations
        x = tf.image.random_saturation(x, lower=0.7, upper=1.3)  # Increased saturation range
        x = tf.image.random_hue(x, max_delta=0.15)  # Increased hue range
        
        # Process each image in the batch individually for jpeg quality
        # since random_jpeg_quality doesn't support batched inputs
        def apply_jpeg_quality(img):
            return tf.image.random_jpeg_quality(img, 70, 100)
            
        # Apply the function to each image in the batch
        x = tf.map_fn(apply_jpeg_quality, x)
        return x, y
    
    # Mixup augmentation implementation with robust batch size handling
    def mixup(ds):
        # Create two datasets with the same content but different shuffling
        ds_one = ds.shuffle(1000, reshuffle_each_iteration=True)
        ds_two = ds.shuffle(1000, reshuffle_each_iteration=True)
        
        # Zip the datasets
        ds_zip = tf.data.Dataset.zip((ds_one, ds_two))
        
        def apply_mixup(images_labels_1, images_labels_2):
            images1, labels1 = images_labels_1
            images2, labels2 = images_labels_2
            
            # Check if shapes are compatible for mixing
            shape1 = tf.shape(images1)
            shape2 = tf.shape(images2)
            
            # Safety check: ensure both batches have the same batch size
            # This prevents shape mismatches when the last batch is smaller
            if shape1[0] != shape2[0]:
                # If batch sizes don't match, just return the first batch unchanged
                # This handles the edge case of the last batch in the dataset
                return images1, labels1
                
            # Get the actual batch size from the current batch
            # This handles variable batch sizes correctly
            batch_size = shape1[0]
            
            # Skip mixup for very small batches to avoid shape issues
            # This is a safety measure for when batch sizes are too small
            if batch_size < 2:
                return images1, labels1
            
            # Generate mixup coefficient from beta distribution
            alpha_param = tf.ones(batch_size) * MIXUP_ALPHA
            beta_param = tf.ones(batch_size) * MIXUP_ALPHA
            
            # Use tf.random.stateless_beta or tf.random.stateless_uniform as fallback
            try:
                # Try using stateless_beta if available (TF 2.4+)
                seed = tf.cast(tf.stack([tf.timestamp() * 1000000, tf.timestamp() * 1000000]), tf.int32)
                gamma = tf.random.stateless_beta(alpha_param, beta_param, seed)
            except AttributeError:
                # Fallback to uniform distribution if beta is not available
                gamma = tf.random.uniform(shape=[batch_size], minval=0.2, maxval=0.8)
            
            # Reshape gamma for proper broadcasting - dynamically based on actual batch size
            gamma_images = tf.reshape(gamma, [batch_size, 1, 1, 1])
            
            # Mix images with dynamic shape handling
            mixed_images = gamma_images * images1 + (1 - gamma_images) * images2
            
            # Reshape gamma for labels - dynamically based on actual batch size
            gamma_labels = tf.reshape(gamma, [batch_size, 1])
            
            # Mix labels
            mixed_labels = gamma_labels * labels1 + (1 - gamma_labels) * labels2
            
            return mixed_images, mixed_labels
        
        return ds_zip.map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)
    
    def configure_dataset(ds, is_training=False):
        # Apply preprocessing before batching
        ds = ds.map(
            lambda x, y: (applications.mobilenet_v2.preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch the dataset
        ds = ds.batch(BATCH_SIZE)
        
        if is_training and AUGMENTATION:
            # Apply standard augmentation
            ds = ds.map(
                lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Apply custom color augmentation
            ds = ds.map(
                color_augmentation,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Apply mixup augmentation with error handling
            try:
                ds = mixup(ds)
            except Exception as e:
                # Fallback to non-mixup dataset if mixup fails
                logger.warning(f"Mixup augmentation failed, using standard augmentation instead: {str(e)}")
                # Continue without mixup
        
        return ds.cache().prefetch(tf.data.AUTOTUNE)
    
    # Load datasets from split directories
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / 'train',
        image_size=IMG_SIZE,
        batch_size=None,
        label_mode='binary',
        shuffle=True,
        seed=42,
        interpolation='bilinear',
        follow_links=False
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / 'val',
        image_size=IMG_SIZE,
        batch_size=None,
        label_mode='binary',
        shuffle=True,
        seed=42,
        interpolation='bilinear',
        follow_links=False
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / 'test',
        image_size=IMG_SIZE,
        batch_size=None,
        label_mode='binary',
        shuffle=False,
        seed=42,
        interpolation='bilinear',
        follow_links=False
    )
    
    # Configure datasets
    train_ds = configure_dataset(train_ds, is_training=True)
    val_ds = configure_dataset(val_ds)
    test_ds = configure_dataset(test_ds)
    
    # Calculate steps per epoch
    total_train_images = sum(
        len(list((dataset_dir / 'train' / cls).glob('*.jpg')))
        for cls in ['nsfw', 'sfw']
    )
    steps_per_epoch = max(1, total_train_images // BATCH_SIZE)
    
    # Add repeat to ensure we don't run out of data during training
    # But make it finite for test_retrain.py by setting a large but finite repeat count
    train_ds = train_ds.repeat(100)  # Repeat 100 times should be more than enough for training
    
    return train_ds, val_ds, test_ds, steps_per_epoch

def get_class_weights(dataset_dir: Path) -> Dict[int, float]:
    """Calculate class weights for imbalanced data with improved balancing"""
    # Calculate weights based on training split for more accurate weighting
    train_nsfw_count = len(list((dataset_dir / 'train' / 'nsfw').glob('*.jpg')))
    train_sfw_count = len(list((dataset_dir / 'train' / 'sfw').glob('*.jpg')))
    
    # Fallback to root directory if train split doesn't exist
    if train_nsfw_count == 0 and train_sfw_count == 0:
        train_nsfw_count = len(list((dataset_dir / 'nsfw').glob('*.jpg')))
        train_sfw_count = len(list((dataset_dir / 'sfw').glob('*.jpg')))
    
    total = train_nsfw_count + train_sfw_count
    
    # Ensure we don't divide by zero and apply smoothing
    train_nsfw_count = max(train_nsfw_count, 1)
    train_sfw_count = max(train_sfw_count, 1)
    
    # Calculate balanced weights with smoothing factor
    weights = {
        0: total / (2 * train_sfw_count),  # SFW class weight
        1: total / (2 * train_nsfw_count)   # NSFW class weight
    }
    
    # Log the class distribution and weights
    logger.info(f"Class distribution - SFW: {train_sfw_count}, NSFW: {train_nsfw_count}")
    logger.info(f"Class weights - SFW: {weights[0]:.4f}, NSFW: {weights[1]:.4f}")
    
    return weights

def classify_image(image_path: str) -> Dict:
    """Classify image with ensemble prediction and test-time augmentation"""
    with model_lock:
        try:
            img = Image.open(image_path)
            # Apply EXIF orientation correction
            img = ImageOps.exif_transpose(img).convert('RGB')
            
            # Convert to numpy array for advanced processing
            img_array = np.array(img)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better local contrast
            # Convert to LAB color space for better perceptual processing
            try:
                import cv2
                # Convert RGB to LAB color space
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                # Split the LAB channels
                l, a, b = cv2.split(lab)
                # Apply CLAHE to L-channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                # Merge the CLAHE enhanced L-channel with the original A and B channels
                enhanced_lab = cv2.merge((cl, a, b))
                # Convert back to RGB color space
                enhanced_img_array = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
                img = Image.fromarray(enhanced_img_array)
            except ImportError:
                # Fallback to PIL-based enhancement if OpenCV is not available
                # Apply adaptive auto-contrast for better brightness/contrast normalization
                img = ImageOps.autocontrast(img, cutoff=2)
                # Enhance image details
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.2)  # Slightly enhance sharpness
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.1)  # Slightly enhance contrast
            
            # Resize to model input size with high-quality resampling
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            
            # Implement ensemble prediction with test-time augmentation (TTA)
            predictions = []
            
            # Original image prediction
            img_array = applications.mobilenet_v2.preprocess_input(
                tf.keras.preprocessing.image.img_to_array(img)
            )
            img_array = np.expand_dims(img_array, axis=0)
            predictions.append(float(model.predict(img_array, verbose=0)[0][0]))
            
            # Test-time augmentations for ensemble prediction
            for _ in range(ENSEMBLE_PREDICTIONS - 1):
                # Apply random augmentations similar to training
                aug_img = img.copy()
                
                # Random horizontal flip
                if np.random.random() > 0.5:
                    aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
                
                # Random brightness/contrast adjustment
                enhancer = ImageEnhance.Brightness(aug_img)
                aug_img = enhancer.enhance(np.random.uniform(0.8, 1.2))
                
                enhancer = ImageEnhance.Contrast(aug_img)
                aug_img = enhancer.enhance(np.random.uniform(0.8, 1.2))
                
                # Convert to array and predict
                aug_array = applications.mobilenet_v2.preprocess_input(
                    tf.keras.preprocessing.image.img_to_array(aug_img)
                )
                aug_array = np.expand_dims(aug_array, axis=0)
                predictions.append(float(model.predict(aug_array, verbose=0)[0][0]))
            
            # Average the ensemble predictions
            confidence = sum(predictions) / len(predictions)
            
            # Apply smoothing to reduce overconfidence
            confidence = 0.9 * confidence + 0.05  # Slight regression to the mean
            
            # Log prediction details for debugging
            logger.info(f"Ensemble predictions: {predictions}, final confidence: {confidence:.4f}")
            
            return {
                "nsfw": confidence > CLASS_THRESHOLD,  # Use the configurable threshold
                "confidence": confidence,
                "threshold": CLASS_THRESHOLD,
                "predictions": predictions,  # Include individual predictions for analysis
                "error": None
            }
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return {
                "nsfw": False,
                "confidence": None,
                "error": str(e)
            }

def retrain_model():
    """Enhanced retraining procedure with robust error handling and proper dataset management"""
    global model
    with model_lock:
        try:
            logger.info("Starting enhanced retraining...")
            dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
            delete_non_image_files(dataset_dir)
            
            try:
                train_ds, val_ds, test_ds, steps_per_epoch = create_data_pipeline(dataset_dir)
            except ValueError as e:
                logger.error(f"Dataset error: {str(e)}")
                raise
            
            class_weights = get_class_weights(dataset_dir)
            
            # Phase 1: Feature extraction with cosine decay learning rate
            model = tf.keras.models.load_model(MODEL_PATH)
            initial_epochs = 1  # Increased epochs for better feature extraction
            
            logger.info("Dataset split complete:")
            logger.info(f"Training samples: {len(train_ds) * BATCH_SIZE}")
            logger.info(f"Validation samples: {len(val_ds) * BATCH_SIZE}")
            logger.info(f"Test samples: {len(test_ds) * BATCH_SIZE}")

            
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
            
            # Enhanced callbacks with more patience
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=7,  # Increased patience
                verbose=1,
                restore_best_weights=True,
                min_delta=0.001
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=4,  # Increased patience
                min_lr=1e-7,
                verbose=1
            )
            
            checkpoint = callbacks.ModelCheckpoint(
                MODEL_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            
            # Custom callback for more comprehensive validation
            class FullValidationCallback(callbacks.Callback):
                def __init__(self, validation_data, log_dir=None):
                    super().__init__()
                    self.validation_data = validation_data
                    self.best_val_loss = float('inf')
                    self.log_dir = log_dir
                    
                def on_epoch_end(self, epoch, logs=None):
                    if epoch % 2 == 0:  # Validate every 2 epochs to save time
                        logger.info("Running full validation set evaluation...")
                        results = self.model.evaluate(self.validation_data, verbose=1)
                        metrics_names = self.model.metrics_names
                        
                        for name, value in zip(metrics_names, results):
                            logs[f'full_val_{name}'] = value
                            
                        logger.info(f"Full validation results: {dict(zip(metrics_names, results))}")
                        
                        # Track best model
                        if results[0] < self.best_val_loss:  # results[0] is val_loss
                            self.best_val_loss = results[0]
                            logger.info(f"New best model with full val_loss: {self.best_val_loss:.4f}")
            
            # Add full validation callback
            full_val_callback = FullValidationCallback(val_ds)
            
            # Phase 1 training with increased steps and better validation
            history = model.fit(
                train_ds,
                steps_per_epoch=steps_per_epoch,
                epochs=initial_epochs,
                validation_data=val_ds,
                validation_steps=steps_per_epoch // 4,  # Increased validation steps
                class_weight=class_weights,
                callbacks=[early_stop, reduce_lr, checkpoint, full_val_callback],
                verbose=1
            )
            
            # Phase 2: Progressive fine-tuning with gradual unfreezing
            fine_tune_epochs = 12  # Further increased fine-tuning epochs
            
            # Get base model layers
            base_model_layers = model.layers[1].layers
            total_layers = len(base_model_layers)
            
            # Progressive unfreezing in stages
            unfreeze_stages = [
                (FINE_TUNE_AT, 4),       # First stage: unfreeze last few layers
                (FINE_TUNE_AT - 10, 4),  # Second stage: unfreeze more layers
                (FINE_TUNE_AT - 20, 4)   # Third stage: unfreeze even more layers
            ]
            
            logger.info("Starting progressive fine-tuning with gradual unfreezing")
            
            for stage, (unfreeze_at, epochs) in enumerate(unfreeze_stages):
                # Ensure index is valid
                unfreeze_idx = max(0, total_layers + unfreeze_at) if unfreeze_at < 0 else unfreeze_at
                
                # Set trainable status
                for i, layer in enumerate(base_model_layers):
                    layer.trainable = (i >= unfreeze_idx)
                
                # Count trainable layers
                trainable_count = sum(1 for layer in base_model_layers if layer.trainable)
                logger.info(f"Stage {stage+1}: Unfreezing from layer {unfreeze_idx}, {trainable_count} trainable layers")
                
                # Compile with appropriate learning rate (decreasing for later stages)
                stage_lr = FINE_TUNE_LR * (0.5 ** stage)  # Decrease LR for each stage
                
                model.compile(
                    optimizer=optimizers.Adam(
                        learning_rate=stage_lr,
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
                
                # Fine-tuning for this stage
                logger.info(f"Fine-tuning stage {stage+1} with learning rate {stage_lr:.2e}")
                model.fit(
                    train_ds,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=val_ds,
                    validation_steps=steps_per_epoch // 4,
                    class_weight=class_weights,
                    callbacks=[reduce_lr, checkpoint, full_val_callback],
                    verbose=1
                )
            
            # Final evaluation on test set
            logger.info("Evaluating final model on test set")
            test_results = model.evaluate(test_ds, verbose=1)
            test_metrics = dict(zip(model.metrics_names, test_results))
            logger.info(f"Test set evaluation: {test_metrics}")
            
            logger.info("Retraining completed successfully")
            
        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}")
            raise