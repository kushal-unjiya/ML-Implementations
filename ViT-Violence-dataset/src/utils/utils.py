import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# Import custom model classes
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_01 import VisionTransformer, Projection, TransformerEncoder, MLP, MHA

def rebuild_model_and_load_weights(model_path):
    """Rebuild the model architecture and load weights"""
    try:
        # Model parameters from training script
        N_HEADS = 3
        CLASSES = 2
        CHANNELS = 3
        DROPOUT = 0.1
        IMG_SIZE = 224
        PATCH_SIZE = 16
        EMBED_SIZE = 256
        MLP_HIDDEN = EMBED_SIZE * 4
        ENCODER_BLOCKS = 6
        SEQUENCE_LENGTH = 16
        
        print("🔧 Rebuilding model architecture...")
        
        # Rebuild the model with exact same parameters
        vit_model = VisionTransformer(
            n_heads=N_HEADS,
            n_classes=CLASSES,
            img_size=IMG_SIZE,
            mlp_dropout=DROPOUT,
            pos_dropout=0.0,
            attn_dropout=0.0,
            embed_size=EMBED_SIZE,
            patch_size=PATCH_SIZE,
            n_blocks=ENCODER_BLOCKS,
            mlpHidden_size=MLP_HIDDEN,
        )

        # Build the model with correct input shape
        vit_model.build([None, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, CHANNELS])
        
        # Compile the model with same settings as training
        vit_model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1.1),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ],
            run_eagerly=False,
        )
        
        # Build with dummy input to ensure proper initialization
        dummy_input = tf.zeros((1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, CHANNELS))
        _ = vit_model(dummy_input)
        
        print("🔧 Loading weights...")
        # Try to load weights only
        vit_model.load_weights(model_path)
        
        print("Model rebuilt and weights loaded successfully! Model rebuilt and weights loaded successfully!")
        return vit_model
        
    except Exception as e:
        print(f"Error rebuilding model: Error rebuilding model: {e}")
        return None

def load_model_with_custom_objects(model_path):
    """Load model with custom objects registered"""
    
    # First try rebuilding architecture and loading weights
    model = rebuild_model_and_load_weights(model_path)
    if model is not None:
        return model
    
    # Fallback to loading full model with custom objects
    custom_objects = {
        'VisionTransformer': VisionTransformer,
        'Projection': Projection,
        'TransformerEncoder': TransformerEncoder,
        'MLP': MLP,
        'MHA': MHA
    }
    
    try:
        print("🔧 Trying to load full model with custom objects...")
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model with custom objects: {e}")
        # Try loading without custom objects as fallback
        try:
            print("🔧 Trying fallback loading...")
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e2:
            print(f"Fallback loading also failed: {e2}")
            return None

def load_video_frames(video_path, sequence_length=16, img_height=224, img_width=224):
    """Load and preprocess video frames"""
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < sequence_length:
            # If video has fewer frames than required, duplicate last frame
            for i in range(total_frames):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (img_width, img_height))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            # Duplicate last frame to reach sequence_length
            if frames:
                last_frame = frames[-1]
                while len(frames) < sequence_length:
                    frames.append(last_frame)
        else:
            # Sample frames uniformly across the video
            frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (img_width, img_height))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
        
        cap.release()
        
        if len(frames) == sequence_length:
            # Normalize frames to [0, 1]
            frames = np.array(frames, dtype=np.float32) / 255.0
            return frames
        else:
            return None
            
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None

def predict_single_video(model, video_path, sequence_length=16, img_height=224, img_width=224):
    """Predict violence probability for a single video"""
    frames = load_video_frames(video_path, sequence_length, img_height, img_width)
    
    if frames is None:
        return None, "Error loading video"
    
    # Add batch dimension
    frames = np.expand_dims(frames, axis=0)
    
    try:
        # Make prediction
        prediction = model.predict(frames, verbose=0)
        probability = float(prediction[0][0])
        
        # Determine class (assuming 0 = NonViolence, 1 = Violence)
        predicted_class = "Violence" if probability > 0.5 else "NonViolence"
        confidence = probability if probability > 0.5 else 1 - probability
        
        return {
            'probability': probability,
            'predicted_class': predicted_class,
            'confidence': confidence
        }, None
        
    except Exception as e:
        return None, f"Error making prediction: {e}"

def find_best_model(checkpoint_dir):
    """Find the best model in checkpoint directory"""
    try:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None
        
        # Look for .h5 files
        model_files = list(checkpoint_path.glob("**/*.h5"))
        
        if not model_files:
            return None
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return str(model_files[0])
        
    except Exception as e:
        print(f"Error finding best model: {e}")
        return None

def create_video_report(video_path, prediction_result):
    """Create a formatted report for video prediction"""
    if prediction_result[1] is not None:  # Error occurred
        return f"""
Error rebuilding model: ERROR ANALYZING VIDEO
=====================================
📁 Video: {os.path.basename(video_path)}
⚠️  Error: {prediction_result[1]}
"""
    
    result = prediction_result[0]
    
    # Choose emoji and color based on prediction
    if result['predicted_class'] == 'Violence':
        emoji = "🚨"
        status = "VIOLENCE DETECTED"
    else:
        emoji = "Model rebuilt and weights loaded successfully!"
        status = "NON-VIOLENT"
    
    return f"""
{emoji} VIDEO ANALYSIS COMPLETE
=====================================
📁 Video: {os.path.basename(video_path)}
📊 Prediction: {result['predicted_class']}
📈 Confidence: {result['confidence']:.2%}
📋 Raw Probability: {result['probability']:.4f}
🎯 Status: {status}
=====================================
"""

def batch_predict_videos(model, video_paths, sequence_length=16, img_height=224, img_width=224):
    """Predict violence for multiple videos"""
    results = []
    
    print(f"🎬 Analyzing {len(video_paths)} videos...")
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"📹 Processing video {i}/{len(video_paths)}: {os.path.basename(video_path)}")
        
        result = predict_single_video(model, video_path, sequence_length, img_height, img_width)
        results.append({
            'video_path': video_path,
            'result': result
        })
    
    return results
