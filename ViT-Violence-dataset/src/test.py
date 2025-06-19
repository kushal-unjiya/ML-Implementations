#!/usr/bin/env python3
"""
Comprehensive Test Script for Real-Life Violence Detection
=========================================================
Single file for model evaluation and testing with visualization
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import cv2

# Reduce TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('ERROR')

# Custom imports
from data_loader.data_loader_01 import * # Imports SEQUENCE_LENGTH, IMG_SIZE, CHANNELS, PATCH_SIZE
from models.model_01 import VisionTransformer, Projection, MHA, MLP, TransformerEncoder # Ensure all are imported
from utils.utils import find_best_model


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model_path, log_dir=None):
        self.model_path = model_path
        if log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / "plots").mkdir(exist_ok=True)
            (self.log_dir / "reports").mkdir(exist_ok=True)
        else:
            self.log_dir = None
        
        self.model = None
        self.test_results = {}
        
    def load_model(self):
        """Load the trained model using tf.keras.models.load_model."""
        try:
            print(f"Loading model from: Loading model from: {self.model_path} using tf.keras.models.load_model")
            
            custom_objects = {
                'VisionTransformer': VisionTransformer,
                'Projection': Projection,
                'MHA': MHA,
                'MLP': MLP,
                'TransformerEncoder': TransformerEncoder
            }
            
            # When loading a model with custom objects, Keras needs to know about them.
            # compile=False is used because we are only doing inference.
            with tf.keras.utils.custom_object_scope(custom_objects):
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
            
            print("Model loaded Model loaded successfully!")
            print(f"Model Model input shape: {self.model.input_shape}")
            print(f"Model Model output shape: {self.model.output_shape}")
            print(f"Parameters: Parameters: {self.model.count_params():,}")
            print("📝 Model summary after loading:")
            self.model.summary()
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            return False
    
    def evaluate_on_test_set(self, test_data, test_df):
        """Evaluate model on test dataset"""
        print("\nEvaluating Evaluating on test dataset...")
        
        # Get predictions
        y_pred_prob = self.model.predict(test_data)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Get true labels
        y_true = []
        for batch_x, batch_y in test_data:
            y_true.extend(batch_y.numpy().flatten())
        y_true = np.array(y_true)
        
        # Calculate metrics
        test_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred_prob.flatten()).numpy().mean()
        test_accuracy = np.mean(y_pred == y_true)
        
        # Store results
        self.test_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob.flatten()
        }
        
        print(f"Model Test Loss: {test_loss:.4f}")
        print(f"Model Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        return self.test_results
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        if not self.test_results:
            print("ERROR: No test results available. Run evaluation first.")
            return
        
        y_true = self.test_results['y_true']
        y_pred = self.test_results['y_pred']
        
        # Classification report
        class_names = ['Non-Violence', 'Violence']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        print("\nCLASSIFICATION REPORT CLASSIFICATION REPORT:")
        print("=" * 60)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        for class_name in class_names:
            metrics = report[class_name]
            print(f"{class_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1-score']:<10.3f} {metrics['support']:<10.0f}")
        
        print("-" * 60)
        print(f"{'Accuracy':<15} {'':<10} {'':<10} {report['accuracy']:<10.3f} {report['macro avg']['support']:<10.0f}")
        print(f"{'Macro Avg':<15} {report['macro avg']['precision']:<10.3f} {report['macro avg']['recall']:<10.3f} "
              f"{report['macro avg']['f1-score']:<10.3f} {report['macro avg']['support']:<10.0f}")
        print(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:<10.3f} {report['weighted avg']['recall']:<10.3f} "
              f"{report['weighted avg']['f1-score']:<10.3f} {report['weighted avg']['support']:<10.0f}")
        
        # Save report
        report_path = self.log_dir / "reports" / f"classification_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nModel Classification report saved to: {report_path}")
        
        return report
    
    def plot_confusion_matrix(self):
        """Generate and plot confusion matrix"""
        if not self.test_results:
            print("ERROR: No test results available. Run evaluation first.")
            return
        
        y_true = self.test_results['y_true']
        y_pred = self.test_results['y_pred']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Violence', 'Violence'],
                   yticklabels=['Non-Violence', 'Violence'])
        plt.title('🎯 Confusion Matrix - Violence Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add accuracy annotation
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)', 
                   ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen'))
        
        plot_path = self.log_dir / "plots" / f"confusion_matrix_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Model Confusion matrix saved to: {plot_path}")
        plt.close()
        
        return cm
    
    def plot_roc_curve(self):
        """Generate ROC curve"""
        if not self.test_results:
            print("ERROR: No test results available. Run evaluation first.")
            return
        
        y_true = self.test_results['y_true']
        y_pred_prob = self.test_results['y_pred_prob']
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('🎯 ROC Curve - Violence Detection', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plot_path = self.log_dir / "plots" / f"roc_curve_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Model ROC curve saved to: {plot_path}")
        plt.close()
        
        return roc_auc
    
    def evaluate_single_video(self, video_path, sequence_length=16):
        """Evaluate a single video file."""
        if not self.model:
            print("ERROR: Model not loaded. Call load_model() first.")
            return None

        print(f"Processing video: Processing video: {video_path}")
        
        # Use the load_video function from data_loader_01
        # We need to instantiate VideoDataset or adapt load_video to be static/standalone
        # For simplicity, let's adapt parts of load_video here or make it callable
        
        frames = []
        video_capture = cv2.VideoCapture(video_path)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            print(f"ERROR: Could not open or read video file: {video_path}")
            video_capture.release()
            return None

        skip_frames_window = max(int(frame_count / sequence_length), 1)
        
        for i in range(frame_count):
            ret, frame = video_capture.read()
            if not ret:
                break
            if i % skip_frames_window == 0 and len(frames) < sequence_length:
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
                image = tf.image.convert_image_dtype(rgb_img, dtype=tf.float32)
                image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) # IMG_SIZE from data_loader
                frames.append(image)
        
        video_capture.release()

        if not frames:
            print("ERROR: Could not extract frames from the video.")
            return None

        # Pad if necessary
        if len(frames) < sequence_length:
            pad_tensor = tf.zeros(shape=[IMG_SIZE, IMG_SIZE, CHANNELS], dtype=tf.float32) # CHANNELS from data_loader
            for _ in range(sequence_length - len(frames)):
                frames.append(pad_tensor)
        
        video_tensor = np.expand_dims(np.asarray(frames), axis=0) # Add batch dimension

        print(f"Processing video: Video tensor shape: {video_tensor.shape}")

        # Make prediction
        prediction_prob = self.model.predict(video_tensor)
        prediction = (prediction_prob > 0.5).astype(int).flatten()[0]
        
        result = "Violence" if prediction == 1 else "Non-Violence"
        print(f"Prediction Prediction for '{Path(video_path).name}': {result} (Probability: {prediction_prob[0][0]:.4f})")
        return result, prediction_prob[0][0]

    def plot_precision_recall_curve(self):
        """Generate Precision-Recall curve"""
        if not self.test_results:
            print("ERROR: No test results available. Run evaluation first.")
            return
        
        y_true = self.test_results['y_true']
        y_pred_prob = self.test_results['y_pred_prob']
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        avg_precision = average_precision_score(y_true, y_pred_prob)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('🎯 Precision-Recall Curve - Violence Detection', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plot_path = self.log_dir / "plots" / f"precision_recall_curve_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Model Precision-Recall curve saved to: {plot_path}")
        plt.close()
        
        return avg_precision
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        if not self.test_results:
            print("ERROR: No test results available. Run evaluation first.")
            return
        
        # Calculate additional metrics
        y_true = self.test_results['y_true']
        y_pred = self.test_results['y_pred']
        y_pred_prob = self.test_results['y_pred_prob']
        
        # Basic metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Average Precision
        avg_precision = average_precision_score(y_true, y_pred_prob)
        
        comprehensive_metrics = {
            'test_accuracy': self.test_results['test_accuracy'],
            'test_loss': self.test_results['test_loss'],
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(y_true)
        }
        
        print("\nModel COMPREHENSIVE EVALUATION REPORT:")
        print("=" * 60)
        print(f"Parameters: Test Accuracy: {comprehensive_metrics['test_accuracy']:.4f} ({comprehensive_metrics['test_accuracy']*100:.2f}%)")
        print(f"📉 Test Loss: {comprehensive_metrics['test_loss']:.4f}")
        print(f"🎯 Precision: {precision:.4f}")
        print(f"Evaluating Recall: {recall:.4f}")
        print(f"⚖️  F1-Score: {f1:.4f}")
        print(f"🔒 Specificity: {specificity:.4f}")
        print(f"Model ROC AUC: {roc_auc:.4f}")
        print(f"Model Average Precision: {avg_precision:.4f}")
        print("-" * 60)
        print(f"Model loaded True Positives: {tp}")
        print(f"Model loaded True Negatives: {tn}")
        print(f"ERROR: False Positives: {fp}")
        print(f"ERROR: False Negatives: {fn}")
        print(f"Model Total Samples: {len(y_true)}")
        print("=" * 60)
        
        # Save comprehensive report
        report_path = self.log_dir / "reports" / f"comprehensive_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(comprehensive_metrics, f, indent=2)
        print(f"\nModel Comprehensive report saved to: {report_path}")
        
        return comprehensive_metrics


def main():
    """Main testing function"""
    print("SINGLE VIDEO VIOLENCE DETECTION TEST SINGLE VIDEO VIOLENCE DETECTION TEST")
    print("=" * 60)
    
    # Updated model path
    model_path = "/Users/zen/Desktop/GUNI/Real-Life-Violence-Detection/checkpoints/20250617-111900/ViolenceDetection_ViT_20250617-111900-01.h5"
    video_path = "/Users/zen/Desktop/GUNI/Real-Life-Violence-Detection/src/dataset/videoplayback.mp4"

    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        return
    if not Path(video_path).exists():
        print(f"ERROR: Video file not found: {video_path}")
        return

    # Initialize evaluator (log_dir is not strictly needed for single video eval)
    evaluator = ModelEvaluator(model_path) 
    
    # Load model
    if not evaluator.load_model():
        return
    
    # Evaluate single video
    result, probability = evaluator.evaluate_single_video(video_path, sequence_length=SEQUENCE_LENGTH) # SEQUENCE_LENGTH from data_loader

    if result:
        print(f"Model loaded Video evaluation completed.")
        print(f"   Video: {video_path}")
        print(f"   Predicted: {result}")
        print(f"   Confidence: {probability:.4f}")
    else:
        print("ERROR: Video evaluation failed.")


if __name__ == "__main__":
    main()
