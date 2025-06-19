import os
import time
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from models.pytorch_model import create_vit_model
from data_loader.pytorch_dataloader import create_data_loaders, VideoDataset


def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


class ViolenceDetectionTester:
    """Tester class for violence detection model."""
    
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.model = None
        
    def load_model(self):
        """Load the trained model."""
        try:
            print(f"📥 Loading model from: {self.model_path}")
            
            self.model = create_vit_model(
                img_size=224,
                patch_size=16,
                embed_dim=256,
                num_heads=8,
                num_layers=6,
                sequence_length=16,
                dropout=0.1
            ).to(self.device)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"📊 Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}")
            else:
                self.model.load_state_dict(checkpoint)
                print("✅ Model weights loaded successfully!")
            
            self.model.eval()
            print(f"📈 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate_single_video(self, video_path, sequence_length=16, img_size=224):
        """Evaluate a single video file."""
        if self.model is None:
            print("❌ Model not loaded. Call load_model() first.")
            return None, None
        
        print(f"\n📹 Processing video: {video_path}")
        
        try:
            frames = self.load_video(video_path, sequence_length, img_size)
            if frames is None:
                return None, None
            
            video_tensor = torch.from_numpy(frames).float().unsqueeze(0).to(self.device)
            video_tensor = video_tensor.permute(0, 1, 4, 2, 3)
            
            print(f"📹 Video tensor shape: {video_tensor.shape}")
            
            with torch.no_grad():
                output = self.model(video_tensor)
                probability = torch.sigmoid(output).item()
                prediction = int(probability > 0.5)
            
            result = "Violence" if prediction == 1 else "Non-Violence"
            print(f"\n🎬 Prediction for '{Path(video_path).name}': {result}")
            print(f"   Confidence: {probability:.4f}")
            
            return result, probability
            
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def load_video(self, video_path, sequence_length=16, img_size=224):
        """Load video frames and preprocess them."""
        frames = []
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Could not open video file: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"❌ No frames found in video: {video_path}")
            cap.release()
            return None
        
        if total_frames <= sequence_length:
            frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (img_size, img_size))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    zero_frame = np.zeros((img_size, img_size, 3), dtype=np.float32)
                    frames.append(zero_frame)
        
        cap.release()
        
        while len(frames) < sequence_length:
            if frames:
                frames.append(frames[-1])
            else:
                zero_frame = np.zeros((img_size, img_size, 3), dtype=np.float32)
                frames.append(zero_frame)
        
        return np.array(frames[:sequence_length])
    
    def evaluate_dataset(self, test_loader):
        """Evaluate model on test dataset."""
        if self.model is None:
            print("❌ Model not loaded. Call load_model() first.")
            return None
        
        print("\n🔍 Evaluating on test dataset...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for videos, targets in tqdm(test_loader, desc="Testing"):
                videos = videos.to(self.device)
                targets = targets.numpy()
                
                outputs = self.model(videos)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                
                all_predictions.extend(predictions.flatten())
                all_targets.extend(targets.flatten())
                all_probabilities.extend(probabilities.flatten())
        
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'targets': all_targets,
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }
        
        print(f"\n📊 Test Results:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, targets, predictions, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Violence', 'Violence'],
                   yticklabels=['Non-Violence', 'Violence'])
        plt.title('Confusion Matrix - Violence Detection')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_classification_report(self, targets, predictions):
        """Generate detailed classification report."""
        class_names = ['Non-Violence', 'Violence']
        report = classification_report(targets, predictions, target_names=class_names)
        
        print("\n📋 CLASSIFICATION REPORT:")
        print("=" * 60)
        print(report)
        
        return report


def find_best_model(checkpoint_dir):
    """Find the best model in checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    best_models = list(checkpoint_path.glob("*_best.pth"))
    if best_models:
        return str(best_models[0])
    
    pth_files = list(checkpoint_path.glob("*.pth"))
    if pth_files:
        latest_model = max(pth_files, key=os.path.getmtime)
        return str(latest_model)
    
    return None


def main():
    """Main testing function."""
    print("🧪 PyTorch Violence Detection Testing")
    print("=" * 60)
    
    model_path = "checkpoints/ViolenceDetection_ViT_20250617-162133_best.pth"
    dataset_dir = "dataset"
    sequence_length = 16
    img_size = 224

    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    video_files = [str(p) for p in Path(dataset_dir).glob("**/*") if p.suffix.lower() in video_extensions]
    if not video_files:
        print(f"❌ No video files found in {dataset_dir}")
        return
    print(f"✅ Found {len(video_files)} video files in {dataset_dir}")

    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return

    device = get_device()
    
    tester = ViolenceDetectionTester(model_path, device)
    
    if not tester.load_model():
        return

    for video_path in video_files:
        result, probability = tester.evaluate_single_video(video_path, sequence_length=sequence_length, img_size=img_size)
        if result is not None:
            print(f"\n✅ Video: {video_path}\n   Result: {result}\n   Confidence: {probability:.4f}")
        else:
            print(f"\n❌ Failed to evaluate video: {video_path}")


if __name__ == "__main__":
    main()
