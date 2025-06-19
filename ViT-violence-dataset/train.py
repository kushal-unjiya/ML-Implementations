
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torch.amp import GradScaler, autocast
from contextlib import nullcontext
from models.pytorch_model import create_vit_model
from data_loader.pytorch_dataloader import create_data_loaders


def get_device():
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


class ViolenceDetectionTrainer:
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=1e-3,
        weight_decay=1e-4,
        save_dir="checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_name = f"ViolenceDetection_ViT_{timestamp}"
        self.log_dir = Path("logs") / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.scaler = GradScaler(device_type='cuda') if self.device.type == 'cuda' else None
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} - Training")
        
        for videos, targets in pbar:
            start = time.time()
            videos, targets = videos.to(self.device), targets.to(self.device).unsqueeze(1)
            data_time = time.time() - start
            print(f'Data loading time: {data_time:.4f}s')
            self.optimizer.zero_grad()
            
            autocast_ctx = autocast(device_type='cuda') if self.device.type == 'cuda' else nullcontext()
            with autocast_ctx:
                outputs = self.model(videos)
                loss = self.criterion(outputs, targets)
            
            if self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            
            predictions = torch.sigmoid(outputs) > 0.5
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_targets, all_predictions)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} - Validation")
            
            for videos, targets in pbar:
                videos = videos.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)
                
                outputs = self.model(videos)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                
                predictions = torch.sigmoid(outputs) > 0.5
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_targets, all_predictions)
        epoch_precision = precision_score(all_targets, all_predictions, zero_division=0)
        epoch_recall = recall_score(all_targets, all_predictions, zero_division=0)
        epoch_f1 = f1_score(all_targets, all_predictions, zero_division=0)
        
        return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1

    def save_model(self, epoch, val_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }
        
        checkpoint_path = self.save_dir / f"{self.run_name}_epoch_{epoch+1:02d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / f"{self.run_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved: {best_path}")
    
    def plot_training_curves(self):
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Loss Evolution', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Accuracy Evolution', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(epochs, self.learning_rates, 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        ax4.bar(['Best Val Acc'], [self.best_val_acc], color='green', alpha=0.7)
        ax4.set_title(f'Best Validation Accuracy: {self.best_val_acc:.4f}', fontweight='bold')
        ax4.set_ylabel('Accuracy')
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plot_path = self.log_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {plot_path}")
        return plot_path
    
    def train(self, num_epochs=50, save_every=5):
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Model checkpoints will be saved to: {self.save_dir}")
        print(f"Logs will be saved to: {self.log_dir}")
        print("="*80)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch)
            
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate_epoch(epoch)
            
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)
            
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Metrics/Precision', val_precision, epoch)
            self.writer.add_scalar('Metrics/Recall', val_recall, epoch)
            self.writer.add_scalar('Metrics/F1', val_f1, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1:3d}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  Precision:  {val_precision:.4f} | Recall:    {val_recall:.4f} | F1: {val_f1:.4f}")
            print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            if is_best:
                print("  NEW BEST MODEL!")
            print("-" * 80)
            
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_model(epoch, val_acc, is_best)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed!")
        print(f"Total time: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch + 1})")
        
        self.save_model(num_epochs - 1, val_acc, False)
        self.plot_training_curves()
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'total_epochs': num_epochs 
        }
        
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to: {history_path}")
        
        self.writer.close()
        return self.best_val_acc


def main():
    print("PyTorch Violence Detection Training")
    print("="*60)
    
    device = get_device()
    CONFIG = {
        'data_path': 'dataset',
        'batch_size': 4,
        'sequence_length': 16,
        'img_size': 224,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_workers': 8,

        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'patch_size': 16,

    }
    
    print("Loading dataset...")
    pin_memory = device.type == 'mps'
    train_loader, val_loader, test_loader, train_df, val_df, test_df = create_data_loaders(
        data_path=CONFIG['data_path'],
        batch_size=CONFIG['batch_size'],
        sequence_length=CONFIG['sequence_length'],
        img_size=CONFIG['img_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=pin_memory
    )
    
    print("Creating model...")
    model = create_vit_model(
        img_size=CONFIG['img_size'],
        patch_size=CONFIG['patch_size'],
        embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        sequence_length=CONFIG['sequence_length'],
        dropout=CONFIG['dropout']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = ViolenceDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    best_acc = trainer.train(
        num_epochs=CONFIG['num_epochs'],
        save_every=5
    )
    
    print(f"\nTraining completed with best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
