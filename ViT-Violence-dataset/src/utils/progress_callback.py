import tensorflow as tf
from tqdm import tqdm
import sys

class TqdmProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to show beautiful tqdm progress bars during training"""
    
    def __init__(self, train_steps, val_steps):
        super().__init__()
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.epoch_pbar = None
        self.train_pbar = None
        self.val_pbar = None
        
    def on_train_begin(self, logs=None):
        print("\nStarting Starting Vision Transformer Training on 8-core GPU")
        print("=" * 60)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_pbar = tqdm(
            total=100, 
            desc=f"Epoch {epoch + 1}/{self.params['epochs']}", 
            position=0,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        # Training progress bar
        self.train_pbar = tqdm(
            total=self.train_steps,
            desc="Training",
            position=1,
            leave=False,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [⏱️{elapsed}<{remaining}, {rate_fmt}]"
        )
        
    def on_train_batch_end(self, batch, logs=None):
        if self.train_pbar:
            # Update with current metrics
            postfix = {}
            if logs:
                if 'loss' in logs:
                    postfix['loss'] = f"{logs['loss']:.4f}"
                if 'accuracy' in logs:
                    postfix['acc'] = f"{logs['accuracy']:.4f}"
            
            self.train_pbar.set_postfix(postfix)
            self.train_pbar.update(1)
            
            # Update epoch progress (training is 70% of epoch)
            epoch_progress = (batch + 1) / self.train_steps * 70
            if self.epoch_pbar:
                self.epoch_pbar.n = int(epoch_progress)
                self.epoch_pbar.refresh()
    
    def on_test_begin(self, logs=None):
        if self.train_pbar:
            self.train_pbar.close()
            
        # Validation progress bar
        self.val_pbar = tqdm(
            total=self.val_steps,
            desc="Validation",
            position=1,
            leave=False,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [⏱️{elapsed}<{remaining}, {rate_fmt}]"
        )
    
    def on_test_batch_end(self, batch, logs=None):
        if self.val_pbar:
            # Update with current metrics
            postfix = {}
            if logs:
                if 'loss' in logs:
                    postfix['val_loss'] = f"{logs['loss']:.4f}"
                if 'accuracy' in logs:
                    postfix['val_acc'] = f"{logs['accuracy']:.4f}"
            
            self.val_pbar.set_postfix(postfix)
            self.val_pbar.update(1)
            
            # Update epoch progress (validation is remaining 30% of epoch)
            epoch_progress = 70 + (batch + 1) / self.val_steps * 30
            if self.epoch_pbar:
                self.epoch_pbar.n = int(epoch_progress)
                self.epoch_pbar.refresh()
    
    def on_epoch_end(self, epoch, logs=None):
        if self.val_pbar:
            self.val_pbar.close()
            
        if self.epoch_pbar:
            self.epoch_pbar.n = 100
            self.epoch_pbar.refresh()
            
            # Show epoch summary
            summary = f"Epoch {epoch + 1} Summary: "
            if logs:
                metrics = []
                if 'loss' in logs:
                    metrics.append(f"loss={logs['loss']:.4f}")
                if 'accuracy' in logs:
                    metrics.append(f"acc={logs['accuracy']:.4f}")
                if 'val_loss' in logs:
                    metrics.append(f"val_loss={logs['val_loss']:.4f}")
                if 'val_accuracy' in logs:
                    metrics.append(f"val_acc={logs['val_accuracy']:.4f}")
                summary += " | ".join(metrics)
            
            self.epoch_pbar.set_description(summary)
            self.epoch_pbar.close()
            print()  # Add spacing between epochs
    
    def on_train_end(self, logs=None):
        print("\n🎉 Training completed successfully!")
        print("=" * 60)
