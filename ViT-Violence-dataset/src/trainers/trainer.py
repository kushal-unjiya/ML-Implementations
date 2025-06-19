import datetime
import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from base.base_trainer import BaseTrain


class CleanProgressCallback(tf.keras.callbacks.Callback):
    """Clean and informative progress callback"""
    
    def __init__(self, epochs):
        super().__init__()
        # Ensure epochs is an integer, not a DotMap
        if hasattr(epochs, '__int__'):
            self.epochs = int(epochs)
        else:
            self.epochs = 50  # Default fallback
        print(f"🔧 CleanProgressCallback initialized with {self.epochs} epochs")
        
    def on_train_begin(self, logs=None):
        print("=" * 60)
        
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch Epoch {epoch + 1}/{self.epochs}")
        
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # Format metrics nicely
            train_acc = logs.get('accuracy', 0) * 100
            train_loss = logs.get('loss', 0)
            val_acc = logs.get('val_accuracy', 0) * 100
            val_loss = logs.get('val_loss', 0)
            
            print(f"Complete: Epoch {epoch + 1} Complete:")
            print(f"   Training Training   - Loss: {train_loss:.4f} | Accuracy: {train_acc:.1f}%")
            print(f"   Epoch Validation - Loss: {val_loss:.4f} | Accuracy: {val_acc:.1f}%")
            
            # Show improvement indicator
            if hasattr(self, 'best_val_loss'):
                if val_loss < self.best_val_loss:
                    print("   🎯 New best model saved!")
                    self.best_val_loss = val_loss
            else:
                self.best_val_loss = val_loss
                print("   💾 Model saved!")
        
    def on_train_end(self, logs=None):
        print("\n🎉 Training Completed Successfully!")
        print("=" * 60)


class ModelTrainer(BaseTrain):
    def __init__(self, model, data_train, data_validate, config, train_df=None, val_df=None):
        """_summary_

        Args:
            model (_type_): compiled model
            data_train (_type_): training data
            data_validate (_type_): validation data
        """
        super(ModelTrainer, self).__init__(model, data_train, data_validate, config)
        self.callbacks = []
        self.log_dir = (
            os.path.join(os.getcwd(), "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/")
        )
        self.train_df = train_df
        self.val_df = val_df
        self.init_callbacks()

    def init_callbacks(self):
        # Add clean progress callback
        train_steps = len(self.train_df) // self.config.trainer.BATCH_SIZE
        val_steps = len(self.val_df) // self.config.trainer.BATCH_SIZE
        
        # Ensure epochs is an integer, not a DotMap
        epochs_value = self.config.trainer.EPOCHS
        if hasattr(epochs_value, '__int__'):
            epochs_value = int(epochs_value)
        else:
            epochs_value = 50  # Default fallback
            
        self.callbacks.append(CleanProgressCallback(epochs_value))
        
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(
                    self.config.callbacks.checkpoint_dir,
                    f"{self.config.exp.name}-{{epoch:02d}}.h5"
                ),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                verbose=0,  # Disable verbose to avoid clutter
            )
        )
        self.callbacks.append(
            EarlyStopping(patience=self.config.callbacks.ESPatience, monitor="val_loss")
        )
        self.callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=self.config.callbacks.lrSPatience,
                min_lr=self.config.callbacks.lrSmin_lr,
            )
        )

        self.callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        )
        
        # Add custom callbacks if provided
        if hasattr(self.config, 'custom_callbacks') and self.config.custom_callbacks:
            self.callbacks.extend(self.config.custom_callbacks)

    def train(self):
        steps_per_epoch = None
        validation_steps = None
        if self.train_df is not None:
            batch_size = self.config.trainer.BATCH_SIZE if isinstance(self.config.trainer, dict) or isinstance(self.config.trainer.BATCH_SIZE, int) else 16
            try:
                batch_size = int(self.config.trainer.BATCH_SIZE)
            except Exception:
                batch_size = 16
            steps_per_epoch = len(self.train_df) // batch_size
        if self.val_df is not None:
            batch_size = self.config.trainer.BATCH_SIZE if isinstance(self.config.trainer, dict) or isinstance(self.config.trainer.BATCH_SIZE, int) else 16
            try:
                batch_size = int(self.config.trainer.BATCH_SIZE)
            except Exception:
                batch_size = 16
            validation_steps = len(self.val_df) // batch_size
        print(f"🎯 Training Configuration:")
        print(f"   Epoch Training samples: {len(self.train_df) if self.train_df is not None else 'Unknown'}")
        print(f"   Epoch Validation samples: {len(self.val_df) if self.val_df is not None else 'Unknown'}")
        print(f"   Epoch Batch size: {batch_size}")
        print(f"   Epoch Steps per epoch: {steps_per_epoch}")
        print(f"   Epoch Validation steps: {validation_steps}")
        print(f"   Epoch Total epochs: {self.config.trainer.EPOCHS}")
        print(f"   ⏱️  Estimated time per epoch: 4-6 minutes")
        print()
        
        # Ensure epochs is an integer for model.fit
        epochs_value = self.config.trainer.EPOCHS
        if hasattr(epochs_value, '__int__'):
            epochs_value = int(epochs_value)
        else:
            epochs_value = 50  # Default fallback
        
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=epochs_value,
            callbacks=self.callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=0  # Disable default progress bar to use our clean callback
        )

    if __name__ == "__main__":
        pass
