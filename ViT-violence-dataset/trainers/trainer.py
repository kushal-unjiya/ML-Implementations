import datetime
import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from base.base_trainer import BaseTrain


class CleanProgressCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, epochs):
        super().__init__()
        if hasattr(epochs, '__int__'):
            self.epochs = int(epochs)
        else:
            self.epochs = 50
        print(f"🔧 CleanProgressCallback initialized with {self.epochs} epochs")
        
    def on_train_begin(self, logs=None):
        print("=" * 60)
        
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n📊 Epoch {epoch + 1}/{self.epochs}")
        
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            train_acc = logs.get('accuracy', 0) * 100
            train_loss = logs.get('loss', 0)
            val_acc = logs.get('val_accuracy', 0) * 100
            val_loss = logs.get('val_loss', 0)
            
            print(f"✅ Epoch {epoch + 1} Complete:")
            print(f"   📈 Training   - Loss: {train_loss:.4f} | Accuracy: {train_acc:.1f}%")
            print(f"   📊 Validation - Loss: {val_loss:.4f} | Accuracy: {val_acc:.1f}%")
            
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

        super(ModelTrainer, self).__init__(model, data_train, data_validate, config)
        self.callbacks = []
        self.log_dir = (
            os.path.join(os.getcwd(), "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/")
        )
        self.train_df = train_df
        self.val_df = val_df
        self.init_callbacks()

    def init_callbacks(self):
        train_steps = len(self.train_df) // self.config.trainer.BATCH_SIZE
        val_steps = len(self.val_df) // self.config.trainer.BATCH_SIZE
        
        epochs_value = self.config.trainer.EPOCHS
        if hasattr(epochs_value, '__int__'):
            epochs_value = int(epochs_value)
        else:
            epochs_value = 50
            
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
                verbose=0,
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
        print(f"   📊 Training samples: {len(self.train_df) if self.train_df is not None else 'Unknown'}")
        print(f"   📊 Validation samples: {len(self.val_df) if self.val_df is not None else 'Unknown'}")
        print(f"   📊 Batch size: {batch_size}")
        print(f"   📊 Steps per epoch: {steps_per_epoch}")
        print(f"   📊 Validation steps: {validation_steps}")
        print(f"   📊 Total epochs: {self.config.trainer.EPOCHS}")
        print(f"   ⏱️  Estimated time per epoch: 4-6 minutes")
        print()
        
        epochs_value = self.config.trainer.EPOCHS
        if hasattr(epochs_value, '__int__'):
            epochs_value = int(epochs_value)
        else:
            epochs_value = 50
        
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=epochs_value,
            callbacks=self.callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=0
        )

    if __name__ == "__main__":
        pass
