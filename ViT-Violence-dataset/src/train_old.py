import datetime
import tensorflow as tf
from tqdm import tqdm
import os

# Reduce TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress INFO messages
tf.get_logger().setLevel('ERROR')  # Only show errors

from dotmap import DotMap

# Configure TensorFlow for optimal 8-core GPU performance
print("TensorFlow version:", tf.__version__)
print("Configuring 8-core GPU...")

# Enable mixed precision for faster training (but use compatible policy)
# Temporarily disable mixed precision to avoid dtype conflicts
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# print("Mixed precision enabled (float16/float32)")
print("Using float32 for compatibility")

# Configure memory growth and parallelism
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Metal GPU detected: {len(physical_devices)} device(s)")
    except RuntimeError as e:
        print(f"GPU error: {e}")

# Optimize for 8-core GPU
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

from data_loader.data_loader_01 import *
from models.model_01 import VisionTransformer
from trainers.trainer import ModelTrainer

TRAIN_CONFIG = {
    "exp": {"name": "Experiment 1"},
    "trainer": {
        "name": "trainer.ModelTrainer",
        "EPOCHS": 100,
        "verbose_training": True,  # Enable verbose for progress tracking
        "save_pickle": True,
        "BATCH_SIZE": 16,  # Reduced to match BATCH_SIZE constant
    },
    "callbacks": {
        "checkpoint_dir": "./checkpoints/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + "/",
        "checkpoint_monitor": "val_loss",
        "checkpoint_mode": "min",
        "checkpoint_save_best_only": True,
        "checkpoint_save_weights_only": False,
        "checkpoint_verbose": 1,
        "ESPatience": 5,
        "lrSPatience": 3,
        "lrSmin_lr": 1e-6,
    },
}
TRAIN_CONFIG = DotMap(TRAIN_CONFIG)

SEED = 42
CLASSES = 2
N_HEADS = 3
CHANNELS = 3
DROPOUT = 0.1
IMG_SIZE = 224
BATCH_SIZE = 16  # Reduced from 32 since mixed precision is disabled
PATCH_SIZE = 16
EMBED_SIZE = 256  # Increased for better accuracy
MLP_HIDDEN = EMBED_SIZE * 4
ENCODER_BLOCKS = 6  # Increased for better accuracy
SEQUENCE_LENGTH = 16
AUTOTUNE = tf.data.AUTOTUNE
DATADIR = "./dataset"


def main():
    print("\n🎯 VISION TRANSFORMER TRAINING - 8-CORE GPU")
    print("=" * 55)
    print("Configuration: Configuration:")
    print(f"   • GPU: 8-core Metal (Mixed Precision: Disabled)")
    print(f"   • Batch Size: {BATCH_SIZE} | Epochs: {TRAIN_CONFIG.trainer.EPOCHS}")
    print(f"   • Image Size: {IMG_SIZE}x{IMG_SIZE} | Frames: {SEQUENCE_LENGTH}")
    print("=" * 55)
    
    # Use GPU strategy
    strategy = tf.distribute.get_strategy()
    
    with strategy.scope():
        # Build Dataset
        print("\n📂 Loading dataset...")
        train, test, validation = build_dataframe(path=DATADIR)
        tr, ts, val = build_dataset(train, test, validation)
        
        print("🧠 Building model...")
        # Build Model

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

        vit_model.build([None, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, CHANNELS])

        vit_model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1.1),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ],
            # Enable mixed precision for 8-core GPU optimization
            run_eagerly=False,
        )

        # Build the model with a dummy input to calculate parameters
        dummy_input = tf.zeros((1, 16, 224, 224, 3))  # [batch, depth, height, width, channels]
        _ = vit_model(dummy_input)  # This builds the model
        
        print("Model ready! Model ready!")
        print(f"Parameters: Parameters: {vit_model.count_params():,}")
        
        TRAINER = ModelTrainer(vit_model, tr, val, TRAIN_CONFIG, train_df=train, val_df=validation)
        TRAINER.train()


if __name__ == "__main__":
    main()
