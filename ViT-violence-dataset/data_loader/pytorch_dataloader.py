
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image


class VideoDataset(Dataset):
    
    def __init__(self, dataframe, sequence_length=16, img_size=224, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.transform = transform
        self.class_names = {"NonViolence": 0, "Violence": 1}
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        video_path = self.dataframe.iloc[idx]['Video_Path']
        label = self.class_names[self.dataframe.iloc[idx]['Labels']]
        
        frames = self.load_video(video_path)
        
        if self.transform:
            transformed_frames = []
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    frame = Image.fromarray(frame)
                transformed_frames.append(self.transform(frame))
            frames = torch.stack(transformed_frames)
        else:
            frames = torch.from_numpy(frames).float()
            if frames.max() > 1.0:
                frames = frames / 255.0
            frames = frames.permute(0, 3, 1, 2)
        
        return frames, torch.tensor(label, dtype=torch.float32)
    
    def load_video(self, video_path):
        frames = []
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            dummy_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            return np.array([dummy_frame] * self.sequence_length)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            dummy_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            return np.array([dummy_frame] * self.sequence_length)
        
        if total_frames <= self.sequence_length:
            frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    zero_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                    frames.append(zero_frame)
        
        cap.release()
        
        while len(frames) < self.sequence_length:
            if frames:
                frames.append(frames[-1])
            else:
                zero_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                frames.append(zero_frame)
        
        return np.array(frames[:self.sequence_length])


def build_dataframe(data_path):
    
    data_path = Path(data_path)
    
    video_paths = list(data_path.glob("*/*.mp4"))
    
    if not video_paths:
        raise ValueError(f"No .mp4 files found in {data_path}")
    
    video_labels = [path.parent.name for path in video_paths]
    
    print(f"📁 Found {len(video_paths)} videos total")
    
    unique_labels, counts = np.unique(video_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"   📁 {label}: {count} videos")
    
    dataframe = pd.DataFrame({
        "Video_Path": video_paths,
        "Labels": video_labels
    })
    
    dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n_total = len(dataframe)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    
    train_df = dataframe[:n_train]
    val_df = dataframe[n_train:n_train + n_val]
    test_df = dataframe[n_train + n_val:]
    
    print(f"📊 Dataset split:")
    print(f"   📈 Train: {len(train_df)} videos")
    print(f"   📊 Validation: {len(val_df)} videos")
    print(f"   📋 Test: {len(test_df)} videos")
    
    return train_df, val_df, test_df


def get_data_transforms(img_size=224, is_training=True):
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_data_loaders(
    data_path,
    batch_size=16,
    sequence_length=16,
    img_size=224,
    num_workers=4,
    pin_memory=True
):
    
    train_df, val_df, test_df = build_dataframe(data_path)
    
    train_transform = get_data_transforms(img_size, is_training=True)
    val_transform = get_data_transforms(img_size, is_training=False)
    
    train_dataset = VideoDataset(
        train_df, 
        sequence_length=sequence_length, 
        img_size=img_size,
        transform=train_transform
    )
    
    val_dataset = VideoDataset(
        val_df, 
        sequence_length=sequence_length, 
        img_size=img_size,
        transform=val_transform
    )
    
    test_dataset = VideoDataset(
        test_df, 
        sequence_length=sequence_length, 
        img_size=img_size,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"✅ Data loaders created successfully!")
    print(f"   🔄 Train loader: {len(train_loader)} batches")
    print(f"   🔄 Val loader: {len(val_loader)} batches")
    print(f"   🔄 Test loader: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, train_df, val_df, test_df


if __name__ == "__main__":
    data_path = "dataset"
    
    if Path(data_path).exists():
        try:
            train_loader, val_loader, test_loader, _, _, _ = create_data_loaders(
                data_path=data_path,
                batch_size=2,
                sequence_length=16,
                img_size=224,
                num_workers=0
            )
            
            for batch_idx, (videos, labels) in enumerate(train_loader):
                print(f"Batch {batch_idx}:")
                print(f"  Videos shape: {videos.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Labels: {labels}")
                if batch_idx >= 2:
                    break
                    
        except Exception as e:
            print(f"Error testing data loader: {e}")
    else:
        print(f"Dataset path {data_path} does not exist")
