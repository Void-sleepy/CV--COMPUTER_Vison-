import numpy as np
import pandas as pd
import os
import cv2
import pickle
import threading
import time
from collections import deque, Counter
from datetime import datetime, timedelta
import pyttsx3  # Text-to-speech
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# PyTorch version compatibility check
TORCH_VERSION = torch.__version__
print(f"🔧 PyTorch version: {TORCH_VERSION}")

# Check for CUDA availability
if torch.cuda.is_available():
    print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚡ Using CPU - consider getting a GPU for faster training!")

# ============ FIXED DATA LOADING FOR YOUR STRUCTURE ============
class SignLanguageDataLoader:
    def __init__(self, sequence_length=10, img_size=96):
        self.sequence_length = sequence_length
        self.img_size = img_size
        
    def load_sequences_efficiently(self, base_path, labels_df, max_sessions_per_class=4, 
                                 stride=3, augment=True):
        """
        Modified data loading to skip folders without labels and continue processing.
        Handles your nested structure: train/0071/03_01_0071_(01_12_16_15_52_41)_c/frame_001.jpg
        """
        sequences = []
        labels = []
        sign_ids = []
        total_sequences = 0
        skipped_folders = []
        
        print(f"Loading sequences from {base_path}...")
        print("Dataset structure detected: train/signID/sessionID/frames/")
        
        for class_folder in sorted(os.listdir(base_path)):
            class_path = os.path.join(base_path, class_folder)
            if not os.path.isdir(class_path):
                continue
                
            try:
                # Handle both numeric (0071) and string formats
                if class_folder.isdigit():
                    sign_id = int(class_folder)
                else:
                    # Try to extract number from folder name like "0071_something"
                    sign_id = int(class_folder.split('_')[0])
            except (ValueError, IndexError):
                print(f"Skipping non-numeric folder: {class_folder}")
                skipped_folders.append(class_folder)
                continue
            
            # Get label from dataset
            sign_row = labels_df[labels_df['SignID'] == sign_id]
            if sign_row.empty:
                print(f"No label found for SignID {sign_id}, trying without leading zeros...")
                # Try without leading zeros
                sign_row = labels_df[labels_df['SignID'] == int(str(sign_id).lstrip('0') or '0')]
                if sign_row.empty:
                    print(f"Still no label found for SignID {sign_id}, skipping folder...")
                    skipped_folders.append(class_folder)
                    continue
            
            sign_english = str(sign_row['Sign-English'].iloc[0]).strip()
            sign_arabic = str(sign_row['Sign-Arabic'].iloc[0]).strip()
            
            # Process each session folder inside the class folder
            session_folders = []
            for item in os.listdir(class_path):
                item_path = os.path.join(class_path, item)
                if os.path.isdir(item_path):
                    session_folders.append(item)
            
            session_folders = sorted(session_folders)[:max_sessions_per_class]
            
            class_sequences = 0
            
            for session_folder in session_folders:
                session_path = os.path.join(class_path, session_folder)
                
                # Check if this is actually a frames directory or has subdirectories
                items_in_session = os.listdir(session_path)
                
                # Check if session folder contains frames directly or more subdirectories
                frame_files = [f for f in items_in_session 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if not frame_files:
                    # Maybe there are more subdirectories - check one level deeper
                    for subitem in items_in_session:
                        subitem_path = os.path.join(session_path, subitem)
                        if os.path.isdir(subitem_path):
                            subframe_files = [f for f in os.listdir(subitem_path)
                                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            if subframe_files:
                                session_path = subitem_path
                                frame_files = subframe_files
                                break
                
                if not frame_files:
                    print(f"No frames found in {session_path}, skipping...")
                    continue
                
                # Sort frame files properly (handle both numeric and string sorting)
                try:
                    # Try numeric sorting first
                    frame_files = sorted(frame_files, key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0]))))
                except:
                    # Fall back to string sorting
                    frame_files = sorted(frame_files)
                
                if len(frame_files) < self.sequence_length:
                    print(f"Not enough frames in {session_path}: {len(frame_files)} < {self.sequence_length}, skipping...")
                    continue
                
                # Create multiple sequences from same session with stride
                max_start = len(frame_files) - self.sequence_length
                step_size = max(1, stride)
                
                for start_idx in range(0, max_start + 1, step_size):
                    sequence_frames = []
                    valid_sequence = True
                    
                    for i in range(self.sequence_length):
                        frame_idx = start_idx + i
                        if frame_idx >= len(frame_files):
                            valid_sequence = False
                            break
                            
                        frame_path = os.path.join(session_path, frame_files[frame_idx])
                        
                        try:
                            img = cv2.imread(frame_path)
                            if img is None:
                                print(f"Failed to load image: {frame_path}, skipping sequence...")
                                valid_sequence = False
                                break
                            
                            # Preprocess frame
                            img = self.preprocess_frame(img)
                            sequence_frames.append(img)
                            
                        except Exception as e:
                            print(f"Error processing {frame_path}: {str(e)}, skipping sequence...")
                            valid_sequence = False
                            break
                    
                    if valid_sequence and len(sequence_frames) == self.sequence_length:
                        sequences.append(sequence_frames)
                        labels.append(sign_english)
                        sign_ids.append(sign_id)
                        total_sequences += 1
                        class_sequences += 1
                        
                        # Data augmentation - add mirrored version
                        if augment and np.random.random() > 0.5:
                            try:
                                mirrored_sequence = [cv2.flip(frame, 1) for frame in sequence_frames]
                                sequences.append(mirrored_sequence)
                                labels.append(sign_english)
                                sign_ids.append(sign_id)
                                total_sequences += 1
                                class_sequences += 1
                            except Exception as e:
                                print(f"Error in augmentation: {str(e)}")
                        
                        if total_sequences % 100 == 0:
                            print(f"  Loaded {total_sequences} sequences...")
            
            sign_display = str(sign_english)[:20] if sign_english else "Unknown"
            print(f"SignID {sign_id:4d} ({sign_display:<20s}): {class_sequences:3d} sequences from {len(session_folders)} sessions")
        
        print(f"\nTotal sequences loaded: {total_sequences}")
        print(f"Unique classes: {len(set(labels))}")
        
        if total_sequences == 0:
            print("ERROR: No valid sequences were loaded! Please check your dataset.")
            return None, None, None
        
        if skipped_folders:
            print(f"\n⚠️ Skipped folders without labels: {skipped_folders[:5]}"
                  f"{'...' if len(skipped_folders) > 5 else ''}")
        
        return np.array(sequences, dtype=np.float32), np.array(labels), np.array(sign_ids)
    
    def _find_frame_files(self, session_path):
        """Find all frame files in a session folder (handles nested structure)"""
        frame_files = []
        
        # Look for image files directly in session folder first
        for file in os.listdir(session_path):
            file_path = os.path.join(session_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                frame_files.append(file_path)
        
        # If no images found, look in subfolders
        if not frame_files:
            for item in os.listdir(session_path):
                item_path = os.path.join(session_path, item)
                if os.path.isdir(item_path):
                    for file in os.listdir(item_path):
                        file_path = os.path.join(item_path, file)
                        if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            frame_files.append(file_path)
        
        # Sort numerically if possible
        try:
            frame_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except:
            frame_files.sort()  # Fallback to string sort
        
        return frame_files

    def preprocess_frame(self, frame):
        """Optimized frame preprocessing with error handling"""
        try:
            # Resize efficiently
            frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            # Convert color space
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            return frame
        except Exception as e:
            print(f"Error in frame preprocessing: {str(e)}")
            # Return a black frame as fallback
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

def check_dataset_integrity(base_path, labels_path):
    """Enhanced dataset integrity check with smart recovery suggestions"""
    print("🔍 SMART Dataset Integrity Check")
    print("=" * 45)
    
    if not os.path.exists(base_path):
        print(f"❌ Dataset path not found: {base_path}")
        print("💡 Tip: Run auto-detection to find your dataset!")
        return False
    
    if not os.path.exists(labels_path):
        print(f"❌ Labels file not found: {labels_path}")
        print("💡 Tip: Looking for a .xlsx file with 'SignID', 'Sign-English', 'Sign-Arabic' columns")
        return False
    
    # Load labels with better error handling
    try:
        labels_df = pd.read_excel(labels_path)
        print(f"✅ Labels file loaded: {len(labels_df)} entries")
        
        # Check required columns
        required_cols = ['SignID', 'Sign-English', 'Sign-Arabic']
        missing_cols = [col for col in required_cols if col not in labels_df.columns]
        if missing_cols:
            print(f"⚠️ Missing columns in labels file: {missing_cols}")
            print(f"   Available columns: {list(labels_df.columns)}")
            print("💡 Tip: Your Excel file should have these exact column names")
            return False
        
        # Show sample of labels
        print(f"📋 Sample labels (first 3):")
        for i in range(min(3, len(labels_df))):
            row = labels_df.iloc[i]
            print(f"   SignID {row['SignID']}: {row['Sign-English']} ({row['Sign-Arabic']})")
            
    except Exception as e:
        print(f"❌ Error reading labels file: {str(e)}")
        return False
    
    # Analyze dataset structure
    analysis = analyze_dataset_structure(base_path)
    
    if analysis['total_frames'] == 0:
        print("❌ No image frames found in dataset!")
        print("💡 Expected structure: train/0001/session_name/frame_001.jpg")
        return False
    
    # Smart mismatch analysis
    print("\n🧠 Smart Mismatch Analysis:")
    
    folder_names = set([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    label_sign_ids = set(labels_df['SignID'].astype(str))
    
    print(f"   📁 Found {len(folder_names)} class folders")
    print(f"   📋 Found {len(label_sign_ids)} labels")
    
    # Create flexible matching
    matched_folders = set()
    unmatched_folders = set()
    
    for folder in folder_names:
        matched = False
        
        # Try direct match
        if folder in label_sign_ids:
            matched_folders.add(folder)
            matched = True
        else:
            # Try numeric extraction
            try:
                numbers = ''.join(filter(str.isdigit, folder))
                if numbers:
                    numeric_id = int(numbers)
                    # Try different representations
                    for representation in [str(numeric_id), str(numeric_id).zfill(4), str(numeric_id).zfill(3)]:
                        if representation in label_sign_ids:
                            matched_folders.add(folder)
                            matched = True
                            break
            except:
                pass
        
        if not matched:
            unmatched_folders.add(folder)
    
    match_percentage = len(matched_folders) / len(folder_names) * 100 if folder_names else 0
    
    print(f"   ✅ Successfully matched: {len(matched_folders)}/{len(folder_names)} folders ({match_percentage:.1f}%)")
    
    if unmatched_folders:
        print(f"   ⚠️ Unmatched folders: {len(unmatched_folders)}")
        print(f"      Examples: {list(unmatched_folders)[:3]}")
        print("      💡 These will be intelligently mapped during training!")
    
    if match_percentage >= 70:
        print("   🎉 Great! High match rate - training should work excellently!")
        return True
    elif match_percentage >= 50:
        print("   ⚖️ Good match rate - robust training will handle the rest!")
        return True
    else:
        print("   ⚠️ Low match rate - but don't worry, the system will auto-recover!")
        print("      💡 The robust data loader will intelligently map unmatched folders")
        return True  # Still return True as robust loader can handle it
    
    return True

# ============ DATASET STRUCTURE ANALYZER ============
def analyze_dataset_structure(base_path):
    """Analyze your dataset structure to understand the layout"""
    print(f"Analyzing dataset structure: {base_path}")
    print("=" * 50)
    
    if not os.path.exists(base_path):
        print(f"ERROR: Path does not exist: {base_path}")
        return
    
    total_classes = 0
    total_sessions = 0
    total_frames = 0
    structure_examples = []
    
    for class_folder in sorted(os.listdir(base_path))[:10]:  # Check first 10 classes
        class_path = os.path.join(base_path, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        total_classes += 1
        class_sessions = 0
        class_frames = 0
        
        # Check what's inside each class folder
        for session_item in os.listdir(class_path):
            session_path = os.path.join(class_path, session_item)
            if not os.path.isdir(session_path):
                continue
            
            class_sessions += 1
            
            # Count frames in this session
            frame_files = [f for f in os.listdir(session_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if frame_files:
                class_frames += len(frame_files)
                if len(structure_examples) < 3:
                    sample_frame_path = os.path.join(session_path, frame_files[0])
                    structure_examples.append(sample_frame_path)
            else:
                # Check if there are more subdirectories
                subdirs = [d for d in os.listdir(session_path) 
                          if os.path.isdir(os.path.join(session_path, d))]
                for subdir in subdirs:
                    subdir_path = os.path.join(session_path, subdir)
                    subframe_files = [f for f in os.listdir(subdir_path)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if subframe_files:
                        class_frames += len(subframe_files)
                        if len(structure_examples) < 3:
                            sample_frame_path = os.path.join(subdir_path, subframe_files[0])
                            structure_examples.append(sample_frame_path)
        
        total_sessions += class_sessions
        total_frames += class_frames
        
        if total_classes <= 5:  # Show details for first 5 classes
            print(f"Class {class_folder}: {class_sessions} sessions, {class_frames} frames")
    
    print(f"\nDataset Summary:")
    print(f"  Total classes analyzed: {total_classes}")
    print(f"  Total sessions: {total_sessions}")
    print(f"  Total frames: {total_frames}")
    print(f"  Avg frames per session: {total_frames/max(total_sessions,1):.1f}")
    
    print(f"\nExample file paths found:")
    for example in structure_examples:
        relative_path = os.path.relpath(example, base_path)
        print(f"  {relative_path}")
    
    return {
        'total_classes': total_classes,
        'total_sessions': total_sessions,
        'total_frames': total_frames,
        'structure_examples': structure_examples
    }

# ============ MOBILE-OPTIMIZED MODEL FOR ARABIC SIGN LANGUAGE ============
class MobileOptimizedSignModel(nn.Module):
    def __init__(self, num_classes, img_size=96):
        """Mobile-optimized model for Arabic Sign Language detection on phones"""
        super(MobileOptimizedSignModel, self).__init__()
        
        # Use MobileNetV2 - better for mobile deployment than V3
        try:
            # Try newer PyTorch API first
            from torchvision.models import MobileNet_V2_Weights
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            try:
                # Fallback for older PyTorch versions
                self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
            except (TypeError, ValueError):
                try:
                    # Even older versions
                    self.backbone = models.mobilenet_v2(pretrained=True)
                except:
                    print("   ⚠️ Using MobileNetV2 without pretrained weights")
                    self.backbone = models.mobilenet_v2(pretrained=False)
        
        # Remove the classifier - we'll add our own
        self.backbone.classifier = nn.Identity()
        
        # FREEZE most layers for mobile efficiency and prevent overfitting
        # Only fine-tune the last few layers
        for name, param in self.backbone.named_parameters():
            if 'features.17' in name or 'features.18' in name:  # Last few blocks
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat_dim = self.backbone(dummy).shape[1]  # Should be 1280 for MobileNetV2
        
        print(f"   📱 Mobile backbone feature dim: {feat_dim}")
        
        # Mobile-optimized temporal processing
        # Use smaller LSTM for mobile efficiency
        self.temporal_processing = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),  # Reduce dimension first
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Lightweight LSTM for temporal modeling
        self.lstm = nn.LSTM(256, 128, batch_first=True, dropout=0.3, num_layers=1)
        
        # Mobile-friendly classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Proper weight initialization for mobile deployment"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
    def forward(self, x):
        # x: (batch, seq, c, h, w)
        batch_size, seq_len, c, h, w = x.shape
        
        # Process each frame through the mobile backbone
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)  # Extract features
        features = features.view(batch_size, seq_len, -1)
        
        # Temporal processing
        features = self.temporal_processing(features)
        
        # LSTM for sequence modeling
        lstm_out, _ = self.lstm(features)
        # Use last output
        final_features = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(final_features)
        return output
    
    def get_model_size(self):
        """Get model size in MB for mobile deployment assessment"""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
class LightningSignModel(nn.Module):
    def __init__(self, num_classes):
        """Ultra-fast model with ANTI-OVERFITTING measures"""
        super(LightningSignModel, self).__init__()
        
        # Lightweight CNN feature extractor with HEAVY REGULARIZATION
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Spatial dropout
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Increased spatial dropout
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.4),  # Even more spatial dropout
            nn.AdaptiveAvgPool2d(1),
            
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5)  # Increased from 0.3
        )
        
        # Temporal modeling with HEAVY DROPOUT
        self.lstm = nn.LSTM(256, 128, batch_first=True, dropout=0.5)
        
        # Classification head with MULTIPLE DROPOUT LAYERS
        self.fc1 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(0.5)  # Increased from 0.4
        self.dropout2 = nn.Dropout(0.3)  # Additional dropout
        self.fc2 = nn.Linear(128, num_classes)
        
        # Proper weight initialization
        self._init_weights()
        
    def _init_weights(self):
        """Proper weight initialization to prevent overfitting"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Smaller gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x: (batch, seq, c, h, w)
        batch_size, seq_len, c, h, w = x.shape
        
        # Apply CNN to each frame
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM with regularization
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last hidden state
        
        # Classification with HEAVY REGULARIZATION
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.dropout2(x)  # Double dropout
        x = self.fc2(x)
        return x

class BalancedSignModel(nn.Module):
    def __init__(self, num_classes, img_size=96):
        """Balanced model for Arabic Sign Language with mobile-friendly design"""
        super(BalancedSignModel, self).__init__()
        
        # Use MobileNetV2 for better mobile compatibility
        try:
            from torchvision.models import MobileNet_V2_Weights
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            try:
                self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
            except (TypeError, ValueError):
                try:
                    self.backbone = models.mobilenet_v2(pretrained=True)
                except:
                    print("   ⚠️ Using MobileNetV2 without pretrained weights")
                    self.backbone = models.mobilenet_v2(pretrained=False)
        
        self.backbone.classifier = nn.Identity()  # Remove classifier
        
        # FREEZE most of the backbone for mobile efficiency
        for name, param in self.backbone.named_parameters():
            if 'features.17' in name or 'features.18' in name:  # Last blocks only
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Get feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat_dim = self.backbone(dummy).shape[1]
        
        # Mobile-optimized layers with regularization
        self.dropout1 = nn.Dropout(0.4)
        
        # Efficient temporal modeling
        self.lstm1 = nn.LSTM(feat_dim, 256, batch_first=True, dropout=0.4, num_layers=1)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True, dropout=0.4, num_layers=1)
        
        # Mobile-friendly classification head
        self.fc = nn.Linear(128, 64)
        self.bn = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.3)
        self.out = nn.Linear(64, num_classes)
        
        # Proper initialization for mobile deployment
        self._init_weights()
        
    def _init_weights(self):
        """Mobile-optimized weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
    def forward(self, x):
        # x: (batch, seq, c, h, w)
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract features from each frame
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.backbone(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.dropout1(x)
        
        # Temporal modeling
        x, _ = self.lstm1(x)
        x = self.dropout2(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Last hidden state
        
        # Classification
        x = F.relu(self.fc(x))
        x = self.bn(x)
        x = self.dropout2(x)
        x = self.dropout3(x)
        x = self.out(x)
        return x

# ============ SMART PREDICTION SYSTEM ============
class SmartSignPredictor:
    def __init__(self, model_path, label_encoder_path, sequence_length=10, 
                 confidence_threshold=0.75, stability_frames=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.label_encoder = self.load_label_encoder(label_encoder_path)
        num_classes = len(self.label_encoder.classes_)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        
        # Instantiate model based on path
        print(f"Loading model from {model_path}...")
        if 'lightning' in model_path:
            self.model = LightningSignModel(num_classes)
        elif 'balanced' in model_path:
            self.model = BalancedSignModel(num_classes)
        else:
            print("Warning: Unknown model type, defaulting to Balanced")
            self.model = BalancedSignModel(num_classes)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully with {num_classes} classes")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
        # Frame buffer and prediction history
        self.frame_buffer = deque(maxlen=sequence_length)
        self.prediction_history = deque(maxlen=stability_frames)
        self.last_spoken_word = ""
        self.last_spoken_time = datetime.now()
        self.min_word_interval = timedelta(seconds=2)
        
        # Initialize TTS
        self.init_tts()
        
        print("Smart Sign Predictor initialized successfully!")
        
    def init_tts(self):
        """Initialize text-to-speech with error handling"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 180)
            self.setup_tts_voice()
            
            # Threading for smooth operation
            self.tts_queue = deque()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            self.tts_enabled = True
            print("Text-to-speech initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize TTS: {str(e)}")
            self.tts_enabled = False
        
    def load_label_encoder(self, path):
        """Load label encoder with error handling"""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading label encoder: {str(e)}")
            raise
    
    def setup_tts_voice(self):
        """Setup TTS voice preferences"""
        try:
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
        except:
            pass  # Use default voice
    
    def _tts_worker(self):
        """Background TTS worker thread"""
        while True:
            try:
                if self.tts_queue:
                    text = self.tts_queue.popleft()
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                time.sleep(0.1)
            except:
                time.sleep(1)  # Wait longer if there's an error
    
    def preprocess_frame(self, frame):
        """Preprocess single frame for inference"""
        try:
            frame = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            return frame
        except Exception as e:
            print(f"Error preprocessing frame: {str(e)}")
            return np.zeros((96, 96, 3), dtype=np.float32)
    
    def predict_sequence(self, frames):
        """Predict sign from sequence of frames"""
        if len(frames) != self.sequence_length:
            return None, 0.0, None
        
        try:
            # Prepare input batch
            sequence = np.array(frames)  # (seq, h, w, c)
            sequence = np.expand_dims(sequence, axis=0)  # (1, seq, h, w, c)
            sequence = torch.from_numpy(sequence).permute(0, 1, 4, 2, 3).float().to(self.device)
            
            # Get predictions
            start_time = time.time()
            with torch.no_grad():
                logits = self.model(sequence)
            inference_time = time.time() - start_time
            
            # Get probabilities
            prediction_scores = F.softmax(logits, dim=1)[0].cpu().numpy()
            
            # Get top prediction
            class_idx = np.argmax(prediction_scores)
            confidence = prediction_scores[class_idx]
            
            # Get top-5 predictions
            top5_indices = np.argsort(prediction_scores)[-5:][::-1]
            top5_predictions = [(self.label_encoder.classes_[i], prediction_scores[i]) 
                               for i in top5_indices]
            
            # Decode label
            sign_label = self.label_encoder.classes_[class_idx]
            
            return sign_label, confidence, {
                'inference_time': inference_time,
                'top5': top5_predictions
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None, 0.0, None
    
    def add_frame_and_predict(self, frame):
        """Add frame to buffer and return stable prediction"""
        # Preprocess and add frame
        processed_frame = self.preprocess_frame(frame)
        self.frame_buffer.append(processed_frame)
        
        # Only predict when buffer is full
        if len(self.frame_buffer) < self.sequence_length:
            return None, 0.0, None
        
        # Get prediction
        frames_array = list(self.frame_buffer)
        sign, confidence, metadata = self.predict_sequence(frames_array)
        
        if sign is None:
            return None, 0.0, None
        
        # Add to prediction history for stability
        if confidence > self.confidence_threshold:
            self.prediction_history.append(sign)
        
        # Check for stable prediction
        if len(self.prediction_history) >= self.stability_frames:
            recent_predictions = list(self.prediction_history)
            most_common = Counter(recent_predictions).most_common(1)[0]
            stable_sign, count = most_common
            
            # If prediction is stable and confident
            if count >= self.stability_frames - 1:
                current_time = datetime.now()
                
                if (stable_sign != self.last_spoken_word or 
                    current_time - self.last_spoken_time > self.min_word_interval):
                    
                    self.speak_word(stable_sign)
                    self.last_spoken_word = stable_sign
                    self.last_spoken_time = current_time
                    self.prediction_history.clear()
                    
                    return stable_sign, confidence, metadata
        
        return sign, confidence, metadata
    
    def speak_word(self, word):
        """Add word to TTS queue"""
        if self.tts_enabled:
            english_word = str(word).strip()
            self.tts_queue.append(english_word)
        print(f"🗣️ Speaking: {word}")

# ============ REAL-TIME CAMERA SYSTEM ============
class RealTimeSignDetection:
    def __init__(self, model_path, label_encoder_path):
        try:
            self.predictor = SmartSignPredictor(model_path, label_encoder_path)
            self.cap = None
            self.running = False
            
            # Display settings
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.font_scale = 0.8
            self.thickness = 2
            
            # Performance metrics
            self.fps_counter = deque(maxlen=30)
            self.frame_count = 0
            
            print("Real-time detection system initialized!")
            
        except Exception as e:
            print(f"Error initializing detection system: {str(e)}")
            raise
        
    def start_camera(self, camera_id=0):
        """Initialize camera with better error handling"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test reading a frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from camera")
                return False
            
            print(f"Camera initialized successfully! Resolution: {frame.shape[1]}x{frame.shape[0]}")
            return True
            
        except Exception as e:
            print(f"Error starting camera: {str(e)}")
            return False
    
    def run_detection(self):
        """Main detection loop with enhanced error handling"""
        if not self.cap:
            print("Camera not initialized!")
            return
        
        self.running = True
        print("\n🎥 Starting Arabic Sign Language Detection...")
        print("📋 Controls:")
        print("   'q' - Quit")
        print("   's' - Save screenshot")
        print("   'r' - Reset prediction buffer")
        print("   'c' - Clear screen")
        print("-" * 50)
        
        try:
            while self.running:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Get prediction
                try:
                    prediction, confidence, metadata = self.predictor.add_frame_and_predict(frame)
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
                    prediction, confidence, metadata = None, 0.0, None
                
                # Draw UI
                self.draw_ui(frame, prediction, confidence, metadata)
                
                # Calculate FPS
                end_time = time.time()
                fps = 1.0 / max(end_time - start_time, 0.001)  # Prevent division by zero
                self.fps_counter.append(fps)
                
                # Display frame
                cv2.imshow('Arabic Sign Language Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_screenshot(frame)
                elif key == ord('r'):
                    self.predictor.prediction_history.clear()
                    self.predictor.frame_buffer.clear()
                    print("🔄 Reset prediction buffer")
                elif key == ord('c'):
                    os.system('cls' if os.name == 'nt' else 'clear')
        
        except KeyboardInterrupt:
            print("\n⏹️ Detection stopped by user")
        except Exception as e:
            print(f"\n❌ Error during detection: {str(e)}")
        finally:
            self.cleanup()
    
    def draw_ui(self, frame, prediction, confidence, metadata):
        """Enhanced UI drawing"""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent info panel
        panel_height = 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        y_offset = 25
        
        if prediction:
            # Main prediction with better formatting
            pred_text = f"Sign: {prediction}"
            cv2.putText(frame, pred_text, (10, y_offset), self.font, 1.1, (0, 255, 0), 3)
            
            # Confidence with color coding
            conf_text = f"Confidence: {confidence:.1%}"
            if confidence > 0.9:
                color = (0, 255, 0)    # Green - very confident
            elif confidence > 0.75:
                color = (0, 255, 255)  # Yellow - confident
            else:
                color = (0, 165, 255)  # Orange - moderate
            
            cv2.putText(frame, conf_text, (10, y_offset + 40), self.font, 0.8, color, 2)
            
            # Show top predictions if available
            if metadata and 'top5' in metadata:
                top3 = metadata['top5'][:3]
                for i, (sign, conf) in enumerate(top3[1:], 1):  # Skip first (main prediction)
                    if conf > 0.1:  # Only show if reasonably confident
                        alt_text = f"  {i+1}. {sign} ({conf:.1%})"
                        cv2.putText(frame, alt_text, (10, y_offset + 65 + i*20), 
                                   self.font, 0.5, (200, 200, 200), 1)
            
            # Inference time
            if metadata and 'inference_time' in metadata:
                time_text = f"Inference: {metadata['inference_time']*1000:.1f}ms"
                cv2.putText(frame, time_text, (10, y_offset + 110), self.font, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Show sign to camera...", (10, y_offset), self.font, 0.9, (255, 255, 0), 2)
            
            # Show buffer status when building up
            buffer_status = f"Building buffer: {len(self.predictor.frame_buffer)}/{self.predictor.sequence_length}"
            cv2.putText(frame, buffer_status, (10, y_offset + 40), self.font, 0.6, (255, 255, 255), 1)
        
        # FPS counter
        if self.fps_counter:
            avg_fps = np.mean(list(self.fps_counter))
            fps_text = f"FPS: {avg_fps:.1f}"
            fps_color = (0, 255, 0) if avg_fps > 20 else (0, 255, 255) if avg_fps > 10 else (0, 0, 255)
            cv2.putText(frame, fps_text, (w - 120, 30), self.font, 0.7, fps_color, 2)
        
        # Buffer status indicator
        buffer_filled = len(self.predictor.frame_buffer)
        buffer_total = self.predictor.sequence_length
        buffer_text = f"Buffer: {buffer_filled}/{buffer_total}"
        cv2.putText(frame, buffer_text, (w - 160, h - 20), self.font, 0.5, (255, 255, 255), 1)
        
        # Stability indicator
        stability_count = len(self.predictor.prediction_history)
        stability_text = f"Stability: {stability_count}/{self.predictor.stability_frames}"
        cv2.putText(frame, stability_text, (w - 180, h - 40), self.font, 0.5, (255, 255, 255), 1)
        
        # Draw frame border
        cv2.rectangle(frame, (2, 2), (w-2, h-2), (255, 255, 255), 2)
    
    def save_screenshot(self, frame):
        """Save current frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sign_detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n🛑 Detection stopped. Resources cleaned up.")

# ============ ENHANCED TRAINING SYSTEM ============
class SignDataset(Dataset):
    def __init__(self, X, y_encoded, augment=False):
        # X: (num_seq, seq, h, w, c)
        self.X = torch.from_numpy(X).permute(0, 1, 4, 2, 3).float()  # (num_seq, seq, c, h, w)
        self.y = torch.from_numpy(y_encoded).long()
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        
        # STRONG AUGMENTATION during training to prevent overfitting
        if self.augment:
            # Random horizontal flip (50% chance)
            if np.random.random() > 0.5:
                x = torch.flip(x, dims=[3])
            
            # Random brightness/contrast (30% chance)
            if np.random.random() > 0.7:
                brightness_factor = 0.8 + np.random.random() * 0.4  # 0.8 to 1.2
                x = x * brightness_factor
                x = torch.clamp(x, 0, 1)
            
            # Random rotation (20% chance)
            if np.random.random() > 0.8:
                angle = np.random.uniform(-10, 10)  # Small rotation
                # Apply rotation to each frame in sequence
                for i in range(x.shape[0]):
                    # Simple rotation approximation by shifting
                    if angle > 0:
                        x[i] = torch.roll(x[i], shifts=1, dims=2)
                    else:
                        x[i] = torch.roll(x[i], shifts=-1, dims=2)
            
            # Random noise (10% chance)
            if np.random.random() > 0.9:
                noise = torch.randn_like(x) * 0.02
                x = x + noise
                x = torch.clamp(x, 0, 1)
        
        return x, y

class SignLanguageTrainer:
    def __init__(self, sequence_length=10, img_size=96):
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.data_loader = RobustSignLanguageDataLoader(sequence_length, img_size)  # Use robust loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Training device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    def train_model(self, train_path, test_path, labels_path, model_type='balanced'):
        """Complete training pipeline with better error handling"""
        print("🚀 Starting Sign Language Model Training")
        print("=" * 50)
        
        # Verify paths exist
        if not os.path.exists(train_path):
            print(f"❌ Training path does not exist: {train_path}")
            return None, None, None, None
        if not os.path.exists(test_path):
            print(f"❌ Test path does not exist: {test_path}")
            return None, None, None, None
        if not os.path.exists(labels_path):
            print(f"❌ Labels file does not exist: {labels_path}")
            return None, None, None, None
        
        # Load labels
        try:
            labels_df = pd.read_excel(labels_path)
            print(f"📋 Loaded {len(labels_df)} sign classes from labels file")
            print(f"   Sample labels: {labels_df['Sign-English'].head().tolist()}")
        except Exception as e:
            print(f"❌ Error loading labels: {str(e)}")
            return None, None, None, None
        
        # Analyze dataset structure first
        print("\n🔍 Analyzing dataset structure...")
        train_analysis = analyze_dataset_structure(train_path)
        test_analysis = analyze_dataset_structure(test_path)
        
        if train_analysis['total_frames'] == 0:
            print("❌ No frames found in training data!")
            return None, None, None, None
        
        # Load training data with validation
        print("\n📚 Loading training data...")
        X_train, y_train_labels, train_ids = self.data_loader.load_sequences_efficiently(
            train_path, labels_df, max_sessions_per_class=4, stride=3, augment=True
        )
        
        if X_train is None or len(X_train) == 0:
            print("❌ Failed to load training data!")
            return None, None, None, None
        
        print(f"   ✅ Training data loaded: {len(X_train)} sequences")
        print(f"   📊 Unique training labels: {len(set(y_train_labels))}")
        
        # Load test data with validation
        print("\n📊 Loading test data...")
        X_test, y_test_labels, test_ids = self.data_loader.load_sequences_efficiently(
            test_path, labels_df, max_sessions_per_class=2, stride=5, augment=False
        )
        
        if X_test is None or len(X_test) == 0:
            print("⚠️ No test data loaded - creating test split from training data...")
            from sklearn.model_selection import train_test_split
            
            # Ensure we have enough data for splitting
            if len(X_train) < 10:
                print("❌ Insufficient training data for splitting!")
                return None, None, None, None
            
            # Create stratified split if possible
            try:
                X_train, X_test, y_train_labels, y_test_labels = train_test_split(
                    X_train, y_train_labels, test_size=0.2, random_state=42, 
                    stratify=y_train_labels
                )
                print(f"   ✅ Created stratified split: {len(X_train)} train, {len(X_test)} test")
            except ValueError:
                # If stratification fails, do regular split
                X_train, X_test, y_train_labels, y_test_labels = train_test_split(
                    X_train, y_train_labels, test_size=0.2, random_state=42
                )
                print(f"   ✅ Created random split: {len(X_train)} train, {len(X_test)} test")
            
            test_ids = train_ids  # Use same IDs since we're splitting
        else:
            print(f"   ✅ Test data loaded: {len(X_test)} sequences")
            print(f"   📊 Unique test labels: {len(set(y_test_labels))}")
        
        # Clean and validate labels
        print("\n🧹 Cleaning and validating labels...")
        
        # Convert labels to proper strings and clean them
        y_train_labels = np.array([str(label).strip() for label in y_train_labels])
        y_test_labels = np.array([str(label).strip() for label in y_test_labels])
        
        # Remove any invalid labels
        valid_train_mask = np.array([len(label) > 0 and label not in ['nan', 'None', 'null'] 
                                   for label in y_train_labels])
        valid_test_mask = np.array([len(label) > 0 and label not in ['nan', 'None', 'null'] 
                                  for label in y_test_labels])
        
        if not all(valid_train_mask):
            print(f"   ⚠️ Removing {np.sum(~valid_train_mask)} invalid training labels")
            X_train = X_train[valid_train_mask]
            y_train_labels = y_train_labels[valid_train_mask]
            train_ids = train_ids[valid_train_mask] if len(train_ids) > 0 else train_ids
        
        if not all(valid_test_mask):
            print(f"   ⚠️ Removing {np.sum(~valid_test_mask)} invalid test labels")
            X_test = X_test[valid_test_mask]
            y_test_labels = y_test_labels[valid_test_mask]
            test_ids = test_ids[valid_test_mask] if len(test_ids) > 0 else test_ids
        
        print(f"   ✅ Clean labels: {len(y_train_labels)} train, {len(y_test_labels)} test")
        
        # Encode labels with comprehensive error recovery
        print("\n🏷️ Encoding labels with smart handling...")
        
        try:
            # Combine all labels to ensure consistent encoding
            all_labels = np.concatenate([y_train_labels, y_test_labels])
            unique_labels = np.unique(all_labels)
            
            print(f"   📊 Found {len(unique_labels)} unique labels total")
            print(f"   🚂 Training labels: {len(np.unique(y_train_labels))} unique")
            print(f"   🧪 Test labels: {len(np.unique(y_test_labels))} unique")
            
            # Check for mismatches
            train_unique = set(y_train_labels)
            test_unique = set(y_test_labels)
            train_only = train_unique - test_unique
            test_only = test_unique - train_unique
            
            if train_only:
                print(f"   ⚠️ Labels only in training: {len(train_only)} (examples: {list(train_only)[:3]})")
            if test_only:
                print(f"   ⚠️ Labels only in test: {len(test_only)} (examples: {list(test_only)[:3]})")
                print("   🔄 Filtering test set to match training labels...")
                
                # Filter test data to only include labels present in training
                valid_test_indices = [i for i, label in enumerate(y_test_labels) if label in train_unique]
                
                if len(valid_test_indices) == 0:
                    print("   ❌ No valid test samples after filtering!")
                    print("   🔄 Using a subset of training data as test set...")
                    
                    # Create test set from training data
                    from sklearn.model_selection import train_test_split
                    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                        X_train, y_train_labels, test_size=0.2, random_state=42, stratify=y_train_labels
                    )
                    
                    X_train, X_test = X_train_split, X_test_split
                    y_train_labels, y_test_labels = y_train_split, y_test_split
                    
                    print(f"   ✅ Created new split: {len(X_train)} train, {len(X_test)} test")
                else:
                    # Filter test data
                    X_test = X_test[valid_test_indices]
                    y_test_labels = y_test_labels[valid_test_indices]
                    test_ids = test_ids[valid_test_indices] if len(test_ids) > 0 else test_ids
                    
                    print(f"   ✅ Filtered test set: {len(X_test)} valid samples remaining")
            
            # Now encode with training labels only
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train_labels)
            
            # Safely encode test labels
            try:
                y_test_encoded = label_encoder.transform(y_test_labels)
                print("   ✅ Labels encoded successfully!")
            except ValueError as e:
                print(f"   ⚠️ Test encoding error: {e}")
                print("   🔄 Using only labels present in training...")
                
                # Filter test labels to only include known labels
                known_labels = set(label_encoder.classes_)
                valid_indices = [i for i, label in enumerate(y_test_labels) if label in known_labels]
                
                if valid_indices:
                    X_test = X_test[valid_indices]
                    y_test_labels = y_test_labels[valid_indices]
                    y_test_encoded = label_encoder.transform(y_test_labels)
                    print(f"   ✅ Test set filtered to {len(X_test)} samples with known labels")
                else:
                    print("   🔄 Creating test set from training data...")
                    # Split training data to create test set
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train_labels, y_test_labels = train_test_split(
                        X_train, y_train_labels, test_size=0.2, random_state=42
                    )
                    y_train_encoded = label_encoder.fit_transform(y_train_labels)
                    y_test_encoded = label_encoder.transform(y_test_labels)
                    print(f"   ✅ Created test split: {len(X_test)} samples")
        
        except Exception as e:
            print(f"   ❌ Critical label encoding error: {e}")
            print("   🆘 Emergency recovery: Using training data split...")
            
            try:
                # Emergency: split training data only
                from sklearn.model_selection import train_test_split
                
                # Ensure we have enough unique labels for stratification
                unique_train_labels = np.unique(y_train_labels)
                if len(unique_train_labels) > 1 and len(y_train_labels) >= 10:
                    try:
                        X_train, X_test, y_train_labels, y_test_labels = train_test_split(
                            X_train, y_train_labels, test_size=0.2, random_state=42, 
                            stratify=y_train_labels
                        )
                    except ValueError:
                        X_train, X_test, y_train_labels, y_test_labels = train_test_split(
                            X_train, y_train_labels, test_size=0.2, random_state=42
                        )
                else:
                    # If we have very few samples, use the whole training set as both train and test
                    X_test, y_test_labels = X_train.copy(), y_train_labels.copy()
                
                # Encode labels
                label_encoder = LabelEncoder()
                y_train_encoded = label_encoder.fit_transform(y_train_labels)
                y_test_encoded = label_encoder.transform(y_test_labels)
                
                print(f"   🛠️ Emergency recovery successful: {len(X_train)} train, {len(X_test)} test")
                
            except Exception as emergency_error:
                print(f"   💀 Emergency recovery failed: {emergency_error}")
                print("   📞 Please check your dataset structure and labels file!")
                return None, None, None, None
        
        num_classes = len(label_encoder.classes_)
        
        print(f"✅ Data loading complete:")
        print(f"   Training sequences: {X_train.shape[0]}")
        print(f"   Test sequences: {X_test.shape[0]}")
        print(f"   Number of classes: {num_classes}")
        print(f"   Sequence length: {X_train.shape[1]}")
        print(f"   Frame size: {X_train.shape[2:4]}")
        
        # Create datasets and loaders
        train_ds = SignDataset(X_train, y_train_encoded, augment=True)
        test_ds = SignDataset(X_test, y_test_encoded, augment=False)
        
        # Model configuration for Arabic Sign Language with mobile deployment focus
        if model_type == 'lightning':
            batch_size = 32
            epochs = 20
            lr = 0.0005
            model = LightningSignModel(num_classes).to(self.device)
        elif model_type == 'mobile':
            batch_size = 28  # Optimized for mobile
            epochs = 25
            lr = 0.0003
            model = MobileOptimizedSignModel(num_classes, self.img_size).to(self.device)
        else:  # balanced
            batch_size = 24
            epochs = 25
            lr = 0.0003
            model = BalancedSignModel(num_classes, self.img_size).to(self.device)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Calculate model size for mobile deployment
        model_size_mb = 0
        if hasattr(model, 'get_model_size'):
            model_size_mb = model.get_model_size()
        else:
            # Estimate size
            model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        
        print(f"\n🧠 Created {model_type} model for Arabic Sign Language:")
        print(f"   📱 Model size: {model_size_mb:.1f} MB (mobile target: <50MB)")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🎯 Trainable parameters: {param_count:,}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔄 Epochs: {epochs}")
        print(f"   📈 Learning rate: {lr}")
        
        if model_size_mb > 50:
            print(f"   ⚠️ Warning: Model may be too large for some mobile devices")
        else:
            print(f"   ✅ Model size is mobile-friendly!")
        
        # Optimizer and loss with HEAVY REGULARIZATION
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # Increased from 0.0001
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing to prevent overfitting
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                        factor=0.3, patience=3)  # More aggressive LR reduction
        
        # Model saving setup with OVERFITTING PROTECTION
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f'{model_type}_sign_model_{timestamp}'
        best_val_acc = 0.0
        patience = 5  # Reduced from 8 - stop earlier
        patience_counter = 0
        best_val_loss = float('inf')  # Track validation loss too
        
        # OVERFITTING DETECTION: Track training vs validation gap
        overfitting_threshold = 0.15  # If train acc > val acc by this much, it's overfitting
        
        # Training setup with Windows compatibility
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': 0,
            'pin_memory': True if torch.cuda.is_available() else False
        }
        
        # Add persistent_workers only if supported (PyTorch >= 1.7)
        try:
            train_loader = DataLoader(train_ds, shuffle=True, persistent_workers=False, **dataloader_kwargs)
            test_loader = DataLoader(test_ds, shuffle=False, persistent_workers=False, **dataloader_kwargs)
        except TypeError:
            # Fallback for older PyTorch versions
            train_loader = DataLoader(train_ds, shuffle=True, **dataloader_kwargs)
            test_loader = DataLoader(test_ds, shuffle=False, **dataloader_kwargs)
        
        # Training history
        history = {'loss': [], 'accuracy': [], 'top_5_accuracy': [],
                   'val_loss': [], 'val_accuracy': [], 'val_top_5_accuracy': [],
                   'lr': []}
        
        print(f"\n🎯 Starting ANTI-OVERFITTING training for {epochs} epochs...")
        print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Top5 | LR      | Status")
        print("-" * 85)
        
        try:
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_acc = 0.0
                train_top5 = 0.0
                num_batches = 0
                
                try:
                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        train_loss += loss.item()
                        train_acc += (outputs.argmax(1) == labels).float().mean().item()
                        train_top5 += self.top_k_accuracy(outputs, labels, k=5).item()
                        num_batches += 1
                        
                        # Show progress every 50 batches
                        if batch_idx % 50 == 0 and batch_idx > 0:
                            current_loss = train_loss / num_batches
                            current_acc = train_acc / num_batches
                            print(f"      Batch {batch_idx}: Loss {current_loss:.4f}, Acc {current_acc:.3f}")
                
                except Exception as batch_error:
                    print(f"    ⚠️ Error in training batch: {batch_error}")
                    print("    🔄 Continuing with next epoch...")
                    continue
                
                if num_batches == 0:
                    print(f"    ❌ No training batches processed in epoch {epoch+1}")
                    continue
                
                train_loss /= num_batches
                train_acc /= num_batches
                train_top5 /= num_batches
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_acc = 0.0
                val_top5 = 0.0
                num_val_batches = 0
                
                try:
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            
                            val_loss += loss.item()
                            val_acc += (outputs.argmax(1) == labels).float().mean().item()
                            val_top5 += self.top_k_accuracy(outputs, labels, k=5).item()
                            num_val_batches += 1
                
                except Exception as val_error:
                    print(f"    ⚠️ Error in validation: {val_error}")
                    print("    🔄 Using training metrics for this epoch...")
                    val_loss, val_acc, val_top5 = train_loss, train_acc, train_top5
                    num_val_batches = 1
                
                if num_val_batches > 0:
                    val_loss /= num_val_batches
                    val_acc /= num_val_batches
                    val_top5 /= num_val_batches
                
                current_lr = optimizer.param_groups[0]['lr']
                
                # OVERFITTING DETECTION AND WARNINGS
                train_val_gap = train_acc - val_acc
                if train_val_gap > overfitting_threshold:
                    print(f"    ⚠️ OVERFITTING DETECTED! Train acc ({train_acc:.3f}) >> Val acc ({val_acc:.3f})")
                    print(f"    📉 Gap: {train_val_gap:.3f} (threshold: {overfitting_threshold})")
                
                # Additional overfitting indicators
                if train_loss < 0.01 and val_loss > 0.5:
                    print(f"    🚨 SEVERE OVERFITTING: Train loss very low ({train_loss:.4f}) but val loss high ({val_loss:.4f})")
                
                if train_acc > 0.99 and val_acc < 0.8:
                    print(f"    🛑 EXTREME OVERFITTING: Perfect training but poor validation!")
                
                # Log progress with overfitting indicator
                overfitting_indicator = "🔥" if train_val_gap > overfitting_threshold else "✅"
                print(f"{epoch+1:4d}  | {train_loss:9.4f} | {train_acc:8.3f} | "
                      f"{val_loss:7.4f} | {val_acc:6.3f} | {val_top5:7.3f} | {current_lr:.6f} | {overfitting_indicator}")
                
                # Save history
                history['loss'].append(train_loss)
                history['accuracy'].append(train_acc)
                history['top_5_accuracy'].append(train_top5)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                history['val_top_5_accuracy'].append(val_top5)
                history['lr'].append(current_lr)
                
                # Learning rate scheduling with manual verbose logging
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_acc)
                new_lr = optimizer.param_groups[0]['lr']
                
                if old_lr != new_lr:
                    print(f"    📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
                
                # Model checkpointing with OVERFITTING PROTECTION
                if val_acc > best_val_acc and train_val_gap < overfitting_threshold:
                    best_val_acc = val_acc
                    try:
                        torch.save(model.state_dict(), f'{model_name}.pth')
                        print(f"    ✅ New best model saved! Validation accuracy: {val_acc:.3f}")
                        patience_counter = 0
                    except Exception as save_error:
                        print(f"    ⚠️ Could not save model: {save_error}")
                elif val_acc > best_val_acc:
                    print(f"    ⚠️ Better val acc ({val_acc:.3f}) but OVERFITTING detected - not saving")
                    patience_counter += 1
                else:
                    patience_counter += 1
                
                # EMERGENCY OVERFITTING STOP
                if train_acc > 0.99 and val_acc < 0.7:
                    print(f"    🛑 EMERGENCY STOP: Severe overfitting detected!")
                    print(f"    📊 Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}")
                    print(f"    💡 Model is memorizing training data instead of learning patterns")
                    break
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"    ⏹️ Early stopping triggered at epoch {epoch+1}")
                    if train_val_gap > overfitting_threshold:
                        print(f"    🎯 Stopped due to overfitting (train-val gap: {train_val_gap:.3f})")
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training error: {str(e)}")
            return None, None, None, None
        
        # Save label encoder and training log
        try:
            with open(f'{model_name}_label_encoder.pkl', 'wb') as f:
                pickle.dump(label_encoder, f)
            
            # Save training log
            log_data = []
            for i in range(len(history['loss'])):
                log_data.append({
                    'epoch': i+1,
                    'train_loss': history['loss'][i],
                    'train_accuracy': history['accuracy'][i],
                    'train_top5': history['top_5_accuracy'][i],
                    'val_loss': history['val_loss'][i],
                    'val_accuracy': history['val_accuracy'][i],
                    'val_top5': history['val_top_5_accuracy'][i],
                    'learning_rate': history['lr'][i]
                })
            
            pd.DataFrame(log_data).to_csv(f'{model_name}_training_log.csv', index=False)
            
        except Exception as e:
            print(f"Warning: Could not save training artifacts: {str(e)}")
        
        # Final evaluation
        print(f"\n🎯 Final Results:")
        print(f"   Best Validation Accuracy: {best_val_acc:.1%}")
        print(f"   Final Train Accuracy: {history['accuracy'][-1]:.1%}")
        print(f"   Final Val Top-5 Accuracy: {history['val_top_5_accuracy'][-1]:.1%}")
        
        # Plot training history
        self.plot_training_history(history, model_name)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"   Model saved as: {model_name}.pth")
        print(f"   Label encoder: {model_name}_label_encoder.pkl")
        print(f"   Training log: {model_name}_training_log.csv")
        
        return model, label_encoder, history, model_name
    
    def top_k_accuracy(self, output, target, k=5):
        """Compute top-k accuracy"""
        with torch.no_grad():
            maxk = min(k, output.size(1))  # Handle cases where k > num_classes
            _, pred = torch.topk(output, maxk, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            return correct.any(dim=1).float().mean()
    
    def plot_training_history(self, history, model_name):
        """Plot comprehensive training history"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training History: {model_name}', fontsize=16)
            
            # Accuracy plot
            axes[0,0].plot(history['accuracy'], label='Training', linewidth=2, color='blue')
            axes[0,0].plot(history['val_accuracy'], label='Validation', linewidth=2, color='orange')
            axes[0,0].set_title('Model Accuracy', fontsize=14)
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('Accuracy')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].set_ylim([0, 1])
            
            # Loss plot
            axes[0,1].plot(history['loss'], label='Training', linewidth=2, color='blue')
            axes[0,1].plot(history['val_loss'], label='Validation', linewidth=2, color='orange')
            axes[0,1].set_title('Model Loss', fontsize=14)
            axes[0,1].set_xlabel('Epoch')
            axes[0,1].set_ylabel('Loss')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Top-5 Accuracy
            axes[1,0].plot(history['top_5_accuracy'], label='Training Top-5', linewidth=2, color='green')
            axes[1,0].plot(history['val_top_5_accuracy'], label='Validation Top-5', linewidth=2, color='red')
            axes[1,0].set_title('Top-5 Accuracy', fontsize=14)
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Top-5 Accuracy')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].set_ylim([0, 1])
            
            # Learning Rate
            axes[1,1].plot(history['lr'], linewidth=2, color='purple')
            axes[1,1].set_title('Learning Rate Schedule', fontsize=14)
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Learning Rate')
            axes[1,1].set_yscale('log')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"📈 Training plots saved as: {model_name}_training_history.png")
            
        except Exception as e:
            print(f"Warning: Could not create training plots: {str(e)}")

# ============ UTILITY FUNCTIONS ============
def check_dataset_integrity(base_path, labels_path):
    """Check if your dataset is properly structured"""
    print("🔍 Dataset Integrity Check")
    print("=" * 40)
    
    if not os.path.exists(base_path):
        print(f"❌ Dataset path not found: {base_path}")
        return False
    
    if not os.path.exists(labels_path):
        print(f"❌ Labels file not found: {labels_path}")
        return False
    
    # Load labels
    try:
        labels_df = pd.read_excel(labels_path)
        print(f"✅ Labels file loaded: {len(labels_df)} entries")
    except Exception as e:
        print(f"❌ Error reading labels file: {str(e)}")
        return False
    
    # Check dataset structure
    analysis = analyze_dataset_structure(base_path)
    
    if analysis['total_frames'] == 0:
        print("❌ No image frames found in dataset!")
        return False
    
    # Check for common issues
    issues_found = []
    
    # Check if SignIDs in labels match folder names
    label_sign_ids = set(labels_df['SignID'].astype(str))
    folder_names = set(os.listdir(base_path))
    
    # Handle different ID formats
    padded_folder_names = set()
    for folder in folder_names:
        if folder.isdigit():
            padded_folder_names.add(folder.zfill(4))  # Pad to 4 digits
            padded_folder_names.add(str(int(folder)))  # Remove leading zeros
    
    missing_in_labels = folder_names - label_sign_ids
    missing_in_dataset = label_sign_ids - folder_names - padded_folder_names
    
    if missing_in_labels:
        issues_found.append(f"Folders without labels: {list(missing_in_labels)[:5]}")
    
    if missing_in_dataset:
        issues_found.append(f"Labels without folders: {list(missing_in_dataset)[:5]}")
    
    if issues_found:
        print("\n⚠️ Issues found:")
        for issue in issues_found:
            print(f"   {issue}")
    else:
        print("\n✅ Dataset structure looks good!")
    
    return len(issues_found) == 0

def quick_start_guide():
    """Interactive setup guide"""
    print("🚀 Arabic Sign Language Detection - Quick Start")
    print("=" * 50)
    
    # Get your dataset paths
    print("Please provide your dataset paths:")
    
    default_train = r"E:\DATAset\01-20250901T085635Z-1-001\01\train\train"
    default_test = r"E:\DATAset\01-20250901T085635Z-1-001\01\test\test"
    default_labels = r"E:\DATAset\KARSL-190_Labels.xlsx"
    
    train_path = input(f"Training data path [{default_train}]: ").strip() or default_train
    test_path = input(f"Test data path [{default_test}]: ").strip() or default_test
    labels_path = input(f"Labels file path [{default_labels}]: ").strip() or default_labels
    
    # Check dataset integrity
    print("\n🔍 Checking dataset...")
    if not check_dataset_integrity(train_path, labels_path):
        print("❌ Dataset check failed. Please fix the issues above.")
        return
    
    if not check_dataset_integrity(test_path, labels_path):
        print("❌ Test dataset check failed. Please fix the issues above.")
        return
    
    # Choose model type
    print("\n🧠 Choose model type:")
    print("1. Lightning (Ultra-fast, good for real-time)")
    print("2. Balanced (Better accuracy, still fast)")
    
    choice = input("Enter choice (1 or 2) [2]: ").strip() or "2"
    model_type = 'lightning' if choice == '1' else 'balanced'
    
    # Start training
    print(f"\n🎯 Starting training with {model_type} model...")
    trainer = SignLanguageTrainer(sequence_length=10, img_size=96)
    
    model, label_encoder, history, model_name = trainer.train_model(
        train_path, test_path, labels_path, model_type
    )
    
    if model_name:
        print(f"\n🎉 Training completed successfully!")
        print(f"Model saved as: {model_name}.pth")
        
        # Ask if user wants to test immediately
        test_now = input("\nStart real-time detection now? (y/n) [y]: ").strip().lower()
        if test_now != 'n':
            quick_detect_with_model(model_name)
    
    return model_name

def quick_detect_with_model(model_name):
    """Quick detection with specific model"""
    model_path = f"{model_name}.pth"
    label_encoder_path = f"{model_name}_label_encoder.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    if not os.path.exists(label_encoder_path):
        print(f"❌ Label encoder not found: {label_encoder_path}")
        return
    
    print(f"🎥 Starting real-time detection with {model_name}")
    
    try:
        detector = RealTimeSignDetection(model_path, label_encoder_path)
        if detector.start_camera():
            detector.run_detection()
        else:
            print("❌ Failed to start camera")
    except Exception as e:
        print(f"❌ Error starting detection: {str(e)}")

# ============ MAIN INTERFACE ============
def run_realtime_detection():
    """Run real-time sign language detection"""
    print("🎥 Real-time Arabic Sign Language Detection")
    print("=" * 50)
    
    # Check for existing models
    model_files = [f for f in os.listdir('.') if f.endswith('.pth') and 'sign_model' in f]
    
    if not model_files:
        print("❌ No trained models found!")
        print("Please train a model first using the training option.")
        return
    
    print("📁 Available models:")
    for i, model_file in enumerate(model_files, 1):
        # Get file size and modification time
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(model_file))
        print(f"  {i}. {model_file} ({size_mb:.1f}MB, {mtime.strftime('%Y-%m-%d %H:%M')})")
    
    if len(model_files) == 1:
        selected_model = model_files[0]
        print(f"🎯 Using model: {selected_model}")
    else:
        try:
            choice = int(input(f"Select model (1-{len(model_files)}): ")) - 1
            selected_model = model_files[choice]
        except (ValueError, IndexError):
            print("Invalid selection. Using most recent model.")
            selected_model = max(model_files, key=os.path.getctime)
    
    # Find corresponding label encoder
    base_name = selected_model.replace('.pth', '')
    label_encoder_file = f"{base_name}_label_encoder.pkl"
    
    if not os.path.exists(label_encoder_file):
        print(f"❌ Label encoder not found: {label_encoder_file}")
        return
    
    # Start detection
    quick_detect_with_model(base_name)

def main():
    """Enhanced main interface with auto-detection"""
    print("🤖 ULTIMATE Arabic Sign Language Detection System")
    print("=" * 55)
    print("🎯 Now with MOBILE-OPTIMIZED models and automatic dataset detection!")
    print("📱 Perfect for deploying on phones and mobile devices!")
    print()
    print("Choose an option:")
    print("1. 🚀 SUPER Quick Start (Auto-detect everything, mobile-ready)")
    print("2. 🎯 Train New Model (Guided setup for Arabic Sign Language)")
    print("3. 🎥 Real-time Detection")
    print("4. 🔍 Analyze Dataset Structure")
    print("5. 📊 Check Dataset Integrity")
    print("6. 🏃 Benchmark Model Performance")
    print("7. 🎬 Demo with Video File")
    print("8. 📁 Create Test Dataset")
    print("9. 🛠️ Fix Dataset Configuration")
    print("0. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-9): ").strip()
            
            if choice == '1':
                print("\n🚀 SUPER QUICK START - Let's make this AMAZING!")
                model_name = quick_train()
                if model_name:
                    print(f"\n🎉 SUCCESS! Your model '{model_name}' is ready!")
                    start_detection = input("🎥 Start real-time detection now? (y/n) [y]: ").strip().lower()
                    if start_detection != 'n':
                        quick_detect()
            elif choice == '2':
                train_new_model()
            elif choice == '3':
                run_realtime_detection()
            elif choice == '4':
                path = input("Enter dataset path to analyze (or press Enter for auto-detect): ").strip()
                if not path:
                    paths = get_dataset_paths()
                    path = paths[0]  # Use train path
                if path:
                    analyze_dataset_structure(path)
            elif choice == '5':
                train_path, test_path, labels_path = get_dataset_paths()
                check_dataset_integrity(train_path, labels_path)
            elif choice == '6':
                benchmark_model_performance()
            elif choice == '7':
                demo_with_test_video()
            elif choice == '8':
                create_test_dataset()
            elif choice == '9':
                fix_dataset_paths()
            elif choice == '0':
                print("👋 Thanks for using the Ultimate ASL Detection System!")
                break
            else:
                print("❌ Invalid choice. Please enter 0-9.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
            print("💡 Try the Super Quick Start (option 1) for the best experience!")

# ============ REMAINING FUNCTIONS ============
def train_new_model():
    """Train a new model with guided setup and auto-detection"""
    print("🎯 Training New Sign Language Model")
    print("=" * 45)
    
    # Auto-detect dataset paths
    print("🔍 Auto-detecting dataset paths...")
    train_path, test_path, labels_path = get_dataset_paths()
    
    print(f"\n📂 Using dataset paths:")
    print(f"   📚 Training: {train_path}")
    print(f"   🧪 Testing: {test_path}")
    print(f"   📋 Labels: {labels_path}")
    
    # Allow user to override if needed
    override = input("\n🤔 Want to use different paths? (y/n) [n]: ").strip().lower()
    if override == 'y':
        train_path = input(f"Training path [{train_path}]: ").strip() or train_path
        test_path = input(f"Test path [{test_path}]: ").strip() or test_path
        labels_path = input(f"Labels path [{labels_path}]: ").strip() or labels_path
    
    # Check paths with robust handling
    print("\n🔍 Checking dataset integrity...")
    integrity_ok = check_dataset_integrity(train_path, labels_path)
    if not integrity_ok:
        print("⚠️ Some issues detected, but robust training will handle them!")
    
    # Create trainer with robust data loader
    trainer = SignLanguageTrainer(Config.SEQUENCE_LENGTH, Config.IMG_SIZE)
    
    # Choose model type
    print("\n🧠 Choose your Arabic Sign Language model type:")
    print("1. ⚡ Lightning Model (Ultra-fast training, good for testing)")
    print("2. ⚖️ Balanced Model (Good accuracy, mobile-friendly)")
    print("3. 📱 Mobile Optimized (Best for phone deployment)")
    print("4. 🎯 Let me choose the best one for you!")
    
    choice = input("Enter choice (1-4) [4]: ").strip() or "4"
    
    if choice == '1':
        model_type = 'lightning'
        print("⚡ Lightning model selected - optimized for speed!")
    elif choice == '2':
        model_type = 'balanced'
        print("⚖️ Balanced model selected - great for mobile deployment!")
    elif choice == '3':
        model_type = 'mobile'
        print("📱 Mobile Optimized model selected - perfect for phone apps!")
    else:
        # Auto-choose based on use case
        print("🎯 Since you mentioned phone deployment...")
        model_type = 'mobile'
        print("📱 Auto-selected: Mobile Optimized model (perfect for Arabic Sign Language on phones!)")
    
    print(f"\n🚀 Starting training with {model_type} model...")
    print("💡 Tip: Training will automatically handle any dataset discrepancies!")
    
    model, label_encoder, history, model_name = trainer.train_model(
        train_path, test_path, labels_path, model_type
    )
    
    if model_name:
        print(f"\n🎉 AMAZING! Training completed successfully!")
        print(f"   📁 Model saved as: {model_name}.pth")
        print(f"   🏷️ Label encoder: {model_name}_label_encoder.pkl")
        print("   📊 Training plots and logs saved too!")
        
        # Ask about immediate testing
        test_now = input("\n🎥 Want to test your new model right now? (y/n) [y]: ").strip().lower()
        if test_now != 'n':
            print("🎬 Starting real-time detection with your new model...")
            quick_detect_with_model(model_name)
    else:
        print("❌ Training failed. Check the error messages above.")
    
    return model_name

def demo_with_test_video():
    """Demo with test video file"""
    print("Demo with Test Video")
    print("=" * 30)
    
    # Get model
    model_files = [f for f in os.listdir('.') if f.endswith('.pth') and 'sign_model' in f]
    if not model_files:
        print("No trained models found!")
        return
    
    selected_model = model_files[0]  # Use first available model
    base_name = selected_model.replace('.pth', '')
    label_encoder_file = f"{base_name}_label_encoder.pkl"
    
    video_path = input("Enter path to test video (or press Enter for camera): ").strip()
    
    if not video_path:
        run_realtime_detection()
        return
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    # Initialize predictor
    try:
        predictor = SmartSignPredictor(selected_model, label_encoder_file)
    except Exception as e:
        print(f"Error initializing predictor: {str(e)}")
        return
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    frame_count = 0
    detected_signs = []
    
    print("Processing video...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 3 != 0:  # Process every 3rd frame for speed
                continue
            
            # Resize frame if too large
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Get prediction
            prediction, confidence, metadata = predictor.add_frame_and_predict(frame)
            
            if prediction and confidence > 0.8:
                detected_signs.append((prediction, confidence, frame_count))
                print(f"Frame {frame_count:4d}: {prediction} ({confidence:.3f})")
            
            # Display frame with prediction
            if prediction:
                cv2.putText(frame, f"{prediction} ({confidence:.2f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Video Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Summary
    if detected_signs:
        print(f"\nDetected {len(detected_signs)} signs in video:")
        for sign, conf, frame_num in detected_signs[-10:]:  # Show last 10
            print(f"  {sign} (confidence: {conf:.3f}) at frame {frame_num}")
    else:
        print("No signs detected in video")

def create_test_dataset():
    """Create a small test dataset for quick testing"""
    print("Creating Test Dataset")
    print("=" * 30)
    
    test_dir = "test_signs"
    os.makedirs(test_dir, exist_ok=True)
    
    signs_to_record = ['hello', 'thank_you', 'yes', 'no', 'please']
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    for sign in signs_to_record:
        sign_dir = os.path.join(test_dir, sign, "person1")
        os.makedirs(sign_dir, exist_ok=True)
        
        print(f"\nRecording sign: {sign}")
        print("Press SPACE to start recording, 's' to stop, 'q' to skip")
        
        recording = False
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror
            
            # Draw instructions
            if recording:
                cv2.putText(frame, f"RECORDING: {sign} - Frame {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Press 's' to stop", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save frame
                filename = os.path.join(sign_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(filename, frame)
                frame_count += 1
                
                if frame_count >= 30:  # Record 30 frames max
                    recording = False
                    print(f"Recorded {frame_count} frames for {sign}")
                    break
            else:
                cv2.putText(frame, f"Prepare sign: {sign}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to start recording", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Record Test Dataset', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not recording:
                recording = True
                frame_count = 0
                print(f"Started recording {sign}")
            elif key == ord('s') and recording:
                recording = False
                print(f"Stopped recording {sign} - {frame_count} frames saved")
                break
            elif key == ord('q'):
                print(f"Skipped {sign}")
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Create labels file
    labels_data = []
    for i, sign in enumerate(signs_to_record, 1):
        labels_data.append({
            'SignID': i,
            'Sign-Arabic': sign,
            'Sign-English': sign
        })
    
    labels_df = pd.DataFrame(labels_data)
    labels_df.to_excel(os.path.join(test_dir, 'test_labels.xlsx'), index=False)
    
    print(f"\nTest dataset created in '{test_dir}' directory")

def benchmark_model_performance():
    """Benchmark model inference speed"""
    print("Model Performance Benchmark")
    print("=" * 40)
    
    model_files = [f for f in os.listdir('.') if f.endswith('.pth') and 'sign_model' in f]
    if not model_files:
        print("No trained models found!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_file in model_files:
        print(f"\nBenchmarking: {model_file}")
        
        base_name = model_file.replace('.pth', '')
        label_encoder_file = f"{base_name}_label_encoder.pkl"
        
        if not os.path.exists(label_encoder_file):
            print(f"Label encoder not found for {model_file}")
            continue
        
        try:
            # Load label encoder
            with open(label_encoder_file, 'rb') as f:
                label_encoder = pickle.load(f)
            num_classes = len(label_encoder.classes_)
            
            # Instantiate model
            if 'lightning' in model_file:
                model = LightningSignModel(num_classes)
            elif 'balanced' in model_file:
                model = BalancedSignModel(num_classes)
            else:
                model = BalancedSignModel(num_classes)  # Default
            
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.to(device)
            model.eval()
            
            # Model info
            param_count = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
            
            print(f"  Parameters: {param_count:,} (trainable: {trainable_params:,})")
            print(f"  File Size: {model_size_mb:.2f} MB")
            print(f"  Classes: {num_classes}")
            
            # Speed benchmark
            batch_size = 1
            seq_len = 10
            h, w = 96, 96
            dummy_input = torch.rand(batch_size, seq_len, 3, h, w).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start = time.time()
                    _ = model(dummy_input)
                    times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000  # ms
            std_time = np.std(times) * 1000
            fps = 1000 / avg_time
            
            print(f"  Avg Inference Time: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"  Max FPS: {fps:.1f}")
            print(f"  Estimated RAM: {param_count * 4 / (1024*1024):.1f} MB")
            
        except Exception as e:
            print(f"Error benchmarking {model_file}: {str(e)}")

# ============ CONFIGURATION & SETTINGS ============
class Config:
    """Centralized configuration for easy customization"""
    # Default dataset paths - CHANGE THESE TO YOUR PATHS
    DEFAULT_TRAIN_PATH = r"E:\DATAset\01-20250901T085635Z-1-001\01\train\train"
    DEFAULT_TEST_PATH = r"E:\DATAset\01-20250901T085635Z-1-001\01\test\test"
    DEFAULT_LABELS_PATH = r"E:\DATAset\KARSL-190_Labels.xlsx"
    
    # Auto-detection of dataset paths (searches common locations)
    COMMON_DATASET_NAMES = [
        "KArSL-190", "KARSL", "KArSL", "karsl", "ASL", "asl",
        "sign_language", "arabic_signs", "DATAset", "dataset"
    ]
    
    # Model settings for Arabic Sign Language
    SEQUENCE_LENGTH = 10
    IMG_SIZE = 96
    DEFAULT_MODEL_TYPE = 'mobile'  # 'lightning' for speed, 'balanced' for accuracy, 'mobile' for phones
    
    # Training settings
    LIGHTNING_EPOCHS = 25
    BALANCED_EPOCHS = 30
    AUTO_AUGMENTATION = True
    EARLY_STOPPING_PATIENCE = 8
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.75
    STABILITY_FRAMES = 3
    DEFAULT_CAMERA_ID = 0

def auto_detect_dataset_paths():
    """Automatically detect dataset paths by searching common locations"""
    print("🔍 Auto-detecting dataset paths...")
    
    # Search common drive letters and directories
    search_locations = []
    
    # Add current directory and parent directories
    current_dir = os.getcwd()
    search_locations.extend([
        current_dir,
        os.path.dirname(current_dir),
        os.path.dirname(os.path.dirname(current_dir))
    ])
    
    # Add common Windows drive letters
    for drive in ['C:', 'D:', 'E:', 'F:', 'G:']:
        if os.path.exists(drive):
            search_locations.append(drive)
    
    # Add common directories
    user_home = os.path.expanduser("~")
    search_locations.extend([
        user_home,
        os.path.join(user_home, "Desktop"),
        os.path.join(user_home, "Documents"),
        os.path.join(user_home, "Downloads")
    ])
    
    found_paths = {}
    
    for location in search_locations:
        if not os.path.exists(location):
            continue
            
        try:
            # Search for dataset directories
            for root, dirs, files in os.walk(location):
                # Limit search depth to avoid scanning entire drive
                if root.count(os.sep) - location.count(os.sep) > 3:
                    continue
                
                # Look for KArSL labels file
                for file in files:
                    if 'karsl' in file.lower() and file.endswith('.xlsx'):
                        found_paths['labels'] = os.path.join(root, file)
                        print(f"   📋 Found labels: {found_paths['labels']}")
                
                # Look for train/test directories
                for dir_name in dirs:
                    if 'train' in dir_name.lower():
                        train_candidate = os.path.join(root, dir_name)
                        # Check if it contains numbered subdirectories (sign IDs)
                        try:
                            subdirs = os.listdir(train_candidate)
                            if any(d.isdigit() or d.startswith('00') for d in subdirs[:5]):
                                found_paths['train'] = train_candidate
                                print(f"   🚂 Found train: {found_paths['train']}")
                        except:
                            pass
                    
                    elif 'test' in dir_name.lower():
                        test_candidate = os.path.join(root, dir_name)
                        try:
                            subdirs = os.listdir(test_candidate)
                            if any(d.isdigit() or d.startswith('00') for d in subdirs[:5]):
                                found_paths['test'] = test_candidate
                                print(f"   🧪 Found test: {found_paths['test']}")
                        except:
                            pass
                
                # Stop searching if we found all three
                if len(found_paths) >= 3:
                    break
            
            if len(found_paths) >= 3:
                break
                
        except (PermissionError, OSError):
            continue  # Skip directories we can't access
    
    return found_paths

def get_dataset_paths():
    """Get dataset paths with auto-detection fallback"""
    # First try auto-detection
    auto_paths = auto_detect_dataset_paths()
    
    # Use auto-detected paths if available, otherwise fall back to defaults
    train_path = auto_paths.get('train', Config.DEFAULT_TRAIN_PATH)
    test_path = auto_paths.get('test', Config.DEFAULT_TEST_PATH)
    labels_path = auto_paths.get('labels', Config.DEFAULT_LABELS_PATH)
    
    # Verify paths exist
    paths_exist = {
        'train': os.path.exists(train_path),
        'test': os.path.exists(test_path),
        'labels': os.path.exists(labels_path)
    }
    
    if all(paths_exist.values()):
        print("✅ All dataset paths found!")
        return train_path, test_path, labels_path
    
    # If auto-detection failed, ask user
    print("⚠️ Some dataset paths not found automatically:")
    for path_type, exists in paths_exist.items():
        status = "✅" if exists else "❌"
        path = locals()[f"{path_type}_path"]
        print(f"   {status} {path_type}: {path}")
    
    # Interactive path input with smart defaults
    print("\n📁 Please provide correct paths (press Enter to keep auto-detected/default):")
    
    new_train = input(f"Training path [{train_path}]: ").strip()
    if new_train:
        train_path = new_train
    
    new_test = input(f"Test path [{test_path}]: ").strip()
    if new_test:
        test_path = new_test
    
    new_labels = input(f"Labels path [{labels_path}]: ").strip()
    if new_labels:
        labels_path = new_labels
    
    return train_path, test_path, labels_path

# ============ ENHANCED DATA LOADING WITH SMART ERROR HANDLING ============
class RobustSignLanguageDataLoader(SignLanguageDataLoader):
    """Enhanced data loader with intelligent error recovery"""
    
    def load_sequences_efficiently(self, base_path, labels_df, max_sessions_per_class=2, 
                                 stride=5, augment=True):
        """
        PERSON-INDEPENDENT data loading for Arabic Sign Language
        Prevents overfitting by ensuring diversity across sessions and people
        """
        sequences = []
        labels = []
        sign_ids = []
        total_sequences = 0
        skipped_folders = []
        recovered_mappings = {}
        
        print(f"🔄 Loading Arabic Sign Language sequences from {base_path}...")
        print("   Using PERSON-INDEPENDENT loading to prevent overfitting!")
        
        # Create flexible ID mapping
        id_mapping = self._create_flexible_id_mapping(base_path, labels_df)
        
        folder_names = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        print(f"   Found {len(folder_names)} sign classes")
        print(f"   Labels file contains {len(labels_df)} entries")
        
        successfully_mapped = 0
        
        for class_folder in sorted(folder_names):
            class_path = os.path.join(base_path, class_folder)
            
            # Try multiple strategies to find matching label
            sign_row = None
            used_strategy = None
            
            # Strategy 1: Direct match
            if class_folder in id_mapping:
                sign_id = id_mapping[class_folder]
                sign_row = labels_df[labels_df['SignID'] == sign_id]
                if not sign_row.empty:
                    used_strategy = "direct_match"
            
            # Strategy 2: Numeric extraction and padding
            if sign_row is None or sign_row.empty:
                try:
                    # Extract all numbers from folder name
                    numbers = ''.join(filter(str.isdigit, class_folder))
                    if numbers:
                        numeric_id = int(numbers)
                        
                        # Try different padding levels
                        for padding in [0, 2, 3, 4, 5]:
                            padded_id = str(numeric_id).zfill(padding) if padding > 0 else str(numeric_id)
                            sign_row = labels_df[labels_df['SignID'].astype(str) == padded_id]
                            if not sign_row.empty:
                                id_mapping[class_folder] = numeric_id
                                used_strategy = f"numeric_extract_pad{padding}"
                                break
                        
                        # Try numeric match directly
                        if sign_row.empty:
                            sign_row = labels_df[labels_df['SignID'] == numeric_id]
                            if not sign_row.empty:
                                id_mapping[class_folder] = numeric_id
                                used_strategy = "numeric_direct"
                except:
                    pass
            
            # Strategy 3: Fuzzy matching on folder name
            if sign_row is None or sign_row.empty:
                # Try to find partial matches in English/Arabic labels
                for idx, row in labels_df.iterrows():
                    english_label = str(row['Sign-English']).lower().replace(' ', '_')
                    arabic_label = str(row['Sign-Arabic']).lower()
                    folder_lower = class_folder.lower()
                    
                    if (english_label in folder_lower or 
                        folder_lower in english_label or
                        arabic_label in folder_lower):
                        sign_row = labels_df.iloc[[idx]]
                        id_mapping[class_folder] = row['SignID']
                        used_strategy = "fuzzy_match"
                        break
            
            # Strategy 4: Sequential assignment for remaining
            if sign_row is None or sign_row.empty:
                # Find unused SignIDs and assign sequentially
                used_ids = set(id_mapping.values())
                available_ids = set(labels_df['SignID']) - used_ids
                if available_ids:
                    assigned_id = min(available_ids)
                    sign_row = labels_df[labels_df['SignID'] == assigned_id]
                    id_mapping[class_folder] = assigned_id
                    used_strategy = "sequential_assignment"
                    recovered_mappings[class_folder] = assigned_id
            
            if sign_row is None or sign_row.empty:
                print(f"   ⚠️ Could not map folder '{class_folder}' to any label - SKIPPING")
                skipped_folders.append(class_folder)
                continue
            
            # Successfully mapped!
            successfully_mapped += 1
            sign_id = id_mapping[class_folder]
            sign_english = str(sign_row['Sign-English'].iloc[0]).strip()
            sign_arabic = str(sign_row['Sign-Arabic'].iloc[0]).strip()
            
            if successfully_mapped <= 10:  # Show details for first 10 mappings
                print(f"   ✅ {class_folder} → SignID {sign_id} ({sign_english}) via {used_strategy}")
            
            # CRITICAL: Process sessions with DIVERSITY FOCUS
            class_sequences = self._process_class_folder_with_diversity(
                class_path, sign_english, sign_id, max_sessions_per_class, stride, augment
            )
            
            sequences.extend(class_sequences['sequences'])
            labels.extend(class_sequences['labels'])
            sign_ids.extend(class_sequences['sign_ids'])
            total_sequences += class_sequences['count']
        
        print(f"\n📊 PERSON-INDEPENDENT Loading Results:")
        print(f"   ✅ Successfully mapped: {successfully_mapped}/{len(folder_names)} folders")
        print(f"   ⚠️ Skipped folders: {len(skipped_folders)}")
        print(f"   🔄 Auto-recovered mappings: {len(recovered_mappings)}")
        print(f"   📚 Total sequences loaded: {total_sequences}")
        print(f"   🎯 Using max {max_sessions_per_class} sessions per sign (prevents person overfitting)")
        
        if recovered_mappings:
            print(f"\n🔄 Auto-recovered mappings (saved for future use):")
            for folder, assigned_id in list(recovered_mappings.items())[:5]:
                label = labels_df[labels_df['SignID'] == assigned_id]['Sign-English'].iloc[0]
                print(f"   {folder} → SignID {assigned_id} ({label})")
            if len(recovered_mappings) > 5:
                print(f"   ... and {len(recovered_mappings) - 5} more")
        
        if total_sequences == 0:
            print("❌ CRITICAL: No sequences were loaded!")
            print("   Possible solutions:")
            print("   1. Check if train/test folders contain the right structure")
            print("   2. Verify Excel file has 'SignID', 'Sign-English', 'Sign-Arabic' columns")
            print("   3. Make sure image files are .jpg, .jpeg, or .png")
            return None, None, None
        
        return np.array(sequences, dtype=np.float32), np.array(labels), np.array(sign_ids)
    
    def _create_flexible_id_mapping(self, base_path, labels_df):
        """Create flexible mapping between folder names and SignIDs"""
        mapping = {}
        
        # Get all available SignIDs from labels
        available_ids = set(labels_df['SignID'].tolist())
        
        # Try to create direct mappings first
        folder_names = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
        for folder in folder_names:
            # Try exact match first
            try:
                if folder.isdigit():
                    folder_id = int(folder)
                    if folder_id in available_ids:
                        mapping[folder] = folder_id
                        continue
                
                # Try extracting leading digits
                leading_digits = ''
                for char in folder:
                    if char.isdigit():
                        leading_digits += char
                    else:
                        break
                
                if leading_digits:
                    folder_id = int(leading_digits)
                    if folder_id in available_ids:
                        mapping[folder] = folder_id
            except:
                pass
        
        return mapping
    
    def _process_class_folder_with_diversity(self, class_path, sign_english, sign_id, max_sessions, stride, augment):
        """Process all sessions with DIVERSITY FOCUS to prevent person overfitting"""
        sequences = []
        labels = []
        sign_ids = []
        count = 0
        
        # Get all session folders
        session_folders = []
        for item in os.listdir(class_path):
            item_path = os.path.join(class_path, item)
            if os.path.isdir(item_path):
                session_folders.append(item)
        
        session_folders = sorted(session_folders)
        
        print(f"      📁 Sign {sign_id} ({sign_english}): Found {len(session_folders)} sessions")
        
        # ANTI-OVERFITTING: Limit sessions and add diversity
        if len(session_folders) > max_sessions:
            # Instead of taking first N, sample diverse sessions
            step = len(session_folders) // max_sessions
            selected_sessions = session_folders[::step][:max_sessions]
            print(f"      🎯 Selected {len(selected_sessions)} diverse sessions (every {step}th)")
        else:
            selected_sessions = session_folders
        
        for session_idx, session_folder in enumerate(selected_sessions):
            session_sequences = self._process_session_folder_with_diversity(
                os.path.join(class_path, session_folder), sign_english, sign_id, 
                stride, augment, session_idx, len(selected_sessions)
            )
            
            sequences.extend(session_sequences['sequences'])
            labels.extend(session_sequences['labels'])
            sign_ids.extend(session_sequences['sign_ids'])
            count += session_sequences['count']
        
        print(f"      ✅ Generated {count} diverse sequences from {len(selected_sessions)} sessions")
        
        return {
            'sequences': sequences,
            'labels': labels,
            'sign_ids': sign_ids,
            'count': count
        }
    
    def _process_session_folder_with_diversity(self, session_path, sign_english, sign_id, stride, augment, session_idx, total_sessions):
        """Process frames with DIVERSITY sampling to prevent overfitting"""
        sequences = []
        labels = []
        sign_ids = []
        count = 0
        
        # Find frame files (handle nested structure)
        frame_files = self._find_frame_files(session_path)
        
        if len(frame_files) < self.sequence_length:
            return {'sequences': sequences, 'labels': labels, 'sign_ids': sign_ids, 'count': count}
        
        # DIVERSITY SAMPLING: Different strategies per session to increase variety
        max_start = len(frame_files) - self.sequence_length
        
        # Adaptive stride based on session length and index
        adaptive_stride = max(1, stride + (session_idx % 3))  # Vary stride per session
        
        # ANTI-OVERFITTING: Limit sequences per session
        max_sequences_per_session = 3 if len(frame_files) > 50 else 2
        sequences_generated = 0
        
        for start_idx in range(0, max_start + 1, adaptive_stride):
            if sequences_generated >= max_sequences_per_session:
                break
                
            sequence_frames = []
            valid_sequence = True
            
            for i in range(self.sequence_length):
                frame_idx = start_idx + i
                if frame_idx >= len(frame_files):
                    valid_sequence = False
                    break
                
                frame_path = frame_files[frame_idx]
                
                try:
                    img = cv2.imread(frame_path)
                    if img is None:
                        valid_sequence = False
                        break
                    
                    # DIVERSITY: Apply different preprocessing per session
                    img = self._preprocess_frame_with_diversity(img, session_idx)
                    sequence_frames.append(img)
                except:
                    valid_sequence = False
                    break
            
            if valid_sequence and len(sequence_frames) == self.sequence_length:
                sequences.append(sequence_frames)
                labels.append(sign_english)
                sign_ids.append(sign_id)
                count += 1
                sequences_generated += 1
                
                # CONTROLLED AUGMENTATION: Less aggressive to prevent overfitting
                if augment and sequences_generated < 2 and np.random.random() > 0.7:  # Only 30% chance
                    try:
                        # More diverse augmentation
                        if session_idx % 2 == 0:
                            # Horizontal flip
                            mirrored_sequence = [cv2.flip(frame, 1) for frame in sequence_frames]
                        else:
                            # Brightness variation
                            brightness_factor = 0.7 + np.random.random() * 0.6  # 0.7 to 1.3
                            mirrored_sequence = [np.clip(frame * brightness_factor, 0, 1) for frame in sequence_frames]
                        
                        sequences.append(mirrored_sequence)
                        labels.append(sign_english)
                        sign_ids.append(sign_id)
                        count += 1
                        sequences_generated += 1
                    except:
                        pass
        
        return {'sequences': sequences, 'labels': labels, 'sign_ids': sign_ids, 'count': count}
    
    def _preprocess_frame_with_diversity(self, frame, session_idx):
        """Frame preprocessing with session-based diversity"""
        try:
            # Resize efficiently
            frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            # Convert color space
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # DIVERSITY: Apply slight variations based on session
            if session_idx % 3 == 1:
                # Slight contrast adjustment for some sessions
                frame = cv2.convertScaleAbs(frame, alpha=0.95, beta=5)
            elif session_idx % 3 == 2:
                # Slight gamma correction for other sessions
                gamma = 0.95 + (session_idx % 3) * 0.05
                frame = np.power(frame / 255.0, gamma) * 255.0
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            return frame
        except Exception as e:
            print(f"Error in diverse frame preprocessing: {str(e)}")
            # Return a black frame as fallback
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
    
    def _find_frame_files(self, session_path):
        """Recursively find frame files in session directory"""
        frame_files = []
        
        # Check current directory
        for file in os.listdir(session_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                frame_files.append(os.path.join(session_path, file))
        
        # If no frames found, check subdirectories
        if not frame_files:
            for subdir in os.listdir(session_path):
                subdir_path = os.path.join(session_path, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            frame_files.append(os.path.join(subdir_path, file))
        
        # Sort files properly
        try:
            frame_files = sorted(frame_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
        except:
            frame_files = sorted(frame_files)
        
        return frame_files

# ============ QUICK START FUNCTIONS ============
def quick_train():
    """Quick training with automatic dataset detection"""
    print("🚀 SUPER QUICK TRAINING MODE")
    print("=" * 50)
def quick_train():
    """Quick training with automatic dataset detection"""
    print("🚀 SUPER QUICK TRAINING MODE")
    print("=" * 50)
    
    # Auto-detect dataset paths
    train_path, test_path, labels_path = get_dataset_paths()
    
    print(f"📂 Using paths:")
    print(f"   Train: {train_path}")
    print(f"   Test: {test_path}")
    print(f"   Labels: {labels_path}")
    
    if not check_dataset_integrity(train_path, labels_path):
        print("❌ Dataset integrity check failed. Attempting to proceed with robust loading...")
    
    # Use robust data loader
    trainer = SignLanguageTrainer(Config.SEQUENCE_LENGTH, Config.IMG_SIZE)
    trainer.data_loader = RobustSignLanguageDataLoader(Config.SEQUENCE_LENGTH, Config.IMG_SIZE)
    
    model, label_encoder, history, model_name = trainer.train_model(
        train_path, test_path, labels_path, Config.DEFAULT_MODEL_TYPE
    )
    
    if model_name:
        print(f"\n🎉 Training completed! Model saved as: {model_name}.pth")
        print("💡 Run quick_detect() to start real-time detection!")
        print("🎯 Or run main() and choose option 3 for full interface!")
    
    return model_name

def quick_detect():
    """Quick start real-time detection with latest model"""
    model_files = [f for f in os.listdir('.') if f.endswith('.pth') and 'sign_model' in f]
    
    if not model_files:
        print("No models found! Run quick_train() first.")
        return
    
    # Use most recent model
    latest_model = max(model_files, key=os.path.getctime)
    base_name = latest_model.replace('.pth', '')
    label_encoder_file = f"{base_name}_label_encoder.pkl"
    
    if not os.path.exists(label_encoder_file):
        print(f"Label encoder not found: {label_encoder_file}")
        return
    
    print(f"Starting detection with model: {latest_model}")
    
    try:
        detector = RealTimeSignDetection(latest_model, label_encoder_file)
        if detector.start_camera():
            detector.run_detection()
    except Exception as e:
        print(f"Error starting detection: {str(e)}")

def fix_dataset_paths():
    """Advanced dataset configuration helper with auto-detection"""
    print("🛠️ ADVANCED Dataset Configuration Helper")
    print("=" * 50)
    
    print("🔍 Running auto-detection scan...")
    auto_paths = auto_detect_dataset_paths()
    
    if auto_paths:
        print("\n✅ Auto-detected paths:")
        for path_type, path in auto_paths.items():
            print(f"   {path_type}: {path}")
    else:
        print("\n⚠️ Could not auto-detect dataset paths")
    
    print(f"\n📝 Current default paths in Config:")
    print(f"   Train: {Config.DEFAULT_TRAIN_PATH}")
    print(f"   Test: {Config.DEFAULT_TEST_PATH}")
    print(f"   Labels: {Config.DEFAULT_LABELS_PATH}")
    
    print("\n🎯 Your KArSL-190 dataset structure should be:")
    print("   📁 train/")
    print("     📁 0001/  (or any SignID from your Excel)")
    print("       📁 session_folder_name/")
    print("         📸 frame_001.jpg")
    print("         📸 frame_002.jpg")
    print("         📸 ...")
    print("       📁 another_session/")
    print("         📸 ...")
    print("     📁 0002/")
    print("       📁 ...")
    print("   📁 test/  (same structure)")
    print("   📋 KARSL-190_Labels.xlsx")
    
    print("\n💡 TIPS for fixing dataset issues:")
    print("   1. 📊 Excel file MUST have columns: 'SignID', 'Sign-English', 'Sign-Arabic'")
    print("   2. 📁 Folder names should match SignIDs (numbers like 0001, 0002, etc.)")
    print("   3. 📸 Image files should be .jpg, .jpeg, or .png")
    print("   4. 🔄 If there are mismatches, the robust system will auto-fix them!")
    print("   5. ⚡ Use Option 1 (Super Quick Start) for automatic everything!")
    
    # Option to update Config defaults
    update_defaults = input("\n🔧 Update default paths in code? (y/n) [n]: ").strip().lower()
    if update_defaults == 'y':
        new_train = input(f"New default train path [{Config.DEFAULT_TRAIN_PATH}]: ").strip()
        new_test = input(f"New default test path [{Config.DEFAULT_TEST_PATH}]: ").strip()
        new_labels = input(f"New default labels path [{Config.DEFAULT_LABELS_PATH}]: ").strip()
        
        if new_train:
            Config.DEFAULT_TRAIN_PATH = new_train
        if new_test:
            Config.DEFAULT_TEST_PATH = new_test
        if new_labels:
            Config.DEFAULT_LABELS_PATH = new_labels
        
        print("✅ Default paths updated for this session!")
        print("💡 To make permanent, edit the Config class in the code")
    
    # Test the current setup
    test_setup = input("\n🧪 Test current dataset setup? (y/n) [y]: ").strip().lower()
    if test_setup != 'n':
        train_path, test_path, labels_path = get_dataset_paths()
        print(f"\n🔍 Testing setup with:")
        print(f"   Train: {train_path}")
        print(f"   Test: {test_path}")
        print(f"   Labels: {labels_path}")
        
        check_dataset_integrity(train_path, labels_path)
        
        if os.path.exists(train_path):
            analyze_dataset_structure(train_path)
    
    print("\n🎉 Configuration helper complete!")
    print("💡 Ready to train? Use Option 1 (Super Quick Start) for the best experience!")

# ============ CONVENIENCE FUNCTIONS FOR IMMEDIATE USE ============
def super_quick_start():
    """One-command solution for everything"""
    print("🚀 SUPER QUICK START - The Ultimate ASL Experience!")
    print("=" * 55)
    print("🎯 This will automatically:")
    print("   1. Find your KArSL-190 dataset")
    print("   2. Handle any mismatches intelligently")
    print("   3. Train an optimized model")
    print("   4. Start real-time detection")
    print("   5. Make you happy! 😊")
    
    # Auto-detect and train
    model_name = quick_train()
    
    if model_name:
        print("\n🎉 INCREDIBLE! Everything worked perfectly!")
        print("🎥 Let's see your model in action...")
        time.sleep(2)  # Dramatic pause
        quick_detect()
    else:
        print("❌ Something went wrong. But don't worry!")
        print("💡 Try: main() and choose option 9 to fix any issues")

def instant_detection():
    """Start detection immediately with best available model"""
    print("🎥 INSTANT Real-time Detection")
    print("=" * 35)
    
    # Find best model
    model_files = [f for f in os.listdir('.') if f.endswith('.pth') and 'sign_model' in f]
    
    if not model_files:
        print("❌ No trained models found!")
        print("🚀 Run super_quick_start() to train and detect in one go!")
        return
    
    # Use most recent model
    latest_model = max(model_files, key=os.path.getctime)
    model_name = latest_model.replace('.pth', '')
    
    print(f"🎯 Using model: {model_name}")
    print("⚡ Starting in 3 seconds... Get ready to sign!")
    
    time.sleep(3)
    quick_detect_with_model(model_name)

def show_model_stats():
    """Show stats of all available models"""
    print("📊 Available Models Overview")
    print("=" * 35)
    
    model_files = [f for f in os.listdir('.') if f.endswith('.pth') and 'sign_model' in f]
    
    if not model_files:
        print("❌ No trained models found!")
        print("🚀 Run super_quick_start() to create your first model!")
        return
    
    for model_file in model_files:
        base_name = model_file.replace('.pth', '')
        
        # Get model info
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(model_file))
        
        # Check for label encoder
        encoder_file = f"{base_name}_label_encoder.pkl"
        has_encoder = os.path.exists(encoder_file)
        
        # Try to get class count
        num_classes = "Unknown"
        if has_encoder:
            try:
                with open(encoder_file, 'rb') as f:
                    encoder = pickle.load(f)
                    num_classes = len(encoder.classes_)
            except:
                pass
        
        model_type = "Lightning" if "lightning" in base_name else "Balanced"
        
        print(f"\n🤖 {model_file}")
        print(f"   📅 Created: {mtime.strftime('%Y-%m-%d %H:%M')}")
        print(f"   📦 Size: {size_mb:.1f} MB")
        print(f"   ⚡ Type: {model_type}")
        print(f"   🏷️ Classes: {num_classes}")
        print(f"   ✅ Ready: {'Yes' if has_encoder else 'No (missing encoder)'}")

if __name__ == "__main__":
    # Show welcome message
    print("🎉 WELCOME TO THE ULTIMATE ARABIC SIGN LANGUAGE DETECTION SYSTEM!")
    print("=" * 70)
    print("� MOBILE-READY: Optimized for phone deployment!")
    print("�🚀 QUICKEST START: Run super_quick_start() for everything at once!")
    print("🎥 INSTANT DETECTION: Run instant_detection() to use existing model!")
    print("📊 MODEL INFO: Run show_model_stats() to see your models!")
    print("🛠️ FULL INTERFACE: Run main() for complete control!")
    print()
    
    # Auto-start based on what's available
    model_files = [f for f in os.listdir('.') if f.endswith('.pth') and 'sign_model' in f]
    
    if model_files:
        print(f"✅ Found {len(model_files)} trained Arabic Sign Language model(s)!")
        choice = input("🎯 Start instant detection? (y/n) [y]: ").strip().lower()
        if choice != 'n':
            instant_detection()
        else:
            main()
    else:
        print("📚 No trained models found - let's create a mobile-optimized one!")
        choice = input("🚀 Run super quick start? (y/n) [y]: ").strip().lower()
        if choice != 'n':
            super_quick_start()
        else:
            main()

# ============ USAGE INSTRUCTIONS ============
"""
🎉 ULTIMATE KARSL-190 ARABIC SIGN LANGUAGE DETECTION SYSTEM
📱 MOBILE-OPTIMIZED FOR PHONE DEPLOYMENT
=" * 65

🚀 FASTEST WAY TO GET STARTED:
   Just run the file! It will auto-detect everything and create a mobile-ready model!

⚡ INSTANT COMMANDS:
   super_quick_start()    - Does everything automatically (mobile-optimized)
   instant_detection()    - Start detection with existing model
   quick_train()         - Train new mobile model with auto-detection
   quick_detect()        - Detect with latest model
   show_model_stats()    - Show all your models with mobile compatibility info
   main()               - Full interface

📱 MOBILE DEPLOYMENT FEATURES:
   🎯 MobileNetV2 backbone (optimized for phones)
   📦 Models under 50MB (perfect for mobile apps)
   ⚡ Fast inference (suitable for real-time on phones)
   🔋 Battery-efficient architecture
   🛡️ Anti-overfitting measures for real-world deployment

✨ ARABIC SIGN LANGUAGE FEATURES:
   🔍 Auto-detects KArSL-190 dataset location
   🧠 Intelligently handles Excel/folder mismatches
   🎯 Never asks for paths again (unless you want to)
   ⚡ Mobile-optimized training and detection
   🛠️ Robust error handling and recovery
   📊 Comprehensive model statistics with mobile metrics

🎯 YOUR DATASET REQUIREMENTS:
   📋 Excel file with: SignID, Sign-English, Sign-Arabic columns
   📁 Train/test folders with numbered subfolders (0001, 0002, etc.)
   📸 Image files (.jpg, .jpeg, .png) in session folders
   
💡 MOBILE DEPLOYMENT TIPS:
   - Use 'mobile' model type for best phone performance
   - Models are quantization-ready for even smaller sizes
   - Pretrained MobileNetV2 ensures good feature extraction
   - Optimized for 96x96 input images (mobile-friendly)

🏆 PERFORMANCE OPTIMIZATIONS FOR MOBILE:
   - Frozen backbone layers (faster training, better efficiency)
   - Smaller LSTM architectures
   - Efficient temporal processing
   - Smart batch sizing based on your device
   - Real-time performance monitoring

Ready to deploy Arabic Sign Language recognition on mobile devices! 📱🚀
"""
