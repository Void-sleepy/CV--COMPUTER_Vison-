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
import torchvision.models as models
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# ============ OPTIMIZED DATA LOADING ============
class SignLanguageDataLoader:
    def __init__(self, sequence_length=10, img_size=96):
        self.sequence_length = sequence_length
        self.img_size = img_size
        
    def load_sequences_efficiently(self, base_path, labels_df, max_sessions_per_class=4, 
                                 stride=3, augment=True):
        """
        Load data as proper sequences with smart sampling
        """
        sequences = []
        labels = []
        sign_ids = []
        total_sequences = 0
        
        print(f"Loading sequences from {base_path}...")
        
        for class_folder in sorted(os.listdir(base_path)):
            class_path = os.path.join(base_path, class_folder)
            if not os.path.isdir(class_path):
                continue
                
            try:
                sign_id = int(class_folder)
            except ValueError:
                print(f"Skipping non-numeric folder: {class_folder}")
                continue
            
            # Get label from dataset
            sign_row = labels_df[labels_df['SignID'] == sign_id]
            if sign_row.empty:
                print(f"No label found for SignID {sign_id}")
                continue
            
            sign_english = str(sign_row['Sign-English'].iloc[0])  # Convert to string
            sign_arabic = str(sign_row['Sign-Arabic'].iloc[0])   # Convert to string
            
            # Process each session (person)
            session_folders = [f for f in os.listdir(class_path) 
                             if os.path.isdir(os.path.join(class_path, f))]
            session_folders = sorted(session_folders)[:max_sessions_per_class]
            
            class_sequences = 0
            
            for session_folder in session_folders:
                session_path = os.path.join(class_path, session_folder)
                
                # Get all frames and sort them
                frame_files = [f for f in os.listdir(session_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                frame_files = sorted(frame_files)
                
                if len(frame_files) < self.sequence_length:
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
                        img = cv2.imread(frame_path)
                        
                        if img is None:
                            valid_sequence = False
                            break
                        
                        # Preprocess frame
                        img = self.preprocess_frame(img)
                        sequence_frames.append(img)
                    
                    if valid_sequence and len(sequence_frames) == self.sequence_length:
                        sequences.append(sequence_frames)
                        labels.append(sign_english)
                        sign_ids.append(sign_id)
                        total_sequences += 1
                        class_sequences += 1
                        
                        # Data augmentation - add mirrored version
                        if augment and np.random.random() > 0.5:
                            mirrored_sequence = [cv2.flip(frame, 1) for frame in sequence_frames]
                            sequences.append(mirrored_sequence)
                            labels.append(sign_english)
                            sign_ids.append(sign_id)
                            total_sequences += 1
                            class_sequences += 1
                        
                        if total_sequences % 200 == 0:
                            print(f"  Loaded {total_sequences} sequences...")
            
            sign_display = str(sign_english)[:15] if sign_english else "Unknown"
            print(f"SignID {sign_id:3d} ({sign_display:<15s}): {class_sequences:3d} sequences from {len(session_folders)} sessions")
        
        print(f"\nTotal sequences loaded: {total_sequences}")
        print(f"Unique classes: {len(set(labels))}")
        
        return np.array(sequences, dtype=np.float32), np.array(labels), np.array(sign_ids)
    
    def preprocess_frame(self, frame):
        """Optimized frame preprocessing"""
        # Resize efficiently
        frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        # Convert color space
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        return frame

# ============ OPTIMIZED MODEL ARCHITECTURES ============
class LightningSignModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        """
        Ultra-fast model for real-time inference
        Optimized for speed over accuracy
        """
        super(LightningSignModel, self).__init__()
        
        # Lightweight CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Temporal modeling with lightweight RNN
        self.lstm = nn.LSTM(256, 128, batch_first=True, dropout=0.2)
        
        # Classification head
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: (batch, seq, c, h, w)
        batch_size, seq_len, c, h, w = x.shape
        
        # Apply CNN to each frame
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last hidden state
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Return logits; apply softmax outside if needed

class BalancedSignModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        """
        Balanced model - good accuracy and reasonable speed
        """
        super(BalancedSignModel, self).__init__()
        
        # Use MobileNetV3 as backbone
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Remove classifier
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get feature dim (run a dummy input to find out)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_shape[1], input_shape[2])
            feat_dim = self.backbone(dummy).shape[1]
        
        self.dropout1 = nn.Dropout(0.2)
        
        # Temporal modeling
        self.lstm1 = nn.LSTM(feat_dim, 128, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True, dropout=0.3)
        
        # Classification
        self.fc = nn.Linear(64, 128)
        self.bn = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)
        self.out = nn.Linear(128, num_classes)
        
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
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Last hidden state
        
        # Classification
        x = F.relu(self.fc(x))
        x = self.bn(x)
        x = self.dropout2(x)
        x = self.out(x)
        return x  # Return logits; apply softmax outside if needed

# ============ SMART PREDICTION SYSTEM ============
class SmartSignPredictor:
    def __init__(self, model_path, label_encoder_path, model_class, input_shape, sequence_length=10, 
                 confidence_threshold=0.75, stability_frames=3):
        # Note: Added model_class and input_shape to instantiate the correct model
        self.model = model_class(input_shape, len(pickle.load(open(label_encoder_path, 'rb')).classes_))
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.label_encoder = self.load_label_encoder(label_encoder_path)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        
        # Frame buffer and prediction history
        self.frame_buffer = deque(maxlen=sequence_length)
        self.prediction_history = deque(maxlen=stability_frames)
        self.last_spoken_word = ""
        self.last_spoken_time = datetime.now()
        self.min_word_interval = timedelta(seconds=2)  # Prevent word spam
        
        # TTS engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 180)  # Speech rate
        self.setup_tts_voice()
        
        # Threading for smooth operation
        self.tts_queue = deque()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
        print("Smart Sign Predictor initialized successfully!")
        print(f"Loaded model with {len(self.label_encoder.classes_)} classes")
        
    def load_label_encoder(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def setup_tts_voice(self):
        """Setup TTS voice preferences"""
        voices = self.tts_engine.getProperty('voices')
        # Try to find a suitable voice (preferably female for accessibility)
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
    
    def _tts_worker(self):
        """Background TTS worker thread"""
        while True:
            if self.tts_queue:
                text = self.tts_queue.popleft()
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"TTS error: {e}")
            time.sleep(0.1)
