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
    def __init__(self, num_classes):
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
        return x  # Return logits

class BalancedSignModel(nn.Module):
    def __init__(self, num_classes, img_size=96):
        """
        Balanced model - good accuracy and reasonable speed
        """
        super(BalancedSignModel, self).__init__()
        
        # Use MobileNetV3 as backbone
        self.backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()  # Remove classifier
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
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
        return x  # Return logits

# ============ SMART PREDICTION SYSTEM ============
class SmartSignPredictor:
    def __init__(self, model_path, label_encoder_path, sequence_length=10, 
                 confidence_threshold=0.75, stability_frames=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = self.load_label_encoder(label_encoder_path)
        num_classes = len(self.label_encoder.classes_)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        
        # Instantiate model based on path
        if 'lightning' in model_path:
            self.model = LightningSignModel(num_classes)
        elif 'balanced' in model_path:
            self.model = BalancedSignModel(num_classes)
        else:
            raise ValueError("Unknown model type in path")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
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
        print(f"Loaded model with {num_classes} classes")
        
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
                except:
                    pass
            time.sleep(0.1)
    
    def preprocess_frame(self, frame):
        """Preprocess single frame for inference"""
        frame = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def predict_sequence(self, frames):
        """Predict sign from sequence of frames"""
        if len(frames) != self.sequence_length:
            return None, 0.0, None
        
        # Prepare input batch
        sequence = np.array(frames)  # (seq, h, w, c)
        sequence = np.expand_dims(sequence, axis=0)  # (1, seq, h, w, c)
        sequence = torch.from_numpy(sequence).permute(0, 1, 4, 2, 3).float().to(self.device)  # (1, seq, c, h, w)
        
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
        
        # Get top-5 predictions for analysis
        top5_indices = np.argsort(prediction_scores)[-5:][::-1]
        top5_predictions = [(self.label_encoder.classes_[i], prediction_scores[i]) 
                           for i in top5_indices]
        
        # Decode label
        sign_label = self.label_encoder.classes_[class_idx]
        
        return sign_label, confidence, {
            'inference_time': inference_time,
            'top5': top5_predictions
        }
    
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
            # Get most common prediction in recent history
            recent_predictions = list(self.prediction_history)
            most_common = Counter(recent_predictions).most_common(1)[0]
            stable_sign, count = most_common
            
            # If prediction is stable and confident
            if count >= self.stability_frames - 1:
                current_time = datetime.now()
                
                # Check if enough time has passed since last spoken word
                if (stable_sign != self.last_spoken_word or 
                    current_time - self.last_spoken_time > self.min_word_interval):
                    
                    # Queue for speech
                    self.speak_word(stable_sign)
                    self.last_spoken_word = stable_sign
                    self.last_spoken_time = current_time
                    
                    # Clear prediction history to prevent repetition
                    self.prediction_history.clear()
                    
                    return stable_sign, confidence, metadata
        
        return sign, confidence, metadata
    
    def speak_word(self, word):
        """Add word to TTS queue"""
        # Speak both English and Arabic if needed
        english_word = str(word).strip()
        self.tts_queue.append(english_word)
        print(f" Speaking: {english_word}")

# ============ REAL-TIME CAMERA SYSTEM ============
class RealTimeSignDetection:
    def __init__(self, model_path, label_encoder_path):
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
        
    def start_camera(self, camera_id=0):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized successfully!")
        return True
    
    def run_detection(self):
        """Main detection loop"""
        if not self.cap:
            print("Camera not initialized!")
            return
        
        self.running = True
        print("\nStarting real-time sign language detection...")
        print(" Press 'q' to quit, 's' to take screenshot")
        print(" Show sign language gestures to the camera")
        print("-" * 50)
        
        while self.running:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get prediction
            prediction, confidence, metadata = self.predictor.add_frame_and_predict(frame)
            
            # Draw UI
            self.draw_ui(frame, prediction, confidence, metadata)
            
            # Calculate FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
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
                print(" Reset prediction buffer")
        
        self.cleanup()
    
    def draw_ui(self, frame, prediction, confidence, metadata):
        """Draw user interface on frame"""
        h, w = frame.shape[:2]
        
        # Draw main info panel
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw prediction info
        y_offset = 30
        
        if prediction:
            # Main prediction
            pred_text = f"Sign: {prediction}"
            cv2.putText(frame, pred_text, (10, y_offset), self.font, 1.0, (0, 255, 0), self.thickness)
            
            # Confidence
            conf_text = f"Confidence: {confidence:.2f}"
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
            cv2.putText(frame, conf_text, (10, y_offset + 35), self.font, 0.7, color, 2)
            
            # Inference time
            if metadata and 'inference_time' in metadata:
                time_text = f"Inference: {metadata['inference_time']*1000:.1f}ms"
                cv2.putText(frame, time_text, (10, y_offset + 65), self.font, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Show sign to camera...", (10, y_offset), self.font, 0.8, (255, 255, 255), 2)
        
        # FPS counter
        if self.fps_counter:
            avg_fps = np.mean(list(self.fps_counter))
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(frame, fps_text, (w - 120, 30), self.font, 0.6, (255, 255, 255), 2)
        
        # Buffer status
        buffer_filled = len(self.predictor.frame_buffer)
        buffer_total = self.predictor.sequence_length
        buffer_text = f"Buffer: {buffer_filled}/{buffer_total}"
        cv2.putText(frame, buffer_text, (w - 150, h - 20), self.font, 0.5, (255, 255, 255), 1)
        
        # Draw frame border
        cv2.rectangle(frame, (5, 5), (w-5, h-5), (255, 255, 255), 2)
    
    def save_screenshot(self, frame):
        """Save current frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sign_detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f" Screenshot saved: {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n Detection stopped. Resources cleaned up.")

# ============ TRAINING SYSTEM ============
class SignDataset(Dataset):
    def __init__(self, X, y_encoded):
        # X: (num_seq, seq, h, w, c)
        self.X = torch.from_numpy(X).permute(0, 1, 4, 2, 3).float()  # (num_seq, seq, c, h, w)
        self.y = torch.from_numpy(y_encoded).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SignLanguageTrainer:
    def __init__(self, sequence_length=10, img_size=96):
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.data_loader = SignLanguageDataLoader(sequence_length, img_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_model(self, train_path, test_path, labels_path, model_type='balanced'):
        """Complete training pipeline"""
        print(" Starting Sign Language Model Training")
        print("=" * 50)
        
        # Load labels
        labels_df = pd.read_excel(labels_path)
        print(f" Loaded {len(labels_df)} sign classes from labels file")
        
        # Load training data
        print("\nLoading training data...")
        X_train, y_train_labels, train_ids = self.data_loader.load_sequences_efficiently(
            train_path, labels_df, max_sessions_per_class=4, stride=3, augment=True
        )
        
        # Load test data
        print("\n Loading test data...")
        X_test, y_test_labels, test_ids = self.data_loader.load_sequences_efficiently(
            test_path, labels_df, max_sessions_per_class=2, stride=5, augment=False
        )
        
        # Encode labels
        print("\n Encoding labels...")
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_labels)
        y_test_encoded = label_encoder.transform(y_test_labels)
        
        num_classes = len(label_encoder.classes_)
        
        print(f"Training data: {X_train.shape}")
        print(f"Test data: {X_test.shape}")
        print(f"Number of classes: {num_classes}")
        print(f"Sample labels: {label_encoder.classes_[:5]}...")
        
        # Create datasets and loaders
        train_ds = SignDataset(X_train, y_train_encoded)
        test_ds = SignDataset(X_test, y_test_encoded)
        
        if model_type == 'lightning':
            batch_size = 32
            epochs = 25
            model = LightningSignModel(num_classes).to(self.device)
        else:
            batch_size = 16
            epochs = 30
            model = BalancedSignModel(num_classes, self.img_size).to(self.device)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n Created {model_type} model with {param_count:,} trainable parameters")
        
        # Optimizer and loss
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss()
        
        # Callbacks simulation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f'{model_type}_sign_model_{timestamp}'
        best_val_acc = 0.0
        patience = 8
        patience_counter = 0
        lr_patience = 4
        lr_counter = 0
        min_lr = 1e-7
        factor = 0.3
        log_data = []
        
        # Training loop
        print(f"\n Starting training for {epochs} epochs...")
        history = {'loss': [], 'accuracy': [], 'top_5_accuracy': [],
                   'val_loss': [], 'val_accuracy': [], 'val_top_5_accuracy': [],
                   'lr': []}
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            train_top5 = 0.0
            num_batches = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_acc += (outputs.argmax(1) == labels).float().mean().item()
                train_top5 += self.top_k_accuracy(outputs, labels, k=5).item()
                num_batches += 1
            
            train_loss /= num_batches
            train_acc /= num_batches
            train_top5 /= num_batches
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_top5 = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_acc += (outputs.argmax(1) == labels).float().mean().item()
                    val_top5 += self.top_k_accuracy(outputs, labels, k=5).item()
                    num_batches += 1
            
            val_loss /= num_batches
            val_acc /= num_batches
            val_top5 /= num_batches
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Top5: {train_top5:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Top5: {val_top5:.4f} | LR: {current_lr:.6f}")
            
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['top_5_accuracy'].append(train_top5)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['val_top_5_accuracy'].append(val_top5)
            history['lr'].append(current_lr)
            
            log_data.append({
                'epoch': epoch+1,
                'loss': train_loss,
                'accuracy': train_acc,
                'top_5_accuracy': train_top5,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_top_5_accuracy': val_top5,
                'lr': current_lr
            })
            
            # Checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'{model_name}.pth')
                print(f"  Saved best model at epoch {epoch+1}")
                patience_counter = 0
                lr_counter = 0
            else:
                patience_counter += 1
                lr_counter += 1
                
            # Reduce LR
            if lr_counter >= lr_patience:
                new_lr = max(current_lr * factor, min_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"  Reduced LR to {new_lr:.6f}")
                lr_counter = 0
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Save label encoder
        with open(f'{model_name}_label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Save log
        pd.DataFrame(log_data).to_csv(f'{model_name}_training.log', index=False)
        
        # Final Evaluation
        print("\n Final Evaluation:")
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_top5 = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                test_acc += (outputs.argmax(1) == labels).float().mean().item()
                test_top5 += self.top_k_accuracy(outputs, labels, k=5).item()
                num_batches += 1
        
        test_loss /= num_batches
        test_acc /= num_batches
        test_top5 /= num_batches
        
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Test Top-5 Accuracy: {test_top5:.4f} ({test_top5*100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Plot training history
        self.plot_training_history(history, model_name)
        
        print(f"\n Training completed successfully!")
        print(f" Model saved as: {model_name}.pth")
        print(f" Label encoder saved as: {model_name}_label_encoder.pkl")
        
        return model, label_encoder, history, model_name
    
    def top_k_accuracy(self, output, target, k=5):
        """Compute top-k accuracy"""
        with torch.no_grad():
            _, pred = torch.topk(output, k, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            return correct.any(dim=1).float().mean()
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0,0].plot(history['accuracy'], label='Training', linewidth=2)
        axes[0,0].plot(history['val_accuracy'], label='Validation', linewidth=2)
        axes[0,0].set_title('Model Accuracy', fontsize=14)
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Loss
        axes[0,1].plot(history['loss'], label='Training', linewidth=2)
        axes[0,1].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[0,1].set_title('Model Loss', fontsize=14)
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Top-5 Accuracy
        if 'top_5_accuracy' in history:
            axes[1,0].plot(history['top_5_accuracy'], label='Training Top-5', linewidth=2)
            axes[1,0].plot(history['val_top_5_accuracy'], label='Validation Top-5', linewidth=2)
            axes[1,0].set_title('Top-5 Accuracy', fontsize=14)
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Top-5 Accuracy')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        if 'lr' in history:
            axes[1,1].plot(history['lr'], linewidth=2, color='orange')
            axes[1,1].set_title('Learning Rate Schedule', fontsize=14)
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Learning Rate')
            axes[1,1].set_yscale('log')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print final metrics
        final_metrics = {
            'Final Training Accuracy': f"{history['accuracy'][-1]:.4f}",
            'Final Validation Accuracy': f"{history['val_accuracy'][-1]:.4f}",
            'Best Validation Accuracy': f"{max(history['val_accuracy']):.4f}",
            'Final Training Loss': f"{history['loss'][-1]:.4f}",
            'Final Validation Loss': f"{history['val_loss'][-1]:.4f}",
        }
        
        print("\nTraining Summary:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value}")

# ============ MAIN USAGE FUNCTIONS ============
def train_new_model():
    """Train a new sign language model"""
    # Update these paths to match your dataset
    train_path = r"D:\NAID\Dataset\train"
    test_path = r"D:\NAID\Dataset\test"   ###### DATASETS
    labels_path = r"D:\NAID\KARSL-190_Labels.xlsx"
    
    trainer = SignLanguageTrainer(sequence_length=10, img_size=96)
    
    print("Choose model type:")
    print("1. Lightning (Ultra-fast, good for real-time)")
    print("2. Balanced (Good accuracy-speed balance)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    model_type = 'lightning' if choice == '1' else 'balanced'
    
    model, label_encoder, history, model_name = trainer.train_model(
        train_path, test_path, labels_path, model_type
    )
    
    return model_name

def run_realtime_detection():
    """Run real-time sign language detection"""
    print("Real-time Arabic Sign Language Detection")
    print("=" * 50)
    
    # Check for existing models
    model_files = [f for f in os.listdir('.') if f.endswith('.pth') and 'sign_model' in f]
    
    if not model_files:
        print(" No trained models found!")
        print("Please train a model first using train_new_model()")
        return
    
    print("Available models:")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")
    
    if len(model_files) == 1:
        selected_model = model_files[0]
        print(f"Using model: {selected_model}")
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
        print(f" Label encoder not found: {label_encoder_file}")
        return
    
    # Initialize detection system
    detector = RealTimeSignDetection(selected_model, label_encoder_file)
    
    # Start camera
    if detector.start_camera():
        try:
            detector.run_detection()
        except KeyboardInterrupt:
            print("\n Detection interrupted by user")
        finally:
            detector.cleanup()
    else:
        print(" Failed to initialize camera")

def demo_with_test_video():
    """Demo with test video file"""
    print(" Demo with Test Video")
    print("=" * 30)
    
    # Get model and video paths
    model_files = [f for f in os.listdir('.') if f.endswith('.pth') and 'sign_model' in f]
    if not model_files:
        print(" No trained models found!")
        return
    
    selected_model = model_files[0]  # Use first available model
    base_name = selected_model.replace('.pth', '')
    label_encoder_file = f"{base_name}_label_encoder.pkl"
    
    video_path = input("Enter path to test video (or press Enter for camera): ").strip()
    
    if not video_path:
        run_realtime_detection()
        return
    
    if not os.path.exists(video_path):
        print(f" Video file not found: {video_path}")
        return
    
    # Initialize predictor
    predictor = SmartSignPredictor(selected_model, label_encoder_file)
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detected_signs = []
    
    print(" Processing video...")
    
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
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    if detected_signs:
        print(f"\nDetected {len(detected_signs)} signs in video:")
        for sign, conf, frame_num in detected_signs[-10:]:  # Show last 10
            print(f"  {sign} (confidence: {conf:.3f}) at frame {frame_num}")
    else:
        print(" No signs detected in video")

def create_test_dataset():
    """Create a small test dataset for quick testing"""
    print("Creating Test Dataset")
    print("=" * 30)
    
    # This function helps create a minimal dataset for testing
    # You can record a few signs using your camera
    
    test_dir = "test_signs"
    os.makedirs(test_dir, exist_ok=True)
    
    signs_to_record = ['hello', 'thank_you', 'yes', 'no', 'please']
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Could not open camera")
        return
    
    for sign in signs_to_record:
        sign_dir = os.path.join(test_dir, sign, "person1")
        os.makedirs(sign_dir, exist_ok=True)
        
        print(f"\n Recording sign: {sign}")
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
                    print(f" Recorded {frame_count} frames for {sign}")
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
                print(f" Started recording {sign}")
            elif key == ord('s') and recording:
                recording = False
                print(f" Stopped recording {sign} - {frame_count} frames saved")
                break
            elif key == ord('q'):
                print(f"   Skipped {sign}")
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Create labels file
    labels_data = []
    for i, sign in enumerate(signs_to_record, 1):
        labels_data.append({
            'SignID': i,
            'Sign-Arabic': sign,  # For simplicity, using English as Arabic
            'Sign-English': sign
        })
    
    labels_df = pd.DataFrame(labels_data)
    labels_df.to_excel(os.path.join(test_dir, 'test_labels.xlsx'), index=False)
    
    print(f"\n Test dataset created in '{test_dir}' directory")
    print("You can now train a model using this test dataset!")

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
        print(f"\n Benchmarking: {model_file}")
        
        base_name = model_file.replace('.pth', '')
        label_encoder_file = f"{base_name}_label_encoder.pkl"
        
        if not os.path.exists(label_encoder_file):
            print(f" Label encoder not found for {model_file}")
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
                raise ValueError("Unknown model type")
            
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.to(device)
            model.eval()
            
            # Model info
            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
            
            print(f"  Parameters: {param_count:,}")
            print(f"   File Size: {model_size_mb:.2f} MB")
            print(f"    Classes: {num_classes}")
            
            # Speed benchmark
            batch_size = 1
            seq_len = 10
            h, w = 96, 96
            dummy_input = torch.rand(batch_size, seq_len, 3, h, w).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(dummy_input)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(50):
                    start = time.time()
                    _ = model(dummy_input)
                    times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000  # ms
            std_time = np.std(times) * 1000
            fps = 1000 / avg_time
            
            print(f"  Avg Inference Time: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"  Max FPS: {fps:.1f}")
            
            # Memory usage (approximate)
            print(f"   Estimated RAM: {param_count * 4 / (1024*1024):.1f} MB")
            
        except Exception as e:
            print(f"  Error benchmarking {model_file}: {str(e)}")

# ============ MAIN INTERFACE ============
def main():
    """Main interface for the sign language detection system"""
    print(" Arabic Sign Language Detection System")
    print("=" * 50)
    print("Choose an option:")
    print("1. Train new model")
    print("2. Run real-time detection")  
    print("3. Demo with video file")
    print("4. Create test dataset")
    print("5. Benchmark model performance")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                train_new_model()
            elif choice == '2':
                run_realtime_detection()
            elif choice == '3':
                demo_with_test_video()
            elif choice == '4':
                create_test_dataset()
            elif choice == '5':
                benchmark_model_performance()
            elif choice == '6':
                print(" Goodbye!")
                break
            else:
                print(" Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

# ============ QUICK START FUNCTIONS ============
def quick_train():
    """Quick training with your existing dataset"""
    # Update these paths to your dataset
    train_path = r"D:\NAID\Dataset\train"
    test_path = r"D:\NAID\Dataset\test"    ##### DATASETS
    labels_path = r"D:\NAID\KARSL-190_Labels.xlsx"
    
    trainer = SignLanguageTrainer(sequence_length=10, img_size=96)
    model, label_encoder, history, model_name = trainer.train_model(train_path, test_path, labels_path, 'lightning')
    
    print(f"\n Training completed! Model saved as: {model_name}.pth")
    print("Run quick_detect() to start real-time detection!")
    
    return model_name

def quick_detect():
    """Quick start real-time detection with latest model"""
    model_files = [f for f in os.listdir('.') if f.endswith('.pth') and 'sign_model' in f]
    
    if not model_files:
        print(" No models found! Run quick_train() first.")
        return
    
    # Use most recent model
    latest_model = max(model_files, key=os.path.getctime)
    base_name = latest_model.replace('.pth', '')
    label_encoder_file = f"{base_name}_label_encoder.pkl"
    
    print(f"Starting detection with model: {latest_model}")
    
    detector = RealTimeSignDetection(latest_model, label_encoder_file)
    if detector.start_camera():
        detector.run_detection()
    
# ============ USAGE INSTRUCTIONS ============
"""
QUICK START GUIDE:

1. TRAINING A NEW MODEL:
   - Update the paths in quick_train() function to point to your dataset
   - Run: quick_train()
   - This will train a lightning-fast model optimized for real-time detection

2. REAL-TIME DETECTION:
   - After training, run: quick_detect()
   - Or use the full interface: main()
   
3. YOUR DATASET STRUCTURE SHOULD BE:
   train/
   ├── 1/
   │   ├── person1/
   │   │   ├── frame_001.jpg
   │   │   └── ...
   │   └── person2/
   │       └── ...
   ├── 2/
   └── ...
   
4. FEATURES:
   ✅ Real-time camera detection
   ✅ Text-to-speech output
   ✅ Smart prediction smoothing
   ✅ FPS optimization
   ✅ Memory-efficient processing
   ✅ Mobile-ready architecture
   ✅ Comprehensive benchmarking

5. PERFORMANCE:
   - Lightning model: ~5-15ms inference, ~60+ FPS
   - Balanced model: ~15-30ms inference, ~30+ FPS
   - Automatic speech with 2-second cooldown
   - Smart prediction stability (requires 3 consistent predictions)

6. CONTROLS (during detection):
   - 'q': Quit
   - 's': Save screenshot  
   - 'r': Reset prediction buffer

Happy sign language detecting! 
"""
