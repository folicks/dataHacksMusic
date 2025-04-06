import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys
from collections import Counter
import os
import json
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def load_large_dataset(data_dir):
    """Load a large dataset of songs from a directory containing JSON files"""
    all_data = []
    
    # Walk through the directory
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    return pd.DataFrame(all_data)

def preprocess_lyrics(text):
    """Enhanced preprocessing of lyrics text"""
    if not isinstance(text, str):
        return []
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep apostrophes
    text = re.sub(r'[^a-zA-Z\s\']', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

def create_vocabulary(texts, min_freq=5):
    """Create vocabulary from texts with minimum frequency threshold"""
    all_words = []
    for text in tqdm(texts, desc="Creating vocabulary"):
        all_words.extend(preprocess_lyrics(text))
    
    word_counts = Counter(all_words)
    # Filter words that appear less than min_freq times
    vocabulary = {word: idx+1 for idx, (word, count) in enumerate(word_counts.most_common()) if count >= min_freq}
    vocabulary['<unk>'] = 0
    vocabulary['<pad>'] = len(vocabulary)
    
    return vocabulary

class LyricsDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_length=200):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize and convert to indices
        tokens = preprocess_lyrics(text)
        indices = [self.word_to_idx.get(word, self.word_to_idx['<unk>']) for word in tokens[:self.max_length]]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices += [self.word_to_idx['<pad>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
            
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class RapClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=200, hidden_dim=256, num_classes=2):
        super(RapClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        x = torch.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in train_pbar:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(classification_report(all_labels, all_preds))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def predict_rap(lyrics, model, word_to_idx, device):
    """Predict if given lyrics are rap or not"""
    model.eval()
    
    # Preprocess lyrics
    tokens = preprocess_lyrics(lyrics)
    indices = [word_to_idx.get(word, 0) for word in tokens]
    
    # Pad or truncate
    if len(indices) < 100:
        indices += [0] * (100 - len(indices))
    else:
        indices = indices[:100]
    
    # Convert to tensor
    text_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(text_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence

def main():
    if len(sys.argv) > 1:
        # If lyrics are provided as command line argument
        lyrics = ' '.join(sys.argv[1:])
        
        # Load model and vocabulary
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        word_to_idx = torch.load('vocabulary.pth')
        model = RapClassifier(len(word_to_idx))
        model.load_state_dict(torch.load('best_model.pth'))
        model.to(device)
        
        prediction, confidence = predict_rap(lyrics, model, word_to_idx, device)
        print(f"\nLyrics: {lyrics}")
        print(f"Prediction: {'Rap' if prediction == 1 else 'Not Rap'}")
        print(f"Confidence: {confidence:.2%}")
    else:
        # Load large dataset
        data_dir = 'lyrics_dataset'  # Directory containing JSON files with lyrics
        print("Loading dataset...")
        df = load_large_dataset(data_dir)
        
        # Ensure we have the required columns
        if 'lyrics' not in df.columns or 'is_rap' not in df.columns:
            raise ValueError("Dataset must contain 'lyrics' and 'is_rap' columns")
        
        print(f"Loaded {len(df)} songs")
        
        # Create vocabulary
        print("Creating vocabulary...")
        word_to_idx = create_vocabulary(df['lyrics'])
        torch.save(word_to_idx, 'vocabulary.pth')
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Create datasets and dataloaders
        train_dataset = LyricsDataset(train_df['lyrics'].values, train_df['is_rap'].values, word_to_idx)
        val_dataset = LyricsDataset(val_df['lyrics'].values, val_df['is_rap'].values, word_to_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RapClassifier(len(word_to_idx)).to(device)
        
        # Train model
        train_model(model, train_loader, val_loader, device, num_epochs=20)
        
        print("\nTo test the classifier, run:")
        print("python rap_classifier.py \"your lyrics here\"")

if __name__ == "__main__":
    main() 
