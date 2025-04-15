import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Emoji sentiment mapping
def load_emoji_sentiment_mapping():
    # This is a simplified version - you might want to expand this
    positive_emojis = ['😊', '😄', '👍', '❤️', '🥰', '😍', '🙏', '👏', '🎉', '✨']
    negative_emojis = ['😢', '😭', '👎', '😠', '😡', '🤬', '😔', '😞', '😒', '💔']
    neutral_emojis = ['😐', '🤔', '🤷', '😶', '📝', '🔍', '📊', '🔄', '🕒', '📱']
    
    emoji_sentiment = {}
    for e in positive_emojis:
        emoji_sentiment[e] = 'positive'
    for e in negative_emojis:
        emoji_sentiment[e] = 'negative'
    for e in neutral_emojis:
        emoji_sentiment[e] = 'neutral'
    
    return emoji_sentiment

# Text preprocessing functions
def preprocess_text(text, emoji_sentiment_map, remove_stopwords=True):
    # Convert to lowercase
    text = text.lower()
    
    # Extract emojis and their sentiment
    emoji_sentiments = []
    for char in text:
        if char in emoji.EMOJI_DATA:
            emoji_text = emoji.demojize(char)
            if char in emoji_sentiment_map:
                emoji_sentiments.append(emoji_sentiment_map[char])
            text = text.replace(char, ' ' + emoji_text + ' ')
    
    # Clean text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    return tokens, emoji_sentiments

# Custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, max_length=100, vocab=None):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        if vocab is None:
            # Build vocabulary
            word_counts = {}
            emoji_sentiment_map = load_emoji_sentiment_mapping()
            
            for text in texts:
                tokens, _ = preprocess_text(text, emoji_sentiment_map)
                for token in tokens:
                    if token not in word_counts:
                        word_counts[token] = 0
                    word_counts[token] += 1
            
            # Keep only words that appear more than once
            self.vocab = {'<PAD>': 0, '<UNK>': 1}
            for word, count in word_counts.items():
                if count > 1:
                    self.vocab[word] = len(self.vocab)
        else:
            self.vocab = vocab
        
        # Preprocess texts
        self.preprocessed_texts = []
        self.emoji_sentiments_list = []
        emoji_sentiment_map = load_emoji_sentiment_mapping()
        
        for text in texts:
            tokens, emoji_sentiments = preprocess_text(text, emoji_sentiment_map)
            self.preprocessed_texts.append(tokens)
            self.emoji_sentiments_list.append(emoji_sentiments)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        tokens = self.preprocessed_texts[idx]
        emoji_sentiments = self.emoji_sentiments_list[idx]
        
        # Convert tokens to indices
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices = indices + [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        # Create tensor
        tensor = torch.tensor(indices, dtype=torch.long)
        label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        
        # Create emoji sentiment feature
        emoji_feature = [0, 0, 0]  # [positive, negative, neutral]
        for sentiment in emoji_sentiments:
            if sentiment == 'positive':
                emoji_feature[0] += 1
            elif sentiment == 'negative':
                emoji_feature[1] += 1
            elif sentiment == 'neutral':
                emoji_feature[2] += 1
        
        # Normalize emoji feature
        total = sum(emoji_feature) + 1e-10
        emoji_feature = [count / total for count in emoji_feature]
        emoji_feature = torch.tensor(emoji_feature, dtype=torch.float)
        
        return tensor, emoji_feature, label

# Bi-LSTM Model
class BiLSTMSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 + 3, output_dim)  # +3 for emoji features
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, emoji_feature):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output shape: [batch_size, seq_len, hidden_dim * 2]
        # hidden shape: [n_layers * 2, batch_size, hidden_dim]
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # hidden shape: [batch_size, hidden_dim * 2]
        
        # Concatenate with emoji features
        combined = torch.cat((hidden, emoji_feature), dim=1)
        # combined shape: [batch_size, hidden_dim * 2 + 3]
        
        return self.fc(self.dropout(combined))

# Training function
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for text, emoji_feature, labels in train_loader:
        text = text.to(device)
        emoji_feature = emoji_feature.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(text, emoji_feature)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Evaluation function
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for text, emoji_feature, labels in data_loader:
            text = text.to(device)
            emoji_feature = emoji_feature.to(device)
            labels = labels.to(device)
            
            output = model(text, emoji_feature)
            loss = criterion(output, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return total_loss / len(data_loader), accuracy, report, conf_matrix

# Main function
def main():
    # Load data
    train_df = pd.read_csv('/Users/krutin/Desktop/sentiment/datasets/train_all.csv')
    val_df = pd.read_csv('/Users/krutin/Desktop/sentiment/datasets/val_all.csv')
    test_df = pd.read_csv('/Users/krutin/Desktop/sentiment/datasets/test_all.csv')
    
    # Create datasets
    train_dataset = SentimentDataset(train_df['sentence'].tolist(), train_df['label'].tolist())
    val_dataset = SentimentDataset(val_df['sentence'].tolist(), val_df['label'].tolist(), vocab=train_dataset.vocab)
    test_dataset = SentimentDataset(test_df['sentence'].tolist(), test_df['label'].tolist(), vocab=train_dataset.vocab)
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = len(train_dataset.vocab)
    embedding_dim = 300
    hidden_dim = 256
    output_dim = len(train_dataset.label_encoder.classes_)
    
    model = BiLSTMSentimentModel(vocab_size, embedding_dim, hidden_dim, output_dim)
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    n_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, val_report, val_conf_matrix = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Epoch: {epoch+1}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        print(f'Validation Report:\n{val_report}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            with open('vocab.pkl', 'wb') as f:
                pickle.dump(train_dataset.vocab, f)
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(train_dataset.label_encoder, f)
            print('Model and auxiliary files saved!')
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pt'))
    test_loss, test_accuracy, test_report, test_conf_matrix = evaluate_model(model, test_loader, criterion, device)
    
    print('Test Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Report:\n{test_report}')
    print(f'Confusion Matrix:\n{test_conf_matrix}')

if __name__ == '__main__':
    main()