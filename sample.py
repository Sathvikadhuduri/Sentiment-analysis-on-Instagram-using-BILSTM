import torch
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
import emoji
import torch.nn as nn
import torch.nn.functional as F

# Load necessary NLTK resources
nltk.download('punkt')

# Load the trained model
class BiLSTMSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 + 3, output_dim)  # +3 for emoji features
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, emoji_feature):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # Combine forward & backward states
        combined = torch.cat((hidden, emoji_feature), dim=1)
        return self.fc(self.dropout(combined))

# Load model, vocab, and label encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model = BiLSTMSentimentModel(len(vocab), 300, 256, len(label_encoder.classes_))
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)
model.eval()

# Emoji sentiment mapping
def load_emoji_sentiment_mapping():
    return {
        '😊': 'positive', '😄': 'positive', '👍': 'positive', '❤️': 'positive', '🥰': 'positive', '😍': 'positive',
        '🙏': 'positive', '👏': 'positive', '🎉': 'positive', '✨': 'positive',
        '😢': 'negative', '😭': 'negative', '👎': 'negative', '😠': 'negative', '😡': 'negative', '🤬': 'negative',
        '😔': 'negative', '😞': 'negative', '😒': 'negative', '💔': 'negative',
        '😐': 'neutral', '🤔': 'neutral', '🤷': 'neutral', '😶': 'neutral', '📝': 'neutral', '🔍': 'neutral',
        '📊': 'neutral', '🔄': 'neutral', '🕒': 'neutral', '📱': 'neutral'
    }

# Preprocess text
def preprocess_text(text):
    emoji_sentiment_map = load_emoji_sentiment_mapping()
    text = text.lower()

    # Extract emoji sentiments
    emoji_sentiments = []
    for char in text:
        if char in emoji.EMOJI_DATA:
            text = text.replace(char, ' ' + emoji.demojize(char) + ' ')
            if char in emoji_sentiment_map:
                emoji_sentiments.append(emoji_sentiment_map[char])

    # Clean text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Convert tokens to indices
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]

    # Pad/truncate
    max_length = 100
    if len(indices) < max_length:
        indices += [vocab['<PAD>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]

    # Convert emoji sentiment to tensor
    emoji_feature = [0, 0, 0]  # [positive, negative, neutral]
    for sentiment in emoji_sentiments:
        if sentiment == 'positive':
            emoji_feature[0] += 1
        elif sentiment == 'negative':
            emoji_feature[1] += 1
        elif sentiment == 'neutral':
            emoji_feature[2] += 1

    total = sum(emoji_feature) + 1e-10
    emoji_feature = [count / total for count in emoji_feature]
    
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device), torch.tensor(emoji_feature, dtype=torch.float).unsqueeze(0).to(device)

# Predict function
def predict_sentiment(sentence):
    text_tensor, emoji_tensor = preprocess_text(sentence)
    
    with torch.no_grad():
        output = model(text_tensor, emoji_tensor)
        prediction = torch.argmax(F.softmax(output, dim=1), dim=1).item()
    
    return label_encoder.inverse_transform([prediction])[0]

# Run inference
if __name__ == "__main__":
    while True:
        sentence = input("Enter a sentence (or type 'exit' to quit): ")
        if sentence.lower() == 'exit':
            break
        sentiment = predict_sentiment(sentence)
        print(f"Predicted Sentiment: {sentiment}\n")