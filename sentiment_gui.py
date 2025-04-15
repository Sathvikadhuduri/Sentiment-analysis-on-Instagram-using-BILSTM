import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog,messagebox
from glob import glob
from os.path import expanduser
from sqlite3 import connect
import pathlib
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import re
import emoji
import csv
import emoji
from instaloader import ConnectionException, Instaloader, Post

# Path to Firefox Cookies (Update it if needed)
path_to_firefox_cookies = "C:\\Users\\Hello\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles\\gtdm9dv9.default-release\\cookies.sqlite"
# Initialize Instaloader with saved session
loader = Instaloader(max_connection_attempts=1)
loader.context._session.cookies.update(
    connect(path_to_firefox_cookies).execute(
        "SELECT name, value FROM moz_cookies WHERE host='.instagram.com'"
    )
)

try:
    username = loader.test_login()
    if not username:
        raise ConnectionException()
except ConnectionException:
    messagebox.showerror("Error", "Cookie import failed. Are you logged in successfully in Firefox?")
    exit()

loader.context.username = username
loader.save_session_to_file()

instagram = Instaloader(download_pictures=False, download_videos=False,
                        download_video_thumbnails=False, save_metadata=False,
                        max_connection_attempts=0)
instagram.load_session_from_file('majorproject2025')

def scrape_data(url):
    SHORTCODE = str(url[28:39])
    post = Post.from_shortcode(instagram.context, SHORTCODE)
    output_path = pathlib.Path('post_data')
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.joinpath(SHORTCODE + '.csv')
    
    with csv_path.open("w", encoding="utf-8", newline="") as post_file:
        field_names = ["post_shortcode", "commenter_username", "comment_text", "comment_likes"]
        post_writer = csv.DictWriter(post_file, fieldnames=field_names)
        post_writer.writeheader()
        
        for x in post.get_comments():
            post_writer.writerow({
                "post_shortcode": post.shortcode,
                "commenter_username": x.owner.username if hasattr(x.owner, 'username') else str(x.owner),
                "comment_text": (emoji.demojize(x.text)).encode('utf-8', errors='ignore').decode() if x.text else "",
                "comment_likes": x.likes_count
            })
    
    return csv_path

# Function for GUI Button
def start_scraping():
    url = url_entry.get().strip()
    if not url.startswith("https://www.instagram.com/p/"):
        messagebox.showerror("Invalid URL", "Please enter a valid Instagram post URL!")
        return
    
    try:
        csv_path = scrape_data(url)
        messagebox.showinfo("Success", f"Scraping Done!\nCSV saved at:\n{csv_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to scrape: {e}")

# Load trained model and related files
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
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
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

# Function to analyze the CSV file
def analyze_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    
    if not file_path:
        return
    
    df = pd.read_csv(file_path, delimiter=",")

    if 'comment_text' not in df.columns or 'comment_likes' not in df.columns:
        messagebox.showerror("Error", "CSV file must contain 'comment_text' and 'comment_likes' columns.")
        return

    comments = df['comment_text'].dropna().tolist()
    comment_likes = df['comment_likes'].fillna(0).astype(int).tolist()  # Ensure likes are integers

    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    likes_count = {"positive": 0, "negative": 0, "neutral": 0}  # ✅ Store likes for each category
    
    # 🔹 Get the total post likes (Only extract once from the post object)
    try:
        post_shortcode = df["post_shortcode"].iloc[0]
        post = Post.from_shortcode(instagram.context, post_shortcode)
        post_likes = post.likes  # ✅ Get total post likes
    except Exception as e:
        post_likes = "Unavailable"  # Handle errors if post_likes is not found

    for comment, likes in zip(comments, comment_likes):
        sentiment = predict_sentiment(comment)
        sentiments[sentiment] += 1
        likes_count[sentiment] += likes  # ✅ Store total likes for each sentiment

    # ✅ Define Colors
    colors = {"positive": "green", "negative": "red", "neutral": "gray"}

    # Create figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6),gridspec_kw={'wspace': 0.4})

    # 🔹 Pie Chart (Sentiment Distribution)
    labels = ["Positive", "Negative", "Neutral"]
    sizes = [sentiments["positive"], sentiments["negative"], sentiments["neutral"]]
    axs[0].pie(sizes, labels=labels, autopct="%1.1f%%", colors=[colors["positive"], colors["negative"], colors["neutral"]])
    axs[0].set_title("Instagram Comment Sentiment Distribution")

    # 🔹 Bar Graph (Number of Comments per Sentiment)
    axs[1].bar(labels, sizes, color=[colors["positive"], colors["negative"], colors["neutral"]])
    axs[1].set_xlabel("Sentiment Categories")
    axs[1].set_ylabel("Number of Comments")
    axs[1].set_title("Sentiment Analysis - Bar Graph")

    # 🔹 Display Summary Below Graphs
    summary_text = (
    f"Total Likes for Post: {post_likes}\n"
    f"Positive Comments: {sentiments['positive']} (Likes: {likes_count['positive']})\n"
    f"Negative Comments: {sentiments['negative']} (Likes: {likes_count['negative']})\n"
    f"Neutral Comments: {sentiments['neutral']} (Likes: {likes_count['neutral']})"
    )
    # Use `plt.subplots_adjust()` to create space for text
    plt.subplots_adjust(bottom=0.3)
    fig.text(0.5, 0.05, summary_text, ha="center", fontsize=12, color="black", bbox=dict(facecolor='white', alpha=0.8))

# 
    # Adjust layout & show the plot
    plt.tight_layout()
    plt.show()





# GUI Setup
root = tk.Tk()
root.title("Instagram Sentiment Analyzer")
root.geometry("550x300")
root.configure(bg="#2C2F33")  # Dark gray background

# Title Label
title_label = tk.Label(root, text="📊 Instagram Sentiment Analysis", font=("Arial", 16, "bold"), bg="#2C2F33", fg="white")
title_label.pack(pady=15)

# Entry Box for URL
url_frame = tk.Frame(root, bg="#2C2F33")
url_frame.pack(pady=5)

tk.Label(url_frame, text="Enter Post URL:", font=("Arial", 12, "bold"), bg="#2C2F33", fg="white").pack(side="left", padx=5)
url_entry = tk.Entry(url_frame, width=40, font=("Arial", 12), bg="#ffffff", fg="#000000", bd=2, relief="solid")
url_entry.pack(side="right", padx=5)

# Buttons Frame
button_frame = tk.Frame(root, bg="#2C2F33")
button_frame.pack(pady=20)

scrape_button = tk.Button(button_frame, text="🔍 Scrape & Save", command=start_scraping, font=("Arial", 12, "bold"))
scrape_button.grid(row=0, column=0, padx=15)

upload_button = tk.Button(button_frame, text="📂 Upload CSV & Analyze", command=analyze_csv, font=("Arial", 12, "bold"))
upload_button.grid(row=0, column=1, padx=15)

# Footer
footer_label = tk.Label(root, text="🔹 Developed by Your Name 🔹", font=("Arial", 10, "italic"), bg="#2C2F33", fg="#AAB7B8")
footer_label.pack(side="bottom", pady=10)

root.mainloop()