import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import nltk
import pickle
import os
import argparse
import sys
from itertools import chain

def get_artist_id(artist_name, access_token):
    url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"q": artist_name}
    
    response = requests.get(url, params=params, headers=headers).json()
    hits = response["response"]["hits"]
    
    for hit in hits:
        if hit["result"]["primary_artist"]["name"].lower() == artist_name.lower():
            return hit["result"]["primary_artist"]["id"]
    return None

def get_artist_songs(artist_name, artist_id, access_token, max_songs=10):
    songs = []
    page = 1
    headers = {"Authorization": f"Bearer {access_token}"}
    
    while len(songs) < max_songs:
        url = f"https://api.genius.com/artists/{artist_id}/songs"
        params = {"per_page": 20, "page": page, "sort": "popularity"}
        response = requests.get(url, params=params, headers=headers).json()
        new_songs = response["response"]["songs"]
        
        if not new_songs:
            break
        
        for song in new_songs:
            if song["primary_artist"]["id"] == artist_id:
                songs.append({
                    "title": song["title"],
                    "url": song["url"],
                    "release_date": song.get("release_date", "N/A"),
                    "artist": artist_name
                })
                if len(songs) >= max_songs:
                    break
        page += 1
        
    return songs

def get_lyrics_from_url(song_url):
    page = requests.get(song_url)
    soup = BeautifulSoup(page.text, 'html.parser')
    lyrics = []

    # Find the div that contains the lyrics
    lyrics_div = soup.find("div", attrs={"data-lyrics-container": "true"})
    
    # Use .decode_contents() to get the raw HTML (including <br> tags)
    lyric_links = lyrics_div.find_all("a", attrs={"data-ignore-on-click-outside": "true"})
    
    for link in lyric_links:
        # Get the raw HTML content
        lyric_html = link.find("span").decode_contents()

        # Replace <br> tags with a whitespace
        lyric_text = re.sub(r"<br\s*/?>", " ", lyric_html)
        
        # Remove bracketed sections (like [Chorus], [Verse 1], etc.)
        lyric_text = re.sub(r"\[.*?\]", "", lyric_text)
        
        # Remove parentheses (if needed)
        lyric_text = re.sub(r"[()]", "", lyric_text)
        
        lyrics.append(lyric_text)

    # Split the lyrics into words, removing punctuation, and keeping apostrophes
    words = [re.sub(r"[^\w']", '', word) for sentence in lyrics for word in sentence.split()]
    words = [word for word in words if word]  # remove empty strings

    return words

def get_kendrick_lyrics(access_token, max_songs=5):
    artist_id = get_artist_id("Kendrick Lamar", access_token)
    if not artist_id:
        return "Artist not found."

    songs = get_artist_songs("Kendrick Lamar", artist_id, access_token, max_songs)
    for song in songs:
        print(f"Fetching lyrics for: {song['title']}")
        lyrics = get_lyrics_from_url(song["url"])
        song["lyrics"] = lyrics
    return songs


def get_artist_lyrics(access_token, artist, max_songs=5):
    artist_id = get_artist_id(artist, access_token)
    if not artist_id:
        return "Artist not found."

    songs = get_artist_songs(artist, artist_id, access_token, max_songs)
    for song in songs:
        print(f"Fetching lyrics for: {song['title']}")
        lyrics = get_lyrics_from_url(song["url"])
        song["lyrics"] = lyrics
    return songs


def get_artist_corpus(df, artist_name):
    artist_lyrics = df[df["Artist"] == artist_name]["Lyrics"].tolist()
    return list(chain.from_iterable(artist_lyrics))

def prepare_sequences(corpus, sequence_length=5):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([" ".join(corpus)])

    total_words = len(tokenizer.word_index) + 1

    # Create input sequences
    input_sequences = []
    for i in range(sequence_length, len(corpus)):
        seq = corpus[i-sequence_length:i+1]
        encoded = tokenizer.texts_to_sequences([" ".join(seq)])[0]
        input_sequences.append(encoded)

    input_sequences = np.array(input_sequences)
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = to_categorical(y, num_classes=total_words)

    return X, y, tokenizer, total_words


def build_model(input_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 64, input_length=input_len))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def generate_lyrics(seed_text, tokenizer, model, max_sequence_len, num_words_to_generate):
    output = seed_text
    current_input = seed_text

    for _ in range(num_words_to_generate):
        encoded = tokenizer.texts_to_sequences([current_input])[0]
        encoded = pad_sequences([encoded], maxlen=max_sequence_len, truncating='pre')
        
        # Predict probabilities for each word
        predicted_probs = model.predict(encoded, verbose=0)[0]
        
        # Get the index of the word with the highest probability
        predicted_index = np.argmax(predicted_probs)
        
        # Find the corresponding word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        
        # Append the predicted word to the output
        output += " " + output_word
        
        # Update the current input sequence by appending the predicted word
        current_input += " " + output_word
        # Ensure the input sequence doesn't exceed max_sequence_len words
        current_input_words = current_input.split()
        if len(current_input_words) > max_sequence_len:
            current_input = " ".join(current_input_words[-max_sequence_len:])
            
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Drake-style lyrics.')
    parser.add_argument('seed_text', type=str, help='The starting text for lyric generation.')
    parser.add_argument('length', type=int, help='The number of words to generate.')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory containing the saved model and tokenizer.')

    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, 'drake_model.keras')
    tokenizer_path = os.path.join(args.model_dir, 'tokenizer.pkl')
    seq_len_path = os.path.join(args.model_dir, 'seq_length.pkl')

    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path) or not os.path.exists(seq_len_path):
        print(f"Error: Model files not found in {args.model_dir}. Please train the model first.", file=sys.stderr)
        sys.exit(1)

    try:
        # Load the model, tokenizer, and sequence length
        model = load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(seq_len_path, 'rb') as f:
            max_sequence_len = pickle.load(f)

        # Generate lyrics
        generated = generate_lyrics(args.seed_text, tokenizer, model, max_sequence_len, args.length)
        print(generated) # Print generated lyrics to stdout

    except Exception as e:
        print(f"Error during generation: {str(e)}", file=sys.stderr)
        sys.exit(1)
