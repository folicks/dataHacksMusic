import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np

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

from itertools import chain

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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def build_model(input_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 64, input_length=input_len))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import numpy as np

nltk.download('punkt')

def generate_lyrics(seed_text, tokenizer, model, length=20):
    result = seed_text.split()
    for _ in range(length):
        encoded = tokenizer.texts_to_sequences([" ".join(result[-5:])])[0]
        encoded = pad_sequences([encoded], maxlen=5, truncating='pre')
        predicted = model.predict(encoded, verbose=0)
        next_word = tokenizer.index_word[np.argmax(predicted)]
        result.append(next_word)
    return " ".join(result)

def main():
    # Define artists to analyze
    pop_culture_artists = ["Drake", "Kendrick Lamar"]  
    
    access_token = "NLSQjUEWly7uv5AK1-fwSwXXl0QN9eQSYlJ2DP_0qpCepn1MLaXozB6VSvkqz1qX"
    all_songs = []
    for artist in pop_culture_artists:
        songs = get_artist_lyrics(access_token, artist, max_songs=5)  
        if isinstance(songs, list):
            all_songs.append(songs)
    
    song_list = []
    for artist_songs in all_songs:
        for song in artist_songs:
            song_list.append([song["title"], song["artist"], song["lyrics"]])
    
    song_info = pd.DataFrame(song_list, columns=["Song", "Artist", "Lyrics"])
    
    # Generate Drake lyrics
    corpus = get_artist_corpus(song_info, "Drake")
    X, y, tokenizer, total_words = prepare_sequences(corpus)
    model = build_model(X.shape[1], total_words)
    model.fit(X, y, epochs=100, verbose=1)  
    
    # Generate sample lyrics
    generated = generate_lyrics("start", tokenizer, model, length=20)
    print("\nGenerated Lyrics:")
    print(generated)

if __name__ == "__main__":
    main()
