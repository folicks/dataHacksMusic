import os
import numpy as np
import pickle
from drake_song_gen import get_artist_lyrics, prepare_sequences, build_model, get_artist_corpus
import pandas as pd

def main():
    print("Training Drake lyrics model...")
    
    # Define artists to analyze
    pop_culture_artists = ["Drake"]
    
    access_token = "NLSQjUEWly7uv5AK1-fwSwXXl0QN9eQSYlJ2DP_0qpCepn1MLaXozB6VSvkqz1qX"
    all_songs = []
    
    # Get Drake lyrics
    songs = get_artist_lyrics(access_token, "Drake", max_songs=100)
    if isinstance(songs, list):
        all_songs.append(songs)
    
    song_list = []
    for artist_songs in all_songs:
        for song in artist_songs:
            song_list.append([song["title"], song["artist"], song["lyrics"]])
    
    song_info = pd.DataFrame(song_list, columns=["Song", "Artist", "Lyrics"])
    
    # Generate Drake lyrics model
    corpus = get_artist_corpus(song_info, "Drake")
    X, y, tokenizer, total_words = prepare_sequences(corpus)
    model = build_model(X.shape[1], total_words)
    
    print(f"Training model with {len(X)} sequences for 100 epochs...")
    model.fit(X, y, epochs=100, verbose=1)
    
    # Save the model and tokenizer
    os.makedirs('models', exist_ok=True)
    model.save('models/drake_model.keras')
    
    # Save tokenizer
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save sequence length
    with open('models/seq_length.pkl', 'wb') as f:
        pickle.dump(X.shape[1], f)
    
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
