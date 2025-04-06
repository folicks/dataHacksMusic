from flask import Flask, request, jsonify, send_from_directory
from drake_song_gen import get_artist_lyrics, prepare_sequences, build_model, generate_lyrics, get_artist_corpus
import pandas as pd
from itertools import chain
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__, static_folder='static')

# Initialize model and tokenizer as global variables
model = None
tokenizer = None
total_words = None

def initialize_model():
    global model, tokenizer, total_words
    
    print("Initializing model...")
    
    # Create a simple test corpus for quick initialization
    test_corpus = ["this", "is", "a", "test", "corpus", "for", "drake", "lyrics", 
                   "generator", "to", "test", "the", "model", "initialization"]
    
    # Prepare sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([" ".join(test_corpus)])
    total_words = len(tokenizer.word_index) + 1
    
    # Create sample sequences
    input_sequences = []
    for i in range(5, len(test_corpus)):
        seq = test_corpus[i-5:i+1]
        encoded = tokenizer.texts_to_sequences([" ".join(seq)])[0]
        input_sequences.append(encoded)
    
    input_sequences = np.array(input_sequences)
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = to_categorical(y, num_classes=total_words)
    
    # Build and compile model
    model = build_model(X.shape[1], total_words)
    model.fit(X, y, epochs=5, verbose=1)
    
    print("Model initialized successfully!")

# Initialize model on startup
print("Starting server...")
initialize_model()

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/generate', methods=['POST'])
def generate():
    global model, tokenizer, total_words
    
    print("Received generate request")
    data = request.json
    seed_text = data.get('seed', 'start')
    length = min(int(data.get('length', 20)), 100)
    
    print(f"Generating lyrics with seed: '{seed_text}', length: {length}")
    
    if not model or not tokenizer:
        print("Error: Model not initialized")
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        # Generate lyrics
        result = seed_text.split()
        for _ in range(length):
            encoded = tokenizer.texts_to_sequences([" ".join(result[-5:])])[0]
            encoded = pad_sequences([encoded], maxlen=5, truncating='pre')
            predicted = model.predict(encoded, verbose=0)
            next_word = tokenizer.index_word[np.argmax(predicted)]
            result.append(next_word)
        
        lyrics = " ".join(result)
        print(f"Generated lyrics: {lyrics[:50]}...")
        return jsonify({'lyrics': lyrics})
    except Exception as e:
        print(f"Error generating lyrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
