from flask import Flask, request, jsonify
from drake_song_gen import get_artist_lyrics, generate_lyrics
import os

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    seed_text = data.get('seed', 'start')
    length = data.get('length', 20)
    
    # You'll need to initialize your model here
    # For now using placeholder
    lyrics = generate_lyrics(seed_text, None, None, length)
    
    return jsonify({'lyrics': lyrics})

@app.route('/')
def home():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
