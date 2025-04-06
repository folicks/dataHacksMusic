from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import numpy as np
import subprocess
import json
import random
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__, static_folder='static', template_folder='static')


# TODO : i should've known where to put this limiter :(
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Sample Drake lyrics for fallback
DRAKE_LYRICS = [
    "Started from the bottom now we're here",
    "I'm just saying you could do better",
    "Running through the six with my woes",
    "Last name ever, first name greatest",
    "I only love my bed and my momma, I'm sorry",
    "God's plan, God's plan",
    "Know yourself, know your worth",
    "They'll tell the story, wow",
    "Look what you've done",
    "I'm upset, fifty thousand on my head, it's disrespect",
    "I'm too good to you, I'm way too good to you",
    "Hotline bling, that can only mean one thing",
    "Jumpman, Jumpman, Jumpman, them boys up to something",
    "I got enemies, got a lot of enemies",
    "You know how that should go",
    "I was running through the six with my woes",
    "She say do you love me, I tell her only partly",
    "Kiki, do you love me? Are you riding?",
    "Nice for what, to these...",
    "I'm working on dying"
]

def generate_lyrics_subprocess(seed_text, length=20):
    """Generate lyrics by calling drake_song_gen.py as a subprocess"""
    try:
        print(f"Running: python drake_song_gen.py \"{seed_text}\" {length}") # Debug print
        result = subprocess.run(
            ['python', 'drake_song_gen.py', seed_text, str(length)], # Use drake_song_gen.py directly
            capture_output=True,
            text=True
        )
        
        print(f"Subprocess stdout: {result.stdout}") # Debug print
        print(f"Subprocess stderr: {result.stderr}") # Debug print
        
        # Get the output
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Subprocess error (stderr): {result.stderr}") # Improved error message
            return None
    except Exception as e:
        print(f"Exception calling subprocess: {str(e)}") # Improved error message
        import traceback
        traceback.print_exc() # Print full traceback
        return None

def generate_lyrics_fallback(seed_text, length=20):
    """Generate lyrics using the fallback method"""
    words = seed_text.split()
    
    # Add random Drake lyrics to reach desired length
    while len(words) < length:
        # Pick a random Drake lyric
        lyric = DRAKE_LYRICS[np.random.randint(0, len(DRAKE_LYRICS))]
        words.extend(lyric.split())
    
    # Trim to exact length
    return " ".join(words[:length])

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/generate', methods=['POST'])
@limiter.limit("10 per minute")
def generate():
    print("Received generate request")
    data = request.json
    seed_text = data.get('seed', 'start')
    length = min(int(data.get('length', 20)), 100)  # Limit to 100 words max
    
    print(f"Generating lyrics with seed: '{seed_text}', length: {length}")
    
    try:
        # Try to use the subprocess method first
        lyrics = generate_lyrics_subprocess(seed_text, length)
        
        if lyrics:
            print(f"Generated lyrics using subprocess: {lyrics[:50]}...")
        else:
            # Use fallback if subprocess fails
            lyrics = generate_lyrics_fallback(seed_text, length)
            print(f"Generated lyrics using fallback (subprocess failed or returned empty): {lyrics[:50]}...") # Clarified fallback reason
        
        return jsonify({'lyrics': lyrics})
    except Exception as e:
        print(f"Error generating lyrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    # Use render_template to process Jinja syntax like url_for
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
