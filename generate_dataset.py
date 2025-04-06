import json
import random
from faker import Faker
import os

# Initialize Faker for generating realistic text
fake = Faker()

# Common rap phrases and patterns
rap_patterns = [
    "I'm {adjective}, {adjective}, {adjective}",
    "Spitting {noun}, dropping {noun}, making {noun}",
    "From the {place} to the {place}, we {verb}",
    "Stacking {noun}, counting {noun}, making {noun}",
    "I'm the {noun} of the {noun}, {verb} like a {noun}",
    "Breaking {noun}, making {noun}, taking {noun}",
    "I'm {adjective} with the {noun}, {verb} like a {noun}",
    "From the {place} to the {place}, we {verb} the {noun}",
    "Making {noun}, taking {noun}, breaking {noun}",
    "I'm the {noun} of the {noun}, {verb} like a {noun}"
]

# Common non-rap phrases and patterns
non_rap_patterns = [
    "I'm {adjective} in the {place}, feeling {adjective}",
    "The {noun} is {adjective}, the {noun} is {adjective}",
    "I {verb} in the {place}, feeling {adjective}",
    "The {noun} is {adjective}, the {noun} is {adjective}",
    "I'm {adjective} in the {place}, feeling {adjective}",
    "The {noun} is {adjective}, the {noun} is {adjective}",
    "I {verb} in the {place}, feeling {adjective}",
    "The {noun} is {adjective}, the {noun} is {adjective}",
    "I'm {adjective} in the {place}, feeling {adjective}",
    "The {noun} is {adjective}, the {noun} is {adjective}"
]

# Vocabulary for different parts of speech
vocabulary = {
    'adjective': ['happy', 'sad', 'blue', 'green', 'red', 'yellow', 'bright', 'dark', 'light', 'heavy',
                 'fast', 'slow', 'quick', 'loud', 'quiet', 'strong', 'weak', 'tall', 'short', 'long'],
    'noun': ['sun', 'moon', 'star', 'sky', 'cloud', 'rain', 'wind', 'tree', 'flower', 'grass',
             'bird', 'dog', 'cat', 'fish', 'house', 'car', 'road', 'street', 'city', 'town'],
    'verb': ['run', 'walk', 'jump', 'dance', 'sing', 'play', 'work', 'sleep', 'eat', 'drink',
             'think', 'feel', 'see', 'hear', 'smell', 'taste', 'touch', 'move', 'stop', 'go'],
    'place': ['park', 'beach', 'forest', 'mountain', 'river', 'lake', 'ocean', 'desert', 'city', 'town',
              'house', 'school', 'store', 'restaurant', 'hotel', 'hospital', 'airport', 'station', 'garden', 'field']
}

def generate_rap_lyrics():
    """Generate rap-like lyrics using patterns and vocabulary"""
    verses = []
    for _ in range(random.randint(2, 4)):  # Generate 2-4 verses
        pattern = random.choice(rap_patterns)
        verse = pattern.format(
            adjective=random.choice(vocabulary['adjective']),
            noun=random.choice(vocabulary['noun']),
            verb=random.choice(vocabulary['verb']),
            place=random.choice(vocabulary['place'])
        )
        verses.append(verse)
    return '\n'.join(verses)

def generate_non_rap_lyrics():
    """Generate non-rap lyrics using patterns and vocabulary"""
    verses = []
    for _ in range(random.randint(2, 4)):  # Generate 2-4 verses
        pattern = random.choice(non_rap_patterns)
        verse = pattern.format(
            adjective=random.choice(vocabulary['adjective']),
            noun=random.choice(vocabulary['noun']),
            verb=random.choice(vocabulary['verb']),
            place=random.choice(vocabulary['place'])
        )
        verses.append(verse)
    return '\n'.join(verses)

def generate_dataset(num_songs=1000):
    """Generate a dataset of songs with a mix of rap and non-rap"""
    dataset = []
    rap_count = 0
    non_rap_count = 0
    
    for _ in range(num_songs):
        # Generate approximately 50% rap and 50% non-rap
        is_rap = random.random() < 0.5
        
        if is_rap and rap_count < num_songs // 2:
            lyrics = generate_rap_lyrics()
            rap_count += 1
        elif not is_rap and non_rap_count < num_songs // 2:
            lyrics = generate_non_rap_lyrics()
            non_rap_count += 1
        else:
            # If we've hit the limit for one category, generate the other
            if rap_count >= num_songs // 2:
                lyrics = generate_non_rap_lyrics()
                non_rap_count += 1
            else:
                lyrics = generate_rap_lyrics()
                rap_count += 1
        
        dataset.append({
            "lyrics": lyrics,
            "is_rap": 1 if is_rap else 0
        })
    
    return dataset

def main():
    # Create the dataset
    print("Generating dataset...")
    dataset = generate_dataset(1000)
    
    # Save to file
    output_file = "lyrics_dataset/large_dataset.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} songs")
    print(f"Rap songs: {sum(1 for song in dataset if song['is_rap'] == 1)}")
    print(f"Non-rap songs: {sum(1 for song in dataset if song['is_rap'] == 0)}")
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    main() 