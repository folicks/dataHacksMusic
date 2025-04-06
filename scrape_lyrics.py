import requests
from bs4 import BeautifulSoup
import re
import argparse
import sys

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

def get_artist_songs(artist_id, access_token, max_songs=10):
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
                    "release_date": song.get("release_date", "N/A")
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
    


# def get_kendrick_lyrics(access_token, max_songs=5):
#     artist_id = get_artist_id("Kendrick Lamar", access_token)
#     if not artist_id:
#         return "Artist not found."

#     songs = get_artist_songs(artist_id, access_token, max_songs)
#     for song in songs:
#         print(f"Fetching lyrics for: {song['title']}")
#         lyrics = get_lyrics_from_url(song["url"])
#         song["lyrics"] = lyrics
#     return songs


# access_token = "NLSQjUEWly7uv5AK1-fwSwXXl0QN9eQSYlJ2DP_0qpCepn1MLaXozB6VSvkqz1qX"
# kendrick_songs = get_kendrick_lyrics(access_token, max_songs=10)

# for song in kendrick_songs:
#     print(f"\n{song['title']} ({song['release_date']})\n")
#     print(song['lyrics'][:500])  # preview first 500 chars
#     print("...\n" + "-"*80)
#     print(song["url"])

def get_artist_lyrics(access_token, artist, max_songs=5):
    artist_id = get_artist_id(artist, access_token)
    if not artist_id:
        return "Artist not found."

    songs = get_artist_songs(artist_id, access_token, max_songs)
    for song in songs:
        print(f"Fetching lyrics for: {song['title']}")
        lyrics = get_lyrics_from_url(song["url"])
        song["lyrics"] = lyrics
    return songs

# pop_culture_artists = ["Taylor Swift", "Billie Eilish", "Rihanna", "Ariana Grande", "The Weekend", "Kendrick Lamar", "Drake", "Lady Gaga", "Justin Bieber"]
def main():
    parser = argparse.ArgumentParser(description='Fetch lyrics for popular artists')
    parser.add_argument('--artist', help='Artist name to fetch lyrics for')
    parser.add_argument('--list', action='store_true', help='List available artists')
    args = parser.parse_args()

    pop_culture_artists = ["Taylor Swift", "Billie Eilish", "Rihanna", "Ariana Grande", 
                          "The Weekend", "Kendrick Lamar", "Drake", "Lady Gaga", "Justin Bieber"]
    
    if args.list:
        print("Available artists:")
        for artist in pop_culture_artists:
            print(f"- {artist}")
        sys.exit(0)
        
    access_token = "NLSQjUEWly7uv5AK1-fwSwXXl0QN9eQSYlJ2DP_0qpCepn1MLaXozB6VSvkqz1qX"
    # all_songs = []
    # for artist in pop_culture_artists:
    #     all_songs.append(get_artist_lyrics(access_token, artist, max_songs=2))

    # for thing in all_songs:
    #     for song in thing:
    if args.artist:
        if args.artist not in pop_culture_artists:
            print(f"Artist '{args.artist}' not in available list. Use --list to see options.")
            sys.exit(1)
            
        songs = get_artist_lyrics(access_token, args.artist, max_songs=5)
        for song in songs:
            print(f"\n{song['title']} ({song['release_date']})\n")
            print(song['lyrics'][:500])  # preview first 500 chars
            print("...\n" + "-"*80)
            print(song["url"])
    else:
        print("Please specify an artist with --artist or see available artists with --list")

if __name__ == "__main__":
    main()
