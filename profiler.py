import json
import time
import spotipy
import requests
import random
from collections import Counter
from spotipy.oauth2 import SpotifyOAuth
from config import LASTFM_API_KEY, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI

# --- GENRE MAPPING (for creating the genre tree) ---
GENRE_MAP = {
    "Hip-Hop/Rap": ["hip-hop", "hip hop", "rap", "trap", "boom bap", "gangsta rap", "horrorcore", "emo rap", "cloud rap", "phonk", "underground hip-hop", "g-funk", "rapcore"],
    "Rock/Alternative": ["rock", "alternative", "alternative rock", "indie rock", "emo", "pop punk", "punk", "90s", "post-punk", "new wave", "nu metal", "garage rock", "post-punk revival"],
    "Indie/Folk": ["indie", "indie pop", "indie folk", "folk", "singer-songwriter", "americana", "country", "dream pop", "shoegaze"],
    "Electronic": ["electronic", "electropop", "house", "dubstep", "trip-hop", "downtempo", "synthpop", "dance", "electronica", "idm", "uk garage"],
    "Metal/Hardcore": ["metal", "metalcore", "post-hardcore", "death metal", "hardcore", "screamo", "deathcore", "djent", "nu-metal"],
    "Pop": ["pop", "pop rock", "power pop"],
    "R&B/Soul": ["rnb", "soul", "funk"]
}

# --- Spotify & Last.fm Helper Functions ---

def get_spotify_data(sp_client):
    """Gathers all necessary data from Spotify."""
    print("Fetching data from Spotify...")
    top_artists_results = sp_client.current_user_top_artists(limit=50, time_range='long_term')
    foundational_artists = [artist['name'] for artist in top_artists_results['items']]
    
    all_liked_tracks = []
    liked_results = sp_client.current_user_saved_tracks(limit=50)
    all_liked_tracks.extend(liked_results['items'])
    while liked_results['next']:
        liked_results = sp_client.next(liked_results)
        all_liked_tracks.extend(liked_results['items'])
        if len(all_liked_tracks) % 200 == 0:
            print(f"  - Fetched {len(all_liked_tracks)} liked songs...")
    
    liked_artists_for_exclusion = {artist['name'] for item in all_liked_tracks for artist in item['track']['artists']}
    
    return {"foundational_artists": foundational_artists, "artist_exclusion_list": list(liked_artists_for_exclusion)}

def get_artist_tags(artist_name):
    """Fetches top tags for a single artist from Last.fm."""
    params = {'method': 'artist.gettoptags', 'artist': artist_name, 'api_key': LASTFM_API_KEY, 'format': 'json'}
    response = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
    if response.status_code != 200: return []
    data = response.json()
    return [tag['name'] for tag in data.get('toptags', {}).get('tag', [])]

def main():
    """Builds and saves a detailed taste profile."""
    scope = "user-top-read user-library-read"
    auth_manager = SpotifyOAuth(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET, redirect_uri=SPOTIFY_REDIRECT_URI, scope=scope)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    spotify_data = get_spotify_data(sp)
    
    print("\nBuilding genre tree from Last.fm tags...")
    tag_counts = Counter()
    for artist in spotify_data["foundational_artists"]:
        tag_counts.update(get_artist_tags(artist))
        time.sleep(0.1)

    genre_tree = {genre: {"total_weight": 0, "subgenres": {}} for genre in GENRE_MAP}
    for tag, weight in tag_counts.items():
        categorized = False
        for genre, keywords in GENRE_MAP.items():
            if any(keyword in tag.lower() for keyword in keywords):
                genre_tree[genre]["subgenres"][tag] = weight
                genre_tree[genre]["total_weight"] += weight
                categorized = True
                break
    
    final_profile = {
        "seed_artists": spotify_data["foundational_artists"],
        "artist_exclusion_list": spotify_data["artist_exclusion_list"],
        "genre_tree": genre_tree
    }

    with open('taste_profile.json', 'w', encoding='utf-8') as f:
        json.dump(final_profile, f, indent=2, ensure_ascii=False)

    print("\n✅ Taste profile built and saved to 'taste_profile.json'")

if __name__ == "__main__":
    main()
