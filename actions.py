import json
import time
import requests
import spotipy
import random
from config import LASTFM_API_KEY

# --- GENRE MAP (must match profiler) ---
GENRE_MAP = {
    "Hip-Hop/Rap": ["hip-hop", "hip hop", "rap", "trap", "boom bap", "gangsta rap", "horrorcore", "emo rap", "cloud rap", "phonk", "underground hip-hop", "g-funk", "rapcore"],
    "Rock/Alternative": ["rock", "alternative", "alternative rock", "indie rock", "emo", "pop punk", "punk", "90s", "post-punk", "new wave", "nu metal", "garage rock", "post-punk revival"],
    "Indie/Folk": ["indie", "indie pop", "indie folk", "folk", "singer-songwriter", "americana", "country", "dream pop", "shoegaze"],
    "Electronic": ["electronic", "electropop", "house", "dubstep", "trip-hop", "downtempo", "synthpop", "dance", "electronica", "idm", "uk garage"],
    "Metal/Hardcore": ["metal", "metalcore", "post-hardcore", "death metal", "hardcore", "screamo", "deathcore", "djent", "nu-metal"],
    "Pop": ["pop", "pop rock", "power pop"],
    "R&B/Soul": ["rnb", "soul", "funk"]
}

# --- Helper Functions ---
def get_artist_tags(artist_name):
    params = {'method': 'artist.gettoptags', 'artist': artist_name, 'api_key': LASTFM_API_KEY, 'format': 'json'}
    response = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
    if response.status_code != 200: return []
    return [tag['name'] for tag in data.get('toptags', {}).get('tag', [])] if (data := response.json()) else []

def map_tags_to_genres(tags):
    artist_genres = set()
    for tag in tags:
        for genre, keywords in GENRE_MAP.items():
            if any(keyword in tag.lower() for keyword in keywords):
                artist_genres.add(genre)
    return artist_genres

def get_similar_artists(artist_name, limit=10):
    params = {'method': 'artist.getsimilar', 'artist': artist_name, 'api_key': LASTFM_API_KEY, 'format': 'json', 'limit': limit}
    response = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
    if response.status_code != 200: return []
    return [artist['name'] for artist in data.get('similarartists', {}).get('artist', [])] if (data := response.json()) else []

def get_artist_info(artist_name):
    params = {'method': 'artist.getinfo', 'artist': artist_name, 'api_key': LASTFM_API_KEY, 'format': 'json'}
    response = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
    return response.json().get('artist') if response.status_code == 200 else None

def get_artist_top_track(artist_name):
    params = {'method': 'artist.gettoptracks', 'artist': artist_name, 'api_key': LASTFM_API_KEY, 'format': 'json', 'limit': 1}
    response = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
    if response.status_code != 200: return None
    tracks = data.get('toptracks', {}).get('track', []) if (data := response.json()) else []
    return f"{tracks[0]['name']} by {tracks[0]['artist']['name']}" if tracks else None

# --- Core Functions ---
def generate_recommendations(popularity_cap, selected_genres, similar_artist_limit, seed_artist_count):
    try:
        with open('taste_profile.json', 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
    except FileNotFoundError:
        return ["Error: 'taste_profile.json' not found. Please run profiler.py first."]

    seed_artists = profile_data.get("seed_artists", [])
    exclusion_list = set(profile_data.get("artist_exclusion_list", []))

    bridge_artists = []
    for artist_name in seed_artists:
        tags = get_artist_tags(artist_name)
        genres = map_tags_to_genres(tags)
        if len(genres) > 1:
            if not selected_genres or any(g in genres for g in selected_genres):
                bridge_artists.append(artist_name)
        time.sleep(0.1)
    
    bridge_artist_seeds = random.sample(bridge_artists, min(len(bridge_artists), seed_artist_count))
    
    final_recs = set()
    for artist_name in bridge_artist_seeds:
        similar_artists_pool = get_similar_artists(artist_name, limit=similar_artist_limit)
        time.sleep(0.1)
        
        explore_count = max(1, similar_artist_limit // 2)
        artists_to_explore = random.sample(similar_artists_pool, min(len(similar_artists_pool), explore_count))

        for sim_artist in artists_to_explore:
            if sim_artist in exclusion_list: continue
            info = get_artist_info(sim_artist)
            time.sleep(0.1)
            if info and 0 < int(info.get('stats', {}).get('listeners', 0)) < popularity_cap:
                top_track = get_artist_top_track(sim_artist)
                if top_track:
                    final_recs.add(top_track)
                time.sleep(0.1)
    
    return sorted(list(final_recs))

def create_spotify_playlist(sp, songs, playlist_name):
    """Creates a Spotify playlist using a pre-authenticated Spotipy client."""
    if not sp:
        return "Error: Not authenticated with Spotify. Please log in."

    user_id = sp.current_user()['id']
    track_uris = []
    for song in songs:
        try:
            track_name, artist_name = song.rsplit(' by ', 1)
            results = sp.search(q=f"track:{track_name} artist:{artist_name}", type='track', limit=1)
            if items := results['tracks']['items']:
                track_uris.append(items[0]['uri'])
        except ValueError:
            pass
            
    if not track_uris: return "Could not find any of the songs on Spotify."
    
    playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True)
    
    for i in range(0, len(track_uris), 100):
        sp.playlist_add_items(playlist['id'], track_uris[i:i+100])
        
    return f"Success! Listen here: {playlist['external_urls']['spotify']}"

