import streamlit as st
import spotipy
import requests
import json
import random
import time
from spotipy.oauth2 import SpotifyOAuth
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI, LASTFM_API_KEY
from collections import Counter

# --- App State Management ---
if 'sp' not in st.session_state:
    st.session_state.sp = None
if 'taste_profile' not in st.session_state:
    st.session_state.taste_profile = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

# --- GENRE MAP ---
GENRE_MAP = {
    "Hip-Hop/Rap": ["hip-hop", "hip hop", "rap", "trap", "boom bap", "gangsta rap", "horrorcore", "emo rap", "cloud rap", "phonk", "underground hip-hop", "g-funk", "rapcore"],
    "Rock/Alternative": ["rock", "alternative", "alternative rock", "indie rock", "emo", "pop punk", "punk", "90s", "post-punk", "new wave", "nu metal", "garage rock", "post-punk revival"],
    "Indie/Folk": ["indie", "indie pop", "indie folk", "folk", "singer-songwriter", "americana", "country", "dream pop", "shoegaze"],
    "Electronic": ["electronic", "electropop", "house", "dubstep", "trip-hop", "downtempo", "synthpop", "dance", "electronica", "idm", "uk garage"],
    "Metal/Hardcore": ["metal", "metalcore", "post-hardcore", "death metal", "hardcore", "screamo", "deathcore", "djent", "nu-metal"],
    "Pop": ["pop", "pop rock", "power pop"],
    "R&B/Soul": ["rnb", "soul", "funk"]
}

# --- Core Logic Functions ---

def build_user_profile(sp):
    """Generates a new taste profile for the currently logged-in user."""
    yield "status", "Fetching your Spotify data..."
    yield "progress", 0.1
    top_artists_results = sp.current_user_top_artists(limit=50, time_range='long_term')
    seed_artists = [artist['name'] for artist in top_artists_results['items']]
    
    yield "status", "Building exclusion list from your liked songs..."
    yield "progress", 0.2
    all_liked_tracks = []
    liked_results = sp.current_user_saved_tracks(limit=50)
    all_liked_tracks.extend(liked_results['items'])
    while liked_results['next']:
        liked_results = sp.next(liked_results)
        all_liked_tracks.extend(liked_results['items'])
    exclusion_list = {artist['name'] for item in all_liked_tracks for artist in item['track']['artists']}

    yield "status", "Analyzing your taste with Last.fm tags..."
    yield "progress", 0.4
    tag_counts = Counter()
    for i, artist_name in enumerate(seed_artists):
        params = {'method': 'artist.gettoptags', 'artist': artist_name, 'api_key': LASTFM_API_KEY, 'format': 'json'}
        response = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
        if response.status_code == 200:
            data = response.json()
            tags = [tag['name'] for tag in data.get('toptags', {}).get('tag', [])]
            tag_counts.update(tags)
        yield "progress", 0.4 + (0.6 * ((i + 1) / len(seed_artists)))
        time.sleep(0.1)

    genre_tree = {genre: {"total_weight": 0, "subgenres": {}} for genre in GENRE_MAP}
    for tag, weight in tag_counts.items():
        for genre, keywords in GENRE_MAP.items():
            if any(keyword in tag.lower() for keyword in keywords):
                genre_tree[genre]["subgenres"][tag] = weight
                genre_tree[genre]["total_weight"] += weight
                break
    
    yield "status", "Profile created!"
    yield "progress", 1.0
    
    st.session_state.taste_profile = {
        "seed_artists": seed_artists,
        "artist_exclusion_list": list(exclusion_list),
        "genre_tree": genre_tree
    }

def generate_recommendations(profile, popularity_cap, subgenre_count, artists_per_subgenre):
    """Generates recommendations using the new Tag-Based Random Discovery method."""
    exclusion_list = set(profile.get("artist_exclusion_list", []))
    genre_tree = profile.get("genre_tree", {})

    # Create a flat list of all sub-genre tags from the user's profile
    all_subgenres = [tag for genre_details in genre_tree.values() for tag in genre_details.get("subgenres", {}).keys()]
    
    if not all_subgenres:
        return ["Could not find any sub-genres in your profile to search."]

    # Randomly select a number of sub-genres to explore
    tags_to_explore = random.sample(all_subgenres, min(len(all_subgenres), subgenre_count))
    
    final_recs = set()
    for tag in tags_to_explore:
        # Get a list of artists for the randomly selected tag
        params = {'method': 'tag.gettopartists', 'tag': tag, 'api_key': LASTFM_API_KEY, 'format': 'json', 'limit': 100}
        response = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
        time.sleep(0.1)
        
        if response.status_code == 200:
            data = response.json()
            artist_pool = [artist['name'] for artist in data.get('topartists', {}).get('artist', [])]
            
            # Randomly sample artists from this niche pool
            artists_to_check = random.sample(artist_pool, min(len(artist_pool), artists_per_subgenre))

            for artist_name in artists_to_check:
                if artist_name in exclusion_list:
                    continue

                info_response = requests.get(f"http://ws.audioscrobbler.com/2.0/?method=artist.getinfo&artist={artist_name}&api_key={LASTFM_API_KEY}&format=json")
                time.sleep(0.1)
                if info_response.status_code == 200:
                    info = info_response.json().get('artist')
                    if info and 0 < int(info.get('stats', {}).get('listeners', 0)) < popularity_cap:
                        top_track_response = requests.get(f"http://ws.audioscrobbler.com/2.0/?method=artist.gettoptracks&artist={artist_name}&limit=1&api_key={LASTFM_API_KEY}&format=json")
                        if top_track_response.status_code == 200:
                            tracks = top_track_response.json().get('toptracks', {}).get('track', [])
                            if tracks:
                                final_recs.add(f"{tracks[0]['name']} by {tracks[0]['artist']['name']}")
    
    return sorted(list(final_recs))

def create_spotify_playlist(sp, songs, playlist_name):
    user_id = sp.current_user()['id']
    track_uris = []
    for song in songs:
        try:
            track_name, artist_name = song.rsplit(' by ', 1)
            results = sp.search(q=f"track:{track_name} artist:{artist_name}", type='track', limit=1)
            if items := results['tracks']['items']:
                track_uris.append(items[0]['uri'])
        except ValueError: pass
    if not track_uris: return "Could not find any songs on Spotify."
    playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True)
    for i in range(0, len(track_uris), 100):
        sp.playlist_add_items(playlist['id'], track_uris[i:i+100])
    return f"Success! Listen here: {playlist['external_urls']['spotify']}"

# --- Authentication ---
auth_manager = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI, scope="user-top-read user-library-read playlist-modify-public", cache_path=".cache"
)
query_params = st.query_params
if "code" in query_params:
    try:
        code = query_params["code"]
        auth_manager.get_access_token(code)
        st.session_state.sp = spotipy.Spotify(auth_manager=auth_manager)
        st.query_params.clear()
    except Exception:
        st.error("Error during authentication.")
        st.session_state.sp = None

# --- UI Layout ---
st.set_page_config(layout="wide")
st.title("🎵 Music Recommender")

if not st.session_state.sp:
    st.write("Please log in with Spotify to begin.")
    auth_url = auth_manager.get_authorize_url()
    st.link_button("Login with Spotify", auth_url)
elif not st.session_state.taste_profile:
    st.header("Step 1: Build Your Taste Profile")
    st.write("To get started, we need to analyze your Spotify listening history.")
    if st.button("Build My Taste Profile"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        for update_type, data in build_user_profile(st.session_state.sp):
            if update_type == "status": status_text.text(data)
            elif update_type == "progress": progress_bar.progress(data)
        st.success("Your taste profile is ready!")
        st.rerun()
else:
    st.success("Your taste profile is loaded. Let's find some music!")
    
    st.sidebar.header("Recommendation Controls")
    pop_cap = st.sidebar.slider("Popularity Cap", 10000, 1000000, 500000, 10000, help="Artists with fewer listeners than this are considered 'lesser-known'.")
    subgenre_count = st.sidebar.slider("Number of Sub-Genres to Explore", 5, 50, 15, help="How many of your niche sub-genres to use as starting points.")
    artists_per_subgenre = st.sidebar.slider("Artists per Sub-Genre", 1, 10, 3, help="How many random artists to check within each selected sub-genre.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Step 2: Get Recommendations")
        if st.button("Find New Music"):
            with st.spinner("Exploring random sub-genres for hidden gems..."):
                recs = generate_recommendations(st.session_state.taste_profile, pop_cap, subgenre_count, artists_per_subgenre)
                st.session_state.recommendations = recs
            st.success(f"Found {len(recs)} new songs!")
    with col2:
        st.header("Step 3: Create Playlist")
        if st.session_state.recommendations:
            playlist_name = st.text_input("Playlist Name:", "Random Discovery")
            if st.button("Create Spotify Playlist"):
                with st.spinner("Creating playlist..."):
                    message = create_spotify_playlist(st.session_state.sp, st.session_state.recommendations, playlist_name)
                    if "Success" in message:
                        st.success("Playlist created!")
                        st.markdown(f"**[Click here to listen]({message.split(' ')[-1]})**")
                    else: st.error(message)
    st.header("Your Recommendations")
    if st.session_state.recommendations:
        st.text_area("Found Songs:", "\n".join(f"{i}. {s}" for i, s in enumerate(st.session_state.recommendations, 1)), height=300)

