import streamlit as st
import spotipy
import requests
import json
import random
import time
import math
from spotipy.oauth2 import SpotifyOAuth
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI, LASTFM_API_KEY
from collections import Counter, defaultdict
from fuzzywuzzy import fuzz
import numpy as np
from datetime import datetime, timedelta

# --- App State Management ---
if 'sp' not in st.session_state:
    st.session_state.sp = None
if 'taste_profile' not in st.session_state:
    st.session_state.taste_profile = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'api_cache' not in st.session_state:
    st.session_state.api_cache = {}

# --- IMPROVED GENRE MAP ---
GENRE_MAP = {
    "Hip-Hop/Rap": {
        "keywords": ["hip-hop", "hip hop", "rap", "trap", "boom bap", "gangsta rap", "horrorcore", "emo rap", "cloud rap", "phonk", "underground hip-hop", "g-funk", "rapcore", "drill", "mumble rap", "conscious rap"],
        "exclude": ["garage rock", "garage punk"]
    },
    "Rock/Alternative": {
        "keywords": ["rock", "alternative", "alternative rock", "indie rock", "emo", "pop punk", "punk", "90s rock", "post-punk", "new wave", "nu metal", "garage rock", "post-punk revival", "grunge", "classic rock", "hard rock"],
        "exclude": ["hip hop", "electronic", "house"]
    },
    "Indie/Folk": {
        "keywords": ["indie", "indie pop", "indie folk", "folk", "singer-songwriter", "americana", "country", "dream pop", "shoegaze", "bedroom pop", "lo-fi", "acoustic"],
        "exclude": ["electronic", "house", "techno"]
    },
    "Electronic": {
        "keywords": ["electronic", "electropop", "house", "dubstep", "trip-hop", "downtempo", "synthpop", "dance", "electronica", "idm", "uk garage", "techno", "ambient", "drum and bass", "trance"],
        "exclude": ["garage rock", "punk rock"]
    },
    "Metal/Hardcore": {
        "keywords": ["metal", "metalcore", "post-hardcore", "death metal", "hardcore", "screamo", "deathcore", "djent", "nu-metal", "black metal", "doom metal", "thrash metal", "progressive metal"],
        "exclude": ["pop", "electronic"]
    },
    "Pop": {
        "keywords": ["pop", "pop rock", "power pop", "electropop", "dance pop", "synth-pop", "bubblegum pop"],
        "exclude": ["metal", "hardcore", "death"]
    },
    "R&B/Soul": {
        "keywords": ["rnb", "r&b", "soul", "funk", "neo-soul", "contemporary r&b", "motown", "gospel"],
        "exclude": ["rock", "metal"]
    },
    "Jazz/Blues": {
        "keywords": ["jazz", "blues", "bebop", "smooth jazz", "fusion", "swing", "big band", "contemporary jazz"],
        "exclude": ["rock", "pop", "electronic"]
    }
}

# --- Rate Limiting and Caching ---
class APIRateLimiter:
    def __init__(self, max_requests_per_second=4):
        self.max_requests_per_second = max_requests_per_second
        self.last_request_time = 0
        self.request_count = 0
        self.reset_time = time.time()
    
    def wait_if_needed(self):
        current_time = time.time()
        
        # Reset counter every second
        if current_time - self.reset_time >= 1.0:
            self.request_count = 0
            self.reset_time = current_time
        
        # If we've hit the limit, wait
        if self.request_count >= self.max_requests_per_second:
            sleep_time = 1.0 - (current_time - self.reset_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.request_count = 0
            self.reset_time = time.time()
        
        self.request_count += 1

rate_limiter = APIRateLimiter()

def cached_lastfm_request(url, params, cache_key=None):
    """Make a cached Last.fm API request with rate limiting."""
    if cache_key and cache_key in st.session_state.api_cache:
        return st.session_state.api_cache[cache_key]
    
    rate_limiter.wait_if_needed()
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if cache_key:
                st.session_state.api_cache[cache_key] = data
            return data
    except requests.RequestException:
        pass
    
    return None

# --- Improved Genre Classification ---
def classify_tag_to_genre(tag_name, tag_count=1):
    """Classify a tag to a genre using fuzzy matching and exclusion rules."""
    tag_lower = tag_name.lower()
    
    best_match = None
    best_score = 0
    
    for genre, config in GENRE_MAP.items():
        # Check exclusions first
        if any(exclude_term in tag_lower for exclude_term in config.get("exclude", [])):
            continue
        
        # Find best keyword match
        for keyword in config["keywords"]:
            if keyword in tag_lower:
                # Exact substring match gets high score
                score = 100
            else:
                # Use fuzzy matching for partial matches
                score = fuzz.ratio(keyword, tag_lower)
            
            # Weight by tag count (more popular tags get preference)
            weighted_score = score * math.log(tag_count + 1)
            
            if weighted_score > best_score and score > 70:  # Minimum similarity threshold
                best_score = weighted_score
                best_match = genre
    
    return best_match

# --- Enhanced Profile Building ---
def build_user_profile(sp):
    """Generates an enhanced taste profile with audio features and temporal weighting."""
    yield "status", "Fetching your Spotify listening history..."
    yield "progress", 0.05
    
    # Get more artists from different time ranges
    artists_data = []
    time_ranges = [('long_term', 1.0), ('medium_term', 0.7), ('short_term', 0.5)]
    
    for time_range, weight in time_ranges:
        try:
            results = sp.current_user_top_artists(limit=50, time_range=time_range)
            for artist in results['items']:
                artists_data.append({
                    'name': artist['name'],
                    'id': artist['id'],
                    'popularity': artist['popularity'],
                    'genres': artist['genres'],
                    'weight': weight,
                    'followers': artist['followers']['total']
                })
        except Exception:
            continue
    
    # Remove duplicates, keeping highest weight
    unique_artists = {}
    for artist in artists_data:
        name = artist['name']
        if name not in unique_artists or artist['weight'] > unique_artists[name]['weight']:
            unique_artists[name] = artist
    
    seed_artists = list(unique_artists.values())
    
    yield "status", "Analyzing your saved tracks..."
    yield "progress", 0.15
    
    # Get exclusion list from saved tracks
    all_liked_tracks = []
    try:
        liked_results = sp.current_user_saved_tracks(limit=50)
        all_liked_tracks.extend(liked_results['items'])
        
        # Get more saved tracks in batches
        offset = 50
        while len(all_liked_tracks) < 500 and liked_results['next']:
            try:
                liked_results = sp.next(liked_results)
                all_liked_tracks.extend(liked_results['items'])
            except:
                break
    except Exception:
        pass
    
    exclusion_list = {artist['name'] for item in all_liked_tracks for artist in item['track']['artists']}
    
    yield "status", "Gathering genre insights from Last.fm..."
    yield "progress", 0.25
    
    # Enhanced tag collection with confidence weighting
    weighted_tag_counts = defaultdict(float)
    genre_tree = {genre: {"total_weight": 0, "subgenres": {}, "artists": []} for genre in GENRE_MAP}
    
    processed_count = 0
    total_artists = len(seed_artists)
    
    for artist_data in seed_artists:
        artist_name = artist_data['name']
        artist_weight = artist_data['weight']
        
        # Add Spotify genres to the mix
        for spotify_genre in artist_data.get('genres', []):
            classified_genre = classify_tag_to_genre(spotify_genre, 10)  # High confidence for Spotify genres
            if classified_genre:
                weighted_tag_counts[spotify_genre] += artist_weight * 2
                genre_tree[classified_genre]["subgenres"][spotify_genre] = weighted_tag_counts[spotify_genre]
                genre_tree[classified_genre]["total_weight"] += artist_weight * 2
                genre_tree[classified_genre]["artists"].append(artist_name)
        
        # Get Last.fm tags
        cache_key = f"artist_tags_{artist_name}"
        params = {
            'method': 'artist.gettoptags',
            'artist': artist_name,
            'api_key': LASTFM_API_KEY,
            'format': 'json'
        }
        
        data = cached_lastfm_request('http://ws.audioscrobbler.com/2.0/', params, cache_key)
        
        if data and 'toptags' in data:
            tags = data['toptags'].get('tag', [])
            for tag_info in tags[:10]:  # Limit to top 10 tags per artist
                tag_name = tag_info['name']
                tag_count = int(tag_info.get('count', 1))
                
                # Weight by artist importance, tag popularity, and recency
                tag_weight = artist_weight * math.log(tag_count + 1)
                weighted_tag_counts[tag_name] += tag_weight
                
                # Classify tag to genre
                classified_genre = classify_tag_to_genre(tag_name, tag_count)
                if classified_genre:
                    genre_tree[classified_genre]["subgenres"][tag_name] = weighted_tag_counts[tag_name]
                    genre_tree[classified_genre]["total_weight"] += tag_weight
                    if artist_name not in genre_tree[classified_genre]["artists"]:
                        genre_tree[classified_genre]["artists"].append(artist_name)
        
        processed_count += 1
        progress = 0.25 + (0.65 * (processed_count / total_artists))
        yield "progress", progress
        yield "status", f"Processed {processed_count}/{total_artists} artists..."
    
    # Get audio features for additional profiling
    yield "status", "Analyzing audio characteristics..."
    yield "progress", 0.9
    
    audio_features_profile = {}
    if all_liked_tracks:
        track_ids = [item['track']['id'] for item in all_liked_tracks[:100] if item['track']['id']]
        try:
            # Get audio features in batches
            all_features = []
            for i in range(0, len(track_ids), 50):
                batch = track_ids[i:i+50]
                features = sp.audio_features(batch)
                if features:
                    all_features.extend([f for f in features if f])
            
            if all_features:
                # Calculate average audio features
                feature_keys = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'tempo']
                audio_features_profile = {}
                
                for key in feature_keys:
                    values = [f[key] for f in all_features if f.get(key) is not None]
                    if values:
                        audio_features_profile[key] = {
                            'mean': np.mean(values),
                            'std': np.std(values)
                        }
        except Exception:
            pass
    
    # Normalize genre weights
    total_weight = sum(details["total_weight"] for details in genre_tree.values())
    if total_weight > 0:
        for genre_data in genre_tree.values():
            genre_data["normalized_weight"] = genre_data["total_weight"] / total_weight
    
    yield "status", "Profile created successfully!"
    yield "progress", 1.0
    
    st.session_state.taste_profile = {
        "seed_artists": [artist['name'] for artist in seed_artists],
        "artist_exclusion_list": list(exclusion_list),
        "genre_tree": genre_tree,
        "audio_features_profile": audio_features_profile,
        "total_artists_analyzed": len(seed_artists),
        "creation_time": datetime.now().isoformat()
    }

# --- Enhanced Recommendation Engine ---
def generate_recommendations(profile, popularity_cap, subgenre_count, artists_per_subgenre, diversity_factor=0.7):
    """Enhanced recommendation generation with anti-bias measures for genre balance."""
    exclusion_list = set(profile.get("artist_exclusion_list", []))
    genre_tree = profile.get("genre_tree", {})
    
    # Get available genres with their weights
    available_genres = [(genre, details["normalized_weight"]) 
                       for genre, details in genre_tree.items() 
                       if details["total_weight"] > 0]
    
    if not available_genres:
        return ["Could not find any genres in your profile to search."]
    
    # ANTI-BIAS MEASURE 1: Genre Selection with Proportional Limits
    genres, weights = zip(*available_genres)
    genre_weights = np.array(weights) / sum(weights)
    
    # Limit any single genre to maximum 50% of selections (adjustable)
    max_genre_proportion = 0.5
    capped_weights = np.minimum(genre_weights, max_genre_proportion)
    capped_weights = capped_weights / capped_weights.sum()  # Re-normalize
    
    # Select genres using the capped weights
    num_genres_to_select = min(subgenre_count, len(genres))
    selected_genres_with_counts = {}
    
    # Distribute subgenre slots across genres more evenly
    for _ in range(subgenre_count):
        selected_genre = np.random.choice(genres, p=capped_weights)
        selected_genres_with_counts[selected_genre] = selected_genres_with_counts.get(selected_genre, 0) + 1
    
    # ANTI-BIAS MEASURE 2: Limit subgenres per genre
    max_subgenres_per_genre = max(3, subgenre_count // len(available_genres) + 1)
    for genre in list(selected_genres_with_counts.keys()):
        if selected_genres_with_counts[genre] > max_subgenres_per_genre:
            selected_genres_with_counts[genre] = max_subgenres_per_genre
    
    # For each selected genre, pick subgenres with bias reduction
    tags_to_explore = []
    for genre, count in selected_genres_with_counts.items():
        subgenres = genre_tree[genre]["subgenres"]
        if subgenres:
            subgenre_items = list(subgenres.items())
            
            if len(subgenre_items) <= count:
                # Use all subgenres if we don't have enough
                for subgenre_name, _ in subgenre_items:
                    tags_to_explore.append((subgenre_name, genre))
            else:
                # ANTI-BIAS MEASURE 3: Subgenre selection with diversity
                subgenre_names, subgenre_weights = zip(*subgenre_items)
                
                # Apply diversity factor to weights (higher diversity = more random)
                if diversity_factor < 0.8:
                    adjusted_weights = np.array(subgenre_weights) ** (1 - diversity_factor)
                else:
                    # High diversity mode: nearly uniform selection
                    adjusted_weights = np.ones(len(subgenre_weights))
                
                adjusted_weights = adjusted_weights / adjusted_weights.sum()
                
                # Select multiple subgenres without replacement
                selected_indices = np.random.choice(
                    len(subgenre_names), 
                    size=count,
                    replace=False,
                    p=adjusted_weights
                )
                
                for idx in selected_indices:
                    tags_to_explore.append((subgenre_names[idx], genre))
    
    if not tags_to_explore:
        return ["Could not select any sub-genres to explore."]
    
    # Enhanced recommendation collection with genre balancing
    recommendations_by_genre = defaultdict(list)
    
    for tag, parent_genre in tags_to_explore:
        cache_key = f"tag_artists_{tag}"
        params = {
            'method': 'tag.gettopartists',
            'tag': tag,
            'api_key': LASTFM_API_KEY,
            'format': 'json',
            'limit': min(200, artists_per_subgenre * 8)  # Get more options
        }
        
        data = cached_lastfm_request('http://ws.audioscrobbler.com/2.0/', params, cache_key)
        
        if data and 'topartists' in data and 'artist' in data['topartists']:
            artist_pool = data['topartists']['artist']
            
            # Score and filter artists
            scored_artists = []
            for artist_data in artist_pool:
                artist_name = artist_data['name']
                
                if artist_name in exclusion_list:
                    continue
                
                # Get artist info for popularity filtering
                info_cache_key = f"artist_info_{artist_name}"
                info_params = {
                    'method': 'artist.getinfo',
                    'artist': artist_name,
                    'api_key': LASTFM_API_KEY,
                    'format': 'json'
                }
                
                info_data = cached_lastfm_request('http://ws.audioscrobbler.com/2.0/', info_params, info_cache_key)
                
                if info_data and 'artist' in info_data:
                    artist_info = info_data['artist']
                    listener_count = int(artist_info.get('stats', {}).get('listeners', 0))
                    
                    if 1000 < listener_count < popularity_cap:
                        # ANTI-BIAS MEASURE 4: Genre-aware scoring
                        popularity_score = min(listener_count / popularity_cap, 1.0)
                        tag_relevance = 1.0 - (float(artist_data.get('rank', 100)) / 100.0)
                        
                        # Reduce genre bonus to prevent amplification
                        genre_bonus = 1.0
                        user_genre_weight = genre_tree[parent_genre]["normalized_weight"]
                        
                        # For over-represented genres, reduce the bonus
                        if user_genre_weight > 0.3:  # If genre is >30% of user's taste
                            genre_bonus = 0.8  # Slight penalty
                        elif artist_name in profile.get("genre_tree", {}).get(parent_genre, {}).get("artists", []):
                            genre_bonus = 0.6  # Known artist penalty
                        else:
                            genre_bonus = 1.1  # Small bonus for new artists in your genres
                        
                        total_score = (popularity_score * 0.4 + 
                                     tag_relevance * 0.4 + 
                                     genre_bonus * 0.2)
                        
                        scored_artists.append((artist_name, total_score, listener_count))
            
            # Select best artists from this tag, but limit per genre
            scored_artists.sort(key=lambda x: x[1], reverse=True)
            max_artists_this_tag = min(artists_per_subgenre, len(scored_artists))
            selected_artists = scored_artists[:max_artists_this_tag]
            
            # Get tracks for selected artists
            for artist_name, score, _ in selected_artists:
                track_cache_key = f"artist_tracks_{artist_name}"
                track_params = {
                    'method': 'artist.gettoptracks',
                    'artist': artist_name,
                    'limit': 8,  # Fewer tracks to check per artist
                    'api_key': LASTFM_API_KEY,
                    'format': 'json'
                }
                
                track_data = cached_lastfm_request('http://ws.audioscrobbler.com/2.0/', track_params, track_cache_key)
                
                if track_data and 'toptracks' in track_data and 'track' in track_data['toptracks']:
                    tracks = track_data['toptracks']['track']
                    if tracks:
                        # Select track with some randomness
                        if len(tracks) >= 3:
                            selected_track = random.choice(tracks[:3])  # Top 3 tracks
                        else:
                            selected_track = random.choice(tracks)
                        
                        recommendation = f"{selected_track['name']} by {selected_track['artist']['name']}"
                        recommendations_by_genre[parent_genre].append((recommendation, score))
    
    # ANTI-BIAS MEASURE 5: Final genre balancing
    final_recs = []
    
    # Ensure no genre contributes more than 60% of final recommendations
    max_per_genre = max(1, int(len(tags_to_explore) * artists_per_subgenre * 0.6))
    
    for genre, recs_with_scores in recommendations_by_genre.items():
        # Sort by score and limit per genre
        recs_with_scores.sort(key=lambda x: x[1], reverse=True)
        limited_recs = recs_with_scores[:max_per_genre]
        final_recs.extend([rec for rec, _ in limited_recs])
    
    # Remove duplicates and shuffle for variety
    seen = set()
    unique_recs = []
    for rec in final_recs:
        if rec not in seen:
            seen.add(rec)
            unique_recs.append(rec)
    
    # Shuffle to mix genres rather than clustering
    random.shuffle(unique_recs)
    
    return unique_recs

def create_spotify_playlist(sp, songs, playlist_name):
    """Creates a Spotify playlist with improved search matching."""
    user_id = sp.current_user()['id']
    track_uris = []
    failed_matches = []
    
    for song in songs:
        try:
            if ' by ' in song:
                track_name, artist_name = song.rsplit(' by ', 1)
                
                # Try multiple search strategies
                search_queries = [
                    f'track:"{track_name}" artist:"{artist_name}"',
                    f'"{track_name}" "{artist_name}"',
                    f'{track_name} {artist_name}'
                ]
                
                found = False
                for query in search_queries:
                    try:
                        results = sp.search(q=query, type='track', limit=5)
                        if results['tracks']['items']:
                            # Find best match
                            best_match = None
                            best_score = 0
                            
                            for track in results['tracks']['items']:
                                # Score based on name similarity and artist match
                                track_score = fuzz.ratio(track_name.lower(), track['name'].lower())
                                artist_score = max([fuzz.ratio(artist_name.lower(), artist['name'].lower()) 
                                                  for artist in track['artists']])
                                
                                combined_score = (track_score + artist_score) / 2
                                if combined_score > best_score and combined_score > 70:
                                    best_score = combined_score
                                    best_match = track
                            
                            if best_match:
                                track_uris.append(best_match['uri'])
                                found = True
                                break
                    except Exception:
                        continue
                
                if not found:
                    failed_matches.append(song)
                        
        except ValueError:
            failed_matches.append(song)
    
    if not track_uris:
        return "Could not find any songs on Spotify."
    
    # Create playlist
    try:
        playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True)
        playlist_id = playlist['id']
        
        # Add tracks in batches
        for i in range(0, len(track_uris), 100):
            sp.playlist_add_items(playlist_id, track_uris[i:i+100])
        
        success_msg = f"Success! Added {len(track_uris)} tracks"
        if failed_matches:
            success_msg += f" ({len(failed_matches)} tracks could not be found)"
        
        return f"{success_msg}. Listen here: {playlist['external_urls']['spotify']}"
        
    except Exception as e:
        return f"Error creating playlist: {str(e)}"

# --- Authentication ---
auth_manager = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID, 
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI, 
    scope="user-top-read user-library-read playlist-modify-public user-read-recently-played", 
    cache_path=".cache"
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

# --- Enhanced UI Layout ---
st.set_page_config(layout="wide", page_title="Enhanced Music Discovery")
st.title("🎵 Enhanced Music Discovery")
st.markdown("*Discover music that matches your taste using advanced profiling and recommendation algorithms*")

if not st.session_state.sp:
    st.write("Please log in with Spotify to begin your musical journey.")
    auth_url = auth_manager.get_authorize_url()
    st.link_button("🎧 Login with Spotify", auth_url)

elif not st.session_state.taste_profile:
    st.header("Step 1: Build Your Enhanced Taste Profile")
    st.write("We'll analyze your Spotify listening history, saved tracks, and audio preferences to create a comprehensive taste profile.")
    
    with st.expander("What we analyze:", expanded=False):
        st.markdown("""
        - **Long-term, medium-term, and recent listening patterns** (weighted by recency)
        - **Genre classification** using fuzzy matching and exclusion rules
        - **Audio characteristics** from your saved tracks (danceability, energy, etc.)
        - **Artist popularity preferences** and discovery patterns
        - **Last.fm tags** with confidence weighting
        """)
    
    if st.button("🔍 Build My Enhanced Taste Profile", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        for update_type, data in build_user_profile(st.session_state.sp):
            if update_type == "status": 
                status_text.text(data)
            elif update_type == "progress": 
                progress_bar.progress(data)
        
        st.success("✅ Your enhanced taste profile is ready!")
        st.rerun()

else:
    # Display profile summary
    profile = st.session_state.taste_profile
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Artists Analyzed", profile.get("total_artists_analyzed", 0))
    with col2:
        active_genres = sum(1 for g in profile["genre_tree"].values() if g["total_weight"] > 0)
        st.metric("Genres Identified", active_genres)
    with col3:
        total_tracks = len(profile.get("artist_exclusion_list", []))
        st.metric("Known Artists", total_tracks)
    
    # Genre distribution
    with st.expander("Your Music Taste Breakdown", expanded=False):
        genre_weights = [(genre, details.get("normalized_weight", 0)) 
                        for genre, details in profile["genre_tree"].items() 
                        if details.get("normalized_weight", 0) > 0]
        
        if genre_weights:
            genre_weights.sort(key=lambda x: x[1], reverse=True)
            for genre, weight in genre_weights:
                st.progress(weight, text=f"{genre}: {weight:.1%}")
    
    st.success("✅ Your enhanced taste profile is loaded. Let's discover some music!")
    
    # Enhanced sidebar controls
    st.sidebar.header("🎛️ Discovery Controls")
    
    st.sidebar.subheader("Popularity Settings")
    pop_cap = st.sidebar.slider(
        "Maximum Popularity", 
        50000, 2000000, 800000, 50000,
        help="Higher values include more mainstream artists"
    )
    
    st.sidebar.subheader("Exploration Depth")
    subgenre_count = st.sidebar.slider(
        "Subgenres to Explore", 
        5, 30, 12,
        help="How many of your subgenres to use as starting points"
    )
    
    artists_per_subgenre = st.sidebar.slider(
        "Artists per Subgenre", 
        2, 8, 4,
        help="How many artists to check within each subgenre"
    )
    
    st.sidebar.subheader("Discovery Style")
    diversity_factor = st.sidebar.slider(
        "Diversity vs Familiarity", 
        0.3, 1.0, 0.7, 0.1,
        help="Lower = more familiar styles, Higher = more diverse exploration"
    )
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🔍 Generate Recommendations")
        if st.button("Find New Music", type="primary"):
            with st.spinner("🎯 Using advanced algorithms to find your perfect matches..."):
                recs = generate_recommendations(
                    profile, pop_cap, subgenre_count, 
                    artists_per_subgenre, diversity_factor
                )
                st.session_state.recommendations = recs
            
            st.success(f"🎉 Discovered {len(recs)} new songs tailored to your taste!")
    
    with col2:
        st.header("📝 Create Playlist")
        if st.session_state.recommendations:
            # Analyze playlist sentiment for AI features
            sentiment, vibe_tags = analyze_playlist_sentiment(
                st.session_state.recommendations, 
                st.session_state.taste_profile
            )
            
            # Generate AI playlist name if Gemini is available
            if GEMINI_AVAILABLE:
                if st.button("🤖 Generate AI Playlist Name", help="Let AI create a creative name based on your music's vibe"):
                    with st.spinner("🎨 AI is analyzing your music's vibe..."):
                        ai_name = generate_playlist_name(sentiment, vibe_tags, st.session_state.taste_profile)
                        st.session_state['suggested_name'] = ai_name
                
                # Use AI-generated name or fallback
                default_name = st.session_state.get('suggested_name', f"Discovery Mix {datetime.now().strftime('%m/%d')}")
            else:
                default_name = f"Discovery Mix {datetime.now().strftime('%m/%d')}"
            
            playlist_name = st.text_input("Playlist Name:", default_name)
            
            # Show sentiment analysis
            with st.expander("🎭 Playlist Vibe Analysis", expanded=False):
                st.write(f"**Primary Sentiment:** {sentiment.title()}")
                st.write(f"**Vibe Tags:** {', '.join(vibe_tags[:4])}")
                if GEMINI_AVAILABLE:
                    st.write("*AI will use this analysis to generate playlist artwork and names*")
            
            # Create playlist button
            create_button_text = "🎵 Create Playlist"
            if GEMINI_AVAILABLE:
                create_button_text += " + AI Cover"
            
            if st.button(create_button_text, type="primary"):
                with st.spinner("Creating your personalized playlist with AI enhancements..."):
                    message = create_spotify_playlist(
                        st.session_state.sp, 
                        st.session_state.recommendations, 
                        playlist_name,
                        sentiment,
                        vibe_tags
                    )
                    
                if "Success" in message:
                    st.success("🎉 Playlist created successfully!")
                    # Extract URL from message
                    url = message.split("Listen here: ")[-1]
                    st.markdown(f"**[🎧 Open in Spotify]({url})**")
                    
                    if GEMINI_AVAILABLE:
                        st.info("🎨 AI cover concept generated! (Note: Actual cover upload requires additional Spotify permissions)")
                else:
                    st.error(message)
        else:
            st.info("Generate recommendations first to create a playlist.")
    
    # Display recommendations
    st.header("🎵 Your Personalized Recommendations")
    if st.session_state.recommendations:
        # Show in a more readable format
        recommendations_text = "\n".join(
            f"{i:2d}. {song}" 
            for i, song in enumerate(st.session_state.recommendations, 1)
        )
        
        st.text_area(
            f"Discovered Songs ({len(st.session_state.recommendations)} tracks):",
            recommendations_text, 
            height=400,
            help="These recommendations are ranked by relevance to your music taste"
        )
        
        # Option to refresh recommendations
        if st.button("🔄 Generate Different Recommendations"):
            with st.spinner("Finding alternative matches..."):
                recs = generate_recommendations(
                    profile, pop_cap, subgenre_count, 
                    artists_per_subgenre, diversity_factor
                )
                st.session_state.recommendations = recs
            st.rerun()
    else:
        st.info("Click 'Find New Music' to get started with personalized recommendations!")

    # Cache management
    if st.sidebar.button("🗑️ Clear Cache"):
        st.session_state.api_cache = {}
        st.sidebar.success("Cache cleared!")
