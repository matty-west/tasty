import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from actions import generate_recommendations, create_spotify_playlist
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI
import json

# --- App State Management ---
# Use Streamlit's session state to store variables between user interactions
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'sp' not in st.session_state:
    st.session_state.sp = None

# --- Authentication ---
def get_auth_manager():
    """Creates the Spotify OAuth manager."""
    # For deployment, this redirect URI must be updated in your Spotify Dev Dashboard
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope="playlist-modify-public", # The permission needed to create playlists
        cache_path=".cache"
    )

auth_manager = get_auth_manager()

# This block handles the redirect back from Spotify's login page
query_params = st.query_params
if "code" in query_params:
    try:
        code = query_params["code"]
        # Exchange the code for an access token
        token_info = auth_manager.get_access_token(code)
        # Save the authenticated Spotify object in the session state
        st.session_state.sp = spotipy.Spotify(auth_manager=auth_manager)
        # Clear the ?code=... from the URL for a cleaner user experience
        st.query_params.clear()
    except Exception as e:
        st.error(f"Error during authentication: {e}")
        st.session_state.sp = None

# --- UI Layout ---
st.set_page_config(layout="wide")
st.title("🎵 Music Recommender")

# --- Login / Main App Display ---
# If the user is not logged in, show the login button.
if not st.session_state.sp:
    st.write("Please log in with Spotify to begin.")
    auth_url = auth_manager.get_authorize_url()
    st.link_button("Login with Spotify", auth_url)
# If the user is logged in, show the full application.
else:
    st.success("Successfully logged in to Spotify!")
    
    # --- Helper Function to Load Genres for UI ---
    def load_genres():
        try:
            with open('taste_profile.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                genres = [g for g, d in data.get('genre_tree', {}).items() if d.get('total_weight', 0) > 0 and g != "Other"]
                return sorted(genres)
        except FileNotFoundError:
            return []

    # --- Sidebar for Controls ---
    st.sidebar.header("Recommendation Controls")
    pop_cap = st.sidebar.slider("Popularity Cap", 10000, 1000000, 500000, 10000)
    st.sidebar.subheader("Genre Bias")
    genre_options = load_genres()
    selected_genres = [g for g in genre_options if st.sidebar.checkbox(g, key=f"genre_{g}")]
    similar_artist_limit = st.sidebar.slider("Discovery Breadth", 5, 50, 25)
    seed_count = st.sidebar.slider("Number of Seed Artists", 5, 50, 15)

    # --- Main Content Columns ---
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Step 1: Get Recommendations")
        if st.button("Find New Music"):
            with st.spinner("Analyzing profile and finding artists..."):
                recs = generate_recommendations(pop_cap, selected_genres, similar_artist_limit, seed_count)
                st.session_state.recommendations = recs
            st.success(f"Found {len(recs)} new songs!")

    with col2:
        st.header("Step 2: Create Playlist")
        if st.session_state.recommendations:
            playlist_name = st.text_input("Playlist Name:", "Last.fm Discoveries")
            if st.button("Create Spotify Playlist"):
                with st.spinner("Creating playlist..."):
                    # Pass the authenticated 'sp' object to the create_spotify_playlist function
                    message = create_spotify_playlist(st.session_state.sp, st.session_state.recommendations, playlist_name)
                    if "Success" in message:
                        st.success("Playlist created!")
                        st.markdown(f"**[Click here to listen]({message.split(' ')[-1]})**")
                    else:
                        st.error(message)

    # --- Results Display ---
    st.header("Your Recommendations")
    if st.session_state.recommendations:
        rec_list_str = "\n".join(f"{i}. {s}" for i, s in enumerate(st.session_state.recommendations, 1))
        st.text_area("Found Songs:", rec_list_str, height=300)
    else:
        st.info("Adjust controls and click 'Find New Music'.")

