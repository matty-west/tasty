# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Spotify Credentials
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

# Last.fm API Key
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")