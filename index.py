import streamlit as st
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import pickle
import time
import socket

# ====================== Configuration ======================
TMDB_API_KEY = "d08779f"
OMDB_API_KEY ="ef8c779f"
csv_path= "movies.csv"
CACHE_DIR = "./cache"
POSTER_TIMEOUT = 8  # seconds
MAX_RETRIES = 2  # for API calls

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# ====================== Initialize Session State ======================
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'recommend_trigger' not in st.session_state:
    st.session_state.recommend_trigger = False

# ====================== Data Caching ======================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_movies():
    import streamlit as st
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    csv_path = "F:/Users/Nitin Sharma/OneDrive/Documents/CT/movies.csv" 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "movies.csv")

    if not os.path.exists(csv_path):
        st.error(f"❌ CSV file not found at: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path, low_memory=False)

    # Calculate similarity matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Extract unique genres
    if 'genres' in df.columns:
        genres = sorted(set(g for gs in df['genres'].dropna() for g in gs.split('|')))
    else:
        genres = []

    # Extract unique years
    if 'release_year' in df.columns:
        years = sorted(df['release_year'].dropna().unique())
    else:
        years = []

    return df, similarity, genres, years


    # Expected columns with descriptions
    expected_cols = {
        'title': 'Movie title',
        'genres': 'Comma-separated genres',
        'overview': 'Plot summary',
        'tagline': 'Movie tagline',
        'cast': 'Main cast members',
        'director': 'Director(s)',
        'year': 'Release year',
        'rating': 'Average rating',
        'runtime': 'Movie duration in minutes'
    }
    
    # Keep only columns that exist in the dataframe
    df = df[[col for col in expected_cols if col in df.columns]].fillna('')
    
    if 'title' not in df.columns:
        st.error("'title' column missing in dataset.")
        return None, None, None, None

    # Create combined features for similarity calculation
    df['combined'] = df.apply(lambda row: ' '.join(
        str(row[col]) for col in df.columns 
        if col != 'title' and col in expected_cols
    ), axis=1)
    df['combined'] = df['combined'].str.lower()

    # Check for cached similarity matrix
    cache_file = os.path.join(CACHE_DIR, "similarity_cache.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            vectorizer, vectors, similarity = pickle.load(f)
    else:
        with st.spinner("Calculating movie similarities (this may take a while)..."):
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            vectors = vectorizer.fit_transform(df['combined'])
            similarity = cosine_similarity(vectors)
            # Cache the results
            with open(cache_file, 'wb') as f:
                pickle.dump((vectorizer, vectors, similarity), f)

    # Extract available genres
    genres = set()
    if 'genres' in df.columns:
        for genre_str in df['genres']:
            genres.update([g.strip() for g in str(genre_str).split(',') if g.strip()])
    
    # Extract available years if column exists
    years = []
    if 'year' in df.columns:
        years = sorted([y for y in df['year'].unique() if str(y).isdigit()], reverse=True)
    
    return df, similarity, sorted(list(genres)), years

# ====================== Robust Poster Functions ======================
@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 1 day
def fetch_tmdb_poster(title, year=None):
    for attempt in range(MAX_RETRIES):
        try:
            params = {
                "api_key": TMDB_API_KEY,
                "query": title,
                "include_adult": False
            }
            if year:
                params["year"] = year
                
            response = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params=params,
                timeout=POSTER_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('results'):
                # Try to find exact match first
                exact_matches = [r for r in data['results'] if r['title'].lower() == title.lower()]
                if exact_matches:
                    result = exact_matches[0]
                else:
                    result = data['results'][0]
                
                path = result.get("poster_path")
                return f"https://image.tmdb.org/t/p/w500{path}" if path else None
                
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:  # Last attempt
                st.warning(f"Couldn't fetch TMDB poster for '{title}'. Using placeholder.")
                return None
            time.sleep(1)  # Wait before retrying
            continue
            
    return None

@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 1 day
def fetch_omdb_poster(title, year=None):
    for attempt in range(MAX_RETRIES):
        try:
            params = {
                "apikey": OMDB_API_KEY,
                "t": title,
                "type": "movie"
            }
            if year:
                params["y"] = year
                
            response = requests.get(
                "http://www.omdbapi.com/",
                params=params,
                timeout=POSTER_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("Poster") and data["Poster"] != "N/A":
                return data["Poster"]
                
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:  # Last attempt
                st.warning(f"Couldn't fetch OMDB poster for '{title}'. Using placeholder.")
                return None
            time.sleep(1)  # Wait before retrying
            continue
            
    return None

def get_movie_poster(title, year=None):
    # Try TMDB first, then OMDB
    poster = fetch_tmdb_poster(title, year) or fetch_omdb_poster(title, year)
    if not poster:
        # If no poster found, use a genre-specific placeholder
        genre_placeholder = "https://via.placeholder.com/300x450/333/cccccc?text=" + title.replace(" ", "+")
        return genre_placeholder
    return poster

# ====================== Recommendation ======================
def recommend(title, df, sim_matrix, n=5, genre_filter=None, year_range=None, min_rating=None):
    title = title.lower().strip()
    matches = df[df['title'].str.lower() == title]
    
    if matches.empty:
        st.warning(f"Movie '{title}' not found in dataset.")
        return pd.DataFrame()
    
    idx = matches.index[0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    
    # Sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get movie indices and similarity scores
    movie_indices = [i[0] for i in sim_scores[1:n*2+1]]  # Get extra in case we filter some out
    movie_scores = [i[1] for i in sim_scores[1:n*2+1]]
    
    # Create recommendations dataframe
    recs = df.iloc[movie_indices].copy()
    recs['similarity'] = movie_scores
    
    # Apply filters
    if genre_filter:
        genre_mask = recs['genres'].str.contains('|'.join(genre_filter), case=False, na=False)
        recs = recs[genre_mask]
    
    if year_range and 'year' in recs.columns:
        year_mask = (recs['year'] >= year_range[0]) & (recs['year'] <= year_range[1])
        recs = recs[year_mask]
    
    if min_rating and 'rating' in recs.columns:
        rating_mask = recs['rating'] >= min_rating
        recs = recs[rating_mask]
    
    # Return top n recommendations
    return recs.head(n)

# ====================== UI Rendering ======================
def render_movie_card(title, df, col, show_add_button=True):
    with col:
        # Get movie details
        movie = df[df['title'] == title].iloc[0]
        year = movie['year'] if 'year' in movie else None
        rating = f"{movie['rating']}/10" if 'rating' in movie and pd.notna(movie['rating']) else "N/A"
        runtime = f"{movie['runtime']} min" if 'runtime' in movie and pd.notna(movie['runtime']) else "N/A"
        
        # Display poster
        poster = get_movie_poster(title, year)
        st.image(poster, use_container_width=True)
        
        # Display basic info
        st.markdown(f"**🎬 {title}**")
        if year or rating or runtime:
            info_text = []
            if year:
                info_text.append(f"📅 {int(year)}")
            if rating:
                info_text.append(f"⭐ {rating}")
            if runtime:
                info_text.append(f"⏱️ {runtime}")
            st.caption(" | ".join(info_text))
        
        # Add to watchlist button
        if show_add_button:
            # Create a unique key for each button using the movie title and current time
            button_key = f"add_{title}_{time.time()}"
            if st.button(f"➕ Add to Watchlist", key=button_key):
                if title not in st.session_state.watchlist:
                    st.session_state.watchlist.append(title)
                    st.success(f"Added '{title}' to your watchlist!")
                    # Force a rerun to update the UI
                    st.experimental_rerun()
        
        # Expandable detailed info
        with st.expander("ℹ️ More Info"):
            if 'genres' in movie:
                st.markdown(f"**🎭 Genres:** {movie['genres']}")
            if 'overview' in movie and pd.notna(movie['overview']):
                st.markdown(f"**📖 Overview:** {movie['overview']}")
            if 'tagline' in movie and pd.notna(movie['tagline']):
                st.markdown(f"**💬 Tagline:** _{movie['tagline']}_")
            if 'director' in movie and pd.notna(movie['director']):
                st.markdown(f"**🎥 Director:** {movie['director']}")
            if 'cast' in movie and pd.notna(movie['cast']):
                st.markdown(f"**👥 Cast:** {movie['cast']}")

def render_recommendations(recs, df):
    if recs.empty:
        st.warning("No recommendations available with current filters.")
        return
    
    st.subheader("🎬 Recommended Movies")
    
    # Display in a responsive grid
    cols_per_row = 3
    cols = st.columns(cols_per_row)
    
    for i, (_, movie) in enumerate(recs.iterrows()):
        render_movie_card(movie['title'], df, cols[i % cols_per_row])
        
        # Start new row after cols_per_row movies
        if (i + 1) % cols_per_row == 0 and (i + 1) < len(recs):
            cols = st.columns(cols_per_row)

def show_watchlist():
    st.sidebar.subheader("📝 Your Watchlist")
    
    if not st.session_state.watchlist:
        st.sidebar.info("Your watchlist is empty. Add some movies!")
        return
    
    for movie in st.session_state.watchlist[:10]:  # Show first 10 to avoid clutter
        cols = st.sidebar.columns([4, 1])
        cols[0].write(f"🎥 {movie}")
        # Create a unique key for each remove button
        if cols[1].button("❌", key=f"remove_{movie}_{time.time()}"):
            st.session_state.watchlist.remove(movie)
            st.sidebar.success(f"Removed '{movie}' from watchlist!")
            # Force a rerun to update the UI
            st.experimental_rerun()
    
    if len(st.session_state.watchlist) > 10:
        st.sidebar.info(f"Showing 10 of {len(st.session_state.watchlist)} movies in watchlist.")
    
    if st.sidebar.button("🗑️ Clear All", key="clear_all"):
        st.session_state.watchlist.clear()
        st.sidebar.success("Watchlist cleared!")
        st.experimental_rerun()

# ====================== Main App ======================
def main():
    # Configure page
    st.set_page_config(
        page_title="AI Movie Recommender",
        layout="wide",
        page_icon="🎬",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius: 6px;
            padding: 0.25rem 0.5rem;
            font-size: 0.9rem;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .css-1aumxhk {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
        }
        .movie-card {
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
        }
        .movie-card:hover {
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading movie data..."):
        df, similarity, genres, years = load_movies()
    
    if df is None:
        st.error("Failed to load movie data. Please check the data source.")
        return
    
    # Main header
    st.title("🎬 AI Movie Recommendation System")
    st.markdown("""
        Discover your next favorite movie based on content similarity. 
        Get personalized recommendations powered by machine learning and enriched with movie posters.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🔍 Search Options")
        
        # Movie selection
        selected_movie = st.selectbox(
            "Choose a movie",
            df['title'].sort_values(),
            index=0,
            help="Select a movie to get similar recommendations"
        )
        
        # Filters
        with st.expander("⚙️ Filters", expanded=True):
            genre_filter = st.multiselect(
                "Filter by genre",
                genres,
                help="Only show recommendations from these genres"
            )
            
            if years:
                year_range = st.slider(
                    "Release year range",
                    min(years),
                    max(years),
                    (min(years), max(years)),
                    help="Only show recommendations within this year range"
                )
            else:
                year_range = None
                
            if 'rating' in df.columns:
                min_rating = st.slider(
                    "Minimum rating",
                    0.0,
                    10.0,
                    6.0,
                    0.5,
                    help="Only show movies with at least this rating"
                )
            else:
                min_rating = None
        
        # Number of recommendations
        num_recs = st.slider(
            "Number of recommendations",
            3,
            15,
            6,
            help="How many recommendations to show"
        )
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Get Recommendations", use_container_width=True):
                st.session_state.recommend_trigger = True
        with col2:
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.session_state.recommend_trigger = False
        
        # Watchlist section
        show_watchlist()
        
        # Footer
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; font-size: 0.8rem; color: #666;">
                <p>Powered by TMDB and OMDB APIs</p>
                <p>Movie data © {}</p>
            </div>
        """.format(datetime.now().year), unsafe_allow_html=True)
    
    # Main content area
    if st.session_state.recommend_trigger:
        with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
            try:
                recs = recommend(
                    selected_movie,
                    df,
                    similarity,
                    num_recs,
                    genre_filter,
                    year_range,
                    min_rating
                )
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                st.stop()
        
        if not recs.empty:
            # Show selected movie info
            st.subheader(f"🎥 You selected: {selected_movie}")
            selected_cols = st.columns([2, 3])
            
            with selected_cols[0]:
                selected_movie_data = df[df['title'] == selected_movie].iloc[0]
                year = selected_movie_data['year'] if 'year' in selected_movie_data else None
                poster = get_movie_poster(selected_movie, year)
                st.image(poster, use_container_width=True)
            
            with selected_cols[1]:
                if 'overview' in selected_movie_data and pd.notna(selected_movie_data['overview']):
                    st.markdown(f"**📖 Overview:** {selected_movie_data['overview']}")
                if 'genres' in selected_movie_data:
                    st.markdown(f"**🎭 Genres:** {selected_movie_data['genres']}")
                if 'director' in selected_movie_data and pd.notna(selected_movie_data['director']):
                    st.markdown(f"**🎥 Director:** {selected_movie_data['director']}")
                if 'cast' in selected_movie_data and pd.notna(selected_movie_data['cast']):
                    st.markdown(f"**👥 Cast:** {selected_movie_data['cast']}")
                if 'rating' in selected_movie_data and pd.notna(selected_movie_data['rating']):
                    st.markdown(f"**⭐ Rating:** {selected_movie_data['rating']}/10")
                if 'runtime' in selected_movie_data and pd.notna(selected_movie_data['runtime']):
                    st.markdown(f"**⏱️ Runtime:** {selected_movie_data['runtime']} minutes")
            
            # Show recommendations
            render_recommendations(recs, df)
        else:
            st.warning("No recommendations found with the current filters. Try adjusting your filters.")

if __name__ == "__main__":
    main()
