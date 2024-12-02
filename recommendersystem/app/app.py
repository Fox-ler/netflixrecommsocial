# Import necessary libraries
import streamlit as st
import pandas as pd
import template as t
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure the layout of the Streamlit page to use a wide format and set a page title
st.set_page_config(layout="wide", page_title="Movie Recommender System")

# App title
st.title("Netlflix Recommender System")
logo_url = "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg"  # URL of the Netflix logo
st.image(logo_url, use_column_width=True)

# Load the dataset containing movie information
df_movies = pd.read_csv(r"C:\Users\Gebruiker\Streamlit\recommendersystem\data\movies.csv")


# Function to shorten the plot text for display
def shorten_text(text, max_sentences=3):
    sentences = text.split('. ')
    shortened_text = '. '.join(sentences[:max_sentences]) + " [...]"
    return shortened_text

# Define a function to create content-based recommendations
def get_content_based_recommendations(movie_id, df, top_n=6):
    df['combined_features'] = df['Plot'].fillna('') + ' ' + df['Title'].fillna('') + ' ' + df['Genre'].fillna('') + ' ' + df['Director'].fillna('') + ' ' + df['Country'].fillna('')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx = df.index[df['movieId'] == movie_id].tolist()[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_movie_indices = [i[0] for i in similarity_scores[1:top_n + 1]]
    return df.iloc[similar_movie_indices]

# Initialize session state for storing the current movieId if not already present
if 'movieId' not in st.session_state:
    st.session_state['movieId'] = 3114  # Default movieId to start

# Search bar to find a movie by title
search_query = st.text_input("Search for a movie", "")
if search_query:
    search_results = df_movies[df_movies['Title'].str.contains(search_query, case=False, na=False)]
    if not search_results.empty:
        selected_movie_title = st.selectbox("Select a Movie", search_results['Title'].tolist())
        st.session_state['movieId'] = search_results[search_results['Title'] == selected_movie_title].iloc[0]['movieId']
    else:
        st.write("No movies found. Please try another title.")

# Filter the dataset to only include the selected movie using the stored movieId
df_movie = df_movies[df_movies['movieId'] == st.session_state['movieId']]

# Display selected movie details in two columns: poster and information
cover, info = st.columns([2, 3])

with cover:
    st.image(df_movie['Poster'].iloc[0], caption="Movie Poster")

with info:
    st.title(df_movie['Title'].iloc[0])
    st.caption(f"{df_movie['Year'].iloc[0]} | {df_movie['Runtime'].iloc[0]} | {df_movie['Genre'].iloc[0]} | {df_movie['imdbRating'].iloc[0]} | {df_movie['Actors'].iloc[0]}")
    st.markdown(f"**Plot:** {shorten_text(df_movie['Plot'].iloc[0])}")



#st.subheader('Recommendations based on Frequently Reviewed Together (frequency)')
#df_freq_reviewed = pd.read_csv(r"C:\Users\Gebruiker\Streamlit\recommendersystem\app\recommendations\recommendations-seeded-freq.csv")
#movie_id = st.session_state['movieId']
#df_recommendations = df_freq_reviewed[df_freq_reviewed['movie_a'] == movie_id].sort_values(by='count', ascending=False)
#df_recommendations = df_recommendations.rename(columns={"movie_b": "movieId"})
#df_recommendations = df_recommendations.merge(df_movies, on='movieId')
#t.recommendations(df_recommendations.head(6))

# Content-Based Recommendations
#st.subheader('Content-Based Recommendations')
#recommended_movies = get_content_based_recommendations(st.session_state['movieId'], df_movies, top_n=6)
#df_recommendations = recommended_movies.rename(columns={"movieId": "movieId"})
#t.recommendations(df_recommendations)

# Define personas and preferences
personal_preferences = {
    "Berend": "Action, Horror, Thriller",
    "Ik": "Comedy, Horror, Action"
}

group_personas = {
    "Berend": "Action, Horror, Thriller",
    "Ik": "Comedy, Horror, Action",
    "Alice": "Drama, Romance, History",
    "Tom": "Sci-Fi, Fantasy, Adventure",
    "Eva": "Documentary, Biography, Mystery"
}

# Function to recommend movies based on combined preferences
def get_recommendations(preferences, df, top_n=6):
    """
    preferences: dict containing user names and their preferred genres
    df: DataFrame containing movie information
    top_n: number of recommendations to return
    """
    combined_genres = set(genre.strip() for user_prefs in preferences.values() for genre in user_prefs.split(","))
    df_filtered = df[df['Genre'].str.contains('|'.join(combined_genres), case=False, na=False)]
    # Add a weight for relevance based on overlapping genres
    df_filtered['relevance_score'] = df_filtered['Genre'].apply(
        lambda genres: sum([1 for genre in combined_genres if genre in genres])
    )
    return df_filtered.sort_values(by='relevance_score', ascending=False).head(top_n)

# Section for recommendations for you and Berend
st.subheader("Recommendations for You and Berend")
st.caption("We have made this recommendation list based on your mutual interests, which are: "
           f"{', '.join(set(genre.strip() for prefs in personal_preferences.values() for genre in prefs.split(',')))}")

personal_recommendations = get_recommendations(personal_preferences, df_movies, top_n=6)
t.recommendations(personal_recommendations)

# Section for recommendations for the group of 5 personas
st.subheader("Recommendations for the Group")
st.caption("We have made this recommendation list based on your mutual interests, which are: "
           f"{', '.join(set(genre.strip() for prefs in group_personas.values() for genre in prefs.split(',')))}")

group_recommendations = get_recommendations(group_personas, df_movies, top_n=6)
t.recommendations(group_recommendations)

# New Section: Match Your Genre with Berend's
st.subheader("Match Your Genre with Berend's")
st.caption("Select your favorite genres to see movies that match your preferences with Berend's.")

# Interactive multi-select input for user to pick their genres
user_genres = st.multiselect(
    "Select Your Preferred Genres",
    options=df_movies['Genre'].str.split(',').explode().str.strip().dropna().unique(),
    default=["Action", "Comedy"]  # Provide a default selection
)

# Combine user-selected genres with Berend's preferences
if user_genres:
    combined_genres = set(user_genres + personal_preferences["Berend"].split(", "))
    st.caption(f"Matching genres are: {', '.join(combined_genres)}")

    # Filter movies that match the combined genres
    df_combined_recommendations = df_movies[df_movies['Genre'].str.contains('|'.join(combined_genres), case=False, na=False)]
    df_combined_recommendations['relevance_score'] = df_combined_recommendations['Genre'].apply(
        lambda genres: sum([1 for genre in combined_genres if genre.strip().lower() in genres.lower()])
    )

    # Sort by relevance and display top 6
    df_combined_recommendations = df_combined_recommendations.sort_values(by='relevance_score', ascending=False).head(6)

    # Display recommendations
    if not df_combined_recommendations.empty:
        t.recommendations(df_combined_recommendations)
    else:
        st.write("No movies found that match the selected genres.")
    if st.button("Load More"):
        st.session_state['recommendation_index'] += BATCH_SIZE
else:
    st.write("Please select at least one genre to see recommendations.")