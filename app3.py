import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from PIL import Image

# Configure Streamlit Page
st.set_page_config(
    page_title="Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI/UX Optimization
st.markdown("""
    <style>
        /* Global Settings */
        body {background-color: #f8f9fa; font-family: 'Segoe UI', sans-serif; color: #2c3e50;}

        /* Title Styling */
        .title {text-align: center; font-size: 2.5rem; font-weight: bold; color: #1e3a8a; margin-bottom: 0.5rem;}

        /* Divider */
        .divider {border-top: 3px solid #ffcc00; margin: 1rem 0;}

        /* Sidebar */
        [data-testid="stSidebar"] {background: #1e3a8a; padding: 20px;}
        .sidebar-title {color: #ffcc00; font-size: 1.5rem; text-align: center;}
        .sidebar-label {color: white; font-weight: bold;}

        /* Button */
        .stButton>button {background: #ffcc00; color: #1e3a8a; font-weight: bold; border-radius: 5px; padding: 10px;}
        .stButton>button:hover {background: #ffd633; color: #1e3a8a;}

        /* Recommendation Cards */
        .recommendation-card {
            padding: 15px;
            margin: 8px 0;
            border-radius: 10px;
            background: #ffffff;
            border: 2px solid #ffcc00;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
        }
        .recommendation-card h4 {color: #e67e22; margin-bottom: 5px;}
        .recommendation-card p {margin: 0; font-size: 14px; color: #2c3e50;}
        
        /* Data Table */
        .dataframe {border: 1px solid #dfe6e9 !important; border-radius: 8px;}
        .dataframe th {background: #1e3a8a !important; color: white !important;}

    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='title'>🍽 Restaurant Recommendation system</h1>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("update_dataset.csv")
    df['Restaurant Name'] = df['Restaurant Name'].str.replace('?', '').str.strip().str.title()
    df['Cuisines'].fillna('Unknown', inplace=True)
    return df

df = load_data()

# Model Caching
@st.cache_data
def compute_similarity(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    cuisine_matrix = vectorizer.fit_transform(data['Cuisines'])
    return linear_kernel(cuisine_matrix, cuisine_matrix)

cosine_sim = compute_similarity(df)
restaurant_index = {name: idx for idx, name in enumerate(df['Restaurant Name'])}

# Recommendation Function
def get_recommendations(name, top_n=5):
    if name not in restaurant_index:
        return f"⚠️ Restaurant '{name}' not found."
    
    idx = restaurant_index[name]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    return df.iloc[[i[0] for i in sim_scores[1:top_n+1]]][['Restaurant Name', 'Cuisines', 'Aggregate rating', 'Price range']]

# Sidebar Components
with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>🔍 Find Your Next Meal</h2>", unsafe_allow_html=True)
    st.image(Image.open('image.png'), width=150)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Controls
    selected_restaurant = st.selectbox("🍽️ Choose a Restaurant", df['Restaurant Name'].unique())
    rec_count = st.slider("🔢 Number of Recommendations", 1, 10, 5)
    
    generate_recs = st.button("🍽️ Get Recommendations")

# Main Content Area
if generate_recs:
    st.subheader(f"🔝 Top {rec_count} Recommendations for You")
    recommendations = get_recommendations(selected_restaurant, rec_count)
    
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        for _, row in recommendations.iterrows():
            with st.container():
                st.markdown(f"""
                    <div class='recommendation-card'>
                        <h4>{row['Restaurant Name']}</h4>
                        <p><b>Cuisines:</b> {row['Cuisines']}</p>
                        <p><b>⭐ Rating:</b> {row['Aggregate rating']}/5</p>
                        <p><b>💰 Price Range:</b> {row['Price range']}/4</p>
                    </div>
                """, unsafe_allow_html=True)

# Raw Data Toggle
if st.checkbox("📊 Show Raw Dataset"):
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
