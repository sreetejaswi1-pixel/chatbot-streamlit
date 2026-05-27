import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# Title
st.title("🎬 Movie Recommendation System")

# Description
st.write("""
Welcome to the Movie Recommendation App 🍿  
Get movie suggestions based on your favorite movie.
""")

# Features
st.header("✨ Features")

st.markdown("""
- 🎥 Movie recommendations
- ⭐ Popular movies
- 🍿 Genre-based suggestions
- 🔍 Easy search option
- 💻 Streamlit interactive UI
""")

# User Input
movie = st.text_input("Enter your favorite movie:")

# Button
if st.button("Recommend Movies"):

    if movie == "":
        st.error("Please enter a movie name.")

    else:

        st.write("### Recommended Movies 🍿")

        movie = movie.lower()

        # Marvel Movies
        if "avengers" in movie:

            st.success("""
1. Iron Man  
2. Captain America  
3. Thor  
4. Doctor Strange  
5. Black Panther  
""")

        # Telugu Movies
        elif "bahubali" in movie:

            st.success("""
1. RRR  
2. KGF  
3. Pushpa  
4. Salaar  
5. Magadheera  
""")

        # Romantic Movies
        elif "titanic" in movie:

            st.success("""
1. The Notebook  
2. Me Before You  
3. La La Land  
4. Dear John  
5. A Walk to Remember  
""")

        # General Movies
        else:

            st.info("""
1. Inception  
2. Interstellar  
3. Avatar  
4. Joker  
5. Spider-Man: No Way Home  
""")

# Footer
st.markdown("---")
st.caption("Built using Streamlit and Python 🚀")