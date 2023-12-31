import streamlit as st
import pickle
import pandas as pd
import requests
import gzip

#def remove_file_folder(filename):
#    try:
#        os.remove(filename)
#        print(f"The file '{filename}' has been deleted successfully.")
#    except FileNotFoundError:
#        print(f"The file '{filename}' does not exist.")
#
#remove_file_folder("similarity.pkl")
#remove_file_folder("zsn6wo0r")
#wget.download("https://ufile.io/zsn6wo0r", "similarity.pkl")

def compress_pickle(input_filename, output_filename):
    with open(input_filename, 'rb') as file:
        data = file.read()

    with gzip.open(output_filename, 'wb') as gz_file:
        gz_file.write(data)

    print(f"The pickle file '{input_filename}' has been compressed to '{output_filename}'.")

def decompress_pickle(input_filename, output_filename):
    with gzip.open(input_filename, 'rb') as gz_file:
        data = gz_file.read()

    with open(output_filename, 'wb') as file:
        file.write(data)

    print(f"The compressed pickle file '{input_filename}' has been decompressed to '{output_filename}'.")


decompress_pickle("similarity.pkl.gz", "similarity.pkl")
def fetch_poster(id):
    api_key = "8cf43ad9c085135b9479ad5cf6bbcbda"
    response = requests.get(f"https://api.themoviedb.org/3/movie/{id}?api_key={api_key}&language=en-US")
    data = response.json()
    return f"https://image.tmdb.org/t/p/w500/{data['poster_path']}"

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_posters

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

st.title('Movie Recommdation System')

selected_movie_name = st.selectbox('Select a movie you like', movies['title'].values)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])
 
    with col2:
        st.text(names[1])
        st.image(posters[1])
 
    with col3:
        st.text(names[2])
        st.image(posters[2])

    with col4:
        st.text(names[3])
        st.image(posters[3])

    with col5:
        st.text(names[4])
        st.image(posters[4])
