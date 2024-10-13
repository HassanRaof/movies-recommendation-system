import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pickle
import streamlit as st
import requests
from PIL import Image
movies=pd.read_csv('C:/Users/ENG-MR/Desktop/machine learning/New folder/top10K-TMDB-movies.csv')
movies=movies[['id','title','overview','genre']]
movies['keywords']=movies['overview']+movies['genre']
new_data= movies.drop(columns=['overview','genre'],axis=1)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=10000,stop_words='english')
vector=cv.fit_transform(new_data['keywords'].values.astype('U')).toarray()
from sklearn.metrics.pairwise import cosine_similarity
sim=cosine_similarity(vector)
def recommand(movies):
    index=new_data[new_data['title']==movies].index[0]
    distance = sorted(list(enumerate(sim[index])), reverse=True, key=lambda vector:vector[1])
    for i in distance[1:6]:
        print(new_data.iloc[i[0]].title)
recommand("The Godfather")
pickle.dump(new_data, open('movies_list.pkl', 'wb'))
pickle.dump(sim, open('sim.pkl', 'wb'))

movies= pickle.load(open('movies_list.pkl', 'rb'))
similarity = pickle.load(open("sim.pkl", 'rb'))
movies_list=movies['title'].values



st.header("Movie Recommender System")
selectvalue=st.selectbox("Select movie from dropdown", movies_list)

image_path = 'C:/Users/ENG-MR/Desktop/nettt.jpg'

# Open and display the image
image = Image.open(image_path)
st.image(image,use_column_width=True)


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index=movies[movies['title']==movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
    recommend_movie=[]
    recommend_poster=[]
    for i in distance[1:6]:
        movies_id=movies.iloc[i[0]].id
        recommend_movie.append(movies.iloc[i[0]].title)
        recommend_poster.append(fetch_poster(movies_id))
    return recommend_movie, recommend_poster

if st.button("Show Recommend"):
    movie_name, movie_poster = recommend(selectvalue)
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.text(movie_name[0])
        st.image(movie_poster[0])
    with col2:
        st.text(movie_name[1])
        st.image(movie_poster[1])
    with col3:
        st.text(movie_name[2])
        st.image(movie_poster[2])
    with col4:
        st.text(movie_name[3])
        st.image(movie_poster[3])
    with col5:
        st.text(movie_name[4])
        st.image(movie_poster[4])

