import numpy
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import *

dataFrames_movies = pd.read_csv("tmdb_5000_movies.csv")
dataFrames_credits = pd.read_csv("tmdb_5000_credits.csv")

dataFrames_movies = dataFrames_movies.merge(dataFrames_credits,on="title")  # combining movies and credits on the basis of the movie title name


dataFrames_movies = dataFrames_movies[['movie_id','title','overview','genres','keywords','cast','crew']] # selecting particular attributes from the dataset which are usefull for making tags


dataFrames_movies.dropna(inplace = True)  # removing the rows which contain null values in any of their attributes


def lower_title(x):
    return x.lower()

def fetch_genres(x):
    l = []
    for i in ast.literal_eval(x):
        l.append(i['name'])
    return l

def fetch_keyword(x):
    l=[]
    for i in ast.literal_eval(x):
        l.append(i['name'])
    return l

def fetch_cast(x):
    l=[]
    counter = 0
    for i in ast.literal_eval(x):
        if counter <= 5 :
            l.append(i['name'])
            counter += 1
        else:
            break
    return l

def fetch_crew(x):
    l=[]
    for i in ast.literal_eval(x):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l


dataFrames_movies['title'] = dataFrames_movies['title'].apply(lower_title)

dataFrames_movies['genres'] = dataFrames_movies['genres'].apply(fetch_genres)
dataFrames_movies['keywords'] = dataFrames_movies['keywords'].apply(fetch_keyword)
dataFrames_movies["cast"] = dataFrames_movies["cast"].apply(fetch_cast)
dataFrames_movies["crew"] = dataFrames_movies["crew"].apply(fetch_crew)

def splitting_of_string(s):
    return s.split()

dataFrames_movies["overview"] = dataFrames_movies["overview"].apply( splitting_of_string )

def remove_spaces(l):
    new_list = []
    for i in l:
        new_list.append( i.replace(" ", "") )
    return new_list

dataFrames_movies['genres'] = dataFrames_movies['genres'].apply( remove_spaces )
dataFrames_movies['keywords'] = dataFrames_movies['keywords'].apply( remove_spaces )
dataFrames_movies['cast'] = dataFrames_movies['cast'].apply( remove_spaces )
dataFrames_movies['crew'] = dataFrames_movies['crew'].apply( remove_spaces )


dataFrames_movies['tags'] = dataFrames_movies['genres'] + dataFrames_movies['keywords'] + dataFrames_movies["cast"] + dataFrames_movies['crew'] + dataFrames_movies["overview"]

final_movie_data = dataFrames_movies[['movie_id','title','tags']]

def list_to_string(l):
    return " ".join(l)

final_movie_data.loc[: ,'tags'] = final_movie_data["tags"].apply( list_to_string )


def stemming(x):
    l=[]
    for i in x.split():
        l.append( PorterStemmer().stem(i) )
    return " ".join(l)

final_movie_data.loc[:,'tags'] = final_movie_data['tags'].apply( stemming )

vectors = CountVectorizer( stop_words = "english" ).fit_transform( final_movie_data['tags'] ).toarray()


array_of_similarity_values_for_all_movies = cosine_similarity(vectors)

numpy.save("array_of_similarity_values_for_all_movies.npy",array_of_similarity_values_for_all_movies)
final_movie_data.to_pickle('final_movie_data.pkl')

# def recommend(x):
#     index = final_movie_data[ final_movie_data['title'] == x ].index[0]
#     similarity_values_for_one_movie = array_of_similarity_values_for_all_movies [ index ]
#     list_of_similar_movies = sorted( list( enumerate( similarity_values_for_one_movie ) ), reverse=True, key=lambda x:x[1] )[1:6]
#     for i in list_of_similar_movies:
#         print( final_movie_data.iloc[ i[0],1 ].title() ) # here title is a string function


# while True:
#     str = input("Enter a movie name : ").lower().strip()
#     recommend(str)
#     print("*"*30)
#     print("Do you want to use it again ? ")
#     ans = input("Enter 'no' to exit or enter 'yes' to use it again : ").strip().lower()
#     if ans == 'no':
#         break



# NEED to check if the input movie name is present or not!!!!!
# try to not use ast
# .index[0] ??? in recommend()
