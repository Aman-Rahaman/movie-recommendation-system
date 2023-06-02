import numpy
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity


# We are having two datasets for movies.
# One is having the casts and crew in the movie and the other one is having the rest of the informations
# like: title, geners, keywords, overview, budget, release date, profit, etc.
dataFrames_movies1 = pd.read_csv("movie_dataset_1.csv")
dataFrames_movies2 = pd.read_csv("movie_dataset_2.csv")


dataFrames_movies = dataFrames_movies1.merge(dataFrames_movies2, on="title")
# Merging the datasets. now I have a dataset having all the informations related to a movie.


dataFrames_movies = dataFrames_movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
# Here we are selecting some particular attributes from the dataset which are usefull for making the tags for each movie.
# Here movie_id is a unique id for each movie which is in the website of TMDB.
# This will be later very helpfull to fetch posters from the website.


dataFrames_movies.dropna(inplace=True) 
# Removing the rows which contain null values in any of their attributes.
# This is because null value in any of these attributes will show wrong result.
# we need the values for the inportant attributes which identifies a movie.


# Function to convert a string into all lower case.
# This I will use to lower case the movie titles so that it becomes easier to check the input movie name in the dataset.
# Python is case sensitive.
# So a movie title in lower case wont be equal to the same movie title but in different case, like title case or upper case.
def lower_title(x):
    return x.lower()


def fetch_genres(x):
    generes = []
    for i in ast.literal_eval(x):
        generes.append(i['name'])
    return generes


def fetch_keyword(x):
    keyword = []
    for i in ast.literal_eval(x):
        keyword.append(i['name'])
    return keyword


def fetch_cast(x):
    casts = []
    counter = 0
    for i in ast.literal_eval(x):
        if counter <= 5:
            casts.append(i['name'])
            counter += 1
        else:
            break
    return casts


def fetch_crew(x):
    crew = []
    for i in ast.literal_eval(x):
        if i['job'] == 'Director':
            crew.append(i['name'])
            break
    return crew


dataFrames_movies['title'] = dataFrames_movies['title'].apply(lower_title)

dataFrames_movies['genres'] = dataFrames_movies['genres'].apply(fetch_genres)
dataFrames_movies['keywords'] = dataFrames_movies['keywords'].apply(fetch_keyword)
dataFrames_movies["cast"] = dataFrames_movies["cast"].apply(fetch_cast)
dataFrames_movies["crew"] = dataFrames_movies["crew"].apply(fetch_crew)


def splitting_of_string(s):
    return s.split()


dataFrames_movies["overview"] = dataFrames_movies["overview"].apply(splitting_of_string)


def remove_spaces(x):
    new_list = []
    for i in x:
        new_list.append(i.replace(" ", ""))
    return new_list


dataFrames_movies['genres'] = dataFrames_movies['genres'].apply(remove_spaces)
dataFrames_movies['keywords'] = dataFrames_movies['keywords'].apply(remove_spaces)
dataFrames_movies['cast'] = dataFrames_movies['cast'].apply(remove_spaces)
dataFrames_movies['crew'] = dataFrames_movies['crew'].apply(remove_spaces)


dataFrames_movies['tags'] = dataFrames_movies['genres'] + dataFrames_movies['keywords'] + dataFrames_movies["cast"] + dataFrames_movies['crew'] + dataFrames_movies["overview"]

final_movie_data = dataFrames_movies[['movie_id', 'title', 'tags']]


def list_to_string(x):
    return " ".join(x)


final_movie_data.loc[:, 'tags'] = final_movie_data["tags"].apply(list_to_string)


def stemming(x):
    stemmed_words = []
    for i in x.split():
        stemmed_words.append(PorterStemmer().stem(i))
    return " ".join(stemmed_words)


final_movie_data.loc[:, 'tags'] = final_movie_data['tags'].apply(stemming)

vectors = CountVectorizer(stop_words="english").fit_transform(final_movie_data['tags']).toarray()


array_of_similarity_values_for_all_movies = cosine_similarity(vectors)

numpy.save("array_of_similarity_values_for_all_movies.npy", array_of_similarity_values_for_all_movies)
final_movie_data.to_pickle('final_movie_data.pkl')



# # NEED to check if the input movie name is present or not!!!!!
# # try to not use ast
# # .index[0] ??? in recommend()
