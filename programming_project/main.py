import numpy
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

dataFrames_movies1 = pd.read_csv("movie_dataset_1.csv")
print("open first csv file")
# This dataset1.csv file contains the TMDB_id, actors names and the other people in the unit.
# TMDB is similar to IMDB. TMDB = 'The Movie Database'

dataFrames_movies2 = pd.read_csv("movie_dataset_2.csv")
print("open second csv file")
# This dataset2.csv file contains a lot of data with respect to each movie.
# Most of the attributes in this are not in much use of use.
# Therefore we will only take some selective attributes.

dataFrames_movies = dataFrames_movies1.merge(dataFrames_movies2, on="title")
print("merging the frames")
# combining movies and credits on the basis of the movie title name.
# now attributes from both the files are in one dataframe.


dataFrames_movies = dataFrames_movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
print("selecting attributes")
# selecting particular attributes from the dataset which are usefull for making tags.
# selecting only those attributes which are helpful in making the tag.


dataFrames_movies.dropna(inplace=True)
# removing the rows which contain null values in any of their attributes
# this is because these attributes are important to make the tags.
# if for a movie we have null values in any of the attributes, the the system will show wrong result for that movie.


# this is a function to convert all the titles into lower case.
# this will help in searching at the end.
def lower_case(x):
    return x.lower()


# This is a function to fetch the actors return a list.
# Here we are using ast.literal_eval because of the format in which the labels are in the dataframe.
# The labels are in the form of dictionaries in a list but the whole thing is a string.
# that's why we are using ast.literal_eval
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


# This is the function to fetch the labels of a particular film from the dataframe and return a list.
# even here we are using ast.literal_eval
def fetch_keyword(x):
    keyword = []
    for i in ast.literal_eval(x):
        keyword.append(i['name'])
    return keyword


# function to fetch the generes from the dataset and return a list.
# even here we are using ast.literal_eval
def fetch_genres(x):
    generes = []
    for i in ast.literal_eval(x):
        generes.append(i['name'])
    return generes


# function to fetch the people behind the unit from the dataset and return a list.
# even here we are using ast.literal_eval
def fetch_crew(x):
    crew = []
    for i in ast.literal_eval(x):
        if i['job'] == 'Director':
            crew.append(i['name'])
            break
    return crew


print("movie_name to lower case")
dataFrames_movies['title'] = dataFrames_movies['title'].apply(lower_case)

print("running fetch generes")
dataFrames_movies['genres'] = dataFrames_movies['genres'].apply(fetch_genres)

print("running fetch labels")
dataFrames_movies['keywords'] = dataFrames_movies['keywords'].apply(fetch_keyword)

print("running fetch actor")
dataFrames_movies["cast"] = dataFrames_movies["cast"].apply(fetch_cast)

print("running fetch unit")
dataFrames_movies["crew"] = dataFrames_movies["crew"].apply(fetch_crew)


# function to split the summary into a list
def splitting_of_string(s):
    return s.split()


# splitting of summary from string to list
dataFrames_movies["overview"] = dataFrames_movies["overview"].apply(splitting_of_string)


# function to remove the back spaces.
def remove_spaces(x):
    new_list = []
    for i in x:
        new_list.append(i.replace(" ", ""))
    return new_list


# removing space from each of the generes so that the generes appear as one
dataFrames_movies['genres'] = dataFrames_movies['genres'].apply(remove_spaces)
print("removing whitespaces from genere")

# removing spaces from labels so that the labels appear as one if their is a space in between
dataFrames_movies['keywords'] = dataFrames_movies['keywords'].apply(remove_spaces)
print("removing whitespaces from labels")

# removing spaces from actor names otherwise actor names won't come as one name if their is any space
dataFrames_movies['cast'] = dataFrames_movies['cast'].apply(remove_spaces)
print("removing whitespaces from actors")

# removing space from each of the unit person's name because of the same logic
dataFrames_movies['crew'] = dataFrames_movies['crew'].apply(remove_spaces)
print("removing whitespaces from unit")


dataFrames_movies['tags'] = dataFrames_movies['genres'] + dataFrames_movies['keywords'] + dataFrames_movies["cast"] + dataFrames_movies['crew'] + dataFrames_movies["overview"]

final_movie_data = dataFrames_movies[['movie_id', 'title', 'tags']]


# function to join a list to string.
# tags are lists before this.
# now this function will convert the tag from a list to a string.
# we can further do stemming of the tags
def list_to_string(x):
    return " ".join(x)


print("converting tags from a list to a string")
final_movie_data.loc[:, 'tags'] = final_movie_data["tags"].apply(list_to_string)


# funtion for stemming
def stemming(x):
    stemmed_words = []
    for i in x.split():
        stemmed_words.append(PorterStemmer().stem(i))
    return " ".join(stemmed_words)


final_movie_data.loc[:, 'tags'] = final_movie_data['tags'].apply(stemming)

vectors = CountVectorizer(stop_words="english").fit_transform(final_movie_data['tags']).toarray()


array_of_similarity_values_for_all_movies = cosine_similarity(vectors)

# saving the cosine similarity values. this is in the form of array.
# we will use it in the gui.
numpy.save("array_of_similarity_values_for_all_movies.npy", array_of_similarity_values_for_all_movies)
print("saving or loading the similarity values")

# saving the final dataframe.
# we will use it in the gui.
final_movie_data.to_pickle('final_movie_data.pkl')
print("saving or loading the final dataframe")

print("Done..")
#in finalproject
