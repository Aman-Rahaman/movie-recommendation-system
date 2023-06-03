import numpy
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

dataFrames1 = pd.read_csv("data1.csv")
print("open first csv file")
# This dataset1.csv file contains the TMDB_id, actors names and the other people in the unit.
# TMDB is similar to IMDB. TMDB = 'The Movie Database'

dataFrames2 = pd.read_csv("data2.csv")
print("open second csv file")
# This dataset2.csv file contains a lot of data with respect to each movie.
# Most of the attributes in this are not in much use of use.
# Therefore we will only take some selective attributes.

dataFrames_films = dataFrames1.merge(dataFrames2, on="movie_name")
print("merging the frames")
# combining movies and credits on the basis of the movie movie_name name.
# now attributes from both the files are in one dataframe.


dataFrames_films = dataFrames_films[['TMDB_id', 'movie_name', 'summary', 'genres', 'labels', 'actors', 'unit']]
print("selecting attributes")
# selecting particular attributes from the dataset which are usefull for making tags.
# selecting only those attributes which are helpful in making the tag.


dataFrames_films.dropna(inplace=True)
# removing the rows which contain null values in any of their attributes
# this is because these attributes are important to make the tags.
# if for a movie we have null values in any of the attributes, the the system will show wrong result for that movie.


# this is a function to convert all the movie_names into lower case.
# this will help in searching at the end.
def lower_case(x):
    return x.lower()


# This is a function to fetch the actors return a list.
# Here we are using ast.literal_eval because of the format in which the labels are in the dataframe.
# The labels are in the form of dictionaries in a list but the whole thing is a string.
# that's why we are using ast.literal_eval
def fetch_actors(x):
    actors = []
    number_of_actors = 0
    for actor in ast.literal_eval(x):
        if number_of_actors <= 5:
            actors.append(actor['name'])
            number_of_actors += 1
        else:
            break
    return actors


# This is the function to fetch the labels of a particular film from the dataframe and return a list.
# even here we are using ast.literal_eval
def fetch_labels(x):
    labels = []
    for label in ast.literal_eval(x):
        labels.append(label['name'])
    return labels


# function to fetch the generes from the dataset and return a list.
# even here we are using ast.literal_eval
def fetch_genres(x):
    generes = []
    for i in ast.literal_eval(x):
        generes.append(i['name'])
    return generes


# function to fetch the people behind the unit from the dataset and return a list.
# even here we are using ast.literal_eval
def fetch_unit(x):
    unit = []
    for i in ast.literal_eval(x):
        if i['job'] == 'Director':
            unit.append(i['name'])
            break
    return unit


print("movie_name to lower case")
dataFrames_films['movie_name'] = dataFrames_films['movie_name'].apply(lower_case)

print("running fetch actor")
dataFrames_films["actors"] = dataFrames_films["actors"].apply(fetch_actors)

print("running fetch labels")
dataFrames_films['labels'] = dataFrames_films['labels'].apply(fetch_labels)

print("running fetch generes")
dataFrames_films['genres'] = dataFrames_films['genres'].apply(fetch_genres)

print("running fetch unit")
dataFrames_films["unit"] = dataFrames_films["unit"].apply(fetch_unit)


# function to split the summary into a list
def splitting_of_string(s):
    return s.split()


# splitting of summary from string to list
dataFrames_films["summary"] = dataFrames_films["summary"].apply(splitting_of_string)


# function to remove the back spaces.
def remove_spaces(x):
    new_list = []
    for i in x:
        new_list.append(i.replace(" ", ""))
    return new_list


# removing space from each of the generes so that the generes appear as one
dataFrames_films['genres'] = dataFrames_films['genres'].apply(remove_spaces)
print("removing whitespaces from genere")

# removing spaces from labels so that the labels appear as one if their is a space in between
dataFrames_films['labels'] = dataFrames_films['labels'].apply(remove_spaces)
print("removing whitespaces from labels")

# removing spaces from actor names otherwise actor names won't come as one name if their is any space
dataFrames_films['actors'] = dataFrames_films['actors'].apply(remove_spaces)
print("removing whitespaces from actors")

# removing space from each of the unit person's name because of the same logic
dataFrames_films['unit'] = dataFrames_films['unit'].apply(remove_spaces)
print("removing whitespaces from unit")


dataFrames_films['tags'] = dataFrames_films['genres'] + dataFrames_films['labels'] +\
                            dataFrames_films["actors"] + dataFrames_films['unit'] + dataFrames_films["summary"]

final_movie_data = dataFrames_films[['TMDB_id', 'movie_name', 'tags']]


# function to join a list to string.
# tags are lists before this.
# now this function will convert the tag from a list to a string.
# we can further do stemming of the tags
def list_to_string(x):
    return " ".join(x)


print("converting tags from a list to a string")
final_movie_data.loc[:, 'tags'] = final_movie_data["tags"].apply(list_to_string)


# function for stemming
def stemming(x):
    stemmed_words = []
    for i in x.split():
        stemmed_words.append(PorterStemmer().stem(i))
    return " ".join(stemmed_words)


print("stemming...")
final_movie_data.loc[:, 'tags'] = final_movie_data['tags'].apply(stemming)

print("vectorization...")
vectors = CountVectorizer(stop_words="english").fit_transform(final_movie_data['tags']).toarray()

print("finding cosine similarity values...")
array_of_similarity_values_for_all_movies = cosine_similarity(vectors)


# saving the cosine similarity values. this is in the form of array.
# we will use it in the gui.
print("saving or loading the similarity values")
numpy.save("array_of_similarity_values_for_all_movies.npy", array_of_similarity_values_for_all_movies)

# saving the final dataframe.
# we will use it in the gui.
print("saving or loading the final dataframe")
final_movie_data.to_pickle('final_movie_data.pkl')

print("Done...")
