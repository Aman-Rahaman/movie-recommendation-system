import numpy
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

dataFrames_movies1 = pd.read_csv("movie_dataset_1.csv")
dataFrames_movies2 = pd.read_csv("movie_dataset_2.csv")

dataFrames_movies = dataFrames_movies1.merge(dataFrames_movies2, on="title")
# combining movies and credits on the basis of the movie title name


dataFrames_movies = dataFrames_movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
# selecting particular attributes from the dataset which are usefull for making tags


dataFrames_movies.dropna(inplace=True)  # removing the rows which contain null values in any of their attributes


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


# def recommend(x):
#     index = final_movie_data[ final_movie_data['title'] == x ].index[0]
#     similarity_values_for_one_movie = array_of_similarity_values_for_all_movies [ index ]
#     list_of_similar_movies = sorted( list( enumerate( similarity_values_for_one_movie ) ), reverse=True, key=lambda x:x[1] )[1:6]
#     return list_of_similar_movies
#     # for i in list_of_similar_movies:
#     #     print( final_movie_data.iloc[ i[0],1 ].title() ) # here title is a string function
#
# #*****************************************************************************
#
# window = Tk()
# window.title("Movie recommendation system")
# window.geometry("600x500")
#
# def search():
#     list_of_recommended_movies = recommend( input_movie_name.get() )
#     for i in list_of_recommended_movies:
#         print( final_movie_data.iloc[ i[0],1 ].title() ) # here title is a string function
#
# upper_frame = LabelFrame(window, text="Enter Movie Name")
# upper_frame.pack(pady=25)
#
# input_movie_name = StringVar()
#
# movie_input = Entry( upper_frame, font=("Helvetica",30), textvariable=input_movie_name )
# movie_input.grid(row=0, column=0, padx=10, pady=10)
#
# button = Button(upper_frame, text="Recommend", command=search)
# button.grid(row=0, column=1, padx=10)
#
# text_box = Text(window, height=20, width=65, wrap=WORD)
# text_box.pack(pady=10)
#
# window.mainloop()
#
# #********************************************************************
#
#
# # while True:
# #     str = input("Enter a movie name : ").lower().strip()
# #     recommend(str)
# #     print("*"*30)
# #     print("Do you want to use it again ? ")
# #     ans = input("Enter 'no' to exit or enter 'yes' to use it again : ").strip().lower()
# #     if ans == 'no':
# #         break
#
#
#
# # NEED to check if the input movie name is present or not!!!!!
# # try to not use ast
# # .index[0] ??? in recommend()
