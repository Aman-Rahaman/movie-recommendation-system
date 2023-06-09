from tkinter import *
import numpy
import pandas as pd
import requests

from urllib.request import urlopen
from PIL import Image, ImageTk
import io

array_of_similarity_values_for_all_movies = numpy.load("array_of_similarity_values_for_all_movies.npy")
final_movie_data = pd.read_pickle('final_movie_data.pkl')

window = Tk()
window.title("Movie recommendation system")
window.geometry("1000x500")
window['background'] = 'pink'


def fetch_poster(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=d71c21a739efb3a0137279c4d08c7612&language=en-US".format(movie_id))
    data = response.json()
    if data["poster_path"] is None:
        return None
    else:
        return "http://image.tmdb.org/t/p/w500/" + data["poster_path"]


def recommend(x):
    if x in final_movie_data['movie_name'].values:
        index = final_movie_data[final_movie_data['movie_name'] == x].index[0]
        similarity_values_for_the_movie = array_of_similarity_values_for_all_movies[index]
        list_of_similar_movies = sorted(list(enumerate(similarity_values_for_the_movie)), reverse=True,
                                        key=lambda x: x[1])[1:6]

        RecommendedMovies = []
        RecommendedMoviesPoster = []
        for i in list_of_similar_movies:
            movie_id = final_movie_data.iloc[i[0], 0]
            RecommendedMovies.append(final_movie_data.iloc[i[0], 1].title())  # here title is a string function
            RecommendedMoviesPoster.append(fetch_poster(movie_id))

        return RecommendedMovies, RecommendedMoviesPoster
    else:
        return False, False


def search():
    movie = input_movie_name.get().strip().lower()

    if movie == "":
        existance_status_label.config(text="Please Enter A Movie Name")
        return

    RecommendedMovies, RecommendedMoviesPoster = recommend(movie)

    if RecommendedMovies is False:
        existance_status_label.config(text="Entered movie not found in the data")

    else:
        existance_status_label.config(text="")
        poster_labels = [poster1, poster2, poster3, poster4, poster5]
        movie_name_labels = [name1, name2, name3, name4, name5]
        for i in range(len(RecommendedMovies)):

            movie_name = RecommendedMovies[i]
            movie_poster_url = RecommendedMoviesPoster[i]

            if movie_poster_url is None:
                poster_labels[i].config(image=img)
                poster_labels[i].image = img
            else:
                u = urlopen(movie_poster_url)
                raw_data = u.read()
                u.close()

                im = Image.open(io.BytesIO(raw_data)).resize((150, 250))

                poster = ImageTk.PhotoImage(im)

                poster_labels[i].config(image=poster)
                poster_labels[i].image = poster

            movie_name_labels[i].config(text=movie_name)


upper_frame = LabelFrame(window, text="Enter Movie Name", background='pink')
upper_frame.pack(pady=25)

input_movie_name = StringVar()

movie_input = Entry(upper_frame, font=("Helvetica", 30), textvariable=input_movie_name)
movie_input.grid(row=0, column=0, padx=10, pady=10)

button = Button(upper_frame, text="Recommend", command=search)
button.grid(row=0, column=1, padx=10)

lower_frame = LabelFrame(window, height=300, width=800)
lower_frame.pack(pady=25)

img = Image.open('grey.jpeg')
img = img.resize((150, 250))
img = ImageTk.PhotoImage(img)

# labels for posters
poster1 = Label(lower_frame, image=img)
poster1.grid(row=0, column=0)

poster2 = Label(lower_frame, image=img)
poster2.grid(row=0, column=1)

poster3 = Label(lower_frame, image=img)
poster3.grid(row=0, column=2)

poster4 = Label(lower_frame, image=img)
poster4.grid(row=0, column=3)

poster5 = Label(lower_frame, image=img)
poster5.grid(row=0, column=4)

# labels for movie name
name1 = Label(lower_frame, text='poster 1')
name1.grid(row=1, column=0)

name2 = Label(lower_frame, text='poster 2')
name2.grid(row=1, column=1)

name3 = Label(lower_frame, text='poster 3')
name3.grid(row=1, column=2)

name4 = Label(lower_frame, text='poster 4')
name4.grid(row=1, column=3)

name5 = Label(lower_frame, text='poster 5')
name5.grid(row=1, column=4)

# label for status bar
existance_status_label = Label(window, text="", background="pink")
existance_status_label.pack()

# text_box = Text(window, height=20, width=70, wrap=WORD)
# text_box.pack(pady=10)

window.mainloop()

# api_id = d71c21a739efb3a0137279c4d08c7612
# image path = https://api.themoviedb.org/3/movie/{movie_id}?api_key=<<api_key>>&language=en-US
