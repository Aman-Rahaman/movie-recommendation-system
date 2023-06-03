from tkinter import *
import numpy
import pandas as pd
import requests
import customtkinter

from urllib.request import urlopen
from PIL import Image, ImageTk
import io

customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
from tkinter import *
import numpy
import pandas as pd
import requests
import customtkinter

from urllib.request import urlopen
from PIL import Image, ImageTk
import io

customtkinter.set_appearance_mode("Light")
customtkinter.set_default_color_theme("blue")


array_of_similarity_values_for_all_movies = numpy.load("array_of_similarity_values_for_all_movies.npy")
final_movie_data = pd.read_pickle('final_movie_data.pkl')

window = customtkinter.CTk()
window.title("Movie recommendation system")
window.geometry("1000x500")
response = requests.get("https://api.themoviedb.org/3/movie/285?api_key=d71c21a739efb3a0137279c4d08c7612&language=en-US")


def fetch_poster(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=d71c21a739efb3a0137279c4d08c7612&language=en-US".format(movie_id))
    data = response.json()
    if data["poster_path"] is None:
        return None
    else:
        return "http://image.tmdb.org/t/p/w500/" + data["poster_path"]


def recommend(x):
    if x in final_movie_data['title'].values:
        index = final_movie_data[final_movie_data['title'] == x].index[0]
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
    movie = movie_input.get()

    if movie == "":
        existance_status_label.configure(text="Please Enter A Movie Name")
        return

    RecommendedMovies, RecommendedMoviesPoster = recommend(movie)

    if RecommendedMovies is False:
        existance_status_label.configure(text="Entered movie not found in the data")

    else:
        existance_status_label.configure(text="")
        poster_labels = [poster1, poster2, poster3, poster4, poster5]
        movie_name_labels = [name1, name2, name3, name4, name5]
        for i in range(len(RecommendedMovies)):

            movie_name = RecommendedMovies[i]
            movie_poster_url = RecommendedMoviesPoster[i]

            if movie_poster_url is None:
                poster_labels[i].configure(image=img)
                poster_labels[i].image = img
            else:
                u = urlopen(movie_poster_url)
                raw_data = u.read()
                u.close()

                im = Image.open(io.BytesIO(raw_data)).resize((150, 250))

                poster = ImageTk.PhotoImage(im)

                poster_labels[i].configure(image=poster)
                poster_labels[i].image = poster

            movie_name_labels[i].configure(text=movie_name)


# first_label = customtkinter.CTkLabel(window, text="Movie Recommendation System")
# first_label.pack()

upper_frame = customtkinter.CTkFrame(window, corner_radius=10)
upper_frame.pack(pady=25)

# input_movie_name = StringVar()

movie_input = customtkinter.CTkEntry(upper_frame, width=400, height=40,
                                     placeholder_text="Enter a movie name", font=("High Tower", 20))
movie_input.grid(row=0, column=0, padx=10, pady=10)

button = customtkinter.CTkButton(upper_frame, text="Recommend me", command=search)
button.grid(row=0, column=1, padx=10)

lower_frame = customtkinter.CTkFrame(window, height=300, width=800, corner_radius=10)
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
name1 = customtkinter.CTkLabel(lower_frame, text='POSTER 1')
name1.grid(row=1, column=0)

name2 = customtkinter.CTkLabel(lower_frame, text='POSTER 2')
name2.grid(row=1, column=1)

name3 = customtkinter.CTkLabel(lower_frame, text='POSTER 3')
name3.grid(row=1, column=2)

name4 = customtkinter.CTkLabel(lower_frame, text='POSTER 4')
name4.grid(row=1, column=3)

name5 = customtkinter.CTkLabel(lower_frame, text='POSTER 5')
name5.grid(row=1, column=4)

# label for status bar
existance_status_label = customtkinter.CTkLabel(window, text="")
existance_status_label.pack()

# text_box = Text(window, height=20, width=70, wrap=WORD)
# text_box.pack(pady=10)

window.mainloop()

# api_id = d71c21a739efb3a0137279c4d08c7612
# image path = https://api.themoviedb.org/3/movie/{movie_id}?api_key=<<api_key>>&language=en-US
