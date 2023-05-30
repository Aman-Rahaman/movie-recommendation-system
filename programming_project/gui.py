from tkinter import *

window = Tk()
window.title("Movie recommendation system")
window.geometry("600x500")
window['background'] = 'pink'

upper_frame = LabelFrame(window, text="Enter Movie Name", bg='pink')
upper_frame.pack(pady=25)

input_movie_name = StringVar()

movie_input = Entry( upper_frame, font=("Helvetica",30), textvariable=input_movie_name )
movie_input.grid(row=0, column=0, padx=10, pady=10)

button = Button(upper_frame, text="Recommend", command=search)
button.grid(row=0, column=1, padx=10)

lower_frame = LabelFrame(window,height=300,width=800)
lower_frame.pack(pady=25)

window.mainloop()
