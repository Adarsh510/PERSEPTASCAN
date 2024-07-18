import tkinter as tk
import subprocess
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk

# Create the main window
root = Tk()
root.title("PerceptaScan")

root.resizable(False, False)  # no_resize
root.iconbitmap("./title_icon.ico")  # title icon

# Set the window size
root.geometry("640x480")

# Load and display a background image
# bg_image = Image.open("bg.jpg")
# bg_photo = ImageTk.PhotoImage(bg_image)

# background_label = ttk.Label(root, image=bg_photo)
# background_label.place(relwidth=1, relheight=1)


# Create a label
# label1 = ttk.Label(root, text="Recognizer", font="Tahoma 20 bold underline")
# label1.pack()


# frame = tk.Frame(root)
# frame.pack()

# label = tk.Label(frame, width=500, height=500)
# label.pack()


# bg
root.configure(background='white')

# label
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
message = tk.Label(root, text="PerceptaScan", bg="light blue", fg="white", width=25, height=3,
                   font=('times', 30, 'bold'))
message.place(x=20, y=20)


# Main function call
def execute_python_file():
    subprocess.call(["python", "D:\Face_Integrated\Real-Time Object and Face Detection\main.py"])


# Function for closing window
def Close():
    root.destroy()


# Create a button for scan
button = Button(root, text="Scan", fg="white", bg="light blue",
                width=7, height=2, activebackground="Red",
                font=('times', 15, ' bold '), command=lambda: [execute_python_file(), Close()])
button.pack(pady=20)
button.place(x=150, y=300)

# Button for closing app
exit_button = Button(root, text="Exit", fg="white", bg="light blue",
                     width=7, height=2, activebackground="Red",
                     font=('times', 15, ' bold '), command=Close)
exit_button.pack(pady=20)
exit_button.place(x=400, y=300)

# Start the Tkinter main loop
root.mainloop()
