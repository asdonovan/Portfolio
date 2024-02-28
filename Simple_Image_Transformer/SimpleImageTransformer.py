# -*- coding: utf-8 -*-
"""
@author: Alec Donovan

Goal: The goal of this project was to gain experience creating a GUI using TKinter and create a simple application
that I am able to use to generate variations of images for image recognition projects.

Description: When the application is ran, a GUI would open that would allow a user to upload an image and be able to modify
the image using simple transformations such as rotation and reflection including various combinations of these
transformations. The user is also able to save these images for future use.
"""

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

"""
 --- Functions ---
"""
# Used to update image shown in left panel
def transformImage(image, transformations):
    # Transformation Functions
    rotate = lambda img: img.rotate(90)
    reflectionLR = lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)
    reflectionTB = lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)
    
    transformationFunctions = (rotate, reflectionLR, reflectionTB)
    
    # Dictate which transformations are applied to image based on checkboxes
    for index, t in enumerate(transformations):
        if t.get() == 1:
            image = transformationFunctions[index](image)

    return image


# Destroy all buttons passed in parameter
def destroyButtons(buttonArray):
    for button in buttonArray:
        if button is not None:
            button.destroy()


# Update Image on left frame
def updateImage(master, image, transformations=None):
    global img
    global b2
    
    # Prevents image from being updated when image button does not exist
    if len(master.children) < 2:
        print('No image to transform')
        return
    
    try:
        b2.destroy() # To prevent overlapping buttons
    except NameError:
        pass # Do nothing if b2 has not been created yet
        
    if transformations is not None:
        image = transformImage(image, transformations)
    resized_image = image.resize((150, 150))
    img = ImageTk.PhotoImage(resized_image)
    
    b2 = Button(master, image=img, command=lambda:destroyButtons([b2, saveImageButton])) # Image is placed in a button. When click, button is deleted
    b2.grid(column=0, row=1)

    saveImageButton.configure(command=lambda:saveImage(image)) # Allows the save button to save current version of image


# Used to open file explorer when 'upload image' button is pressed
def openFile(master):
    global saveImageButton
    global image
    
    f_types = [('Jpg Files', '*.jpg')]
    filePath = filedialog.askopenfilename(filetypes=f_types)
    if filePath is not None:
        image = Image.open(filePath)
        
        # Save Image Button is created
        saveImageButton = Button(master, text="Save Image", width=15)
        saveImageButton.grid(column=0, row=2, pady=10,)
        
        updateImage(master, image)
    else:
        print('Error Uploading File')
        

# Used to save currently dispalyed image
def saveImage(img):
    fileName = filedialog.asksaveasfile(mode='w', defaultextension='.jpg')
    if not fileName:
        print('Unable to save Image')
        return
    img.save(fileName)



"""
 --- Main ---
"""
# Create main window for application
root = Tk()

# Paned Window -> Main panel
pw = ttk.PanedWindow(root, orient=HORIZONTAL)
pw.pack(fill=BOTH, expand=True)
frame1 = ttk.Frame(pw, relief=SUNKEN)
frame1.columnconfigure(0, weight=1)
frame2 = ttk.Frame(pw, relief=SUNKEN)
frame2.columnconfigure(0, weight=1)

# Frame 1
# Upload Image Button
uploadImageButton = Button(frame1, text="Upload Image", width=15, command=lambda:openFile(frame1))
uploadImageButton.grid(column=0, row=0, pady=10,)

# Add frame to main panel
pw.add(frame1, weight=1)

# Frame 2
# Add Edit Option Label
transformOptionsLabel = Label(frame2, text='Transform Options',  font=("Arial", 10))
transformOptionsLabel.pack(side=TOP, pady=10)

# Add Rotation Option
checkButtonVal1 = IntVar()
checkbox1 = tk.Checkbutton(frame2, text="Rotate 90", variable=checkButtonVal1, font=("Arial", 10))
checkbox1.pack(pady=5)

# Add Reflection Left/Right Option
checkButtonVal2 = IntVar()
checkbox2 = tk.Checkbutton(frame2, text="Reflect Left/Right", variable=checkButtonVal2, font=("Arial", 10))
checkbox2.pack(pady=5)

# Add Reflection Top/Bottom Option
checkButtonVal3 = IntVar()
checkbox3 = tk.Checkbutton(frame2, text="Reflect Top/Bottom", variable=checkButtonVal3, font=("Arial", 10))
checkbox3.pack(pady=5)

# Add Transform Button
checkButtons = [checkButtonVal1, checkButtonVal2, checkButtonVal3]
transformButton = Button(frame2, text="Transform", command=lambda:updateImage(frame1, image, checkButtons))
transformButton.pack(side=BOTTOM, pady=5)

pw.add(frame2,weight=1)


#Resize Window and edit title
root.geometry("600x250")
root.title("Image Editor")

# Keep window open
root.mainloop()

