import tensorflow as tf
from tensorflow import keras
import cv2 # Uses OpenCV to read images from a file
import imghdr # Used to identify image type
import os
import matplotlib.pyplot as plt # Python vizualtion library
import numpy as np # Array processing package
import math

class SimpleImageClassifier:
    """
    Author: Alec Donovan
    
    Description: The goal of this project was to create a class that creates a simple image classifier using tensor flow and keras. This project is by no means optimized
    and was used for me to practice and learn concepts related to using tensorflow, keras and other ML practices.
    
    Attributes:
        * labels -> List containing label names
        * data -> BatchDataset containing images and associated labels
        
    Methods:
        Public:
        * get_label_count() -> Get count of how many classes are in the image dataset
        * draw_label() -> Plot 1 image of each class with the title containing the label number and the class name
        * scale_data(scaling_function) -> Return a MapDataset containing the data set that was scaled using the scaling_function
        * split_data(data_to_split, split_size) -> Splits data into train, validation, and test data. Returns a tuple of these 3 sets
        * create_default_model() -> Creates and compiles a default (ideally all purpose) model
        * train_and_evaluate(model, train_set, validation_set, test_set, epochs) -> Trains the model and evaluates result using test set
        * evaluate_model(history, test_set) -> Evaluate performance of model using a test_set
        * test_single_image(img_dir) -> Test an image on the current model
        
        Private:
        * __clean_images(img_dir) -> Finds and remove any potentially problematic images from the dataset. Should only be used in the constructor
    """
    
    def __init__(self, img_dir, batch_size=32, even_image_quantity=False):
        self.labels = os.listdir(img_dir)
        self.__clean_images(img_dir, even_image_quantity)
        self.data = keras.utils.image_dataset_from_directory(img_dir, batch_size=batch_size, )
        data_iter = self.data.as_numpy_iterator()
        print(f'{len(self.data)} image batche(s) of size {len(data_iter.next()[0])} created')


    def get_label_count(self):
        return len(self.labels)

    
    def draw_labels(self):
        diction = {} # Used to keep track of which labels have been found
        figure, axis = plt.subplots(ncols=self.get_label_count(), figsize=(15,15)) # Used to plot images
        plot_filled = False # Used to dictate when to terminate iterations
        data_iter = self.data.as_numpy_iterator() # Used for iterating throug batches
        batch_count = 0 # Used to prevent data_iter from throwing an exception at the end of iteration
        
        while not plot_filled and batch_count <= len(self.data):
            batch = data_iter.next()
            for i in range(len(batch[0])): # For each image in batch
                image = batch[0][i]
                label = batch[1][i]
                if label not in diction: # If this image class has not been seen yet 
                    axis[label].imshow(image.astype(np.int64))
                    axis[label].title.set_text(f'Class {label} - {self.labels[label]}')
                    diction[label]  = False 
            if len(diction) == self.get_label_count(): # If the dictionary has the same length of labels, then we signal the loop to stop
                plot_filled = True
            else: # If there is still an image class we have not found yet, iterate to next batch
                batch_count += 1

    def scale_data(self, scaling_function=lambda x,y: (x/255.0, y)): # Use this default lambda if no function is given
        return self.data.map(scaling_function)

    
    def split_data(self, data_to_split, split_size = [80,10,10]):
        if len(data_to_split) < 3: # Ensure there are at least 1 batch for each split
            print('Not enough batches to make splits. At least 3 are needed. Currently there are', len(data_to_split))
            return (None, None, None)
        if sum(split_size) > 1: # Scale values if needed
            split_size = np.array(split_size) / 100.
        if sum(split_size) != 1.0: # Ensure any manual splits are valid 
            print('Split percentages do not add to 100%. No operation done')
            return (None, None, None)
        
        train_batch_size = math.floor(split_size[0] * len(data_to_split))
        validation_batch_size = math.ceil(split_size[1] * len(data_to_split))
        test_batch_size = math.ceil(split_size[2] * len(data_to_split))
        train = data_to_split.take(train_batch_size)
        validation = data_to_split.skip(train_batch_size).take(validation_batch_size) 
        test = data_to_split.skip(train_batch_size + validation_batch_size).take(test_batch_size) 
        
        print('Total Batches:', len(data_to_split))
        print('Train Batch Count:', train_batch_size)
        print('Validation Batch Count:', validation_batch_size)
        print('Test Batch Size:', test_batch_size)
        
        return train, validation, test

    
    def create_default_model(self):
        output_activation = 'softmax' # Better for multi class classifcation
        output_size = len(self.labels)
        loss_function = tf.losses.SparseCategoricalCrossentropy()
        if len(self.labels) == 2:
            output_size = 1
            output_activation = 'sigmoid' # Better for binary classification
            loss_function = tf.losses.BinaryCrossentropy()
            
        tf.keras.backend.clear_session() # Clear any saved models
            
        model = keras.models.Sequential([
        # Input Layer
        keras.layers.Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256,256,3)), # 16, 3 by 3 kernal with a stride of 1. Assuming that images are not grayscaled
        # Hidden Layers
        keras.layers.MaxPooling2D(), # Take max value in input window
        keras.layers.Conv2D(16, (3,3), 1, activation='relu'), # 16, 3 by 3 kernal with a stride of 1
        keras.layers.MaxPooling2D(), # Take max value in input window
        keras.layers.Conv2D(32, (3,3), 1, activation='relu'), # 16, 3 by 3 kernal with a stride of 1
        keras.layers.MaxPooling2D(), # Take max value in input window
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        # Output Layers
        keras.layers.Dense(output_size, activation=output_activation)
        ])
        
        model.compile('adam', loss=loss_function, metrics=['accuracy']) # Will only use adam optimizer for now
        return model

    
    def train_and_evaluate(self, model, train, validation, test, epochs=20):
        history = model.fit(train, epochs=epochs, validation_data=validation)
        
        fig = plt.figure() # Plot loss
        plt.plot(history.history['loss'], color='blue', label='loss') # Training loss = blue
        plt.plot(history.history['val_loss'], color='green', label='val_loss') # validation loss = green
        fig.suptitle('Loss')
        plt.legend(loc="upper right")
        plt.show()
        
        fig = plt.figure() # Plot accuracy
        plt.plot(history.history['accuracy'], color='blue', label='accuracy') # Training accuracy = blue
        plt.plot(history.history['val_accuracy'], color='green', label='val_accuracy') # validation accuracy = green
        fig.suptitle('Accuracy')
        plt.legend(loc="lower right")
        plt.show()
        
        return history
    
    
    def evaluate_model(self, model, test):
        lam = lambda x: np.argmax(tf.nn.softmax(x))
        i = 0

        for batch in test.as_numpy_iterator(): # For each test batch
            print(f'Batch {i}:')
            image, true_label = batch
            prediction_label = model.predict(image, verbose=0) # Predict label
            if len(self.labels) <= 3:
                prediction_label = np.round(prediction_label).flatten().astype(int)
            else:
                prediction_label = np.fromiter(map(lam, prediction_label), dtype=np.int_) # Get label with highest softmax 
            label_comparitor = [true_label[j] == prediction_label[j] for j in range(len(true_label))] # Compare predicted label vs target label
            i += 1

            print(true_label, '<- True labels')
            print(prediction_label, '<- Predicted labels')
            print('Incorrectly labeled images:', label_comparitor.count(False), end='\n\n')
            
            
    def test_single_image(self, model, img_dir):
        # This is an unoptimal way to do this step, but the goal was to practice using tensorflow datasets
        testing_images = keras.utils.image_dataset_from_directory(img_dir, shuffle=False)
        plt.imshow(testing_images.as_numpy_iterator().next()[0][0].astype(np.int64)) # diplay image being tested
        
        testing_images = testing_images.map(lambda x,y: (x/255.0, y)) # scale the image
        image = testing_images.as_numpy_iterator().next()[0]
        prediction = model.predict(image, verbose=0)
        
        if len(self.labels) > 2: # if there are more then 2 image classes, take softmax
            prediction_label = np.argmax(tf.nn.softmax(prediction[-1]))
        elif prediction < 0.5:
            prediction_label = 0
        else:
            prediction_label = 1
            
        plt.title('I predict... {}'.format(self.labels[prediction_label]))
        plt.axis('off')
        plt.show()
        
    
    def __clean_images(self, img_dir, even_image_quantity): # Private method. Should only need to be used on init
        count_invalid_type = 0 # Used to count how many files are removed due to invalid type
        count_exception = 0 # Used to cont how many files are removed due to exceptions reading the images
        images_removed = {} # Stores the path and type of removed images in a dictionary. {path : type}
        print('Removing bad or corrupted images...')
        for image_dir in os.listdir(img_dir): # For each directory in the image directory
            image_dir_path = img_dir + '/' + image_dir
            for image in os.listdir(image_dir_path): # For each image in image class
                image_path = image_dir_path + '/' + image
                try: # Attempt to open file and retrieve file type
                    image_check = cv2.imread(image_path) # Open image as a numpy array. See above
                    type_check = imghdr.what(image_path) # Obtain file extension
                    if type_check not in ['bmp', 'png', 'jpeg', 'jpg']: # If not a valid file extension
                        count_invalid_type += 1
                        images_removed[image_path] = type_check
                        os.remove(image_path)
                except Exception as e: # Catch any exceptions when trying to open and retrieve image extension
                    print('Exception Occured:', e)
                    count_exception += 1 
                    images_removed[image_path] = None
                    os.remove(image_path)

        print('Images removed due to invalid type:', count_invalid_type)
        print('Images removed due to issues opening image:', count_exception)
        print('Total removed images:', count_exception + count_invalid_type, end ='\n\n')
            
        for i in os.listdir(img_dir):
            print('{}: {} images'.format(i, len(os.listdir(img_dir + '/' + i))))
        print()