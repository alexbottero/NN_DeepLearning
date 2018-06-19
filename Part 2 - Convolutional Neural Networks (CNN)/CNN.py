#Part 1: Building CNN

#Import Modules
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Int CNN
classifier=Sequential()

#Step 1: Convolution
classifier.add(Convolution2D(filters=32,kernel_size=3,strides=1,input_shape=(64,64,3),activation="relu"))

#Step 2: Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#add convoultion layer
classifier.add(Convolution2D(filters=32,kernel_size=3,strides=1,activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3:Flattenning
classifier.add(Flatten())

#Step 4:fully completed layer 
classifier.add(Dense(units=128,activation="relu"))
classifier.add(Dense(units=1,activation="sigmoid"))
# ifmore than 2 categories use SoftMax

#Compilation
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#if more categorical_crossentropy

#Train CNN on image
#avoid overtraining
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        #uptate weight
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        #8000/32
        steps_per_epoch=250,
        epochs=25,
        validation_data=test_set,
        #2000/32 
        validation_steps=63)

#cat or dog
import numpy as np
from keras.preprocessing import image

test_image=image.load_img("dataset/single_prediction/cat_or_dog_2.jpg",target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)


result=classifier.predict(test_image)
if result[0][0]==1:
    print("it's a dog")
else:
    print("it's a cat")