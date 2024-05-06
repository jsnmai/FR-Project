import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential,load_model,save_model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np


#################################################################
def data_gen():
    train_datagen = ImageDataGenerator(rescale = 1./255,
      rotation_range=25,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

#################################################################

    batch_size = 64
    target_size = (64, 64)
    input_shape=(64, 64, 3)
    seed=1337
    adam = 0.001
    fre= -20
    FC = 2048
    E = 1
    patience = 3
    verbose = 1
    factor = 0.50
    min_lr = 0.0001
    steps_per_epoch=256
    validation_steps=256
    epochs=8

    test_datagen = ImageDataGenerator( rescale = 1.0/255)

    train_generator = train_datagen.flow_from_directory('../Dataset/Train',
                                                        batch_size =batch_size ,
                                                        class_mode = 'binary',
                                                        seed=seed,
                                                        target_size = target_size )     

    validation_generator =  test_datagen.flow_from_directory('../Dataset/Validation',
                                                            batch_size  = batch_size,
                                                            class_mode  = 'binary',
                                                            seed=seed,
                                                            target_size = target_size)

    #################################################################

    base_model = tf.keras.applications.VGG16(input_shape=input_shape,include_top=False,weights="imagenet")

    #################################################################

    for layer in base_model.layers[:fre]:
        layer.trainable=False

#################################################################

    model=Sequential()
    model.add(base_model)
    model.add(layers.Dropout(.2))

    model.add(Conv2D(512, (3, 3),strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Dropout(.1))
    model.add(Conv2D(128, (3, 3),strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Dropout(.1))
    model.add(Conv2D(384, (3, 3),strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Dropout(.1))
    model.add(Conv2D(384, (3, 3),strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Dropout(.1))
    model.add(Conv2D(500, (3, 3),strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,strides=(2,2), padding='same'))

    # Add new layers
    model.add(Flatten())
    model.add(Dense(FC , activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(Dense(FC , activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(Dense(FC, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(Dense(E, activation='sigmoid'))

    model.summary()

#################################################################

    model.compile(optimizer=Adam(adam),
                loss='binary_crossentropy'
                ,metrics=['accuracy'])

#################################################################

    lrd = ReduceLROnPlateau(monitor = 'val_loss',
                            patience = patience,
                            verbose = verbose ,
                            factor = factor,
                            min_lr = min_lr)



    es = EarlyStopping(verbose=verbose, patience=patience)
    return model

#################################################################


def classify(model, imagepath):
    imge = image.load_img(imagepath, target_size=(64, 64))
    X = image.img_to_array(imge)
    X = np.expand_dims(X, axis=0)

    images = np.vstack([X])
    classes = model.predict(images, batch_size=1)
    print(classes[0])
    if classes[0]<0.5:
        return "male"
    else:
        return "female"

def plot_data(male_count, female_coumt):
    male_percent = male_count / (male_count + female_coumt) * 100
    female_percent = female_coumt / (male_count + female_coumt) * 100

    labels = ['Classifications']
    fig, graph = plt.subplots(figsize=(10, 2))  
    graph.barh(labels, [male_percent], label='Male', color='blue')
    graph.barh(labels, [female_percent], left=[male_percent], label='Female', color='pink')
    
    graph.set_xlabel('Percentage (%)')
    graph.set_title('Gender Classification Percentages')
    plt.xlim(0, 100)
    graph.legend()

    plt.show()

if __name__ == "__main__" :
    model = data_gen()
    paths = ['Dataset/Validation/Male', 'Dataset/Validation/Female']
    male_count = 0
    female_count = 0

    for path in paths:
        image_count = 0
        for file in os.listdir(path):
            if image_count >= 50:  
                break
            image_path = os.path.join(path, file)
            print(image_path)
            result = classify(model, image_path)
            if result == 'male':
                male_count += 1
            if result == 'female':
                female_count += 1
            image_count += 1

    plot_data(male_count, female_count)
