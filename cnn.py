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
    classes = model.predict(images, batch_size=8)
    print(classes[0])
    if classes[0]<0.5:
        return "male"
    else:
        return "female"
    # plt.imshow(imge)

def plot_average(datasets, m_avg, f_avg):
    genders = {
        'CASIA-Webface': (m_avg, f_avg),
        'SFace-60': (m_avg, f_avg)
    }

    x = np.arange(len(datasets))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in genders.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=2)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    fig.suptitle('Dataset Gender Distribution', fontsize=20)
    ax.set_ylabel('Percentage Identified')
    ax.set_title('Gender Attributes')
    ax.set_xticks(x + width, genders)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 100)

    plt.show()

if __name__ == "__main__" :
    males = 0
    females = 0
    model = data_gen()
    datasets = ("Baseline", "SFace-60")
    i = 0
    for subfolder in os.listdir("../Dataset/Test/"):
        subname = os.path.join("../Dataset/Test/", subfolder)
        for filename in os.listdir(subname):
            f = os.path.join(subname, filename)
            # checking if it is a file
            if os.path.isfile(f):
                if i < 50:
                    print(f)
                    gender = classify(model, f)
                    if gender == "male":
                        males += 1
                        i += 1
                    elif gender == "female":
                        females += 1
                        i += 1
                    pass
                elif i == 51:
                    print("done")
            else:
                print("not a good path to images")
                pass
    plot_average(datasets, males, females)
