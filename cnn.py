import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
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

batch_size = 6
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
steps_per_epoch=10
validation_steps=10
epochs=6

### Datagen ###
sf_train_datagen = ImageDataGenerator(rescale = 1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

sf_test_datagen = ImageDataGenerator( rescale = 1.0/255)

sf_train_generator = sf_train_datagen.flow_from_directory('Kaggle/Train',
                                                    batch_size =batch_size ,
                                                    class_mode = 'binary',
                                                    seed=seed,
                                                    target_size = target_size)

sf_validation_generator =  sf_test_datagen.flow_from_directory('Kaggle/Test',
                                                    batch_size  = batch_size,
                                                    class_mode  = 'binary',
                                                    seed=seed,
                                                    target_size = target_size)

sf_model = load_model("VGG16.h5")
sf_model.compile(optimizer=Adam(adam),
                loss='binary_crossentropy',
                metrics=['accuracy'])
# sf_model.summary()

ca_train_datagen = ImageDataGenerator(rescale = 1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

ca_test_datagen = ImageDataGenerator( rescale = 1.0/255)

ca_train_generator = ca_train_datagen.flow_from_directory('casia-images/casia_validation',
                                                batch_size = batch_size,
                                                class_mode = "binary",
                                                seed=seed,
                                                target_size = target_size )   
  
ca_validation_generator = ca_test_datagen.flow_from_directory('casia-images/casia_validation',
                                                batch_size = batch_size,
                                                class_mode = "binary",
                                                seed=seed,
                                                target_size = target_size )  

ca_model = load_model("VGG16.h5")
ca_model.compile(optimizer=Adam(adam),
                loss='binary_crossentropy',
                metrics=['accuracy'])
# casia_model.summary()

### Plotting Accuracies ###
def plot_accuracy(model, data):
    if data == "CASIA":
        validation_data=ca_validation_generator
        hist = model.fit(ca_train_generator,steps_per_epoch=steps_per_epoch,
                         validation_data=validation_data,
                         validation_steps=validation_steps,epochs=epochs)
    elif data == "SFace":
        validation_data=sf_validation_generator
        hist = model.fit(sf_train_generator,steps_per_epoch=steps_per_epoch,
                         validation_data=validation_data,
                         validation_steps=validation_steps,epochs=epochs)
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history['val_accuracy'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(["Accuracy","Validation Accuracy","Loss","Validation Loss"])
    plt.title(data + " classifier accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.show()
    return

plot_accuracy(sf_model, "SFace")
plot_accuracy(ca_model, "CASIA")

### Plotting Data ###
def plot_data(ca_male, ca_female, sf_male, sf_female):
    
    ca_male_per = ca_male / (ca_male + ca_female) * 100
    ca_female_per = ca_female / (ca_male + ca_female) * 100

    sf_male_per = sf_male / (sf_male + sf_female) * 100
    sf_female_per = sf_female / (sf_male + sf_female) * 100
    
    labels = ["SFace-60", "CASIA"]
    males = [sf_male_per, ca_male_per]
    females = [sf_female_per, ca_female_per]
    fig, graph = plt.subplots(layout='constrained')

    b1 = graph.barh(labels, males, label='Male', color='green')
    b2 = graph.barh(labels, females, left=males, label='Female', color='orange')

    graph.set_title('Gender Classification Percentages')
    graph.set_yticks(labels)
    graph.set_xlabel('Percentage (%)')
    plt.xlim(0, 100)
    graph.legend(loc='upper left', ncols=1)
    
    plt.show()

### Classifying datasets ###
def count(paths, data):
    male_count = 0
    female_count = 0

    if "casia" in data:
        model = ca_model
        max_img = 50 if data == "casia test" else 500
    elif "sf" in data:
        # model = sf_model
        model = ca_model
        max_img = 50 if data == "sf test" else 500
    for path in paths:
        image_count = 0
        for file in os.listdir(path):
            if file[:-4:-1] in ["gpj", "gnp"]:
                if image_count >= max_img:  
                    break
                image_path = os.path.join(path, file)
                print(image_path)
                img = image.load_img(image_path,target_size=target_size)
                img = np.asarray(img)
                # plt.imshow(img)
                img = np.expand_dims(img, axis=0)
                output = model.predict_on_batch(img)
                if output[0] > 0.5:
                    # print("female")
                    female_count += 1
                else:
                    # print("male")
                    male_count += 1
                image_count += 1
    return male_count, female_count

### Classifying CASIA and SFace-60 datasets of 50M/50F ###
casia_paths = ['casia-images/casia_validation/females','casia-images/casia_validation/males']
sface_paths = ['sfacesubset/train/female', 'sfacesubset/train/male']
ca_male, ca_female = count(casia_paths, "casia test")
sf_male, sf_female = count(sface_paths, "sf test")
plot_data(ca_male, ca_female, sf_male, sf_female)
print("casia: M-" + str(ca_male) + " F-" + str(ca_female))
print("sface: M-" + str(sf_male) + " F-" + str(sf_female))

### Classifying random sample of CASIA and SFace-60 images ###
casia_paths = ['casia-images/train']
sface_paths = ['sfacesubset/test']
ca_male, ca_female = count(casia_paths, "casia")
sf_male, sf_female = count(sface_paths, "sf")
plot_data(ca_male, ca_female, sf_male, sf_female)
print("casia: M-" + str(ca_male) + " F-" + str(ca_female))
print("sface: M-" + str(sf_male) + " F-" + str(sf_female))