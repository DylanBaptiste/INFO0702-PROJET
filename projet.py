import matplotlib

matplotlib.use("Agg")
import numpy as np 
import pandas as pd 
import re
import os
import nltk 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.models import Sequential
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.constraints import max_norm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

from keras.layers import Dense, Dropout, Conv2D, InputLayer, Input, MaxPooling2D, Flatten, Concatenate, GlobalAveragePooling2D

from sklearn.model_selection import train_test_split, GridSearchCV
from keras.constraints import maxnorm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from imutils import paths
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

# 4. La fonction d’erreur est : loss = ‘categorical-crossentropy’. Ok
# 5.La méthode d’optimisation est ‘Adam’                           OK
# 6.Taux d’apprentissage initial : lr=0.001                        Ca se defini où ?
# 7.activation='relu', strides=2, padding='same'                   OK

img_rows, img_cols = 224, 224
channels = 3
BASE_PATH = "AFF11"
CLASSES = ["ash", "beech", "blackpine", "fir", "hornbeam", "larch", "mountainoak", "scotspine", "spruce", "swissstonepine", "sycamoremaple"]
TRAIN = '_train'
TEST = '_test'
BATCH_SIZE = 1

def miniSqueezeNet(activation='relu', init_mode='uniform', loss='categorical_crossentropy', optimizer='adam', padding='same'):
    
    input_1 = Input((img_rows, img_cols, channels))
    conv1 = Conv2D(96, kernel_size=7, activation='relu', strides=2, padding='same')(input_1)
    maxpool1 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(conv1)

    # 1er Fire Module
    fire1_squeeze = Conv2D(16, kernel_size=1, activation='relu', padding='same')(maxpool1)
    fire1_expand1 = Conv2D(64, kernel_size=1, activation='relu', padding='same')(fire1_squeeze)
    fire1_expand2 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(fire1_squeeze)
    concatenate_1 = Concatenate(axis=3)([fire1_expand1, fire1_expand2])
    
    # 2eme Fire Module
    fire2_squeeze = Conv2D(16, kernel_size=1, activation='relu', padding='same')(concatenate_1)
    fire2_expand1 = Conv2D(64, kernel_size=1, activation='relu', padding='same')(fire2_squeeze)
    fire2_expand2 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(fire2_squeeze)
    concatenate_2 = Concatenate()([fire2_expand1, fire2_expand2])
    
    # 3eme Fire Module
    fire3_squeeze = Conv2D(16,  kernel_size=1, activation='relu', padding='same')(concatenate_2)
    fire3_expand1 = Conv2D(128, kernel_size=1, activation='relu', padding='same')(fire3_squeeze)
    fire3_expand2 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(fire3_squeeze)
    concatenate_3 = Concatenate()([fire3_expand1, fire3_expand2])
    
    # max pooling
    maxpool2 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(concatenate_3)
    
    # 4eme Fire Module
    fire4_squeeze = Conv2D(16,  kernel_size=1, activation='relu', padding='same')(maxpool2)
    fire4_expand1 = Conv2D(128, kernel_size=1, activation='relu', padding='same')(fire4_squeeze)
    fire4_expand2 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(fire4_squeeze)
    concatenate_4 = Concatenate()([fire4_expand1, fire4_expand2])
    
    # FC
    dropout1 = Dropout(0.5)(concatenate_4)
    conv2 = Conv2D(len(CLASSES), kernel_size=1, activation='relu', padding='valid')(dropout1)
    global_average_pooling2d_1 = GlobalAveragePooling2D()(conv2)
    output = Dense(len(CLASSES), activation='softmax')(global_average_pooling2d_1)
    
    model = Model(inputs=input_1, outputs=output)
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model



# chemins vers les repertoires train, val et test
trainPath = os.path.sep.join([BASE_PATH, BASE_PATH+TRAIN])
testPath = os.path.sep.join([BASE_PATH, BASE_PATH+TEST])

# nbr total des image dans chacun des repo train test
totalTrain = len(list(paths.list_images(trainPath)))
totalTest  = len(list(paths.list_images(testPath)))

# instancier un objet ImageDataGenerator pour l'augmentation des donnees train
# 3. Appliquer une augmentation des données de train, avec un flip horizontal.
trainAug = ImageDataGenerator(horizontal_flip=True)
# instancier un objet ImageDataGenerator pour l'augmentation des donnees test
testAug = ImageDataGenerator()


# 2. Les images en entrée vont être normalisées par rapport à la moyenne des plans RGB des images de ImageNet. 
# definir la moyenne des images ImageNet par plan RGB pour normaliser les images de la base Food-11
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
testAug.mean = mean

# initialiser le generateur de train
trainGen = trainAug.flow_from_directory(
    trainPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True,
    batch_size=BATCH_SIZE)

# initialiser le generateur de test
testGen = testAug.flow_from_directory(
    testPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE)

model = miniSqueezeNet()
print("[INFO] training...")
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BATCH_SIZE,
    validation_data=testGen,
    validation_steps=totalTest // BATCH_SIZE,
    epochs=2)


