import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from imutils import paths
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.applications import NASNetMobile
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, InputLayer, Input, MaxPooling2D, Flatten, Concatenate, GlobalAveragePooling2D
from keras.optimizers import Adam

img_rows, img_cols = 224, 224
channels = 3
BASE_PATH = "AFF11"
#BASE_PATH = "AFF11_light"
CLASSES = ["ash", "beech", "blackpine", "fir", "hornbeam", "larch", "mountainoak", "scotspine", "spruce", "swissstonepine", "sycamoremaple"]
TRAIN = '_train'
TEST = '_test'
batch_size = 32
epochs = 20

# repertoire de sauvegarde du modele apres entrainement
MODEL_ALL_PATH = os.path.sep.join(["NASNetMobile", "NASNetMobile.model"])

# graphe d'historique d'entrainement
TRAIN_ALL_PLOT_PATH = os.path.sep.join(["NASNetMobile", "train.png"])


# chemins vers les repertoires train, val et test
trainPath = os.path.sep.join([BASE_PATH, BASE_PATH+TRAIN])
testPath = os.path.sep.join([BASE_PATH, BASE_PATH+TEST])

# nbr total des image dans chacun des repo train test
totalTrain = len(list(paths.list_images(trainPath)))
totalTest  = len(list(paths.list_images(testPath)))

# instancier un objet ImageDataGenerator pour l'augmentation des donnees train
# 1. Appliquer une augmentation des données de train, avec un flip horizontal.
trainAug = ImageDataGenerator(horizontal_flip=True)
# instancier un objet ImageDataGenerator pour l'augmentation des donnees test
testAug = ImageDataGenerator()


# initialiser les generateurs
trainGen = trainAug.flow_from_directory(trainPath, class_mode="categorical", target_size=(224, 224), color_mode="rgb", shuffle=True,  batch_size=batch_size)
testGen  = testAug.flow_from_directory(testPath,   class_mode="categorical", target_size=(224, 224), color_mode="rgb", shuffle=False, batch_size=batch_size)

# 2. Les images en entrée vont être normalisées par rapport à la moyenne des plans RGB des images de ImageNet. 
# definir la moyenne des images ImageNet par plan RGB pour normaliser les images de la base Food-11
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
testAug.mean = mean


# ENTRAINEMENT DE LA FC

#3. Charger un model pré-appris sur ImageNet sans la dernière couche FC.
print("[INFO] Chargement de NASNetMobile...")
baseModel = NASNetMobile(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
print("[INFO] Chargement fini...")

# Redefinition de la FC
#5.Définir une nouvelle couche FC identique à l’ancienne (il faut respecter la structure
#originale) mais cette fois ci initialisée aléatoirement. Le nombre de neurones en sortie
#est égal au nombre de classes de la base à traiter, le taux d’apprentissage doit être une
#valeur assez faible de 0.001.
headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(len(CLASSES), activation="softmax")(headModel)

#6. Reconstruire le nouveau modèle
# on empile la nouvelle FC sur la base
model = Model(inputs=baseModel.input, outputs=headModel)    

# geler (ou bien freeze) toute les couche basale du modèle, càd on ne va pas changer leur poids mais laisser les poids
# appris sur imagenet
for layer in baseModel.layers:
    layer.trainable = False
    
# visualiser les couches à entrainer
# normalement que la FC est entrainable
for layer in model.layers:
    print("{}: {}".format(layer, layer.trainable))
   
# compiler le nouveau model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# entrainer la couche FC pour 20 epoch (rappel: toutes les autres couches sont gelees donc leurs poids resteront
# inchangés
print("[INFO] training head...")
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // batch_size,
    validation_data=testGen,
    validation_steps=totalTest // batch_size,
    epochs=epochs)

model.save('NASNetMobile\\FCtrain.model')

"""
from keras.models import load_model

model = load_model('NASNetMobile\\FCtrain.model')

for layer in model.layers:
    print("{}: {}".format(layer, layer.trainable))
"""