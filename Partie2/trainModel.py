"""
Entrainement de NASNetMobile apres avoir entrainé la FC
7.a Après  avoir  entrainer  la  nouvelle  FC,  Geler  la  première  couche  et  faire  un  fine-tuning sur les couches supérieures. 
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from imutils import paths
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

img_rows, img_cols = 224, 224
channels = 3
BASE_PATH = "AFF11"
#BASE_PATH = "AFF11_light"
CLASSES = ["ash", "beech", "blackpine", "fir", "hornbeam", "larch", "mountainoak", "scotspine", "spruce", "swissstonepine", "sycamoremaple"]
TRAIN = '_train'
TEST = '_test'
BATCH_SIZE = 8

# repertoire de sauvegarde du modele apres entrainement
MODEL_ALL_PATH = os.path.sep.join(["NASNetMobile", "NASNetMobile.model"])

# graphe d'historique d'entrainement
TRAIN_ALL_PLOT_PATH = os.path.sep.join(["NASNetMobile", "train.png"])

# chemins vers les repertoires train, val et test
trainPath = os.path.sep.join(['..', BASE_PATH, BASE_PATH+TRAIN])
testPath = os.path.sep.join(['..', BASE_PATH, BASE_PATH+TEST])

# nbr total des image dans chacun des repo train test
totalTrain = len(list(paths.list_images(trainPath)))
totalTest  = len(list(paths.list_images(testPath)))

# instancier un objet ImageDataGenerator pour l'augmentation des donnees train
# 1. Appliquer une augmentation des données de train, avec un flip horizontal.
trainAug = ImageDataGenerator(horizontal_flip=True)
# instancier un objet ImageDataGenerator pour l'augmentation des donnees test
testAug = ImageDataGenerator()


# 2. Les images en entrée vont être normalisées par rapport à la moyenne des plans RGB des images de ImageNet. 
# definir la moyenne des images ImageNet par plan RGB pour normaliser les images de la base Food-11
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
testAug.mean = mean

#3. Charger un model pré-appris sur ImageNet sans la dernière couche FC.
#print("[INFO] Chargement de NASNetMobile...")
#baseModel = NASNetMobile(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
#print("[INFO] Chargement fini...")
# FCmodel = NASNetMobile(weights="imagenet", include_top=True, input_tensor=Input(shape=(224, 224, 3)))

#4. Faire un model.summary() avec et sans la dernière couche afin de voir la couche à
#redéfinir (un globalaveragepooling, un dropout, un flatten, etc) ceci diffère d’un modèle
#à l’autre.

#print(baseModel.summary())
#print('############################################################################################################')
#print(FCmodel.summary())


# recherche du batchsize "optimal"
epochs = 20
batchs_size = [128, 64, 32]

for batch_size in batchs_size:

    # initialisation des generateurs
    trainGen = trainAug.flow_from_directory(trainPath, class_mode="categorical", target_size=(224, 224), color_mode="rgb", shuffle=True,  batch_size=batch_size)
    testGen  = testAug.flow_from_directory(testPath,   class_mode="categorical", target_size=(224, 224), color_mode="rgb", shuffle=False, batch_size=batch_size)
    
    # chargement du model avec la FC entrainé sur la nouvelle base
    print("[INFO] chargement du model avec la FC entrainé sur la nouvelle base...")
    model = load_model('Trained_FC\\FCtrain.model')
    print("[INFO] chargement fini.")
    
    # degelé toutes les couches sauf la premiere
    for layer in model.layers[1:]:
        layer.trainable = True
    
    # compiler le nouveau model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    print("[INFO] training avec un batch_size de {}...".format(batch_size))
    H = model.fit_generator(trainGen, steps_per_epoch=totalTrain // batch_size, validation_data=testGen, validation_steps=totalTest // batch_size, epochs=epochs)
    
    print("[INFO] creation du plot d'accuracy...")
    fig = plt.figure(figsize=(10, 5))  
    plt.title('{} epochs {} batch size'.format(epochs, batch_size))
    plt.plot(H.history["val_accuracy"], label="test accuracy")
    plt.plot(H.history["accuracy"],     label="train accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.legend()
    fig.savefig('NASNetMobile_{}_epoch_{}_batchsize.jpg'.format(epochs, batch_size), bbox_inches='tight', dpi=150)

"""
# une fois le batch size trouvé ont le test sur 100 epoch
"""

"""
epochs = 100
batch_size = 32

# initialiser les generateurs
trainGen = trainAug.flow_from_directory(trainPath, class_mode="categorical", target_size=(224, 224), color_mode="rgb", shuffle=True,  batch_size=batch_size)
testGen  = testAug.flow_from_directory(testPath,   class_mode="categorical", target_size=(224, 224), color_mode="rgb", shuffle=False, batch_size=batch_size)

# Redefinition de la FC
#5.Définir une nouvelle couche FC identique à l’ancienne (il faut respecter la structure
#originale) mais cette fois ci initialisée aléatoirement. Le nombre de neurones en sortie
#est égal au nombre de classes de la base à traiter, le taux d’apprentissage doit être une
#valeur assez faible de 0.001.
headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(len(CLASSES), activation="softmax")(headModel)

#6. Reconstruire le nouveau modèle
model = Model(inputs=baseModel.input, outputs=headModel)    
#7.Faire un fine-tuning sur la nouvelle FC pour un certain nombre d’epoch, afin
#d’apprendre des poids adaptés à la nouvelle base. Ici les autres couches sont à priori
#toutes gelées.
for layer in baseModel.layers:
    layer.trainable = False
    
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
print("[INFO] training...")
H = model.fit_generator(trainGen, steps_per_epoch=totalTrain // batch_size, validation_data=testGen, validation_steps=totalTest // batch_size, epochs=epochs)

# fonction qui dessine les graphes de l'historique d'entrainement
def plot_training(H, N, plotPath):
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath, dpi=300)
 
# plot training
plot_training(H, epochs, TRAIN_ALL_PLOT_PATH)

# serialiser le modèle (sauvegarde sur le disque)
print("[INFO] serializing network...")
model.save(MODEL_ALL_PATH)

# tester le modèle
print("[INFO] evaluating after fine-tuning network...")
predIdxs = model.predict_generator(testGen, steps=(totalTest // BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)

# afficher les performances de classification
print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = 'partie2_history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv: 
hist_csv_file = 'partie2_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
"""