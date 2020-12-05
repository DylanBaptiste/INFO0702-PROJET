import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import os
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator
from miniSqueezeNet import miniSqueezeNet

CLASSES = ["ash", "beech", "blackpine", "fir", "hornbeam", "larch", "mountainoak", "scotspine", "spruce", "swissstonepine", "sycamoremaple"]
TRAIN = '_train'
TEST = '_test'
img_rows, img_cols = 224, 224
channels = 3
input_shape = (img_rows, img_cols, channels)
BASE_NAME = "AFF11"

# chemins vers les repertoires train, val et test
trainPath = os.path.sep.join(['../' + BASE_NAME, BASE_NAME+TRAIN])
testPath = os.path.sep.join(['../'  + BASE_NAME, BASE_NAME+TEST])

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

"""
epochs = [20,40,60,80,100] 
batchs_size = [8, 16, 32, 64, 128]
for epoch in epochs:
    for batch_size in batchs_size:

        # initialiser les generateurs
        trainGen = trainAug.flow_from_directory(trainPath, class_mode="categorical", target_size=(224, 224), color_mode="rgb", shuffle=True,  batch_size=batch_size)
        testGen  = testAug.flow_from_directory(testPath,   class_mode="categorical", target_size=(224, 224), color_mode="rgb", shuffle=False, batch_size=batch_size)
        
        model = miniSqueezeNet()
        print("[INFO] training...")
        H = model.fit_generator(trainGen, steps_per_epoch=totalTrain // batch_size, validation_data=testGen, validation_steps=totalTest // batch_size, epochs=epoch)
        
        fig = plt.figure(figsize=(10, 5))  
        plt.title('{} epochs {} batch size .jpg'.format(epoch, batch_size))
        plt.plot(H.history["val_loss"],     label="train loss")
        plt.plot(H.history["val_accuracy"], label="train accuracy")
        plt.plot(H.history["loss"],         label="loss")
        plt.plot(H.history["accuracy"],     label="accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        fig.savefig('{}_epoch_{}_batchsize.jpg'.format(epoch, batch_size), bbox_inches='tight', dpi=300)
"""


epochs = 20
batchs_size = [8, 16, 32, 64, 128]


for batch_size in batchs_size:

    # initialiser les generateurs
    trainGen = trainAug.flow_from_directory(trainPath, class_mode="categorical", target_size=(224, 224), color_mode="rgb", shuffle=True,  batch_size=batch_size)
    testGen  = testAug.flow_from_directory(testPath,   class_mode="categorical", target_size=(224, 224), color_mode="rgb", shuffle=False, batch_size=batch_size)
    
    model = miniSqueezeNet()
    print("[INFO] training...")
    H = model.fit_generator(trainGen, steps_per_epoch=totalTrain // batch_size, validation_data=testGen, validation_steps=totalTest // batch_size, epochs=epochs)
    
    fig = plt.figure(figsize=(10, 5))  
    plt.title('{} epochs {} batch size'.format(epochs, batch_size))
    plt.plot(H.history["val_accuracy"], label="test accuracy")
    plt.plot(H.history["accuracy"],     label="train accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("")
    plt.legend()
    fig.savefig('{}_epoch_{}_batchsize.jpg'.format(epochs, batch_size), bbox_inches='tight', dpi=300)


"""
d'apres les resultats 32 batch size semble etre optimal
"""

"""
epoch = 100
batch_size = 32


# initialiser les generateurs
trainGen = trainAug.flow_from_directory(trainPath, class_mode="categorical", target_size=(224, 224), color_mode="rgb", shuffle=True,  batch_size=batch_size)
testGen  = testAug.flow_from_directory(testPath,   class_mode="categorical", target_size=(224, 224), color_mode="rgb", shuffle=False, batch_size=batch_size)

model = miniSqueezeNet()
print("[INFO] training...")
H = model.fit_generator(trainGen, steps_per_epoch=totalTrain // batch_size, validation_data=testGen, validation_steps=totalTest // batch_size, epochs=epoch)

fig = plt.figure(figsize=(10, 5))  
plt.title('{} epochs {} batch size'.format(epoch, batch_size))
plt.plot(H.history["val_accuracy"], label="test accuracy")
plt.plot(H.history["accuracy"],     label="train accuracy")
plt.xlabel("Epochs")
plt.ylabel("")
plt.legend()
fig.savefig('{}_epoch_{}_batchsize.jpg'.format(epoch, batch_size), bbox_inches='tight', dpi=300)

"""



