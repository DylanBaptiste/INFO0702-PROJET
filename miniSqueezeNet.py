from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.optimizers import Adam

def miniSqueezeNet(input_shape=(224, 224, 3), nb_classes=11):
    
    input_1 = Input(input_shape)
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
    conv2 = Conv2D(nb_classes, kernel_size=1, activation='relu', padding='valid')(dropout1)
    global_average_pooling2d_1 = GlobalAveragePooling2D()(conv2)
    output = Dense(nb_classes, activation='softmax')(global_average_pooling2d_1)
    
    model = Model(inputs=input_1, outputs=output)
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model