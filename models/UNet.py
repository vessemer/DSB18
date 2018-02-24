from keras.models import Model
from keras.layers import Input, concatenate, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K


K.set_image_data_format('channels_last')  # TF dimension ordering in this code

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet(shape, filters_init, optimizer='adam', metrics=[], activation='relu', loss='binary_crossentropy'):
    inputs = Input(shape)
    conv1 = Conv2D(filters_init, (3, 3), activation=activation, padding='same')(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(filters_init, (3, 3), activation=activation, padding='same')(conv1)
    pool1 = Conv2D(filters_init, (3, 3), strides=(2, 2), padding='same')(conv1)
    conv1 = Conv2D(filters_init, (3, 3), activation=activation, padding='same')(conv1)

    conv2 = Conv2D(filters_init * 2, (3, 3), activation=activation, padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(filters_init * 2, (3, 3), activation=activation, padding='same')(conv2)
    pool2 = Conv2D(filters_init, (3, 3), strides=(2, 2), padding='same')(conv2)
    conv2 = Conv2D(filters_init * 2, (3, 3), activation=activation, padding='same')(conv2)

    conv3 = Conv2D(filters_init * 4, (3, 3), activation=activation, padding='same')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(filters_init * 4, (3, 3), activation=activation, padding='same')(conv3)
    pool3 = Conv2D(filters_init, (3, 3), strides=(2, 2), padding='same')(conv3)

    conv4 = Conv2D(filters_init * 8, (3, 3), activation=activation, padding='same')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(filters_init * 8, (3, 3), activation=activation, padding='same')(conv4)

    up6 = concatenate([Conv2DTranspose(filters_init * 4, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv6 = Conv2D(filters_init * 4, (3, 3), activation=activation, padding='same')(up6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(filters_init * 4, (3, 3), activation=activation, padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(filters_init * 2, (2, 2), strides=(2, 2), padding='same')(conv6), conv2], axis=3)
    conv7 = Conv2D(filters_init * 2, (3, 3), activation=activation, padding='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(filters_init * 2, (3, 3), activation=activation, padding='same')(conv7)

    up9 = concatenate([Conv2DTranspose(filters_init, (2, 2), strides=(2, 2), padding='same')(conv7), conv1], axis=3)
    conv9 = Conv2D(filters_init, (3, 3), activation=activation, padding='same')(up9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(filters_init, (3, 3), activation=activation, padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
