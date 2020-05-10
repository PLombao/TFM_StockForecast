from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def create_basic_ann(optimizer='adagrad',
                     kernel_initializer='glorot_uniform',
                     dropout=0.2):
    """
    Crea una red neuronal basica para regresion
    """
    model = Sequential()
    model.add(Dense(8, activation='relu', kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(8, activation='relu', kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer))
    model.compile(loss=root_mean_squared_error, optimizer=optimizer)

    return model