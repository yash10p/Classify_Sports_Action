import os
import numpy as np
import cv2
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Average, Merge
from keras.layers import Convolution2D as Conv2D, MaxPooling2D, TimeDistributed, LSTM, Lambda, BatchNormalization, GlobalAveragePooling1D
from keras.optimizers import Adam, SGD
from keras.utils import plot_model, np_utils
from keras.metrics import categorical_accuracy
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import matplotlib.pyplot as plt
import load_preprocess as lp
import load_preprocess_flow as lpf
import load_preprocess_rgb as lpr

dir_path = os.path.dirname(os.path.realpath(__file__))

epochs = 20
learning_rate = 0.0001
decay_rate = learning_rate / epochs
momentum = 0.8
seed = 42

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
def make_flow_model():
    model = Sequential()

    # define CNN model
    model.add(TimeDistributed(Conv2D(64, kernel_size=(7, 7), strides=(2,2)), input_shape=(20, 32, 32, 3)))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2,2))))
    model.add(TimeDistributed(BatchNormalization()))
    
    model.add(TimeDistributed(Conv2D(128, (5, 5), padding="same")))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2,2))))
    
    model.add(TimeDistributed(Conv2D(128, (3, 3), padding="same")))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(256, activation='relu')))
    model.add(TimeDistributed(Dropout(0.6)))
    
    # define LSTM model
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(Dense(9, activation='softmax'))
    model.add(GlobalAveragePooling1D(name="global_avg_flow"))
    
    return model
    
def make_rgb_model():
    model = Sequential()

    # define CNN model
    model.add(TimeDistributed(Conv2D(64, kernel_size=(5, 5), strides=(2,2)), input_shape=(20, 32, 32, 3)))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2,2))))
    model.add(TimeDistributed(BatchNormalization()))
    
    model.add(TimeDistributed(Conv2D(128, (3, 3))))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(256, activation='relu')))
    model.add(TimeDistributed(Dropout(0.7)))
    
    # define LSTM model
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(TimeDistributed(Dense(9, activation='softmax')))
    model.add(GlobalAveragePooling1D(name="global_avg_rgb"))
    
    return model

def build_model():
    rgb_model = make_rgb_model()
    flow_model = make_flow_model()

    input1 = keras.layers.Input(shape=(20,32,32,3))
    input2 = keras.layers.Input(shape=(20,32,32,3))
    x1 = rgb_model(input1)
    x2 = flow_model(input2)
    
    out = keras.layers.multiply([x1, x2])
    model = keras.models.Model(inputs=[input1, input2], outputs=out)
    
    #model = Sequential()
    #model.add(Merge([rgb_model, flow_model], mode='ave'))
    #model.add(Average([rgb_model, flow_model]))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy', 'top_k_categorical_accuracy'])
    print(model.summary(90))
    plot_model(model, to_file='cnn_lstm.png')

    return model
    
def train_model():
    model = build_model()
    FLOW, RGB, Y = lpr.load_data()
    #X, Y = shuffle(X, Y, random_state=42)
    print "FLOW: ", FLOW.shape
    print "RGB: ", RGB.shape
    
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)
    checkpoint_callback = ModelCheckpoint('my_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    for train_index, test_index in sss.split(FLOW, Y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        flow_train, flow_test = FLOW[train_index], FLOW[test_index]
        rgb_train, rgb_test = RGB[train_index], RGB[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    
    #x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    model.fit([flow_train, rgb_train], y_train,
          batch_size = 2,
          epochs=epochs,
          validation_data=([flow_test, rgb_test], y_test),
          callbacks=[early_stopping_callback, checkpoint_callback],
          shuffle=True)
    
    #for train, test in kfold.split(RGB, np.zeros(shape=(RGB.shape[0], 1))):
    #    model.fit([FLOW[train], RGB[train]], Y[train],
    #          batch_size = 2,
    #          epochs=epochs,
    #          validation_data=([FLOW[test], RGB[test]], Y[test]),
    #          callbacks=[early_stopping_callback, checkpoint_callback],
    #          shuffle=True)

        #scores = model.evaluate(X[test], Y[test], verbose=0)
        #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

              
    #score,acc = model.evaluate(x_test, y_test)
    #print 'score: ', score, 'accuraccy: ', acc
    
    #model.save("my_model.h5")
    return model
    

if __name__ == "__main__":
    model = train_model()
    
    #model = load_model('my_model.h5')
    #vid_path = '/home/rohit/projects/NNFL/project/data/3.avi'
    #predict(vid_path, model)
