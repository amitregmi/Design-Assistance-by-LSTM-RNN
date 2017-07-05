# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:41:13 2017

@author: Amit_Regmi
"""
import os
import json
import nltk
import gensim
import numpy as np
import gensim 
import pickle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.cross_validation import train_test_split
import theano
theano.config.optimizer = "None"


#%%
os.chdir('C:\Users\Amit_Regmi\.spyder\Test_Example\LSTM_Recommendation')
with open('DesignSequenceQA.pickle') as f:
    vec_x, vec_y = pickle.load(f)
    
vec_x = np.array(vec_x, dtype = np.float64)
vec_y = np.array(vec_y, dtype = np.float64)

x_train,x_test,y_train,y_test = train_test_split(vec_x,vec_y,test_size=0.2,random_state=1)         
#%%          
model = Sequential()
model.add(LSTM(output_dim=32,input_shape=x_train.shape[1:],return_sequences=True,init='glorot_normal',inner_init='glorot_normal',activation='sigmoid'))
model.add(LSTM(output_dim=32,input_shape=x_train.shape[1:],return_sequences=True,init='glorot_normal',inner_init='glorot_normal',activation='sigmoid')) 
model.add(LSTM(output_dim=32,input_shape=x_train.shape[1:],return_sequences=True,init='glorot_normal',inner_init='glorot_normal',activation='sigmoid')) 
model.add(LSTM(output_dim=32,input_shape=x_train.shape[1:],return_sequences=True,init='glorot_normal',inner_init='glorot_normal',activation='sigmoid'))    
model.add(Dropout(0.5))
model.add(Dense(y_test.shape[2], activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(y_test.shape[2], activation='softmax'))
model.compile(loss='cosine_proximity',optimizer='adam',metrics=['accuracy'])   

model.fit(x_train, y_train, batch_size=128, nb_epoch=1000, validation_data=(x_test,y_test))
model.save('LSTM1000.h5')
model.fit(x_train, y_train, batch_size=128, nb_epoch=1000, validation_data=(x_test,y_test))
model.save('LSTM2000.h5')
model.fit(x_train, y_train, batch_size=128, nb_epoch=1000, validation_data=(x_test,y_test))
model.save('LSTM3000.h5')
model.fit(x_train, y_train, batch_size=128, nb_epoch=1000, validation_data=(x_test,y_test))
model.save('LSTM4000.h5')
model.fit(x_train, y_train, batch_size=128, nb_epoch=1000, validation_data=(x_test,y_test))
model.save('LSTM5000.h5')

predictions = model.predict(x_test)
# summarize performance of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
#%%              
