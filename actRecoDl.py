from keras.datasets import mnist
from keras import models
from keras.layers import Dense
from keras.utils import to_categorical
from random import shuffle
import re
import numpy as np


#preprocessing

def preprocess(samples_list) :
    tr_data = []
    tr_label = []
    for sample_list in samples_list :
        help_list =[]
        tr_label.append(sample_list[1])
        help_list.append(float(sample_list[3]))
        help_list.append(float(sample_list[4]))
        help_list.append(float(sample_list[5]))
        tr_data.append(help_list)

    # print(tr_data)
    # print(tr_label)
    label = ['Walking','Jogging','Sitting','Standing','LyingDown','Stairs']
    indexes = []
    for tl in tr_label :
        if tl in label :
            index = label.index(tl)
            indexes.append(index)
    return (indexes , tr_data)



def vectorize_sequence(sequences , dimension=6):
    results = np.zeros((len(sequences),dimension))
    for i , sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

db = open("rawData.txt","r")
lines = db.readlines()
# print(lines)
samples = []
for line in lines :
    samples.extend(line.split())
# print(samples)

samples_list = []
for sample in samples :
    samples_list.append(sample[:-1].split(','))
print(len(samples_list))
shuffle(samples_list)
test = samples_list[:1000000]
train = samples_list[1000000:]



#train data
train_indexes , tr_data = preprocess(train)
train_labels = vectorize_sequence(train_indexes)
train_data = np.array(tr_data)

#test_data
test_indexes , ts_data = preprocess(test)
test_labels = vectorize_sequence(test_indexes)
test_data = np.array(ts_data)



#network structue
network = models.Sequential()
network.add(Dense(100,activation='relu' , input_shape=(3,)))
network.add(Dense(64,activation='relu'))
network.add(Dense(6,activation='softmax'))

#optimization setting
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#training
network.fit(train_data,train_labels , epochs=6 , batch_size=128)

#testing
score = network.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
