from keras.datasets import mnist
from keras import models
from keras.layers import Dense
from keras.utils import to_categorical
from random import shuffle
import re
import numpy as np
import matplotlib.pyplot as plt


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


def build_model():
    network = models.Sequential()
    network.add(Dense(100,activation='relu' , input_shape=(3,)))
    network.add(Dense(64,activation='relu'))
    network.add(Dense(6,activation='softmax'))

    #optimization setting
    network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return network



k = 4
num_val_samples = len(train_data) // k
num_epochs = 15
batch_size = 128
all_mae_histories = []
histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_labels[i * num_val_samples : (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([train_labels[:i * num_val_samples],
                                         train_labels[(i + 1) * num_val_samples:]],
                                        axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs,
              batch_size=batch_size, verbose=0, validation_data=(val_data, val_targets))
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    histories.append(history)


[history.history.keys() for history in histories]
average_mae_history = [np.mean([x[i] for x in all_mae_histories])
                           for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


points = np.array(average_mae_history)
plt.plot(points[10:])
# plt.yscale('log')
plt.show()
