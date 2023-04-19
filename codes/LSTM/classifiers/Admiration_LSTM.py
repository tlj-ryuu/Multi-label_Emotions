import numpy as np
from Preprocess import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.models import model_from_json


def preLoad(path, index):
    data_path = path + 'GoEmotions\\'
    label_path = path + 'result_data\\'

    train_set = read_tsv(data_path + 'train.tsv')
    dev_set = read_tsv(data_path + 'dev.tsv')
    test_set = read_tsv(data_path + 'test.tsv')
    print('---------------------------------read dataset over---------------------------------')

    train_label = read_label(label_path + 'train_labels.csv', index)
    dev_label = read_label(label_path + 'dev_labels.csv', index)
    test_label = read_label(label_path + 'test_labels.csv', index)
    print('---------------------------------read label over---------------------------------')

    train_set = tokenize_and_vectorize(train_set)
    dev_set = tokenize_and_vectorize(dev_set)
    test_set = tokenize_and_vectorize(test_set)
    print('---------------------------------tokenize vectorize over---------------------------------')

    ##let the longest text in the training set be maxlen for padding or truncate
    # maxlen = 0
    # for i in range(len(train_set)):
    #     length = len(train_set[i])
    #     if length > maxlen:
    #         maxlen = length
    #
    # print(maxlen)
    # 42

    maxlen = 42
    embedding_dim = 300

    train_set = pad_trunc(train_set, maxlen)
    dev_set = pad_trunc(dev_set, maxlen)
    test_set = pad_trunc(test_set, maxlen)
    print('---------------------------------pad trunc over---------------------------------')

    ## Never print all of them, they can take a long time
    # print(dev_set[0])

    ##Remolding into a numpy data structure for efficient storage
    train_set = np.reshape(train_set, (len(train_set), maxlen, embedding_dim))
    dev_set = np.reshape(dev_set, (len(dev_set), maxlen, embedding_dim))
    test_set = np.reshape(test_set, (len(test_set), maxlen, embedding_dim))
    train_label = np.array(train_label)
    dev_label = np.array(dev_label)
    test_label = np.array(test_label)
    # print(dev_set[0])
    # print('------------------------------------------------------------------------')
    # print(dev_set)

    return train_set, train_label, dev_set, dev_label, test_set, test_label


def mainModel(path, train_set, train_label, dev_set, dev_label):
    num_neurons = 50
    batch_size = 32
    maxlen = 42
    embedding_size = 300
    epochs = 2

    model = Sequential()
    model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_size)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(train_set, train_label, batch_size=batch_size, epochs=epochs, validation_data=(dev_set, dev_label))

    ##save model
    model_structure = model.to_json()
    with open(path + 'result_data\\Admiration_LSTM.json', 'w') as j:
        j.write(model_structure)

    model.save_weights(path + 'result_data\\Admiration_LSTM_weights.h5')


def testModel(path, test_set, test_label):
    ##reload model
    with open(path + 'result_data\\Admiration_LSTM.json', 'r') as j:
        json_str = j.read()
    model = model_from_json(json_str)

    model.load_weights(path + 'result_data\\Admiration_LSTM_weights.h5')

    pre = model.predict(test_set)
    pre_label = (pre > 0.5).astype('int')

    print('real_label:')
    print(test_label)
    print('pre_prob:')
    print(pre)
    print('pre_label:')
    print(pre_label)

    ## calculate accuracy
    cnt = 0
    slen = len(test_label)
    for i in range(slen):
        if test_label[i] == pre_label[i][0]:
            cnt += 1
    acc = cnt / slen
    print('Accuracy: {:.5f}'.format(acc))


if __name__ == '__main__':
    path = 'D:\\计算机毕业设计\\'
    train_set, train_label, dev_set, dev_label, test_set, test_label = preLoad(path, 9)
    mainModel(path,train_set, train_label, dev_set, dev_label)
    print('-----------------------------training over------------------------------------')
    testModel(path, test_set, test_label)



