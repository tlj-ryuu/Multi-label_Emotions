## 这是博客上人家的实现方式，我自己稍微改了一下
# https://blog.csdn.net/weixin_46277779/article/details/125718106

import numpy as np
from Preprocess import *
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, LSTM,Multiply,Layer
from keras.models import model_from_json

np.random.seed(99)


# 自定义注意力层
from keras import initializers, constraints, activations, regularizers
from keras import backend as K
from keras.layers import Input

## 自定义的注意力机制层
class Attention(Layer):
    # 返回值：返回的不是attention权重，而是每个timestep乘以权重后相加得到的向量。
    # 输入:输入是rnn的timesteps，也是最长输入序列的长度
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],), initializer=self.init, name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer, constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],), initializer='zero', name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer, constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None  ## 后面的层不需要mask了，所以这里可以直接返回none

    def call(self, x, mask=None):
        features_dim = self.features_dim  ## 这里应该是 step_dim是我们指定的参数，它等于input_shape[1],也就是rnn的timesteps
        step_dim = self.step_dim

        # 输入和参数分别reshape再点乘后，tensor.shape变成了(batch_size*timesteps, 1),之后每个batch要分开进行归一化
        # 所以应该有 eij = K.reshape(..., (-1, timesteps))

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)  # RNN一般默认激活函数为tanh, 对attention来说激活函数差别不大，因为要做softmax
        a = K.exp(eij)
        if mask is not None:  ## 如果前面的层有mask，那么后面这些被mask掉的timestep肯定是不能参与计算输出的，也就是将他们attention权重设为0
            a *= K.cast(mask, K.floatx())  ## cast是做类型转换，keras计算时会检查类型，可能是因为用gpu的原因

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)  # a = K.expand_dims(a, axis=-1) , axis默认为-1， 表示在最后扩充一个维度。比如shape = (3,)变成 (3, 1)
        ## 此时a.shape = (batch_size, timesteps, 1), x.shape = (batch_size, timesteps, units)
        weighted_input = x * a
        # weighted_input的shape为 (batch_size, timesteps, units), 每个timestep的输出向量已经乘上了该timestep的权重
        # weighted_input在axis=1上取和，返回值的shape为 (batch_size, 1, units)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):  ## 返回的结果是c，其shape为 (batch_size, units)
        return input_shape[0], self.features_dim


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


    inputs = Input(name='inputs', shape=(maxlen, embedding_size), dtype='float64')

    attention_probs = Dense(embedding_size, activation='softmax', name='attention_vec')(inputs)
    attention_mul = Multiply()([inputs, attention_probs])
    mlp = Dense(num_neurons)(attention_mul)  # 原始的全连接
    fla = Flatten()(mlp)
    output = Dense(1, activation='sigmoid')(fla)
    model = Model(inputs=[inputs], outputs=output)

    model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(train_set, train_label, batch_size=batch_size, epochs=epochs, validation_data=(dev_set, dev_label))

    # ##save model
    # model_structure = model.to_json()
    # with open(path + 'result_data\\Admiration_Attention_v1.json', 'w') as j:
    #     j.write(model_structure)
    #
    # model.save_weights(path + 'result_data\\Admiration_Attention_v1_weights.h5')


def testModel(path, test_set, test_label):
    ##reload model
    with open(path + 'result_data\\Admiration_Attention_v1.json', 'r') as j:
        json_str = j.read()
    model = model_from_json(json_str)

    model.load_weights(path + 'result_data\\Admiration_Attention_v1_weights.h5')

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
    # testModel(path, test_set, test_label)