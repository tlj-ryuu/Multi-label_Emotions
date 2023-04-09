import numpy as np
from Preprocess import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.models import model_from_json

def prepare_test(path):
    maxlen = 42
    embedding_dim = 300
    data_path = path + 'GoEmotions\\'
    label_path = path + 'result_data\\'
    test_set = read_tsv(data_path + 'test.tsv')
    # print(test_set)

    test_label = []
    for cell in test_set:
        test_label.append(cell[1])


    test_set = tokenize_and_vectorize(test_set)
    test_set = pad_trunc(test_set, maxlen)
    test_set = np.reshape(test_set, (len(test_set), maxlen, embedding_dim))
    # test_label = np.array(test_label)
    # print(test_label)

    print('--------------------------data preprocess over--------------------------------')

    return test_set,test_label

def Multi_Level_Model(path):
    maxlen = 42
    embedding_dim = 300

    test_set,test_label=prepare_test(path)



    test_set = test_set[:15]
    test_label = test_label[:15]

    pre_res = []

    for i in range(len(test_set)):
        ## change into a format that the model can predict
        sample = np.reshape(np.array([test_set[i]]), (1, maxlen, embedding_dim))

        for first_level in ['Positive_LSTM', 'Negative_LSTM', 'Ambiguous_LSTM']:
            with open(path + 'result_data\\' + first_level + '.json', 'r') as j:
                json_str = j.read()
            model = model_from_json(json_str)
            model.load_weights(path + 'result_data\\' + first_level + '_weights.h5')

            f_pro = model.predict(sample)[0][0]
            # print(pre)
            f_label = (f_pro > 0.5).astype('int')
            # print('{}: predicted-probability:{} predicted-label:{}'.format(first_level,f_pro,f_label))

            ## 加个清理内存的函数!!!!!!!!

            if first_level == 'Positive_LSTM'  and f_label == 1:
                for second_level in ['Joy_ekman_LSTM']:
                    with open(path + 'result_data\\' + second_level + '.json', 'r') as j:
                        json_str = j.read()
                    model = model_from_json(json_str)
                    model.load_weights(path + 'result_data\\' + second_level + '_weights.h5')

                    f_pro = model.predict(sample)[0][0]
                    # print(pre)
                    f_label = (f_pro > 0.5).astype('int')

                    ## 加个清理内存的函数!!!!!!!!

            if first_level == 'Ambiguous_LSTM' and f_label == 1:
                for second_level in ['Surprise_ekman_LSTM']:
                    with open(path + 'result_data\\' + second_level + '.json', 'r') as j:
                        json_str = j.read()
                    model = model_from_json(json_str)
                    model.load_weights(path + 'result_data\\' + second_level + '_weights.h5')

                    f_pro = model.predict(sample)[0][0]
                    # print(pre)
                    f_label = (f_pro > 0.5).astype('int')

                    ## 加个清理内存的函数!!!!!!!!

            if first_level == 'Negative_LSTM' and f_label == 1:
                for second_level in ['Sadness_ekman_LSTM','Fear_ekman_LSTM','Disgust_ekman_LSTM','Anger_ekman_LSTM']:
                    with open(path + 'result_data\\' + second_level + '.json', 'r') as j:
                        json_str = j.read()
                    model = model_from_json(json_str)
                    model.load_weights(path + 'result_data\\' + second_level + '_weights.h5')

                    f_pro = model.predict(sample)[0][0]
                    # print(pre)
                    f_label = (f_pro > 0.5).astype('int')

                    ## 加个清理内存的函数!!!!!!!!











if __name__ == '__main__':
    path = 'D:\\计算机毕业设计\\'
    Multi_Level_Model(path)