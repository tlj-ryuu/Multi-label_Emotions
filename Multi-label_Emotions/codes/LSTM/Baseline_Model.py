import numpy as np
from Preprocess import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K

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


def Cal_MAcc(pre_res,test_label):
    '''
    e.g.
    pre_res = [['sad',anger'],['joy']]
    test_label = [(1,),(3,4)]
    '''
    n = len(test_label)
    MAcc = 0
    for i in range(n):
        y_p = set(emo2num(pre_res[i]))
        y_t = set(test_label[i])
        MAcc += (len(y_p & y_t) / len(y_p | y_t))
    MAcc = MAcc / n
    return MAcc


def Cal_MF1(pre_res,test_label):
    '''
        e.g.
        pre_res = [['sad',anger'],['joy']]
        test_label = [(1,),(3,4)]
    '''
    n = len(test_label)
    MF1 = 0
    for i in range(n):
        y_p = set(emo2num(pre_res[i]))
        y_t = set(test_label[i])
        MF1 += (len(y_p & y_t) / (len(y_p) + len(y_t)))
    MF1 = 2 * MF1 / n
    return MF1



def Multi_Level_Model(path):
    maxlen = 42
    embedding_dim = 300

    test_set,test_label=prepare_test(path)


    # test_set = test_set[:100]
    # test_label = test_label[:100]
    # print("test size : ",len(test_label))

    pre_res = [[] for _ in range(len(test_set))]


    for i in range(len(test_set)):

        print('------------------------------i------------------------------------:',i)
        ## change into a format that the model can predict
        sample = np.reshape(np.array([test_set[i]]), (1, maxlen, embedding_dim))
        emtions = ["Joy_LSTM", "Amusement_LSTM", "Approval_LSTM", "Excitement_LSTM", "Gratitude_LSTM", "Love_LSTM", "Optimism_LSTM", "Relief_LSTM", "Pride_LSTM",
            "Admiration_LSTM", "Desire_LSTM", "Caring_LSTM","Surprise_LSTM", "Realization_LSTM", "Confusion_LSTM", "Curiosity_LSTM",
                       "Sadness_LSTM", "Disappointment_LSTM", "Embarrassment_LSTM", "Grief_LSTM", "Remorse_LSTM","Fear_LSTM", "Nervousness_LSTM",
                       "Disgust_LSTM","Anger_LSTM", "Annoyance_LSTM", "Disapproval_LSTM","Neutral_LSTM"]
        for emtion in emtions:
            with open(path + 'result_data\\' + emtion + '.json', 'r') as j:
                json_str = j.read()
            model = model_from_json(json_str)
            model.load_weights(path + 'result_data\\' + emtion + '_weights.h5')

            f_pro = model.predict(sample)[0][0]
            # print(pre)
            f_label = (f_pro > 0.5).astype('int')

            # print('{}: predicted-probability:{} predicted-label:{}'.format(emtion, f_pro, f_label))

            if f_label == 1:
                emo = emtion.split('_LSTM')[0].lower()
                pre_res[i].append(emo)

            ## 加个清理内存的函数!!!!!!!!
            K.clear_session()
            tf.compat.v1.reset_default_graph()

        print('------------------------------over-------------------------------------')

    print(pre_res)
    MAcc = Cal_MAcc(pre_res,test_label)
    MF1 = Cal_MF1(pre_res,test_label)
    print('MAcc: ',MAcc)
    print('MF1: ',MF1)



if __name__ == '__main__':
    print("################################Baseline_Model###################################")
    path = 'D:\\计算机毕业设计\\Multi-label_Emotions\\'
    Multi_Level_Model(path)
