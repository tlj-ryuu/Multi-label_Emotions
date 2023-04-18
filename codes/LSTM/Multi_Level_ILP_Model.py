import numpy as np
from Preprocess import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K
from ILP import *

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


    test_set = test_set[:100]
    test_label = test_label[:100]
    print("test size : ", len(test_label))

    pre_res = [[] for _ in range(len(test_set))]

    t_cutoff = 0.65
    for i in range(len(test_set)):
        neutral_cnt = 0
        print('------------------------------i------------------------------------:',i)
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

            print('{}: predicted-probability:{} predicted-label:{}'.format(first_level, f_pro, f_label))


            ## 加个清理内存的函数!!!!!!!!
            K.clear_session()
            tf.compat.v1.reset_default_graph()

            if f_label == 0:

                neutral_cnt+=1


            if neutral_cnt == 3:
                pre_res[i].append('neutral')
                continue

            if first_level == 'Positive_LSTM'  and f_label == 1:
                for second_level in ['Joy_ekman_LSTM']:
                    with open(path + 'result_data\\' + second_level + '.json', 'r') as j:
                        json_str = j.read()
                    model = model_from_json(json_str)
                    model.load_weights(path + 'result_data\\' + second_level + '_weights.h5')

                    s_pro = model.predict(sample)[0][0]
                    # print(pre)
                    s_label = (s_pro > 0.5).astype('int')

                    ## 加个清理内存的函数!!!!!!!!
                    K.clear_session()
                    tf.compat.v1.reset_default_graph()
                    # print(s_label)
                    print('{}: predicted-probability:{} predicted-label:{}'.format(second_level, s_pro, s_label))

                    P3 = []
                    isEmptyFlag = True
                    if s_label == 1:
                        joy_ekmans = ["Joy_LSTM", "Amusement_LSTM", "Approval_LSTM", "Excitement_LSTM", "Gratitude_LSTM", "Love_LSTM", "Optimism_LSTM", "Relief_LSTM", "Pride_LSTM",
            "Admiration_LSTM", "Desire_LSTM", "Caring_LSTM"]
                        for third_level in joy_ekmans:
                            with open(path + 'result_data\\' + third_level + '.json', 'r') as j:
                                json_str = j.read()
                            model = model_from_json(json_str)
                            model.load_weights(path + 'result_data\\' + third_level + '_weights.h5')

                            t_pro = model.predict(sample)[0][0]
                            # print(pre)
                            P3.append(t_pro)
                            t_label = (t_pro > t_cutoff).astype('int')

                            if t_label == 1:
                                isEmptyFlag = False
                                emo = third_level.split('_LSTM')[0].lower()
                                pre_res[i].append(emo)

                            ## 加个清理内存的函数!!!!!!!!
                            K.clear_session()
                            tf.compat.v1.reset_default_graph()
                            # print(t_label)
                            print('{}: predicted-probability:{} predicted-label:{}'.format(third_level, t_pro, t_label))

                        ## 对于空标签进行整数线性规划
                        if isEmptyFlag:
                            selected = ILP(P3)
                            for e_i in range(len(selected)):
                                if selected[e_i] == 1:
                                    emo = joy_ekmans[e_i].split('_LSTM')[0].lower()
                                    pre_res[i].append(emo)



            if first_level == 'Ambiguous_LSTM' and f_label == 1:

                for second_level in ['Surprise_ekman_LSTM']:
                    with open(path + 'result_data\\' + second_level + '.json', 'r') as j:
                        json_str = j.read()
                    model = model_from_json(json_str)
                    model.load_weights(path + 'result_data\\' + second_level + '_weights.h5')

                    s_pro = model.predict(sample)[0][0]
                    # print(pre)
                    s_label = (s_pro > 0.5).astype('int')

                    ## 加个清理内存的函数!!!!!!!!
                    K.clear_session()
                    tf.compat.v1.reset_default_graph()
                    # print(s_label)
                    print('{}: predicted-probability:{} predicted-label:{}'.format(second_level, s_pro, s_label))

                    P3 = []
                    isEmptyFlag = True
                    if s_label == 1:
                        surprise_ekmans = ["Surprise_LSTM", "Realization_LSTM", "Confusion_LSTM", "Curiosity_LSTM"]
                        for third_level in surprise_ekmans:
                            with open(path + 'result_data\\' + third_level + '.json', 'r') as j:
                                json_str = j.read()
                            model = model_from_json(json_str)
                            model.load_weights(path + 'result_data\\' + third_level + '_weights.h5')

                            t_pro = model.predict(sample)[0][0]
                            # print(pre)
                            t_label = (t_pro > t_cutoff).astype('int')

                            if t_label == 1:
                                isEmptyFlag = False
                                emo = third_level.split('_LSTM')[0].lower()
                                pre_res[i].append(emo)

                            ## 加个清理内存的函数!!!!!!!!
                            K.clear_session()
                            tf.compat.v1.reset_default_graph()
                            # print(t_label)
                            print('{}: predicted-probability:{} predicted-label:{}'.format(third_level, t_pro, t_label))

                        ## 对空标签进行整数线性规划
                        if isEmptyFlag:
                            selected = ILP(P3)
                            for e_i in range(len(selected)):
                                if selected[e_i] == 1:
                                    emo = surprise_ekmans[e_i].split('_LSTM')[0].lower()
                                    pre_res[i].append(emo)





            if first_level == 'Negative_LSTM' and f_label == 1:

                for second_level in ['Sadness_ekman_LSTM','Fear_ekman_LSTM','Disgust_ekman_LSTM','Anger_ekman_LSTM']:
                    with open(path + 'result_data\\' + second_level + '.json', 'r') as j:
                        json_str = j.read()
                    model = model_from_json(json_str)
                    model.load_weights(path + 'result_data\\' + second_level + '_weights.h5')

                    s_pro = model.predict(sample)[0][0]
                    # print(pre)
                    s_label = (s_pro > 0.5).astype('int')

                    ## 加个清理内存的函数!!!!!!!!
                    K.clear_session()
                    tf.compat.v1.reset_default_graph()
                    # print(s_label)
                    print('{}: predicted-probability:{} predicted-label:{}'.format(second_level, s_pro, s_label))

                    P3 = []
                    isEmptyFlag = True
                    if s_label == 1 and second_level == 'Sadness_ekman_LSTM':
                        sadness_ekmans = ["Sadness_LSTM", "Disappointment_LSTM", "Embarrassment_LSTM", "Grief_LSTM", "Remorse_LSTM"]
                        for third_level in sadness_ekmans:
                            with open(path + 'result_data\\' + third_level + '.json', 'r') as j:
                                json_str = j.read()
                            model = model_from_json(json_str)
                            model.load_weights(path + 'result_data\\' + third_level + '_weights.h5')

                            t_pro = model.predict(sample)[0][0]
                            # print(pre)
                            t_label = (t_pro > t_cutoff).astype('int')

                            if t_label == 1:
                                isEmptyFlag = False
                                emo = third_level.split('_LSTM')[0].lower()
                                pre_res[i].append(emo)

                            ## 加个清理内存的函数!!!!!!!!
                            K.clear_session()
                            tf.compat.v1.reset_default_graph()
                            # print(t_label)
                            print('{}: predicted-probability:{} predicted-label:{}'.format(third_level, t_pro, t_label))

                        ## 对空标签进行整数线性规划
                        if isEmptyFlag:
                            selected = ILP(P3)
                            for e_i in range(len(selected)):
                                if selected[e_i] == 1:
                                    emo = sadness_ekmans[e_i].split('_LSTM')[0].lower()
                                    pre_res[i].append(emo)

                    P3 = []
                    isEmptyFlag = True
                    if s_label == 1 and second_level == 'Fear_ekman_LSTM':
                        fear_ekmans = ["Fear_LSTM", "Nervousness_LSTM"]
                        for third_level in fear_ekmans:
                            with open(path + 'result_data\\' + third_level + '.json', 'r') as j:
                                json_str = j.read()
                            model = model_from_json(json_str)
                            model.load_weights(path + 'result_data\\' + third_level + '_weights.h5')

                            t_pro = model.predict(sample)[0][0]
                            # print(pre)
                            t_label = (t_pro > t_cutoff).astype('int')

                            if t_label == 1:
                                isEmptyFlag = False
                                emo = third_level.split('_LSTM')[0].lower()
                                pre_res[i].append(emo)

                            ## 加个清理内存的函数!!!!!!!!
                            K.clear_session()
                            tf.compat.v1.reset_default_graph()
                            # print(t_label)
                            print('{}: predicted-probability:{} predicted-label:{}'.format(third_level, t_pro, t_label))

                        ## 对空标签进行整数线性规划
                        if isEmptyFlag:
                            selected = ILP(P3)
                            for e_i in range(len(selected)):
                                if selected[e_i] == 1:
                                    emo = fear_ekmans[e_i].split('_LSTM')[0].lower()
                                    pre_res[i].append(emo)

                    P3 = []
                    isEmptyFlag = True
                    if s_label == 1 and second_level == 'Disgust_ekman_LSTM':
                        disgust_ekmans = ["Disgust_LSTM"]
                        for third_level in disgust_ekmans:
                            with open(path + 'result_data\\' + third_level + '.json', 'r') as j:
                                json_str = j.read()
                            model = model_from_json(json_str)
                            model.load_weights(path + 'result_data\\' + third_level + '_weights.h5')

                            t_pro = model.predict(sample)[0][0]
                            # print(pre)
                            t_label = (t_pro > t_cutoff).astype('int')

                            if t_label == 1:
                                isEmptyFlag = False
                                emo = third_level.split('_LSTM')[0].lower()
                                pre_res[i].append(emo)

                            ## 加个清理内存的函数!!!!!!!!
                            K.clear_session()
                            tf.compat.v1.reset_default_graph()
                            # print(t_label)
                            print('{}: predicted-probability:{} predicted-label:{}'.format(third_level, t_pro, t_label))
                        ## 对空标签进行整数线性规划
                        if isEmptyFlag:
                            selected = ILP(P3)
                            for e_i in range(len(selected)):
                                if selected[e_i] == 1:
                                    emo = disgust_ekmans[e_i].split('_LSTM')[0].lower()
                                    pre_res[i].append(emo)

                    P3 = []
                    isEmptyFlag = True
                    if s_label == 1 and second_level == 'Anger_ekman_LSTM':
                        anger_ekmans = ["Anger_LSTM", "Annoyance_LSTM", "Disapproval_LSTM"]
                        for third_level in anger_ekmans:
                            with open(path + 'result_data\\' + third_level + '.json', 'r') as j:
                                json_str = j.read()
                            model = model_from_json(json_str)
                            model.load_weights(path + 'result_data\\' + third_level + '_weights.h5')

                            t_pro = model.predict(sample)[0][0]
                            # print(pre)
                            t_label = (t_pro > t_cutoff).astype('int')

                            if t_label == 1:
                                isEmptyFlag = False
                                emo = third_level.split('_LSTM')[0].lower()
                                pre_res[i].append(emo)

                            ## 加个清理内存的函数!!!!!!!!
                            K.clear_session()
                            tf.compat.v1.reset_default_graph()
                            # print(t_label)
                            print('{}: predicted-probability:{} predicted-label:{}'.format(third_level, t_pro, t_label))

                        ## 对空标签进行整数线性规划
                        if isEmptyFlag:
                            selected = ILP(P3)
                            for e_i in range(len(selected)):
                                if selected[e_i] == 1:
                                    emo = anger_ekmans[e_i].split('_LSTM')[0].lower()
                                    pre_res[i].append(emo)

        print('------------------------------over-------------------------------------')

    print(pre_res)
    MAcc = Cal_MAcc(pre_res,test_label)
    MF1 = Cal_MF1(pre_res,test_label)
    print('MAcc: ',MAcc)
    print('MF1: ',MF1)



if __name__ == '__main__':
    print("################################Multi_Level_Model_testversion###################################")
    path = 'D:\\计算机毕业设计\\'
    Multi_Level_Model(path)
