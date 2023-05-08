import numpy as np
import pandas as pd
import random
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.casual import casual_tokenize
from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format(r'F:\GoogleNews-vectors-negative300.bin.gz',binary=True,limit=200000)

def read_tsv(filepath):
    dataset = []
    with open(filepath,encoding='utf-8') as f:
        for line in f.readlines():
            tmp = line.strip().split('\t')
            ## multilabels processing
            labels = tuple(map(int,tmp[1].strip().split(',')))
            dataset.append((tmp[0],labels))
    return dataset

def tokenize_and_vectorize(dataset):
    vectorized = []
    for sample in dataset:

        ## casual tokenize can be used for language processing on social networks and is good at emoticons and long, nonstandard sentences like awesommmmmeeee,
        ## but actually experimenting with TreebankWordTokenizer works not only with list symbols but also with don 't
        # tokens2 = casual_tokenize(sample[0])
        tokens = TreebankWordTokenizer().tokenize(sample[0])
        # print(tokens)

        ## process the [NAME]or[RELIGION]
        combined = []
        i = 0
        # print(tokens)
        while i < len(tokens)-1:
            if tokens[i]=='[' and tokens[i+1]=='NAME':
                combined.append('[NAME]')
                i += 3
                continue
            if tokens[i]=='[' and tokens[i+1]=='RELIGION':
                combined.append('[RELIGION]')
                i += 3
                continue

            if tokens[i].isalpha():
                combined.append(tokens[i].lower())
            else:
                combined.append(tokens[i])
            i += 1
        # print(tokens)
        # print(combined)

        sample_vecs = []
        for token in combined:
            try:
                sample_vecs.append(word_vectors[token])
            ## case where the word does not exist in wordvec
            except KeyError:
                UNK = []
                for i in range(300):
                    UNK.append(random.uniform(-0.25,0.25))
                sample_vecs.append(np.array(UNK))
                # print(np.array(UNK))
        vectorized.append(sample_vecs)

    return vectorized

def emo2num(subemo):
    '''
    transform emotions list to index list
    '''
    emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion'
        , 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement'
        , 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief'
        , 'remorse', 'sadness', 'surprise', 'neutral']

    numlist = []
    for emo in subemo:
        numlist.append(emotions.index(emo))
    return numlist



def collect_labels(dataset,name):
    '''
    There are three levels in total, the first layer has 3 categories,
    the second layer has 6 categories, the third layer has 28 categories,
    according to the binary method needs to build 37 pairs of 0,1 tags
    '''

    emotions = ['admiration','amusement','anger','annoyance','approval','caring','confusion'
        ,'curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement'
        ,'fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief'
        ,'remorse','sadness','surprise','neutral']
    sentiment = ['positive','negative','ambiguous']
    ekman = ['anger_ekman','disgust_ekman','fear_ekman','joy_ekman','sadness_ekman','surprise_ekman']
    labels = {}
    ## initialization

    for em in sentiment:
        labels[em] = [-1]*len(dataset)
    for em in ekman:
        labels[em] = [-1]*len(dataset)
    for em in emotions:
        labels[em] = [-1]*len(dataset)


    positive = ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"]
    negative = ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"]
    ambiguous = ["realization", "surprise", "curiosity", "confusion"]
    anger_ekman = ["anger", "annoyance", "disapproval"]
    disgust_ekman = ["disgust"]
    fear_ekman = ["fear", "nervousness"]
    joy_ekman = ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride",
            "admiration", "desire", "caring"]
    sadness_ekman =  ["sadness", "disappointment", "embarrassment", "grief", "remorse"]
    surprise_ekman = ["surprise", "realization", "confusion", "curiosity"]


    emotion_dict = {}
    emotion_dict['positive'] = emo2num(positive)
    emotion_dict['negative'] = emo2num(negative)
    emotion_dict['ambiguous'] = emo2num(ambiguous)
    emotion_dict['anger_ekman'] = emo2num(anger_ekman)
    emotion_dict['disgust_ekman']= emo2num(disgust_ekman)
    emotion_dict['fear_ekman'] = emo2num(fear_ekman)
    emotion_dict['joy_ekman']   = emo2num(joy_ekman)
    emotion_dict['sadness_ekman'] = emo2num(sadness_ekman)
    emotion_dict['surprise_ekman']= emo2num(surprise_ekman)
    for basic in emotions:
        emotion_dict[basic] = [emotions.index(basic)]


    for i in range(len(dataset)):
        # print('--------------start-----------------',i)
        sample_label=dataset[i][1]
        # print('sample_label',sample_label)
        for em in labels:
            # print('em',em)
            for sam in sample_label:
                # print('sam',sam)
                # print('emtion_dict',emotion_dict[em])
                if sam in emotion_dict[em]:
                    labels[em][i] = 1
                else:
                    if labels[em][i]!=1:
                        labels[em][i]=0


        # print('---------------end---------------------',i)

    ##write into the csv
    path = 'D:\\计算机毕业设计\\result_data\\' + name + '_labels.csv'
    tx = pd.DataFrame(labels)
    tx.to_csv(path, encoding='utf-8', mode='a', header=True, index=False, sep=',')

    return labels

def pad_trunc(vectorized,maxlen):
    '''
    pad with zero or truncate to maxlen
    '''
    embedding_dim = 300

    new = [sample[:maxlen] + [np.array([0.] * embedding_dim)] * (maxlen - len(sample)) for sample in vectorized]
    return new

def read_label(filepath,index):
    res = []
    with open(filepath) as f:
        for line in f.readlines():
            tmp = line.strip().split(',')[index]
            if tmp=='0' or tmp=='1':
                res.append(int(tmp))
    return res


if __name__=='__main__':
    train_path=r'D:\计算机毕业设计\GoEmotions\train.tsv'
    train_data = read_tsv(train_path)

    cnt = 0
    all = 0
    for tmp in train_data:
        all += 1
        if (len(tmp[1])>1):
            cnt += 1
    print('所有数为{}，多标签数为{}'.format(all,cnt))

    dev_path = r'D:\计算机毕业设计\GoEmotions\dev.tsv'
    dev_data = read_tsv(dev_path)

    cnt = 0
    all = 0
    for tmp in dev_data:
        all += 1
        if (len(tmp[1]) > 1):
            cnt += 1
    print('所有数为{}，多标签数为{}'.format(all, cnt))

    test_path = r'D:\计算机毕业设计\GoEmotions\test.tsv'
    test_data = read_tsv(test_path)

    cnt = 0
    all = 0
    for tmp in test_data:
        all += 1
        if (len(tmp[1]) > 1):
            print(tmp[1])
            cnt += 1
    print('所有数为{}，多标签数为{}'.format(all, cnt))

    # train_labels = collect_labels(dataset=train_data,name='train')
    # dev_labels = collect_labels(dataset=dev_data,name='dev')
    # test_labels = collect_labels(dataset=test_data,name='test')


    # print(test_data)
    # vectorized = tokenize_and_vectorize(dev_data)
    #
    # # print(vectorized)
    # pad = pad_trunc(vectorized,42)
    # print(pad[1])
    # print(len(pad[1]))
    # for i in range(len(pad[0])):
    #     print(pad[0][i])
    #     print('----------------------------------------------------------------------------')

    # print(len(vectorized[-3]))


    # labels = collect_labels(dataset=test_data,name='test')

    # print(labels['curiosity'][13])
    # print(labels['positive'][13])
    # print(labels['ambiguous'][13])
    # print(labels['surprise_ekman'][13])
    # print(labels['joy_ekman'][13])
    # print(labels['sadness_ekman'][13])
    # print(labels['admiration'][13])
