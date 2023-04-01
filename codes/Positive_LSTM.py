import numpy as np
from Preprocess import *
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,LSTM

def preLoad(path):
    dataset = read_tsv(path)

