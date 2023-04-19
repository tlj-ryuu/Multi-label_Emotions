from keras.layers import Input, LSTM, Dense, Attention


# 定义Attention层
attention = Attention()(lstm)