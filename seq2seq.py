# -*- coding = utf-8 -*-
# @Time : 2021/4/26 13:56
# @Author : 陈凡亮
# @File : seq2seq.py
# @Software : PyCharm

import tensorflow as tf
from tensorflow import keras

# 编码器
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoding_units, batch_size):
        super(Encoder, self).__init__()

        self.batch_size = batch_size
        self.encoding_units = encoding_units

        # 输入是（batch_size,sequence_length）输出是（batch_size,sequence_length,embedding_dim）
        # embedding 创建一个（vocab_size,embedding_dim）大小的查询表对象
        # 对于任意单词编号i,v=table[i]
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)

        self.gru = keras.layers.GRU(self.encoding_units,
                                    dropout=0.5,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)

        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def init_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoding_units))

# 注意力机制
class bahdanauAttention(tf.keras.Model):

    def __init__(self, units):
        super(bahdanauAttention, self).__init__()
        self.w1 = tf.keras.layers.Dense(units)
        self.w2 = tf.keras.layers.Dense(units)
        self.v = tf.keras.layers.Dense(1)

    def call(self, decoder_hidden, code_encoder_output, sbt_encoder_output):
        # print("初始输入：")
        # print("hidden",decoder_hidden.shape)
        # print("code",code_encoder_output.shape)
        # print("sbt",sbt_encoder_output.shape)

        # hidden state扩维
        # encoder 的输出为（batch_size,sequnece_length,units）,而hidden为（batch_size,units）
        # 为了使二者能够相加，需要对hidden进行扩维
        hidden_with_time_axis = tf.expand_dims(decoder_hidden, axis=1)

        # 计算code——encoder
        # score=FC(tanh(FC(EO)+FC(H)))
        code_score = self.v(tf.nn.tanh(self.w1(code_encoder_output) + self.w2(hidden_with_time_axis)))
        # attention_weihts=softmax(score,acis=1)
        code_attention_weights = tf.nn.softmax(code_score, axis=1)
        # context=sum(attention_weigths*EO,axis=1)
        code_context = tf.reduce_sum(code_attention_weights * code_encoder_output, axis=1)

        # 计算sbt_encoder

        sbt_score = self.v(tf.nn.tanh(self.w1(sbt_encoder_output) + self.w2(hidden_with_time_axis)))
        sbt_attention_weights = tf.nn.softmax(sbt_score, axis=1)
        sbt_context = tf.reduce_sum(sbt_attention_weights * sbt_encoder_output, axis=1)

        # 二者相加
        attention_weights = code_attention_weights + sbt_attention_weights
        context = code_context + sbt_context
        # print("at结束",context.shape)
        return context, attention_weights

# 解码器
class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, decoding_units, batch_size):
        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.decoding_units = decoding_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = keras.layers.GRU(self.decoding_units,
                                    dropout=0.5,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = keras.layers.Dense(vocab_size)
        # self.drop=keras.layers.Dropout(0.5)
        self.attention = bahdanauAttention(self.decoding_units)

    def call(self, x, hidden, code_encoding_output, sbt_encodeing_output):
        context_vector, attention_weights = self.attention(hidden, code_encoding_output, sbt_encodeing_output)
        # print("context:",context_vector.shape)
        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights


# 超参数
# batch_size=128
# units=256
# embedding_dim=256
# vocab_maxlen=30000

