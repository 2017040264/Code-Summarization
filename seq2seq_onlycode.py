# -*- coding = utf-8 -*-
# @Time : 2021/4/29 19:42
# @Author : 陈凡亮
# @File : seq2seq_onlycode.py
# @Software : PyCharm

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm


class sEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoding_units, batch_size):
        super(sEncoder, self).__init__()

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


class bahdanauAttention(tf.keras.Model):

    def __init__(self, units):
        super(bahdanauAttention, self).__init__()
        self.w1 = tf.keras.layers.Dense(units)
        self.w2 = tf.keras.layers.Dense(units)
        self.v = tf.keras.layers.Dense(1)

    def call(self, decoder_hidden, code_encoder_output):
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

        # print("at结束",context.shape)
        return code_context


class sDecoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, decoding_units, batch_size):
        super(sDecoder, self).__init__()

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

    def call(self, x, hidden, code_encoding_output):
        context_vector = self.attention(hidden, code_encoding_output)
        # print("context:",context_vector.shape)
        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state


def seq2seq_onlycode_evaluate(code,units,nl_lang,nl_maxlen,train_code_encoder,decoder):
    code = tf.expand_dims(code, axis=0)

    # print(code.shape)
    result = ''

    # hidden是一维的，输入只有一个
    hidden = [tf.zeros((1, units))]
    # code:(1,500) hidden(1,2)
    # out:(1,500,2)
    code_encoding_out, code_encoding_hidden = train_code_encoder(code, hidden)

    decoding_hidden = code_encoding_hidden

    # decoder input shape=(1,1)
    decoding_input = tf.expand_dims([nl_lang.word_index['<start>']], 0)

    for t in range(nl_maxlen):
        pred, decoding_hidden = decoder(decoding_input, decoding_hidden, code_encoding_out)

        # 取预测结果中概率最大的值
        pred_id = tf.argmax(pred[0]).numpy()

        if nl_lang.index_word[pred_id] == '<end>':
            return result

        result += nl_lang.index_word[pred_id] + ' '

        # fed back
        decoding_input = tf.expand_dims([pred_id], 0)

    return result


def seq2seq_onlycode_translate(tensor, path,units,nl_lang,nl_maxlen,train_code_encoder,decoder,num=None):
    if num==None:
        num=len(tensor)
    with open(path, 'w+') as targ:
        for i in tqdm(range(num)):
            can = seq2seq_onlycode_evaluate(tensor[i],units,nl_lang,nl_maxlen,train_code_encoder,decoder)
            #print(can)
            targ.write(can + '\n')