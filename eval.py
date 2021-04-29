# -*- coding = utf-8 -*-
# @Time : 2021/4/26 14:06
# @Author : 陈凡亮
# @File : eval.py
# @Software : PyCharm

import tensorflow as tf
from tqdm import tqdm
from nltk.translate.bleu_score import *
from transformer import create_masks

def seq2seq_evaluate(code, sbt,units,nl_lang,nl_maxlen,train_code_encoder,train_sbt_encoder,decoder):
    code = tf.expand_dims(code, axis=0)
    sbt = tf.expand_dims(sbt, axis=0)
    # print(code.shape)
    result = ''

    # hidden是一维的，输入只有一个
    hidden = [tf.zeros((1, units))]
    # code:(1,500) hidden(1,2)
    # out:(1,500,2)
    code_encoding_out, code_encoding_hidden = train_code_encoder(code, hidden)
    sbt_encoding_out, sbt_encoding_hidden = train_sbt_encoder(sbt, hidden)

    decoding_hidden = code_encoding_hidden

    # decoder input shape=(1,1)
    decoding_input = tf.expand_dims([nl_lang.word_index['<start>']], 0)

    for t in range(nl_maxlen):
        pred, decoding_hidden, _ = decoder(decoding_input, decoding_hidden, code_encoding_out, sbt_encoding_out)

        # 取预测结果中概率最大的值
        pred_id = tf.argmax(pred[0]).numpy()

        if nl_lang.index_word[pred_id] == '<end>':
            return result

        result += nl_lang.index_word[pred_id] + ' '

        # fed back
        decoding_input = tf.expand_dims([pred_id], 0)

    return result


def seq2seq_result(code_tensor,sbt_tensor,path,units,nl_lang,nl_maxlen,train_code_encoder,train_sbt_encoder,decoder,num=None):
    if num==None:
        num=len(code_tensor)

    with open(path,'w+') as targ:
        #for i in tqdm(range(len(code_tensor))):
        for i in tqdm(range(num)):
            #print(c[i])
            #print(s[i])
            can=seq2seq_evaluate(code_tensor[i],sbt_tensor[i],units,nl_lang,nl_maxlen,train_code_encoder,train_sbt_encoder,decoder)
            #print(can)
            targ.write(can+'\n')


def transformer_evaluate(tensor,nl_lang,nl_maxlen,transformer):
    result = ''
    encoder_input = tf.expand_dims(tensor, axis=0)
    output = tf.expand_dims([nl_lang.word_index['<start>']], axis=0)
    for i in range(nl_maxlen):

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)

        # 从 seq_len 维度选择最后一个词
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        pre = tf.argmax(predictions, axis=-1)
        predicted_id = tf.cast(pre, tf.int32)
        pre = pre.numpy()[0][0]

        # 如果 predicted_id 等于结束标记，就返回结果
        if nl_lang.index_word[pre] == '<end>':
            return result

        result += nl_lang.index_word[pre] + ' '

        # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
        output = tf.concat([output, predicted_id], axis=-1)

    return result


def transformer_result(tensor,path,nl_lang,nl_maxlen,transformer,num=None):
    if num==None:
        num=len(tensor)

    #print('num= ',num)
    with open(path, 'w+') as re:
        for i in tqdm(range(num)):
            result = transformer_evaluate(tensor[i],nl_lang,nl_maxlen,transformer)
            #print(result)
            re.write(result + '\n')


# start seq2trans集成
def seq2trans_evaluate(code, sbt,units,nl_lang,nl_maxlen,train_code_encoder,train_sbt_encoder,decoder,transformer):
    code = tf.expand_dims(code, axis=0)
    sbt = tf.expand_dims(sbt, axis=0)
    # print(code.shape)
    result = ''

    # hidden是一维的，输入只有一个
    hidden = [tf.zeros((1, units))]
    # code:(1,500) hidden(1,2)
    # out:(1,500,2)
    code_encoding_out, code_encoding_hidden = train_code_encoder(code, hidden)
    sbt_encoding_out, sbt_encoding_hidden = train_sbt_encoder(sbt, hidden)

    decoding_hidden = code_encoding_hidden

    # decoder input shape=(1,1)
    decoding_input = tf.expand_dims([nl_lang.word_index['<start>']], 0)
    # transformer 的输入
    output=decoding_input

    for t in range(nl_maxlen):

        pred, decoding_hidden, _ = decoder(decoding_input, decoding_hidden, code_encoding_out, sbt_encoding_out)
        # 取预测结果中概率最大的值
        seq2seq_max=tf.reduce_max(pred,axis=-1).numpy()[0]


        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(code, output)
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = transformer(code, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
        transformer_max=tf.reduce_max(predictions,axis=-1).numpy()[0][0]

        # seq2seq模型得到的预测概率大
        if seq2seq_max>=transformer_max:
            pred_id = tf.argmax(pred[0]).numpy()
        else:
            # 从 seq_len 维度选择最后一个词
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            pred_id = tf.argmax(predictions, axis=-1)
            predicted_id = tf.cast(pred_id, tf.int32)
            #print('predicted_id:',predicted_id)
            pred_id = pred_id.numpy()[0][0]


        if nl_lang.index_word[pred_id] == '<end>':
            return result

        result += nl_lang.index_word[pred_id] + ' '

        # fed back
        decoding_input = tf.expand_dims([pred_id], 0)

        # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
        predicted_id = tf.cast(pred_id, tf.int32)
        predicted_id=tf.expand_dims(predicted_id,axis=0)
        predicted_id = tf.expand_dims(predicted_id, axis=0)
        output = tf.concat([output, predicted_id], axis=-1)

    return  result

def seq2trans_result(code, sbt,path,units,nl_lang,nl_maxlen,train_code_encoder,train_sbt_encoder,decoder,transformer,num=None):
    if num==None:
        num=len(code)
    print('num= ', num)
    with open(path,'w+') as targ:
        #for i in tqdm(range(len(code_tensor))):
        for i in tqdm(range(num)):
            #print(c[i])
            #print(s[i])
            can=seq2trans_evaluate(code[i],sbt[i],units,nl_lang,nl_maxlen,train_code_encoder,train_sbt_encoder,decoder,transformer)
            #print(can)
            targ.write(can+'\n')
# end seq2trans集成

# bleu评分
def sentenceBleu(model):
    # 使用test数据集计算sentence_BLEU
    with open('dataset/source_code.txt', 'r') as s:
        slines = s.readlines()
        slines = slines[:20000]
    with open('dataset/comment.txt', 'r') as nl:
        lines = nl.readlines()
        lines = lines[:20000]

    if model=='seq2seq':
        with open("seq2seq/result.txt", 'r') as re:
            results = re.readlines()
    elif model=='transformer':
        with open("transformer/result.txt", 'r') as re:
            results = re.readlines()
    elif model=='seq2trans':
        with open("ensemble/result.txt", 'r') as re:
            results = re.readlines()
    else:
        print('model参数不对，请修改( seq2seq 或者 tranformer )')

    total_score = 0
    cc = SmoothingFunction()
    for i in tqdm(range(len(results))):
        # print('代码：',slines[i])
        # print('标准注释：',lines[i])
        # print('生成注释：',results[i])
        reference = lines[i].split()
        candidate = results[i].split()

        score = sentence_bleu(reference, candidate, smoothing_function=cc.method4)
        # print('BLEU评分',score)
        # print('\n')
        total_score += score

    # print(total_score)
    ava_score=total_score / len(results)
    #print(ava_score)
    return ava_score
