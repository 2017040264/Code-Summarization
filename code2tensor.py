# -*- coding = utf-8 -*-
# @Time : 2021/4/26 13:40
# @Author : 陈凡亮
# @File : code2tensor.py
# @Software : PyCharm

import tensorflow as tf

# 从文件中读取内容
def creat_dataset(a, b, c):
    # a : code
    # b: sbt
    # c: nl
    with open(a, encoding='utf-8') as tc:
        lines1 = tc.readlines()
        for i in range(len(lines1)):
            lines1[i] = "<start> " + lines1[i].strip('\n') + " <end>"
    with open(b, encoding='utf-8') as ts:
        lines2 = ts.readlines()
        for i in range(len(lines2)):
            lines2[i] = "<start> " + lines2[i].strip('\n') + " <end>"
    with open(c, encoding="utf-8") as tn:
        lines3 = tn.readlines()
        for i in range(len(lines3)):
            lines3[i] = "<start> " + lines3[i].strip('\n') + " <end>"

    if (len(lines1) != len(lines2) or len(lines1) != len(lines3) or len(lines2) != len(lines3)):
        print("数据量不匹配")
    # else:
    # print(len(lines1))
    return lines1, lines2, lines3


# 将text转为tensor
def tokenize(vocab_maxlen,lang, maxlen):
    # lang:文本
    # maxlen: 最大长度  code与sbt是500，nl是30

    # 1. 创建分词器
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_maxlen + 1, filters='', oov_token="<unk>")
    # 2. 将文本内容添加进来
    lang_tokenizer.fit_on_texts(lang)
    # 3. 将词列表转换为词索引表
    tensor = lang_tokenizer.texts_to_sequences(lang)
    # 进行末尾pad,在末尾进行截断（保留前面的）
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=maxlen, padding='post', truncating='post')

    return tensor, lang_tokenizer


# 将tensor进行组合[[code,sbt],[code,sbt],...]
def seq2seq_getTensor(code_tensor,sbt_tensor,nl_tensor):
    code_sbt_tensor = tf.stack([code_tensor, sbt_tensor], axis=1)

    test_code_sbt_tensor = code_sbt_tensor[:20000]
    val_code_sbt_tensor = code_sbt_tensor[20000:40000]
    train_code_sbt_tensor = code_sbt_tensor[40000:]

    test_nl_tensor = nl_tensor[:20000]
    val_nl_tensor = nl_tensor[20000:40000]
    train_nl_tensor = nl_tensor[40000:]
    return test_code_sbt_tensor,val_code_sbt_tensor,train_code_sbt_tensor,test_nl_tensor,val_nl_tensor,train_nl_tensor


# 获取测试tensor
def seq2seq_getTestTensor(code_tensor,sbt_tensor):
    return code_tensor[:20000],sbt_tensor[:20000]

def tranformer_getTestTensor(code_tensor):
    return code_tensor[:20000]