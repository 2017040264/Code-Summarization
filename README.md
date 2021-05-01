# Code-Summarization
My graduation task.The goal is : Give you code,give me comments.

# 文件说明
## dataset
该文件夹下数据集处理文件，因为数据集过大(>25MB)，数据集无法上传至github,可以参考由此链接获取：
https://github.com/xing-hu/EMSE-DeepCom
该链接指向的是清华大学学者的github项目地址。可在他们项目中获取数据集(他们将数据集放到了谷歌云盘里面，给出了链接)。

也可以访问AI Studio 上面我建立的数据集：https://aistudio.baidu.com/aistudio/datasetdetail/73043

## seq2seq
该文件夹存放使用训练好的seq2seq模型得到的测试结果
checkpoints获取路径：https://www.kaggle.com/chenfanliang/seq2seq

## seq2seq_onlycode
该文件夹存放使用训练好的seq2seq_onlycode模型得到的测试结果
checkpoints获取路径：https://www.kaggle.com/chenfanliang/seq2seq-onlywith-code

## tranformer
该问价夹存放使用训练好的transformer模型得到的预测结果
checkpoints获取路径：https://www.kaggle.com/chenfanliang/transfor

## ensembel(经过测试，这种思路并未得到一个优于两个模型的结果，均介于二者之间)
该文件夹存放了seq2trans模型的预测结果。所谓的seq2trans模型，就是seq2seq+transformer模型的合体，在预测的时候，
二者同时预测，取概率最大的的结果。

举例：
code: public synchronized void info ( string msg ) { log record record = new log record ( level . info , msg ) ; log ( record ) ; }

预测时，解码器第一个输入为<start>,然后seq2seq模型预测结果的最大值概率为0.6，下标为10；而tranformer模型的预测结果最大值概率为0.7，下标为11.
那么，如果 ws*0.6>=wt*0.7(如果ws>wt,反之不加等号)，我们选择下标10，反之选择下标11。以此类推。

## code2tensor.py
将code转成tensor

## main.py
使用模型进行预测

## seq2seq.py
搭建seq2seq模型。因为保存模型使用的是checkpoints，所以需要原始的网络模型进行复原

##seq2seq_onlycode.py
搭建不适用AST的seq2seq模型。

## transformer.py
搭建transformer模型。

## eval.py
生成预测结果、BLEU评分。