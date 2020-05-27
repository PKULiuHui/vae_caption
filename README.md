# vae_caption

本文件是基于CVAE的Image Caption任务的项目。

## 1.Data
此Project基于Mscoco数据集进行，其中coco_dataset：coco caption原始数据，我们采用Andrej Karpathy的数据切分方法。位于39服务器/home/liuhui/vae_caption/目录下。
data：使用rnn_attn/create_input_files.py生成的数据，包含resize了的图片和caption

## 2.Code
本文件包含不同的对照试验，其中包含基于RNN的image caption模型，基于RNN的CVAE模型，基于Transformer的CVAE模型以及对CVAE模型中间隐变量进行重构的实验。本项目参考[sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning#implementation)。
其中包含的文件夹说明如下：
rnn_attn：使用LSTM + Image attention生成caption，并对attention权重进行可视化，
rnn_cvae：使用LSTM+CVAE对Image Caption进行生成的Model，其中包含CNN Fine-tune的部分。
T_cvae：使用Transformer + Image attention生成caption。 使用beam search, 其中beam-width = 10.
rnn_cvae_z和T_cvae_z：对rnn+cvae和Transformer+cvae的隐变量进行重构和MSE loss的实验代码。


