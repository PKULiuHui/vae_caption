# vae_caption

coco_dataset：coco caption原始数据，以及Andrej Karpathy的数据切分方法。位于39服务器/home/liuhui/vae_caption/目录下

data：使用rnn_attn/create_input_files.py生成的数据，包含resize了的图片和caption

rnn_attn：使用LSTM + Image attention生成caption，并对attention权重进行可视化，参考[sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning#implementation)，做了点微调可以在pytorch1.3上运行了