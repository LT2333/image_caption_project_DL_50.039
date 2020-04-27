There are 6 models that trained from 3 experiments, you can choose to use any pair of encoder and ecoder :
    *Please put them under models folder. (e.g. image_caption_project_DL_50.039\models\my-encoder-5-3000-t4-resnext.ckpt)

    i) my-decoder-5-3000-t4.ckpt and my-encoder-5-3000-t4.ckpt
    This is trained from ResNet50 with batchnormalization after the linear layer in the decoder, the threshold for the vocab is 4.

    ii) my-decoder-5-2000-t10.ckpt and my-encoder-5-2000-t10.ckpt
    This is trained from ResNet50 with batchnormalization after the linear layer in the decoder, the threshold for the vocab is 10.

    iii) my-decoder-5-3000-t4-resnext.ckpt and my-encoder-5-3000-t4-resnext.ckpt  (recommended)
    This is trained from ResNext101_32x8d with dropout(0.2) layer before the last layer in the decoder, the threshold for the vocab is 4.

There are 2 vocab files that we generated using Snowball Stemmer:
    *Please put them under data folder. (e.g. image_caption_project_DL_50.039\data\vocab_stemmed_t4.pkl)

    i) vocab_stemmed_t4.pkl
        Generated with threshold = 4 (minimum word count = 4)
    ii) vocab_stemmed_t10.pkl
        Generated with threshold = 10 (minimum word count = 10)

Public accessible link for the models: 
https://sutdapac-my.sharepoint.com/:f:/g/personal/lutong_zhao_mymail_sutd_edu_sg/Eoh1-OLFc7ZGp0eButfOUeQBoNQuAnRpbO11Fs73gcbETg?e=NDsPff

Public accessible link for the vocabs:
https://sutdapac-my.sharepoint.com/:f:/g/personal/lutong_zhao_mymail_sutd_edu_sg/EpSceXm8_w5DgIgcoA0tc7sBba9sLRZEc_i5v1Ibxntjbg?e=qCS8rd