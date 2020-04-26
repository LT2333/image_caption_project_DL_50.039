# image_caption_project_DL_50.039
The Image Captioning project for SUTD 50.039 Deep Learning
<br>

*collaborators: Zhao Lutong (1002872), Tang Xiaoyue (1002968), Wang Zijia (1002885)*

# A video demo on our project
https://www.dropbox.com/sh/qfabcdkb6y6ulgq/AABfJeCEvYbmFCTgDyF2Ovgea?dl=0

# Reference 
Adapted code from:
https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning


# Image Captioning
The goal of image captioning is to convert a given input image into a natural language description. The encoder-decoder framework is widely used for this task. The image encoder is a convolutional neural network (CNN). In this tutorial, we used [resnet-152](https://arxiv.org/abs/1512.03385) model pretrained on the [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/) image classification dataset. The decoder is a GRU network. 

![alt text](png/model.png)

#### Training phase
For the encoder part, the pretrained CNN extracts the feature vector from a given input image. The feature vector is linearly transformed to have the same dimension as the input dimension of the GRU network. For the decoder part, source and target texts are predefined. For example, if the image description is **"Giraffes standing next to each other"**, the source sequence is a list containing **['\<start\>', 'Giraffes', 'standing', 'next', 'to', 'each', 'other']** and the target sequence is a list containing **['Giraffes', 'standing', 'next', 'to', 'each', 'other', '\<end\>']**. Using these source and target sequences and the feature vector, the GRU decoder is trained as a language model conditioned on the feature vector.

#### Test phase
In the test phase, the encoder part is almost same as the training phase. The only difference is that batchnorm layer uses moving average and variance instead of mini-batch statistics. This can be easily implemented using [encoder.eval()](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/sample.py#L37). For the decoder part, there is a significant difference between the training phase and the test phase. In the test phase, the GRU decoder can't see the image description. To deal with this problem, the GRU decoder feeds back the previosly generated word to the next input. This can be implemented using a for-loop.


## Usage 

#### 1) Using GUI
```bash
conda install kivy -c conda-forge  
python gui.py 
```
*If you counter problem of pygame on Mac system:*
    ```
    pip install pygame
    ``` 

#### 2) Test the model from the terminal

```bash
python sample.py --image='png/example.png'
```



<br>

## Prepare model

### Step 1. Prepare environment 
The package pycocotools requires cython and a C compiler to install correctly.
If you are using Linux or Mac system:
```bash
cd ../
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install
cd ../../
```

If you are using Windows, you can install pycocotools as follows:

```
pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
```

### Step 2. Option 1. Using Pretrained model
If you do not want to train the model from scratch, you can use our 3 trained models. You can download the pretrained model [here](https://sutdapac-my.sharepoint.com/personal/lutong_zhao_mymail_sutd_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flutong%5Fzhao%5Fmymail%5Fsutd%5Fedu%5Fsg%2FDocuments%2FDL%5FProject%2Fmodels&originalPath=aHR0cHM6Ly9zdXRkYXBhYy1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9sdXRvbmdfemhhb19teW1haWxfc3V0ZF9lZHVfc2cvRW9oMS1PTEZjN1pHcDBlQnV0Zk9VZVFCTDhfbDN0bzRCTGlVd0NkNDhhYjdJQT9ydGltZT15ZWhGN3dMcTEwZw) and the vocabulary file [here](https://sutdapac-my.sharepoint.com/personal/lutong_zhao_mymail_sutd_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flutong%5Fzhao%5Fmymail%5Fsutd%5Fedu%5Fsg%2FDocuments%2FDL%5FProject%2Fvocabs&originalPath=aHR0cHM6Ly9zdXRkYXBhYy1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9sdXRvbmdfemhhb19teW1haWxfc3V0ZF9lZHVfc2cvRXBTY2VYbThfdzVEZ0lnY29BMHRjN3NCYmE5c0xSWkVjX2k1djFJYnhudGpiZz9ydGltZT04Z0V5LXdMcTEwZw). You should save encoder and decoder files to `./models/` and vocab files to `./data/` folder.



<br>

### Step 2.  Option 2.Train from scratch

#### i). Download the dataset

```bash
pip install -r requirements.txt
chmod +x download.sh
./download.sh
```

#### ii). Preprocessing

```bash
python build_vocab.py   
python resize.py
```

#### iii). Train the model

```bash
python train.py    
```
<br>

###  Preview for our GUI
![Preview](png/group_people.png)
