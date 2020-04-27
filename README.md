# image_caption_project_DL_50.039
The Image Captioning project for SUTD 50.039 Deep Learning
<br>

*Collaborators: Tang Xiaoyue (1002968), Wang Zijia (1002885), Zhao Lutong (1002872) (alphabetical order)*

# A video demo on our project
https://www.dropbox.com/sh/qfabcdkb6y6ulgq/AABfJeCEvYbmFCTgDyF2Ovgea?dl=0


# Prepare model

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
If you do not want to train the model from scratch, you can use our 3 trained models. You can download the pretrained model [here](https://sutdapac-my.sharepoint.com/:f:/g/personal/lutong_zhao_mymail_sutd_edu_sg/Eoh1-OLFc7ZGp0eButfOUeQBoNQuAnRpbO11Fs73gcbETg?e=NDsPff) and the vocabulary file [here](https://sutdapac-my.sharepoint.com/:f:/g/personal/lutong_zhao_mymail_sutd_edu_sg/EpSceXm8_w5DgIgcoA0tc7sBba9sLRZEc_i5v1Ibxntjbg?e=qCS8rd). You should save encoder and decoder files to `./models/` and vocab files to `./data/` folder.

For detailed information about this section, please read [readme_models.txt](readme_models.txt)

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

# Usage 

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

###  Preview for our GUI
![Preview](png/group_people.png)



# Reference 
Adapted code from:
https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
