# image_caption_project_DL_50.039
The Image Captioning project for SUTD 50.039 Deep Learning
<br>

*collaborators: Zhao Lutong (1002872), Tang Xiaoyue (1002968), Wang Zijia (1002885)*

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
