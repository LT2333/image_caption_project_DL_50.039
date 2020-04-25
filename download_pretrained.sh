mkdir data
wget https://www.dropbox.com/s/26adb7y9m98uisa/vocap.zip -P ./data/
unzip data/vocap.zip -d ./data/
rm data/vocap.zip

mkdir models
wget https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip -P ./models/
unzip models/pretrained_model.zip -d ./models/
rm models/pretrained_model.zip