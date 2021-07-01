#mk directories for the dataset
mkdir -p mmca/{images,audio,train,test}
mkdir -p mmca/audio/{mel,wav}

#download the filenames.pick for the train and test partitions
cd mmca/train
gdown https://drive.google.com/uc?id=1GdeTdBpi_IV7AuBpJAhLElqjswRmOy-7

cd ../test
gdown https://drive.google.com/uc?id=1JNxgdvPMI_HHUq2-JUuJp8L7cD-74OAf

#download the text files for the image captions
cd ../
gdown https://drive.google.com/uc?id=1X1EFCyralNN2Bg3LhelL_lShrSrmTitW
unzip text.zip
rm text.zip

#download the image files and only keep the .jpg image files in mmca/images
gdown https://drive.google.com/uc?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv
unzip CelebAMask-HQ.zip
rm CelebAMask-HQ.zip
mv CelebAMask-HQ/CelebA-HQ-img/*.jpg images/
rm -r CelebAMask-HQ
