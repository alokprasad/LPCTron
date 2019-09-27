# LPCTron
Tacotron2 + LPCNET for complete End-to-End TTS System

Thanks to https://github.com/MlWoo/ for most of changes
( Check this issue Also https://github.com/MlWoo/LPCNet/issues/4)

## Prerequisites
librosa , tqdm , matplotlib, lws , unidecode , inflect, falcon, scipy , numpy, keras

sudo apt-get install python-pyaudio
sudo apt-get install portaudio19-dev

## Quick Run

Checkout the LPCTron , and run tts.sh it converts text_file ( Tacotron2/text.txt ) to test.wav 


## Steps of Integration

* Checkout Tacatron2 (https://github.com/alokprasad/LPCTron/blob/master/Tacotron-2/Tacotron2-lpcnet_changes.diff)



## Training Tacotron2 specially to be used by LPCNET vocoder ( Instead of Wavenet Vocoder)

```
python3 preprocess.py --base_dir /media/alok/ws/sandbox/lpc_tacatron2/dataset --dataset LJSpeech-1.1
```

this will generate /dataset/training_data folder 

```
├── training_data (1)
│   ├── audio (*)
│   ├── linear
│   └── mels
    └── train.txt

*all three folder contain npy array, train.txt contains text for each audio.
```
* this is replaced by script for training to be used by LPCNET

## Training LPCNET with LJPSpeech
Since Tacatron2 is trained with LPSpeech its good idea to train with Same data. LPCNET uses a single Merged PCM 
for training. So you need to remove wav headers from each file and merge them for training.

===============================================================================


sudo apt-get install python3-tk 

pip3 install lws --user

python -m pip uninstall pip && sudo apt install python3-pip --reinstall


python3 train.py --input_dir ../dataset/training_data --tacotron_input ../dataset/training_data/train.txt --model='Tacotron'

python3 synthesize.py --model='Tacotron' --mode='eval'


sudo apt-get install python3-tk 

pip3 install lws --user
python -m pip uninstall pip && sudo apt install python3-pip --reinstall

Download
https://jmvalin.ca/misc_stuff/lpcnet_models/lpcnet15_384_10_G16_100.h5

apt-get install -y google-perftools
LD_PRELOAD = /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

python3 train.py --input_dir ../dataset/training_data --tacotron_input ../dataset/training_data/train.txt --model='Tacotron'

python3 synthesize.py --model='Tacotron' --mode='eval'

./test_lpcnet f32_for_lpcnet.f32 test.s16

ffmpeg -f s16le -ar 16k -ac 1 -i test.s16 test-out.wav



