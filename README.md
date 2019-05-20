# LPCTron
Tacotron2 + LPCNET for complete End-to-End TTS System

Thanks to https://github.com/MlWoo/ for most of changes

## Steps of Integration

* Checkout Tacatron2 (Was facing some sihttps://github.com/alokprasad/LPCTron/blob/master/Tacotron-2/Tacotron2-lpcnet_changes.diff



## Training Tacotron2 specially to be used by LPCNET vocoder ( Instead of Wavenet Vocoder)

python3 preprocess.py --base_dir /media/alok/ws/sandbox/lpc_tacatron2/dataset --dataset LJSpeech-1.1
this will generate /dataset/training_data folder 
```
├── training_data (1)
│   ├── audio (*)
│   ├── linear
│   └── mels
```
* this is replaced by script for training to be used by LPCNET

## Training LPCNET with LJPSpeech
Since Tacatron2 is trained with LPSpeech its good idea to train with Same data. LPCNET uses a single Merged PCM 
for training. So you need to remove wav headers from each file and merge them for training.






============================================================================================================================
sudo apt-get install python3-tk 

pip3 install lws --user

python -m pip uninstall pip && sudo apt install python3-pip --reinstall
