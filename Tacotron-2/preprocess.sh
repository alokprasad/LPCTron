python3 preprocess.py --base_dir ../dataset --dataset LJSpeech-1.1
#this generates dataset/training_data folder 
#
#├── training_data (1)
#│   ├── audio
#│   ├── linear
#│   └── mels

#Now use LPCNET dump_data to convert PCM to NPY and place it in audio 
#Training TACATRON2 with lpcnet as Vocoder is not done with audio(dump_data PCM 0 and mels

#wav->s16->to npy as meta[0] audio file
#audio-LJ024-0006.npy 
#convert to PCM ( header removal)
mkdir -p ../dataset/LJSpeech-1.1/pcms
for i in ../dataset/LJSpeech-1.1/wavs/*.wav
do sox $i -r 16000 -c 1 -t sw - > ../dataset/LJSpeech-1.1/pcms/audio-$(basename "$i" | cut -d. -f1).s16
done

#f32/npy
mkdir -p ../dataset/LJSpeech-1.1/f32
for i in ../dataset/LJSpeech-1.1/pcms/*.s16
do
#(dump_data compileted with taco=1)
cd ../LPCNet
make clean && make dump_data taco=1
cd -
../LPCNet/dump_data -test $i ../dataset/LJSpeech-1.1/f32/$(basename "$i" | cut -d. -f1).npy
done

cp ../dataset/LJSpeech-1.1/f32/* ../dataset/training_data/audio/.
#copy f32 npy files to training_data/audio

python3 train.py --input_dir ../dataset/training_data --tacotron_input ../dataset/training_data/train.txt --model='Tacotron'
#models path
#/logs-Tacotron/taco_pretrained/*
