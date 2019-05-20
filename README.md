# LPCTron
Tacotron2 + LPCNET for complete End-to-End TTS System

Thanks to https://github.com/MlWoo/ for most of changes


#Training LPCNET with LJPSpeech
Since Tacatron2 is trained with LPSpeech its good idea to train with Same data. LPCNET uses a single Merged PCM 
for training. So you need to remove wav headers from each file and merge them for training.


