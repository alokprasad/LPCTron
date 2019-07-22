cd Tacotron-2
python3 synthesize.py --model='Tacotron' --mode='eval' --text_list='text.txt'
cd -
echo "Synthesizing Audio from Features..."
#(compile with make test_lpcnet taco=1)
cd LPCNet
#make clean && make test_lpcnet taco=1
cd -
echo "**************************************"
echo "Feature to PCM Conversion Started ..."
rm -rf test.s16 test.wav
time ./LPCNet/test_lpcnet Tacotron-2/f32_for_lpcnet.f32 test.s16
echo "**************************************"
echo "Converting PCM To wav..."
ffmpeg -f s16le -ar 16k -ac 1 -i test.s16 test.wav
