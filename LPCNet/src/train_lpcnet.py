#!/usr/bin/python3
'''Copyright (c) 2018 Mozilla

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# Train a LPCNet model (note not a Wavenet model)

import lpcnet
import sys
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import Callback
from ulaw import ulaw2lin, lin2ulaw
import keras.backend as K
import h5py
from keras.callbacks import TensorBoard
import os as os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()

# use this option to reserve GPU memory, e.g. for running more than
# one thing at a time.  Best to disable for GPUs with small memory
config.gpu_options.per_process_gpu_memory_fraction = 0.9

set_session(tf.Session(config=config))

nb_epochs = 120

# Try reducing batch_size if you run out of memory on your GPU
batch_size  = 64
model, _, _ = lpcnet.new_lpcnet_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

feature_file = sys.argv[1]
pcm_file = sys.argv[2]     # 16 bit unsigned short PCM samples
frame_size = 160
nb_features = 55
nb_used_features = model.nb_used_features
feature_chunk_size = 15
pcm_chunk_size = frame_size*feature_chunk_size

# u for unquantised, load 16 bit PCM samples and convert to mu-law

data = np.fromfile(pcm_file, dtype='uint8')
nb_frames = len(data)//(4*pcm_chunk_size)

features = np.fromfile(feature_file, dtype='float32')

# limit to discrete number of frames
data = data[:nb_frames*4*pcm_chunk_size]
features = features[:nb_frames*feature_chunk_size*nb_features]

features = np.reshape(features, (nb_frames*feature_chunk_size, nb_features))

sig = np.reshape(data[0::4], (nb_frames, pcm_chunk_size, 1))
pred = np.reshape(data[1::4], (nb_frames, pcm_chunk_size, 1))
in_exc = np.reshape(data[2::4], (nb_frames, pcm_chunk_size, 1))
out_exc = np.reshape(data[3::4], (nb_frames, pcm_chunk_size, 1))
del data

print("ulaw std = ", np.std(out_exc))

features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
features = features[:, :, :nb_used_features]
features[:,:,18:36] = 0

periods = (.1 + 50*features[:,:,36:37]+100).astype('int16')

in_data = np.concatenate([sig, pred], axis=-1)
del sig
del pred

#save model
class save_model(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("epoch %d completed .."%epoch)
        if ((epoch+1) % 1) == 0:
            #val_loss = logs['val_loss']
            loss=logs['loss']
            modelfile = "model_loss-%.3f_%d_.hdf5"%(loss,epoch)
            #modelfile = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
            #modelfile =  "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
            model.save(modelfile)
        #ModelCheckpoint(filepath='/tmp/newweightsi_{epoch:02d}.hdf5', monitor='acc',verbose=1, period=1 ,save_best_only=False,mode='max')
        f= open("checkpoint","w+")
        f1= open("checkpoint.log","a")
        f1.write(modelfile)
        f1.write("\n")
        f.write(modelfile)
        f.close()
        f1.close()

checkpoint = save_model()

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

adaptation = False
if adaptation:
    f= open("checkpoint","r")
    latest_model = f.read().replace('\n', '')
    model.load_weights(latest_model)
    print("***Successfully loaded checkpoint %s"% latest_model)


#model.load_weights('lpcnet9b_384_10_G16_01.h5')
model.compile(optimizer=Adam(0.001, amsgrad=True, decay=5e-5), loss='sparse_categorical_crossentropy')
#model.fit([in_data, in_exc, features, periods], out_exc, batch_size=batch_size, epochs=nb_epochs, validation_split=0.0, callbacks=[checkpoint, lpcnet.Sparsify(2000, 40000, 400, (0.05, 0.05, 0.2)),TrainValTensorBoard(histogram_freq = 5)])

model.fit([in_data, in_exc, features, periods], out_exc, batch_size=batch_size, epochs=nb_epochs, validation_split=0.0, callbacks=[checkpoint, lpcnet.Sparsify(2000, 40000, 400, (0.05, 0.05, 0.2))])
