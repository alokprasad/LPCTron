import os
import numpy as np
import tensorflow as tf
from librosa import effects
from tacotron.models import create_model
from tacotron.utils.text import text_to_sequence
from tacotron.utils import plot
from datasets import audio
from datetime import datetime
import sounddevice as sd
import pyaudio
import wave
from infolog import log


class Synthesizer:
    def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron'):
        log('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
        targets = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'mel_targets')
        with tf.variable_scope('model') as scope:
            self.model = create_model(model_name, hparams)
            if gta:
                self.model.initialize(inputs, input_lengths, targets, gta=gta)
            else:        
                self.model.initialize(inputs, input_lengths)
            self.mel_outputs = self.model.mel_outputs
            self.alignment = self.model.alignments[0]

        self.gta = gta
        self._hparams = hparams

        log('Loading checkpoint: %s' % checkpoint_path)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)


    def synthesize(self, text, index, out_dir, log_dir, mel_filename):
        hparams = self._hparams
        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        seq = text_to_sequence(text, cleaner_names)
        feed_dict = {
            self.model.inputs: [np.asarray(seq, dtype=np.int32)],
            self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
        }

        if self.gta:
            feed_dict[self.model.mel_targets] = np.load(mel_filename).reshape(1, -1, 80)

        if self.gta or not hparams.predict_linear:
            mels, alignment = self.session.run([self.mel_outputs, self.alignment], feed_dict=feed_dict)

        else:
            linear, mels, alignment = self.session.run([self.linear_outputs, self.mel_outputs, self.alignment], feed_dict=feed_dict)
            linear = linear.reshape(-1, hparams.num_freq)

        mels = mels.reshape(-1, hparams.num_mels) #Thanks to @imdatsolak for pointing this out

        #convert checkpoint to frozen model
        minimal_graph = tf.graph_util.convert_variables_to_constants(self.session, self.session.graph_def, ["model/inference/add"])
        tf.train.write_graph(minimal_graph, '.', 'inference_model.pb', as_text=False)

        npy_data = mels.reshape((-1,))
        print(mels)
        print("==============================================")
        print(npy_data)
        npy_data.tofile("f32_for_lpcnet.f32")

        return
