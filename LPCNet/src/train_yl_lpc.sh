#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python train_lpcnet.py ../yl_features.f32 ../yl_data.u8

