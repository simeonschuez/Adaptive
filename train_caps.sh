#!/bin/bash

vocab_path='./data/coco_vocab.pkl'
image_dir='./data/images/mscoco/'
caption_path='./data/captions/annotations/'
splits_path='./data/splits/'
batch_size=60
eval_size=28
model_name='adaptive-captions'

python train_caption.py --vocab_path $vocab_path --image_dir $image_dir --caption_path $caption_path --splits_path $splits_path --batch_size $batch_size --eval_size $eval_size --model_name $model_name
