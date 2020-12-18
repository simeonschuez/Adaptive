#!/bin/bash
module load nvidia/cuda/10.1.168
module load nvidia/cudnn/7.5.1.10
module load tools/python/3.7

echo "running on node $HOSTNAME"

vocab_path='./data/refcoco_vocab.pkl'
image_dir='./data/images/mscoco/'
refcoco_path='./data/refcoco/'
batch_size=60
eval_size=28
model_name='adaptive-refcoco'

python train_reg.py --num_epochs 3 --vocab_path $vocab_path --image_dir $image_dir --refcoco_path $refcoco_path --batch_size $batch_size --eval_size $eval_size --model_name $model_name
