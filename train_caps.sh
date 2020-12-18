#!/bin/bash
#SBATCH --job-name=train_captioning_model
#SBATCH --partition=gpu_test
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

module load nvidia/cuda/10.1.168
module load nvidia/cudnn/7.5.1.10
module load tools/python/3.7

echo "running on node $HOSTNAME"

vocab_path='./data/coco_vocab.pkl'
image_dir='./data/images/mscoco/'
caption_path='./data/captions/'
splits_path='./data/splits/karpathy/caption_datasets/'
batch_size=60
eval_size=28
model_name='adaptive-captions'

srun python train_caption.py --vocab_path $vocab_path --image_dir $image_dir --caption_path $caption_path --splits_path $splits_path --batch_size $batch_size --eval_size $eval_size --model_name $model_name
