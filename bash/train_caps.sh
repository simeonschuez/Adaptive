
vocab_path='/home/simeon/Dokumente/Code/Uni/Repos/Adaptive/data/coco_vocab.pkl'
image_dir='/home/simeon/Dokumente/Code/Data/COCO/'
caption_path='/home/simeon/Dokumente/Code/Data/COCO/'
splits_path='/home/simeon/Dokumente/Code/Data/COCO/splits/karpathy/caption_datasets/'
batch_size=60
eval_size=28
model_name='adaptive-captions'

cd ..

python train_caption.py
  --vocab_path $vocab_path
  --image_dir $image_dir
  --caption_path $caption_path
  --splits_path $splits_path
  --batch_size $batch_size
  --eval_size $eval_size
  --model_name $model_name

cd bash
