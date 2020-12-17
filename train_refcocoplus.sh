
vocab_path='./data/refcoco_vocab.pkl'
image_dir='./data/images/mscoco/'
refcoco_path='./data/refcoco+/'
batch_size=60
eval_size=28
model_name='adaptive-refcocoplus'

cd ..

python train_reg.py
  --vocab_path $vocab_path
  --image_dir $image_dir
  --refcoco_path $refcoco_path
  --batch_size $batch_size
  --eval_size $eval_size
  --model_name $model_name

cd bash
