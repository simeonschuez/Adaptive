
vocab_path='/home/simeon/Dokumente/Code/Uni/Repos/Adaptive/data/refcoco_vocab.pkl'
image_dir='/home/simeon/Dokumente/Code/Uni/Repos/Adaptive/data/images/mscoco/'
refcoco_path='/home/simeon/Dokumente/Code/Uni/Repos/Adaptive/data/refcoco/'
batch_size=60
eval_size=28
model_name='adaptive-refcoco'

cd ..

python train_reg.py
  --vocab_path $vocab_path
  --image_dir $image_dir
  --refcoco_path $refcoco_path
  --batch_size $batch_size
  --eval_size $eval_size
  --model_name $model_name

cd bash
