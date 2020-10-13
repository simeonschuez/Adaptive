wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip ./

unzip -p ./caption_datasets.zip dataset_coco.json > ./coco_splits.json
rm ./caption_datasets.zip
