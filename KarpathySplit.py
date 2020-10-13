import json
from os import makedirs
from os.path import join, isdir
import argparse

def main(args):

    # make output dir if it doesn't exist yet
    if not isdir(args.out_dir):
        makedirs(args.out_dir)
        print('created output directory: {}'.format(args.out_dir))

    print('load Karpathy splits...')
    with open(args.karpathy_path) as file:
        k_data = json.load(file)['images']

    # get corresponding ids for splits
    train_ids = [i['cocoid'] for i in k_data if i['split'] == 'train']
    restval_ids = [i['cocoid'] for i in k_data if i['split'] == 'restval']
    train_ids = train_ids + restval_ids  # add restval ids to train_ids
    val_ids = [i['cocoid'] for i in k_data if i['split'] == 'val']
    test_ids = [i['cocoid'] for i in k_data if i['split'] == 'test']

    print('# images: \ntrain: {}, val: {}, test: {}'.format(
            len(train_ids), len(val_ids), len(test_ids)
        ))

    print('load coco data...')
    val = json.load(
        open(join(args.coco_dir, 'captions_val2014.json'), 'r')
        )
    train = json.load(
        open(join(args.coco_dir, 'captions_train2014.json'), 'r')
        )

    # Merge together
    imgs = val['images'] + train['images']
    annots = val['annotations'] + train['annotations']

    print('split images into val, test, train...')
    dataset = {}
    dataset['val'] = [i for i in imgs if i['id'] in val_ids]
    dataset['test'] = [i for i in imgs if i['id'] in test_ids]
    dataset['train'] = [i for i in imgs if i['id'] in train_ids]

    # setup json data
    json_data = {}
    info = train['info']
    licenses = train['licenses']

    # Group annotations by image ids
    itoa = {}
    for a in annots:
        imgid = a['image_id']
        if imgid not in itoa:
            itoa[imgid] = []
        itoa[imgid].append(a)

    for subset in ['val', 'test', 'train']:

        json_data[subset] = {
            'type': 'caption', 'info': info, 'licenses': licenses,
            'images': [], 'annotations': []
            }

        for img in dataset[subset]:

            img_id = img['id']
            anns = itoa[img_id]

            json_data[subset]['images'].append(img)
            json_data[subset]['annotations'].extend(anns)

        print('{} split: {} images | {} captions'.format(
            subset,
            len(json_data[subset]['images']),
            len(json_data[subset]['annotations'])
            ))

        out_path = join(args.out_dir, 'karpathy_split_'+subset+'.json')
        json.dump(json_data[subset], open(out_path, 'w'))
        print('saved to file', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', default='self', help='To make it runnable in jupyter'
        )
    parser.add_argument(
        '--coco_dir', type=str, help='path to coco annotations'
        )
    parser.add_argument(
        '--karpathy_path', type=str, help='path to Karpathy split file',
        default='./data/coco_splits.json'
        )

    parser.add_argument(
        '--out_dir', type=str, help='directory for output data',
        default='./data/annotations'
    )

    args = parser.parse_args()

    main(args)
