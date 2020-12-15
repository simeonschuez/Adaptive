import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'nlg-eval'))

import string
from nlgeval import NLGEval
from torch.autograd import Variable
import torch


def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def no_punct(s):
    return s.translate(str.maketrans('', '', string.punctuation))


def eval_captioning_model(model, vocab, eval_data_loader, eval_df):

    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)  # loads the models

    model.eval()

    hypotheses = []
    references = []
    print('---------------------Start evaluation on MS-COCO dataset-----------------------')

    for i, (images, _, _, image_ids, _) in enumerate(eval_data_loader):

        images = to_var(images)

        # get token ids from model sampler
        generated_captions, _, _ = model.sampler(images)

        # move generated captions to device
        if torch.cuda.is_available():
            captions = generated_captions.cpu().data.numpy()
        else:
            captions = generated_captions.data.numpy()

        # Build caption based on Vocabulary and the '<end>' token
        for image_idx in range(captions.shape[0]):

            img_id = int(image_ids[image_idx])  # the current image id

            sampled_ids = captions[image_idx]  # the current caption

            # transform caption ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                else:
                    sampled_caption.append(word)

            # combine words and add to hypotheses list
            sentence = ' '.join(sampled_caption)
            hypotheses.append(sentence)

            # add all (normalized) reference captions as list to references list
            refs = eval_df.loc[eval_df.image_id == img_id].caption\
                .map(str.strip).map(str.lower).map(no_punct)\
                .to_list()
            references.append(refs)

    # calculate cider score from hypotheses and references
    metrics_dict = nlgeval.compute_metrics(
        ref_list=references, hyp_list=hypotheses)
    cider = metrics_dict['CIDEr']

    return cider


def eval_reg_model(model, vocab, eval_data_loader, eval_df):

    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)  # loads the models

    model.eval()
    # Generated captions to be compared with GT
    hypotheses = []
    references = []
    print('---------------------Start evaluation on MS-COCO dataset-----------------------')

    for i, (images, _, location_features, _, ann_ids, _) in enumerate(eval_data_loader):

        images = to_var(images)
        generated_captions, _, _ = model.sampler(images, location_features)

        if torch.cuda.is_available():
            captions = generated_captions.cpu().data.numpy()
        else:
            captions = generated_captions.data.numpy()

        # Build caption based on Vocabulary and the '<end>' token
        for image_idx in range(captions.shape[0]):

            ann_id = int(ann_ids[image_idx])

            sampled_ids = captions[image_idx]
            sampled_caption = []

            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                else:
                    sampled_caption.append(word)

            sentence = ' '.join(sampled_caption)
            hypotheses.append(sentence)

            refs = eval_df.loc[eval_df.ann_id == ann_id].caption\
                .map(str.strip).map(str.lower).map(no_punct)\
                .to_list()

            references.append(refs)

    # calculate cider score from hypotheses and references
    metrics_dict = nlgeval.compute_metrics(
        ref_list=references, hyp_list=hypotheses)
    cider = metrics_dict['CIDEr']

    return cider
