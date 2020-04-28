from torchtext.data import Dataset
from torchtext.data import Example
from torchtext.vocab import Vectors

from torchtext import data
import torch
import json
import os
from collections import Counter

from spacy.lang.en.stop_words import STOP_WORDS
import spacy

import envs


stop_words = [x for x in STOP_WORDS]

# ignore _ and eou, they are used as token to indicate one person ends the talk
stop_words.extend(["_", "eou"]) 


TEXT = data.Field(tokenize='spacy', sequential=True,
        stop_words=stop_words, include_lengths=True)

LABEL = data.LabelField(dtype=torch.float, sequential=False)


fields = {
        'text': ('text', TEXT),
        'label': ('label', LABEL)
    }

class DailyDialog(Dataset):
    def __init__(self, path, text_field, label_field, samples=None, cap=None):
        fields = {
            'text': ('text', TEXT),
            'label': ('label', LABEL)
            }
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if samples:
            examples = []
            for ix in samples:
                examples.append(Example.fromdict(data[ix], fields))
        else:
            examples = []
            for d in data:
                examples.append(Example.fromdict(d, fields))
        if cap:
            if not isinstance(cap, int):
                raise("cap needs to be an instance of int, got {}".format(cap))
            if cap < len(examples):
                examples = examples[:cap]
        if isinstance(fields, dict):
            fields, fields_dict = [], fields
            for field in fields_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)
                    
        super(DailyDialog, self).__init__(examples, fields)


        
entire_data = DailyDialog(path=os.path.join(envs.DATA_DIR,
    'train.json'), text_field=TEXT, 
    label_field=LABEL, samples=None, cap=None)


vectors = Vectors(name='glove.6B.100d.txt', cache=os.path.join(
    envs.VECTOR_DIR, 'glove'))


TEXT.build_vocab(entire_data, vectors=vectors)
LABEL.build_vocab(entire_data)


def write_class_weight():
    labels = [x.label for x in entire_data]
    cnt = Counter(labels)
    class_weights = []
    for i in range(len(cnt)):
        class_weights.append(cnt[i])
    return class_weights

# print(write_class_weight())
