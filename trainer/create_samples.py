import os

import pandas as pd
from shutil import copyfile


def run():
    test = pd.read_csv('../input/sample_submission.csv')
    train = pd.read_csv('../input/train.csv')

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index('fname', inplace=True)
    test.set_index('fname', inplace=True)
    train['label_idx'] = train.label.apply(lambda elem: label_idx[elem])
    for key, value in train.label.items():
        if not os.path.exists('../Samples/'+value+'/'):
            os.makedirs('../Samples/'+value+'/')
        copyfile('../input/audio_train/'+key, '../Samples/'+value+'/'+key)


run()
print("done")