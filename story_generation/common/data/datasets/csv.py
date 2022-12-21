import random
import os

import pandas as pd

from story_generation.common.data.datasets.abstract_dataset import Dataset
from story_generation.common.data.split_paragraphs import split_texts


class CSVDataset(Dataset):
    def __init__(self, args):
        print('loading data')
        random.seed(args.seed)
        self.args = args
        self.debug = args.debug
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.splits = {}
        df = pd.read_csv(args.data_dir, delimiter=',', quotechar='"', skipinitialspace=True)
        all_examples = [text.strip() for text in getattr(df, args.csv_column).tolist() if type(text) == str][:args.limit]
        train_end = int(len(all_examples) * args.split_sizes[0])
        valid_end = int(len(all_examples) * (args.split_sizes[0] + args.split_sizes[1]))
        self.splits['train'] = all_examples[:train_end]
        self.splits['valid'] = all_examples[train_end:valid_end]
        self.splits['test'] = all_examples[valid_end:]

        print('done loading data')
        print('split sizes:')
        for key in ['train', 'valid', 'test']:
            print(key, len(self.splits[key]))

    def load_long_texts(self, split='train', limit=None):
        texts = self.splits[split]
        return split_texts(texts if limit is None else texts[:limit], mode='none')
        
    def load_short_texts(self, split='train', limit=None):
        texts = self.splits[split]
        return split_texts(texts if limit is None else texts[:limit], mode='none')
        
    def pandas_format(self, split, long_name='content', short_name='title', limit=None):
        raise NotImplementedError

    def shuffle(self, split, seed=None):
        assert split in ['train', 'valid', 'test']
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.splits[split])

