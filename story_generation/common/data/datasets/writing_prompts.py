import random
import os
import string
import multiprocessing as mp

from transformers import BartTokenizerFast
import pandas as pd

from story_generation.common.data.datasets.abstract_dataset import Dataset
from story_generation.common.data.split_paragraphs import split_texts


def preprocess(texts):
    # remaining known edge cases: 
    # people who use ' as quotation marks (definitely not me)
    all_fixed = []
    for text in texts:
        if text.startswith('['):
            text = ']'.join(text.split(']')[1:]) # remove leading brackets
        fixed = ''
        text = text.replace('<newline> <newline>', '<newline>')
        text = text.replace(u'\u2018', "'").replace(u'\u2019', "'")
        text = text.replace(u'\u201d', '').replace(u'\u201c', '')
        while '  ' in text:
            text = text.replace('  ', ' ')
        text = text.replace('``', '"')
        text = text.replace("''", '"')
        tokens = text.split()
        fixed = ''
        in_quotes = False
        for tok in tokens:
            if tok == '<newline>':
                fixed += '\n'
            elif '<newline>' in tok:
                print(tok)
            elif tok.startswith('"') and not in_quotes:
                fixed += ' ' + tok
            elif all([c in string.punctuation for c in tok]):
                fixed += tok
            elif tok.startswith("'"):
                fixed += tok
            elif tok.startswith("n't"):
                fixed += tok
            # https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
            elif fixed.endswith("'") and tok in ['s', 't', 'll', 'd', 're', 'm', 'n', 've', 'cause', 'cept', 'ight', 'bout', 'ye', 'en', 'er', 'em', 'gainst', 'day', 'am', 'neath', 'clock', 'round', 'til', 'tis', 'tween', 'twere', 'twas', 'all', 'know']:
                fixed += tok
            else:
                if fixed.endswith('"') and in_quotes:
                    fixed += tok
                else:
                    fixed += ' ' + tok
            if '"' in tok:
                in_quotes = not in_quotes
        fixed = fixed.replace('( ', ' (').replace('[ ', ' [').replace('{ ', ' {')
        all_fixed.append(fixed)
    return tuple(all_fixed)


class WritingPromptsDataset(Dataset):
    def __init__(self, args):
        print('loading data')
        random.seed(args.seed)
        self.args = args
        self.debug = args.debug
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        
        tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')
        self.splits = {}
        for split in ['train', 'valid', 'test']:
            self.splits[split] = []
            with open(os.path.join(args.data_dir, split + '.wp_target'), 'r') as rf1, \
                open(os.path.join(args.data_dir, split + '.wp_source'), 'r') as rf2:
                contents = [line for line in rf1]
                summaries = [line for line in rf2]
                assert len(contents) == len(summaries)
                tokenized_contents = tokenizer.batch_encode_plus(contents, max_length=args.length_limit+1, truncation=True)['input_ids']
                tokenized_summaries = tokenizer.batch_encode_plus(summaries, max_length=min(args.length_limit, args.summary_length_limit)+1, truncation=True)['input_ids']
                for i in range(len(contents)):
                    tokenized_content = tokenized_contents[i]
                    if len(tokenized_content) > args.length_limit or len(tokenized_content) < args.lower_length_limit:
                        continue
                    tokenized_summary = tokenized_summaries[i]
                    if len(tokenized_summary) > min(args.summary_length_limit, args.length_limit):
                        continue
                    content, summary = contents[i], summaries[i]
                    self.splits[split].append((content.strip(), summary.strip()))
                    if args.limit is not None and len(self.splits[split]) >= args.limit:
                        break
                    if args.debug and len(self.splits[split]) >= 10:
                        break
        os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid warnings later
        for split in ['train', 'valid', 'test']:
            with mp.Pool(20) as pool:
                self.splits[split] = pool.map(preprocess, self.splits[split])
            
        print('done loading data')
        print('split sizes:')
        for key in ['train', 'valid', 'test']:
            print(key, len(self.splits[key]))

    def load_long_texts(self, split='train', limit=None):
        texts = [d[0] for d in self.splits[split]]
        return split_texts(texts if limit is None else texts[:limit], mode='none')
        
    def load_short_texts(self, split='train', limit=None):
        texts = [d[1] for d in self.splits[split]]
        texts = split_texts(texts if limit is None else texts[:limit], mode='none')
        return texts

    def pandas_format(self, split, long_name='content', short_name='title', limit=None):
        pandas_data = self.splits[split]
        if limit is not None:
            pandas_data = pandas_data[:limit]
        return pd.DataFrame(pandas_data, columns=[long_name, short_name])

    def shuffle(self, split, seed=None):
        assert split in ['train', 'valid', 'test']
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.splits[split])