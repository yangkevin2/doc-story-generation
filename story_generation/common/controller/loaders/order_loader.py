import random

import numpy as np
import torch
from transformers import AutoTokenizer

from story_generation.common.data.split_paragraphs import split_paragraphs

class OrderSplitLoader(torch.utils.data.IterableDataset):
    def __init__(self, contents, summaries, tokenizer_model, append_mask_token=False, **kwargs):
        super(OrderSplitLoader).__init__()
        if append_mask_token:
            raise NotImplementedError
        self.contents = contents
        self.tokenizer_model = tokenizer_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.pos = 0
        self.eval = kwargs['eval'] if 'eval' in kwargs else False
            

    def __len__(self):
        return len(self.contents)

    def __iter__(self):
        return self
    
    def __next__(self):
        increment = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None: # # in a worker process
            increment = worker_info.num_workers
            worker_id = worker_info.id
            if self.pos == 0:
                self.pos = worker_id
        valid = False
        while not valid:
            if self.pos >= len(self.contents):
                raise StopIteration
            if self.eval:
                random.seed(self.pos)
            content = self.contents[self.pos].split('\t')
            if len(content) < 2:
                content = split_paragraphs(self.contents[self.pos], mode='sentence')
                if len(content) < 2:
                    self.pos += increment
                    continue

            idx0 = random.choice(range(len(content)))
            content0 = content[idx0]
            example = {}
            copy_content = [s.strip() for s in content]
            if random.random() < 0.5: # wrong order, label 0
                copy_content = [s for i, s in enumerate(content) if i != idx0]
                insert_idx = random.choice(range(len(copy_content)+1))
                while insert_idx == idx0:
                    insert_idx = random.choice(range(len(copy_content)+1))
                copy_content.insert(insert_idx, '*' + content0 + '*') # use * to mark which sentence we're focusing on, since in practice we'll be looking at a specific sentence
                example['prefix'] = self.tokenizer.encode('\n\n'.join(copy_content))
                example['labels'] = np.array([0])
            else: # correct order, label 1
                copy_content[idx0] = '*' + content0 + '*'
                example['prefix'] = self.tokenizer.encode('\n\n'.join(copy_content))
                example['labels'] = np.array([1])

            valid = True
            self.pos += increment
        return [example]
    