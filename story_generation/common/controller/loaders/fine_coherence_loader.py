import random

import numpy as np
import torch
from transformers import AutoTokenizer

from story_generation.common.data.split_paragraphs import split_paragraphs, group_chunks
from story_generation.common.util import *

class FineCoherenceSplitLoader(torch.utils.data.IterableDataset):
    def __init__(self, contents, summaries, tokenizer_model, append_mask_token=False, **kwargs):
        super(FineCoherenceSplitLoader).__init__()
        if append_mask_token:
            raise NotImplementedError
        assert len(contents) == len(summaries)
        self.contents = contents
        self.summaries = summaries
        self.tokenizer_model = tokenizer_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.append_mask_token = append_mask_token
        self.tokenized_info = kwargs['tokenized_info'] if 'tokenized_info' in kwargs else False
        self.negative_categories = kwargs['negative_categories'] if 'negative_categories' in kwargs else ['other', 'repeat', 'shuffle']
        self.generate_negatives = kwargs['generate_negatives'] if 'generate_negatives' in kwargs else False
        if self.generate_negatives:
            assert 'num_negatives' in kwargs
            self.num_negatives = kwargs['num_negatives']
        self.eval = kwargs['eval'] if 'eval' in kwargs else False
        self.pos = 0

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
            # try:
            base_content = self.contents[self.pos].split('\t')
            summaries = self.summaries[self.pos].split('\t')
            if len(base_content) != len(summaries):
                self.pos += increment
                continue

            # segment into chunks (assume pre-segmented)
            sentences = base_content
            if len(sentences) < 2:
                self.pos += increment
                continue

            # cutoff at some sentence
            try:
                cutoff = random.randint(0, len(sentences)-1)
            except:
                self.pos += increment
                continue
            prefix = '' # actually don't use this part here
            summary = summaries[cutoff]

            # select true, repetition, shuffled sentence, random other story
            possible_modes = ['true']
            if 'other' in self.negative_categories:
                possible_modes.append('other')
            if cutoff > 0 and 'repeat' in self.negative_categories: # can't repeat if don't have anything to repeat yet
                possible_modes.append('repeat')
            if cutoff < len(sentences) - 1 and 'shuffle' in self.negative_categories: # no shuffle if only 1 sentence left
                possible_modes.append('shuffle')
            
            if self.generate_negatives:
                completions = set()
                all_examples = []
                true_example, true_completion = self.create_example('true', sentences, cutoff, prefix, summary)
                all_examples.append(true_example)
                completions.add(true_completion)
                for _ in range(self.num_negatives):
                    while True:
                        mode = random.choice(possible_modes)
                        if mode == 'true':
                            continue
                        neg_example, neg_completion = self.create_example(mode, sentences, cutoff, prefix, summary)
                        if neg_completion not in completions:
                            all_examples.append(neg_example)
                            completions.add(neg_completion)
                            break
            else:
                mode = random.choice(possible_modes)
                example, _ = self.create_example(mode, sentences, cutoff, prefix, summary)
                all_examples = example

            valid = True
            self.pos += increment
        return all_examples
    
    def create_example(self, mode, sentences, cutoff, prefix, summary, max_extra_prefix_chunk_length=32):
        is_true_prefix = random.random() < 0.5
        if is_true_prefix:
            # true extra prefix
            prefix_sentences = group_chunks(split_paragraphs(sentences[cutoff], mode='sentence'), max_chunk_length=max_extra_prefix_chunk_length)
        else:
            # fake extra prefix
            other_content_sentences = self.contents[random.randint(0, len(self.contents)-1)].split('\t')
            other_content_sentence = random.choice(other_content_sentences).strip()
            prefix_sentences = group_chunks(split_paragraphs(other_content_sentence, mode='sentence'), max_chunk_length=max_extra_prefix_chunk_length)
        extra_prefix_cutoff = random.randint(0, len(prefix_sentences)-1)
        extra_prefix = ' '.join(prefix_sentences[:extra_prefix_cutoff])
        prefix = extra_prefix.strip()
        tokenized_prefix = [self.tokenizer.eos_token_id] + self.tokenizer.encode(prefix) if 'bart' in self.tokenizer_model else self.tokenizer.encode(prefix)
        tokenized_summary = [self.tokenizer.eos_token_id] + self.tokenizer.encode(summary) if 'bart' in self.tokenizer_model else self.tokenizer.encode(summary)

        if mode == 'true':
            if is_true_prefix:
                separate_completion = prefix_sentences[extra_prefix_cutoff]
            else:
                true_sentences = group_chunks(split_paragraphs(sentences[cutoff], mode='sentence'), max_chunk_length=max_extra_prefix_chunk_length)
                separate_completion = random.choice(true_sentences)
            completion = prefix.strip() + ' ' + separate_completion
            label = np.array([1])
        # create repetition example
        elif mode == 'repeat':
            separate_completion_chunk = random.choice(sentences[:cutoff]).strip() # random already used sentence
            separate_completion = random.choice(group_chunks(split_paragraphs(separate_completion_chunk, mode='sentence'), max_chunk_length=max_extra_prefix_chunk_length))
            completion = prefix.strip() + ' ' + separate_completion
            label = np.array([0])
        # create shuffled sentence example
        elif mode == 'shuffle':
            separate_completion_chunk = random.choice(sentences[cutoff+1:]).strip() # random out of order sentence
            separate_completion = random.choice(group_chunks(split_paragraphs(separate_completion_chunk, mode='sentence'), max_chunk_length=max_extra_prefix_chunk_length))
            completion = prefix.strip() + ' ' + separate_completion
            label = np.array([0])
        # create random other story example
        elif mode == 'other':
            if is_true_prefix:
                other_content_sentences = self.contents[random.randint(0, len(self.contents)-1)].split('\t')
                separate_completion_chunk = random.choice(other_content_sentences).strip()
                separate_completion = random.choice(group_chunks(split_paragraphs(separate_completion_chunk, mode='sentence'), max_chunk_length=max_extra_prefix_chunk_length))
            else:
                separate_completion = prefix_sentences[extra_prefix_cutoff]
            completion = prefix.strip() + ' ' + separate_completion
            label = np.array([0])
        
        tokenized_completion = [self.tokenizer.eos_token_id] + self.tokenizer.encode(completion) if 'bart' in self.tokenizer_model else self.tokenizer.encode(completion)
        loss_mask = np.array([0 for _ in range(len(tokenized_prefix))] + [1 for _ in range(len(tokenized_completion) - len(tokenized_prefix))])

        prefix_summary = concatenate_summary_text(summary, prefix)
        tokenized_prefix_summary = [self.tokenizer.eos_token_id] + self.tokenizer.encode(prefix_summary) if 'bart' in self.tokenizer_model else self.tokenizer.encode(prefix_summary)
        completion_summary = concatenate_summary_text(summary, completion)
        tokenized_completion_summary = [self.tokenizer.eos_token_id] + self.tokenizer.encode(completion_summary) if 'bart' in self.tokenizer_model else self.tokenizer.encode(completion_summary)
        completion_summary_loss_mask = np.array([0 for _ in range(len(tokenized_prefix_summary))] + [1 for _ in range(len(tokenized_completion_summary) - len(tokenized_prefix_summary))])

        if self.tokenized_info:
            # prefix_info: 'input_ids', 'attention_mask' (all 1)
            prefix_info = self.tokenizer(prefix, return_tensors='pt')
            # completion_info: 'input_ids', 'attention_mask'
            completion_info = self.tokenizer(separate_completion, return_tensors='pt')
            # reversed_prefix_sentence_info: 'input_ids', 'attention_mask'
            reversed_prefix_sentence_info = self.tokenizer(list(reversed([s for s in sentences[:cutoff] if len(s.strip()) > 0])), return_tensors='pt', padding=True)
        else:
            prefix_info, completion_info, reversed_prefix_sentence_info = None, None, None
                
        example = {'prefix': tokenized_completion, # you actually want to run on all of the completion, and then mask out the tokenized_prefix sometimes
                   'labels': label, 
                   'summary': tokenized_summary, 
                   'prefix_summary': tokenized_completion_summary,
                   'loss_mask': loss_mask,
                   'prefix_summary_loss_mask': completion_summary_loss_mask,
                   'prefix_info': prefix_info, 
                   'completion_info': completion_info,
                   'reversed_prefix_sentence_info': reversed_prefix_sentence_info,
                  }

        return example, completion