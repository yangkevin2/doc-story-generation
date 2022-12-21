import time
import logging

import torch
from transformers import AutoTokenizer
import openai

from story_generation.common.summarizer.models.abstract_summarizer import AbstractSummarizer
from story_generation.common.data.split_paragraphs import cut_last_sentence

GPT3_END = 'THE END.'
PRETRAINED_MODELS = ['ada', 'babbage', 'curie', 'davinci', 'text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-001', 'text-davinci-002', 'text-davinci-003']

class GPT3Summarizer(AbstractSummarizer):
    def __init__(self, args):
        assert args.gpt3_model is not None
        self.model = args.gpt3_model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.args = args
        self.controller = None

    @torch.no_grad()
    def __call__(self, texts, suffixes=None, max_tokens=None, top_p=None, temperature=None, retry_until_success=True, stop=None, logit_bias=None, num_completions=1, cut_sentence=False, model_string=None):
        assert type(texts) == list
        if logit_bias is None:
            logit_bias = {}
        if suffixes is not None:
            assert len(texts) == len(suffixes)
        if model_string is None:
            logging.warning('model string not provided, using default model')
        if self.controller is None:
            return self._call_helper(texts, suffixes=suffixes, max_tokens=max_tokens, top_p=top_p, temperature=temperature, retry_until_success=retry_until_success, stop=stop, logit_bias=logit_bias, num_completions=num_completions, cut_sentence=cut_sentence, model_string=model_string)
        else:
            raise NotImplementedError
    
    @torch.no_grad()
    def _call_helper(self, texts, suffixes=None, max_tokens=None, top_p=None, temperature=None, retry_until_success=True, stop=None, logit_bias=None, num_completions=1, cut_sentence=False, model_string=None):
        assert model_string in PRETRAINED_MODELS

        if logit_bias is None:
            logit_bias = {}

        outputs = []
        for i in range(len(texts)):
            text = texts[i]
            prompt = text
                        
            retry = True
            num_fails = 0
            while retry:
                try:
                    context_length = len(self.tokenizer.encode(prompt))
                    if context_length > self.args.max_context_length:
                        logging.warning('context length' + ' ' + str(context_length) + ' ' + 'exceeded artificial context length limit' + ' ' + str(self.args.max_context_length))
                        time.sleep(5) # similar interface to gpt3 query failing and retrying
                        assert False
                    if max_tokens is None:
                        max_tokens = min(self.args.max_tokens, self.args.max_context_length - context_length)
                    engine = self.model if model_string is None else model_string
                    if engine == 'text-davinci-001':
                        engine = 'text-davinci-002' # update to latest version
                    logging.log(21, 'PROMPT')
                    logging.log(21, prompt)
                    logging.log(21, 'MODEL STRING:' + ' ' + self.model if model_string is None else model_string)
                    completion = openai.Completion.create(
                        engine=engine,
                        prompt=prompt,
                        suffix=suffixes[i] if suffixes is not None else None,
                        max_tokens=max_tokens,
                        temperature=temperature if temperature is not None else self.args.summarizer_temperature,
                        top_p=top_p if top_p is not None else self.args.summarizer_top_p,
                        frequency_penalty=self.args.summarizer_frequency_penalty,
                        presence_penalty=self.args.summarizer_presence_penalty,
                        stop=stop,
                        logit_bias=logit_bias,
                        n=num_completions)
                    retry = False
                except Exception as e: 
                    logging.warning(str(e))
                    retry = retry_until_success
                    num_fails += 1
                    if num_fails > 20:
                        raise e
                    if retry:
                        logging.warning('retrying...')
                        time.sleep(num_fails)
            outputs += [completion['choices'][j]['text'] for j in range(num_completions)]
        if cut_sentence:
            for i in range(len(outputs)):
                if len(outputs[i].strip()) > 0:
                    outputs[i] = cut_last_sentence(outputs[i])
        engine = self.model if model_string is None else model_string
        logging.log(21, 'OUTPUTS')
        logging.log(21, str(outputs))
        logging.log(21, 'GPT3 CALL' + ' ' + engine + ' ' + str(len(self.tokenizer.encode(texts[0])) + sum([len(self.tokenizer.encode(o)) for o in outputs])))
        return outputs