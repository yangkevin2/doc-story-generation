import argparse
import os
import pickle

from story_generation.edit_module.entity import *
from story_generation.plan_module.plan import *
from story_generation.common.util import *
from story_generation.common.data.data_util import add_data_args
from story_generation.common.summarizer.summarizer_util import add_summarizer_args, load_summarizer
from story_generation.common.summarizer.models.opt_summarizer import OPTSummarizer
from story_generation.common.summarizer.models.gpt3_summarizer import GPT3Summarizer
from story_generation.common.data.split_paragraphs import *

def join_story(previous_text, top_level_outline_items):
    current_story = ''
    for item in top_level_outline_items:
        if item in previous_text:
            current_story += ''.join(previous_text[item])
    return current_story

if __name__=='__main__':
    parser = argparse.ArgumentParser() # parameter defaults are set to values used in paper
    parser = add_general_args(parser)
    parser = add_data_args(parser)
    parser = add_summarizer_args(parser)

    parser.add_argument('--load-outline-file', type=str, help='load outline from this file')
    parser.add_argument('--extension-method', type=str, default='opt', choices=['gpt3', 'opt'], help='model/method to use for main story generation')
    parser.add_argument('--max-continuation-substeps', type=int, default=5, help='max number of continuation candidates to generate at each step')
    parser.add_argument('--save-complete-file', type=str, help='save completed beam object to this file')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_complete_file), exist_ok=True)

    model = GPT3Summarizer(args) if args.extension_method == 'gpt3' else OPTSummarizer(args)
    tokenizer = model.tokenizer

    plan_info = load_plan_info(args.load_outline_file)
    top_level_outline_items = [item.text for item in plan_info['outline']]
    premise = plan_info['premise']

    previous_text = {}
    for item_idx, outline_item in enumerate(top_level_outline_items):
        previous_text[outline_item] = []
        base_prefix = f'Premise: {premise}\n\nCurrent Story Outline: {outline_item}'
        for i in range(args.max_continuation_substeps):
            if item_idx == 0 and i == 0:
                prefix = base_prefix + '\n\nWrite a story according to this premise, starting with the current outline.\n\n-----------------------------\n\nChapter 1\n\n'
                print(prefix)
            else:
                prefix = base_prefix + '\n\nWrite a story according to this premise, continuing from the current outline.\n\n-----------------------------\n\n'
                print(prefix)
                prefix_length = len(tokenizer.encode(prefix))
                story = join_story(previous_text, top_level_outline_items)
                left_truncated_story = tokenizer.decode(tokenizer.encode(story)[-(args.max_context_length - args.max_tokens - prefix_length - 1):]) # avoid off by one errors by doing an extra -1
                prefix += left_truncated_story
            if args.extension_method == 'gpt3':
                continuation = model([prefix], max_tokens=args.max_tokens, model_string='davinci', logit_bias={50256:-100})[0] # don't let it end prematurely
            else:
                continuation = model(
                    [prefix],
                    max_tokens=args.max_tokens,
                    logit_bias_decay=args.summarizer_frequency_penalty_decay
                )[0]
            previous_text[outline_item].append(continuation)

    with open(args.save_complete_file, 'wb') as wf:
        plan_info['passages'] = previous_text
        pickle.dump(plan_info, wf)

    for i, item in enumerate(top_level_outline_items):
        print(f'\n\n\n\nOutline Item {i}: {item}')
        print(f'Story Passage {i}:\n\n', ''.join(previous_text[item]))
