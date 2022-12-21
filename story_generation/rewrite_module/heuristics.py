import re

import torch
import Levenshtein

from story_generation.edit_module.entity import *
from story_generation.common.util import *
from story_generation.common.data.split_paragraphs import *

# most of the model-based reranking infra is in common/controller

def detect_first_second_person(text):
    text = text.replace("“", '"').replace("”", '"').replace('\n', ' ')
    text = text.split('"')
    for i in range(0, len(text), 2): # all the sections that are outside of quotations
        # actually not gonna detect "you" since some informal third person constructions can use it
        if any([s in ' ' + text[i] + ' ' for s in ["I ", "I'", ' my ', 'My ', ' me ', 'Me.', 'Me ', ' you ', " you'", 'You ', "You'", ' we ', 'We ', "We'", '?', '!']]):
            return True
    return False


def calculate_repetition_length_penalty(generation, prompt_sentences, levenshtein_repetition_threshold=0.8, max_length=None, tokenizer=None, is_outline=False, repetition_length=5, verbose=False):
    if verbose:
        logging.log(21, 'calculate repetition length penalty generation:' + generation)
    if len(generation.strip()) == 0:
        if verbose:
            logging.log(21, 'calculate repetition length penalty: empty generation')
        return 10
    if max_length is not None:
        if len(tokenizer.encode(generation)) > max_length:
            if verbose:
                logging.log(21, 'calculate repetition length penalty: too long')
            return 10
    if any([s.lower() in generation.lower() for s in ['\nRelevant', '\nContext', '\nComment', 'Summar', '\nSupporting', '\nEvidence', '\nStages', '\nText', '\nAssum', '\n1.', '\n1)', '\nRelationship', '\nMain Character', '\nCharacter', '\nConflict:', '\nPlot', 'protagonist', '\nEdit ', '\nPremise', 'Suspense', 'www', '.com', 'http', '[', ']', 'copyright', 'chapter', '\nNote', 'Full Text', 'narrat', '\n(', 'All rights reserved', 'The story', 'This story', '(1)', 'passage', '\nRundown', 'playdown', 'episode', 'plot device', 'java', '\nQuestion', '\nDiscuss', '\nAnalysis', '</s>', 'he writer', 'text above', 'above text', 'described previous', 'previously described', 'The book', 'ee also', '\nShare', '\nLink', 'ontinue read', ' scene', '\nScene', 'work of fiction', 'the author', 'This book', 'main character', 'written by']]): # it's repeating parts of the prompt/reverting to analysis
        if verbose:
            logging.log(21, 'calculate repetition length penalty: bad word case insensitive')
        return 10
    if any([s in generation for s in ['TBA', 'POV']]):
        if verbose:
            logging.log(21, 'calculate repetition length penalty: bad word case sensitive')
        return 10
    generation_paragraphs = split_paragraphs(generation, mode='newline')
    for i, paragraph in enumerate(generation_paragraphs):
        if len(paragraph.strip()) == 0:
            continue
        if ':' in ' '.join(paragraph.strip().split()[:10]) or paragraph.strip().endswith(':'): # there's a colon in the first few words, so it's probably a section header for some fake analysis, or ends with a colon
            if verbose:
                logging.log(21, 'calculate repetition length penalty: colon')
            return 10
        elif (i > 0 or generation.startswith('\n')) and paragraph[0] in "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~": # punctuation other than quotes
            if verbose:
                logging.log(21, 'calculate repetition length penalty: punctuation at start')
            return 10 # don't want paragraphs that start with punctuation
        for p in prompt_sentences:
            p_paragraphs = split_paragraphs(p, mode='newline')[1:-1]
            if paragraph in p_paragraphs: # if any of the complete paragraphs from the prompt are repeated verbatim as complete paragraphs
                if verbose:
                    logging.log(21, 'calculate repetition length penalty: repeated verbatim paragraph')
                return 10
    penalty = 0
    # if re.search('\W\W', generation.replace('\n', 'a').replace(' ', 'a').replace('\t', 'a')) is not None: # multiple consecutive non-spacing punctuations
    no_consecutive_punc = "!#$%&*+,-./:;<=>?@[\\]^_`{|}~"
    if re.search('[' + ''.join([re.escape(p) for p in no_consecutive_punc]) + ']{2,}', generation) is not None:
        if verbose:
            logging.log(21, 'calculate repetition length penalty: multiple consecutive non-spacing punctuations')
        penalty += 5
    for p in prompt_sentences:
        split = p.lower().split(' ')
        for i in range(repetition_length+1, len(split)):
            if ' '.join(split[i-repetition_length:i]) in generation.lower(): # somewhat penalize repeated strings of 5 words or more for each prompt sentence
                if verbose:
                    logging.log(21, 'calculate repetition length penalty: repeated string of ' + str(repetition_length) + ' words or more compared to repetition string: ' + p)
                penalty += 1
                # break
    split = generation.lower().split(' ')
    for i in range(repetition_length+1, len(split)):
        if ' '.join(split[i-repetition_length:i]) in ' '.join(split[i:]): # penalize repetition within the generation itself
            if verbose:
                logging.log(21, 'calculate repetition length penalty: repeated string of ' + str(repetition_length) + ' words or more within generation')
            penalty += 1
            # break
    for i in range(1, len(split)):
        if len(split[i]) > 4 and split[i] == split[i-1]: # consecutive repeated non-common words
            if verbose:
                logging.log(21, 'calculate repetition length penalty: consecutive repeated non-common words')
            penalty += 1
    mildly_bad_strings = ['\n\n\n\n\n', 'novel', 'passage', 'perspective', 'point of view', 'summar', 'paragraph', 'sentence', 'example', 'analy', 'section', 'character', 'review', 'readers', '(', ')', 'blog', 'website', 'comment', 'This book', 'The book', 'ficti']
    if not is_outline:
        mildly_bad_strings += ['1.', '2.', '3.', '4.', '5.']
    num_mildly_bad_strings = sum([1 for s in mildly_bad_strings if s in generation.lower()])
    if num_mildly_bad_strings > 0:
        if verbose:
            logging.log(21, 'calculate repetition length penalty: mildly bad strings')
        penalty += num_mildly_bad_strings # discourage multiple of these strings appearing, since it's likely that this is resulting from GPT3 generating story analysis
    generation_sentences = split_paragraphs(generation, mode='sentence')
    for g in generation_sentences:
        for p in prompt_sentences:
            if Levenshtein.ratio(g, p) > levenshtein_repetition_threshold:
                if verbose:
                    logging.log(21, 'calculate repetition length penalty: levenshtein ratio too high vs prompt sentence: ' + p)
                penalty += 1
    return penalty


if __name__=='__main__':
    import pdb; pdb.set_trace()