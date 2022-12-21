import argparse
import csv
import os
from multiprocessing import Pool

from tqdm import tqdm

from story_generation.common.util import add_general_args
from story_generation.common.summarizer.summarizer_util import add_summarizer_args, load_summarizer

def generate_example(args):
    summarizer = load_summarizer(args)
    premise = summarizer(['Write a premise for a short story.'], model_string=args.gpt3_model)[0].strip()
    story = summarizer([premise + '\n\nA professional author wrote a lengthy story based on this premise, as follows:'], model_string=args.gpt3_model)[0].strip()
    return premise, story

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser = add_general_args(parser)
    parser = add_summarizer_args(parser)
    parser.add_argument('--save-csv', type=str, required=True, help='save to this csv file')
    parser.add_argument('--total-examples', type=int, default=1000, help='total number of examples to generate')
    parser.add_argument('--num-threads', type=int, default=1, help='number of threads to use')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    with open(args.save_csv, 'w') as wf:
        writer = csv.writer(wf)
        writer.writerow(['premise', 'story'])
        with Pool(args.num_threads) as pool:
            for premise, story in tqdm(pool.imap(generate_example, [args] * args.total_examples), total=args.total_examples):
                if len(premise) > 0 and len(story) > 0:
                    writer.writerow([premise, story])
