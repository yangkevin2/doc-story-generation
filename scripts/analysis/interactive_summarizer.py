import argparse
import logging

from story_generation.edit_module.entity import *
from story_generation.common.util import *
from story_generation.common.data.data_util import add_data_args
from story_generation.common.summarizer.summarizer_util import add_summarizer_args, load_summarizer

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser = add_general_args(parser)
    parser = add_data_args(parser)
    parser = add_summarizer_args(parser)
    args = parser.parse_args()
    logging.basicConfig(format='%(message)s', filename=args.log_file, level=args.log_level)

    summarizer = load_summarizer(args)

    import pdb; pdb.set_trace()