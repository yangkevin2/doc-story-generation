import argparse

from story_generation.common.util import add_general_args
from story_generation.common.data.data_util import add_data_args, load_dataset
from story_generation.common.controller.controller_util import add_controller_args, load_controller

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser = add_general_args(parser)
    parser = add_data_args(parser)
    parser = add_controller_args(parser)
    parser.add_argument('--controller-interactive', action='store_true', help='interactive mode for testing controller')
    args = parser.parse_args()

    controller = load_controller(args, 0)
    if args.controller_interactive:
        import pdb; pdb.set_trace() # for inspection

    assert args.controller_save_dir is not None

    dataset = load_dataset(args)
    dataset.shuffle('train')
    controller.fit(dataset)
