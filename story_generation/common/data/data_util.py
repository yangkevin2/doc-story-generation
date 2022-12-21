from story_generation.common.data.datasets.writing_prompts import WritingPromptsDataset
from story_generation.common.data.datasets.csv import CSVDataset
from story_generation.common.data.datasets.alignment import AlignmentDataset

DATASET_CHOICES=['writing_prompts', 'csv', 'alignment']
# if providing a csv, shold give the full path to csv in data-dir. only for inference. 

def add_data_args(parser):
    parser.add_argument('--dataset', type=str, choices=DATASET_CHOICES, help='dataset format')
    parser.add_argument('--data-dir', type=str, help='data directory')
    parser.add_argument('--split-sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1], help='train/val/test proportions for datasets where not provided')
    parser.add_argument('--limit', type=int, default=None, help='limit the number of examples')
    parser.add_argument('--length-limit', type=int, default=1000000, help='limit the number of words per example')
    parser.add_argument('--lower-length-limit', type=int, default=0, help='limit the number of words per example')
    parser.add_argument('--summary-length-limit', type=int, default=1000000, help='limit the number of words in the summary')
    parser.add_argument('--csv-column', type=str, help='column name to use as input for csv when using csv dataset')
    parser.add_argument('--num-workers', type=int, default=20, help='number of workers for data loading')
    return parser

def load_dataset(args):
    if args.dataset == 'writing_prompts':
        dataset = WritingPromptsDataset(args)
    elif args.dataset == 'csv':
        dataset = CSVDataset(args)
    elif args.dataset == 'alignment':
        dataset = AlignmentDataset(args)
    else:
        raise NotImplementedError
    return dataset