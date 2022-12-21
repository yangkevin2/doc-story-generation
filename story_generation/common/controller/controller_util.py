from story_generation.common.controller.models.fudge_controller import FudgeController
from story_generation.common.controller.models.longformer_classifier import LongformerClassifier

CONTROLLER_CHOICES=['fudge_controller', 'longformer_classifier', 'none']
LOADER_CHOICES=['coherence', 'fine_coherence', 'alignment', 'order', 'none']

def add_controller_args(parser):
    parser.add_argument('--controller', type=str, nargs='*', choices=CONTROLLER_CHOICES, help='model architecture. in main.py, first for relevance, second for coherence, third for OPT fudge control, fourth for outline ordering')
    parser.add_argument('--controller-model-string', type=str,  nargs='*', help='model string')
    parser.add_argument('--loader', type=str, nargs='*', choices=LOADER_CHOICES, help='loader for controller')
    parser.add_argument('--controller-save-dir', type=str, default=None, help='directory to save controller')
    parser.add_argument('--controller-load-dir', type=str, nargs='*', default=[''], help='directory to load controller')
    parser.add_argument('--controller-epochs', type=int, default=1, help='number of epochs for controller finetuning')
    parser.add_argument('--control-strength', type=float, nargs='*', default=[1.0], help='strength of control for controller inference')
    parser.add_argument('--fudge-top-k', type=int, nargs='*', default=[100], help='top k for fudge inference')
    parser.add_argument('--fudge-batch-size', type=int, default=256, help='batch size for fudge inference')
    parser.add_argument('--controller-num-negatives', type=int, default=1, help='number of negative samples for controller contrastive training')
    parser.add_argument('--coherence-negative-categories', type=str, nargs='*', default=['other', 'repeat', 'shuffle'], help='types of negatives for coherence')
    parser.add_argument('--controller-lr', type=float, default=5e-5, help='learning rate for controller finetuning')
    return parser

def load_controller(args, index):
    if args.controller[index] == 'none':
        return None
    elif args.controller[index] == 'fudge_controller':
        controller = FudgeController(args, index)
        if len(args.controller_load_dir[index]) > 0:
            controller.load(args.controller_load_dir[index])
    elif args.controller[index] == 'longformer_classifier':
        controller = LongformerClassifier(args, index)
        if len(args.controller_load_dir[index]) > 0:
            controller.load(args.controller_load_dir[index])
    else:
        raise NotImplementedError
    return controller