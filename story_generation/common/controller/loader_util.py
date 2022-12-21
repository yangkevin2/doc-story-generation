import torch

from story_generation.common.controller.loaders.coherence_loader import CoherenceSplitLoader
from story_generation.common.controller.loaders.fine_coherence_loader import FineCoherenceSplitLoader
from story_generation.common.controller.loaders.alignment_loader import AlignmentSplitLoader
from story_generation.common.controller.loaders.order_loader import OrderSplitLoader

def get_loader(loader_name, dataset, split, collate_fn, batch_size=32, append_mask_token=False, num_workers=20, tokenizer_model='roberta-base', **kwargs):
    assert split in ['train', 'valid', 'test']
    if loader_name == 'coherence':
        loader_class = CoherenceSplitLoader
    elif loader_name == 'fine_coherence':
        loader_class = FineCoherenceSplitLoader
    elif loader_name == 'alignment':
        loader_class = AlignmentSplitLoader
    elif loader_name == 'order':
        loader_class = OrderSplitLoader
    else:
        raise NotImplementedError
    print('loading texts for data loader')
    contents, summaries = dataset.load_long_texts(split), dataset.load_short_texts(split)
    print('done loading texts')
    return torch.utils.data.DataLoader(loader_class(contents, summaries, tokenizer_model, append_mask_token=False, **kwargs), batch_size=batch_size, pin_memory=True, collate_fn=collate_fn, num_workers=num_workers)
