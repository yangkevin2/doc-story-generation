import os

from typing import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, AutoTokenizer, AutoModel

from story_generation.common.controller.models.abstract_controller import AbstractController
from story_generation.common.util import AverageMeter, pad_to_max_length, pad_mask
from story_generation.common.controller.loader_util import get_loader
from story_generation.common.util import *


class FudgeController(AbstractController):
    def __init__(self, args, index):
        self.type = 'fudge'
        self.index = index
        self.model_string = args.controller_model_string[index] if args.controller_model_string[index] != 'none' else 'gpt2'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args
        self.trained = False
        self.loader_type = self.args.loader[self.index]
        if self.args.loader[self.index] in ['coherence', 'fine_coherence']:
            # use our own custom class to run on all prefixes, with input conditioning
            if 'gpt2' in self.model_string:
                self.model = GPT2ForAutoregressiveClassification(AutoModel.from_pretrained(self.model_string)).to(self.device)
            elif 'facebook/opt' in self.model_string:
                self.model = OPTForAutoregressiveClassification(AutoModel.from_pretrained(self.model_string)).to(self.device)
            else:
                raise NotImplementedError
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        else:
            raise NotImplementedError
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_string)
        self.optimizer = AdamW(self.model.parameters(), lr=args.controller_lr)
    
    def reset_cache(self):
        self.past_key_values = None
        self.encoder_outputs = None
    
    @torch.no_grad()
    def evaluate_overall_texts(self, texts):
        positive_log_probs = []
        for text in texts: # not batched for now; larger controllers of this type can be memory heavy
            batch = self.tokenizer(text, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(**batch)
            positive_log_probs.append(F.log_softmax(logits, dim=-1)[0, -1, 1].item())
        positive_log_probs = torch.Tensor(positive_log_probs).to(self.device)
        return positive_log_probs * self.args.control_strength[self.index]
    
    @torch.no_grad()
    def __call__(self, lm_logits, input_ids, keyword_ids=None, control_strength=None, **kwargs):
        """
        lm_logits: beam x 1 x vocab
        input_ids: beam x seqlen
        optionally, top_logits and top_indices, both beam x 1 x topk
        """
        # get the top k next tokens, up to some limit
        if 'top_logits' not in kwargs or kwargs['top_logits'] is None:
            top_logits, top_indices = torch.topk(lm_logits, k=self.args.fudge_top_k[self.index], dim=-1) # beam x 1 x topk
        else:
            top_logits, top_indices = kwargs['top_logits'], kwargs['top_indices']
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size = self.args.fudge_batch_size
        # get past key values
        past_key_values = None
        # TODO fix the past key values, code below currently broken, though the fudge model is lightweight enough relative to the base generator that it's not that big a deal anyway
        # if self.past_key_values is not None and input_ids.shape[1] > 5:
        #     past_key_values_idxs = []
        #     past_decoder_input_ids, past_top_indices, past_key_values = self.past_key_values
        #     for ids in input_ids:
        #         for i, past_ids in enumerate(past_decoder_input_ids):
        #             if ids[:-1].shape != past_ids.shape:
        #                 import pdb; pdb.set_trace() # somehow this bug showed up once?
        #             if (ids[:-1] - past_ids).abs().max() == 0:
        #                 past_beam_idx = i
        #                 break
        #         past_top_indices_idx = (past_top_indices[past_beam_idx][0] == ids[-1]).nonzero().item()
        #         past_key_values_idx = self.args.fudge_top_k[self.index] * past_beam_idx + past_top_indices_idx
        #         past_key_values_idxs.append(past_key_values_idx)
        #     new_past_key_values = []
        #     for i in range(len(past_key_values)):
        #         new_past_key_values_section = []
        #         for j in range(len(past_key_values[i])):
        #             new_past_key_values_section.append(torch.stack(sum([[past_key_values[i][j][past_idx] for _ in range(self.args.fudge_top_k[self.index])] for past_idx in past_key_values_idxs], []), dim=0).to(self.device))
        #         new_past_key_values.append(tuple(new_past_key_values_section))
        #     new_past_key_values = tuple(new_past_key_values)
        #     past_key_values = new_past_key_values

        # construct prefixes for controller
        controller_prefixes = torch.cat([input_ids.unsqueeze(1).repeat(1, top_indices.shape[-1], 1), top_indices.permute(0, 2, 1)], dim=2) # beam x topk x decoder_len+1
        controller_prefixes = controller_prefixes.flatten(0, 1) # beam*topk x decoder_len+1
        # feed them through the controller

        controller_output_logits = []
        for i in range(0, controller_prefixes.shape[0], batch_size):
            controller_outputs = self.model(controller_prefixes[i:i+batch_size])
            controller_output_logits.append(controller_outputs[:, -1, :])
        controller_output_logits = torch.cat(controller_output_logits, dim=0)
        controller_output_logits = F.log_softmax(controller_output_logits, dim=-1)

        if self.loader_type in ['coherence', 'fine_coherence', 'token_fudge']:
            final_control_logits = controller_output_logits[:, 1].view(top_logits.shape[0], top_logits.shape[-1]) # beam x topk of positive label logprob

        # add logprobs to lm_logits
        new_lm_logits = torch.zeros_like(lm_logits) - 1e8
        control_strength = self.args.control_strength[self.index] if control_strength is None else control_strength
        new_lm_logits = torch.scatter(new_lm_logits, 2, top_indices, top_logits + control_strength * final_control_logits.unsqueeze(1)) # add fudge keyword logits to appropriate positions
        
        return new_lm_logits
    
    def fit(self, dataset):
        # largely following https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
        best_val_loss = 1e8
        for epoch in range(self.args.controller_epochs):
            dataset.shuffle('train')
            train_loader = get_loader(
                self.args.loader[self.index], 
                dataset, 'train', 
                fudge_collate, 
                batch_size=self.args.batch_size, 
                append_mask_token=True, 
                tokenizer_model=self.model_string,
                num_workers=self.args.num_workers, 
            )
            loop = tqdm(train_loader, leave=True)
            loss_meter = AverageMeter('loss', ':6.4f')
            for batch in loop:
                # initialize calculated gradients (from prev step)
                self.optimizer.zero_grad()
                # pull all tensor batches required for training
                if batch['encoder_ids'].shape[0] < self.args.batch_size: # don't do the last batch if smaller
                    continue
                output = self.model(input_ids=batch['encoder_ids'].to(self.device))
                # mask loss based on length, also expand labels based on length
                labels = batch['labels'].unsqueeze(1).expand(-1, output.shape[1], -1).flatten(0, 1).to(self.device)
                if labels.shape[-1] == 1:
                    labels = labels.squeeze(-1).long()
                loss = self.criterion(output.flatten(0, 1), labels)
                loss_mask = batch['loss_mask'].flatten().to(self.device)
                loss = loss * loss_mask
                loss = loss.sum() / loss_mask.sum()

                loss.backward()
                # update parameters
                self.optimizer.step()
                loss_meter.update(loss.detach().item(), batch['encoder_ids'].shape[0])
                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())
            print('Training epoch {} average loss {}'.format(epoch, loss_meter.avg))
            
            valid_loader = get_loader(
                self.args.loader[self.index], 
                dataset, 
                'valid', 
                fudge_collate, 
                batch_size=self.args.batch_size,
                append_mask_token=True, 
                tokenizer_model=self.model_string, 
                num_workers=self.args.num_workers, 
                eval=True
            )
            loop = tqdm(valid_loader, leave=True)
            loss_meter = AverageMeter('loss', ':6.4f')
            with torch.no_grad():
                for batch in loop:
                    # initialize calculated gradients (from prev step)
                    # pull all tensor batches required for training
                    output = self.model(input_ids=batch['encoder_ids'].to(self.device))

                    # mask loss based on length, also expand labels based on length
                    labels = batch['labels'].unsqueeze(1).expand(-1, output.shape[1], -1).flatten(0, 1).to(self.device)
                    if labels.shape[-1] == 1:
                        labels = labels.squeeze(-1).long()
                    loss = self.criterion(output.flatten(0, 1), labels)
                    loss_mask = batch['loss_mask'].flatten().to(self.device)
                    loss = loss * loss_mask
                    loss = loss.sum() / loss_mask.sum()

                    loss_meter.update(loss.item(), batch['encoder_ids'].shape[0])
                    # print relevant info to progress bar
                    loop.set_description(f'Epoch {epoch}')
                    loop.set_postfix(loss=loss.item())
                print('Validation epoch {} average loss {}'.format(epoch, loss_meter.avg))
            if loss_meter.avg < best_val_loss:
                print('Found new best model. Saving...')
                best_val_loss = loss_meter.avg
                self.save(os.path.join(self.args.controller_save_dir, 'model_best.pth.tar'))

        self.trained = True

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': self.args
        }, path)

    def load(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
        except:
            checkpoint = torch.load(os.path.join(path, 'model_best.pth.tar'), map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.trained = True


class GPT2ForAutoregressiveClassification(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.classification_head = nn.Linear(model.config.n_embd, 2)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.classification_head(outputs.last_hidden_state)

        return logits


class OPTForAutoregressiveClassification(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.classification_head = nn.Linear(model.config.word_embed_proj_dim, 2)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.classification_head(outputs.last_hidden_state)

        return logits


def fudge_collate(batch):
    inputs = [torch.LongTensor(p['prefix_summary']) for p in batch]
    input_ids = torch.stack(pad_to_max_length(inputs, 0), dim=0)
    lengths = torch.LongTensor([len(p['prefix_summary']) for p in batch])
    labels = torch.stack([torch.from_numpy(p['labels']) for p in batch], dim=0)
    if batch[0]['prefix_summary_loss_mask'] is None:
        loss_mask = pad_mask(lengths).permute(1, 0)
    else:
        loss_mask = torch.stack(pad_to_max_length([torch.from_numpy(p['prefix_summary_loss_mask']) for p in batch], 0), dim=0)
    return {'encoder_ids': input_ids, 
            'labels': labels, 
            'loss_mask': loss_mask}