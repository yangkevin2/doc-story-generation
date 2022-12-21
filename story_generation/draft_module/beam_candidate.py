from copy import deepcopy
import string
import logging

import torch
import numpy as np

from story_generation.plan_module.outline import *
from story_generation.edit_module.entity import *
from story_generation.rewrite_module.heuristics import *
from story_generation.common.util import *
from story_generation.common.controller.loaders.alignment_loader import create_prefix_completion
from story_generation.common.data.split_paragraphs import *
from story_generation.common.summarizer.models.gpt3_summarizer import GPT3_END

class BeamCandidate:
    def __init__(self, 
                 args, 
                 all_entities_dict,
                 infer_attributes_string,
                 outline,
                 model=None,
                 opt_model=None,
                 controllers=None,
                 step=0, 
                 alignment_score=-1e8, 
                 best_alignment_so_far=-1e8,
                 alignment_history=None,
                 all_paragraphs=None,
                 outline_sections=None,
                 detailed_outline_section_history=None,
                 paragraphs_by_outline_section=None):
        self.args = args
        self.all_entities_dict = all_entities_dict
        self.infer_attributes_string = infer_attributes_string
        self.outline = outline
        self.model = model
        self.opt_model = opt_model
        self.controllers = controllers
        self.step = step
        self.alignment_score = alignment_score
        self.best_alignment_so_far = best_alignment_so_far
        self.alignment_history = alignment_history if alignment_history is not None else []
        self.all_paragraphs = all_paragraphs if all_paragraphs is not None else []
        self.outline_sections = outline_sections if outline_sections is not None else []
        self.detailed_outline_section_history = detailed_outline_section_history if detailed_outline_section_history is not None else []
        self.paragraphs_by_outline_section = paragraphs_by_outline_section if paragraphs_by_outline_section is not None else {}
        self.is_consistent = False
    
    def story(self, demarcated_outline_section=None):
        out = ''
        for p in self.all_paragraphs:
            if demarcated_outline_section is not None and demarcated_outline_section in self.paragraphs_by_outline_section and p in self.paragraphs_by_outline_section[demarcated_outline_section]:
                out += '<SECTION START>' + p + '<SECTION END>'
            else:
                out += p
        return out
    
    def previous_passage(self, max_tokens, suffix=None):
        if len(self.all_paragraphs) == 0:
            return ''
        passage = self.story()
        if len(self.story().strip()) == 0:
            return ''
        if suffix is not None:
            passage = passage[:len(passage) - len(suffix)].rstrip()
        if len(passage.strip()) == 0:
            return ''
        passage = self.model.tokenizer.decode(self.model.tokenizer.encode(passage)[-max_tokens:])
        return cut_first_sentence(passage)
    
    def print_section(self, section_idx):
        return ''.join(self.paragraphs_by_outline_section[self.outline_sections[section_idx]])
    
    def create_updated_entities(self, new_passage, cached_update_dict=None):
        # detect and make entries for new entities, run inference for description / is_character on new entities, update attributes
        new_entities_dict = {k: v for k, v in self.all_entities_dict.items()}
        entities = [str(ent) for ent in detect_entities(new_passage)]
        matched_entities, new_entities, _ = deduplicate_match_entities(entities, self.all_entities_dict.keys())
        new_entities_dict = {k: v for k, v in self.all_entities_dict.items()}
        for ent in new_entities:
            entity = Entity(ent)
            entity.infer_description(new_passage, self.model, max_length=self.args.entity_description_max_length)
            entity.infer_is_character(new_passage, self.model)
            entity.infer_attributes(new_passage, self.model, other_names=[name for name in matched_entities if name != entity.name] + [name for name in new_entities if name != entity.name])
            new_entities_dict[ent] = entity
        for ent in matched_entities:
            if cached_update_dict is not None and ent in cached_update_dict:
                new_entities_dict[ent] = cached_update_dict[ent]
            else:
                new_entities_dict[ent].infer_attributes(new_passage, self.model, other_names=[name for name in matched_entities if name != ent] + list(new_entities), detect_contradictions=False)
        complete_mutual_relations(new_entities_dict, self.model)
        return new_entities_dict
    
    def detect_attribute_contradictions(self, completion, detect_contradictions=True):
        matched_entities, new_entities, _ = deduplicate_match_entities(detect_entities(completion, add_dpr_entities=False, all_entities_dict=self.all_entities_dict), self.all_entities_dict.keys())
        matched_entities = list(matched_entities)
        contradictions = {}
        cached_update_dict = {}
        copied_entities = {k: v for k, v in self.all_entities_dict.items()}
        for ent in matched_entities:
            entity = copied_entities[ent]
            contradictions[ent] = entity.infer_attributes(completion, self.model, detect_contradictions=detect_contradictions, other_names=[name for name in matched_entities if name != entity.name] + list(new_entities))
            cached_update_dict[ent] = entity
        _, additional_contradictions = complete_mutual_relations(copied_entities, self.model)
        for ent in additional_contradictions:
            for key in additional_contradictions[ent]:
                if ent not in contradictions:
                    contradictions[ent] = {}
                contradictions[ent][key] = additional_contradictions[ent][key]
        return matched_entities, contradictions, cached_update_dict

    def condense_outline_sections(self, outline, section_list, i):
        # condense the previous parts of the outline that will be put in the prompt as a summary 
        # specifically, leaves before the immediately preceding node will be collapsed up into their parents where possible
        logging.log(23, 'CONDENSING OUTLINE')
        logging.log(23, 'BEFORE')
        logging.log(23, str([n.text for n in self.outline_sections]))
        current_leaf = section_list[i]
        assert current_leaf.text in self.outline_sections[-1].text
        keep_nodes = [current_leaf] if i == 0 else [current_leaf, section_list[i-1]]
        self.outline_sections = [node for node in outline.collapse_around(keep_nodes, up_to=current_leaf)] # keep the current leaf and up to 1 previous leaf
        logging.log(23, 'AFTER')
        logging.log(23, str([n.text for n in self.outline_sections]))

    def construct_prompt(self, node, selected_entities=[]):
        presumed_max_prompt_length = self.args.max_context_length - self.args.max_tokens
        if len(self.all_paragraphs) == 0:
            prompt = 'Premise: ' + self.all_entities_dict['Premise'].description.replace('Premise:', '').strip() + ' ' +  self.all_entities_dict['Setting'].description.replace('Setting:', '').strip() + '\n\n\n\n'
        else: 
            # if it's the first piece in a group of outline pieces, add the higher-level section in the hierarchy as a "premise" to guide generation
            prompt = ''
            parent_texts = []
            current_node = node
            current_scene = current_node.scene
            while True:
                parent = current_node.parent
                if parent is not None and len(parent.text.strip()) > 0:
                    parent_texts.append(parent.text)
                    current_node = parent
                else:
                    break
            if len(parent_texts) > 0:
                parent_texts = reversed(parent_texts)
                prompt = 'Premise: ' + ' '.join(parent_texts) + '\n\n\n\n'

        tokenizer = self.opt_model.tokenizer if self.opt_model is not None else self.model.tokenizer
        prompt += 'This book was authored by a well-known novelist, and received glowing reviews from critics, who praised the interesting dialogue and interactions between characters.'
        if len(selected_entities) > 0:
            selected_entity_strings = []
            for ent in selected_entities:
                desc = self.all_entities_dict[ent].get_outline_description_up_to_node(node, max_tokens=128, tokenizer=tokenizer)
                assert len(tokenizer.encode(desc)) <= 128
                selected_entity_strings.append(desc)
            while sum([len(tokenizer.encode(desc)) for desc in selected_entity_strings]) > self.args.max_entity_context_tokens:
                selected_entity_strings = selected_entity_strings[:-1]
            logging.log(22, 'SELECTED ENTITIES: ' + str(selected_entities))
            logging.log(22, 'SELECTED ENTITY STRINGS: ' + str(selected_entity_strings))
            prompt += '\n\n\n\nRelevant Context:\n\n' + '\n\n'.join(selected_entity_strings)
        else:
            logging.warning('No selected entities')
        if self.step > 1:
            prompt += '\n\n\n\nPrevious story summary: ' + ' '.join([n.text for n in self.outline_sections[:-1]])
        previous_text = self.previous_passage(self.args.previous_prompt_length)
        if len(self.all_paragraphs) > 0:
            previous_passage = self.previous_passage(int(self.args.max_context_length/2), suffix=previous_text)
            if len(tokenizer.encode(previous_passage)) > int(self.args.max_context_length/4): # no need to do this extra summary if it's really short
                max_preceding_summary_tokens = int(self.args.previous_prompt_length / 2)
                preceding_summary = self.model([previous_passage + '\n\nSummarize the events in this passage.'], max_tokens=max_preceding_summary_tokens, model_string='text-curie-001', cut_sentence=True)[0].strip().replace('\n\n', ' ')
                if len(tokenizer.encode(preceding_summary)) == max_preceding_summary_tokens:
                    logging.warning('Warning: preceding events summary is too long, truncating')
                prompt += '\n\n\n\nEvents immediately prior to the upcoming passage: ' + preceding_summary
        next_node = node.successor()
        if next_node is not None and self.args.include_future_context:
            next_text = ' ' + next_node.text.strip()
        else:
            next_text = ''
        if self.step == 1:
            prompt += '\n\n\n\nChapter 1 Summary: ' + node.text.strip() + next_text
        else:
            previous_node = self.detailed_outline_section_history[-2]
            previous_scene = previous_node.scene
            previous_text_entities, _, _ = deduplicate_match_entities(detect_entities(previous_text), self.all_entities_dict.keys())
            prompt += '\n\n\n\nThe characters currently in the scene are ' + ', '.join(list(previous_text_entities) + [e for e in previous_node.selected_entities if e not in previous_text_entities]) + '.'
            prompt += '\n\n\n\nIn the upcoming passage, ' + self.detailed_outline_section_history[-2].text + ' ' + node.text.strip() + next_text
            logging.log(22, 'PREVIOUS SCENE: ' + str(previous_scene))
            logging.log(22, 'CURRENT SCENE: ' + str(current_scene))
            if not is_same_scene(previous_scene, current_scene):
                prompt += '\n\n\n\nThis part of the story initially takes place in ' + previous_scene + ' The characters then move to ' + current_scene
            else:
                prompt += '\n\n\n\nThis part of the story takes place in ' + current_scene
        prompt += '\n\n\n\nFull text below:\n\n--------------------------------\n\n'
        if len(self.all_paragraphs) == 0:
            prompt = prompt + 'Chapter 1\n\n'
        prompt = prompt + previous_text
        prompt = prompt.replace('\n\n\n\n', '\n\n')
        if len(tokenizer.encode(prompt)) > presumed_max_prompt_length:
            # truncate if too long; *very* rarely happens
            logging.log(22, 'WARNING: CONTEXT TOO LONG, TRUNCATING')
            prompt = tokenizer.decode(tokenizer.encode(prompt)[-presumed_max_prompt_length:]) # left truncate prompt to fit our imposed limit on context window size
        return prompt
    
    @torch.no_grad()
    def edit_update_contradictions(self):
        assert not self.is_consistent
        completion = self.all_paragraphs[-1]
        autoregressive_context = self.all_paragraphs[-2].lstrip(string.punctuation) if len(self.all_paragraphs) > 1 else ''
        matched_entities, contradictions, cached_update_dict = self.detect_attribute_contradictions(completion.strip(), detect_contradictions=True)
        edited_sentences = set()
        if any([len(contradictions[ent]) > 0 for ent in matched_entities]) and len(autoregressive_context) > 0: # don't do it on the first paragraph, if we don't have autoregressive context to help check we're not messing something up
            logging.log(23, 'editing completion based on contradictions')
            logging.log(23, 'AUTOREGRESSIVE CONTEXT ' + autoregressive_context)
            logging.log(23, 'BEFORE ' + completion)
            for ent in matched_entities:
                for contradiction_key in contradictions[ent]:
                    for contradicted_sentence in contradictions[ent][contradiction_key][0]['text'].strip().split('\n'):
                        if contradicted_sentence in edited_sentences: # no need to edit again if the sentence was contradicted more than once
                            continue
                        edited_sentences.add(contradicted_sentence)
                        instruction = 'Edit so that ' + contradicted_sentence + ' Keep the text unchanged as much as possible.'
                        logging.log(23, 'INSTRUCTION ' + instruction)
                        completion = gpt3_edit(completion, instruction, prefix=None if len(autoregressive_context.strip()) == 0 else autoregressive_context).strip()
                        if len(self.model.tokenizer.encode(completion)) > self.args.max_tokens + 64: # give some leeway for editing to expand text
                            logging.warning('WARNING: completion is too long after editing. Truncating...')
                            completion = self.model.tokenizer.decode(self.model.tokenizer.encode(completion)[:self.args.max_tokens + 64])
                            completion = cut_last_sentence(completion)
            logging.log(23, 'AFTER ' + completion)
            _, _, cached_update_dict = self.detect_attribute_contradictions(completion.strip(), detect_contradictions=False) # only reupdate the cache, and allow appending any new entries; presumably GPT3 fixed any "real" contradictions
        self.all_paragraphs[-1] = completion
        self.paragraphs_by_outline_section[self.outline_sections[-1]][-1] = completion
        self.all_entities_dict = self.create_updated_entities(completion.strip(), cached_update_dict=cached_update_dict)
        self.is_consistent = True

    @torch.no_grad()
    def extend(self, node):
        # return a list of up to max_beam_size new BeamCandidates with their respective alignment scores before moving on to the next outline sentence
        logging.log(25, 'extension step ' + str(self.step))
        logging.log(23, 'outline section: ' + node.text)
        self.step += 1
        self.alignment_score = -1e8
        self.best_alignment_so_far = -1e8
        self.alignment_history = []
        self.outline_sections.append(node)
        self.detailed_outline_section_history.append(node)
        self.paragraphs_by_outline_section[node] = []
        completed_candidates = []
        beam = [self]
        substep = 0
        while len(completed_candidates) < self.args.max_beam_size:
            logging.log(25, 'substep ' + str(substep))
            next_candidates = []
            for beam_idx, prev_candidate in enumerate(beam):
                candidates = []
                for candidate in prev_candidate.extend_single(node, batch_size=self.args.max_candidates, top_p=self.args.draft_top_p, substep=substep):
                    candidates.append(candidate)
                    logging.log(25, 'beam idx ' + str(beam_idx) + ' single extension with score ' + str(candidates[-1].alignment_score))
                if max([c.alignment_score for c in candidates]) < self.args.skip_threshold: # try generating a few more
                    for candidate in prev_candidate.extend_single(node, batch_size=self.args.max_candidates, top_p=self.args.draft_top_p, substep=substep):
                        candidates.append(candidate)
                    logging.log(25, 'beam idx ' + str(beam_idx) + ' extra single extension with score ' + str(candidates[-1].alignment_score))
                candidates = sorted(candidates, key=lambda x: x.alignment_score, reverse=True)
                logging.log(25, 'best candidate with score ' + str(candidates[0].alignment_score) + ':\n' + candidates[0].all_paragraphs[-1])
                if (candidates[0].alignment_score < prev_candidate.best_alignment_so_far - self.args.continuation_threshold and prev_candidate.best_alignment_so_far >= self.args.early_stop_threshold):
                    # early termination of expansion of this outline point: below early stop threshold, and continuation not as good
                    logging.log(25, 'beam idx ' + str(beam_idx) + ' adding completed candidate with early stop score ' + str(prev_candidate.alignment_score) + ' and best alignment score ' + str(prev_candidate.best_alignment_so_far))
                    assert self.args.no_editor or prev_candidate.is_consistent
                    completed_candidates.append(prev_candidate)
                elif candidates[0].alignment_score < self.args.skip_threshold:
                    logging.log(25, 'beam idx ' + str(beam_idx) + ' adding acceptable candidate with score ' + str(prev_candidate.alignment_score) + ' and best alignment score ' + str(prev_candidate.best_alignment_so_far))
                    assert self.args.no_editor or prev_candidate.is_consistent
                    completed_candidates.append(prev_candidate)
                else:
                    if candidates[0].alignment_score < prev_candidate.best_alignment_so_far:
                        logging.log(25, 'continuation with slightly worse score')
                    next_candidates.extend(candidates)
            next_candidates = sorted(next_candidates, key=lambda x: x.alignment_score, reverse=True)[:self.args.max_beam_size]
            beam = next_candidates
            if len(completed_candidates) > 0: # just early stop, don't bother spending time continuing
                beam = []
            if not self.args.no_editor:
                for c in beam:
                    c.edit_update_contradictions()
            substep += 1
            if substep >= self.args.max_continuation_substeps: # fill out the rest of the completed candidates
                for c in beam:
                    logging.log(25, 'beam idx ' + str(beam_idx) + ' adding completed candidate with score ' + str(c.alignment_score) + ' and best alignment score ' + str(c.best_alignment_so_far))
                    assert self.args.no_editor or c.is_consistent
                    completed_candidates.append(c)
                break
        completed_candidates = [c for c in completed_candidates if c is not None]
        if len(completed_candidates) == 0:
            completed_candidates = [self]
        # sort first by best alignment so far and then by current alignment score
        return sorted(completed_candidates, key=lambda x: x.best_alignment_so_far * 10000 + x.alignment_score, reverse=True)[:self.args.max_beam_size]
    
    def calculate_alignment(self, completions, prompt, node):
        if self.args.max_candidates == 1:
            return np.zeros(len(completions)) # in this case, we're doing no reranking, and this will also prevent the reranking from being used to decide when to stop. 
        unstripped_completions = completions
        completions = [c.strip() for c in completions]
        repetition_penalty = np.array([calculate_repetition_length_penalty(c, [prompt]) for c in completions])
        last_prompt_paragraph = split_paragraphs(prompt, mode='newline')[-1]
        is_first_person = np.array([1 if detect_first_second_person(last_prompt_paragraph + c) - detect_first_second_person(last_prompt_paragraph) else 0 for c in completions]) # could have some false positives if the quotations are off, but whatever.
        repetition_penalty += is_first_person
        alignment_score = 0

        # for relevance, since we generate shorter passages at a time before refreshing the prompt compared to re3, 
        # we'll compute the min of two relevance scores for slightly better robustness. 
        # specifically we align against 
        # (1) the text generated thus far for this outline node, and 
        # (2) the text generated thus far for the outline node, prefixed by the last generated part of the preceding outline node (if it exists)
        if self.args.controller[0] == 'longformer_classifier':
            previous_outline_section = self.detailed_outline_section_history[-2] if len(self.detailed_outline_section_history) > 1 else None
            if previous_outline_section is not None and len(self.paragraphs_by_outline_section[previous_outline_section]) > 0:
                previous_text = self.paragraphs_by_outline_section[previous_outline_section][-1]
            else:
                previous_text = ''
            alignment_input = [create_prefix_completion(''.join(self.paragraphs_by_outline_section[node]) + c, node.text)[1] for c in unstripped_completions]
            prefix_alignment_input = [create_prefix_completion(previous_text + ''.join(self.paragraphs_by_outline_section[node]) + c, node.text)[1] for c in unstripped_completions]
            logging.log(22, 'prefix alignment input 0: ' + str(prefix_alignment_input[0]))
        else:
            raise NotImplementedError
        relevance_scores = self.controllers[0].evaluate_overall_texts(alignment_input).cpu().numpy() # logprob for alignment with outline
        logging.log(22, 'relevance scores: ' + str(['%.2f' % score for score in relevance_scores]))
        prefix_relevance_scores = self.controllers[0].evaluate_overall_texts(prefix_alignment_input).cpu().numpy()
        logging.log(22, 'prefix relevance scores: ' + str(['%.2f' % score for score in prefix_relevance_scores]))
        relevance_scores = np.array([min(rs, prs) for rs, prs in zip(relevance_scores, prefix_relevance_scores)]) # take the minimum of the two for robustness, to avoid occasional fake signal from where the continuation got cut off
        logging.log(22, 'min relevance scores: ' + str(['%.2f' % score for score in relevance_scores]))
        alignment_score += relevance_scores * self.args.control_strength[0]

        # similar relevance checking for scenes/characters
        current_node = node
        if len(self.detailed_outline_section_history) > 1:
            previous_node = self.detailed_outline_section_history[-2]
            extra_relevance_strings = []
            if not is_same_scene(current_node.scene, previous_node.scene):
                extra_relevance_strings.append(('The characters move to ' + current_node.scene, self.args.control_strength[0] * 0.5))
            for character in current_node.selected_entities:
                if character not in previous_node.selected_entities:
                    extra_relevance_strings.append((character + ' enters the scene.', self.args.control_strength[0] * 0.2))
            for ers, cs in extra_relevance_strings:
                logging.log(22, 'scene/char relevance string: ' + ers)
                extra_alignment_input = [create_prefix_completion(''.join(self.paragraphs_by_outline_section[node]) + c, ers)[1] for c in unstripped_completions]
                extra_prefix_alignment_input = [create_prefix_completion(previous_text + ''.join(self.paragraphs_by_outline_section[node]) + c, ers)[1] for c in unstripped_completions]
                extra_relevance_scores = self.controllers[0].evaluate_overall_texts(extra_alignment_input).cpu().numpy()
                logging.log(22, 'scene/char relevance scores: ' + str(['%.2f' % score for score in extra_relevance_scores]))
                extra_prefix_relevance_scores = self.controllers[0].evaluate_overall_texts(extra_prefix_alignment_input).cpu().numpy()
                logging.log(22, 'scene/char prefix relevance scores: ' + str(['%.2f' % score for score in extra_prefix_relevance_scores]))
                extra_relevance_scores = np.array([min(rs, prs) for rs, prs in zip(extra_relevance_scores, extra_prefix_relevance_scores)])
                logging.log(22, 'min scene/char relevance scores: ' + str(['%.2f' % score for score in extra_relevance_scores]))
                alignment_score += extra_relevance_scores * cs

        # coherence reranker
        if len(self.story().strip()) > 0:
            coherence_scores = self.controllers[1]([self.previous_passage(1000) for _ in range(len(completions))], completions).cpu().numpy() # logprob for alignment with previous story, up to 1k prev tokens
            logging.log(22, 'coherence scores: ' + str(['%.2f' % score for score in coherence_scores]))
            alignment_score += coherence_scores * self.args.control_strength[1]
        else:
            alignment_score += -1 * self.args.control_strength[1] # add some baseline level to prevent early stopping

        # heuristics e.g. repetition penalty
        logging.log(22, 'repetition: ' + str(['%.2f' % score for score in -repetition_penalty * self.args.repetition_penalty_weight]))
        alignment_score += -repetition_penalty * self.args.repetition_penalty_weight
        return alignment_score

    def extend_single(self, node, batch_size=1, top_p=None, substep=0):
        if self.args.generation_outline_levels is not None and self.args.generation_outline_levels == 1:
            assert self.step == len(self.outline_sections)
        selected_entities = node.selected_entities
        prompt = self.construct_prompt(node, selected_entities=selected_entities)
        logging.log(22, 'PROMPT')
        logging.log(22, prompt)

        if self.args.extension_method == 'gpt3':
            completions = self.model([prompt], 
                                     model_string=self.args.draft_model_string, 
                                     num_completions=batch_size, 
                                     top_p=top_p, 
                                     temperature=self.args.summarizer_temperature, 
                                     cut_sentence=self.args.cut_sentence, 
                                     logit_bias={50256:-100}) # don't let it end prematurely

        elif self.args.extension_method == 'opt-control':
            current_control_strength = min(self.args.control_strength[2] + substep * self.args.control_strength_substep_increment, self.args.max_control_strength)
            
            # OPT logit biases, including some decaying bias against the prompt
            exclude_strings = stopwords.words('english') + list("!\"“”‘’'(),-.:;?") + ['\n', '\n\n'] + selected_entities
            assert '\n\nFull text below:\n\n' in prompt
            previous_paragraph = prompt.split('\n\nFull text below:\n\n')[-1].strip()
            opt_control_logit_bias = self.opt_model.create_logit_bias_for_prompt(
                previous_paragraph, 
                bias=-self.args.summarizer_frequency_penalty,
                decay=self.args.summarizer_frequency_penalty_decay
            )
            prompt_logit_bias_string = prompt[:len(prompt) - len(previous_paragraph)]
            for character in self.all_entities_dict:
                prompt_logit_bias_string = prompt_logit_bias_string.replace(self.all_entities_dict[character].description, '') # don't bias against char descriptions?
            opt_control_logit_bias_prompt = self.opt_model.create_logit_bias_for_prompt(
                prompt_logit_bias_string,
                bias=-self.args.summarizer_prompt_penalty,
                exclude_strings=exclude_strings,
            )
            for key in opt_control_logit_bias_prompt:
                if key in opt_control_logit_bias:
                    opt_control_logit_bias[key] = min(opt_control_logit_bias[key], opt_control_logit_bias_prompt[key])
                else:
                    opt_control_logit_bias[key] = opt_control_logit_bias_prompt[key]
            opt_control_logit_bias[2] = -1e8 # ban </s>

            current_controllers = [self.controllers[2]]
            current_control_texts = [concatenate_summary_text(node.text, "")] # control string for event             
            current_control_strengths = [current_control_strength]
            current_node = node
            if len(self.detailed_outline_section_history) > 1:
                # control strings for new scenes/characters (generally not necessary for the very first passage of the story)
                previous_node = self.detailed_outline_section_history[-2]
                if not is_same_scene(current_node.scene, previous_node.scene):
                    current_controllers.append(self.controllers[2])
                    current_control_texts.append(concatenate_summary_text('The characters move to ' + current_node.scene, ""))
                    current_control_strengths.append(current_control_strength * 0.5)
                section_entities, _, _ = deduplicate_match_entities(detect_entities(''.join(self.paragraphs_by_outline_section[node])), self.all_entities_dict.keys(), prioritized_names=current_node.selected_entities)
                logging.log(22, 'section entities: ' + str(section_entities))
                for character in current_node.selected_entities:
                    if character not in previous_node.selected_entities and character not in section_entities: # don't need to keep controlling for char after appearing
                        current_controllers.append(self.controllers[2])
                        current_control_texts.append(concatenate_summary_text(character + ' enters the scene.', ""))
                        current_control_strengths.append(current_control_strength * 0.2)

            logging.log(22, 'control texts: ' + str(current_control_texts))
            logging.log(22, 'control strengths' + str(current_control_strengths))
            completions = self.opt_model.generate_with_controller(
                current_controllers,
                current_control_texts,
                prompt, 
                control_strengths=current_control_strengths,
                max_tokens=self.args.max_tokens,
                temperature=self.args.opt_summarizer_temperature,
                logit_bias=opt_control_logit_bias,
                num_completions=batch_size,
                cut_sentence=self.args.cut_sentence,
                logit_bias_decay=self.args.summarizer_frequency_penalty_decay
            )
        else:
            raise NotImplementedError

        for i in range(len(completions)):
            logging.log(22, 'COMPLETION: ' + completions[i])
            while '\n\n\n' in completions[i]: # just improve the formatting a bit
                completions[i] = completions[i].replace('\n\n\n', '\n\n')
        for i in range(len(completions)):
            _, _, replacements = deduplicate_match_entities(detect_entities(completions[i].strip()), self.all_entities_dict.keys())
            if not self.args.no_editor:
                for key, value in replacements.items():
                    completions[i] = completions[i].replace(key, value)
        alignment_score = self.calculate_alignment(completions, prompt, node)
        new_candidates = []
        for c, s in zip(completions, alignment_score):
            new_paragraphs_by_outline_section = deepcopy(self.paragraphs_by_outline_section)
            new_paragraphs_by_outline_section[node].append(c)
            new_candidates.append(BeamCandidate(self.args, 
                                self.all_entities_dict,
                                self.infer_attributes_string,
                                self.outline,
                                model=self.model, 
                                opt_model=self.opt_model,
                                controllers=self.controllers, 
                                step=self.step, 
                                alignment_score=s, 
                                best_alignment_so_far=max(s, self.best_alignment_so_far),
                                alignment_history=self.alignment_history + [s],
                                all_paragraphs=deepcopy(self.all_paragraphs) + [c], 
                                outline_sections=[o for o in self.outline_sections],
                                detailed_outline_section_history=[o for o in self.detailed_outline_section_history],
                                paragraphs_by_outline_section=new_paragraphs_by_outline_section))
        return new_candidates
    
    def complete_ending(self):
        node = self.outline_sections[-1]
        if node not in self.paragraphs_by_outline_section:
            self.paragraphs_by_outline_section[node] = []
        selected_entities = node.selected_entities
        prompt = self.construct_prompt(node, selected_entities=selected_entities)
        completions = gpt3_insert(prompt, 
                                 '\n\n\n\n' + GPT3_END, 
                                 top_p=self.args.draft_top_p, 
                                 temperature=self.args.summarizer_temperature,
                                 n=self.args.max_candidates, 
                                 max_tokens=self.args.max_tokens, 
                                 frequency_penalty=self.args.summarizer_frequency_penalty,
                                 presence_penalty=self.args.summarizer_presence_penalty)
        completions = [c.replace('\n\n\n\n', '\n\n') for c in completions]
        alignment_score = self.calculate_alignment(completions, prompt, node)
        logging.log(23, 'ENDING ALIGNMENT SCORES ' + str(alignment_score))
        ranked_completions = sorted(zip(completions, alignment_score), key=lambda x: x[1], reverse=True)
        ending = ranked_completions[0][0]
        should_continue = len(self.model.tokenizer.encode(ending))==self.args.max_tokens # ending didn't finish writing; should generate more toward the ending after this
        ending = cut_last_sentence(ending)
        logging.log(23, 'ENDING' + ' ' + ending)
        new_paragraphs_by_outline_section = deepcopy(self.paragraphs_by_outline_section)
        new_paragraphs_by_outline_section[node].append(ending)
        new_candidate = BeamCandidate(self.args, 
                            self.all_entities_dict,
                            self.infer_attributes_string,
                            self.outline,
                            model=self.model, 
                            opt_model=self.opt_model,
                            controllers=self.controllers, 
                            step=self.step, 
                            alignment_score=self.alignment_score, # TODO if you ever need this, should fix this to something proper
                            best_alignment_so_far=self.best_alignment_so_far,
                            alignment_history=self.alignment_history + [self.alignment_score],
                            all_paragraphs=deepcopy(self.all_paragraphs) + [ending], 
                            outline_sections=[o for o in self.outline_sections],
                            detailed_outline_section_history=[o for o in self.detailed_outline_section_history],
                            paragraphs_by_outline_section=new_paragraphs_by_outline_section)
        if not self.args.no_editor:
            new_candidate.edit_update_contradictions()
        return new_candidate, should_continue