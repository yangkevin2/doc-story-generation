import logging
from collections.abc import Sequence
from copy import deepcopy
import uuid

from story_generation.common.util import *
from story_generation.rewrite_module.heuristics import *
from story_generation.plan_module.name_util import *
from story_generation.edit_module.entity import *


SCENE_PREFIX = 'This scene is located in'


def display_contiguous_context_section(ordered_nodes): # assume nodes are in order
    return '\n\n'.join([node.number() + node.text for node in ordered_nodes])


def is_imperative_interrogative(text):
    # detect if text is an imperative or interrogative, which typically means something went wrong in the outline generation
    if '?' in text:
        return True
    return pos_tag(text)[0].tag == 'VB' # heuristic for imperative: verb for first word


def simple_outline_section_check(piece, tokenizer=None, min_piece_length=None, max_piece_length=None, repetition_strings=None, repetition_length=5, levenshtein_ratio=0.8):
    logging.log(22, 'simple outline section check piece: ' + piece)
    if any([s in piece for s in '_=+[]\\/{}|<>^&*#@~`']):
        logging.log(22, 'bad punc character')
        return False
    if len(piece.strip()) == 0:
        logging.log(22, 'simple check failure: empty piece')
        return False
    if piece[-1] not in '.?!':
        logging.log(22, 'simple check failure: no punctuation at end')
        return False
    if piece[0] != piece[0].upper():
        logging.log(22, 'simple check failure: not capitalized')
        return False
    if tokenizer is not None and min_piece_length is not None and len(tokenizer.encode(piece)) < min_piece_length:
        logging.log(22, 'simple check failure: too short')
        return False
    if tokenizer is not None and max_piece_length is not None and len(tokenizer.encode(piece)) > max_piece_length:
        logging.log(22, 'simple check failure: too long')
        return False
    if is_imperative_interrogative(piece):
        logging.log(22, 'simple check failure: imperative or interrogative')
        return False
    if piece.strip().split()[0].endswith('.'):
        logging.log(22, 'simple check failure: first word ends with period')
        return False
    if repetition_strings is not None and calculate_repetition_length_penalty(piece, repetition_strings, is_outline=True, repetition_length=repetition_length, levenshtein_repetition_threshold=levenshtein_ratio) > 0:
        logging.log(22, 'simple check failure: repetition')
        return False
    return True


class OutlinePiece(Sequence):
    def __init__(self, text, parent):
        self.text = text.strip()
        self.selected_entities = []
        self.scene = ''
        self.children = []
        self.parent = parent
        self.id = str(uuid.uuid4())
        super().__init__()
    
    def __hash__(self):
        return hash(self.id)
        
    @classmethod
    def num_converter(cls, depth):
        if depth == 0:
            return lambda num: '' # assume the root node should be empty
        if depth % 3 == 1:
            return str
        elif depth % 3 == 2:
            return num_to_char
        elif depth % 3 == 0:
            return num_to_roman
    
    @classmethod
    def indent(cls, depth):
        if depth == 0:
            return ''
        return '\t' * (depth-1)
    
    def __str__(self):
        ordered_nodes = [node for node in self.root().depth_first_traverse()]
        return '\n\n'.join([node.number() + node.text + ' Scene: ' + node.scene + ' Characters: ' + ', '.join(node.selected_entities) for node in ordered_nodes])
    
    def __len__(self):
        return len(self.children)
    
    def __getitem__(self, index):
        return self.children[index]
    
    def __eq__(self, other):
        return self.id == other.id

    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root()
    
    def depth(self):
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.depth()
        
    def number(self, shift=0, convert=True):
        if self.parent is None:
            num = 1
        else:
            num = self.parent.children.index(self) + 1
        if convert:
            depth = self.depth() + shift
            if depth == 0:
                return ''
            else:
                return '\t' * (depth-1) + OutlinePiece.num_converter(depth)(num) + '. '
        else:
            return num
    
    def predecessor(self, max_depth=None):
        nodes = list(self.root().depth_first_traverse(max_depth=max_depth))
        return nodes[nodes.index(self)-1] if nodes.index(self) > 0 else None
    
    def successor(self, max_depth=None):
        nodes = list(self.root().depth_first_traverse(max_depth=max_depth))
        return nodes[nodes.index(self)+1] if nodes.index(self) < len(nodes)-1 else None
    
    def clear_children(self):
        self.children = []
    
    def get_node_by_id(self, id):
        for node in self.root().depth_first_traverse():
            if node.id == id:
                return node
        return None
    
    def depth_first_traverse(self, include_self=True, max_depth=None):
        if include_self:
            if max_depth is None or self.depth() <= max_depth:
                yield self
        for child in self.children:
            yield from child.depth_first_traverse(max_depth=max_depth)
    
    def is_before(self, other, equal_ok=True):
        nodes = list(self.root().depth_first_traverse())
        return (nodes.index(self) <= nodes.index(other)) if equal_ok else (nodes.index(self) < nodes.index(other))
    
    def path_to_root(self):
        if self.parent is None:
            return [self]
        else:
            return [self] + self.parent.path_to_root()
    
    def siblings(self):
        if self.parent is None:
            return []
        else:
            return [child for child in self.parent.children if child != self]
    
    def index_in_outline(self):
        nodes = list(self.root().depth_first_traverse())
        return nodes.index(self)

    def context_nodes(self, ordered=False):
        # set of all siblings of nodes on the path to root from this node, plus all their children
        context = set()
        context.add(self)
        path_to_root = [self]
        current_node = self
        while current_node.parent is not None:
            context.add(current_node.parent)
            path_to_root.append(current_node.parent)
            current_node = current_node.parent
        for node in path_to_root:
            context.update(node.children)
            for child in node.children:
                context.update(child.children)
        if ordered:
            return sorted(context, key=lambda node: node.index_in_outline()) # inefficient but whatever
        else:
            return context
    
    def prefix(self, prune=True, include_self=True, omit_numbers_up_to=0, include_scene=False):
        ordered_nodes = [node for node in self.root().depth_first_traverse()]
        prefix_nodes = ordered_nodes[:ordered_nodes.index(self) + (1 if include_self else 0)]
        if prune:
            prefix_nodes = [node for node in prefix_nodes if node in self.context_nodes()]
        prefix = ''
        for node in prefix_nodes:
            prefix += '\n\n'
            if node.depth() >= omit_numbers_up_to:
                prefix += node.number() + node.text
            else:
                prefix += node.text
            if include_scene and len(node.scene) > 0:
                prefix += SCENE_PREFIX + ' ' + node.scene
        return prefix.strip()
    
    def suffix(self, prune=True, omit_numbers_up_to=0, shift_beginning_up_to=None, include_scene=False):
        ordered_nodes = [node for node in self.root().depth_first_traverse()]
        suffix_nodes = ordered_nodes[ordered_nodes.index(self)+1:]
        if shift_beginning_up_to is not None and len(suffix_nodes) > 0:
            # shift the depth of the suffix to help the GPT3 insertion API maintain long-range coherence, and adjust numbers accordingly
            shift = shift_beginning_up_to - suffix_nodes[0].depth()
        else:
            shift = 0
        if prune:
            suffix_nodes = [node for node in suffix_nodes if node in self.context_nodes()]
        returned_suffix = ''
        for node in suffix_nodes:
            returned_suffix += '\n\n'
            if node.depth() >= omit_numbers_up_to:
                if shift == 0:
                    number = node.number()
                else:
                    if node.number(convert=False) == 1:
                        number = node.number(shift=shift)
                    else:
                        if all([n in suffix_nodes for n in node.parent.children]):
                            number = node.number(shift=shift)
                        else:
                            previous_in_prefix, previous_in_suffix = 0, 0
                            for n in node.parent.children:
                                if n == node:
                                    break
                                if n in suffix_nodes:
                                    previous_in_suffix += 1
                                else:
                                    previous_in_prefix += 1
                            last_prefix_sibling = node.parent.children[previous_in_prefix-1]
                            for i in range(shift-1):
                                if last_prefix_sibling is None:
                                    break
                                last_prefix_sibling = last_prefix_sibling.children[-1] if len(last_prefix_sibling.children) > 0 else None
                            if last_prefix_sibling is None:
                                number = OutlinePiece.num_converter(node.depth()+shift)(previous_in_suffix + 1)
                            else:
                                number = OutlinePiece.num_converter(node.depth()+shift)(len(last_prefix_sibling.children) + previous_in_suffix + 1)
                            number = '\t' * (node.depth() + shift - 1) + number + '. '
                returned_suffix += number + node.text
            else:
                returned_suffix += node.text
            if include_scene and len(node.scene) > 0:
                returned_suffix += SCENE_PREFIX + ' ' + node.scene
        returned_suffix = returned_suffix.rstrip()
        if len(returned_suffix.strip()) == 0:
            return 'The End.'
        else:
            return returned_suffix
    
    def list_children_at_depth(self, depth): # note: depth is relative to the root, not to the current node
        return [child for child in self.depth_first_traverse() if child.depth() == depth]        
    
    def leaves(self):
        if len(self.children) == 0:
            return [self]
        else:
            return sum([child.leaves() for child in self.children], [])
    
    def collapse_around(self, keep_nodes, up_to=None):
        if not any([node in [n for n in self.depth_first_traverse(include_self=False)] for node in keep_nodes]):
            return [self]
        else:
            return sum([child.collapse_around(keep_nodes, up_to=up_to) for child in self.children if (up_to is None or child.is_before(up_to))], [])

    def expand(self, 
               fixed_prefix, 
               additional_prefix, 
               model, 
               model_string,
               characters,
               character_strings,
               infer_attributes_string, 
               max_tokens=256, 
               n=1, 
               min_sections=2,
               max_sections=5, 
               min_piece_length=3, 
               max_piece_length=50, 
               num_attempts=1, 
               logit_bias=None, 
               repetition_strings=None, 
               temperature=None,
               char_model_string=None):
        # expand the children under this node

        if num_attempts > 1:
            raise NotImplementedError # need to handle re-copying the saved character info across attempts
        if logit_bias is None:
            logit_bias = {}
        if repetition_strings is None:
            repetition_strings = []
        saved_characters = deepcopy(characters)
        saved_character_strings = deepcopy(character_strings)
        saved_infer_attributes_string = deepcopy(infer_attributes_string)
        self.clear_children()
        converter = OutlinePiece.num_converter(self.depth()+1)
        all_node_texts = [node.text for node in self.root().depth_first_traverse(include_self=True) if len(node.text.strip()) > 0]
        before_non_path_node_texts = [node.text for node in self.root().depth_first_traverse(include_self=True) if len(node.text.strip()) > 0 and node not in self.path_to_root() and node.is_before(self)]
        after_non_path_node_texts = [node.text for node in self.root().depth_first_traverse(include_self=True) if len(node.text.strip()) > 0 and node not in self.path_to_root() and self.is_before(node)]

        prefix = fixed_prefix + self.prefix(omit_numbers_up_to=self.depth()-1) + ('\n\n' + OutlinePiece.indent(self.depth()+1) + additional_prefix if len(additional_prefix.strip()) > 0 else '')
        self.children.append(OutlinePiece('dummy', self)) # add a dummy node to shift the numbering in the suffix appropriately for the prompt
        suffix = '\n\n' + self.children[-1].suffix(omit_numbers_up_to=self.depth()-1, shift_beginning_up_to=self.depth()+1) # shifting the depth to trick gpt3 to continue the narration more fluidly, rather than sometimes going out of order
        self.children = self.children[:-1] # remove dummy
        logging.log(22, 'prefix: ' + prefix)
        logging.log(22, 'suffix: ' + suffix)
        has_next = True
        for _ in range(num_attempts):
            self.clear_children()
            success = False
            children = []
            for sec_num in range(max_sections):
                if not has_next:
                    if len(children) >= min_sections:
                        success = True
                    break
                
                # generate initial outline sections
                prefix += '\n\n' + OutlinePiece.indent(self.depth()+1) + converter(sec_num+1) + '.'
                logging.log(21, 'current prefix: ' + prefix.replace('__CHARACTERS__', characters))
                logging.log(21, 'current suffix: ' + suffix)
                outline_sections = model([prefix.replace('__CHARACTERS__', characters)], suffixes=[suffix] if len(suffix.strip()) > 0 else None, stop=[' ' + converter(sec_num+2)+'.', '\n' + converter(sec_num+2)+'.', '.' + converter(sec_num+2)+'.', '\t' + converter(sec_num+2)+'.'], logit_bias=logit_bias, max_tokens=max_piece_length, num_completions=n, model_string=model_string, temperature=temperature)
                logging.log(22, 'initial outline sections: ' + str(outline_sections))
                
                # processing + basic filtering
                if self.depth() > 0:
                    suffix_beginning = OutlinePiece.num_converter(self.depth())(self.number(convert=False)+1) + '.'
                    for pre_tok in [' ', '\n', '\t', '.']:
                        outline_sections = [o.split(pre_tok + suffix_beginning)[0].strip() if pre_tok + suffix_beginning in o else o for o in outline_sections]
                outline_sections = [o.split('\n')[0].strip() + ('\n' if '\n' in o else '') for o in outline_sections] # prob unnecessary: + ('.' if o.split('\n')[0].strip()[-1] not in '.?!' else '')
                outline_sections = [(o[o.index('.')+1:].lstrip() if (len(o.strip()) > 0 and '.' in o.strip().split()[0]) else o) + ('\n' if '\n' in o else '') for o in outline_sections] # strip the lettering in the beginning if it has it for some reason
                outline_sections = [(split_paragraphs(o.strip(), mode='sentence')[0].strip() if len(split_paragraphs(o.strip(), mode='sentence')) > 0 else '') + ('\n' if '\n' in o else '') for o in outline_sections]
                outline_sections = [o for o in outline_sections if len(model.tokenizer.encode(o)) < max_piece_length] # filter out examples that are cutoff via max length (too long without newlines)
                logging.log(22, 'after max piece length' + str(max_piece_length) + ': ' + str(len(outline_sections)))
                outline_sections = [o for o in outline_sections if len(o.strip().split()) >= min_piece_length and len(o.strip(string.whitespace + string.punctuation)) > 0]
                logging.log(22, 'after min piece length' + str(min_piece_length) + ': ' + str(len(outline_sections)))
                outline_sections = [o.replace('\t', '') for o in outline_sections]
                outline_sections = [o for o in outline_sections if not any([text in o for text in (all_node_texts + [c.text for c in children]) if len(text.split()) > 10])] # don't take super repetitive stuff even when in the same part of the outline
                logging.log(22, 'after major repetition: ' + str(len(outline_sections)))
                
                # more careful repetition checks
                for (repetition_length, levenshtein_ratio) in [(5, 0.8)]:
                    filtered_outline_sections = [o for o in outline_sections if simple_outline_section_check(o.strip(), tokenizer=model.tokenizer, min_piece_length=min_piece_length, max_piece_length=max_piece_length, repetition_strings=repetition_strings + [c.text for c in children], repetition_length=repetition_length, levenshtein_ratio=levenshtein_ratio)]
                    if len(filtered_outline_sections) > 0:
                        break
                    else:
                        logging.log(22, 'no outline sections passed simple check with repetition length ' + str(repetition_length))
                outline_sections = filtered_outline_sections
                logging.log(22, 'after repetition filtering:' + str(outline_sections))
                if len(outline_sections) == 0: # failure, too repetitive
                    break

                # entailment checks
                if len(before_non_path_node_texts) > 0: # shouldn't be entailed by stuff that happened before
                    nonentailed_outline_sections = []
                    for o in outline_sections:
                        entailed_by_max_prob = softmax(score_entailment(before_non_path_node_texts, [o for _ in range(len(before_non_path_node_texts))])[0], axis=1)[:, 2].max()
                        if entailed_by_max_prob > 0.5:
                            continue
                        nonentailed_outline_sections.append(o)
                    outline_sections = nonentailed_outline_sections
                logging.log(22, 'after pre nonentailed: ' + str(len(outline_sections)))
                if len(after_non_path_node_texts) > 0: # shouldn't entail stuff that happened after
                    nonentailed_outline_sections = []
                    for o in outline_sections:
                        entail_other_max_prob = softmax(score_entailment([o for _ in range(len(after_non_path_node_texts))], after_non_path_node_texts)[0], axis=1)[:, 2].max()
                        if entail_other_max_prob > 0.5:
                            continue
                        nonentailed_outline_sections.append(o)
                    outline_sections = nonentailed_outline_sections
                logging.log(22, 'after post nonentailed: ' + str(len(outline_sections)))
                if len(outline_sections) == 0: # failure, too repetitive
                    break
                
                # reranker scores
                if len(self.text) > 0:
                    if sec_num == 0:
                        distances = [-s for s in sentence_similarity(self.text.strip().split(',')[0].strip(), [o.strip() for o in outline_sections])] # just for relative sorting. pick the one closest to the parent initially, up to the first comma
                        for o, d in zip(outline_sections, distances):
                            logging.log(21, 'distance to parent: ' + str(d) + ' for ' + o)
                    else:
                        text_only_prefix = self.prefix(omit_numbers_up_to=1e8)
                        for c in children:
                            text_only_prefix += '\n\n' + c.text
                        text_only_suffix_source = self if len(children) == 0 else children[-1]
                        text_only_suffix = text_only_suffix_source.suffix(omit_numbers_up_to=1e8)
                        order_scores = []
                        for o in outline_sections:
                            query = text_only_prefix + '\n\n*' + o.strip() + '*\n\n' + text_only_suffix
                            logging.log(21, 'order query:')
                            logging.log(21, query)
                            order_scores.append(get_outline_order_controller().evaluate_overall_texts([query]))
                            logging.log(21, 'order score' + str(order_scores[-1]))
                        distances = [-s for s in order_scores] # use NLL; it's just for relative sorting
                else:
                    distances = [0 for _ in outline_sections]

                # sort + pick best candidate, if we have any left
                outline_sections = sorted(outline_sections, key=lambda o: distances[outline_sections.index(o)])
                if sec_num < min_sections - 1:
                    outline_sections = outline_sections # don't do this re-sorting, actually
                else:
                    outline_sections = [o for o in outline_sections if not o.endswith('\n')] + [o for o in outline_sections if o.endswith('\n')] # try to end outline if possible
                if len(outline_sections) == 0:
                    break
                selected_outline_section = outline_sections[0]
                logging.log(22, 'selected outline section: ' + str(selected_outline_section.strip()))

                # add the selected outline section to the tree, and determine if we need to generate more children or if we're done, based on whether the current one ends with a newline
                children.append(OutlinePiece(selected_outline_section.strip(), self))
                self.children.append(children[-1])
                characters, character_strings, infer_attributes_string = children[-1].select_characters(model, model_string, characters, character_strings, infer_attributes_string, predecessors=[self] + children[:-1], char_model_string=char_model_string)
                has_next = selected_outline_section.endswith('\n') or sec_num < min_sections - 1
                prefix += ' ' + selected_outline_section.strip()
                self.children.append(OutlinePiece('dummy', self)) # add a dummy node to shift the numbering in the suffix appropriately for the prompt
                suffix = '\n\n' + self.children[-1].suffix(omit_numbers_up_to=self.depth()-1, shift_beginning_up_to=self.depth()+1) # shifting the depth to trick gpt3 to continue the narration more fluidly, rather than sometimes going out of order
                self.children = self.children[:-1] # remove dummy
            if success:
                break
        if success:
            return True, characters, character_strings, infer_attributes_string
        else:
            logging.warning('Warning: didn\'t find acceptable outline')
            self.clear_children()
            return False, saved_characters, saved_character_strings, saved_infer_attributes_string
    
    def recursively_expand_to_depth(self, 
                                    target_depth, 
                                    fixed_prefix, 
                                    additional_prefix, 
                                    model, 
                                    model_string, 
                                    characters, 
                                    character_strings, 
                                    infer_attributes_string, 
                                    max_tokens=256, 
                                    n=1, 
                                    min_sections=2, 
                                    max_sections=5, 
                                    min_piece_length=3, 
                                    max_piece_length=50, 
                                    num_attempts=1, 
                                    logit_bias=None, 
                                    repetition_strings=[], 
                                    temperature=None, 
                                    expand_self=False, 
                                    resample_self=True,
                                    char_model_string=None):
        # wrapper around expand() that can do multiple attempts, picks some repetition strings, and tries to expand to a target depth if needed,
        # though in practice we usually just expand 1 level at a time. 

        if logit_bias is None:
            logit_bias = {}
        for level in range(self.depth(), target_depth):
            if level == self.depth() and not expand_self:
                continue
            logging.log(25, 'Expanding outline level ' + str(level))
            outline_nodes = self.list_children_at_depth(level)
            logging.log(25, 'Total nodes: ' + str(len(outline_nodes)))
            for i, node in enumerate(outline_nodes):
                logging.log(25, 'Expanding node ' + str(i))
                current_temperature = temperature if temperature is not None else model.args.summarizer_temperature + 0.4
                for j in range(num_attempts):
                    logging.log(25, 'Expansion attempt ' + str(j) + ' with temperature ' + str(current_temperature))
                    success, characters, character_strings, infer_attributes_string = node.expand(fixed_prefix,
                                additional_prefix,
                                model, 
                                model_string, 
                                characters,
                                character_strings,
                                infer_attributes_string,
                                max_tokens=max_tokens, 
                                num_attempts=1,
                                n=n, 
                                min_sections=min_sections, 
                                max_sections=max_sections, 
                                min_piece_length=min_piece_length,
                                max_piece_length=max_piece_length,
                                logit_bias=logit_bias,
                                repetition_strings=repetition_strings + [node.text for node in self.root().depth_first_traverse() if node not in self.path_to_root()],
                                temperature=current_temperature,
                                char_model_string=char_model_string) # it may fail, and that's fine
                    if success:
                        break
                    else:
                        current_temperature += 0.1 # try again with a higher temperature
            logging.log(25, 'Done expanding outline level ' + str(level))
            logging.log(23, str(self))
        return characters, character_strings, infer_attributes_string
    
    def select_scene(self, model, model_string, infer_attributes_string):
        # add scene for a given outline item, based on context
        logging.log(22, 'detecting scene for outline section: ' + self.text)
        prefix = infer_attributes_string + '\n\nOutline:\n\n' + self.prefix(omit_numbers_up_to=self.depth()-1, include_scene=True) + ' ' + SCENE_PREFIX
        suffix = '\n\n' + self.suffix(omit_numbers_up_to=self.depth()-1, include_scene=True)
        logging.log(21, 'SCENE PREFIX: ' + prefix)
        logging.log(21, 'SCENE SUFFIX: ' + suffix)
        logit_bias = get_repetition_logit_bias(model.tokenizer, ' not unspecified unimportant undisclosed unknown', -100) # some bad strings that we don't want
        output = model([prefix], suffixes=[suffix], stop=['\n'], num_completions=5, model_string=model_string, logit_bias=logit_bias) # ban not, unimportant, unspecified
        filter_words = ['Chapter', 'present', 'future', 'scene', '"', 'movie']
        output = [o for o in output if all([f.lower() not in o.lower() for f in filter_words])]
        if len(output) == 0 or len(output[0].strip()) == 0:
            logging.warning('Warning: no scene detected')
        else:
            try:
                self.scene = split_paragraphs(output[0].strip(), mode='sentence')[0].strip()
            except:
                logging.warning('Warning: failed to add scene: ' + output[0])
        logging.log(22, 'scene: ' + self.scene)

    def select_characters(self, model, model_string, characters, character_strings, infer_attributes_string, num_samples=5, num_sample_sets=3, max_iters=10, name_max_length=20, max_characters=5, predecessors=None, char_model_string=None):
        # select from inventory of characters which are going to appear in this outline point

        def update_character_description(entity, detected_description, character_strings):
            entity.add_new_outline_description(self, model, model_string, character_strings, num_samples=num_samples, additional_description=detected_description, other_nodes=predecessors[1:] if predecessors is not None else None)

        def description_contradiction(name, detected_description, character_strings, threshold=0.5):
            entity = character_strings[name]
            premise = self.text + ' ' + entity.get_outline_description_up_to_node(self)
            _, noncontradiction_nll = score_entailment(premise, detected_description)
            if math.log(threshold) > -noncontradiction_nll:
                logging.log(22, 'DESCRIPTION CONTRADICTION DETECTED: ' + premise + ' ' + detected_description + str(noncontradiction_nll))
                return True
            return False

        def create_characters_prefix(seed=None, context=None):
            values = list(character_strings.values())
            descs = ['Full Name: ' + entry.name + ' ' + entry.get_outline_description_up_to_node(self) for entry in values]
            if context is not None:
                similarities = sentence_similarity(context, descs)
                descs = sorted(descs, key=lambda d: similarities[descs.index(d)]) # sort by similarity to context, highest similarity last (most recent)
            else:
                if seed is not None:
                    random.seed(seed)
                random.shuffle(descs)
            return '\n\n'.join(descs) + '\n\n----------------------------------\n\n'
        
        def unify_names(possible_names, character_strings, selected_entities):
            for i in range(len(possible_names)):
                matched_entities, _, _ = deduplicate_match_entities([possible_names[i]], character_strings.keys(), prioritized_names=selected_entities)
                if len(matched_entities) > 0:
                    if len(matched_entities) > 1:
                        logging.warning('WARNING: multiple matches for name', possible_names[i])
                    possible_names[i] = list(matched_entities)[0]
            return possible_names
        
        def add_name(name, selected_entities, characters, character_strings, infer_attributes_string, add_directly=True):
            success = False
            used_name_words = set()
            for existing_name in character_strings.keys():
                used_name_words.update(existing_name.strip().split())
            name = name.strip()
            matched_entities, _, _ = deduplicate_match_entities([name], character_strings.keys(), prioritized_names=selected_entities)
            if len(matched_entities) > 0:
                assert len(matched_entities) == 1
                name = list(matched_entities)[0]
                success = True # count as success even if name already in list
                if name not in selected_entities:
                    if add_directly:
                        selected_entities.append(name)
                    used_name_words.update(name.strip().split())
            else: 
                success = False
            return success, name, selected_entities, characters, character_strings, infer_attributes_string
        
        def return_logging(selected_entities):
            logging.log(22, 'final pre-truncation ' + str(selected_entities))  
            self.selected_entities = selected_entities[:max_characters]
            if len(self.selected_entities) == 0: # just take previous node's entities
                logging.warning('WARNING: no entities found, taking previous node\'s entities')
                predecessor = None
                if predecessors is not None:
                    predecessor = predecessors[-1]
                elif self.predecessor(max_depth=self.depth()) is not None:
                    predecessor = self.predecessor(max_depth=self.depth())
                if predecessor is not None:
                    self.selected_entities = deepcopy(predecessor.selected_entities)
            logging.log(22, 'final: ' + str(self.selected_entities))

        # detect characters mentioned in the outline point; these might be unnamed descriptions or pronouns
        logging.log(22, 'detecting chars for outline section: ' + self.text)
        detected_chars_unnamed = []
        prompt_question = 'List all characters mentioned in this sentence.'
        prefix = self.text.strip() + '\n\n{}\n'.format(prompt_question)
        has_next = True
        for i in range(max_iters):
            if not has_next:
                break
            prefix += '\n{}.'.format(i+1)
            samples = model([prefix], stop=None, model_string=model_string, num_completions=num_samples * num_sample_sets, max_tokens=name_max_length)
            logging.log(22, str(samples))
            has_next = sum([1 for s in samples if str(i+2) + '.' in s]) >= num_samples * num_sample_sets / 2 # we got a stop token indicating we want to continue the list
            samples = [s.strip().split('\n')[0] for s in samples]
            samples = [s.strip().split(str(i+2) + '.')[0] for s in samples]
            num_empty_samples = sum([len(s.strip()) == 0 for s in samples])
            if num_empty_samples > num_samples * num_sample_sets / 2:
                break
            samples = [s for s in samples if len(s.strip()) > 0 and s.strip() not in prompt_question]
            if len(samples) == 0:
                break
            best_name = Counter(samples).most_common(1)[0][0].strip()
            if best_name.lower() in prompt_question:
                best_name = best_name.lower()
            tags = pos_tag(best_name)
            if all([t.tag not in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP'] for t in tags]):
                break
            if sum([1 for s in samples if best_name in s]) < 3:
                break
            is_entity, logprobs = infer_is_character(best_name, self.text, model, model_string, plural=True, threshold=0.2, return_logprobs=True) # high recall, lower precision
            logging.log(21, str(logprobs))
            if is_entity:
                detected_chars_unnamed.append(best_name)
                prefix += ' ' + best_name
            else:
                break
        logging.log(22, 'detected chars unnamed: ' + str(detected_chars_unnamed))
        selected_entities, _, _ = deduplicate_match_entities(detect_entities(self.text), character_strings.keys())
        selected_entities = list(selected_entities)
        for name in selected_entities:
            update_character_description(character_strings[name], '', character_strings)
        updated_names = set(selected_entities)
        if len(selected_entities) >= max_characters:
            return_logging(selected_entities)
            return characters, character_strings, infer_attributes_string
        
        if predecessors is None:
            predecessors = [self.predecessor(max_depth=self.depth())]
            while predecessors[0] is not None and predecessors[0].depth() == self.depth():
                predecessors = [predecessors[0].predecessor(max_depth=self.depth())] + predecessors
            predecessors = [p for p in predecessors if p is not None and len(p.text.strip()) > 0]
        if len(predecessors) > 0:
            predecessor_chars = []
            for p in predecessors:
                for e in p.selected_entities:
                    if e not in predecessor_chars:
                        predecessor_chars.append(e)
            predecessor_text = 'The characters in the following context include: ' + ', '.join(predecessor_chars) + '. '
            predecessor_text += '\n\nPrevious context: ' + ' '.join([p.text for p in predecessors])
        else:
            predecessor_text = ''
        logging.log(21, 'predecessor text: ' + predecessor_text)
        for character in detected_chars_unnamed:
            # check if name already matched, and if so select it
            logging.log(22, 'processing possibly unnamed characters: ' + character)
            matched_entities, _, _ = deduplicate_match_entities([character], character_strings.keys(), prioritized_names=selected_entities)
            if len(matched_entities) > 0:
                logging.log(22, 'matched to existing character: ' + str(matched_entities))
                assert len(matched_entities) == 1
                name = list(matched_entities)[0]
                success, name, selected_entities, characters, character_strings, infer_attributes_string = add_name(name, selected_entities, characters, character_strings, infer_attributes_string, add_directly=False)
                if success and name not in updated_names:
                    detected_description = '{} is {}.'.format(character, name)
                    if not description_contradiction(name, detected_description, character_strings):
                        selected_entities.append(name)
                        update_character_description(character_strings[name], detected_description, character_strings)
                        updated_names.add(name)
                        if len(selected_entities) >= max_characters:
                            return_logging(selected_entities)
                            return characters, character_strings, infer_attributes_string

            # detect if the entity string refers to a single character or a group
            is_group = infer_is_group(character, self.text, model, model_string)
            if not is_group:
                # prompt to try to detect the char's name, and select it if found
                single_prompt = predecessor_text + '\n\nCurrent passage: ' + self.text + '\n\n{}\'s full name:'.format(character.strip())
                logging.log(21, 'single prompt: ' + single_prompt)
                possible_names = []
                current_prompt = create_characters_prefix(context=predecessor_text + self.text + '\n\nWho is {}?'.format(character.strip())) + single_prompt
                logging.log(21, 'FINAL SINGLE PROMPT: ' + current_prompt)
                possible_names += model([current_prompt], model_string=model_string if char_model_string is None else char_model_string, num_completions=num_samples*num_sample_sets, max_tokens=name_max_length)
                logging.log(22, 'possible single names: ' + str(possible_names))
                for stop_s in ['\n', ',', '-', ';']:
                    possible_names = [n.strip().split(stop_s)[0].strip() for n in possible_names if len(n.strip()) > 0]
                possible_names = [split_paragraphs(n, mode='sentence')[0].strip() for n in possible_names]
                for indicator in ['called', 'known as', 'named', 'name is', 'name will be', 'name:', ' is']:
                    possible_names = [n.split(indicator)[1].strip() if indicator in n else n for n in possible_names]
                possible_names = [n.rstrip(string.punctuation) for n in possible_names]
                possible_names = [n for n in possible_names if not any([word in n for word in BANNED_NAME_WORDS])]
                possible_names = [n for n in possible_names if simple_name_check(n) and n.lower() not in single_prompt]
                logging.log(21, 'pre-unified single names: ' + str(possible_names))
                possible_names = unify_names(possible_names, character_strings, selected_entities)
                logging.log(21, 'unified single names: ' + str(possible_names))
                if len(possible_names) > 0:
                    possible_names_counter = Counter(possible_names)
                    name = possible_names_counter.most_common(1)[0][0]
                    if sum([1 for n in possible_names if name in n]) >= num_samples * num_sample_sets / 3:
                        logging.log(22, 'adding single name: ' + name)
                        success, name, selected_entities, characters, character_strings, infer_attributes_string = add_name(name, selected_entities, characters, character_strings, infer_attributes_string, add_directly=False)
                        if success and name not in updated_names:
                            detected_description = '{} is {}.'.format(character, name)
                            if not description_contradiction(name, detected_description, character_strings):
                                selected_entities.append(name)
                                update_character_description(character_strings[name], detected_description, character_strings)
                                updated_names.add(name)
                                if len(selected_entities) >= max_characters:
                                    return_logging(selected_entities)
                                    return characters, character_strings, infer_attributes_string
                        continue
                    else:
                        possible_names = [] # go into group prediction branch
            else:
                # try to find up to 2 characters that fit in this group
                added_names = []
                group_prompt = predecessor_text + '\n\nCurrent passage: ' + self.text + '\n\n{}\'s full names:\n\n'.format(character.strip())
                logging.log(21, 'group prompt: ' + group_prompt)
                has_next = True
                for i in range(2):
                    if not has_next:
                        break
                    logging.log(22, 'group name: ' + character)
                    if i >= 1:
                        group_prompt += '{}.'.format(i+1)
                    possible_names = []
                    possible_names += model([create_characters_prefix(context=predecessor_text + self.text + '\n\nWho are {}?'.format(character.strip())) + group_prompt], stop=[' and', ' the', 'The '], model_string=model_string if char_model_string is None else char_model_string, num_completions=num_samples*num_sample_sets, max_tokens=name_max_length)
                    logging.log(22, 'possible group names: ' + str(possible_names))
                    if i == 0:
                        group_prompt += '{}.'.format(i+1) # for some reason having this at step 0 messes it up
                    has_next = (i == 0) or sum([1 for n in possible_names if str(i+2) + '.' in n]) >= num_samples * num_sample_sets / 2
                    possible_names = [n for n in possible_names if len(n.strip()) > 0]
                    possible_names = [n.replace(str(i+1) + '.', '') for n in possible_names]
                    for stop_s in ['\n', ',', '-', ';']:
                        possible_names = [n.strip().split(stop_s)[0].strip() for n in possible_names if len(n.strip()) > 0]
                    possible_names = [split_paragraphs(n, mode='sentence')[0].strip() for n in possible_names]
                    for indicator in ['called', 'known as', 'named', 'name is', 'name will be', 'name:', ' is', ' are']:
                        possible_names = [n.split(indicator)[1].strip() if indicator in n else n for n in possible_names]
                    possible_names = [n for n in possible_names if not any([word in n for word in BANNED_NAME_WORDS])]
                    possible_names = [n for n in possible_names if simple_name_check(n) and n not in added_names and n.lower() not in group_prompt + 'list']
                    logging.log(21, 'pre-unified group names: ' + str(possible_names))
                    possible_names = unify_names(possible_names, character_strings, selected_entities)
                    logging.log(21, 'unified group names: ' + str(possible_names))
                    if len(possible_names) > 0:
                        name = Counter(possible_names).most_common(1)[0][0].strip()
                        if sum([1 for n in possible_names if name in n]) < 3:
                            break
                        logging.log(22, 'adding group name: ' + name)
                        added_names.append(name)
                        success, name, selected_entities, characters, character_strings, infer_attributes_string = add_name(name, selected_entities, characters, character_strings, infer_attributes_string, add_directly=False)
                        if success:
                            if name not in updated_names:
                                processed_character = character.strip()
                                detected_description = 'One of {} is {}.'.format(processed_character, name)
                                if not description_contradiction(name, detected_description, character_strings):
                                    selected_entities.append(name)
                                    update_character_description(character_strings[name], detected_description, character_strings)
                                    updated_names.add(name)
                                    if len(selected_entities) >= max_characters:
                                        return_logging(selected_entities)
                                        return characters, character_strings, infer_attributes_string
                        else:
                            break
                        group_prompt += ' ' + name + '\n\n'
                    else:
                        break

        return_logging(selected_entities)
        return characters, character_strings, infer_attributes_string


if __name__=='__main__':
    import pdb; pdb.set_trace()