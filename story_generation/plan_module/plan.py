from copy import deepcopy
import pickle
import string
import logging

from story_generation.edit_module.entity import *
from story_generation.rewrite_module.heuristics import *
from story_generation.plan_module.outline import *
from story_generation.plan_module.name_util import *
from story_generation.common.util import *
from story_generation.common.data.split_paragraphs import *

def create_character_summary(character_strings):
    character_summary = ''
    keys = list(character_strings.keys())
    for i in range(len(keys)):
        character_summary += "{}.\n\nFull Name: {}\n\nCharacter Portrait: {}".format(str(i+1), keys[i], character_strings[keys[i]].description)
        if i != len(keys) - 1:
            character_summary += '\n\n'
    return character_summary


def generate_initial_entity_strings(premise, setting, instruct_model, min_entities=3, max_entities=3, max_description_length=48, model_string='text-davinci-002'):
    # TODO figure out alternative stopping criterion for generating initial characters?
    initial_characters_prompt = "Premise: " + premise.strip() + '\n\n' + 'Setting: ' + setting.strip() + '\n\nList the names and details of all major characters.'
    banned_name_words = deepcopy(BANNED_NAME_WORDS)
    name_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, ' '.join(NAME_BIAS_WORDS), bias=-5, bias_common_tokens=True)
    name_logit_bias[198] = -5 # also penalize newline, although we want it eventually eventually

    character_strings = {}
    characters_prompt = initial_characters_prompt
    used_name_words = set()
    for i in range(max_entities):
        previous_characters_prompt = characters_prompt
        characters_prompt += '\n\n' + str(i+1) +'.\n\nFull Name:'
        for _ in range(5):
            name_continuations = instruct_model([characters_prompt], top_p=1, temperature=1.2, logit_bias=name_logit_bias, stop=['\n', '(', ':'], num_completions=10, max_tokens=10, model_string=model_string)
            filtered_name_continuations = []
            for name in name_continuations:
                name_is_good = True
                for word in name.strip().split():
                    if word.strip(string.punctuation) not in characters_prompt and sum([1 for n in name_continuations if word in n]) >= 2: # >=2 because it's in the name itself and at least 1 other
                        name_is_good = False
                        logging.log(23, 'bad name word ' + word + ' in ' + name + ' due to repetition')
                        for tok in instruct_model.tokenizer.encode(word) + instruct_model.tokenizer.encode(' ' + word):
                            name_logit_bias[tok] = -100
                    if word not in used_name_words:
                        for used_word in used_name_words:
                            if word.lower() in used_word.lower() or used_word.lower() in word.lower():
                                name_is_good = False
                                logging.log(23, 'bad name word ' + word + ' in ' + name + ' due to overlap with used word ' + used_word)
                                break
                if not name_is_good:
                    continue
                if not any([key.strip() in name.strip() or name.strip() in key.strip() for key in character_strings]) and len(name.strip()) > 0 and all([piece.strip()[0].isupper() for piece in name.strip().split()]) and all([word.lower() not in name.lower() for word in banned_name_words]): # check that names are capitalized to filter out some bad cases
                    if not any([word.strip('"') not in initial_characters_prompt and word.lower() in initial_characters_prompt.lower() for word in name.strip().split()]) and sum([1 for letter in name if letter.isupper()]) == len(name.strip().split()): # don't allow cases where it dodged our checks by changing case
                        if simple_name_check(name):
                            filtered_name_continuations.append(name)
            if len(filtered_name_continuations) > 0:
                break
        if len(filtered_name_continuations) == 0:
            if len(character_strings) >= min_entities: # just settle for fewer characters
                break
            else:
                logging.warning('Warning: failed to generate enough characters')
                raise ValueError
        filtered_name_continuations = sorted(filtered_name_continuations, key=lambda x: abs(2 - len(x.strip().split()))) # ideally want the full name, not just the first word, and want roughly 2 words
        selected_name = filtered_name_continuations[0].strip()
        name_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, selected_name, bias=-1, bias_common_tokens=True, existing_logit_bias=name_logit_bias, increment=True) # bias against already used names
        name_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, selected_name.strip().split()[0], bias=-100, bias_common_tokens=True, existing_logit_bias=name_logit_bias) # bias heavily against already used first names
        banned_name_words.append(selected_name.strip().split()[0])
        # characters_prompt += ' ' + selected_name
        characters_prompt += ' ' + selected_name + '\n\nCharacter Portrait: ' + selected_name.strip() + ' is'
        found_acceptable_description = False
        logging.log(21, 'CHARACTERS PROMPT', characters_prompt)
        for j in range(5):
            description_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, initial_characters_prompt + ' ' + ' '.join(NAME_BIAS_WORDS), bias=-2**(j+1), bias_common_tokens=False)
            name_tokens = set(sum([instruct_model.tokenizer.encode(ent) + instruct_model.tokenizer.encode(' ' + ent) for ent in character_strings.keys()], []))
            for tok in name_tokens:
                if tok in description_logit_bias:
                    del description_logit_bias[tok]
            descriptions = instruct_model([characters_prompt], stop='\n', logit_bias=description_logit_bias, num_completions=10, max_tokens=max_description_length, cut_sentence=True, model_string=model_string)
            logging.log(21, 'DESCRIPTIONS', descriptions)
            descriptions = [d for d in descriptions if len(d.strip()) > 0 and len(instruct_model.tokenizer.encode(d)) < max_description_length] # not empty, and terminated naturally rather than due to max length
            descriptions = sorted(descriptions, key=lambda d: calculate_repetition_length_penalty(d, [characters_prompt]))
            if len(descriptions) > 0 and calculate_repetition_length_penalty(descriptions[0], [characters_prompt]) < 1:
                found_acceptable_description = True
                break
        if not found_acceptable_description:
            logging.log(22, 'Warning: no acceptable description found for character ' + selected_name)
            characters_prompt = previous_characters_prompt
            continue
        description = descriptions[0]
        description = description[:len(description) - len(description.lstrip())] + split_paragraphs(description, mode='sentence')[0].strip()
        characters_prompt += description
        character_strings[selected_name.strip()] = Entity(selected_name.strip(), description=selected_name.strip() + ' is' + description, is_character=True)
        used_name_words.update(selected_name.strip().split())
    infer_attributes_string = premise.strip() + '\n\n' + setting.strip() + '\n\n' + '\n\n'.join([ent.description for ent in character_strings.values()])
    return create_character_summary(character_strings), character_strings, infer_attributes_string


def generate_outline(args, premise, setting, characters, character_strings, infer_attributes_string, instruct_model, max_tokens, min_sections=2, max_sections=5, outline_levels=1, model_string='text-davinci-002', previous_outline=None):
    premise_setting_chars = "Premise: " + premise.strip() + '\n\n' + 'Setting: ' + setting.strip() + '\n\n' + 'Characters:\n\n__CHARACTERS__' # __CHARACTERS__ will be replaced later
    outline_prompt = '\n\nOutline the main plot points of the story.'
    found_acceptable_outline = False
    banned_outline_tokens = set(sum([instruct_model.tokenizer.encode(ent) + instruct_model.tokenizer.encode(' ' + ent) for ent in ['He', 'She', 'They', 'It', ' He', ' She', ' They', ' It', 'How', 'What', ' How', ' What', 'Fill', ' Fill', 'Add', ' Add']], []))
    bad_outline_words = ['is', 'are', 'was', 'had', 'has', 'have', 'she', 'he', 'they']
    bad_outline_tokens = sum([[word, ' ' + word, word[0].upper() + word[1:], ' ' + word[0].upper() + word[1:]] for word in bad_outline_words], [])
    bad_outline_tokens = set(sum([instruct_model.tokenizer.encode(ent) + instruct_model.tokenizer.encode(' ' + ent) for ent in bad_outline_tokens], []))
    if previous_outline is not None:
        outline = previous_outline
        found_acceptable_outline = True
    else:
        for attempt_num in range(5):
            logging.log(24, 'Generating initial outline attempt ' + str(attempt_num))
            outline_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, outline_prompt, -2**(attempt_num+1))
            name_tokens = set(sum([instruct_model.tokenizer.encode(ent) + instruct_model.tokenizer.encode(' ' + ent) for ent in character_strings.keys()], []))
            for tok in name_tokens:
                if tok in outline_logit_bias:
                    del outline_logit_bias[tok]
            for tok in banned_outline_tokens:
                outline_logit_bias[tok] = -100
            for tok in bad_outline_tokens:
                outline_logit_bias[tok] = -10
            outline = OutlinePiece('', None)
            found_acceptable_outline, characters, character_strings, infer_attributes_string = outline.expand(premise_setting_chars, 
                            outline_prompt, 
                            instruct_model, 
                            model_string, 
                            characters,
                            character_strings,
                            infer_attributes_string,
                            max_tokens=max_tokens, 
                            n=10, 
                            min_sections=min_sections, 
                            max_sections=max_sections, 
                            repetition_strings=[characters],
                            logit_bias=outline_logit_bias,
                            char_model_string=args.outline_char_model_string)
            if found_acceptable_outline:
                break
            
    if not found_acceptable_outline:
        logging.warning('Warning: didn\'t find acceptable top-level outline after 5 attempts')
        raise ValueError
    
    logging.log(23, str(outline))

    outline_expansion_logit_bias = {}
    for tok in banned_outline_tokens:
        outline_expansion_logit_bias[tok] = -100
    for tok in bad_outline_tokens:
        outline_expansion_logit_bias[tok] = -5
    for level in range(outline_levels+1):
        logging.log(23, 'Top-level resampling at outline level ' + str(level))
        if level > 0 and len(list(outline.list_children_at_depth(level+1))) > 0: # restarting from a previous outline that already generated at this depth
            assert previous_outline is not None
            continue
        outline_nodes = outline.list_children_at_depth(level)
        if level == outline_levels: # don't need to expand again, we're done
            break
        for node in outline_nodes: # recursively generate, then resample at each level of the hierarchy to improve global->local quality/consistency
            characters, character_strings, infer_attributes_string = node.recursively_expand_to_depth(
                node.depth()+1,
                premise_setting_chars + '\n\nOutline:\n\n', 
                'List the main events that occur under this heading, starting from the beginning.',
                instruct_model, 
                model_string,
                characters,
                character_strings,
                infer_attributes_string,
                max_tokens=max_tokens, 
                num_attempts=3,
                n=10, 
                min_sections=min_sections, 
                max_sections=max_sections, 
                min_piece_length=3,
                max_piece_length=50,
                repetition_strings=[characters],
                logit_bias=outline_expansion_logit_bias,
                temperature=None,
                expand_self=(level > 0), # we already expanded once at the very top level, no need to redo
                char_model_string=args.outline_char_model_string,
            )

    # remove unused chars
    original_characters = list(character_strings.keys())
    used_characters = set(sum([node.selected_entities for node in outline.depth_first_traverse()], []))
    character_strings = {char: character_strings[char] for char in used_characters}
    characters = create_character_summary(character_strings)
    infer_attributes_string = '\n\n'.join([premise, setting] + [character_strings[char].description for char in original_characters if char in used_characters])

    for leaf in outline.leaves():
        leaf.select_scene(instruct_model, model_string, infer_attributes_string)

    logging.log(25, 'Finished resampling outline')
    logging.log(23, str(outline))
    outline_last_node = list(outline.depth_first_traverse())[-1]
    for ent in character_strings.values():
        logging.log(23, ent.name + ': ' + str(ent.get_outline_description_up_to_node(outline_last_node)))

    return outline, (characters, character_strings, infer_attributes_string)


def load_plan_info(plan_file):
    with open(plan_file, 'rb') as f:
        save_info = pickle.load(f)
    return save_info


def generate_plan_info(args, instruct_model, include_outline=True, model_string='text-davinci-002'):
    while True:
        try:
            if args.premise is not None:
                premise = args.premise
            else:
                premise_prompt = "Write a premise for a short story."
                max_premise_tokens = 128
                premise = (instruct_model([premise_prompt], top_p=1, temperature=1.2, max_tokens=max_premise_tokens, model_string=model_string)[0]) # more diversity with premises with higher temp
                if len(instruct_model.tokenizer.encode(premise)) == max_premise_tokens: # likely we got cutoff instead of ending naturally
                    logging.warning('premise too long, retrying')
                    raise ValueError
                premise = premise.strip()

            logging.log(25, 'Premise: ' + premise)

            success = False
            for attempt_num in range(10): # avoid resampling good premises for fairness
                try:
                    if args.outline_restart_pkl is None:
                        setting_prompt = "Premise: " + premise.strip() + '\n\nDescribe the setting of the story.\n\nThe story is set in'
                        settings = []
                        for i in range(5):
                            banned_setting_words = ['unknown', 'unnamed', 'unspecified', 'Unknown', 'Unnamed', 'Unspecified']
                            setting_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, setting_prompt, -2**(i+1))
                            settings = instruct_model([setting_prompt], num_completions=10, logit_bias=setting_logit_bias, max_tokens=32, cut_sentence=True, model_string=model_string)
                            settings = [split_paragraphs(s, mode='sentence')[0] for s in settings]
                            settings = [s.strip() for s in settings if calculate_repetition_length_penalty(s, [premise]) == 0 and not any([w in s.lower() for w in banned_setting_words])]
                            settings = ['The story is set in ' + s for s in settings]
                            if len(settings) > 0:
                                break
                        setting = settings[0]

                        logging.log(25, 'Setting: ' + setting)

                        logging.log(24, 'Generating characters and outline...')

                        characters, character_strings, infer_attributes_string = generate_initial_entity_strings(premise, setting, instruct_model, max_entities=args.max_characters, max_description_length=args.entity_description_max_length, model_string=model_string)

                        logging.log(25, 'Initial Characters (will be filtered down later): ' + str(characters))

                        for entity in character_strings.values():
                            logging.log(23, entity)

                        outline = None
                        if not include_outline:
                            break
                    else:
                        save_info = load_plan_info(args.outline_restart_pkl)
                        premise = save_info['premise']
                        setting = save_info['setting']
                        characters = save_info['characters']
                        character_strings = save_info['character_strings']
                        infer_attributes_string = save_info['infer_attributes_string']
                        outline = save_info['outline']
                    outline_max_tokens = 128
                    outline, (characters, character_strings, infer_attributes_string) = generate_outline(args, premise, setting, characters, character_strings, infer_attributes_string, instruct_model, outline_max_tokens, outline_levels=args.outline_levels, previous_outline=outline)

                    logging.log(25, 'FINAL PLAN')
                    logging.log(25, 'Premise: ' + premise)
                    logging.log(25, 'Setting: ' + setting)
                    logging.log(25, 'Characters:')
                    for entity in character_strings.values():
                        logging.log(25, entity.name + ': ' + entity.description)
                    logging.log(25, 'Outline:')
                    logging.log(25, str(outline))
                    success = True
                    break
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logging.log(23, 'Plan generation failed: ' + str(e))
            if not success:
                logging.warning('WARNING: Could not generate a valid setup after 10 attempts.')
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            logging.warning('Exception ' + str(e))
            continue
    plan_info = {'premise': premise,
                'setting': setting,
                'characters': characters,
                'character_strings': character_strings,
                'outline': outline,
                'infer_attributes_string': infer_attributes_string}
    return plan_info


def infer_initial_attributes_from_plan(plan_info, instruct_model):
    character_strings = plan_info['character_strings']
    infer_attributes_string = plan_info['infer_attributes_string']
    made_changes = False
    for entity in character_strings.values():
        if len(entity.attributes) == 0 and entity.is_character: # unlikely that we inferred nothing from an initial setup passage
            made_changes = True
            entity.infer_attributes(infer_attributes_string, instruct_model, other_names=[name for name in character_strings.keys() if name != entity.name])
    if made_changes:
        complete_mutual_relations(character_strings, instruct_model)