import argparse
import os
from copy import deepcopy
import pickle
import logging

from story_generation.edit_module.entity import *
from story_generation.draft_module.beam_candidate import BeamCandidate
from story_generation.plan_module.plan import *
from story_generation.common.util import *
from story_generation.common.data.data_util import add_data_args
from story_generation.common.summarizer.summarizer_util import add_summarizer_args
from story_generation.common.summarizer.models.opt_summarizer import OPTSummarizer
from story_generation.common.summarizer.models.gpt3_summarizer import GPT3Summarizer
from story_generation.common.controller.controller_util import add_controller_args, load_controller
from story_generation.common.data.split_paragraphs import *


if __name__=='__main__':
    parser = argparse.ArgumentParser() # parameter defaults are set to values used in paper
    parser = add_general_args(parser)
    parser = add_data_args(parser)
    parser = add_summarizer_args(parser)
    parser = add_controller_args(parser)

    # SAVE/LOAD PLAN/LOGS
    parser.add_argument('--load-outline-file', type=str, help='load outline from this file')
    parser.add_argument('--save-outline-file', type=str, help='save outline to this file')
    parser.add_argument('--save-complete-file', type=str, help='save completed beam object to this file')
    parser.add_argument('--inspect-setup', action='store_true', help='inspect setup with pdb')
    parser.add_argument('--outline-restart-pkl', type=str, help='restart outline generation from this pickle file of a lower depth outline')

    # ALTERNATE MODES
    parser.add_argument('--setup-only', action='store_true', help='exit after generating the premise/setup/outline')
    parser.add_argument('--no-editor', action='store_true', help='do not use editor to edit text for detected contradictions. note: not heavily tested with editor, but leftover from re3 if you want to try it')

    # SEARCH SIZE / BEAM PARAMETERS
    parser.add_argument('--max-candidates', type=int, default=8, help='max number of candidates to generate at each step by each beam candidate')
    parser.add_argument('--max-beam-size', type=int, default=1, help='max number of beam candidates to generate at each step')
    parser.add_argument('--beam-max-difference', type=float, default=1, help='max difference between beam scores')

    # OUTLINE PARAMETERS
    parser.add_argument('--premise', type=str, default=None, help='premise to use for generation; will make a new one if None')
    parser.add_argument('--max-characters', type=int, default=10, help='max number of characters to generate for outline')
    parser.add_argument('--entity-description-max-length', type=int, default=48, help='max number of tokens to use per initial base entity description')
    parser.add_argument('--outline-levels', type=int, default=3, help='num levels of hierarchy in outline')
    parser.add_argument('--generation-outline-levels', type=int, default=None, help='num levels of hierarchy in outline to use at generation time; use all if None')
    parser.add_argument('--plan-model-string', type=str, default='text-davinci-002', help='gpt3 model string to use in planning')
    parser.add_argument('--outline-char-model-string', type=str, default='text-davinci-002', help='gpt3 model string to use when predicting char names during planning for unnamed char strings')

    # EARLY STOPPING / LENGTH PARAMETERS
    parser.add_argument('--continuation-threshold', type=float, default=0, help='if alignment score is worse by at least this much, move on to next outline point; 10000 turns this off')
    parser.add_argument('--early-stop-threshold', type=float, default=-0.5, help='early skipping to next section disallowed if alignment score is worse than this; 10000 turns this off')
    parser.add_argument('--skip-threshold', type=float, default=-10, help='early skipping to next section if best alignment score is worse than this; -10000 turns this off')
    parser.add_argument('--max-continuation-substeps', type=int, default=8, help='max number of continuation candidates to generate at each step')
    parser.add_argument('--max-ending-continuations', type=int, default=3, help='max number of continuation steps for ending the story')

    # PROMPT PARAMETERS
    parser.add_argument('--previous-prompt-length', type=int, default=256, help='length of previously generated text in prompt')
    parser.add_argument('--max-entity-context-tokens', type=int, default=256, help='max number of tokens to use for entity context')
    
    # GENERATION PARAMETERS
    parser.add_argument('--extension-method', type=str, default='opt-control', choices=['gpt3', 'opt-control'], help='model/method to use for main story generation')
    parser.add_argument('--repetition-penalty-weight', type=float, default=5, help='weight of repetition penalty')
    parser.add_argument('--draft-top-p', type=float, default=1, help='top p to use during drafting')
    parser.add_argument('--draft-model-string', type=str, default='davinci', help='gpt3 model string to use in extending story, if using gpt3')
    parser.add_argument('--cut-sentence', action='store_true', default=False, help='cut incomplete sentence at end of generation')
    parser.add_argument('--control-strength-substep-increment', type=float, default=3, help='increment for control strength for each substep')
    parser.add_argument('--max-control-strength', type=float, default=10, help='max control strength after incrementing')
    parser.add_argument('--include-future-context', action='store_true', help='include future context in prompt')
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.getLogger().handlers = []
    logging.basicConfig(format='%(message)s', filename=args.log_file, level=args.log_level)

    gpt3_model = GPT3Summarizer(args) # naming is a relic of some old preliminary experiments; it's just a gpt3 interface
    controllers = [load_controller(args, i) for i in range(len(args.controller))]
    assert len(controllers) >= 3 # in order: relevance, coherence, fudge, outline order (outline order only needed if you don't specify a load_outline_file)
    
    # load or create new plan/outline
    if args.load_outline_file is not None:
        plan_info = load_plan_info(args.load_outline_file)
    else:
        if os.path.exists(args.save_outline_file):
            print('save outline file already exists, exiting')
            sys.exit()
        assert len(controllers) == 4
        assert controllers[3] is not None
        set_outline_order_controller(controllers[3])
        plan_info = generate_plan_info(args, gpt3_model, model_string=args.plan_model_string)
        if args.save_outline_file is not None:
            os.makedirs(os.path.dirname(args.save_outline_file), exist_ok=True)
            with open(args.save_outline_file, 'wb') as f:
                pickle.dump(plan_info, f)
    if args.setup_only:
        print('done making setup')
        if args.inspect_setup:
            import pdb; pdb.set_trace()
        sys.exit()

    if os.path.exists(args.save_complete_file):
        print('save complete file already exists, exiting')
        sys.exit()
    
    for i in range(3):
        assert controllers[i] is not None
    
    if not args.no_editor: # fill in the attributes if we need them, if they're not already present in the save
        infer_initial_attributes_from_plan(plan_info, gpt3_model)
    
    premise = plan_info['premise']
    setting = plan_info['setting']
    character_strings = plan_info['character_strings']
    outline = plan_info['outline']
    infer_attributes_string = premise + '\n\n' + setting + '\n\n' + '\n\n'.join([c.description for c in character_strings.values()])
    
    # delete nodes after specified depth if we're generating with limited-depth version of the outline
    if args.generation_outline_levels is not None: 
        nodes_at_depth = [node for node in outline.depth_first_traverse() if node.depth() == args.generation_outline_levels]
        for node in nodes_at_depth:
            node.children = []
        for leaf in outline.leaves(): # we have new leaves so we should recompute the scenes at leaves
            leaf.select_scene(gpt3_model, args.plan_model_string, infer_attributes_string) # since in practice scenes are only detected at leaves, reselect scenes at the new leaves
    
    # initialize entities and story objects
    opt_model = OPTSummarizer(args) if args.extension_method != 'gpt3' else None
    all_entities_dict = deepcopy(character_strings)
    all_entities_dict['Premise'] = Entity('Premise', description='Premise: ' + premise.strip(), is_character=False)
    all_entities_dict['Setting'] = Entity('Setting', description='Setting: ' + setting.strip(), is_character=False)
    all_paragraphs = []
    previous_alignment_score = -1e8
    beam = [BeamCandidate(args, 
                          all_entities_dict, 
                          infer_attributes_string,
                          outline,
                          model=gpt3_model,
                          opt_model=opt_model,
                          controllers=controllers)]
    if not args.no_editor:
        for candidate in beam:
            candidate.all_entities_dict = candidate.create_updated_entities('\n\n'.join([node.text for node in outline.leaves()]))
    section_list = outline.leaves()
    logging.log(25, 'num sections: ' + str(len(section_list)))
    outline_sections = [node for node in section_list]
    outline_sections[-1].text += ' This is the end of the story.'

    # machinery for restarting from partial story pkl
    for i in range(len(outline_sections)-1, -1, -1):
        if os.path.exists(args.save_complete_file + '.temp' + str(i)):
            logging.log(25, 'found temp file for section ' + str(i) + ', restarting from there')
            with open(args.save_complete_file + '.temp' + str(i), 'rb') as f:
                beam = pickle.load(f)
                for b in beam:
                    b.controllers = controllers
                    b.model = gpt3_model
                    b.opt_model = opt_model
            break
    
    # main generation loop
    for i in range(len(outline_sections)):
        logging.log(25, '\n\n\n\niteration at step ' + str(i))
        outline_section = outline_sections[i]
        if outline_section in beam[0].detailed_outline_section_history:
            logging.log(25, 'already generated this section')
            continue
        extensions = sum([b.extend(outline_section) for b in beam], [])
        extensions = sorted(extensions, key=lambda x: x.best_alignment_so_far, reverse=True)
        # pick the best extension plus up to max_beam_size that are below some alignment threshold
        new_beam = [extensions[0]]
        for extension in extensions[1:args.max_beam_size]:
            if extension.best_alignment_so_far > extensions[0].best_alignment_so_far - args.beam_max_difference: # variable beam size
                new_beam.append(extension)
        beam = new_beam
        for b in beam:
            b.condense_outline_sections(outline, section_list, i)
        logging.log(25, '\n\n\n\nend of iteration ' + str(i))
        for entity in beam[0].all_entities_dict.values():
            logging.log(22, entity)
        logging.log(24, beam[0].story(demarcated_outline_section=outline_section))
        with open(args.save_complete_file + '.temp' + str(i), 'wb') as f:
            for b in beam:
                b.controllers = None
            pickle.dump(beam, f)
            for b in beam:
                b.controllers = controllers
        if i > 0 and os.path.exists(args.save_complete_file + '.temp' + str(i-1)):
            os.remove(args.save_complete_file + '.temp' + str(i-1))
    
    # story-ending machinery using gpt3, without fudge
    for i in range(len(beam)):
        should_continue = True
        num_attempts = 0
        while should_continue:
            logging.log(25, 'BEAM ' + str(i) + ' ENDING ATTEMPT ' + str(num_attempts))
            beam[i], should_continue = beam[i].complete_ending()
            num_attempts += 1
            if num_attempts >= args.max_ending_continuations:
                break

    logging.log(25, '\n\n\n\nFINAL STORY')
    logging.log(25, beam[0].story())
    if args.save_complete_file is not None:
        with open(args.save_complete_file, 'wb') as wf:
            for b in beam:
                b.controllers = None
            pickle.dump(beam, wf)
