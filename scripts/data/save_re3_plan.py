import argparse
import pickle
import json

if __name__=='__main__': # for saving 1 level outlines with re3
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input pkl file', required=True)
    parser.add_argument('-o', '--output', help='Output json file', required=True)
    args = parser.parse_args()

    with open(args.input, 'rb') as rf:
        plan = pickle.load(rf)
    
    with open(args.output, 'w') as wf:
        save_info = {'premise': plan['premise'],
                     'setting': plan['setting'],
                     'outline_sections': [node.text for node in plan['outline'].children],
                     'character_info': [(ent.name, ent.description) for ent in plan['character_strings'].values()],}
        json.dump(save_info, wf, indent=4)
