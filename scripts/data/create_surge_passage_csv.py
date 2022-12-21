import argparse
import os
import random
import pickle
import csv

def parse_indices(indices):
    for index in indices.split(','):
        if '-' in index:
            start, end = index.split('-')
            for i in range(int(start), int(end) + 1):
                yield i
        else:
            yield int(index)

def process_passage(passage):
    passage = passage.replace('\n', '<br>')
    while '<br><br>' in passage:
        passage = passage.replace('<br><br>', '<br>')
    passage = passage.replace('<br>', '<br><br>')
    return passage.strip()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--premise-dir', help='Directory containing premise files', required=True)
    parser.add_argument('--data-dir1', type=str, required=True)
    parser.add_argument('--data-dir2', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--indices', type=str, required=True)
    parser.add_argument('--num-annotations', type=int, default=1)
    args = parser.parse_args()

    random.seed(0)

    indices = list(parse_indices(args.indices))

    stories = {}
    suffixes = set()
    for i in indices:
        print(i)
        stories[i] = {}
        with open(os.path.join(args.premise_dir, str(i) + '.pkl'), 'rb') as f:
            save_info = pickle.load(f)
        outline = save_info['outline']
        top_level_outline_items = [item for item in outline]
        for item in top_level_outline_items:
            stories[i][item] = {}
        for data_dir in [args.data_dir1, args.data_dir2]:
            suffix = data_dir.rstrip('/').split('/')[-1]
            suffixes.add(suffix)
            print(suffix)
            with open(os.path.join(data_dir, str(i) + '.pkl'), 'rb') as f:
                story_info = pickle.load(f)
                if type(story_info) == list:
                    sections = story_info[0].paragraphs_by_outline_section
                else:
                    sections = story_info['passages']
                if 're3' in suffix:
                    for key in list(sections.keys()): # text representation
                        if 'This is the end of the story.' in key:
                            sections[key.replace('This is the end of the story.', '').strip()] = sections[key]
                current_outline_item, current_passage = None, ''
                for node in outline.depth_first_traverse(include_self=False):
                    if node in top_level_outline_items:
                        print(node.text)
                        if current_outline_item is not None:
                            stories[i][current_outline_item][suffix] = process_passage(current_passage)
                            print(len(process_passage(current_passage).split()))
                        current_outline_item = node
                        current_passage = ''
                    if node in sections:
                        current_passage += ' ' + ''.join(sections[node])
                    if node.text in sections:
                        current_passage += ' ' + ''.join(sections[node.text])
                stories[i][current_outline_item][suffix] = process_passage(current_passage)
                print(len(process_passage(current_passage).split()))
        assert all([len(stories[i][item]) == 2 for item in top_level_outline_items])

    suffixes = list(suffixes)
    csv_rows = []
    individual_rows = {suffixes[0]: [], suffixes[1]: []}
    for i in indices:
        with open(os.path.join(args.premise_dir, str(i) + '.pkl'), 'rb') as f:
            save_info = pickle.load(f)
        for oi, outline_item in enumerate(list(stories[i].keys())):
            for j in range(args.num_annotations):
                random.shuffle(suffixes)
                line_id = str(i) + '-' + str(oi) + '-' + '-'.join(suffixes) + '-' + str(j)
                csv_rows.append({'id': line_id, 'premise': save_info['premise'], 'outline_item': outline_item.text, 'passage1': stories[i][outline_item][suffixes[0]], 'passage2': stories[i][outline_item][suffixes[1]]})
                individual_rows[suffixes[0]].append({'id': str(i) + '-' + str(oi) + '-' + suffixes[0] + '-' + str(j), 'premise': save_info['premise'], 'outline_item': outline_item, 'passage': stories[i][outline_item][suffixes[0]]})
                individual_rows[suffixes[1]].append({'id': str(i) + '-' + str(oi) + '-' + suffixes[1] + '-' + str(j), 'premise': save_info['premise'], 'outline_item': outline_item, 'passage': stories[i][outline_item][suffixes[1]]})
    random.shuffle(csv_rows)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'compare.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'premise', 'outline_item', 'passage1', 'passage2'])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    for suffix in suffixes:
        with open(os.path.join(args.save_dir, suffix + '.csv'), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'premise', 'outline_item', 'passage'])
            writer.writeheader()
            rows = individual_rows[suffix]
            random.shuffle(rows)
            writer.writerows(rows)
    
    individual_rows = sum(individual_rows.values(), [])
    random.shuffle(individual_rows)
    with open(os.path.join(args.save_dir, 'individual_relevance.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'premise', 'outline_item', 'passage'])
        writer.writeheader()
        writer.writerows(individual_rows)
        
