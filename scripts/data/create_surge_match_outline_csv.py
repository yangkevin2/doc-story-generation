from argparse import ArgumentParser
import os
import pickle
import csv
import random

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Input file', nargs='*', required=True)
    parser.add_argument('-o', '--output', help='Output file', required=True)
    parser.add_argument('--include-before-after', action='store_true')
    args = parser.parse_args()

    rows = []
    for fname in args.input:
        with open(fname, 'rb') as rf:
            data = pickle.load(rf)
            beam = data[0]
            leaves = list(beam.outline.leaves())
            for i, leaf in enumerate(leaves):
                generation = ''.join(beam.paragraphs_by_outline_section[leaf]).replace('\n', '<br>')
                if args.include_before_after:
                    before = ''.join(beam.paragraphs_by_outline_section[leaves[i-1]]).replace('\n', '<br>') if i > 0 else ''
                    after = ''.join(beam.paragraphs_by_outline_section[leaves[i+1]]).replace('\n', '<br>') if i < len(leaves)-1 else ''
                    generation = before + generation + after
                generation = generation.replace('<br><br>', '<br>')
                current_id = os.path.join(os.path.dirname(fname).split('/')[-1], os.path.basename(fname).split('.')[0]) + '___' + str(i)
                rows.append({'id': current_id, 'event': leaf.text.replace('This is the end of the story.', '').replace('\n', '<br>').strip(), 'passage': generation})
    random.shuffle(rows)
    with open(args.output, 'w') as wf:
        writer = csv.DictWriter(wf, fieldnames=['id', 'event', 'passage'])
        writer.writeheader()
        writer.writerows(rows)

