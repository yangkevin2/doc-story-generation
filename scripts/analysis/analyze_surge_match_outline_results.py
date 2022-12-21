import csv
import argparse
from collections import defaultdict
from scipy.stats import ttest_rel

QUESTION = 'Does the given event occur in its entirety within the passage?'

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Analyze results from a Passage Contains Event Surge Task.')
    parser.add_argument('-i', '--input', help='Input CSV file', nargs='*', required=True)
    args = parser.parse_args()
    result_dict = defaultdict(lambda:defaultdict(lambda:0))
    pairs = defaultdict(lambda:{})
    for in_file in args.input:
        with open(in_file, 'r') as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                outline_idx = row['id'].split('/')[-1]
                id = row['id'].split('___')[0]
                id = '_'.join(id.split('_')[:-1])
                result_dict[id]['total'] += 1
                if row[QUESTION] == 'Yes':
                    result_dict[id]['yes'] += 1
                    pairs[outline_idx][id] = 1
                elif row[QUESTION] == 'No':
                    result_dict[id]['no'] += 1
                    pairs[outline_idx][id] = 0
                else:
                    raise NotImplementedError
    for id in result_dict:
        print('\n', id)
        for key in ['yes', 'no']:
            print(key, result_dict[id][key], result_dict[id][key]/result_dict[id]['total'])
    keys = list(result_dict.keys())
    pairs1 = [pairs[outline_idx][keys[0]] for outline_idx in pairs]
    pairs2 = [pairs[outline_idx][keys[1]] for outline_idx in pairs]
    print(ttest_rel(pairs1, pairs2))
