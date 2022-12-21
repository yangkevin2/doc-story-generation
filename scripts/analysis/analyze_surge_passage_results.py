import csv
import argparse
from collections import defaultdict
from scipy.stats import ttest_rel
import numpy as np

from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

KEYS = ['coherence', 'relevance', 'interesting']

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Analyze results from a MTurk HIT.')
    parser.add_argument('-i', '--input', help='Input CSV file', nargs='*', required=True)
    parser.add_argument('--num-assignments', type=int, default=3)
    args = parser.parse_args()
    results = {}
    for in_file in args.input:
        with open(in_file, 'r') as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                id_info = row['id'].split('-')
                pair_num = id_info[0] + '-' + id_info[1]
                if pair_num not in results:
                    results[pair_num] = []
                storyA = id_info[2]
                storyB = id_info[3]
                possible_fields = [storyA, storyB]
                info = {storyA: {}, storyB: {}}
                info['premise'] = row['premise']
                result = row['Which passage seems more interesting?']
                if result.startswith('Both'):
                    info[storyA]['interesting'] = 1
                    info[storyB]['interesting'] = 1
                elif result.startswith('Passage A'):
                    info[storyA]['interesting'] = 1
                    info[storyB]['interesting'] = 0
                elif result.startswith('Passage B'):
                    info[storyA]['interesting'] = 0
                    info[storyB]['interesting'] = 1
                else:
                    assert result.startswith('Neither')
                    info[storyA]['interesting'] = 0
                    info[storyB]['interesting'] = 0

                result = row['Which passage has a more coherent overall plot?']
                if result.startswith('Both'):
                    info[storyA]['coherence'] = 1
                    info[storyB]['coherence'] = 1
                elif result.startswith('Passage A'):
                    info[storyA]['coherence'] = 1
                    info[storyB]['coherence'] = 0
                elif result.startswith('Passage B'):
                    info[storyA]['coherence'] = 0
                    info[storyB]['coherence'] = 1
                else:
                    assert result.startswith('Neither')
                    info[storyA]['coherence'] = 0
                    info[storyB]['coherence'] = 0

                result = row['Which passage is better focused on the given sub-event?']
                if result.startswith('Both'):
                    info[storyA]['relevance'] = 1
                    info[storyB]['relevance'] = 1
                elif result.startswith('Passage A'):
                    info[storyA]['relevance'] = 1
                    info[storyB]['relevance'] = 0
                elif result.startswith('Passage B'):
                    info[storyA]['relevance'] = 0
                    info[storyB]['relevance'] = 1
                else:
                    assert result.startswith('Neither')
                    info[storyA]['relevance'] = 0
                    info[storyB]['relevance'] = 0

                results[pair_num].append(info)

    total = len(results) * args.num_assignments
    field_counts = []
    fleiss_tables = defaultdict(lambda: [[], [], []])
    for field in possible_fields:
        counts = defaultdict(lambda: [])
        for r in results.values():
            assert len(r) == args.num_assignments
            for j in range(len(r)):
                info = r[j][field]
                for key in KEYS:
                    counts[key].append(info[key])
                    fleiss_tables[key][j].append(info[key])
        print(field)
        for key in KEYS:
            print(key + ':', "{:.3f}".format(sum(counts[key]) / total))
        print('\n\n')
        field_counts.append(counts)
    
    print('total', len(field_counts[0]['interesting']))
    print('\n')

    for key in field_counts[0]:
        print(key, "{:.2f}".format(ttest_rel(field_counts[0][key], field_counts[1][key]).pvalue))
    
    for key in field_counts[0]:
        print(key, 'fleiss kappa', fleiss_kappa(aggregate_raters(np.array(fleiss_tables[key]).transpose())[0]))

