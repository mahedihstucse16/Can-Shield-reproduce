# -*- coding: utf-8 -*-
"""Per-draw table for the constant-size composition ablation (seeds 0-4).
Usage: python3 -I make_per_draw_csv.py <checkpoint_dir> <out.csv>"""
import csv, glob, json, os, statistics, sys
SRC, OUT = sys.argv[1], sys.argv[2]
COMP = {'flooding_only': 'Flooding only',
        'flooding_gear_rpm': 'Flooding + gear + RPM spoofing',
        'all_four_with_fuzzing': 'All four, fuzzing included'}
REP = {'raw10': 'Raw 10', 'eng8': 'Engineered 8'}
BAL = {'class_weight': 'class weighted', 'manuscript': 'unbalanced'}
rows = []
for p in glob.glob(os.path.join(SRC, '*.json')):
    d = json.load(open(p))
    if d['seed'] not in (0, 1, 2, 3, 4):
        continue
    rows.append(d)
order = lambda d: (list(BAL).index(d['balance']), list(COMP).index(d['composition']),
                   list(REP).index(d['representation']), d['seed'])
rows.sort(key=order)
with open(OUT, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['composition', 'representation', 'balancing', 'seed',
                'attack_ids', 'f1', 'precision', 'recall', 'fp'])
    for d in rows:
        w.writerow([COMP[d['composition']], REP[d['representation']], BAL[d['balance']],
                    d['seed'], d['training_attack_ids'], '%.2f' % d['f1'],
                    '%.2f' % d['precision'], '%.2f' % d['recall'], d['fp']])
print(len(rows), 'rows')
g = {}
for d in rows:
    g.setdefault((d['composition'], d['representation'], d['balance']), []).append(d)
for k, v in g.items():
    f = [x['f1'] for x in v]; pr = [x['precision'] for x in v]
    print(k, len(v), '%.2f %.2f | %.2f %.2f' % (statistics.mean(f), statistics.stdev(f),
          statistics.mean(pr), statistics.stdev(pr)), [x['training_attack_ids'] for x in v])
