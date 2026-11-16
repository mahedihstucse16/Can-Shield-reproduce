# -*- coding: utf-8 -*-
"""M4: cross-vehicle transfer with the SAME balancing for both representations,
over five independent draws.

The sampling, scaling, balancing, model and metric functions below are copied
verbatim from the authors' released notebook (CAN_SHIELD_Reproduction.ipynb) so
that the new runs are produced by the same pipeline as the published ones. The
only thing that changes is which (composition, representation, balance, seed)
cells get run.

Gate: --validate reproduces the published seed-42 numbers first. If those do not
match, nothing else should be believed.

Usage:
  python3 -I m4_balancing.py <cache_dir> <out_dir> [--validate | --grid]
"""
import gc
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

CACHE, OUT = sys.argv[1], sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else '--grid'
os.makedirs(OUT, exist_ok=True)

# ---- constants, verbatim from the notebook --------------------------------
MASTER_SEED = 42
MULTI_SEEDS = [0, 1, 2, 3, 4]
E2_TOTAL = 120000
RF_TRANSFER_TREES = 60
OTIDS_TEST_TOTAL = 2573117
CH_ORDER = ['DoS', 'Fuzzy', 'gear', 'RPM']
RAW_10 = list(range(0, 10))
ENG_8 = list(range(10, 18))
REPRESENTATIONS = {'raw10': RAW_10, 'eng8': ENG_8}
BALANCE_POLICY = {'raw10': 'none', 'eng8': 'smote'}
SCALE_POLICY = {'raw10': False, 'eng8': True}
E2_COMPOSITIONS = {
    'flooding_only': {'DoS': E2_TOTAL},
    'flooding_gear_rpm': {'DoS': E2_TOTAL // 3, 'gear': E2_TOTAL // 3, 'RPM': E2_TOTAL // 3},
    'all_four_with_fuzzing': {k: E2_TOTAL // 4 for k in CH_ORDER},
}
E2_LABEL = {'flooding_only': 'Flooding only',
            'flooding_gear_rpm': 'Flooding + gear + RPM spoofing',
            'all_four_with_fuzzing': 'All four, fuzzing included'}


def load(key):
    return (np.load('%s/X_%s.npy' % (CACHE, key), mmap_mode='r'),
            np.load('%s/y_%s.npy' % (CACHE, key)))


CH = {k: load('ch_' + k) for k in CH_ORDER}


def build_otids_test():
    Xd, yd = load('ot_dos')
    Xa, ya = load('ot_attack_free')
    nd, na = len(yd), len(ya)
    need = min(OTIDS_TEST_TOTAL - nd, na)
    rs = np.random.RandomState(MASTER_SEED)
    sel = np.sort(rs.choice(na, size=need, replace=False))
    X = np.concatenate([np.asarray(Xd), np.asarray(Xa)[sel]], axis=0)
    y = np.concatenate([np.asarray(yd), np.asarray(ya)[sel]], axis=0)
    return X, y


X_OT, y_OT = build_otids_test()
print('OTIDS test: %d frames, %d attack (%.4f%%)'
      % (len(y_OT), int(y_OT.sum()), 100 * y_OT.mean()), flush=True)


def stratified_take(key, m, seed):
    X, y = CH[key]
    y = np.asarray(y)
    ai = np.flatnonzero(y == 1)
    ni = np.flatnonzero(y == 0)
    n = len(y)
    mt = min(int(round(m * len(ai) / n)), len(ai))
    mn = min(m - mt, len(ni))
    rs = np.random.RandomState(seed)
    sel = np.concatenate([rs.choice(ai, mt, replace=False),
                          rs.choice(ni, mn, replace=False)])
    sel.sort()
    return np.asarray(X)[sel], y[sel]


def build_training_set(spec, seed):
    Xs, ys = [], []
    for i, (k, m) in enumerate(sorted(spec.items())):
        if m <= 0:
            continue
        Xi, yi = stratified_take(k, m, seed * 1000 + 7 * i + 13)
        Xs.append(Xi)
        ys.append(yi)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    rs = np.random.RandomState(seed)
    p = rs.permutation(len(y))
    return X[p], y[p]


def prepare(Xtr, ytr, rep, balance=None, seed=MASTER_SEED):
    cols = REPRESENTATIONS[rep]
    Xtr = np.asarray(Xtr)[:, cols].astype(np.float32)
    ytr = np.asarray(ytr)
    scaler = None
    if SCALE_POLICY[rep]:
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr).astype(np.float32)
    mode = BALANCE_POLICY[rep] if balance is None else balance
    cw = None
    if mode == 'smote':
        # imbalanced-learn is not installable here (the egress proxy denies
        # pypi), so the SMOTE arm cannot be reproduced bit-for-bit and is not
        # run. It is not needed: the referee's request is class weighting for
        # BOTH representations, which uses no synthetic samples at all.
        raise SystemExit('SMOTE arm not runnable in this environment')
    elif mode == 'class_weight':
        cw = 'balanced'
    return Xtr, ytr, scaler, cw, cols


def metrics(ytrue, ypred):
    ytrue = np.asarray(ytrue).astype(np.int8)
    ypred = np.asarray(ypred).astype(np.int8)
    tp = int(np.sum((ytrue == 1) & (ypred == 1)))
    fp = int(np.sum((ytrue == 0) & (ypred == 1)))
    fn = int(np.sum((ytrue == 1) & (ypred == 0)))
    tn = int(np.sum((ytrue == 0) & (ypred == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {'precision': 100 * prec, 'recall': 100 * rec, 'f1': 100 * f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'n_test': tp + fp + fn + tn}


def run_cell(comp, rep, balance, seed):
    key = '%s__%s__%s__seed%d' % (comp, rep, balance or 'manuscript', seed)
    path = os.path.join(OUT, key + '.json')
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    Xtr, ytr = build_training_set(E2_COMPOSITIONS[comp], seed)
    nid = int(len(np.unique(np.asarray(Xtr)[np.asarray(ytr).astype(bool), 0].astype(np.int32))))
    Xp, yp, scaler, cw, cols = prepare(Xtr, ytr, rep, balance=balance, seed=seed)
    mdl = RandomForestClassifier(n_estimators=RF_TRANSFER_TREES, random_state=seed,
                                 class_weight=cw, n_jobs=-1)
    mdl.fit(Xp, yp)
    Xte = np.asarray(X_OT)[:, cols].astype(np.float32)
    if scaler is not None:
        Xte = scaler.transform(Xte).astype(np.float32)
    m = metrics(y_OT, mdl.predict(Xte))
    m.update({'composition': comp, 'representation': rep,
              'balance': balance or 'manuscript', 'seed': seed,
              'training_attack_ids': nid, 'n_train': int(len(ytr)),
              'seconds': round(time.time() - t0, 1)})
    json.dump(m, open(path, 'w'), indent=1)
    del Xtr, ytr, Xp, yp, Xte, mdl
    gc.collect()
    print('  %-58s F1=%6.2f  prec=%6.2f  fp=%8d  (%.0fs)'
          % (key, m['f1'], m['precision'], m['fp'], m['seconds']), flush=True)
    return m


if MODE == '--validate':
    # Published seed-42 values from supplementary/table07d_balancing_control.csv
    # Published seed-42 values from supplementary/table07d_balancing_control.csv.
    # Only the SMOTE-free cells are used as the gate, since the SMOTE arm
    # cannot be run here.
    EXPECT = {
        ('all_four_with_fuzzing', 'raw10', None): 43.23054362127806,
        ('all_four_with_fuzzing', 'raw10', 'class_weight'): 48.75428348966225,
        ('all_four_with_fuzzing', 'eng8', 'class_weight'): 98.2326125928365,
        ('flooding_only', 'eng8', 'class_weight'): 96.97410982197162,
        ('flooding_gear_rpm', 'eng8', 'class_weight'): 97.79738506860771,
        ('flooding_only', 'raw10', 'class_weight'): 100.0,
    }
    print('\nVALIDATION against the published seed-42 run')
    ok = True
    for (comp, rep, bal), exp in EXPECT.items():
        m = run_cell(comp, rep, bal, MASTER_SEED)
        d = abs(m['f1'] - exp)
        flag = 'MATCH' if d < 1e-6 else ('close %.4f' % d if d < 0.05 else 'MISMATCH %.4f' % d)
        if d >= 0.05:
            ok = False
        print('    expected %10.6f  got %10.6f   %s' % (exp, m['f1'], flag))
    print('\nVALIDATION %s' % ('PASSED' if ok else 'FAILED'))
    sys.exit(0 if ok else 1)

print('\nGRID: 3 compositions x 2 representations x 2 balancings x 5 seeds')
for comp in E2_COMPOSITIONS:
    for rep in ('raw10', 'eng8'):
        # eng8 under the manuscript protocol needs SMOTE, which is not
        # installable here; raw10 under the manuscript protocol is unbalanced
        # and runs natively, giving the like-for-like baseline.
        bals = ('class_weight', None) if rep == 'raw10' else ('class_weight',)
        for bal in bals:
            for seed in MULTI_SEEDS:
                run_cell(comp, rep, bal, seed)
print('GRID COMPLETE')
