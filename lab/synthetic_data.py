"""Generate fictional telemetry, audit it, and measure its downstream utility.

All partitions, including evaluation, are simulated. This demonstrates the
workflow; it does not demonstrate transfer to real production telemetry.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

FEATURES = ['queue_depth', 'error_percent']
VERSION = 'fictional-telemetry-v1'


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def simulate(count, seed, origin, coverage='full', label_noise=0.0):
    """Sample independent rows; flip labels only after applying the fixed oracle."""
    rng = np.random.default_rng(seed)
    values = rng.random((count, 2))
    if coverage == 'narrow':
        values[:, 1] *= .25
    # The noisy decision rule is an assumption of this simulator, not an observed law.
    clean = (values[:, 0] + .7 * values[:, 1] + rng.normal(0, .04, count) > .85).astype(int)
    flipped = rng.random(count) < label_noise
    labels = np.where(flipped, 1-clean, clean)
    rows = []
    for i, (q, e) in enumerate(values):
        rows.append({'id': f'{origin}-{i:05d}', 'origin': origin, 'generator': VERSION,
                     'queue_depth': round(float(q * 100), 6), 'error_percent': round(float(e * 10), 6),
                     'label': int(labels[i]), 'oracle_label': int(clean[i]), 'label_flipped': bool(flipped[i])})
    return rows


def dataset(seed, samples, coverage, label_noise):
    # Separate streams ensure changing the synthetic mix cannot change held-out rows.
    return {
        'seed': simulate(60, seed, 'seed', 'narrow'),
        'synthetic': simulate(samples, seed+1, 'synthetic', coverage, label_noise),
        'validation': simulate(400, seed+2, 'validation'),
        'test': simulate(400, seed+3, 'test'),
    }


def arrays(rows):
    # Provenance and the oracle are audit metadata: never training features.
    return np.array([[r[k] for k in FEATURES] for r in rows], dtype=float), np.array([r['label'] for r in rows])


def audit(data):
    seen_ids, seen_features = set(), set()
    summary = {}
    for name, rows in data.items():
        if not rows:
            raise ValueError('Each partition must contain rows.')
        fingerprints = set()
        for row in rows:
            if row['id'] in seen_ids:
                raise ValueError('Duplicate identifier across partitions.')
            seen_ids.add(row['id'])
            x = np.array([row[k] for k in FEATURES], dtype=float)
            if not np.isfinite(x).all() or not (0 <= x[0] <= 100 and 0 <= x[1] <= 10):
                raise ValueError('Telemetry is outside the declared feature contract.')
            if row['label'] not in (0, 1) or row['oracle_label'] not in (0, 1):
                raise ValueError('Labels must be zero or one.')
            fingerprints.add(tuple(x))
        if fingerprints & seen_features or len(fingerprints) != len(rows):
            raise ValueError('Exact feature duplicates found within or across partitions.')
        seen_features |= fingerprints
        summary[name] = {'rows': len(rows), 'positive_fraction': float(np.mean([r['label'] for r in rows])),
                         'flipped_fraction': float(np.mean([r['label_flipped'] for r in rows])),
                         'label_agreement_with_oracle': float(np.mean([r['label'] == r['oracle_label'] for r in rows])),
                         'high_error_fraction': float(np.mean([r['error_percent'] > 2.5 for r in rows])),
                         'origin': name, 'all_simulated': True}
    return {'partitions': summary, 'exact_duplicate_count': 0,
            'note': 'Exact deduplication cannot rule out near duplicates, shared templates, or distribution mismatch.'}


def train(rows):
    x, y = arrays(rows)
    if len(np.unique(y)) != 2:
        raise ValueError('Both labels are required. Increase sample count or choose another seed.')
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, random_state=0))
    return model.fit(x, y)


def measure(model, rows):
    x, y = arrays(rows)
    predicted = model.predict(x)
    result = {name: float(fn(y, predicted)) for name, fn in [('accuracy', accuracy_score), ('balanced_accuracy', balanced_accuracy_score)]}
    result.update({name: float(fn(y, predicted, zero_division=0)) for name, fn in [('precision', precision_score), ('recall', recall_score), ('f1', f1_score)]})
    result['confusion_matrix'] = confusion_matrix(y, predicted, labels=[0, 1]).tolist()
    high = x[:, 1] > 2.5
    result['high_error_accuracy'] = float(accuracy_score(y[high], predicted[high])) if high.any() else None
    return result


def predict(folder, inputs):
    x = np.asarray(inputs, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Use one or more [queue_depth, error_percent] rows.')
    if np.any((x[:, 0] < 0) | (x[:, 0] > 100) | (x[:, 1] < 0) | (x[:, 1] > 10)):
        raise ValueError('Queue depth must be 0–100; error percent must be 0–10.')
    models = joblib.load(Path(folder)/'model.joblib')
    return {name: {'class': model.predict(x).tolist(), 'incident_probability': model.predict_proba(x)[:, 1].tolist()}
            for name, model in models.items()}


def run(args):
    if not 20 <= args.samples <= 5000 or not 0 <= args.label_noise <= 1 or not 0 <= args.seed <= 2**32-4:
        raise ValueError('Use 20–5000 synthetic rows, label noise 0–1, and a nonnegative seed below 2**32-3.')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty output directory; existing evidence is never overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = dataset(args.seed, args.samples, args.coverage, args.label_noise)
    quality = audit(data)
    branches = {'seed_only': data['seed'], 'synthetic_only': data['synthetic'], 'mixed': data['seed']+data['synthetic']}
    models = {name: train(rows) for name, rows in branches.items()}
    evaluation = {name: measure(model, data[args.split]) for name, model in models.items()}
    for name, rows in data.items():
        with (output/(name+'.csv')).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    joblib.dump(models, output/'model.joblib')
    write(output/'model-manifest.json', {'format':'superlab-synthetic', 'feature_order':FEATURES,
          'ranges':[[0,100],[0,10]], 'label_meaning':{'0':'no simulated incident','1':'simulated incident'}, 'generator':VERSION})
    example = [[35.0, 1.0], [60.0, 8.0]]
    write(output/'example-input.json', example)
    reloaded = predict(output, example)
    reload_ok = all(np.array_equal(model.predict_proba(example)[:, 1], reloaded[name]['incident_probability']) for name, model in models.items())
    if not reload_ok:
        raise RuntimeError('Reload verification failed.')
    hashes = {name:hashlib.sha256((output/(name+'.csv')).read_bytes()).hexdigest() for name in data}
    write(output/'data-card.json', {'purpose':'Fictional telemetry teaching exercise; never use for operational decisions.',
          'all_partitions_simulated':True, 'generator':VERSION, 'seed':args.seed, 'settings':vars(args),
          'features':FEATURES, 'audit_only_fields':['id','origin','generator','oracle_label','label_flipped'],
          'rule':'queue_depth/100 + 0.7*error_percent/10 + normal_noise(0,0.04) > 0.85',
          'lineage':'Independent seed/augmentation/validation/test random streams; no real records or model downloads.',
          'evaluation_limit':'Held-out simulator draws test this declared world, not real-world transfer.',
          'dataset_sha256':hashes, 'quality':quality})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for axis, name in zip(ax, ['seed', 'synthetic']):
        x, y = arrays(data[name]);axis.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', s=15, alpha=.65, vmin=0, vmax=1)
        axis.set(xlabel='Queue depth', ylabel='Error percent', xlim=(0,100), ylim=(0,10), title=f'{name}: blue = 0; red = 1')
    fig.savefig(output/'coverage.png', dpi=130);plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.bar(list(evaluation), [v['balanced_accuracy'] for v in evaluation.values()], color=['#73856a','#b0c993','#274c3b'])
    ax.axhline(.5, color='#7d625b', linestyle='--', label='Constant-label reference')
    ax.set(ylim=(0,1), ylabel='Balanced accuracy', title=f'Independent simulated {args.split} rows');ax.legend()
    fig.savefig(output/'utility.png', dpi=130);plt.close(fig)
    report = {'schema':'superlab.report.v1', 'experiment':'synthetic-data', 'split':args.split, 'settings':vars(args),
              'metrics':{name+'_balanced_accuracy':v['balanced_accuracy'] for name,v in evaluation.items()},
              'evaluation':evaluation, 'quality':quality, 'training_rows':{k:len(v) for k,v in branches.items()},
              'synthetic_fraction_in_mixed':args.samples/(args.samples+60), 'reload_verified':reload_ok,
              'reloaded_prediction':reloaded, 'dataset_sha256':hashes,
              'artifacts':['coverage.png','utility.png','data-card.json','seed.csv','synthetic.csv','validation.csv','test.csv','model-manifest.json'],
              'seconds':time.perf_counter()-started,
              'limitations':['All examples are simulated; an independent real-world evaluation is still needed.',
                             'The oracle is known only in this teaching simulator; real generated labels need independent checks.',
                             'The same simulator generates training and evaluation; no claim of reality or formal privacy.',
                             'A single run does not establish a general benefit; compare several seeds without tuning on the final test.']}
    write(output/'report.json', report)
    print(json.dumps({'output':str(output), 'metrics':report['metrics'], 'reload_verified':reload_ok}, indent=2))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=int,default=400);p.add_argument('--label-noise',type=float,default=0.0)
    p.add_argument('--coverage',choices=['full','narrow'],default='full');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--output',required=True)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
