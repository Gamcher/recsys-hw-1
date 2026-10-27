"""Amazon Reviews 2018: deterministic, train-only catalogue selection."""
import argparse
import gzip
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

BASE = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/'
SOURCES = {
    'Video_Games_5.json.gz': BASE + 'categoryFilesSmall/Video_Games_5.json.gz',
    'meta_Video_Games.json.gz': BASE + 'metaFiles2/meta_Video_Games.json.gz',
}


def download(root):
    """TLS verified download; validate entire gzip/JSON stream before rename."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in SOURCES.items():
        dest = root / name
        if not dest.exists():
            part = dest.with_suffix('.part')
            try:
                with urllib.request.urlopen(url, timeout=120) as response, part.open('wb') as out:
                    shutil.copyfileobj(response, out)
                for _ in records(part):
                    pass
                part.replace(dest)
            finally:
                part.unlink(missing_ok=True)
        digest = hashlib.sha256()
        with dest.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        manifest[name] = {'url': url, 'sha256': digest.hexdigest()}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def eda(reviews, metadata):
    times = pd.to_datetime(reviews.unixReviewTime, unit='s', utc=True, errors='coerce')
    pairs = reviews[['reviewerID', 'asin']].dropna().drop_duplicates()
    user_activity = pairs.groupby('reviewerID').size()
    item_popularity = pairs.groupby('asin').size()
    return {
        'unique_pairs': len(pairs),
        'sparsity_unique_pairs': 1 - len(pairs) / (len(user_activity) * len(item_popularity)) if len(pairs) else None,
        'user_activity_unique_items': user_activity.describe(percentiles=[.5, .9, .99]).to_dict(),
        'item_popularity_unique_users': item_popularity.describe(percentiles=[.5, .9, .99]).to_dict(),
        'reviews_shape': list(reviews.shape), 'metadata_shape': list(metadata.shape),
        'users': int(reviews.reviewerID.nunique()), 'items': int(reviews.asin.nunique()),
        'missing_reviews': reviews.isna().sum().to_dict(),
        'missing_metadata': metadata.isna().sum().to_dict(),
        'empty_metadata': {c: int(metadata[c].map(lambda x: x == '' or x == [] if isinstance(x, (str, list)) else False).sum()) for c in metadata},
        'exact_duplicate_reviews': int(reviews.astype(str).duplicated().sum()),
        'duplicate_user_item': int(reviews.duplicated(['reviewerID', 'asin']).sum()),
        'duplicate_metadata_asin': int(metadata.duplicated('asin').sum()),
        'rating_counts': reviews.overall.value_counts(dropna=False).sort_index().to_dict(),
        'time_min': str(times.min()), 'time_max': str(times.max()),
        'invalid_timestamps': int(times.isna().sum()),
        'reviews_per_year': times.dt.year.value_counts().sort_index().to_dict(),
        'metadata_columns': list(metadata.columns),
        'metadata_coverage': float(reviews.asin.isin(metadata.asin).mean()),
        'brands_top': metadata.brand.value_counts().head(20).to_dict() if 'brand' in metadata else {},
        'category_examples': metadata.category.head(5).tolist() if 'category' in metadata else [],
    }


def prepare(reviews, max_items=5000, min_interactions=100000, max_interactions=2000000,
            min_entities=1000):
    """Global chronological 80/10/10, ties kept together; freeze train vocabulary.

    Keep earliest user/item event, never consult future ratings for deduplication.
    Holdout cold-start rows are excluded and explicitly counted.
    """
    cols = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
    frame = reviews[cols].copy()
    frame = frame.dropna(subset=cols)
    frame = frame[frame.overall.between(1, 5) & (frame.unixReviewTime > 0)]
    frame = frame.sort_values(['unixReviewTime', 'reviewerID', 'asin', 'overall'], kind='stable')
    frame = frame.drop_duplicates(['reviewerID', 'asin'], keep='first')
    if frame.empty:
        raise ValueError('No valid interactions')
    # Cutoffs use timestamps only, never holdout popularity, text or ratings.
    t1, t2 = frame.unixReviewTime.quantile([.8, .9], interpolation='lower')
    raw = [frame[frame.unixReviewTime <= t1],
           frame[(frame.unixReviewTime > t1) & (frame.unixReviewTime <= t2)],
           frame[frame.unixReviewTime > t2]]
    counts = raw[0].groupby('asin').size().rename('count').reset_index()
    catalogue = counts.sort_values(['count', 'asin'], ascending=[False, True]).head(max_items).asin
    train = raw[0][raw[0].asin.isin(catalogue)].copy()
    # Fixed split budgets; retain most recent events with stable tie order.
    train = train.tail(int(max_interactions * .8)).copy()
    users, items = sorted(train.reviewerID.unique()), sorted(train.asin.unique())
    splits = [train] + [x[x.reviewerID.isin(users) & x.asin.isin(items)].tail(int(max_interactions * .1)).copy() for x in raw[1:]]
    if len(train) < min_interactions or min(len(users), len(items)) < min_entities:
        raise ValueError(f'Infeasible train constraints: interactions={len(train)}, users={len(users)}, items={len(items)}')
    if any(x.empty for x in splits):
        raise ValueError('Empty temporal split')
    user_map, item_map = dict(zip(users, range(len(users)))), dict(zip(items, range(len(items))))
    for part in splits:
        part['user_idx'] = part.reviewerID.map(user_map)
        part['item_idx'] = part.asin.map(item_map)
    assert splits[0].unixReviewTime.max() < splits[1].unixReviewTime.min()
    assert splits[1].unixReviewTime.max() < splits[2].unixReviewTime.min()
    report = {'cutoffs_unix': [int(t1), int(t2)], 'raw_split_rows': [len(x) for x in raw],
              'prepared_split_rows': [len(x) for x in splits],
              'excluded_rows': [len(x)-len(y) for x, y in zip(raw, splits)],
              'train_users': len(users), 'train_items': len(items),
              'ease_float64_matrix_bytes': len(items)**2 * 8,
              'constraint_scope': 'at least 100k in train by default; at most 2M across all splits',
              'user_mapping': user_map, 'item_mapping': item_map}
    return splits, report


def save_prepared(reviews, metadata, data_dir='data'):
    """Write audit and deterministic train-only preparation artifacts."""
    out = Path(data_dir) / 'processed'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'eda.json').write_text(json.dumps(eda(reviews, metadata), indent=2, default=str))
    splits, report = prepare(reviews)
    for name, split in zip(['train', 'validation', 'test'], splits):
        split.to_csv(out / f'{name}.csv.gz', index=False, compression={'method':'gzip','mtime':0})
    # Deliberately exclude prices, salesRank, also_buy/view, review aggregates and text.
    safe = [c for c in ['asin', 'brand', 'category', 'title'] if c in metadata]
    metadata = metadata[safe].drop_duplicates('asin', keep='first')
    metadata = metadata[metadata.asin.isin(splits[0].asin)]
    metadata.to_json(out / 'item_metadata.jsonl', orient='records', lines=True)
    (out / 'preparation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('mapping')}, indent=2))

    return splits, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    raw = args.data_dir / 'raw'
    if args.download:
        download(raw)
    reviews = pd.DataFrame(records(raw / 'Video_Games_5.json.gz'))
    metadata = pd.DataFrame(records(raw / 'meta_Video_Games.json.gz'))
    save_prepared(reviews, metadata, args.data_dir)


if __name__ == '__main__':
    main()
