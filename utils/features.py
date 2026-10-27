"""Only training catalogue defines categorical vocabularies; no review text."""
import html
import re
import numpy as np


def clean(value):
    if not isinstance(value,str):
        return '<UNK>'
    text = re.sub('<[^>]*>', ' ', html.unescape(value))
    text = re.sub(r'^\s*by\s+', '', text, flags=re.I)
    return ' '.join(text.lower().split()) or '<UNK>'


class ItemFeatures:
    def __init__(self, include_metadata=True):
        self.include_metadata = include_metadata

    def fit(self, train, metadata, n_users, n_items):
        self.n_users = n_users
        # Reserved feature IDs are disjoint for users, items, brands and platforms.
        fields = np.arange(n_items,dtype=np.int64)[:,None] + n_users
        self.vocabularies = {}
        next_id = n_users+n_items
        if self.include_metadata:
            meta = metadata.drop_duplicates('asin').set_index('asin')
            training_items = train[['asin','item_idx']].drop_duplicates().sort_values('item_idx')
            for field in ['brand','platform']:
                values = ['<UNK>'] * n_items
                for row in training_items.itertuples():
                    if row.asin not in meta.index:
                        continue
                    record = meta.loc[row.asin]
                    if field == 'brand':
                        values[row.item_idx] = clean(record.get('brand'))
                    else:
                        categories = record.get('category', [])
                        # Level 2 is usually a platform (PC, Xbox, etc.); never tail marketing text.
                        values[row.item_idx] = clean(categories[1]) if isinstance(categories,list) and len(categories)>1 else '<UNK>'
                vocab = {token:next_id+j for j,token in enumerate(sorted(set(values)|{'<UNK>'}))}
                next_id += len(vocab)
                self.vocabularies[field] = vocab
                fields = np.column_stack([fields,[vocab[v] for v in values]])
        self.item_fields = fields
        self.n_features = next_id
        return self

    def transform(self, users, items):
        return np.column_stack([np.asarray(users,dtype=np.int64),self.item_fields[np.asarray(items)]])
