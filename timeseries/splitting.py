"""Reproducible train/validation/test partitions of untransformed samples."""
import hashlib
import math
import numpy as np
from torch.utils.data import Subset, ConcatDataset
from .datasets._subsample import stratified_indices_from_labels

DEFAULT_DATA_SPLIT = {'train': 80, 'validation': 10, 'test': 10}


def load_dataset_partitions(build_raw, *, root, max_samples, seed, data_split, **kwargs):
    """Cap and split raw samples before fitting preprocessing transforms."""
    percentages = validate_data_split(DEFAULT_DATA_SPLIT if data_split is None else data_split)
    if seed is None:
        raise ValueError('SEED must be an integer for reproducible dataset partitions')

    original_train, original_test, class_names, _ = build_raw(
        root=root, max_samples=max_samples, seed=seed, **kwargs)
    pool = ConcatDataset([original_train, original_test])
    if max_samples is not None and len(pool) > int(max_samples):
        selected = stratified_indices_from_labels(
            dataset_labels(pool), int(max_samples), seed=seed, min_per_class=1)
        pool = Subset(pool, selected)

    subsets, metadata = split_dataset(pool, percentages, seed)
    metadata['num_samples'] = len(pool)
    return subsets, class_names, metadata


def validate_data_split(split):
    if set(split) != set(DEFAULT_DATA_SPLIT):
        raise ValueError('DATA_SPLIT requires exactly train, validation, and test percentages')
    values = {key: float(split[key]) for key in DEFAULT_DATA_SPLIT}
    if any(not math.isfinite(v) or v <= 0 for v in values.values()) or not math.isclose(sum(values.values()), 100):
        raise ValueError('DATA_SPLIT percentages must be positive and sum to 100')
    return values


def dataset_labels(dataset):
    """Read stored labels where possible to avoid loading waveforms/events."""
    if isinstance(dataset, Subset):
        labels = dataset_labels(dataset.dataset)
        return [labels[i] for i in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        return [y for ds in dataset.datasets for y in dataset_labels(ds)]
    if hasattr(dataset, 'old_to_new') and hasattr(dataset, 'subset'):
        return [dataset.old_to_new[int(y)] for y in dataset_labels(dataset.subset)]
    if hasattr(dataset, 'segments'):
        labels = dataset_labels(dataset.base)
        return [labels[i] for i, *_ in dataset.segments]
    for attribute in ('targets', 'labels', 'y'):
        labels = getattr(dataset, attribute, None)
        if labels is not None and len(labels) == len(dataset):
            return [int(y) for y in labels]
    if hasattr(dataset, 'samples'):
        mapping = getattr(dataset, 'cid_to_y', None)
        return [mapping[y] if mapping is not None else int(y) for _, y, _ in dataset.samples]
    if hasattr(dataset, 'rows'):
        if 'classID' in dataset.rows:
            return dataset.rows['classID'].astype(int).tolist()
        return [dataset.class_to_idx[c] for c in dataset.rows['category']]
    if hasattr(dataset, 'items'):
        return [int(y) for _, y in dataset.items]
    if hasattr(dataset, '_file_labels'):
        return [dataset.word_to_idx['silence' if i == -1 else dataset._file_labels[i]] for i in dataset._indices]
    if hasattr(dataset, 'ds'):
        return dataset_labels(dataset.ds)
    return [int(dataset[i][1]) for i in range(len(dataset))]


def split_dataset(dataset, percentages, seed):
    """Stratify before transforms, assigning every item to exactly one set.

    Round class counts by largest remainder. Classes with at least three items
    contribute to all sets. Actual percentages can differ due to rounding.
    """
    percentages = validate_data_split(percentages)
    labels = np.asarray(dataset_labels(dataset))
    rng = np.random.default_rng(seed)
    indices = {key: [] for key in percentages}
    ratios = np.array(list(percentages.values())) / 100
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        rng.shuffle(members)
        exact = len(members) * ratios
        counts = np.floor(exact).astype(int)
        for i in np.argsort(-(exact - counts), kind='stable')[:len(members) - counts.sum()]:
            counts[i] += 1
        if len(members) >= 3:
            for i in np.flatnonzero(counts == 0):
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[i] += 1
        start = 0
        for key, count in zip(indices, counts):
            indices[key].extend(members[start:start + count].tolist())
            start += count
    if any(not ids for ids in indices.values()):
        raise ValueError('Too few samples for three nonempty sets; increase MAX_SAMPLES or revise DATA_SPLIT')
    indices = {key: sorted(ids) for key, ids in indices.items()}
    subsets = {key: Subset(dataset, ids) for key, ids in indices.items()}
    counts = {key: len(ids) for key, ids in indices.items()}
    metadata = dict(split_method='stratified_sample', split_seed=seed, data_split=percentages,
        split_counts=counts, split_percentages={key: 100 * n / len(dataset) for key, n in counts.items()},
        split_class_counts={key: {str(label): int(np.sum(labels[ids] == label)) for label in np.unique(labels)}
                            for key, ids in indices.items()},
        split_indices_sha256={key: hashlib.sha256(str(ids).encode()).hexdigest() for key, ids in indices.items()})
    return subsets, metadata


def split_report_fields(metadata):
    """Flat split columns shared by experiment and comparison exports."""
    fields = {}
    for name in DEFAULT_DATA_SPLIT:
        fields[f'{name}_samples'] = metadata.get('split_counts', {}).get(name)
        fields[f'{name}_percent_requested'] = metadata.get('data_split', {}).get(name)
        fields[f'{name}_percent_actual'] = metadata.get('split_percentages', {}).get(name)
    return fields
