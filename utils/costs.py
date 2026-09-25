"""Aggregate algorithmic costs using the window lengths seen during training."""
from collections import Counter
from copy import deepcopy


def training_costs(learner, profile: Counter, *, original_samples, batch,
                   fp_bytes=4, alpha=1.0, beta=1.0):
    """Mean work per B original sequences; peak memory at the longest window.

    profile[T] counts processed instances (windows, or unwindowed sequences)
    during one data pass. Collecting it while training avoids an extra dataset
    scan and handles variable lengths, overlap, and padded short sequences.
    B is the configured comparison batch size, including for a partial final
    minibatch. Forward-Forward counts a complete layerwise training sweep.
    """
    if not profile or original_samples <= 0:
        raise ValueError("Cannot estimate costs for an empty training pass")
    parts = [learner.estimate_costs(
        batch=batch, time_steps=T, fp_bytes=fp_bytes, alpha=alpha, beta=beta,
        num_windows=count / original_samples,
    ) for T, count in sorted(profile.items())]
    result = deepcopy(max(parts, key=lambda x: x['memory']['total_bytes']))
    for category in ('compute', 'access'):
        components = Counter()
        for part in parts:
            components.update(part[category]['components'])
        result[category] = dict(components=dict(components), total_scalars=sum(components.values()))
    result['time_proxy']['value'] = (alpha * result['compute']['total_scalars']
                                     + beta * result['access']['total_scalars'])
    result['windows_per_sample'] = sum(profile.values()) / original_samples
    result['processed_timesteps_per_sample'] = sum(T * n for T, n in profile.items()) / original_samples
    result['profile_source'] = 'first_training_epoch'
    result['original_samples'] = original_samples
    result['instance_length_counts'] = dict(sorted(profile.items()))
    result['per_original_sample'] = {
        'compute_scalars': result['compute']['total_scalars'] / batch,
        'access_scalars': result['access']['total_scalars'] / batch,
        'time_proxy': result['time_proxy']['value'] / batch,
    }
    return result
