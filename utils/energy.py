# utils/energy.py
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class EnergyCoefficients:
    """
    Inference-only proxy coefficients.

    Literature-grounded 45nm-inspired values:
      - dense_mac_pj      = 3.7 pJ
      - synop_pj          = 0.9 pJ
      - neuron_update_pj  = 0.1 pJ
      - memory_access_pj  = 20.0 pJ
    """
    dense_mac_pj: float = 3.7
    synop_pj: float = 0.9
    neuron_update_pj: float = 0.1
    memory_access_pj: float = 20.0


ENERGY_COEFFS = EnergyCoefficients()


def compute_energy_breakdown(
    activity: Dict[str, Any],
    coeffs: EnergyCoefficients = ENERGY_COEFFS,
) -> Dict[str, Any]:
    """
    Returns average inference proxy energy PER SAMPLE.
    """
    n = max(1, int(activity["num_samples"]))

    avg_synops = float(activity["synaptic_operations"]) / n
    avg_neuron_updates = float(activity["neuron_updates"]) / n
    avg_input_macs = float(activity["input_mac_ops"]) / n

    energy_input_layer_pj = avg_input_macs * coeffs.dense_mac_pj
    energy_synop_pj = avg_synops * coeffs.synop_pj
    energy_neuron_update_pj = avg_neuron_updates * coeffs.neuron_update_pj

    avg_memory_accesses = (
        avg_input_macs
        + avg_synops
        + 2.0 * avg_neuron_updates
    )
    energy_memory_pj = avg_memory_accesses * coeffs.memory_access_pj

    energy_total_pj = (
        energy_input_layer_pj
        + energy_synop_pj
        + energy_neuron_update_pj
        + energy_memory_pj
    )

    total_safe = max(energy_total_pj, 1e-30)
    breakdown_pct = {
        "input_layer":   100.0 * energy_input_layer_pj / total_safe,
        "synop":         100.0 * energy_synop_pj / total_safe,
        "neuron_update": 100.0 * energy_neuron_update_pj / total_safe,
        "memory":        100.0 * energy_memory_pj / total_safe,
    }

    return {
        # Detailed breakdowns
        # "energy_input_layer_pj": energy_input_layer_pj,
        # "energy_synop_pj": energy_synop_pj,
        # "energy_neuron_update_pj": energy_neuron_update_pj,
        # "energy_memory_pj": energy_memory_pj,
        # "memory_accesses_per_sample": avg_memory_accesses,
        # "energy_total_pj": energy_total_pj,
        # "energy_total_nj": energy_total_pj / 1e3,
        # "energy_total_uj": energy_total_pj / 1e6,
        # "energy_breakdown_pct": breakdown_pct,
        # "energy_coeffs": {
        #     "dense_mac_pj": coeffs.dense_mac_pj,
        #     "synop_pj": coeffs.synop_pj,
        #     "neuron_update_pj": coeffs.neuron_update_pj,
        #     "memory_access_pj": coeffs.memory_access_pj,
        # },

        # Simplified summary metrics
        "energy_total_uj": energy_total_pj / 1e6,
        "energy_breakdown_pct": breakdown_pct,
    }
