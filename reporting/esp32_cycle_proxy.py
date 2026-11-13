"""Export the ESP32 reference cycle proxy from saved report values."""
from pathlib import Path
import pandas as pd


def esp32_cycle_proxy_data(window_reference, fast_capacity=512*1024,
                           slow_capacity=8*1024**2):
    rows=[]
    for r in window_reference.itertuples():
        for setting,prefix in [('full_sequence','baseline'),('windowed','selected')]:
            memory=getattr(r,prefix+'_memory_bytes')
            rows.append(dict(dataset=r.dataset,learner=r.learner,setting=setting,
                run_id=getattr(r,prefix+'_run_id'),batch_size=r.batch_size,
                compute_scalars=getattr(r,prefix+'_compute_scalars'),
                access_scalars=getattr(r,prefix+'_access_scalars'),
                memory_bytes=memory,fast_capacity_bytes=fast_capacity,
                total_capacity_bytes=fast_capacity+slow_capacity,
                fast_fraction=getattr(r,prefix+'_fast_fraction'),
                effective_beta=getattr(r,prefix+'_effective_beta'),
                cycle_proxy=getattr(r,prefix+'_estimated_cost_units'),
                cycle_proxy_millions=getattr(r,prefix+'_estimated_cost_units')/1e6,
                fits_fast_memory=memory<=fast_capacity,
                fits_total_memory=memory<=fast_capacity+slow_capacity))
    return pd.DataFrame(rows)


if __name__=='__main__':
    folder=Path(__file__).resolve().parents[1]/'results/results_overview'
    data=esp32_cycle_proxy_data(pd.read_csv(folder/'windowed_cost_reference_costs.csv'))
    data.to_csv(folder/'esp32_cycle_proxy.csv',index=False)
    print(f'Exported {len(data)} cycle proxies and nominal memory-fit flags.')
