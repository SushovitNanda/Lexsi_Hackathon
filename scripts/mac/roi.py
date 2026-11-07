"""Transparent clinic-volume cost scenarios; no bundled market prices or clinical claims."""
import argparse
import json
import math
from pathlib import Path


def calculate(s):
    names = ['requests_per_month', 'input_tokens_per_request', 'output_tokens_per_request',
             'api_input_per_million', 'api_output_per_million', 'hardware_cost',
             'amortization_months', 'hardware_allocation', 'available_hours_per_month',
             'utilization', 'requests_per_second', 'wall_power_watts', 'electricity_per_kwh',
             'other_local_monthly_cost']
    for name in names:
        v = s[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f'{name} must be a finite nonnegative number')
    if s['amortization_months'] == 0 or not 0 <= s['hardware_allocation'] <= 1 or not 0 <= s['utilization'] <= 1:
        raise ValueError('Invalid amortization, allocation or utilization')
    if not s.get('currency') or not s.get('assumptions_source'):
        raise ValueError('Currency and assumptions provenance are required')
    n = s['requests_per_month']
    api_each = (s['input_tokens_per_request'] * s['api_input_per_million'] +
                s['output_tokens_per_request'] * s['api_output_per_million']) / 1_000_000
    hardware = s['hardware_cost'] * s['hardware_allocation'] / s['amortization_months']
    energy = s['wall_power_watts'] / 1000 * s['available_hours_per_month'] * s['electricity_per_kwh']
    local = hardware + energy + s['other_local_monthly_cost']
    api = n * api_each
    capacity = 3600 * s['available_hours_per_month'] * s['utilization'] * s['requests_per_second']
    break_even = local / api_each if api_each > 0 else None
    return {'currency': s['currency'], 'illustrative_only': s.get('illustrative_only', True),
        'assumptions_source': s['assumptions_source'], 'monthly_api_cost': api,
        'monthly_local_cost': local, 'monthly_hardware_cost': hardware, 'monthly_energy_cost': energy,
        'monthly_savings': api - local, 'savings_fraction_vs_api': (api-local)/api if api else None,
        'cost_return_fraction_on_local_cost': (api-local)/local if local else None,
        'capacity_requests_per_month': capacity, 'demand_within_capacity': n <= capacity,
        'break_even_requests_per_month': break_even,
        'break_even_within_capacity': break_even <= capacity if break_even is not None else False,
        'interpretation': 'Cost scenario only; capacity, safety, clinical utility and latency require separate validation. '
        'Fixed local monthly budget assumes the stated powered hours; it is not an autoscaling or queueing model.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    scenario = json.loads(a.scenario.read_text())
    result = {'scenario': scenario, 'result': calculate(scenario)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['result'], indent=2))


if __name__ == '__main__':
    main()
