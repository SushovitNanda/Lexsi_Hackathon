import copy
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/mac'))
from roi import calculate


def scenario():
    return json.loads((ROOT / 'configs/mac/planning/roi_scenario.example.json').read_text())


def test_cost_units_and_break_even_capacity():
    s = scenario()
    r = calculate(s)
    assert r['monthly_api_cost'] == 16
    assert r['monthly_energy_cost'] == pytest.approx(.24)
    assert r['monthly_local_cost'] == pytest.approx(600*.5/36 + .24 + 5)
    assert r['capacity_requests_per_month'] == 27000
    s['requests_per_month'] = r['break_even_requests_per_month']
    assert calculate(s)['monthly_savings'] == pytest.approx(0)
    s['requests_per_month'] = 27001
    assert not calculate(s)['demand_within_capacity']


def test_zero_price_or_zero_demand_does_not_invent_roi():
    s = scenario()
    s.update(api_input_per_million=0, api_output_per_million=0)
    r = calculate(s)
    assert r['break_even_requests_per_month'] is None
    assert r['savings_fraction_vs_api'] is None
    assert not r['break_even_within_capacity']


@pytest.mark.parametrize('field,value', [('utilization', 1.1), ('hardware_allocation', -1), ('amortization_months', 0), ('requests_per_second', float('nan'))])
def test_invalid_scenario_rejected(field, value):
    s = scenario(); s[field] = value
    with pytest.raises(ValueError):
        calculate(s)
