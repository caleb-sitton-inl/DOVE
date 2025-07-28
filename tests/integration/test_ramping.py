# Copyright 2024, Battelle Energy Alliance, LLC
"""
Test script for ramping constraints in DOVE.

This script creates a simple system with ramping-constrained components
and tests the implementation of both ramp rate and ramp frequency constraints.
"""

import os

import numpy as np
import pandas as pd
import pytest

from dove.core import (
    Converter,
    Cost,
    RatioTransfer,
    Resource,
    Sink,
    Source,
    System,
)


@pytest.mark.integration()
def test_ramping():
    # Time periods - 24 hours
    hours = list(range(11))

    # Create resources
    steam = Resource("steam")
    electricity = Resource("electricity")

    # Time Series Grid Demand

    npp = Source(
        name="npp",
        produces=steam,
        max_capacity_profile=np.full(len(hours), 700),
    )

    npp_bop = Converter(
        name="npp_bop",
        consumes=[steam],
        produces=[electricity],
        max_capacity_profile=np.full(len(hours), 200),
        capacity_resource=electricity,
        transfer_fn=RatioTransfer(
            input_resources={steam: 1.0}, output_resources={electricity: 0.333}
        ),
        ramp_limit=0.5,
        ramp_freq=2,
    )

    ngcc = Source(
        name="ngcc",
        produces=electricity,
        max_capacity_profile=np.full(len(hours), 400),
        cashflows=[Cost(name="ngcc_cost", alpha=0.3)],
    )

    # Create a load with varying demand
    # Demand profile with morning and evening peaks
    demand_profile = np.array([1e-5, 200, 200, 300, 100, 100, 1e-5, 200, 400, 200, 1e-5])

    grid = Sink(
        name="grid", consumes=electricity, max_capacity_profile=demand_profile, flexibility="fixed"
    )

    # Create and populate the system
    system = System(time_index=hours)
    system.add_resource(electricity)
    system.add_resource(steam)
    system.add_component(npp)
    system.add_component(npp_bop)
    system.add_component(ngcc)
    system.add_component(grid)

    # Solve the system
    results = system.solve(model_type="price_taker")
    print(results)

    gold_file_dirname = os.path.dirname(os.path.abspath(__file__))
    gold_path = os.path.join(gold_file_dirname, "test_ramping_gold.csv")
    expected = pd.read_csv(gold_path)

    pd.testing.assert_frame_equal(results, expected, check_like=True, atol=1e-8)
