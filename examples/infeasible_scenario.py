# Copyright 2024, Battelle Energy Alliance, LLC
"""This example demonstrates the use of the infeasibility diagnosis tool."""

import dove.core as dc

steam = dc.Resource("steam")
electricity = dc.Resource("electricity")

source = dc.Source(
    "steam_source",
    produces=steam,
    installed_capacity=1,
)
converter = dc.Converter(
    name="steam_turbine",
    consumes=[steam],
    produces=[electricity],
    installed_capacity=2,
    capacity_resource=steam,
    transfer_fn=dc.RatioTransfer(input_resources={steam: 1.0}, output_resources={electricity: 0.9}),
)
sink = dc.Sink(
    "elec_sink",
    consumes=electricity,
    installed_capacity=1.5,
    flexibility="fixed",
)

sys = dc.System(components=[source, converter, sink], resources=[steam, electricity])
results = sys.solve(diagnose_infeasible=True)
