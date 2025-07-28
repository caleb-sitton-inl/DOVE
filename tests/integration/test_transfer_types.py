# Copyright 2024, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
""" """

import pandas as pd
import pytest

import dove.core as dc


@pytest.mark.integration()
def test_transfers():
    funding = dc.Resource("funding")
    labor = dc.Resource("labor")
    collaboration = dc.Resource("collaboration")
    work = dc.Resource("work")

    funding_source = dc.Source(
        name="FundingSource",
        produces=funding,
        max_capacity_profile=[200],
        flexibility="fixed",
    )

    labor_source = dc.Source(
        name="LaborSource",
        produces=labor,
        max_capacity_profile=[500],
        flexibility="fixed",
    )

    collaboration_source = dc.Source(
        name="CollaborationSource",
        produces=collaboration,
        max_capacity_profile=[100],
        flexibility="fixed",
    )

    balance_ratio_1 = dc.Converter(
        name="BalanceRatio1",
        consumes=[funding],
        produces=[work],
        capacity_resource=funding,
        max_capacity_profile=[100],
        transfer_fn=dc.RatioTransfer(input_resources={funding: 1.0}, output_resources={work: 0.25}),
    )

    balance_ratio_2 = dc.Converter(
        name="BalanceRatio2",
        consumes=[collaboration],
        produces=[funding, work],
        max_capacity_profile=[100],
        capacity_resource=collaboration,
        transfer_fn=dc.RatioTransfer(
            input_resources={collaboration: 1.0}, output_resources={funding: 0.2, work: 0.1}
        ),
    )

    quadratic = dc.Converter(
        name="Quadratic",
        consumes=[funding, labor],
        produces=[work],
        capacity_resource=funding,
        max_capacity_profile=[100],
        transfer_fn=dc.PolynomialTransfer(
            [
                (0.9, {funding: 1}),
                (1, {labor: 1}),
                (1e-6, {funding: 1, labor: 2}),
            ]
        ),
    )

    work_sink = dc.Sink(
        name="Milestones",
        consumes=work,
        max_capacity_profile=[6e3],
        cashflows=[dc.Revenue("proposals", alpha=1.0)],
    )

    funding_sink = dc.Sink(
        name="Outsource",
        consumes=funding,
        max_capacity_profile=[150],
        cashflows=[dc.Cost("contracts", alpha=1.0)],
    )

    labor_sink = dc.Sink(
        name="BusyWork",
        consumes=labor,
        max_capacity_profile=[500],
        cashflows=[dc.Cost("other_work", alpha=1.0)],
    )

    sys = dc.System(
        components=[
            funding_source,
            labor_source,
            collaboration_source,
            balance_ratio_1,
            balance_ratio_2,
            quadratic,
            work_sink,
            funding_sink,
            labor_sink,
        ],
        resources=[funding, labor, collaboration, work],
    )
    results = sys.solve(solver="ipopt")
    print(results)

    expected = pd.DataFrame(
        {
            "FundingSource_funding_produces": [199.99999800250433],
            "LaborSource_labor_produces": [500.0000049977196],
            "CollaborationSource_collaboration_produces": [99.99999900925043],
            "BalanceRatio1_work_produces": [25.000000249498818],
            "BalanceRatio1_funding_consumes": [-100.00000099799527],
            "BalanceRatio2_funding_produces": [19.999999801850084],
            "BalanceRatio2_work_produces": [9.999999900925042],
            "BalanceRatio2_collaboration_consumes": [-99.99999900925043],
            "Quadratic_work_produces": [615.0000066545259],
            "Quadratic_funding_consumes": [-100.00000099883447],
            "Quadratic_labor_consumes": [-500.000005005333],
            "Milestones_work_consumes": [-650.0000068049497],
            "Outsource_funding_consumes": [-19.999995807524677],
            "BusyWork_labor_consumes": [7.613425183555376e-09],
            "objective": [630.0000110050385],
        }
    )

    pd.testing.assert_frame_equal(results, expected, check_like=True, atol=1e-8)
