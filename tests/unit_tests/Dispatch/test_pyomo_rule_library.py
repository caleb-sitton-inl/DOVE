# Copyright 2024, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

# import __init__  # Running __init__ here enables importing from DOVE and RAVEN

import unittest
from unittest.mock import MagicMock, patch

import pyomo.environ as pyo
from pyomo.core.expr.relational_expr import InequalityExpression

import dove.dispatch.pyomo_rule_library as prl

class TestPyomoRuleLibrary(unittest.TestCase):

  def setUp(self):

    # For convenience, code needed for multiple tests is put here

    self.model = pyo.ConcreteModel() # For testing with pyomo objects
    self.mockModel = MagicMock(name="mockModel") # For testing math with dicts
    self.resources = pyo.Set(initialize=[0, 1])
    self.times = pyo.Set(initialize=[0, 1])
    self.large_eps = 1e8

    # Make sure all manually started patchers are stopped
    self.addCleanup(patch.stopall)

  # Testing with pyomo objects is helpful, but it doesn't check the math of the rules
  # Dicts are effective for testing the math, so for the below tests, we'll test with pyomo objects
  # and then with dicts to check both the typical behavior and the math

  def testChargeRule(self):

    chargeVals = {(0, 0): 0, (0, 1): 0, (1, 0): -2, (1, 1): -1}
    binVals = {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 1}

    # Test with pyomo objects

    # Set up inputs
    self.model.chargeVar = pyo.Var(self.resources, self.times, initialize=0, domain=pyo.NonPositiveReals)
    self.model.chargeVar.set_values(chargeVals)

    self.model.binVar = pyo.Var(self.resources, self.times, initialize=0, domain=pyo.Binary)
    self.model.binVar.set_values(binVals)

    # Check that none of the cases error out and all return a mathematical expression in terms of the inputs
    for r in self.resources:
      for t in self.times:
        charge_rule = prl.charge_rule("chargeVar", "binVar", self.large_eps, r, self.model, t)
        self.assertIsInstance(charge_rule, InequalityExpression)
        ruleString = charge_rule.to_string()
        self.assertIn(f"chargeVar[{r},{t}]", ruleString)
        self.assertIn(f"binVar[{r},{t}]", ruleString)

    # Test with dicts

    self.mockModel.chargeVar = chargeVals
    self.mockModel.binVar = binVals

    # Should return True when not charging or when (charging and binary=0)
    self.assertTrue(prl.charge_rule("chargeVar", "binVar", self.large_eps, 0, self.mockModel, 0))
    self.assertTrue(prl.charge_rule("chargeVar", "binVar", self.large_eps, 0, self.mockModel, 1))
    self.assertTrue(prl.charge_rule("chargeVar", "binVar", self.large_eps, 1, self.mockModel, 0))
    self.assertFalse(prl.charge_rule("chargeVar", "binVar", self.large_eps, 1, self.mockModel, 1))

  def testDischargeRule(self):

    dischargeVals = {(0, 0): 0, (0, 1): 0, (1, 0): 2, (1, 1): 1}
    binVals = {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 1}

    # Test with pyomo objects

    # Set up inputs
    self.model.dischargeVar = pyo.Var(self.resources, self.times, initialize=0, domain=pyo.NonNegativeReals)
    self.model.dischargeVar.set_values(dischargeVals)

    self.model.binVar = pyo.Var(self.resources, self.times, initialize=0, domain=pyo.Binary)
    self.model.binVar.set_values(binVals)

    # Check that none of the cases error out and all return a mathematical expression in terms of the inputs
    for r in self.resources:
      for t in self.times:
        discharge_rule = prl.discharge_rule("dischargeVar", "binVar", self.large_eps, r, self.model, t)
        self.assertIsInstance(discharge_rule, InequalityExpression)
        ruleString = discharge_rule.to_string()
        self.assertIn(f"dischargeVar[{r},{t}]", ruleString)
        self.assertIn(f"binVar[{r},{t}]", ruleString)

    # Test with dicts

    self.mockModel.dischargeVar = dischargeVals
    self.mockModel.binVar = binVals

    # Should return True when not discharging or when (discharging and binary=1)
    self.assertTrue(prl.discharge_rule("dischargeVar", "binVar", self.large_eps, 0, self.mockModel, 0))
    self.assertTrue(prl.discharge_rule("dischargeVar", "binVar", self.large_eps, 0, self.mockModel, 1))
    self.assertFalse(prl.discharge_rule("dischargeVar", "binVar", self.large_eps, 1, self.mockModel, 0))
    self.assertTrue(prl.discharge_rule("dischargeVar", "binVar", self.large_eps, 1, self.mockModel, 1))


if __name__ == "__main__":
  unittest.main()
