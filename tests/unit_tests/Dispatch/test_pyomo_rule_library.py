# Copyright 2024, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

# import __init__  # Running __init__ here enables importing from DOVE and RAVEN

import unittest
from unittest.mock import MagicMock, patch

import pyomo.environ as pyo
from pyomo.core.expr.relational_expr import InequalityExpression, EqualityExpression
import numpy as np

import dove.dispatch.pyomo_rule_library as prl
from dove.components import Component
from dove.interactions import Storage

class TestPyomoRuleLibrary(unittest.TestCase):

  def setUp(self):

    # For convenience, code needed for multiple tests is put here

    self.model = pyo.ConcreteModel()
    self.resources = pyo.Set(initialize=[0, 1])
    self.times = pyo.Set(initialize=[0, 1])
    self.large_eps = 1e8
    self.mockComponent = MagicMock(name="mockComponent", spec=Component)

    # Make sure all manually started patchers are stopped
    self.addCleanup(patch.stopall)

  def testChargeRule(self):

    # Set up inputs
    chargeVals = {(0, 0): 0, (0, 1): 0, (1, 0): -2, (1, 1): -1}
    binVals =    {(0, 0): 0, (0, 1): 1, (1, 0):  0, (1, 1):  1}

    self.model.chargeVar = pyo.Var(self.resources, self.times, initialize=0, domain=pyo.NonPositiveReals)
    self.model.chargeVar.set_values(chargeVals)

    self.model.binVar = pyo.Var(self.resources, self.times, initialize=0, domain=pyo.Binary)
    self.model.binVar.set_values(binVals)

    # Call the method under test and check return values
    for r in self.resources:
      for t in self.times:
        charge_rule = prl.charge_rule("chargeVar", "binVar", self.large_eps, r, self.model, t)
        self.assertIsInstance(charge_rule, InequalityExpression)
        # Check return value (should return True when not charging or when (charging and binary=0))
        if r == 1 and t == 1: # At (1, 1), it's charging and binary=1
          self.assertFalse(pyo.value(charge_rule))
        else:
          self.assertTrue(pyo.value(charge_rule))

  def testDischargeRule(self):

    # Set up inputs
    dischargeVals = {(0, 0): 0, (0, 1): 0, (1, 0): 2, (1, 1): 1}
    binVals =       {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 1}

    self.model.dischargeVar = pyo.Var(self.resources, self.times, initialize=0, domain=pyo.NonNegativeReals)
    self.model.dischargeVar.set_values(dischargeVals)

    self.model.binVar = pyo.Var(self.resources, self.times, initialize=0, domain=pyo.Binary)
    self.model.binVar.set_values(binVals)

    # Call method under test and check return values
    for r in self.resources:
      for t in self.times:
        discharge_rule = prl.discharge_rule("dischargeVar", "binVar", self.large_eps, r, self.model, t)
        self.assertIsInstance(discharge_rule, InequalityExpression)
        # Check return value (should return True when not discharging or when (discharging and binary=1))
        if r == 1 and t == 0: # At (1, 0), it's discharging and binary=0
          self.assertFalse(pyo.value(discharge_rule))
        else:
          self.assertTrue(pyo.value(discharge_rule))

  def testLevelRule(self):

    # Add one more timestep
    self.times = pyo.Set(initialize=[0, 1, 2])
    self.model.Times = np.array([0, 2, 3])

    self.mockComponent.interaction.mock_add_spec(Storage)
    self.mockComponent.interaction.sqrt_rte = 0.5

    # First half (r=0) are correct level values; second half (r=1) are wrong
    # Note that "correct" means relative to the previous levelVal, NOT the previous correct level
    chargeVals =    {(0, 0): -2.0, (0, 1): -4.0, (0, 2): 0.0, (1, 0): 0.0, (1, 1): -8.0, (1, 2): 0.0}
    dischargeVals = {(0, 0):  0.0, (0, 1):  0.0, (0, 2): 2.0, (1, 0): 1.0, (1, 1):  1.0, (1, 2): 0.0}
    levelVals =     {(0, 0):  2.0, (0, 1):  6.0, (0, 2): 2.0, (1, 0): 2.0, (1, 1):  2.0, (1, 2): 0.0}
    correctLevels = {(0, 0):  2.0, (0, 1):  6.0, (0, 2): 2.0, (1, 0): 0.0, (1, 1):  6.0, (1, 2): 2.0}

    # Set up inputs
    self.model.levelVar = pyo.Var(self.resources, self.times, initialize=0)
    self.model.levelVar.set_values(levelVals)
    self.model.chargeVar = pyo.Var(self.resources, self.times, initialize=0, domain=pyo.NonPositiveReals)
    self.model.chargeVar.set_values(chargeVals)
    self.model.dischargeVar = pyo.Var(self.resources, self.times, initialize=0, domain=pyo.NonNegativeReals)
    self.model.dischargeVar.set_values(dischargeVals)

    # Call the function under test and check return values
    for r in self.resources:
      initial = 0 if (r == 0) else 4 # Another input
      for t in self.times:
        levelRule = prl.level_rule(self.mockComponent, "levelVar", "chargeVar", "dischargeVar", initial, r, self.model, t)
        self.assertIsInstance(levelRule, EqualityExpression)
        self.assertEqual(pyo.value(levelRule.arg(0)), levelVals[r, t]) # Check LHS value
        self.assertEqual(pyo.value(levelRule.arg(1)), correctLevels[r, t]) # Check RHS value
        # Check return value
        if r == 0:
          self.assertTrue(pyo.value(levelRule))
        else:
          self.assertFalse(pyo.value(levelRule))

  def testPeriodicLevelRule(self):

    self.model.T = pyo.Set(initialize=[0, 2])
    levelVals = {(0, 0):  0.0, (0, 1):  2.0, (1, 0): 2.0, (1, 1):  0.0}

    # Set up inputs
    self.model.levelVar = pyo.Var(self.resources, self.times, initialize=0)
    self.model.levelVar.set_values(levelVals)

    initial = {self.mockComponent: 2.0}

    # Call the function under test and check return values
    for r in self.resources:
      for t in self.times:
        periodicLevelRule = prl.periodic_level_rule(self.mockComponent, "levelVar", initial, r, self.model, t)
        self.assertIsInstance(periodicLevelRule, EqualityExpression)
        # Check return value
        if r == 0:
          self.assertTrue(pyo.value(periodicLevelRule))
        else:
          self.assertFalse(pyo.value(periodicLevelRule))

  def testCapacityRule(self):

    # Set up patcher for prod_limit_rule
    prodLimitRulePatcher = patch("dove.dispatch.pyomo_rule_library.prod_limit_rule")
    mockProdLimitRule = prodLimitRulePatcher.start()

    # Test with negative caps
    negCaps = [-4.0, 0.0]
    returned = prl.capacity_rule("comp_production", 0, negCaps, self.model, 0)
    mockProdLimitRule.assert_called_with("comp_production", 0, negCaps, "lower", 0, self.model)
    self.assertEqual(returned, mockProdLimitRule.return_value)

    # Test with positive caps
    posCaps = [0.0, 4.0]
    returned = prl.capacity_rule("comp_production", 1, posCaps, self.model, 1)
    mockProdLimitRule.assert_called_with("comp_production", 1, posCaps, "upper", 1, self.model)
    self.assertEqual(returned, mockProdLimitRule.return_value)

  def testProdLimitRule(self):

    # Add one more timestep
    self.times = pyo.Set(initialize=[0, 1, 2])

    # Set up inputs
    prodVals = {(0, 0):  1, (0, 1):  4, (0, 2): 4, (1, 0): 0, (1, 1):  4, (1, 2): 5}
    limits = [2, 4, 3] # Just use the same limits for both resources

    self.model.prodVar = pyo.Var(self.resources, self.times)
    self.model.prodVar.set_values(prodVals)

    # Test with normal inputs
    for r in self.resources:
      kind = "lower" if (r == 0) else "upper"
      for t in self.times:
        # Call function under test
        prodLimit = prl.prod_limit_rule("prodVar", r, limits, kind, t, self.model)
        self.assertIsInstance(prodLimit, InequalityExpression)
        # Check return value
        if (r == 0 and (t == 0)) or (r == 1 and (t == 2)):
          self.assertFalse(pyo.value(prodLimit))
        else:
          self.assertTrue(pyo.value(prodLimit))

    # Test with bad kind value
    kind = "middle"
    with self.assertRaises(TypeError):
      prl.prod_limit_rule("prodVar", 0, limits, kind, 0, self.model)


if __name__ == "__main__":
  unittest.main()
