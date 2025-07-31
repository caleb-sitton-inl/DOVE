# Copyright 2024, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
``dove.core.cashflow``
======================

Module containing cash flow classes for financial modeling in DOVE.

This module defines classes for representing financial cash flows in energy system models,
including both costs and revenues. These cash flows can vary over time and have
configurable scaling factors.

Classes
--------
    CashFlow: Abstract base class for all cash flows
    Cost: Represents expenses or negative cash flows
    Revenue: Represents income or positive cash flows
"""

from __future__ import annotations

import inspect
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.dove.core import Component, Resource, Storage

if TYPE_CHECKING:
    from collections.abc import Callable

TimeDependent: TypeAlias = list[float] | NDArray[np.float64]
Feature: TypeAlias = tuple[Component, Resource, str]


@dataclass
class CashFlow(ABC):
    """
    Abstract base class representing a financial cash flow.

    This class serves as the foundation for different types of cash flows in the system.
    Cash flows have associated pricing profiles that can vary over time. If a price profile
    is not provided, it will default to using alpha as the fixed recurring value for the
    length of the simulation.

    Parameters
    ----------
    name : str
        Identifier for the cash flow.
    price_profile : TimeDependent, optional
        Time-dependent pricing data, defaults to empty list.
    alpha : float, optional
        Scaling factor for the cash flow magnitude, defaults to 1.0.
    dprime : float, optional
        Adjustment factor for price calculations, defaults to 1.0.
    scalex : float, optional
        Horizontal scaling factor for time-dependent functions, defaults to 1.0.
    price_is_levelized : bool, optional
        Flag indicating if the price is levelized, defaults to False. (Not Implemented)
    sign : int, optional
        Direction of the cash flow (positive or negative), defaults to 0.
    """

    name: str
    price_profile: TimeDependent = field(default_factory=list)
    alpha: float = 1.0
    dprime: float = 1.0
    scalex: float = 1.0
    price_is_levelized: bool = False
    sign: int = 0

    def __post_init__(self) -> None:
        """
        Process the cashflow's price_profile after initialization.
        """
        # Convert price_profile
        self.price_profile = np.asarray(self.price_profile, float).ravel()

    def evaluate(self, t: int, dispatch: float) -> float:
        """
        Returns the cashflow's dollar value at the given timestep t, provided a dispatch quantity.
        Recall that a positive value indicates a revenue and a negative value indicates a cost.
        """
        value = self.sign * self.alpha * ((dispatch / self.dprime) ** self.scalex)
        if len(self.price_profile) > 0:
            if t > len(self.price_profile) - 1:
                available = (
                    "[0]" if len(self.price_profile) == 1 else f"[0, {len(self.price_profile) - 1}]"
                )
                raise IndexError(
                    f"{self.name}: timestep {t} is outside of range for provided price_profile "
                    f"data (available range is {available})"
                )

            value *= self.price_profile[t]
        return value


@dataclass
class Cost(CashFlow):
    """
    Represents a negative cash flow or cost.

    This class is a subclass of the `CashFlow` abstract base class that specifically
    represents costs or expenses. A `Cost` instance always has a negative sign,
    which is enforced by setting the `sign` class attribute to -1.

    Attributes
    ----------
    sign : int
        Fixed value of -1 to indicate that this cash flow represents a cost.

    Examples
    --------
    A recurring cost of $1,000.00 per time period:

    >>> cost = Cost(name="Recurring Cost", alpha=1000.0)

    Note that not specifying a `price_profile` will default to using `alpha` as
    the fixed value for the length of the simulation.

    A time-dependent cost with a specific price profile:

    >>> cost = Cost(
    ...     name="Time-Dependent Cost",
    ...     price_profile=[0.5, 1.0, 1.5],
    ...     alpha=1000.0)
    """

    sign: int = -1


@dataclass
class Revenue(CashFlow):
    """
    A class representing revenue in a financial context.

    Revenue is a subclass of CashFlow with a positive sign (+1), indicating
    incoming cash flow. It represents income generated from the sale of goods,
    services, or other business activities.

    Attributes
    ----------
    sign : int
        The sign of the cash flow, set to +1 for revenue (incoming cash).
    """

    sign: int = +1


@dataclass
class CustomCashFlow:
    """
    A class that allows the user to calculate their own cash flow based on a feature of the system.

    Attributes
    ----------
    name: str
        Identifier for the cash flow
    features: list[Feature]
        A list of the system features that are needed to evaluate this cash flow.
    evaluation_function: Callable[[int, list[float]], float]
        A user-defined function that calculates the value of the cash flow at the provided timestep
        based on a list of values for input features. These input features will be found on the
        model using information from the features arg, then their values will be provided as
        arguments to the evaluation_function, in the same order as the features were listed.
    """

    name: str
    features: list[Feature]
    evaluation_function: Callable[[int, list[Any]], float]

    def __post_init__(self) -> None:
        # Validate features
        for feat in self.features:
            comp, res, feat_type = feat  # Unpack tuple
            if isinstance(comp, Storage):
                if comp.resource != res:
                    raise ValueError(
                        f"{self.name}: The resource provided for feature {feat} does not "
                        f"match the resource of the component ({res} != {comp.resource})."
                    )
                if feat_type not in ["SOC", "charge", "discharge"]:
                    raise ValueError(
                        f"{self.name}: The feature type '{feat_type}' provided "
                        f"for feature {feat} is not accepted with a storage "
                        "component. Please use 'SOC', 'charge', or 'discharge'."
                    )
            else:
                if feat_type not in ["produces", "consumes"]:
                    raise ValueError(
                        f"{self.name}: The feature type '{feat_type}' provided "
                        f"for feature {feat} is not accepted with a non-storage "
                        "component. Please use 'produces' or 'consumes'."
                    )
                if res not in getattr(comp, feat_type):
                    raise ValueError(
                        f"{self.name}: The resource '{res}' provided for "
                        f"feature {feat} is not found in the '{feat_type}' "
                        f"'attribute of the provided component '{comp.name}'"
                    )

        # Confirm that features correspond to evaluation_function args
        eval_func_signature = inspect.signature(self.evaluation_function)
        if len(eval_func_signature.parameters) != len(self.features) + 1:
            raise ValueError(
                f"{self.name}: The list of features to fetch from the model does not have a "
                "length equal to the number of features accepted by the evaluation_function "
                f"({len(self.features)} != {len(eval_func_signature.parameters) - 1})."
            )
