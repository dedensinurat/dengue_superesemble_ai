"""Expanding-window cross-validation splitter (MODEL_SPEC C.9).

Generates walk-forward evaluation splits where:
- Training set only contains months <= origin.
- Test set contains target months origin + h (h=1..max_lead).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    """A single expanding window train/test split."""

    origin: str
    train_months: list[str]
    test_targets: list[tuple[str, int]]  # (target_month, lead)


class ExpandingWindowSplitter:
    """Generates train/test splits with expanding training window.

    Guarantees strict anti-leakage: training months never exceed origin.
    """

    def __init__(
        self,
        initial_train_months: int = 60,
        step: int = 1,
        max_lead: int = 6,
        retrain_every: int = 1,
    ) -> None:
        self.initial_train_months = initial_train_months
        self.step = step
        self.max_lead = max_lead
        self.retrain_every = retrain_every

    def split(self, months: list[str] | pd.Series) -> list[Split]:
        """Generate expanding window splits.

        Args:
            months: Ordered list or Series of month strings 'YYYY-MM'.

        Returns:
            List of Split objects.
        """
        sorted_months = sorted(list(set(months)))
        n_total = len(sorted_months)

        if n_total <= self.initial_train_months:
            return []

        splits: list[Split] = []
        # Walk forward from initial_train_months - 1 up to n_total - 2
        for i in range(self.initial_train_months - 1, n_total - 1, self.step):
            origin = sorted_months[i]
            train_months = sorted_months[: i + 1]

            test_targets: list[tuple[str, int]] = []
            for h in range(1, self.max_lead + 1):
                target_idx = i + h
                if target_idx < n_total:
                    test_targets.append((sorted_months[target_idx], h))

            if test_targets:
                splits.append(
                    Split(
                        origin=origin,
                        train_months=train_months,
                        test_targets=test_targets,
                    )
                )

        return splits
