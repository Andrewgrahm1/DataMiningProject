"""
Strategy protocol: one interface for backtest, paper, and live.
"""

from datetime import datetime
from typing import Protocol

import pandas as pd

from lib.framework.orders import Order
from lib.framework.portfolio import Portfolio


class Strategy(Protocol):
    """
    Strategy interface: given current time, market snapshot, and portfolio,
    return a list of orders to submit.
    Implement this for backtest and (later) paper/live; same code path.
    """

    def next(
        self,
        current_time: datetime,
        market_snapshot: pd.DataFrame,
        portfolio: Portfolio,
    ) -> list[Order]:
        """
        Called each bar (or tick in live). Return orders to submit this step.

        Args:
            current_time: Current bar/time.
            market_snapshot: Bar(s) at this time (MultiIndex symbol/timestamp or single bar).
            portfolio: Current portfolio state (read-only; do not mutate).

        Returns:
            List of orders to submit; may be empty.
        """
