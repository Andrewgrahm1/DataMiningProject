"""
Portfolio state: positions, cash, trade history.
Apply fills to update state; compute equity from current prices.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from lib.framework.orders import Fill, OrderSide


@dataclass
class Position:
    """A single position: symbol, quantity, and cost basis (total cost so far)."""

    symbol: str
    quantity: int
    cost_basis: float  # total cash spent (positive) or received (negative) for this position

    @property
    def avg_price(self) -> float:
        """Average price per share; 0 if no quantity."""
        if self.quantity == 0:
            return 0.0
        return self.cost_basis / self.quantity


@dataclass
class Portfolio:
    """
    Portfolio state: positions by symbol, cash, and history of fills (trades).
    Use apply_fill to update from broker fills; use equity() with current prices for valuation.
    """

    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    trade_history: List[Fill] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cash < 0:
            raise ValueError("cash cannot be negative")

    def position(self, symbol: str) -> Position:
        """Return position for symbol; zero position if not held."""
        return self.positions.get(symbol, Position(symbol=symbol, quantity=0, cost_basis=0.0))

    def apply_fill(self, fill: Fill) -> None:
        """
        Update positions and cash for a fill; deduct fill.fee from cash; append fill to trade_history.
        Sell quantity cannot exceed current position (no short selling).
        """
        self.trade_history.append(fill)
        pos = self.positions.get(fill.symbol, Position(symbol=fill.symbol, quantity=0, cost_basis=0.0))
        cost_delta = fill.price * fill.qty
        if fill.side == OrderSide.SELL:
            if pos.quantity < fill.qty:
                raise ValueError(
                    f"Cannot sell {fill.qty} {fill.symbol}: position is {pos.quantity}"
                )
            cost_delta = -cost_delta
            new_qty = pos.quantity - fill.qty
            new_basis = pos.cost_basis - (pos.avg_price * fill.qty) if pos.quantity else 0.0
        else:
            new_qty = pos.quantity + fill.qty
            new_basis = pos.cost_basis + cost_delta
        self.cash -= cost_delta  # buy: cash decreases; sell: cash increases
        self.cash -= fill.fee
        if new_qty == 0:
            self.positions.pop(fill.symbol, None)
        else:
            self.positions[fill.symbol] = Position(symbol=fill.symbol, quantity=new_qty, cost_basis=new_basis)

    def equity(self, snapshot: pd.DataFrame | None = None) -> float:
        """
        Total equity: cash + sum(position value at given prices).
        If snapshot is provided (DataFrame with MultiIndex symbol/timestamp and 'close' column),
        uses close price per symbol for valuation. If snapshot is None or empty, uses cost basis (book value).
        """
        value = self.cash
        if snapshot is not None and not snapshot.empty and "close" in snapshot.columns:
            if isinstance(snapshot.index, pd.MultiIndex) and "symbol" in snapshot.index.names:
                for sym, pos in self.positions.items():
                    try:
                        p = float(snapshot.xs(sym, level="symbol")["close"].iloc[-1])
                    except KeyError:
                        p = pos.avg_price
                    value += pos.quantity * p
            else:
                p = float(snapshot["close"].iloc[-1])
                for pos in self.positions.values():
                    value += pos.quantity * p
        else:
            for pos in self.positions.values():
                value += pos.quantity * pos.avg_price
        return value
