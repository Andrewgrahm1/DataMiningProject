# experiments

This folder is for **custom experiments** to find and explore trading strategies.

Add your experiment scripts here (e.g. backtests, strategy ideas, parameter sweeps). Code in `experiments/` is separate from the shared library in `lib/` and can be run ad hoc as you iterate on strategies. For machine-learning experiments, prefer calling training helpers from **`lib.models`** rather than duplicating fit/tune logic in the experiment script.
