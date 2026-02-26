# DataMiningProject

Project for **data mining**: shared libraries and experiments for finding a **profitable trading strategy**, with tests and tooling.

---

## Quick start

1. **Clone** the repo.
2. **Add your Alpaca API key** so verify and data-fetching work. Create a file `etc/private/alpaca_key.json` with your keys in either format:
   - `{"APCA-API-KEY-ID": "your-key-id", "APCA-API-SECRET-KEY": "your-secret-key"}`
   - or `{"api_key": "your-key-id", "secret_key": "your-secret-key"}`
   Get keys from your [Alpaca dashboard](https://app.alpaca.markets/). The `etc/private/` folder is git-ignored so your keys are not committed. See [etc/private/README.md](etc/private/README.md).
3. **Initialize** the project (creates a virtual environment and installs dependencies):
   - Windows: `bin\init.bat`
   - macOS/Linux: `python3 bin/init.py`
4. **Lint and test:** from the project root, run `bin\ctl.bat verify` (Windows) or `python bin/ctl.py verify` (with venv activated).

See [bin/README.md](bin/README.md) for full details on `init.py`, `ctl.py`, and `ctl.bat`.

---

## Directory layout

| Directory      | Purpose |
|----------------|--------|
| **[bin/](bin/)**       | Scripts: setup (`init.py` / `init.bat`) and control (`ctl.py` / `ctl.bat`) for lint and tests. |
| **[etc/](etc/)**       | Data, configs, and private keys. Data and secrets are git-ignored. |
| **[experiments/](experiments/)** | Custom experiments to find and explore trading strategies. |
| **[lib/](lib/)**       | Shared libraries for data manipulation (stock data, utils) used by experiments. |
| **[tests/](tests/)**   | Unit and integration tests for code in `lib`. |

Each directory has its own README with more detail.
