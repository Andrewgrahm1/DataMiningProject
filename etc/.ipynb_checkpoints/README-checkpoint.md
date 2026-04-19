# etc

This is the **etc** directory of the project. It holds **data**, **configs**, and **private keys** used by the algorithmic trading strategies project.

---

## Contents

| Folder / file   | Purpose |
|-----------------|--------|
| **`data/`**     | Data files (e.g. CSVs, exports). Contents are git-ignored. |
| **`private/`** | Secrets (API keys, credentials). Contents are git-ignored; never commit real keys. |
| **`pylintrc`**  | Pylint configuration used by `bin/ctl.py verify`. |

---

## data

Put project data here (e.g. `trade_data.csv`, backtest outputs). Everything in `data/` is ignored by git so local or generated files are not committed. See `data/README.md` for details.

## private

Put private keys and credentials here (e.g. `alpaca_key.json`). Everything in `private/` is ignored by git. **Do not commit real secrets.** See `private/README.md` for details.
