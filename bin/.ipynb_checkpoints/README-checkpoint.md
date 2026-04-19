# bin

This is the **bin** directory of the project. It contains scripts that are needed to build, run, and maintain the algorithmic trading strategies project.

---

## First-time setup: `init.py`

After cloning the repository, run **`init.py`** to initialize the project:

- **Windows:**  
  `bin\init.bat`

- **macOS/Linux:**  
  `python3 bin/init.py`

This script:

1. Creates (or recreates) a Python virtual environment in the project root (`venv`)
2. Installs required dependencies (e.g. `pytz`, `alpaca-py`, `pandas`, `pandas_market_calendars`, `pylint`)

You only need to run it once after a fresh clone, or when you want to reset the environment.

---

## Activating the virtual environment

To activate the venv in your current terminal (so you can run `python`, `pip`, or scripts like `experiments\test.py`):

- **Windows — PowerShell (from project root):**  
  `. .\bin\activate.ps1`  
  Use the leading dot and space so the script runs in your current session. Your prompt will show `(venv)`.

- **macOS/Linux:**  
  `source venv/bin/activate`

---

## Control script: `ctl.py` and `ctl.bat`

The control script provides a command-line interface for project tasks: running linting and tests.

### Using `ctl.bat` (Windows)

From the project root, run:

```bat
bin\ctl.bat <command> [options]
```

`ctl.bat` invokes `ctl.py` using the project’s virtual environment (`venv`), so you don’t need to activate the venv first.

### Using `ctl.py` (any platform)

From the project root, with the project’s virtual environment **activated** (or with `venv`’s Python on your `PATH`):

```bash
python bin/ctl.py <command> [options]
```

On Windows (with venv activated):

```bat
python bin\ctl.py <command> [options]
```

---

### Commands

| Command   | Description |
|----------|-------------|
| **`verify`** | Run pylint on code in `lib/` (using `etc/pylintrc` if present), then run the full test suite. Pass extra arguments after `verify` to forward them to pylint (e.g. `ctl.bat verify --disable=C0114`). |
| **`test`**   | Run unit and/or integration tests. |

**`test` options:**

- `--skip-unit` / `-su` — skip unit tests
- `--skip-integration` / `-si` — skip integration tests

**Examples:**

```bat
bin\ctl.bat verify
bin\ctl.bat test
bin\ctl.bat test --skip-integration
bin\ctl.bat verify --disable=C0114
```

```bash
python bin/ctl.py verify
python bin/ctl.py test -si
```

**Note:** Both `verify` and `test` require the project virtual environment to exist. If you see “Virtual environment not found”, run `python bin/init.py` (or `python bin\init.py` on Windows) first.
