# Trader

A quantitative trading system with multiple strategy experiments, featuring RAG-based risk control and hierarchical multi-asset portfolio management.

## Documentation

- [Changelog](CHANGELOG.md) - Project development history and updates
- [Report](docs/报告.pdf) - Detailed project report (in Chinese)

## Requirements

- Python 3.12 - 3.14
- Poetry (for dependency management)

## Project Structure

```
Trader/
├── cmd/                    # Command-line executables
├── trader/                 # Core trading system modules
│   ├── agent/              # Trading agents
│   ├── backtest/           # Backtesting engine
│   ├── dataloader/         # Data loading utilities
│   ├── features/           # Feature engineering
│   ├── news/               # News data processing
│   ├── predictor/          # Price prediction models
│   ├── rag/                # RAG-based risk control
│   └── risk/               # Risk management modules
├── experiments/            # Trading strategy experiments
├── data/                   # Data storage (CSV, SQLite)
├── output/                 # Experiment outputs
│   ├── backtest/           # Backtest reports and visualizations
│   ├── predictor/          # Model outputs
│   └── debug/              # Debug information
└── script/                 # Utility scripts
```

## Quick Start

### 1. Environment Setup

Install dependencies using Poetry:

```bash
poetry install
```

### 2. Configure Environment Variables

Create a `.env` file in the project root and configure your API keys:

```bash
# Copy environment variable template (if exists)
cp .env.example .env

# Edit .env file and add your DEEPSEEK API Key
# DEEPSEEK_API_KEY=your_api_key_here
```

### 3. Initialize Database

Initialize the SQLite database for storing market data:

```bash
"./script/0. init_db.sh"
```

### 4. Prepare News Data

Process and prepare news data for RAG-based analysis:

```bash
"./script/1. prepare_news.sh"
```

## Running Experiments

The project includes multiple trading strategy experiments organized in the `experiments/` directory. Each experiment is a self-contained directory with a `main.py` file.

### Interactive Experiment Runner

Use the interactive experiment runner to select and run experiments:

```bash
"./script/2. run experiment.sh"
```

This will display all available experiments in an interactive menu. You can:
- Select individual experiments to run
- Run entire experiment series (e.g., all 6.x series experiments)
- Exit the runner

### Available Experiments

Experiments are organized by series:

- **Series 1.x**: Single-asset strategies (DCA, Turtle, Logistic with various risk controls)
- **Series 2.x**: Multi-asset strategies with different risk management approaches
- **Series 3.x**: Agent-based strategies using predictors
- **Series 4.x**: Hierarchical multi-asset strategies with reflection mechanisms
- **Series 5.x**: Single stock (MRNA.O) experiments with different reflection strategies
- **Series 6.x**: Hierarchical multi-asset with RAG reflection (few-shot learning variants)

### Experiment Outputs

After running an experiment, results are saved to `output/backtest/` with the following structure:

- **Trading records**: Detailed transaction logs
- **Equity curves**: Account value over time
- **Performance metrics**: Returns, Sharpe ratio, max drawdown, etc.
- **Visualizations**: Charts and graphs (PNG format)
- **Reports**: Markdown-formatted analysis reports

## Additional Scripts

The `script/` directory contains utility scripts for various tasks:

- `0. init_db.sh`: Initialize the database schema
- `1. prepare_news.sh`: Process and prepare news data
- `2. run experiment.sh`: Interactive experiment runner
- `A. visualize features.sh`: Visualize feature distributions
- `B. train Predictor.sh`: Train prediction models
- `C. visualize Predictor.sh`: Visualize model predictions
- `D. top K news.sh`: Analyze top K news articles

## Development

The project uses Poetry for dependency management. All Python code should be run within the Poetry environment:

```bash
poetry run python <script>
```

Or activate the Poetry shell:

```bash
poetry shell
python <script>
```

