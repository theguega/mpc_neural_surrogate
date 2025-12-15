# MPC Surrogate

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A project for approximating Model Predictive Control (MPC) policies using neural networks for a 3-degree-of-freedom (3DOF) robot arm simulation in MuJoCo.

## Overview

This repository implements a surrogate modeling approach to replace computationally expensive MPC controllers with fast neural network approximations. The project focuses on trajectory optimization and control for robotic manipulation tasks, enabling real-time performance while maintaining control accuracy.

## Features

- **MuJoCo Simulation**: Realistic physics simulation of a 3DOF robot arm
- **MPC Implementation**: Full MPC controller using CasADi for optimal control
- **Neural Network Surrogates**: Trainable NN models to approximate MPC policies
- **Data Generation**: Scripts for collecting MPC trajectory data
- **Evaluation Tools**: Closed-loop testing and performance comparison
- **Visualization**: Comprehensive plotting and analysis tools

## Installation

### Prerequisites

- Python 3.12 or higher
- uv package manager (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mpc-surrogate.git
   cd mpc-surrogate
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. (Optional) Install development dependencies:
   ```bash
   uv sync --group dev
   ```

## Usage

### Data Generation

Generate MPC trajectory data for training:

```bash
python scripts/data_generator.py
```

### Training

Train neural network surrogates using the provided Jupyter notebook:

```bash
jupyter notebook scripts/mpc_surrogate_training.ipynb
```

### Interactive Simulation

Launch the MuJoCo interactive viewer:

```bash
python scripts/interactive_mujoco_launcher.py
```

### Closed-Loop Testing

Evaluate controller performance:

```bash
python scripts/closed_loop_eval.py
```

### Dataset Analysis

Analyze closed-loop evaluation results and generate comparison plots:

```bash
python scripts/analyze_closed_loop.py
```

### Visualization

Visualize scikit-learn baseline results:

```bash
python scripts/visualize_scikit_results.py
```

### Dataset Scaling

Test model performance with different dataset sizes:

```bash
python scripts/dataset_scale.py
```

### Dataset Replay

Replay dataset trajectories in MuJoCo for visualization:

```bash
python scripts/replay_dataset_mujoco.py
```

## Project Structure

```
├── data/                    # Generated datasets
├── docs/                    # Research report, figures, and documentation
├── models/                  # MuJoCo model files
├── results/                 # Evaluation results and plots
├── scripts/                 # Utility scripts and notebooks
│   ├── analyze_closed_loop.py       # Closed-loop performance analysis
│   ├── closed_loop_eval.py          # Closed-loop evaluation
│   ├── data_generator.py            # MPC data generation
│   ├── dataset_scale.py             # Dataset scaling experiments
│   ├── interactive_mujoco_launcher.py # Interactive MuJoCo viewer
│   ├── mpc_surrogate_training.ipynb # Main training notebook
│   ├── replay_dataset_mujoco.py     # Dataset visualization
│   ├── scikit_learn_baseline.py     # Baseline model training
│   └── visualize_scikit_results.py  # Results visualization
├── src/mpc_surrogate/       # Core package
│   ├── mpc_controller.py    # MPC implementation
│   ├── mujoco_env.py        # MuJoCo environment wrapper
│   └── utils.py             # Helper functions
└── tests/                   # Unit tests
```

## Development

### Testing

Run the test suite:

```bash
pytest
```

### Code Quality

Format and lint code:

```bash
ruff check . --fix
ruff format .
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Theo - [GitHub](https://github.com/theguega)
Dexter - [GitHub](https://github.com/dexterteo4)

## Citation

If you use this code in your research, please cite the relevant papers from the `docs/refs.bib` file or refer to this repository:

```bibtex
@misc{mpc-surrogate,
  title={Approximating Model Predictive Control policies using neural networks},
  author={Theo Guegan, Wen Jie Dexter Teo},
  year={2025},
  url={https://github.com/your-username/mpc-surrogate}
}
```
