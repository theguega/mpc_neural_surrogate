# MCP Surrogate

Approximating Model Predictive Control (MPC) policies using neural networks.

## Description

This project implements surrogate models for MPC controllers, allowing for faster inference by approximating complex optimization-based control policies with neural networks. The approach combines reinforcement learning environments, optimization tools, and machine learning techniques to create efficient control policies.

## Installation

### Prerequisites
- Python >= 3.12
- uv (for dependency management)

### Install from source
```bash
git clone <repository-url>
cd mcp_surrogate
uv sync
```

### Development installation
```bash
uv sync --dev
```

## Usage

Run the main script:
```bash
mcp-surrogate
```

## Dependencies

### Core
- **casadi**: Optimization framework for MPC
- **gym**: Reinforcement learning environments
- **mujoco**: Physics simulation engine
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **h5py**: HDF5 file format support

### Development
- **ipython**: Interactive Python shell
- **pytest**: Testing framework
- **black**: Code formatting
- **ruff**: Linting and code quality
- **jupyter**: Interactive notebooks
- **wandb**: Experiment tracking
- **pre-commit**: Git hooks

## Development

### Code Quality
This project uses:
- **Black** for code formatting
- **Ruff** for linting
- Pre-commit hooks for automated quality checks

### Testing
```bash
pytest
```

### Linting
```bash
ruff check .
black --check .
```

## Author

Theo - [130441797+theguega@users.noreply.github.com](mailto:130441797+theguega@users.noreply.github.com)

## License

[Add license information here]
