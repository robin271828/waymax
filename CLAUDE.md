# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Waymax is a JAX-based, hardware-accelerated simulator for autonomous driving research built on the Waymo Open Motion Dataset (WOMD). It supports both closed-loop simulation for planning/control research and open-loop behavior prediction, with all components written in JAX for GPU/TPU acceleration.

## Installation & Setup

```bash
# Install Waymax
pip install --upgrade pip
pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax

# Install JAX with GPU support (if needed)
# See https://github.com/jax-ml/jax#installation
```

### Waymo Open Motion Dataset Access

Before running any code that loads data, authenticate with GCP:

```bash
# Via gcloud CLI
gcloud auth login <your_email>
gcloud auth application-default login

# Or in Colab notebooks
from google.colab import auth
auth.authenticate_user()
```

## Running Tests

Tests use pytest and are colocated with source files (named `*_test.py`):

```bash
# Run all tests
python3 -m pytest waymax/

# Run specific test file
python3 -m pytest waymax/env/base_environment_test.py

# Run specific test
python3 -m pytest waymax/env/base_environment_test.py::TestClass::test_method
```

## Architecture

### Core Module Structure

**datatypes/** - Core data structures (all JAX PyTrees):
- `SimulatorState`: Top-level state containing all scenario information
- `Trajectory`: Object trajectories with position, velocity, yaw, bounding boxes
- `RoadgraphPoints`: Road network representation (lanes, edges, etc.)
- `Action`: Control inputs for objects
- `Observation`: Coordinate-frame-transformed views of state
- `Paths`: SDC route paths from roadgraph connectivity

**dataloader/** - WOMD data loading utilities:
- `simulator_state_generator()`: Main function to create iterators over WOMD scenarios
- Pre-configured dataset paths in `config.py`: `WOD_1_0_0_TRAINING`, `WOD_1_1_0_TRAINING`, etc.
- Supports batching, shuffling, parallel loading via TensorFlow datasets

**dynamics/** - Vehicle dynamics models for state transitions:
- `InvertibleBicycleModel`: Kinematically realistic (acceleration, steering curvature)
- `DeltaLocal`/`DeltaGlobal`: Position-based displacement models
- `StateDynamics`: Direct state setting model
- `DiscreteActionSpaceWrapper`: Discretizes continuous action spaces

**env/** - RL environment interfaces:
- `BaseEnvironment`: Multi-agent environment with configurable dynamics
- `PlanningAgentEnvironment`: Single-agent focused environment for ego planning
- `AbstractEnvironment`: Base stateless environment interface
- Wrappers: `dm_env_wrapper.py` (dm-env), `brax_wrapper.py` (Brax)

**agents/** - Simulated agent implementations:
- `WaypointFollowingPolicy`/`IDMRoutePolicy`: IDM-based agents for realistic behavior
- `create_expert_actor()`: Log-replay agent
- `create_constant_speed_actor()`: Simple baseline
- `actor_core_factory()`: Combines multiple agents

**metrics/** - Evaluation metrics:
- `OverlapMetric`: Object-object collision detection
- `OffroadMetric`, `WrongWayMetric`: Roadgraph violations
- `OffRouteMetric`, `ProgressionMetric`: Route following (requires `include_sdc_paths=True`)
- `LogDivergenceMetric`: MSE from logged trajectory
- `KinematicInfeasibilityMetric`: Physics violations

**rewards/** - Reward function for RL:
- `LinearCombinationReward`: Weighted sum of metrics

**visualization/** - Rendering utilities:
- `viz.py`: Top-down visualization of scenarios
- Used extensively in Colab notebooks

**utils/** - Shared utilities:
- `geometry.py`: Geometric operations (coordinate transforms, etc.)
- `test_utils.py`: Test data generators and fixtures

### Configuration System

The `config.py` module contains frozen dataclasses for all configuration:

- `DatasetConfig`: Data loading (path, batching, shuffling, roadgraph/object limits)
- `EnvironmentConfig`: Environment behavior (controlled objects, metrics, rewards, sim agents)
- `ObservationConfig`: Observation generation (history length, roadgraph top-k, coordinate frame)
- `MetricsConfig`: Which metrics to compute
- `SimAgentConfig`: Sim agent configuration for non-controlled objects
- `WaymaxConfig`: Top-level config combining data + environment

Pre-configured dataset paths: `WOD_1_0_0_*`, `WOD_1_1_0_*`, `WOD_1_2_0_*`, `WOD_1_3_0_*`, `WOD_1_3_1_*` for training/validation/testing splits.

### Coordinate Frames

Waymax supports three coordinate systems via `CoordinateFrame` enum:
- `GLOBAL`: WOMD global coordinates
- `SDC`: Centered on ego vehicle (SDC)
- `OBJECT`: Centered on each object individually

Use `datatypes.observation` functions to transform between frames.

### Key Design Patterns

1. **Stateless Environments**: Environments don't hold state; `state` is passed to `step()`, `reset()`, etc.
2. **JAX PyTrees**: All data structures are JAX-compatible PyTrees (vmap/pmap/jit friendly)
3. **Masking**: Most arrays have `.valid` masks to handle variable-length data
4. **Object Types**: `ObjectType` enum controls which objects are user-controlled vs. sim-agent-controlled
5. **Timesteps**: WOMD has 10 warmup timesteps (history) + 1 current + 80 future timesteps

## Common Workflows

### Loading Data and Running Simulation

```python
from waymax import config, dataloader, env, dynamics, datatypes

# Load data
data_config = config.WOD_1_1_0_TRAINING
scenarios = dataloader.simulator_state_generator(data_config)

# Create environment
dynamics_model = dynamics.InvertibleBicycleModel()
env_config = config.EnvironmentConfig()
waymax_env = env.BaseEnvironment(dynamics_model, env_config)

# Rollout
state = waymax_env.reset(next(scenarios))
while not state.is_done:
    action = datatypes.Action(data=..., valid=...)  # Your policy
    state = waymax_env.step(state, action)
```

### Using Sim Agents

Configure `sim_agents` in `EnvironmentConfig` to control non-ego objects:

```python
env_config = config.EnvironmentConfig(
    controlled_object=config.ObjectType.SDC,  # User controls SDC only
    sim_agents=[
        config.SimAgentConfig(
            agent_type=config.SimAgentType.IDM,
            controlled_objects=config.ObjectType.MODELED,
        )
    ],
)
```

### Computing Metrics

```python
# Via environment (configured metrics only)
metrics = waymax_env.metrics(state)

# Directly compute specific metric
from waymax.metrics import OverlapMetric
overlap_metric = OverlapMetric()
overlap_result = overlap_metric.compute(state)
```

## Important Constraints

- **Python 3.10+** required
- **JAX-native code**: Avoid non-JAX operations inside JIT-compiled functions
- **Dataset access**: Requires Waymo Open Dataset approval and GCP authentication
- **Non-commercial license**: See LICENSE file for terms
- **Fixed scenario length**: WOMD scenarios are 91 timesteps (10 warmup + 1 current + 80 future)
- **Route metrics**: Require `include_sdc_paths=True` in `DatasetConfig` (only available in WOMD 1.3.1+)

## Development Notes

- All files include Waymax License Agreement header
- Test files colocated with source (`*_test.py`)
- No build system; pure Python package installed via pip
- Documentation hosted at https://waymo-research.github.io/waymax/docs/
- Tutorials in `docs/notebooks/` directory (Colab-ready)
