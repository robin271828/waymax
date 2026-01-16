# Waymax Repository Guide

This guide provides a comprehensive overview of how the Waymax repository is structured and how the simulator works.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Core Concepts](#core-concepts)
- [Data Flow](#data-flow)
- [Module Deep Dive](#module-deep-dive)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Common Patterns](#common-patterns)

## Overview

Waymax is a JAX-based autonomous driving simulator built on the Waymo Open Motion Dataset (WOMD). The repository is structured as a pure Python package with all code in the `waymax/` directory. The key design principle is that **everything is a JAX PyTree**, enabling full hardware acceleration and functional programming patterns.

### Key Characteristics

- **Stateless Design**: Environments don't hold state; state is explicitly passed to all functions
- **Functional**: Pure functions that can be JIT-compiled, vmap'd, and pmap'd
- **Hardware Accelerated**: Runs on CPU, GPU, or TPU through JAX
- **Data-Driven**: Directly consumes Waymo Open Motion Dataset TFRecords

## Repository Structure

```
waymax/
├── agents/              # Simulated agents (IDM, constant speed, expert)
├── config.py            # Configuration dataclasses for everything
├── dataloader/          # WOMD data loading and parsing
├── datatypes/           # Core data structures (all JAX PyTrees)
├── dynamics/            # Vehicle dynamics models
├── env/                 # RL environment interfaces
│   └── wrappers/       # dm-env and Brax adapters
├── metrics/             # Evaluation metrics
├── rewards/             # Reward functions
├── utils/               # Geometry and test utilities
└── visualization/       # Rendering utilities

docs/
├── notebooks/           # Colab tutorials
└── *.rst               # Sphinx documentation

setup.py                 # Package setup
CLAUDE.md               # Claude Code integration guide
pixi.toml               # Pixi dependency management
test_simulation.py      # Quick simulation test script
```

## Core Concepts

### 1. SimulatorState

The top-level state container that holds everything about a scenario:

```python
@dataclass
class SimulatorState:
    sim_trajectory: Trajectory      # Simulated object trajectories
    log_trajectory: Trajectory      # Logged/ground-truth trajectories
    object_metadata: ObjectMetadata # Object IDs, types, flags
    roadgraph_points: RoadgraphPoints  # Road network
    log_traffic_light: TrafficLights   # Traffic signals
    sdc_paths: Paths  # Optional: SDC route paths
    timestep: int     # Current timestep (0-90)
```

**Key Points:**
- Immutable (frozen dataclass)
- Contains both sim and log trajectories
- `timestep` ranges from 0 to 90 (10 warmup + 1 current + 80 future)
- Replace fields with `.replace()` to create new states

### 2. Trajectory

Container for object motion over time:

```python
@dataclass
class Trajectory:
    x, y, z: Array[num_objects, num_timesteps]
    vel_x, vel_y: Array[num_objects, num_timesteps]
    yaw: Array[num_objects, num_timesteps]
    length, width, height: Array[num_objects, num_timesteps]
    valid: Array[num_objects, num_timesteps]  # boolean mask
    timestamp_micros: Array[num_objects, num_timesteps]
```

**Key Points:**
- All arrays have shape `(num_objects, num_timesteps)`
- `valid` mask handles variable number of objects
- Positions are in global coordinates by default
- Can transform to SDC or object-centric frames

### 3. Coordinate Frames

Waymax supports three coordinate systems:

1. **GLOBAL** (default): WOMD world coordinates
2. **SDC**: Ego-vehicle centered (SDC at origin, heading is x-axis)
3. **OBJECT**: Each object centered in its own frame

Transform between frames using `datatypes.observation_from_state()` or `datatypes.transform_trajectory()`.

### 4. Action

Control inputs for objects:

```python
@dataclass
class Action:
    data: Array[num_objects, action_dim]
    valid: Array[num_objects]  # boolean mask
```

- `action_dim` depends on dynamics model (e.g., 2 for InvertibleBicycleModel)
- Only controlled objects need valid actions
- Non-controlled objects can have `valid=False`

### 5. ObjectType

Controls which objects the user controls vs. sim agents:

- `SDC`: Only the ego vehicle
- `MODELED`: Objects marked for prediction (tracks_to_predict)
- `VALID`: All valid objects in the scene
- `NON_SDC`: All objects except the ego vehicle

Configure via `EnvironmentConfig.controlled_object`.

## Data Flow

### Typical Simulation Loop

```
1. Load Scenario
   └─> dataloader.simulator_state_generator(config)
       └─> Yields SimulatorState objects

2. Reset Environment
   └─> env.reset(scenario)
       ├─> Copies log trajectory to sim trajectory for init_steps
       ├─> Sets timestep = init_steps - 1
       └─> Returns initial SimulatorState

3. Simulation Loop
   While not state.is_done:

   a. Get Action
      └─> Your policy generates action
          └─> Action(data, valid)

   b. Step Environment
      └─> env.step(state, action)
          ├─> dynamics.forward(state, action)  # Update controlled objects
          ├─> Apply sim agents to non-controlled objects
          ├─> Increment timestep
          └─> Returns new SimulatorState

   c. Get Reward (optional)
      └─> env.reward(state, action)
          └─> Evaluates metrics and computes linear combination

   d. Get Metrics (optional)
      └─> env.metrics(state)
          └─> Returns dict of MetricResult objects

4. Done
   └─> state.is_done == True when timestep >= 90
```

### Data Loading Pipeline

```
TFRecord Files (WOMD)
    ↓
[dataloader_utils.tf_examples_dataset]
    ↓ Parse TensorFlow Examples
[womd_dataloader.preprocess_serialized_womd_data]
    ↓ Convert to dict of tf.Tensors
[womd_factories.simulator_state_from_womd_dict]
    ↓ Convert to JAX arrays and create PyTrees
SimulatorState (JAX PyTree)
```

**Key Functions:**
- `simulator_state_generator(config)`: High-level API
- `tf_examples_dataset()`: TensorFlow dataset creation
- `womd_factories.*_from_womd_dict()`: Conversion utilities

## Module Deep Dive

### agents/

Provides intelligent agent implementations for realistic simulation.

**Main Classes:**
- `WaymaxActorCore`: Base interface for agents
- `IDMRoutePolicy`: Intelligent Driver Model with route following
- `WaypointFollowingPolicy`: Follow waypoints with simple dynamics
- `create_expert_actor()`: Log-replay agent
- `create_constant_speed_actor()`: Simple baseline

**Usage Pattern:**
```python
# In EnvironmentConfig
sim_agents=[
    config.SimAgentConfig(
        agent_type=config.SimAgentType.IDM,
        controlled_objects=config.ObjectType.MODELED,
    )
]
```

Sim agents automatically control non-user-controlled objects during `env.step()`.

### dataloader/

Handles loading and parsing WOMD data.

**Key Files:**
- `womd_dataloader.py`: Main API (`simulator_state_generator`)
- `womd_factories.py`: Conversion from dict to PyTrees
- `womd_utils.py`: Feature descriptions and aggregation
- `dataloader_utils.py`: TensorFlow dataset utilities

**Pre-configured Datasets in config.py:**
```python
WOD_1_0_0_TRAINING, WOD_1_0_0_VALIDATION, WOD_1_0_0_TESTING
WOD_1_1_0_TRAINING, WOD_1_1_0_VALIDATION, WOD_1_1_0_TESTING
WOD_1_2_0_TRAINING, ...
WOD_1_3_0_TRAINING, ...
WOD_1_3_1_TRAINING, ...  # Includes sdc_paths
```

### datatypes/

Core data structures - all are JAX PyTrees.

**Key Types:**
- `SimulatorState`: Top-level state
- `Trajectory`: Object motion over time
- `RoadgraphPoints`: Road network representation
- `Action`, `TrajectoryUpdate`: Control inputs
- `Observation`: Coordinate-transformed view
- `ObjectMetadata`: Object properties and flags
- `Paths`: SDC route paths (for route metrics)

**Important Operations (operations.py):**
- `dynamic_slice()`: Slice along arbitrary axis
- `dynamic_index()`: Index along arbitrary axis
- `update_by_mask()`: Conditional update with mask
- `masked_mean()`: Mean ignoring invalid values

### dynamics/

Vehicle dynamics models for state transitions.

**Available Models:**
1. **InvertibleBicycleModel** (Recommended)
   - Action: `[acceleration, steering_curvature]`
   - Kinematically realistic
   - Invertible (can compute action from trajectory)

2. **DeltaLocal**
   - Action: `[dx, dy, dyaw]` relative to current pose
   - Position-based, no physics

3. **DeltaGlobal**
   - Action: `[dx, dy, dyaw]` in global frame
   - Position-based, no physics

4. **StateDynamics**
   - Action: `[x, y, yaw, vel_x, vel_y]`
   - Direct state setting

**Discretization:**
```python
# Discretize any dynamics model
discrete_dynamics = dynamics.DiscreteActionSpaceWrapper(
    base_dynamics,
    bins_per_dimension=[5, 5]  # 5x5 = 25 discrete actions
)
```

### env/

RL environment interfaces.

**Main Environments:**

1. **BaseEnvironment**
   - Multi-agent environment
   - All objects can be controlled
   - Configure controlled objects via `ObjectType`

2. **PlanningAgentEnvironment**
   - Single-agent (SDC only)
   - Specialized for ego planning
   - Sim agents control all other objects

**Methods:**
- `reset(scenario)`: Initialize from scenario
- `step(state, action)`: Transition to next state
- `reward(state, action)`: Compute reward
- `metrics(state)`: Compute all configured metrics
- `action_spec()`, `observation_spec()`: Get specs

**Wrappers:**
- `dm_env_wrapper.py`: Wraps stateless env in stateful dm-env interface
- `brax_wrapper.py`: Brax environment adapter

### metrics/

Evaluation metrics for agent performance.

**Available Metrics:**
- `OverlapMetric`: Object-object collision detection
- `OffroadMetric`: Off-roadgraph violations
- `WrongWayMetric`: Wrong-way driving
- `OffRouteMetric`: Route-following accuracy (requires sdc_paths)
- `ProgressionMetric`: Forward progress along route
- `LogDivergenceMetric`: MSE from logged trajectory
- `KinematicInfeasibilityMetric`: Physics violations

**Usage:**
```python
# Via environment
env_config = config.EnvironmentConfig(
    metrics=config.MetricsConfig(
        metrics_to_run=('overlap', 'offroad', 'log_divergence')
    )
)
metrics = env.metrics(state)  # Returns dict of MetricResults

# Directly
overlap_metric = metrics.OverlapMetric()
result = overlap_metric.compute(state)  # Returns MetricResult
```

**MetricResult:**
```python
@dataclass
class MetricResult:
    value: Array  # Per-object metric values
    valid: Array  # Mask for valid metric values
```

### visualization/

Rendering utilities for visualization.

**Main Module: viz.py**
- `plot_simulator_state()`: Top-down view of scenario
- `plot_observation()`: Visualize observations
- `plot_trajectory_buffer()`: Multi-step trajectory visualization

Used extensively in Colab notebooks in `docs/notebooks/`.

## Development Workflow

### Setup

```bash
# Using pixi (recommended)
pixi install
pixi run test-sim

# Or using pip
pip install -e .
pytest waymax/
```

### Running Tests

```bash
# All tests
pixi run test
# Or: pytest waymax/

# Specific test file
pytest waymax/env/base_environment_test.py

# Specific test
pytest waymax/env/base_environment_test.py::BaseEnvironmentTest::test_reset

# With verbose output
pytest -v waymax/
```

### Adding a New Dynamics Model

1. Create file in `waymax/dynamics/`
2. Inherit from `AbstractDynamics`
3. Implement required methods:
   ```python
   class MyDynamics(AbstractDynamics):
       @property
       def action_spec(self) -> specs.BoundedArray:
           # Return action specification

       def forward(self, state, action) -> SimulatorState:
           # Apply action and return new state
   ```
4. Add tests in `*_test.py`
5. Export in `waymax/dynamics/__init__.py`

### Adding a New Metric

1. Create file in `waymax/metrics/`
2. Inherit from `AbstractMetric`
3. Implement `compute()`:
   ```python
   class MyMetric(AbstractMetric):
       def compute(self, state: SimulatorState) -> MetricResult:
           # Compute per-object metric values
           return MetricResult(value=values, valid=mask)
   ```
4. Register metric:
   ```python
   from waymax.metrics import register_metric
   register_metric('my_metric', MyMetric)
   ```
5. Add tests

## Testing

### Test Organization

- Tests colocated with source: `*_test.py` files next to `*.py` files
- Use `tf.test.TestCase` for TensorFlow integration
- Use `parameterized.TestCase` for parameterized tests

### Test Utilities

`waymax/utils/test_utils.py` provides:
- `ROUTE_DATA_PATH`: Path to test TFRecord
- `make_test_dataset()`: Create test dataset
- `simulated_trajectory_*()`: Create synthetic trajectories

### Example Test

```python
class MyTest(tf.test.TestCase):
    def setUp(self):
        dataset_config = config.DatasetConfig(
            path=test_utils.ROUTE_DATA_PATH,
            data_format=config.DataFormat.TFRECORD,
        )
        self.scenario = next(dataloader.simulator_state_generator(dataset_config))

    def test_something(self):
        # Your test here
        self.assertAllClose(expected, actual)
```

## Common Patterns

### 1. JAX Transformations

```python
# JIT compile for speed
@jax.jit
def fast_step(state, action):
    return env.step(state, action)

# Vectorize over batch dimension
batched_step = jax.vmap(env.step, in_axes=(0, 0))

# Parallelize across devices
parallel_step = jax.pmap(env.step, in_axes=(0, 0))
```

### 2. Working with Masks

```python
# Filter valid objects
valid_objects = trajectory.valid.any(axis=-1)  # Any valid timestep
valid_positions = jnp.where(valid_objects[:, None],
                             trajectory.xyz,
                             jnp.nan)

# Masked operations
mean_speed = datatypes.masked_mean(
    jnp.sqrt(trajectory.vel_x**2 + trajectory.vel_y**2),
    trajectory.valid
)
```

### 3. Slicing Trajectories

```python
# Get history (first 11 timesteps)
history = datatypes.dynamic_slice(trajectory, start=0, size=11, axis=-1)

# Get current timestep
current = datatypes.dynamic_index(trajectory, state.timestep, axis=-1)

# Update future with new trajectory
new_traj = datatypes.dynamic_update_slice_in_dim(
    trajectory, new_values, start=state.timestep+1, axis=-1
)
```

### 4. Coordinate Transforms

```python
# Transform to SDC frame
sdc_obs = datatypes.sdc_observation_from_state(state)

# Transform to global frame
global_traj = datatypes.transform_trajectory(
    trajectory,
    transform_matrix,  # 3x3 transform
)
```

### 5. Custom Reward Function

```python
from waymax.env import typedefs
from waymax import datatypes

def my_reward_fn(state: datatypes.SimulatorState,
                 action: datatypes.Action) -> jax.Array:
    # Compute custom reward
    # Return shape: (num_objects,)
    return rewards

# Use in environment
env_config = config.EnvironmentConfig(compute_reward=False)
env = env.BaseEnvironment(dynamics_model, env_config)

# Manually compute reward
reward = my_reward_fn(state, action)
```

### 6. Rollout Collection

```python
from waymax.env import rollout

# Collect full rollout
output = rollout.rollout(
    env,
    scenario,
    policy_fn=my_policy,  # Callable: state -> action
    num_steps=80,
)

# Access rollout data
states = output.states  # List[SimulatorState]
actions = output.actions  # List[Action]
rewards = output.rewards  # List[Array]
```

## Advanced Topics

### Multi-Device Training

```python
# Replicate state across devices
devices = jax.devices()
replicated_state = jax.device_put_replicated(state, devices)

# Parallel step
@jax.pmap
def parallel_step(state, action):
    return env.step(state, action)

new_states = parallel_step(replicated_state, replicated_actions)
```

### Custom Observation Function

```python
from waymax.env import typedefs

def my_obs_fn(state: datatypes.SimulatorState) -> MyObservation:
    # Extract custom observations
    return obs

# Use in environment
env_config = config.EnvironmentConfig(observation=None)
env = env.BaseEnvironment(dynamics_model, env_config)

# Manually compute observation
obs = my_obs_fn(state)
```

### Scenario Filtering

```python
# Load only specific scenarios
def filter_fn(womd_dict):
    # Return True to keep scenario
    num_objects = womd_dict['state/id'].shape[0]
    return num_objects > 10

scenarios = dataloader.simulator_state_generator(config)
filtered = (s for s in scenarios if filter_fn(s))
```

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Run `pixi install` or `pip install -e .`

2. **"CUDA not available" warning**
   - Install CUDA-enabled JAX: See https://github.com/jax-ml/jax#installation
   - Or ignore - CPU is fine for development

3. **"Invalid action shape" errors**
   - Check `action_spec = env.action_spec()`
   - Ensure action.data has correct shape

4. **Metrics require sdc_paths**
   - Set `include_sdc_paths=True` in DatasetConfig
   - Only available in WOMD 1.3.1+

5. **Out of memory errors**
   - Reduce `batch_dims` in DatasetConfig
   - Reduce `max_num_objects`
   - Use smaller scenarios

## Resources

- **Documentation**: https://waymo-research.github.io/waymax/docs/
- **Paper**: https://arxiv.org/abs/2310.08710
- **WOMD**: https://waymo.com/open/data/motion/
- **Tutorials**: `docs/notebooks/` directory

## Contributing

This is a fork of the official Waymax repository. For contributions to the upstream:

1. Sign the CLA: https://cla.developers.google.com/
2. Follow Google's Open Source Community Guidelines
3. Submit PRs to: https://github.com/waymo-research/waymax

---

**Last Updated**: 2026-01-15

This guide reflects the structure of Waymax v0.1.0. For the latest information, consult the official documentation.
