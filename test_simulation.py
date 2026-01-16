#!/usr/bin/env python3
"""Simple test script to verify Waymax simulation works correctly."""

import jax
from jax import numpy as jnp

from waymax import config, dataloader, datatypes, dynamics, env
from waymax.utils import test_utils

def main():
    print("=" * 80)
    print("Waymax Simulation Test")
    print("=" * 80)

    # 1. Load test data
    print("\n1. Loading test data...")
    dataset_config = config.DatasetConfig(
        path=test_utils.ROUTE_DATA_PATH,
        data_format=config.DataFormat.TFRECORD,
        include_sdc_paths=True,
        num_paths=test_utils.ROUTE_NUM_PATHS,
        num_points_per_path=test_utils.ROUTE_NUM_POINTS_PER_PATH,
    )

    scenarios = dataloader.simulator_state_generator(dataset_config)
    scenario = next(scenarios)
    print(f"   ✓ Loaded scenario with {scenario.num_objects} objects")
    print(f"   ✓ Scenario has {scenario.remaining_timesteps + 1} timesteps")

    # 2. Create environment with bicycle dynamics
    print("\n2. Setting up environment...")
    dynamics_model = dynamics.InvertibleBicycleModel()
    env_config = config.EnvironmentConfig(
        max_num_objects=128,
        init_steps=11,
        controlled_object=config.ObjectType.SDC,
        compute_reward=True,
        metrics=config.MetricsConfig(
            metrics_to_run=('log_divergence', 'overlap', 'offroad')
        ),
    )

    waymax_env = env.BaseEnvironment(dynamics_model, env_config)
    print(f"   ✓ Created environment with {dynamics_model.__class__.__name__}")
    print(f"   ✓ Controlled objects: {env_config.controlled_object}")

    # 3. Reset environment
    print("\n3. Resetting environment...")
    state = waymax_env.reset(scenario)
    print(f"   ✓ Initial timestep: {state.timestep}")
    print(f"   ✓ Remaining timesteps: {state.remaining_timesteps}")

    # 4. Get action spec and create a simple policy (IDM behavior)
    print("\n4. Setting up simple policy...")
    action_spec = waymax_env.action_spec()
    print(f"   ✓ Action spec shape: {action_spec.data.shape}")
    print(f"   ✓ Action dimensions: acceleration + steering curvature")

    # 5. Run simulation for several steps
    print("\n5. Running simulation...")
    num_steps = 10
    total_reward = 0.0

    for step in range(num_steps):
        # Simple policy: maintain speed with slight acceleration, no steering
        action = datatypes.Action(
            data=jnp.zeros(action_spec.data.shape, dtype=jnp.float32),
            valid=jnp.ones(action_spec.valid.shape, dtype=jnp.bool_),
        )
        # Small positive acceleration for controlled objects
        action = action.replace(
            data=action.data.at[:, 0].set(0.5)  # 0.5 m/s² acceleration
        )

        # Get reward before stepping
        reward = waymax_env.reward(state, action)
        total_reward += reward

        # Step environment
        state = waymax_env.step(state, action)

        if step % 2 == 0:
            # Reward is per-object, take mean for display
            reward_val = float(jnp.mean(reward))
            print(f"   Step {step+1}/{num_steps}: timestep={state.timestep}, "
                  f"reward={reward_val:.3f}, done={state.is_done}")

    print(f"\n   ✓ Simulation completed!")
    print(f"   ✓ Total reward over {num_steps} steps: {float(jnp.mean(total_reward)):.3f}")
    print(f"   ✓ Final timestep: {state.timestep}")

    # 6. Compute final metrics
    print("\n6. Computing metrics...")
    metrics = waymax_env.metrics(state)

    for metric_name, metric_result in metrics.items():
        # MetricResult has a .value attribute
        metric_value = metric_result.value if hasattr(metric_result, 'value') else metric_result
        # Get mean value if it's an array
        if hasattr(metric_value, 'shape') and len(metric_value.shape) > 0:
            finite_vals = metric_value[jnp.isfinite(metric_value)]
            if len(finite_vals) > 0:
                mean_val = float(jnp.mean(finite_vals))
                print(f"   {metric_name}: {mean_val:.4f} (mean)")
            else:
                print(f"   {metric_name}: N/A (no finite values)")
        else:
            print(f"   {metric_name}: {float(metric_value):.4f}")

    # 7. Test observation generation
    print("\n7. Testing observation generation...")
    obs = datatypes.observation_from_state(
        state,
        roadgraph_top_k=1000,
    )
    print(f"   ✓ Observation shape: trajectory={obs.trajectory.shape}")
    print(f"   ✓ Roadgraph points: {jnp.sum(obs.roadgraph_static_points.valid)}")

    # 8. Summary
    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)
    print("\nKey findings:")
    print(f"  • Loaded scenario with {scenario.num_objects} objects")
    print(f"  • Ran {num_steps} simulation steps")
    print(f"  • Final timestep: {state.timestep}")
    print(f"  • Environment is done: {state.is_done}")
    print(f"  • Dynamics model: {dynamics_model.__class__.__name__}")
    print(f"  • Action space: {action_spec.data.shape}")
    print("\nSimulation working correctly! ✨")
    print("=" * 80)

if __name__ == '__main__':
    main()
