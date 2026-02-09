#!/usr/bin/env python3
"""
Multi-Agent Closed-Loop Simulation with Waymax

This script demonstrates how to run closed-loop simulation with multiple
agents controlling different objects in the scene.

Agent types available:
- IDMRoutePolicy: Intelligent Driver Model - follows logged route with adaptive speed
- create_constant_speed_actor: Maintains constant speed along logged path
- create_expert_actor: Replays logged trajectory exactly

Usage:
    python multiagent_simulation.py

    # With real WOMD data (requires GCP authentication):
    python multiagent_simulation.py --use-womd

    # Save visualization video:
    python multiagent_simulation.py --save-video
"""

import argparse
import dataclasses
from datetime import datetime
import os
from pathlib import Path
from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np

from waymax import agents
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax import metrics as _metrics
from waymax import visualization
from waymax.utils import test_utils


def load_scenario(use_womd: bool = False, max_num_objects: int = 32):
    """Load a scenario from test data or WOMD."""
    if use_womd:
        # Load from Waymo Open Motion Dataset (requires GCP auth)
        print("Loading from WOMD (requires GCP authentication)...")
        config = dataclasses.replace(
            _config.WOD_1_1_0_VALIDATION,
            max_num_objects=max_num_objects,
        )
    else:
        # Load from test data (no auth required)
        print("Loading from test data...")
        config = _config.DatasetConfig(
            path=test_utils.ROUTE_DATA_PATH,
            data_format=_config.DataFormat.TFRECORD,
            include_sdc_paths=True,
            max_num_objects=max_num_objects,
            num_paths=test_utils.ROUTE_NUM_PATHS,
            num_points_per_path=test_utils.ROUTE_NUM_POINTS_PER_PATH,
        )

    data_iter = dataloader.simulator_state_generator(config=config)
    scenario = next(data_iter)
    return scenario


def create_multi_agent_actors(
    dynamics_model: dynamics.DynamicsModel,
    max_num_objects: int,
) -> list:
    """
    Create multiple agents with different behaviors.

    Agent assignment:
    - Object 0 (SDC): IDM policy (adaptive speed based on distance to lead vehicle)
    - Object 1: IDM policy
    - Object 2: Constant speed (5 m/s)
    - Object 3-4: Expert/log replay (follows exact logged trajectory)
    - Objects 5+: Static (speed = 0)
    """
    obj_idx = jnp.arange(max_num_objects)

    actors = []

    # 1. IDM agent for SDC and object 1
    # IDM adjusts speed based on distance to vehicles ahead
    idm_actor = agents.IDMRoutePolicy(
        is_controlled_func=lambda state: (obj_idx == 0) | (obj_idx == 1)
    )
    actors.append(("IDM (objects 0,1)", idm_actor))

    # 2. Constant speed agent for object 2
    constant_speed_actor = agents.create_constant_speed_actor(
        speed=5.0,  # 5 m/s
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: obj_idx == 2,
    )
    actors.append(("Constant Speed 5m/s (object 2)", constant_speed_actor))

    # 3. Expert/log replay for objects 3-4
    expert_actor = agents.create_expert_actor(
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: (obj_idx == 3) | (obj_idx == 4),
    )
    actors.append(("Expert Replay (objects 3,4)", expert_actor))

    # 4. Static objects (speed = 0) for objects 5+
    static_actor = agents.create_constant_speed_actor(
        speed=0.0,
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: obj_idx > 4,
    )
    actors.append(("Static (objects 5+)", static_actor))

    return actors


def run_simulation(
    scenario: datatypes.SimulatorState,
    actors: list,
    dynamics_model: dynamics.DynamicsModel,
    max_num_objects: int,
    verbose: bool = True,
) -> list[datatypes.SimulatorState]:
    """
    Run closed-loop multi-agent simulation.

    Returns list of states for each timestep.
    """
    # Configure environment
    env_config = dataclasses.replace(
        _config.EnvironmentConfig(),
        max_num_objects=max_num_objects,
        # User controls all valid objects via our multi-agent setup
        controlled_object=_config.ObjectType.VALID,
    )

    env = _env.BaseEnvironment(
        dynamics_model=dynamics_model,
        config=env_config,
    )

    # JIT compile for performance
    jit_step = jax.jit(env.step)
    jit_select_actions = [jax.jit(actor.select_action) for _, actor in actors]

    # Reset environment
    states = [env.reset(scenario)]
    initial_state = states[0]

    if verbose:
        print(f"\nRunning simulation for {initial_state.remaining_timesteps} timesteps...")

    # Simulation loop
    for step in range(initial_state.remaining_timesteps):
        current_state = states[-1]

        # Get action from each agent
        outputs = [
            jit_select_action({}, current_state, None, None)
            for jit_select_action in jit_select_actions
        ]

        # Merge actions from all agents
        merged_action = agents.merge_actions(outputs)

        # Step environment
        next_state = jit_step(current_state, merged_action)
        states.append(next_state)

        if verbose and step % 20 == 0:
            print(f"  Step {step + 1}/{initial_state.remaining_timesteps}")

    if verbose:
        print(f"  Simulation complete! Final timestep: {states[-1].timestep}")

    return states


def compute_metrics(
    states: list[datatypes.SimulatorState],
) -> dict:
    """Compute evaluation metrics on the final state."""
    final_state = states[-1]

    metrics_results = {}

    # Overlap (collision) metric
    overlap_metric = _metrics.OverlapMetric()
    overlap_result = overlap_metric.compute(final_state)
    metrics_results["overlap"] = overlap_result

    # Offroad metric
    offroad_metric = _metrics.OffroadMetric()
    offroad_result = offroad_metric.compute(final_state)
    metrics_results["offroad"] = offroad_result

    # Log divergence (how far from logged trajectory)
    log_divergence_metric = _metrics.LogDivergenceMetric()
    log_divergence_result = log_divergence_metric.compute(final_state)
    metrics_results["log_divergence"] = log_divergence_result

    return metrics_results


def get_output_dir() -> Path:
    """Get the outputs directory path, creating it if needed."""
    # Find the repo root (parent of examples/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    output_dir = repo_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@dataclasses.dataclass
class VisualizationConfig:
    """Configuration for what to show in visualization."""
    show_log_trajectory: bool = False  # Show ground truth logged path
    show_sdc_paths: bool = False  # Show SDC route options
    show_velocity_vectors: bool = False  # Show velocity arrows
    show_future_trajectory: bool = False  # Show future logged trajectory
    show_metrics: bool = False  # Show metrics overlay
    show_agent_ids: bool = True  # Show agent ID labels
    show_legend: bool = False  # Show color legend
    future_steps: int = 20  # How many future steps to show


def plot_enhanced_state(
    ax,
    state: datatypes.SimulatorState,
    viz_config,
    vis_options: VisualizationConfig,
    metrics_results: Optional[dict] = None,
):
    """Plot simulator state with enhanced visualization options."""
    # Base visualization
    visualization.plot_simulator_state_matplotlib(
        ax, state, viz_config, use_log_traj=False
    )

    timestep = int(state.timestep)

    # Show logged trajectory (ground truth) as dotted line
    if vis_options.show_log_trajectory:
        log_traj = state.log_trajectory
        for obj_idx in range(log_traj.num_objects):
            if not log_traj.valid[obj_idx, timestep]:
                continue
            # Future logged trajectory
            future_valid = log_traj.valid[obj_idx, timestep:]
            if np.any(future_valid):
                future_x = log_traj.x[obj_idx, timestep:][future_valid]
                future_y = log_traj.y[obj_idx, timestep:][future_valid]
                ax.plot(future_x, future_y, 'g--', alpha=0.5, linewidth=1,
                       label='Log trajectory' if obj_idx == 0 else None)

    # Show SDC paths (route options)
    if vis_options.show_sdc_paths and state.sdc_paths is not None:
        paths = state.sdc_paths
        for i in range(paths.shape[0]):
            if not np.any(paths.valid[i]):
                continue
            valid_mask = paths.valid[i]
            x = paths.x[i][valid_mask]
            y = paths.y[i][valid_mask]
            # Cyan for on-route, light green for off-route
            color = 'cyan' if paths.on_route[i, 0] else 'lightgreen'
            ax.plot(x, y, '.', color=color, ms=2, alpha=0.3,
                   label='SDC path' if i == 0 else None)

    # Show velocity vectors
    if vis_options.show_velocity_vectors:
        sim_traj = state.sim_trajectory
        # Collect all velocity vectors for quiver plot
        xs, ys, vxs, vys = [], [], [], []
        for obj_idx in range(sim_traj.num_objects):
            if not sim_traj.valid[obj_idx, timestep]:
                continue
            x = float(sim_traj.x[obj_idx, timestep])
            y = float(sim_traj.y[obj_idx, timestep])
            vx = float(sim_traj.vel_x[obj_idx, timestep])
            vy = float(sim_traj.vel_y[obj_idx, timestep])
            # Skip if velocity is near zero
            speed = np.sqrt(vx**2 + vy**2)
            if speed < 0.1:
                continue
            xs.append(x)
            ys.append(y)
            vxs.append(vx)
            vys.append(vy)

        if xs:
            # Use quiver for better arrow rendering
            # Red color for high visibility, scale adjusts arrow length
            ax.quiver(xs, ys, vxs, vys,
                     color='red', scale=25, width=0.012,
                     headwidth=4, headlength=5, headaxislength=4,
                     zorder=100, alpha=0.95,
                     label='Velocity')

    # Show future trajectory (where agents will go based on log)
    if vis_options.show_future_trajectory:
        log_traj = state.log_trajectory
        future_end = min(timestep + vis_options.future_steps, log_traj.num_timesteps)
        for obj_idx in range(log_traj.num_objects):
            if not log_traj.valid[obj_idx, timestep]:
                continue
            future_slice = slice(timestep, future_end)
            future_valid = log_traj.valid[obj_idx, future_slice]
            if np.any(future_valid):
                future_x = log_traj.x[obj_idx, future_slice][future_valid]
                future_y = log_traj.y[obj_idx, future_slice][future_valid]
                ax.plot(future_x, future_y, 'b:', alpha=0.4, linewidth=1)

    # Show metrics overlay
    if vis_options.show_metrics and metrics_results:
        # Get view bounds
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        text_x = xlim[0] + 5
        text_y = ylim[1] - 5

        metrics_text = []
        for name, result in metrics_results.items():
            if hasattr(result, 'value'):
                val = result.value
                valid = result.valid if hasattr(result, 'valid') else np.ones_like(val, dtype=bool)
                valid_vals = val[valid]
                if len(valid_vals) > 0:
                    mean_val = float(np.mean(valid_vals))
                    metrics_text.append(f"{name}: {mean_val:.2f}")

        if metrics_text:
            ax.text(text_x, text_y, '\n'.join(metrics_text),
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Show legend
    if vis_options.show_legend:
        ax.legend(loc='upper right', fontsize=6)


def save_visualization(
    states: list[datatypes.SimulatorState],
    output_path: Optional[str] = None,
    fps: int = 10,
    vis_options: Optional[VisualizationConfig] = None,
    metrics_results: Optional[dict] = None,
):
    """Save simulation as MP4 video (or GIF if .gif extension)."""
    try:
        import imageio.v3 as iio
    except ImportError:
        print("Install imageio to save visualization: pip install imageio imageio-ffmpeg")
        return

    import io
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from PIL import Image
    from waymax.visualization import utils as viz_utils

    if vis_options is None:
        vis_options = VisualizationConfig()

    # Default to outputs directory with MP4 format and timestamp
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(get_output_dir() / f"simulation_{timestamp}.mp4")

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualization...")
    imgs = []
    viz_config = viz_utils.VizConfig(show_agent_id=vis_options.show_agent_ids)

    # Fixed figure size for consistent frame dimensions
    # Use dimensions divisible by 16 for video codec compatibility (macro_block_size)
    frame_size = (640, 640)  # 640 is divisible by 16
    dpi = 100
    fig_width = frame_size[0] / dpi
    fig_height = frame_size[1] / dpi

    for i, state in enumerate(states):
        # Create figure with fixed size
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        # Use enhanced plotting
        plot_enhanced_state(ax, state, viz_config, vis_options, metrics_results)

        # Save to buffer and convert to numpy array
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, pad_inches=0.1)
        buf.seek(0)
        img = Image.open(buf)
        # Resize to ensure consistent dimensions
        img = img.resize(frame_size, Image.Resampling.LANCZOS)
        imgs.append(np.array(img.convert('RGB')))
        plt.close(fig)
        buf.close()

        if i % 20 == 0:
            print(f"  Rendering frame {i + 1}/{len(states)}")

    print(f"Saving to {output_path}...")

    # Choose format based on extension
    if output_path.endswith('.gif'):
        iio.imwrite(output_path, imgs, fps=fps, loop=0)
    else:
        # MP4 with h264 codec for broad compatibility
        iio.imwrite(
            output_path,
            imgs,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',  # For compatibility with most players
        )

    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Closed-Loop Simulation with Waymax"
    )
    parser.add_argument(
        "--use-womd",
        action="store_true",
        help="Use real WOMD data (requires GCP authentication)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save simulation as MP4 video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for visualization (default: outputs/simulation_output.mp4). Use .gif extension for GIF format.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for video output (default: 10)",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=32,
        help="Maximum number of objects to load",
    )

    # Visualization options
    viz_group = parser.add_argument_group("visualization options")
    viz_group.add_argument(
        "--show-log-traj",
        action="store_true",
        help="Show ground truth logged trajectory (green dashed)",
    )
    viz_group.add_argument(
        "--show-sdc-paths",
        action="store_true",
        help="Show SDC route options (cyan=on-route, green=off-route)",
    )
    viz_group.add_argument(
        "--show-velocity",
        action="store_true",
        help="Show velocity vectors (orange arrows)",
    )
    viz_group.add_argument(
        "--show-future",
        action="store_true",
        help="Show future logged trajectory (blue dotted)",
    )
    viz_group.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show metrics overlay on video",
    )
    viz_group.add_argument(
        "--show-legend",
        action="store_true",
        help="Show color legend",
    )
    viz_group.add_argument(
        "--hide-agent-ids",
        action="store_true",
        help="Hide agent ID labels",
    )
    viz_group.add_argument(
        "--show-all",
        action="store_true",
        help="Enable all visualization options",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Multi-Agent Closed-Loop Simulation")
    print("=" * 70)

    # 1. Load scenario
    print("\n1. Loading scenario...")
    scenario = load_scenario(
        use_womd=args.use_womd,
        max_num_objects=args.max_objects,
    )
    print(f"   Loaded scenario with {scenario.num_objects} objects")
    print(f"   Scenario duration: {scenario.remaining_timesteps + 1} timesteps (9 seconds)")

    # 2. Setup dynamics model
    # StateDynamics allows direct state setting (simpler)
    # InvertibleBicycleModel is more realistic but requires compatible actions
    print("\n2. Setting up dynamics model...")
    dynamics_model = dynamics.StateDynamics()
    print(f"   Using {dynamics_model.__class__.__name__}")

    # 3. Create multi-agent actors
    print("\n3. Creating multi-agent actors...")
    actors = create_multi_agent_actors(dynamics_model, args.max_objects)
    for name, _ in actors:
        print(f"   - {name}")

    # 4. Run simulation
    print("\n4. Running closed-loop simulation...")
    states = run_simulation(
        scenario=scenario,
        actors=actors,
        dynamics_model=dynamics_model,
        max_num_objects=args.max_objects,
    )

    # 5. Compute metrics
    print("\n5. Computing metrics...")
    metrics_results = compute_metrics(states)

    for metric_name, result in metrics_results.items():
        value = result.value
        valid = result.valid if hasattr(result, "valid") else jnp.ones_like(value, dtype=bool)

        # Compute mean over valid values
        valid_values = value[valid]
        if len(valid_values) > 0:
            mean_val = float(jnp.mean(valid_values))
            max_val = float(jnp.max(valid_values))
            print(f"   {metric_name}: mean={mean_val:.4f}, max={max_val:.4f}")
        else:
            print(f"   {metric_name}: N/A")

    # 6. Save visualization (optional)
    if args.save_video:
        # Build visualization config from args
        vis_options = VisualizationConfig(
            show_log_trajectory=args.show_log_traj or args.show_all,
            show_sdc_paths=args.show_sdc_paths or args.show_all,
            show_velocity_vectors=args.show_velocity or args.show_all,
            show_future_trajectory=args.show_future or args.show_all,
            show_metrics=args.show_metrics or args.show_all,
            show_agent_ids=not args.hide_agent_ids,
            show_legend=args.show_legend or args.show_all,
        )
        save_visualization(
            states,
            args.output,
            fps=args.fps,
            vis_options=vis_options,
            metrics_results=metrics_results,
        )

    # Summary
    print("\n" + "=" * 70)
    print("Simulation Summary")
    print("=" * 70)
    print(f"  Objects simulated: {scenario.num_objects}")
    print(f"  Timesteps: {len(states)}")
    print(f"  Dynamics model: {dynamics_model.__class__.__name__}")
    print(f"  Agent types: {len(actors)}")
    print("\nAgent behaviors:")
    print("  - IDM: Adapts speed based on distance to leading vehicle")
    print("  - Constant Speed: Maintains fixed velocity along path")
    print("  - Expert Replay: Follows exact logged trajectory")
    print("  - Static: Remains stationary")
    print("=" * 70)


if __name__ == "__main__":
    main()
