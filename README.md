# EKF Localization with Landmark Observations

A from-scratch implementation of an **Extended Kalman Filter (EKF)** for 2D mobile-robot localization, deployed as a **ROS 2 Humble** node inside Docker. The filter fuses noisy wheel-encoder odometry with sparse range-bearing landmark observations to produce a corrected pose estimate.

---

## Demo — Three Noise Scenarios

<p align="center">
  <img src="analysis/plots/ekf_localization1.gif" width="30%">
  <img src="analysis/plots/ekf_localization2.gif" width="30%">
  <img src="analysis/plots/ekf_localization3.gif" width="30%">
</p>
<p align="center">
  <b>Low Noise</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>High Noise</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>Moderate Noise</b>
</p>
<p align="center">
  <i>Green = Ground Truth &nbsp;|&nbsp; Red dashed = Noisy Odometry &nbsp;|&nbsp; Blue = EKF Estimate &nbsp;|&nbsp; Blue ellipse = 2σ covariance</i>
</p>

> To regenerate the GIFs after any code change: `python3 analysis/ekf_gif.py`

---

## Quick Results

| Method | RMSE (m) | vs Odometry |
|--------|----------|-------------|
| Raw Odometry | 0.1303 | — |
| `robot_localization` (odom-only) | 0.0513 | +60.6% better |
| **This EKF (landmark-fused)** | **0.0126** | **+90.3% better** |

Full breakdown → [`docs/RESULTS.md`](docs/RESULTS.md)  
Comparison with `robot_localization` → [`docs/COMPARISON.md`](docs/COMPARISON.md)

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/SIMULATION.md`](docs/SIMULATION.md) | Math framework, noise scenarios, GIF plots, re-run instructions |
| [`docs/RVIZ.md`](docs/RVIZ.md) | ROS 2 node architecture, Docker setup, RViz topics, covariance ellipse screenshots |
| [`docs/RESULTS.md`](docs/RESULTS.md) | All quantitative results — update trace, trajectory table, RMSE across scenarios |
| [`docs/COMPARISON.md`](docs/COMPARISON.md) | Head-to-head comparison with `robot_localization` package |

---

## Repository Structure

```
ekf_localisation_project/
│
├── README.md
├── docs/
│   ├── SIMULATION.md
│   ├── RVIZ.md
│   ├── RESULTS.md
│   └── COMPARISON.md
│
├── ros2_ws/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── src/ekf_package/
│       ├── package.xml
│       ├── setup.py
│       └── ekf_package/
│           ├── __init__.py
│           ├── ekf.py                    # Core EKF math — no ROS dependency
│           ├── ekf_node.py               # ROS 2 node (subscribe / filter / publish)
│           └── simulation_publisher.py   # Publishes .npy data at 10 Hz
│
└── analysis/
    ├── ekf.py                  # Standalone copy of EKF (no ROS)
    ├── simulation.py           # Generates .npy scenario data
    ├── compare_rmse.py         # RMSE bar chart + error curves (3 scenarios)
    ├── ekf_all_scenarios_plot.py  # Trajectory plots for all scenarios
    ├── ekf_gif.py              # Animated GIFs
    └── plots/                  # Auto-created output directory
        └── rviz_plots/
```

---

## Running the Project

### ROS 2 (Docker)

```bash
cd ros2_ws
docker-compose up --build -d

# Terminal 1 — build
docker exec -it ekf_localisation_dev_container bash
colcon build --symlink-install && source install/setup.bash

# Terminal 2 — EKF node
ros2 run ekf_package ekf_node

# Terminal 3 — RViz
rviz2 -d /ros2_ws/src/rviz_config.rviz

# Terminal 4 — simulation
ros2 run ekf_package simulation_publisher
```

See [`docs/RVIZ.md`](docs/RVIZ.md) for the full multi-terminal procedure including `robot_localization`.

### Standalone Python (no ROS)

```bash
cd analysis/
python3 simulation.py            # generate data (once)
python3 ekf_all_scenarios_plot.py
python3 compare_rmse.py
python3 ekf_gif.py
```

Requirements: `pip install numpy matplotlib pillow`

---

## References

- S. Thrun, W. Burgard, D. Fox — *Probabilistic Robotics*, MIT Press, 2005
- R. Siegwart, I. R. Nourbakhsh — *Introduction to Autonomous Mobile Robots*, MIT Press, 2011
- T. Moore, D. Stouch — *A Generalized Extended Kalman Filter Implementation for the Robot Operating System*, IAS-13, 2014
- [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/)
