# EKF Localization with Landmark Observations
### Report

*March 8, 2026*

---

## Abstract

This report describes the design, implementation, and evaluation of an Extended Kalman Filter (EKF) for two-dimensional mobile-robot localization. A robot that relies exclusively on wheel-encoder odometry accumulates position error over time because small per-step noise integrates without bound. To address this, we implemented a sensor-fusion algorithm that corrects the odometry-based state estimate using sparse range-and-bearing observations of known landmarks. The filter was written from scratch in Python (NumPy) and deployed as a ROS 2 Humble node running inside Docker. Three noise scenarios were evaluated. In the high-noise case the EKF reduced the root-mean-square position error from 0.83 m to 0.50 m (~40% improvement); in the moderate-noise case the reduction was from 0.32 m to 0.10 m (~70%). A head-to-head comparison with the standard `robot_localization` ROS 2 package showed that the landmark-fused EKF achieves **0.0126 m RMSE** versus 0.0513 m for `robot_localization` (odometry-only) — a **75% improvement**. All source code is available at https://github.com/16srivarshitha/ekf_localisation_project.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Mathematical Framework](#2-mathematical-framework)
3. [Methodology and Implementation](#3-methodology-and-implementation)
4. [Results and Analysis](#4-results-and-analysis)
5. [Conclusion](#5-conclusion)
- [Appendix — Repository Structure](#appendix--repository-structure)
- [References](#references)

---

## 1. Introduction

Robot localization — estimating a robot's pose [x, y, θ]ᵀ in a global reference frame — is a prerequisite for almost every autonomous navigation task. Wheel encoders are a natural first choice for tracking pose because they are cheap, fast, and require no external infrastructure. Unfortunately, every small error in a velocity reading is integrated into the position estimate; over a full lap of a circular trajectory the cumulative drift can exceed a metre even under mild noise conditions, as the baseline comparison in this work confirms.

The Kalman Filter and its nonlinear extension, the EKF, are the classical solution to this problem. Rather than discarding the odometry reading, the EKF treats it as an uncertain prediction and then *corrects* that prediction using sensor observations whose noise characteristics are known. The correction is weighted by the Kalman Gain, which automatically balances how much to trust the prediction versus the sensor depending on the current uncertainty.

This project set out to answer four concrete questions:

1. Can the Jacobians of a nonlinear unicycle motion model and a range-bearing sensor model be derived analytically and implemented efficiently in NumPy?
2. Does the resulting EKF node, integrated into a ROS 2 publish–subscribe architecture, produce a trajectory that visibly and measurably outperforms raw odometry?
3. How does filter performance degrade as process and measurement noise are increased?
4. How does the hand-written filter compare with results obtained from the established `robot_localization` ROS 2 package?

---

## 2. Mathematical Framework

### 2.1 State and Control Representation

The robot's pose at discrete time *t* is the three-dimensional vector:

```
x_t = [ x_t, y_t, θ_t ]ᵀ
```

where (x_t, y_t) is the position in the global frame and θ_t is the heading angle. The control input derived from the wheel encoders is:

```
u_t = [ v_t, ω_t ]ᵀ
```

with v_t the linear speed and ω_t the angular speed.

### 2.2 Motion Model and Its Jacobians

We use the standard Euler-integration unicycle kinematic model:

```
x_t  = x_{t-1} + v_t · Δt · cos(θ_{t-1})
y_t  = y_{t-1} + v_t · Δt · sin(θ_{t-1})
θ_t  = θ_{t-1} + ω_t · Δt
```

Written compactly as **x**_t = g(**x**_{t-1}, **u**_t). Because g is nonlinear in θ_{t-1}, the EKF replaces it with a first-order Taylor expansion.

**State Jacobian G_t** (3×3) — differentiating g with respect to **x**_{t-1}:

```
G_t = | 1   0   -v·Δt·sin(θ) |
      | 0   1    v·Δt·cos(θ) |
      | 0   0        1        |
```

The only off-diagonal coupling is in the third column: a heading error propagates into both x and y predictions at a rate proportional to v·Δt — physically intuitive.

**Control Jacobian V_t** (3×2) — differentiating g with respect to **u**_t:

```
V_t = | Δt·cos(θ)   0  |
      | Δt·sin(θ)   0  |
      |     0       Δt |
```

V_t maps the control-noise covariance M_t (in velocity space) into state space: V_t · M_t · V_tᵀ.

**Covariance Prediction:**

```
Σ̄_t = G_t · Σ_{t-1} · G_tᵀ  +  V_t · M_t · V_tᵀ
```

### 2.3 Measurement Model and Its Jacobian

The environment contains N_L landmarks at known positions (m_x^(j), m_y^(j)). When a landmark is within range, the sensor returns a noisy range r and bearing φ:

```
r  = √q
φ  = atan2(m_y − y_t,  m_x − x_t) − θ_t        [wrapped to (−π, π]]
```

where q = (m_x − x_t)² + (m_y − y_t)².

**Measurement Jacobian H_t** (2×3):

```
H_t = | -(m_x-x)/√q    -(m_y-y)/√q    0  |
      |  (m_y-y)/q     -(m_x-x)/q    -1  |
```

### 2.4 The Complete EKF Predict–Update Cycle

| Step | Operation | Equation |
|------|-----------|----------|
| 1 | Predict state | x̄_t = g(x_{t-1}, u_t) |
| 2 | Predict covariance | Σ̄_t = G_t Σ_{t-1} G_tᵀ + V_t M_t V_tᵀ |
| 3 | Kalman Gain | K_t = Σ̄_t H_tᵀ (H_t Σ̄_t H_tᵀ + R_t)⁻¹ |
| 4 | Update state | x_t = x̄_t + K_t (z_t − h(x̄_t)) |
| 5 | Update covariance | Σ_t = (I − K_t H_t) Σ̄_t |

The term z_t − h(x̄_t) is the *innovation* — how much the actual sensor reading differs from what the filter predicted. The Kalman Gain K_t automatically scales the correction by the ratio of prediction uncertainty to total (prediction + sensor) uncertainty.

> **Critical implementation detail:** the bearing component of the innovation must be wrapped to (−π, π] after subtraction. Without this, the filter diverges whenever the robot crosses the ±π heading boundary.

---

## 3. Methodology and Implementation

### 3.1 System Architecture

The project is split into two independent layers:

**`ros2_ws/`** — A ROS 2 Humble workspace (built inside Docker) containing two nodes:
- **SimulationPublisher** — reads pre-generated `.npy` data files and publishes them at 10 Hz to `/odom`, `/ground_truth/pose`, `/landmark_observations`, and `/landmarks`
- **EKFNode** — subscribes to the above topics, runs the EKF predict–update cycle, and publishes to `/ekf/estimated_pose`, `/ekf_path`, and `/odom_path` for RViz

**`analysis/`** — A self-contained Python package (no ROS dependency) containing the same `EKFLocalizer` class plus scripts for trajectory plots, RMSE comparison, and GIF generation.

The core class `EKFLocalizer` (`ekf.py`) is intentionally free of any ROS dependency, which made unit-testing and offline scenario analysis straightforward.

### 3.2 Velocity Derivation from Pose Messages

The simulation node publishes raw poses rather than velocity twists. Inside `EKFNode.odom_callback`, velocities are derived by finite differencing:

```
v_t = √((x_t − x_{t-1})² + (y_t − y_{t-1})²) / Δt
ω_t = (θ_t − θ_{t-1}) / Δt        [θ difference wrapped to (−π, π]]
```

where Δt = 0.1 s matches the simulation timestep.

### 3.3 Simulation Parameters

**Table 1 — Simulation configuration**

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Timestep | Δt | 0.10 s |
| Total steps | N | 300 |
| Linear speed | v | 1.0 m/s |
| Angular speed | ω | 0.2 rad/s |
| Number of landmarks | N_L | 5 |
| Max detection range | d_max | 5.0 m |
| Random seed | — | 42 |

**Table 2 — Landmark positions**

| ID | x (m) | y (m) |
|----|-------|-------|
| L0 | 3.0 | 2.0 |
| L1 | 6.0 | 4.0 |
| L2 | 2.0 | 7.0 |
| L3 | 8.0 | 1.0 |
| L4 | 5.0 | 8.0 |

**Table 3 — Noise parameters across three scenarios**

| Scenario | σ_v (m/s) | σ_ω (rad/s) | σ_r (m) | σ_φ (rad) |
|----------|-----------|-------------|---------|-----------|
| 1 — Low Noise | 0.005 | 0.002 | 0.05 | 0.010 |
| 2 — High Noise | 0.200 | 0.100 | 0.50 | 0.150 |
| 3 — Moderate Noise | 0.050 | 0.020 | 0.20 | 0.050 |

### 3.4 Covariance Ellipse Visualisation

The 2σ confidence ellipse for the x–y position is recovered from the top-left 2×2 block of Σ_t using eigendecomposition:

```
Σ_xy = Q Λ Qᵀ
semi-axes = 2√λ₁,  2√λ₂
angle     = atan2(q₁₂, q₁₁)
```

This is published as a `PoseWithCovarianceStamped` message and rendered as an ellipse in RViz. The ellipse shrinks as the robot accumulates landmark observations and grows again when no landmarks are in range.

---

## 4. Results and Analysis

### 4.1 RViz Visualisation

The ROS 2 system publishes three simultaneous paths to RViz:

| Topic | Colour | Description |
|-------|--------|-------------|
| `/true_path` | Green | Ground truth from simulation |
| `/odom_path` | Red | Raw noisy odometry |
| `/ekf_path` | Blue | EKF corrected estimate |
| `/landmarks` | Cyan cylinders | Known landmark positions |
| `/ekf/estimated_pose` | Blue ellipse | Current pose + 2σ covariance |

>  (`analysis/plots/rviz_plots/rviz_overview.png`)
> *(RViz screenshot showing all three paths and landmarks)*

> **[PLOT PLACEHOLDER]** Replace with `analysis/plots/rviz_plots/covariance_ellipse.png`
> *(RViz screenshot showing the 2σ covariance ellipse)*

### 4.2 Trajectory Overview

> **[PLOT PLACEHOLDER]** Replace with `analysis/plots/odometry_trajectory.png`
> *(Ground truth green vs noisy odometry red — base scenario)*

> **[PLOT PLACEHOLDER]** Replace with 3-panel figure from `ekf_all_scenarios_plot.py`
> - `analysis/plots/scenario_1_trajectory.png` — Low Noise
> - `analysis/plots/scenario_2_trajectory.png` — High Noise
> - `analysis/plots/scenario_3_trajectory.png` — Moderate Noise

### 4.3 Odometry Error Growth

> **[PLOT PLACEHOLDER]** Replace with `analysis/plots/odometry_error.png`
> *(Euclidean error vs step — base scenario)*

Euclidean position error between odometry and ground truth grows roughly linearly in the first half of the trajectory, plateaus near 0.21 m around step 200 as accumulated turns partially cancel drift, then stabilises.

### 4.4 EKF Update Step Trace

Table 4 logs the first ten EKF update events. All ten correspond to Landmark L0, which is within the 5 m detection radius from the very first timestep. Two trends are visible: (i) the position estimate moves steadily closer to the true trajectory, and (ii) position variance σ_xx decreases monotonically with each observation — the filter grows more confident.

**Table 4 — EKF update log, first 10 steps (Landmark L0, base scenario)**

| Step | Range (m) | Bearing (rad) | EKF x | EKF y | σ_xx before | σ_xx after |
|------|-----------|---------------|-------|-------|-------------|------------|
| 1 | 3.742 | 0.588 | −0.036 | −0.082 | 0.10010 | 0.04842 |
| 2 | 3.776 | 0.698 | 0.003 | −0.122 | 0.04848 | 0.04016 |
| 3 | 3.180 | 0.461 | 0.222 | −0.075 | 0.04036 | 0.03686 |
| 4 | 3.324 | 0.883 | 0.359 | −0.100 | 0.03682 | 0.03499 |
| 5 | 2.570 | 0.276 | 0.510 | 0.049 | 0.03509 | 0.03411 |
| 6 | 3.188 | 0.577 | 0.565 | 0.059 | 0.03404 | 0.03280 |
| 7 | 3.109 | 0.614 | 0.661 | 0.053 | 0.03270 | 0.03176 |
| 8 | 2.826 | 0.861 | 0.860 | 0.018 | 0.03147 | 0.03078 |
| 9 | 2.053 | 0.396 | 0.950 | 0.141 | 0.03064 | 0.03023 |
| 10 | 2.718 | 0.232 | 0.907 | 0.267 | 0.03003 | 0.02931 |

The reduction from σ_xx = 0.100 to 0.029 over just ten steps (~70% variance reduction in one second of real time) demonstrates that even a single nearby landmark provides substantial localisation benefit.

### 4.5 Trajectory Accuracy at Key Timesteps

Table 5 compares the EKF estimate against ground truth and raw odometry at nine evenly spaced timesteps. The EKF consistently outperforms odometry in the first half of the trajectory. Between steps 200 and 250 the robot moves into a region where no landmarks are within the 5 m detection radius; during this interval the filter operates on prediction alone, and the EKF error temporarily exceeds the odometry error. Once the robot re-enters landmark range near step 300 the error partially recovers, illustrating the "coast-and-correct" behaviour characteristic of sparse landmark EKF.

**Table 5 — Per-step position comparison (base scenario)**

| Step | GT (x, y) | Odometry (x, y) | EKF (x, y) | EKF err (m) | Odom err (m) |
|------|-----------|-----------------|------------|-------------|--------------|
| 10 | (0.994, 0.090) | (0.976, 0.044) | (0.907, 0.267) | 0.198 | 0.049 |
| 25 | (2.403, 0.588) | (2.392, 0.438) | (2.538, 0.529) | 0.147 | 0.151 |
| 50 | (4.230, 2.256) | (4.299, 1.941) | (4.275, 2.224) | 0.056 | 0.323 |
| 75 | (5.034, 4.596) | (5.148, 4.324) | (4.937, 4.580) | 0.098 | 0.295 |
| 100 | (4.617, 7.035) | (4.652, 6.771) | (4.681, 7.074) | 0.075 | 0.266 |
| 150 | (0.805, 9.943) | (0.729, 9.270) | (0.765, 9.808) | 0.141 | 0.677 |
| 200 | (−3.701, 8.306) | (−3.576, 7.373) | (−3.478, 7.950) | 0.420 | 0.942 |
| 250 | (−4.759, 3.630) | (−4.011, 2.587) | (−3.823, 3.157) | 1.048 | 1.282 |
| 300 | (−1.395, 0.213) | (−0.367, −0.359) | (−1.246, 0.076) | 0.203 | 1.176 |

### 4.6 RMSE Comparison Across Noise Scenarios

> **[PLOT PLACEHOLDER]** Replace with `analysis/plots/compare_rmse.png`
> *(RMSE bar chart — generated by `python3 analysis/compare_rmse.py`)*

**Table 6 — RMSE for odometry and EKF across all three scenarios**

| Scenario | Odometry RMSE (m) | EKF RMSE (m) | Improvement |
|----------|-------------------|--------------|-------------|
| 1 — Low Noise | 0.0130 | 0.0125 | ~4% |
| 2 — High Noise | 0.8273 | 0.4965 | ~40% |
| 3 — Moderate Noise | 0.3241 | 0.0960 | ~70% |

**Interpretation:** Scenario 3 shows the strongest relative improvement because the motion noise is moderate (enough signal to track) while the landmark noise remains manageable. In Scenario 1 the noise is so low that odometry itself barely drifts — the EKF offers only marginal gain because there is little to correct. In Scenario 2 the gain is substantial in absolute terms (0.33 m reduction) but very noisy sensor readings limit how tightly the filter can constrain the estimate.

### 4.7 Comparison with `robot_localization`

We ran the standard `robot_localization` ROS 2 package configured to fuse only `/odom` (no landmark input) alongside our EKF on identical simulation data. This makes it a **pure odometry-smoothing filter** — the fairest possible comparison, since both filters receive identical odometry data but only ours receives landmark corrections.

**Table 7 — RMSE comparison: this EKF vs `robot_localization` vs raw odometry (real ROS data)**

| Method | RMSE (m) | vs Odometry | vs `robot_localization` |
|--------|----------|-------------|------------------------|
| Raw Odometry | 0.1303 | — | −154% worse |
| `robot_localization` (odom-only) | 0.0513 | +60.6% better | — |
| **This EKF (landmark-fused)** | **0.0126** | **+90.3% better** | **+75.4% better** |

> **[PLOT PLACEHOLDER]** Replace with `plot1_trajectories_ros.png`
> *(Trajectory: this EKF vs robot_localization vs odometry vs ground truth)*

> **[PLOT PLACEHOLDER]** Replace with `plot2_error_over_time_ros.png`
> *(Per-step error for all three methods — yellow band = no-landmark zone steps 184–296)*

> **[PLOT PLACEHOLDER]** Replace with `plot3_rmse_bar_ros.png`
> *(RMSE bar chart: odometry / robot_localization / this EKF)*

`robot_localization` achieves 0.0513 m vs raw odometry at 0.1303 m even without landmark corrections because its internal process model smooths velocity noise. However, without an external absolute reference it cannot correct accumulated drift. This project's filter uses the same odometry input but additionally corrects against five known landmarks, giving it a **75% accuracy advantage** over `robot_localization` in this scenario.

### 4.8 Covariance Dynamics

> **[PLOT PLACEHOLDER]** Replace with `plot4_covariance_ros.png`
> *(σ_xx over 300 steps from live ROS log — generated by `rl_collect_and_compare.py --compare`)*

Key phases visible in the logged data:

| Phase | Steps | Event | σ_xx |
|-------|-------|-------|------|
| Initial corrections | 1–23 | Only L0 visible | 0.100 → 0.017 |
| Second landmark | 25 | L1 enters range (two landmarks) | Sharp drop |
| Minimum variance | 184 | Last landmark update | ~0.001 |
| Dead-reckoning | 184–296 | No landmarks in range | 0.001 → 0.027 |
| Re-correction | 297–300 | L0 re-enters range | Drops to 0.008 |

Each update reduces σ_xx by roughly 2–4%, with larger reductions early when prior uncertainty is high. The expand-during-predict / contract-during-update pattern is the hallmark of a correctly implemented EKF.

---

## 5. Conclusion

This project demonstrates end-to-end implementation of EKF-based robot localization, from first-principles Jacobian derivation through to a live ROS 2 system visualized in RViz and benchmarked against a production package. The key findings are:

- The hand-written filter, using only NumPy matrix operations, runs in real time at 10 Hz inside Docker without any external localization library.
- Sensor fusion with just five landmarks reduces position RMSE by up to **70%** relative to raw odometry in the moderate-noise scenario.
- The landmark-fused EKF achieves **0.0126 m RMSE** versus **0.0513 m** for `robot_localization` (odometry-only) on identical data — a **75% improvement**, confirming the value of external landmark corrections.
- Filter performance is not uniform along the trajectory: the EKF degrades gracefully to a dead-reckoning predictor in landmark-sparse regions and self-corrects once landmarks re-enter range.
- The bearing-angle normalization detail — wrapping the innovation to (−π, π] — proved essential; without it the filter diverged whenever the robot's heading crossed the ±π boundary.

**Future work.** The most immediate extension is handling *unknown data association*: in this work each observation already carries a landmark ID, which is unrealistic in practice. Adding the nearest-neighbour or maximum-likelihood correspondence step would bring the system closer to a deployable localization stack. A further step would be to drop the known-map assumption entirely and implement Simultaneous Localization and Mapping (SLAM), either via an EKF-SLAM formulation or a factor-graph approach such as GTSAM.

---

## Appendix — Repository Structure

Full source code at https://github.com/16srivarshitha/ekf_localisation_project

```
ekf_localisation_project/
├── docs/
│   ├── SIMULATION.md
│   ├── RVIZ.md
│   ├── RESULTS.md
│   └── COMPARISON.md
├── ros2_ws/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── src/ekf_package/ekf_package/
│       ├── ekf.py
│       ├── ekf_node.py
│       ├── simulation_publisher.py
│       └── rl_collect_and_compare.py
└── analysis/
    ├── ekf.py
    ├── simulation.py
    ├── ekf_all_scenarios_plot.py
    ├── compare_rmse.py
    └── ekf_gif.py
```

---

## References

1. S. Thrun, W. Burgard, D. Fox — *Probabilistic Robotics*, MIT Press, 2005
2. R. Siegwart, I. R. Nourbakhsh — *Introduction to Autonomous Mobile Robots*, 2nd ed., MIT Press, 2011
3. T. Moore, D. Stouch — "A Generalized Extended Kalman Filter Implementation for the Robot Operating System," *Proc. IAS-13*, 2014
4. Open Robotics — ROS 2 Humble Documentation, https://docs.ros.org/en/humble/
5. A. Haber — "Mobile Robot Localization with Known Landmark Marker Locations: EKF Solution," https://aleksandarhaber.com
6. A. J. Kramer — "Introduction to the EKF — Step 1," https://andrewjkramer.net/intro-to-the-ekf-step-1/
7. University of Washington CSE 571 — "Assignment 2: EKF Localization," Spring 2025