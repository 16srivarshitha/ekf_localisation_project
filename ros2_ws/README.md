# Extended Kalman Filter (EKF) Localization in ROS 2

##  Project Overview
This project implements an **Extended Kalman Filter (EKF)** from scratch to localize a robot in a 2D environment. The system fuses noisy **Odometry data** (Prediction Step) with **Landmark Observations** (Update Step) to estimate the robot's pose $(x, y, \theta)$ and covariance $\Sigma$.

**Key Features:**
*   **Custom EKF Math Engine:** Implemented using NumPy (Linearization via Jacobians).
*   **ROS 2 Node:** Subscribes to simulation data and publishes estimated poses.
*   **Velocity Calculation:** Derives linear ($v$) and angular ($\omega$) velocities from raw pose data.
*   **Data Integration:** seamless integration with external simulation scripts.

---

##  Project Structure
```text
/ros2_ws
│
├── src/
│   ├── ekf_package/           # The ROS 2 Package 
│   │   ├── ekf_package/
│   │   │   ├── ekf.py         # Core EKF Math Class
│   │   │   └── ekf_node.py    # ROS 2 Node Wrapper
│   │   ├── package.xml
│   │   └── setup.py
│   │
│   ├── simulation.py          # Data Generator 
│   └── simulation_publisher.py # Simulation ROS Publisher 
│
├── landmarks.npy              # Generated Map Data
└── Dockerfile                 # Environment Setup
```

---

##  Setup & Installation

### 1. Start the Environment
This project runs inside a Docker container with ROS 2 Humble.
```bash
# In the project root
docker-compose up -d
docker exec -it ekf_dev_container bash
```

### 2. Generate Simulation Data
Before running the node, we must generate the ground truth, odometry, and landmark maps.
```bash
# Inside the container
cd /ros2_ws/src
python3 simulation.py
```
*Output: Generates `.npy` files (landmarks, odometry, etc.) required for the EKF.*

### 3. Build the EKF Package
```bash
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
```

---

##  How to Run

You will need **two terminals** (both inside the Docker container).

### Terminal 1: Run the EKF Node
This node waits for data, computes the position, and publishes the result.
```bash
source install/setup.bash
ros2 run ekf_package ekf_node
```

### Terminal 2: Run the Simulation
This script plays back the pre-recorded noisy data to test the EKF.
```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws/src
python3 simulation_publisher.py
```

---

##  Results & Verification

The EKF successfully reduces uncertainty when landmarks are observed. Below is a log capture showing the **Covariance Matrix** (uncertainty) shrinking during an Update step.

**1. Prediction Phase (Drift):**
As the robot moves without seeing landmarks, uncertainty (variance in X) slowly increases:
```text
Variance X: 0.113...
Variance X: 0.115...
Variance X: 0.116...
```

**2. Update Phase (Correction):**
The moment a landmark is observed, the EKF corrects the position and uncertainty collapses:
```text
Variance X: 0.030...  <-- (75% Confidence Gain)
```

**Output Topic:**
To visualize the live data stream:
```bash
ros2 topic echo /ekf/estimated_pose
```

---

##  Technical Implementation Details

### 1. Motion Model (Prediction)
Since the simulation provided raw poses $(x, y, \theta)$ instead of velocities, the node calculates control inputs dynamically:
$$ v = \frac{\sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}}{dt} $$
$$ \omega = \frac{(\theta_t - \theta_{t-1})}{dt} $$

### 2. Observation Model (Update)
The filter calculates the Expected Measurement $(r, \phi)$ vs Actual Measurement to compute the **Kalman Gain ($K$)**:
$$ K_t = \bar{\Sigma}_t H_t^T (H_t \bar{\Sigma}_t H_t^T + R_t)^{-1} $$

### 3. Coordinate Frames
*   **Map Frame:** Global static frame for landmarks.
*   **Odom Frame:** Noisy accumulation frame.
*   **Output:** `PoseWithCovarianceStamped` published to `/ekf/estimated_pose`.

---
