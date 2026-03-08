import numpy as np


class EKFLocalizer:
    def __init__(self, initial_state, initial_covariance, control_noise_cov, measurement_noise_cov):
        self.mu    = initial_state.reshape(3, 1)
        self.Sigma = initial_covariance.copy()
        self.M     = control_noise_cov          # 2*2  process noise (velocity space)
        self.R     = measurement_noise_cov      # 2*2  measurement noise

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def predict(self, u, dt):
        # Capture theta at linearisation point (BEFORE state update)
        v, omega, theta = u[0], u[1], self.mu[2, 0]

        # State prediction — unicycle model (eq. 3-5)
        self.mu[0, 0] += v * dt * np.cos(theta)
        self.mu[1, 0] += v * dt * np.sin(theta)
        self.mu[2, 0]  = self.normalize_angle(self.mu[2, 0] + omega * dt)

        # State Jacobian G_t  (eq. 6)
        G_t = np.array([
            [1.0, 0.0, -v * dt * np.sin(theta)],
            [0.0, 1.0,  v * dt * np.cos(theta)],
            [0.0, 0.0,  1.0]
        ])

        # Control Jacobian V_t  (eq. 7)
        V_t = np.array([
            [dt * np.cos(theta), 0.0],
            [dt * np.sin(theta), 0.0],
            [0.0,                dt ]
        ])

        # Covariance prediction  (eq. 8)
        self.Sigma = G_t @ self.Sigma @ G_t.T + V_t @ self.M @ V_t.T
        return self.mu, self.Sigma

    def update(self, z, landmark_pos):
        m_x, m_y = landmark_pos
        x, y, theta = self.mu[0, 0], self.mu[1, 0], self.mu[2, 0]

        dx = m_x - x
        dy = m_y - y
        q  = dx**2 + dy**2
        sqrt_q = np.sqrt(q)

        # Predicted measurement h(x)  (eq. 9-10)
        z_pred = np.array([
            [sqrt_q],
            [self.normalize_angle(np.arctan2(dy, dx) - theta)]
        ])

        # Innovation — bearing wrapped to (-pi, pi]  (Section 2.4)
        y_res = np.array([[z[0]], [z[1]]]) - z_pred
        y_res[1, 0] = self.normalize_angle(y_res[1, 0])

        # Measurement Jacobian H_t  (eq. 11)
        H_t = np.array([
            [-dx / sqrt_q, -dy / sqrt_q,  0.0],
            [ dy / q,      -dx / q,       -1.0]
        ])

        # Kalman gain  (Step 3)
        S   = H_t @ self.Sigma @ H_t.T + self.R
        K_t = self.Sigma @ H_t.T @ np.linalg.inv(S)

        # State and covariance update  (Steps 4-5)
        self.mu       = self.mu + K_t @ y_res
        self.mu[2, 0] = self.normalize_angle(self.mu[2, 0])
        self.Sigma    = (np.eye(3) - K_t @ H_t) @ self.Sigma

        return self.mu, self.Sigma