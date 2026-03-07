import numpy as np

class EKFLocalizer:
    def __init__(self, initial_state, initial_covariance, control_noise_cov, measurement_noise_cov):
        self.mu = initial_state.reshape(3, 1)
        self.Sigma = initial_covariance
        self.M = control_noise_cov
        self.R = measurement_noise_cov

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def predict(self, u, dt):
        v, omega, theta = u[0], u[1], self.mu[2, 0]

        # State Prediction
        self.mu[0, 0] += v * dt * np.cos(theta)
        self.mu[1, 0] += v * dt * np.sin(theta)
        self.mu[2, 0] = self.normalize_angle(self.mu[2, 0] + omega * dt)

        # Jacobians
        G_t = np.array([
            [1.0, 0.0, -v * dt * np.sin(theta)],
            [0.0, 1.0,  v * dt * np.cos(theta)],
            [0.0, 0.0,  1.0]
        ])
        V_t = np.array([
            [dt * np.cos(theta), 0.0],
            [dt * np.sin(theta), 0.0],
            [0.0,                dt ]
        ])

        # Covariance Prediction
        self.Sigma = G_t @ self.Sigma @ G_t.T + V_t @ self.M @ V_t.T
        return self.mu, self.Sigma

    def update(self, z, landmark_pos):
        m_x, m_y = landmark_pos
        x, y, theta = self.mu[0, 0], self.mu[1, 0], self.mu[2, 0]

        # Distance to landmark
        dx = m_x - x
        dy = m_y - y
        q = dx**2 + dy**2
        sqrt_q = np.sqrt(q)

        # 1. Predicted Measurement (h function)
        z_pred_range = sqrt_q
        z_pred_bearing = self.normalize_angle(np.arctan2(dy, dx) - theta)
        z_pred = np.array([[z_pred_range], [z_pred_bearing]])

        # Actual Measurement from sensor
        z_actual = np.array([[z[0]], [z[1]]])

        # Measurement residual (Innovation: Actual - Predicted)
        y_residual = z_actual - z_pred
        y_residual[1, 0] = self.normalize_angle(y_residual[1, 0]) # Keep angles stable

        # 2. Measurement Jacobian (H_t)
        H_t = np.array([
            [-dx / sqrt_q, -dy / sqrt_q,  0.0],
            [ dy / q,      -dx / q,      -1.0]
        ])

        # 3. Kalman Gain (K_t) -> The magic trust factor
        S = H_t @ self.Sigma @ H_t.T + self.R
        K_t = self.Sigma @ H_t.T @ np.linalg.inv(S)

        # 4. State and Covariance Update
        self.mu = self.mu + K_t @ y_residual
        self.mu[2, 0] = self.normalize_angle(self.mu[2, 0])
        
        I = np.eye(3)
        self.Sigma = (I - K_t @ H_t) @ self.Sigma

        return self.mu, self.Sigma