import torch
import numpy as np

class KalmanFilter:
    def __init__(self, X_t, error_covariance, acceleration_std, R, dt=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X_t = torch.tensor(X_t, dtype=torch.float32, device=self.device).clone().detach()  # [6,]
        self.F_t = torch.tensor(error_covariance, dtype=torch.float32, device=self.device).clone().detach()  # [6,6]
        self.acceleration_std = torch.tensor(acceleration_std, dtype=torch.float32, device=self.device).clone().detach()  # [3,]
        self.R = torch.tensor(R, dtype=torch.float32, device=self.device)  # [3,3]
        self.dt = dt

        self.state_dim = self.X_t.shape[0]
        self.meas_dim = 3  # Only position is observed

    def compute_Q(self, B):
        Q_a = torch.diag(self.acceleration_std ** 2).to(self.device)  # [3,3]
        return B @ Q_a @ B.T  # [6,6]

    def predict(self):
        A = torch.eye(self.state_dim, device=self.device)
        for j in range(3):
            A[j, 3 + j] = self.dt

        Xhat_t = A @ self.X_t

        B = torch.zeros((self.state_dim, 3), device=self.device)
        for j in range(3):
            B[j, j] = 0.5 * self.dt ** 2
            B[3 + j, j] = self.dt

        Q_t = self.compute_Q(B)
        Fhat_t = A @ self.F_t @ A.T + Q_t

        print("Predicted VELOCITY of source:", Xhat_t[3:])
        return Xhat_t, Fhat_t

    def compute_kalman_gain(self, Fhat_t):
        # Observation matrix: only selects the first 3 states (position)
        H_t = torch.zeros((self.meas_dim, self.state_dim), device=self.device)
        H_t[:, :self.meas_dim] = torch.eye(self.meas_dim, device=self.device)

        S = H_t @ Fhat_t @ H_t.T + self.R  # Innovation covariance
        K_t = Fhat_t @ H_t.T @ torch.linalg.inv(S)

        I = torch.eye(self.state_dim, device=self.device)
        self.F_t = (I - K_t @ H_t) @ Fhat_t
        return K_t, H_t

    def update(self, Xhat_t, Fhat_t, Z_t):
        # Z_t: measurement vector (position from SRP), shape [3,]
        Z_t = torch.tensor(Z_t, dtype=torch.float32, device=self.device).flatten()

        K_t, H_t = self.compute_kalman_gain(Fhat_t)
        innovation = Z_t - H_t @ Xhat_t  # Measurement residual

        delta = K_t @ innovation
        self.X_t = Xhat_t + delta

        return self.X_t, self.F_t
        

