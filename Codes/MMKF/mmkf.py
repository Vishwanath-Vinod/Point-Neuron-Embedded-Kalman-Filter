import torch
import numpy as np

class KalmanFilter:
    def __init__(self, X_t, error_covariance, process_std, R, dt=1, model='linear'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.X_t = torch.tensor(X_t, dtype=torch.float32, device=self.device).clone().detach()
        self.F_t = torch.tensor(error_covariance, dtype=torch.float32, device=self.device).clone().detach()
        self.process_std = torch.tensor(process_std, dtype=torch.float32, device=self.device).clone().detach()
        self.R = torch.tensor(R, dtype=torch.float32, device=self.device)
        self.dt = dt

        # Infer dimensions
        self.state_dim = 9 # [x,y,z,vx,vy,vz,ax,ay,az]
        self.meas_dim = 3  # Only position observed

    def predict(self):
        A = torch.eye(self.state_dim, device=self.device)
        B = torch.zeros((self.state_dim, 3), device=self.device)

        if self.model == 'stationary':
            # State: [x,y,z, 0, 0, 0, 0, 0, 0]
            for j in range(3):
                B[j, j] = self.dt

        elif self.model == 'linear':
            # State: [x,y,z,vx,vy,vz, 0, 0, 0]
            for j in range(3):
                A[j, 3+j] = self.dt
            for j in range(3):
                B[j, j] = 0.5 * self.dt**2
                B[3+j, j] = self.dt

        elif self.model == 'accelerated':
            # State: [x,y,z,vx,vy,vz,ax,ay,az]
            for j in range(3):
                A[j, 3+j] = self.dt
                A[j, 6+j] = 0.5 * self.dt**2
                A[3+j, 6+j] = self.dt
            for j in range(3):
                B[j, j] = (1/6) * self.dt**3
                B[3+j, j] = 0.5 * self.dt**2
                B[6+j, j] = self.dt

        # KF prediction
        Q = torch.diag(self.process_std**2).to(self.device)
        Q_t = B @ Q @ B.T
        Xhat_t = A @ self.X_t
        Fhat_t = A @ self.F_t @ A.T + Q_t
        return Xhat_t, Fhat_t

    def compute_kalman_gain(self, Fhat_t):
        H_t = torch.zeros((self.meas_dim, self.state_dim), device=self.device)
        H_t[:, :self.meas_dim] = torch.eye(self.meas_dim, device=self.device)

        S = H_t @ Fhat_t @ H_t.T + self.R
        K_t = Fhat_t @ H_t.T @ torch.linalg.inv(S)

        I = torch.eye(self.state_dim, device=self.device)
        self.F_t = (I - K_t @ H_t) @ Fhat_t
        return K_t, H_t, S

    def update(self, Xhat_t, Fhat_t, Z_t):
        Z_t = torch.tensor(Z_t, dtype=torch.float32, device=self.device).flatten()

        K_t, H_t, S = self.compute_kalman_gain(Fhat_t)
        innovation = Z_t - H_t @ Xhat_t
        self.X_t = Xhat_t + K_t @ innovation
        self.F_t = Fhat_t - K_t @ H_t @ Fhat_t
        return self.X_t, self.F_t, innovation, S


class MultipleModelKalmanFilter:
    def __init__(self, models):
        """
        models: list of KalmanFilter instances
        mu_init: initial model probabilities
        """
        self.models = models
        self.M = len(models)
        self.device = models[0].device
        self.mu = torch.ones(self.M, device=self.device) / self.M

    def step(self, Z_t):
        log_likelihoods = torch.zeros(self.M, device=self.device)
        for i, kf in enumerate(self.models):
            Xhat_t, Fhat_t = kf.predict()
            kf.X_t, kf.F_t, innovation, S = kf.update(Xhat_t, Fhat_t, Z_t)
            if kf.model != 'stationary':
                velocity_update = (kf.X_t[:3] - Xhat_t[:3]) / kf.dt
                kf.X_t[3:6] = velocity_update
            if kf.model == 'accelerated':
                acc_update = (kf.X_t[3:6] - Xhat_t[3:6]) / kf.dt
                kf.X_t[6:9] = acc_update
            invS = torch.linalg.inv(S)
            maha = innovation.T @ invS @ innovation  # Mahalanobis distance
            log_det = torch.logdet(S + 1e-9*torch.eye(S.size(0), device=self.device))
            log_likelihoods[i] = -0.5 * (maha + log_det + kf.meas_dim * torch.log(torch.tensor(2.0 * np.pi, device=self.device)))

        # Update probabilities in log-space
        log_mu = torch.log(self.mu + 1e-12) + log_likelihoods
        self.mu = torch.softmax(log_mu, dim=0)
        #print("PROBS: ", self.mu)

        # Merge states (IMM combination)
        X_comb = torch.zeros_like(self.models[0].X_t)
        for i, kf in enumerate(self.models):
            X_comb += self.mu[i] * kf.X_t
            #print(f"MIXING {kf.X_t} probs: {self.mu[i]} final: {X_comb}")

        return X_comb, self.mu
