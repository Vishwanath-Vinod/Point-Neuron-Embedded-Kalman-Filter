import torch
import numpy as np
from point_neuron import PointNeuron

class PNEKF:
    def __init__(self, X_t, MicCoord, error_covariance, acceleration_std, R, PnWeight,
                 Frequency=900, source_pns_num=1, wall_pns_num=68, dt=1):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X_t = torch.tensor(X_t, dtype=torch.float32, device=self.device).clone().detach()
        self.F_t = torch.tensor(error_covariance, dtype=torch.float32, device=self.device).clone().detach()
        self.acceleration_std = torch.tensor(acceleration_std, dtype=torch.float32, device=self.device).clone().detach()
        self.MicCoord = torch.tensor(MicCoord, dtype=torch.float32, device=self.device)
        self.pn_weight = torch.tensor(PnWeight, dtype=torch.cfloat, device=self.device)
        self.R = torch.tensor(R, dtype=torch.float32, device=self.device)

        self.state_dim = self.X_t.shape[0]
        self.num_mics = self.MicCoord.shape[0]

        self.dt = dt
        self.source_pns_num = source_pns_num
        self.wall_pns_num = wall_pns_num
        self.total_pns = source_pns_num + wall_pns_num

        c = 343
        self.k = 2 * np.pi * Frequency / c

    def compute_Q(self, B):
        """Process noise covariance Q = B Q_a B^T, where Q_a = diag(acceleration_std^2)"""
        Q_a = torch.diag(self.acceleration_std**2).to(self.device)  # [3, 3]
        Q = B @ Q_a @ B.T  # [state_dim, state_dim]
        return Q

    def predict(self):
        """EKF Predict step with motion model for circular PNs and velocity"""
        A = torch.eye(self.state_dim, device=self.device)
        for i in range(self.source_pns_num):
            for j in range(3):  # x, y, z
                A[3 * i + j, 3*self.source_pns_num +3*self.wall_pns_num + 3*i + j] = self.dt
        Xhat_t = A @ self.X_t

        B = torch.zeros((self.state_dim, 3), device=self.device)
        for i in range(self.source_pns_num):
            for j in range(3):
                B[3 * i + j, j] = 0.5 * self.dt ** 2
        B[-3:, :] = torch.eye(3, device=self.device) * self.dt

        Q_t = self.compute_Q(B)
        Fhat_t = A @ self.F_t @ A.T + Q_t
        print("VELOCITY OF 3 SOURCEs: ",Xhat_t[-3*self.source_pns_num:])
        return Xhat_t, Fhat_t

    def compute_jacobian(self):
        """Compute Jacobian of Re[z_pred] and Im[z_pred] w.r.t. x"""
        x = self.X_t.clone().detach().requires_grad_(True)  # shape: [3P]
        pn_coord = x[:-3*self.source_pns_num].view(self.total_pns, 3) 

        diff = pn_coord[:, None, :] - self.MicCoord[None, :, :]  # (P, Q, 3)
        dist_sq = torch.sum(diff ** 2, dim=2)
        dist = torch.sqrt(dist_sq)  # (P, Q)

        hn = torch.exp(1j * self.k * dist) / (dist * 4 * np.pi)  # (P, Q)
        dist_source = torch.norm(pn_coord, dim=1, keepdim=True).repeat(1, self.num_mics)
        scale = dist_source * torch.exp(-1j * self.k * dist_source)  # (P, Q)

        z_pred = torch.sum(self.pn_weight[:, None] * hn * scale, dim=0)  # (Q,)

        H_real = torch.zeros((self.num_mics, self.state_dim), device=self.device)
        H_imag = torch.zeros((self.num_mics, self.state_dim), device=self.device)

        for i in range(self.num_mics):
            if x.grad is not None:
                x.grad.zero_()
            z_pred[i].real.backward(retain_graph=True)
            H_real[i] = x.grad.detach().clone()
            x.grad.zero_()
            z_pred[i].imag.backward(retain_graph=True)
            H_imag[i] = x.grad.detach().clone()

        H_t = torch.cat([H_real, H_imag], dim=0)# [2Q, state_dim]
        ht_norm = torch.norm(H_t, p='fro')
        return H_t,z_pred

    def compute_kalman_gain(self, Fhat_t, H_t):
        """Kalman gain: K = Fhat_t H^T (H Fhat_t H^T + R)^-1"""
        S = H_t @ Fhat_t @ H_t.T + self.R
        K_t = Fhat_t @ H_t.T @ torch.linalg.inv(S)
        return K_t

    def update(self, Xhat_t, Fhat_t, Z_t):
        """EKF update step"""
        Z_t = torch.tensor(Z_t, dtype=torch.cfloat, device=self.device)
        Z_combined = torch.cat([Z_t.real, Z_t.imag]).to(self.device)
        H_t,z_pred = self.compute_jacobian()
        ht_norm = torch.norm(H_t, p='fro')
        Z_hat = torch.cat([z_pred.real, z_pred.imag])
        #Z_hat = H_t @ Xhat_t
        K_t = self.compute_kalman_gain(Fhat_t, H_t)
        K_t[-3*self.source_pns_num:, :] = 0  
        error = (Z_combined - Z_hat)
        delta = K_t @ error
        self.X_t = (Xhat_t + delta).detach()
        I = torch.eye(self.state_dim, device=self.device)
        self.F_t = ((I - K_t @ H_t) @ Fhat_t).detach()

        return self.X_t, self.F_t
    
    def update_iekf(self, Xhat_t, Fhat_t, Z_t, max_iter=1,tol=1e-6):
        """Iterated EKF update step"""
        Z_t = torch.tensor(Z_t, dtype=torch.cfloat, device=self.device)
        Z_combined = torch.cat([Z_t.real, Z_t.imag]).to(self.device)

        # Initial guess: use predicted state
        x_iter = Xhat_t.clone().detach()

        for _ in range(max_iter):
            # Linearize around current estimate
            self.X_t = x_iter.clone().detach()
            H_t, z_pred = self.compute_jacobian()  # must depend on self.X_t
            Z_hat = torch.cat([z_pred.real, z_pred.imag])
            error = Z_combined - Z_hat

            # Kalman gain
            K_t = self.compute_kalman_gain(Fhat_t, H_t)
            K_t[-3*self.source_pns_num:, :] = 0  # don’t update velocity via measurement
            # State correction
            delta = K_t @ error
            x_new = Xhat_t + delta

            # Convergence check
            if torch.norm(delta) < tol:
                break

            x_iter = x_new.clone().detach()

        # Final update
        self.X_t = x_iter
        I = torch.eye(self.state_dim, device=self.device)
        self.F_t = ((I - K_t @ H_t) @ Fhat_t).detach()

        return self.X_t, self.F_t

    


