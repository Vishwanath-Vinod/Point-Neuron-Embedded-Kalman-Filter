import numpy as np
from point_neuron import PointNeuron

class OMP:

    @staticmethod
    def initialize(k, MicPrimaryField, MICCAR, InCoord, InWeight,IterN,freeze_bias=False):
        StepW = 1e-3
        StepC = [0.0005, 0.0005, 0.0005]
        Lambda = 0.005
        model = PointNeuron(k, MicPrimaryField, MICCAR, InCoord, InWeight, StepW, StepC, Lambda, IterN)

        if freeze_bias:
            PnCoord, PnWeight, PnWscale, Loss = model.train_freeze_biases()
        else:
            PnCoord, PnWeight, PnWscale, Loss = model.pseudo_inverse()

        # Convert tensors to numpy
        PnCoord = PnCoord.cpu().numpy()
        PnWeight = PnWeight.cpu().numpy()

        return PnCoord,PnWeight
    
    @staticmethod
    def localize(y, PnCoord_direct,MicCoord, k, sparsity=10, thres=0):
        """
        Solve sparse approximation using Orthogonal Matching Pursuit (OMP)
        for complex-valued signals. Returns a solution with sparsity = k.
        y(microphone measurement) = Psi(Normalized Green Function)*weight(weight of point neuron)
        M (Number of Microphones) M*N N (Number of locations on Candidate Grid) 
        """
        diff = PnCoord_direct[:, None, :] - MicCoord[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        hn = np.exp(1j * k * dist) / (dist * 4 * np.pi)
        dist_source = np.linalg.norm(PnCoord_direct, axis=1, keepdims=True)
        scale = dist_source * np.exp(-1j * k * dist_source)
        Psi = (hn * scale).T

        resid = y.copy()
        c = []
        Psi_edt = Psi.copy()
        ompErr = np.full(sparsity, np.nan, dtype=np.float64)

        for i in range(sparsity):
            ti = OMP.cmpxMutlIncoh(Psi_edt, resid)
            if ti not in c:
                c.append(ti)

            Psi_c = Psi[:, c]
            Pi = Psi_c @ np.linalg.pinv(Psi_c)
            resid = y - Pi @ y
            Psi_edt[:, c] = 0  # zero-out selected columns
            ompErr[i] = np.linalg.norm(resid)

            if ompErr[i] < thres:
                break

        x = np.zeros(Psi.shape[1], dtype=np.complex128)
        x_c = np.linalg.pinv(Psi[:, c]) @ y
        x[c] = x_c
        unique_indices = []
        threshold = 0.15  # meters (tune as per grid spacing)
        for idx in c:
            pos = PnCoord_direct[idx]
            if not any(np.linalg.norm(pos - PnCoord_direct[j]) < threshold for j in unique_indices):
                unique_indices.append(idx)
            if len(unique_indices) >= 3:  # only keep top 3 unique
                break
        return x, ompErr,unique_indices

    @staticmethod
    def cmpxMutlIncoh(Psi, Res):
        """
        Return index of the column in Psi most correlated with Res.
        """
        Psi_norm = OMP.colNorm(Psi)
        Res_norm = OMP.colNorm(Res.reshape(-1, 1)).flatten()
        incoh = np.abs(Psi_norm.conj().T @ Res_norm)
        return int(np.argmax(incoh))

    @staticmethod
    def colNorm(Matrix):
        """
        Normalize each column of the matrix.
        """
        norms = np.linalg.norm(Matrix, axis=0, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return Matrix / norms
    
    @staticmethod
    def dereverb(PnCoord_reverb,PnWeight_reverb,k,MicCoord,MicPrimaryField):

        diff = PnCoord_reverb[:, None, :] - MicCoord[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        hn = np.exp(1j * k * dist) / (dist * 4 * np.pi)
        dist_source = np.linalg.norm(PnCoord_reverb, axis=1, keepdims=True)
        scale = dist_source * np.exp(-1j * k * dist_source)
        mic_reverb = np.sum(PnWeight_reverb[:, None] * hn * scale, axis=0)
        MicDereverb = MicPrimaryField - mic_reverb

        return MicDereverb


