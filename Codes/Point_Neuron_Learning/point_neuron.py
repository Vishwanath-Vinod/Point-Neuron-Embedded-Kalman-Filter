# Creation   : 28-05-2025
# Author: Vishwanath Vinod
# Description:
#       The function iteratively updates the locations
#       and weights of point neurons by gradient descent.
#
# Shape:
#       1) P: Number of Point Neurons (Virtual sources) determined by the InCoord and InWeight shape
#       2) Q: Number of Microphones given by the shape of MicField and MicCoord
#
# Inputs:
#       1) k        : The wavenumber.
#       2) MicField : The microphone measured sound field in the frequency 
#                     domain (Q-by-1).
#       3) MicCoord : the Cartesian coordinates of the microphone points 
#                     (Q-by-3).
#       4) InCoord  : the Cartesian coordinates of the initial neurons 
#                     (P-by-3).
#       5) InWeight : the initial weight of point neurons (P-by-1).
#       6) StepC    : Step size for updating point neuron location 
#                     in xyz (3-by-1).
#       7) StepW    : Step size for updating point neuron weights.
#       8) Lambda   : Model complexity penalty (L1 norm).
#       9) IterN    : No of iterations.
#
# 
# Outputs:
#       1) PnCoord  : The optimal Cartesian location of point neurons 
#                     (P-by-3).
#       2) PnWeight : The optimal weights of point neurons (P-by-3).
#       3) PnWscale : Scale parts of the mix-wave function (P-by-1).
#       4) Loss     : Loss of the network.


#########################################################################

import torch
import numpy as np
from tqdm import tqdm

class PointNeuron:
    def __init__(self, k, MicField, MicCoord, InCoord, InWeight, StepW, StepC, Lambda, IterN):
        '''
        Initializing the input parameters and other network settings similar to original code
        '''
        self.k = k
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MicField = torch.tensor(MicField, dtype=torch.cfloat,device=self.device)   #(Q,)
        self.MicCoord = torch.tensor(MicCoord, dtype=torch.float32,device=self.device)  #(Q,3)
        self.InCoord = torch.tensor(InCoord, dtype=torch.float32,device=self.device)
        self.InWeight = torch.tensor(InWeight, dtype=torch.cfloat,device=self.device)
        
        self.StepW = StepW
        self.StepC = StepC        # (3,) step sizes for x, y, z
        self.Lambda = Lambda
        self.IterN = IterN
        self.P = InCoord.shape[0]
        self.Q = MicCoord.shape[0]


        self.grad_b = 0.3
        self.loss_threshold = 5e-7
        self.optimal_loss = 2e7

    def calculate_loss(self,pn_coord,pn_weight,mic_pred,scale):
        '''
        Calculate the loss function given a set of locations and strengths for each microphone
        '''
        loss = torch.sum(torch.abs(mic_pred - self.MicField) ** 2) +self.Lambda *torch.sum(torch.abs(pn_weight))
        #print("Weight sum",self.Lambda * torch.sum(torch.abs(pn_weight)))
        if loss.detach().item() < self.optimal_loss:
            self.optimal_loss = loss.detach().item()
            self.best_coord = pn_coord.detach().clone()
            self.best_weight = pn_weight.detach().clone()
            self.best_scale = scale[:, 0].detach().clone()
        
        return loss
    
    def train_autograd(self):
        '''
        Updating bias and weights of each neuron via backpropagation.
        Gradients are calculated using torch.autograd.
        '''
        # Initialize base parameters without grad
        bias = self.InCoord.clone().detach().to(torch.float32).to(self.device)
        weight = self.InWeight.clone().detach().to(torch.cfloat).to(self.device)
        loss_list = []
        self.best_scale = torch.zeros(self.P, dtype=torch.cfloat, device=self.device)

        for i in tqdm(range(self.IterN)):
            # Create new tensors with grad enabled from current parameters
            pn_coord = bias.clone().detach().requires_grad_(True)
            pn_weight = weight.clone().detach().requires_grad_(True)
            diff = pn_coord[:, None, :] - self.MicCoord[None, :, :]  # (P, Q, 3)
            dist_sq = torch.sum(diff ** 2, dim=2)
            dist = torch.sqrt(dist_sq)  # (P, Q)
            #print("distance",torch.min(dist))
            hn = torch.exp(1j * self.k * dist) / ((dist) * 4 * np.pi)  # (P, Q)
            dist_source = torch.norm(pn_coord, dim=1, keepdim=True).repeat(1, self.Q)
            #print("distance source",torch.min(dist_source))
            scale = dist_source * torch.exp(-1j * self.k * dist_source)  # (P, Q)
            mic_pred = torch.sum(pn_weight[:, None] * hn * scale, dim=0)  # (Q,)
            loss = self.calculate_loss(pn_coord, pn_weight, mic_pred, scale)
            loss_list.append(loss.item())
            
            if i > 0 and i % 100 == 0:
                print(f"Loss in {i}th iteration is: {loss.item()}")

            # Backpropagation
            loss.backward()
            #torch.nn.utils.clip_grad_norm_([pn_coord, pn_weight], max_norm=10.0)
            with torch.no_grad():
                max_delta = 0.2 * self.grad_b
                for p in range(self.P):
                    # Update weights (complex)
                    grad_w = pn_weight.grad[p]
                    if grad_w is not None:
                        weight[p] -= self.StepW * grad_w
                    if p == 0:
                        continue  
                    # Update coords (real)
                    for j in range(3):
                        grad_c = pn_coord.grad[p, j]
                        if grad_c is not None:
                            delta = self.StepC[j] * grad_c.item()
                            if abs(delta) > max_delta:
                                delta = max_delta * delta / abs(delta)
                            bias[p, j] -= delta  
                    #if p==15:
                        #print(f"At time {i+1} location is {bias[p,:]}")
        return self.best_coord, self.best_weight, self.best_scale, loss_list
    
    def train_freeze_biases(self):
        '''
        Updating only weights of each neuron via backpropagation.
        Bias (coordinates) are frozen.
        Gradients are calculated using torch.autograd.
        '''
        # Freeze bias
        bias = self.InCoord.clone().detach().to(torch.float32).to(self.device)
        weight = self.InWeight.clone().detach().to(torch.cfloat).to(self.device)
        loss_list = []
        self.best_scale = torch.zeros(self.P, dtype=torch.cfloat, device=self.device)

        for i in tqdm(range(self.IterN)):
            # Only weights require gradients
            pn_coord = bias.clone()  # frozen, no grad
            pn_weight = weight.clone().detach().requires_grad_(True)

            # Compute distances and Green's function
            diff = pn_coord[:, None, :] - self.MicCoord[None, :, :]  # (P, Q, 3)
            dist = torch.norm(diff, dim=2)  # (P, Q)
            hn = torch.exp(1j * self.k * dist) / (dist * 4 * np.pi + 1e-8)
            dist_source = torch.norm(pn_coord, dim=1, keepdim=True).repeat(1, self.Q)
            scale = dist_source * torch.exp(-1j * self.k * dist_source)  # (P, Q)

            mic_pred = torch.sum(pn_weight[:, None] * hn * scale, dim=0)  # (Q,)
            loss = self.calculate_loss(pn_coord, pn_weight, mic_pred, scale)
            loss_list.append(loss.item())
            '''
            if i > 0 and i % 100 == 0:
                print(f"Loss at iteration {i}: {loss.item()}")
            '''
            # Backpropagation
            loss.backward()
            with torch.no_grad():
                for p in range(self.P):
                    grad_w = pn_weight.grad[p]
                    if grad_w is not None:
                        weight[p] -= self.StepW * grad_w  # Update weights only
            

        return self.best_coord, self.best_weight, self.best_scale, loss_list
    
    def train_update_first_n_bias(self, n=10):
        '''
        Updates:
            - The **first `n` neuron biases** (coordinates)
            - All neuron weights
        '''
        bias = self.InCoord.clone().detach().to(torch.float32).to(self.device)
        weight = self.InWeight.clone().detach().to(torch.cfloat).to(self.device)
        loss_list = []
        self.best_scale = torch.zeros(self.P, dtype=torch.cfloat, device=self.device)

        for i in tqdm(range(self.IterN)):
            pn_weight = weight.clone().detach().requires_grad_(True)

            # Use separate variable for the first `n` biases
            bias_n = bias[:n].detach().clone().requires_grad_(True)

            # Reconstruct full bias tensor with gradient tracking for first `n`
            pn_coord = bias.clone()
            pn_coord[:n] = bias_n  # First n are trainable

            diff = pn_coord[:, None, :] - self.MicCoord[None, :, :]  # (P, M, 3)
            dist = torch.norm(diff, dim=2)  # (P, M)
            hn = torch.exp(1j * self.k * dist) / (dist * 4 * np.pi + 1e-8)

            dist_source = torch.norm(pn_coord, dim=1, keepdim=True).repeat(1, self.Q)
            scale = dist_source * torch.exp(-1j * self.k * dist_source)

            mic_pred = torch.sum(pn_weight[:, None] * hn * scale, dim=0)
            loss = self.calculate_loss(pn_coord, pn_weight, mic_pred, scale)
            loss_list.append(loss.item())

            loss.backward()

            with torch.no_grad():
                max_delta = 0.2 * self.grad_b

                # Update weights
                for p in range(self.P):
                    grad_w = pn_weight.grad[p]
                    if grad_w is not None:
                        weight[p] -= self.StepW * grad_w

                # Update the first `n` bias vectors
                if bias_n.grad is not None:
                    for i_bias in range(n):
                        for j in range(3):
                            grad_c = bias_n.grad[i_bias, j].item()
                            delta = self.StepC[j] * grad_c
                            if abs(delta) > max_delta:
                                delta = max_delta * delta / abs(delta)
                            bias[i_bias, j] -= delta

            if i % 100 == 0:
                
                # Weighted average of first n neuron coordinates
                weights = pn_weight[:n].detach()
                coords = bias[:n].detach()
                # Take magnitude of complex weights
                weights_abs = torch.abs(weights)

                # Normalize weights
                weight_sum = weights_abs.sum()
                if weight_sum.item() > 0:
                    weights_normalized = weights_abs / weight_sum
                else:
                    weights_normalized = torch.ones_like(weights_abs) / len(weights_abs)

                # Weighted average
                avg_coord = (weights_normalized.view(-1, 1) * coords).sum(dim=0)
                coord_str = ', '.join([f'{c:.4f}' for c in avg_coord.tolist()])
                print(f"Iter {i}: Weighted Avg Coord[0:{n}] = [{coord_str}]")
                
                coords = bias[:n].detach()
                weights = pn_weight[:n].detach()
                # Use absolute value if weights are complex
                abs_weights = torch.abs(weights)

                # Find index of maximum weight
                max_idx = torch.argmax(abs_weights)

                # Get the corresponding coordinate
                max_coord = coords[max_idx]
                coord_str = ', '.join([f'{c:.4f}' for c in max_coord.tolist()])
                print(f"Iter {i}: Best Estimate of Source = [{coord_str}]")
                
        return self.best_coord, self.best_weight, self.best_scale, loss_list
    
    def pseudo_inverse(self):
        bias = self.InCoord.clone().detach().to(torch.float32).to(self.device)
        self.best_coord = bias.clone()
        self.best_scale = torch.zeros(self.P, dtype=torch.cfloat, device=self.device)
        loss_list = []

        pn_coord = bias.clone()  # fixed positions (P, 3)

        # Compute distances and Green's function
        diff = pn_coord[:, None, :] - self.MicCoord[None, :, :]  # (P, Q, 3)
        dist = torch.norm(diff, dim=2)  # (P, Q)
        hn = torch.exp(1j * self.k * dist) / (dist * 4 * np.pi + 1e-8)  # (P, Q)

        dist_source = torch.norm(pn_coord, dim=1, keepdim=True).repeat(1, self.Q)  # (P, Q)
        scale = dist_source * torch.exp(-1j * self.k * dist_source)  # (P, Q)
        self.best_scale = scale.clone()

        # Construct matrix H of shape (Q, P)
        H = (hn * scale).T  # (Q, P)
        # Solve for weights using pseudo-inverse
        H_pinv = torch.linalg.pinv(H)  # (P, Q)
        weight = H_pinv @ self.MicField  # (P,)
        self.best_weight = weight.detach()

        # Calculate loss (optional)
        mic_pred = H @ self.best_weight  # (Q,)
        loss = self.calculate_loss(pn_coord, self.best_weight, mic_pred, scale)
        loss_list.append(loss.item())

        return self.best_coord, self.best_weight, self.best_scale, loss_list
    
    def train_along_walls(self):
        '''
        Updating bias and weights of each neuron via backpropagation.
        Gradients are calculated using torch.autograd.
        Wall constraint: PNs move only along walls (no normal update).
        '''
        room_size = torch.tensor([3.2, 3.6, 2.2], device=self.device)

        # Initialize base parameters without grad
        bias = self.InCoord.clone().detach().to(torch.float32).to(self.device)
        weight = self.InWeight.clone().detach().to(torch.cfloat).to(self.device)
        loss_list = []
        self.best_scale = torch.zeros(self.P, dtype=torch.cfloat, device=self.device)

        for i in tqdm(range(self.IterN)):
            # Create new tensors with grad enabled
            pn_coord = bias.clone().detach().requires_grad_(True)
            pn_weight = weight.clone().detach().requires_grad_(True)

            diff = pn_coord[:, None, :] - self.MicCoord[None, :, :]  # (P, Q, 3)
            dist = torch.sqrt(torch.sum(diff ** 2, dim=2))           # (P, Q)
            #print("distance",torch.min(dist))
            hn = torch.exp(1j * self.k * dist) / (dist * 4 * np.pi+1e-8)  # (P, Q)
            #print("hn max", torch.max(abs(hn)))
            dist_source = torch.norm(pn_coord, dim=1, keepdim=True).repeat(1, self.Q)
            #print("distance source",torch.min(dist_source))
            scale = dist_source * torch.exp(-1j * self.k * dist_source)  # (P, Q)
            #print("Scale",torch.max(abs(scale)))
            mic_pred = torch.sum(pn_weight[:, None] * hn * scale, dim=0) # (Q,)

            # Loss
            loss = self.calculate_loss(pn_coord, pn_weight, mic_pred, scale)
            loss_list.append(loss.item())

            if i > 0 and i % 250 == 0:
                print(f"Loss in {i}th iteration is: {loss.item()}")

            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_([pn_coord, pn_weight], max_norm=1.0)
            with torch.no_grad():
                max_delta = 0.1 * self.grad_b
                margin = 0.05
                for p in range(self.P):
                    # Update weights (complex)
                    grad_w = pn_weight.grad[p]
                    if grad_w is not None:
                        weight[p] -= self.StepW * grad_w

                    # Update coords (real) – but skip first PN
                    if p == 0:
                        continue  

                    # Update only along tangential axes
                    for j in range(3):

                        grad_c = pn_coord.grad[p, j]
                        if grad_c is not None:
                            delta = self.StepC[j] * grad_c.item()
                            if abs(delta) > max_delta:
                                delta = max_delta * delta / abs(delta)
                            bias[p, j] -= delta
                        # If PN moved inside the room, push it back outside
                        if 0 + margin < bias[p, j] < room_size[j] - margin:
                            # Decide direction: push to nearest wall + margin
                            dist_to_low  = bias[p, j] - 0
                            dist_to_high = room_size[j] - bias[p, j]

                            if dist_to_low < dist_to_high:
                                bias[p, j] = -margin  # push outside left wall
                            else:
                                bias[p, j] = room_size[j] + margin  # push outside right wall
                            

        return self.best_coord, self.best_weight, self.best_scale, loss_list


