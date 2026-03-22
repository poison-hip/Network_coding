import torch
import torch.nn as nn
import math

class DDPM(nn.Module):
    def __init__(
        self,
        timestep : int,
        esp_model : nn.Module,
    ):
        super().__init__()
        self.timestep = timestep
        self.esp_model = esp_model
        self.belt = torch.linspace(1e-4, 0.01, timestep)
        self.alpha = 1 - self.belt
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.register_buffer("alpha", self.alpha)
        self.register_buffer("belt", self.belt)
        self.register_buffer("alpha_bar", self.alpha_bar)
        
    def q_sample(self, x_0:torch.Tensor, t:torch.Tensor, esp):
        mean = torch.sqrt(self.alpha_bar[t])[:,None,None,None] * x_0
        var = torch.sqrt(1 - self.alpha_bar[t])[:,None,None,None] * esp
        return mean + var
    
    def p_sample(self, x_t:torch.Tensor, t:torch.Tensor, z):
        esp_theta = self.esp_model(x_t, t)
        mean = torch.sqrt(1 / self.alpha[t])[:,None,None,None] * x_t
        val = torch.sqrt(1 / self.alpha[t]) * self.belt[t] /torch.sqrt( (1 - self.alpha_bar[t]))[:,None,None,None] * esp_theta
        return mean - val + torch.sqrt(self.belt[t]) * z