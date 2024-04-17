import numpy as np 
import torch.nn  as nn
import soundfile as sf 
import torch
import torch.nn.functional as f

class contrastive_loss(nn.Module):
    def __init__(self, tau=1, normalize=True):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize
    
    def forward(self, xi, xj):
        x = torch.cat((xi, xj), dim=0)
        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)
        sim_mat = torch.exp(sim_mat / self.tau)

        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:    
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)
        sim_match = torch.cat((sim_match, sim_match), dim=0)
        norm_sum =torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match /(torch.sum(sim_mat, dim=-1) - norm_sum)))
        return loss



class CLAP_loss(nn.Module):
    def __init__(self, tau=1, normalize=True):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize
    def forward(self, audio, text):
        audio_weight = (audio@audio.T).detach()
        text_weight =(text@text.T).detach()
        return loss 