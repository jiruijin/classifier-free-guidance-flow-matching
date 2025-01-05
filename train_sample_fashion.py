
"""
This scipt does a classification-free-guidance flow matching model training on FashionMNIST dataset.
The used probability path is conditional optimal-transport (linear) path.
Two ODE solver methods are available: 'midpoint' and 'euler'.

The code is modified from:
https://github.com/TeaPearce/Conditional_Diffusion_MNIST 
https://github.com/facebookresearch/flow_matching


"""


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.utils import save_image, make_grid
from torchvision import transforms
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

##### U-Net Block #####
class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414  # approximate scaling
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

##### Flow U-Net (Predict dx/dt) #####
class ContextFlow(nn.Module):
    """A 'flow matching' U-Net that predicts dx/dt(velocity) for images."""
    def __init__(self, in_channels, n_feat=256, n_classes=10):
        super(ContextFlow, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2*n_feat, 2*n_feat, 7, 7),
            nn.GroupNorm(8, 2*n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        # Output = in_channels, same shape as x
        self.out = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )


    def forward(self, x, t, c, context_mask):
        """
        x: (B, in_channels, H, W)
        t: (B, 1) or (B,1,1,1)   # scalar time
        c: (B,)                  # class labels
        context_mask: (B,)       # 0 or 1 for classifier-free dropout
        """
        # Prepare shapes
        if len(t.shape) == 1:
            t = t[:, None, None, None]  # (B,1,1,1)
        if len(c.shape) > 1:
            c = c.squeeze()

        # Initial conv
        x_0 = self.init_conv(x)    
        down1 = self.down1(x_0)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)  

        # one-hot
        c_onehot = F.one_hot(c, num_classes=self.n_classes).float()
        # mask out context if needed
        context_mask = context_mask[:, None].float()  # shape (B,1)
        context_mask = context_mask.repeat(1, self.n_classes)  # (B,n_classes)
        context_mask = (1 - context_mask) # flip 0->1, 1->0
        c_onehot = c_onehot * context_mask

        # embed
        cemb1 = self.contextembed1(c_onehot).view(-1, 2*self.n_feat, 1, 1)
        cemb2 = self.contextembed2(c_onehot).view(-1, self.n_feat, 1, 1)

        temb1 = self.timeembed1(t).view(-1, 2*self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)   

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2)
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x_0), 1))
        return out  # interpret as dx/dt      


##### Flow-Matching Trainer & Sampler #####
class FlowMatchingWrapper:
    """
    Wraps the ContextFlow model with:
    - a training function that does flow matching on (x_0, x_1) 
      with linear interpolation,
    - a sampling function that integrates from t=0 to t=1.
    """
    def __init__(self, model: nn.Module, device="cuda", drop_prob=0.1):
        self.model = model.to(device)
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
    
    def train_step(self,x_0, x_1, c):
        """
        x_0, x_1: (B, 1, 28, 28) - e.g. noise and real images
        c: (B,) class labels
        """
        B = x_0.shape[0]
        # sample random t in [0,1]
        t = torch.rand(B, device=self.device)
        
        # linear interpolation
        x_t = (1 - t.view(-1,1,1,1))*x_0 + t.view(-1,1,1,1)*x_1
        dx_true = x_1 - x_0  # shape (B, 1, 28, 28)

        # classifier-free dropout
        context_mask = torch.bernoulli(torch.zeros_like(c.float())+self.drop_prob).to(self.device)

        # predict dx/dt
        dx_pred = self.model(x_t, t, c, context_mask)
        # ground-truth dx/dt = (x_1 - x_0), independent of t
        # so MSE
        loss = self.loss_mse(dx_pred, dx_true)
        return loss
    
    def sample(self, n_sample, c, shape=(1,28,28), n_steps=20, method='midpoint', guide_w=0.0):
        """
        Integrate from t=0 -> t=1 in n_steps. 
        Start x(0) ~ noise (or any prior).
        c: shape (n_sample,) with labels
        context_mask=0 to keep conditioning at test time (no dropout).

        If guide_w=0.0 => standard unconditional pass
        If guide_w>0 => CFG: 
            flow_uncond = f(x, t, c=null_mask) 
            flow_cond   = f(x, t, c=label_mask)
            flow_guided = guide_w*flow_cond + (1-guide_w)*flow_uncond
        """
        x = torch.randn(n_sample, *shape).to(self.device)  # x_0
        t_vals = torch.linspace(0, 1, n_steps+1, device=self.device)

        # At test time, we always want the unconditional pass "mask=1" to skip label
        # as well as the conditional pass "mask=0"
        context_mask_cond = torch.zeros_like(c).to(self.device)
        context_mask_uncond = torch.ones_like(c).to(self.device)
        
        if method.lower() == 'midpoint':
            for i in range(n_steps):
                t_start, t_end = t_vals[i], t_vals[i+1]
                dt = t_end - t_start
                t_mid = t_start + 0.5*dt

                # Evaluate flows at t_start
                f_cond_start   = self.model(x, t_start, c, context_mask_cond)
                f_uncond_start = self.model(x, t_start, c, context_mask_uncond)
                flow_guided_start = guide_w * f_cond_start + (1-guide_w) * f_uncond_start

                # Midpoint
                x_mid = x + 0.5*dt*flow_guided_start

                # Evaluate flows at t_mid
                f_cond_mid   = self.model(x_mid, t_mid, c, context_mask_cond)
                f_uncond_mid = self.model(x_mid, t_mid, c, context_mask_uncond)
                # variant version of classifier free guidance
                flow_guided_mid = guide_w * f_cond_mid + (1-guide_w) * f_uncond_mid

                # Euler step
                x = x + dt*flow_guided_mid   

        elif method.lower() == 'euler':
            for i in range(n_steps):
                t_start, t_end = t_vals[i], t_vals[i+1]
                dt = t_end - t_start

                # Evaluate flows at t_start
                f_cond_start   = self.model(x, t_start, c, context_mask_cond)
                f_uncond_start = self.model(x, t_start, c, context_mask_uncond)
                flow_guided_start = guide_w * f_cond_start + (1-guide_w) * f_uncond_start

                # Euler step
                x = x + dt*flow_guided_start

        return x

def train_fashion_flow():
    os.makedirs('./data/fashion_flow_outputs', exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epoch = 300
    batch_size = 256
    n_classes = 10
    ode_solver = 'midpoint'  # or 'euler'

    # Guidance weight
    ws_test = [0.0, 1.0, 2.0]

    # Create the Flow model and wrapper
    flow_model = ContextFlow(in_channels=1, n_feat=128, n_classes=n_classes)
    flow = FlowMatchingWrapper(flow_model, device=device, drop_prob=0.1)

    # Data
    tf = transforms.Compose([transforms.ToTensor()])
    dataset = FashionMNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    optim = torch.optim.Adam(flow.model.parameters(), lr=1e-4)

    # Track best model by minimal training loss
    best_loss = float('inf')
    best_model_path = "./data/fashion_flow_outputs/best_model.pth"

    for ep in range(n_epoch):
        flow.model.train()
        losses = []
        for x, c in tqdm(dataloader):
            x = x.to(device)
            c = c.to(device)

            # x_1 = real images
            x_1 = x
            # x_0 = noise
            x_0 = torch.randn_like(x_1)

            loss = flow.train_step(x_0, x_1, c)
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()

        epoch_loss = sum(losses)/len(losses)
        print(f"[Epoch {ep}] Loss: {epoch_loss:.4f}")

        # Check if best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(flow.model.state_dict(), best_model_path)
            print(f"  => Saved best model (loss={best_loss:.4f}) at epoch {ep}")

        # Evaluate and save images at each epoch for different w
        flow.model.eval() 
        with torch.no_grad():
            # We'll generate 4 samples per class, for all 10 classes => total 40
            n_sample = n_classes*4
            # For c, let's cycle through classes
            c_batch = torch.arange(n_classes, device=device).repeat(int(n_sample/n_classes))

            for w in ws_test:
                x_gen = flow.sample(n_sample=n_sample, c=c_batch, shape=(1,28,28), n_steps=20, method=ode_solver, guide_w=w)
                # create grid
                grid = make_grid(x_gen, nrow=n_classes, normalize=True)
                save_path = f"./data/fashion_flow_outputs/flow_ep{ep}_w{w}.png"
                save_image(grid, save_path)
                print(f"Saved samples with w={w} at epoch {ep}")

if __name__ == "__main__":
    train_fashion_flow()