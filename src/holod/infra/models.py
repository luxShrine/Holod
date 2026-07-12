from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: change new in enums/text to physics based CNN


def physics_informed_loss(loss_fn: Any, prediction: float, label: float):
    # TODO: implement basic idea of being aware of the physics:
    # calculate the reconstruction of the prediction depth
    # calculate the reconstruction of the true depth
    # calculate metric, what metric, focus, tamara coeff?

    # H(fx,fy) = exp(i k z) * exp(-i π λ z (fx^2 + fy^2))
    prediction_recon = np.random.random((5, 5))
    label_recon = np.random.random((5, 5))
    phy_loss = np.sqrt((prediction_recon**2) - (label_recon**2))
    num_loss = loss_fn(prediction, label)
    return phy_loss + num_loss


class NeuralNetwork(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        # conv2d (ks=3, s=1) -> (N, OC, h - 2, w - 2)
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 128, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # avg pool to 1x1, allowing for size agnostic images passed in
        self.gap = nn.AdaptiveAvgPool2d((3, 3))  # Any -> (3, 3)
        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # -> (N, 64, h-2, w-2)
        x = F.relu(self.conv2(x))  # -> (N, 128, h-4, w-4)
        x = F.relu(self.conv3(x))  # -> (N, 256, h-2, w-2)
        x = F.relu(self.conv4(x))  # -> (N, 128, h-4, w-4)
        x = self.pool(x)  # -> (N, 128, (h-4)/2, (h-4)/2)
        x = self.dropout1(x)
        x = self.gap(x)  # -> (N, 128, 3, 3)
        x = torch.flatten(x, 1)  #  -> (N, 128 * 3 * 3)
        x = F.relu(self.fc1(x))  # -> (N, 128)
        x = self.dropout2(x)
        x = self.fc2(x)  # -> (N, num_classes)
        return F.log_softmax(x, dim=1)


# ---------- focusnet torch implementation ----------


@torch.no_grad()
def fft_mag2d(x: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Compute |FFT2| with centered spectrum, per-sample, per-channel.

    Args:
        x: (N, C, H, W) real input (hologram).
        eps: small bias for numerical stability.

    Returns:
        (N, C, H, W) nonnegative magnitude.

    """
    # FFT expects complex; cast real -> complex
    x_c = torch.view_as_complex(torch.stack((x, torch.zeros_like(x)), dim=-1))
    f = torch.fft.fft2(x_c, dim=(-2, -1), norm="backward")
    f = torch.fft.fftshift(f, dim=(-2, -1))
    mag = torch.abs(f)
    if eps:
        mag = mag + eps
    return mag.real  # still real


def make_input_2ch(holod: torch.Tensor, use_fft: bool = True) -> torch.Tensor:
    """Build the 2‑channel input described in the repo's Fourier2D layer.

    Args:
        holod: (N, 1, H, W) real hologram.

    """
    assert holod.ndim == 4 and holod.size(1) == 1, "expect (N,1,H,W)"
    if use_fft:
        mag = fft_mag2d(holod)
        x = torch.cat([holod, mag], dim=1)  # -> (N, 2, H, W)
    else:
        x = torch.cat([holod, holod], dim=1)
    return x


# ---------- Model ----------


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        p = (k // 2) if p is None else p
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FocusNetTorch(nn.Module):
    """Compact CNN for DLHM depth, inspired by the TF/Keras FocusNET repo.

    - Input: (N, 1, H, W) hologram; expanded to 2 channels [holod, |FFT(holod)|] internally.
    - Output: (N, num_classes) logits for classification, or (N, 1) normalized depth
      for regression (num_classes=1), matching the other backbones.
    """

    def __init__(
        self,
        num_classes: int = 1,
        width: int = 32,
        use_fft: bool = True,
        head: Literal["mlp", "linear"] = "mlp",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_fft = use_fft

        # Feature extractor; stem sees the 2-channel input built in forward()
        self.stem = ConvBlock(2, width, k=3)
        self.layer1 = nn.Sequential(
            ConvBlock(width, width, k=3),
            ConvBlock(width, width, k=3),
        )
        self.down1 = ConvBlock(width, width * 2, k=3, s=2)  # 128x128
        self.layer2 = nn.Sequential(
            ConvBlock(width * 2, width * 2, k=3),
            ConvBlock(width * 2, width * 2, k=3),
        )
        self.down2 = ConvBlock(width * 2, width * 4, k=3, s=2)  # 64x64
        self.layer3 = nn.Sequential(
            ConvBlock(width * 4, width * 4, k=3),
            ConvBlock(width * 4, width * 4, k=3),
        )

        # Global pooling + head
        self.pool = nn.AdaptiveAvgPool2d(1)
        emb_dim = width * 4
        if head == "mlp":
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, emb_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(emb_dim // 2, num_classes),
            )
        else:
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(emb_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network on a (N, 1, H, W) real hologram.

        Returns:
            (N, num_classes) raw logits (classification) or values (regression).

        """
        x = make_input_2ch(x, use_fft=self.use_fft)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x)
