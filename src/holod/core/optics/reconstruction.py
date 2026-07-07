from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2

from holod.infra.dataclasses import AutoConfig
from holod.infra.log import get_logger
from holod.infra.util.image_processing import crop_max_square
from holod.infra.util.types import (
    SENSOR_PIXEL_PITCH_M,
    AnalysisType,
    Arr32,
    ModelType,
    ReconstructionMethod,
)

if TYPE_CHECKING:
    from PIL.Image import Image as ImageType

__all__ = [
    "dlhm_effective_z_mm",
    "focus_score",
    "gradient_tamura",
    "recon_inline",
    "tamura_coefficient",
    "torch_recon",
]
logger = get_logger(__name__)
i = 1j  # just for my sanity


def torch_recon(
    img_file_path: str,
    wavelength: float,
    ckpt_file: str,
    crop_size: int = 512,
    dx: float = SENSOR_PIXEL_PITCH_M,
):
    """Reconstruct amplitude/phase from a hologram image using a depth predicted by a model."""
    pil_image: ImageType = Image.open(img_file_path).convert("RGB")
    pil_image_crop = crop_max_square(pil_image)
    np.asarray(crop_max_square(pil_image))

    # build architecture + load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt: dict[str, Any] = torch.load(ckpt_file, map_location=device, weights_only=False)

    ## WARN: Fragile loading process...
    bin_centers = ckpt["bin_centers"]
    # z normalization stats; absent on classification and pre-existing checkpoints
    z_avg = ckpt.get("z_avg")
    z_std = ckpt.get("z_std")
    cfg = AutoConfig(
        analysis=AnalysisType.REG if bin_centers is None else AnalysisType.CLASS,
        backbone=ckpt["model_type"],
        num_classes=ckpt["num_classes"],
    )

    model = cfg.create_model().to(device)
    incomp = model.load_state_dict(ckpt["model_state_dict"])
    if getattr(incomp, "missing_keys", None) or getattr(incomp, "unexpected_keys", None):
        logger.warning(
            "State dict load: missing=%s unexpected=%s",
            getattr(incomp, "missing_keys", []),
            getattr(incomp, "unexpected_keys", []),
        )

    model.eval()

    # TODO: create wrapper around models such that this repeated functionalility from
    # TransformedDataset compressed
    rgb_models = [ModelType.ENET, ModelType.RESNET, ModelType.VIT]

    transforms: list[nn.Module] = [
        # convert PIL to tensor
        v2.PILToTensor(),
        # ToTensor preserves original datatype, this ensures it is proper input type
        v2.ToDtype(torch.uint8, scale=True),
        v2.CenterCrop(size=crop_size),
        # normalize across channels, expects float
        v2.ToDtype(torch.float32, scale=True),
    ]
    if cfg.backbone in rgb_models:
        transforms.extend(
            [
                v2.Grayscale(num_output_channels=3),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    # load & preprocess image
    # shape [batch=1, C, H, W]
    input_image_tens = v2.Compose(transforms)(pil_image_crop).unsqueeze(0).to(device)

    logger.info("Creating prediction...")
    # prediction
    with torch.no_grad():
        pred = model(input_image_tens)  # shape [Batch, Classes]

        if cfg.analysis == AnalysisType.CLASS:
            # rescales elements for range [0,1] & sum to one
            probs = torch.softmax(pred, dim=1)
            # discrete estimate
            index = int(probs.argmax(1).item())  # ensure integer type
            centers = np.asarray(bin_centers, dtype=np.float32)  # retrive that depth in mm
            z_expect = float(centers[index])
        elif cfg.analysis == AnalysisType.REG:
            z_expect = float(
                pred.squeeze()  # convert to scalar first
            )
            if z_avg is not None and z_std is not None:
                # reverse the label normalization applied during training
                z_expect = z_expect * z_std + z_avg
            else:
                logger.warning(
                    "Checkpoint lacks z_avg/z_std normalization stats; "
                    + "regression output taken as-is."
                )

    # torch expects float32
    intensity_image = np.asarray(crop_max_square(pil_image).convert("L"), np.float32) / 255.0
    amp, phase = recon_inline(intensity_image, wavelength=wavelength, z=z_expect * 1e-3, px=dx)
    focus_tc = focus_score(intensity_image, wavelength, z_expect * 1e-3, dx)

    hologram = np.array(pil_image_crop)
    # TODO: collect these returns into a dataclass for easier access/processing
    return hologram, amp, phase, focus_tc


def recon_inline(
    intensity: Arr32,
    wavelength: float,
    z: float,
    px: float,
    field0: npt.NDArray[np.complex64] | None = None,
    reference: npt.NDArray[np.complex64] | float | None = None,
    method: ReconstructionMethod = ReconstructionMethod.FRESNEL,
    pad: bool = True,
) -> tuple[Arr32, Arr32]:
    """Reconstruct amplitude and phase by scalar diffraction propagation.

    Returns:
        amplitude : float32 array (H, W)
        phase     : float32 array (H, W), in radians.

    """
    # NOTE: assumption: reference ~1, object weak => field amplitude ~ sqrt(I)
    # if actual reference exists, use that instead of sqrt.

    # Input checking
    I = np.asarray(intensity, dtype=np.float32)
    if field0 is None:
        if reference is None:
            if I.min() < 0:
                # shift if passed in high-pass or signed data
                I = I - I.min()
            # if values are large, bring to [0,1]
            if I.max() > 1.0:
                logger.warning(f"Intensity measured at {I.max()}, rescaling back to [0,1]")
                I = I / I.max()
            field0 = np.sqrt(I).astype(np.complex64)
        else:
            # normalize measured field relative to reference
            if np.isscalar(reference):
                R = np.full_like(I, fill_value=reference, dtype=np.complex64)
            else:
                R = np.asarray(reference, dtype=np.complex64)

            # from identity: I = |R+O|^2 -> estimate E = I/R
            field0 = (I / np.abs(R)).astype(np.complex64)

    H0, W0 = field0.shape  # rows=y, cols=x
    if pad:
        pad_y = H0
        pad_x = W0
        field = np.pad(
            field0,
            ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
            mode="constant",
        )
    else:
        field = field0

    logger.info("Computing Reconstruction...")

    H, W = field.shape
    k = 2.0 * np.pi / wavelength

    #  Frequency grids
    # NOTE: np.fft.fftfreq(N, d) returns cycles/meter, might need to multiply by 2π
    # to return proper units
    fx = np.fft.fftfreq(W, d=px)  # length W (x corresponds to columns)
    fy = np.fft.fftfreq(H, d=px)  # length H (y corresponds to rows)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")  # shapes (H, W)

    match method:
        case ReconstructionMethod.FRESNEL:
            # Fresnel-FFT transfer function:
            # H(fx,fy) = exp(i k z) * exp(-i π λ z (fx^2 + fy^2))
            Hf = np.exp(i * k * z) * np.exp(-i * np.pi * wavelength * z * (FX**2 + FY**2))
            U1 = np.fft.ifft2(np.fft.fft2(field) * Hf)
        case ReconstructionMethod.ANGULAR:
            # Angular spectrum:
            # kz = sqrt(k^2 - (2π fx)^2 - (2π fy)^2)
            # H = exp(i z kz)
            kx = 2.0 * np.pi * FX
            ky = 2.0 * np.pi * FY
            k_sq = k**2
            kxy_sq = kx**2 + ky**2
            # evanescent components: where kxy_sq > k^2 ⇒ imaginary kz (decays). Keep complex sqrt.
            kz = np.sqrt(np.maximum(0, k_sq - kxy_sq)) + (i * np.sqrt(np.maximum(0, kxy_sq - k_sq)))
            Ha = np.exp(i * z * kz)
            U1 = np.fft.ifft2(np.fft.fft2(field) * Ha)
        case _:
            raise ValueError("method must be 'fresnel' or 'angular'")

    # Remove padding
    if pad:
        sy0 = (H - H0) // 2
        sx0 = (W - W0) // 2
        U1 = U1[sy0 : sy0 + H0, sx0 : sx0 + W0]

    amplitude = np.abs(U1).astype(np.float32)
    phase = np.angle(U1).astype(np.float32)

    return amplitude, phase


def tamura_coefficient(image: npt.NDArray[np.floating[Any]]) -> float:
    """Tamura coefficient ``TC = sqrt(std / mean)`` of a non-negative image.

    A contrast statistic widely used as a holographic autofocus criterion
    (Memmolo et al., Opt. Lett. 36, 2011).
    """
    arr = np.asarray(image, dtype=np.float64)
    mean = float(arr.mean())
    if mean <= 0.0:
        return 0.0
    return float(np.sqrt(arr.std() / mean))


def gradient_tamura(amplitude: npt.NDArray[np.floating[Any]]) -> float:
    """Tamura coefficient of the gradient magnitude of a reconstructed amplitude.

    Edge energy concentrates at the in-focus plane, so this score is *maximized*
    at the correct reconstruction depth   higher is sharper. Empirically it peaks
    at the true focus on this project's DLHM holograms where the plain amplitude
    Tamura coefficient stays flat.
    """
    gy, gx = np.gradient(np.asarray(amplitude, dtype=np.float64))
    return tamura_coefficient(np.sqrt(gx**2 + gy**2))


def focus_score(intensity: Arr32, wavelength: float, z: float, px: float) -> float:
    """Label-free focus sharpness of a hologram reconstructed at depth ``z`` (m).

    Suppresses the DC reference term (uses the hologram contrast ``I/mean(I) - 1``,
    without which the propagated field is dominated by the background and carries no
    focus signal), backpropagates it, and returns the gradient-Tamura sharpness of
    the resulting amplitude. Higher is sharper.
    """
    meas = np.asarray(intensity, dtype=np.float32)
    contrast = (meas / max(float(meas.mean()), 1e-12) - 1.0).astype(np.complex64)
    (amplitude, _phase) = recon_inline(meas, wavelength=wavelength, z=z, px=px, field0=contrast)
    return gradient_tamura(amplitude)


def dlhm_effective_z_mm(z_source_sample_mm: float, l_source_screen_mm: float) -> float:
    """Plane-wave-equivalent reconstruction distance for a DLHM hologram (mm).

    In DLHM the sample sits ``z`` from the point source and the sensor at ``L``; by
    the Fresnel scaling theorem the recorded hologram, backpropagated on the sensor
    grid with a plane wave, refocuses at ``M * (L - z)`` with ``M = L / z``. Returns
    ``z`` unchanged (no correction) when the geometry is degenerate (``z <= 0`` or
    ``z >= L``), as can happen with unconstrained regression predictions.
    """
    if not 0.0 < z_source_sample_mm < l_source_screen_mm:
        logger.warning(
            f"DLHM geometry degenerate (z={z_source_sample_mm} mm, L={l_source_screen_mm} mm); "
            + "using the uncorrected depth for reconstruction."
        )
        return z_source_sample_mm
    magnification = l_source_screen_mm / z_source_sample_mm
    return magnification * (l_source_screen_mm - z_source_sample_mm)
