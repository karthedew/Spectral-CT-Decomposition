import numpy as np
from scipy.signal import convolve
from numba import njit

# --- Constants ---
NX = 512
NY = 512
XIMAGESIDE = 18.0  # cm
YIMAGESIDE = 18.0  # cm
RADIUS = 50.0      # cm
SRC_TO_DET = 100.0 # cm
NVIEWS = 256
NBINS = 1024
SLEN = 2 * np.pi

# Derived
FANANGLE2 = np.arcsin((XIMAGESIDE / 2.) / RADIUS)
DETLEN = 2 * np.tan(FANANGLE2) * SRC_TO_DET
DU = DETLEN / NBINS
DS = SLEN / NVIEWS
DUP = DU * RADIUS / SRC_TO_DET
U0 = -DETLEN / 2.

# --- Ramp filter ---
def ramp_kernel(n, du):
    filt = np.zeros(n)
    mid = n // 2
    for i in range(n):
        k = i - mid
        if k == 0:
            filt[i] = 1 / (4 * du ** 2)
        elif k % 2 == 0:
            filt[i] = 0
        else:
            filt[i] = -1 / (np.pi ** 2 * du ** 2 * k ** 2)
    return filt

RAMP = ramp_kernel(2 * NBINS - 1, DUP) * DUP

# --- Backprojection ---
@njit
def circularFanbeamBackProjection(sinogram, nx, ny, image):
    dx = XIMAGESIDE / nx
    dy = YIMAGESIDE / ny
    x0 = -XIMAGESIDE / 2.
    y0 = -YIMAGESIDE / 2.

    for sindex in range(NVIEWS):
        s = sindex * DS
        xsource = RADIUS * np.cos(s)
        ysource = RADIUS * np.sin(s)
        xDetCenter = (RADIUS - SRC_TO_DET) * np.cos(s)
        yDetCenter = (RADIUS - SRC_TO_DET) * np.sin(s)
        eux = -np.sin(s)
        euy = np.cos(s)
        ewx = np.cos(s)
        ewy = np.sin(s)

        for iy in range(ny):
            pix_y = y0 + dy * (iy + 0.5)
            for ix in range(nx):
                pix_x = x0 + dx * (ix + 0.5)
                frad = np.sqrt(pix_x ** 2 + pix_y ** 2)
                fphi = np.arctan2(pix_y, pix_x)
                if frad <= XIMAGESIDE / 2.:
                    bigu = (RADIUS + frad * np.sin(s - fphi - np.pi / 2.)) / RADIUS
                    bpweight = 1. / (bigu * bigu)
                    ew_dot = (pix_x - xsource) * ewx + (pix_y - ysource) * ewy
                    rayratio = -SRC_TO_DET / ew_dot
                    det_x = xsource + rayratio * (pix_x - xsource)
                    det_y = ysource + rayratio * (pix_y - ysource)
                    upos = (det_x - xDetCenter) * eux + (det_y - yDetCenter) * euy
                    if (upos - U0 >= DU / 2.) and (upos - U0 < DETLEN - DU / 2.):
                        bin_loc = (upos - U0) / DU + 0.5
                        nbin1 = int(bin_loc) - 1
                        nbin2 = nbin1 + 1
                        frac = bin_loc - int(bin_loc)
                        det_value = (1 - frac) * sinogram[sindex, nbin1] + frac * sinogram[sindex, nbin2]
                    else:
                        det_value = 0.0
                    image[ix, iy] += bpweight * det_value * DS

# --- FBP Wrapper ---
def fbp(sino: np.ndarray) -> np.ndarray:
    """
    Perform fan-beam filtered backprojection on a (256, 1024) sinogram.
    Returns a (512, 512) reconstructed image.
    """
    assert sino.shape == (NVIEWS, NBINS), f"Expected shape {(NVIEWS, NBINS)}, got {sino.shape}"

    # Detector weighting
    uarray = np.linspace(U0 + DU / 2., U0 + DU / 2. + DETLEN - DU, NBINS)
    uarray *= RADIUS / SRC_TO_DET
    data_weight = RADIUS / np.sqrt(RADIUS ** 2 + uarray ** 2)

    # Filtering
    filtered = np.zeros_like(sino)
    for i in range(NVIEWS):
        weighted = data_weight * sino[i, :]
        filtered[i, :] = convolve(weighted, RAMP, mode='same')

    # Backprojection
    recon = np.zeros((NX, NY), dtype=np.float32)
    circularFanbeamBackProjection(filtered, NX, NY, recon)
    return recon
