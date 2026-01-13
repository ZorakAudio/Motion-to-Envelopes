#!/usr/bin/env python3
# Camera_Movement_Analyzer_v5F_cli_overlay_ffmpeg_3Dvector.py
# v5F: robust second pass with cached scales; 3D vector overlay; global XY scale;
#      live smoothing; Z=4 decimals; text auto-fit; CSV parity with robust overlay.

import os, sys, math, csv, argparse, subprocess, shutil, time, threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from queue import Queue


import heapq
import numpy as np
import cv2 as cv
import csv
import os


# --- adaptive prefilter (from ROI Flux Tracker) ---
import numpy as np, cv2 as cv

HUD_BAND_H, GRAPH_H, GRAPH_HISTORY_W = 60, 120, 600
ORB_NFEATURES, RANSAC_THRESH, AFF_MAXIT, AFF_CONF = 2000, 1.5, 4000, 0.995
FLOW_SCALE = 0.65
FB_OPTS = dict(pyr_scale=0.5, levels=5, winsize=27, iterations=4, poly_n=7, poly_sigma=1.3, flags=0)

# --- performance toggles ---
# ORB is often the #1 CPU hog at high resolutions. Run ORB on a smaller image and scale points back up.
ORB_SCALE = 0.65  # 1.0 = full-res ORB; ~FLOW_SCALE is usually fine

# Adaptive blur is expensive (Sobel + multiple blurs + per-pixel weights). Keep it, but allow hard-disable.
ENABLE_ADAPTIVE_BLUR = False

# Debug per-frame flow statistics are VERY expensive (extra Farneback + Sobels). Leave off for normal runs.
ENABLE_FLOW_DEBUG_STATS = False


def _lap_mad_sigma(img_u8):
    k = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], np.float32)
    r = cv.filter2D(img_u8.astype(np.float32), -1, k)
    return 1.4826 * np.median(np.abs(r))

def _coherence(img_u8):
    gx = cv.Sobel(img_u8, cv.CV_32F, 1, 0, 3)
    gy = cv.Sobel(img_u8, cv.CV_32F, 0, 1, 3)
    Jxx, Jxy, Jyy = np.mean(gx*gx), np.mean(gx*gy), np.mean(gy*gy)
    t = np.sqrt((Jxx-Jyy)**2 + 4.0*Jxy*Jxy)
    lam1 = 0.5*((Jxx+Jyy) + t); lam2 = 0.5*((Jxx+Jyy) - t)
    return float((lam1 - lam2) / (lam1 + lam2 + 1e-9))

def _adaptive_blur(prev_s, curr_s):
    if not ENABLE_ADAPTIVE_BLUR:
        return prev_s, curr_s, 0.0, 0.0
    sigma_n = 0.5*(_lap_mad_sigma(prev_s)+_lap_mad_sigma(curr_s)) / 255.0
    coh     = 0.5*(_coherence(prev_s)+_coherence(curr_s))
    noise_score = np.clip(sigma_n*(1.0 - coh), 0.0, 1.0)

    SIGMA_MAX = 1.25
    sigma = float(np.interp(noise_score, [0.003, 0.02], [0.0, SIGMA_MAX]))

    f32 = prev_s.astype(np.float32)
    mu  = cv.blur(f32, (5,5))
    var = cv.blur(f32*f32, (5,5)) - mu*mu
    w   = (sigma_n**2) / (sigma_n**2 + np.maximum(var, 1e-9))
    w   = cv.blur(w, (3,3)).astype(np.float32)
    w   = np.clip(w, 0.0, 1.0)

    if sigma <= 1e-6:
        return prev_s, curr_s, 0.0, 0.0

    b0 = cv.GaussianBlur(prev_s, (0,0), sigma)
    b1 = cv.GaussianBlur(curr_s, (0,0), sigma)
    prev_f = ((1.0-w)*prev_s + w*b0).astype(np.uint8)
    curr_f = ((1.0-w)*curr_s + w*b1).astype(np.uint8)
    return prev_f, curr_f, sigma, float(np.mean(w))

def gauss_blur1d(z, sigma):
    """
    Simple 1D Gaussian blur for control-rate signals.
    Used for entropy: split slow trend vs residual.
    z: 1D array
    sigma: in *samples* (frames), not seconds.
    """
    z = np.asarray(z, np.float64)
    if z.size == 0:
        return z
    sigma = float(max(1e-6, sigma))
    R = int(np.ceil(3.0 * sigma))
    if R <= 0:
        return z.copy()
    kx = np.arange(-R, R + 1, dtype=np.float64)
    k = np.exp(-0.5 * (kx / sigma) ** 2)
    k /= (np.sum(k) + 1e-12)
    zp = np.pad(z, (R, R), mode="reflect")
    return np.convolve(zp, k, mode="valid")


def adaptive_blur_from_curv(gray_u8, curv01,
                            blur_small=1, blur_big=7,
                            t0=0.10, t1=0.35):
    """
    gray_u8: uint8 image
    curv01:  float32 in [0,1] same size as gray_u8
    """
    # weight for "structure": 0=flat, 1=strong structure
    w = (curv01 - t0) / max(1e-6, (t1 - t0))
    w = np.clip(w, 0.0, 1.0).astype(np.float32)

    # two blurs only (fast enough)
    if blur_small <= 1:
        a = gray_u8
    else:
        a = cv.GaussianBlur(gray_u8, (0, 0), blur_small)

    b = cv.GaussianBlur(gray_u8, (0, 0), blur_big)

    # keep structure sharp (more of a), flatten flats (more of b)
    out = (w * a.astype(np.float32) + (1.0 - w) * b.astype(np.float32))
    return out.astype(np.uint8)


def _inject_gauss_struct(img: np.ndarray, eps: float = 0.12) -> np.ndarray:
    """
    Multi-scale Laplacian-of-Gaussian residual injection.
    Adds tiny, smooth curvature so classical flow has gradients to latch onto,
    especially in flat / low-texture regions.
    """
    img_f = img.astype(np.float32)
    out = img_f.copy()

    # σ = 1.0, 2.0, 3.5 with decaying weights
    for sigma, w in [(1.0, eps),
                     (2.0, eps * 0.5),
                     (3.5, eps * 0.25)]:
        if sigma <= 0.0 or w == 0.0:
            continue
        blur = cv.GaussianBlur(img_f, (0, 0), sigma)
        LoG  = img_f - blur          # curvature-ish residual
        out += w * LoG

    return np.clip(out, 0.0, 255.0).astype(np.uint8)

def _inject_gauss_struct(img: np.ndarray, eps: float = 0.12) -> np.ndarray:
    """
    Multi-scale Laplacian-of-Gaussian residual injection.
    Adds tiny, smooth curvature so classical flow has gradients to latch onto,
    especially in flat / low-texture regions.
    """
    img_f = img.astype(np.float32)
    out = img_f.copy()

    # σ = 1.0, 2.0, 3.5 with decaying weights
    for sigma, w in [(1.0, eps),
                     (2.0, eps * 0.5),
                     (3.5, eps * 0.25)]:
        if sigma <= 0.0 or w == 0.0:
            continue
        blur = cv.GaussianBlur(img_f, (0, 0), sigma)
        LoG  = img_f - blur          # curvature-ish residual
        out += w * LoG

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# --- PATCH VZ25_GLOB_A: IG-LoG curvature energy + 2.5D vz from full-frame flow ---

def _iglog_energy_frame(img_u8: np.ndarray) -> np.ndarray:
    """
    Multi-scale IG-LoG curvature energy for a full grayscale frame.
    Returns |accumulated Laplacian| as float32.
    """
    if img_u8.ndim == 3:
        img_u8 = cv.cvtColor(img_u8, cv.COLOR_BGR2GRAY)
    pf = img_u8.astype(np.float32) / 255.0
    acc = np.zeros_like(pf, np.float32)

    for sigma, w in [(1.0, 1.0),
                     (2.0, 0.5),
                     (3.5, 0.25)]:
        if sigma <= 0.0 or w == 0.0:
            continue
        blur = cv.GaussianBlur(pf, (0, 0), sigma)
        hp   = pf - blur
        lap  = cv.Laplacian(hp, cv.CV_32F, ksize=3)
        acc += w * lap

    return np.abs(acc)


def _vz25_global_from_flow(prev_gray: np.ndarray,
                           gray: np.ndarray,
                           scale: float = FLOW_SCALE) -> float:
    """
    2.5D curvature-depth for the whole frame:

      1) Downscale frames to 'scale'.
      2) Run Farneback to get fx, fy.
      3) Compute IG-LoG curvature on the *current* scaled frame.
      4) Project flow along radial direction from frame center.
      5) Weight radial flow by curvature & motion magnitude → vz25 map.
      6) Return robust median of vz25 over active motion pixels.

    Returns vz25 in scaled-flow units per frame (caller applies fps and z_div_gain).
    """
    if prev_gray is None or gray is None:
        return 0.0

    h, w = gray.shape[:2]
    W = max(8, int(round(w * scale)))
    H = max(8, int(round(h * scale)))

    prev_s = cv.resize(prev_gray, (W, H), cv.INTER_AREA)
    curr_s = cv.resize(gray,      (W, H), cv.INTER_AREA)

    # same prefilter path as divergence: curvature injection + adaptive blur
    prev_s = _inject_gauss_struct(prev_s)
    curr_s = _inject_gauss_struct(curr_s)
    
    curv_prev = _iglog_energy_frame(prev_s)
    curv_curr = _iglog_energy_frame(curr_s)

    # normalize curv to 0..1 (copy your robust 5–95% normalization style)
    lo = float(np.percentile(curv_prev, 5.0)); hi = float(np.percentile(curv_prev, 95.0))
    curv01_prev = np.clip((curv_prev - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)

    prev_s = adaptive_blur_from_curv(prev_s, curv01_prev)
    # repeat for curr if you want symmetry:
    lo = float(np.percentile(curv_curr, 5.0)); hi = float(np.percentile(curv_curr, 95.0))
    curv01_curr = np.clip((curv_curr - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)
    curr_s = adaptive_blur_from_curv(curr_s, curv01_curr)

    sigma = 0.0
    blur_strength = 0.0

    FB = dict(FB_OPTS)
    if sigma > 0.0:
        FB["flags"]   = cv.OPTFLOW_FARNEBACK_GAUSSIAN
        FB["winsize"] = max(FB.get("winsize", 27),
                            27 + int(24 * min(1.0, blur_strength)))

    flow = cv.calcOpticalFlowFarneback(prev_s, curr_s, None, **FB)
    fx = flow[..., 0].astype(np.float32)
    fy = flow[..., 1].astype(np.float32)

    # 1) curvature magnitude
    curv = _iglog_energy_frame(curr_s)
    if not np.any(curv > 1e-8):
        return 0.0

    lo = float(np.percentile(curv, 5.0))
    hi = float(np.percentile(curv, 95.0))
    rng = max(hi - lo, 1e-9)
    curv01 = np.clip((curv - lo) / rng, 0.0, 1.0)

    # 2) radial direction from frame center (in scaled coords)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    dx = xx - cx
    dy = yy - cy
    rad = np.sqrt(dx*dx + dy*dy) + 1e-6
    ux = dx / rad
    uy = dy / rad

    # radial component of flow (outward +, inward -)
    v_rad = fx * ux + fy * uy

    # 3) gate by motion magnitude
    vmag = np.sqrt(fx*fx + fy*fy)
    m95 = float(np.percentile(vmag, 95.0)) or 1e-9
    mag01 = np.clip(vmag / m95, 0.0, 1.0)

    # 4) curvature-weighted radial motion → vz25 map
    vz_map = np.sign(v_rad) * curv01 * mag01

    # robust pooling over active motion
    m = (vmag > 1e-6)
    if np.count_nonzero(m) < 32:
        return 0.0

    return float(np.median(vz_map[m]))

# --- END PATCH VZ25_GLOB_A ---
def _robust01_from_u8_absdiff(a_u8: np.ndarray, b_u8: np.ndarray,
                             p_lo: float = 10.0, p_hi: float = 90.0,
                             blur_sigma: float = 1.0, gamma: float = 0.85) -> np.ndarray:
    """HG-like structural motion salience:
    absdiff on structure maps -> robust normalize -> optional blur.
    Returns float32 in [0,1].
    """
    d = cv.absdiff(a_u8, b_u8).astype(np.float32)
    if blur_sigma and blur_sigma > 0.0:
        d = cv.GaussianBlur(d, (0, 0), float(blur_sigma))
    lo = float(np.percentile(d, p_lo))
    hi = float(np.percentile(d, p_hi))
    rng = max(hi - lo, 1e-9)
    x = np.clip((d - lo) / rng, 0.0, 1.0)
    if gamma and gamma != 1.0:
        x = np.power(x, float(gamma), dtype=np.float32)
    return x.astype(np.float32)

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median. Arrays are 1D."""
    if values is None or weights is None:
        return 0.0
    v = np.asarray(values, np.float64).ravel()
    w = np.asarray(weights, np.float64).ravel()
    if v.size == 0 or w.size == 0 or v.size != w.size:
        return 0.0
    m = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    v = v[m]; w = w[m]
    if v.size == 0:
        return 0.0
    idx = np.argsort(v)
    v = v[idx]; w = w[idx]
    c = np.cumsum(w)
    cutoff = 0.5 * c[-1]
    j = int(np.searchsorted(c, cutoff, side="left"))
    j = max(0, min(j, v.size - 1))
    return float(v[j])


# --- PERF PATCH: compute Farneback once per frame, reuse for div + vz25 ---
def _compute_flow_pack(prev_gray: np.ndarray, gray: np.ndarray, scale: float = FLOW_SCALE):
    """Shared Farneback path used by divergence + vz25 (and bg gating if you add it).
    Returns dict with fx, fy, vmag, curv01, H, W.
    """
    if prev_gray is None or gray is None:
        return None
    h, w = gray.shape[:2]
    W = max(8, int(round(w * scale)))
    H = max(8, int(round(h * scale)))

    prev_s = cv.resize(prev_gray, (W, H), cv.INTER_AREA)
    curr_s = cv.resize(gray,      (W, H), cv.INTER_AREA)

    prev_s = _inject_gauss_struct(prev_s)
    curr_s = _inject_gauss_struct(curr_s)

    curv_prev = _iglog_energy_frame(prev_s)
    curv_curr = _iglog_energy_frame(curr_s)

    # normalize curv to 0..1 (copy your robust 5–95% normalization style)
    lo = float(np.percentile(curv_prev, 5.0)); hi = float(np.percentile(curv_prev, 95.0))
    curv01_prev = np.clip((curv_prev - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)

    prev_s = adaptive_blur_from_curv(prev_s, curv01_prev)
    # repeat for curr if you want symmetry:
    lo = float(np.percentile(curv_curr, 5.0)); hi = float(np.percentile(curv_curr, 95.0))
    curv01_curr = np.clip((curv_curr - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)
    curr_s = adaptive_blur_from_curv(curr_s, curv01_curr)

    sigma = 0.0
    blur_strength = 0.0


    FB = dict(FB_OPTS)
    if sigma > 0.0:
        FB["flags"]   = cv.OPTFLOW_FARNEBACK_GAUSSIAN
        FB["winsize"] = max(FB.get("winsize", 27), 27 + int(24 * min(1.0, blur_strength)))

    flow = cv.calcOpticalFlowFarneback(prev_s, curr_s, None, **FB)
    fx = flow[..., 0].astype(np.float32)
    fy = flow[..., 1].astype(np.float32)
    vmag = np.sqrt(fx*fx + fy*fy)

    curv = _iglog_energy_frame(curr_s)
    if np.any(curv > 1e-8):
        lo = float(np.percentile(curv, 5.0))
        hi = float(np.percentile(curv, 95.0))
        curv01 = np.clip((curv - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)
    else:
        curv01 = np.zeros((H, W), np.float32)

    # HG-like structural motion salience (0..1) for weighting div/vz25:
    # detect structure on IG-LoG map, but measure *change* (absdiff) between frames.
    try:
        Sprev = _struct_u8_from_gray(prev_s)
        Scurr = _struct_u8_from_gray(curr_s)
        hg01 = _robust01_from_u8_absdiff(Sprev, Scurr, p_lo=10.0, p_hi=90.0, blur_sigma=1.0, gamma=0.85)
    except Exception:
        hg01 = np.ones((H, W), np.float32)

    return dict(fx=fx, fy=fy, vmag=vmag, curv01=curv01, hg01=hg01, H=H, W=W)


def _divergence_from_pack(pack) -> float:
    if not pack:
        return 0.0
    fx = pack["fx"]; fy = pack["fy"]
    mag = pack["vmag"]
    hg01 = pack.get("hg01", None)

    # original coherent gating
    mthr = np.percentile(mag, 70.0)
    mask = (mag >= mthr)
    ang  = np.arctan2(fy, fx)
    a0   = np.median(ang[mask]) if np.count_nonzero(mask) else 0.0
    mask &= (np.cos(ang - a0) >= 0.8)

    dvx_dx = cv.Sobel(fx, cv.CV_32F, 1, 0, 3) / 8.0
    dvy_dy = cv.Sobel(fy, cv.CV_32F, 0, 1, 3) / 8.0
    div = dvx_dx + dvy_dy

    # HG-guided weighting: don't change the differential math, change WHERE it's trusted.
    if hg01 is None:
        hg01 = np.ones_like(div, np.float32)
    else:
        hg01 = hg01.astype(np.float32)

    m95 = float(np.percentile(mag, 95.0)) or 1e-9
    mag01 = np.clip(mag / m95, 0.0, 1.0).astype(np.float32)
    w = (hg01 * mag01).astype(np.float32)

    mask2 = mask & (w >= 0.10)
    if np.count_nonzero(mask2) >= 24:
        return _weighted_median(div[mask2], w[mask2])

    mask3 = (w >= 0.08)
    if np.count_nonzero(mask3) >= 64:
        return _weighted_median(div[mask3], w[mask3])

    if np.count_nonzero(mask) >= 16:
        return float(np.median(div[mask]))
    return float(np.median(div))


def _vz25_from_pack(pack) -> float:
    if not pack:
        return 0.0
    fx = pack["fx"]; fy = pack["fy"]
    curv01 = pack["curv01"]
    hg01 = pack.get("hg01", None)
    H = int(pack["H"]); W = int(pack["W"])

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    dx = xx - cx
    dy = yy - cy
    rad = np.sqrt(dx*dx + dy*dy) + 1e-6
    ux = dx / rad
    uy = dy / rad

    v_rad = fx * ux + fy * uy
    vmag = pack["vmag"]
    m95 = float(np.percentile(vmag, 95.0)) or 1e-9
    mag01 = np.clip(vmag / m95, 0.0, 1.0)

    if hg01 is None:
        hg01 = np.ones((H, W), np.float32)
    else:
        hg01 = hg01.astype(np.float32)

    vz_map = np.sign(v_rad) * curv01 * mag01 * hg01

    m = (vmag > 1e-6) & (hg01 >= 0.08)
    if np.count_nonzero(m) < 32:
        m = (vmag > 1e-6)
    if np.count_nonzero(m) < 32:
        return 0.0

    w = (hg01 * mag01).astype(np.float32)
    return _weighted_median(vz_map[m], w[m])

# --- END PERF PATCH ---

# ====================== Global motion backends: FB / HG / Hybrid ======================
# FB   = existing ORB/RANSAC affine 2D estimate (current behavior)
# HG   = IG-LoG structural phase-correlation + LK median (translation-only, robust on low texture)
# Hybrid = soft switch between FB and HG based on confidence-weighted magnitude (no incoherent blending)

# HG params (operate on FLOW_SCALE-sized frames; units are full-res px/frame after dividing by FLOW_SCALE)
HG_PC_DOWNSCALE_GLOB = 0.5     # additional downscale for phase correlation on structural map (speed/stability)
HG_PC_MIN_RESP_GLOB  = 0.12    # phaseCorr response gate (0..1-ish)
HG_GFTT_MAX_CORNERS  = 260     # LK features
HG_GFTT_QL           = 0.01
HG_GFTT_MIN_DIST     = 7
HG_LK_WIN            = 21
HG_LK_MAXLEVEL       = 3

HYB_ATTACK = 0.35   # how fast we switch *towards* HG when HG wins
HYB_RELEASE = 0.18  # how fast we switch *back* towards FB
HYB_DOT_OPPOSE = -0.15  # if direction cosine is below this, don't blend (pick winner)


def _fb_conf_from_orb_matches(nmatch: int) -> float:
    """Rough confidence for the existing ORB/RANSAC affine estimate (0..1)."""
    try:
        n = int(nmatch)
    except Exception:
        n = 0
    # Below ~8 matches: basically guessy. Past ~40: "good enough".
    return float(np.clip((n - 8.0) / 32.0, 0.0, 1.0))

def _orb_struct_detect_and_compute(orb, gray_orb_u8: np.ndarray, max_kp: int = ORB_NFEATURES):
    """Structure-guided ORB:
    - Detect keypoints on IG-LoG structural map (more invariant to lighting / flat texture)
    - Compute descriptors on grayscale (keeps descriptor distinctiveness)

    Returns (keypoints, descriptors).
    """
    try:
        struct_u8 = _struct_u8_from_gray(gray_orb_u8)
        kps = orb.detect(struct_u8, None)
        if not kps:
            return [], None
        if max_kp is not None and int(max_kp) > 0 and len(kps) > int(max_kp):
            kps = sorted(kps, key=lambda k: float(getattr(k, "response", 0.0)), reverse=True)[:int(max_kp)]
        kps, des = orb.compute(gray_orb_u8, kps)
        return kps, des
    except Exception:
        # hard fallback to vanilla ORB
        try:
            return orb.detectAndCompute(gray_orb_u8, None)
        except Exception:
            return [], None


def _struct_u8_from_gray(gray_u8: np.ndarray) -> np.ndarray:
    """
    Cheap global structural map for HG:
      - use IG-LoG energy (already present for vz25/div) as an "edge/curvature" measure
      - robust-normalize to 0..255
    """
    try:
        E = _iglog_energy_frame(gray_u8)
        if E is None or E.size == 0:
            return gray_u8.astype(np.uint8)
        lo = float(np.percentile(E, 5.0))
        hi = float(np.percentile(E, 95.0))
        rng = max(hi - lo, 1e-9)
        M = np.clip((E - lo) / rng, 0.0, 1.0)
        # gamma < 1 emphasizes weaker structure a bit (helps flat scenes)
        M = np.power(M, 0.65, dtype=np.float32)
        return np.clip(M * 255.0, 0, 255).astype(np.uint8)
    except Exception:
        return gray_u8.astype(np.uint8)


def _phasecorr_shift(prev_u8: np.ndarray, curr_u8: np.ndarray, down: float = HG_PC_DOWNSCALE_GLOB):
    """Return (dx, dy, resp) in *input image pixels* (not full-res)."""
    if prev_u8 is None or curr_u8 is None:
        return 0.0, 0.0, 0.0
    H, W = prev_u8.shape[:2]
    if H < 8 or W < 8:
        return 0.0, 0.0, 0.0

    d = float(np.clip(down, 0.25, 1.0))
    if d != 1.0:
        Wd = max(8, int(round(W * d)))
        Hd = max(8, int(round(H * d)))
        p = cv.resize(prev_u8, (Wd, Hd), cv.INTER_AREA)
        c = cv.resize(curr_u8, (Wd, Hd), cv.INTER_AREA)
    else:
        p, c = prev_u8, curr_u8

    pf = p.astype(np.float32)
    cf = c.astype(np.float32)

    try:
        win = cv.createHanningWindow((pf.shape[1], pf.shape[0]), cv.CV_32F)
        (dx, dy), resp = cv.phaseCorrelate(pf, cf, win)
    except Exception:
        # fallback without window
        (dx, dy), resp = cv.phaseCorrelate(pf, cf)

    # back to pre-downscale pixel units
    dx = float(dx) / d
    dy = float(dy) / d
    return dx, dy, float(resp)



def _cut_struct_phasecorr(prev_gray: np.ndarray, gray: np.ndarray,
                          scale: float = FLOW_SCALE) -> Tuple[float, float]:
    """Hard-cut detector using STRUCTURE (IG-LoG map) + phase correlation.

    Returns:
      resp : phase correlation peak response (higher = more consistent)
      ad01 : mean absdiff on structural maps, normalized to [0,1]
    """
    if prev_gray is None or gray is None:
        return 1.0, 0.0
    h, w = gray.shape[:2]
    W = max(8, int(round(w * float(scale))))
    H = max(8, int(round(h * float(scale))))
    prev_s = cv.resize(prev_gray, (W, H), cv.INTER_AREA)
    curr_s = cv.resize(gray,      (W, H), cv.INTER_AREA)

    Mp = _struct_u8_from_gray(prev_s)
    Mc = _struct_u8_from_gray(curr_s)

    # structural change magnitude
    ad01 = float(np.mean(cv.absdiff(Mp, Mc))) / 255.0

    # structural phase correlation response
    _dx, _dy, resp = _phasecorr_shift(Mp, Mc, down=HG_PC_DOWNSCALE_GLOB)
    resp = float(resp)
    if not np.isfinite(resp):
        resp = 0.0
    return resp, ad01

def _hg_global_translation(prev_gray: np.ndarray, gray: np.ndarray, scale: float = FLOW_SCALE, state: Optional[dict] = None):
    """
    Hypergraph-ish global translation estimator.
    Returns:
      dx_pf, dy_pf  (full-res pixels / frame)
      conf          (0..1)
      dbg           (dict)
    """
    if prev_gray is None or gray is None:
        return 0.0, 0.0, 0.0, {"ok": False, "why": "no_prev"}

    h, w = gray.shape[:2]
    Ws = max(8, int(round(w * float(scale))))
    Hs = max(8, int(round(h * float(scale))))
    if Ws < 8 or Hs < 8:
        return 0.0, 0.0, 0.0, {"ok": False, "why": "too_small"}

    prev_s = cv.resize(prev_gray, (Ws, Hs), cv.INTER_AREA)
    curr_s = cv.resize(gray,      (Ws, Hs), cv.INTER_AREA)

    # structural maps (uint8)
    Mp = _struct_u8_from_gray(prev_s)
    Mc = _struct_u8_from_gray(curr_s)

    # phase correlation (coarse)
    dx_pc, dy_pc, resp = _phasecorr_shift(Mp, Mc, down=HG_PC_DOWNSCALE_GLOB)

    # LK refinement around the coarse shift
    dx_lk = dy_lk = 0.0
    lk_conf = 0.0
    n_inl = 0

    try:
        p0 = cv.goodFeaturesToTrack(Mp, maxCorners=int(HG_GFTT_MAX_CORNERS),
                                    qualityLevel=float(HG_GFTT_QL),
                                    minDistance=float(HG_GFTT_MIN_DIST),
                                    blockSize=7, useHarrisDetector=False)
        if p0 is not None and int(p0.shape[0]) >= 8:
            p0 = p0.astype(np.float32)
            p1_init = p0 + np.array([[[dx_pc, dy_pc]]], dtype=np.float32)

            p1, st, err = cv.calcOpticalFlowPyrLK(
                prev_s, curr_s, p0, p1_init,
                winSize=(int(HG_LK_WIN), int(HG_LK_WIN)),
                maxLevel=int(HG_LK_MAXLEVEL),
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 25, 0.01),
                flags=cv.OPTFLOW_USE_INITIAL_FLOW
            )
            if p1 is not None and st is not None:
                st = st.reshape(-1).astype(np.uint8)
                good = (st == 1)
                n_inl = int(np.count_nonzero(good))
                if n_inl >= 8:
                    d = (p1.reshape(-1, 2)[good] - p0.reshape(-1, 2)[good]).astype(np.float32)
                    med = np.median(d, axis=0)
                    dx_lk = float(med[0]); dy_lk = float(med[1])

                    # coherence: how tightly displacements cluster around the median
                    dev = np.sqrt((d[:, 0] - med[0])**2 + (d[:, 1] - med[1])**2)
                    mad = float(np.median(dev)) if dev.size else 999.0

                    # inlier fraction + spread gate
                    frac = float(n_inl) / float(max(1, int(p0.shape[0])))
                    spread_score = float(np.exp(-mad / 2.0))  # ~2px MAD => ~0.37
                    lk_conf = float(np.clip(frac * spread_score, 0.0, 1.0))
    except Exception:
        pass

    # fuse: if LK looks reliable, prefer it; else keep phaseCorr
    use_lk = (lk_conf >= 0.18 and n_inl >= 10)
    if use_lk:
        # small blend keeps phaseCorr bias helpful on huge steps
        w_lk = float(np.clip(0.65 + 0.35 * lk_conf, 0.65, 0.95))
        dx_s = (1.0 - w_lk) * dx_pc + w_lk * dx_lk
        dy_s = (1.0 - w_lk) * dy_pc + w_lk * dy_lk
    else:
        dx_s, dy_s = dx_pc, dy_pc

    # confidence: mix phaseCorr response + LK coherence
    conf = float(np.clip(0.60 * np.clip(resp, 0.0, 1.0) + 0.40 * lk_conf, 0.0, 1.0))

    # hard gate: totally untrusted phaseCorr response and no LK
    if resp < float(HG_PC_MIN_RESP_GLOB) and lk_conf < 0.15:
        conf *= 0.35

    dx_pf = float(dx_s) * (1.0 / float(scale))
    dy_pf = float(dy_s) * (1.0 / float(scale))

    dbg = {"ok": True, "resp": float(resp), "lk_conf": float(lk_conf), "n_inl": int(n_inl),
           "dx_s": float(dx_s), "dy_s": float(dy_s)}
    return dx_pf, dy_pf, conf, dbg


def _hybrid_blend_step(alpha_prev: float,
                       fb_dx: float, fb_dy: float, fb_conf: float,
                       hg_dx: float, hg_dy: float, hg_conf: float) -> Tuple[float, float, float, dict]:
    """
    Hybrid = choose "the moving one" (confidence-weighted magnitude), but soft-switch to avoid flicker.
    Returns (dx, dy, alpha, dbg), where alpha is HG weight.
    """
    fb_dx = float(fb_dx); fb_dy = float(fb_dy)
    hg_dx = float(hg_dx); hg_dy = float(hg_dy)
    fb_conf = float(np.clip(fb_conf, 0.0, 1.0))
    hg_conf = float(np.clip(hg_conf, 0.0, 1.0))

    mag_fb = float(math.hypot(fb_dx, fb_dy))
    mag_hg = float(math.hypot(hg_dx, hg_dy))

    # score = magnitude * confidence (with a small floor so tiny conf doesn't kill everything)
    score_fb = mag_fb * (0.20 + 0.80 * fb_conf)
    score_hg = mag_hg * (0.20 + 0.80 * hg_conf)

    # direction agreement
    dot = fb_dx * hg_dx + fb_dy * hg_dy
    cos = float(dot / (mag_fb * mag_hg + 1e-9)) if (mag_fb > 1e-9 and mag_hg > 1e-9) else 1.0

    if cos <= float(HYB_DOT_OPPOSE):
        target = 1.0 if (score_hg >= score_fb) else 0.0  # avoid incoherent cancellation
    else:
        target = float(score_hg / (score_hg + score_fb + 1e-9))

    a = float(np.clip(alpha_prev, 0.0, 1.0))
    rate = float(HYB_ATTACK if target > a else HYB_RELEASE)
    a = a + rate * (target - a)
    a = float(np.clip(a, 0.0, 1.0))

    dx = (1.0 - a) * fb_dx + a * hg_dx
    dy = (1.0 - a) * fb_dy + a * hg_dy

    dbg = {"alpha": a, "target": target, "cos": cos,
           "mag_fb": mag_fb, "mag_hg": mag_hg,
           "score_fb": score_fb, "score_hg": score_hg}
    return float(dx), float(dy), a, dbg

# ==================== End global motion backends ====================



# ---------- Robust normalizer (per-axis; MAD + percentile + gamma) ----------
class RobustArrowNormalizer:
    def __init__(self, z_thresh: float = 6.0, prctl: float = 95.0, gamma: float = 1.0):
        self.z_thresh = float(z_thresh); self.prctl = float(prctl); self.gamma = float(gamma)
        self.sx = 1.0; self.sy = 1.0; self.sz = 1.0

    @staticmethod
    def _mad_stats(a: np.ndarray):
        if a.size == 0: return 0.0, 0.0
        med = float(np.median(a)); mad = float(np.median(np.abs(a - med)))
        return med, (mad if mad > 1e-12 else 0.0)

    def _scale(self, a: np.ndarray) -> float:
        if a.size == 0: return 1.0
        med, mad = self._mad_stats(a)
        if mad == 0.0:
            s = float(np.percentile(np.abs(a), self.prctl)) or 1.0
            return max(s, 1e-6)
        z = np.abs((a - med) / (1.4826 * mad))
        inl = a[z < self.z_thresh]
        if inl.size < max(10, 0.1*a.size): inl = a
        s = float(np.percentile(np.abs(inl), self.prctl)) or 1.0
        return max(s, 1e-6)

    def fit(self, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray):
        vx = np.asarray(vx, float); vy = np.asarray(vy, float); vz = np.asarray(vz, float)
        self.sx = self._scale(vx); self.sy = self._scale(vy); self.sz = self._scale(vz)
        return self

    def _norm(self, v: float, s: float) -> float:
        s = 1.0 if s <= 0 else s
        x = np.clip(v / s, -1.0, 1.0)
        if self.gamma != 1.0:
            x = np.sign(x) * (abs(x) ** self.gamma)
        return float(x)

    def vx(self, v: float) -> float: return self._norm(v, self.sx)
    def vy(self, v: float) -> float: return self._norm(v, self.sy)
    def vz(self, v: float) -> float: return self._norm(v, self.sz)

# ---------- Profiles ----------
PROFILES = {
    "cinematic": dict(smooth_vel="sg", sg_win_ms=300, sg_poly=3,
                      smooth_ang="lp1", lp_ang_hz=3.0, lp_vel_hz=1.0, kalman_q=0.05, kalman_r=0.3,
                      max_rate_vel=0.0, max_rate_ang=0.0, bias_tau_s=10.0,
                      dir_dead=0.25, dir_hyst_on=0.65, dir_hyst_off=0.30, flip_hold_ms=400,
                      conf_min_matches=120, conf_parallax=0.15,
                      gamma=1.15, resample_hz=100.0, flux_z_weight=1.0,
                      sign_vx="auto", sign_vy="auto", sign_vz="auto", sign_wyaw="auto", sign_wpitch="auto", sign_wroll="auto",
                      no_bursts=False, burst_thresh=0.6, burst_ms=120.0, burst_blend=0.5),
    "handheld": dict(smooth_vel="sg", sg_win_ms=220, sg_poly=3,
                     smooth_ang="lp1", lp_ang_hz=4.0, lp_vel_hz=1.2, kalman_q=0.06, kalman_r=0.35,
                     max_rate_vel=0.0, max_rate_ang=0.0, bias_tau_s=6.0,
                     dir_dead=0.18, dir_hyst_on=0.50, dir_hyst_off=0.20, flip_hold_ms=220,
                     conf_min_matches=100, conf_parallax=0.12,
                     gamma=1.10, resample_hz=120.0, flux_z_weight=1.0,
                     sign_vx="auto", sign_vy="auto", sign_vz="auto", sign_wyaw="auto", sign_wpitch="auto", sign_wroll="auto",
                     no_bursts=False, burst_thresh=0.55, burst_ms=110.0, burst_blend=0.55),
    "action": dict(smooth_vel="sg", sg_win_ms=180, sg_poly=3,
                   smooth_ang="lp1", lp_ang_hz=5.0, lp_vel_hz=1.5, kalman_q=0.08, kalman_r=0.4,
                   max_rate_vel=0.0, max_rate_ang=0.0, bias_tau_s=4.0,
                   dir_dead=0.12, dir_hyst_on=0.45, dir_hyst_off=0.18, flip_hold_ms=160,
                   conf_min_matches=90, conf_parallax=0.10,
                   gamma=1.10, resample_hz=150.0, flux_z_weight=1.2,
                   sign_vx="auto", sign_vy="auto", sign_vz="auto", sign_wyaw="auto", sign_wpitch="auto", sign_wroll="auto",
                   no_bursts=False, burst_thresh=0.5, burst_ms=100.0, burst_blend=0.6),
    "stabilized": dict(smooth_vel="sg", sg_win_ms=360, sg_poly=3,
                       smooth_ang="lp1", lp_ang_hz=2.0, lp_vel_hz=0.8, kalman_q=0.04, kalman_r=0.3,
                       max_rate_vel=0.0, max_rate_ang=0.0, bias_tau_s=12.0,
                       dir_dead=0.30, dir_hyst_on=0.70, dir_hyst_off=0.35, flip_hold_ms=600,
                       conf_min_matches=140, conf_parallax=0.18,
                       gamma=1.20, resample_hz=80.0, flux_z_weight=0.8,
                       sign_vx="auto", sign_vy="auto", sign_vz="auto", sign_wyaw="auto", sign_wpitch="auto", sign_wroll="auto",
                       no_bursts=True, burst_thresh=0.6, burst_ms=120.0, burst_blend=0.5),
    "vr_comfort": dict(smooth_vel="sg", sg_win_ms=320, sg_poly=3,
                       smooth_ang="lp1", lp_ang_hz=2.5, lp_vel_hz=1.0, kalman_q=0.05, kalman_r=0.3,
                       max_rate_vel=0.0, max_rate_ang=0.0, bias_tau_s=10.0,
                       dir_dead=0.25, dir_hyst_on=0.70, dir_hyst_off=0.30, flip_hold_ms=500,
                       conf_min_matches=130, conf_parallax=0.16,
                       gamma=1.20, resample_hz=100.0, flux_z_weight=1.0,
                       sign_vx="auto", sign_vy="auto", sign_vz="auto", sign_wyaw="auto", sign_wpitch="auto", sign_wroll="auto",
                       no_bursts=True, burst_thresh=0.6, burst_ms=120.0, burst_blend=0.5),
    "raw": dict(smooth_vel="lp1", lp_vel_hz=6.0, sg_win_ms=150, sg_poly=3,
                smooth_ang="lp1", lp_ang_hz=8.0, kalman_q=0.10, kalman_r=0.5,
                max_rate_vel=0.0, max_rate_ang=0.0, bias_tau_s=2.0,
                dir_dead=0.05, dir_hyst_on=0.20, dir_hyst_off=0.10, flip_hold_ms=100,
                conf_min_matches=60, conf_parallax=0.06,
                gamma=1.00, resample_hz=150.0, flux_z_weight=1.0,
                sign_vx="auto", sign_vy="auto", sign_vz="auto", sign_wyaw="auto", sign_wpitch="auto", sign_wroll="auto",
                no_bursts=True, burst_thresh=0.6, burst_ms=120.0, burst_blend=0.5),
}



# --- DUP FRAME DETECTION ---
DUP_FRAME_MAD_THRESH = 0.7  # tune if needed

def is_near_duplicate_frame(prev_gray: np.ndarray, curr_gray: np.ndarray,
                            thresh: float = DUP_FRAME_MAD_THRESH) -> bool:
    if prev_gray is None or curr_gray is None:
        return False
    if prev_gray.shape != curr_gray.shape:
        return False
    diff = cv.absdiff(prev_gray, curr_gray)
    m = float(np.mean(diff))
    return m < thresh


# --- Frame-level curvature (2nd-order intensity energy at FLOW_SCALE) ---
def frame_curvature(prev_gray: np.ndarray, gray: np.ndarray,
                    scale: float = FLOW_SCALE) -> float:
    if prev_gray is None or gray is None:
        return 0.0
    h, w = gray.shape[:2]
    W = max(8, int(round(w*scale))); H = max(8, int(round(h*scale)))
    prev_s = cv.resize(prev_gray, (W, H), cv.INTER_AREA)
    curr_s = cv.resize(gray,      (W, H), cv.INTER_AREA)

    prev_s = cv.GaussianBlur(prev_s, (3,3), 0)
    curr_s = cv.GaussianBlur(curr_s, (3,3), 0)

    Ix  = cv.Sobel(curr_s, cv.CV_32F, 1, 0, ksize=3)
    Iy  = cv.Sobel(curr_s, cv.CV_32F, 0, 1, ksize=3)
    Ixx = cv.Sobel(Ix,     cv.CV_32F, 1, 0, ksize=3)
    Iyy = cv.Sobel(Iy,     cv.CV_32F, 0, 1, ksize=3)

    It  = (curr_s.astype(np.float32) - prev_s.astype(np.float32))
    Itt = cv.Laplacian(It, cv.CV_32F, ksize=3)

    energy = np.mean(Ixx*Ixx + Iyy*Iyy + 0.5*Itt*Itt)
    return float(np.sqrt(max(energy, 0.0)))

def debug_flow_mag_and_curl(prev_gray: np.ndarray,
                            gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute dense Farneback flow, magnitude, and curl on a downscaled grid,
    then upsample to full-res for statistics.

    This uses the same FB_OPTS and FLOW_SCALE as the main pipeline, but
    NO adaptive blur – we want the raw structure for debugging.
    """
    if prev_gray is None or gray is None:
        return None, None

    h, w = gray.shape[:2]
    W = max(8, int(round(w * FLOW_SCALE)))
    H = max(8, int(round(h * FLOW_SCALE)))

    prev_s = cv.resize(prev_gray, (W, H), cv.INTER_AREA)
    curr_s = cv.resize(gray,      (W, H), cv.INTER_AREA)

    flow = cv.calcOpticalFlowFarneback(prev_s, curr_s, None, **FB_OPTS)
    fx = flow[..., 0].astype(np.float32)
    fy = flow[..., 1].astype(np.float32)

    mag = np.sqrt(fx * fx + fy * fy)

    # curl ω = ∂v/∂x − ∂u/∂y
    dv_dx = cv.Sobel(fy, cv.CV_32F, 1, 0, ksize=3) / 8.0
    du_dy = cv.Sobel(fx, cv.CV_32F, 0, 1, ksize=3) / 8.0
    curl = dv_dx - du_dy

    # Upsample to full-res for consistent per-frame stats
    mag_full  = cv.resize(mag,  (w, h), interpolation=cv.INTER_LINEAR)
    curl_full = cv.resize(curl, (w, h), interpolation=cv.INTER_LINEAR)
    return mag_full, curl_full


def divergence_from_flow(prev_gray: np.ndarray, gray: np.ndarray, scale: float = FLOW_SCALE) -> float:
    if prev_gray is None or gray is None: return 0.0
    h, w = gray.shape[:2]
    W = max(8, int(round(w*scale))); H = max(8, int(round(h*scale)))
    prev_s = cv.resize(prev_gray, (W, H), cv.INTER_AREA)
    curr_s = cv.resize(gray,      (W, H), cv.INTER_AREA)

    # --- Gaussian curvature injection (same idea as ROI tracker) ---
    prev_s = _inject_gauss_struct(prev_s)
    curr_s = _inject_gauss_struct(curr_s)
    # ---------------------------------------------------------------

    # 1) noise-adaptive, edge-aware blur
    curv_prev = _iglog_energy_frame(prev_s)
    curv_curr = _iglog_energy_frame(curr_s)

    # normalize curv to 0..1 (copy your robust 5–95% normalization style)
    lo = float(np.percentile(curv_prev, 5.0)); hi = float(np.percentile(curv_prev, 95.0))
    curv01_prev = np.clip((curv_prev - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)

    prev_s = adaptive_blur_from_curv(prev_s, curv01_prev)
    # repeat for curr if you want symmetry:
    lo = float(np.percentile(curv_curr, 5.0)); hi = float(np.percentile(curv_curr, 95.0))
    curv01_curr = np.clip((curv_curr - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)
    curr_s = adaptive_blur_from_curv(curr_s, curv01_curr)

    sigma = 0.0
    blur_strength = 0.0



    # 2) Farnebäck opts: enable GAUSSIAN + enlarge winsize when noisy
    FB = dict(FB_OPTS)
    if sigma > 0.0:
        FB["flags"]   = cv.OPTFLOW_FARNEBACK_GAUSSIAN
        FB["winsize"] = max(FB.get("winsize", 27), 27 + int(24 * min(1.0, blur_strength)))

    flow = cv.calcOpticalFlowFarneback(prev_s, curr_s, None, **FB)
    fx = flow[...,0].astype(np.float32); fy = flow[...,1].astype(np.float32)

    # 3) robust gating: keep strong + coherent vectors (as in ROI)
    mag  = np.sqrt(fx*fx + fy*fy)
    mthr = np.percentile(mag, 70.0)
    mask = (mag >= mthr)
    ang  = np.arctan2(fy, fx)
    a0   = np.median(ang[mask]) if np.count_nonzero(mask) else 0.0
    mask &= (np.cos(ang - a0) >= 0.8)  # ±36.9°

    dvx_dx = cv.Sobel(fx, cv.CV_32F, 1, 0, 3) / 8.0
    dvy_dy = cv.Sobel(fy, cv.CV_32F, 0, 1, 3) / 8.0
    div = dvx_dx + dvy_dy
    if np.count_nonzero(mask) >= 16:
        return float(np.median(div[mask]))
    return float(np.median(div))

def build_unified_flow_overlay(analyzer: 'Analyzer',
                               prev_gray: np.ndarray,
                               gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Unified global flow overlay:

      - Base:  MAG (|flow|) heatmap in VIRIDIS
      - DIV:   divergence (∂u/∂x + ∂v/∂y) in custom BWR (blue=negative, red=positive)
      - CURL:  curl (∂v/∂x - ∂u/∂y) in TWILIGHT (signed)

    All three are in true physical scale; visualization uses
    running global 95th percentile for each channel.

    Returns:
      dbg_col  : HxWx3 BGR composite overlay
      dbg_alpha: HxW float32 alpha in [0,1] for blending into GUI
    """
    if prev_gray is None or gray is None:
        return None, None

    h, w = gray.shape[:2]
    W = max(8, int(round(w * FLOW_SCALE)))
    H = max(8, int(round(h * FLOW_SCALE)))
    prev_s = cv.resize(prev_gray, (W, H), cv.INTER_AREA)
    curr_s = cv.resize(gray,      (W, H), cv.INTER_AREA)

    # Inject curvature first so debug overlay shows the real processing path
    prev_s = _inject_gauss_struct(prev_s)
    curr_s = _inject_gauss_struct(curr_s)

    # Same adaptive blur as divergence_from_flow — "with processing"
    curv_prev = _iglog_energy_frame(prev_s)
    curv_curr = _iglog_energy_frame(curr_s)

    # normalize curv to 0..1 (copy your robust 5–95% normalization style)
    lo = float(np.percentile(curv_prev, 5.0)); hi = float(np.percentile(curv_prev, 95.0))
    curv01_prev = np.clip((curv_prev - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)

    prev_s = adaptive_blur_from_curv(prev_s, curv01_prev)
    # repeat for curr if you want symmetry:
    lo = float(np.percentile(curv_curr, 5.0)); hi = float(np.percentile(curv_curr, 95.0))
    curv01_curr = np.clip((curv_curr - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)
    curr_s = adaptive_blur_from_curv(curr_s, curv01_curr)

    sigma = 0.0
    blur_strength = 0.0



    FB = dict(FB_OPTS)
    if sigma > 0.0:
        FB["flags"]   = cv.OPTFLOW_FARNEBACK_GAUSSIAN
        FB["winsize"] = max(FB.get("winsize", 27),
                            27 + int(24 * min(1.0, blur_strength)))

    flow = cv.calcOpticalFlowFarneback(prev_s, curr_s, None, **FB)
    fx = flow[..., 0].astype(np.float32)
    fy = flow[..., 1].astype(np.float32)

    # --- true fields ---
    mag  = np.sqrt(fx * fx + fy * fy)
    dvx_dx = cv.Sobel(fx, cv.CV_32F, 1, 0, ksize=3) / 8.0
    dvy_dy = cv.Sobel(fy, cv.CV_32F, 0, 1, ksize=3) / 8.0
    div = dvx_dx + dvy_dy
    dv_dx = cv.Sobel(fy, cv.CV_32F, 1, 0, ksize=3) / 8.0
    du_dy = cv.Sobel(fx, cv.CV_32F, 0, 1, ksize=3) / 8.0
    curl = dv_dx - du_dy

    # --- update global 95th percentiles (approx "global normalization") ---
    def _update_p95(current: float, arr: np.ndarray) -> float:
        arr_abs = np.abs(arr)
        if arr_abs.size == 0:
            return current
        p = float(np.percentile(arr_abs, 95.0))
        if not np.isfinite(p) or p <= 1e-9:
            return current
        return max(current, p)

    analyzer._dbg_mag95  = _update_p95(analyzer._dbg_mag95,  mag)
    analyzer._dbg_div95  = _update_p95(analyzer._dbg_div95,  div)
    analyzer._dbg_curl95 = _update_p95(analyzer._dbg_curl95, curl)

    mag95  = analyzer._dbg_mag95
    div95  = analyzer._dbg_div95
    curl95 = analyzer._dbg_curl95

    # --- normalized fields for visualization ONLY (data is still true-scale) ---
    mag_n  = np.clip(mag / (mag95 + 1e-12),  0.0, 1.0).astype(np.float32)
    div_s  = np.clip(div / (div95 + 1e-12), -1.0, 1.0).astype(np.float32)
    curl_s = np.clip(curl / (curl95 + 1e-12), -1.0, 1.0).astype(np.float32)

    # base MAG color (0..1 -> 0..255)
    mag_u8 = (mag_n * 255.0).astype(np.uint8)
    col_mag_s = cv.applyColorMap(mag_u8, cv.COLORMAP_VIRIDIS)

    # --- custom BWR for DIV: -1 -> blue, 0 -> white, +1 -> red ---
    # div_s in [-1,1]
    t = (div_s * 0.5 + 0.5)[..., None]  # [-1,1] -> [0,1], shape HxWx1
    # blue to white for t in [0,0.5], white to red for t in (0.5,1]
    # Start blue=(255,0,0), white=(255,255,255), red=(0,0,255) in BGR
    col_div_s = np.zeros((H, W, 3), np.float32)
    left_mask  = (t <= 0.5).astype(np.float32)
    right_mask = 1.0 - left_mask

    # left: blue -> white
    tl = np.clip(t * 2.0, 0.0, 1.0)  # [0,0.5]->[0,1]
    # BGR: blue=(255,0,0) to white=(255,255,255)
    col_div_s += left_mask * np.concatenate([
        255.0 * np.ones_like(tl),          # B stays 255
        255.0 * tl,                        # G 0->255
        255.0 * tl                         # R 0->255
    ], axis=2)

    # right: white -> red
    tr = np.clip((t - 0.5) * 2.0, 0.0, 1.0)  # (0.5,1]->[0,1]
    # BGR: white=(255,255,255) to red=(0,0,255)
    col_div_s += right_mask * np.concatenate([
        255.0 * (1.0 - tr),                # B 255->0
        255.0 * (1.0 - tr),                # G 255->0
        255.0 * np.ones_like(tr),         # R stays 255
    ], axis=2)

    col_div_s = np.clip(col_div_s, 0.0, 255.0).astype(np.uint8)

    # CURL color (signed, TWILIGHT)
    curl_u8 = ((curl_s * 0.5 + 0.5) * 255.0).astype(np.uint8)
    col_curl_s = cv.applyColorMap(curl_u8, cv.COLORMAP_TWILIGHT)

    # upsample to full resolution
    col_mag   = cv.resize(col_mag_s,   (w, h), interpolation=cv.INTER_LINEAR)
    col_div   = cv.resize(col_div_s,   (w, h), interpolation=cv.INTER_LINEAR)
    col_curl  = cv.resize(col_curl_s,  (w, h), interpolation=cv.INTER_LINEAR)
    mag_n_full    = cv.resize(mag_n,        (w, h), interpolation=cv.INTER_LINEAR)
    div_abs_full  = cv.resize(np.abs(div_s),(w, h), interpolation=cv.INTER_LINEAR)
    curl_abs_full = cv.resize(np.abs(curl_s),(w, h), interpolation=cv.INTER_LINEAR)

    # --- composite:
    # base = MAG; then DIV; then CURL on top
    base     = col_mag.astype(np.float32)  / 255.0
    div_col  = col_div.astype(np.float32)  / 255.0
    curl_col = col_curl.astype(np.float32) / 255.0

    # per-pixel alpha: small values barely show; large values penetrate
    a_div  = (div_abs_full  ** 0.8) * 0.7
    a_curl = (curl_abs_full ** 0.8) * 0.7

    a_div  = np.clip(a_div,  0.0, 1.0)[..., None]
    a_curl = np.clip(a_curl, 0.0, 1.0)[..., None]

    comp = base * (1.0 - a_div) + div_col * a_div
    comp = comp * (1.0 - a_curl) + curl_col * a_curl

    dbg_col = (np.clip(comp, 0.0, 1.0) * 255.0).astype(np.uint8)

    # final overlay alpha driven by MAG (so static areas don’t get painted)
    dbg_alpha = (mag_n_full ** 0.7) * 0.85
    dbg_alpha = np.clip(dbg_alpha, 0.0, 1.0).astype(np.float32)

    return dbg_col, dbg_alpha




def make_flow_debug(prev_gray, gray):
    """
    Debug panel for Ctrl+D.

    Left:  flow magnitude heatmap (JET).
    Right: curl (rotation) heatmap (TWILIGHT, signed).
    Output: combined image (H x 2W x 3).
    """
    if prev_gray is None or gray is None:
        return None

    h, w = gray.shape[:2]

    # Use same scale + adaptive blur logic as divergence_from_flow
    W = max(8, int(round(w * FLOW_SCALE)))
    H = max(8, int(round(h * FLOW_SCALE)))
    prev_s = cv.resize(prev_gray, (W, H), cv.INTER_AREA)
    curr_s = cv.resize(gray,      (W, H), cv.INTER_AREA)

    prev_s = _inject_gauss_struct(prev_s)
    curr_s = _inject_gauss_struct(curr_s)

    curv_prev = _iglog_energy_frame(prev_s)
    curv_curr = _iglog_energy_frame(curr_s)

    # normalize curv to 0..1 (copy your robust 5–95% normalization style)
    lo = float(np.percentile(curv_prev, 5.0)); hi = float(np.percentile(curv_prev, 95.0))
    curv01_prev = np.clip((curv_prev - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)

    prev_s = adaptive_blur_from_curv(prev_s, curv01_prev)
    # repeat for curr if you want symmetry:
    lo = float(np.percentile(curv_curr, 5.0)); hi = float(np.percentile(curv_curr, 95.0))
    curv01_curr = np.clip((curv_curr - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)
    curr_s = adaptive_blur_from_curv(curr_s, curv01_curr)

    sigma = 0.0
    blur_strength = 0.0



    FB = dict(FB_OPTS)
    if sigma > 0.0:
        FB["flags"]   = cv.OPTFLOW_FARNEBACK_GAUSSIAN
        FB["winsize"] = max(FB.get("winsize", 27), 27 + int(24 * min(1.0, blur_strength)))

    flow = cv.calcOpticalFlowFarneback(prev_s, curr_s, None, **FB)
    fx = flow[..., 0].astype(np.float32)
    fy = flow[..., 1].astype(np.float32)

    # --- magnitude heatmap ---
    mag = np.sqrt(fx * fx + fy * fy)
    mag_norm = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    heat_mag = cv.applyColorMap(mag_norm, cv.COLORMAP_JET)

    # --- curl heatmap (rotation): ω = ∂v/∂x - ∂u/∂y ---
    dv_dx = cv.Sobel(fy, cv.CV_32F, 1, 0, ksize=3)
    du_dy = cv.Sobel(fx, cv.CV_32F, 0, 1, ksize=3)
    curl = dv_dx - du_dy

    # Signed normalization: -max → 0, 0 → 127, +max → 255
    cmin, cmax = float(np.min(curl)), float(np.max(curl))
    cabs = max(abs(cmin), abs(cmax), 1e-6)
    curl_scaled = ((curl / (2.0 * cabs)) + 0.5)  # [-1,1] -> [0,1]
    curl_u8 = np.clip(curl_scaled * 255.0, 0, 255).astype(np.uint8)
    heat_curl = cv.applyColorMap(curl_u8, cv.COLORMAP_TWILIGHT)

    # Combine side-by-side: [ mag | curl ]
    panel = np.hstack([heat_mag, heat_curl])
    return panel



def even2(n: int) -> int: return n if (n % 2 == 0) else (n - 1 if n > 1 else 2)

# ---------- text (auto-fit; no overflow) ----------
def _text_autoscale(img, txt, scale, margin=10):
    max_w = img.shape[1] - 2*margin
    s = float(scale)
    (w, h), _ = cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, s, 2)
    if w > max_w:
        s = max(0.35, s * (max_w / max(1, w)))
    return s

def draw_text(img, txt, y, color=(255,255,255), scale=0.6, margin=10):
    s = _text_autoscale(img, txt, scale, margin)
    cv.putText(img, txt, (margin, y), cv.FONT_HERSHEY_SIMPLEX, s, (0,0,0), 2, cv.LINE_AA)
    cv.putText(img, txt, (margin, y), cv.FONT_HERSHEY_SIMPLEX, s, color, 1, cv.LINE_AA)

def draw_text_right(img, txt, y, color=(230,230,230), scale=0.6, margin=10):
    s = _text_autoscale(img, txt, scale, margin)
    (w, h), _ = cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, s, 2)
    x = img.shape[1] - margin - w
    cv.putText(img, txt, (x, y), cv.FONT_HERSHEY_SIMPLEX, s, (0,0,0), 2, cv.LINE_AA)
    cv.putText(img, txt, (x, y), cv.FONT_HERSHEY_SIMPLEX, s, color, 1, cv.LINE_AA)

def rotmat_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0]); singular = sy < 1e-6
    if not singular:
        roll  = math.degrees(math.atan2(R[2,1], R[2,2]))
        pitch = math.degrees(math.atan2(-R[2,0], sy))
        yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))
    else:
        roll  = math.degrees(math.atan2(-R[1,2], R[1,1])); pitch = math.degrees(math.atan2(-R[2,0], sy)); yaw = 0.0
    return yaw, pitch, roll

@dataclass
class FrameRow:
    frame: int; time: float
    dx: float; dy: float; rot2d_deg: float; scale2d: float; trans_mag: float
    yaw_deg: float; pitch_deg: float; roll_deg: float
    tx: float; ty: float; tz: float; tnorm: float
    orb_matches: int; E_inliers: int; H_inliers: int; parallax: float
    vx_px_s: float = 0.0; vy_px_s: float = 0.0; vz_rel_s: float = 0.0
    wyaw_deg_s: float = 0.0; wpitch_deg_s: float = 0.0; wroll_deg_s: float = 0.0
    speed2d_px_s: float = 0.0; speed3_rel: float = 0.0
    flux: float = 0.0; acc: float = 0.0; jerk: float = 0.0
    divz: float = 0.0
    curv: float = 0.0  # NEW: frame-level curvature (intensity 2nd-order energy)
    vz25: float = 0.0  # NEW: frame-level 2.5D curvature-depth (per-frame units)

    

# ---------- smoothing ----------
def lp1(x, fps, cutoff_hz):
    if cutoff_hz <= 0: return x.copy()
    a = 2*math.pi*cutoff_hz / (fps + 2*math.pi*cutoff_hz)
    y = np.empty_like(x, dtype=np.float64); y[0] = x[0]
    for i in range(1, len(x)): y[i] = y[i-1] + a*(x[i]-y[i-1])
    return y
def gauss_blur1d(z, sigma):
    """
    Simple 1D Gaussian blur for control-rate signals.
    sigma is in *samples* (frames), not seconds.
    """
    z = np.asarray(z, np.float64)
    if z.size == 0:
        return z
    sigma = float(max(1e-6, sigma))
    R = int(np.ceil(3.0 * sigma))
    if R <= 0:
        return z.copy()
    kx = np.arange(-R, R + 1, dtype=np.float64)
    k = np.exp(-0.5 * (kx / sigma) ** 2)
    k /= (np.sum(k) + 1e-12)
    zp = np.pad(z, (R, R), mode="reflect")
    return np.convolve(zp, k, mode="valid")


def robust_z(x, clip=4.0):
    """
    Median/MAD z-score, clipped to [-clip, clip].
    Keeps structure, normalizes amplitude, robust to outliers.
    """
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return x
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-12
    z = (x - med) / (1.4826 * mad)
    return np.clip(z, -clip, clip)


def log1d_series(x, fps, sigma_ms=80.0):
    """
    1D LoG in time: Gaussian smoothing + 2nd temporal derivative.
    x       : 1D array over frames
    fps     : frame rate
    sigma_ms: Gaussian sigma in milliseconds (temporal scale of interest)
    """
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return x

    dt = 1.0 / max(1e-6, float(fps))
    sigma_frames = max(1.0, (sigma_ms / 1000.0) * fps)

    x_blur = gauss_blur1d(x, sigma_frames)
    d1 = np.gradient(x_blur, dt, edge_order=2)
    d2 = np.gradient(d1,      dt, edge_order=2)
    return d2


def savgol_coeffs(window: int, poly: int):
    if window % 2 == 0: window += 1
    window = max(window, poly + 2 if (poly+2)%2==1 else poly+3)
    half = window // 2; x = np.arange(-half, half+1, dtype=np.float64)
    A = np.vander(x, poly+1, increasing=True); pinv = np.linalg.pinv(A)
    return pinv[0]

def savgol_smooth(x, fps, win_ms=200.0, poly=3):
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n <= 2:
        return x.copy()  # nothing meaningful to smooth; avoid crashes

    # target window from milliseconds → frames
    win = int(round((win_ms / 1000.0) * float(fps)))

    # enforce odd window and minimum for given poly
    if win % 2 == 0: win += 1
    win = max(win, poly + 2 if ((poly + 2) % 2 == 1) else (poly + 3))

    # CRITICAL: mirrorable maximum so xp[i:i+win] always has length==win
    win = min(win, 2 * n - 1)
    if win % 2 == 0: win -= 1  # keep odd after clamping

    # if data is very short, reduce polynomial to keep (win >= poly+2)
    poly = min(poly, win - 2)

    half = win // 2
    c = savgol_coeffs(win, poly)  # unchanged API; returns length==win

    # mirror padding now guaranteed to produce exactly 'half' samples each side
    padL = x[1:half+1][::-1]
    padR = x[-half-1:-1][::-1] if half > 0 else np.array([], dtype=x.dtype)
    xp = np.concatenate([padL, x, padR])

    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = np.dot(c, xp[i:i+win])  # shapes now always align
    return y


def kalman_cv(x, fps, q=0.05, r=0.3):
    dt = 1.0 / max(1e-6, fps)
    F = np.array([[1.0, dt],[0.0, 1.0]], dtype=np.float64); H = np.array([[1.0, 0.0]], dtype=np.float64)
    Q = np.array([[q*dt*dt*dt/3.0, q*dt*dt/2.0],[q*dt*dt/2.0, q*dt]], dtype=np.float64); R = np.array([[r*r]], dtype=np.float64)
    x_state = np.array([[x[0]],[0.0]], dtype=np.float64); P = np.eye(2, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        x_state = F @ x_state; P = F @ P @ F.T + Q
        z = np.array([[x[i]]], dtype=np.float64); yv = z - (H @ x_state)
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        x_state = x_state + K @ yv; P = (np.eye(2) - K @ H) @ P; out[i] = x_state[0,0]
    return out

def rate_limit(x, fps, max_rate_per_s):
    if max_rate_per_s <= 0: return x
    dx_max = max_rate_per_s / fps; y = np.empty_like(x, dtype=np.float64); y[0] = x[0]
    for i in range(1, len(x)):
        step = x[i] - y[i-1]; step = min(dx_max, max(-dx_max, step)); y[i] = y[i-1] + step
    return y

def clip_spikes(x, thresh_sigma=3.5):
    if len(x) < 3: return x
    dx = np.diff(x, prepend=x[0]); std = np.std(dx) or 1.0; y = x.copy().astype(np.float64)
    for i in range(1, len(x)): y[i] = y[i-1] if abs(dx[i]) > thresh_sigma*std else x[i]
    return y

def interpolate_over_dups_1d(values: np.ndarray, dup_flags: List[bool]) -> np.ndarray:
    v = np.asarray(values, np.float64)
    d = np.asarray(dup_flags, bool)
    n = len(v)
    if n == 0 or len(d) != n:
        return v

    out = v.copy()
    i = 0
    while i < n:
        if not d[i]:
            i += 1
            continue
        start = i - 1
        j = i
        while j < n and d[j]:
            j += 1
        end = j

        v0 = out[start] if start >= 0 else out[end-1]
        v1 = out[end]   if end < n   else out[start+1]

        span = end - start
        if span <= 0:
            i = j
            continue

        for k in range(start+1, end):
            alpha = (k - start) / span
            out[k] = (1.0 - alpha)*v0 + alpha*v1

        i = j

    return out


def smooth_series(x, fps, method, lp_hz, sg_win_ms, sg_poly, k_q, k_r, max_rate=0.0):
    x = clip_spikes(x, 3.5)
    if method == "sg": y = savgol_smooth(x, fps, win_ms=sg_win_ms, poly=sg_poly)
    elif method == "kalman": y = kalman_cv(x, fps, q=k_q, r=k_r)
    else: y = lp1(x, fps, cutoff_hz=lp_hz)
    if max_rate > 0: y = rate_limit(y, fps, max_rate)
    return y

from collections import deque
# ---------- FFmpeg writer ----------
class FFMpegWriter:
    def __init__(self, out_path: Path, fps: float, W: int, H: int,
                 ffmpeg_bin: str = "ffmpeg",
                 vcodec: str = "libx264", crf: float = 18.0, preset: str = "medium",
                 pix_fmt_out: str = "yuv420p", extra_args: Optional[List[str]] = None):
        self.W, self.H = int(W), int(H)
        self.fps = float(fps)
        self.out_path = Path(out_path)
        self.ffmpeg_bin = shutil.which(ffmpeg_bin) or ffmpeg_bin
        if not shutil.which(self.ffmpeg_bin):
            raise RuntimeError(f"ffmpeg not found: {ffmpeg_bin}")
        args = [self.ffmpeg_bin, "-y",
                "-hide_banner", "-loglevel", "error", "-nostdin",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.W}x{self.H}",
                "-r", f"{self.fps:.6f}", "-i", "-",
                "-an",
                "-c:v", vcodec]
        if vcodec in ("libx264","libx265","libsvtav1"):
            args += ["-preset", preset]
            if vcodec != "libsvtav1": args += ["-crf", str(int(round(crf)))]
            args += ["-pix_fmt", pix_fmt_out]
        elif vcodec.startswith("prores"):
            if vcodec != "prores_ks":
                vcodec = "prores_ks"; args[args.index("-c:v")+1] = vcodec
            args += ["-profile:v","3","-pix_fmt","yuv422p10le"]
        if extra_args: args += list(extra_args)
        args += [str(self.out_path)]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        self._stderr_buf = deque(maxlen=64)
        def _drain_out(pipe):
            try:
                while pipe.read(65536):
                    pass
            except Exception:
                pass

        def _drain_err(pipe):
            try:
                while True:
                    chunk = pipe.read(4096)
                    if not chunk:
                        break
                    self._stderr_buf.append(chunk)
            except Exception:
                pass

        self._t_out = threading.Thread(target=_drain_out, args=(self.proc.stdout,), daemon=True); self._t_out.start()
        self._t_err = threading.Thread(target=_drain_err, args=(self.proc.stderr,), daemon=True); self._t_err.start()

    def _stderr_tail(self, max_bytes=32768):
        data = b"".join(list(self._stderr_buf))
        if len(data) > max_bytes:
            data = data[-max_bytes:]
        return data.decode("utf-8", "ignore")

    def write(self, frame_bgr: np.ndarray):
        if frame_bgr.shape[1] != self.W or frame_bgr.shape[0] != self.H:
            raise ValueError(f"Frame size mismatch: got {frame_bgr.shape[1]}x{frame_bgr.shape[0]}, expected {self.W}x{self.H}")
        # fail fast if ffmpeg exited
        if self.proc.poll() is not None:
            raise RuntimeError(f"ffmpeg terminated (rc={self.proc.returncode}).\n--- ffmpeg stderr tail ---\n{self._stderr_tail()}")
        try:
            self.proc.stdin.write(np.asarray(frame_bgr, dtype=np.uint8).tobytes())
        except BrokenPipeError:
            raise RuntimeError(f"ffmpeg pipe closed.\n--- ffmpeg stderr tail ---\n{self._stderr_tail()}")

    def close(self):
        try:
            if self.proc and self.proc.stdin:
                self.proc.stdin.close()
        finally:
            if self.proc:
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.proc.kill(); self.proc.wait()

# ---------- Analyzer ----------
class Analyzer:
    def __init__(self, path: Path, args, out_dir: Path, write_annotated: bool = True, show_window: bool = False):
        self.args = args; self.path = Path(path); self.out_dir = Path(out_dir); self.out_dir.mkdir(exist_ok=True)
        self.cap = cv.VideoCapture(str(self.path)); assert self.cap.isOpened(), f"Cannot open: {path}"
        self.fps = float(self.cap.get(cv.CAP_PROP_FPS) or 30.0)
        self.W = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)); self.H = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.N = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        f = 0.9 * max(self.W, self.H)
        self.K = np.array([[f,0,self.W/2.0],[0,f,self.H/2.0],[0,0,1]], dtype=np.float64)
        self.orb = cv.ORB_create(nfeatures=ORB_NFEATURES, scaleFactor=1.2, nlevels=8, edgeThreshold=16, patchSize=31, fastThreshold=7)
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        self._burnin_frames = int(float(getattr(self.args, "burnin_sec", 2.0)) * self.fps)
        self._frame_i = 0
        self._did_restart = False

        self._cut_hold = 0


        self.rows: List[FrameRow] = []
        self.g_trans=[]; self.g_rot2d=[]; self.g_roll=[]; self.g_parax=[]
        self._ov_hist = {}

        # global motion backend (translation/pose split):
        #   Hybrid (default): HG provides dx/dy; ORB provides rot/scale + 3D correspondences (no competition on translation)
        #   ORB:  ORB provides dx/dy + rot/scale (+3D)
        #   HG:   HG provides dx/dy only (pose neutral)
        mm = str(getattr(self.args, "mode", "Hybrid") or "Hybrid").strip().lower()
        if mm in ("fb", "feature", "features", "orb"):
            mm = "orb"
        elif mm in ("hyb", "hybrid", "mix"):
            mm = "hybrid"
        elif mm in ("hg", "hypergraph"):
            mm = "hg"
        else:
            mm = "hybrid"
        self.motion_mode = mm
        self._hg_state: dict = {}

        self._hyb_alpha = 0.0
        self._hg_state: dict = {}
        self.write_annotated = bool(write_annotated); self.show_window = bool(show_window)
        self.outW = even2(self.W); self.outH = even2(self.H + HUD_BAND_H + GRAPH_H)
        stem = self.path.stem
        self.out_raw = self.out_dir / f"{stem}_camera_motion_raw.csv"
        self.out_mod = self.out_dir / f"{stem}_modulators.csv"
        self.out_mp4 = self.out_dir / f"{stem}_analyzed_ffmpeg.mp4"
        self.out_mp4_robust = self.out_dir / f"{stem}_analyzed_ffmpeg_robust.mp4"
        self.sx_dim = self.W / 2.0
        self.sy_dim = self.H / 2.0
        self.sxy_dim = min(self.W, self.H) / 2.0

        # live overlay smoothing
        self.live_lp_hz = float(getattr(self.args, "live_lp_hz", self.args.lp_vel_hz))
        self.vx_live = self.vy_live = self.vz_live = 0.0
        self._live_initialized = False

        # 3D vector XY global scale warm-up/lock for first pass
        self._sxy_running = 1.0
        self._sxy_lock = None
        self.v3_lock_s = float(getattr(self.args, "v3_lock_s", 1.0))

        # robust export opts
        self.robust_export = bool(getattr(self.args, "robust_export", False))
        self.robust_z = float(getattr(self.args, "robust_z", 6.0))
        self.robust_prctl = float(getattr(self.args, "robust_prctl", 95.0))
        self.robust_gamma = float(getattr(self.args, "robust_gamma", 1.0))

        self.vw = None
        if self.write_annotated:
            self.vw = FFMpegWriter(self.out_mp4, fps=self.fps, W=self.outW, H=self.outH,
                                   ffmpeg_bin=args.ffmpeg_bin, vcodec=args.vcodec, crf=args.crf,
                                   preset=args.preset, pix_fmt_out=args.pix_fmt)
        if self.show_window:
            cv.namedWindow("Camera Analyzer v5F", cv.WINDOW_NORMAL)
            cv.resizeWindow("Camera Analyzer v5F", self.W, self.H + HUD_BAND_H + GRAPH_H)

        # --- PATCH 1: seed ORB cache (Analyzer.__init__) ---
        ok, prev = self.cap.read(); assert ok, "No frames in input."
        self.prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY); self.frame_idx = 0
        # ORB on downscaled image for speed; scale keypoints back to full-res coords
        oh, ow = self.prev_gray.shape[:2]
        oW = max(32, int(round(ow * ORB_SCALE)))
        oH = max(32, int(round(oh * ORB_SCALE)))
        self._orb_sx = ow / float(oW)
        self._orb_sy = oh / float(oH)
        prev_orb = cv.resize(self.prev_gray, (oW, oH), cv.INTER_AREA) if ORB_SCALE != 1.0 else self.prev_gray
        self.kp_prev, self.des_prev = _orb_struct_detect_and_compute(self.orb, prev_orb, max_kp=ORB_NFEATURES)  # cache


        self._fq = Queue(maxsize=8)
        self.dup_flags: List[bool] = []  # per-frame duplicate flags aligned with rows

        # debug: unified global flow overlay (0=OFF, 1=ON)
        self.debug_flow_mode = 0

        # debug stats (per-frame mag/curl CSV)
        self.flow_debug_rows = []  # list of tuples (frame_idx, time_s, mean_mag, mean_abs_curl, corr)

        # global display scales (approx global 95th percentiles)
        self._dbg_mag95  = 1e-6
        self._dbg_div95  = 1e-6
        self._dbg_curl95 = 1e-6



        def _reader():
            while True:
                ok, fr = self.cap.read()
                if not ok: self._fq.put(None); break
                self._fq.put(fr)

        self._reader_t = threading.Thread(target=_reader, daemon=True); self._reader_t.start()

    def _finite_or(self, x, default=0.0):
        try:
            x = float(x)
        except Exception:
            return float(default)
        return x if np.isfinite(x) else float(default)

        
    def _compute_divergence_map(self, prev_gray, gray):
        h, w = gray.shape[:2]
        W = max(8, int(round(w * FLOW_SCALE)))
        H = max(8, int(round(h * FLOW_SCALE)))

        prev_s = cv.resize(prev_gray, (W, H), cv.INTER_AREA)
        curr_s = cv.resize(gray,      (W, H), cv.INTER_AREA)
        flow = cv.calcOpticalFlowFarneback(prev_s, curr_s, None, **FB_OPTS)

        fx = flow[...,0].astype(np.float32)
        fy = flow[...,1].astype(np.float32)

        dvx_dx = cv.Sobel(fx, cv.CV_32F, 1, 0, 3) / 8.0
        dvy_dy = cv.Sobel(fy, cv.CV_32F, 0, 1, 3) / 8.0

        div = dvx_dx + dvy_dy
        return cv.resize(div, (w, h), interpolation=cv.INTER_LINEAR)


    def _apply_signs(self, vx, vy, vz):
        if getattr(self.args, "invert_x", False): vx = -vx
        if getattr(self.args, "invert_y", False): vy = -vy
        return vx, vy, vz

    # ---- overlay runtime helpers ----
    def _running_p95_norm(self, key: str, val: float, nmax: int = 600):
        val = self._finite_or(val, 0.0)
        buf = self._ov_hist.setdefault(key, [])
        buf.append(float(val))
        if len(buf) > nmax:
            del buf[0]

        a = np.asarray(buf, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return 0.0

        p95 = float(np.percentile(np.abs(a), 95.0))
        if not np.isfinite(p95) or p95 <= 1e-6:
            p95 = 1.0
        return float(np.clip(val / p95, -1.0, 1.0))


    def _lp1_step(self, y_prev, x, cutoff_hz):
        a = 2*math.pi*cutoff_hz / (self.fps + 2*math.pi*cutoff_hz)
        return y_prev + a*(x - y_prev)
    

    def estimate_2d_cached(self, gray, keep=800):
        # prev is cached on self.{kp_prev, des_prev}
        oh, ow = gray.shape[:2]
        oW = max(32, int(round(ow * ORB_SCALE)))
        oH = max(32, int(round(oh * ORB_SCALE)))
        sx = ow / float(oW)
        sy = oh / float(oH)
        orb_img = cv.resize(gray, (oW, oH), cv.INTER_AREA) if ORB_SCALE != 1.0 else gray
        kp2, des2 = _orb_struct_detect_and_compute(self.orb, orb_img, max_kp=ORB_NFEATURES)
        if self.des_prev is None or des2 is None:
            self.kp_prev, self.des_prev = kp2, des2
            return 0.0, 0.0, 0.0, 1.0, 0, None, None

        matches = self.bf.match(self.des_prev, des2)   # crossCheck=True already set
        # partial sort == same top-K as full sort, cheaper than sorting all M
        best = heapq.nsmallest(keep, matches, key=lambda m: m.distance)

        # NEW: minimum correspondences for a stable affine
        if best is None or len(best) < 12:
            self.kp_prev, self.des_prev = kp2, des2
            return 0.0, 0.0, 0.0, 1.0, (0 if best is None else len(best)), None, None

        pts1 = np.float32([self.kp_prev[m.queryIdx].pt for m in best]).reshape(-1,1,2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in best]).reshape(-1,1,2)
        if ORB_SCALE != 1.0:
            pts1[..., 0] *= sx; pts1[..., 1] *= sy
            pts2[..., 0] *= sx; pts2[..., 1] *= sy

        A, _ = cv.estimateAffinePartial2D(pts1, pts2, method=cv.RANSAC,
                                        ransacReprojThreshold=RANSAC_THRESH,
                                        maxIters=AFF_MAXIT, confidence=AFF_CONF)
        # NEW: reject NaN/Inf matrices (OpenCV can do this on degenerate inputs)
        if A is None or (not np.isfinite(A).all()):
            self.kp_prev, self.des_prev = kp2, des2
            return 0.0, 0.0, 0.0, 1.0, len(best), pts1, pts2

        a, b = float(A[0,0]), float(A[0,1]); dx, dy = float(A[0,2]), float(A[1,2])
        scale = float(math.sqrt(max(1e-12, a*a + b*b)))
        rot2d = float(math.degrees(math.atan2(b, a)))

        self.kp_prev, self.des_prev = kp2, des2      # move window: curr → prev
        return dx, dy, rot2d, scale, len(best), pts1, pts2


    # ---- 3D vector widget (ADDITIVE overlay) ----
    def draw_vector3_widget(self, img: np.ndarray, vx_ps: float, vy_ps: float, vz_ps: float,
                            normalizer: Optional['RobustArrowNormalizer']=None, sxy_pixels: Optional[float]=None):
        H, W = img.shape[:2]; cx, cy = W // 2, H // 2

        # XY normalized by ONE global pixel scale (not per-axis)
        if sxy_pixels is None:
            sxy = max(1e-6, self._ov_hist.get('ov3_sxy_max', 1.0))
        else:
            sxy = max(1e-6, float(sxy_pixels))
        nx = float(np.clip(vx_ps / sxy, -1.0, 1.0))
        ny = float(np.clip(vy_ps / sxy, -1.0, 1.0))

        # Z robust-normalized for the tip indicator only
        if normalizer is None:
            nz = self._running_p95_norm('ov3_vz', vz_ps)
        else:
            nz = normalizer.vz(vz_ps)

        vmag = float(np.clip(math.sqrt(nx*nx + ny*ny + nz*nz), 0.0, 1.0))
        if vmag < 1e-6: return

        k = 0.40 * min(W, H)
        ex = int(round(cx + k * nx)); ey = int(round(cy + k * ny))
        th = max(2, int(2 + 10 * vmag))

        cv.arrowedLine(img, (cx, cy), (ex, ey), (255,255,255), th, tipLength=0.18)
        cv.arrowedLine(img, (cx+1, cy+1), (ex+1, ey+1), (0,0,0), max(1, th//2), tipLength=0.18)

        zmag = abs(nz); r = max(4, int(6 + 16 * zmag))
        if nz >= 0:
            cv.circle(img, (ex, ey), r, (0,210,0), -1, cv.LINE_AA)       # toward
            cv.circle(img, (ex, ey), r, (20,20,20), 1, cv.LINE_AA)
        else:
            cv.circle(img, (ex, ey), r, (255,0,255), 2, cv.LINE_AA)      # away
            s = int(r*0.70); c = (255,0,255); lw = max(2, th//3)
            cv.line(img, (ex - s, ey - s), (ex + s, ey + s), c, lw, cv.LINE_AA)
            cv.line(img, (ex - s, ey + s), (ex + s, ey - s), c, lw, cv.LINE_AA)

        az = math.degrees(math.atan2(ny, nx)) if (abs(nx)+abs(ny))>1e-9 else 0.0
        el = math.degrees(math.atan2(nz, math.hypot(nx, ny)))
        # draw_text_right(img, f"V3 |n|={vmag:0.3f}  az={az:+.1f}°  el={el:+.1f}°", 50)

    # ---- legacy arrows + ring + graphs + 3D vector ----
    def draw_overlay_symbols(self, frame_bgr: np.ndarray, vx_ps: float, vy_ps: float, vz_ps: float, flux: float,
                             px: Optional[float]=None, py: Optional[float]=None, pz: Optional[float]=None,
                             normalizer: Optional['RobustArrowNormalizer']=None, sxy_pixels: Optional[float]=None):
        H, W = frame_bgr.shape[:2]; cx, cy = W // 2, H // 2

        # ---- finite firewall (prevents NaN -> int() crashes) ----
        vx_ps = self._finite_or(vx_ps, 0.0)
        vy_ps = self._finite_or(vy_ps, 0.0)
        vz_ps = self._finite_or(vz_ps, 0.0)
        flux  = self._finite_or(flux,  0.0)

        # XY by geometry; Z still robust
        nx = float(np.clip(vx_ps / max(1e-9, self.sx_dim), -1.0, 1.0))
        ny = float(np.clip(vy_ps / max(1e-9, self.sy_dim), -1.0, 1.0))

        # Guard nz too (normalizer path can propagate NaN)
        nz = normalizer.vz(vz_ps) if normalizer is not None else self._running_p95_norm('ov_vz', vz_ps)
        nz = self._finite_or(nz, 0.0)


        nf = np.clip(self._running_p95_norm('ov_flux', flux), 0.0, 1.0)
        C_X=(255,255,0); C_Y=(0,165,255); C_Z=(255,0,255); C_FL=(0,210,0)

        # X arrow
        Lx = int(0.35 * W * abs(nx)); thx = max(2, int(2 + 6 * abs(nx)))
        if Lx > 0:
            dirx = 1 if nx >= 0 else -1; p1=(cx - dirx*10, cy); p2=(cx + dirx*Lx, cy)
            # cv.arrowedLine(frame_bgr, p1, p2, C_X, thx, tipLength=0.2)
            draw_text(frame_bgr, f"X {vx_ps:+.0f} px/s", 28, C_X, 0.65)
        # Y arrow
        Ly = int(0.35 * H * abs(ny)); thy = max(2, int(2 + 6 * abs(ny)))
        if Ly > 0:
            diry = 1 if ny >= 0 else -1; p1=(cx, cy - diry*10); p2=(cx, cy + diry*Ly)
            # cv.arrowedLine(frame_bgr, p1, p2, C_Y, thy, tipLength=0.2)
            draw_text(frame_bgr, f"Y {vy_ps:+.0f} px/s", 52, C_Y, 0.65)

        # Z ring + cardinal ticks (out/in by sign)
        R0 = int(0.08 * min(W, H)); Rz = int(R0 + 0.30 * min(W, H) * abs(nz)); thz = max(2, int(2 + 5 * abs(nz)))
        # cv.circle(frame_bgr, (cx, cy), Rz, C_Z, thz, cv.LINE_AA)
        for ang in (0, 90, 180, 270):
            rad = math.radians(ang); s = int(Rz * 0.70); e = int(Rz * 1.05)
            sx = int(cx + s*math.cos(rad)); sy = int(cy + s*math.sin(rad)); ex = int(cx + e*math.cos(rad)); ey = int(cy + e*math.sin(rad))
            # if nz >= 0: cv.arrowedLine(frame_bgr, (sx, sy), (ex, ey), C_Z, max(1, thz-1), tipLength=0.45)
            # else:       cv.arrowedLine(frame_bgr, (ex, ey), (sx, sy), C_Z, max(1, thz-1), tipLength=0.45)

        # Z readout — 4 decimals, auto-fit
        draw_text(frame_bgr, f"Z {vz_ps:+.4f} rel/s", 76, C_Z, 0.65)

        # Flux outer ring
        # Rf = int(0.46 * min(W, H)); thf = int(2 + 14 * nf); cv.circle(frame_bgr, (cx, cy), Rf, C_FL, thf, cv.LINE_AA)
        draw_text(frame_bgr, f"Flux {flux:.0f}", 100, C_FL, 0.65)

        # POS summary upper-right (when provided)
        # if px is not None and py is not None and pz is not None:
            # draw_text_right(frame_bgr, f"POS X:{px:+.1f}  Y:{py:+.1f}  Z:{pz:+.1f}", 26)

        # Added 3D vector widget with global XY scale
        self.draw_vector3_widget(frame_bgr, vx_ps, vy_ps, vz_ps, normalizer=normalizer, sxy_pixels=sxy_pixels)

        # Mini graphs
        pw = int(0.32 * W); ph = int(0.20 * H); ox, oy = 10, H - ph - 10
        cv.rectangle(frame_bgr, (ox, oy), (ox+pw, oy+ph), (40,40,40), 2, cv.LINE_AA)
        for k, val in (('plot_vx', vx_ps), ('plot_vy', vy_ps), ('plot_vz', vz_ps), ('plot_flux', flux)):
            buf = self._ov_hist.setdefault(k, []); buf.append(float(val))
            if len(buf) > pw: del buf[0]
        def norm_signed(buf):
            if not buf: return []
            p95 = np.percentile(np.abs(buf), 95.0) or 1.0
            return np.clip(np.array(buf, dtype=np.float32)/float(p95), -1.0, 1.0)
        def norm_pos(buf):
            if not buf: return []
            p95 = np.percentile(np.array(buf, dtype=np.float32), 95.0) or 1.0
            return np.clip(np.array(buf, dtype=np.float32)/float(p95), 0.0, 1.0)
        sx = norm_signed(self._ov_hist['plot_vx']); sy = norm_signed(self._ov_hist['plot_vy']); sz = norm_signed(self._ov_hist['plot_vz']); sf = norm_pos(self._ov_hist['plot_flux'])
        def draw_series(vals, color, signed=True):
            if len(vals) < 2: return
            L = len(vals)
            for i in range(1, L):
                x1 = ox + int((i-1) * pw / max(1, (L-1))); x2 = ox + int(i * pw / max(1, (L-1)))
                if signed:
                    y1 = oy + int((1.0 - (vals[i-1] + 1.0)/2.0) * ph); y2 = oy + int((1.0 - (vals[i] + 1.0)/2.0) * ph)
                else:
                    y1 = oy + int((1.0 - vals[i-1]) * ph); y2 = oy + int((1.0 - vals[i]) * ph)
                cv.line(frame_bgr, (x1,y1), (x2,y2), color, 2, cv.LINE_AA)
        draw_series(sx, C_X, True); draw_series(sy, C_Y, True); draw_series(sz, C_Z, True); draw_series(sf, C_FL, False)

    def estimate_2d(self, prev_gray, gray, keep=800):
        kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(gray, None)
        if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
            return 0.0, 0.0, 0.0, 1.0, 0, None, None
        matches = self.bf.match(des1, des2)
        if len(matches) < 12:
            return 0.0, 0.0, 0.0, 1.0, len(matches), None, None
        matches = sorted(matches, key=lambda m: m.distance)[:keep]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        A, _inl = cv.estimateAffinePartial2D(pts1, pts2, method=cv.RANSAC, ransacReprojThreshold=RANSAC_THRESH, maxIters=AFF_MAXIT, confidence=AFF_CONF)
        if A is None: return 0.0, 0.0, 0.0, 1.0, len(matches), pts1, pts2
        a, b = float(A[0,0]), float(A[0,1]); dx, dy = float(A[0,2]), float(A[1,2])
        scale = float(math.sqrt(max(1e-12, a*a + b*b))); rot2d = float(math.degrees(math.atan2(b, a)))
        return dx, dy, rot2d, scale, len(matches), pts1, pts2

    def estimate_3d_parallax(self, pts1, pts2):
        if pts1 is None or pts2 is None or len(pts1) < 12: return 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        H, maskH = cv.findHomography(pts1, pts2, cv.RANSAC, RANSAC_THRESH, maxIters=4000, confidence=0.995); Hinl = int(maskH.sum()) if maskH is not None else 0
        try: E, _ = cv.findEssentialMat(pts1, pts2, self.K, method=cv.RANSAC, prob=0.999, threshold=RANSAC_THRESH)
        except cv.error: E = None
        cands=[]
        if E is not None:
            if E.ndim == 2 and E.shape == (3,3): cands=[E]
            elif E.ndim == 3 and E.shape[:2] == (3,3): cands = [E[:,:,i] for i in range(E.shape[2])]
            elif E.size % 9 == 0:
                Es = E.reshape((-1,3,3)); cands = [Es[i] for i in range(Es.shape[0])]
        best_R = np.eye(3, dtype=np.float64); best_t = np.zeros((3,1), dtype=np.float64); best_score = -1
        for Ei in cands:
            try: _, Ri, ti, maskPose = cv.recoverPose(Ei, pts1, pts2, self.K); score = int(maskPose.sum()) if maskPose is not None else 0
            except cv.error: continue
            if score > best_score: best_score, best_R, best_t = score, Ri, ti
        E_inl = max(0, best_score); yaw, pitch, roll = rotmat_to_euler(best_R)
        tx=float(best_t[0,0]); ty=float(best_t[1,0]); tz=float(best_t[2,0]); tnorm=float(np.linalg.norm(best_t))
        M=int(pts1.shape[0]); parallax=max(0.0, float(E_inl - Hinl))/float(max(1, M))
        return E_inl, Hinl, parallax, yaw, pitch, roll, tx, ty, tz, tnorm

    def draw_graph_strip(self, w: int) -> np.ndarray:
        g = np.zeros((GRAPH_H, w, 3), np.uint8)
        def tail(x: List[float], W: int): return x[-W:] if len(x) > W else x
        T = max(20, w - 20)
        series = [(tail(self.g_trans,T),(0,255,0)),(tail(self.g_rot2d,T),(0,165,255)),(tail(self.g_roll,T),(0,255,255)),(tail(self.g_parax,T),(255,0,255))]
        y_bot = GRAPH_H - 5
        for vals, color in series:
            if len(vals) < 2: continue
            v = np.array(vals, dtype=np.float32); v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            p95 = np.percentile(np.abs(v), 95) or 1.0
            v = np.clip(v / p95, -1.0, 1.0)
            y = y_bot - ((v + 1.0) * 0.5 * (GRAPH_H - 10)).astype(np.int32)
            for i in range(1, len(y)):
                cv.line(g, (10+i-1, int(y[i-1])), (10+i, int(y[i])), color, 1, cv.LINE_AA)
        cv.line(g, (10, y_bot), (w-10, y_bot), (50,50,50), 1, cv.LINE_AA)
        return g
    
    def next(self):
        # --- PATCH 7: use prefetched frames (Analyzer.next) ---
        item = self._fq.get()
        if item is None:
            return False, self.frame_idx, None, None, None
            
        frame = item
        gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        time_s = self.frame_idx / self.fps

        # --- HARD CUT DETECTION (structure phaseCorr) ---
        cut_now = False
        try:
            if self.prev_gray is not None:
                resp, ad01 = _cut_struct_phasecorr(self.prev_gray, gray, scale=FLOW_SCALE)
                resp_th = float(getattr(self.args, "cut_resp_thresh", 0.08))
                ad_th   = float(getattr(self.args, "cut_absdiff_thresh", 0.18))
                cut_now = (resp < resp_th) and (ad01 >= ad_th)
                if cut_now:
                    self._cut_hold = int(getattr(self.args, "cut_hold_frames", 2))
        except Exception:
            cut_now = False

        if self._cut_hold > 0:
            # Hold for N frames after cut: emit neutral motion and reset output-facing smoothers
            self._cut_hold -= 1
            self.dup_flags.append(False)

            dx = dy = 0.0
            rot2d, scale2d = 0.0, 1.0
            trans_mag = 0.0
            yaw = pitch = roll = 0.0
            tx = ty = tz = 0.0
            tnorm = 0.0
            parallax = 0.0
            divz = 0.0
            curv_val = 0.0
            vz25_raw = 0.0
            nmatch = 0
            pts1 = pts2 = None

            row = FrameRow(frame=self.frame_idx, time=time_s,
                        dx=dx, dy=dy, rot2d_deg=rot2d, scale2d=scale2d, trans_mag=trans_mag,
                        yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                        tx=tx, ty=ty, tz=tz, tnorm=tnorm,
                        orb_matches=nmatch,
                        E_inliers=0, H_inliers=0, parallax=parallax,
                        divz=float(divz), curv=float(curv_val), vz25=float(vz25_raw))
            self.rows.append(row)

            # Reset output-facing state so normalization doesn't get poisoned
            self._live_initialized = False
            self.vx_live = self.vy_live = self.vz_live = 0.0
            self._ov_hist = {}

            # Re-seed ORB cache on the new scene (pose becomes meaningful again after the cut)
            try:
                oh, ow = gray.shape[:2]
                oW = max(32, int(round(ow * ORB_SCALE)))
                oH = max(32, int(round(oh * ORB_SCALE)))
                prev_orb = cv.resize(gray, (oW, oH), cv.INTER_AREA) if ORB_SCALE != 1.0 else gray
                self.kp_prev, self.des_prev = _orb_struct_detect_and_compute(self.orb, prev_orb, max_kp=ORB_NFEATURES)
            except Exception:
                self.kp_prev, self.des_prev = [], None

            # Advance state
            self.prev_gray = gray
            self.frame_idx += 1

            # HUD / overlay for neutral frame (still writes annotated video if enabled)
            band = np.full((HUD_BAND_H, self.W, 3), (25,25,25), np.uint8)
            ytxt = 24
            draw_text(band, f"{self.path.name}  {self.W}x{self.H}@{self.fps:.2f}  f {self.frame_idx}/{self.N}  t {time_s:7.3f}s", ytxt); ytxt+=22
            draw_text(band, f"2D: dx={dx:+6.2f} dy={dy:+6.2f} rot2d={rot2d:+6.2f}° scale={scale2d:0.4f} |T|={trans_mag:6.2f}", ytxt, (200,255,200)); ytxt+=20
            draw_text(band, f"3D: yaw={yaw:+6.2f}° pitch={pitch:+6.2f}° roll={roll:+6.2f}° t=({tx:+.2f},{ty:+.2f},{tz:+.2f}) para={parallax:0.3f}", ytxt, (200,220,255))

            self.g_trans.append(trans_mag); self.g_rot2d.append(abs(rot2d)); self.g_roll.append(abs(roll)); self.g_parax.append(parallax)
            for b in (self.g_trans, self.g_rot2d, self.g_roll, self.g_parax):
                if len(b) > GRAPH_HISTORY_W: del b[0]
            graph = self.draw_graph_strip(self.W)

            gui = frame.copy()
            self.draw_overlay_symbols(gui, 0.0, 0.0, 0.0, 0.0, normalizer=None, sxy_pixels=self.sxy_dim)
            out = np.vstack([gui, band, graph])
            if out.shape[1] != self.outW or out.shape[0] != self.outH:
                out = cv.resize(out, (self.outW, self.outH), cv.INTER_AREA)
            if self.vw is not None: self.vw.write(out)
            if self.show_window: cv.imshow("Camera Analyzer v5F", out)

            metrics = dict(frame=self.frame_idx-1, t=time_s,
                        dx=dx, dy=dy, yaw=yaw, rot2d=rot2d, trans=trans_mag, cut=int(cut_now))
            return True, self.frame_idx-1, gui, row, metrics


        # --- DUP: detect once and reuse ---
        is_dup = is_near_duplicate_frame(self.prev_gray, gray, DUP_FRAME_MAD_THRESH)
        self.dup_flags.append(bool(is_dup))

        if is_dup and self.rows:
            # Reuse last kinematics, new time/frame index only
            last = self.rows[-1]
            dx, dy        = last.dx, last.dy
            rot2d         = last.rot2d_deg
            scale2d       = last.scale2d
            trans_mag     = last.trans_mag
            yaw           = last.yaw_deg
            pitch         = last.pitch_deg
            roll          = last.roll_deg
            tx, ty, tz    = last.tx, last.ty, last.tz
            tnorm         = last.tnorm
            parallax      = last.parallax
            divz          = last.divz
            nmatch        = last.orb_matches
            curv_val      = last.curv
            vz25_raw      = last.vz25
            pts1 = pts2   = None
        else:
            # Fresh flow + 2D/3D solve + curvature + 2.5D depth
            pack = _compute_flow_pack(self.prev_gray, gray, scale=FLOW_SCALE)
            divz = _divergence_from_pack(pack)
            curv_val = frame_curvature(self.prev_gray, gray, scale=FLOW_SCALE)
            vz25_raw = _vz25_from_pack(pack)

            # --- global motion backend: HG translation + optional ORB pose ---
            dx_hg, dy_hg, hg_conf, _hg_dbg = _hg_global_translation(
                self.prev_gray, gray, scale=FLOW_SCALE, state=self._hg_state
            )

            if self.motion_mode == "hg":
                # HG-only: translation-only, pose neutral (fastest)
                dx, dy = dx_hg, dy_hg
                rot2d, scale2d, nmatch, pts1, pts2 = 0.0, 1.0, 0, None, None
            else:
                # ORB or Hybrid: run ORB (structure-guided) to get pose + 3D correspondences
                dx_orb, dy_orb, rot2d, scale2d, nmatch, pts1, pts2 = self.estimate_2d_cached(gray)

                if self.motion_mode == "orb":
                    # ORB-only: translation+pose from ORB affine
                    dx, dy = dx_orb, dy_orb
                else:
                    # HYBRID (default agreement): HG owns translation; ORB supplies pose only
                    dx, dy = dx_hg, dy_hg

            trans_mag = float(math.hypot(dx, dy))
            (E_inl, H_inl, parallax,
            yaw, pitch, roll, tx, ty, tz, tnorm) = self.estimate_3d_parallax(pts1, pts2)


        row = FrameRow(frame=self.frame_idx, time=time_s,
                dx=dx, dy=dy, rot2d_deg=rot2d, scale2d=scale2d, trans_mag=trans_mag,
                yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                tx=tx, ty=ty, tz=tz, tnorm=tnorm,
                orb_matches=nmatch,
                E_inliers=getattr(self, "last_E_inliers", 0),
                H_inliers=getattr(self, "last_H_inliers", 0),
                parallax=parallax,
                divz=float(divz),
                curv=float(curv_val),
                vz25=float(vz25_raw))


        self.rows.append(row)
        # --- DEBUG: log flow mag vs curl statistics for CSV ---
        if ENABLE_FLOW_DEBUG_STATS:
            try:
                mag_full, curl_full = debug_flow_mag_and_curl(self.prev_gray, gray)
                div_full = self._compute_divergence_map(self.prev_gray, gray)  # <- add this

                if mag_full is not None and curl_full is not None:
                    m = mag_full.astype(np.float32)
                    c = np.abs(curl_full.astype(np.float32))

                    mean_mag = float(np.mean(m))
                    mean_abs_curl = float(np.mean(c))
                    mean_abs_div = float(np.mean(np.abs(div_full)))


                    # correlation between |curl| and |mag|
                    m_flat = m.ravel()
                    c_flat = c.ravel()
                    corr = np.nan
                    if m_flat.size >= 4:
                        # avoid constant vectors → NaN
                        if np.std(m_flat) > 1e-9 and np.std(c_flat) > 1e-9:
                            cc = np.corrcoef(m_flat, c_flat)
                            corr = float(cc[0, 1])

                    self.flow_debug_rows.append(
                        (self.frame_idx, time_s, mean_mag, mean_abs_curl, mean_abs_div, corr)
                    )

            except Exception as e:
                # don't ever let debug stats crash the run
                print("[debug] flow mag/curl stats failed:", repr(e))

        self.prev_gray = gray
        self.frame_idx += 1

        # HUD
        band = np.full((HUD_BAND_H, self.W, 3), (25,25,25), np.uint8)
        ytxt = 24
        draw_text(band, f"{self.path.name}  {self.W}x{self.H}@{self.fps:.2f}  f {self.frame_idx}/{self.N}  t {time_s:7.3f}s", ytxt); ytxt+=22
        draw_text(band, f"2D: dx={dx:+6.2f} dy={dy:+6.2f} rot2d={rot2d:+6.2f}° scale={scale2d:0.4f} |T|={trans_mag:6.2f}", ytxt, (200,255,200)); ytxt+=20
        draw_text(band, f"3D: yaw={yaw:+6.2f}° pitch={pitch:+6.2f}° roll={roll:+6.2f}° t=({tx:+.2f},{ty:+.2f},{tz:+.2f}) para={parallax:0.3f}", ytxt, (200,220,255))

        self.g_trans.append(trans_mag); self.g_rot2d.append(abs(rot2d)); self.g_roll.append(abs(roll)); self.g_parax.append(parallax)
        for b in (self.g_trans, self.g_rot2d, self.g_roll, self.g_parax):
            if len(b) > GRAPH_HISTORY_W: del b[0]
        graph = self.draw_graph_strip(self.W)

        # live overlay values = smoothed per-frame rates
        vx_raw = dx * self.fps
        vy_raw = dy * self.fps

        vx_raw = 0.0 if not np.isfinite(vx_raw) else vx_raw
        vy_raw = 0.0 if not np.isfinite(vy_raw) else vy_raw


        # base signals in per-second units
        vz_div_raw  = float(divz)      * self.fps * self.args.z_div_gain
        vz25_raw_ps = float(vz25_raw)  * self.fps * self.args.z_div_gain
        vz_div_raw = 0.0 if not np.isfinite(vz_div_raw) else vz_div_raw
        vz25_raw_ps = 0.0 if not np.isfinite(vz25_raw_ps) else vz25_raw_ps

        mode = str(getattr(self.args, "vz_mode", "hybrid")).lower()
        mix  = float(getattr(self.args, "vz25_mix", 0.70))
        mix  = float(np.clip(mix, 0.0, 1.0))

        if mode == "curv":
            vz_raw = vz25_raw_ps
        elif mode == "hybrid":
            vz_raw = vz_div_raw + vz25_raw_ps
        else:  # "div" or unknown
            vz_raw = vz_div_raw

        vx_raw, vy_raw, vz_raw = self._apply_signs(vx_raw, vy_raw, vz_raw)


        if not self._live_initialized:
            self.vx_live, self.vy_live, self.vz_live = vx_raw, vy_raw, vz_raw
            self._live_initialized = True
        else:
            self.vx_live = self._lp1_step(self.vx_live, vx_raw, self.live_lp_hz)
            self.vy_live = self._lp1_step(self.vy_live, vy_raw, self.live_lp_hz)
            self.vz_live = self._lp1_step(self.vz_live, vz_raw, self.live_lp_hz)

        vx_ps, vy_ps, vz_ps = self.vx_live, self.vy_live, self.vz_live
        flux  = math.sqrt(vx_ps*vx_ps + vy_ps*vy_ps + (self.args.flux_z_weight * vz_ps*vz_ps))

        # update running global XY max (first pass) and lock after v3_lock_s
        self._sxy_running = max(self._sxy_running, math.hypot(vx_ps, vy_ps))
        self._ov_hist['ov3_sxy_max'] = self._sxy_running
        if self._sxy_lock is None and (self.frame_idx / self.fps) >= self.v3_lock_s:
            self._sxy_lock = self._sxy_running
        sxy_firstpass = self._sxy_lock if self._sxy_lock is not None else self._sxy_running

        
        # overlay on video
        gui = frame.copy()
        sxy_firstpass = self.sxy_dim
        self.draw_overlay_symbols(gui, vx_ps, vy_ps, vz_ps, flux,
                                  normalizer=None, sxy_pixels=sxy_firstpass)

        # --- DEBUG unified MAG+DIV+CURL overlay ---
        if getattr(self, "debug_flow_mode", 0) != 0:
            dbg_col, dbg_alpha = build_unified_flow_overlay(self, self.prev_gray, gray)
            if dbg_col is not None and dbg_alpha is not None:
                a = dbg_alpha.astype(np.float32)[..., None]  # HxWx1
                base = gui.astype(np.float32)
                over = dbg_col.astype(np.float32)
                gui = (base * (1.0 - a) + over * a).astype(np.uint8)

        
        # output frame (video + HUD + graph)
        out = np.vstack([gui, band, graph])



        # output frame (video + HUD + graph)
        out = np.vstack([gui, band, graph])
        if out.shape[1] != self.outW or out.shape[0] != self.outH: out = cv.resize(out, (self.outW, self.outH), cv.INTER_AREA)
        if self.vw is not None: self.vw.write(out)
        if self.show_window: cv.imshow("Camera Analyzer v5F", out)
        metrics = dict(frame=self.frame_idx-1, t=time_s,
                    dx=dx, dy=dy, yaw=yaw, rot2d=rot2d, trans=trans_mag)
        return True, self.frame_idx-1, gui, row, metrics


    # ---------- post (export) ----------
    @staticmethod
    def ewma_bias(x: np.ndarray, fps: float, tau_s: float) -> np.ndarray:
        alpha = (1.0/fps) / max(1e-6, tau_s)
        y = np.empty_like(x, dtype=np.float64); y[0] = x[0]
        for i in range(1, len(x)): y[i] = (1.0 - alpha)*y[i-1] + alpha*x[i]
        return y

    @staticmethod
    def norm_sym_p95(x: np.ndarray, gamma: float = 1.15) -> np.ndarray:
        p95 = np.percentile(np.abs(x), 95.0) or 1.0
        y = np.clip(x / p95, -1.0, 1.0)
        return np.sign(y) * (np.abs(y) ** gamma)

    def robust_integrate(self, v: np.ndarray, fps: float,
                         xn: np.ndarray, conf: np.ndarray,
                         dead: float, lp_hz: float, bias_tau_s: float,
                         jerk: Optional[np.ndarray] = None, burst_gain: float = 0.5) -> np.ndarray:
        dt = 1.0 / max(1e-6, fps)
        g = np.clip((np.abs(xn) - dead) / (1.0 - dead + 1e-9), 0.0, 1.0) * conf
        if jerk is not None and jerk.size == v.size:
            jn = np.clip(np.abs(jerk)/(np.percentile(np.abs(jerk),95)+1e-9), 0.0, 1.0)
            g = np.maximum(g, burst_gain * jn)
        v_eff = v * g
        p = np.cumsum(v_eff) * dt
        pb = self.ewma_bias(p, fps, tau_s=bias_tau_s)
        p_hp = p - pb
        p_sm = lp1(p_hp, fps, cutoff_hz=lp_hz)
        return p_sm

    def render_export_robust(self):
        if not getattr(self, "_series_cache", None):
            return
        ser = self._series_cache
        vx_s = ser["vx_s"]; vy_s = ser["vy_s"]; vz_s = ser["vz_s"]; flux = ser["flux"]
        px = ser["px"]; py = ser["py"]; pz = ser["pz"]
        v3_sxy_max = ser.get("v3_sxy_max", float(np.max(np.hypot(vx_s, vy_s))) if vx_s.size else 1.0)

        # Reuse cached robust scales if present; otherwise fit
        scales = ser.get("normer_scales", None)
        if scales:
            normer = RobustArrowNormalizer(self.robust_z, self.robust_prctl, self.robust_gamma)
            normer.sx, normer.sy, normer.sz = float(scales[0]), float(scales[1]), float(scales[2])
        else:
            normer = RobustArrowNormalizer(self.robust_z, self.robust_prctl, self.robust_gamma).fit(vx_s, vy_s, vz_s)

        cap = cv.VideoCapture(str(self.path)); assert cap.isOpened(), f"Cannot reopen: {self.path}"
        vw = FFMpegWriter(self.out_mp4_robust, fps=self.fps, W=self.outW, H=self.outH,
                          ffmpeg_bin=self.args.ffmpeg_bin, vcodec=self.args.vcodec, crf=self.args.crf,
                          preset=self.args.preset, pix_fmt_out=self.args.pix_fmt)

        self.g_trans=[]; self.g_rot2d=[]; self.g_roll=[]; self.g_parax=[]

        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= len(self.rows): break
            gui = frame.copy()

            vx_ps = float(vx_s[i]); vy_ps = float(vy_s[i]); vz_ps = float(vz_s[i]); fl = float(flux[i]); r = self.rows[i]
            # only use robust scaling for Z
            sz = float(ser.get("normer_scales", (None,None,None))[2] or 1.0)
            normerZ = RobustArrowNormalizer(self.robust_z, self.robust_prctl, self.robust_gamma)
            normerZ.sz = sz
            sxy = self.sxy_dim
            self.draw_overlay_symbols(gui, vx_ps, vy_ps, vz_ps, fl,
                                      px=float(px[i]), py=float(py[i]), pz=float(pz[i]),
                                      normalizer=normerZ, sxy_pixels=v3_sxy_max)

            band = np.full((HUD_BAND_H, self.W, 3), (25,25,25), np.uint8)
            ytxt = 24
            draw_text(band, f"{self.path.name}  {self.W}x{self.H}@{self.fps:.2f}  f {i}/{self.N}  t {r.time:7.3f}s", ytxt); ytxt+=22
            draw_text(band, f"2D: dx={r.dx:+6.2f} dy={r.dy:+6.2f} rot2d={r.rot2d_deg:+6.2f}° scale={r.scale2d:0.4f} |T|={r.trans_mag:6.2f}", ytxt, (200,255,200)); ytxt+=20
            draw_text(band, f"3D: yaw={r.yaw_deg:+6.2f}° pitch={r.pitch_deg:+6.2f}° roll={r.roll_deg:+6.2f}° t=({r.tx:+.2f},{r.ty:+.2f},{r.tz:+.2f}) para={r.parallax:0.3f}", ytxt, (200,220,255))

            self.g_trans.append(r.trans_mag); self.g_rot2d.append(abs(r.rot2d_deg)); self.g_roll.append(abs(r.roll_deg)); self.g_parax.append(r.parallax)
            for b in (self.g_trans, self.g_rot2d, self.g_roll, self.g_parax):
                if len(b) > GRAPH_HISTORY_W: del b[0]
            graph = self.draw_graph_strip(self.W)

            out = np.vstack([gui, band, graph])
            if out.shape[1] != self.outW or out.shape[0] != self.outH: out = cv.resize(out, (self.outW, self.outH), cv.INTER_AREA)
            vw.write(out)
            i += 1

        cap.release(); vw.close()

    def finalize_and_export(self):
        if not self.rows: return
        fps, args = self.fps, self.args

        # gather arrays
        t   = np.array([r.time for r in self.rows], float)
        dx  = np.array([r.dx for r in self.rows], float)
        dy  = np.array([r.dy for r in self.rows], float)
        yaw = np.array([r.yaw_deg for r in self.rows], float)
        pit = np.array([r.pitch_deg for r in self.rows], float)
        rol = np.array([r.roll_deg for r in self.rows], float)
        divz = np.array([getattr(r, 'divz', 0.0) for r in self.rows], float)
        vz25 = np.array([getattr(r, 'vz25', 0.0) for r in self.rows], float)
        curv_series = np.array([getattr(r, 'curv', 0.0) for r in self.rows], float)


        # unwrap angles
        yaw_u = np.rad2deg(np.unwrap(np.deg2rad(yaw)))
        pit_u = np.rad2deg(np.unwrap(np.deg2rad(pit)))
        rol_u = np.rad2deg(np.unwrap(np.deg2rad(rol)))

        # raw rates
        vx = dx * fps
        vy = dy * fps

        vz_div_raw  = divz * fps * args.z_div_gain
        vz25_raw_ps = vz25 * fps * args.z_div_gain

        mode = str(getattr(args, "vz_mode", "hybrid")).lower()
        mix  = float(getattr(args, "vz25_mix", 0.70))
        mix  = float(np.clip(mix, 0.0, 1.0))

        if mode == "curv":
            vz_rel = vz25_raw_ps
        elif mode == "hybrid":
            vz_rel = vz_div_raw + vz25_raw_ps
        else:  # "div" or unknown
            vz_rel = vz_div_raw

        vx, vy, vz_rel = self._apply_signs(vx, vy, vz_rel)

        # Interpolate over duplicate frames so envelopes don’t staircase
        if getattr(self, "dup_flags", None) and len(self.dup_flags) == len(vx):
            vx = interpolate_over_dups_1d(vx, self.dup_flags)
            vy = interpolate_over_dups_1d(vy, self.dup_flags)
            vz_rel = interpolate_over_dups_1d(vz_rel, self.dup_flags)

        wyaw = np.gradient(yaw_u, t, edge_order=2) if t.size >= 3 else np.zeros_like(yaw_u)
        wpit = np.gradient(pit_u, t, edge_order=2) if t.size >= 3 else np.zeros_like(pit_u)
        wrol = np.gradient(rol_u, t, edge_order=2) if t.size >= 3 else np.zeros_like(rol_u)

        # smoothing
        def S_vel(x): return smooth_series(x, fps, args.smooth_vel, args.lp_vel_hz, args.sg_win_ms, args.sg_poly, args.kalman_q, args.kalman_r, args.max_rate_vel)
        def S_ang(x): return smooth_series(x, fps, args.smooth_ang, args.lp_ang_hz, args.sg_win_ms, args.sg_poly, args.kalman_q, args.kalman_r, args.max_rate_ang)

        # --- Normalization to match robust video ---
        # XY: dimension-based (W/2, H/2); Z: robust (MAD+percentile)
        vx_s = S_vel(vx); vy_s = S_vel(vy); vz_s = S_vel(vz_rel)
        vx_n = np.clip(vx_s / self.sx_dim, -1.0, 1.0)
        vy_n = np.clip(vy_s / self.sy_dim, -1.0, 1.0)

        nz_fit = RobustArrowNormalizer(self.robust_z, self.robust_prctl, self.robust_gamma).fit(
            np.zeros_like(vx_s), np.zeros_like(vy_s), vz_s
        )
        def _rob_z(arr):
            z = np.clip(arr / max(1e-6, nz_fit.sz), -1.0, 1.0)
            g = self.robust_gamma
            return np.sign(z) * (np.abs(z) ** g) if g != 1.0 else z

        vz_n = _rob_z(vz_s)

        # Treat these as the canonical normalized lanes
        vx_r, vy_r, vz_r = vx_n, vy_n, vz_n

        # Cache scales for robust pass
        v3_sxy_ref = self.sxy_dim
        normer_scales = (self.sx_dim, self.sy_dim, float(nz_fit.sz))
        wyaw_s = S_ang(wyaw); wpit_s = S_ang(wpit); wrol_s = S_ang(wrol)

        # metrics
        speed2d = np.sqrt(vx_s**2 + vy_s**2)
        speed3  = np.sqrt(vx_s**2 + vy_s**2 + vz_s**2)
        flux    = np.sqrt(vx_s**2 + vy_s**2 + (args.flux_z_weight * vz_s)**2)
        acc  = np.gradient(flux, t, edge_order=2) if t.size >= 3 else np.zeros_like(flux)
        jerk = np.gradient(acc,  t, edge_order=2) if t.size >= 3 else np.zeros_like(acc)
        # --- Primary motion-based entropy from global flux ---

        # Event entropy: deviation from a slow trend (~0.35 s)
        sigma_event = max(1.0, 0.35 * fps)
        flux_slow   = gauss_blur1d(flux, sigma_event)
        flux_res    = flux - flux_slow

        ent_event   = np.abs(flux_res)
        p95_event   = float(np.percentile(ent_event, 95.0)) or 1e-9
        entropy_event01 = np.clip(ent_event / p95_event, 0.0, 1.0)

        # Micro entropy: fast, noisy residual (~0.10 s), MAD-whitened
        sigma_micro = max(1.0, 0.10 * fps)
        flux_fast   = gauss_blur1d(flux, sigma_micro)
        flux_hp     = flux - flux_fast

        z_hp        = robust_z(flux_hp, clip=4.0)
        micro_core  = np.tanh(0.7 * z_hp)   # signed, compressed chaos
        ent_micro   = np.abs(micro_core)

        p95_micro   = float(np.percentile(ent_micro, 95.0)) or 1e-9
        entropy_micro01 = np.clip(ent_micro / p95_micro, 0.0, 1.0)

        # --- Secondary: smoothed LoG² entropy from curvature (G(LoG(LoG(curv)))) ---

        if curv_series.size >= 5:
            # LoG¹ in time on curvature
            log1 = log1d_series(curv_series, fps, sigma_ms=80.0)
            # LoG²
            log2 = log1d_series(log1, fps, sigma_ms=80.0)

            # Smooth the LoG² field in time (Gσ₂ ∘ LoG²)
            sigma_log2_smooth = max(1.0, 0.10 * fps)  # ~0.10 s
            log2_smooth = gauss_blur1d(log2, sigma_log2_smooth)

            # Signed, whitened LoG²_smooth (for optional bipolar use)
            z_log2    = robust_z(log2_smooth, clip=4.0)
            log2_core = np.tanh(0.7 * z_log2)  # keep sign, compress extremes

            # Magnitude → 0..1 artifact/curvature entropy
            log2_abs  = np.abs(log2_core)
            p95_log2  = float(np.percentile(log2_abs, 95.0)) or 1e-9
            entropy_log2_01 = np.clip(log2_abs / p95_log2, 0.0, 1.0)
        else:
            log2_core       = np.zeros_like(flux)
            entropy_log2_01 = np.zeros_like(flux)

        for i, r in enumerate(self.rows):
            r.vx_px_s = float(vx_s[i]); r.vy_px_s = float(vy_s[i]); r.vz_rel_s = float(vz_s[i])
            r.speed2d_px_s = float(speed2d[i]); r.speed3_rel = float(speed3[i])
            r.flux = float(flux[i]); r.acc = float(acc[i]); r.jerk = float(jerk[i])

        # robust positions
        conf = np.ones_like(t, dtype=np.float64)
        xn_vx_p95 = self.norm_sym_p95(vx_s, args.gamma)
        xn_vy_p95 = self.norm_sym_p95(vy_s, args.gamma)
        xn_vz_p95 = self.norm_sym_p95(vz_s, args.gamma)

        px = self.robust_integrate(vx_s, fps, xn_vx_p95, conf, args.pos_dead, args.pos_lp_hz, args.pos_bias_tau_s, jerk=jerk, burst_gain=args.pos_burst_gain)
        py = self.robust_integrate(vy_s, fps, xn_vy_p95, conf, args.pos_dead, args.pos_lp_hz, args.pos_bias_tau_s, jerk=jerk, burst_gain=args.pos_burst_gain)
        pz = self.robust_integrate(vz_s, fps, xn_vz_p95, conf, args.pos_dead, max(0.1, args.pos_lp_hz * 0.8), args.pos_bias_tau_s, jerk=jerk, burst_gain=args.pos_burst_gain)


        # 3D vector helpers
        v3_sxy_max = float(np.max(np.hypot(vx_s, vy_s))) if vx_s.size else 1.0
        v3_mag = np.sqrt(vx_r**2 + vy_r**2 + vz_r**2)
        v3_mag01 = np.clip(v3_mag / math.sqrt(3.0), 0.0, 1.0) ** args.gamma
        azim_deg = np.degrees(np.arctan2(vy_r, vx_r))
        elev_deg = np.degrees(np.arctan2(vz_r, np.hypot(vx_r, vy_r)))

        # RAW CSV
        with open(self.out_raw, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(asdict(self.rows[0]).keys()))
            for r in self.rows:
                w.writerow(list(asdict(r).values()))

        # Modulators CSV (resampled)
        sr = max(1.0, float(args.resample_hz))
        T_end = float(t[-1]); t_out = np.arange(0.0, T_end + 1.0 / sr, 1.0 / sr)
        def rs(x): return np.interp(t_out, t, x)

        p_lo_val = np.percentile(flux, args.p_lo); p_hi_val = np.percentile(flux, args.p_hi)
        denom = max(1e-9, (p_hi_val - p_lo_val))
        flux_n = np.clip((flux - p_lo_val) / denom, 0.0, 1.0) ** args.gamma
        acc_n  = self.norm_sym_p95(acc, args.gamma)

        out_cols = {
            "time": t_out,

            # Normalized lanes for control (match robust video)
            "vx_s": rs(vx_n), "vy_s": rs(vy_n), "vz_s": rs(vz_n),

            # Aliases (back-compat)
            "vx_r": rs(vx_n), "vy_r": rs(vy_n), "vz_r": rs(vz_n),
            "vx_cam": rs(vx_n), "vy_cam": rs(vy_n), "vz_cam": rs(vz_n),

            # Absolute velocities (px/s) if you ever need them
            "vx_px_s": rs(vx_s), "vy_px_s": rs(vy_s), "vz_rel_s": rs(vz_s),

            # Angular lanes
            "wyaw_s": rs(self.norm_sym_p95(wyaw_s, args.gamma)),
            "wpitch_s": rs(self.norm_sym_p95(wpit_s, args.gamma)),
            "wroll_s":  rs(self.norm_sym_p95(wrol_s, args.gamma)),

            # Amplitude helpers
            "flux_n": rs(flux_n), "acc_s": rs(acc_n),
            "amp_low":  rs(flux_n),
            "amp_mid":  rs(0.6 * flux_n),
            "amp_high": rs(0.3 * flux_n),
            "fm_low_hz":  rs(0.5 + 2.5 * flux_n),
            "fm_mid_hz":  rs(1.0 + 5.0 * flux_n),
            "fm_high_hz": rs(3.0 + 15.0 * flux_n),

            # Entropy lanes
            "cam_entropy_event01":  rs(entropy_event01),   # macro motion-change spikes
            "cam_entropy_motion01": rs(entropy_micro01),   # primary motion micro-chaos
            "cam_entropy_log2_01":  rs(entropy_log2_01),   # smoothed LoG² curvature entropy
            "cam_log2_signed":      rs(np.clip(log2_core, -1.0, 1.0)),
            "entropy_flux01": rs(entropy_micro01),  # default: motion-based micro entropy


            # Positions
            "px": rs(px), "py": rs(py), "pz": rs(pz),
            "px_n": rs(self.norm_sym_p95(px, args.gamma)),
            "py_n": rs(self.norm_sym_p95(py, args.gamma)),
            "pz_n": rs(self.norm_sym_p95(pz, args.gamma)),
            "posx01": rs(0.5 + 0.5 * self.norm_sym_p95(px, args.gamma)),
            "posy01": rs(0.5 + 0.5 * self.norm_sym_p95(py, args.gamma)),
            "posz01": rs(0.5 + 0.5 * self.norm_sym_p95(pz, args.gamma)),

            # Direction 0..1 lanes
            "dirx01": rs(0.5 + 0.5 * vx_n),
            "diry01": rs(0.5 + 0.5 * vy_n),
            "dirz01": rs(0.5 + 0.5 * vz_n),

            # 3D vector helpers
            "v3_mag01": rs(np.clip(np.sqrt(vx_n**2 + vy_n**2 + vz_n**2) / math.sqrt(3.0), 0.0, 1.0) ** args.gamma),
            "v3_azim_deg": rs(np.degrees(np.arctan2(vy_n, vx_n))),
            "v3_elev_deg": rs(np.degrees(np.arctan2(vz_n, np.hypot(vx_n, vy_n)))),
        }

        # after jerk is computed and before writing out_cols:
        thr = float(np.percentile(np.abs(jerk), 85)) if jerk.size else 1.0
        impact01 = (np.abs(jerk) >= max(1e-9, thr)).astype(float)

        # when building out_cols (resample with rs() like other lanes):
        out_cols["impact01"] = rs(impact01)



        with open(self.out_mod, "w", newline="") as f:
            w = csv.writer(f)
            headers = list(out_cols.keys()); w.writerow(headers)
            L = len(t_out)
            for i in range(L):
                w.writerow([f"{float(out_cols[h][i]):.6f}" for h in headers])

        # cache for robust pass — THIS prevents the 'normer_scales' error
        self._series_cache = dict(
            vx_s=vx_s, vy_s=vy_s, vz_s=vz_s, flux=flux,
            px=px, py=py, pz=pz, t=t,
            normer_scales=normer_scales,
            v3_sxy_max=(v3_sxy_max if v3_sxy_max > 1e-6 else 1.0)
        )
        self._series_cache.update({
            "vx_s": vx_s, "vy_s": vy_s, "vz_s": vz_s, "flux": flux,
            "px": px, "py": py, "pz": pz, "t": t,
            "normer_scales": normer_scales,
            "v3_sxy_max": v3_sxy_ref
        })

        # optional REAPER push
        if getattr(self.args, 'reaper_push', False):
            t0 = float(t[0]) if t.size else 0.0
            t1 = float(t[-1]) if t.size else t0
            try:
                emit_reaper_push(str(self.out_mod), t0, t1, version='v5F')
            except Exception as e:
                print(f"[reaper] push failed: {e}")
        
        # --- DEBUG CSV: curl vs magnitude stats ---
        if self.flow_debug_rows and self.out_dir:
            dbg_path = os.path.join(self.out_dir, "curl_mag_debug.csv")
            try:
                # inside finalize_and_export(), in the curl/mag CSV block
                with open(dbg_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["frame", "time_s", "mean_mag", "mean_abs_curl",
                                "mean_abs_div", "corr_abs_curl_vs_mag"])
                    for fr, t, mm, mc, md, corr in self.flow_debug_rows:
                        w.writerow([
                            fr,
                            f"{t:.6f}",
                            f"{mm:.9g}",
                            f"{mc:.9g}",
                            f"{md:.9g}",
                            "" if np.isnan(corr) else f"{corr:.6f}",
                        ])

                print(f"[debug] wrote curl/mag stats CSV → {dbg_path}")
            except Exception as e:
                print("[debug] failed to write curl/mag CSV:", repr(e))


# ---------- REAPER push (FluxBridge-compatible) ----------
REAPER_BRIDGE_INBOX = os.environ.get("REAPER_BRIDGE_DIR")

def _discover_inbox_default():
    candidates = []
    home = os.path.expanduser("~")
    candidates += [
        os.path.join(os.getenv("APPDATA",""), "REAPER", "Scripts", "FluxBridge", "inbox"),
        os.path.join(home, "Library", "Application Support", "REAPER", "Scripts", "FluxBridge", "inbox"),
        os.path.join(home, ".config", "REAPER", "Scripts", "FluxBridge", "inbox"),
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            ready = os.path.join(p, "BRIDGE_READY.txt")
            if os.path.isfile(ready): return p
    p = os.path.abspath("./FluxBridge_inbox")
    os.makedirs(p, exist_ok=True)
    return p

def _bridge_inbox():
    p = REAPER_BRIDGE_INBOX or _discover_inbox_default()
    os.makedirs(p, exist_ok=True)
    os.makedirs(os.path.join(p, "processed"), exist_ok=True)
    return p

def _py2lua(obj):
    if obj is None: return "nil"
    if isinstance(obj, (int, float)): return ("%0.10f" % obj).rstrip("0").rstrip(".")
    if isinstance(obj, str): return "'" + obj.replace('/mnt/c/', 'C:\\').replace("\\","/").replace("'", "\\'") + "'"
    if isinstance(obj, list): return "{" + ",".join(_py2lua(v) for v in obj) + "}"
    if isinstance(obj, dict):
        items=[]
        for k,v in obj.items():
            if isinstance(k, str) and k.isidentifier():
                items.append(k + "=" + _py2lua(v))
            else:
                items.append("[" + _py2lua(k) + "]=" + _py2lua(v))
        return "{" + ",".join(items) + "}"
    return "'" + str(obj).replace("'", "\\'") + "'"

def emit_reaper_push(csv_path: str, t0: float, t1: float, version: str = "v5F"):
    inbox = _bridge_inbox()
    agg = dict(
        name=f"CameraMotion_{version}",
        columns=dict(
            env="flux_n",
            dirx="dirx01",
            diry="diry01",
            posx="posx01",
            posy="posy01",
            dirz="dirz01",
            posz="posz01",
            entropy_motion="cam_entropy_motion01",
            entropy_log2="cam_entropy_log2_01",

        )
    )
    job = {
        "csv": os.path.abspath(csv_path).replace('/mnt/c/', 'C:\\').replace("\\","/"),
        "start_sec": float(t0),
        "end_sec": float(t1),
        "scene_id": 0,
        "agg_track": agg,
        "roi_tracks": [],
        "version": version
    }
    lua = "return " + _py2lua(job) + "\n"
    fn  = os.path.join(inbox, f"push_camera_{int(time.time())}_{version}.rpush.lua")
    tmp = fn + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: f.write(lua)
    os.replace(tmp, fn)
    print(f"[reaper] push → {fn}")

# ---------- CLI ----------
def build_arg_parser():
    ap = argparse.ArgumentParser(description="Camera Motion Analyzer v5F — robust second pass + 3D vector overlay with global XY scale; CSV parity.")
    ap.add_argument("video", type=Path, help="Input video file")
    # profiles
    ap.add_argument("--profile", choices=["cinematic","handheld","action","stabilized","vr_comfort","raw"], default="cinematic")
    ap.add_argument("--cinematic", action="store_true"); ap.add_argument("--handheld", action="store_true")
    ap.add_argument("--action", action="store_true"); ap.add_argument("--stabilized", action="store_true")
    ap.add_argument("--vr-comfort", dest="vr_comfort", action="store_true"); ap.add_argument("--raw", action="store_true")
    # output / behavior
    ap.add_argument("--out-dir", type=Path, default=Path("analyzed_output"))
    ap.add_argument("--no-window", action="store_true", help="Disable preview window")
    ap.add_argument("--no-annotated", action="store_true", help="Do not write annotated MP4")
    ap.add_argument("--print-profile", action="store_true", help="Print effective parameters and exit")
    # normalization
    ap.add_argument("--p-lo", type=float, default=5.0); ap.add_argument("--p-hi", type=float, default=95.0)
    # FFmpeg settings
    ap.add_argument("--ffmpeg-bin", default="ffmpeg")
    ap.add_argument("--vcodec", default="libx264", choices=["libx264","libx265","libsvtav1","prores_ks"], help="Output codec")
    ap.add_argument("--crf", type=float, default=18.0, help="CRF for x264/x265 (lower = higher quality)")
    ap.add_argument("--preset", default="medium", help="x264/x265 preset")
    ap.add_argument("--pix-fmt", default="yuv420p", help="Output pixel format")

    # Z-divergence gain & REAPER
    ap.add_argument("--z-div-gain", type=float, default=1.0, help="Gain applied to divergence-based Z")
    ap.add_argument("--reaper-push", action="store_true", help="After export, push lanes to REAPER FluxBridge inbox")
    # 2.5D vs divergence blend
    ap.add_argument("--vz-mode", choices=["div", "curv", "hybrid"], default="hybrid",
                    help="How to derive Z: 'div' (pure divergence), 'curv' (IG-LoG 2.5D), 'hybrid' (blend)")
    ap.add_argument("--vz25-mix", type=float, default=0.70,
                    help="Blend factor when vz-mode='hybrid' (0=all divergence, 1=all 2.5D curvature)")
    
        # global XY motion backend (dx/dy)
    ap.add_argument("--mode", choices=["ORB","HG","Hybrid"], default="Hybrid",
                    help="Global motion backend for dx/dy: FB=feature-based ORB affine (default), HG=IG-LoG structural phaseCorr+LK, Hybrid=soft-switch between FB/HG")

# Robust export (two-pass)
    ap.add_argument("--no-robust-export", action="store_true", help="Disable second-pass robust annotated export")
    ap.add_argument("--robust-z", type=float, default=6.0, help="MAD z-score threshold to drop spikes for normalization")
    ap.add_argument("--robust-prctl", type=float, default=95.0, help="Percentile for arrow scaling after outlier cull (e.g., 95)")
    ap.add_argument("--robust-gamma", type=float, default=1.0, help="Gamma on normalized magnitude")

    # Live overlay controls
    ap.add_argument("--live-lp-hz", type=float, default=None, help="Low-pass (Hz) for *live* overlay smoothing; default = profile lp_vel_hz")
    ap.add_argument("--v3-lock-s", type=float, default=1.0, help="Seconds before locking XY scale for 3D vector in first pass")

    # Burn-in smoothing / filtering

    ap.add_argument("--burnin-sec", type=float, default=2.0,
                help="Warm up internal motion/normalization stats for N seconds, then discard those frames and start outputs fresh.")
    ap.add_argument("--restart-after-burnin", type=int, default=1,
                help="After burn-in, reset running filters/alphas and start output fresh (1=yes,0=no).")

    # Hard-cut detection (structural phase correlation)
    ap.add_argument("--cut-resp-thresh", type=float, default=0.08,
                help="Hard cut if structural phaseCorr response falls below this (0..1-ish). Lower = less sensitive.")
    ap.add_argument("--cut-absdiff-thresh", type=float, default=0.18,
                help="Hard cut if structural mean absdiff exceeds this (0..1). Higher = less sensitive.")
    ap.add_argument("--cut-hold-frames", type=int, default=2,
                help="After detecting a cut, output neutral lanes for N frames and reset output-facing smoothers.")


    return ap

def resolve_profile(args):
    pf = args.profile
    for name, flag in [("cinematic", args.cinematic),("handheld", args.handheld),("action", args.action),
                       ("stabilized", args.stabilized),("vr_comfort", args.vr_comfort),("raw", args.raw)]:
        if flag: pf = name
    if pf not in PROFILES: pf = "cinematic"
    base = PROFILES[pf].copy()
    # position defaults per profile
    if pf in ("stabilized","vr_comfort"):
        base.update(dict(pos_dead=0.20, pos_lp_hz=0.20, pos_bias_tau_s=20.0, pos_burst_gain=0.4))
    elif pf in ("action","handheld"):
        base.update(dict(pos_dead=0.12, pos_lp_hz=0.30, pos_bias_tau_s=12.0, pos_burst_gain=0.6))
    else:
        base.update(dict(pos_dead=0.15, pos_lp_hz=0.25, pos_bias_tau_s=15.0, pos_burst_gain=0.5))
    # pass-throughs
    for k in ("z_div_gain","reaper_push","vz_mode","vz25_mix","mode","burnin_sec","restart_after_burnin","cut_resp_thresh","cut_absdiff_thresh","cut_hold_frames"):
        base[k] = getattr(args, k)
    base["robust_export"] = not getattr(args, "no_robust_export", False)
    for k in ("robust_z","robust_prctl","robust_gamma"): base[k] = getattr(args, k)
    base["p_lo"] = float(args.p_lo); base["p_hi"] = float(args.p_hi)
    base["ffmpeg_bin"] = args.ffmpeg_bin; base["vcodec"] = args.vcodec
    base["crf"] = args.crf; base["preset"] = args.preset; base["pix_fmt"] = args.pix_fmt
    # live overlay smoothing default
    base["live_lp_hz"] = args.live_lp_hz if args.live_lp_hz is not None else base["lp_vel_hz"]
    base["v3_lock_s"] = args.v3_lock_s
    ns = argparse.Namespace(**base); ns.profile_name = pf
    return ns

def main():
    ap = build_arg_parser(); cli = ap.parse_args()
    assert cli.video.is_file(), f"File not found: {cli.video}"
    prof = resolve_profile(cli)
    if cli.print_profile:
        from pprint import pprint; pprint({k: getattr(prof,k) for k in sorted(vars(prof).keys())}); return
    a = Analyzer(cli.video, prof, out_dir=cli.out_dir,
                 write_annotated=(not cli.no_annotated),
                 show_window=(not cli.no_window))
    while True:
        ok, idx, gui, row, metrics = a.next()
        if not ok:
            break
        # Analyzer.next() already appends to self.rows; don’t double-append
        # a.rows.append(row)

        if a.show_window:
            key = cv.waitKey(max(1, int(1000.0/a.fps))) & 0xFF

            # quit
            if key in (27, ord('q'), ord('Q')):
                break

            if key in (ord('d'), ord('D')):
                a.debug_flow_mode = 0 if a.debug_flow_mode else 1
                label = "OFF" if a.debug_flow_mode == 0 else "MAG+DIV+CURL (unified)"
                print(f"[debug] global flow overlay: {label}")




    a.cap.release()
    if a.vw is not None: a.vw.close()
    if a.show_window: cv.destroyAllWindows()
    a.finalize_and_export()
    # robust second-pass export unless disabled
    if getattr(a.args, "robust_export", False):
        try:
            a.render_export_robust()
        except Exception as e:
            print(f"[robust-export] failed: {e}")
    print(str(a.out_raw)); print(str(a.out_mod));
    if a.vw is not None: print(str(a.out_mp4))
    if getattr(a.args, "robust_export", False): print(str(a.out_mp4_robust))

if __name__ == "__main__":
    main()
