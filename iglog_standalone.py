import os
import cv2 as cv
import numpy as np
import math

# ---------------- IG-LoG core (copied from Local Motion Tracker.py) ----------------

def _to_u8_norm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-9:
        return np.zeros_like(a, dtype=np.uint8)
    x = (a - mn) / (mx - mn)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

def iglog_map(gray_u8: np.ndarray, sigma: float) -> np.ndarray:
    g = gray_u8.astype(np.float32) / 255.0
    k = int(max(3, int(math.ceil(sigma * 6)) | 1))
    gs = cv.GaussianBlur(g, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv.BORDER_REFLECT)
    log = cv.Laplacian(gs, ddepth=cv.CV_32F, ksize=3, borderType=cv.BORDER_REFLECT)

    sigma2 = max(0.8, sigma * 1.75)
    k2 = int(max(3, int(math.ceil(sigma2 * 6)) | 1))
    slow = cv.GaussianBlur(log, (k2, k2), sigmaX=sigma2, sigmaY=sigma2, borderType=cv.BORDER_REFLECT)
    return log - slow

def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def zero_crossings(log_f32: np.ndarray, thresh: float = 0.0) -> np.ndarray:
    a = log_f32
    s = np.sign(a).astype(np.int8)
    if thresh > 0:
        s[np.abs(a) < thresh] = 0
    H, W = s.shape[:2]
    z = np.zeros((H, W), dtype=np.uint8)
    hchg = (s[:, 1:] * s[:, :-1]) < 0
    z[:, 1:] |= (hchg.astype(np.uint8) * 255)
    vchg = (s[1:, :] * s[:-1, :]) < 0
    z[1:, :] |= (vchg.astype(np.uint8) * 255)
    return z

def extrema_map(log_f32: np.ndarray, abs_thresh: float) -> np.ndarray:
    a = np.abs(log_f32)
    if abs_thresh <= 0:
        abs_thresh = float(np.percentile(a, 95))
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dil = cv.dilate(a, k)
    peaks = (a >= dil - 1e-12) & (a >= abs_thresh)
    return (peaks.astype(np.uint8) * 255)

def _dens01(u8_bw: np.ndarray) -> float:
    return float(np.mean(u8_bw > 0))

def _u8_from_percentile(x_f32: np.ndarray, p: float = 99.0) -> np.ndarray:
    x = x_f32.astype(np.float32)
    s = float(np.percentile(x, p))
    if s < 1e-9:
        return np.zeros_like(x, dtype=np.uint8)
    y = np.clip(x * (255.0 / s), 0, 255)
    return y.astype(np.uint8)

def mass_grad_u8(gray_u8: np.ndarray, sigma_mass: float = 6.0, p: float = 99.0) -> np.ndarray:
    """
    Upgrade A: low-frequency gradient magnitude on a heavily blurred grayscale.
    Captures smooth shading-defined volume (hips, torso bulges) that IG-LoG drops.
    """
    g = gray_u8.astype(np.float32) / 255.0
    # heavy blur to kill texture and keep only mass shading
    gb = cv.GaussianBlur(g, (0, 0), sigmaX=sigma_mass, sigmaY=sigma_mass, borderType=cv.BORDER_REFLECT)
    gx = cv.Sobel(gb, cv.CV_32F, 1, 0, ksize=3, borderType=cv.BORDER_REFLECT)
    gy = cv.Sobel(gb, cv.CV_32F, 0, 1, ksize=3, borderType=cv.BORDER_REFLECT)
    mag = cv.magnitude(gx, gy)
    return _u8_from_percentile(mag, p=p)


def iglog_struct_maps_gray_u8(gray_u8: np.ndarray,
                              sigma: float = 1.2,
                              zc_thresh: float = 0.004,
                              ext_abs_thresh: float = 0.03):
    ig = iglog_map(gray_u8, sigma=float(sigma))
    abs_u8 = _u8_from_percentile(np.abs(ig), p=99.0)
    zc_u8  = zero_crossings(ig, thresh=float(zc_thresh))
    ext_u8 = extrema_map(ig, abs_thresh=float(ext_abs_thresh))

    clean = cv.add(zc_u8, ext_u8)
    rescue = np.clip(
        0.70 * abs_u8.astype(np.float32) +
        0.20 * zc_u8.astype(np.float32) +
        0.10 * ext_u8.astype(np.float32),
        0, 255
    ).astype(np.uint8)

    score = _dens01(zc_u8) + _dens01(ext_u8)
    return {"abs": abs_u8, "zc": zc_u8, "ext": ext_u8, "clean": clean, "rescue": rescue, "score": float(score)}

def adaptive_struct_map(abs_u8, zc_u8, ext_u8,
                        state: dict,
                        T_low=0.0015, T_high=0.0040,
                        decay=0.94,
                        abs_cap=200,
                        force_abs_on: bool=False,
                        warp_dx: float = 0.0,
                        warp_dy: float = 0.0):
    score = _dens01(zc_u8) + _dens01(ext_u8)

    abs_on = bool(state.get("abs_on", False))
    if force_abs_on:
        abs_on = True
    else:
        if (not abs_on) and score < T_low:
            abs_on = True
        elif abs_on and score > T_high:
            abs_on = False

    acc = state.get("abs_acc", None)
    if acc is None or acc.shape != abs_u8.shape:
        acc = abs_u8.astype(np.uint8)
    else:
        if (abs(warp_dx) + abs(warp_dy)) >= 0.5:
            M = np.float32([[1.0, 0.0, float(warp_dx)],
                            [0.0, 1.0, float(warp_dy)]])
            acc = cv.warpAffine(acc, M, (acc.shape[1], acc.shape[0]),
                                flags=cv.INTER_LINEAR,
                                borderMode=cv.BORDER_CONSTANT,
                                borderValue=0)
        acc = np.maximum((acc.astype(np.float32) * float(decay)).astype(np.uint8), abs_u8)

    if abs_cap is not None and abs_cap > 0:
        acc = np.minimum(acc, np.uint8(abs_cap))

    state["abs_acc"] = acc
    state["abs_on"]  = abs_on

    if abs_on:
        M_f = (0.70 * acc.astype(np.float32) +
               0.20 * zc_u8.astype(np.float32) +
               0.10 * ext_u8.astype(np.float32))
    else:
        M_f = (0.70 * zc_u8.astype(np.float32) +
               0.30 * ext_u8.astype(np.float32))

    M_u8 = np.clip(M_f, 0, 255).astype(np.uint8)
    return M_u8, abs_on, float(score)

# ---------------- end IG-LoG core ----------------

def downscale_max_side(frame_bgr: np.ndarray, max_side: int):
    if not max_side or max_side <= 0:
        return frame_bgr, 1.0
    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return frame_bgr, 1.0
    s = float(max_side) / float(m)
    out = cv.resize(frame_bgr, (int(w*s), int(h*s)), interpolation=cv.INTER_AREA)
    return out, s

def _to_bgr(u8: np.ndarray) -> np.ndarray:
    return cv.cvtColor(u8, cv.COLOR_GRAY2BGR)

def _label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    cv.putText(img_bgr, text, (15, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)
    return img_bgr

def make_grid_1x3(a_u8, b_u8, c_u8, la="A", lb="B", lc="C"):
    A = _label(_to_bgr(a_u8), la)
    B = _label(_to_bgr(b_u8), lb)
    C = _label(_to_bgr(c_u8), lc)
    return np.concatenate([A, B, C], axis=1)

def make_grid_2x3(top_u8s, bot_u8s, top_labels, bot_labels):
    top = make_grid_1x3(top_u8s[0], top_u8s[1], top_u8s[2], *top_labels)
    bot = make_grid_1x3(bot_u8s[0], bot_u8s[1], bot_u8s[2], *bot_labels)
    return np.concatenate([top, bot], axis=0)


def convert_video_to_iglog(video_path: str,
                           sigma: float = 1.2,
                           zc_thresh: float = 0.004,
                           ext_abs_thresh: float = 0.03,
                           force_abs_on: bool = True,
                           sigma_mass: float = 0.5,
                           codec: str = "mp4v", debug_mode: int = 0):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = float(cap.get(cv.CAP_PROP_FPS) or 30.0)
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)

    in_dir = os.path.dirname(os.path.abspath(video_path))
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(in_dir, f"{base}-iglog.mp4")

    fourcc = cv.VideoWriter_fourcc(*codec)
    #vw = cv.VideoWriter(out_path, fourcc, fps, (W, H), isColor=True)
    vw = None

    # if not vw.isOpened():
        # cap.release()
        # raise RuntimeError("VideoWriter failed. Try installing ffmpeg or changing codec.")

    st = {}  # <-- PERSIST across frames (the “less noisy” part)
    n = 0
    max_side = 480  # long edge cap; preserves aspect ratio

    while True:
        ok, fr = cap.read()
        if not ok:
            break

        gray = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
        maps = iglog_struct_maps_gray_u8(gray, sigma=sigma, zc_thresh=zc_thresh, ext_abs_thresh=ext_abs_thresh)
        #M_u8, abs_on, score = adaptive_struct_map(maps["abs"], maps["zc"], maps["ext"], st, force_abs_on=force_abs_on)
        # MASS channel
        mass_u8 = mass_grad_u8(gray, sigma_mass=sigma_mass, p=99.0)

        # --- build HYBRID = (MASS gated by ABS) * (ZC|EXT) ---
        abs_n  = maps["abs"].astype(np.float32) / 255.0
        mass_n = mass_u8.astype(np.float32) / 255.0

        w = smoothstep(abs_n, lo=0.15, hi=0.45)          # tune later
        hyb = w * mass_n + (1.0 - w) * abs_n

        valid_u8 = cv.max(maps["zc"], maps["ext"])          # 0 or 255
        valid_f  = valid_u8.astype(np.float32) / 255.0

        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        valid_u8 = cv.dilate(valid_u8, k, iterations=1)     # try 1..2


        # soften it: blur the gate so it becomes a permit field, not a guillotine
        valid_soft = cv.GaussianBlur(valid_f, (0,0), sigmaX=1.2, sigmaY=1.2, borderType=cv.BORDER_REFLECT)

        # keep some floor so it’s never pitch black
        valid_soft = 0.25 + 0.75 * valid_soft               # floor=0.25, tune 0.15..0.4
        hyb *= valid_soft

        # compensate for gating-induced darkening
        mean_gate = np.mean(valid_soft)
        if mean_gate > 1e-3:
            hyb /= mean_gate
        hyb = np.clip(hyb, 0.0, 1.0)



        hybrid_u8 = np.clip(hyb * 255.0, 0, 255).astype(np.uint8)

        # --- visualize ABS | MASS | HYBRID ---
        abs_bgr    = cv.cvtColor(maps["abs"], cv.COLOR_GRAY2BGR)
        mass_bgr   = cv.cvtColor(mass_u8,    cv.COLOR_GRAY2BGR)
        hybrid_bgr = cv.cvtColor(hybrid_u8,  cv.COLOR_GRAY2BGR)

        cv.putText(abs_bgr,    "ABS",    (15, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)
        cv.putText(mass_bgr,   "MASS",   (15, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)
        cv.putText(hybrid_bgr, "HYBRID", (15, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)

        zc_u8  = maps["zc"]
        ext_u8 = maps["ext"]
        zc_or_ext = cv.max(zc_u8, ext_u8)

        if debug_mode == 0:
            # HYBRID only (still BGR for writer)
            combo = _to_bgr(hybrid_u8)
            combo = _label(combo, "HYBRID")

        elif debug_mode == 1:
            # 1x3: ABS | MASS | HYBRID
            combo = make_grid_1x3(maps["abs"], mass_u8, hybrid_u8, "ABS", "MASS", "HYBRID")

        elif debug_mode == 2:
            # 1x3: ZC | EXT | ZC|EXT
            combo = make_grid_1x3(zc_u8, ext_u8, zc_or_ext, "ZC", "EXT", "ZC|EXT")

        else:
            # 2x3 grid: top ABS/MASS/HYBRID, bottom ZC/EXT/ZC|EXT
            combo = make_grid_2x3(
                (maps["abs"], mass_u8, hybrid_u8),
                (zc_u8, ext_u8, zc_or_ext),
                ("ABS", "MASS", "HYBRID"),
                ("ZC", "EXT", "ZC|EXT")
            )

        if vw is None:
            out_h, out_w = combo.shape[:2]
            vw = cv.VideoWriter(out_path, fourcc, fps, (out_w, out_h), isColor=True)
            if not vw.isOpened():
                cap.release()
                raise RuntimeError("VideoWriter failed. Try installing ffmpeg or changing codec.")
        vw.write(combo)

        n += 1
        if n % 120 == 0:
            print(f"[iglog] frames={n}")
        st = {}

    vw.release()
    cap.release()
    print(f"[iglog] wrote: {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--sigma", type=float, default=1.2)
    ap.add_argument("--zc", type=float, default=0.004)
    ap.add_argument("--ext", type=float, default=0.03)
    ap.add_argument("--force-abs", action="store_true", default=True)
    ap.add_argument("--sigma-mass", type=float, default=0.5)
    ap.add_argument("--debug", type=int, default=0,
                help="0=HYBRID only, 1=ABS|MASS|HYBRID, 2=ZC|EXT|ZC|EXT, 3=2x3 grid")

    args = ap.parse_args()

    convert_video_to_iglog(args.video, sigma=args.sigma, zc_thresh=args.zc, ext_abs_thresh=args.ext, force_abs_on=args.force_abs, sigma_mass=args.sigma_mass, debug_mode=args.debug)