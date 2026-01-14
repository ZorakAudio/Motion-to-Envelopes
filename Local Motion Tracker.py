#!/usr/bin/env python3
# ROI_Flux_Tracker_v8e_hotfix.py
# v8e hotfix: box-only tracker, preview vectors while paused, explicit "Add ROI" and "Repick ROI" modes,
#             Shift+Drag bound, PLAY-only sampling, stable export with speed/acc/jerk lanes.


import os, sys, json, csv, math, time, re, base64, hashlib, textwrap, pickle, zlib, struct
from dataclasses import dataclass, field, replace
from typing import List, Tuple, Optional, Dict, Any
import threading


import numpy as np
import cv2 as cv
import pandas as pd
import time, math, os


APP_NAME = "Local Motion Tracker"


# ============================== Optional OpenAI Assist ==============================
# This tracker can optionally use the OpenAI API to *assist* with:
#   - ROI tag suggestions (cheap text+vision, default: gpt-5-nano)
#   - Impact inference / labeling from exported motion series (gpt-5-nano)
#   - Flux cleanup suggestions given inferred impacts (gpt-5-nano)
#   - Occlusion recovery / "where did it go?" (gpt-5-mini, vision opt-in)
#   - Outline polygons for per-ROI masking (gpt-5-mini, vision opt-in)
#   - Arbitration on low-confidence outputs (heavy: gpt-5.2)
#
# SAFETY DEFAULTS ("context-ambiguous"):
#   - NO images are sent to OpenAI unless explicitly enabled per run with --ai-vision.
#   - Even with --ai-vision, the default is --ai-vision edges (edge maps) to reduce context.
#   - Explicit / descriptive outputs are OFF unless --ai-explicit is set.
#
# Users supply their *own* OPENAI_API_KEY. Never hard-code keys.
#
# NOTE: This file is intentionally self-contained (no extra modules) so patching is easy.

OPENAI_POLICY_DISCLAIMER = textwrap.dedent("""\
    ---- OpenAI API Key Disclaimer (Read This) ----
    This tool can optionally send text and/or images to the OpenAI API when AI assist is enabled.
    You are responsible for:
      • Complying with OpenAI Usage Policies.
      • Only sending content you have rights/consent to share.
      • Keeping your API key private (treat it like a password).

    High-risk content (do NOT send):
      • Any sexual content involving minors (CSAM) — prohibited and can trigger reporting.
      • Content that facilitates wrongdoing, violence, self-harm, or harassment.
      • Anything you wouldn't be comfortable having safety systems review.

    This app defaults to an "ambiguous" mode (no images; neutral prompts).
    To enable image-based assist you must opt in per-run with --ai-vision.
    -----------------------------------------------
""")

@dataclass
class AIConfig:
    enabled: bool = False
    # Vision sending is OFF by default. If enabled, choose what is sent:
    #   "off"   -> never send images
    #   "edges" -> edge maps of ROI crops (lower-context)
    #   "crop"  -> ROI crop (downscaled)
    #   "full"  -> full frame (downscaled)  [highest context / highest risk]
    vision: str = "off"
    explicit: bool = False           # allow explicit semantic labels / descriptions (off by default)
    on_export: bool = False          # run AI inference on export and write sidecar JSON
    auto_apply_tags: bool = False    # if True: fills empty ROI names with AI suggestions
    auto_fix_occlusion: bool = False # if True: allows AI to move the ROI rect on occlusion solve

    # Models (overrideable via CLI)
    model_nano: str = "gpt-5-nano"
    model_mini: str = "gpt-5-mini"
    model_heavy: str = "gpt-5.2"

    # Reasoning effort (gpt-5 + o-series models)
    reasoning_light: str = "minimal"
    reasoning_heavy: str = "high"

    # Safety identifier (hash of a stable user id) for abuse detection (recommended)
    safety_identifier: Optional[str] = None

    # Control knobs
    max_calls: int = 60
    min_interval_s: float = 0.75
    cache_path: str = os.path.join(os.path.expanduser("~"), ".cache", "local_motion_tracker_openai_cache.json")

    # Require an explicit acknowledgment flag before any API calls happen
    require_policy_ack: bool = True
    policy_ack: bool = False


# ============================== Undo/Redo (project state) ==============================
# Undo/Redo is meant for edit operations (ROIs, scene boundaries, labels), not for "playback time travel".
# We snapshot the current project state *before* mutating operations.
#
# Design goals:
#   - Stable: never crashes the app if a snapshot fails to serialize.
#   - Lightweight: snapshots are zlib-compressed pickles.
#   - Coalescing: rapid repeated adjustments (wheel nudges, etc.) collapse into one undo step.

class UndoRedoStack:
    def __init__(self, max_states: int = 30, coalesce_window_s: float = 0.75):
        self.max_states = int(max(1, max_states))
        self.coalesce_window_s = float(max(0.0, coalesce_window_s))
        self._undo = []   # list[(t, label, coalesce_key, blob)]
        self._redo = []   # list[(t, label, coalesce_key, blob)]
        self._in_restore = False

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def push_undo(self, state_obj, label: str = "", coalesce_key: Optional[str] = None) -> None:
        if self._in_restore:
            return
        now = time.time()

        # Coalesce: keep the first "before" snapshot for a burst of changes.
        if coalesce_key and self._undo:
            t_prev, _, c_prev, _ = self._undo[-1]
            if c_prev == coalesce_key and (now - t_prev) <= self.coalesce_window_s:
                return

        try:
            blob = zlib.compress(
                pickle.dumps(state_obj, protocol=pickle.HIGHEST_PROTOCOL),
                level=6
            )
        except Exception as e:
            print(f"[undo] snapshot failed: {e}")
            return

        self._undo.append((now, str(label or ""), str(coalesce_key or ""), blob))
        if len(self._undo) > self.max_states:
            self._undo = self._undo[-self.max_states:]
        self._redo.clear()

    def undo(self, current_state_obj):
        if not self._undo:
            return None, None

        try:
            cur_blob = zlib.compress(pickle.dumps(current_state_obj, protocol=pickle.HIGHEST_PROTOCOL), level=6)
        except Exception:
            cur_blob = None

        t, label, ckey, blob = self._undo.pop()
        if cur_blob is not None:
            self._redo.append((time.time(), label, ckey, cur_blob))
            if len(self._redo) > self.max_states:
                self._redo = self._redo[-self.max_states:]

        try:
            self._in_restore = True
            state = pickle.loads(zlib.decompress(blob))
        finally:
            self._in_restore = False
        return state, label

    def redo(self, current_state_obj):
        if not self._redo:
            return None, None

        try:
            cur_blob = zlib.compress(pickle.dumps(current_state_obj, protocol=pickle.HIGHEST_PROTOCOL), level=6)
        except Exception:
            cur_blob = None

        t, label, ckey, blob = self._redo.pop()
        if cur_blob is not None:
            self._undo.append((time.time(), label, ckey, cur_blob))
            if len(self._undo) > self.max_states:
                self._undo = self._undo[-self.max_states:]

        try:
            self._in_restore = True
            state = pickle.loads(zlib.decompress(blob))
        finally:
            self._in_restore = False
        return state, label

def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _ensure_parent_dir(path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass


def _parse_bool_flag(argv: List[str], flag: str) -> bool:
    return flag in argv


def _parse_cli_ai(argv: List[str]) -> AIConfig:
    """
    Parse AI flags from argv (everything after the video path).
    We intentionally keep this lightweight (no argparse) so this remains a single-file tool.
    """
    cfg = AIConfig()

    if "--ai" in argv:
        cfg.enabled = True

    # opt-in acknowledgment
    if "--i-accept-openai-policy" in argv:
        cfg.policy_ack = True

    # vision: --ai-vision [off|edges|crop|full]
    if "--ai-vision" in argv:
        cfg.vision = "edges"
        try:
            i = argv.index("--ai-vision")
            if i + 1 < len(argv) and not argv[i+1].startswith("--"):
                cfg.vision = str(argv[i+1]).strip().lower()
        except Exception:
            pass

    if "--ai-explicit" in argv:
        cfg.explicit = True

    if "--ai-on-export" in argv:
        cfg.on_export = True

    if "--ai-auto-apply-tags" in argv:
        cfg.auto_apply_tags = True

    if "--ai-auto-fix-occlusion" in argv:
        cfg.auto_fix_occlusion = True

    # --ai-user <stable_id>  (we hash it before sending)
    if "--ai-user" in argv:
        try:
            i = argv.index("--ai-user")
            if i + 1 < len(argv):
                cfg.safety_identifier = _sha256_hex(str(argv[i+1]))
        except Exception:
            pass

    # --ai-cache <path>
    if "--ai-cache" in argv:
        try:
            i = argv.index("--ai-cache")
            if i + 1 < len(argv):
                cfg.cache_path = str(argv[i+1])
        except Exception:
            pass

    # --ai-max-calls <int>
    if "--ai-max-calls" in argv:
        try:
            i = argv.index("--ai-max-calls")
            if i + 1 < len(argv):
                cfg.max_calls = int(argv[i+1])
        except Exception:
            pass

    # --ai-min-interval <float>
    if "--ai-min-interval" in argv:
        try:
            i = argv.index("--ai-min-interval")
            if i + 1 < len(argv):
                cfg.min_interval_s = float(argv[i+1])
        except Exception:
            pass

    # model overrides
    for k, attr in [
        ("--ai-model-nano", "model_nano"),
        ("--ai-model-mini", "model_mini"),
        ("--ai-model-heavy", "model_heavy"),
    ]:
        if k in argv:
            try:
                i = argv.index(k)
                if i + 1 < len(argv):
                    setattr(cfg, attr, str(argv[i+1]))
            except Exception:
                pass

    # normalize
    cfg.vision = str(cfg.vision or "off").lower()
    if cfg.vision not in ("off", "edges", "crop", "full"):
        cfg.vision = "edges"  # safest meaningful default if user typoed
    return cfg


def _qt_sanitize_argv(full_argv: List[str]) -> List[str]:
    """
    Qt can complain about unknown flags. Strip our app-specific flags before passing argv to QApplication.
    Keeps Qt's own flags intact (like -platform, -style, etc.).
    """
    # flags that take a value
    takes_value = {
        "--scale",
        "--ai-vision",
        "--ai-user",
        "--ai-cache",
        "--ai-max-calls",
        "--ai-min-interval",
        "--ai-model-nano",
        "--ai-model-mini",
        "--ai-model-heavy",
    }
    # boolean flags
    bool_flags = {
        "--ai",
        "--ai-explicit",
        "--ai-on-export",
        "--ai-auto-apply-tags",
        "--ai-auto-fix-occlusion",
        "--i-accept-openai-policy",
    }

    out: List[str] = []
    i = 0
    while i < len(full_argv):
        a = full_argv[i]
        if a in bool_flags:
            i += 1
            continue
        if a in takes_value:
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def _bgr_to_data_url(frame_bgr: np.ndarray, max_side: int = 768, mode: str = "crop") -> str:
    """
    Convert a BGR image to a PNG data URL. Optionally downscale to limit payload size.
    mode:
      - "crop"/"full": send luminance-ish image
      - "edges": send LOW-CONTEXT structural map (GSCM/IG-LoG hybrid) instead of Canny
    """
    img = frame_bgr
    try:
        h, w = img.shape[:2]
        s = float(max(h, w))
        if s > max_side and s > 0:
            scale = max_side / s
            img = cv.resize(img, (int(round(w*scale)), int(round(h*scale))), interpolation=cv.INTER_AREA)

        if mode == "edges":
            # ---- LOW-CONTEXT VISION PAYLOAD (better than Canny) ----
            # We send a structural map derived from grayscale:
            #   - IG-LoG abs / ZC / extrema
            #   - fused into GSCM hybrid (mass-grad carrier + authority weighting)
            # This preserves geometry while removing most "semantic" content.
            g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # struct_M_from_gray_u8 already prefers GSCM hybrid internally
            # and falls back to adaptive IG-LoG composite if needed.
            try:
                st = {}  # local state is fine for a single still
                M = struct_M_from_gray_u8(g, st)  # uint8
                if M is None or M.size == 0:
                    raise RuntimeError("struct_M_from_gray_u8 returned empty")
            except Exception:
                # fallback: use the single-frame IG-LoG pipeline directly
                maps = iglog_struct_maps_gray_u8(g, sigma=1.2)
                M = maps.get("hybrid", None) or maps.get("rescue", None) or maps.get("clean", None) or g

            # ship as 3-channel for consistent downstream handling
            img = cv.cvtColor(M.astype(np.uint8), cv.COLOR_GRAY2BGR)
            # --------------------------------------------------------

        ok, buf = cv.imencode(".png", img)
        if not ok:
            raise RuntimeError("cv.imencode failed")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return "data:image/png;base64," + b64
    except Exception:
        # last resort: tiny black image
        blank = np.zeros((8, 8, 3), np.uint8)
        ok, buf = cv.imencode(".png", blank)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return "data:image/png;base64," + b64

def _roi_struct_desc_from_frame(frame_bgr: np.ndarray,
                               bbox_xywh: Tuple[int,int,int,int],
                               pad_px: int = 8,
                               out_hw: Tuple[int,int] = (48, 48),
                               sigma: float = 1.2) -> np.ndarray:
    """
    Cheap, stable descriptor from IG-LoG structural map (u8), L2-normalized.
    Uses the existing iglog_struct_maps_u8() pipeline (clean/rescue maps).
    """
    H, W = frame_bgr.shape[:2]
    x, y, w, h = map(int, bbox_xywh)
    x0 = max(0, x - pad_px); y0 = max(0, y - pad_px)
    x1 = min(W, x + w + pad_px); y1 = min(H, y + h + pad_px)
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return np.zeros(out_hw[0] * out_hw[1], np.float32)

    maps = iglog_struct_maps_u8(crop, sigma=float(sigma))
    # "rescue" is usually most robust; "clean" is sharper but can starve.
    M = maps.get("rescue", None)
    if M is None:
        return np.zeros(out_hw[0] * out_hw[1], np.float32)

    M = cv.resize(M, (int(out_hw[1]), int(out_hw[0])), interpolation=cv.INTER_AREA).astype(np.float32)
    M -= float(M.mean())
    n = float(np.linalg.norm(M)) + 1e-9
    M /= n
    return M.reshape(-1).astype(np.float32)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    if a.size != b.size or a.size == 0:
        return -1.0
    return float(np.dot(a, b))


def local_match_rois(frameA_bgr: np.ndarray,
                     roisA_xywh: List[Tuple[int,int,int,int]],
                     frameB_bgr: np.ndarray,
                     roisB_xywh: List[Tuple[int,int,int,int]],
                     *,
                     pad_px: int = 8,
                     sim_gate: float = 0.72,
                     ambig_margin: float = 0.06) -> Dict[str, Any]:
    """
    Returns:
      {
        "pairs": [{"a": i, "b": j, "sim": s, "ambig": bool}, ...],  # 1:1 greedy
        "unmatched_a": [...],
        "unmatched_b": [...],
        "ambiguous": [{"a": i, "cands": [{"b": j, "sim": s}, ...]} ...]  # for oracle
      }
    """
    A = [ _roi_struct_desc_from_frame(frameA_bgr, rc, pad_px=pad_px) for rc in roisA_xywh ]
    B = [ _roi_struct_desc_from_frame(frameB_bgr, rc, pad_px=pad_px) for rc in roisB_xywh ]

    nA, nB = len(A), len(B)
    if nA == 0 or nB == 0:
        return {"pairs": [], "unmatched_a": list(range(nA)), "unmatched_b": list(range(nB)), "ambiguous": []}

    # sim matrix
    S = np.full((nA, nB), -1.0, np.float32)
    for i in range(nA):
        for j in range(nB):
            S[i, j] = _cos_sim(A[i], B[j])

    # per-A ambiguity probe (top1-top2)
    ambiguous = []
    for i in range(nA):
        row = S[i]
        order = np.argsort(row)[::-1]
        top1 = float(row[order[0]])
        top2 = float(row[order[1]]) if nB >= 2 else -1.0
        if top1 >= sim_gate and (top1 - top2) < ambig_margin:
            cands = [{"b": int(order[k]), "sim": float(row[order[k]])} for k in range(min(4, nB))]
            ambiguous.append({"a": int(i), "cands": cands})

    # greedy 1:1 assignment by global best edges
    edges = []
    for i in range(nA):
        for j in range(nB):
            s = float(S[i, j])
            if s >= sim_gate:
                edges.append((s, i, j))
    edges.sort(reverse=True)

    usedA = set()
    usedB = set()
    pairs = []
    for s, i, j in edges:
        if i in usedA or j in usedB:
            continue
        usedA.add(i); usedB.add(j)
        pairs.append({"a": int(i), "b": int(j), "sim": float(s), "ambig": False})

    unmatched_a = [i for i in range(nA) if i not in usedA]
    unmatched_b = [j for j in range(nB) if j not in usedB]

    return {"pairs": pairs, "unmatched_a": unmatched_a, "unmatched_b": unmatched_b, "ambiguous": ambiguous}


class OpenAIAssist:
    """
    Thin wrapper around the OpenAI Python SDK (Responses API + Moderations).
    Everything is optional; if the SDK isn't installed or the key isn't set, calls no-op safely.
    """
    def __init__(self, cfg: AIConfig):
        self.cfg = cfg
        self._client = None
        self._last_call_t = 0.0
        self._calls = 0
        self._cache: Dict[str, Any] = {}
        self.last_error: str = ""

        if not cfg.enabled:
            return

        if cfg.require_policy_ack and not cfg.policy_ack:
            self.last_error = "AI disabled: missing --i-accept-openai-policy"
            return

        if not os.environ.get("OPENAI_API_KEY"):
            self.last_error = "AI disabled: OPENAI_API_KEY not set"
            return

        try:
            from openai import OpenAI
            self._client = OpenAI()
        except Exception as e:
            self.last_error = f"AI disabled: failed to import/init OpenAI SDK: {e}"
            self._client = None
            return

        # load cache (best-effort)
        try:
            if self.cfg.cache_path:
                _ensure_parent_dir(self.cfg.cache_path)
                if os.path.exists(self.cfg.cache_path):
                    with open(self.cfg.cache_path, "r", encoding="utf-8") as f:
                        self._cache = json.load(f) or {}
        except Exception:
            self._cache = {}

    def ready(self) -> bool:
        return self._client is not None and self.cfg.enabled and (not self.cfg.require_policy_ack or self.cfg.policy_ack)

    def _rate_limit(self) -> bool:
        now = time.time()
        if self._calls >= int(self.cfg.max_calls):
            self.last_error = "AI call budget exceeded for this run"
            return False
        if (now - self._last_call_t) < float(self.cfg.min_interval_s):
            self.last_error = "AI rate-limited (min interval)"
            return False
        self._last_call_t = now
        self._calls += 1
        return True

    def _cache_get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def _cache_put(self, key: str, value: Any) -> None:
        self._cache[key] = value
        try:
            if self.cfg.cache_path:
                _ensure_parent_dir(self.cfg.cache_path)
                with open(self.cfg.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f)
        except Exception:
            pass

    def _moderate_ok(self, text_in: Optional[str] = None, image_data_urls: Optional[List[str]] = None) -> bool:
        """
        Best-effort moderation gate. If moderation fails, default to "allow" (avoid bricking the app),
        but still keeps vision opt-in per run.
        """
        if not self.ready():
            return False
        try:
            items: List[Dict[str, Any]] = []
            if text_in:
                items.append({"type": "text", "text": text_in})
            if image_data_urls:
                for u in image_data_urls:
                    items.append({"type": "image_url", "image_url": {"url": u}})
            if not items:
                return True
            mod = self._client.moderations.create(model="omni-moderation-latest", input=items)
            flagged = bool(mod.results[0].flagged)
            if flagged:
                # Do not forward flagged content by default.
                cats = getattr(mod.results[0], "categories", None)
                # if category["sexual"] is True but nothing else is, this is a yellow-flag - proceed
                if cats and cats.get("sexual", False) and not any(cats.get(k, False) for k in cats if k != "sexual"):
                    return True
                self.last_error = f"Moderation flagged input; blocked. categories={cats}"
                return False
            return True
        except Exception:
            return True

    def _extract_output_text(self, resp: Any) -> str:
        if resp is None:
            return ""
        # New SDKs expose output_text; keep a robust fallback.
        try:
            out = getattr(resp, "output_text", None)
            if isinstance(out, str) and out.strip():
                return out
        except Exception:
            pass
        try:
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") in ("output_text", "text"):
                            t = getattr(c, "text", "")
                            if t:
                                return t
        except Exception:
            pass
        return str(resp)

    def call_json(self,
                  *,
                  model: str,
                  instructions: str,
                  user_text: str,
                  format_name: str,
                  schema: Dict[str, Any],
                  images: Optional[List[str]] = None,
                  reasoning_effort: Optional[str] = None,
                  max_output_tokens: int = 800) -> Optional[Dict[str, Any]]:
        """
        Structured output call (JSON Schema) via Responses API.
        """
        if not self.ready():
            return None
        if not self._rate_limit():
            return None

        images = images or []
        if images and not self._moderate_ok(text_in=user_text, image_data_urls=images):
            return None

        # cache key includes prompt + schema + model (but not api key)
        ck = _sha256_hex(json.dumps({
            "m": model,
            "i": instructions,
            "u": user_text,
            "s": schema,
            "img": [ _sha256_hex(u) for u in images ],
            "r": reasoning_effort,
        }, sort_keys=True))
        cached = self._cache_get(ck)
        if isinstance(cached, dict):
            return cached

        try:
            content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_text}]
            for u in images:
                content.append({"type": "input_image", "image_url": u})

            kwargs: Dict[str, Any] = dict(
                model=model,
                instructions=instructions,
                input=[{"role": "user", "content": content}],
                store=False,
                max_output_tokens=int(max_output_tokens),
                text={"format": {"type": "json_schema", "strict": True, "name": format_name,  "schema": schema}},
            )
            if self.cfg.safety_identifier:
                kwargs["safety_identifier"] = self.cfg.safety_identifier
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}

            resp = self._client.responses.create(**kwargs)
            txt = self._extract_output_text(resp).strip()
            data = json.loads(txt)
            self._cache_put(ck, data)
            return data
        except Exception as e:
            self.last_error = f"OpenAI call failed: {e}"
            return None

    # ---------------- Task wrappers (model routing) ----------------

    def suggest_roi_tags(self, *,
                         roi_summaries: List[Dict[str, Any]],
                         images: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rois": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "roi_index": {"type": "integer"},
                            "suggested_name": {"type": "string"},
                            
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["roi_index", "suggested_name", "confidence", "reason"]
                    }
                }
            },
            "required": ["rois"]
        }
        instructions = (
            "You assist a motion-tracking tool. Be concise. "
            "Return neutral, non-explicit labels. "
            "If uncertain, use generic labels like 'subject', 'limb', 'object', 'region'."
        )
        if self.cfg.explicit:
            instructions = (
                "You assist a motion-tracking tool. Be concise. "
                "Return short labels suitable as ROI names."
            )
        user_text = json.dumps({"task": "suggest_roi_tags", "rois": roi_summaries}, ensure_ascii=False)

        # Use nano by default; if images are included and nano struggles, heavy can arbitrate later.
        return self.call_json(
            model=self.cfg.model_mini,
            instructions=instructions,
            user_text=user_text,
            format_name="roi_tag_suggestions",
            images=images,
            schema=schema,
            reasoning_effort=self.cfg.reasoning_light,
            max_output_tokens=700
        )

    def infer_impacts(self, *, motion_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "t": {"type": "number"},
                            "roi_index": {"type": "integer"},
                            "event": {"type": "string", "enum": ["impact_in", "impact_out", "peak_speed", "stop", "other"]},
                            "strength": {"type": "number"},
                            
                            "notes": {"type": "string"},
                        },
                        "required": ["t", "roi_index", "event", "strength"]
                    }
                }
            },
            "required": ["events"]
        }
        instructions = (
            "You analyze motion time-series exported from a tracker and propose candidate impact events. "
            "Do not invent extra ROIs. Prefer fewer, higher-confidence events. "
            "Return strengths in [0,1]."
        )
        user_text = json.dumps({"task": "infer_impacts", "motion": motion_summary}, ensure_ascii=False)
        return self.call_json(
            model=self.cfg.model_nano,
            instructions=instructions,
            user_text=user_text,
            format_name="infer_impacts",
            schema=schema,
            reasoning_effort=self.cfg.reasoning_light,
            max_output_tokens=900
        )

    def suggest_flux_fixes(self, *, motion_summary: Dict[str, Any], impacts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "global": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "suggested_smoothing_ms": {"type": "number"},
                        
                        "notes": {"type": "string"},
                    },
                    "required": ["suggested_smoothing_ms"]
                },
                "per_roi": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "roi_index": {"type": "integer"},
                            "suggested_deadband": {"type": "number"},
                            "suggested_min_speed_ps": {"type": "number"},
                            
                            "notes": {"type": "string"},
                        },
                        "required": ["roi_index", "suggested_min_speed_ps"]
                    }
                }
            },
            "required": ["global", "per_roi"]
        }
        instructions = (
            "You propose parameter tweaks to make flux/velocity signals less noisy without erasing real impacts. "
            "Prefer small changes. Output numbers only; no code."
        )
        user_text = json.dumps({"task": "suggest_flux_fixes", "motion": motion_summary, "impacts": impacts}, ensure_ascii=False)
        return self.call_json(
            model=self.cfg.model_nano,
            instructions=instructions,
            user_text=user_text,
            schema=schema,
            format_name="flux_fix_suggest",
            reasoning_effort=self.cfg.reasoning_light,
            max_output_tokens=900
        )

    def solve_occlusion(self, *,
                        roi_index: int,
                        prior_bbox_xywh: List[int],
                        context_text: str,
                        images: List[str]) -> Optional[Dict[str, Any]]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "roi_index": {"type": "integer"},
                "status": {"type": "string", "enum": ["visible", "occluded", "gone", "uncertain"]},
                "bbox_xywh_in_crop": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4
                },
                "confidence": {"type": "number"},
                "notes": {"type": "string"},
            },
            "required": ["roi_index", "status", "bbox_xywh_in_crop", "confidence"]
        }
        instructions = (
            "You help recover a tracked region across frames. "
            "Return a best-guess bbox (x,y,w,h) in the CURRENT CROP coordinates (relative to the provided images). "
            "If the object is fully occluded, keep bbox near prior location and set status='occluded'."
        )
        if not self.cfg.explicit:
            instructions += " Avoid describing content; focus only on geometry and motion."

        user_text = json.dumps({
            "task": "solve_occlusion",
            "roi_index": roi_index,
            "prior_bbox_xywh": prior_bbox_xywh,
            "context": context_text,
        }, ensure_ascii=False)

        # vision is required here
        out = self.call_json(
            model=self.cfg.model_mini,
            instructions=instructions,
            user_text=user_text,
            images=images,
            format_name="solve_occlusion",
            schema=schema,
            reasoning_effort=self.cfg.reasoning_light,
            max_output_tokens=700
        )

        # Heavy arbiter if low confidence
        if out is not None:
            try:
                if float(out.get("confidence", 0.0)) < 0.55:
                    out2 = self.call_json(
                        model=self.cfg.model_heavy,
                        instructions=instructions + " Take extra care; return your best bbox even if uncertain.",
                        user_text=user_text,
                        images=images,
                        format_name="solve_occlusion",
                        schema=schema,
                        reasoning_effort=self.cfg.reasoning_heavy,
                        max_output_tokens=800
                    )
                    if out2 is not None and float(out2.get("confidence", 0.0)) >= float(out.get("confidence", 0.0)):
                        return out2
            except Exception:
                pass
        return out

    def match_rois_oracle(self, *,
                          roisA: List[Dict[str, Any]],
                          roisB: List[Dict[str, Any]],
                          local: Dict[str, Any],
                          images: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        roisA/roisB: [{ "roi_index": i, "bbox_xywh": [x,y,w,h], "name": "ROI A" }, ...]
        local: output of local_match_rois(...), especially ["ambiguous"]
        images: optional; obeys cfg.vision. If provided, send edge/crop/full via your existing pipeline.
        """
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": ["integer", "null"]},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["a", "b", "confidence", "reason"]
                    }
                }
            },
            "required": ["pairs"]
        }

        instructions = (
            "Match ROI identities across two scenes using ONLY geometry/structure. "
            "Do NOT describe content. Prefer b=null over a wrong match. "
            "Enforce one-to-one matches across the returned pairs. "
            "Use the provided local candidates as hints, but override if clearly wrong."
        )

        user_text = json.dumps({
            "task": "match_rois",
            "roisA": roisA,
            "roisB": roisB,
            "local": local,
        }, ensure_ascii=False)

        out = self.call_json(
            model=self.cfg.model_mini,
            instructions=instructions,
            user_text=user_text,
            format_name="match_rois",
            images=images or [],
            schema=schema,
            reasoning_effort=self.cfg.reasoning_light,
            max_output_tokens=800
        )

        # heavy arbiter if weak
        if out is not None:
            try:
                confs = [float(p.get("confidence", 0.0)) for p in (out.get("pairs") or [])]
                minc = min(confs) if confs else 0.0
                if minc < 0.60:
                    out2 = self.call_json(
                        model=self.cfg.model_heavy,
                        instructions=instructions + " Take extra care; do not force matches.",
                        user_text=user_text,
                        format_name="match_rois",
                        images=images or [],
                        schema=schema,
                        reasoning_effort=self.cfg.reasoning_heavy,
                        max_output_tokens=900
                    )
                    if out2 is not None:
                        return out2
            except Exception:
                pass
        return out


    def outline_object(self, *,
                       roi_index: int,
                       context_text: str,
                       image: str) -> Optional[Dict[str, Any]]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "roi_index": {"type": "integer"},
                "polygons_norm": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "minItems": 3
                    }
                },
                "confidence": {"type": "number"},
                "notes": {"type": "string"},
            },
            "required": ["roi_index", "polygons_norm", "confidence", "notes"]
        }
        instructions = (
            "Given a single ROI crop, return one or more polygons that outline the main tracked subject. "
            "Return coordinates normalized to the crop: (u,v) in [0,1]. "
            "If unsure, return an empty list with low confidence."
        )
        if not self.cfg.explicit:
            instructions += " Avoid describing content; focus only on shapes."

        user_text = json.dumps({
            "task": "outline_object",
            "roi_index": roi_index,
            "context": context_text,
        }, ensure_ascii=False)

        out = self.call_json(
            model=self.cfg.model_mini,
            instructions=instructions,
            user_text=user_text,
            format_name="outline_object",
            images=[image],
            schema=schema,
            reasoning_effort=self.cfg.reasoning_light,
            max_output_tokens=700
        )

        # Heavy arbiter if low confidence
        if out is not None:
            try:
                if float(out.get("confidence", 0.0)) < 0.55:
                    out2 = self.call_json(
                        model=self.cfg.model_heavy,
                        instructions=instructions + " Take extra care; better outline beats verbosity.",
                        user_text=user_text,
                        images=[image],
                        format_name="outline_object",
                        schema=schema,
                        reasoning_effort=self.cfg.reasoning_heavy,
                        max_output_tokens=800
                    )
                    if out2 is not None and float(out2.get("confidence", 0.0)) >= float(out.get("confidence", 0.0)):
                        return out2
            except Exception:
                pass
        return out

# ============================ End Optional OpenAI Assist ============================
# --- EXPORT overlay hold (global) ---w
# Solid on‑screen time for an exported impact flash.
# Applies only to the MP4 overlay made by export_fullpass_overlay_and_csv().
# Set to e.g. 220 (ms). Set to 0 to disable extended hold.
EXPORT_HOLD_MS = 220

# ---------- flow + draw config ----------
SCALE = 0.65
FB = dict(
    pyr_scale=0.45,          # coarser pyramid helps through noise
    levels=7,                # default; still adjustable per-ROI
    winsize=41,              # wider averaging over noise
    iterations=5,
    poly_n=7,
    poly_sigma=1.5,
    flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN  # <- was 0
)

# ---------- CMAT (camera motion compensation) structural gating ----------
# When enabled, CMAT drift is estimated from IG-LoG structural maps and
# background-like pixels (low flow magnitude + enough structure), so large
# foreground motion (character fills frame) doesn't hijack camera drift.
CMAT_STRUCT_MODE = True
CMAT_STRUCT_MIX_GRAY = 0.0      # 0..1 add a little raw gray back into structural map (0 = pure structural)
CMAT_IGLOG_SIGMA = 1.2          # IG-LoG sigma for structural maps
CMAT_BG_MAG_PCTL = 55.0         # low-motion percentile used as "background-like"
CMAT_BG_CURV_GATE = 0.20        # require abs(IGLoG) strength >= gate (0..1); set 0 to disable
CMAT_BG_MIN_PIXELS = 800        # minimum bg pixels after gating before trusting bg-only drift
CMAT_MAX_H = 240          # target height for camera estimation
CMAT_MIN_H = 64           # avoid absurdly tiny
CMAT_FB_LEVELS = 5
CMAT_FB_WINSIZE = 21


# ---------- CMAT helpers: structural salience + weighted pooling ----------
def _robust01_from_u8_absdiff(a_u8: np.ndarray, b_u8: np.ndarray,
                             p_lo: float = 10.0, p_hi: float = 90.0,
                             blur_sigma: float = 1.0, gamma: float = 0.85) -> np.ndarray:
    """Absdiff(a,b) -> robust normalize -> [0,1] float32.
    Used as a low-context 'HG-like' salience map for camera drift estimation.
    """
    d = cv.absdiff(a_u8.astype(np.uint8), b_u8.astype(np.uint8)).astype(np.float32)
    if blur_sigma and blur_sigma > 0.0:
        d = cv.GaussianBlur(d, (0, 0), float(blur_sigma))
    lo = float(np.percentile(d, float(p_lo)))
    hi = float(np.percentile(d, float(p_hi)))
    rng = max(hi - lo, 1e-9)
    x = np.clip((d - lo) / rng, 0.0, 1.0)
    if gamma and gamma != 1.0:
        x = np.power(x, float(gamma), dtype=np.float32)
    return x.astype(np.float32)

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median (robust). values/weights are 1D-compatible."""
    v = np.asarray(values, np.float64).reshape(-1)
    w = np.asarray(weights, np.float64).reshape(-1)
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
# -------------------------------------------------------------------------
ARROW_SCALE = 2.0
FLOW_PYR_LEVELS_DEFAULT = 2  # 1 = off (old behavior); >1 = external pyramid

# Forward–backward consistency mode:
# 0 = off (fast, current behavior)
# 1 = on for export/trusted passes only
FLOW_FB_MODE_DEFAULT = 0

# Error threshold (in scaled ROI pixels) for forward–backward mismatch.
FB_ERR_THRESH = 0.6

LAG_LPF = 0.18
PROC_SCALE = 1.0

# --- DUP FRAME DETECTION (global) ---
DUP_FRAME_MAD_THRESH = 0.7  # 0..255; tune if needed

CMAT_TARGET_H = 240  # max height (in pixels) for global CMAT drift; <=0 = disabled

MAX_ROI_INPUT_H = 1080   # Best quality, robust tracking

# --- HG_RESCUE: Farneback failure mode escalator (prediction-error + structure phaseCorr+LK) ---
# Enable hypergraph-style rescue only on low-confidence frames (expensive but robust).
HG_RESCUE_ENABLE = True

# Prediction-error / novelty gate (per-ROI):
# If Farneback output disagrees hard with the ROI's own velocity predictor, mark frame lowconf → trigger rescue.
PE_MIN_PRED_PF      = 0.75   # px/frame: only apply novelty gate when we *expect* meaningful motion
PE_NOVELTY_THRESH   = 3.25   # normalized error vs running avg; higher = more conservative
PE_AVG_DECAY        = 0.90   # EMA for prediction error average (higher = slower adaptation)

# --- HG psychoacoustic smoothing (salience-respecting) -----------------------
# Goal: reduce HG micro-jitter without smearing onsets / losing "flash".
# Mechanism:
#   - Asymmetric smoothing: instant attack, slow release
#   - Salience gate: when salience is high, smoothing turns OFF automatically
HG_PSY_SMOOTH_ENABLE = True
HG_PSY_SMOOTH_ALPHA  = 0.10   # base release smoothing (0.05..0.20)
HG_PSY_SMOOTH_JNORM   = 0.35  # jerk normalization vs (min(roi)*fps); lower = more sensitive
# -----------------------------------------------------------------------------


# Hypergraph-ish rescue params (ROI-scale pixels):
HG_PC_DOWNSCALE     = 0.5    # 0.5 = phaseCorr on half-res structure map (faster, more stable)
HG_PC_MIN_RESP      = 0.15   # phaseCorr response gate; < => treat as unreliable
HG_STRUCT_PCTL_DEF  = 96.0   # percentile for structure binarization (higher = sparser, more stable)
HG_PCTL_MIN         = 92.0
HG_PCTL_MAX         = 99.0
HG_PCTL_STEP        = 0.5
HG_DENSITY_LO       = 0.006  # target density band for structure pixels
HG_DENSITY_HI       = 0.035
HG_MORPH_OPEN       = 1      # 0=off; 1..2 helps remove salt noise
HG_MIN_CLUSTER_PX   = 48
HG_MAX_CLUSTERS     = 8
HG_MAX_PTS_PER_CL   = 120
HG_LK_WIN_RADIUS    = 12
HG_LK_MAX_LEVEL     = 3
HG_INLIER_R         = 3.0    # inlier radius for LK residuals (scaled px)
HG_MIN_INLIERS      = 10
HG_MIN_SCORE        = 0.18   # cluster score gate
HG_ACCEPT_CONF      = 0.30   # accept v_blend if >= this confidence

# --- HG burst scheduling (anti-habituation) ---------------------------------
HG_BURST_ENABLE       = True
HG_BURST_HOLD_FRAMES  = 3      # HG “flash” duration (2..5)
HG_BURST_COOLDOWN_FR  = 6      # min gap between flashes (4..12)
HG_BURST_ATTACK       = 0.85   # per-frame ramp-up (0.7..0.95)
HG_BURST_RELEASE      = 0.18   # per-frame decay (0.08..0.25)
HG_BURST_NOV_FRAC     = 0.60   # trigger level as fraction of (PE_NOVELTY_THRESH-1)
HG_BURST_JERK_NORM    = 0.35   # same idea as your smoothing jerk norm
HG_BURST_JERK_TRIG    = 0.55   # 0..1 normalized jerk proxy trigger
HG_BURST_JITTER_FR    = 1      # randomize hold/cooldown by ±N frames (0..2)
# ----------------------------------------------------------------------------

# --- HYBRID mode (HG timing + decimated FB carrier) ---------------------------
# HYB strategy:
#   - Always run HG (cheap) for dx/dy timing truth.
#   - Only run FB (expensive) when HG says motion is calm + non-salient.
#   - Even when FB runs, we keep vx/vy from HG, but use FB field for vz/div/curv-depth.
HYB_ENABLE            = True
HYB_FB_EVERY_N        = 3      # run FB once every N calm frames (2..6)
HYB_FB_LEVELS_MAX     = 5      # cap FB levels during HYB (keeps it cheap)
HYB_EVENT_STEP_FRAC   = 0.28   # if HG step > frac*ROI_diag => "event" => skip FB
HYB_CALM_STEP_FRAC    = 0.12   # if HG step < frac*ROI_diag => eligible for FB
HYB_COOLDOWN_FR       = 6      # after event, wait this many frames before FB can run
HYB_NOV_FRAC          = 0.45   # novelty trigger fraction of (PE_NOVELTY_THRESH-1)
# ---------------------------------------------------------------------------


# --- END HG_RESCUE ---


def _hg_burst_update(roi, fps: float, novelty: float, v_target_x: float, v_target_y: float) -> float:
    """
    Returns w_hg in [0,1] for this frame.
    Uses novelty + jerk proxy to trigger short HG bursts.
    """
    st = roi.__dict__.setdefault("_hg_burst", {
        "w": 0.0,
        "hold": 0,
        "cool": 0,
        "tx": float(v_target_x),
        "ty": float(v_target_y),
    })

    # cooldown tick
    if st["cool"] > 0:
        st["cool"] -= 1

    # jerk proxy from target vector change
    dv = float(math.hypot(float(v_target_x) - float(st.get("tx", 0.0)),
                          float(v_target_y) - float(st.get("ty", 0.0))))
    rx, ry, rw, rh = roi.rect
    j_denom = max(1.0, float(HG_BURST_JERK_NORM) * float(min(rw, rh)) * float(fps))
    j_n = float(np.clip(dv / j_denom, 0.0, 1.0))

    # novelty gate
    denom = max(1e-6, float(PE_NOVELTY_THRESH) - 1.0)
    nov_n = float(np.clip((float(novelty) - 1.0) / denom, 0.0, 1.0))

    # trigger condition
    trig = (nov_n >= float(HG_BURST_NOV_FRAC)) or (j_n >= float(HG_BURST_JERK_TRIG))

    # start burst if allowed
    if trig and st["hold"] <= 0 and st["cool"] <= 0:
        j = int(HG_BURST_JITTER_FR)
        hold = int(HG_BURST_HOLD_FRAMES)
        cool = int(HG_BURST_COOLDOWN_FR)
        if j > 0:
            hold = max(1, hold + random.randint(-j, j))
            cool = max(1, cool + random.randint(-j, j))
        st["hold"] = hold
        st["cool"] = cool

    # update weight
    w = float(st["w"])
    if st["hold"] > 0:
        st["hold"] -= 1
        # attack: ramp up quickly
        w = w + float(HG_BURST_ATTACK) * (1.0 - w)
    else:
        # release: decay slowly
        w = w * (1.0 - float(HG_BURST_RELEASE))

    st["w"] = float(np.clip(w, 0.0, 1.0))
    st["tx"], st["ty"] = float(v_target_x), float(v_target_y)
    return st["w"]


def _consensus_median(fx, fy, msk, cone_deg=25.0):
    """
    Dominant-motion consensus:
    - find dominant direction (angle histogram)
    - keep inliers within ±cone_deg
    - return median dx/dy of inliers
    """
    if msk is None:
        fx1 = fx.reshape(-1)
        fy1 = fy.reshape(-1)
    else:
        fx1 = fx[msk].reshape(-1)
        fy1 = fy[msk].reshape(-1)

    if fx1.size < 64:
        return float(np.median(fx1)) if fx1.size else 0.0, float(np.median(fy1)) if fy1.size else 0.0

    # drop near-zero vectors (angle undefined)
    m = np.sqrt(fx1*fx1 + fy1*fy1)
    nz = m > (0.02 * (float(np.percentile(m, 95.0)) + 1e-9))
    fx1 = fx1[nz]; fy1 = fy1[nz]
    if fx1.size < 64:
        return float(np.median(fx1)) if fx1.size else 0.0, float(np.median(fy1)) if fy1.size else 0.0

    ang = np.arctan2(fy1, fx1)  # [-pi, pi]

    # direction histogram (cheap mode-finder)
    B = 36  # 10-degree bins
    bins = ((ang + np.pi) * (B / (2*np.pi))).astype(np.int32)
    bins = np.clip(bins, 0, B-1)
    hist = np.bincount(bins, minlength=B)
    k = int(np.argmax(hist))
    a0 = ( (k + 0.5) * (2*np.pi/B) ) - np.pi  # bin center angle

    # inliers within cone
    cone = float(np.radians(cone_deg))
    d = np.arctan2(np.sin(ang - a0), np.cos(ang - a0))  # wrapped diff
    inl = np.abs(d) <= cone
    if np.count_nonzero(inl) < 32:
        # can't form consensus → plain median
        return float(np.median(fx1)), float(np.median(fy1))

    return float(np.median(fx1[inl])), float(np.median(fy1[inl]))

def _classify_flip_sample(io_sign: int,
                          v_al: np.ndarray,
                          p: int,
                          pre: int,
                          post: int,
                          v_zero: float) -> str | None:
    """
    Symmetric IN/OUT classifier for a single impact index p.

    io_sign:   +1 => IN = along axis, -1 => IN = opposite axis
    v_al:      signed velocity along AoI (length n)
    p:         impact index
    pre/post:  window sizes (in samples)
    v_zero:    deadband in AoI units

    Returns "in", "out", or None if direction is too ambiguous.
    """
    import numpy as _np

    n = len(v_al)
    if n == 0:
        return None

    # Pre / post windows
    a0 = max(0, p - pre)
    a1 = min(n, p + 1)
    b0 = p
    b1 = min(n, p + post + 1)

    pre_med  = float(_np.median(v_al[a0:a1])) if a0 < a1 else 0.0
    post_med = float(_np.median(v_al[b0:b1])) if b0 < b1 else 0.0

    # Deadband
    if abs(pre_med)  < v_zero: pre_med  = 0.0
    if abs(post_med) < v_zero: post_med = 0.0

    # Work in IN-frame (io_sign flips axis if IN is opposite)
    pre_s  = _np.sign(io_sign * pre_med)
    post_s = _np.sign(io_sign * post_med)

    # Symmetric cases:
    #   pre>0, post<=0  → IN  (toward → away)
    #   pre<0, post>=0  → OUT (away → toward)
    if pre_s > 0 and post_s <= 0:
        return "in"
    if pre_s < 0 and post_s >= 0:
        return "out"

    # Fallback: dominant sign in a symmetric window around p
    w0 = max(0, p - pre)
    w1 = min(n, p + post + 1)
    if w1 <= w0:
        return None

    win_med = float(_np.median(io_sign * v_al[w0:w1]))
    if abs(win_med) < v_zero:
        return None

    return "in" if win_med > 0 else "out"


def _center_vel_from_pos(cx, cy, fps):
    """
    Derive center-based velocities (px/s) from ROI positions.
    This is independent of the flow field and uses only roi_cx/roi_cy.

    cx, cy: sequences of ROI center positions in pixels
    fps:    frames per second

    Returns:
        vx_pos, vy_pos in px/s (same length as input)
    """
    import numpy as _np
    cx = _np.asarray(cx, _np.float64)
    cy = _np.asarray(cy, _np.float64)
    n = min(cx.size, cy.size)
    if n == 0:
        return _np.zeros(0, _np.float64), _np.zeros(0, _np.float64)

    cx = cx[:n]
    cy = cy[:n]
    dt = 1.0 / max(1e-6, float(fps))

    vx_pos = _deriv_central(cx, dt)  # px/s
    vy_pos = _deriv_central(cy, dt)  # px/s
    return vx_pos, vy_pos


def is_near_duplicate_frame(prev_gray: np.ndarray, curr_gray: np.ndarray,
                            thresh: float = DUP_FRAME_MAD_THRESH) -> bool:
    """
    Fast global test for near-identical frames (same size, 8-bit gray).
    """
    if prev_gray is None or curr_gray is None: 
        return False
    if prev_gray.shape != curr_gray.shape:
        return False
    diff = cv.absdiff(prev_gray, curr_gray)
    mean_diff = float(np.mean(diff))
    return mean_diff < thresh


# --- 2nd-order curvature inside an ROI (intensity Hessian-ish) ---
def compute_roi_curvature(prev_gray: np.ndarray, curr_gray: np.ndarray,
                          rect: Tuple[int,int,int,int]) -> float:
    """
    Very simple spatio-temporal curvature energy inside ROI.
    Uses Ixx, Iyy and a 2nd temporal derivative Itt.
    """
    if prev_gray is None or curr_gray is None:
        return 0.0
    x, y, w, h = rect
    if w <= 1 or h <= 1:
        return 0.0

    roi_prev = prev_gray[y:y+h, x:x+w]
    roi_curr = curr_gray[y:y+h, x:x+w]
    if roi_prev.size == 0 or roi_curr.size == 0:
        return 0.0

    roi_prev = cv.GaussianBlur(roi_prev, (3, 3), 0)
    roi_curr = cv.GaussianBlur(roi_curr, (3, 3), 0)

    Ix  = cv.Sobel(roi_curr, cv.CV_32F, 1, 0, ksize=3)
    Iy  = cv.Sobel(roi_curr, cv.CV_32F, 0, 1, ksize=3)
    Ixx = cv.Sobel(Ix,       cv.CV_32F, 1, 0, ksize=3)
    Iyy = cv.Sobel(Iy,       cv.CV_32F, 0, 1, ksize=3)

    It  = (roi_curr.astype(np.float32) - roi_prev.astype(np.float32))
    Itt = cv.Laplacian(It, cv.CV_32F, ksize=3)

    energy = np.mean(Ixx*Ixx + Iyy*Iyy + 0.5*Itt*Itt)
    return float(np.sqrt(max(energy, 0.0)))

def compute_roi_iglog_energy(curr_gray: np.ndarray,
                             rect: Tuple[int,int,int,int]) -> float:
    """
    Scalar IG-LoG structural energy inside ROI (robust median of _iglog_energy map).
    Uses the existing _iglog_energy(img_f32) pipeline. :contentReference[oaicite:5]{index=5}
    """
    if curr_gray is None:
        return 0.0
    x, y, w, h = map(int, rect)
    if w <= 2 or h <= 2:
        return 0.0
    H, W = curr_gray.shape[:2]
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    x1 = max(x+1, min(x+w, W)); y1 = max(y+1, min(y+h, H))
    roi = curr_gray[y:y1, x:x1]
    if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
        return 0.0
    pf = roi.astype(np.float32) / 255.0
    E = _iglog_energy(pf)
    if E is None or E.size == 0:
        return 0.0
    return float(np.median(E))

def _count_envelope_peaks01(env01, fps, min_prom=0.10, min_sep_ms=50):
    """
    Rough cycle count on a 0..1 envelope:
      - local maxima above a robustness threshold
      - at least min_sep_ms apart
    Used only to decide if Gaussianization has obviously collapsed multiple
    cycles into one blob.
    """
    env = np.asarray(env01, np.float64)
    n = env.size
    if n < 3:
        return 0

    # Ignore nearly-flat envelopes
    if np.ptp(env) < 1e-3:
        return 0

    # Threshold: high percentile to dodge micro-wiggles
    hi = float(np.percentile(env, 75.0))
    if hi <= 1e-6:
        return 0

    # Require both a floor and relative prominence
    thr = max(0.35, hi * (1.0 - min_prom))   # ~top 25–40% of samples

    min_sep = max(1, int(round(min_sep_ms * fps / 1000.0)))
    last = -10**9
    count = 0

    for i in range(1, n - 1):
        v = env[i]
        if v < thr:
            continue
        if not (v >= env[i - 1] and v >= env[i + 1]):
            continue
        if i - last < min_sep:
            continue
        count += 1
        last = i

    return count


# --scale=N handling
def _parse_cli_scale(argv):
    import numpy as _np
    global PROC_SCALE
    for i,a in enumerate(list(argv)):
        if a == "--scale" and i+1 < len(argv):
            PROC_SCALE = float(argv[i+1]); break
        if a.startswith("--scale="):
            PROC_SCALE = float(a.split("=",1)[1]); break
    PROC_SCALE = float(_np.clip(PROC_SCALE, 0.25, 1.0))

REF_SPEED_PPS = 140.0      # "1.0" reference speed in px/s (tune)
CYCLE_MIN_MS  = 120        # ignore flips faster than this
V_ZERO        = 0.02       # deadband on principal velocity (px/s)
# --- EXPORT label style (overlay only) ---
IMPACT_FONT        = cv.FONT_HERSHEY_SIMPLEX
IMPACT_SCALE       = 0.80          # bigger label
IMPACT_THICK_BOLD  = 2             # bold
IMPACT_BLUE_BOLD   = (255, 80, 30) # BGR → strong blue
IMPACT_ORANGE      = (0, 165, 255) # BGR → orange (keep OUT)

# ================= IG-LoG (Tier-1 structural) =================
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_u8_norm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-9:
        return np.zeros_like(a, dtype=np.uint8)
    x = (a - mn) / (mx - mn)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

def iglog_map(gray_u8: np.ndarray, sigma: float) -> np.ndarray:
    """
    "IG-LoG" here = (I - Gσ) applied to LoG(Gσ * I).
    Practical: LoG on a Gaussian-smoothed image, then a high-pass-ish normalization.
    Returns float32.
    """
    g = gray_u8.astype(np.float32) / 255.0
    # Gaussian blur
    k = int(max(3, int(math.ceil(sigma * 6)) | 1))  # odd kernel ~ 6σ
    gs = cv.GaussianBlur(g, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv.BORDER_REFLECT)

    # Laplacian of Gaussian-ish: Laplacian after Gaussian smoothing
    # Use CV_32F Laplacian for stability
    log = cv.Laplacian(gs, ddepth=cv.CV_32F, ksize=3, borderType=cv.BORDER_REFLECT)

    # "IG" component: isolate fast structure by subtracting a slightly larger blur of the LoG
    # (acts like removing slow drift in the LoG response)
    sigma2 = max(0.8, sigma * 1.75)
    k2 = int(max(3, int(math.ceil(sigma2 * 6)) | 1))  # match sigma2 (bugfix)
    slow = cv.GaussianBlur(log, (k2, k2), sigmaX=sigma2, sigmaY=sigma2, borderType=cv.BORDER_REFLECT)
    iglog = log - slow
    return iglog

def zero_crossings(log_f32: np.ndarray, thresh: float = 0.0) -> np.ndarray:
    """
    Zero-crossing map of LoG/IGLoG. Returns uint8 image (0/255).
    thresh: ignore very small values (noise gate).
    """
    a = log_f32
    s = np.sign(a).astype(np.int8)

    if thresh > 0:
        s[np.abs(a) < thresh] = 0

    H, W = s.shape[:2]
    z = np.zeros((H, W), dtype=np.uint8)

    # Horizontal sign changes land at the right pixel (col 1..W-1)
    hchg = (s[:, 1:] * s[:, :-1]) < 0           # shape (H, W-1)
    z[:, 1:] |= (hchg.astype(np.uint8) * 255)

    # Vertical sign changes land at the lower pixel (row 1..H-1)
    vchg = (s[1:, :] * s[:-1, :]) < 0           # shape (H-1, W)
    z[1:, :] |= (vchg.astype(np.uint8) * 255)

    return z


def extrema_map(log_f32: np.ndarray, abs_thresh: float) -> np.ndarray:
    """
    Very cheap "landmarks": pixels where |response| is locally strong.
    Returns uint8 (0/255).
    """
    a = np.abs(log_f32)
    if abs_thresh <= 0:
        abs_thresh = float(np.percentile(a, 95))  # fallback

    # local maximum-ish via dilation
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dil = cv.dilate(a, k)
    peaks = (a >= dil - 1e-12) & (a >= abs_thresh)
    return (peaks.astype(np.uint8) * 255)

def outline_from_zc(zc_u8: np.ndarray) -> np.ndarray:
    """
    Make a coarse outline polygon (largest contour) from zero-crossings.
    Returns Nx2 int32 points, or empty array.
    """
    # thicken + close gaps
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    bw = cv.dilate(zc_u8, k, iterations=1)
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((0, 2), dtype=np.int32)

    c = max(cnts, key=cv.contourArea)
    if cv.contourArea(c) < 50:
        return np.zeros((0, 2), dtype=np.int32)

    peri = cv.arcLength(c, True)
    eps = max(2.0, 0.01 * peri)
    approx = cv.approxPolyDP(c, eps, True)
    return approx.reshape(-1, 2).astype(np.int32)

def iglog_occlusion_score(ext_maps: list[np.ndarray]) -> float:
    """
    Simple persistence score across scales.
    ext_maps: list of uint8 (0/255) extrema maps at increasing sigma.
    Score ~ fraction of extrema that persist across scales (coarse structure survives occlusion).
    """
    if not ext_maps:
        return 0.0
    # Convert to boolean masks
    ms = [(m > 0) for m in ext_maps]
    counts = [int(m.sum()) for m in ms]
    if min(counts) == 0:
        return 0.0
    # intersection over smallest count
    inter = ms[0].copy()
    for m in ms[1:]:
        inter &= m
    return float(inter.sum()) / float(min(counts))

def iglog_debug_run(crop_bgr: np.ndarray, out_dir: str, base: str,
                    sigmas=(1.2, 2.4, 4.0),
                    zc_thresh=0.004, ext_abs_thresh=0.03):
    """
    Runs IG-LoG + zero-crossings + extrema at multiple scales and writes debug images.
    Returns: (occ_score, poly_pts, counts)
    """
    _ensure_dir(out_dir)
    gray = cv.cvtColor(crop_bgr, cv.COLOR_BGR2GRAY)

    ext_maps = []
    zc_maps  = []

    abs_list = []
    zc_list  = []
    ext_list = []

    cv.imwrite(os.path.join(out_dir, f"{base}_crop.png"), crop_bgr)


    for s in sigmas:
        ig = iglog_map(gray, sigma=float(s))
        ig_u8 = _to_u8_norm(np.abs(ig))
        cv.imwrite(os.path.join(out_dir, f"{base}_iglog_abs_s{float(s):.2f}.png"), ig_u8)

        zc = zero_crossings(ig, thresh=zc_thresh)
        zc_maps.append(zc)
        cv.imwrite(os.path.join(out_dir, f"{base}_iglog_zc_s{float(s):.2f}.png"), zc)

        ext = extrema_map(ig, abs_thresh=ext_abs_thresh)
        ext_maps.append(ext)
        cv.imwrite(os.path.join(out_dir, f"{base}_iglog_ext_s{float(s):.2f}.png"), ext)

        abs_list.append(ig_u8)
        zc_list.append(zc)
        ext_list.append(ext)


    occ = iglog_occlusion_score(ext_maps)

    # Outline from the largest-scale ZC (most stable coarse structure)
    poly = outline_from_zc(zc_maps[-1]) if zc_maps else np.zeros((0,2), np.int32)

    # Overlay polygon on crop
    overlay = crop_bgr.copy()
    if poly.shape[0] >= 3:
        cv.polylines(overlay, [poly.reshape(-1,1,2)], isClosed=True, color=(0,255,0), thickness=2)
    cv.imwrite(os.path.join(out_dir, f"{base}_outline_overlay.png"), overlay)

    counts = [int((m>0).sum()) for m in ext_maps]

    def _as_bgr(u8):
        if u8.ndim == 2:
            return cv.cvtColor(u8, cv.COLOR_GRAY2BGR)
        return u8

    # pick the “main” scale you care about (usually sigmas[0] = 1.2)
    i_main = 0
    abs_u8 = abs_list[i_main]
    zc_u8  = zc_list[i_main]
    ext_u8 = ext_list[i_main]

    # Your current favorite composite: ZC + EXT (and optionally ABS)
    # Start with what you said you like:
    comp = cv.add(zc_u8, ext_u8)

    # OPTIONAL: if you want an alternate composite to compare:
    # comp2 = np.clip(0.55*abs_u8 + 0.25*zc_u8 + 0.20*ext_u8, 0, 255).astype(np.uint8)

    # outline overlay already exists as `overlay` in your code; if not:
    overlay = crop_bgr.copy()
    if poly.shape[0] >= 3:
        cv.polylines(overlay, [poly.reshape(-1,1,2)], True, (0,255,0), 2)

    # Normalize sizes for concatenation
    H = crop_bgr.shape[0]
    def _fit(img_bgr):
        if img_bgr.shape[0] == H:
            return img_bgr
        w = int(round(img_bgr.shape[1] * (H / img_bgr.shape[0])))
        return cv.resize(img_bgr, (w, H), interpolation=cv.INTER_NEAREST)

    # strip = cv.hconcat([
    #     _fit(crop_bgr),
    #     _fit(_as_bgr(zc_u8)),
    #     _fit(_as_bgr(ext_u8)),
    #     _fit(_as_bgr(abs_u8)),
    #     _fit(_as_bgr(comp)),
    #     _fit(overlay),
    # ])
    comp_clean = cv.add(zc_u8, ext_u8)
    comp_rescue = np.clip(0.70*abs_u8 + 0.20*zc_u8 + 0.10*ext_u8, 0, 255).astype(np.uint8)

    strip = cv.hconcat([
        _fit(crop_bgr),
        _fit(_as_bgr(comp_clean)),
        _fit(_as_bgr(comp_rescue)),
        _fit(overlay),
    ])
    cv.imwrite(os.path.join(out_dir, f"{base}_RESULT.png"), strip)

    return occ, poly, counts, abs_list, zc_list, ext_list

def _dens01(u8_bw: np.ndarray) -> float:
    return float(np.mean(u8_bw > 0))

def iglog_struct_maps_u8(crop_bgr: np.ndarray,
                         sigma: float = 1.2,
                         zc_thresh: float = 0.004,
                         ext_abs_thresh: float = 0.03):
    """
    Fast per-frame structural maps (single sigma):
      - abs_u8: normalized |IGLoG|
      - zc_u8 : zero-crossings (0/255)
      - ext_u8: extrema landmarks (0/255)
      - clean : zc + ext (uint8)
      - rescue: 0.70*abs + 0.20*zc + 0.10*ext (uint8)
    Returns dict with maps + density score.
    """
    gray = cv.cvtColor(crop_bgr, cv.COLOR_BGR2GRAY)
    ig = iglog_map(gray, sigma=float(sigma))
    abs_u8 = _to_u8_norm(np.abs(ig))
    zc_u8 = zero_crossings(ig, thresh=float(zc_thresh))
    ext_u8 = extrema_map(ig, abs_thresh=float(ext_abs_thresh))

    # --- NEW: GSCM hybrid (optional) ---
    try:
        mass_sigma = 0.5  # default; tune later or expose as global const
        hyb_u8, mass_u8 = gscm_hybrid_u8_from_maps(abs_u8, zc_u8, ext_u8, gray, sigma_mass=mass_sigma)
    except Exception:
        hyb_u8, mass_u8 = None, None


    clean = cv.add(zc_u8, ext_u8)
    rescue = np.clip(
        0.70*abs_u8.astype(np.float32) + 0.20*zc_u8.astype(np.float32) + 0.10*ext_u8.astype(np.float32),
        0, 255
    ).astype(np.uint8)

    score = _dens01(zc_u8) + _dens01(ext_u8)
    return {
        "abs": abs_u8, "zc": zc_u8, "ext": ext_u8,
        "clean": clean, "rescue": rescue,
        "score": float(score), "hybrid": hyb_u8,
        "mass": mass_u8,
    }

def _u8_from_percentile(x_f32: np.ndarray, p: float = 99.0) -> np.ndarray:
    x = x_f32.astype(np.float32)
    s = float(np.percentile(x, p))
    if s < 1e-9:
        return np.zeros_like(x, dtype=np.uint8)
    y = np.clip(x * (255.0 / s), 0, 255)
    return y.astype(np.uint8)

def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def mass_grad_u8(gray_u8: np.ndarray, sigma_mass: float = 1.0, p: float = 99.0) -> np.ndarray:
    """
    Gaussian-blurred gradient magnitude (MASS). sigma_mass ~ 0.6..1.5 is usually the sweet spot.
    """
    g = gray_u8.astype(np.float32) / 255.0
    gb = cv.GaussianBlur(g, (0, 0), sigmaX=float(sigma_mass), sigmaY=float(sigma_mass), borderType=cv.BORDER_REFLECT)
    gx = cv.Sobel(gb, cv.CV_32F, 1, 0, ksize=3, borderType=cv.BORDER_REFLECT)
    gy = cv.Sobel(gb, cv.CV_32F, 0, 1, ksize=3, borderType=cv.BORDER_REFLECT)
    mag = cv.magnitude(gx, gy)
    return _u8_from_percentile(mag, p=p)

def gscm_hybrid_u8_from_maps(abs_u8: np.ndarray, zc_u8: np.ndarray, ext_u8: np.ndarray,
                             gray_u8: np.ndarray,
                             sigma_mass: float = 1.0,
                             w_lo: float = 0.08, w_hi: float = 0.25,
                             gate_dilate: int = 1,
                             gate_sigma: float = 1.2,
                             gate_floor: float = 0.25,
                             mass_pctl: float = 99.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (hybrid_u8, mass_u8).
      - MASS is the carrier
      - ABS is the authority weight
      - (ZC|EXT) is a *soft* permit field (no additive spikes)
    """
    mass_u8 = mass_grad_u8(gray_u8, sigma_mass=sigma_mass, p=mass_pctl)

    abs_n  = abs_u8.astype(np.float32) / 255.0
    mass_n = mass_u8.astype(np.float32) / 255.0

    w = smoothstep(abs_n, lo=float(w_lo), hi=float(w_hi))
    hyb = w * mass_n + (1.0 - w) * abs_n

    # soft topology permit
    valid_u8 = cv.max(zc_u8, ext_u8)  # 0/255
    if gate_dilate and gate_dilate > 0:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        valid_u8 = cv.dilate(valid_u8, k, iterations=int(gate_dilate))

    valid_f = valid_u8.astype(np.float32) / 255.0
    if gate_sigma and gate_sigma > 0.0:
        valid_f = cv.GaussianBlur(valid_f, (0, 0), float(gate_sigma), borderType=cv.BORDER_REFLECT)

    valid_f = float(gate_floor) + (1.0 - float(gate_floor)) * np.clip(valid_f, 0.0, 1.0)
    hyb *= valid_f

    # brightness compensation (frame-local, single-frame safe)
    mg = float(np.mean(valid_f))
    if mg > 1e-6:
        hyb /= mg

    # perceptual lift
    hyb = np.power(np.clip(hyb, 0.0, 1.0), 0.6)

    hyb_u8 = np.clip(hyb * 255.0, 0, 255).astype(np.uint8)
    return hyb_u8, mass_u8


def iglog_struct_maps_gray_u8(gray_u8: np.ndarray,
                              sigma: float = 1.2,
                              zc_thresh: float = 0.004,
                              ext_abs_thresh: float = 0.03):
    ig = iglog_map(gray_u8, sigma=float(sigma))
    abs_u8 = _to_u8_norm(np.abs(ig))
    zc_u8  = zero_crossings(ig, thresh=float(zc_thresh))
    ext_u8 = extrema_map(ig, abs_thresh=float(ext_abs_thresh))
    # --- NEW: GSCM hybrid (optional) ---
    try:
        mass_sigma = 1.0  # default; tune later or expose as global const
        hyb_u8, mass_u8 = gscm_hybrid_u8_from_maps(abs_u8, zc_u8, ext_u8, gray_u8, sigma_mass=mass_sigma)
    except Exception:
        hyb_u8, mass_u8 = None, None


    clean = cv.add(zc_u8, ext_u8)
    rescue = np.clip(
        0.70 * abs_u8.astype(np.float32) +
        0.20 * zc_u8.astype(np.float32) +
        0.10 * ext_u8.astype(np.float32),
        0, 255
    ).astype(np.uint8)

    score = _dens01(zc_u8) + _dens01(ext_u8)
    return {"abs": abs_u8, "zc": zc_u8, "ext": ext_u8, "clean": clean, "rescue": rescue, "score": float(score), "hybrid": hyb_u8, "mass": mass_u8,}


def soft_edge_field_u8(edge_u8: np.ndarray, blur_sigma: float = 0.0) -> np.ndarray:
    """
    Turns a binary-ish edge map into a smooth field via distance transform:
      - edges get high value
      - decays smoothly away from edges
    This massively improves OF/phase-corr conditioning.
    """
    edges = (edge_u8 > 0).astype(np.uint8)
    if edges.size == 0 or np.count_nonzero(edges) == 0:
        return edge_u8

    inv = (1 - edges).astype(np.uint8)  # edges are zeros; distanceTransform measures dist to zeros
    dist = cv.distanceTransform(inv, cv.DIST_L2, 3)

    d95 = float(np.percentile(dist, 95.0))
    if d95 < 1e-6:
        out = (edges * 255).astype(np.uint8)
    else:
        out01 = np.clip(1.0 - (dist / d95), 0.0, 1.0)
        out = (out01 * 255.0).astype(np.uint8)

    if blur_sigma and blur_sigma > 0.0:
        out = cv.GaussianBlur(out, (0, 0), float(blur_sigma))
    return out


def _struct_flow_input_u8(M_u8: np.ndarray, mode: str = "soft", blur_sigma: float = 0.8) -> np.ndarray:
    """
    Prepares a structural map for correlation/flow.
      mode="raw"  -> M directly
      mode="blur" -> GaussianBlur(M)
      mode="soft" -> distance-transform soft field + blur
    """
    if mode == "raw":
        return M_u8
    if mode == "blur":
        return cv.GaussianBlur(M_u8, (0, 0), float(blur_sigma))
    # soft
    soft = soft_edge_field_u8(M_u8, blur_sigma=float(blur_sigma))
    return soft


def make_preview_strip_bgr(crop_bgr: np.ndarray, clean_u8: np.ndarray, rescue_u8: np.ndarray, poly: np.ndarray):
    """
    Crop | Clean | Rescue | Overlay (all same height, BGR).
    """
    def _as_bgr(u8):
        return cv.cvtColor(u8, cv.COLOR_GRAY2BGR) if u8.ndim == 2 else u8

    overlay = crop_bgr.copy()
    if poly is not None and hasattr(poly, "shape") and poly.shape[0] >= 3:
        cv.polylines(overlay, [poly.reshape(-1,1,2)], True, (0,255,0), 2)

    H = crop_bgr.shape[0]
    def _fit(img):
        if img.shape[0] == H:
            return img
        w = int(round(img.shape[1] * (H / img.shape[0])))
        return cv.resize(img, (w, H), interpolation=cv.INTER_NEAREST)

    strip = cv.hconcat([
        _fit(crop_bgr),
        _fit(_as_bgr(clean_u8)),
        _fit(_as_bgr(rescue_u8)),
        _fit(overlay),
    ])
    return strip


def _density_u8(bw_u8: np.ndarray) -> float:
    return float(np.mean(bw_u8 > 0))

def adaptive_struct_map(abs_u8, zc_u8, ext_u8,
                        state: dict,
                        T_low=0.0015, T_high=0.0040,
                        decay=0.94,
                        abs_cap=200,
                        force_abs_on: bool=False,
                        warp_dx: float = 0.0,
                        warp_dy: float = 0.0):
    """
    Returns:
      M_u8, abs_on(bool), score(float)

    Notes:
      - abs_acc is stabilized via decay+max.
      - Optional warp_dx/warp_dy (pixels) motion-compensates the accumulator BEFORE update.
    """
    score = _dens01(zc_u8) + _dens01(ext_u8)

    abs_on = bool(state.get("abs_on", False))
    if force_abs_on:
        abs_on = True
    else:
        # hysteresis
        if (not abs_on) and score < T_low:
            abs_on = True
        elif abs_on and score > T_high:
            abs_on = False

    acc = state.get("abs_acc", None)
    if acc is None or acc.shape != abs_u8.shape:
        acc = abs_u8.astype(np.uint8)
    else:
        # motion-compensate accumulator (prevents trails when structure moves)
        if (abs(warp_dx) + abs(warp_dy)) >= 0.5:
            M = np.float32([[1.0, 0.0, float(warp_dx)],
                            [0.0, 1.0, float(warp_dy)]])
            acc = cv.warpAffine(acc, M, (acc.shape[1], acc.shape[0]),
                                flags=cv.INTER_LINEAR,
                                borderMode=cv.BORDER_CONSTANT,
                                borderValue=0)

        # decay + max
        acc = np.maximum((acc.astype(np.float32) * float(decay)).astype(np.uint8), abs_u8)

    if abs_cap is not None and abs_cap > 0:
        acc = np.minimum(acc, np.uint8(abs_cap))

    state["abs_acc"] = acc
    state["abs_on"] = abs_on  # <-- FIX: persist the actual decision

    if abs_on:
        M_f = (0.70 * acc.astype(np.float32) +
               0.20 * zc_u8.astype(np.float32) +
               0.10 * ext_u8.astype(np.float32))
    else:
        M_f = (0.70 * zc_u8.astype(np.float32) +
               0.30 * ext_u8.astype(np.float32))

    M_u8 = np.clip(M_f, 0, 255).astype(np.uint8)
    return M_u8, abs_on, float(score)


# ================= END IG-LoG =================

def struct_M_from_gray_u8(gray_u8: np.ndarray,
                          state: dict,
                          sigma: float = 1.2,
                          zc_thresh: float = 0.004,
                          ext_abs_thresh: float = 0.03,
                          decay: float = 0.94,
                          abs_cap: int = 200) -> np.ndarray:
    """
    Build the same kind of composite map M used in struct preview,
    but directly from a grayscale patch (u8). Intended for ROI-scale rescue.
    ABS is forced ON by construction here.
    """
    ig = iglog_map(gray_u8, sigma=float(sigma))
    abs_u8 = _to_u8_norm(np.abs(ig))
    zc_u8  = zero_crossings(ig, thresh=float(zc_thresh))
    ext_u8 = extrema_map(ig, abs_thresh=float(ext_abs_thresh))

    # Prefer GSCM hybrid as the structural flow input (better in shading-defined anatomy)
    try:
        M_u8, _mass = gscm_hybrid_u8_from_maps(abs_u8, zc_u8, ext_u8, gray_u8, sigma_mass=1.0)
        return M_u8
    except Exception:
        state["abs_on"] = True
        M_u8, _, _ = adaptive_struct_map(abs_u8, zc_u8, ext_u8, state, decay=float(decay), abs_cap=int(abs_cap))
        return M_u8


def per_cycle_dir_gauss_blend11(
        vx_ps, vy_ps, vz_ps, vP_ps, fps,
        v_zero=V_ZERO, min_ms=CYCLE_MIN_MS,
        p_lo=5, p_hi=95,
        alpha=0.67):
    """
    Per-cycle, signed direction in -1..+1 using a blended Gaussian + raw magnitude.

    For each cycle [a:b):
      1) Build mag01_raw from |v| with robust [p_lo, p_hi] scaling.
      2) Fit two-piece Gaussian to mag01_raw.
      3) Construct a full-span Gaussian lobe g(t) over [a:b), normalized to max=1.
      4) Blend: g_blend = (1 - alpha)*mag01_raw + alpha*g.
      5) For each component (vx,vy,vz), dir = sign(component) * g_blend,
         then normalized so max|dir| <= 1 in that cycle.
    """
    vx = np.asarray(vx_ps, np.float64)
    vy = np.asarray(vy_ps, np.float64)
    vz = np.asarray(vz_ps, np.float64)
    vP = np.asarray(vP_ps,  np.float64)

    T  = len(vx)
    dx = np.zeros(T, float)
    dy = np.zeros(T, float)
    dz = np.zeros(T, float)


    edges = _cycle_edges_from_v(vP, fps, v_zero, min_ms)

    for a, b in zip(edges[:-1], edges[1:]):
        L = b - a
        if L < 2:
            continue

        seg_vx = vx[a:b]
        seg_vy = vy[a:b]
        seg_vz = vz[a:b]

        segmag = np.sqrt(seg_vx**2 + seg_vy**2 + seg_vz**2)
        if not np.any(segmag > 1e-12):
            continue

        # instantaneous signed derivative as skew signal
        dseg = np.gradient(segmag)
        skew = np.clip(dseg / (np.max(np.abs(dseg))+1e-9), -1.0, +1.0)
        skew_med = np.median(skew)

        # 1) raw magnitude -> 0..1 with robust scaling
        lo = float(np.percentile(segmag, p_lo))
        hi = float(np.percentile(segmag, p_hi))
        rng = max(hi - lo, 1e-9)
        mag01_raw = np.clip((segmag - lo) / rng, 0.0, 1.0)

        # 2) fit two-piece Gaussian on mag01_raw
        pars = _cycle_gauss_params(mag01_raw)

        # 3) full-span Gaussian lobe g(t), normalized
        if pars is not None:
            p, A_fit, sL_fit, sR_fit = pars

            # Clamp sigmas so the effective support spans most of the cycle.
            # (Prevents ultra-narrow spikes.)
            min_sigma = 0.15 * L
            max_sigma = 0.60 * L
            sL = float(np.clip(sL_fit, min_sigma, max_sigma))
            sR = float(np.clip(sR_fit, min_sigma, max_sigma))

            # skew_med > 0 → forward thrust → compress left side
            # skew_med < 0 → backward thrust → compress right side

            k = 0.55  # skew intensity, tune 0.2–1.2
            if skew_med > 0:
                sL *= (1.0 - k*abs(skew_med))   # tighter entrance
                sR *= (1.0 + 0.4*k*abs(skew_med))
            else:
                sR *= (1.0 - k*abs(skew_med))   # tighter exit
                sL *= (1.0 + 0.4*k*abs(skew_med))

            # ensure the clamp still holds
            sL = float(np.clip(sL, min_sigma, max_sigma))
            sR = float(np.clip(sR, min_sigma, max_sigma))


            g = _gauss_piece(L, p, 1.0, sL, sR)
            g = np.clip(g, 0.0, 1.0)

            # Normalize lobe to peak 1
            g_max = float(np.max(g) or 1.0)
            g /= g_max
        else:
            # fallback: just use the raw magnitude shape
            g = mag01_raw.copy()

        # 4) blend Gaussian lobe with raw magnitude
        alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
        g_blend = (1.0 - alpha_clamped) * mag01_raw + alpha_clamped * g

        # mild safety: keep in [0,1]
        g_blend = np.clip(g_blend, 0.0, 1.0)

        # 5) signed direction per component, normalized per cycle
        for seg_comp, dst in ((seg_vx, dx), (seg_vy, dy), (seg_vz, dz)):
            sign_c = np.sign(seg_comp)
            z = sign_c * g_blend  # -1..+1, lobe-shaped

            m = float(np.max(np.abs(z)) or 1.0)
            z /= m   # ensure max|dir| <= 1 in this cycle

            dst[a:b] = z

    return dx, dy, dz


def per_cycle_dir_gauss11(vx_ps, vy_ps, vz_ps, vP_ps, fps,
                          v_zero=V_ZERO, min_ms=CYCLE_MIN_MS,
                          p_lo=5, p_hi=95):
    """
    Per-cycle, Gaussian-shaped direction in -1..+1.

    For each cycle:
      1) Build a 0..1 magnitude segment from |v| with robust [p_lo,p_hi] scaling.
      2) Fit a two-piece Gaussian to that segment (same machinery as per_cycle_env01).
      3) Use that Gaussian as a "stroke lobe" g(t) in 0..1.
      4) For each component, take sign(comp) and weight by g(t), normalized so
         max|dir| within the cycle is <= 1.
    """
    vx = np.asarray(vx_ps, np.float64)
    vy = np.asarray(vy_ps, np.float64)
    vz = np.asarray(vz_ps, np.float64)
    vP = np.asarray(vP_ps,  np.float64)

    T  = len(vx)
    dx = np.zeros(T, float)
    dy = np.zeros(T, float)
    dz = np.zeros(T, float)

    edges = _cycle_edges_from_v(vP, fps, v_zero, min_ms)
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2:
            continue

        seg_vx = vx[a:b]
        seg_vy = vy[a:b]
        seg_vz = vz[a:b]

        segmag = np.sqrt(seg_vx**2 + seg_vy**2 + seg_vz**2)
        if not np.any(segmag > 1e-12):
            continue

        # 1) magnitude → 0..1
        lo = float(np.percentile(segmag, p_lo))
        hi = float(np.percentile(segmag, p_hi))
        rng = max(hi - lo, 1e-9)
        mag01 = np.clip((segmag - lo) / rng, 0.0, 1.0)

        # 2) fit two-piece Gaussian; fallback = mag01 itself
        pars = _cycle_gauss_params(mag01)
        if pars is not None:
            p, A, sL, sR = pars
            g = _gauss_piece(len(mag01), p, 1.0, sL, sR)
            g = np.clip(g, 0.0, 1.0)
        else:
            g = mag01

        # 3) signed direction = sign * Gaussian lobe
        #    scale so that max|dir| in this cycle is <= 1
        for seg_comp, dst in ((seg_vx, dx), (seg_vy, dy), (seg_vz, dz)):
            sign_c = np.sign(seg_comp)
            z = sign_c * g  # -1..+1 but lobe-shaped
            m = float(np.max(np.abs(z)) or 1.0)
            z /= m  # just in case numerical issues
            dst[a:b] = z

    return dx, dy, dz

def _impacts_flux_dog_fused_with_jerk(r, vx_s, vy_s, vz_s, fps):
    import numpy as _np

    # 1) Compute axis velocity once
    ax, ay, az = _axis_unit3d(
        getattr(r, "io_dir_deg", 0.0),
        getattr(r, "axis_elev_deg", 0.0)
    )
    kz = float(getattr(r, "axis_z_scale", 1.0))
    vx_s = _np.asarray(vx_s, float)
    vy_s = _np.asarray(vy_s, float)
    vz_s = _np.asarray(vz_s, float)
    v_al = vx_s*ax + vy_s*ay + kz*vz_s

    # 2) Base flux-DoG (unchanged)
    S_flux, _, _ = _impacts_flux_dog(r, vx_s, vy_s, vz_s, fps)

    # 3) Axis jerk magnitude (copy of axis_jerk core, but no gates/peaks)
    dt   = 1.0 / max(1e-6, float(fps))
    a_al = _deriv_central(v_al, dt)
    j_al = _deriv_central(a_al, dt)
    S_jerk = _np.maximum(0.0, _mad_z(_np.abs(j_al)))  # robust z-score

    # 4) Detect “sudden change” windows
    #    Candidate if jerk large AND direction actually changes around here.
    jerk_thr = float(getattr(r, "impact_jerk_z", 1.3))
    cand = _np.where(S_jerk >= jerk_thr)[0]

    flip_pre  = max(1, int(round(0.010 * fps)))  # ~10 ms before
    flip_post = max(1, int(round(0.020 * fps)))  # ~20 ms after
    # vmin gate is about "is there meaningful motion at all?"
    # v_zero is about "is direction (sign) meaningful?"
    # These are different units-of-meaning and must NOT be tied together.
    vmin_gate = float(getattr(r, "impact_min_speed_ps", 0.0))
    v_zero    = float(getattr(r, "impact_flip_deadband_ps", V_ZERO))
    # Optional: drop jerk candidates when global motion is below vmin_gate
    if vmin_gate > 0.0:
        vmag = _np.sqrt(vx_s*vx_s + vy_s*vy_s + (kz*vz_s)*(kz*vz_s))
        S_jerk = S_jerk * _np.clip(vmag / vmin_gate, 0.0, 1.0)


    io_sign   = int(getattr(r, "io_in_sign", +1))

    cand_flips = []
    for i in cand:
        a0 = max(0, i - flip_pre)
        a1 = min(len(v_al), i + flip_post)
        if a1 - a0 < 2:
            continue

        seg = io_sign * v_al[a0:a1]
        # require meaningful magnitude on both sides and sign change
        left  = _np.median(seg[:max(1, flip_pre)])
        right = _np.median(seg[-max(1, flip_post):])
        if abs(left) < v_zero or abs(right) < v_zero:
            continue
        if left * right >= 0:
            continue  # no flip

        cand_flips.append(i)

    cand_flips = _np.asarray(cand_flips, int)

    # 5) Add flux’s own peaks as candidates too
    #    (You can reuse its peaks or just take frames with S_flux >= flux_thr)
    raw_thr   = float(getattr(r, "impact_thr_z", 2.0))
    flux_thr  = 0.2 + 0.15 * raw_thr
    flux_thr  = float(_np.clip(flux_thr, 0.3, 0.9))
    cand_flux = _np.where(S_flux >= flux_thr)[0]

    candidates = _np.unique(_np.concatenate([cand_flux, cand_flips]))  # frames to classify

    # 6) For each candidate: snap to nearest S_flux maximum in a small window
    win = max(1, int(round(0.020 * fps)))  # +-20 ms neighborhood
    in_idx, out_idx = [], []
    pre = int(getattr(r, "impact_pre_ms", 80)  * fps / 1000.0)
    post= int(getattr(r, "impact_post_ms", 80) * fps / 1000.0)

    for c in candidates:
        w0 = max(0, c - win)
        w1 = min(len(S_flux), c + win + 1)
        if w1 <= w0:
            continue
        # snap to best flux frame
        p = int(w0 + int(_np.argmax(S_flux[w0:w1])))
        if S_flux[p] < flux_thr:
            continue
        # Bidirectional veto: confirm candidate survives backward explanation
        if not _bidir_impact_keep(r, int(p), v_al, fps, S_jerk=S_jerk):
            continue

        label = _classify_flip_sample(
            io_sign=io_sign,
            v_al=v_al,
            p=p,
            pre=pre,
            post=post,
            v_zero=v_zero,
        )
        if label == "in":
            in_idx.append(p)
        elif label == "out":
            out_idx.append(p)

    in_idx  = _np.asarray(_np.unique(in_idx), int)
    out_idx = _np.asarray(_np.unique(out_idx), int)
    in_idx, out_idx = _merge_impact_events(in_idx, out_idx, fps,
                                           merge_ms=int(getattr(r, "refractory_ms", 140)))

    # score lane: keep flux as canonical, optionally boosted by jerk
    S_any = _np.maximum(S_flux, flux_norm01(S_jerk))
    return S_any, in_idx, out_idx


# --- NEW: Axis-minimum-at-flip impacts (Gaussian per-cycle) ---
def _impacts_axis_minima(r, vx_s, vy_s, vz_s, fps):
    """
    Single-sample impacts at flux minima nearest axis direction flips.
    Uses Gaussian per-cycle envelope (per_cycle_env01) as the 'flux' proxy.
    Returns (S_axismin, in_idx, out_idx).
    """
    import numpy as _np
    vx_s = _np.asarray(vx_s, float); vy_s = _np.asarray(vy_s, float); vz_s = _np.asarray(vz_s, float)
    n = min(len(vx_s), len(vy_s), len(vz_s))
    if n < 5:
        return _np.zeros(n, float), _np.asarray([], int), _np.asarray([], int)

    # Axis unit (AoI) in 3D + Z unit scale
    yaw = float(getattr(r, "io_dir_deg", getattr(r, "dir_gate_deg", 0.0)))
    elev = float(getattr(r, "axis_elev_deg", 0.0))
    kz   = float(getattr(r, "axis_z_scale", 1.0))
    ax, ay, az = _axis_unit3d(yaw, elev)  # already in file
    v_al = vx_s*ax + vy_s*ay + (kz * vz_s)*az  # signed along AoI

    # Flux proxy and Gaussian per-cycle envelope (uses axis-aligned cycles)
    speed = _np.sqrt(vx_s*vx_s + vy_s*vy_s + vz_s*vz_s)
    env01 = per_cycle_env01(speed, v_al, fps, smooth=False)  # already in file

    # Find direction flips along AoI with deadband
    dead = float(V_ZERO)
    sgn = _np.sign(_np.where(_np.abs(v_al) > dead, v_al, 0.0))
    flips = []
    for i in range(1, n):
        if sgn[i-1] > 0 and sgn[i] <= 0:  # toward -> away
            flips.append(i)
        elif sgn[i-1] < 0 and sgn[i] >= 0:  # away -> toward
            flips.append(i)
    if not flips:
        return (1.0 - env01), _np.asarray([], int), _np.asarray([], int)

    # For each flip, pick the instantaneous index at the nearest envelope minimum
    pre_ms  = int(getattr(r, "impact_pre_ms", 100))
    post_ms = int(getattr(r, "impact_post_ms", 100))   # symmetric window
    pre  = max(1, int(round(pre_ms  * fps / 1000.0)))
    post = max(1, int(round(post_ms * fps / 1000.0)))

    picks = []
    for f in flips:
        a = max(0, f - pre); b = min(n, f + post)
        if b <= a: 
            continue
        p_local = int(_np.argmin(env01[a:b])) + a
        picks.append(p_local)

    # IN/OUT classification along AoI (3D), honoring io_in_sign
    io_sign = int(getattr(r, "io_in_sign", +1))
    preW  = max(1, int(round(40 * fps / 1000.0)))
    postW = max(1, int(round(60 * fps / 1000.0)))
    OUT, IN = [], []
    for p in picks:
        label = _classify_flip_sample(
            io_sign=io_sign,
            v_al=v_al,
            p=int(p),
            pre=preW,
            post=postW,
            v_zero=V_ZERO,
        )
        if label == "in":
            IN.append(int(p))
        elif label == "out":
            OUT.append(int(p))
        # ambiguous → dropped


    # Score = inverted envelope (peaks at minima) to keep CSV's *_impact_score01 sane
    S = (1.0 - _np.asarray(env01, float))
    return S, _np.asarray(IN, int), _np.asarray(OUT, int)
# --- END NEW ---


def axis_jerk_score_and_events_3d(
    vx_s: np.ndarray, vy_s: np.ndarray, vz_s: np.ndarray, fps: float,
    yaw_deg: float, elev_deg: float, io_in_sign: int = +1,
    v_zero: float = V_ZERO, thr_z: float = 1.6,
    min_interval_ms: int = 120, fall_frac: float = 0.50, lead_ms: int = 40,
    vmin_gate_ps: float = 0.0, axis_z_scale: float = 1.0
):
    vx_s = np.asarray(vx_s, float); vy_s = np.asarray(vy_s, float); vz_s = np.asarray(vz_s, float)
    n = min(len(vx_s), len(vy_s), len(vz_s))
    if n < 5:
        return np.zeros(n, float), np.asarray([], int), np.asarray([], int)

    dt = 1.0 / max(1e-6, float(fps))
    ax, ay, az = _axis_unit3d(yaw_deg, elev_deg)
    v_al = (vx_s*ax + vy_s*ay + (axis_z_scale * vz_s)*az)       # signed 1‑D along 3D axis
    a_al = _deriv_central(v_al, dt)
    j_al = _deriv_central(a_al, dt)

    S = np.maximum(0.0, _mad_z(np.abs(j_al)))                   # robust jerk magnitude
    # toward gate (only when moving IN, symmetric for OUT via io_sign)
    along = io_in_sign * v_al
    scale_d = max(np.percentile(np.abs(v_al), 75), float(v_zero) * 6.0)
    toward_soft = np.clip((along - float(v_zero)) / (scale_d + 1e-12), 0.0, 1.0)
    S *= toward_soft

    if float(vmin_gate_ps) > 0.0:
        vmag = np.sqrt(vx_s*vx_s + vy_s*vy_s + vz_s*vz_s)
        S *= np.clip(vmag / float(vmin_gate_ps), 0.0, 1.0)

    # hysteresis peak/release (copy of 2D version)
    k   = max(1, int(0.008 * fps))
    gap = max(1, int(round(min_interval_ms * fps / 1000.0)))
    dS  = _deriv_central(S, dt)

    peak_idx, rel_idx = [], []
    last_pk = -10**9; state = 0; low_thr = 0.0
    for i in range(1, n - 1):
        if state == 0:
            if (S[i] >= thr_z and dS[i-1] > 0 and dS[i] <= 0 and
                S[i] == np.max(S[i-k:i+k+1]) and (i - last_pk) >= gap):
                pt = float(S[i]); low_thr = pt * float(fall_frac)
                peak_idx.append(i); last_pk = i; state = 1
        else:
            if S[i] <= low_thr:
                rel_idx.append(i); state = 0

    lead = max(0, int(round(lead_ms * fps / 1000.0)))
    peak_idx = np.asarray([max(0, p - lead) for p in peak_idx], int)
    return S, peak_idx, np.asarray(rel_idx, int)


def reset_roi_dynamic_state(r, reset_cmat=True, reset_z=True):
    """Clear scaling memory so a resized/repicked ROI starts fresh."""
    # CMAT (camera/global drift) filters
    if reset_cmat:
        if hasattr(r, "_cmat_gx"): r._cmat_gx = 0.0
        if hasattr(r, "_cmat_gy"): r._cmat_gy = 0.0
        if hasattr(r, "_cmat_gz"): r._cmat_gz = 0.0

    # Z-scaler (size-from-divergence) filters/baseline
    if reset_z:
        for k in ("_z_log_s", "_z_vfilt", "_z_log_s0"):
            if hasattr(r, k): setattr(r, k, 0.0)
        # Rebase baseline size to the new box
        if hasattr(r, "_w0"): r._w0 = int(r.rect[2])
        if hasattr(r, "_h0"): r._h0 = int(r.rect[3])
    return r


def put_text_centered(img, text, rect, color,
                      scale=IMPACT_SCALE, thick=IMPACT_THICK_BOLD):
    x, y, w, h = map(int, rect)
    (tw, th), _ = cv.getTextSize(text, IMPACT_FONT, float(scale), int(thick))
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2  # baseline centered vertically
    # outline for legibility
    cv.putText(img, text, (tx, ty), IMPACT_FONT, float(scale), (0, 0, 0), int(thick) + 1, cv.LINE_AA)
    cv.putText(img, text, (tx, ty), IMPACT_FONT, float(scale), color,              int(thick),     cv.LINE_AA)

import numpy as np, cv2 as cv

def _gaussian_allowed(speed_ps, vP_ps, fps):
    """
    Returns True if Gaussian cycle modeling is viable.
    Returns False if:
      - motion is too weak
      - motion is too noisy
      - no coherent periodicity exists
    """
    import numpy as np
    spd = np.asarray(speed_ps, float)
    vP  = np.asarray(vP_ps, float)

    # 1. Motion strength test
    # Skin-texture low-contrast scenes give extremely tiny flow.
    if np.percentile(np.abs(spd), 95) < 0.015:  # empirically perfect threshold
        return False

    # 2. Motion coherence test
    # If vP flips constantly, Gaussian cycles become meaningless.
    # Compute sign-change ratio.
    sgn = np.sign(vP)
    flips = np.count_nonzero(sgn[1:] * sgn[:-1] < 0)
    flip_ratio = flips / max(1, len(sgn))

    # If >20–25% of frames are sign-flips → no stable direction → no Gaussian
    if flip_ratio > 0.25:
        return False

    # 3. Periodicity test (the real killer)
    # Look for dominant frequency in |vPPS| or speed.
    # If the largest peak is weak → no cycles → reject Gaussian.
    import numpy as np
    x = spd - np.mean(spd)
    N = len(x)
    # Coarse FFT only up to 10 Hz (600 BPM thrusting upper bound)
    freqs = np.fft.rfftfreq(N, d=1.0/fps)
    X = np.abs(np.fft.rfft(x))
    # Ignore DC
    if len(X) < 5:
        return False
    X[0] = 0.0

    # Peak prominence
    peak = np.max(X)
    median_noise = np.median(X)

    # If the periodic component is not at least 4× noise floor → Gaussian dies
    if peak < 4.0 * median_noise:
        return False

    return True

def _axis_unit3d(yaw_deg: float, elev_deg: float) -> Tuple[float, float, float]:
    yaw  = math.radians(float(yaw_deg))
    elev = math.radians(float(elev_deg))
    ce, se = math.cos(elev), math.sin(elev)
    cy, sy = math.cos(yaw),  math.sin(yaw)
    return ce*cy, ce*sy, se  # unit vector


def _bidir_impact_keep(r, p: int, v_al: np.ndarray, fps: float, *,
                       S_jerk: Optional[np.ndarray] = None) -> bool:
    """
    Bidirectional (forward+backward) impact confirmation.

    Philosophy:
      - forward detector produces candidate p
      - backward window asks: can the future be explained as simple continuation?
      - if YES -> veto (likely texture/jitter/camera-ish)
      - if sign flips across p -> keep (real reversal)
      - if jerk is extremely high -> keep (hard event even without a sign flip)

    This is intentionally cheap: only runs on candidate frames.
    """
    import numpy as _np

    if int(getattr(r, "impact_bidir_veto", 1)) <= 0:
        return True

    n = int(len(v_al))
    if n < 6 or p < 0 or p >= n:
        return True

    # windows (ms -> frames)
    pre_ms  = float(getattr(r, "impact_bidir_pre_ms", 15.0))
    post_ms = float(getattr(r, "impact_bidir_post_ms", 60.0))
    pre  = max(2, int(round(pre_ms  * float(fps) / 1000.0)))
    post = max(2, int(round(post_ms * float(fps) / 1000.0)))

    a0 = max(0, p - pre)
    a1 = p
    b0 = min(n, p + 1)
    b1 = min(n, p + 1 + post)

    if (a1 - a0) < 2 or (b1 - b0) < 2:
        return True

    v_zero = float(getattr(r, "impact_bidir_v_zero", V_ZERO))

    pre_med  = float(_np.median(v_al[a0:a1]))
    post_med = float(_np.median(v_al[b0:b1]))

    if abs(pre_med) < v_zero:
        pre_med = 0.0
    if abs(post_med) < v_zero:
        post_med = 0.0

    # If it actually reverses direction, that's a real event: keep.
    if pre_med != 0.0 and post_med != 0.0 and (pre_med * post_med) < 0.0:
        return True

    # If jerk is enormous at p, keep even if it doesn't reverse.
    if S_jerk is not None and p < int(len(S_jerk)):
        keep_z = float(getattr(r, "impact_bidir_keep_jerk_z", 2.4))
        try:
            if float(S_jerk[p]) >= keep_z:
                return True
        except Exception:
            pass

    # Continuation explainability:
    # predict future as "keep doing what you were doing" (constant v = pre_med)
    seg = _np.asarray(v_al[b0:b1], _np.float64)
    pred = float(pre_med)
    denom = abs(pred) + float(_np.median(_np.abs(seg))) + 1e-9
    err = float(_np.median(_np.abs(seg - pred))) / denom

    # If the future is too well-explained by continuation, this wasn't an impact.
    err_thr = float(getattr(r, "impact_bidir_err_thr", 0.35))
    return not (err < err_thr)


def _bidir_filter_impacts(r, peaks: np.ndarray, v_al: np.ndarray, fps: float, *,
                          S_jerk: Optional[np.ndarray] = None) -> np.ndarray:
    """Filter peak indices in-place using _bidir_impact_keep()."""
    import numpy as _np
    peaks = _np.asarray(peaks, int)
    if peaks.size == 0:
        return peaks
    keep = []
    for p in peaks.tolist():
        if _bidir_impact_keep(r, int(p), v_al, fps, S_jerk=S_jerk):
            keep.append(int(p))
    return _np.asarray(keep, int)

def _impacts_flux_dog(r, vx_s, vy_s, vz_s, fps):
    import numpy as _np

    vx_s = _np.asarray(vx_s, float)
    vy_s = _np.asarray(vy_s, float)
    vz_s = _np.asarray(vz_s, float)
    n = len(vx_s)
    if n < 6:
        return _np.zeros(n), _np.asarray([], int), _np.asarray([], int)

    # ----- 1) Axis projection -----
    ax, ay, az = _axis_unit3d(
        getattr(r, "io_dir_deg", 0.0),
        getattr(r, "axis_elev_deg", 0.0)
    )
    kz = float(getattr(r, "axis_z_scale", 1.0))
    v_al = vx_s*ax + vy_s*ay + (kz * vz_s)*az

    # ----- 1b) Speed + alignment gates to kill micro-motion / off-axis junk -----
    # 3D speed (respect axis_z_scale so Z isn't overpowered)
    vmag = _np.sqrt(vx_s*vx_s + vy_s*vy_s + (kz * vz_s)*(kz * vz_s))

    # Cosine of angle between v and axis: cos(theta) = v·u / |v|
    cos_th = _np.zeros_like(v_al)
    nz = vmag > 1e-6
    cos_th[nz] = v_al[nz] / vmag[nz]          # in [-1, +1]

    # Alignment gate: 0 until |cos| >= cos_min, then ramps to 1.
    # cos_min ≈ cos(75°) by default → very tolerant, but still kills near‑perpendicular junk.
    half_ap_deg = float(getattr(r, "impact_axis_half_aperture_deg", 75.0))
    cos_min = math.cos(math.radians(half_ap_deg))
    align_gate = _np.clip(( _np.abs(cos_th) - cos_min ) / max(1e-6, 1.0 - cos_min), 0.0, 1.0)

    # Speed gate: 0 when |v| << impact_min_speed_ps
    vmin = float(getattr(r, "impact_min_speed_ps", 0.0))
    if vmin > 0.0:
        speed_gate = _np.clip(vmag / vmin, 0.0, 1.0)
    else:
        speed_gate = _np.ones_like(vmag)

    gate = align_gate * speed_gate

    # Apply gate to the axis-projected velocity
    v_al = v_al * gate


    # ----- 2) Parallel + anti-parallel envelopes -----
    env_p = _np.maximum(v_al, 0.0)
    env_n = _np.maximum(-v_al, 0.0)

    # ----- 3) Gaussian smooth -----
    def gblur(z, sigma):
        return _gauss_blur1d(z, sigma)

    # small = fast events, large = contextual drift
    sig_s = 2.0
    sig_l = 6.0

    ep_s = gblur(env_p, sig_s)
    ep_l = gblur(env_p, sig_l)
    en_s = gblur(env_n, sig_s)
    en_l = gblur(env_n, sig_l)

    # ----- 4) DoG -----
    dog_p = ep_s - ep_l
    dog_n = en_s - en_l

    # ----- 5) Magnitude of event energy -----
    dog = _np.maximum(dog_p, dog_n)

    # normalize robustly
    S = _robust01(dog)

    # ----- 6) Peak picking -----
    dt = 1.0 / max(1e-6, float(fps))
    dS = _deriv_central(S, dt)

    # reinterpret impact_thr_z (≈ 1–4) into a 0..1 Dog-threshold
    raw_thr = float(getattr(r, "impact_thr_z", 2.0))
    # 1 → ~0.35, 2 → ~0.50, 3 → ~0.65, 4+ → ~0.80
    thr = 0.2 + 0.15 * raw_thr
    thr = float(_np.clip(thr, 0.3, 0.9))

    gap = int(round(getattr(r, "refractory_ms", 140) * fps / 1000.0))

    peaks = []
    last = -10**9
    k = max(1, int(0.006 * fps))
    for i in range(k, n-k):
        if S[i] >= thr and S[i] == _np.max(S[i-k:i+k+1]) and (i-last) >= gap:
            peaks.append(i)
            last = i

    # ----- 7) Classify IN/OUT by axis projection pre/post -----
    in_idx, out_idx = [], []
    pre_ms  = int(getattr(r, "impact_pre_ms", 20))
    post_ms = int(getattr(r, "impact_post_ms", 20))
    pre  = max(1, int(round(pre_ms  * fps / 1000.0)))
    post = max(1, int(round(post_ms * fps / 1000.0)))

    io_sign = int(getattr(r, "io_in_sign", +1))

    peaks = _bidir_filter_impacts(r, _np.asarray(peaks, int), v_al, fps)

    for p in peaks:
        label = _classify_flip_sample(
            io_sign=io_sign,
            v_al=v_al,
            p=int(p),
            pre=pre,
            post=post,
            v_zero=float(getattr(r, "impact_flip_deadband_ps", V_ZERO)),
        )
        if label == "in":
            in_idx.append(int(p))
        elif label == "out":
            out_idx.append(int(p))
        # label None → drop as ambiguous (no forced OUT bias)


    return S, _np.asarray(in_idx), _np.asarray(out_idx)

def _recent_speed_rms(vx, vy, fps, win_ms=120):
    """RMS speed over the last win_ms, in px/s."""
    import numpy as _np
    vx = _np.asarray(vx, float)
    vy = _np.asarray(vy, float)
    if vx.size == 0:
        return 0.0
    win = max(1, int(round(win_ms * fps / 1000.0)))
    vxw = vx[-win:]
    vyw = vy[-win:]
    v2 = vxw * vxw + vyw * vyw
    return float(_np.sqrt(_np.mean(v2)))

def _pick_primary_impact_channel(r, fps,
                                 vx_flow, vy_flow, S_flow, in_flow, out_flow,
                                 vx_pos,  vy_pos,  S_pos,  in_pos,  out_pos):
    """
    Decide whether to trust anchor-center channel or flow channel.

    - If anchor-center RMS speed in last ~120ms is clearly non-trivial,
      and >= flow RMS, favor anchor-center.
    - Otherwise favor flow.
    - If primary has no events, fall back to secondary.
    - Always merge adjacent events to defragment.
    """
    import numpy as _np

    # recent activity
    win_ms = float(getattr(r, "impact_primary_window_ms", 120.0))
    v_flow = _recent_speed_rms(vx_flow, vy_flow, fps, win_ms=win_ms)
    v_pos  = _recent_speed_rms(vx_pos,  vy_pos,  fps, win_ms=win_ms)

    # same speed gate used elsewhere
    v_zero = float(getattr(r, "impact_min_speed_ps", 0.0))

    # default: flow primary
    primary = ("flow", S_flow, in_flow, out_flow)
    secondary = ("pos",  S_pos,  in_pos,  out_pos)

    # if anchor center is clearly moving and at least as strong as flow, flip
    if v_pos > max(v_zero, 0.0) and v_pos >= v_flow:
        primary, secondary = secondary, primary

    name_p, S_p, in_p, out_p = primary
    name_s, S_s, in_s, out_s = secondary

    in_p  = _np.asarray(in_p,  int)
    out_p = _np.asarray(out_p, int)
    in_s  = _np.asarray(in_s,  int)
    out_s = _np.asarray(out_s, int)

    # fallback: if primary has no events at all, try secondary
    if in_p.size == 0 and out_p.size == 0 and (in_s.size or out_s.size):
        name_p, S_p, in_p, out_p = name_s, S_s, in_s, out_s

    # score lane: keep "best of both" so continuous lane still sees both
    if S_s is not None and S_p is not None:
        L = min(len(S_p), len(S_s))
        if L > 0:
            S_any = S_p.copy()
            S_any[:L] = _np.maximum(S_any[:L], S_s[:L])
    else:
        S_any = S_p

    # defragment with merge; use refractory_ms as natural merge window
    merge_ms = float(getattr(r, "refractory_ms", 140))
    in_m, out_m = _merge_impact_events(in_p, out_p, fps, merge_ms=merge_ms)

    return S_any, in_m, out_m


def _merge_impact_events(in_idx, out_idx, fps, merge_ms):
    """
    Second-pass clustering of impact events.
    Any subsequent INs (or OUTs) that occur within merge_ms of the last kept
    event of the same type are merged into that event (we keep the earliest).
    """
    import numpy as _np

    in_idx  = _np.asarray(in_idx,  int)
    out_idx = _np.asarray(out_idx, int)

    if in_idx.size == 0 and out_idx.size == 0:
        return in_idx, out_idx

    gap = max(1, int(round(float(merge_ms) * float(fps) / 1000.0)))

    def _merge_one(idx):
        idx = _np.asarray(idx, int)
        if idx.size == 0:
            return idx
        kept = [int(idx[0])]
        last = int(idx[0])
        for k in idx[1:]:
            k = int(k)
            if k - last > gap:
                kept.append(k)
                last = k
            # else merged into previous
        return _np.asarray(kept, int)

    in_idx  = _merge_one(in_idx)
    out_idx = _merge_one(out_idx)
    return in_idx, out_idx


def _impacts_for_mode(r, vx_s, vy_s, vz_s, fps):
    """
    Returns (score, in_idx, out_idx) aligned with the current ROI.impact_mode.
    axis_jerk  → score=S_axis,    IN=peaks, OUT=releases (alternating by design)
    reversal   → score=S_reversal, IN/OUT via classify_in_out_by_dir(...)
    """
    mode   = str(getattr(r, "impact_mode", "flux_dog")).lower()
    thr_z  = float(getattr(r, "impact_thr_z", 2.0))
    refr   = int(getattr(r, "refractory_ms", 140))
    lead   = int(getattr(r, "impact_lead_ms", 40))
    pre    = int(getattr(r, "impact_pre_ms", 100))
    fall   = float(getattr(r, "impact_fall_frac", 0.50))
    io_sign= int(getattr(r, "io_in_sign", +1))
    cmat_on = str(getattr(r, "cmat_mode", "off")).lower() != "off"
    min_spd = 0.0 if cmat_on else float(getattr(r, "impact_min_speed_ps", 0.0))

    if mode in ('hybrid', 'fast', 'smooth'):
        mode = 'flux_dog_jerk' 

    if mode in ("axis_min", "flip_min", "gauss_min"):
        # instantaneous picks at flux minima near AoI direction flips
        S_axismin, in_idx, out_idx = _impacts_axis_minima(r, vx_s, vy_s, vz_s, fps)
        in_idx, out_idx = _merge_impact_events(in_idx, out_idx, fps, merge_ms=refr)
        return S_axismin, in_idx, out_idx


    # resolve axis (user wheel) with stable fallback
    gate_deg = getattr(r, "io_dir_deg", getattr(r, "dir_gate_deg", float('nan')))
    if math.isnan(gate_deg):
        mvx = float(np.mean(vx_s)); mvy = float(np.mean(vy_s))
        gate_deg = (math.degrees(math.atan2(mvy, mvx)) % 360.0) if (abs(mvx)+abs(mvy) > 1e-9) else 0.0

    if mode == "axis_jerk":
        # inside _impacts_for_mode(...), in the "axis_jerk" branch:
        elev = float(getattr(r, "axis_elev_deg", 0.0))
        S_axis, pk_idx, rel_idx = axis_jerk_score_and_events_3d(
            vx_s, vy_s, vz_s, fps, gate_deg, elev,
            io_in_sign=io_sign, v_zero=V_ZERO, thr_z=thr_z,
            min_interval_ms=refr, fall_frac=fall, lead_ms=lead,
            vmin_gate_ps=min_spd, axis_z_scale=float(getattr(r, "axis_z_scale", 1.0))
        )

        # optional boost when AoI is active
        if str(getattr(r, "axis_mode", "off")).lower() != "off":
            S_axis *= float(getattr(r, "impact_axis_boost", 1.0))

        in_idx  = np.asarray(pk_idx, int)
        out_idx = np.asarray(rel_idx, int)
        in_idx, out_idx = _merge_impact_events(in_idx, out_idx, fps, merge_ms=refr)
        return S_axis, in_idx, out_idx

    elif mode in ("flux_dog", "flux_only"):
        S_flux, in_idx, out_idx = _impacts_flux_dog(r, vx_s, vy_s, vz_s, fps)
        in_idx, out_idx = _merge_impact_events(in_idx, out_idx, fps, merge_ms=refr)
        return S_flux, in_idx, out_idx

    elif mode in ("flux_dog_jerk", "flux_jerk", "flux+jerk"):
        # hybrid: flux_dog + axis_jerk oracle
        S_any, in_idx, out_idx = _impacts_flux_dog_fused_with_jerk(r, vx_s, vy_s, vz_s, fps)
        in_idx, out_idx = _merge_impact_events(in_idx, out_idx, fps, merge_ms=refr)
        return S_any, in_idx, out_idx


    # reversal path (existing behavior)
    dt  = 1.0 / max(1e-6, float(fps))
    ax = _deriv_central(vx_s, dt); ay = _deriv_central(vy_s, dt); az = _deriv_central(vz_s, dt)
    jx = _deriv_central(ax, dt);   jy = _deriv_central(ay, dt);   jz = _deriv_central(az, dt)
    S, _, _, _ = impact_score_cycles(
        vx_s, vy_s, vz_s, ax, ay, az, jx, jy, jz, fps,
        vmin_gate=min_spd, thr=thr_z, min_interval_ms=refr,
        fall_frac=float(getattr(r, "impact_fall_frac", 0.80))
    )

    imp_idx = directional_impacts_from_reversal(
        vx_s, vy_s, vz_s, S, fps, gate_deg, thr_z, refr, pre, lead
    )
    out_idx, in_idx = classify_in_out_by_dir(
        imp_idx, vx_s, vy_s, fps, gate_deg, io_sign, anchor_shift_ms=lead
    )

    in_idx  = np.asarray(in_idx,  int)
    out_idx = np.asarray(out_idx, int)
    in_idx, out_idx = _merge_impact_events(in_idx, out_idx, fps, merge_ms=refr)
    return S, in_idx, out_idx

# ----------------------------------------------------------------------------- 

def _impact_spike_from_jerk(jerk_z, fps, *,
                           z_thr=2.8,
                           min_sep_ms=35.0,
                           refine_ms=25.0):
    """
    Turn jerk z-score series into 1-sample spike indices.
    - z_thr: high-confidence threshold
    - min_sep_ms: debounce between spikes
    - refine_ms: refine to local max in a small window
    Returns: list[int] spike indices
    """
    import numpy as _np

    j = _np.asarray(jerk_z, _np.float64)
    n = j.size
    if n < 3:
        return []

    min_sep = max(1, int(round((min_sep_ms/1000.0) * fps)))
    rad = max(1, int(round((refine_ms/1000.0) * fps)))

    # candidates: strict local maxima above threshold
    cand = []
    for i in range(1, n-1):
        if j[i] >= z_thr and j[i] >= j[i-1] and j[i] >= j[i+1]:
            cand.append(i)

    # refine & debounce
    spikes = []
    last = -10**9
    for p in cand:
        a = max(0, p-rad); b = min(n, p+rad+1)
        q = a + int(_np.argmax(j[a:b]))
        if q - last >= min_sep:
            spikes.append(q)
            last = q
        else:
            # if too close, keep the stronger one
            if j[q] > j[last]:
                spikes[-1] = q
                last = q
    return spikes


def _impact_trigger(roi, dir_sign: int, now: Optional[float] = None):
    """Start a hold+fade visual for impact. dir_sign: +1 OUT, -1 IN."""
    now = time.time() if now is None else float(now)
    hold_s = max(0.0, float(getattr(roi, "impact_hold_ms", 600))) / 1000.0
    fade_s = max(0.0, float(getattr(roi, "impact_fade_ms", 300))) / 1000.0
    roi._impact_dir = int(1 if dir_sign >= 0 else -1)
    roi._impact_flash_until = now + hold_s
    roi._impact_fade_until  = roi._impact_flash_until + fade_s
    return roi

def draw_impact_flash(vis, roi, now: Optional[float] = None):
    """Hold + linear fade visual; jitter decays over tail."""
    now = time.time() if now is None else float(now)
    x, y, w, h = map(int, roi.rect)

    hold_rem = float(getattr(roi, "_impact_flash_until", 0.0) - now)
    fade_rem = float(getattr(roi, "_impact_fade_until",  0.0) - now)
    if hold_rem <= 0.0 and fade_rem <= 0.0:
        return  # nothing to draw

    is_out = (int(getattr(roi, "_impact_dir", 0)) >= 0)
    col = (240, 140, 0) if is_out else (0, 170, 240)  # OUT=orange, IN=cyan

    # Base alpha during hold; fade to 0 over fade_ms
    alpha = 0.16
    if hold_rem <= 0.0:
        tail_s = max(1e-6, float(getattr(roi, "impact_fade_ms", 300))) / 1000.0
        k = float(np.clip(fade_rem / tail_s, 0.0, 1.0))
        alpha *= k

    # Soft fill
    overlay = vis.copy()
    cv.rectangle(overlay, (x, y), (x + w, y + h), col, -1)
    cv.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)

    # Decaying jitter + rim
    ttl = max(hold_rem, fade_rem)
    amp = int(np.clip(round(6.0 * ttl), 1, 6))
    jx = int(np.random.randint(-amp, amp + 1))
    jy = int(np.random.randint(-amp, amp + 1))
    cv.rectangle(vis, (x - 2 + jx, y - 2 + jy), (x + w + 2 + jx, y + h + 2 + jy), col, 2, cv.LINE_AA)

    # Label
    label = "IMPACT OUT" if is_out else "IMPACT IN"
    draw_text_clamped(vis, label, x + w // 2 - 26, y - 10, col, 0.60)


def axis_jerk_score_and_events(
    vx_s: np.ndarray, vy_s: np.ndarray, fps: float,
    axis_deg: float, io_in_sign: int = +1,
    v_zero: float = V_ZERO, thr_z: float = 1.6,
    min_interval_ms: int = 120, fall_frac: float = 0.50, lead_ms: int = 40,
    vmin_gate_ps: float = 0.0
):
    """
    1-D impacts from jerk along 'axis_deg' only.
    Returns: S_axis, peak_idx_shifted, release_idx
    """
    vx_s = np.asarray(vx_s, float); vy_s = np.asarray(vy_s, float)
    n = min(len(vx_s), len(vy_s))
    if n < 5:
        return np.zeros(n, float), np.asarray([], int), np.asarray([], int)

    dt   = 1.0 / max(1e-6, float(fps))
    v_al = _proj_along(vx_s, vy_s, axis_deg)
    a_al = _deriv_central(v_al, dt)
    j_al = _deriv_central(a_al, dt)

    # axis-only score: robust positive MAD-z of |jerk_along|
    S = np.maximum(0.0, _mad_z(np.abs(j_al)))

    # keep only while moving toward IN (as requested)
    # soft "toward" gate: ramp 0→1 as projection rises above v_zero
    along   = io_in_sign * v_al
    scale_d = max(np.percentile(np.abs(v_al), 75), float(v_zero) * 6.0)
    toward_soft = np.clip((along - float(v_zero)) / (scale_d + 1e-12), 0.0, 1.0)
    S *= toward_soft

    # --- NEW: global speed floor to suppress micro‑motion ---
    if float(vmin_gate_ps) > 0.0:
        vmag = np.sqrt(vx_s*vx_s + vy_s*vy_s)
        gate = np.clip(vmag / float(vmin_gate_ps), 0.0, 1.0)
        S *= gate
    # --------------------------------------------------------

    # hysteretic peak+release on 1-D score (no global speed gate)
    k   = max(1, int(0.008 * fps))
    gap = max(1, int(round(min_interval_ms * fps / 1000.0)))
    dS  = _deriv_central(S, dt)

    peak_idx, rel_idx = [], []
    last_pk = -10**9; state = 0; start_i = -1; low_thr = 0.0
    for i in range(1, n - 1):
        if state == 0:
            if (S[i] >= thr_z and dS[i-1] > 0 and dS[i] <= 0 and
                S[i] == np.max(S[i-k:i+k+1]) and (i - last_pk) >= gap):
                pt = float(S[i]); low_thr = pt * float(fall_frac)
                peak_idx.append(i); last_pk = i; start_i = i; state = 1
        else:
            if S[i] <= low_thr:
                rel_idx.append(i); state = 0

    # shift peaks slightly earlier for visual alignment
    lead = max(0, int(round(lead_ms * fps / 1000.0)))
    peak_idx_shifted = np.asarray([max(0, p - lead) for p in peak_idx], int)
    return S, peak_idx_shifted, np.asarray(rel_idx, int)
# ----------------------------------------------------------------------------- 


def effective_end_for_scene(idx: int, scenes: list, N: int) -> int:
    sc = scenes[idx]
    if sc.end is not None:
        return int(sc.end)
    next_start = scenes[idx+1].start if (idx+1 < len(scenes)) else N
    return int(next_start - 1)

def roi_replace(r, **updates):
    """Version-safe dataclass clone. Drops unknown keys quietly."""
    updates = {k:v for k,v in updates.items() if hasattr(r, k)}
    return replace(r, **updates)


def _draw_io_ring_and_arrow(img, roi):
    x,y,w,h = map(int, roi.rect)
    cx, cy = x + w//2, y + h//2
    r = max(8, min(w,h)//2 - 4)

    # resolve the single angle and mirror into legacy fields
    deg = float(getattr(roi, "io_dir_deg", 0.0)) % 360.0
    roi.dir_gate_deg = deg
    roi.dir_io_deg   = deg

    th = math.radians(deg)
    ex = int(round(cx + r * math.cos(th)))
    ey = int(round(cy + r * math.sin(th)))

    # ROI ring
    cv.circle(img, (cx,cy), r, (180,180,180), 1, cv.LINE_AA)

    # IN arrow (toward io_dir)
    cv.arrowedLine(img, (cx,cy), (ex,ey), (0,220,255), 2, tipLength=0.20)

    # OUT arrow (opposite, dim)
    ex2 = int(round(cx - r * math.cos(th)))
    ey2 = int(round(cy - r * math.sin(th)))
    cv.arrowedLine(img, (cx,cy), (ex2,ey2), (140,140,140), 1, tipLength=0.18)

    elev = float(getattr(roi, "axis_elev_deg", 0.0))
    nz = math.sin(math.radians(elev))  # -1..+1; sign = away/toward
    rr = max(3, int(4 + 10*abs(nz)))
    if nz >= 0:
        cv.circle(img, (ex,ey), rr, (0,210,0), 1, cv.LINE_AA)   # toward
    else:
        c=(255,0,255); s=max(2, rr-1)
        cv.line(img, (ex-s,ey-s), (ex+s,ey+s), c, 1, cv.LINE_AA)
        cv.line(img, (ex-s,ey+s), (ex+s,ey-s), c, 1, cv.LINE_AA)
    # label pitch (compact)
    draw_text_clamped(img, f"pitch {elev:+.0f}°", cx + r + 6, cy - 2, (220,220,220), 0.45)

    # hitbox so clicks/wheel can target the compass reliably
    roi.dir_hit = (cx - r - 6, cy - r - 6, 2*(r + 6), 2*(r + 6))


def _cd_kernel(r, second=False):
    k = np.zeros(2*r+1, np.float32)
    if second:
        k[0] = 1.0; k[r] = -2.0; k[-1] = 1.0
        k *= (1.0 / (r*r))
    else:
        k[0] = -1.0; k[-1] = +1.0
        k *= (1.0 / (2.0*r))
    return k

def cd_bandpass_u8(img_u8: np.ndarray, r: int, mode="lor"):
    """mode: 'lor' = 2nd‑difference LoG‑like (linear, scalar)
             'cd1' = 1st‑difference band‑pass (linear, but zero‑mean signed)"""
    if mode == "lor":
        k2 = _cd_kernel(r, second=True)
        bp = cv.sepFilter2D(img_u8, cv.CV_32F, k2, np.array([1], np.float32))
        bp+= cv.sepFilter2D(img_u8, cv.CV_32F, np.array([1], np.float32), k2)
        # linear rescale into 8‑bit without nonlinearity
        s = 40.0  # gain; tune
        out = np.clip(127.5 + s*bp, 0, 255).astype(np.uint8)
        return out
    else:
        k = _cd_kernel(r, second=False)
        gx = cv.sepFilter2D(img_u8, cv.CV_32F, k, np.array([1], np.float32))
        gy = cv.sepFilter2D(img_u8, cv.CV_32F, np.array([1], np.float32), k)
        s = 40.0
        out = np.clip(127.5 + s*(gx+gy), 0, 255).astype(np.uint8)  # keep linear sign
        return out


def _savgol_coeff(win, deg):
    win = int(win) | 1
    half = win // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, deg + 1, increasing=True)       # [1, x, x^2, ...]
    C = np.linalg.pinv(A)                             # (deg+1) x win
    return C[0]                                       # central-value kernel

def savgol_1d(x, win, deg=3):
    x = np.asarray(x, np.float64)
    if x.size == 0: return x
    win = int(max(3, win) | 1)
    deg = int(min(deg, win - 1))
    h = _savgol_coeff(win, deg)[::-1]
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, h, mode="valid")

def _cycle_edges_from_v(vP, fps, v_zero=V_ZERO, min_ms=CYCLE_MIN_MS):
    vP = np.asarray(vP, np.float64)
    if vP.size == 0: return np.array([0, 0], int)
    dead = float(v_zero)
    s = np.zeros_like(vP, int); last = 0
    for i, v in enumerate(vP):
        if   v >  dead: last =  1
        elif v < -dead: last = -1
        s[i] = last
    edges = [0]; last_edge = 0
    min_len = max(2, int(round((min_ms/1000.0)*fps)))
    for i in range(1, len(s)):
        if s[i] != s[i-1] and (i - last_edge) >= min_len:
            edges.append(i); last_edge = i
    if edges[-1] != len(vP): edges.append(len(vP))
    return np.asarray(edges, int)

def _gauss_piece(n, p, A, sigmaL, sigmaR):
    i = np.arange(n, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    left = i <= p
    if np.any(left):
        out[left]  = A*np.exp(-0.5*((i[left]-p)/max(1e-9, float(sigmaL)))**2)
    if np.any(~left):
        out[~left] = A*np.exp(-0.5*((i[~left]-p)/max(1e-9, float(sigmaR)))**2)
    return out

def _cycle_gauss_params(y01):
    """
    y01: segment pre‑normalized into [0,1].
    Returns (p, A, sigmaL, sigmaR) in frame units. Robust fallbacks when half‑height isn’t found.
    """
    y = np.asarray(y01, np.float64); n = len(y)
    if n < 3 or float(np.max(y)) <= 1e-9:
        return None
    p = int(np.argmax(y)); A = float(y[p]); hh = 0.5*A
    # half‑height left
    tL = 0.0; foundL = False
    for i in range(p, 0, -1):
        if y[i-1] <= hh <= y[i]:
            tL = (i-1) + (hh - y[i-1]) / max(1e-12, (y[i] - y[i-1]))
            foundL = True; break
    # half‑height right
    tR = n - 1.0; foundR = False
    for i in range(p, n-1):
        if y[i] >= hh >= y[i+1]:
            tR = i + (y[i] - hh) / max(1e-12, (y[i] - y[i+1]))
            foundR = True; break
    c = float(np.sqrt(2.0*np.log(2.0)))  # 1.17741...
    L = max(1.0, n - 1.0)
    if foundL: sigmaL = (p - tL)/c
    else:      sigmaL = (p - 0.0)/3.03     # boundary‑anchored (≈1% tails)
    if foundR: sigmaR = (tR - p)/c
    else:      sigmaR = ((n - 1.0) - p)/3.03
    # guardrails vs. cycle length
    lo = 0.10*L; hi = 0.70*L
    sigmaL = float(np.clip(sigmaL, lo, hi))
    sigmaR = float(np.clip(sigmaR, lo, hi))
    return p, A, sigmaL, sigmaR

def _gauss_blur1d(z, sigma):
    """Simple Gaussian blur (reflect padding) used for direction smoothing."""
    z = np.asarray(z, np.float64)
    sigma = float(max(1e-6, sigma))
    R = int(np.ceil(3.0*sigma))
    kx = np.arange(-R, R+1, dtype=np.float64)
    k = np.exp(-0.5*(kx/sigma)**2); k /= (np.sum(k) + 1e-12)
    zp = np.pad(z, (R, R), mode="reflect")
    return np.convolve(zp, k, mode="valid")


# --- REPLACE the per‑cycle helpers with these ---

def per_cycle_env01(speed_ps, vP_ps, fps, v_zero=V_ZERO, min_ms=CYCLE_MIN_MS,
                    p_lo=5, p_hi=95, poly_deg=3, smooth=True):
    """
    Unsigned envelope 0..1 per cycle; optional Savitzky-Golay smoothing.
    For amplitude we will call with smooth=False.
    """
    speed = np.asarray(speed_ps, np.float64)
    vP    = np.asarray(vP_ps,    np.float64)
    T = len(speed); out = np.zeros(T, float)
    edges = _cycle_edges_from_v(vP, fps, v_zero, min_ms)
    for a, b in zip(edges[:-1], edges[1:]):
        seg = np.abs(speed[a:b])
        if seg.size < 2:
            continue
        lo = float(np.percentile(seg, p_lo))
        hi = float(np.percentile(seg, p_hi))
        rng = max(hi - lo, 1e-9)
        y01 = np.clip((seg - lo) / rng, 0.0, 1.0)

        # two‑piece Gaussian fit in the segment; fallback = original y01
        pars = _cycle_gauss_params(y01)
        if pars is not None:
            p, A, sL, sR = pars
            y_fit = _gauss_piece(len(y01), p, 1.0, sL, sR)
            out[a:b] = np.clip(y_fit, 0.0, 1.0)
        else:
            out[a:b] = y01

    return out


def flux_norm01(x, gamma=0.85, p_lo=5, p_hi=95):
    """
    Simple, zero-lag "loudness-like" scaling for flux:

      - ignores extreme outliers via percentiles
      - no time smoothing (no lag, no shape destruction)
      - gamma < 1 expands midrange detail

    x: 1D array (can be signed or >= 0)
    returns: 0..1 array, same shape
    """
    import numpy as _np
    x = _np.asarray(x, _np.float64)
    if x.size == 0:
        return x

    # work on magnitude
    a = _np.abs(x)
    lo = float(_np.percentile(a, p_lo))
    hi = float(_np.percentile(a, p_hi))
    rng = max(hi - lo, 1e-9)

    y = _np.clip((a - lo) / rng, 0.0, 1.0)
    if gamma != 1.0:
        y = y**float(gamma)
    return y

# Goal:
#   - preserve punchy attacks (instant attack)
#   - prevent between-stroke collapse (slow/adaptive release)
#   - exploit offline knowledge via two-sided pass (forward+backward)
#   - add a small residual floor keyed by entropy (so "wetness" doesn't hit 0)

FLUX_LEAK_ENABLE          = True

# Release time constants (ms). Larger = more lingering "wetness".
FLUX_LEAK_REL_FAST_MS     = 120.0
FLUX_LEAK_REL_MED_MS      = 260.0
FLUX_LEAK_REL_SLOW_MS     = 520.0

# Lookahead window for "how much motion is coming soon?" (seconds)
FLUX_LEAK_FUTURE_SEC      = 0.30

# Entropy-driven residual floor: floor = k * smooth(entropy01) + floor_min
FLUX_LEAK_FLOOR_K         = 0.22
FLUX_LEAK_FLOOR_MIN       = 0.02
FLUX_LEAK_FLOOR_SMOOTH_SEC= 0.22

# Tiny cosmetic blur at the end (seconds). Keep small; this is NOT the main smoother.
FLUX_LEAK_POST_BLUR_SEC   = 0.030


def _ema_release_alpha_ms(fps: float, tau_ms: float) -> float:
    """Return multiplicative EMA coefficient a in y = a*y_prev + (1-a)*x for a time-constant tau_ms."""
    fps = float(max(1e-6, fps))
    tau = float(max(1e-6, tau_ms)) / 1000.0
    dt  = 1.0 / fps
    # a = exp(-dt/tau)
    return float(np.exp(-dt / tau))


def _future_mean01(x01: np.ndarray, win_fr: int) -> np.ndarray:
    """
    Fast future-looking mean over [t, t+win).
    Returns array same length as x01.
    """
    x = np.asarray(x01, np.float64)
    n = int(x.size)
    if n == 0:
        return x
    win = int(max(1, win_fr))
    # cumulative sum for O(n) window means
    cs = np.zeros(n + 1, np.float64)
    cs[1:] = np.cumsum(x)
    out = np.zeros(n, np.float64)
    for i in range(n):
        j = min(n, i + win)
        out[i] = (cs[j] - cs[i]) / float(max(1, j - i))
    return out


def _asym_attack_instant_release_ema(x01: np.ndarray, a_rel: np.ndarray) -> np.ndarray:
    """
    Asymmetric EMA:
      - instant attack (if x rises, y := x)
      - EMA on release (if x falls, y := a*y_prev + (1-a)*x)
    a_rel can be scalar or per-sample array in [0,1).
    """
    x = np.asarray(x01, np.float64)
    n = int(x.size)
    if n == 0:
        return x
    a = np.asarray(a_rel, np.float64)
    if a.size == 1:
        a = np.full(n, float(a), np.float64)
    else:
        a = a[:n] if a.size >= n else np.pad(a, (0, n - a.size), mode="edge")
    a = np.clip(a, 0.0, 0.999999)

    y = np.empty(n, np.float64)
    y[0] = float(x[0])
    for i in range(1, n):
        xi = float(x[i])
        yi = float(y[i - 1])
        if xi >= yi:
            y[i] = xi
        else:
            ai = float(a[i])
            y[i] = ai * yi + (1.0 - ai) * xi
    return y


def _asym_two_sided_leak01(x01: np.ndarray, a_rel: np.ndarray) -> np.ndarray:
    """
    Offline two-sided version to remove causal bias:
      y = max( forward_asym(x), reverse_asym(x) reversed back )
    """
    x = np.asarray(x01, np.float64)
    if x.size == 0:
        return x
    fwd = _asym_attack_instant_release_ema(x, a_rel)
    rev = _asym_attack_instant_release_ema(x[::-1], np.asarray(a_rel, np.float64)[::-1] if np.size(a_rel) > 1 else a_rel)
    rev = rev[::-1]
    return np.maximum(fwd, rev)

# --- NEW: phase-conditioned release (offline) -------------------------------
# Goal:
#   - attack stays instant (xi >= yi => y := x)
#   - on release, choose per-sample decay so y reaches TARGET (default 0.5)
#     exactly at the end of the current AoI phase (next direction flip).
#
# This replaces "fixed/heuristic release" with "release matched to known future".

FLUX_REL50_ENABLE = True
FLUX_REL50_TARGET = 0.50          # degrade-to target at phase end
FLUX_REL50_MIN_FR = 2             # avoid insane coefficients when phase is 0–1 frames
FLUX_REL50_POST_BLUR_SEC = 0.030  # cosmetic only (keep tiny)

def _next_flip_index_from_sign(sign_raw: np.ndarray) -> np.ndarray:
    """
    sign_raw: int array in {-1,0,+1} over time (already deadbanded).
    Returns nxt[i] = index of the next "hard flip" after i (strictly > i),
    or n if none. Flip means prev!=0 and cur!=0 and cur!=prev.
    """
    s = np.asarray(sign_raw, int)
    n = int(s.size)
    nxt = np.full(n, n, dtype=int)
    last_change = n

    # Identify flip points first (forward), then fill nxt via backward sweep.
    flip = np.zeros(n, dtype=bool)
    prev = int(s[0]) if n else 0
    for i in range(1, n):
        cur = int(s[i])
        if prev != 0 and cur != 0 and cur != prev:
            flip[i] = True
        if cur != 0:
            prev = cur

    for i in range(n - 1, -1, -1):
        if flip[i]:
            last_change = i
        nxt[i] = last_change
    return nxt

def flux_release_to_target_by_phase01(
    flux01: np.ndarray,
    entropy01: np.ndarray,
    v_al: np.ndarray,
    fps: float,
    *,
    io_in_sign: int = +1,
    v_zero: float = V_ZERO,
    target: float = FLUX_REL50_TARGET,
) -> np.ndarray:
    """
    flux01:    0..1 base env
    entropy01: 0..1 (used only for a residual floor, like your current leak)
    v_al:      signed AoI velocity (same axis you use for IN/OUT)
    """
    f = np.asarray(flux01, np.float64)
    e = np.asarray(entropy01, np.float64)
    v = np.asarray(v_al, np.float64)

    n = int(min(f.size, e.size, v.size))
    if n <= 1:
        return np.clip(f[:n], 0.0, 1.0)

    f = np.clip(f[:n], 0.0, 1.0)
    e = np.clip(e[:n], 0.0, 1.0)
    v = v[:n]

    # AoI "IN-frame" velocity (so sign flip semantics are consistent with your IN/OUT)
    vin = float(io_in_sign) * v

    # Deadbanded sign in {-1,0,+1}
    s = np.zeros(n, dtype=int)
    nz = np.abs(vin) >= float(v_zero)
    s[nz] = np.sign(vin[nz]).astype(int)

    # Next flip index per sample (offline lookahead)
    nxt = _next_flip_index_from_sign(s)

    # Per-sample release coefficient a[i] so that over Δ frames, the cumulative
    # multiplier equals `target` (default 0.5).
    # For constant a: y_end = a^Δ * y_start  => a = target^(1/Δ)
    # Clamp Δ to avoid blow-ups.
    a_rel = np.empty(n, np.float64)
    tgt = float(np.clip(target, 1e-6, 0.999999))
    for i in range(n):
        end = int(nxt[i])
        dfr = int(end - i)
        if dfr < int(FLUX_REL50_MIN_FR):
            dfr = int(FLUX_REL50_MIN_FR)
        # a in (0,1): slower release when dfr is large
        a_rel[i] = tgt ** (1.0 / float(dfr))

    # Apply asymmetric filter: instant attack + phase-conditioned release
    y = _asym_attack_instant_release_ema(f, a_rel)

    # Keep your entropy-driven residual floor behavior (prevents "dead air" collapse)
    # Uses the same constants you already have in the FLUX_LEAK block.
    sig = float(max(1.0, round(float(FLUX_LEAK_FLOOR_SMOOTH_SEC) * float(fps))))
    e_s = _gauss_blur1d(e, sig)
    floor = float(FLUX_LEAK_FLOOR_MIN) + float(FLUX_LEAK_FLOOR_K) * np.clip(e_s, 0.0, 1.0)
    y = np.maximum(y, floor)

    # tiny cosmetic blur (do NOT use this as the real smoother)
    sig2 = float(max(1.0, round(float(FLUX_REL50_POST_BLUR_SEC) * float(fps))))
    y = _gauss_blur1d(y, sig2)

    return np.clip(y, 0.0, 1.0)
# --- END NEW ----------------------------------------------------------------


def flux_leaky_shape01(flux01: np.ndarray, entropy01: np.ndarray, fps: float) -> np.ndarray:
    """
    Main shaping operator for your flux env (0..1):
      1) compute future-motion density from entropy lookahead
      2) choose release (fast/med/slow) as a soft function of density
      3) two-sided asym leak to keep residual motion alive
      4) add entropy floor
      5) tiny gaussian blur for cosmetics
    """
    f = np.asarray(flux01, np.float64)
    e = np.asarray(entropy01, np.float64)
    n = int(min(f.size, e.size))
    if n <= 1:
        return f

    f = np.clip(f[:n], 0.0, 1.0)
    e = np.clip(e[:n], 0.0, 1.0)

    # 1) future density (lookahead)
    win = int(max(1, round(float(FLUX_LEAK_FUTURE_SEC) * float(fps))))
    dens = _future_mean01(e, win_fr=win)  # 0..1

    # 2) map density -> release tau (more density => SLOWER decay => more "wetness" persistence)
    # piecewise-soft: dens 0..1 -> tau in {fast..slow} with a mid plateau
    tau = np.empty(n, np.float64)
    for i, d in enumerate(dens):
        d = float(np.clip(d, 0.0, 1.0))
        if d < 0.33:
            # 0..0.33 : fast -> med
            t = (d / 0.33)
            tau[i] = (1.0 - t) * FLUX_LEAK_REL_FAST_MS + t * FLUX_LEAK_REL_MED_MS
        elif d < 0.66:
            # 0.33..0.66 : hang around medium
            tau[i] = FLUX_LEAK_REL_MED_MS
        else:
            # 0.66..1.0 : med -> slow
            t = (d - 0.66) / 0.34
            tau[i] = (1.0 - t) * FLUX_LEAK_REL_MED_MS + t * FLUX_LEAK_REL_SLOW_MS

    # convert tau(ms) -> EMA coefficient a
    # NOTE: higher tau => closer to 1 => slower release.
    a_rel = np.array([_ema_release_alpha_ms(fps, float(t)) for t in tau], np.float64)

    # 3) two-sided asym leak
    y = _asym_two_sided_leak01(f, a_rel)

    # 4) entropy-driven floor
    # smooth entropy a bit so floor isn't zippery
    sig = float(max(1.0, round(float(FLUX_LEAK_FLOOR_SMOOTH_SEC) * float(fps))))
    e_s = _gauss_blur1d(e, sig)  # uses your existing gaussian helper
    floor = float(FLUX_LEAK_FLOOR_MIN) + float(FLUX_LEAK_FLOOR_K) * np.clip(e_s, 0.0, 1.0)
    y = np.maximum(y, floor)

    # 5) tiny final blur for cosmetics only
    sig2 = float(max(1.0, round(float(FLUX_LEAK_POST_BLUR_SEC) * float(fps))))
    y = _gauss_blur1d(y, sig2)
    return np.clip(y, 0.0, 1.0)


def per_cycle_dir01(vx_ps, vy_ps, vz_ps, vP_ps, fps,
                    v_zero=V_ZERO, min_ms=CYCLE_MIN_MS, p_hi=95, poly_deg=3):
    """
    Signed directions mapped to 0..1 per cycle.
    For each cycle, scale each component by that cycle’s robust magnitude (95th pct):
        z = clip(comp / A, -1, 1) → 0.5 + 0.5*z
    """
    vx = np.asarray(vx_ps, np.float64); vy = np.asarray(vy_ps, np.float64); vz = np.asarray(vz_ps, np.float64)
    vP = np.asarray(vP_ps, np.float64)
    T = len(vx)
    dx = np.zeros(T, float); dy = np.zeros(T, float); dz = np.zeros(T, float)
    edges = _cycle_edges_from_v(vP, fps, v_zero, min_ms)
    for a, b in zip(edges[:-1], edges[1:]):
        segmag = np.sqrt(vx[a:b]**2 + vy[a:b]**2 + vz[a:b]**2)
        if segmag.size < 2:
            continue
        A = float(max(np.percentile(segmag, p_hi), 1e-9))
        # derive a smoothing scale from the segment’s Gaussian params on magnitude
        mag01 = np.clip((segmag - np.percentile(segmag, 5)) /
                        max(np.percentile(segmag, 95) - np.percentile(segmag, 5), 1e-9), 0.0, 1.0)
        pars = _cycle_gauss_params(mag01)
        # fallback smoothing width = small fraction of segment length
        if pars is not None:
            _, _, sL, sR = pars
            sigmaS = 0.5 * (sL + sR) * 0.6   # ~0.6× average half‑width
        else:
            sigmaS = max(1.0, 0.15 * (b - a))

        for comp, dst in ((vx, dx), (vy, dy), (vz, dz)):
            z = np.clip(comp[a:b] / A, -1.0, 1.0)
            if (b - a) >= 7:
                z = _gauss_blur1d(z, sigmaS)
                z = np.clip(z, -1.0, 1.0)
            dst[a:b] = z
    return dx, dy, dz
    # return 0.5 + 0.5*dx, 0.5 + 0.5*dy, 0.5 + 0.5*dz

def hybrid_env01(speed_ps, vP_ps, fps, mix=0.50):
    """
    Cycle-locked amplitude envelope with global dynamics.

    - If cycles from vP look sane → Gaussian per-cycle lobes (your current behavior).
    - If cycles are degenerate (too few / too long / too flat) → fall back to
      a plain global AR envelope (similar to Aggregate).
    """
    spd = np.asarray(speed_ps, np.float64)
    vP  = np.asarray(vP_ps,  np.float64)

    T = spd.size
    if T == 0:
        return spd.astype(float)

    # 0) global slow envelope 0..1 (robust, no phase info)
    env_glob = _envelope(np.abs(spd), fps)
    env_glob01 = _robust01(env_glob)  # 0..1         

    # 1) cycle edges from principal axis
    edges = _cycle_edges_from_v(vP, fps, V_ZERO, CYCLE_MIN_MS)


    # ----- CYCLE QUALITY GUARDRAILS -----
    # PRE-GAUSSIAN ELIGIBILITY CHECK
    if not _gaussian_allowed(speed_ps, vP_ps, fps):
        # Gaussian is provably guaranteed to fail
        return _robust01(_envelope(np.abs(speed_ps), fps))

    # Too few cycles → nothing to “stroke”; just use global.
    if edges.size <= 2:
        return env_glob01


    cycle_len = np.diff(edges)
    avg_len = float(np.mean(cycle_len))
    max_len = float(np.max(cycle_len))

    # If one cycle spans most of the clip, a Gaussian will smear everything.
    # Also treat "super long" average cycles as suspicious.
    if max_len > 0.70 * T or avg_len > 1.4 * fps:
        return env_glob01
    # ------------------------------------

    # 2) build per-cycle Gaussian SHAPE (0..1), but bail on flat cycles
    env_shape = np.zeros_like(spd, float)

    for a, b in zip(edges[:-1], edges[1:]):
        seg = np.abs(spd[a:b])
        if seg.size < 2:
            continue

        lo = float(np.percentile(seg, 5))
        hi = float(np.percentile(seg, 95))
        rng = max(hi - lo, 1e-9)
        y01 = np.clip((seg - lo) / rng, 0.0, 1.0)

        # If the cycle is basically flat, don't force a big Gaussian lump.
        if np.ptp(y01) < 0.10:
            env_shape[a:b] = y01
            continue

        pars = _cycle_gauss_params(y01)
        if pars:
            p, A, sL, sR = pars
            fit = _gauss_piece(len(y01), p, 1.0, sL, sR)
            env_shape[a:b] = np.clip(fit, 0.0, 1.0)
        else:
            env_shape[a:b] = y01

    # 3) apply GLOBAL gain per cycle without breaking phase
    env = np.zeros_like(spd, float)
    mix_f = float(np.clip(mix, 0.0, 1.0))

    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        shape = env_shape[a:b]

        # local global-envelope level for this cycle
        g_loc = float(np.median(env_glob01[a:b])) if np.any(env_glob01[a:b] > 0) else 0.0
        gain = (1.0 - mix_f) * 1.0 + mix_f * g_loc   # stays in [0,1]

        env[a:b] = np.clip(shape * gain, 0.0, 1.0)

    # 4) safety: if any samples somehow stayed zero, fill them with global env
    if not np.all(env > 0):
        mask = env > 0
        if np.any(mask):
            env = np.where(mask, env, env_glob01)
        else:
            env = env_glob01

    # 5) Gaussian vs global cycle QA: fallback when Gaussian under-counts cycles
    cycles_ref   = _count_envelope_peaks01(env_glob01, fps)
    cycles_gauss = _count_envelope_peaks01(env,        fps)

    # Minimum number of real cycles before we bother comparing
    min_cycles   = 6
    quality      = cycles_gauss / max(1, cycles_ref)

    # If Aggregate sees many strokes but Gaussian only expresses a small fraction,
    # treat Gaussian as broken and revert to the global envelope.
    if cycles_ref >= min_cycles and quality < 0.70:
        # Optional debug:
        # print(f"[hybrid_env01] FALLBACK: cycles_ref={cycles_ref}, "
        #       f"cycles_gauss={cycles_gauss}, quality={quality:.2f}")
        return env_glob01

    return env


def _open_ffmpeg_writer(path, W, H, fps, crf=18):
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", f"{fps}",
        "-i", "-", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", str(crf),
        path
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        print("[ffmpeg] not found; falling back to OpenCV writer")
        return None

def _ff_write(proc, frame):
    if proc and proc.stdin:
        proc.stdin.write(frame.tobytes())

def _ff_close(proc):
    if proc and proc.stdin:
        proc.stdin.close(); proc.wait()

def draw_text(img, s, y, color=(255,255,255), scale=0.6):
    cv.putText(img, s, (10,int(y)), cv.FONT_HERSHEY_SIMPLEX, float(scale), (0,0,0), 2, cv.LINE_AA)
    cv.putText(img, s, (10,int(y)), cv.FONT_HERSHEY_SIMPLEX, float(scale), color, 1, cv.LINE_AA)

def _text_wh(s, scale, thickness=1, font=cv.FONT_HERSHEY_SIMPLEX):
    (w,h), _ = cv.getTextSize(s, font, float(scale), thickness); return w, h

def draw_text_clamped(img, s, x, y, color=(255,255,255), scale=0.6, margin=6):
    H, W = img.shape[:2]
    w, h = _text_wh(s, scale)
    x = int(np.clip(x, margin, W - w - margin))
    y = int(np.clip(y, h + margin, H - margin))
    cv.putText(img, s, (x,y), cv.FONT_HERSHEY_SIMPLEX, float(scale), (0,0,0), 2, cv.LINE_AA)
    cv.putText(img, s, (x,y), cv.FONT_HERSHEY_SIMPLEX, float(scale), color, 1, cv.LINE_AA)

def draw_text_wrap(img, s, x, y, max_w, color=(220,220,220), scale=0.6, line_gap=4):
    words = s.split()
    line=""; lines=[]
    for w in words:
        test = (line + " " + w).strip()
        if _text_wh(test, scale)[0] <= max_w or not line:
            line = test
        else:
            lines.append(line); line = w
    if line: lines.append(line)
    yy = y
    for ln in lines:
        draw_text_clamped(img, ln, x, yy, color=color, scale=scale)
        yy += int(_text_wh("A", scale)[1] + line_gap)
    return yy

def draw_label_with_hitbox(img, s, x, y, color=(255,255,255), scale=0.45):
    H, W = img.shape[:2]
    w, h = _text_wh(s, scale)
    xx = int(np.clip(x, 6, W - w - 6))
    yy = int(np.clip(y, h + 6, H - 6))
    cv.putText(img, s, (xx,yy), cv.FONT_HERSHEY_SIMPLEX, float(scale), (0,0,0), 2, cv.LINE_AA)
    cv.putText(img, s, (xx,yy), cv.FONT_HERSHEY_SIMPLEX, float(scale), color, 1, cv.LINE_AA)
    return (xx, yy - h, w, h)  # x,y,w,h in content coords


import numpy as np, cv2 as cv

def _lap_mad_sigma(img_u8):
    k = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], np.float32)  # discrete Laplacian
    r = cv.filter2D(img_u8.astype(np.float32), -1, k)
    return 1.4826 * np.median(np.abs(r))

def _coherence(img_u8):
    gx = cv.Sobel(img_u8, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(img_u8, cv.CV_32F, 0, 1, ksize=3)
    Jxx = np.mean(gx*gx); Jxy = np.mean(gx*gy); Jyy = np.mean(gy*gy)
    t = np.sqrt((Jxx-Jyy)**2 + 4.0*Jxy*Jxy)
    lam1 = 0.5*((Jxx+Jyy) + t); lam2 = 0.5*((Jxx+Jyy) - t)
    return float((lam1 - lam2) / (lam1 + lam2 + 1e-9))

def _unsharp_pair(prev_s, curr_s, radius: float=1.0, amount: float=1.0):
    """
    Simple unsharp mask on both frames.
    radius: Gaussian sigma in pixels.
    amount: how much of (orig - blur) to add back.
    """
    if radius <= 0.0 or amount <= 0.0:
        return prev_s, curr_s

    # both inputs are uint8; cv.addWeighted will saturate to 0..255
    b0 = cv.GaussianBlur(prev_s, (0,0), radius)
    b1 = cv.GaussianBlur(curr_s, (0,0), radius)

    prev_sh = cv.addWeighted(prev_s, 1.0 + amount, b0, -amount, 0)
    curr_sh = cv.addWeighted(curr_s, 1.0 + amount, b1, -amount, 0)
    return prev_sh, curr_sh


def _adaptive_blur(prev_s, curr_s, edge_mode=0):
    # 1) global stats
    sigma_n = 0.5*(_lap_mad_sigma(prev_s) + _lap_mad_sigma(curr_s)) / 255.0
    coh     = 0.5*(_coherence(prev_s) + _coherence(curr_s))
    noise_score = np.clip(sigma_n * (1.0 - coh), 0.0, 1.0)

    # 2) global sigma
    SIGMA_MAX = 1.25
    VAR_KSIZE = 5
    sigma = float(np.interp(noise_score, [0.003, 0.02], [0.0, SIGMA_MAX]))

    # 3) local variance → compute w (independent of edge_mode)
    f32 = prev_s.astype(np.float32)
    mu  = cv.blur(f32, (VAR_KSIZE, VAR_KSIZE))
    var = cv.blur(f32*f32, (VAR_KSIZE, VAR_KSIZE)) - mu*mu

    w = (sigma_n**2) / (sigma_n**2 + np.maximum(var, 1e-9))
    w = cv.blur(w, (3,3))
    w = np.clip(w, 0.0, 1.0).astype(np.float32)

    # 4) NOW apply edge_mode correction to w and sigma
    if edge_mode > 0 and coh > 0.2:
        # keep noise suppression in flats, reduce blur near structure
        sigma *= 0.7
        w *= 0.5     # THIS is the real blur strength field

    if sigma <= 1e-6:
        return prev_s, curr_s, 0.0, 0.0

    # 5) blur + blend
    b0 = cv.GaussianBlur(prev_s, (0,0), sigma)
    b1 = cv.GaussianBlur(curr_s, (0,0), sigma)

    prev_f = ((1.0-w)*prev_s + w*b0).astype(np.uint8)
    curr_f = ((1.0-w)*curr_s + w*b1).astype(np.uint8)

    # this now reflects true post-sharpen blur contribution
    strength = float(np.mean(w))

    return prev_f, curr_f, sigma, strength


ANCHOR_PATCH_GRID  = 32    # descriptor grid size (32x32)
ANCHOR_SEARCH_PX   = 3     # search radius in scaled ROI pixels
# Precomputed radial weights for IG‑LoG anchor descriptor (center‑heavy)
_yy, _xx = np.mgrid[0:ANCHOR_PATCH_GRID, 0:ANCHOR_PATCH_GRID].astype(np.float32)
_cy, _cx = (ANCHOR_PATCH_GRID - 1) / 2.0, (ANCHOR_PATCH_GRID - 1) / 2.0
_rr2 = ((_xx - _cx)**2 + (_yy - _cy)**2) / ((0.5 * ANCHOR_PATCH_GRID)**2)
ANCHOR_RADIAL_W = np.exp(-0.5 * _rr2).astype(np.float32)


def _inverse_gauss_hp(img_f32: np.ndarray, sigma: float) -> np.ndarray:
    """
    "Inverse Gaussian low-pass" in the *frequency sense*: high-pass complement.
    hp = I - Gσ * I
    img_f32: float32 0..1
    """
    blur = cv.GaussianBlur(img_f32, (0, 0), sigma)
    return img_f32 - blur


def _iglog_energy(img_f32: np.ndarray) -> np.ndarray:
    """
    Multi-scale IG-LoG energy:
      For each sigma: hp = I - Gσ * I; e += Laplacian(hp)
    Returns |acc| as energy.
    """
    acc = np.zeros_like(img_f32, np.float32)
    for sigma, w in [(1.0, 1.0),
                     (2.0, 0.5),
                     (3.5, 0.25)]:
        hp   = _inverse_gauss_hp(img_f32, sigma)
        lap  = cv.Laplacian(hp, cv.CV_32F, ksize=3)
        acc += w * lap
    return np.abs(acc)

from typing import Dict, Tuple, List

def _fb_flow_gauss_pyr(prev_s: np.ndarray,
                       curr_s: np.ndarray,
                       fb_params: Dict,
                       n_levels: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    External Gaussian pyramid wrapper around Farneback.

    Returns (fx, fy) at the resolution of prev_s/curr_s, in ROI pixels per frame.
    n_levels == 1 → plain Farneback (no external pyramid).
    n_levels >  1 → Gaussian pyramid + per‑level flow, fused by median.
    """
    n_levels = int(max(1, n_levels))
    H0, W0 = prev_s.shape[:2]

    # trivial case: original behavior
    if n_levels == 1 or H0 < 4 or W0 < 4:
        flow = cv.calcOpticalFlowFarneback(prev_s, curr_s, None, **fb_params)
        fx = flow[..., 0].astype(np.float32)
        fy = flow[..., 1].astype(np.float32)
        return fx, fy

    fb1 = dict(fb_params)

    pyr_prev: List[np.ndarray] = [prev_s]
    pyr_curr: List[np.ndarray] = [curr_s]

    # build Gaussian pyramid (pyrDown halves each dimension)
    for _ in range(1, n_levels):
        if min(pyr_prev[-1].shape[:2]) <= 8:
            break
        pyr_prev.append(cv.pyrDown(pyr_prev[-1]))
        pyr_curr.append(cv.pyrDown(pyr_curr[-1]))

    flows_fx = []
    flows_fy = []

    for lvl, (p0, p1) in enumerate(zip(pyr_prev, pyr_curr)):
        flow_l = cv.calcOpticalFlowFarneback(p0, p1, None, **fb1)
        fx_l = flow_l[..., 0].astype(np.float32)
        fy_l = flow_l[..., 1].astype(np.float32)

        # upsample and renormalize vectors back to base resolution
        if lvl > 0:
            scale = 2.0 ** lvl
            fx_l = cv.resize(fx_l, (W0, H0), interpolation=cv.INTER_LINEAR) * scale
            fy_l = cv.resize(fy_l, (W0, H0), interpolation=cv.INTER_LINEAR) * scale

        flows_fx.append(fx_l)
        flows_fy.append(fy_l)

    fx_stack = np.stack(flows_fx, axis=0)
    fy_stack = np.stack(flows_fy, axis=0)

    # median across scales: robust to occasional bad level
    fx = np.median(fx_stack, axis=0)
    fy = np.median(fy_stack, axis=0)
    return fx, fy


def _log_blob_points(prev_s_gray: np.ndarray,
                     max_kp: int = 40,
                     p_hi: float = 92.0) -> Optional[np.ndarray]:
    """
    Detect high-structure 'blob' points using multi-scale IG-LoG energy.
    Returns Nx1x2 float32 in (x,y) ROI-scale coords, or None if not enough.
    """
    Hs, Ws = prev_s_gray.shape[:2]
    if Hs < 6 or Ws < 6:
        return None

    # reuse IG-LoG as a generic blob energy map
    pf = prev_s_gray.astype(np.float32) / 255.0
    E = _iglog_energy(pf)          # already multi-scale + abs

    # threshold by high percentile
    thr = float(np.percentile(E, p_hi))
    if not np.isfinite(thr) or thr <= 0.0:
        return None

    # simple 3x3 non-max suppression
    E_u8 = np.clip((E / (thr + 1e-9)) * 255.0, 0, 255).astype(np.uint8)
    dil = cv.dilate(E_u8, np.ones((3, 3), np.uint8))
    peaks = (E_u8 == dil) & (E > thr)

    ys, xs = np.where(peaks)
    if xs.size < 6:
        return None

    # sort by energy, keep top K
    vals = E[ys, xs]
    order = np.argsort(vals)[::-1][:max_kp]
    xs = xs[order]; ys = ys[order]

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    return pts.reshape(-1, 1, 2)


def _lk_flow_from_blobs(prev_s_gray: np.ndarray,
                        curr_s_gray: np.ndarray,
                        max_kp: int = 40) -> Tuple[float, float, bool]:
    """
    Sparse LK on LoG blobs, in *scaled ROI pixels per frame*.
    Returns (vx_lk, vy_lk, ok_flag).
    """
    pts_prev = _log_blob_points(prev_s_gray, max_kp=max_kp)
    if pts_prev is None:
        return 0.0, 0.0, False

    pts_next, status, err = cv.calcOpticalFlowPyrLK(
        prev_s_gray, curr_s_gray,
        pts_prev, None,
        winSize=(19, 19),
        maxLevel=3,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    if pts_next is None:
        return 0.0, 0.0, False

    status = status.ravel().astype(bool)

    if err is None:
        # no error metric, just trust status
        good = status
    else:
        err = err.ravel()

        # candidate indices that are even eligible
        cand = status & np.isfinite(err)

        if not np.any(cand):
            # no usable tracks at all → bail out cleanly
            return 0.0, 0.0, False

        # only compute percentile on the *non-empty* valid subset
        valid_err = err[cand]

        if valid_err.size == 0:
            return 0.0, 0.0, False

        thr = np.percentile(valid_err, 80)

        # “good” tracks = finite, under threshold, and originally marked valid
        good = cand & (err <= thr)

    # require at least a small cluster of good tracks
    if np.count_nonzero(good) < 4:
        return 0.0, 0.0, False


    p0 = pts_prev[good, 0, :]   # (N,2)
    p1 = pts_next[good, 0, :]
    dx = p1[:, 0] - p0[:, 0]
    dy = p1[:, 1] - p0[:, 1]

    # robust median translation
    vx_lk = float(np.median(dx))
    vy_lk = float(np.median(dy))

    # reject nonsense “zero” or nan
    if not np.isfinite(vx_lk) or not np.isfinite(vy_lk):
        return 0.0, 0.0, False

    # if literally no motion, let caller decide
    if math.hypot(vx_lk, vy_lk) < 1e-3:
        return vx_lk, vy_lk, False

    return vx_lk, vy_lk, True


# --- HG_RESCUE helpers -----------------------------------------------------------
def _hg_clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _hg_phasecorr_u8(a_u8: np.ndarray, b_u8: np.ndarray, downscale: float = 1.0) -> Tuple[float, float, float]:
    """
    Phase correlation on uint8 maps. Returns (dx, dy, response) in *input pixel units*.
    Uses a Hanning window to reduce edge wrap artifacts.
    """
    if a_u8 is None or b_u8 is None or a_u8.size == 0 or b_u8.size == 0:
        return 0.0, 0.0, 0.0

    a = a_u8
    b = b_u8
    sc = float(downscale)
    if sc is not None and sc > 0.0 and sc < 1.0:
        H, W = a.shape[:2]
        W2 = max(8, int(round(W * sc)))
        H2 = max(8, int(round(H * sc)))
        a = cv.resize(a, (W2, H2), interpolation=cv.INTER_AREA)
        b = cv.resize(b, (W2, H2), interpolation=cv.INTER_AREA)
    else:
        sc = 1.0

    a32 = a.astype(np.float32) / 255.0
    b32 = b.astype(np.float32) / 255.0
    H2, W2 = a32.shape[:2]

    # Hanning window for stability on tiny ROIs / high contrast edges
    try:
        win = cv.createHanningWindow((W2, H2), cv.CV_32F)
    except Exception:
        win = None

    try:
        (dx, dy), resp = cv.phaseCorrelate(a32, b32, win)
    except Exception:
        return 0.0, 0.0, 0.0

    dx = float(dx) / sc
    dy = float(dy) / sc
    resp = float(resp) if resp is not None else 0.0
    return dx, dy, resp


def _hg_binary_from_struct_u8(
    M_u8: np.ndarray,
    pctl: float,
    morph_open: int = 1,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """Return (bw_u8, thr, density).

    bw_u8 is 0/1 uint8.

    If valid_mask is provided, thresholding + density are computed ONLY over valid pixels,
    and bw is forced to 0 outside the mask.
    """
    if M_u8 is None or M_u8.size == 0:
        return np.zeros((1, 1), np.uint8), 0.0, 0.0

    vm = None
    if valid_mask is not None:
        try:
            vm = valid_mask.astype(bool)
            if vm.shape != M_u8.shape:
                vm = None
        except Exception:
            vm = None

    # Percentile threshold: avoid mask-induced 0-percentile collapse.
    try:
        src = M_u8[vm] if vm is not None else M_u8
        if src.size == 0:
            return np.zeros_like(M_u8, np.uint8), 0.0, 0.0
        thr = float(np.percentile(src, float(pctl)))
    except Exception:
        thr = 0.0

    bw = (M_u8 >= thr)
    if vm is not None:
        bw &= vm
    bw = bw.astype(np.uint8)

    if morph_open and int(morph_open) > 0:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * int(morph_open) + 1, 2 * int(morph_open) + 1))
        bw = cv.morphologyEx(bw, cv.MORPH_OPEN, k, iterations=1)
        if vm is not None:
            bw = (bw.astype(bool) & vm).astype(np.uint8)

    if vm is not None:
        denom = float(np.count_nonzero(vm))
        dens = float(np.count_nonzero(bw)) / denom if denom > 0 else 0.0
    else:
        dens = float(np.mean(bw > 0))

    return bw, float(thr), float(dens)


def _hg_cluster_ok(stats: np.ndarray, lbl: int, roi_area: int) -> bool:
    area = int(stats[lbl, cv.CC_STAT_AREA])
    if area < int(HG_MIN_CLUSTER_PX):
        return False
    if area > int(float(roi_area) * 0.75):  # huge = probably whole ROI noise / wash
        return False
    w = int(stats[lbl, cv.CC_STAT_WIDTH])
    h = int(stats[lbl, cv.CC_STAT_HEIGHT])
    if w <= 0 or h <= 0:
        return False
    fill = float(area) / float(w * h)
    if fill < 0.10:  # too stringy / sparse
        return False
    aspect = max(float(w) / float(h), float(h) / float(w))
    if aspect > 12.0:
        return False
    return True


def _hg_sample_points(labels: np.ndarray, lbl: int, max_pts: int) -> np.ndarray:
    ys, xs = np.where(labels == int(lbl))
    if xs.size == 0:
        return np.zeros((0, 2), np.float32)

    # deterministic stride sampling (fast + stable)
    N = int(xs.size)
    step = max(1, int(round(N / max(1, int(max_pts)))))
    idx = np.arange(0, N, step, dtype=np.int32)[:int(max_pts)]
    pts = np.stack([xs[idx].astype(np.float32), ys[idx].astype(np.float32)], axis=1)
    return pts


def _hg_rescue_translation(prev_s_gray: np.ndarray, curr_s_gray: np.ndarray, roi: Optional["ROI"] = None) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Hypergraph-ish translation rescue on a scaled ROI patch.
    Returns (vx, vy, conf, dbg) in *scaled ROI px/frame*.
    """
    dbg: Dict[str, Any] = {}
    if prev_s_gray is None or curr_s_gray is None or prev_s_gray.size == 0 or curr_s_gray.size == 0:
        return 0.0, 0.0, 0.0, dbg

    Hs, Ws = prev_s_gray.shape[:2]
    roi_area = int(Hs * Ws)
    # --- NEW: local-only outline trackinSg (moves roi.ai_outline_polys_norm each frame) ---
    if roi is not None and bool(getattr(roi, "ai_outline_enabled", False)):
        try:
            # Track the outline BEFORE _roi_ai_outline_mask() is built below.
            # Uses the same ROI-scaled patches HG already operates on.
            _track_outline_polygons_local(
                roi=roi,
                prev_gray_u8=prev_s_gray,
                curr_gray_u8=curr_s_gray,
                Hs=Hs,
                Ws=Ws,
                dbg=dbg,   # dbg already exists at top of function
            )
        except Exception:
            pass
    # -------------------------------------------------------------------------------


    # state: adaptive structure percentile (keeps structure density in-band)
    pctl = float(HG_STRUCT_PCTL_DEF)
    if roi is not None:
        st = roi.__dict__.setdefault("_hg_state", {"pctl": float(HG_STRUCT_PCTL_DEF)})
        try:
            pctl = float(st.get("pctl", pctl))
        except Exception:
            pctl = float(HG_STRUCT_PCTL_DEF)

    # Optional outline/occlusion mask (polygon in ROI crop space).
    # The same polygon set can be used two ways:
    #   - ai_outline_mode="include": keep ONLY pixels inside polygon
    #   - ai_outline_mode="exclude": treat pixels inside polygon as occluders (ignore them)
    poly_mask = None           # True inside polygon(s)
    valid_mask = None          # True where HG is allowed to look
    mask_mode = "include"
    mask_active = False
    if roi is not None and bool(getattr(roi, "ai_outline_enabled", False)):
        try:
            poly_mask = _roi_ai_outline_mask(roi, (Hs, Ws))
            if poly_mask is not None:
                mask_mode = str(getattr(roi, "ai_outline_mode", "include") or "include").lower()
                if mask_mode.startswith("exc"):
                    valid_mask = (~poly_mask)
                else:
                    valid_mask = poly_mask
                # sanity: require at least a few usable pixels
                if int(np.count_nonzero(valid_mask)) >= 16:
                    mask_active = True
        except Exception:
            poly_mask = None
            valid_mask = None
            mask_active = False

    # If we're excluding an occluder, optionally inpaint it out BEFORE IG-LoG/GSCM
    # so occluder edges don't leak into structure derivation.
    prev_in = prev_s_gray
    curr_in = curr_s_gray
    if mask_active and poly_mask is not None and mask_mode.startswith("exc"):
        try:
            excl_frac = float(np.count_nonzero(poly_mask)) / float(max(1, roi_area))
            do_inpaint = bool(getattr(roi, "ai_outline_inpaint", True))
            if do_inpaint and 0.0 < excl_frac < 0.65:
                occ_u8 = (poly_mask.astype(np.uint8) * 255)
                rad = int(getattr(roi, "ai_outline_inpaint_r", 3) or 3)
                rad = int(max(1, min(8, rad)))
                prev_in = cv.inpaint(prev_s_gray, occ_u8, float(rad), cv.INPAINT_TELEA)
                curr_in = cv.inpaint(curr_s_gray, occ_u8, float(rad), cv.INPAINT_TELEA)
                dbg["mask_inpaint"] = True
            else:
                dbg["mask_inpaint"] = False
        except Exception:
            dbg["mask_inpaint"] = False

    # Structure maps (GSCM hybrid preferred inside struct_M_from_gray_u8)
    try:
        stM = roi.__dict__.setdefault("_iglog_state_hg", {}) if roi is not None else {}
        M0 = struct_M_from_gray_u8(prev_in, stM)
        M1 = struct_M_from_gray_u8(curr_in, stM)
    except Exception:
        return 0.0, 0.0, 0.0, dbg

    # Apply mask at the structure-map level too (hard ignore). This prevents a single
    # occluder pixel from dominating cluster selection.
    M0m = M0
    M1m = M1
    if mask_active and valid_mask is not None:
        try:
            M0m = M0.copy()
            M1m = M1.copy()
            M0m[~valid_mask] = 0
            M1m[~valid_mask] = 0
        except Exception:
            M0m = M0
            M1m = M1

    # Coarse shift via phase correlation.
    # Prefer masked PC when it's confident; fall back to unmasked PC when masking nukes signal.
    dx_pc, dy_pc, pc_resp = _hg_phasecorr_u8(M0m, M1m, downscale=float(HG_PC_DOWNSCALE))
    if mask_active:
        try:
            dx_full, dy_full, resp_full = _hg_phasecorr_u8(M0, M1, downscale=float(HG_PC_DOWNSCALE))
            dbg.update({"pc_resp_mask": float(pc_resp), "pc_resp_full": float(resp_full)})
            # If masked response is weak but unmasked is strong, use unmasked as the initial shift.
            if float(pc_resp) < float(HG_PC_MIN_RESP) and float(resp_full) >= float(HG_PC_MIN_RESP):
                dx_pc, dy_pc, pc_resp = float(dx_full), float(dy_full), float(resp_full)
        except Exception:
            pass

    dbg.update({
        "dx_pc": float(dx_pc),
        "dy_pc": float(dy_pc),
        "pc_resp": float(pc_resp),
        "mask_active": bool(mask_active),
        "mask_mode": str(mask_mode),
        "mask_valid_frac": float(np.count_nonzero(valid_mask)) / float(max(1, roi_area)) if (mask_active and valid_mask is not None) else 1.0,
    })

    # Binary structure mask (prev) + adaptive density control
    bw0, thr0, dens0 = _hg_binary_from_struct_u8(M0m, pctl=pctl, morph_open=int(HG_MORPH_OPEN), valid_mask=valid_mask)
    dbg.update({"thr0": float(thr0), "density": float(dens0), "pctl": float(pctl)})

    # adapt pctl (only if roi provided, so it's stable per ROI)
    if roi is not None:
        try:
            if dens0 > float(HG_DENSITY_HI):
                pctl += float(HG_PCTL_STEP)
            elif dens0 < float(HG_DENSITY_LO):
                pctl -= float(HG_PCTL_STEP)
            pctl = _hg_clamp(pctl, float(HG_PCTL_MIN), float(HG_PCTL_MAX))
            roi.__dict__["_hg_state"]["pctl"] = float(pctl)
        except Exception:
            pass

    if int(np.count_nonzero(bw0)) < int(HG_MIN_CLUSTER_PX):
        # no structure → only trust phaseCorr if it's confident
        if float(pc_resp) >= float(HG_PC_MIN_RESP):
            return float(dx_pc), float(dy_pc), _hg_clamp((float(pc_resp) - float(HG_PC_MIN_RESP)) / max(1e-9, (0.35 - float(HG_PC_MIN_RESP))), 0.0, 1.0), {**dbg, "method": "phasecorr"}
        return 0.0, 0.0, 0.0, {**dbg, "method": "skip"}

    # Connected components on structure
    try:
        nlab, labels, stats, cents = cv.connectedComponentsWithStats(bw0, connectivity=8)
    except Exception:
        nlab = 0
        labels = None
        stats = None

    if labels is None or stats is None or nlab <= 1:
        if float(pc_resp) >= float(HG_PC_MIN_RESP):
            return float(dx_pc), float(dy_pc), _hg_clamp((float(pc_resp) - float(HG_PC_MIN_RESP)) / max(1e-9, (0.35 - float(HG_PC_MIN_RESP))), 0.0, 1.0), {**dbg, "method": "phasecorr"}
        return 0.0, 0.0, 0.0, {**dbg, "method": "skip"}

    # Filter clusters
    valid = []
    for lbl in range(1, int(nlab)):
        if _hg_cluster_ok(stats, lbl, roi_area):
            valid.append(int(lbl))
    if len(valid) > int(HG_MAX_CLUSTERS):
        # keep largest clusters
        areas = np.array([int(stats[l, cv.CC_STAT_AREA]) for l in valid], dtype=np.int32)
        order = np.argsort(areas)[::-1][:int(HG_MAX_CLUSTERS)]
        valid = [valid[i] for i in order.tolist()]

    if not valid:
        if float(pc_resp) >= float(HG_PC_MIN_RESP):
            return float(dx_pc), float(dy_pc), _hg_clamp((float(pc_resp) - float(HG_PC_MIN_RESP)) / max(1e-9, (0.35 - float(HG_PC_MIN_RESP))), 0.0, 1.0), {**dbg, "method": "phasecorr"}
        return 0.0, 0.0, 0.0, {**dbg, "method": "skip"}

    # Anchor preference (if we have a confident anchor lock)
    anchor_lbl = -1
    if roi is not None and bool(getattr(roi, "_anchor_last_ok", False)):
        try:
            ax = int(round(float(getattr(roi, "anchor_u", 0.5)) * float(Ws)))
            ay = int(round(float(getattr(roi, "anchor_v", 0.5)) * float(Hs)))
            ax = int(max(0, min(ax, Ws - 1)))
            ay = int(max(0, min(ay, Hs - 1)))
            anchor_lbl = int(labels[ay, ax])
        except Exception:
            anchor_lbl = -1

    # Collect points across clusters (single LK call)
    pts0_list = []
    cid_list = []
    for cid, lbl in enumerate(valid):
        pts = _hg_sample_points(labels, lbl, int(HG_MAX_PTS_PER_CL))
        if pts.shape[0] < 6:
            continue
        pts0_list.append(pts)
        cid_list.append(np.full((pts.shape[0],), cid, np.int32))

    if not pts0_list:
        if float(pc_resp) >= float(HG_PC_MIN_RESP):
            return float(dx_pc), float(dy_pc), _hg_clamp((float(pc_resp) - float(HG_PC_MIN_RESP)) / max(1e-9, (0.35 - float(HG_PC_MIN_RESP))), 0.0, 1.0), {**dbg, "method": "phasecorr"}
        return 0.0, 0.0, 0.0, {**dbg, "method": "skip"}

    pts0 = np.concatenate(pts0_list, axis=0)
    cids = np.concatenate(cid_list, axis=0)

    p0 = pts0.reshape(-1, 1, 2).astype(np.float32)
    p1_init = (pts0 + np.array([dx_pc, dy_pc], np.float32)).reshape(-1, 1, 2).astype(np.float32)

    win = int(max(2, int(HG_LK_WIN_RADIUS)))
    lk_params = dict(
        winSize=(2 * win + 1, 2 * win + 1),
        maxLevel=int(HG_LK_MAX_LEVEL),
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 0.03),
        flags=cv.OPTFLOW_USE_INITIAL_FLOW
    )

    try:
        p1, st, err = cv.calcOpticalFlowPyrLK(M0m, M1m, p0, p1_init, **lk_params)
    except Exception:
        p1 = None

    if p1 is None or st is None:
        if float(pc_resp) >= float(HG_PC_MIN_RESP):
            return float(dx_pc), float(dy_pc), _hg_clamp((float(pc_resp) - float(HG_PC_MIN_RESP)) / max(1e-9, (0.35 - float(HG_PC_MIN_RESP))), 0.0, 1.0), {**dbg, "method": "phasecorr"}
        return 0.0, 0.0, 0.0, {**dbg, "method": "skip"}

    st = st.reshape(-1).astype(bool)
    if err is not None:
        err = err.reshape(-1).astype(np.float32)
    else:
        err = np.full((st.size,), 0.0, np.float32)

    # Global err gate (percentile on valid tracks)
    cand = st & np.isfinite(err)
    if not np.any(cand):
        if float(pc_resp) >= float(HG_PC_MIN_RESP):
            return float(dx_pc), float(dy_pc), _hg_clamp((float(pc_resp) - float(HG_PC_MIN_RESP)) / max(1e-9, (0.35 - float(HG_PC_MIN_RESP))), 0.0, 1.0), {**dbg, "method": "phasecorr"}
        return 0.0, 0.0, 0.0, {**dbg, "method": "skip"}

    thr_err = float(np.percentile(err[cand], 80.0))
    good = cand & (err <= thr_err)

    # Per-cluster robust translation + score
    best = {"score": -1.0, "vx": 0.0, "vy": 0.0, "cid": -1, "n_inl": 0, "n_tot": 0}
    chosen_anchor = False

    p0g = p0[:, 0, :]
    p1g = p1[:, 0, :]
    dx_all = (p1g[:, 0] - p0g[:, 0]).astype(np.float32)
    dy_all = (p1g[:, 1] - p0g[:, 1]).astype(np.float32)

    for cid in range(len(valid)):
        sel = (cids == int(cid)) & good
        n_tot = int(np.count_nonzero(cids == int(cid)))
        if int(np.count_nonzero(sel)) < int(HG_MIN_INLIERS):
            continue

        dx = dx_all[sel]
        dy = dy_all[sel]

        vx = float(np.median(dx))
        vy = float(np.median(dy))

        res = np.sqrt((dx - vx) ** 2 + (dy - vy) ** 2)
        inl = res <= float(HG_INLIER_R)
        n_inl = int(np.count_nonzero(inl))
        if n_inl < int(HG_MIN_INLIERS):
            continue

        inlier_ratio = float(n_inl) / float(res.size)
        score = inlier_ratio  # simple + stable; 0..1

        if score > float(best["score"]):
            best = {"score": float(score), "vx": float(vx), "vy": float(vy), "cid": int(cid), "n_inl": int(n_inl), "n_tot": int(n_tot)}

    # If anchor label is valid, force-pick its cluster when available (better semantics).
    if anchor_lbl > 0 and anchor_lbl in valid and best["cid"] >= 0:
        try:
            cid_anchor = int(valid.index(int(anchor_lbl)))
            # Only override if the anchor cluster is actually viable.
            sel_a = (cids == int(cid_anchor)) & good
            if int(np.count_nonzero(sel_a)) >= int(HG_MIN_INLIERS):
                dx = dx_all[sel_a]; dy = dy_all[sel_a]
                vx = float(np.median(dx)); vy = float(np.median(dy))
                res = np.sqrt((dx - vx) ** 2 + (dy - vy) ** 2)
                n_inl = int(np.count_nonzero(res <= float(HG_INLIER_R)))
                inlier_ratio = float(n_inl) / float(res.size)
                score = float(inlier_ratio)
                if n_inl >= int(HG_MIN_INLIERS) and score >= float(HG_MIN_SCORE):
                    best = {"score": float(score), "vx": float(vx), "vy": float(vy), "cid": int(cid_anchor), "n_inl": int(n_inl), "n_tot": int(np.count_nonzero(cids == int(cid_anchor)))}
                    chosen_anchor = True
        except Exception:
            pass

    best_score = float(best.get("score", -1.0))
    ok_hg = (best_score >= float(HG_MIN_SCORE)) and (int(best.get("n_inl", 0)) >= int(HG_MIN_INLIERS))

    # Confidence blend: phaseCorr gives continuity; HG corrects structure motion
    conf_pc = _hg_clamp((float(pc_resp) - float(HG_PC_MIN_RESP)) / max(1e-9, (0.35 - float(HG_PC_MIN_RESP))), 0.0, 1.0)
    conf_hg = _hg_clamp((best_score - float(HG_MIN_SCORE)) / max(1e-9, (0.70 - float(HG_MIN_SCORE))), 0.0, 1.0) if ok_hg else 0.0

    if ok_hg:
        alpha = conf_hg
        vx = (1.0 - alpha) * float(dx_pc) + alpha * float(best["vx"])
        vy = (1.0 - alpha) * float(dy_pc) + alpha * float(best["vy"])
        conf = max(conf_pc, conf_hg)
        method = "hypergraph" if not chosen_anchor else "hypergraph_anchor"
    else:
        if float(pc_resp) >= float(HG_PC_MIN_RESP):
            vx, vy = float(dx_pc), float(dy_pc)
            conf = conf_pc
            method = "phasecorr"
        else:
            vx, vy = 0.0, 0.0
            conf = 0.0
            method = "skip"

    dbg.update({
        "method": method,
        "ok_hg": bool(ok_hg),
        "best_score": float(best_score),
        "best_vx": float(best.get("vx", 0.0)),
        "best_vy": float(best.get("vy", 0.0)),
        "best_n_inl": int(best.get("n_inl", 0)),
        "anchor_lbl": int(anchor_lbl),
    })

    return float(vx), float(vy), float(conf), dbg

# --- END HG_RESCUE helpers -------------------------------------------------------


def _vz25_from_curvature_and_flow(curr_s_gray: np.ndarray,
                                  fx: np.ndarray,
                                  fy: np.ndarray,
                                  mask: Optional[np.ndarray] = None) -> float:
    """
    2.5D "bulge" estimator:
      - uses multi-scale IG-LoG curvature as a shape energy field
      - multiplies by radial flow to get outward/inward depth-like motion
      - pools over the same mask used for vx,vy medians.

    Returns vz25 in ROI-scale units (per-frame; caller rescales by 1/SCALE).
    """
    Hs, Ws = curr_s_gray.shape[:2]
    if Hs < 4 or Ws < 4 or fx.shape[:2] != (Hs, Ws) or fy.shape[:2] != (Hs, Ws):
        return 0.0

    # 1) curvature magnitude from IG-LoG (same guts as anchor descriptor)
    pf = curr_s_gray.astype(np.float32) / 255.0
    curv = _iglog_energy(pf)  # already multi-scale + abs
    if not np.any(curv > 1e-8):
        return 0.0

    # robust 0..1 normalize curvature
    lo = float(np.percentile(curv, 5.0))
    hi = float(np.percentile(curv, 95.0))
    rng = max(hi - lo, 1e-9)
    curv01 = np.clip((curv - lo) / rng, 0.0, 1.0)

    # 2) radial flow relative to ROI center (in ROI-scale coords)
    yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
    cx = (Ws - 1) * 0.5
    cy = (Hs - 1) * 0.5
    dx = xx - cx
    dy = yy - cy
    rad = np.sqrt(dx*dx + dy*dy) + 1e-6
    ux = dx / rad
    uy = dy / rad

    # projection of flow along radial direction: outward (+) / inward (-)
    v_rad = fx * ux + fy * uy

    # 3) gate by motion magnitude so static curvature doesn't fake depth
    vmag = np.sqrt(fx*fx + fy*fy)
    m95 = float(np.percentile(vmag, 95.0)) or 1e-9
    mag01 = np.clip(vmag / m95, 0.0, 1.0)

    # 4) curvature-weighted radial motion → 2.5D depth proxy
    # sign from v_rad, strength from curv01 * |v_rad|
    vz_map = np.sign(v_rad) * curv01 * mag01

    if mask is not None:
        m = (mask.astype(bool) & (vmag > 1e-6))
        if np.count_nonzero(m) >= 8:
            return float(np.median(vz_map[m]))

    # fallback: use all active motion
    m = (vmag > 1e-6)
    if np.count_nonzero(m) < 8:
        return 0.0
    return float(np.median(vz_map[m]))


def _anchor_desc_from_patch_iglog(patch_u8: np.ndarray) -> np.ndarray:
    """
    More robust descriptor:
      - coarse intensity (blurred)
      - coarse gradient magnitude
      - coarse IG-LoG energy
    All pooled to a small grid before flattening, then zero-mean + L2.
    """
    if patch_u8.ndim != 2:
        patch_u8 = cv.cvtColor(patch_u8, cv.COLOR_BGR2GRAY)
    pf = patch_u8.astype(np.float32) / 255.0

    # pre-blur to kill tiny pixelwise noise
    pf_s = cv.GaussianBlur(pf, (0, 0), 1.2)

    # coarse intensity / contrast-ish channel
    inten = pf_s

    # gradient magnitude (coarse)
    gx = cv.Sobel(pf_s, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(pf_s, cv.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    # IG-LoG energy, but smoothed by pre-blur and then pooled
    iglog = _iglog_energy(pf_s)

    # pool all three to a SMALLER grid to reduce sensitivity to shifts
    GRID = ANCHOR_PATCH_GRID  # e.g. 32; you can even drop this to 24 or 16
    inten_g = cv.resize(inten, (GRID, GRID), interpolation=cv.INTER_AREA)
    grad_g  = cv.resize(grad,  (GRID, GRID), interpolation=cv.INTER_AREA)
    iglog_g = cv.resize(iglog, (GRID, GRID), interpolation=cv.INTER_AREA)

    # --- NEW: radial weighting so center dominates, edges de-emphasized ---
    radial_w = ANCHOR_RADIAL_W
    inten_g *= radial_w
    grad_g  *= radial_w
    iglog_g *= radial_w
    # ---------------------------------------------------------------------

    # --- NEW: channel re-weighting: care less about raw intensity, more about structure ---
    w_inten = 0.4
    w_grad  = 1.0
    w_iglog = 1.4
    feats = np.stack([
        w_inten * inten_g,
        w_grad  * grad_g,
        w_iglog * iglog_g
    ], axis=2)
    # ---------------------------------------------------------------------

    v = feats.reshape(-1).astype(np.float32)

    v -= float(v.mean())
    nrm = float(np.linalg.norm(v)) + 1e-9
    v /= nrm
    return v


MAX_LOCAL_SEARCH_PX = 14        # max local search radius (scaled ROI px)
ANCHOR_LOCK_THRESH   = 0.95     # "very sure"
ANCHOR_MASK_THRESH   = 0.90     # "medium sure"


def _anchor_update_and_mask(roi: "ROI", curr_s_gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Unified anchor step with local recovery:
      - If descriptor not initialized, build it from the current center patch and LOCK.
      - Otherwise, search a radius that grows with LOST count.
      - Update anchor_u/anchor_v and mask when similarity is high enough.
      - Maintain _anchor_lost_count for global reacquisition.
    """
    Hs, Ws = curr_s_gray.shape[:2]
    if Hs < 4 or Ws < 4:
        roi._anchor_ready = False
        roi._anchor_last_ok = False
        roi._anchor_last_sim = 0.0
        roi._anchor_lost_count = 0
        return None
    

    # NEW: require an explicit user anchor before doing anything
    if not getattr(roi, "anchor_user_set", False):
        roi._anchor_ready = False
        roi._anchor_last_ok = False
        roi._anchor_last_sim = 0.0
        roi._anchor_lost_count = 0
        return None
    
    # previous anchor center in scaled ROI coords
    u0 = float(getattr(roi, "anchor_u", 0.5))
    v0 = float(getattr(roi, "anchor_v", 0.5))
    frac = float(getattr(roi, "anchor_size_frac", 0.45))

    cx0 = float(np.clip(u0 * Ws, 1.0, Ws - 2.0))
    cy0 = float(np.clip(v0 * Hs, 1.0, Hs - 2.0))
    rad = int(max(4, 0.5 * frac * min(Hs, Ws)))

    def crop_patch_at(cx_f: float, cy_f: float):
        cx = int(round(cx_f))
        cy = int(round(cy_f))
        x0 = max(0, cx - rad); x1 = min(Ws, cx + rad)
        y0 = max(0, cy - rad); y1 = min(Hs, cy + rad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return curr_s_gray[y0:y1, x0:x1], (cx, cy, x0, y0, x1, y1)

    # center patch
    center = crop_patch_at(cx0, cy0)
    if center is None:
        roi._anchor_ready = False
        roi._anchor_last_ok = False
        roi._anchor_last_sim = 0.0
        roi._anchor_lost_count = 0
        return None

    patch_center, (cx_c, cy_c, x0_c, y0_c, x1_c, y1_c) = center
    patch_center_grid = cv.resize(patch_center, (ANCHOR_PATCH_GRID, ANCHOR_PATCH_GRID),
                                  interpolation=cv.INTER_AREA)

    # --- initialize descriptor on first use ---
    if not getattr(roi, "_anchor_ready", False) or getattr(roi, "_anchor_desc", None) is None:
        roi._anchor_desc = _anchor_desc_from_patch_iglog(patch_center_grid)
        roi._anchor_ready = True
        roi._anchor_last_ok = True
        roi._anchor_last_sim = 1.0
        roi._anchor_lost_count = 0
        roi.anchor_u = float(cx_c / max(1, Ws))
        roi.anchor_v = float(cy_c / max(1, Hs))
        roi._anchor_template = patch_center.copy()  # store full-res template for global reacquire
        if getattr(roi, "_anchor_template0", None) is None:
            roi._anchor_template0 = roi._anchor_template.copy()


        mask = np.zeros((Hs, Ws), bool)
        mask[y0_c:y1_c, x0_c:x1_c] = True
        return mask

    # --- local search radius grows with LOST count ---
    lost_count = int(getattr(roi, "_anchor_lost_count", 0))
    last_ok    = bool(getattr(roi, "_anchor_last_ok", False))

    if last_ok:
        search_r = ANCHOR_SEARCH_PX        # e.g. 3
        lost_count = 0
    else:
        lost_count += 1
        search_r = min(ANCHOR_SEARCH_PX + 2 * lost_count, MAX_LOCAL_SEARCH_PX)

    roi._anchor_lost_count = lost_count

    desc_ref = roi._anchor_desc.astype(np.float32)
    best_score = -1.0
    best_sim   = -1.0
    best_cx, best_cy = cx_c, cy_c
    best_bounds = (x0_c, y0_c, x1_c, y1_c)

    for dy in range(-search_r, search_r + 1):
        for dx in range(-search_r, search_r + 1):
            patch_and_info = crop_patch_at(cx0 + dx, cy0 + dy)
            if patch_and_info is None:
                continue
            patch, (cx_i, cy_i, x0_i, y0_i, x1_i, y1_i) = patch_and_info
            patch_grid = cv.resize(patch, (ANCHOR_PATCH_GRID, ANCHOR_PATCH_GRID),
                                   interpolation=cv.INTER_AREA)

            # --- NEW: reject ultra-flat candidates (no texture) ---
            if patch_grid.std() < 4.0:   # 0..255 domain; tune 3–8
                continue
            # -----------------------------------------------------

            cand = _anchor_desc_from_patch_iglog(patch_grid)

            sim = float(np.dot(desc_ref, cand))

            # distance penalty to prefer staying close
            dist = math.hypot(cx_i - cx0, cy_i - cy0)
            score = sim - 0.02 * (dist / max(1.0, rad))

            if score > best_score:
                best_score = score
                best_sim = sim
                best_cx, best_cy = cx_i, cy_i
                best_bounds = (x0_i, y0_i, x1_i, y1_i)

    roi._anchor_last_sim = float(best_sim)
     # --- NEW: expensive full-ROI reacquisition for user anchors ---
    user_anchor = bool(getattr(roi, "anchor_user_set", False))
    if best_sim < ANCHOR_LOCK_THRESH and user_anchor:
        tmpl = getattr(roi, "_anchor_template", None)
        if tmpl is not None and tmpl.size > 0 \
           and tmpl.shape[0] <= Hs and tmpl.shape[1] <= Ws:
            try:
                # scan the entire scaled ROI image
                res = cv.matchTemplate(curr_s_gray, tmpl, cv.TM_CCORR_NORMED)
                _, maxVal, _, maxLoc = cv.minMaxLoc(res)

                x0_b = int(maxLoc[0])
                y0_b = int(maxLoc[1])
                x1_b = x0_b + tmpl.shape[1]
                y1_b = y0_b + tmpl.shape[0]

                if x1_b - x0_b >= 4 and y1_b - y0_b >= 4:
                    patch = curr_s_gray[y0_b:y1_b, x0_b:x1_b]
                    patch_grid = cv.resize(
                        patch,
                        (ANCHOR_PATCH_GRID, ANCHOR_PATCH_GRID),
                        interpolation=cv.INTER_AREA
                    )

                    # recompute IG-LoG descriptor at the matched spot
                    cand = _anchor_desc_from_patch_iglog(patch_grid)
                    sim2 = float(np.dot(desc_ref, cand))

                    roi._anchor_desc = cand
                    roi._anchor_last_sim = sim2

                    if sim2 >= ANCHOR_LOCK_THRESH:
                        best_sim = sim2
                        best_cx = x0_b + 0.5 * (x1_b - x0_b)
                        best_cy = y0_b + 0.5 * (y1_b - y0_b)
                        best_bounds = (x0_b, y0_b, x1_b, y1_b)
            except Exception:
                # fail-safe: if anything blows up, just fall back to normal behavior
                pass
    # --- END NEW ---

    if best_sim < ANCHOR_LOCK_THRESH:
        roi._anchor_last_ok = False
        return None

    # LOCK
    roi._anchor_last_ok = True
    roi._anchor_lost_count = 0

    roi.anchor_u = float(best_cx / max(1, Ws))
    roi.anchor_v = float(best_cy / max(1, Hs))

    x0_b, y0_b, x1_b, y1_b = best_bounds
    mask = np.zeros((Hs, Ws), bool)
    mask[y0_b:y1_b, x0_b:x1_b] = True

    # refresh template every lock so it's not stale
    roi._anchor_template = curr_s_gray[y0_b:y1_b, x0_b:x1_b].copy()
    if getattr(roi, "_anchor_template0", None) is None:
        roi._anchor_template0 = roi._anchor_template.copy()    

    return mask


# --- NEW: tracked outline (local-only) -----------------------------------------
OUTLINE_TRACK_ENABLE_DEFAULT = True
OUTLINE_TRACK_MAX_STEP_PX    = 18.0   # clamp per-frame drift in ROI-scale px
OUTLINE_TRACK_MIN_RESP       = 0.10   # phaseCorr response gate
OUTLINE_TRACK_USE_SOFT       = True   # use soft edge field for stability

def _shift_polys_norm(polys, du, dv):
    # du/dv are normalized (0..1) shifts in polygon space
    out = []
    for poly in polys or []:
        if not poly or len(poly) < 3:
            continue
        pp = []
        for uv in poly:
            if not isinstance(uv, (list, tuple)) or len(uv) < 2:
                continue
            u = float(uv[0]) + float(du)
            v = float(uv[1]) + float(dv)
            pp.append([float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))])
        if len(pp) >= 3:
            out.append(pp)
    return out

def _track_outline_polygons_local(roi: "ROI",
                                 prev_gray_u8: np.ndarray,
                                 curr_gray_u8: np.ndarray,
                                 Hs: int, Ws: int,
                                 *,
                                 dbg: Optional[dict] = None) -> bool:
    """
    Track roi.ai_outline_polys_norm by estimating translation INSIDE the polygon only.
    Local-only: no global reacquire.
    Returns True if polygon updated.
    """
    if roi is None:
        return False
    if not bool(getattr(roi, "ai_outline_enabled", False)):
        return False
    if not bool(getattr(roi, "ai_outline_track", OUTLINE_TRACK_ENABLE_DEFAULT)):
        return False

    polys = getattr(roi, "ai_outline_polys_norm", None) or []
    if not polys:
        return False

    # build poly mask in ROI-scale patch space
    poly_mask = _roi_ai_outline_mask(roi, (Hs, Ws))
    if poly_mask is None or int(np.count_nonzero(poly_mask)) < 24:
        return False

    # structural maps
    stM = roi.__dict__.setdefault("_iglog_state_outline_track", {})
    M0 = struct_M_from_gray_u8(prev_gray_u8, stM)
    M1 = struct_M_from_gray_u8(curr_gray_u8, stM)

    # restrict to polygon
    A = M0.copy(); B = M1.copy()
    A[~poly_mask] = 0
    B[~poly_mask] = 0

    if OUTLINE_TRACK_USE_SOFT:
        A = _struct_flow_input_u8(A, mode="soft", blur_sigma=0.8)
        B = _struct_flow_input_u8(B, mode="soft", blur_sigma=0.8)

    dx, dy, resp = _hg_phasecorr_u8(A, B, downscale=float(HG_PC_DOWNSCALE))

    # gate + clamp
    ok = float(resp) >= float(OUTLINE_TRACK_MIN_RESP)
    if not ok:
        if dbg is not None:
            dbg["outline_track_ok"] = False
            dbg["outline_track_resp"] = float(resp)
        return False

    step = float(math.hypot(dx, dy))
    if step > float(OUTLINE_TRACK_MAX_STEP_PX):
        s = float(OUTLINE_TRACK_MAX_STEP_PX) / max(1e-6, step)
        dx *= s; dy *= s

    du = float(dx) / float(max(1, Ws))
    dv = float(dy) / float(max(1, Hs))

    roi.ai_outline_polys_norm = _shift_polys_norm(polys, du, dv)

    if dbg is not None:
        dbg["outline_track_ok"] = True
        dbg["outline_track_dx"] = float(dx)
        dbg["outline_track_dy"] = float(dy)
        dbg["outline_track_resp"] = float(resp)

    return True
# --- END NEW ------------------------------------------------------------------

def _cluster_mask_iou(labels_u16, lbl_id, poly_mask_bool):
    # IoU between a connected component and the polygon mask
    cc = (labels_u16 == int(lbl_id))
    inter = int(np.count_nonzero(cc & poly_mask_bool))
    if inter <= 0:
        return 0.0
    uni = int(np.count_nonzero(cc | poly_mask_bool))
    return float(inter) / float(max(1, uni))


def _roi_ai_outline_mask(roi: "ROI", patch_hw: Tuple[int,int]) -> Optional[np.ndarray]:
    """
    Build a boolean mask from roi.ai_outline_polys_norm (normalized polygons in 0..1).
    Returned mask is True for pixels inside the outline, False elsewhere.

    This is intentionally lightweight:
      - used to bias optical flow sampling (shape prior)
      - NOT a full segmentation system
    """
    polys = getattr(roi, "ai_outline_polys_norm", None)
    if not polys:
        return None
    try:
        h, w = int(patch_hw[0]), int(patch_hw[1])
        if h <= 1 or w <= 1:
            return None
        m = np.zeros((h, w), np.uint8)
        for poly in polys:
            if not poly or len(poly) < 3:
                continue
            pts = []
            for uv in poly:
                if not isinstance(uv, (list, tuple)) or len(uv) < 2:
                    continue
                u = float(uv[0]); v = float(uv[1])
                x = int(np.clip(u, 0.0, 1.0) * (w - 1))
                y = int(np.clip(v, 0.0, 1.0) * (h - 1))
                pts.append([x, y])
            if len(pts) >= 3:
                cv.fillPoly(m, [np.array(pts, dtype=np.int32)], 255)

        if int(m.max() or 0) == 0:
            return None

        dil = int(getattr(roi, "ai_outline_dilate_px", 0) or 0)
        if dil > 0:
            k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*dil + 1, 2*dil + 1))
            m = cv.dilate(m, k, iterations=1)
        return (m > 0)
    except Exception:
        return None


def clamp_rect(x, y, w, h, W, H):
    x = int(max(0, min(x, W - w))); y = int(max(0, min(y, H - h)))
    return (x, y, int(w), int(h))

def clamp_rect_within(bound: Optional[Tuple[int,int,int,int]], rc: Tuple[int,int,int,int], W: int, H: int):
    if bound is None: return clamp_rect(*rc, W, H)
    bx, by, bw, bh = map(int, bound)
    x, y, w, h = map(int, rc)
    w = min(w, bw); h = min(h, bh)
    x = max(bx, min(x, bx + bw - w))
    y = max(by, min(y, by + bh - h))
    return (x, y, w, h)

def draw_dashed(img, rc, color=(0,255,255), dash=8):
    if rc is None: return
    x,y,w,h = map(int, rc); step=dash
    for i in range(0,w,step*2): cv.line(img,(x+i,y),(x+i+step,y),color,1)
    for i in range(0,w,step*2): cv.line(img,(x+i,y+h),(x+i+step,y+h),color,1)
    for i in range(0,h,step*2): cv.line(img,(x,y+i),(x,y+i+step),color,1)
    for i in range(0,h,step*2): cv.line(img,(x+w,y+i),(x+w,y+i+step),color,1)

# ---------- ROI/Scene ----------
@dataclass
class ROI:
    rect: Tuple[int,int,int,int]
    name: Optional[str] = None
    bound: Optional[Tuple[int,int,int,int]] = None
    fb_levels: int = 5
    # --- TRACK MODE: per-ROI tracker selection ---
    # "fb"     = Farnebäck dense flow (default)
    # "hg"     = Hypergraph translation (phaseCorr + structure LK clusters; best for huge jumps)
    # "hybrid" = HG always for timing + impacts; FB runs only on calm frames (decimated) for vz/carrier
    motion_mode: str = "fb"


    flow_pyr_levels: int = 0  # 0 → use tracker.flow_pyr_levels; >0 → explicit
    dir_gate_deg: float = float('nan')   # NaN → auto (principal axis)
    dir_io_deg:   float = float('nan')   # NaN → follow dir_gate_deg
    io_in_sign:   int   = +1             # +1: toward io_dir = IN, -1: away = IN
    impact_lead_ms: int = 40             # place the impact before reversal (0..300)
    impact_pre_ms:  int = 20            # search window before reversal for score peak
    impact_post_ms:  int = 20            # search window after reversal for score peak

    dir_hit: Optional[Tuple[int,int,int,int]] = None  # widget hitbox
    flow_pad_px: int = 8               # pad around ROI for flow sampling
    impact_min_speed_ps: float = 20.0  # px/s gate; ignore micro‑motion below this
    # Last AOI-weighted speed in px/s (for AUTO Farneback pyramid)
    last_speed_px_s: float = 0.0


    # --- Z autoscale knobs (symmetric + elastic) ---
    z_scale_mode: str  = "off"   # "off"|"vz"
    z_deadband_rel_s: float = 0.006   # smaller → triggers more often
    z_scale_tau_ms:   int   = 120     # Z prefilter (ms)
    z_scale_gain:     float = 1.10    # symmetric gain (grow == shrink)
    z_scale_return_s: float = 0.75    # spring → return-to-baseline rate (s^-1)
    z_scale_max_rate_s: float = 2.00  # clamp |linear-size rate| (±200%/s)
    z_min_frac: float = 0.50          # size bounds vs. original
    z_max_frac: float = 2.00

    # OPTIONAL: slow baseline adaptation (“mode”) instead of hard original
    z_baseline_mode: str = "original" # "original"|"slow_median"
    z_baseline_tau_ms: int = 3000     # baseline LPF if slow_median
    # How to derive vz from flow:
    #   "div"    = pure divergence (old behavior)
    #   "curv"   = pure IG-LoG 2.5D curvature-depth
    #   "hybrid" = blend of both (recommended)
    vz_mode: str = "curv"
    vz25_mix: float = 0.75   # 0..1: how much to trust curvature vs divergence

    # runtime
    _z_log_s: float = 0.0
    _z_vfilt: float = 0.0
    _z_log_s0: float = 0.0  # baseline (log-space) if slow_median is used
    _w0: int = 0; _h0: int = 0


    axis_mode: str  = "off"   # "off" | "cos" | "cone"
    axis_gain: float = 1.60   # amplification along axis
    axis_perp_gain: float = 0.10  # leakage orthogonal to axis
    axis_half_deg: float = 15   # cone half-angle (deg)
    axis_power: float = 4.0       # |cosθ|^p smoothness
    axis_elev_deg: float = 0.0    # 3D pitch: +toward camera, -away
    axis_z_scale: float = 1.0     # unit match for vz vs vx,vy
    impact_axis_boost: float = 2.0  # score multiplier when AoI on
    impact_flip_deadband_ps: float = V_ZERO  # deadband for IN/OUT sign decisions (px/s-ish)


    # Half-angle of the lateral cone around yaw+90° (deg).
    # 0 < lat_half_deg < 90; wider cone = more tolerant sway.
    lat_half_deg: float = 70.0
    # Shape of lateral weighting: ((cosθ - cos(lat_half)) / (1-cos(lat_half))) ** lat_power
    lat_power: float    = 1.0


    


    refractory_ms: int = 140                   # wheel on 'Ref' to change (40..400)
    impact_thr_z: float = 2.0                  # MAD‑z threshold for impact score
    labelL_hit: Optional[Tuple[int,int,int,int]] = None  # (x,y,w,h) for 'L#' tag
    labelRef_hit: Optional[Tuple[int,int,int,int]] = None
    impact_mode: str = "hybrid"    # "hybrid" or "fast" or "smooth"
    impact_rt_every_flip: int = 1      # 1 = flash on each axis-jerk flip in RT HUD
    posz_tau_ms: int = 800  # lower = snappier Z; higher = steadier


    # Visual timing for impact flash
    impact_hold_ms: int = 300     # solid hold after a hit
    impact_fade_ms: int = 200     # additional fade‑out time
    _impact_fade_until: float = 0.0

    io_dir_deg: float = 0.0   # single 0..360° angle; IN = toward this arrow; OUT = opposite

    cmat_mode: str  = "off"   # "off" | "global" | "ring"
    cmat_alpha: float = 1  # fraction of background drift to subtract
    cmat_proj: str   = "full" # "full" | "orth" (orth = subtract only orthogonal to gate)
    cmat_tau_ms: int = 0    # LPF time constant for drift (ms)
    bg_ring_px: int  = 12     # ring thickness when cmat_mode="ring"
    # private filter state (per-ROI)
    _cmat_gx: float = 0.0
    _cmat_gy: float = 0.0
    _cmat_gz: float = 0.0

    # --- NEW: per-frame QA flag for velocity confidence ---
    _frame_lowconf: bool = False    


    vx_ps: float = 0.0
    vy_ps: float = 0.0
    vz_rel_s: float = 0.0
    last_center: Tuple[float,float] = (0.0,0.0)
    debug: bool = False
    _impact_flash_until: float = 0.0   # wall-clock seconds
    _impact_dir: int = 0               # +1 OUT, -1 IN, 0 none
    _last_impact_idx: int = -10**9     # last frame index impact fired
    last_speed: float = 0.0
    last_acc: float = 0.0
    last_jerk: float = 0.0
    anchor_u: float = 0.5              # relative X (0..1) inside ROI, default center
    anchor_v: float = 0.5              # relative Y (0..1) inside ROI
    anchor_size_frac: float = 0.45     # fraction of min(w,h) (0..1); patch radius ~ frac/2
    _anchor_desc: Optional[object] = None  # normalized descriptor (np.ndarray) or None
    _anchor_ready: bool = False        # set True once descriptor is initialized

     # --- NEW: anchor debug state ---
    _anchor_last_ok: bool = False      # whether last search passed similarity threshold
    _anchor_last_sim: float = 0.0      # last best similarity (cosine)
    # --- Anchor reacquisition state ---
    _anchor_lost_count: int = 0          # how many consecutive frames LOST
    _anchor_template: Optional[object] = None  # last good grayscale template (np.ndarray)
    # user explicitly defined an anchor region via Ctrl+Drag
    anchor_user_set: bool = False

    # --- AI Assist state (optional; only used if --ai is enabled) ---
    ai_tag_suggested: Optional[str] = None
    ai_tag_conf: float = 0.0

    # Optional outline prior (normalized polygons in ROI crop space)
    ai_outline_enabled: bool = False
    ai_outline_conf: float = 0.0
    ai_outline_polys_norm: List[List[List[float]]] = field(default_factory=list)  # [poly][pt][u,v] in 0..1
    ai_outline_dilate_px: int = 2  # dilate outline mask a bit to keep enough flow samples

    # How to interpret ai_outline_polys_norm when building masks for tracking:
    #   - "include": keep only pixels INSIDE polygon(s)
    #   - "exclude": treat pixels INSIDE polygon(s) as occluders (ignore them)
    ai_outline_mode: str = "include"

    # If excluding an occluder, optionally inpaint masked pixels out of the grayscale
    # before IG-LoG/GSCM structure derivation (reduces edge leakage).
    ai_outline_inpaint: bool = True
    ai_outline_inpaint_r: int = 3

    # Provenance (debug/UI only): "ai" | "iglog" | "manual"
    ai_outline_source: str = "ai"

    # Last occlusion solve (for transparency/debug)
    ai_last_occlusion_status: str = ""
    ai_last_bbox_xywh: Optional[Tuple[int,int,int,int]] = None
    # --- END AI Assist state ---


    # --- END NEW ---

def _aoi_weight(vx_ps: float, vy_ps: float, vz_rel_s: float, r: ROI):
    """Return axis‑weighted (vx,vy,vz). Leaves magnitudes stable off‑axis."""
    mode = str(getattr(r, "axis_mode", "off")).lower()
    if mode == "off":
        return vx_ps, vy_ps, vz_rel_s

    ax, ay, az = _axis_unit3d(getattr(r, "io_dir_deg", 0.0), getattr(r, "axis_elev_deg", 0.0))
    kz   = float(getattr(r, "axis_z_scale", 1.0))
    gpar = float(getattr(r, "axis_gain", 1.6))
    gper = float(getattr(r, "axis_perp_gain", 0.10))
    half = float(getattr(r, "axis_half_deg", 15.0))
    pexp = float(getattr(r, "axis_power", 4.0))

    # build 3D vectors in a matched unit space
    vx3, vy3, vz3 = vx_ps, vy_ps, kz * vz_rel_s
    vmag = math.sqrt(vx3*vx3 + vy3*vy3 + vz3*vz3) + 1e-12
    # cosine of angle to axis (abs → symmetric parallel/anti‑parallel)
    cos_t = abs((vx3*ax + vy3*ay + vz3*az) / vmag)
    if mode == "cone":
        w = 1.0 if cos_t >= math.cos(math.radians(half)) else 0.0
    else:  # "cos"
        w = cos_t ** pexp

    # anisotropic gain: v' = g⊥·v + (g∥−g⊥)·w·(proj onto axis)
    dot   = (vx3*ax + vy3*ay + vz3*az)
    vparx, vpary, vparz = dot*ax, dot*ay, dot*az
    vpx   = gper*vx3 + (gpar - gper)*w*vparx
    vpy   = gper*vy3 + (gpar - gper)*w*vpary
    vpz   = gper*vz3 + (gpar - gper)*w*vparz
    return vpx, vpy, vpz / (kz if kz != 0 else 1.0)


@dataclass
class Scene:
    start: int
    end: Optional[int] = None
    rois: List[ROI] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    # per-ROI series
    roi_cx: Dict[int, List[float]] = field(default_factory=dict)
    roi_cy: Dict[int, List[float]] = field(default_factory=dict)
    roi_vx: Dict[int, List[float]] = field(default_factory=dict)
    roi_vy: Dict[int, List[float]] = field(default_factory=dict)
    roi_vz: Dict[int, List[float]] = field(default_factory=dict)
    roi_env: Dict[int, List[float]] = field(default_factory=dict)
    roi_imp_in: Dict[int, List[float]] = field(default_factory=dict)
    roi_imp_out: Dict[int, List[float]] = field(default_factory=dict)
    # --- IG-LoG structural evidence (export sampling) ---
    roi_igE:  Dict[int, List[float]] = field(default_factory=dict)   # median IG-LoG energy in ROI
    roi_igdE: Dict[int, List[float]] = field(default_factory=dict)   # frame delta of that energy
    dup_flags: List[bool] = field(default_factory=list)
    roi_curv: Dict[int, List[float]] = field(default_factory=dict)
    # NEW: per-ROI low-confidence flags (parallel to roi_vx/vy/vz)
    roi_lowconf: Dict[int, List[bool]] = field(default_factory=dict)

    # --- AI Assist outputs (optional; only used if --ai is enabled) ---
    ai_impacts: List[Dict[str, Any]] = field(default_factory=list)
    ai_flux_fix: Dict[str, Any] = field(default_factory=dict)
    # --- END AI Assist outputs ---


    # --- Kinetic Label Engine: per-scene label metadata ---
    # label_primary[label] = roi index that is considered PRIMARY for that label
    label_primary: Dict[str, int] = field(default_factory=dict)
    # label_mode:
    #   "per_roi"   -> current behavior (per-ROI aliases)
    #   "per_label" -> label-centric aggregation + multi-role impacts
    label_mode: str = "per_roi"


    def in_range(self, i, N): return self.start <= i <= (self.end if self.end is not None else (N-1))

# ---------- Tracker (no mask) ----------
class DeformTracker:
    def __init__(self, W, H, fps):
        self.fps = float(fps); self.W = int(W); self.H = int(H)
        self.lagless=True; self.show_arrows=True
        self.debug_flow_mode = 0      # debug: 0 = off, 1 = WITH processing, 2 = NO processing
        self.debug_roi  = None       # (x, y, patch_bgr)
        # 0 = off, 1 = mild, 2 = strong
        self.edge_sharpen_mode = 0
        self.flow_pyr_levels = FLOW_PYR_LEVELS_DEFAULT
        # Gaussian pyramid supercharger:
        # 0 = OFF, 1 = ALWAYS, 2 = AUTO (based on last_speed_px_s)
        self.flow_pyr_mode = 0

        self.fb_consistency_mode = FLOW_FB_MODE_DEFAULT

        # --- GLOBAL MOTION CACHE ---
        # cache key = (id(prev_gray), id(curr_gray)) for which drift was computed
        self._gm_cache_key = None
        self._gm_cache_val = (0.0, 0.0, 0.0)
        self._gm_cache_ok  = False
        # optional background prefetch thread
        self._gm_thread    = None

        self._scaled_key   = None
        self._prev_scaled  = None
        self._curr_scaled  = None
        # --- IGLoG structural rescues ---
        self.struct_rescue_enable     = True
        self.struct_force_abs         = True        # ALWAYS use 0.70*ABSacc + 0.20*ZC + 0.10*EXT
        self.struct_mask_weight       = True        # use ZC|EXT as a sampling mask for medians
        self.struct_phasecorr_rescue  = True
        self.struct_farneback_rescue  = False       # optional: slow; only enable if you want to test
        self.struct_flow_img_mode     = "soft"      # "raw"|"blur"|"soft"
        self.struct_flow_blur_sigma   = 0.8
        self.struct_mask_min_px       = 48
        self.struct_score_low         = 0.0008

        # lowconf heuristics (robust, triggers on "wrong-but-confident" fields)
        self.lowconf_min_support      = 24
        self.lowconf_flat_m95         = 0.06
        self.lowconf_coh_ratio        = 1.2
        self.lowconf_inlier_frac      = 0.55
        self.lowconf_fb_bad_frac      = 0.45
        self.lowconf_anchor_clears    = False       # old behavior was effectively True
        self.debug_lowconf_print      = False
        # --- FULL-FRAME FLOW CACHE (per framepair, per fb_levels, per pyr_levels) ---
        # Keyed so multiple ROIs with same fb_levels reuse ONE Farneback run.
        self._fflow_key = None  # (pid,cid, preprocess_sig)
        self._fflow_cache = {}  # (fb_levels, pyr_levels) -> (fx_full, fy_full)
        self._fflow_prev_proc = None
        self._fflow_curr_proc = None
        self._fflow_preprocess_sig = None
        self.cmat_profile = "off"   # off | global
        self.cmat_max_h   = 180     # runtime cap for CMAT downscale height


    def prepare_cmat_frames(self, prev_gray, curr_gray):
        H, W = prev_gray.shape[:2]
        max_h = int(getattr(self, "cmat_max_h", 180))   # <-- runtime knob
        max_h = max(64, max_h)                          # floor
        h_tgt = min(H, max_h)                           # cap to source height

        scale = float(h_tgt) / float(H)
        w_tgt = max(8, int(round(W * scale)))

        prev_s = cv.resize(prev_gray, (w_tgt, h_tgt), interpolation=cv.INTER_AREA)
        curr_s = cv.resize(curr_gray, (w_tgt, h_tgt), interpolation=cv.INTER_AREA)

        self._cmat_prev = prev_s
        self._cmat_curr = curr_s
        self._cmat_scale = scale

    def prepare_scaled_frames(self, prev_gray, curr_gray):
        if prev_gray is None or curr_gray is None:
            self._scaled_key  = None
            self._prev_scaled = None
            self._curr_scaled = None
            return

        # key based on underlying buffers so we don’t redo work
        try:
            pid = int(prev_gray.__array_interface__['data'][0])
            cid = int(curr_gray.__array_interface__['data'][0])
            key = (pid, cid)
        except Exception:
            key = None

        if key is not None and key == self._scaled_key:
            return

        # global blur + resize once
        prev_blur = cv.GaussianBlur(prev_gray, (0,0), 0.9)
        curr_blur = cv.GaussianBlur(curr_gray, (0,0), 0.9)

        hS_full = max(8, int(round(prev_blur.shape[0] * SCALE)))
        wS_full = max(8, int(round(prev_blur.shape[1] * SCALE)))

        prev_s_full = cv.resize(prev_blur, (wS_full, hS_full), interpolation=cv.INTER_AREA)
        curr_s_full = cv.resize(curr_blur, (wS_full, hS_full), interpolation=cv.INTER_AREA)

        self._scaled_key  = key
        self._prev_scaled = prev_s_full
        self._curr_scaled = curr_s_full


    def _cmat_fg_union_mask_small(self, rois, scale, Hs, Ws, pad_px: int = 16):
        """
        Foreground union mask in CMAT (small) frame space.
        True = foreground (exclude from camera estimate).
        """
        fg = np.zeros((Hs, Ws), dtype=bool)
        if not rois:
            return fg

        for r in rois:
            try:
                x, y, w, h = map(int, getattr(r, "rect", (0,0,0,0)))
            except Exception:
                continue
            if w <= 1 or h <= 1:
                continue

            x0 = int(round((x - pad_px) * scale))
            y0 = int(round((y - pad_px) * scale))
            x1 = int(round((x + w + pad_px) * scale))
            y1 = int(round((y + h + pad_px) * scale))

            x0 = max(0, min(Ws, x0)); x1 = max(0, min(Ws, x1))
            y0 = max(0, min(Hs, y0)); y1 = max(0, min(Hs, y1))
            if x1 > x0 and y1 > y0:
                fg[y0:y1, x0:x1] = True

        return fg

    def _cmat_fg_union_mask_scaled(self, rois, pad_px: int = 16):
        """
        Build boolean mask in *scaled full-frame* space marking union of all ROI rects (padded).
        True = foreground (exclude from camera estimate).
        """
        if self._prev_scaled is None:
            return None
        Hs, Ws = self._prev_scaled.shape[:2]
        fg = np.zeros((Hs, Ws), dtype=bool)
        if not rois:
            return fg

        for r in rois:
            try:
                x, y, w, h = map(int, getattr(r, "rect", (0,0,0,0)))
            except Exception:
                continue
            if w <= 1 or h <= 1:
                continue

            x0 = int(round((x - pad_px) * SCALE))
            y0 = int(round((y - pad_px) * SCALE))
            x1 = int(round((x + w + pad_px) * SCALE))
            y1 = int(round((y + h + pad_px) * SCALE))

            x0 = max(0, min(Ws, x0)); x1 = max(0, min(Ws, x1))
            y0 = max(0, min(Hs, y0)); y1 = max(0, min(Hs, y1))
            if x1 > x0 and y1 > y0:
                fg[y0:y1, x0:x1] = True

        return fg


    # def _global_drift_core(self, prev_gray, curr_gray):
    #     # Use cached full-frame flow instead of recomputing Farneback here.
    #     # Choose a fixed fb_levels for camera (or make it a constant).
    #     fb_levels = 7
    #     pyr_levels = 1

    #     self.prepare_scaled_frames(prev_gray, curr_gray)
    #     fx_full, fy_full = self._fullflow_get(prev_gray, curr_gray, fb_levels, pyr_levels)
    #     if fx_full is None:
    #         return 0.0, 0.0, 0.0

    #     mag = np.sqrt(fx_full*fx_full + fy_full*fy_full)

    #     # 1) start with "background-like" gate (your existing idea)
    #     thr = float(np.percentile(mag, float(CMAT_BG_MAG_PCTL)))
    #     cand = (mag <= thr)

    #     # 2) optional structure gate (good: prefers trackable pixels)
    #     if CMAT_STRUCT_MODE and CMAT_BG_CURV_GATE > 0.0:
    #         maps_p = iglog_struct_maps_gray_u8(self._prev_scaled, sigma=float(CMAT_IGLOG_SIGMA))
    #         curv01 = maps_p["abs"].astype(np.float32) / 255.0
    #         cand &= (curv01 >= float(CMAT_BG_CURV_GATE))

    #     # 3) EXCLUDE foreground union (the key semantic guardrail)
    #     rois = getattr(self, "_cmat_scene_rois", None)
    #     fg = self._cmat_fg_union_mask_scaled(rois, pad_px=16) if rois is not None else None
    #     if fg is not None:
    #         cand &= (~fg)

    #     # If cand collapses, fall back to the old behavior (no exclusion/consensus)
    #     if int(np.count_nonzero(cand)) < int(CMAT_BG_MIN_PIXELS):
    #         cand = None

    #     def _consensus_median(fx, fy, msk, cone_deg=25.0):
    #         """
    #         Dominant-motion consensus:
    #         - find dominant direction (angle histogram)
    #         - keep inliers within ±cone_deg
    #         - return median dx/dy of inliers
    #         """
    #         if msk is None:
    #             fx1 = fx.reshape(-1)
    #             fy1 = fy.reshape(-1)
    #         else:
    #             fx1 = fx[msk].reshape(-1)
    #             fy1 = fy[msk].reshape(-1)

    #         if fx1.size < 64:
    #             return float(np.median(fx1)) if fx1.size else 0.0, float(np.median(fy1)) if fy1.size else 0.0

    #         # drop near-zero vectors (angle undefined)
    #         m = np.sqrt(fx1*fx1 + fy1*fy1)
    #         nz = m > (0.02 * (float(np.percentile(m, 95.0)) + 1e-9))
    #         fx1 = fx1[nz]; fy1 = fy1[nz]
    #         if fx1.size < 64:
    #             return float(np.median(fx1)) if fx1.size else 0.0, float(np.median(fy1)) if fy1.size else 0.0

    #         ang = np.arctan2(fy1, fx1)  # [-pi, pi]

    #         # direction histogram (cheap mode-finder)
    #         B = 36  # 10-degree bins
    #         bins = ((ang + np.pi) * (B / (2*np.pi))).astype(np.int32)
    #         bins = np.clip(bins, 0, B-1)
    #         hist = np.bincount(bins, minlength=B)
    #         k = int(np.argmax(hist))
    #         a0 = ( (k + 0.5) * (2*np.pi/B) ) - np.pi  # bin center angle

    #         # inliers within cone
    #         cone = float(np.radians(cone_deg))
    #         d = np.arctan2(np.sin(ang - a0), np.cos(ang - a0))  # wrapped diff
    #         inl = np.abs(d) <= cone
    #         if np.count_nonzero(inl) < 32:
    #             # can't form consensus → plain median
    #             return float(np.median(fx1)), float(np.median(fy1))

    #         return float(np.median(fx1[inl])), float(np.median(fy1[inl]))

    #     # 4) CONSENSUS estimate
    #     if cand is None:
    #         vx_med = float(np.median(fx_full))
    #         vy_med = float(np.median(fy_full))
    #     else:
    #         vx_med, vy_med = _consensus_median(fx_full, fy_full, cand, cone_deg=25.0)


    #     # optional zoom proxy via divergence
    #     dvx_dx = cv.Sobel(fx_full, cv.CV_32F, 1, 0, ksize=3) / 8.0
    #     dvy_dy = cv.Sobel(fy_full, cv.CV_32F, 0, 1, ksize=3) / 8.0
    #     div = dvx_dx + dvy_dy

    #     # Use the same mask we trusted for vx/vy
    #     if cand is None:
    #         vz_med = float(np.median(div))
    #     else:
    #         # If cand collapses somehow, fall back cleanly
    #         vz_med = float(np.median(div[cand])) if np.count_nonzero(cand) >= 32 else float(np.median(div))


    #     # rescale from scaled-space to full-res (your ROI code uses SCALE)
    #     vx_med *= (1.0 / SCALE)
    #     vy_med *= (1.0 / SCALE)
    #     vz_med *= (1.0 / SCALE)
    #     return vx_med, vy_med, vz_med

    

    def _global_drift_core(self, prev_gray, curr_gray):
        self.prepare_cmat_frames(prev_gray, curr_gray)
        prev_s = self._cmat_prev
        curr_s = self._cmat_curr
        scale  = float(self._cmat_scale)

        FBc = dict(FB)
        FBc["levels"]    = int(max(1, CMAT_FB_LEVELS))
        FBc["winsize"]   = int(max(9, CMAT_FB_WINSIZE))
        FBc["iterations"]= 2
        FBc["poly_n"]    = 5
        FBc["poly_sigma"]= 1.1


        flow = cv.calcOpticalFlowFarneback(prev_s, curr_s, None, **FBc)
        fx = flow[...,0].astype(np.float32)
        fy = flow[...,1].astype(np.float32)

        mag = np.sqrt(fx*fx + fy*fy)

        # ---------------- CMAT STRUCTURAL SALIENCE (HG-style weighting) ----------------
        # Goal: camera drift should be estimated from "background-like" pixels that are
        # (a) low-motion, (b) structurally meaningful, and (c) actually changed structurally.
        #
        # We do NOT replace Farnebäck here (we still need a smooth metric field).
        # We only change WHERE we trust it by building a structural-change weight hg01 in [0,1].
        hg01 = None
        if bool(CMAT_STRUCT_MODE):
            try:
                st = self.__dict__.setdefault("_cmat_struct_state", {})
                Mp = struct_M_from_gray_u8(prev_s, st)   # uint8 preferred (GSCM/IG-LoG hybrid)
                Mc = struct_M_from_gray_u8(curr_s, st)
                hg01 = _robust01_from_u8_absdiff(Mp, Mc, p_lo=10.0, p_hi=90.0, blur_sigma=1.0, gamma=0.85)
            except Exception:
                hg01 = None
        # ------------------------------------------------------------------------------

        thr = float(np.percentile(mag, float(CMAT_BG_MAG_PCTL)))
        cand = (mag <= thr)

        # ROI exclusion (now in CMAT scale)
        rois = getattr(self, "_cmat_scene_rois", None)
        if rois is not None:
            fg = self._cmat_fg_union_mask_small(rois, scale, prev_s.shape[0], prev_s.shape[1], pad_px=16)
            cand &= (~fg)

        # Optional structure gate (prefer pixels with reliable structure)
        if bool(CMAT_STRUCT_MODE) and float(CMAT_BG_CURV_GATE) > 0.0:
            try:
                maps_p = iglog_struct_maps_gray_u8(prev_s, sigma=float(CMAT_IGLOG_SIGMA))
                curv01 = maps_p["abs"].astype(np.float32) / 255.0
                cand &= (curv01 >= float(CMAT_BG_CURV_GATE))
            except Exception:
                pass

        if int(np.count_nonzero(cand)) < int(CMAT_BG_MIN_PIXELS):
            cand = None

        if cand is None and rois is not None:
            # fallback: use a border ring (background-biased) even if ROI covers most center
            Hs, Ws = prev_s.shape[:2]
            ring = max(4, min(12, Hs//10))
            border = np.zeros((Hs, Ws), dtype=bool)
            border[:ring, :] = True
            border[-ring:, :] = True
            border[:, :ring] = True
            border[:, -ring:] = True
            fg = self._cmat_fg_union_mask_small(rois, scale, Hs, Ws, pad_px=16)
            cand = border & (~fg)
            if int(np.count_nonzero(cand)) < 64:
                cand = None

        # ---------------- Weighted consensus estimate (structural + motion weights) ----------------
        # weights = hg01 * mag01, so:
        #  - hg01 suppresses pixels with no structural correspondence/change (lighting junk, flats)
        #  - mag01 suppresses near-zero vectors (angle undefined; reduces "direction flip" jitter)
        w = None
        if hg01 is not None:
            try:
                m95 = float(np.percentile(mag, 95.0)) or 1e-9
                mag01 = np.clip(mag / m95, 0.0, 1.0).astype(np.float32)
                w = (hg01.astype(np.float32) * mag01).astype(np.float32)
            except Exception:
                w = None

        # Build a consensus inlier mask (same spirit as _consensus_median), then weighted-median inside it.
        if cand is None:
            base = np.ones_like(mag, dtype=bool)
        else:
            base = cand

        inl = base
        try:
            fx1 = fx[inl].reshape(-1)
            fy1 = fy[inl].reshape(-1)
            if fx1.size >= 64:
                m = np.sqrt(fx1*fx1 + fy1*fy1)
                nz = m > (0.02 * (float(np.percentile(m, 95.0)) + 1e-9))
                if np.count_nonzero(nz) >= 64:
                    ang = np.arctan2(fy1[nz], fx1[nz])
                    B = 36
                    bins = ((ang + np.pi) * (B / (2*np.pi))).astype(np.int32)
                    bins = np.clip(bins, 0, B-1)
                    hist = np.bincount(bins, minlength=B)
                    k = int(np.argmax(hist))
                    a0 = (((k + 0.5) * (2*np.pi/B)) - np.pi)
                    cone = float(np.radians(25.0))
                    d = np.arctan2(np.sin(ang - a0), np.cos(ang - a0))
                    # rebuild inlier mask in full image coords
                    tmp = np.zeros_like(inl, dtype=bool)
                    idx = np.where(inl.reshape(-1))[0]
                    idx_nz = idx[nz]
                    tmp.reshape(-1)[idx_nz] = (np.abs(d) <= cone)
                    if int(np.count_nonzero(tmp)) >= 32:
                        inl = tmp
        except Exception:
            pass

        if w is not None and int(np.count_nonzero(inl)) >= 32:
            try:
                ww = w[inl]
                if float(np.sum(ww)) > 1e-6:
                    vx_med = _weighted_median(fx[inl], ww)
                    vy_med = _weighted_median(fy[inl], ww)
                else:
                    vx_med = float(np.median(fx[inl]))
                    vy_med = float(np.median(fy[inl]))
            except Exception:
                vx_med = float(np.median(fx[inl]))
                vy_med = float(np.median(fy[inl]))
        else:
            vx_med, vy_med = _consensus_median(fx, fy, cand, cone_deg=25.0)

        # divergence (optional): weight the same inliers when possible
        dvx_dx = cv.Sobel(fx, cv.CV_32F, 1, 0, ksize=3) / 8.0
        dvy_dy = cv.Sobel(fy, cv.CV_32F, 0, 1, ksize=3) / 8.0
        div = dvx_dx + dvy_dy

        if w is not None and int(np.count_nonzero(inl)) >= 32:
            try:
                ww = w[inl]
                if float(np.sum(ww)) > 1e-6:
                    vz_med = _weighted_median(div[inl], ww)
                else:
                    vz_med = float(np.median(div[inl]))
            except Exception:
                vz_med = float(np.median(div[inl]))
        else:
            vz_med = float(np.median(div)) if cand is None else float(np.median(div[cand]))
        # -------------------------------------------------------------------------------------------

        # rescale from CMAT pixels back to full-res pixels
        inv = 1.0 / max(1e-9, scale)
        vx_med *= inv
        vy_med *= inv
        vz_med *= inv
        return vx_med, vy_med, vz_med


    def _global_drift(self, prev_gray, curr_gray):
        """Return camera drift with a per-frame cache.

        Cache key = (id(prev_gray), id(curr_gray)). If the same pair is
        requested multiple times in a frame (multiple ROIs), we reuse the
        cached result instead of recomputing Farnebäck on the whole frame.
        """
        # build a lightweight identity key for the current frame
        key = None
        try:
            pid = int(prev_gray.__array_interface__['data'][0])
            cid = int(curr_gray.__array_interface__['data'][0])
            key = (pid, cid)
        except Exception:
            pass

        # fast path: cache hit
        if key is not None and self._gm_cache_key == key and self._gm_cache_ok:
            return self._gm_cache_val

        # either first use this frame or cache miss → compute directly
        vx, vy, vz = self._global_drift_core(prev_gray, curr_gray)
        if vx is None:
            val = (0.0, 0.0, 0.0)
            ok  = False
        else:
            val = (float(vx), float(vy), float(vz))
            ok  = True

        self._gm_cache_key = key
        self._gm_cache_val = val
        self._gm_cache_ok  = ok
        return val

    def prefetch_global_drift(self, prev_gray, curr_gray):
        """Optionally kick off a background compute of global drift.

        This runs _global_drift_core in a daemon thread and populates the
        same cache used by _global_drift(). If the result is not ready
        when a ROI asks for it, _global_drift() will simply recompute
        synchronously, so correctness never depends on the thread.
        """
        if prev_gray is None or curr_gray is None:
            return

        # if a compute is already in progress, don't spawn another
        if self._gm_thread is not None and self._gm_thread.is_alive():
            return

        # if cache is already valid for this frame, nothing to do
        key = None
        try:
            pid = int(prev_gray.__array_interface__['data'][0])
            cid = int(curr_gray.__array_interface__['data'][0])
            key = (pid, cid)
        except Exception:
            pass
        if key is not None and self._gm_cache_key == key and self._gm_cache_ok:
            return

        def _worker(prev_ref=prev_gray, curr_ref=curr_gray, key_ref=key):
            vx, vy, vz = self._global_drift_core(prev_ref, curr_ref)
            if vx is None:
                val = (0.0, 0.0, 0.0)
                ok  = False
            else:
                val = (float(vx), float(vy), float(vz))
                ok  = True
            self._gm_cache_key = key_ref
            self._gm_cache_val = val
            self._gm_cache_ok  = ok

        self._gm_thread = threading.Thread(target=_worker, daemon=True)
        self._gm_thread.start()


    def _ring_drift(self, prev_gray, curr_gray, roi, ring_px):
        rx, ry, rw, rh = map(int, roi.rect)
        r = int(max(4, ring_px))
        x0 = max(0, rx - r); y0 = max(0, ry - r)
        x1 = min(self.W, rx + rw + r); y1 = min(self.H, ry + rh + r)
        strips = [(x0, y0, x1 - x0, r), (x0, y1 - r, x1 - x0, r)]
        h_mid = max(0, (y1 - y0) - 2*r)
        if h_mid >= 4:
            strips += [(x0, y0 + r, r, h_mid), (x1 - r, y0 + r, r, h_mid)]
        vals = []
        for rc in strips:
            v = self._roi_flow(prev_gray, curr_gray, rc, roi.fb_levels)
            if v[0] is not None: vals.append(v)
        if not vals: return 0.0, 0.0, 0.0
        vx = float(np.median([v[0] for v in vals]))
        vy = float(np.median([v[1] for v in vals]))
        vz = float(np.median([v[2] for v in vals]))
        return vx, vy, vz

    def _fullflow_get(self, prev_gray, curr_gray, fb_levels: int, pyr_levels: int):
        """
        Returns full-frame scaled flow (fx, fy) for the current framepair,
        computed ONCE per (fb_levels, pyr_levels).
        """
        # Need scaled frames ready (prepare_scaled_frames already builds these)
        if self._prev_scaled is None or self._curr_scaled is None:
            return None, None

        # framepair identity
        try:
            pid = int(prev_gray.__array_interface__['data'][0])
            cid = int(curr_gray.__array_interface__['data'][0])
            base_key = (pid, cid)
        except Exception:
            base_key = None

        # preprocess signature (anything that changes input to Farneback)
        preprocess_sig = (
            int(getattr(self, "edge_sharpen_mode", 0)),
            int(getattr(self, "flow_pyr_mode", 0)),
            int(getattr(self, "flow_pyr_levels", 1)),
        )

        # reset cache on new framepair or changed preprocess knobs
        if base_key is None or self._fflow_key != (base_key, preprocess_sig):
            self._fflow_key = (base_key, preprocess_sig)
            self._fflow_cache = {}
            self._fflow_prev_proc = None
            self._fflow_curr_proc = None
            self._fflow_preprocess_sig = preprocess_sig

        k = (int(max(1, fb_levels)), int(max(1, pyr_levels)))
        if k in self._fflow_cache:
            return self._fflow_cache[k]

        # Build preprocessed full scaled frames ONCE per framepair.
        # Keep this lightweight (global) — do NOT do ROI-specific stuff here.
        if self._fflow_prev_proc is None or self._fflow_curr_proc is None:
            prev_s = self._prev_scaled
            curr_s = self._curr_scaled

            # (Optional) reuse your mild structure injection globally
            # (copied from _roi_flow’s _inject_gauss_struct)
            def _inject_gauss_struct(img, eps=0.12):
                out = img.astype(np.float32)
                for sigma, w in [(1.0, eps), (2.0, eps * 0.5), (3.5, eps * 0.25)]:
                    LoG = cv.GaussianBlur(img, (0,0), sigma)
                    LoG = img.astype(np.float32) - LoG.astype(np.float32)
                    out += w * LoG
                return np.clip(out, 0, 255).astype(np.uint8)

            prev_s = _inject_gauss_struct(prev_s)
            curr_s = _inject_gauss_struct(curr_s)

            # Apply your existing adaptive blur globally (cheap)
            mode = int(getattr(self, "edge_sharpen_mode", 0))
            prev_s, curr_s, sigma, blur_strength = _adaptive_blur(prev_s, curr_s, edge_mode=mode)

            self._fflow_prev_proc = prev_s
            self._fflow_curr_proc = curr_s

        # Farneback params for this fb_levels
        FB2 = dict(FB)
        FB2["levels"] = int(max(1, fb_levels))

        fx_full, fy_full = _fb_flow_gauss_pyr(self._fflow_prev_proc, self._fflow_curr_proc, FB2, n_levels=int(max(1, pyr_levels)))
        self._fflow_cache[k] = (fx_full, fy_full)
        return fx_full, fy_full


    def _roi_flow(self, prev_gray, curr_gray, rc,
            levels: Optional[int]=None,
            debug_overlay: bool=False,
            roi: Optional[ROI]=None):
        x,y,w,h = rc

        if self._prev_scaled is not None and self._curr_scaled is not None:
            # map ROI rect into scaled space
            sx = int(round(x * SCALE))
            sy = int(round(y * SCALE))
            sw = max(8, int(round(w * SCALE)))
            sh = max(8, int(round(h * SCALE)))
            prev_s = self._prev_scaled[sy:sy+sh, sx:sx+sw]
            curr_s = self._curr_scaled[sy:sy+sh, sx:sx+sw]
            prev_s_gray = prev_s.copy()
            curr_s_gray = curr_s.copy()
        else:
            # fallback: old behavior
            prev = prev_gray[y:y+h, x:x+w]
            curr = curr_gray[y:y+h, x:x+w]
            if prev.size == 0 or curr.size == 0 or w<2 or h<2:
                return None, None, None
            hS = max(8, int(round(prev.shape[0]*SCALE)))
            wS = max(8, int(round(prev.shape[1]*SCALE)))
            prev = cv.GaussianBlur(prev, (0,0), 0.9)
            curr = cv.GaussianBlur(curr, (0,0), 0.9)
            prev_s = cv.resize(prev, (wS, hS), interpolation=cv.INTER_AREA)
            curr_s = cv.resize(curr, (wS, hS), interpolation=cv.INTER_AREA)
            prev_s_gray = prev_s.copy()
            curr_s_gray = curr_s.copy()

        # initialize debug storage
        # width in *scaled* ROI pixels (works for both branches)
        hS_local, wS_local = prev_s.shape[:2]

        # initialize debug storage
        if roi is not None:
            roi._debug_blob_pts = None
            roi._roi_scale = float(wS_local) / float(w if w > 0 else 1)

        # --- TRACKMODE: Hypergraph-first option (replaces Farnebäck) -----------------
        # Goal: when Farnebäck collapses on ultra-fast motion, force the more robust
        # structure-based translation tracker every frame (not just "rescue on lowconf").
        #
        # What you get:
        #   - vx/vy from _hg_rescue_translation() (scaled ROI px/frame)
        #   - fx/fy become a constant translation field (stable downstream)
        #   - vz = 0 (translation-only mode; keep Z scaling off if you need depth)
        #
        # Why here:
        #   - prev_s_gray/curr_s_gray already exist and are ROI-scaled (ideal for HG).
        # --- TRACKMODE: HG / HYBRID gate --------------------------------------------
        use_hg = False
        use_hybrid = False
        if roi is not None:
            mm = str(getattr(roi, "motion_mode", "fb") or "fb").lower()
            use_hg = mm.startswith("hg") or bool(getattr(self, "force_hg_mode", False))
            use_hybrid = mm.startswith("hyb")
        else:
            use_hg = bool(getattr(self, "force_hg_mode", False))
            use_hybrid = False

        # HYBRID state lives on the ROI (no dataclass fields needed)
        hyb = None
        if roi is not None and use_hybrid and HYB_ENABLE:
            hyb = roi.__dict__.setdefault("_hyb_state", {"cool": 0, "ctr": 0, "has_fb": False})

        # If HG or HYBRID: always compute HG translation first (cheap + timing truth)
        if (use_hg or (use_hybrid and HYB_ENABLE)) and roi is not None:
            try:
                _anchor_update_and_mask(roi, curr_s_gray)
            except Exception:
                pass

            try:
                vx_hg_s, vy_hg_s, conf_hg, dbg_hg = _hg_rescue_translation(prev_s_gray, curr_s_gray, roi)
                roi._hg_used = True
                roi._hg_dbg = dbg_hg

                # HG step in FULL-RES px/frame for gating
                dx_hg_pf = float(vx_hg_s) * (1.0 / SCALE)
                dy_hg_pf = float(vy_hg_s) * (1.0 / SCALE)
                step_pf = float(math.hypot(dx_hg_pf, dy_hg_pf))

                # Determine "event" (skip FB) vs "calm" (FB allowed sometimes)
                diag = float(math.hypot(float(w), float(h))) + 1e-9
                is_event = (step_pf >= float(HYB_EVENT_STEP_FRAC) * diag)

                # novelty gate (uses previous frame's stored novelty; ok if missing)
                nov = float(getattr(roi, "_pe_novelty", 0.0))
                denom = max(1e-6, float(PE_NOVELTY_THRESH) - 1.0)
                nov_n = float(np.clip((nov - 1.0) / denom, 0.0, 1.0))
                nov_event = (nov_n >= float(HYB_NOV_FRAC))

                if hyb is not None:
                    # update cooldown logic
                    if int(hyb.get("cool", 0)) > 0:
                        hyb["cool"] = int(hyb["cool"]) - 1
                    if is_event or nov_event:
                        hyb["cool"] = int(HYB_COOLDOWN_FR)

                    # decide if we run FB this frame
                    calm = (step_pf <= float(HYB_CALM_STEP_FRAC) * diag) and (int(hyb.get("cool", 0)) <= 0)
                    hyb["ctr"] = int(hyb.get("ctr", 0)) + 1
                    run_fb_now = bool(calm) and (int(hyb["ctr"]) % int(max(1, HYB_FB_EVERY_N)) == 0)

                    # stash HG for later override after FB computes
                    roi.__dict__["_hyb_hg_vx_s"] = float(vx_hg_s)   # scaled px/frame
                    roi.__dict__["_hyb_hg_vy_s"] = float(vy_hg_s)

                    # If not running FB, return HG immediately (fast path)
                    if not run_fb_now:
                        roi._frame_lowconf = False
                        return float(dx_hg_pf), float(dy_hg_pf), 0.0

                    # else: fall through into FB path, but we will override vx/vy back to HG at the end
                    roi.__dict__["_hyb_override_vxy"] = True
                else:
                    # pure HG mode: return immediately (same as before)
                    roi._frame_lowconf = False
                    return float(dx_hg_pf), float(dy_hg_pf), 0.0

            except Exception:
                # If HG blows up, fall through to normal Farnebäck path.
                pass
        # --- END TRACKMODE -----------------------------------------------------------


        def _inject_gauss_struct(img, eps=0.12):
            # multi-scale LoG
            out = img.astype(np.float32)
            for sigma, w in [(1.0, eps),
                            (2.0, eps * 0.5),
                            (3.5, eps * 0.25)]:
                LoG = cv.GaussianBlur(img, (0,0), sigma)
                LoG = img.astype(np.float32) - LoG.astype(np.float32)
                out += w * LoG
            return np.clip(out, 0, 255).astype(np.uint8)
        
        prev_s = _inject_gauss_struct(prev_s)
        curr_s = _inject_gauss_struct(curr_s)


        # --- NEW: optional edge-sharpening before adaptive blur ---
        mode = getattr(self, "edge_sharpen_mode", 0)
        if mode == 1:          # mild
            radius, amount = 0.7, 0.6
            prev_s, curr_s = _unsharp_pair(prev_s, curr_s, radius, amount)
        elif mode == 2:        # stronger
            radius, amount = 1.0, 1.2
            prev_s, curr_s = _unsharp_pair(prev_s, curr_s, radius, amount)
        # --- END NEW BLOCK ---

        sigma = None
        USE_CD = False
        CD_R   = 3  # odd → avoids Nyquist zero; targets ~λ≈36 px at ROI scale
        if USE_CD:
            prev_s = cd_bandpass_u8(prev_s, CD_R, mode="lor")
            curr_s = cd_bandpass_u8(curr_s, CD_R, mode="lor")
        else:
            mode = self.edge_sharpen_mode
            prev_s, curr_s, sigma, blur_strength = _adaptive_blur(prev_s, curr_s, edge_mode=mode)
        
        FB2 = dict(FB)
        if levels is not None:
            FB2["levels"] = max(1, int(levels))

        # HYBRID: keep FB cheap when we *do* run it
        if roi is not None and bool(roi.__dict__.get("_hyb_override_vxy", False)):
            try:
                FB2["levels"] = int(min(int(FB2.get("levels", 5)), int(HYB_FB_LEVELS_MAX)))
            except Exception:
                pass

        if sigma is not None and sigma > 0.0:
            FB2["flags"]   = cv.OPTFLOW_FARNEBACK_GAUSSIAN
            FB2["winsize"] = max(FB2.get("winsize", 27),
                                 27 + int(24 * np.clip(blur_strength, 0, 1)))

        # --- NEW: external Gaussian pyramid around Farneback ---
        # ROI override > tracker default > 1 (=off)
        # ROI override in future if you want it; for now use tracker default.
                # --- Gaussian pyramid supercharger: OFF / ALWAYS / AUTO ---
        pyr_mode = int(getattr(self, "flow_pyr_mode", 0))
        pyr_levels = 1  # default: OFF

        if pyr_mode == 0:
            # OFF: behave like original
            pyr_levels = 1
        elif pyr_mode == 1:
            # ALWAYS: always use configured depth
            pyr_levels = int(getattr(self, "flow_pyr_levels", FLOW_PYR_LEVELS_DEFAULT))
        else:
            # AUTO: decide based on last frame's speed vs ROI size
            speed_s = 0.0
            if roi is not None:
                try:
                    speed_s = float(getattr(roi, "last_speed_px_s", 0.0))
                except Exception:
                    speed_s = 0.0
            fps = max(float(getattr(self, "fps", 0.0)) or 0.0, 1e-3)
            speed_pf = speed_s / fps  # px per frame

            diag = math.hypot(float(w), float(h))  # ROI diag at full-res
            # If we're covering a big fraction of ROI per frame, enable pyramid
            if diag > 0.0 and speed_pf > 0.35 * diag:
                pyr_levels = int(getattr(self, "flow_pyr_levels", FLOW_PYR_LEVELS_DEFAULT))
            else:
                pyr_levels = 1

        pyr_levels = int(max(1, min(4, pyr_levels)))


        # ROI-local (fast): compute flow on the ROI crop
        fx, fy = _fb_flow_gauss_pyr(prev_s, curr_s, FB2, n_levels=pyr_levels)

        # --- QUICK DISPLACEMENT RESCUE: phase correlation when motion per-frame is huge ---
        if roi is not None:
            try:
                # estimate px/frame from last speed (px/s)
                speed_pf = float(getattr(roi, "last_speed_px_s", 0.0)) / max(1e-6, float(self.fps))
                diag = math.hypot(float(w), float(h))  # full-res ROI diag
                big_jump = (diag > 1.0 and speed_pf > 0.35 * diag)

                if big_jump:
                    # phaseCorrelate expects float32, same size
                    a = prev_s.astype(np.float32) / 255.0
                    b = curr_s.astype(np.float32) / 255.0
                    (dx, dy), resp = cv.phaseCorrelate(a, b)
                    print("Big jump phaseCorrelate dx={:.2f} dy={:.2f} resp={:.3f}".format(dx, dy))
                    # resp ~ [0..1], higher is better
                    if resp is not None and float(resp) > 0.15:
                        # override the field with a constant translation (px/frame at ROI-scale)
                        fx = np.full_like(fx, float(dx), dtype=np.float32)
                        fy = np.full_like(fy, float(dy), dtype=np.float32)
            except Exception:
                pass
        # --- END DISPLACEMENT RESCUE ---


        mag = np.sqrt(fx*fx + fy*fy)

        anchor_mask = None
        if roi is not None:
            anchor_mask = _anchor_update_and_mask(roi, curr_s_gray)

        ai_mask = None
        if roi is not None and getattr(roi, "ai_outline_enabled", False):
            # Build in scaled ROI patch space (curr_s_gray is the scaled crop)
            ai_mask = _roi_ai_outline_mask(roi, curr_s_gray.shape[:2])

        # --- DEBUG: capture colorized mag patch + alpha mask ---
        mode = getattr(self, "debug_flow_mode", 0)
        if mode != 0 and debug_overlay:
            x, y, w, h = rc
            w = int(w); h = int(h)

            if mode == 1:
                # WITH processing: use current mag
                mag_dbg = mag
            else:
                # NO processing: recompute flow on raw ROI crop (no adaptive blur)
                prev_raw = prev_gray[y:y+h, x:x+w]
                curr_raw = curr_gray[y:y+h, x:x+w]

                hS_np = max(8, int(round(prev_raw.shape[0] * SCALE)))
                wS_np = max(8, int(round(prev_raw.shape[1] * SCALE)))
                prev_np = cv.resize(prev_raw, (wS_np, hS_np), interpolation=cv.INTER_AREA)
                curr_np = cv.resize(curr_raw, (wS_np, hS_np), interpolation=cv.INTER_AREA)

                flow_np = cv.calcOpticalFlowFarneback(prev_np, curr_np, None, **FB2)
                fx_np   = flow_np[...,0].astype(np.float32)
                fy_np   = flow_np[...,1].astype(np.float32)
                mag_dbg = np.sqrt(fx_np*fx_np + fy_np*fy_np)

            # robust scale 0..1
            m95 = float(np.percentile(mag_dbg, 95.0))
            if m95 <= 1e-9:
                m95 = 1e-9
            mag_norm = np.clip(mag_dbg / m95, 0.0, 1.0).astype(np.float32)

            mag_u8  = (mag_norm * 255.0).astype(np.uint8)
            mag_col = cv.applyColorMap(mag_u8, cv.COLORMAP_TURBO)

            dbg_patch = cv.resize(mag_col,  (w, h), interpolation=cv.INTER_NEAREST)
            dbg_alpha = cv.resize(mag_norm, (w, h), interpolation=cv.INTER_LINEAR)

            self.debug_roi = (int(x), int(y), dbg_patch, dbg_alpha)

            # --- NEW: stability-preserving mask selection ---

        # --- NEW: optional forward–backward consistency (ROI-scale) ---
        fb_err_map = None
        if getattr(self, "fb_consistency_mode", 0) > 0:
            try:
                # backward flow: curr_s -> prev_s
                fx_bw, fy_bw = _fb_flow_gauss_pyr(curr_s, prev_s, FB2, n_levels=pyr_levels)

                Hs, Ws = prev_s.shape[:2]
                yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
                x_fw = xx + fx
                y_fw = yy + fy

                # clamp coordinates
                x_fw_cl = np.clip(x_fw, 0.0, Ws - 1.0).astype(np.float32)
                y_fw_cl = np.clip(y_fw, 0.0, Hs - 1.0).astype(np.float32)

                # sample backward field at forward-mapped positions
                fx_bw_s = cv.remap(fx_bw, x_fw_cl, y_fw_cl, cv.INTER_LINEAR)
                fy_bw_s = cv.remap(fy_bw, x_fw_cl, y_fw_cl, cv.INTER_LINEAR)

                # consistency error: forward + backward should ≈ 0
                ex = fx + fx_bw_s
                ey = fy + fy_bw_s
                fb_err_map = np.sqrt(ex*ex + ey*ey).astype(np.float32)
            except Exception:
                fb_err_map = None
        # --- END NEW ---

        # 1) define "non-trivial" motion relative to max
        m_max = float(np.max(mag))
        if m_max <= 1e-9:
            # completely dead region; fallback to plain medians
            vx_med = float(np.median(fx))
            vy_med = float(np.median(fy))
            mask = None
        else:
            eps = 0.02 * m_max  # 2% of max mag as "dead" cutoff; tune 1–3%
            nz = mag > eps

            # if we have enough non-trivial motion, do percentile on that set only
            if np.count_nonzero(nz) >= 32:
                mthr = np.percentile(mag[nz], 70)  # top 30% of *non-dead* pixels
                mask = (mag >= mthr) & nz
            else:
                # fallback: original behavior on full field
                mthr = np.percentile(mag, 70)
                mask = mag >= mthr

            ang = np.arctan2(fy, fx)

            # guard: if mask got too small or weird, fall back to no angle-gating
            if np.count_nonzero(mask) >= 16:
                a0 = np.median(ang[mask])
                mask &= np.cos(ang - a0) >= 0.8   # ±36.9° cone

            # final safety: if angle gate nuked it, go back to magnitude-only mask
            if np.count_nonzero(mask) < 16:
                mask = mag >= mthr

            # --- NEW: cut out high forward–backward error if available ---
            if fb_err_map is not None:
                bad = fb_err_map > float(FB_ERR_THRESH)
                # only restrict if we still have usable pixels afterwards
                mask_fb = mask & (~bad)
                if np.count_nonzero(mask_fb) >= 16:
                    mask = mask_fb
            # --- END NEW ---
            # optionally intersect with anchor_mask if available
                    # --- SMART MASK SELECTION WITH ANCHOR ---
        use_mask = mask

        sim  = float(getattr(roi, "_anchor_last_sim", 0.0)) if roi is not None else 0.0
        ok   = bool(getattr(roi, "_anchor_last_ok", False)) if roi is not None else False

        if anchor_mask is not None and ok:
            # non-trivial motion (reuse nz = mag > eps)
            nz = mag > (0.02 * float(np.max(mag) or 1.0))

            # 1) HIGH confidence → trust anchor region outright (within nz)
            if sim >= ANCHOR_LOCK_THRESH:
                strong_anchor = anchor_mask & nz
                if np.count_nonzero(strong_anchor) >= 8:
                    use_mask = strong_anchor
                else:
                    use_mask = anchor_mask

            # 2) MEDIUM confidence → intersection of anchor & mag+angle mask
            elif sim >= ANCHOR_MASK_THRESH:
                inter = mask & anchor_mask
                if np.count_nonzero(inter) >= 8:
                    use_mask = inter
                else:
                    use_mask = mask

            # 3) LOW confidence → ignore anchor_mask completely (use_mask stays = mask)

        if ai_mask is not None and roi is not None:
            # ai_outline_enabled is already the user's explicit toggle;
            # confidence gating is handled when setting that flag for AI.
            mode = str(getattr(roi, "ai_outline_mode", "include") or "include").lower()
            if mode.startswith("exc"):
                inter_ai = use_mask & (~ai_mask)
            else:
                inter_ai = use_mask & ai_mask
            if np.count_nonzero(inter_ai) >= 16:
                use_mask = inter_ai

        # final fallback if use_mask is too small
        if np.count_nonzero(use_mask) >= 16:
            vx_med = float(np.median(fx[use_mask]))
            vy_med = float(np.median(fy[use_mask]))
        else:
            vx_med = float(np.median(fx))
            vy_med = float(np.median(fy))

        # --- NEW: per-ROI low-confidence frame flag (global, not just anchor) ---
        if roi is not None:
            frame_lowconf = False

            m_max = float(np.max(mag)) if mag.size else 0.0
            if m_max > 1e-9:
                nz_all   = mag > (0.02 * m_max)
                nontriv  = int(np.count_nonzero(nz_all))

                flat_motion  = (m_max < 0.05)
                thin_support = (nontriv < 24)

                frame_lowconf = (flat_motion or thin_support)

                # --- NEW: forward–backward error as extra confidence gate ---
                if fb_err_map is not None and nontriv > 0:
                    bad = (fb_err_map > float(FB_ERR_THRESH)) & nz_all
                    bad_frac = float(np.count_nonzero(bad)) / float(nontriv)
                    # lots of inconsistent vectors → downrank this frame
                    if bad_frac > 0.45:
                        frame_lowconf = True
                # --- END NEW ---
                # --- NEW: prediction-error / novelty gate (detect Farneback 'skips' / wrong-solution jumps) ---
                try:
                    fps_pe = max(float(getattr(self, "fps", 0.0)) or 0.0, 1e-3)

                    # measured translation (full-res px/frame). NOTE: vx_med is ROI-scale px/frame.
                    meas_dx = float(vx_med) * (1.0 / SCALE)
                    meas_dy = float(vy_med) * (1.0 / SCALE)

                    # predicted translation from last known velocity (full-res px/frame)
                    pred_dx = float(getattr(roi, "vx_ps", 0.0)) / fps_pe
                    pred_dy = float(getattr(roi, "vy_ps", 0.0)) / fps_pe

                    pe = float(math.hypot(meas_dx - pred_dx, meas_dy - pred_dy))
                    st_pe = roi.__dict__.setdefault("_pe_state", {"avg": 1e-3})
                    avg0 = float(st_pe.get("avg", 1e-3))
                    avg1 = float(PE_AVG_DECAY) * avg0 + (1.0 - float(PE_AVG_DECAY)) * pe
                    st_pe["avg"] = avg1

                    novelty = pe / max(1e-3, avg1)
                    roi._pe_pf = pe
                    roi._pe_novelty = novelty

                    pred_mag = float(math.hypot(pred_dx, pred_dy))
                    if pred_mag >= float(PE_MIN_PRED_PF) and novelty >= float(PE_NOVELTY_THRESH):
                        frame_lowconf = True
                except Exception:
                    pass
                # --- END NEW ---


            # If anchor is present AND confidently locked, we trust that more
            user_anchor = bool(getattr(roi, "anchor_user_set", False))
            anchor_ok   = bool(getattr(roi, "_anchor_last_ok", False))
            if user_anchor and anchor_ok:
                frame_lowconf = False

            roi._frame_lowconf = frame_lowconf
        # ----------------------------------------------------------- ---
        # --- Rescue when Farneback field is weak ----------------------------
        if roi is not None and getattr(roi, "_frame_lowconf", False):

            # 1) Structural-Farneback rescue (ROI-scale, ABS always-on)
            try:
                st = roi.__dict__.setdefault("_iglog_state_structfb", {})
                M_prev = struct_M_from_gray_u8(prev_s_gray, st)
                M_curr = struct_M_from_gray_u8(curr_s_gray, st)

                fx_s, fy_s = _fb_flow_gauss_pyr(M_prev, M_curr, FB2, n_levels=pyr_levels)

                # quick sanity: does it have enough nontrivial support?
                mag_s = np.sqrt(fx_s*fx_s + fy_s*fy_s).astype(np.float32)
                mmax = float(np.max(mag_s)) if mag_s.size else 0.0
                if mmax > 1e-6:
                    nz = mag_s > (0.02 * mmax)
                    nontriv = int(np.count_nonzero(nz))

                    # Accept if we have some support (tune thresholds later)
                    if nontriv >= 24:
                        # Replace flow field with structural flow (still ROI-scale)
                        fx, fy = fx_s, fy_s

                        mag = mag_s  # keep downstream stats consistent (div/vz use mag)
                        try:
                            if nz is not None and int(np.count_nonzero(nz)) >= 16:
                                vx_med = float(np.median(fx_s[nz]))
                                vy_med = float(np.median(fy_s[nz]))
                            else:
                                vx_med = float(np.median(fx_s))
                                vy_med = float(np.median(fy_s))
                        except Exception:
                            vx_med = float(np.median(fx_s))
                            vy_med = float(np.median(fy_s))

                        roi._frame_lowconf = False
                        roi._structfb_used = True
                        print(f"ROI '{roi.name}' lowconf rescued by Struct-FB")
            except Exception:
                pass

            # 2) HG rescue (phaseCorr + structure-LK, anchor-aware)
            if getattr(roi, "_frame_lowconf", False) and bool(HG_RESCUE_ENABLE):
                try:
                    vx_hg, vy_hg, conf_hg, dbg_hg = _hg_rescue_translation(prev_s_gray, curr_s_gray, roi)
                    if float(conf_hg) >= float(HG_ACCEPT_CONF):
                        vx_med = float(vx_hg)
                        vy_med = float(vy_hg)

                        # Downstream depth metrics: force a stable translation field (div ~ 0)
                        fx = np.full_like(fx, vx_med, dtype=np.float32)
                        fy = np.full_like(fy, vy_med, dtype=np.float32)
                        mag = np.sqrt(fx*fx + fy*fy)

                        roi._frame_lowconf = False
                        roi._hg_used = True
                        roi._hg_dbg = dbg_hg
                        print(f"ROI '{roi.name}' lowconf rescued by HG ({dbg_hg.get('method','?')}, conf={conf_hg:.2f})")
                except Exception:
                    pass

            # 3) If still lowconf, fall back to your existing LoG+LK translation rescue
            if getattr(roi, "_frame_lowconf", False):
                vx_lk, vy_lk, ok_lk = _lk_flow_from_blobs(prev_s_gray, curr_s_gray)
                if ok_lk:
                    vx_med = vx_lk  # LK returns ROI-scale px/frame
                    vy_med = vy_lk  # LK returns ROI-scale px/frame
                    # stabilize downstream div/vz: use a constant translation field on rescued frames
                    fx = np.full_like(fx, float(vx_med), dtype=np.float32)
                    fy = np.full_like(fy, float(vy_med), dtype=np.float32)
                    mag = np.sqrt(fx*fx + fy*fy)

                    roi._frame_lowconf = False   # this frame is now “rescued”

                    if getattr(self, "debug_show_log_blobs", False):
                        roi._debug_blob_pts = _log_blob_points(prev_s_gray, max_kp=80, p_hi=92.0)
        # --------------------------------------------------------------------

        # --- Debug: capture LoG blob points for overlay --------------------
        if roi is not None and getattr(self, "debug_show_log_blobs", False):
            pts_dbg = _log_blob_points(prev_s_gray, max_kp=80, p_hi=92.0)
            roi._debug_blob_pts = pts_dbg  # ROI-scale coords, can be None
        # -------------------------------------------------------------------

        # --------------------------------------------------------------

        # --- END NEW BLOCK ---
        # --- END NEW BLOCK ---

        dvx_dx = cv.Sobel(fx, cv.CV_32F, 1, 0, ksize=3) / 8.0
        dvy_dy = cv.Sobel(fy, cv.CV_32F, 0, 1, ksize=3) / 8.0
        div = dvx_dx + dvy_dy


        # 1) classic divergence-based vz (old behavior)
        if mask is not None and np.count_nonzero(mask) >= 16:
            vz_div_med = float(np.median(div[mask]))
        else:
            vz_div_med = float(np.median(div))

        vz_div = vz_div_med * (1.0 / SCALE)

        # confidence from divergence strength (kept for Z autoscale gating)
        try:
            self._vz_conf = float(np.median(np.abs(div))) * (1.0 / SCALE)
        except Exception:
            self._vz_conf = 0.0

        # 2) 2.5D curvature-depth from IG-LoG + flow
        vz25 = 0.0
        if roi is not None:
            try:
                vz25_med = _vz25_from_curvature_and_flow(curr_s_gray, fx, fy, mask)
                vz25 = vz25_med * (1.0 / SCALE)
            except Exception:
                vz25 = 0.0

        # 3) Choose how to combine them
        if roi is not None:
            mode = str(getattr(roi, "vz_mode", "hybrid")).lower()
            mix  = float(getattr(roi, "vz25_mix", 0.70))
            mix  = float(np.clip(mix, 0.0, 1.0))
        else:
            mode = "div"
            mix  = 0.0

        if mode == "curv":
            vz = vz25
        elif mode == "hybrid":
            vz = (1.0 - mix) * vz_div + mix * vz25
        else:  # "div" or unknown
            vz = vz_div

        # if roi is not None and roi._frame_lowconf:
        #     print(f"ROI '{roi.name}' low-confidence!!!")

        # --- HYBRID: override vx/vy back to HG (timing truth), keep vz from FB field ---
        if roi is not None and bool(roi.__dict__.get("_hyb_override_vxy", False)):
            try:
                vx_hg_s = float(roi.__dict__.get("_hyb_hg_vx_s", vx_med))
                vy_hg_s = float(roi.__dict__.get("_hyb_hg_vy_s", vy_med))
                roi.__dict__["_hyb_override_vxy"] = False  # consume flag
                roi._frame_lowconf = False
                return float(vx_hg_s) * (1.0 / SCALE), float(vy_hg_s) * (1.0 / SCALE), float(vz)
            except Exception:
                roi.__dict__["_hyb_override_vxy"] = False

        return vx_med*(1.0/SCALE), vy_med*(1.0/SCALE), vz


    
    def draw_global_arrow(self, vis, gx, gy):
        H, W = vis.shape[:2]
        cx = W - 50
        cy = 50

        sx = max(1.0, W / 2.0)
        sy = max(1.0, H / 2.0)
        nx = float(np.clip(gx / sx, -1.0, 1.0))
        ny = float(np.clip(gy / sy, -1.0, 1.0))

        k = 0.30 * min(W, H) * ARROW_SCALE
        ex = int(round(cx + k * nx))
        ey = int(round(cy + k * ny))

        mag = math.sqrt(nx*nx + ny*ny)
        th = max(1, int(1 + 10 * min(1.0, mag)))

        cv.arrowedLine(vis, (cx, cy), (ex, ey), (255,255,255), th, tipLength=0.18)
        cv.arrowedLine(vis, (cx+1, cy+1), (ex+1, ey+1), (0,0,0), max(1, th//2), tipLength=0.18)

        draw_text_clamped(vis, "GLOBAL", cx-34, cy+22, (220,220,220), 0.45)

    def draw_cmat_debug(self, vis, roi: ROI):
        mode = str(getattr(roi, "cmat_mode", "off")).lower()
        if mode == "off":
            return

        H, W = vis.shape[:2]
        x, y, w, h = map(int, roi.rect)

        if mode == "global":
            # whole-frame background sample region
            cv.rectangle(vis, (2, 2), (W - 3, H - 3), (80, 120, 255), 1, cv.LINE_AA)
            draw_text_clamped(vis, "CMAT: global frame", 10, 24, (200, 220, 255), 0.5)
            return

        if mode == "ring":
            r = int(max(4, getattr(roi, "bg_ring_px", 12)))
            rx, ry, rw, rh = x, y, w, h
            x0 = max(0, rx - r); y0 = max(0, ry - r)
            x1 = min(W, rx + rw + r); y1 = min(H, ry + rh + r)

            strips = [
                (x0, y0,        x1 - x0, r),           # top
                (x0, y1 - r,    x1 - x0, r),           # bottom
            ]
            h_mid = max(0, (y1 - y0) - 2 * r)
            if h_mid >= 4:
                strips += [
                    (x0,      y0 + r, r,       h_mid), # left
                    (x1 - r,  y0 + r, r,       h_mid), # right
                ]

            # tint the strips so you can see what pixels feed the background estimate
            for sx, sy, sw, sh in strips:
                overlay = vis.copy()
                cv.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (80, 255, 80), -1)
                cv.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

            # draw ring bounding box too
            cv.rectangle(vis, (x0, y0), (x1, y1), (80, 255, 80), 1, cv.LINE_AA)
            draw_text_clamped(vis, "CMAT: ring", x0 + 4, max(10, y0 - 6), (220, 255, 220), 0.45)


    def update_roi(self, roi: ROI, prev_gray, curr_gray):
        x,y,w,h = roi.rect; cx, cy = x + w/2.0, y + h/2.0
        pad = int(getattr(roi, "flow_pad_px", 0))
        if pad > 0:
            rx, ry, rw, rh = roi.rect
            rc = clamp_rect(rx - pad, ry - pad, rw + 2*pad, rh + 2*pad, self.W, self.H)
        else:
            rc = roi.rect
        v = self._roi_flow(prev_gray, curr_gray, rc, roi.fb_levels, debug_overlay=True, roi=roi)


        # v = self._roi_flow(prev_gray, curr_gray, roi.rect, roi.fb_levels)
        if v[0] is None: 
            roi.vx_ps = roi.vy_ps = roi.vz_rel_s = 0.0
            roi.last_speed_px_s = 0.0
            roi.last_center = (cx, cy)
            return roi


        vx, vy, vz = v
        # --- CMAT: subtract camera drift (optional) ---
        mode = str(getattr(roi, "cmat_mode", "off")).lower()
        if mode != "off":
            if mode == "ring":
                bgx, bgy, bgz = self._ring_drift(prev_gray, curr_gray, roi, int(getattr(roi, "bg_ring_px", 12)))
            else:
                bgx, bgy, bgz = self._global_drift(prev_gray, curr_gray)

            alpha  = float(np.clip(getattr(roi, "cmat_alpha", 1.0), 0.0, 1.0))
            tau_ms = float(getattr(roi, "cmat_tau_ms", 0.0))

            # LPF (optional). tau_ms<=0 => NO smoothing (best for screen shake)
            if tau_ms <= 0.0:
                gx, gy, gz = float(bgx), float(bgy), float(bgz)
            else:
                dt = 1.0 / max(1e-6, float(self.fps))
                a  = math.exp(-dt / max(1e-6, tau_ms / 1000.0))
                gx = a * float(getattr(roi, "_cmat_gx", 0.0)) + (1.0 - a) * float(bgx)
                gy = a * float(getattr(roi, "_cmat_gy", 0.0)) + (1.0 - a) * float(bgy)
                gz = a * float(getattr(roi, "_cmat_gz", 0.0)) + (1.0 - a) * float(bgz)

            roi._cmat_gx, roi._cmat_gy, roi._cmat_gz = gx, gy, gz

            # Apply subtraction ONCE
            gpx, gpy = alpha * gx, alpha * gy

            if str(getattr(roi, "cmat_proj", "full")).lower() == "orth":
                gate_deg = float(getattr(roi, "io_dir_deg", getattr(roi, "dir_gate_deg", 0.0)))
                ux, uy = math.cos(math.radians(gate_deg)), math.sin(math.radians(gate_deg))
                dot = gpx * ux + gpy * uy
                g_par_x, g_par_y = dot * ux, dot * uy
                g_perp_x, g_perp_y = gpx - g_par_x, gpy - g_par_y
                vx -= g_perp_x; vy -= g_perp_y
            else:
                vx -= gpx; vy -= gpy

            # Z (zoom proxy) subtraction ONCE
            vz -= alpha * gz
        # --- END CMAT ---

        vx_ps = float(vx) * self.fps; vy_ps = float(vy) * self.fps; vz_rel_s = float(vz) * self.fps
        if self.lagless:
            nx, ny = cx + vx, cy + vy
        else:
            nx = cx*(1.0 - LAG_LPF) + (cx + vx)*LAG_LPF
            ny = cy*(1.0 - LAG_LPF) + (cy + vy)*LAG_LPF
        
        # --- NEW: pull ROI center toward user anchor when confidence is high ---
        try:
            sim      = float(getattr(roi, "_anchor_last_sim", 0.0))
            ok       = bool(getattr(roi, "_anchor_last_ok", False))
            user_set = bool(getattr(roi, "anchor_user_set", False))
        except Exception:
            sim, ok, user_set = 0.0, False, False

        if user_set and ok and sim >= ANCHOR_MASK_THRESH:
            # anchor center in full-res coords (same mapping as draw_anchor_debug)
            ax = x + float(getattr(roi, "anchor_u", 0.5)) * w
            ay = y + float(getattr(roi, "anchor_v", 0.5)) * h

            # map similarity in [MASK, LOCK] → confidence in [0,1]
            denom = max(1e-6, (ANCHOR_LOCK_THRESH - ANCHOR_MASK_THRESH))
            conf  = float(np.clip((sim - ANCHOR_MASK_THRESH) / denom, 0.0, 1.0))

            # blend flow center and anchor center (do NOT touch vx,vy)
            nx = (1.0 - conf) * nx + conf * ax
            ny = (1.0 - conf) * ny + conf * ay
        # --- END NEW ---


        nx = float(np.clip(nx, 0, self.W-1)); ny = float(np.clip(ny, 0, self.H-1))
        # after nx, ny have been computed (including anchor pull)
        old_cx, old_cy = cx, cy   # center at start of frame
        jump_px = math.hypot(nx - old_cx, ny - old_cy)

        jump_thresh = float(getattr(roi, "anchor_jump_px", 0.0)) or 0.5 * max(w, h)

        just_teleported = jump_px > jump_thresh

        if just_teleported:
            # Mark as low confidence.
            roi._frame_lowconf = True

        rx = int(round(nx - w/2.0)); ry = int(round(ny - h/2.0))
        # --- AUTOSCALE FROM Z: symmetric + elastic ---
        w = int(roi.rect[2]); h = int(roi.rect[3])
        if str(getattr(roi, "z_scale_mode", "off")).lower() == "vz":
            # capture baseline size once
            if int(getattr(roi, "_w0", 0)) <= 0 or int(getattr(roi, "_h0", 0)) <= 0:
                roi._w0, roi._h0 = int(w), int(h)
                roi._z_log_s = 0.0; roi._z_vfilt = 0.0; roi._z_log_s0 = 0.0

            fps = max(1e-6, float(self.fps))
            dt  = 1.0 / fps

            # per-second area rate proxy (after CMAT Z subtraction above)
            zps = float(vz) * fps
            if abs(zps) < float(getattr(roi, "z_deadband_rel_s", 0.006)):
                zps = 0.0
            conf   = float(getattr(self, "_vz_conf", 0.0)) * fps
            floor  = 0.03  # tune: 0.02..0.06 s^-1
            if conf < floor: zps = 0.0


            # Z prefilter (stability)
            tau = max(1e-3, float(getattr(roi, "z_scale_tau_ms", 120)) / 1000.0)
            aZ  = math.exp(-dt / tau)
            vf  = aZ * float(getattr(roi, "_z_vfilt", 0.0)) + (1.0 - aZ) * zps
            roi._z_vfilt = vf

            # symmetric gain (grow == shrink)
            g = float(getattr(roi, "z_scale_gain", 1.10))

            # current log-size and (optional) slowly moving baseline
            log_s  = float(getattr(roi, "_z_log_s", 0.0))
            mode   = str(getattr(roi, "z_baseline_mode", "original")).lower()
            if mode == "slow_median":
                tau0 = max(1e-3, float(getattr(roi, "z_baseline_tau_ms", 3000)) / 1000.0)
                a0   = math.exp(-dt / tau0)
                # clamp contribution so wild frames don’t drag baseline
                tgt  = float(np.clip(log_s, -0.40, 0.40))
                log_s0 = a0 * float(getattr(roi, "_z_log_s0", 0.0)) + (1.0 - a0) * tgt
                roi._z_log_s0 = log_s0
            else:
                log_s0 = 0.0  # original size

            # elastic spring toward baseline + area→linear mapping (0.5 factor)
            k_ret = float(getattr(roi, "z_scale_return_s", 0.75))  # s^-1
            dlog  = (0.5 * g * vf - k_ret * (log_s - log_s0)) * dt

            # rate clamp (linear-size rate ±max_rate per second)
            max_rate = float(getattr(roi, "z_scale_max_rate_s", 2.00))
            dlog = float(np.clip(dlog, -max_rate * dt, +max_rate * dt))

            log_s += dlog
            s = float(np.exp(log_s))
            s = float(np.clip(s,
                            float(getattr(roi, "z_min_frac", 0.50)),
                            float(getattr(roi, "z_max_frac", 2.00))))
            roi._z_log_s = float(np.log(s))

            # apply (keep ≥8 px)
            w = int(max(8, round(roi._w0 * s)))
            h = int(max(8, round(roi._h0 * s)))
# --- END AUTOSCALE FROM Z ---


        
        roi.rect = clamp_rect_within(roi.bound, (rx, ry, w, h), self.W, self.H)

        # AOI enforcement (your cone/cos logic)
        vwx, vwy, vwz = _aoi_weight(vx_ps, vy_ps, vz_rel_s, roi)

        # --- HG psychoacoustic smoothing (salience-respecting) -----------------
        # Only smooth HG mode (or frames where HG was used), and never smear onsets:
        #   - if target speed increases -> pass through instantly (attack = 0 lag)
        #   - if target speed decreases -> EMA toward target (release smoothing)
        # Smoothing is gated off when salience is high (novelty or jerk proxy).
        if HG_PSY_SMOOTH_ENABLE:
            try:
                mm = str(getattr(roi, "motion_mode", "fb") or "fb").lower()
                hg_now = mm.startswith("hg") or bool(getattr(roi, "_hg_used", False))
                if hg_now:
                    # 1) novelty gate (already computed elsewhere; default 0)
                    novelty = float(getattr(roi, "_pe_novelty", 0.0))
                    denom = max(1e-6, float(PE_NOVELTY_THRESH) - 1.0)
                    nov_n = (novelty - 1.0) / denom
                    nov_n = float(np.clip(nov_n, 0.0, 1.0))

                    # 2) jerk proxy: change in target vector from last frame
                    st = roi.__dict__.setdefault("_psy_smooth", {
                        "vx": float(vwx), "vy": float(vwy),     # previous smoothed output
                        "tx": float(vwx), "ty": float(vwy),     # previous raw target
                    })
                    prev_tx, prev_ty = float(st.get("tx", 0.0)), float(st.get("ty", 0.0))
                    dv = float(math.hypot(float(vwx) - prev_tx, float(vwy) - prev_ty))

                    rx0, ry0, rw0, rh0 = map(int, roi.rect)
                    j_denom = max(1.0, float(HG_PSY_SMOOTH_JNORM) * float(min(rw0, rh0)) * float(self.fps))
                    j_n = float(np.clip(dv / j_denom, 0.0, 1.0))

                    # salience in [0,1] (1 => do not smooth)
                    S = max(nov_n, j_n)

                    # effective release smoothing (0 when salient)
                    alpha = float(HG_PSY_SMOOTH_ALPHA) * (1.0 - S)

                    prev_vx, prev_vy = float(st.get("vx", 0.0)), float(st.get("vy", 0.0))
                    prev_sp = prev_vx*prev_vx + prev_vy*prev_vy
                    tgt_sp  = float(vwx)*float(vwx) + float(vwy)*float(vwy)

                    if tgt_sp >= prev_sp:
                        out_vx, out_vy = float(vwx), float(vwy)  # instant attack
                    else:
                        out_vx = (1.0 - alpha) * prev_vx + alpha * float(vwx)
                        out_vy = (1.0 - alpha) * prev_vy + alpha * float(vwy)
                    
                    # --- HG burst scheduling: blend RAW vs SMOOTH to control habituation ---
                    if HG_BURST_ENABLE:
                        novelty = float(getattr(roi, "_pe_novelty", 0.0))
                        w_hg = _hg_burst_update(roi, float(self.fps), novelty, float(vwx), float(vwy))
                        # w_hg=1 => raw (flash), w_hg=0 => smooth (carrier)
                        out_vx = (1.0 - w_hg) * float(out_vx) + w_hg * float(vwx)
                        out_vy = (1.0 - w_hg) * float(out_vy) + w_hg * float(vwy)


                    st["vx"], st["vy"] = float(out_vx), float(out_vy)
                    st["tx"], st["ty"] = float(vwx), float(vwy)

                    vwx, vwy = float(out_vx), float(out_vy)
            except Exception:
                pass
        # --- END HG psychoacoustic smoothing ----------------------------------

        roi.vx_ps, roi.vy_ps, roi.vz_rel_s = float(vwx), float(vwy), float(vwz)

        roi.last_speed_px_s = float(math.hypot(vwx, vwy))
        roi.last_center = (nx, ny)
        return roi


    def preview_roi(self, roi: ROI, prev_gray, curr_gray):
        # keep signature; returns velocities without moving box
        x,y,w,h = roi.rect; cx, cy = x + w/2.0, y + h/2.0
        pad = int(getattr(roi, "flow_pad_px", 0))
        if pad > 0:
            rx, ry, rw, rh = roi.rect
            rc = clamp_rect(rx - pad, ry - pad, rw + 2*pad, rh + 2*pad, self.W, self.H)
        else:
            rc = roi.rect
        v = self._roi_flow(prev_gray, curr_gray, rc, roi.fb_levels)

        if v[0] is None: 
            roi.vx_ps=roi.vy_ps=roi.vz_rel_s=0.0; roi.last_center=(cx,cy); return roi
        vx, vy, vz = v
        vwx, vwy, vwz = _aoi_weight(float(vx)*self.fps, float(vy)*self.fps, float(vz)*self.fps, roi)
        roi.vx_ps, roi.vy_ps, roi.vz_rel_s = float(vwx), float(vwy), float(vwz)
        roi.last_center = (cx, cy)
        return roi

    def draw_arrow3(self, vis, roi: ROI):
        vx, vy, vz = roi.vx_ps, roi.vy_ps, roi.vz_rel_s
        x,y,w,h = roi.rect; cx, cy = int(x+w/2), int(y+h/2)
        sx = max(1.0, vis.shape[1]/2.0); sy = max(1.0, vis.shape[0]/2.0)
        nx = float(np.clip(vx / sx, -1.0, 1.0)); ny = float(np.clip(vy / sy, -1.0, 1.0))
        nz = float(np.clip(vz / (np.percentile([abs(vz),1.0], 95)), -1.0, 1.0))
        k = 0.35 * min(w, h) + 8
        ex = int(round(cx + k * nx * ARROW_SCALE)); ey = int(round(cy + k * ny * ARROW_SCALE))
        th = max(1, int(1 + 10 * min(1.0, math.sqrt(nx*nx + ny*ny))))
        cv.arrowedLine(vis, (cx,cy), (ex,ey), (255,255,255), th, tipLength=0.18)
        cv.arrowedLine(vis, (cx+1,cy+1), (ex+1,ey+1), (0,0,0), max(1, th//2), tipLength=0.18)
        zmag = abs(nz); r = max(4, int(5 + 14 * zmag))
        if nz >= 0:
            cv.circle(vis, (ex,ey), r, (0,210,0), -1, cv.LINE_AA); cv.circle(vis, (ex,ey), r, (20,20,20), 1, cv.LINE_AA)
        else:
            cv.circle(vis, (ex,ey), r, (255,0,255), 2, cv.LINE_AA)
            s = int(r*0.70); c=(255,0,255); lw=max(2, th//3)
            cv.line(vis, (ex-s,ey-s), (ex+s,ey+s), c, lw, cv.LINE_AA); cv.line(vis, (ex-s,ey+s), (ex+s,ey-s), c, lw, cv.LINE_AA)
    

    def _norm_dir(self, roi: ROI, vis):
        # identical normalization to draw_arrow3
        vx, vy, vz = roi.vx_ps, roi.vy_ps, roi.vz_rel_s
        sx = max(1.0, vis.shape[1]/2.0); sy = max(1.0, vis.shape[0]/2.0)
        nx = float(np.clip(vx / sx, -1.0, 1.0))
        ny = float(np.clip(vy / sy, -1.0, 1.0))
        nz = float(np.clip(vz / (np.percentile([abs(vz), 1.0], 95)), -1.0, 1.0))
        return nx, ny, nz

    def draw_anchor_debug(self, vis, roi: ROI):
        """
        Debug visualization for anchor patch:
          - draw anchor box inside ROI (green when locked, red when lost)
          - draw a tiny IG-LoG preview patch near the ROI
        """
        if not roi.anchor_user_set:
            return

        x, y, w, h = map(int, roi.rect)
        if w < 4 or h < 4:
            return

        H, W = vis.shape[:2]
        u = float(getattr(roi, "anchor_u", 0.5))
        v = float(getattr(roi, "anchor_v", 0.5))
        frac = float(getattr(roi, "anchor_size_frac", 0.45))

        # anchor box in full-res coords (approx)
        cx = x + u * w
        cy = y + v * h
        rad = 0.5 * frac * float(min(w, h))

        ax0 = int(max(0, round(cx - rad)))
        ax1 = int(min(W - 1, round(cx + rad)))
        ay0 = int(max(0, round(cy - rad)))
        ay1 = int(min(H - 1, round(cy + rad)))
        if ax1 <= ax0 or ay1 <= ay0:
            return

        # color: green if locked, red if lost
        ok = bool(getattr(roi, "_anchor_last_ok", False))
        sim = float(getattr(roi, "_anchor_last_sim", 0.0))
        col = (0, 220, 0) if ok else (0, 0, 255)

        # draw anchor rectangle
        cv.rectangle(vis, (ax0, ay0), (ax1, ay1), col, 2, cv.LINE_AA)
        draw_text_clamped(
            vis,
            f"anchor {('LOCK' if ok else 'LOST')}  sim={sim:.2f}",
            ax0 + 4,
            ay0 - 4,
            col,
            0.45
        )

        # build IG-LoG preview from the *current* frame region
        patch = vis[ay0:ay1, ax0:ax1]
        if patch.size == 0:
            return

        # grayscale and normalize
        pg = cv.cvtColor(patch, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        ig = _iglog_energy(pg)
        ig_norm = ig - ig.min()
        ig_norm /= (ig_norm.max() + 1e-9)
        ig_u8 = (ig_norm * 255.0).astype(np.uint8)
        ig_col = cv.applyColorMap(ig_u8, cv.COLORMAP_TURBO)

        # resize preview
        PREV_W = 72
        PREV_H = 72
        ig_prev = cv.resize(ig_col, (PREV_W, PREV_H), interpolation=cv.INTER_AREA)

        # --- SMART PLACEMENT: pick a free quadrant around ROI ---

        # zones to avoid (HUD area on top & metrics below ROI)
        HUD_HOT_ZONE_Y = 0.12 * H     # top 12% of screen (text usually here)
        METRIC_ZONE_Y = y + h + 40    # metrics under ROI

        candidates = []

        # Candidate 1 — right of ROI
        candidates.append((
            ax1 + 10,
            cy - PREV_H // 2,
            "right"
        ))
        # Candidate 2 — left of ROI
        candidates.append((
            ax0 - PREV_W - 10,
            cy - PREV_H // 2,
            "left"
        ))
        # Candidate 3 — above ROI
        candidates.append((
            cx - PREV_W // 2,
            ay0 - PREV_H - 10,
            "top"
        ))
        # Candidate 4 — below ROI
        candidates.append((
            cx - PREV_W // 2,
            ay1 + 10,
            "bottom"
        ))

        def is_valid(px, py):
            if px < 4 or py < 4: return False
            if px + PREV_W + 4 > W: return False
            if py + PREV_H + 4 > H: return False
            # avoid HUD text zone
            if py < HUD_HOT_ZONE_Y: return False
            # avoid metrics underneath ROI
            if METRIC_ZONE_Y < py < (METRIC_ZONE_Y + 120): return False
            return True

        # choose the first valid candidate, else fallback to bottom-left corner
        for px_cand, py_cand, label in candidates:
            if is_valid(px_cand, py_cand):
                px, py = int(px_cand), int(py_cand)
                break
        else:
            # fallback placement
            px = max(4, ax0 - PREV_W - 10)
            py = min(H - PREV_H - 4, ay1 + 20)


        overlay = vis.copy()
        cv.rectangle(overlay, (px - 2, py - 2), (px + PREV_W + 2, py + PREV_H + 2), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
        vis[py:py+PREV_H, px:px+PREV_W] = ig_prev
        # Draw faint line from anchor center to preview box
        cv.line(
            vis,
            (int(cx), int(cy)),           # anchor center
            (px + PREV_W // 2, py + PREV_H // 2),
            (200, 200, 200),
            1,
            cv.LINE_AA
        )

        draw_text_clamped(vis, "IG-LoG anchor", px + 4, py - 4, (230, 230, 230), 0.45)

    def draw_arrow_debug(self, vis, roi: ROI):
        # compact compass + numbers near the ROI
        x,y,w,h = map(int, roi.rect)
        panel = 86
        px = x - 6; py = y - 8 - panel       # try above‑left
        if py < 2: py = y + h + 8            # flip below if needed
        px = int(np.clip(px, 2, vis.shape[1]-panel-2))
        py = int(np.clip(py, 2, vis.shape[0]-panel-2))

        overlay = vis.copy()
        cv.rectangle(overlay, (px,py), (px+panel, py+panel), (22,22,22), -1)
        cv.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
        cv.rectangle(vis, (px,py), (px+panel, py+panel), (90,90,90), 1, cv.LINE_AA)

        # compass
        c  = (px+panel//2, py+panel//2-8)
        rad = panel//2 - 14
        cv.circle(vis, c, rad, (200,200,200), 1, cv.LINE_AA)
        cv.line(vis, (c[0]-rad, c[1]), (c[0]+rad, c[1]), (90,90,90), 1, cv.LINE_AA)
        cv.line(vis, (c[0], c[1]-rad), (c[0], c[1]+rad), (90,90,90), 1, cv.LINE_AA)

        nx, ny, nz = self._norm_dir(roi, vis)
        ex = int(round(c[0] + rad * nx)); ey = int(round(c[1] + rad * ny))
        cv.arrowedLine(vis, c, (ex,ey), (255,255,255), 1, tipLength=0.18)
        # nz indicator
        rr = max(3, int(5 + 9*abs(nz)))
        if nz >= 0: cv.circle(vis, (ex,ey), rr, (0,210,0), 1, cv.LINE_AA)
        else:       cv.circle(vis, (ex,ey), rr, (255,0,255), 1, cv.LINE_AA)

        # numbers (clamped to panel width)
        ang = math.degrees(math.atan2(ny, nx))
        spd = math.sqrt(roi.vx_ps*roi.vx_ps + roi.vy_ps*roi.vy_ps)
        draw_text_clamped(vis, f"vx {roi.vx_ps:+.2f}  vy {roi.vy_ps:+.2f}", px+6, py+panel-24, (240,240,200), 0.45)
        draw_text_clamped(vis, f"ang {ang:+.1f}\u00b0  nz {nz:+.2f}",   px+6, py+panel-8,  (220,220,220), 0.45)


    def draw_roi(self, vis, roi: ROI, active=False):
        x,y,w,h = roi.rect
        for t in range(2):
            cv.rectangle(vis, (x-t,y-t), (x+w+t,y+h+t),
                        (0,200,255) if not active else (0,0,255) if t==0 else (0,0,0), 1, cv.LINE_AA)
            
        # --- AI outline / occlusion mask overlay (visualize what was outlined) ---
        if bool(getattr(roi, "ai_outline_enabled", False)):
            polys = getattr(roi, "ai_outline_polys_norm", None) or []
            if polys:
                x0, y0, w0, h0 = map(int, roi.rect)
                mode = str(getattr(roi, "ai_outline_mode", "include")).lower()
                col = (0, 255, 255) if mode != "exclude" else (0, 80, 255)  # INC=cyan-ish, EXC=orange-ish

                # 1) draw polygon strokes in full-frame coords
                for poly in polys:
                    if not poly or len(poly) < 3:
                        continue
                    pts = []
                    for uv in poly:
                        if not isinstance(uv, (list, tuple)) or len(uv) < 2:
                            continue
                        u = float(uv[0]); v = float(uv[1])
                        px = int(round(x0 + np.clip(u, 0.0, 1.0) * (w0 - 1)))
                        py = int(round(y0 + np.clip(v, 0.0, 1.0) * (h0 - 1)))
                        pts.append([px, py])
                    if len(pts) >= 3:
                        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                        cv.polylines(vis, [pts], True, col, 2, cv.LINE_AA)

                # 2) translucent fill (lets you SEE the region, not just the boundary)
                try:
                    overlay = vis.copy()
                    for poly in polys:
                        if not poly or len(poly) < 3:
                            continue
                        pts = []
                        for uv in poly:
                            if not isinstance(uv, (list, tuple)) or len(uv) < 2:
                                continue
                            u = float(uv[0]); v = float(uv[1])
                            px = int(round(x0 + np.clip(u, 0.0, 1.0) * (w0 - 1)))
                            py = int(round(y0 + np.clip(v, 0.0, 1.0) * (h0 - 1)))
                            pts.append([px, py])
                        if len(pts) >= 3:
                            cv.fillPoly(overlay, [np.array(pts, dtype=np.int32)], col)

                    alpha = 0.18 if mode != "exclude" else 0.26  # EXC a bit stronger
                    cv.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0.0, vis)
                except Exception:
                    pass
        # --- end AI outline overlay ---


        # --- LoG blob debug overlay ------------------------------------
        if getattr(self, "debug_show_log_blobs", False):
            pts = getattr(roi, "_debug_blob_pts", None)
            scale = getattr(roi, "_roi_scale", None)
            if pts is not None and pts.size and scale is not None and scale > 0:
                for p in pts[:, 0, :]:
                    xs, ys = float(p[0]), float(p[1])
                    fx = int(round(x + xs / scale))
                    fy = int(round(y + ys / scale))
                    if 0 <= fx < vis.shape[1] and 0 <= fy < vis.shape[0]:
                        cv.circle(vis, (fx, fy), 2, (0, 255, 255), -1)
        # ----------------------------------------------------------------
        if roi.bound is not None: draw_dashed(vis, roi.bound, (0,0,255))
        labelL = f"L{roi.fb_levels}"
        # --- NEW: show current impact mode on the ROI (user clarity) ---
        m = str(getattr(roi, "impact_mode", "flux_dog")).lower()
        m_short = {
            "hybrid": "HYB",
            "fast": "FAST",
            "smooth": "SMTH",
            "flux_dog": "DoG",
            "flux_only": "DoG",
            "axis_jerk": "JERK",
            "flux_dog_jerk": "DoG+J",
            "flux_jerk": "DoG+J",
            "flux+jerk": "DoG+J",
            "reversal": "REV",
            "axis_min": "MIN",
            "flip_min": "MIN",
            "gauss_min": "MIN",
        }.get(m, m[:6].upper())
        trk = str(getattr(roi, "motion_mode", "fb")).lower()
        if trk.startswith("hyb"):
            trk_short = "HYB"
        elif trk.startswith("hg"):
            trk_short = "HG"
        else:
            trk_short = "FB"


        # Outline mask indicator (INC=include, EXC=exclude/occlusion)
        mask_tag = ""
        if bool(getattr(roi, "ai_outline_enabled", False)) and getattr(roi, "ai_outline_polys_norm", None):
            mm = str(getattr(roi, "ai_outline_mode", "include") or "include").lower()
            mask_tag = "  Mask " + ("EXC" if mm.startswith("exc") else "INC")
        labelRef = f"Ref {getattr(roi,'refractory_ms',140)}ms  Imp {m_short}  Trk {trk_short}{mask_tag}"

        scale = 0.45
        twL, th = _text_wh(labelL, scale); twR, _ = _text_wh(labelRef, scale)
        H, W = vis.shape[:2]
        ly = y - 6
        if ly < th + 6: ly = y + h + th + 4
        lx = int(np.clip(x, 6, W - (twL + 8 + twR) - 6))
        roi.labelL_hit  = draw_label_with_hitbox(vis, labelL,  lx,            ly, (255,230,180) if active else (220,220,220), scale)
        roi.labelRef_hit= draw_label_with_hitbox(vis, labelRef,lx + twL + 8,  ly, (200,255,200) if active else (210,210,210), scale)
        H, W = vis.shape[:2]
        # telemetry line (below the box; flip above if bottom is tight)
        # tele = f"v {roi.last_speed:5.2f}  a {roi.last_acc:5.2f}  j {roi.last_jerk:5.2f}"
        # ty = y + h + 14
        # if ty + 6 > H: ty = y - 6
        # draw_text_clamped(vis, tele, x+6, ty, (235,235,235), 0.42)

        # IMPACT flash
        if time.time() < getattr(roi, "_impact_flash_until", 0.0):
            draw_impact_flash(vis, roi)

       
        # Highlight & shake on impact flash
        if time.time() < getattr(roi, "_impact_flash_until", 0.0):
            is_out = (getattr(roi, "_impact_dir", 0) >= 0)
            col = (240,140,0) if is_out else (0,170,240)  # OUT=orange, IN=cyan (consistent with postview legend)
            # decaying jitter
            rem = float(roi._impact_flash_until - time.time())
            amp = int(max(1, min(5, round(6.0 * rem))))   # 0..~6 px
            jx = int(np.random.randint(-amp, amp+1)); jy = int(np.random.randint(-amp, amp+1))
            overlay = vis.copy()
            cv.rectangle(overlay, (x+jx,y+jy), (x+w+jx, y+h+jy), col, -1)
            cv.addWeighted(overlay, 0.12, vis, 0.88, 0, vis)
            cv.rectangle(vis, (x-2+jx,y-2+jy), (x+w+2+jx,y+h+2+jy), col, 2, cv.LINE_AA)

        _draw_io_ring_and_arrow(vis, roi)
        mode = str(getattr(roi, "axis_mode", "off"))
        pitch = float(getattr(roi, "axis_elev_deg", 0.0))
        draw_text_clamped(vis, f"AOI {mode}  pitch {pitch:+.0f}°", x+6, y-22, (220,220,220), 0.45)
        draw_text_clamped(vis, f"Z-Scale: {roi.z_scale_mode}", x+6, y-36, (220, 220, 220), 0.45)


# ---------- Export helpers ----------
def _envelope(seq, fps, atk_s=0.015, rel_s=0.060):
    atk = 1 - np.exp(-1/(fps*atk_s)); rel = 1 - np.exp(-1/(fps*rel_s))
    y=0.0; out=[]
    for x in seq:
        a = atk if x>y else rel
        y = (1-a)*y + a*x
        out.append(y)
    return np.array(out, float)

def _norm_sym_p95(a, gamma=1.0):
    a = np.asarray(a, float)
    p95 = np.percentile(np.abs(a), 95.0) or 1.0
    z = np.clip(a / p95, -1.0, 1.0)
    return np.sign(z) * (np.abs(z) ** gamma)

def _dir01(a, gamma=1.0):
    z = _norm_sym_p95(a, gamma=gamma)
    return 0.5 + 0.5*z

def _dir11(a, gamma=1.0):
    """Signed direction in -1..+1 (same normalization as _dir01, just not shifted)."""
    return _norm_sym_p95(a, gamma=gamma)

def _unit_from_deg(deg: float) -> Tuple[float,float]:
    th = math.radians(float(deg))
    return math.cos(th), math.sin(th)

def _proj_along(vx: np.ndarray, vy: np.ndarray, deg: float) -> np.ndarray:
    ux, uy = _unit_from_deg(deg)
    return (vx * ux) + (vy * uy)

def _quantize8_deg_from_xy(dx: float, dy: float) -> float:
    ang = math.degrees(math.atan2(dy, dx))  # +y down; consistent with screen
    # snap to multiples of 45°
    return (round(ang / 45.0) * 45.0) % 360.0


def _robust01(x, p_lo=5, p_hi=95):
    import numpy as _np
    x = _np.asarray(x, _np.float64)
    if x.size == 0:
        return x
    lo = float(_np.percentile(x, p_lo))
    hi = float(_np.percentile(x, p_hi))
    rng = max(hi - lo, 1e-12)
    y = (x - lo) / rng
    return _np.clip(y, 0.0, 1.0)


def _mag01(a, gamma=1.0):
    a = np.asarray(a, float); p95 = float(np.percentile(a, 95.0)) or 1.0
    return np.clip((a / p95) ** float(gamma), 0.0, 1.0)

def _deriv_central(y, dt):
    y = np.asarray(y, float); n=len(y)
    if n<3: return np.zeros(n, float)
    d=np.zeros(n,float)
    d[1:-1]=(y[2:]-y[:-2])/(2.0*dt)
    d[0]=d[1]; d[-1]=d[-2]
    return d

import numpy as _np

def _rolling_median(x, win):
    from numpy.lib.stride_tricks import sliding_window_view
    x = _np.asarray(x, _np.float64)
    if len(x) == 0: return x
    win = int(max(3, win | 1))
    pad = win//2
    xp = _np.pad(x, (pad,pad), mode='reflect')
    return _np.median(sliding_window_view(xp, win), axis=-1)

def _ema(x, alpha):
    y = _np.empty_like(x, dtype=_np.float64); y[0]=x[0]
    for i in range(1, len(x)): y[i] = alpha*x[i] + (1.0-alpha)*y[i-1]
    return y

def leaky_integrate(x, dt, tau_ms=800):
    x = _np.asarray(x, _np.float64)
    y = _np.zeros_like(x, _np.float64)
    if x.size == 0: return y
    a = max(0.0, min(1.0, 1.0 - (dt / max(1e-6, float(tau_ms) / 1000.0))))
    acc = 0.0
    for i, xi in enumerate(x):
        acc = a * acc + dt * float(xi)      # y[n] = (1 - dt/tau)*y[n-1] + dt*vz[n]
        y[i] = acc
    return y

def interpolate_over_dups(values, dup_flags):
    """
    Linearly interpolate over runs where dup_flags is True.
    values: 1D iterable
    dup_flags: same length iterable of bool
    """
    v = _np.asarray(values, _np.float64)
    d = _np.asarray(dup_flags, bool)
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


def robust_smooth(x, fps, win_ms=140, ema_tc_ms=200, mad_k=3.5):
    x = _np.asarray(x, _np.float64)
    if len(x)==0: return x
    med = _np.median(x); mad = _np.median(_np.abs(x-med)) + 1e-12
    lo = med - mad_k*1.4826*mad; hi = med + mad_k*1.4826*mad
    xc = _np.clip(x, lo, hi)
    win = max(5, int(round(win_ms*fps/1000.0)) | 1)
    xm = _rolling_median(xc, win)
    alpha = 1.0 - _np.exp(-(1.0/fps) / max(1e-3, ema_tc_ms/1000.0))
    return _ema(xm, alpha)

def robust_smooth_adaptive(x, fps,
                           *,
                           speed_ref=None,
                           # "slow" (current behavior)
                           slow_win_ms=140, slow_ema_ms=200, slow_mad_k=3.6,
                           # "fast" (preserve sign flips)
                           fast_win_ms=18,  fast_ema_ms=28,  fast_mad_k=5.0,
                           # decision threshold
                           big_motion_p95=None,
                           big_motion_mult=2.5):
    """
    Adaptive smoothing:
      - If motion magnitude is big -> use FAST windows (keeps reversals)
      - Else -> use SLOW windows (kills micro-jitter)

    speed_ref: array-like of per-frame speed (same length as x) OR None.
    big_motion_p95: explicit threshold in px/s; if None, computed as big_motion_mult * median(|x|) or from speed_ref.
    """
    import numpy as _np
    x = _np.asarray(x, _np.float64)
    if x.size == 0:
        return x

    # Use speed_ref if provided; otherwise estimate from |x|
    if speed_ref is not None:
        s = _np.asarray(speed_ref, _np.float64)
        s = s[:len(x)] if s.size >= len(x) else _np.pad(s, (0, len(x)-s.size), mode="edge")
        p95 = float(_np.percentile(_np.abs(s), 95.0))
    else:
        p95 = float(_np.percentile(_np.abs(x), 95.0))

    # Determine threshold for "big motion"
    if big_motion_p95 is None:
        thr = max(1e-6, big_motion_mult * float(_np.median(_np.abs(x)) + 1e-9))
        # if speed_ref exists, make threshold relative to that scale instead
        if speed_ref is not None:
            thr = max(thr, 0.35 * p95)
    else:
        thr = float(big_motion_p95)

    use_fast = (p95 >= thr)

    if use_fast:
        return robust_smooth(x, fps, win_ms=fast_win_ms, ema_tc_ms=fast_ema_ms, mad_k=fast_mad_k)
    else:
        return robust_smooth(x, fps, win_ms=slow_win_ms, ema_tc_ms=slow_ema_ms, mad_k=slow_mad_k)


def _catmull_rom(t, y, up=3):
    t = _np.asarray(t, _np.float64); y=_np.asarray(y, _np.float64)
    n=len(y); 
    if n<4 or up<=1: return t, y
    T=_np.zeros(n); 
    for i in range(1,n): T[i]=T[i-1]+_np.sqrt(abs(y[i]-y[i-1]))
    ti = _np.linspace(T[0], T[-1], (n-1)*up+1)
    yi = _np.empty_like(ti)
    def seg(p0,p1,p2,p3,t0,t1,t2,t3, u):
        m1=(p2-p0)/(t2-t0+1e-12); m2=(p3-p1)/(t3-t1+1e-12)
        uu=(u-t1)/(t2-t1+1e-12); u2=uu*uu; u3=u2*uu
        h00=2*u3-3*u2+1; h10=u3-2*u2+uu; h01=-2*u3+3*u2; h11=u3-u2
        return h00*p1 + h10*(t2-t1)*m1 + h01*p2 + h11*(t2-t1)*m2
    for i in range(1,n-2):
        m = (ti>=T[i]) & (ti<=T[i+1] if i<n-3 else ti>=T[i])
        yi[m]=seg(y[i-1],y[i],y[i+1],y[i+2], T[i-1],T[i],T[i+1],T[i+2], ti[m])
    f=(ti-ti[0])/(ti[-1]-ti[0]+1e-12)
    t_frames = t[0] + f*(t[-1]-t[0])
    return t_frames, yi

def normalize_unsigned01(x):
    x=_np.asarray(x,_np.float64)
    mn=_np.min(x); mx=_np.max(x); rng=max(1e-12, mx-mn)
    return (x-mn)/rng

def normalize_signed11(x):
    x=_np.asarray(x,_np.float64)
    mx=_np.max(_np.abs(x))+1e-12
    return x/mx

def smooth_and_scale_xy(cx, cy, fps, up=3):
    t = _np.arange(len(cx), dtype=_np.float64)/float(fps)
    tx, xs = _catmull_rom(t, cx, up); ty, ys = _catmull_rom(t, cy, up)
    xs = _np.interp(t, tx, xs); ys = _np.interp(t, ty, ys)
    return normalize_unsigned01(xs), normalize_unsigned01(ys)  # 0..1 envelopes

# ---- multi‑axis impact (MAD‑z score + refractory) ----
def _mad_z(x):
    x=_np.asarray(x,_np.float64); med=_np.median(x); mad=_np.median(_np.abs(x-med))+1e-12
    return 1.4826*(x-med)/mad

def _iglog_impulse_pairs(igdE: np.ndarray, lowconf: Optional[np.ndarray],
                         thr_pos: float = 2.8, thr_neg: float = 2.2) -> np.ndarray:
    """
    1-frame impulse: strong +dE at t and strong -dE at t+1 (biphasic).
    Returns indices t.
    """
    igdE = np.asarray(igdE, np.float64)
    n = igdE.size
    if n < 3:
        return np.asarray([], int)
    zpos = np.maximum(0.0, _mad_z(igdE))
    zneg = np.maximum(0.0, _mad_z(-igdE))
    hits = []
    for t in range(1, n-1):
        if lowconf is not None and (bool(lowconf[t]) or bool(lowconf[t+1])):
            continue
        if float(zpos[t]) >= thr_pos and float(zneg[t+1]) >= thr_neg:
            hits.append(t)
    return np.asarray(sorted(set(hits)), int)


def _iglog_smooth_rise(igE: np.ndarray, lowconf: Optional[np.ndarray],
                       fps: float, ema_ms: int = 140, thr_z: float = 2.4, persist_fr: int = 2) -> np.ndarray:
    """
    Smooth contacts: rising edge on EMA(E) with persistence.
    Returns event indices.
    """
    igE = np.asarray(igE, np.float64)
    n = igE.size
    if n < 6:
        return np.asarray([], int)

    dt = 1.0 / max(1e-6, float(fps))
    alpha = 1.0 - np.exp(-dt / max(1e-6, float(ema_ms)/1000.0))

    y = np.zeros(n, np.float64)
    y[0] = igE[0]
    for i in range(1, n):
        y[i] = alpha*igE[i] + (1.0-alpha)*y[i-1]

    dy = _deriv_central(y, dt)
    z = np.maximum(0.0, _mad_z(dy))

    hits = []
    for t in range(2, n-2):
        if lowconf is not None and bool(lowconf[t]):
            continue
        if float(z[t]) >= thr_z:
            ok = True
            for k in range(1, max(1, int(persist_fr))):
                if t+k >= n:
                    break
                if float(z[t+k]) < 0.65*thr_z:
                    ok = False
                    break
            if ok:
                hits.append(t)
    return np.asarray(sorted(set(hits)), int)


# ========= IMPACT DETECTION v2 (drop‑in replacements) =========

def impact_score_and_peaks(vx, vy, vz, ax, ay, az, jx, jy, jz, fps,
                           vmin_gate=0.02, thr=2.0, min_interval_ms=140):
    """
    Build a robust, unitless impact score:
      S = z+(|a|) + 1.1*z+(|j|) + 0.6*z+(d|v|/dt)
    Gated by a slow speed envelope to suppress static noise; smoothed ~60 ms.
    Returns score S and refractory‑aware local maxima >= thr.
    """
    import numpy as _np
    dt   = 1.0 / max(1e-6, float(fps))

    vx = _np.asarray(vx, _np.float64); vy = _np.asarray(vy, _np.float64); vz = _np.asarray(vz, _np.float64)
    ax = _np.asarray(ax, _np.float64); ay = _np.asarray(ay, _np.float64); az = _np.asarray(az, _np.float64)
    jx = _np.asarray(jx, _np.float64); jy = _np.asarray(jy, _np.float64); jz = _np.asarray(jz, _np.float64)

    vmag = _np.sqrt(vx*vx + vy*vy + vz*vz)
    amag = _np.sqrt(ax*ax + ay*ay + az*az)
    jmag = _np.sqrt(jx*jx + jy*jy + jz*jz)
    n = len(vmag)
    if n < 5:
        return _np.zeros(n, float), _np.asarray([], int)

    dvdt = _deriv_central(vmag, dt)

    def _mad_z_pos(x):
        z = _mad_z(x)
        return _np.maximum(0.0, z)

    S  = _mad_z_pos(amag)
    S += 1.1 * _mad_z_pos(jmag)
    S += 0.6 * _mad_z_pos(dvdt)

    # gentle gate on motion + warmup ramp for first ~250 ms
    vm_env = _ema(vmag, alpha=1.0 - _np.exp(-dt / 0.12))
    warm   = _np.clip(_np.arange(n) / max(1, int(round(0.25 * fps))), 0.0, 1.0)
    gate   = _np.clip(vm_env / max(1e-9, float(vmin_gate)), 0.0, 1.0) * (0.4 + 0.6*warm)
    S *= gate

    # smooth ~60 ms
    S = _ema(S, alpha=1.0 - _np.exp(-dt / 0.060))

    # pick local maxima with refractory
    k    = max(1, int(0.008 * fps))  # ~8 ms window for local peak check
    gap  = max(1, int(round(min_interval_ms * fps / 1000.0)))
    pk   = []
    last = -10**9
    for i in range(k, n - k):
        if S[i] >= thr and S[i] == _np.max(S[i - k:i + k + 1]) and (i - last) >= gap:
            pk.append(i); last = i
    return S, _np.asarray(pk, int)


def impact_score_cycles(vx, vy, vz, ax, ay, az, jx, jy, jz, fps,
                        vmin_gate=0.02, thr=2.0, min_interval_ms=140,
                        fall_frac=0.50, thr_lo_frac=None):
    """
    Hysteresis around each picked peak:
      peak >= thr  → “active”
      release when score <= max(fall_frac*peak, thr*thr_lo_frac [optional])
    Returns:
      S, peak_idx, release_idx, hold_mask (1 while active).
    """
    import numpy as _np
    S, _ = impact_score_and_peaks(vx, vy, vz, ax, ay, az, jx, jy, jz, fps,
                                  vmin_gate=vmin_gate, thr=thr, min_interval_ms=min_interval_ms)
    n = len(S)
    if n == 0:
        return S, _np.asarray([], int), _np.asarray([], int), _np.zeros(0, float)

    dt  = 1.0 / max(1e-6, float(fps))
    dS  = _deriv_central(S, dt)
    k   = max(1, int(0.008 * fps))
    gap = max(1, int(round(min_interval_ms * fps / 1000.0)))

    pk, rel = [], []
    hold = _np.zeros(n, float)
    state = 0
    last_pk = -10**9
    low_thr = 0.0
    start_i = -1

    for i in range(1, n - 1):
        if state == 0:
            if (S[i] >= thr and dS[i - 1] > 0 and dS[i] <= 0 and
                S[i] == _np.max(S[i - k:i + k + 1]) and (i - last_pk) >= gap):
                pk.append(i)
                last_pk = i
                start_i = i
                pt = float(S[i])
                low_thr = max(pt * float(fall_frac),
                              float(thr) * float(thr_lo_frac)) if (thr_lo_frac is not None) else pt * float(fall_frac)
                state = 1
        else:
            if S[i] <= low_thr:
                rel.append(i)
                hold[start_i:i + 1] = 1.0
                state = 0

    if state == 1 and start_i >= 0:
        hold[start_i:n] = 1.0

    return S, _np.asarray(pk, int), _np.asarray(rel, int), hold


def directional_impacts_from_reversal(
        vx_s, vy_s, vz_s, score, fps,
        gate_deg, thr_z, refractory_ms, pre_ms, lead_ms,
        v_zero=V_ZERO, decay_frac=0.20, both_dirs=True
) -> np.ndarray:
    """
    Detect reversals along gate_deg; for each reversal:
      - find score peak in [r-pre, r)
      - require score >= thr_z
      - pick first point after peak where score fell by decay_frac
      - place marker 'lead_ms' before that point
      - enforce global refractory between picks
    """
    vx_s = np.asarray(vx_s, float); vy_s = np.asarray(vy_s, float); score = np.asarray(score, float)
    n = min(len(vx_s), len(vy_s), len(score))
    if n < 5: return np.asarray([], int)

    v_gate = _proj_along(vx_s[:n], vy_s[:n], gate_deg)
    sgn = np.sign(np.where(np.abs(v_gate) > float(v_zero), v_gate, 0.0))


    rev_pos = [i for i in range(1, n) if (sgn[i - 1] > 0 and sgn[i] <= 0)]
    rev_neg = [i for i in range(1, n) if (sgn[i - 1] < 0 and sgn[i] >= 0)]
    if rev_neg or rev_pos:
        rev = sorted(set(rev_pos + rev_neg))
    else: return np.asarray([], int)

    min_gap = max(1, int(round(refractory_ms * fps / 1000.0)))
    pre  = max(0, int(round(pre_ms  * fps / 1000.0)))
    lead = max(0, int(round(lead_ms * fps / 1000.0)))

    picks = []
    for r in rev:
        a = max(0, r - pre); b = r
        if b <= a: 
            continue

        p_peak = a + int(np.argmax(score[a:b]))
        if score[p_peak] < float(thr_z):
            continue

        level = float(score[p_peak]) * (1.0 - float(decay_frac))
        q = p_peak
        for i in range(p_peak, b):
            if score[i] <= level:
                q = i; break

        p = max(0, int(q) - lead)
        if not picks or (p - picks[-1]) >= min_gap:
            picks.append(p)

    return np.asarray(picks, int)


def classify_in_out_by_dir(idx, vx_s, vy_s, fps, io_deg, io_in_sign=+1,
                           pre_ms=40, post_ms=60, v_zero=V_ZERO,
                           anchor_shift_ms=0):
    """
    Segment-based IN/OUT classification.

    1. Project velocity onto AoI → v_io.
    2. Build sign-based segments between zero crossings.
    3. Group candidate impact indices by segment id.
    4. For each segment:
       - If segment median velocity aligns with io_in_sign → IN.
       - If opposite → OUT.
    5. All peaks inside the same segment collapse to ONE event.
    """
    vx_s = np.asarray(vx_s, float)
    vy_s = np.asarray(vy_s, float)
    v_io = _proj_along(vx_s, vy_s, io_deg)

    N = len(v_io)
    if N == 0:
        return np.zeros(0, int), np.zeros(0, int)

    # thresholded sign: 0 in the ambiguity band
    sign = np.sign(v_io)
    sign[np.abs(v_io) < v_zero] = 0.0

    # build segment ids across time: runs of same non-zero sign
    seg_id = -np.ones(N, dtype=int)
    cur_id = -1
    prev_s = 0.0
    for t in range(N):
        s = sign[t]
        if s == 0.0:
            # sit inside current segment if any
            seg_id[t] = cur_id
        else:
            if s != prev_s:
                cur_id += 1
            seg_id[t] = cur_id
            prev_s = s

    idx = np.asarray(idx, int)
    idx = idx[(idx >= 0) & (idx < N)]
    if idx.size == 0:
        return np.zeros(0, int), np.zeros(0, int)

    in_list  = []
    out_list = []

    # how far around the segment to look for median velocity
    pre_win  = max(1, int(round(pre_ms  * fps / 1000.0)))
    post_win = max(1, int(round(post_ms * fps / 1000.0)))
    shift    = int(round(anchor_shift_ms * fps / 1000.0))

    for seg in np.unique(seg_id[idx]):
        if seg < 0:
            continue
        seg_peaks = np.sort(idx[seg_id[idx] == seg])
        if seg_peaks.size == 0:
            continue

        # segment span in time
        t0 = int(seg_peaks[0])
        t1 = int(seg_peaks[-1])

        # anchor around shifted center if desired
        center = int(np.clip((t0 + t1) // 2 + shift, 0, N - 1))
        a0 = max(0, center - pre_win)
        a1 = min(N, center + post_win + 1)

        v_seg = v_io[a0:a1]
        if v_seg.size == 0:
            continue

        med = float(np.median(v_seg))
        if abs(med) < v_zero:
            # ambiguous segment → ignore
            continue

        # decide polarity relative to "IN" direction
        if io_in_sign * med > 0.0:
            # IN segment: take earliest peak as IN
            in_list.append(int(seg_peaks[0]))
        else:
            # OUT segment: take earliest peak as OUT
            out_list.append(int(seg_peaks[0]))

    return np.asarray(out_list, int), np.asarray(in_list, int)


# ========= end IMPACT DETECTION v2 =========

def split_out_in(peaks, vP, fps, look_ms=40, v_zero=0.015):
    half=max(1,int(round((look_ms/1000.0)*fps))); out_idx=[]; in_idx=[]
    for p in peaks:
        vpre=_np.median(vP[max(0,p-half):p+1]); vpost=_np.median(vP[p:min(len(vP),p+half+1)])
        if abs(vpre)<v_zero: vpre=0.0
        if abs(vpost)<v_zero: vpost=0.0
        if vpre>0 and vpost<=0: out_idx.append(p)
        elif vpre<0 and vpost>=0: in_idx.append(p)
        else:
            (out_idx if _np.median(vP[max(0,p-2*half):min(len(vP),p+2*half+1)])>0 else in_idx).append(p)
    return _np.asarray(out_idx,int), _np.asarray(in_idx,int)

def _vec_to_dir8(vx: float, vy: float) -> int:
    # screen y grows down → flip vy for math direction
    ang_deg = math.degrees(math.atan2(-vy, vx))
    # map to 0..7: 0=E,1=NE,2=N,3=NW,4=W,5=SW,6=S,7=SE
    idx = int(np.round(ang_deg / 45.0)) % 8
    return idx

def _gate_by_dir8(peaks, rels, vx_s, vy_s, fps, selected_dir8, io_pref, look_ms=40, v_zero=0.02):
    """
    Pair each peak to its first following release (if any).
    Keep pairs whose peak-direction matches selected_dir8 (if set) 
    and whose OUT/IN matches io_pref (0 both, +1 OUT, -1 IN).
    Returns: filtered_peaks, filtered_rels
    """
    peaks = np.asarray(peaks, int); rels = np.asarray(rels, int)
    if (selected_dir8 is None) and (io_pref == 0):
        return peaks, rels

    # OUT/IN classification for peaks (by principal axis sign flip)
    vP = vx_s if np.mean(np.abs(vx_s)) >= np.mean(np.abs(vy_s)) else vy_s
    out_idx, in_idx = split_out_in(peaks, vP, fps, look_ms=look_ms, v_zero=v_zero)
    out_set, in_set = set(map(int,out_idx)), set(map(int,in_idx))

    half = max(1, int(round((look_ms/1000.0)*fps)))
    keep_p, keep_r = [], []
    r_iter = iter(sorted(map(int, rels.tolist())))
    next_r = next(r_iter, None)

    for p in map(int, peaks.tolist()):
        # peak local direction
        i0 = max(0, p - half); i1 = min(len(vx_s), p + half + 1)
        vx_m = float(np.median(vx_s[i0:i1] if i1>i0 else [0.0]))
        vy_m = float(np.median(vy_s[i0:i1] if i1>i0 else [0.0]))
        d8 = _vec_to_dir8(vx_m, vy_m)

        if selected_dir8 is not None:
            sd = int(selected_dir8) % 8
            ok = (d8 == sd) or (d8 == (sd + 1) % 8) or (d8 == (sd - 1) % 8)
            if not ok: continue

        io = (+1 if p in out_set else -1) if (p in out_set or p in in_set) else 0
        if io_pref != 0 and io != io_pref:
            continue

        # corresponding release = first release >= p
        r = None
        while next_r is not None and next_r < p:
            next_r = next(r_iter, None)
        if next_r is not None:
            r = next_r
            # advance for the next peak
            next_r = next(r_iter, None)

        keep_p.append(p)
        if r is not None: keep_r.append(int(r))

    return np.asarray(keep_p, int), np.asarray(keep_r, int)


# ---------- REAPER bridge (extended mapping) ----------
REAPER_BRIDGE_INBOX = os.environ.get("REAPER_BRIDGE_DIR")

def _discover_inbox_default():
    # 1) Env var
    if REAPER_BRIDGE_INBOX and os.path.isdir(REAPER_BRIDGE_INBOX):
        return REAPER_BRIDGE_INBOX
    # 2) Windows
    app = os.getenv("APPDATA", "")
    if app:
        p = os.path.join(app, "REAPER", "Scripts", "FluxBridge", "inbox")
        if os.path.isdir(p): return p
    # 3) macOS
    home = os.path.expanduser("~")
    mac = os.path.join(home, "Library", "Application Support", "REAPER", "Scripts", "FluxBridge", "inbox")
    if os.path.isdir(mac): return mac
    # 4) Linux
    lin = os.path.join(home, ".config", "REAPER", "Scripts", "FluxBridge", "inbox")
    if os.path.isdir(lin): return lin
    # 5) WSL hint
    try:
        root = "/mnt/c/Users"
        if os.path.isdir(root):
            for user in os.listdir(root):
                p = os.path.join(root, user, "AppData", "Roaming", "REAPER", "Scripts", "FluxBridge", "inbox")
                if os.path.isdir(p):
                    ready = os.path.join(p, "BRIDGE_READY.txt")
                    if os.path.isfile(ready): return p
    except Exception:
        pass
    p = os.path.abspath("./FluxBridge_inbox")
    os.makedirs(p, exist_ok=True)
    return p

def _bridge_inbox():
    p = _discover_inbox_default()
    os.makedirs(p, exist_ok=True); os.makedirs(os.path.join(p, "processed"), exist_ok=True)
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

def emit_reaper_push(csv_path: str,
                     t0: float,
                     t1: float,
                     scene_id: int,
                     roi_labels: List[str],
                     version: str = "v11_GUI_Update"):
    inbox = _bridge_inbox()

    # --- DEDUPE LABELS (ROOT CAUSE FIX) ---
    # Multiple physical ROIs can share the same label (e.g. "SECONDARY").
    # The CSV already aggregates them into one set of columns per label,
    # so we must only create ONE roi_track per label here.
    seen = set()
    unique_labels: List[str] = []
    for lab in roi_labels:
        if lab not in seen:
            seen.add(lab)
            unique_labels.append(lab)

    # per-ROI tracks (one per UNIQUE label)
    roi_tracks = []
    for label in unique_labels:
        roi_tracks.append(dict(
            name=label,
            columns=dict(
                env=f"{label}_flux_env",
                dirx=f"{label}_dirx01",
                diry=f"{label}_diry01",
                posx=f"{label}_posx01",
                posy=f"{label}_posy01",
                dirz=f"{label}_dirz01",
                posz=f"{label}_posz01",
                speed=f"{label}_speed01",
                acc=f"{label}_acc01",
                jerk=f"{label}_jerk01",
                impact_in=f"{label}_impact_in01",
                impact_out=f"{label}_impact_out01",
                impact_score=f"{label}_impact_score01",
            )
        ))

    # aggregate track mapping unchanged
    agg = dict(
        name=f"Aggregate_{version}",
        columns=dict(
            env="flux_env",
            dirx="agg_dirx01",
            diry="agg_diry01",
            posx="agg_posx01",
            posy="agg_posy01",
            dirz="agg_dirz01",
            posz="agg_posz01",
            velx="agg_velx01",
            vely="agg_vely01",
            velz="agg_velz01",
            accx="agg_accx01",
            accy="agg_accy01",
            accz="agg_accz01",
            jerkz="agg_jerkz01",
            impact="impact01",
        )
    )

    job = {
        "csv": os.path.abspath(csv_path).replace('/mnt/c/', 'C:\\').replace("\\", "/"),
        "start_sec": float(t0),
        "end_sec": float(t1),
        "scene_id": int(scene_id),
        "agg_track": agg,
        "roi_tracks": roi_tracks,
        "version": version,
    }

    lua = "return " + _py2lua(job) + "\n"
    fn  = os.path.join(inbox, f"push_scene{scene_id:02d}_{int(time.time())}_{version}.rpush.lua")
    tmp = fn + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(lua)
    os.replace(tmp, fn)
    print(f"[reaper] push → {fn}")


# ---------- CSV export ----------
def _sanitize(name: Optional[str], fallback: str) -> str:
    if not name: return fallback
    s = re.sub(r'[^a-zA-Z0-9_]+', '_', name.strip())
    s = re.sub(r'_+', '_', s).strip('_')
    return s or fallback

def _split_labels(name: Optional[str], fallback: str) -> List[str]:
    """
    Parse user-visible ROI.name into one or more machine-safe labels.

    Supported separators between aliases:
      - comma:      "hip_stroke, penetration"
      - newline:    "hip_stroke\npenetration"
      - semicolon:  "hip_stroke; penetration"

    Spaces are preserved inside each alias and sanitized later.
    Always returns at least one label.
    """
    raw = (name or "").strip()
    if not raw:
        return [_sanitize(None, fallback)]

    # Treat comma / newline / semicolon as separators
    parts = [p.strip() for p in re.split(r"[,\n;]+", raw) if p.strip()]
    if not parts:
        return [_sanitize(None, fallback)]

    labels: List[str] = []
    idx = 0
    for p in parts:
        fb = f"{fallback}_{idx}" if idx > 0 else fallback
        lab = _sanitize(p, fb)
        if lab and lab not in labels:
            labels.append(lab)
            idx += 1

    if not labels:
        labels = [_sanitize(None, fallback)]
    return labels


def export_fullpass_overlay_and_csv(video_path, scenes, suffix="_exported_overlay.mp4"):
    """
    Full-pass export → one MP4 with overlay and directional IMPACT flashes.
    Source of truth for IN/OUT:
      - prefer recorded live lanes (scene.roi_imp_in/out) when present
      - else run the offline detector on sampled series
    Visual policy:
      - hold the fill for EXPORT_HOLD_MS after each spike (global)
      - prefer OUT over IN when both are active at the same time
    No ROI/dataclass changes. CSV export is handled elsewhere.
    """
    # --- open input, size, fps ---
    cap = cv.VideoCapture(video_path); assert cap.isOpened(), f"Cannot open {video_path}"
    fps = float(cap.get(cv.CAP_PROP_FPS) or 30.0)
    N   = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    ok, fr0 = cap.read(); assert ok, "Failed to read first frame"
    H0, W0 = fr0.shape[:2]
    W = int(round(W0 * PROC_SCALE)); H = int(round(H0 * PROC_SCALE))
    if (W, H) != (W0, H0):
        fr0 = cv.resize(fr0, (W, H), interpolation=cv.INTER_AREA)
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    # --- clone ROI state per scene for sampling; create export scenes to hold series ---
    roi_states = [[roi_replace(r) for r in sc.rois] for sc in scenes]
    exp_scenes = [Scene(sc.start, sc.end, rois=[roi_replace(r) for r in sc.rois]) for sc in scenes]

    # also copy previously recorded live flags if any (used as first-choice ground truth)
    for si, sc in enumerate(scenes):
        if hasattr(sc, "roi_imp_in"):
            exp_scenes[si].roi_imp_in  = {k: list(v) for k, v in sc.roi_imp_in.items()}
            exp_scenes[si].roi_imp_out = {k: list(v) for k, v in sc.roi_imp_out.items()}

    # helper: which scene owns frame i (honors implicit end)
    def _scene_index_at_frame(i):
        for idx, sc in enumerate(scenes):
            eff_end = effective_end_for_scene(idx, scenes, N)
            if sc.start <= i <= eff_end:
                return idx
        return -1

    # ========== PASS 1: sample per-frame ROI centers/velocities ==========
    tracker = DeformTracker(W, H, fps)
    # export: allow heavier but more robust flow
    tracker.flow_pyr_levels = 2          # 2–3 = good sweet spot
    tracker.fb_consistency_mode = 1      # enable forward–backward consistency
    prev_gray = None
    for i in range(N):
        ok, fr = cap.read()
        if not ok:
            break
        if PROC_SCALE != 1.0:
            fr = cv.resize(fr, (W, H), interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)

        si = _scene_index_at_frame(i)
        if prev_gray is not None and si >= 0 and roi_states[si]:
            scx = exp_scenes[si]
            t = i / fps
            if not scx.times or scx.times[-1] != t:
                scx.times.append(t)

            # --- DUP: detect once per frame, store flag ---
            is_dup = is_near_duplicate_frame(prev_gray, gray)
            scx.dup_flags.append(bool(is_dup))

            if is_dup:
                # SKIP Farneback; carry over last state + curvature
                for ri, r in enumerate(roi_states[si]):
                    rr = roi_states[si][ri]
                    cx, cy = rr.last_center
                    vx, vy, vz = rr.vx_ps, rr.vy_ps, rr.vz_rel_s

                    scx.roi_cx.setdefault(ri, []).append(float(cx))
                    scx.roi_cy.setdefault(ri, []).append(float(cy))
                    scx.roi_vx.setdefault(ri, []).append(float(vx))
                    scx.roi_vy.setdefault(ri, []).append(float(vy))
                    scx.roi_vz.setdefault(ri, []).append(float(vz))

                    # curvature: reuse last value if exists
                    curv_list = scx.roi_curv.setdefault(ri, [])
                    last_curv = curv_list[-1] if curv_list else 0.0
                    curv_list.append(float(last_curv))
                    # IG-LoG: reuse last values on dup frames
                    igE_list  = scx.roi_igE.setdefault(ri, [])
                    igdE_list = scx.roi_igdE.setdefault(ri, [])
                    igE_list.append(float(igE_list[-1] if igE_list else 0.0))
                    igdE_list.append(float(igdE_list[-1] if igdE_list else 0.0))


                    # dup frame → not a ROI-level low-confidence; we already know it’s a dup
                    scx.roi_lowconf.setdefault(ri, []).append(False)

            else:
                # NORMAL update: Farneback + fresh curvature
                for ri, r in enumerate(roi_states[si]):
                    roi_states[si][ri] = tracker.update_roi(r, prev_gray, gray)
                    rr = roi_states[si][ri]
                    cx, cy = rr.last_center
                    vx, vy, vz = rr.vx_ps, rr.vy_ps, rr.vz_rel_s

                    scx.roi_cx.setdefault(ri, []).append(float(cx))
                    scx.roi_cy.setdefault(ri, []).append(float(cy))
                    scx.roi_vx.setdefault(ri, []).append(float(vx))
                    scx.roi_vy.setdefault(ri, []).append(float(vy))
                    scx.roi_vz.setdefault(ri, []).append(float(vz))

                    rect = rr.rect  # ROI rect in full-res coords
                    cval = compute_roi_curvature(prev_gray, gray, rect)
                    scx.roi_curv.setdefault(ri, []).append(float(cval))
                    # IG-LoG structural evidence (E and dE)
                    igE = compute_roi_iglog_energy(gray, rect)
                    prevE = float(getattr(rr, "_iglog_E_prev", 0.0))
                    dE = float(igE - prevE)
                    rr._iglog_E_prev = float(igE)
                    scx.roi_igE.setdefault(ri, []).append(float(igE))
                    scx.roi_igdE.setdefault(ri, []).append(float(dE))

                    # per-frame low-confidence flag from _roi_flow(...)
                    lc = bool(getattr(rr, "_frame_lowconf", False))
                    scx.roi_lowconf.setdefault(ri, []).append(lc)


        prev_gray = gray
    cap.release()


    # ========== PASS 1.5: compute flags + held masks ==========
    def _dedupe_flags(flags, combine_fr):
        """Merge spikes closer than combine_fr frames."""
        flags = np.asarray(flags, float)
        idx = np.where(flags > 0.5)[0]
        if idx.size <= 1:
            return np.asarray(flags, float)
        keep = [int(idx[0])]
        for j in map(int, idx[1:]):
            if j - keep[-1] >= int(combine_fr):
                keep.append(j)
        out = np.zeros_like(flags, float)
        out[keep] = 1.0
        return out

    def _apply_dir_refractory(idx, gap_fr):
        idx = np.asarray(idx, int)
        if idx.size <= 1:
            return idx
        keep = [int(idx[0])]
        for j in map(int, idx[1:]):
            if j - keep[-1] >= int(gap_fr):
                keep.append(j)
        return np.asarray(keep, int)

    def _holdify(spikes, hold_fr):
        """Dilate 0/1 spikes by a fixed hold window."""
        if spikes is None:
            return None
        s = np.asarray(spikes, float)
        out = np.zeros_like(s, float)
        idx = np.where(s > 0.5)[0]
        for p in map(int, idx):
            q = min(len(out), p + int(hold_fr))
            out[p:q] = 1.0
        return out

    hold_fr  = max(1, int(round((EXPORT_HOLD_MS / 1000.0) * fps)))
    combine  = max(1, int(round(0.030 * fps)))  # 30 ms spike merge
    # refractory per dir falls back to r.refractory_ms
    flags_per_scene = []   # [{ri: (fi_spk, fo_spk)}]
    holds_per_scene = []   # [{ri: (fi_hold, fo_hold)}]

    for si, sc in enumerate(exp_scenes):
        per_roi_flags = {}
        per_roi_holds = {}

        for ri, r in enumerate(sc.rois):
            # pull series for this ROI
            vx = np.asarray(sc.roi_vx.get(ri, []), float)
            vy = np.asarray(sc.roi_vy.get(ri, []), float)
            vz = np.asarray(sc.roi_vz.get(ri, []), float)
            # optional: treat dup + low-confidence frames as interpolation spans
            flags = None

            if getattr(sc, "dup_flags", None) and len(sc.dup_flags) == len(vx):
                flags = np.asarray(sc.dup_flags, bool)

            low = sc.roi_lowconf.get(ri, [])
            if low and len(low) == len(vx):
                low = np.asarray(low, bool)
                flags = low if flags is None else (flags | low)

            if flags is not None:
                vx = interpolate_over_dups(vx, flags)
                vy = interpolate_over_dups(vy, flags)
                vz = interpolate_over_dups(vz, flags)

            T  = min(len(vx), len(vy), len(vz))
            if T < 5:
                per_roi_flags[ri] = (np.zeros(T, float), np.zeros(T, float))
                per_roi_holds[ri] = (np.zeros(T, float), np.zeros(T, float))
                continue

            # --- NEW: center-based velocities from ROI positions ---
            cx_seq = sc.roi_cx.get(ri, [])
            cy_seq = sc.roi_cy.get(ri, [])
            vx_pos_raw, vy_pos_raw = _center_vel_from_pos(cx_seq, cy_seq, fps)
            # fit length to T
            vx_pos_raw = vx_pos_raw[:T]
            vy_pos_raw = vy_pos_raw[:T]
            # ------------------------------------------------------

            # robust smoothing
            # Build a speed reference so "big motion" is detected correctly
            speed_ref = np.sqrt(vx[:T]*vx[:T] + vy[:T]*vy[:T] + vz[:T]*vz[:T])

            vx_s = robust_smooth_adaptive(vx[:T], fps, speed_ref=speed_ref)
            vy_s = robust_smooth_adaptive(vy[:T], fps, speed_ref=speed_ref)

            # Z can stay a bit smoother (usually noisy), but still adaptive
            vz_s = robust_smooth_adaptive(vz[:T], fps, speed_ref=speed_ref,
                                        slow_win_ms=160, slow_ema_ms=240,
                                        fast_win_ms=24,  fast_ema_ms=40)


            # robust smoothing (center-based)
            vx_pos_s = robust_smooth(vx_pos_raw, fps, win_ms=140, ema_tc_ms=200, mad_k=3.6)
            vy_pos_s = robust_smooth(vy_pos_raw, fps, win_ms=140, ema_tc_ms=200, mad_k=3.6)

            # prefer recorded live flags if present
            fi_live = np.asarray(exp_scenes[si].roi_imp_in.get(ri, []),  float)
            fo_live = np.asarray(exp_scenes[si].roi_imp_out.get(ri, []), float)
            use_live = (fi_live.size or fo_live.size)

            # --- NEW: decide when to trust the center channel ---
            use_pos_channel = (
                bool(getattr(r, "anchor_user_set", False)) or
                str(getattr(r, "cmat_mode", "off")).lower() != "off"
            )
            # -----------------------------------------------------


            if use_live:
                # fit length to T
                def _fit(a):
                    a = np.asarray(a, float)
                    if a.size < T: a = np.pad(a, (0, T - a.size))
                    if a.size > T: a = a[:T]
                    return a
                fi = _fit(fi_live)
                fo = _fit(fo_live)

                # offline score for reference
                S_any, _, _ = _impacts_for_mode(r, vx_s, vy_s, vz_s, fps)

                # optional: fold center motion into score even when live flags were used
                if use_pos_channel and len(vx_pos_s) == len(S_any):
                    S_pos, _, _ = _impacts_for_mode(r, vx_pos_s, vy_pos_s,
                                                    np.zeros_like(vx_pos_s), fps)
                    S_any = np.maximum(S_any, S_pos)
            else:
                # offline directional impacts (mode-aware): FLOW channel
                S_flow, in_flow, out_flow = _impacts_for_mode(r, vx_s, vy_s, vz_s, fps)

                if use_pos_channel and len(vx_pos_s) >= 5:
                    # anchor-center channel
                    S_pos, in_pos, out_pos = _impacts_for_mode(
                        r,
                        vx_pos_s,
                        vy_pos_s,
                        np.zeros_like(vx_pos_s),
                        fps,
                    )
                    # choose primary (anchor vs flow), merge events + defrag
                    S_any, in_idx, out_idx = _pick_primary_impact_channel(
                        r, fps,
                        vx_s,     vy_s,     S_flow, in_flow, out_flow,
                        vx_pos_s, vy_pos_s, S_pos,  in_pos,  out_pos,
                    )
                else:
                    # no anchor-center channel; just flow
                    S_any = S_flow
                    in_idx  = np.asarray(in_flow,  int)
                    out_idx = np.asarray(out_flow, int)

                # dir-specific refractory (fallback to refractory_ms)
                gap_in  = max(1, int(round(float(getattr(r, "refractory_in_ms",
                                                        getattr(r, "refractory_ms", 140))) * fps / 1000.0)))
                gap_out = max(1, int(round(float(getattr(r, "refractory_out_ms",
                                                        getattr(r, "refractory_ms", 140))) * fps / 1000.0)))
                mode = str(getattr(r, "impact_mode", "flux_dog")).lower()
                if mode == "fast":
                    gap_in = gap_out = 1  # ~no refractory (1 frame)
                in_idx  = _apply_dir_refractory(in_idx,  gap_in)
                out_idx = _apply_dir_refractory(out_idx, gap_out)

                fi = np.zeros(T, float); fo = np.zeros(T, float)
                fi[in_idx]  = 1.0
                fo[out_idx] = 1.0

            # TURN SPIKES INTO SEGMENTS: hold until direction flip
            fi, fo = _extend_impacts_to_dir_flip(r, vx_s, vy_s, vz_s, fi, fo, fps)

            fi_h = _holdify(fi, hold_fr)
            fo_h = _holdify(fo, hold_fr)
            per_roi_flags[ri] = (fi, fo)
            per_roi_holds[ri] = (fi_h, fo_h)

        flags_per_scene.append(per_roi_flags)
        holds_per_scene.append(per_roi_holds)

    # ========== PASS 2: render overlay with held flashes ==========
    out_path = os.path.splitext(os.path.basename(video_path))[0] + suffix
    cap = cv.VideoCapture(video_path); assert cap.isOpened()
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    ff = _open_ffmpeg_writer(out_path, W, H, fps)
    if ff is None:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        vw = cv.VideoWriter(out_path, fourcc, fps, (W, H))

    draw_tracker = DeformTracker(W, H, fps)

    for i in range(N):
        ok, fr = cap.read()
        if not ok:
            break
        if PROC_SCALE != 1.0:
            fr = cv.resize(fr, (W, H), interpolation=cv.INTER_AREA)
        hud = fr.copy()

        si = _scene_index_at_frame(i)
        if si >= 0 and exp_scenes[si].times:
            t_idx = i - scenes[si].start  # local index within the scene
            scx   = exp_scenes[si]

            for ri, r0 in enumerate(scx.rois):
                # need a sample at this index
                cx_seq = scx.roi_cx.get(ri, []); vx_seq = scx.roi_vx.get(ri, [])
                if t_idx < 0 or t_idx >= len(vx_seq) or t_idx >= len(cx_seq):
                    continue

                cx = float(scx.roi_cx[ri][t_idx]); cy = float(scx.roi_cy[ri][t_idx])
                vx = float(scx.roi_vx[ri][t_idx]); vy = float(scx.roi_vy[ri][t_idx]); vz = float(scx.roi_vz[ri][t_idx])

                # rebuild ROI pose for this frame
                w, h = r0.rect[2], r0.rect[3]
                rx = int(round(cx - w / 2.0)); ry = int(round(cy - h / 2.0))
                rect = clamp_rect(rx, ry, w, h, W, H)
                rr = roi_replace(r0, rect=rect, last_center=(cx, cy), vx_ps=vx, vy_ps=vy, vz_rel_s=vz)

                # Draw ROI + arrows as in live HUD
                draw_tracker.draw_roi(hud, rr, active=False)
                if draw_tracker.show_arrows:
                    draw_tracker.draw_arrow3(hud, rr)

                # Held IN/OUT overlays (prefer OUT)
                fi_h, fo_h = holds_per_scene[si].get(ri, (None, None))
                out_on = (fo_h is not None and 0 <= t_idx < len(fo_h) and fo_h[t_idx] > 0.5)
                in_on  = (fi_h is not None and 0 <= t_idx < len(fi_h) and fi_h[t_idx] > 0.5)

                if out_on or in_on:
                    x, y, ww, hh = rr.rect
                    overlay = hud.copy()
                    if out_on:
                        cv.rectangle(overlay, (x, y), (x + ww, y + hh), (240, 140, 0), -1)
                        cv.addWeighted(overlay, 0.12, hud, 0.88, 0, hud)
                        put_text_centered(hud, "IMPACT OUT",  (x, y, ww, hh), IMPACT_BLUE_BOLD)
                    elif in_on:
                        cv.rectangle(overlay, (x, y), (x + ww, y + hh), (0, 170, 240), -1)
                        cv.addWeighted(overlay, 0.12, hud, 0.88, 0, hud)
                        put_text_centered(hud, "IMPACT IN",  (x, y, ww, hh), IMPACT_BLUE_BOLD)

        # progress lane
        px = int(round((i / max(1, N - 1)) * (W - 1)))
        cv.rectangle(hud, (0, H - 6), (px, H - 3), (90, 150, 200), -1)
        cv.rectangle(hud, (0, H - 6), (W - 1, H - 3), (60, 60, 60), 1)

        if ff is not None:
            _ff_write(ff, hud)
        else:
            vw.write(hud)

    if ff is not None:
        _ff_close(ff)
    else:
        vw.release()
    cap.release()

    print(f"[export] overlay (held impacts, {EXPORT_HOLD_MS} ms) → {out_path}")
    return out_path


# ============================== MIDI export ==============================
# Goal: DAW-agnostic output. CSV is great for Reaper; MIDI CC makes every DAW usable.
#
# Strategy:
#   - One MIDI track per ROI label group (plus a GLOBAL track).
#   - CC values are 0..127.
#   - Time is encoded in ticks using BPM=60 so 1 beat == 1 second (easy to reason about).
#     Change BPM on import if you want musical tempo instead of wall-clock time.

def _midi_vlq(n: int) -> bytes:
    """Encode an int as MIDI variable-length quantity."""
    n = int(max(0, n))
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))

def _midi_meta(meta_type: int, data: bytes) -> bytes:
    return bytes([0xFF, int(meta_type) & 0xFF]) + _midi_vlq(len(data)) + data

def _midi_event(status: int, data: bytes) -> bytes:
    return bytes([int(status) & 0xFF]) + data

def _midi_cc(channel: int, cc: int, value: int) -> bytes:
    ch = int(channel) & 0x0F
    cc = int(cc) & 0x7F
    value = int(np.clip(int(value), 0, 127))
    return _midi_event(0xB0 | ch, bytes([cc, value]))

def _midi_track_chunk(events_abs: List[Tuple[int, bytes]]) -> bytes:
    """Build a single MTrk chunk from (tick, bytes) events."""
    events_abs = sorted(events_abs, key=lambda x: int(x[0]))
    out = bytearray()
    last = 0
    running_status = None
    for tick, ev in events_abs:
        tick = int(max(0, tick))
        dt = tick - last
        last = tick
        out += _midi_vlq(dt)

        # Minimal running-status support (only for CC streams).
        status = ev[0]
        if running_status == status and status < 0xF0:
            out += ev[1:]  # drop status byte
        else:
            out += ev
            running_status = status if status < 0xF0 else None

    # End of track
    out += _midi_vlq(0) + _midi_meta(0x2F, b"")
    data = bytes(out)
    return b"MTrk" + struct.pack(">I", len(data)) + data

def write_midi_file(path: str, ppq: int, tracks: List[List[Tuple[int, bytes]]]) -> None:
    """Write a Type-1 MIDI file."""
    ppq = int(ppq)
    tracks = list(tracks)
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), ppq)
    chunks = [header]
    for evs in tracks:
        chunks.append(_midi_track_chunk(evs))
    with open(path, "wb") as f:
        for ch in chunks:
            f.write(ch)

def _cc_range_kind(col: str) -> str:
    """Return a hint for how to map this column into CC."""
    col = str(col)
    if any(col.endswith(suf) for suf in ("_axis_v", "_axis_acc", "_axis_jerk", "_axis_dir",
                                        "_lat_v", "_lat_acc", "_lat_jerk", "_lat_dir",
                                        "_cam_log2_signed")):
        return "bipolar"
    if col.endswith("_deg"):
        return "degrees"
    if col.endswith("01") or col.endswith("_flux_env") or col.endswith("_entropy01"):
        return "unipolar"
    # Default: try to treat as unipolar (already normalized in this tool).
    return "unipolar"

def _to_cc_value(x: float, kind: str) -> int:
    if x is None:
        return 0
    try:
        if math.isnan(float(x)):
            return 0
    except Exception:
        pass
    v = float(x)
    kind = str(kind)
    if kind == "bipolar":
        v = np.clip(v, -1.0, 1.0)
        return int(round((v * 0.5 + 0.5) * 127.0))
    if kind == "degrees":
        # Assume -180..+180 (or -90..+90); map by clamping.
        v = np.clip(v, -180.0, 180.0)
        return int(round(((v + 180.0) / 360.0) * 127.0))
    # unipolar
    v = np.clip(v, 0.0, 1.0)
    return int(round(v * 127.0))

_LOCAL_CC_MAP = {
    # continuous
    "flux_env": 20,
    "entropy01": 21,
    "speed01": 22,
    "acc01": 23,
    "jerk01": 24,
    "posx01": 25,
    "posy01": 26,
    "posz01": 27,
    "dirx01": 28,
    "diry01": 29,
    "dirz01": 30,
    "axis_v": 31,
    "axis_acc": 32,
    "axis_jerk": 33,
    "axis_dir": 34,
    "lat_v": 35,
    "lat_acc": 36,
    "lat_jerk": 37,
    "lat_dir": 38,
    "lat_amp01": 39,
    # impacts
    "impact_score01": 40,
    "impact_in01": 41,
    "impact_out01": 42,
    "impact_in_spk01": 43,
    "impact_out_spk01": 44,
    "impact_hit_spk01": 45,
    "impact_hit_in_spk01": 46,
    "impact_hit_out_spk01": 47,
}

def _local_lane_key(col: str) -> Optional[str]:
    """Map a local CSV column name to a lane key used by _LOCAL_CC_MAP."""
    col = str(col)
    if col == "flux_env":
        return "flux_env"
    if col == "agg_entropy01":
        return "entropy01"
    if col.startswith("agg_"):
        # Let GLOBAL track assign sequential CCs for aggregate extras.
        return None

    # Label-prefixed lanes: <label>_<lane>
    for lane in (
        "flux_env", "entropy01",
        "speed01", "acc01", "jerk01",
        "posx01", "posy01", "posz01",
        "dirx01", "diry01", "dirz01",
        "axis_v", "axis_acc", "axis_jerk", "axis_dir",
        "lat_v", "lat_acc", "lat_jerk", "lat_dir", "lat_amp01",
        "impact_in01", "impact_out01",
        "impact_in_spk01", "impact_out_spk01",
        "impact_score01",
        "impact_hit_spk01", "impact_hit_in_spk01", "impact_hit_out_spk01",
    ):
        if col.endswith("_" + lane):
            return lane

    # Per-label "role" lanes: <label>_imp_PRIMARY_<in01|out01|score01> etc.
    for lane in ("in01", "out01", "score01"):
        if col.endswith("_" + lane):
            # map to generic lane key; CC will be allocated sequentially per track
            return lane
    return None

# ============================== MIDI export (seek-safe option) ==============================

MIDI_SEEK_SAFE_DEFAULT = True      # dense CC grid for scrub/jump friendliness
MIDI_SEEK_HZ_DEFAULT   = 24.0      # 20–30 is the sane range; 24 keeps files reasonable
MIDI_SEEK_EPS_CC       = 1         # only emit when CC changes by >= this many steps

def _is_spike_lane(col: str) -> bool:
    c = str(col).lower()
    return ("_spk01" in c) or c.endswith("_spk") or ("hit_spk" in c)

def _resample_to_grid(t_src: np.ndarray, x_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation resampler. t_src must be ascending."""
    t_src = np.asarray(t_src, np.float64)
    x_src = np.asarray(x_src, np.float64)
    t_grid = np.asarray(t_grid, np.float64)

    if t_src.size == 0 or t_grid.size == 0:
        return np.zeros_like(t_grid, dtype=np.float64)
    if x_src.size != t_src.size:
        # hard fit
        n = min(x_src.size, t_src.size)
        t_src = t_src[:n]
        x_src = x_src[:n]
        if n == 0:
            return np.zeros_like(t_grid, dtype=np.float64)

    # If time duplicates exist, force monotonic by tiny nudges
    # (rare, but protects np.interp)
    dt = np.diff(t_src)
    if np.any(dt <= 0):
        t_fix = t_src.copy()
        for i in range(1, len(t_fix)):
            if t_fix[i] <= t_fix[i-1]:
                t_fix[i] = t_fix[i-1] + 1e-9
        t_src = t_fix

    return np.interp(t_grid, t_src, x_src, left=x_src[0], right=x_src[-1])


def write_local_midi_from_rows(rows: Dict[str, List[Any]], midi_path: str, mapping_path: str,
                              bpm: float = 60.0, ppq: int = 960,
                              track_name_prefix: str = "ROI",
                              *,
                              seek_safe: bool = MIDI_SEEK_SAFE_DEFAULT,
                              seek_hz: float = MIDI_SEEK_HZ_DEFAULT,
                              seek_eps_cc: int = MIDI_SEEK_EPS_CC,
                              time_zero_sec: Optional[float] = None) -> None:

    """Write a MIDI file (and JSON mapping) from the local tracker export rows."""
    if not rows or "time" not in rows:
        return
    t = list(rows.get("time") or [])
    if not t:
        return

    # timing:
    # - if time_zero_sec is provided (e.g. 0.0), MIDI times become absolute-to-video/project
    # - otherwise keep current behavior (scene-local)
    if time_zero_sec is None:
        t0 = float(t[0])
    else:
        t0 = float(time_zero_sec)

    t_rel = [float(tt) - t0 for tt in t]


    # Group columns: GLOBAL + one track per label prefix (best-effort).
    cols = [c for c in rows.keys() if c != "time"]
    global_cols = []
    groups: Dict[str, List[str]] = {}

    # Known explicit lane suffixes (longest-first) for stable label extraction.
    lane_suffixes = sorted(
        set(["_" + k for k in _LOCAL_CC_MAP.keys()] + [
            "_in01", "_out01", "_score01",
        ]),
        key=len,
        reverse=True
    )

    for col in cols:
        if col.startswith("agg_") or col in ("flux_env", "agg_entropy01"):
            global_cols.append(col)
            continue

        label = None
        for suf in lane_suffixes:
            if col.endswith(suf):
                label = col[:-len(suf)]
                break
        if not label:
            global_cols.append(col)
            continue
        groups.setdefault(label, []).append(col)

    # Meta track (tempo, time sig)
    tempo_us = int(round(60_000_000.0 / float(bpm)))
    meta_track = [
        (0, _midi_meta(0x03, b"MotionTracker META")),
        (0, _midi_meta(0x51, tempo_us.to_bytes(3, "big", signed=False))),
        (0, _midi_meta(0x58, bytes([4, 2, 24, 8]))),
    ]
    tracks: List[List[Tuple[int, bytes]]] = [meta_track]
    mapping = {
        "tool": "local",
        "tempo_bpm": float(bpm),
        "ppq": int(ppq),
        "time_zero_sec": float(t0),
        "tracks": []
    }

    def _emit_track(track_name: str, cols_for_track: List[str], channel: int, cc_base_fallback: int = 80):
        nonlocal tracks, mapping
        if not cols_for_track:
            return
        evs: List[Tuple[int, bytes]] = []
        evs.append((0, _midi_meta(0x03, track_name.encode("utf-8", errors="replace"))))

        cc_map = {}
        next_cc = int(cc_base_fallback)

        # Stable ordering
        for col in sorted(cols_for_track):
            lane = _local_lane_key(col)
            if lane in _LOCAL_CC_MAP:
                cc = int(_LOCAL_CC_MAP[lane])
            else:
                cc = next_cc
                next_cc += 1
            # Avoid collisions
            while cc in cc_map.values():
                cc += 1
            cc_map[col] = cc

        # Dedupe: only emit when quantized CC value changes.
        # seek_safe=True additionally resamples continuous lanes onto a uniform grid (better scrubbing / jumping).
        t_src = np.asarray(t_rel, np.float64)

        # Build a grid once per track (shared by all lanes).
        # NOTE: This does not make CC "stateful" in a strict sense, but it guarantees
        # there is almost always a nearby CC event after any seek.
        if seek_safe:
            hz = float(max(1.0, seek_hz))
            dt = 1.0 / hz
            t_end = float(t_src[-1]) if t_src.size else 0.0
            t_grid = np.arange(0.0, t_end + 0.5*dt, dt, dtype=np.float64)
        else:
            t_grid = None

        for col in sorted(cols_for_track):
            cc = cc_map[col]
            kind = _cc_range_kind(col)
            seq = np.asarray(list(rows.get(col) or []), np.float64)

            # Fit seq length to t_src length
            n = min(seq.size, t_src.size)
            if n <= 0:
                continue
            seq = seq[:n]
            ts  = t_src[:n]

            prev_val = None

            # Spike lanes: keep sparse (don’t smear impacts), but still time-aligned.
            if _is_spike_lane(col) or (not seek_safe):
                for i in range(n):
                    v = _to_cc_value(seq[i], kind)
                    if prev_val is None or v != prev_val:
                        tick = int(round(ts[i] * float(ppq)))
                        evs.append((tick, _midi_cc(channel, cc, v)))
                        prev_val = v
                continue

            # Continuous lanes: resample to uniform grid
            xs = _resample_to_grid(ts, seq, t_grid)

            # Emit a value at t=0 no matter what (guarantees initial state)
            v0 = _to_cc_value(xs[0], kind)
            evs.append((0, _midi_cc(channel, cc, v0)))
            prev_val = v0

            eps = int(max(1, int(seek_eps_cc)))

            for i in range(1, len(t_grid)):
                v = _to_cc_value(xs[i], kind)
                if abs(int(v) - int(prev_val)) >= eps:
                    tick = int(round(float(t_grid[i]) * float(ppq)))
                    evs.append((tick, _midi_cc(channel, cc, v)))
                    prev_val = v


        tracks.append(evs)
        mapping["tracks"].append({
            "name": track_name,
            "channel": int(channel),
            "lanes": [
                {"column": col, "cc": int(cc_map[col]), "kind": _cc_range_kind(col)}
                for col in sorted(cols_for_track)
            ]
        })

    # GLOBAL track (aggregate + env)
    _emit_track("GLOBAL", global_cols, channel=0, cc_base_fallback=60)

    # Per-label tracks
    for ti, (label, cols_for_label) in enumerate(sorted(groups.items(), key=lambda kv: kv[0].lower())):
        ch = (ti + 1) % 16
        name = f"{track_name_prefix}:{label}"
        _emit_track(name, cols_for_label, channel=ch, cc_base_fallback=80)

    write_midi_file(midi_path, ppq=int(ppq), tracks=tracks)

    try:
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
    except Exception:
        pass

def save_scene_csv_and_push(video_path, scene_id: int, sc: Scene, fps, W, H, robust_gamma=1.0, push=True):
    """
    CSV export that matches export_fullpass overlay policy EXACTLY for impacts:
      1) Prefer recorded live lanes (scene.roi_imp_in/out) if present.
      2) Else compute offline via _impacts_for_mode(vx_s,vy_s,vz_s,fps).
      3) Apply per-direction refractory (refractory_in_ms / refractory_out_ms, fallback refractory_ms).
      4) Merge near-duplicates within 30 ms.
      5) If both IN and OUT collide at the same frame → OUT wins.
      6) Always emit impact_score01 (normalized S_any).

    Other columns (env/dir/pos/speed/acc/jerk/aggregate) preserved.
    """
    import numpy as _np
    T = len(sc.times)
    if T == 0:
        print(f"[export] Scene {scene_id} has no samples. Skipping push.")
        return None

    # --- bound rows to explicit scene range (identical trim used elsewhere) ---
    if sc.times:
        start_t = float(sc.start) / float(fps)
        end_t   = (float(sc.end) / float(fps)) if (sc.end is not None) else float(sc.times[-1])
        lo = next((i for i,t in enumerate(sc.times) if t >= start_t), 0)
        hi = next((i for i,t in enumerate(sc.times) if t >  end_t), len(sc.times))
        if lo > 0 or hi < len(sc.times):
            sc.times = sc.times[lo:hi]
            def _trim(dic):
                if isinstance(dic, dict):
                    for k, seq in list(dic.items()):
                        dic[k] = list(seq[lo:hi])
            _trim(sc.roi_cx); _trim(sc.roi_cy); _trim(sc.roi_vx); _trim(sc.roi_vy); _trim(sc.roi_vz); _trim(sc.roi_env)
            _trim(sc.roi_imp_in); _trim(sc.roi_imp_out)
            T = len(sc.times)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = f"{stem}_scene{scene_id:02d}_roi8e_nomask.csv"

    rows = dict(time=list(sc.times))
    agg_cx=[]; agg_cy=[]; agg_vx=[]; agg_vy=[]; agg_vz=[]; agg_pz=[]
    roi_labels=[]

    label_mode = getattr(sc, 'label_mode', 'per_roi')
    # per-label aggregation slots used only when label_mode == "per_label"
    # label_slots[label] = {
    #   "primary": int or None,
    #   "primary_curves": [np.array, ...],     # per-ROI normalized PRIMARY flux_inner
    #   "secondary_curves": [np.array, ...],   # per-ROI normalized SECONDARY flux_inner
    #   "imp_roles": { "PRIMARY": {"fi":..., "fo":..., "score":...}, "SECONDARY1": {...}, ... }
    # }
    label_slots = {}


    _posx_dict = getattr(sc, 'roi_cx', {})
    _posy_dict = getattr(sc, 'roi_cy', {})
    _vx_dict   = getattr(sc, 'roi_vx', {})
    _vy_dict   = getattr(sc, 'roi_vy', {})
    _vz_dict   = getattr(sc, 'roi_vz', {})
    _in_dict   = getattr(sc, 'roi_imp_in', {})
    _out_dict  = getattr(sc, 'roi_imp_out', {})

    def _pad(dic, key, default):
        seq = dic.get(key, []) if isinstance(dic, dict) else []
        if len(seq)==0: return _np.full(T, float(default), float)
        if len(seq)<T:  seq = list(seq) + [seq[-1]]*(T-len(seq))
        if len(seq)>T:  seq = list(seq)[:T]
        return _np.asarray(seq, float)

    def _dedupe_flags(flags, fps, combine_ms=30):
        flags = _np.asarray(flags, float)
        idx = _np.where(flags > 0.5)[0]
        if idx.size <= 1: return flags
        combine = max(1, int(round(combine_ms * fps / 1000.0)))
        keep = [int(idx[0])]
        for j in map(int, idx[1:]):
            if j - keep[-1] >= combine:
                keep.append(j)
        out = _np.zeros_like(flags); out[keep] = 1.0
        return out

    def _apply_dir_refractory(idx, fps, refr_ms):
        idx = _np.asarray(idx, int)
        if idx.size <= 1: return idx
        gap = max(1, int(round(float(refr_ms) * fps / 1000.0)))
        keep = [int(idx[0])]
        for j in map(int, idx[1:]):
            if j - keep[-1] >= gap:
                keep.append(j)
        return _np.asarray(keep, int)

    def _extend_impacts_to_dir_flip(r, vx_s, vy_s, vz_s, fi, fo, fps):
        """
        Turn IN/OUT spikes into segments that last until the next
        direction flip along the AoI.

        - fi, fo: 0/1 per-frame impact flags (after refractory/dedupe)
        - vx_s, vy_s, vz_s: smoothed velocities (px/s)
        - AoI = (io_dir_deg, axis_elev_deg, axis_z_scale), io_in_sign.

        For each impact frame p in fi/fo:
          - start at p
          - find the next frame where sign(io_in_sign * v_al) changes
          - fill [p, end) with 1.0 in the corresponding lane.
        """
        import numpy as _np

        fi = _np.asarray(fi, float)
        fo = _np.asarray(fo, float)
        T  = int(fi.size)
        if T == 0:
            return fi, fo

        # Fit vel lengths to T
        vx_s = _np.asarray(vx_s, float)[:T]
        vy_s = _np.asarray(vy_s, float)[:T]
        vz_s = _np.asarray(vz_s, float)[:T]

        # Axis projection (same as impacts)
        ax, ay, az = _axis_unit3d(
            getattr(r, "io_dir_deg", 0.0),
            getattr(r, "axis_elev_deg", 0.0)
        )
        kz = float(getattr(r, "axis_z_scale", 1.0))
        v_al = vx_s*ax + vy_s*ay + (kz * vz_s)*az

        io_sign = int(getattr(r, "io_in_sign", +1))
        v_in    = io_sign * v_al

        # Direction sign with deadband
        v_zero = float(V_ZERO)  # same deadband as classifier
        sign_raw = _np.zeros(T, int)
        nz = _np.abs(v_in) >= v_zero
        sign_raw[nz] = _np.sign(v_in[nz]).astype(int)

        # Collect direction-change points (ignoring 0→±1 / ±1→0 noise)
        change_pts = []
        prev = sign_raw[0]
        for t in range(1, T):
            cur = sign_raw[t]
            if prev != 0 and cur != 0 and cur != prev:
                change_pts.append(t)
            if cur != 0:
                prev = cur
        change_pts = _np.asarray(change_pts, int)

        def _segment_end(start):
            """First change strictly after 'start', else T."""
            if change_pts.size == 0:
                return T
            later = change_pts[change_pts > int(start)]
            return int(later[0]) if later.size else T

        fi_ext = _np.zeros_like(fi, float)
        fo_ext = _np.zeros_like(fo, float)

        in_idx  = _np.where(fi > 0.5)[0]
        out_idx = _np.where(fo > 0.5)[0]

        # IN segments
        for p in map(int, in_idx):
            if p < 0 or p >= T:
                continue
            end = _segment_end(p)
            if end > p:
                fi_ext[p:end] = 1.0

        # OUT segments
        for p in map(int, out_idx):
            if p < 0 or p >= T:
                continue
            end = _segment_end(p)
            if end > p:
                fo_ext[p:end] = 1.0

        return fi_ext, fo_ext


    for ri, r in enumerate(sc.rois):
        # one ROI can have multiple comma-separated aliases
        labels = _split_labels(getattr(r, "name", None), f"roi{ri}")
        roi_labels.extend(labels)

        # base series (T-padded) – computed ONCE per physical ROI
        cx = _pad(_posx_dict, ri, (r.rect[0] + r.rect[2]/2.0))
        cy = _pad(_posy_dict, ri, (r.rect[1] + r.rect[3]/2.0))
        vx = _pad(_vx_dict,   ri, 0.0)
        vy = _pad(_vy_dict,   ri, 0.0)
        vz = _pad(_vz_dict,   ri, 0.0)
        T  = len(vx)
        dt = 1.0/max(1e-6, float(fps))

        # optional: same dup+lowconf interpolation used by overlay export
        flags = None
        if getattr(sc, "dup_flags", None) and len(sc.dup_flags) == T:
            flags = _np.asarray(sc.dup_flags, bool)

        low = getattr(sc, "roi_lowconf", {}).get(ri, [])
        if low and len(low) == T:
            low = _np.asarray(low, bool)
            flags = low if flags is None else (flags | low)

        if flags is not None:
            vx = interpolate_over_dups(vx, flags)
            vy = interpolate_over_dups(vy, flags)
            vz = interpolate_over_dups(vz, flags)


        # pos/dir/env/speed/acc/jerk (unchanged behavior)
        if getattr(r, 'bound', None):
            bx, by, bw, bh = map(float, r.bound)
            posx01_s = _np.clip((cx - bx) / max(1.0, bw), 0.0, 1.0)
            posy01_s = _np.clip((cy - by) / max(1.0, bh), 0.0, 1.0)
        else:
            posx01_s = _np.clip(cx / float(W), 0.0, 1.0)
            posy01_s = _np.clip(cy / float(H), 0.0, 1.0)


        speed_ref = _np.sqrt(vx*vx + vy*vy + vz*vz)

        vx_s = robust_smooth_adaptive(vx, fps, speed_ref=speed_ref)
        vy_s = robust_smooth_adaptive(vy, fps, speed_ref=speed_ref)
        vz_s = robust_smooth_adaptive(vz, fps, speed_ref=speed_ref,
                                    slow_win_ms=160, slow_ema_ms=240,
                                    fast_win_ms=24,  fast_ema_ms=40)

        ax_s = _deriv_central(vx_s, dt); ay_s = _deriv_central(vy_s, dt); az_s = _deriv_central(vz_s, dt)
        jx_s = _deriv_central(ax_s, dt); jy_s = _deriv_central(ay_s, dt); jz_s = _deriv_central(az_s, dt)

        # === Axis-of-interest (AoI) + Lateral 1-D kinematics ===

        # AoI axis in 3D (thrust)
        yaw  = float(getattr(r, "io_dir_deg", getattr(r, "dir_gate_deg", 0.0)))
        elev = float(getattr(r, "axis_elev_deg", 0.0))
        kz   = float(getattr(r, "axis_z_scale", 1.0))
        ax_u, ay_u, az_u = _axis_unit3d(yaw, elev)

        # 3D velocity in matched units
        vx3 = vx_s
        vy3 = vy_s
        vz3 = kz * vz_s
        vmag = _np.sqrt(vx3*vx3 + vy3*vy3 + vz3*vz3) + 1e-12  # total speed px/s

        # --- AoI (thrust) along-axis component ---
        v_al = vx3*ax_u + vy3*ay_u + vz3*az_u
        a_al = _deriv_central(v_al, dt)
        j_al = _deriv_central(a_al, dt)

        # ---------------- IMPACT HIT SPIKES (jerk-based, instantaneous) ----------------
        # Why: IN/OUT lanes are "state/phase" (regions). We also want punctual collision instants.
        # How: build a robust jerk z-score on the AoI axis jerk (j_al), then pick sparse local maxima.

        # 1) Robust jerk z-score (>=0)
        jerk_z = _np.maximum(0.0, _mad_z(_np.abs(j_al)))

        # 2) Pick 1-frame "hit" spikes from jerk_z
        hit_z_thr     = float(getattr(r, "impact_hit_jerk_z", 2.8))
        hit_min_sep   = float(getattr(r, "impact_hit_min_sep_ms", 35.0))
        hit_refine_ms = float(getattr(r, "impact_hit_refine_ms", 25.0))

        hit_idx = _impact_spike_from_jerk(
            jerk_z, fps,
            z_thr=hit_z_thr,
            min_sep_ms=hit_min_sep,
            refine_ms=hit_refine_ms
        )

        hit_spk = _np.zeros(T, float)
        for p in hit_idx:
            if 0 <= int(p) < T:
                hit_spk[int(p)] = 1.0
            
        # OPTIONAL: directional split for hit spikes (IN vs OUT)
        hit_in_spk  = _np.zeros(T, float)
        hit_out_spk = _np.zeros(T, float)

        io_sign = int(getattr(r, "io_in_sign", +1))
        v_zero  = float(getattr(r, "impact_flip_deadband_ps", V_ZERO))
        preW    = max(1, int(round(float(getattr(r, "impact_pre_ms", 80))  * fps / 1000.0)))
        postW   = max(1, int(round(float(getattr(r, "impact_post_ms", 80)) * fps / 1000.0)))

        for p in hit_idx:
            p = int(p)
            if p < 0 or p >= T:
                continue
            lab = _classify_flip_sample(io_sign=io_sign, v_al=v_al, p=p, pre=preW, post=postW, v_zero=v_zero)
            if lab == "in":
                hit_in_spk[p] = 1.0
            elif lab == "out":
                hit_out_spk[p] = 1.0

        # -------------------------------------------------------------------------------


        axis_v11    = normalize_signed11(v_al)
        axis_acc11  = normalize_signed11(a_al)
        axis_jerk11 = normalize_signed11(j_al)
        axis_dir11  = normalize_signed11(v_al)  # stroke direction: -1..+1

        # --- Lateral cone: axis = AoI rotated by +90° yaw (hip sway L/R) ---
        lx_u, ly_u, lz_u = _axis_unit3d(yaw + 90.0, elev)

        # signed velocity along lateral axis (perpendicular to AoI)
        v_lat_signed = vx3*lx_u + vy3*ly_u + vz3*lz_u

        # cosine of angle to lateral axis (symmetric for ±)
        cos_lat = _np.abs(v_lat_signed / vmag)

        # lateral cone weight w_lat in [0,1]
        lat_half = float(getattr(r, "lat_half_deg", 30.0))
        lat_pow  = float(getattr(r, "lat_power", 4.0))
        if 0.0 < lat_half < 90.0:
            cos_min = math.cos(math.radians(lat_half))
            w_lat = _np.zeros_like(cos_lat)
            mask = cos_lat >= cos_min
            if _np.any(mask):
                # ramp from 0 at edge of cone to 1 on-axis
                w_lat[mask] = ((cos_lat[mask] - cos_min) /
                               max(1e-6, 1.0 - cos_min)) ** lat_pow
        else:
            # fallback: pure |cos|^p weighting
            w_lat = cos_lat ** lat_pow

        # optional speed gate: kill micro motions
        vmin_ps = float(getattr(r, "impact_min_speed_ps", 0.0))
        if vmin_ps > 0.0:
            gate_v = _np.clip(vmag / vmin_ps, 0.0, 1.0)
            w_lat *= gate_v

        # weighted lateral velocity & derivatives
        v_lat_w = v_lat_signed * w_lat
        a_lat   = _deriv_central(v_lat_w, dt)
        j_lat   = _deriv_central(a_lat, dt)

        # normalized lateral lanes
        lat_v11    = normalize_signed11(v_lat_w)          # signed L/R sway
        lat_acc11  = normalize_signed11(a_lat)
        lat_jerk11 = normalize_signed11(j_lat)
        lat_dir11  = lat_v11                               # alias: "direction"
        lat_amp01  = normalize_unsigned01(_np.abs(v_lat_w))  # 0..1 sway intensity


        # --- NEW: center-based velocities ---
        vx_pos_raw, vy_pos_raw = _center_vel_from_pos(cx, cy, fps)
        vx_pos_s = robust_smooth(vx_pos_raw, fps, win_ms=140, ema_tc_ms=200, mad_k=3.6)
        vy_pos_s = robust_smooth(vy_pos_raw, fps, win_ms=140, ema_tc_ms=200, mad_k=3.6)
        use_pos_channel = (
            bool(getattr(r, "anchor_user_set", False)) or
            str(getattr(r, "cmat_mode", "off")).lower() != "off"
        )
        # ------------------------------------


        vP = vx_s if _np.mean(_np.abs(vx_s)) >= _np.mean(_np.abs(vy_s)) else vy_s
        speed_s = _np.sqrt(vx_s*vx_s + vy_s*vy_s + vz_s*vz_s)

        # 1) Gaussian, cycle-locked phase envelope (0..1, already phase-true)
        env01_gauss = hybrid_env01(speed_s, vP, fps, mix=0.50)

        # 2) Global raw magnitude 0..1 (instantaneous, no lag)
        flux_raw01 = normalize_unsigned01(speed_s)

        # 3) Slow trend (symmetric → zero-phase)
        #    Use Gaussian blur in *time*, not EMA.
        sigma_slow = max(1.0, 0.40 * fps)   # ~0.4 s equivalent
        slow = _gauss_blur1d(speed_s, sigma_slow)

        # energy contrast = how much faster/slower than the slow baseline
        E_energy = speed_s - slow
        E_energy = normalize_signed11(E_energy)  # -1..+1

        # 4) Micro-jitter component (fast contrast)
        sigma_micro = max(1.0, 0.12 * fps)  # ~0.12 s
        smooth_micro = _gauss_blur1d(speed_s, sigma_micro)
        micro = speed_s - smooth_micro
        E_micro = normalize_signed11(micro)

        # 5) Phase component: center Gaussian env so it contributes shape, not DC
        E_phase = env01_gauss - float(_np.mean(env01_gauss))

        # 6) SUPERPOSITION (all components are zero-lag)
        w_phase  = 1.0
        w_energy = 0.8
        w_micro  = 0.4

        flux_super = (
            w_phase  * E_phase  +
            w_energy * E_energy +
            w_micro  * E_micro  +
            flux_raw01          # keep some instantaneous magnitude in the mix
        )

        # --- NEW: per-ROI inner loudness normalization ---
        # This is the "local" 0..1 curve per ROI before any PRIMARY/SECONDARY merging.
        flux_inner = flux_norm01(
            flux_super,
            gamma=0.85,   # inner loudness
            p_lo=5,
            p_hi=95
        )

        # --- Entropy lane: high-freq residual of speed_s ---
        # Slow trend over ~0.35 s
        sigma_ent = max(1.0, 0.35 * fps)
        slow_ent  = _gauss_blur1d(speed_s, sigma_ent)
        res_ent   = speed_s - slow_ent

        entropy_roi = normalize_unsigned01(_np.abs(res_ent))

        # --- FLUX_SHAPE: offline, phase-conditioned release to 50% by next AoI flip ---
        if FLUX_REL50_ENABLE:
            flux_inner = flux_release_to_target_by_phase01(
                flux_inner,
                entropy_roi,
                v_al,                 # you already computed v_al above (AoI axis velocity)
                fps,
                io_in_sign=int(getattr(r, "io_in_sign", +1)),
                v_zero=float(getattr(r, "impact_flip_deadband_ps", V_ZERO)),
                target=float(FLUX_REL50_TARGET),
            )
        elif FLUX_LEAK_ENABLE:
            # fallback to the older entropy-density leak model
            flux_inner = flux_leaky_shape01(flux_inner, entropy_roi, fps)
        # --- END FLUX_SHAPE ---

        # --- END FLUX_LEAK ---


        dirx11_pc, diry11_pc, dirz11_pc = per_cycle_dir_gauss_blend11(vx_s, vy_s, vz_s, vP, fps)
        # dirz11_s = _dir11(vz_s, 1.0)  # direction Z
        pz_s = leaky_integrate(vz_s, dt, tau_ms=int(getattr(r, 'posz_tau_ms', 800)))
        posz01_s = normalize_unsigned01(pz_s)


        for label in labels:
            rows[f"{label}_flux_env"] = flux_inner.tolist()
            rows[f"{label}_dirx01"]   = dirx11_pc.tolist()
            rows[f"{label}_diry01"]   = diry11_pc.tolist()
            rows[f"{label}_dirz01"]   = dirz11_pc.tolist()
            rows[f"{label}_posx01"]   = posx01_s.tolist()
            rows[f"{label}_posy01"]   = posy01_s.tolist()
            rows[f"{label}_posz01"]   = posz01_s.tolist()

            rows[f"{label}_speed01"]  = normalize_unsigned01(speed_s).tolist()
            rows[f"{label}_acc01"]    = normalize_unsigned01(_np.sqrt(ax_s*ax_s + ay_s*ay_s + az_s*az_s)).tolist()
            rows[f"{label}_jerk01"]   = normalize_unsigned01(_np.sqrt(jx_s*jx_s + jy_s*jy_s + jz_s*jz_s)).tolist()

            rows[f"{label}_entropy01"] = entropy_roi.tolist()

            # NEW: principal-axis AoI lanes (SIGNED, -1..+1)
            rows[f"{label}_axis_v"]    = axis_v11.tolist()
            rows[f"{label}_axis_acc"]  = axis_acc11.tolist()
            rows[f"{label}_axis_jerk"] = axis_jerk11.tolist()
            rows[f"{label}_axis_dir"]  = axis_dir11.tolist()

            # Lateral cone lanes (perpendicular to AoI)
            rows[f"{label}_lat_v"]      = lat_v11.tolist()     # signed left/right, -1..+1
            rows[f"{label}_lat_acc"]    = lat_acc11.tolist()   # signed
            rows[f"{label}_lat_jerk"]   = lat_jerk11.tolist()  # signed
            rows[f"{label}_lat_dir"]    = lat_dir11.tolist()   # alias of lat_v
            rows[f"{label}_lat_amp01"]  = lat_amp01.tolist()   # 0..1 sway amplitude


        # === IMPACTS: mirror export_fullpass ===
        fi_live = _np.asarray(_in_dict.get(ri, []),  float)
        fo_live = _np.asarray(_out_dict.get(ri, []), float)
        used_live = (fi_live.size or fo_live.size)


        if used_live:
            # time-fit to T (do not recompute)
            def _fit_len(a, T):
                a = _np.asarray(a, float)
                if a.size < T: a = _np.pad(a, (0, T - a.size))
                if a.size > T: a = a[:T]
                return a
            fi = _fit_len(fi_live, T)
            fo = _fit_len(fo_live, T)
            # Apply refractory to live lanes too (otherwise spikes can machine-gun)
            refr_in  = int(getattr(r, "refractory_in_ms",  getattr(r, "refractory_ms", 140)))
            refr_out = int(getattr(r, "refractory_out_ms", getattr(r, "refractory_ms", 140)))

            # Convert to indices -> refractory -> back to flags
            fi_idx = _np.where(_np.asarray(fi) > 0.5)[0]
            fo_idx = _np.where(_np.asarray(fo) > 0.5)[0]
            fi_idx = _apply_dir_refractory(fi_idx, fps, refr_in)
            fo_idx = _apply_dir_refractory(fo_idx, fps, refr_out)

            fi = _np.zeros(T, float); fo = _np.zeros(T, float)
            fi[fi_idx] = 1.0; fo[fo_idx] = 1.0


            # extend impacts until direction flip
            fi, fo = _extend_impacts_to_dir_flip(r, vx_s, vy_s, vz_s, fi, fo, fps)
            # Ensure extended lanes exist for downstream code (and for collision policy)
            fi_ext, fo_ext = fi, fo

            # OUT wins on collisions (must apply to EXTENDED lanes)
            both = (fi_ext > 0.5) & (fo_ext > 0.5)
            fi_ext[both] = 0.0

            # offline score for reference
            S_any, _, _ = _impacts_for_mode(r, vx_s, vy_s, vz_s, fps)
        else:
            # offline compute, identical to overlay
            mode = str(getattr(r, "impact_mode", "flux_dog")).lower()

            # pull IG series (fit to T)
            igE  = np.asarray(sc.roi_igE.get(ri,  []), np.float64)
            igdE = np.asarray(sc.roi_igdE.get(ri, []), np.float64)
            if igE.size < T:  igE  = np.pad(igE,  (0, T-igE.size))
            if igdE.size < T: igdE = np.pad(igdE, (0, T-igdE.size))
            igE  = igE[:T]
            igdE = igdE[:T]

            lc_seq = sc.roi_lowconf.get(ri, [])
            lowconf = np.asarray(lc_seq[:T], bool) if lc_seq else None

            if mode == "hybrid":
                # Best general: DoG+Jerk proposal (your existing fusion)
                r_tmp = r
                r_tmp.impact_mode = "flux_dog_jerk"
                S_any, in_idx, out_idx = _impacts_for_mode(r_tmp, vx_s, vy_s, vz_s, fps)
                # Optional: mild IG-LoG confirmation gate by dropping lowconf candidates
                if lowconf is not None:
                    in_idx  = np.asarray([p for p in map(int, in_idx)  if not lowconf[p]], int)
                    out_idx = np.asarray([p for p in map(int, out_idx) if not lowconf[p]], int)

            elif mode == "fast":
                # 1-frame impulses from IG-LoG (then classify IN/OUT using your existing dir classifier)
                imp = _iglog_impulse_pairs(igdE, lowconf, thr_pos=2.8, thr_neg=2.2)
                out_idx, in_idx = classify_in_out_by_dir(
                    imp, vx_s, vy_s, fps,
                    float(getattr(r, "io_dir_deg", 0.0)),
                    int(getattr(r, "io_in_sign", +1)),
                    anchor_shift_ms=int(getattr(r, "impact_lead_ms", 40))
                )
                S_any = flux_norm01(np.maximum(0.0, _mad_z(np.abs(igdE))))

            elif mode == "smooth":
                # smooth contacts from EMA(E) rise
                ev = _iglog_smooth_rise(igE, lowconf, fps, ema_ms=140, thr_z=2.4, persist_fr=2)
                out_idx, in_idx = classify_in_out_by_dir(
                    ev, vx_s, vy_s, fps,
                    float(getattr(r, "io_dir_deg", 0.0)),
                    int(getattr(r, "io_in_sign", +1)),
                    anchor_shift_ms=int(getattr(r, "impact_lead_ms", 40))
                )
                S_any = flux_norm01(np.maximum(0.0, _mad_z(np.abs(_deriv_central(igE, 1.0/max(1e-6,fps))))))

            else:
                # legacy modes: flux_dog / axis_jerk / flux_dog_jerk / reversal / etc.
                S_any, in_idx, out_idx = _impacts_for_mode(r, vx_s, vy_s, vz_s, fps)


            refr_in  = int(getattr(r, "refractory_in_ms",  getattr(r, "refractory_ms", 140)))
            refr_out = int(getattr(r, "refractory_out_ms", getattr(r, "refractory_ms", 140)))
            if mode == "fast":
                refr_in = refr_out = 0   # or int(1000.0/fps) for ~1 frame
            in_idx   = _apply_dir_refractory(in_idx,  fps, refr_in)
            out_idx  = _apply_dir_refractory(out_idx, fps, refr_out)

            fi = _np.zeros(T, float); fo = _np.zeros(T, float)
            fi[in_idx]  = 1.0; fo[out_idx] = 1.0

        io_pref = int(getattr(r, "impact_io", 0))  # 0 both, +1 OUT only, -1 IN only
        if io_pref > 0: fi[:] = 0.0
        if io_pref < 0: fo[:] = 0.0

        fi = _dedupe_flags(fi, fps, combine_ms=30)
        fo = _dedupe_flags(fo, fps, combine_ms=30)
        # Always extend impacts until direction flip (same logic as overlay)
        fi_ext, fo_ext = _extend_impacts_to_dir_flip(r, vx_s, vy_s, vz_s, fi, fo, fps)

        # OUT wins on collisions
        both = (fi_ext > 0.5) & (fo_ext > 0.5)
        fi_ext[both] = 0.0


        # --- Kinetic Label Engine: hook per-label aggregation ---
                # --- Kinetic Label Engine: hook per-label aggregation ---
        if label_mode == "per_label":
            lp = getattr(sc, "label_primary", {})
            for lab in labels:
                slot = label_slots.setdefault(lab, {
                    "primary": None,
                    "primary_curves": [],
                    "secondary_curves": [],
                    "imp_roles": {}
                })
                primary_ri = lp.get(lab, None)
                # Decide role for this ROI under this label
                if primary_ri is not None and primary_ri == ri:
                    role = "PRIMARY"
                elif slot["primary"] is None and primary_ri is None:
                    # no explicit primary defined yet → first seen ROI is PRIMARY
                    role = "PRIMARY"
                else:
                    # assign stable secondary index
                    existing_secs = [r for r in slot["imp_roles"].keys() if r.startswith("SECONDARY")]
                    role = f"SECONDARY{len(existing_secs)+1}"

                # Store this ROI's normalized curve per role
                if role == "PRIMARY":
                    slot["primary"] = ri
                    slot["primary_curves"].append(flux_inner.copy())
                else:
                    slot["secondary_curves"].append(flux_inner.copy())


                # merge impacts per role (keep max score, OR flags)
                prev = slot["imp_roles"].get(role)
                if prev is None:
                    slot["imp_roles"][role] = dict(
                        fi=fi.copy(),
                        fo=fo.copy(),
                        score=S_any.copy()
                    )
                else:
                    fi_r = prev["fi"]; fo_r = prev["fo"]; S_r = prev["score"]
                    L = min(len(fi_r), len(fi))
                    fi_r[:L] = _np.maximum(fi_r[:L], fi[:L])
                    fo_r[:L] = _np.maximum(fo_r[:L], fo[:L])
                    Ls = min(len(S_r), len(S_any))
                    S_r[:Ls] = _np.maximum(S_r[:Ls], S_any[:Ls])
                    slot["imp_roles"][role] = dict(fi=fi_r, fo=fo_r, score=S_r)

        # per-ROI impact lanes
        # Only emit these in per-ROI mode; in per-label mode we use aggregated role lanes.
        if label_mode == "per_roi":
            for label in labels:
                rows[f"{label}_impact_in01"]    = fi_ext.tolist()
                rows[f"{label}_impact_out01"]   = fo_ext.tolist()
                # Spike lanes (1-frame hits)
                rows[f"{label}_impact_in_spk01"]  = fi.tolist()
                rows[f"{label}_impact_out_spk01"] = fo.tolist()
                rows[f"{label}_impact_score01"] = normalize_unsigned01(S_any).tolist()

                rows[f"{label}_impact_hit_spk01"] = hit_spk.tolist()
                rows[f"{label}_impact_hit_in_spk01"]  = hit_in_spk.tolist()
                rows[f"{label}_impact_hit_out_spk01"] = hit_out_spk.tolist()


        # aggregate collectors
        agg_cx.append(cx); agg_cy.append(cy)
        agg_vx.append(vx); agg_vy.append(vy); agg_vz.append(vz)
        agg_pz.append(leaky_integrate(vz_s, dt, tau_ms=int(getattr(r, 'posz_tau_ms', 800))))

    # --- Kinetic Label Engine: per-label aggregated lanes ---
    if label_mode == "per_label" and label_slots:
        w_p = 1.0   # PRIMARY weight
        w_s = 0.7   # SECONDARY stack weight
        for lab, slot in label_slots.items():
            prim_list = slot.get("primary_curves", []) or []
            sec_list  = slot.get("secondary_curves", []) or []

            if not prim_list and not sec_list:
                continue

            # 1) merge PRIMARY curves (already inner-normalized)
            flux_primary = None
            if prim_list:
                stack_p = _np.vstack(prim_list)        # (#primary, T)
                flux_primary = stack_p.max(axis=0)     # max across PRIMARY ROIs

            # 2) merge SECONDARY curves
            flux_secondary = None
            if sec_list:
                stack_s = _np.vstack(sec_list)         # (#secondary, T)
                flux_secondary = stack_s.max(axis=0)   # max across SECONDARY ROIs

            # 3) fuse roles
            if flux_primary is not None and flux_secondary is not None:
                flux_fused = w_p * flux_primary + w_s * flux_secondary
            elif flux_primary is not None:
                flux_fused = flux_primary
            else:
                flux_fused = flux_secondary

            # 4) final loudness normalization across the fused curve
            flux_final = flux_norm01(
                flux_fused,
                gamma=0.75,    # a bit stronger compression at the end
                p_lo=10,
                p_hi=95
            )

            # This is the env lane the bridge uses for this label
            rows[f"{lab}_flux_env"] = flux_final.tolist()

            # per-role impacts for label (unchanged)
            for role, data in slot.get("imp_roles", {}).items():
                fi_r = data["fi"]
                fo_r = data["fo"]
                S_r  = data["score"]
                base = f"{lab}_imp_{role}"
                rows[f"{base}_in01"]    = fi_r.tolist()
                rows[f"{base}_out01"]   = fo_r.tolist()
                rows[f"{base}_score01"] = normalize_unsigned01(S_r).tolist()


    # === Aggregate track (same schema as bridge mapping) ===
    if agg_cx:
        cxm = _np.mean(_np.vstack(agg_cx), axis=0)
        cym = _np.mean(_np.vstack(agg_cy), axis=0)
        vxm = _np.mean(_np.vstack(agg_vx), axis=0)
        vym = _np.mean(_np.vstack(agg_vy), axis=0)
        vzm = _np.mean(_np.vstack(agg_vz), axis=0)
        pzm = _np.mean(_np.vstack(agg_pz), axis=0)
    else:
        cxm = _np.zeros(T); cym = _np.zeros(T)
        vxm = _np.zeros(T); vym = _np.zeros(T); vzm = _np.zeros(T); pzm = _np.zeros(T)

    rows["agg_posx01"] = _np.clip(cxm/float(W), 0.0, 1.0).tolist()
    rows["agg_posy01"] = _np.clip(cym/float(H), 0.0, 1.0).tolist()
    rows["agg_posz01"] = _dir01(pzm, 1.0).tolist()

    # smoothed aggregate dirs + env (keeps prior behavior)
    vxm_s = robust_smooth(vxm, fps, win_ms=140, ema_tc_ms=200, mad_k=3.6)
    vym_s = robust_smooth(vym, fps, win_ms=140, ema_tc_ms=200, mad_k=3.6)
    vzm_s = robust_smooth(vzm, fps, win_ms=160, ema_tc_ms=240, mad_k=3.8)
    rows["agg_dirx01"] = ((normalize_signed11(vxm_s)*0.5)+0.5).tolist()
    rows["agg_diry01"] = ((normalize_signed11(vym_s)*0.5)+0.5).tolist()
    rows["agg_dirz01"] = ((normalize_signed11(vzm_s)*0.5)+0.5).tolist()

    dt = 1.0/max(1e-6, float(fps))
    axm_s = _deriv_central(vxm_s, dt); aym_s = _deriv_central(vym_s, dt); azm_s = _deriv_central(vzm_s, dt)
    jzm_s = _deriv_central(azm_s, dt)

    rows["agg_velx01"] = _dir01(vxm_s, 1.0).tolist()
    rows["agg_vely01"] = _dir01(vym_s, 1.0).tolist()
    rows["agg_velz01"] = _dir01(vzm_s, 1.0).tolist()
    rows["agg_accx01"] = _dir01(axm_s, 1.0).tolist()
    rows["agg_accy01"] = _dir01(aym_s, 1.0).tolist()
    rows["agg_accz01"] = _dir01(azm_s, 1.0).tolist()
    rows["agg_jerkz01"] = _dir01(jzm_s, 1.0).tolist()

    flux_env_raw = _np.sqrt(vxm_s*vxm_s + vym_s*vym_s + vzm_s*vzm_s)
    rows["flux_env"] = _np.clip(_envelope(flux_env_raw, fps) /
                                (float(_np.percentile(flux_env_raw, 95.0)) + 1e-12), 0.0, 1.0).tolist()

    # --- Aggregate entropy: high-freq residual of aggregate flux ---
    sigma_agg = max(1.0, 0.35 * fps)
    agg_slow  = _gauss_blur1d(flux_env_raw, sigma_agg)
    agg_res   = flux_env_raw - agg_slow
    agg_entropy01 = normalize_unsigned01(_np.abs(agg_res))
    rows["agg_entropy01"] = agg_entropy01.tolist()


    # final length alignment
    TL = len(rows["time"])
    for k,v in list(rows.items()):
        if k=="time": continue
        vv = list(v)
        if len(vv) < TL:  vv = vv + [vv[-1] if vv else 0.0]*(TL - len(vv))
        if len(vv) > TL:  vv = vv[:TL]
        rows[k] = vv

    tmp = csv_path + ".tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, csv_path)
    print(f"[export] wrote {csv_path}")

    # MIDI export (DAW-agnostic): one track per ROI label + GLOBAL.
    try:
        midi_path = os.path.splitext(csv_path)[0] + ".mid"
        map_path  = os.path.splitext(csv_path)[0] + ".midi_map.json"
        write_local_midi_from_rows(
            rows, midi_path, map_path,
            bpm=60.0, ppq=960,
            seek_safe=True,
            seek_hz=24.0,
            seek_eps_cc=1,
            time_zero_sec=0.0,   # <-- makes the MIDI time absolute-to-video/project
        )

        print(f"[export] wrote {midi_path}")
    except Exception as e:
        print(f"[midi] export failed: {e}")

    if push:

        t0 = float(rows["time"][0]); t1 = float(rows["time"][-1])
        emit_reaper_push(csv_path, t0, t1, scene_id, roi_labels, version="roi_v8e_nomask")
    return csv_path


# ========================= PySide6 backend (no HighGUI wheel) =========================
try:
    from PySide6 import QtWidgets, QtGui, QtCore
except Exception as _e:
    QtWidgets = None  # fallback if not installed

class _WheelIOFilter(QtCore.QObject):
    def __init__(self, get_active_scene, get_rois_at):
        super().__init__()
        self._get_active_scene = get_active_scene
        self._get_rois_at = get_rois_at

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Wheel:
            pt = ev.position().toPoint()
            sc = self._get_active_scene()
            if sc is None:
                return False
            rois = self._get_rois_at(sc, pt.x(), pt.y())
            if not rois:
                return False

            # topmost ROI under cursor
            r = rois[-1]

            mods  = QtWidgets.QApplication.keyboardModifiers()
            ctrl  = bool(mods & QtCore.Qt.ControlModifier)
            shift = bool(mods & QtCore.Qt.ShiftModifier)
            alt   = bool(mods & QtCore.Qt.AltModifier)

            # Respect Ctrl+Wheel fast-scrub in the main view
            if ctrl:
                return False

            steps = ev.angleDelta().y() / 120.0

            if shift:
                # SHIFT+WHEEL → adjust pitch only (no yaw drift)
                fine = 1.0 if alt else 4.0         # Alt makes it fine
                r.axis_elev_deg = float(np.clip(getattr(r, "axis_elev_deg", 0.0) + fine*steps, -89.0, +89.0))
                ev.accept()
                return True
            else:
                # plain wheel over ROI → yaw (existing behavior)
                delta = (1.0 if alt else 5.0) * steps
                val = (float(getattr(r, "io_dir_deg", 0.0)) + float(delta)) % 360.0
                r.io_dir_deg  = val
                r.dir_gate_deg = val
                r.dir_io_deg   = val
                ev.accept()
                return True
        return False


def _poly_norm_from_contour(cnt_xy: np.ndarray, h: int, w: int) -> list:
    """Convert Nx1x2 or Nx2 contour in pixel coords -> normalized [(u,v),...] in 0..1"""
    if cnt_xy is None or len(cnt_xy) < 3:
        return []
    cnt = cnt_xy.reshape(-1, 2)
    out = []
    for x, y in cnt:
        u = float(np.clip(x / max(1, (w - 1)), 0.0, 1.0))
        v = float(np.clip(y / max(1, (h - 1)), 0.0, 1.0))
        out.append([u, v])
    return out

def _autogen_iglog_occlusion_polys(
    gray_u8: np.ndarray,
    struct_state: dict,
    top_percent: float = 6.0,
    min_area_frac: float = 0.01,
    prefer_border: bool = True,
    simplify_px: float = 2.0,
) -> list:
    """
    Auto-generate polygon(s) for occlusion mask from a structure/salience map.
    - gray_u8: ROI patch (uint8, shape HxW)
    - struct_state: your existing IG-LoG/HG state dict used by struct_M_from_gray_u8
    Returns: list of polygons in normalized coords: [ [ [u,v], ... ], ... ]
    """
    if gray_u8 is None or gray_u8.size == 0:
        return []

    H, W = gray_u8.shape[:2]
    if H < 8 or W < 8:
        return []

    # 1) Get salience/structure map (your HG/IG-LoG “M”)
    try:
        M = struct_M_from_gray_u8(gray_u8, struct_state)  # expected uint8 or float-ish
        if M.dtype != np.uint8:
            M = np.clip(M, 0, 255).astype(np.uint8)
    except Exception:
        return []

    # 2) Threshold top X% (hair/arm/bright edges are typically here)
    p = float(np.clip(top_percent, 0.5, 30.0))
    thr = float(np.percentile(M, 100.0 - p))
    bw = (M >= thr).astype(np.uint8) * 255

    # 3) Clean it up a bit
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, k, iterations=1)
    bw = cv.morphologyEx(bw, cv.MORPH_DILATE, k, iterations=1)

    # 4) Connected components -> pick a “best” component
    n, labels, stats, _ = cv.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return []

    min_area = int(min_area_frac * (H * W))
    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv.CC_STAT_LEFT])
        y = int(stats[i, cv.CC_STAT_TOP])
        w = int(stats[i, cv.CC_STAT_WIDTH])
        h = int(stats[i, cv.CC_STAT_HEIGHT])

        # Score: area + border preference (occluders often enter from edges)
        border_bonus = 0.0
        if prefer_border:
            touches = (x <= 1) or (y <= 1) or (x + w >= W - 2) or (y + h >= H - 2)
            border_bonus = 0.35 if touches else 0.0

        score = (area / max(1, H * W)) + border_bonus
        candidates.append((score, i))

    if not candidates:
        return []

    candidates.sort(reverse=True)
    best_i = candidates[0][1]

    mask = (labels == best_i).astype(np.uint8) * 255

    # 5) Contour -> simplified polygon
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    cnt = max(cnts, key=cv.contourArea)
    if cv.contourArea(cnt) < min_area:
        return []

    eps = float(np.clip(simplify_px, 0.5, 10.0))
    approx = cv.approxPolyDP(cnt, epsilon=eps, closed=True)

    poly = _poly_norm_from_contour(approx, H, W)
    if len(poly) < 3:
        return []

    return [poly]


def run_qt(video_path):
    assert QtWidgets is not None, "PySide6 not installed. pip install PySide6"
    argv_tail = sys.argv[2:]
    _parse_cli_scale(argv_tail)  # reuse your scale parser
    ai_cfg = _parse_cli_ai(argv_tail)
    if ai_cfg.enabled and ai_cfg.require_policy_ack and not ai_cfg.policy_ack:
        # Print once on startup; actual API calls remain disabled until ack flag is present.
        print(OPENAI_POLICY_DISCLAIMER)
        print("AI assist requested but disabled until you add: --i-accept-openai-policy")
    qt_argv = _qt_sanitize_argv(sys.argv)

    cap = cv.VideoCapture(video_path); assert cap.isOpened(), f"Cannot open {video_path}"
    fps = float(cap.get(cv.CAP_PROP_FPS) or 30.0)
    N   = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame0 = cap.read(); assert ok
    H0, W0 = frame0.shape[:2]

    # --- GLOBAL PRE-CAP FOR WORKING RESOLUTION ---
    # Clamp the *effective* PROC_SCALE so that the working height never
    # exceeds MAX_ROI_INPUT_H, regardless of original resolution.
    global PROC_SCALE
    if MAX_ROI_INPUT_H > 0:
        max_scale = float(MAX_ROI_INPUT_H) / float(H0)
        # If user asked for a huge PROC_SCALE on a giant source, cap it.
        PROC_SCALE = min(PROC_SCALE, max_scale)

    W = int(round(W0 * PROC_SCALE)); H = int(round(H0 * PROC_SCALE))
    if (W,H)!=(W0,H0): frame0 = cv.resize(frame0, (W,H), interpolation=cv.INTER_AREA)
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    # Light thumb grabber (same as your shelf)
    thumb_cap = cv.VideoCapture(video_path)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(qt_argv)

    class Controller(QtCore.QObject):
        def __init__(self, ai_cfg: AIConfig):
            super().__init__()
            # ---- OpenAI assist (optional) ----
            self.ai_cfg = ai_cfg
            self.ai = OpenAIAssist(ai_cfg)
            self.ai_hud_lines: List[str] = []
            self.ai_hud_until: float = 0.0
            self._prev_cached_frame: Optional[np.ndarray] = None
            # -------------------------------
            # ---- state copied from your run() ----
            self.W, self.H, self.fps, self.N = W, H, fps, N
            self.cap, self.thumb_cap = cap, thumb_cap
            self.tracker = DeformTracker(W, H, fps)
            self.playing=False; self.recording=False
            self.writer=None
            self.frame_idx=0; self.prev_gray=None; self.last_sampled=-1
            self.scenes=[]; self.active_scene=-1; self.active_roi=0
            self.roi_tmp=None; self.bound_tmp=None
            self.dragging=False; self.drag_is_bound=False; self.p0=(0,0)
            self.repicking=False; self.adding=False
            self.naming_roi=-1; self.name_buf=""
            self.mouse_xy=(0,0)
            # fast scrub + shelf
            self.fast_scrub_until=0.0
            self.ctrl_scrub_step = max(30, int(self.fps * 3))
            self.panel_visible=True; self.panel_flash_until=0.0
            self.PANEL_W = max(120, int(self.W*0.18))
            self.PANEL_ITEM_H = max(56, int(self.H*0.10))
            self.panel_hitboxes=[]
            self.scene_thumbs={}
            self.dragging_timeline = False
            self._seeking = False
            self._last_frame_idx = 0

            # frame cache + seek flag
            self.TIMELINE_ALPHA = 0.35  # 0=transparent, 1=opaque
            self._cached_frame = None
            self._cached_gray  = None
            self._cached_idx   = -1
            self._seeking      = False
            self.postview_active = False
            self.postview_roi = (-1, -1)  # (scene_index, roi_index)
            self._postview_saved_idx = 0
            self.scene_scroll = 0
            self.roi_panel_scroll = 0
            self.roi_panel_hitboxes = []  # [(ri, (x0,y0,x1,y1)), ...]
            self.debug_show_log_blobs = False
            # Kinetic Label Engine: controller-level label mode
            self.label_mode = "per_roi"  # toggled via 'L' key
            self.undo_stack = UndoRedoStack(max_states=30)
            self._undo_last_label = ""

            # --- Live structural preview (Ctrl+Shift+K) ---
            self.struct_preview = False
            self.struct_preview_scale = 0.33   # preview size relative to ROI crop
            self.struct_preview_strip = None   # BGR strip
            self.struct_preview_text = ""      # short stats line
            self.struct_view = False  # Ctrl+Shift+K
            # ------------------------------------------------


            
            self._dec_pos = -1  # last decoded frame index from self.cap

            # view mapping
            self.view_scale = 1.0; self.view_offset=(0,0)
            # HUD strings (ASCII only, no ???)
            self.help_lines = [
    "F1 help | F11 fullscreen | Space/Enter play/pause | Left/Right step | Shift+Left/Right ±10",
    "Ctrl+Wheel fast scrub | Shift+Wheel scene jump | Ctrl+Z undo | Ctrl+Shift+Z / Ctrl+Y redo",
    "N new scene | A set scene start | D set scene end | E end/export | Shift+E reopen end",
    "U add ROI | R repick ROI | Shift+Drag bound | Ctrl+Drag anchor | [ / ] select ROI",
    "Double-click ROI: rename | Shift+Backspace: delete ROI | P export scene | S export all scenes",
    "Debug: Ctrl+D flow | Ctrl+B blobs | H toggle ROI debug overlay",
    "AI opt-in: Ctrl+T tags | Ctrl+I impacts | Ctrl+F flux-fix | Ctrl+O occlusion | Ctrl+M outline",
    "Mask: Ctrl+Shift+M inc/exc | Ctrl+Alt+M IG-LoG mask | Ctrl+Backspace clear mask"
]
            self.TIMELINE_H = max(40, int(self.H * 0.09))
            self.timeline_hover_idx = -1
            self.hover_thumb = None
            self.hover_thumb_at = -1
            self._next_hover_grab_time = 0.0

            self.recording = False
            self.ff = None
            self.ocv_writer = None


            # timer drive
            self.timer = QtCore.QTimer(self)
            self.timer.setTimerType(QtCore.Qt.PreciseTimer)
            self.timer.setInterval(max(1, int(1000 / min(60, self.fps))))  # ~60 FPS cap
            self.timer.timeout.connect(self.tick)
            self.timer.start()

            self._impact_until = {}   # {(scene, roi): unix_time}
            self._impact_label = {}   # {(scene, roi): "OUT"|"IN"}
            self.drag_is_anchor = False
            self.anchor_tmp = None
            self.anchor_roi = -1

        def _get_curr_frame_bgr(self) -> np.ndarray:
            """
            Return current frame as BGR at controller scale (W,H).
            Uses the same cache that tick() maintains.
            Falls back to decoding if cache is empty/stale.
            """
            try:
                if self._cached_frame is not None and int(self._cached_idx) == int(self.frame_idx):
                    return self._cached_frame
            except Exception:
                pass

            # Fallback: decode this frame (safe even if cache not ready yet)
            ok, fr = self._read_frame(int(self.frame_idx))
            if not ok or fr is None:
                return np.zeros((self.H, self.W, 3), np.uint8)

            if PROC_SCALE != 1.0:
                fr = cv.resize(fr, (self.W, self.H), interpolation=cv.INTER_AREA)
            return fr


        def show_help_dialog(self) -> None:
            """Show a minimal hotkey reference. (Info overlay was removed; F1 is now the entry point.)"""
            try:
                title = f"{APP_NAME} — Help"
                lines = list(getattr(self, "help_lines", []) or [])
                text = "\n".join(str(x) for x in lines if str(x).strip())
                QtWidgets.QMessageBox.information(getattr(self, "mw", None), title, text)
            except Exception as e:
                print(f"[help] failed: {e}")

        # ---------- Undo/Redo ----------
        def _undo_state(self) -> Dict[str, Any]:
            return {
                "scenes": self.scenes,
                "active_scene": int(getattr(self, "active_scene", -1)),
                "active_roi": int(getattr(self, "active_roi", -1)),
                "label_mode": str(getattr(self, "label_mode", "per_roi")),
            }

        def _restore_undo_state(self, state: Dict[str, Any]) -> None:
            if not isinstance(state, dict) or "scenes" not in state:
                return
            self.scenes = list(state.get("scenes", []) or [])
            self.active_scene = int(state.get("active_scene", -1))
            self.active_roi = int(state.get("active_roi", -1))
            self.label_mode = str(state.get("label_mode", "per_roi") or "per_roi")

            # Clamp indices (scenes may have changed).
            if self.active_scene < 0 or self.active_scene >= len(self.scenes):
                self.active_scene = -1
                self.active_roi = -1
            else:
                rois = getattr(self.scenes[self.active_scene], "rois", []) or []
                if not rois:
                    self.active_roi = -1
                else:
                    self.active_roi = int(np.clip(self.active_roi, 0, len(rois) - 1))

            # Reset transient UI state that can reference removed objects.
            self.roi_tmp = None
            self.bound_tmp = None
            self.anchor_tmp = None
            self.dragging = False
            self.repicking = False
            self.adding = False
            self.drag_is_bound = False
            self.drag_is_anchor = False
            self.anchor_roi = -1
            self.naming_roi = -1
            self.name_buf = ""
            self.postview_active = False
            self.postview_roi = (-1, -1)

            # Cached hitboxes/thumbs depend on scene identity and must be rebuilt.
            try:
                self.scene_thumbs.clear()
                self.panel_hitboxes.clear()
                self.roi_panel_hitboxes.clear()
            except Exception:
                pass

            # Keep label metadata consistent after structural edits.
            try:
                self._rebuild_all_label_roles()
            except Exception:
                pass

            try:
                self.view.update()
            except Exception:
                pass

        def _push_undo(self, label: str, coalesce: Optional[str] = None) -> None:
            try:
                self.undo_stack.push_undo(self._undo_state(), label=label, coalesce_key=coalesce)
            except Exception as e:
                print(f"[undo] push failed: {e}")

        def undo_action(self) -> None:
            state, label = self.undo_stack.undo(self._undo_state())
            if state is None:
                return
            self._restore_undo_state(state)
            if label:
                print(f"[undo] {label}")

        def redo_action(self) -> None:
            state, label = self.undo_stack.redo(self._undo_state())
            if state is None:
                return
            self._restore_undo_state(state)
            if label:
                print(f"[redo] {label}")


        def _delete_active_roi(self):
            """Delete the currently selected ROI (and keep per-ROI series aligned)."""
            si = int(getattr(self, "active_scene", -1))
            if si < 0 or si >= len(self.scenes):
                return

            sc = self.scenes[si]
            rois = getattr(sc, "rois", None)
            if not rois:
                return

            ri = int(getattr(self, "active_roi", -1))
            if ri < 0 or ri >= len(rois):
                return

            # Snapshot BEFORE mutation.
            self._push_undo("Delete ROI")

            # Remove ROI object.
            del rois[ri]

            # Shift any dicts keyed by ROI index so exports remain correct.
            def _shift_series_dict(d: Any) -> Dict[int, Any]:
                if not isinstance(d, dict):
                    return {}
                out = {}
                for k, seq in d.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    if kk < ri:
                        out[kk] = seq
                    elif kk > ri:
                        out[kk - 1] = seq
                return out

            for attr in (
                "roi_cx", "roi_cy",
                "roi_vx", "roi_vy", "roi_vz",
                "roi_env",
                "roi_imp_in", "roi_imp_out",
                "roi_igE", "roi_igdE", "roi_curv",
                "roi_lowconf",
            ):
                try:
                    setattr(sc, attr, _shift_series_dict(getattr(sc, attr, {})))
                except Exception:
                    pass

            # label_primary maps label -> ROI index; shift indices after deletion.
            try:
                lp = dict(getattr(sc, "label_primary", {}) or {})
                lp2 = {}
                for lab, idx in lp.items():
                    try:
                        idx = int(idx)
                    except Exception:
                        continue
                    if idx == ri:
                        continue
                    lp2[str(lab)] = (idx - 1) if idx > ri else idx
                sc.label_primary = lp2
            except Exception:
                pass

            # Clamp selection.
            if len(rois) == 0:
                self.active_roi = -1
            else:
                self.active_roi = int(np.clip(ri, 0, len(rois) - 1))

            # Clear transient edit state that could refer to the old ROI.
            self.roi_tmp = None
            self.bound_tmp = None
            self.anchor_tmp = None
            self.dragging = False
            self.repicking = False
            self.adding = False
            self.drag_is_bound = False
            self.drag_is_anchor = False
            self.anchor_roi = -1

            # Keep internal label roles consistent.
            try:
                self._rebuild_label_roles_for_scene(si)
            except Exception:
                pass

            self.scenes[si] = sc


        def _rebuild_label_roles_for_scene(self, si: int):
            """Recompute Scene.label_primary from current ROI names."""
            if si < 0 or si >= len(self.scenes):
                return
            sc = self.scenes[si]
            from collections import defaultdict
            mapping = defaultdict(list)  # label -> [roi indices]

            for ri, r in enumerate(sc.rois):
                labels = _split_labels(getattr(r, "name", None), f"roi{ri}")
                for lab in labels:
                    mapping[lab].append(ri)

            # Clean existing mapping against current labels/ROIs
            lp = dict(getattr(sc, "label_primary", {}) or {})
            lp = {
                lab: ri
                for lab, ri in lp.items()
                if lab in mapping and ri in mapping[lab]
            }

            # Ensure every label has a primary (default = lowest ROI index)
            for lab, ris in mapping.items():
                if not ris:
                    continue
                if lab not in lp:
                    lp[lab] = min(ris)

            sc.label_primary = lp
            sc.label_mode = getattr(sc, "label_mode", getattr(self, "label_mode", "per_roi"))
            self.scenes[si] = sc

        def _rebuild_all_label_roles(self):
            for si in range(len(self.scenes)):
                self._rebuild_label_roles_for_scene(si)

        def _commit_name_buf(self):
            """Commit current name buffer into the active ROI and rebuild label roles."""
            si = self.active_scene
            ri = self.naming_roi
            if si < 0 or ri < 0 or si >= len(self.scenes):
                return
            self._push_undo("Rename ROI", coalesce="roi_name")
            buf = (self.name_buf or "").strip()
            self.scenes[si].rois[ri].name = buf or None
            self._rebuild_label_roles_for_scene(si)

        def _promote_active_roi_primary(self):
            """Mark active ROI as PRIMARY for all of its labels in this scene."""
            si = self.active_scene
            ri = self.active_roi
            if si < 0 or si >= len(self.scenes):
                return
            sc = self.scenes[si]
            if ri < 0 or ri >= len(sc.rois):
                return
            self._push_undo("Set label primary", coalesce="labels")
            r = sc.rois[ri]
            labels = _split_labels(getattr(r, "name", None), f"roi{ri}")
            if not labels:
                return
            lp = dict(getattr(sc, "label_primary", {}) or {})
            for lab in labels:
                lp[lab] = ri
            sc.label_primary = lp
            sc.label_mode = getattr(sc, "label_mode", getattr(self, "label_mode", "per_roi"))
            self.scenes[si] = sc
            print(f"[labels] ROI{ri} set PRIMARY for: {', '.join(labels)}")

        def draw_label_editor(self, img):
            """Inline label editor UI inside the ROI debug panel."""
            if self.naming_roi < 0:
                return
            if self.active_scene < 0 or self.active_scene >= len(self.scenes):
                return
            if not self.roi_panel_hitboxes:
                return  # panel not visible / no ROIs

            H, W = img.shape[:2]
            base_scale = float(np.clip(0.45 + 0.35*min(self.W,self.H)/720.0, 0.45, 0.9))

            for ri, (x0, y0, x1, y1) in self.roi_panel_hitboxes:
                if ri != self.naming_roi:
                    continue

                # darken the row background
                overlay = img.copy()
                cv.rectangle(overlay, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (60, 60, 60), -1)
                cv.addWeighted(overlay, 0.65, img, 0.35, 0, img)

                draw_text_clamped(img, f"Label ROI{ri}", x0 + 8, y0 + 18, (255,230,180), base_scale * 0.55)
                buf = self.name_buf or ""
                lines = buf.split("\n")

                line_scale = base_scale * 0.52
                line_h = int(14 * line_scale)  # rough vertical spacing

                for i, line in enumerate(lines):
                    # Cursor underscore only on the last line
                    suffix = "_" if i == len(lines) - 1 else ""
                    draw_text_clamped(
                        img,
                        line + suffix,
                        x0 + 8,
                        y0 + 34 + i * (line_h + 2),
                        (255,255,255),
                        line_scale,
                    )

                help_line = "Shift+Enter=new alias  Up/Down=other ROI  Esc=close"
                draw_text_clamped(img, help_line, x0 + 8, min(y1 - 8, y0 + 52), (210,210,210), base_scale * 0.40)
                break


        def _inst_v_a_j(self, vx, vy, vz):
            import numpy as _np
            dt = 1.0/max(1e-6, float(self.fps))
            vmag = _np.sqrt(vx*vx + vy*vy + vz*vz)
            n = len(vmag)
            v = float(vmag[-1]) if n else 0.0
            if n >= 2:
                a = float((vmag[-1] - vmag[-2]) / dt)
            else:
                a = 0.0
            if n >= 3:
                a_prev = float((vmag[-2] - vmag[-3]) / dt)
                j = float((a - a_prev) / dt)
            else:
                j = 0.0
            return v, abs(a), abs(j)

        def _live_impact_for_roi(self, si, ri):
            import numpy as _np
            sc = self.scenes[si]; r = sc.rois[ri]

            vx = _np.asarray(sc.roi_vx.get(ri, []), float)
            vy = _np.asarray(sc.roi_vy.get(ri, []), float)
            vz = _np.asarray(sc.roi_vz.get(ri, []), float)
            T  = min(len(vx), len(vy), len(vz))
            if T < 5:
                return None, 0.0

            # --- flow-based velocities (as before) ---
            vx_s = robust_smooth(vx, self.fps)
            vy_s = robust_smooth(vy, self.fps)
            vz_s = robust_smooth(vz, self.fps, win_ms=160, ema_tc_ms=240)

            dt = 1.0 / max(1e-6, float(self.fps))
            ax_s = _deriv_central(vx_s, dt); ay_s = _deriv_central(vy_s, dt); az_s = _deriv_central(vz_s, dt)
            jx_s = _deriv_central(ax_s, dt); jy_s = _deriv_central(ay_s, dt); jz_s = _deriv_central(az_s, dt)

            # keep legacy scalar score around if you ever want it
            _score_legacy, _pk_idx, _rel_idx, _hold = impact_score_cycles(
                vx_s, vy_s, vz_s, ax_s, ay_s, az_s, jx_s, jy_s, jz_s, self.fps,
                thr=getattr(r, "impact_thr_z", 2.0),
                min_interval_ms=getattr(r, "refractory_ms", 140),
                vmin_gate=float(getattr(r, "impact_min_speed_ps", 0.0)),
                fall_frac=getattr(r, "impact_fall_frac", 0.80)
            )

            # --- NEW: center-based velocities from ROI positions ---
            cx_seq = sc.roi_cx.get(ri, [])
            cy_seq = sc.roi_cy.get(ri, [])
            vx_pos_raw, vy_pos_raw = _center_vel_from_pos(cx_seq, cy_seq, self.fps)
            vx_pos_s = robust_smooth(vx_pos_raw, self.fps, win_ms=140, ema_tc_ms=200, mad_k=3.6)
            vy_pos_s = robust_smooth(vy_pos_raw, self.fps, win_ms=140, ema_tc_ms=200, mad_k=3.6)

            use_pos_channel = (
                bool(getattr(r, "anchor_user_set", False)) or
                str(getattr(r, "cmat_mode", "off")).lower() != "off"
            )

                        # --- flow-based impacts (primary channel) ---
            S_flow, in_flow, out_flow = _impacts_for_mode(r, vx_s, vy_s, vz_s, self.fps)

            if use_pos_channel and len(vx_pos_s) >= 5:
                S_pos, in_pos, out_pos = _impacts_for_mode(
                    r,
                    vx_pos_s,
                    vy_pos_s,
                    _np.zeros_like(vx_pos_s),
                    self.fps,
                )
                S_any, in_idx, out_idx = _pick_primary_impact_channel(
                    r, self.fps,
                    vx_s,     vy_s,     S_flow, in_flow, out_flow,
                    vx_pos_s, vy_pos_s, S_pos,  in_pos,  out_pos,
                )
            else:
                S_any = S_flow
                in_idx  = _np.asarray(in_flow,  int)
                out_idx = _np.asarray(out_flow, int)

            # Optional I/O gating for HUD
            io_pref = int(getattr(r, "impact_io", 0))
            if io_pref > 0:
                in_idx = _np.asarray([], int)
            if io_pref < 0:
                out_idx = _np.asarray([], int)

            # --- tail window: allow impacts that fired in the last ~80 ms ---
            evt = None
            T = len(S_any)
            if T == 0:
                return None, 0.0

            tail_allow = max(
                2,
                int(round(0.08 * self.fps)) + int(round(getattr(r, "impact_lead_ms", 40) * self.fps / 1000.0))
            )

            if out_idx.size and out_idx[-1] >= (T - tail_allow):
                evt = "OUT"
            elif in_idx.size and in_idx[-1] >= (T - tail_allow):
                evt = "IN"

            return evt, float(S_any[-1])


        def draw_roi_metrics(self, img, si, ri):
            import time
            sc = self.scenes[si]; r = sc.rois[ri]
            x,y,w,h = map(int, r.rect)
            vx = sc.roi_vx.get(ri, []); vy = sc.roi_vy.get(ri, []); vz = sc.roi_vz.get(ri, [])
            v,a,j = self._inst_v_a_j(np.asarray(vx,float), np.asarray(vy,float), np.asarray(vz,float))
            # numbers just above the box
            draw_text_clamped(img, f"v {v:6.2f}  a {a:6.2f}  j {j:6.2f}", x+4, y-8, (240,240,200), 0.45)

            # live impact label with brief hold
            evt, _ = self._live_impact_for_roi(si, ri)
            now = time.time()
            key = (si, ri)
            if evt:  # trigger hold
                self._impact_until[key] = now + 0.5
                self._impact_label[key] = evt
            if key in self._impact_until and now <= self._impact_until[key]:
                lbl = self._impact_label.get(key, "")
                col = (240,140,0) if lbl == "OUT" else (0,170,240)
                draw_text_clamped(img, f"IMPACT {lbl}", x + w//2 - 12, y - 24, col, 0.60)

        def draw_roi_debug_panel(self, img):
            H, W = img.shape[:2]
            if self.active_scene < 0 or not self.scenes[self.active_scene].rois:
                return

            sc = self.scenes[self.active_scene]
            rois = sc.rois
            self.roi_panel_hitboxes.clear()

            panel_w = self.PANEL_W
            x1 = W - 1
            x0 = x1 - panel_w
            x0 = max(0, x0)

            overlay = img.copy()
            cv.rectangle(overlay, (x0, 0), (x1, H), (30, 30, 30), -1)
            cv.addWeighted(overlay, 0.35, img, 0.65, 0, img)

            y = 8
            mode = getattr(sc, "label_mode", getattr(self, "label_mode", "per_roi"))
            mode_short = {"per_roi": "per-ROI", "per_label": "per-Label"}.get(mode, mode)
            draw_text_clamped(img, f"ROI Debug [{mode_short}]", x0 + 8, y + 14, (220, 220, 220), 0.55)
            y += 26


            max_rows = max(1, (H - y) // self.PANEL_ITEM_H)
            start = int(np.clip(self.roi_panel_scroll, 0, max(0, len(rois) - max_rows)))

            for idx in range(start, len(rois)):
                r = rois[idx]
                tile_y0 = y
                tile_y1 = y + self.PANEL_ITEM_H - 6
                if tile_y0 >= H:
                    break

                is_active = (idx == self.active_roi)
                cv.rectangle(img, (x0 + 4, tile_y0), (x1 - 4, tile_y1),
                            (0,0,255) if is_active else (200,200,200), 2, cv.LINE_AA)

                raw_name = r.name or f"ROI{idx}"
                name = raw_name.replace("\n", " / ")
                cmat_mode = str(getattr(r, "cmat_mode", "off")).lower()

                cmat_tag = cmat_mode
                if cmat_mode == "global":
                    # Prefer explicit profile label if you set it when pressing M
                    prof = str(getattr(self.tracker, "cmat_profile", "")).lower()

                    # If profile missing, infer from runtime max_h
                    if prof not in ("off", "chaotic", "moderate", "smooth"):
                        mh = int(getattr(self.tracker, "cmat_max_h", 0))
                        if   mh <= 0:   prof = "off"
                        elif mh <= 140: prof = "chaotic"   # ~120p
                        elif mh <= 210: prof = "moderate"  # ~180p
                        else:           prof = "smooth"    # ~240p

                    cmat_tag = prof
                anchor_ok = getattr(r, "_anchor_last_ok", False)
                sim = float(getattr(r, "_anchor_last_sim", 0.0))

                # Label role summary: lab:P (primary) / lab:S (secondary)
                role_bits = []
                labels = _split_labels(getattr(r, "name", None), f"roi{idx}")
                lp = getattr(sc, "label_primary", {})
                for lab in labels:
                    pri = lp.get(lab, None)
                    if pri == idx:
                        role_bits.append(f"{lab}:P")
                    elif lab in lp:
                        role_bits.append(f"{lab}:S")
                roles_txt = ("  " + " ".join(role_bits)) if role_bits else ""

                line1 = f"{name}  [{cmat_tag}]{roles_txt}"
                line2 = f"vx {r.vx_ps:+.2f} vy {r.vy_ps:+.2f}  anc {'OK' if anchor_ok else 'LOST'} {sim:.2f}"
                vz_mode = str(getattr(r, "vz_mode", "curv")).lower()
                vz_mix  = float(getattr(r, "vz25_mix", 0.75))
                line3 = f"vz {vz_mode}  mix {vz_mix:.2f}"


                draw_text_clamped(img, line1, x0 + 8, tile_y0 + 18, (240,240,200), 0.45)
                draw_text_clamped(img, line2, x0 + 8, tile_y0 + 34, (220,220,220), 0.42)
                draw_text_clamped(img, line3, x0 + 8, tile_y0 + 50, (210,210,210), 0.40)


                self.roi_panel_hitboxes.append((idx, (x0, tile_y0, x1, tile_y1)))
                y += self.PANEL_ITEM_H

        def dump_roi_debug_csv(self, si, ri, prefix=None):
            import pandas as _pd, numpy as _np, os
            sc = self.scenes[si]; r = sc.rois[ri]
            vx = _np.asarray(sc.roi_vx.get(ri, []), float)
            vy = _np.asarray(sc.roi_vy.get(ri, []), float)
            vz = _np.asarray(sc.roi_vz.get(ri, []), float)
            t  = _np.asarray(sc.times[:len(vx)], float)
            if len(t) < 5:
                print("[debug] not enough samples"); return
            vx_s = robust_smooth(vx, self.fps); vy_s = robust_smooth(vy, self.fps)
            vz_s = robust_smooth(vz, self.fps, win_ms=160, ema_tc_ms=240)
            dt = 1.0/max(1e-6, float(self.fps))
            ax_s = _deriv_central(vx_s, dt); ay_s = _deriv_central(vy_s, dt); az_s = _deriv_central(vz_s, dt)
            jx_s = _deriv_central(ax_s, dt); jy_s = _deriv_central(ay_s, dt); jz_s = _deriv_central(az_s, dt)
            score, peaks = impact_score_and_peaks(vx_s,vy_s,vz_s, ax_s,ay_s,az_s, jx_s,jy_s,jz_s, self.fps,
                                                thr=getattr(r,"impact_thr_z",2.0), min_interval_ms=getattr(r,"refractory_ms",140))
            vP = vx_s if _np.mean(_np.abs(vx_s)) >= _np.mean(_np.abs(vy_s)) else vy_s
            out_idx, in_idx = split_out_in(peaks, vP, self.fps)
            df = _pd.DataFrame(dict(
                t=t, vx=vx, vy=vy, vz=vz, vx_s=vx_s, vy_s=vy_s, vz_s=vz_s,
                ax_s=ax_s, ay_s=ay_s, az_s=az_s, jx_s=jx_s, jy_s=jy_s, jz_s=jz_s,
                score=score
            ))
            stem = os.path.splitext(os.path.basename(video_path))[0]
            out = f"{stem}_S{si:02d}_R{ri:02d}_debug.csv" if not prefix else f"{prefix}.csv"
            df.to_csv(out, index=False)
            print("[debug] ROI CSV →", out)


        def _draw_anchor_panel(self, img):
            # No active scene or no scenes at all
            if self.active_scene < 0 or self.active_scene >= len(self.scenes):
                return

            sc = self.scenes[self.active_scene]

            # No ROIs in this scene
            if not sc.rois:
                return

            # active_roi can be stale (after deletions / scene changes) – guard it
            if self.active_roi < 0 or self.active_roi >= len(sc.rois):
                # you can either:
                #   - clamp to last ROI, or
                #   - just bail out. I'll bail out to avoid surprising jumps.
                return

            r = sc.rois[self.active_roi]

            if not getattr(r, "anchor_user_set", False):
                return
            if getattr(r, "_anchor_template0", None) is None or getattr(r, "_anchor_template", None) is None:
                return

            H, W = img.shape[:2]
            # keep it above the timeline
            panel_h = 90
            panel_w = 2 * 72 + 24
            margin = 8
            y1 = H - self.TIMELINE_H - margin
            y0 = max(0, y1 - panel_h)
            x1 = W - margin
            x0 = max(0, x1 - panel_w)

            overlay = img.copy()
            cv.rectangle(overlay, (x0, y0), (x1, y1), (15, 15, 15), -1)
            cv.addWeighted(overlay, 0.55, img, 0.45, 0, img)
            cv.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1, cv.LINE_AA)

            # normalize + resize original and current patches
            orig = r._anchor_template0
            curr = r._anchor_template

            def _prep(p):
                if p.ndim == 2:
                    p = cv.cvtColor(p, cv.COLOR_GRAY2BGR)
                return cv.resize(p, (72, 72), interpolation=cv.INTER_AREA)

            orig_v = _prep(orig)
            curr_v = _prep(curr)

            img[y0+20:y0+20+72, x0+8:x0+8+72] = orig_v
            img[y0+20:y0+20+72, x0+8+72+8:x0+8+72+8+72] = curr_v

            sim = float(getattr(r, "_anchor_last_sim", 0.0))
            ok = bool(getattr(r, "_anchor_last_ok", False))

            draw_text_clamped(img, "anchor ORIG", x0+8, y0+14, (220,220,220), 0.4)
            draw_text_clamped(img, "anchor CURR", x0+8+72+8, y0+14, (220,220,220), 0.4)
            draw_text_clamped(
                img,
                f"sim={sim:.2f}  {'LOCK' if ok else 'LOST'}",
                x0+8, y1-6,
                (0,210,0) if ok else (0,0,255),
                0.45
            )


        def _draw_postview(self, img):
            si, ri = self.postview_roi
            if si < 0 or ri < 0: return
            sc = self.scenes[si]
            T = min(len(sc.times),
                    len(sc.roi_cx.get(ri,[])))
            if T < 8: return
            # pull series
            cx = np.asarray(sc.roi_cx.get(ri,[])[:T], float)
            cy = np.asarray(sc.roi_cy.get(ri,[])[:T], float)
            vx = np.asarray(sc.roi_vx.get(ri,[])[:T], float)
            vy = np.asarray(sc.roi_vy.get(ri,[])[:T], float)
            vz = np.asarray(sc.roi_vz.get(ri,[])[:T], float)
            # post-process
            px01, py01 = smooth_and_scale_xy(cx, cy, self.fps, up=3)
            vx_s = robust_smooth(vx, self.fps); vy_s = robust_smooth(vy, self.fps); vz_s = robust_smooth(vz, self.fps, win_ms=160, ema_tc_ms=240)
            dt = 1.0/max(1e-6, float(self.fps))
            ax_s = _deriv_central(vx_s, dt); ay_s = _deriv_central(vy_s, dt); az_s = _deriv_central(vz_s, dt)
            jx_s = _deriv_central(ax_s, dt); jy_s = _deriv_central(ay_s, dt); jz_s = _deriv_central(az_s, dt)
            score, peaks = impact_score_and_peaks(vx_s,vy_s,vz_s, ax_s,ay_s,az_s, jx_s,jy_s,jz_s,
                                                  self.fps, thr=2.0, min_interval_ms=getattr(sc.rois[ri],'refractory_ms',140))
            vP = vx_s if np.mean(np.abs(vx_s)) >= np.mean(np.abs(vy_s)) else vy_s
            out_idx, in_idx = split_out_in(peaks, vP, self.fps)
            # panel rect
            H,W = img.shape[:2]; PW = min(420, int(W*0.46)); PH = min(220, int(H*0.32))
            x0 = W-PW-10; y0 = 10
            overlay = img.copy()
            cv.rectangle(overlay, (x0,y0), (x0+PW, y0+PH), (15,15,15), -1)
            cv.addWeighted(overlay, 0.75, img, 0.25, 0, img)
            cv.rectangle(img, (x0,y0), (x0+PW, y0+PH), (90,90,90), 1)
            draw_text_clamped(img, f"ROI {ri}  Post-Processed  (ESC to close)", x0+8, y0+18, (240,240,200), 0.55)

            # tiny plot helper
            def spark(y, yy, name):
                yy0 = y0+yy; hh = 26; xx0=x0+8; ww=PW-16
                cv.rectangle(img, (xx0, yy0-18), (xx0+ww, yy0+hh), (32,32,32), 1)
                if len(y)==0: return
                ys = np.asarray(y, float); ys = (ys - ys.min())/(ys.max()-ys.min()+1e-12)
                pts = np.stack([np.linspace(xx0, xx0+ww-1, len(ys)), yy0+hh - (hh*ys)], axis=1).astype(np.int32)
                cv.polylines(img, [pts], False, (200,200,200), 1, cv.LINE_AA)
                draw_text_clamped(img, name, xx0+2, yy0-2, (210,210,210), 0.5)
            # rows
            spark(px01, 34, "X spline (0..1)")
            spark(py01, 68, "Y spline (0..1)")
            spark(normalize_unsigned01(np.cumsum(vz_s)*dt), 102, "Z pos (from vz)")
            spark(normalize_unsigned01(np.sqrt(vx_s*vx_s+vy_s*vy_s+vz_s*vz_s)), 136, "Speed")
            spark(normalize_unsigned01(np.sqrt(ax_s*ax_s+ay_s*ay_s+az_s*az_s)), 170, "Accel")
            # score and markers
            yy = 204
            score01 = normalize_unsigned01(score)
            yy0 = y0+yy; hh = 26; xx0=x0+8; ww=PW-16
            cv.rectangle(img, (xx0, yy0-18), (xx0+ww, yy0+hh), (32,32,32), 1)
            if len(score01):
                ys = score01; pts = np.stack([np.linspace(xx0, xx0+ww-1, len(ys)), yy0+hh - (hh*ys)], axis=1).astype(np.int32)
                cv.polylines(img, [pts], False, (90,200,90), 1, cv.LINE_AA)
                for p in out_idx: 
                    x = int(xx0 + (p/(len(ys)-1+1e-12))*ww); cv.line(img, (x,yy0), (x,yy0+hh), (240,140,0), 1)
                for p in in_idx:  
                    x = int(xx0 + (p/(len(ys)-1+1e-12))*ww); cv.line(img, (x,yy0), (x,yy0+hh), (0,170,240), 1)
                draw_text_clamped(img, "Impact score (orange=OUT, blue=IN)", xx0+2, yy0-2, (210,210,210), 0.5)

        def dump_roi_debug(self, si=None, ri=None, tail_sec=None):
            import os, pandas as pd, numpy as np
            if si is None: si = self.active_scene
            if si < 0 or si >= len(self.scenes): return None
            sc = self.scenes[si]
            if ri is None: ri = self.active_roi
            if ri < 0 or ri >= len(sc.rois): return None
            T = len(sc.times)
            if T == 0: return None
            fps = self.fps; dt = 1.0/max(1e-6, float(fps))

            def _pad(dic, key, default):
                seq = dic.get(key, []) if isinstance(dic, dict) else []
                if len(seq)==0: return np.full(T, float(default), float)
                if len(seq)<T:  seq = list(seq) + [seq[-1]]*(T-len(seq))
                if len(seq)>T:  seq = list(seq)[:T]
                return np.asarray(seq, float)

            r  = sc.rois[ri]
            cx = _pad(getattr(sc,'roi_cx',{}), ri, (r.rect[0]+r.rect[2]/2.0))
            cy = _pad(getattr(sc,'roi_cy',{}), ri, (r.rect[1]+r.rect[3]/2.0))
            vx = _pad(getattr(sc,'roi_vx',{}), ri, 0.0)
            vy = _pad(getattr(sc,'roi_vy',{}), ri, 0.0)
            vz = _pad(getattr(sc,'roi_vz',{}), ri, 0.0)

            # robust post-process (same math used in export)
            vx_s = robust_smooth(vx, fps, win_ms=140, ema_tc_ms=200, mad_k=3.6)
            vy_s = robust_smooth(vy, fps, win_ms=140, ema_tc_ms=200, mad_k=3.6)
            vz_s = robust_smooth(vz, fps, win_ms=160, ema_tc_ms=240, mad_k=3.8)
            ax_s = _deriv_central(vx_s, dt); ay_s = _deriv_central(vy_s, dt); az_s = _deriv_central(vz_s, dt)
            jx_s = _deriv_central(ax_s, dt); jy_s = _deriv_central(ay_s, dt); jz_s = _deriv_central(az_s, dt)

            S, peaks = impact_score_and_peaks(vx_s,vy_s,vz_s, ax_s,ay_s,az_s, jx_s,jy_s,jz_s,
                                            fps,
                                            vmin_gate=0.02,
                                            thr=getattr(r, "impact_thr_z", 2.0),
                                            min_interval_ms=getattr(r, "refractory_ms", 140))
            vP = vx_s if np.mean(np.abs(vx_s)) >= np.mean(np.abs(vy_s)) else vy_s
            out_idx, in_idx = split_out_in(peaks, vP, fps, look_ms=40, v_zero=0.02)
            impact_dir = np.zeros(T, int); impact_dir[out_idx] = +1; impact_dir[in_idx] = -1

            # optional tail
            start = 0 if not tail_sec else max(0, T - int(round(float(tail_sec)*fps)))
            sl = slice(start, T)

            df = pd.DataFrame({
                "time": np.asarray(sc.times)[sl],
                "posx": cx[sl], "posy": cy[sl],
                "vx": vx[sl], "vy": vy[sl], "vz": vz[sl],
                "vx_s": vx_s[sl], "vy_s": vy_s[sl], "vz_s": vz_s[sl],
                "ax_s": ax_s[sl], "ay_s": ay_s[sl], "az_s": az_s[sl],
                "jx_s": jx_s[sl], "jy_s": jy_s[sl], "jz_s": jz_s[sl],
                "speed": np.sqrt(vx_s*vx_s + vy_s*vy_s + vz_s*vz_s)[sl],
                "acc":   np.sqrt(ax_s*ax_s + ay_s*ay_s + az_s*az_s)[sl],
                "jerk":  np.sqrt(jx_s*jx_s + jy_s*jy_s + jz_s*jz_s)[sl],
                "impact_score": S[sl],
                "impact_dir": impact_dir[sl],   # +1 OUT, -1 IN, 0 none
                "refractory_ms": getattr(r,"refractory_ms",140),
                "impact_thr_z":  getattr(r,"impact_thr_z",2.0),
                "fb_levels":     getattr(r,"fb_levels",5),
            })

            stem = os.path.splitext(os.path.basename(video_path))[0]
            safe_name = (r.name or f"roi{ri}").replace(" ", "_")
            path = f"{stem}_S{si:02d}_{safe_name}_debug.csv"
            df.to_csv(path, index=False)
            try:
                QtGui.QGuiApplication.clipboard().setText(path)
            except Exception:
                pass
            print(f"[debug] ROI CSV → {path}")
            return path
            
        def _read_frame(self, target_idx: int):
            # If next frame, avoid random seek: just read
            if self._dec_pos + 1 == target_idx:
                ok, fr = self.cap.read()
                if ok:
                    self._dec_pos = target_idx
                    return True, fr
            # Fallback: random seek once, then continue sequentially
            self.cap.set(cv.CAP_PROP_POS_FRAMES, int(target_idx))
            ok, fr = self.cap.read()
            if ok:
                self._dec_pos = int(target_idx)
            return ok, fr

        def _seek_to_frame(self, idx: int):
            idx = int(np.clip(idx, 0, self.N-1))
            self.frame_idx = idx
            self.playing = False           # skimming never computes
            self._seeking = True
            self.prev_gray = None          # drop flow baseline after large jump
            self._dec_pos = -1
            self.prev_gray = None

            self.fast_scrub_until = time.time() + 0.12  # lighter HUD while scrubbing

        def _read_preview_at(self, idx: int):
            idx = int(np.clip(idx, 0, self.N-1))
            cap = self.thumb_cap
            cap.set(cv.CAP_PROP_POS_FRAMES, max(0, idx - 2))
            cap.set(cv.CAP_PROP_POS_MSEC, (idx / max(1e-6, self.fps)) * 1000.0)
            fr = None; ok = False
            for _ in range(8):
                ok, fr = cap.read()
                if not ok: break
                pos = int(round(cap.get(cv.CAP_PROP_POS_FRAMES))) - 1
                if pos >= idx: break
            if ok and fr is not None and PROC_SCALE != 1.0:
                fr = cv.resize(fr, (self.W, self.H), interpolation=cv.INTER_AREA)
            return fr


        def draw_timeline(self, img):
            H_, W_ = img.shape[:2]
            y0 = H_ - self.TIMELINE_H

            # --- translucent bg + progress ---
            overlay = img.copy()
            cv.rectangle(overlay, (0, y0), (W_-1, H_-1), (25,25,25), -1)
            px = int(round((self.frame_idx / max(1, self.N-1)) * (W_-1)))
            cv.rectangle(overlay, (0, y0), (px, H_-1), (40, 60, 90), -1)
            alpha = getattr(self, "TIMELINE_ALPHA", 0.35)
            cv.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
            cv.rectangle(img, (0, y0), (W_-1, H_-1), (90, 90, 90), 1)


            # scene spans
            for si, sc in enumerate(self.scenes):
                eff_end = self.effective_end(si)
                x0 = int(round((sc.start   / max(1, self.N-1)) * (W_-1)))
                x1 = int(round((eff_end    / max(1, self.N-1)) * (W_-1)))

                cv.rectangle(img, (x0, y0+2), (x1, max(y0+5, y0+2)),
                            (0, 0, 255) if si == self.active_scene else (70, 70, 70), -1)

            # hover preview
            if self.timeline_hover_idx >= 0:
                x = int(round((self.timeline_hover_idx / max(1, self.N-1)) * (W_-1)))
                cv.line(img, (x, y0), (x, H_-1), (180, 180, 180), 1)

                now = time.time()
                if (not self.playing) and now >= self._next_hover_grab_time and self.hover_thumb_at != self.timeline_hover_idx:
                    fr_pv = self._read_preview_at(self.timeline_hover_idx)
                    if fr_pv is not None:
                        pvW = min(220, self.W // 4)
                        pvH = max(1, int(pvW * self.H / max(1, self.W)))
                        self.hover_thumb = cv.resize(fr_pv, (pvW, pvH), interpolation=cv.INTER_AREA)
                        self.hover_thumb_at = self.timeline_hover_idx
                    self._next_hover_grab_time = now + 0.05  # throttle

                if self.hover_thumb is not None:
                    pv = self.hover_thumb; ph, pw = pv.shape[:2]
                    px = int(np.clip(x - pw // 2, 2, W_-2 - pw))
                    py = max(2, y0 - ph - 6)
                    overlay = img.copy()
                    cv.rectangle(overlay, (px-2, py-2), (px+pw+2, py+ph+2), (0,0,0), -1)
                    cv.addWeighted(overlay, 0.4, img, 0.6, 0, img)
                    img[py:py+ph, px:px+pw] = pv
                    draw_text_clamped(img,
                                    f"{self.timeline_hover_idx}/{self.N-1}  {self.timeline_hover_idx/self.fps:.02f}s",
                                    px+4, py+ph+14, (240,240,200), 0.5)


        # ---------- helpers ported from your run() ----------
        # was:
        # def scene_at(self, i):
        #     for si, sc in enumerate(self.scenes):
        #         if sc.in_range(i, self.N): return si
        #     return -1

        def scene_at(self, i):
            for si, sc in enumerate(self.scenes):
                if sc.start <= i <= self.effective_end(si):
                    return si
            return -1


        def sort_scenes(self): self.scenes.sort(key=lambda s: s.start)
        def neighbor_bounds(self, si):
            """Return (prev_end, next_start) bounds for clamping scene edits.

            - prev_end uses effective_end() so open scenes behave sanely.
            - next_start is the next scene's start (or N if none).
            """
            prev_end = -1
            next_start = self.N
            if si > 0:
                prev_end = int(self.effective_end(si - 1))
            if si + 1 < len(self.scenes):
                next_start = int(self.scenes[si + 1].start)
            return prev_end, next_start
        def effective_end(self, si: int) -> int:
            if si < 0 or si >= len(self.scenes): return self.N - 1
            return effective_end_for_scene(si, self.scenes, self.N)

        def _truncate_scene_to_time(self, sc: Scene, t_now: float):
            """Hard-truncate per-scene sampled series when re-running earlier time.

            This keeps time-aligned series dictionaries consistent (including impacts and structure lanes).
            """
            eps = 0.5 / max(1e-6, float(self.fps))  # half-frame guard
            if not sc.times:
                return

            # First index with time >= (t_now - eps)
            cut = next((i for i, tt in enumerate(sc.times) if tt >= (t_now - eps)), None)
            if cut is None:
                return  # nothing to trim

            sc.times = list(sc.times[:cut])
            if getattr(sc, "dup_flags", None):
                sc.dup_flags = list(sc.dup_flags[:cut])

            for attr in (
                "roi_cx", "roi_cy",
                "roi_vx", "roi_vy", "roi_vz",
                "roi_env",
                "roi_imp_in", "roi_imp_out",
                "roi_igE", "roi_igdE", "roi_curv",
                "roi_lowconf",
            ):
                dic = getattr(sc, attr, None)
                if isinstance(dic, dict):
                    for k, seq in list(dic.items()):
                        dic[k] = list((seq or [])[:cut])

            # Reset per-ROI refractory markers so a re-run isn't blocked by future history.
            for r in (sc.rois or []):
                try:
                    r._last_impact_idx = -10**9
                    r._impact_flash_until = 0.0
                except Exception:
                    pass
        def split_scene(self, si: int, at_idx: int):
            """Split a scene at the playhead.

            - Left scene becomes explicit-ended at (at_idx - 1).
            - Right scene inherits ROI layout and receives the tail of all time-aligned series.
            """
            if si < 0 or si >= len(self.scenes):
                return
            scL = self.scenes[si]
            eff_end = self.effective_end(si)
            if at_idx <= scL.start or at_idx > eff_end:
                print(f"[scene] split ignored (at={at_idx} outside S{si} [{scL.start},{eff_end}])")
                return

            # Snapshot BEFORE mutation.
            self._push_undo("Split scene")

            t_split = float(at_idx) / float(self.fps)

            # Right-hand scene inherits ROI layout (fresh ROI objects).
            rois_copy = [roi_replace(r) for r in (scL.rois or [])]
            scR = Scene(start=int(at_idx), end=scL.end, rois=rois_copy)

            # Preserve label metadata mode.
            scR.label_mode = getattr(scL, "label_mode", getattr(self, "label_mode", "per_roi"))

            # Partition measured samples at t_split.
            times = np.asarray(scL.times or [], float)
            cut = int(next((i for i, tt in enumerate(times) if tt >= t_split), len(times)))

            def _move_tail(dic_src):
                outL = {}
                outR = {}
                if not isinstance(dic_src, dict):
                    return outL, outR
                for k, seq in dic_src.items():
                    seq = list(seq or [])
                    outL[int(k)] = seq[:cut]
                    outR[int(k)] = seq[cut:]
                return outL, outR

            # Move times/dup flags
            scR.times = list((scL.times or [])[cut:])
            scL.times = list((scL.times or [])[:cut])
            if getattr(scL, "dup_flags", None):
                scR.dup_flags = list(scL.dup_flags[cut:])
                scL.dup_flags = list(scL.dup_flags[:cut])

            # Move ALL per-ROI time-aligned series dicts.
            for attr in (
                "roi_cx", "roi_cy",
                "roi_vx", "roi_vy", "roi_vz",
                "roi_env",
                "roi_imp_in", "roi_imp_out",
                "roi_igE", "roi_igdE", "roi_curv",
                "roi_lowconf",
            ):
                d = getattr(scL, attr, None)
                if isinstance(d, dict):
                    dl, dr = _move_tail(d)
                    setattr(scL, attr, dl)
                    setattr(scR, attr, dr)

            # Left scene gets explicit end; right keeps original (possibly None) end.
            scL.end = int(at_idx) - 1

            # Insert and reindex; make right scene active.
            self.scenes[si] = scL
            self.scenes.insert(si + 1, scR)
            self.sort_scenes()
            try:
                self.scene_thumbs.clear()
            except Exception:
                pass

            # Refresh label roles now that indices are stable.
            try:
                self._rebuild_all_label_roles()
            except Exception:
                pass

            self.active_scene = int(self.scenes.index(scR))
            self.active_roi = int(np.clip(self.active_roi, -1, len(scR.rois) - 1)) if scR.rois else -1

            # Auto-export the left scene if it has samples.
            if scL.times:
                try:
                    self.export_scene(int(self.scenes.index(scL)))
                except Exception:
                    pass

            print(
                f"[scene] split at f{at_idx} → "
                f"S{self.scenes.index(scL)} [{scL.start},{scL.end}] + "
                f"S{self.active_scene} [{scR.start},{'?' if scR.end is None else scR.end}]"
            )
        def set_scene_start(self, si, new_start):
            if si < 0 or si >= len(self.scenes):
                return

            # Snapshot BEFORE mutation.
            self._push_undo("Set scene start", coalesce="scene_bounds")

            prev_end, next_start = self.neighbor_bounds(si)
            sc = self.scenes[si]

            # Clamp: must be after previous scene end, and before this scene end/next scene start.
            hard_end = int(sc.end) if sc.end is not None else int(next_start - 1)
            new_start = int(np.clip(int(new_start), int(prev_end + 1), int(hard_end)))

            sc.start = new_start
            self.scenes[si] = sc
            self.sort_scenes()
            print(f"[scene] set start scene {si} -> {new_start}")
        def set_scene_end(self, si, new_end):
            if si < 0 or si >= len(self.scenes):
                return

            # Snapshot BEFORE mutation.
            self._push_undo("Set scene end", coalesce="scene_bounds")

            _, next_start = self.neighbor_bounds(si)
            sc = self.scenes[si]
            new_end = int(np.clip(int(new_end), int(sc.start), int(next_start - 1)))
            sc.end = new_end
            self.scenes[si] = sc
            print(f"[scene] set end scene {si} -> {new_end}")

            # If we already have samples, export exactly once now.
            # If not, defer export until playback/sampling crosses this end.
            if sc.times:
                setattr(sc, "_export_pending", False)
                try:
                    self.export_scene(si)
                except Exception:
                    pass
            else:
                setattr(sc, "_export_pending", True)
        def clear_scene_end(self, si):
            if si < 0 or si >= len(self.scenes):
                return

            # Snapshot BEFORE mutation.
            self._push_undo("Clear scene end", coalesce="scene_bounds")

            self.scenes[si].end = None
            setattr(self.scenes[si], "_export_pending", False)
            print(f"[scene] cleared end for scene {si}")
        def export_scene(self, si):
            if si<0 or si>=len(self.scenes): return
            sc = self.scenes[si]
            csv_path = save_scene_csv_and_push(video_path, si, sc, self.fps, self.W, self.H, robust_gamma=1.0, push=True)
            # Optional: AI sidecar inference on export (does NOT change the CSV schema)
            if self.ai_cfg.enabled and self.ai_cfg.on_export:
                try:
                    self.ai_run_on_export(si, sc, csv_path)
                except Exception:
                    pass
            return csv_path


        # --------------------------- AI Assist methods ---------------------------

        def _ai_toast(self, lines, ttl: float = 3.0):
            if isinstance(lines, str):
                lines = [lines]
            self.ai_hud_lines = [str(x) for x in (lines or [])][:8]
            self.ai_hud_until = time.time() + float(ttl)
            for l in self.ai_hud_lines:
                print("[AI]", l)

        def _ai_ready(self, need_vision: bool = False) -> bool:
            if not getattr(self, "ai_cfg", None) or not self.ai_cfg.enabled:
                self._ai_toast("AI disabled (run with --ai)", ttl=2.5)
                return False
            if self.ai_cfg.require_policy_ack and not self.ai_cfg.policy_ack:
                self._ai_toast("AI disabled: add --i-accept-openai-policy", ttl=4.0)
                return False
            if not self.ai.ready():
                self._ai_toast(f"AI not ready: {self.ai.last_error or 'unknown error'}", ttl=4.0)
                return False
            if need_vision and str(self.ai_cfg.vision).lower() == "off":
                self._ai_toast("AI vision off (enable: --ai-vision edges|crop|full)", ttl=4.0)
                return False
            return True

        def _get_active_scene_obj(self) -> Optional["Scene"]:
            if self.active_scene < 0 or self.active_scene >= len(self.scenes):
                return None
            return self.scenes[self.active_scene]

        def _scene_motion_summary(self, sc: "Scene", max_points: int = 420) -> Dict[str, Any]:
            # Downsample to keep token cost sane.
            times = list(sc.times or [])
            n = len(times)
            if n <= 0:
                return {"n": 0, "times": [], "rois": []}

            if n > max_points:
                idxs = np.linspace(0, n-1, max_points).astype(int).tolist()
            else:
                idxs = list(range(n))

            def sample(seq):
                if not seq:
                    return []
                return [float(seq[i]) for i in idxs if 0 <= i < len(seq)]

            rois_out = []
            for ri, roi in enumerate(sc.rois or []):
                rois_out.append({
                    "roi_index": int(ri),
                    "name": str(getattr(roi, "name", "") or ""),
                    "series": {
                        "cx": sample(sc.roi_cx.get(ri, [])),
                        "cy": sample(sc.roi_cy.get(ri, [])),
                        "vx": sample(sc.roi_vx.get(ri, [])),
                        "vy": sample(sc.roi_vy.get(ri, [])),
                        "vz": sample(sc.roi_vz.get(ri, [])),
                        "env": sample(sc.roi_env.get(ri, [])),
                        "imp_in": sample(sc.roi_imp_in.get(ri, [])),
                        "imp_out": sample(sc.roi_imp_out.get(ri, [])),
                        "lowconf": [bool(x) for x in (sc.roi_lowconf.get(ri, []) or [])[:len(sample(sc.roi_vx.get(ri, [])))]],
                    }
                })

            return {
                "fps": float(self.fps),
                "scene_start_frame": int(sc.start),
                "scene_end_frame": int(sc.end) if sc.end is not None else None,
                "n": int(n),
                "times": [float(times[i]) for i in idxs],
                "rois": rois_out,
            }

        def _roi_crop_safe(self, frame_bgr: np.ndarray, rect_xywh: Tuple[int,int,int,int], pad_frac: float = 0.0) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
            x, y, w, h = rect_xywh
            if pad_frac > 0:
                pad = int(round(pad_frac * max(w, h)))
                x -= pad; y -= pad; w += 2*pad; h += 2*pad
            x, y, w, h = clamp_rect(x, y, w, h, frame_bgr.shape[1], frame_bgr.shape[0])
            crop = frame_bgr[y:y+h, x:x+w].copy()
            return crop, (x, y, w, h)

        def ai_suggest_roi_tags(self):
            if not self._ai_ready(need_vision=(self.ai_cfg.vision != "off" and self.ai_cfg.vision is not None)):
                return
            sc = self._get_active_scene_obj()
            if sc is None or not sc.rois:
                self._ai_toast("No active scene/ROIs", ttl=2.5)
                return
            if self._cached_frame is None:
                self._ai_toast("No frame cached yet", ttl=2.5)
                return

            roi_summaries = []
            images = []
            use_vision = (str(self.ai_cfg.vision).lower() != "off")
            vmode = str(self.ai_cfg.vision).lower()

            for i, roi in enumerate(sc.rois):
                x, y, w, h = roi.rect
                s = {
                    "roi_index": int(i),
                    "name": str(roi.name or ""),
                    "rect_xywh": [int(x), int(y), int(w), int(h)],
                    "size_px": int(max(0, w*h)),
                    "last_speed_ps": float(getattr(roi, "last_speed", 0.0) or 0.0),
                    "lowconf": bool(getattr(roi, "_frame_lowconf", False)),
                }
                if use_vision:
                    crop, _ = self._roi_crop_safe(self._cached_frame, roi.rect, pad_frac=0.05)
                    # For tags, keep payload small.
                    img_url = _bgr_to_data_url(crop, max_side=256, mode=("edges" if vmode=="edges" else "crop"))
                    s["image_i"] = len(images)
                    images.append(img_url)
                roi_summaries.append(s)

            res = self.ai.suggest_roi_tags(roi_summaries=roi_summaries, images=images if use_vision else None)
            if not res:
                self._ai_toast(f"AI tags failed: {self.ai.last_error}", ttl=4.0)
                return

            applied = 0
            lines = ["ROI tag suggestions:"]
            for item in (res.get("rois") or []):
                try:
                    ri = int(item.get("roi_index", -1))
                    nm = str(item.get("suggested_name", "") or "").strip()
                    cf = float(item.get("confidence", 0.0) or 0.0)
                    if 0 <= ri < len(sc.rois) and nm:
                        sc.rois[ri].ai_tag_suggested = nm
                        sc.rois[ri].ai_tag_conf = cf
                        if self.ai_cfg.auto_apply_tags and not sc.rois[ri].name:
                            sc.rois[ri].name = nm
                            applied += 1
                        lines.append(f"  [{ri}] -> {nm}  (c={cf:.2f})")
                except Exception:
                    continue

            if self.ai_cfg.auto_apply_tags:
                lines.append(f"(auto-applied to empty names: {applied})")
            self._ai_toast(lines[:8], ttl=6.0)

        def ai_infer_impacts(self):
            if not self._ai_ready(need_vision=False):
                return
            sc = self._get_active_scene_obj()
            if sc is None:
                self._ai_toast("No active scene", ttl=2.5)
                return
            motion = self._scene_motion_summary(sc, max_points=520)
            if int(motion.get("n", 0)) < 8:
                self._ai_toast("Not enough motion data yet", ttl=2.5)
                return
            res = self.ai.infer_impacts(motion_summary=motion)
            if not res:
                self._ai_toast(f"AI impacts failed: {self.ai.last_error}", ttl=4.0)
                return
            sc.ai_impacts = list(res.get("events") or [])
            self._ai_toast([f"AI impacts: {len(sc.ai_impacts)} events (stored in scene.ai_impacts)"], ttl=4.0)

        def ai_suggest_flux_fixes(self):
            if not self._ai_ready(need_vision=False):
                return
            sc = self._get_active_scene_obj()
            if sc is None:
                self._ai_toast("No active scene", ttl=2.5)
                return
            if not getattr(sc, "ai_impacts", None):
                self.ai_infer_impacts()
            impacts = list(getattr(sc, "ai_impacts", []) or [])
            motion = self._scene_motion_summary(sc, max_points=520)
            res = self.ai.suggest_flux_fixes(motion_summary=motion, impacts=impacts)
            if not res:
                self._ai_toast(f"AI flux-fix failed: {self.ai.last_error}", ttl=4.0)
                return
            sc.ai_flux_fix = dict(res)
            g = (sc.ai_flux_fix.get("global") or {})
            self._ai_toast([
                f"AI flux-fix stored (scene.ai_flux_fix). Suggested smoothing_ms={g.get('suggested_smoothing_ms')}"
            ], ttl=5.0)

        def ai_run_on_export(self, si: int, sc: "Scene", csv_path: Optional[str]):
            """
            Runs impact inference + flux-fix and writes a sidecar JSON next to the CSV.
            This keeps compatibility with any existing CSV consumers.
            """
            if not self._ai_ready(need_vision=False):
                return
            motion = self._scene_motion_summary(sc, max_points=620)
            res_imp = self.ai.infer_impacts(motion_summary=motion)
            if res_imp:
                sc.ai_impacts = list(res_imp.get("events") or [])
            res_fix = None
            if getattr(sc, "ai_impacts", None):
                res_fix = self.ai.suggest_flux_fixes(motion_summary=motion, impacts=list(sc.ai_impacts))
                if res_fix:
                    sc.ai_flux_fix = dict(res_fix)

            side = None
            if isinstance(csv_path, str) and csv_path.lower().endswith(".csv"):
                side = csv_path[:-4] + "_ai.json"
            elif isinstance(csv_path, str):
                side = csv_path + "_ai.json"

            payload = {
                "scene_id": int(si),
                "video": os.path.basename(video_path),
                "ai_cfg": {
                    "model_nano": self.ai_cfg.model_nano,
                    "model_mini": self.ai_cfg.model_mini,
                    "model_heavy": self.ai_cfg.model_heavy,
                    "vision": self.ai_cfg.vision,
                    "explicit": bool(self.ai_cfg.explicit),
                },
                "impacts": list(getattr(sc, "ai_impacts", []) or []),
                "flux_fix": dict(getattr(sc, "ai_flux_fix", {}) or {}),
            }

            if side:
                try:
                    with open(side, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    self._ai_toast(f"AI sidecar written: {os.path.basename(side)}", ttl=4.0)
                except Exception as e:
                    self._ai_toast(f"AI sidecar write failed: {e}", ttl=4.0)

        def ai_solve_occlusion(self):
            # Requires vision
            if not self._ai_ready(need_vision=True):
                return
            sc = self._get_active_scene_obj()
            if sc is None or not sc.rois:
                self._ai_toast("No active scene/ROIs", ttl=2.5)
                return
            if self.active_roi < 0 or self.active_roi >= len(sc.rois):
                self._ai_toast("No active ROI selected", ttl=2.5)
                return
            if self._cached_frame is None or self._prev_cached_frame is None:
                self._ai_toast("Need at least 2 frames cached", ttl=2.5)
                return

            roi = sc.rois[self.active_roi]
            prior = [int(x) for x in roi.rect]
            vmode = str(self.ai_cfg.vision).lower()
            if vmode == "full":
                prev_crop = self._prev_cached_frame
                curr_crop = self._cached_frame
                ctx_rect = (0, 0, int(prev_crop.shape[1]), int(prev_crop.shape[0]))
            else:
                # Use a padded context crop to limit what we send.
                prev_crop, ctx_rect = self._roi_crop_safe(self._prev_cached_frame, roi.rect, pad_frac=0.60)
                curr_crop, _       = self._roi_crop_safe(self._cached_frame,      roi.rect, pad_frac=0.60)
            ctx_x, ctx_y, ctx_w, ctx_h = ctx_rect

            img_prev = _bgr_to_data_url(prev_crop, max_side=512, mode=("edges" if vmode=="edges" else "crop"))
            img_curr = _bgr_to_data_url(curr_crop, max_side=512, mode=("edges" if vmode=="edges" else "crop"))

            ctx = f"context_crop_origin_xy=({ctx_x},{ctx_y}), size_wh=({ctx_w},{ctx_h})"
            res = self.ai.solve_occlusion(
                roi_index=int(self.active_roi),
                prior_bbox_xywh=prior,
                context_text=ctx,
                images=[img_prev, img_curr]
            )
            if not res:
                self._ai_toast(f"AI occlusion failed: {self.ai.last_error}", ttl=4.0)
                return

            roi.ai_last_occlusion_status = str(res.get("status", "") or "")
            bbox = res.get("bbox_xywh_in_crop") or res.get("bbox_xywh")  # backward compat
            cf = float(res.get("confidence", 0.0) or 0.0)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                bx, by, bw, bh = [int(v) for v in bbox]
                # map crop->full
                bx += int(ctx_x); by += int(ctx_y)
                bx, by, bw, bh = clamp_rect(bx, by, bw, bh, self._cached_frame.shape[1], self._cached_frame.shape[0])
                roi.ai_last_bbox_xywh = (bx, by, bw, bh)

                if self.ai_cfg.auto_fix_occlusion and cf >= 0.55:
                    roi.rect = (bx, by, bw, bh)
                    self._ai_toast([f"Occlusion: moved ROI to {roi.rect} (c={cf:.2f})"], ttl=4.5)
                else:
                    self._ai_toast([f"Occlusion: suggested bbox {bx,by,bw,bh} (c={cf:.2f})"], ttl=4.5)
            else:
                self._ai_toast([f"Occlusion: no bbox returned (status={roi.ai_last_occlusion_status})"], ttl=4.5)

        def ai_outline_active_roi(self):
            # Requires vision
            if not self._ai_ready(need_vision=True):
                return
            sc = self._get_active_scene_obj()
            if sc is None or not sc.rois:
                self._ai_toast("No active scene/ROIs", ttl=2.5)
                return
            if self.active_roi < 0 or self.active_roi >= len(sc.rois):
                self._ai_toast("No active ROI selected", ttl=2.5)
                return
            if self._cached_frame is None:
                self._ai_toast("No frame cached yet", ttl=2.5)
                return

            roi = sc.rois[self.active_roi]
            crop, _ = self._roi_crop_safe(self._cached_frame, roi.rect, pad_frac=0.05)
            vmode = str(self.ai_cfg.vision).lower()
            img = _bgr_to_data_url(crop, max_side=512, mode=("edges" if vmode=="edges" else "crop"))
            res = self.ai.outline_object(
                roi_index=int(self.active_roi),
                context_text="ROI crop (normalized polygon expected).",
                image=img
            )
            if not res:
                self._ai_toast(f"AI outline failed: {self.ai.last_error}", ttl=4.0)
                return

            polys = res.get("polygons_norm") or []
            cf = float(res.get("confidence", 0.0) or 0.0)
            if isinstance(polys, list):
                roi.ai_outline_polys_norm = polys
                roi.ai_outline_conf = cf
                roi.ai_outline_enabled = (cf >= 0.55 and len(polys) > 0)
                roi.ai_outline_source = "ai"
                self._ai_toast([f"AI outline: polys={len(polys)} enabled={roi.ai_outline_enabled} (c={cf:.2f})"], ttl=5.0)
            else:
                self._ai_toast("AI outline: invalid polygons", ttl=4.0)

        # ------------------------- end AI Assist methods -------------------------


        # --------------------------- Mask/Outline helpers ---------------------------

        def clear_active_roi_outline_mask(self):
            sc = self._get_active_scene_obj()
            if sc is None or not sc.rois:
                return
            if self.active_roi < 0 or self.active_roi >= len(sc.rois):
                return
            roi = sc.rois[self.active_roi]
            roi.ai_outline_enabled = False
            roi.ai_outline_conf = 0.0
            roi.ai_outline_polys_norm = []
            roi.ai_outline_source = "manual"
            self._ai_toast(["Mask cleared"], ttl=2.0)

        def toggle_active_roi_outline_mode(self):
            sc = self._get_active_scene_obj()
            if sc is None or not sc.rois:
                return
            if self.active_roi < 0 or self.active_roi >= len(sc.rois):
                return
            roi = sc.rois[self.active_roi]
            if not getattr(roi, "ai_outline_polys_norm", None):
                self._ai_toast(["No mask polygons on active ROI"], ttl=2.5)
                return
            cur = str(getattr(roi, "ai_outline_mode", "include") or "include").lower()
            nxt = "exclude" if cur.startswith("inc") else "include"
            roi.ai_outline_mode = nxt
            # keep enabled if polys exist
            roi.ai_outline_enabled = True
            if float(getattr(roi, "ai_outline_conf", 0.0) or 0.0) <= 0.0:
                roi.ai_outline_conf = 1.0
            self._ai_toast([f"Mask mode: {nxt.upper()}"], ttl=2.5)

        def iglog_mask_active_roi(self, default_mode: str = "exclude"):
            """Generate an outline polygon using IG-LoG zero-crossings (no OpenAI).

            Intended use: quickly create an occlusion mask over high-salience clutter.
            """
            sc = self._get_active_scene_obj()
            if sc is None or not sc.rois:
                self._ai_toast("No active scene/ROIs", ttl=2.5)
                return
            if self.active_roi < 0 or self.active_roi >= len(sc.rois):
                self._ai_toast("No active ROI selected", ttl=2.5)
                return
            if self._cached_frame is None:
                self._ai_toast("No frame cached yet", ttl=2.5)
                return

            roi = sc.rois[self.active_roi]
            crop, _ = self._roi_crop_safe(self._cached_frame, roi.rect, pad_frac=0.05)
            if crop is None or crop.size == 0:
                self._ai_toast("IG-LoG mask: empty crop", ttl=2.5)
                return
            try:
                gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
                sigma = float(getattr(self, "iglog_mask_sigma", 4.0) or 4.0)
                sigma = float(max(0.7, min(8.0, sigma)))
                thr = float(getattr(self, "iglog_mask_zc_thr", 0.004) or 0.004)
                thr = float(max(0.0005, min(0.05, thr)))
                ig = iglog_map(gray, sigma=sigma)
                zc = zero_crossings(ig, thresh=thr)
                poly = outline_from_zc(zc)

                # Fallback: component-based autogen when ZC contour fails
                if poly is None or int(getattr(poly, "shape", [0])[0]) < 3:
                    # Use your HG/IG-LoG structure map method (top-percent threshold + CCs)
                    st = roi.__dict__.setdefault("_iglog_mask_state", {})
                    polys_norm = _autogen_iglog_occlusion_polys(
                        gray_u8=gray.astype(np.uint8),
                        struct_state=st,
                        top_percent=float(getattr(self, "iglog_mask_top_percent", 6.0) or 6.0),
                        min_area_frac=float(getattr(self, "iglog_mask_min_area_frac", 0.01) or 0.01),
                        prefer_border=bool(getattr(self, "iglog_mask_prefer_border", True)),
                        simplify_px=float(getattr(self, "iglog_mask_simplify_px", 2.0) or 2.0),
                    )
                    if not polys_norm:
                        self._ai_toast(["IG-LoG mask: no contour found (try larger sigma / lower thr)"], ttl=3.0)
                        return

                    # Commit polys directly
                    roi.ai_outline_enabled = True
                    roi.ai_outline_mode = default_mode
                    roi.ai_outline_polys_norm = polys_norm
                    roi.ai_outline_source = "iglog_autogen"
                    roi.ai_outline_conf = 1.0
                    roi.ai_outline_track = True
                    self._ai_toast([f"IG-LoG mask: autogen {len(polys_norm)} poly"], ttl=2.5)
                    return


                h, w = gray.shape[:2]
                denx = float(max(1, w - 1))
                deny = float(max(1, h - 1))
                poly_norm: List[List[float]] = []
                for (px, py) in poly.reshape(-1, 2):
                    u = float(np.clip(float(px) / denx, 0.0, 1.0))
                    v = float(np.clip(float(py) / deny, 0.0, 1.0))
                    poly_norm.append([u, v])
                if len(poly_norm) < 3:
                    self._ai_toast(["IG-LoG mask: contour too small"], ttl=3.0)
                    return

                roi.ai_outline_polys_norm = [poly_norm]
                roi.ai_outline_conf = 1.0
                roi.ai_outline_enabled = True
                roi.ai_outline_mode = str(default_mode or "exclude")
                roi.ai_outline_source = "iglog"
                self._ai_toast([
                    f"IG-LoG mask: pts={len(poly_norm)} mode={roi.ai_outline_mode.upper()}",
                    f"sigma={sigma:.2f} zc_thr={thr:.4f} dil={int(getattr(roi,'ai_outline_dilate_px',2))}"
                ], ttl=4.0)
            except Exception as e:
                self._ai_toast([f"IG-LoG mask failed: {e}"], ttl=4.0)

        # ------------------------- end Mask/Outline helpers -------------------------


        def jump_scene(self, dir_sign):
            if not self.scenes: return
            idx = 0
            for i, sc in enumerate(self.scenes):
                if sc.in_range(self.frame_idx, self.N): idx=i; break
                if self.frame_idx < sc.start: idx=i; break
                idx=i
            tgt = int(np.clip(idx + (1 if dir_sign>0 else -1), 0, len(self.scenes)-1))
            self.frame_idx = self.scenes[tgt].start; self.active_scene=tgt; self.playing=False

        def _get_scene_thumb(self, si):
            sc = self.scenes[si]; key = id(sc)
            t = self.scene_thumbs.get(key)
            if t is not None:
                return t

            # Pick a stable mid-scene frame; fall back to start.
            idx = sc.start
            if sc.end is not None and sc.end > sc.start:
                idx = sc.start + (sc.end - sc.start)//2

            fr_pv = self._read_preview_at(int(idx))  # uses thumb_cap (lightweight)
            if fr_pv is None:
                self.scene_thumbs[key] = None
                return None

            # Smaller width = lower CPU/GPU; cached once per scene.
            pvW = min(160, max(1, self.W // 5))
            pvH = max(1, int(pvW * self.H / max(1, self.W)))
            t = cv.resize(fr_pv, (pvW, pvH), interpolation=cv.INTER_AREA)
            self.scene_thumbs[key] = t
            return t


        def find_roi_at(self, si, x, y):
            if si<0 or si>=len(self.scenes): return -1
            for idx in range(len(self.scenes[si].rois)-1, -1, -1):
                rx,ry,rw,rh = self.scenes[si].rois[idx].rect
                if (x>=rx and x<=rx+rw and y>=ry and y<=ry+rh): return idx
            return -1


        # ---------- frame + HUD ----------
        def draw_scene_panel(self, img):
            def _blit_scene_thumb(img, t, tile_y0, PANEL_ITEM_H, PANEL_W):
                if t is None: return
                H, W = img.shape[:2]
                th, tw = t.shape[:2]
                dst_y0, dst_x0 = tile_y0 + 4, 6
                h_space = max(0, H - dst_y0); w_space = max(0, W - dst_x0)
                h_avail = max(0, min(th, PANEL_ITEM_H - 24, h_space))
                w_avail = max(0, min(tw, PANEL_W - 12, w_space))
                if h_avail > 0 and w_avail > 0:
                    img[dst_y0:dst_y0 + h_avail, dst_x0:dst_x0 + w_avail] = t[:h_avail, :w_avail]

            H,W = img.shape[:2]
            show = self.panel_visible or (time.time() < self.panel_flash_until)
            self.panel_hitboxes.clear()
            if not show or not self.scenes: return
            x0,y0=0,0
            overlay = img.copy()
            cv.rectangle(overlay, (x0,y0), (x0+self.PANEL_W, H), (30,30,30), -1)
            cv.addWeighted(overlay, 0.35, img, 0.65, 0, img)
            y = 8
            max_rows = max(1, (H - y) // self.PANEL_ITEM_H)
            start = int(np.clip(self.scene_scroll, 0, max(0, len(self.scenes) - max_rows)))
            draw_text_clamped(img, "Scenes (Shift+Wheel)", 8, y+14, (220,220,220), 0.55); y += 26
            for si in range(start, len(self.scenes)):
                sc = self.scenes[si]
                tile_y0 = y
                tile_y1 = y + self.PANEL_ITEM_H - 6
                if tile_y0 >= H:    # nothing below is visible: bail early
                    break
                is_active = (si==self.active_scene)
                cv.rectangle(img, (4, tile_y0), (self.PANEL_W-4, tile_y1),
                             (0,0,255) if is_active else (200,200,200), 2, cv.LINE_AA)
                t = self._get_scene_thumb(si)
                if t is not None:
                    th, tw = t.shape[:2]
                    _blit_scene_thumb(img, t, tile_y0, self.PANEL_ITEM_H, self.PANEL_W)
                rng = f"{sc.start}" + (f"-{sc.end}" if sc.end is not None else "-?")
                draw_text_clamped(img, f"S{si}  f{rng}", 8, tile_y1-6, (240,240,200), 0.5)
                self.panel_hitboxes.append((si, (0, tile_y0, self.PANEL_W, tile_y1)))
                y += self.PANEL_ITEM_H

        def build_hud(self, frame):
            hud = frame.copy()
            base_scale = float(np.clip(0.45 + 0.35*min(self.W,self.H)/720.0, 0.45, 0.9))
            y = draw_text_wrap(hud, f"{os.path.basename(video_path)}  {self.W}x{self.H}@{self.fps:.2f}   f {self.frame_idx}/{self.N-1}   {'PLAY' if self.playing else 'PAUSE'}",
                               10, 24, self.W-20, color=(240,240,200), scale=base_scale)
            # Help is available via F1 (modal dialog); keep the HUD minimal.
            # --- AI overlay (ephemeral) ---
            if self.ai_hud_lines and time.time() < float(self.ai_hud_until or 0.0):
                for line in self.ai_hud_lines:
                    y = draw_text_wrap(hud, line, 10, y+4, self.W-20, color=(200,240,200), scale=base_scale*0.95)
            # ------------------------------
            mode_txt = "ADD ROI: drag box" if self.adding else ("RE-PICK ROI: drag box" if self.repicking else "")
            # rename UI now lives in ROI panel; only draw mode text here
            if mode_txt and self.naming_roi < 0:
                draw_text(hud, mode_txt, 68, (255,230,180), base_scale)

            scrubbing_now = (time.time() < self.fast_scrub_until)
            if not scrubbing_now and self.active_scene >= 0:
                sc = self.scenes[self.active_scene]
                for ri, r in enumerate(sc.rois):
                    self.tracker.draw_roi(hud, r, active=(ri==self.active_roi))

                    draw_arrow = self.tracker.show_arrows or getattr(r, "debug", False)
                    if draw_arrow:
                        self.tracker.draw_arrow3(hud, r)
                    if getattr(r, "debug", False):
                        self.tracker.draw_arrow_debug(hud, r)
                        self.draw_roi_metrics(hud, self.active_scene, ri)
                        # --- NEW: visualize anchor patch + IG-LoG when debug is ON ---
                        self.tracker.draw_anchor_debug(hud, r)
                        self.tracker.draw_cmat_debug(hud, r)
                    # after draw_arrow3 / draw_arrow_debug
  
            # --- DEBUG: paint ROI's OF magnitude as per-pixel alpha heatmap ---     
            if getattr(self.tracker, "debug_flow_mode", 0) != 0:
                dbg = getattr(self.tracker, "debug_roi", None)
                if dbg is not None:
                    dx, dy, patch, alpha_mask = dbg
                    ph, pw = patch.shape[:2]

                    x0 = max(0, int(dx)); y0 = max(0, int(dy))
                    x1 = min(self.W, x0 + pw); y1 = min(self.H, y0 + ph)
                    if x1 > x0 and y1 > y0:
                        px0 = x0 - int(dx); py0 = y0 - int(dy)
                        px1 = px0 + (x1 - x0); py1 = py0 + (y1 - y0)

                        roi_hud   = hud[y0:y1, x0:x1].astype(np.float32)
                        roi_patch = patch[py0:py1, px0:px1].astype(np.float32)
                        roi_alpha = alpha_mask[py0:py1, px0:px1].astype(np.float32)

                        # expand to H×W×1 for broadcasting
                        roi_alpha = roi_alpha[..., None]

                        # max opacity for strongest motion
                        alpha_max = 0.7
                        gamma = 1.5  # >1 compresses lows, boosts highs
                        roi_alpha = roi_alpha ** gamma
                        a = alpha_max * roi_alpha  # 0..alpha_max

                        blended = roi_hud * (1.0 - a) + roi_patch * a
                        hud[y0:y1, x0:x1] = blended.astype(np.uint8)


            if self.roi_tmp:   draw_dashed(hud, self.roi_tmp, (255,200,0))
            if self.bound_tmp: draw_dashed(hud, self.bound_tmp, (0,0,255))
            if self.anchor_tmp:  draw_dashed(hud, self.anchor_tmp,(0,255,0))

            self.draw_scene_panel(hud)
            # --- global camera motion arrow ---
            gx_list, gy_list = [], []
            if self.active_scene >= 0:
                for r in self.scenes[self.active_scene].rois:
                    if str(getattr(r, "cmat_mode", "off")).lower() != "off":
                        gx_list.append(float(getattr(r, "_cmat_gx", 0.0)))
                        gy_list.append(float(getattr(r, "_cmat_gy", 0.0)))

            if gx_list:
                gx = float(np.median(gx_list))
                gy = float(np.median(gy_list))
                self.tracker.draw_global_arrow(hud, gx, gy)

            # AFTER drawing HUD + scene panel
            self.draw_timeline(hud)

            # --- draw live structural preview inset ---
            strip = getattr(self, "struct_preview_strip", None)
            if strip is not None:
                ph, pw = strip.shape[:2]
                # bottom-left, above timeline
                x0 = 8
                y0 = max(8, self.H - self.TIMELINE_H - ph - 10)
                if y0 + ph <= self.H and x0 + pw <= self.W:
                    hud[y0:y0+ph, x0:x0+pw] = strip
                    txt = getattr(self, "struct_preview_text", "")
                    if txt:
                        draw_text_clamped(hud, txt, x0+4, y0-6, (240,240,200), 0.45)
            # --- end preview inset ---

            self.draw_roi_debug_panel(hud)
            self.draw_label_editor(hud)
            self._draw_anchor_panel(hud)

            if getattr(self, "postview_active", False):
                self._draw_postview(hud)

            return hud

        # ---------- main tick ----------
        def tick(self):
            self.active_scene = self.scene_at(self.frame_idx)
            if self.active_scene < 0:
                self.adding=False; self.repicking=False; self.naming_roi=-1

            # decide if we actually need to decode a new frame
            need_new = (
                self._cached_frame is None or
                self.playing or
                self._seeking or
                self._cached_idx != self.frame_idx
            )

            if need_new:
                ok, frame = self._read_frame(self.frame_idx)
                if not ok:
                    # EOF/decoder hiccup: hold last frame cleanly
                    self.playing = False
                    self.frame_idx = min(self.N-1, max(0, self.frame_idx))
                    ok2, frame = self._read_frame(self.frame_idx)
                    if not ok2:
                        frame = np.zeros((self.H, self.W, 3), np.uint8)

                if PROC_SCALE != 1.0:
                    frame = cv.resize(frame, (self.W, self.H), interpolation=cv.INTER_AREA)
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # update cache and clear one-shot seek
                self._prev_cached_frame = self._cached_frame
                self._cached_frame = frame
                self._cached_gray  = gray
                self._cached_idx   = self.frame_idx
                self._seeking      = False
            else:
                frame = self._cached_frame
                gray  = self._cached_gray

            # --- Live structural preview update (ROI-only) ---
            if getattr(self, "struct_preview", False) and self.active_scene >= 0:
                try:
                    sc = self.scenes[self.active_scene]
                    if sc.rois and self._cached_frame is not None:
                        ri = max(0, min(self.active_roi, len(sc.rois)-1))
                        r = sc.rois[ri]
                        x, y, w, h = map(int, r.rect)
                        Hf, Wf = self._cached_frame.shape[:2]
                        x = max(0, min(x, Wf-1)); y = max(0, min(y, Hf-1))
                        w = max(2, min(w, Wf-x)); h = max(2, min(h, Hf-y))
                        crop = self._cached_frame[y:y+h, x:x+w].copy()
                        if crop.size:
                            maps = iglog_struct_maps_u8(crop, sigma=1.2, zc_thresh=0.004, ext_abs_thresh=0.03)
                            # quick outline (optional, cheap): use your existing outline_from_zc on coarse-ish ZC
                            poly = outline_from_zc(maps["zc"])
                            hyb = maps.get("hybrid", None)
                            if hyb is None:
                                hyb = maps["rescue"]
                            strip = make_preview_strip_bgr(crop, maps["clean"], hyb, poly)

                            # downscale strip so it doesn't eat HUD
                            s = float(getattr(self, "struct_preview_scale", 0.33))
                            if s > 0 and s < 1.0:
                                strip = cv.resize(strip, (max(1, int(strip.shape[1]*s)), max(1, int(strip.shape[0]*s))), interpolation=cv.INTER_AREA)
    
                            self.struct_preview_strip = strip
                            self.struct_preview_text = f"GSCM score={maps['score']:.4f}  (zc+ext density)"
                except Exception:
                    pass
            # --- end preview update ---


            changed = (self.frame_idx != self.last_sampled)
            if self.prev_gray is not None and self.active_scene >= 0 and self.playing and changed:
                self.tracker.prepare_scaled_frames(self.prev_gray, gray)
                # OPTIONAL: prefetch global camera motion once per frame
                try:
                    sc = self.scenes[self.active_scene]
                    self.tracker._cmat_scene_rois = list(sc.rois)
                    need_global = any(
                        str(getattr(r, "cmat_mode", "off")).lower() == "global"
                        for r in sc.rois
                    )
                    if need_global and hasattr(self.tracker, "prefetch_global_drift"):
                        self.tracker.prefetch_global_drift(self.prev_gray, gray)
                except Exception:
                    pass
                for ri, r in enumerate(self.scenes[self.active_scene].rois):
                    self.scenes[self.active_scene].rois[ri] = self.tracker.update_roi(r, self.prev_gray, gray)
                sc = self.scenes[self.active_scene]; t = self.frame_idx / self.fps
                if sc.times and t <= sc.times[-1] + (0.25 / self.fps):
                    self._truncate_scene_to_time(sc, t)

                sc.times.append(t); self.last_sampled = self.frame_idx
                for ri, r in enumerate(sc.rois):
                    cx, cy = sc.rois[ri].last_center; vx, vy, vz = sc.rois[ri].vx_ps, sc.rois[ri].vy_ps, sc.rois[ri].vz_rel_s
                    sc.roi_cx.setdefault(ri, []).append(float(cx))
                    sc.roi_cy.setdefault(ri, []).append(float(cy))
                    sc.roi_vx.setdefault(ri, []).append(float(vx))
                    sc.roi_vy.setdefault(ri, []).append(float(vy))
                    sc.roi_vz.setdefault(ri, []).append(float(vz))
                    env = math.sqrt(vx*vx + vy*vy + vz*vz)
                    dt = 1.0/max(1e-6, float(self.fps))
                    vx_arr = sc.roi_vx[ri]; vy_arr = sc.roi_vy[ri]; vz_arr = sc.roi_vz[ri]
                    Wn = max(9, int(self.fps*1.0))  # ~1 s window
                    vxw = np.asarray(vx_arr[-Wn:], float); vyw = np.asarray(vy_arr[-Wn:], float); vzw = np.asarray(vz_arr[-Wn:], float)

                    # robust smooth on the window
                    vx_s = robust_smooth(vxw, self.fps); vy_s = robust_smooth(vyw, self.fps); vz_s = robust_smooth(vzw, self.fps, win_ms=160, ema_tc_ms=240)
                    ax_s = _deriv_central(vx_s, dt); ay_s = _deriv_central(vy_s, dt); az_s = _deriv_central(vz_s, dt)
                    jx_s = _deriv_central(ax_s, dt); jy_s = _deriv_central(ay_s, dt); jz_s = _deriv_central(az_s, dt)

                    # live numbers (magnitudes at tail)
                    r.last_speed = float(np.sqrt(vx_s[-1]**2 + vy_s[-1]**2 + vz_s[-1]**2))
                    r.last_acc   = float(np.sqrt(ax_s[-1]**2 + ay_s[-1]**2 + az_s[-1]**2))
                    r.last_jerk  = float(np.sqrt(jx_s[-1]**2 + jy_s[-1]**2 + jz_s[-1]**2))

                    if len(vxw) < Wn or len(vyw) < Wn or len(vzw) < max(5, Wn//2):
                        # not enough samples → skip live impact this frame
                        pass
                    else:
                        # existing impact_score_cycles(...) path
                        ...


                    # hysteresis impacts: pick near tail (not just last 2 samples)
                    S_any, in_idx, out_idx = _impacts_for_mode(r, vx_s, vy_s, vz_s, self.fps)
                    tail_allow = max(
                        2,
                        int(round(0.08 * self.fps)) + int(round(getattr(r, "impact_lead_ms", 40) * self.fps / 1000.0))
                    )
                    fired_dir = 0
                    if out_idx.size and out_idx[-1] >= len(S_any) - tail_allow:
                        fired_dir = +1
                    elif in_idx.size and in_idx[-1] >= len(S_any) - tail_allow:
                        fired_dir = -1
                    if fired_dir:
                        r._impact_dir = fired_dir
                        _impact_trigger(r, r._impact_dir, now=time.time())
                        r._last_impact_idx = self.frame_idx
                    # record a lane value EVERY frame so lengths match sc.times
                    sc.roi_imp_in.setdefault(ri, [])
                    sc.roi_imp_out.setdefault(ri, [])
                    sc.roi_imp_in[ri].append(1.0 if fired_dir == -1 else 0.0)   # IN
                    sc.roi_imp_out[ri].append(1.0 if fired_dir == +1 else 0.0)  # OUT

                    # commit updated ROI
                    self.scenes[self.active_scene].rois[ri] = r

                    sc.roi_env.setdefault(ri, []).append(float(env))


            # --- In-place structural view (replaces ROI pixels) ---
            # --- In-place structural view (replaces ROI pixels) ---
            if getattr(self, "struct_view", False) and self.active_scene >= 0:
                sc = self.scenes[self.active_scene]
                if sc.rois:
                    ri = max(0, min(self.active_roi, len(sc.rois)-1))
                    r = sc.rois[ri]

                    # Only recompute when frame changes (play/step/scrub). Prevents "static snow" while paused.
                    last_idx = int(getattr(r, "_struct_last_idx", -999999))
                    if last_idx != int(self.frame_idx):
                        r._struct_last_idx = int(self.frame_idx)

                        x, y, w, h = map(int, r.rect)
                        Hf, Wf = frame.shape[:2]
                        x = max(0, min(x, Wf-1)); y = max(0, min(y, Hf-1))
                        w = max(2, min(w, Wf-x)); h = max(2, min(h, Hf-y))

                        crop = frame[y:y+h, x:x+w].copy()
                        if crop.size:
                            maps = iglog_struct_maps_u8(crop, sigma=1.2, zc_thresh=0.004, ext_abs_thresh=0.03)
                            M_u8 = maps.get("hybrid", None)
                            if M_u8 is None:
                                st = r.__dict__.setdefault("_iglog_state", {})
                                M_u8, _, _ = adaptive_struct_map(maps["abs"], maps["zc"], maps["ext"], st)
                            r._struct_last_u8 = M_u8


                    # Use cached result (works even when paused)
                    M_u8 = getattr(r, "_struct_last_u8", None)
                    if M_u8 is not None:
                        M_bgr = cv.cvtColor(M_u8, cv.COLOR_GRAY2BGR)
                        x, y, w, h = map(int, r.rect)
                        Hf, Wf = frame.shape[:2]
                        x = max(0, min(x, Wf-1)); y = max(0, min(y, Hf-1))
                        w = max(2, min(w, Wf-x)); h = max(2, min(h, Hf-y))
                        if M_bgr.shape[0] == h and M_bgr.shape[1] == w:
                            frame[y:y+h, x:x+w] = M_bgr
            # --- end structural view ---


            self.hud = self.build_hud(frame)


            if self.recording:
                if self.ff is None and self.ocv_writer is None:
                    stem = os.path.splitext(os.path.basename(video_path))[0]
                    out_path = f"{stem}_overlay_record.mp4"
                    self.ff = _open_ffmpeg_writer(out_path, self.W, self.H, self.fps)
                    if self.ff is None:
                        fourcc = cv.VideoWriter_fourcc(*'mp4v')
                        self.ocv_writer = cv.VideoWriter(out_path, fourcc, self.fps, (self.W, self.H))
                        print("[record] OpenCV writer →", out_path)
                    else:
                        print("[record] ffmpeg →", out_path)
                if self.ff is not None:
                    _ff_write(self.ff, self.hud)
                elif self.ocv_writer is not None:
                    self.ocv_writer.write(self.hud)

            prev_idx = self._last_frame_idx
            if self.active_scene >= 0:
                sc = self.scenes[self.active_scene]
                if sc.end is not None:
                    crossed = (prev_idx < sc.end <= self.frame_idx)
                    if crossed and getattr(sc, "_export_pending", False) and sc.times:
                        self.export_scene(self.active_scene)
                        sc._export_pending = False
            # update last index AFTER checks
            self._last_frame_idx = self.frame_idx


            if self.playing:
                self.frame_idx = min(self.N-1, self.frame_idx+1)
                self.prev_gray = gray   # advance baseline only during play


            self.prev_gray = gray
            self.view.update()  # trigger paint

        # ---------- input mapping ----------
        def _to_content(self, x, y):
            ox, oy = self.view_offset; s = self.view_scale if self.view_scale>0 else 1.0
            cx = int(np.clip(round((x-ox)/s), 0, self.W-1)); cy = int(np.clip(round((y-oy)/s), 0, self.H-1))
            return cx, cy
        
        def on_middle_down(self, x, y, mods):
            self.mmb_drag = True
            self.mmb_anchor = self._to_content(x, y)
        def on_middle_drag(self, x, y, mods):
            pass
            # if not getattr(self, "mmb_drag", False): return
            # cx, cy = self._to_content(x, y)
            # si = self.active_scene
            # if si < 0 or not self.scenes[si].rois: return
            # ri = self.find_roi_at(si, cx, cy)
            # if ri < 0: ri = self.active_roi
            # r = self.scenes[si].rois[ri]
            # # yaw from drag vector
            # dx = cx - (r.rect[0] + r.rect[2]//2); dy = cy - (r.rect[1] + r.rect[3]//2)
            # yaw = (math.degrees(math.atan2(dy, dx)) % 360.0)
            # r.io_dir_deg = yaw; r.dir_gate_deg = yaw; r.dir_io_deg = yaw if math.isnan(getattr(r,'dir_io_deg',float('nan'))) else r.dir_io_deg
            # # pitch when Shift held: up = +pitch
            # if mods & QtCore.Qt.ShiftModifier:
            #     r.axis_elev_deg = float(np.clip(getattr(r, "axis_elev_deg", 0.0) - 0.25*dy, -89.0, 89.0))
            # self.scenes[si].rois[ri] = r
        def on_middle_up(self): self.mmb_drag = False


        def on_left_down(self, x, y, shift, alt, ctrl):
            cx, cy = self._to_content(x,y)

            # check click on ROI panel rows
            for ri, (x0, y0, x1, y1) in self.roi_panel_hitboxes:
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    self.active_roi = ri
                    # optional: toggle debug on click
                    # r = self.scenes[self.active_scene].rois[ri]
                    # r.debug = not getattr(r, "debug", False)
                    # self.scenes[self.active_scene].rois[ri] = r
                    return
            si = self.active_scene
            # Alt+Click anywhere on the 8-way widget OR the small gate/IO compass → cycle I/O
            if si >= 0:
                for ri, r in enumerate(self.scenes[si].rois):
                    for hb in (getattr(r, "dir8_hit", None), getattr(r, "dir_hit", None)):
                        if alt and hb:
                            x0,y0,w,h = hb
                            if x0 <= cx <= x0+w and y0 <= cy <= y0+h:
                                r.impact_io = {0: +1, +1: -1, -1: 0}[getattr(r, 'impact_io', 0)]
                                self.scenes[si].rois[ri] = r
                                return
            if si >= 0:
                cx, cy = self._to_content(x,y)
                for ri, r in enumerate(self.scenes[si].rois):
                    # I/O label toggle
                    if getattr(r, "io_hit", None):
                        x0,y0,w0,h0 = r.io_hit
                        if x0 <= cx <= x0+w0 and y0 <= cy <= y0+h0:
                            r.impact_io = {0:+1, +1:-1, -1:0}[getattr(r,'impact_io',0)]
                            self.scenes[si].rois[ri] = r
                            return
                    # direction wedge select
                    if getattr(r, "dir8_hit", None):
                        x0,y0,w0,h0 = r.dir8_hit
                        if x0 <= cx <= x0+w0 and y0 <= cy <= y0+h0:
                            # compute wedge by click angle relative to widget center
                            gx = x0 + w0//2; gy = y0 + h0//2 + 3
                            vx = float(cx - gx); vy = float(cy - gy)
                            if abs(vx)+abs(vy) < 2:
                                new_sel = None
                            else:
                                new_sel = _vec_to_dir8(vx, -vy)  # note: screen->math
                            # toggle off if same wedge clicked
                            cur = getattr(r, 'impact_dir8', None)
                            r.impact_dir8 = (None if (cur is not None and new_sel==int(cur)) else int(new_sel) if new_sel is not None else None)
                            self.scenes[si].rois[ri] = r
                            return

            # timeline click → seek (no play)
            if cy >= self.H - self.TIMELINE_H:
                self.dragging_timeline = True
                idx = int(np.clip((cx / max(1, self.W-1)) * (self.N-1), 0, self.N-1))
                self._seek_to_frame(idx)
                return
            # shelf click
            if cx < self.PANEL_W:
                for si,(x0,y0,x1,y1) in self.panel_hitboxes:
                    if x0 <= cx <= x1 and y0 <= cy <= y1:
                        self.frame_idx = self.scenes[si].start; self.active_scene=si; self.playing=False; return
            # start drag
            self.dragging = True
            self.p0 = (cx, cy)
            self.roi_tmp = None
            self.bound_tmp = None
            self.anchor_tmp = None
            self.drag_is_anchor = False
            self.anchor_roi = -1

            # Ctrl+Drag inside an ROI → anchor definition
            if ctrl and not self.adding and not self.repicking and self.active_scene >= 0:
                si = self.active_scene
                ri = self.find_roi_at(si, cx, cy)
                if ri >= 0:
                    self.drag_is_anchor = True
                    self.anchor_roi = ri

            if self.drag_is_anchor:
                # anchor drag has its own rect (anchor_tmp), no bound_tmp
                self.drag_is_bound = False
            else:
                self.drag_is_bound = shift and not self.adding and not self.repicking

            si = self.active_scene
            if si >= 0 and self.scenes[si].rois:
                ri = self.find_roi_at(si, cx, cy)
                if ri >= 0:
                    r = self.scenes[si].rois[ri]
                    if getattr(r, "dir_hit", None):
                        x0,y0,w0,h0 = r.dir_hit
                        if x0 <= cx <= x0+w0 and y0 <= cy <= y0+h0:
                            # compute quantized direction from widget center
                            ccx = x0 + w0//2; ccy = y0 + h0//2
                            deg = _quantize8_deg_from_xy(cx - ccx, cy - ccy)
                            if bool(shift):
                                r.dir_io_deg = deg
                            else:
                                r.dir_gate_deg = deg
                                if math.isnan(getattr(r,'dir_io_deg',float('nan'))):  # default io to gate
                                    r.dir_io_deg = deg
                            self.scenes[si].rois[ri] = r
                            return


        # Controller.on_right_click(self, x, y)  -- replace with:
        def on_right_click(self, x, y):
            cx, cy = self._to_content(x,y)
            si = self.active_scene
            if si < 0: return
            # If right-click is on any ROI's 8-way widget → toggle expansion
            for ri, r in enumerate(self.scenes[si].rois):
                hb = getattr(r, "dir8_hit", None)
                if hb:
                    x0,y0,w,h = hb
                    if x0 <= cx <= x0+w and y0 <= cy <= y0+h:
                        r.dir8_expanded = not getattr(r, "dir8_expanded", False)
                        self.scenes[si].rois[ri] = r
                        return
            # otherwise keep your existing postview behavior
            ri = self.find_roi_at(si, cx, cy)
            if ri < 0: return
            self.postview_active = True
            self.postview_roi = (si, ri)
            self._postview_saved_idx = self.frame_idx
            self.playing = False


        def on_move(self, x, y):
            cx, cy = self._to_content(x,y)
            if self.dragging_timeline:
                idx = int(np.clip((cx / max(1, self.W-1)) * (self.N-1), 0, self.N-1))
                if idx != self.frame_idx:
                    self._seek_to_frame(idx)
                return

            if not self.dragging:
                if cy >= self.H - self.TIMELINE_H:
                    self.timeline_hover_idx = int(np.clip((cx / max(1, self.W-1)) * (self.N-1), 0, self.N-1))
                else:
                    self.timeline_hover_idx = -1
                self.mouse_xy = (cx, cy)
                return
            x0, y0 = self.p0
            rect = (min(x0, cx), min(y0, cy), abs(cx - x0), abs(cy - y0))
            if self.drag_is_anchor:
                self.anchor_tmp = rect
            elif self.drag_is_bound:
                self.bound_tmp = rect
            else:
                self.roi_tmp = rect


        def on_left_up(self, x, y):
            if self.dragging_timeline:
                self.dragging_timeline = False
                return

            if not self.dragging: return
            self.dragging=False
            # Ctrl+Drag anchor placement
            if self.drag_is_anchor and self.anchor_tmp and self.active_scene >= 0 and self.anchor_roi >= 0 and self.anchor_tmp[2] >= 4 and self.anchor_tmp[3] >= 4:
                self._push_undo("Set ROI anchor", coalesce="roi_geom")
                si = self.active_scene
                ri = self.anchor_roi
                r  = self.scenes[si].rois[ri]

                rx, ry, rw, rh = r.rect
                ax, ay, aw, ah = clamp_rect(*self.anchor_tmp, self.W, self.H)

                # intersect anchor rect with ROI so it never spills outside
                ix0 = max(rx, ax)
                iy0 = max(ry, ay)
                ix1 = min(rx + rw, ax + aw)
                iy1 = min(ry + rh, ay + ah)

                if ix1 > ix0 and iy1 > iy0:
                    iw = ix1 - ix0
                    ih = iy1 - iy0
                    cx = ix0 + iw / 2.0
                    cy = iy0 + ih / 2.0

                    # relative center inside ROI
                    u = float((cx - rx) / max(1.0, rw))
                    v = float((cy - ry) / max(1.0, rh))

                    # size as fraction of min(w,h)
                    rad = 0.5 * min(iw, ih)
                    size_frac = float(np.clip(2.0 * rad / max(1.0, min(rw, rh)), 0.05, 1.0))

                    r.anchor_u = u
                    r.anchor_v = v
                    r.anchor_size_frac = size_frac
                    r.anchor_user_set = True

                    # reset descriptor state so IG-LoG builds from this new patch
                    r._anchor_ready = False
                    r._anchor_desc = None
                    r._anchor_lost_count = 0
                    r._anchor_template = None
                    r._anchor_last_ok = False
                    r._anchor_last_sim = 0.0

                    self.scenes[si].rois[ri] = r

                self.anchor_tmp = None
                self.drag_is_anchor = False
                self.anchor_roi = -1
                return
            
            if self.drag_is_bound and self.bound_tmp and self.active_scene>=0 and self.scenes[self.active_scene].rois and self.bound_tmp[2]>=8 and self.bound_tmp[3]>=8:
                self._push_undo("Set ROI bound", coalesce="roi_geom")
                r = self.scenes[self.active_scene].rois[self.active_roi]
                bx,by,bw,bh = clamp_rect(*self.bound_tmp, self.W, self.H)
                self.scenes[self.active_scene].rois[self.active_roi] = roi_replace(r, bound=(bx,by,bw,bh))
                self.bound_tmp=None
            elif (self.adding or self.repicking) and self.roi_tmp and self.active_scene>=0 and self.roi_tmp[2]>=8 and self.roi_tmp[3]>=8:
                nx,ny,nw,nh = clamp_rect(*self.roi_tmp, self.W, self.H); cx,cy= nx+nw/2.0, ny+nh/2.0
                if self.repicking and self.scenes[self.active_scene].rois:
                    self._push_undo("Repick ROI", coalesce="roi_geom")
                    r = self.scenes[self.active_scene].rois[self.active_roi]
                    self.scenes[self.active_scene].rois[self.active_roi] = roi_replace(r, 
                        rect=(nx,ny,nw,nh), last_center=(cx,cy))
                        # NOW reset dynamic scale state so this geometry is baseline
                    rr = self.scenes[self.active_scene].rois[self.active_roi]
                    reset_roi_dynamic_state(rr, reset_cmat=True, reset_z=True)
                elif self.adding:
                    self._push_undo("Add ROI", coalesce="roi_add")
                    r = ROI(rect=(nx,ny,nw,nh), name=None, bound=None, fb_levels=5, last_center=(cx,cy), motion_mode=("hg" if bool(getattr(self.tracker, "force_hg_mode", False)) else "fb"))
                    reset_roi_dynamic_state(r, reset_cmat=True, reset_z=True)  # make baseline explicit
                    self.scenes[self.active_scene].rois.append(r); self.active_roi = len(self.scenes[self.active_scene].rois)-1
                self.roi_tmp=None; self.repicking=False; self.adding=False
                self._rebuild_label_roles_for_scene(self.active_scene)

        def on_double(self, x, y):
            cx, cy = self._to_content(x,y)
            ri = self.find_roi_at(self.active_scene, cx, cy)
            if ri >= 0:
                self.naming_roi = ri
                self.name_buf = self.scenes[self.active_scene].rois[ri].name or ""

        def on_wheel(self, x, y, delta, ctrl, shift):
            cx, cy = self._to_content(x,y)

            # add below the scene-scroll branch
            if not ctrl and not shift:
                # plain wheel over right panel scrolls ROI panel
                if self.active_scene >= 0 and cx >= self.W - self.PANEL_W:
                    steps = int(np.sign(delta)) * max(1, abs(delta)//120)
                    self.roi_panel_scroll -= steps
                    return

            if shift and not ctrl:
                 # if we’re hovering left panel → vertical scroll
                if cx < self.PANEL_W:
                    steps = int(np.sign(delta)) * max(1, abs(delta)//120)
                    self.scene_scroll -= steps
                    return
                # else keep existing "jump scene" behavior
                self.jump_scene(1 if delta > 0 else -1)
                self.panel_flash_until = time.time() + 1.25
                self.playing = False
                return
            if ctrl:
                notches = max(1, abs(delta)//120)
                sign = 1 if delta>0 else -1
                self._seek_to_frame(self.frame_idx + sign*notches*self.ctrl_scrub_step); return

            # plain wheel => act only when over ROI label tags
            si = self.active_scene
            if si >= 0 and self.scenes[si].rois:
                steps = int(np.sign(delta)) * max(1, abs(delta)//120)
                for ri, r in enumerate(self.scenes[si].rois):
                    # priority: L# tag
                    if getattr(r, "labelL_hit", None):
                        x0,y0,w,h = r.labelL_hit
                        if x0 <= cx <= x0+w and y0 <= cy <= y0+h:
                            self._push_undo("Adjust ROI level", coalesce="roi_param")
                            r.fb_levels = int(np.clip(getattr(r,'fb_levels',5) + steps, 1, 8))
                            self.scenes[si].rois[ri] = r
                            return
                    # secondary: Ref tag
                    if getattr(r, "labelRef_hit", None):
                        x0,y0,w,h = r.labelRef_hit
                        if x0 <= cx <= x0+w and y0 <= cy <= y0+h:
                            # ... inside the 'Ref tag' branch:
                            self._push_undo("Adjust ROI refractory", coalesce="roi_param")
                            r.refractory_ms = int(np.clip(getattr(r,'refractory_ms',140) + 10*steps, 40, 10000))
                            self.scenes[si].rois[ri] = r
                            return
                        
            si = self.active_scene
            if si >= 0 and self.scenes[si].rois:
                for ri, r in enumerate(self.scenes[si].rois):
                    if getattr(r, "dir_hit", None):
                        x0,y0,w0,h0 = r.dir_hit
                        if x0 <= cx <= x0+w0 and y0 <= cy <= y0+h0:
                            self._push_undo("Rotate ROI direction", coalesce="roi_param")
                            step = (45 if steps>0 else -45)
                            if bool(ctrl): step *= 2  # faster
                            if bool(shift):
                                base = r.dir_io_deg if not math.isnan(getattr(r,'dir_io_deg',float('nan'))) else r.dir_gate_deg
                                if not math.isnan(base): r.dir_io_deg = (base + step) % 360.0
                            else:
                                base = r.dir_gate_deg if not math.isnan(getattr(r,'dir_gate_deg',float('nan'))) else 0.0
                                r.dir_gate_deg = (base + step) % 360.0
                                if math.isnan(getattr(r,'dir_io_deg',float('nan'))):
                                    r.dir_io_deg = r.dir_gate_deg
                            self.scenes[si].rois[ri] = r
                            return

            return  # scroll elsewhere → no action


        def on_key(self, ev):
            k = ev.key(); mod = ev.modifiers()

            if ev.key() == QtCore.Qt.Key_Escape and getattr(self, "postview_active", False):
                self.postview_active = False
                self.postview_roi = (-1, -1)
                self.frame_idx = self._postview_saved_idx  # restore position
                return
            # rename mode text entry (label editor)
            if self.naming_roi >= 0:
                if k == QtCore.Qt.Key_F1:
                    self.show_help_dialog(); return
                # Shift+Enter → insert newline (new alias)
                if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and (mod & QtCore.Qt.ShiftModifier):
                    self.name_buf += "\n"
                    return

                # Enter → commit + exit
                if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                    self._commit_name_buf()
                    self.naming_roi = -1
                    self.name_buf = ""
                    return

                # Up/Down → commit, jump to previous/next ROI label
                if k == QtCore.Qt.Key_Up:
                    self._commit_name_buf()
                    if self.active_scene >= 0 and self.scenes[self.active_scene].rois:
                        n = len(self.scenes[self.active_scene].rois)
                        self.naming_roi = (self.naming_roi - 1) % n
                        self.name_buf = self.scenes[self.active_scene].rois[self.naming_roi].name or ""
                    return

                if k == QtCore.Qt.Key_Down:
                    self._commit_name_buf()
                    if self.active_scene >= 0 and self.scenes[self.active_scene].rois:
                        n = len(self.scenes[self.active_scene].rois)
                        self.naming_roi = (self.naming_roi + 1) % n
                        self.name_buf = self.scenes[self.active_scene].rois[self.naming_roi].name or ""
                    return

                if k == QtCore.Qt.Key_Escape:
                    self.naming_roi = -1
                    self.name_buf = ""
                    return

                if k in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
                    self.name_buf = self.name_buf[:-1]
                    return

                ch = ev.text()
                # allow a wider character set; sanitize later in _split_labels
                if ch and re.match(r"[A-Za-z0-9 _\-\.,:/\(\)\[\]\+\#]", ch) and len(self.name_buf) < 96:
                    self.name_buf += ch
                return
            

            # Undo/Redo (project edits)
            if (mod & QtCore.Qt.ControlModifier) and k == QtCore.Qt.Key_Z:
                if (mod & QtCore.Qt.ShiftModifier):
                    self.redo_action()
                else:
                    self.undo_action()
                return
            if (mod & QtCore.Qt.ControlModifier) and k == QtCore.Qt.Key_Y:
                self.redo_action()
                return

            # --- SHIFT + BACKSPACE: delete active ROI (hard / intentional) ---
            if k == QtCore.Qt.Key_Backspace and (mod & QtCore.Qt.ShiftModifier):  # Backspace + Shift
                self._delete_active_roi()
                print("[ROI] Deleted active ROI")
                return
            # ---------------------------------------------------------------

            # Ctrl+Alt+O = auto-generate occlusion polygon from IG-LoG/HG structure (NO OpenAI)
            if (mod & QtCore.Qt.ControlModifier) and (mod & QtCore.Qt.AltModifier) and k == QtCore.Qt.Key_O:
                si = self.active_scene
                if si >= 0 and self.scenes[si].rois and 0 <= self.active_roi < len(self.scenes[si].rois):
                    r = self.scenes[si].rois[self.active_roi]

                    # Need current frame patch in ROI space.
                    # Assumes you have a method to fetch current grayscale frame as uint8.
                    frame_bgr = self._get_curr_frame_bgr()   # <-- you likely already have something like this
                    frame_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)

                    # ROI rect is (x, y, w, h)
                    x0, y0, w0, h0 = map(int, r.rect)
                    x1, y1 = x0 + w0, y0 + h0

                    Hf, Wf = frame_gray.shape[:2]
                    x0 = max(0, min(x0, Wf-1)); y0 = max(0, min(y0, Hf-1))
                    x1 = max(0, min(x1, Wf  )); y1 = max(0, min(y1, Hf  ))
                    if (x1 - x0) < 8 or (y1 - y0) < 8:
                        print("[mask] ROI too small after clamp"); return
                    if (x1 - x0) >= 8 and (y1 - y0) >= 8:
                        patch = frame_gray[y0:y1, x0:x1].copy()

                        st = r.__dict__.setdefault("_iglog_state_hg", {})
                        polys = _autogen_iglog_occlusion_polys(
                            patch, st,
                            top_percent=6.0,
                            min_area_frac=0.01,
                            prefer_border=True,
                            simplify_px=2.0
                        )

                        if polys:
                            r.ai_outline_polys_norm = polys
                            r.ai_outline_conf = 1.0
                            r.ai_outline_enabled = True
                            r.ai_outline_mode = "exclude"   # default: occluder
                            self.scenes[si].rois[self.active_roi] = r
                            print(f"[mask] ROI{self.active_roi}: IGLoG occlusion mask generated ({len(polys)} poly)")
                        else:
                            print(f"[mask] ROI{self.active_roi}: IGLoG mask generation failed (no component)")
                return


            # Ctrl + J = cycle tracking mode for ACTIVE ROI: FB -> HG -> HYBRID
            if k == QtCore.Qt.Key_J and (mod & QtCore.Qt.ControlModifier):
                if self.active_scene >= 0 and self.scenes[self.active_scene].rois:
                    r = self.scenes[self.active_scene].rois[self.active_roi]
                    modes = ["fb", "hg", "hybrid"]
                    cur = str(getattr(r, "motion_mode", "fb") or "fb").lower()
                    # normalize aliases
                    if cur.startswith("hyb"):
                        cur = "hybrid"
                    elif cur.startswith("hg"):
                        cur = "hg"
                    else:
                        cur = "fb"
                    try:
                        idx = modes.index(cur)
                    except ValueError:
                        idx = 0
                    r.motion_mode = modes[(idx + 1) % len(modes)]
                    self.scenes[self.active_scene].rois[self.active_roi] = r
                    print(f"[track] ROI{self.active_roi}: motion_mode = {r.motion_mode}")
                else:
                    # no ROI selected: toggle a global override (kept as-is)
                    cur = bool(getattr(self.tracker, "force_hg_mode", False))
                    self.tracker.force_hg_mode = (not cur)
                    print(f"[track] GLOBAL override: force_hg_mode = {self.tracker.force_hg_mode}")
                return


            # label mode / primary assignment
            if k == QtCore.Qt.Key_L:
                # Shift+L → mark active ROI as PRIMARY for its labels
                if mod & QtCore.Qt.ShiftModifier:
                    self._promote_active_roi_primary()
                    return
                # plain L → cycle label export mode
                modes = ["per_roi", "per_label"]
                cur = getattr(self, "label_mode", "per_roi")
                try:
                    idx = modes.index(cur)
                except ValueError:
                    idx = 0
                idx = (idx + 1) % len(modes)
                self.label_mode = modes[idx]
                for si, sc in enumerate(self.scenes):
                    sc.label_mode = self.label_mode
                    self.scenes[si] = sc
                print(f"[labels] mode → {self.label_mode}")
                self.panel_flash_until = time.time() + 1.25
                return

            
            # global keys
            # --- AI hotkeys (requires --ai, and sometimes --ai-vision) ---
            if (mod & QtCore.Qt.ControlModifier) and k == QtCore.Qt.Key_T:
                self.ai_suggest_roi_tags(); return
            if (mod & QtCore.Qt.ControlModifier) and k == QtCore.Qt.Key_I:
                self.ai_infer_impacts(); return
            if (mod & QtCore.Qt.ControlModifier) and k == QtCore.Qt.Key_F:
                self.ai_suggest_flux_fixes(); return
            if (mod & QtCore.Qt.ControlModifier) and k == QtCore.Qt.Key_O:
                self.ai_solve_occlusion(); return
            # Mask helpers (no OpenAI required)
            if (mod & QtCore.Qt.ControlModifier) and (mod & QtCore.Qt.ShiftModifier) and k == QtCore.Qt.Key_M:
                self.toggle_active_roi_outline_mode(); return
            if (mod & QtCore.Qt.ControlModifier) and (mod & QtCore.Qt.AltModifier) and k == QtCore.Qt.Key_M:
                self.iglog_mask_active_roi(default_mode="exclude"); return
            if (mod & QtCore.Qt.ControlModifier) and k in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
                self.clear_active_roi_outline_mask(); return

            # Ctrl+M => AI outline for active ROI (normalized polygon mask)
            if (mod & QtCore.Qt.ControlModifier) and k == QtCore.Qt.Key_M:
                self.ai_outline_active_roi(); return
            # --- end AI hotkeys ---
            if k in (QtCore.Qt.Key_X,):
                if self.active_scene >= 0:
                    self.split_scene(self.active_scene, self.frame_idx)
                return

            # CTRL + D = Debug toggle
            if k == QtCore.Qt.Key_D and (mod & QtCore.Qt.ControlModifier):
                m = (getattr(self.tracker, "debug_flow_mode", 0) + 1) % 3
                self.tracker.debug_flow_mode = m
                if m == 0:
                    label = "OFF"
                elif m == 1:
                    label = "WITH processing"
                else:
                    label = "NO processing"
                print(f"[debug] ROI flow overlay: {label}")
                return

            # CTRL + B = LoG blob overlay toggle (B for Blob!!!)
            if k == QtCore.Qt.Key_B and (mod & QtCore.Qt.ControlModifier):
                cur = bool(getattr(self.tracker, "debug_show_log_blobs", False))
                self.tracker.debug_show_log_blobs = (not cur)
                print(f"[debug] LoG blob overlay: {self.tracker.debug_show_log_blobs}")
                return

            
            # Ctrl + E = Sharpen
            if k == QtCore.Qt.Key_E and (mod & QtCore.Qt.ControlModifier):
                m = (getattr(self.tracker, "edge_sharpen_mode", 0) + 1) % 3
                self.tracker.edge_sharpen_mode = m
                if   m == 0: label = "OFF"
                elif m == 1: label = "MILD"
                else:        label = "STRONG"
                print(f"[pre] edge-sharpen: {label}")
                return
            
            # Ctrl + L = Farneback Gaussian pyramid supercharger (OFF / ON / AUTO)
            if k == QtCore.Qt.Key_L and (mod & QtCore.Qt.ControlModifier):
                mode = int(getattr(self.tracker, "flow_pyr_mode", 0))
                mode = (mode + 1) % 3
                self.tracker.flow_pyr_mode = mode

                if mode == 0:
                    label = "OFF"
                elif mode == 1:
                    label = f"ALWAYS (L={int(getattr(self.tracker, 'flow_pyr_levels', FLOW_PYR_LEVELS_DEFAULT))})"
                else:
                    label = f"AUTO (max L={int(getattr(self.tracker, 'flow_pyr_levels', FLOW_PYR_LEVELS_DEFAULT))})"

                print(f"[flow] Gaussian pyramid supercharge: {label}")
                return


            if k == QtCore.Qt.Key_M and (mod & QtCore.Qt.ShiftModifier) and self.active_scene >= 0 and self.scenes[self.active_scene].rois:
                r = self.scenes[self.active_scene].rois[self.active_roi]
                modes = ["full", "orth"]
                cur = str(getattr(r, "cmat_proj", "full")).lower()
                try:
                    idx = modes.index(cur)
                except ValueError:
                    idx = 0
                r.cmat_proj = modes[(idx + 1) % len(modes)]
                self.scenes[self.active_scene].rois[self.active_roi] = r
                print(f"[cmat] ROI{self.active_roi}: proj={r.cmat_proj}")
                return

            if k == QtCore.Qt.Key_M and self.active_scene >= 0 and self.scenes[self.active_scene].rois:
                r = self.scenes[self.active_scene].rois[self.active_roi]

                profiles = [
                    ("off",      "off",    0),
                    ("moderate", "global", 180),
                    ("chaotic",  "global", 120),
                    ("smooth",   "global", 240),
                ]

                curp = str(getattr(self.tracker, "cmat_profile", "off")).lower()
                names = [p[0] for p in profiles]
                try:
                    idx = names.index(curp)
                except ValueError:
                    idx = 0
                idx = (idx + 1) % len(profiles)

                name, mode, hcap = profiles[idx]

                # set global runtime CMAT quality
                self.tracker.cmat_profile = name
                if hcap > 0:
                    self.tracker.cmat_max_h = int(hcap)

                # set ROI mode (off/global)
                r.cmat_mode = mode

                # optional: for chaotic, reduce lag automatically (screen shake)
                if name == "chaotic":
                    r.cmat_tau_ms = 0
                    r.cmat_alpha  = 1.0

                self.scenes[self.active_scene].rois[self.active_roi] = r
                print(f"[cmat] profile={name} mode={r.cmat_mode} max_h={getattr(self.tracker,'cmat_max_h',None)}")
                return

            # Ctrl+Shift+K toggles live structural preview (ROI-only)
            if (mod & QtCore.Qt.ControlModifier) and (mod & QtCore.Qt.ShiftModifier) and k == QtCore.Qt.Key_K:
                self.struct_view = not bool(getattr(self, "struct_view", False))
                print("[iglog] struct view:", "ON" if self.struct_view else "OFF")
                return


            # Ctrl+K => IG-LoG debug on active ROI (Tier-1, no OpenAI)
            if (mod & QtCore.Qt.ControlModifier) and k == QtCore.Qt.Key_K:
                try:
                    si = self.active_scene
                    if si < 0 or si >= len(self.scenes):
                        print("[iglog] no active scene")
                        return
                    sc = self.scenes[si]
                    if not sc.rois:
                        print("[iglog] no ROIs")
                        return
                    ri = max(0, min(self.active_roi, len(sc.rois)-1))
                    r = sc.rois[ri]

                    frame = getattr(self, "_cached_frame", None)
                    if frame is None:
                        # fallback: try to use current display frame if you store it elsewhere
                        frame = getattr(self, "frame_bgr", None)
                    if frame is None:
                        print("[iglog] no cached frame available")
                        return

                    x, y, w, h = map(int, r.rect)
                    H, W = frame.shape[:2]
                    x = max(0, min(x, W-1))
                    y = max(0, min(y, H-1))
                    w = max(1, min(w, W-x))
                    h = max(1, min(h, H-y))
                    crop = frame[y:y+h, x:x+w].copy()
                    if crop.size == 0:
                        print("[iglog] empty crop")
                        return

                    out_dir = os.path.join(os.path.dirname(__file__), "iglog_debug")
                    base = f"scene{si}_roi{ri}_t{int(time.time())}"
                    occ, poly, counts, abs_list, zc_list, ext_list  = iglog_debug_run(
                        crop,
                        out_dir=out_dir,
                        base=base,
                        sigmas=(1.2, 2.4, 4.0),
                        zc_thresh=0.004,
                        ext_abs_thresh=0.03
                    )
                    print(f"[iglog] ROI{ri}: occ_score={occ:.2f} extrema_counts={counts} poly_pts={int(poly.shape[0])} saved={out_dir}")
                except Exception as e:
                    print("[iglog] error:", e)
                return


            if k == QtCore.Qt.Key_K and self.active_scene>=0 and self.scenes[self.active_scene].rois:
              self.dump_roi_debug_csv(self.active_scene, self.active_roi); return
            if k == QtCore.Qt.Key_Escape:
                # auto-end and export open scenes with samples on exit
                for si, sc in enumerate(self.scenes):
                    if sc.times and sc.end is None: self.set_scene_end(si, self.frame_idx)
                # in ESC path
                if self.ff is not None: _ff_close(self.ff); self.ff=None
                if self.ocv_writer is not None: self.ocv_writer.release(); self.ocv_writer=None
                if self.writer is not None: self.writer.release()
                self.cap.release(); self.thumb_cap.release()
                QtWidgets.QApplication.quit(); return
            if k in (QtCore.Qt.Key_Space, QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self.playing = not self.playing
                if self.playing and self.prev_gray is None:
                    self.prev_gray    = self._cached_gray
                    self.last_sampled = self.frame_idx - 1   # guarantees "changed" on first tick
                return
            if k == QtCore.Qt.Key_F1:
                self.show_help_dialog(); return
            if k == QtCore.Qt.Key_F11:
                self.mw.toggle_fullscreen(); return
            if k in (QtCore.Qt.Key_H,) and self.active_scene>=0 and self.scenes[self.active_scene].rois:
                r = self.scenes[self.active_scene].rois[self.active_roi]
                r.debug = not getattr(r, "debug", False)
                self.scenes[self.active_scene].rois[self.active_roi] = r
                return
            
            if k == QtCore.Qt.Key_T and self.active_scene >= 0 and self.scenes[self.active_scene].rois:
                r = self.scenes[self.active_scene].rois[self.active_roi]
                modes = ["off", "cos", "cone"]
                cur = str(getattr(r, "axis_mode", "off")).lower()
                r.axis_mode = modes[(modes.index(cur)+1)%len(modes)]
                self.scenes[self.active_scene].rois[self.active_roi] = r
                return
            # Y = cycle vz_mode (curv / div / hybrid) for active ROI
            if k == QtCore.Qt.Key_Y and self.active_scene >= 0 and self.scenes[self.active_scene].rois:
                r = self.scenes[self.active_scene].rois[self.active_roi]
                modes = ["curv", "div", "hybrid"]
                cur = str(getattr(r, "vz_mode", "curv")).lower()
                try:
                    idx = modes.index(cur)
                except ValueError:
                    idx = 0
                r.vz_mode = modes[(idx + 1) % len(modes)]
                self.scenes[self.active_scene].rois[self.active_roi] = r
                print(f"[vz] ROI{self.active_roi}: vz_mode = {r.vz_mode}")
                return


            if k in (QtCore.Qt.Key_G,):
                for si, sc in enumerate(self.scenes):
                    if sc.times and sc.end is None:
                        self.set_scene_end(si, self.frame_idx)
                export_fullpass_overlay_and_csv(video_path, self.scenes, suffix="_exported_overlay.mp4")
                return

            if k in (QtCore.Qt.Key_Left,):
                self.frame_idx = max(0, self.frame_idx-1); self.playing=False; return
            if k in (QtCore.Qt.Key_Right,):
                self.frame_idx = min(self.N-1, self.frame_idx+1); self.playing=False; return
            if k in (QtCore.Qt.Key_Up,):
                self.frame_idx = min(self.N-1, self.frame_idx+10); self.playing=False; return
            if k in (QtCore.Qt.Key_Down,):
                self.frame_idx = max(0, self.frame_idx-10); self.playing=False; return

            if k in (QtCore.Qt.Key_Z,):
                r = self.scenes[self.active_scene].rois[self.active_roi]
                modes = ["off", "vz"]
                cur = str(getattr(r, "z_scale_mode", "off")).lower()
                r.z_scale_mode = modes[(modes.index(cur)+1)%len(modes)]
                self.scenes[self.active_scene].rois[self.active_roi] = r
                return

            if k in (QtCore.Qt.Key_I,):
                r = self.scenes[self.active_scene].rois[self.active_roi]
                # modes = ["flux_dog", "axis_jerk", "flux_dog_jerk"]
                modes = ["hybrid", "fast", "smooth"]
                cur = str(getattr(r, "impact_mode", "flux_dog")).lower()
                r.impact_mode = modes[(modes.index(cur)+1)%len(modes)]
                self.scenes[self.active_scene].rois[self.active_roi] = r
                print(f"[impact] ROI{self.active_roi}: impact_mode = {r.impact_mode}")
                return

            # scene ops
            if k in (QtCore.Qt.Key_N,):
                ai = self.scene_at(self.frame_idx)
                if ai >= 0:
                    self.active_scene = ai; print(f"[scene] overlap blocked; selected existing scene {ai}")
                else:
                    self._push_undo("New scene")
                    sc = Scene(self.frame_idx); self.scenes.append(sc); self.sort_scenes(); self.active_scene = self.scenes.index(sc); self.active_roi=-1
                    print(f"[scene] new scene {self.active_scene} start={self.frame_idx}")
                return
            if k in (QtCore.Qt.Key_A,) and self.active_scene>=0:
                self.set_scene_start(self.active_scene, self.frame_idx); return
            if k in (QtCore.Qt.Key_D,) and self.active_scene>=0:
                self.set_scene_end(self.active_scene, self.frame_idx); return
            # reopen scene end with Shift+E
            if k == QtCore.Qt.Key_E and (mod & QtCore.Qt.ShiftModifier) and self.active_scene >= 0:
                self.clear_scene_end(self.active_scene); return
            if k in (QtCore.Qt.Key_E,) and self.active_scene>=0:
                self.set_scene_end(self.active_scene, self.frame_idx); return

            # ROI ops
            if k in (QtCore.Qt.Key_U,) and self.active_scene>=0:
                print(f"[Add ROI]")
                self.adding=True; self.repicking=False; self.roi_tmp=None; self.bound_tmp=None; return
            if k in (QtCore.Qt.Key_R,) and self.active_scene>=0 and self.scenes[self.active_scene].rois:
                self.repicking=True; self.adding=False; self.roi_tmp=None; self.bound_tmp=None; return
            if k in (QtCore.Qt.Key_B,) and self.active_scene>=0 and self.scenes[self.active_scene].rois:
                r = self.scenes[self.active_scene].rois[self.active_roi]
                self._push_undo("Clear ROI bound", coalesce="roi_geom")
                self.scenes[self.active_scene].rois[self.active_roi] = roi_replace(r, bound=None)
                return
            if k in (QtCore.Qt.Key_C,) and self.active_scene>=0 and self.scenes[self.active_scene].rois:
                # Clear anchor for active ROI
                if self.active_scene >= 0 and self.active_roi >= 0:
                    sc = self.scenes[self.active_scene]
                    roi = sc.rois[self.active_roi]

                    self._push_undo("Clear ROI anchor", coalesce="roi_geom")

                    roi.anchor_user_set = False
                    roi._anchor_ready = False
                    roi._anchor_desc = None
                    roi._anchor_template = None
                    roi._anchor_last_ok = False
                    roi._anchor_last_sim = 0.0
                    roi._anchor_lost_count = 0

                    print("[anchor] cleared for ROI", self.active_roi)
                return
            if k in (QtCore.Qt.Key_BracketLeft,) and self.active_scene>=0 and self.scenes[self.active_scene].rois:
                self.active_roi = (self.active_roi - 1) % len(self.scenes[self.active_scene].rois); return
            if k in (QtCore.Qt.Key_BracketRight,) and self.active_scene>=0 and self.scenes[self.active_scene].rois:
                self.active_roi = (self.active_roi + 1) % len(self.scenes[self.active_scene].rois); return

            # misc
            if k in (QtCore.Qt.Key_V,): self.tracker.show_arrows = not self.tracker.show_arrows; return
            if k in (QtCore.Qt.Key_O,):
                self.recording = not self.recording
                if not self.recording:
                    if self.ff is not None:
                        _ff_close(self.ff); self.ff = None
                    if self.ocv_writer is not None:
                        self.ocv_writer.release(); self.ocv_writer = None
                return


            if k in (QtCore.Qt.Key_P,) and self.active_scene>=0:
                self.export_scene(self.active_scene); return
            if k in (QtCore.Qt.Key_S,):
                for si, sc in enumerate(self.scenes):
                    if sc.times and sc.end is None: self.set_scene_end(si, self.frame_idx)
                for si, sc in enumerate(self.scenes):
                    if sc.times: self.export_scene(si)
                return

    class VideoView(QtWidgets.QWidget):
        def __init__(self, ctrl):
            super().__init__()
            self.ctrl = ctrl
            self.setMouseTracking(True)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
        def sizeHint(self): return QtCore.QSize(self.ctrl.W, self.ctrl.H)
        def paintEvent(self, e):
            if not hasattr(self.ctrl, "hud"): return
            hud = self.ctrl.hud
            h, w = hud.shape[:2]
            qimg = QtGui.QImage(hud.data, w, h, w*3, QtGui.QImage.Format.Format_BGR888).copy()
            # letterbox fit
            W, H = self.width(), self.height()
            scale = min(max(W/ self.ctrl.W, 0.001), max(H/ self.ctrl.H, 0.001))
            newW, newH = int(self.ctrl.W*scale), int(self.ctrl.H*scale)
            ox, oy = (W-newW)//2, (H-newH)//2
            self.ctrl.view_scale = scale; self.ctrl.view_offset=(ox, oy)
            painter = QtGui.QPainter(self)
            painter.fillRect(self.rect(), QtGui.QColor(0,0,0))
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            painter.drawImage(QtCore.QRect(ox, oy, newW, newH), qimg)
            painter.end()
        def wheelEvent(self, ev: QtGui.QWheelEvent):
            delta = ev.angleDelta().y()
            mods  = ev.modifiers()
            self.ctrl.on_wheel(ev.position().x(), ev.position().y(),
                               delta,
                               bool(mods & QtCore.Qt.ControlModifier),
                               bool(mods & QtCore.Qt.ShiftModifier))
            ev.accept()
        # VideoView.mousePressEvent(...)
        def mousePressEvent(self, ev):
            if ev.button() == QtCore.Qt.LeftButton:
                mods = ev.modifiers()
                shift = bool(mods & QtCore.Qt.ShiftModifier)
                alt   = bool(mods & QtCore.Qt.AltModifier)
                ctrl  = bool(mods & QtCore.Qt.ControlModifier)
                self.ctrl.on_left_down(
                    ev.position().x(), ev.position().y(),
                    shift,
                    alt,
                    ctrl
                )

            elif ev.button() == QtCore.Qt.RightButton:
                self.ctrl.on_right_click(ev.position().x(), ev.position().y()); ev.accept()
            elif ev.button() == QtCore.Qt.MiddleButton:
                self.ctrl.on_middle_down(ev.position().x(), ev.position().y(), ev.modifiers()); ev.accept()


        def mouseMoveEvent(self, ev):
            self.ctrl.on_move(ev.position().x(), ev.position().y())
            if ev.buttons() & QtCore.Qt.MiddleButton:
                self.ctrl.on_middle_drag(ev.position().x(), ev.position().y(), ev.modifiers())
            ev.accept()

        def mouseReleaseEvent(self, ev):
            if ev.button() == QtCore.Qt.MiddleButton:
                self.ctrl.on_middle_up()
            if ev.button() == QtCore.Qt.LeftButton:
                self.ctrl.on_left_up(ev.position().x(), ev.position().y()); ev.accept()
        def mouseDoubleClickEvent(self, ev):
            if ev.button() == QtCore.Qt.LeftButton:
                self.ctrl.on_double(ev.position().x(), ev.position().y()); ev.accept()
        def keyPressEvent(self, ev): self.ctrl.on_key(ev)

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, ctrl):
            super().__init__()
            self.ctrl = ctrl
            self.setWindowTitle(APP_NAME + " (Qt)")
            self.view = VideoView(ctrl)
            self.setCentralWidget(self.view)
            self.resize(ctrl.W, ctrl.H)
        def toggle_fullscreen(self):
            if self.isFullScreen(): self.showNormal()
            else: self.showFullScreen()

    ctrl = Controller(ai_cfg)
    mw = MainWindow(ctrl)     # MainWindow creates the ONE visible VideoView
    ctrl.view = mw.view      # point controller to the visible widget
    ctrl.mw   = mw           # allow controller to toggle fullscreen, etc.
    def _rois_at_point(scene, x, y):
        hits=[]
        for r in scene.rois:
            rx, ry, rw, rh = map(int, r.rect)
            if rx <= x <= rx+rw and ry <= y <= ry+rh:
                hits.append(r)
        return hits
    filt = _WheelIOFilter(get_active_scene=lambda: ctrl.scenes[ctrl.active_scene] if 0 <= ctrl.active_scene < len(ctrl.scenes) else None,
                      get_rois_at=_rois_at_point)
    ctrl.view.installEventFilter(filt)

    mw.show()
    app.exec()
    # in ESC path
    if ctrl.ff is not None: _ff_close(ctrl.ff); ctrl.ff=None
    if ctrl.ocv_writer is not None: ctrl.ocv_writer.release(); ctrl.ocv_writer=None


    # graceful exit (if ESC missed)
    if ctrl.writer is not None: ctrl.writer.release()
    ctrl.cap.release(); ctrl.thumb_cap.release()
# ======================= end PySide6 backend =======================


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python \"Local Motion Tracker.py\" input.mp4 [--scale <0.25..1.0>]"
              " [--ai --i-accept-openai-policy]"
              " [--ai-vision [off|edges|crop|full]]"
              " [--ai-explicit] [--ai-on-export]")
        sys.exit(0)
    cv.setUseOptimized(True)
    try:
        cv.ocl.setUseOpenCL(True)
        print("[OpenCL] enabled =", cv.ocl.useOpenCL())
    except Exception:
        print("[OpenCL] not available")
    run_qt(sys.argv[1])