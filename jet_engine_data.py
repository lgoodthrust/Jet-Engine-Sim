from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Iterable, List
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------- Data classes ---------------------------

@dataclass(frozen=True)
class EngineGeometry:
    Zr: int  # rotor blade count
    Zs: int  # stator vane count (OGV/NGV)

@dataclass(frozen=True)
class OperatingPoint:
    rpm: float

@dataclass(frozen=True)
class ToneIndex:
    k: int               # blade-passing harmonic order (k >= 1)
    n: int               # stator index (|n| >= 0); n=0 -> pure rotor harmonic
    family: str          # '+' or '-' for Tyler–Sofrin families
    s: int = 0           # sideband order (>=0); 0 = on-harmonic
    sideband_sign: int = 0  # -1, 0, +1

# --------------------------- Core acoustics ---------------------------

def shaft_hz(rpm: float) -> float:
    return rpm / 60.0

def bpf_hz(rpm: float, Zr: int) -> float:
    return shaft_hz(rpm) * Zr

def tyler_sofrin_mode_m(k: int, n: int, family: str, Zr: int, Zs: int) -> int:
    if family == '+':
        return k * Zr + abs(n) * Zs
    elif family == '-':
        return k * Zr - abs(n) * Zs
    raise ValueError("family must be '+' or '-'")

def tone_frequency_hz(op: OperatingPoint, geom: EngineGeometry, idx: ToneIndex) -> float:
    f_shaft = shaft_hz(op.rpm)
    f_bpf   = bpf_hz(op.rpm, geom.Zr)
    f = idx.k * f_bpf
    if idx.s > 0 and idx.sideband_sign != 0:
        f += idx.sideband_sign * idx.s * f_shaft
    return max(0.0, f)

# --------------------------- Amplitude model ---------------------------

@dataclass(frozen=True)
class AmplitudeParams:
    A0_db: float = 100.0
    rpm_ref: float = 6000.0
    rpm_gain_db_per_20log: float = 1.0
    k_decay_db_per_order: float = 10.0
    n_decay_db: float = 4.0
    sideband_decay_db: float = 6.0
    jitter_db: float = 1.5
    clamp_min_db: float = 20.0
    clamp_max_db: float = 150.0

def synth_amplitude_db(op: OperatingPoint,
                       geom: EngineGeometry,
                       idx: ToneIndex,
                       P: AmplitudeParams) -> float:
    rpm_ratio = max(1e-6, op.rpm / P.rpm_ref)
    base = P.A0_db + P.rpm_gain_db_per_20log * 20.0 * math.log10(rpm_ratio)
    k_term = 0.0 if idx.k <= 1 else P.k_decay_db_per_order * math.log10(idx.k)
    n_term = P.n_decay_db * abs(idx.n)
    sb_term = P.sideband_decay_db * (idx.s if idx.s > 0 else 0)
    fam_bias = 0.6 if idx.family == '+' else 0.0
    jitter = random.gauss(0.0, P.jitter_db)
    A = base - k_term - n_term - sb_term + fam_bias + jitter
    return float(np.clip(A, P.clamp_min_db, P.clamp_max_db))

# --------------------------- Normalization ---------------------------

@dataclass(frozen=True)
class NormalizationConfig:
    """
    Per-RPM normalization to keep the *sum* of partials under control.
    mode:
      - 'rms'  : assumes random-ish phases (root-sum-of-squares). Good default.
      - 'peak' : worst-case in-phase sum (very conservative).
      - 'none' : no normalization (not recommended).
    """
    mode: str = "rms"
    target_rms: float = 0.2     # ~ -14 dBFS RMS per RPM slice before headroom
    target_peak: float = 0.8    # used if mode='peak'
    headroom_db: float = 6.0    # extra pad applied after compute
    dbfs_floor: float = -120.0  # floor for amplitude_dbfs

EPS = 1e-12

def db_to_lin(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, db / 20.0)

def apply_per_rpm_normalization(df: pd.DataFrame, cfg: NormalizationConfig) -> pd.DataFrame:
    df = df.copy()
    df["amp_rel"] = db_to_lin(df["amplitude_db"].to_numpy())

    # ----- compute per-RPM scale vectorized -----
    g = df.groupby("rpm")

    if cfg.mode == "none":
        denom = 1.0
        scale = np.full(len(df), 1.0, dtype=float)

    elif cfg.mode == "peak":
        # worst-case in-phase: sum of amplitudes
        denom = g["amp_rel"].transform("sum").to_numpy() + EPS
        target = cfg.target_peak

        scale = (target / denom)
        # headroom pad
        scale *= 10.0 ** (-cfg.headroom_db / 20.0)
        scale = np.clip(scale, 0.0, 1.0)

    elif cfg.mode == "rms":
        # uncorrelated RMS of sum: sqrt(sum(a^2)/2)
        # 1) sum of squares per group
        sumsq = g["amp_rel"].transform(lambda s: float(np.sum(s.values**2)))
        denom = np.sqrt(sumsq.to_numpy() / 2.0) + EPS
        target = cfg.target_rms

        scale = (target / denom)
        scale *= 10.0 ** (-cfg.headroom_db / 20.0)
        scale = np.clip(scale, 0.0, 1.0)

    else:
        raise ValueError("NormalizationConfig.mode must be 'rms', 'peak', or 'none'.")

    # ----- apply scale per row -----
    amp_lin = df["amp_rel"].to_numpy() * scale
    amp_lin = np.minimum(1.0, amp_lin)
    amp_dbfs = 20.0 * np.log10(np.maximum(EPS, amp_lin))
    amp_dbfs = np.maximum(cfg.dbfs_floor, amp_dbfs)

    df["amplitude_lin"]  = amp_lin
    df["amplitude_dbfs"] = amp_dbfs
    df["rpm_norm_scale"] = scale
    df["rpm_norm_mode"]  = cfg.mode
    return df

# --------------------------- Dataset generation ---------------------------

def generate_indices(K_max: int,
                     N_max: int,
                     sidebands: int,
                     include_rotor_only: bool = True) -> Iterable[ToneIndex]:
    for k in range(1, K_max + 1):
        if include_rotor_only:
            yield ToneIndex(k=k, n=0, family='+', s=0, sideband_sign=0)
            for s in range(1, sidebands + 1):
                for sign in (-1, +1):
                    yield ToneIndex(k=k, n=0, family='+', s=s, sideband_sign=sign)
        for n in range(1, N_max + 1):
            for fam in ('+', '-'):
                yield ToneIndex(k=k, n=n, family=fam, s=0, sideband_sign=0)
                for s in range(1, sidebands + 1):
                    for sign in (-1, +1):
                        yield ToneIndex(k=k, n=n, family=fam, s=s, sideband_sign=sign)

def generate_dataset(geom: EngineGeometry,
                     rpms: Iterable[float],
                     K_max: int = 8,
                     N_max: int = 3,
                     sidebands: int = 1,
                     amp_params: AmplitudeParams = AmplitudeParams(),
                     norm_cfg: NormalizationConfig = NormalizationConfig(),
                     seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    rows: List[dict] = []
    indices = list(generate_indices(K_max=K_max, N_max=N_max, sidebands=sidebands, include_rotor_only=True))

    for rpm in rpms:
        op = OperatingPoint(rpm=rpm)
        f_shaft = shaft_hz(rpm)
        f_bpf   = bpf_hz(rpm, geom.Zr)
        for idx in indices:
            f = tone_frequency_hz(op, geom, idx)
            if f <= 0.0:
                continue
            m = tyler_sofrin_mode_m(idx.k, idx.n, idx.family, geom.Zr, geom.Zs)
            A_db = synth_amplitude_db(op, geom, idx, amp_params)
            label = "rotor_harmonic" if idx.n == 0 else (f"TS_{idx.family}")
            sb_lbl = "center" if idx.s == 0 else ("upper" if idx.sideband_sign > 0 else "lower")
            rows.append({
                "rpm": rpm,
                "f_shaft_hz": f_shaft,
                "bpf_hz": f_bpf,
                "Zr": geom.Zr,
                "Zs": geom.Zs,
                "k": idx.k,
                "n": idx.n,
                "family": idx.family,
                "mode_m": m,
                "sideband_order": idx.s,
                "sideband_sign": idx.sideband_sign,
                "sideband_label": sb_lbl,
                "frequency_hz": f,
                "amplitude_db": A_db,   # modeled RELATIVE dB (not dBFS!)
                "component": label,
            })

    df = pd.DataFrame(rows).sort_values(
        ["rpm", "frequency_hz", "k", "n", "family", "sideband_order", "sideband_sign"]
    ).reset_index(drop=True)

    # <<< NEW: keep the mix under control per RPM slice >>>
    df = apply_per_rpm_normalization(df, norm_cfg)
    return df

# --------------------------- Config (no argparse) ---------------------------

def get_config() -> dict:
    return {
        "Zr": 16,
        "Zs": 32,
        "rpm_min": 2000.0,
        "rpm_max": 20_000.0,
        "rpm_steps": 300,
        "K_max": 8,
        "N_max": 3,
        "sidebands": 2,
        "seed": 54,
        "out": r"D:\code stuff\AAA\py scripts\GitHub Projects\Jet Engine Sim\turbine_acoustics.csv",
        # amplitude knobs
        "A0_db": 100.0,
        "rpm_ref": 8000.0,
        "rpm_gain": 1.0,
        "k_decay": 10.0,
        "n_decay": 4.0,
        "sb_decay": 6.0,
        "jitter_db": 2.0,
        # normalization knobs
        "norm_mode": "rms",      # 'rms' | 'peak' | 'none'
        "norm_target_rms": 0.2,  # used if mode='rms'
        "norm_target_peak": 0.8, # used if mode='peak'
        "norm_headroom_db": 6.0,
    }

def main():
    args = get_config()

    if args["rpm_steps"] < 2:
        raise ValueError("rpm_steps must be >= 2")
    if args["Zr"] <= 0 or args["Zs"] <= 0:
        raise ValueError("Zr and Zs must be positive")

    geom = EngineGeometry(Zr=args["Zr"], Zs=args["Zs"])
    rpms = np.linspace(args["rpm_min"], args["rpm_max"], num=int(args["rpm_steps"]), endpoint=True).round(6).tolist()

    amp_params = AmplitudeParams(
        A0_db=args["A0_db"],
        rpm_ref=args["rpm_ref"],
        rpm_gain_db_per_20log=args["rpm_gain"],
        k_decay_db_per_order=args["k_decay"],
        n_decay_db=args["n_decay"],
        sideband_decay_db=args["sb_decay"],
        jitter_db=args["jitter_db"],
    )

    norm_cfg = NormalizationConfig(
        mode=args["norm_mode"],
        target_rms=args["norm_target_rms"],
        target_peak=args["norm_target_peak"],
        headroom_db=args["norm_headroom_db"],
    )

    tones_per_rpm = args["K_max"] * (1 + 2*args["sidebands"]) + (2*args["N_max"]) * (1 + 2*args["sidebands"]) * args["K_max"]
    est_rows = int(tones_per_rpm * len(rpms))
    print(f"Estimated rows: ~{est_rows:,} (RPM points={len(rpms)}, tones/rpm≈{tones_per_rpm})")

    df = generate_dataset(
        geom=geom,
        rpms=rpms,
        K_max=max(1, args["K_max"]),
        N_max=max(0, args["N_max"]),
        sidebands=max(0, args["sidebands"]),
        amp_params=amp_params,
        norm_cfg=norm_cfg,
        seed=args["seed"],
    )

    out_path = Path(args["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6g")
    print(f"Wrote {len(df):,} rows to {out_path.resolve()}")
    print("Columns you’ll want in the synth: frequency_hz, amplitude_lin (already headroomed).")

if __name__ == "__main__":
    main()
