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
                "amplitude_db": A_db,
                "component": label,
            })

    df = pd.DataFrame(rows).sort_values(
        ["rpm", "frequency_hz", "k", "n", "family", "sideband_order", "sideband_sign"]
    )
    df.reset_index(drop=True, inplace=True)
    return df

# --------------------------- Config (no argparse) ---------------------------

def get_config() -> dict:
    """
    Use underscore_keys to match the rest of the code.
    Adjust values as needed.
    """
    return {
        "Zr": 24,                      # rotor blades
        "Zs": 48,                      # stator vanes
        "rpm_min": 100.0,              # NOTE: very low -> dense low-freq lines
        "rpm_max": 10_000.0,
        "rpm_steps": 300,             # inclusive steps; consider reducing if file is huge
        "K_max": 8,
        "N_max": 3,
        "sidebands": 1,
        "seed": 42,
        "out": r"D:\code stuff\AAA\py scripts\GitHub Projects\Jet Engine Sim\turbine_acoustics.csv",
        # amplitude knobs
        "A0_db": 100.0,
        "rpm_ref": 6000.0,
        "rpm_gain": 1.0,
        "k_decay": 10.0,
        "n_decay": 4.0,
        "sb_decay": 6.0,
        "jitter_db": 1.5,
    }

def main():
    args = get_config()

    # Basic checks
    if args["rpm_steps"] < 2:
        raise ValueError("rpm_steps must be >= 2")
    if args["Zr"] <= 0 or args["Zs"] <= 0:
        raise ValueError("Zr and Zs must be positive")

    geom = EngineGeometry(Zr=args["Zr"], Zs=args["Zs"])
    rpms = np.linspace(args["rpm_min"], args["rpm_max"], num=int(args["rpm_steps"]), endpoint=True).tolist()

    amp_params = AmplitudeParams(
        A0_db=args["A0_db"],
        rpm_ref=args["rpm_ref"],
        rpm_gain_db_per_20log=args["rpm_gain"],
        k_decay_db_per_order=args["k_decay"],
        n_decay_db=args["n_decay"],
        sideband_decay_db=args["sb_decay"],
        jitter_db=args["jitter_db"],
    )

    # Rough row-count estimate to avoid surprises:
    # tones_per_rpm ≈ (K_max * (1 + 2*sidebands))   [n=0]
    #              + (2 * N_max * (1 + 2*sidebands)) [n>=1, families ±]
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
        seed=args["seed"],
    )

    out_path = Path(args["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path.resolve()}")

if __name__ == "__main__":
    main()
