from __future__ import annotations
import threading, time, math, random
from dataclasses import dataclass
from typing import Dict, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd
import sounddevice as sd

# ============================ Utilities ============================

def db_to_lin(db): return np.power(10.0, np.asarray(db) / 20.0)
def lin_to_db(x, eps: float = 1e-12): x = np.maximum(np.asarray(x), eps); return 20.0 * np.log10(x)
def clamp(x, lo, hi): return hi if x > hi else lo if x < lo else x
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

ToneKey = Tuple[int, int, str, int, int]  # (k, n, family, sideband_order, sideband_sign)

@dataclass
class EngineGeometry:
    Zr: int
    Zs: int

@dataclass
class ToneProfile:
    key: ToneKey
    rpm_grid: np.ndarray
    amp_db_grid: np.ndarray
    # derived fade info
    rpm_on: float
    rpm_off: float
    w_on: float
    w_off: float

    def amp_db(self, rpm: float) -> float:
        return float(np.interp(rpm, self.rpm_grid, self.amp_db_grid,
                               left=self.amp_db_grid[0], right=self.amp_db_grid[-1]))

    def amp_lin(self, rpm: float) -> float:
        return float(db_to_lin(self.amp_db(rpm)))

# ============================ Lightweight biquad ============================

class Biquad:
    def __init__(self):
        self.b0 = self.b1 = self.b2 = 0.0
        self.a0 = 1.0
        self.a1 = self.a2 = 0.0
        self.z1 = 0.0
        self.z2 = 0.0

    def set_peaking(self, fs: float, f0: float, Q: float, gain_db: float):
        f0 = max(1.0, min(0.45*fs, f0))
        A = 10**(gain_db/40.0)
        w0 = 2.0*math.pi*f0/fs
        alpha = math.sin(w0)/(2.0*Q)
        cosw0 = math.cos(w0)
        b0 = 1 + alpha*A
        b1 = -2*cosw0
        b2 = 1 - alpha*A
        a0 = 1 + alpha/A
        a1 = -2*cosw0
        a2 = 1 - alpha/A
        self.b0, self.b1, self.b2 = b0/a0, b1/a0, b2/a0
        self.a0, self.a1, self.a2 = 1.0, a1/a0, a2/a0

    def set_highpass(self, fs: float, f0: float, Q: float=0.707):
        f0 = max(1.0, min(0.45*fs, f0))
        w0 = 2.0*math.pi*f0/fs
        alpha = math.sin(w0)/(2.0*Q)
        cosw0 = math.cos(w0)
        b0 =  (1+cosw0)/2
        b1 = -(1+cosw0)
        b2 =  (1+cosw0)/2
        a0 =  1 + alpha
        a1 = -2*cosw0
        a2 =  1 - alpha
        self.b0, self.b1, self.b2 = b0/a0, b1/a0, b2/a0
        self.a0, self.a1, self.a2 = 1.0, a1/a0, a2/a0

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x, dtype=np.float64)
        b0,b1,b2,a1,a2 = self.b0,self.b1,self.b2,self.a1,self.a2
        z1,z2 = self.z1,self.z2
        for i in range(x.shape[0]):
            xn = float(x[i])
            yn = b0*xn + z1
            z1 = b1*xn - a1*yn + z2
            z2 = b2*xn - a2*yn
            y[i] = yn
        self.z1,self.z2 = z1,z2
        return y

# ============================ Tone Bank with smooth fade ============================

class ToneBank:
    """
    - Builds amplitude interpolants from CSV
    - Derives frequencies each block
    - Assigns micro-detune
    - Precomputes soft fade windows (rpm_on/off, logistic width) per tone
    """
    def __init__(self, df: pd.DataFrame,
                 rng_seed: int = 1337,
                 detune_cents: float = 20.0,
                 onset_rel_db: float = -30.0,    # onset at (max_db + onset_rel_db)
                 fade_width_frac: float = 0.03,  # 3% of rpm span
                 fade_min_rpm: float = 150.0):
        self.geom = EngineGeometry(int(df["Zr"].iloc[0]), int(df["Zs"].iloc[0]))
        self.keys: List[ToneKey] = []
        self.profiles: Dict[ToneKey, ToneProfile] = {}
        self.detune: Dict[ToneKey, float] = {}
        self.rpm_min = float(df["rpm"].min())
        self.rpm_max = float(df["rpm"].max())
        self.rpm_span = max(1.0, self.rpm_max - self.rpm_min)
        self.onset_rel_db = float(onset_rel_db)
        self.fade_width_frac = float(fade_width_frac)
        self.fade_min_rpm = float(fade_min_rpm)

        required = {"rpm","k","n","family","sideband_order","sideband_sign","amplitude_db","Zr","Zs"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        # Build profiles + fade windows
        for (k, n, fam, s, ss), g in df.groupby(["k","n","family","sideband_order","sideband_sign"], sort=True):
            g = g.sort_values("rpm")
            rpm_grid = g["rpm"].to_numpy(dtype=float)
            amp_db_grid = g["amplitude_db"].to_numpy(dtype=float)
            max_db = float(np.max(amp_db_grid))
            thr = max_db + self.onset_rel_db  # e.g., -30 dB below the peak

            above = amp_db_grid >= thr
            if np.any(above):
                idxs = np.flatnonzero(above)
                rpm_on = float(rpm_grid[idxs[0]])
                rpm_off = float(rpm_grid[idxs[-1]])
            else:
                rpm_on = float(rpm_grid[0] + 0.1*(rpm_grid[-1]-rpm_grid[0]))
                rpm_off = float(rpm_grid[-1])

            w = max(self.fade_min_rpm, self.fade_width_frac * self.rpm_span)
            prof = ToneProfile(
                key=(int(k), int(n), str(fam), int(s), int(ss)),
                rpm_grid=rpm_grid,
                amp_db_grid=amp_db_grid,
                rpm_on=rpm_on, rpm_off=rpm_off,
                w_on=w, w_off=w
            )
            self.keys.append(prof.key)
            self.profiles[prof.key] = prof

        # Fixed micro-detune per tone (uniform ± detune_cents)
        rng = random.Random(rng_seed)
        def cents_to_ratio(c): return 2.0**(c/1200.0)
        for k in self.keys:
            c = rng.uniform(-detune_cents, detune_cents)
            self.detune[k] = cents_to_ratio(c)

    # ---- Frequency functions (derived) ----
    def f_shaft(self, rpm: float) -> float: return rpm / 60.0
    def bpf(self, rpm: float) -> float: return self.f_shaft(rpm) * self.geom.Zr

    def tone_freq_hz(self, rpm: float, key: ToneKey) -> float:
        k, _, _, s, ss = key
        f = k * self.bpf(rpm)
        if s > 0 and ss != 0: f += ss * s * self.f_shaft(rpm)
        return max(0.0, f * self.detune[key])

    # ---- Amplitude & smooth fade gate ----
    def tone_amp_lin(self, rpm: float, key: ToneKey) -> float:
        return self.profiles[key].amp_lin(rpm)

    def tone_gate(self, rpm: float, key: ToneKey,
                  k_onset_rpm_per_order: float = 250.0,
                  ts_extra_onset_rpm: float = 150.0) -> float:
        """Smooth logistic fade: g_on(rpm) * g_off(rpm). Later onset for higher k and TS families."""
        prof = self.profiles[key]
        k, n, fam, s, ss = key
        # Later appearance for higher orders (k>=1) and TS families (n>0)
        onset_shift = (k-1) * k_onset_rpm_per_order + (ts_extra_onset_rpm if n > 0 else 0.0)
        rpm_on = prof.rpm_on + onset_shift
        rpm_off = prof.rpm_off  # usually end of range
        w_on = prof.w_on
        w_off = prof.w_off
        g_on = sigmoid((rpm - rpm_on) / max(1.0, w_on))
        g_off = sigmoid((rpm_off - rpm) / max(1.0, w_off))
        return float(g_on * g_off)

# ============================ Broadband noise bed ============================

class NoiseBed:
    def __init__(self, fs: int, hp_hz: float = 20.0):
        self.fs = int(fs)
        self.brown_state = 0.0
        self.hp = Biquad(); self.hp.set_highpass(self.fs, hp_hz, Q=0.707)
        self.formant = Biquad()

    def update_formant(self, center_hz: float, Q: float, gain_db: float):
        self.formant.set_peaking(self.fs, center_hz, Q=Q, gain_db=gain_db)

    def render(self, n: int, tilt_mix: float = 0.3) -> np.ndarray:
        w = np.random.uniform(-1.0, 1.0, size=n).astype(np.float64)
        y = np.empty_like(w)
        state = self.brown_state
        for i in range(n):
            state = 0.995*state + w[i]
            y[i] = state
        self.brown_state = state
        x = (1.0 - tilt_mix)*y + tilt_mix*w
        x = self.hp.process(x)
        x = self.formant.process(x)
        x *= 0.02
        return x

# ============================ Real-time Synth ============================

class EngineSynth:
    def __init__(self,
                 bank: ToneBank,
                 samplerate: int = 48_000,
                 buffer_ms: float = 100.0,
                 gain: float = 0.12,
                 max_active_tones: int = 220,
                 amp_slew_per_block_db: float = 4.0,
                 spectral_tilt_per_k_db: float = 0.25,
                 rotor_db_trim: float = 0.0,
                 ts_db_trim: float = -2.0,
                 noise_db_at_idle: float = -18.0,
                 noise_db_at_max: float  = -8.0,
                 noise_formant_Q: float = 1.0,
                 noise_formant_gain_db: float = 6.0,
                 rpm_smooth_tau_blocks: float = 1.5,
                 out_hpf_hz: float = 18.0,
                 # NEW fade params
                 k_onset_rpm_per_order: float = 250.0,
                 ts_extra_onset_rpm: float = 150.0,
                 ):
        self.bank = bank
        self.fs = int(samplerate)
        self.block = max(16, int(round(self.fs * buffer_ms / 1000.0)))
        self.gain = float(gain)
        self.max_active = int(max_active_tones)
        self.amp_slew_db = float(amp_slew_per_block_db)
        self.spectral_tilt_per_k_db = float(spectral_tilt_per_k_db)
        self.rotor_trim_lin = float(db_to_lin(rotor_db_trim))
        self.ts_trim_lin    = float(db_to_lin(ts_db_trim))
        self.noise_idle_lin = float(db_to_lin(noise_db_at_idle))
        self.noise_max_lin  = float(db_to_lin(noise_db_at_max))
        self.noise_Q = float(noise_formant_Q)
        self.noise_formant_gain_db = float(noise_formant_gain_db)
        self.rpm_alpha = float(1.0 / max(1.0, rpm_smooth_tau_blocks))
        self.k_onset_rpm_per_order = float(k_onset_rpm_per_order)
        self.ts_extra_onset_rpm = float(ts_extra_onset_rpm)

        # Shared RPM
        self._rpm_raw = 3000.0
        self._rpm_smoothed = 3000.0
        self._rpm_lock = threading.Lock()

        # State per tone
        self._keys = list(self.bank.keys)
        self._phase = np.zeros(len(self._keys), dtype=np.float64)
        self._last_amp = np.zeros(len(self._keys), dtype=np.float64)

        # Noise & output filters
        self.noise = NoiseBed(self.fs, hp_hz=out_hpf_hz)
        self.out_hpf = Biquad(); self.out_hpf.set_highpass(self.fs, out_hpf_hz, Q=0.707)

        # Stream
        self.stream = sd.OutputStream(
            samplerate=self.fs,
            channels=1,
            blocksize=self.block,
            dtype="float32",
            callback=self._callback,
            latency="low",
        )

    # ---------- RPM control ----------
    def set_rpm(self, rpm: float):
        with self._rpm_lock:
            self._rpm_raw = float(max(0.0, rpm))

    def get_rpm(self) -> float:
        with self._rpm_lock:
            return float(self._rpm_raw)

    # ---------- Helpers ----------
    def _tone_trim_lin(self, key: ToneKey) -> float:
        return self.rotor_trim_lin if key[1] == 0 else self.ts_trim_lin

    def _tone_tilt_db(self, key: ToneKey) -> float:
        k = key[0]
        return 0.0 if k <= 1 else -self.spectral_tilt_per_k_db * (k - 1)

    def _noise_gain_for_rpm(self, rpm: float) -> float:
        t = 0.0 if self.bank.rpm_max <= self.bank.rpm_min else (rpm - self.bank.rpm_min) / (self.bank.rpm_max - self.bank.rpm_min)
        t = clamp(t, 0.0, 1.0)
        g_db = (1.0 - t) * lin_to_db(self.noise_idle_lin) + t * lin_to_db(self.noise_max_lin)
        return float(db_to_lin(g_db))

    # ---------- Audio callback ----------
    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(status, flush=False)

        # Smooth RPM
        rpm_now = self.get_rpm()
        self._rpm_smoothed = (1.0 - self.rpm_alpha) * self._rpm_smoothed + self.rpm_alpha * rpm_now
        rpm = self._rpm_smoothed

        # 1) Frequencies for all tones
        all_freqs = np.fromiter((self.bank.tone_freq_hz(rpm, k) for k in self._keys),
                                dtype=np.float64, count=len(self._keys))
        valid = (all_freqs >= 50.0) & (all_freqs <= 20_000.0)
        valid_idx = np.nonzero(valid)[0]
        freqs = all_freqs[valid]
        keys  = [self._keys[i] for i in valid_idx]
        phase = self._phase[valid_idx]
        lastA = self._last_amp[valid_idx]

        # 2) Target amplitudes (linear) with trims, tilt, and NEW smooth fade gate
        amps = np.empty(len(keys), dtype=np.float64)
        for i, key in enumerate(keys):
            a = self.bank.tone_amp_lin(rpm, key)
            a *= self._tone_trim_lin(key)
            a *= float(db_to_lin(self._tone_tilt_db(key)))
            g = self.bank.tone_gate(rpm, key,
                                    k_onset_rpm_per_order=self.k_onset_rpm_per_order,
                                    ts_extra_onset_rpm=self.ts_extra_onset_rpm)
            amps[i] = a * g

        # 3) Keep top-N loudest
        if self.max_active < len(keys):
            idx_rel = np.argpartition(amps, -self.max_active)[-self.max_active:]
            idx_rel.sort()
            freqs = freqs[idx_rel]
            amps  = amps[idx_rel]
            phase = phase[idx_rel]
            lastA = lastA[idx_rel]
            active_idx_global = valid_idx[idx_rel]
        else:
            active_idx_global = valid_idx

        # 4) Amplitude slew-limit in dB
        eps = 1e-12
        target_db = 20.0 * np.log10(np.maximum(amps, eps))
        curr_db   = 20.0 * np.log10(np.maximum(lastA, eps))
        delta_db  = np.clip(target_db - curr_db, -self.amp_slew_db, self.amp_slew_db)
        smoothed_db = curr_db + delta_db
        A = np.power(10.0, smoothed_db / 20.0)

        # 5) Generate tones
        t_idx = np.arange(frames, dtype=np.float64)
        omega = (2.0 * np.pi * freqs) / self.fs
        phases = phase[:, None] + omega[:, None] * t_idx[None, :]
        tones  = (A[:, None] * np.sin(phases)).sum(axis=0)

        # 6) Noise bed (formant ~ 0.8*BPF)
        bpf = self.bank.bpf(rpm)
        self.noise.update_formant(center_hz=max(60.0, 0.8*bpf),
                                  Q=1.0, gain_db=6.0)
        noise_block = self.noise.render(frames, tilt_mix=0.3)
        noise_gain = self._noise_gain_for_rpm(rpm)
        chunk = tones + noise_gain * noise_block

        # 7) Update states
        new_phase = (phase + omega * frames) % (2.0 * np.pi)
        self._phase[active_idx_global] = new_phase
        self._last_amp[active_idx_global] = A

        # 8) Output: HPF + soft clip
        x = self.out_hpf.process(self.gain * chunk.astype(np.float64))
        x = x / (1.0 + np.abs(x))
        outdata[:, 0] = x.astype(np.float32)

    def __enter__(self): self.stream.start(); return self
    def __exit__(self, exc_type, exc, tb): self.stream.stop(); self.stream.close()

# ============================ Driver / Demo ============================

def load_tone_bank(csv_path: str | Path) -> ToneBank:
    df = pd.read_csv(csv_path)
    return ToneBank(df,
                    rng_seed=1337,
                    detune_cents=18.0,
                    onset_rel_db=-30.0,   # consider -35..-25
                    fade_width_frac=0.03, # 3% of rpm span
                    fade_min_rpm=120.0)

def demo_rpm_controller(synth: EngineSynth, rpm_idle: float, rpm_max: float, dwell_s: float = 0.8, steps: int = 350):
    try:
        while True:
            for v in np.linspace(rpm_idle, rpm_max, steps): synth.set_rpm(v); time.sleep(0.05)
            time.sleep(dwell_s)
            for v in np.linspace(rpm_max, rpm_idle, steps): synth.set_rpm(v); time.sleep(0.05)
            time.sleep(dwell_s)
    except KeyboardInterrupt:
        pass

def main():
    CSV_PATH   = r"D:\code stuff\AAA\py scripts\GitHub Projects\Jet Engine Sim\turbine_acoustics.csv"
    SR         = 48_000
    BUF_MS     = 100.0
    GAIN       = 0.15
    MAX_TONES  = 255
    AMP_SLEW   = 4.0

    RPM_IDLE   = 1_000.0
    RPM_MAX    = 10_000.0

    # Build + run
    bank = load_tone_bank(CSV_PATH)
    print(f"Starting v2.1 @ {SR} Hz, ~{BUF_MS:.0f} ms blocks, max_tones={MAX_TONES}")
    with EngineSynth(bank,
                     samplerate=SR,
                     buffer_ms=BUF_MS,
                     gain=GAIN,
                     max_active_tones=MAX_TONES,
                     amp_slew_per_block_db=AMP_SLEW,
                     spectral_tilt_per_k_db=0.25,
                     rotor_db_trim=0.0,
                     ts_db_trim=-2.0,
                     noise_db_at_idle=-18.0,
                     noise_db_at_max=-8.0,
                     rpm_smooth_tau_blocks=1.5,
                     out_hpf_hz=18.0,
                     k_onset_rpm_per_order=250.0,
                     ts_extra_onset_rpm=150.0
                     ) as synth:
        demo_rpm_controller(synth, RPM_IDLE, RPM_MAX, dwell_s=0.6, steps=500)

if __name__ == "__main__":
    main()
