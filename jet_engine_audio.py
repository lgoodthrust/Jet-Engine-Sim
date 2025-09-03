import numpy as np
import sounddevice as sd
import threading, time

SR      = 48_000
BUFFER  = 10.0
BLOCK   = max(32, int(SR * BUFFER/1000.0))
GAIN_TONE  = 0.15
GAIN_NOISE = 0.15

def lav_float(val: float) -> float:
    MAX = 1e8
    MIN = -1e-8 if val >= 0 else -1e8
    return max(min(MAX, val), MIN)

def base_freq_func_01(x: float) -> float:
    x = lav_float(x)
    return x*3

def base_freq_func_02(x: float) -> float:
    x = lav_float(x)
    return x*2.95

def base_freq_func_03(x: float) -> float:
    x = lav_float(x)
    return x*2.9

def base_freq_func_04(x: float) -> float:
    x = lav_float(x)
    return x*2.85

def base_freq_func_05(x: float) -> float:
    x = lav_float(x)
    return x*2.8

def base_freq_func_06(x: float) -> float:
    x = lav_float(x)
    return x*2.75

def base_freq_func_07(x: float) -> float:
    x = lav_float(x)
    return x*2.7

def base_freq_func_08(x: float) -> float:
    x = lav_float(x)
    return x*2.65

def base_freq_func_09(x: float) -> float:
    x = lav_float(x)
    return x*2.6

def base_freq_func_10(x: float) -> float:
    x = lav_float(x)
    return x*2.1

def base_freq_func_11(x: float) -> float:
    x = lav_float(x)
    return x*2.05

def base_freq_func_12(x: float) -> float:
    x = lav_float(x)
    return x*2.0

def base_freq_func_13(x: float) -> float:
    x = lav_float(x)
    return x*1.35

def base_freq_func_14(x: float) -> float:
    x = lav_float(x)
    return x*1.3

def base_freq_func_15(x: float) -> float:
    x = lav_float(x)
    return x*1.25

TONE_FUNCS = [
    base_freq_func_01,
    base_freq_func_02,
    base_freq_func_03,
    base_freq_func_04,
    base_freq_func_05,
    base_freq_func_06,
    base_freq_func_07,
    base_freq_func_08,
    base_freq_func_09,
    base_freq_func_10,
    base_freq_func_11,
    base_freq_func_12,
    base_freq_func_13,
    base_freq_func_14,
    base_freq_func_15,
]

TONE_WEIGHTS = np.ones(len(TONE_FUNCS), dtype=np.float64)

def base_noise_func_01(x: float) -> float:
    x = lav_float(x)
    return max(0, x*100 - 900)

def base_noise_func_02(x: float) -> float:
    x = lav_float(x)
    return max(0, x*90 - 600)

def base_noise_func_03(x: float) -> float:
    x = lav_float(x)
    return max(0, x*80 - 300)

NOISE_FUNCS = [

]

NOISE_WEIGHTS = np.array([1.0, 1.0, 1.0], dtype=np.float64)
NOISE_Q       = np.array([1.0, 1.0, 1.0], dtype=np.float64)

class Param:
    def __init__(self, v=1.0):
        self._v = float(v); self._lock = threading.Lock()
    def get(self):
        with self._lock: return self._v
    def set(self, v):
        with self._lock: self._v = float(v)

x_param = Param(1.0)

reverb_wet        = Param(0.25)
reverb_decay      = Param(0.75)
reverb_room       = Param(1.0)
reverb_predelay_ms= Param(25.0)
reverb_damp       = Param(0.25)

class DampedComb:
    def __init__(self, delay_samples: int, fs: float):
        self.n = int(max(2, delay_samples))
        self.buf = np.zeros(self.n, dtype=np.float64)
        self.widx = 0
        self.lp_state = 0.0
        self.fs = fs

    def process(self, x: np.ndarray, feedback: float, damp: float) -> np.ndarray:

        y = np.empty_like(x)

        d = float(np.clip(damp, 0.0, 1.0))
        fb = float(np.clip(feedback, 0.0, 0.9995))

        for i in range(x.shape[0]):
            r = self.buf[self.widx]

            self.lp_state = (1.0 - d) * r + d * self.lp_state
            y[i] = r
            self.buf[self.widx] = x[i] + fb * self.lp_state
            self.widx += 1
            if self.widx >= self.n:
                self.widx = 0
        return y

class Allpass:
    def __init__(self, delay_samples: int, gain: float):
        self.n = int(max(2, delay_samples))
        self.buf = np.zeros(self.n, dtype=np.float64)
        self.widx = 0
        self.g = float(gain)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        g = self.g
        for i in range(x.shape[0]):
            bufout = self.buf[self.widx]
            v = x[i] + (-g) * bufout
            y[i] = bufout + g * v
            self.buf[self.widx] = v
            self.widx += 1
            if self.widx >= self.n:
                self.widx = 0
        return y

class PreDelay:
    def __init__(self, max_ms: float, fs: float):
        self.max_n = int(max_ms * 0.001 * fs) + 2
        self.buf = np.zeros(self.max_n, dtype=np.float64)
        self.widx = 0
        self.fs = fs
        self.current_n = 0

    def set_time_ms(self, ms: float):
        n = int(np.clip(ms, 0.0, self.max_n * 1000.0 / self.fs) * 0.001 * self.fs)
        self.current_n = int(np.clip(n, 0, self.max_n - 1))

    def process(self, x: np.ndarray) -> np.ndarray:
        n = self.current_n
        y = np.empty_like(x)
        for i in range(x.shape[0]):
            ridx = self.widx - n
            if ridx < 0: ridx += self.max_n
            y[i] = self.buf[ridx]
            self.buf[self.widx] = x[i]
            self.widx += 1
            if self.widx >= self.max_n:
                self.widx = 0
        return y

class Reverb:
    def __init__(self, fs: float):
        self.fs = float(fs)
        scale = fs / 44100.0

        comb_base = np.array([1557, 1617, 1491, 1422], dtype=np.int32)
        ap_base   = np.array([225,   556], dtype=np.int32)

        self.comb_base = (comb_base * scale).astype(int)
        self.ap_base   = (ap_base   * scale).astype(int)

        self._build_filters(room=1.0)

        self.predelay = PreDelay(max_ms=200.0, fs=fs)

    def _build_filters(self, room: float):

        room = float(np.clip(room, 0.5, 1.5))
        comb_delays = np.maximum(2, (self.comb_base * room).astype(int))
        ap_delays   = np.maximum(2, (self.ap_base   * room).astype(int))

        self.combs = [DampedComb(d, self.fs) for d in comb_delays]

        self.ap1 = Allpass(ap_delays[0], gain=0.5)
        self.ap2 = Allpass(ap_delays[1], gain=0.5)
        self._last_room = room

    def process(self, x: np.ndarray,
                wet: float,
                decay: float,
                room: float,
                predelay_ms: float,
                damp: float) -> np.ndarray:

        if abs(room - getattr(self, "_last_room", room)) > 1e-3:
            self._build_filters(room)

        self.predelay.set_time_ms(predelay_ms)

        xpd = self.predelay.process(x)

        comb_out = np.zeros_like(x)
        for c in self.combs:
            comb_out += c.process(xpd, feedback=np.clip(decay, 0.05, 0.98), damp=np.clip(damp, 0.0, 1.0))
        comb_out *= (1.0 / max(len(self.combs), 1))

        y = self.ap1.process(comb_out)
        y = self.ap2.process(y)

        wet = float(np.clip(wet, 0.0, 1.0))
        return (1.0 - wet) * x + wet * y

_reverb = Reverb(SR)

N_TONE  = len(TONE_FUNCS)
N_NOISE = len(NOISE_FUNCS)

tone_phases = np.zeros(N_TONE, dtype=np.float64)

noise_z1 = np.zeros(N_NOISE, dtype=np.float64)
noise_z2 = np.zeros(N_NOISE, dtype=np.float64)

class Slew:
    def __init__(self, rate_per_sec=6.0, start=1.0):
        self.current = float(start); self.rate = float(rate_per_sec)
    def step_to(self, target, frames):
        max_delta = self.rate * (frames / SR)
        delta = float(target) - self.current
        if abs(delta) > max_delta:
            self.current += np.sign(delta) * max_delta
        else:
            self.current = float(target)
        return self.current

slew_x = Slew(start=x_param.get())

def tone_freqs_from_x(x: float) -> np.ndarray:
    return np.array([f(x) for f in TONE_FUNCS], dtype=np.float64)

def noise_centers_from_x(x: float) -> np.ndarray:
    return np.array([f(x) for f in NOISE_FUNCS], dtype=np.float64)

def biquad_bandpass_coeffs(fc: float, Q: float, fs: float):

    fc = float(np.clip(fc, 10.0, fs * 0.45))
    Q  = float(max(Q, 1e-4))
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * Q)
    b0 =   Q * alpha
    b1 =   0.0
    b2 =  -Q * alpha
    a0 =   1.0 + alpha
    a1 =  -2.0 * np.cos(w0)
    a2 =   1.0 - alpha

    b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0
    return b0, b1, b2, a1, a2

twopi = 2.0 * np.pi
safe_eps = 1e-8
_rng = np.random.default_rng()

def audio_callback(outdata, frames, time_info, status):
    global tone_phases, noise_z1, noise_z2

    x = slew_x.step_to(x_param.get(), frames)

    f_tone = tone_freqs_from_x(x)
    phase_inc = f_tone / SR
    n = np.arange(frames, dtype=np.float64)
    ph_block = (tone_phases[:, None] + np.outer(phase_inc, n)) % 1.0
    tone = np.sin(twopi * ph_block) * TONE_WEIGHTS[:, None]
    tone_mix = tone.sum(axis=0) if N_TONE else 0.0
    tone_phases = (tone_phases + frames * phase_inc) % 1.0

    noise_mix = np.zeros(frames, dtype=np.float64)
    if N_NOISE:
        centers = noise_centers_from_x(x)
        for i in range(N_NOISE):
            fc = centers[i]
            b0, b1, b2, a1, a2 = biquad_bandpass_coeffs(fc, NOISE_Q[i], SR)

            xw = _rng.standard_normal(frames).astype(np.float64)


            y = np.empty(frames, dtype=np.float64)
            z1, z2 = noise_z1[i], noise_z2[i]
            for nidx in range(frames):
                xn = xw[nidx]
                yn = b0*xn + z1
                z1_new = b1*xn - a1*yn + z2
                z2     = b2*xn - a2*yn
                z1     = z1_new
                y[nidx] = yn
            noise_z1[i], noise_z2[i] = z1, z2
            noise_mix += NOISE_WEIGHTS[i] * y

    tone_norm  = np.sqrt(max((TONE_WEIGHTS**2).sum(), safe_eps))
    noise_norm = np.sqrt(max((NOISE_WEIGHTS**2).sum(), safe_eps))

    y = 0.0
    if N_TONE:
        y += (GAIN_TONE / tone_norm) * tone_mix
    if N_NOISE:
        y += (GAIN_NOISE / noise_norm) * (noise_mix / np.sqrt(2.0))

    y = _reverb.process(
        y.astype(np.float64), # pyright: ignore[reportAttributeAccessIssue]
        wet        = reverb_wet.get(),
        decay      = reverb_decay.get(),
        room       = reverb_room.get(),
        predelay_ms= reverb_predelay_ms.get(),
        damp       = reverb_damp.get(),
    )
    outdata[:, 0] = y.astype(np.float32)

def sweep_thread():
    try:
        A = 100
        while True:
                A = A + (A*0.01) + 0.1
                x_param.set(A)
                print(A)
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    t = threading.Thread(target=sweep_thread, daemon=True)
    t.start()
    with sd.OutputStream(samplerate=SR, channels=1, dtype='float32',
                         blocksize=BLOCK, latency='low',
                         callback=audio_callback):
        print(f"Running tone+noise bank: BLOCK={BLOCK} (~{1000*BLOCK/SR:.2f} ms)")
        while True:
            time.sleep(1)
