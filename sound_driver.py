# sound_driver.py
import multiprocessing as mp
import jet_engine_csv_sound_profile_generator_sub as jecsvspg
import os

def audio_process(rpm_queue: mp.Queue):
    # Init synth ONCE
    script_dir = os.path.dirname(__file__)
    CSV = "turbine_acoustics.csv"
    csv_path = os.path.join(script_dir, CSV)
    bank = jecsvspg.load_tone_bank(csv_path)

    with jecsvspg.EngineSynth(bank,
                    samplerate=48_000,         # Audio sample rate in Hz (48 kHz standard for good fidelity)
                    buffer_ms=50.0,           # Buffer size in milliseconds (controls latency vs stability)
                    gain=0.15,                 # Master output gain applied after mixing tones/noise
                    max_active_tones=512,      # Maximum number of tones simultaneously synthesized

                    # -------- Amplitude / spectral shaping --------
                    amp_slew_per_block_db=3.0, # Max amplitude change (dB) per audio block, smooths fast jumps
                    spectral_tilt_per_k_db=0.25, # Additional roll-off in dB per rotor harmonic order k
                    rotor_db_trim=-1.0,         # Gain trim (dB) applied to rotor tones
                    ts_db_trim=-2.0,           # Gain trim (dB) applied to tone–stator interaction components

                    # -------- Broadband noise shaping --------
                    noise_db_at_idle=-5.0,    # Broadband noise floor (dB) at idle RPM
                    noise_db_at_max=-1.0,      # Broadband noise floor (dB) at max RPM

                    # -------- RPM dynamics --------
                    rpm_smooth_tau_blocks=0.5, # Time constant (in audio blocks) for smoothing RPM input
                    out_hpf_hz=25.0,           # High-pass filter cutoff frequency (Hz) to remove rumble/DC

                    # -------- Harmonic onset shaping --------
                    k_onset_rpm_per_order=250.0, # RPM threshold per harmonic order k before it fades in
                    ts_extra_onset_rpm=150.0,    # Extra RPM threshold for tone–stator harmonics

                    # reverbs
                    reverb_wet=0.35,     # overall reverb mix
                    reverb_room=0.05,    # longer tails
                    reverb_damp=0.75     # darker tail (higher = darker)
        ) as synth:


        current_rpm = 3.0
        synth.set_rpm(current_rpm)

        while True:
            try:
                while not rpm_queue.empty():
                    current_rpm = rpm_queue.get_nowait()
                synth.set_rpm(current_rpm)
            except Exception as e:
                print("Audio process error:", e)
