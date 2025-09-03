# sound_driver.py
import multiprocessing as mp
import jet_engine_csv_sound_profile_generator_sub as jecsvspg

def audio_process(rpm_queue: mp.Queue):
    # Init synth ONCE
    CSV_PATH = r"D:\code stuff\AAA\py scripts\GitHub Projects\Jet Engine Sim\turbine_acoustics.csv"
    bank = jecsvspg.load_tone_bank(CSV_PATH)

    with jecsvspg.EngineSynth(bank,
                              samplerate=48_000,
                              buffer_ms=100.0,
                              gain=0.15,
                              max_active_tones=512,
                              amp_slew_per_block_db=4.0,
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

        current_rpm = 3.0
        synth.set_rpm(current_rpm)

        while True:
            try:
                while not rpm_queue.empty():
                    current_rpm = rpm_queue.get_nowait()
                synth.set_rpm(current_rpm)
            except Exception as e:
                print("Audio process error:", e)
