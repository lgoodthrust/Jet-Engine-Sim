import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QTimer
import multiprocessing as mp
import multiprocessing as mp
from sound_driver import audio_process

def lav_float(val: float):
    MAX = 2.5e5
    MIN = -2.5e-5 if val >= 0 else -2.5e5
    clamped = max(min(MAX, val), MIN)
    if clamped != val:
        print("CLAMPING!", val, "->", clamped)
    return clamped

def comp_graph(x: float) -> float:
    x = lav_float(x)
    return ((x**1.125)/10_000)

def start_graph(x: float) -> float:
    x = lav_float(x)
    return max(0, (-x)/15 + 500)

class BraytonCycleEngine:
    def __init__(self):

        # --- Ambient / gas properties (ISA SL approx) ---
        self.atm_pressure      = 101_325.0   # [Pa] ambient static pressure at sea level
        self.atm_temperature   = 288.15      # [K]  ambient static temperature at sea level
        self.air_specific_heat = 1005.0      # [J/(kg·K)] cp of dry air (≈ constant over your range)
        self.gas_constant      = 287.05      # [J/(kg·K)] specific gas constant for air (R)
        self.gamma             = 1.40        # [-] ratio of specific heats (cp/cv) for air

        # --- Component efficiencies / overall cycle settings ---
        self.efficiency_compressor  = 0.875  # [-] adiabatic/isentrope efficiency of compressor (η_c)
        self.combustion_efficiency  = 0.98   # [-] fraction of LHV released and absorbed by flow (η_b)
        self.efficiency_turbine     = 0.855  # [-] adiabatic/isentrope efficiency of turbine (η_t)

        # NOTE: This is the compressor pressure ratio (π_c). 1.05 is essentially no compression.
        # Small turboshaft/RC microturbine: π_c ≈ 2–6. Model engines ~3–4 are common.
        self.compression_ratio      = 3.0   # [-] compressor total-pressure ratio π_c

        # --- Combustor / fuel model ---
        self.TIT_max   = 1630.0              # [K] turbine temperature limit (a.k.a. Tt4,max)
        self.fuel_LHV  = 43e6                # [J/kg] lower heating value (Jet-A/kerosene ≈ 42–43 MJ/kg)

        # Fuel–air ratio bounds (mass_fuel / mass_air). Stoich for kerosene ≈ 0.068.
        # Idle FAR ~0.005–0.012; max FAR often 0.03–0.06 (temp-limited by TIT).
        self.f_min     = 0.0                 # [-] minimum FAR clamp (usually 0, you’ll clamp to idle separately)
        self.f_max     = 0.06                # [-] maximum FAR (don’t exceed to respect TIT and surge margin)
        self.f_idle    = 0.0                 # [-] idle FAR (reasonable)

        # --- Flow / map shaping helpers (model-specific “knobs”) ---
        self.pi_c_cap  = 5.0                 # [-] hard cap on π_c vs. corrected speed (prevents runaway)
        self.mdot_gain = 3.5                 # [kg/s per (normalized speed)] mass-flow gain factor; tune to match map

        # --- Rotor dynamics / losses (lumped) ---
        self.rotor_inertia        = 15.0      # [kg·m²] lumped polar moment of inertia of spool/prop/geartrain
        self.friction_loss        = 1.0       # [N·m] constant parasitic torque (bearings, seals) at any speed
        self.k_drag_visc          = 3.0      # [N·m per krpm] viscous torque coefficient (≈ linear in ω or small quadratic)
        self.tau_static           = 1.0       # [N·m] static (Coulomb) friction breakaway torque at very low RPM
        self.static_drag_cut_rpm  = 500.0     # [rpm] below this, include τ_static; above it, drop to dynamic model

        self.starter_active = False
        self.throttle = 0.0
        self.EngineRpm = 0.0
        self.prev_rpm = 0.0
        self.air_mass_flow = 0.0
        self.T2 = self.atm_temperature
        self.T3 = self.atm_temperature

    def starter(self) -> float:
        if not self.starter_active:
            return 0.0
        return max(0.0, start_graph(self.EngineRpm))

    def toggle_starter(self) -> None:
        self.starter_active = not self.starter_active

    def compressor(self) -> float:
        T1 = self.atm_temperature
        raw = comp_graph(self.EngineRpm)

        self.compression_ratio = max(1.0, min(1.0 + raw, self.pi_c_cap))

        tau_c = self.compression_ratio ** ((self.gamma - 1.0) / self.gamma)
        T2s = T1 * tau_c
        self.T2 = T1 + (T2s - T1) / max(self.efficiency_compressor, 1e-6)

        self.air_mass_flow = max(1e-3, self.mdot_gain * raw)
        return self.T2

    def combustor(self):
        if self.throttle <= 0.0 and self.EngineRpm > 800.0:
            f = self.f_idle
        else:
            f = 0.01 + 0.05 * self.throttle
        f = float(np.clip(f, self.f_min, self.f_max))

        if self.air_mass_flow <= 1e-3:
            self.T3 = self.T2
            return self.atm_pressure * self.compression_ratio, self.T3

        dT = (self.combustion_efficiency * f * self.fuel_LHV) / ((1.0 + f) * self.air_specific_heat)
        self.T3 = min(self.T2 + dT, self.TIT_max)

        P3 = self.atm_pressure * self.compression_ratio
        return P3, self.T3

    def turbine(self, P3):
        if self.air_mass_flow <= 0.0:
            return 0.0, self.T3

        P4 = self.atm_pressure
        tau_t = (P4 / max(P3, 1.0)) ** ((self.gamma - 1.0) / self.gamma)
        T4s = self.T3 * tau_t
        T4 = self.T3 - self.efficiency_turbine * (self.T3 - T4s)

        P_turb = self.air_mass_flow * self.air_specific_heat * (self.T3 - T4)
        omega = max(1.0, (self.EngineRpm * 2.0 * np.pi) / 60.0)
        tau_turb = max(0.0, P_turb / omega)
        return tau_turb, T4

    def nozzle(self, T4):
        if self.air_mass_flow <= 0.0:
            return 0.0
        Ve = np.sqrt(self.gamma * self.gas_constant * max(T4, 120.0))
        return max(0.0, self.air_mass_flow * Ve)

    def update_engine(self, throttle: float, dt: float):
        self.throttle = float(np.clip(throttle, 0.0, 1.0))

        T2 = self.compressor()
        P3, T3 = self.combustor()
        tau_turb, T4 = self.turbine(P3)
        thrust = self.nozzle(T4)

        omega = max(0.0, (self.EngineRpm * 2.0 * np.pi) / 60.0)
        tau_drag = self.k_drag_visc * omega + (self.tau_static if self.EngineRpm < self.static_drag_cut_rpm else 0.0)
        tau_fric = max(0.0, self.friction_loss)
        tau_start = self.starter()

        tau_net = tau_turb + tau_start - tau_drag - tau_fric

        domega = (tau_net / max(self.rotor_inertia, 1e-6))
        omega = max(0.0, omega + domega * dt)
        self.EngineRpm = omega * 60.0 / (2.0 * np.pi)

        rpm_rate = (self.EngineRpm - self.prev_rpm) / max(dt, 1e-6)
        self.prev_rpm = self.EngineRpm

        drag_force_like = tau_drag
        return T2, self.atm_temperature, P3, -1, tau_turb, thrust, drag_force_like, rpm_rate

class EngineGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.engine = BraytonCycleEngine()
        self.init_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_engine)
        self.timer.start(35)

    def init_ui(self):
        self.setWindowTitle("Jet Engine Throttle Control")
        self.slider = QSlider(Qt.Vertical) # pyright: ignore[reportAttributeAccessIssue]
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)

        self.A1 = QLabel("ABC: 0")
        self.A2 = QLabel("ABC: 0")
        self.A3 = QLabel("ABC: 0")
        self.A4 = QLabel("ABC: 0")
        self.A5 = QLabel("ABC: 0")

        self.start_button = QPushButton("Start Engine")
        self.start_button.clicked.connect(self.engine.toggle_starter)

        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.A1)
        layout.addWidget(self.A2)
        layout.addWidget(self.A3)
        layout.addWidget(self.A4)
        layout.addWidget(self.A5)
        layout.addWidget(self.start_button)
        self.setLayout(layout)
        self.show()

    def update_engine(self):
        throttle = self.slider.value() / 100.0
        dt = self.timer.interval() / 1000.0
        v1, v2, v3, v4, v5, v6, v7, v8 = self.engine.update_engine(throttle, dt)

        self.A1.setText(f"RPM: {self.engine.EngineRpm:.1f}")
        self.A2.setText(f"RPM rate: {v8:.1f} RPM/s")
        self.A3.setText(f"Turb torque: {v5:.2f} N·m")
        self.A4.setText(f"Thrust: {v6:.2f} N")
        self.A5.setText(f"Drag torque: {v7:.2f} N·m")
        self.start_button.setText("Stop Starter" if self.engine.starter_active else "Start Engine")

        sounds(self.engine.EngineRpm)

if __name__ == "__main__":
    rpm_queue = mp.Queue()
    audio_proc = mp.Process(target=audio_process, args=(rpm_queue,), daemon=True)
    audio_proc.start()

    def sounds(rpm: float):
        if not rpm_queue.full():
            rpm_queue.put(rpm)

    app = QApplication(sys.argv)
    window = EngineGUI()
    sys.exit(app.exec_())