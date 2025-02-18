import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QTimer

#\\times10\^\{([\-, \d]*)\}
#**$1

def lav_float(val:float):
    MAX = 100000000.0#(sys.float_info.max- 1.0)
    MIN = -100000000.0#(sys.float_info.min + 1.0)
    if MIN > val > MAX:
        print("CLAMPPING!")
    return max(min(MAX, val), MIN)


def comp_graph(rpm:float) -> float:
    rpm = lav_float(rpm)
    x = rpm
    y=(0.000005*(x**1.995))/550
    return y


def drag_graph(rpm:float) -> float:
    rpm = lav_float(rpm)
    x = rpm
    y=(0.00000035*(x**2.27))-13/550
    return y


def start_graph(rpm:float) -> float:
    rpm = lav_float(rpm)
    x = rpm
    y=(-0.001*x)+1000
    return y
 

class BraytonCycleEngine:
    def __init__(self):
        ### System Constants ###
        self.atm_pressure = 101325  # Pascals (Standard atmospheric pressure)
        self.atm_temperature = 288.15  # Kelvin (15°C, standard atmosphere)
        self.air_specific_heat = 1005  # J/kg-K
        self.air_mol_mass = 0.02897  # Using molar mass of air
        self.air_density = 1.225  # kg/m³
        self.gas_constant = 287.05  # J/(kg·K)
        self.gamma = 1.4  # Heat capacity ratio

        ### Engine Parameters ###
        self.chamber_volume = 0.05 # m^3
        self.efficiency_compressor = 0.90
        self.combustion_efficiency = 0.98
        self.efficiency_turbine = 0.85
        self.rotor_inertia = 15  # rotor assembaly inertia
        self.friction_loss = 10.0  # Nm

        ### Fuel Constants ###
        self.fuel_energy_density = 113e5  # J/kg
        self.ideal_air_fuel_ratio = 2.0  # real ratio for fuel

        ### Engine State Variables ###
        self.starter_active = False
        self.throttle = 0.0 # Inital throttle value
        self.EngineRpm = 0  # Initial RPM
        self.prev_rpm = 0
        self.exhaust_temperature = self.atm_temperature
        self.chamber_pressure = self.atm_pressure
        self.air_mass_flow = 0.0  # Inital mass flow rate (kg/s)

    def starter(self):
        if self.starter_active and start_graph(self.EngineRpm) > 0.0:
            torque = start_graph(self.EngineRpm)
            self.EngineRpm += torque / self.rotor_inertia
        else:
            torque = 0
        return torque
    
    def compressor(self):
        T1 = self.atm_temperature
        
        self.compression_ratio = (comp_graph(self.EngineRpm) * self.efficiency_compressor)
        
        T2 = T1 * (self.compression_ratio ** ((self.gamma - 1) / self.gamma))
        T2_real = T1 + (T2 - T1) / self.efficiency_compressor
        
        self.air_mass_flow = comp_graph(self.EngineRpm)

        return T2_real

    def combustor(self):
        fuel_mass_flow = (self.air_mass_flow / self.ideal_air_fuel_ratio) * self.throttle

        if self.throttle == 0:
            fuel_mass_flow = 0.0
        
        heat_added = fuel_mass_flow * self.fuel_energy_density * self.combustion_efficiency
        delta_T = heat_added / (self.air_mass_flow * self.air_specific_heat + 1e-10)
        self.exhaust_temperature = delta_T + self.atm_temperature

        # Calculate number of moles in the combustion chamber
        num_moles = self.air_mass_flow / self.air_mol_mass

        # **New chamber pressure equation with fuel energy contribution**
        fuel_pressure_contribution = heat_added / self.chamber_volume  # Energy density effect
        self.chamber_pressure = ((num_moles * self.gas_constant * self.exhaust_temperature) / self.chamber_volume) + self.atm_pressure + fuel_pressure_contribution
        

        #self.chamber_pressure = comp_graph(self.EngineRpm) * (self.throttle) + self.atm_pressure
        
        return self.chamber_pressure, self.exhaust_temperature

    def turbine(self):
        if self.air_mass_flow <= 0:
            return 0

        # Expansion work formula: W = m * Cp * ΔT * η
        T4_ideal = self.exhaust_temperature * (self.atm_pressure / self.chamber_pressure) ** ((self.gamma - 1) / self.gamma)
        T4_real = self.exhaust_temperature - (self.exhaust_temperature - T4_ideal) * self.efficiency_turbine
        work_extracted = self.air_mass_flow * self.air_specific_heat * (self.exhaust_temperature - T4_real)

        # Convert to torque
        turbine_torque = work_extracted / (2 * np.pi * ((self.EngineRpm + 1.0) / 60))  # Nm

        return max(turbine_torque, 0)


    def nozzle(self):
        if self.air_mass_flow <= 0:
            return 0

        exit_velocity = np.sqrt(2 * self.air_specific_heat * (self.exhaust_temperature - self.atm_temperature))
        thrust = self.air_mass_flow * exit_velocity

        return max(thrust, 0)

    def update_engine(self, throttle):
        self.throttle = throttle
        temp = self.atm_temperature
        compression = self.compressor()
        chamber_pressure = self.combustor()[0]
        turbine_torque = self.turbine()
        thrust = self.nozzle()
        drag_force = drag_graph(self.EngineRpm)

        if self.starter_active:
            self.starter()
        
        rpm_change = (turbine_torque - drag_force - self.friction_loss) / self.rotor_inertia
        self.EngineRpm += rpm_change
        
        self.EngineRpm = max(self.EngineRpm, 0)

        rpm_change = self.EngineRpm - self.prev_rpm
        self.prev_rpm = self.EngineRpm
        
        return compression, temp, chamber_pressure, -1, turbine_torque, thrust, drag_force, rpm_change

    def toggle_starter(self):
        self.starter_active = not self.starter_active


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
        self.slider = QSlider(Qt.Vertical)
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
        v1, v2, v3, v4, v5, v6, v7, v8 = self.engine.update_engine(throttle)
        self.A1.setText(f"RPM: {self.engine.EngineRpm:.1f}")
        self.A2.setText(f"RPM delta: {v8:.2f} RPM/s")
        self.A3.setText(f"turb torque: {v5:.2f} N")
        self.A4.setText(f"thrust: {v6:.2f} N")
        self.A5.setText(f"drag: {v7:.2f} N")
        self.start_button.setText("Stop Starter" if self.engine.starter_active else "Start Engine")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EngineGUI()
    sys.exit(app.exec_())
