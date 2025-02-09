import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QTimer


class BraytonCycleEngine:
    def __init__(self):
        ### System Constants ###
        self.atm_pressure = 101325  # Pascals (Standard atmospheric pressure)
        self.atm_temperature = 288.15  # Kelvin (15°C, standard atmosphere)
        self.air_specific_heat = 1005  # J/kg-K
        self.gas_constant = 287.05  # J/(kg·K)
        self.air_density = 1.225  # kg/m³
        self.gamma = 1.4  # Heat capacity ratio

        ### Engine Parameters ###
        self.cross_sectional_area = 0.5  # m²
        self.chamber_volume = 0.02  # m³ (approximate size for a jet engine combustor)
        self.combustion_efficiency = 0.98
        self.compressor_expo = 2.0  # Reduced to prevent excessive airflow at low RPM
        self.compression_ratio_base = 2.0  # Base compression ratio
        self.compression_ratio_rpm_factor = 0.0005  # Compression ratio scaling with RPM
        self.base_mass_flow_rate = 20.0  # kg/s
        self.efficiency_compressor = 0.85
        self.efficiency_turbine = 0.90
        self.rotor_inertia = 125  # Increased inertia for realistic spool-up time
        self.drag_coefficient = 0.25
        self.friction_loss = 8.0  # Nm
        self.k_velocity_scale = 0.08

        ### Starter Motor ###
        self.starter_torque = 300.0  # Nm
        self.starter_max_rpm = 5000  # Max starter RPM

        ### Fuel Constants ###
        self.fuel_energy_density = 43.0e6  # J/kg
        self.ideal_air_fuel_ratio = 15.0  # More realistic ratio for jet engines

        ### Engine State Variables ###
        self.starter_active = False
        self.throttle = 0.0
        self.EngineRpm = 0  # Initial RPM
        self.exhaust_temperature = self.atm_temperature
        self.chamber_pressure = self.atm_pressure
        self.air_mass_flow = 0.0  # kg/s

    def starter(self):
        if self.starter_active:
            torque = self.starter_torque * max(0, 1 - self.EngineRpm / self.starter_max_rpm)
            rpm_change = torque / self.rotor_inertia
            self.EngineRpm += rpm_change

    def compressor(self):
        T1 = self.atm_temperature
        
        self.compression_ratio = self.compression_ratio_base + (self.EngineRpm * self.compression_ratio_rpm_factor)
        
        T2 = T1 * (self.compression_ratio ** ((self.gamma - 1) / self.gamma))
        T2_real = T1 + (T2 - T1) / self.efficiency_compressor
        
        self.air_mass_flow = self.base_mass_flow_rate * (self.EngineRpm / 10000) ** self.compressor_expo  
        return T2_real, T1

    def combustor(self):
        fuel_mass_flow = min(self.air_mass_flow / self.ideal_air_fuel_ratio, 1.5) * self.throttle  
        
        heat_added = fuel_mass_flow * self.fuel_energy_density * self.combustion_efficiency
        delta_T = heat_added / (self.air_mass_flow * self.air_specific_heat + 1e-6)
        self.exhaust_temperature = delta_T + self.atm_temperature
        
        num_moles = self.air_mass_flow / 0.02897  # Using molar mass of air
        fuel_pressure_contribution = heat_added / self.chamber_volume
        self.chamber_pressure = ((num_moles * self.gas_constant * self.exhaust_temperature) / self.chamber_volume) + self.atm_pressure + fuel_pressure_contribution
        
        return self.chamber_pressure, self.exhaust_temperature

    def turbine(self):
        if self.air_mass_flow <= 0:
            return 0

        T4_ideal = self.exhaust_temperature * (self.atm_pressure / self.chamber_pressure) ** ((self.gamma - 1) / self.gamma)
        T4_real = self.exhaust_temperature - (self.exhaust_temperature - T4_ideal) * self.efficiency_turbine

        power_extracted = self.air_mass_flow * self.air_specific_heat * (self.exhaust_temperature - T4_real)
        turbine_torque = power_extracted / (2 * np.pi * (self.EngineRpm + 1) / 60)

        return turbine_torque

    def nozzle(self):
        if self.air_mass_flow <= 0:
            return 0

        exit_velocity = np.sqrt(2 * self.air_specific_heat * (self.exhaust_temperature - self.atm_temperature))
        thrust = self.air_mass_flow * exit_velocity
        return max(thrust, 0)

    def update_engine(self, throttle):
        self.throttle = throttle
        self.compressor()
        self.combustor()
        turbine_torque = self.turbine()
        thrust = self.nozzle()
        
        drag_force = 0.5 * self.drag_coefficient * self.air_density * self.cross_sectional_area * (self.EngineRpm * self.k_velocity_scale) ** 2
        
        rpm_change = (turbine_torque - drag_force - self.friction_loss) / self.rotor_inertia
        self.EngineRpm += rpm_change
        
        if self.starter_active:
            self.starter()
        
        self.EngineRpm = max(self.EngineRpm, 0)
        return thrust, drag_force

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
        
        self.label_rpm = QLabel("RPM: 0")
        self.label_thrust = QLabel("Thrust: 0 N")
        self.label_drag = QLabel("Drag: 0 N")
        
        self.start_button = QPushButton("Start Engine")
        self.start_button.clicked.connect(self.engine.toggle_starter)
        
        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.label_rpm)
        layout.addWidget(self.label_thrust)
        layout.addWidget(self.label_drag)
        layout.addWidget(self.start_button)
        self.setLayout(layout)
        self.show()

    def update_engine(self):
        throttle = self.slider.value() / 100.0
        thrust, drag = self.engine.update_engine(throttle)
        self.label_rpm.setText(f"RPM: {self.engine.EngineRpm:.0f}")
        self.label_thrust.setText(f"Thrust: {thrust:.2f} N")
        self.label_drag.setText(f"Drag: {drag:.2f} N")
        self.start_button.setText("Stop Starter" if self.engine.starter_active else "Start Engine")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EngineGUI()
    sys.exit(app.exec_())
