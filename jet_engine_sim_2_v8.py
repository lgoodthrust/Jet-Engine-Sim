#### INFO ### DO NOT TOUCH THESE COMMENTS!!!
#standard: IUPAC

### sources ###

# https://en.wikipedia.org/wiki/Centrifugal_compressor
# https://en.wikipedia.org/wiki/Axial_compressor
# https://en.wikipedia.org/wiki/Combustor
# https://en.wikipedia.org/wiki/Turbine
# https://en.wikipedia.org/wiki/Overall_pressure_ratio
# https://en.wikipedia.org/wiki/Compressor_map
# https://www.grc.nasa.gov/www/k-12/airplane/compexp.html

### formulas ###

# air specific heat capacity
# capacity = heat engergy / ( air mass * temperature delta )

# Angular momentum
# ang momentum = mass * velocity * radius

# inertia
# inertia = ang momentum / ang velocity (rpm)

# drag
# Drag = drag coeff * fluid density * ( velocity**2 / 2 ) * surface area

# volume
# density = mass / volume
 
# Isentropic Compression / Expansion (heat capacity ratio)
# https://www.grc.nasa.gov/www/k-12/airplane/compexp.html


### simulated model (sequence) ###

# atmosphereic starting condictions / air
# compressor / compression
# combustion / combustor / fuel
# expantion / combustion / fuel
# turbine / recovery
# nozzle / thrust


import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QTimer


class BraytonCycleEngine:
    def __init__(self):
        ### System Constants ###
        self.atm_pressure = 100000  # Pascals
        self.atm_temperature = 273.15  # Kelvin
        self.air_specific_heat = 1005  # J/kg-K
        self.gas_constant = 287.05  # J/(kg·K)
        self.air_density = 1.225  # kg/m³
        self.gamma = 1.4  # Heat capacity ratio

        ### Engine Parameters ###
        self.cross_sectional_area = 5.0  # m²
        self.compression_ratio = 5.0
        self.base_mass_flow_rate = 15.0  # kg/s
        self.efficiency_compressor = 0.9
        self.efficiency_turbine = 0.9
        self.rotor_inertia = 100
        self.rotor_static_drag = 50.0
        self.friction_loss = 0.05
        self.drag_coefficient = 0.4
        self.k_velocity_scale = 0.01

        ### Starter Motor ###
        self.starter_torque = 750.0  # Nm
        self.starter_max_rpm = 15000  # Max starter RPM

        ### Fuel Constants ###
        self.fuel_energy_density = 43.0e6  # J/kg
        self.ideal_air_fuel_ratio = 9.0  # Air-to-fuel ratio

        ### Engine State Variables ###
        self.starter_active = False
        self.throttle = 0.0
        self.EngineRpm = 0  # Initial RPM
        self.exhaust_temperature = self.atm_temperature
        self.chamber_pressure = 0.0 # Pascals
        self.air_mass_flow = 0.0 # m^3/s

    ### 🛠 **Starter Motor Model (Realistic Torque Curve)**
    def starter(self):
        if self.starter_active:
            torque = self.starter_torque * max(0, 1 - self.EngineRpm / self.starter_max_rpm)
            rpm_change = torque / self.rotor_inertia
            self.EngineRpm += rpm_change

    ### 🛠 **Compressor Model (Airflow Increases with RPM)**
    def compressor(self):
        """Simulate air compression and mass flow through the engine."""
        if self.EngineRpm < 50:  # Below 50 RPM, airflow is too weak
            self.air_mass_flow = 0
            return 0, self.atm_temperature

        T1 = self.atm_temperature

        T2 = T1 * (self.compression_ratio ** ((self.gamma - 1) / self.gamma))

        T2_real = T1 + (T2 - T1) / self.efficiency_compressor
        self.air_mass_flow = self.base_mass_flow_rate * (self.EngineRpm / 10000)  # Scaled mass flow
        return T2_real

    ### 🛠 **Combustor Model (Air-Fuel Mixing & Ignition)**
    def combustor(self): # modify to return "chamber_pressure" as the amount that compressed air as expanded to combustion
        """Burns fuel based on available air mass and throttle input."""
        if self.air_mass_flow <= 0:  # No air = no combustion
            return self.atm_pressure, self.exhaust_temperature

        fuel_mass_flow = self.air_mass_flow / (self.ideal_air_fuel_ratio / (self.throttle + 1e-6)) # prevent divison by 0
        heat_added = self.fuel_energy_density * fuel_mass_flow

        self.exhaust_temperature += heat_added / (self.air_mass_flow * self.air_specific_heat + 1e-6) # prevent divison by 0
        return self.atm_pressure, self.exhaust_temperature

    ### 🛠 **Turbine Model (Extract Power from Exhaust)**
    def turbine(self):
        """Extracts work from the exhaust stream to sustain rotation."""
        if self.air_mass_flow <= 0:
            return 0  # No work if there's no airflow

        P4 = self.atm_pressure
        T4_ideal = self.exhaust_temperature * (P4 / self.atm_pressure) ** ((self.gamma - 1) / self.gamma)
        T4_real = self.exhaust_temperature - (self.exhaust_temperature - T4_ideal) * self.efficiency_turbine

        power_extracted = self.air_mass_flow * self.air_specific_heat * (self.exhaust_temperature - T4_real)

        if self.EngineRpm > 50:
            turbine_torque = power_extracted / self.EngineRpm
        else:
            turbine_torque = 0

        return turbine_torque

    ### 🛠 **Nozzle Model (Thrust Calculation)**
    def nozzle(self): # thrust doesnt have ause at the moment
        """Calculate exhaust velocity."""
        if self.air_mass_flow <= 0:
            return 0  # No thrust without airflow

        exit_velocity = np.sqrt(2 * self.air_specific_heat * (self.exhaust_temperature - self.atm_temperature))
        thrust = self.air_mass_flow * exit_velocity
        return max(thrust, 0)

    ### 🛠 **Update Engine: Combine All Components**
    def update_engine(self, throttle):
        self.throttle = throttle

        # Engine cycle sequence:
        T2 = self.compressor()
        P3, T3 = self.combustor()
        turbine_torque = self.turbine()
        thrust = self.nozzle()

        # Compute net forces & RPM changes
        drag_force = 0.5 * self.drag_coefficient * self.air_density * self.cross_sectional_area * (self.EngineRpm * self.k_velocity_scale) ** 2
        rpm_change = ((turbine_torque - drag_force) / self.rotor_inertia) * 1.05  # Slight efficiency boost
        self.EngineRpm += rpm_change

        # Apply starter motor if active
        if self.starter_active:
            self.starter()

        # Prevent negative RPM
        self.EngineRpm = max(self.EngineRpm, 0)

        return thrust, drag_force, T2, T3

    def toggle_starter(self):
        """Turn the starter on/off"""
        self.starter_active = not self.starter_active
        print("Starter:", "ON" if self.starter_active else "OFF")


### **🔥 GUI: Throttle & Engine Feedback Display**
class SliderExample(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.brayton_engine = BraytonCycleEngine()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_engine)
        self.timer.start(30)

    def init_ui(self):
        self.setWindowTitle("Throttle Control")
        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)

        self.label_rpm = QLabel("RPM: 0", self)
        self.compression = QLabel("compression: 0 bar", self)
        self.combustion = QLabel("combustion: 0 +m^3", self)
        self.label_thrust = QLabel("Thrust: 0 N", self)
        self.label_drag = QLabel("Drag: 0 N", self)
        

        # Starter Button
        self.start_button = QPushButton("Start Engine")
        self.start_button.clicked.connect(self.toggle_starter)

        self.slider.valueChanged.connect(self.update_engine)

        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.label_rpm)
        layout.addWidget(self.compression)
        layout.addWidget(self.combustion)
        layout.addWidget(self.label_thrust)
        layout.addWidget(self.label_drag)
        layout.addWidget(self.start_button)  # Add button to UI
        self.setLayout(layout)
        self.show()

    def toggle_starter(self):
        """Toggle the starter motor when the button is clicked."""
        self.brayton_engine.toggle_starter()
        self.start_button.setText("Stop Starter" if self.brayton_engine.starter_active else "Start Engine")

    def update_engine(self):
        throttle = self.slider.value() / 100.0
        thrust, drag, compression, combustion = self.brayton_engine.update_engine(throttle)
        print(compression)
        print(combustion)
        self.label_rpm.setText(f"RPM: {self.brayton_engine.EngineRpm:.2f}")
        self.compression.setText(f"pressure: {compression:.2f} bar")
        self.combustion.setText(f"expantion: {combustion:.2f} +m^3")
        self.label_thrust.setText(f"Thrust: {thrust:.2f} N")
        self.label_drag.setText(f"Drag: {drag:.2f} N")


### **Run GUI**
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SliderExample()
    sys.exit(app.exec_())
