#### INFO ###
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
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer




class BraytonCycleEngine:
    def __init__(self):
        ### system constants ###
        self.atm_pressure = 100000 # pascals
        self.atm_temperature = 273.15 # kelvin
        self.air_specific_heat = 1.005 # kJ/kg-K
        self.gas_constant = 287.05 # 
        self.air_density = 1.225 # 
        self.gamma = 1.4 #
       

        ### engine parameters ###
        self.cross_sectional_area = 5.0 # m**2
        self.compression_ratio = 2.0
        self.base_mass_flow_rate = 10.0 # kg/s
        self.efficiency_compressor = 0.85
        self.efficiency_turbine = 0.85
        self.rotor_inertia = 100 
        self.rotor_static_drag = 50.0
        self.friction_loss = 0.05
        self.drag_coefficient = 0.55
        self.starter_torque = 10.0 # Nm
        self.starter_max_rpm = 15000 # maximum starterRPM

        ### constant parameters ###
        self.exhaust_temperature = self.atm_temperature
        self.k_velocity_scale = 0.01
        self.heating_rate = 0.02
        self.cooling_rate = 0.03

        ### fuel constants ###
        self.fuel_energy_density = 43.0e6
        self.ideal_air_fuel_ratio = 9.0

        ### independent variables ###
        self.starter_active = False
        self.THROTTLE = 0.0

        ### dependent variables ###
        self.fuel_lambda = 1.0 # ideal afr on a 1+ biploar scale
        self.EngineRpm = 0 # RPM

    def starter(self):
        pass

    def compressor(self):
        pass
    
    def combustor(self):
        pass

    def turbine(self):
        pass

    def nozzle(self):
        pass
        
        


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

        self.label_throttle = QLabel(f"Throttle: {self.slider.value()}%", self)
        self.label_rpm = QLabel("RPM: 0", self)
        self.label_thrust = QLabel("Thrust: 0 N", self)
        self.label_drag = QLabel("Drag: 0 N", self)

        self.slider.valueChanged.connect(self.update_label)

        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.label_throttle)
        layout.addWidget(self.label_rpm)
        layout.addWidget(self.label_thrust)
        layout.addWidget(self.label_drag)
        self.setLayout(layout)
        self.show()

    def update_label(self):
        self.label_throttle.setText(f"Throttle: {self.slider.value()}%")

    def update_engine(self):
        throttle = self.slider.value() / 100.0
        self.brayton_engine.update_engine(throttle)
        self.label_rpm.setText(f"RPM: {self.brayton_engine.EngineRpm:.2f}")
        self.label_thrust.setText(f"Thrust: {self.brayton_engine.simulate(throttle)[0]:.2f} N")
        self.label_drag.setText(f"Drag: {self.brayton_engine.simulate(throttle)[1]:.2f} N")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SliderExample()
    sys.exit(app.exec_())
