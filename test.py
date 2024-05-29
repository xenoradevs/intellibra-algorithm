import numpy as np
import h5py
import pandas as pd

# --- Sensor Setup ---
num_sensors = 10
radius = 0.03  
sampling_frequency = 40e6 

# --- ADC Settings ---
adc_resolution_bits = 12   
adc_voltage_range = (-2.5, 2.5) 

# --- Piezoelectric Sensor ---
sensor_sensitivity = 5e-12  # 5 pC/N
max_acoustic_pressure = 1e6  

# --- Tissue Model ---
layer_properties = [
    {'thickness': 0.02, 'speed_of_sound': 1540, 'attenuation': 0.5},  
    {'thickness': 0.05, 'speed_of_sound': 1580, 'attenuation': 0.8},  
    {'thickness': 0.08, 'speed_of_sound': 1570, 'attenuation': 0.6}  
] 

# --- Simulation --- 
num_samples = 8192
reflector_depths = [0.04, 0.08, 0.14]  
reflector_strengths = [0.8, 0.5, 0.3]  

#(spherical coordinates)
theta = np.linspace(0, 2 * np.pi, num_sensors, endpoint=False) 
phi = np.full_like(theta, 0.5 * np.pi)  # Assuming half-circle 
radius = 0.05 

def quantize(data, bits, vmin, vmax):
    resolution = 2**bits  
    step_size = (vmax - vmin) / resolution 
    quantized = np.round((data - vmin) / step_size).astype(int) 
    return quantized


def calculate_reflection_coefficient(z1, z2): 
    return (z2 - z1) / (z2 + z1)

def simulate_layered_reflections(pressure_wave):
    interface_depths = np.cumsum([layer['thickness'] for layer in layer_properties]) 
    current_depth = 0

    for i, interface in enumerate(interface_depths[:-1]):  # Change here
        distance_to_interface = interface - current_depth
        travel_time = 2 * distance_to_interface / layer_properties[i]['speed_of_sound']
        delay = int(travel_time * sampling_frequency)

        R = calculate_reflection_coefficient(layer_properties[i]['speed_of_sound'], layer_properties[i+1]['speed_of_sound'])
        pressure_wave[delay:] += R * pressure_wave[delay:]

        attenuation_factor = np.exp(-layer_properties[i]['attenuation'] * distance_to_interface * sampling_frequency * 1e-6) 
        pressure_wave *= attenuation_factor            
        current_depth = interface 

    return pressure_wave


# Simulate sensor readings
sensor_data = np.zeros((num_sensors, num_samples))  

interface_depths = np.cumsum([layer['thickness'] for layer in layer_properties]) 
for i, depth in enumerate(reflector_depths):
    pressure_wave = reflector_strengths[i] * max_acoustic_pressure * np.random.randn(num_samples)
    pressure_wave = simulate_layered_reflections(pressure_wave) 

    voltage_signal = sensor_sensitivity  * pressure_wave 

    # Calculate delay
    layer_index = next(x[0] for x in enumerate(interface_depths) if x[1] > depth) 
    travel_time = 2 * depth / layer_properties[layer_index]['speed_of_sound']
    delay = int(travel_time * sampling_frequency)

    if delay < num_samples:  # Check if delay is within bounds
        voltage_signal[delay:] += voltage_signal[delay]  
        sensor_data[:, delay:] += quantize(voltage_signal[delay:], adc_resolution_bits, *adc_voltage_range) 
    else:
        print(f"Warning: Echo from reflector at {depth} arrives beyond signal length - Skipping")



    voltage_signal[delay:] += voltage_signal[delay]  
    sensor_data[:, delay:] += quantize(voltage_signal[delay:], adc_resolution_bits, *adc_voltage_range) 

    


# --- Metadata ---
metadata = {
    'sampling_frequency_hz': sampling_frequency,
    'speed_of_sound_mps': 1540, 
    'sensor_array_type': 'uniform_circular', 
    'num_sensors': num_sensors,
}

# --- HDF5 Creation ---
with h5py.File('ultrasound_data.h5', 'w') as hf:
    hf.create_dataset('sensor_data', data=sensor_data) 

    # Combine sensor positions
    sensor_positions = np.vstack((theta, phi, radius)).T 
    hf.create_dataset('sensor_positions', data=sensor_positions)

    hf.create_dataset('timestamps', data=np.arange(num_samples) / sampling_frequency)

    for key, value in metadata.items():
        hf.attrs[key] = value

# --- CSV Creation --- 
df = pd.DataFrame(sensor_data.T)  
timestamps = np.arange(num_samples) / sampling_frequency
df['timestamp'] = timestamps 
df.to_csv('ultrasound_data.csv', index=False, header=False)




















# ... (Your existing code)

# ... Your code to calculate sensor positions 
# Example placeholder (spherical coordinates) - replace this!
