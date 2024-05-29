import yaml
import numpy as np
import h5py
import pandas as pd


def load_config(config_file_path):
    """Loads simulation parameters from a YAML configuration file."""
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_ultrasound_pulse(center_freq, bandwidth, pulse_duration, sampling_freq):
    """Generates a modulated Gaussian pulse for ultrasound simulation."""
    time_axis = np.arange(0, pulse_duration, 1/sampling_freq)
    carrier = np.sin(2 * np.pi * center_freq * time_axis)
    gaussian_envelope = np.exp(-(time_axis - pulse_duration/2)**2 / (2 * bandwidth**2))
    pulse = carrier * gaussian_envelope
    return pulse

def generate_noise(data_shape, noise_type, noise_level):
    """Generates additive noise for ultrasound simulation."""
    if noise_type == 'thermal':
        noise = np.random.randn(*data_shape) * noise_level 
    elif noise_type == 'electronic':
        noise = np.random.randn(*data_shape) * noise_level
    else:
        raise ValueError("Unsupported noise type. Use 'thermal' or 'electronic'.")
    return noise

def calculate_attenuation(distance, speed_of_sound, attenuation_coeff, frequency):
    """Calculates frequency-dependent attenuation factor."""
    attenuation_per_meter = attenuation_coeff * frequency  # Assuming n=1
    total_attenuation = np.exp(-attenuation_per_meter * distance * 2)
    return total_attenuation

def simulate_layered_reflections(pressure_wave): #Modify with your reflections model
    interface_depths = np.cumsum([layer['thickness'] for layer in layer_properties]) 
    current_depth = 0

    for i, interface in enumerate(interface_depths[:-1]): 
        distance_to_interface = interface - current_depth
        travel_time = 2 * distance_to_interface / layer_properties[i]['speed_of_sound']
        delay = int(travel_time * sampling_frequency)

        R = calculate_reflection_coefficient(layer_properties[i]['speed_of_sound'], layer_properties[i+1]['speed_of_sound'])
        pressure_wave[delay:] += R * pressure_wave[delay:]

        attenuation_factor = calculate_attenuation(distance_to_interface, 
                                                   layer_properties[i]['speed_of_sound'], 
                                                   layer_properties[i]['attenuation'], 
                                                   config['center_frequency'])

        pressure_wave *= attenuation_factor            
        current_depth = interface 

    return pressure_wave

def simulate_ultrasound(config):
    pulse = generate_ultrasound_pulse(config['center_frequency'], 
                                      config['bandwidth'], 
                                      config['pulse_duration'],
                                      config['sampling_frequency'])

    rf_data = np.zeros((config['num_sensors'], len(pulse))) 
    for depth, strength in zip(config['reflector_depths'], 
                               config['reflector_strengths']):
        reflected_wave = simulate_reflections(pulse, depth, config)
        rf_data += strength * reflected_wave

    if config['noise_type']:
        rf_data += generate_noise(rf_data.shape, config['noise_type'], 
                                  config['noise_level'])

    timestamps = np.arange(0, len(pulse)/config['sampling_frequency'], 1/config['sampling_frequency'])
    return rf_data, timestamps

def store_as_csv(rf_data, timestamps, sim_config):
    df = pd.DataFrame(rf_data.T)
    df.columns = ['Sensor ' + str(i + 1) for i in range(sim_config['num_sensors'])]
    df['Timestamp'] = timestamps
    df.to_csv('ultrasound_data.csv', index=False)

def store_as_hdf5(rf_data, timestamps, sim_config, simulation_index):
    """Stores RF data, timestamps, and metadata in an HDF5 file, 
       supporting multiple simulations.
    """
    with h5py.File('ultrasound_data.h5', 'a') as hf:  
        if 'rf_data' not in hf:
            hf.create_dataset('rf_data', data=rf_data, 
                              maxshape=(None, rf_data.shape[1], rf_data.shape[2]))
        else:
            hf['rf_data'].resize(hf['rf_data'].shape[0] + 1, axis=0)
            hf['rf_data'][simulation_index] = rf_data

        metadata_grp_name = f'metadata_{simulation_index}'
        if metadata_grp_name not in hf:
            metadata_group = hf.create_group(metadata_grp_name) 
            for key, value in sim_config.items():
                metadata_group.attrs[key] = value

# --- Simulation Configuration --- (example - replace with your YAML config)
layer_properties = [
    {'thickness': 0.02, 'speed_of_sound': 1540, 'attenuation': 0.5},  
    {'thickness': 0.05, 'speed_of_sound': 1580, 'attenuation': 0.8},  
    {'thickness': 0.08, 'speed_of_sound': 1570, 'attenuation': 0.6}  
]
config_file_path = 'simulation_config.yaml'

def main():
    # Load configuration from YAML file
    sim_config = load_config(config_file_path)

    sim_index = 0
    for _ in range(sim_config['num_simulations']):  
        rf_data, timestamps = simulate_ultrasound(sim_config)
        store_as_csv(rf_data, timestamps, sim_config)
        store_as_hdf5(rf_data, timestamps, sim_config, sim_index)
        sim_index += 1 

if __name__ == "__main__":
    main()
