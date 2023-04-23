import os
import numpy as np
import pandas as pd
import json

from distutils.dir_util import copy_tree
from utils import njit, copy_filter_hp_to_other_clock_frequency
from utils import timing
from pathlib import Path
from snn.graphs import plot_network
from matplotlib import pyplot as plt
from snn.resonator import create_sine_wave, create_excitatory_inhibitory_resonator
import gc


def test_frequency(network, test_size=10_000_000, start_freq=0, step=1 / 200000, clk_freq=1536000):
    batch_size = 50_000
    shift = 0
    while test_size > 0:
        sine_size = min(batch_size, test_size)
        sine_wave, freqs = create_sine_wave(sine_size, clk_freq, start_freq, step, shift)

        network.input_full_data(sine_wave)

        shift = freqs[-1]
        start_freq += sine_size * step
        test_size -= sine_size

def custom_resonator_output_spikes(
        freq0,
        clk_freq=int(1.536 * (10 ** 6)) * 2,
        step=1/12_000,
        save_figure=False,
        spectrum=None,
        path=None,
        plot=True
):
    my_resonator = create_excitatory_inhibitory_resonator(freq0=freq0, clk_freq=clk_freq)
    log_neuron_potentials = []
    for i in log_neuron_potentials:
        my_resonator.log_membrane_potential(i)
    my_resonator.log_out_spikes(-1)
    start_freq = 0
    spectrum = spectrum or 2 * freq0
    test_size = int(spectrum / step)
    spikes_neuron = my_resonator.neurons[-1]

    spikes_neuron.membrane_sample_max_window = np.zeros(1).astype('float32')
    test_frequency(
        my_resonator,
        start_freq=start_freq,
        step=step,
        test_size=test_size,
        clk_freq=clk_freq
   )
    #freq0 = int(freq0)

    for i in log_neuron_potentials:
        membrane_neuron = my_resonator.neurons[i]
        y_membrane = membrane_neuron.membrane_potential_graph()
        x = np.linspace(start_freq, start_freq + spectrum, len(y_membrane))
        plt.title(f'membrane potential f={freq0}, neuron={i}')
        plt.plot(x, y_membrane)
        plt.show()

    y_spikes = spikes_neuron.out_spikes[:spikes_neuron.index]

    if path is not None:
        print(os.getcwd())
        np.savez_compressed(path, spikes=y_spikes)

    spikes_window_size = 5000
    y_spikes = np.convolve(y_spikes, np.ones(spikes_window_size, dtype=int), 'valid')
    x = np.linspace(start_freq, start_freq + spectrum, len(y_spikes))
    plt.title(f'spikes in window of {spikes_window_size} freq: {freq0}')
    plt.plot(x, y_spikes)
    print("freq0=",freq0)
    if save_figure:
        plt.savefig(f'../filters/clk_{clk_freq}/figures/f_{freq0}.PNG', bbox_inches='tight')
        plt.close()
    elif plot:
        plt.show()

"""
clk_filters = {
    153600: (60,65),
    #696172: (30, 63),
    #331510: (14, 30),
    #154705: (8, 14),
    #88402: (4, 8),
    #16830: (0, 4),
}
for clk_freq, (lower_f, higher_f) in clk_filters.items():
    for fname in os.listdir(f'../filters/clk_{clk_freq}/parameters'):
        f0 = int(fname[2:-5])
        spikes_file = '../spikes_output/{}.npz'.format(int(f0))
        if Path(spikes_file).is_file():
            continue
        if not lower_f <= f0 <= higher_f:
            continue
        custom_resonator_output_spikes(f0, clk_freq, 1/20_000, save_figure=False,spectrum=40, path='./spikes_output/{}.npz'.format(int(f0)), plot=False)
"""

def generate_figures(clk_freq, step):
    if not os.path.isdir(f'../filters/clk_{clk_freq}/figures'):
        os.makedirs(f'../filters/clk_{clk_freq}/figures')
    already_exist = os.listdir(f'../filters/clk_{clk_freq}/figures')
    already_exist = [f'{int(f[2:-4]):.3f}' for f in already_exist]
    for fname in os.listdir(f'../filters/clk_{clk_freq}/parameters'):
        if fname[2:-5] in already_exist:
            continue
        f0 = float(fname[2:-5])
        print(f0)
        custom_resonator_output_spikes(
            f0,
            clk_freq=clk_freq,
            step=1/step,
            save_figure=True
        )
        gc.collect()

def copy_filter_hp_to_other_clock_frequency(clk_old, clk_new):
    copy_tree(f"../filters/clk_{clk_old}/parameters", f"../filters/clk_{clk_new}/parameters")
    clk_new_dirname = f'../filters/clk_{clk_new}/parameters'
    scale_factor = clk_new / clk_old
    for fname in os.listdir(clk_new_dirname):
        new_filter = int(fname[2:].split('.')[0]) * scale_factor
        new_filter_file_name = f'{clk_new_dirname}/f_{new_filter:.3f}.json'
        os.rename(f'{clk_new_dirname}/{fname}', new_filter_file_name)

        with open(new_filter_file_name, 'r') as f:
            filter_parameters = json.load(f)
            filter_parameters['f0'] *= scale_factor
            filter_parameters['f_resonator'] *= scale_factor

        with open(new_filter_file_name, 'w') as f:
            json.dump(filter_parameters, f)

#copy_filter_hp_to_other_clock_frequency(1536000, 15360)
generate_figures(15360, 2_000_000)
