#! /usr/bin/bash
N=240
~/.julia/v0.6/Pope/scripts/noise_analysis.jl 20181205_A --maxchannels=$N --replaceoutput
~/.julia/v0.6/Pope/scripts/basis_create.jl 20181205_B 20181205_A/20181205_A_noise.hdf5 --maxchannels=$N --n_basis=4 --replaceoutput
~/.julia/v0.6/Pope/scripts/ljh2off.jl 20181205_B/20181205_B_model.hdf5 20181205_ --endings=B C
~/.julia/v0.6/Pope/scripts/basis_plots.jl 20181205_B/20181205_B_model.hdf5
~/.julia/v0.6/Pope/scripts/noise_plots.jl 20181205_A/20181205_A_noise.hdf5