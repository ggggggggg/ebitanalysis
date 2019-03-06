#! /usr/bin/bash
N=240
~/.julia/v0.6/Pope/scripts/noise_analysis.jl 20181204_F --maxchannels=$N --replaceoutput --dontcrash
~/.julia/v0.6/Pope/scripts/basis_create.jl 20181204_H 20181204_F/20181204_F_noise.hdf5 --maxchannels=$N --n_basis=5 --replaceoutput
~/.julia/v0.6/Pope/scripts/ljh2off.jl 20181204_H/20181204_H_model.hdf5 20181204_ --endings=D E H I J K L
~/.julia/v0.6/Pope/scripts/basis_plots.jl 20181204_H/20181204_H_model.hdf5
~/.julia/v0.6/Pope/scripts/noise_plots.jl 20181204_F/20181204_F_noise.hdf5
