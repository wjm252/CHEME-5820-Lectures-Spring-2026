# setup paths -
const _ROOT = pwd();
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_DATA = joinpath(_ROOT, "data");

# check do we have a Manifest.toml file?
using Pkg;
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.add(path="https://github.com/varnerlab/VLDataScienceMachineLearningPackage.jl.git")
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages
using VLDataScienceMachineLearningPackage
using CSV
using DataFrames
using FileIO
using StatsPlots
using Plots
using Colors
using Statistics
using LinearAlgebra
using Distances
using JLD2
using PrettyTables
using Clustering
using ProgressMeter

# load my codes -
include(joinpath(_PATH_TO_SRC, "Compute.jl"));