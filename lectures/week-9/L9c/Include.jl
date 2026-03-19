# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_IMAGES = joinpath(_ROOT, "images");

# load external packages -
using Pkg
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.add(path="https://github.com/varnerlab/VLDataScienceMachineLearningPackage.jl.git")
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# using statements -
using VLDataScienceMachineLearningPackage
using Statistics
using JLD2
using LinearAlgebra
using Plots
using Distances
using NNlib
using Distributions
using PrettyTables
using DataFrames
using StatsBase
using IJulia

# load source files -
include(joinpath(_PATH_TO_SRC, "CBOW.jl"));
include(joinpath(_PATH_TO_SRC, "SkipGram.jl"));
include(joinpath(_PATH_TO_SRC, "NegativeSampling.jl"));
