# __precompile__()

module PrivateBoostedDensities

using Flux, Distributions, Mamba, Cubature, ProgressMeter
using Base.Iterators: repeated, partition
using Flux: onehotbatch, argmax, crossentropy, throttle
using Flux.Tracker

for f in ("qdensity", 
          "sampling", 
          "gaussian_mixture", 
          "metrics", 
          "random_gaussians",
          "run_privacy_experiment")
    include("$f.jl")
end

export QDensity, GaussianMixture, RandomGaussian,
    push!, normalise!, grad, logpdf_gradlogpdf, 
    kl, integrate_pdf, coverage, expected_log_likelihood,
    allocate_train_valid, initialise,
    mu, boosted_alpha, mean_boosted_alpha, run_privacy_experiment
end