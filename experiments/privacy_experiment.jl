using PrivateBoostedDensities, Distributions, Flux

using Plots; gr(size=(800, 600))

# For optional plotting purposes
include( "./utilities.jl")

# Set up some kind of target distribution
p = MvNormal(1,1)

# Set up the initial distribution
q0 = MvNormal(1,1)

# Pick an architechture for the classifier
model = Chain(Dense(1, 25, tanh), Dense(25,25, tanh),  Dense(25,25, tanh),  Dense(25, 1), x->σ.(x))

# Set the number of samples to be used at each iteration
num_p, num_q = 10_000, 10_000

# Set the privacy parameter
eps = 1

# Run the model, storing the final density and relevant metrics
q, train_history = run_privacy_experiment(p, q0, num_p, num_q, iter = 2, num_epochs = 5, model = model, verbose = true, run_boosting_metrics = true, optimiser = p -> Flux.ADAM(p), eps_=eps ,seed = 1337 + 2)
