function mu(samps, m::Flux.Chain)
    mu = m(samps) |> Flux.Tracker.data |> vec
    mx = maximum(abs, mu)
    return mean(mu)/mx
end


epochs(inputs, n::Int) = (d for _ in Base.OneTo(n) for d in inputs)

function batches(inputs, batch_size::Int)
    last_dim(arr, ind) = view(arr, ntuple((_)->(:), ndims(arr) - 1)..., ind)

    n = last(size(first(inputs)))
    return (map(x->last_dim(x, batch), inputs) for batch in Iterators.partition(Base.OneTo(n), batch_size))
end

@views function allocate_train_valid(dim, num_p, num_q; train_fraction = 3/4)
    p_samps = Array{Float32, length(dim) + 1}(dim..., num_p)
    q_samps = Array{Float32, length(dim) + 1}(dim..., num_q)

    train_p_inputs = p_samps[:, Base.OneTo(Int(num_p * train_fraction))]
    train_q_inputs = q_samps[:, Base.OneTo(Int(num_q * train_fraction))]

    test_p_inputs  = p_samps[:, (Int(num_p * train_fraction) + 1):end]
    test_q_inputs  = q_samps[:, (Int(num_q * train_fraction) + 1):end]

    return p_samps, q_samps, train_p_inputs, train_q_inputs, test_p_inputs, test_q_inputs
end

function allocate_train_valid(_p_samps::Array{T,N}, _q_samps::Array{T,N}; train_fraction = 3/4) where {T,N}
    num_p, num_q = map(last, (size(p_samps), size(q_samps)))
    p_samps, q_samps, train_p_inputs, train_q_inputs, test_p_inputs, test_q_inputs = allocate_train_valid(N-1, num_p, num_q, train_fraction = train_fraction)
    p_samps[:] = _p_samps
    q_samps[:] = _q_samps

    return p_samps, q_samps, train_p_inputs, train_q_inputs, test_p_inputs, test_q_inputs
end

function run_privacy_experiment(p ::Distribution,            # Target distribution 
                        q0::Distribution,                    # Initial distribution
                        num_p::Int = 10_000,
                        num_q::Int = 10_000;
                        batch_size = div(num_p + num_q, 4),
                        iter::Int  = 5,
                        early_stop::Real    = 0.1,            # when test error is `early_stop` below train error, stop early
                        weak_boost::Real    = Inf,            # when we have achieved a weak learner, stop early (set this to a small positive number to enable)
                        verbose::Bool              = false,
                        run_boosting_metrics::Bool = false,   # time consuming to have these on since we will renormalise Q every iteration
                        optimiser::Function = p -> Nesterov(p, 0.01),
                        sampler::Function   = rand,
                        num_epochs::Int     = 10,
                        eps_ = 0.001,
                        seed::Int           = 1337,
                        model::Flux.Chain   = Chain(Dense(2, 20, softplus), Dense(20, 20, softplus),Dense(20, 20, softplus),  Dense(20, 2), softmax))

    # Set random seed
    srand(seed)

    # Initialize the density
    q = QDensity(q0)

    # Start validation by splitting the dataset
    p_samps, q_samps, train_p_samps, train_q_samps, test_p_samps, test_q_samps =
        allocate_train_valid(1, num_p, num_q; train_fraction = 3/4)

    # Sample from the target
    rand!(p, p_samps)

    # Initailize variables that will keep document the training track
    train_history    = Vector{Any}[]
    boosting_history = Dict{Function,Float64}[]
    boosting_metrics = (coverage, expected_log_likelihood, kl)
    if run_boosting_metrics
        push!(boosting_history, Dict(met => met(p, q, q_samps, p_samps) for met in boosting_metrics))
    end

    # Begin the Boosting process
    for i in Base.OneTo(iter)
        if verbose
            info("Sampling Q")
        end

        # Sample from Q_i
        q_samps[:] = sampler(q, num_q)

        # Initialize classifier c_i
        m          = deepcopy(model)

        # Set up the optimiser and objective function (cross entropy loss) for classifier c_i
        opt        = optimiser(Flux.params(m))
        obj(p_samps, q_samps) = -mean(log.(m(p_samps))) - mean(log1p.(-m(q_samps)))

        if verbose
            train_progress = Progress(num_epochs, 1, "Training classifier ($i of $iter): ")
        end

        # Begin training
        evalcb() = begin

            # Compute the training and testing accuracy based on cross-entropy loss
            train_ce = (obj(train_p_samps, train_q_samps) |> Flux.Tracker.data)[]
            test_ce  = (obj(test_p_samps,  test_q_samps)  |> Flux.Tracker.data)[]

            if verbose
                ProgressMeter.next!(train_progress, showvalues = [("cross entropy (train)", train_ce),
                                                                  ("cross entropy (test)",  test_ce)])
            end

            # Evaluate Boosting metric so that we may check the weak learning assumption for early stopping
            mu_p =  mu(p_samps,  m[1:end-1])
            mu_q = -mu(q_samps,  m[1:end-1])

            push!(train_history, [i, test_ce, mu_p, mu_q])

            if (train_ce + early_stop <= test_ce) || (min(mu_p, mu_q) >= weak_boost)
                println()
                info("Stopping early")
                return :stop
            end
        end

        # Compute theta parameter
        θ = (eps_ / (4*log(2) + eps_)) ^ i

        # Train the classifier c_i
        Flux.train!(obj, epochs(batches((train_p_samps, train_q_samps), batch_size), num_epochs), opt, cb = Flux.throttle(evalcb, 2))

        # Boost c_i onto q_i
        push!(q, (m[1:end-1], θ))

        if run_boosting_metrics
            if verbose
                info("Running the boosting metrics")
            end
            push!(boosting_history, Dict(met => met(p, q, q_samps, p_samps) for met in boosting_metrics))
        end
    end

    return q, (train_history, boosting_history)
end
