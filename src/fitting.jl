function reg(A, λ, L)
    iszero(λ) && return zero(eltype(A))
    L==1 && return λ * sum(abs, A)
    L==2 && return λ * sum(abs2, A)
    error("L must be 1 or 2. Get $L")
end
function reg!(Ā, A, λ, L)
    iszero(λ) && return nothing
    L==1 && return @.(Ā += λ * sign(A))
    L==2 && return @.(Ā += 2λ * A)
    error("L must be 1 or 2. Get $L")
end
function reg(θ::SpinModel, regs::NamedTuple=(;))
    regparams = Iterators.filter(key -> hasproperty(θ,key), keys(regs)) 
    return sum(reg(getproperty(θ, s), regs[s]...) for s ∈ regparams)
end
function reg!(θ̄::T, θ::T, regs::NamedTuple=(;)) where T<:SpinModel
    for s ∈ Iterators.filter(key -> hasproperty(θ,key), keys(regs))
        reg!(getproperty(θ̄, s), getproperty(θ, s), regs[s]...)
    end
end


function ratiomatch_reg(z, θ::T, θ̄::T, buffer; gradient=true, regs::NamedTuple=(;)) where T<:SpinModel
    L = ratiomatch(z, θ, θ̄, buffer, gradient)
    gradient && reg!(θ̄, θ, regs)
    return L + reg(θ, regs)
end
ratiomatch_reg(z, θ::SpinModel, buffer; regs=(;)) = ratiomatch(z, θ, buffer) + reg(θ, regs)


function Flux.update!(opt::Flux.Optimise.AbstractOptimiser, θ::T, θ̄::T) where T<:SpinModel
    for s ∈ paramnames(θ)
        Flux.update!(opt, getproperty(θ,s), getproperty(θ̄,s))
    end
end

@with_kw struct FitOption{R}
    q::Int
    N::Int
    M::Int
    P::Int=0
    reg_J::Tuple{R,Int} = (1f-3, 2)
    reg_h::Tuple{R,Int} = reg_J
    reg_W::Tuple{R,Int} = (1f-3, 2)
    reg_b::Tuple{R,Int} = reg_W
    gpu::Bool = CUDA.functional()   # use cuda (if available)
    batchsize::Int = 256            # minibatch for sgd
    epochs::Int = 1000              # sgd epochs
    seed::Int = 0                   # random number seed
    showevery::Int = 50          
    progTol::R = 10^-6              # rel change in obejctive function stopping condition
end
FitOption(X; kws...) = FitOption{eltype(X)}(;q=size(X,1), N=size(X,2), M=size(X,3), kws...)

function savename(x::FitOption)
    n  = "P=$(x.P)"
    n *= @sprintf "_lambdaJ=1e%.1f.L%d" log10(x.reg_J[1]) x.reg_J[2]
    if x.P > 0
        n *= @sprintf "_lambdaW=%.1f.L%d" log10(x.reg_W[1]) x.reg_W[2]
    end
    return n
end

function savename(x::Dict{Symbol,Any})
    λh, Lh = x[:reg_h]
    if iszero(λh)
        n = @sprintf "regh=0"
    else
        n = @sprintf "L%d_regh=1e%.1f" Lh log10(λh)
    end
    if haskey(x, :P)
        P = x[:P]
        n *= @sprintf "_P=%i" P
    end
    LJ=0
    if haskey(x, :reg_J)
        λJ, LJ = x[:reg_J]
        if iszero(λJ)
            n *= @sprintf "_regJ=0"
        elseif LJ == Lh
            n *= @sprintf "_regJ=1e%.1f" log10(λJ)
        else
            n *= @sprintf "_L%d_regJ=1e%.1f" LJ log10(λJ)
        end
    end
    if haskey(x, :reg_W)
        λW, LW = x[:reg_W]
        if iszero(λW)
            n *= @sprintf "_regW=0"
        elseif  LW == LJ
            n *= @sprintf "_regW=1e%.1f" log10(λW)
        elseif (LW == Lh & LJ == 0)
            n *= @sprintf "_regW=1e%.1f" log10(λW)
        else
            n *= @sprintf "_L%d_regW=1e%.1f" LW log10(λW)
        end
    end
    return n
end

function pull_from_sweep( sweep_dicts, 
        requested::Vector{Symbol}; kwargs...)
    # takes the sweep_dict with all specified params and fitted results and returns a N-column-matrix, 
    # return_request, where return_request[:,i] are values of requested[i]
    # kwargs are fixed values of other params that constrain which sweep_dict you pull from
    # the tuples containing reg values have to be of type Float32
    for (key, value) = kwargs
        if typeof(value) <: Tuple  && typeof(value[1]) != Float32 
            println("error: regularization value must be of Float32")
            return
        end
    end
    # initialize return_request matrix
    n_req = length(requested);
    return_request = Matrix{Any}(nothing, 0, n_req)
    # main loop for pulling from sweep
    for d in sweep_dicts
        if all( [ d[key] == value for (key, value) in kwargs] )
            return_request = [return_request ; zeros(1, n_req)]
            [ return_request[end,i] = d[ requested[i] ] for i in 1:n_req ]   
        end
    end 
    return return_request              
end


sgdfit!(θ, X; kws...) = sgdfit!( θ, X, FitOption(X;kws...) )
function sgdfit!(θ::SpinModel, X::AbstractArray{<:AbstractFloat,3}, args::FitOption)
    (; q, N, M, reg_J, reg_h, reg_W, reg_b, batchsize, epochs, showevery, progTol) = args
    
    # for keeping gradients
    θ̄ = initθ(θ)

    # training loss
    l, b = Float32[], RatioMatchBuffer(X, θ)
    l!() = push!(l, ratiomatch(X, θ, b))

    # minibatch loss & gradient
    rs     = (J = reg_J, h = reg_h, W = reg_W, b = reg_b)  # regularization
    buffer = RatioMatchBuffer(θ, batchsize)
    ℓ(x)   = ratiomatch_reg(x, θ, θ̄, buffer; regs = rs)

    # trace regularization penalty
    r = Float32[]
    r!() = push!(r, reg(θ, rs))

    @printf "------------------ Begin SGD ------------------\n"
    x_loader = Flux.DataLoader(X; batchsize = batchsize, shuffle = true, partial = false)
    opt = ADAM()
    t0 = time()
    exitflag=""
    for i ∈ 0:epochs
        i > 0 && for x ∈ x_loader
            ℓ(x); Flux.update!(opt, θ, θ̄)
        end
        # trace losses and regularization penalty
        l!(); r!();
        iszero(i) && @printf "epoch  reg. loss   train loss  time\n"
        iszero(i) && @printf "-----  ----------  ----------  ----\n"
        aug_loss_current = l[end]+r[end]
        if iszero(i % showevery)
            @printf("%5d  %10.6f  %10.6f  %4.0f\n",
                i, aug_loss_current, l[end], time()-t0
            )
        end
        
        # stopping conditions
        i > 0 && begin
            aug_loss_last = l[end-1]+r[end-1]
            # rel_change = abs(aug_loss_last - aug_loss_current) / aug_loss_last
            # rel_change <= progTol && (exitflag = "rel change in aug loss below progTol $(progTol)"; break)
            rel_change = abs(l[end-1] - l[end]) / l[end-1]
            rel_change <= progTol && (exitflag = "rel change in bare loss below progTol $(progTol)"; break)
        end
        i == epochs && ( exitflag="maxiter reached" ; break ) 
        
    end
    traces = Dict{Symbol,Any}(zip(
        (:epochs, :time_run, :f_training, :regularization, :exitflag),
        ( epochs, time()-t0,           l,               r,  exitflag)
    ))
    @printf "%s\n" exitflag
    @printf "------------------- End SGD -------------------\n"
    return traces
end

"""
    krandperm(N, k)
Partition `randperm(N)` into `k` parts. 
The last part is bigger than the others if `N%k ≠ 0`.
"""
function krandperm(N::Integer, k::Integer)
    p, n = randperm(N), N ÷ k
    ntuple(i -> (a = (i-1)*n; i==k ? p[a+1:end] : p[a .+ (1:n)]), k)
end
"""
    kfold(X::AbstractArray, k)
Partition `X` along the last dimensions into `k` parts. 
The last part is bigger than the others if `N%k ≠ 0`.
"""
function kfold(X::AbstractArray, k)
    inds = krandperm(size(X, ndims(X)), k)
    cols = ntuple(i -> Colon(), ndims(X)-1)
    ntuple(i -> X[cols..., inds[i]], length(inds))
end


# kfold_sgdfit!(θ, 𝕏; kws...) = kfold_sgdfit!(θ, 𝕏, FitOption{eltype(𝕏[1])}(; kws...))
function kfold_sgdfit!(θ::SpinModel, 𝕏::NTuple; kws...)
    Θ = ntuple(i -> initθ(θ), length(𝕏) )      # models at last epoch
    𝕋 = ntuple(i -> Dict{Symbol,Any}(), length(𝕏)) # traces
    𝕍 = zeros(eltype(θ), length(𝕏)) # validation losses of trained model
    for (n, X̂, θ, traces) in zip(eachindex(𝕏), 𝕏, Θ, 𝕋)
        @printf "k-fold: %d of %d\n" n length(𝕏)
        X = cat(𝕏[filter(≠(n), eachindex(𝕏))]..., dims = 3) # training set
        merge!(traces, sgdfit!(θ, X; kws...))
        𝕍[n] = ratiomatch(X̂, θ)
    end 
    return Θ, 𝕋, 𝕍
end


function initθ(mdl::Symbol, q,N,P=0)
    θ = nothing
    h = similar(Float32[], q, N)
    J = similar(Float32[],q,N,q,N)
    SpinModels.random_h!(h, 0.1)
    SpinModels.random_J!(J, 0.1)
    if mdl == :Pairwise
        h,J = CUDA.functional() ? ( cu(h), cu(J) ) : ( h,J )
        θ = Pairwise(h,J)  
    elseif mdl == :RBM
        W = similar(Float32[], P,q,N)
        b = similar(Float32[], P)
        SpinModels.random_W!(W, 0.1)
        SpinModels.random_b!(b, 0.1)
        h,W,b = CUDA.functional() ? ( cu(h), cu(W), cu(b) ) : ( h,W,b )
        θ = RBM(h,W,b)
    elseif mdl == :SRBM
        W = similar(Float32[], P,q,N)
        b = similar(Float32[], P)
        SpinModels.random_W!(W, 0.1)
        SpinModels.random_b!(b, 0.1)
        h,J,W,b = CUDA.functional() ? ( cu(h),cu(J),cu(W),cu(b) ) : ( h,J,W,b )
        θ = SRBM( h,J,W,b )
    end
    return θ
end

function initθ(θ::SpinModel)
    (; q,N) = θ
    h = similar( Float32[], q, N )
    J = similar( Float32[],q,N,q,N )
    SpinModels.random_h!(h, 0.1)
    SpinModels.random_J!(J, 0.1)
    if typeof(θ)<:Pairwise
        h,J = CUDA.functional() ? ( cu(h), cu(J) ) : ( h,J )
        θpw = Pairwise(h,J)
        return θpw
    elseif typeof(θ)<:RBM
        (; P) = θ
        W = similar(Float32[], P,q,N)
        b = similar(Float32[], P)
        SpinModels.random_W!(W, 0.1)
        SpinModels.random_b!(b, 0.1)
        h,W,b = CUDA.functional() ? ( cu(h), cu(W), cu(b)  ) : ( h,W,b )
        θrbm = RBM(h,W,b)
        return θrbm
    elseif typeof(θ)<:SRBM
        (; P) = θ
        W = similar(Float32[], P,q,N)
        b = similar(Float32[], P)
        SpinModels.random_W!(W, 0.1)
        SpinModels.random_b!(b, 0.1)
        h,J,W,b = CUDA.functional() ? ( cu(h),cu(J),cu(W),cu(b) ) : ( h,J,W,b )
        θsrbm = SRBM( h,J,W,b )
        return θsrbm 
    end
end
