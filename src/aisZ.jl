"""Annealed Importance sampling. For use with SpinModels
Returns traces and estimate of log(Z) for given model"""

Base.@kwdef struct OptionAIS
    anneal_steps::Int = 100
    showevery::Int = 10
    chains::Int = 100
    gibbs::Bool = true # use gibbs sampling or MH. note: gibbs not reliable on gpu
    silent::Bool = false
    samplingDict::Dict{Symbol,Any} = Dict( :traceevery => 1, :max_sampling_steps => 100, :stopafter => 10 )
    #stopafter => stop sampling if Emean hasnt decreased for stopafter*tracevery steps at a given iteration
    gpu::Bool = CUDA.functional()
end                                                                   
    
function AISlogZ(θ::SpinModel, opt::OptionAIS, T=1)
    (; anneal_steps, showevery, chains, gibbs, silent, samplingDict, gpu ) = opt
    # init samples
    (; q,N ) = θ
    FloatType = eltype(θ)
    T=eltype(θ)(T)
    
    z = FloatType.(Flux.onehotbatch(rand(1:q, N, chains), 1:q))

    z = gpu ? CuArray(z) : z
    
    # init temps to anneal over
    βs = FloatType.( collect(0:anneal_steps-1)./anneal_steps )

    # init energy functions to use for sampling
    Efuncs = init_energies(z, θ, T)

    # init sampler
    sampler = init_sampler(z; gibbs=gibbs)
        t0 = time()

    draw!(i,z,Eᵦᵢ,sampler,traces)=AISdraw( i, z, Eᵦᵢ, sampler, traces, t0; samplingDict... )
    
    # function to calculate expectation value of
    Ẽ(X) = Efuncs[1](X) ./ anneal_steps 

    # initialize traces
    traces = ntuple(i -> FloatType[], 5) # Eᵦ₍ᵢ₋₁₎_mean, Eᵦ₍ᵢ₋₁₎_std, accrate, time, total_steps_per_anneal_iteration
    push!(traces[1],FloatType(Inf))
    Ƶ = zeros( FloatType , anneal_steps ) #container for ratio of  ( Zᵦᵢ/Zᵦᵢ₋₁ )

    for (i,βᵢ₋₁) in enumerate(βs)
        Eᵦ₍ᵢ₋₁₎ = get_sampler_energy_βᵢ(βᵢ₋₁, Efuncs... ; gibbs=gibbs)
        draw!(i,z,Eᵦ₍ᵢ₋₁₎,sampler,traces)
        Ƶ[i] = mean( exp.(-Ẽ(z)) )

        isone(i) && @printf "         i    Zᵦᵢ/Zᵦ₍ᵢ₋₁₎  mean(Eᵦ₍ᵢ₋₁₎)  std(Eᵦ₍ᵢ₋₁₎)    rate  time (s)  sample steps\n"
        isone(i) && @printf "----------  ------------  -------------  -----------   ------  --------  ------------\n"
        ((i%showevery == 0) && (silent==false)) && begin
            @printf(
            "%10d  %11.6e %14.5e  %11.4e  %6.4f  %7.2e  %12i\n", 
            i, Ƶ[i], traces[1][end], traces[2][end], traces[3][end], traces[4][end], traces[5][end]  )
        end   
    end
    
    # calculate average energy in order to get entropy estimate
    # do one more sampling step (for many more sampling steps) to get samples, z, at T=1
    Efinal = get_sampler_energy_βᵢ(FloatType(1), Efuncs... ; gibbs=gibbs)
    AISdraw(anneal_steps+1, z, Efinal, sampler, traces, t0;
        traceevery=10, max_sampling_steps=20_000, stopafter=100)
    
    traces = Dict{Symbol,Any}(zip( (:meanEᵦ₍ᵢ₋₁₎, :stdEᵦ₍ᵢ₋₁₎, :accrate, :time, :sample_steps_used), traces ))
    traces[:Ƶ] = Ƶ 
    popfirst!(traces[:meanEᵦ₍ᵢ₋₁₎])
    logZ₀ = sum(log.(sum(exp.(θ.h./T), dims=1)))
    logZ = sum(log.(Ƶ))+logZ₀
    meanE = mean( energy(z,θ)) # z are samples taken at the given temp T, at this point in the algorithm
    results = Dict{Symbol,Any}( 
        :logZ    => logZ ,
        :meanE   => meanE,
        :entropy => logZ + meanE/T,
        :T       => T
        )

    return traces, results, z
end

### helper functions ###

function init_energies(z, θ, T=1)
    (; q,N) = θ
    T = eltype(θ).(T)
    #init models
    θᵢₙₜ = copy(θ); copyto!(θᵢₙₜ.h, zeros(eltype(θ),q,N)); #interacting model
    θₕ = SpinModels.Profile(θ.h./T)
    #energy buffers
    energybufferᵢₙₜ = EnergyBuffer(z, θᵢₙₜ)
    energybufferₕ = EnergyBuffer(z, θₕ)
    #energy functions
    Eᵢₙₜ(X) = energy(X, θᵢₙₜ, energybufferᵢₙₜ)./T
    Eₕ(X) = energy(X, θₕ, energybufferₕ)
    ∇Eᵢₙₜ(X) =  ∂x_energy(X, θᵢₙₜ, energybufferᵢₙₜ)./T
    return ( Eᵢₙₜ, Eₕ, ∇Eᵢₙₜ, θₕ ) 
end

function get_sampler_energy_βᵢ(βᵢ, Eᵢₙₜ, Eₕ, ∇Eᵢₙₜ, θₕ ; gibbs=true)
    # Eᵢ(X) = rmul!(Eᵦ(X), βᵢ) + E₀(X)
    function Eᵢ(X) ; rmul!(Eᵢₙₜ(X), βᵢ) + Eₕ(X) end
    if gibbs
        function ∇Eᵢ(X); rmul!(∇Eᵢₙₜ(X), βᵢ) .- θₕ.h end
        function ∂Eᵢ(X); ( Eᵢ(X) , ∇Eᵢ(X) ) end
        return ∂Eᵢ
    else
        return Eᵢ
    end
end

function init_sampler(z; gibbs=true)
    gibbs ? ( return GibbsWithGradients(z) ) :  ( return MetropolisHastings(z) )
end
    

function AISdraw(anneal_iteration, z, Eᵢ, sampler, traces, t0;
        traceevery=1, max_sampling_steps=200, stopafter=10)
    
    counter = 0
    E_mean_min = traces[1][end]
    for k in 1:max_sampling_steps
        sampler(z, Eᵢ)
        
        k%traceevery == 0 && begin
            trace!(traces, z, Eᵢ, sampler, t0)
            #check if E_mean has decreased
            if (traces[1][end] < E_mean_min) && (traces[1][end] ≉ E_mean_min)
                E_mean_min = traces[1][end]
                counter=0
            else
                counter+=1
                if counter >= stopafter
                    push!(traces[5], k)
                    break
                end
            end
        end
        
        if k == (max_sampling_steps)
            push!(traces[5], k)
            @warn """max sampling steps reached for anneal step $anneal_iteration. 
            Z estimate may be inaccurate due to poor mixing of samples.
            Consider increasing max_sampling_steps or anneal_steps."""
        end                                 
    end
end

function trace!(traces, z, Eᵢ, sampler, t0)
    ith_energy = Eᵢ(z)
    ith_energy = typeof(ith_energy)<:Tuple ? ith_energy[1] : ith_energy
    Eᵢ_mean = mean(ith_energy)
    Eᵢ_std  = std(ith_energy)
    
    for (i,x) in enumerate( [Eᵢ_mean, Eᵢ_std, mean(acceptrate(sampler)), time()-t0] )
        push!(traces[i], x)
    end
end
