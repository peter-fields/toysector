struct AllStatesBuffer{K}
    allstates::AbstractMatrix
    allstates_1hot::AbstractArray{T,3} where T <: Any
    probs::AbstractVector{K}
    energies::AbstractVector{K}
    δ::AbstractVector # container for delta function that defines states with inf energy
    
    function AllStatesBuffer( q::A, N::A, θtrue::Sector, FloatType ) where A <: Integer
        qᴺ = q^N
        log( qᴺ ) > 21. && (@warn "too many states to enumerate")
        allstates = get_all_states( q, N )
        allstates_1hot = Flux.onehotbatch( Array(allstates'), 1:q )
        probs = zeros(FloatType, qᴺ )
        energies = zeros(FloatType, qᴺ )
        δ = sectorenergy(allstates_1hot, θtrue ) .== θtrue.FloatType(Inf)
        allstates_1hot = CUDA.functional() ? cu(Float32.(allstates_1hot)) : allstates_1hot
        return new{FloatType}( allstates, allstates_1hot, probs, energies, δ )
    end
end

function f_pos_rate( θfit, all_states_buffer::AllStatesBuffer )
    (; allstates_1hot, probs, energies, δ) = all_states_buffer
    copyto!(energies, energy( allstates_1hot, θfit )  ) 
    probs .= exp.(-energies)
    probs ./= sum(probs)
    f_pos = sum( δ .* probs )
    H = - sum( probs .* log.(probs) )
    return f_pos, H
end

function f_pos_rate( θsec, θfit, sec_energy_buffer, M )
    sampler_opt = SamplerOption(
        M = M,
        traceevery = 100,
        showevery = 200,
        steps = 100_000,
        seed = 0, # 0 does not reseed rng after each MCMC function call
        method = :MetropolisHastings,
        gpu = CUDA.functional(),
        stopafter = 10,
        FloatType = Float32
       )
    _, samps_from_fit = drawsamples( θfit, 1.0, sampler_opt )
    samps_from_fit = samps_from_fit[:,29:35,:]
    Etrue = sectorenergy( samps_from_fit, θsec, sec_energy_buffer )
    f_pos = countmap(Etrue)[Inf32] ./ M
    return f_pos, nothing
end

function get_f_pos_rate( θsec, θfit, sec_energy_buffer, all_states_buffer, M )
    (; q,N) = θfit
    qᴺ = q^N
    if log(qᴺ) < 21.
        return f_pos_rate( θfit, all_states_buffer ) 
    else
        return f_pos_rate( θsec, θfit, sec_energy_buffer, M )
    end
end

function get_all_states( q, N )
    # q is number of amino acid type
    # N is length of sequnce
    mapreduce(permutedims, vcat,  [ digits(x, base=q, pad=N ) for x in collect(0:q^N-1) ] ) .+ 1
end

function H_toymodel(θtoy_pw::Pairwise, θtoy_sec::Sector; Npairs = 5, Nnoise = 18)
    # sector entropy
    (; S, q, Ξ, m, f) = θtoy_sec
    choos(x,S,q) = binomial(S,x)*((q-1)^x)
    c(x) = choos(x,S,q)
    mutes = collect(0:m)
    boltz = exp.(-f.(mutes))
    Zsec = Ξ * sum( c.(mutes) .* boltz ) 
    Hsec = sum((Ξ/Zsec) * c.(mutes) .*  boltz .* f.(mutes) ) + log(Zsec)
    # isolated coupling entropy
    Ziso(J₁₂) = sum(exp.(2*J₁₂)) #partition function of 2 coupled potts spins with h=0 for both
    Hiso(J₁₂) = begin Z = Ziso(J₁₂); -(1/Z)*sum( exp.(2J₁₂) .* (2J₁₂) ) + log(Z) end #entropy
    Hisos = Npairs * Hiso( θtoy_pw.J[:,1,:,2] )
    # noise entropy
    Hnoise = Nnoise * log( θtoy_pw.q )
    
    return Dict{Symbol, Any}( zip( 
            (:Hsec, :Hisos, :Hnoise, :Htotal), 
            (Hsec, Hisos, Hnoise, Hsec+Hisos+Hnoise)    ) 
            )
end

function get_H_fitted_model( θ, all_states_buffer )
    (; allstates_1hot, probs, energies ) = all_states_buffer
    energies .= energy( allstates_1hot, θ )
    
end