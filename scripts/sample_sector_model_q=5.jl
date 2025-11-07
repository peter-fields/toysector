"""
Make toy model with 5 isolated pairwise interactions, a synergistic sector of size S=7, with number of patterns Ξ=5, and max number of mutations away from ideal pattern m=3, with q=5 amino acid types. 
Tsec controls spacing of energy levels of mutations from ideal pattern as 1/Tsec. 
"""

using DrWatson
@quickactivate "proteins"
using SpinModels, Optim, Parameters, CUDA, LinearAlgebra, Statistics, Printf, Flux
using BSON, Distributions, Random 
include(srcdir("toysector.jl"))
include(srcdir("samplers.jl"))
import StatsBase: Weights, sample, StatsBase

function sample_sector_model( θ::Sector, M )
    # M is number of samples
    (; q, S, FloatType) = θ
    function scramble_patterns( z, q,M,S,θ,FloatType ) 
        function scramble_pattern( pos_to_mutate, samples::AbstractArray{T,3}, idx, q ) where T
            p = pos_to_mutate
            q_to_flip_from = findall( samples[:,p,idx].==1 )[1][1]
            samples[:,p,idx] .= T.(Flux.onehotbatch( rand( deleteat!( collect(1:q), q_to_flip_from ) , 
                length(p) ), 1:q ))
        end
        boltsec(x)=binomial(S,x)*((q-1)^x)*exp(-θ.f(x))
        probs = boltsec.( collect(0:θ.m ) ) 
        probs ./= sum(probs)
        # @show probs
        n_mutations_per_sample = StatsBase.sample(0:θ.m, Weights(probs), M)
        pos_to_mutate = [ StatsBase.sample( 1:θ.S, m , replace=false) for m in n_mutations_per_sample];
        for (i,p) in enumerate(pos_to_mutate)
            length(p) == 0 && continue
            scramble_pattern( p, z, i, q )
        end
        return n_mutations_per_sample
    end

    z₀ = FloatType.( (!).(permutedims(θ.patterns, (2,3,1) )) )
    z = z₀[ : , : , rand(1:θ.Ξ, M) ]
    n_mutations_per_sample = scramble_patterns( z, q,M,S,θ,FloatType ) 
    return z 
end
function sample_N_iso_pairs( rawJ, q, M, Npairs )
        function sample_iso_pw( rawJ, q, M )
        # rawJ is coupling strength when AA1 is same as AA2
        # q is num of amino acids
        # M is number samples
        Jmat = diagm(fill(rawJ, q))
        Z = sum(exp.(Jmat) )
        probs = exp.(Jmat) ./ Z
        cidxs = CartesianIndices(probs)
        z = zeros(Int,2,M)
        sss   = StatsBase.sample(1:length(probs),  Weights(probs[:]), M )
        for (i,x) in enumerate(cidxs[sss] )
            z[1,i] = x[1]
            z[2,i] = x[2]
        end
        return z
    end
    samps = zeros(Int, 2*Npairs, M)
    # @show size(samps)
    for k in 1:2:(2*Npairs)
        samps[ k:(k+1), : ] .= sample_iso_pw( rawJ, q, M )
    end
    return Flux.onehotbatch( samps, 1:q )
end
function sample_noise( q, M, Nnoise )
    Flux.onehotbatch( rand(1:q, (Nnoise,M)), 1:q )
end



function sample_toy_model( M, rawJ, θsec ; Npairs=5, Nnoise=18 )
    (; q ) = θsec
    z_sec = sample_sector_model( θsec, M )
    z_pw  = sample_N_iso_pairs( rawJ, q, M, Npairs )
    z_noise = sample_noise( q,M, Nnoise )
    return cat(z_pw, z_noise, z_sec, dims=2)
end

# function main(;  M=500,  seed=123, 
#             FloatType=Float32, Npairs=5, Nnoise=18, q=5, rawJ=1, Ξ=5, S=7, m=3  )
#     # M is number of samples
#     sampledir(args...) = datadir("toysector_q=5","nsamples=$M", args...)
#     isdir(sampledir()) || mkpath( sampledir() )
    
#     sector_temps = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
#     Random.seed!(seed)
    
#     for Tsec in sector_temps
#         filename = sampledir(@sprintf("sector_temp=%.2f_samples.bson", Tsec))
#         _, θsec = init_toy_model(Tsec; FloatType=FloatType, 
#                 rawJ=rawJ, Ξ=Ξ, q=q, S=S, m=m, npairwise=2*Npairs )
#         z = sample_toy_model( M, rawJ, θsec ; Npairs=Npairs, Nnoise=Nnoise)
#         bson(filename, samples = z)
#     end
    
# end
    
# seeds    = [123, 456, 98765, 12345, 123456789, 111000111,  321, 4444 ]
# nsamples = [500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
# for (M, seed) in zip(nsamples, seeds) 
#     main( ; M=M, seed=seed )
# end

