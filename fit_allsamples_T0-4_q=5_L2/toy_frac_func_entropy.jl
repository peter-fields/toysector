using DrWatson
@quickactivate "toysector"
# include(srcdir("makeLocalPref.jl"))
# makeLocalPref11_5()
using SpinModels, Optim, Parameters, Flux
# rmLocalPref()
# makeLocalPref()
using CUDA
using LinearAlgebra, Statistics, Printf
# rmLocalPref()
using BSON, Distributions, Random 
# using PyPlot
import StatsBase.countmap
#using GLMNet

### define fitting functions
include( srcdir( "fitting.jl") )

include( srcdir(  "samplers.jl"  ) )
include( srcdir( "aisZ.jl" ) )
#include( srcdir( "helpers.jl" ) )

#CM_logit_L2 = BSON.load( srcdir("CM_logit_L2") )[:CM_logit_L2]

# CUDA.versioninfo()

jobid = parse(Int64, ARGS[1])

### specify hyperparams for each model to sweep
modls = [:Pairwise , :RBM] #, :SRBM]
params_sets = Dict{Symbol,Any}(
    :Pairwise => Dict{Symbol, Any}(
        :reg_J => [(10f0^log10λ, 2) for log10λ ∈ -8:1], # -8
        :reg_h => [(10f0^log10λ, 2) for log10λ ∈ 2:2],  # -4
        :epochs => 20_000,
        :showevery => 500,
        :progTol => 10^-6
    )
     ,
     :RBM => Dict{Symbol, Any}(
         :reg_W => [(10f0^log10λ, 2) for log10λ ∈ -8:1],
         :reg_h => [(10f0^log10λ, 2) for log10λ ∈ 2:2],
         :P => [50],
         :epochs => 20_000,
         :showevery => 500,
         :progTol => 10^-6
     )
#     ,
#     :SRBM => Dict{Symbol, Any}(
#         :reg_J => [(10f0^log10λ, 2) for log10λ ∈ -8:2:0],
#         :reg_W => [(10f0^log10λ, 2) for log10λ ∈ -8:2:0],
#         :reg_h => [(10f0^log10λ, 2) for log10λ ∈ -4:2:2],
#         :P => collect(10:20:50),
#         :epochs => 20_000,
#         :showevery => 50,
#         :progTol => 10^-6
#     )
)

### global variables for pulling models
seed = 123
Random.seed!(seed)
calc_frac_func = false
calc_H = true
q,N = (5,35)

sampledir(args...) = datadir("toysector_q=5", "nsamples=500", args...)

fitsdir(args...) = sampledir("fitted_models","sector_temp=0.40", args...)

##### NOTHING BELOW HERE SHOULD BE EDITED #####
option_ais =OptionAIS(
		anneal_steps=1000, 
		showevery=20, 
		chains=5000, 
		gibbs=false,
		silent=false,
		samplingDict=Dict(:traceevery=>1, :stopafter=>50, :max_sampling_steps=>1000),
		gpu = CUDA.functional() 
	)


### make containers for in-sample and estimated out-of-sample loss for each model's sweep over hyperparams
sweep_dicts = Dict{Symbol, Any}()
for mdl in modls
    sweep_dicts[mdl] = dict_list( params_sets[mdl] )
end
    
@show CUDA.functional()
### for each model type (pairwise, rbm, sRBM)
for mdl in modls
    mdldir(args...) = fitsdir(string(mdl), args...)
    isdir(mdldir()) || mkpath(mdldir())
    
    # swp_ds_string = mdldir( string(mdl)*"_best_results.bson")
    
    ### do hyperparam sweep
    swp_ds = [ sweep_dicts[mdl][jobid] ]#alias
    @printf "\n"
    @printf "Begin calculations of H and frac_func of %s model\n" string(mdl)
    for d in swp_ds
        @printf "\n"
        @printf "-----------------------------------------------\n"
        @printf "%s\n" savename(d)
        P = haskey(d, :P) ? d[:P] : 0
        θ = initθ(mdl, q, N, P)
        
        if isfile( mdldir(savename(d)*".bson") )
            @printf "model alrealy fit. loading into memory\n"
        else
            @error "model $(savename(d)) not found." 
        end
        merge!(d, BSON.load(mdldir(savename(d)*".bson"))[:d])
        

        BSON.@load mdldir(savename(d)*"_model.bson") vecθ
        copyto!(θ, vecθ)
        
        @printf "-----------------------------------------------\n"
        
        
        ###calculate false_pos_rate and entropy
        @show typeof(θ)
        @printf "calculating entropy\n" 
        fit_entropy, z_from_fit = calc_H ? begin
            ais_trace, ais_result, z_from_fit = AISlogZ(θ, option_ais)
            @show ais_result
            (ais_result[:entropy], z_from_fit)
        end : nothing
        @printf "\n"
        calc_frac_func && (@printf "calculating frac func\n")
        frac_func = calc_frac_func ? begin
            # _, z_from_fit = drawsamples( θ, 1. , SamplerOption(seed=0, M=10_000, stopafter=10) )

            probs = logit_predict_samples( CM_logit_L2, z_from_fit )
            length( findall(x -> x>0.5, probs ) ) / length(probs) 
        end : nothing
        
        calc_frac_func && ( @printf "frac_func = %f" frac_func )
        @printf "\n"
	
        
        ### put results in dictionary
        results = Dict{Symbol, Any}( :frac_func => frac_func, 
                :fit_entropy => fit_entropy )  
        merge!(d, results)
        @show keys(d)
        
        ### save results
        BSON.@save mdldir(savename(d)*".bson") d
        
    end
    #also save the vector of dictionaries with results
#     BSON.@save swp_ds_string swp_ds
end

@printf "finished!" 
return    



