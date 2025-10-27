using DrWatson
using Pkg
@quickactivate "toysector"
using CSV, DelimitedFiles, HypothesisTests, BSON, Distributions, Random 
using SpinModels, Optim, Parameters, CUDA, LinearAlgebra, Statistics, Printf, Flux
using BenchmarkTools, PyPlot
import StatsBase.countmap
import DrWatson.savename

include( srcdir( "fitting.jl"   ) )
include( srcdir( "toysector.jl" ) )
include( srcdir( "false_positive_funcs.jl" ) )
include( srcdir( "samplers.jl"             ) )
include( srcdir( "aisZ.jl" ))

folds = 5
seed = 123
Random.seed!(seed)
calc_f_pos = true
nsamples_for_fpos = 10_000;
calc_H = true

q,N = (5,35)

### which samples to fit ###
samples_to_fit_all = Dict{Symbol, Any}(
    :M => [500], #nsamples
#     :T_sec => [0.2, 0.25, 0.3, 0.4, 0.5, 0.6]  
    :T_sec => [1.,0.4, 0.2],
    :m => [1,2,3]
    )

modls = [:Pairwise]# , :RBM]#, :SRBM]

params_sets = Dict{Symbol,Any}(
        :Pairwise => Dict{Symbol, Any}(
        :reg_J => [(10f0^log10λ, 2) for log10λ ∈ -8:1],
        :reg_h => [(10f0^log10λ, 2) for log10λ ∈ 2:2],
        :epochs => 20_000,
        :showevery => 500,
        :progTol => 10^-6
    )
     # ,
     #     :RBM => Dict{Symbol, Any}(
     #     :reg_W => [(10f0^log10λ, 2) for log10λ ∈ -8:-1],
     #     :reg_h => [(10f0^log10λ, 2) for log10λ ∈ -4:2],
     #     :P => [10,20,30,40,50],
     #     :epochs => 20_000,
     #     :showevery => 500,
     #     :progTol => 10^-6
     # )
#     ,
#     :SRBM => Dict{Symbol, Any}(
#         :reg_J => [(10f0^log10λ, 2) for log10λ ∈ -6:-1],
#         :reg_W => [(10f0^log10λ, 2) for log10λ ∈ -8:-1],
#         :reg_h => [(10f0^log10λ, 2) for log10λ ∈ -4:2],
#         :P => collect(10:20:50),
#         :epochs => 20_000,
#         :showevery => 50,
#         :progTol => 10^-6
#     )
)

samples_to_fit_dict_list = dict_list(samples_to_fit_all)

job_id = parse(Int64, ARGS[1])

### helper funcs
function get_fpos_from_samps(results,θsec)
    z  = results[:z_from_fit]
    H_ = results[:fit_entropy][:entropy]
    E_under_gt = sectorenergy(z[:,29:35,:], θsec, 
        SectorEnergyBuffer(θsec, size(results[:z_from_fit],3)))
    n_fpos = E_under_gt .== Inf32
    # @show n_fpos
    
    f_pos= sum(n_fpos)/length(n_fpos)
    return f_pos, H_
end
function write_ents_f_pos_to_d(Tsec, m, Tsamp, results,d, mdl_str)
    if Tsamp == 1
        θpw, θsec = init_toy_model( Tsec, m=m )
        f_pos, H_= get_fpos_from_samps(results,θsec)
        if isnothing(d[:true_entropy])
            d[:true_entropy] = H_toymodel( θpw, θsec )[:Htotal]
        end
        if isnothing(d[:false_pos_rate])
            d[:false_pos_rate]=f_pos
        end
        if isnothing(d[:fit_entropy])
            d[:fit_entropy]=H_
        end
    end
    BSON.@save mdl_str*".bson" d
end

### run loops to calculate entropies

samples_to_fit_info=samples_to_fit_dict_list[job_id]

    sampledir(args...) = datadir("toysector_q=5", 
        "nsamples=$(samples_to_fit_info[:M])", args...)

    fitsdir(args...) = sampledir("fitted_models", 
        @sprintf("sector_temp=%.2f_m=%i", 
            samples_to_fit_info[:T_sec], samples_to_fit_info[:m]), 
        args...)

    Tsec, m = (samples_to_fit_info[:T_sec], samples_to_fit_info[:m])

    sweep_dicts = Dict{Symbol, Any}()

    Tsamps = collect(1:-0.1:0.1)
    # Tsamps = [0.1, 0.2, 0.3]


    ##### NOTHING BELOW HERE SHOULD BE EDITED #####
    option_ais =OptionAIS(
            anneal_steps=1000, #1000
            showevery=50, 
            chains=5000, #5000
            gibbs=false,
            silent=false,
            samplingDict=Dict(:traceevery=>1, :stopafter=>50, :max_sampling_steps=>2000),#1000
            gpu = CUDA.functional() 
        )

    @printf("GETTING ENTS AT DIFF T FOR MODELS FIT TO DATA FOR Tsec = %.2f and m=%i\n", 
            samples_to_fit_info[:T_sec], samples_to_fit_info[:m])
    
    ### make containers for in-sample and estimated out-of-sample loss for each model's sweep over hyperparams
    sweep_dicts = Dict{Symbol, Any}()
    for mdl in modls
        sweep_dicts[mdl] = dict_list( params_sets[mdl] )
    end

    @show CUDA.functional()
    ### for each model type (pairwise, rbm, sRBM)
    for mdl in modls
        mdldir(args...) = fitsdir(string(mdl), args...)

        # swp_ds_string = mdldir( string(mdl)*"_best_results.bson")

        ### do hyperparam sweep
        swp_ds =  sweep_dicts[mdl] #alias
        @printf "\n"
        @printf "Begin calculations of H and frac_func of %s model\n" string(mdl)
    #     d = swp_ds[job_id]
        for d in swp_ds
            @printf "\n"
            @printf "-----------------------------------------------\n"
            @printf "%s\n" savename(d)

            P = haskey(d, :P) ? d[:P] : 0
            θ = initθ(mdl, q, N, P)

            mdl_str = mdldir(savename(d))

            if isfile( mdl_str*".bson" )
                @printf "model alrealy fit. loading into memory\n"
            else
                @error "model $(mdl_str) not found." 
            end
            merge!(d, BSON.load(mdl_str*".bson")[:d])


            BSON.@load mdl_str*"_model.bson" vecθ
            copyto!(θ, vecθ)

            @printf "-----------------------------------------------\n"

            ###calculate entropy
            @show typeof(θ)
            for Tsamp in Tsamps
                # check if samples file already exists
                smpl_str=mdldir(savename(d)*"_samples_T=$(Tsamp).bson")
                if isfile(smpl_str)
                    # re-write any arrays that cu-arrays into regular arrays
                    results = BSON.load(smpl_str)[:results]
                    z_from_fit=results[:z_from_fit]
                    results[:z_from_fit] = collect(z_from_fit)
                    BSON.@save mdldir(savename(d)*"_samples_T=$(Tsamp).bson") results
                    @printf "samples at T=%f already exist for %s. skipping\n" Tsamp savename(d)

                    # if Tsamp == 1, put ent and fpos into the corresponding model dict
                    write_ents_f_pos_to_d(Tsec, m, Tsamp, results,d, mdl_str)

                    continue
                end

                @printf "calculating entropy for T = %f\n" Tsamp 

                ais_trace, fit_entropy, z_from_fit = calc_H ? begin
                    ais_trace, ais_result, z_from_fit = AISlogZ(θ, option_ais, Tsamp)
                    @show ais_result
                    (ais_trace, ais_result, z_from_fit)
                end : (nothing, nothing, nothing)

            #         calc_frac_func && ( @printf "frac_func = %f" frac_func )
            #         @printf "\n"

                z_from_fit = collect(z_from_fit)

                ### put results in dictionary
                results = Dict{Symbol, Any}(
                        :z_from_fit => z_from_fit, :ais_trace=>ais_trace,
                        :fit_entropy => fit_entropy )
                
                _, θsec = init_toy_model( Tsec, m=m )
                f_pos, H_= get_fpos_from_samps(results,θsec)
                
                write_ents_f_pos_to_d(Tsec, m, Tsamp, results,d, mdl_str)
                
                results[:f_pos] = f_pos
                
                BSON.@save mdldir(savename(d)*"_samples_T=$(Tsamp).bson") results
            end

        end
        #also save the vector of dictionaries with results
    #     BSON.@save swp_ds_string swp_ds
    end