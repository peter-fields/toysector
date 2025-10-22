using DrWatson
@quickactivate "proteins"
using SpinModels, Optim, Parameters, CUDA, LinearAlgebra, Statistics, Printf, Flux
using BSON, Distributions, Random 
using PyPlot
import StatsBase.countmap

### define fitting functions
include(projectdir("newtoymodel", "fitting.jl"))
include( srcdir( "toysector.jl" ) )
include( srcdir( "false_positive_funcs.jl" ) )

"""
This script will do cross-val fits of pw, rbm, and srbm to 1 dataset.
Stopping conditons for fitting are such that the relative change in train loss between epochs falls below progTol
Single floating point precision is used
"""

#=
to put in sbatch later
JULIA_CUDA_USE_BINARYBUILDER=false julia fitP30.jl $SLURM_ARRAY_TASK_ID
=#

### specify lists of sector temps and nsamples you want to fit
jobid = parse(Int64, ARGS[1])

samples_to_fit_all = Dict{Symbol, Any}(
    :M => [500], #nsamples
    :T_sec => [0.4, 1.0, 1.4, 2.25, 3.25, 4.25, 5.25] ) 

### specify hyperparams for each model to sweep
modls = [:Pairwise, :RBM] #, :SRBM]
params_sets = Dict{Symbol,Any}(
    :Pairwise => Dict{Symbol, Any}(
        :reg_J => [(10f0^log10λ, 2) for log10λ ∈ -4:0],
        :reg_h => [(10f0^log10λ, 2) for log10λ ∈ -4:0],
        :epochs => 8000,
        :showevery => 500,
        :progTol => 10^-6
    ),

    # :RBM => Dict{Symbol, Any}(
    #     :reg_W => [(10f0^log10λ, 2) for log10λ ∈ -4:0],
    #     :reg_h => [(10f0^log10λ, 2) for log10λ ∈ -4:0],
    #     :P => 30,
    #     :epochs => 8000,
    #     :showevery => 500,
    #     :progTol => 10^-6
    # )
    # ,
    # :SRBM => Dict{Symbol, Any}(
    #     :reg_J => [(10f0^log10λ, 2) for log10λ ∈ -6:-1],
    #     :reg_W => [(10f0^log10λ, 2) for log10λ ∈ -8:-1],
    #     :reg_h => [(10f0^log10λ, 2) for log10λ ∈ -6:-1],
    #     :P => collect(10:20:50),
    #     :epochs => 4000,
    #     :showevery => 50,
    #     :progTol => 10^-6
    # )
)

### load data and partition it
folds = 5
seed = 123
Random.seed!(seed)
calc_f_pos = true

samples_to_fit_info = dict_list(samples_to_fit_all)[jobid]
sampledir(args...) = datadir("toysector_q=5", "nsamples=$(samples_to_fit_info[:M])", args...)
z = BSON.load(
    sampledir( @sprintf("sector_temp=%.2f_samples.bson", samples_to_fit_info[:T_sec]) ) 
    )[:samples];
z = Float32.(z)
q,N,_=size(z)
# z=z[:, 29:35, :] # this line gets only the sector
# fitsdir(args...) = sampledir("fitted_models_sec_only", @sprintf("sector_temp=%.2f", samples_to_fit_info[:T_sec]), args...)

fitsdir(args...) = sampledir("fitted_models", @sprintf("sector_temp=%.2f", samples_to_fit_info[:T_sec]), args...)

##### NOTHING BELOW HERE SHOULD BE EDITED #####

### containers for f_pos_calculations
_, θsec =  init_toy_model( samples_to_fit_info[:T_sec] )
all_states_buffer = log(qᴺ) < 21. ? AllStatesBuffer( q, N, θsec, θsec.FloatType ) : nothing
sec_energy_buffer = SectorEnergyBuffer( θsec , 20_000 )

### make into cuda array if possible
z = CUDA.functional() ? cu(z) : z
@printf "%s\n" "Array type: $(typeof(z))"
𝕏 = kfold( Float32.(z), folds )


### make containers for in-sample and estimated out-of-sample loss for each model's sweep over hyperparams
sweep_dicts = Dict{Symbol, Any}()
for mdl in modls
    sweep_dicts[mdl] = dict_list( params_sets[mdl] )
end
    

### for each model type (pairwise, rbm, sRBM)
for mdl in modls
    mdldir(args...) = fitsdir(string(mdl), args...)
    isdir(mdldir()) || mkpath(mdldir())
    
    ### check if sweep is already done
    swp_ds_string = mdldir( string(mdl)*"_all_results.bson")
    isfile(swp_ds_string) && begin 
        sweep_dicts[mdl]  = BSON.load(swp_ds_string)[:swp_ds] 
	end
    
    ### do hyperparam sweep
    swp_ds = sweep_dicts[mdl] #alias
    @printf "\n"
    @printf "Begin sweep of %s model\n" string(mdl)
    for d in sweep_dicts[mdl]
        @printf "\n"
        @printf "-----------------------------------------------\n"
        @printf "%s\n" savename(d)
	    isfile(mdldir(savename(d)*".bson")) && begin
            merge!(d, BSON.load(mdldir(savename(d)*".bson"))[:d])
            @printf "model alrealy fit. skipping\n" 
            continue
        end
        
        @printf "-----------------------------------------------\n"
        ### 5 fold fit
        θ = mdl == :Pairwise ? Pairwise(; q=q , N=N, similarto=z) : eval(mdl)(q=q,N=N,P=d[:P], similarto=z)
        SpinModels.random!(θ)
        _, 𝕋, 𝕍 = kfold_sgdfit!(θ, 𝕏; d... )
        ### fit all data to model
        SpinModels.random!(θ)
        @printf "train on whole dataset\n"
        intrace = sgdfit!(θ, z; d... )
        inloss = intrace[:f_training][end] 
        
        ###calculate false_pos_rate
        f_pos_rate = calc_f_pos ? get_f_pos_rate( θsec, θ, sec_energy_buffer, all_states_buffer ) : nothing
        
        ### put results in dictionary
        results = Dict{Symbol, Any}( :𝕍 => 𝕍, :𝕋 => 𝕋, :intrace => intrace, :inloss => inloss, 
             :mean𝕍 => mean(𝕍), :std𝕍 => std(𝕍), :f_pos_rate => f_pos_rate ) 
        merge!(d, results)
        
        
        ### save results
        BSON.@save mdldir(savename(d)*".bson") d
        BSON.@save mdldir(savename(d)*"_model.bson") collect(veccopy(θ))
    end
    #also save the vector of dictionaries with results
    BSON.@save swp_ds_string swp_ds
end
    
### after all hyperparam sweeps are finished:   
### parse each crossval for best hyperparams
losses_pw   = pull_from_sweep(sweep_dicts[:Pairwise], 
    [:reg_h, :reg_J, :mean𝕍] 
    )
best_idx_pw = findmin(losses_pw[:,end])[2]
best_pw_reg_h, best_pw_reg_J, _ = losses_pw[best_idx_pw, :]

losses_rbm  = pull_from_sweep(sweep_dicts[:RBM], 
    [:reg_h, :reg_W, :P, :mean𝕍] 
    )
best_idx_rbm = findmin(losses_rbm[:,end])[2]
best_rbm_reg_h, best_rbm_reg_W, best_rbm_P, _ = losses_rbm[best_idx_rbm, :]

# losses_srbm = pull_from_sweep(sweep_dicts_SRBM, 
#     [:reg_h, :reg_J, :reg_W, :P, :mean𝕍] 
#     )






### put that model in accessible part of folder
best_model_dict_pw=filter( d -> ( (d[:reg_J] == best_pw_reg_J) && (d[:reg_h] == best_pw_reg_h) ) , sweep_dicts[:Pairwise] )[1]
cp( fitsdir("Pairwise", savename(best_model_dict_pw)*"_model.bson") , fitsdir(savename(best_model_dict_pw)*"_bestmodel.bson") ; force=true) 
cp( fitsdir("Pairwise", savename(best_model_dict_pw)*".bson") , fitsdir(savename(best_model_dict_pw)*"_best.bson") ; force = true) 

best_model_dict_rbm=filter( d -> ( (d[:reg_W] == best_rbm_reg_W) && (d[:reg_h] == best_rbm_reg_h) && (d[:P] == best_rbm_P)) , sweep_dicts[:RBM] )[1]
cp( fitsdir("RBM", savename(best_model_dict_rbm)*"_model.bson") , fitsdir(savename(best_model_dict_rbm)*"_bestmodel.bson") , force=true) 
cp( fitsdir("RBM", savename(best_model_dict_rbm)*".bson") , fitsdir(savename(best_model_dict_rbm)*"_best.bson") , force=true)


### make learning curve of lJ and lW at fixed lh
function plot_losses( ax, losses, calc_f_pos)
    lambdas = map(x -> log10(x), map( x -> x[1]  , losses[:,1])) 
    ax.errorbar( lambdas , losses[:,2], yerr = losses[:,3],label = "mean validation loss")
    ax.plot(lambdas, losses[:,4], label = "full loss")
    calc_f_pos && ( ax.plot( lambdas, losses[:,5] , label = "false positive rate")  )
    ax.grid(alpha=0.5)
    ylabel("loss")
    legend()
end

losses_pw   = pull_from_sweep(sweep_dicts[:Pairwise], 
    [:reg_J, :mean𝕍, :std𝕍, :inloss, :f_pos_rate] ;
    reg_h = best_pw_reg_h
    )
losses_rbm   = pull_from_sweep(sweep_dicts[:RBM], 
    [:reg_W, :mean𝕍, :std𝕍, :inloss, :f_pos_rate] ;
    reg_h = best_rbm_reg_h
    )

fig, ax = subplots(dpi=200)
plot_losses( ax, losses_pw, calc_f_pos )
xlabel(L"log_{10}(\lambda_J)")
title("pairwise with lambda_h = $best_pw_reg_h, M=$(samples_to_fit_info[:M])")
savefig(fitsdir("pairwise_learning_curve.png"))


fig, ax = subplots(dpi=200)
plot_losses( ax, losses_rbm, calc_f_pos )
xlabel(L"log_{10}(\lambda_W)")
title("rbm with P = $best_rbm_P, lambda_h = $best_rbm_reg_h, M=$(samples_to_fit_info[:M])")
savefig(fitsdir("rbm_learning_curve.png"))
