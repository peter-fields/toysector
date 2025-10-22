"""
type and energy function for sector:

by default, attractive patterns of a sector are defined such that no patters are overlapping.
e.g. for 3 amino acids and sector of length 4 we have AAAA, BBBB, CCCC
Ξ,q,S = (# patterns), (# spin states), (# length of sector)
m = # of mutations above which energy is infinte. default is ceil(  S/2 - 1 )
default FloatType is Float32 for GPU
"""

using SpinModels

### Container for syngistic model of sector
struct Sector{A} <: SpinModel{A}
    patterns::A
    Ξ::Integer   
    q::Integer   
    S::Integer   
    f::Function  #energy function to determine spacing of energy levels for n-point mutations
    m::Integer   
    FloatType::DataType 
    function Sector( patterns::AbstractArray{Bool,3}, 
                    f::Function, m::T, FloatType::DataType ) where T<:Integer
        Ξ, q, S = size(patterns)
        Ξ > q && @warn "number of patterns exceeds number of amino acids. some patterns may be overlapping"
        return new{typeof(patterns)}(patterns, Ξ, q, S, f, m, FloatType)      
    end
    function Sector(patterns::AbstractArray{Bool,3}, f::Function)
        m=ceil(size(patterns,3)/2-1)
        Sector(patterns, f, m, Float32)
    end
    function Sector(patterns::AbstractArray{Bool,3}, f::Function, FloatType::DataType)
        m=ceil(size(patterns,3)/2-1)
        Sector(patterns, f, m, FloatType)
    end
    Sector(patterns::AbstractArray{Bool,3}, f::Function, m::Integer) = Sector(patterns, f, m, Float32)
    
    function Sector(; Ξ::T, q::T, S::T, f::Function, m::T=Int(ceil(S/2-1)), FloatType::DataType=Float32) where T<:Int
        patterns =  collect(Flux.onehotbatch( Int.(reduce(hcat,(  ξ*ones(S)  for ξ in 1:Ξ ) ) ) , 1:q ))
        @. patterns = !patterns
        patterns = permutedims( patterns, (3,1,2) )
        return Sector(patterns, f, m, FloatType)
    end
end 

### energy function for sectors
sectorenergy(z::AbstractMatrix, θ) = sectorenergy( z, θ, SectorEnergyBuffer(θ, 1))
sectorenergy(z::AbstractArray{<:Any,3}, θ) = sectorenergy( z, θ, SectorEnergyBuffer(θ, size(z, 3)) )
function sectorenergy(z, θ::Sector, sectorbuffer)
    (; patterns, q, S, f, m, FloatType) = θ
    (; C, Cₘᵢₙ) = sectorbuffer
    
    qS=q*S
    infinity = convert(FloatType, Inf)
    replacer(x) = x>m ? infinity : x #assign infinite energy to # of mutations above threshold
    
    ### energy = f(number of mutations away from an ideal pattern)
    mul!( C, reshape(patterns, :, qS) , reshape(z, qS, :) )
    minimum!(Cₘᵢₙ, C)
    replace!( replacer , Cₘᵢₙ)
    @. Cₘᵢₙ = f(Cₘᵢₙ) 
    return vec(Cₘᵢₙ)
end

### buffer for energy function
struct SectorEnergyBuffer{V,V2}
    C::V
    Cₘᵢₙ::V2
    function SectorEnergyBuffer(θ::Sector, M::Integer)
        (; Ξ, FloatType) = θ
        C = zeros(FloatType, Ξ, M)
        Cₘᵢₙ = similar(C,1, M)
        return new{typeof(C),typeof(Cₘᵢₙ)}(C, Cₘᵢₙ)
    end
    function SectorEnergyBuffer(Ξ::Integer, M::Integer, FloatType)
        C = zeros(FloatType, Ξ, M)
        Cₘᵢₙ = similar(C,1, M)
        return new{typeof(C),typeof(Cₘᵢₙ)}(C, Cₘᵢₙ)
    end
    SectorEnergyBuffer(z::AbstractArray{<:Any,3}, θ::Sector) = SectorEnergyBuffer(θ, size(z,3))
end

function init_toy_model( Tsec ; FloatType=Float32, rawJ=1, Ξ=5, q=5, S=7, m=3, npairwise=10 )
    θpw = begin
        J0 = diagm(fill(rawJ, q)); Jtoy = zeros(FloatType, q,npairwise,q,npairwise)
        for i ∈ 1:2:npairwise; Jtoy[:,i,:,i+1] .= J0; end
        SpinModels.dropselfcoupling!(Jtoy); SpinModels.symmetrizecoupling!(Jtoy);
        zerosum!( Pairwise(   zeros(eltype(Jtoy),q,npairwise)   , Jtoy ));
    end
    Tsec = FloatType(Tsec)
    levels(x)=x/Tsec #assigns energy to different numbers of mutations
    θsec = Sector(; Ξ=Ξ, q=q, S=S, f=levels, m=m, FloatType=FloatType ) ; 
    return θpw, θsec
end