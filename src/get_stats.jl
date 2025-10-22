# % function takes MSA. It must be formatted such that spin takes on a value
# % from [1, q] inclusive, where q is number of values a spin can take. 
# % Each row is a sequence 

function nchoose2(v)
    #get all n choose 2 pairings of items in v
    l = length(v)
    pairs = []
    for i in 1:l
        for j in 1:l 
            if i<j
                push!( pairs, [v[i], v[j]]  )
            end
        end
    end
    return pairs
end

function nchoose3(v)
    #get all n choose 3 pairings of items in v
    l = length(v)
    trips = []
    for i in 1:l
        for j in 1:l
            for k in 1:l
                if i<j && j<k
                    push!( trips, [ v[i], v[j], v[k] ]  )
                end
            end
        end
    end
    return trips
end

function get_freq_ijl(MSA, i, j, l, nsamp, count_ijl)
    count_ijl .= 0.0     
    for k = 1:nsamp
        count_ijl[ MSA[k,i] , MSA[k,j] , MSA[k,l] ] += 1
    end
    freq_ijl = count_ijl ./ nsamp
    
#     return freq_ijl
    
end


function get_counts(MSA::AbstractMatrix{Int}, q::Integer)
    # get frequencies for first and second order stats
    # q is of different spin-values a site can take
    nsamp = size(MSA, 1);
    seq_len = size(MSA, 2);
    
    freqs_ij = zeros(q,q,seq_len,seq_len);
    
    for i = 1:seq_len
        for j = 1:seq_len
            if i < j || i == j
                count_ij = zeros(q,q); #initialize the qxq matrix where each entry will
#                 % have the number of times that pair of spins occurs in i'th and j'th position of
#                 % the MSA respectively
                
            
                
                for k = 1:nsamp
                    count_ij[ MSA[k,i], MSA[k,j] ] += 1;
                end
                
                freqs_ij[:,:,i,j] = count_ij ./ nsamp ;
            end
        end
    end
    
    freqs_i = zeros(q,seq_len);
    
    for i = 1:seq_len
        count_i = zeros(q,1); #initialize the q-long array where each entry will
#         % have the number of times that spin-k appears at site i
        for k = 1:nsamp
            count_i[ MSA[k,i] ] += 1;
        end
        
        freqs_i[:,i] = count_i / nsamp;
    end
    
    return freqs_i, freqs_ij #, freqs_ijl #, freqs_ijlp

end


function get_stats(MSA::Matrix{Int}, q::Int, ids...)

# %%% get normalized histogram-like arrays containing pairwise 
freq_i, freq_ij = get_counts(MSA, q);


# % get the correlation coeefficient matrices for all position pairs i,j 

nsamp = size(MSA,1);
s_len = size(MSA,2); #%length of each sequence in the sample array

C  = zeros(q,q,    s_len,s_len);
# C3 = zeros(q,q,q,  s_len,s_len,s_len);
# C4 = zeros(q,q,q,q,s_len,s_len,s_len,s_len)
    
# % C(:,:,i,j) is the qxq matrix of coefficients for all pairs of 
# % amino acids at positions i and j. 

froC = zeros(s_len, s_len); #%initialize matrix with Frobenius norm of each Cij matrix
allC = []; 
allC3 = zeros( binomial(s_len,3) * q^3 ) ;
froC3 = zeros( s_len, s_len, s_len )
allfroC3 = zeros( binomial(s_len,3) )

sec_ids = Vector{Any}(undef,0)
# @show ids
    for (i, idxs) in enumerate(ids)
        # @show i, idxs
        d = Dict{Symbol,Any}()
        d[:idxs] = idxs
        d[:secpairs] = nchoose2(d[:idxs])
        d[:allC2] = []
        d[:allfroC2] = [] 
        d[:sectrips] = nchoose3(d[:idxs])
        d[:allC3] = []
        d[:allfroC3] = []
        d[:freq_i] = freq_i[:, d[:idxs] ][:]
        push!(sec_ids, d)
        # # @show d
    end
# @show typeof(sec_ids)
    
for i = 1:s_len 
    for j = 1:s_len
        if i < j
            Cij = freq_ij[:,:,i,j] - freq_i[:,i] * freq_i[:,j]'
            C[:,:,i,j] .= Cij;
            froC[i,j] = norm( Cij[:] );  
            allC = [allC ; Cij[:] ] ;
            
            for d in sec_ids
                if [i,j] in d[:secpairs]
                        # @show [i,j]
                    append!( d[:allC2], Cij[:] )
                    push!(   d[:allfroC2], froC[i,j] ) 
                end
            end
                
        end 
    end
end

    C3ijl    = zeros(q,q,q)
    freq_ijl = zeros(q,q,q)
    count_ijl = zeros(q,q,q)
    kk = 1
    tt = 1
for i = 1:s_len #%normally s_len goes here
    for j = 1:s_len
        for l = 1:s_len
            
            if i < j && j < l
# %                 C3ijl = C3(:,:,:,i,j,l);
                    
                
                freq_ijl .= get_freq_ijl(MSA, i,j,l, nsamp, count_ijl )
                    
                
                    
                for a = 1:q
                    for b = 1:q
                        for c = 1:q
                        C3ijl[a,b,c] = freq_ijl[a,b,c] - 
                             freq_ij[a,b,i,j]*freq_i[c,l] - 
                             freq_ij[a,c,i,l]*freq_i[b,j] - 
                             freq_ij[b,c,j,l]*freq_i[a,i] +
                             2*freq_i[a,i]*freq_i[b,j]*freq_i[c,l];
                        end
                    end
                end
#                 C3ijl = C3[:,:,:,i,j,l];
                allC3[kk:(kk+q^3-1)] .= C3ijl[:]
                froC3[i,j,l] = norm(C3ijl[:])
                allfroC3[tt] = norm(C3ijl[:])
                tt+=1
                kk += q^3
                        
                for d in sec_ids
                    if [i,j,l] in d[:sectrips]
                        append!( d[:allC3], C3ijl[:] )
                        push!(   d[:allfroC3], norm(C3ijl[:]) ) 
                    end
                end
                    
            end
   
        end
    end
end
    

    stats = Dict{Symbol, Any}()
    stats[:freq_i]    = freq_i
    stats[:freq_ij]   = freq_ij
    stats[:C2]        = C
    stats[:froC2]     = froC
    stats[:allC2]     = allC
    stats[:allC3]     = allC3
    stats[:froC3]     = froC3
    stats[:allfroC3]  = allfroC3
    stats[:allfroC2]  = froC[ froC .!= 0.0 ]
    stats[:secstats]  = sec_ids
    
    return stats

end