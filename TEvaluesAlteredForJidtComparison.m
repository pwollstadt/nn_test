function [te] = TEvaluesAlteredForJidtComparison(ts_1,ts_2,dim,tau,u,k_th,TheilerT,runOnlyGroupSearch)
%
% JL -- this function copied from TRENTOOL 3.0 altered for the comparison to JIDT.
% MI calculation is removed.
% 

% TRANSFERENTROPYVALUES computes the transfer entropy (TE) among a given
% pair of time series. source (ts_1) -> target (ts_2)
%
% This function sis called by the transferentropy.
%
% REFERENCE INFORMATION
%   - The concept of TE appears in Schreiber's article,
%     "Measuring Information Transfer", Phys. Rev. Lett. 85, 461 - 464 (2000).
%   - For the estimation of probability densities needed for the TE
%     computation, the function implements the Kraskov-Stoegbauer-Grassberger
%     estimator described in Kraskov et al. "Estimating mutual information",
%     Phys. Rev. E 69 (6) 066138, (2004).
%
% * DEPENDENCIES
%   - Package TSTOOL is used at nearest neighbors searches
%     required for the KSG estimator.
%
% INPUT PARAMETERS
%   - cfg       = configuration structure
%   - ts_1      = time series 1
%   - ts_2      = time series 2 (ts_2 should be of equal length than ts_1)
%   - dim       = embedding dimension
%   - tau       = embedding delay in number of sampled points
%   - u         = points ahead for the advance vector in number of sampled
%                 points
%   - k_th      = number of neighbors for fixed mass search (controls 
%                 balance of bias/statistical errors)
%   - TheilerT  = number of temporal neighbors excluded to avoid serial 
%                 correlations (Theiler correction)
%
% OUTPUT PARAMETERS
%   - te = transfer entropy time series 1 -> time series 2
%
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation;
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY;
%
% Version 2.0 by Michael Lindner, Raul Vicente, Michael Wibral
% Frankfurt 2012

if (nargin < 8)
	runOnlyGroupSearch = true;
end

tic

%% Preprocessing of the data to be read by TSTOOL %%

% Z-scoring the time series
z_data_1 = zscore(ts_1);
z_data_2 = zscore(ts_2);

%% Creating the embedding vectors

% Computing effective lengths

T = length(ts_1);     % length of full time series
M = T-(dim-1)*tau;    % number of points inside the time series ready for delay embedding
L = M-u;              % number of points inside the time series ready for advance and delay embedding
WOI = 1:L;          % Window of interest


% Initialization of embedding vectors
pointset_2 = zeros(L,dim);
pointset_p2 = zeros(L,dim+1);
pointset_21 = zeros(L,2*dim);
pointset_p21 = zeros(L,2*dim+1);

% Embedding vectors

for ii = 1:L           % Marginal distributions
    for jj = 1:dim
        pointset_2(ii,jj) = z_data_2(ii+(dim-1)*tau-(jj-1)*tau);
    end
end


for ii = 1:L           % Join distributions of marginal and own future state
    for jj = 1:dim+1
        if jj == 1
            pointset_p2(ii,jj) = z_data_2(ii+(dim-1)*tau+u);
        else
            pointset_p2(ii,jj) = z_data_2(ii+(dim-1)*tau-(jj-2)*tau);
        end
    end
end




for ii = 1:L           % Join distributions of the two time series
    for jj = 1:2*dim
        if jj <= dim
            pointset_21(ii,jj) = z_data_2(ii+(dim-1)*tau-(jj-1)*tau);
        else
            pointset_21(ii,jj) = z_data_1(ii+(dim-1)*tau-(jj-dim-1)*tau);
        end
    end
end

for ii = 1:L           % Join distributions of join marginal and own future states
    for jj = 1:2*dim+1
        if jj == 1
            pointset_p21(ii,jj) = z_data_2(ii+(dim-1)*tau+u);
        elseif jj > 1 && jj <= dim+1
            pointset_p21(ii,jj) = z_data_2(ii+(dim-1)*tau-(jj-2)*tau);
        else
            pointset_p21(ii,jj) = z_data_1(ii+(dim-1)*tau-(jj-dim-2)*tau);
        end
    end
end


embeddingCompletedTime = toc;
fprintf('Embedding completed after: %.3f s\n', embeddingCompletedTime);

%% Nearest neighbors search (fixed mass)




% Preprocessing for nearest neighbor searches
atria_2 = nn_prepare(pointset_2,'maximum');
atria_p2 = nn_prepare(pointset_p2,'maximum');
atria_21 = nn_prepare(pointset_21,'maximum');
atria_p21 = nn_prepare(pointset_p21,'maximum');

timeToEndAtria = toc;
atriaTime = timeToEndAtria - embeddingCompletedTime;
fprintf('Atria completed in: %.3f s\n', atriaTime);

% Finding the k_th nearest neighbor
% [index_2, distance_2] = nn_search(pointset_2,atria_2,WOI,k_th,TheilerT);
% [index_p2, distance_p2] = nn_search(pointset_p2,atria_p2,WOI,k_th,TheilerT);
% [index_21, distance_21] = nn_search(pointset_21,atria_21,WOI,k_th,TheilerT);
[index_p21, distance_p21] = nn_search(pointset_p21,atria_p21,WOI,k_th,TheilerT);

timeToEndNnSearch = toc;
nnSearchTime = timeToEndNnSearch - timeToEndAtria;
fprintf('kth NN searches completed in: %.3f s\n', nnSearchTime);

%% Nearest neighbor search (fixed radius)

if (~runOnlyGroupSearch)
	ncount_p21_p2 = zeros(L,1);
	ncount_p21_21 = zeros(L,1);
	ncount_p21_2 = zeros(L,1);

	for i=1:L
	    [count_p21_p2, neighbors_p21_p2] = range_search(pointset_p2,atria_p2,i,distance_p21(i,k_th)-eps,TheilerT);
	    [count_p21_21, neighbors_p21_21] = range_search(pointset_21,atria_21,i,distance_p21(i,k_th)-eps,TheilerT);
	    [count_p21_2,  neighbors_p21_2]  = range_search(pointset_2,atria_2,i,distance_p21(i,k_th)-eps,TheilerT);
	    ncount_p21_p2(i) = count_p21_p2;
	    ncount_p21_21(i) = count_p21_21;
	    ncount_p21_2(i)  = count_p21_2;
	    if ((L > 10000) && (mod(i,1000) == 0))
		timeFixedRSoFar = toc - timeToEndNnSearch;
	    	fprintf('Done %d neighbour searches ... (%.3f s for searches; projecting final time %.3f s)\n', i, timeFixedRSoFar, timeToEndNnSearch + (timeFixedRSoFar / i * L));
	    end
	end


	%% Transfer entropy
	te = psi(k_th)+mean(psi(ncount_p21_2+1)-psi(ncount_p21_p2+1)-psi(ncount_p21_21+1));

	timeToEndRSearch = toc;
	fixedRSearchTime = timeToEndRSearch - timeToEndNnSearch;
	fprintf('Fixed R searches completed in: %.3f s\n', fixedRSearchTime);
else
	timeToEndRSearch = toc;
	fixedRSearchTime = timeToEndRSearch - timeToEndNnSearch;
	fprintf('No fixed R searches performed\n');
end

% Try using a group call here:
[count_p21_p2, neighbors_p21_p2] = range_search(pointset_p2,atria_p2,1:L,distance_p21(:,k_th)-eps,TheilerT);
[count_p21_21, neighbors_p21_21] = range_search(pointset_21,atria_21,1:L,distance_p21(:,k_th)-eps,TheilerT);
[count_p21_2,  neighbors_p21_2]  = range_search(pointset_2,atria_2,1:L,distance_p21(:,k_th)-eps,TheilerT);
%% Transfer entropy
teGroup = psi(k_th)+mean(psi(count_p21_2+1)-psi(count_p21_p2+1)-psi(count_p21_21+1));
timeToEndGroupFixedRSearch = toc;
groupFixedRSearchTime = timeToEndGroupFixedRSearch - timeToEndRSearch;
fprintf('Group Fixed R searches completed in: %.3f s (result %.4f)\n', groupFixedRSearchTime, teGroup);
fprintf('Using a group search would give a runtime of %.3f s\n', (timeToEndGroupFixedRSearch - fixedRSearchTime));
if (runOnlyGroupSearch)
	te = teGroup;
end

return;
