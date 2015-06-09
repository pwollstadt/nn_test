function checkTEResults(numObservations, kHistory, knn)
% checkTEResults(numObservations) - Oliver Cliff and Joe Lizier
%
% numObservations -- time series length to check
% kHistory -- embedding dimension for BOTH source and target
% kNNs -- number of nearest neighbours

% For JL, I need to run :
%  cd ~/Work/Software/TStool/
%  tstoolInit
%
% TEvalues.m has been copied to this directory and modified as TEvaluesAlteredForJidtComparison

knn = '4';

covariance=0.4;
sourceArray=randn(numObservations, 1);
destArray = [0; covariance*sourceArray(1:numObservations-1) + (1-covariance)*randn(numObservations - 1, 1)];
sourceArray2=randn(numObservations, 1); % Uncorrelated source

sourceArrayJIDT = zscore(sourceArray);
destArrayJIDT = zscore(destArray);

javaaddpath('/home/joseph/Work/Investigations/JavaCode/sharedProjects/InformationDynamics/infodynamics.jar');

fprintf('JIDT calculation: ');
tic
teCalc=javaObject('infodynamics.measures.continuous.kraskov.TransferEntropyCalculatorKraskov');
teCalc.setProperty('k', knn); % Use Kraskov parameter K=4 for 4 nearest points
teCalc.setProperty('NORMALISE', 'FALSE'); % Leave raw to get same values as for TRENTOOL
teCalc.setProperty('NUM_THREADS', '1'); % Force to use single-threaded
% teCalc.setDebug(true); % Uncomment if you want to be sure about how many threads are running
teCalc.initialise(kHistory, 1, kHistory, 1, 1); % Use history length kLength for source and target)
teCalc.setObservations(sourceArrayJIDT, destArrayJIDT);
resultJIDT = teCalc.computeAverageLocalOfObservations();
toc

fprintf('TRENTOOL calculation: ');
tic
resultTRENTOOL = TEvaluesAlteredForJidtComparison(sourceArray, destArray, kHistory, 1, 1, str2double(knn), 0);
toc

fprintf('[JIDT TRENTOOL] = [%.4f %.4f]\n', ...
            resultJIDT, resultTRENTOOL);


