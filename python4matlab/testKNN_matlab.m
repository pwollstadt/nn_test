%% DATA INITIALIZATION
chunksize=4096;
nchunks = 100; %Number of chunks must be 1 if we want to compare with TSTOOL output
kth = 6;
pointdim = 8;
thelier = 0;
radius = 0.0;
gpuid = 1;
datalength = nchunks * chunksize;
CompareCudaOpenCL = false;

% CONVERT INPUT MATRIX TO SINGLE VECTOR
%pointset = [-12.1 23.4 -20.6 21.6 -8.5 23.7 -10.1 8.5;	 5.3 -9.2  8.2 -15.3  15.1  -9.2  5.5 -15.1]';
pointset = rand(datalength,pointdim,'single');
queryset = pointset;

% Save Initialization Data in prueba.mat
save ('KNNData', 'chunksize', 'nchunks', 'kth', 'pointdim', 'thelier', 'gpuid', '-V7.3');
save ('KNNData', 'pointset', '-append');


%% GPU EXECUTION
if CompareCudaOpenCL
    disp(['EXECUTING KNN GPU... (K:' num2str(kth) ') on ' num2str(size(queryset,1)) ' points of dimension ' num2str(size(queryset,2)) ' in ' num2str(nchunks) ' chunks']);
  
    timegpustart = tic();
    [indexescuda,distancescuda]=fnearneigh_multigpu(single(pointset),single(queryset),kth,thelier,nchunks,gpuid);
    timegpu = toc(timegpustart);
    disp(['Time for KNN mex file:' num2str(timegpu)]);

    
    save ('KNNData', 'indexescuda', '-append');
    save ('KNNData', 'distancescuda', '-append');
end

timegpustart2 = tic();
systemCommand = ['./matlab_to_pyOpenCL_KNN.py ']
system(systemCommand);

%Load OpenCL results from prueba.mat
load ('KNNData.mat', 'distancesopencl', 'indexesopencl');
timegpu2 = toc(timegpustart2);
disp(['Time for OpenCL file:' num2str(timegpu2)]);


if CompareCudaOpenCL
    disp('COMPARING OUTPUT...');

    compindex=sum(indexescuda,2)==sum(indexesopencl,2);
    resultind=all(compindex);
    if resultind 
        disp('Comparing indexes...PASSED');
    else
        disp('Comparing indexes...FAIL. DIFFERENT INDEXES FOUND (DISTANCES PROBABLY EQUAL)');
    end
    eps=1e-5;
    compdist=abs(distancescuda-distancesopencl)<eps;
    resultdist=all(compdist);
    if resultdist
        disp('Comparing distances...PASSED');
    else
        disp('Comparing distances...FAIL. DIFFERENT DISTANCES FOUND');
    end
end
