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
radius = rand(datalength,1,'single');


% Save Initialization Data in prueba.mat
save ('RSAllData', 'chunksize', 'nchunks', 'kth', 'pointdim', 'thelier', 'gpuid', '-V7.3');
save ('RSAllData', 'pointset', '-append');
save ('RSAllData', 'radius', '-append');


%% GPU EXECUTION
if CompareCudaOpenCL
    disp(['EXECUTING RS GPU... (Radius variable for each point) on ' num2str(size(queryset,1)) ' points of dimension ' num2str(size(queryset,2)) ' in ' num2str(nchunks) ' chunks']);
  
    timegpustart = tic();
    npointscuda=range_search_all_gpu(single(pointset),single(queryset),single(radius),thelier,nchunks);    
    timegpu = toc(timegpustart);
    disp(['Time for RS mex file:' num2str(timegpu)]);
    
    save ('RSAllData', 'npointscuda', '-append');
end

timegpustart = tic();
systemCommand = ['./matlab_to_pyOpenCL_RSAll.py ']
system(systemCommand);

%Load OpenCL results from prueba.mat
load ('RSAllData.mat', 'npointsopencl');
timegpu2 = toc(timegpustart2);
disp(['Time for OpenCL file:' num2str(timegpu2)]);

if CompareCudaOpenCL
    disp('COMPARING OUTPUT...');

    compnum=sum(npointscuda,2)==sum(npointsopencl,2);
    resultnum=all(compnum);
    if resultnum 
        disp('Comparing count...PASSED');
    else
        disp('Comparing count...FAIL. DIFFERENT COUNT FOUND');
    end
end
