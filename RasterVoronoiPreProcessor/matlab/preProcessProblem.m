function [nfpSizes, rasterNfps] = preProcessProblem(inputFname, outputfName, scale, method, distScale)

if ~exist('method','var')
    method = 'sqreuclidean';
end
if ~exist('scale','var')
    scale = 1';
end
if ~exist('distScale','var')
    distScale = 1';
end
if ~strcmp(method,'euclidean') && ~strcmp(method,'sqreuclidean') && ~strcmp(method,'manhattan')
    error('rasterizeNfp:InvalidaMethod', 'available methods: euclidean, sqreuclidean and manhattan');
end
tic
fprintf('Loading problem file %s...\n', inputFname);
nfps = loadNofitPolygons( inputFname, scale );
toc
fprintf('Finished loading problem file, rasterizing %d nofit polygons...\n', size(nfps,2));
%rasterNfps = repmat(struct('matrix', [], 'x', 0, 'y', 0), 1, size(nfps,2));
rasterNfps = nfps;
%totalElements = 0;
nfpSizes = zeros(size(nfps,2),2);
fprintf('0.00%%');
prevPercent = 0;
for i = 1:size(nfps,2)
    matrix = rasterizeNfp(nfps(i).polygon, nfps(i).x, nfps(i).y, nfps(i).w, nfps(i).h, method, distScale);
    rasterNfps(i).matrix = matrix;
    nfpSizes(i,:) = [size(matrix,1) size(matrix,2)];
    if prevPercent >= 10
        fprintf('\b\b\b\b\b\b%03.2f%%', 100*i/size(nfps,2));
    else
        fprintf('\b\b\b\b\b%03.2f%%', 100*i/size(nfps,2));
    end
    prevPercent = 100*i/size(nfps,2);
end

fprintf('\nSaving problem data file...\n');
fileID = fopen(fullfile(fileparts(inputFname),outputfName),'w');
fwrite(fileID,size(nfps,2),'integer*4');
for i = 1:size(nfps,2)
    fwrite(fileID,nfpSizes(i,1),'integer*4');
    fwrite(fileID,nfpSizes(i,2),'integer*4');
    fwrite(fileID,rasterNfps(i).x,'integer*4');
    fwrite(fileID,rasterNfps(i).y,'integer*4');
end
for i = 1:size(nfps,2)
    fwrite(fileID,rasterNfps(i).matrix,'integer*4');
end 
fclose(fileID);
fprintf('Finished saving problem data file...\n');
toc

%Update xml
fprintf('Updating xml problem file...\n');
fid = fopen(inputFname,'rt') ;
X = fread(fid);
fclose(fid);
X = char(X.');
% replace string S1 with string S2
Y = strrep(X, '<raster>', strcat('<raster data="',outputfName,'">'));
fid2 = fopen(inputFname,'wt') ;
fwrite(fid2,Y) ;
fclose (fid2) ;

fprintf('Finished pre-processing!\n');
end