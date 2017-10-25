function [nfpSizes, rasterNfps] = preProcessProblem(inputFname, outputfName)
tic
nfps = loadNofitPolygons( inputFname );
toc
rasterNfps = repmat(struct('matrix', [], 'x', 0, 'y', 0), 1, size(nfps,2));
%totalElements = 0;
nfpSizes = zeros(size(nfps,2),2);
for i = 1:size(nfps,2)
    [matrix, rp] = rasterizeNfp(nfps{i});
    rasterNfps(i) = struct('matrix', matrix, 'x', rp(1), 'y', rp(2));
    nfpSizes(i,1) = size(matrix,1); nfpSizes(i,2) = size(matrix,2);
end

fileID = fopen(outputfName,'w');
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
toc

end