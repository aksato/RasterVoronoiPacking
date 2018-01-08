function rasterVoronoiNfp = rasterizeNfp(curNfp, x0, y0, w, h, method, distScale)

curX = curNfp(:,1);
curY = curNfp(:,2);
%RP = [-floor(min(curX)) -floor(min(curY))];
RP = [x0 y0];
curNfp(:,1) = curX + RP(1);
curNfp(:,2) = curY + RP(2);
%rasterNfp = zeros(ceil(max(curX)) - floor(min(curX)) + 1, ceil(max(curY)) - floor(min(curY)) + 1);
rasterNfp = zeros(w, h);
for j = 1:size(rasterNfp,2)
    for i = 1:size(rasterNfp,1)
        [in,on] = inpolygon(i-1,j-1,curNfp(:,1),curNfp(:,2));
        rasterNfp(i,j) = in & ~on;
    end
end

rasterVoronoiNfp = padarray(rasterNfp,[1 1]);
rasterVoronoiNfp = 1 - rasterVoronoiNfp;
if strcmp(method,'sqreuclidean') || strcmp(method,'euclidean')
    rasterVoronoiNfp = bwdist(rasterVoronoiNfp);
else
    rasterVoronoiNfp = bwdist(rasterVoronoiNfp,'cityblock');
end
rasterVoronoiNfp(1,:) = [];
rasterVoronoiNfp(:,1) = [];
rasterVoronoiNfp(size(rasterVoronoiNfp,1),:) = [];
rasterVoronoiNfp(:,size(rasterVoronoiNfp,2)) = [];
if strcmp(method,'sqreuclidean')
    rasterVoronoiNfp = rasterVoronoiNfp.^2;
elseif strcmp(method,'euclidean')
    rasterVoronoiNfp = round(distScale*rasterVoronoiNfp);
else
    rasterVoronoiNfp = round(rasterVoronoiNfp);
end

end