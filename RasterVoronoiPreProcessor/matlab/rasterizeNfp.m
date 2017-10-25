function [rasterVoronoiNfp, RP] = rasterizeNfp(curNfp)

    curX = curNfp(:,1);
    curY = curNfp(:,2);
    RP = [-floor(min(curX)) -floor(min(curY))];
    curNfp(:,1) = curX + RP(1);
    curNfp(:,2) = curY + RP(2);
    rasterNfp = zeros(ceil(max(curX)) - floor(min(curX)) + 1, ceil(max(curY)) - floor(min(curY)) + 1);
    for j = 1:size(rasterNfp,2)
        for i = 1:size(rasterNfp,1)
            [in,on] = inpolygon(i-1,j-1,curNfp(:,1),curNfp(:,2));
            rasterNfp(i,j) = in & ~on;
        end
    end

    rasterVoronoiNfp = padarray(rasterNfp,[1 1]);
    rasterVoronoiNfp = 1 - rasterVoronoiNfp;
    rasterVoronoiNfp = bwdist(rasterVoronoiNfp);
    rasterVoronoiNfp(1,:) = [];
    rasterVoronoiNfp(:,1) = [];
    rasterVoronoiNfp(size(rasterVoronoiNfp,1),:) = [];
    rasterVoronoiNfp(:,size(rasterVoronoiNfp,2)) = [];
    rasterVoronoiNfp = rasterVoronoiNfp.^2;

end