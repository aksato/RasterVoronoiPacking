function nfps = loadNofitPolygons( filename, scale )

    c=xml2struct(filename);
    numPolygons = size(c.nesting.polygons.polygon,2);

    %Count only nofit polygons
    nonNfpPolCount = 0;
    for i=1:numPolygons
        polName = c.nesting.polygons.polygon{i}.Attributes.id;
        nonNfpPolCount = nonNfpPolCount + isempty(strfind(polName,'nfpPolygon'));
    end
    numNfpPolygons = numPolygons - nonNfpPolCount;

    %nfps = cell(1,numNfpPolygons);
    nfps = repmat(struct('polygon', [], 'matrix', [], 'x', 0, 'y', 0, 'w', 0, 'h', 0), 1, numNfpPolygons);
    nfpId = 0;
    for i=1:numPolygons
        polName = c.nesting.polygons.polygon{i}.Attributes.id;
        if isempty(strfind(polName,'nfpPolygon'))
            continue;
        end
        polygon = c.nesting.polygons.polygon{i}.lines.segment;
        curNfp = zeros(size(polygon,2),2);
        for j = 1:size(polygon,2)
            curNfp(j,1) = str2double(polygon{j}.Attributes.x0);
            curNfp(j,2) = str2double(polygon{j}.Attributes.y0);
        end
        curNfp = scale*curNfp;
        %nfps{nfpId + 1} = curNfp;
        x0 = str2double(c.nesting.raster.rnfp{nfpId + 1}.resultingImage.Attributes.x0);
        y0 = str2double(c.nesting.raster.rnfp{nfpId + 1}.resultingImage.Attributes.y0);
        w = str2double(c.nesting.raster.rnfp{nfpId + 1}.resultingImage.Attributes.width);
        h = str2double(c.nesting.raster.rnfp{nfpId + 1}.resultingImage.Attributes.height);
        nfps(nfpId + 1) = struct('polygon', curNfp,'matrix', [], 'x', x0, 'y', y0, 'w', w, 'h', h);
        nfpId = nfpId + 1;
    end

end