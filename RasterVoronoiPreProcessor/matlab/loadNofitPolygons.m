function nfps = loadNofitPolygons( filename )

    c=xml2struct(filename);
    numPolygons = size(c.nesting.polygons.polygon,2);

    %Count only nofit polygons
    nonNfpPolCount = 0;
    for i=1:numPolygons
        polName = c.nesting.polygons.polygon{i}.Attributes.id;
        nonNfpPolCount = nonNfpPolCount + isempty(strfind(polName,'nfpPolygon'));
    end
    numNfpPolygons = numPolygons - nonNfpPolCount;

    nfps = cell(1,numNfpPolygons);
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
        nfps{nfpId + 1} = curNfp;
        nfpId = nfpId + 1;
    end

end