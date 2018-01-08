function plotNfp( coords )
    %fill(coords(:,1),coords(:,2),'g')
    coordsTemp = coords;
    coordsTemp(end+1,:) =  coordsTemp(1,:);
    plot(coordsTemp(:,1),coordsTemp(:,2))
end