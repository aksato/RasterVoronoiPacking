scale = 0.01;
fileID = fopen('OpTA001.txt','r');
formatSpec = '%d';
contents = fscanf(fileID,formatSpec);
count = 1;
containersCount = contents(count); count = count+1;
piecesCount = zeros(containersCount,1);
for i=1:containersCount
    piecesCount(i) = contents(count); count = count+1;
end
width = contents(count); count = count+1;
height = contents(count); count = count+1;

%positions = zeros(sum(piecesCount),2);
pieces = cell(sum(piecesCount),1);
pieceCount = 1;
deltax = 0;
for i=1:containersCount
    for j=1:piecesCount(i)
        numNodes = contents(count); count = count+1;
        pieces{pieceCount} = zeros(numNodes,2);
        for k=1:numNodes
            pieces{pieceCount}(k,1) = deltax + contents(count); count = count+1;
            pieces{pieceCount}(k,2) = contents(count); count = count+1;
        end
        pieces{pieceCount} = scale * pieces{pieceCount};
        pieceCount = pieceCount + 1;
    end
    deltax = deltax + width;
end
fclose(fileID);

fileID = fopen('bestSol.pgf','w');
fprintf(fileID,'\\begin{tikzpicture}\n');
fprintf(fileID,'\\draw (0,0) -- (%f, 0) -- (%f,%f) -- (0,%f) -- cycle;\n',containersCount*scale*width,containersCount*scale*width,scale*height,scale*height);
for i=1:size(pieces,1)
    curPiece = pieces{i};
    fprintf(fileID,'\\draw[fill = gray!20]');
    for j=1:size(curPiece,1)
        fprintf(fileID,' (%f,%f) --',curPiece(j,1),curPiece(j,2));
    end
    fprintf(fileID,' cycle;\n');
end
fprintf(fileID,'\\end{tikzpicture}');
fclose(fileID);