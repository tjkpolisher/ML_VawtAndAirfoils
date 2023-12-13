
function [pos,faces,startFace,connecntivity] = load_pos_faces_boundary(path,pathData)

scriptName= char([path, '/script.sh']);
scriptShortName= char([path, '/script_short.sh']);

%copyFile = char([directory,'/nuTilda_orig']);
if ~exist(scriptName, 'file')
    copyFile = '/Users/sahucko/Documents/MATLAB/3/constant/polyMesh/script.sh';
    cmd = char(['cp ',copyFile,' ',path]); 
    system(cmd);
end

if ~exist(scriptShortName, 'file')
    copyFile = '/Users/sahucko/Documents/MATLAB/3/constant/polyMesh/script_short.sh';
    cmd = char(['cp ',copyFile,' ',path]); 
    system(cmd);
end

if nargin == 2
    cmd = char(['source ',scriptName,' ',path,' ',pathData]);
elseif nargin == 1
    cmd = char(['source ',scriptShortName,' ',path]);
end
system(cmd);

fileFaces = char([path,'/faces1']);
faces = importdata(fileFaces) + 1;

filePoints = char([path,'/points1']);
pos = importdata(filePoints);

ix = isnan(faces(:,end));
t_quad = faces(~ix,:) ;
t_tri = faces(ix,1:3) ;

fileStartFace = char([path,'/startFace']);
s = importdata(fileStartFace);
startFace = s.data;
startFace = [0;startFace];

fileOwner = char([path,'/owner1']);
owner = importdata(fileOwner) + 1;

n = max(owner);
connecntivity = zeros(n,6);
allIdx = 1:length(owner);
for i = 1:n
    k = allIdx(owner == i);
    m = length(k);
    connecntivity(i,1:m) = k;
    if mod(i,10000) == 0
        disp([num2str(i),' iterations done out of ',num2str(n)])
    end
   1; 
end

1;


