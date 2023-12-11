function [p,t,e,n1,n2]=pmesh_inside_1_fast(pv_bdr,pv_in1,hmax,nref,plotidx)

p=[];
for i=1:size(pv_bdr,1)-1
  pp=pv_bdr(i:i+1,:);
  L=sqrt(sum(diff(pp,[],1).^2,2));
  if L>hmax
    n=ceil(L/hmax);
    pp=interp1([0,1],pp,(0:n)/n);
  end
  p=[p;pp(1:end-1,:)];
end
n1 = size(p,1);

for i=1:size(pv_in1,1)-1
  pp=pv_in1(i:i+1,:);
  L=sqrt(sum(diff(pp,[],1).^2,2));
  if L>hmax
    n=ceil(L/hmax);
    pp=interp1([0,1],pp,(0:n)/n);
  end
  p=[p;pp(1:end-1,:)];
end
n2 = size(p,1);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while 1
  t=delaunayn(p);
  t=removeoutsidetris(p,t,pv_bdr);
  t=removeoutsidetris_inside(p,t,pv_in1);
  if strcmp(plotidx,'on')
      %tplot(p,t)
  end
  area=triarea(p,t);
  [~,idx]=sort(area);
  maxarea = area(idx(end));
  
  %[maxarea,ix]=max(area);
%   if maxarea<hmax^2/2, break; end
%   pc=circumcenter(p(t(ix,:),:));
%   p(end+1,:)=pc;

  if maxarea<hmax^2/2, break; end
  ix = idx(end-9:end);
  pc=circumcenter_fast(p(t(ix,1),:),p(t(ix,2),:),p(t(ix,3),:));
  p(end+1:end+10,:)=pc;

end

for iref=1:nref
  p=[p;edgemidpoints(p,t)];
  t=delaunayn(p);
  t=removeoutsidetris(p,t,pv_bdr);
  t=removeoutsidetris_inside(p,t,pv_in1);
  if strcmp(plotidx,'on')
      %tplot(p,t)
  end
end

e=boundary_nodes(t);

function pmid=edgemidpoints(p,t)
  
pmid=[(p(t(:,1),:)+p(t(:,2),:))/2;
      (p(t(:,2),:)+p(t(:,3),:))/2;
      (p(t(:,3),:)+p(t(:,1),:))/2];
pmid=unique(pmid,'rows');

function a=triarea(p,t)

d12=p(t(:,2),:)-p(t(:,1),:);
d13=p(t(:,3),:)-p(t(:,1),:);
a=abs(d12(:,1).*d13(:,2)-d12(:,2).*d13(:,1))/2;

function t=removeoutsidetris(p,t,pv)

pmid=(p(t(:,1),:)+p(t(:,2),:)+p(t(:,3),:))/3;
isinside=inpolygon(pmid(:,1),pmid(:,2),pv(:,1),pv(:,2));
t=t(isinside,:);

function t=removeoutsidetris_inside(p,t,pv)

pmid=(p(t(:,1),:)+p(t(:,2),:)+p(t(:,3),:))/3;
isinside=inpolygon(pmid(:,1),pmid(:,2),pv(:,1),pv(:,2));
t=t(~isinside,:);

function pc=circumcenter(p)

dp1=p(2,:)-p(1,:);
dp2=p(3,:)-p(1,:);

mid1=(p(1,:)+p(2,:))/2;
mid2=(p(1,:)+p(3,:))/2;
  
s=[-dp1(2),dp2(2);dp1(1),-dp2(1)]\[-mid1+mid2]';
pc=mid1+s(1)*[-dp1(2),dp1(1)];


function allpc=circumcenter_fast(p1,p2,p3)

n = length(p1);
allpc = zeros(n,2);   
for q=1:n

    dp1=p2(q,:)-p1(q,:);
    dp2=p3(q,:)-p1(q,:);

    mid1=(p1(q,:)+p2(q,:))/2;
    mid2=(p1(q,:)+p3(q,:))/2;

    
    
    s=[-dp1(2),dp2(2);dp1(1),-dp2(1)]\[-mid1+mid2]';
    pc=mid1+s(1)*[-dp1(2),dp1(1)];
    allpc(q,:) = pc;
end