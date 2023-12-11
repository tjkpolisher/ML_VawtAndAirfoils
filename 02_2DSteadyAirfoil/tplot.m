function tplot(p,t,u)
%clf
if nargin<3
  patch('vertices',p,'faces',t,'facecol',[.9,0.9,.9],'edgecol','k');
else
  patch('vertices',p,'faces',t,'facevertexcdata',u,'facecol','interp', ...
        'edgecol','none');
  colorbar
end
set(gcf,'renderer','zbuffer');
axis equal
%drawnow