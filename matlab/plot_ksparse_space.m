box = makebox(3,1);
L = 1.5;

xyplane = Polyhedron('He',[0,0,1,0]) & box;
yzplane = Polyhedron('He',[1,0,0,0]) & box;
xzplane = Polyhedron('He',[0,1,0,0]) & box;
arrows = [zeros(3,6) ; L*[eye(3), -eye(3)]];

clf
hold on;
plot([xyplane], 'color', 'white')
plot([yzplane], 'color', 'white')
plot([xzplane], 'color', 'white')

q = quiver3(arrows(1,:),arrows(2,:),arrows(3,:),arrows(4,:),arrows(5,:),arrows(6,:), 'k', 'linestyle','-');
q.Marker

axis([-L,L,-L,L,-L,L])
axis equal