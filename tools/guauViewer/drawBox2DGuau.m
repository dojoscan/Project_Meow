function drawBox2DGuau(h,object,classDict,plotColours)

axes(h)
% show rectangular bounding boxes
pos = [object.x1,object.y1,object.x2-object.x1+1,object.y2-object.y1+1];
rectangle('Position',pos,'EdgeColor',plotColours(classDict(object.type)),...
        'LineWidth',1)
%rectangle('Position',pos,'EdgeColor','b', 'parent', h.axes)

% draw label
label_text = [sprintf(object.type(1:3)) ', ' num2str(object.conf)];
x = (object.x1+object.x2)/2;
y = object.y1;
text(x,max(y-5,40),label_text,'color','w',...
   'BackgroundColor','k','HorizontalAlignment','center',...
   'VerticalAlignment','bottom','FontWeight','bold',...
   'FontSize',7);

