function im_out = drawBox2DMeow(im_in,object,classDict,plotColours)

% show rectangular bounding boxes
pos = [object.x1,object.y1,object.x2-object.x1+1,object.y2-object.y1+1];
im = insertShape(im_in, 'Rectangle',pos,'Color',plotColours(classDict(object.type)),...
        'LineWidth',1);

% draw label
label_text = [sprintf(object.type(1:3)) ', ' num2str(object.conf)];
x = (object.x1+object.x2)/2;
y = object.y1;
im_out = insertText(im, [x max(y-5,40)],label_text,'TextColor','w',...
   'BoxColor','black','AnchorPoint','CenterBottom', 'FontSize',8);

