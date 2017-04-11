function image = readLabelsTest(label_dir, nr_img, gtBool)

image = [];

for img_idx = 1:nr_img
    
    % parse input file
    fid = fopen(sprintf('%s/%06d.txt',label_dir,img_idx-1),'r');
    if gtBool
        C   = textscan(fid,'%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f','delimiter', ' ');
    else
        C   = textscan(fid,'%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f','delimiter', ' ');
    end
    fclose(fid);

    % for all objects do
    objects = [];
    for o = 1:numel(C{1})

        % extract label, truncation, occlusion : REMOVE TRUNC, OCC, ADD CONF
        lbl = C{1}(o);                   % for converting: cell -> string
        objects(o).type       = lbl{1};  % 'Car', 'Pedestrian', ...

        % extract 2D bounding box in 0-based coordinates
        objects(o).x1 = C{5}(o); % left
        objects(o).y1 = C{6}(o); % top
        objects(o).x2 = C{7}(o); % right
        objects(o).y2 = C{8}(o); % bottom
        
        if gtBool
            objects(o).truncation = C{2}(o); % truncated pixel ratio ([0..1])
            objects(o).occlusion  = C{3}(o); % 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown7
        else
            objects(o).conf = C{16}(o); % confidence score
        end
    end
    
    image(img_idx).object = objects;
    
end
