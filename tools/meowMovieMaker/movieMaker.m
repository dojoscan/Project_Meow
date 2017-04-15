path_to_images = 'C:\Users\Donal\Dropbox\KITTI\data\testing\image\';
path_to_dets = 'C:\Users\Donal\Dropbox\KITTI\output\predictions\';
conf_thresh = 0.5;
im_size = [375, 1242];

% Create dictionary for plotting
classes = {'Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc','DontCare'};
class_num = num2cell(1:size(classes,2));
class_dict = containers.Map(classes,class_num);
plot_colours = ['y','m','c','r','g','b','w','k','w','k'];

% Directories
image_list = [dir([path_to_images, '\*.png']);dir([path_to_images, '\*.jpg'])];
no_img = length(D(not([image_list.isdir])));
det_list = dir([path_to_dets, '\*.txt']);

vid = VideoWriter('guau.avi');
vid.FrameRate = 8;
open(vid)

for img_idx = 1:no_img
    
    im_data = image_list(img_idx);
    det_data = det_list(img_idx);
    im = imread([im_data.folder '/' im_data.name]);
    objects = readLabelsMeow([det_data.folder '/' det_data.name]);

    for obj_idx=1:numel(objects)
        object = objects(obj_idx);
        if object.conf >= conf_thresh
            im = drawBox2DMeow(im,objects(obj_idx),class_dict,plot_colours);
        end
    end
    
    im = imresize(im,im_size);
    frame = im2frame(im);
    writeVideo(vid, frame);
    
end

close(vid)

