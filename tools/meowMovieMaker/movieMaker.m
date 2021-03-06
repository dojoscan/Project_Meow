path_to_images = 'C:\Users\Donal\Desktop\Thesis\Data\CrowdAI\images\';
path_to_dets = 'C:\Users\Donal\Desktop\Thesis\Data\CrowdAI\predictions\no_pre_fs\';
conf_thresh = 0.75;
%im_size = [375, 1242];
frame_rate = 10;

% Create dictionary for plotting
classes = {'Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc','DontCare'};
class_num = num2cell(1:size(classes,2));
class_dict = containers.Map(classes,class_num);
plot_colours = {'yellow','magneta','cyan','red','green','blue','white','black','white','black'};

% Directories
image_list = [dir([path_to_images, '\*.png']);dir([path_to_images, '\*.jpg'])];
no_img = length(image_list(not([image_list.isdir])));
det_list = dir([path_to_dets, '\*.txt']);

vid = VideoWriter('C:\Users\Donal\Desktop\forget_nopre_100k_crowdAI.avi');
vid.FrameRate = frame_rate;
open(vid)

for img_idx = 1:no_img
    
    im_data = image_list(img_idx);
    det_data = det_list(img_idx);
    im = imread([path_to_images im_data.name]);
    %im = imresize(im,im_size);
    im = im(300:900,:,:);
    objects = readLabelsMeow([path_to_dets det_data.name]);

    for obj_idx=1:numel(objects)
        object = objects(obj_idx);
        if object.conf >= conf_thresh
            im = drawBox2DMeow(im,objects(obj_idx),class_dict,plot_colours);
        end
    end
    
    frame = im2frame(im);
    writeVideo(vid, frame);
    
end

close(vid)

