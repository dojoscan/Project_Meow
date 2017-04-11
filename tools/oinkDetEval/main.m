% Evaluation script for KITTI

%path_to_gt_dir = 'C:/Users/Donal/Dropbox/KITTI/data/training/label';
path_to_gt_dir = 'C:/Master Chalmers/2 year/volvo thesis/code0/training/label';
path_to_det_dir = 'C:/log_ckpt_thesis/output/predictions';
%path_to_det_dir = 'C:/Users/Donal/Dropbox/KITTI/output/predictions';

D = dir([path_to_gt_dir, '\*.txt']);
nr_img = length(D(not([D.isdir])));

% run readLabelsTest
im_gt = readLabelsTest(path_to_gt_dir, nr_img, true);
im_det = readLabelsTest(path_to_det_dir, nr_img, false);

for im_idx = 1:nr_img
    
    % sort det by conf
    objs = im_det(im_idx).object;
    [~,sx]=sort([objs.conf]);
    objs=objs(sx);
    im_det(im_idx).object = objs;
    
    % split by class
    for index = 1:im_det(im_idxnr_obj
        
        
    end
    
end

% discard don't care detection

% assign det to gt


