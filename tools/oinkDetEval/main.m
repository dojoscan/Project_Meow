% Evaluation script for KITTI

path_to_gt_dir = 'C:/Users/Donal/Dropbox/KITTI/data/training/label';
path_to_det_dir = 'C:/Users/Donal/Dropbox/KITTI/output/predictions';
%path_to_gt_dir = 'C:/Master Chalmers/2 year/volvo thesis/code0/training/label';
%path_to_det_dir = 'C:/log_ckpt_thesis/output/predictions';

D = dir([path_to_gt_dir, '\*.txt']);
nr_img = length(D(not([D.isdir])));

% read gt and detctions from txt files
gt = readLabelsTest(path_to_gt_dir, nr_img, true);
det = readLabelsTest(path_to_det_dir, nr_img, false);

for im_idx = 1:nr_img
    
    
    % sort det by conf
%     objs = im_det(im_idx).object;
%     [~,sx]=sort([objs.conf]);
%     objs=objs(sx);
%     im_det(im_idx).object = objs;
    
    % calc iou between all gt and dets
    im_gt = gt(im_idx);
    im_det = det(im_idx);
    bb_gt = extractBbox(im_gt.object);
    bb_det = extractBbox(im_det.object);
    iou = bboxOverlapRatio(bb_det, bb_gt);
    % assign det to gt
    [max_iou, asgn] = max(iou,[],2);
    sort_iou, sort_ind = sort(max_iou);
    sort_asgn = asgn(sort_ind);
    ref_det = -1*ones(1,im_gt.nr_obj);
    for obj_idx = 1:im_det.nr_obj
        if strcmp(im_det(obj_idx).type, 'Car')
            
        elseif strcmp(im_det(obj_idx).type, 'Car')
        % check if ped/cyc
        end
    end
end


function bbox = extractBbox(data)
    bbox = [extractfield(data, 'x1')', extractfield(data, 'y1')', ...
        extractfield(data, 'x2')' - extractfield(data, 'x1')', ...
        extractfield(data, 'y2')' - extractfield(data, 'y1')'];
end

