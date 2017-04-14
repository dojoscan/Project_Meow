% Evaluation script for KITTI

% TO DO
% check detection height!

path_to_gt_dir = 'C:/Users/Donal/Dropbox/KITTI/data/training/label';
path_to_det_dir = 'C:/Users/Donal/Desktop/output/predictions';
%path_to_gt_dir = 'C:/Master Chalmers/2 year/volvo thesis/code0/training/label';
%path_to_det_dir = 'C:/log_ckpt_thesis/output/predictions';
min_height = 25;

D = dir([path_to_gt_dir, '\*.txt']);
nr_img = length(D(not([D.isdir])));

classes = {'Car','Pedestrian', 'Cyclist', 'DontCare', 'Person_sitting', ...
                                           'Van', 'Truck', 'Tram', 'Misc'};
class_num = num2cell(1:size(classes,2));
class_dict = containers.Map(classes,class_num);
overlap_thresh = [0.7, 0.5, 0.5, 0.5, 0.5, 0.7, 1, 1, 1];
confuse_vec = [1, 2, 3, 4, 2, 1, 5, 6, 7];

% read gt and detctions from txt files and store in structs
gt = readLabelsTest(path_to_gt_dir, nr_img, true);
det = readLabelsTest(path_to_det_dir, nr_img, false);

img_info = []; % store iou and DC/MC/FP/TP assignment

for im_idx = 1:nr_img
    
    im_gt = gt(im_idx);
    im_det = det(im_idx);
    % calc iou between all gt and dets
    bb_gt = extractBbox(im_gt.object);
    bb_det = extractBbox(im_det.object);
    iou = bboxOverlapRatio(bb_det, bb_gt);
    
    [max_iou, asgn] = max(iou,[],2); % asgn: gt idx assigned to each det
    [sort_iou, sort_idx] = sort(max_iou); % sort_idx: det idx sorted by iou (ascend)
    sort_asgn = asgn(sort_idx);
    
    sort_label_gt = zeros(im_gt.nr_obj,1); % det idx assigned to each gt (0 for no assignment)
    sort_label_det = 2*ones(im_det.nr_obj,1); % DC=0/MC=1/FP=2/TP=3 for each det
    
    for det_idx = 1:im_det.nr_obj
        det_class = im_det.object(sort_idx(det_idx)).type;
        gt_class = im_gt.object(sort_asgn(det_idx)).type;
        if sort_iou(det_idx) >= overlap_thresh(class_dict(gt_class)) % check if iou is greater than threshold for gt class
            if strcmp(gt_class,det_class)   % predicted correct class
                sort_label_gt(sort_asgn(det_idx)) = det_idx;
            elseif strcmp(gt_class,'DontCare') || confuse_vec(class_dict(gt_class))...
                   == confuse_vec(class_dict(det_class)) || bb_det(sort_idx(det_idx),4) ...
                   -bb_det(sort_idx(det_idx),2) < min_height  % gt is DC or confused or too small
                sort_label_det(det_idx) = 0;    % not incl. in eval.
            else
                sort_label_det(det_idx) = 1;    % misclassification (subclass of FP)
            end
        end
    end
    
    for gt_idx = 1:im_gt.nr_obj
        if sort_label_gt(gt_idx) > 0    % gt has been correctly assigned a det
            sort_label_det(sort_label_gt(gt_idx)) = 3; % set det to TP
        end
    end
    
    % store iou and det labels in struct
    [~, idx_rev] = sort(sort_idx);  % reverse sort
    img_info(im_idx).lbl = sort_label_det(idx_rev);
    img_info(im_idx).iou = max_iou;
    
end

% calculate precision and recall


function bbox = extractBbox(data)
    bbox = [extractfield(data, 'x1')', extractfield(data, 'y1')', ...
        extractfield(data, 'x2')' - extractfield(data, 'x1')', ...
        extractfield(data, 'y2')' - extractfield(data, 'y1')'];
end

    % sort det by conf
%     objs = im_det(im_idx).object;
%     [~,sx]=sort([objs.conf]);
%     objs=objs(sx);
%     im_det(im_idx).object = objs;