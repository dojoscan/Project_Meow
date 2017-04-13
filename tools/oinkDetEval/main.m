% Evaluation script for KITTI

path_to_gt_dir = 'C:/Users/Donal/Dropbox/KITTI/data/training/label';
path_to_det_dir = 'C:/Users/Donal/Dropbox/KITTI/output/predictions';
%path_to_gt_dir = 'C:/Master Chalmers/2 year/volvo thesis/code0/training/label';
%path_to_det_dir = 'C:/log_ckpt_thesis/output/predictions';

D = dir([path_to_gt_dir, '\*.txt']);
nr_img = length(D(not([D.isdir])));

classes = {'Car','Pedestrian', 'Cyclist', 'DontCare', 'Van', 'Truck', ...
                                        'Person_sitting', 'Tram', 'Misc'};
class_num = num2cell(1:size(classes,2));
class_dict = containers.Map(classes,class_num);
overlap_thresh = [0.7, 0.5, 0.5, 1, 1, 1, 1, 1, 1];
% read gt and detctions from txt files and store in structs
gt = readLabelsTest(path_to_gt_dir, nr_img, true);
det = readLabelsTest(path_to_det_dir, nr_img, false);

for im_idx = 1:nr_img
    
    % calc iou between all gt and dets
    im_gt = gt(im_idx);
    im_det = det(im_idx);
    bb_gt = extractBbox(im_gt.object);
    bb_det = extractBbox(im_det.object);
    iou = bboxOverlapRatio(bb_det, bb_gt);
    [max_iou, asgn] = max(iou,[],2); % asgn: gt idx assigned to each det
    sort_iou, sort_ind = sort(max_iou); % sort_ind: det idx sorted by iou (ascend)
    sort_asgn = asgn(sort_ind);
    ref_det = zeros(1,im_gt.nr_obj); % det idx assigned to each gt (0 for no assignment)
    for obj_idx = 1:im_det.nr_obj
        ref_det(sort_asgn(obj_idx)) = sort_ind(obj_idx);
        % Need to create labels TP/FP/FN/MC/DC for each det
        if strcmp(im_gt(sort_asgn(obj_idx)).type, 'DontCare')
            
        end
    end
    % Loop over gt
%         if sort_iou(obj_idx) >= overlap_thresh(class_dict(im_det(sort_ind(obj_idx)).type)) % det and gt have iou greater than threshold
%             if strcmp(im_det(sort_ind(obj_idx)).type, im_gt(sort_asgn(obj_idx)).type) % det and gt are same class
%                 
%             else
%                 
%             end
%         end
    
end


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