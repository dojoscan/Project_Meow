% Evaluation script for KITTI

function mainConf()

path_to_gt_dir = 'C:/Users/Donal/Desktop/output/predictions/no_pre_val_pred/val_labels';
path_to_det_dir_array = {'C:\Users\Donal\Desktop\even_ckpt_val\squeeze', ...
    'C:\Users\Donal\Desktop\even_ckpt_val\res_squeeze', ...
    'C:\Users\Donal\Desktop\even_ckpt_val\forget_squeeze'} ;
%path_to_gt_dir = 'C:/Master Chalmers/2 year/volvo thesis/code0/training/label';
%path_to_det_dir = 'C:/log_ckpt_thesis/output/predictions';
min_height = 25;

D_gt = dir([path_to_gt_dir, '\*.txt']);
nr_img = length(D_gt(not([D_gt.isdir])));

classes = {'Car','Pedestrian', 'Cyclist', 'DontCare', 'Person_sitting', ...
                                           'Van', 'Truck', 'Tram', 'Misc'};
class_num = num2cell(1:size(classes,2));
class_dict = containers.Map(classes,class_num);
overlap_thresh = [0.7, 0.5, 0.5, 0.5, 0.5, 0.7, 1, 1, 1];
confuse_vec = [1, 2, 3, 4, 2, 1, 5, 6, 7];
inc = 0.05;
conf_bins = 0:0.05:1;
conf_centres = inc/2:inc:1;
nr_bins = size(conf_bins,2)-1;
nr_methods = 3;
nr_imp_classes = 3; % car, pedestrian, cyclists
tp_freq = zeros(nr_methods, nr_bins);
det_freq = zeros(nr_methods, nr_bins);

gt = readLabelsTest(nr_img, path_to_gt_dir, D_gt, true);

for method = 1:nr_methods
    
    path_to_det_dir = path_to_det_dir_array{method};
    D_det = dir([path_to_det_dir, '\*.txt']);
    det = readLabelsTest(nr_img, path_to_det_dir, D_det, false);
    
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

        % assign TP and generate distance histogram
        for gt_idx = 1:im_gt.nr_obj
            class = class_dict(im_gt.object(gt_idx).type);
            if class < 4 && sort_label_gt(gt_idx) > 0    % gt has been correctly assigned a det
                sort_label_det(sort_label_gt(gt_idx)) = 3; % set det to TP
            end
        end
        
        [~, idx_rev] = sort(sort_idx);  % reverse sort
        label_det = sort_label_det(idx_rev);
        
        for det_nr =1:im_det.nr_obj
           conf_score = im_det.object(det_nr).conf;
           bin_ind = discretize(conf_score, conf_bins);
           det_freq(method,bin_ind) = det_freq(method,bin_ind)+1;
           if label_det(det_nr) == 3
               tp_freq(method,bin_ind) = tp_freq(method,bin_ind)+1;
           end
        end
    end
end

tp_frac = tp_freq./det_freq;
tp_frac(isnan(tp_frac)) = 0;

for method = 1:nr_methods
    plotHist(tp_frac(method,:), conf_bins, 'Precision', conf_centres)
end

end


function plotHist(data, bins, lbl, conf_centres)
    figure()
    h = histogram('BinEdges',bins,'BinCounts',data);
    hold on
    [curve,~] = fit(conf_centres',data','gauss2');
    plot(curve)
    xlabel('Object Confindence Score')
    ylabel(lbl)
end

function bbox = extractBbox(data)
    bbox = [extractfield(data, 'x1')', extractfield(data, 'y1')', ...
        extractfield(data, 'x2')' - extractfield(data, 'x1')', ...
        extractfield(data, 'y2')' - extractfield(data, 'y1')'];
end
