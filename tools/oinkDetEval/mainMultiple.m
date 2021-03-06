% Evaluation script for KITTI for the predictions from multiple runs

function mainMultiple()

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
dist_bins = 0:5:100;
nr_bins = size(dist_bins,2)-1;
nr_methods = size(path_to_det_dir_array,2);
nr_imp_classes = 3; % car, pedestrian, cyclists
gt_freq = zeros(nr_methods, nr_imp_classes, nr_bins); % gt's in each dist bin
tp_freq = gt_freq;    % nr. tp in each dist bin
iou_av = gt_freq;     % average iou in each bin
conf_av = gt_freq;    % average conf in each bin

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

        [~, idx_rev] = sort(sort_idx);  % reverse sort
        label_det = sort_label_det(idx_rev);

        % assign TP and generate distance histogram
        for gt_idx = 1:im_gt.nr_obj
            class = class_dict(im_gt.object(gt_idx).type);
            dist = im_gt.object(gt_idx).distance;
            bin_ind = discretize(dist, dist_bins);
            if class < 4
                gt_freq(method, class, bin_ind) = gt_freq(method, class, bin_ind)+1;
                if sort_label_gt(gt_idx) > 0    % gt has been correctly assigned a det
                    sort_label_det(sort_label_gt(gt_idx)) = 3; % set det to TP
                    tp_freq(method, class, bin_ind) = tp_freq(method, class, bin_ind)+1;
                    conf_av(method, class, bin_ind) = conf_av(method, class, bin_ind) + im_det.object(sort_idx(sort_label_gt(gt_idx))).conf;
                    iou_av(method, class, bin_ind) = iou_av(method, class, bin_ind) + sort_iou(sort_label_gt(gt_idx));
                end
            end
        end
    end
end

conf_av = conf_av./tp_freq;
iou_av = iou_av./tp_freq;
tp_frac = tp_freq./gt_freq;
    
conf_av(isnan(conf_av)) = 0;
iou_av(isnan(iou_av)) = 0;
tp_frac(isnan(tp_frac)) = 0;

plotHist(sum(gt_freq,2), dist_bins, 'Nr. of Groundtruths');
plotHistMult(sum(tp_frac,2)/3, dist_bins, 'Overall Recall', nr_methods);
plotHistMult(sum(conf_av,2)/3, dist_bins, 'Average Confidence of All TP Detections', nr_methods);
plotHistMult(sum(iou_av,2)/3, dist_bins, 'Average IOU of All TP Detections', nr_methods);
    
    for class = 1:3
        plotHist(gt_freq(:,class,:), dist_bins, ['Nr. of ' classes(class) ' Groundtruths']);
        plotHistMult(tp_frac(:,class,:), dist_bins, [classes(class) ' Recall'], nr_methods);
        plotHistMult(conf_av(:,class,:), dist_bins, ['Average Confidence of TP ' classes(class) ' Detections'], nr_methods);
        plotHistMult(iou_av(:,class,:), dist_bins, ['Average IOU of TP ' classes(class) '  Detections'], nr_methods);
    end

end


function plotHist(data, dist_bins, lbl)
    figure()
    histogram('BinEdges',dist_bins,'BinCounts',squeeze(data(1,:,:)));
    xlabel('Distance From Ego-Vehicle (m)')
    ylabel(lbl)
end

function plotHistMult(data, dist_bins, lbl, nr_methods)
    figure()
    colours = 'rgb';
    for i=1:nr_methods
        histogram('BinEdges',dist_bins,'BinCounts',squeeze(data(i,:,:)), ...
            'FaceAlpha', 0.1, 'FaceColor', colours(i), 'EdgeColor', colours(i), 'LineWidth', 1.5);
        hold on
    end
    xlabel('Distance From Ego-Vehicle (m)')
    ylabel(lbl)
    legend('SqueezeDet','Residual SqueezeDet','Gated SqueezeDet')
    hold off
end

function bbox = extractBbox(data)
    bbox = [extractfield(data, 'x1')', extractfield(data, 'y1')', ...
        extractfield(data, 'x2')' - extractfield(data, 'x1')', ...
        extractfield(data, 'y2')' - extractfield(data, 'y1')'];
end
