% Evaluation distributions of objects in KITTI data-set

function main()

%path_to_gt_dir = 'C:\Users\Donal\Desktop\label';
path_to_gt_dir = 'C:\Users\Donal\Desktop\output\predictions\no_pre_val_pred\val_labels';
%path_to_gt_dir = 'C:/Master Chalmers/2 year/volvo thesis/code0/training/label';

D_gt = dir([path_to_gt_dir, '\*.txt']);
nr_img = length(D_gt(not([D_gt.isdir])));

classes = {'Car','Pedestrian', 'Cyclist', 'DontCare', 'Person_sitting', ...
                                           'Van', 'Truck', 'Tram', 'Misc'};
class_num = num2cell(1:size(classes,2));
class_dict = containers.Map(classes,class_num);
bins = -5:5:100;
nr_bins = size(bins,2)-1;
nr_imp_classes = 3; % car, pedestrian, cyclists
gt_freq = zeros(nr_imp_classes, nr_bins); % gt's in each dist bin

% read gt and detctions from txt files and store in structs
gt = readLabelsTest(nr_img, path_to_gt_dir, D_gt, true);

for im_idx = 1:nr_img
    
    im_gt = gt(im_idx);

    for gt_idx = 1:im_gt.nr_obj
        class = class_dict(im_gt.object(gt_idx).type);
        dist = im_gt.object(gt_idx).distance;
        bin_ind = discretize(dist, bins);
        if class < 4
            gt_freq(class, bin_ind) = gt_freq(class, bin_ind)+1;
        end
    end
end

plotHistMult(gt_freq, bins,'Nr. Of Groundtruths',3);

end

function plotHistMult(data, bins, lbl, nr_methods)
    figure()
    colours = 'cmk';
    for i=1:nr_methods
        histogram('BinEdges',bins,'BinCounts',data(i,:), ...
            'FaceAlpha', 0.1, 'FaceColor', colours(i), 'EdgeColor', colours(i), 'LineWidth', 1.5);
        hold on
    end
    xlabel('Distance From Ego-Vehicle (m)')
    ylabel(lbl)
    legend('Cars','Pedestrians','Cyclists')
    hold off
end

function bbox = extractBbox(data)
    bbox = [extractfield(data, 'x1')', extractfield(data, 'y1')', ...
        extractfield(data, 'x2')' - extractfield(data, 'x1')', ...
        extractfield(data, 'y2')' - extractfield(data, 'y1')'];
end
