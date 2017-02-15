read_dir = 'C:\Users\Donal\Desktop\cifar-10-batches-mat\data_batch_';
save_dir = 'C:\Users\Donal\Dropbox\CIFAR10\Data\';
py_path = 'C:\Users\Donal\Dropbox\CIFAR10\Data\';
all_labels = [];
out_dim = [32 32 3];
image_counter = 0;
for batch_index = 1:5
    batch_data = load([read_dir num2str(batch_index) '.mat']);
    all_labels = vertcat(all_labels,batch_data.labels);
%     for image_index = 1:size(batch_data.data,1)
%         image = batch_data.data(image_index,:);
%         image = imrotate(reshape(image,out_dim),270);
%         imwrite(image,[save_dir 'images\' sprintf('%06d.png',image_counter)])
%         image_counter = image_counter+1;
%     end
end

fid = fopen([save_dir 'labels.txt'],'wt');
for i = 1:size(all_labels,1)
    fprintf(fid,'%s %d\n',strcat(,sprintf('%06d.png',input_index)),all_labels(i));
end
fclose(fid)