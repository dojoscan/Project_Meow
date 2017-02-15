addpath('C:\Users\Donal\Desktop\kitti code')
read_dir = 'C:\Users\Donal\Desktop\kitti code\test\image';
label_dir = 'C:\Users\Donal\Desktop\kitti code\test\label';
save_dir = 'C:\Users\Donal\Dropbox\CIFAR10\Data\test\';
no_images = 50;
out_dim = [32 32];
output_counter = 0;
fid = fopen([save_dir 'labels.txt'],'wt');

for input_index = 0:no_images-1
   image = imread([read_dir '\' sprintf('%06d.png',input_index)]);
   objects = readLabelsTest(label_dir,input_index);
   for obj_idx=1:numel(objects)
      object = objects(obj_idx);
      rect = [object.x1,object.y1,object.x2-object.x1+1,object.y2-object.y1+1];
      image_crop = imcrop(image,rect);
      image_resize = imresize(image_crop, out_dim);
      imwrite(image_resize,[save_dir 'images\' sprintf('%06d.png',output_counter)])
      fprintf(fid,'%d %.2f %.2f %.2f %.2f \n',input_index,rect);
      output_counter = output_counter+1;
   end
end

fclose(fid)