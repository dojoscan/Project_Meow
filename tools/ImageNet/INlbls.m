folders = dir('\Users\Donal\Desktop\Thesis\Data\TinyImageNet\training\image');

labels = [];
for syn = 1:6
   image_list = dir([folders(syn+2).folder '/' folders(syn+2).name]);
   labels = vertcat(labels,ones(length(image_list(not([image_list.isdir]))),1).*(syn-1)); 
end

fid=fopen('labels.txt','wt');
fprintf(fid,'%d\n',labels);
fclose(fid);