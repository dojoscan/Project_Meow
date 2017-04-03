username = 'dojoscan';
accesskey = '5f697d45aba2d38f80c85dbc823ec354dc69d72d';
save_dir = 'C:\Users\Donal\Desktop\Thesis\ImageNet\images';
fid = fopen('C:\Users\Donal\Desktop\Thesis\ImageNet\im_net_ids.txt');
tline = fgets(fid);
former_nr_images = 0;
index = 0;
labels = [];

while ischar(tline)

    url = ['http://www.image-net.org/download/synset?wnid=' tline(1:9) '&username=' username '&accesskey=' accesskey '&release=latest&src=stanford'];
    untar(url,save_dir)
    
    D = dir([save_dir, '\*.JPEG']);
    total_nr_images = length(D(not([D.isdir])));
    nr_images = total_nr_images-former_nr_images;
    former_nr_images = total_nr_images;
    
    labels = vertcat(labels, ones(nr_images,1)*index);
    
    tline = fgets(fid);
    index = index + 1;
    
end

% TODO: split val/training

fid = fopen('C:\Users\Donal\Desktop\Thesis\ImageNet\class.txt','wt');
fprintf(fid,'%d\n',labels);
fclose(fid);