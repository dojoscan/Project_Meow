username = 'dojoscan';
accesskey = '5f697d45aba2d38f80c85dbc823ec354dc69d72d';
save_dir = 'C:\Users\Donal\Desktop\Thesis\Data\ImageNet\images';
fid = fopen('im_net_ids.txt');
tline = fgets(fid);

while ischar(tline)

    url = ['http://www.image-net.org/download/synset?wnid=' tline(1:9) '&username=' username '&accesskey=' accesskey '&release=latest&src=stanford'];
    untar(url,[save_dir '\' tline(1:9)])
    tline = fgets(fid);
    
end
