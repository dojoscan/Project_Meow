data = synsets(1:1000);
Afields = fieldnames(data);
Acell = struct2cell(data);
sz = size(Acell);
Acell = reshape(Acell, sz(1), []);
Acell = Acell';
Acell = sortrows(Acell, 2);
Acell = reshape(Acell', sz);
data_sorted = cell2struct(Acell, Afields, 1);

labels = [];
for syn = 1:1000
   labels = horzcat(labels,data_sorted(syn).ILSVRC2012_ID*ones(1,data_sorted(syn).num_train_images)); 
end

fid=fopen('labels.txt','wt');
fprintf(fid,'%d\n',labels);
fclose(fid);