matnm ="C:\Users\labuser\PycharmProjects\hover_net\out\Lymphocyte\pannuke_lymph_50_fix\mat\FTE301-20X-normal.mat";
js = "C:\Users\labuser\PycharmProjects\hover_net\out\Lymphocyte\pannuke_lymph_50_fix\json\FTE301-20X-normal.json";
val = jsondecode(fileread(js));

hovernet_init_pred = load(matnm);
xy=hovernet_init_pred.inst_centroid;
ct = csv{:,1};
idx =[];
bwL = hovernet_init_pred.inst_map;
bwL2 = hovernet_init_pred.inst_map;
% iterate annotated one
for i=1:length(xy)
    idxn = hovernet_init_pred.inst_map(xy(i,2),xy(i,1));
    idx(end+1)=idxn;
    if idxn~=0
        bwL(bwL==idxn)=100000;
    end
end
bwL(bwL~=100000)=0;

%label with manual cell type annotation 
bwL3 = bwL2.*bwL;
inst_map = bwlabel(bwL3);
%pair new labeled mask to annotaiton order
newidx = [];
for i=1:length(xy)
    newidx(end+1)=inst_map(xy(i,2),xy(i,1));
end
%assign manual cell type to new labeled mask
newtypes = zeros([max(inst_map(:)),1]);
for i=1:length(newidx)
    if newidx(i)~=0
        newtypes(newidx(i))=ct(i);
    end
end
type_map = inst_map;
for i=1:max(inst_map(:))
    type_map(inst_map==i)=newtypes(i);
end
matname = fullfile(dst,"FTE302-20X-STIC.mat");

save(matname,'type_map','inst_map');