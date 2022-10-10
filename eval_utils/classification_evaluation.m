% get cell level classification

gt_csv = "\\babyserverdw5\Digital pathology image lib\JHU\Ie-Ming Shih\lymphocytes\0830\FTE301-20X-normal\CellCounter_FTE301-20X-normal.csv";
gt = readtable(gt_csv);
imsize = size(inst_map);
% relabel indices
c1 = table2array(gt(:,4));
c2 = table2array(gt(:,3));
gtind = sub2ind(imsize, c1, c2);
% subset map by indices
gtindL = inst_map(gtind);
% get rid of non-segmented regions
zero_col = gtindL==0;
pred = inst_type(gtindL(~zero_col));
truth = table2array(gt(~zero_col, 1));
% compare pred vs ground truth
comp = [truth pred];
count2 = hist3(single(comp), [5 5]);
acc = sum(pred == truth) / length(pred)

save('FTE301-20X-STIC_class_acc.mat', "truth", "pred", "acc")