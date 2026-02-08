clc
clear
close all

addpath('/Users/liyang/Documents/Study/NUS/Forest Resilience/Code/matlab tools/');
what_data = 'AC1 sudden drop Forzieri 0.5 th02 1bpstep air';
what_data_historical = 'AC1 sudden drop Forzieri 0.5 th02 1bpstep air historical';
what_data_S2 = 'AC1 sudden drop Forzieri 0.5 th02 1bpstep air scenario2';
Figout_word = 'Forzieri 0.5 th02 1bpstep air scenario2';
EVI_word = 'Forzieri 0.5 th02'

root_dir = '/Users/liyang/Documents/Study/';
data_dir = sprintf('%sNUS/Forest Resilience/Results/%s/',root_dir,what_data);
data_dir_historical = sprintf('%sNUS/Forest Resilience/Results/%s/',root_dir,what_data_historical);
data_dir_S2 = sprintf('%sNUS/Forest Resilience/Results/%s/',root_dir,what_data_S2);

%% Climate zone https://www.gloh2o.org/koppen/
Koppen_dir = sprintf('%sData/koppen_geiger_tif/1991_2020/koppen_geiger_0p1.tif',root_dir);
Koppen = imread(Koppen_dir);
Koppen = double(Koppen);
Koppen(Koppen==0)=NaN;

Koppens_1 = imresize(Koppen, [3600, 7200], 'nearest');

%% AC1_multiyr_mean
AC1_multiyr_mean_dir = sprintf('%sNUS/Forest Resilience/Data/Global AC1/%s/AC1_multiyr_mean.mat',root_dir,EVI_word);
AC1_multiyr_mean = loadMatData(AC1_multiyr_mean_dir);

AC1_multiyr_mean_S2_dir = sprintf('%sNUS/Forest Resilience/Results/AC1/Scenarios prediction/2 degree/AC1_prediction_S2.mat',root_dir);
AC1_multiyr_mean_S2 = loadMatData(AC1_multiyr_mean_S2_dir);

%%
Twindow = '1bpstep';
years = arrayfun(@num2str, 2002:2015, 'UniformOutput', false);

layers = length(years);
deltaEVI_global_all  = NaN(3600,7200,layers);
postprcp_global_all  = NaN(3600,7200,layers);
postairt_global_all  = NaN(3600,7200,layers);
postVPD_global_all   = NaN(3600,7200,layers);
postdnrad_global_all = NaN(3600,7200,layers);
postsm_global_all    = NaN(3600,7200,layers);

postprcp_global_all_S2  = NaN(3600,7200,layers);
postairt_global_all_S2  = NaN(3600,7200,layers);
postVPD_global_all_S2   = NaN(3600,7200,layers);
postdnrad_global_all_S2 = NaN(3600,7200,layers);
postsm_global_all_S2    = NaN(3600,7200,layers);

LC_multiyear = [];
for i = 1:length(years)
    year_cur = str2double(years{i});

    %% 
    LC_cur_dir = sprintf('%sData/MCD12C1/MCD12C1_01_23_mat/LC_%d.mat',root_dir,year_cur);
    LC = loadMatData(LC_cur_dir);
    LC_reclassified = NaN(size(LC));

    LC_reclassified (LC==1 )=101;   % ENF
    LC_reclassified (LC==2 )=102;   % EBF
    LC_reclassified (LC==3 )=103;   % DNF
    LC_reclassified (LC==4 )=104;   % DBF
    LC_reclassified (LC==5 )=105;   % MF
    LC_reclassified (LC==8 | LC==9)=106; % Woody Savanna & Savanna
    LC_reclassified (LC==6 | LC==7 | LC==10)=107;        % Grassland
    LC_reclassified (LC==12| LC==14)=108;   % Cropland

    LC_multiyear = cat(3,LC_multiyear,LC_reclassified);

    for row = 0:9
        for col = 0:9           
            EVI_cur_dir = sprintf('%s%s spatial/delta EVI/%d/delta_EVI_%s_%d_%d_%d.mat',data_dir,Twindow,year_cur,Twindow,year_cur,row,col);
            deltaEVI_seg = loadMatData(EVI_cur_dir);
            prcp_cur_dir = sprintf('%spost prcp spatial/%d/post_prcp_%s_%d_%d_%d.mat',data_dir_historical,year_cur,Twindow,year_cur,row,col);
            postprcp_seg = loadMatData(prcp_cur_dir);
            airt_cur_dir = sprintf('%spost air temp spatial/%d/post_air_temp_%s_%d_%d_%d.mat',data_dir_historical,year_cur,Twindow,year_cur,row,col);
            postairt_seg = loadMatData(airt_cur_dir);
            VPD_cur_dir = sprintf('%spost VPD spatial/%d/post_VPD_%s_%d_%d_%d.mat',data_dir_historical,year_cur,Twindow,year_cur,row,col);
            postVPD_seg = loadMatData(VPD_cur_dir);
            dnrad_cur_dir = sprintf('%spost dnrad spatial/%d/post_dnrad_%s_%d_%d_%d.mat',data_dir_historical,year_cur,Twindow,year_cur,row,col);
            postdnrad_seg = loadMatData(dnrad_cur_dir);
            sm_cur_dir    = sprintf('%spost soil moisture spatial/%d/post_sm_%s_%d_%d_%d.mat',data_dir_historical,year_cur,Twindow,year_cur,row,col);
            postsm_seg = loadMatData(sm_cur_dir);

            row_start = row*360+1;
            row_end   = row*360+360;
            col_start = col*720+1;
            col_end   = col*720+720;

            deltaEVI_global_all(row_start:row_end,col_start:col_end,i) = deltaEVI_seg;
            postprcp_global_all(row_start:row_end,col_start:col_end,i)= postprcp_seg;
            postairt_global_all(row_start:row_end,col_start:col_end,i) = postairt_seg;
            postVPD_global_all(row_start:row_end,col_start:col_end,i) = postVPD_seg;
            postdnrad_global_all(row_start:row_end,col_start:col_end,i) = postdnrad_seg;
            postsm_global_all(row_start:row_end,col_start:col_end,i) = postsm_seg;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % S2
            prcp_S2_cur_dir = sprintf('%spost prcp spatial/%d/post_prcp_%s_%d_%d_%d.mat',data_dir_S2,year_cur,Twindow,year_cur,row,col);
            postprcp_seg_S2 = loadMatData(prcp_S2_cur_dir);
            airt_S2_cur_dir = sprintf('%spost air temp spatial/%d/post_air_temp_%s_%d_%d_%d.mat',data_dir_S2,year_cur,Twindow,year_cur,row,col);
            postairt_seg_S2 = loadMatData(airt_S2_cur_dir);
            VPD_S2_cur_dir = sprintf('%spost VPD spatial/%d/post_VPD_%s_%d_%d_%d.mat',data_dir_S2,year_cur,Twindow,year_cur,row,col);
            postVPD_seg_S2 = loadMatData(VPD_S2_cur_dir);
            dnrad_S2_cur_dir = sprintf('%spost dnrad spatial/%d/post_dnrad_%s_%d_%d_%d.mat',data_dir_S2,year_cur,Twindow,year_cur,row,col);
            postdnrad_seg_S2 = loadMatData(dnrad_S2_cur_dir);
            sm_S2_cur_dir    = sprintf('%spost sm spatial/%d/post_sm_%s_%d_%d_%d.mat',data_dir_S2,year_cur,Twindow,year_cur,row,col);
            postsm_seg_S2    = loadMatData(sm_S2_cur_dir);

            postprcp_global_all_S2(row_start:row_end,col_start:col_end,i)  = postprcp_seg_S2;
            postairt_global_all_S2(row_start:row_end,col_start:col_end,i)  = postairt_seg_S2;
            postVPD_global_all_S2(row_start:row_end,col_start:col_end,i)   = postVPD_seg_S2;
            postdnrad_global_all_S2(row_start:row_end,col_start:col_end,i) = postdnrad_seg_S2;
            postsm_global_all_S2(row_start:row_end,col_start:col_end,i)    = postsm_seg_S2;

        end
    end
end

%% each LC
no_change_mask = all(LC_multiyear == LC_multiyear(:,:,1), 3);
LC_no_change = NaN(size(LC_multiyear, 1), size(LC_multiyear, 2));
LC_no_change(no_change_mask) = LC_multiyear(no_change_mask);

%% whole (regardless of LC, year, and climate zone)
% ensure each deltaEVI has a corresponding deltaAC1, deltaclim.
delta_EVI_all    = [];
post_prcp_all    = [];
post_airt_all    = [];
post_VPD_all     = [];
post_dnrad_all   = [];
post_sm_all      = [];
LC_no_change_all = [];
AC1_multiyr_mean_all = [];

post_prcp_S2_all  = [];
post_airt_S2_all  = [];
post_VPD_S2_all   = [];
post_dnrad_S2_all = [];
post_sm_S2_all    = [];
AC1_multiyr_mean_S2_all = [];

row_idx_all = [];
col_idx_all = [];

for i = 1:length(years)
    deltaEVI_global_single_yr  = deltaEVI_global_all(:,:,i);
    postprcp_global_single_yr  = postprcp_global_all(:,:,i);
    postairt_global_single_yr  = postairt_global_all(:,:,i);
    postVPD_global_single_yr   = postVPD_global_all(:,:,i);
    postdnrad_global_single_yr = postdnrad_global_all(:,:,i);
    postsm_global_single_yr    = postsm_global_all(:,:,i);

    postprcp_S2_global_single_yr  = postprcp_global_all_S2(:,:,i);
    postairt_S2_global_single_yr  = postairt_global_all_S2(:,:,i);
    postVPD_S2_global_single_yr   = postVPD_global_all_S2(:,:,i);
    postdnrad_S2_global_single_yr = postdnrad_global_all_S2(:,:,i);
    postsm_S2_global_single_yr    = postsm_global_all_S2(:,:,i);

    CombinedMask = ~isnan(deltaEVI_global_single_yr) & ~isnan(postprcp_global_single_yr)& ~isnan(postairt_global_single_yr) & ~isnan(postVPD_global_single_yr) ...
        & ~isnan(postdnrad_global_single_yr) & ~isnan(postsm_global_single_yr) & ~isnan(LC_no_change)...
        & ~isnan(postprcp_S2_global_single_yr) & ~isnan(postairt_S2_global_single_yr)& ~isnan(postVPD_S2_global_single_yr)...
        & ~isnan(postdnrad_S2_global_single_yr) & ~isnan(postsm_S2_global_single_yr);

    delta_EVI_all  = [delta_EVI_all;deltaEVI_global_single_yr(CombinedMask)];
    post_prcp_all  = [post_prcp_all;postprcp_global_single_yr(CombinedMask)];
    post_airt_all  = [post_airt_all;postairt_global_single_yr(CombinedMask)];
    post_VPD_all   = [post_VPD_all;postVPD_global_single_yr(CombinedMask)];
    post_dnrad_all = [post_dnrad_all;postdnrad_global_single_yr(CombinedMask)];
    post_sm_all    = [post_sm_all;postsm_global_single_yr(CombinedMask)];

    post_prcp_S2_all  = [post_prcp_S2_all;postprcp_S2_global_single_yr(CombinedMask)];
    post_airt_S2_all  = [post_airt_S2_all;postairt_S2_global_single_yr(CombinedMask)];
    post_VPD_S2_all   = [post_VPD_S2_all;postVPD_S2_global_single_yr(CombinedMask)];
    post_dnrad_S2_all = [post_dnrad_S2_all;postdnrad_S2_global_single_yr(CombinedMask)];
    post_sm_S2_all    = [post_sm_S2_all;postsm_S2_global_single_yr(CombinedMask)];

    LC_no_change_all = [LC_no_change_all;LC_no_change(CombinedMask)];
    AC1_multiyr_mean_all = [AC1_multiyr_mean_all;AC1_multiyr_mean(CombinedMask)];
    AC1_multiyr_mean_S2_all = [AC1_multiyr_mean_S2_all;AC1_multiyr_mean_S2(CombinedMask)];

    [row_idx, col_idx] = find(CombinedMask);
    row_idx_all = [row_idx_all;row_idx];
    col_idx_all = [col_idx_all;col_idx];
end

%%
p5 = prctile(delta_EVI_all, 1);
p95 = prctile(delta_EVI_all, 99);

% Create a logical index for values within the 5th to 95th percentile range
idx = delta_EVI_all >= p5 & delta_EVI_all <= p95;

% Filter each variable using this index
delta_EVI_all = delta_EVI_all(idx);
AC1_multiyr_mean_all = AC1_multiyr_mean_all(idx);
AC1_multiyr_mean_S2_all =AC1_multiyr_mean_S2_all(idx);
post_prcp_all = post_prcp_all(idx);
post_prcp_S2_all = post_prcp_S2_all(idx);
post_airt_all = post_airt_all(idx);
post_airt_S2_all = post_airt_S2_all(idx);
post_sm_all = post_sm_all(idx);
post_sm_S2_all = post_sm_S2_all(idx);
post_VPD_all = post_VPD_all(idx);
post_VPD_S2_all = post_VPD_S2_all(idx);
post_dnrad_all = post_dnrad_all(idx);
post_dnrad_S2_all = post_dnrad_S2_all(idx);

LC_no_change_all = LC_no_change_all(idx);
row_idx_all = row_idx_all(idx);
col_idx_all = col_idx_all(idx);

%%
data = [delta_EVI_all, AC1_multiyr_mean_all, AC1_multiyr_mean_S2_all, post_prcp_all, post_prcp_S2_all, post_airt_all, post_airt_S2_all, ...
    post_VPD_all, post_VPD_S2_all, post_dnrad_all, post_dnrad_S2_all, post_sm_all, post_sm_S2_all, LC_no_change_all, row_idx_all, col_idx_all];
data = rmmissing(data);
X = data(:,2:end);
y = data(:,1);

%%
varNames = {'AC1 mean','AC1 mean S2','prcp','prcp S2','airt','airt S2','VPD','VPD S2','solar rad','solar rad S2', 'sm', 'sm S2', 'Land Cover','row idx','col idx'};
X_table = array2table(X, 'VariableNames', varNames);
X_table.('Land Cover') = categorical(X_table.('Land Cover'));
varfun(@class, X_table, 'OutputFormat', 'cell');

%% 
LCLabels = {'ENF','EBF','DNF','DBF','MF','Savanna','Shrub & Grass','Crop'};
lcRaw = X_table.("Land Cover");

lcCodes = str2double(string(lcRaw));

% Identify valid rows (1 to 30)
valid = lcCodes >= 101 & lcCodes <= 108;

% Pre-fill with "Unknown" (or any placeholder)
lclabels = repmat("Unknown", height(X_table), 1);

% Assign valid labels only where index is valid
lclabels(valid) = string(LCLabels(lcCodes(valid)-100));

X_table.("Land Cover") = lclabels;
X_table.('Land Cover') = categorical(X_table.('Land Cover'));

%%
X_train = X_table(:,[1,3,5,7,9,11,13]);
y_train = y;

nCols = size(X_train,2);
catCols = nCols;
inputs.catCols = catCols;

% inputs.FontSize = 14;

rf_model = fitrensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 50, 'Learners', 'Tree','CategoricalPredictors', inputs.catCols);

%%
X_test = X_table(:,[2,4,6,8,10,12,13]);
X_test.Properties.VariableNames = {'AC1 mean','prcp','airt','VPD','solar rad','sm','Land Cover'};
X_test.('Land Cover') = categorical(X_table.('Land Cover'));

y_pred = predict(rf_model, X_test); % delta EVI in S2

row_idx_final = table2array(X_table(:,14));
col_idx_final = table2array(X_table(:,15));
linear_idx = sub2ind([3600, 7200], row_idx_final, col_idx_final);

deltaEVI_img = nan(3600,7200);
deltaEVI_img(linear_idx) = y;

deltaEVI_pred_img = nan(3600,7200);
deltaEVI_pred_img(linear_idx) = y_pred;
