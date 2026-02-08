clc
clear

addpath('/Users/liyang/Documents/Study/NUS/Forest Resilience/Code/matlab tools/');
what_data = 'AC1 sudden drop Forzieri 0.5 th02 1bpstep air';
what_data_history = 'AC1 sudden drop Forzieri 0.5 th02 1bpstep air historical';
Figout_word = 'Forzieri 0.5 th02 1bpstep air historical LCrevised';
EVI_word = 'Forzieri 0.5 th02'

root_dir = '/Users/liyang/Documents/Study/';
data_dir = sprintf('%sNUS/Forest Resilience/Results/%s/',root_dir,what_data);
data_dir_history = sprintf('%sNUS/Forest Resilience/Results/%s/',root_dir,what_data_history);

%% Climate zone https://www.gloh2o.org/koppen/
Koppen_dir = sprintf('%sData/koppen_geiger_tif/1991_2020/koppen_geiger_0p1.tif',root_dir);
Koppen = imread(Koppen_dir);
Koppen = double(Koppen);
Koppen(Koppen==0)=NaN;

Koppens_1 = imresize(Koppen, [3600, 7200], 'nearest');

%% EVI_multiyr_mean
EVI_multiyr_mean_dir = sprintf('%sNUS/Forest Resilience/Data/Global EVI/%s/EVI_multiyr_mean_beast.mat',root_dir,EVI_word);
EVI_multiyr_mean = loadMatData(EVI_multiyr_mean_dir);

%% AC1_multiyr_mean
AC1_multiyr_mean_dir = sprintf('%sNUS/Forest Resilience/Data/Global AC1/%s/AC1_multiyr_mean.mat',root_dir,EVI_word);
AC1_multiyr_mean = loadMatData(AC1_multiyr_mean_dir);

%% MAP MAT
MAP_dir = sprintf('%sData/TerraClimate/historical/mat/ppt/MAP.mat',root_dir);
MAP = loadMatData(MAP_dir);
MAP(MAP==0)=NaN;
MAP(MAP>4000)=4000;

MAT_dir = sprintf('%sData/TerraClimate/historical/mat/temp/MAT.mat',root_dir);
MAT = loadMatData(MAT_dir);

%% soil
phh2o_dir = sprintf('%sData/Soil Grid/mat/phh2o_0_30.mat',root_dir);
phh2o = loadMatData(phh2o_dir);
sand_dir = sprintf('%sData/Soil Grid/mat/sand_0_30.mat',root_dir);
sand = loadMatData(sand_dir);
clay_dir = sprintf('%sData/Soil Grid/mat/clay_0_30.mat',root_dir);
clay = loadMatData(clay_dir);
nitrogen_dir = sprintf('%sData/Soil Grid/mat/nitrogen_0_30.mat',root_dir);
nitrogen = loadMatData(nitrogen_dir);
soc_dir = sprintf('%sData/Soil Grid/mat/soc_0_30.mat',root_dir);
soc = loadMatData(soc_dir);

%%
Twindow = '1bpstep';
years = arrayfun(@num2str, 2002:2022, 'UniformOutput', false);

layers = length(years);
deltaEVI_global_all  = NaN(3600,7200,layers);
deltaAC1_global_all  = NaN(3600,7200,layers);
postprcp_global_all  = NaN(3600,7200,layers);
postairt_global_all   = NaN(3600,7200,layers);
postsm_global_all    = NaN(3600,7200,layers);
postVPD_global_all   = NaN(3600,7200,layers);
postdnrad_global_all = NaN(3600,7200,layers);
deltaprcp_global_all = NaN(3600,7200,layers); 
deltaairt_global_all  = NaN(3600,7200,layers);
deltasm_global_all   = NaN(3600,7200,layers);
deltaVPD_global_all  = NaN(3600,7200,layers);

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
    % for row = 5:5
    %     for col = 3:3
            
            EVI_cur_dir = sprintf('%s%s spatial/delta EVI/%d/delta_EVI_%s_%d_%d_%d.mat',data_dir,Twindow,year_cur,Twindow,year_cur,row,col);
            deltaEVI_seg = loadMatData(EVI_cur_dir);
            AC1_cur_dir = sprintf('%s%s spatial/delta AC1/%d/delta_AC1_%s_%d_%d_%d.mat',data_dir,Twindow,year_cur,Twindow,year_cur,row,col);
            deltaAC1_seg = loadMatData(AC1_cur_dir);
            prcp_cur_dir = sprintf('%spost prcp spatial/%d/post_prcp_%s_%d_%d_%d.mat',data_dir_history,year_cur,Twindow,year_cur,row,col);
            postprcp_seg = loadMatData(prcp_cur_dir);
            airt_cur_dir = sprintf('%spost air temp spatial/%d/post_air_temp_%s_%d_%d_%d.mat',data_dir_history,year_cur,Twindow,year_cur,row,col);
            postairt_seg = loadMatData(airt_cur_dir);
            sm_cur_dir = sprintf('%spost soil moisture spatial/%d/post_sm_%s_%d_%d_%d.mat',data_dir_history,year_cur,Twindow,year_cur,row,col);
            postsm_seg = loadMatData(sm_cur_dir);
            VPD_cur_dir = sprintf('%spost VPD spatial/%d/post_VPD_%s_%d_%d_%d.mat',data_dir_history,year_cur,Twindow,year_cur,row,col);
            postVPD_seg = loadMatData(VPD_cur_dir);
            dnrad_cur_dir = sprintf('%spost dnrad spatial/%d/post_dnrad_%s_%d_%d_%d.mat',data_dir_history,year_cur,Twindow,year_cur,row,col);
            postdnrad_seg = loadMatData(dnrad_cur_dir);

            deltaprcp_cur_dir = sprintf('%sdelta prcp spatial/%d/delta_prcp_%s_%d_%d_%d.mat',data_dir_history,year_cur,Twindow,year_cur,row,col);
            deltaprcp_seg = loadMatData(deltaprcp_cur_dir);
            deltaairt_cur_dir = sprintf('%sdelta air temp spatial/%d/delta_air_temp_%s_%d_%d_%d.mat',data_dir_history,year_cur,Twindow,year_cur,row,col);
            deltaairt_seg = loadMatData(deltaairt_cur_dir);
            deltasm_cur_dir = sprintf('%sdelta soil moisture spatial/%d/delta_sm_%s_%d_%d_%d.mat',data_dir_history,year_cur,Twindow,year_cur,row,col);
            deltasm_seg = loadMatData(deltasm_cur_dir);
            deltaVPD_cur_dir = sprintf('%sdelta VPD spatial/%d/delta_VPD_%s_%d_%d_%d.mat',data_dir_history,year_cur,Twindow,year_cur,row,col);
            deltaVPD_seg = loadMatData(deltaVPD_cur_dir);

            row_start = row*360+1;
            row_end   = row*360+360;
            col_start = col*720+1;
            col_end   = col*720+720;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            deltaEVI_global_all(row_start:row_end,col_start:col_end,i) = deltaEVI_seg;
            deltaAC1_global_all(row_start:row_end,col_start:col_end,i) = deltaAC1_seg;
            postprcp_global_all(row_start:row_end,col_start:col_end,i)= postprcp_seg;
            postairt_global_all(row_start:row_end,col_start:col_end,i) = postairt_seg;
            postsm_global_all(row_start:row_end,col_start:col_end,i)  = postsm_seg;
            postVPD_global_all(row_start:row_end,col_start:col_end,i) = postVPD_seg;
            postdnrad_global_all(row_start:row_end,col_start:col_end,i) = postdnrad_seg;

            deltaprcp_global_all(row_start:row_end,col_start:col_end,i)= deltaprcp_seg;
            deltaairt_global_all(row_start:row_end,col_start:col_end,i) = deltaairt_seg;
            deltasm_global_all(row_start:row_end,col_start:col_end,i)  = deltasm_seg;
            deltaVPD_global_all(row_start:row_end,col_start:col_end,i) = deltaVPD_seg;
        end
    end
end

%% each LC
no_change_mask = all(LC_multiyear == LC_multiyear(:,:,1), 3);
LC_no_change = NaN(size(LC_multiyear, 1), size(LC_multiyear, 2));
LC_no_change(no_change_mask) = LC_multiyear(no_change_mask);

%% whole (regardless of LC, year, and climate zone)
delta_EVI_all  = [];
delta_AC1_all  = [];
post_prcp_all  = [];
post_airt_all   = [];
post_sm_all    = [];
post_VPD_all   = [];
post_dnrad_all = [];
Koppen_CZ_all  = [];
LC_no_change_all = [];
delta_prcp_all = [];
delta_airt_all   = [];
delta_sm_all    = [];
delta_VPD_all   = [];
EVI_multiyr_mean_all = [];
AC1_multiyr_mean_all = [];

MAP_all = [];
MAT_all = [];

phh2o_all = [];
sand_all = [];
clay_all = [];
nitrogen_all = [];
soc_all = [];

for i = 1:length(years)
    deltaEVI_global_single_yr  = deltaEVI_global_all(:,:,i);
    deltaAC1_global_single_yr  = deltaAC1_global_all(:,:,i);
    postprcp_global_single_yr = postprcp_global_all(:,:,i);
    postairt_global_single_yr  = postairt_global_all(:,:,i);
    postsm_global_single_yr   = postsm_global_all(:,:,i);
    postVPD_global_single_yr  = postVPD_global_all(:,:,i);
    postdnrad_global_single_yr= postdnrad_global_all(:,:,i);

    deltaprcp_global_single_yr = deltaprcp_global_all(:,:,i);
    deltaairt_global_single_yr  = deltaairt_global_all(:,:,i);
    deltasm_global_single_yr   = deltasm_global_all(:,:,i);
    deltaVPD_global_single_yr  = deltaVPD_global_all(:,:,i);

    CombinedMask = ~isnan(deltaEVI_global_single_yr) & ~isnan(deltaAC1_global_single_yr) & ~isnan(postprcp_global_single_yr)...
        & ~isnan(postairt_global_single_yr) & ~isnan(postsm_global_single_yr) & ~isnan(postVPD_global_single_yr) ...
        & ~isnan(Koppens_1) & ~isnan(LC_no_change)...
        & ~isnan(deltaprcp_global_single_yr) & ~isnan(deltaairt_global_single_yr) & ~isnan(deltasm_global_single_yr) & ~isnan(deltaVPD_global_single_yr)...
        & ~isnan(postdnrad_global_single_yr);

    delta_EVI_all  = [delta_EVI_all;deltaEVI_global_single_yr(CombinedMask)];
    delta_AC1_all  = [delta_AC1_all;deltaAC1_global_single_yr(CombinedMask)];
    post_prcp_all = [post_prcp_all;postprcp_global_single_yr(CombinedMask)];
    post_airt_all  = [post_airt_all;postairt_global_single_yr(CombinedMask)];
    post_sm_all   = [post_sm_all;postsm_global_single_yr(CombinedMask)];
    post_VPD_all  = [post_VPD_all;postVPD_global_single_yr(CombinedMask)];
    post_dnrad_all= [post_dnrad_all;postdnrad_global_single_yr(CombinedMask)];
    % timing_all    = [timing_all;timing_global_single_yr(CombinedMask)];
    Koppen_CZ_all = [Koppen_CZ_all;Koppens_1(CombinedMask)];
    LC_no_change_all = [LC_no_change_all;LC_no_change(CombinedMask)];
    delta_prcp_all = [delta_prcp_all;deltaprcp_global_single_yr(CombinedMask)];
    delta_airt_all  = [delta_airt_all;deltaairt_global_single_yr(CombinedMask)];
    delta_sm_all   = [delta_sm_all;deltasm_global_single_yr(CombinedMask)];
    delta_VPD_all  = [delta_VPD_all;deltaVPD_global_single_yr(CombinedMask)];

    EVI_multiyr_mean_all = [EVI_multiyr_mean_all;EVI_multiyr_mean(CombinedMask)];
    AC1_multiyr_mean_all = [AC1_multiyr_mean_all;AC1_multiyr_mean(CombinedMask)];

    MAP_all = [MAP_all;MAP(CombinedMask)];
    MAT_all = [MAT_all;MAT(CombinedMask)];

    phh2o_all = [phh2o_all;phh2o(CombinedMask)];
    sand_all  = [sand_all;sand(CombinedMask)];
    clay_all = [clay_all;clay(CombinedMask)];
    nitrogen_all = [nitrogen_all;nitrogen(CombinedMask)];
    soc_all = [soc_all;soc(CombinedMask)];
end
%%
p5 = prctile(delta_EVI_all, 1);
p95 = prctile(delta_EVI_all, 99);

% Create a logical index for values within the 5th to 95th percentile range
idx = delta_EVI_all >= p5 & delta_EVI_all <= p95;

% idx = true(size(delta_EVI_all));

% Filter each variable using this index
delta_EVI_all = delta_EVI_all(idx);
delta_AC1_all = delta_AC1_all(idx);
post_prcp_all = post_prcp_all(idx);
delta_prcp_all = delta_prcp_all(idx);
post_airt_all = post_airt_all(idx);
delta_airt_all = delta_airt_all(idx);
post_sm_all = post_sm_all(idx);
delta_sm_all = delta_sm_all(idx);
post_VPD_all = post_VPD_all(idx);
delta_VPD_all = delta_VPD_all(idx);
post_dnrad_all = post_dnrad_all(idx);
Koppen_CZ_all = Koppen_CZ_all(idx);
LC_no_change_all = LC_no_change_all(idx);
EVI_multiyr_mean_all = EVI_multiyr_mean_all(idx);
AC1_multiyr_mean_all = AC1_multiyr_mean_all(idx);
phh2o_all = phh2o_all(idx);
sand_all = sand_all(idx);
clay_all = clay_all(idx);
nitrogen_all = nitrogen_all(idx);
soc_all = soc_all(idx);

MAP_all = MAP_all(idx);
MAT_all = MAT_all(idx);

%%
data_all = [delta_EVI_all, EVI_multiyr_mean_all, AC1_multiyr_mean_all, delta_AC1_all, post_prcp_all, delta_prcp_all, post_airt_all, delta_airt_all, post_sm_all, delta_sm_all, post_VPD_all, delta_VPD_all,...
    post_dnrad_all,...
    phh2o_all, sand_all, clay_all, nitrogen_all, soc_all,...
    Koppen_CZ_all, LC_no_change_all,MAP_all,MAT_all];
data = rmmissing(data_all);
X = data(:,2:end);
y = data(:,1);

%%
varNames = {'EVI mean','AC1 mean','ΔAC1','prcp','Δprcp','airt','Δairt','sm','Δsm','VPD','ΔVPD','solar rad','ph','sand','clay','nitrogen','soc','Climate Zone','Land Cover','MAP','MAT'};
X_table = array2table(X, 'VariableNames', varNames);
X_table.('Climate Zone') = categorical(X_table.('Climate Zone'));
X_table.('Land Cover') = categorical(X_table.('Land Cover'));
varfun(@class, X_table, 'OutputFormat', 'cell');

%% change categorical feature labels
climateLabels = {'Af', 'Am', 'Aw', 'BWh', 'BWk', 'BSh', 'BSk', ...
                 'Csa', 'Csb', 'Csc', 'Cwa', 'Cwb', 'Cwc', ...
                 'Cfa', 'Cfb', 'Cfc', 'Dsa', 'Dsb', 'Dsc', 'Dsd', ...
                 'Dwa', 'Dwb', 'Dwc', 'Dwd', 'Dfa', 'Dfb', 'Dfc', 'Dfd', ...
                 'ET', 'EF'};

zoneRaw = X_table.("Climate Zone");

% Convert categorical (e.g., '9') to numeric (e.g., 9)
zoneCodes = str2double(string(zoneRaw));  % this gives correct numeric codes

% Identify valid rows (1 to 30)
valid = zoneCodes >= 1 & zoneCodes <= 30;

% Pre-fill with "Unknown" (or any placeholder)
zonelabels = repmat("Unknown", height(X_table), 1);

% Assign valid labels only where index is valid
zonelabels(valid) = string(climateLabels(zoneCodes(valid)));

X_table.("Climate Zone") = zonelabels;

%% 
LCLabels = {'ENF','EBF','DNF','DBF','MF','Savanna','Shrub & Grass','Crop'};
lcRaw = X_table.("Land Cover");

lcCodes = str2double(string(lcRaw));

valid = lcCodes >= 101 & lcCodes <= 108;

% Pre-fill with "Unknown" (or any placeholder)
lclabels = repmat("Unknown", height(X_table), 1);

% Assign valid labels only where index is valid
lclabels(valid) = string(LCLabels(lcCodes(valid)-100));

X_table.("Land Cover") = lclabels;

X_table.('Climate Zone') = categorical(X_table.('Climate Zone'));
X_table.('Land Cover') = categorical(X_table.('Land Cover'));
