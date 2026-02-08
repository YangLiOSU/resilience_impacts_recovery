clc
clear
close all

root_dir = '/Users/liyang/Documents/Study/';
addpath('/Users/liyang/Documents/Study/NUS/Forest Resilience/Code/matlab tools/');
Figout_word = 'Forzieri 0.5 th02 1bpstep air historical LCrevised AC1';

mask_dir = sprintf('%sNUS/Forest Resilience/Results/AC1/Data_avail_mask/Global Forzieri 0.5/data_avail_mask.mat',root_dir);
data_avail_mask = loadMatData(mask_dir);

%% 
row_num = 360;
col_num = 720;
AC1_multiyr_mean = NaN(3600,7200);

for seg_num_row = 0:9
    for seg_num_col = 0:9
        fn_AC1 = sprintf('%sNUS/Forest Resilience/Results/AC1/Global Forzieri 0.5/AC1_global_%d_%d.mat',root_dir,seg_num_row,seg_num_col);
        Data1 = load(fn_AC1);
        AC1_t = Data1.AC1_global_t; 
        filteredDates = (year(AC1_t) >= 2002 & year(AC1_t) <= 2015);
        AC1_TS_all = Data1.AC1_global;
        AC1_TS_all = AC1_TS_all(:,:,filteredDates);
        
        local_mean = NaN(row_num, col_num);  % Local to the parfor loop
        for row = 1:row_num
            for col = 1:col_num
                data_avail_mask_pix = data_avail_mask(seg_num_row*row_num+row,seg_num_col*col_num+col);

                if  data_avail_mask_pix == 1
                    AC1_TS_curSite = squeeze(AC1_TS_all(row,col,:));

                    % plot(AC1_TS_curSite)
                    % close all
                    local_mean(row,col) = mean(AC1_TS_curSite,'omitnan');
                end            
            end
        end
        AC1_multiyr_mean(seg_num_row*row_num+1:(seg_num_row+1)*row_num, ...
                 seg_num_col*col_num+1:(seg_num_col+1)*col_num) = local_mean;
    end
end

%% water availability: 
% precipitation
prcp_dir = sprintf('%sData/TerraClimate/historical/mat/ppt/ppt_monthly.mat',root_dir);
prcp_monthly = loadMatData(prcp_dir);

prcp_monthly_02_15 = prcp_monthly(:,:,25:192); % year 2002 to 2015
[nRow, nCol, nTime] = size(prcp_monthly_02_15);
nYears = nTime / 12; 

prcp_02_15_diff = zeros(nRow, nCol, nYears);
for i = 1:nYears
    idx = (i-1)*12 + (1:12);     % Indices for the 12 layers of year i
    yearlyData = prcp_monthly_02_15(:, :, idx);  % Get the 12-month stack for year i
    
    % Compute max - min along the 3rd dimension
    prcp_02_15_diff(:, :, i) = max(yearlyData, [], 3) - min(yearlyData, [], 3);
end
prcp_02_15_mean = mean(prcp_monthly_02_15,3,'omitnan');
prcp_02_15_variance = var(prcp_monthly_02_15,0,3,'omitnan');
prcp_02_15_diff_mean = mean(prcp_02_15_diff,3,'omitnan');
clear prcp_monthly prcp_monthly_02_15 prcp_02_15_diff

% VPD
VPD_dir = sprintf('%sData/TerraClimate/historical/mat/vpd/vpd_monthly.mat',root_dir);
VPD_monthly = loadMatData(VPD_dir);
VPD_monthly_02_15 = VPD_monthly(:,:,25:192); % year 2002 to 2015

VPD_02_15_diff = zeros(nRow, nCol, nYears);
for i = 1:nYears
    idx = (i-1)*12 + (1:12);     % Indices for the 12 layers of year i
    yearlyData = VPD_monthly_02_15(:, :, idx);  % Get the 12-month stack for year i
    
    % Compute max - min along the 3rd dimension
    VPD_02_15_diff(:, :, i) = max(yearlyData, [], 3) - min(yearlyData, [], 3);
end
VPD_02_15_mean = mean(VPD_monthly_02_15,3,'omitnan');
VPD_02_15_variance = var(VPD_monthly_02_15,0,3,'omitnan');
VPD_02_15_diff_mean = mean(VPD_02_15_diff,3,'omitnan');
clear VPD_monthly VPD_monthly_02_15 VPD_02_15_diff

%%
years = arrayfun(@num2str, 2002:2015, 'UniformOutput', false);
LC_multiyear = [];
for i = 1:length(years)
    year_cur = str2double(years{i});

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
end

no_change_mask = all(LC_multiyear == LC_multiyear(:,:,1), 3);
LC_no_change = NaN(size(LC_multiyear, 1), size(LC_multiyear, 2));
LC_no_change(no_change_mask) = LC_multiyear(no_change_mask);

%%  dataset
y_AC1 = AC1_multiyr_mean(data_avail_mask);
x1_prcp_mean = prcp_02_15_mean(data_avail_mask);
x2_VPD_mean  = VPD_02_15_mean(data_avail_mask);
x3_prcp_variance = prcp_02_15_variance(data_avail_mask);
x4_VPD_variance = VPD_02_15_variance(data_avail_mask);
x5_prcp_diff  = prcp_02_15_diff_mean(data_avail_mask);
x6_VPD_diff   = VPD_02_15_diff_mean(data_avail_mask);
x7_LC = LC_no_change(data_avail_mask);
[row_idx, col_idx] = find(data_avail_mask);

data_train = [y_AC1, x1_prcp_mean, x2_VPD_mean, x3_prcp_variance, x4_VPD_variance, x5_prcp_diff, x6_VPD_diff, x7_LC];
varNames = {'AC1','prcp mean','VPD mean','prcp variance','VPD variance','prcp range','VPD range','Land Cover'};
X_table = array2table(data_train, 'VariableNames', varNames);
X_table.('Land Cover') = categorical(X_table.('Land Cover'));

X_table_clean = rmmissing(X_table);
isDefined = ~ismissing(X_table_clean{:,8});
X_table_clean = X_table_clean(isDefined, :);
y_train_clean = X_table_clean(:,1);
clear X_table

%%
X_table = X_table_clean(:,[2,3,6,7,8]);
y = table2array(y_train_clean);

inputs.xticklabels = {'prcp mean','VPD mean','prcp diff','VPD diff','Land Cover'};
var_num = size(X_table,2);
nCols = size(X_table,2);
catCols = nCols;
inputs.FigTab_out = sprintf('%sNUS/Forest Resilience/Figure/resilience drop/%s/RF/Benchmark_deltaclim_soil_lc/',root_dir,Figout_word);
if ~exist(inputs.FigTab_out, 'dir')
    mkdir(inputs.FigTab_out)
end

inputs.catCols = catCols;
inputs.FontSize = 16;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cv = cvpartition(height(X_table), 'HoldOut', 0.2);  % 70% train, 30% test
idxTrain = training(cv);
idxTest = test(cv);

X_train = X_table(idxTrain, :);
y_train = y(idxTrain);

X_test = X_table(idxTest, :);
y_test = y(idxTest);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rf_model = fitrensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 50, 'Learners', 'Tree','CategoricalPredictors', inputs.catCols);
explainer = shapley(rf_model,X_train,'QueryPoints',X_test,'UseParallel',true);
