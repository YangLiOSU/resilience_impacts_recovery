clc
clear
close all

cd '/Users/liyang/Desktop/'
load('Xtable_y_historical_LCrevised_MAP_MAT.mat')
Figout_word = 'Forzieri 0.5 th02 1bpstep air historical LCrevised';
root_dir = '/Users/liyang/Documents/Study/';
addpath('/Users/liyang/Documents/Study/Toolbox/violin/daviolinplot/')
addpath('/Users/liyang/Documents/Study/Toolbox/Perceptually uniform colormaps/')

X_table_ori = X_table;
clear X_table;

%% 
newNames = {'EVI mean','AC1 mean','ΔAC1','prcp','Δprcp','T_{air}','ΔT_{air}','sm','Δsm','VPD','ΔVPD','solar rad.','ph','sand','clay','nitrogen','soc','climate zone','land cover','MAP','MAT'};
X_table_ori.Properties.VariableNames = newNames;
for i = 6:6
    if i ==1 % benchmark
        X_table = X_table_ori(:,[1,2,3,4,6,8,10,12]);
        inputs.xticklabels = {'EVI mean','AC1 mean','ΔAC1','prcp','skt','sm','VPD','solar rad'};
        var_num = size(X_table,2);
        nCols = size(X_table,2);
        catCols = [];
        inputs.FigTab_out = sprintf('%sNUS/Forest Resilience/Figure/resilience drop/%s/RF/Benchmark/',root_dir,Figout_word);   

    elseif i ==2 % benchmark + delta_clim
        X_table = X_table_ori(:,[1,2,3,4,5,6,7,8,9,10,11,12]);
        inputs.xticklabels = {'EVI mean','AC1 mean','ΔAC1','prcp','Δprcp','skt','Δskt','sm','Δsm','VPD','ΔVPD','solar rad'};
        var_num = size(X_table,2);
        nCols = size(X_table,2);
        catCols = [];
        inputs.FigTab_out = sprintf('%sNUS/Forest Resilience/Figure/resilience drop/%s/RF/Benchmark_deltaClim/',root_dir,Figout_word);

    elseif i ==3 % benchmark + delta_clim + soil
        X_table = X_table_ori(:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]);
        inputs.xticklabels = {'EVI mean','AC1 mean','ΔAC1','prcp','Δprcp','skt','Δskt','sm','Δsm','VPD','ΔVPD','solar rad','ph','sand','clay','nitrogen','soc'};
        var_num = size(X_table,2);
        nCols = size(X_table,2);
        catCols = [];
        inputs.FigTab_out = sprintf('%sNUS/Forest Resilience/Figure/resilience drop/%s/RF/Benchmark_deltaClim_soil/',root_dir,Figout_word);

    elseif i ==4 % benchmark + lc
        X_table = X_table_ori(:,[1,2,3,4,6,8,10,12,20]);
        inputs.xticklabels = {'EVI mean','AC1 mean','ΔAC1','prcp','skt','sm','VPD','solar rad','land cover'};
        var_num = size(X_table,2);
        nCols = size(X_table,2);
        catCols = nCols;
        inputs.FigTab_out = sprintf('%sNUS/Forest Resilience/Figure/resilience drop/%s/RF/Benchmark_lc/',root_dir,Figout_word);     

    elseif i ==5 % benchmark + delta clim +lc
        X_table = X_table_ori(:,[1,2,3,4,5,6,7,8,9,10,11,12,20]);
        inputs.xticklabels = {'EVI mean','AC1 mean','ΔAC1','prcp','Δprcp','skt','Δskt','sm','Δsm','VPD','ΔVPD','solar rad','land cover'};
        var_num = size(X_table,2);
        nCols = size(X_table,2);
        catCols = nCols;
        inputs.FigTab_out = sprintf('%sNUS/Forest Resilience/Figure/resilience drop/%s/RF/Benchmark_deltaclim_lc/',root_dir,Figout_word);

    elseif i ==6 % benchmark + delta clim + soil + lc
        X_table = X_table_ori(:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19]);
        inputs.xticklabels = {'EVI mean','AC1 mean','ΔAC1','prcp','Δprcp','T_{air}','ΔT_{air}','sm','Δsm','VPD','ΔVPD','solar rad','ph','sand','clay','nitrogen','soc','land cover'};
        var_num = size(X_table,2);
        nCols = size(X_table,2);
        catCols = nCols;
        inputs.FigTab_out = sprintf('%sNUS/Forest Resilience/Figure/Final figures 2025/Fig. 2 revised/',root_dir);

    elseif i ==7 % benchmark + delta clim + soil + cz + lc
        X_table = X_table_ori(:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]);
        inputs.xticklabels = {'EVI mean','AC1 mean','ΔAC1','prcp','Δprcp','skt','Δskt','sm','Δsm','VPD','ΔVPD','solar rad','ph','sand','clay','nitrogen','soc','climate zone','land cover'};
        var_num = size(X_table,2);
        nCols = size(X_table,2);
        catCols = [nCols-1:nCols];
        inputs.FigTab_out = sprintf('%sNUS/Forest Resilience/Figure/resilience drop/%s/RF/Benchmark_deltaclim_soil_cz_lc/',root_dir,Figout_word);

    elseif i ==8 % benchmark + delta clim + soil + cz + lc + yr
        X_table = X_table_ori(:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]);
        inputs.xticklabels = {'EVI mean','AC1 mean','ΔAC1','prcp','Δprcp','skt','Δskt','sm','Δsm','VPD','ΔVPD','solar rad','ph','sand','clay','nitrogen','soc','dist. year','climate zone','land cover'};
        var_num = size(X_table,2);
        nCols = size(X_table,2);
        catCols = [nCols-2:nCols];
        inputs.FigTab_out = sprintf('%sNUS/Forest Resilience/Figure/resilience drop/%s/RF/Benchmark_deltaclim_soil_cz_lc_yr/',root_dir,Figout_word);

    elseif i ==9 % benchmark + delta clim + soil + yr
        X_table = X_table_ori(:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]);
        inputs.xticklabels = {'EVI mean','AC1 mean','ΔAC1','prcp','Δprcp','skt','Δskt','sm','Δsm','VPD','ΔVPD','solar rad','ph','sand','clay','nitrogen','soc','dist. year'};
        var_num = size(X_table,2);
        nCols = size(X_table,2);
        catCols = nCols;
        inputs.FigTab_out = sprintf('%sNUS/Forest Resilience/Figure/resilience drop/%s/RF/Benchmark_deltaclim_soil_yr/',root_dir,Figout_word);

    elseif i ==10 % benchmark + delta clim + soil + yr % Yanlan's suggestion
        X_table = X_table_ori(:,[2,4,6,8,10,12,19]);
        inputs.xticklabels = {'AC1 mean','prcp','skt','sm','VPD','solar rad','land cover'};
        var_num = size(X_table,2);
        nCols = size(X_table,2);
        catCols = nCols;
        inputs.FigTab_out = sprintf('%sNUS/Forest Resilience/Figure/resilience drop/%s/RF/Benchmark_deltaclim_soil_yr/',root_dir,Figout_word);

    end

    if ~exist(inputs.FigTab_out, 'dir')
        mkdir(inputs.FigTab_out);
    end
    inputs.catCols = catCols;
    inputs.FontSize = 16;

    %%
    cv = cvpartition(height(X_table), 'HoldOut', 0.2); 
    idxTrain = training(cv);
    idxTest = test(cv);

    X_train = X_table(idxTrain, :);
    y_train = y(idxTrain);

    X_test = X_table(idxTest, :);
    y_test = y(idxTest);

    rf_model = fitrensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 50, 'Learners', 'Tree','CategoricalPredictors', inputs.catCols);

    %%
    inputs.title1 = sprintf('ΔEVI (training data)');
    inputs.title2 = sprintf('Predictor Importance');
    check1 = RFplot(X_train,y_train,inputs,rf_model)

    %%
    inputs.title1 = sprintf('ΔEVI (testing data)');
    % inputs.title2 = sprintf('Predictor Importance (testing data)');
    check2 = RFplot(X_test,y_test,inputs,rf_model)

    %%
    explainer = shapley(rf_model,X_train,'QueryPoints',X_test,'UseParallel',true);

    %%
    title = sprintf('Shapley Value bars');
    inputs.outname1 = sprintf('%s%s.jpg', inputs.FigTab_out, title);

    title = sprintf('Shapley Value dots');
    inputs.outname2 = sprintf('%s%s.jpg', inputs.FigTab_out, title);

    inputs.var_num = var_num;
    check2 = shapleyplot(explainer,inputs)

    %%
    title = sprintf('histogram')
    inputs.outname3 = sprintf('%s%s.jpg', inputs.FigTab_out, title);
    X_table{:,2}=0.4;
    y_AC1_fixed = predict(rf_model, X_table);
    check3 = hg(y,y_AC1_fixed,inputs)

    %%
    inputs.FigTab_out_sub = sprintf('%seach_feature/',inputs.FigTab_out);
    if ~exist(inputs.FigTab_out_sub, 'dir')
        mkdir(inputs.FigTab_out_sub);
    end
    check4 = shapleyPDP_num(explainer,inputs)

    %%
    inputs.FigTab_out_sub = sprintf('%seach_feature/',inputs.FigTab_out);
    if ~exist(inputs.FigTab_out_sub, 'dir')
        mkdir(inputs.FigTab_out_sub); 
    end
    figure
    check5 = shapleyPDP_cat(explainer,inputs);
    close all

end

%%
function check = hg(y1,y2,inputs)
fig = figure('Theme','light','Position',[100,600,800,300]);

% ---- common binning (same edges) ----
xlimVals = [-0.05 0.06];                  % your chosen range
edges = linspace(xlimVals(1), xlimVals(2), 401);  % ~0.001 bin width

% ---- plot as probability density (area = 1) ----
hold on
h1 = histogram(y1,           'BinEdges',edges, 'Normalization','pdf', ...
    'FaceColor',[0 0.447 0.741], 'FaceAlpha',0.6, 'EdgeColor','none', ...
    'DisplayName','Observed \DeltaEVI');

h2 = histogram(y2, 'BinEdges',edges, 'Normalization','pdf', ...
    'FaceColor',[0.85 0.325 0.098], 'FaceAlpha',0.45,'EdgeColor','none', ...
    'DisplayName','Modeled \DeltaEVI with fixed AC1');

% ---- axes & labels ----
xlim(xlimVals)
xlabel('\DeltaEVI','FontSize',16)
ylabel('Probability density','FontSize',16)
legend('Location','northeast')
set(gca,'FontSize',15)
grid on
box on
hold off

% ---- mean ----
mu_y1 = mean(y1(:), 'omitnan');
str_y1 = sprintf('%.6e', mu_y1);  
tokens_y1 = regexp(str_y1, '([-+]?\d\.\d+)e([+-]?\d+)', 'tokens');
man_y1  = str2double(tokens_y1{1}{1});
expn_y1 = str2double(tokens_y1{1}{2});

mu_y2 = mean(y2(:), 'omitnan');
str_y2 = sprintf('%.6e', mu_y2);  
tokens_y2 = regexp(str_y2, '([-+]?\d\.\d+)e([+-]?\d+)', 'tokens');
man_y2  = str2double(tokens_y2{1}{1});
expn_y2 = str2double(tokens_y2{1}{2});

txt = sprintf('Mean\\_observed = %.2f\\times10^{%d}\nMean\\_modeled = %.2f\\times10^{%d}', man_y1, expn_y1,man_y2, expn_y2);

text(0.5,0.6,txt, ...
    'Units','normalized', ...
    'HorizontalAlignment','left', ...
    'VerticalAlignment','top', ...
    'FontSize',15, ...
    'Color','k', ...
    'Interpreter','tex', ...   % <<-- Use TeX, not LaTeX or none
    'Clipping','off', ...
    'BackgroundColor','w', 'Margin',2);

% ---- save ----
print(fig,'-djpeg','-r600',inputs.outname3);
check = 1;

end

%%
function check = shapleyplot(explainer,inputs)
figure(1); clf
fig = figure('Theme','light',...
    'Position', [100, 100, 300, 500]);  % width=1200px, height=800px
h = plot(explainer, NumImportantPredictors=inputs.var_num); 

sortedTbl = sortrows(explainer.MeanAbsoluteShapley, 'Value', 'descend');
barNames = sortedTbl.Predictor;

catColor  = [106, 121, 173]/255;  % Categorical: dist.year, Land Cover, Climate Zone
cliColor  = [162, 195, 234]/255;   % Climate: skt, prcp, solar rad, Δsm, Δskt, VPD, ΔVPD
vegColor  = [58, 103, 84]/255;   % Vegetation: EVI mean, AC1 mean
soilColor = [146, 116, 91]/255; 


barColors = zeros(length(barNames),3); 

for i = 1:length(barNames)
    name = barNames{i};
    if ismember(name, {'EVI mean', 'AC1 mean','ΔAC1'})
        barColors(i,:) = vegColor;
    elseif ismember(name, {'prcp','Δprcp','T_{air}','ΔT_{air}','sm','Δsm','VPD','ΔVPD','solar rad.'})
        barColors(i,:) = cliColor;
    elseif ismember(name, {'ph','sand','clay','nitrogen','soc'})
        barColors(i,:) = soilColor;
    else
        barColors(i,:) = catColor;  % If none of the above, assign categorical color
    end
end

h.FaceColor = 'flat'; % important: set to 'flat' to use CData
h.CData = flipud(barColors);

title('') 
ax = gca; 
xlabel(ax,'Mean |Shapley|','FontSize',7)
ax.XAxis.FontSize = 7; 
ax.YAxis.FontSize = 7; 
ax.YAxis.TickLabelInterpreter = 'tex'; 
hold off

print(gcf, '-djpeg', '-r600', inputs.outname1);
check = 1;
end

%%
function check = shapleyPDP_num(explainer,inputs)
predictorNames = explainer.Shapley.Predictor;
predictorNames(18)=[];
% predictorNames = predictorNames([6,7]);
numFeatures = length(predictorNames);

for i = 1:numFeatures
    fig = figure('Theme','light','Position', [100, 600, 800, 600]); 
    x_feature = explainer.QueryPoints{:, i};  % x-axis: feature value
    y_shap = explainer.Shapley.Value(i,:)';  % transpose to 17607x1

    p1 = prctile(x_feature, 1);
    p99 = prctile(x_feature, 99);

    idx = x_feature >= p1 & x_feature <= p99;
    x_feature = x_feature(idx);
    y_shap = y_shap(idx);
    
    scatter(x_feature, y_shap, 10, 'filled', 'MarkerFaceAlpha', 0.3);
    hold on

    % Plot the fitted smooth line
    [x_feature_sorted, sortIdx] = sort(x_feature);
    y_shap_sorted = y_shap(sortIdx);

    smooth_y = smoothdata(y_shap_sorted, 'movmean', 10000);

    plot(x_feature_sorted, smooth_y, 'b-', 'LineWidth', 2);

    xlabel(predictorNames{i}, 'Interpreter', 'none');
    ylabel('Shapley Value');
    % title(predictorNames{i}, 'Interpreter', 'none');
    grid on;
    hold off

    xlim([min(x_feature), max(x_feature)]);
    
    ax = gca;  % get current axis
    ax.XAxis.FontSize = 30;  % x tick labels
    ax.YAxis.FontSize = 30;  % y tick labels
    ax.XLabel.FontSize = 32; % x axis label
    ax.YLabel.FontSize = 32; % y axis label
    set(get(gca, 'XLabel'), 'Interpreter', 'tex');   % or 'latex' or 'none'

axs = findall(gcf,'Type','axes','-not','Tag','legend','-not','Tag','Colorbar');

for k = 1:numel(axs)
    ax = axs(k);
    t  = ax.YTick; t = t(isfinite(t));
    if numel(t) < 2 || all(t==0), continue; end

    % Use the tick step to choose exponent (e.g., 0.005 -> n = -3)
    dt   = diff(unique(t)); dt = dt(dt~=0);
    step = min(abs(dt));                 % characteristic tick step
    n    = floor(log10(step));           % exponent from step

    ax.YRuler.ExponentMode = 'manual';
    ax.YRuler.Exponent     = n;          % shows ×10^n outside

    % Pick tick precision from the mantissa step
    mantStep = step / 10^n;              % e.g., 0.005 / 1e-3 = 5
    if mantStep >= 1
        fmt = '%.0f';
    elseif mantStep >= 0.1
        fmt = '%.1f';
    else
        fmt = '%.2f';
    end
    ytickformat(ax, fmt);
end

    OutName = sprintf('%s%s.jpg', inputs.FigTab_out_sub, predictorNames{i});
    % print(gcf, '-djpeg', '-r600', OutName);

end
check = 1;
end

%%
function check = shapleyPDP_cat(explainer,inputs)
predictorNames = explainer.Shapley.Predictor;
predictorNames(1:17)=[];
numFeatures = length(predictorNames);

x_feature = explainer.QueryPoints{:, 18};  % x-axis: feature value
y_shap = explainer.Shapley.Value(18,:)';  % transpose to 17607x1

% landcover_order = {'DBF', 'DNF', 'EBF', 'ENF', 'MF', 'Shrub', 'Savanna','Grassland','Cropland'};
landcover_order = {'DBF', 'DNF', 'EBF', 'ENF', 'MF', 'Savanna','Shrub & Grass','Crop'};
x = categorical(x_feature, landcover_order, 'Ordinal', true);
x_feature_cp = double(x);

fig = figure('Theme','light','Position', [100, 600, 800, 600]); 
daviolinplot_9cats(y_shap,'groups',x_feature_cp);
    
    hold on;
    yline(0, '--', 'LineWidth', 2, 'Color', [153,153,153]./255);  % black dashed line at y = 0

    xlabel('land cover');
    ylabel('Shapley Value');

    ax = gca;  % get current axis
    set(gca, 'XTick', 1:9, 'XTickLabel', landcover_order);
    ax.XAxis.FontSize = 30;  % x tick labels
    ax.YAxis.FontSize = 30;  % y tick labels
    ax.XLabel.FontSize = 32; % x axis label
    ax.YLabel.FontSize = 32; % y axis label
    

    OutName = sprintf('%s%s.jpg', inputs.FigTab_out_sub, 'Landcover');
    % print(gcf, '-djpeg', '-r600', OutName);

check = 1;
end
%%
function check = RFplot(X_table,y,inputs,rf_model)
% Fit a regression tree model
y_pred = predict(rf_model, X_table);

% Compute statistics
R_squared = 1 - sum((y - y_pred).^2) / sum((y - mean(y)).^2);
RMSE = sqrt(mean((y_pred - y).^2));

fprintf('R-squared: %.3f\nRMSE: %.4f\n', R_squared, RMSE);

figure('Theme','light','Position',[100 100 800 600]);
hexbin_plot(y, y_pred, 60);
colormap(gca,flipud(bone))
cb = colorbar;
cb.FontSize = 16;
cb.Label.String = 'Density (log_{10}(count))';
cb.Label.FontSize = 22;
% cb.Label.FontWeight = 'bold';
axis tight; 
box off;

% 1:1 reference line
min_val = min(min(y), min(y_pred));
max_val = max(max(y), max(y_pred));
plot([min_val, max_val],[min_val, max_val], 'b--','LineWidth',2);

% Stats annotation
stat_text = sprintf('R^{2} = %.3f\nRMSE = %.4f', R_squared, RMSE);
text(min_val + 0.05*(max_val-min_val), max_val - 0.15*(max_val-min_val), stat_text,...
    'FontSize',22,'FontWeight','bold','BackgroundColor','w','EdgeColor','k');

% Formatting plot
set(gca,'FontSize',18)
xlabel('Observed \DeltaEVI','FontSize',22);
ylabel('Predicted \DeltaEVI','FontSize',22);
title(inputs.title1,'FontSize',22);
% legend('Data points','1:1 Line','Location','best');
% grid on;
hold off;

%% save
OutName = sprintf('%s%s.jpg', inputs.FigTab_out, inputs.title1);
print(gcf, '-djpeg', '-r600', OutName);

%% Plot the importance
% Extract predictor importance
importance = predictorImportance(rf_model);

% Normalize importance (optional but recommended)
importance = importance / sum(importance);

[sortedVals, sortIdx] = sort(importance, 'descend');
sortedLabels = inputs.xticklabels(sortIdx);

figure('Theme','light');
h = bar(sortedVals, 'FaceColor', [0.2 0.7 0.9]);
xticks(1:size(X_table,2));
xticklabels(sortedLabels);
% set(gca,'TickLabelInterpreter','latex')
xlabel('Predictors','FontSize',20);
ylabel('Relative Importance','FontSize',20);
title(inputs.title2,'FontSize',22);

%%
barNames = sortedLabels;

catColor  = [106, 121, 173]/255;  % Categorical: dist.year, Land Cover, Climate Zone
cliColor  = [162, 195, 234]/255;   % Climate: skt, prcp, solar rad, Δsm, Δskt, VPD, ΔVPD
vegColor  = [58, 103, 84]/255;   % Vegetation: EVI mean, AC1 mean
soilColor = [146, 116, 91]/255; 


barColors = zeros(length(sortedLabels),3); 

for i = 1:length(sortedLabels)
    name = sortedLabels{i};
    if ismember(name, {'EVI mean', 'AC1 mean','ΔAC1'})
        barColors(i,:) = vegColor;
    elseif ismember(name, {'prcp','Δprcp','T_{air}','ΔT_{air}','sm','Δsm','VPD','ΔVPD','solar rad'})
        barColors(i,:) = cliColor;
    elseif ismember(name, {'ph','sand','clay','nitrogen','soc'})
        barColors(i,:) = soilColor;
    else
        barColors(i,:) = catColor;  % If none of the above, assign categorical color
    end
end

h.FaceColor = 'flat'; % important: set to 'flat' to use CData
h.CData = barColors;

% grid on;
hold off

%% save
OutName = sprintf('%s%s.jpg', inputs.FigTab_out, inputs.title2);
% print(gcf, '-djpeg', '-r600', OutName);

check = 1
end

function hexbin_plot(x, y, nx)
    x = x(:); y = y(:);
    good = isfinite(x) & isfinite(y);
    x = x(good); y = y(good);

    % Set hex size based on x-range and desired bins across x
    xmin = min(x); xmax = max(x);
    dx = (xmax - xmin) / nx;        % hex width scale
    r  = dx / sqrt(3);              % hex "radius" (center to vertex)

    % Convert (x,y) to axial hex coordinates (q,r) using pointy-top layout
    % First shift so xmin starts at 0 for stability
    xs = (x - xmin) / r;
    ys = (y - min(y)) / r;

    q = (sqrt(3)/3) * xs - (1/3) * ys;
    s = (2/3) * ys;
    % cube coords: (q, r, -q-r) with r := s
    cube_x = q;
    cube_z = s;
    cube_y = -cube_x - cube_z;

    % Cube rounding
    rx = round(cube_x); ry = round(cube_y); rz = round(cube_z);
    x_diff = abs(rx - cube_x);
    y_diff = abs(ry - cube_y);
    z_diff = abs(rz - cube_z);

    fix_x = (x_diff > y_diff) & (x_diff > z_diff);
    fix_y = (y_diff > z_diff) & ~fix_x;
    fix_z = ~(fix_x | fix_y);

    rx(fix_x) = -ry(fix_x) - rz(fix_x);
    ry(fix_y) = -rx(fix_y) - rz(fix_y);
    rz(fix_z) = -rx(fix_z) - ry(fix_z);

    % Convert rounded cube coords back to centers in data coords
    % pointy-top hex axial -> cartesian
    cx = r * (sqrt(3) * rx + sqrt(3)/2 * rz) + xmin;
    cy = r * (3/2 * rz) + min(y);

    % Count points per hex center
    C = [cx, cy];
    [Cu, ~, ic] = unique(C, 'rows');
    counts = accumarray(ic, 1);

    % Draw hexagons
    hold on;
    theta = (0:5) * (pi/3) + pi/6;  % pointy-top
    hx = r * cos(theta);
    hy = r * sin(theta);

    % Color by counts (log helps if very dense)
    vals = log10(counts);   % try counts instead if you prefer linear
    for k = 1:size(Cu,1)
        xv = Cu(k,1) + hx;
        yv = Cu(k,2) + hy;
        patch(xv, yv, vals(k), 'EdgeColor', 'none'); % no borders looks like your 2nd fig
    end
    colormap(parula);
    set(gca, 'YDir', 'normal');
end
