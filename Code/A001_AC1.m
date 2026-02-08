clc
clear
close all

%% 
for seg_num_row = 0:9
    for seg_num_col = 0:9

    tic
    root_dir = '/Users/liyang/Documents/Study/';
    fn_EVI = sprintf('%sData/MOD13C1/mat/segment images/EVI_TS_seg_%d_%d.mat',root_dir,seg_num_row,seg_num_col)   
    Data1 = load(fn_EVI);
    EVI_TS = Data1.EVI_TS_seg;

    fn_t = sprintf('%sData/MOD13C1/mat/t_all.mat',root_dir)
    Data2 = load(fn_t);
    t_all = Data2.t_all;
    clear Data1 Data2

    row_num = 360;
    col_num = 720;
    pixel_num = row_num*col_num;

    data_per_year = 23;
    srchwin = 69; %3-year search window

    % time = t_all(:,4); 
    date = t_all(:,1:3);
    time = datetime(date);

    %%
    startYear = t_all(1,1);
    endYear = t_all(end,1);

    firstDate = datetime(date(1,:));

    % Initialize empty array for storing data dates
    MOD13C1_Dates = [];

    % Loop through each year
    for year = startYear:endYear
        % If it's the first year, start from Feb 18
        if year == startYear
            currentDate = firstDate;
        else
            % In subsequent years, restart from Jan 1
            currentDate = datetime(year,1,1);
        end

        % Generate 16-day intervals within the year
        while currentDate.Year == year && currentDate <= datetime(year,12,31)
            MOD13C1_Dates = [MOD13C1_Dates; currentDate]; % Store the date
            currentDate = currentDate + days(16); % Move to next 16-day interval
        end
    end

    MOD13C1_Dates.Format = 'yyyy-MM-dd';
    layers = length(MOD13C1_Dates);
    EVI_TS_all = nan(row_num, col_num, layers);

    [~, loc] = ismember(time, MOD13C1_Dates);
    [found, idx] = ismember(time, MOD13C1_Dates);
    EVI_TS_all(:,:,loc(loc > 0)) = EVI_TS;

    %% 
    result_out = sprintf('%sData/MOD13C1/mat/segment images full Forzieri 0.4/',root_dir)
    savename = sprintf('%sEVI_TS_seg_%d_%d.mat',result_out,seg_num_row,seg_num_col);
    save(savename,'EVI_TS_all','-v7.3');
    check=1;
        
    clear EVI_TS
    %%
    AC1_global = [];
    Residual_Forzieri = [];
    stationary_kpss = NaN(row_num,col_num);

    count = 0;
    parfor row = 1:row_num
        for col = 1:col_num

            EVI_TS_curSite = squeeze(EVI_TS_all(row,col,:));
            EVI_TS_curSite(EVI_TS_curSite<=0|EVI_TS_curSite>1)=NaN;
            if sum(isnan(EVI_TS_curSite))>(0.4*length(EVI_TS_curSite))
                pixelTS_AC1 = NaN(1,1,480); 
                AC1_global(row,col,:) = pixelTS_AC1;

                pixelTS_residual = NaN(1,1,549);
                Residual_Forzieri(row,col,:) = pixelTS_residual;
                continue
            end
            
            %% deseasonality
            % Extract Month Index
            DOY = day(MOD13C1_Dates,'dayofyear');
            unique_DOY = unique(DOY);

            % Compute Seasonal Mean and Standard Deviation
            Xmu = nan(data_per_year,1); % Preallocate
            Xsd = nan(data_per_year,1); % Preallocate

            for d = unique_DOY'
                % Find indices corresponding to DOY `d`
                doy_idx = (DOY == d);

                % Extract values for the same doy across years
                doy_values = EVI_TS_curSite(doy_idx);

                % Compute seasonal mean and standard deviation
                Xmu(d) = nanmean(doy_values);
                Xsd(d) = nanstd(doy_values);
            end

            % Deseasonalization
            EVI_deseasoned = nan(size(EVI_TS_curSite)); % Preallocate for deseasonalized EVI

            for d = unique_DOY'
                % Find indices corresponding to DOY `d`
                doy_idx = (DOY == d);

                % Subtract values mean to remove seasonality
                EVI_deseasoned(doy_idx) = EVI_TS_curSite(doy_idx) - Xmu(d);
            end

            %% Detrend
            pixelTS_residual = detrend(EVI_deseasoned,'omitnan'); % Remove linear trend
            Residual_Forzieri(row,col,:) = pixelTS_residual;
            h = kpsstest(pixelTS_residual,'Trend',true);
            stationary_kpss(row,col) = h;
            h
            %%
            % figure;
            % plot(MOD13C1_Dates, EVI_TS_curSite, '-b', 'DisplayName', 'Original EVI');
            % 
            % hold on
            % plot(MOD13C1_Dates, pixelTS_residual, '-g', 'DisplayName', 'Deseasonalized Detrended EVI');
            % close all

            %% calculate AC1
            pixelTS_AC1 = calculate_AC1(pixelTS_residual,srchwin);
            AC1_global(row,col,:) = pixelTS_AC1;
        end
    end

    %% save the EVI residual Forzieri
    output_dir = sprintf('%sData/MOD13C1/mat/EVI residual Forzieri 0.4/',root_dir);
    output_name = sprintf('%sEVI_residual_%d_%d.mat',output_dir,seg_num_row,seg_num_col);
    save(output_name,'Residual_Forzieri','MOD13C1_Dates','-v7.3')
    clear Residual_Forzieri stationary_kpss EVI_TS_all
    
    %% save the timing of the moving-window AC1 of the current pixel   
    pixelTS_AC1_t_start = ceil(1/2.*srchwin)+1;
    pixelTS_AC1_t_end   = length(MOD13C1_Dates)-floor(1/2.*srchwin);
    pixelTS_AC1_t       = MOD13C1_Dates(pixelTS_AC1_t_start:pixelTS_AC1_t_end,:);
    AC1_global_t = pixelTS_AC1_t;
    
    output_dir = sprintf('%sNUS/Forest Resilience/Results/AC1/Global Forzieri 0.4/',root_dir);
    output_name = sprintf('%sAC1_global_%d_%d.mat',output_dir,seg_num_row,seg_num_col);
    save(output_name,'AC1_global','AC1_global_t','-v7.3')

    clear AC1_global AC1_global_t
    toc
    end
end

%% save the revised time->MOD13C1_Dates
clear year t_all
t_all = NaN(length(MOD13C1_Dates),4);
t_all(:,1) = year(MOD13C1_Dates);
t_all(:,2) = month(MOD13C1_Dates);
t_all(:,3) = day(MOD13C1_Dates);
t_all(:,4) = t_all(:,1)+(datenum(t_all(:,1),t_all(:,2),t_all(:,3))-datenum(t_all(:,1),1,0))./365.25;

output_dir = sprintf('%sData/MOD13C1/mat/',root_dir);
output_name = sprintf('%st_all_full.mat',output_dir);
save(output_name,'t_all','-v7.3')


%%
function AC1 = calculate_AC1(TS_residual,srchwin)
    AC1 = [];
    for srchwin_start = 1:size(TS_residual,1)-srchwin
        TS_residual_srchwin = TS_residual(srchwin_start:srchwin_start+srchwin,1);
    
        % Calculate the mean of the time series
        mean_residual_srchwin = mean(TS_residual_srchwin,'omitnan');
    
        % Calculate the numerator of the autocorrelation formula
        numerator = sum(((TS_residual_srchwin(1:end-1)-mean_residual_srchwin).* (TS_residual_srchwin(2:end)-mean_residual_srchwin)),'omitnan');
    
        % Calculate the denominator of the autocorrelation formula
        denominator = sum(((TS_residual_srchwin-mean_residual_srchwin).^2),'omitnan');
    
        % Calculate the lag-1 autocorrelation
        AC1_curSrchWin = numerator./denominator;

        AC1 = cat(1,AC1,AC1_curSrchWin);
    end
end
