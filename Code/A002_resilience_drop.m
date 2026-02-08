clc
clear
close all

root_dir = '/Users/liyang/Documents/Study/';
addpath('/Users/liyang/Documents/Study/NUS/Forest Resilience/Code/matlab tools')
in_word  = 'AC1 sudden drop Forzieri 0.5 th02 1bpstep air';
out_word = 'AC1 sudden drop Forzieri 0.5 th02 1bpstep air historical';
Twindow = '1bpstep';

%%
startDate = datetime(2000,1,15);
endDate = datetime(2023,12,15);
dates = startDate:calmonths(1):endDate;
clim_t = [year(dates)', month(dates)', day(dates)'];
clim_t_datetime = datetime(clim_t);

%%
tic

for seg_num_row = 0:9
    for seg_num_col = 0:9

        delta_prcp_1bpstep = {};
        delta_air_temp_1bpstep = {};
        delta_VPD_1bpstep = {};
        delta_sm_1bpstep = {};

        post_prcp_1bpstep = {};
        post_air_temp_1bpstep = {};
        post_VPD_1bpstep = {};
        post_sm_1bpstep = {};
        post_dnrad_1bpstep = {};

        % resilience drop
        fn_resilience_drop_start = sprintf('%sNUS/Forest Resilience/Results/%s/Resilience drop timing start/Resilience_drop_timing_start_%d_%d.mat',root_dir,in_word,seg_num_row,seg_num_col);
        Resilience_drop_timing_start = loadMatData(fn_resilience_drop_start);
        fn_resilience_drop_end = sprintf('%sNUS/Forest Resilience/Results/%s/Resilience drop timing end/Resilience_drop_timing_end_%d_%d.mat',root_dir,in_word,seg_num_row,seg_num_col);
        Resilience_drop_timing_end = loadMatData(fn_resilience_drop_end);

        % clim data
        prcp_dir = sprintf('%sData/TerraClimate/historical/mat/ppt/segments/precipitation_%d_%d.mat',root_dir,seg_num_row,seg_num_col);
        prcp_seg = loadMatData(prcp_dir);

        air_temp_dir = sprintf('%sData/TerraClimate/historical/mat/temp/segments/air_temp_%d_%d.mat',root_dir,seg_num_row,seg_num_col);
        air_temp_seg = loadMatData(air_temp_dir);

        VPD_dir = sprintf('%sData/TerraClimate/historical/mat/vpd/segments/VPD_%d_%d.mat',root_dir,seg_num_row,seg_num_col);
        VPD_seg = loadMatData(VPD_dir);

        dnrad_dir = sprintf('%sData/TerraClimate/historical/mat/srad/segments/downward_rad_%d_%d.mat',root_dir,seg_num_row,seg_num_col);
        dnrad_seg = loadMatData(dnrad_dir);

        sm_dir = sprintf('%sData/TerraClimate/historical/mat/soil/segments/soil_moisture_%d_%d.mat',root_dir,seg_num_row,seg_num_col);
        sm_seg = loadMatData(sm_dir);

        if all(cellfun(@isempty, Resilience_drop_timing_start), 'all')  
            delta_prcp_1bpstep = repmat({[]}, 360, 720);
            delta_air_temp_1bpstep = repmat({[]}, 360, 720);
            delta_VPD_1bpstep = repmat({[]}, 360, 720);
            delta_sm_1bpstep = repmat({[]}, 360, 720);

            post_prcp_1bpstep = repmat({[]}, 360, 720);
            post_air_temp_1bpstep = repmat({[]}, 360, 720);
            post_VPD_1bpstep = repmat({[]}, 360, 720);
            post_dnrad_1bpstep = repmat({[]}, 360, 720);
            post_sm_1bpstep = repmat({[]}, 360, 720);

            %%
            result_out = sprintf('%sNUS/Forest Resilience/Results/%s/delta prcp/',root_dir,out_word);
            if ~(exist(result_out,'dir') ==7)
                mkdir(result_out);
            end

            savename = sprintf('%sdelta_prcp_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
            save(savename,'delta_prcp_1bpstep','-v7.3');

            result_out = sprintf('%sNUS/Forest Resilience/Results/%s/post prcp/',root_dir,out_word);
            if ~(exist(result_out,'dir') ==7)
                mkdir(result_out);
            end

            savename = sprintf('%spost_prcp_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
            save(savename,'post_prcp_1bpstep','-v7.3');

            clear delta_prcp_1bpstep post_prcp_1bpstep
            %%
            result_out = sprintf('%sNUS/Forest Resilience/Results/%s/delta air temp/',root_dir,out_word);
            if ~(exist(result_out,'dir') ==7)
                mkdir(result_out);
            end

            savename = sprintf('%sdelta_air_temp_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
            save(savename,'delta_air_temp_1bpstep','-v7.3');

            result_out = sprintf('%sNUS/Forest Resilience/Results/%s/post air temp/',root_dir,out_word);
            if ~(exist(result_out,'dir') ==7)
                mkdir(result_out);
            end

            savename = sprintf('%spost_air_temp_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
            save(savename,'post_air_temp_1bpstep','-v7.3');

            clear delta_air_temp_1bpstep post_air_temp_1bpstep
            %%
            result_out = sprintf('%sNUS/Forest Resilience/Results/%s/delta VPD/',root_dir,out_word);
            if ~(exist(result_out,'dir') ==7)
                mkdir(result_out);
            end

            savename = sprintf('%sdelta_VPD_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
            save(savename,'delta_VPD_1bpstep','-v7.3');

            result_out = sprintf('%sNUS/Forest Resilience/Results/%s/post VPD/',root_dir,out_word);
            if ~(exist(result_out,'dir') ==7)
                mkdir(result_out);
            end

            savename = sprintf('%spost_VPD_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
            save(savename,'post_VPD_1bpstep','-v7.3');

            clear delta_VPD_1bpstep post_VPD_1bpstep

            %%
            result_out = sprintf('%sNUS/Forest Resilience/Results/%s/delta sm/',root_dir,out_word);
            if ~(exist(result_out,'dir') ==7)
                mkdir(result_out);
            end

            savename = sprintf('%sdelta_sm_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
            save(savename,'delta_sm_1bpstep','-v7.3');

            result_out = sprintf('%sNUS/Forest Resilience/Results/%s/post sm/',root_dir,out_word);
            if ~(exist(result_out,'dir') ==7)
                mkdir(result_out);
            end

            savename = sprintf('%spost_sm_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
            save(savename,'post_sm_1bpstep','-v7.3');

            clear delta_sm_1bpstep post_sm_1bpstep

            %%
            result_out = sprintf('%sNUS/Forest Resilience/Results/%s/post dnrad/',root_dir,out_word);
            if ~(exist(result_out,'dir') ==7)
                mkdir(result_out);
            end

            savename = sprintf('%spost_dnrad_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
            save(savename,'post_dnrad_1bpstep','-v7.3');

            clear post_dnrad_1bpstep
        end

        %%
        row_num = 360;
        col_num = 720;

        %%
        parfor row = 1:row_num
            for col = 1:col_num

                %% resilience drop
                Resilience_drop_timing_start_cur = Resilience_drop_timing_start{row,col};
                Resilience_drop_timing_end_cur   = Resilience_drop_timing_end{row,col};
                Resilience_drop_count = length(Resilience_drop_timing_start_cur);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if Resilience_drop_count <1

                    delta_prcp_1bpstep{row,col}  = [];
                    delta_air_temp_1bpstep{row,col}  = [];
                    delta_VPD_1bpstep{row,col}  = [];
                    delta_sm_1bpstep{row,col}  = [];

                    post_prcp_1bpstep{row,col}  = [];
                    post_air_temp_1bpstep{row,col}  = [];
                    post_VPD_1bpstep{row,col}  = [];
                    post_dnrad_1bpstep{row,col}  = [];
                    post_sm_1bpstep{row,col}  = [];
                    continue
                end

                %% AC1
                prcp_TS          = squeeze(prcp_seg(row,col,:));
                air_temp_TS      = squeeze(air_temp_seg(row,col,:));
                VPD_TS           = squeeze(VPD_seg(row,col,:));
                dnrad_TS         = squeeze(dnrad_seg(row,col,:));
                sm_TS            = squeeze(sm_seg(row,col,:));

                %% delta AC1 (AC1_post-AC1_pre),del
                delta_prcp_1bpstep_cur  = [];
                delta_air_temp_1bpstep_cur  = [];
                delta_VPD_1bpstep_cur  = [];
                delta_sm_1bpstep_cur  = [];

                post_prcp_1bpstep_cur = [];
                post_air_temp_1bpstep_cur = [];
                post_VPD_1bpstep_cur = [];
                post_dnrad_1bpstep_cur = [];
                post_sm_1bpstep_cur = [];
                %%
                for i = 1:Resilience_drop_count
                    timing_start = Resilience_drop_timing_start_cur(1,i);
                    timing_start_yy = floor(timing_start);
                    fractional_part = timing_start - timing_start_yy;
                    days_in_year = 365.25;
                    day_of_year = round(fractional_part * days_in_year);
                    timing_start_date = datetime(timing_start_yy, 1, 0) + days(day_of_year);
                    timing_start_yy = year(timing_start_date);
                    timing_start_mm = month(timing_start_date);
                    timing_start_dd = day(timing_start_date);

                    timing_end   = Resilience_drop_timing_end_cur(1,i);
                    timing_end_yy = floor(timing_end);
                    fractional_part = timing_end - timing_end_yy;
                    day_of_year = round(fractional_part * days_in_year);
                    timing_end_date = datetime(timing_end_yy, 1, 0) + days(day_of_year);
                    timing_end_yy = year(timing_end_date);
                    timing_end_mm = month(timing_end_date);
                    timing_end_dd = day(timing_end_date);

                    % AC1_global_t from 2001-8-29 to 2022-6-26
                    if timing_start_date<datetime(2002,8,29)|timing_end_date>datetime(2021,06,26)
                        continue
                    end

                    %% 1 year diff
                    pre_1bpstep =  [datetime(timing_start_yy-1,timing_start_mm,timing_start_dd),datetime(timing_end_yy-1,timing_end_mm,timing_end_dd)];
                    post_1bpstep = [datetime(timing_start_yy+1,timing_start_mm,timing_start_dd),datetime(timing_end_yy+1,timing_end_mm,timing_end_dd)];

                    clim_idx_pre_1bpstep    = (clim_t_datetime >= pre_1bpstep(1)) & (clim_t_datetime < pre_1bpstep(2));
                    clim_idx_post_1bpstep   = (clim_t_datetime > post_1bpstep(1)) & (clim_t_datetime <= post_1bpstep(2));

                    if sum(clim_idx_pre_1bpstep)==0|sum(clim_idx_post_1bpstep)==0
                        continue
                    end

                    %%
                    % prcp
                    prcp_pre_1bpstep       = mean(prcp_TS(clim_idx_pre_1bpstep),'omitnan');
                    prcp_post_1bpstep      = mean(prcp_TS(clim_idx_post_1bpstep),'omitnan');
                    diff_prcp_1bpstep      = prcp_post_1bpstep-prcp_pre_1bpstep;
                    delta_prcp_1bpstep_cur = cat(2,delta_prcp_1bpstep_cur,diff_prcp_1bpstep);
                    post_prcp_1bpstep_cur  = cat(2,post_prcp_1bpstep_cur,prcp_post_1bpstep);

                    % air temp
                    air_temp_pre_1bpstep       = mean(air_temp_TS(clim_idx_pre_1bpstep),'omitnan');
                    air_temp_post_1bpstep      = mean(air_temp_TS(clim_idx_post_1bpstep),'omitnan');
                    diff_air_temp_1bpstep      = air_temp_post_1bpstep-air_temp_pre_1bpstep;
                    delta_air_temp_1bpstep_cur = cat(2,delta_air_temp_1bpstep_cur,diff_air_temp_1bpstep);
                    post_air_temp_1bpstep_cur  = cat(2,post_air_temp_1bpstep_cur,air_temp_post_1bpstep);

                    % VPD
                    VPD_pre_1bpstep       = mean(VPD_TS(clim_idx_pre_1bpstep),'omitnan');
                    VPD_post_1bpstep      = mean(VPD_TS(clim_idx_post_1bpstep),'omitnan');
                    diff_VPD_1bpstep      = VPD_post_1bpstep-VPD_pre_1bpstep;
                    delta_VPD_1bpstep_cur = cat(2,delta_VPD_1bpstep_cur,diff_VPD_1bpstep);
                    post_VPD_1bpstep_cur  = cat(2,post_VPD_1bpstep_cur,VPD_post_1bpstep);

                    % dnrad
                    dnrad_post_1bpstep      = mean(dnrad_TS(clim_idx_post_1bpstep),'omitnan');
                    post_dnrad_1bpstep_cur  = cat(2,post_dnrad_1bpstep_cur,dnrad_post_1bpstep);

                    % sm
                    sm_pre_1bpstep       = mean(sm_TS(clim_idx_pre_1bpstep),'omitnan');
                    sm_post_1bpstep      = mean(sm_TS(clim_idx_post_1bpstep),'omitnan');
                    diff_sm_1bpstep      = sm_post_1bpstep-sm_pre_1bpstep;
                    delta_sm_1bpstep_cur = cat(2,delta_sm_1bpstep_cur,diff_sm_1bpstep);
                    post_sm_1bpstep_cur  = cat(2,post_sm_1bpstep_cur,sm_post_1bpstep);


                end
                
                %%
                delta_prcp_1bpstep{row,col}  = delta_prcp_1bpstep_cur;
                delta_air_temp_1bpstep{row,col}  = delta_air_temp_1bpstep_cur;
                delta_VPD_1bpstep{row,col}  = delta_VPD_1bpstep_cur;
                delta_sm_1bpstep{row,col}  = delta_sm_1bpstep_cur;

                post_prcp_1bpstep{row,col}  = post_prcp_1bpstep_cur;
                post_air_temp_1bpstep{row,col}  = post_air_temp_1bpstep_cur;
                post_VPD_1bpstep{row,col}  = post_VPD_1bpstep_cur;
                post_dnrad_1bpstep{row,col}  = post_dnrad_1bpstep_cur;
                post_sm_1bpstep{row,col}  = post_sm_1bpstep_cur;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            end
        end


        %%
        result_out = sprintf('%sNUS/Forest Resilience/Results/%s/delta prcp/',root_dir,out_word);
        if ~(exist(result_out,'dir') ==7)
            mkdir(result_out);
        end

        savename = sprintf('%sdelta_prcp_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
        save(savename,'delta_prcp_1bpstep','-v7.3');

        result_out = sprintf('%sNUS/Forest Resilience/Results/%s/post prcp/',root_dir,out_word);
        if ~(exist(result_out,'dir') ==7)
            mkdir(result_out);
        end

        savename = sprintf('%spost_prcp_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
        save(savename,'post_prcp_1bpstep','-v7.3');

        clear delta_prcp_1bpstep post_prcp_1bpstep
        %%
        result_out = sprintf('%sNUS/Forest Resilience/Results/%s/delta air temp/',root_dir,out_word);
        if ~(exist(result_out,'dir') ==7)
            mkdir(result_out);
        end

        savename = sprintf('%sdelta_air_temp_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
        save(savename,'delta_air_temp_1bpstep','-v7.3');

        result_out = sprintf('%sNUS/Forest Resilience/Results/%s/post air temp/',root_dir,out_word);
        if ~(exist(result_out,'dir') ==7)
            mkdir(result_out);
        end

        savename = sprintf('%spost_air_temp_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
        save(savename,'post_air_temp_1bpstep','-v7.3');

        clear delta_air_temp_1bpstep post_air_temp_1bpstep
        %%
        result_out = sprintf('%sNUS/Forest Resilience/Results/%s/delta VPD/',root_dir,out_word);
        if ~(exist(result_out,'dir') ==7)
            mkdir(result_out);
        end

        savename = sprintf('%sdelta_VPD_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
        save(savename,'delta_VPD_1bpstep','-v7.3');

        result_out = sprintf('%sNUS/Forest Resilience/Results/%s/post VPD/',root_dir,out_word);
        if ~(exist(result_out,'dir') ==7)
            mkdir(result_out);
        end

        savename = sprintf('%spost_VPD_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
        save(savename,'post_VPD_1bpstep','-v7.3');

        clear delta_VPD_1bpstep post_VPD_1bpstep

        %%
        result_out = sprintf('%sNUS/Forest Resilience/Results/%s/post dnrad/',root_dir,out_word);
        if ~(exist(result_out,'dir') ==7)
            mkdir(result_out);
        end

        savename = sprintf('%spost_dnrad_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
        save(savename,'post_dnrad_1bpstep','-v7.3');

        clear post_dnrad_1bpstep

        %%
        result_out = sprintf('%sNUS/Forest Resilience/Results/%s/delta sm/',root_dir,out_word);
        if ~(exist(result_out,'dir') ==7)
            mkdir(result_out);
        end

        savename = sprintf('%sdelta_sm_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
        save(savename,'delta_sm_1bpstep','-v7.3');

        result_out = sprintf('%sNUS/Forest Resilience/Results/%s/post sm/',root_dir,out_word);
        if ~(exist(result_out,'dir') ==7)
            mkdir(result_out);
        end

        savename = sprintf('%spost_sm_1bpstep_%d_%d.mat',result_out,seg_num_row,seg_num_col);
        save(savename,'post_sm_1bpstep','-v7.3');

        clear delta_sm_1bpstep post_sm_1bpstep
    end
end
toc
check = 1;
% end
