% Execute this file with MATLAB
% (Ideally I would have made it compatible with GNU Octave, but the dates and tables make it difficult.)

% Code written by PA Absil. There is NO WARRANTY of prediction accuracy.

% SHR_01PA.m - Started by PA on Sun 05 Jul 2020
% SHR_02PA.m - Started by PA on Sun 05 Jul 2020
%    Yields a remarkably good fit.
% SHR_03PA.m - Started by PA on Sun 05 Jul 2020
%    Version sent to Ousmane
% SHR_04PA.m - Started by PA on Sun 05 Jul 2020
% SHR_05PA.m - Started by PA on Thu 09 Jul 2020
%    Prepare more systematic experiments.
% SHR_06PA.m - Started by PA on Thu 09 Jul 2020
%    Try to estimate gamma by optimization too, and put new_in and new_out in the cost.
%    Figure 7 is quite remarkable for the prediction. I record this version.
% SHR_07PA.m - Started by PA on Thu 09 Jul 2020
%    Go on, trying to find suitable values for c_H, c_E, c_L.
%    Figure 7 shows a rather good fitting over the whole range of 100 values. I record this version.
% SHR_08PA.m - Started by PA on Thu 09 Jul 2020
%    Initialize 3D optimization with 2D optimum.
%    Compute RMSE_test and RMSE_train.
%    I record this version, which gives an MAPE_test of 4.5114%
% SHR_09PA.m - Started by PA on Fri 10 Jul 2020
%    ! Before this version, the cost function was using the raw new_out.
%    Improve function syntax.
%    Try H(0) as decision variable.
%    Set tolerance for fminsearch.
%    I record this version, which give an MAPE_test of 5.7255%
% SHR_10PA.m - Started by PA on Fri 10 Jul 2020
%    Plot computed H for several train periods.
%    This version shows an interesting plot in Figure 8.
% SHR_11PA.m - Started by PA on Fri 20 Jul 2020
%    Consider French data.
% SHR_11PA.m - Started by PA on Fri 20 Jul 2020
%    Use several train_t_start **and train_t_end**.
% SHR_12PA.m - Started by PA on Sat 11 Jul 2020
%    Play around with various French departments.
% SHR_12PA.m - Started by PA on Mon 13 Jul 2020
%    Try polynomial fitting?

%    PATODO:   Shield train phase from test data even better.



sw_dataset = 'BEL';  % !! % 'BEL' or 'FRA'
department = 'all';  % Choose French department number, e.g., '75'. Choose 'all' to sum all departments.
force_recreate_data = 0;  % If 1, forces to recreate mat file from csv file.
                          % !! Make sure you set it to 1 if you have changed the data file.
test_duration = 10;  % Duration of the test period.
show_totinout = 0;  % If 1, shows plots of total, in and out and check their discrepancy.
save_figures = 0;  % If 1, some figures will be saved in fig and eps format.

switch sw_dataset
    case 'BEL'
% ***********************************************************************************
% Load data Belgium
% *******

% The data comes from https://epistat.sciensano.be/Data/COVID19BE_HOSP.csv.
% This link was provided on 22 June 2020 by Alexey Medvedev on the "Re R0 estimation" channel of the O365G-covidata team on MS-Teams.
% The link can be reached from https://epistat.wiv-isp.be/covid/
% Some explanations can be found at https://epistat.sciensano.be/COVID19BE_codebook.pdf

if force_recreate_data | ~isfile(['MAT_files/data_saved_BEL.mat'])
    recreate_data = 1;
else
    recreate_data = 0;
end
recreate_data

if recreate_data
    
    % Load the data:
    % data_raw = readtable('COVID19BE_HOSP_2020-07-05.csv', 'HeaderLines',1);
    data_raw = readtable('Belgium/COVID19BE_HOSP_2020-07-12.csv');

    % Merge provinces into table "data":
    clear data;
    data = data_raw(1,[1,4:10]);  
    i_new = 1;  % current row in table "data"
    i_old = 1;  % current row in table "data_raw"
    data(i_new,:) = data_raw(i_old,[1,4:10]);
    for i_old = 2:size(data_raw,1)    % descend in the rows of table "data_raw"
        if table2array(data_raw(i_old,1)) == table2array(data_raw(i_old-1,1))  % if the date is the same as on the previous row
            data(i_new,2:8) = array2table( table2array(data(i_new,2:8)) + table2array(data_raw(i_old,4:10)) );   % accumulate the values
        else
            i_new = i_new + 1;
            data(i_new,:) = data_raw(i_old,[1,4:10]);  % otherwise create a new row in "data" and put it the values of the current row
        end
    end

    save('MAT_files/data_saved_BEL','data');

else   % load "data" from MAT-file
    load('MAT_files/data_saved_BEL','data');   % load table "data"
end
    
% Extract relevant data and recompute new_out:
% Source: Some variable names taken from https://rpubs.com/JMBodart/Covid19-hosp-be
data_length = size(data,1);
data_num = table2array(data(:,2:end));
dates = table2array(data(:,1));
col_total_in = 2; col_new_in = 6; col_new_out = 7;
total_in = data_num(:,col_total_in);
new_in = data_num(:,col_new_in);
new_out_raw = data_num(:,col_new_out);   % there will be a non-raw due to the "Problem" mentioned below.
new_delta = new_in - new_out_raw;
cum_new_delta = cumsum(new_delta);
total_in_chg = [0;diff(total_in)];
% Problem: new_delta and total_in_chg are different, though they are sometimes close. 
% Cum_new_delta does not go back to something close to zero, whereas it should. Hence I should not trust it.
% I'm going to trust total_in and new_in. I deduce new_out_fixed by:
new_out = new_in - total_in_chg;   % fixed new_out

data_totinout = [total_in,new_in,new_out];


% ***********************************************************************************
% Load data France
% *******
    case 'FRA'  % if sw_dataset is 'FRA'
    
% The data comes from https://www.data.gouv.fr/en/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/, see donnees-hospitalieres-covid19-2020-07-10-19h00.csv

if force_recreate_data | ~isfile(['MAT_files/data_saved_FRA_',department,'.mat'])
    recreate_data = 1;
else
    recreate_data = 0;
end
recreate_data

if recreate_data

    % Load the data:
    % data_raw = readtable('COVID19BE_HOSP_2020-07-05.csv', 'HeaderLines',1);
    data_raw = readtable('France/donnees-hospitalieres-covid19-2020-07-10-19h00_corrected.csv');
    % "corrected" because some dates had the format dd/mm/yyyy instead of yyyy-mm-dd. Moreover, 2020-07-03 was repeated.

    clear data;
    i_new = 0;
    
    if strcmp(department,'all')    % if we sum all departments
        date_prev = '';
        for i_old = 1:size(data_raw,1)    % descend in the rows of table "data_raw"
            if data_raw.sexe(i_old) == 0;   % Consider only sex "0", i.e., total female+male
                if data_raw.jour(i_old) == date_prev  % if the date is the same as the previous one
                    data(i_new,4:7) = array2table( table2array(data(i_new,4:7)) + table2array(data_raw(i_old,4:7)) );   % accumulate the values
                else
                    i_new = i_new + 1;
                    data(i_new,:) = data_raw(i_old,:);  % otherwise create a new row in "data" and put it the values of the current row
                    data(i_new,1) = {''};  % erase department
                end
                date_prev = data_raw.jour(i_old);
            end
        end
    else   % in this case, department is a specific department number
        for i_old = 1:size(data_raw,1)
            if strcmp(data_raw.dep{i_old},department) & data_raw.sexe(i_old) == 0
                i_new = i_new + 1;
                data(i_new,:) = data_raw(i_old,:); 
            end
        end
    end
    

    save(['MAT_files/data_saved_FRA_',department],'data');

else   % load "data" from MAT-file
    load(['MAT_files/data_saved_FRA_',department],'data');   % load table "data"
end

% Extract relevant data and recompute new_out:
% Source: Some variable names taken from https://rpubs.com/JMBodart/Covid19-hosp-be
data_length = size(data,1);
data_num = table2array(data(:,4:end));
dates = table2array(data(:,3));
total_in = data_num(:,1);
new_out = [0;diff(data_num(:,3) + data_num(:,4))];
total_in_chg = [0;diff(total_in)];
new_in = new_out + total_in_chg;

data_totinout = [total_in,new_in,new_out];

end  % end switch


% ***********************************************************************************
% Select train periods
% *******

train_t_start_vals = (length(total_in)-test_duration) - [30:-1:28];    
train_t_start_vals = (length(total_in)-test_duration) - [90:-1:85]; % !! 
train_t_end_vals = (length(total_in)-test_duration)*ones(size(train_t_start_vals));

% Use this to override the train interval specification:
%train_t_start_vals = [11:24];    
% Sugg: 18 (meaning day 18 of the data) or [18:24] (meaning days 18 to 24)
%train_t_end_vals = train_t_start_vals + 14;
%train_t_start_vals = [1];  % !!
%train_t_end_vals = length(total_in)-1;

% ***********************************************************************************
% Data plots
% *******

close('all')
fig_cnt = 0;

if strcmp(sw_dataset,'BEL') & show_totinout
    fig_cnt = fig_cnt + 1;
    figure(fig_cnt)
    plot(dates,total_in,dates,cum_new_delta);
    leg = legend('total_in','sum(new_in-new_out)');
    set(leg,'Interpreter','none')  % to avoid interpreting "_" as subscript symbol

    fig_cnt = fig_cnt + 1;
    figure(fig_cnt)
    plot(dates,new_delta,dates,total_in_chg);
    leg = legend('new_delta','total_in_chg');
    set(leg,'Interpreter','none')


    fig_cnt = fig_cnt + 1;
    figure(fig_cnt)
    plot(dates,new_out_raw,dates,new_out);
    leg = legend('new_out_raw','new_out');
    set(leg,'Interpreter','none')
end


fig_cnt_mark = fig_cnt;

% {{{{{{{ 
% Start loop on train_t_start values

for period_cnt = 1:length(train_t_start_vals)
    
    fig_cnt = fig_cnt_mark;   % restart figure counter
    
    
    % ***********************************************************************************
    % Train/test data split
    % *******

    % Specify start and end of train period:
    %train_t_start = 18; % t_i in paper % sugg: 18 (1st of April) (Works well with 18 or larger)
    %train_t_end = train_t_start + train_duration;  % t_f in paper  % sugg: +14

    %train_t_start = 1; % t_i in paper % sugg: 18 (Works well with 18 or larger)
    %train_t_end = train_t_start+12;  % t_f in paper % sugg: +14

    train_t_start = train_t_start_vals(period_cnt);
    train_t_end = train_t_end_vals(period_cnt);

    tspan_train = [train_t_start,train_t_end];

    % The test data is defined to be all the data that occurs after train_t_end.

    % Replace test data by NaN in *_train variables.
    total_in_train = total_in; total_in_train(train_t_end+1:end) = NaN;   
    new_in_train = new_in; new_in_train(train_t_end+1:end) = NaN;
    new_out_train = new_out; new_out_train(train_t_end+1:end) = NaN;
    data_totinout_train = data_totinout; data_totinout_train(train_t_end+1:end,:) = NaN;
    % ! Make sure to use only these _train variables in the train phase.


    % ***********************************************************************************
    % Estimate gamma
    % *******

    % Model for gamma: new_out = gamma * total_in

    % Estimator by ratio of means:
    %gamma_hat_RM = sum(new_out(1:train_t_end))/sum(total_in(1:train_t_end));
    gamma_hat_RM = sum(new_out_train(1:train_t_end))/sum(total_in_train(1:train_t_end));

    % Estimator by least squares:
    gamma_hat_LS = total_in_train(1:train_t_end)\new_out_train(1:train_t_end);

    gamma_hat_RM
    gamma_hat_LS

    % Estimator by ratio of means on all data (test and train):  not legitimate
    %gamma_hat_all_RM = sum(new_out_train)/sum(total_in_train);

    % Estimator by least squares on all data (test and train):
    % gamma_hat_all_LS = total_in\new_out;
    % gamma_hat_all_RM
    % gamma_hat_all_LS

    % I obtain this:
    % gamma_hat_RM =
    %     0.0694
    % gamma_hat_LS =
    %     0.0683
    % They are thus quite close. Let's keep:
    gamma = gamma_hat_RM;
    %gamma = gamma_hat_all_RM;  % not legitimate

    % ***********************************************************************************
    % Objective function for the estimation of bar_beta and S_bar_init
    % *******

    c_H = 1; c_E = 0; c_L = 0;  % !! % coefficients of the terms of the cost function. Default: c_H = 1; c_E = 1; c_L = 1 (it gives a good MAPE_test)
    c_HEL = [c_H,c_E,c_L];

    fun = @(x)phi(x,data_totinout_train,tspan_train,gamma,c_HEL);  % function phi is given at the end of this script

    % Try a few values of the parameters in order to figure out a suitable initial point for fminsearch:
    %x = [1e-1/2e4,2e4];  % change this as you wish. Idea: [1e-1/1e4,1e4], [1e-1/2e4,2e4]
    %fun(x)

    % * Plot the objective function:

    % beta_bar_vals = linspace(1e-1/1e5,1e0/1e4,10);
    % S_bar_init_vals = linspace(1e4,1e5,10);

    % beta_bar_vals = linspace(1e-1/1e5,1e0/1e4,10);
    % S_bar_init_vals = linspace(1e4,2e4,10);

    beta_bar_vals = linspace(2e-6,40e-6,10);
    S_bar_init_vals = linspace(1e3,2e4,10);

    % beta_bar_vals = linspace(1e-3,1e-2,10);
    % S_bar_init_vals = linspace(1e4,1e5,10);

    [X,Y] = meshgrid(beta_bar_vals,S_bar_init_vals);
    Z = NaN(size(X));
    for i = 1:size(X,1)
        for j = 1:size(X,2);
            Z(i,j) = fun([X(i,j),Y(i,j)]);
        end
    end
    fig_cnt = fig_cnt + 1;
    figure(fig_cnt)
    %contour(X,Y,Z);
    contourf(X,Y,Z);

    fig_name_root = ['Figures/',mfilename,'_',sw_dataset,'_traintstart',num2str(train_t_start_vals(1)),'_traintstop',num2str(train_t_end_vals(1)),'_c',num2str(c_H),num2str(c_E),num2str(c_L)];

    
    % ***********************************************************************************
    % Optimization on beta_bar and S_bar_init
    % *******

    x_init = [1e-5,1e4];  % sugg: [2e-5,1e4]
    fun_init = fun(x_init);
    fminsearch_options.TolFun = fun_init * 1e-6; 
    fminsearch_options.TolX = 0;
    fminsearch_options.MaxFunEvals = 1e6;
    fminserach_options.MaxIter = 1e6;
    x_opt = fminsearch(fun,x_init,fminsearch_options);
    beta_bar_opt = x_opt(1); S_bar_init_opt = x_opt(2);
    fun_opt = fun(x_opt);  % value of the minimum

    % Plot the objective function around the found minimizer:
    beta_bar_vals = linspace(x_opt(1)/2,x_opt(1)*2,100);
    S_bar_init_vals = linspace(x_opt(2)/2,x_opt(2)*2,100);
    beta_bar_vals = linspace(0,x_opt(1)*2,100);
    S_bar_init_vals = linspace(0,x_opt(2)*2,100);
    [X,Y] = meshgrid(beta_bar_vals,S_bar_init_vals);
    Z = NaN(size(X));
    for i = 1:size(X,1)
        for j = 1:size(X,2);
            % Z(i,j) = fun([X(i,j),Y(i,j)]);
            Z(i,j) = log10(fun([X(i,j),Y(i,j)])-fun_opt*.99);
        end
    end
    fig_cnt = fig_cnt + 1;
    figure(fig_cnt)
    %contour(X,Y,Z);
    contourf(X,Y,Z);
    %title('log(fun - fun\_opt*.99))')
    if save_figures 
        saveas(gcf,[fig_name_root,'_contour'],'fig');
        saveas(gcf,[fig_name_root,'_contour'],'epsc');        
    end

    % Plot true and simulated H, and simulated S_bar:
    fig_cnt = fig_cnt + 1;
    figure(fig_cnt)
    res = results(beta_bar_opt,gamma,S_bar_init_opt,total_in(train_t_start),tspan_train,dates,data_totinout);  % results() does the plot, and returns info in res.
    disp(['MAPE_train (beta_bar and S_bar_init optimized): ', num2str(res.MAPE_train)])
    disp(['MAPE_test (beta_bar and S_bar_init optimized): ', num2str(res.MAPE_test)])
    %title('Optimized wrt beta\_bar and S\_bar\_init')
    if save_figures 
        saveas(gcf,[fig_name_root,'_2Dopt'],'fig');
        saveas(gcf,[fig_name_root,'_2Dopt'],'epsc');        
    end
    


    % ***********************************************************************************
    % Optimization on beta_bar, S_bar_init, and gamma
    % *******

    fun_gamma = @(x)phi_gamma(x,data_totinout_train,tspan_train,c_HEL);  % function phi is given at the end of this script
    x_init_gamma = [x_opt,gamma_hat_RM];  % sugg: [x_opt,gamma_hat_RM]
    x_opt_gamma = fminsearch(fun_gamma,x_init_gamma,fminsearch_options);
    beta_bar_opt_gamma = x_opt_gamma(1); S_bar_init_opt_gamma = x_opt_gamma(2); gamma_opt_gamma = x_opt_gamma(3);
    %fun_opt_gamma = fun_gamma(x_opt_gamma);  % value of the minimum


    % Plot true and simulated H, and simulated S_bar:
    fig_cnt = fig_cnt + 1;
    figure(fig_cnt)
    res_gamma = results(beta_bar_opt_gamma,gamma_opt_gamma,S_bar_init_opt_gamma,total_in(train_t_start),tspan_train,dates,data_totinout);
    disp(['MAPE_train (beta_bar, S_bar_init, and gamma optimized): ', num2str(res_gamma.MAPE_train)])
    disp(['MAPE_test (beta_bar, S_bar_init, and gamma optimized): ', num2str(res_gamma.MAPE_test)])
    %title('Optimized wrt beta\_bar, S\_bar\_init, and gamma')
    if save_figures 
        saveas(gcf,[fig_name_root,'_3Dopt'],'fig');
        saveas(gcf,[fig_name_root,'_3Dopt'],'epsc');        
    end


    % ***********************************************************************************
    % Optimization on beta_bar, S_bar_init, gamma, and H_init
    % *******

    fun_gammaHinit = @(x)phi_gammaHinit(x,data_totinout_train,tspan_train,c_HEL);  % function phi is given at the end of this script
    x_init_gammaHinit = [x_opt,gamma_hat_RM,data_totinout_train(train_t_start,1)];  % sugg: [x_opt,gamma_hat_RM,data_totinout_train(train_t_start,1)]
    [x_opt_gammaHinit,fval,exitflag,output] = fminsearch(fun_gammaHinit,x_init_gammaHinit,fminsearch_options);
    beta_bar_opt_gammaHinit = x_opt_gammaHinit(1); S_bar_init_opt_gammaHinit = x_opt_gammaHinit(2); gamma_opt_gammaHinit = x_opt_gammaHinit(3); H_init_opt_gammaHinit = x_opt_gammaHinit(4);
    %fun_opt_gammaHinit = fun_gammaHinit(x_opt_gammaHinit);  % value of the minimum


    % Plot true and simulated H, and simulated S_bar:
    fig_cnt = fig_cnt + 1;
    figure(fig_cnt)
    res_gammaHinit = results(beta_bar_opt_gammaHinit,gamma_opt_gammaHinit,S_bar_init_opt_gammaHinit,H_init_opt_gammaHinit,tspan_train,dates,data_totinout);
    disp(['MAPE_train (beta_bar, S_bar_init, gamma, and H_init optimized): ', num2str(res_gammaHinit.MAPE_train)])
    disp(['MAPE_test (beta_bar, S_bar_init, gamma, and H_init optimized): ', num2str(res_gammaHinit.MAPE_test)])
    disp(['Relative stability margin for beta at the end: ', num2str(res_gammaHinit.L(end)/res_gammaHinit.E(end))])
    %title('Optimized wrt beta\_bar, S\_bar\_init, gamma, and H\_init')
    if save_figures 
        saveas(gcf,[fig_name_root,'_4Dopt'],'fig');
        saveas(gcf,[fig_name_root,'_4Dopt'],'epsc');        
    end

    
    % ***********************************************************************************
    % Polynomial fitting
    % *******
    
    % For the sajke of comparison, let's check the fitting and prediction capabilities of the simplest model class: polynomial.
    
    deg = 8;  % degree of the polynomial
    ts = [1:length(total_in)]';
    poly = polyfit(ts(train_t_start:train_t_end),total_in_train(train_t_start:train_t_end),deg);
    H_poly = polyval(poly,ts);

    fig_cnt = fig_cnt + 1;
    figure(fig_cnt)
    plot(dates,total_in,'k-', dates(train_t_start:train_t_end),H_poly(train_t_start:train_t_end),'b--', dates(train_t_end+1:end),H_poly(train_t_end+1:end),'r-.');
    hold on
    leg = legend('total_in','H_train','H_pred');
    set(leg,'Interpreter','none')
    %title('Polynomial fit and extrapolation')
    if save_figures 
        saveas(gcf,[fig_name_root,'_polyfit'],'fig');
        saveas(gcf,[fig_name_root,'_polyfit'],'epsc');        
    end
    
    res_poly.H = H_poly;
    res_poly.RMSE_train = norm(H_poly(train_t_start:train_t_end) - total_in(train_t_start:train_t_end)) / sqrt(train_t_end-train_t_start+1);
    res_poly.RMSE_test = norm(H_poly(train_t_end+1:end) - total_in(train_t_end+1:end)) / sqrt(data_length-train_t_end);
    % Mean absolute percentage error:
    res_poly.MAPE_train = 100 * sum(abs(H_poly(train_t_start:train_t_end) - total_in(train_t_start:train_t_end))./total_in(train_t_start:train_t_end)) / (train_t_end-train_t_start+1);
    res_poly.MAPE_test = 100 * sum(abs(H_poly(train_t_end+1:end) - total_in(train_t_end+1:end))./total_in(train_t_end+1:end)) / (data_length-train_t_end); 
    
    
end 
% end loop on train_t_start
% }}}}}}}


% ***********************************************************************************
% Local functions
% *******

function [S_bar,H,E,L] = simu(beta_bar,gamma,S_bar_init,H_init,tspan)
    simu_t_start = tspan(1);
    simu_t_end = tspan(2);
    S_bar = NaN(simu_t_end,1);  % set storage
    H = S_bar;  % set storage
    E = S_bar;  % set storage
    L = S_bar;  % set storage
    S_bar(simu_t_start) = S_bar_init;
    H(simu_t_start) = H_init;
    for t = simu_t_start:simu_t_end-1
        S_bar(t+1) = S_bar(t) - beta_bar * S_bar(t) * H(t);
        H(t+1) = H(t) + beta_bar * S_bar(t) * H(t) - gamma * H(t);
        E(t+1) = beta_bar * S_bar(t) * H(t);
        L(t+1) = gamma * H(t);
    end
end

function cost = phi(x,data_totinout_train,tspan_train,gamma,c_HEL)   % x := [beta_bar,S_bar_init]
    % Extract variables from input:
    c_H = c_HEL(1); c_E = c_HEL(2); c_L = c_HEL(3);  % coefficients of the terms of the cost function.
    beta_bar = x(1); S_bar_init = x(2);
    train_t_start = tspan_train(1); train_t_end = tspan_train(2);
    % Simulate SH model:
    [S_bar,H,E,L] = simu(beta_bar,gamma,S_bar_init,data_totinout_train(train_t_start,1),[train_t_start,train_t_end]);
    % Compute the cost (discrepancy between observed and simulated):
    cost = c_H * (norm(H(train_t_start:train_t_end)-data_totinout_train(train_t_start:train_t_end,1)))^2 + c_E * (norm(E(train_t_start+1:train_t_end)-data_totinout_train(train_t_start+1:train_t_end,2)))^2 + c_L * (norm(L(train_t_start+1:train_t_end)-data_totinout_train(train_t_start+1:train_t_end,3)))^2;
end

function cost = phi_gamma(x,data_totinout_train,tspan_train,c_HEL)   % x := [beta_bar,S_bar_init,gamma]
    c_H = c_HEL(1); c_E = c_HEL(2); c_L = c_HEL(3);  % coefficients of the terms of the cost function.
    beta_bar = x(1); S_bar_init = x(2); gamma = x(3);
    train_t_start = tspan_train(1); train_t_end = tspan_train(2);
    [S_bar,H,E,L] = simu(beta_bar,gamma,S_bar_init,data_totinout_train(train_t_start,1),[train_t_start,train_t_end]);
    cost = c_H * (norm(H(train_t_start:train_t_end)-data_totinout_train(train_t_start:train_t_end,1)))^2 + c_E * (norm(E(train_t_start+1:train_t_end)-data_totinout_train(train_t_start+1:train_t_end,2)))^2 + c_L * (norm(L(train_t_start+1:train_t_end)-data_totinout_train(train_t_start+1:train_t_end,3)))^2;
end

function cost = phi_gammaHinit(x,data_totinout_train,tspan_train,c_HEL)   % x := [beta_bar,S_bar_init,gamma,H_init]
    c_H = c_HEL(1); c_E = c_HEL(2); c_L = c_HEL(3);  % coefficients of the terms of the cost function.
    beta_bar = x(1); S_bar_init = x(2); gamma = x(3);
    H_init = x(4);
    train_t_start = tspan_train(1); train_t_end = tspan_train(2);
    [S_bar,H,E,L] = simu(beta_bar,gamma,S_bar_init,H_init,[train_t_start,train_t_end]);
    cost = c_H * (norm(H(train_t_start:train_t_end)-data_totinout_train(train_t_start:train_t_end,1)))^2 + c_E * (norm(E(train_t_start+1:train_t_end)-data_totinout_train(train_t_start+1:train_t_end,2)))^2 + c_L * (norm(L(train_t_start+1:train_t_end)-data_totinout_train(train_t_start+1:train_t_end,3)))^2;
end

function res = results(beta_bar,gamma,S_bar_init,H_init,tspan_train,dates,data_totinout)
% tspan_train in the argument is needed to distinguish between train and test parts.
    total_in = data_totinout(:,1);
    data_length = length(total_in);
    train_t_start = tspan_train(1); train_t_end = tspan_train(2);
    [S_bar,H,E,L] = simu(beta_bar,gamma,S_bar_init,H_init,[tspan_train(1),size(data_totinout,1)]);
    res.S_bar = S_bar; res.H = H; res.E = E; res.L = L;
    
    %  plot(dates,total_in,'k-', dates(train_t_start:train_t_end),H(train_t_start:train_t_end),'b--', dates(train_t_end+1:end),H(train_t_end+1:end),'r-.');
    % hold on
    
    % show_S_bar = 1;
    % if show_S_bar
    %     plot(dates(train_t_start:train_t_end),S_bar(train_t_start:train_t_end),'m--', dates(train_t_end+1:end),S_bar(train_t_end+1:end),'m-.');
    %     %leg = legend('S_bar_train','S_bar_pred');
    %     leg = legend('total_in','H_train','H_pred','S_bar_train','S_bar_pred');
    %     set(leg,'Interpreter','none')
    % else
    %     leg = legend('total_in','H_train','H_pred');
    %     set(leg,'Interpreter','none')
    % end
    % %ylim([0,1.2*max(data_totinout(:,1))]);  % set vertical axis limits.

    
    subplot(1,2,1)
    plot(dates,total_in,'k-', dates(train_t_start:train_t_end),H(train_t_start:train_t_end),'b--', dates(train_t_end+1:end),H(train_t_end+1:end),'r-.');
    hold on
    leg = legend('total_in','H_train','H_pred');
    set(leg,'Interpreter','none')

    subplot(1,2,2)
    plot(dates(train_t_start:train_t_end),S_bar(train_t_start:train_t_end),'b--', dates(train_t_end+1:end),S_bar(train_t_end+1:end),'r-.');
    hold on
    %ylim_orig = ylim;
    %ylim([0,ylim_orig(2)]);
    leg = legend('S_bar_train','S_bar_pred');
    set(leg,'Interpreter','none')
    
    
    % Root mean square error:
    res.RMSE_train = norm(H(train_t_start:train_t_end) - total_in(train_t_start:train_t_end)) / sqrt(train_t_end-train_t_start+1);
    res.RMSE_test = norm(H(train_t_end+1:end) - total_in(train_t_end+1:end)) / sqrt(data_length-train_t_end);
    % Mean absolute percentage error:
    res.MAPE_train = 100 * sum(abs(H(train_t_start:train_t_end) - total_in(train_t_start:train_t_end))./total_in(train_t_start:train_t_end)) / (train_t_end-train_t_start+1);
    res.MAPE_test = 100 * sum(abs(H(train_t_end+1:end) - total_in(train_t_end+1:end))./total_in(train_t_end+1:end)) / (data_length-train_t_end); 
end
