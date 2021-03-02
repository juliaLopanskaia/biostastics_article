%% simulation to understand biostatistics
clear
N_per_day=20; 
N_days=5;

true_exp_mean=0.8;
true_control_mean=1;

inter_day_SD=0.1; %um/s
sigma=0.5; %intra_day_SD;

mean_exp=true_exp_mean+inter_day_SD*randn(1,N_days);
mean_control=true_control_mean+inter_day_SD*randn(1,N_days);

day_exp=mean_exp+sigma*randn(N_per_day,N_days);
day_control=mean_control+sigma*randn(N_per_day,N_days);
mexp=mean(day_exp,1);
mctr=mean(day_control,1);

figure
hold on
plot(ones(N_per_day,1)+0.05/sqrt(N_days)*randn(N_per_day,1), day_exp,'.','MarkerSize',11)

colord=get(gca,'colororder');
nc=size(colord,1);

if N_days<=nc
    set(gca,'colororder',colord(1:N_days,:))
    col=colord(1:N_days,:);
else
    integ=fix(N_days/nc);
    remain=rem(N_days,nc);
    col=repmat(colord,[integ+1 1]);
    col=col(1:integ*nc+remain,1:3);
end

scatter(ones(1,N_days), mexp, 600, col,'+','LineWidth',2)

plot(2*ones(N_per_day,1)+0.05/sqrt(N_days)*randn(N_per_day,1), day_control,'.', 'MarkerSize',11)
scatter(2*ones(1,N_days), mctr, 600, col,'+','LineWidth',2)
xlim([0 3]);

exp_pooled=day_exp(:);
ctr_pooled=day_control(:);

exp_dayly=mexp(:);
ctr_dayly=mctr(:);

ICC=inter_day_SD^2/(inter_day_SD^2+sigma^2)

[h,p] = ttest2(exp_pooled,ctr_pooled);
if h==0
    fprintf('Means are the same (pooled data), p-value is %d \n',p)
else
    fprintf('Means are different (pooled data), p-value is %d \n',p)
end

[h,p] = ttest2(exp_dayly,ctr_dayly);
if h==0
    fprintf('Means are the same (per-day means), p-value is %d \n',p)
else
    fprintf('Means are different (per-day means), p-value is %d \n',p)
end

fprintf('Mean value in experiment %.2f \n',mean(exp_dayly))
fprintf('Mean value in control %.2f \n',mean(ctr_dayly))