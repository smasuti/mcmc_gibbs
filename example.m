% Example of using the gibbs sampling.
% Author : Sagar Masuti
% Date   : 04-Aug-2018
% 
% 03-Apr-2019: Added burn-in and step.
% -------------------------------------------------------------------------
clear all;
% close all;

% Number of dimensions.
nd=2;

% number of iteration/number of samples to be generated.
iter=5000;

% Number of samples in each dimension to compute the conditional
% distribution
ncond=100;

% Lower and Upper bounds within which to search and sample
bounds=[0.30 0.7;
           0 0.477];
       
% Burn in Samples
burnin=1000;

% step 
step=1;

% True mean of the posterior for synthetic data generation
m1=4; m2=2;

x=0:0.1:5;
d=m1*x+m2+0.01;

% Initial values.
m0=[log10(3) log10(1)];

ms=zeros(ncond,nd);
for i=1:nd
    ms(:,i)=ones(ncond,1)*m0(i);
end

% Calculating the misfit for the current sample.
model=@(m) m(1)*x+m(2);
misfit=@(d,dpre) (sum((d-dpre).^2));
residual=@(m) misfit(d,model(10.^m));

% Get the samples.
[allsamples,~]=gibbs_sample(iter,bounds,m0,residual,ncond);

samples=allsamples(burnin:step:end,:);
%% Plotting the results.
figure(1);clf;
subplot(2,2,3);
scatter(10.^samples(:,1),10.^samples(:,2),30,'filled');
hold on
y=(0:0.1:4);
x=(0:0.1:7);
plot(m1*ones(length(y),1),y,'k--');
plot(x,m2*ones(length(x),1),'k--');
% xlim([1 3]);
% ylim([2 6]);
% xlim([3.5 4.5]);
% ylim([1 3]);
xlim([3.7 4.4]);ylim([1 2.8]);
xlabel('$m_1$','Interpreter','latex');
ylabel('$m_2$','Interpreter','latex');

% Plot the 0-90 percentile error ellipse.
plot_error_ellipse(10.^samples);

subplot(2,2,1);
histogram(10.^samples(:,1),100);
xlim([3.5 4.5]);
xlabel('$m_1$','Interpreter','latex');

subplot(2,2,4);
histogram(10.^samples(:,2),100);
xlim([1 3]);
xlabel('$m_2$','Interpreter','latex');

