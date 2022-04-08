% Script to perform Gibbs sampling. 
% Author : Sagar Masuti
% Date   : 04-Aug-2018
% -------------------------------------------------------------------------
function [samples,misfit]=gibbs_sample(sample_size,bounds,m0,residual,varargin)
% INPUT:
%   sample_size        = Number of samples to be generated.
%   bounds             = Bounds on the input parameter.
%   m0                 = Inital sample.
%   varargin
%   1) ncond           = Number of samples in to compute conditional
%                        distribution 
%
%   Output is samples
% -------------------------------------------------------------------------

% Number of dimensions.
nd=size(bounds,1);

% number of iteration/number of samples to be generated.
iter=sample_size;

% Number of samples in each dimension to compute the conditional
% distribution
if(nargin>4)
    ncond=varargin{1};
else
    ncond=100;
end

samples=zeros(iter,nd);
rand('seed' ,12345);

ms=ones(ncond,1)*m0;
misfit=zeros(iter,1);
randomscan=0;

accept=0;
for i=1:iter
    for j=1:nd
        mcond=unifrnd(bounds(j,1),bounds(j,2),ncond,1);
        ms(:,j)=mcond;
        logliklihood=zeros(ncond,1);
        for k=1:ncond
            logliklihood(k)=residual(ms(k,:));
        end


        if (randomscan==1)
            [~,index]=min(logliklihood);
            lmax=exp(-logliklihood(index)/2); 
            while (accept==0)
                ran=unifrnd(min(mcond),max(mcond));
                ms(1,j)=ran;
                lmdash=exp(-residual(ms(1,:))/2);
                s=rand;
                if (s<(lmdash/lmax))
                    accept=1;
                    samples(i,j)=ran;
                    ms(1:end,j)=ones(ncond,1)*samples(i,j);
                end
            end
            accept=0;   
        else
% accepting bestfit
%         [~,index]=min(logliklihood);
%         samples(i,j)=mcond(index);
%         ms(1:end,j)=ones(ncond,1)*mcond(index);

            fx=exp(-logliklihood)/sum(exp(-logliklihood));
            gmu=sum(mcond.*fx);
            gvar=sum((mcond.^2).*fx)-gmu^2;
            new_sample=normrnd(gmu,sqrt(gvar));
            samples(i,j)=new_sample;
            ms(1:end,j)=ones(ncond,1)*new_sample;
        end
    end
    misfit(i)=residual(samples(i,:));
    fprintf('Gibbs sampler iteration, misfit: %d %.2e\n',i,misfit(i));
end
end


