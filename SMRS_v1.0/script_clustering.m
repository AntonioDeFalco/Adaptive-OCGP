close all; 

clear all; 

%D = 30; d = 3; N = 100;
%X = (orth(randn(D,d)) * randn(d,N))';

X = [[ones(100,1) zeros(100,1)]; [zeros(100,1) ones(100,1)]; ones(1,2); randn(100,2)]; 

plot(X(:,1), X(:,2),'*');

[repInd,C] = smrs(X',10,0,true);

hold on; 
plot(X(repInd,1),X(repInd,2),'rx')

plot(X(101,1),X(101,2),'ko')

[center,U,obj_fcn] = fcm(X,2);

plot([center([1 2],1)],[center([1 2],2)],'*','color','m')
%
%
%
% EEG signals -> 
%   
%
%
% proteomic signals -> 
%
%
%
% ecg signals ->
%