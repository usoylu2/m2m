%%
clear; close all; clc;

[rf,header] = RPread(['./FR4.rf']);
header.h
header.txf
header.sf
rf = rf(:,:,-1);

fs=header.sf;    % Sampling Frequency
c=1540;   % Sound Velocity [m/s]
x=linspace(0,0.038,size(rf,2));   % 2.6 rad
dx=(x(end)-x(1))/(length(x)-1);   % 
z=(1/fs)*c*(1:header.h)/2;   % Axial Distance 30 mm
dz=(z(end)-z(1))/(length(z)-1);   % 1.925*10^-5   0.01925 um  1mm aprox.
fc=header.txf;   % Central Frequency
lambda=c/(fc);

figure; plot(z*1e3,rf(:,1)); xlabel('mm')
figure; plot(z*1e3,rf(:,50)); xlabel('mm')
figure; plot(z*1e3,rf(:,100)); xlabel('mm')
% keyboard;