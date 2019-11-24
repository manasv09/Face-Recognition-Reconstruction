%% MyMainScript

tic;
%% Your code here
clc
clear all
close all

myOutlier('../../../att_faces/s', 112, 92, 32, 6, 4, 0);
% k = 0 for the plot generation and k = n; n != 0 for accuracy at k = n

toc;
