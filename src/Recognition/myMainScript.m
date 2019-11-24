%% MyMainScript

tic;
%% Your code here

clc
clear all
close all 
% k = 0 for the plot generation and k = n; n != 0 for accuracy at k = n
mySVD('../../../att_faces/s', 112, 92, 32, 6, 4, 0); 
myEIG('../../../att_faces/s', 112, 92, 32, 6, 4, 0);
% e = 0 for including the top 3 eigen vectors
% e = 1 for excluding the top 3 eigen vectors
myYaleSVD('../../../CroppedYale/yaleB', 192, 168, 39, 40, 24, 0, 0);
myYaleSVD('../../../CroppedYale/yaleB', 192, 168, 39, 40, 24, 0, 1);

toc;
