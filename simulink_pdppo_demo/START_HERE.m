%START_HERE One-click entry point for the Simulink PD-PPO demo.
%
% Open MATLAB, set the current folder to this project directory, and run:
%
%   START_HERE
%
% The script adds the local MATLAB folder to the path and launches the demo.

demoRoot = fileparts(mfilename("fullpath"));
addpath(fullfile(demoRoot, "matlab"));
run_demo();
