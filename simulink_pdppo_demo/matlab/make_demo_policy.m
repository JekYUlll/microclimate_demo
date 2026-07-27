function policy = make_demo_policy()
%MAKE_DEMO_POLICY Save a frozen scheduler policy for the Simulink demo.
% The policy is deterministic and lightweight. It preserves the PD-PPO runtime
% interface: score channels first, then project scores to an executable subset.

rootDir = fileparts(fileparts(mfilename("fullpath")));
outDir = fullfile(rootDir, "frozen_policy");
if ~exist(outDir, "dir")
    mkdir(outDir);
end

policy = struct();
policy.name = "frozen_pdppo_demo_policy";
policy.created = "2026-06-13";
policy.numChannels = 8;
policy.channelNames = ["weather"; "pyranometer"; "surface_ir"; ...
    "highres_wind"; "thermo_hygro"; "particle_counter"; "laser"; "fc4_flux"];
policy.cost = [0.42; 0.36; 0.38; 0.58; 0.52; 0.68; 0.82; 0.86];
policy.budget = 1.70;
policy.minActivationSteps = 6;
policy.dutyMax = 0.72 * ones(8, 1);
policy.stabilityBonus = 0.20;

% Feature vector = [state(8); aoi(8); previous_mask(8); event_state(1)].
W = zeros(8, 25);

% State features: wind, temp, humidity, pressure, radiation, snow temp,
% particle proxy, flux proxy.
W(1, [1 2 3 4]) = [0.55 0.25 0.20 0.20];
W(2, 5) = 0.85;
W(3, 6) = 0.75;
W(4, 1) = 0.80;
W(5, [2 3]) = [0.50 0.45];
W(6, 7) = 0.80;
W(7, [7 8]) = [0.55 0.45];
W(8, 8) = 0.95;

% AoI features encourage revisiting stale channels.
for i = 1:8
    W(i, 8 + i) = 0.35;
end

% Previous-mask features reduce unnecessary switching.
for i = 1:8
    W(i, 16 + i) = 0.20;
end

% Event feature boosts wind, particle, laser, and flux channels.
W(:, 25) = [0.10; -0.10; 0.05; 0.45; 0.05; 0.85; 1.05; 1.20];

policy.W = W;
policy.bias = [0.20; -0.02; 0.02; 0.04; 0.03; -0.05; -0.08; -0.12];

save(fullfile(outDir, "pdppo_demo_policy.mat"), "policy");
end
