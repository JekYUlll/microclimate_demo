function data = generate_demo_inputs(nSteps)
%GENERATE_DEMO_INPUTS Create deterministic demo observations for Simulink.

if nargin < 1
    nSteps = 240;
end

rootDir = fileparts(fileparts(mfilename("fullpath")));
outDir = fullfile(rootDir, "data");
if ~exist(outDir, "dir")
    mkdir(outDir);
end

rng(41);
t = (0:nSteps-1)';
event = double((t > 45 & t < 82) | (t > 138 & t < 176) | sin(2*pi*t/95) > 0.82);

wind = clip01(0.35 + 0.20*sin(2*pi*t/80) + 0.32*event + 0.03*randn(nSteps,1));
airTemp = clip01(0.50 + 0.10*sin(2*pi*t/120 + 1.2) - 0.05*event);
humidity = clip01(0.55 + 0.18*cos(2*pi*t/90) + 0.06*event);
pressure = clip01(0.50 + 0.04*sin(2*pi*t/160));
radiation = clip01(0.50 + 0.45*sin(2*pi*t/120 - 0.8));
snowTemp = clip01(0.45 + 0.08*sin(2*pi*t/110) - 0.08*event);
particle = clip01(0.10 + 0.18*wind + 0.60*event + 0.03*randn(nSteps,1));
flux = clip01(0.06 + 0.22*wind.^2 + 0.55*event + 0.03*randn(nSteps,1));

state = [wind, airTemp, humidity, pressure, radiation, snowTemp, particle, flux];

% AoI input is only a demo signal. The scheduler also keeps its own previous
% mask and dwell counters internally.
aoi = zeros(nSteps, 8);
for i = 1:8
    aoi(:, i) = mod(t + 3*i, 24) / 24;
end

obs = [state, aoi, event];
pdppo_demo_obs = timeseries(obs, t);

data = struct("t", t, "event", event, "obs", obs, "pdppo_demo_obs", pdppo_demo_obs);
save(fullfile(outDir, "demo_inputs.mat"), "pdppo_demo_obs", "obs", "t", "event");
end

function y = clip01(x)
y = min(max(x, 0), 1);
end
