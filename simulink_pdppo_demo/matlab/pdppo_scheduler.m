function maskRow = pdppo_scheduler(obs)
%PDPO_SCHEDULER Frozen PD-PPO-style scheduler for Simulink demonstration.
% Input:  [state(8), AoI(8), blowing-snow event state(1)]
% Output: 1-by-8 executable channel mask.

persistent policy prevMask holdRemaining dutyCount stepCount

if isempty(policy)
    thisDir = fileparts(mfilename("fullpath"));
    rootDir = fileparts(thisDir);
    s = load(fullfile(rootDir, "frozen_policy", "pdppo_demo_policy.mat"), "policy");
    policy = s.policy;
    prevMask = zeros(policy.numChannels, 1);
    holdRemaining = zeros(policy.numChannels, 1);
    dutyCount = zeros(policy.numChannels, 1);
    stepCount = 0;
end

obs = double(obs(:));
if numel(obs) < 17
    obs = [obs; zeros(17 - numel(obs), 1)];
end

state = min(max(obs(1:8), 0), 1);
aoi = min(max(obs(9:16), 0), 1);
eventState = double(obs(17) > 0.5);

features = [state; aoi; prevMask; eventState];
scores = policy.W * features + policy.bias + policy.stabilityBonus * prevMask;

mask = feasibility_projector(scores, policy, prevMask, holdRemaining, dutyCount, stepCount);

stepCount = stepCount + 1;
dutyCount = dutyCount + mask;
newOn = (mask > 0.5) & (prevMask < 0.5);
holdRemaining(newOn) = policy.minActivationSteps;
holdRemaining(mask > 0.5) = max(holdRemaining(mask > 0.5) - 1, 0);
holdRemaining(mask < 0.5) = 0;
prevMask = mask;

maskRow = double(mask(:)');
end
