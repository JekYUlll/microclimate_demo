function partial = pdppo_apply_mask(obsAndMask)
%PDPO_APPLY_MASK Convert full demo observations into scheduler-induced missing data.
% Input from Simulink mux: [obs(17), mask(8)].

x = double(obsAndMask(:));
if numel(x) < 25
    x = [x; zeros(25 - numel(x), 1)];
end

fullState = x(1:8);
mask = x(18:25) > 0.5;

partialState = fullState;
partialState(~mask) = NaN;
partial = partialState(:)';
end
