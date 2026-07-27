function mask = feasibility_projector(scores, policy, prevMask, holdRemaining, dutyCount, stepCount)
%FEASIBILITY_PROJECTOR Convert channel scores to an executable subset.

scores = double(scores(:));
cost = double(policy.cost(:));
budget = double(policy.budget);
n = numel(cost);

required = (prevMask(:) > 0.5) & (holdRemaining(:) > 0);
mask = double(required);

% If required channels exceed the budget, keep the highest-scoring required set.
while cost' * mask > budget && any(mask)
    idx = find(mask > 0.5);
    [~, localMin] = min(scores(idx));
    mask(idx(localMin)) = 0;
end

remaining = budget - cost' * mask;
dutyRatio = dutyCount(:) ./ max(stepCount, 1);

[~, order] = sort(scores, "descend");
for k = 1:numel(order)
    i = order(k);
    if mask(i) > 0.5
        continue;
    end
    if dutyRatio(i) > policy.dutyMax(i)
        continue;
    end
    if cost(i) <= remaining + 1e-9
        mask(i) = 1;
        remaining = remaining - cost(i);
    end
end

% Keep the demo alive even for pathological inputs.
if ~any(mask)
    feasible = find(cost <= budget);
    [~, j] = max(scores(feasible));
    mask(feasible(j)) = 1;
end

mask = double(mask(:));
end
