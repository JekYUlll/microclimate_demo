function [t, values2d] = extract_sim_signal(signalStruct)
%EXTRACT_SIM_SIGNAL Convert a To Workspace structure-with-time signal to 2-D.

t = signalStruct.time(:);
values = signalStruct.signals.values;

if ndims(values) == 3
    values2d = squeeze(values);
    if size(values2d, 1) ~= numel(t)
        values2d = values2d';
    end
else
    values2d = values;
end

if size(values2d, 1) ~= numel(t)
    values2d = values2d';
end
end
