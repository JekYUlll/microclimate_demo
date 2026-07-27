function export_demo_outputs(simOut, data)
%EXPORT_DEMO_OUTPUTS Write full data, mask, and partial observations as CSV.

rootDir = fileparts(fileparts(mfilename("fullpath")));
outDir = fullfile(rootDir, "outputs");
if ~exist(outDir, "dir")
    mkdir(outDir);
end

channelNames = ["weather", "pyranometer", "surface_ir", "highres_wind", ...
    "thermo_hygro", "particle_counter", "laser", "fc4_flux"];

[tMask, mask] = extract_sim_signal(simOut.get("pdppo_mask_out"));
[tPartial, partial] = extract_sim_signal(simOut.get("pdppo_partial_out"));

fullState = data.obs(:, 1:8);
tFull = data.t(:);

fullTable = array2table([tFull, fullState], "VariableNames", ["time", channelNames]);
maskTable = array2table([tMask, mask], "VariableNames", ["time", channelNames]);
partialTable = array2table([tPartial, partial], "VariableNames", ["time", channelNames]);

writetable(fullTable, fullfile(outDir, "full_reference.csv"));
writetable(maskTable, fullfile(outDir, "channel_mask.csv"));
writetable(partialTable, fullfile(outDir, "simulink_partial_observations.csv"));
end
