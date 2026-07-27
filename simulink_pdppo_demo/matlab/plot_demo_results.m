function plot_demo_results(simOut, data)
%PLOT_DEMO_RESULTS Plot event signal and scheduled channel masks.

rootDir = fileparts(fileparts(mfilename("fullpath")));
outDir = fullfile(rootDir, "outputs");
if ~exist(outDir, "dir")
    mkdir(outDir);
end

y = simOut.get("pdppo_mask_out");
t = y.time(:);
values = y.signals.values;

if ndims(values) == 3
    mask = squeeze(values);
    if size(mask, 1) == 1
        mask = squeeze(values(1, :, :))';
    end
else
    mask = values;
end

if size(mask, 1) ~= numel(t)
    mask = mask';
end

names = ["weather", "pyranometer", "surface IR", "wind", ...
    "thermo-hygro", "particle", "laser", "FC4"];

fig = figure("Name", "Frozen PD-PPO Simulink Demo", "Color", "w");
tiledlayout(fig, 2, 1, "TileSpacing", "compact");

nexttile;
stairs(data.t, data.event, "LineWidth", 1.4);
ylim([-0.1, 1.1]);
xlabel("time step");
ylabel("event");
title("Blowing-snow event state");
grid on;

nexttile;
imagesc(t, 1:8, mask');
colormap(gca, [0.94 0.94 0.94; 0.20 0.47 0.73]);
caxis([0 1]);
yticks(1:8);
yticklabels(names);
xlabel("time step");
ylabel("channel");
title("Scheduled channel mask");

exportgraphics(fig, fullfile(outDir, "demo_schedule.png"), "Resolution", 180);
savefig(fig, fullfile(outDir, "demo_schedule.fig"));
end
