function run_demo()
%RUN_DEMO Create and run the frozen PD-PPO Simulink demonstration.

rootDir = fileparts(fileparts(mfilename("fullpath")));
matlabDir = fullfile(rootDir, "matlab");
addpath(matlabDir);

if ~exist(fullfile(rootDir, "outputs"), "dir")
    mkdir(fullfile(rootDir, "outputs"));
end

policyFile = fullfile(rootDir, "frozen_policy", "pdppo_demo_policy.mat");
if ~exist(policyFile, "file")
    make_demo_policy();
end
data = generate_demo_inputs(240);
assignin("base", "pdppo_demo_obs", data.pdppo_demo_obs);

pdppo_reset();
mdl = create_pdppo_demo_model();

simOut = sim(mdl, "StopTime", num2str(data.t(end)));
export_demo_outputs(simOut, data);
plot_demo_results(simOut, data);

fprintf("Demo complete.\n");
fprintf("Model:   %s\n", fullfile(rootDir, mdl + ".slx"));
fprintf("Figure:  %s\n", fullfile(rootDir, "outputs", "demo_schedule.png"));
end
