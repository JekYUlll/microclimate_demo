function mask = pdppo_demo_step(obs)
%PDPO_DEMO_STEP Simulink Interpreted MATLAB Function entry point.
mask = pdppo_scheduler(obs);
end
