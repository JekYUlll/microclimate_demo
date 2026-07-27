#!/usr/bin/env python3
"""Export fixed demo models used by the Simulink/Python demonstration.

The exported files are intentionally small and deterministic. They are not
trained during the course demo; they provide fixed inference-time parameters
for the scheduler and the downstream reconstruction model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "frozen_policy"
MODEL_DIR = ROOT / "frozen_model"

CHANNELS = [
    "weather",
    "pyranometer",
    "surface_ir",
    "highres_wind",
    "thermo_hygro",
    "particle_counter",
    "laser",
    "fc4_flux",
]


def build_scheduler_policy() -> dict:
    """Return the fixed scheduler parameters used by Simulink."""

    weights = np.zeros((8, 25), dtype=np.float64)

    weights[0, [0, 1, 2, 3]] = [0.55, 0.25, 0.20, 0.20]
    weights[1, 4] = 0.85
    weights[2, 5] = 0.75
    weights[3, 0] = 0.80
    weights[4, [1, 2]] = [0.50, 0.45]
    weights[5, 6] = 0.80
    weights[6, [6, 7]] = [0.55, 0.45]
    weights[7, 7] = 0.95

    for i in range(8):
        weights[i, 8 + i] = 0.35
        weights[i, 16 + i] = 0.20

    weights[:, 24] = [0.10, -0.10, 0.05, 0.45, 0.05, 0.85, 1.05, 1.20]

    return {
        "name": "frozen_pdppo_demo_policy",
        "created": "2026-06-13",
        "numChannels": 8,
        "channelNames": CHANNELS,
        "cost": np.array([0.42, 0.36, 0.38, 0.58, 0.52, 0.68, 0.82, 0.86]),
        "budget": 1.70,
        "minActivationSteps": 6,
        "dutyMax": np.full(8, 0.72),
        "stabilityBonus": 0.20,
        "W": weights,
        "bias": np.array([0.20, -0.02, 0.02, 0.04, 0.03, -0.05, -0.08, -0.12]),
    }


def build_reconstructor() -> dict[str, np.ndarray]:
    """Return fixed parameters for the downstream reconstruction model."""

    coupling = np.zeros((8, 8), dtype=np.float64)

    # Weather core, thermo-hygro, and wind channels carry broad context.
    coupling[0, [1, 2, 3, 4]] = [0.14, 0.10, 0.08, 0.12]
    coupling[1, [0, 2, 4]] = [0.10, 0.08, 0.16]
    coupling[2, [0, 1, 4]] = [0.08, 0.08, 0.12]
    coupling[3, [0, 5, 6, 7]] = [0.10, 0.10, 0.08, 0.08]
    coupling[4, [0, 1, 2]] = [0.12, 0.14, 0.12]

    # Snow-particle, laser, and flux channels are coupled in event periods.
    coupling[5, [3, 6, 7]] = [0.10, 0.18, 0.20]
    coupling[6, [3, 5, 7]] = [0.08, 0.22, 0.22]
    coupling[7, [3, 5, 6]] = [0.10, 0.24, 0.24]

    return {
        "channels": np.array(CHANNELS, dtype=object),
        "baseline": np.array([0.42, 0.50, 0.45, 0.36, 0.50, 0.18, 0.16, 0.14]),
        "decay": np.array([0.82, 0.90, 0.88, 0.84, 0.88, 0.78, 0.74, 0.74]),
        "coupling": coupling,
        "event_gain": np.array([0.03, -0.02, -0.03, 0.12, 0.02, 0.34, 0.42, 0.46]),
        "observed_blend": np.array([0.92, 0.96, 0.95, 0.92, 0.95, 0.88, 0.86, 0.86]),
    }


def save_policy(policy: dict) -> None:
    POLICY_DIR.mkdir(parents=True, exist_ok=True)

    matlab_policy = {
        "name": policy["name"],
        "created": policy["created"],
        "numChannels": np.array([[policy["numChannels"]]], dtype=np.float64),
        "channelNames": np.array(policy["channelNames"], dtype=object).reshape(-1, 1),
        "cost": policy["cost"].reshape(-1, 1),
        "budget": np.array([[policy["budget"]]], dtype=np.float64),
        "minActivationSteps": np.array([[policy["minActivationSteps"]]], dtype=np.float64),
        "dutyMax": policy["dutyMax"].reshape(-1, 1),
        "stabilityBonus": np.array([[policy["stabilityBonus"]]], dtype=np.float64),
        "W": policy["W"],
        "bias": policy["bias"].reshape(-1, 1),
    }
    savemat(POLICY_DIR / "pdppo_demo_policy.mat", {"policy": matlab_policy})

    serializable = {}
    for key, value in policy.items():
        serializable[key] = value.tolist() if isinstance(value, np.ndarray) else value
    (POLICY_DIR / "pdppo_demo_policy.json").write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )


def save_reconstructor(model: dict[str, np.ndarray]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_DIR / "channel_reconstructor.npz", **model)


def main() -> None:
    save_policy(build_scheduler_policy())
    save_reconstructor(build_reconstructor())
    print(f"wrote {POLICY_DIR / 'pdppo_demo_policy.mat'}")
    print(f"wrote {POLICY_DIR / 'pdppo_demo_policy.json'}")
    print(f"wrote {MODEL_DIR / 'channel_reconstructor.npz'}")


if __name__ == "__main__":
    main()
