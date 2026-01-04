from __future__ import annotations

"""
Computation helpers for wind-blown snow load parameters.

Implements formulas described in the patent text:
- Mass, momentum, energy flux from particle spectra
- Impact pressure estimation with restitution
- Snow density/hardness/viscosity surrogate models
"""

from dataclasses import dataclass
from math import exp, pi
from typing import Iterable, Sequence


@dataclass
class ParticleBin:
    """Represents a binned particle population."""

    diameter_m: float          # equivalent diameter (meters)
    velocity_m_s: float        # characteristic velocity (m/s)
    number_density_m3: float   # particles per cubic meter
    restitution: float = 0.0   # coefficient of restitution e_i (0-1)


def _particle_volume(diameter_m: float) -> float:
    """Volume of a sphere with given diameter."""
    radius = diameter_m / 2.0
    return 4.0 / 3.0 * pi * radius**3


def compute_fluxes(
    bins: Iterable[ParticleBin],
    rho_ice: float = 917.0,
) -> dict[str, float]:
    """
    Compute per-area mass, momentum, and energy fluxes.

    Formulas (per bin i):
      mass flux:      m_dot = rho_ice * V_i * n_i * v_i
      momentum flux:  Phi_p = rho_ice * V_i * n_i * v_i^2
      energy flux:    Phi_E = 0.5 * rho_ice * V_i * n_i * v_i^3
    where V_i = particle volume, n_i = number density, v_i = velocity.
    """
    mass_flux = 0.0
    momentum_flux = 0.0
    energy_flux = 0.0
    for b in bins:
        vol = _particle_volume(b.diameter_m)
        mass_flux += rho_ice * vol * b.number_density_m3 * b.velocity_m_s
        momentum_flux += rho_ice * vol * b.number_density_m3 * (b.velocity_m_s**2)
        energy_flux += 0.5 * rho_ice * vol * b.number_density_m3 * (b.velocity_m_s**3)
    return {
        "mass_flux_kg_m2_s": mass_flux,
        "momentum_flux_kg_m_s2": momentum_flux,
        "energy_flux_W_m2": energy_flux,
    }


def compute_impact_pressure(
    bins: Iterable[ParticleBin],
    rho_ice: float = 917.0,
) -> float:
    """
    Estimate impact pressure per unit area:
      p_impact = sum rho_ice * V_i * n_i * v_i * (1 + e_i)
    """
    pressure = 0.0
    for b in bins:
        vol = _particle_volume(b.diameter_m)
        pressure += rho_ice * vol * b.number_density_m3 * b.velocity_m_s * (1.0 + b.restitution)
    return pressure


def estimate_density(
    energy_flux: float,
    rho0: float = 100.0,
    alpha: float = 1e-3,
) -> float:
    """Snow density surrogate: rho = rho0 + alpha * E_avg."""
    return rho0 + alpha * energy_flux


def estimate_hardness(
    energy_flux: float,
    density: float,
    H0: float = 0.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Hardness surrogate: H = H0 + beta * Phi_E + gamma * rho."""
    return H0 + beta * energy_flux + gamma * density


def estimate_viscosity(
    density: float,
    snow_temp_c: float,
    eta0: float = 1.0,
    A: float = 1.0,
    T0: float = -5.0,
    ) -> float:
    """
    Viscosity surrogate: eta = eta0 * exp(A * rho / (T_s - T0))
    Note: ensure (T_s - T0) != 0 to avoid singularities.
    """
    denom = max(snow_temp_c - T0, 1e-3)
    arg = min(A * density / denom, 50.0)  # clamp to avoid overflow
    return eta0 * exp(arg)


def summarize_loads(
    bins: Sequence[ParticleBin],
    rho_ice: float = 917.0,
    rho0: float = 100.0,
    alpha: float = 1e-3,
    H0: float = 0.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    eta0: float = 1.0,
    A: float = 1.0,
    snow_temp_c: float = -10.0,
) -> dict[str, float]:
    """
    Convenience wrapper to compute fluxes, impact pressure, and derived snow properties.
    """
    fluxes = compute_fluxes(bins, rho_ice=rho_ice)
    impact_p = compute_impact_pressure(bins, rho_ice=rho_ice)
    density = estimate_density(fluxes["energy_flux_W_m2"], rho0=rho0, alpha=alpha)
    hardness = estimate_hardness(fluxes["energy_flux_W_m2"], density, H0=H0, beta=beta, gamma=gamma)
    viscosity = estimate_viscosity(density, snow_temp_c, eta0=eta0, A=A)
    return {
        **fluxes,
        "impact_pressure_pa": impact_p,
        "snow_density_kg_m3": density,
        "snow_hardness": hardness,
        "snow_viscosity": viscosity,
    }
