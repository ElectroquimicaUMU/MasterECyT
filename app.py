# app.py
# Streamlit app: Cronoamperometría reversible en electrodo esférico
# Cambios solicitados:
# 1) Descargar solo I_total(t)
# 2) Regresión solo con I_total(t)
# 3) Selección de t_min y t_max con slider de rango

import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# ----------------------------
# Parámetros FIJOS (no editables por el usuario)
# ----------------------------
F = 96485.0  # C/mol
R = 8.314    # J/mol/K
T = 298.0    # K

D = 1.0e-9          # m^2/s
a = 25e-6           # m
E0 = 0.0            # V
c_total = 1.0       # mol/m^3

# Parámetros eléctricos (fijos)
Ru = 50.0           # ohm
Cdl = 20e-6         # F

# Discretización fija (dependiente de tp para control de coste)
TARGET_STEPS = 4000
DT_MIN = 1e-5
DT_MAX = 5e-3
DR = 2e-6

DOMAIN_FACTOR = 6.0  # r_max = a + factor*sqrt(D*tp)


# ----------------------------
# Utilidades
# ----------------------------
def _parse_float(text: str) -> float:
    return float(text.strip().replace(",", "."))


def eta(E: float, E0_: float) -> float:
    return (F / (R * T)) * (E - E0_)


def c_surf_nernst(c_tot: float, E: float, E0_: float) -> float:
    x = math.exp(eta(E, E0_))
    return c_tot * x / (1.0 + x)


def thomas_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = diag.size
    c_star = np.empty(n - 1, dtype=float)
    d_star = np.empty(n, dtype=float)

    c_star[0] = upper[0] / diag[0]
    d_star[0] = rhs[0] / diag[0]

    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c_star[i - 1]
        c_star[i] = upper[i] / denom
        d_star[i] = (rhs[i] - lower[i - 1] * d_star[i - 1]) / denom

    denom = diag[-1] - lower[-1] * c_star[-1]
    d_star[-1] = (rhs[-1] - lower[-1] * d_star[-2]) / denom

    x = np.empty(n, dtype=float)
    x[-1] = d_star[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]

    return x


def solve_spherical_reversible_chronoamperometry(E_app: float, tpulse: float):
    """Devuelve: times, jF (A/m^2), r, c_ox_final, r_max, dt"""
    if tpulse <= 0:
        raise ValueError("tpulse debe ser > 0")

    dt = tpulse / TARGET_STEPS
    dt = min(max(dt, DT_MIN), DT_MAX)
    nsteps = int(math.ceil(tpulse / dt))
    dt = tpulse / nsteps

    r_max = a + DOMAIN_FACTOR * math.sqrt(D * tpulse)
    if r_max < a + 5 * DR:
        r_max = a + 5 * DR

    n = int(math.ceil((r_max - a) / DR)) + 1
    r = a + np.arange(n) * DR
    r_face = 0.5 * (r[:-1] + r[1:])

    c0 = c_surf_nernst(c_total, E_app, E0)

    C = np.full(n, c_total, dtype=float)
    C[0] = c0

    lam = D * dt / (DR ** 2)

    Nunk = n - 1
    diag = np.zeros(Nunk, dtype=float)
    lower = np.zeros(Nunk - 1, dtype=float)
    upper = np.zeros(Nunk - 1, dtype=float)

    for k in range(Nunk - 1):
        i = k + 1
        A = lam * (r_face[i - 1] ** 2 / (r[i] ** 2))
        B = lam * (r_face[i] ** 2 / (r[i] ** 2))
        diag[k] = 1.0 + A + B
        if k > 0:
            lower[k - 1] = -A
        upper[k] = -B

    i_last = n - 1
    A_last = lam * (r_face[-1] ** 2 / (r[i_last] ** 2))
    diag[-1] = 1.0 + A_last
    lower[-1] = -A_last

    times = np.empty(nsteps, dtype=float)
    jF = np.empty(nsteps, dtype=float)

    A_i1 = lam * (r_face[0] ** 2 / (r[1] ** 2))

    for k in range(nsteps):
        rhs = np.empty(Nunk, dtype=float)
        rhs[:-1] = C[1:-1].copy()
        rhs[0] += A_i1 * c0
        rhs[-1] = C[-1]

        C_unk = thomas_tridiagonal(lower, diag, upper, rhs)
        C[0] = c0
        C[1:] = C_unk

        dc_dr_a = (-3.0 * C[0] + 4.0 * C[1] - C[2]) / (2.0 * DR)
        N_mol = -D * dc_dr_a

        times[k] = (k + 1) * dt
        jF[k] = F * N_mol

    return times, jF, r, C, r_max, dt


def capacitive_current(E_app: float, t: np.ndarray) -> np.ndarray:
    tau = Ru * Cdl
    return (E_app / Ru) * np.exp(-t / tau)


def regression_lnI_lnT(t: np.ndarray, I: np.ndarray):
    """Regresión y = m x + b con x=ln(t), y=ln(|I|)."""
    t = np.asarray(t, dtype=float)
    I = np.asarray(I, dtype=float)

    mask = np.isfinite(t) & np.isfinite(I) & (t > 0) & (np.abs(I) > 0)
    t2 = t[mask]
    I2 = I[mask]
    if t2.size < 2:
        return None

    x = np.log(t2)
    y = np.log(np.abs(I2))
    m, b = np.polyfit(x, y, 1)

    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"m": float(m), "b": float(b), "r2": r2, "x": x, "y": y, "yhat": yhat, "n": int(x.size)}


def export_txt_itotal(runs):
    lines = []
    lines.append("# Cronoamperometría esférica reversible: I_total(t) (A)")
    lines.append("# I_total = I_F + I_cap ;  I_cap = (E/Ru)*exp(-t/(Ru*Cdl))")
    lines.append(f"# Ru[ohm]={Ru}, Cdl[F]={Cdl}")
    lines.append("# Columnas: t[s]\tI_total[A]")
    for run in runs:
        lines.append("")
        lines.append(f"# --- RUN {run['id']} ---")
        lines.append(f"# E[V]={run['E']}, tpulse[s]={run['tpulse']}, r_max[m]={run['r_max']}, dt[s]={run['dt']}")
        for ti, it in zip(run["times"], run["I_total"]):
            lines.append(f"{ti:.12g}\t{it:.12g}")
    return "\n".join(lines) + "\n"


def export_txt_profile_final(runs, species):
    lines = []
    lines.append("# Perfiles (al final del pulso, t=tp)")
    if species == "Oxidada (c_ox)":
        lines.append("# Columnas: dist[um]\tc_ox[mol/m^3]")
    elif species == "Reducida (c_red)":
        lines.append("# Columnas: dist[um]\tc_red[mol/m^3]  (c_total constante)")
    else:
        lines.append("# Columnas: dist[um]\tc_ox[mol/m^3]\tc_red[mol/m^3]  (c_total constante)")

    for run in runs:
        lines.append("")
        lines.append(f"# --- RUN {run['id']} ---")
        lines.append(f"# E[V]={run['E']}, tpulse[s]={run['tpulse']}, r_max[m]={run['r_max']}, dt[s]={run['dt']}")
        dist_um = run["dist_um"]
        c_ox = run["c_ox_final"]
        c_red = c_total - c_ox

        if species == "Oxidada (c_ox)":
            for x, co in zip(dist_um, c_ox):
                lines.append(f"{x:.12g}\t{co:.12g}")
        elif species == "Reducida (c_red)":
            for x, cr in zip(dist_um, c_red):
                lines.append(f"{x:.12g}\t{cr:.12g}")
        else:
            for x, co, cr in zip(dist_um, c_ox, c_red):
                lines.append(f"{x:.12g}\t{co:.12g}\t{cr:.12g}")

    return "\n".join(lines) + "\n"


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Cronoamperometría esférica reversible", layout="wide")
st.title("Cronoamperometría (electrodo esférico, proceso reversible)")

if "runs" not in st.session_state:
    st.
