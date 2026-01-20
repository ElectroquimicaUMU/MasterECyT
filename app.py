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

# Sistema físico
D = 1.0e-9          # m^2/s
a = 5000e-6         # m (radio electrodo)
E0 = -0.5           # V (E0')

# Parámetros eléctricos (fijos)
Ru = 500.0           # ohm
Cdl = 100e-6         # F

# Discretización (fija, pero dt se adapta a tp)
TARGET_STEPS = 4000
DT_MIN = 1e-5
DT_MAX = 5e-3
DR = 2e-6           # m

# Dominio externo
DOMAIN_FACTOR = 1.  # r_max = a + factor*sqrt(D*tp)


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
    """
    Resuelve difusión esférica reversible (c_ox) con:
      - c_ox(a,t) fijada por Nernst (E_app)
      - flujo nulo en r_max

    Devuelve:
      times, jF (A/m^2), r, c_ox_final, r_max, dt
    """
    if tpulse <= 0:
        raise ValueError("tpulse debe ser > 0")

    # dt adaptativo
    dt = tpulse / TARGET_STEPS
    dt = min(max(dt, DT_MIN), DT_MAX)
    nsteps = int(math.ceil(tpulse / dt))
    dt = tpulse / nsteps  # termina exactamente en tpulse

    # dominio
    r_max = a + DOMAIN_FACTOR * math.sqrt(D * tpulse)
    if r_max < a + 5 * DR:
        r_max = a + 5 * DR

    # malla radial
    n = int(math.ceil((r_max - a) / DR)) + 1
    r = a + np.arange(n) * DR
    r_face = 0.5 * (r[:-1] + r[1:])  # caras i+1/2

    # contorno en el electrodo
    c0 = c_surf_nernst(c_total, E_app, E0)

    # inicial
    C = np.full(n, c_total, dtype=float)
    C[0] = c0

    lam = D * dt / (DR ** 2)

    # incógnitas: C[1]..C[n-1]
    Nunk = n - 1
    diag = np.zeros(Nunk, dtype=float)
    lower = np.zeros(Nunk - 1, dtype=float)
    upper = np.zeros(Nunk - 1, dtype=float)

    # filas i=1..n-2
    for k in range(Nunk - 1):
        i = k + 1
        A = lam * (r_face[i - 1] ** 2 / (r[i] ** 2))
        B = lam * (r_face[i] ** 2 / (r[i] ** 2))
        diag[k] = 1.0 + A + B
        if k > 0:
            lower[k - 1] = -A
        upper[k] = -B

    # última fila i=n-1 con flujo nulo en la cara externa
    i_last = n - 1
    A_last = lam * (r_face[-1] ** 2 / (r[i_last] ** 2))
    diag[-1] = 1.0 + A_last
    lower[-1] = -A_last

    times = np.empty(nsteps, dtype=float)
    jF = np.empty(nsteps, dtype=float)

    # término Dirichlet para i=1
    A_i1 = lam * (r_face[0] ** 2 / (r[1] ** 2))

    for k in range(nsteps):
        rhs = np.empty(Nunk, dtype=float)
        rhs[:-1] = C[1:-1].copy()
        rhs[0] += A_i1 * c0
        rhs[-1] = C[-1]

        C_unk = thomas_tridiagonal(lower, diag, upper, rhs)
        C[0] = c0
        C[1:] = C_unk

        # flujo en r=a
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


# --- NUEVO: regresión |I| vs t^{-1/2} ---
def regression_I_vs_tinvhalf(t: np.ndarray, I: np.ndarray):
    """Regresión y = m x + b con x=t^{-1/2}, y=|I|."""
    t = np.asarray(t, dtype=float)
    I = np.asarray(I, dtype=float)

    mask = np.isfinite(t) & np.isfinite(I) & (t > 0)
    t2 = t[mask]
    I2 = np.abs(I[mask])
    if t2.size < 2:
        return None

    x = 1.0 / np.sqrt(t2)
    y = I2
    m, b = np.polyfit(x, y, 1)

    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"m": float(m), "b": float(b), "r2": r2, "x": x, "y": y, "yhat": yhat, "n": int(x.size)}


def export_txt_itotal(runs):
    lines = []
    lines.append("# I (A)")
    lines.append("# I = I_F + I_cap ; I_cap = (E/Ru)*exp(-t/(Ru*Cdl))")
    lines.append(f"# Ru[ohm]={Ru}, Cdl[F]={Cdl}")
    lines.append("# Columnas: t[s]\tI[A]")
    for run in runs:
        lines.append("")
        lines.append(f"# --- RUN {run['id']} ---")
        lines.append(f"# E[V]={run['E']}, tpulse[s]={run['tpulse']}, dt[s]={run['dt']}, r_max[m]={run['r_max']}")
        for ti, it in zip(run["times"], run["I_total"]):
            lines.append(f"{ti:.12g}\t{it:.12g}")
    return "\n".join(lines) + "\n"


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Cronoamperometría esférica reversible", layout="wide")
st.title("Cronoamperometría")

if "runs" not in st.session_state:
    st.session_state.runs = []
if "run_id" not in st.session_state:
    st.session_state.run_id = 1

st.sidebar.header("Entradas (ajustables)")
E_text = st.sidebar.text_input("Potencial aplicado E [V]", value="0.10")
tp_text = st.sidebar.text_input("Duración del pulso tp [s]", value="5.0")
c_text = st.sidebar.text_input("Concentración inicial [mM]", value="1.0")

E_valid = True
tp_valid = True
c_valid = True

try:
    E_app = _parse_float(E_text)
except Exception:
    E_valid = False
    E_app = float("nan")
    st.sidebar.error("E inválido. Ej.: 0.10 o -0.25")

try:
    tpulse = _parse_float(tp_text)
    if tpulse <= 0:
        tp_valid = False
        st.sidebar.error("tp debe ser > 0.")
except Exception:
    tp_valid = False
    tpulse = float("nan")
    st.sidebar.error("tp inválido. Ej.: 5 o 12.5")

# Concentración inicial (mM) -> (mol/m^3). Numéricamente: 1 mM = 1 mol/m^3
try:
    c_mM = _parse_float(c_text)
    if c_mM <= 0:
        c_valid = False
        st.sidebar.error("La concentración inicial debe ser > 0.")
    c_total = float(c_mM)  # mol/m^3
except Exception:
    c_valid = False
    c_total = float("nan")
    st.sidebar.error("Concentración inválida. Ej.: 1.0")

sim_enabled = E_valid and tp_valid and c_valid

colb1, colb2 = st.sidebar.columns(2)
btn_add = colb1.button("Simular + añadir", disabled=not sim_enabled)
btn_clear = colb2.button("Limpiar", disabled=len(st.session_state.runs) == 0)

if btn_clear:
    st.session_state.runs = []
    st.session_state.run_id = 1

with st.sidebar.expander("Parámetros del sistema"):
    st.write(f"D = {D:.2e} m²/s")
    st.write(f"A = {4.0 * math.pi * (a ** 2):.2e} m²")
    st.write(f"E0' = {E0:.3g} V")
    st.write(f"c_total = {c_total:.3g} mM")
    st.write(f"Ru = {Ru:.3g} Ω")
    st.write(f"Cdl = {Cdl:.3g} F")

if btn_add and sim_enabled:
    with st.spinner("Resolviendo..."):
        times, jF, r, c_ox_final, r_max, dt = solve_spherical_reversible_chronoamperometry(E_app, tpulse)

    # Faradaica total: I_F = 4πa² jF
    area = 4.0 * math.pi * (a ** 2)
    I_F = area * jF

    # Capacitiva y total
    I_cap = capacitive_current(E_app, times)
    I_total = I_F + I_cap

    st.session_state.runs.append(
        {
            "id": st.session_state.run_id,
            "E": float(E_app),
            "tpulse": float(tpulse),
            "times": times,
            "I_total": I_total,
            "I_F": I_F,
            "I_cap": I_cap,
            "dt": float(dt),
            "r_max": float(r_max),
        }
    )
    st.session_state.run_id += 1

if len(st.session_state.runs) == 0:
    st.info("Introduce E, tp y concentración inicial, luego pulsa “Simular + añadir”.")
    st.stop()

# Tabla
st.subheader("Curvas almacenadas")
rows = []
for run in st.session_state.runs:
    rows.append(
        {
            "ID": run["id"],
            "E [V]": run["E"],
            "tp [s]": run["tpulse"],
            "dt [s]": run["dt"],
            "r_max [m]": run["r_max"],
        }
    )
st.dataframe(rows, use_container_width=True, hide_index=True)

ids = [r["id"] for r in st.session_state.runs]
selected_ids = st.multiselect("Selecciona curvas a mostrar", options=ids, default=ids)
selected = [r for r in st.session_state.runs if r["id"] in selected_ids]
if not selected:
    st.warning("No hay curvas seleccionadas.")
    st.stop()

# Descarga SOLO I_total
st.markdown("### Descarga")
txt_it = export_txt_itotal(selected)
st.download_button(
    "Descargar I(t) .txt",
    data=txt_it,
    file_name="I_vs_t.txt",
    mime="text/plain",
    use_container_width=True,
)

# Gráficas: |I_total| y regresión
st.markdown("### Visualización")
col_left, col_right = st.columns(2)

with col_left:
    fig, ax = plt.subplots()
    for run in selected:
        ax.plot(run["times"], np.abs(run["I_total"]), label=f"ID {run['id']}: E={run['E']:.3g} V, tp={run['tpulse']:.3g} s")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("|I| [A]")
    ax.set_title("Corriente (valor absoluto)")
    ax.grid(True)
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)

# Slider de rango para regresión (sólo Itotal)
t_min_possible = float(min(run["times"][0] for run in selected))
t_max_possible = float(max(run["times"][-1] for run in selected))

# valores iniciales
t0_lo = max(t_min_possible, min(0.01, t_max_possible * 0.1))
t0_hi = max(t0_lo, min(t_max_possible, max(0.1, t_max_possible * 0.8)))

# paso del slider (evitar step=0)
step = max((t_max_possible - t_min_possible) / 500.0, t_min_possible)

with col_right:
    st.markdown("**Regresión: ln|I_total| vs ln(t)**")
    t_min_reg, t_max_reg = st.slider(
        "Rango de tiempos [s]",
        min_value=t_min_possible,
        max_value=t_max_possible,
        value=(t0_lo, t0_hi),
        step=step,
    )

    fig, ax = plt.subplots()
    summary = []

    for run in selected:
        t = run["times"]
        I = run["I_total"]
        mask = (t >= t_min_reg) & (t <= t_max_reg)
        reg = regression_lnI_lnT(t[mask], I[mask])
        if reg is None:
            summary.append({"ID": run["id"], "m": np.nan, "b": np.nan, "R2": np.nan, "N": int(np.sum(mask))})
            continue

        ax.scatter(reg["x"], reg["y"], s=12, label=f"ID {run['id']} datos")
        ax.plot(reg["x"], reg["yhat"], linewidth=2, label=f"ID {run['id']} fit: m={reg['m']:.3g}, R²={reg['r2']:.4f}")
        summary.append({"ID": run["id"], "m": reg["m"], "b": reg["b"], "R2": reg["r2"], "N": reg["n"]})

    ax.set_xlabel("ln(t)")
    ax.set_ylabel("ln(|I|)")
    ax.set_title("Ajuste lineal en el rango seleccionado")
    ax.grid(True)
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)

    st.dataframe(summary, use_container_width=True, hide_index=True)

# --- NUEVO: Gráfica y regresión |I| vs t^{-1/2} (mismo rango t_min_reg..t_max_reg) ---
st.markdown("### Regresión: |I| vs t$^{-1/2}$")

fig, ax = plt.subplots()
summary2 = []

for run in selected:
    t = run["times"]
    I = run["I_total"]
    mask = (t >= t_min_reg) & (t <= t_max_reg)

    reg2 = regression_I_vs_tinvhalf(t[mask], I[mask])
    if reg2 is None:
        summary2.append({"ID": run["id"], "m": np.nan, "b": np.nan, "R2": np.nan, "N": int(np.sum(mask))})
        continue

    ax.scatter(reg2["x"], reg2["y"], s=12, label=f"ID {run['id']} datos")
    ax.plot(reg2["x"], reg2["yhat"], linewidth=2, label=f"ID {run['id']} fit: m={reg2['m']:.3g}, R²={reg2['r2']:.4f}")
    summary2.append({"ID": run["id"], "m": reg2["m"], "b": reg2["b"], "R2": reg2["r2"], "N": reg2["n"]})

ax.set_xlabel(r"t$^{-1/2}$ [s$^{-1/2}$]")
ax.set_ylabel("|I| [A]")
ax.set_title("Ajuste lineal en el rango seleccionado")
ax.grid(True)
ax.legend(fontsize=8)
st.pyplot(fig, use_container_width=True)

st.dataframe(summary2, use_container_width=True, hide_index=True)

st.caption(
    "I_total = I_F + I_cap, con I_cap=(E/Ru)·exp(-t/(Ru·Cdl)). "
    "Regresiones: ln|I_total| vs ln(t) y |I_total| vs t^{-1/2} en el rango de tiempos seleccionado."
)
