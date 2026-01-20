# app.py
# Streamlit app: Cronoamperometría reversible en electrodo esférico
# Cambios:
# 1) Reporta CORRIENTE I(t) (A), no densidad de corriente.
# 2) I_total = I_faradaica + I_capacitiva, con:
#       I_cap(t) = (E/Ru) * exp(-t/(Ru*Cdl))
#    Ru y Cdl se definen en el código (no editables).
# 3) Gráfico y análisis de regresión lineal: ln|I| vs ln(t) en un rango de tiempos seleccionable.

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

# Sistema físico (ajusta aquí si quieres cambiar el caso base)
D = 1.0e-9          # m^2/s
a = 25e-6           # m (radio electrodo)
E0 = 0.0            # V (E0')
c_total = 1.0       # mol/m^3 (c_ox + c_red = constante)

# Parámetros eléctricos (NO editables)
Ru = 50.0           # ohm (resistencia de disolución)
Cdl = 20e-6         # F (capacitancia doble capa)

# Discretización numérica (fija, pero depende de tp para mantener coste razonable)
TARGET_STEPS = 4000          # pasos de tiempo objetivo
DT_MIN = 1e-5                # s
DT_MAX = 5e-3                # s
DR = 2e-6                    # m (paso radial fijo)

# Dominio externo (fijo como regla semi-infinita, pero calculado a partir de tp)
DOMAIN_FACTOR = 6.0          # r_max = a + factor*sqrt(D*tp)


# ----------------------------
# Utilidades
# ----------------------------
def _parse_float(text: str) -> float:
    return float(text.strip().replace(",", "."))


def eta(E: float, E0_: float) -> float:
    return (F / (R * T)) * (E - E0_)


def c_surf_nernst(c_tot: float, E: float, E0_: float) -> float:
    """c_ox en la superficie para un par reversible Ox + e- <-> Red, con c_tot constante."""
    x = math.exp(eta(E, E0_))
    return c_tot * x / (1.0 + x)


def thomas_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Algoritmo de Thomas para tridiagonal."""
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
    """Resuelve c_ox(r,t) en electrodo esférico con:
       - c_ox(a,t) = Nernst(E_app) (pulso constante durante tpulse)
       - flujo nulo en r=r_max (dc/dr=0)

       Discretización conservativa (volúmenes finitos) e implícita.
       Devuelve: times, j_F(t) (A/m^2), r (m), c_ox_final (perfil final), r_max, dt
    """
    if tpulse <= 0:
        raise ValueError("tpulse debe ser > 0")

    # dt adaptativo (no editable por el usuario)
    dt = tpulse / TARGET_STEPS
    dt = min(max(dt, DT_MIN), DT_MAX)
    nsteps = int(math.ceil(tpulse / dt))
    dt = tpulse / nsteps  # cierra exactamente en tpulse

    # dominio externo (no editable, pero calculado)
    r_max = a + DOMAIN_FACTOR * math.sqrt(D * tpulse)
    if r_max < a + 5 * DR:
        r_max = a + 5 * DR

    # mallado
    n = int(math.ceil((r_max - a) / DR)) + 1
    r = a + np.arange(n) * DR
    r_face = 0.5 * (r[:-1] + r[1:])  # caras i+1/2

    # condición superficial
    c0 = c_surf_nernst(c_total, E_app, E0)

    # estado inicial: bulk uniforme (antes del pulso)
    C = np.full(n, c_total, dtype=float)
    C[0] = c0  # al iniciar el pulso

    lam = D * dt / (DR ** 2)

    # Desconocidas: C[1]..C[n-1]
    Nunk = n - 1
    diag = np.zeros(Nunk, dtype=float)
    lower = np.zeros(Nunk - 1, dtype=float)
    upper = np.zeros(Nunk - 1, dtype=float)

    # Construcción coeficientes conservativos
    for k in range(Nunk - 1):
        i = k + 1
        A = lam * (r_face[i - 1] ** 2 / (r[i] ** 2))  # cara i-1/2
        B = lam * (r_face[i] ** 2 / (r[i] ** 2))      # cara i+1/2
        diag[k] = 1.0 + A + B
        if k > 0:
            lower[k - 1] = -A
        upper[k] = -B

    # Último nodo i=n-1 con flujo nulo en la cara externa
    i_last = n - 1
    A_last = lam * (r_face[-1] ** 2 / (r[i_last] ** 2))  # cara entre n-2 y n-1
    diag[-1] = 1.0 + A_last
    lower[-1] = -A_last

    times = np.empty(nsteps, dtype=float)
    jF = np.empty(nsteps, dtype=float)  # densidad faradaica en A/m^2

    # precomputar el A de i=1 para el RHS (término Dirichlet)
    A_i1 = lam * (r_face[0] ** 2 / (r[1] ** 2))

    for k in range(nsteps):
        rhs = np.empty(Nunk, dtype=float)
        rhs[:-1] = C[1:-1].copy()
        rhs[0] += A_i1 * c0
        rhs[-1] = C[-1]

        C_unk = thomas_tridiagonal(lower, diag, upper, rhs)
        C[0] = c0
        C[1:] = C_unk

        # Flujo molar en r=a: N = -D dc/dr
        dc_dr_a = (-3.0 * C[0] + 4.0 * C[1] - C[2]) / (2.0 * DR)
        N_mol = -D * dc_dr_a

        times[k] = (k + 1) * dt
        jF[k] = F * N_mol  # n=1 fijo

    return times, jF, r, C, r_max, dt


def capacitive_current(E_app: float, t: np.ndarray) -> np.ndarray:
    """I_cap(t) = (E/Ru) * exp(-t/(Ru*Cdl))"""
    tau = Ru * Cdl
    return (E_app / Ru) * np.exp(-t / tau)


def regression_lnI_lnT(t: np.ndarray, I: np.ndarray):
    """Regresión lineal y = m x + b con x=ln(t), y=ln(|I|). Devuelve m,b,R2 y arrays filtrados."""
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

    return {"m": float(m), "b": float(b), "r2": r2, "t": t2, "x": x, "y": y, "yhat": yhat}


def export_txt_currents(runs):
    lines = []
    lines.append("# Cronoamperometría esférica reversible: corrientes (A)")
    lines.append("# I_total = I_F + I_cap ;  I_cap = (E/Ru)*exp(-t/(Ru*Cdl))")
    lines.append(f"# Ru[ohm]={Ru}, Cdl[F]={Cdl}")
    lines.append("# Columnas: t[s]\tI_F[A]\tI_cap[A]\tI_total[A]\t|I_total|[A]")
    for run in runs:
        lines.append("")
        lines.append(f"# --- RUN {run['id']} ---")
        lines.append(f"# E[V]={run['E']}, tpulse[s]={run['tpulse']}, r_max[m]={run['r_max']}, dt[s]={run['dt']}")
        t = run["times"]
        IF = run["I_F"]
        Icap = run["I_cap"]
        Itot = run["I_total"]
        for ti, a1, a2, a3 in zip(t, IF, Icap, Itot):
            lines.append(f"{ti:.12g}\t{a1:.12g}\t{a2:.12g}\t{a3:.12g}\t{abs(a3):.12g}")
    return "\n".join(lines) + "\n"


def export_txt_profile_final(runs, species):
    lines = []
    lines.append("# Cronoamperometría esférica reversible: perfiles (al final del pulso, t=tp)")
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
    st.session_state.runs = []
if "run_id" not in st.session_state:
    st.session_state.run_id = 1

st.sidebar.header("Entradas (ajustables)")
E_text = st.sidebar.text_input("Potencial aplicado E [V]", value="0.10")
tp_text = st.sidebar.text_input("Duración del pulso tp [s]", value="5.0")

E_valid = True
tp_valid = True

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

sim_enabled = E_valid and tp_valid

colb1, colb2 = st.sidebar.columns(2)
btn_add = colb1.button("Simular + añadir", disabled=not sim_enabled)
btn_clear = colb2.button("Limpiar", disabled=len(st.session_state.runs) == 0)

if btn_clear:
    st.session_state.runs = []
    st.session_state.run_id = 1

with st.sidebar.expander("Parámetros fijados en el código"):
    st.write(f"D = {D:.2e} m²/s")
    st.write(f"a = {a:.2e} m")
    st.write(f"E0' = {E0:.3g} V")
    st.write(f"c_total = {c_total:.3g} mol/m³")
    st.write(f"Ru = {Ru:.3g} Ω")
    st.write(f"Cdl = {Cdl:.3g} F")
    st.write(f"Δr = {DR:.2e} m")
    st.write(f"dt adaptativo: tp/{TARGET_STEPS} con límites [{DT_MIN}, {DT_MAX}] s")
    st.write(f"r_max = a + {DOMAIN_FACTOR}·sqrt(D·tp)  (flujo nulo en r_max)")

if btn_add and sim_enabled:
    with st.spinner("Resolviendo difusión esférica (implícito)..."):
        times, jF, r, c_ox_final, r_max, dt = solve_spherical_reversible_chronoamperometry(E_app, tpulse)

    # Corriente faradaica total (A): I_F = 4πa^2 * jF
    area = 4.0 * math.pi * (a ** 2)
    I_F = area * jF

    # Corriente capacitiva (A), y total
    I_cap = capacitive_current(E_app, times)
    I_total = I_F + I_cap

    dist_um = (r - a) * 1e6

    st.session_state.runs.append(
        {
            "id": st.session_state.run_id,
            "E": float(E_app),
            "tpulse": float(tpulse),
            "times": times,
            "jF": jF,
            "I_F": I_F,
            "I_cap": I_cap,
            "I_total": I_total,
            "r": r,
            "dist_um": dist_um,
            "c_ox_final": c_ox_final.copy(),
            "r_max": float(r_max),
            "dt": float(dt),
        }
    )
    st.session_state.run_id += 1

if len(st.session_state.runs) == 0:
    st.info("Introduce E y tp, luego pulsa “Simular + añadir”.")
    st.stop()

# Tabla de corridas
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

# Descargas
st.markdown("### Descargas")
txt_curr = export_txt_currents(selected)
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "Descargar corrientes (I_F, I_cap, I_total) .txt",
        data=txt_curr,
        file_name="currents_spherical_reversible.txt",
        mime="text/plain",
        use_container_width=True,
    )

species = st.radio(
    "Perfil a representar (al final del pulso)",
    ["Oxidada (c_ox)", "Reducida (c_red)", "Ambas (c_ox y c_red)"],
    horizontal=True,
)
txt_prof = export_txt_profile_final(selected, species)
with dl2:
    st.download_button(
        "Descargar perfiles finales .txt",
        data=txt_prof,
        file_name="profiles_final_spherical_reversible.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ----------------------------
# Gráficas: corriente (izq) y perfiles (der)
# ----------------------------
st.markdown("### Visualización")
col_left, col_right = st.columns(2)

with col_left:
    fig, ax = plt.subplots()
    for run in selected:
        ax.plot(run["times"], np.abs(run["I_total"]), label=f"ID {run['id']}: E={run['E']:.3g} V, tp={run['tpulse']:.3g} s")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("|I_total| [A]")
    ax.set_title("Corriente total (valor absoluto): |I_F + I_cap|")
    ax.grid(True)
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)

    with st.expander("Ver componentes (I_F e I_cap)"):
        fig2, ax2 = plt.subplots()
        for run in selected:
            ax2.plot(run["times"], run["I_F"], label=f"ID {run['id']} I_F")
            ax2.plot(run["times"], run["I_cap"], linestyle="--", label=f"ID {run['id']} I_cap")
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("I [A]")
        ax2.set_title("Componentes: faradaica y capacitiva")
        ax2.grid(True)
        ax2.legend(fontsize=8)
        st.pyplot(fig2, use_container_width=True)

with col_right:
    fig, ax = plt.subplots()
    ax.set_ylim(0.0, c_total)  # eje y fijo: 0..concentración inicial
    for run in selected:
        dist = run["dist_um"]
        c_ox = run["c_ox_final"]
        c_red = c_total - c_ox
        tag = f"ID {run['id']} (t=tp={run['tpulse']:.3g}s)"
        if species == "Oxidada (c_ox)":
            ax.plot(dist, c_ox, label=tag)
        elif species == "Reducida (c_red)":
            ax.plot(dist, c_red, label=tag)
        else:
            ax.plot(dist, c_ox, label=f"{tag} c_ox")
            ax.plot(dist, c_red, linestyle="--", label=f"{tag} c_red")
    ax.set_xlabel("Distancia a la superficie (µm)")
    ax.set_ylabel("c [mol/m³]")
    ax.set_title("Perfiles de concentración (final del pulso)")
    ax.grid(True)
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)

# ----------------------------
# Regresión ln|I| vs ln(t)
# ----------------------------
st.markdown("### Regresión: ln|I| vs ln(t)")

t_global_max = float(max(run["times"][-1] for run in selected))
reg_col1, reg_col2, reg_col3 = st.columns(3)

with reg_col1:
    t_min_reg = st.number_input("t_min [s]", value=float(min(0.01, t_global_max * 0.1)), min_value=0.0, format="%.6g")
with reg_col2:
    t_max_reg = st.number_input("t_max [s]", value=float(min(t_global_max, max(0.1, t_global_max * 0.8))), min_value=0.0, format="%.6g")
with reg_col3:
    use_total = st.selectbox("Señal para regresión", ["|I_total|", "|I_F|"], index=0)

if t_max_reg <= t_min_reg:
    st.error("El rango debe cumplir t_max > t_min.")
else:
    fig, ax = plt.subplots()
    summary = []

    for run in selected:
        t = run["times"]
        if use_total == "|I_total|":
            I_sig = run["I_total"]
        else:
            I_sig = run["I_F"]

        mask = (t >= t_min_reg) & (t <= t_max_reg)
        t_sel = t[mask]
        I_sel = I_sig[mask]

        reg = regression_lnI_lnT(t_sel, I_sel)
        if reg is None:
            summary.append({"ID": run["id"], "m": np.nan, "b": np.nan, "R2": np.nan, "N": int(np.sum(mask))})
            continue

        # Scatter + fit
        ax.scatter(reg["x"], reg["y"], s=12, label=f"ID {run['id']} datos")
        ax.plot(reg["x"], reg["yhat"], linewidth=2, label=f"ID {run['id']} fit: m={reg['m']:.3g}, R²={reg['r2']:.4f}")

        summary.append({"ID": run["id"], "m": reg["m"], "b": reg["b"], "R2": reg["r2"], "N": int(reg["x"].size)})

    ax.set_xlabel("ln(t)")
    ax.set_ylabel(f"ln({use_total})")
    ax.set_title("Ajuste lineal en el rango seleccionado")
    ax.grid(True)
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)

    st.dataframe(summary, use_container_width=True, hide_index=True)

st.caption(
    "Modelo: difusión esférica reversible con condición superficial tipo Nernst y flujo nulo en r_max. "
    "La corriente total incluye un término capacitivo RC: I_cap=(E/Ru)·exp(-t/(Ru·Cdl))."
)
