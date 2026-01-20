import numpy as np

# --- Constantes ---
F = 96485.0  # C/mol
R = 8.314    # J/mol/K
T = 298.0    # K


def eta(E: float, E0: float) -> float:
    """Sobretensión adimensional: (F/RT)(E - E0)."""
    return (F / (R * T)) * (E - E0)


def c_surf_nernst(c_bulk: float, E: float, E0: float) -> float:
    """Concentración superficial bajo condición de Nernst (forma logística)."""
    x = np.exp(eta(E, E0))
    return c_bulk * x / (1.0 + x)


def _thomas_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Resuelve un sistema tridiagonal Ax=rhs con el algoritmo de Thomas."""
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


def solve_diffusion_implicit_planar(
    *,
    D: float,
    delta_x: float,
    delta_t: float,
    max_t: float,
    max_x: float,
    c_bulk: float,
    E: float,
    E0: float,
):
    """Difusión lineal hacia un electrodo plano con:
       - c(0,t) fijada por Nernst (Dirichlet)
       - flujo nulo en x=max_x: dc/dx = 0 (Neumann)

    PDE:  ∂c/∂t = D ∂²c/∂x², x∈[0,max_x]
    """
    if D <= 0:
        raise ValueError("D debe ser > 0")
    if delta_x <= 0 or delta_t <= 0:
        raise ValueError("delta_x y delta_t deben ser > 0")
    if max_t <= 0:
        raise ValueError("max_t debe ser > 0")
    if max_x <= 0:
        raise ValueError("max_x debe ser > 0")
    if max_x < 5 * delta_x:
        raise ValueError("max_x debe ser al menos ~5*Δx (mejor bastante mayor)")

    n = int(np.ceil(max_x / delta_x)) + 1
    m = int(np.ceil(max_t / delta_t))

    lam = D * delta_t / (delta_x ** 2)

    c0 = c_surf_nernst(c_bulk, E, E0)

    # Estado inicial
    C = np.full(n, c_bulk)
    C[0] = c0

    # Desconocidas: C[1]..C[n-1] (tamaño n-1)
    Nunk = n - 1
    diag = np.empty(Nunk, dtype=float)
    lower = np.empty(Nunk - 1, dtype=float)
    upper = np.empty(Nunk - 1, dtype=float)

    # Filas interiores (corresponden a i=1..n-2)
    diag[:-1] = 1.0 + 2.0 * lam
    lower[:-1] = -lam
    upper[:] = -lam

    # Última fila: condición Neumann (flujo nulo) => C[n-1] - C[n-2] = 0
    diag[-1] = 1.0
    lower[-1] = -1.0  # coeficiente de C[n-2] en la última ecuación
    # upper no se usa en la última fila (no existe superdiagonal)

    times = np.empty(m)
    j = np.empty(m)
    profiles = np.empty((m, n))

    for k in range(m):
        rhs = np.empty(Nunk, dtype=float)

        # Ecuaciones interiores: -lam*C[i-1] + (1+2lam)C[i] - lam*C[i+1] = C_old[i]
        rhs[:-1] = C[1:-1].copy()

        # Incorporar C[0]=c0 conocida en la primera ecuación interior (i=1)
        rhs[0] += lam * c0

        # Última ecuación: C[n-1] - C[n-2] = 0
        rhs[-1] = 0.0

        C_unk = _thomas_tridiagonal(lower, diag, upper, rhs)
        C[0] = c0
        C[1:] = C_unk

        # Flujo molar hacia el electrodo: N = -D (dc/dx)|_{x=0}
        dc_dx_0 = (-3.0 * C[0] + 4.0 * C[1] - C[2]) / (2.0 * delta_x)
        N_mol = -D * dc_dx_0

        times[k] = (k + 1) * delta_t
        j[k] = F * N_mol  # n=1 fijo
        profiles[k] = C

    x = np.arange(n) * delta_x
    return times, j, x, profiles

def solve_diffusion_implicit_spherical(
    *,
    D: float,
    delta_r: float,
    delta_t: float,
    max_t: float,
    a: float,
    r_max: float,
    c_bulk: float,
    E: float,
    E0: float,
):
    """Difusión esférica con:
       - c(a,t) fijada por Nernst (Dirichlet)
       - flujo nulo en r=r_max: dc/dr = 0 (Neumann)
    Discretización conservativa:  ∂c/∂t = (D/r^2) ∂/∂r (r^2 ∂c/∂r)
    Implícito, sistema tridiagonal.
    """
    if D <= 0:
        raise ValueError("D debe ser > 0")
    if delta_r <= 0 or delta_t <= 0:
        raise ValueError("delta_r y delta_t deben ser > 0")
    if max_t <= 0:
        raise ValueError("max_t debe ser > 0")
    if a <= 0:
        raise ValueError("a debe ser > 0")
    if r_max <= a:
        raise ValueError("r_max debe ser > a")
    if r_max < a + 5 * delta_r:
        raise ValueError("r_max debe ser al menos ~a+5*Δr (mejor bastante mayor)")

    # Mallado radial
    n = int(np.ceil((r_max - a) / delta_r)) + 1
    m = int(np.ceil(max_t / delta_t))
    r = a + np.arange(n) * delta_r

    # Caras (i+1/2)
    r_face = 0.5 * (r[:-1] + r[1:])  # tamaño n-1

    # Condición en electrodo (Dirichlet en r=a)
    c0 = c_surf_nernst(c_bulk, E, E0)

    # Estado inicial (puedes mantener c_bulk uniforme)
    C = np.full(n, c_bulk, dtype=float)
    C[0] = c0

    # Desconocidas: C[1]..C[n-1] (tamaño n-1)
    Nunk = n - 1
    diag = np.zeros(Nunk, dtype=float)
    lower = np.zeros(Nunk - 1, dtype=float)
    upper = np.zeros(Nunk - 1, dtype=float)

    # Construcción de coeficientes implícitos (conservativos)
    # Para nodo físico i (1..n-2): coeficientes basados en r_{i±1/2}^2 / r_i^2
    # Map: incógnita k corresponde a i=k+1
    for k in range(Nunk - 1):  # k=0..n-3 => i=1..n-2
        i = k + 1
        A = (D * delta_t / (delta_r ** 2)) * (r_face[i - 1] ** 2 / (r[i] ** 2))  # (i-1/2)
        B = (D * delta_t / (delta_r ** 2)) * (r_face[i] ** 2 / (r[i] ** 2))      # (i+1/2)

        diag[k] = 1.0 + A + B
        if k > 0:
            lower[k - 1] = -A
        upper[k] = -B

    # Última ecuación (i = n-1) con flujo nulo en cara externa:
    # flujo en (n-1/2) existe; flujo en (n+1/2) = 0  => solo contribuye la cara interna
    i = n - 1
    A_last = (D * delta_t / (delta_r ** 2)) * (r_face[-1] ** 2 / (r[i] ** 2))  # cara (n-3/2) en notación de r_face[-1]=r_{n-3/2}
    # OJO: r_face[-1] es la cara entre n-2 y n-1, que es la "cara interna" del último control.
    diag[-1] = 1.0 + A_last
    lower[-1] = -A_last  # conecta con C[n-2] (que es incógnita k=Nunk-2)

    times = np.empty(m)
    j = np.empty(m)
    profiles = np.empty((m, n))

    for step in range(m):
        rhs = np.empty(Nunk, dtype=float)

        # RHS para i=1..n-2
        rhs[:-1] = C[1:-1].copy()

        # Incorporar Dirichlet en i=1: término A*C0 pasa al RHS
        # Aquí A corresponde a i=1 => cara i-1/2 = r_face[0]
        A_i1 = (D * delta_t / (delta_r ** 2)) * (r_face[0] ** 2 / (r[1] ** 2))
        rhs[0] += A_i1 * c0

        # RHS última ecuación (i=n-1)
        rhs[-1] = C[-1]

        # Resolver tridiagonal para C[1..n-1]
        C_unk = _thomas_tridiagonal(lower, diag, upper, rhs)
        C[0] = c0
        C[1:] = C_unk

        # Flujo molar en r=a (hacia el electrodo): N = -D (dc/dr)|_a
        dc_dr_a = (-3.0 * C[0] + 4.0 * C[1] - C[2]) / (2.0 * delta_r)
        N_mol = -D * dc_dr_a

        times[step] = (step + 1) * delta_t
        j[step] = F * N_mol  # n=1 fijo
        profiles[step] = C

    return times, j, r, profiles


