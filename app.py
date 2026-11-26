"""
Proyecto Integrador U4 - INC-2505 Análisis y Exploración de Datos
Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego
Dashboard en Streamlit · estilo liquid glass (Hitos 1-4)
"""
import io

import duckdb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Invernadero · Rendimiento e inferencia", page_icon="🌱", layout="wide")

# ---------------------------------------------------------------------------
# PALETA (tema agro: bosque, menta y cosecha dorada)
# ---------------------------------------------------------------------------
BG = "#0b2018"
MENTA = "#7dd3a8"     # Tratamiento A
DORADO = "#f4c95d"    # Tratamiento B / líneas de ajuste
TEXTO = "#e6f4ec"
SUAVE = "#9fbfae"
COLORES_TRAT = {"Tratamiento_A": MENTA, "Tratamiento_B": DORADO}
CMAP = LinearSegmentedColormap.from_list("agro", ["#d98e3f", "#16352a", "#6ee7a1"])

# ---------------------------------------------------------------------------
# ESTILO LIQUID GLASS (CSS)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; color: #e6f4ec; }

.stApp {
    background:
        radial-gradient(circle at 12% 8%,  rgba(52,211,153,.28), transparent 38%),
        radial-gradient(circle at 88% 18%, rgba(244,201,93,.16), transparent 34%),
        radial-gradient(circle at 55% 95%, rgba(20,184,166,.22), transparent 45%),
        linear-gradient(135deg, #06140f 0%, #0d2a1f 55%, #08201a 100%);
    background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }

/* Vidrio base */
.glass, [class*="st-key-card"] {
    background: linear-gradient(145deg, rgba(255,255,255,.10), rgba(255,255,255,.04));
    backdrop-filter: blur(20px) saturate(170%);
    -webkit-backdrop-filter: blur(20px) saturate(170%);
    border: 1px solid rgba(255,255,255,.14);
    border-radius: 24px;
    box-shadow: 0 10px 34px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.22);
}
[class*="st-key-card"] { padding: 1.1rem 1.3rem 0.6rem 1.3rem; }

/* Hero */
.hero { padding: 1.6rem 1.9rem; margin-bottom: 1.1rem; }
.hero h1 { font-size: 1.85rem; font-weight: 600; letter-spacing: -.02em; margin: 0 0 .35rem 0; }
.hero p  { color: #9fbfae; margin: 0 0 .9rem 0; font-weight: 300; }
.pill {
    display: inline-block; padding: .22rem .75rem; margin: 0 .4rem .3rem 0;
    border-radius: 999px; font-size: .74rem; color: #cfe9da;
    background: rgba(125,211,168,.12); border: 1px solid rgba(125,211,168,.28);
}

/* KPIs */
.kpi { padding: 1.15rem 1.4rem; height: 100%; }
.kpi-label { font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; color: #9fbfae; }
.kpi-value { font-size: 2.1rem; font-weight: 600; letter-spacing: -.02em; margin: .2rem 0; }
.kpi-sub   { font-size: .8rem; color: #9fbfae; }
.ok  { color: #7dd3a8; }
.bad { color: #f08a6b; }

/* Títulos de tarjeta */
.card-title { font-size: .95rem; font-weight: 600; margin: 0 0 .1rem 0; }
.card-sub   { font-size: .76rem; color: #9fbfae; margin: 0 0 .5rem 0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,.05);
    backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
    border-right: 1px solid rgba(255,255,255,.10);
}
section[data-testid="stSidebar"] h2 { font-size: 1.05rem; font-weight: 600; }

/* Tabs minimalistas */
button[data-baseweb="tab"] { background: transparent; color: #9fbfae; }
button[data-baseweb="tab"][aria-selected="true"] { color: #f4c95d; }
div[data-baseweb="tab-highlight"] { background-color: #f4c95d; }

/* Detalles */
hr { border-color: rgba(255,255,255,.10); }
div[data-testid="stExpander"] {
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12); border-radius: 18px;
}
pre { background: rgba(0,0,0,.25) !important; border-radius: 14px; }
.footer { text-align:center; color:#6f8f7e; font-size:.74rem; margin-top:1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Utilidades de gráficas (fondo transparente, tipografía clara)
# ---------------------------------------------------------------------------
def nueva_fig(w=6.0, h=3.7):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    return fig, ax


def pulir(ax, xlabel=None, ylabel=None, grid_y=True):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=SUAVE, labelsize=8, length=0)
    ax.grid(False)
    if grid_y:
        ax.grid(axis="y", color=(1, 1, 1, 0.08), linewidth=0.8)
    ax.set_xlabel(xlabel or "", color=SUAVE, fontsize=9)
    ax.set_ylabel(ylabel or "", color=SUAVE, fontsize=9)


def mostrar(fig):
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=200, bbox_inches="tight")
    plt.close(fig)
    st.image(buf, width="stretch")


def tarjeta_titulo(titulo, sub):
    st.markdown(f'<div class="card-title">{titulo}</div><div class="card-sub">{sub}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HITO 1: Telemetría en DuckDB (semilla determinista 0.42)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_telemetria() -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    con.execute("SELECT setseed(0.42)")  # Paso 1.1

    # Paso 1.2: variables aleatorias base (N = 100 parcelas)
    con.execute("""
        CREATE TEMP TABLE base AS
        SELECT
            i                                   AS parcela_id,
            ROUND(100.0 + random() * 200.0, 2)  AS volumen_riego,     -- 100-300 m3/ha
            ROUND(50.0  + random() * 100.0, 2)  AS fertilizante_kg,   -- 50-150 kg/ha
            random()                            AS p_trat,            -- prob. de grupo
            (random() * 1.6 - 0.8)              AS ruido              -- -0.8 a +0.8 ton/ha
        FROM range(1, 101) t(i)
    """)

    # Ecuación: 5.0 + 0.035*riego + 0.020*fert + efecto_trat + ruido
    con.execute("""
        CREATE TABLE telemetria AS
        SELECT
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            CASE WHEN p_trat < 0.50 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END
                AS tipo_tratamiento,
            ROUND(
                5.0
                + 0.035 * volumen_riego
                + 0.020 * fertilizante_kg
                + CASE WHEN p_trat >= 0.50 THEN 1.8 ELSE 0.0 END
                + ruido
            , 2) AS rendimiento_ton
        FROM base
    """)

    # Paso 1.3: limpieza / extracción SQL antes de pasar a Python
    df = con.execute("""
        SELECT DISTINCT *
        FROM telemetria
        WHERE volumen_riego IS NOT NULL
          AND fertilizante_kg IS NOT NULL
          AND rendimiento_ton IS NOT NULL
        ORDER BY parcela_id
    """).df()
    con.close()

    # Temperatura (solo para el heatmap que la pide). No influye en el
    # rendimiento: es una variable ambiental independiente.
    rng = np.random.default_rng(42)
    df["temperatura_c"] = np.round(rng.uniform(18.0, 32.0, len(df)), 2)
    return df


df_full = cargar_telemetria()

# ---------------------------------------------------------------------------
# HITO 4.2: filtros dinámicos
# ---------------------------------------------------------------------------
st.sidebar.header("🌿 Filtros")
trats = st.sidebar.multiselect(
    "Tipo de tratamiento",
    options=sorted(df_full["tipo_tratamiento"].unique()),
    default=sorted(df_full["tipo_tratamiento"].unique()),
)
r_min, r_max = float(df_full.volumen_riego.min()), float(df_full.volumen_riego.max())
rango_riego = st.sidebar.slider("Volumen de riego (m³/ha)", r_min, r_max, (r_min, r_max))
incluir_trat = st.sidebar.checkbox("Incluir tratamiento en el modelo OLS", value=False)
st.sidebar.caption("Los KPIs, gráficas y pruebas se recalculan con los filtros.")

df = df_full[
    df_full.tipo_tratamiento.isin(trats)
    & df_full.volumen_riego.between(*rango_riego)
].copy()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<div class="glass hero">
  <h1>🌱 Optimización inferencial del rendimiento agrícola</h1>
  <p>Eficiencia de riego en invernaderos de jitomate y aguacate</p>
  <span class="pill">DuckDB · semilla 0.42</span>
  <span class="pill">N = 100 parcelas</span>
  <span class="pill">Regresión OLS · 4.2</span>
  <span class="pill">t de Student · 4.3</span>
  <span class="pill">{len(df)} parcelas en vista</span>
</div>
""",
    unsafe_allow_html=True,
)

if len(df) < 10:
    st.warning("Pocos datos con los filtros actuales. Amplía el rango o selecciona más tratamientos.")
    st.stop()

# ---------------------------------------------------------------------------
# HITO 2: correlación + OLS (Subtema 4.2)
# ---------------------------------------------------------------------------
X = df[["volumen_riego", "fertilizante_kg"]].copy()
if incluir_trat:
    X["tratamiento_B"] = (df.tipo_tratamiento == "Tratamiento_B").astype(float)
X = sm.add_constant(X)
modelo = sm.OLS(df["rendimiento_ton"], X).fit()
df["residuo"] = modelo.resid  # Y_real - Y_predicho
rmse = float(np.sqrt(np.mean(modelo.resid ** 2)))

corr = df[["volumen_riego", "fertilizante_kg", "temperatura_c", "rendimiento_ton"]].corr(method="pearson")
corr.columns = corr.index = ["Riego", "Fertilizante", "Temperatura", "Rendimiento"]

# ---------------------------------------------------------------------------
# HITO 3: t-Student de dos muestras independientes (Subtema 4.3)
# H0: mu_B - mu_A <= 0   vs   H1: mu_B - mu_A > 0   (alpha = 0.05)
# ---------------------------------------------------------------------------
ya = df.loc[df.tipo_tratamiento == "Tratamiento_A", "rendimiento_ton"]
yb = df.loc[df.tipo_tratamiento == "Tratamiento_B", "rendimiento_ton"]
prueba_ok = len(ya) > 2 and len(yb) > 2
if prueba_ok:
    t_stat, p_val = stats.ttest_ind(yb, ya, equal_var=True, alternative="greater")
    dif = yb.mean() - ya.mean()
    gl = len(ya) + len(yb) - 2
    sp2 = ((len(ya) - 1) * ya.var(ddof=1) + (len(yb) - 1) * yb.var(ddof=1)) / gl
    se = np.sqrt(sp2 * (1 / len(ya) + 1 / len(yb)))
    t_crit = stats.t.ppf(0.975, gl)
    ic = (dif - t_crit * se, dif + t_crit * se)  # IC 95% bilateral de mu_B - mu_A

# ---------------------------------------------------------------------------
# KPIs (tarjetas de vidrio)
# ---------------------------------------------------------------------------
def kpi(label, value, sub):
    return f'<div class="glass kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'


if prueba_ok:
    clase = "ok" if p_val < 0.05 else "bad"
    veredicto = "Se rechaza H₀ · B supera a A" if p_val < 0.05 else "No se rechaza H₀"
    k3 = kpi("P-valor · prueba t (B > A)", f'<span class="{clase}">{p_val:.2e}</span>', veredicto)
else:
    k3 = kpi("P-valor · prueba t (B > A)", "N/D", "Selecciona ambos tratamientos")

c_a, c_b, c_c = st.columns(3, gap="medium")
c_a.markdown(kpi("Rendimiento promedio global", f"{df.rendimiento_ton.mean():.2f} <small style='font-size:.9rem;color:#9fbfae'>ton/ha</small>", f"Mín {df.rendimiento_ton.min():.1f} · Máx {df.rendimiento_ton.max():.1f}"), unsafe_allow_html=True)
c_b.markdown(kpi("R² ajustado del modelo", f"{modelo.rsquared_adj * 100:.1f} %", f"RMSE = {rmse:.3f} ton/ha"), unsafe_allow_html=True)
c_c.markdown(k3, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# HITO 4.1: cuadrícula 2x2 de gráficas
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="medium")

with col1:
    with st.container(key="card_heatmap"):
        tarjeta_titulo("Mapa de calor de correlación", "Pearson (r) entre insumos, temperatura y rendimiento")
        fig, ax = nueva_fig()
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap=CMAP, vmin=-1, vmax=1,
            linewidths=3, linecolor=BG, square=False,
            annot_kws={"color": TEXTO, "fontsize": 9},
            cbar_kws={"shrink": 0.85}, ax=ax,
        )
        ax.tick_params(colors=SUAVE, labelsize=8, length=0)
        plt.setp(ax.get_xticklabels(), rotation=0)
        cb = ax.collections[0].colorbar
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=SUAVE, labelsize=7, length=0)
        mostrar(fig)

with col2:
    with st.container(key="card_scatter"):
        tarjeta_titulo("Riego vs rendimiento", "Dispersión con recta OLS y banda de confianza al 95 %")
        fig, ax = nueva_fig()
        sns.regplot(data=df, x="volumen_riego", y="rendimiento_ton", scatter=False, ci=95,
                    color=DORADO, line_kws={"linewidth": 2}, ax=ax)
        sns.scatterplot(data=df, x="volumen_riego", y="rendimiento_ton", hue="tipo_tratamiento",
                        palette=COLORES_TRAT, s=34, alpha=0.85, edgecolor="none", ax=ax)
        pulir(ax, "Volumen de riego (m³/ha)", "Rendimiento (ton/ha)")
        leg = ax.legend(frameon=False, fontsize=8, loc="upper left", title=None)
        for t in leg.get_texts():
            t.set_color(SUAVE)
        mostrar(fig)

col3, col4 = st.columns(2, gap="medium")

with col3:
    with st.container(key="card_box"):
        tarjeta_titulo("Tratamiento A vs B", "Medianas, cuartiles (Q1, Q3), atípicos y media (◆)")
        fig, ax = nueva_fig()
        orden = [t for t in ["Tratamiento_A", "Tratamiento_B"] if t in trats]
        sns.boxplot(
            data=df, x="tipo_tratamiento", y="rendimiento_ton", order=orden,
            hue="tipo_tratamiento", palette=COLORES_TRAT, legend=False,
            width=0.45, linewidth=1.1, saturation=0.9, showmeans=True,
            medianprops={"color": BG, "linewidth": 2},
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": BG, "markersize": 6},
            boxprops={"edgecolor": "none"}, whiskerprops={"color": SUAVE}, capprops={"color": SUAVE},
            flierprops={"markerfacecolor": SUAVE, "markeredgecolor": "none", "markersize": 4},
            ax=ax,
        )
        pulir(ax, "", "Rendimiento (ton/ha)")
        ax.set_xticks(range(len(orden)))
        ax.set_xticklabels([o.replace("_", " ") for o in orden], color=TEXTO, fontsize=9)
        mostrar(fig)

with col4:
    with st.container(key="card_resid"):
        tarjeta_titulo("Residuos del modelo", "Histograma con densidad (KDE) · supuesto de normalidad")
        fig, ax = nueva_fig()
        sns.histplot(df["residuo"], kde=True, stat="density", bins=15, color=MENTA,
                     edgecolor="none", alpha=0.55, line_kws={"linewidth": 2, "color": DORADO}, ax=ax)
        ax.lines[0].set_color(DORADO)
        ax.axvline(0, color=TEXTO, linestyle=(0, (4, 4)), linewidth=1, alpha=0.6)
        pulir(ax, "Residuo = Y real − Y predicho (ton/ha)", "Densidad")
        mostrar(fig)
        _, p_sw = stats.shapiro(df["residuo"])
        st.caption(f"Shapiro-Wilk sobre residuos: p = {p_sw:.3f}")

st.write("")

# ---------------------------------------------------------------------------
# Resultados para el reporte
# ---------------------------------------------------------------------------
with st.container(key="card_resultados"):
    t1, t2 = st.tabs(["Modelo OLS · 4.2", "Prueba de hipótesis · 4.3"])

    with t1:
        mejor = corr.loc[["Riego", "Fertilizante"], "Rendimiento"].abs()
        st.write(
            f"El factor con mayor asociación al rendimiento es **{mejor.idxmax()}** "
            f"(|r| = {mejor.max():.2f}). El modelo explica ≈ **{modelo.rsquared_adj * 100:.1f} %** "
            f"de la variabilidad (R² ajustado = {modelo.rsquared_adj:.4f}) con un "
            f"**RMSE de {rmse:.3f} ton/ha**."
        )
        st.code(modelo.summary().as_text())

    with t2:
        if prueba_ok:
            st.markdown("**H₀:** μ_B − μ_A ≤ 0  ·  **H₁:** μ_B − μ_A > 0  ·  α = 0.05")
            st.write(
                f"Media A = **{ya.mean():.2f}** ton/ha (n = {len(ya)}) · "
                f"Media B = **{yb.mean():.2f}** ton/ha (n = {len(yb)}) · "
                f"Diferencia = **{dif:.2f}** ton/ha"
            )
            st.write(f"t = {t_stat:.3f} · gl = {gl} · p-valor = {p_val:.3e}")
            st.write(f"IC 95 % de μ_B − μ_A: **[{ic[0]:.2f}, {ic[1]:.2f}]** ton/ha")
            if p_val < 0.05:
                st.success(
                    "Se rechaza H₀: hay evidencia estadística de que el Tratamiento B "
                    "incrementa el rendimiento promedio. Se recomienda autorizar la "
                    "inversión para escalarlo al invernadero."
                )
            else:
                st.error(
                    "No se rechaza H₀: no hay evidencia suficiente de que el Tratamiento B "
                    "mejore el rendimiento. No se recomienda escalar la inversión."
                )
        else:
            st.info("Selecciona ambos tratamientos para ejecutar la prueba t.")

st.write("")
with st.expander("Ver tabla de datos (SQL → pandas)"):
    st.dataframe(df.drop(columns="residuo"), width="stretch")

st.markdown('<div class="footer">INC-2505 · Análisis y Exploración de Datos · Proyecto Integrador U4</div>', unsafe_allow_html=True)