import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from scipy.stats import poisson
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================================================
# CONFIGURACION
# =========================================================

st.set_page_config(
    page_title="Valoracion de Activos",
    layout="wide"
)

st.title("Análisis de portaforlio")

st.markdown("""
            
-Angie Loango
            
-Suleily Moreno
            
-Carolina Umbarila
            
Aplicacion desarrollada para comparar:

- Movimiento Browniano Geometrico (GBM)
- Heston
- Saltos de Merton
- Opciones Reales con Arbol Binomial
""")

TICKERS = ["TSLA", "NVDA", "BA", "UNH"]

DAYS_TO_PREDICT = 21
SIMULATIONS = 10000

# =========================================================
# DESCARGA DE DATOS
# =========================================================

@st.cache_data(ttl=3600)
def download_data(ticker):

    try:

        data = yf.download(
            ticker,
            period="2y",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or data.empty:

            st.warning(
                f"No se pudieron descargar datos para {ticker}"
            )

            return None

        if isinstance(data.columns, pd.MultiIndex):

            data.columns = data.columns.get_level_values(0)

        if "Close" not in data.columns:

            st.warning(
                f"No existe columna Close para {ticker}"
            )

            return None

        data = data[["Close"]].dropna()

        if len(data) < 30:

            st.warning(
                f"No hay suficientes datos para {ticker}"
            )

            return None

        data["Returns"] = np.log(
            data["Close"] / data["Close"].shift(1)
        )

        data = data.dropna()

        return data

    except Exception as e:

        st.error(
            f"Error descargando {ticker}: {e}"
        )

        return None

# =========================================================
# GBM
# =========================================================

def simulate_gbm(
    S0,
    mu,
    sigma,
    days=DAYS_TO_PREDICT,
    sims=SIMULATIONS
):

    dt = 1 / 252

    paths = np.zeros((days, sims))

    paths[0] = S0

    for t in range(1, days):

        z = np.random.normal(size=sims)

        paths[t] = paths[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * z
        )

    return paths


# =========================================================
# HESTON
# =========================================================

def simulate_heston(
    S0,
    mu,
    v0,
    kappa=2,
    theta=0.04,
    xi=0.3,
    rho=-0.5,
    days=DAYS_TO_PREDICT,
    sims=SIMULATIONS
):

    dt = 1 / 252

    prices = np.zeros((days, sims))
    variances = np.zeros((days, sims))

    prices[0] = S0
    variances[0] = v0

    for t in range(1, days):

        z1 = np.random.normal(size=sims)

        z2 = (
            rho * z1
            + np.sqrt(1 - rho**2)
            * np.random.normal(size=sims)
        )

        variances[t] = np.abs(
            variances[t - 1]
            + kappa * (theta - variances[t - 1]) * dt
            + xi
            * np.sqrt(np.maximum(variances[t - 1], 0))
            * np.sqrt(dt)
            * z2
        )

        prices[t] = prices[t - 1] * np.exp(
            (mu - 0.5 * variances[t - 1]) * dt
            + np.sqrt(np.maximum(variances[t - 1], 0))
            * np.sqrt(dt)
            * z1
        )

    return prices


# =========================================================
# MERTON MEJORADO
# =========================================================

def simulate_merton(
    S0,
    mu,
    sigma,
    lam=2.5,
    mu_j=0,
    sigma_j=0.08,
    days=DAYS_TO_PREDICT,
    sims=SIMULATIONS
):

    dt = 1 / 252

    paths = np.zeros((days, sims))

    paths[0] = S0

    for t in range(1, days):

        z = np.random.normal(size=sims)

        jumps = poisson.rvs(
            mu=lam * dt,
            size=sims
        )

        jump_sizes = np.random.normal(
            mu_j,
            sigma_j,
            size=sims
        ) * jumps

        paths[t] = paths[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * z
            + jump_sizes
        )

    return paths


# =========================================================
# RMSE
# =========================================================

def evaluate_model(real_prices, simulated):

    real_prices = np.asarray(
        real_prices,
        dtype=float
    ).ravel()

    pred = simulated.mean(axis=1)

    pred = pred[: len(real_prices)]

    rmse = np.sqrt(
        mean_squared_error(real_prices, pred)
    )

    return rmse, pred


# =========================================================
# GRAFICOS
# =========================================================

def plot_simulation(title, simulation):

    st.subheader(title)

    fig = go.Figure()

    for i in range(min(100, simulation.shape[1])):

        fig.add_trace(
            go.Scatter(
                y=simulation[:, i],
                mode="lines",
                opacity=0.25,
                showlegend=False,
            )
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# OPCIONES REALES
# =========================================================

def real_option_binomial(
    S,
    K,
    T,
    r,
    sigma,
    steps=5
):

    dt = T / steps

    u = np.exp(sigma * np.sqrt(dt))

    d = 1 / u

    p = (
        np.exp(r * dt) - d
    ) / (u - d)

    stock_tree = np.zeros(
        (steps + 1, steps + 1)
    )

    for i in range(steps + 1):

        for j in range(i + 1):

            stock_tree[j, i] = (
                S
                * (u ** (i - j))
                * (d ** j)
            )

    option_tree = np.zeros(
        (steps + 1, steps + 1)
    )

    for j in range(steps + 1):

        option_tree[j, steps] = max(
            stock_tree[j, steps] - K,
            0
        )

    for i in range(steps - 1, -1, -1):

        for j in range(i + 1):

            hold = np.exp(-r * dt) * (
                p * option_tree[j, i + 1]
                + (1 - p)
                * option_tree[j + 1, i + 1]
            )

            exercise = max(
                stock_tree[j, i] - K,
                0
            )

            option_tree[j, i] = max(
                hold,
                exercise
            )

    return option_tree[0, 0], stock_tree


# =========================================================
# ANALISIS PRINCIPAL
# =========================================================

results = []

for ticker in TICKERS:

    st.header(f"📊 Accion: {ticker}")

    data = download_data(ticker)

    train = data.iloc[:-DAYS_TO_PREDICT]

    test = data.iloc[-DAYS_TO_PREDICT:]

    S0 = float(train["Close"].iloc[-1])

    mu = float(
        train["Returns"].mean() * 252
    )

    sigma = float(
        train["Returns"].std()
        * np.sqrt(252)
    )

    v0 = sigma**2

    # =====================================================
    # MERTON MEJORADO
    # =====================================================

    threshold = (
        1.5
        * train["Returns"].std()
    )

    jumps = train[
        np.abs(train["Returns"]) > threshold
    ]

    lam = max(
        len(jumps)
        / len(train)
        * 252,
        0.5
    )

    mu_j = jumps["Returns"].mean()

    sigma_j = max(
        jumps["Returns"].std(),
        0.03
    )

    # =====================================================
    # SIMULACIONES
    # =====================================================

    gbm = simulate_gbm(
        S0,
        mu,
        sigma
    )

    heston = simulate_heston(
        S0,
        mu,
        v0
    )

    merton = simulate_merton(
        S0,
        mu,
        sigma * 1.5,
        lam=lam,
        mu_j=mu_j,
        sigma_j=sigma_j
    )

    # =====================================================
    # BACKTESTING
    # =====================================================

    rmse_gbm, pred_gbm = evaluate_model(
        test["Close"].values,
        gbm
    )

    rmse_heston, pred_heston = evaluate_model(
        test["Close"].values,
        heston
    )

    rmse_merton, pred_merton = evaluate_model(
        test["Close"].values,
        merton
    )

    rmse_table = pd.DataFrame({
        "Modelo": [
            "GBM",
            "Heston",
            "Merton"
        ],

        "RMSE": [
            rmse_gbm,
            rmse_heston,
            rmse_merton
        ]
    })

    best_model = rmse_table.loc[
        rmse_table["RMSE"].idxmin(),
        "Modelo"
    ]

    model_map = {
        "GBM": gbm,
        "Heston": heston,
        "Merton": merton
    }

    final_prices = model_map[
        best_model
    ][-1]

    p5 = np.percentile(final_prices, 5)

    p50 = np.percentile(final_prices, 50)

    p95 = np.percentile(final_prices, 95)

    results.append({

        "Accion": ticker,

        "Modelo ganador": best_model,

        "RMSE GBM": round(rmse_gbm, 2),

        "RMSE Heston": round(rmse_heston, 2),

        "RMSE Merton": round(rmse_merton, 2),

        "Precio esperado": round(p50, 2),

        "Rango 5%": round(p5, 2),

        "Rango 95%": round(p95, 2),
    })

    # =====================================================
    # GRAFICOS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Precio Historico")

        fig = px.line(
            data,
            y="Close"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Retornos Logaritmicos"
        )

        fig2 = px.line(
            data,
            y="Returns"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    plot_simulation(
        "Simulacion GBM",
        gbm
    )

    plot_simulation(
        "Simulacion Heston",
        heston
    )

    plot_simulation(
        "Simulacion Merton",
        merton
    )

    st.subheader("RMSE por Modelo")

    st.dataframe(rmse_table)

    st.success(
        f"Bajo el modelo {best_model}, "
        f"el precio podria ubicarse entre "
        f"{round(p5,2)} y {round(p95,2)}"
    )


# =========================================================
# RESULTADOS FINALES
# =========================================================

st.header("📋 Tabla Final")

results_df = pd.DataFrame(results)

st.dataframe(
    results_df,
    use_container_width=True
)

# =========================================================
# RECOMENDACION
# =========================================================

st.header("💼 Recomendacion Ejecutiva")

best_asset = results_df.sort_values(
    "Precio esperado",
    ascending=False
).iloc[0]

lowest_risk = (
    results_df["Rango 95%"]
    - results_df["Rango 5%"]
).idxmin()

highest_risk = (
    results_df["Rango 95%"]
    - results_df["Rango 5%"]
).idxmax()

st.write(
    f"✅ Mejor expectativa: "
    f"{best_asset['Accion']}"
)

st.write(
    f"✅ Menor incertidumbre: "
    f"{results_df.iloc[lowest_risk]['Accion']}"
)

st.write(
    f"⚠️ Mayor riesgo: "
    f"{results_df.iloc[highest_risk]['Accion']}"
)

st.warning("""
Los modelos no garantizan resultados futuros.
Solo representan escenarios probabilisticos.
""")


# =========================================================
# ANALISIS DE PORTAFOLIO
# =========================================================

st.header("📈 Analisis de Portafolio")

st.markdown("""
Distribucion optima sugerida del portafolio basada
en el precio esperado y el nivel de riesgo de cada accion.
""")

# ---------------------------------------------------------
# PESOS BASADOS EN PRECIO ESPERADO AJUSTADO POR RIESGO
# ---------------------------------------------------------

results_df["Rango"] = (
    results_df["Rango 95%"] - results_df["Rango 5%"]
)

results_df["Score"] = (
    results_df["Precio esperado"]
    / results_df["Rango"].replace(0, np.nan)
)

score_sum = results_df["Score"].sum()

results_df["Peso (%)"] = (
    results_df["Score"] / score_sum * 100
).round(2)

# ---------------------------------------------------------
# GRAFICO DE PASTEL
# ---------------------------------------------------------

st.subheader("🥧 Distribucion del Portafolio")

fig_pie = px.pie(
    results_df,
    names="Accion",
    values="Peso (%)",
    color_discrete_sequence=px.colors.qualitative.Bold,
    hole=0.35,
)

fig_pie.update_traces(
    textposition="outside",
    textinfo="label+percent",
    pull=[0.05] * len(results_df),
)

fig_pie.update_layout(
    showlegend=True,
    legend=dict(orientation="h", y=-0.15),
    margin=dict(t=40, b=60, l=20, r=20),
)

st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------
# TABLA DE PESOS
# ---------------------------------------------------------

st.subheader("📋 Pesos por Accion")

portfolio_table = results_df[[
    "Accion",
    "Modelo ganador",
    "Precio esperado",
    "Rango 5%",
    "Rango 95%",
    "Peso (%)"
]].copy()

st.dataframe(
    portfolio_table,
    use_container_width=True
)

# ---------------------------------------------------------
# QUE HACER
# ---------------------------------------------------------

st.subheader("✅ Que hacer con este portafolio")

mejor = results_df.sort_values(
    "Peso (%)", ascending=False
).iloc[0]["Accion"]

menor_riesgo = results_df.loc[
    results_df["Rango"].idxmin(), "Accion"
]

st.success(f"""
**Recomendaciones de accion:**

- **Sobreponderar {mejor}**: tiene el mejor ratio
  retorno esperado / riesgo del portafolio.
- **Mantener {menor_riesgo}**: presenta el rango
  de incertidumbre mas acotado; ideal como ancla
  defensiva del portafolio.
- Rebalancear mensualmente segun los precios
  observados versus los rangos simulados.
- Usar los percentiles 5% y 95% como bandas de
  alerta para stop-loss y toma de ganancias.
- Diversificar entre los modelos ganadores:
  si distintas acciones tienen distintos modelos
  optimos (GBM / Heston / Merton), el portafolio
  captura diferentes regimenes de mercado.
""")

# ---------------------------------------------------------
# QUE NO HACER
# ---------------------------------------------------------

st.subheader("🚫 Que NO hacer con este portafolio")

mayor_riesgo = results_df.loc[
    results_df["Rango"].idxmax(), "Accion"
]

menor_peso = results_df.sort_values(
    "Peso (%)"
).iloc[0]["Accion"]

st.error(f"""
**Errores a evitar:**

- ❌ **No concentrar todo el capital en {mayor_riesgo}**:
  tiene el rango de incertidumbre mas amplio;
  una posicion excesiva eleva el riesgo total
  del portafolio de forma desproporcionada.
- ❌ **No ignorar los percentiles 5%**: representan
  escenarios adversos reales, no solo estadisticos.
  Ignorarlos lleva a subcapitalizar coberturas.
- ❌ **No sobreponderar {menor_peso} solo por tener
  menor peso sugerido**: el peso bajo refleja
  un ratio retorno/riesgo menos favorable segun
  los modelos, no necesariamente mal desempeno absoluto.
- ❌ **No tomar los precios esperados como garantia**:
  son medianas de simulaciones probabilisticas,
  no predicciones deterministicas.
- ❌ **No rebalancear con demasiada frecuencia**:
  los costos de transaccion y el deslizamiento
  pueden erosionar el retorno ajustado al riesgo.
- ❌ **No usar estos modelos como unica fuente
  de decision**: complementar siempre con analisis
  fundamental y contexto macroeconomico.
""")


# =========================================================
# OPCIONES REALES Y ARBOL BINOMIAL (VERSION CORREGIDA)
# =========================================================

st.header("🏗️ Opciones Reales y Árbol Binomial")

st.markdown("""
Esta sección evalúa un proyecto tecnológico
mediante Opciones Reales usando decisiones
estratégicas más realistas y balanceadas.
""")

# =========================================================
# INPUTS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    investment = st.number_input(
        "Inversión Inicial",
        value=100000
    )

    # MÁS REALISTA
    project_value = st.number_input(
        "Valor esperado del proyecto",
        value=540000
    )

    volatility = st.slider(
    "Volatilidad",
    0.1,
    1.0,
    0.81
)

with col2:

    risk_free = st.slider(
        "Tasa libre de riesgo",
        0.01,
        0.2,
        0.05
    )

    years = 2

    steps = st.slider(
        "Pasos del árbol",
        3,
        10,
        4
    )

# =========================================================
# PARAMETROS ESTRATEGICOS MEJORADOS
# =========================================================

st.subheader("⚙️ Configuración Estratégica")

EXPAND_THRESHOLD = st.slider(
    "Umbral para EXPANDIR",
    min_value=1.8,
    max_value=4.0,
    value=3.9,
    step=0.1
)

EXECUTE_THRESHOLD = st.slider(
    "Umbral para EJECUTAR",
    min_value=1.1,
    max_value=2.0,
    value=1.60,
    step=0.05
)

WAIT_THRESHOLD = st.slider(
    "Umbral para ESPERAR",
    min_value=0.8,
    max_value=1.2,
    value=1.05,
    step=0.05
)

REDUCE_THRESHOLD = st.slider(
    "Umbral para REDUCIR",
    min_value=0.4,
    max_value=0.9,
    value=0.4,
    step=0.05
)

EXPANSION_GAIN = st.slider(
    "Ganancia por expansión",
    0.05,
    0.40,
    0.12
)

max_expansion_budget = st.number_input(
    "Presupuesto máximo expansión",
    min_value=10000,
    max_value=5000000,
    value=50000,
    step=10000
)

# =========================================================
# MODELO BINOMIAL
# =========================================================

option_value, stock_tree = real_option_binomial(
    S=project_value,
    K=investment,
    T=years,
    r=risk_free,
    sigma=volatility,
    steps=steps
)

npv = project_value - investment

metric1, metric2 = st.columns(2)

metric1.metric(
    "VAN Tradicional",
    f"${npv:,.0f}"
)

metric2.metric(
    "Valor Opción Real",
    f"${option_value:,.0f}"
)

if option_value > npv:

    st.success("""
La flexibilidad estratégica agrega valor.
""")

else:

    st.warning("""
La opción real no agrega valor significativo.
""")

# =========================================================
# GRAFICO DEL ARBOL
# =========================================================

st.subheader("🌳 Árbol Binomial de Decisiones")

fig, ax = plt.subplots(figsize=(15, 9))

dt_plot = years / steps

u_plot = np.exp(
    volatility * np.sqrt(dt_plot)
)

d_plot = 1 / u_plot

p_plot = (
    np.exp(risk_free * dt_plot) - d_plot
) / (u_plot - d_plot)

option_tree_plot = np.zeros(
    (steps + 1, steps + 1)
)

decision_tree = np.full(
    (steps + 1, steps + 1),
    "",
    dtype=object
)

# =========================================================
# NODOS TERMINALES
# =========================================================

for j in range(steps + 1):

    node_value = stock_tree[j, steps]

    payoff = max(
        node_value - investment,
        0
    )

    ratio = node_value / investment

    # =====================================================
    # NUEVA LOGICA ESTRATEGICA
    # =====================================================

    if ratio >= EXPAND_THRESHOLD:

        expansion_bonus = min(
            node_value * EXPANSION_GAIN,
            max_expansion_budget * 0.25
        )

        payoff += expansion_bonus

        decision_tree[j, steps] = "EXPANDIR"

    elif ratio >= EXECUTE_THRESHOLD:

        payoff *= 1.0

        decision_tree[j, steps] = "EJECUTAR"

    elif ratio >= WAIT_THRESHOLD:

        payoff *= 0.92

        decision_tree[j, steps] = "ESPERAR"

    elif ratio >= REDUCE_THRESHOLD:

        payoff *= 0.55

        decision_tree[j, steps] = "REDUCIR"

    else:

        payoff = 0

        decision_tree[j, steps] = "ABANDONAR"

    option_tree_plot[j, steps] = payoff

# =========================================================
# BACKWARD INDUCTION
# =========================================================

for i in range(steps - 1, -1, -1):

    for j in range(i + 1):

        node_value = stock_tree[j, i]

        ratio = node_value / investment

        hold = np.exp(-risk_free * dt_plot) * (

            p_plot * option_tree_plot[j, i + 1]
            + (1 - p_plot)
            * option_tree_plot[j + 1, i + 1]
        )

        exercise = max(
            node_value - investment,
            0
        )

        # =================================================
        # NUEVA LOGICA ESTRATEGICA
        # =================================================

        if ratio >= EXPAND_THRESHOLD:

            expansion_bonus = min(
                node_value * EXPANSION_GAIN,
                max_expansion_budget * 0.25
            )

            expanded_value = (
                exercise + expansion_bonus
            )

            option_tree_plot[j, i] = max(
                hold,
                expanded_value
            )

            decision_tree[j, i] = "EXPANDIR"

        elif ratio >= EXECUTE_THRESHOLD:

            option_tree_plot[j, i] = max(
                hold,
                exercise
            )

            if exercise > hold:

                decision_tree[j, i] = "EJECUTAR"

            else:

                decision_tree[j, i] = "ESPERAR"

        elif ratio >= WAIT_THRESHOLD:

            option_tree_plot[j, i] = hold * 0.92

            decision_tree[j, i] = "ESPERAR"

        elif ratio >= REDUCE_THRESHOLD:

            option_tree_plot[j, i] = hold * 0.55

            decision_tree[j, i] = "REDUCIR"

        else:

            option_tree_plot[j, i] = 0

            decision_tree[j, i] = "ABANDONAR"

# =========================================================
# COLORES
# =========================================================

color_map = {

    "EXPANDIR":  "#9b59b6",
    "EJECUTAR":  "#2ecc71",
    "ESPERAR":   "#3498db",
    "REDUCIR":   "#f39c12",
    "ABANDONAR": "#e74c3c",
}

# =========================================================
# DIBUJAR ARBOL
# =========================================================

for i in range(steps + 1):

    for j in range(i + 1):

        x = i

        y = -j + i / 2

        decision = decision_tree[j, i]

        color = color_map.get(
            decision,
            "#95a5a6"
        )

        option_type_map = {
            "EXPANDIR":  "Opción de crecimiento",
            "EJECUTAR":  "Opción de inversión",
            "ESPERAR":   "Opción de diferimiento",
            "REDUCIR":   "Opción de contracción",
            "ABANDONAR": "Opción de abandono",
        }

        option_type_label = option_type_map.get(
            decision,
            ""
        )

        ax.scatter(
            x,
            y,
            s=900,
            color=color,
            edgecolors="white",
            linewidths=2,
            zorder=3
        )

        ax.text(
            x,
            y + 0.45,
            f"${stock_tree[j, i]:,.0f}",
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold"
        )

        ax.text(
            x,
            y,
            decision,
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold"
        )

        ax.text(
            x,
            y - 0.38,
            option_type_label,
            ha="center",
            va="center",
            fontsize=5,
            color=color,
            fontstyle="italic"
        )

        if i < steps:

            ax.plot(
                [x, x + 1],
                [y, y + 0.5],
                color="#bdc3c7",
                linewidth=1.5,
                zorder=1
            )

            ax.plot(
                [x, x + 1],
                [y, y - 0.5],
                color="#bdc3c7",
                linewidth=1.5,
                zorder=1
            )

# =========================================================
# TITULO
# =========================================================

ax.set_title(
    "Árbol Binomial de Decisiones Estratégicas",
    fontsize=16,
    fontweight="bold",
    pad=20
)

ax.axis("off")

# =========================================================
# LEYENDA
# =========================================================

legend_elements = [

    Patch(
        facecolor="#9b59b6",
        label="Expandir negocio"
    ),

    Patch(
        facecolor="#2ecc71",
        label="Ejecutar proyecto"
    ),

    Patch(
        facecolor="#3498db",
        label="Esperar"
    ),

    Patch(
        facecolor="#f39c12",
        label="Reducir operación"
    ),

    Patch(
        facecolor="#e74c3c",
        label="Abandonar"
    ),
]

ax.legend(
    handles=legend_elements,
    loc="lower right",
    fontsize=10,
    framealpha=0.95
)

st.pyplot(fig)

# =========================================================
# METRICAS ESTRATEGICAS
# =========================================================

st.subheader("📊 Análisis Estratégico")

expand_count = np.sum(
    decision_tree == "EXPANDIR"
)

execute_count = np.sum(
    decision_tree == "EJECUTAR"
)

wait_count = np.sum(
    decision_tree == "ESPERAR"
)

reduce_count = np.sum(
    decision_tree == "REDUCIR"
)

abandon_count = np.sum(
    decision_tree == "ABANDONAR"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Expansiones",
    int(expand_count)
)

col2.metric(
    "Ejecuciones",
    int(execute_count)
)

col3.metric(
    "Esperas",
    int(wait_count)
)

col1, col2 = st.columns(2)

col1.metric(
    "Reducciones",
    int(reduce_count)
)

col2.metric(
    "Abandonos",
    int(abandon_count)
)

# =========================================================
# DECISION INICIAL
# =========================================================

st.markdown(
    "### 📌 Decisión óptima inicial"
)

decision_raiz = decision_tree[0, 0]

if decision_raiz == "EXPANDIR":

    st.success("""
Se recomienda expandir el proyecto.

Existe suficiente upside estratégico
para justificar crecimiento agresivo.
""")

elif decision_raiz == "EJECUTAR":

    st.info("""
Se recomienda ejecutar el proyecto.

El proyecto tiene valor positivo
y condiciones financieras favorables.
""")

elif decision_raiz == "ESPERAR":

    st.warning("""
Se recomienda esperar.

Existe incertidumbre importante y
conviene esperar nueva información.
""")

elif decision_raiz == "REDUCIR":

    st.warning("""
Se recomienda reducir exposición.

El proyecto requiere ajustes
antes de expandirse.
""")

else:

    st.error("""
Se recomienda abandonar el proyecto.

Los escenarios esperados destruyen valor.
""")

# =========================================================
# ESTRATEGIA DE NEGOCIO
# =========================================================

st.subheader("📌 Estrategia de Negocio Recomendada")

if option_value > npv and volatility > 0.6:

    st.success("""
### Estrategia: Esperar y Expandirse Gradualmente

El proyecto presenta alta incertidumbre, pero tambien
alto valor estrategico.

#### Recomendacion ejecutiva:
- No invertir todo el capital inmediatamente.
- Implementar una entrada progresiva al mercado.
- Priorizar fases piloto y pruebas de mercado.
- Mantener flexibilidad para expandirse rapidamente
  si el escenario favorable se materializa.

#### Justificacion financiera:
La alta volatilidad aumenta el valor de la opcion real,
ya que permite aprovechar escenarios positivos mientras
se limita parcialmente el riesgo de perdida.

#### Perfil recomendado:
- Empresas tecnologicas
- IA
- Blockchain
- Startups de crecimiento acelerado
""")

elif option_value > npv and volatility <= 0.6:

    st.info("""
### Estrategia: Inversion Inmediata

El proyecto tiene valor positivo tanto en VAN tradicional
como en Opciones Reales.

#### Recomendacion ejecutiva:
- Ejecutar la inversion en el corto plazo.
- Aprovechar la estabilidad relativa del proyecto.
- Priorizar eficiencia operativa y escalamiento.

#### Justificacion financiera:
La volatilidad moderada reduce el beneficio de esperar,
por lo que capturar flujos tempranos genera mayor valor.

#### Perfil recomendado:
- Expansiones corporativas
- Infraestructura
- Proyectos de software empresarial
""")

elif option_value < npv and volatility > 0.7:

    st.warning("""
### Estrategia: Esperar Informacion Adicional

El proyecto presenta demasiada incertidumbre
respecto al valor esperado.

#### Recomendacion ejecutiva:
- Posponer la inversion temporalmente.
- Obtener mas informacion del mercado.
- Esperar reduccion de volatilidad.
- Monitorear tasas, demanda y competencia.

#### Justificacion financiera:
La incertidumbre extrema incrementa el riesgo de
escenarios negativos significativos.

#### Riesgos principales:
- Sobrevaloracion del proyecto
- Cambios regulatorios
- Caidas abruptas de demanda
""")

else:

    st.error("""
### Estrategia: No Ejecutar el Proyecto

El proyecto no genera suficiente valor financiero
ni estrategico bajo las condiciones actuales.

#### Recomendacion ejecutiva:
- Reestructurar el proyecto.
- Reducir costos iniciales.
- Buscar alianzas estrategicas.
- Evaluar mercados alternativos.

#### Justificacion financiera:
Ni el VAN ni la opcion real compensan adecuadamente
el riesgo asumido.

#### Conclusion:
Actualmente el proyecto destruye valor.
""")
