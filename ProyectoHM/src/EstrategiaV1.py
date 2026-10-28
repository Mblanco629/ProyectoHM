import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from .CargaDatos import calcular_indicadores, descargar_datos

# =========================================================
# GENERACIÓN DE SEÑALES
# =========================================================
def generar_senal(
    sma_50, sma_200, prob_subida, rsi,
    rsi_entrada: float = 30.0, rsi_salida: float = 70.0,
    probabilidad_minima: float = 0.50,
    probabilidad_maxima_venta: float = 0.30,
):
    """Combina la tendencia, la IA y el RSI para generar una recomendación."""

#• Al inicio de los datos históricos, las medias móviles (como la sma_200) no tienen suficientes 
# velas previas para calcularse y devuelven NaN. Si intentas operar con un NaN, 
# el código fallaría. Ante la duda, la función adopta una postura segura y devuelve MANTENER.

    if (
        pd.isna(sma_50) or pd.isna(sma_200)
        or pd.isna(prob_subida) or pd.isna(rsi)
    ):
        return "MANTENER (HOLD)"

# • Qué hace: Si el RSI está en su rango normal (entre 30 y 70), significa que el mercado está en una zona de equilibrio o consolidación.
# • Resultado: No se toma ninguna acción operativa. Directamente devuelve MANTENER, ahorrando tiempo de cómputo al no evaluar el resto de condiciones.

    if rsi_entrada <= rsi <= rsi_salida:
        return "MANTENER (HOLD)"

# Para que la función te diga que compres, se tienen que cumplir las 3 condiciones estrictamente al mismo tiempo:
# 1. RSI < 30: El precio ha caído rápido y el activo está "barato" o en sobreventa.
# 2. SMA 50 > SMA 200: Existe una tendencia alcista de fondo (conocido como Cruce de Oro si ocurre recientemente). Esto asegura que no estás comprando un activo en caída libre infinita.
# 3. Probabilidad > 50%: Tu modelo de IA respalda la operación confirmando que la tendencia matemática apunta a una subida.

    if (
        rsi < rsi_entrada
        and sma_50 > sma_200
        and prob_subida > probabilidad_minima
    ):
        return "COMPRA (BUY)"

# Para que la función te diga que vendas, exige la tormenta perfecta bajista:
# 1. RSI > 70: El precio ha subido demasiado rápido y está "caro" o sobrecomprado.
# 2. SMA 50 < SMA 200: La tendencia general es bajista (Cruce de la Muerte), lo que sugiere presiones de venta a largo plazo.
# 3. Probabilidad < 30%: La IA certifica que el escenario de subida es muy poco probable, apoyando la decisión de salir del activo o abrir una posición en corto.

    if (
        rsi > rsi_salida
        and sma_50 < sma_200
        and prob_subida < probabilidad_maxima_venta
    ):
        return "VENTA (SELL)"
    return "MANTENER (HOLD)"

# =========================================================
# BACKTESTING (Simulación de Rentabilidad)
# =========================================================
def ejecutar_backtest(df):
    """Calcula el rendimiento de la estrategia frente a simplemente retener el activo."""
    print("Ejecutando simulación de backtesting...")
    df = df.copy()
    
    # 1. Retorno diario del mercado (Buy & Hold)
    # pct_change() calcula qué porcentaje subió o bajó el precio respecto a ayer
    df['Retorno_Mercado'] = df['Close'].pct_change()
    
    # 2. Retorno de nuestra estrategia
    # IMPORTANTE: shift(1) desplaza la señal un día. Si el algoritmo decide "Comprar" 
    # hoy con los datos de cierre, la rentabilidad real se captura mañana. 
    # Olvidar el shift(1) causa "look-ahead bias" (ver el futuro trampa).
    df['Retorno_Estrategia'] = df['Senal_Trading'].shift(1) * df['Retorno_Mercado']
    
    # 3. Evolución del capital acumulado (Asumiendo que empezamos con capital = 1.0)
    df['Capital_Mercado'] = (1 + df['Retorno_Mercado']).cumprod()
    df['Capital_Estrategia'] = (1 + df['Retorno_Estrategia']).cumprod()
    
    return df.dropna()

# =========================================================
# EJECUTAR ESTRATEGIA IA (Función principal)
# =========================================================

def ejecutar_estrategia(
    ticker, fecha_inicio, fecha_fin,
    rsi_entrada: float = 30.0, rsi_salida: float = 70.0,
    probabilidad_minima: float = 0.50,
    probabilidad_maxima_venta: float = 0.30,
    periodo_media_corta: int = 50, periodo_media_larga: int = 200,
):
    """Entrena y evalúa el backtest de la estrategia para un activo.

    Descarga los datos OHLCV del ticker, calcula los indicadores técnicos y
    prepara como objetivo si el cierre de la siguiente sesión será superior al
    cierre actual. Divide los registros cronológicamente: el 80 % inicial se
    usa para entrenar un clasificador Random Forest y el 20 % final para
    generar probabilidades, recomendaciones y resultados del backtest. No se
    mezclan aleatoriamente las observaciones.

    Las señales se basan en las medias móviles, el RSI y la probabilidad
    estimada de subida. El backtest compara la estrategia con mantener el
    activo y retrasa un día la señal antes de aplicar los retornos, para evitar
    usar una señal calculada al cierre como si se hubiera podido ejecutar
    antes de ese cierre. También genera un gráfico interactivo con precios,
    indicadores, señales y evolución del capital.

    Args:
        ticker: Símbolo del activo que se descarga desde Yahoo Finance, por
            ejemplo ``"SAN.MC"``.
        fecha_inicio: Fecha inicial del histórico, en el formato aceptado por
            ``yfinance.download``.
        fecha_fin: Fecha final del histórico, en el formato aceptado por
            ``yfinance.download``.
        rsi_entrada: Umbral inferior del RSI para considerar una compra.
            Valor predeterminado: ``30.0``.
        rsi_salida: Umbral superior del RSI para considerar una venta.
            Valor predeterminado: ``70.0``.
        probabilidad_minima: Probabilidad mínima de subida para habilitar una
            compra, expresada como fracción entre 0 y 1 (por ejemplo, ``0.50``
            equivale a 50 %).
        probabilidad_maxima_venta: Probabilidad máxima de subida para habilitar
            una venta, expresada como fracción entre 0 y 1 (por ejemplo,
            ``0.30`` equivale a 30 %).
        periodo_media_corta: Ventana de cálculo de la media móvil corta, en
            sesiones. Debe ser positiva y menor que ``periodo_media_larga``.
        periodo_media_larga: Ventana de cálculo de la media móvil larga, en
            sesiones. Debe ser positiva y mayor que ``periodo_media_corta``.

    Returns:
        Una tupla de cinco elementos, en este orden:

        * HTML del gráfico interactivo de Plotly, sin las etiquetas
          ``<html>`` ni ``<body>``.
        * Rentabilidad porcentual del mercado (estrategia Buy & Hold).
        * Rentabilidad porcentual acumulada del backtest de la estrategia.
        * Probabilidad estimada de subida para el último registro del período
          de prueba, como fracción entre 0 y 1.
        * Última señal de trading como entero: ``1`` para compra, ``-1`` para
          venta y ``0`` para mantener.

    Raises:
        ValueError: Si los períodos de medias móviles no son positivos o su
            orden es incorrecto, si no hay suficientes registros para separar
            entrenamiento y prueba, si el entrenamiento contiene una sola
            clase objetivo o si el backtest no produce resultados.
        ValueError: También puede propagarse desde la descarga de datos cuando
            no hay histórico disponible para el ticker y el período indicados.
    """
    df_crudo = descargar_datos(ticker, fecha_inicio, fecha_fin)
    if periodo_media_corta < 1 or periodo_media_larga < 1:
        raise ValueError("Los períodos de las medias móviles deben ser positivos.")
    if periodo_media_corta >= periodo_media_larga:
        raise ValueError(
            "El período de la media corta debe ser menor que el de la media larga."
        )
    df_indicadores = calcular_indicadores(
        df_crudo, periodo_media_corta, periodo_media_larga
    )

    print("Preparando datos para la Inteligencia Artificial...")
    df_indicadores['Target'] = (
        df_indicadores['Close'].shift(-1) > df_indicadores['Close']
    ).astype(int)
    df_ml = df_indicadores.dropna().copy()

    features = ['SMA_corta', 'SMA_larga', 'Volatilidad', 'RSI']
    train_size = int(len(df_ml) * 0.8)
    if train_size < 2 or len(df_ml) - train_size < 2:
        raise ValueError(
            "El período es demasiado corto para entrenar y probar la estrategia. "
            "Selecciona un rango de fechas más amplio."
        )

    X_train = df_ml[features].iloc[:train_size]
    y_train = df_ml['Target'].iloc[:train_size]
    X_test = df_ml[features].iloc[train_size:]
    if y_train.nunique() < 2:
        raise ValueError(
            "Los datos de entrenamiento no contienen subidas y bajadas suficientes. "
            "Prueba con otro período."
        )

    print("Entrenando el modelo Random Forest...")
    modelo_ia = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    modelo_ia.fit(X_train, y_train)

    df_ml['Prob_Subida'] = np.nan
    df_ml.loc[X_test.index, 'Prob_Subida'] = modelo_ia.predict_proba(X_test)[:, 1]

    print("Generando señales de trading en base a las predicciones...")

    df_ml['Recomendacion'] = df_ml.apply(
        lambda fila: generar_senal(
            fila['SMA_corta'], fila['SMA_larga'],
            fila['Prob_Subida'], fila['RSI'], rsi_entrada, rsi_salida,
            probabilidad_minima, probabilidad_maxima_venta,
        ),
        axis=1,
    )
    df_ml['Senal_Trading'] = df_ml['Recomendacion'].map(
        {
            "COMPRA (BUY)": 1,
            "VENTA (SELL)": -1,
            "MANTENER (HOLD)": 0,
        }
    )

    df_prueba = df_ml.iloc[train_size:].copy()
    recomendacion_manana = df_prueba['Recomendacion'].iloc[-1]
    probabilidad_subida = float(df_prueba['Prob_Subida'].iloc[-1])
    ultima_senal_trading = int(df_prueba['Senal_Trading'].iloc[-1])
    print(f"Recomendación para mañana: {recomendacion_manana}")
    print(f"Probabilidad estimada de subida: {probabilidad_subida:.2%}")
    print(f"Último valor de Senal_Trading: {ultima_senal_trading}")
    df_resultados = ejecutar_backtest(df_prueba)
    if df_resultados.empty:
        raise ValueError(
            "No hay suficientes resultados para mostrar. "
            "Selecciona un rango de fechas más amplio."
        )

    grafico = graficar_resultados(
        df_resultados, ticker, recomendacion_manana,
        rsi_entrada, rsi_salida,
        periodo_media_corta, periodo_media_larga,
        probabilidad_minima, probabilidad_maxima_venta,
    )
    rentabilidad_mercado = (
        df_resultados['Capital_Mercado'].iloc[-1] - 1
    ) * 100
    rentabilidad_estrategia = (
        df_resultados['Capital_Estrategia'].iloc[-1] - 1
    ) * 100
    return (
        grafico, rentabilidad_mercado, rentabilidad_estrategia,
        probabilidad_subida, ultima_senal_trading,
    )


# =========================================================
# VISUALIZACIÓN DE RESULTADOS
# =========================================================
def graficar_resultados(
    df, ticker, recomendacion_manana,
    rsi_entrada: float = 30.0, rsi_salida: float = 70.0,
    periodo_media_corta: int = 50, periodo_media_larga: int = 200,
    probabilidad_minima: float = 0.50,
    probabilidad_maxima_venta: float = 0.30,
):
    """Genera un gráfico interactivo con señales e indicadores de la estrategia."""
    print("Generando gráficos...")
    figura = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.34, 0.16, 0.16, 0.16, 0.18],
        subplot_titles=(
            'Precio, señales y bandas Bollinger', 'RSI (14)',
            'Probabilidad de subida', 'Capital acumulado', 'Volumen',
        ),
    )

    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['Close'], mode='lines',
            name='Precio de cierre', line={'color': 'black'},
        ),
        row=1, col=1,
    )
    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['SMA_corta'], mode='lines',
            name=f'Media móvil corta ({periodo_media_corta})',
            line={'color': 'orange'},
        ),
        row=1, col=1,
    )
    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['SMA_larga'], mode='lines',
            name=f'Media móvil larga ({periodo_media_larga})',
            line={'color': 'purple'},
        ),
        row=1, col=1,
    )
    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['Bollinger_Inferior'], mode='lines',
            name='Banda Bollinger inferior',
            line={'color': 'rgba(72, 167, 255, 0.45)', 'width': 1},
        ),
        row=1, col=1,
    )
    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['Bollinger_Superior'], mode='lines',
            name='Banda Bollinger superior',
            line={'color': 'rgba(72, 167, 255, 0.45)', 'width': 1},
            fill='tonexty', fillcolor='rgba(72, 167, 255, 0.10)',
        ),
        row=1, col=1,
    )
    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['Bollinger_Media'], mode='lines',
            name='Media Bollinger (20)',
            line={'color': 'rgba(72, 167, 255, 0.8)', 'width': 1},
        ),
        row=1, col=1,
    )
    compras = df[df['Senal_Trading'] == 1]
    ventas = df[df['Senal_Trading'] == -1]
    figura.add_trace(
        go.Scatter(
            x=compras.index, y=compras['Close'], mode='markers',
            name='Compra (BUY)',
            marker={'symbol': 'triangle-up', 'color': 'green', 'size': 10},
        ),
        row=1, col=1,
    )
    figura.add_trace(
        go.Scatter(
            x=ventas.index, y=ventas['Close'], mode='markers',
            name='Venta (SELL)',
            marker={'symbol': 'triangle-down', 'color': 'red', 'size': 10},
        ),
        row=1, col=1,
    )
    figura.add_annotation(
        x=0.01, y=0.98, xref='x domain', yref='y domain',
        text=f'Recomendación para mañana: {recomendacion_manana}',
        showarrow=False, bgcolor='white', opacity=0.85,
    )

    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['RSI'], mode='lines', name='RSI (14)',
            line={'color': 'teal'},
        ),
        row=2, col=1,
    )
    figura.add_hline(
        y=rsi_entrada, line_dash='dash', line_color='green',
        annotation_text=f'Umbral de entrada ({rsi_entrada:g})',
        row=2, col=1,
    )
    figura.add_hline(
        y=rsi_salida, line_dash='dash', line_color='red',
        annotation_text=f'Umbral de salida ({rsi_salida:g})',
        row=2, col=1,
    )
    figura.update_yaxes(title_text='RSI', range=[0, 100], row=2, col=1)

    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['Prob_Subida'] * 100, mode='lines',
            name='Probabilidad de subida', line={'color': 'blue'},
        ),
        row=3, col=1,
    )
    figura.add_hline(
        y=probabilidad_minima * 100, line_dash='dash', line_color='green',
        annotation_text=f'Umbral de compra ({probabilidad_minima * 100:g}%)',
        row=3, col=1,
    )
    figura.add_hline(
        y=probabilidad_maxima_venta * 100, line_dash='dash',
        line_color='red',
        annotation_text=(
            f'Umbral de venta ({probabilidad_maxima_venta * 100:g}%)'
        ),
        row=3, col=1,
    )
    figura.update_yaxes(
        title_text='Prob. subida (%)', range=[0, 100], row=3, col=1
    )

    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['Capital_Mercado'], mode='lines',
            name='Mercado (Buy & Hold)',
            line={'color': 'gray'}, opacity=0.6,
        ),
        row=4, col=1,
    )
    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['Capital_Estrategia'], mode='lines',
            name='Estrategia IA', line={'color': 'blue', 'width': 2},
        ),
        row=4, col=1,
    )
    figura.add_trace(
        go.Bar(
            x=df.index, y=df['Volume'], name='Volumen',
            marker_color='rgba(72, 167, 255, 0.45)',
        ),
        row=5, col=1,
    )
    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['Volumen_Medio_20'], mode='lines',
            name='Volumen medio (20 sesiones)',
            line={'color': 'orange', 'width': 2},
        ),
        row=5, col=1,
    )

    figura.update_layout(
        title=f'Señales de trading y backtest ({ticker})',
        height=900, hovermode='x unified',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02},
        margin={'l': 60, 'r': 30, 't': 110, 'b': 50},
    )
    figura.update_yaxes(title_text='Precio', row=1, col=1)
    figura.update_yaxes(title_text='Multiplicador', row=4, col=1)
    figura.update_yaxes(title_text='Acciones', row=5, col=1)
    figura.update_xaxes(title_text='Fecha', row=5, col=1)
    figura.update_xaxes(showgrid=True)
    figura.update_yaxes(showgrid=True)

    # Resumen de texto en consola
    rent_mercado = (df['Capital_Mercado'].iloc[-1] - 1) * 100
    rent_estrategia = (df['Capital_Estrategia'].iloc[-1] - 1) * 100
    print(f"\n--- RESUMEN FINAL ---")
    print(f"Rentabilidad Mercado: {rent_mercado:.2f}%")
    print(f"Rentabilidad Estrategia: {rent_estrategia:.2f}%")

    return figura.to_html(
        full_html=False, include_plotlyjs='cdn',
        config={'responsive': True},
    )
