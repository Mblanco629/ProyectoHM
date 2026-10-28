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
    filtro_tendencia, filtro_momento, filtro_volatilidad, filtro_volumen,
):
    """Genera compra unicamente cuando se cumplen los cuatro filtros."""
    if all((
        filtro_tendencia, filtro_momento,
        filtro_volatilidad, filtro_volumen,
    )):
        return "COMPRA (BUY)"
    return "MANTENER (HOLD)"


# =========================================================
# BACKTESTING (Simulacion de Rentabilidad)
# =========================================================
def ejecutar_backtest(df):
    """Simula posiciones largas con stop, objetivo y salidas anticipadas."""
    print("Ejecutando simulacion de backtesting...")
    df = df.copy()
    df['Retorno_Mercado'] = df['Close'].pct_change()
    df['Retorno_Estrategia'] = 0.0
    df['Senal_Trading'] = 0
    df['Stop_Loss'] = np.nan
    df['Take_Profit'] = np.nan
    df['Precio_Salida'] = np.nan

    en_posicion = False
    entrada = riesgo = stop = objetivo = None
    sobrecompra_alcanzada = False

    for i in range(1, len(df)):
        anterior = df.iloc[i - 1]
        actual = df.iloc[i]

        if not en_posicion and bool(anterior['Senal_Entrada']):
            precio_entrada = float(anterior['Close'])
            atr = anterior['ATR']
            minimo_local = anterior['Minimo_Local']
            if pd.notna(atr) and atr > 0:
                niveles_stop = [precio_entrada - 1.5 * float(atr)]
                if pd.notna(minimo_local):
                    niveles_stop.append(
                        float(minimo_local) - 0.1 * float(atr)
                    )
                niveles_validos = [
                    nivel for nivel in niveles_stop
                    if 0 < nivel < precio_entrada
                ]
                if niveles_validos:
                    entrada = precio_entrada
                    stop = max(niveles_validos)
                    riesgo = entrada - stop
                    objetivo = entrada + 2 * riesgo
                    en_posicion = True
                    sobrecompra_alcanzada = False
                    df.loc[df.index[i - 1], 'Senal_Trading'] = 1
                    df.loc[df.index[i - 1], 'Stop_Loss'] = stop
                    df.loc[df.index[i - 1], 'Take_Profit'] = objetivo

        if not en_posicion:
            continue

        indice_actual = df.index[i]
        precio_cierre_previo = float(anterior['Close'])
        apertura = float(actual['Open'])
        minimo = float(actual['Low'])
        maximo = float(actual['High'])
        precio_salida = None

        df.loc[indice_actual, 'Stop_Loss'] = stop
        df.loc[indice_actual, 'Take_Profit'] = objetivo

        if apertura <= stop:
            precio_salida = apertura
        elif apertura >= objetivo:
            precio_salida = apertura
        elif minimo <= stop:
            precio_salida = stop
        elif maximo >= objetivo:
            precio_salida = objetivo
        else:
            if maximo >= entrada + riesgo:
                stop = entrada
            df.loc[indice_actual, 'Stop_Loss'] = stop
            if minimo <= stop:
                precio_salida = stop

        if precio_salida is None:
            rsi_actual = actual['RSI']
            rsi_previo = anterior['RSI']
            if pd.notna(rsi_actual) and rsi_actual >= 70:
                sobrecompra_alcanzada = True

            salida_rsi = (
                sobrecompra_alcanzada
                and pd.notna(rsi_actual)
                and pd.notna(rsi_previo)
                and rsi_actual < rsi_previo
            )
            salida_ema = (
                pd.notna(actual['EMA_20'])
                and actual['Close'] < actual['EMA_20']
            )
            if salida_rsi or salida_ema:
                precio_salida = float(actual['Close'])

        if precio_salida is not None:
            df.loc[indice_actual, 'Retorno_Estrategia'] = (
                precio_salida / precio_cierre_previo
            ) - 1
            df.loc[indice_actual, 'Senal_Trading'] = -1
            df.loc[indice_actual, 'Precio_Salida'] = precio_salida
            en_posicion = False
            entrada = riesgo = stop = objetivo = None
            sobrecompra_alcanzada = False
        else:
            df.loc[indice_actual, 'Retorno_Estrategia'] = (
                float(actual['Close']) / precio_cierre_previo
            ) - 1

    df['Capital_Mercado'] = (1 + df['Retorno_Mercado']).cumprod()
    df['Capital_Estrategia'] = (1 + df['Retorno_Estrategia']).cumprod()
    return df.dropna(subset=['Retorno_Mercado'])


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

    Las entradas requieren precio sobre EMA 20/50, RSI entre 50 y 65,
    expansion de Bollinger o ruptura de su banda superior, y ruptura del
    maximo previo con rango amplio y volumen superior a su media de 20 sesiones.
    El backtest aplica stop por minimo local o 1.5 ATR, objetivo 1:2,
    breakeven al alcanzar 1R y salidas por RSI o cierre bajo EMA 20.

    Args:
        ticker: Símbolo del activo que se descarga desde Yahoo Finance, por
            ejemplo ``"SAN.MC"``.
        fecha_inicio: Fecha inicial del histórico, en el formato aceptado por
            ``yfinance.download``.
        fecha_fin: Fecha final del histórico, en el formato aceptado por
            ``yfinance.download``.
        rsi_entrada, rsi_salida, probabilidad_minima,
        probabilidad_maxima_venta: Parámetros conservados por compatibilidad
            con el formulario común de estrategias; las reglas nuevas usan
            niveles fijos de RSI y no filtran por la probabilidad de IA.
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

    cierre_previo = df_indicadores['Close'].shift(1)
    rango_verdadero = pd.concat(
        [
            df_indicadores['High'] - df_indicadores['Low'],
            (df_indicadores['High'] - cierre_previo).abs(),
            (df_indicadores['Low'] - cierre_previo).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df_indicadores['ATR'] = rango_verdadero.rolling(
        window=14, min_periods=14
    ).mean()
    df_indicadores['EMA_20'] = df_indicadores['Close'].ewm(
        span=20, adjust=False, min_periods=20
    ).mean()
    df_indicadores['EMA_50'] = df_indicadores['Close'].ewm(
        span=50, adjust=False, min_periods=50
    ).mean()
    df_indicadores['Minimo_Local'] = df_indicadores['Low'].shift(1).rolling(
        window=5, min_periods=5
    ).min()
    df_indicadores['Ancho_Bollinger'] = (
        df_indicadores['Bollinger_Superior']
        - df_indicadores['Bollinger_Inferior']
    ) / df_indicadores['Bollinger_Media']
    umbral_compresion = df_indicadores['Ancho_Bollinger'].shift(1).rolling(
        window=120, min_periods=20
    ).quantile(0.2)
    df_indicadores['Bollinger_Abriendose'] = (
        df_indicadores['Ancho_Bollinger'].shift(1).le(umbral_compresion)
        & df_indicadores['Ancho_Bollinger'].gt(
            df_indicadores['Ancho_Bollinger'].shift(1)
        )
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

    print("Generando senales con los cuatro filtros...")
    precio_rompe_banda = (
        df_ml['Close'] > df_ml['Bollinger_Superior']
    )
    filtro_tendencia = (
        (df_ml['Close'] > df_ml['EMA_20'])
        & (df_ml['Close'] > df_ml['EMA_50'])
    )
    filtro_momento = df_ml['RSI'].between(50, 65, inclusive='neither')
    filtro_volatilidad = (
        df_ml['Bollinger_Abriendose'] | precio_rompe_banda
    )
    filtro_volumen = (
        (df_ml['Close'] > df_ml['High'].shift(1))
        & ((df_ml['High'] - df_ml['Low']) >= df_ml['ATR'])
        & (df_ml['Volume'] > df_ml['Volumen_Medio_20'])
    )
    df_ml['Recomendacion'] = [
        generar_senal(tendencia, momento, volatilidad, volumen)
        for tendencia, momento, volatilidad, volumen in zip(
            filtro_tendencia, filtro_momento,
            filtro_volatilidad, filtro_volumen,
        )
    ]
    df_ml['Senal_Entrada'] = df_ml['Recomendacion'].eq("COMPRA (BUY)")

    df_prueba = df_ml.iloc[train_size:].copy()
    probabilidad_subida = float(df_prueba['Prob_Subida'].iloc[-1])
    df_resultados = ejecutar_backtest(df_prueba)
    if df_resultados.empty:
        raise ValueError(
            "No hay suficientes resultados para mostrar. "
            "Selecciona un rango de fechas más amplio."
        )
    ultima_senal_trading = int(df_resultados['Senal_Trading'].iloc[-1])
    recomendacion_manana = df_prueba['Recomendacion'].iloc[-1]
    print(f"Recomendacion para manana: {recomendacion_manana}")
    print(f"Probabilidad estimada de subida: {probabilidad_subida:.2%}")
    print(f"Ultimo valor de Senal_Trading: {ultima_senal_trading}")

    grafico = graficar_resultados(
        df_resultados, ticker, recomendacion_manana,
        50.0, 65.0,
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
    rsi_entrada: float = 50.0, rsi_salida: float = 65.0,
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
            x=df.index, y=df['EMA_20'], mode='lines',
            name='EMA 20', line={'color': 'blue', 'width': 1},
        ),
        row=1, col=1,
    )
    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['EMA_50'], mode='lines',
            name='EMA 50', line={'color': 'brown', 'width': 1},
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
    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['Stop_Loss'], mode='lines',
            name='Stop Loss', line={'color': 'red', 'dash': 'dash'},
        ),
        row=1, col=1,
    )
    figura.add_trace(
        go.Scatter(
            x=df.index, y=df['Take_Profit'], mode='lines',
            name='Take Profit (1:2)', line={'color': 'green', 'dash': 'dash'},
        ),
        row=1, col=1,
    )
    compras = df[df['Senal_Trading'] == 1]
    ventas = df[df['Senal_Trading'] == -1]
    ventas_cierre = ventas['Precio_Salida'].fillna(ventas['Close'])
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
            x=ventas.index, y=ventas_cierre, mode='markers',
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
    figura.add_hline(
        y=70, line_dash='dot', line_color='darkred',
        annotation_text='Salida RSI sobrecompra (70)',
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
