"""Limpieza de datos de mercado y cálculo de indicadores técnicos."""

import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf


def descargar_datos(ticker, fecha_inicio, fecha_fin):
    """Descarga datos históricos de Yahoo Finance y los limpia."""
    print(f"Descargando datos de {ticker}...")
    df = yf.download(ticker, start=fecha_inicio, end=fecha_fin)
    return limpiar_datos(df, ticker)


def limpiar_datos(df, ticker):
    """Normaliza columnas OHLCV y descarta filas incompletas."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    if df.empty:
        raise ValueError(
            f"No se encontraron datos para {ticker} en el período indicado."
        )

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df.dropna()


def calcular_indicadores(
    df, periodo_media_corta: int = 50, periodo_media_larga: int = 200,
):
    """Añade medias, RSI, Bollinger y volumen medio de 20 sesiones.

    Las bandas de Bollinger usan una ventana de 20 sesiones y dos desviaciones
    estándar. La función descarta las filas iniciales que no tienen suficientes
    observaciones para calcular todos los indicadores.
    """
    print("Calculando indicadores técnicos...")
    df = df.copy()

    df["SMA_corta"] = ta.sma(
        close=df["Close"], length=periodo_media_corta
    )
    df["SMA_larga"] = ta.sma(
        close=df["Close"], length=periodo_media_larga
    )
    df["Volatilidad"] = ta.stdev(close=df["Close"], length=20)
    rsi = ta.rsi(close=df["Close"], length=14).clip(lower=0, upper=100)
    sin_variacion = (
        df["Close"].diff().abs().rolling(window=14, min_periods=14).sum() == 0
    )
    df["RSI"] = rsi.mask(sin_variacion, 50)

    bandas_bollinger = ta.bbands(close=df["Close"], length=20, std=2)
    df["Bollinger_Inferior"] = bandas_bollinger["BBL_20_2.0"]
    df["Bollinger_Media"] = bandas_bollinger["BBM_20_2.0"]
    df["Bollinger_Superior"] = bandas_bollinger["BBU_20_2.0"]
    df["Volumen_Medio_20"] = ta.sma(close=df["Volume"], length=20)

    return df.dropna()
