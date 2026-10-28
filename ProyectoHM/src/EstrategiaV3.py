"""Optimiza y valida una estrategia de cruce de medias móviles simples.

La optimización compara períodos rápidos de 5 a 35 sesiones con períodos
lentos de 20 a 140 sesiones, seleccionando la combinación con mayor ratio
Sharpe en el 70 % inicial de los datos. La combinación ganadora se evalúa
también en el 30 % final, reservado para validación.

Además, un clasificador Random Forest se entrena con el mismo tramo inicial y
se valida en el tramo final para estimar la dirección del cierre siguiente.
Las variables se construyen únicamente con información disponible al cierre de
cada sesión.

Las señales se aplican en la sesión siguiente al cruce para evitar operar con
información que aún no estaba disponible al generarse la señal. La simulación
descuenta una comisión del 0,2 % cuando cambia la posición. Los datos se
descargan mediante ``CargaDatos.descargar_datos``.
"""

import math
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.CargaDatos import descargar_datos
else:
    from .CargaDatos import descargar_datos


def _detectar_cruces(cierre, periodo_rapido, periodo_lento):
    """Devuelve cruces de medias compartidos por el backtest y la señal."""
    media_rapida = cierre.rolling(periodo_rapido).mean()
    media_lenta = cierre.rolling(periodo_lento).mean()
    cruce_alcista = (
        (media_rapida > media_lenta)
        & (media_rapida.shift(1) <= media_lenta.shift(1))
    )
    cruce_bajista = (
        (media_rapida < media_lenta)
        & (media_rapida.shift(1) >= media_lenta.shift(1))
    )
    return cruce_alcista, cruce_bajista


def _evaluar_parametros(
    precios, periodo_rapido, periodo_lento, inicio_evaluacion=0
):
    """Simula una combinación de medias y calcula Sharpe y rentabilidad.

    La posición pasa a largo en un cruce alcista y a efectivo en uno bajista.
    Las señales se desplazan una sesión y los retornos se calculan desde
    ``inicio_evaluacion``, lo que permite evaluar solo el tramo de validación
    sin perder el histórico necesario para calcular las medias.

    Args:
        precios: DataFrame con una columna ``Close`` y un índice temporal.
        periodo_rapido: Ventana, en sesiones, de la media rápida.
        periodo_lento: Ventana, en sesiones, de la media lenta.
        inicio_evaluacion: Posición desde la que incluir retornos en las
            métricas.

    Returns:
        Tupla con el ratio Sharpe anualizado y la rentabilidad acumulada.
        El Sharpe es infinito negativo cuando no se puede calcular.
    """
    cierre = precios["Close"].astype(float)
    cruce_alcista, cruce_bajista = _detectar_cruces(
        cierre, periodo_rapido, periodo_lento
    )

    posicion = 0
    posiciones = []
    for compra, venta in zip(cruce_alcista, cruce_bajista):
        if compra:
            posicion = 1
        elif venta:
            posicion = 0
        posiciones.append(posicion)

    posiciones = pd.Series(posiciones, index=precios.index).shift(1).fillna(0)
    cambios_posicion = posiciones.diff().abs().fillna(posiciones.abs())
    retornos = cierre.pct_change().fillna(0) * posiciones - 0.002 * cambios_posicion
    retornos = retornos.iloc[inicio_evaluacion:]
    desviacion = retornos.std(ddof=1)
    ratio_sharpe = (
        float(retornos.mean() / desviacion * math.sqrt(252))
        if pd.notna(desviacion) and desviacion > 0
        else float("-inf")
    )
    rentabilidad = float((1 + retornos).prod() - 1)
    return ratio_sharpe, rentabilidad


def _recomendacion_siguiente_sesion(datos, periodo_rapido, periodo_lento):
    """Obtiene la acción siguiente según el estado del cruce optimizado."""
    cierre = datos["Close"].astype(float)
    cruce_alcista, cruce_bajista = _detectar_cruces(
        cierre, periodo_rapido, periodo_lento
    )

    if cruce_alcista.iloc[-1]:
        return "COMPRA (BUY)"
    if cruce_bajista.iloc[-1]:
        return "VENTA (SELL)"

    posicion = 0
    for compra, venta in zip(cruce_alcista, cruce_bajista):
        if compra:
            posicion = 1
        elif venta:
            posicion = 0
    return "MANTENER (HOLD)" if posicion else "ESPERAR (SIN POSICIÓN)"


def _entrenar_modelo_ia(datos, corte):
    """Entrena RandomForest en el tramo inicial y mide su predicción final.

    Las variables predictoras solo usan precios disponibles al cierre de cada
    sesión. La etiqueta representa si el cierre de la sesión siguiente sube.
    Se excluye de entrenamiento la última sesión previa al corte porque su
    etiqueta depende del primer precio del tramo de validación.

    Args:
        datos: DataFrame cronológico con una columna ``Close``.
        corte: Posición donde empieza el tramo de validación.

    Returns:
        Métricas de clasificación en validación y señal/probabilidad para la
        última sesión disponible.

    Raises:
        ValueError: Si el histórico no proporciona suficientes observaciones
            o el tramo de entrenamiento contiene una única clase.
    """
    cierre = datos["Close"].astype(float)
    retornos = cierre.pct_change()
    cierre_siguiente = cierre.shift(-1)
    variables = pd.DataFrame(
        {
            "retorno_1": retornos,
            "retorno_5": cierre.pct_change(5),
            "retorno_10": cierre.pct_change(10),
            "distancia_sma_5": cierre / cierre.rolling(5).mean() - 1,
            "distancia_sma_20": cierre / cierre.rolling(20).mean() - 1,
            "volatilidad_20": retornos.rolling(20).std(),
            "posicion": range(len(datos)),
            "sube_siguiente_sesion": cierre_siguiente.gt(cierre)
            .where(cierre_siguiente.notna())
            .astype("Int64"),
        },
        index=datos.index,
    )
    columnas_variables = [
        "retorno_1",
        "retorno_5",
        "retorno_10",
        "distancia_sma_5",
        "distancia_sma_20",
        "volatilidad_20",
    ]
    muestras = variables.dropna(subset=columnas_variables).copy()
    entrenamiento = muestras.loc[
        (muestras["posicion"] < corte - 1)
        & muestras["sube_siguiente_sesion"].notna()
    ]
    validacion = muestras.loc[
        (muestras["posicion"] >= corte)
        & muestras["sube_siguiente_sesion"].notna()
    ]
    ultima = muestras.iloc[[-1]]

    if len(entrenamiento) < 20 or len(validacion) < 5 or ultima.empty:
        raise ValueError(
            "El período es demasiado corto para entrenar y validar el modelo "
            "Random Forest. Selecciona un rango de fechas más amplio."
        )

    etiquetas_entrenamiento = entrenamiento[
        "sube_siguiente_sesion"
    ].astype(int)
    if etiquetas_entrenamiento.nunique() < 2:
        raise ValueError(
            "Los datos de entrenamiento del modelo no contienen suficientes "
            "subidas y bajadas. Prueba con otro período."
        )

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    modelo.fit(
        entrenamiento[columnas_variables],
        etiquetas_entrenamiento,
    )

    etiquetas_validacion = validacion[
        "sube_siguiente_sesion"
    ].astype(int)
    predicciones = modelo.predict(validacion[columnas_variables])
    prediccion_actual = int(modelo.predict(ultima[columnas_variables])[0])
    probabilidades_actuales = modelo.predict_proba(ultima[columnas_variables])[0]
    indice_clase_alza = list(modelo.classes_).index(1)

    return {
        "accuracy_validacion": float(
            accuracy_score(etiquetas_validacion, predicciones)
        ),
        "precision_validacion": float(
            precision_score(
                etiquetas_validacion, predicciones, zero_division=0
            )
        ),
        "recall_validacion": float(
            recall_score(etiquetas_validacion, predicciones, zero_division=0)
        ),
        "f1_validacion": float(
            f1_score(etiquetas_validacion, predicciones, zero_division=0)
        ),
        "senal_actual": "ALZA" if prediccion_actual == 1 else "BAJA",
        "probabilidad_alza_actual": float(
            probabilidades_actuales[indice_clase_alza]
        ),
        "muestras_entrenamiento": int(len(entrenamiento)),
        "muestras_validacion": int(len(validacion)),
    }


def optimizar_parametros(datos):
    """Optimiza las medias en el tramo inicial y evalúa la ganadora al final.

    Descarta filas sin precio de cierre, reserva el 70 % inicial del histórico
    para buscar parámetros y usa el 30 % restante para medir su rendimiento
    fuera de la muestra de optimización. El Sharpe se anualiza con 252 sesiones
    bursátiles y los cálculos consideran una comisión del 0,2 % por cambio de
    posición. En la misma partición cronológica entrena y evalúa un
    clasificador Random Forest para predecir la dirección del cierre siguiente.

    Args:
        datos: DataFrame de precios que debe incluir la columna ``Close``.

    Returns:
        Diccionario con ``periodo_rapido``, ``periodo_lento``,
        ``sharpe_entrenamiento``, ``rentabilidad_entrenamiento``,
        ``sharpe_prueba`` y ``rentabilidad_prueba``, más ``modelo_ia`` con las
        métricas de validación y la predicción para la última sesión, además de
        ``recomendacion_backtest`` para la siguiente sesión. El Sharpe de prueba
        es ``None`` si no se puede calcular.

    Raises:
        ValueError: Si faltan precios de cierre, el histórico es demasiado
            corto o no se encuentra una combinación con un Sharpe válido.
    """
    if datos.empty or "Close" not in datos:
        raise ValueError("No hay precios de cierre para optimizar la estrategia.")

    datos = datos.dropna(subset=["Close"]).copy()
    corte = int(len(datos) * 0.7)
    datos_entrenamiento = datos.iloc[:corte]
    if len(datos_entrenamiento) < 21:
        raise ValueError(
            "El período es demasiado corto para optimizar y validar la estrategia. "
            "Selecciona un rango de fechas más amplio."
        )

    periodos_lentos = [
        periodo
        for periodo in range(20, min(150, len(datos_entrenamiento)), 10)
    ]
    if not periodos_lentos:
        raise ValueError(
            "El período es demasiado corto para optimizar las medias móviles."
        )

    mejor = None
    for periodo_lento in periodos_lentos:
        for periodo_rapido in range(5, 40, 5):
            if periodo_rapido >= periodo_lento:
                continue
            sharpe, rentabilidad = _evaluar_parametros(
                datos_entrenamiento, periodo_rapido, periodo_lento
            )
            if mejor is None or sharpe > mejor["sharpe_entrenamiento"]:
                mejor = {
                    "periodo_rapido": periodo_rapido,
                    "periodo_lento": periodo_lento,
                    "sharpe_entrenamiento": sharpe,
                    "rentabilidad_entrenamiento": rentabilidad,
                }

    if mejor is None or not math.isfinite(mejor["sharpe_entrenamiento"]):
        raise ValueError(
            "No se encontraron operaciones válidas para optimizar. "
            "Prueba con otro ticker o un período más amplio."
        )

    sharpe_prueba, rentabilidad_prueba = _evaluar_parametros(
        datos,
        mejor["periodo_rapido"],
        mejor["periodo_lento"],
        inicio_evaluacion=corte,
    )
    mejor["sharpe_prueba"] = (
        sharpe_prueba if math.isfinite(sharpe_prueba) else None
    )
    mejor["rentabilidad_prueba"] = rentabilidad_prueba
    mejor["recomendacion_backtest"] = _recomendacion_siguiente_sesion(
        datos, mejor["periodo_rapido"], mejor["periodo_lento"]
    )
    mejor["modelo_ia"] = _entrenar_modelo_ia(datos, corte)
    return mejor


def ejecutar_estrategia(ticker, fecha_inicio, fecha_fin):
    """Descarga datos de mercado y optimiza los períodos de las medias.

    Args:
        ticker: Símbolo del activo admitido por Yahoo Finance.
        fecha_inicio: Fecha inicial que acepta
            :func:`CargaDatos.descargar_datos`.
        fecha_fin: Fecha final que acepta
            :func:`CargaDatos.descargar_datos`.

    Returns:
        Diccionario de parámetros óptimos y métricas producido por
        :func:`optimizar_parametros`.

    Raises:
        ValueError: Si la descarga no contiene datos utilizables o el
            histórico es insuficiente para optimizar y validar la estrategia.
    """
    datos = descargar_datos(ticker, fecha_inicio, fecha_fin)
    return optimizar_parametros(datos)
