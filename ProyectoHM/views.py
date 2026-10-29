"""Vistas HTTP del sitio: noticias, análisis de estrategias y respuestas simples."""

import re
from datetime import date, timedelta
from urllib.error import URLError
from urllib.request import Request, urlopen

from django.http import HttpResponse
from django.shortcuts import render
from django.utils import timezone
from xml.etree import ElementTree

from ProyectoHM.src.CargaDatos import calcular_indicadores, descargar_datos
from ProyectoHM.src.EstrategiaV1 import (
    ejecutar_estrategia_completa as ejecutar_estrategia_v1,
)
from ProyectoHM.src.EstrategiaV2 import ejecutar_estrategia as ejecutar_estrategia_v2
from ProyectoHM.src.EstrategiaV3 import ejecutar_estrategia

EXPANSION_RSS = "https://www.expansion.com/rss/mercados.xml"
# Direcciones de los feeds que aparecen en la página de inicio.

CINCO_DIAS_RSS = (
    "https://feeds.elpais.com/mrss-s/pages/ep/site/"
    "cincodias.elpais.com/section/ultimas-noticias/portada"
)


def _local_name(tag):
    """Obtiene el nombre de una etiqueta XML sin el espacio de nombres."""
    return tag.rsplit("}", 1)[-1]


def _feed_items(url):
    """Descarga un RSS y extrae como máximo cinco noticias con título y enlace.

    Se usa un agente de usuario identificable, un tiempo máximo de espera y
    un límite de tamaño para acotar la descarga. El resultado contiene el
    título, la URL y, cuando está disponible, la fecha de publicación.

    Raises:
        URLError: Si falla la conexión.
        TimeoutError: Si la descarga supera el tiempo de espera.
        ValueError: Si el contenido excede el límite permitido.
        ElementTree.ParseError: Si el contenido no es XML válido.
    """
    request = Request(url, headers={"User-Agent": "TradingDjango/1.0"})
    with urlopen(request, timeout=8) as response:
        content = response.read(2 * 1024 * 1024 + 1)

    if len(content) > 2 * 1024 * 1024:
        raise ValueError("El RSS supera el tamaño máximo permitido.")

    root = ElementTree.fromstring(content)
    items = []

    for entry in root.iter():
        if _local_name(entry.tag) not in {"item", "entry"}:
            continue

        fields = {_local_name(child.tag): child for child in entry}
        title = fields.get("title")
        link = fields.get("link")
        published = (
            fields.get("pubDate")
            or fields.get("published")
            or fields.get("updated")
        )
        link_url = ""
        if link is not None:
            link_url = link.get("href") or (link.text or "").strip()

        if title is not None and title.text and link_url:
            items.append(
                {
                    "title": title.text.strip(),
                    "url": link_url,
                    "published": published.text.strip()
                    if published is not None and published.text
                    else "",
                }
            )
        if len(items) == 5:
            break

    return items


def _news_feed(name, url):
    """Prepara los datos de una fuente de noticias para mostrarlos en la web.

    Si la descarga o el análisis del RSS falla, devuelve igualmente la fuente
    con una descripción del error para que la página pueda informar al usuario.
    """
    feed = {"name": name, "url": url, "items": [], "error": ""}
    try:
        feed["items"] = _feed_items(url)
    except (ElementTree.ParseError, URLError, TimeoutError, ValueError) as error:
        feed["error"] = str(error) or "No se pudo leer este RSS."
    return feed


def home(request):
    """Muestra la página de inicio con noticias de dos fuentes RSS."""
    feeds = [
        _news_feed("Expansión", EXPANSION_RSS),
        _news_feed("Cinco Días", CINCO_DIAS_RSS),
    ]
    return render(request, "home.html", {"feeds": feeds})


def chatbot(request):
    """Muestra la página del chatbot de trading."""
    return render(request, "chatbot.html")


def indicadores(request):
    """Muestra los indicadores técnicos de un ticker en una fecha dada."""
    hoy = timezone.localdate()
    valores = {"ticker": "", "fecha": hoy.isoformat()}
    error = ""
    indicadores_calculados = None
    fecha_resultado = None

    if request.method == "POST":
        valores.update(
            {key: request.POST.get(key, "").strip() for key in valores}
        )
        valores["ticker"] = valores["ticker"].upper()

        if not re.fullmatch(r"[A-Z0-9.^=-]{1,15}", valores["ticker"]):
            error = "Introduce un ticker válido (máximo 15 caracteres)."

        try:
            fecha = date.fromisoformat(valores["fecha"])
        except ValueError:
            error = error or "Introduce una fecha válida."
        else:
            if fecha > hoy:
                error = error or "La fecha no puede ser futura."

        if not error:
            try:
                datos = descargar_datos(
                    valores["ticker"],
                    (fecha - timedelta(days=3 * 365)).isoformat(),
                    (fecha + timedelta(days=1)).isoformat(),
                )
                datos = calcular_indicadores(datos)
            except ValueError as exc:
                error = str(exc)
            else:
                filas_fecha = [
                    indice.date() <= fecha for indice in datos.index
                ]
                if not any(filas_fecha):
                    error = (
                        "No hay suficientes datos para calcular los indicadores "
                        "en esa fecha."
                    )
                else:
                    fila = datos.loc[filas_fecha].iloc[-1]
                    fecha_resultado = datos.index[filas_fecha][-1].date()
                    indicadores_calculados = [
                        {
                            "nombre": nombre,
                            "valor": int(fila[columna])
                            if columna == "Volume"
                            else float(fila[columna]),
                            "decimales": 0 if columna == "Volume" else 4,
                        }
                        for columna, nombre in (
                            ("Open", "Apertura"),
                            ("High", "Máximo"),
                            ("Low", "Mínimo"),
                            ("Close", "Cierre"),
                            ("Volume", "Volumen"),
                            ("SMA_corta", "Media móvil simple (50 sesiones)"),
                            ("SMA_larga", "Media móvil simple (200 sesiones)"),
                            ("Volatilidad", "Volatilidad (desviación estándar, 20 sesiones)"),
                            ("RSI", "RSI (14 sesiones)"),
                            ("Bollinger_Inferior", "Banda de Bollinger inferior"),
                            ("Bollinger_Media", "Media de Bollinger"),
                            ("Bollinger_Superior", "Banda de Bollinger superior"),
                            ("Volumen_Medio_20", "Volumen medio (20 sesiones)"),
                        )
                    ]

    return render(
        request,
        "indicadores.html",
        {
            "valores": valores,
            "error": error,
            "hoy": hoy.isoformat(),
            "ticker": valores["ticker"],
            "fecha_resultado": fecha_resultado,
            "indicadores": indicadores_calculados,
        },
    )


def estrategia3(request):
    """Valida ticker y fecha, ejecuta la estrategia V3 y muestra sus métricas.

    El formulario utiliza la fecha recibida como fecha de corte y analiza los
    cinco años anteriores. La fecha final se amplía un día para incluir la
    fecha seleccionada en la consulta histórica. En una petición GET muestra
    el formulario; en un POST conserva los valores, valida los datos y presenta
    los resultados o el mensaje de error en ``estrategia3.html``.
    """
    ticker = ""
    fecha = timezone.localdate().isoformat()
    error = ""
    metricas = None

    if request.method == "POST":
        ticker = request.POST.get("ticker", "").strip().upper()
        fecha = request.POST.get("fecha", "").strip()

        if not re.fullmatch(r"[A-Z0-9.^=-]{1,15}", ticker):
            error = "Introduce un ticker válido (máximo 15 caracteres)."
        else:
            try:
                fecha_corte = date.fromisoformat(fecha)
            except ValueError:
                error = "Introduce una fecha válida."
            else:
                if fecha_corte > timezone.localdate():
                    error = "La fecha no puede ser futura."
                else:
                    try:
                        resultado = ejecutar_estrategia(
                            ticker,
                            fecha_corte - timedelta(days=5 * 365),
                            fecha_corte + timedelta(days=1),
                        )
                    except ValueError as exc:
                        error = str(exc)
                    else:
                        modelo_ia = resultado["modelo_ia"]
                        metricas = {
                            **resultado,
                            "rentabilidad_entrenamiento_pct": (
                                resultado["rentabilidad_entrenamiento"] * 100
                            ),
                            "rentabilidad_prueba_pct": (
                                resultado["rentabilidad_prueba"] * 100
                            ),
                            "accuracy_validacion_pct": (
                                modelo_ia["accuracy_validacion"] * 100
                            ),
                            "probabilidad_alza_actual_pct": (
                                modelo_ia["probabilidad_alza_actual"] * 100
                            ),
                        }

    return render(
        request,
        "estrategia3.html",
        {"ticker": ticker, "fecha": fecha, "error": error, "metricas": metricas},
    )


def estrategia_v2(request):
    """Muestra el formulario de la estrategia V2 y ejecuta el análisis enviado.

    Valida el ticker, el intervalo de fechas y el orden de los períodos de las
    medias móviles antes de llamar a ``EstrategiaV2.ejecutar_estrategia``.
    Cuando el análisis termina, entrega sus rentabilidades, probabilidad,
    última señal y gráfico a ``estrategiav2.html``. Para el formulario o los
    errores de entrada utiliza ``estrategia_v2.html``.
    """
    hoy = timezone.localdate()
    valores = {
        "ticker": "",
        "fecha_inicio": (hoy - timedelta(days=5 * 365)).isoformat(),
        "fecha_fin": hoy.isoformat(),
        "periodo_media_corta": "50",
        "periodo_media_larga": "200",
    }

    if request.method == "POST":
        valores.update(
            {key: request.POST.get(key, "").strip() for key in valores}
        )
        valores["ticker"] = valores["ticker"].upper()

    error = ""
    if request.method == "POST":
        if not re.fullmatch(r"[A-Z0-9.^=-]{1,15}", valores["ticker"]):
            error = "Introduce un ticker válido (máximo 15 caracteres)."

        try:
            fecha_inicio = date.fromisoformat(valores["fecha_inicio"])
            fecha_fin = date.fromisoformat(valores["fecha_fin"])
        except ValueError:
            error = error or "Introduce un rango de fechas válido."
        else:
            if fecha_inicio >= fecha_fin:
                error = error or "La fecha de inicio debe ser anterior a la fecha de fin."
            elif fecha_fin > hoy:
                error = error or "La fecha de fin no puede ser futura."

        try:
            periodo_media_corta = int(valores["periodo_media_corta"])
            periodo_media_larga = int(valores["periodo_media_larga"])
        except ValueError:
            error = error or "Los períodos de las medias deben ser números enteros."
        else:
            if periodo_media_corta < 1 or periodo_media_larga < 1:
                error = error or "Los períodos de las medias deben ser positivos."
            elif periodo_media_corta >= periodo_media_larga:
                error = error or "La media corta debe tener un período menor que la larga."

        if not error:
            try:
                (
                    grafico,
                    rentabilidad_mercado,
                    rentabilidad_estrategia,
                    probabilidad_subida,
                    ultima_senal_trading,
                ) = ejecutar_estrategia_v2(
                    valores["ticker"],
                    fecha_inicio.isoformat(),
                    fecha_fin.isoformat(),
                    periodo_media_corta=periodo_media_corta,
                    periodo_media_larga=periodo_media_larga,
                )
            except ValueError as exc:
                error = str(exc)
            else:
                senales = {1: "COMPRA", -1: "VENTA", 0: "MANTENER"}
                return render(
                    request,
                    "estrategiav2.html",
                    {
                        "ticker": valores["ticker"],
                        "valores": valores,
                        "rentabilidad_mercado": rentabilidad_mercado,
                        "rentabilidad_estrategia": rentabilidad_estrategia,
                        "probabilidad_subida_pct": probabilidad_subida * 100,
                        "ultima_senal_trading": senales.get(
                            ultima_senal_trading, "DESCONOCIDA"
                        ),
                        "grafico": grafico,
                    },
                )

    return render(
        request,
        "estrategia_v2.html",
        {"valores": valores, "error": error, "hoy": hoy.isoformat()},
    )


def estrategia_v1(request):
    """Valida el formulario V1 y muestra las métricas y el gráfico del backtest."""
    hoy = timezone.localdate()
    valores = {
        "ticker": "",
        "fecha_inicio": (hoy - timedelta(days=5 * 365)).isoformat(),
        "fecha_fin": hoy.isoformat(),
        "rsi_entrada": "30",
        "rsi_salida": "70",
        "probabilidad_minima": "50",
        "probabilidad_maxima_venta": "30",
        "periodo_media_corta": "50",
        "periodo_media_larga": "200",
    }

    if request.method == "POST":
        valores.update(
            {key: request.POST.get(key, "").strip() for key in valores}
        )
        valores["ticker"] = valores["ticker"].upper()

    error = ""
    if request.method == "POST":
        if not re.fullmatch(r"[A-Z0-9.^=-]{1,15}", valores["ticker"]):
            error = "Introduce un ticker válido (máximo 15 caracteres)."

        try:
            fecha_inicio = date.fromisoformat(valores["fecha_inicio"])
            fecha_fin = date.fromisoformat(valores["fecha_fin"])
        except ValueError:
            error = error or "Introduce un rango de fechas válido."
        else:
            if fecha_inicio >= fecha_fin:
                error = error or "La fecha de inicio debe ser anterior a la fecha de fin."
            elif fecha_fin > hoy:
                error = error or "La fecha de fin no puede ser futura."

        try:
            rsi_entrada = float(valores["rsi_entrada"])
            rsi_salida = float(valores["rsi_salida"])
            probabilidad_minima = float(valores["probabilidad_minima"]) / 100
            probabilidad_maxima_venta = (
                float(valores["probabilidad_maxima_venta"]) / 100
            )
            periodo_media_corta = int(valores["periodo_media_corta"])
            periodo_media_larga = int(valores["periodo_media_larga"])
        except ValueError:
            error = error or "Revisa los valores numéricos del formulario."
        else:
            if not 0 <= rsi_entrada <= 100 or not 0 <= rsi_salida <= 100:
                error = error or "Los umbrales de RSI deben estar entre 0 y 100."
            elif not 0 <= probabilidad_minima <= 1 or not 0 <= probabilidad_maxima_venta <= 1:
                error = error or "Las probabilidades deben estar entre 0 y 100."
            elif periodo_media_corta < 1 or periodo_media_larga < 1:
                error = error or "Los períodos de las medias deben ser positivos."
            elif periodo_media_corta >= periodo_media_larga:
                error = error or "La media corta debe tener un período menor que la larga."

        if not error:
            try:
                (
                    grafico,
                    rentabilidad_mercado,
                    rentabilidad_estrategia,
                    probabilidad_subida,
                    ultima_senal_trading,
                    grafico_tecnico,
                    rentabilidad_mercado_tecnica,
                    rentabilidad_tecnica,
                    ultima_senal_tecnica,
                ) = ejecutar_estrategia_v1(
                    valores["ticker"],
                    fecha_inicio.isoformat(),
                    fecha_fin.isoformat(),
                    rsi_entrada=rsi_entrada,
                    rsi_salida=rsi_salida,
                    probabilidad_minima=probabilidad_minima,
                    probabilidad_maxima_venta=probabilidad_maxima_venta,
                    periodo_media_corta=periodo_media_corta,
                    periodo_media_larga=periodo_media_larga,
                )
            except ValueError as exc:
                error = str(exc)
            else:
                senales = {1: "COMPRA", -1: "VENTA", 0: "MANTENER"}
                return render(
                    request,
                    "estrategiav1.html",
                    {
                        "ticker": valores["ticker"],
                        "valores": valores,
                        "rentabilidad_mercado": rentabilidad_mercado,
                        "rentabilidad_estrategia": rentabilidad_estrategia,
                        "probabilidad_subida_pct": probabilidad_subida * 100,
                        "ultima_senal_trading": senales.get(
                            ultima_senal_trading, "DESCONOCIDA"
                        ),
                        "grafico": grafico,
                        "rentabilidad_mercado_tecnica": (
                            rentabilidad_mercado_tecnica
                        ),
                        "rentabilidad_tecnica": rentabilidad_tecnica,
                        "ultima_senal_tecnica": senales.get(
                            ultima_senal_tecnica, "DESCONOCIDA"
                        ),
                        "grafico_tecnico": grafico_tecnico,
                    },
                )

    return render(
        request,
        "estrategiasV1.html",
        {"valores": valores, "error": error, "hoy": hoy.isoformat()},
    )
