from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase
from django.urls import reverse


class GraficoTickerTests(SimpleTestCase):
    def test_get_muestra_formulario_y_opcion_de_menu(self):
        response = self.client.get(reverse("grafico_ticker"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="ticker"')
        self.assertContains(response, "Gráfico de ticker")

    @patch("ProyectoHM.views.descargar_datos")
    def test_post_renderiza_datos_y_permite_cambiar_tipo(self, descargar_datos):
        descargar_datos.return_value = pd.DataFrame(
            {
                "Open": [10.0, 11.0],
                "High": [12.0, 13.0],
                "Low": [9.0, 10.0],
                "Close": [11.0, 12.0],
                "Volume": [100, 200],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

        response = self.client.post(
            reverse("grafico_ticker"),
            {
                "ticker": "aapl",
                "fecha_inicio": "2024-01-01",
                "fecha_fin": "2024-01-03",
                "chart_type": "line",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.context["datos_grafico"][0],
            {
                "time": "2024-01-02",
                "open": 10.0,
                "high": 12.0,
                "low": 9.0,
                "close": 11.0,
            },
        )
        self.assertContains(response, "lightweight-charts@4.2.3")
        self.assertContains(response, 'value="line" selected')
        descargar_datos.assert_called_once_with(
            "AAPL", "2024-01-01", "2024-01-04"
        )

    @patch("ProyectoHM.views.descargar_datos")
    def test_post_invalido_no_descarga_datos(self, descargar_datos):
        response = self.client.post(
            reverse("grafico_ticker"),
            {
                "ticker": "INVALID TICKER",
                "fecha_inicio": "2024-01-03",
                "fecha_fin": "2024-01-01",
                "chart_type": "unknown",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Introduce un ticker válido")
        descargar_datos.assert_not_called()
