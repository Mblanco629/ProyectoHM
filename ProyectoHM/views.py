from urllib.error import URLError
from urllib.request import Request, urlopen

from django.http import HttpResponse
from django.shortcuts import render
from django.utils import timezone
from xml.etree import ElementTree

EXPANSION_RSS = "https://www.expansion.com/rss/mercados.xml"
CINCO_DIAS_RSS = (
    "https://feeds.elpais.com/mrss-s/pages/ep/site/"
    "cincodias.elpais.com/section/ultimas-noticias/portada"
)


def _local_name(tag):
    return tag.rsplit("}", 1)[-1]


def _feed_items(url):
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
    feed = {"name": name, "url": url, "items": [], "error": ""}
    try:
        feed["items"] = _feed_items(url)
    except (ElementTree.ParseError, URLError, TimeoutError, ValueError) as error:
        feed["error"] = str(error) or "No se pudo leer este RSS."
    return feed


def home(request):
    feeds = [
        _news_feed("Expansión", EXPANSION_RSS),
        _news_feed("Cinco Días", CINCO_DIAS_RSS),
    ]
    return render(request, "home.html", {"feeds": feeds})


def saludo(request):
    return HttpResponse("Hola mundo, esta es mi primera vista en Django")


def hola(request):
    fecha_hora = timezone.localtime()
    return HttpResponse(f"Hola, la fecha y hora actual es: {fecha_hora:%d/%m/%Y %H:%M:%S}")
