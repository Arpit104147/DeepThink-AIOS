import os
import json
import re
import urllib.parse
import requests
from bs4 import BeautifulSoup  # Inherited from system packages if available

# ─────────────────────────────────────────────────────────────────────────
# Grounding-quality knobs (Phase 2.2)
#
# PER_SOURCE_CHAR_CAP     Cap for text extracted from a single source. The
#                          old code used 15 000 chars per source which
#                          crowded out the other 2-3 sources entirely;
#                          2 500 lets 3-5 sources coexist in ~8-12k of ctx.
# MAX_SCRAPE_ATTEMPTS     After domain-dedup, how many URLs we're willing
#                          to hit per query. Bounded to keep latency low.
# PARAGRAPH_KEEP_TOP_N    From each scraped page we keep at most this many
#                          paragraphs, chosen by keyword overlap with the
#                          query. Prevents boilerplate menu text from
#                          dominating the extracted context.
# MIN_USEFUL_TEXT_CHARS   Below this length a scraped source is treated as
#                          empty (captcha stubs, error pages, redirects).
# ─────────────────────────────────────────────────────────────────────────
PER_SOURCE_CHAR_CAP = 2_500
MAX_SCRAPE_ATTEMPTS = 3
PARAGRAPH_KEEP_TOP_N = 12
MIN_USEFUL_TEXT_CHARS = 400

class WebSearch:
    def __init__(self, google_api_key=None, google_cx=None, searxng_url=None):
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        self.google_cx = google_cx or os.environ.get("GOOGLE_CX")
        # Default to a reliable public instance if not specified
        self.searxng_url = searxng_url or os.environ.get("SEARXNG_URL", "https://searx.be")
        # Persistent session for TCP connection pooling (reuses SSL handshakes)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"})

    def search(self, query, max_results=5):
        """
        Search the web. Priority: Google API -> SearXNG -> DuckDuckGo.
        """
        if self.google_api_key and self.google_cx:
            print(f"Searching Google for: '{query}'")
            return self._google_search(query, max_results)
        elif self.searxng_url:
            print(f"Searching SearXNG ({self.searxng_url}) for: '{query}'")
            return self._searxng_search(query, max_results)
        else:
            print(f"Searching DuckDuckGo (free fallback) for: '{query}'")
            return self._ddg_search_api(query, max_results)

    def _searxng_search(self, query, max_results=5):
        """Search using SearXNG JSON API with multiple instance fallbacks."""
        # Try multiple public SearXNG instances in case one is down
        instances = [
            self.searxng_url,
            "https://search.ononoki.org",
            "https://searx.tiekoetter.com",
            "https://search.sapti.me",
        ]
        safe_query = urllib.parse.quote(query)
        
        for instance_url in instances:
            try:
                url = f"{instance_url.rstrip('/')}/search?q={safe_query}&format=json"
                response = self._session.get(url, timeout=5.0)
                
                # Skip if we got HTML instead of JSON (captcha/error page)
                content_type = response.headers.get('content-type', '')
                if 'json' not in content_type and not response.text.strip().startswith('{'):
                    print(f"SearXNG {instance_url} returned non-JSON. Trying next...")
                    continue
                    
                data = response.json()
                results = []
                if "results" in data:
                    for item in data["results"][:max_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "link": item.get("url", ""),
                            "snippet": item.get("content", "")
                        })
                if results:
                    return results
            except Exception as e:
                print(f"SearXNG {instance_url} failed: {str(e)}. Trying next...")
                continue
        
        print("All SearXNG instances failed. Falling back to DuckDuckGo...")
        return self._ddg_search_api(query, max_results)

    def _google_search(self, query, max_results=5):
        """Search using Google Custom Search JSON API."""
        try:
            safe_query = urllib.parse.quote(query)
            url = f"https://www.googleapis.com/customsearch/v1?key={self.google_api_key}&cx={self.google_cx}&q={safe_query}&num={max_results}"
            
            response = self._session.get(url, timeout=5)
            data = response.json()
                
            results = []
            if "items" in data:
                for item in data["items"]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
            return results
        except Exception as e:
            print(f"Google search API failed: {str(e)}. Falling back to DuckDuckGo...")
            return self._ddg_search_api(query, max_results)

    def _ddg_search_api(self, query, max_results=5):
        """
        Search using duckduckgo_search library if available,
        or fall back to a light web request scraper if not.
        """
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Try new package name first, fall back to old one
                try:
                    from ddgs import DDGS
                except ImportError:
                    from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "link": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
                if results:
                    return results
                else:
                    print("DuckDuckGo search library returned 0 results. Trying HTML scraper...")
                    return self._ddg_html_scraper(query, max_results)
        except ImportError:
            # Fallback to direct HTML search scraping or HTML API
            return self._ddg_html_scraper(query, max_results)
        except Exception as e:
            print(f"DuckDuckGo search failed: {str(e)}")
            return self._ddg_html_scraper(query, max_results)

    def _ddg_html_scraper(self, query, max_results=5):
        """Scrape DuckDuckGo HTML search page as a robust fallback without library dependencies."""
        try:
            # DuckDuckGo HTML version is lightweight and scrapeable
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            
            response = self._session.get(url, timeout=5.0)
            html = response.content
                
            soup = BeautifulSoup(html, "html.parser")
            results = []
            
            # Find search result divs on ddg html page
            for result_div in soup.find_all("div", class_="result"):
                if len(results) >= max_results:
                    break
                    
                title_elem = result_div.find("a", class_="result__a")
                snippet_elem = result_div.find("a", class_="result__snippet")
                
                if title_elem:
                    title = title_elem.text.strip()
                    link = title_elem.get("href", "")
                    
                    # Handle redirect link mapping
                    if "uddg=" in link or link.startswith("//"):
                        if link.startswith("//"):
                            full_url = "https:" + link
                        else:
                            full_url = link
                        parsed = urllib.parse.urlparse(full_url)
                        qs = urllib.parse.parse_qs(parsed.query)
                        if "uddg" in qs:
                            link = qs["uddg"][0]
                            
                    snippet = snippet_elem.text.strip() if snippet_elem else ""
                    
                    if link:
                        results.append({
                            "title": title,
                            "link": link,
                            "snippet": snippet
                        })
            return results
        except Exception as e:
            print(f"HTML scraper fallback failed: {str(e)}")
            return []

    def scrape_url(self, url):
        """Deep scrape the full text of a webpage."""
        try:
            print(f"Deep scraping: {url}")
            response = self._session.get(url, timeout=8.0)
            if response.status_code != 200:
                print(f"Failed to scrape {url} (Status: {response.status_code})")
                return ""
            
            # Check for common bot-protection/blocking pages
            lower_text = response.text.lower()
            block_markers = ["cloudflare", "captcha", "attention required", "access denied", "checking your browser", "ddos protection", "robot check"]
            if any(marker in lower_text for marker in block_markers):
                print(f"Scrape block detected (Cloudflare/Captcha/Access Denied) for {url}")
                return ""

            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove junk elements
            for el in soup(["script", "style", "nav", "footer", "header", "aside"]):
                el.decompose()
                
            text = soup.get_text(separator="\n")
            # Clean up excessive whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = "\n".join(lines)
            
            # Cap at 15000 characters to prevent context overflow
            return cleaned_text[:15000]
        except Exception as e:
            print(f"Failed to deep scrape {url}: {str(e)}")
            return ""

    # ────────────────────────────────────────────────────────────────────
    # Phase 2.2 — grounding-quality helpers.
    #
    # These new methods work *on top of* the existing search()/scrape_url()
    # so we don't have to touch callers that only want raw results. The
    # orchestrator's RAG path should prefer `search_and_scrape()`.
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_by_domain(results):
        """
        Keep at most one hit per registered netloc (foo.example.com counts
        as `example.com`). Preserves input ordering so the highest-ranked
        result from each domain wins.
        """
        seen = set()
        deduped = []
        for r in results:
            link = (r.get("link") or "").strip()
            if not link:
                continue
            try:
                netloc = urllib.parse.urlparse(link).netloc.lower()
            except Exception:
                netloc = link
            # Reduce "en.wikipedia.org" and "de.wikipedia.org" to the same
            # domain family; likewise "docs.python.org" and "www.python.org".
            parts = netloc.split(".")
            root = ".".join(parts[-2:]) if len(parts) >= 2 else netloc
            if root in seen:
                continue
            seen.add(root)
            deduped.append(r)
        return deduped

    @staticmethod
    def _query_keywords(query):
        """Extract lowercase content words from `query` for relevance scoring."""
        _STOP = {
            "a", "an", "and", "or", "the", "of", "in", "on", "at", "to", "for",
            "is", "are", "was", "were", "be", "been", "being", "with", "by",
            "as", "it", "its", "this", "that", "these", "those", "how", "what",
            "when", "where", "why", "which", "who", "whom", "do", "does", "did",
            "can", "could", "should", "would", "will", "shall", "may", "might",
        }
        tokens = re.findall(r"[a-z0-9][a-z0-9\-']+", query.lower())
        return set(t for t in tokens if len(t) > 2 and t not in _STOP)

    def _relevance_filter(self, cleaned_text, keywords):
        """
        Split `cleaned_text` into paragraphs, score each by keyword overlap
        with the query, and return the concatenation of the top-N most
        relevant paragraphs (capped by PER_SOURCE_CHAR_CAP). This replaces
        the naive "first N thousand chars" truncation which usually captured
        nav / boilerplate rather than the answer.

        If `keywords` is empty (very short query), just return the head of
        the text — falling back to the old behavior.
        """
        if not cleaned_text:
            return ""
        if not keywords:
            return cleaned_text[:PER_SOURCE_CHAR_CAP]

        # Split on blank lines so a paragraph is a semantic unit, not one line.
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned_text) if p.strip()]
        if not paragraphs:
            # Fall back to line-based splitting if the page had no blank lines.
            paragraphs = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]

        scored = []
        for p in paragraphs:
            # Skip fragments that are obviously navigation / cookie banners.
            if len(p) < 40:
                continue
            p_words = set(re.findall(r"[a-z0-9][a-z0-9\-']+", p.lower()))
            overlap = len(keywords & p_words)
            if overlap == 0:
                continue
            # Reward density: a short paragraph with 3 hits beats a huge
            # paragraph with 3 hits — this favors precision over recall.
            density = overlap / max(1, len(p_words))
            scored.append((overlap + density, p))

        if not scored:
            # Nothing matched — return the head of the page as a last resort
            # (better than empty; still capped).
            return cleaned_text[:PER_SOURCE_CHAR_CAP]

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = []
        total = 0
        for _, p in scored[:PARAGRAPH_KEEP_TOP_N]:
            if total + len(p) > PER_SOURCE_CHAR_CAP:
                remaining = PER_SOURCE_CHAR_CAP - total
                if remaining > 200:
                    chosen.append(p[:remaining] + "…")
                break
            chosen.append(p)
            total += len(p)
        return "\n\n".join(chosen)

    def search_and_scrape(self, query, max_results=5, max_scrapes=None):
        """
        High-level RAG helper (Phase 2.2).

        1. Run `search(query, max_results)`.
        2. Domain-dedup so a single site can't dominate.
        3. Deep-scrape up to `max_scrapes` (default MAX_SCRAPE_ATTEMPTS).
        4. Filter each scraped page down to the paragraphs most relevant
           to the query (PARAGRAPH_KEEP_TOP_N × PER_SOURCE_CHAR_CAP cap).
        5. Return a structured dict:

            {
              "empty": bool,                       # True ⇔ no usable text
              "sources_scraped": int,              # pages that produced text
              "sources_blocked": int,              # pages that returned "" 
              "context": str,                      # joined, source-labeled
              "sources": [                         # per-source metadata
                  {"title", "link", "netloc", "chars"},
                  ...
              ]
            }

        The `context` is pre-labeled with `[SOURCE 1: <netloc>]` blocks so
        the LLM can attribute claims back to a specific origin. Callers
        should check `empty` and *omit* any "Web-scraped context:" header
        when it is True — injecting an empty block encourages the model to
        hallucinate rather than admit ignorance.
        """
        if max_scrapes is None:
            max_scrapes = MAX_SCRAPE_ATTEMPTS

        raw = self.search(query, max_results=max_results) or []
        deduped = self._dedup_by_domain(raw)[:max_scrapes]

        keywords = self._query_keywords(query)
        parts = []
        sources_meta = []
        sources_scraped = 0
        # 1. Check for real-time meteorological weather report first
        weather_table = self.fetch_live_weather(query)
        if weather_table:
            parts.append(weather_table)
            sources_scraped += 1

        # 2. Check for real-time financial market quotes
        financial_table = self.fetch_financial_quote(query)
        if financial_table:
            parts.append(financial_table)
            sources_scraped += 1

        for idx, r in enumerate(deduped, start=1):
            link = r.get("link", "")
            title = r.get("title", "")
            try:
                netloc = urllib.parse.urlparse(link).netloc.lower() or "unknown"
            except Exception:
                netloc = "unknown"

            raw_text = self.scrape_url(link) if link else ""
            if not raw_text or len(raw_text) < MIN_USEFUL_TEXT_CHARS:
                # Blocked / captcha / too-short redirect page.
                sources_blocked += 1
                continue

            filtered = self._relevance_filter(raw_text, keywords)
            if not filtered or len(filtered) < MIN_USEFUL_TEXT_CHARS // 2:
                sources_blocked += 1
                continue

            sources_scraped += 1
            parts.append(
                f"[SOURCE {sources_scraped}: {netloc}] {title}\n{filtered}"
            )
            sources_meta.append({
                "title": title,
                "link": link,
                "netloc": netloc,
                "chars": len(filtered),
            })

        context = "\n\n---\n\n".join(parts)
        return {
            "empty": sources_scraped == 0,
            "sources_scraped": sources_scraped,
            "sources_blocked": sources_blocked,
            "context": context,
            "sources": sources_meta,
        }

    def fetch_live_weather(self, query: str):
        """
        Extracts location from weather inquiry and fetches live real-time
        meteorological conditions from Open-Meteo and wttr.in.
        """
        query_lower = query.lower()
        weather_keywords = ["weather", "temperature", "rain", "climate", "forecast", "humidity", "wind", "temp"]
        if not any(kw in query_lower for kw in weather_keywords):
            return None

        # Extract target city/location name
        city = None
        match = re.search(r'(?:weather|temperature|forecast|rain|climate|humidity|temp)\s+(?:in|at|of|for)?\s*([a-zA-Z\s]+?)(?:\s+now|\s+today|\s+currently|\s*\?|$)', query, re.IGNORECASE)
        if match:
            city = match.group(1).strip()
        else:
            cleaned = re.sub(r'(how is the|what is the|tell me the|current|weather|condition|temperature|today|now|in|at|of|for|\?)', '', query, flags=re.IGNORECASE).strip()
            if cleaned and len(cleaned) >= 3:
                city = cleaned

        if not city:
            return None

        # 1. Try Open-Meteo (Global Open Meteorological Network)
        try:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1&language=en&format=json"
            geo_res = self._session.get(geo_url, timeout=3.5)
            if geo_res.ok:
                geo_data = geo_res.json()
                if "results" in geo_data and len(geo_data["results"]) > 0:
                    loc = geo_data["results"][0]
                    lat = loc.get("latitude")
                    lon = loc.get("longitude")
                    loc_name = f"{loc.get('name', city.title())}, {loc.get('admin1', '')} ({loc.get('country', '')})".replace(",  (", " (")

                    fc_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset&timezone=auto"
                    fc_res = self._session.get(fc_url, timeout=3.5)
                    if fc_res.ok:
                        fc_data = fc_res.json()
                        curr = fc_data.get("current", {})
                        daily = fc_data.get("daily", {})

                        wmo_codes = {
                            0: "Clear Sky ☀️", 1: "Mainly Clear 🌤️", 2: "Partly Cloudy ⛅", 3: "Overcast ☁️",
                            45: "Foggy 🌫️", 48: "Depositing Rime Fog 🌫️", 51: "Light Drizzle 🌦️", 53: "Moderate Drizzle 🌦️",
                            55: "Dense Drizzle 🌧️", 61: "Slight Rain 🌧️", 63: "Moderate Rain 🌧️", 65: "Heavy Rain 🌧️",
                            71: "Slight Snow ❄️", 73: "Moderate Snow ❄️", 75: "Heavy Snow ❄️",
                            80: "Slight Rain Showers 🌦️", 81: "Moderate Rain Showers 🌧️", 82: "Violent Rain Showers ⛈️",
                            95: "Thunderstorm ⛈️", 96: "Thunderstorm with Hail ⛈️"
                        }
                        w_desc = wmo_codes.get(curr.get("weather_code", 0), "Partly Cloudy ⛅")
                        temp_c = curr.get("temperature_2m", "N/A")
                        feels_c = curr.get("apparent_temperature", "N/A")
                        hum = curr.get("relative_humidity_2m", "N/A")
                        wind = curr.get("wind_speed_10m", "N/A")
                        precip = curr.get("precipitation", 0.0)

                        min_t = daily.get("temperature_2m_min", ["N/A"])[0]
                        max_t = daily.get("temperature_2m_max", ["N/A"])[0]
                        sunrise = str(daily.get("sunrise", ["N/A"])[0]).split("T")[-1]
                        sunset = str(daily.get("sunset", ["N/A"])[0]).split("T")[-1]

                        return (
                            f"### ⛅ Real-Time Meteorological Data: {loc_name}\n\n"
                            f"| Metric / Parameter | Live Value |\n"
                            f"| :--- | :--- |\n"
                            f"| **Current Condition** | {w_desc} |\n"
                            f"| **Temperature** | **{temp_c}°C** (Feels like: {feels_c}°C) |\n"
                            f"| **Today's Forecast Range** | Low: {min_t}°C / High: {max_t}°C |\n"
                            f"| **Relative Humidity** | {hum}% |\n"
                            f"| **Wind Speed** | {wind} km/h |\n"
                            f"| **Precipitation** | {precip} mm |\n"
                            f"| **Sunrise / Sunset** | 🌅 {sunrise} / 🌇 {sunset} |\n"
                        )
        except Exception as e:
            print(f"Open-Meteo weather fetch notice: {e}")

        # 2. Try wttr.in fallback
        try:
            w_url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
            w_res = self._session.get(w_url, timeout=3.5)
            if w_res.ok:
                w_data = w_res.json()
                curr = w_data.get("current_condition", [{}])[0]
                today = w_data.get("weather", [{}])[0]
                desc = curr.get("weatherDesc", [{}])[0].get("value", "Clear")
                temp_c = curr.get("temp_C", "N/A")
                feels_c = curr.get("FeelsLikeC", "N/A")
                hum = curr.get("humidity", "N/A")
                wind = curr.get("windspeedKmph", "N/A")
                min_t = today.get("mintempC", "N/A")
                max_t = today.get("maxtempC", "N/A")

                return (
                    f"### ⛅ Real-Time Weather Data: {city.title()}\n\n"
                    f"| Parameter | Live Reading |\n"
                    f"| :--- | :--- |\n"
                    f"| **Current Condition** | {desc} |\n"
                    f"| **Temperature** | **{temp_c}°C** (Feels like: {feels_c}°C) |\n"
                    f"| **Temperature Range** | Min: {min_t}°C / Max: {max_t}°C |\n"
                    f"| **Relative Humidity** | {hum}% |\n"
                    f"| **Wind Speed** | {wind} km/h |\n"
                )
        except Exception as e:
            print(f"wttr.in weather fetch notice: {e}")

        return None

    def fetch_financial_quote(self, query: str):
        """
        Detects stock, index, commodity, or crypto tickers and fetches
        real-time 5-day historical OHLC data from Yahoo Finance API.
        """
        import datetime
        q_lower = query.lower()

        # Known mapping for common companies / commodities / cryptos
        common_symbols = {
            "reliance": "RELIANCE.NS",
            "reliance industries": "RELIANCE.NS",
            "tcs": "TCS.NS",
            "tata consultancy": "TCS.NS",
            "infosys": "INFY.NS",
            "infy": "INFY.NS",
            "hdfc": "HDFCBANK.NS",
            "hdfc bank": "HDFCBANK.NS",
            "icici": "ICICIBANK.NS",
            "state bank of india": "SBIN.NS",
            "sbi": "SBIN.NS",
            "tata motors": "TATAMOTORS.NS",
            "apple": "AAPL",
            "aapl": "AAPL",
            "microsoft": "MSFT",
            "msft": "MSFT",
            "nvidia": "NVDA",
            "nvda": "NVDA",
            "tesla": "TSLA",
            "tsla": "TSLA",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "meta": "META",
            "bitcoin": "BTC-USD",
            "btc": "BTC-USD",
            "ethereum": "ETH-USD",
            "eth": "ETH-USD",
            "gold": "GC=F",
            "silver": "SI=F",
            "crude oil": "CL=F",
            "nifty": "^NSEI",
            "nifty 50": "^NSEI",
            "sensex": "^BSESN",
        }

        matched_symbol = None
        for name, sym in common_symbols.items():
            if name in q_lower:
                matched_symbol = sym
                break

        # If not in common symbols but query mentions "stock price" or "share price", try Yahoo search
        if not matched_symbol and any(k in q_lower for k in ["stock price", "share price", "stock of", "shares of", "ticker"]):
            clean = re.sub(r"(tell me the|what is the|stock price of|share price of|last \d+ days|lowest price in \d+ days|today|now|\?)", "", q_lower, flags=re.I).strip()
            if clean:
                try:
                    search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(clean)}&quotesCount=1&newsCount=0"
                    s_res = self._session.get(search_url, timeout=3.0)
                    if s_res.status_code == 200:
                        quotes = s_res.json().get("quotes", [])
                        if quotes:
                            matched_symbol = quotes[0].get("symbol")
                except Exception:
                    pass

        if not matched_symbol:
            return None

        try:
            chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{matched_symbol}?range=5d&interval=1d"
            res = self._session.get(chart_url, timeout=4.0)
            if res.status_code == 200:
                data = res.json()
                chart = data.get("chart", {}).get("result", [])
                if chart:
                    result = chart[0]
                    meta = result.get("meta", {})
                    symbol = meta.get("symbol", matched_symbol)
                    currency = meta.get("currency", "INR")
                    regular_price = meta.get("regularMarketPrice")
                    timestamps = result.get("timestamp", [])
                    quote = result.get("indicators", {}).get("quote", [{}])[0]
                    opens = quote.get("open", [])
                    highs = quote.get("high", [])
                    lows = quote.get("low", [])
                    closes = quote.get("close", [])

                    rows = []
                    valid_lows = []
                    valid_highs = []
                    valid_closes = []

                    for t, o, h, l, c in zip(timestamps, opens, highs, lows, closes):
                        if o is not None and c is not None and l is not None and h is not None:
                            dt = datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d")
                            rows.append(f"| {dt} | {o:.2f} | {h:.2f} | {l:.2f} | {c:.2f} |")
                            valid_lows.append(l)
                            valid_highs.append(h)
                            valid_closes.append(c)

                    if rows:
                        recent_rows = rows[-3:] if len(rows) >= 3 else rows
                        recent_lows = valid_lows[-3:] if len(valid_lows) >= 3 else valid_lows
                        lowest_3d = min(recent_lows) if recent_lows else "N/A"
                        highest_3d = max(valid_highs[-3:]) if valid_highs else "N/A"
                        latest_close = valid_closes[-1] if valid_closes else regular_price

                        table_md = (
                            f"[LIVE FINANCIAL MARKET DATA: {symbol}]\n"
                            f"Currency: {currency} | Latest Current Price: {latest_close:.2f} {currency}\n"
                            f"Lowest Price (Last 3 Trading Days): {lowest_3d:.2f} {currency}\n"
                            f"Highest Price (Last 3 Trading Days): {highest_3d:.2f} {currency}\n\n"
                            f"| Date | Open ({currency}) | High ({currency}) | Low ({currency}) | Close ({currency}) |\n"
                            f"| :--- | :--- | :--- | :--- | :--- |\n"
                            + "\n".join(recent_rows)
                        )
                        return table_md
        except Exception as e:
            print(f"Financial quote fetch error for {matched_symbol}: {e}")
        return None


if __name__ == "__main__":
    # Test search
    ws = WebSearch()
    res = ws.search("Intel Core i5-1235u Xe graphics specification", 3)
    print(json.dumps(res, indent=2))

