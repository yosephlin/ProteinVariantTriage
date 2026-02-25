import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .cache import cache_key, load_cache, make_meta, save_cache

DEFAULT_TIMEOUT = 30

class OfflineError(RuntimeError):
    pass

def fetch_json(
    url: str,
    provider: str,
    cache_dir: Path,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    offline: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 4,
    backoff_base: float = 1.0,
) -> Tuple[Any, Dict[str, Any]]:
    key = cache_key(url, params)
    cached = load_cache(cache_dir, provider, key)
    if cached is not None:
        return cached
    if offline:
        raise OfflineError(f"Offline mode enabled and no cache entry for {provider}:{key}")

    full_url = url
    if params:
        full_url = f"{url}?{urlencode(params)}"

    base_headers = {"User-Agent": "swarm-context-pack", "Accept": "application/json"}
    if headers:
        base_headers.update(headers)
    req = Request(full_url, headers=base_headers)
    attempt = 0
    while True:
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                meta = make_meta(url, params, resp.status, dict(resp.headers))
                save_cache(cache_dir, provider, key, data, meta)
                return data, meta
        except HTTPError as e:
            status = getattr(e, "code", None)
            retry_after = None
            if hasattr(e, "headers") and e.headers:
                try:
                    retry_after = float(e.headers.get("Retry-After"))
                except Exception:
                    retry_after = None
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = retry_after if retry_after is not None else backoff_base * (2 ** attempt)
                time.sleep(wait)
                attempt += 1
                continue
            raise
        except URLError:
            if attempt < max_retries:
                time.sleep(backoff_base * (2 ** attempt))
                attempt += 1
                continue
            raise
