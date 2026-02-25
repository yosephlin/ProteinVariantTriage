import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

def _stable_param_string(params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return ""
    pairs = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, (list, tuple)):
            for item in v:
                pairs.append((k, str(item)))
        else:
            pairs.append((k, str(v)))
    return urlencode(pairs)

def cache_key(url: str, params: Optional[Dict[str, Any]] = None) -> str:
    param_str = _stable_param_string(params)
    full = url if not param_str else f"{url}?{param_str}"
    return hashlib.sha1(full.encode("utf-8")).hexdigest()

def cache_paths(cache_dir: Path, provider: str, key: str) -> Tuple[Path, Path]:
    base = cache_dir / provider
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{key}.json", base / f"{key}.meta.json"

def load_cache(cache_dir: Path, provider: str, key: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
    data_path, meta_path = cache_paths(cache_dir, provider, key)
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        data = json.loads(data_path.read_text())
        meta = json.loads(meta_path.read_text())
        return data, meta
    except Exception:
        return None

def save_cache(cache_dir: Path, provider: str, key: str, data: Any, meta: Dict[str, Any]) -> None:
    data_path, meta_path = cache_paths(cache_dir, provider, key)
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

def make_meta(url: str, params: Optional[Dict[str, Any]], status_code: int, headers: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "url": url,
        "params": params or {},
        "status_code": status_code,
        "timestamp": time.time(),
        "headers": headers or {},
    }
