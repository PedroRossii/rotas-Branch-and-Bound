import os
import requests
import pandas as pd
import time
from typing import Tuple, Optional

CACHE_CSV = 'geocode_cache.csv'
# Lê chave da variável de ambiente quando disponível; em última instância usa a chave embutida
GOOGLE_GEOCODING_API_KEY = 'AIzaSyAwKapaVOhpPdaE7rQdvAQ9eCWRQ-Vrll8'

# Carrega cache do disco (se existir)
def _load_cache():
    cache = {}
    if os.path.exists(CACHE_CSV):
        try:
            df = pd.read_csv(CACHE_CSV, dtype={'cache_key': str, 'latitude': float, 'longitude': float})
            for _, r in df.iterrows():
                key = r['cache_key']
                lat = r['latitude'] if not pd.isna(r['latitude']) else None
                lon = r['longitude'] if not pd.isna(r['longitude']) else None
                cache[key] = (lat, lon) if (lat is not None and lon is not None) else None
        except Exception:
            # arquivo possivelmente corrompido — ignorar
            pass
    return cache


def _save_cache(cache: dict):
    rows = []
    for key, val in cache.items():
        lat = val[0] if val is not None else None
        lon = val[1] if val is not None else None
        rows.append({'cache_key': key, 'latitude': lat, 'longitude': lon})
    df = pd.DataFrame(rows)
    df.to_csv(CACHE_CSV, index=False)


GEOCODING_CACHE = _load_cache()


def geocode_address(address: str, city: str = '', state: str = 'PR', sleep: float = 0.5) -> Optional[Tuple[float, float]]:
    """
    Geocodifica um endereço usando Google Geocoding API.
    Retorna (lat, lon) ou None se falhar. Usa cache em memória e salva em disco.
    """
    cache_key = f"{address}|{city}|{state}".strip()
    if cache_key in GEOCODING_CACHE:
        return GEOCODING_CACHE[cache_key]

    query = f"{address}, {city}, {state}, Brasil" if address else f"{city}, {state}, Brasil"
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': query,
        'key': GOOGLE_GEOCODING_API_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        if data.get('status') == 'OK' and data.get('results'):
            loc = data['results'][0]['geometry']['location']
            result = (loc['lat'], loc['lng'])
            GEOCODING_CACHE[cache_key] = result
            _save_cache(GEOCODING_CACHE)
            time.sleep(sleep)
            return result
        else:
            GEOCODING_CACHE[cache_key] = None
            _save_cache(GEOCODING_CACHE)
            time.sleep(sleep)
            return None
    except Exception as e:
        # registrar e retornar None
        print(f"Erro geocodificando {query}: {e}")
        GEOCODING_CACHE[cache_key] = None
        _save_cache(GEOCODING_CACHE)
        time.sleep(sleep)
        return None


def geocode_municipalities(municipios_df: pd.DataFrame, sleep: float = 0.5) -> pd.DataFrame:
    """
    Geocodifica municípios únicos e retorna DataFrame com (municipio, latitude, longitude).
    Usa cache para evitar requests repetidas e salva cache em disco.
    municipios_df: DataFrame com colunas ['municipio', 'cod_ibge', ...]
    """
    unique_municipios = municipios_df[['municipio']].drop_duplicates().reset_index(drop=True)
    print(f'Geocodificando {len(unique_municipios)} municípios...')

    coords = []
    for idx, row in unique_municipios.iterrows():
        municipio = row['municipio']
        lat_lon = geocode_address('', city=municipio, state='PR', sleep=sleep)
        if lat_lon:
            coords.append({'municipio': municipio, 'latitude': lat_lon[0], 'longitude': lat_lon[1]})
        else:
            coords.append({'municipio': municipio, 'latitude': None, 'longitude': None})
        if (idx + 1) % 10 == 0:
            print(f'  {idx + 1} / {len(unique_municipios)} geocodificados')

    # garantir que cache foi salva
    _save_cache(GEOCODING_CACHE)
    return pd.DataFrame(coords)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula distância em km entre dois pontos (lat, lon) usando fórmula haversine.
    """
    import math
    R = 6371.0  # raio da Terra em km
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def build_distance_matrix_from_coords(municipios_coords_df: pd.DataFrame) -> tuple:
    """
    Constrói matriz de distâncias a partir de coordenadas.
    Retorna (matriz numpy, lista de municípios em ordem).
    """
    import numpy as np
    n = len(municipios_coords_df)
    mat = np.zeros((n, n), dtype=float)
    municipios = municipios_coords_df['municipio'].tolist()

    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = municipios_coords_df.iloc[i][['latitude', 'longitude']].values
            lat2, lon2 = municipios_coords_df.iloc[j][['latitude', 'longitude']].values
            if pd.notna(lat1) and pd.notna(lon1) and pd.notna(lat2) and pd.notna(lon2):
                dist = haversine_distance(lat1, lon1, lat2, lon2)
            else:
                dist = float('inf')
            mat[i, j] = dist
            mat[j, i] = dist

    return mat, municipios
