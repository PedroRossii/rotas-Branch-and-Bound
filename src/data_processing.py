import pandas as pd
import os


def load_and_aggregate(path='enderecos_pr_filtered.csv', uf_filter=None):
    """Carrega CSV já filtrado (PR) e retorna DataFrame agregado por município.

    Se arquivo filtrado não existe, tenta carregar arquivo original e filtrar.
    Retorna DataFrame com colunas: cod_ibge (int), municipio (str), count (int)
    """
    # Tentar carregar arquivo filtrado (mais rápido)
    if os.path.exists(path) and 'filtered' in path:
        df = pd.read_csv(path, sep=',', encoding='utf-8')
        print(f'Carregado CSV filtrado: {path}')
    else:
        # Carregar arquivo original e filtrar
        original_path = 'enderecos_empresas_wnames_codibge.csv'
        if not os.path.exists(original_path):
            raise FileNotFoundError(f'Arquivo {original_path} não encontrado')
        df = pd.read_csv(original_path, sep=',', dtype=str, encoding='utf-8', low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if uf_filter and 'uf' in df.columns:
            df = df[df['uf'] == uf_filter]
    
    # Garantir cod_ibge como int
    if 'cod_ibge' in df.columns:
        df['cod_ibge'] = pd.to_numeric(df['cod_ibge'], errors='coerce')
    
    # Agrupar por município
    agg = df.groupby(['cod_ibge', 'municipio'], as_index=False).size().reset_index(drop=True)
    agg.columns = ['cod_ibge', 'municipio', 'count']
    agg = agg.dropna(subset=['cod_ibge'])
    agg['cod_ibge'] = agg['cod_ibge'].astype(int)
    agg = agg.sort_values('count', ascending=False).reset_index(drop=True)
    return agg


def get_raw_data(path='enderecos_pr_filtered.csv'):
    """Carrega dados brutos (PR) sem agregar."""
    if os.path.exists(path) and 'filtered' in path:
        return pd.read_csv(path, sep=',', encoding='utf-8')
    else:
        original_path = 'enderecos_empresas_wnames_codibge.csv'
        df = pd.read_csv(original_path, sep=',', dtype=str, encoding='utf-8', low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        return df[df['uf'] == 'PR'].copy()
