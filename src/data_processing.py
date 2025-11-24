import pandas as pd
import os

def load_and_aggregate(path='enderecos_curitiba_filtered.csv'):
    """
    Carrega CSV de Curitiba e retorna DataFrame agregado por BAIRRO.
    Retorna: DataFrame com colunas ['bairro', 'count']
    """
    if not os.path.exists(path):
        # Fallback para tentar achar o arquivo antigo se o nome mudou
        if os.path.exists('enderecos_pr_filtered.csv'):
            path = 'enderecos_pr_filtered.csv'
        else:
            raise FileNotFoundError(f'Arquivo {path} não encontrado')
            
    df = pd.read_csv(path, sep=',', encoding='utf-8')
    
    # Agrupar por BAIRRO
    # Contamos quantas empresas existem em cada bairro
    agg = df.groupby(['bairro'], as_index=False).size().reset_index(drop=True)
    agg.columns = ['bairro', 'count']
    
    # Remove bairros inválidos ou vazios se houver
    agg = agg[agg['bairro'].str.len() > 2]
    
    # Ordenar pelos bairros com mais empresas (centros comerciais)
    agg = agg.sort_values('count', ascending=False).reset_index(drop=True)
    
    return agg

def get_raw_data(path='enderecos_curitiba_filtered.csv'):
    """Carrega dados brutos."""
    if os.path.exists(path):
        return pd.read_csv(path, sep=',', encoding='utf-8')
    else:
        # Fallback
        return pd.read_csv('enderecos_pr_filtered.csv', sep=',', encoding='utf-8')