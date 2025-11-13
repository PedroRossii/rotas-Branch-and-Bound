import pandas as pd
import os

"""
Script para pré-processar o CSV original (leitura em chunks):
- Filtra apenas registros com UF == 'PR'
- Remove colunas desnecessárias
- Salva em novo CSV comprimido para leitura rápida
"""

INPUT_CSV = 'enderecos_empresas_wnames_codibge.csv'
OUTPUT_CSV = 'enderecos_pr_filtered.csv'

if os.path.exists(OUTPUT_CSV):
    print(f'Arquivo {OUTPUT_CSV} já existe. Pulando pré-processamento.')
else:
    print(f'Lendo {INPUT_CSV} em chunks...')
    
    chunks = []
    chunk_size = 50000
    
    for chunk in pd.read_csv(INPUT_CSV, sep=',', dtype=str, encoding='utf-8', low_memory=False, chunksize=chunk_size):
        # Filtrar PR
        chunk_pr = chunk[chunk['uf'] == 'PR'].copy()
        
        # Manter apenas colunas necessárias
        colunas_manter = ['cnpj', 'logradouro', 'numero', 'complemento', 'bairro', 'cep', 'municipio', 'cod_ibge']
        chunk_pr = chunk_pr[colunas_manter]
        
        chunks.append(chunk_pr)
        print(f'  Processados {len(chunks) * chunk_size} registros...')
    
    df_pr = pd.concat(chunks, ignore_index=True)
    print(f'Registros com UF=PR: {len(df_pr)}')
    
    # Remover duplicatas por CNPJ
    df_pr = df_pr.drop_duplicates(subset=['cnpj'], keep='first')
    print(f'Registros após remover duplicatas: {len(df_pr)}')
    
    # Salvar
    df_pr.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f'Salvo em {OUTPUT_CSV}')
    print(f'Tamanho do arquivo: {os.path.getsize(OUTPUT_CSV) / 1024 / 1024:.2f} MB')
