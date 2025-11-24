import pandas as pd
import os

"""
Script para pré-processar o CSV original (leitura em chunks):
- Filtra apenas registros de CURITIBA (PR)
- Remove colunas desnecessárias
- Normaliza nomes de bairros
- Salva em novo CSV comprimido para leitura rápida
"""

INPUT_CSV = 'enderecos_empresas_wnames_codibge.csv'
OUTPUT_CSV = 'enderecos_curitiba_filtered.csv'

if os.path.exists(OUTPUT_CSV):
    print(f'Arquivo {OUTPUT_CSV} já existe. Pulando pré-processamento.')
else:
    print(f'Lendo {INPUT_CSV} em chunks...')
    
    chunks = []
    chunk_size = 50000
    
    # Iterar sobre o arquivo original
    for chunk in pd.read_csv(INPUT_CSV, sep=',', dtype=str, encoding='utf-8', low_memory=False, chunksize=chunk_size):
        # Normalizando para maiúsculo para garantir o match
        if 'municipio' in chunk.columns and 'uf' in chunk.columns:
            chunk['municipio'] = chunk['municipio'].str.upper().str.strip()
            chunk['uf'] = chunk['uf'].str.upper().str.strip()
            
            # Filtra UF = PR e MUNICIPIO = CURITIBA
            chunk_cwb = chunk[(chunk['uf'] == 'PR') & (chunk['municipio'] == 'CURITIBA')].copy()
            
            # Manter apenas colunas necessárias se elas existirem
            cols_existentes = [c for c in ['cnpj', 'bairro', 'cep', 'municipio'] if c in chunk_cwb.columns]
            chunk_cwb = chunk_cwb[cols_existentes]
            
            chunks.append(chunk_cwb)
            print(f'  Processados {len(chunks) * chunk_size} registros (aprox)...')
    
    if chunks:
        df_cwb = pd.concat(chunks, ignore_index=True)
        print(f'Registros brutos de Curitiba encontrados: {len(df_cwb)}')
        
        # Remover duplicatas por CNPJ
        if 'cnpj' in df_cwb.columns:
            df_cwb = df_cwb.drop_duplicates(subset=['cnpj'], keep='first')
        
        # Limpar bairros (remover vazios ou genéricos)
        if 'bairro' in df_cwb.columns:
            df_cwb = df_cwb.dropna(subset=['bairro'])
            # Padronizar Capitalização (Ex: "CENTRO" -> "Centro")
            df_cwb['bairro'] = df_cwb['bairro'].str.strip().str.title()
            # Filtro básico para remover sujeira (bairros com nomes muito curtos geralmente são erros)
            df_cwb = df_cwb[df_cwb['bairro'].str.len() > 2]
        
        print(f'Registros finais após limpeza: {len(df_cwb)}')
        
        # Salvar
        df_cwb.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f'Salvo em {OUTPUT_CSV}')
        print(f'Tamanho do arquivo: {os.path.getsize(OUTPUT_CSV) / 1024 / 1024:.2f} MB')
    else:
        print("Nenhum registro de Curitiba encontrado ou erro na leitura.")