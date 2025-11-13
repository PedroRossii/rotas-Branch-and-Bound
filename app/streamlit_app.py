import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from src.data_processing import load_and_aggregate, get_raw_data
from src.geocoding import geocode_municipalities, build_distance_matrix_from_coords
from src.heuristics import nearest_neighbor
from src.bb_tsp import branch_and_bound_tsp

# Adiciona o diretório raiz ao path (corrigido)
# Isso é mais robusto se o script está em 'app/streamlit_app.py'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

st.set_page_config(layout='wide', page_title='Sistema de Otimização de Rotas - Branch and Bound')

st.title('🗺️ Sistema de Otimização de Rotas com Branch and Bound')
st.markdown('**Problema do Caixeiro Viajante (TSP) aplicado a municípios do Paraná**')

DATA_PATH = 'enderecos_pr_filtered.csv'

# Sidebar
with st.sidebar:
    st.header('📋 Navegação')
    section = st.radio(
        'Escolha a seção:',
        ['📊 EDA', '⚙️ Otimização', '📈 Comparação', '🔬 Sensibilidade']
    )
    
    st.markdown('---')
    st.info("""
    **Sistema de Otimização** Branch and Bound + TSP   
    Dataset: Empresas do PR
    """)

# SEÇÃO EDA
if section == '📊 EDA':
    st.header('Análise Exploratória de Dados')
    
    try:
        agg = load_and_aggregate(DATA_PATH)
        raw = get_raw_data(DATA_PATH)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Municípios', f"{len(agg):,}")
        col2.metric('Empresas', f"{len(raw):,}")
        col3.metric('Máximo/Município', f"{agg['count'].max():,}")
        col4.metric('Média/Município', f"{agg['count'].mean():.0f}")
        
        st.subheader('Top 20 Municípios')
        st.dataframe(agg.head(20), use_container_width=True)
        
        tab1, tab2, tab3 = st.tabs(['Histograma', 'Boxplot', 'Estatísticas'])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(agg['count'], bins=50, color='steelblue', edgecolor='black')
            ax.set_xlabel('Número de Empresas')
            ax.set_ylabel('Frequência')
            ax.set_title('Distribuição de Empresas por Município')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.boxplot(agg['count'], vert=False, patch_artist=True)
            ax.set_xlabel('Número de Empresas')
            ax.set_title('Boxplot: Identificação de Outliers')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with tab3:
            stats = agg['count'].describe()
            st.dataframe(stats, use_container_width=True)
            
            percentis = {
                '25%': agg['count'].quantile(0.25),
                '50%': agg['count'].quantile(0.50),
                '75%': agg['count'].quantile(0.75),
                '90%': agg['count'].quantile(0.90),
                '95%': agg['count'].quantile(0.95)
            }
            st.write('Percentis:', percentis)
    
    except Exception as e:
        st.error(f'Erro ao carregar dados EDA: {e}')
        st.error(f"Verifique se o arquivo '{DATA_PATH}' está no diretório correto.")


# SEÇÃO OTIMIZAÇÃO
elif section == '⚙️ Otimização':
    st.header('Execução do Algoritmo Branch and Bound')

    # --- 1. Inicialização do Estado da Sessão ---
    # Isso garante que as variáveis existam
    if 'run_complete' not in st.session_state:
        st.session_state.run_complete = False
        st.session_state.results = {}

    # --- 2. Callback para Limpar o Estado ---
    # Será chamada se os sliders mudarem
    def reset_run_state():
        st.session_state.run_complete = False
        st.session_state.results = {}

    # --- 3. Controles (Sliders) ---
    col1, col2 = st.columns(2)
    with col1:
        # Adiciona o on_change para limpar resultados antigos se o slider mudar
        sample_size = st.slider('Municípios', 4, 20, 8, on_change=reset_run_state)
    with col2:
        time_limit = st.slider('Tempo Limite (s)', 5, 300, 30, on_change=reset_run_state)
    
    if sample_size > 14:
        st.warning('⚠️ >14 municípios pode não encontrar ótimo no tempo limite')
    
    # --- 4. Botão de Execução (APENAS CÁLCULO) ---
    if st.button('▶️ Executar', type='primary', use_container_width=True):
        try:
            with st.spinner('Carregando dados...'):
                agg = load_and_aggregate(DATA_PATH)
                sample = agg.head(sample_size).reset_index(drop=True)
            
            st.success(f'✅ {sample_size} municípios carregados')
            
            with st.expander('Ver municípios selecionados'):
                st.dataframe(sample)
            
            with st.spinner('Geocodificando...'):
                coords = geocode_municipalities(sample)
                dist_mat, municipios = build_distance_matrix_from_coords(coords)
            
            st.success('✅ Geocodificação completa')
            
            # Nearest Neighbor
            t0 = time.time()
            tour_h, cost_h = nearest_neighbor(dist_mat, start=0)
            time_nn = time.time() - t0
            
            # Branch and Bound
            with st.spinner(f'Executando B&B (limite: {time_limit}s)...'):
                res = branch_and_bound_tsp(dist_mat, time_limit=time_limit)
            
            # --- 5. Salvar Resultados no Estado da Sessão ---
            st.session_state.results = {
                'cost_h': cost_h,
                'tour_h': tour_h,
                'time_nn': time_nn,
                'res': res,
                'coords': coords,
                'municipios': municipios,
                'coords_df': coords.set_index('municipio').reindex(municipios).reset_index()
            }
            st.session_state.run_complete = True # Sinaliza que a execução terminou

        except Exception as e:
            st.error(f'Erro durante a execução: {e}')
            st.session_state.run_complete = False # Falha na execução

    # --- 6. Bloco de Exibição Persistente ---
    # Este bloco é executado SE os resultados existirem no state
    if st.session_state.run_complete:
        
        # AVISO DE SUCESSO MOVIDO PARA CÁ, PARA SÓ APARECER UMA VEZ
        st.success('✅ Execução Concluída! Resultados disponíveis na aba "Comparação".')

        # --- TODO O BLOCO DE RESULTADOS FOI MOVIDO PARA A ABA "COMPARAÇÃO" ---
        # (O código que estava aqui, de métricas e mapa, foi removido)


# SEÇÃO COMPARAÇÃO
elif section == '📈 Comparação':
    st.header('Comparação de Desempenho')
    
    # --- ALTERAÇÃO PRINCIPAL: VERIFICAR O SESSION STATE ---
    # Verifica se a otimização já foi executada com sucesso
    if 'run_complete' in st.session_state and st.session_state.run_complete:
        st.success("Usando dados reais da última execução da 'Otimização'!")
        
        # Recupera os dados do session_state
        results = st.session_state.results
        cost_h = results['cost_h']
        tour_h = results['tour_h'] # Necessário para métricas e mapa
        time_nn = results['time_nn']
        res = results['res']
        municipios = results['municipios'] # Necessário para o mapa
        coords_df = results['coords_df'] # Necessário para o mapa

        # --- CÓDIGO MOVIDO DA ABA "OTIMIZAÇÃO" ---
        # Nearest Neighbor
        st.subheader('🎯 Heurística Nearest Neighbor')
        col1, col2, col3 = st.columns(3)
        col1.metric('Custo', f'{cost_h:.2f} km')
        col2.metric('Tempo', f'{time_nn:.4f} s')
        col3.metric('Tour', f"{len(tour_h)} nós")
        
        # Branch and Bound
        st.subheader('🌳 Branch and Bound')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Custo Ótimo', f'{res.best_cost:.2f} km')
        col2.metric('Nós Expandidos', f'{res.nodes_expanded:,}')
        col3.metric('Profundidade', res.max_depth)
        col4.metric('Tempo', f'{res.time_seconds:.2f} s')
        # --- FIM DO CÓDIGO MOVIDO ---

        # Cria o DataFrame com dados reais
        real_data = pd.DataFrame({
            'Algoritmo': ['Nearest Neighbor', 'Branch and Bound'],
            'Custo (km)': [cost_h, res.best_cost],
            'Tempo (s)': [time_nn, res.time_seconds],
            'Nós Expandidos': [0, res.nodes_expanded] # NN não expande nós
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            # Plota os dados reais
            ax.bar(real_data['Algoritmo'], real_data['Custo (km)'], color=['orange', 'green'])
            ax.set_ylabel('Custo (km)')
            ax.set_title('Comparação de Custo (Resultados Reais)')
            for i, v in enumerate(real_data['Custo (km)']):
                ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            # Plota os dados reais
            ax.bar(real_data['Algoritmo'], real_data['Tempo (s)'], color=['orange', 'green'])
            ax.set_ylabel('Tempo (s)')
            ax.set_title('Comparação de Tempo (Resultados Reais)')
            # Adiciona escala de log para melhor visualização, pois a diferença é grande
            ax.set_yscale('log')
            for i, v in enumerate(real_data['Tempo (s)']):
                ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
            st.pyplot(fig)
        
        # Mostra a tabela de dados reais
        st.dataframe(real_data, use_container_width=True)

        # --- CÓDIGO DO MAPA MOVIDO DA ABA "OTIMIZAÇÃO" ---
        st.subheader('🗺️ Visualização no Mapa')
        valid = coords_df.dropna(subset=['latitude', 'longitude'])
        
        if len(valid) >= 2:
            center_lat = valid['latitude'].mean()
            center_lon = valid['longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
            
            for _, row in valid.iterrows():
                folium.Marker([row['latitude'], row['longitude']], 
                                popup=row['municipio']).add_to(m)
            
            # Rota NN
            coords_nn = []
            for i in tour_h:
                if i < len(municipios):
                    r = coords_df[coords_df['municipio'] == municipios[i]]
                    if not r.empty and pd.notna(r.iloc[0]['latitude']):
                        coords_nn.append([r.iloc[0]['latitude'], r.iloc[0]['longitude']])
            if coords_nn:
                folium.PolyLine(coords_nn, color='orange', weight=4).add_to(m)
            
            # Rota B&B
            if res.best_tour:
                coords_bb = []
                for i in res.best_tour:
                    if i < len(municipios):
                        r = coords_df[coords_df['municipio'] == municipios[i]]
                        if not r.empty and pd.notna(r.iloc[0]['latitude']):
                            coords_bb.append([r.iloc[0]['latitude'], r.iloc[0]['longitude']])
                if coords_bb:
                    folium.PolyLine(coords_bb, color='green', weight=6).add_to(m)
            
            st.markdown('**Legenda:** 🟠 NN | 🟢 B&B')
            st_folium(m, width=1200, height=600)
        # --- FIM DO CÓDIGO DO MAPA ---

    else:
        # Se a otimização não foi executada, mostra os dados de exemplo
        st.info("Execute a '⚙️ Otimização' primeiro para gerar dados reais. Mostrando dados de exemplo.")
        
        exemplo = pd.DataFrame({
            'Algoritmo': ['Nearest Neighbor', 'Branch and Bound'],
            'Custo (km)': [845.3, 710.5],
            'Tempo (s)': [0.002, 18.5],
            'Nós Expandidos': [0, 8342]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(exemplo['Algoritmo'], exemplo['Custo (km)'], color=['orange', 'green'])
            ax.set_ylabel('Custo (km)')
            ax.set_title('Comparação de Custo (Exemplo)')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(exemplo['Algoritmo'], exemplo['Tempo (s)'], color=['orange', 'green'])
            ax.set_ylabel('Tempo (s)')
            ax.set_title('Comparação de Tempo (Exemplo)')
            st.pyplot(fig)
        
        st.dataframe(exemplo, use_container_width=True)

# SEÇÃO SENSIBILIDADE
elif section == '🔬 Sensibilidade':
    st.header('Análise de Sensibilidade')
    
    tab1, tab2 = st.tabs(['Tempo Limite', 'Tamanho'])
    
    with tab1:
        st.subheader('Impacto do Tempo Limite')
        
        test_size = st.number_input('Municípios', 4, 12, 8, key='sens_time_size')
        time_limits = st.multiselect('Tempos (s)', [5, 10, 30, 60, 120], [10, 30, 60], key='sens_time_limits')
        
        if st.button('Testar Tempos'):
            if not time_limits:
                st.warning('Selecione pelo menos um tempo')
            else:
                results = []
                agg = load_and_aggregate(DATA_PATH)
                sample = agg.head(test_size).reset_index(drop=True)
                coords = geocode_municipalities(sample)
                dist_mat, municipios = build_distance_matrix_from_coords(coords)
                
                progress = st.progress(0)
                for i, tl in enumerate(sorted(time_limits)):
                    st.text(f'Testando {tl}s...')
                    res = branch_and_bound_tsp(dist_mat, time_limit=tl)
                    results.append({
                        'Tempo Limite (s)': tl,
                        'Custo (km)': res.best_cost,
                        'Nós Expandidos': res.nodes_expanded,
                        'Tempo Real (s)': res.time_seconds
                    })
                    progress.progress((i + 1) / len(time_limits))
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_results['Tempo Limite (s)'], df_results['Custo (km)'], 
                        marker='o', linewidth=2)
                ax.set_xlabel('Tempo Limite (s)')
                ax.set_ylabel('Custo (km)')
                ax.set_title('Custo vs Tempo Limite')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
    
    with tab2:
        st.subheader('Impacto do Tamanho')
        
        time_fixed = st.slider('Tempo fixo (s)', 10, 120, 60, key='sens_size_time')
        sizes = st.multiselect('Tamanhos', [4, 6, 8, 10, 12], [4, 8, 12], key='sens_size_sizes')
        
        if st.button('Testar Tamanhos'):
            if not sizes:
                st.warning('Selecione pelo menos um tamanho')
            else:
                results = []
                agg = load_and_aggregate(DATA_PATH)
                
                progress = st.progress(0)
                for i, sz in enumerate(sorted(sizes)):
                    st.text(f'Testando n={sz}...')
                    sample = agg.head(sz).reset_index(drop=True)
                    coords = geocode_municipalities(sample)
                    dist_mat, municipios = build_distance_matrix_from_coords(coords)
                    
                    res = branch_and_bound_tsp(dist_mat, time_limit=time_fixed)
                    results.append({
                        'Municípios': sz,
                        'Custo (km)': res.best_cost,
                        'Nós Expandidos': res.nodes_expanded,
                        'Tempo (s)': res.time_seconds
                    })
                    progress.progress((i + 1) / len(sizes))
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_results['Municípios'], df_results['Nós Expandidos'], 
                        marker='o', linewidth=2)
                ax.set_xlabel('Número de Municípios')
                ax.set_ylabel('Nós Expandidos')
                ax.set_title('Escalabilidade: Nós Expandidos vs Tamanho')
                ax.set_yscale('log')
                ax.grid(alpha=0.3)
                st.pyplot(fig)