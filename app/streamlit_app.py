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

# Adiciona o diretório raiz ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

st.set_page_config(layout='wide', page_title='Otimização de Rotas - Curitiba')

st.title('🗺️ Sistema de Otimização de Rotas - Curitiba')
st.markdown('**Problema do Caixeiro Viajante (TSP) aplicado a Bairros de Curitiba**')

DATA_PATH = 'enderecos_curitiba_filtered.csv'

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
    Escopo: Curitiba (Bairros)
    """)

# SEÇÃO EDA
if section == '📊 EDA':
    st.header('Análise de Bairros de Curitiba')
    
    try:
        agg = load_and_aggregate(DATA_PATH)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Total Bairros', f"{len(agg):,}")
        total_empresas = agg['count'].sum()
        col2.metric('Total Empresas', f"{total_empresas:,}")
        col3.metric('Máximo/Bairro', f"{agg['count'].max():,}")
        col4.metric('Média/Bairro', f"{agg['count'].mean():.0f}")
        
        st.subheader('Top 20 Bairros com mais Empresas')
        st.dataframe(agg.head(20), use_container_width=True)
        
        tab1, tab2, tab3 = st.tabs(['Histograma', 'Boxplot', 'Estatísticas'])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(agg['count'], bins=30, color='steelblue', edgecolor='black')
            ax.set_xlabel('Número de Empresas')
            ax.set_ylabel('Frequência')
            ax.set_title('Distribuição de Empresas por Bairro')
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
        st.warning(f"Verifique se o arquivo '{DATA_PATH}' foi gerado pelo preprocess.py.")


# SEÇÃO OTIMIZAÇÃO
elif section == '⚙️ Otimização':
    st.header('Execução do Algoritmo Branch and Bound')

    if 'run_complete' not in st.session_state:
        st.session_state.run_complete = False
        st.session_state.results = {}

    def reset_run_state():
        st.session_state.run_complete = False
        st.session_state.results = {}

    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.slider('Quantidade de Bairros', 4, 20, 8, on_change=reset_run_state)
    with col2:
        time_limit = st.slider('Tempo Limite (s)', 5, 300, 30, on_change=reset_run_state)
    
    if sample_size > 14:
        st.warning('⚠️ >14 bairros pode não encontrar ótimo no tempo limite')
    
    if st.button('▶️ Executar', type='primary', use_container_width=True):
        try:
            with st.spinner('Carregando dados...'):
                agg = load_and_aggregate(DATA_PATH)
                sample = agg.head(sample_size).reset_index(drop=True)
            
            st.success(f'✅ {sample_size} bairros carregados')
            
            with st.expander('Ver bairros selecionados'):
                st.dataframe(sample)
            
            with st.spinner('Geocodificando bairros...'):
                coords = geocode_municipalities(sample)
                dist_mat, locais = build_distance_matrix_from_coords(coords)
            
            st.success('✅ Geocodificação completa')
            
            # Nearest Neighbor
            t0 = time.time()
            tour_h, cost_h = nearest_neighbor(dist_mat, start=0)
            time_nn = time.time() - t0
            
            # Branch and Bound
            with st.spinner(f'Executando B&B (limite: {time_limit}s)...'):
                res = branch_and_bound_tsp(dist_mat, time_limit=time_limit)
            
            st.session_state.results = {
                'cost_h': cost_h,
                'tour_h': tour_h,
                'time_nn': time_nn,
                'res': res,
                'coords': coords,
                'locais': locais,
                'coords_df': coords.set_index('bairro').reindex(locais).reset_index()
            }
            st.session_state.run_complete = True

        except Exception as e:
            st.error(f'Erro durante a execução: {e}')
            st.session_state.run_complete = False

    if st.session_state.run_complete:
        st.success('✅ Execução Concluída! Resultados disponíveis na aba "Comparação".')


# SEÇÃO COMPARAÇÃO
elif section == '📈 Comparação':
    st.header('Comparação de Desempenho')
    
    if 'run_complete' in st.session_state and st.session_state.run_complete:
        st.success("Usando dados reais da última execução!")
        
        results = st.session_state.results
        cost_h = results['cost_h']
        tour_h = results['tour_h']
        time_nn = results['time_nn']
        res = results['res']
        locais = results['locais']
        coords_df = results['coords_df']

        # Métricas
        st.subheader('Resultados Detalhados')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('NN Custo', f'{cost_h:.2f} km')
        col2.metric('B&B Custo (Ótimo)', f'{res.best_cost:.2f} km')
        improvement = ((cost_h - res.best_cost) / cost_h * 100) if cost_h > 0 else 0
        col3.metric('Melhoria', f'{improvement:.2f}%')
        col4.metric('B&B Tempo', f'{res.time_seconds:.2f} s')
        
        # Gráficos
        real_data = pd.DataFrame({
            'Algoritmo': ['Nearest Neighbor', 'Branch and Bound'],
            'Custo (km)': [cost_h, res.best_cost],
            'Tempo (s)': [time_nn, res.time_seconds],
            'Nós Expandidos': [0, res.nodes_expanded]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(real_data['Algoritmo'], real_data['Custo (km)'], color=['orange', 'green'])
            ax.set_ylabel('Custo (km)')
            ax.set_title('Comparação de Custo')
            for i, v in enumerate(real_data['Custo (km)']):
                ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(real_data['Algoritmo'], real_data['Tempo (s)'], color=['orange', 'green'])
            ax.set_ylabel('Tempo (s)')
            ax.set_title('Comparação de Tempo (Log Scale)')
            ax.set_yscale('log')
            st.pyplot(fig)
        
        # Mapa
        st.subheader('🗺️ Visualização das Rotas (Curitiba)')
        st.markdown('**Legenda:** 🟠 Nearest Neighbor | 🟢 Branch and Bound (Ótimo)')
        
        valid = coords_df.dropna(subset=['latitude', 'longitude'])
        
        if len(valid) >= 2:
            center_lat = valid['latitude'].mean()
            center_lon = valid['longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            
            # Marcadores
            for _, row in valid.iterrows():
                folium.Marker([row['latitude'], row['longitude']], 
                                popup=row['bairro'],
                                icon=folium.Icon(color='blue', icon='building', prefix='fa')).add_to(m)
            
            # Rota NN
            if tour_h:
                coords_nn = []
                for i in tour_h:
                    if i < len(locais):
                        r = coords_df[coords_df['bairro'] == locais[i]]
                        if not r.empty and pd.notna(r.iloc[0]['latitude']):
                            coords_nn.append([r.iloc[0]['latitude'], r.iloc[0]['longitude']])
                if coords_nn:
                    folium.PolyLine(coords_nn, color='orange', weight=4, opacity=0.6, tooltip="NN").add_to(m)
            
            # Rota B&B
            if res.best_tour:
                coords_bb = []
                for i in res.best_tour:
                    if i < len(locais):
                        r = coords_df[coords_df['bairro'] == locais[i]]
                        if not r.empty and pd.notna(r.iloc[0]['latitude']):
                            coords_bb.append([r.iloc[0]['latitude'], r.iloc[0]['longitude']])
                if coords_bb:
                    folium.PolyLine(coords_bb, color='green', weight=6, opacity=0.8, tooltip="B&B").add_to(m)
            
            st_folium(m, width=1200, height=600)
            
            st.markdown("### Sequência de Visita (Bairros)")
            rota_nomes = [locais[i] for i in res.best_tour if i < len(locais)]
            st.write(" ➡️ ".join(rota_nomes))

    else:
        st.info("Execute a '⚙️ Otimização' primeiro para gerar dados reais.")


# SEÇÃO SENSIBILIDADE
elif section == '🔬 Sensibilidade':
    st.header('Análise de Sensibilidade')
    
    tab1, tab2 = st.tabs(['Tempo Limite', 'Tamanho'])
    
    # --- ABA 1: Impacto do Tempo Limite ---
    with tab1:
        st.subheader('Impacto do Tempo Limite')
        
        test_size = st.number_input('Quantidade de Bairros', 4, 15, 8, key='sens_time_size')
        time_limits = st.multiselect('Tempos (s)', [5, 10, 30, 60, 120], [10, 30, 60], key='sens_time_limits')
        
        if st.button('Testar Tempos'):
            if not time_limits:
                st.warning('Selecione pelo menos um tempo')
            else:
                results = []
                agg = load_and_aggregate(DATA_PATH)
                sample = agg.head(test_size).reset_index(drop=True)
                coords = geocode_municipalities(sample)
                dist_mat, _ = build_distance_matrix_from_coords(coords)
                
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
    
    # --- ABA 2: Impacto do Tamanho ---
    with tab2:
        st.subheader('Impacto do Tamanho (Número de Bairros)')
        
        time_fixed = st.slider('Tempo fixo (s)', 10, 120, 60, key='sens_size_time')
        sizes = st.multiselect('Tamanhos', [4, 6, 8, 10, 12, 14], [4, 8, 12], key='sens_size_sizes')
        
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
                    dist_mat, _ = build_distance_matrix_from_coords(coords)
                    
                    res = branch_and_bound_tsp(dist_mat, time_limit=time_fixed)
                    results.append({
                        'Bairros': sz,
                        'Custo (km)': res.best_cost,
                        'Nós Expandidos': res.nodes_expanded,
                        'Tempo (s)': res.time_seconds
                    })
                    progress.progress((i + 1) / len(sizes))
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_results['Bairros'], df_results['Nós Expandidos'], 
                        marker='o', linewidth=2)
                ax.set_xlabel('Número de Bairros')
                ax.set_ylabel('Nós Expandidos')
                ax.set_title('Escalabilidade: Nós Expandidos vs Tamanho')
                ax.set_yscale('log')
                ax.grid(alpha=0.3)
                st.pyplot(fig)