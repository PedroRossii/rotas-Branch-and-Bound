"""
Script principal do sistema de otimização de rotas
Pode ser usado como CLI ou para lançar a interface Streamlit
"""

import argparse
import time
import sys
import subprocess
import os

from src.data_processing import load_and_aggregate
from src.geocoding import geocode_municipalities, build_distance_matrix_from_coords
from src.heuristics import nearest_neighbor
from src.bb_tsp import branch_and_bound_tsp


def run_cli(sample_size, time_limit):
    """Executa otimização via linha de comando"""
    path = 'enderecos_pr_filtered.csv'
    
    print('='*70)
    print('SISTEMA DE OTIMIZAÇÃO DE ROTAS - BRANCH AND BOUND')
    print('='*70)
    print()
    
    print('[1/5] Carregando e agregando dados...')
    agg = load_and_aggregate(path)
    print(f'      ✓ {len(agg)} municípios carregados')
    
    sample = agg.head(sample_size).reset_index(drop=True)
    print(f'      ✓ Amostra de {sample_size} municípios selecionada')
    print()
    print(sample[['municipio', 'count']].to_string(index=False))
    print()
    
    print('[2/5] Geocodificando municípios via Google Maps API...')
    coords = geocode_municipalities(sample)
    print(f'      ✓ {len(coords)} coordenadas obtidas')
    print()
    
    print('[3/5] Construindo matriz de distâncias (Haversine)...')
    dist, municipios = build_distance_matrix_from_coords(coords)
    print(f'      ✓ Matriz {dist.shape[0]}x{dist.shape[1]} construída')
    print()
    
    print('[4/5] Executando heurística Nearest Neighbor...')
    t0 = time.time()
    tour_h, cost_h = nearest_neighbor(dist, start=0)
    time_nn = time.time() - t0
    print(f'      ✓ Custo: {cost_h:.2f} km')
    print(f'      ✓ Tempo: {time_nn:.4f} s')
    print(f'      ✓ Tour: {" → ".join([municipios[i][:15] for i in tour_h[:5]])}...')
    print()
    
    print(f'[5/5] Executando Branch and Bound (limite: {time_limit}s)...')
    res = branch_and_bound_tsp(dist, time_limit=time_limit)
    print(f'      ✓ Custo ótimo: {res.best_cost:.2f} km')
    print(f'      ✓ Nós expandidos: {res.nodes_expanded:,}')
    print(f'      ✓ Profundidade máxima: {res.max_depth}')
    print(f'      ✓ Tempo: {res.time_seconds:.2f} s')
    print()
    
    # Resumo final
    improvement = ((cost_h - res.best_cost) / cost_h * 100) if cost_h > res.best_cost else 0
    
    print('='*70)
    print('RESUMO DOS RESULTADOS')
    print('='*70)
    print()
    print(f'Nearest Neighbor:')
    print(f'  • Custo: {cost_h:.2f} km')
    print(f'  • Tempo: {time_nn:.4f} s')
    print()
    print(f'Branch and Bound:')
    print(f'  • Custo: {res.best_cost:.2f} km')
    print(f'  • Tempo: {res.time_seconds:.2f} s')
    print(f'  • Nós expandidos: {res.nodes_expanded:,}')
    print()
    
    if improvement > 0:
        print(f'✓ Melhoria: {improvement:.2f}% ({cost_h - res.best_cost:.2f} km economizados)')
    else:
        print(f'⚠ Nenhuma melhoria encontrada (NN já era ótima ou tempo insuficiente)')
    
    print()
    print(f'Tour completo (índices): {res.best_tour}')
    print(f'Tour completo (municípios):')
    for i, idx in enumerate(res.best_tour):
        if idx < len(municipios):
            print(f'  {i+1}. {municipios[idx]}')
    print()
    print('='*70)


def run_streamlit():
    """Lança a interface Streamlit"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_app = os.path.join(project_dir, 'app', 'streamlit_app.py')
    
    print('Iniciando interface Streamlit...')
    print('O navegador será aberto automaticamente em http://localhost:8501')
    print('Pressione Ctrl+C para encerrar')
    print()
    
    cmd = [sys.executable, '-m', 'streamlit', 'run', streamlit_app]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Otimização de Rotas com Branch and Bound',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Lançar interface gráfica (padrão)
  python Main.py
  
  # Executar CLI com 8 municípios e 30s de limite
  python Main.py --cli --sample-size 8 --time-limit 30
  
  # Executar CLI com 12 municípios e 60s de limite
  python Main.py --cli --sample-size 12 --time-limit 60
  
Para mais informações, consulte README.md ou GUIA_RAPIDO.md
        """
    )
    
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Executar em modo linha de comando (ao invés de abrir Streamlit)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=8,
        help='Número de municípios a otimizar (padrão: 8, recomendado: 4-16)'
    )
    
    parser.add_argument(
        '--time-limit',
        type=float,
        default=30.0,
        help='Tempo limite do Branch and Bound em segundos (padrão: 30)'
    )
    
    args = parser.parse_args()
    
    # Se --cli foi passado, executar CLI
    if args.cli:
        try:
            run_cli(args.sample_size, args.time_limit)
        except KeyboardInterrupt:
            print('\n\nExecução interrompida pelo usuário.')
            sys.exit(0)
        except Exception as e:
            print(f'\n❌ Erro durante execução: {e}')
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Caso contrário, lançar Streamlit
        try:
            run_streamlit()
        except KeyboardInterrupt:
            print('\n\nStreamlit encerrado.')
            sys.exit(0)
        except Exception as e:
            print(f'\n❌ Erro ao iniciar Streamlit: {e}')
            print('\nTente executar manualmente: streamlit run app/streamlit_app.py')
            sys.exit(1)


if __name__ == '__main__':
    main()
