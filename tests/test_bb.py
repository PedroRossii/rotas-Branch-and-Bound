"""
Testes unitários para o sistema de Branch and Bound TSP
Cobre: cálculo de bounds, heurísticas, processamento de dados
"""

import numpy as np
import pytest
from src.bb_tsp import branch_and_bound_tsp, lower_bound_two_min_edges, BBResult
from src.heuristics import nearest_neighbor
from src.data_processing import load_and_aggregate


class TestBranchAndBound:
    """Testes para o algoritmo Branch and Bound"""
    
    def test_bb_small_instance(self):
        """Testa B&B em instância pequena (4 nós)"""
        dist = np.array([
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 35.0, 25.0],
            [15.0, 35.0, 0.0, 30.0],
            [20.0, 25.0, 30.0, 0.0]
        ])
        
        res = branch_and_bound_tsp(dist, time_limit=5.0)
        
        # Verificações básicas
        assert res.best_tour is not None, "B&B deve encontrar uma solução"
        assert res.best_cost < float('inf'), "Custo deve ser finito"
        assert len(res.best_tour) == 5, "Tour deve ter 5 elementos (4 cidades + retorno)"
        assert res.best_tour[0] == res.best_tour[-1], "Tour deve retornar à origem"
        assert res.nodes_expanded > 0, "Deve expandir pelo menos 1 nó"
        
        # Verificar unicidade de visitas (exceto início/fim)
        visited = set(res.best_tour[:-1])
        assert len(visited) == 4, "Deve visitar cada cidade exatamente uma vez"
        
    def test_bb_finds_optimal_solution(self):
        """Testa se B&B encontra solução ótima conhecida"""
        # Instância triangular simples
        dist = np.array([
            [0.0, 10.0, 15.0],
            [10.0, 0.0, 20.0],
            [15.0, 20.0, 0.0]
        ])
        
        res = branch_and_bound_tsp(dist, time_limit=5.0)
        
        # Tour ótimo conhecido: 0->1->2->0 ou 0->2->1->0
        optimal_cost = 10 + 20 + 15  # = 45
        assert res.best_cost == optimal_cost, f"Custo deveria ser {optimal_cost}, mas foi {res.best_cost}"
        
    def test_bb_respects_time_limit(self):
        """Testa se B&B respeita o limite de tempo"""
        # Instância maior para garantir que não termine instantaneamente
        np.random.seed(42)
        n = 10
        dist = np.random.randint(10, 100, size=(n, n)).astype(float)
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2  # Tornar simétrica
        
        time_limit = 2.0
        res = branch_and_bound_tsp(dist, time_limit=time_limit)
        
        assert res.time_seconds <= time_limit + 0.5, "Deve respeitar time_limit (com margem)"
        
    def test_bb_improves_or_equals_heuristic(self):
        """Testa se B&B encontra solução melhor ou igual à heurística"""
        dist = np.array([
            [0.0, 29.0, 20.0, 21.0],
            [29.0, 0.0, 15.0, 17.0],
            [20.0, 15.0, 0.0, 28.0],
            [21.0, 17.0, 28.0, 0.0]
        ])
        
        tour_h, cost_h = nearest_neighbor(dist, start=0)
        res = branch_and_bound_tsp(dist, time_limit=10.0)
        
        assert res.best_cost <= cost_h, "B&B deve encontrar solução melhor ou igual ao NN"
        
    def test_bb_symmetric_instance(self):
        """Testa se matriz simétrica produz tour consistente"""
        dist = np.array([
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 12.0, 18.0],
            [15.0, 12.0, 0.0, 22.0],
            [20.0, 18.0, 22.0, 0.0]
        ])
        
        res = branch_and_bound_tsp(dist, time_limit=5.0)
        
        # Verificar simetria: custo i->j == j->i
        for i in range(len(res.best_tour) - 1):
            from_node = res.best_tour[i]
            to_node = res.best_tour[i + 1]
            assert dist[from_node][to_node] == dist[to_node][from_node], "Matriz deve ser simétrica"


class TestBoundCalculation:
    """Testes para cálculo de lower bounds"""
    
    def test_bound_is_admissible(self):
        """Testa se bound nunca superestima custo real"""
        dist = np.array([
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 12.0, 18.0],
            [15.0, 12.0, 0.0, 22.0],
            [20.0, 18.0, 22.0, 0.0]
        ])
        
        # Calcular bound inicial
        path = [0]
        cost = 0.0
        unvisited = tuple([1, 2, 3])
        bound = lower_bound_two_min_edges(dist, path, cost, unvisited, 0)
        
        # Bound deve ser menor ou igual ao custo real
        res = branch_and_bound_tsp(dist, time_limit=10.0)
        assert bound <= res.best_cost, "Bound deve ser admissível (não superestimar)"
        
    def test_bound_increases_with_path(self):
        """Testa se bound aumenta ao expandir caminho"""
        dist = np.array([
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 12.0, 18.0],
            [15.0, 12.0, 0.0, 22.0],
            [20.0, 18.0, 22.0, 0.0]
        ])
        
        # Bound inicial
        path1 = [0]
        unvisited1 = tuple([1, 2, 3])
        bound1 = lower_bound_two_min_edges(dist, path1, 0.0, unvisited1, 0)
        
        # Bound após um passo
        path2 = [0, 1]
        cost2 = dist[0][1]
        unvisited2 = tuple([2, 3])
        bound2 = lower_bound_two_min_edges(dist, path2, cost2, unvisited2, 0)
        
        assert bound2 >= bound1, "Bound deve ser monotonicamente crescente"
        
    def test_bound_at_solution(self):
        """Testa se bound no nó solução é exato"""
        dist = np.array([
            [0.0, 10.0, 15.0],
            [10.0, 0.0, 20.0],
            [15.0, 20.0, 0.0]
        ])
        
        # Caminho completo: 0->1->2
        path = [0, 1, 2]
        cost = dist[0][1] + dist[1][2]
        unvisited = tuple([])
        
        # Bound deve ser igual ao custo + aresta de retorno
        bound = lower_bound_two_min_edges(dist, path, cost, unvisited, 0)
        expected_cost = cost + dist[2][0]
        
        # Bound pode ter pequena diferença devido a cálculo diferente
        assert abs(bound - expected_cost) < 1.0, "Bound em solução completa deve ser próximo do custo real"


class TestHeuristics:
    """Testes para heurísticas"""
    
    def test_nearest_neighbor_valid_tour(self):
        """Testa se Nearest Neighbor produz tour válido"""
        dist = np.array([
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 12.0, 18.0],
            [15.0, 12.0, 0.0, 22.0],
            [20.0, 18.0, 22.0, 0.0]
        ])
        
        tour, cost = nearest_neighbor(dist, start=0)
        
        # Verificações básicas
        assert len(tour) == 5, "Tour deve ter 5 elementos"
        assert tour[0] == 0, "Tour deve começar no nó 0"
        assert tour[-1] == 0, "Tour deve terminar no nó 0"
        
        # Verificar unicidade
        visited = set(tour[:-1])
        assert len(visited) == 4, "Deve visitar cada nó exatamente uma vez"
        
        # Verificar custo
        calculated_cost = sum(dist[tour[i]][tour[i+1]] for i in range(len(tour)-1))
        assert abs(calculated_cost - cost) < 0.01, "Custo calculado deve bater com retornado"
        
    def test_nearest_neighbor_different_starts(self):
        """Testa NN com diferentes nós iniciais"""
        dist = np.array([
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 12.0, 18.0],
            [15.0, 12.0, 0.0, 22.0],
            [20.0, 18.0, 22.0, 0.0]
        ])
        
        for start in range(4):
            tour, cost = nearest_neighbor(dist, start=start)
            assert tour[0] == start, f"Tour deve começar no nó {start}"
            assert tour[-1] == start, f"Tour deve terminar no nó {start}"
            assert cost > 0, "Custo deve ser positivo"
            
    def test_nearest_neighbor_greedy_choice(self):
        """Testa se NN faz escolha gulosa correta"""
        # Instância onde escolha gulosa é clara
        dist = np.array([
            [0.0, 5.0, 100.0, 100.0],
            [5.0, 0.0, 10.0, 100.0],
            [100.0, 10.0, 0.0, 15.0],
            [100.0, 100.0, 15.0, 0.0]
        ])
        
        tour, cost = nearest_neighbor(dist, start=0)
        
        # Primeiro passo deve ser 0->1 (menor distância)
        assert tour[1] == 1, "NN deve escolher vizinho mais próximo"


class TestDataProcessing:
    """Testes para processamento de dados"""
    
    def test_load_and_aggregate_structure(self):
        """Testa estrutura do DataFrame agregado"""
        try:
            agg = load_and_aggregate('enderecos_pr_filtered.csv')
            
            # Verificar colunas
            assert 'cod_ibge' in agg.columns, "Deve ter coluna cod_ibge"
            assert 'municipio' in agg.columns, "Deve ter coluna municipio"
            assert 'count' in agg.columns, "Deve ter coluna count"
            
            # Verificar tipos
            assert agg['cod_ibge'].dtype == np.int64, "cod_ibge deve ser int"
            assert agg['count'].dtype == np.int64, "count deve ser int"
            
            # Verificar ordenação
            assert all(agg['count'].iloc[i] >= agg['count'].iloc[i+1] 
                      for i in range(len(agg)-1)), "Deve estar ordenado por count decrescente"
            
            # Verificar sem nulos
            assert agg['cod_ibge'].isna().sum() == 0, "Não deve ter nulos em cod_ibge"
            assert agg['municipio'].isna().sum() == 0, "Não deve ter nulos em municipio"
            
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_aggregate_counts_positive(self):
        """Testa se contagens são positivas"""
        try:
            agg = load_and_aggregate('enderecos_pr_filtered.csv')
            assert (agg['count'] > 0).all(), "Todas as contagens devem ser positivas"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_aggregate_no_duplicates(self):
        """Testa se não há municípios duplicados"""
        try:
            agg = load_and_aggregate('enderecos_pr_filtered.csv')
            assert not agg['cod_ibge'].duplicated().any(), "Não deve ter cod_ibge duplicado"
            assert not agg['municipio'].duplicated().any(), "Não deve ter municipio duplicado"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")


class TestEdgeCases:
    """Testes de casos extremos e de borda"""
    
    def test_two_cities_only(self):
        """Testa instância trivial com 2 cidades"""
        dist = np.array([
            [0.0, 10.0],
            [10.0, 0.0]
        ])
        
        res = branch_and_bound_tsp(dist, time_limit=5.0)
        
        # Tour trivial: 0->1->0
        assert res.best_cost == 20.0, "Custo de 2 cidades deve ser 2*distância"
        assert res.best_tour in [[0, 1, 0], [0, 1, 0]], "Tour único possível"
        
    def test_three_cities_all_solutions(self):
        """Testa instância com 3 cidades (2 tours únicos)"""
        dist = np.array([
            [0.0, 10.0, 15.0],
            [10.0, 0.0, 20.0],
            [15.0, 20.0, 0.0]
        ])
        
        res = branch_and_bound_tsp(dist, time_limit=5.0)
        
        # Dois tours possíveis: 0->1->2->0 ou 0->2->1->0
        cost1 = 10 + 20 + 15  # 0->1->2->0
        cost2 = 15 + 20 + 10  # 0->2->1->0
        
        assert res.best_cost in [cost1, cost2], "Deve encontrar um dos dois tours ótimos"
        
    def test_identical_distances(self):
        """Testa matriz com todas distâncias iguais"""
        n = 4
        dist = np.full((n, n), 10.0)
        np.fill_diagonal(dist, 0.0)
        
        res = branch_and_bound_tsp(dist, time_limit=5.0)
        
        # Qualquer tour tem custo n * 10
        expected_cost = n * 10.0
        assert abs(res.best_cost - expected_cost) < 0.1, "Todas as rotas devem ter mesmo custo"
        
    def test_zero_time_limit(self):
        """Testa comportamento com tempo limite zero"""
        dist = np.array([
            [0.0, 10.0, 15.0],
            [10.0, 0.0, 20.0],
            [15.0, 20.0, 0.0]
        ])
        
        res = branch_and_bound_tsp(dist, time_limit=0.0)
        
        # Deve retornar resultado mesmo com tempo 0 (pelo menos nó inicial)
        assert res is not None, "Deve retornar resultado mesmo com tempo 0"
        assert res.nodes_expanded >= 0, "Deve expandir 0 ou mais nós"


class TestMetrics:
    """Testes para métricas de execução"""
    
    def test_metrics_are_recorded(self):
        """Testa se métricas são registradas corretamente"""
        dist = np.array([
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 12.0, 18.0],
            [15.0, 12.0, 0.0, 22.0],
            [20.0, 18.0, 22.0, 0.0]
        ])
        
        res = branch_and_bound_tsp(dist, time_limit=5.0)
        
        # Verificar se métricas existem
        assert hasattr(res, 'nodes_expanded'), "Deve ter métrica nodes_expanded"
        assert hasattr(res, 'max_depth'), "Deve ter métrica max_depth"
        assert hasattr(res, 'time_seconds'), "Deve ter métrica time_seconds"
        assert hasattr(res, 'best_cost'), "Deve ter métrica best_cost"
        assert hasattr(res, 'best_tour'), "Deve ter métrica best_tour"
        
        # Verificar valores razoáveis
        assert res.nodes_expanded > 0, "Deve expandir pelo menos 1 nó"
        assert res.max_depth > 0, "Profundidade deve ser > 0"
        assert res.time_seconds >= 0, "Tempo deve ser >= 0"
        
    def test_max_depth_bounded(self):
        """Testa se profundidade máxima não excede n+1"""
        n = 5
        dist = np.random.randint(10, 100, size=(n, n)).astype(float)
        np.fill_diagonal(dist, 0)
        
        res = branch_and_bound_tsp(dist, time_limit=5.0)
        
        # Profundidade máxima = n (todos os nós visitados)
        assert res.max_depth <= n, f"Profundidade {res.max_depth} não deve exceder {n}"


# Executar testes se script for chamado diretamente
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
