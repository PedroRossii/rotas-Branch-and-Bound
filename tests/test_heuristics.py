"""
Testes unitários para heurísticas
"""

import numpy as np
import pytest
from src.heuristics import nearest_neighbor


class TestNearestNeighbor:
    """Testes detalhados para heurística Nearest Neighbor"""
    
    def test_nn_basic_functionality(self):
        """Testa funcionalidade básica do NN"""
        dist = np.array([
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 25.0, 30.0],
            [15.0, 25.0, 0.0, 35.0],
            [20.0, 30.0, 35.0, 0.0]
        ])
        
        tour, cost = nearest_neighbor(dist, start=0)
        
        assert isinstance(tour, list), "Tour deve ser lista"
        assert isinstance(cost, (int, float)), "Custo deve ser numérico"
        assert cost > 0, "Custo deve ser positivo"
        
    def test_nn_visits_all_cities(self):
        """Testa se NN visita todas as cidades"""
        n = 6
        dist = np.random.randint(10, 100, size=(n, n)).astype(float)
        np.fill_diagonal(dist, 0)
        
        tour, cost = nearest_neighbor(dist, start=0)
        
        # Verificar que todas as cidades foram visitadas
        visited = set(tour[:-1])
        assert len(visited) == n, f"Deve visitar {n} cidades, visitou {len(visited)}"
        assert visited == set(range(n)), "Deve visitar todas as cidades de 0 a n-1"
        
    def test_nn_returns_to_start(self):
        """Testa se NN retorna à cidade inicial"""
        dist = np.array([
            [0.0, 10.0, 15.0],
            [10.0, 0.0, 20.0],
            [15.0, 20.0, 0.0]
        ])
        
        for start in range(3):
            tour, cost = nearest_neighbor(dist, start=start)
            assert tour[0] == start, f"Deve iniciar na cidade {start}"
            assert tour[-1] == start, f"Deve retornar à cidade {start}"
            
    def test_nn_cost_calculation(self):
        """Testa se custo é calculado corretamente"""
        dist = np.array([
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 25.0, 30.0],
            [15.0, 25.0, 0.0, 35.0],
            [20.0, 30.0, 35.0, 0.0]
        ])
        
        tour, cost = nearest_neighbor(dist, start=0)
        
        # Recalcular custo manualmente
        calculated_cost = 0
        for i in range(len(tour) - 1):
            calculated_cost += dist[tour[i]][tour[i+1]]
            
        assert abs(calculated_cost - cost) < 0.01, "Custo reportado deve bater com custo calculado"
        
    def test_nn_greedy_first_step(self):
        """Testa se NN faz escolha gulosa no primeiro passo"""
        dist = np.array([
            [0.0, 5.0, 100.0, 100.0],
            [5.0, 0.0, 50.0, 100.0],
            [100.0, 50.0, 0.0, 100.0],
            [100.0, 100.0, 100.0, 0.0]
        ])
        
        tour, cost = nearest_neighbor(dist, start=0)
        
        # Primeira escolha deve ser 0->1 (menor distância de 0)
        assert tour[1] == 1, "Primeira escolha deve ser cidade mais próxima"
        
    def test_nn_deterministic(self):
        """Testa se NN é determinístico"""
        dist = np.random.randint(10, 100, size=(5, 5)).astype(float)
        np.fill_diagonal(dist, 0)
        
        tour1, cost1 = nearest_neighbor(dist, start=0)
        tour2, cost2 = nearest_neighbor(dist, start=0)
        
        assert tour1 == tour2, "NN deve ser determinístico"
        assert cost1 == cost2, "Custo deve ser idêntico em execuções repetidas"
        
    def test_nn_different_starts_different_costs(self):
        """Testa que diferentes inícios podem gerar custos diferentes"""
        dist = np.array([
            [0.0, 10.0, 50.0, 100.0],
            [10.0, 0.0, 20.0, 80.0],
            [50.0, 20.0, 0.0, 30.0],
            [100.0, 80.0, 30.0, 0.0]
        ])
        
        costs = []
        for start in range(4):
            tour, cost = nearest_neighbor(dist, start=start)
            costs.append(cost)
            
        # Pelo menos dois custos diferentes (matriz não simétrica perfeita)
        assert len(set(costs)) > 1, "Diferentes pontos de partida devem gerar custos diferentes"
        
    def test_nn_no_self_loops(self):
        """Testa que NN não cria auto-loops (cidade para ela mesma)"""
        dist = np.array([
            [0.0, 10.0, 15.0],
            [10.0, 0.0, 20.0],
            [15.0, 20.0, 0.0]
        ])
        
        tour, cost = nearest_neighbor(dist, start=0)
        
        # Verificar que não há transições i->i (exceto no retorno final)
        for i in range(len(tour) - 2):
            assert tour[i] != tour[i+1], f"Não deve ter auto-loop: {tour[i]}->{tour[i+1]}"
            
    def test_nn_handles_symmetric_matrix(self):
        """Testa NN com matriz simétrica"""
        dist = np.array([
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 12.0, 18.0],
            [15.0, 12.0, 0.0, 22.0],
            [20.0, 18.0, 22.0, 0.0]
        ])
        
        tour, cost = nearest_neighbor(dist, start=0)
        
        # Verificar que custo reverso é igual
        reverse_cost = 0
        for i in range(len(tour) - 1):
            from_city = tour[i]
            to_city = tour[i+1]
            assert dist[from_city][to_city] == dist[to_city][from_city], "Matriz deve ser simétrica"
            
    def test_nn_two_cities(self):
        """Testa NN com apenas 2 cidades (caso trivial)"""
        dist = np.array([
            [0.0, 10.0],
            [10.0, 0.0]
        ])
        
        tour, cost = nearest_neighbor(dist, start=0)
        
        assert tour == [0, 1, 0], "Com 2 cidades, tour deve ser 0->1->0"
        assert cost == 20.0, "Custo deve ser 2 * distância"
        
    def test_nn_large_instance(self):
        """Testa NN em instância maior"""
        n = 20
        np.random.seed(42)
        dist = np.random.randint(10, 100, size=(n, n)).astype(float)
        np.fill_diagonal(dist, 0)
        
        tour, cost = nearest_neighbor(dist, start=0)
        
        # Verificações de validade
        assert len(tour) == n + 1, f"Tour deve ter {n+1} elementos"
        assert len(set(tour[:-1])) == n, "Deve visitar cada cidade uma vez"
        assert cost > 0, "Custo deve ser positivo"
        
    def test_nn_with_zero_distances(self):
        """Testa NN quando há distâncias zero (além da diagonal)"""
        dist = np.array([
            [0.0, 0.0, 15.0],
            [0.0, 0.0, 20.0],
            [15.0, 20.0, 0.0]
        ])
        
        tour, cost = nearest_neighbor(dist, start=0)
        
        # Deve funcionar mesmo com distâncias zero
        assert tour is not None, "Deve retornar tour válido"
        assert cost >= 0, "Custo deve ser não-negativo"


class TestHeuristicComparisons:
    """Testes comparativos entre heurísticas"""
    
    def test_nn_vs_random(self):
        """Testa se NN é melhor que tour aleatório (na média)"""
        np.random.seed(42)
        n = 8
        dist = np.random.randint(10, 100, size=(n, n)).astype(float)
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2  # Simétrica
        
        # NN
        _, cost_nn = nearest_neighbor(dist, start=0)
        
        # Tour aleatório (várias tentativas)
        random_costs = []
        for _ in range(10):
            random_tour = [0] + list(np.random.permutation(range(1, n))) + [0]
            random_cost = sum(dist[random_tour[i]][random_tour[i+1]] for i in range(len(random_tour)-1))
            random_costs.append(random_cost)
            
        avg_random_cost = np.mean(random_costs)
        
        # NN deve ser melhor que a média de tours aleatórios
        assert cost_nn < avg_random_cost, "NN deve ser melhor que tour aleatório médio"
        
    def test_nn_best_start(self):
        """Testa qual ponto de partida dá melhor resultado para NN"""
        dist = np.array([
            [0.0, 10.0, 50.0, 100.0],
            [10.0, 0.0, 20.0, 80.0],
            [50.0, 20.0, 0.0, 30.0],
            [100.0, 80.0, 30.0, 0.0]
        ])
        
        results = []
        for start in range(4):
            tour, cost = nearest_neighbor(dist, start=start)
            results.append((start, cost))
            
        # Identificar melhor início
        best_start, best_cost = min(results, key=lambda x: x[1])
        
        # Verificar que há diferença entre inícios
        costs = [c for s, c in results]
        assert max(costs) - min(costs) > 0, "Diferentes inícios devem produzir custos diferentes"


class TestHeuristicPerformance:
    """Testes de desempenho das heurísticas"""
    
    def test_nn_speed(self):
        """Testa se NN é rápido o suficiente"""
        import time
        
        n = 50
        np.random.seed(42)
        dist = np.random.randint(10, 100, size=(n, n)).astype(float)
        np.fill_diagonal(dist, 0)
        
        start_time = time.time()
        tour, cost = nearest_neighbor(dist, start=0)
        elapsed = time.time() - start_time
        
        # NN deve ser muito rápido (< 1 segundo para n=50)
        assert elapsed < 1.0, f"NN muito lento: {elapsed:.3f}s para n={n}"
        
    def test_nn_memory_efficient(self):
        """Testa se NN não consome memória excessiva"""
        import sys
        
        n = 100
        dist = np.random.randint(10, 100, size=(n, n)).astype(float)
        
        # Executar NN
        tour, cost = nearest_neighbor(dist, start=0)
        
        # Tour deve ter tamanho proporcional a n, não exponencial
        tour_size = sys.getsizeof(tour)
        assert tour_size < 10000, "Tour não deve consumir memória excessiva"


# Executar testes se script for chamado diretamente
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
