import heapq
import time
import math


class BBResult:
    def __init__(self):
        self.best_cost = float('inf')
        self.best_tour = None
        self.nodes_expanded = 0
        self.max_depth = 0
        self.time_seconds = 0.0


def lower_bound_two_min_edges(dist_matrix, path, cost_so_far, unvisited, start):
    # Bound baseado na soma das duas menores arestas para cada nó não visitado
    n = dist_matrix.shape[0]
    lb = cost_so_far
    # para o nó atual: considerar menor aresta até um não visitado (se houver)
    last = path[-1]
    if unvisited:
        min_from_last = min(dist_matrix[last][j] for j in unvisited)
    else:
        min_from_last = dist_matrix[last][start]
    lb += min_from_last
    # para cada nó não visitado, somar suas duas menores arestas
    s = 0.0
    for i in unvisited:
        row = dist_matrix[i]
        # considerar arestas para todos exceto ele mesmo
        mins = sorted([row[j] for j in range(n) if j != i])
        # pegar duas menores
        s += mins[0] + mins[1]
    # dividir por 2 (cada aresta contada duas vezes)
    lb += s / 2.0
    return lb


def branch_and_bound_tsp(dist_matrix, time_limit=30.0):
    start_time = time.time()
    n = dist_matrix.shape[0]
    result = BBResult()
    # nó inicial: path [0]
    # prioridade pela bound
    heap = []

    init_path = [0]
    init_cost = 0.0
    unvisited_init = tuple(i for i in range(1, n))
    init_bound = lower_bound_two_min_edges(dist_matrix, init_path, init_cost, unvisited_init, 0)
    heapq.heappush(heap, (init_bound, init_cost, init_path, unvisited_init))

    while heap:
        if time.time() - start_time > time_limit:
            break
        bound, cost_so_far, path, unvisited = heapq.heappop(heap)
        result.nodes_expanded += 1
        result.max_depth = max(result.max_depth, len(path))
        # poda: se bound >= best_cost então não expandir
        if bound >= result.best_cost:
            continue
        if not unvisited:
            # fechar tour
            total_cost = cost_so_far + dist_matrix[path[-1]][path[0]]
            if total_cost < result.best_cost:
                result.best_cost = total_cost
                result.best_tour = path + [path[0]]
            continue
        # expandir: escolher próximo nó entre unvisited
        for idx, v in enumerate(unvisited):
            new_path = path + [v]
            new_cost = cost_so_far + dist_matrix[path[-1]][v]
            new_unvisited = list(unvisited)
            new_unvisited.pop(idx)
            new_unvisited = tuple(new_unvisited)
            new_bound = lower_bound_two_min_edges(dist_matrix, new_path, new_cost, new_unvisited, 0)
            # poda por custo parcial
            if new_bound < result.best_cost:
                heapq.heappush(heap, (new_bound, new_cost, new_path, new_unvisited))
    result.time_seconds = time.time() - start_time
    return result
