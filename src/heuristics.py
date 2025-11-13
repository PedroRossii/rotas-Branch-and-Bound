import numpy as np


def nearest_neighbor(dist_matrix, start=0):
    n = dist_matrix.shape[0]
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start
    cost = 0.0
    for _ in range(n - 1):
        # escolher vizinho mais próximo não visitado
        dists = dist_matrix[current]
        next_idx = None
        best = float('inf')
        for j in range(n):
            if not visited[j] and dists[j] < best:
                best = dists[j]
                next_idx = j
        if next_idx is None:
            break
        tour.append(next_idx)
        visited[next_idx] = True
        cost += best
        current = next_idx
    # voltar ao início
    cost += dist_matrix[current, start]
    tour.append(start)
    return tour, cost
