import numpy as np


def build_distance_matrix(cod_ibge_series):
    """Cria matriz de custos a partir dos códigos IBGE como proxy de distância.

    A distância entre i e j é: abs(cod_ibge_i - cod_ibge_j) / 1000.0
    (hipótese de demonstração — substitua por geocoding em produção)
    """
    codes = cod_ibge_series.to_numpy(dtype=float)
    n = len(codes)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            mat[i, j] = abs(codes[i] - codes[j]) / 1000.0
    return mat
