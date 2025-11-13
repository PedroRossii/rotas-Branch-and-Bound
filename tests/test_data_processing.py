"""
Testes unitários para processamento de dados
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import load_and_aggregate, get_raw_data


class TestLoadAndAggregate:
    """Testes para função load_and_aggregate"""
    
    def test_returns_dataframe(self):
        """Testa se retorna um DataFrame"""
        try:
            result = load_and_aggregate('enderecos_pr_filtered.csv')
            assert isinstance(result, pd.DataFrame), "Deve retornar um DataFrame"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_has_required_columns(self):
        """Testa se DataFrame tem colunas necessárias"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            required_cols = ['cod_ibge', 'municipio', 'count']
            
            for col in required_cols:
                assert col in df.columns, f"Deve ter coluna {col}"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_column_types(self):
        """Testa se colunas têm tipos corretos"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            assert df['cod_ibge'].dtype == np.int64, "cod_ibge deve ser int64"
            assert df['count'].dtype == np.int64, "count deve ser int64"
            assert df['municipio'].dtype == object, "municipio deve ser string"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_no_null_values(self):
        """Testa se não há valores nulos nas colunas críticas"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            assert df['cod_ibge'].isna().sum() == 0, "cod_ibge não deve ter nulos"
            assert df['municipio'].isna().sum() == 0, "municipio não deve ter nulos"
            assert df['count'].isna().sum() == 0, "count não deve ter nulos"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_sorted_by_count(self):
        """Testa se DataFrame está ordenado por count (decrescente)"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            # Verificar ordenação
            is_sorted = all(df['count'].iloc[i] >= df['count'].iloc[i+1] 
                           for i in range(len(df)-1))
            assert is_sorted, "DataFrame deve estar ordenado por count (decrescente)"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_positive_counts(self):
        """Testa se todas as contagens são positivas"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            assert (df['count'] > 0).all(), "Todas as contagens devem ser positivas"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_no_duplicate_cod_ibge(self):
        """Testa se não há códigos IBGE duplicados"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            assert not df['cod_ibge'].duplicated().any(), "Não deve ter cod_ibge duplicado"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_no_duplicate_municipios(self):
        """Testa se não há municípios duplicados"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            assert not df['municipio'].duplicated().any(), "Não deve ter municipio duplicado"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_valid_pr_cod_ibge_range(self):
        """Testa se códigos IBGE estão no range do Paraná"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            # Códigos do PR: 4100000 - 4199999
            in_range = df['cod_ibge'].between(4100000, 4199999)
            assert in_range.all(), "Todos os cod_ibge devem estar no range do PR"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_top_municipality_curitiba(self):
        """Testa se Curitiba está entre os top municípios"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            # Curitiba deve estar no top 5 (maior município do PR)
            top_5 = df.head(5)
            curitiba_in_top = top_5['municipio'].str.contains('Curitiba', case=False).any()
            
            assert curitiba_in_top, "Curitiba deve estar no top 5 municípios"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")


class TestGetRawData:
    """Testes para função get_raw_data"""
    
    def test_returns_dataframe(self):
        """Testa se retorna um DataFrame"""
        try:
            result = get_raw_data('enderecos_pr_filtered.csv')
            assert isinstance(result, pd.DataFrame), "Deve retornar um DataFrame"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_has_more_rows_than_aggregate(self):
        """Testa se dados brutos têm mais linhas que agregado"""
        try:
            raw = get_raw_data('enderecos_pr_filtered.csv')
            agg = load_and_aggregate('enderecos_pr_filtered.csv')
            
            assert len(raw) > len(agg), "Dados brutos devem ter mais linhas que agregado"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_has_detailed_columns(self):
        """Testa se dados brutos têm colunas detalhadas"""
        try:
            df = get_raw_data('enderecos_pr_filtered.csv')
            
            # Deve ter colunas de endereço detalhado
            expected_cols = ['cnpj', 'municipio', 'cod_ibge']
            for col in expected_cols:
                assert col in df.columns, f"Deve ter coluna {col}"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")


class TestDataQuality:
    """Testes de qualidade dos dados"""
    
    def test_cod_ibge_seven_digits(self):
        """Testa se códigos IBGE têm 7 dígitos"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            # Códigos IBGE têm 7 dígitos (1000000 - 9999999)
            valid_length = df['cod_ibge'].between(1000000, 9999999)
            assert valid_length.all(), "Todos os cod_ibge devem ter 7 dígitos"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_municipio_not_empty(self):
        """Testa se nomes de municípios não estão vazios"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            assert (df['municipio'].str.len() > 0).all(), "Nomes de municípios não devem estar vazios"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_count_consistency(self):
        """Testa consistência das contagens"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            # Soma de counts deve ser <= número de linhas dos dados brutos
            total_count = df['count'].sum()
            assert total_count > 0, "Soma de contagens deve ser positiva"
            
            # Total deve ser razoável (entre 1k e 10M para PR)
            assert 1000 <= total_count <= 10_000_000, "Total de empresas deve estar em range razoável"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_distribution_reasonable(self):
        """Testa se distribuição de empresas é razoável"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            # Top município não deve ter > 50% do total
            top_count = df['count'].iloc[0]
            total_count = df['count'].sum()
            top_percentage = top_count / total_count
            
            assert top_percentage < 0.5, "Top município não deve ter >50% das empresas"
            
            # Verificar que não há concentração extrema
            top_10_count = df.head(10)['count'].sum()
            top_10_percentage = top_10_count / total_count
            
            assert top_10_percentage < 0.8, "Top 10 municípios não devem ter >80% das empresas"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")


class TestDataTransformations:
    """Testes de transformações de dados"""
    
    def test_aggregation_preserves_municipalities(self):
        """Testa se agregação preserva todos os municípios únicos"""
        try:
            raw = get_raw_data('enderecos_pr_filtered.csv')
            agg = load_and_aggregate('enderecos_pr_filtered.csv')
            
            # Número de municípios únicos deve ser igual
            unique_raw = raw['municipio'].nunique()
            unique_agg = len(agg)
            
            assert unique_raw == unique_agg, "Agregação deve preservar todos os municípios únicos"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_aggregation_sums_correctly(self):
        """Testa se agregação soma corretamente"""
        try:
            raw = get_raw_data('enderecos_pr_filtered.csv')
            agg = load_and_aggregate('enderecos_pr_filtered.csv')
            
            # Soma de counts deve ser igual ao número de linhas em raw
            total_count = agg['count'].sum()
            total_rows = len(raw)
            
            assert total_count == total_rows, "Soma de counts deve ser igual ao total de linhas"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")


class TestEdgeCases:
    """Testes de casos extremos"""
    
    def test_handles_nonexistent_file(self):
        """Testa comportamento com arquivo inexistente"""
        with pytest.raises(FileNotFoundError):
            load_and_aggregate('arquivo_inexistente.csv')
            
    def test_sampling_works(self):
        """Testa se amostragem funciona corretamente"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            # Testar amostragem de diferentes tamanhos
            for sample_size in [5, 10, 20]:
                sample = df.head(sample_size)
                assert len(sample) == min(sample_size, len(df)), f"Amostra deve ter {sample_size} linhas"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")


class TestPerformance:
    """Testes de desempenho"""
    
    def test_load_is_reasonably_fast(self):
        """Testa se carregamento é rápido o suficiente"""
        import time
        
        try:
            start = time.time()
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            elapsed = time.time() - start
            
            # Carregamento deve ser < 10 segundos
            assert elapsed < 10.0, f"Carregamento muito lento: {elapsed:.2f}s"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_aggregate_memory_efficient(self):
        """Testa se agregação não consome memória excessiva"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            # DataFrame agregado deve ser pequeno (< 1 MB)
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            assert memory_mb < 1.0, f"DataFrame muito grande: {memory_mb:.2f} MB"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")


class TestStatistics:
    """Testes estatísticos sobre os dados"""
    
    def test_count_statistics_reasonable(self):
        """Testa se estatísticas de contagem são razoáveis"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            mean_count = df['count'].mean()
            median_count = df['count'].median()
            std_count = df['count'].std()
            
            # Verificações de sanidade
            assert mean_count > 0, "Média deve ser positiva"
            assert median_count > 0, "Mediana deve ser positiva"
            assert std_count > 0, "Desvio padrão deve ser positivo"
            
            # Mediana deve ser menor que média (distribuição assimétrica esperada)
            assert median_count < mean_count, "Distribuição deve ser assimétrica à direita"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")
            
    def test_no_extreme_outliers(self):
        """Testa se não há outliers extremos"""
        try:
            df = load_and_aggregate('enderecos_pr_filtered.csv')
            
            q1 = df['count'].quantile(0.25)
            q3 = df['count'].quantile(0.75)
            iqr = q3 - q1
            
            # Outliers extremos: > Q3 + 3*IQR
            upper_bound = q3 + 3 * iqr
            extreme_outliers = (df['count'] > upper_bound).sum()
            
            # Deve haver poucos outliers extremos (< 5% dos dados)
            outlier_percentage = extreme_outliers / len(df)
            assert outlier_percentage < 0.05, f"Muitos outliers: {outlier_percentage:.2%}"
        except FileNotFoundError:
            pytest.skip("Arquivo de dados não encontrado")


# Executar testes se script for chamado diretamente
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
