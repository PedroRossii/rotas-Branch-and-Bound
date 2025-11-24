# Sistema de Otimização de Rotas com Branch and Bound

## 📋 Descrição do Projeto

Sistema completo em Python que implementa o algoritmo Branch and Bound para resolver o problema do Caixeiro Viajante (TSP) aplicado a rotas de manutenção entre bairros do município de Curitiba, utilizando dados reais de empresas brasileiras.

## 📊 Dataset Utilizado

### Fonte dos Dados
- **Nome**: EnderecosEmpresasComHeaders
- **Origem**: Kaggle
- **Link**: https://www.kaggle.com/datasets/hiratasan/enderecosempresascomheaders?resource=download
- **Tamanho Original**: ~7GB (44+ milhões de registros)
- **Tamanho Filtrado (PR - Curitiba)**: ~45MB (registros únicos de Curitiba - Paraná)

### Variáveis Relevantes

| Variável | Tipo | Descrição |
|----------|------|-----------|
| `cnpj` | String | Cadastro Nacional de Pessoa Jurídica (identificador único) |
| `bairro` | String | Bairro da empresa |
| `cep` | String | Código de Endereçamento Postal |
| `municipio` | String | Nome do município |

### Contexto e Problema

**Contexto**: Uma empresa de manutenção técnica precisa visitar diferentes bairros de Curitiba para prestar serviços a empresas cadastradas. O número de empresas por município varia significativamente, afetando a prioridade e frequência de visitas.

**Problema a Resolver**: Determinar a rota ótima que minimize a distância total percorrida entre os bairros selecionados, garantindo que todos sejam visitados exatamente uma vez antes de retornar ao ponto de origem (Problema do Caixeiro Viajante - TSP).

**Aplicação Prática**: 
- Planejamento de rotas de manutenção preventiva
- Otimização de logística de visitação
- Redução de custos operacionais e tempo de deslocamento
- Priorização de municípios com maior concentração de empresas

## 🏗️ Arquitetura do Sistema

```
rotasoperacionais-A1/
│
│
├── src/                            # Código-fonte principal
│   ├── __init__.py
│   ├── data_processing.py          # Carregamento e agregação de dados
│   ├── geocoding.py                # Geocodificação e distâncias
│   ├── heuristics.py               # Heurística Nearest Neighbor
│   ├── bb_tsp.py                   # Implementação Branch and Bound
│   └── distance.py                 # Cálculos de distância
│
├── app/                            # Interface Streamlit
│   └── streamlit_app.py            # Dashboard interativo
│
├── tests/                          # Testes unitários
│   ├── test_bb.py
│   ├── test_heuristics.py
│   └── test_data_processing.py
│
├── Main.py                         # Script CLI principal
├── preprocess.py                   # Pré-processamento inicial
├── requirements.txt                # Dependências
│── README.md                       # Este arquivo
│── MODELAGEM.md                    # Modelo matemático formal
│── enderecos_curitiba_filtered.csv # Arquivo gerqado após rodar o preprocess.py com os dados filtrados do dataset original
└── geocode_cache.csv               # Cache da geocodificação para caso haja algum problema com a API
```

## 🚀 Instalação e Configuração

### Requisitos
- Python 3.8+
- Chave da API Google Maps Geocoding (opcional, há fallback)

### Instalação

1. Clone ou baixe o projeto:
```bash
Baixe o projeto no Git: https://github.com/PedroRossii/rotas-Branch-and-Bound
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. (Primeira execução) Execute o pré-processamento:
OBS: No Git o arquivo salvo já é o filtrado, então podemos pular essa etapa.
```bash
python preprocess.py
```
Este comando irá:
- Filtrar apenas registros do Paraná (UF='PR') e do municipio de Curitiba (municipio='Curitiba')
- Remover duplicatas por CNPJ
- Criar arquivo otimizado `enderecos_curitiba_filtered.csv`

## 📖 Como Usar

### Modo 1: Interface Gráfica (Recomendado)

Execute o dashboard interativo:
```bash
python -m streamlit run app/streamlit_app.py
```

O sistema abrirá em `http://localhost:8501` com quatro seções:

#### 1️⃣ **EDA (Análise Exploratória)**
- Estatísticas descritivas completas
- Visualizações: histogramas, boxplots, gráficos de barras
- Identificação de outliers e padrões
- Análise de percentis e distribuições

#### 2️⃣ **Otimização**
- Seleção de municípios (4-20)
- Geocodificação automática via Google Maps API
- Execução de heurística Nearest Neighbor
- Execução de Branch and Bound com métricas detalhadas
- Visualização da rota no mapa interativo
- Comparação de resultados

#### 3️⃣ **Comparação**
- Gráficos comparativos de custo e tempo
- Tabelas de métricas lado a lado
- Análise de melhoria percentual

#### 4 **Sensibilidade**
- Testes do impacto do tempo de limite para rodar o algoritmo
- Testes do impacto do número de bairros para rodar o algoritmo

### Modo 2: Linha de Comando (CLI)

Execute com parâmetros personalizados:
```bash
python Main.py --sample-size 8 --time-limit 30
```

**Parâmetros disponíveis:**
- `--sample-size`: Número de municípios (padrão: 8)
- `--time-limit`: Tempo máximo em segundos para B&B (padrão: 30)

## 🧮 Modelagem do Problema

### Definição Formal

**Variáveis de Decisão:**
- `x_ij ∈ {0, 1}`: 1 se a rota passa diretamente de i para j, 0 caso contrário
- `u_i ∈ ℕ`: Ordem de visitação do nó i (eliminação de sub-tours)

**Função Objetivo:**
```
Minimizar: Σ Σ d_ij * x_ij
         i j≠i
```
Onde `d_ij` é a distância (em km) entre os municípios i e j.

**Restrições:**
1. Cada nó é visitado exatamente uma vez (saída):
   ```
   Σ x_ij = 1, ∀i
   j≠i
   ```

2. Cada nó é visitado exatamente uma vez (entrada):
   ```
   Σ x_ij = 1, ∀j
   i≠j
   ```

3. Eliminação de sub-tours (MTZ):
   ```
   u_i - u_j + n*x_ij ≤ n-1, ∀i,j ≠ 0, i≠j
   ```

### Relaxação e Bound

**Método de Relaxação**: Relaxação Linear Fracionária com Base nas Duas Menores Arestas

Para cada nó não visitado, calculamos a soma das duas menores arestas conectadas a ele. Esta soma fornece um limite inferior otimista do custo adicional necessário.

**Fórmula do Bound:**
```
LB(path) = custo_acumulado + min_aresta_atual + (Σ(min1_i + min2_i) / 2)
```

Onde:
- `custo_acumulado`: Distância já percorrida no caminho parcial
- `min_aresta_atual`: Menor distância do nó atual até qualquer nó não visitado
- `min1_i, min2_i`: Duas menores arestas do nó i não visitado

**Justificativa**: Em qualquer tour completo, cada nó deve ter grau 2 (entrar e sair uma vez). Usar as duas menores arestas de cada nó fornece um limite inferior admissível.

### Critérios de Poda

1. **Poda por Bound**: Se `LB(node) ≥ best_cost`, descarta o nó
2. **Poda por Viabilidade**: Rotas que violam restrições são eliminadas
3. **Poda por Tempo**: Interrompe busca após `time_limit` segundos

### Estratégia de Busca

**Best-First Search** usando heap (fila de prioridade):
- Prioriza nós com menor bound
- Maximiza chances de encontrar soluções ótimas rapidamente
- Reduz espaço de busca efetivamente

## 📈 Métricas de Execução

O sistema registra automaticamente:
- **Nós Expandidos**: Total de estados explorados
- **Profundidade Máxima**: Maior profundidade da árvore de busca
- **Tempo de Execução**: Duração total em segundos
- **Soluções Viáveis**: Número de tours completos encontrados
- **Melhor Custo**: Distância total do melhor tour (km)
- **Taxa de Melhoria**: Percentual de melhoria sobre heurística

## 🧪 Testes

Execute os testes unitários:
```bash
pytest tests/ -v
```

**Cobertura de Testes:**
- ✅ Cálculo de bounds
- ✅ Geração de estados válidos
- ✅ Poda de ramos inviáveis
- ✅ Validação de soluções ótimas
- ✅ Heurísticas de referência
- ✅ Processamento de dados

## 📊 Análise de Sensibilidade

O sistema permite avaliar o impacto de diferentes parâmetros:

1. **Variação de Tempo Limite**: 
   - Testes com 10s, 30s, 60s, 120s
   - Análise de trade-off qualidade vs tempo

2. **Tamanho da Instância**:
   - Testes com 4, 8, 12, 16, 20 municípios
   - Análise de escalabilidade

## 🔍 Validação e Comparação

### Heurística de Referência: Nearest Neighbor

**Algoritmo Guloso Construtivo:**
1. Inicia em um nó arbitrário
2. Em cada passo, visita o nó mais próximo ainda não visitado
3. Retorna ao início após visitar todos

**Complexidade**: O(n²)

**Vantagens**: Rápido, simples, fornece bound superior

**Desvantagens**: Solução pode ser 25-40% pior que o ótimo

### Comparação de Desempenho

| Métrica | Nearest Neighbor | Branch & Bound |
|---------|------------------|----------------|
| Tempo | < 1ms | 5-120s |
| Qualidade | Sub-ótima | Ótima* |
| Escalabilidade | Excelente | Limitada |
| Garantias | Nenhuma | Otimalidade* |

*Dentro do tempo limite estabelecido

## 🗺️ Geocodificação

### Google Maps Geocoding API

O sistema utiliza a API do Google Maps para converter nomes de bairros em coordenadas GPS:

**Cache Inteligente:**
- Armazena resultados em `geocode_cache.csv`
- Evita requisições duplicadas
- Reduz custos e tempo de execução

**Fallback**: Sistema funciona mesmo sem chave API usando cache existente

### Cálculo de Distâncias

**Fórmula de Haversine**: Calcula distância ortodrômica (great-circle) entre dois pontos na esfera terrestre.

```python
d = 2R × arcsin(√(sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)))
```

Onde R = 6371 km (raio médio da Terra)

**Precisão**: ±0.5% para distâncias < 1000km

## 📝 Decisões de Pré-processamento

Todas as decisões estão documentadas em `docs/DECISOES_PREPROCESSAMENTO.md`:

1. **Filtro Geográfico**: Apenas Curitiba-PR (reduz 99% dos dados)
2. **Remoção de Duplicatas**: Por CNPJ (mantém primeira ocorrência)
3. **Tratamento de Nulos**: Remoção de registros sem bairro
4. **Padronização**: UTF-8, trim de espaços, tipos consistentes
5. **Agregação**: Contagem por bairro para priorização

## 🎯 Resultados Esperados

### Instâncias Pequenas (4-8 bairros)
- **B&B**: Solução ótima em < 10s
- **Melhoria**: 5-15% sobre NN
- **Nós Expandidos**: 50-500

### Instâncias Médias (10-12 bairros)
- **B&B**: Solução ótima ou near-ótima em 30-60s
- **Melhoria**: 8-20% sobre NN
- **Nós Expandidos**: 1000-5000

### Instâncias Grandes (15-20 bairros)
- **B&B**: Melhor solução encontrada em tempo limite
- **Melhoria**: 10-25% sobre NN
- **Nós Expandidos**: 5000+

## 🤝 Contribuições

Projeto acadêmico desenvolvido para a disciplina de Pesquisa Operacional.

## 📄 Licença

Uso educacional - Dados públicos da Receita Federal do Brasil

## 👥 Autores
Cassiano Duarte
Luiz Eduardo Aben Athar Ribeiro
Pedro Ferreira Rossi
Wellerson Barauna

Desenvolvido como trabalho acadêmico - 2025
