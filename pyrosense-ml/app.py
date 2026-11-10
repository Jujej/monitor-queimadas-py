from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import logging
import time
from pathlib import Path
from pandas.api.types import is_numeric_dtype
from werkzeug.utils import secure_filename

import etl  # pipeline de ETL (etl.py)

"""
Todos
- 


"""


# Módulo de treino (treino.py). Se o scikit-learn não carregar (ex.: DLL bloqueada pelo
# Windows), o app continua subindo e só as rotas de ML respondem 503.
try:
    import treino
    TREINO_DISPONIVEL, ERRO_TREINO = True, None
except Exception as e:  # ImportError, "DLL load failed" etc.
    treino = None
    TREINO_DISPONIVEL, ERRO_TREINO = False, str(e)
    print(f"⚠️ Módulo de treino indisponível ({e}). As rotas de ML responderão 503.")

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Caminhos (ETL)
# --------------------------------------------------------------------------- #
log = logging.getLogger("app")  # a configuração de logging vem do etl.py

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = etl.RAW_DIR / "focos_br_todos-sats_2025.csv"                    # base histórica
UPLOAD_DIR = etl.UPLOAD_DIR                                 # uploads do usuário (CSV)
PROCESSED_CSV = etl.PROCESSED_DIR / "queimadas_tocantins.csv"  # saída do etl.py

etl.criar_diretorios()  # garante data/raw, data/uploads e data/processed

# Estado em memória: só a base. O modelo treinado fica salvo em disco pelo treino.py.
df_atual = None


def gerar_dados_iniciais(n=400):
    """Gera dados de demonstração de queimadas (usado só se não houver base real)."""
    np.random.seed(42)
    biomas = ['Amazônia', 'Cerrado', 'Mata Atlântica', 'Pantanal', 'Caatinga']
    estados = ['MT', 'PA', 'MS', 'MG', 'PR', 'GO', 'AM']

    df = pd.DataFrame({
        'latitude': np.round(np.random.uniform(-24.0, -3.0, n), 4),
        'longitude': np.round(np.random.uniform(-63.0, -41.0, n), 4),
        'satelite': np.random.choice(['AQUA_M-T', 'TERRA_M-M', 'GOES-16', 'NOAA-20'], n),
        'bioma': np.random.choice(biomas, n, p=[0.35, 0.30, 0.15, 0.10, 0.10]),
        'estado': np.random.choice(estados, n),
        'dias_sem_chuva': np.random.randint(0, 60, n),
        'precipitacao': np.round(np.random.exponential(2.5, n), 2),
        'frp': np.round(np.random.uniform(5.0, 250.0, n), 1)
    })

    condicoes = [
        (df['dias_sem_chuva'] > 25) & (df['frp'] > 100),
        (df['dias_sem_chuva'] > 10) | (df['frp'] > 50)
    ]
    df['risco_fogo'] = np.select(condicoes, ['Alto', 'Médio'], default='Baixo')
    return df

def ler_kml(file_stream):
    """Converte arquivo KML de queimadas em DataFrame."""
    tree = ET.parse(file_stream)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    dados = []
    for pm in root.findall('.//kml:Placemark', ns):
        nome = pm.find('kml:name', ns)
        coords = pm.find('.//kml:coordinates', ns)
        if coords is not None and coords.text:
            partes = coords.text.strip().split(',')
            if len(partes) >= 2:
                dados.append({
                    'satelite': nome.text if nome is not None else 'Foco_KML',
                    'longitude': float(partes[0]),
                    'latitude': float(partes[1]),
                    'bioma': 'Desconhecido',
                    'estado': 'BR',
                    'dias_sem_chuva': np.random.randint(5, 40),
                    'precipitacao': 0.0,
                    'frp': np.round(np.random.uniform(20.0, 150.0), 1),
                    'risco_fogo': np.random.choice(['Baixo', 'Médio', 'Alto'])
                })
    return pd.DataFrame(dados)

# --------------------------------------------------------------------------- #
# Helpers de dados / ETL
# --------------------------------------------------------------------------- #
def carregar_processado():
    """Lê o CSV tratado pelo ETL. `riscofogo` vira `risco_fogo`, nome que o front já usa como alvo padrão."""
    df = pd.read_csv(PROCESSED_CSV)
    return df.rename(columns={'riscofogo': 'risco_fogo'})

def carregar_dados_iniciais():
    """Usa os dados do ETL (rodando-o se preciso); sem base real, cai nos dados de demonstração."""
    log.info("Carregando base de dados...")
    inicio = time.perf_counter()
    try:
        if not PROCESSED_CSV.exists() and RAW_CSV.exists():
            uploads = sorted(UPLOAD_DIR.glob("*.csv"))
            log.info("Nenhum dado processado encontrado; executando o ETL (base + %d upload(s))...", len(uploads))
            etl.executar_etl(caminho_base=RAW_CSV, caminhos_upload=uploads)
        if PROCESSED_CSV.exists():
            log.info("Lendo dados processados: %s", PROCESSED_CSV)
            df = carregar_processado()
            log.info("Base carregada: %s linhas, %d colunas em %.1fs",
                     f"{len(df):,}", df.shape[1], time.perf_counter() - inicio)
            return df
    except Exception as e:
        log.error("Falha ao carregar a base real: %s", e)
    log.warning("Base real não encontrada em %s. Usando dados de demonstração.", RAW_CSV)
    return gerar_dados_iniciais()

def tipos_colunas(df):
    """Separa colunas categóricas e numéricas, ocultando data bruta e colunas *_enc do ETL."""
    cols = [c for c in df.columns if c != 'datahora' and not c.endswith('_enc')]
    cat = [c for c in cols if not is_numeric_dtype(df[c])]
    num = [c for c in cols if is_numeric_dtype(df[c])]
    return cols, cat, num

def classe_risco(serie):
    """Converte risco numérico (0–1, escala do INPE) em Baixo / Médio / Alto."""
    return pd.cut(serie, bins=[-np.inf, 0.4, 0.7, np.inf],
                  labels=['Baixo', 'Médio', 'Alto'], right=False).astype(str)

df_atual = carregar_dados_iniciais()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dados', methods=['GET'])
def obter_dados():
    global df_atual
    if df_atual is None or df_atual.empty:
        return jsonify({'erro': 'Base vazia'}), 400

    colunas, colunas_cat, colunas_num = tipos_colunas(df_atual)

    cat_bar = request.args.get('cat_bar', colunas_cat[0] if colunas_cat else None)
    cat_pie = request.args.get('cat_pie', colunas_cat[min(1, len(colunas_cat)-1)] if colunas_cat else None)

    bar_data = df_atual[cat_bar].astype(str).value_counts().to_dict() if cat_bar in df_atual else {}
    pie_data = df_atual[cat_pie].astype(str).value_counts().to_dict() if cat_pie in df_atual else {}

    col_lat = next((c for c in df_atual.columns if 'lat' in c.lower()), None)
    col_lon = next((c for c in df_atual.columns if 'lon' in c.lower()), None)
    pontos_mapa = []
    if col_lat and col_lon:
        df_mapa = df_atual.dropna(subset=[col_lat, col_lon])
        df_mapa = df_mapa.sample(min(len(df_mapa), 300), random_state=42)

        # Rótulo de risco (o front colore o marcador procurando Alto/Médio/Baixo no texto)
        if 'risco_fogo' in df_mapa.columns:
            riscos = (classe_risco(df_mapa['risco_fogo']) if is_numeric_dtype(df_mapa['risco_fogo'])
                      else df_mapa['risco_fogo'].astype(str))
        else:
            riscos = pd.Series('N/A', index=df_mapa.index)

        for idx, row in df_mapa.iterrows():
            local = f"Município: {row['municipio']} | " if 'municipio' in df_mapa.columns else ""
            pontos_mapa.append({
                'lat': float(row[col_lat]),
                'lon': float(row[col_lon]),
                'info': f"{local}Bioma: {row.get('bioma', 'N/A')} | Risco: {riscos[idx]} | FRP: {row.get('frp', 'N/A')}"
            })

    opcoes_categorias = {col: df_atual[col].dropna().astype(str).unique().tolist() for col in colunas_cat}
    medias_numericas = {col: round(float(df_atual[col].median()), 2) for col in colunas_num}

    return jsonify({
        'total_registros': len(df_atual),
        'total_colunas': len(df_atual.columns),
        'valores_nulos': int(df_atual.isna().sum().sum()),
        'colunas': colunas,
        'colunas_cat': colunas_cat,
        'colunas_num': colunas_num,
        'amostra_tabela': df_atual.head(15).fillna('').to_dict(orient='records'),
        'grafico_barras': {'labels': list(bar_data.keys()), 'valores': list(bar_data.values()), 'coluna': cat_bar},
        'grafico_pizza': {'labels': list(pie_data.keys()), 'valores': list(pie_data.values()), 'coluna': cat_pie},
        'pontos_mapa': pontos_mapa,
        'opcoes_categorias': opcoes_categorias,
        'medias_numericas': medias_numericas
    })

# --------------------------------------------------------------------------- #
# Indicadores e sazonalidade (calculados a partir da base carregada)
# --------------------------------------------------------------------------- #
MESES_ABREV = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
_cache_datas = {'df': None, 'datas': None}


def _datas_da_base():
    """
    `datahora` da base atual como datetime. É calculado uma vez por base: o cache é
    invalidado sozinho quando o upload troca o `df_atual` por outro DataFrame.
    Levanta ValueError se a base não tem datas (ex.: dados de demonstração).
    """
    if df_atual is None or df_atual.empty or 'datahora' not in df_atual.columns:
        raise ValueError("A base atual não tem a coluna 'datahora' (provavelmente são os dados de "
                         "demonstração). Carregue a base real do INPE.")
    if _cache_datas['df'] is not df_atual:
        _cache_datas['datas'] = pd.to_datetime(df_atual['datahora'], errors='coerce')
        _cache_datas['df'] = df_atual
    datas = _cache_datas['datas']
    if datas.notna().sum() == 0:
        raise ValueError("Nenhuma data válida em 'datahora'.")
    return datas


def _variacao_pct(atual, anterior):
    """Variação % entre dois valores; None quando não há base de comparação (vazio ou zero)."""
    if atual is None or anterior is None or pd.isna(atual) or pd.isna(anterior) or anterior == 0:
        return None
    return round(float((atual - anterior) / anterior * 100), 1)


def _numero(valor, casas=2):
    """float arredondado, ou None se for NaN (jsonify não aceita NaN como JSON válido)."""
    return None if valor is None or pd.isna(valor) else round(float(valor), casas)


@app.route('/api/kpis', methods=['GET'])
def obter_kpis():
    """
    Indicadores do mês mais recente da base (substituem os KPIs fixos do HTML).

    - Município mais crítico: o que teve mais focos no mês de referência.
    - Todas as variações comparam com o MESMO recorte de dias do mês anterior. Assim,
      se a base termina no meio do mês, não se compara um mês parcial com um mês cheio.
    - `variacao_pct` vem como null quando o mês anterior teve 0 focos (sem base de comparação).
    """
    try:
        datas = _datas_da_base()
    except ValueError as e:
        return jsonify({'erro': str(e)}), 400
    if 'municipio' not in df_atual.columns:
        return jsonify({'erro': "A base atual não tem a coluna 'municipio'."}), 400

    ultimo = datas.max()
    ref = ultimo.to_period('M')
    ant = ref - 1
    periodos = datas.dt.to_period('M')
    mask_ref = periodos == ref
    mask_ant = (periodos == ant) & (datas.dt.day <= ultimo.day)

    focos_ref = df_atual.loc[mask_ref, 'municipio'].value_counts()
    focos_ant = df_atual.loc[mask_ant, 'municipio'].value_counts()
    ranking = [
        {'municipio': str(mun), 'focos': int(n), 'focos_mes_anterior': int(focos_ant.get(mun, 0)),
         'variacao_pct': _variacao_pct(int(n), int(focos_ant.get(mun, 0)))}
        for mun, n in focos_ref.head(5).items()
    ]

    risco_ref = risco_ant = None
    if 'risco_fogo' in df_atual.columns:
        risco = pd.to_numeric(df_atual['risco_fogo'], errors='coerce')
        risco_ref, risco_ant = risco[mask_ref].mean(), risco[mask_ant].mean()

    total_ref, total_ant = int(mask_ref.sum()), int(mask_ant.sum())
    return jsonify({
        'periodo_referencia': {
            'ano': int(ref.year), 'mes': int(ref.month), 'rotulo': f'{MESES_ABREV[ref.month - 1]}/{ref.year}',
            'ultimo_dia_com_dados': int(ultimo.day), 'parcial': bool(ultimo.day < ref.days_in_month),
        },
        'periodo_comparacao': {'ano': int(ant.year), 'mes': int(ant.month), 'dias_comparados': int(ultimo.day)},
        'total_focos': {'atual': total_ref, 'anterior': total_ant, 'variacao_pct': _variacao_pct(total_ref, total_ant)},
        'risco_medio': {'atual': _numero(risco_ref, 3), 'anterior': _numero(risco_ant, 3),
                        'variacao_pct': _variacao_pct(risco_ref, risco_ant)},
        'municipio_mais_critico': ranking[0] if ranking else None,
        'top_municipios': ranking,
    })


@app.route('/api/sazonalidade', methods=['GET'])
def obter_sazonalidade():
    """
    Focos por mês e ano, no formato que o heatmap do ApexCharts consome.

    - `series`: uma por ano (ordem crescente); cada ponto é {x: 'Jan', y: focos}.
    - `y` é null nos meses FORA do período coberto pela base (antes do primeiro e depois do
      último registro): célula vazia no gráfico, em vez de um falso "0 focos".
    - `faixas`: cortes da escala de cores calculados dos próprios dados (tercis), já no formato
      de `colorScale.ranges` (from/to), para o front não depender de limites fixos.
    """
    try:
        datas = _datas_da_base()
    except ValueError as e:
        return jsonify({'erro': str(e)}), 400

    periodos = datas.dropna().dt.to_period('M')
    contagem = periodos.value_counts()
    primeiro, ultimo = periodos.min(), periodos.max()

    series = []
    for ano in range(primeiro.year, ultimo.year + 1):
        pontos = []
        for mes in range(1, 13):
            p = pd.Period(year=ano, month=mes, freq='M')
            pontos.append({'x': MESES_ABREV[mes - 1],
                           'y': int(contagem.get(p, 0)) if primeiro <= p <= ultimo else None})
        series.append({'name': str(ano), 'data': pontos})

    valores = [pt['y'] for s in series for pt in s['data'] if pt['y']]
    maximo = max(valores) if valores else 0
    corte1, corte2 = (int(round(q)) for q in np.quantile(valores, [1 / 3, 2 / 3])) if valores else (0, 0)
    ultima_data = datas.max()

    return jsonify({
        'meses': MESES_ABREV,
        'anos': [s['name'] for s in series],
        'series': series,
        'faixas': [
            {'from': 0, 'to': corte1, 'name': 'Baixo'},
            {'from': corte1 + 1, 'to': corte2, 'name': 'Moderado'},
            {'from': corte2 + 1, 'to': max(maximo, corte2 + 1), 'name': 'Crítico'},
        ],
        'periodo': {'inicio': str(primeiro), 'fim': str(ultimo)},
        'mes_parcial': (None if ultima_data.day >= ultimo.days_in_month
                        else {'ano': int(ultimo.year), 'mes': int(ultimo.month)}),
        'total_focos': int(contagem.sum()),
    })


@app.route('/api/upload', methods=['POST'])
def upload_arquivo():
    global df_atual
    if 'arquivo' not in request.files:
        return jsonify({'erro': 'Nenhum arquivo enviado'}), 400

    arq = request.files['arquivo']
    nome = arq.filename.lower()
    destino = None
    inicio = time.perf_counter()
    log.info("Upload recebido: %s", arq.filename)

    try:
        # 1) Converte qualquer formato aceito para um DataFrame e guarda como CSV em data/uploads
        if nome.endswith('.csv'):
            df_novo = pd.read_csv(arq, sep=None, engine='python')
        elif nome.endswith('.json'):
            df_novo = pd.read_json(arq)
        elif nome.endswith('.xml'):
            df_novo = pd.read_xml(arq)
        elif nome.endswith('.kml'):
            df_novo = ler_kml(arq)
        else:
            return jsonify({'erro': 'Formato não suportado. Use CSV, JSON, XML ou KML.'}), 400

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        destino = UPLOAD_DIR / (secure_filename(Path(arq.filename).stem) + '.csv')
        df_novo.to_csv(destino, index=False)

        # 2) Valida e reprocessa base + uploads com o ETL
        log.info("Upload salvo em %s (%s linhas); validando...", destino, f"{len(df_novo):,}")
        etl.carregar_upload(destino)
        uploads = sorted(UPLOAD_DIR.glob("*.csv"))
        log.info("Reprocessando base + %d upload(s) com o ETL...", len(uploads))
        etl.executar_etl(caminho_base=RAW_CSV if RAW_CSV.exists() else None, caminhos_upload=uploads)

        df_atual = carregar_processado()
        # dados novos: o modelo salvo ficou desatualizado; o front deve re-treinar
        log.info("Base atualizada: %s linhas (%.1fs no total)", f"{len(df_atual):,}", time.perf_counter() - inicio)
        return jsonify({'mensagem': f'Arquivo {arq.filename} carregado com sucesso! ({len(df_atual)} linhas na base)',
                        'retreinar': True})

    except etl.ETLError as e:
        log.error("Upload rejeitado: %s", e)
        if destino is not None:
            destino.unlink(missing_ok=True)  # não deixa um upload inválido quebrar os próximos ETLs
        return jsonify({'erro': str(e)}), 400
    except Exception as e:
        if destino is not None:
            destino.unlink(missing_ok=True)
        return jsonify({'erro': f'Falha ao processar arquivo: {str(e)}'}), 500

def _ml_indisponivel():
    return jsonify({'erro': f'Módulo de treino indisponível: {ERRO_TREINO}'}), 503

@app.route('/api/treinar', methods=['POST'])
def treinar_modelo():
    """
    Body JSON (todos opcionais):
        modelo            'decision_tree' | 'random_forest' | 'knn'   (padrão: random_forest)
        parametro         profundidade máxima (árvores) ou k (kNN), 1-50   (padrão: 8)
        features          lista de colunas; se vazia usa o conjunto padrão
        conjunto          'completo' | 'sem_clima'  (usado quando `features` está vazia)
        ano_teste         ano a partir do qual os dados vão para o teste
        validacao_cruzada true/false (padrão false: deixa o treino interativo mais rápido)
    """
    if not TREINO_DISPONIVEL:
        return _ml_indisponivel()

    dados = request.json or {}
    if dados.get('alvo') not in (None, '', 'risco_fogo', 'riscofogo'):
        return jsonify({'erro': "O modelo prevê o risco de fogo; o alvo deve ser 'risco_fogo'."}), 400
    if df_atual is None or df_atual.empty:
        return jsonify({'erro': 'Base vazia'}), 400

    try:
        parametro = max(1, min(int(dados.get('parametro', 8)), 50))
        ano_teste = int(dados['ano_teste']) if dados.get('ano_teste') else None
    except (TypeError, ValueError):
        return jsonify({'erro': 'parametro e ano_teste devem ser números inteiros.'}), 400

    try:
        resultado = treino.treinar(
            df_atual,
            modelo=dados.get('modelo') or 'random_forest',
            parametro=parametro,
            features=dados.get('features') or None,
            conjunto=dados.get('conjunto') or 'completo',
            ano_teste=ano_teste,
            validacao_cruzada=bool(dados.get('validacao_cruzada', False)),
        )
        return jsonify(resultado)
    except treino.TreinoError as e:
        log.warning("Treino rejeitado: %s", e)
        return jsonify({'erro': str(e)}), 400
    except Exception as e:
        log.exception("Falha inesperada no treino")
        return jsonify({'erro': f'Falha no treino: {e}'}), 500

@app.route('/api/prever', methods=['POST'])
def prever():
    """Body: {"entradas": {"mes": 8, "diasemchuva": 30, "bioma": "CERRADO", ...}}"""
    if not TREINO_DISPONIVEL:
        return _ml_indisponivel()

    entradas = (request.json or {}).get('entradas', {})
    try:
        return jsonify(treino.prever(entradas))
    except treino.TreinoError as e:
        return jsonify({'erro': str(e)}), 400
    except Exception as e:
        log.exception("Falha inesperada na predição")
        return jsonify({'erro': f'Erro na predição: {e}'}), 500

@app.route('/api/modelo', methods=['GET'])
def info_modelo():
    """Resultado do último treino salvo (métricas, features, split). Evita re-treinar a cada carga da página."""
    if not TREINO_DISPONIVEL:
        return _ml_indisponivel()
    if not treino.METRICAS_PATH.exists():
        return jsonify({'erro': 'Nenhum modelo treinado ainda.'}), 404
    return app.response_class(treino.METRICAS_PATH.read_text(encoding='utf-8'), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, port=5000)