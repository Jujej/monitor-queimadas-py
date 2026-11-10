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


# Tenta carregar o scikit-learn; se o Windows bloquear a DLL (_libsvm), ativa o modo compatível em NumPy
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    SKLEARN_DISPONIVEL = True
    print("✅ Scikit-learn carregado com sucesso!")
except ImportError as e:
    SKLEARN_DISPONIVEL = False
    print(f"⚠️ Aviso: Bloqueio de DLL do Windows detectado ({e}). Usando motor ML nativo em NumPy/Pandas.")

    class LabelEncoder:
        def fit_transform(self, y):
            self.classes_, y_idx = np.unique(y, return_inverse=True)
            return y_idx
        def transform(self, y):
            return np.array([np.where(self.classes_ == val)[0][0] if val in self.classes_ else 0 for val in y])
        def inverse_transform(self, idx):
            return self.classes_[idx]

    def train_test_split(X, y, test_size=0.25, random_state=42):
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        corte = int(len(X) * (1 - test_size))
        idx_treino, idx_teste = indices[:corte], indices[corte:]
        return X.iloc[idx_treino], X.iloc[idx_teste], y[idx_treino], y[idx_teste]

    def accuracy_score(y_true, y_pred):
        return float(np.mean(y_true == y_pred))

    class ModeloNativoML:
        """Implementação leve para quando o Smart App Control do Windows bloqueia DLLs do sklearn."""
        def __init__(self, tipo='decision_tree', param=5):
            self.tipo = tipo
            self.param = max(1, int(param))

        def fit(self, X, y):
            self.X_train = np.asarray(X, dtype=float)
            self.y_train = np.asarray(y, dtype=int)
            self.classes_ = np.unique(self.y_train)
            self.std = np.std(self.X_train, axis=0) + 1e-8
            return self

        def predict_proba(self, X):
            X_arr = np.asarray(X, dtype=float)
            probs = []
            # Ajusta vizinhos efetivos conforme algoritmo e parâmetro escolhido no slider
            k = min(len(self.X_train), self.param if self.tipo == 'knn' else max(3, 15 - self.param))
            for x in X_arr:
                dists = np.linalg.norm((self.X_train - x) / self.std, axis=1)
                idx = np.argsort(dists)[:k]
                vizinhos = self.y_train[idx]
                p = [np.mean(vizinhos == c) for c in self.classes_]
                probs.append(p)
            return np.array(probs)

        def predict(self, X):
            probs = self.predict_proba(X)
            return self.classes_[np.argmax(probs, axis=1)]

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

# Variáveis globais em memória para manter o estado da sessão
df_atual = None
modelo_atual = None
encoders_atuais = {}
encoder_alvo = None
features_atuais = []
alvo_atual = None


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

def preparar_alvo(serie):
    """Alvo numérico contínuo vira classes (risco_fogo: limiares fixos; demais: tercis)."""
    if is_numeric_dtype(serie) and serie.nunique() > 10:
        if serie.name == 'risco_fogo':
            return classe_risco(serie)
        return pd.qcut(serie.rank(method='first'), 3, labels=['Baixo', 'Médio', 'Alto']).astype(str)
    return serie.astype(str)

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

@app.route('/api/upload', methods=['POST'])
def upload_arquivo():
    global df_atual, modelo_atual
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
        modelo_atual = None  # dados novos invalidam o modelo anterior
        log.info("Base atualizada: %s linhas (%.1fs no total)", f"{len(df_atual):,}", time.perf_counter() - inicio)
        return jsonify({'mensagem': f'Arquivo {arq.filename} carregado com sucesso! ({len(df_atual)} linhas na base)'})

    except etl.ETLError as e:
        log.error("Upload rejeitado: %s", e)
        if destino is not None:
            destino.unlink(missing_ok=True)  # não deixa um upload inválido quebrar os próximos ETLs
        return jsonify({'erro': str(e)}), 400
    except Exception as e:
        if destino is not None:
            destino.unlink(missing_ok=True)
        return jsonify({'erro': f'Falha ao processar arquivo: {str(e)}'}), 500

@app.route('/api/treinar', methods=['POST'])
def treinar_modelo():
    global df_atual, modelo_atual, encoders_atuais, encoder_alvo, features_atuais, alvo_atual

    dados = request.json or {}
    alvo_atual = dados.get('alvo')
    features_atuais = [f for f in dados.get('features', []) if f != alvo_atual]
    tipo_modelo = dados.get('modelo')
    try:
        # Parâmetro vindo do campo do front (profundidade máxima); limitado a um intervalo seguro
        param_valor = max(1, min(int(dados.get('parametro', 5)), 50))
    except (TypeError, ValueError):
        return jsonify({'erro': 'O parâmetro do modelo deve ser um número inteiro.'}), 400

    if not alvo_atual or not features_atuais:
        return jsonify({'erro': 'Selecione a variável alvo e pelo menos uma variável preditora.'}), 400
    if alvo_atual not in df_atual.columns or any(f not in df_atual.columns for f in features_atuais):
        return jsonify({'erro': 'Alvo ou variáveis preditoras inválidas para a base atual.'}), 400

    df_ml = df_atual[features_atuais + [alvo_atual]].dropna().copy()
    encoders_atuais = {}

    for col in features_atuais:
        if not is_numeric_dtype(df_ml[col]):
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            encoders_atuais[col] = le

    encoder_alvo = LabelEncoder()
    y = encoder_alvo.fit_transform(preparar_alvo(df_ml[alvo_atual]).astype(str))
    X = df_ml[features_atuais]

    if len(encoder_alvo.classes_) < 2:
        return jsonify({'erro': 'O alvo tem apenas uma classe; escolha outra variável.'}), 400

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if SKLEARN_DISPONIVEL:
        if tipo_modelo == 'decision_tree':
            modelo_atual = DecisionTreeClassifier(max_depth=param_valor, random_state=42)
        elif tipo_modelo == 'knn':
            k = min(param_valor, len(X_train))
            modelo_atual = KNeighborsClassifier(n_neighbors=max(1, k))
        else:
            modelo_atual = RandomForestClassifier(n_estimators=min(param_valor * 10, 200), max_depth=param_valor, random_state=42)
    else:
        modelo_atual = ModeloNativoML(tipo=tipo_modelo, param=param_valor)

    modelo_atual.fit(X_train, y_train)
    y_pred = modelo_atual.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return jsonify({
        'acuracia': round(acc * 100, 2),
        'amostras_treino': len(X_train),
        'amostras_teste': len(X_test),
        'classes': encoder_alvo.classes_.tolist()
    })

@app.route('/api/prever', methods=['POST'])
def prever():
    global modelo_atual, encoders_atuais, encoder_alvo, features_atuais

    if modelo_atual is None:
        return jsonify({'erro': 'Treine o modelo primeiro antes de realizar uma predição!'}), 400

    entradas = request.json.get('entradas', {})
    linha = {}

    try:
        for col in features_atuais:
            val = entradas.get(col)
            if col in encoders_atuais:
                le = encoders_atuais[col]
                if val in le.classes_:
                    linha[col] = le.transform([val])[0]
                else:
                    linha[col] = 0
            else:
                linha[col] = float(val)

        df_in = pd.DataFrame([linha])
        pred_cod = modelo_atual.predict(df_in)[0]
        classe_final = encoder_alvo.inverse_transform([pred_cod])[0]

        probabilidades = {}
        if hasattr(modelo_atual, 'predict_proba'):
            probs = modelo_atual.predict_proba(df_in)[0]
            for cls, p in zip(encoder_alvo.classes_, probs):
                probabilidades[cls] = round(float(p) * 100, 1)

        return jsonify({
            'predicao': classe_final,
            'probabilidades': probabilidades
        })
    except Exception as e:
        return jsonify({'erro': f'Erro na predição: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)