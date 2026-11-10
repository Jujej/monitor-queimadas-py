"""
etl.py — Pipeline de ETL para focos de queimadas do INPE (Tocantins).

Etapas:
    1. Extract   : lê o CSV base e/ou CSVs enviados por upload e concatena tudo.
    2. Clean     : filtra TOCANTINS, trata nulos e remove duplicatas.
    3. Transform : converte datas, cria ano/mes/dia e a variação % mensal de focos
                   por município.
    4. Encode    : aplica LabelEncoder em colunas categóricas e salva os encoders.
    5. Load      : grava o resultado em data/processed/ (CSV e, se possível, Parquet).

Dependências: pandas e scikit-learn (se o sklearn não puder ser importado, usa um
              LabelEncoder equivalente) e pyarrow (opcional, só para Parquet).
"""

from __future__ import annotations

import csv
import hashlib
import logging
import time
import unicodedata
from pathlib import Path
from typing import Iterable

import pickle

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import LabelEncoder
except ImportError:  # ex.: DLL do scikit-learn bloqueada no Windows
    class LabelEncoder:
        """Substituto mínimo, com a mesma interface básica do LabelEncoder do sklearn."""

        def fit(self, y):
            self.classes_ = np.unique(np.asarray(y))
            return self

        def transform(self, y):
            mapa = {c: i for i, c in enumerate(self.classes_)}
            return np.array([mapa[v] for v in y])

        def fit_transform(self, y):
            return self.fit(y).transform(y)

        def inverse_transform(self, idx):
            return self.classes_[np.asarray(idx)]

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("etl_queimadas")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # base histórica
UPLOAD_DIR = DATA_DIR / "uploads"       # CSVs enviados pelo usuário
PROCESSED_DIR = DATA_DIR / "processed"  # saída do ETL

# Leitura em blocos: permite logar o progresso e filtrar o estado enquanto lê,
# sem nunca manter o CSV inteiro (milhões de linhas) na memória.
TAMANHO_CHUNK = 200_000

ESTADO_ALVO = "TOCANTINS"

# Só estas colunas são lidas do CSV (as demais, como `pais` e `id_area_industrial`,
# nem chegam a ser processadas). Para ler tudo: ler_csv(..., somente_uteis=False).
COLUNAS_UTEIS = {
    "datahora", "satelite", "estado", "municipio", "bioma", "riscofogo",
    "latitude", "longitude", "diasemchuva", "precipitacao", "frp",
}

# Colunas sem as quais o dataset não faz sentido para este projeto.
COLUNAS_OBRIGATORIAS = ["datahora", "estado", "municipio", "latitude", "longitude"]

# Colunas categóricas que serão convertidas com LabelEncoder.
COLUNAS_CATEGORICAS = ["municipio", "bioma", "satelite"]

# Colunas numéricas do INPE onde -999 significa "sem dado".
COLUNAS_NUMERICAS_SENTINELA = ["diasemchuva", "precipitacao", "riscofogo", "frp"]
VALOR_SENTINELA = -999

# Colunas que NÃO devem ser preenchidas com a mediana: é o alvo do modelo.
# Imputar o alvo fabricaria rótulos (todas as linhas sem dado virariam "a classe
# da mediana"). Elas ficam como NaN e o treino descarta essas linhas.
COLUNAS_SEM_IMPUTACAO = ["riscofogo"]

# Nomes alternativos usados pelo INPE em versões diferentes dos arquivos -> nome canônico.
# Ex.: o CSV "focos_br_todos-sats" traz `data_pas`, `numero_dias_sem_chuva` e `risco_fogo`.
ALIASES_COLUNAS = {
    "data_pas": "datahora",
    "data_hora_gmt": "datahora",
    "datahora_gmt": "datahora",
    "numero_dias_sem_chuva": "diasemchuva",
    "dias_sem_chuva": "diasemchuva",
    "risco_fogo": "riscofogo",
}

# Colunas com mais nulos que isso (após filtrar o estado) são descartadas:
# não ajudam o modelo e, com dropna no treino, apagariam todas as linhas.
LIMITE_NULOS_COLUNA = 0.95


class ETLError(Exception):
    """Erro de negócio do pipeline (arquivo inválido, colunas ausentes etc.)."""


def criar_diretorios() -> dict[str, Path]:
    """Cria data/raw, data/uploads e data/processed caso ainda não existam."""
    pastas = {"raw": RAW_DIR, "uploads": UPLOAD_DIR, "processed": PROCESSED_DIR}
    for nome, pasta in pastas.items():
        try:
            if pasta.exists():
                log.debug("Diretório já existe: %s", pasta)
            else:
                pasta.mkdir(parents=True, exist_ok=True)
                log.info("Diretório criado: %s", pasta)
        except OSError as exc:
            raise ETLError(f"Não foi possível criar o diretório '{nome}' ({pasta}): {exc}") from exc
    return pastas


# --------------------------------------------------------------------------- #
# Utilitários
# --------------------------------------------------------------------------- #
def _remover_acentos(texto: str) -> str:
    """'Município' -> 'Municipio'."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    )


def _mapear_colunas(originais: list) -> dict:
    """
    Mapeia cada nome original do cabeçalho para o nome canônico: minúsculas, sem
    acentos/espaços e com os aliases do INPE aplicados (ex.: data_pas -> datahora).
    """
    normalizados = {o: _remover_acentos(str(o)).strip().lower() for o in originais}
    presentes = set(normalizados.values())
    mapa = dict(normalizados)
    for original, nome in normalizados.items():
        destino = ALIASES_COLUNAS.get(nome)
        if destino and destino not in presentes:
            mapa[original] = destino
            presentes.add(destino)
            log.info("Coluna '%s' tratada como '%s'", nome, destino)
    return mapa


def _normalizar_texto(serie: pd.Series) -> pd.Series:
    """Texto em MAIÚSCULAS, sem acentos e sem espaços extras (para comparar/deduplicar)."""
    # Normaliza só os valores únicos (poucos) e mapeia: bem mais rápido em milhões de linhas.
    s = serie.astype("string").str.strip()
    mapa = {v: _remover_acentos(v).upper() for v in s.dropna().unique()}
    return s.map(mapa)


# --------------------------------------------------------------------------- #
# 1. EXTRACT
# --------------------------------------------------------------------------- #
def _caminho_cache(caminho: Path, estado: str | None, somente_uteis: bool) -> Path:
    """Nome do cache: muda se o arquivo mudar (tamanho/data) ou se os parâmetros de leitura mudarem."""
    st = caminho.stat()
    parametros = f"{estado}|{sorted(COLUNAS_UTEIS) if somente_uteis else 'todas'}"
    sufixo = hashlib.md5(parametros.encode()).hexdigest()[:8]
    return PROCESSED_DIR / "cache" / f"{caminho.stem}__{st.st_size}__{st.st_mtime_ns}__{sufixo}.pkl"


def ler_csv(
    caminho: str | Path,
    filtro_estado: str | None = None,
    somente_uteis: bool = True,
    usar_cache: bool = True,
) -> pd.DataFrame:
    """
    Lê um CSV do INPE da forma mais enxuta possível:

    - valida o cabeçalho ANTES de ler o arquivo (falha rápida em CSV errado);
    - lê só as colunas úteis (`usecols`);
    - filtra o estado bloco a bloco, descartando o resto logo na leitura;
    - guarda o resultado em cache (data/processed/cache); enquanto o arquivo não
      mudar, as próximas execuções (ex.: após um upload) pulam a leitura do CSV.

    Detecta o separador (',' ou ';') e testa encodings comuns. Levanta ETLError se
    o arquivo não existir, estiver ilegível/vazio ou não tiver as colunas mínimas.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise ETLError(f"Arquivo não encontrado: {caminho}")

    cache = _caminho_cache(caminho, filtro_estado, somente_uteis) if usar_cache else None
    if cache is not None and cache.exists():
        try:
            df = pd.read_pickle(cache)
            log.info("Cache encontrado para %s: %s linhas (CSV não precisou ser relido)",
                     caminho.name, f"{len(df):,}")
            return df
        except Exception as exc:  # cache corrompido/incompatível: relê o CSV
            log.warning("Cache inválido (%s); relendo o CSV.", exc)

    log.info("Lendo %s (%.1f MB)...", caminho.name, caminho.stat().st_size / 1e6)
    inicio = time.perf_counter()
    alvo = _remover_acentos(filtro_estado).upper() if filtro_estado else None

    ultimo_erro: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(caminho, "r", encoding=encoding, newline="") as f:
                amostra = f.read(8192)
            try:
                sep = csv.Sniffer().sniff(amostra, delimiters=",;\t|").delimiter
            except csv.Error:
                sep = ","

            # Cabeçalho primeiro: nomes canônicos, validação e colunas a ler.
            cabecalho = pd.read_csv(caminho, sep=sep, encoding=encoding, nrows=0)
            mapa = _mapear_colunas(list(cabecalho.columns))
            faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in mapa.values()]
            if faltando:
                raise ETLError(
                    f"Colunas ausentes em {caminho.name}: {faltando}. "
                    f"Colunas encontradas: {list(mapa.values())}"
                )
            usecols = [o for o, c in mapa.items() if c in COLUNAS_UTEIS] if somente_uteis else None

            pedacos, lidas, mantidas = [], 0, 0
            for pedaco in pd.read_csv(caminho, sep=sep, encoding=encoding,
                                      usecols=usecols, chunksize=TAMANHO_CHUNK):
                pedaco = pedaco.rename(columns=mapa)
                lidas += len(pedaco)
                if alvo is not None:
                    mascara = (_normalizar_texto(pedaco["estado"]) == alvo).fillna(False)
                    pedaco = pedaco[mascara.to_numpy(dtype=bool)]
                if len(pedaco):
                    pedacos.append(pedaco)
                    mantidas += len(pedaco)
                if lidas % TAMANHO_CHUNK == 0:  # bloco cheio: provavelmente há mais
                    log.info("  %s: %s linhas lidas, %s mantidas...", caminho.name,
                             f"{lidas:,}", f"{mantidas:,}")
            break
        except UnicodeDecodeError as exc:
            ultimo_erro = exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ETLError(f"CSV inválido ou vazio ({caminho.name}): {exc}") from exc
    else:
        raise ETLError(f"Não foi possível decodificar {caminho.name}: {ultimo_erro}")

    if lidas == 0:
        raise ETLError(f"O arquivo {caminho.name} não tem linhas.")
    if not pedacos:
        raise ETLError(f"Nenhuma linha de '{filtro_estado}' em {caminho.name} ({lidas:,} linhas lidas).")
    df = pedacos[0] if len(pedacos) == 1 else pd.concat(pedacos, ignore_index=True)

    log.info(
        "Lido %s: %s linhas lidas, %s mantidas, %d colunas em %.1fs",
        caminho.name, f"{lidas:,}", f"{mantidas:,}", df.shape[1], time.perf_counter() - inicio,
    )

    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            for antigo in cache.parent.glob(f"{caminho.stem}__*.pkl"):  # remove versões antigas
                antigo.unlink(missing_ok=True)
            df.to_pickle(cache)
            log.info("Cache salvo: %s", cache.name)
        except OSError as exc:
            log.warning("Não foi possível salvar o cache: %s", exc)
    return df


def validar_colunas(df: pd.DataFrame, origem: str = "dataset") -> None:
    """Garante que as colunas mínimas existem; caso contrário, informa quais faltam."""
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ETLError(
            f"Colunas ausentes em {origem}: {faltando}. "
            f"Colunas encontradas: {list(df.columns)}"
        )


def carregar_base(caminho_base: str | Path) -> pd.DataFrame:
    """Lê o CSV base (já existente no projeto)."""
    df = ler_csv(caminho_base, filtro_estado=ESTADO_ALVO)
    validar_colunas(df, origem=Path(caminho_base).name)
    return df


def carregar_upload(caminho_upload: str | Path) -> pd.DataFrame:
    """Lê um CSV enviado pelo usuário e valida as colunas mínimas."""
    df = ler_csv(caminho_upload, filtro_estado=ESTADO_ALVO)
    validar_colunas(df, origem=f"upload '{Path(caminho_upload).name}'")
    return df


def concatenar_bases(dataframes: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatena base antiga + novos uploads. Colunas que existem só em alguns
    arquivos são mantidas (preenchidas com NaN nos demais).
    """
    lista = [d for d in dataframes if d is not None and not d.empty]
    if not lista:
        raise ETLError("Nenhum dado para concatenar.")
    df = pd.concat(lista, ignore_index=True, sort=False)
    log.info("Concatenação: %d linhas no total", len(df))
    return df


# --------------------------------------------------------------------------- #
# 2. LIMPEZA E FILTRAGEM
# --------------------------------------------------------------------------- #
def filtrar_estado(df: pd.DataFrame, estado: str = ESTADO_ALVO) -> pd.DataFrame:
    """Mantém apenas as linhas do estado desejado (comparação sem acento/caixa)."""
    df = df.copy()
    df["estado"] = _normalizar_texto(df["estado"])
    antes = len(df)
    df = df[df["estado"] == _remover_acentos(estado).upper()]
    log.info("Filtro estado=%s: %d -> %d linhas", estado, antes, len(df))
    if df.empty:
        raise ETLError(f"Nenhuma linha encontrada para o estado '{estado}'.")
    return df.reset_index(drop=True)


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Troca -999 (sentinela do INPE) por NaN.
    - Padroniza textos categóricos.
    - Descarta linhas sem informação essencial (data, município, coordenadas).
    - Preenche nulos restantes: numéricos com a mediana (exceto o alvo `riscofogo`,
      que fica NaN), categóricos com 'DESCONHECIDO'.
    - Remove duplicatas.
    """
    df = df.copy()
    n_inicial = len(df)

    # Sentinela -999 -> NaN nas colunas numéricas conhecidas
    for col in COLUNAS_NUMERICAS_SENTINELA:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(VALOR_SENTINELA, np.nan)

    # Descarta colunas praticamente vazias (ex.: id_area_industrial)
    vazias = [
        c for c in df.columns
        if c not in COLUNAS_OBRIGATORIAS and df[c].isna().mean() > LIMITE_NULOS_COLUNA
    ]
    if vazias:
        log.warning("Colunas descartadas por serem >%d%% vazias: %s", LIMITE_NULOS_COLUNA * 100, vazias)
        df = df.drop(columns=vazias)

    # Coordenadas precisam ser numéricas
    for col in ("latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Padroniza texto das categóricas
    for col in COLUNAS_CATEGORICAS:
        if col in df.columns:
            df[col] = _normalizar_texto(df[col])

    # Linhas sem campos essenciais não servem para análise nem para o modelo
    df = df.dropna(subset=COLUNAS_OBRIGATORIAS)

    # Nulos restantes
    for col in COLUNAS_CATEGORICAS:
        if col in df.columns:
            df[col] = df[col].fillna("DESCONHECIDO")
    for col in COLUNAS_NUMERICAS_SENTINELA:
        if col in COLUNAS_SEM_IMPUTACAO:
            if col in df.columns:
                log.info("Coluna '%s' mantida sem imputação: %s nulos (%.1f%%).",
                         col, f"{int(df[col].isna().sum()):,}", df[col].isna().mean() * 100)
            continue
        if col in df.columns and df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())

    # Duplicatas (mesmo foco detectado/registrado mais de uma vez)
    antes_dup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.info(
        "Limpeza: %d -> %d linhas (%d descartadas por nulos essenciais, %d duplicadas)",
        n_inicial, len(df), n_inicial - antes_dup, antes_dup - len(df),
    )
    if df.empty:
        raise ETLError("Todas as linhas foram descartadas na limpeza.")
    return df


# --------------------------------------------------------------------------- #
# 3. TRANSFORMAÇÃO E FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
def criar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    """Converte `datahora` para datetime e cria `ano`, `mes` e `dia`."""
    df = df.copy()
    df["datahora"] = pd.to_datetime(df["datahora"], errors="coerce")

    invalidas = df["datahora"].isna().sum()
    if invalidas:
        log.warning("%d linhas com datahora inválida foram descartadas", invalidas)
        df = df.dropna(subset=["datahora"])

    df["ano"] = df["datahora"].dt.year.astype("int16")
    df["mes"] = df["datahora"].dt.month.astype("int8")
    df["dia"] = df["datahora"].dt.day.astype("int8")
    return df.reset_index(drop=True)


def calcular_variacao_mensal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agrupa focos por município e mês e calcula a variação % em relação ao mês anterior.

    Meses sem nenhum foco entram como 0 (reindex num calendário contínuo), para que
    a comparação seja sempre com o mês imediatamente anterior e não com o último
    mês que teve registro.

    Retorna:
        df            : dataset original com `focos_mes` e `variacao_focos_pct`.
        df_variacao   : tabela agregada (municipio, ano, mes, focos, variacao_pct),
                        útil direto para os KPIs do front-end.
    """
    df = df.copy()
    df["periodo"] = df["datahora"].dt.to_period("M")

    contagem = df.groupby(["municipio", "periodo"]).size().rename("focos").reset_index()

    # Calendário contínuo por município
    todos_periodos = pd.period_range(df["periodo"].min(), df["periodo"].max(), freq="M")
    municipios = contagem["municipio"].unique()
    grade = pd.MultiIndex.from_product([municipios, todos_periodos], names=["municipio", "periodo"])
    contagem = (
        contagem.set_index(["municipio", "periodo"])
        .reindex(grade, fill_value=0)
        .reset_index()
        .sort_values(["municipio", "periodo"])
    )

    # Variação % vs mês anterior. Mês anterior = 0 gera inf/NaN -> 0 (sem base de comparação).
    contagem["variacao_pct"] = (
        contagem.groupby("municipio")["focos"].pct_change(fill_method=None) * 100
    )
    contagem["variacao_pct"] = (
        contagem["variacao_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0).round(2)
    )
    contagem["ano"] = contagem["periodo"].dt.year.astype("int16")
    contagem["mes"] = contagem["periodo"].dt.month.astype("int8")

    # Traz as métricas de volta para cada linha do dataset original
    df = df.merge(
        contagem[["municipio", "periodo", "focos", "variacao_pct"]].rename(
            columns={"focos": "focos_mes", "variacao_pct": "variacao_focos_pct"}
        ),
        on=["municipio", "periodo"],
        how="left",
    ).drop(columns="periodo")

    df_variacao = contagem[["municipio", "ano", "mes", "focos", "variacao_pct"]].reset_index(drop=True)
    return df, df_variacao


# --------------------------------------------------------------------------- #
# 4. ENCODING PARA ML
# --------------------------------------------------------------------------- #
def codificar_categoricas(
    df: pd.DataFrame,
    colunas: list[str] | None = None,
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Aplica LabelEncoder criando colunas `<col>_enc` (as originais são mantidas
    para dashboards).

    - Sem `encoders`: ajusta (fit) novos encoders com os dados atuais.
    - Com `encoders`: reutiliza os existentes (inferência); categorias inéditas
      viram -1 em vez de quebrar.
    """
    df = df.copy()
    colunas = colunas or COLUNAS_CATEGORICAS
    ajustar = encoders is None
    encoders = {} if ajustar else dict(encoders)

    for col in colunas:
        if col not in df.columns:
            log.warning("Coluna categórica '%s' ausente; pulando encoding.", col)
            continue

        valores = df[col].astype(str)
        if ajustar:
            enc = LabelEncoder().fit(valores)
            encoders[col] = enc
            df[f"{col}_enc"] = enc.transform(valores)
        else:
            enc = encoders[col]
            mapa = {classe: i for i, classe in enumerate(enc.classes_)}
            df[f"{col}_enc"] = valores.map(mapa).fillna(-1).astype(int)

    return df, encoders


def salvar_encoders(encoders: dict[str, LabelEncoder], pasta: Path = PROCESSED_DIR) -> Path:
    """Salva os encoders para reutilizar na inferência."""
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / "encoders.pkl"
    with open(caminho, "wb") as f:
        pickle.dump(encoders, f)
    log.info("Encoders salvos em %s", caminho)
    return caminho


def carregar_encoders(pasta: Path = PROCESSED_DIR) -> dict[str, LabelEncoder]:
    """Carrega os encoders salvos (para as rotas Flask / inferência)."""
    caminho = pasta / "encoders.pkl"
    if not caminho.exists():
        raise ETLError(f"Encoders não encontrados em {caminho}. Rode o ETL primeiro.")
    with open(caminho, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# 5. LOAD
# --------------------------------------------------------------------------- #
def salvar_processado(
    df: pd.DataFrame,
    df_variacao: pd.DataFrame,
    pasta: Path = PROCESSED_DIR,
) -> dict[str, Path]:
    """
    Grava o dataset final e a tabela de variação mensal em data/processed/.
    CSV é sempre gerado; Parquet só se `pyarrow`/`fastparquet` estiver instalado.
    """
    pasta.mkdir(parents=True, exist_ok=True)
    saidas: dict[str, Path] = {}

    for nome, dados in (("queimadas_tocantins", df), ("variacao_mensal_municipio", df_variacao)):
        caminho_csv = pasta / f"{nome}.csv"
        dados.to_csv(caminho_csv, index=False)
        saidas[f"{nome}_csv"] = caminho_csv

        try:
            caminho_pq = pasta / f"{nome}.parquet"
            dados.to_parquet(caminho_pq, index=False)
            saidas[f"{nome}_parquet"] = caminho_pq
        except ImportError:
            log.warning("pyarrow não instalado; Parquet de '%s' não foi gerado.", nome)

    for chave, caminho in saidas.items():
        log.info("Salvo [%s]: %s", chave, caminho)
    return saidas


# --------------------------------------------------------------------------- #
# ORQUESTRAÇÃO
# --------------------------------------------------------------------------- #
def executar_etl(
    caminho_base: str | Path | None = None,
    caminhos_upload: Iterable[str | Path] | None = None,
    pasta_saida: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """
    Executa o pipeline completo e devolve o DataFrame final.

    Uploads com problema (colunas ausentes, arquivo ilegível) são ignorados com
    aviso, sem derrubar o processamento da base principal.
    """
    inicio = time.perf_counter()
    criar_diretorios()
    log.info("=== ETL iniciado ===")

    # 1. Extract
    log.info("[1/5] Extração: lendo base e uploads...")
    fontes: list[pd.DataFrame] = []
    if caminho_base:
        fontes.append(carregar_base(caminho_base))

    uploads = list(caminhos_upload or [])
    for i, caminho in enumerate(uploads, start=1):
        log.info("      Upload %d/%d: %s", i, len(uploads), Path(caminho).name)
        try:
            fontes.append(carregar_upload(caminho))
        except ETLError as exc:
            log.error("Upload ignorado: %s", exc)

    df = concatenar_bases(fontes)

    # 2. Clean
    log.info("[2/5] Limpeza e filtragem (estado=%s)...", ESTADO_ALVO)
    df = filtrar_estado(df)
    df = limpar_dados(df)

    # 3. Transform
    log.info("[3/5] Transformação: datas e variação mensal de focos...")
    df = criar_features_temporais(df)
    df = df.drop_duplicates().reset_index(drop=True)  # garante unicidade após conversões
    df, df_variacao = calcular_variacao_mensal(df)

    # 4. Encode
    log.info("[4/5] Encoding de variáveis categóricas...")
    df, encoders = codificar_categoricas(df)
    salvar_encoders(encoders, pasta_saida)

    # 5. Load
    log.info("[5/5] Gravando resultados em %s...", pasta_saida)
    salvar_processado(df, df_variacao, pasta_saida)

    log.info(
        "=== ETL concluído em %.1fs: %s linhas, %d colunas ===",
        time.perf_counter() - inicio, f"{len(df):,}", df.shape[1],
    )
    return df


# --------------------------------------------------------------------------- #
# EXEMPLO DE USO
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    criar_diretorios()

    # Caminhos fictícios — troque pelos arquivos reais do projeto.
    CSV_BASE = RAW_DIR / "focos_2025.csv"
    CSV_UPLOAD = UPLOAD_DIR / "focos_2024.csv"

    try:
        resultado = executar_etl(
            caminho_base=CSV_BASE,
            caminhos_upload=[CSV_UPLOAD],
        )
        print(resultado.head())
        print("\nColunas finais:", list(resultado.columns))
    except ETLError as erro:
        log.error("Falha no ETL: %s", erro)
    except Exception:
        log.exception("Erro inesperado no ETL")
        raise