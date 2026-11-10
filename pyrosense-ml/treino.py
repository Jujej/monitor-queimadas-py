"""
treino.py — Treinamento do modelo de risco de fogo (PyroSense ML).

Lê o dataset gerado pelo etl.py (data/processed/queimadas_tocantins.csv), treina um
classificador que prevê a classe de risco (Baixo / Médio / Alto) e guarda o modelo
em disco. Não depende do Flask: o back-end só precisa chamar `treinar()` e `prever()`.

Decisões de metodologia (e por quê):
    - ALVO sem imputação: linhas sem `riscofogo` (-999 no INPE) são descartadas do
      treino, em vez de receberem a mediana.
    - SPLIT TEMPORAL: treina no passado e testa no futuro. Focos do mesmo dia e da
      mesma região são quase iguais; um split aleatório colocaria "gêmeos" em treino
      e teste e inflaria a acurácia.
    - PRÉ-PROCESSAMENTO DENTRO DO PIPELINE: imputação, encoding e escala são
      ajustados só com o treino (sem vazar estatísticas do teste) e viajam junto com
      o modelo salvo, então a predição usa exatamente o mesmo tratamento.
    - BASELINE + MÉTRICAS COMPLETAS: acurácia sozinha engana com classes
      desbalanceadas. Reportamos baseline, acurácia balanceada, F1 macro, métricas
      por classe e matriz de confusão.
    - DOIS CONJUNTOS DE FEATURES: o `riscofogo` do INPE é calculado em grande parte
      a partir de dias sem chuva e precipitação. Com essas colunas como entrada, o
      modelo tende a só reaprender a fórmula do INPE. O conjunto "sem_clima" mostra
      quanto o modelo acerta sem elas.

Uso na linha de comando:
    python treino.py                                   # random forest, conjunto completo
    python treino.py --modelo decision_tree --parametro 6
    python treino.py --conjunto sem_clima
    python treino.py --comparar                        # tabela de modelos x conjuntos
    python treino.py --prever '{"latitude": -10.2, "longitude": -48.3, "mes": 8}'

Uso no código:
    import treino
    resultado = treino.treinar(modelo="random_forest", parametro=8)
    resposta  = treino.prever({"latitude": -10.2, "longitude": -48.3, "mes": 8, ...})
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import etl  # caminhos e logging compartilhados com o pipeline de ETL

log = logging.getLogger("treino")

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
SEED = 42

PROCESSED_CSV = etl.PROCESSED_DIR / "queimadas_tocantins.csv"
MODELO_PATH = etl.PROCESSED_DIR / "modelo_risco.joblib"
METRICAS_PATH = etl.PROCESSED_DIR / "modelo_risco_metricas.json"

# Alvo: `riscofogo` (0 a 1, escala do INPE) discretizado em 3 classes.
# Mesmos limiares do app.py (classe_risco), para o front e o modelo concordarem.
COLUNA_ALVO = "riscofogo"
CLASSES = ["Baixo", "Médio", "Alto"]
LIMIARES = (0.4, 0.7)  # [0, 0.4) Baixo | [0.4, 0.7) Médio | [0.7, ...) Alto

# Conjuntos de features. O padrão NÃO inclui `municipio` (alta cardinalidade e
# redundante com latitude/longitude), mas ele pode ser pedido em `features=`.
FEATURES_BASE = ["latitude", "longitude", "mes", "frp", "bioma", "satelite"]
FEATURES_CLIMA = ["diasemchuva", "precipitacao"]
CONJUNTOS_FEATURES = {
    "completo": FEATURES_BASE + FEATURES_CLIMA,
    "sem_clima": FEATURES_BASE,
}

# Colunas do ETL que NÃO podem ser features: são calculadas com a base inteira
# (inclusive o futuro do próprio mês), então não existem na hora de prever.
COLUNAS_PROIBIDAS = {"focos_mes", "variacao_focos_pct", "ano", "dia", "datahora"}

MODELOS = ("decision_tree", "random_forest", "knn")


class TreinoError(Exception):
    """Erro de negócio do treino/predição (dados insuficientes, entrada inválida etc.)."""


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
def carregar_dataset(caminho: str | Path | None = None) -> pd.DataFrame:
    """Lê o CSV processado pelo ETL e padroniza o nome do alvo."""
    caminho = Path(caminho) if caminho else PROCESSED_CSV
    if not caminho.exists():
        raise TreinoError(f"Dataset processado não encontrado em {caminho}. Rode o ETL primeiro.")
    df = pd.read_csv(caminho, parse_dates=["datahora"])
    return df.rename(columns={"risco_fogo": COLUNA_ALVO})


def classificar_risco(serie: pd.Series, limiares: tuple[float, float] = LIMIARES) -> pd.Series:
    """Converte o risco numérico (0-1) em Baixo / Médio / Alto. NaN continua NaN."""
    return pd.cut(
        serie,
        bins=[-np.inf, limiares[0], limiares[1], np.inf],
        labels=CLASSES,
        right=False,
    )


def _resolver_features(
    df: pd.DataFrame, features: list[str] | None, conjunto: str
) -> tuple[list[str], list[str]]:
    """Define as features finais e devolve (features, avisos)."""
    avisos: list[str] = []

    if features:
        pedidas = list(dict.fromkeys(features))  # tira repetidas, mantém a ordem
    else:
        if conjunto not in CONJUNTOS_FEATURES:
            raise TreinoError(f"Conjunto '{conjunto}' inválido. Use: {list(CONJUNTOS_FEATURES)}.")
        pedidas = list(CONJUNTOS_FEATURES[conjunto])

    final = []
    for col in pedidas:
        if col == COLUNA_ALVO or col == "risco_fogo":
            avisos.append(f"'{col}' é o alvo e foi removida das features.")
        elif col in COLUNAS_PROIBIDAS:
            avisos.append(f"'{col}' foi removida das features: não está disponível na hora de prever.")
        elif col.endswith("_enc"):
            avisos.append(f"'{col}' (encoding do ETL) foi removida: o pipeline faz o próprio encoding.")
        elif col not in df.columns:
            avisos.append(f"Feature '{col}' não existe na base e foi ignorada.")
        else:
            final.append(col)

    if not final:
        raise TreinoError("Nenhuma feature válida para treinar.")
    return final, avisos


def preparar_dados(
    df: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, int]]:
    """
    Monta X, y e as datas, descartando linhas sem alvo ou sem data.
    Devolve também um resumo do que foi descartado.
    """
    if COLUNA_ALVO not in df.columns:
        raise TreinoError(f"Coluna alvo '{COLUNA_ALVO}' ausente na base.")

    alvo = pd.to_numeric(df[COLUNA_ALVO], errors="coerce")
    datas = pd.to_datetime(df["datahora"], errors="coerce")
    valido = alvo.notna() & datas.notna()

    descartes = {
        "linhas_total": int(len(df)),
        "sem_alvo": int(alvo.isna().sum()),
        "sem_data": int(datas.isna().sum()),
        "linhas_usadas": int(valido.sum()),
    }
    if descartes["linhas_usadas"] < 100:
        raise TreinoError(
            f"Só {descartes['linhas_usadas']} linhas com alvo e data válidos: insuficiente para treinar."
        )

    X = df.loc[valido, features].reset_index(drop=True)
    y = classificar_risco(alvo[valido]).astype(str).reset_index(drop=True)
    datas = datas[valido].reset_index(drop=True)
    return X, y, datas, descartes


def dividir_temporal(
    datas: pd.Series, frac_teste: float = 0.2, ano_teste: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split por tempo: treino = passado, teste = futuro.

    - `ano_teste`: tudo a partir desse ano vai para teste.
    - senão: os `frac_teste` dias mais recentes vão para teste (o corte é num dia
      inteiro, para um mesmo dia não ficar dividido entre treino e teste).
    """
    if ano_teste is not None:
        teste = (datas.dt.year >= ano_teste).to_numpy()
    else:
        if not 0.05 <= frac_teste <= 0.5:
            raise TreinoError("frac_teste deve ficar entre 0.05 e 0.5.")
        dias = datas.dt.normalize()
        posicao = min(int(len(dias) * (1 - frac_teste)), len(dias) - 1)
        corte = dias.sort_values().iloc[posicao]
        teste = (dias >= corte).to_numpy()

    treino = ~teste
    if treino.sum() < 50 or teste.sum() < 50:
        raise TreinoError(
            f"Split temporal inviável (treino={int(treino.sum())}, teste={int(teste.sum())}). "
            "Ajuste frac_teste/ano_teste ou carregue mais dados."
        )
    return np.flatnonzero(treino), np.flatnonzero(teste)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def montar_pipeline(
    modelo: str,
    parametro: int,
    numericas: list[str],
    categoricas: list[str],
    n_arvores: int = 200,
) -> Pipeline:
    """
    Pipeline completo: imputação -> encoding/escala -> classificador.

    `parametro`: profundidade máxima (árvore e floresta) ou número de vizinhos (kNN).
    Árvores e floresta usam class_weight='balanced' para não ignorar a classe rara.
    """
    if modelo not in MODELOS:
        raise TreinoError(f"Modelo '{modelo}' inválido. Use: {list(MODELOS)}.")
    parametro = max(1, int(parametro))

    passos_num: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if modelo == "knn":
        passos_num.append(("scaler", StandardScaler()))  # kNN é sensível à escala

    if modelo == "knn":
        # Ordem numérica entre categorias não faz sentido para distância: one-hot.
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    passos_cat = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="DESCONHECIDO")),
        ("encoder", encoder),
    ]

    transformadores = []
    if numericas:
        transformadores.append(("num", Pipeline(passos_num), numericas))
    if categoricas:
        transformadores.append(("cat", Pipeline(passos_cat), categoricas))
    pre = ColumnTransformer(transformadores)

    if modelo == "decision_tree":
        clf = DecisionTreeClassifier(
            max_depth=parametro, min_samples_leaf=5, class_weight="balanced", random_state=SEED
        )
    elif modelo == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=n_arvores, max_depth=parametro, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
        )
    else:
        clf = KNeighborsClassifier(n_neighbors=parametro, n_jobs=-1)

    return Pipeline([("pre", pre), ("modelo", clf)])


def _arredondar(valor: float, casas: int = 4) -> float:
    return round(float(valor), casas)


def avaliar(y_real: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    """Métricas globais, por classe e matriz de confusão (ordem fixa Baixo/Médio/Alto)."""
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_real, y_pred, labels=CLASSES, zero_division=0
    )
    matriz = confusion_matrix(y_real, y_pred, labels=CLASSES)
    return {
        "acuracia": _arredondar(accuracy_score(y_real, y_pred)),
        "acuracia_balanceada": _arredondar(balanced_accuracy_score(y_real, y_pred)),
        "f1_macro": _arredondar(f1_score(y_real, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "por_classe": {
            c: {
                "precisao": _arredondar(prec[i]),
                "recall": _arredondar(rec[i]),
                "f1": _arredondar(f1[i]),
                "suporte": int(sup[i]),
            }
            for i, c in enumerate(CLASSES)
        },
        "matriz_confusao": {"rotulos": CLASSES, "valores": matriz.tolist()},
    }


def _importancias(pipeline: Pipeline, top: int = 10) -> list[dict[str, float]] | None:
    """Importância das features (só modelos de árvore)."""
    clf = pipeline.named_steps["modelo"]
    if not hasattr(clf, "feature_importances_"):
        return None
    nomes = [n.split("__", 1)[-1] for n in pipeline.named_steps["pre"].get_feature_names_out()]
    pares = sorted(zip(nomes, clf.feature_importances_), key=lambda p: p[1], reverse=True)
    return [{"feature": n, "importancia": _arredondar(v)} for n, v in pares[:top]]


def _distribuicao(y: pd.Series) -> dict[str, float]:
    freq = y.value_counts(normalize=True)
    return {c: _arredondar(freq.get(c, 0.0)) for c in CLASSES}


def _periodo(datas: pd.Series) -> dict[str, str]:
    return {"inicio": datas.min().date().isoformat(), "fim": datas.max().date().isoformat()}


# --------------------------------------------------------------------------- #
# Treino
# --------------------------------------------------------------------------- #
def treinar(
    df: pd.DataFrame | None = None,
    modelo: str = "random_forest",
    parametro: int = 8,
    conjunto: str = "completo",
    features: list[str] | None = None,
    frac_teste: float = 0.2,
    ano_teste: int | None = None,
    n_arvores: int = 200,
    validacao_cruzada: bool = True,
    n_splits_cv: int = 3,
    salvar: bool = True,
) -> dict[str, Any]:
    """
    Treina, avalia e (opcionalmente) salva o modelo. Devolve um dicionário
    serializável em JSON.

    Compatível com o /api/treinar atual: as chaves `acuracia` (em %),
    `amostras_treino`, `amostras_teste` e `classes` continuam presentes.
    """
    inicio = time.perf_counter()
    df = carregar_dataset() if df is None else df.rename(columns={"risco_fogo": COLUNA_ALVO})
    parametro = max(1, min(int(parametro), 50))

    feats, avisos = _resolver_features(df, features, conjunto)
    X, y, datas, descartes = preparar_dados(df, feats)
    for aviso in avisos:
        log.warning(aviso)
    log.info("Dados: %s linhas usadas de %s (%s sem alvo descartadas).",
             f"{descartes['linhas_usadas']:,}", f"{descartes['linhas_total']:,}", f"{descartes['sem_alvo']:,}")

    # Ordena por tempo (necessário para o split e para a validação cruzada temporal)
    ordem = np.argsort(datas.to_numpy(), kind="stable")
    X, y, datas = X.iloc[ordem].reset_index(drop=True), y.iloc[ordem].reset_index(drop=True), datas.iloc[ordem].reset_index(drop=True)

    idx_treino, idx_teste = dividir_temporal(datas, frac_teste, ano_teste)
    X_tr, X_te = X.iloc[idx_treino], X.iloc[idx_teste]
    y_tr, y_te = y.iloc[idx_treino], y.iloc[idx_teste]
    if y_tr.nunique() < 2:
        raise TreinoError("O período de treino tem uma única classe de risco; não há o que aprender.")

    numericas = [c for c in feats if is_numeric_dtype(X[c])]
    categoricas = [c for c in feats if c not in numericas]

    pipe = montar_pipeline(modelo, parametro, numericas, categoricas, n_arvores)

    # Baseline: sempre a classe mais frequente do treino
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    metricas_base = avaliar(y_te, base.predict(X_te))

    pipe.fit(X_tr, y_tr)
    metricas = avaliar(y_te, pipe.predict(X_te))
    metricas_treino = avaliar(y_tr, pipe.predict(X_tr))

    # Validação cruzada temporal só no treino (o teste fica intocado)
    cv = None
    if validacao_cruzada:
        try:
            res = cross_validate(
                montar_pipeline(modelo, parametro, numericas, categoricas, n_arvores),
                X_tr, y_tr, cv=TimeSeriesSplit(n_splits=n_splits_cv),
                scoring={"f1_macro": "f1_macro", "acuracia_balanceada": "balanced_accuracy"},
                error_score=np.nan,
            )
            cv = {
                "n_splits": n_splits_cv,
                "f1_macro_media": _arredondar(np.nanmean(res["test_f1_macro"])),
                "f1_macro_desvio": _arredondar(np.nanstd(res["test_f1_macro"])),
                "acuracia_balanceada_media": _arredondar(np.nanmean(res["test_acuracia_balanceada"])),
            }
        except Exception as exc:  # CV é diagnóstico: não derruba o treino
            log.warning("Validação cruzada pulada: %s", exc)

    # Categorias vistas no treino (para resolver entradas na hora de prever)
    categorias: dict[str, list[str]] = {}
    if categoricas:
        enc = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["encoder"]
        categorias = {c: [str(v) for v in cats] for c, cats in zip(categoricas, enc.categories_)}

    gap = metricas_treino["f1_macro"] - metricas["f1_macro"]
    if gap > 0.15:
        avisos.append(
            f"Possível overfitting: F1 macro {metricas_treino['f1_macro']:.2f} no treino "
            f"contra {metricas['f1_macro']:.2f} no teste. Tente reduzir o parâmetro."
        )

    resultado: dict[str, Any] = {
        # --- chaves do contrato antigo do /api/treinar ---
        "acuracia": round(metricas["acuracia"] * 100, 2),
        "amostras_treino": int(len(idx_treino)),
        "amostras_teste": int(len(idx_teste)),
        "classes": CLASSES,
        # --- novidades ---
        "baseline_acuracia": round(metricas_base["acuracia"] * 100, 2),
        "metricas_teste": metricas,
        "metricas_treino": {"acuracia": metricas_treino["acuracia"], "f1_macro": metricas_treino["f1_macro"]},
        "metricas_baseline": {"acuracia": metricas_base["acuracia"], "f1_macro": metricas_base["f1_macro"]},
        "validacao_cruzada": cv,
        "importancias": _importancias(pipe),
        "split": {
            "tipo": "temporal",
            "treino": {**_periodo(datas.iloc[idx_treino]), "distribuicao_classes": _distribuicao(y_tr)},
            "teste": {**_periodo(datas.iloc[idx_teste]), "distribuicao_classes": _distribuicao(y_te)},
        },
        "dados": descartes,
        "config": {
            "modelo": modelo, "parametro": parametro, "n_arvores": n_arvores if modelo == "random_forest" else None,
            "conjunto": None if features else conjunto,
            "features": feats, "numericas": numericas, "categoricas": categoricas,
            "limiares_risco": list(LIMIARES),
        },
        "avisos": avisos,
        "duracao_s": round(time.perf_counter() - inicio, 1),
    }

    if salvar:
        pacote = {
            "pipeline": pipe,
            "features": {"numericas": numericas, "categoricas": categoricas},
            "categorias": categorias,
            "medianas": {c: _arredondar(X_tr[c].median(), 4) for c in numericas},
            "classes": CLASSES,
            "limiares": LIMIARES,
            "config": resultado["config"],
            "metricas": {"acuracia": metricas["acuracia"], "f1_macro": metricas["f1_macro"]},
            "treinado_em": datetime.now().isoformat(timespec="seconds"),
            "versao_sklearn": sklearn.__version__,
        }
        resultado["caminho_modelo"] = str(salvar_modelo(pacote))
        METRICAS_PATH.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Treino concluído em %.1fs: acurácia=%.2f%% | baseline=%.2f%% | F1 macro=%.3f",
             resultado["duracao_s"], resultado["acuracia"], resultado["baseline_acuracia"], metricas["f1_macro"])
    return resultado


def comparar(
    df: pd.DataFrame | None = None,
    modelos: tuple[str, ...] = MODELOS,
    conjuntos: tuple[str, ...] = ("completo", "sem_clima"),
    parametros: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Treina cada combinação modelo x conjunto (sem salvar) e devolve uma tabela ordenada por F1 macro."""
    parametros = parametros or {"decision_tree": 8, "random_forest": 10, "knn": 15}
    df = carregar_dataset() if df is None else df
    linhas = []
    for conjunto in conjuntos:
        for modelo in modelos:
            r = treinar(df, modelo=modelo, parametro=parametros.get(modelo, 8), conjunto=conjunto,
                        validacao_cruzada=False, salvar=False)
            m = r["metricas_teste"]
            linhas.append({
                "modelo": modelo, "conjunto": conjunto, "parametro": r["config"]["parametro"],
                "acuracia": m["acuracia"], "acuracia_balanceada": m["acuracia_balanceada"],
                "f1_macro": m["f1_macro"], "baseline_acuracia": r["metricas_baseline"]["acuracia"],
            })
    return sorted(linhas, key=lambda l: l["f1_macro"], reverse=True)


# --------------------------------------------------------------------------- #
# Persistência e predição
# --------------------------------------------------------------------------- #
_cache: dict[str, Any] = {"mtime": None, "pacote": None}


def salvar_modelo(pacote: dict[str, Any], caminho: Path = MODELO_PATH) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, caminho, compress=3)
    _cache["mtime"] = None  # força recarregar na próxima predição
    log.info("Modelo salvo em %s (%.1f MB)", caminho, caminho.stat().st_size / 1e6)
    return caminho


def carregar_modelo(caminho: Path = MODELO_PATH) -> dict[str, Any]:
    """Carrega o modelo salvo (com cache em memória; recarrega se o arquivo mudar)."""
    if not caminho.exists():
        raise TreinoError("Nenhum modelo treinado encontrado. Treine o modelo primeiro.")
    mtime = caminho.stat().st_mtime_ns
    if _cache["pacote"] is None or _cache["mtime"] != mtime:
        _cache["pacote"] = joblib.load(caminho)
        _cache["mtime"] = mtime
    return _cache["pacote"]


def _sem_acento_maiusculo(texto: str) -> str:
    sem = "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))
    return sem.strip().upper()


def prever(entradas: dict[str, Any], pacote: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Prevê a classe de risco para uma linha de entrada.

    - Campos ausentes/vazios viram NaN e são imputados pelo próprio pipeline
      (listados em `campos_ausentes`).
    - Categorias desconhecidas são aceitas (listadas em `categorias_desconhecidas`),
      mas a predição fica menos confiável.
    - Valor numérico inválido levanta TreinoError com o nome do campo.
    """
    pacote = pacote or carregar_modelo()
    numericas = pacote["features"]["numericas"]
    categoricas = pacote["features"]["categoricas"]
    entradas = entradas or {}

    linha: dict[str, Any] = {}
    ausentes, desconhecidas = [], []

    for col in numericas:
        bruto = entradas.get(col)
        if bruto is None or (isinstance(bruto, str) and not bruto.strip()):
            linha[col] = np.nan
            ausentes.append(col)
            continue
        try:
            linha[col] = float(str(bruto).replace(",", "."))
        except ValueError:
            raise TreinoError(f"Valor inválido para '{col}': {bruto!r} (esperado um número).") from None

    for col in categoricas:
        bruto = entradas.get(col)
        if bruto is None or (isinstance(bruto, str) and not bruto.strip()):
            linha[col] = np.nan
            ausentes.append(col)
            continue
        conhecidas = pacote["categorias"].get(col, [])
        texto = str(bruto)
        if texto in conhecidas:
            linha[col] = texto
        elif _sem_acento_maiusculo(texto) in conhecidas:  # o ETL guarda texto em MAIÚSCULAS sem acento
            linha[col] = _sem_acento_maiusculo(texto)
        else:
            linha[col] = texto
            desconhecidas.append(col)

    df_in = pd.DataFrame([linha], columns=numericas + categoricas)
    pipe = pacote["pipeline"]
    classe = str(pipe.predict(df_in)[0])
    probs = dict(zip(pipe.classes_, pipe.predict_proba(df_in)[0]))

    return {
        "predicao": classe,
        "probabilidades": {c: round(float(probs.get(c, 0.0)) * 100, 1) for c in pacote["classes"]},
        "campos_ausentes": ausentes,
        "categorias_desconhecidas": desconhecidas,
    }
