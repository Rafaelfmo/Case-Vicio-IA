from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).resolve().parent
PREP = BASE / "artefatos" / "preparacao"
OUT = BASE / "artefatos" / "modelagem"
OUT.mkdir(parents=True, exist_ok=True)

questions = pd.read_csv(PREP / "questoes_enriquecidas.csv").fillna("")
themes = pd.read_csv(PREP / "temas_enriquecidos.csv").fillna("")
truth = pd.read_csv(PREP / "folha_verdade_preservada.csv").fillna("")
terms = {}
for line in (BASE / "artefatos" / "termos_temas.txt").read_text(encoding="utf-8").splitlines():
    if "|" in line and not line.startswith("#"):
        key, value = line.split("|", 1)
        terms[int(key)] = value

oab = pd.read_parquet(
    "hf://datasets/eduagarcia/oab_exams/data/train-00000-of-00001.parquet"
)
oab = oab.loc[~oab["nullified"].astype(bool)]

def choices(value):
    if isinstance(value, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in value
        )
    return str(value) if pd.notna(value) else ""

qtext = questions["texto_normalizado"].astype(str).tolist()
theme_ids = themes["tema_id"].astype(int).tolist()
theme_docs = [
    f"{row.documento_enriquecido} {terms.get(int(row.tema_id), '')}".lower()
    for row in themes.itertuples()
]
oab_text = (
    oab["question"].fillna("").astype(str) + " " + oab["choices"].map(choices)
).tolist()
corpus = qtext + theme_docs + oab_text

word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
word.fit(corpus)
S_lex = (word.transform(qtext) @ word.transform(theme_docs).T).toarray()
char = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2
)
char.fit(corpus)
S_char = (char.transform(qtext) @ char.transform(theme_docs).T).toarray()
S_emb = np.load(BASE / "artefatos" / "emb_questoes.npy") @ np.load(
    BASE / "artefatos" / "emb_temas.npy"
).T
signals = np.stack([S_lex, S_char, S_emb], axis=-1)

parent = themes.set_index("tema_id")["parent_id"].to_dict()
def ancestors(theme_id):
    result, current, seen = [], int(theme_id), set()
    while current not in seen and pd.notna(parent.get(current)) and str(parent.get(current)).strip():
        seen.add(current)
        current = int(parent[current])
        result.append(current)
    return result

theme_ix = {value: index for index, value in enumerate(theme_ids)}
R = np.zeros((len(theme_ids), len(theme_ids)))
for a, a_id in enumerate(theme_ids):
    for b, b_id in enumerate(theme_ids):
        if a_id == b_id or b_id in ancestors(a_id) or a_id in ancestors(b_id):
            R[a, b] = 1
R = np.maximum(R, np.eye(len(theme_ids)))

truth_map = {}
for row in truth.itertuples():
    labels = set()
    if str(row.tema_central_id).strip():
        labels.add(int(float(row.tema_central_id)))
    if str(row.extras_ids).strip():
        labels.update(int(x) for x in str(row.extras_ids).split(";") if x.strip())
    truth_map[row.id] = (labels, bool(int(row.fora_da_taxonomia or 0)))
q_ix = {value: index for index, value in enumerate(questions["id"])}
labeled = np.array([q_ix[value] for value in truth_map if value in q_ix], dtype=int)
Y = np.zeros((len(labeled), len(theme_ids)))
fora = np.zeros(len(labeled), dtype=bool)
for position, question_index in enumerate(labeled):
    labels, is_fora = truth_map[questions.iloc[question_index]["id"]]
    fora[position] = is_fora
    for label in labels:
        if label in theme_ix:
            Y[position, theme_ix[label]] = 1
Y_accept = (Y @ R.T > 0).astype(float)

def fit_weights(X, target):
    def objective(weights):
        probability = softmax(X @ weights, axis=1)
        return -np.log((probability * target).sum(axis=1) + 1e-9).mean() + 1e-5 * (weights ** 2).sum()
    result = minimize(objective, np.full(X.shape[-1], 3.0), method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x

def correct(prediction, labels, is_fora):
    return not is_fora and bool(labels) and any(
        R[prediction, theme_ix[label]] > 0 for label in labels if label in theme_ix
    )

def run_oof():
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        for test_pos in np.array_split(rng.permutation(len(labeled)), 10):
            test_pos = np.array(test_pos)
            train_pos = np.array([x for x in range(len(labeled)) if x not in set(test_pos)])
            train_pos = train_pos[~fora[train_pos]]
            weights = fit_weights(signals[labeled[train_pos]], Y_accept[train_pos])
            accepted = softmax(signals[labeled[test_pos]] @ weights, axis=1) @ R.T
            for local, position in enumerate(test_pos):
                prediction = int(accepted[local].argmax())
                qid = questions.iloc[labeled[position]]["id"]
                labels, is_fora = truth_map[qid]
                rows.append({
                    "question_index": labeled[position],
                    "confidence": float(accepted[local, prediction]),
                    "correct": correct(prediction, labels, is_fora),
                })
    return pd.DataFrame(rows)

oof = run_oof()
ordered = oof.sort_values("confidence", ascending=False).reset_index(drop=True)
cumulative = ordered["correct"].astype(float).cumsum() / np.arange(1, len(ordered) + 1)
valid = np.flatnonzero(cumulative.to_numpy() >= 0.96)
threshold = float(ordered.loc[int(valid[-1]), "confidence"]) if len(valid) else 1.01
selected = oof["confidence"] >= threshold
metrics = pd.DataFrame([{
    "threshold": threshold,
    "precision_oof": float(oof.loc[selected, "correct"].mean()),
    "coverage_oof": float(selected.mean()),
    "target_precision": 0.96,
}])
weights = fit_weights(signals[labeled[~fora]], Y_accept[~fora])
accepted = softmax(signals @ weights, axis=1) @ R.T
best = accepted.argmax(axis=1)
ranked = np.sort(accepted, axis=1)
decisions = np.where(accepted[np.arange(len(questions)), best] >= threshold, "publicar", "revisar")
militar = questions["disciplina_origem"].str.contains("Militar", case=False, na=False)
decisions[militar.to_numpy()] = "fora"
full = pd.DataFrame({
    "id": questions["id"],
    "decisao": decisions,
    "tema_central_id": [theme_ids[x] for x in best],
    "confianca": accepted[np.arange(len(questions)), best],
    "margem_top1_top2": ranked[:, -1] - ranked[:, -2],
})
full.loc[militar, "tema_central_id"] = pd.NA
full.to_csv(OUT / "classificacao_completa.csv", index=False)
oof.to_csv(OUT / "resultados_oof.csv", index=False)
metrics.to_csv(OUT / "metricas_modelo.csv", index=False)

publicacao = full.copy()
publicacao["extras_ids"] = ""
publicacao["motivo"] = np.where(
    publicacao["decisao"].eq("publicar"),
    "",
    np.where(publicacao["decisao"].eq("fora"), "sem tema aplicável na taxonomia", "abaixo do limiar OOF ou ambígua"),
)
publicacao["tema_central_id"] = pd.to_numeric(publicacao["tema_central_id"], errors="coerce").astype("Int64")
publicacao["confianca"] = publicacao["confianca"].clip(0, 1).round(6)
publicacao = publicacao[["id", "decisao", "tema_central_id", "extras_ids", "confianca", "motivo"]]
publicacao.to_csv(BASE / "publicacao.csv", index=False)

outside = publicacao[~publicacao["id"].isin(truth["id"])]
coverage = float((outside["decisao"] == "publicar").mean())
precision = float(metrics.iloc[0]["precision_oof"])
(BASE / "previsao.txt").write_text(
    f"""precisao_prevista: {precision:.2f}
precisao_intervalo: {max(0, precision - .03):.2f}-{min(1, precision + .03):.2f}
cobertura_prevista: {coverage:.2f}
cobertura_intervalo: {max(0, coverage - .08):.2f}-{min(1, coverage + .08):.2f}

Estimativa baseada em validação cruzada OOF repetida nos rótulos da folha-verdade.
O threshold foi escolhido no maior prefixo de confiança com precisão OOF de pelo menos 96%.
O corpus externo eduagarcia/oab_exams foi usado somente para ajustar o vocabulário e o IDF do TF-IDF.
Não foram usados gabaritos, respostas ou rótulos da OAB. A classificação final roda offline, sem LLM.
""",
    encoding="utf-8",
)
print(metrics.to_string(index=False))
print(publicacao["decisao"].value_counts().to_string())
