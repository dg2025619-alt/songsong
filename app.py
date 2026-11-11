"""제로음료 효과 '판정(분류)' — 모델 3종 + 베이스라인 시각 비교
분류 대상: 지표 × 비교 칸마다  개선 / 불분명 / 악화  (모든 지표는 '낮을수록 개선'으로 가정)
실행: pip install -r requirements.txt  →  streamlit run app.py
"""
import io
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import optimize, stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).resolve().parent
METRIC_KO = {"Body Weight": "체중", "BMI": "BMI", "Body Fat": "체지방률", "Waist Circumference": "허리둘레",
             "HbA1c": "당화혈색소", "Fasting Blood Glucose": "공복혈당", "2-Hour Post-Prandial Glucose": "식후 2시간 혈당"}
# edge: (Treatment1, Treatment2, 표시이름). 행의 MD = Treatment2 − Treatment1 (제로음료 비교는 MD<0 이면 제로음료가 더 낮음)
EDGES = {"A": ("SSB", "LNCSB", "제로음료 vs 설탕음료"), "B": ("Water", "LNCSB", "제로음료 vs 물"),
         "C": ("SSB", "Water", "물 vs 설탕음료")}
PAPER_EDGE = {"LNCSBs for SSBs": "A", "LNCSBs for Water": "B"}
MODELS = ["베이스라인(고정효과)", "① 랜덤효과+HK", "② 다수준+RVE", "③ 네트워크", "논문 보고값"]
MCOLOR = ["#7F7F7F", "#4C78A8", "#F58518", "#54A24B", "#B279A2"]
NAIVE = "다수결(기준선=찍는 것)"  # 항상 학습 데이터의 가장 흔한 클래스를 예측하는 무지성 분류기
CLS, CODE = ["개선", "불분명", "악화"], {"개선": -1, "불분명": 0, "악화": 1}
CCOLOR = {"개선": "#8FD18F", "불분명": "#E3E3E3", "악화": "#F4A6A6"}
X_EDGE = {"A": [1, 0], "B": [1, -1], "C": [0, 1]}          # 기본 파라미터: d_L(제로−설탕), d_W(물−설탕)
CONTRAST = {"A": np.array([1, 0]), "B": np.array([1, -1]), "C": np.array([0, 1])}  # A: 제로−설탕, B: 제로−물


# ───────────────────────── 데이터 ─────────────────────────
def find_excel():
    """저장소 안의 엑셀 파일을 이름과 무관하게 찾음 (파일명 변형·macOS 한글 자모 분리(NFD) 대비)"""
    for files in (sorted(APP_DIR.glob("*.xlsx")), sorted(APP_DIR.rglob("*.xlsx")), sorted(Path.cwd().glob("*.xlsx"))):
        files = [f for f in files if not f.name.startswith("~$") and ".venv" not in f.parts]
        pref = [f for f in files if "제로음료" in unicodedata.normalize("NFC", f.name)]
        if pref or files:
            return (pref or files)[0]
    return None


@st.cache_data
def load(raw: bytes):
    xl = pd.ExcelFile(io.BytesIO(raw))
    d = xl.parse("개별연구_원자료").dropna(subset=["지표"])
    d = d.rename(columns={"연구(저자·연도)": "study", "Mean Difference": "y", "Standard Error": "se",
                          "Treatment 1": "t1", "Treatment 2": "t2"})
    d["metric"] = d["지표"].map(METRIC_KO)
    d["edge"] = [next(k for k, (a, b, _) in EDGES.items() if (a, b) == (x, z)) for x, z in zip(d.t1, d.t2)]
    d["paper"] = d["study"].str.replace(r"\s*\(.*\)", "", regex=True).str.strip()  # 같은 논문 = 같은 군집
    s = xl.parse("종합분석_결과").dropna(subset=["지표"])
    s["metric"], s["edge"] = s["지표"].map(METRIC_KO), s["비교"].map(PAPER_EDGE)
    return d[["metric", "단위", "study", "paper", "edge", "y", "se"]], s.dropna(subset=["edge"])


# ───────────────────────── 모델 ─────────────────────────
def res(mu, se, df, **kw):
    c = 1.96 if not np.isfinite(df) else stats.t.ppf(0.975, df)
    return dict(mu=mu, se=se, df=df, lo=mu - c * se, hi=mu + c * se, **kw)


def reml_tau2(y, se):
    def nll(t):
        v = se**2 + t
        w = 1 / v
        mu = (w * y).sum() / w.sum()
        return 0.5 * (np.log(v).sum() + np.log(w.sum()) + (w * (y - mu) ** 2).sum())
    return optimize.minimize_scalar(nll, bounds=(0, 10 * np.var(y) + 1), method="bounded").x


def fe(y, se):  # 베이스라인: 역분산 가중 고정효과
    w = 1 / se**2
    return res((w * y).sum() / w.sum(), w.sum() ** -0.5, np.inf)


def re_hk(y, se):  # ① 랜덤효과(REML) + Hartung-Knapp + 예측구간
    k = len(y)
    if k < 2:
        return res(y[0], se[0], np.inf, tau2=0.0)
    t2 = reml_tau2(y, se)
    w = 1 / (se**2 + t2)
    mu = (w * y).sum() / w.sum()
    v = max(1.0, (w * (y - mu) ** 2).sum() / (k - 1)) / w.sum()
    r = res(mu, np.sqrt(v), k - 1, tau2=t2)
    if k >= 3:
        c = stats.t.ppf(0.975, k - 2) * np.sqrt(t2 + v)
        r["pi"] = (mu - c, mu + c)
    return r


def three_level(y, se, cl):  # ② 3수준(효과크기 ⊂ 논문) REML + 군집 강건분산(CR1)
    ids = pd.factorize(cl)[0]
    G = ids.max() + 1
    if G == len(y):  # 한 논문에 한 행뿐이면 랜덤효과와 같음
        r = re_hk(y, se)
        return {**r, "G": G, "tw": 0.0, "tb": r.get("tau2", 0.0)}
    blocks = [np.where(ids == g)[0] for g in range(G)]

    def parts(th):
        tw, tb = np.exp(np.clip(th, -12, 4))  # 분산 성분 범위 제한(수치 안정)
        out = []
        for b in blocks:
            S = np.diag(se[b] ** 2 + tw) + tb
            out.append((b, np.linalg.inv(S), np.linalg.slogdet(S)[1]))
        A = sum(np.ones(len(b)) @ W @ np.ones(len(b)) for b, W, _ in out)
        mu = sum(np.ones(len(b)) @ W @ y[b] for b, W, _ in out) / A
        return out, A, mu

    def nll(th):
        out, A, mu = parts(th)
        return 0.5 * (sum(l for *_, l in out) + np.log(A) + sum((y[b] - mu) @ W @ (y[b] - mu) for b, W, _ in out))

    th = np.clip(optimize.minimize(nll, np.log([0.01, 0.01]), method="Nelder-Mead").x, -12, 4)
    out, A, mu = parts(th)
    u = np.array([np.ones(len(b)) @ W @ (y[b] - mu) for b, W, _ in out])
    var = (u**2).sum() / A**2 * G / (G - 1) if G > 1 else 1 / A
    return res(mu, np.sqrt(var), G - 1 if G > 1 else np.inf, G=G, tw=np.exp(th[0]), tb=np.exp(th[1]))


def nma(g, edge):  # ③ 네트워크 메타분석(빈도주의, 랜덤효과): 세 비교를 한 모델로 동시에 추정
    X = np.array([X_EDGE[e] for e in g.edge], float)
    y, se, c = g.y.values, g.se.values, CONTRAST[edge]
    rk = np.linalg.matrix_rank
    if rk(np.vstack([X, c])) > rk(X):  # 이 대비를 추정할 연결이 없음
        return None
    w0 = 1 / se**2
    M = np.linalg.pinv((X.T * w0) @ X)
    b0 = M @ ((X.T * w0) @ y)
    dfq = len(y) - rk(X)
    C = w0.sum() - np.trace(M @ ((X.T * w0**2) @ X))
    t2 = max(0.0, ((w0 * (y - X @ b0) ** 2).sum() - dfq) / C) if dfq > 0 and C > 0 else 0.0
    w = 1 / (se**2 + t2)
    M = np.linalg.pinv((X.T * w) @ X)
    b = M @ ((X.T * w) @ y)
    return res(c @ b, np.sqrt(c @ M @ c), dfq if dfq > 0 else np.inf, tau2=t2)


def consistency(gm):
    """직접 근거 vs 간접 근거(다른 두 비교로 계산) — 크게 다르면 네트워크가 불일치"""
    pool = {e: re_hk(gm[gm.edge == e].y.values, gm[gm.edge == e].se.values) for e in "ABC" if (gm.edge == e).any()}
    rows = []
    for tgt, (o1, o2, sgn) in {"A": ("B", "C", 1), "B": ("A", "C", -1)}.items():
        if tgt in pool and o1 in pool and o2 in pool:
            ind = pool[o1]["mu"] + sgn * pool[o2]["mu"]
            ise = np.hypot(pool[o1]["se"], pool[o2]["se"])
            z = (pool[tgt]["mu"] - ind) / np.hypot(pool[tgt]["se"], ise)
            rows.append(dict(비교=EDGES[tgt][2], 직접=round(pool[tgt]["mu"], 2), 간접=round(ind, 2),
                             z=round(z, 2), 판정="❌ 불일치" if abs(z) > 1.96 else "✅"))
    return rows


# ───────────────────────── 분류(결정 규칙) ─────────────────────────
def classify(r, d=0.0):
    if r is None:
        return None
    return "개선" if r["hi"] < -d else "악화" if r["lo"] > d else "불분명"


def probs(r, d=0.0):
    cdf = (lambda x: stats.norm.cdf(x)) if not np.isfinite(r["df"]) else (lambda x: stats.t.cdf(x, r["df"]))
    pb, pw = cdf((-d - r["mu"]) / r["se"]), 1 - cdf((d - r["mu"]) / r["se"])
    return pb, max(0.0, 1 - pb - pw), pw


def drop_outliers(d, thr):
    drop = set()
    for _, idx in d.groupby(["metric", "edge"]).groups.items():
        g = d.loc[idx]
        if len(g) < 3:
            continue
        for i in g.index:
            o = g.drop(i)
            t2 = reml_tau2(o.y.values, o.se.values)
            w = 1 / (o.se.values**2 + t2)
            mu, sm = (w * o.y.values).sum() / w.sum(), w.sum() ** -0.5
            if abs((g.loc[i, "y"] - mu) / np.sqrt(g.loc[i, "se"] ** 2 + t2 + sm**2)) >= thr:
                drop.add(i)
    return d[~d.index.isin(drop)].reset_index(drop=True)


# ───────────────────────── 분류 성능 평가 (leave-one-paper-out) ─────────────────────────
# 과제: 개별 연구 결과 1건이 개선/불분명/악화 중 무엇으로 관측될지 예측
#  - 정답: 그 연구 자신의 95% CI (개선=CI가 0보다 아래, 악화=0보다 위, 그 외=불분명)
#  - 예측: 같은 논문의 모든 행을 뺀 나머지 자료로 모델을 적합 → 예측분포 → 클래스 확률
#  → 정답을 만든 연구가 학습에 들어가지 않으므로 순환(데이터 누수)이 없음
def obs_class(y, se):
    return "개선" if y + 1.96 * se < 0 else "악화" if y - 1.96 * se > 0 else "불분명"


def _predict(model, tr_edge, tr_all, edge):
    """(예측 평균, 관측오차를 제외한 예측 분산) 또는 None"""
    try:
        if model == MODELS[3]:
            r = nma(tr_all, edge)
            return None if r is None else (r["mu"], r.get("tau2", 0.0) + r["se"] ** 2)
        if len(tr_edge) < 2:
            return None
        y, se = tr_edge.y.values, tr_edge.se.values
        if model == MODELS[0]:
            r = fe(y, se)
            return r["mu"], r["se"] ** 2
        if model == MODELS[1]:
            r = re_hk(y, se)
            return r["mu"], r.get("tau2", 0.0) + r["se"] ** 2
        r = three_level(y, se, tr_edge.paper.values)
        return r["mu"], r.get("tw", 0.0) + r.get("tb", 0.0) + r["se"] ** 2
    except Exception:
        return None


DIRS = ["낮음(−)", "높음(+)"]
TASKS = {"유의성 3클래스 (개선/불분명/악화)": CLS, "방향 2클래스 (제로음료 쪽이 낮음/높음)": DIRS}


@st.cache_data
def loo_eval(d):
    """행마다 모델별 예측분포(평균 m, 분산 v = 모델 불확실성 + 그 연구의 관측오차)를 저장"""
    rows = []
    for m, gm in d.groupby("metric"):
        for p in gm.paper.unique():
            te, tr = gm[gm.paper == p], gm[gm.paper != p]
            freq = pd.Series([obs_class(a, b) for a, b in zip(tr.y, tr.se)]).value_counts()
            nv = np.array([freq.get(k, 0) + 1 for k in CLS], float)
            nv /= nv.sum()  # 라플라스 평활
            p_low = ((tr.y < 0).sum() + 1) / (len(tr) + 2)
            for e in ("A", "B"):
                t = te[te.edge == e]
                if t.empty:
                    continue
                pred = {mod: _predict(mod, tr[tr.edge == e], tr, e) for mod in MODELS[:4]}
                for _, r in t.iterrows():
                    rec = dict(metric=m, edge=e, study=r.study, se=r.se, true_sig=obs_class(r.y, r.se),
                               true_dir=DIRS[0] if r.y < 0 else DIRS[1], ns_low=p_low)
                    for mod, pr in pred.items():
                        rec[f"{mod}|m"], rec[f"{mod}|v"] = (pr[0], pr[1] + r.se**2) if pr else (np.nan, np.nan)
                    for k, v in zip(CLS, nv):
                        rec[f"ns|{k}"] = v
                    rows.append(rec)
    return pd.DataFrame(rows)


def class_probs(ev, name, task):
    """모델/기준선이 각 행에 주는 클래스 확률 (행 × 클래스)"""
    if name == NAIVE:
        return ev[[f"ns|{k}" for k in CLS]].values if task == "sig" else np.c_[ev["ns_low"], 1 - ev["ns_low"]]
    m, sd, se = ev[f"{name}|m"].values, np.sqrt(ev[f"{name}|v"].values), ev["se"].values
    if task == "dir":
        pl = stats.norm.cdf(-m / sd)
        return np.c_[pl, 1 - pl]
    pb, pw = stats.norm.cdf((-1.96 * se - m) / sd), 1 - stats.norm.cdf((1.96 * se - m) / sd)
    return np.c_[pb, np.clip(1 - pb - pw, 0, 1), pw]


def score_models(ev, names, task):
    """공통 행(모든 모델이 예측 가능)에서 정확도·macro-F1·log-loss·Brier 계산"""
    classes, tcol = (CLS, "true_sig") if task == "sig" else (DIRS, "true_dir")
    ok = ev.dropna(subset=[f"{n}|m" for n in names if n != NAIVE]).reset_index(drop=True)
    Y = np.array([[t == k for k in classes] for t in ok[tcol]], float)
    out, cms = [], {}
    for n in names:
        P = class_probs(ok, n, task)
        pred = np.array(classes)[P.argmax(1)]
        cm = pd.crosstab(pd.Categorical(ok[tcol], classes), pd.Categorical(pred, classes), dropna=False)
        f1s = []
        for k in classes:
            if cm.loc[k].sum() > 0:
                tp, fp, fn = cm.loc[k, k], cm[k].sum() - cm.loc[k, k], cm.loc[k].sum() - cm.loc[k, k]
                f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        out.append({"모델": n, "정확도": (pred == ok[tcol].values).mean(), "macro-F1": np.mean(f1s),
                    "log-loss": -(Y * np.log(np.clip(P, 1e-6, 1))).sum(1).mean(), "Brier": ((P - Y) ** 2).sum(1).mean()})
        cms[n] = cm
    return pd.DataFrame(out), cms, ok, classes, tcol


# ───────────────────────── 분류 모델 3종 (지도학습) ─────────────────────────
# 입력(특징): 지표, 비교, 하위군(감미료·성별 등), 발표연도, 지표 안에서 표준화한 SE 크기  ← 효과크기(MD)는 넣지 않음(정답 누수 방지)
# 출력(클래스): 방향(낮음/높음) 또는 유의성(개선/불분명/악화)
# 평가: leave-one-paper-out (같은 논문 행은 통째로 검증용으로 빼서 예측)
CLF_NAMES = ["로지스틱 회귀", "랜덤 포레스트", "그래디언트 부스팅"]
CF_COLOR = ["#4C78A8", "#54A24B", "#E45756", "#BBBBBB"]
CAT, NUM = ["metric", "edge", "sub"], ["year", "log_se_z"]


def make_features(d):
    x = d[d.edge.isin(["A", "B"])].reset_index(drop=True).copy()  # 제로음료가 들어간 비교만
    x["year"] = pd.to_numeric(x.study.str.extract(r"(\d{4})")[0], errors="coerce")
    x["year"] = x["year"].fillna(x["year"].median())
    sub = x.study.str.extract(r"\((.*?)\)")[0].fillna("없음")
    x["sub"] = sub.where(sub.map(sub.value_counts()) >= 3, "기타")
    ls = np.log(x.se)
    x["log_se_z"] = ls.groupby(x.metric).transform(lambda v: (v - v.mean()) / (v.std(ddof=0) or 1.0))
    x["true_sig"] = [obs_class(a, b) for a, b in zip(x.y, x.se)]
    x["true_dir"] = np.where(x.y < 0, DIRS[0], DIRS[1])
    return x


def clf_pipe(name):
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT), ("n", StandardScaler(), NUM)])
    est = {CLF_NAMES[0]: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000),
           CLF_NAMES[1]: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", random_state=0),
           CLF_NAMES[2]: GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
           NAIVE: DummyClassifier(strategy="prior")}[name]
    return Pipeline([("pre", pre), ("clf", est)])


def _clean(n):
    for a, b in [("c__metric_", "지표="), ("c__edge_A", "비교=제로 vs 설탕"), ("c__edge_B", "비교=제로 vs 물"), ("c__sub_", "하위군="),
                 ("n__year", "발표연도"), ("n__log_se_z", "SE 크기(정밀도)")]:
        n = n.replace(a, b)
    return n


@st.cache_data
def run_classifiers(x, task):
    classes, tcol = (CLS, "true_sig") if task == "sig" else (DIRS, "true_dir")
    y, K, names = np.array([classes.index(v) for v in x[tcol]]), len(classes), CLF_NAMES + [NAIVE]
    F, oof = x[CAT + NUM], {n: np.zeros((len(x), len(classes))) for n in names}
    for tr, te in LeaveOneGroupOut().split(F, y, groups=x.paper):
        for n in names:
            if len(np.unique(y[tr])) < 2:  # 학습에 클래스가 하나뿐이면 사전확률로 대체
                pr = np.bincount(y[tr], minlength=K) + 1.0
                oof[n][te] = pr / pr.sum()
                continue
            m = clf_pipe(n).fit(F.iloc[tr], y[tr])
            full = np.zeros((len(te), K))
            full[:, m.classes_] = m.predict_proba(F.iloc[te])
            oof[n][te] = 0.98 * full + 0.02 / K  # 0확률 방지용 약한 평활(모든 모델 동일)
    Y, rows, cms = np.eye(K)[y], [], {}
    for n, P in oof.items():
        pred = P.argmax(1)
        try:
            auc = roc_auc_score(y == 0, P[:, 0]) if K == 2 else roc_auc_score(Y, P, average="macro")
        except ValueError:
            auc = np.nan
        rows.append({"모델": n, "정확도": (pred == y).mean(), "균형정확도": balanced_accuracy_score(y, pred),
                     "macro-F1": f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0), "AUC": auc,
                     "log-loss": log_loss(y, P, labels=list(range(K))), "Brier": ((P - Y) ** 2).sum(1).mean()})
        cms[n] = confusion_matrix(y, pred, labels=list(range(K)))
    fitted = {n: clf_pipe(n).fit(F, y) for n in CLF_NAMES}  # 전체 자료로 적합(중요도·직접 예측용)
    imps = {}
    for n, m in fitted.items():
        est, names_f = m.named_steps["clf"], [_clean(f) for f in m.named_steps["pre"].get_feature_names_out()]
        v = np.abs(est.coef_).mean(0) if n == CLF_NAMES[0] else est.feature_importances_
        imps[n] = pd.Series(v / v.sum(), index=names_f).sort_values(ascending=False)
    return dict(scores=pd.DataFrame(rows), cms=cms, oof=oof, y=y, classes=classes, tcol=tcol, fitted=fitted, imps=imps)


@st.cache_data
def run(d, s):
    cells = []
    for m in [x for x in METRIC_KO.values() if x in set(d.metric)]:
        gm = d[d.metric == m]
        for e in "AB":
            g, r = gm[gm.edge == e], {m: None for m in MODELS}  # 추정 불가인 모델은 None
            if len(g):
                y, se = g.y.values, g.se.values
                r[MODELS[0]], r[MODELS[1]], r[MODELS[2]] = fe(y, se), re_hk(y, se), three_level(y, se, g.paper.values)
            r[MODELS[3]] = nma(gm, e)
            p = s[(s.metric == m) & (s.edge == e)]
            r[MODELS[4]] = None if p.empty else dict(
                mu=p.iloc[0]["종합 평균차이(MD)"], lo=p.iloc[0]["95% CI 하한"], hi=p.iloc[0]["95% CI 상한"],
                se=(p.iloc[0]["95% CI 상한"] - p.iloc[0]["95% CI 하한"]) / 3.92, df=np.inf)
            cells.append(dict(metric=m, edge=e, comp=EDGES[e][2], unit=gm["단위"].iloc[0], k=len(g), res=r))
    return cells


# ───────────────────────── 화면 ─────────────────────────
st.set_page_config(page_title="제로음료 판정 모델 비교", layout="wide")
st.title("제로음료, 몸에 좋은가? — 모델 3종 판정 비교")
st.caption("각 (지표 × 비교) 칸을 **개선 / 불분명 / 악화** 3개 클래스로 분류하고, 모델별 판정이 어디서 갈리는지 눈으로 확인합니다. "
           "클래스는 정답 라벨이 아니라 모델 추정치(95% CI)로 만든 결정 규칙입니다.")
st.warning("부호 가정: MD<0 이면 제로음료 쪽이 더 낮음(모든 지표 '낮을수록 개선'). 논문 합산값과 방향이 맞는 것은 확인했지만 원문 PDF로 재확인하세요. "
           "②는 같은 논문 행 사이 표본 공분산을, ③은 다중군 연구의 행 간 상관을 무시한 근사입니다.", icon="⚠️")

path = find_excel()
if path is not None:
    raw = path.read_bytes()
else:
    st.error("엑셀 파일을 저장소에서 찾지 못했습니다. 아래에 직접 업로드하거나, 엑셀을 app.py와 같은 폴더에 올린 뒤 다시 배포하세요.")
    up = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type="xlsx")
    if up is None:
        st.stop()
    raw = up.getvalue()
d_all, s_all = load(raw)
with st.sidebar:
    st.header("설정")
    sel = st.multiselect("분석할 지표", list(METRIC_KO.values()), default=list(METRIC_KO.values()))
    ex = st.checkbox("이상치 제외 (leave-one-out)", value=False)
    thr = st.slider("이상치 기준 |z|", 2.0, 5.0, 3.0, 0.5, disabled=not ex)
    with st.expander("임상적 의미 기준 δ (지표별)"):
        st.caption("0이면 '신뢰구간이 0을 지나는가'만 봅니다. δ>0이면 CI 전체가 ±δ 밖이어야 개선/악화로 분류합니다.")
        delta = {m: st.number_input(m, min_value=0.0, value=0.0, step=0.1, key=f"d_{m}") for m in sel}

if not sel:
    st.info("지표를 하나 이상 선택하세요.")
    st.stop()
d = d_all[d_all.metric.isin(sel)].reset_index(drop=True)
if ex:
    n0 = len(d)
    d = drop_outliers(d, thr)
    st.info(f"이상치 {n0 - len(d)}건 제외 후 분석")
cells = run(d, s_all)
for c in cells:
    c["cls"] = {k: classify(v, delta[c["metric"]]) for k, v in c["res"].items()}

task_name = st.radio(
    "분류 과제", list(TASKS), index=1, horizontal=True,
    help="방향: 연구 결과가 0보다 낮은지 높은지(기본). 유의성: 연구 하나의 CI가 0을 넘는지(대부분 '불분명'이라 클래스가 쏠림).")
task = "sig" if task_name.startswith("유의성") else "dir"

# ── 분류 모델 3종 학습·평가
X = make_features(d)
cf = run_classifiers(X, task) if len(X) >= 10 and X.paper.nunique() >= 4 else None

if cf is not None:  # 헤드라인
    csc = cf["scores"]
    top = csc[csc["모델"].isin(CLF_NAMES)].sort_values("log-loss").iloc[0]
    base = csc[csc["모델"] == NAIVE].iloc[0]
    with st.container(border=True):
        st.subheader(f"🏁 분류 모델 3종 중 1위: {top['모델']}")
        h1, h2, h3 = st.columns(3)
        h1.metric("AUC", f"{top['AUC']:.2f}", f"{top['AUC'] - base['AUC']:+.2f} vs 기준선")
        h2.metric("균형정확도", f"{top['균형정확도']:.1%}", f"{top['균형정확도'] - base['균형정확도']:+.1%} vs 기준선")
        h3.metric("log-loss (낮을수록 좋음)", f"{top['log-loss']:.3f}", f"{top['log-loss'] - base['log-loss']:+.3f} vs 기준선", delta_color="inverse")
        if top["log-loss"] < base["log-loss"]:
            st.success("1위 모델이 다수결 기준선보다 잘 분류합니다.")
        else:
            st.warning("어느 분류 모델도 다수결 기준선을 넘지 못했습니다. 주어진 특징(지표·비교·하위군·연도·SE)만으로는 연구 결과의 클래스를 신뢰성 있게 예측하기 어렵다는 뜻입니다.")

tc, tm, t1, t2, t3 = st.tabs(["⭐ 분류 모델 3종", "📎 메타분석 모델의 판정 성능", "① 판정표 한눈에", "② 칸별 자세히", "③ 진단(이질성·일치성)"])

with tc:
    if cf is None:
        st.info("분류 모델을 학습하기에 데이터가 부족합니다(논문 4편 이상, 10행 이상 필요).")
    else:
        cl, cms_, oof_, y_ = cf["classes"], cf["cms"], cf["oof"], cf["y"]
        st.subheader("분류 문제 정의")
        st.markdown(
            "| 항목 | 내용 |\n|---|---|\n"
            f"| **분류 대상** | 제로음료가 들어간 비교의 연구 결과 {len(X)}건 (논문 {X.paper.nunique()}편) |\n"
            f"| **클래스** | {' / '.join(cl)} " + ("(MD의 부호)" if task == "dir" else "(그 연구 자신의 95% CI, 0 기준)") + " |\n"
            "| **입력 특징** | 지표 · 비교(제로 vs 설탕/물) · 하위군(감미료·성별 등) · 발표연도 · 지표 안에서 표준화한 SE 크기 |\n"
            "| **입력에서 뺀 것** | 효과크기(MD) — 넣으면 정답이 그대로 들어가는 누수 |\n"
            "| **평가** | leave-one-paper-out: 같은 논문의 행을 통째로 빼고 예측 |\n"
            f"| **기준선** | {NAIVE}: 학습 데이터에서 가장 흔한 클래스만 예측 |\n"
            "| **분류 모델 3종** | ① 로지스틱 회귀(해석 쉬움) · ② 랜덤 포레스트(비선형·상호작용) · ③ 그래디언트 부스팅(순차 보정) |")
        st.caption(f"클래스 분포: " + " / ".join(f"{k} {int((y_ == i).sum())}" for i, k in enumerate(cl)))

        st.subheader("성능 비교")
        st.dataframe(cf["scores"].set_index("모델").style.format(
            {"정확도": "{:.1%}", "균형정확도": "{:.1%}", "macro-F1": "{:.2f}", "AUC": "{:.2f}", "log-loss": "{:.3f}", "Brier": "{:.3f}"})
            .highlight_max(subset=["정확도", "균형정확도", "macro-F1", "AUC"], color="#cde8cd")
            .highlight_min(subset=["log-loss", "Brier"], color="#cde8cd"), width="stretch")
        st.caption("정확도·균형정확도·macro-F1·AUC는 높을수록, log-loss·Brier는 낮을수록 좋습니다. AUC 0.5는 무작위 수준입니다. "
                   "기준선의 AUC가 0.5보다 낮게 나오는 것은 leave-one-paper-out에서 학습 분포가 검증 분포와 반대로 치우치기 때문에 생기는 알려진 현상이니 무시하세요. "
                   f"행 {len(X)}건·논문 {X.paper.nunique()}편의 작은 표본이라 점수 차이가 작으면 우연일 수 있습니다.")
        g1, g2 = st.columns(2)
        with g1:
            if len(cl) == 2:
                f = go.Figure()
                for n, col in zip(CLF_NAMES + [NAIVE], CF_COLOR):
                    fpr, tpr, _ = roc_curve(y_ == 0, oof_[n][:, 0])
                    f.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=n, line=dict(color=col, width=3 if n != NAIVE else 1)))
                f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="gray"), showlegend=False))
                f.update_layout(title=f"ROC 곡선 (양성 = {cl[0]})", height=360, xaxis_title="거짓양성률", yaxis_title="참양성률",
                                margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25))
                st.plotly_chart(f, width="stretch")
            else:
                st.info("ROC 곡선은 2클래스 과제에서만 표시합니다.")
        with g2:
            sc_ = cf["scores"]
            f = go.Figure(go.Bar(x=sc_["모델"], y=sc_["균형정확도"], marker_color=CF_COLOR, text=[f"{v:.0%}" for v in sc_["균형정확도"]],
                                 textposition="outside"))
            f.update_layout(title="균형정확도", height=360, yaxis=dict(range=[0, 1], tickformat=".0%"),
                            margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-15)
            st.plotly_chart(f, width="stretch")

        st.subheader("혼동행렬 (행=실제, 열=예측)")
        for col, n in zip(st.columns(4), CLF_NAMES + [NAIVE]):
            f = go.Figure(go.Heatmap(z=cms_[n], x=cl, y=cl, text=cms_[n], texttemplate="%{text}", showscale=False, colorscale="Blues", xgap=2, ygap=2))
            f.update_yaxes(autorange="reversed")
            f.update_layout(title=dict(text=n, font=dict(size=13)), height=300, margin=dict(l=10, r=10, t=50, b=10))
            col.plotly_chart(f, width="stretch")

        st.subheader("모델이 무엇을 보고 판단했나 (특징 중요도)")
        for col, n, c in zip(st.columns(3), CLF_NAMES, CF_COLOR):
            im = cf["imps"][n].head(8)[::-1]
            f = go.Figure(go.Bar(x=im.values, y=im.index, orientation="h", marker_color=c))
            f.update_layout(title=dict(text=n, font=dict(size=13)), height=330, margin=dict(l=10, r=10, t=50, b=10), xaxis_tickformat=".0%")
            col.plotly_chart(f, width="stretch")
        st.caption("로지스틱 회귀는 계수 절댓값, 나머지는 모델 내장 중요도이며 각각 합이 100%가 되게 맞췄습니다. 전체 자료로 다시 적합한 모델 기준입니다.")

        st.subheader(f"지표 × 비교별 '{cl[0]}' 예측 확률 (검증 예측의 평균)")
        grp = X.assign(**{n: oof_[n][:, 0] for n in CLF_NAMES}, 실제=(y_ == 0).astype(float))
        grp["칸"] = grp.metric + " · " + grp.edge.map({"A": EDGES["A"][2], "B": EDGES["B"][2]})
        tbl = grp.groupby("칸")[["실제"] + CLF_NAMES].mean().reindex(
            [f"{m} · {EDGES[e][2]}" for m in METRIC_KO.values() for e in "AB" if f"{m} · {EDGES[e][2]}" in set(grp["칸"])])
        f = go.Figure(go.Heatmap(z=tbl.values, x=[f"실제 {cl[0]} 비율"] + CLF_NAMES, y=tbl.index, text=[[f"{v:.0%}" for v in r] for r in tbl.values],
                                 texttemplate="%{text}", zmin=0, zmax=1, colorscale="Blues", showscale=False, xgap=3, ygap=3))
        f.update_yaxes(autorange="reversed")
        f.update_xaxes(side="top")
        f.update_layout(height=max(380, 36 * len(tbl) + 90), margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(f, width="stretch")
        st.caption("각 칸에서 모델들이 '" + cl[0] + "' 쪽이라고 본 평균 확률과 실제 비율을 비교합니다. 둘이 비슷하면 모델이 그 칸의 경향을 잡은 것입니다.")

        with st.expander("직접 예측해 보기"):
            e1, e2, e3, e4 = st.columns(4)
            q_m = e1.selectbox("지표", [m for m in METRIC_KO.values() if m in set(X.metric)], key="q_m")
            q_e = e2.radio("비교", ["A", "B"], format_func=lambda e: EDGES[e][2], key="q_e")
            q_s = e3.selectbox("하위군", sorted(X["sub"].unique()), key="q_s")
            q_y = e4.slider("발표연도", int(X.year.min()), int(X.year.max()), int(X.year.median()), key="q_y")
            q_z = st.slider("SE 크기(지표 안 표준화, +면 부정확한 연구)", -2.0, 2.0, 0.0, 0.1, key="q_z")
            row = pd.DataFrame([dict(metric=q_m, edge=q_e, sub=q_s, year=q_y, log_se_z=q_z)])
            for col, n in zip(st.columns(3), CLF_NAMES):
                pr = cf["fitted"][n].predict_proba(row)[0]
                col.metric(n, f"P({cl[0]}) = {pr[0]:.0%}", f"예측: {cl[int(np.argmax(pr))]}", delta_color="off")

with tm:
    st.subheader("메타분석 모델은 연구 결과를 얼마나 잘 분류하나")
    st.caption("앞 탭의 지도학습 분류 모델과 달리, 메타분석 모델(베이스라인·①②③)이 만든 예측분포로 같은 과제를 푼 결과입니다. 정답 연구를 뺀 나머지로 적합합니다(leave-one-paper-out).")
    ev = loo_eval(d)
    NAMES = MODELS[:4] + [NAIVE]
    ev_ok = ev.dropna(subset=[f"{n}|m" for n in MODELS[:4]]) if len(ev) else ev
    sc = pd.DataFrame()
    if len(ev_ok) >= 5:
        sc, cms, ok, classes, tcol = score_models(ev, NAMES, task)

    if len(sc):
        _top = sc[sc["모델"].isin(MODELS[:4])].sort_values("log-loss").iloc[0]
        _nv = sc[sc["모델"] == NAIVE].iloc[0]
        best = _top["모델"]
        _msg = f"{task_name} 기준 1위 {best}: log-loss {_top['log-loss']:.3f} (다수결 기준선 {_nv['log-loss']:.3f}) — "
        (st.success if _top["log-loss"] < _nv["log-loss"] else st.warning)(_msg + ("기준선보다 잘 예측합니다." if _top["log-loss"] < _nv["log-loss"] else "기준선을 넘지 못했습니다."))
    st.subheader("평가 과제")
    if task == "sig":
        st.markdown("**과제**: 연구 한 건이 **개선 / 불분명 / 악화** 중 무엇으로 관측될지 예측합니다. 정답은 그 연구 자신의 95% CI(0 기준)입니다.")
    else:
        st.markdown("**과제**: 연구 한 건의 효과가 0보다 **낮을지(제로음료 쪽이 더 낮음) / 높을지** 예측합니다. 정답은 그 연구 MD의 부호입니다.")
    st.markdown("**예측**: 같은 논문을 통째로 뺀 나머지 자료로 모델을 적합해 클래스 확률을 계산합니다(leave-one-paper-out). "
                "정답을 만든 연구가 학습에 들어가지 않아 순환 평가가 아닙니다.  \n"
                f"**기준선**: '{NAIVE}'은 학습 데이터에서 가장 흔한 클래스만 찍는 분류기로, 모델이 이보다 나아야 의미가 있습니다.")
    if sc.empty:
        st.info("평가할 데이터가 부족합니다.")
    else:
        n_true = ok[tcol].value_counts().reindex(classes, fill_value=0)
        st.caption(f"평가 대상 {len(ok)}건 · 실제 클래스 분포: " + " / ".join(f"{k} {n_true[k]}" for k in classes))
        st.dataframe(sc.set_index("모델").style.format({"정확도": "{:.1%}", "macro-F1": "{:.2f}", "log-loss": "{:.3f}", "Brier": "{:.3f}"})
                     .highlight_max(subset=["정확도", "macro-F1"], color="#cde8cd")
                     .highlight_min(subset=["log-loss", "Brier"], color="#cde8cd"), width="stretch")
        st.caption("정확도·macro-F1은 높을수록, log-loss·Brier는 낮을수록 좋습니다. 클래스가 한쪽으로 쏠리면 정확도만으로는 기준선과 구분이 안 되므로 log-loss, Brier를 함께 보세요.")
        b1, b2 = st.columns(2)
        for col, key, ttl in [(b1, "정확도", "정확도 (높을수록 좋음)"), (b2, "log-loss", "log-loss (낮을수록 좋음)")]:
            f = go.Figure(go.Bar(x=sc["모델"], y=sc[key], marker_color=MCOLOR[:4] + ["#CCCCCC"],
                                 text=[f"{v:.2f}" for v in sc[key]], textposition="outside"))
            f.update_layout(title=ttl, height=340, margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=-20)
            col.plotly_chart(f, width="stretch")
        st.subheader("혼동행렬 (행=실제, 열=예측)")
        for col, n in zip(st.columns(len(NAMES)), NAMES):
            cm = cms[n]
            f = go.Figure(go.Heatmap(z=cm.values, x=classes, y=classes, text=cm.values, texttemplate="%{text}", showscale=False,
                                     colorscale="Blues", xgap=2, ygap=2))
            f.update_yaxes(autorange="reversed")
            f.update_layout(title=dict(text=n, font=dict(size=12)), height=300, margin=dict(l=10, r=10, t=50, b=10))
            col.plotly_chart(f, width="stretch")
        st.subheader("가장 크게 빗나간 연구")
        bp = class_probs(ok, best, task)
        ok2 = ok.assign(예측=np.array(classes)[bp.argmax(1)], 정답확률=[bp[i, classes.index(t)] for i, t in enumerate(ok[tcol])])
        miss = ok2[ok2["예측"] != ok2[tcol]].sort_values("정답확률").head(10)
        st.dataframe(miss[["metric", "study", tcol, "예측", "정답확률"]].rename(columns={"metric": "지표", "study": "연구", tcol: "실제"})
                     .assign(정답확률=lambda x: x["정답확률"].map("{:.1%}".format)), hide_index=True, width="stretch")
        st.caption(f"{best}가 틀린 연구 중 정답 클래스에 가장 낮은 확률을 준 사례입니다. 이상치(예: Peters 2016, Madjd 2017)가 여기에 나타나는지 확인해 보세요.")


with t1:
    ylab = [f"{c['metric']} · {c['comp']}" for c in cells]
    z = [[CODE.get(c["cls"][m], np.nan) if c["cls"][m] else np.nan for m in MODELS] for c in cells]
    txt = [[("—" if c["res"][m] is None else f"<b>{c['res'][m]['mu']:+.2f}</b><br>[{c['res'][m]['lo']:.2f}, {c['res'][m]['hi']:.2f}]")
            for m in MODELS] for c in cells]
    fig = go.Figure(go.Heatmap(z=z, x=MODELS, y=ylab, text=txt, texttemplate="%{text}", zmin=-1, zmax=1, showscale=False,
                               colorscale=[[0, CCOLOR["개선"]], [0.5, CCOLOR["불분명"]], [1, CCOLOR["악화"]]], xgap=3, ygap=3,
                               hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    fig.update_layout(height=max(500, 52 * len(cells) + 90), margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("🟩 개선(CI 전체가 −δ 아래)  ⬜ 불분명  🟥 악화(CI 전체가 +δ 위) · 칸 안 숫자는 MD [95% CI] · '—'은 해당 모델로 추정 불가")

    st.subheader("모델별 판정 개수")
    cnt = [dict(비교=EDGES[e][2], 모델=m, **{k: sum(c["cls"][m] == k for c in cells if c["edge"] == e) for k in CLS})
           for e in "AB" for m in MODELS]
    st.dataframe(pd.DataFrame(cnt), hide_index=True, width="stretch")

    st.subheader("판정이 갈리는 칸")
    dis = []
    for c in cells:
        v = {m: c["cls"][m] for m in MODELS[:4] if c["cls"][m]}
        if len(set(v.values())) > 1:
            dis.append({"지표": c["metric"], "비교": c["comp"], **{m: c["cls"][m] or "—" for m in MODELS}})
    if dis:
        st.dataframe(pd.DataFrame(dis), hide_index=True, width="stretch")
    else:
        st.success("베이스라인과 모델 3종의 판정이 모든 칸에서 같습니다.")

    st.subheader("모델 간 판정 일치율")
    agr = pd.DataFrame([[np.nan if not (pair := [(c["cls"][a], c["cls"][b]) for c in cells if c["cls"][a] and c["cls"][b]])
                         else 100 * np.mean([x == y for x, y in pair]) for b in MODELS] for a in MODELS], index=MODELS, columns=MODELS)
    st.dataframe(agr.round(0).style.format("{:.0f}%", na_rep="—"), width="stretch")

with t2:
    c1, c2 = st.columns(2)
    m_sel = c1.selectbox("지표", [x for x in METRIC_KO.values() if x in set(d.metric)])
    e_sel = c2.radio("비교", ["A", "B"], format_func=lambda e: EDGES[e][2], horizontal=True)
    cell = next(c for c in cells if c["metric"] == m_sel and c["edge"] == e_sel)
    g = d[(d.metric == m_sel) & (d.edge == e_sel)].copy()
    g["lab"] = g.study + g.groupby("study").cumcount().map(lambda i: "" if i == 0 else f" ({i + 1})")
    dl = delta[m_sel]
    fig = go.Figure()
    if len(g):
        fig.add_trace(go.Scatter(x=g.y, y=g.lab, mode="markers", name="개별 연구",
                                 marker=dict(color="#999", symbol="square", size=8),
                                 error_x=dict(type="data", array=(1.96 * g.se).tolist(), color="#bbb")))
    for name, col in zip(MODELS, MCOLOR):
        r = cell["res"][name]
        if r is None:
            continue
        fig.add_trace(go.Scatter(x=[r["mu"]], y=[name], mode="markers", name=name,
                                 marker=dict(symbol="diamond", size=14, color=col),
                                 error_x=dict(type="data", symmetric=False, array=[r["hi"] - r["mu"]],
                                              arrayminus=[r["mu"] - r["lo"]], color=col, thickness=3)))
        if r.get("pi"):
            fig.add_trace(go.Scatter(x=list(r["pi"]), y=[name, name], mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=col, dash="dot", width=2)))
    fig.add_vline(x=0, line_color="black")
    if dl > 0:
        for x in (-dl, dl):
            fig.add_vline(x=x, line_dash="dot", line_color="gray")
    fig.update_layout(height=max(420, 26 * len(g) + 330), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=f"평균차이 MD ({cell['unit']})  ◀ 제로음료가 더 낮음 | 더 높음 ▶   (점선 = 랜덤효과 예측구간)",
                      yaxis=dict(categoryorder="array", categoryarray=list(g.lab[::-1]) + MODELS[::-1]),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    st.subheader("클래스 확률 (모델이 추정한 효과 분포 기준)")
    pr = go.Figure()
    avail = [m for m in MODELS if cell["res"][m]]
    P = {m: probs(cell["res"][m], dl) for m in avail}
    for i, k in enumerate(CLS):
        pr.add_trace(go.Bar(y=avail, x=[P[m][i] for m in avail], orientation="h", name=k, marker_color=CCOLOR[k],
                            text=[f"{P[m][i]:.0%}" for m in avail], textposition="inside"))
    pr.update_layout(barmode="stack", height=60 * len(avail) + 120, xaxis=dict(range=[0, 1], tickformat=".0%"),
                     yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(pr, width="stretch")
    st.caption("확률 = 추정치와 표준오차(정규/t 근사)로 계산한 P(효과<−δ), P(−δ~δ), P(효과>δ). δ=0이면 '불분명' 확률은 0이 되고 개선/악화 확률만 남습니다.")
    rows = []
    for m in avail:
        r = cell["res"][m]
        rows.append({"모델": m, "MD": round(r["mu"], 3), "95% CI": f"[{r['lo']:.2f}, {r['hi']:.2f}]", "판정": cell["cls"][m],
                     "P(개선)": f"{P[m][0]:.0%}", "P(악화)": f"{P[m][2]:.0%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with t3:
    st.subheader("이질성 · 다수준 구조")
    dg = []
    for c in cells:
        r1, r2 = c["res"][MODELS[1]], c["res"][MODELS[2]]
        if r1:
            dg.append({"지표": c["metric"], "비교": c["comp"], "행 수": c["k"], "논문 수(G)": r2.get("G", ""),
                       "τ²(랜덤효과)": round(r1.get("tau2", 0), 3), "τ²(논문 간)": round(r2.get("tb", 0), 3),
                       "τ²(논문 내)": round(r2.get("tw", 0), 3),
                       "예측구간": f"[{r1['pi'][0]:.2f}, {r1['pi'][1]:.2f}]" if r1.get("pi") else "—(연구<3)"})
    st.dataframe(pd.DataFrame(dg), hide_index=True, width="stretch")
    st.caption("τ²가 크면 연구 간 효과가 크게 다르다는 뜻이고, 예측구간이 0을 지나면 '다음 연구에서 반대 방향이 나올 수 있음'을 뜻합니다. "
               "행 수 ≫ 논문 수이면 ①의 연구 독립 가정이 과대평가될 수 있어 ②를 함께 보세요.")
    st.subheader("네트워크 일치성 (직접 vs 간접 근거)")
    cons = [{"지표": m, **r} for m in [x for x in METRIC_KO.values() if x in set(d.metric)] for r in consistency(d[d.metric == m])]
    if cons:
        st.dataframe(pd.DataFrame(cons), hide_index=True, width="stretch")
    else:
        st.info("세 비교가 모두 있는 지표가 없어 일치성 검정을 할 수 없습니다.")
