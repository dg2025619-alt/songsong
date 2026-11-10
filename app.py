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

APP_DIR = Path(__file__).resolve().parent
METRIC_KO = {"Body Weight": "체중", "BMI": "BMI", "Body Fat": "체지방률", "Waist Circumference": "허리둘레",
             "HbA1c": "당화혈색소", "Fasting Blood Glucose": "공복혈당", "2-Hour Post-Prandial Glucose": "식후 2시간 혈당"}
# edge: (Treatment1, Treatment2, 표시이름). 행의 MD = Treatment2 − Treatment1 (제로음료 비교는 MD<0 이면 제로음료가 더 낮음)
EDGES = {"A": ("SSB", "LNCSB", "제로음료 vs 설탕음료"), "B": ("Water", "LNCSB", "제로음료 vs 물"),
         "C": ("SSB", "Water", "물 vs 설탕음료")}
PAPER_EDGE = {"LNCSBs for SSBs": "A", "LNCSBs for Water": "B"}
MODELS = ["베이스라인(고정효과)", "① 랜덤효과+HK", "② 다수준+RVE", "③ 네트워크", "논문 보고값"]
MCOLOR = ["#7F7F7F", "#4C78A8", "#F58518", "#54A24B", "#B279A2"]
CLS, CODE = ["개선", "불분명", "악화"], {"개선": -1, "불분명": 0, "악화": 1}
CCOLOR = {"개선": "#8FD18F", "불분명": "#E3E3E3", "악화": "#F4A6A6"}
X_EDGE = {"A": [1, 0], "B": [1, -1], "C": [0, 1]}          # 기본 파라미터: d_L(제로−설탕), d_W(물−설탕)
CONTRAST = {"A": np.array([1, 0]), "B": np.array([1, -1])}  # A: 제로−설탕, B: 제로−물


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

t1, t2, t3 = st.tabs(["① 판정표 한눈에", "② 칸별 자세히", "③ 진단(이질성·일치성)"])

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
