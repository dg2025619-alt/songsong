# 제로음료 효과 판정 — 분류 모델 · 메타분석 모델 비교 앱

데이터가 `app.py` 안에 내장되어 있어 **엑셀 파일 없이** 코드만으로 실행·배포됩니다.

## 폴더 구조
```
프로젝트폴더/
├─ app.py                  ← 앱 본체 (데이터 내장)
├─ requirements.txt
├─ .gitignore
└─ .streamlit/
   └─ config.toml
```

## 로컬 실행
```
pip install -r requirements.txt
streamlit run app.py
```

## GitHub → Streamlit Community Cloud 배포
```
git init
git add .
git commit -m "제로음료 판정 모델 비교 앱"
git branch -M main
git remote add origin https://github.com/<내계정>/<저장소이름>.git
git push -u origin main
```
share.streamlit.io → **Create app** → 저장소/브랜치(`main`)/Main file path(`app.py`) 지정 → Deploy

## 데이터를 바꾸고 싶다면
- **임시로**: 앱 사이드바 "다른 엑셀로 분석하기(선택)"에 같은 시트 구성(`개별연구_원자료`, `종합분석_결과`)의 엑셀을 올리세요.
- **영구적으로**: `app.py` 위쪽의 `STUDY_CSV`, `SUMMARY_CSV` 텍스트를 새 데이터의 CSV로 교체하세요.

## 문제 해결
| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError` | `requirements.txt`가 저장소 루트에 있는지, 패키지 이름 오타가 없는지 확인 (plotly, scipy, scikit-learn 필수) |
| `width="stretch"` 관련 오류 | `streamlit>=1.50.0`으로 올리기 |
| 설치가 오래 걸리거나 호환 오류 | Advanced settings에서 Python 3.12 선택 |

## 해석할 때 주의
- 부호는 "MD<0이면 제로음료가 더 낮음", 모든 지표는 "낮을수록 개선"으로 가정했습니다. 원문 PDF로 재확인하세요.
- 성인 대상 임상시험의 연구 단위 요약값 기준이라 청소년에게 그대로 일반화할 수 없습니다.
- 분류 모델은 표본이 작아(79행, 논문 14편) 점수 차이가 작으면 우연일 수 있습니다.어 독립으로 가정합니다.
- 성인 대상 임상시험 요약값 기준이라 청소년에게 그대로 일반화할 수 없습니다.
