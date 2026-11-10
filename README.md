# songsong# 제로음료 효과 판정 — 모델 비교 앱

(지표 × 비교) 칸을 개선 / 불분명 / 악화로 분류하고, 베이스라인(고정효과)과 모델 3종(랜덤효과, 다수준, 네트워크 메타분석)의 판정을 시각으로 비교하는 Streamlit 앱입니다.

## 폴더 구조
```
프로젝트폴더/
├─ app.py
├─ requirements.txt
├─ 제로음료_임상시험_원자료_체중혈당.xlsx
├─ .gitignore
└─ .streamlit/
   └─ config.toml
```

## 1) 내 컴퓨터에서 실행
```
python -m venv .venv
source .venv/bin/activate          # 윈도우: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
브라우저에서 http://localhost:8501 이 열립니다.

## 2) GitHub에 올리기
```
git init
git add .
git commit -m "제로음료 판정 모델 비교 앱"
git branch -M main
git remote add origin https://github.com/<내계정>/<저장소이름>.git
git push -u origin main
```
엑셀 파일도 반드시 같이 올려야 합니다(앱이 이 파일을 읽습니다).

## 3) Streamlit Community Cloud로 배포 (새 주소 만들기)
1. https://share.streamlit.io 에 GitHub 계정으로 로그인
2. **Create app** → **Deploy a public app from GitHub**
3. Repository: `<내계정>/<저장소이름>`, Branch: `main`, Main file path: `app.py`
4. (선택) App URL에서 원하는 주소 이름 지정
5. **Advanced settings**에서 Python 3.11 또는 3.12 선택 후 **Deploy**

몇 분 뒤 `https://<앱이름>.streamlit.app` 형태의 주소가 생깁니다. 이후 `git push`만 하면 자동으로 다시 배포됩니다.

## 문제 해결
| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError` | `requirements.txt` 누락 또는 오타 — 파일이 저장소 루트에 있는지 확인 |
| `엑셀 파일을 찾을 수 없습니다` | 엑셀 파일명이 코드의 `EXCEL_FILE`과 같은지, 저장소에 올라갔는지 확인 |
| `width="stretch"` 관련 오류 | Streamlit 버전이 낮은 것 — `streamlit>=1.50.0`으로 올리기 |
| 한글 파일명이 깨짐 | 영문 이름으로 바꾸고 `app.py`의 `EXCEL_FILE`만 같이 수정 |
| 데이터를 바꿨는데 반영 안 됨 | 앱 메뉴(⋮) → Clear cache 후 새로고침 |

## 해석할 때 주의
- 부호는 "MD<0이면 제로음료가 더 낮음", 모든 지표는 "낮을수록 개선"으로 가정했습니다. 원문 PDF로 재확인하세요.
- ③ 네트워크 모델은 빈도주의 근사이며 다중군 연구의 행 간 상관을 무시합니다. ② 다수준 모델도 같은 논문 행 사이의 표본 공분산을 알 수 없어 독립으로 가정합니다.
- 성인 대상 임상시험 요약값 기준이라 청소년에게 그대로 일반화할 수 없습니다.
