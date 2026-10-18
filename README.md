# 🍊 제주어 ↔ 표준어 양방향 번역기

표준어를 제주어로, 제주어를 표준어로 번역하는 신경망 번역 모델입니다.
모델 하나가 두 방향을 모두 처리하며, 입력 앞에 붙는 방향 태그(`<ko2je>`, `<je2ko>`)로 방향을 구분합니다.

- 기반 모델: [KoBART](https://huggingface.co/gogamza/kobart-base-v2) (`gogamza/kobart-base-v2`)
- 학습 데이터: JIT 데이터셋 (카카오브레인), 필요하면 AI Hub 제주 방언 데이터 추가
- 데모: _(Hugging Face Spaces 링크를 여기에 넣으세요)_

```
<ko2je> 어서 오세요   →  혼저 옵서예
<je2ko> 고맙수다      →  고맙습니다
```

## 폴더 구조

```
jeju-translator/
├── src/jeju_translator/   # 번역기 코드 (공통 설정, Translator 클래스)
├── scripts/
│   ├── preprocess.py      # 병렬 코퍼스 → 양방향 학습 데이터
│   ├── train.py           # 학습 (+ Hub 업로드)
│   └── evaluate.py        # 방향별 BLEU/chrF 평가
├── app/                   # Gradio 웹 데모
├── notebooks/colab_train.ipynb   # Colab에서 처음부터 끝까지 실행
└── data/                  # 데이터 안내 + 동작 확인용 예시 문장
```

## 빠른 시작

### 처음 한 번: 이 폴더를 GitHub 저장소로 올리기

GitHub에서 **빈 저장소** `jeju-translator`를 만든 뒤(README·.gitignore 체크 해제), 압축을 푼 폴더에서:

```bash
git init
git add .
git commit -m "Initial commit: bidirectional Jeju translator"
git branch -M main
git remote add origin https://github.com/<아이디>/jeju-translator.git
git push -u origin main
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<아이디>/jeju-translator/blob/main/notebooks/colab_train.ipynb)
← 링크의 `<아이디>`를 바꾸면 이 배지로 Colab 노트북이 바로 열립니다.

### 0. 설치

```bash
git clone https://github.com/<아이디>/jeju-translator.git
cd jeju-translator
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .        # jeju_translator 패키지와 jeju-translate 명령 설치
```

> GPU가 없다면 `notebooks/colab_train.ipynb`를 Google Colab에서 여는 것이 가장 쉽습니다.

### 1. 동작 확인 (예시 데이터, 1~2분)

```bash
python scripts/preprocess.py --raw_dir data/sample
python scripts/train.py --epochs 1 --batch_size 4
jeju-translate --direction ko2je "어서 오세요"
```

예시 데이터는 10문장뿐이라 번역 결과는 엉터리입니다. 오류 없이 끝까지 돌아가는지만 확인하세요.

### 2. 실제 데이터 준비

[data/README.md](data/README.md)를 따라 JIT 데이터셋을 `data/raw/`에 넣은 뒤:

```bash
python scripts/preprocess.py --raw_dir data/raw
```

### 3. 학습

```bash
# 먼저 일부만 빠르게 돌려서 확인
python scripts/train.py --limit 2000 --epochs 1

# 전체 학습 (Colab T4 GPU 기준 몇 시간 걸릴 수 있음)
python scripts/train.py --epochs 5
```

GPU 메모리가 부족하면 `--batch_size 16 --grad_accum 2`처럼 나눠서 학습합니다.
결과 모델은 `outputs/final/`에 저장됩니다.

### 4. 평가

```bash
python scripts/evaluate.py --model outputs/final
```

방향별 점수가 표로 출력되고, `results/samples_ko2je.tsv`, `results/samples_je2ko.tsv`에서 실제 번역 사례를 볼 수 있습니다.

| 방향 | 문장 수 | BLEU | chrF |
|---|---:|---:|---:|
| 표준어 → 제주어 | – | – | – |
| 제주어 → 표준어 | – | – | – |

_(학습 후 결과로 채워 넣으세요)_

### 5. 번역해 보기

```bash
jeju-translate --direction ko2je "어서 오세요" "많이 드세요"
jeju-translate --direction je2ko "혼저 옵서예"
jeju-translate --model <아이디>/kobart-jeju-bidirectional --direction je2ko "고맙수다"   # Hub 모델
```

파이썬 코드에서:

```python
from jeju_translator import Translator
tr = Translator("outputs/final")          # 또는 Hub 이름
tr.translate("어서 오세요", "ko2je")
tr.translate(["혼저 옵서예", "고맙수다"], "je2ko")
```

### 6. 웹 데모

```bash
python app/app.py                         # http://127.0.0.1:7860
```

### 7. 공개하기

1. **모델 → Hugging Face Hub**
   ```bash
   huggingface-cli login
   python scripts/train.py --epochs 5 --push_to_hub <아이디>/kobart-jeju-bidirectional
   ```
2. **데모 → Hugging Face Spaces** (SDK: Gradio)
   Space 루트에 `app/app.py` → `app.py`, `app/requirements.txt` → `requirements.txt`, `src/jeju_translator/` 폴더를 올리고,
   Settings → Variables에 `MODEL_ID=<아이디>/kobart-jeju-bidirectional`을 등록합니다.
3. **코드 → GitHub**: 이 README의 데모 링크와 평가 표를 채운 뒤 push합니다.

## 개선 아이디어

- 더 큰 모델 비교: `google/mt5-base`, `facebook/nllb-200-distilled-600M` (`--model` 옵션으로 바꿀 수 있음)
- AI Hub 제주 방언 데이터 추가
- 역번역(back-translation)으로 데이터 늘리기
- 제주어사전 기반 단어 풀이 기능
- 제주어 화자의 결과 검토 (제주어는 표기가 일정하지 않아 자동 점수만으로는 한계가 있음)

## 데이터 출처와 라이선스

- **코드**: MIT License ([LICENSE](LICENSE))
- **JIT 데이터셋**: 카카오브레인, Park et al., *Jejueo Datasets for Machine Translation and Speech Synthesis* (LREC 2020). 비상업 라이선스이므로 학습된 모델도 같은 조건을 따릅니다.
- **AI Hub 데이터**: 사용 시 AI Hub 이용 약관을 따르며, 원본 데이터는 재배포하지 않습니다.

제주어는 유네스코가 지정한 소멸위기 언어입니다. 이 프로젝트가 제주어를 기록하고 잇는 데 작은 보탬이 되기를 바랍니다.
