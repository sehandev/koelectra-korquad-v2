# Fine-tuning KoELECTRA

```bash
python -m pip install -r requirements.txt
python main.py
```

## Data

KorQuAD v2.1 train → data/train.json

KorQuAD v2.1 dev → data/dev.json

### 데이터 변경

1. 데이터를 train.json, dev.json에 적절한 비율로 나누어 저장
2. main.py의 read_korquad_v2를 수정
→ return값으로 contexts, questions, answers 필요
