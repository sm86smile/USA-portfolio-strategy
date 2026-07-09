# 강환국 ETF 전략 대시보드

Streamlit 앱으로 LAA, VAA 공격형, 오리지널 듀얼 모멘텀의 ETF 비중, 투자금, 리밸런싱 일정을 계산합니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## API Key 설정

로컬에서 `.streamlit/secrets.toml` 파일을 만들고 아래 내용을 입력하세요.

```toml
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
```

`secrets.toml`은 GitHub에 올리면 안 됩니다.
