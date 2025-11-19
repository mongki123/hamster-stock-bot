import os
import datetime
import base64

from dotenv import load_dotenv
import tweepy
from openai import OpenAI

import yfinance as yf
import fear_and_greed
from zoneinfo import ZoneInfo

# Load secrets (.env or Replit Secrets)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

# 🔐 /run 보호용 토큰 (원하는 걸로 바꿔써)
# SECRET_TOKEN = os.getenv("BOT_SECRET_TOKEN", "hamster-secret-token")

# ==== OpenAI Client ====
client = OpenAI(api_key=OPENAI_API_KEY)

# ==== Twitter Clients ====
# v1.1 (이미지 업로드용 - 지금은 안 씀)
auth_v1 = tweepy.OAuth1UserHandler(
    TWITTER_API_KEY,
    TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_SECRET,
)
twitter_v1 = tweepy.API(auth_v1)

# v2 → 텍스트 트윗용
twitter_v2 = tweepy.Client(
    consumer_key=TWITTER_API_KEY,
    consumer_secret=TWITTER_API_SECRET,
    access_token=TWITTER_ACCESS_TOKEN,
    access_token_secret=TWITTER_ACCESS_SECRET,
)

# 🔥 Flask 앱 생성
# app = Flask(__name__)

SECTOR_ETFS = {
    "기술": "XLK",
    "커뮤니케이션": "XLC",
    "헬스케어": "XLV",
    "금융": "XLF",
    "산업재": "XLI",
    "경기소비재": "XLY",
    "필수소비재": "XLP",
    "에너지": "XLE",
    "유틸리티": "XLU",
    "부동산": "XLRE",
    "소재": "XLB",
}


def get_top_sector_line() -> str | None:
    """전일 대비 수익률 기준으로 가장 잘 간 섹터 한 줄 설명을 만들어 줌."""
    results: list[tuple[str, float]] = []

    for name, ticker in SECTOR_ETFS.items():
        df = yf.download(
            ticker,
            period="5d",  # 최근 5일만 받으면 충분
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        df = df.sort_index()

        if len(df) < 2:
            # 데이터가 하루밖에 없으면 패스
            continue

        close_today = float(df["Close"].iloc[-1])
        close_prev = float(df["Close"].iloc[-2])

        if close_prev == 0:
            continue

        pct = (close_today - close_prev) / close_prev * 100.0
        results.append((name, pct))

    if not results:
        return None

    # 수익률 높은 순으로 정렬
    results.sort(key=lambda x: x[1], reverse=True)
    top_name, top_pct = results[0]

    direction = "상승" if top_pct >= 0 else "하락"
    # 예: "오늘 가장 강했던 섹터는 기술 섹터로, 전일 대비 +1.23% 상승했어."
    line = f"오늘 가장 강했던 섹터는 {top_name} 섹터로, 전일 대비 {top_pct:+.2f}% {direction}했어."
    return line


# -------------------------------
# 날짜 관련 (KST 기준)
# -------------------------------
def get_today_kst() -> datetime.date:
    now_kst = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
    return now_kst.date()


# -------------------------------
# 전날이 실제 미국 거래일인지 체크
# -------------------------------
def was_us_market_open_on(date_obj: datetime.date) -> bool:
    """
    전날(KST) 날짜 기준으로, 그 날이 실제 미국장에서 거래가 있었는지 여부 체크.
    ^GSPC( S&P500 ) 데이터를 10일 치 받아와서 날짜 존재 여부로 판단.
    """
    df = yf.download("^GSPC",
                     period="10d",
                     interval="1d",
                     progress=False,
                     auto_adjust=False)
    traded_dates = [idx.date() for idx in df.index]
    return date_obj in traded_dates


def get_symbol_change(symbol: str, target_date: datetime.date):
    """
    해당 symbol의 target_date 종가와 전일 대비 등락률(%)을 반환.
    """
    df = yf.download(symbol,
                     period="10d",
                     interval="1d",
                     progress=False,
                     auto_adjust=False)
    df = df.sort_index()

    dates = [idx.date() for idx in df.index]
    if target_date not in dates:
        raise ValueError(f"{symbol} 에 해당 날짜 데이터가 없음: {target_date}")
    idx_pos = dates.index(target_date)
    if idx_pos == 0:
        raise ValueError(f"{symbol} 에 대해 이전 거래일 데이터가 부족함")

    # close_today = float(df.iloc[idx_pos]["Close"])
    # close_prev = float(df.iloc[idx_pos - 1]["Close"])

    close_today = float(df["Close"].iloc[idx_pos])
    close_prev = float(df["Close"].iloc[idx_pos - 1])

    pct = (close_today / close_prev - 1.0) * 100.0
    return close_today, pct


def fmt_pct(pct: float) -> str:
    """
    등락률을 +0.8% / -1.2% 형태로 포맷.
    """
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def get_fear_greed_value() -> str:
    """
    CNN Fear & Greed Index 현재 값 가져오기 (실패하면 'N/A').
    """
    try:
        data = fear_and_greed.get()
        # data.value, data.description, data.last_updated
        value = int(data.value)
        return str(value)
    except Exception as e:
        print("⚠️ 공포·탐욕 지수 조회 실패:", e)
        return "N/A"


# -------------------------------
# 실제 미장 데이터 구성
# -------------------------------
def fetch_market_info(target_date: datetime.date) -> dict:
    """
    target_date 기준으로
    - 다우 / S&P500 / 나스닥 등락률
    - 주요섹터 등락률
    - 공포·탐욕 지수
    를 가져와서 dict로 반환.
    """
    # 주요 지수
    dji_close, dji_pct = get_symbol_change("^DJI", target_date)
    spx_close, spx_pct = get_symbol_change("^GSPC", target_date)
    ixic_close, ixic_pct = get_symbol_change("^IXIC", target_date)

    fear_greed = get_fear_greed_value()
    sector_line = get_top_sector_line()

    sectors_str = sector_line if sector_line else ""

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "dow": fmt_pct(dji_pct),
        "sp500": fmt_pct(spx_pct),
        "nasdaq": fmt_pct(ixic_pct),
        "fear_greed": fear_greed,
        "sectors": sectors_str,
        # 실제 뉴스 API는 안 쓰고, GPT가 지수/섹터 흐름 기반으로
        # '분위기 설명'만 하도록 할 거라 리스트는 비워둠
        "news": [],
        "fx_oil_rate": "주요 환율·유가·금리 특이사항은 생략",
    }


# -------------------------------
# GPT 프롬프트 (거래일용)
# -------------------------------
def build_prompt_for_market_day(market_info: dict) -> str:
    date = market_info["date"]
    dow = market_info["dow"]
    sp500 = market_info["sp500"]
    nasdaq = market_info["nasdaq"]
    fear_greed = market_info["fear_greed"]
    sectors = market_info["sectors"]
    fx_oil_rate = market_info["fx_oil_rate"]

    return f"""
너는 X 계정 “주식하는 동물”을 운영하는 햄스터 캐릭터야.
반말, 친근한 공감톤, 살짝 개그 섞기.
매수 추천이나 특정 종목 선동은 절대 금지.
글은 한국어로 작성해.
해시태그, 줄바꿈, 이모티콘 포함 본문 전체 길이를 "무조건 100~110자 사이"로 맞춰줘.

출력 구조:
1) 본문
2) 맨 마지막 줄에 반드시 아래 형식 추가:
🎨 오늘 햄스터 이미지: {{이미지 묘사 한 줄}}

[오늘 정보 입력]
- 날짜(미국 기준 거래일): {date}
- 미국장 지수 등락률:
  - 다우: {dow}
  - S&P500: {sp500}
  - 나스닥: {nasdaq}
  - 공포와 탐욕 지수: {fear_greed}
- 주요 섹터 움직임: {sectors}
- 환율/유가/금리(선택): {fx_oil_rate}

주의:
- 실제 뉴스 헤드라인을 지어내지 말고,
  위 숫자/섹터 흐름을 기반으로 “분위기 설명”만 해줘.
- 다우, S&P500, 나스닥 등락률은 반드시 포함하나, 주요 섹터 움직임은 등락률 제외 흐름만 알려줘.

글쓰기 조건:
- 해시태그, 줄바꿈 포함 본문 전체 길이를 "무조건 100~110자 사이"로 맞춰줘.
- “어제 미장 요약” 형식.
- 햄스터 멘트(공감+개그) 1줄 포함.
- 모바일 X에서 보기 좋은 줄바꿈은 필수야!
- 귀여운 이모티콘 사용 가능.
- 마지막 줄 형식:
  🎨 오늘 햄스터 이미지: {{한 장면을 상상할 수 있는 묘사}}
- 해시태그 1~3개 (#미국주식 #미장요약 #주식하는햄스터 등)
"""


def generate_morning_tweet(market_info: dict) -> str:
    prompt = build_prompt_for_market_day(market_info)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.8,
        messages=[
            {
                "role": "system",
                "content": "너는 X에 글 쓰는 한국어 햄스터 캐릭터야."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )
    return response.choices[0].message.content.strip()


# -------------------------------
# GPT 프롬프트 (휴장/주말용)
# -------------------------------
def build_prompt_for_offday(today: datetime.date,
                            yesterday: datetime.date) -> str:
    today_str = today.strftime("%Y-%m-%d")
    y_str = yesterday.strftime("%Y-%m-%d")

    return f"""
너는 X 계정 “주식하는 동물”을 운영하는 햄스터 캐릭터야.
반말, 친근한 공감톤, 살짝 개그 섞기.
매수 추천이나 특정 종목 선동은 절대 금지.
글은 한국어로 작성해.

상황:
- 오늘 날짜(KST): {today_str}
- 전날(KST): {y_str}
- 전날은 미국장이 열리지 않은 날이야 (주말/공휴일 등 휴장).
- 그래서 오늘은 미장 숫자 요약 대신,
  햄스터의 일상 / 투자 멘탈 / 공부 / 휴식과 관련된 가벼운 글을 올리려고 해.

글쓰기 조건:
- “어제 미장은 어떠한 이유로 쉬어갔고, 햄스터는 대신 이런 생각을 했다” 느낌으로 자연스럽게 풀어줘.
- 실제 지수/수치 언급은 최소화하고, 휴장일이라는 사실만 언급.
- 햄스터의 다짐, 공부 계획, 마음가짐 등을 1~2줄 포함.
- 해시태그, 줄바꿈 포함 본문 전체 길이를 "무조건 130자 이내"로 맞춰줘.
- 모바일 X에서 보기 좋은 줄바꿈은 필수야!
- 귀여운 이모티콘 사용 가능.
- 마지막 줄 형식:
  🎨 오늘 햄스터 이미지: {{한 장면을 상상할 수 있는 묘사}}
- 해시태그 3~5개 (#미국주식 #휴장일 #주식하는햄스터 등)

출력 형식:
{{본문 전체}}
"""


def generate_offday_tweet(today: datetime.date,
                          yesterday: datetime.date) -> str:
    prompt = build_prompt_for_offday(today, yesterday)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.8,
        messages=[
            {
                "role": "system",
                "content": "너는 X에 글 쓰는 한국어 햄스터 캐릭터야."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )
    return response.choices[0].message.content.strip()


# -------------------------------
# 본문에서 이미지 설명 추출 + 제거
# -------------------------------
def split_tweet_and_image_prompt(full_text: str):
    """
    전체 텍스트에서 '🎨 오늘 햄스터 이미지:' 줄을 찾아
    - 트윗 본문 (이미지 줄 제거)
    - 이미지 프롬프트 (내부 사용용)
    로 분리.
    """
    lines = full_text.split("\n")
    image_prompt = None
    tweet_lines = []

    for line in lines:
        if line.startswith("🎨 오늘 햄스터 이미지:"):
            image_prompt = line.replace("🎨 오늘 햄스터 이미지:", "").strip()
        else:
            tweet_lines.append(line)

    tweet_text = "\n".join(tweet_lines).strip()
    return tweet_text, image_prompt


def trim_tweet_length(text: str, max_len: int = 140) -> str:
    """
    트윗이 너무 길면 max_len 기준으로 잘라서 반환.
    """
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


# -------------------------------
# 트윗 업로드 (현재: 텍스트만)
# -------------------------------
def post_to_x_with_image(tweet_text: str, image_prompt: str | None):
    """
    (임시 버전)
    - 이미지 생성/업로드는 비활성화
    - 텍스트만 트윗 시도
    """
    try:
        resp = twitter_v2.create_tweet(text=tweet_text)
        print("✅ 트윗 업로드 완료:", resp)
    except Exception as e:
        print("❌ 트윗 업로드 실패:", e)


# -------------------------------
# 메인 실행
# -------------------------------
def run_bot():
    today_kst = get_today_kst()
    yesterday_kst = today_kst - datetime.timedelta(days=1)

    print("오늘(KST):", today_kst)
    print("전날(KST):", yesterday_kst)

    # 전날이 실제 미국 거래일인지 판단
    if was_us_market_open_on(yesterday_kst):
        print("📈 전날은 미국장이 열린 날 → 미장 요약 모드")
        market_info = fetch_market_info(yesterday_kst)
        full_text = generate_morning_tweet(market_info)
    else:
        print("🛌 전날은 미국장이 휴장 → 일상/멘탈 글 모드")
        full_text = generate_offday_tweet(today_kst, yesterday_kst)

    print("=== GPT 생성 원본 ===")
    print(full_text)
    print("=====================")

    tweet_text, image_prompt = split_tweet_and_image_prompt(full_text)

    # 길이 제한 적용 (140자 초과하면 컷)
    tweet_text = trim_tweet_length(tweet_text, max_len=140)

    print("=== 최종 트윗 본문 ===")
    print(tweet_text)
    print("=====================")

    # 현재는 텍스트만 업로드 (이미지 업로드는 비활성화)
    post_to_x_with_image(tweet_text, image_prompt)


# 깃허브 실행으로 인한 불필요 코드
"""
# -------------------------------
# Flask 라우트: 헬스체크 & 봇 실행
# -------------------------------
@app.route("/")
def index():
    return "OK - hamster bot alive"


@app.route("/run")
def run_route():
    token = request.args.get("token")
    if token != SECRET_TOKEN:
        return "Unauthorized", 401

    try:
        print("=== /run 호출됨 → 햄스터 봇 실행 시작 ===")
        run_bot()
        print("=== 햄스터 봇 실행 완료 ===")
        return "Done", 200
    except Exception as e:
        print("❌ run_bot 실행 중 에러:", e)
        return "Error", 500

"""

if __name__ == "__main__":
    run_bot()
