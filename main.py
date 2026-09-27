import os
import datetime
import tempfile
import base64

from dotenv import load_dotenv
import tweepy
from openai import OpenAI
import yfinance as yf
import fear_and_greed
from zoneinfo import ZoneInfo
import pandas as pd
import requests


# =========================
# 환경변수
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 가 없습니다.")

if not all([TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET]):
    raise ValueError("Twitter/X 키가 비어 있습니다.")


# =========================
# 클라이언트
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)

auth_v1 = tweepy.OAuth1UserHandler(
    TWITTER_API_KEY,
    TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_SECRET,
)
twitter_v1 = tweepy.API(auth_v1)

twitter_v2 = tweepy.Client(
    consumer_key=TWITTER_API_KEY,
    consumer_secret=TWITTER_API_SECRET,
    access_token=TWITTER_ACCESS_TOKEN,
    access_token_secret=TWITTER_ACCESS_SECRET,
)


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


# =========================
# yfinance 유틸
# =========================
def extract_close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")

    close_col = df["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_series = close_col.iloc[:, 0]
    else:
        close_series = close_col

    return close_series.dropna()


def get_top_sector_line() -> str | None:
    results: list[tuple[str, float]] = []

    for name, ticker in SECTOR_ETFS.items():
        df = yf.download(
            ticker,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if df is None or df.empty:
            continue

        df = df.sort_index()
        try:
            close_series = extract_close_series(df)
        except Exception:
            continue

        closes = close_series.to_numpy()
        if closes.size < 2:
            continue

        close_today = float(closes[-1])
        close_prev = float(closes[-2])
        if close_prev == 0:
            continue

        pct = (close_today - close_prev) / close_prev * 100.0
        results.append((name, pct))

    if not results:
        return None

    results.sort(key=lambda x: x[1], reverse=True)
    top_name, top_pct = results[0]
    direction = "상승" if top_pct >= 0 else "하락"
    return f"오늘 가장 강했던 섹터는 {top_name} 섹터로, 전일 대비 {top_pct:+.2f}% {direction}했어."


def get_today_kst() -> datetime.date:
    return datetime.datetime.now(ZoneInfo("Asia/Seoul")).date()


def was_us_market_open_on(date_obj: datetime.date) -> bool:
    df = yf.download(
        "^GSPC",
        period="10d",
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        return False

    traded_dates = [idx.date() for idx in df.index]
    return date_obj in traded_dates


def get_symbol_change(symbol: str, target_date: datetime.date):
    df = yf.download(
        symbol,
        period="10d",
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise ValueError(f"{symbol} 데이터가 비어 있음")

    df = df.sort_index()
    dates = [idx.date() for idx in df.index]

    if target_date not in dates:
        raise ValueError(f"{symbol} 에 해당 날짜 데이터가 없음: {target_date}")

    idx_pos = dates.index(target_date)
    if idx_pos == 0:
        raise ValueError(f"{symbol} 에 대해 이전 거래일 데이터가 부족함")

    prev_date = dates[idx_pos - 1]

    df_target = df.loc[df.index.map(lambda x: x.date() == target_date)]
    df_prev = df.loc[df.index.map(lambda x: x.date() == prev_date)]

    if df_target.empty or df_prev.empty:
        raise ValueError(f"{symbol} 날짜 슬라이스 실패: {target_date} / {prev_date}")

    close_today = float(extract_close_series(df_target).to_numpy()[-1])
    close_prev = float(extract_close_series(df_prev).to_numpy()[-1])

    if close_prev == 0:
        raise ValueError(f"{symbol} 이전 종가가 0이라 등락률 계산 불가")

    pct = (close_today / close_prev - 1.0) * 100.0
    return close_today, pct


def fmt_pct(pct: float) -> str:
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def get_fear_greed_value() -> str:
    try:
        data = fear_and_greed.get()
        return str(int(data.value))
    except Exception as e:
        print("⚠️ 공포·탐욕 지수 조회 실패:", e)
        return "N/A"


def fetch_market_info(target_date: datetime.date) -> dict:
    _, dji_pct = get_symbol_change("^DJI", target_date)
    _, spx_pct = get_symbol_change("^GSPC", target_date)
    _, ixic_pct = get_symbol_change("^IXIC", target_date)

    sector_line = get_top_sector_line()

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "dow": fmt_pct(dji_pct),
        "sp500": fmt_pct(spx_pct),
        "nasdaq": fmt_pct(ixic_pct),
        "fear_greed": get_fear_greed_value(),
        "sectors": sector_line if sector_line else "",
        "fx_oil_rate": "주요 환율·유가·금리 특이사항은 생략",
    }


# =========================
# GPT 프롬프트
# =========================
def build_prompt_for_market_day(market_info: dict) -> str:
    return f"""
너는 X 계정 “주식하는 동물”을 운영하는 햄스터 캐릭터야.
반말, 친근한 공감톤, 살짝 개그 섞기.
매수 추천이나 특정 종목 선동은 절대 금지.
글은 한국어로 작성해.

출력 구조:
1) 본문
2) 맨 마지막 줄에 반드시 아래 형식 추가:
🎨 오늘 햄스터 이미지: {{이미지 묘사 한 줄}}

[오늘 정보 입력]
- 날짜(미국 기준 거래일): {market_info["date"]}
- 미국장 지수 등락률:
  - 다우: {market_info["dow"]}
  - S&P500: {market_info["sp500"]}
  - 나스닥: {market_info["nasdaq"]}
  - 공포와 탐욕 지수: {market_info["fear_greed"]}
- 주요 섹터 움직임: {market_info["sectors"]}
- 환율/유가/금리(선택): {market_info["fx_oil_rate"]}

주의:
- 실제 뉴스 헤드라인을 지어내지 말고, 위 숫자/섹터 흐름을 기반으로 분위기만 설명해.
- 다우, S&P500, 나스닥 등락률은 숫자로 반드시 포함.
- 주요 섹터는 등락률 없이 흐름만 알려줘.

글쓰기 조건:
- 해시태그, 줄바꿈 포함 본문 전체 길이를 무조건 100자 이내로 맞춰줘.
- “어제 미장 요약” 형식.
- 햄스터 멘트(공감+개그) 1줄 포함.
- 모바일 X에서 보기 좋은 줄바꿈 필수.
- 귀여운 이모티콘 사용 가능.
- 마지막 줄 형식:
  🎨 오늘 햄스터 이미지: {{한 장면을 상상할 수 있는 묘사}}
- 해시태그 1~3개 (#미국주식 #미장요약 #주식하는햄스터 등)
"""


def build_prompt_for_offday(today: datetime.date, yesterday: datetime.date) -> str:
    return f"""
너는 X 계정 “주식하는 동물”을 운영하는 햄스터 캐릭터야.
반말, 친근한 공감톤, 살짝 개그 섞기.
매수 추천이나 특정 종목 선동은 절대 금지.
글은 한국어로 작성해.

상황:
- 오늘 날짜(KST): {today.strftime("%Y-%m-%d")}
- 전날(KST): {yesterday.strftime("%Y-%m-%d")}
- 전날은 미국장이 열리지 않은 날이야 (주말/공휴일 등 휴장).
- 그래서 오늘은 미장 숫자 요약 대신,
  햄스터의 일상 / 투자 멘탈 / 공부 / 휴식과 관련된 가벼운 글을 올리려고 해.

글쓰기 조건:
- “어제 미장은 쉬어갔고, 햄스터는 대신 이런 생각을 했다” 느낌으로 자연스럽게 풀어줘.
- 실제 지수/수치 언급은 최소화하고, 휴장일이라는 사실만 언급.
- 햄스터의 다짐, 공부 계획, 마음가짐 등을 1~2줄 포함.
- 해시태그, 줄바꿈 포함 본문 전체 길이를 무조건 130자 이내로 맞춰줘.
- 모바일 X에서 보기 좋은 줄바꿈 필수.
- 귀여운 이모티콘 사용 가능.
- 마지막 줄 형식:
  🎨 오늘 햄스터 이미지: {{한 장면을 상상할 수 있는 묘사}}
- 해시태그 3~5개 (#미국주식 #휴장일 #주식하는햄스터 등)

출력 형식:
{{본문 전체}}
"""


def generate_morning_tweet(market_info: dict) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.8,
        messages=[
            {"role": "system", "content": "너는 X에 글 쓰는 한국어 햄스터 캐릭터야."},
            {"role": "user", "content": build_prompt_for_market_day(market_info)},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_offday_tweet(today: datetime.date, yesterday: datetime.date) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.8,
        messages=[
            {"role": "system", "content": "너는 X에 글 쓰는 한국어 햄스터 캐릭터야."},
            {"role": "user", "content": build_prompt_for_offday(today, yesterday)},
        ],
    )
    return response.choices[0].message.content.strip()


def split_tweet_and_image_prompt(full_text: str):
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
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


# =========================
# 이미지 생성 + X 업로드
# =========================
def generate_hamster_image(image_prompt: str) -> str:
    prompt = (
        "A cute chubby hamster character, chibi illustration, Korean webtoon style, "
        "soft warm lighting, clean background, high quality, no text, no watermark, no letters. "
        f"Scene: {image_prompt}"
    )

    result = client.images.generate(
        model="gpt-image-2",
        prompt=prompt,
        size="1024x1024",
        quality="medium",
    )

    item = result.data[0]
    image_b64 = getattr(item, "b64_json", None)
    image_url = getattr(item, "url", None)

    if image_b64:
        img_bytes = base64.b64decode(image_b64)
    elif image_url:
        img_bytes = requests.get(image_url, timeout=60).content
    else:
        raise ValueError("이미지 데이터가 없습니다.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(img_bytes)
    tmp.close()
    return tmp.name


def post_to_x_with_image(tweet_text: str, image_prompt: str | None):
    image_path = None
    media_ids = None

    try:
        if image_prompt:
            print("🎨 이미지 생성 중:", image_prompt)
            image_path = generate_hamster_image(image_prompt)
            media = twitter_v1.media_upload(filename=image_path)
            media_ids = [media.media_id]
            print("✅ 미디어 업로드 완료:", media.media_id)

        if media_ids:
            resp = twitter_v2.create_tweet(
                text=tweet_text,
                media_ids=media_ids,
            )
        else:
            print("⚠️ 이미지 프롬프트가 없어서 텍스트만 업로드")
            resp = twitter_v2.create_tweet(text=tweet_text)

        print("✅ 트윗 업로드 완료:", resp)

    except Exception as e:
        print("❌ 트윗 업로드 실패:", e)
        raise
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


# =========================
# 메인
# =========================
def run_bot():
    today_kst = get_today_kst()
    yesterday_kst = today_kst - datetime.timedelta(days=1)

    print("오늘(KST):", today_kst)
    print("전날(KST):", yesterday_kst)

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
    tweet_text = trim_tweet_length(tweet_text, max_len=140)

    print("=== 최종 트윗 본문 ===")
    print(tweet_text)
    print("=== 이미지 프롬프트 ===")
    print(image_prompt)
    print("=====================")

    post_to_x_with_image(tweet_text, image_prompt)


if __name__ == "__main__":
    run_bot()
