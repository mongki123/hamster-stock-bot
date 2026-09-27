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
# Env
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")

if not all([TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET]):
    raise ValueError("X API keys are missing.")


# =========================
# Clients
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
    "Technology": "XLK",
    "Communication": "XLC",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

HAMSTER_STYLE = (
    "The same cute chubby hamster mascot every time, round body, tiny paws, "
    "soft brown fur, big shiny eyes, small hoodie, chibi illustration, "
    "Korean webtoon style, clean composition, warm soft lighting, "
    "no text, no letters, no watermark, no logo, high quality"
)


# =========================
# Time / market helpers
# =========================
def now_et() -> datetime.datetime:
    return datetime.datetime.now(ZoneInfo("America/New_York"))


def extract_close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")

    close_col = df["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_series = close_col.iloc[:, 0]
    else:
        close_series = close_col

    return close_series.dropna()


def was_us_market_open_on(date_obj: datetime.date) -> bool:
    df = yf.download(
        "^GSPC",
        period="15d",
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        return False

    traded_dates = [idx.date() for idx in df.index]
    return date_obj in traded_dates


def previous_calendar_day(date_obj: datetime.date) -> datetime.date:
    return date_obj - datetime.timedelta(days=1)


def resolve_target_session(now: datetime.datetime) -> tuple[str, datetime.date]:
    """
    Weekday after 16:20 ET -> today's US session if it existed.
    Weekday before close -> previous session.
    Weekend / holiday -> offday.
    """
    today = now.date()

    if today.weekday() >= 5:
        return "offday", previous_calendar_day(today)

    after_close = now.hour > 16 or (now.hour == 16 and now.minute >= 20)
    candidate = today if after_close else previous_calendar_day(today)

    for _ in range(7):
        if was_us_market_open_on(candidate):
            if candidate == today or not after_close:
                return "market", candidate
            if candidate != today and today.weekday() < 5 and after_close and not was_us_market_open_on(today):
                return "offday", candidate
            return "market", candidate
        candidate = previous_calendar_day(candidate)

    return "offday", previous_calendar_day(today)


def get_top_sector_line() -> str:
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
        return "Sector leadership was mixed."

    results.sort(key=lambda x: x[1], reverse=True)
    top_name, top_pct = results[0]
    if top_pct >= 0.15:
        return f"{top_name} led the tape."
    if top_pct <= -0.15:
        return f"{top_name} was the weakest area."
    return "No sector really took over."


def get_symbol_change(symbol: str, target_date: datetime.date):
    df = yf.download(
        symbol,
        period="15d",
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise ValueError(f"{symbol} data is empty")

    df = df.sort_index()
    dates = [idx.date() for idx in df.index]
    if target_date not in dates:
        raise ValueError(f"{symbol} has no bar for {target_date}")

    idx_pos = dates.index(target_date)
    if idx_pos == 0:
        raise ValueError(f"{symbol} has no previous session")

    prev_date = dates[idx_pos - 1]
    df_target = df.loc[df.index.map(lambda x: x.date() == target_date)]
    df_prev = df.loc[df.index.map(lambda x: x.date() == prev_date)]

    if df_target.empty or df_prev.empty:
        raise ValueError(f"{symbol} date slice failed")

    close_today = float(extract_close_series(df_target).to_numpy()[-1])
    close_prev = float(extract_close_series(df_prev).to_numpy()[-1])
    if close_prev == 0:
        raise ValueError(f"{symbol} previous close is 0")

    pct = (close_today / close_prev - 1.0) * 100.0
    return close_today, pct


def fmt_pct(pct: float) -> str:
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def get_fear_greed_value() -> str:
    try:
        data = fear_and_greed.get()
        value = int(data.value)
        label = str(getattr(data, "description", "") or "").strip()
        if label:
            return f"{value} ({label})"
        return str(value)
    except Exception as e:
        print("Fear & Greed lookup failed:", e)
        return "N/A"


def fetch_market_info(target_date: datetime.date) -> dict:
    _, dji_pct = get_symbol_change("^DJI", target_date)
    _, spx_pct = get_symbol_change("^GSPC", target_date)
    _, ixic_pct = get_symbol_change("^IXIC", target_date)

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "dow": fmt_pct(dji_pct),
        "sp500": fmt_pct(spx_pct),
        "nasdaq": fmt_pct(ixic_pct),
        "fear_greed": get_fear_greed_value(),
        "sectors": get_top_sector_line(),
    }


# =========================
# Prompts
# =========================
def build_prompt_for_market_day(market_info: dict) -> str:
    return f"""
You are the hamster behind the X account "Market Hamster". (@hamstocky).
Write in casual, clear, global English.
Sound like a tiny market intern with jokes, not an analyst.
Never recommend buying or selling anything.
Never mention a specific stock ticker.
Never invent headlines or macro events.
Use only the data below.

DATA
- US session date: {market_info["date"]}
- Dow: {market_info["dow"]}
- S&P 500: {market_info["sp500"]}
- Nasdaq: {market_info["nasdaq"]}
- Fear & Greed: {market_info["fear_greed"]}
- Sector flow: {market_info["sectors"]}

TWEET RULES
- This is a US market close recap.
- Must include Dow, S&P 500, and Nasdaq percentages.
- Mention sector flow as a vibe, not a lecture.
- One short hamster line. Dry, cute, slightly self-aware.
- Max 3 emojis.
- Max 2 hashtags: #USMarkets #MarketClose
- Mobile line breaks.
- Whole tweet including hashtags and line breaks MUST be under 220 characters.
- English only.

OUTPUT
1) tweet body only
2) last line exactly like this:
🎨 today's hamster image: one vivid scene matching the market mood
"""


def build_prompt_for_offday(today_et: datetime.date, previous_day: datetime.date) -> str:
    return f"""
You are the hamster behind the X account "Market Hamster". (@hamstocky).
Write in casual, clear, global English.
Never recommend stocks.
Never invent index numbers.

CONTEXT
- Today in New York: {today_et.strftime("%Y-%m-%d")}
- Previous day: {previous_day.strftime("%Y-%m-%d")}
- US markets are closed.

TWEET RULES
- Say the market is closed.
- Then 1-2 lines about patience, rest, studying, or not forcing trades.
- Keep it light and memorable.
- Max 3 emojis.
- Hashtags: #USMarkets #MarketClosed
- Whole tweet including hashtags and line breaks MUST be under 220 characters.
- English only.

OUTPUT
1) tweet body only
2) last line exactly like this:
🎨 today's hamster image: one vivid off-duty hamster scene
"""


def generate_morning_tweet(market_info: dict) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": "You write short, addictive market-close tweets for a global X audience. No financial advice.",
            },
            {"role": "user", "content": build_prompt_for_market_day(market_info)},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_offday_tweet(today_et: datetime.date, previous_day: datetime.date) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": "You write short, addictive market-close tweets for a global X audience. No financial advice.",
            },
            {"role": "user", "content": build_prompt_for_offday(today_et, previous_day)},
        ],
    )
    return response.choices[0].message.content.strip()


def split_tweet_and_image_prompt(full_text: str):
    lines = full_text.split("\n")
    image_prompt = None
    tweet_lines = []

    for line in lines:
        lowered = line.strip().lower()
        if lowered.startswith("🎨 today's hamster image:") or lowered.startswith("today's hamster image:"):
            image_prompt = line.split(":", 1)[1].strip()
        else:
            tweet_lines.append(line)

    tweet_text = "\n".join(tweet_lines).strip()
    return tweet_text, image_prompt


def trim_tweet_length(text: str, max_len: int = 220) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


# =========================
# Image + post
# =========================
def generate_hamster_image(image_prompt: str) -> str:
    scene = image_prompt or "a hamster watching market candles on a tiny monitor"
    prompt = (
        f"{HAMSTER_STYLE}. Scene: {scene}. "
        "Keep the hamster recognizable as the same character."
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
        raise ValueError("No image payload returned.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(img_bytes)
    tmp.close()
    return tmp.name


def post_to_x_with_image(tweet_text: str, image_prompt: str | None):
    image_path = None
    media_ids = None

    try:
        if image_prompt:
            print("Generating image:", image_prompt)
            try:
                image_path = generate_hamster_image(image_prompt)
                media = twitter_v1.media_upload(filename=image_path)
                media_ids = [media.media_id]
                print("Media uploaded:", media.media_id)
            except Exception as image_error:
                print("Image failed, posting text only:", image_error)

        if media_ids:
            resp = twitter_v2.create_tweet(text=tweet_text, media_ids=media_ids)
        else:
            resp = twitter_v2.create_tweet(text=tweet_text)

        print("Tweet posted:", resp)
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


# =========================
# Main
# =========================
def run_bot():
    current = now_et()
    mode, target_date = resolve_target_session(current)

    print("Now ET:", current.isoformat())
    print("Mode:", mode)
    print("Target date:", target_date)

    if mode == "market":
        print("US session recap mode")
        market_info = fetch_market_info(target_date)
        full_text = generate_morning_tweet(market_info)
    else:
        print("Market closed mode")
        full_text = generate_offday_tweet(current.date(), target_date)

    print("=== RAW MODEL OUTPUT ===")
    print(full_text)
    print("========================")

    tweet_text, image_prompt = split_tweet_and_image_prompt(full_text)
    tweet_text = trim_tweet_length(tweet_text, max_len=220)

    if not image_prompt:
        image_prompt = "the same chubby hamster at a tiny desk, watching red and green candles, warm lamp light"

    print("=== FINAL TWEET ===")
    print(tweet_text)
    print("=== IMAGE PROMPT ===")
    print(image_prompt)
    print("====================")

    post_to_x_with_image(tweet_text, image_prompt)


if __name__ == "__main__":
    run_bot()
