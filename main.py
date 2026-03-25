import html
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from sentiment_agent import SentimentAgent


load_dotenv()


class Config:
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets/v2")
    ALPACA_NEWS_URL = os.getenv("ALPACA_NEWS_URL", "https://data.alpaca.markets/v1beta1/news")

    TICKER = os.getenv("TICKER", "AAPL")
    NEWS_LIMIT = int(os.getenv("NEWS_LIMIT", "200"))
    LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", str(24 * 180)))
    STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
    CONVICTION_THRESHOLD = float(os.getenv("CONVICTION_THRESHOLD", "7.0"))
    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))
    EXECUTE_ORDERS = os.getenv("EXECUTE_ORDERS", "false").lower() == "true"


class MarketDataHandler:
    def __init__(self):
        self.api_key = Config.ALPACA_API_KEY
        self.secret_key = Config.ALPACA_SECRET_KEY
        self.data_url = Config.ALPACA_DATA_URL

    def _alpaca_headers(self) -> Dict[str, str]:
        if not self.api_key or not self.secret_key:
            raise ValueError("Missing Alpaca API credentials in environment variables.")
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    def get_historical_alpaca(
        self,
        ticker: str,
        start: str,
        end: str,
        timeframe: str = "1Day"
    ) -> pd.DataFrame:
        url = f"{self.data_url}/stocks/{ticker}/bars"
        params = {
            "start": start,
            "end": end,
            "timeframe": timeframe,
            "limit": 10000,
            "adjustment": "raw",
            "feed": "iex",
        }

        response = requests.get(
            url,
            headers=self._alpaca_headers(),
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        bars = data.get("bars", [])
        if not bars:
            raise ValueError(f"No Alpaca historical data returned for {ticker}")

        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(None)
        df = df.rename(columns={
            "t": "Date",
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
        })
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace=True)
        return df.sort_index()

    def get_historical_yfinance(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            raise ValueError(f"No yfinance data returned for {ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)

        return df.sort_index()

    def get_historical(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        try:
            if start and end:
                return self.get_historical_alpaca(ticker, start, end, timeframe="1Day")
            return self.get_historical_yfinance(ticker, period=period, interval=interval)
        except Exception as e:
            print(f"[WARN] Alpaca primary failed for {ticker}: {e}")
            print("[INFO] Falling back to yfinance.")
            return self.get_historical_yfinance(ticker, period=period, interval=interval)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["MA50"] = out["Close"].rolling(50).mean()
        out["Return"] = out["Close"].pct_change()
        return out


class NewsFetcher:
    def __init__(self):
        self.api_key = Config.ALPACA_API_KEY
        self.secret_key = Config.ALPACA_SECRET_KEY
        self.news_url = Config.ALPACA_NEWS_URL

    def _alpaca_headers(self) -> Dict[str, str]:
        if not self.api_key or not self.secret_key:
            raise ValueError("Missing Alpaca API credentials in environment variables.")
        return {
            "Apca-Api-Key-Id": self.api_key,
            "Apca-Api-Secret-Key": self.secret_key,
        }

    def get_headlines_alpaca(
        self,
        ticker: str,
        limit: int = 200,
        lookback_hours: int = 24 * 180,
        page_limit: int = 50
    ) -> List[Dict]:
        now_utc = datetime.now(timezone.utc)
        start_utc = now_utc - timedelta(hours=lookback_hours)

        all_items = []
        seen = set()
        next_page_token = None

        while len(all_items) < limit:
            batch_limit = min(page_limit, limit - len(all_items))

            params = {
                "symbols": ticker,
                "start": start_utc.isoformat(),
                "end": now_utc.isoformat(),
                "limit": batch_limit,
                "sort": "desc",
            }

            if next_page_token:
                params["page_token"] = next_page_token

            response = requests.get(
                self.news_url,
                headers=self._alpaca_headers(),
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()

            news_batch = data.get("news", [])
            if not news_batch:
                break

            for item in news_batch:
                headline = html.unescape(item.get("headline", "").strip())
                summary = html.unescape(item.get("summary", "").strip())
                created_at = item.get("created_at")
                url = item.get("url", "")

                dedup_key = (headline, created_at)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                all_items.append({
                    "ticker": ticker,
                    "title": headline,
                    "summary": summary,
                    "published_at": created_at,
                    "source": item.get("source", "alpaca"),
                    "url": url,
                    "raw_text": f"{headline}. {summary}".strip(),
                })

                if len(all_items) >= limit:
                    break

            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

            time.sleep(0.2)

        return all_items

    def get_headlines_yfinance(
        self,
        ticker: str,
        limit: int = 50
    ) -> List[Dict]:
        tk = yf.Ticker(ticker)
        news_items = getattr(tk, "news", []) or []

        normalized = []
        seen = set()

        for item in news_items[:limit]:
            title = html.unescape((item.get("title") or "").strip())
            summary = html.unescape((item.get("summary") or "").strip())
            published_ts = item.get("providerPublishTime")

            if published_ts:
                published_at = datetime.fromtimestamp(published_ts, tz=timezone.utc).isoformat()
            else:
                published_at = None

            dedup_key = (title, published_at)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            normalized.append({
                "ticker": ticker,
                "title": title,
                "summary": summary,
                "published_at": published_at,
                "source": item.get("publisher", "yfinance"),
                "url": item.get("link", ""),
                "raw_text": f"{title}. {summary}".strip(),
            })

        return normalized

    def get_headlines(
        self,
        ticker: str,
        limit: int = 200,
        lookback_hours: int = 24 * 180
    ) -> List[Dict]:
        try:
            return self.get_headlines_alpaca(
                ticker=ticker,
                limit=limit,
                lookback_hours=lookback_hours
            )
        except Exception as e:
            print(f"[WARN] Alpaca news failed for {ticker}: {e}")
            print("[INFO] Falling back to yfinance news.")
            return self.get_headlines_yfinance(ticker=ticker, limit=min(limit, 50))


class AlpacaExecutor:
    def __init__(self):
        self.api_key = Config.ALPACA_API_KEY
        self.secret_key = Config.ALPACA_SECRET_KEY
        self.base_url = Config.ALPACA_BASE_URL

        if not self.api_key or not self.secret_key:
            raise ValueError("Missing Alpaca credentials in environment variables.")

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def get_account(self) -> Dict:
        url = f"{self.base_url}/v2/account"
        response = requests.get(url, headers=self.headers, timeout=20)
        response.raise_for_status()
        return response.json()

    def get_position(self, ticker: str) -> Optional[Dict]:
        url = f"{self.base_url}/v2/positions/{ticker}"
        response = requests.get(url, headers=self.headers, timeout=20)

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return response.json()

    def has_position(self, ticker: str) -> bool:
        return self.get_position(ticker) is not None

    def _get_last_trade_price(self, ticker: str) -> float:
        market_data_url = Config.ALPACA_DATA_URL
        url = f"{market_data_url}/stocks/{ticker}/trades/latest"
        md_headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

        try:
            response = requests.get(url, headers=md_headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            return float(data["trade"]["p"])
        except Exception as e:
            print(f"[WARN] Alpaca latest trade failed for {ticker}: {e}")
            print("[INFO] Falling back to yfinance latest close.")
            hist = yf.download(ticker, period="5d", interval="1d", progress=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if hist.empty:
                raise ValueError(f"Could not fetch fallback price for {ticker}")
            return float(hist["Close"].dropna().iloc[-1])

    def submit_market_buy_with_stop(
        self,
        ticker: str,
        qty: int,
        stop_loss_pct: float = 0.02
    ) -> Dict:
        current_price = self._get_last_trade_price(ticker)
        stop_price = round(current_price * (1 - stop_loss_pct), 2)

        payload = {
            "symbol": ticker,
            "qty": qty,
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "order_class": "bracket",
            "stop_loss": {
                "stop_price": stop_price
            }
        }

        url = f"{self.base_url}/v2/orders"
        response = requests.post(url, headers=self.headers, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()

    def submit_market_sell(self, ticker: str, qty: int) -> Dict:
        payload = {
            "symbol": ticker,
            "qty": qty,
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
        }

        url = f"{self.base_url}/v2/orders"
        response = requests.post(url, headers=self.headers, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()


def generate_trade_decision(
    latest_row: pd.Series,
    sentiment_summary: Dict,
    holding: bool,
    conviction_threshold: float = 7.0
) -> str:
    close_price = float(latest_row["Close"])
    ma50 = latest_row["MA50"]

    price_above_ma = pd.notna(ma50) and close_price > float(ma50)
    positive_sentiment = float(sentiment_summary["weighted_score"]) > 0.0
    strong_conviction = float(sentiment_summary["avg_conviction"]) >= conviction_threshold

    if not holding and price_above_ma and positive_sentiment and strong_conviction:
        return "BUY"

    if holding and (
        (pd.notna(ma50) and close_price < float(ma50)) or
        sentiment_summary["overall_sentiment"] == "NEGATIVE"
    ):
        return "SELL"

    return "HOLD"


def run_live_decision_cycle(
    ticker: str,
    market_df: pd.DataFrame,
    sentiment_summary: Dict,
    qty: int = 1,
    stop_loss_pct: float = 0.02,
    conviction_threshold: float = 7.0,
    execute_orders: bool = False,
):
    latest = market_df.iloc[-1]

    try:
        executor = AlpacaExecutor()
        holding = executor.has_position(ticker)
    except Exception as e:
        print(f"[WARN] Live executor unavailable: {e}")
        holding = False
        executor = None

    decision = generate_trade_decision(
        latest_row=latest,
        sentiment_summary=sentiment_summary,
        holding=holding,
        conviction_threshold=conviction_threshold
    )

    print("========== LIVE DECISION ==========")
    print(f"Ticker: {ticker}")
    print(f"Latest Close: {latest['Close']:.2f}")
    print(f"MA50: {latest['MA50']:.2f}" if pd.notna(latest["MA50"]) else "MA50: N/A")
    print(f"Overall Sentiment: {sentiment_summary['overall_sentiment']}")
    print(f"Avg Conviction: {sentiment_summary['avg_conviction']}")
    print(f"Holding Now: {holding}")
    print(f"Decision: {decision}")

    if not execute_orders:
        return {"decision": decision, "executed": False}

    if executor is None:
        raise RuntimeError("Cannot execute orders because AlpacaExecutor is unavailable.")

    if decision == "BUY":
        order = executor.submit_market_buy_with_stop(
            ticker=ticker,
            qty=qty,
            stop_loss_pct=stop_loss_pct
        )
        return {"decision": decision, "executed": True, "order": order}

    if decision == "SELL":
        position = executor.get_position(ticker)
        sell_qty = int(float(position["qty"])) if position else qty
        order = executor.submit_market_sell(ticker=ticker, qty=sell_qty)
        return {"decision": decision, "executed": True, "order": order}

    return {"decision": decision, "executed": False}


def prepare_backtest_dataset(
    price_df: pd.DataFrame,
    daily_sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    df = price_df.copy().reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["trade_date"] = df["Date"].dt.normalize()

    sent = daily_sentiment_df.copy()
    sent["date"] = pd.to_datetime(sent["date"])
    sent = sent.sort_values("date").reset_index(drop=True)

    trading_days = df["trade_date"].drop_duplicates().sort_values().tolist()

    def get_next_trading_day(news_date):
        news_day = news_date.normalize()
        for d in trading_days:
            if d > news_day:
                return d
        return pd.NaT

    sent["effective_date"] = sent["date"].apply(get_next_trading_day)
    sent = sent.dropna(subset=["effective_date"]).copy()

    def label_from_score(x):
        if x > 0.05:
            return "POSITIVE"
        elif x < -0.05:
            return "NEGATIVE"
        else:
            return "NEUTRAL"

    agg_rows = []
    for effective_date, g in sent.groupby("effective_date"):
        conviction_sum = g["avg_conviction"].sum()
        weighted_sentiment = (
            (g["daily_sentiment"] * g["avg_conviction"]).sum() / conviction_sum
            if conviction_sum > 0 else 0.0
        )

        agg_rows.append({
            "trade_date": effective_date,
            "daily_sentiment": weighted_sentiment,
            "avg_conviction": float(g["avg_conviction"].mean()),
            "news_count": int(g["news_count"].sum()),
        })

    sent_agg = pd.DataFrame(agg_rows)
    if not sent_agg.empty:
        sent_agg["sentiment_signal"] = sent_agg["daily_sentiment"].apply(label_from_score)
    else:
        sent_agg = pd.DataFrame(columns=[
            "trade_date",
            "daily_sentiment",
            "avg_conviction",
            "news_count",
            "sentiment_signal"
        ])

    merged = df.merge(sent_agg, on="trade_date", how="left")
    merged = merged.sort_values("Date").reset_index(drop=True)

    merged["daily_sentiment"] = merged["daily_sentiment"].ffill(limit=1)
    merged["avg_conviction"] = merged["avg_conviction"].ffill(limit=1)
    merged["sentiment_signal"] = merged["sentiment_signal"].ffill(limit=1)
    merged["news_count"] = merged["news_count"].fillna(0)

    merged["daily_sentiment"] = merged["daily_sentiment"].fillna(0.0)
    merged["avg_conviction"] = merged["avg_conviction"].fillna(0.0)
    merged["sentiment_signal"] = merged["sentiment_signal"].fillna("NEUTRAL")

    merged = merged.drop_duplicates(subset=["Date"]).copy()
    return merged.set_index("Date")


def run_rule_based_backtest(
    merged_df: pd.DataFrame,
    stop_loss_pct: float = 0.02,
    conviction_threshold: float = 7.0,
    initial_capital: float = 10000.0
) -> pd.DataFrame:
    df = merged_df.copy()
    df["position"] = 0
    df["strategy_return"] = 0.0
    df["benchmark_return"] = df["Close"].pct_change().fillna(0)

    in_position = False
    entry_price = None
    positions = []

    for i in range(len(df)):
        row = df.iloc[i]

        close_price = float(row["Close"])
        ma50 = row["MA50"]
        sentiment = row["sentiment_signal"]
        conviction = float(row["avg_conviction"]) if pd.notna(row["avg_conviction"]) else 0.0
        daily_sentiment = float(row["daily_sentiment"]) if pd.notna(row["daily_sentiment"]) else 0.0

        if not in_position:
            buy_cond = (
                pd.notna(ma50) and
                close_price > float(ma50) and
                daily_sentiment > 0.0 and
                conviction >= conviction_threshold
            )

            if buy_cond:
                in_position = True
                entry_price = close_price
                positions.append(1)
            else:
                positions.append(0)

        else:
            stop_hit = close_price <= entry_price * (1 - stop_loss_pct)

            sell_cond = (
                (pd.notna(ma50) and close_price < float(ma50)) or
                (sentiment == "NEGATIVE") or
                stop_hit
            )

            if sell_cond:
                in_position = False
                entry_price = None
                positions.append(0)
            else:
                positions.append(1)

    df["position"] = positions

    df["strategy_return"] = (
        df["position"].shift(1).fillna(0) *
        df["Close"].pct_change().fillna(0)
    )

    df["strategy_equity"] = initial_capital * (1 + df["strategy_return"]).cumprod()
    df["benchmark_equity"] = initial_capital * (1 + df["benchmark_return"]).cumprod()

    return df


def calculate_metrics(equity_df: pd.DataFrame) -> dict:
    strat_ret = equity_df["strategy_return"]
    bench_ret = equity_df["benchmark_return"]

    def sharpe(r):
        std = r.std()
        if std == 0 or pd.isna(std):
            return 0.0
        return (r.mean() / std) * np.sqrt(252)

    def max_drawdown(equity):
        rolling_max = equity.cummax()
        dd = equity / rolling_max - 1
        return dd.min()

    metrics = {
        "Strategy Total Return": float(round((equity_df["strategy_equity"].iloc[-1] / equity_df["strategy_equity"].iloc[0] - 1) * 100, 2)),
        "Benchmark Total Return": float(round((equity_df["benchmark_equity"].iloc[-1] / equity_df["benchmark_equity"].iloc[0] - 1) * 100, 2)),
        "Strategy Sharpe": float(round(sharpe(strat_ret), 2)),
        "Benchmark Sharpe": float(round(sharpe(bench_ret), 2)),
        "Strategy Max Drawdown": float(round(max_drawdown(equity_df["strategy_equity"]) * 100, 2)),
        "Benchmark Max Drawdown": float(round(max_drawdown(equity_df["benchmark_equity"]) * 100, 2)),
    }
    return metrics


def plot_equity_curve(equity_df: pd.DataFrame, ticker: str, save_path: str = "equity_curve.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df["strategy_equity"], label=f"{ticker} Strategy")
    plt.plot(equity_df.index, equity_df["benchmark_equity"], label="Buy & Hold Benchmark")
    plt.title(f"Equity Curve: {ticker} Strategy vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def save_metrics_table(metrics: Dict, save_path: str = "final_metrics_table.csv") -> pd.DataFrame:
    metrics_table = pd.DataFrame({
        "Metric": [
            "Total Return (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)"
        ],
        "Strategy": [
            metrics["Strategy Total Return"],
            metrics["Strategy Sharpe"],
            metrics["Strategy Max Drawdown"]
        ],
        "Benchmark": [
            metrics["Benchmark Total Return"],
            metrics["Benchmark Sharpe"],
            metrics["Benchmark Max Drawdown"]
        ]
    })
    metrics_table.to_csv(save_path, index=False)
    return metrics_table


def main():
    ticker = Config.TICKER

    print(f"\n===== RUNNING PIPELINE FOR {ticker} =====\n")

    market_handler = MarketDataHandler()
    news_fetcher = NewsFetcher()
    sentiment_agent = SentimentAgent()

    price_df = market_handler.get_historical(ticker, period="1y", interval="1d")
    price_df = market_handler.add_indicators(price_df)

    news_items = news_fetcher.get_headlines(
        ticker=ticker,
        limit=Config.NEWS_LIMIT,
        lookback_hours=Config.LOOKBACK_HOURS
    )

    print(price_df.tail())
    print("Price DF shape:", price_df.shape)
    print("News fetched:", len(news_items))
    print("Sample news:", news_items[:2])

    sentiment_df = sentiment_agent.analyze_news(news_items)

    print("\nArticle-level sentiment sample:")
    print(sentiment_df[[
        "ticker",
        "title",
        "positive_score",
        "negative_score",
        "neutral_score",
        "sentiment",
        "model_confidence",
        "conviction_score"
    ]].head(10))

    sentiment_summary = sentiment_agent.summarize_sentiment(sentiment_df)
    print("\nRecent Sentiment Summary:")
    print(sentiment_summary)

    daily_sentiment_df = sentiment_agent.build_daily_sentiment(sentiment_df)
    print("\nDaily sentiment head:")
    print(daily_sentiment_df.head(10))
    print("\nDaily sentiment distribution:")
    print(daily_sentiment_df["sentiment_signal"].value_counts(dropna=False))

    live_result = run_live_decision_cycle(
        ticker=ticker,
        market_df=price_df,
        sentiment_summary=sentiment_summary,
        qty=1,
        stop_loss_pct=Config.STOP_LOSS_PCT,
        conviction_threshold=Config.CONVICTION_THRESHOLD,
        execute_orders=Config.EXECUTE_ORDERS
    )
    print("\nLive result:")
    print(live_result)

    merged_df = prepare_backtest_dataset(price_df, daily_sentiment_df)

    print("\nSentiment signal distribution in merged_df:")
    print(merged_df["sentiment_signal"].value_counts(dropna=False))

    equity_df = run_rule_based_backtest(
        merged_df,
        stop_loss_pct=Config.STOP_LOSS_PCT,
        conviction_threshold=Config.CONVICTION_THRESHOLD,
        initial_capital=Config.INITIAL_CAPITAL
    )

    metrics = calculate_metrics(equity_df)
    print("\nMetrics:")
    print(metrics)

    print("\nPosition changes:", int((equity_df["position"].diff().fillna(0) != 0).sum()))
    print("\nBacktest sample:")
    print(equity_df[["Close", "MA50", "sentiment_signal", "avg_conviction", "position"]].tail(30))

    print("\nArticle-level sentiment distribution:")
    print(sentiment_df["sentiment"].value_counts(dropna=False))

    print("\nDaily sentiment preview:")
    print(daily_sentiment_df.tail(20))

    metrics_table = save_metrics_table(metrics, save_path="final_metrics_table.csv")
    print("\nFinal Metrics Table:")
    print(metrics_table)

    plot_equity_curve(equity_df, ticker, save_path="equity_curve.png")
    print("\nSaved files: final_metrics_table.csv, equity_curve.png")


if __name__ == "__main__":
    main()