from typing import Dict, List

import pandas as pd
from transformers import pipeline


class SentimentAgent:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.classifier = pipeline(
            task="text-classification",
            model=model_name,
            tokenizer=model_name,
            top_k=None
        )

    @staticmethod
    def _extract_scores(raw_output) -> Dict[str, float]:
        if isinstance(raw_output, list) and len(raw_output) > 0:
            if isinstance(raw_output[0], list):
                raw_output = raw_output[0]

        score_dict = {"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0}

        for item in raw_output:
            label = str(item["label"]).strip().upper()
            score = float(item["score"])

            if label == "POSITIVE":
                score_dict["POSITIVE"] = score
            elif label == "NEGATIVE":
                score_dict["NEGATIVE"] = score
            elif label == "NEUTRAL":
                score_dict["NEUTRAL"] = score
            elif label == "LABEL_0":
                score_dict["NEGATIVE"] = score
            elif label == "LABEL_1":
                score_dict["NEUTRAL"] = score
            elif label == "LABEL_2":
                score_dict["POSITIVE"] = score

        return score_dict

    @staticmethod
    def _classify_sentiment(pos: float, neg: float, neu: float) -> tuple[str, float]:
        if pos >= neg and pos >= neu:
            return "POSITIVE", pos
        elif neg >= pos and neg >= neu:
            return "NEGATIVE", neg
        else:
            return "NEUTRAL", neu

    def analyze_news(self, news_items: List[Dict]) -> pd.DataFrame:
        rows = []

        for item in news_items:
            text = item.get("raw_text", "").strip()
            if not text:
                continue

            raw_output = self.classifier(text[:512])
            score_dict = self._extract_scores(raw_output)

            pos = score_dict["POSITIVE"]
            neg = score_dict["NEGATIVE"]
            neu = score_dict["NEUTRAL"]

            sentiment, confidence = self._classify_sentiment(pos, neg, neu)
            conviction = round(confidence * 10, 2)

            rows.append({
                "ticker": item.get("ticker"),
                "title": item.get("title"),
                "summary": item.get("summary"),
                "published_at": item.get("published_at"),
                "source": item.get("source"),
                "url": item.get("url"),
                "text": text,
                "positive_score": round(pos, 4),
                "negative_score": round(neg, 4),
                "neutral_score": round(neu, 4),
                "sentiment": sentiment,
                "model_confidence": round(confidence, 4),
                "conviction_score": conviction,
            })

        df = pd.DataFrame(rows)

        if not df.empty:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
            df = df.dropna(subset=["published_at"]).copy()

        return df

    def summarize_sentiment(self, sentiment_df: pd.DataFrame) -> Dict:
        if sentiment_df.empty:
            return {
                "overall_sentiment": "NEUTRAL",
                "avg_conviction": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "news_count": 0,
                "weighted_score": 0.0,
            }

        score_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        temp = sentiment_df.copy()
        temp["numeric_sentiment"] = temp["sentiment"].map(score_map).fillna(0)
        temp["weighted"] = temp["numeric_sentiment"] * temp["conviction_score"]

        conviction_sum = temp["conviction_score"].sum()
        weighted_score = 0.0 if conviction_sum == 0 else temp["weighted"].sum() / conviction_sum

        if weighted_score > 0.05:
            overall = "POSITIVE"
        elif weighted_score < -0.05:
            overall = "NEGATIVE"
        else:
            overall = "NEUTRAL"

        return {
            "overall_sentiment": overall,
            "avg_conviction": float(round(temp["conviction_score"].mean(), 2)),
            "positive_count": int((temp["sentiment"] == "POSITIVE").sum()),
            "negative_count": int((temp["sentiment"] == "NEGATIVE").sum()),
            "neutral_count": int((temp["sentiment"] == "NEUTRAL").sum()),
            "news_count": int(len(temp)),
            "weighted_score": float(round(weighted_score, 4)),
        }

    def build_daily_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        if sentiment_df.empty:
            return pd.DataFrame(columns=[
                "date",
                "daily_sentiment",
                "avg_conviction",
                "news_count",
                "sentiment_signal"
            ])

        score_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        temp = sentiment_df.copy()
        temp["date"] = temp["published_at"].dt.date
        temp["numeric_sentiment"] = temp["sentiment"].map(score_map).fillna(0)
        temp["weighted"] = temp["numeric_sentiment"] * temp["conviction_score"]

        grouped = temp.groupby("date").agg(
            weighted_sum=("weighted", "sum"),
            conviction_sum=("conviction_score", "sum"),
            avg_conviction=("conviction_score", "mean"),
            news_count=("title", "count"),
            positive_count=("sentiment", lambda s: int((s == "POSITIVE").sum())),
            negative_count=("sentiment", lambda s: int((s == "NEGATIVE").sum())),
            neutral_count=("sentiment", lambda s: int((s == "NEUTRAL").sum())),
        ).reset_index()

        grouped["daily_sentiment"] = grouped.apply(
            lambda row: row["weighted_sum"] / row["conviction_sum"] if row["conviction_sum"] != 0 else 0.0,
            axis=1
        )

        def map_daily_signal(row):
            if row["daily_sentiment"] > 0.05 and row["positive_count"] >= row["negative_count"]:
                return "POSITIVE"
            elif row["daily_sentiment"] < -0.05 and row["negative_count"] > row["positive_count"]:
                return "NEGATIVE"
            else:
                return "NEUTRAL"

        grouped["sentiment_signal"] = grouped.apply(map_daily_signal, axis=1)

        return grouped[[
            "date",
            "daily_sentiment",
            "avg_conviction",
            "news_count",
            "sentiment_signal"
        ]]