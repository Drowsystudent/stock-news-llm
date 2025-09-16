import requests
import pandas as pd
import yfinance as yf

# --- Fetch old news (GDELT) ---
def fetch_news(query, start="20100101000000", end="20151231000000", max_records=250):
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={query}&mode=ArtList&maxrecords={max_records}"
        f"&format=json&sort=DateAsc&startdatetime={start}&enddatetime={end}"
    )
    res = requests.get(url).json()
    if "articles" not in res:
        return pd.DataFrame()
    return pd.DataFrame(res["articles"])

# --- Fetch prices ---
def fetch_prices(ticker, start="2010-01-01", end="2015-01-01"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    return df.reset_index()

# --- Join news + prices ---
def label_articles(news, prices, ticker="NVDA", horizon=3):
    news["date"] = pd.to_datetime(news["seendate"].str[:8], format="%Y%m%d")
    prices["date"] = pd.to_datetime(prices["Date"])
    out = []
    for _, row in news.iterrows():
        pub_date = row["date"]
        p0 = prices[prices["date"] >= pub_date]
        if len(p0) == 0: continue
        price_at_pub = p0.iloc[0]["Close"]

        p_future = prices[prices["date"] >= pub_date + pd.Timedelta(days=horizon)]
        if len(p_future) == 0: continue
        price_after = p_future.iloc[0]["Close"]

        ret = (price_after / price_at_pub - 1)
        label = "UP" if ret > 0 else "DOWN"
        out.append({
            "ticker": ticker,
            "date": pub_date,
            "title": row.get("title",""),
            "url": row.get("url",""),
            "snippet": row.get("content",""),
            "return": ret,
            "label": label
        })
    return pd.DataFrame(out)

# --- Example run ---
if __name__ == "__main__":
    news = fetch_news("NVIDIA OR AMD", start="20100101000000", end="20101231000000")
    prices = fetch_prices("NVDA", start="2010-01-01", end="2011-01-01")
    dataset = label_articles(news, prices, ticker="NVDA")
    dataset.to_csv("nvda_news_labeled.csv", index=False)
    print("Saved dataset with", len(dataset), "rows")

