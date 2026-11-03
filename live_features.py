"""
live_features.py — v3 add-on for nse_analysis_api.py
================================================================================
Adds LIVE (market-hours-aware) data + new tools on top of your v2 API.

NEW ENDPOINTS
-------------
  GET  /market/status              open / pre_open / closed + IST time
  GET  /live/quotes?symbols=...    live price, change, day range, volume, sparkline
  GET  /live/indices               live Nifty, Bank Nifty, Sensex, IT, VIX
  GET  /live/stream?symbols=...    Server-Sent Events push (alternative to polling)
  GET  /live/analyze/{symbol}      live quote + full v2 analysis in ONE call
  GET  /intraday/{symbol}          1m/5m/15m candles + VWAP for live charts
  GET  /movers                     Nifty 50 gainers, losers, turnover leaders, breadth
  GET  /sectors                    sector heatmap (sector indices, live)
  POST /alerts/check               evaluate price / RSI / volume alert rules
  POST /portfolio/value            live P&L, day P&L, weights for holdings

ALSO: during market hours the daily-history cache shrinks from 5 min to 60 s,
so /analyze, /watchlist, /screener etc. also reflect the market as it moves.

HOW TO ATTACH (add at the very bottom of nse_analysis_api.py):
    import sys, live_features
    live_features.attach(sys.modules[__name__])

NOTE: Yahoo Finance data for NSE is free and can be delayed (often up to ~15 min).
Fine for a personal terminal; use a broker API (Kite / Upstox / Angel) for true tick data.
NOT INVESTMENT ADVICE.
"""

import os
import json
import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

IST = ZoneInfo("Asia/Kolkata")
API_KEY = os.getenv("API_KEY")
STALE_AFTER_SEC = 20 * 60  # flag quotes older than 20 min while market is open

M = None  # the main module, injected by attach()
router = APIRouter()


# ── AUTH (header OR ?api_key= so EventSource/SSE can authenticate too) ───

def _auth(x_api_key: Optional[str] = Header(default=None), api_key: Optional[str] = Query(default=None)):
    if API_KEY and (x_api_key or api_key) != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── MARKET STATUS ─────────────────────────────────────────────────────────

def market_status():
    now = datetime.now(IST)
    t, wd = now.time(), now.weekday()
    if wd >= 5:
        state, label = "closed", "Closed (weekend)"
    elif dtime(9, 0) <= t < dtime(9, 15):
        state, label = "pre_open", "Pre-open session"
    elif dtime(9, 15) <= t <= dtime(15, 30):
        state, label = "open", "Market open"
    else:
        state, label = "closed", "Closed"
    return {
        "state": state, "label": label,
        "ist_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Exchange holidays are not tracked; quotes flagged 'stale' help spot them.",
    }


@router.get("/market/status", dependencies=[Depends(_auth)])
def market_status_endpoint():
    return market_status()


# ── LIVE QUOTES ───────────────────────────────────────────────────────────

def _to_ist(index):
    return index.tz_convert(IST) if index.tz is not None else index.tz_localize(IST)


def _intraday(ticker, interval="1m", period="1d", ttl=20):
    key = f"intra:{ticker}:{interval}:{period}"

    def fetch():
        return yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)

    df = M.cached(key, ttl, fetch)
    if df is None or df.empty:
        with M._cache_lock:
            M._cache.pop(key, None)  # never cache failures
        return None
    return df


def live_quote(symbol):
    ticker = M.to_ticker(symbol)
    try:
        daily = M.get_history(ticker, period="1y", ttl=3600)["Close"]
        df = _intraday(ticker)
        sr = M.safe_round

        if df is not None:
            idx = _to_ist(df.index)
            last_ts = idx[-1]
            price = float(df["Close"].iloc[-1])
            same_day = daily.index[-1].date() == last_ts.date()
            prev_close = float(daily.iloc[-2]) if (same_day and len(daily) > 1) else float(daily.iloc[-1])
            closes = df["Close"].dropna()
            step = max(1, len(closes) // 40)
            spark = [round(float(x), 2) for x in closes.iloc[::step]]
            if spark and spark[-1] != round(price, 2):
                spark.append(round(price, 2))
            day_open, day_high, day_low = float(df["Open"].iloc[0]), float(df["High"].max()), float(df["Low"].min())
            volume = int(df["Volume"].sum())
        else:  # fallback to daily bars (e.g. intraday feed unavailable)
            last_ts = _to_ist(daily.index)[-1]
            price = float(daily.iloc[-1])
            prev_close = float(daily.iloc[-2]) if len(daily) > 1 else price
            spark, day_open, day_high, day_low, volume = [], None, None, None, None

        change = price - prev_close
        change_pct = change / prev_close * 100 if prev_close else None
        status = market_status()
        age = (datetime.now(IST) - last_ts.to_pydatetime()).total_seconds()
        return {
            "symbol": ticker,
            "price": sr(price), "prev_close": sr(prev_close),
            "change": sr(change), "change_pct": sr(change_pct),
            "direction": "up" if change > 0 else "down" if change < 0 else "flat",
            "open": sr(day_open), "day_high": sr(day_high), "day_low": sr(day_low),
            "volume": volume,
            "sparkline": spark,
            "last_update": last_ts.isoformat(),
            "stale": bool(status["state"] == "open" and age > STALE_AFTER_SEC),
            "market_state": status["state"],
        }
    except HTTPException as e:
        return {"symbol": ticker, "error": e.detail}
    except Exception as e:
        return {"symbol": ticker, "error": str(e)}


def _quotes(symbols):
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(live_quote, symbols))


def _parse_symbols(symbols, limit):
    out = [s.strip() for s in symbols.split(",") if s.strip()][:limit]
    if not out:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")
    return out


@router.get("/live/quotes", dependencies=[Depends(_auth)])
def live_quotes(symbols: str):
    """/live/quotes?symbols=RELIANCE,TCS,INFY (max 30). Poll every 10-30 s."""
    syms = _parse_symbols(symbols, 30)
    return {"server_time": datetime.now(IST).isoformat(), "market": market_status(), "quotes": _quotes(syms)}


@router.get("/live/indices", dependencies=[Depends(_auth)])
def live_indices():
    names = list(M.INDICES.items())
    quotes = _quotes([t for _, t in names])
    rows = [{"name": n, **q} for (n, _), q in zip(names, quotes)]
    return {"server_time": datetime.now(IST).isoformat(), "market": market_status(), "indices": rows}


@router.get("/live/stream", dependencies=[Depends(_auth)])
def live_stream(symbols: str, interval: int = 10):
    """
    Server-Sent Events. Browser:
        const es = new EventSource(`${API}/live/stream?symbols=TCS,INFY&api_key=KEY`);
        es.addEventListener('quotes', e => setData(JSON.parse(e.data)));
    Stream ends after 10 min; EventSource reconnects automatically.
    """
    syms = _parse_symbols(symbols, 15)
    interval = min(max(interval, 5), 60)

    def gen():
        end = time.time() + 600
        yield "retry: 3000\n\n"
        while time.time() < end:
            payload = {"market": market_status(), "quotes": _quotes(syms)}
            yield f"event: quotes\ndata: {json.dumps(payload)}\n\n"
            time.sleep(interval)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/live/analyze/{symbol}", dependencies=[Depends(_auth)])
def live_analyze(symbol: str):
    """Everything a stock-detail page needs in one request."""
    return {
        "live": live_quote(symbol),
        "market": market_status(),
        "analysis": M.analyze_stock(symbol),
    }


# ── INTRADAY CANDLES + VWAP ───────────────────────────────────────────────

@router.get("/intraday/{symbol}", dependencies=[Depends(_auth)])
def intraday(symbol: str, interval: str = "5m", days: int = 1):
    """Intraday OHLCV + VWAP. interval: 1m,2m,5m,15m,30m,60m. days: 1-5 (1m/2m) or 1-30."""
    if interval not in ("1m", "2m", "5m", "15m", "30m", "60m"):
        raise HTTPException(status_code=400, detail="interval must be one of 1m,2m,5m,15m,30m,60m")
    days = min(max(days, 1), 5 if interval in ("1m", "2m") else 30)
    ticker = M.to_ticker(symbol)
    df = _intraday(ticker, interval, f"{days}d", ttl=30)
    if df is None:
        raise HTTPException(status_code=404, detail=f"No intraday data for {ticker}.")

    idx = _to_ist(df.index)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day_key = np.array(idx.date)
    cum_pv = (tp * df["Volume"]).groupby(day_key).cumsum()
    cum_v = df["Volume"].groupby(day_key).cumsum().replace(0, np.nan)
    vwap = cum_pv / cum_v

    rows = []
    for i, ts in enumerate(idx):
        r = df.iloc[i]
        rows.append({
            "time": int(ts.timestamp()), "datetime": ts.isoformat(),
            "open": M.safe_round(r["Open"]), "high": M.safe_round(r["High"]),
            "low": M.safe_round(r["Low"]), "close": M.safe_round(r["Close"]),
            "volume": int(r["Volume"]) if not pd.isna(r["Volume"]) else None,
            "vwap": M.safe_round(vwap.iloc[i]),
        })
    return {"symbol": ticker, "interval": interval, "days": days, "market": market_status(), "candles": rows}


# ── MOVERS & BREADTH ──────────────────────────────────────────────────────

def _movers_data():
    quotes = [q for q in _quotes(M.NIFTY50) if q and "error" not in q and q.get("change_pct") is not None]
    for q in quotes:
        q["turnover_cr"] = M.safe_round((q["price"] or 0) * (q["volume"] or 0) / 1e7)

    def slim(q):
        return {k: q.get(k) for k in ("symbol", "price", "change_pct", "volume", "turnover_cr", "direction")}

    by_chg = sorted(quotes, key=lambda q: q["change_pct"], reverse=True)
    by_turn = sorted(quotes, key=lambda q: q["turnover_cr"] or 0, reverse=True)
    adv = sum(1 for q in quotes if q["change_pct"] > 0)
    dec = sum(1 for q in quotes if q["change_pct"] < 0)
    breadth = None
    if quotes:
        breadth = "Broad strength" if adv / len(quotes) >= 0.7 else \
                  "Broad weakness" if dec / len(quotes) >= 0.7 else "Mixed"
    return {
        "universe": "NIFTY 50", "counted": len(quotes),
        "breadth": {"advances": adv, "declines": dec, "unchanged": len(quotes) - adv - dec, "read": breadth},
        "top_gainers": [slim(q) for q in by_chg[:5]],
        "top_losers": [slim(q) for q in by_chg[::-1][:5]],
        "turnover_leaders": [slim(q) for q in by_turn[:5]],
    }


@router.get("/movers", dependencies=[Depends(_auth)])
def movers():
    """Top gainers / losers / turnover leaders + advance-decline. Cached 60 s (first call ~15-30 s)."""
    data = M.cached("movers", 60, _movers_data)
    return {"server_time": datetime.now(IST).isoformat(), "market": market_status(), **data}


# ── SECTOR HEATMAP ────────────────────────────────────────────────────────

SECTORS = {
    "Bank": "^NSEBANK", "IT": "^CNXIT", "Auto": "^CNXAUTO", "Pharma": "^CNXPHARMA",
    "FMCG": "^CNXFMCG", "Metal": "^CNXMETAL", "Energy": "^CNXENERGY", "Realty": "^CNXREALTY",
    "Financial Services": "^CNXFIN", "Media": "^CNXMEDIA", "PSU Bank": "^CNXPSUBANK",
}


@router.get("/sectors", dependencies=[Depends(_auth)])
def sectors():
    """Sector performance, strongest first. Some Yahoo sector tickers may be missing; they return an error row."""
    def build():
        names = list(SECTORS.items())
        qs = _quotes([t for _, t in names])
        rows = [{"sector": n, "ticker": t, "price": q.get("price"), "change_pct": q.get("change_pct"),
                 "error": q.get("error")} for (n, t), q in zip(names, qs)]
        ok = sorted([r for r in rows if r["change_pct"] is not None], key=lambda r: r["change_pct"], reverse=True)
        bad = [r for r in rows if r["change_pct"] is None]
        return ok + bad

    rows = M.cached("sectors", 60, build)
    return {"server_time": datetime.now(IST).isoformat(), "market": market_status(), "sectors": rows}


# ── ALERTS (stateless: your frontend stores the rules) ───────────────────

ALERT_TYPES = {"price_above", "price_below", "day_change_above", "day_change_below",
               "rsi_above", "rsi_below", "volume_spike"}


class AlertRule(BaseModel):
    id: Optional[str] = None
    symbol: str
    type: str
    value: Optional[float] = None


class AlertRequest(BaseModel):
    alerts: List[AlertRule]


@router.post("/alerts/check", dependencies=[Depends(_auth)])
def alerts_check(req: AlertRequest):
    """
    Body: {"alerts":[{"id":"a1","symbol":"TCS","type":"price_below","value":3500}]}
    types: price_above, price_below, day_change_above, day_change_below,
           rsi_above, rsi_below, volume_spike (value not needed)
    Returns each rule with triggered true/false and the current value.
    """
    rules = req.alerts[:50]
    if not rules:
        raise HTTPException(status_code=400, detail="Send at least one alert rule")

    tickers = sorted({M.to_ticker(r.symbol) for r in rules})
    quotes = {q["symbol"]: q for q in _quotes(tickers)}
    tech_cache = {}

    def tech(ticker):
        if ticker not in tech_cache:
            t, _, _ = M.get_technical_snapshot(M.get_history(ticker))
            tech_cache[ticker] = t
        return tech_cache[ticker]

    results = []
    for r in rules:
        ticker = M.to_ticker(r.symbol)
        q = quotes.get(ticker, {})
        base = {"id": r.id, "symbol": ticker, "type": r.type, "value": r.value}
        if r.type not in ALERT_TYPES:
            results.append({**base, "error": f"type must be one of {sorted(ALERT_TYPES)}"})
            continue
        if "error" in q:
            results.append({**base, "error": q["error"]})
            continue
        if r.type != "volume_spike" and r.value is None:
            results.append({**base, "error": "value is required for this alert type"})
            continue
        try:
            if r.type == "price_above":
                cur = q["price"]; hit = cur is not None and cur >= r.value
            elif r.type == "price_below":
                cur = q["price"]; hit = cur is not None and cur <= r.value
            elif r.type == "day_change_above":
                cur = q["change_pct"]; hit = cur is not None and cur >= r.value
            elif r.type == "day_change_below":
                cur = q["change_pct"]; hit = cur is not None and cur <= r.value
            elif r.type == "rsi_above":
                cur = tech(ticker)["rsi_14"]; hit = cur is not None and cur >= r.value
            elif r.type == "rsi_below":
                cur = tech(ticker)["rsi_14"]; hit = cur is not None and cur <= r.value
            else:  # volume_spike
                cur = tech(ticker)["volume_ratio_20d"]; hit = bool(tech(ticker)["volume_spike"])
            results.append({**base, "current": cur, "triggered": bool(hit)})
        except Exception as e:
            results.append({**base, "error": str(getattr(e, "detail", e))})

    return {"checked_at": datetime.now(IST).isoformat(), "market": market_status(),
            "triggered_count": sum(1 for x in results if x.get("triggered")), "results": results}


# ── PORTFOLIO ─────────────────────────────────────────────────────────────

class Holding(BaseModel):
    symbol: str
    qty: float
    avg_price: float


class PortfolioRequest(BaseModel):
    holdings: List[Holding]


@router.post("/portfolio/value", dependencies=[Depends(_auth)])
def portfolio_value(req: PortfolioRequest):
    """Body: {"holdings":[{"symbol":"TCS","qty":10,"avg_price":3400}]} -> live P&L. Max 50 holdings."""
    hs = req.holdings[:50]
    if not hs:
        raise HTTPException(status_code=400, detail="Send at least one holding")
    sr = M.safe_round
    tickers = sorted({M.to_ticker(h.symbol) for h in hs})
    quotes = {q["symbol"]: q for q in _quotes(tickers)}

    rows, tot_inv, tot_val, tot_day = [], 0.0, 0.0, 0.0
    for h in hs:
        ticker = M.to_ticker(h.symbol)
        q = quotes.get(ticker, {})
        if "error" in q or q.get("price") is None:
            rows.append({"symbol": ticker, "qty": h.qty, "avg_price": h.avg_price,
                         "error": q.get("error", "no price")})
            continue
        invested = h.qty * h.avg_price
        value = h.qty * q["price"]
        day_pnl = h.qty * (q["price"] - q["prev_close"])
        tot_inv += invested; tot_val += value; tot_day += day_pnl
        rows.append({
            "symbol": ticker, "qty": h.qty, "avg_price": h.avg_price,
            "price": q["price"], "day_change_pct": q["change_pct"],
            "invested": sr(invested), "value": sr(value),
            "pnl": sr(value - invested), "pnl_pct": sr((value / invested - 1) * 100) if invested else None,
            "day_pnl": sr(day_pnl),
        })

    ok = [r for r in rows if "error" not in r]
    for r in ok:
        r["weight_pct"] = sr(r["value"] / tot_val * 100) if tot_val else None
    ok_sorted = sorted(ok, key=lambda r: r["pnl_pct"] if r["pnl_pct"] is not None else 0)
    top = max(ok, key=lambda r: r["weight_pct"] or 0, default=None)

    return {
        "as_of": datetime.now(IST).isoformat(), "market": market_status(),
        "totals": {
            "invested": sr(tot_inv), "current_value": sr(tot_val),
            "pnl": sr(tot_val - tot_inv),
            "pnl_pct": sr((tot_val / tot_inv - 1) * 100) if tot_inv else None,
            "day_pnl": sr(tot_day),
            "day_pnl_pct": sr(tot_day / (tot_val - tot_day) * 100) if (tot_val - tot_day) else None,
        },
        "best_performer": ok_sorted[-1]["symbol"] if ok_sorted else None,
        "worst_performer": ok_sorted[0]["symbol"] if ok_sorted else None,
        "concentration_warning": (f"{top['symbol']} is {top['weight_pct']}% of the portfolio"
                                  if top and (top["weight_pct"] or 0) > 30 else None),
        "holdings": rows,
    }


# ── ATTACH ────────────────────────────────────────────────────────────────

def attach(module):
    """Call once at the bottom of nse_analysis_api.py."""
    global M
    M = module

    original_get_history = module.get_history

    def live_aware_get_history(ticker, period="2y", ttl=300):
        # default 5-min cache shrinks to 60 s while the market is open
        if ttl == 300 and market_status()["state"] in ("open", "pre_open"):
            ttl = 60
        return original_get_history(ticker, period, ttl)

    module.get_history = live_aware_get_history
    module.app.version = "3.0"
    module.app.include_router(router)
