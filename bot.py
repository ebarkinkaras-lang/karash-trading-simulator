#!/usr/bin/env python3
"""
KARASH Trading Simulator — Bot
Her hafta içi GitHub Actions tarafından çalıştırılır.
slots.json okur → yfinance ile veri çeker → indikatör skoru hesaplar
→ AL/SAT/BEKLE karar verir → pozisyonları günceller → slots.json'a yazar → git commit atar.
"""

import json
import os
import sys
import logging
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from indicator_engine import compute_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("karash-bot")

SLOTS_PATH = Path("data/slots.json")
SLOTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─── Borsa → yfinance suffix mapping ──────────────────────────────────────────
EXCHANGE_CONFIG = {
    "BIST": {
        "suffix": ".IS",
        "currency": "TRY",
        "universe_file": "bist_tickers.txt",  # bist100 sembol listesi
        "fallback": ["THYAO.IS", "ASELS.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS",
                     "SAHOL.IS", "GARAN.IS", "AKBNK.IS", "YKBNK.IS", "BIMAS.IS",
                     "TUPRS.IS", "PGSUS.IS", "TCELL.IS", "ARCLK.IS", "TTKOM.IS",
                     "FROTO.IS", "VESTL.IS", "MGROS.IS", "TOASO.IS", "PETKM.IS"],
    },
    "NASDAQ": {
        "suffix": "",
        "currency": "USD",
        "fallback": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
                     "AVGO", "COST", "NFLX", "AMD", "ADBE", "ASML", "QCOM",
                     "INTC", "INTU", "TXN", "AMAT", "MU", "LRCX"],
    },
    "NYSE": {
        "suffix": "",
        "currency": "USD",
        "fallback": ["JPM", "V", "WMT", "JNJ", "XOM", "PG", "MA", "HD",
                     "CVX", "BAC", "KO", "PEP", "ABBV", "MRK", "T",
                     "DIS", "VZ", "BMY", "GE", "F"],
    },
    "Xetra": {
        "suffix": ".DE",
        "currency": "EUR",
        "fallback": ["SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "MBG.DE",
                     "BMW.DE", "BAYN.DE", "BAS.DE", "MUV2.DE", "ADS.DE",
                     "IFX.DE", "RWE.DE", "VOW3.DE", "DBK.DE", "HEN3.DE",
                     "EOAN.DE", "CON.DE", "FRE.DE", "LIN.DE", "DHER.DE"],
    },
    "LSE": {
        "suffix": ".L",
        "currency": "GBP",
        "fallback": ["SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L",
                     "RIO.L", "GSK.L", "REL.L", "NG.L", "LLOY.L",
                     "BARC.L", "BT-A.L", "VOD.L", "III.L", "RR.L",
                     "EXPN.L", "PRU.L", "LGEN.L", "STAN.L", "IMB.L"],
    },
}

# ─── İndikatör kodu → parametre override ──────────────────────────────────────
INDICATOR_PARAMS = {
    "MC-Hull": {"hull_period": 16},
    "RSI14": {"rsi_len": 14},
    "RSI7": {"rsi_len": 7},
    "MACD": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
    "EMA9": {"ema_fast": 9},
    "EMA21": {"ema_mid": 21},
    "EMA50": {"ema_slow": 50},
    "EMA200": {"ema_trend": 200},
    "BB20": {"bb_len": 20, "bb_mult": 2.0},
    "STOCH": {"stoch_k": 14, "stoch_d": 3, "stoch_smooth": 3},
    "ATR14": {"atr_len": 14},
    "DEMA": {"dema_fast_len": 12, "dema_slow_len": 26, "dema_sig_len": 9},
    "kNN": {"knn_k": 5, "knn_lookback": 100},
    "SuperTrend": {"st_factor": 3.0, "st_period": 10},
}


def load_slots() -> list:
    if SLOTS_PATH.exists():
        with open(SLOTS_PATH) as f:
            try:
                return json.load(f)
            except:
                return []
    return []


def save_slots(slots: list):
    with open(SLOTS_PATH, "w") as f:
        json.dump(slots, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"slots.json yazıldı ({len(slots)} slot)")


def get_indicator_params(indicator_codes: list) -> dict:
    """Seçilen indikatör kodlarından birleşik parametre dict'i üretir."""
    merged = {}
    for code in indicator_codes:
        if code and code in INDICATOR_PARAMS:
            merged.update(INDICATOR_PARAMS[code])
    return merged


def fetch_universe(exchange: str) -> list[str]:
    """Borsanın hisse evrenini döner."""
    cfg = EXCHANGE_CONFIG.get(exchange, EXCHANGE_CONFIG["NASDAQ"])
    universe_file = Path(cfg.get("universe_file", ""))
    if universe_file.exists():
        tickers = [t.strip() for t in universe_file.read_text().splitlines() if t.strip()]
        log.info(f"{exchange}: {len(tickers)} sembol universe dosyasından okundu")
        return tickers
    log.info(f"{exchange}: fallback listesi kullanılıyor ({len(cfg['fallback'])} sembol)")
    return cfg["fallback"]


def fetch_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """yfinance ile OHLCV çeker; daha dayanıklı sütun yönetimi ile."""
    try:
        # progress=False ve auto_adjust=True önemli
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        
        if df is None or len(df) < 220:
            return None

        # Multi-index başlıkları temizle (Bazen yfinance ('Close', 'THYAO.IS') şeklinde döner)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Tüm sütun isimlerini küçük harfe çevir
        df.columns = [str(c).lower() for c in df.columns]
        
        # Sadece ihtiyacımız olanları al
        required = ["open", "high", "low", "close", "volume"]
        # Eğer yfinance sütun isimlerini farklı verdiyse (örn: 'adj close') eşleştir
        if "close" not in df.columns and "adj close" in df.columns:
            df = df.rename(columns={"adj close": "close"})
            
        df = df[required].dropna()
        return df
    except Exception as e:
        log.error(f"{ticker} veri işleme hatası: {e}")
        return None


def score_ticker(ticker: str, params: dict) -> dict | None:
    df = fetch_ohlcv(ticker)
    if df is None:
        return None
    result = compute_score(df, params)
    result["ticker"] = ticker
    result["price"] = round(float(df["close"].iloc[-1]), 4)
    result["date"] = str(df.index[-1].date())
    return result


def decide_positions(scored: list[dict], capital: float, strong_thresh: float = 50) -> list[dict]:
    """
    Sinyal gücüne göre:
      score >= strong_thresh * 1.5  → Tier-A (güçlü al)
      score >= strong_thresh        → Tier-B (normal al)
    Max 5 hisse seçilir; güçlü sinyale daha fazla sermaye ayrılır.
    """
    buys = [s for s in scored if s["signal"] == "AL"]
    if not buys:
        return []

    buys.sort(key=lambda x: x["score"], reverse=True)
    buys = buys[:5]  # max 5

    weights = []
    for b in buys:
        w = 3.0 if b["score"] >= strong_thresh * 1.5 else 2.0 if b["score"] >= strong_thresh * 1.2 else 1.0
        weights.append(w)

    total_w = sum(weights)
    positions = []
    for i, b in enumerate(buys):
        alloc = capital * (weights[i] / total_w)
        qty = max(1, int(alloc / b["price"]))
        cost = qty * b["price"]
        positions.append({
            "ticker": b["ticker"],
            "score": b["score"],
            "signal": b["signal"],
            "price": b["price"],
            "qty": qty,
            "cost": round(cost, 4),
            "components": b.get("components", {}),
        })
    return positions


def run_slot(slot: dict) -> dict:
    """Tek bir slotu çalıştırır, güncellenmiş slot dict'i döner."""
    if not slot.get("active", False):
        return slot
    if slot.get("stopped", False):
        return slot

    exchange = slot.get("exchange", "NASDAQ")
    currency = EXCHANGE_CONFIG.get(exchange, {}).get("currency", "USD")
    indicators = slot.get("indicators", [])
    params = get_indicator_params(indicators)
    params["strong_thresh"] = slot.get("strong_thresh", 50)

    capital_total = slot.get("capital", 10000)
    today = str(date.today())

    # Mevcut açık pozisyonlar
    open_positions: list[dict] = slot.get("open_positions", [])
    trade_log: list[dict] = slot.get("trade_log", [])
    cash: float = slot.get("cash", capital_total)

    universe = fetch_universe(exchange)
    log.info(f"Slot '{slot['name']}' ({exchange}): {len(universe)} sembol taranıyor…")

    scored = []
    for ticker in universe:
        result = score_ticker(ticker, params)
        if result:
            scored.append(result)

    log.info(f"  {len(scored)} sembol başarıyla skorlandı")

    # ── SATIŞ: açık pozisyonları değerlendir ─────────────────────────────────
    updated_positions = []
    for pos in open_positions:
        result = next((s for s in scored if s["ticker"] == pos["ticker"]), None)
        if result is None:
            updated_positions.append(pos)
            continue
        current_price = result["price"]
        pnl_pct = (current_price - pos["buy_price"]) / pos["buy_price"] * 100

        # Sat: SAT sinyali VEYA %15 stop-loss VEYA %25 take-profit
        should_sell = (
            result["signal"] == "SAT"
            or pnl_pct <= -15
            or pnl_pct >= 25
        )
        if should_sell:
            proceeds = pos["qty"] * current_price
            cash += proceeds
            trade_log.append({
                "date": today,
                "action": "SAT",
                "ticker": pos["ticker"],
                "qty": pos["qty"],
                "price": current_price,
                "pnl_pct": round(pnl_pct, 2),
                "score": result["score"],
                "cash_after": round(cash, 2),
                "reason": "SAT sinyali" if result["signal"] == "SAT" else ("stop-loss" if pnl_pct <= -15 else "take-profit"),
            })
            log.info(f"  SAT: {pos['ticker']} × {pos['qty']} @ {current_price} ({pnl_pct:+.1f}%)")
        else:
            updated_positions.append(pos)

    # ── ALIŞ: yeni pozisyonlar ────────────────────────────────────────────────
    held_tickers = {p["ticker"] for p in updated_positions}
    candidates = [s for s in scored if s["signal"] == "AL" and s["ticker"] not in held_tickers]

    new_positions = decide_positions(candidates, cash * 0.9, params["strong_thresh"])  # %90 cash kullan

    for pos in new_positions:
        if pos["cost"] <= cash:
            cash -= pos["cost"]
            updated_positions.append({
                "ticker": pos["ticker"],
                "qty": pos["qty"],
                "buy_price": pos["price"],
                "buy_date": today,
                "score_at_buy": pos["score"],
            })
            trade_log.append({
                "date": today,
                "action": "AL",
                "ticker": pos["ticker"],
                "qty": pos["qty"],
                "price": pos["price"],
                "score": pos["score"],
                "cost": pos["cost"],
                "cash_after": round(cash, 2),
                "components": pos["components"],
            })
            log.info(f"  AL:  {pos['ticker']} × {pos['qty']} @ {pos['price']} (skor={pos['score']})")

    # ── Portföy değeri & kar/zarar ────────────────────────────────────────────
    ticker_prices = {s["ticker"]: s["price"] for s in scored}
    portfolio_value = cash + sum(
        p["qty"] * ticker_prices.get(p["ticker"], p["buy_price"])
        for p in updated_positions
    )
    total_pnl_pct = (portfolio_value - capital_total) / capital_total * 100

    slot.update({
        "cash": round(cash, 2),
        "portfolio_value": round(portfolio_value, 2),
        "pnl_pct": round(total_pnl_pct, 2),
        "currency": currency,
        "open_positions": updated_positions,
        "trade_log": trade_log[-500:],  # son 500 işlemi tut
        "last_run": today,
        "tickers_scanned": len(scored),
    })
    return slot


def main():
    slots = load_slots()
    if not slots:
        log.warning("slots.json boş veya yok. HTML arayüzünden slot oluştur.")
        sys.exit(0)

    updated = []
    for i, slot in enumerate(slots):
        log.info(f"─── Slot {i+1}/{len(slots)}: {slot.get('name', '?')} ───")
        try:
            updated_slot = run_slot(slot)
            updated.append(updated_slot)
        except Exception as e:
            log.error(f"Slot '{slot.get('name')}' hata: {e}", exc_info=True)
            updated.append(slot)

    save_slots(updated)
    log.info("✅ Tüm slotlar işlendi.")


if __name__ == "__main__":
    main()
