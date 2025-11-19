import os
import json
import time
import hmac
import base64
import hashlib
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional

import requests
import gspread
from google.oauth2.service_account import Credentials
import alpaca_trade_api as tradeapi


# =========================
# Config / constants
# =========================

SPREADSHEET_NAME = "Active-Investing"
WORKSHEET_NAME = "Automation-Screener"

# Column indices (0-based) based on your description:
COL_SYMBOL = 2      # Column C
COL_IS_CRYPTO = 3   # Column D (TRUE => Kraken, FALSE => Alpaca)
COL_SIGNAL = 18     # Column S
COL_ICON = 22       # Column W

BASE_ORDER_FRACTION = float(os.getenv("BASE_ORDER_FRACTION", "0.05"))
MIN_SIGNAL_MULTIPLIER = float(os.getenv("MIN_SIGNAL_MULTIPLIER", "0.25"))
MIN_ORDER_DOLLARS = float(os.getenv("MIN_ORDER_DOLLARS", "1.0"))

ALPACA_MIN_ORDER_NOTIONAL = float(os.getenv("ALPACA_MIN_ORDER_NOTIONAL", "1.0"))
KRAKEN_MIN_ORDER_NOTIONAL = float(os.getenv("KRAKEN_MIN_ORDER_NOTIONAL", "5.0"))

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET")
KRAKEN_API_BASE_URL = os.getenv("KRAKEN_API_BASE_URL", "https://api.kraken.com")


# Map sheet symbols to Kraken pairs.
# ⚠️ You MUST adjust this to match how you write crypto tickers in your sheet.
KRAKEN_PAIR_MAP = {
    "BTC-USD": "XBTUSD",
    "ETH-USD": "ETHUSD",
    "SOL-USD": "SOLUSD",
    # add more as needed...
}


@dataclass
class TradeSignal:
    symbol: str
    is_crypto: bool  # True => route to Kraken, False => Alpaca
    signal_pct: float  # between 0 and -100
    row_index: int     # for logging/debugging


# =========================
# Google Sheet client
# =========================

def get_gspread_client() -> gspread.Client:
    svc_json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not svc_json_str:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON env var is required")

    svc_info = json.loads(svc_json_str)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    credentials = Credentials.from_service_account_info(svc_info, scopes=scopes)
    return gspread.authorize(credentials)


def read_signals_from_sheet() -> List[TradeSignal]:
    client = get_gspread_client()
    sheet = client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)
    all_values = sheet.get_all_values()

    if not all_values or len(all_values) < 2:
        return []

    header = all_values[0]
    data_rows = all_values[1:]

    signals: List[TradeSignal] = []

    for idx, row in enumerate(data_rows, start=2):  # row 2 in sheet is first data row
        # Be defensive about row length
        if len(row) <= max(COL_SYMBOL, COL_IS_CRYPTO, COL_SIGNAL, COL_ICON):
            continue

        icon = (row[COL_ICON] or "").strip()
        if "🟢" not in icon:
            continue

        symbol = (row[COL_SYMBOL] or "").strip().upper()
        if not symbol:
            continue

        is_crypto_str = (row[COL_IS_CRYPTO] or "").strip().upper()
        is_crypto = is_crypto_str == "TRUE"

        signal_str = (row[COL_SIGNAL] or "").strip()
        if not signal_str:
            continue

        try:
            signal_pct = float(signal_str)
        except ValueError:
            print(f"Row {idx}: invalid signal value '{signal_str}', skipping.")
            continue

        signals.append(TradeSignal(symbol=symbol, is_crypto=is_crypto,
                                   signal_pct=signal_pct, row_index=idx))

    return signals


# =========================
# Order sizing
# =========================

def compute_order_notional(available_funds: float, signal_pct: float) -> float:
    """
    - Start from BASE_ORDER_FRACTION of available_funds.
    - signal_pct in [0, -100]:
        closer to 0  -> smaller orders
        closer to -100 -> larger orders
    - We keep a floor multiplier so weak signals don't shrink into dust.
    """
    if available_funds <= 0:
        return 0.0

    base_value = available_funds * BASE_ORDER_FRACTION

    # Clamp signal into [-100, 0]
    clamped = max(-100.0, min(0.0, signal_pct))
    strength = -clamped / 100.0  # 0.0 .. 1.0

    # Interpolate between floor and 1
    multiplier = MIN_SIGNAL_MULTIPLIER + (1.0 - MIN_SIGNAL_MULTIPLIER) * strength

    notional = base_value * multiplier

    # Keep from constantly producing silly tiny sizes
    if notional < MIN_ORDER_DOLLARS:
        return 0.0

    return notional


# =========================
# Alpaca broker
# =========================

class AlpacaBroker:
    def __init__(self):
        if not (ALPACA_API_KEY and ALPACA_API_SECRET):
            raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET must be set")

        self.api = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            ALPACA_BASE_URL,
            api_version="v2",
        )
        self.min_notional = ALPACA_MIN_ORDER_NOTIONAL

    def is_market_open(self) -> bool:
        """
        Returns True if the US stock market is currently open
        according to Alpaca's market clock.
        """
        try:
            clock = self.api.get_clock()
            return bool(getattr(clock, "is_open", False))
        except Exception as e:
            print(f"Alpaca: failed to fetch market clock: {e}")
            # Be conservative: if we can't tell, don't trade.
            return False

    def get_available_funds(self) -> float:
        account = self.api.get_account()
        # You could also use account.cash; buying_power includes leverage if enabled
        return float(account.cash)

    def has_position(self, symbol: str) -> bool:
        try:
            pos = self.api.get_position(symbol)
            qty = float(pos.qty)
            return qty > 0
        except Exception as e:
            # Alpaca throws an error if no position exists
            if "position does not exist" in str(e).lower():
                return False
            raise

    def has_open_buy_order(self, symbol: str) -> bool:
        orders = self.api.list_orders(
            status="open",
            direction="desc",
            nested=False,
        )
        for o in orders:
            if o.symbol.upper() == symbol.upper() and o.side.lower() == "buy":
                return True
        return False

    def get_best_bid(self, symbol: str) -> Optional[float]:
        try:
            quote = self.api.get_latest_quote(symbol)
            bid = float(quote.bp)
            if bid > 0:
                return bid
        except Exception as e:
            print(f"Alpaca: failed to fetch best bid for {symbol}: {e}")
        return None

    def place_order(self, symbol: str, notional: float) -> None:
        if notional < self.min_notional:
            print(f"Alpaca: notional ${notional:.2f} below Alpaca min ${self.min_notional:.2f}, skipping {symbol}")
            return

        best_bid = self.get_best_bid(symbol)

        if best_bid is not None:
            qty = round(notional / best_bid, 3)  # fractional shares
            if qty * best_bid < self.min_notional:
                print(f"Alpaca: qty rounding makes order below min notional, skipping {symbol}")
                return

            print(f"Alpaca: placing LIMIT buy {symbol}, qty={qty}, limit={best_bid}")
            self.api.submit_order(
                symbol=symbol,
                qty=str(qty),
                side="buy",
                type="limit",
                time_in_force="day",
                limit_price=str(best_bid),
            )
        else:
            # Fallback: market notional order (no best-bid quote available)
            print(f"Alpaca: no best bid, placing MARKET notional ${notional:.2f} for {symbol}")
            self.api.submit_order(
                symbol=symbol,
                notional=str(notional),
                side="buy",
                type="market",
                time_in_force="day",
            )


# =========================
# Kraken broker
# =========================

class KrakenBroker:
    def __init__(self):
        if not (KRAKEN_API_KEY and KRAKEN_API_SECRET):
            raise RuntimeError("KRAKEN_API_KEY and KRAKEN_API_SECRET must be set")
        self.base_url = KRAKEN_API_BASE_URL.rstrip("/")
        self.api_key = KRAKEN_API_KEY
        self.api_secret = KRAKEN_API_SECRET
        self.min_notional = KRAKEN_MIN_ORDER_NOTIONAL

    def _sign(self, url_path: str, data: dict) -> str:
        # Based on Kraken’s official Python example
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + postdata).encode()
        message = url_path.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()

    def _private(self, method: str, data: Optional[dict] = None) -> dict:
        if data is None:
            data = {}
        data["nonce"] = int(time.time() * 1000)
        url_path = f"/0/private/{method}"
        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._sign(url_path, data),
        }
        url = self.base_url + url_path
        resp = requests.post(url, headers=headers, data=data, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if result.get("error"):
            raise RuntimeError(f"Kraken error {method}: {result['error']}")
        return result["result"]

    def _public(self, method: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}/0/public/{method}"
        resp = requests.get(url, params=params or {}, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if result.get("error"):
            raise RuntimeError(f"Kraken public error {method}: {result['error']}")
        return result["result"]

    def symbol_to_pair(self, symbol: str) -> Optional[str]:
        return KRAKEN_PAIR_MAP.get(symbol.upper())

    def get_available_funds(self) -> float:
        balances = self._private("Balance")
        # Most accounts hold USD as ZUSD
        usd = balances.get("ZUSD") or balances.get("USD")
        if usd is None:
            return 0.0
        return float(usd)

    def has_open_buy_order(self, pair: str) -> bool:
        open_orders = self._private("OpenOrders")
        open_dict = open_orders.get("open", {})
        for _, order in open_dict.items():
            desc = order.get("descr", {})
            if desc.get("pair") == pair and desc.get("type") == "buy":
                return True
        return False

    def get_best_bid(self, pair: str) -> Optional[float]:
        ticker = self._public("Ticker", {"pair": pair})
        # ticker result key may be the canonical pair name
        k = next(iter(ticker.keys()))
        bid_str = ticker[k]["b"][0]  # 'b' = [best bid, whole lot volume, lot volume]
        try:
            bid = float(bid_str)
            if bid > 0:
                return bid
        except ValueError:
            pass
        return None

    def place_order(self, symbol: str, notional: float) -> None:
        pair = self.symbol_to_pair(symbol)
        if not pair:
            print(f"Kraken: no pair mapping for symbol {symbol}, skipping.")
            return

        if notional < self.min_notional:
            print(f"Kraken: notional ${notional:.2f} below Kraken min ${self.min_notional:.2f}, skipping {symbol}")
            return

        best_bid = self.get_best_bid(pair)
        if best_bid is None:
            print(f"Kraken: could not fetch best bid for {pair}, skipping order for {symbol}")
            return

        volume = notional / best_bid
        volume = round(volume, 8)  # crypto precision

        if volume * best_bid < self.min_notional:
            print(f"Kraken: volume rounding makes order below min notional, skipping {symbol}")
            return

        data = {
            "pair": pair,
            "type": "buy",
            "ordertype": "limit",   # best-bid limit
            "price": str(best_bid),
            "volume": str(volume),
            # "timeinforce": "GTC",  # Kraken uses GTC/IOC/GTD. No pure "day" TIF.
        }

        print(f"Kraken: placing LIMIT buy {symbol} ({pair}), vol={volume}, limit={best_bid}")
        self._private("AddOrder", data=data)


# =========================
# Main orchestration
# =========================

def process_signal(signal: TradeSignal, alpaca: AlpacaBroker, kraken: KrakenBroker) -> None:
    venue = "Kraken" if signal.is_crypto else "Alpaca"

    if signal.is_crypto:
        broker = kraken
        symbol_for_venue = signal.symbol
    else:
        broker = alpaca
        symbol_for_venue = signal.symbol

    print(f"Row {signal.row_index}: {symbol_for_venue} | venue={venue} | signal={signal.signal_pct}")

    # For Alpaca (stocks): only trade when the regular market is open
    if not signal.is_crypto:
        if not alpaca.is_market_open():
            print("  Alpaca: market is closed, skipping stock order.")
            return

    # Check for duplicate unfilled orders first
    if signal.is_crypto:
        pair = kraken.symbol_to_pair(symbol_for_venue)
        if not pair:
            print(f"  No Kraken pair mapping for {symbol_for_venue}, skipping.")
            return
        if kraken.has_open_buy_order(pair):
            print(f"  Kraken: open BUY order already exists for {pair}, skipping.")
            return
    else:
        if alpaca.has_open_buy_order(symbol_for_venue):
            print(f"  Alpaca: open BUY order already exists for {symbol_for_venue}, skipping.")
            return
        # Also check positions so we don't double up the asset
        if alpaca.has_position(symbol_for_venue):
            print(f"  Alpaca: position already exists for {symbol_for_venue}, skipping.")
            return

    # Determine available funds on the selected venue
    available_funds = broker.get_available_funds()
    print(f"  {venue} available funds: ${available_funds:.2f}")

    notional = compute_order_notional(available_funds, signal.signal_pct)
    if notional <= 0:
        print(f"  Computed notional is ${notional:.2f}, skipping (too small / zero).")
        return

    print(f"  Computed notional for {symbol_for_venue}: ${notional:.2f}")

    # Final venue-specific min-check and order placement
    broker.place_order(symbol_for_venue, notional)


def main():
    print("=== Automated Buyer Bot run starting ===")
    signals = read_signals_from_sheet()
    if not signals:
        print("No signals found with 🟢 icon. Exiting.")
        return

    print(f"Found {len(signals)} signals with 🟢")

    alpaca = AlpacaBroker()
    kraken = KrakenBroker()

    for signal in signals:
        try:
            process_signal(signal, alpaca, kraken)
        except Exception as e:
            print(f"Error processing row {signal.row_index} ({signal.symbol}): {e}")

    print("=== Automated Buyer Bot run complete ===")


if __name__ == "__main__":
    main()
