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
# Used as overrides for weird/legacy names; otherwise we auto-convert symbols.
KRAKEN_PAIR_MAP = {
    # Common special cases
    "BTC-USD": "XBTUSD",
    "BTC/USD": "XBTUSD",
    "ETH-USD": "ETHUSD",
    "ETH/USD": "ETHUSD",
    "SOL-USD": "SOLUSD",
    "SOL/USD": "SOLUSD",
    # Add more explicit overrides as needed...
}

# =========================
# Debug helpers
# =========================

DEBUG = os.getenv("DEBUG", "1") == "1"
VERBOSE_DEBUG = os.getenv("VERBOSE_DEBUG", "0") == "1"


def dlog(msg: str) -> None:
    """High-level debug logging (can be disabled with DEBUG=0)."""
    if DEBUG:
        print(msg)


def vlog(msg: str) -> None:
    """Extra noisy debug logging (enable with VERBOSE_DEBUG=1)."""
    if VERBOSE_DEBUG:
        print(msg)


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
    dlog("=== [DEBUG] Reading signals from Google Sheet ===")
    client = get_gspread_client()
    sheet = client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)
    all_values = sheet.get_all_values()

    if not all_values or len(all_values) < 2:
        dlog("[DEBUG] Sheet is empty or has no data rows.")
        return []

    header = all_values[0]
    data_rows = all_values[1:]
    dlog(f"[DEBUG] Header row: {header}")
    dlog(f"[DEBUG] Number of data rows: {len(data_rows)}")

    signals: List[TradeSignal] = []

    for idx, row in enumerate(data_rows, start=2):  # row 2 in sheet is first data row
        # Be defensive about row length
        if len(row) <= max(COL_SYMBOL, COL_IS_CRYPTO, COL_SIGNAL, COL_ICON):
            vlog(f"[DEBUG] Row {idx} skipped: too short (len={len(row)}). Raw row: {row}")
            continue

        icon = (row[COL_ICON] or "").strip()
        symbol_raw = (row[COL_SYMBOL] or "").strip()
        is_crypto_str = (row[COL_IS_CRYPTO] or "").strip()
        signal_str = (row[COL_SIGNAL] or "").strip()

        vlog(
            f"[DEBUG] Row {idx} raw -> icon='{icon}', symbol='{symbol_raw}', "
            f"is_crypto_col='{is_crypto_str}', signal='{signal_str}'"
        )

        if "🟢" not in icon:
            vlog(f"[DEBUG] Row {idx} skipped: no 🟢 icon.")
            continue

        symbol = symbol_raw.upper()
        if not symbol:
            vlog(f"[DEBUG] Row {idx} skipped: empty symbol.")
            continue

        is_crypto = is_crypto_str.strip().upper() == "TRUE"

        if not signal_str:
            vlog(f"[DEBUG] Row {idx} skipped: empty signal cell.")
            continue

        try:
            signal_pct = float(signal_str)
        except ValueError:
            vlog(f"[DEBUG] Row {idx} skipped: invalid signal value '{signal_str}'.")
            continue

        dlog(
            f"[DEBUG] Row {idx} accepted as signal: "
            f"symbol={symbol}, is_crypto={is_crypto}, signal_pct={signal_pct}"
        )

        signals.append(
            TradeSignal(
                symbol=symbol,
                is_crypto=is_crypto,
                signal_pct=signal_pct,
                row_index=idx,
            )
        )

    dlog(f"[DEBUG] Total signals loaded: {len(signals)}")
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
    dlog(
        f"[DEBUG] compute_order_notional: available_funds={available_funds:.2f}, "
        f"signal_pct={signal_pct}"
    )

    if available_funds <= 0:
        dlog("[DEBUG] available_funds <= 0, returning 0.0")
        return 0.0

    base_value = available_funds * BASE_ORDER_FRACTION

    # Clamp signal into [-100, 0]
    clamped = max(-100.0, min(0.0, signal_pct))
    strength = -clamped / 100.0  # 0.0 .. 1.0

    # Interpolate between floor and 1
    multiplier = MIN_SIGNAL_MULTIPLIER + (1.0 - MIN_SIGNAL_MULTIPLIER) * strength

    notional = base_value * multiplier

    dlog(
        f"[DEBUG] compute_order_notional: base_value={base_value:.2f}, "
        f"clamped={clamped}, strength={strength:.3f}, "
        f"multiplier={multiplier:.3f}, notional={notional:.2f}"
    )

    # Keep from constantly producing silly tiny sizes
    if notional < MIN_ORDER_DOLLARS:
        dlog(
            f"[DEBUG] notional {notional:.2f} < MIN_ORDER_DOLLARS {MIN_ORDER_DOLLARS:.2f}, "
            "returning 0.0"
        )
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
            dlog("[DEBUG] Alpaca.is_market_open: fetching market clock")
            clock = self.api.get_clock()
            is_open = bool(getattr(clock, "is_open", False))
            dlog(f"[DEBUG] Alpaca.is_market_open: is_open={is_open}")
            return is_open
        except Exception as e:
            print(f"Alpaca: failed to fetch market clock: {e}")
            # Be conservative: if we can't tell, don't trade.
            return False

    def get_available_funds(self) -> float:
        dlog("[DEBUG] Alpaca.get_available_funds: fetching account")
        account = self.api.get_account()
        funds = float(account.cash)
        dlog(f"[DEBUG] Alpaca.get_available_funds: cash={funds}")
        # You could also use account.buying_power; cash is simpler for now
        return funds

    def has_position(self, symbol: str) -> bool:
        try:
            dlog(f"[DEBUG] Alpaca.has_position: checking position for {symbol}")
            pos = self.api.get_position(symbol)
            qty = float(pos.qty)
            has_pos = qty > 0
            dlog(f"[DEBUG] Alpaca.has_position: qty={qty}, has_position={has_pos}")
            return has_pos
        except Exception as e:
            # Alpaca throws an error if no position exists
            if "position does not exist" in str(e).lower():
                dlog(f"[DEBUG] Alpaca.has_position: no existing position for {symbol}")
                return False
            dlog(f"[DEBUG] Alpaca.has_position: unexpected error: {e}")
            raise

    def has_open_buy_order(self, symbol: str) -> bool:
        dlog(f"[DEBUG] Alpaca.has_open_buy_order: listing open orders for {symbol}")
        orders = self.api.list_orders(
            status="open",
            direction="desc",
            nested=False,
        )
        for o in orders:
            # Avoid logging every order in production; use VERBOSE_DEBUG if needed
            vlog(f"[DEBUG] Alpaca.has_open_buy_order: open order {o.id} {o.symbol} {o.side}")
            if o.symbol.upper() == symbol.upper() and o.side.lower() == "buy":
                dlog(f"[DEBUG] Alpaca.has_open_buy_order: open BUY order exists for {symbol}")
                return True
        dlog(f"[DEBUG] Alpaca.has_open_buy_order: no open BUY order for {symbol}")
        return False

    def get_best_bid(self, symbol: str) -> Optional[float]:
        try:
            dlog(f"[DEBUG] Alpaca.get_best_bid: fetching latest quote for {symbol}")
            quote = self.api.get_latest_quote(symbol)
            bid = float(quote.bp)
            dlog(f"[DEBUG] Alpaca.get_best_bid: bid={bid}")
            if bid > 0:
                return bid
        except Exception as e:
            print(f"Alpaca: failed to fetch best bid for {symbol}: {e}")
        return None

    def place_order(self, symbol: str, notional: float) -> None:
        dlog(f"[DEBUG] Alpaca.place_order: symbol={symbol}, notional={notional:.2f}")
        if notional < self.min_notional:
            dlog(
                f"[DEBUG] Alpaca.place_order: notional ${notional:.2f} below Alpaca min "
                f"${self.min_notional:.2f}, skipping {symbol}"
            )
            return

        best_bid = self.get_best_bid(symbol)

        if best_bid is not None:
            qty = round(notional / best_bid, 3)  # fractional shares
            est_notional = qty * best_bid
            dlog(
                f"[DEBUG] Alpaca.place_order: best_bid={best_bid}, "
                f"qty={qty}, est_notional={est_notional:.2f}"
            )
            if est_notional < self.min_notional:
                dlog(
                    f"[DEBUG] Alpaca.place_order: qty rounding makes order below min notional "
                    f"({est_notional:.2f} < {self.min_notional:.2f}), skipping {symbol}"
                )
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
        vlog(f"[DEBUG] Kraken._private: POST {url} method={method}")
        resp = requests.post(url, headers=headers, data=data, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if result.get("error"):
            dlog(f"[DEBUG] Kraken._private: error from {method}: {result['error']}")
            raise RuntimeError(f"Kraken error {method}: {result['error']}")
        return result["result"]

    def _public(self, method: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}/0/public/{method}"
        vlog(f"[DEBUG] Kraken._public: GET {url} params={params or {}}")
        resp = requests.get(url, params=params or {}, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if result.get("error"):
            dlog(f"[DEBUG] Kraken._public: error from {method}: {result['error']}")
            raise RuntimeError(f"Kraken public error {method}: {result['error']}")
        return result["result"]

    def symbol_to_pair(self, symbol: str) -> Optional[str]:
        s = symbol.upper().strip()
        if s in KRAKEN_PAIR_MAP:
            pair = KRAKEN_PAIR_MAP[s]
        else:
            # Default: strip common separators, e.g. "ATOM/USD" -> "ATOMUSD"
            pair = s.replace("-", "").replace("/", "")
        dlog(f"[DEBUG] Kraken.symbol_to_pair: symbol={symbol} -> pair={pair}")
        return pair

    def get_available_funds(self) -> float:
        balances = self._private("Balance")
        vlog(f"[DEBUG] Kraken.get_available_funds: raw balances={balances}")
        usd = balances.get("ZUSD") or balances.get("USD")
        if usd is None:
            dlog("[DEBUG] Kraken.get_available_funds: no ZUSD/USD balance found, returning 0.0")
            return 0.0
        funds = float(usd)
        dlog(f"[DEBUG] Kraken.get_available_funds: using USD/ZUSD={funds}")
        return funds

    def has_open_buy_order(self, pair: str) -> bool:
        open_orders = self._private("OpenOrders")
        open_dict = open_orders.get("open", {})
        dlog(
            f"[DEBUG] Kraken.has_open_buy_order: checking pair={pair}, "
            f"num_open_orders={len(open_dict)}"
        )
        for oid, order in open_dict.items():
            desc = order.get("descr", {})
            vlog(f"[DEBUG] Kraken.has_open_buy_order: order_id={oid}, desc={desc}")
            if desc.get("pair") == pair and desc.get("type") == "buy":
                dlog(f"[DEBUG] Kraken.has_open_buy_order: found open BUY {oid} for pair={pair}")
                return True
        dlog(f"[DEBUG] Kraken.has_open_buy_order: no open BUY for pair={pair}")
        return False

    def get_best_bid(self, pair: str) -> Optional[float]:
        dlog(f"[DEBUG] Kraken.get_best_bid: requesting ticker for pair={pair}")
        ticker = self._public("Ticker", {"pair": pair})
        k = next(iter(ticker.keys()))
        bid_str = ticker[k]["b"][0]  # 'b' = [best bid, whole lot volume, lot volume]
        try:
            bid = float(bid_str)
            dlog(f"[DEBUG] Kraken.get_best_bid: pair={pair}, bid={bid}")
            if bid > 0:
                return bid
        except ValueError:
            dlog(f"[DEBUG] Kraken.get_best_bid: invalid bid_str='{bid_str}'")
        return None

    def place_order(self, symbol: str, notional: float) -> None:
        dlog(f"[DEBUG] Kraken.place_order: symbol={symbol}, notional={notional:.2f}")
        pair = self.symbol_to_pair(symbol)
        if not pair:
            dlog(f"[DEBUG] Kraken.place_order: no pair mapping for symbol {symbol}, skipping.")
            return

        if notional < self.min_notional:
            dlog(
                f"[DEBUG] Kraken.place_order: notional ${notional:.2f} below Kraken min "
                f"${self.min_notional:.2f}, skipping {symbol}"
            )
            return

        best_bid = self.get_best_bid(pair)
        if best_bid is None:
            dlog(
                f"[DEBUG] Kraken.place_order: could not fetch best bid for {pair}, "
                f"skipping order for {symbol}"
            )
            return

        volume = notional / best_bid
        volume = round(volume, 8)  # crypto precision
        dlog(
            f"[DEBUG] Kraken.place_order: best_bid={best_bid}, "
            f"raw_volume={notional / best_bid}, rounded_volume={volume}"
        )

        if volume * best_bid < self.min_notional:
            dlog(
                f"[DEBUG] Kraken.place_order: volume rounding makes order below min notional "
                f"({volume * best_bid:.2f} < {self.min_notional:.2f}), skipping {symbol}"
            )
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

    dlog(
        f"=== [DEBUG] Processing row {signal.row_index}: symbol={symbol_for_venue}, "
        f"venue={venue}, signal={signal.signal_pct} ==="
    )

    # For Alpaca (stocks): only trade when the regular market is open
    if not signal.is_crypto:
        dlog("[DEBUG] process_signal: non-crypto, checking Alpaca market status...")
        if not alpaca.is_market_open():
            dlog("[DEBUG] Alpaca: market is closed, skipping stock order.")
            return

    # Check for duplicate unfilled orders first
    if signal.is_crypto:
        dlog("[DEBUG] process_signal: crypto path, checking Kraken open orders...")
        pair = kraken.symbol_to_pair(symbol_for_venue)
        if not pair:
            dlog(f"[DEBUG] No Kraken pair mapping for {symbol_for_venue}, skipping.")
            return
        if kraken.has_open_buy_order(pair):
            dlog(f"[DEBUG] Kraken: open BUY order already exists for {pair}, skipping.")
            return
    else:
        dlog("[DEBUG] process_signal: stock path, checking Alpaca open orders and positions...")
        if alpaca.has_open_buy_order(symbol_for_venue):
            dlog(f"[DEBUG] Alpaca: open BUY order already exists for {symbol_for_venue}, skipping.")
            return
        # Also check positions so we don't double up the asset
        if alpaca.has_position(symbol_for_venue):
            dlog(f"[DEBUG] Alpaca: position already exists for {symbol_for_venue}, skipping.")
            return

    # Determine available funds on the selected venue
    available_funds = broker.get_available_funds()
    dlog(f"[DEBUG] process_signal: {venue} available funds: ${available_funds:.2f}")

    notional = compute_order_notional(available_funds, signal.signal_pct)
    if notional <= 0:
        dlog(f"[DEBUG] process_signal: computed notional is ${notional:.2f}, skipping (too small / zero).")
        return

    dlog(f"[DEBUG] process_signal: final notional for {symbol_for_venue}: ${notional:.2f}")

    # Final venue-specific min-check and order placement
    broker.place_order(symbol_for_venue, notional)


def main():
    print("=== Automated Buyer Bot run starting ===")
    signals = read_signals_from_sheet()
    if not signals:
        print("No signals found with 🟢 icon. Exiting.")
        return

    dlog(f"[DEBUG] main: Found {len(signals)} signals with 🟢")

    alpaca = AlpacaBroker()
    kraken = KrakenBroker()

    for signal in signals:
        try:
            process_signal(signal, alpaca, kraken)
        except Exception as e:
            dlog(f"[DEBUG] Error processing row {signal.row_index} ({signal.symbol}): {e}")

    print("=== Automated Buyer Bot run complete ===")


if __name__ == "__main__":
    main()
