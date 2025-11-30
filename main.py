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
COL_SIGNAL = 18     # Column S / pct_diff
COL_ICON = 23       # Column X ("Buy symbol" / icon)

BASE_ORDER_FRACTION = float(os.getenv("BASE_ORDER_FRACTION", "0.10"))
MIN_SIGNAL_MULTIPLIER = float(os.getenv("MIN_SIGNAL_MULTIPLIER", "0.25"))
MIN_ORDER_DOLLARS = float(os.getenv("MIN_ORDER_DOLLARS", "1.0"))

ALPACA_MIN_ORDER_NOTIONAL = float(os.getenv("ALPACA_MIN_ORDER_NOTIONAL", "1.0"))
KRAKEN_MIN_ORDER_NOTIONAL = float(os.getenv("KRAKEN_MIN_ORDER_NOTIONAL", "1.0"))

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
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

        # Only process rows with the green icon
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
            vlog(f"[DEBUG] Alpaca.has_open_buy_order: open order {o.id} {o.symbol} {o.side}")
            if o.symbol.upper() == symbol.upper() and o.side.lower() == "buy":
                dlog(f"[DEBUG] Alpaca.has_open_buy_order: open BUY order exists for {symbol}")
                return True
        dlog(f"[DEBUG] Alpaca.has_open_buy_order: no open BUY order for {symbol}")
        return False

    def get_best_bid(self, symbol: str) -> Optional[float]:
        # Still here if you ever want it, but no longer used for order placement.
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

        # Pure MARKET notional order
        print(f"Alpaca: placing MARKET notional ${notional:.2f} buy for {symbol}")
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

        # Track last nonce we used (per process) to ensure monotonicity
        self._last_nonce = 0

        # Cache for pair -> base asset lookups
        self._pair_base_cache: dict[str, str] = {}

        # Cache for full pair info (includes ordemin, etc.)
        self._pair_info_cache: dict[str, dict] = {}

    def _next_nonce(self) -> int:
        """
        Generate a strictly increasing nonce based on a *very* large
        time-based value (nanoseconds since epoch), so it's bigger
        than any previous ms/µs-based nonce Kraken has seen.
        """
        now = int(time.time() * 1_000_000_000)  # ns since epoch
        if now <= self._last_nonce:
            now = self._last_nonce + 1
        self._last_nonce = now
        return now

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
        data["nonce"] = self._next_nonce()
        url_path = f"/0/private/{method}"
        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._sign(url_path, data),
        }
        url = self.base_url + url_path
        vlog(f"[DEBUG] Kraken._private: POST {url} method={method}, nonce={data['nonce']}")
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

    # === pair info & base-asset helpers =======================

    def _get_pair_info(self, pair: str) -> Optional[dict]:
        """
        Fetch and cache the AssetPairs entry for a given pair.
        This includes fields like 'base', 'ordemin', etc.
        """
        if pair in self._pair_info_cache:
            return self._pair_info_cache[pair]

        try:
            info = self._public("AssetPairs", {"pair": pair})
            if not info:
                dlog(f"[DEBUG] Kraken._get_pair_info: empty info for pair={pair}")
                return None

            k = next(iter(info.keys()))
            pair_info = info[k]
            self._pair_info_cache[pair] = pair_info
            vlog(f"[DEBUG] Kraken._get_pair_info: cached info for pair={pair}: {pair_info}")
            return pair_info
        except Exception as e:
            dlog(f"[DEBUG] Kraken._get_pair_info: error fetching info for pair={pair}: {e}")
            return None

    def _get_base_asset_for_pair(self, pair: str) -> Optional[str]:
        """
        Look up the base asset for a given trading pair using AssetPairs.
        Uses a small cache to avoid repeated API calls for the same pair.
        """
        if pair in self._pair_base_cache:
            return self._pair_base_cache[pair]

        pair_info = self._get_pair_info(pair)
        if not pair_info:
            dlog(f"[DEBUG] Kraken._get_base_asset_for_pair: no pair_info for pair={pair}")
            return None

        base_asset = pair_info.get("base")
        dlog(f"[DEBUG] Kraken._get_base_asset_for_pair: pair={pair}, base={base_asset}")
        if base_asset:
            self._pair_base_cache[pair] = base_asset
            return base_asset

        return None

    def has_position(self, pair: str) -> bool:
        """
        Returns True if we currently hold any of the base asset for this pair.
        For example, for pair XBTUSD it checks whether our XXBT balance > 0.
        """
        base_asset = self._get_base_asset_for_pair(pair)
        if not base_asset:
            dlog(f"[DEBUG] Kraken.has_position: could not determine base asset for {pair}.")
            return False

        try:
            balances = self._private("Balance")
            vlog(f"[DEBUG] Kraken.has_position: balances={balances}")
            bal_str = balances.get(base_asset)
            if not bal_str:
                dlog(
                    f"[DEBUG] Kraken.has_position: no balance entry for base_asset={base_asset}, "
                    f"treating as no position."
                )
                return False

            amount = float(bal_str)
            has_pos = amount > 0
            dlog(
                f"[DEBUG] Kraken.has_position: base_asset={base_asset}, "
                f"amount={amount}, has_position={has_pos}"
            )
            return has_pos
        except Exception as e:
            dlog(f"[DEBUG] Kraken.has_position: error checking position for pair={pair}: {e}")
            # On error, be permissive so we don't deadlock trading completely.
            # If you prefer to block on error, return True instead.
            return False

    # ===============================================================

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

    def _get_fallback_price(self, pair: str) -> Optional[float]:
        """
        Use last trade price from Ticker as a backup for sizing
        when we can't get a clean best bid.
        """
        try:
            ticker = self._public("Ticker", {"pair": pair})
            k = next(iter(ticker.keys()))
            last_str = ticker[k]["c"][0]  # 'c' = [last trade price, lot volume]
            price = float(last_str)
            dlog(f"[DEBUG] Kraken._get_fallback_price: pair={pair}, last_price={price}")
            if price > 0:
                return price
        except Exception as e:
            dlog(f"[DEBUG] Kraken._get_fallback_price: error for pair={pair}: {e}")
        return None

    def place_order(self, symbol: str, notional: float) -> None:
        dlog(f"[DEBUG] Kraken.place_order: symbol={symbol}, notional={notional:.2f}")
        pair = self.symbol_to_pair(symbol)
        if not pair:
            dlog(f"[DEBUG] Kraken.place_order: no pair mapping for symbol {symbol}, skipping.")
            return

        if notional < self.min_notional:
            dlog(
                f"[DEBUG] Kraken.place_order: initial notional ${notional:.2f} below Kraken "
                f"config min ${self.min_notional:.2f}, skipping {symbol}"
            )
            return

        # We always do MARKET orders now, but still need a price to size volume.
        price_for_sizing: Optional[float] = None

        best_bid = self.get_best_bid(pair)
        if best_bid is not None and best_bid > 0:
            price_for_sizing = best_bid
            dlog(
                f"[DEBUG] Kraken.place_order: using best_bid={best_bid} "
                f"for sizing {symbol} ({pair})"
            )
        else:
            dlog(
                f"[DEBUG] Kraken.place_order: no clean best bid for {pair}, "
                "trying fallback last trade price for sizing."
            )
            fallback_price = self._get_fallback_price(pair)
            if fallback_price is not None and fallback_price > 0:
                price_for_sizing = fallback_price
                dlog(
                    f"[DEBUG] Kraken.place_order: using fallback_price={fallback_price} "
                    f"for sizing {symbol} ({pair})"
                )

        if price_for_sizing is None:
            dlog(
                f"[DEBUG] Kraken.place_order: no usable price for {pair}, "
                f"skipping order for {symbol}"
            )
            return

        # Initial volume based on our sizing logic
        raw_volume = notional / price_for_sizing
        volume = round(raw_volume, 8)  # crypto precision
        est_notional = volume * price_for_sizing

        dlog(
            f"[DEBUG] Kraken.place_order: price_for_sizing={price_for_sizing}, "
            f"raw_volume={raw_volume}, rounded_volume={volume}, "
            f"est_notional={est_notional:.2f}"
        )

        # ===== NEW: bump up to Kraken's ordemin (per-pair volume minimum) =====
        pair_info = self._get_pair_info(pair)
        ordermin: Optional[float] = None
        if pair_info is not None:
            ord_str = pair_info.get("ordemin")
            if ord_str is not None:
                try:
                    ordermin = float(ord_str)
                except ValueError:
                    dlog(
                        f"[DEBUG] Kraken.place_order: invalid ordemin '{ord_str}' "
                        f"for pair={pair}"
                    )

        if ordermin is not None and ordermin > 0 and volume < ordermin:
            dlog(
                f"[DEBUG] Kraken.place_order: volume {volume} below Kraken ordemin "
                f"{ordemin} for {pair}, bumping volume up to ordemin."
            )
            volume = ordermin
            est_notional = volume * price_for_sizing
            dlog(
                f"[DEBUG] Kraken.place_order: after ordemin bump: "
                f"volume={volume}, est_notional={est_notional:.2f}"
            )
        # ===== END NEW BLOCK =====

        # Re-check against our own notional floor using the final volume
        if est_notional < self.min_notional:
            dlog(
                f"[DEBUG] Kraken.place_order: final est_notional ${est_notional:.2f} "
                f"below Kraken config min ${self.min_notional:.2f} after ordemin bump, "
                f"skipping {symbol}"
            )
            return

        data = {
            "pair": pair,
            "type": "buy",
            "ordertype": "market",
            "volume": str(volume),
        }

        print(f"Kraken: placing MARKET buy {symbol} ({pair}), vol={volume}")
        self._private("AddOrder", data=data)


# =========================
# Main orchestration
# =========================

def process_signal(
    signal: TradeSignal,
    alpaca: Optional[AlpacaBroker],
    kraken: Optional[KrakenBroker],
) -> None:
    """
    Route a single TradeSignal to the appropriate broker. Supports running
    with only one broker configured (e.g., Kraken-only or Alpaca-only).
    """

    # Decide venue & ensure the right broker exists
    if signal.is_crypto:
        if kraken is None:
            dlog(
                f"[DEBUG] process_signal: crypto signal for {signal.symbol}, "
                "but Kraken is not configured. Skipping."
            )
            return
        venue = "Kraken"
        broker = kraken
        symbol_for_venue = signal.symbol
    else:
        if alpaca is None:
            dlog(
                f"[DEBUG] process_signal: stock signal for {signal.symbol}, "
                "but Alpaca is not configured. Skipping."
            )
            return
        venue = "Alpaca"
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
        # also skip if we already hold the base asset for this pair
        if kraken.has_position(pair):
            dlog(
                f"[DEBUG] Kraken: position already exists for base asset of {pair}, "
                f"skipping."
            )
            return
    else:
        dlog("[DEBUG] process_signal: stock path, checking Alpaca open orders and positions...")
        if alpaca.has_open_buy_order(symbol_for_venue):
            dlog(
                f"[DEBUG] Alpaca: open BUY order already exists for "
                f"{symbol_for_venue}, skipping."
            )
            return
        # Also check positions so we don't double up the asset
        if alpaca.has_position(symbol_for_venue):
            dlog(
                f"[DEBUG] Alpaca: position already exists for {symbol_for_venue}, "
                f"skipping."
            )
            return

    # Determine available funds on the selected venue
    available_funds = broker.get_available_funds()
    dlog(f"[DEBUG] process_signal: {venue} available funds: ${available_funds:.2f}")

    notional = compute_order_notional(available_funds, signal.signal_pct)
    if notional <= 0:
        dlog(
            f"[DEBUG] process_signal: computed notional is ${notional:.2f}, "
            "skipping (too small / zero)."
        )
        return

    dlog(
        f"[DEBUG] process_signal: final notional for {symbol_for_venue}: "
        f"${notional:.2f}"
    )

    # Final venue-specific min-check and order placement
    broker.place_order(symbol_for_venue, notional)


def main():
    print("=== Automated Buyer Bot run starting ===")
    signals = read_signals_from_sheet()
    if not signals:
        print("No signals found with 🟢 icon. Exiting.")
        return

    dlog(f"[DEBUG] main: Found {len(signals)} signals with 🟢")

    # Try to bring up Alpaca, but don't die if it fails.
    alpaca: Optional[AlpacaBroker] = None
    try:
        alpaca = AlpacaBroker()
    except Exception as e:
        dlog(
            f"[DEBUG] AlpacaBroker init failed ({e}). "
            "Continuing without Alpaca (stocks disabled)."
        )

    # Try to bring up Kraken, but don't die if it fails.
    kraken: Optional[KrakenBroker] = None
    try:
        kraken = KrakenBroker()
    except Exception as e:
        dlog(
            f"[DEBUG] KrakenBroker init failed ({e}). "
            "Continuing without Kraken (crypto disabled)."
        )

    if alpaca is None and kraken is None:
        print("No brokers configured (both Alpaca and Kraken failed). Exiting.")
        return

    for signal in signals:
        try:
            process_signal(signal, alpaca, kraken)
        except Exception as e:
            dlog(f"[DEBUG] Error processing row {signal.row_index} ({signal.symbol}): {e}")

    print("=== Automated Buyer Bot run complete ===")


if __name__ == "__main__":
    main()
