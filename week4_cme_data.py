# ============================================================
# Week 4: Market option prices for MAE / RMSE vs model
#
# 1) Official CME: DataMine API (requires purchased dataset + API ID).
# 2) Practical proxy: Yahoo Finance OCC option symbols (NOT CME tape;
#    many single-name options clear via OCC; use only if course allows).
#
# survey-year: grid-search monthly (3rd Friday) expiries × strikes, count Yahoo
#              daily bars in [start,end) — use to preview e.g. 2024 coverage.
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Install yfinance: pip install yfinance") from e


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
OUT_DIR = ROOT / "data"


def third_friday(year: int, month: int) -> date:
    """Standard US equity monthly options expiry (third Friday)."""
    first = date(year, month, 1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + timedelta(weeks=2)


def monthly_expirations_iso(year: int, months: list[int] | None = None) -> list[str]:
    """YYYY-MM-DD strings for each month's third Friday."""
    mlist = months or list(range(1, 13))
    out = []
    for m in mlist:
        try:
            out.append(third_friday(year, m).isoformat())
        except Exception:
            continue
    return out


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def occ_symbol(root: str, expiration: str, right: str, strike: float) -> str:
    """
    Build OCC option ticker for Yahoo, e.g. JPM260515C00150000
    expiration: 'YYYY-MM-DD'
    right: 'C' or 'P'
    strike: e.g. 150.0 -> 8-digit strike field 00150000
    """
    dt = datetime.strptime(expiration, "%Y-%m-%d")
    yymmdd = dt.strftime("%y%m%d")
    strike_int = int(round(float(strike) * 1000))
    strike_part = f"{strike_int:08d}"
    r = right.upper()[:1]
    if r not in ("C", "P"):
        raise ValueError("right must be C or P")
    return f"{root.upper()}{yymmdd}{r}{strike_part}"


def pick_most_liquid_occ(
    root: str,
    right: str,
    max_expirations: int = 12,
    prefer_near_strike: float | None = None,
    strike_band: float | None = None,
) -> tuple[str, dict]:
    """
    Scan recent Yahoo option expirations and pick the (expiration, strike) with
    highest volume + openInterest score. Returns (occ_symbol, meta dict).
    If prefer_near_strike is set (e.g. spot price), optionally filter to strikes
    within strike_band (absolute dollars), else take global max liquidity.
    """
    t = yf.Ticker(root)
    exps = list(t.options[:max_expirations])
    best = None  # (score, symbol, meta)
    r = right.upper()
    for exp in exps:
        chain = t.option_chain(exp)
        leg = chain.calls if r == "C" else chain.puts
        df = leg.copy()
        if df.empty:
            continue
        vol = df["volume"].fillna(0).astype(float)
        oi = df["openInterest"].fillna(0).astype(float)
        df = df.assign(_score=vol + 0.1 * oi)
        if prefer_near_strike is not None and strike_band is not None:
            sub = df[(df["strike"] - prefer_near_strike).abs() <= strike_band].copy()
            if sub.empty:
                continue
            row = sub.sort_values("_score", ascending=False).iloc[0]
        else:
            row = df.sort_values("_score", ascending=False).iloc[0]
        score = float(row["_score"])
        strike = float(row["strike"])
        sym = occ_symbol(root, exp, r, strike)
        meta = {
            "root": root.upper(),
            "expiration": exp,
            "right": r,
            "strike": strike,
            "volume": float(row["volume"]) if row["volume"] == row["volume"] else None,
            "openInterest": float(row["openInterest"]) if row["openInterest"] == row["openInterest"] else None,
            "liquidity_score": score,
        }
        cand = (score, sym, meta)
        if best is None or score > best[0]:
            best = cand
    if best is None:
        raise RuntimeError("No option chain rows found; check symbol or network.")
    _score, sym, meta = best
    return sym, meta


def fetch_yahoo_option_history(
    symbol: str, start: str | None, end: str | None
) -> pd.DataFrame:
    """Daily bars for one OCC option; Close used as traded-price proxy."""
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        return df
    df = df.rename_axis("Date").reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == "" else c[0] for c in df.columns]
    close_col = "Close" if "Close" in df.columns else df.columns[-2]
    out = df[["Date", close_col]].copy()
    out = out.rename(columns={close_col: "market_price_close"})
    out["occ_symbol"] = symbol
    return out


def survey_option_liquidity_year(
    root: str,
    year: int,
    strikes: list[float],
    rights: str,
    start: str,
    end: str,
    months: list[int] | None = None,
    sleep_s: float = 0.25,
) -> pd.DataFrame:
    """
    For each standard monthly expiry in `year` (subset `months` if given) and each
    strike/right, query Yahoo daily history length inside [start, end]. Use this
    to see how many bars exist for e.g. 2024 before picking a contract.
    """
    rows = []
    exps = monthly_expirations_iso(year, months)
    rlist = []
    if "C" in rights.upper():
        rlist.append("C")
    if "P" in rights.upper():
        rlist.append("P")
    if not rlist:
        rlist = ["C"]

    total = len(exps) * len(strikes) * len(rlist)
    done = 0
    for exp in exps:
        for k in strikes:
            for r in rlist:
                sym = occ_symbol(root, exp, r, k)
                df = fetch_yahoo_option_history(sym, start, end)
                n = len(df)
                rows.append(
                    {
                        "occ_symbol": sym,
                        "expiration": exp,
                        "strike": k,
                        "right": r,
                        "n_trading_days": n,
                        "first_date": df["Date"].min() if n else None,
                        "last_date": df["Date"].max() if n else None,
                    }
                )
                done += 1
                if done % 15 == 0 or done == total:
                    print(f"Progress {done}/{total} ({sym} -> {n} rows)", file=sys.stderr)
                time.sleep(sleep_s)
    out = pd.DataFrame(rows)
    return out.sort_values("n_trading_days", ascending=False).reset_index(drop=True)


def fetch_option_chain_snapshot(
    root: str, expiration: str, strike_target: float, right: str
) -> pd.DataFrame:
    """Single-time snapshot: nearest strike row from Yahoo option chain."""
    t = yf.Ticker(root)
    if expiration not in t.options:
        near = min(t.options, key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - datetime.strptime(expiration, "%Y-%m-%d")).days))
        print(f"Note: expiration {expiration} not listed; using {near}", file=sys.stderr)
        expiration = near
    chain = t.option_chain(expiration)
    leg = chain.calls if right.upper() == "C" else chain.puts
    leg = leg.copy()
    leg["distance"] = (leg["strike"] - strike_target).abs()
    row = leg.sort_values("distance").iloc[0]
    mid = None
    bid, ask = row.get("bid"), row.get("ask")
    if bid is not None and ask is not None and bid == bid and ask == ask:
        mid = (float(bid) + float(ask)) / 2.0
    return pd.DataFrame(
        [
            {
                "asof_utc": datetime.utcnow().isoformat() + "Z",
                "root": root,
                "expiration": expiration,
                "right": right.upper(),
                "strike": float(row["strike"]),
                "lastPrice": float(row["lastPrice"]) if row.get("lastPrice") == row.get("lastPrice") else None,
                "bid": float(bid) if bid == bid else None,
                "ask": float(ask) if ask == ask else None,
                "mid": mid,
                "volume": float(row["volume"]) if row.get("volume") == row.get("volume") else None,
                "openInterest": float(row["openInterest"]) if row.get("openInterest") == row.get("openInterest") else None,
            }
        ]
    )


def fetch_datamine_file(
    api_id: str,
    api_secret: str,
    file_id: str,
    out_path: Path,
) -> Path:
    """
    Download a file you already purchased on CME DataMine.
    Credentials: CME DataMine -> My Profile -> APIs (see cmegroup.com/datamine).
    """
    token_url = "https://auth.cmegroup.com/as/token.oauth2"
    r = requests.post(
        token_url,
        data={"grant_type": "client_credentials"},
        auth=(api_id, api_secret),
        timeout=60,
    )
    r.raise_for_status()
    token = r.json()["access_token"]
    # Endpoint may change; verify in current DataMine API docs if this fails.
    url = "https://datamine.new.cmegroup.com/cme/api/v2/download"
    resp = requests.get(
        url, headers={"Authorization": f"Bearer {token}"}, params={"fid": file_id}, timeout=120
    )
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return out_path


def load_manual_table(path: Path, date_col: str, price_col: str) -> pd.DataFrame:
    """Load CME (or any) CSV you exported manually: need Date + actual price columns."""
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    return df[[date_col, price_col]].rename(columns={date_col: "Date", price_col: "market_price"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 4: CME / market option price helpers")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_yh = sub.add_parser("yahoo-history", help="Download daily history for one OCC option (Yahoo)")
    p_yh.add_argument("--symbol", help="Full OCC symbol, e.g. JPM260515C00150000")
    p_yh.add_argument("--root", default="JPM")
    p_yh.add_argument("--expiration", help="YYYY-MM-DD")
    p_yh.add_argument("--strike", type=float)
    p_yh.add_argument("--right", choices=("C", "P"), default="C")
    p_yh.add_argument("--start", default="2018-01-01")
    p_yh.add_argument("--end", default=None)
    p_yh.add_argument(
        "--auto-liquid",
        action="store_true",
        help="Pick highest (volume+OI) contract among recent expirations (more Yahoo bars).",
    )
    p_yh.add_argument(
        "--near-spot",
        action="store_true",
        help="With --auto-liquid, restrict to strikes within --strike-band of live spot.",
    )
    p_yh.add_argument(
        "--strike-band",
        type=float,
        default=25.0,
        help="Dollar width each side of spot for --near-spot (default 25).",
    )
    p_yh.add_argument(
        "-o",
        "--output",
        default=str(OUT_DIR / "week4_yahoo_option_history.csv"),
    )

    p_sn = sub.add_parser("snapshot", help="Current chain row nearest to strike (Yahoo)")
    p_sn.add_argument("--root", default="JPM")
    p_sn.add_argument("--expiration", required=True)
    p_sn.add_argument("--strike", type=float, default=None)
    p_sn.add_argument("--right", choices=("C", "P"), default="C")
    p_sn.add_argument("-o", "--output", default=str(OUT_DIR / "week4_option_snapshot.csv"))

    p_dm = sub.add_parser("datamine", help="Download purchased DataMine file by fid")
    p_dm.add_argument("--file-id", required=True, dest="file_id")
    p_dm.add_argument("-o", "--output", required=True)

    p_sv = sub.add_parser(
        "survey-year",
        help="Probe Yahoo OCC history row counts for a calendar year (e.g. 2024 monthly expiries × strikes).",
    )
    p_sv.add_argument("--year", type=int, required=True)
    p_sv.add_argument("--root", default="JPM")
    p_sv.add_argument(
        "--strikes",
        type=str,
        default="135,140,145,150,155,160,165,170,175,180,185,190,195,200",
        help="Comma-separated strikes to try (e.g. 150,155,160).",
    )
    p_sv.add_argument(
        "--rights",
        type=str,
        default="C",
        help="C, P, or CP for both.",
    )
    p_sv.add_argument(
        "--months",
        type=str,
        default=None,
        help="Optional: e.g. 1,2,3 or 6-9 to limit months (default all 12).",
    )
    p_sv.add_argument(
        "--start",
        type=str,
        default=None,
        help="YYYY-MM-DD window start (default: --year-01-01).",
    )
    p_sv.add_argument(
        "--end",
        type=str,
        default=None,
        help="YYYY-MM-DD window end exclusive (default: next year 01-01).",
    )
    p_sv.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Seconds between Yahoo requests (rate limit courtesy).",
    )
    p_sv.add_argument(
        "-o",
        "--output",
        default=str(OUT_DIR / "year_option_survey.csv"),
    )

    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "yahoo-history":
        sym = args.symbol
        meta = {}
        if args.auto_liquid:
            spot = None
            if args.near_spot:
                t = yf.Ticker(args.root)
                spot = t.fast_info.get("last_price") or t.fast_info.get("regular_market_price")
                if spot is None:
                    spot = float(t.history(period="5d")["Close"].iloc[-1])
                print(f"Using spot ~ {spot:.2f} for strike filter", file=sys.stderr)
            sym, meta = pick_most_liquid_occ(
                args.root,
                args.right,
                max_expirations=12,
                prefer_near_strike=spot if args.near_spot else None,
                strike_band=args.strike_band if args.near_spot else None,
            )
            print(
                f"Auto-liquid: {sym}  exp={meta['expiration']} K={meta['strike']} "
                f"score={meta['liquidity_score']:.0f}",
                file=sys.stderr,
            )
        elif not sym:
            cfg = load_config()
            if not args.expiration:
                print("Provide --expiration YYYY-MM-DD, or --symbol, or --auto-liquid.", file=sys.stderr)
                sys.exit(1)
            strike = (
                args.strike if args.strike is not None else float(cfg["contract_terms"]["X"])
            )
            sym = occ_symbol(args.root, args.expiration, args.right, strike)
            meta = {"expiration": args.expiration, "strike": strike, "right": args.right}
            print(f"Using symbol {sym}", file=sys.stderr)
            if strike < 200:
                print(
                    "Hint: strikes far from current spot are often illiquid on Yahoo; "
                    "expect very few daily bars. Try --auto-liquid or --near-spot.",
                    file=sys.stderr,
                )
        else:
            print(f"Using symbol {sym}", file=sys.stderr)

        df = fetch_yahoo_option_history(sym, args.start, args.end)
        if df.empty:
            print("No rows returned; check symbol / dates / listing.", file=sys.stderr)
            sys.exit(1)
        for k, v in meta.items():
            df[k] = v
        if len(df) < 15 and not args.auto_liquid:
            print(
                f"Only {len(df)} rows — often because the contract rarely trades. "
                "Re-run with --auto-liquid (and optionally --near-spot).",
                file=sys.stderr,
            )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} rows -> {args.output}")

    elif args.mode == "snapshot":
        cfg = load_config()
        strike = args.strike if args.strike is not None else float(cfg["contract_terms"]["X"])
        df = fetch_option_chain_snapshot(args.root, args.expiration, strike, args.right)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(df.to_string(index=False))
        print(f"Saved -> {args.output}")

    elif args.mode == "datamine":
        api_id = os.environ.get("CME_DATAMINE_API_ID")
        api_secret = os.environ.get("CME_DATAMINE_API_SECRET")
        if not api_id or not api_secret:
            print("Set CME_DATAMINE_API_ID and CME_DATAMINE_API_SECRET", file=sys.stderr)
            sys.exit(1)
        out = fetch_datamine_file(api_id, api_secret, args.file_id, Path(args.output))
        print(f"Saved -> {out}")

    elif args.mode == "survey-year":
        strikes = [float(x.strip()) for x in args.strikes.split(",") if x.strip()]
        months = None
        if args.months:
            months = []
            for part in args.months.replace(" ", "").split(","):
                if "-" in part:
                    a, b = part.split("-", 1)
                    months.extend(range(int(a), int(b) + 1))
                else:
                    months.append(int(part))
        start = args.start or f"{args.year}-01-01"
        end = args.end or f"{args.year + 1}-01-01"
        if not monthly_expirations_iso(args.year, months):
            print("No expirations to survey (check --year / --months).", file=sys.stderr)
            sys.exit(1)
        res = survey_option_liquidity_year(
            args.root,
            args.year,
            strikes,
            args.rights,
            start,
            end,
            months=months,
            sleep_s=args.sleep,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.output, index=False)
        print("\n=== Top 15 by Yahoo trading days with data ===")
        print(res.head(15).to_string(index=False))
        nz = (res["n_trading_days"] > 0).sum()
        print(
            f"\nSummary: {len(res)} contracts tried, {nz} with at least 1 bar, "
            f"max n={res['n_trading_days'].max()}, median(n)={res['n_trading_days'].median():.1f}",
            file=sys.stderr,
        )
        print(f"\nSaved full table -> {args.output}")


if __name__ == "__main__":
    main()
