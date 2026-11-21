"""
404 Chips Not Found — Version 16 (FINAL TOURNAMENT BUILD)
Fixes from Version 15:
1. Exact zero-tail top-X% preflop equity (_eq_vs_top) + calibrated 169-hand HU
   equity ordering (eliminates dominated preflop calls vs Angelo, Nits, Limp-Shovers).
2. Removes postflop bluff_floor double-counting, tightens preflop carry-over,
   adds opponent call-street narrowing, and tracks BB limp-shoves explicitly.
3. Counter-Angelo & Counter-Nit gears: 2.2x button opens, wider defense & 0.56+
   bluff-raises vs cheap flop stabs, and automatic cheap bet scaling vs Nits.
4. Smart 7-2 Side-Prize Shove: jams 7-2 when ahead in chips (>=1.15x) or
   eff_bb <= 35 (dodging 200 BB suicide busts while maximizing 7-2 leaderboard EV).
"""

import itertools
import random
import time
import eval7

# ── Core Tuning & Safety Limits ──────────────────────────────────────────────
SEVEN_TWO_SHOVE = True   # Smart 7-2 shove (when ahead in chips or <= 35 BB)
THINK_S = 0.140          # Hard Monte Carlo deadline (140 ms << 2000 ms limit)
POSTFLOP_SIMS = 900      # Range-sampled flop/turn rollouts (~2-5 ms typical)
PREFLOP_SIMS = 450       # Cached preflop equity sims vs exact top X% range
PUSH_FOLD_BB = 15.0      # Effective stack <= 15 BB enters Nash push/fold mode
BIG_BET = 1.05           # Bets > 1.05x pot count as overbets
LOOSE_MIN = 0.08         # Narrowest range width assigned to an action
MAX_K = 12.0             # Order-statistic cap ("best of 12 hands")
COMMIT_PREMIUM = 0.08    # Max deep-stack postflop commitment premium

# Freezeout tournament-life premium by total BBs in play (both stacks + pot)
LIFE_PREMIUM = (
    (20.0, 0.000), (25.0, 0.012), (33.0, 0.036), (50.0, 0.068),
    (67.0, 0.096), (100.0, 0.122), (133.0, 0.144), (200.0, 0.167), (400.0, 0.221),
)

# Heads-Up Nash equilibrium SB push & BB call fractions by eff_bb (no-ante baseline)
NASH_PUSH_NO_ANTE = (
    (1.0, 1.00), (2.0, 0.96), (3.0, 0.86), (4.0, 0.79), (5.0, 0.74),
    (6.0, 0.70), (8.0, 0.63), (10.0, 0.58), (12.0, 0.53), (15.0, 0.46), (20.0, 0.37),
)
NASH_CALL_NO_ANTE = (
    (1.0, 1.00), (2.0, 0.94), (3.0, 0.72), (4.0, 0.59), (5.0, 0.52),
    (6.0, 0.46), (8.0, 0.41), (10.0, 0.37), (12.0, 0.34), (15.0, 0.30), (20.0, 0.25),
)

RANKS = "AKQJT98765432"
RANK_VAL = {r: 12 - i for i, r in enumerate(RANKS)}  # '2'->0 .. 'A'->12
SUITS = "shdc"
ALL_CARDS = [r + s for r in RANKS for s in SUITS]
CARD_RECORDS = [(c, eval7.Card(c), RANK_VAL[c[0]], c[1]) for c in ALL_CARDS]
CARD = {c: card for c, card, _, _ in CARD_RECORDS}
VOLUNTARY = ("fold", "check", "call", "raise", "all_in")


# ── 169 Starting-Hand Classes & Calibrated HU Equity Ordering ────────────────

def _build_hand_classes():
    names = []
    for i, a in enumerate(RANKS):
        for j, b in enumerate(RANKS):
            if i == j:
                names.append(a + b)
            elif i < j:
                names.append(a + b + "s")
            else:
                names.append(b + a + "o")

    combos = [6 if len(n) == 2 else (4 if n[2] == "s" else 12) for n in names]
    # Calibrated to exact HU all-in equities vs a random hand
    hi_base = (0.315, 0.323, 0.330, 0.338, 0.342, 0.346, 0.368, 0.388, 0.412, 0.435, 0.464, 0.501, 0.546)
    scores = []
    for n in names:
        hi = RANK_VAL[n[0]]
        lo = RANK_VAL[n[1]]
        if hi == lo:
            sc = 0.503 + 0.0345 * hi - 0.00045 * hi * hi
        else:
            gap = hi - lo - 1
            if hi < 11:
                conn = 0.028 if gap == 0 else (0.018 if gap == 1 else (0.009 if gap == 2 else 0.0))
            else:
                conn = 0.014 if gap == 0 else (0.008 if gap == 1 else (0.003 if gap == 2 else 0.0))
            wheel = 0.010 if (hi == 12 and lo <= 3) else 0.0
            suited = 0.022 if n.endswith("s") else 0.0
            sc = hi_base[hi] + 0.0082 * lo + 0.00014 * lo * lo + conn + wheel + suited
        scores.append(sc)

    order = sorted(range(169), key=lambda idx: -scores[idx])
    cum_w = [0]
    rank_end = [0.0] * 169
    for pos, cls_idx in enumerate(order):
        cum_w.append(cum_w[-1] + combos[cls_idx])
        rank_end[cls_idx] = cum_w[pos + 1] / 1326.0

    key_to_rank_end = {}
    for cls_idx, n in enumerate(names):
        hi = RANK_VAL[n[0]]
        lo = RANK_VAL[n[1]]
        suited = n.endswith("s")
        key_to_rank_end[(hi, lo, suited)] = rank_end[cls_idx]

    class_index = {n: k for k, n in enumerate(names)}
    return names, class_index, rank_end, key_to_rank_end


NAMES, CLASS_INDEX, RANK_END, KEY_RANK_END = _build_hand_classes()


def _hand_class(c1: str, c2: str) -> int:
    r1, r2 = c1[0], c2[0]
    if r1 == r2:
        return CLASS_INDEX[r1 + r2]
    if RANK_VAL[r1] < RANK_VAL[r2]:
        r1, r2 = r2, r1
    return CLASS_INDEX[r1 + r2 + ("s" if c1[1] == c2[1] else "o")]


def _interp(table: tuple, x: float) -> float:
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / float(x1 - x0)
    return table[-1][1]


# ── Match-Long Opponent State & Caches ───────────────────────────────────────

def _fresh_stats() -> dict:
    return {
        "open_n": 0, "open_shove": 0, "open_raise": 0, "open_limp": 0, "open_fold": 0,
        "sopen_n": 0, "sopen_shove": 0,
        "limp_n": 0, "limp_shove": 0, "limp_raise": 0,
        "opp_limp_iso_n": 0, "opp_limp_shove": 0,
        "raise_n": 0, "raise_shove": 0, "raise_3bet": 0, "raise_fold": 0, "raise_call": 0,
        "shove_n": 0, "shove_call": 0,
        "bet_faced_n": 0, "bet_fold": 0,
        "fbet_n": 0, "fbet": 0,
        "lbet_n": 0, "lbet": 0,
        "rz_n": 0, "rz": 0,
        "big_n": 0, "big": 0,
        "agg_n": 0, "agg": 0,
        "shown_flop_n": 0, "shown_flop_weak": 0,
        "shown_late_n": 0, "shown_late_weak": 0,
        "shown_big_n": 0, "shown_big_weak": 0,
        "sbet_n": 0, "sbet_fold": 0,
        "nbet_n": 0, "nbet_fold": 0,
        "rr_n": 0, "rr_fold": 0,
        "any_n": 0, "any_fold": 0,
    }


STATS = _fresh_stats()
_DONE_HANDS = set()
_BOARDS = {}
_DEPTH = {}
_FIRST_HAND_ID = None
_PREFLOP_CACHE = {}   # {(cls_idx, bucket_x_0_to_20): equity}


def _reset_if_new_match(state: dict) -> None:
    global _FIRST_HAND_ID
    hand_num = state.get("hand_num")
    hand_id = state.get("hand_id")
    if hand_num == 0:
        if _FIRST_HAND_ID is not None and hand_id != _FIRST_HAND_ID:
            STATS.clear()
            STATS.update(_fresh_stats())
            _DONE_HANDS.clear()
            _BOARDS.clear()
            _DEPTH.clear()
        _FIRST_HAND_ID = hand_id


def _rate(hits: float, n: float, prior: float, strength: float) -> float:
    return (hits + prior * strength) / float(n + strength)


# ── Exact River Strength & Exact Zero-Tail Preflop Range Equity ──────────────

def _raw_river_strength(cards: list, board: list) -> float:
    """Exact unweighted equity across all 990 opponent combos on a 5-card board."""
    known = set(cards) | set(board)
    deck = [card for c, card, _, _ in CARD_RECORDS if c not in known]
    brd = [CARD[c] for c in board]
    mine = eval7.evaluate([CARD[c] for c in cards] + brd)
    ev = eval7.evaluate
    opp_buf = brd + [None, None]
    score = 0.0
    n = 0
    for c1, c2 in itertools.combinations(deck, 2):
        opp_buf[5] = c1
        opp_buf[6] = c2
        other = ev(opp_buf)
        if mine > other:
            score += 1.0
        elif mine == other:
            score += 0.5
        n += 1
    return score / n if n else 0.5


def _raw_mc_equity(cards: list, board: list, sims: int = 120) -> float:
    """Fast unweighted equity vs a random hand (used for Biggest Bluffer <0.39 check)."""
    known = set(cards) | set(board)
    deck = [card for c, card, _, _ in CARD_RECORDS if c not in known]
    hero = [CARD[c] for c in cards]
    brd = [CARD[c] for c in board]
    need = 5 - len(brd)
    ev = eval7.evaluate
    sample = random.sample
    wins = 0.0
    for _ in range(sims):
        s = sample(deck, 2 + need)
        fb = brd + s[2:]
        h = ev(hero + fb)
        o = ev(s[:2] + fb)
        if h > o:
            wins += 1.0
        elif h == o:
            wins += 0.5
    return wins / sims


def _eq_vs_top(cards: list, keep_frac: float, sims: int = PREFLOP_SIMS) -> float:
    """
    Cached preflop equity of `cards` against STRICTLY the strongest `keep_frac`
    of starting hands (zero garbage tail!).
    """
    cls_idx = _hand_class(cards[0], cards[1])
    x = max(0.05, min(1.0, keep_frac))
    bucket = int(round(x * 20))
    cache_key = (cls_idx, bucket)
    if cache_key in _PREFLOP_CACHE:
        return _PREFLOP_CACHE[cache_key]

    x_eff = max(0.05, bucket / 20.0)
    used = set(cards)
    deck_recs = [(card, r, s) for c, card, r, s in CARD_RECORDS if c not in used]
    deck_cards = [rec[0] for rec in deck_recs]
    hero = [CARD[c] for c in cards]
    ev = eval7.evaluate
    sample = random.sample

    if x_eff >= 0.98:
        wins = 0.0
        n_sims = max(sims, 600)
        for _ in range(n_sims):
            s = sample(deck_cards, 7)
            fb = s[2:]
            hs = ev(hero + fb)
            os_ = ev(s[:2] + fb)
            if hs > os_:
                wins += 1.0
            elif hs == os_:
                wins += 0.5
        eq = wins / n_sims
    else:
        valid_combos = [
            (c1, c2)
            for (c1, r1, s1), (c2, r2, s2) in itertools.combinations(deck_recs, 2)
            if KEY_RANK_END[(r1, r2, s1 == s2) if r1 >= r2 else (r2, r1, s1 == s2)] <= x_eff
        ]
        if not valid_combos:
            valid_combos = [(c1, c2) for (c1, _, _), (c2, _, _) in itertools.combinations(deck_recs, 2)]

        choice = random.choice
        wins = 0.0
        for _ in range(sims):
            c1, c2 = choice(valid_combos)
            draw7 = sample(deck_cards, 7)
            fb = [x_ for x_ in draw7 if x_ is not c1 and x_ is not c2][:5]
            hs = ev(hero + fb)
            os_ = ev([c1, c2] + fb)
            if hs > os_:
                wins += 1.0
            elif hs == os_:
                wins += 0.5
        eq = wins / sims

    _PREFLOP_CACHE[cache_key] = eq
    return eq


# ── Match History Replay & Continuous Opponent Counters ──────────────────────

def _replay(entries: list, me: str) -> list:
    """Rebuild pot sizes per street to classify every postflop opponent action."""
    pot = 0
    bets = {}
    street = "preflop"
    out = []
    last = None  # (who, size_frac, was_raise)
    for e in entries:
        act = e.get("action")
        st = e.get("street")
        if act == "show" or act is None:
            continue
        if st != street and st in ("flop", "turn", "river"):
            street, bets, last = st, {}, None
        who = e.get("bot_id")
        amount = int(e.get("amount") or 0)
        mine = bets.get(who, 0)
        if act == "ante":
            pot += amount
            continue
        if act in ("small_blind", "big_blind"):
            bets[who] = mine + amount
            pot += amount
            continue
        if act not in VOLUNTARY:
            continue

        top = max(bets.values()) if bets else 0
        to_call = max(0, top - mine)
        rec = {
            "street": street,
            "facing": to_call > 0,
            "kind": "none",
            "big": False,
            "size": 0.0,
            "action": act,
            "faced_size": last[1] if last and last[0] == me else 0.0,
            "faced_raise": bool(last and last[0] == me and last[2]),
        }
        if act == "call":
            bets[who] = mine + amount
            pot += amount
        elif act in ("raise", "all_in"):
            if amount > top:
                size = (amount - top) / float(max(1, pot + to_call))
                rec["kind"] = "raise" if to_call > 0 else "bet"
                rec["size"] = size
                rec["big"] = size > BIG_BET or (act == "all_in" and size > 0.75)
                last = (who, size, to_call > 0)
            pot += max(0, amount - mine)
            bets[who] = max(mine, amount)

        if who != me and street != "preflop":
            out.append(rec)
    return out


def _digest(entries: list, me: str, hand_num=None) -> None:
    """Update continuous STATS from one completed hand."""
    sb_id = None
    for e in entries:
        if e.get("action") == "small_blind":
            sb_id = e.get("bot_id")
    if sb_id is None:
        return

    acts = [e for e in entries if e.get("action") in VOLUNTARY]
    pre = [e for e in acts if e.get("street") == "preflop"]
    s = STATS

    for e in pre:
        if e.get("bot_id") != me:
            s["agg_n"] += 1
            s["agg"] += int(e["action"] in ("raise", "all_in"))
            if e["action"] != "check":
                s["any_n"] += 1
                s["any_fold"] += int(e["action"] == "fold")

    if pre:
        first = pre[0]
        if sb_id != me and first.get("bot_id") != me:
            s["open_n"] += 1
            a = first["action"]
            depth = None
            for back in range(0, 4):
                if hand_num is not None and (hand_num - back) in _DEPTH:
                    depth = _DEPTH[hand_num - back]
                    break
            if depth is not None and depth <= 25:
                s["sopen_n"] += 1
                s["sopen_shove"] += int(a == "all_in")
            if a == "all_in":
                s["open_shove"] += 1
            elif a == "raise":
                s["open_raise"] += 1
            elif a == "call":
                s["open_limp"] += 1
                # Track opponent limp-shoves when we isolate from the Big Blind
                if (
                    len(pre) >= 3
                    and pre[1].get("bot_id") == me
                    and pre[1].get("action") == "raise"
                    and pre[2].get("bot_id") != me
                ):
                    s["opp_limp_iso_n"] += 1
                    if pre[2].get("action") in ("raise", "all_in"):
                        s["opp_limp_shove"] += 1
            elif a == "fold":
                s["open_fold"] += 1
        elif sb_id == me and first.get("bot_id") == me and len(pre) > 1 and pre[1].get("bot_id") != me:
            mine, theirs = first["action"], pre[1]["action"]
            if mine == "call":
                s["limp_n"] += 1
                if theirs == "all_in":
                    s["limp_shove"] += 1
                elif theirs == "raise":
                    s["limp_raise"] += 1
            elif mine == "raise":
                s["raise_n"] += 1
                if theirs == "all_in":
                    s["raise_shove"] += 1
                elif theirs == "raise":
                    s["raise_3bet"] += 1
                elif theirs == "fold":
                    s["raise_fold"] += 1
                elif theirs == "call":
                    s["raise_call"] += 1
            elif mine == "all_in":
                s["shove_n"] += 1
                if theirs in ("call", "all_in"):
                    s["shove_call"] += 1

    recs = _replay(entries, me)
    for rec in recs:
        s["agg_n"] += 1
        s["agg"] += int(rec["kind"] != "none")
        s["big_n"] += 1
        if rec["big"]:
            s["big"] += 1
        if rec["facing"]:
            s["rz_n"] += 1
            s["bet_faced_n"] += 1
            folded = int(rec["action"] == "fold")
            s["bet_fold"] += folded
            s["any_n"] += 1
            s["any_fold"] += folded
            key = "rr" if rec["faced_raise"] else ("sbet" if rec["faced_size"] <= 0.45 else "nbet")
            s[key + "_n"] += 1
            s[key + "_fold"] += folded
            if rec["kind"] == "raise" and not rec["big"]:
                s["rz"] += 1
        else:
            key = "fbet" if rec["street"] == "flop" else "lbet"
            s[key + "_n"] += 1
            if rec["kind"] == "bet" and not rec["big"]:
                s[key] += 1

    # Showdown honesty / bluff verification
    shown = None
    for e in entries:
        if e.get("action") == "show" and e.get("bot_id") != me:
            shown = e.get("cards")
    board = _BOARDS.get(hand_num)
    if shown and len(shown) == 2 and board and len(board) == 5:
        bet_flop = any(r["kind"] != "none" and r["street"] == "flop" for r in recs)
        bet_late = any(r["kind"] != "none" and r["street"] != "flop" for r in recs)
        if bet_flop or bet_late:
            weak = int(_raw_river_strength(shown, board) < 0.40)
            if bet_flop:
                s["shown_flop_n"] += 1
                s["shown_flop_weak"] += weak
            if bet_late:
                s["shown_late_n"] += 1
                s["shown_late_weak"] += weak
            if any(r["big"] for r in recs):
                s["shown_big_n"] += 1
                s["shown_big_weak"] += weak


def _update_stats(state: dict) -> None:
    me = state.get("your_bot_id")
    cur = state.get("hand_num", 0)
    groups = {}
    for e in state.get("match_action_log") or []:
        h = e.get("hand_num")
        if h is None or h >= cur or h in _DONE_HANDS:
            continue
        groups.setdefault(h, []).append(e)
    for h, entries in groups.items():
        if not entries or entries[0].get("action") not in ("ante", "small_blind"):
            continue
        _DONE_HANDS.add(h)
        _digest(entries, me, h)


# ── Context & Exact Odds Helpers ─────────────────────────────────────────────

def _context(s: dict) -> dict:
    me = s["your_bot_id"]
    opp = next((p for p in s.get("players", []) if p.get("bot_id") != me), None)
    bb = max(1, int((s.get("blinds") or {}).get("bb", 50)))
    ante = int((s.get("blinds") or {}).get("ante", 0))
    my_stack = int(s["your_stack"])
    my_bet = int(s["your_bet_this_street"])
    my_total = my_stack + my_bet

    if opp is None:
        opp_stack, opp_bet, opp_all_in, opp_id = my_stack, 0, False, "opp"
    else:
        opp_stack = int(opp.get("stack", my_stack))
        opp_bet = int(opp.get("bet_this_street", 0))
        opp_all_in = bool(opp.get("is_all_in")) or opp_stack <= 0
        opp_id = opp.get("bot_id", "opp")

    opp_total = opp_stack + opp_bet
    unmatched = max(0, opp_bet - my_total)
    return {
        "me": me,
        "opp_id": opp_id,
        "bb": bb,
        "ante": ante,
        "stack": my_stack,
        "opp_stack": opp_stack,
        "my_bet": my_bet,
        "owed": min(int(s["amount_owed"]), my_stack),
        "pot": max(1, int(s["pot"]) - unmatched),
        "eff": min(my_total, opp_total),
        "in_play": my_stack + opp_stack + int(s["pot"]),
        "opp_all_in": opp_all_in,
    }


def _showdown_odds(c: dict) -> float:
    risk = min(c["stack"], c["eff"] - c["my_bet"])
    if risk <= 0:
        return 0.0
    final_pot = c["pot"] + risk + max(0, risk - c["owed"])
    return risk / float(max(1, final_pot))


def _life_premium(c: dict, shove_rate: float) -> float:
    """Extra equity required to gamble for tournament life while blinds are small."""
    in_play_bb = c["in_play"] / float(c["bb"])
    base = _interp(LIFE_PREMIUM, in_play_bb)
    share = (c["stack"] + c["my_bet"]) / float(max(1, c["in_play"]))
    uneven = 1.0 - 0.7 * min(1.0, abs(share - 0.5) / 0.5)
    often = min(1.0, max(0.25, shove_rate / 0.60))
    return base * uneven * often


def _commit_premium(c: dict, risk: int) -> float:
    """Extra equity asked for when putting a deep stack in relative to the pot."""
    before = max(1, c["pot"] - c["owed"])
    overall = _rate(STATS["agg"], STATS["agg_n"], 0.30, 6)
    fade = min(1.0, max(0.0, 1.0 - (overall - 0.30) / 0.30))
    bb_scale = min(1.0, (risk / float(c["bb"])) / 80.0)
    pot_scale = min(1.0, (risk / float(before)) / 1.5)
    return COMMIT_PREMIUM * fade * bb_scale * pot_scale


def _nash_push_frac(eff_bb: float, ante_bb: float) -> float:
    base = _interp(NASH_PUSH_NO_ANTE, eff_bb)
    if ante_bb > 0:
        base = min(1.0, base + min(0.14, ante_bb * 1.15))
    return base


def _nash_call_frac(eff_bb: float, ante_bb: float) -> float:
    base = _interp(NASH_CALL_NO_ANTE, eff_bb)
    if ante_bb > 0:
        base = min(1.0, base + min(0.06, ante_bb * 0.45))
    return base


def _shove_ev_positive(cards: list, eff_bb: float, ante_bb: float, call_rate: float) -> bool:
    """Exact chip EV comparison: SB open-shove vs folding (accounts for antes & call_rate)."""
    cr = max(0.05, min(1.0, call_rate))
    eq = _eq_vs_top(cards, cr)
    ev_diff = (1.0 - cr) * (1.5 + 2.0 * ante_bb) + cr * (eq * (2.0 * eff_bb + 2.0 * ante_bb) - eff_bb + 0.5)
    return ev_diff > 0.05


# ── Preflop Engine ───────────────────────────────────────────────────────────

def _preflop(s: dict, c: dict) -> tuple:
    cards = s["your_cards"]
    cls_idx = _hand_class(cards[0], cards[1])
    st = STATS
    bb, owed, pot = c["bb"], c["owed"], c["pot"]
    eff_bb = c["eff"] / float(bb)
    ante_bb = c["ante"] / float(bb)
    log = [a for a in (s.get("action_log") or []) if a.get("action") in VOLUNTARY]
    mine = [a["action"] for a in log if a.get("bot_id") == c["me"]]
    theirs = [a["action"] for a in log if a.get("bot_id") != c["me"]]
    i_am_sb = s.get("sb_seat") == s.get("seat_to_act")
    is_seven_two = {cards[0][0], cards[1][0]} == {"7", "2"}
    dbg = {"hand": NAMES[cls_idx], "eff_bb": round(eff_bb, 1)}

    all_shoves = st["open_shove"] + st["limp_shove"] + st["raise_shove"]
    all_spots = st["open_n"] + st["limp_n"] + st["raise_n"]
    shove_overall = _rate(all_shoves, all_spots, 0.06, 4)
    premium = _life_premium(c, shove_overall)

    # 0) Smart 7-2 Side-Prize Shove: jam when ahead in chips (cannot bust!) OR
    #    eff_bb <= 35 (high pot/blind reward), dodging suicidal jams into stations.
    if SEVEN_TWO_SHOVE and is_seven_two and not c["opp_all_in"] and not mine:
        any_fold = _rate(st["any_fold"], st["any_n"], 0.35, 6)
        shove_call_rate = (st["shove_call"] / float(st["shove_n"])) if st["shove_n"] >= 3 else 0.40
        is_sticky = (st["any_n"] >= 8 and any_fold < 0.18) or (st["shove_n"] >= 3 and shove_call_rate > 0.55)
        has_lead = c["stack"] >= c["opp_stack"] * 1.15
        facing_small = (owed <= bb * 3.5)
        if facing_small and (has_lead or (eff_bb <= 35.0 and not is_sticky)):
            dbg["spot"] = "72_shove"
            return {"action": "all_in"}, dbg

    # 1) Facing an all-in or stack-committing raise
    raised = bool(theirs) and theirs[-1] in ("raise", "all_in")
    if owed > 0 and (c["opp_all_in"] or (raised and owed >= 0.35 * c["stack"])):
        go = {"action": "call"} if c["opp_all_in"] else {"action": "all_in"}
        odds = _showdown_odds(c)
        if not raised:
            eq = _eq_vs_top(cards, 1.0)
            dbg.update(spot="vs_blind_all_in", eq=round(eq, 3), need=round(odds, 3))
            return (go if eq >= odds else {"action": "fold"}), dbg

        # Opponent limped SB and shoved over our BB isolation raise!
        if not i_am_sb and mine == ["raise"] and len(theirs) >= 2 and theirs[0] == "call":
            x = _rate(st["opp_limp_shove"], st["opp_limp_iso_n"], 0.14 if eff_bb > 15 else 0.25, 4)
            need = odds + premium
            eq = _eq_vs_top(cards, x)
            dbg.update(spot="vs_limp_shove", rng=round(x, 2), eq=round(eq, 3), need=round(need, 3))
            return (go if eq >= need else {"action": "fold"}), dbg

        if not mine and not i_am_sb and eff_bb <= 25:
            usual = _nash_push_frac(min(eff_bb, 20.0), ante_bb)
            x = _rate(st["sopen_shove"], st["sopen_n"], usual, 4)
            eq = _eq_vs_top(cards, x)
            need = odds + (premium if eff_bb > 15 else 0.0)
            dbg.update(spot="vs_short_shove", rng=round(x, 2), eq=round(eq, 3), need=round(need, 3))
            return (go if eq >= need else {"action": "fold"}), dbg

        if not mine and not i_am_sb:
            x = _rate(st["open_shove"] - st["sopen_shove"], st["open_n"] - st["sopen_n"], 0.06, 4)
        elif mine == ["call"]:
            x = _rate(st["limp_shove"], st["limp_n"], 0.32 if eff_bb <= 12 else 0.08, 4)
        elif mine == ["raise"]:
            prior = 0.25 if eff_bb <= 15 else (0.12 if eff_bb <= 30 else 0.05)
            x = _rate(st["raise_shove"], st["raise_n"], prior, 4)
        else:
            x = 0.10 if eff_bb <= 20 else 0.05

        x = max(x, 0.85 * shove_overall)
        need = odds + premium
        eq = _eq_vs_top(cards, x)
        dbg.update(spot="vs_shove", rng=round(x, 2), eq=round(eq, 3), need=round(need, 3))
        return (go if eq >= need else {"action": "fold"}), dbg

    # 2) Small Blind (Button) first to act
    if i_am_sb and not mine and not theirs:
        jam_back = _rate(st["raise_shove"] + st["limp_shove"], st["raise_n"] + st["limp_n"], 0.05, 4)
        if eff_bb <= PUSH_FOLD_BB:
            eq_calls = _nash_call_frac(eff_bb, ante_bb)
            calls = _rate(st["shove_call"], st["shove_n"], eq_calls, 4)
            dbg.update(spot="push_fold", calls=round(calls, 2))
            if _shove_ev_positive(cards, eff_bb, ante_bb, calls):
                return {"action": "all_in"}, dbg
            return {"action": "fold"}, dbg

        if jam_back > 0.50:
            eq = _eq_vs_top(cards, max(jam_back, 0.85 * shove_overall))
            dbg.update(spot="vs_jammer", eq=round(eq, 3), need=round(0.5 + premium, 3))
            if eq >= 0.50 + premium:
                return {"action": "raise", "amount": 2 * bb}, dbg
            return {"action": "fold"}, dbg

        fold_to_open = _rate(st["raise_fold"], st["raise_n"], 0.40, 6)
        open_share = 0.98 if fold_to_open > 0.50 else (0.82 if ante_bb == 0 else 0.88)
        dbg.update(spot="open", fto=round(fold_to_open, 2))
        if RANK_END[cls_idx] <= open_share or (SEVEN_TWO_SHOVE and is_seven_two and fold_to_open >= 0.35):
            size = 2.25 if eff_bb >= 25 else 2.0
            return {"action": "raise", "amount": int(size * bb)}, dbg
        return {"action": "fold"}, dbg

    # 3) Big Blind facing a limp (owed == 0)
    if owed == 0:
        opp_lsh = _rate(st["opp_limp_shove"], st["opp_limp_iso_n"], 0.08, 4)
        dbg.update(spot="vs_limp")
        if eff_bb <= PUSH_FOLD_BB:
            if RANK_END[cls_idx] <= (0.36 if ante_bb == 0 else 0.42):
                return {"action": "all_in"}, dbg
        else:
            iso_cut = 0.20 if opp_lsh >= 0.18 else 0.32
            if RANK_END[cls_idx] <= iso_cut:
                mult = 2.5 if eff_bb <= 22 else 3.0
                return {"action": "raise", "amount": int(mult * bb)}, dbg
        return {"action": "check"}, dbg

    odds = owed / float(pot + owed)

    # 4) Big Blind facing an opening raise
    if not mine and theirs == ["raise"]:
        x = max(0.12, _rate(st["open_raise"] + st["open_shove"], st["open_n"], 0.72, 6))
        eq = _eq_vs_top(cards, x)
        dbg.update(spot="vs_open", rng=round(x, 2), eq=round(eq, 3), odds=round(odds, 3))
        if eff_bb <= 20:
            if eq >= 0.56:
                return {"action": "all_in"}, dbg
            if eq >= odds + 0.06:
                return {"action": "call"}, dbg
            return {"action": "fold"}, dbg
        if RANK_END[cls_idx] <= 0.10 or eq >= 0.63:
            return {"action": "raise", "amount": int(3.0 * s["current_bet"])}, dbg
        if eq >= odds + 0.055:
            return {"action": "call"}, dbg
        return {"action": "fold"}, dbg

    # 5) We opened and faced a 3-bet
    if mine == ["raise"]:
        x = max(0.05, _rate(st["raise_3bet"] + st["raise_shove"], st["raise_n"], 0.20, 6))
        eq = _eq_vs_top(cards, x)
        dbg.update(spot="vs_3bet", rng=round(x, 2), eq=round(eq, 3), odds=round(odds, 3))
        if eq >= 0.62:
            if eff_bb <= 45:
                return {"action": "all_in"}, dbg
            return {"action": "raise", "amount": int(2.4 * s["current_bet"])}, dbg
        if eq >= odds + 0.04:
            return {"action": "call"}, dbg
        return {"action": "fold"}, dbg

    # 6) Raising war (4-bet+)
    eq = _eq_vs_top(cards, 0.05)
    dbg.update(spot="raise_war", eq=round(eq, 3), odds=round(odds, 3))
    if eq >= 0.60:
        return {"action": "all_in"}, dbg
    if eq >= odds + 0.05:
        return {"action": "call"}, dbg
    return {"action": "fold"}, dbg


# ── Postflop Continuous Range Reading & Unified Bayesian Equity ──────────────

def _looseness(kind: str, big: bool, street: str, size: float) -> float:
    """
    Continuous range width behind an opponent's postflop bet/raise (1.0 = any hand,
    0.08 = nutted). Anchored to MDF before data, then adapts smoothly to empirical
    betting frequencies and showdown honesty/bluff rates.
    """
    st = STATS
    size = max(0.1, size)
    base = min(1.0, (1.0 + 2.0 * size) / ((1.0 + size) ** 2))
    overall = _rate(st["agg"], st["agg_n"], 0.30, 6)
    boost = min(3.0, max(0.6, overall / 0.30))

    if big:
        f = _rate(st["big"], st["big_n"], min(0.60, 0.10 * boost), 5)
        if f >= 0.20:
            loose = base + (1.0 - base) * min(1.0, (f - 0.20) / 0.40)
        else:
            loose = base * max(0.25, f / 0.20)
        weak_share = _rate(st["shown_big_weak"], st["shown_big_n"], 0.20, 3)
        if weak_share > 0.20:
            target = max(base, 0.80)
            if target > loose:
                loose += (target - loose) * min(1.0, (weak_share - 0.20) / 0.30)
        else:
            honest = max(LOOSE_MIN, f)
            if loose > honest:
                loose = honest + (loose - honest) * (weak_share / 0.20)
        return loose

    if kind == "raise":
        f, usual = _rate(st["rz"], st["rz_n"], min(0.60, 0.12 * boost), 6), 0.30
    elif street == "flop":
        f, usual = _rate(st["fbet"], st["fbet_n"], min(0.90, 0.50 * (boost ** 0.5)), 6), 0.50
    else:
        f, usual = _rate(st["lbet"], st["lbet_n"], min(0.90, 0.40 * (boost ** 0.5)), 6), 0.40

    if f >= usual:
        loose = base + (1.0 - base) * min(1.0, (f - usual) / (0.85 - usual))
    else:
        loose = base * max(0.25, f / usual)

    if street == "flop" and kind == "bet" and not big:
        weak_share = _rate(st["shown_flop_weak"], st["shown_flop_n"], 0.30, 2)
    else:
        weak_share = _rate(st["shown_late_weak"], st["shown_late_n"], 0.30, 2)

    honest = max(LOOSE_MIN, f)
    if loose > honest:
        loose = honest + (loose - honest) * min(1.0, weak_share / 0.30)
    return loose


def _read_hand_postflop(s: dict, c: dict) -> dict:
    """Summarize preflop range width, continuous postflop looseness, and street context."""
    st = STATS
    me = c["me"]
    street = s["street"]
    loose = 1.0
    was_aggressor = False
    my_raise_seen = False
    pre_keep = 0.95
    hero_bets_street = 0
    opp_checked_street = False
    opp_called_post = 0

    for a in s.get("action_log") or []:
        act = a.get("action")
        ast = a.get("street")
        who = a.get("bot_id")
        if ast == "preflop" and act in ("raise", "all_in"):
            if who == me:
                was_aggressor = my_raise_seen = True
            else:
                was_aggressor = False
                if my_raise_seen:
                    f = _rate(st["raise_3bet"] + st["raise_shove"], st["raise_n"], 0.20, 6)
                elif s.get("sb_seat") != s.get("seat_to_act"):
                    f = _rate(st["open_raise"] + st["open_shove"], st["open_n"], 0.72, 6)
                else:
                    f = 0.30
                pre_keep = min(pre_keep, max(0.10, f))
        elif ast in ("flop", "turn", "river"):
            if who != me and act == "call":
                opp_called_post += 1
            if ast == street:
                if who == me and act in ("raise", "all_in"):
                    hero_bets_street += 1
                elif who != me and act == "check":
                    opp_checked_street = True

    recs = [r for r in _replay(s.get("action_log") or [], me) if r["kind"] != "none"]
    for idx, r in enumerate(recs):
        pending = c["owed"] > 0 and idx == len(recs) - 1 and r["street"] == street
        size, big = r["size"], r["big"]
        if pending:
            size = c["owed"] / float(max(1, c["pot"] - c["owed"]))
            big = size > BIG_BET or (c["opp_all_in"] and size > 0.75)
        li = _looseness(r["kind"], big, r["street"], size)
        loose *= li if pending else (li ** 0.5)

    loose = max(LOOSE_MIN, min(1.0, loose))
    return {
        "loose": loose,
        "pre_keep": pre_keep,
        "was_aggressor": was_aggressor,
        "hero_bets_street": hero_bets_street,
        "opp_checked_street": opp_checked_street,
        "opp_called_post": opp_called_post,
        "opp_bet_count": len(recs),
    }


def _their_bet_rate(street: str) -> float:
    st = STATS
    if street == "flop":
        return _rate(st["fbet"], st["fbet_n"], 0.50, 6)
    return _rate(st["lbet"], st["lbet_n"], 0.40, 6)


def _bayesian_postflop_equity(
    hole: list,
    board: list,
    ctx: dict,
    street: str,
    deadline: float,
    sims: int = POSTFLOP_SIMS,
) -> tuple:
    """
    Clean order-statistic Bayesian combo equity (zero bluff_floor inflation):
    - Maps continuous `loose` directly to the exact best-of-k density bp^(k-1).
    - Applies tight preflop range carry-over (0.08 out-of-range) and trims air
      when opponent already called a previous postflop street.
    - Exact 990-combo river evaluation; time-bounded flop/turn rollouts.
    """
    known = set(hole) | set(board)
    deck_recs = [(card, r, s) for c, card, r, s in CARD_RECORDS if c not in known]
    deck_cards = [rec[0] for rec in deck_recs]
    n_deck = len(deck_cards)

    hero = [CARD[c] for c in hole]
    brd = [CARD[c] for c in board]
    n_board = len(brd)
    ev = eval7.evaluate

    combos = list(itertools.combinations(deck_recs, 2))
    n_combos = len(combos)

    eval_buf = brd + [None, None]
    scores = [0] * n_combos
    for i, ((c1, _, _), (c2, _, _)) in enumerate(combos):
        eval_buf[n_board] = c1
        eval_buf[n_board + 1] = c2
        scores[i] = ev(eval_buf)

    order = sorted(range(n_combos), key=scores.__getitem__)
    board_pct = [0.0] * n_combos
    denom = max(1, n_combos - 1)
    for rank_idx, combo_idx in enumerate(order):
        board_pct[combo_idx] = rank_idx / float(denom)

    loose = ctx["loose"]
    k = min(MAX_K, (2.0 - loose) / loose)
    power = max(0.0, k - 1.0)
    pre_keep = ctx["pre_keep"]
    checked_cap = ctx["opp_checked_street"] and ctx["opp_bet_count"] == 0
    cap_mult = max(0.30, 1.0 - 0.75 * _their_bet_rate(street)) if checked_cap else 1.0
    opp_called = ctx["opp_called_post"] > 0 and _rate(STATS["bet_fold"], STATS["bet_faced_n"], 0.35, 6) > 0.18
    out_pre_w = 0.08 if n_board == 3 else 0.14

    weights = [1.0] * n_combos
    for i, ((_, r1, s1), (_, r2, s2)) in enumerate(combos):
        pkey = (r1, r2, s1 == s2) if r1 >= r2 else (r2, r1, s1 == s2)
        w_pre = 1.0 if KEY_RANK_END[pkey] <= pre_keep else out_pre_w

        bp = board_pct[i]
        if power > 0.01:
            w_post = bp ** power
        elif checked_cap and bp >= 0.80:
            w_post = cap_mult
        else:
            w_post = 1.0

        if opp_called and bp < 0.28:
            w_post *= 0.35

        weights[i] = w_pre * w_post

    if n_board == 5:
        hero_score = ev(hero + brd)
        w_wins = 0.0
        w_total = 0.0
        raw_wins = 0.0
        for i, opp_score in enumerate(scores):
            if hero_score > opp_score:
                out = 1.0
            elif hero_score == opp_score:
                out = 0.5
            else:
                out = 0.0
            w = weights[i]
            w_wins += w * out
            w_total += w
            raw_wins += out
        return (w_wins / w_total if w_total > 0 else 0.5), (raw_wins / n_combos)

    cum_w = list(itertools.accumulate(weights))
    if cum_w[-1] <= 0.0:
        return 0.5, None

    sampled = random.choices(combos, cum_weights=cum_w, k=sims)
    clock = time.perf_counter
    randrange = random.randrange
    wins = 0.0
    n = 0

    if n_board == 4:
        h_buf = hero + brd + [None]
        o_buf = [None, None] + brd + [None]
        for (c1, _, _), (c2, _, _) in sampled:
            if n and (n & 63) == 0 and clock() > deadline:
                break
            r1 = deck_cards[randrange(n_deck)]
            while r1 == c1 or r1 == c2:
                r1 = deck_cards[randrange(n_deck)]
            h_buf[6] = r1
            o_buf[0] = c1
            o_buf[1] = c2
            o_buf[6] = r1
            hs = ev(h_buf)
            os_ = ev(o_buf)
            if hs > os_:
                wins += 1.0
            elif hs == os_:
                wins += 0.5
            n += 1
    else:
        h_buf = hero + brd + [None, None]
        o_buf = [None, None] + brd + [None, None]
        for (c1, _, _), (c2, _, _) in sampled:
            if n and (n & 63) == 0 and clock() > deadline:
                break
            r1 = deck_cards[randrange(n_deck)]
            while r1 == c1 or r1 == c2:
                r1 = deck_cards[randrange(n_deck)]
            r2 = deck_cards[randrange(n_deck)]
            while r2 == c1 or r2 == c2 or r2 == r1:
                r2 = deck_cards[randrange(n_deck)]
            h_buf[5] = r1
            h_buf[6] = r2
            o_buf[0] = c1
            o_buf[1] = c2
            o_buf[5] = r1
            o_buf[6] = r2
            hs = ev(h_buf)
            os_ = ev(o_buf)
            if hs > os_:
                wins += 1.0
            elif hs == os_:
                wins += 0.5
            n += 1

    return (wins / n if n else 0.5), None


def _is_wet(board: list) -> bool:
    flop = board[:3]
    ranks = sorted(RANK_VAL[c[0]] for c in flop)
    if len(set(ranks)) < 3:
        return False
    return len(set(c[1] for c in flop)) < 3 or (ranks[2] - ranks[0] <= 4)


def _bet_plan(s: dict, c: dict, eq: float, raw_eq, ctx: dict, deep: bool, dbg: dict) -> dict:
    """
    Unreadable situation-based betting:
    - Size depends on street, board texture, checked-to, and opponent fold rate
      (NEVER hand strength).
    - Sizes UP uniformly against stations (fold_all < 0.45) and DOWN uniformly
      against Nits (fold_all > 0.55) so bluffs print chips at minimal risk.
    """
    st = STATS
    me = c["me"]
    street = s["street"]
    pot, stack = c["pot"], c["stack"]
    acts = [a for a in (s.get("action_log") or []) if a.get("action") in VOLUNTARY]
    checked_to = ctx["opp_checked_street"]
    earlier = {"turn": "flop", "river": "turn"}.get(street)
    barrel = any(
        a.get("street") == earlier and a.get("bot_id") == me and a.get("action") in ("raise", "all_in")
        for a in acts
    )

    if street == "flop":
        frac = 0.34 if checked_to else 0.50
        if _is_wet(s["community_cards"]):
            frac += 0.13
    elif street == "turn":
        frac = 0.66
    else:
        frac = 0.76

    fold_guess = _rate(st["any_fold"], st["any_n"], 0.40, 6)
    fold_all = _rate(st["bet_fold"], st["bet_faced_n"], fold_guess, 4)
    if frac <= 0.45:
        fold_here = _rate(st["sbet_fold"], st["sbet_n"], fold_guess, 4)
    else:
        fold_here = _rate(st["nbet_fold"], st["nbet_n"], fold_guess, 4)
    dbg["ftb"] = round(fold_here, 2)

    # Uniform sizing adjustments: bigger vs stations, cheaper vs Nits
    if fold_all < 0.45:
        frac *= 1.0 + 0.65 * min(1.0, (0.45 - fold_all) / 0.25)
    elif fold_all > 0.54:
        frac *= max(0.68, 1.0 - 0.85 * (fold_all - 0.54))

    value_at = 0.53 + 0.12 * fold_all

    # Push value hard when holding 7-2 postflop with a real hand
    is_seven_two = {s["your_cards"][0][0], s["your_cards"][1][0]} == {"7", "2"}
    if SEVEN_TWO_SHOVE and is_seven_two and eq >= 0.56:
        bet = int(max(frac, 0.90) * pot)
        if bet >= 0.55 * stack:
            return {"action": "all_in"}
        return {"action": "raise", "amount": s["current_bet"] + bet}

    if eq >= value_at:
        if not checked_to:
            their_bet = _their_bet_rate(street)
            trap = 0.60 * (their_bet - 0.46) + (0.10 if street != "river" else 0.0)
            if random.random() < min(0.45, max(0.0, trap)):
                dbg["trap"] = 1
                return {"action": "check"}
        bet = int(frac * pot * random.uniform(0.96, 1.04))
        if bet >= 0.60 * stack and (not deep or eq >= value_at + 2.0 * _commit_premium(c, stack)):
            return {"action": "all_in"}
        return {"action": "raise", "amount": s["current_bet"] + bet}

    # Balanced bluff using the exact same situation-based bet size
    bet = int(frac * pot * random.uniform(0.96, 1.04))
    pays = frac / (1.0 + frac) - 0.05
    if fold_here >= max(0.16, pays) and bet < 0.58 * stack and not (street == "river" and eq >= 0.35):
        if checked_to:
            base = {"flop": 0.50, "turn": 0.44, "river": 0.30}[street]
        elif street == "flop":
            base = 0.50 if ctx["was_aggressor"] else 0.20
        else:
            base = 0.30 if barrel else 0.16

        if random.random() < min(0.88, base * min(2.0, fold_here / 0.38)):
            chk_raw = raw_eq if raw_eq is not None else _raw_mc_equity(s["your_cards"], s["community_cards"], sims=110)
            if chk_raw < 0.39 or fold_here >= 0.56:
                dbg["bluff"] = 1
                return {"action": "raise", "amount": s["current_bet"] + bet}

    return {"action": "check"}


def _postflop(s: dict, c: dict, deadline: float) -> tuple:
    st = STATS
    pot, owed, stack = c["pot"], c["owed"], c["stack"]
    street = s["street"]
    ctx = _read_hand_postflop(s, c)
    eq, raw_eq = _bayesian_postflop_equity(s["your_cards"], s["community_cards"], ctx, street, deadline)

    dbg = {
        "eq": round(eq, 3),
        "loose": round(ctx["loose"], 2),
        "pre": round(ctx["pre_keep"], 2),
    }
    deep = (stack / float(c["bb"])) > 40.0

    if owed == 0:
        if c["opp_all_in"]:
            return {"action": "check"}, dbg
        return _bet_plan(s, c, eq, raw_eq, ctx, deep, dbg), dbg

    odds = owed / float(pot + owed)
    premium = _commit_premium(c, min(owed, stack))
    size_now = owed / float(max(1, pot - owed))
    dbg["odds"] = round(odds, 3)
    dbg["prem"] = round(premium, 3)

    if c["opp_all_in"] or owed >= stack:
        return ({"action": "call"} if eq >= odds + 0.02 + premium else {"action": "fold"}), dbg

    # Value raising with escalation cap (prevents 150 BB one-pair click-back wars)
    hero_bets_st = ctx["hero_bets_street"]
    raise_at = 0.90 if hero_bets_st >= 2 else (0.85 if hero_bets_st == 1 else 0.78)
    if eq >= raise_at:
        raise_to = s["current_bet"] + int(0.75 * (pot + owed) * random.uniform(0.95, 1.05))
        if raise_to - c["my_bet"] >= 0.50 * stack:
            if not deep or eq >= raise_at + _commit_premium(c, stack):
                return {"action": "all_in"}, dbg
            return {"action": "call"}, dbg
        return {"action": "raise", "amount": raise_to}, dbg

    # Defend slightly wider vs cheap flop stabs (<= 0.45x pot); slightly tighter vs passive Nits
    overall_agg = _rate(st["agg"], st["agg_n"], 0.30, 6)
    if street == "river":
        margin = 0.0
    elif street == "flop" and size_now <= 0.45 and overall_agg >= 0.25:
        margin = -0.01
    else:
        margin = 0.02
    if overall_agg < 0.20:
        margin += 0.03

    # Exploitative bluff-raise against frequent flop/turn c-bettors who fold to raises
    if (
        street != "river"
        and c["my_bet"] == 0
        and size_now <= BIG_BET
        and eq < odds + margin + premium + 0.14
        and _their_bet_rate(street) >= 0.56
    ):
        fold_guess = _rate(st["any_fold"], st["any_n"], 0.40, 6)
        fold_to_raise = _rate(st["rr_fold"], st["rr_n"], min(0.60, fold_guess + 0.05), 3)
        raise_to = s["current_bet"] + int(0.75 * (pot + owed))
        if (
            fold_to_raise >= 0.25
            and (raise_to - c["my_bet"]) < 0.40 * stack
            and random.random() < min(0.42, 0.22 * fold_to_raise / 0.42)
        ):
            chk_raw = raw_eq if raw_eq is not None else _raw_mc_equity(s["your_cards"], s["community_cards"], sims=100)
            if chk_raw < 0.39:
                dbg["bluff"] = 2
                return {"action": "raise", "amount": raise_to}, dbg

    if eq >= odds + margin + premium:
        return {"action": "call"}, dbg
    return {"action": "fold"}, dbg


# ── Safety Wrapper & Entry Point ─────────────────────────────────────────────

def _legal(act, s: dict) -> dict:
    """Guarantee a strictly legal move; never accidentally call via check."""
    owed = int(s.get("amount_owed") or 0)
    stack = int(s.get("your_stack") or 0)
    my_bet = int(s.get("your_bet_this_street") or 0)
    passive = {"action": "check"} if owed == 0 else {"action": "fold"}

    if not isinstance(act, dict):
        return passive

    a = act.get("action")
    out = None
    if a in ("fold", "check"):
        out = passive
    elif a == "call":
        out = {"action": "call"} if owed > 0 else {"action": "check"}
    elif a in ("raise", "all_in"):
        me = s.get("your_bot_id")
        opp = next((p for p in s.get("players", []) if p.get("bot_id") != me), None)
        opp_total = (int(opp["stack"]) + int(opp["bet_this_street"])) if opp else (stack + my_bet)
        opp_done = (opp is None) or bool(opp.get("is_all_in")) or int(opp.get("stack", 0)) <= 0
        if opp_done or stack <= owed:
            out = {"action": "call"} if owed > 0 else {"action": "check"}
        elif a == "all_in":
            out = {"action": "all_in"}
        else:
            amount = int(act.get("amount") or 0)
            min_r = int(s.get("min_raise_to") or 0)
            amount = max(min_r, min(amount, opp_total))
            if amount - my_bet >= stack:
                out = {"action": "all_in"}
            else:
                out = {"action": "raise", "amount": amount}

    if out is None:
        return passive
    if "debug" in act and isinstance(act["debug"], dict):
        out["debug"] = act["debug"]
    return out


def _think(s: dict, t0: float) -> dict:
    _reset_if_new_match(s)
    try:
        _update_stats(s)
    except Exception:
        pass
    c = _context(s)
    hnum = s.get("hand_num", 0)
    _BOARDS[hnum] = list(s.get("community_cards") or [])
    if hnum not in _DEPTH:
        _DEPTH[hnum] = c["eff"] / float(c["bb"])

    if s["street"] == "preflop":
        act, dbg = _preflop(s, c)
    else:
        act, dbg = _postflop(s, c, t0 + THINK_S)

    dbg["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    act["debug"] = dbg
    return act


def decide(game_state: dict) -> dict:
    t0 = time.perf_counter()
    try:
        if game_state.get("type") == "warmup":
            _eq_vs_top(["As", "Kh"], 0.70, sims=250)
            _eq_vs_top(["7s", "2d"], 0.50, sims=250)
            return {"action": "check"}
        act = _think(game_state, t0)
    except Exception:
        act = None
    try:
        return _legal(act, game_state)
    except Exception:
        try:
            return {"action": "check"} if int(game_state.get("amount_owed") or 0) == 0 else {"action": "fold"}
        except Exception:
            return {"action": "fold"}