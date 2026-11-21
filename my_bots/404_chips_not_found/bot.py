"""
404 Chips Not Found — Version 6 (Bayesian Range Engine + Freezeout Discipline)
Upgrades over Version 5:
1. Bayesian Range-Weighted Equity: scores all ~1,000 opponent hands on the board
   and weights them by opponent archetype + current-hand actions (exact on river,
   900 range-sampled rollouts on flop/turn, ~5-10 ms per decision).
2. Freezeout Patience vs. Maniacs: faster detection (hand 4+) and stack-depth
   calling floors (0.65 eq deep -> 0.58 mid -> 0.52 shallow) to avoid early flips.
3. Continuous fold-rate blending inside BALANCED to exploit semi-nits/semi-stations.
4. Keeps exact raw-equity check (< 0.40) for Biggest Bluffer + safe 7-2 hook.
"""

import itertools
import random
import eval7

ALL_CARD_STRS = [r + s for r in "23456789TJQKA" for s in "shdc"]
RANK_ORDER = {r: i for i, r in enumerate("23456789TJQKA")}
CARD_RECORDS = [(c, eval7.Card(c), RANK_ORDER[c[0]], c[1]) for c in ALL_CARD_STRS]
CARD_MAP = {c: card for c, card, _, _ in CARD_RECORDS}


def _build_preflop_percentiles() -> dict:
    """Rank all 169 starting hand classes from 0.0 (72o) to 1.0 (AA) at import time."""
    hi_pts = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]
    scored = []
    for hi in range(13):
        for lo in range(hi + 1):
            for suited in (False, True):
                if hi == lo and suited:
                    continue
                if hi == lo:
                    score = max(5.0, hi_pts[hi] * 2.0) + hi * 0.05
                else:
                    gap = hi - lo - 1
                    gap_pen = (0.0, -1.0, -2.0, -4.0)[gap] if gap < 4 else -5.0
                    bonus = 1.0 if (gap <= 1 and hi < 10) else 0.0
                    score = hi_pts[hi] + (2.0 if suited else 0.0) + gap_pen + bonus + lo * 0.22
                scored.append((score, (hi, lo, suited)))
    scored.sort(key=lambda x: x[0])
    total = max(1, len(scored) - 1)
    return {key: idx / total for idx, (_, key) in enumerate(scored)}


PREFLOP_RANK_PCT = _build_preflop_percentiles()

# Match-long memory (resets automatically per match)
HAND_HISTORY = {}          # {hand_num: list of action dicts}
HAND_BOARDS = {}           # {hand_num: [5 river board card strs]}
SHOWDOWN_BLUFF_CACHE = {}  # {hand_num: bool}
PREFLOP_EQ_CACHE = {}      # {(hi, lo, suited): float}


def raw_river_equity(your_cards: list, community_cards: list) -> float:
    """Exact unweighted river equity across all 990 opponent hands."""
    hero = [CARD_MAP[c] for c in your_cards]
    board = [CARD_MAP[c] for c in community_cards]
    hero_score = eval7.evaluate(hero + board)

    used = set(your_cards) | set(community_cards)
    deck = [card for c, card, _, _ in CARD_RECORDS if c not in used]

    opp_buf = board + [None, None]
    wins = 0.0
    total = 0
    for c1, c2 in itertools.combinations(deck, 2):
        opp_buf[5] = c1
        opp_buf[6] = c2
        opp_score = eval7.evaluate(opp_buf)
        if hero_score > opp_score:
            wins += 1.0
        elif hero_score == opp_score:
            wins += 0.5
        total += 1
    return wins / total if total else 0.5


def monte_carlo_raw(your_cards: list, community_cards: list, sims: int = 200) -> float:
    """Fast unweighted Monte Carlo equity against a random hand."""
    used = set(your_cards) | set(community_cards)
    deck = [card for c, card, _, _ in CARD_RECORDS if c not in used]

    hero = [CARD_MAP[c] for c in your_cards]
    board = [CARD_MAP[c] for c in community_cards]
    draw_count = 2 + (5 - len(board))

    wins = 0.0
    for _ in range(sims):
        sample = random.sample(deck, draw_count)
        full_board = board + sample[2:]
        h = eval7.evaluate(hero + full_board)
        o = eval7.evaluate(sample[:2] + full_board)
        if h > o:
            wins += 1.0
        elif h == o:
            wins += 0.5
    return wins / sims


def get_preflop_equity(your_cards: list) -> float:
    """Cached 1,000-sim preflop equity for clean, noise-free preflop play."""
    r0, r1 = RANK_ORDER[your_cards[0][0]], RANK_ORDER[your_cards[1][0]]
    suited = your_cards[0][1] == your_cards[1][1]
    key = (max(r0, r1), min(r0, r1), suited)
    if key not in PREFLOP_EQ_CACHE:
        PREFLOP_EQ_CACHE[key] = monte_carlo_raw(your_cards, [], sims=1000)
    return PREFLOP_EQ_CACHE[key]


def parse_current_hand_context(game_state: dict, my_id: str) -> dict:
    """Summarize what the opponent has done so far in THIS hand."""
    opp_pre = "check"
    opp_post_bets = 0
    opp_post_calls = 0
    opp_post_checks = 0

    for e in game_state.get("action_log") or []:
        if e.get("bot_id") == my_id:
            continue
        act = e.get("action")
        st = e.get("street")
        if act in ("small_blind", "big_blind", "ante", "show"):
            continue
        if st == "preflop":
            if act in ("raise", "all_in"):
                opp_pre = "raise"
            elif act == "call" and opp_pre != "raise":
                opp_pre = "call"
        else:
            if act in ("raise", "all_in"):
                opp_post_bets += 1
            elif act == "call":
                opp_post_calls += 1
            elif act == "check":
                opp_post_checks += 1

    return {
        "opp_pre": opp_pre,
        "opp_post_bets": opp_post_bets,
        "opp_post_calls": opp_post_calls,
        "opp_post_checks": opp_post_checks,
    }


def range_weighted_postflop_equity(
    your_cards: list,
    community_cards: list,
    hand_ctx: dict,
    profile: dict,
    sims: int = 900,
) -> tuple:
    """
    Bayesian range-weighted postflop equity:
    1. Scores all ~1,000 remaining opponent hole-card combos on the current board.
    2. Weights each combo by preflop range + postflop action line + opponent style.
    3. Exact weighted enumeration on the river; 900 range-sampled rollouts on flop/turn.
    Returns (range_equity, exact_raw_equity_or_None).
    """
    used = set(your_cards) | set(community_cards)
    deck_recs = [(card, r, s) for c, card, r, s in CARD_RECORDS if c not in used]
    deck_cards = [rec[0] for rec in deck_recs]
    n_deck = len(deck_cards)

    hero = [CARD_MAP[c] for c in your_cards]
    board = [CARD_MAP[c] for c in community_cards]
    n_board = len(board)

    combos = list(itertools.combinations(deck_recs, 2))
    n_combos = len(combos)

    # 1. Evaluate every possible opponent combo on the current board (5, 6, or 7 cards)
    eval_buf = board + [None, None]
    scores = [0] * n_combos
    for i, ((c1, _, _), (c2, _, _)) in enumerate(combos):
        eval_buf[n_board] = c1
        eval_buf[n_board + 1] = c2
        scores[i] = eval7.evaluate(eval_buf)

    order = sorted(range(n_combos), key=scores.__getitem__)
    board_pct = [0.0] * n_combos
    denom = max(1, n_combos - 1)
    for rank_idx, combo_idx in enumerate(order):
        board_pct[combo_idx] = rank_idx / denom

    # 2. Compute posterior probability weight for each opponent hand
    style = profile["style"]
    vpip = profile["vpip"]
    agg = profile["agg"]
    bluffs = profile["bluffs"]
    opp_pre = hand_ctx["opp_pre"]
    opp_bets = hand_ctx["opp_post_bets"]
    opp_calls = hand_ctx["opp_post_calls"]
    opp_checks = hand_ctx["opp_post_checks"]

    # Preflop range width cutoff
    if style == "MANIAC":
        pre_keep = 0.92
    elif opp_pre == "raise":
        pre_keep = 0.28 if style == "NIT" else (0.55 if style == "CALLING_STATION" else 0.48)
    elif opp_pre == "call":
        pre_keep = max(0.32, min(0.95, vpip + 0.12))
    else:
        pre_keep = 0.95  # Checked BB: almost any two cards

    pre_cutoff = 1.0 - pre_keep
    bluff_boost = min(0.60, bluffs * 0.20 + max(0.0, agg - 0.25) * 0.8)

    weights = [1.0] * n_combos
    for i, ((_, r1, s1), (_, r2, s2)) in enumerate(combos):
        pkey = (r1, r2, s1 == s2) if r1 >= r2 else (r2, r1, s1 == s2)
        w_pre = 1.0 if PREFLOP_RANK_PCT[pkey] >= pre_cutoff else 0.15

        bp = board_pct[i]
        if style == "MANIAC":
            w_post = 1.0
        elif opp_bets >= 2:
            if bp >= 0.75:
                w_post = 3.5
            elif bp >= 0.45:
                w_post = 0.75
            else:
                w_post = 0.12 + bluff_boost
        elif opp_bets == 1:
            if style == "NIT":
                w_post = 3.6 if bp >= 0.70 else (0.45 if bp >= 0.45 else 0.08 + bluff_boost * 0.5)
            elif style == "CALLING_STATION":
                w_post = 4.0 if bp >= 0.78 else (0.45 if bp >= 0.50 else 0.08)
            else:
                w_post = 2.6 if bp >= 0.60 else (0.90 if bp >= 0.35 else 0.22 + bluff_boost)
        elif opp_checks > 0 and opp_calls == 0:
            # Opponent checked: monsters are less likely (capped range)
            w_post = 0.40 if bp >= 0.82 else 1.0
        elif opp_calls > 0 and style != "CALLING_STATION":
            w_post = 0.45 if bp < 0.25 else 1.0
        else:
            w_post = 1.0

        weights[i] = w_pre * w_post

    # 3A. Exact weighted river enumeration (zero noise + free exact raw equity)
    if n_board == 5:
        hero_score = eval7.evaluate(hero + board)
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

    # 3B. Flop / Turn: range-sampled Monte Carlo rollouts
    cum_w = list(itertools.accumulate(weights))
    sampled = random.choices(combos, cum_weights=cum_w, k=sims)

    wins = 0.0
    if n_board == 4:
        # Turn: sample 1 river card per weighted opponent hand
        h_buf = hero + board + [None]
        o_buf = [None, None] + board + [None]
        for (c1, _, _), (c2, _, _) in sampled:
            r1 = deck_cards[random.randrange(n_deck)]
            while r1 == c1 or r1 == c2:
                r1 = deck_cards[random.randrange(n_deck)]
            h_buf[6] = r1
            o_buf[0] = c1
            o_buf[1] = c2
            o_buf[6] = r1
            hs = eval7.evaluate(h_buf)
            os_ = eval7.evaluate(o_buf)
            if hs > os_:
                wins += 1.0
            elif hs == os_:
                wins += 0.5
    else:
        # Flop: sample 2 distinct runout cards per weighted opponent hand
        h_buf = hero + board + [None, None]
        o_buf = [None, None] + board + [None, None]
        for (c1, _, _), (c2, _, _) in sampled:
            r1 = deck_cards[random.randrange(n_deck)]
            while r1 == c1 or r1 == c2:
                r1 = deck_cards[random.randrange(n_deck)]
            r2 = deck_cards[random.randrange(n_deck)]
            while r2 == c1 or r2 == c2 or r2 == r1:
                r2 = deck_cards[random.randrange(n_deck)]
            h_buf[5] = r1
            h_buf[6] = r2
            o_buf[0] = c1
            o_buf[1] = c2
            o_buf[5] = r1
            o_buf[6] = r2
            hs = eval7.evaluate(h_buf)
            os_ = eval7.evaluate(o_buf)
            if hs > os_:
                wins += 1.0
            elif hs == os_:
                wins += 0.5

    return (wins / sims), None


def update_memory(game_state: dict) -> None:
    """Save match_action_log by hand_num and remember river boards for showdown reads."""
    hand_num = game_state.get("hand_num")
    if hand_num == 0 and len(HAND_HISTORY) > 1:
        HAND_HISTORY.clear()
        HAND_BOARDS.clear()
        SHOWDOWN_BLUFF_CACHE.clear()

    board = game_state.get("community_cards") or []
    if hand_num is not None and len(board) == 5:
        HAND_BOARDS[hand_num] = list(board)

    log = game_state.get("match_action_log") or []
    grouped = {}
    for entry in log:
        h = entry.get("hand_num")
        if h is not None:
            grouped.setdefault(h, []).append(entry)

    for h, entries in grouped.items():
        if len(entries) >= len(HAND_HISTORY.get(h, [])):
            HAND_HISTORY[h] = entries


def profile_opponent(my_id: str) -> dict:
    """
    Fast opponent profiler:
    - Detects hyper-aggressive MANIACs as early as hand 4 (2+ shoves or >55% agg).
    - Classifies NIT, CALLING_STATION, MANIAC, or BALANCED by hand 8.
    - Tracks showdown river bluffs from revealed 'show' entries.
    """
    hands_seen = 0
    vpip_hands = 0
    faced_bet = 0
    folded_to_bet = 0
    passive_actions = 0
    aggr_actions = 0
    all_in_count = 0
    caught_bluffs = 0

    for h, entries in HAND_HISTORY.items():
        hands_seen += 1
        opp_put_chips_preflop = False
        opp_agg_postflop = False
        current_street = None
        hero_bet_this_street = False

        for e in entries:
            action = e.get("action")
            bot_id = e.get("bot_id")
            street = e.get("street")

            if action in ("small_blind", "big_blind", "ante"):
                continue

            if action == "show":
                if bot_id != my_id and opp_agg_postflop:
                    if h not in SHOWDOWN_BLUFF_CACHE:
                        opp_cards = e.get("cards") or []
                        river_board = HAND_BOARDS.get(h)
                        if len(opp_cards) == 2 and river_board and len(river_board) == 5:
                            SHOWDOWN_BLUFF_CACHE[h] = raw_river_equity(opp_cards, river_board) < 0.45
                        else:
                            SHOWDOWN_BLUFF_CACHE[h] = False
                    if SHOWDOWN_BLUFF_CACHE[h]:
                        caught_bluffs += 1
                continue

            if street != current_street:
                current_street = street
                hero_bet_this_street = False

            if bot_id == my_id:
                if action in ("raise", "all_in"):
                    hero_bet_this_street = True
            else:
                if street == "preflop" and action in ("call", "raise", "all_in"):
                    opp_put_chips_preflop = True
                if street in ("flop", "turn", "river") and action in ("raise", "all_in"):
                    opp_agg_postflop = True

                if hero_bet_this_street:
                    faced_bet += 1
                    if action == "fold":
                        folded_to_bet += 1
                    hero_bet_this_street = False

                if action in ("check", "call"):
                    passive_actions += 1
                elif action in ("raise", "all_in"):
                    aggr_actions += 1
                    if action == "all_in":
                        all_in_count += 1

        if opp_put_chips_preflop:
            vpip_hands += 1

    vpip = vpip_hands / hands_seen if hands_seen > 0 else 0.60
    fold_rate = folded_to_bet / faced_bet if faced_bet > 0 else 0.35
    total_non_fold = passive_actions + aggr_actions
    agg_rate = aggr_actions / total_non_fold if total_non_fold > 0 else 0.25
    all_in_rate = all_in_count / hands_seen if hands_seen > 0 else 0.0

    style = "BALANCED"
    # Fast-trigger Maniac detection so early shovers can't ambush hands 1-7
    if hands_seen >= 3 and (all_in_count >= 2 or (total_non_fold >= 6 and agg_rate >= 0.55)):
        style = "MANIAC"
    elif hands_seen >= 8:
        if agg_rate >= 0.42 or all_in_rate >= 0.20:
            style = "MANIAC"
        elif faced_bet >= 5 and fold_rate >= 0.50:
            style = "NIT"
        elif vpip <= 0.42 and agg_rate < 0.35:
            style = "NIT"
        elif faced_bet >= 5 and fold_rate <= 0.22 and agg_rate <= 0.22:
            style = "CALLING_STATION"

    return {
        "style": style,
        "hands": hands_seen,
        "vpip": round(vpip, 2),
        "fold": round(fold_rate, 2),
        "faced_bet": faced_bet,
        "agg": round(agg_rate, 2),
        "bluffs": caught_bluffs,
    }


def get_strategy_dials(profile: dict) -> dict:
    """Return exploitative dials with continuous fold-rate blending for BALANCED."""
    style = profile["style"]
    is_bluffer = profile["bluffs"] >= 2

    if style == "NIT":
        dials = {
            "preflop_raise_eq": 0.32,
            "bluff_freq": 0.85,
            "bluff_max_eq": 0.55,
            "bluff_bet_mult": 0.35,
            "value_bet_eq": 0.55,
            "big_value_eq": 0.72,
            "flop_big_mult": 0.60,
            "late_big_mult": 0.75,
            "small_bet_mult": 0.35,
            "call_margin": 0.06,
            "big_bet_guard": 0.65,
        }
    elif style == "CALLING_STATION":
        dials = {
            "preflop_raise_eq": 0.52,
            "bluff_freq": 0.00,
            "bluff_max_eq": 0.38,
            "bluff_bet_mult": 0.50,
            "value_bet_eq": 0.52,
            "big_value_eq": 0.65,
            "flop_big_mult": 0.95,
            "late_big_mult": 1.25,
            "small_bet_mult": 0.65,
            "call_margin": 0.03,
            "big_bet_guard": 0.60,
        }
    elif style == "MANIAC":
        dials = {
            "preflop_raise_eq": 0.55,
            "bluff_freq": 0.00,
            "bluff_max_eq": 0.38,
            "bluff_bet_mult": 0.50,
            "value_bet_eq": 0.56,
            "big_value_eq": 0.67,
            "flop_big_mult": 0.80,
            "late_big_mult": 0.95,
            "small_bet_mult": 0.50,
            "call_margin": -0.02,
            "big_bet_guard": 0.65,  # Dynamically scaled by stack depth in decide()
        }
    else:
        dials = {
            "preflop_raise_eq": 0.48,
            "bluff_freq": 0.10,
            "bluff_max_eq": 0.38,
            "bluff_bet_mult": 0.45,
            "value_bet_eq": 0.58,
            "big_value_eq": 0.72,
            "flop_big_mult": 1.20,
            "late_big_mult": 1.45,
            "small_bet_mult": 0.60,
            "call_margin": 0.03,
            "big_bet_guard": 0.58,
        }
        # Continuous exploit blending for semi-nits / semi-stations inside BALANCED
        if profile["faced_bet"] >= 6:
            f = profile["fold"]
            if f >= 0.40:
                dials["bluff_freq"] = min(0.35, 0.10 + (f - 0.38) * 1.5)
                dials["preflop_raise_eq"] = 0.44
            elif f <= 0.28:
                dials["bluff_freq"] = 0.02
                dials["late_big_mult"] = 1.50

    if is_bluffer and style != "MANIAC":
        dials["call_margin"] = max(-0.02, dials["call_margin"] - 0.05)
        dials["big_bet_guard"] = max(0.50, dials["big_bet_guard"] - 0.05)

    return dials


def safe_raise(game_state: dict, target_total: int, debug: dict) -> dict:
    """Ensure raises are legal; convert near-all-ins into clean all-ins."""
    your_stack = game_state["your_stack"]
    your_bet = game_state["your_bet_this_street"]
    min_raise_to = game_state["min_raise_to"]
    max_total = your_stack + your_bet

    amount = max(min_raise_to, int(target_total))
    chips_needed = amount - your_bet
    if chips_needed >= your_stack * 0.85 or amount >= max_total:
        return {"action": "all_in", "debug": debug}
    return {"action": "raise", "amount": amount, "debug": debug}


def get_effective_stack(game_state: dict) -> int:
    my_id = game_state["your_bot_id"]
    my_stack = game_state["your_stack"]
    for p in game_state.get("players", []):
        if p["bot_id"] != my_id:
            return min(my_stack, p["stack"])
    return my_stack


def maniac_big_guard(eff_stack_bb: float) -> float:
    """
    Freezeout patience vs. Maniac shoves:
    Deep-stacked, fold 55% coin flips and wait 3-4 hands to bust them with 65%+.
    Tighten down cleanly as blinds rise.
    """
    if eff_stack_bb >= 50:
        return 0.65
    if eff_stack_bb >= 25:
        return 0.58
    if eff_stack_bb > 12:
        return 0.52
    return 0.47


def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        return {"action": "check"}

    my_id = game_state["your_bot_id"]
    update_memory(game_state)
    profile = profile_opponent(my_id)
    style = profile["style"]
    is_maniac = style == "MANIAC"
    is_bluffer = profile["bluffs"] >= 2
    dials = get_strategy_dials(profile)

    street = game_state["street"]
    pot = game_state["pot"]
    owed = game_state["amount_owed"]
    can_check = game_state["can_check"]
    your_cards = game_state["your_cards"]
    board = game_state["community_cards"]
    your_stack = game_state["your_stack"]
    your_bet = game_state["your_bet_this_street"]
    current_bet = game_state["current_bet"]
    bb = max(1, game_state["blinds"]["bb"])

    seat = game_state.get("seat_to_act")
    button = game_state.get("button_seat")
    is_button = seat is not None and button is not None and seat == button
    is_seven_two = {c[0] for c in your_cards} == {"7", "2"}

    # 1. Calculate equity (cached preflop, Bayesian range-weighted postflop)
    raw_eq = None
    if street == "preflop":
        equity = get_preflop_equity(your_cards)
        raw_eq = equity
    else:
        hand_ctx = parse_current_hand_context(game_state, my_id)
        equity, raw_eq = range_weighted_postflop_equity(your_cards, board, hand_ctx, profile, sims=900)

    pot_odds = owed / (pot + owed) if owed > 0 else 0.0
    eff_stack_bb = get_effective_stack(game_state) / bb

    # Modest extra caution on late streets/re-raises (range model already does heavy lifting)
    facing_reraise = (your_bet > bb if street == "preflop" else your_bet > 0) and owed > 0
    late_pad = 0.03 if (street in ("turn", "river") and not is_maniac) else 0.0
    reraise_pad = 0.04 if (facing_reraise and not is_maniac) else 0.0
    required_call_eq = pot_odds + dials["call_margin"] + late_pad + reraise_pad

    if style == "NIT" and not is_bluffer and street != "preflop" and owed > 0:
        required_call_eq = max(required_call_eq, 0.56 if street == "flop" else 0.62)

    big_bet_mult = dials["flop_big_mult"] if street == "flop" else dials["late_big_mult"]

    debug = {
        "style": style,
        "h": profile["hands"],
        "fold": profile["fold"],
        "agg": profile["agg"],
        "blf": profile["bluffs"],
        "req_eq": round(equity, 3),
        "req": round(required_call_eq, 3),
    }

    # 2. SHORT-STACK ENDGAME MODE (Preflop when effective stack <= 12 BB)
    if street == "preflop" and eff_stack_bb <= 12:
        if owed == 0 or owed <= bb:
            if eff_stack_bb <= 6:
                shove_eq = 0.36 if style == "NIT" else 0.44
            else:
                shove_eq = 0.40 if style == "NIT" else 0.49
            if equity >= shove_eq:
                return {"action": "all_in", "debug": debug}
            return {"action": "check", "debug": debug} if can_check else {"action": "fold", "debug": debug}
        else:
            if style == "NIT" and not is_bluffer:
                call_eq = max(pot_odds + 0.10, 0.56)
            elif is_maniac:
                call_eq = max(pot_odds, 0.47)
            else:
                call_eq = max(pot_odds + 0.03, 0.47)
            if equity >= call_eq:
                return {"action": "all_in", "debug": debug}
            return {"action": "fold", "debug": debug}

    # 3. PREFLOP STRATEGY (Normal Stacks)
    if street == "preflop":
        if owed <= bb and current_bet <= bb:
            if can_check:
                iso_eq = max(dials["preflop_raise_eq"] + 0.06, 0.55)
                if equity >= iso_eq:
                    return safe_raise(game_state, int(bb * 3), debug)
                return {"action": "check", "debug": debug}
            else:
                if equity >= dials["preflop_raise_eq"]:
                    return safe_raise(game_state, int(bb * 2.5), debug)
                if is_seven_two and eff_stack_bb >= 25 and style in ("NIT", "BALANCED"):
                    return safe_raise(game_state, int(bb * 2.5), debug)
                if equity >= (0.46 if is_maniac else 0.42):
                    return {"action": "call", "debug": debug}
                return {"action": "fold", "debug": debug}
        else:
            # Facing a preflop raise or shove
            if is_maniac:
                guard_val = maniac_big_guard(eff_stack_bb)
                if owed > your_stack * 0.35:
                    if equity >= max(pot_odds, guard_val):
                        return {"action": "call", "debug": debug}
                    return {"action": "fold", "debug": debug}
                if equity >= 0.65:
                    return safe_raise(game_state, int(current_bet * 3), debug)
                if equity >= max(pot_odds, 0.50):
                    return {"action": "call", "debug": debug}
                return {"action": "fold", "debug": debug}

            three_bet_eq = 0.68 if style == "NIT" else 0.64
            if equity >= three_bet_eq:
                return safe_raise(game_state, int(current_bet * 3), debug)
            big_guard = dials["big_bet_guard"] if owed > your_stack * 0.40 else 0.0
            if equity >= max(required_call_eq, big_guard):
                return {"action": "call", "debug": debug}
            return {"action": "fold", "debug": debug}

    # 4. POSTFLOP STRATEGY (Flop, Turn, River)
    if can_check:
        val_eq = dials["value_bet_eq"] - (0.02 if is_button else 0.0)
        bluff_prob = dials["bluff_freq"]
        if bluff_prob > 0:
            if is_button:
                bluff_prob = min(0.92, bluff_prob + 0.08)
            elif style == "BALANCED":
                bluff_prob *= 0.5

        if equity >= dials["big_value_eq"]:
            return safe_raise(game_state, int(pot * big_bet_mult), debug)
        if equity >= val_eq:
            return safe_raise(game_state, int(pot * dials["small_bet_mult"]), debug)

        # Check raw equity vs random hand when bluffing (aligns with <40% Biggest Bluffer rule)
        if bluff_prob > 0 and equity < dials["bluff_max_eq"]:
            check_eq = raw_eq if raw_eq is not None else monte_carlo_raw(your_cards, board, sims=150)
            if check_eq < dials["bluff_max_eq"] and random.random() < bluff_prob:
                return safe_raise(game_state, int(pot * dials["bluff_bet_mult"]), debug)

        if is_seven_two and is_button and eff_stack_bb >= 25 and pot <= bb * 8 and style in ("NIT", "BALANCED"):
            return safe_raise(game_state, int(pot * 0.50), debug)
        return {"action": "check", "debug": debug}

    else:
        # Facing a postflop bet or raise
        raise_eq = 0.72 if is_maniac else (0.80 if style == "NIT" else 0.77)
        if equity >= raise_eq:
            target = max(int(current_bet * 2.5), current_bet + int(pot * big_bet_mult))
            return safe_raise(game_state, target, debug)

        if is_maniac:
            big_guard = maniac_big_guard(eff_stack_bb) if owed > your_stack * 0.35 else 0.0
        else:
            big_guard = dials["big_bet_guard"] if owed > your_stack * 0.45 else 0.0

        if equity >= max(required_call_eq, big_guard):
            return {"action": "call", "debug": debug}
        return {"action": "fold", "debug": debug}