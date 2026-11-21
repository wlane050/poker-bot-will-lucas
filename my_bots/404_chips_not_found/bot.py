"""
404 Chips Not Found — Version 14 (FINAL: Diagnostic-Hardened Bayesian + EV Engine)
Fixes proven by diagnose.py leak analysis:
1. Eliminates permanent false MANIAC classifications (requires deep-stack all-ins
   >= 18 BB + high aggression rate, bounded early trigger hands 3-7).
2. Escalates Bayesian range narrowing for postflop 3-bets/4-bets and massive
   deep overbets (>= 1.8x pot), and caps postflop re-raising wars without the nuts.
3. Tightens EV optimizer overbet/shove eligibility (>= 0.82 / >= 0.87) while
   improving flop/turn protection betting on medium-strong value hands.
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

# Match-long memory (resets cleanly per match)
HAND_HISTORY = {}          # {hand_num: list of action dicts}
HAND_BOARDS = {}           # {hand_num: [5 river board card strs]}
SHOWDOWN_BLUFF_CACHE = {}  # {hand_num: bool}
PREFLOP_EQ_CACHE = {}      # {(hi, lo, suited): float}
FIRST_HAND_ID = None


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


def get_preflop_raw_equity(your_cards: list) -> float:
    """Cached 1,000-sim preflop equity vs a random hand."""
    r0, r1 = RANK_ORDER[your_cards[0][0]], RANK_ORDER[your_cards[1][0]]
    suited = your_cards[0][1] == your_cards[1][1]
    key = (max(r0, r1), min(r0, r1), suited)
    if key not in PREFLOP_EQ_CACHE:
        PREFLOP_EQ_CACHE[key] = monte_carlo_raw(your_cards, [], sims=1000)
    return PREFLOP_EQ_CACHE[key]


def range_weighted_preflop_equity(your_cards: list, keep_frac: float, sims: int = 400) -> float:
    """Estimate preflop equity against opponent's top `keep_frac` range."""
    if keep_frac >= 0.92:
        return get_preflop_raw_equity(your_cards)

    used = set(your_cards)
    deck_recs = [(card, r, s) for c, card, r, s in CARD_RECORDS if c not in used]
    deck_cards = [rec[0] for rec in deck_recs]
    hero = [CARD_MAP[c] for c in your_cards]

    combos = list(itertools.combinations(deck_recs, 2))
    cutoff = 1.0 - max(0.08, min(0.95, keep_frac))

    weights = [
        1.0 if PREFLOP_RANK_PCT[(r1, r2, s1 == s2) if r1 >= r2 else (r2, r1, s1 == s2)] >= cutoff else 0.08
        for (_, r1, s1), (_, r2, s2) in combos
    ]
    cum_w = list(itertools.accumulate(weights))
    sampled = random.choices(combos, cum_weights=cum_w, k=sims)

    wins = 0.0
    for (c1, _, _), (c2, _, _) in sampled:
        draw7 = random.sample(deck_cards, 7)
        board5 = [x for x in draw7 if x is not c1 and x is not c2][:5]
        hs = eval7.evaluate(hero + board5)
        os_ = eval7.evaluate([c1, c2] + board5)
        if hs > os_:
            wins += 1.0
        elif hs == os_:
            wins += 0.5
    return wins / sims


def parse_current_hand_context(game_state: dict, my_id: str) -> dict:
    """Track preflop lines, postflop streets, AND same-street re-raise escalations."""
    hero_raised_pre = False
    opp_pre = "check"
    opp_limped = False
    opp_3bet = False
    opp_limp_shoved = False

    opp_post_bets = 0
    opp_post_calls = 0
    opp_post_checks = 0
    opp_late_bets = 0
    opp_raised_hero_post = False

    cur_street = None
    hero_bets_street = 0
    opp_raises_hero_street = 0
    opp_bets_street = 0

    for e in game_state.get("action_log") or []:
        act = e.get("action")
        st = e.get("street")
        if act in ("small_blind", "big_blind", "ante", "show"):
            continue

        if st != cur_street:
            cur_street = st
            hero_bets_street = 0
            opp_raises_hero_street = 0
            opp_bets_street = 0

        if e.get("bot_id") == my_id:
            if st == "preflop" and act in ("raise", "all_in"):
                hero_raised_pre = True
            elif st in ("flop", "turn", "river") and act in ("raise", "all_in"):
                hero_bets_street += 1
        else:
            if st == "preflop":
                if act == "call" and opp_pre == "check" and not hero_raised_pre:
                    opp_pre = "limp"
                    opp_limped = True
                elif act == "call" and opp_pre == "check":
                    opp_pre = "call"
                elif act in ("raise", "all_in"):
                    if opp_limped and hero_raised_pre:
                        opp_limp_shoved = True
                        opp_pre = "limp_shove"
                    elif hero_raised_pre:
                        opp_3bet = True
                        opp_pre = "3bet"
                    else:
                        opp_pre = "raise"
            else:
                if act in ("raise", "all_in"):
                    opp_post_bets += 1
                    opp_bets_street += 1
                    if hero_bets_street > 0:
                        opp_raised_hero_post = True
                        opp_raises_hero_street += 1
                    if st in ("turn", "river"):
                        opp_late_bets += 1
                elif act == "call":
                    opp_post_calls += 1
                elif act == "check":
                    opp_post_checks += 1

    # Current-street counters only apply if we are still on that street
    active_st = game_state.get("street")
    if cur_street != active_st:
        hero_bets_street = 0
        opp_raises_hero_street = 0
        opp_bets_street = 0

    return {
        "opp_pre": opp_pre,
        "opp_limped": opp_limped,
        "opp_3bet": opp_3bet,
        "opp_limp_shoved": opp_limp_shoved,
        "opp_post_bets": opp_post_bets,
        "opp_post_calls": opp_post_calls,
        "opp_post_checks": opp_post_checks,
        "opp_late_bets": opp_late_bets,
        "opp_raised_hero_post": opp_raised_hero_post,
        "hero_bets_street": hero_bets_street,
        "opp_raises_hero_street": opp_raises_hero_street,
        "opp_bets_street": opp_bets_street,
    }


def range_weighted_postflop_distribution(
    your_cards: list,
    community_cards: list,
    hand_ctx: dict,
    profile: dict,
    owed_to_pot: float,
    eff_stack_bb: float,
    sims: int = 750,
) -> tuple:
    """
    Bayesian range-weighted postflop evaluation returning 20 strength buckets:
    - Sharp range narrowing on postflop 3-bets/4-bets, massive overbets (>=1.8x pot),
      and triple barrels.
    - Exact 990-combo weighted enumeration on river; 750 range-sampled flop/turn sims.
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

    style = profile["style"]
    vpip = profile["vpip"]
    agg = profile["agg"]
    bluffs = profile["bluffs"]
    is_trapper = profile["is_trapper"]

    opp_pre = hand_ctx["opp_pre"]
    opp_bets = hand_ctx["opp_post_bets"]
    opp_calls = hand_ctx["opp_post_calls"]
    opp_checks = hand_ctx["opp_post_checks"]
    opp_raised_hero = hand_ctx["opp_raised_hero_post"]
    opp_raises_st = hand_ctx["opp_raises_hero_street"]
    opp_bets_st = hand_ctx["opp_bets_street"]

    # Detect postflop 3-bet/4-bet or massive deep-stack overbet (>=1.8x pot)
    is_post_3bet = opp_raises_st >= 2 or (opp_raises_st >= 1 and opp_bets_st >= 2)
    is_huge_overbet = owed_to_pot >= 1.8 and eff_stack_bb > 16

    if style == "MANIAC":
        pre_keep = 0.94
    elif opp_pre in ("3bet", "limp_shove"):
        pre_keep = 0.22
    elif opp_pre == "raise":
        pre_keep = 0.30 if style == "NIT" else (0.56 if style == "CALLING_STATION" else max(0.45, min(0.75, vpip)))
    elif opp_pre in ("call", "limp"):
        pre_keep = max(0.35, min(0.95, vpip + 0.12))
    else:
        pre_keep = 0.95

    pre_cutoff = 1.0 - pre_keep
    bluff_boost = min(0.55, bluffs * 0.20 + max(0.0, agg - 0.28) * 0.85)

    weights = [1.0] * n_combos
    combo_bins = [0] * n_combos
    for i, ((_, r1, s1), (_, r2, s2)) in enumerate(combos):
        pkey = (r1, r2, s1 == s2) if r1 >= r2 else (r2, r1, s1 == s2)
        w_pre = 1.0 if PREFLOP_RANK_PCT[pkey] >= pre_cutoff else 0.14

        bp = board_pct[i]
        combo_bins[i] = 19 if bp >= 0.95 else int(bp * 20)

        if style == "MANIAC":
            # Mild tilt even vs maniacs when they re-raise or shove
            if is_post_3bet or opp_raised_hero:
                w_post = 1.45 if bp >= 0.65 else (1.0 if bp >= 0.35 else 0.72)
            else:
                w_post = 1.0
        elif is_post_3bet:
            # Postflop 3-bet/4-bet war: overwhelmingly top ~12% monsters
            if bp >= 0.88:
                w_post = 8.5
            elif bp >= 0.74:
                w_post = 0.90
            else:
                w_post = 0.03 + bluffs * 0.05
        elif is_huge_overbet:
            # Massive deep overbet / shove (e.g., 5x-60x pot): top ~14% monsters
            if bp >= 0.86:
                w_post = 7.5
            elif bp >= 0.70:
                w_post = 0.80
            else:
                w_post = 0.04 + bluffs * 0.05
        elif opp_raised_hero:
            # Postflop check-raise or raise over our bet: top ~20%
            if bp >= 0.80:
                w_post = 5.5
            elif bp >= 0.58:
                w_post = 0.60
            else:
                w_post = 0.06 + bluffs * 0.06
        elif opp_bets >= 3:
            # Triple barrel across flop/turn/river
            if bp >= 0.80:
                w_post = 4.6
            elif bp >= 0.55:
                w_post = 0.70
            else:
                w_post = 0.10 + bluff_boost * 0.6
        elif opp_bets == 2:
            if bp >= 0.74:
                w_post = 3.6
            elif bp >= 0.45:
                w_post = 0.75
            else:
                w_post = 0.12 + bluff_boost
        elif opp_bets == 1:
            if style == "NIT" or (is_trapper and hand_ctx["opp_late_bets"] > 0):
                w_post = 3.8 if bp >= 0.70 else (0.45 if bp >= 0.45 else 0.08 + bluff_boost * 0.4)
            elif style == "CALLING_STATION":
                w_post = 4.0 if bp >= 0.78 else (0.45 if bp >= 0.50 else 0.08)
            else:
                w_post = 2.5 if bp >= 0.58 else (0.90 if bp >= 0.35 else 0.22 + bluff_boost)
        elif opp_checks > 0 and opp_calls == 0:
            w_post = 0.85 if is_trapper else (0.38 if bp >= 0.80 else 1.0)
        elif opp_calls > 0 and style != "CALLING_STATION":
            w_post = 0.45 if bp < 0.24 else 1.0
        else:
            w_post = 1.0

        weights[i] = w_pre * w_post

    bin_w = [0.0] * 20
    bin_win = [0.0] * 20

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
            b = combo_bins[i]
            bin_w[b] += w
            bin_win[b] += w * out
            w_wins += w * out
            w_total += w
            raw_wins += out
        eq = w_wins / w_total if w_total > 0 else 0.5
        return eq, (raw_wins / n_combos), bin_w, bin_win

    cum_w = list(itertools.accumulate(weights))
    sampled_idx = random.choices(range(n_combos), cum_weights=cum_w, k=sims)

    wins = 0.0
    if n_board == 4:
        h_buf = hero + board + [None]
        o_buf = [None, None] + board + [None]
        for idx in sampled_idx:
            (c1, _, _), (c2, _, _) = combos[idx]
            r1 = deck_cards[random.randrange(n_deck)]
            while r1 == c1 or r1 == c2:
                r1 = deck_cards[random.randrange(n_deck)]
            h_buf[6] = r1
            o_buf[0] = c1
            o_buf[1] = c2
            o_buf[6] = r1
            hs = eval7.evaluate(h_buf)
            os_ = eval7.evaluate(o_buf)
            out = 1.0 if hs > os_ else (0.5 if hs == os_ else 0.0)
            b = combo_bins[idx]
            bin_w[b] += 1.0
            bin_win[b] += out
            wins += out
    else:
        h_buf = hero + board + [None, None]
        o_buf = [None, None] + board + [None, None]
        for idx in sampled_idx:
            (c1, _, _), (c2, _, _) = combos[idx]
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
            out = 1.0 if hs > os_ else (0.5 if hs == os_ else 0.0)
            b = combo_bins[idx]
            bin_w[b] += 1.0
            bin_win[b] += out
            wins += out

    return (wins / sims), None, bin_w, bin_win


def equity_when_called(bin_w: list, bin_win: list, total_w: float, continue_frac: float) -> float:
    """Compute hero's equity against the top `continue_frac` of villain's posterior range."""
    if total_w <= 0.0:
        return 0.5
    target = max(1e-9, min(1.0, continue_frac)) * total_w
    acc_w = 0.0
    acc_win = 0.0
    for b in range(19, -1, -1):
        bw = bin_w[b]
        if bw <= 0.0:
            continue
        if acc_w + bw >= target:
            need = target - acc_w
            acc_win += bin_win[b] * (need / bw)
            acc_w = target
            break
        acc_w += bw
        acc_win += bin_win[b]
    return acc_win / acc_w if acc_w > 0.0 else 0.5


def estimate_fold_prob(bet_frac: float, profile: dict, street: str) -> float:
    """Blend pot-odds pressure with empirical size-bucketed fold rates."""
    style = profile["style"]
    theo_fold = bet_frac / (1.0 + bet_frac)

    if style == "CALLING_STATION":
        return min(0.10, 0.02 + 0.05 * theo_fold)
    if style == "MANIAC":
        return min(0.14, 0.04 + 0.08 * theo_fold)

    if bet_frac <= 0.60:
        obs_n = profile["faced_small"]
        obs_f = profile["folded_small"]
        prior_f = 0.68 if style == "NIT" else 0.28
    else:
        obs_n = profile["faced_big"]
        obs_f = profile["folded_big"]
        prior_f = 0.78 if style == "NIT" else 0.44

    emp_fold = (obs_f + prior_f * 4.0) / (obs_n + 4.0)

    if style == "NIT":
        pf = emp_fold * (0.88 + 0.24 * theo_fold)
        return max(0.45, min(0.88, pf))

    conf = min(0.65, obs_n / 10.0)
    pf = (1.0 - conf) * (0.90 * theo_fold) + conf * emp_fold

    if profile["is_trapper"] and street in ("turn", "river"):
        pf *= 0.82
    return max(0.05, min(0.82, pf))


def choose_ev_optimal_bet(
    game_state: dict,
    range_eq: float,
    raw_eq,
    bin_w: list,
    bin_win: list,
    hand_ctx: dict,
    profile: dict,
    is_button: bool,
    eff_stack: int,
    debug: dict,
) -> dict:
    """
    Postflop Expected Value (EV) optimizer when can_check is True.
    Reserves huge overbets/shoves for genuine monsters (>=0.82 / >=0.87) while
    betting medium-strong hands cleanly for value and protection.
    """
    pot = max(1, game_state["pot"])
    street = game_state["street"]
    style = profile["style"]
    your_bet = game_state["your_bet_this_street"]
    min_add = max(1, game_state["min_raise_to"] - your_bet)
    max_add = min(game_state["your_stack"], eff_stack)

    if max_add <= 0:
        return {"action": "check", "debug": debug}

    total_w = sum(bin_w)
    # Checking back on flop/turn forfeits equity realization via free cards
    if street == "river":
        realize = 1.00 if is_button else 0.94
    else:
        realize = 0.90 if is_button else 0.85
    ev_check = range_eq * pot * realize

    # Candidate bet sizes scaled to hand strength & opponent archetype
    if style == "NIT":
        fracs = [0.36, 0.65, 0.95]
    elif style == "CALLING_STATION":
        fracs = [0.55, 0.95, 1.35]
    elif style == "MANIAC":
        fracs = [0.50, 0.85, 1.15]
    else:
        fracs = [0.40, 0.68, 1.15]
        if range_eq >= 0.82:
            fracs.append(1.55)

    candidate_adds = []
    seen = set()
    for f in fracs:
        add = max(min_add, min(max_add, int(pot * f)))
        if add not in seen:
            seen.add(add)
            candidate_adds.append(add)

    # Only shove >1.15x pot directly with genuine monsters (>=0.87), or normal <=1.15x SPR shoves
    if max_add not in seen and (
        (range_eq >= 0.87 and max_add <= pot * 2.2)
        or (range_eq >= 0.76 and max_add <= pot * 1.15)
    ):
        candidate_adds.append(max_add)

    uncontested = hand_ctx["opp_post_calls"] == 0 and hand_ctx["opp_post_bets"] == 0
    is_bluff_hand = range_eq < 0.42

    best_ev = ev_check
    best_add = 0

    for add in candidate_adds:
        frac = add / pot

        if is_bluff_hand:
            if style in ("CALLING_STATION", "MANIAC"):
                continue
            if style != "NIT":
                if not uncontested or frac > 0.72:
                    continue
                if profile["is_trapper"] or (profile["faced_bet"] >= 6 and profile["fold"] <= 0.28):
                    continue

        pf = estimate_fold_prob(frac, profile, street)
        eq_called = equity_when_called(bin_w, bin_win, total_w, 1.0 - pf)

        if style == "CALLING_STATION":
            raise_tax = 0.0
        elif style == "NIT":
            raise_tax = 0.03 * add
        elif profile["is_trapper"]:
            raise_tax = 0.11 * add
        else:
            raise_tax = 0.07 * add

        ev_bet = pf * pot + (1.0 - pf) * (eq_called * (pot + 2.0 * add) - add - raise_tax)

        if is_bluff_hand:
            req_edge = 0.02 * pot if style == "NIT" else (0.05 * pot if is_button else 0.09 * pot)
        else:
            req_edge = 0.02 * pot

        if ev_bet > best_ev and ev_bet > ev_check + req_edge:
            best_ev = ev_bet
            best_add = add

    if best_add <= 0:
        return {"action": "check", "debug": debug}

    # Verify raw equity < 0.39 on non-Nit bluffs (aligns with Biggest Bluffer prize rule)
    if is_bluff_hand and style != "NIT":
        raw_checked = raw_eq if raw_eq is not None else monte_carlo_raw(
            game_state["your_cards"], game_state["community_cards"], sims=140
        )
        if raw_checked >= 0.39:
            return {"action": "check", "debug": debug}
        bluff_cap = 0.52 if (is_button and profile["fold"] >= 0.42) else (0.24 if is_button else 0.10)
        if random.random() >= bluff_cap:
            return {"action": "check", "debug": debug}
    elif is_bluff_hand and style == "NIT":
        if random.random() >= (0.90 if is_button else 0.72):
            return {"action": "check", "debug": debug}

    debug["ev_b"] = round(best_ev - ev_check, 1)
    if best_add >= max_add * 0.90:
        return {"action": "all_in", "debug": debug}
    return safe_raise(game_state, your_bet + best_add, debug)


def update_memory(game_state: dict) -> None:
    """Save match_action_log by hand_num and reset cleanly on a new hand_num==0 match."""
    global FIRST_HAND_ID
    hand_num = game_state.get("hand_num")
    hand_id = game_state.get("hand_id")

    if hand_num == 0:
        if FIRST_HAND_ID is not None and hand_id != FIRST_HAND_ID:
            HAND_HISTORY.clear()
            HAND_BOARDS.clear()
            SHOWDOWN_BLUFF_CACHE.clear()
        FIRST_HAND_ID = hand_id

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
    Precision opponent profiler:
    - Separates true MANIACs (high overall aggression + deep all-ins) from
      short-stack shovers and limp-shovers (zero permanent false MANIAC locks).
    - Tracks size-aware fold curves (small <=0.60x vs big >0.60x), showdown bluffs,
      limps, limp-shoves, and passive-early/aggressive-late trappers.
    """
    hands_seen = 0
    vpip_hands = 0
    pre_raise_hands = 0
    limp_hands = 0
    limp_reraises = 0
    faced_bet = 0
    folded_to_bet = 0
    faced_small = 0
    folded_small = 0
    faced_big = 0
    folded_big = 0
    passive_actions = 0
    aggr_actions = 0
    flop_seen_hands = 0
    flop_agg_hands = 0
    trap_hands = 0
    deep_all_ins = 0
    caught_bluffs = 0

    for h, entries in HAND_HISTORY.items():
        hands_seen += 1
        opp_put_chips_preflop = False
        opp_raised_preflop = False
        opp_limped_preflop = False
        opp_limp_reraised = False
        saw_flop = False
        opp_flop_agg = False
        opp_late_agg = False
        opp_agg_postflop = False
        hand_had_deep_all_in = False

        current_street = None
        hero_bet_this_street = False
        hero_bet_is_big = False
        pot_est = 0
        bb_est = 50
        street_bets = {}

        for e in entries:
            action = e.get("action")
            bot_id = e.get("bot_id")
            street = e.get("street")
            amt = e.get("amount") or 0

            if action == "ante":
                pot_est += amt
                continue
            if action in ("small_blind", "big_blind"):
                if action == "big_blind" and amt > 0:
                    bb_est = amt
                street_bets[bot_id] = street_bets.get(bot_id, 0) + amt
                pot_est += amt
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
                hero_bet_is_big = False
                street_bets.clear()
                if street == "flop":
                    saw_flop = True

            pot_before = max(1, pot_est)
            if action == "call":
                pot_est += amt
                street_bets[bot_id] = street_bets.get(bot_id, 0) + amt
                added = amt
            elif action in ("raise", "all_in"):
                prev = street_bets.get(bot_id, 0)
                added = max(0, amt - prev)
                pot_est += added
                street_bets[bot_id] = max(prev, amt)
            else:
                added = 0

            if bot_id == my_id:
                if action in ("raise", "all_in"):
                    hero_bet_this_street = True
                    hero_bet_is_big = (added / pot_before) > 0.60
            else:
                if street == "preflop":
                    if action == "call" and not opp_put_chips_preflop:
                        opp_put_chips_preflop = True
                        opp_limped_preflop = True
                    elif action in ("raise", "all_in"):
                        if opp_limped_preflop:
                            opp_limp_reraised = True
                        opp_put_chips_preflop = True
                        opp_raised_preflop = True

                if street in ("flop", "turn", "river") and action in ("raise", "all_in"):
                    opp_agg_postflop = True
                    if street == "flop":
                        opp_flop_agg = True
                    else:
                        opp_late_agg = True

                if hero_bet_this_street:
                    faced_bet += 1
                    if street != "preflop":
                        if hero_bet_is_big:
                            faced_big += 1
                            if action == "fold":
                                folded_big += 1
                        else:
                            faced_small += 1
                            if action == "fold":
                                folded_small += 1
                    if action == "fold":
                        folded_to_bet += 1
                    hero_bet_this_street = False

                if action in ("check", "call"):
                    passive_actions += 1
                elif action in ("raise", "all_in"):
                    aggr_actions += 1
                    # Only count deep-stack all-ins (>=18 BB) toward maniac detection
                    if action == "all_in" and amt >= bb_est * 18:
                        hand_had_deep_all_in = True

        if opp_put_chips_preflop:
            vpip_hands += 1
        if opp_raised_preflop:
            pre_raise_hands += 1
        if opp_limped_preflop:
            limp_hands += 1
        if opp_limp_reraised:
            limp_reraises += 1
        if hand_had_deep_all_in:
            deep_all_ins += 1
        if saw_flop:
            flop_seen_hands += 1
            if opp_flop_agg:
                flop_agg_hands += 1
            if opp_late_agg and not opp_flop_agg and not opp_raised_preflop:
                trap_hands += 1

    vpip = vpip_hands / hands_seen if hands_seen > 0 else 0.60
    pre_raise_rate = pre_raise_hands / hands_seen if hands_seen > 0 else 0.35
    limp_rate = limp_hands / hands_seen if hands_seen > 0 else 0.0
    fold_rate = folded_to_bet / faced_bet if faced_bet > 0 else 0.35
    total_non_fold = passive_actions + aggr_actions
    agg_rate = aggr_actions / total_non_fold if total_non_fold > 0 else 0.25
    flop_agg_rate = flop_agg_hands / flop_seen_hands if flop_seen_hands > 0 else 0.30
    deep_all_in_rate = deep_all_ins / hands_seen if hands_seen > 0 else 0.0

    style = "BALANCED"
    # Bounded early trigger (hands 3-7 ONLY) requiring genuine maniac aggression
    if 3 <= hands_seen <= 7 and (
        (deep_all_ins >= 2 and agg_rate >= 0.42)
        or (total_non_fold >= 7 and agg_rate >= 0.56)
    ):
        style = "MANIAC"
    elif hands_seen >= 8:
        if agg_rate >= 0.45 or (deep_all_in_rate >= 0.22 and agg_rate >= 0.36):
            style = "MANIAC"
        elif faced_bet >= 5 and fold_rate >= 0.50:
            style = "NIT"
        elif vpip <= 0.42 and agg_rate < 0.35:
            style = "NIT"
        elif faced_bet >= 5 and fold_rate <= 0.22 and agg_rate <= 0.22:
            style = "CALLING_STATION"
    elif hands_seen >= 5:
        if faced_bet >= 4 and fold_rate >= 0.65:
            style = "NIT"
        elif faced_bet >= 4 and fold_rate <= 0.15 and agg_rate <= 0.20:
            style = "CALLING_STATION"

    is_trapper = (
        hands_seen >= 6
        and trap_hands >= 2
        and pre_raise_rate <= 0.20
        and flop_agg_rate <= 0.22
        and style != "MANIAC"
    )
    is_limper = hands_seen >= 5 and limp_rate >= 0.30

    return {
        "style": style,
        "hands": hands_seen,
        "vpip": round(vpip, 2),
        "limp": round(limp_rate, 2),
        "fold": round(fold_rate, 2),
        "faced_bet": faced_bet,
        "faced_small": faced_small,
        "folded_small": folded_small,
        "faced_big": faced_big,
        "folded_big": folded_big,
        "agg": round(agg_rate, 2),
        "bluffs": caught_bluffs,
        "is_trapper": is_trapper,
        "is_limper": is_limper,
        "limp_reraises": limp_reraises,
    }


def get_strategy_dials(profile: dict) -> dict:
    """Return preflop and calling/raising dials for each archetype."""
    style = profile["style"]
    is_bluffer = profile["bluffs"] >= 2
    is_trapper = profile["is_trapper"]

    if style == "NIT":
        dials = {
            "preflop_raise_eq": 0.32,
            "flop_big_mult": 0.60,
            "late_big_mult": 0.75,
            "call_margin": 0.06,
            "big_bet_guard": 0.65,
        }
    elif style == "CALLING_STATION":
        dials = {
            "preflop_raise_eq": 0.51,
            "flop_big_mult": 0.95,
            "late_big_mult": 1.25,
            "call_margin": 0.03,
            "big_bet_guard": 0.60,
        }
    elif style == "MANIAC":
        dials = {
            "preflop_raise_eq": 0.55,
            "flop_big_mult": 0.85,
            "late_big_mult": 1.00,
            "call_margin": -0.02,
            "big_bet_guard": 0.65,
        }
    else:
        dials = {
            "preflop_raise_eq": 0.46,
            "flop_big_mult": 1.15,
            "late_big_mult": 1.40,
            "call_margin": 0.02,
            "big_bet_guard": 0.58,
        }
        if profile["faced_bet"] >= 6 and profile["fold"] >= 0.42:
            dials["preflop_raise_eq"] = 0.42

    if is_trapper and not is_bluffer:
        dials["call_margin"] = 0.07
        dials["big_bet_guard"] = 0.68
    elif is_bluffer and style != "MANIAC":
        dials["call_margin"] = max(-0.02, dials["call_margin"] - 0.04)
        dials["big_bet_guard"] = max(0.53, dials["big_bet_guard"] - 0.05)

    return dials


def safe_raise(game_state: dict, target_total: int, debug: dict) -> dict:
    """Ensure raises are legal; apply subtle jitter and convert near-all-ins to all-in."""
    your_stack = game_state["your_stack"]
    your_bet = game_state["your_bet_this_street"]
    min_raise_to = game_state["min_raise_to"]
    max_total = your_stack + your_bet

    jitter = random.uniform(0.94, 1.06)
    amount = max(min_raise_to, int(target_total * jitter))
    chips_needed = amount - your_bet
    if chips_needed >= your_stack * 0.85 or amount >= max_total:
        return {"action": "all_in", "debug": debug}
    return {"action": "raise", "amount": amount, "debug": debug}


def get_opponent_info(game_state: dict, my_id: str) -> tuple:
    """Return (effective_stack, opp_is_all_in)."""
    my_stack = game_state["your_stack"]
    for p in game_state.get("players", []):
        if p.get("bot_id") != my_id:
            opp_stack = p.get("stack", my_stack)
            opp_all_in = bool(p.get("is_all_in") or p.get("state") == "all_in")
            return min(my_stack, opp_stack), opp_all_in
    return my_stack, False


def maniac_big_guard(eff_stack_bb: float) -> float:
    """Freezeout patience vs. Maniac shoves scaled by effective stack in BBs."""
    if eff_stack_bb >= 50:
        return 0.65
    if eff_stack_bb >= 25:
        return 0.58
    if eff_stack_bb > 12:
        return 0.52
    return 0.47


def preflop_villain_keep_frac(
    hand_ctx: dict,
    profile: dict,
    eff_stack_bb: float,
    owed: int,
    your_stack: int,
    opp_all_in: bool,
) -> float:
    """Estimate opponent's preflop raising/shoving range width for range-weighted equity."""
    style = profile["style"]
    if style == "MANIAC":
        return 0.92
    if hand_ctx["opp_limp_shoved"]:
        return 0.16
    if hand_ctx["opp_3bet"]:
        return 0.15 if style == "NIT" else 0.24
    if opp_all_in or owed >= your_stack * 0.45:
        if eff_stack_bb <= 10:
            return 0.35 if style == "NIT" else 0.64
        if eff_stack_bb <= 16:
            return 0.28 if style == "NIT" else 0.56
        return 0.22 if style == "NIT" else 0.36
    if style == "NIT":
        return 0.30
    return max(0.48, min(0.80, profile["vpip"] + 0.05))


def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        return {"action": "check"}

    my_id = game_state["your_bot_id"]
    update_memory(game_state)
    profile = profile_opponent(my_id)
    style = profile["style"]
    is_maniac = style == "MANIAC"
    is_bluffer = profile["bluffs"] >= 2
    is_trapper = profile["is_trapper"]
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
    blinds = game_state.get("blinds") or {}
    bb = max(1, blinds.get("bb", 100))
    ante = blinds.get("ante", 0)

    seat = game_state.get("seat_to_act")
    button = game_state.get("button_seat")
    is_button = seat is not None and button is not None and seat == button
    is_seven_two = {c[0] for c in your_cards} == {"7", "2"}

    eff_stack, opp_all_in = get_opponent_info(game_state, my_id)
    eff_stack_bb = eff_stack / bb
    pot_odds = owed / (pot + owed) if owed > 0 else 0.0
    owed_to_pot = owed / max(1, pot - owed) if owed > 0 else 0.0
    hand_ctx = parse_current_hand_context(game_state, my_id)

    # 1. Compute Equity (Range-Weighted preflop vs raises + 20-bin Bayesian postflop)
    raw_eq = None
    bin_w = None
    bin_win = None
    if street == "preflop":
        raw_eq = get_preflop_raw_equity(your_cards)
        if owed > bb or current_bet > bb:
            keep_frac = preflop_villain_keep_frac(
                hand_ctx, profile, eff_stack_bb, owed, your_stack, opp_all_in
            )
            equity = range_weighted_preflop_equity(your_cards, keep_frac, sims=400)
        else:
            equity = raw_eq
    else:
        equity, raw_eq, bin_w, bin_win = range_weighted_postflop_distribution(
            your_cards, board, hand_ctx, profile, owed_to_pot, eff_stack_bb, sims=750
        )

    facing_reraise = (your_bet > bb if street == "preflop" else your_bet > 0) and owed > 0
    late_pad = 0.03 if (street in ("turn", "river") and not is_maniac) else 0.0
    if is_trapper and hand_ctx["opp_late_bets"] > 0 and not is_bluffer:
        late_pad += 0.05
    reraise_pad = 0.05 if (facing_reraise and not is_maniac) else 0.0
    required_call_eq = pot_odds + dials["call_margin"] + late_pad + reraise_pad

    if style == "NIT" and not is_bluffer and street != "preflop" and owed > 0:
        required_call_eq = max(required_call_eq, 0.56 if street == "flop" else 0.62)

    big_bet_mult = dials["flop_big_mult"] if street == "flop" else dials["late_big_mult"]

    debug = {
        "style": style,
        "h": profile["hands"],
        "fold": profile["fold"],
        "agg": profile["agg"],
        "trp": int(is_trapper),
        "eq": round(equity, 3),
        "req": round(required_call_eq, 3),
    }

    # 2. SHORT-STACK ENDGAME MODE (Preflop when effective stack <= 12 BB)
    if street == "preflop" and eff_stack_bb <= 12:
        if owed == 0 or (owed <= bb and current_bet <= bb):
            if can_check:
                shove_eq = 0.48 if eff_stack_bb <= 7 else 0.54
                if raw_eq >= shove_eq:
                    return {"action": "all_in", "debug": debug}
                return {"action": "check", "debug": debug}
            else:
                if eff_stack_bb <= 6:
                    shove_eq = 0.36 if style == "NIT" else 0.43
                else:
                    shove_eq = 0.40 if style == "NIT" else 0.48
                if raw_eq >= shove_eq:
                    return {"action": "all_in", "debug": debug}
                return {"action": "fold", "debug": debug}
        else:
            call_pad = 0.05 if style == "NIT" else (0.00 if is_maniac else 0.02)
            if equity >= max(pot_odds + call_pad, 0.42):
                return {"action": "all_in", "debug": debug}
            return {"action": "fold", "debug": debug}

    # 3. PREFLOP STRATEGY (Normal & Medium-Short Stacks > 12 BB)
    if street == "preflop":
        if owed <= bb and current_bet <= bb:
            if can_check:
                if profile["is_limper"] or profile["limp_reraises"] >= 1:
                    iso_eq = 0.58
                elif style in ("NIT", "CALLING_STATION"):
                    iso_eq = 0.50
                else:
                    iso_eq = 0.54
                if raw_eq >= iso_eq:
                    iso_mult = 2.5 if eff_stack_bb <= 22 else 3.0
                    return safe_raise(game_state, int(bb * iso_mult), debug)
                return {"action": "check", "debug": debug}
            else:
                ante_bonus = 0.02 if ante > 0 else 0.0
                open_eq = max(0.30, dials["preflop_raise_eq"] - ante_bonus)
                open_mult = 2.0 if eff_stack_bb <= 22 else 2.25
                if raw_eq >= open_eq:
                    return safe_raise(game_state, int(bb * open_mult), debug)
                if is_seven_two and eff_stack_bb >= 25 and style in ("NIT", "BALANCED"):
                    return safe_raise(game_state, int(bb * open_mult), debug)
                if raw_eq >= (0.47 if is_maniac else 0.42):
                    return {"action": "call", "debug": debug}
                return {"action": "fold", "debug": debug}
        else:
            # Facing a preflop raise, 3-bet, limp-shove, or deep shove
            if is_maniac:
                guard_val = maniac_big_guard(eff_stack_bb)
                if owed > your_stack * 0.35:
                    if equity >= max(pot_odds, guard_val):
                        return {"action": "call", "debug": debug}
                    return {"action": "fold", "debug": debug}
                if raw_eq >= 0.64:
                    return safe_raise(game_state, int(current_bet * 3), debug)
                if equity >= max(pot_odds + 0.01, 0.49):
                    return {"action": "call", "debug": debug}
                return {"action": "fold", "debug": debug}

            if hand_ctx["opp_limp_shoved"] or hand_ctx["opp_3bet"]:
                if equity >= 0.62 and not opp_all_in:
                    return safe_raise(game_state, int(current_bet * 2.6), debug)
                big_floor = 0.52 if owed > your_stack * 0.40 else 0.44
                if equity >= max(pot_odds + 0.03, big_floor):
                    return {"action": "call", "debug": debug}
                return {"action": "fold", "debug": debug}

            # 13-18 BB BB Re-Shove Gear over standard opens
            if 12 < eff_stack_bb <= 18 and not opp_all_in and owed <= bb * 3.5:
                reshove_eq = 0.61 if style == "NIT" else 0.56
                if equity >= reshove_eq or raw_eq >= (0.63 if style == "NIT" else 0.59):
                    return {"action": "all_in", "debug": debug}

            three_bet_eq = 0.68 if style == "NIT" else 0.63
            if raw_eq >= three_bet_eq and not opp_all_in:
                return safe_raise(game_state, int(current_bet * 2.8), debug)

            big_guard = (0.48 if eff_stack_bb <= 18 else dials["big_bet_guard"]) if owed > your_stack * 0.40 else 0.0
            if equity >= max(pot_odds + 0.02, 0.37, big_guard):
                return {"action": "call", "debug": debug}
            return {"action": "fold", "debug": debug}

    # 4. POSTFLOP STRATEGY (Flop, Turn, River)
    if can_check:
        # OOP Check-Call Discipline vs preflop aggressor
        opp_was_pre_agg = hand_ctx["opp_pre"] in ("raise", "3bet")
        if (
            not is_button
            and opp_was_pre_agg
            and style not in ("NIT", "CALLING_STATION")
            and not is_trapper
            and (street == "flop" or (street == "turn" and hand_ctx["opp_post_bets"] >= 1))
        ):
            if equity < 0.78:
                return {"action": "check", "debug": debug}
            if street == "flop" and equity >= 0.80 and random.random() < 0.35:
                return {"action": "check", "debug": debug}

        # Low-risk 7-2 in-position stab in small uncontested pots (Most Won Off 7-2 side prize)
        uncontested = hand_ctx["opp_post_calls"] == 0 and hand_ctx["opp_post_bets"] == 0
        if (
            is_seven_two
            and is_button
            and uncontested
            and equity < 0.50
            and eff_stack_bb >= 25
            and pot <= bb * 8
            and style in ("NIT", "BALANCED")
        ):
            return safe_raise(game_state, your_bet + int(pot * 0.48), debug)

        return choose_ev_optimal_bet(
            game_state, equity, raw_eq, bin_w, bin_win, hand_ctx, profile, is_button, eff_stack, debug
        )

    else:
        # Facing a postflop bet, check-raise, 3-bet, or massive overbet
        hero_bets_st = hand_ctx["hero_bets_street"]
        if is_maniac:
            raise_eq = 0.82 if hero_bets_st >= 1 else 0.74
        elif hero_bets_st >= 2:
            raise_eq = 0.91  # Only 4-bet/5-bet with the stone nuts
        elif facing_reraise:
            raise_eq = 0.86  # Cap raising wars vs postflop raises/3-bets
        else:
            raise_eq = 0.82 if style == "NIT" else 0.78

        if is_trapper and hand_ctx["opp_late_bets"] > 0 and not is_bluffer:
            raise_eq = min(0.92, raise_eq + 0.06)

        if equity >= raise_eq and not opp_all_in:
            target = max(int(current_bet * 2.4), current_bet + int(pot * big_bet_mult))
            return safe_raise(game_state, target, debug)

        if is_maniac:
            big_guard = maniac_big_guard(eff_stack_bb) if owed > your_stack * 0.35 else 0.0
        else:
            # Strict guard against massive deep overbets (>=1.8x pot) so we never pay off with 1 pair
            if owed_to_pot >= 1.8 and owed > your_stack * 0.35 and eff_stack_bb > 20:
                big_guard = max(dials["big_bet_guard"], 0.60)
            else:
                big_guard = dials["big_bet_guard"] if owed > your_stack * 0.45 else 0.0

        if equity >= max(required_call_eq, big_guard):
            return {"action": "call", "debug": debug}
        return {"action": "fold", "debug": debug}