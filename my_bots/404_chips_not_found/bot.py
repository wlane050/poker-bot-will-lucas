"""
404 Chips Not Found — Version 2 (Exploitative Profiler)
Reads opponent habits from match_action_log and adapts its strategy
against Nits, Calling Stations, and Maniacs.
"""

import random
import eval7

ALL_CARD_STRS = [r + s for r in "23456789TJQKA" for s in "shdc"]
CARD_MAP = {c: eval7.Card(c) for c in ALL_CARD_STRS}

# Match-long memory (resets automatically at the start of every new match)
HAND_HISTORY = {}  # {hand_num: list of action dicts}


def estimate_equity(your_cards: list, community_cards: list, sims: int = 250) -> float:
    """Estimate win probability (0.0 to 1.0) via Monte Carlo simulation."""
    used = set(your_cards) | set(community_cards)
    deck = [CARD_MAP[c] for c in ALL_CARD_STRS if c not in used]

    hero = [CARD_MAP[c] for c in your_cards]
    board = [CARD_MAP[c] for c in community_cards]
    draw_count = 2 + (5 - len(board))

    wins = 0.0
    for _ in range(sims):
        sample = random.sample(deck, draw_count)
        opp_hand = sample[:2]
        full_board = board + sample[2:]

        hero_score = eval7.evaluate(hero + full_board)
        opp_score = eval7.evaluate(opp_hand + full_board)

        if hero_score > opp_score:
            wins += 1.0
        elif hero_score == opp_score:
            wins += 0.5

    return wins / sims


def update_memory(game_state: dict) -> None:
    """
    Save actions from match_action_log into HAND_HISTORY.
    match_action_log only keeps the last 200 entries, so saving by hand_num
    lets us remember the whole match without double-counting.
    """
    log = game_state.get("match_action_log") or []
    grouped = {}
    for entry in log:
        h = entry.get("hand_num")
        if h is not None:
            grouped.setdefault(h, []).append(entry)

    for h, entries in grouped.items():
        # Only overwrite if we have a more complete record of that hand
        if len(entries) >= len(HAND_HISTORY.get(h, [])):
            HAND_HISTORY[h] = entries


def profile_opponent(my_id: str) -> dict:
    """
    Analyze HAND_HISTORY to measure 4 opponent habits and classify their style:
    - BALANCED (default during hands 1-10 or against solid bots)
    - CALLING_STATION (rarely folds, rarely raises)
    - NIT (folds way too often, plays very few hands)
    - MANIAC (raises/shoves constantly)
    """
    hands_seen = 0
    vpip_hands = 0
    faced_bet = 0
    folded_to_bet = 0
    passive_actions = 0  # check, call
    aggr_actions = 0     # raise, all_in
    all_in_count = 0

    for entries in HAND_HISTORY.values():
        hands_seen += 1
        opp_put_chips_preflop = False
        current_street = None
        hero_bet_this_street = False

        for e in entries:
            action = e.get("action")
            bot_id = e.get("bot_id")
            street = e.get("street")

            # Skip blind posts and showdown reveals
            if action in ("small_blind", "big_blind", "ante", "show"):
                continue

            if street != current_street:
                current_street = street
                hero_bet_this_street = False

            if bot_id == my_id:
                if action in ("raise", "all_in"):
                    hero_bet_this_street = True
            else:
                # Opponent action
                if street == "preflop" and action in ("call", "raise", "all_in"):
                    opp_put_chips_preflop = True

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

    # Wait until we have at least 10 hands before locking in an exploit
    style = "BALANCED"
    if hands_seen >= 10:
        if agg_rate >= 0.45 or all_in_rate >= 0.22:
            style = "MANIAC"
        elif faced_bet >= 6 and fold_rate >= 0.52:
            style = "NIT"
        elif vpip <= 0.42 and agg_rate < 0.35:
            style = "NIT"
        elif faced_bet >= 6 and fold_rate <= 0.22 and agg_rate <= 0.22:
            style = "CALLING_STATION"

    return {
        "style": style,
        "hands": hands_seen,
        "vpip": round(vpip, 2),
        "fold": round(fold_rate, 2),
        "agg": round(agg_rate, 2),
    }


def get_strategy_dials(style: str) -> dict:
    """
    Return the tuning knobs for each opponent archetype.
    This is the core of our exploitative strategy.
    """
    if style == "NIT":
        # Punish folders: steal wide, bluff often with medium bets,
        # but respect them when they suddenly wake up and bet big.
        return {
            "preflop_raise_eq": 0.43,
            "bluff_freq": 0.50,
            "value_bet_eq": 0.52,
            "big_bet_mult": 0.55,
            "small_bet_mult": 0.40,
            "call_margin": 0.09,
            "big_bet_guard": 0.66,
        }
    if style == "CALLING_STATION":
        # Never bluff someone who never folds; bet huge with good hands!
        return {
            "preflop_raise_eq": 0.52,
            "bluff_freq": 0.00,
            "value_bet_eq": 0.52,
            "big_bet_mult": 0.85,
            "small_bet_mult": 0.55,
            "call_margin": 0.05,
            "big_bet_guard": 0.62,
        }
    if style == "MANIAC":
        # Stop bluffing them; widen our calls and let them shove into our good hands.
        return {
            "preflop_raise_eq": 0.55,
            "bluff_freq": 0.05,
            "value_bet_eq": 0.58,
            "big_bet_mult": 0.70,
            "small_bet_mult": 0.45,
            "call_margin": -0.02,
            "big_bet_guard": 0.48,
        }
    # BALANCED default
    return {
        "preflop_raise_eq": 0.52,
        "bluff_freq": 0.15,
        "value_bet_eq": 0.56,
        "big_bet_mult": 0.65,
        "small_bet_mult": 0.40,
        "call_margin": 0.05,
        "big_bet_guard": 0.58,
    }


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


def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        return {"action": "check"}

    # 1. Update opponent memory and pick our exploitative dials
    update_memory(game_state)
    profile = profile_opponent(game_state["your_bot_id"])
    dials = get_strategy_dials(profile["style"])

    street = game_state["street"]
    pot = game_state["pot"]
    owed = game_state["amount_owed"]
    can_check = game_state["can_check"]
    your_stack = game_state["your_stack"]
    current_bet = game_state["current_bet"]
    bb = max(1, game_state["blinds"]["bb"])

    # 2. Calculate hand strength and pot odds
    equity = estimate_equity(game_state["your_cards"], game_state["community_cards"])
    pot_odds = owed / (pot + owed) if owed > 0 else 0.0
    eff_stack_bb = get_effective_stack(game_state) / bb

    debug = {
        "style": profile["style"],
        "h": profile["hands"],
        "fold": profile["fold"],
        "agg": profile["agg"],
        "eq": round(equity, 3),
        "odds": round(pot_odds, 3),
    }

    # 3. SHORT-STACK ENDGAME MODE (Preflop when effective stack <= 12 BB)
    if street == "preflop" and eff_stack_bb <= 12:
        # Shove slightly wider against Nits who fold too much
        shove_eq = 0.45 if profile["style"] == "NIT" else 0.50
        if owed == 0 or owed <= bb:
            if equity >= shove_eq:
                return {"action": "all_in", "debug": debug}
            return {"action": "check", "debug": debug} if can_check else {"action": "fold", "debug": debug}
        else:
            call_eq = max(pot_odds + dials["call_margin"], 0.46)
            if equity >= call_eq:
                return {"action": "all_in", "debug": debug}
            return {"action": "fold", "debug": debug}

    # 4. PREFLOP STRATEGY
    if street == "preflop":
        if owed <= bb and current_bet <= bb:
            if equity >= dials["preflop_raise_eq"]:
                return safe_raise(game_state, int(bb * 2.5), debug)
            if can_check:
                return {"action": "check", "debug": debug}
            if equity >= 0.42:
                return {"action": "call", "debug": debug}
            return {"action": "fold", "debug": debug}
        else:
            # Facing a preflop raise
            if equity >= 0.64:
                return safe_raise(game_state, int(current_bet * 3), debug)
            big_guard = dials["big_bet_guard"] if owed > your_stack * 0.40 else 0.0
            if equity >= max(pot_odds + dials["call_margin"], big_guard):
                return {"action": "call", "debug": debug}
            return {"action": "fold", "debug": debug}

    # 5. POSTFLOP STRATEGY (Flop, Turn, River)
    if can_check:
        if equity >= 0.68:
            return safe_raise(game_state, int(pot * dials["big_bet_mult"]), debug)
        if equity >= dials["value_bet_eq"]:
            return safe_raise(game_state, int(pot * dials["small_bet_mult"]), debug)
        if equity < 0.38 and random.random() < dials["bluff_freq"]:
            # Under 40% equity bluff: punishes folders and can score Biggest Bluffer!
            return safe_raise(game_state, int(pot * 0.50), debug)
        return {"action": "check", "debug": debug}

    else:
        # Facing a postflop bet or raise
        if equity >= 0.78:
            target = max(int(current_bet * 2.5), current_bet + int(pot * dials["big_bet_mult"]))
            return safe_raise(game_state, target, debug)

        big_guard = dials["big_bet_guard"] if owed > your_stack * 0.45 else 0.0
        if equity >= max(pot_odds + dials["call_margin"], big_guard):
            return {"action": "call", "debug": debug}
        return {"action": "fold", "debug": debug}