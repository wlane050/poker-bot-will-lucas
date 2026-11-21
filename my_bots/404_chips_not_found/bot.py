"""
404 Chips Not Found — Version 1 (Baseline)
Clean equity + pot-odds + short-stack push/fold bot.
"""

import random
import eval7

# Build all 52 cards once when the bot loads so we don't recreate them every hand.
ALL_CARD_STRS = [r + s for r in "23456789TJQKA" for s in "shdc"]
CARD_MAP = {c: eval7.Card(c) for c in ALL_CARD_STRS}


def estimate_equity(your_cards: list, community_cards: list, sims: int = 250) -> float:
    """
    Estimate our chance of winning (0.0 to 1.0) against a random opponent hand
    by dealing out `sims` random endings from the remaining deck.
    """
    used = set(your_cards) | set(community_cards)
    deck = [CARD_MAP[c] for c in ALL_CARD_STRS if c not in used]

    hero = [CARD_MAP[c] for c in your_cards]
    board = [CARD_MAP[c] for c in community_cards]
    cards_needed_for_board = 5 - len(board)
    draw_count = 2 + cards_needed_for_board  # 2 for opponent + rest of the board

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


def safe_raise(game_state: dict, target_total: int, debug: dict) -> dict:
    """
    Make sure our raise is legal. If the raise would commit almost all our chips,
    just go all-in cleanly instead of leaving a tiny stack behind.
    """
    your_stack = game_state["your_stack"]
    your_bet = game_state["your_bet_this_street"]
    min_raise_to = game_state["min_raise_to"]
    max_total = your_stack + your_bet

    # Ensure raise is at least the minimum legal raise
    amount = max(min_raise_to, int(target_total))

    # If this raise uses 85%+ of our remaining stack, just shove all-in
    chips_needed = amount - your_bet
    if chips_needed >= your_stack * 0.85 or amount >= max_total:
        return {"action": "all_in", "debug": debug}

    return {"action": "raise", "amount": amount, "debug": debug}


def get_effective_stack(game_state: dict) -> int:
    """The smaller of our stack and our opponent's stack (the most chips at risk)."""
    my_id = game_state["your_bot_id"]
    my_stack = game_state["your_stack"]
    for p in game_state.get("players", []):
        if p["bot_id"] != my_id:
            return min(my_stack, p["stack"])
    return my_stack


def decide(game_state: dict) -> dict:
    # Handle the one-time warmup call before hand 1
    if game_state.get("type") == "warmup":
        return {"action": "check"}

    street = game_state["street"]
    pot = game_state["pot"]
    owed = game_state["amount_owed"]
    can_check = game_state["can_check"]
    your_stack = game_state["your_stack"]
    current_bet = game_state["current_bet"]
    bb = max(1, game_state["blinds"]["bb"])

    # 1. Measure our hand strength and the "price" to call (pot odds)
    equity = estimate_equity(game_state["your_cards"], game_state["community_cards"])
    pot_odds = owed / (pot + owed) if owed > 0 else 0.0
    eff_stack_bb = get_effective_stack(game_state) / bb

    debug = {
        "eq": round(equity, 3),
        "odds": round(pot_odds, 3),
        "eff_bb": round(eff_stack_bb, 1),
    }

    # 2. SHORT-STACK ENDGAME MODE (Preflop when effective stack <= 12 big blinds)
    # Rising blinds punish passive bots; push all-in with solid hands or fold.
    if street == "preflop" and eff_stack_bb <= 12:
        if owed == 0 or owed <= bb:
            # We're first to act or facing a limp: shove any above-average hand
            if equity >= 0.50:
                return {"action": "all_in", "debug": debug}
            return {"action": "check", "debug": debug} if can_check else {"action": "fold", "debug": debug}
        else:
            # Facing a raise or all-in: call if our equity beats the pot odds
            if equity >= max(pot_odds + 0.03, 0.48):
                return {"action": "all_in", "debug": debug}
            return {"action": "fold", "debug": debug}

    # 3. PREFLOP STRATEGY (Normal stacks)
    if street == "preflop":
        if owed <= bb and current_bet <= bb:
            # Unopened pot (we're on the button) or opponent just limped
            if equity >= 0.54:
                # Raise to 2.5x the big blind to steal blinds & build pots
                return safe_raise(game_state, int(bb * 2.5), debug)
            if can_check:
                return {"action": "check", "debug": debug}
            # Cheap to complete the small blind with decent hands
            if equity >= 0.42:
                return {"action": "call", "debug": debug}
            return {"action": "fold", "debug": debug}
        else:
            # Facing a preflop raise
            if equity >= 0.64:
                # Strong hand: re-raise to ~3x their bet
                return safe_raise(game_state, int(current_bet * 3), debug)
            # Don't call off our whole stack preflop with a mediocre hand
            big_bet_guard = 0.56 if owed > your_stack * 0.4 else 0.0
            if equity >= max(pot_odds + 0.05, big_bet_guard):
                return {"action": "call", "debug": debug}
            return {"action": "fold", "debug": debug}

    # 4. POSTFLOP STRATEGY (Flop, Turn, River)
    if can_check:
        # Nobody has bet yet on this street
        if equity >= 0.68:
            # Strong hand: bet 65% of the pot for value
            return safe_raise(game_state, int(pot * 0.65), debug)
        if equity >= 0.56:
            # Good medium hand: smaller 40% pot bet to farm small pots
            return safe_raise(game_state, int(pot * 0.40), debug)
        if equity < 0.35 and random.random() < 0.15:
            # Occasional 15% bluff to punish bots that fold whenever they miss
            return safe_raise(game_state, int(pot * 0.50), debug)
        return {"action": "check", "debug": debug}

    else:
        # Facing a bet or raise postflop
        if equity >= 0.78:
            # Monster hand: raise for maximum value
            target = max(int(current_bet * 2.5), current_bet + int(pot * 0.6))
            return safe_raise(game_state, target, debug)

        # Protect our stack: if they bet huge (>45% of our chips), require a real hand
        big_bet_guard = 0.58 if owed > your_stack * 0.45 else 0.0
        if equity >= max(pot_odds + 0.05, big_bet_guard):
            return {"action": "call", "debug": debug}

        return {"action": "fold", "debug": debug}