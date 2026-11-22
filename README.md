# 404_chips_not_found — Heads-Up No-Limit Hold'em Poker Bot

Autonomous heads-up No-Limit Texas Hold'em bot built in Python (`eval7`) for the **IEEE uOttawa × Larus Poker Bot Hackathon**.

* **Result:** Tied for **3rd place out of 21 teams** in the live knockout bracket & winner of the **"Most Won Off 7-2"** prize.
* **Entry Point:** [`my_bots/404_chips_not_found/bot.py`](my_bots/404_chips_not_found/bot.py)
* **Performance:** ~1.4 ms average decision time, 0 crashes/timeouts under 0.5 CPU / 768 MB Docker sandbox limits.

## Key Highlights
* **Continuous Bayesian Profiling:** Tracks ~30 granular opponent frequencies and showdown reveals using Beta-prior smoothing instead of rigid opponent archetypes.
* **Range-Weighted Equity:** Exact top-X% preflop range lookups, exact 990-combo river enumeration, and time-bounded flop/turn Monte Carlo rollouts anchored to Minimum Defense Frequency (MDF).
* **Situation-Based Bet Sizing:** Ties bet sizes to street, board texture, and opponent fold rates rather than hand strength so bluffs and value bets remain indistinguishable.
* **Tournament & Endgame Math:** Ante-aware Nash push/fold at <= 15 BB, freezeout stack-preservation premiums, and a conditional 7-2 shove heuristic tailored to the tournament's side-prize scoring rules.
