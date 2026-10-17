"""Planner: a one-ply afterstate agent with a state-dependent evaluator in victory-point units.

V(state) = E(me) - beta * E(opponent), where E(player) estimates how many victory points that
player is worth now plus what its economy will turn into before the game ends:

  E = VP
    + H * (production value + build-rate value)       H = rounds left, shrinks as anyone nears 10
    + hand value (cards, completed builds, missing cards, discard risk)
    + expansion (ready settlement spots, spots one and two roads away)
    + army / longest-road races, development cards at their expected value

Build rates are "builds per round" for a city, a settlement and a development card, computed
from the player's production *after* converting surplus resources at its own port/bank rates, so
the evaluator sees complementarity (wheat without ore is not a city) instead of raw pip counts.

Chance is handled by expectation rather than by one sampled outcome:
  * ROLL is scored at the pre-roll state (production is already in E as an expectation);
  * BUY_DEVELOPMENT_CARD is scored with the drawn card replaced by the expected card;
  * MOVE_ROBBER is scored analytically (robber moved, steal at a fixed expected value), never by
    executing a random steal;
  * PLAY_KNIGHT_CARD and PLAY_ROAD_BUILDING are followed through their forced follow-ups (the
    robber move, the free roads) before scoring.

Only information the agent may see is used: the opponent is valued with public VPs plus the
expected hidden VP cards among its unplayed development cards, and its unplayed cards at their
expected value, never by type.
"""
import json
import os
import random

from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import ActionPrompt, ActionType, CITY, SETTLEMENT
from catanatron.models.map import number_probability

RES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
RIDX = {r: i for i, r in enumerate(RES)}
DEV_TYPES = ("KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY")
DEV_DECK = {"KNIGHT": 14, "VICTORY_POINT": 5, "ROAD_BUILDING": 2, "YEAR_OF_PLENTY": 2,
            "MONOPOLY": 2}

C_SETTLE = (1, 1, 1, 1, 0)
C_CITY = (0, 0, 0, 2, 3)
C_DEV = (0, 0, 1, 1, 1)
C_ROAD = (1, 1, 0, 0, 0)

WIN_VALUE = 1e6

DEFAULT_W = {
    "beta": 1.0,          # weight on the opponent's E
    "hk": 3.5,            # rounds of play left per VP still missing
    "hmin": 2.0,          # floor on the horizon
    "p_total": 0.15,      # VP per card of production, per round of horizon (v1: 0.10; C1)
    "p_city": 0.6,        # VP per city-per-round of build rate, per round of horizon
    "p_settle": 0.5,
    "p_dev": 0.15,
    "p_var": 0.02,        # per distinct resource produced, per round of horizon
    "h_card": 0.10,       # VP per card in hand that some reachable build needs
    "h_surplus": 0.03,    # VP per card in hand beyond what any reachable build needs
    "h_city": 0.35,       # bonus: can build a city right now (with trades)
    "h_settle": 0.35,     # bonus: can build a settlement right now on a ready spot
    "h_dev": 0.05,
    "h_miss_city": 0.04,  # per card still missing for a city
    "h_miss_settle": 0.04,
    "discard": 0.06,      # per card we would lose to a 7
    "spot_now": 0.9,      # VP per unit spot quality (cards/round) of the best ready spot
    "spot_now2": 0.3,     # ... of the second best ready spot
    "spot_r1": 0.45,      # best spot one road away
    "spot_r2": 0.15,      # best spot two roads away
    "spot_div": 0.15,     # spot quality bonus per resource the player does not yet produce
    "road_len": 0.02,
    "lr_margin": 0.12,    # per road of lead (clipped to +-3)
    "knight_played": 0.10,
    "army_margin": 0.20,  # per knight of lead (clipped to +-2)
    "dev_knight": 0.30,   # a knight in hand
    "dev_prog": 0.35,     # road building / year of plenty / monopoly in hand
    "steal": 0.20,        # expected value of a robber steal (gain + opponent's loss)
    "gate_soft": 0.85,    # v5: share of build-rate potential kept when no target exists right now
    "vp_floor": 1,        # v5c: never END_TURN while an affordable city/settlement is legal
}


def _load_weights():
    w = dict(DEFAULT_W)
    raw = os.environ.get("CATAN_PLANNER_W")
    if raw:
        w.update(json.loads(raw))
    return w


W = _load_weights()


# ---------------------------------------------------------------------------
# Static board geometry, cached per map object
# ---------------------------------------------------------------------------
_MAP_CACHE = {}


def _map_info(catan_map):
    key = id(catan_map)
    hit = _MAP_CACHE.get(key)
    if hit is not None and hit[0] is catan_map:
        return hit[1]
    if len(_MAP_CACHE) > 16:
        _MAP_CACHE.clear()
    land = catan_map.land_nodes
    node_tiles = {}
    for n, tiles in catan_map.adjacent_tiles.items():
        node_tiles[n] = tuple((RIDX[t.resource], 2.0 * number_probability(t.number), t)
                              for t in tiles if t.resource is not None)
    neigh = {n: tuple(u for u in STATIC_GRAPH.neighbors(n) if u in land) for n in land}
    info = {"node_tiles": node_tiles, "neigh": neigh}
    _MAP_CACHE[key] = (catan_map, info)
    return info


# ---------------------------------------------------------------------------
# Small economics helpers
# ---------------------------------------------------------------------------
def _trade_rates(board, color):
    ports = board.get_player_port_resources(color)
    base = 3 if None in ports else 4
    return [2 if r in ports else base for r in RES]


def _build_rate(P, t, c):
    """Builds of cost c per round from production P, trading surplus at rates t."""
    total_c = c[0] + c[1] + c[2] + c[3] + c[4]
    hi = (P[0] + P[1] + P[2] + P[3] + P[4]) / total_c
    lo = 0.0
    for _ in range(9):
        mid = 0.5 * (lo + hi)
        deficit = surplus = 0.0
        for r in range(5):
            d = mid * c[r] - P[r]
            if d > 0:
                deficit += d
            else:
                surplus -= d / t[r]
        if deficit <= surplus:
            lo = mid
        else:
            hi = mid
    return lo


def _missing(h, t, c):
    deficit = 0
    tradeable = 0
    for r in range(5):
        d = c[r] - h[r]
        if d > 0:
            deficit += d
        elif d < 0:
            tradeable += (-d) // t[r]
    return max(0, deficit - tradeable)


_NO_TRADE = (10 ** 9,) * 5


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class Context:
    """Per-decision facts the evaluator needs but cannot read from a (masked) world."""

    def __init__(self, obs):
        self.me = obs.me
        # The horizon is fixed from the ROOT position for the whole decision. Recomputing it per
        # afterstate from the leader's VP made every VP-gaining build shrink the horizon and so
        # wipe out more production value than the VP itself was worth: the agent refused to
        # build as it neared 10.
        lead = max(obs.my_actual_vps, obs.opp_public_vps)
        self.H = max(W["hmin"], W["hk"] * (10 - lead))
        unseen = dict(DEV_DECK)
        for d in DEV_TYPES:
            unseen[d] -= obs.my_devs[d] + obs.played_devs["me"][d] + obs.played_devs["opp"][d]
        total = max(1, sum(max(0, v) for v in unseen.values()))
        self.p_dev = {d: max(0, unseen[d]) / total for d in DEV_TYPES}
        self.p_vp = self.p_dev["VICTORY_POINT"]
        # value of one unseen card that is not a VP card (VP part is counted in vp)
        self.unknown_dev_value = (self.p_dev["KNIGHT"] * W["dev_knight"]
                                  + (self.p_dev["ROAD_BUILDING"] + self.p_dev["YEAR_OF_PLENTY"]
                                     + self.p_dev["MONOPOLY"]) * W["dev_prog"])


def _player_terms(game, color, other, ctx, is_me, drawn=None):
    """E(color) without the horizon factor applied to production (returned separately)."""
    st = game.state
    ps = st.player_state
    board = st.board
    info = _map_info(board.map)
    node_tiles = info["node_tiles"]
    i = st.color_to_index[color]
    j = st.color_to_index[other]
    k = f"P{i}_"
    ko = f"P{j}_"

    # --- victory points ---
    if is_me:
        vp = ps[k + "ACTUAL_VICTORY_POINTS"]
    else:
        n_dev = sum(ps[k + d + "_IN_HAND"] for d in DEV_TYPES)
        vp = ps[k + "VICTORY_POINTS"] + n_dev * ctx.p_vp

    # --- production (cards per round, both players' rolls), robber-aware ---
    robber_tile = board.map.land_tiles.get(board.robber_coordinate)
    bbc = st.buildings_by_color[color]
    settlements = bbc[SETTLEMENT]
    cities = bbc[CITY]
    P = [0.0, 0.0, 0.0, 0.0, 0.0]
    for n in settlements:
        for r, p, t in node_tiles[n]:
            if t is not robber_tile:
                P[r] += p
    for n in cities:
        for r, p, t in node_tiles[n]:
            if t is not robber_tile:
                P[r] += 2.0 * p
    produced = [x > 0 for x in P]
    variety = sum(produced)
    t = _trade_rates(board, color)

    settle_left = ps[k + "SETTLEMENTS_AVAILABLE"]
    city_left = ps[k + "CITIES_AVAILABLE"]
    roads_left = ps[k + "ROADS_AVAILABLE"]

    # --- expansion spots ---
    now, r1, r2 = _spots(board, color, info["neigh"])

    def q(n):
        v = 0.0
        new = set()
        for r, p, _t in node_tiles[n]:
            v += p
            if not produced[r]:
                new.add(r)
        return v + W["spot_div"] * len(new)

    spot_val = 0.0
    if settle_left > 0:
        if now:
            qs = sorted((q(n) for n in now), reverse=True)
            spot_val += W["spot_now"] * qs[0]
            if len(qs) > 1:
                spot_val += W["spot_now2"] * qs[1]
        if roads_left > 0:
            if r1:
                spot_val += W["spot_r1"] * max(q(n) for n in r1)
            if r2 and roads_left > 1:
                spot_val += W["spot_r2"] * max(q(n) for n in r2)

    has_spot = settle_left > 0 and (bool(now) or (roads_left > 0 and bool(r1)))
    # v5: soft gates. v2 gated the city rate on "a settlement to upgrade exists" and the settlement
    # rate on "a spot is ready or one road away", so the build that used the LAST target switched
    # its whole future-rate value off: upgrading the last settlement or settling the last reachable
    # spot scored below END_TURN (~12% of affordable cities/settlements declined vs F). Future
    # targets can still be created (new settlements -> new city targets, roads -> new spots), so
    # potential without a current target is discounted, not zeroed.
    gs = W["gate_soft"]
    if settlements and city_left > 0:
        city_g = 1.0
    elif city_left > 0 and settle_left > 0 and (has_spot or roads_left > 0):
        city_g = gs
    else:
        city_g = 0.0
    if has_spot:
        settle_g = 1.0
    elif settle_left > 0 and roads_left > 0:
        settle_g = gs
    else:
        settle_g = 0.0
    rate_city = city_g * _build_rate(P, t, C_CITY) if city_g else 0.0
    rate_settle = settle_g * _build_rate(P, t, C_SETTLE) if settle_g else 0.0
    rate_dev = _build_rate(P, t, C_DEV) if st.development_listdeck else 0.0
    prod_val = (W["p_total"] * (P[0] + P[1] + P[2] + P[3] + P[4])
                + W["p_city"] * rate_city + W["p_settle"] * rate_settle
                + W["p_dev"] * rate_dev + W["p_var"] * variety)

    # --- hand ---
    # Completion is judged on the cards actually held (no trades), so a trade that completes a
    # build is visibly worth making; cards beyond what any reachable build needs are surplus and
    # worth less, so trading surplus away for a needed card is a gain, not a loss.
    h = [ps[k + r + "_IN_HAND"] for r in RES]
    n_hand = h[0] + h[1] + h[2] + h[3] + h[4]
    can_city = bool(settlements) and city_left > 0
    can_settle = settle_left > 0 and bool(now)
    need = [0, 0, 0, 0, 0]
    targets = [C_DEV] if st.development_listdeck else []
    if can_city:
        targets.append(C_CITY)
    if can_settle or has_spot:
        targets.append(C_SETTLE)
    if roads_left > 0 and (r1 or r2):
        targets.append(C_ROAD)
    for c in targets:
        for r in range(5):
            if c[r] > need[r]:
                need[r] = c[r]
    useful = sum(min(h[r], need[r]) for r in range(5))
    hand_val = W["h_card"] * useful + W["h_surplus"] * (n_hand - useful)
    if can_city:
        m = _missing(h, _NO_TRADE, C_CITY)
        hand_val += W["h_city"] if m == 0 else -W["h_miss_city"] * m
    if can_settle:
        m = _missing(h, _NO_TRADE, C_SETTLE)
        hand_val += W["h_settle"] if m == 0 else -W["h_miss_settle"] * m
    if st.development_listdeck and _missing(h, _NO_TRADE, C_DEV) == 0:
        hand_val += W["h_dev"]
    if n_hand > 7:
        hand_val -= W["discard"] * (n_hand // 2)

    # --- races ---
    rl = ps[k + "LONGEST_ROAD_LENGTH"]
    rl_o = ps[ko + "LONGEST_ROAD_LENGTH"]
    kp = ps[k + "PLAYED_KNIGHT"]
    kp_o = ps[ko + "PLAYED_KNIGHT"]
    race_val = (W["road_len"] * rl + W["lr_margin"] * _clip(rl - rl_o, -3, 3)
                + W["knight_played"] * kp + W["army_margin"] * _clip(kp - kp_o, -2, 2))

    # --- development cards in hand ---
    if is_me:
        devs = {d: ps[k + d + "_IN_HAND"] for d in DEV_TYPES}
        dev_val = 0.0
        if drawn is not None and devs.get(drawn, 0) > 0:
            # the card the sampled world drew is not known at decision time: use the expectation
            devs[drawn] -= 1
            if drawn == "VICTORY_POINT":
                vp -= 1
            vp += ctx.p_vp
            dev_val += ctx.unknown_dev_value
        dev_val += W["dev_knight"] * devs["KNIGHT"] + W["dev_prog"] * (
            devs["ROAD_BUILDING"] + devs["YEAR_OF_PLENTY"] + devs["MONOPOLY"])
    else:
        n_dev = sum(ps[k + d + "_IN_HAND"] for d in DEV_TYPES)
        dev_val = n_dev * ctx.unknown_dev_value

    return vp, prod_val, hand_val + spot_val + race_val + dev_val


def _spots(board, color, neigh):
    comps = board.connected_components.get(color)
    if not comps:
        return [], [], []
    network = set()
    for c in comps:
        network |= c
    buildings = board.buildings
    roads = board.roads
    buildable = board.board_buildable_ids

    def enemy(n):
        b = buildings.get(n)
        return b is not None and b[0] != color

    now = [n for n in network if n in buildable]
    seen = set(network)
    lvl1, frontier2 = [], []
    for v in network:
        if enemy(v):
            continue
        for u in neigh[v]:
            if u in seen or (v, u) in roads:
                continue
            seen.add(u)
            if enemy(u):
                continue
            frontier2.append(u)
            if u in buildable:
                lvl1.append(u)
    lvl2 = []
    for v in frontier2:
        for u in neigh[v]:
            if u in seen or (v, u) in roads:
                continue
            seen.add(u)
            if not enemy(u) and u in buildable:
                lvl2.append(u)
    return now, lvl1, lvl2


def evaluate(game, ctx, drawn=None):
    me = ctx.me
    other = next(c for c in game.state.colors if c != me)
    ps = game.state.player_state
    if ps[f"P{game.state.color_to_index[me]}_ACTUAL_VICTORY_POINTS"] >= game.vps_to_win:
        return WIN_VALUE
    vp_m, prod_m, rest_m = _player_terms(game, me, other, ctx, True, drawn)
    vp_o, prod_o, rest_o = _player_terms(game, other, me, ctx, False)
    H = ctx.H
    e_m = vp_m + H * prod_m + rest_m
    e_o = vp_o + H * prod_o + rest_o
    return e_m - W["beta"] * e_o


# ---------------------------------------------------------------------------
# Action scoring
# ---------------------------------------------------------------------------
def _robber_value(game, action, ctx):
    coord, victim = action.value
    board = game.state.board
    saved = board.robber_coordinate
    board.robber_coordinate = coord
    try:
        v = evaluate(game, ctx)
    finally:
        board.robber_coordinate = saved
    if victim is not None:
        v += W["steal"]
    return v


def _value_after(game, ctx, depth=0):
    """Score a successor, following our own forced follow-ups (robber, free roads)."""
    st = game.state
    if game.winning_color() is not None:
        return WIN_VALUE if game.winning_color() == ctx.me else -WIN_VALUE
    if depth < 3 and st.current_color() == ctx.me:
        if st.current_prompt == ActionPrompt.MOVE_ROBBER:
            return max(_robber_value(game, a, ctx) for a in game.playable_actions)
        if st.is_road_building:
            # Greedy, not exhaustive: the best first free road, then the best second one. Trying
            # every pair cost ~50 ms and recurred on every decision while the card was held.
            best_g, best_v = None, None
            for a in game.playable_actions:
                if a.action_type != ActionType.BUILD_ROAD:
                    continue
                g = game.copy()
                g.execute(a)
                v = evaluate(g, ctx)
                if best_v is None or v > best_v:
                    best_g, best_v = g, v
            if best_g is not None:
                return _value_after(best_g, ctx, depth + 1)
    return evaluate(game, ctx)


def score_actions(world, playable, ctx):
    scores = []
    base = None
    for a in playable:
        at = a.action_type
        if at in (ActionType.ROLL, ActionType.END_TURN):
            if base is None:
                base = evaluate(world, ctx)
            v = base
        elif at == ActionType.MOVE_ROBBER:
            v = _robber_value(world, a, ctx)
        else:
            g = world.copy()
            try:
                g.execute(a)
            except Exception:
                scores.append(float("-inf"))
                continue
            if at == ActionType.BUY_DEVELOPMENT_CARD:
                rec = g.state.action_records[-1]
                v = evaluate(g, ctx, drawn=rec.result)
            else:
                v = _value_after(g, ctx)
        scores.append(v)
    return scores


# ---------------------------------------------------------------------------
# CPU safeguard. The per-game allowance stops growing after 96 decisions (1.5 s cap), so a very
# long game can run it down; past the allowance the harness plays random moves. With a normal
# budget none of this triggers and decisions are exactly v1's.
# ---------------------------------------------------------------------------
LEAN_CPU_S = 0.20       # below this: evaluate at most LEAN_MAX of each bulky action type
EMERGENCY_CPU_S = 0.05  # below this: no simulation at all, cheap priority policy
LEAN_MAX = 3


def _lean_subset(world, playable, me):
    """Indices to evaluate when CPU is scarce: every action except roads and bank trades, plus
    the LEAN_MAX most promising of each by a cheap, copy-free prior."""
    board = world.state.board
    buildable = board.board_buildable_ids
    ps = world.state.player_state
    k = f"P{world.state.color_to_index[me]}_"
    hand = [ps[k + r + "_IN_HAND"] for r in RES]
    keep, roads, trades = [], [], []
    for i, a in enumerate(playable):
        if a.action_type == ActionType.BUILD_ROAD:
            u, v = a.value
            far = v if board.get_node_color(u) == me or u in _network_nodes(board, me) else u
            near_spot = far in buildable or any(n in buildable for n in STATIC_GRAPH.neighbors(far))
            roads.append((1 if near_spot else 0, i))
        elif a.action_type == ActionType.MARITIME_TRADE:
            give = a.value[0]
            get = a.value[-1]
            trades.append((hand[RIDX[give]] - hand[RIDX[get]], i))
        else:
            keep.append(i)
    roads.sort(reverse=True)
    trades.sort(reverse=True)
    return keep + [i for _, i in roads[:LEAN_MAX]] + [i for _, i in trades[:LEAN_MAX]]


def _network_nodes(board, color):
    out = set()
    for c in board.connected_components.get(color, ()):
        out |= c
    return out


def _emergency_choice(obs, playable):
    """No copies, no evaluation: build if possible, hurt the opponent with the robber, discard
    the most plentiful card, otherwise roll / end the turn."""
    order = (ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT)
    for t in order:
        for i, a in enumerate(playable):
            if a.action_type == t:
                return i
    board = obs.board
    robber = [(i, a) for i, a in enumerate(playable) if a.action_type == ActionType.MOVE_ROBBER]
    if robber:
        def hurt(a):
            coord, victim = a.value
            tile = board.map.land_tiles[coord]
            mine = theirs = 0
            for n in tile.nodes.values():
                b = board.buildings.get(n)
                if b is not None:
                    w = 2 if b[1] == CITY else 1
                    if b[0] == obs.me:
                        mine += w
                    else:
                        theirs += w
            p = number_probability(tile.number) if tile.number else 0.0
            return (theirs - 2 * mine) * p + (0.01 if victim is not None else 0.0)
        return max(robber, key=lambda ia: hurt(ia[1]))[0]
    discards = [(i, a) for i, a in enumerate(playable) if a.action_type == ActionType.DISCARD_RESOURCE]
    if discards:
        return max(discards, key=lambda ia: obs.my_resources[ia[1].value])[0]
    for t in (ActionType.ROLL, ActionType.END_TURN):
        for i, a in enumerate(playable):
            if a.action_type == t:
                return i
    return 0


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions):
        if len(playable_actions) == 1:
            return 0
        cpu_left = obs.cpu_left
        if cpu_left is not None and cpu_left < EMERGENCY_CPU_S:
            try:
                return _emergency_choice(obs, playable_actions)
            except Exception:
                return 0
        saved = random.getstate()  # simulated steals/draws must not touch the real dice stream
        try:
            world = obs.sample_world(self.rng)
            ctx = Context(obs)
            if cpu_left is not None and cpu_left < LEAN_CPU_S:
                idx = _lean_subset(world, playable_actions, obs.me)
                sub = score_actions(world, [playable_actions[i] for i in idx], ctx)
                return idx[max(range(len(sub)), key=sub.__getitem__)]
            scores = score_actions(world, playable_actions, ctx)
            best = max(range(len(scores)), key=scores.__getitem__)
            if W["vp_floor"] and playable_actions[best].action_type == ActionType.END_TURN:
                vp_builds = [i for i, a in enumerate(playable_actions)
                             if a.action_type in (ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT)]
                if vp_builds:
                    best = max(vp_builds, key=scores.__getitem__)
            return best
        except Exception:
            return self._fallback(obs, playable_actions)
        finally:
            random.setstate(saved)

    def _fallback(self, obs, playable_actions):
        try:
            from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn
            fn = get_value_fn("base_fn", DEFAULT_WEIGHTS)
            world = obs.sample_world(self.rng)
            best_i, best_v = 0, None
            for i, a in enumerate(playable_actions):
                g = world.copy()
                try:
                    g.execute(a)
                except Exception:
                    continue
                v = fn(g, obs.me)
                if best_v is None or v > best_v:
                    best_i, best_v = i, v
            return best_i
        except Exception:
            return 0
