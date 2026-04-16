import numpy as np

from limit_order_book.limit_order_book import LimitOrderBook, LimitOrder, MarketOrder
from simulation.agents import RLAgent


def test_normalize_action_simplex():
    agent = RLAgent(action_book_levels=2, observation_book_levels=2, volume=10,
                    terminal_time=10, start_time=0, time_delta=1, priority=0)

    raw = np.array([1.0, -2.0, 0.5, 0.0], dtype=np.float32)
    out = agent._normalize_action_simplex(raw)
    assert out.shape == raw.shape
    assert np.all(out >= 0)
    assert np.isclose(float(np.sum(out)), 1.0)


def test_get_bilateral_order_allocation_simple():
    # register both rl_agent and a counterparty to create a present book
    lob = LimitOrderBook(list_of_agents=["rl_agent", "other"], level=5)

    # create some existing resting orders for rl_agent
    lo1 = LimitOrder("rl_agent", side='ask', price=101, volume=3, time=0)
    lo2 = LimitOrder("rl_agent", side='bid', price=99, volume=2, time=0)
    # create a counterparty to ensure market structure
    lo3 = LimitOrder("other", side='ask', price=102, volume=5, time=0)
    lo4 = LimitOrder("other", side='bid', price=98, volume=5, time=0)

    lob.process_order(lo1)
    lob.process_order(lo2)
    lob.process_order(lo3)
    lob.process_order(lo4)

    agent = RLAgent(action_book_levels=2, observation_book_levels=2, volume=10,
                    terminal_time=10, start_time=0, time_delta=1, priority=0)

    bid_vols, bid_orders, ask_vols, ask_orders = agent.get_bilateral_order_allocation(lob, n_levels=2)

    # Expect ask_vols[0] to include the 101 ask volume we added and bid_vols[0] the 99 bid volume
    assert isinstance(ask_vols, list) and isinstance(bid_vols, list)
    assert ask_vols[0] == 3
    assert bid_vols[0] == 2
    # overflow (last element) should be zero in this simple case
    assert ask_vols[-1] == 0
    assert bid_vols[-1] == 0


def test_generate_bilateral_orders_creates_expected_orders():
    # Construct LOB with a counterparty so best bid/ask are defined
    lob = LimitOrderBook(list_of_agents=["rl_agent", "other"], level=5)
    lob.process_order(LimitOrder("other", side='bid', price=100, volume=5, time=0))
    lob.process_order(LimitOrder("other", side='ask', price=102, volume=5, time=0))

    agent = RLAgent(action_book_levels=2, observation_book_levels=2, volume=10,
                    terminal_time=10, start_time=0, time_delta=1, priority=0)

    # Simple bilateral actions: place full non-market volume at best levels on both sides
    per_side_len = agent.action_space_length
    bid_action = np.zeros(per_side_len, dtype=np.float32)
    ask_action = np.zeros(per_side_len, dtype=np.float32)
    # allocate all to L0 (index 1 because index 0 is market)
    bid_action[1] = 1.0
    ask_action[1] = 1.0

    orders = agent.generate_order(lob, time=0, action=(bid_action, ask_action))

    # Expect at least two limit orders (one bid, one ask) in the returned list
    assert orders is not None
    has_bid = any(isinstance(o, LimitOrder) and o.side == 'bid' for o in orders)
    has_ask = any(isinstance(o, LimitOrder) and o.side == 'ask' for o in orders)
    assert has_bid and has_ask
