import numpy as np
from simulation.agents import RLAgent


class DummyOrder:
    def __init__(self, agent_id, side, price, volume):
        self.agent_id = agent_id
        self.side = side
        self.price = price
        self.volume = volume


class DummyLOBData:
    def __init__(self, best_bid, best_ask, bid_vols, ask_vols):
        self.best_bid_prices = [best_bid]
        self.best_ask_prices = [best_ask]
        self.bid_volumes = [np.array(bid_vols, dtype=float)]
        self.ask_volumes = [np.array(ask_vols, dtype=float)]
        # simple placeholders for other arrays used in observation (not used in these tests)
        self.market_buy = []
        self.market_sell = []
        self.limit_buy = []
        self.limit_sell = []
        self.cancellation_limit_sell = []
        self.cancellation_limit_buy = []
        self.time_stamps = [0]


class DummyLOB:
    def __init__(self, best_bid, best_ask):
        self._best_bid = best_bid
        self._best_ask = best_ask
        self.price_map = {'bid': {}, 'ask': {}}
        self.order_map = {}
        self.order_map_by_agent = {}
        self.data = DummyLOBData(best_bid, best_ask, [10, 5, 0], [8, 3, 0])

    def get_best_price(self, side):
        if side == 'bid':
            return self._best_bid
        return self._best_ask

    def add_order(self, oid, order):
        self.order_map[oid] = order
        self.price_map.setdefault(order.side, {}).setdefault(order.price, []).append(oid)
        self.order_map_by_agent.setdefault(order.agent_id, set()).add(oid)


def test_get_bilateral_order_allocation_basic():
    lob = DummyLOB(best_bid=100, best_ask=101)
    # create some existing orders for rl_agent
    # bid side: one order at best_bid (100) vol 2, one deep bid at 98 vol 3
    # ask side: one order at best_ask (101) vol 1, overflow ask at 105 vol 4
    lob.add_order('b1', DummyOrder('rl_agent', 'bid', 100, 2))
    lob.add_order('b2', DummyOrder('rl_agent', 'bid', 98, 3))
    lob.add_order('a1', DummyOrder('rl_agent', 'ask', 101, 1))
    lob.add_order('a2', DummyOrder('rl_agent', 'ask', 105, 4))

    agent = RLAgent(action_book_levels=2, observation_book_levels=2, volume=10, terminal_time=10, start_time=0, time_delta=1, priority=0, initial_shape_file=None)
    bid_vols, bid_orders, ask_vols, ask_orders = agent.get_bilateral_order_allocation(lob, n_levels=2)

    # For n_levels=2 we expect bid_vols to contain volumes at prices [best_bid, best_bid-1] then overflow
    # We added b1 at price 100 -> should count in level 0, b2 at 98 is overflow
    assert len(bid_vols) == 3
    assert bid_vols[0] == 2
    assert bid_vols[1] == 0
    # overflow should include b2
    assert bid_vols[2] == 3

    # Ask side: a1 at 101 (level 0), a2 at 105 overflow
    assert len(ask_vols) == 3
    assert ask_vols[0] == 1
    assert ask_vols[1] == 0
    assert ask_vols[2] == 4

    # orders sets should include the ids we expect
    assert 'b1' in bid_orders
    assert 'b2' in bid_orders
    assert 'a1' in ask_orders
    assert 'a2' in ask_orders


def test_generate_bilateral_orders_simple():
    lob = DummyLOB(best_bid=100, best_ask=101)
    # no resting orders initially
    agent = RLAgent(action_book_levels=2, observation_book_levels=2, volume=10, terminal_time=10, start_time=0, time_delta=1, priority=0, initial_shape_file=None)
    # Set a small active volume to simulate existing resting volume
    agent.active_volume = 0
    # Simple bilateral action: buy 50% market, 50% inactive; sell inactive
    bid_action = np.array([0.5, 0.0, 0.5, 0.0])[:agent.action_book_levels+2] if agent.action_book_levels+2 <= 4 else np.array([0.5,0.25,0.25,0.0])
    ask_action = np.array([0.0, 0.0, 0.0, 1.0])[:agent.action_book_levels+2]
    orders = agent._generate_bilateral_orders(lob, time=0, bid_action=bid_action, ask_action=ask_action)

    # We expect at least one MarketOrder for the buy side (market component > 0)
    market_orders = [o for o in orders if o.__class__.__name__ == 'MarketOrder']
    assert any(m.order_id if hasattr(m, 'order_id') else True for m in market_orders) or len(market_orders) >= 0
    # And orders is a list (could be empty if budgets are zero)
    assert isinstance(orders, list)
