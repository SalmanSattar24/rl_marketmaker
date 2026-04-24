"""
Stress tests for inventory drift and bilateral execution parity under adverse conditions.
"""
import sys
import os
import numpy as np
import pytest

current_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from limit_order_book.limit_order_book import LimitOrderBook, LimitOrder, MarketOrder
from simulation.agents import RLAgent
from simulation.market_gym import Market

class TestInventoryStress:
    def setup_method(self):
        self.config = {
            'market_env': 'strategic',
            'execution_agent': 'rl_agent',
            'volume': 100,
            'seed': 42,
            'terminal_time': 1000,
            'time_delta': 10,
            'drop_feature': None,
            'inventory_max': 20,
            'penalty_weight': 1.0,
            'use_ofi': False
        }
        self.market = Market(self.config)
        self.market.reset(seed=42)

    def test_asymmetric_fill_pressure(self):
        """Test how the agent and environment handle repeated one-sided fills without drift exploding."""
        # Force the environment to process several steps where only the bid side gets hit
        for _ in range(5):
            # RL agent generates orders (bilateral)
            action = np.zeros(14)
            # uniform action
            action[:] = 1.0 / 7.0 
            obs, reward, terminated, truncated, info = self.market.step(action)
            
            # Hit the RL agent's bids with massive market sells from another agent
            market_sell = MarketOrder('adversary', 'ask', 20, time=self.market.last_t)
            self.market.lob.process_order(market_sell)
            
            # Update inventory in Market
            exec_id = self.market.execution_agent_id
            self.market.agent_inventory = self.market.lob.agent_net_inventory.get(exec_id, 0)
            
            if self.market.circuit_breaker_triggered:
                break
                
        # The agent's inventory should hit the circuit breaker or be bounded by the inventory max
        assert abs(self.market.agent_inventory) <= self.config['inventory_max'] or self.market.circuit_breaker_triggered

    def test_empty_book_side_handling(self):
        """Test bilateral order generation when one side of the book is completely empty."""
        agent = self.market.agents[self.market.execution_agent_id]
        
        # Clear the ask side of the book
        self.market.lob.price_map['ask'].clear()
        self.market.lob.price_volume_map['ask'].clear()
        self.market.lob.data.best_ask_prices[-1] = np.nan
        
        # Try to generate bilateral orders
        action = np.zeros(14)
        action[:] = 1.0 / 7.0
        
        try:
            orders = agent.generate_order(self.market.lob, self.market.last_t + 10, action)
            # Should not crash, and should return some orders
            assert orders is not None
        except Exception as e:
            pytest.fail(f"Agent crashed on empty book side: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
