Right now, your bilateral pipeline is stable, but parts of behavior may still use fallback/approximate paths in some situations.
The improvement means:

Make bilateral order generation truly end-to-end

Ensure both bid and ask decisions are fully honored in order placement logic every step (not fallback to unilateral behavior when edge cases happen).
Add targeted stress tests for inventory drift

Simulate asymmetric pressure (e.g., repeated buy-side fills with weak sell-side fills).
Verify inventory remains controlled and does not drift beyond expected bounds.
Validate circuit-breaker/closeout behavior under these skewed scenarios.
Why this is high leverage
It directly impacts live strategy realism and risk behavior.
It catches subtle bugs that don’t appear in average-case tests.
It improves confidence that RL policy behavior matches intended bilateral market-making mechanics under stress.
If you want, I can start this next as a tight test-first patch series (write failing scenario tests first, then implement minimal fixes to pass).
