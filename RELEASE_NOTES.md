# Release notes

## v0.2-stable-batch-fixes

### Summary
Stability and correctness checkpoint for bilateral MM simulation/training stack, including terminal behavior fixes, observation-shape hardening, cleanup, and test hygiene.

### Included changes

- **Terminal closeout + reward decomposition**
  - Added deterministic terminal inventory closeout path.
  - Added explicit reward decomposition fields (realized, inventory, terminal).

- **Dynamic observation sizing + OFI toggle**
  - Replaced hardcoded observation-size assumptions with config-aware dynamic sizing.
  - Added optional OFI feature flag integration and observation-length guardrails.

- **Stale file cleanup**
  - Removed obsolete one-off patch scripts and temporary summary artifacts.

- **Integration test warning cleanup**
  - Replaced list-of-ndarray tensor construction with `np.stack(...)` + `torch.as_tensor(...)` in integration tests.

### Verification

- Full test run result at this checkpoint: **29 passed**.
