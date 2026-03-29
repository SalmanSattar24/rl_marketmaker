# TRAINING CODE FOR BILATERAL MM PHASE 4
# Copy this into cell 19 of bilateral_mm_phase4.ipynb (replace the [EVAL] evaluation code temporarily)
# Run this cell first, then run the evaluation cells after

print("=" * 70)
print("TRAINING BILATERAL AGENT")
print("=" * 70)
print()

# Setup optimizer
optimizer = torch.optim.Adam(bilateral_agent.parameters(),
                            lr=TRAIN_PARAMS['learning_rate'])

training_returns = []
training_losses = []
start_time = time.time()

for iteration in range(TRAIN_PARAMS['num_iterations']):
    # Collect batch of trajectories
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_values = []
    batch_log_probs = []
    batch_entropies = []

    for step in range(TRAIN_PARAMS['num_steps']):
        obs, _ = market.reset(seed=42 + iteration * TRAIN_PARAMS['num_steps'] + step)
        ep_return = 0

        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            # Get action distribution from agent
            with torch.no_grad():
                actions, log_prob, entropy, value = bilateral_agent.get_action_and_value(obs_tensor)

            batch_states.append(obs_tensor.detach())
            batch_actions.append(actions)
            batch_log_probs.append(log_prob.detach())
            batch_entropies.append(entropy.detach())
            batch_values.append(value.detach())

            # Execute action in environment
            bid_action, ask_action = actions
            env_action = (bid_action[0].cpu().numpy(), ask_action[0].cpu().numpy())
            obs, reward, terminated, truncated, info = market.step(env_action)

            batch_rewards.append(reward)
            ep_return += reward

            if terminated:
                break

        training_returns.append(ep_return)

    # Compute advantages (simple GAE with lambda=1.0, gamma=1.0)
    # For terminal tasks: advantage = sum_future_rewards - value
    batch_returns = []
    batch_advantages = []

    cumulative_return = 0
    for i in range(len(batch_rewards) - 1, -1, -1):
        cumulative_return = batch_rewards[i] + TRAIN_PARAMS['gamma'] * cumulative_return
        batch_returns.insert(0, cumulative_return)

        if i < len(batch_values):
            advantage = cumulative_return - batch_values[i].item()
            batch_advantages.insert(0, advantage)

    # Convert to tensors
    returns_tensor = torch.tensor(batch_returns, dtype=torch.float32).to(device)
    advantages_tensor = torch.tensor(batch_advantages, dtype=torch.float32).to(device)

    # Re-compute log probs and values for gradient update
    optimizer.zero_grad()

    total_loss = 0
    for i, (state, action, old_log_prob) in enumerate(zip(batch_states, batch_actions, batch_log_probs)):
        if i >= len(batch_advantages):
            break

        _, log_prob, entropy, value = bilateral_agent.get_action_and_value(state, action=action)

        # Actor loss: policy gradient with baseline
        actor_loss = -(log_prob * advantages_tensor[i])

        # Critic loss: MSE on value function
        value_loss = 0.5 * (value.squeeze() - returns_tensor[i]) ** 2

        # Entropy bonus for exploration
        entropy_bonus = -entropy * TRAIN_PARAMS['entropy_coef']

        # Combined loss
        sample_loss = actor_loss + TRAIN_PARAMS['vf_coef'] * value_loss + entropy_bonus
        total_loss = total_loss + sample_loss

    # Backprop and update
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(bilateral_agent.parameters(), max_norm=0.5)
    optimizer.step()

    avg_loss = (total_loss.item() / len(batch_states)) if len(batch_states) > 0 else 0
    training_losses.append(avg_loss)

    if (iteration + 1) % 50 == 0:
        elapsed = time.time() - start_time
        avg_return = np.mean(training_returns[-50:])
        avg_loss_window = np.mean(training_losses[-50:])
        print(f"[{iteration+1:3d}/{TRAIN_PARAMS['num_iterations']}] "
              f"Return: {avg_return:8.2f} | Loss: {avg_loss_window:8.4f} | "
              f"Time: {elapsed:6.1f}s")

print(f"\n[OK] Training complete in {time.time() - start_time:.1f}s")
print(f"[INFO] Final 20-episode mean return: {np.mean(training_returns[-20:]):.4f}")
print("="*70 + "\n")

print("[INFO] Agent is now trained! Run the evaluation cells below.")
