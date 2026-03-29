import nbformat as nbf
import os
import re

# ======================================================================
# CONFIGURATION
# ======================================================================
NOTEBOOK_PATH = "bilateral_mm_agent.ipynb"

def update_step_7_vector_action():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    modified = False
    for cell in nb.cells:
        if cell.cell_type == "code" and "envs.step(env_action)" in cell.source:
            # FIX 1: Concatenate bid and ask actions into a single 14-dim vector for AsyncVectorEnv
            # FIX 2: Ensure logging uses the original split actions for quota projection if needed
            
            new_source = cell.source.replace(
                'env_action = (bid.cpu().numpy(), ask.cpu().numpy())',
                'env_action = np.concatenate([bid.cpu().numpy(), ask.cpu().numpy()], axis=1)'
            )
            
            if new_source != cell.source:
                cell.source = new_source
                modified = True
                print("Updated step 7 training loop to use concatenated vector actions.")

    if modified:
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Successfully updated {NOTEBOOK_PATH}")
    else:
        print("No matches found in Step 7 for vector action fix.")

if __name__ == "__main__":
    update_step_7_vector_action()
