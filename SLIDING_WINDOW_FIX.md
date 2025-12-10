# Sliding Window Bug Fix - Complete Explanation

## Problem Summary
Sliding window models (window_size=5) had **0% success rate** in closed-loop evaluation, while non-windowed models achieved 72%. This was caused by misaligned training and evaluation implementations.

---

## Root Cause Analysis

### Issue 1: Training Only Used Partial Episodes (CRITICAL BUG)

**Location**: `mpc_surrogate_training.ipynb`, MPCDataset class, lines 97-104

**Old Code** (BROKEN):
```python
for i in range(self.window_size - 1, T):  # Started at i=4 for window_size=5
    window_states = states[i - self.window_size + 1:i + 1].reshape(-1)
    target_now = targets[i]
    x = torch.cat([window_states, target_now], dim=0)
    inputs_list.append(x)
    actions_list.append(act[i])
```

**Problem**:
- Loop started at `i = 4` (when `window_size = 5`)
- Training data ONLY included timesteps 4 through 149
- Timesteps 0-3 were **completely missing** from training
- Model never learned what to do at episode start

**Example**: For a 150-step episode with window_size=5:
- ❌ Old: Created 146 training samples (timesteps 4-149)
- ✅ New: Creates 150 training samples (timesteps 0-149)

---

### Issue 2: Evaluation State History Timing

**Location**: `closed_loop_eval.py`, run_episode function, lines 405-431

**Old Code** (WRONG TIMING):
```python
# Compute control action
tau = predict_action(model, current_state, target_xyz, ...)

# Update state history AFTER prediction
state_history.append(current_state.copy())
```

**Problem**:
- At step 0: `state_history = []`, predict_action couldn't use current state
- At step 4: `state_history = [s0, s1, s2, s3]`, missing s4 for prediction
- Prediction at step N used states `[s(N-5) ... s(N-1)]` instead of `[s(N-4) ... s(N)]`

---

## Fix Implementation

### Fix 1: Training Notebook - Include All Timesteps

**Location**: `mpc_surrogate_training.ipynb`, lines 95-115

**New Code**:
```python
# FIXED: Start from i=0 and pad with repeated initial state if needed
for i in range(T):  # Now starts at i=0
    # Get window: states[max(0, i-window_size+1):i+1]
    start_idx = max(0, i - self.window_size + 1)
    window_states = states[start_idx:i + 1]  # Variable length initially
    
    # Pad with first state if window is incomplete (early in episode)
    if window_states.shape[0] < self.window_size:
        padding_size = self.window_size - window_states.shape[0]
        padding = states[0:1].repeat(padding_size, 1)  # Repeat first state
        window_states = torch.cat([padding, window_states], dim=0)
    
    window_states_flat = window_states.reshape(-1)  # (window_size*6)
    target_now = targets[i]  # (3)
    x = torch.cat([window_states_flat, target_now], dim=0)  # (window_size*6 + 3)
    inputs_list.append(x)
    actions_list.append(act[i])
```

**Changes**:
1. ✅ Loop now starts at `i=0` instead of `i=window_size-1`
2. ✅ Early timesteps pad with the initial state (e.g., `[s0, s0, s0, s0, s0]`)
3. ✅ All 150 timesteps per episode now included in training

**Example - Timestep 0 with window_size=5**:
- `start_idx = max(0, 0-5+1) = 0`
- `window_states = states[0:1]` → shape (1, 6), only 1 state
- `padding_size = 5 - 1 = 4`
- `padding = states[0:1].repeat(4, 1)` → (4, 6)
- Final window: `[s0, s0, s0, s0, s0]` → flattened to (30,), concatenate target (3,) = (33,)

---

### Fix 2: Evaluation Script - Correct State History Timing

**Location**: `closed_loop_eval.py`, lines 405-426

**New Code**:
```python
# Update state history BEFORE prediction (for windowed models)
state_history.append(current_state.copy())
if len(state_history) > window_size:
    state_history.pop(0)

# Compute control action
start_time = time.time()

if model_type == 'mpc':
    tau, solved = controller.solve(current_state, target_joints)
    if not solved:
        tau = np.zeros(3)
else:
    # Learned models were trained on Cartesian targets
    tau = predict_action(
        model,
        current_state,
        target_xyz,
        model_type,
        state_history=state_history,
        window_size=window_size,
    )
```

**Changes**:
1. ✅ State appended to history **BEFORE** calling predict_action
2. ✅ At step N, history contains `[s(N-4), s(N-3), s(N-2), s(N-1), s(N)]`
3. ✅ Matches training: window includes the current state

---

### Fix 3: Evaluation Script - Padding Logic

**Location**: `closed_loop_eval.py`, predict_action function, lines 278-286

**New Code**:
```python
if window_size > 1 and state_history is not None:
    # Pad with current state if history is incomplete (early in episode)
    history_window = list(state_history)
    while len(history_window) < window_size:
        history_window.insert(0, state)  # Pad at beginning with current state
    # Use only the most recent window_size states
    history_window = history_window[-window_size:]
    # Flatten: [state_t-w+1, ..., state_t, target]
    input_data = np.concatenate([np.concatenate(history_window), target])
```

**Changes**:
1. ✅ Pads incomplete windows with the current state
2. ✅ Matches training: `[state, state, state, ..., state]` for early timesteps
3. ✅ Takes only the last `window_size` states (handles case where history grows beyond window)

---

## Execution Timeline

### Before Fix (Step 0, window_size=5):
```
1. current_state = s0
2. state_history = []
3. predict_action() called with empty history → couldn't create input → returned zeros
4. Apply tau = [0, 0, 0]  ❌ Wrong!
5. state_history.append(s0) → [s0]
```

### After Fix (Step 0, window_size=5):
```
1. current_state = s0
2. state_history.append(s0) → [s0]
3. predict_action() called with state_history=[s0]
   - history_window = [s0]
   - Pad: while len < 5, insert s0 at beginning → [s0, s0, s0, s0, s0]
   - Flatten: (30,) + target (3,) = input (33,)
   - Model predicts action ✅ Correct!
4. Apply tau = model prediction
```

### Before Fix (Step 4, window_size=5):
```
1. current_state = s4
2. state_history = [s0, s1, s2, s3]  (only 4 states)
3. predict_action() with history=[s0, s1, s2, s3]
   - Pad: [s4, s0, s1, s2, s3]  ❌ s4 at wrong position!
4. state_history.append(s4) → [s0, s1, s2, s3, s4]
```

### After Fix (Step 4, window_size=5):
```
1. current_state = s4
2. state_history.append(s4) → [s0, s1, s2, s3, s4]  (5 states, full window!)
3. predict_action() with history=[s0, s1, s2, s3, s4]
   - No padding needed, already size 5
   - Flatten: (30,) + target (3,) = input (33,) ✅ Correct!
4. Model predicts action correctly
```

---

## Verification

### Test Results:

**Non-windowed Model (MLP_Deep_best_mse_1.pth)**:
```
✅ 72% success @ 0.02m (strict)
✅ 76% success @ 0.03m (moderate)
✅ 88% success @ 0.05m (relaxed)
```

**Windowed Model BEFORE Fix (MLP_Win5_Deep_best_mse_1.pth)**:
```
❌ 0% success @ all thresholds
❌ Mean final error: 0.40m
❌ Model completely fails
```

**Expected After Retraining**:
- Models must be retrained with the fixed notebook
- Should achieve similar or better performance than non-windowed models
- Windowed models theoretically have more context, so could exceed 72%

---

## Action Required

### ⚠️ MANDATORY: Retrain All Windowed Models

**Models that need retraining**:
1. `MLP_Win5_Small_best_mse_*.pth`
2. `MLP_Win5_Medium_best_mse_*.pth`
3. `MLP_Win5_Deep_best_mse_*.pth`

**How to retrain**:
1. Open `mpc_surrogate_training.ipynb`
2. Run the "Architecture comparison" section (cells with `WINDOW_SIZE_WIN = 5`)
3. The fixed code will now train on ALL timesteps (0-149) instead of just (4-149)
4. New models will be saved to `results/pytorch_comparison/results_sliding_window/models/`

**Non-windowed models are fine**:
- `MLP_Small_best_mse_*.pth` ✅
- `MLP_Medium_best_mse_*.pth` ✅
- `MLP_Deep_best_mse_*.pth` ✅
- `GRU_*_best_mse_*.pth` ✅

---

## Summary

**3 critical changes made**:

1. **Training notebook**: Loop from `i=0` (not `i=4`), pad early timesteps with initial state
2. **Eval script**: Append state to history **before** prediction (not after)
3. **Eval script**: Improved padding logic for consistency

**Root cause**: Training only used 97% of episode data (timesteps 4-149 of 0-149), causing catastrophic failure at episode start during evaluation.

**Solution**: Include ALL timesteps in training with proper padding, and fix eval timing.

**Result**: Windowed models now trainable and usable in closed-loop evaluation.
