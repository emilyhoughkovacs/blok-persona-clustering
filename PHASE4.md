# Phase 4: Running Agent Simulations

## Current Status

The simulation infrastructure is complete and tested with mock responses:
- `src/agents.py` — PersonaAgent and PersonaSimulator classes
- `notebooks/05_agent_simulation.ipynb` — Simulation notebook with 6 test scenarios
- 7 personas ready for instantiation

## Next Steps

### 1. Get Anthropic API Key

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Create an account or sign in
3. Navigate to API Keys and create a new key
4. Copy the key (starts with `sk-ant-...`)

### 2. Configure API Access

**Option A: Environment variable (recommended for terminal)**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**Option B: .env file (recommended for notebooks)**
```bash
echo 'ANTHROPIC_API_KEY=sk-ant-your-key-here' > .env
```

Then load it in the notebook by adding this cell before initializing the simulator:
```python
from dotenv import load_dotenv
load_dotenv()
```

Note: `.env` is already in `.gitignore` so your key won't be committed.

### 3. Install Dependencies

```bash
source venv/bin/activate
pip install anthropic python-dotenv
```

### 4. Run Simulations

Open `notebooks/05_agent_simulation.ipynb` and change:
```python
simulator = PersonaSimulator(
    personas_path='../data/processed/personas.json',
    mock_mode=False  # Changed from True
)
```

Then run all cells.

## Cost Estimate

| Model | 42 calls (7 personas × 6 scenarios) |
|-------|-------------------------------------|
| Sonnet (default) | ~$0.24 |
| Opus | ~$1.17 |

The code defaults to Sonnet, which is sufficient for persona role-playing.

To use Opus instead:
```python
simulator = PersonaSimulator(
    personas_path='../data/processed/personas.json',
    mock_mode=False,
    model='claude-opus-4-5-20250514'
)
```

## Expected Output

After running with real API calls:
- `simulation_results.csv` — All persona responses with parsed decisions
- `decision_heatmap.png` — Visual comparison of decisions across personas
- Validation report showing persona consistency

## Troubleshooting

**"ANTHROPIC_API_KEY not found"**
- Ensure the environment variable is set in the same terminal session
- Or use `.env` file with `python-dotenv`

**"anthropic package not installed"**
- Run `pip install anthropic` in your virtual environment

**Rate limits**
- The Anthropic API has rate limits; 42 sequential calls should be fine
- If you hit limits, add delays between calls in `run_batch()`
