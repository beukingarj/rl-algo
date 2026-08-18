# rl-algo

Experimentele reinforcement-learning pipeline voor het handelen in aandelen.
Een custom `gym`-omgeving simuleert een simpele long/cash-strategie (koop of
hou cash) op historische koersdata, en een RL-agent (A2C/DQN via
`stable-baselines3`) leert daarop een handelsbeleid.

## Projectstructuur

| Bestand | Beschrijving |
|---|---|
| `extract_data.py` | Haalt koersdata op via `yfinance`, berekent features/indicatoren en splitst in train/val/test. |
| `trading_env.py` | Custom `gym`-omgeving: action space (koop/verkoop), reward-functie, balans- en transactiekosten-boekhouding. |
| `train_and_evaluate.py` | Zet train/val/test-omgevingen op, traint een A2C/DQN-model en evalueert het. |
| `SaveOnBestTrainingRewardCallback.py` | Callback die tijdens training periodiek op de validatieset evalueert en het beste model + genormaliseerde env wegschrijft naar `./logs/best_model/`. |
| `evaluate_policy.py` | Aangepaste variant van `stable_baselines3.common.evaluation.evaluate_policy` (retourneert ook `info`). |
| `main.py` | Notebook-achtig script (`#%%`-cellen) dat data-extractie, training en evaluatie aan elkaar knoopt. |

## Installatie

```bash
python -m venv ./venv
```

Windows:
```bat
venv\Scripts\activate.bat
```

macOS/Linux:
```bash
source venv/bin/activate
```

Dependencies installeren:
```bash
pip install -r requirements.txt
```

## Gebruik

`main.py` is opgezet als Jupyter-notebook-in-`.py`-vorm (`#%%`-cellen,
bruikbaar met de Jupyter-extensie van VS Code of via `jupytext`). Het
doorloopt:

1. Data ophalen en features maken (`extract_data`)
2. Model trainen over een reeks learning rates (`train_and_evaluate`)
3. Resultaten evalueren

Getrainde modellen en genormaliseerde environments worden weggeschreven naar
`./logs/best_model/<model_type>/`.

## CI / GitHub Actions

Deze repo heeft een minimale CI-pipeline in `.github/workflows/ci.yml`, die
draait op GitHub-hosted runners (`ubuntu-latest`, geen eigen infrastructuur
nodig). Bij elke push naar `main`, elke pull request en handmatig
(`workflow_dispatch`) installeert hij `requirements.txt` en compileert alle
`.py`-bestanden (`python -m py_compile`) als snelle syntax-/importcheck.

Er zijn nog geen geautomatiseerde tests (zie `.claude/agents/README.md` voor
de AI-squad die dit kan uitbreiden); voeg `pytest` en een `tests/`-map toe
en breid de workflow uit zodra er testbare logica is.

## AI Squad

Dit project heeft een set scoped Claude Code subagents in `.claude/agents/`
voor data-, RL/env- en documentatiewerk. Zie `.claude/agents/README.md` voor
de rolverdeling en tokenregels.
