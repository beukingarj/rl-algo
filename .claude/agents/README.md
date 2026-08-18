# AI Squad — rl-algo

Kleine, scherp afgebakende set subagents voor dit project. Bewust klein
gehouden: bij 7 bestanden en ~35KB code kost een groot multi-agent-team
meer tokens (orchestratie + cold-start context per spawn) dan het
oplevert.

| Rol | Bestand | Scope | Model |
|---|---|---|---|
| Orchestrator | — (hoofdsessie) | routering, kleine fixes (<5 file-reads) zelf afhandelen | huidig sessiemodel |
| Data Engineer | `data-engineer.md` | `extract_data.py` | haiku |
| RL/Env Engineer | `rl-engineer.md` | `trading_env.py`, `train_and_evaluate.py`, `SaveOnBestTrainingRewardCallback.py` | sonnet |
| Docs/Explainer | `docs-writer.md` | README, docstrings, commit messages | haiku |
| Reviewer/QA | — (bestaande `code-review` skill) | diff van elke wijziging, effort low/medium | sonnet |

Geen aparte Architect/Planner-rol: te klein project om te rechtvaardigen,
ad-hoc plan-mode volstaat. Geen aparte experiment-tracker-rol: pas zinnig
bij veel hyperparameter-sweeps/logging.

## Tokenregels
- **Batchen, niet spawnen per bug.** Meerdere kleine fixes in één
  sessie/commit i.p.v. een subagent per bug.
- **Explore-agent** voor puur zoekwerk ("waar staat X") i.p.v.
  general-purpose — leest excerpts, geen volledige files.
- **Achtergrondtaken** (bv. een lange trainingsrun volgen) via een
  background agent, zodat de hoofdcontext niet volloopt met poll-ruis.
- **Modeltier volgt foutkosten, niet taakgrootte**: mechanisch werk
  (data-extractie, docs) → haiku; alles wat trading-logica/reward raakt →
  sonnet.
- **rl-engineer laadt zijn drie kernbestanden samen** in één run in
  plaats van los, om dubbel context laden te voorkomen.
