# Modern Causal Inference: Methods and Applications
### An Essential Hands-on Guide with DoWhy, EconML, CausalML & causal-learn in Python
**3-hour workshop · Jupyter notebooks · synthetic banking data with known ground truth**

---

## What's in this folder

```
causal_inference_workshop/
├── README.md                     ← you are here (setup + facilitator guide)
├── requirements.txt              ← pinned, tested-together versions
├── causal_utils.py               ← synthetic data generators + plotting helpers
├── notebooks/                    ← STUDENT versions (clean; solutions hidden in "Click to reveal")
│   ├── 00_Welcome_Agenda_and_Setup.ipynb
│   ├── 01_Foundations_and_DoWhy.ipynb
│   ├── 02_Heterogeneous_Effects_EconML.ipynb
│   ├── 03_Uplift_Modeling_CausalML.ipynb
│   ├── 04_Causal_Discovery_causal-learn.ipynb
│   └── 05_Capstone_and_Wrap-up.ipynb
└── instructor_solutions/         ← same notebooks, solutions as live cells, fully executed with outputs
```

**All data is simulated** (no customer data), so the notebooks can be shared freely and every estimate can be checked against the true effect.

| Scenario | Data | Used in | Causal question |
|---|---|---|---|
| A. Credit line increase | Observational, confounded, 10K customers | 01, 02 | Effect of a CLI on 6-month spend; for whom is it worth the cost? |
| B. Premium-card cross-sell | Randomized A/B test, 20K customers | 03 | Whom should we contact (uplift targeting)? |
| C. Transaction monitoring | Linear SEM, 8 variables | 04 | What causes what? Effect of velocity on chargebacks |
| D. Fee waiver & attrition | Observational, confounding by indication | 05 | Do waivers reduce churn, and for whom? (capstone) |

---

## Participant setup (send this a week before the session)

```bash
python -m venv causal-env
source causal-env/bin/activate          # Windows: causal-env\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name causal-env
jupyter lab
```
Then open `notebooks/00_Welcome_Agenda_and_Setup.ipynb` and run the environment check; every library should show ✅.

**Troubleshooting**
- `causalml` sometimes compiles from source (needs a C/C++ compiler). If it fails, try `conda install -c conda-forge causalml`, or use a Python version with a prebuilt wheel.
- Behind a corporate proxy / internal package mirror: `pip install --index-url <your-mirror> -r requirements.txt`.
- Graphviz is **not** required — all graphs are drawn with networkx/matplotlib.
- Runtime: each notebook runs end-to-end in under 1 minute on a laptop (Module 1 ≈ 45 s because of refuters).

---

## Facilitator guide

### Agenda (180 min)
| Time | Module | Teach | Hands-on | If you're running late… |
|---|---|---|---|---|
| 0:00–0:10 | 00 Welcome | Prediction vs intervention, ladder of causation, naive-comparison teaser | Env check | Skip the ladder table |
| 0:10–0:50 | 01 DoWhy | Potential outcomes, DAG building blocks, collider demo, Model→Identify→Estimate→Refute, bad controls | Ex 1.1 (wrong graph), Ex 1.2 (ATT) | Make Ex 1.2 take-home |
| 0:50–1:00 | ☕ Break | | | |
| 1:00–1:45 | 02 EconML | DML intuition (hand-rolled), estimator zoo, R-score, CIs, interpreters, cost-aware policy | Ex 2.1 (cost sensitivity), Ex 2.2 (EconML inside DoWhy) | Demo Ex 2.2 instead of hands-on |
| 1:45–2:15 | 03 CausalML | Four customer types, meta-learners, uplift forest, gain/Qini/AUUC, profit vs depth | Ex 3.1 (sleeping dogs) | Ex 3.2 is already optional |
| 2:15–2:20 | ☕ Break | | | |
| 2:20–2:45 | 04 causal-learn | Equivalence classes, PC/GES, tiers, LiNGAM, FCI, discovered graph → DoWhy | Ex 4.1 (fragility) | Skip FCI details, keep the `<->` punchline |
| 2:45–3:00 | 05 Wrap-up | Decision guide, workflow ↔ governance, pitfalls, resources | Capstone is take-home | Q&A |

### Key "aha" moments to land (numbers from the instructor notebooks)
1. **Module 00/01:** naive difference **$1,133** vs true ATE **$367** — confounding inflates the effect ~3×. Most adjusted estimators land at **$349–$418** (coarse stratification: $248); the IV gives **$365**.
2. **Module 01:** adjusting for the mediator `new_balance` drops the estimate from **$301 → −$3**. *"More controls" can be wrong.*
3. **Module 01, Ex 1.1:** a wrong graph produces a badly biased estimate that still **passes the placebo refuter**. Refuters test the estimator, not the graph.
4. **Module 02:** forest-based CATE models beat LinearDML on non-linear heterogeneity; the **R-score** picks good models without ground truth; a depth-2 tree recovers the true segments (utilization > 0.6 → negative effect).
5. **Module 02:** a CATE-based policy nearly doubles net value vs "treat everybody" (**$661K vs $347K** on the test set; oracle $687K).
6. **Module 03:** profit peaks targeting ~**40–45%** of customers; contacting **everyone loses money**. The response model ranks "sure things", not persuadables.
7. **Module 04:** PC leaves `account_age — new_device` undirected (identifiability limit); domain knowledge or LiNGAM orients it. FCI flags the hidden confounder as `txn_amount <-> velocity`. On the discovered graph DoWhy estimates **0.718** vs truth **0.720** (naive 1.238).
8. **Module 05:** naive analysis says fee waivers *increase* churn (+0.6 pp); the true effect is **−5.5 pp**, concentrated in recent complainers.

### Discussion prompts for a banking audience
- Where in our portfolio do we have **natural experiments** (randomized holdouts, score cut-offs, phased rollouts)?
- Which existing "propensity" models are actually used for **intervention decisions**, and should they be uplift models instead?
- How would we document a DAG and a sensitivity analysis for **model risk review**?
- Fair-lending angle: heterogeneous effects and policy trees must still be checked for disparate impact before deployment.

### Tested environment
Python 3.13 · DoWhy 0.14 · EconML 0.17.0 · CausalML 0.18.0 · causal-learn 0.1.4.8 · scikit-learn 1.9.1 · LightGBM 4.7.0 · pandas 3.1 · NumPy 2.5.

**Known library quirks handled in the notebooks**
- DoWhy updates an identified-estimand object in place when an estimator runs. The notebooks pass a fresh copy to each estimator, and give refuters each estimate's own estimand (`est.target_estimand`).
- When calling EconML through DoWhy, pass `effect_modifiers=` to `estimate_effect` as well.
- CausalML's plotting changes matplotlib's global style; Module 3 resets it with `plt.style.use("default")`.
