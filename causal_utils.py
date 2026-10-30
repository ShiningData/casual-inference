"""
causal_utils.py
Shared synthetic data generators for the Modern Causal Inference tutorial.

Every dataset is SIMULATED with a KNOWN causal structure and KNOWN ground-truth
effects, so attendees can check how close each method gets to the truth.
No real customer data is used anywhere in this tutorial.
"""
import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------
# 1. Credit Line Increase (CLI) -> 6-month spend   (observational, confounded)
# --------------------------------------------------------------------------
def make_credit_line_data(n=10_000, seed=42):
    """
    Business question: What is the causal effect of granting a credit line
    increase (CLI) on a customer's incremental 6-month card spend ($)?

    Causal structure (ground truth):
        income, credit_score, tenure, utilization, age  -> CLI (treatment)
        income, credit_score, tenure, utilization, age  -> spend (outcome)
        promo_email (randomized nudge, instrument)      -> CLI only
        CLI -> new_balance (mediator / post-treatment)  -> spend
        region_code: irrelevant noise variable

    Heterogeneous effect:
        true_cate = 150 + 700*sigmoid((income_k - 70)/12) + 3*(credit_score - 700)
                    - 700*(utilization > 0.6)
        (non-linear: lift saturates with income, rises with score, and
         customers above 60% utilization respond much less)
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 75, n)
    income_k = np.clip(rng.lognormal(np.log(70), 0.4, n), 20, 400)      # $000s
    credit_score = np.clip(rng.normal(690 + 0.25 * (income_k - 70), 55, n), 450, 850)
    tenure = np.clip(rng.gamma(2.0, 3.0, n), 0, 40)                       # years
    utilization = np.clip(rng.beta(2, 3, n) + 0.002 * (700 - credit_score) / 10, 0, 1)
    region_code = rng.integers(0, 5, n)                                    # noise
    promo_email = rng.binomial(1, 0.5, n)                                  # randomized

    # Treatment assignment: bank's policy favours high score / high income
    logit = (-0.5 + 0.02 * (credit_score - 690) + 0.015 * (income_k - 70)
             + 0.05 * tenure - 1.5 * (utilization - 0.4) + 1.2 * promo_email)
    p_treat = _sigmoid(logit)
    cli = rng.binomial(1, p_treat)

    true_cate = (150 + 700 * _sigmoid((income_k - 70) / 12)
                 + 3 * (credit_score - 700) - 700 * (utilization > 0.6))

    # Mediator: balance after the increase (post-treatment!)
    new_balance = 2000 + 15 * income_k + 1500 * cli + rng.normal(0, 500, n)

    baseline = (500 + 25 * income_k + 4 * (credit_score - 650) + 60 * tenure
                - 1200 * utilization + 5 * age)
    # Part of the CLI effect flows THROUGH the mediator (0.2 * 1500 = $300),
    # the rest is direct, so the TOTAL effect of cli on spend equals true_cate.
    spend = (baseline + 0.2 * (new_balance - 2000)
             + (true_cate - 300) * cli + rng.normal(0, 800, n))

    df = pd.DataFrame(dict(
        age=age, income_k=income_k.round(1), credit_score=credit_score.round(0),
        tenure=tenure.round(1), utilization=utilization.round(3),
        region_code=region_code, promo_email=promo_email, cli=cli,
        new_balance=new_balance.round(0), spend=spend.round(2),
        true_cate=true_cate.round(2), propensity_true=p_treat.round(4)))
    return df


CREDIT_LINE_EDGES = [
    ("age", "cli"), ("age", "spend"),
    ("income_k", "cli"), ("income_k", "spend"), ("income_k", "credit_score"),
    ("income_k", "new_balance"),
    ("credit_score", "cli"), ("credit_score", "spend"), ("credit_score", "utilization"),
    ("tenure", "cli"), ("tenure", "spend"),
    ("utilization", "cli"), ("utilization", "spend"),
    ("promo_email", "cli"),
    ("cli", "new_balance"), ("cli", "spend"), ("new_balance", "spend"),
]


# Hand-placed layout for readable plots of the credit-line DAG
CREDIT_LINE_POS = {
    "promo_email": (-3.0, 1.0), "age": (-1.5, 2.0), "income_k": (0.0, 2.6),
    "credit_score": (1.5, 2.0), "utilization": (3.0, 1.4), "tenure": (-2.6, -1.0),
    "cli": (-1.3, 0.0), "new_balance": (0.4, -1.3), "spend": (2.2, 0.0),
}


def edges_to_dot(edges):
    return "digraph {" + " ".join(f"{a} -> {b};" for a, b in edges) + "}"


CREDIT_LINE_DAG = """
digraph {
    age -> cli; age -> spend;
    income_k -> cli; income_k -> spend; income_k -> credit_score; income_k -> new_balance;
    credit_score -> cli; credit_score -> spend; credit_score -> utilization;
    tenure -> cli; tenure -> spend;
    utilization -> cli; utilization -> spend;
    promo_email -> cli;
    cli -> new_balance; cli -> spend;
    new_balance -> spend;
}
"""
# NOTE: new_balance is a MEDIATOR (post-treatment). $300 of the average effect
# flows cli -> new_balance -> spend. Adjusting for it estimates only the DIRECT
# effect (~ATE - 300) -- the classic BAD-CONTROL trap shown in Module 1.


# --------------------------------------------------------------------------
# 2. Cross-sell campaign uplift   (randomized experiment / RCT)
# --------------------------------------------------------------------------
def make_campaign_data(n=20_000, seed=7):
    """
    Business question: Which customers should receive a premium-card
    cross-sell offer? Treatment is RANDOMIZED (50/50 A/B test).

    Four latent segments (ground truth):
        persuadables  : convert only if treated        (positive uplift)
        sure_things   : convert anyway                  (~0 uplift)
        lost_causes   : never convert                   (~0 uplift)
        sleeping_dogs : treatment annoys them           (negative uplift)
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.integers(21, 75, n),
        "tenure": np.clip(rng.gamma(2, 3, n), 0, 40).round(1),
        "num_products": rng.poisson(2, n) + 1,
        "digital_engagement": rng.beta(2, 2, n).round(3),     # app logins share
        "avg_balance_k": np.clip(rng.lognormal(np.log(8), 0.8, n), 0, 200).round(2),
        "recent_complaint": rng.binomial(1, 0.08, n),
        "has_mortgage": rng.binomial(1, 0.3, n),
    })
    treatment = rng.binomial(1, 0.5, n)

    base_logit = (-3.0 + 0.4 * X.num_products + 0.03 * X.avg_balance_k
                  + 0.3 * X.has_mortgage)
    uplift_logit = (1.6 * (X.digital_engagement - 0.4)
                    + 0.6 * (X.tenure < 5)
                    - 2.2 * X.recent_complaint          # sleeping dogs
                    - 0.015 * (X.age - 45))
    p0 = _sigmoid(base_logit)
    p1 = _sigmoid(base_logit + uplift_logit)
    p = np.where(treatment == 1, p1, p0)
    conversion = rng.binomial(1, p)

    df = X.copy()
    df["treatment"] = treatment
    df["conversion"] = conversion
    df["true_uplift"] = (p1 - p0).round(5)
    return df


# --------------------------------------------------------------------------
# 3. Structural causal model for causal discovery
# --------------------------------------------------------------------------
DISCOVERY_NODES = ["account_age", "merchant_risk", "txn_amount", "foreign_txn", "new_device",
                   "velocity", "alert_score", "chargeback"]

# Ground-truth edges (parent -> child)
DISCOVERY_EDGES = [
    ("account_age", "new_device"),
    ("merchant_risk", "txn_amount"),
    ("merchant_risk", "alert_score"),
    ("foreign_txn", "txn_amount"),
    ("foreign_txn", "velocity"),
    ("new_device", "velocity"),
    ("txn_amount", "alert_score"),
    ("velocity", "alert_score"),
    ("alert_score", "chargeback"),
    ("new_device", "chargeback"),
]


def make_discovery_data(n=5_000, seed=1, noise="gaussian"):
    """
    Linear structural equation model over transaction-monitoring features.
    noise='gaussian' -> suited to PC / GES (Markov equivalence class only)
    noise='uniform'  -> non-Gaussian, lets LiNGAM recover full edge directions
    """
    rng = np.random.default_rng(seed)

    def e(scale=1.0):
        if noise == "gaussian":
            return rng.normal(0, scale, n)
        return rng.uniform(-scale * np.sqrt(3), scale * np.sqrt(3), n)

    merchant_risk = e()
    foreign_txn = e()
    account_age = e()
    new_device = -0.7 * account_age + e(0.7)
    txn_amount = 0.8 * merchant_risk + 0.7 * foreign_txn + e(0.6)
    velocity = 0.6 * foreign_txn + 0.9 * new_device + e(0.6)
    alert_score = 0.5 * merchant_risk + 0.7 * txn_amount + 0.8 * velocity + e(0.6)
    chargeback = 0.9 * alert_score + 0.6 * new_device + e(0.6)

    df = pd.DataFrame(dict(account_age=account_age, merchant_risk=merchant_risk, txn_amount=txn_amount,
                           foreign_txn=foreign_txn, new_device=new_device,
                           velocity=velocity, alert_score=alert_score,
                           chargeback=chargeback))
    return df[DISCOVERY_NODES]


def true_adjacency(nodes=DISCOVERY_NODES, edges=DISCOVERY_EDGES):
    """A[i, j] = 1 means nodes[i] -> nodes[j]."""
    idx = {v: k for k, v in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=int)
    for a, b in edges:
        A[idx[a], idx[b]] = 1
    return A


def compare_graphs(A_true, A_est):
    """
    Compare an estimated adjacency (A_est[i,j]=1 for i->j; set both [i,j] and
    [j,i] = 1 for an undirected edge) against the truth.
    Returns skeleton precision/recall/F1, # correctly-oriented edges and SHD.
    """
    n = A_true.shape[0]
    skel_t = ((A_true + A_true.T) > 0).astype(int)
    skel_e = ((A_est + A_est.T) > 0).astype(int)
    iu = np.triu_indices(n, 1)
    t, e_ = skel_t[iu], skel_e[iu]
    tp = int(((t == 1) & (e_ == 1)).sum())
    fp = int(((t == 0) & (e_ == 1)).sum())
    fn = int(((t == 1) & (e_ == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    oriented_ok, undirected, reversed_ = 0, 0, 0
    for i, j in zip(*np.where(A_true == 1)):
        if A_est[i, j] == 1 and A_est[j, i] == 0:
            oriented_ok += 1
        elif A_est[i, j] == 1 and A_est[j, i] == 1:
            undirected += 1
        elif A_est[j, i] == 1:
            reversed_ += 1
    # SHD: missing + extra + mis-oriented (undirected counted as 1 error)
    shd = fn + fp + reversed_ + undirected
    return pd.Series(dict(skeleton_precision=round(prec, 3), skeleton_recall=round(rec, 3),
                          skeleton_f1=round(f1, 3), correctly_oriented=oriented_ok,
                          left_undirected=undirected, reversed=reversed_,
                          true_edges=int(A_true.sum()), SHD=shd))


def plot_dag(edges, nodes=None, ax=None, title=None, undirected=None, pos=None):
    """Lightweight DAG plot with networkx (no graphviz binary needed)."""
    import matplotlib.pyplot as plt
    import networkx as nx
    G = nx.DiGraph()
    if nodes is not None:
        G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    if pos is None:
        try:
            for layer, layer_nodes in enumerate(nx.topological_generations(G)):
                for node in layer_nodes:
                    G.nodes[node]["layer"] = layer
            pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
            pos = {k: (v[0], -v[1]) for k, v in pos.items()}
        except Exception:
            pos = nx.spring_layout(G, seed=3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#dbe9f6", node_size=2300,
                           edgecolors="#2f5d8a")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=18,
                           node_size=2300, edge_color="#2f5d8a")
    if undirected:
        U = nx.Graph(); U.add_edges_from(undirected)
        nx.draw_networkx_edges(U, pos, ax=ax, style="dashed", edge_color="#c0392b",
                               width=2)
    ax.set_title(title or "")
    ax.axis("off")
    return ax


# --------------------------------------------------------------------------
# 4. Capstone: proactive fee waiver -> 12-month attrition  (observational)
# --------------------------------------------------------------------------
FEE_WAIVER_EDGES = [
    ("recent_complaint", "fee_waiver"), ("recent_complaint", "churn_12m"),
    ("balance_k", "fee_waiver"), ("balance_k", "churn_12m"),
    ("num_products", "fee_waiver"), ("num_products", "churn_12m"),
    ("tenure", "churn_12m"), ("tenure", "fee_waiver"),
    ("digital_user", "churn_12m"),
    ("fee_waiver", "churn_12m"),
]


def make_fee_waiver_data(n=15_000, seed=2026):
    """
    Business question: Do proactive fee waivers granted by branch staff reduce
    12-month attrition, and for whom? Staff tend to grant waivers to customers
    who recently complained -- who are also far more likely to leave.
    `true_cate` is the change in churn probability caused by the waiver
    (negative = waiver reduces churn).
    """
    rng = np.random.default_rng(seed)
    recent_complaint = rng.binomial(1, 0.15, n)
    balance_k = np.clip(rng.lognormal(np.log(12), 0.9, n), 0, 500).round(2)
    num_products = rng.poisson(1.5, n) + 1
    tenure = np.clip(rng.gamma(2, 3.5, n), 0, 40).round(1)
    digital_user = rng.binomial(1, 0.6, n)

    p_waiver = _sigmoid(-2.4 + 3.0 * recent_complaint + 0.015 * balance_k
                        + 0.35 * (num_products > 2) + 0.02 * tenure)
    fee_waiver = rng.binomial(1, p_waiver)

    base = (-1.8 + 2.4 * recent_complaint - 0.004 * balance_k - 0.25 * num_products
            - 0.06 * tenure + 0.3 * digital_user)
    effect_logit = -0.15 - 1.4 * recent_complaint - 0.004 * np.minimum(balance_k, 100)
    p0, p1 = _sigmoid(base), _sigmoid(base + effect_logit)
    churn = rng.binomial(1, np.where(fee_waiver == 1, p1, p0))
    return pd.DataFrame(dict(recent_complaint=recent_complaint, balance_k=balance_k,
                             num_products=num_products, tenure=tenure,
                             digital_user=digital_user, fee_waiver=fee_waiver,
                             churn_12m=churn, true_cate=(p1 - p0).round(5)))
