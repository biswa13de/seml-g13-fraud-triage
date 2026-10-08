const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, PageBreak,
  LevelFormat,
} = require("docx");

const REPO = "/Users/biswa/Downloads/BITS Pilani/sem-3/SE for ML/seml-g13-fraud-triage";
const IMG = (p) => path.join(REPO, p);

const PAGE_WIDTH_DXA = 11906;
const MARGIN_DXA = 1134;
const CONTENT_WIDTH_DXA = PAGE_WIDTH_DXA - 2 * MARGIN_DXA;

const COLORS = { accent: "9A3324", soft: "666666", good: "2E7D32", bad: "B00020", head: "2F5496" };

function imgDims(relPath, maxWidthDxa, maxHeightIn) {
  const out = execSync(
    `python3 -c "from PIL import Image; im=Image.open('${IMG(relPath)}'); print(im.size[0], im.size[1])"`
  ).toString().trim();
  const [wPx, hPx] = out.split(" ").map(Number);
  const maxWidthIn = maxWidthDxa / 1440;
  let widthIn = maxWidthIn;
  let heightIn = (widthIn * hPx) / wPx;
  if (heightIn > maxHeightIn) {
    heightIn = maxHeightIn;
    widthIn = (heightIn * wPx) / hPx;
  }
  return { width: Math.round(widthIn * 96), height: Math.round(heightIn * 96) };
}
function Img(relPath, maxWidthDxa, maxHeightIn) {
  return new ImageRun({ type: "png", data: fs.readFileSync(IMG(relPath)), transformation: imgDims(relPath, maxWidthDxa, maxHeightIn) });
}
function CenteredImage(relPath, maxWidthDxa, maxHeightIn, caption) {
  const out = [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: [Img(relPath, maxWidthDxa, maxHeightIn)] })];
  if (caption) out.push(new Paragraph({ spacing: { before: 60, after: 240 }, children: [new TextRun({ text: caption, italics: true, size: 20, color: COLORS.soft })] }));
  return out;
}

function H1(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } }); }
function H2(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 } }); }
function H3(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100 } }); }
function P(text, opts = {}) { return new Paragraph({ spacing: { after: 160, line: 276 }, children: [new TextRun({ text, ...opts })] }); }
function Para(runs, opts = {}) { return new Paragraph({ spacing: { after: 160, line: 276 }, ...opts, children: runs }); }
function Caption(text) { return new Paragraph({ spacing: { before: 60, after: 240 }, children: [new TextRun({ text, italics: true, size: 20, color: COLORS.soft })] }); }
function Bullet(text, level = 0) {
  return new Paragraph({ numbering: { reference: "bullets", level }, spacing: { after: 80 }, children: [new TextRun({ text })] });
}
function PageBr() { return new Paragraph({ children: [new PageBreak()] }); }

function cell(text, opts = {}) {
  const { bold = false, shade = null, width = null, align = AlignmentType.LEFT, color = null, italics = false } = opts;
  return new TableCell({
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text: String(text), bold, italics, color: color || undefined, size: 20 })] })],
  });
}
function simpleTable(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scaled = widths.map((w) => Math.round((w / total) * CONTENT_WIDTH_DXA));
  return new Table({
    width: { size: CONTENT_WIDTH_DXA, type: WidthType.DXA },
    columnWidths: scaled,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, { bold: true, shade: COLORS.head, color: "FFFFFF", width: scaled[i] })) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((v, i) => cell(v, { shade: ri % 2 ? "F2F2F2" : null, width: scaled[i] })) })),
    ],
  });
}

// ===================================================================== BODY
const body = [];

// ---------------------------------------------------------- 1. Problem statement
body.push(H1("1. Domain and Problem Statement"));
body.push(P(
  "India's UPI rails process billions of payments a month, and fraud has grown with them: account " +
  "takeover after SIM-swap or phishing, “collect-request” scams, mule accounts that receive and " +
  "quickly move stolen funds, and rapid low-value probing. A typical Payment Service Provider (PSP) today " +
  "relies on static rules (amount limits, blocklists), which miss new fraud patterns and decline too many " +
  "genuine payments. Every flagged payment goes to an undifferentiated manual-review queue, so analysts " +
  "spend their time on low-risk alerts while high-value fraud settles before anyone looks at it."
));
body.push(P("We propose a Real-Time Fraud Risk Triage System. For each payment, before authorisation, it:"));
body.push(Bullet("estimates fraud probability with an ML model using transaction and behavioural features,"));
body.push(Bullet("maps that risk to a cost-optimal action — ALLOW, STEP_UP (extra authentication such as OTP or biometric), or BLOCK — inside a strict latency budget,"));
body.push(Bullet("returns human-readable reason codes for every decision, and"));
body.push(Bullet("opens prioritised cases for fraud analysts, whose verdicts flow back as labels for retraining."));

body.push(H2("Why ML instead of rules"));
body.push(P(
  "Fraud patterns change adversarially, the signal comes from many weak interacting features, labelled " +
  "history exists (chargebacks and complaints), and an occasional wrong answer is tolerable because " +
  "STEP_UP and human review act as safety nets — exactly the conditions under which the course material " +
  "recommends using ML as predictions rather than hand-written rules."
));

body.push(H2("Scope"));
body.push(P("In scope: scoring, triage decision, reason codes, case queue, monitoring, and a cost-based policy. Out of scope: the actual NPCI switch integration, customer KYC, and chargeback processing (simulated)."));

// ---------------------------------------------------------- 2. Requirements
body.push(H1("2. Requirement Specifications and Measurable Goals"));
body.push(P("Goals are organised in four layers, each attached to a SMART measure and a way to measure it."));

body.push(H2("2.1 Layered Goals"));
body.push(simpleTable(
  ["Layer", "Goal", "Measure (SMART)", "How Measured"],
  [
    ["Organisational", "Reduce fraud losses", "≤ 2 bps of payment value within 6 months (baseline 5 bps)", "Monthly finance report"],
    ["Organisational", "Protect customer experience", "False-decline rate ≤ 0.5% of genuine payments", "Declines later confirmed genuine"],
    ["System", "Intervene before money moves", "≥ 85% of fraud value gets STEP_UP or BLOCK", "Offline replay on test month"],
    ["System", "Decide in real time", "p95 ≤ 150 ms, p99 ≤ 300 ms", "Load test + /metrics"],
    ["User (analyst)", "Work the riskiest cases first", "≥ 80% of confirmed fraud in top 20% of queue", "Case-service analytics"],
    ["User (customer)", "Little friction", "STEP_UP rate ≤ 3% of all payments", "Decision-mix metric"],
    ["Model", "Rank fraud well on imbalanced data", "PR-AUC ≥ 0.80; Recall ≥ 85% at FPR ≤ 1%", "Time-based hold-out test"],
    ["Model", "Stay current", "Retrain when PSI > 0.2 or weekly", "Monitoring service"],
  ],
  [14, 22, 34, 30]
));

body.push(H2("2.2 Functional Requirements"));
body.push(simpleTable(["ID", "Requirement"], [
  ["FR1", "Score each payment request and return {decision, risk_score, reason_codes[≤3], model_version, latency_ms}."],
  ["FR2", "Apply deterministic hard rules first: blocklisted payee VPA → BLOCK; amount above the hard limit → BLOCK."],
  ["FR3", "Map the score to ALLOW / STEP_UP / BLOCK using cost-based thresholds."],
  ["FR4", "Publish every decision as a payment.decided event."],
  ["FR5", "Create an analyst case for every BLOCK and for STEP_UP above ₹10,000. Priority = score × amount."],
  ["FR6", "Let analysts list cases and resolve them (CONFIRMED_FRAUD / GENUINE). Each resolution becomes a label."],
  ["FR7", "Update the payer/payee's behavioural (velocity) features after every payment."],
  ["FR8", "Monitor decision mix, score drift (PSI) and latency, and raise an alert when thresholds are breached."],
  ["FR9", "If the ML scorer is unhealthy, fall back to rules only and return degraded=true. Never hang."],
  ["FR10", "Version every model in a registry. Serving loads the champion alias."],
], [10, 90]));

body.push(H2("2.3 Assumptions"));
body.push(Bullet("A1: Fraud labels arrive late (up to 30 days), so training uses only mature labels."));
body.push(Bullet("A2: Fraudsters adapt, so the data distribution drifts and retraining is required."));
body.push(Bullet("A3: Upstream sends a well-formed payment message with payer/payee account IDs, type, amount and pre-transaction balances."));
body.push(Bullet("A4: Fraud base rate is low (0.1–1%; ~0.13% overall, ~0.3% within TRANSFER/CASH_OUT in our dataset)."));
body.push(Bullet("A5: PaySim (simulated mobile money) is a reasonable public proxy for UPI P2P payments; it lacks device/geo signals, which a real deployment would add."));

// ---------------------------------------------------------- 3. GR4ML
body.push(PageBr());
body.push(H1("3. GR4ML Views"));
body.push(P("We model the system in GR4ML across its three views, following the notation and legends taught in Session 3."));

body.push(H2("3.1 Business View (Why?)"));
body.push(...CenteredImage("docs/diagrams/gr4ml_business_view.png", CONTENT_WIDTH_DXA, 6.6,
  "Figure 1. Business View — four actors desire a strategic goal that splits (AND) into three decision goals; each is answered by a question goal, each answered by an insight."));
body.push(simpleTable(["Element", "Instances"], [
  ["Actor", "Fraud Risk Manager, Customer (Payer), Fraud Analyst, Compliance Officer"],
  ["Strategic Goal", "G0: Grow digital payments safely"],
  ["Decision Goal", "D1: triage action per payment · D2: case review priority · D3: block payee VPA"],
  ["Question Goal", "Q1: fraud probability? · Q2: which risk factors? · Q3: expected loss of case?"],
  ["Insight", "Payment Fraud Risk Predictive Model · Risk Reason Codes · Case Priority Score"],
  ["Indicator", "Fraud loss (bps of GMV) · False-decline rate (%)"],
], [25, 75]));

body.push(H2("3.2 Analytics Design View (What?)"));
body.push(...CenteredImage("docs/diagrams/gr4ml_analytics_design_view.png", CONTENT_WIDTH_DXA, 6.6,
  "Figure 2. Analytics Design View — four candidate algorithms and their influence on the softgoals from Section 4. Three were trained and compared in training/train.py (Section 7.3); Isolation Forest was considered and rejected without training."));
body.push(simpleTable(["Element", "Instances"], [
  ["Analytics Goal", "AG1 Classification (Prediction) · AG2 Triage policy (Prescription) · AG3 Explanation (Description)"],
  ["Algorithm", "Logistic Regression, Random Forest, Isolation Forest, LightGBM (champion)"],
  ["Softgoal", "Detection accuracy, Low inference latency, Interpretability, Tolerance to missing values, Adaptability to drift"],
  ["Satisfied / Denied", "LightGBM satisfied (tied best on validation accuracy; fastest to score and explain one payment; smallest; quickest to retrain) · Random Forest denied (equally accurate, but 6–12x slower per payment and 6x larger) · Logistic Regression denied (validation PR-AUC 0.924) · Isolation Forest denied (considered, not trained: unsupervised, ignores the labels we have)"],
], [25, 75]));

body.push(H2("3.3 Data Preparation View (How?)"));
body.push(...CenteredImage("docs/diagrams/gr4ml_data_preparation_view.png", CONTENT_WIDTH_DXA, 5.4,
  "Figure 3. Data Preparation View — the operator chain from training/build_features.py, with the leaked post-transaction columns marked as dropped."));
body.push(simpleTable(["Element", "Instances"], [
  ["Entity (source)", "PaySimTransaction (step, type, amount, nameOrig, oldbalanceOrg, nameDest, oldbalanceDest, isFraud)"],
  ["Entity (output)", "TrainingFeatureTable (txn_id, 13 features, isFraud)"],
  ["Operator", "Filter · Leakage removal · Clean · Windowed aggregation · Derive · Encode · Split · Imbalance handling"],
  ["Note", "Point-in-time correctness; one shared feature function between training and serving (no train/serve skew)"],
], [25, 75]));

// ---------------------------------------------------------- 4. Quality attributes
body.push(H1("4. Top-3 Quality Requirements"));
body.push(P("Each is written as a QA scenario and verified by an automated test, so a regression is caught rather than just documented."));

body.push(H3("1. Performance (latency and throughput)"));
body.push(P("Under 100 payments/s normal load, the system returns a triage decision with p95 ≤ 150 ms and p99 ≤ 300 ms. Model inference p95 ≤ 50 ms."));
body.push(Para([new TextRun({ text: "Why: ", bold: true, italics: true, color: COLORS.soft }), new TextRun({ text: "the decision sits inside the payment authorisation path. A slow risk check makes payments time out, hurting both the customer and the PSP's success-rate metrics — a business loss even with perfect accuracy.", italics: true, color: COLORS.soft })]));
body.push(P("Design response: model loaded in memory once, feature store in Redis (O(1) lookups), async side-effects through events so cases and monitoring add no latency, a compact tree model."));
body.push(Para([new TextRun({ text: "Verified by: ", bold: true }), new TextRun({ text: "tests/test_quality_performance.py and scripts/load_test.py." })]));

body.push(H3("2. Model accuracy (cost-sensitive detection)"));
body.push(P("On a time-based hold-out month, Recall ≥ 85% at FPR ≤ 1% and PR-AUC ≥ 0.80. Precision of BLOCK decisions ≥ 90%."));
body.push(Para([new TextRun({ text: "Why: ", bold: true, italics: true, color: COLORS.soft }), new TextRun({ text: "a missed fraud is a direct monetary loss; a false BLOCK loses a customer. Plain accuracy is meaningless at a ~1% base rate, so we use PR-AUC and recall at a fixed FPR.", italics: true, color: COLORS.soft })]));
body.push(P("Design response: LightGBM with class weights, expected-cost thresholds, a 3-way decision (STEP_UP absorbs uncertainty), an analyst feedback loop."));
body.push(Para([new TextRun({ text: "Verified by: ", bold: true }), new TextRun({ text: "tests/test_quality_model.py, a quality gate that blocks registration as champion." })]));

body.push(H3("3. Explainability"));
body.push(P("Every STEP_UP or BLOCK decision returns the top-3 reason codes in plain language (e.g. “Transfer drains 100% of the sender's balance”), adding ≤ 10 ms p95."));
body.push(Para([new TextRun({ text: "Why: ", bold: true, italics: true, color: COLORS.soft }), new TextRun({ text: "analysts must justify actions on customer complaints and disputes; regulators expect decisions on customer funds to be explainable; reason codes also speed up case resolution.", italics: true, color: COLORS.soft })]));
body.push(P("Design response: SHAP TreeExplainer and a feature-to-text mapping."));
body.push(Para([new TextRun({ text: "Verified by: ", bold: true }), new TextRun({ text: "tests/test_quality_explainability.py." })]));

body.push(H2("Also considered (ranked lower)"));
body.push(P("Availability/reliability (handled architecturally by the heartbeat and fallback tactic, FR9), robustness, fairness (no demographic features are used), privacy (hashing VPAs in logs), and scalability. Trade-offs: accuracy vs. latency (a bigger model is slower); explainability vs. latency (SHAP adds cost); accuracy vs. friction (the threshold trade-off)."));

// ---------------------------------------------------------- 5. Architecture
body.push(H1("5. System Architecture"));
body.push(P("ML components are shown in blue, non-ML in amber, and the asynchronous event bus in purple. Solid arrows are synchronous calls; dashed arrows are published events."));
body.push(...CenteredImage("docs/diagrams/system_architecture.png", CONTENT_WIDTH_DXA, 6.6,
  "Figure 4. System architecture — matches docker-compose.yml component-for-component."));
body.push(simpleTable(["Component", "ML / non-ML", "Responsibility"], [
  ["triage-api", "non-ML", "Auth, hard rules, heartbeat-gated scorer call, cost policy, publishes decisions"],
  ["scoring", "ML", "Pipe-and-filter inference: validate → features → predict → SHAP explain"],
  ["Online Feature Store (Redis)", "ML", "24h receiver velocity, read before score, written after decision"],
  ["MLflow Registry", "ML", "Experiment tracking, model versioning, champion alias"],
  ["case-service", "non-ML", "Opens/lists/resolves analyst cases; verdicts become labels"],
  ["feature-updater", "non-ML (consumer)", "Writes velocity state asynchronously so it never adds request latency"],
  ["monitor", "non-ML", "PSI drift, decision mix, p95 latency, alerts"],
  ["Analyst Console", "non-ML", "Streamlit UI: simulate payments, work the case queue, view metrics"],
], [26, 20, 54]));

// ---------------------------------------------------------- 6. Patterns
body.push(H1("6. Architectural Patterns"));
body.push(P("Each pattern is written as a {Context, Problem, Solution} triple, with the reason it fits this system and its consequences."));

body.push(H2("6.1 Microservices"));
body.push(Para([new TextRun({ text: "Context: ", bold: true }), new TextRun({ text: "a fraud platform with parts that change and scale at different rates — the model is retrained weekly, rules change daily, case management changes rarely." })]));
body.push(Para([new TextRun({ text: "Problem: ", bold: true }), new TextRun({ text: "in a monolith, a model update means redeploying everything, scoring load can't be scaled on its own, and a bug in case management can take down authorisation (single point of failure)." })]));
body.push(Para([new TextRun({ text: "Solution: ", bold: true }), new TextRun({ text: "split by single responsibility into triage-api, scoring-service, case-service, feature-updater and monitor-service, each with its own API or data and its own container." })]));
body.push(Para([new TextRun({ text: "Fits because: ", bold: true, color: COLORS.good }), new TextRun({ text: "scoring is the hot path, so it scales on its own (docker compose up --scale scoring=3). The model can be swapped without touching the gateway. Different owners: the data-science team owns scoring, the fraud-ops team owns cases." })]));
body.push(Para([new TextRun({ text: "Trade-offs: ", bold: true, color: COLORS.bad }), new TextRun({ text: "network hops add latency (measured, within budget), more operational overhead, distributed failures — which is why we add the heartbeat tactic and a fallback." })]));

body.push(H2("6.2 Event-Driven Architecture (publish–subscribe)"));
body.push(Para([new TextRun({ text: "Context: ", bold: true }), new TextRun({ text: "each decision triggers several follow-up actions: open a case, update velocity features, update monitoring." })]));
body.push(Para([new TextRun({ text: "Problem: ", bold: true }), new TextRun({ text: "doing these synchronously adds latency to the authorisation path (violating QA1) and couples the gateway to every downstream service." })]));
body.push(Para([new TextRun({ text: "Solution: ", bold: true }), new TextRun({ text: "triage-api publishes one payment.decided event to Redis Streams. Independent consumer groups subscribe, with at-least-once delivery, acknowledgement (XACK) and replay of pending messages after a crash (XAUTOCLAIM)." })]));
body.push(Para([new TextRun({ text: "Fits because: ", bold: true, color: COLORS.good }), new TextRun({ text: "it keeps the hot path fast (QA1), lets us add consumers without changing the producer, and tolerates a consumer being down." })]));
body.push(Para([new TextRun({ text: "Trade-offs: ", bold: true, color: COLORS.bad }), new TextRun({ text: "eventual consistency — velocity features lag by milliseconds, which is acceptable here. Consumers must be idempotent: we dedupe on txn_id." })]));

body.push(H2("Supporting patterns and tactics"));
body.push(P("Pipe-and-filter inside the scoring service · Feature Store pattern · Model Registry pattern (MLflow aliases) · Heartbeat tactic · Real-time serving pattern · Fallback/degradation (rules-only mode)."));

// ---------------------------------------------------------- 7. Implementation
body.push(PageBr());
body.push(H1("7. Implementation"));
body.push(P("Code: https://github.com/biswa13de/seml-g13-fraud-triage. Stack: Python 3.11, FastAPI, LightGBM, SHAP, MLflow (its own tracking server, see Section 7.4), Redis 7 (Streams + online feature store), SQLite (cases), Streamlit, pytest, Docker Compose, GitHub Actions."));

body.push(H2("7.1 Dataset"));
body.push(P("Kaggle PaySim (ealaxi/paysim1): 6.36M simulated mobile-money transactions, 8,213 frauds, licence CC-BY-SA-4.0. Mapping to UPI: nameOrig = payer, nameDest = payee (M… = merchant), TRANSFER = P2P, CASH_OUT = cash-out, step = hour. Post-transaction balance columns (newbalanceOrig/Dest) are dropped — they leak the outcome and aren't known at authorisation time."));

body.push(H2("7.2 Feature Pipeline"));
body.push(P("One function, common/features.py:compute_features(), is called by both the offline training pipeline and the online scoring service, so the two paths can never drift apart. tests/test_features_parity.py proves this by replaying raw transactions through the same online feature store used at training time and asserting byte-identical output against the offline pipeline."));

body.push(H2("7.3 Model Training and Selection"));
body.push(P(
  "Three algorithms were trained and logged to MLflow (training/train.py). The selection rule was fixed " +
  "before looking at the results, and it uses only the validation split. The test split plays no part in " +
  "the choice; it is kept for the final quality gate (training/register.py)."
));
body.push(Bullet("Step 1: any model whose validation PR-AUC is within 0.001 of the best counts as tied on accuracy."));
body.push(Bullet("Step 2: among the tied models, pick the one with the lowest p95 time to score one payment, because the service scores one payment per request inside the QA1 latency budget."));
body.push(P("For each model we also measured SHAP explanation time for one payment, model size and training time (Random Forest is timed single-threaded, which is how it would be served):"));
const cmp = JSON.parse(fs.readFileSync(path.join(REPO, "data/model_comparison.json"), "utf8"));
const champ = JSON.parse(fs.readFileSync(path.join(REPO, "data/champion.json"), "utf8"));
const NAMES = { lightgbm: "LightGBM", random_forest: "Random Forest", logistic_regression: "Logistic Regression" };
const f = (v, d) => (v === null || Number.isNaN(v) ? "n/a" : Number(v).toFixed(d));
body.push(simpleTable(
  ["Algorithm", "Valid PR-AUC", "Valid Recall @ 1% FPR", "Score 1 payment p95 (ms)", "SHAP 1 payment p95 (ms)", "Size (MB)", "Train (s)"],
  cmp.map((r) => [
    NAMES[r.algorithm] + (r.algorithm === champ.algorithm ? " (champion)" : ""),
    f(r.valid_pr_auc, 5), f(r.valid_recall_at_1pct_fpr, 4), f(r.single_row_p95_ms, 2),
    r.algorithm === "logistic_regression" ? "n/a" : f(r.shap_single_row_p95_ms, 2),
    f(r.model_size_mb, 1), f(r.train_seconds, 1),
  ]),
  [24, 12, 14, 14, 14, 10, 10]
));
body.push(Para([new TextRun({ text: "Why LightGBM wins. ", bold: true }), new TextRun({ text:
  "Random Forest and LightGBM are tied on validation accuracy (both PR-AUC about 1.000 and recall 1.0 at 1% FPR), " +
  "so accuracy cannot separate them; Logistic Regression is clearly behind (PR-AUC 0.924) and is out at step 1. " +
  "Between the two tied models, LightGBM scores one payment 6–12x faster across our runs (about 0.3 ms vs " +
  "1.7–3.5 ms p95), explains it with SHAP about 7x faster, is about 6x smaller and retrains 5–7x faster. " +
  "Those are the costs our quality requirements depend on: the latency budget (QA1), cheap per-decision " +
  "explanations (QA3), and cheap retraining when the data drifts (the “adaptability to drift” softgoal)." })]));
body.push(P(
  "Two points for transparency. First, on the test split Random Forest is marginally ahead (PR-AUC 0.99998 vs " +
  "0.99974); we deliberately did not use that to choose, because choosing on test data leaks it into the " +
  "decision and leaves no unbiased final check. LightGBM then passed the quality gate on that test split " +
  "(PR-AUC 0.9997, Recall at 1% FPR 0.9994). Second, latency figures vary between runs and machines, which " +
  "is why we report a range; the ordering (LightGBM fastest) held in every run.",
  { italics: true, color: COLORS.soft }
));
body.push(P(
  "One honest finding worth stating plainly: fraud is nearly separable in PaySim by two legitimate " +
  "pre-transaction signals — fraudulent transfers drain 97.9% of the sender's balance (vs. 42.5% for " +
  "genuine ones), and 64.9% of the time the receiving account had a zero balance before the payment " +
  "(vs. 13.8%). This is a documented property of the simulator, not a leak in our pipeline (we verified no " +
  "post-transaction columns are used), and it is why all three models score very highly. Our cost-threshold " +
  "and latency results, not raw accuracy, are where real differentiation shows up."
));

body.push(H2("7.4 Model Registry: a Real Debugging Story"));
body.push(P(
  "MLflow's local-filesystem artifact store resolves every artifact path as absolute, relative to the " +
  "caller's current working directory at the time of logging. That broke the moment a Docker container " +
  "(a different filesystem, a different CWD) tried to load a model trained on the host: “No such artifact: " +
  "''”. The fix was to run MLflow as its own tracking server (mlflow server … --serve-artifacts) so every " +
  "client — training, registration, and all running services — resolves paths against the same server, " +
  "not its own filesystem. This is documented in docker-compose.yml and is the reason an mlflow service " +
  "exists as its own container rather than a bare sqlite file."
));

body.push(H2("7.5 Cost-Based Thresholds"));
body.push(P("training/select_thresholds.py grid-searches two cutoffs on the validation split to minimise an expected-cost function (fraud allowed costs the full amount; a false STEP_UP costs ₹50 friction; a false BLOCK costs ₹500 churn; a STEP_UP'd fraud still carries 20% residual risk), subject to the STEP_UP rate staying under 3%."));
body.push(simpleTable(["Parameter", "Value"], [
  ["t_stepup", "0.000401"],
  ["t_block", "0.000737"],
  ["Resulting STEP_UP rate", "0.36%"],
  ["Expected cost (validation)", "₹75,900"],
], [40, 60]));

body.push(H2("7.6 Testing"));
body.push(P("33 tests, all passing, covering: data quality, feature parity (no train/serve skew), the QA2 model-quality gate with directional sanity checks, QA3 explainability latency, QA1 performance, API contract validation, event pub-sub with consumer-group crash replay, and the FR9 fallback/heartbeat/hard-rules logic. See the screenshot in Section 7.8 and the test files under tests/."));

body.push(H2("7.7 A Second Real Bug, Found by Running the System"));
body.push(P(
  "Unit tests with mocks never exercised the full HTTP round-trip between services. Once the whole stack " +
  "ran together in Docker, every /v1/payments/triage call initially returned 500: triage-api was reading " +
  "a request timer's elapsed value while still inside its own measurement block, before the value was set. " +
  "Fixed by restructuring the handler so the response is only built after the timing block closes. " +
  "A second bug surfaced the same way: Redis's blocking XREADGROUP needs the client library's own socket " +
  "timeout set comfortably above the server-side block duration, or the client gives up and the consumer " +
  "silently stops reading new events. Both are the kind of integration-only bug the course material " +
  "(Hidden Technical Debt, Sculley et al.) warns about — correct in isolation, wrong once wired together."
));

body.push(H2("7.8 Screenshots and Evidence"));
body.push(P("The stack was brought up fresh (docker compose down -v, retrain, docker compose up -d --build) and seeded with a small, readable set of scenario payments (scripts/seed_demo.sh) before capturing these."));

const shots = [
  ["01_compose_ps.png", "Figure 5. All 8 containers up; scoring, triage-api, redis and mlflow show (healthy) from their own healthchecks — each service can be built, deployed and scaled independently.", 5.5],
  ["02_mlflow_comparison.png", "Figure 6. MLflow comparing the three trained models' runs side by side. LightGBM and Random Forest are effectively tied on accuracy; Section 7.3 explains how LightGBM was chosen between them.", 6.0],
  ["03_mlflow_registry.png", "Figure 7. The registered model fraud-triage-model, version 1, aliased champion — scoring-service always loads whichever version currently holds this alias.", 6.0],
  ["04a_swagger_allow.png", "Figure 8a. A normal ₹500 transfer via Swagger: very low risk score, ALLOW, 48 ms.", 6.0],
  ["04b_swagger_block.png", "Figure 8b. A transfer that empties the sender's balance into a fresh, zero-balance account: score 0.9999, BLOCK, with plain-language reason codes.", 6.0],
  ["05_swagger_422.png", "Figure 9. A negative amount is rejected with 422 before it ever reaches the model — schema validation at the edge.", 5.6],
  ["06_redis_stream.png", "Figure 10. XINFO GROUPS on payment.decided shows all three consumers with lag: 0 — every published event has been read by all three independently.", 5.6],
  ["07_case_queue.png", "Figure 11. The case queue sorted by priority: the ₹45,000 mule case (priority 44,999) ranks above the ₹9,500 account-takeover case (priority 9,500).", 5.0],
  ["07b_case_resolved.png", "Figure 12. After “Confirm fraud”, the case moves to Resolved with verdict CONFIRMED_FRAUD — saved as a label for the next retraining round.", 3.0],
  ["08_monitoring.png", "Figure 13. The monitoring tab: decision mix and PSI drift, both computed live from the same event stream.", 5.0],
  ["09a_fallback_degraded.png", "Figure 14a. After stopping the scoring container, a new payment still returns a response — STEP_UP with degraded: true — instead of hanging or failing.", 2.2],
  ["09b_fallback_recovered.png", "Figure 14b. A few seconds after restarting scoring, the gateway's own heartbeat detects it is back up, with no restart of the gateway needed.", 1.8],
  ["10_scaling.png", "Figure 15. docker compose up --scale scoring=2 adds a second scoring container while everything else keeps running — the model-serving part scales independently.", 2.4],
  ["11a_pytest.png", "Figure 16. All 33 tests pass, covering data quality, the model-quality gate, explainability, performance, API contracts and the event pipeline.", 6.0],
  ["11b_load_test.png", "Figure 17. 2000 requests at 50 concurrent: p95 148.0 ms, p99 212.2 ms — both inside the QA1 target (p95 ≤ 150 ms, p99 ≤ 300 ms).", 1.6],
];
for (const [file, cap, maxH] of shots) {
  body.push(...CenteredImage(`docs/screenshots/${file}`, CONTENT_WIDTH_DXA, maxH, cap));
}

// ---------------------------------------------------------- 8. Limitations
body.push(PageBr());
body.push(H1("8. Limitations and Future Work"));
body.push(Bullet("PaySim has no device, geo, or app-behaviour signals that a real UPI deployment would have; the near-separable fraud signal here (balance-draining transfers) is a property of the simulator and may not transfer directly to production data."));
body.push(Bullet("The SQLite case store and single-node Redis are demo-scale; a production PSP would use a managed Postgres and a clustered Redis/Kafka."));
body.push(Bullet("No shadow or canary deployment for new models — promotion is immediate once the quality gate passes. A real rollout would run the challenger alongside the champion first."));
body.push(Bullet("Fairness auditing is limited to “no demographic features used”; a full audit would check for proxy discrimination through behavioural features."));
body.push(Bullet("Kubernetes was out of scope; Docker Compose was used throughout, with the architecture designed so a move to Kubernetes would mostly be a deployment-manifest change, not a code change."));

// ---------------------------------------------------------- 9. Appendix
body.push(H1("9. Appendix — Key Source Files"));
body.push(P("Full source is at https://github.com/biswa13de/seml-g13-fraud-triage. Key files referenced in this report:"));
body.push(simpleTable(["File", "Purpose"], [
  ["common/features.py", "The single compute_features() function shared by training and serving"],
  ["common/feature_store.py", "Online Feature Store pattern (Redis + in-memory backends)"],
  ["training/build_features.py", "Data Preparation View, implemented (Section 3.3)"],
  ["training/train.py", "3-model MLflow comparison (Section 7.3)"],
  ["training/select_thresholds.py", "Cost-based triage thresholds (Section 7.5)"],
  ["training/register.py", "Quality gate + Model Registry pattern"],
  ["services/scoring/pipeline.py", "Pipe-and-filter inference pattern"],
  ["services/scoring/explain.py", "SHAP reason-code generation (QA3)"],
  ["services/triage_api/main.py", "Gateway: rules, heartbeat, policy, event publish"],
  ["services/triage_api/heartbeat.py", "Heartbeat tactic (FR9)"],
  ["docker-compose.yml", "All 8 services, matching Figure 4"],
  ["tests/", "33 tests (Section 7.6)"],
  ["13.ipynb", "Dataset, EDA, training comparison, SHAP, live API calls"],
], [35, 65]));

body.push(H1("Group Contribution"));
body.push(P("The contribution table on the cover page reflects real, verifiable work: commit history on the GitHub repository backs the percentages shown there."));

// ===================================================================== DOCUMENT ASSEMBLY
const COVER = [
  new Paragraph({ spacing: { before: 800 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI", bold: true, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [new TextRun({ text: "Work Integrated Learning Programmes Division", size: 20, color: COLORS.soft })] }),

  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600 }, children: [new TextRun({ text: "AIMLZG546 — Software Engineering for Machine Learning", bold: true, size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 600 }, children: [new TextRun({ text: "Assignment I", bold: true, size: 36, color: COLORS.accent })] }),

  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 }, children: [new TextRun({ text: "Real-Time Digital Payment Fraud Risk Triage System", bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 600 }, children: [new TextRun({ text: "Domain: Financial Technology (FinTech)", size: 22, color: COLORS.soft })] }),

  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 400 }, children: [new TextRun({ text: "Group No. 13", bold: true, size: 26 })] }),

  simpleTable(
    ["Sl. No", "BITS ID", "Name", "Contribution (Qualitative)", "% Contribution"],
    [
      ["1", "2025ae05576", "Anirudh Anand", "Report lead. Domain and problem statement, ML formulation, requirements and measurable goals (Sections 1 and 2), Business View, final report and PDF.", "25"],
      ["2", "2025ae05178", "Aniketh Paul", "GR4ML Analytics Design and Data Preparation views, top three quality requirements (Sections 3.2, 3.3, 4). Data part of the pipeline: EDA, ingest, clean, features, split.", "25"],
      ["3", "2025ae05898", "Pushadapu Sanjay Kumar", "Architecture diagram (ML and non-ML components), the two patterns (Sections 5 and 6). FastAPI service, input validation, prediction logging, automated tests.", "25"],
      ["4", "2025af05111", "Biswajeet Mahato (Group Lead)", "Model part of the pipeline: train and evaluate filters, threshold on validation, MLflow tracking and registry. End-to-end run, screenshots, Sections 7.2, 7.3, 8, 9.", "25"],
      ["", "", "", "Total", "100"],
    ],
    [5, 12, 18, 51, 9]
  ),
  new Paragraph({ spacing: { before: 400 }, children: [new TextRun({ text: "Repository: ", bold: true }), new TextRun({ text: "https://github.com/biswa13de/seml-g13-fraud-triage" })] }),
  new Paragraph({ children: [new TextRun({ text: "Submission date: ", bold: true }), new TextRun({ text: "09 October 2026" })] }),
  PageBr(),
];

const numbering = {
  config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 440, hanging: 260 } } } }] },
  ],
};

const doc = new Document({
  numbering,
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { bold: true, size: 30, color: COLORS.head }, paragraph: { spacing: { before: 360, after: 160 } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { bold: true, size: 25, color: COLORS.head }, paragraph: { spacing: { before: 280, after: 120 } } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { bold: true, size: 22, italics: true }, paragraph: { spacing: { before: 200, after: 100 } } },
    ],
  },
  sections: [
    {
      properties: { page: { margin: { top: MARGIN_DXA, bottom: MARGIN_DXA, left: MARGIN_DXA, right: MARGIN_DXA } } },
      children: [...COVER, ...body],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(REPO, "13.docx"), buf);
  console.log("Wrote 13.docx,", buf.length, "bytes");
});
