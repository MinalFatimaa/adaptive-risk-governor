import React, {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Cpu,
  Database,
  Eye,
  FileSearch,
  GitBranch,
  Gauge,
  Layers3,
  Network,
  Play,
  RefreshCw,
  Shield,
  ShieldAlert,
  Sparkles,
  UserRound,
  Users,
  Zap,
} from "lucide-react";


// ============================================================================
// API
// ============================================================================

async function getDashboard() {
  const response = await fetch("/api/dashboard");

  if (!response.ok) {
    throw new Error("Unable to load dashboard");
  }

  return response.json();
}


async function runPhase18(seed = 42) {
  const response = await fetch("/api/evaluation/phase18", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ seed }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));

    throw new Error(
      body.detail || "Phase 18 evaluation failed"
    );
  }

  return response.json();
}


// ============================================================================
// HELPERS
// ============================================================================

function percent(value) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `${(Number(value) * 100).toFixed(1)}%`;
}


function money(value) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `₹${Number(value).toLocaleString("en-IN", {
    maximumFractionDigits: 0,
  })}`;
}


function number(value) {
  if (value === null || value === undefined) {
    return "—";
  }

  return Number(value).toLocaleString("en-IN");
}


function riskClass(value) {
  const numeric = Number(value);

  if (numeric >= 0.7) {
    return "risk-high";
  }

  if (numeric >= 0.4) {
    return "risk-medium";
  }

  return "risk-low";
}


function riskLabel(value) {
  const numeric = Number(value);

  if (numeric >= 0.7) {
    return "HIGH";
  }

  if (numeric >= 0.4) {
    return "MEDIUM";
  }

  return "LOW";
}


// ============================================================================
// APP
// ============================================================================

export default function App() {
  const [page, setPage] = useState("command");

  const [dashboard, setDashboard] = useState(null);

  const [evaluation, setEvaluation] = useState(null);

  const [loading, setLoading] = useState(true);

  const [evaluationLoading, setEvaluationLoading] =
    useState(false);

  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);

        const data = await getDashboard();

        setDashboard(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, []);


  async function executeEvaluation() {
    try {
      setEvaluationLoading(true);
      setError(null);

      const data = await runPhase18(42);

      setEvaluation(data.result);
    } catch (err) {
      setError(err.message);
    } finally {
      setEvaluationLoading(false);
    }
  }


  if (loading) {
    return (
      <div className="boot-screen">
        <div className="boot-logo">
          <Shield />
        </div>

        <div className="boot-title">
          ADAPTIVE RISK GOVERNOR
        </div>

        <div className="boot-status">
          INITIALIZING RISK INTELLIGENCE
        </div>

        <div className="boot-loader">
          <span />
        </div>
      </div>
    );
  }


  return (
    <div className="app-shell">

      <Sidebar
        page={page}
        setPage={setPage}
      />


      <main className="main-area">

        <TopBar
          page={page}
          dashboard={dashboard}
        />


        {error && (
          <div className="error-banner">
            <AlertTriangle size={17} />

            <span>{error}</span>

            <button
              onClick={() => setError(null)}
            >
              Dismiss
            </button>
          </div>
        )}


        {page === "command" && (
          <CommandCenter
            dashboard={dashboard}
            evaluation={evaluation}
            onInvestigate={() =>
              setPage("investigation")
            }
            onEvaluate={executeEvaluation}
            evaluationLoading={evaluationLoading}
          />
        )}


        {page === "investigation" && (
          <InvestigationPage
            onGovernor={() =>
              setPage("governor")
            }
          />
        )}


        {page === "evidence" && (
          <EvidencePage />
        )}


        {page === "governor" && (
          <GovernorPage />
        )}


        {page === "evaluation" && (
          <EvaluationPage
            evaluation={evaluation}
            onEvaluate={executeEvaluation}
            loading={evaluationLoading}
          />
        )}

      </main>
    </div>
  );
}


// ============================================================================
// SIDEBAR
// ============================================================================

function Sidebar({
  page,
  setPage,
}) {
  const items = [
    {
      id: "command",
      label: "Command Center",
      icon: Gauge,
    },
    {
      id: "investigation",
      label: "Case Investigation",
      icon: FileSearch,
    },
    {
      id: "evidence",
      label: "Evidence Agent",
      icon: BrainCircuit,
    },
    {
      id: "governor",
      label: "Risk Governor",
      icon: Shield,
    },
    {
      id: "evaluation",
      label: "Evaluation",
      icon: Activity,
    },
  ];


  return (
    <aside className="sidebar">

      <div className="brand">

        <div className="brand-mark">
          <Shield size={22} />
        </div>

        <div>
          <div className="brand-name">
            ADAPTIVE
          </div>

          <div className="brand-sub">
            RISK GOVERNOR
          </div>
        </div>

      </div>


      <div className="system-status">

        <span className="status-dot" />

        SYSTEM OPERATIONAL

      </div>


      <nav className="nav">

        <div className="nav-section-label">
          INTELLIGENCE
        </div>

        {items.map((item) => {

          const Icon = item.icon;

          return (
            <button
              key={item.id}
              className={
                page === item.id
                  ? "nav-item active"
                  : "nav-item"
              }
              onClick={() =>
                setPage(item.id)
              }
            >

              <Icon size={18} />

              <span>
                {item.label}
              </span>

              {page === item.id && (
                <ChevronRight
                  size={15}
                  className="nav-chevron"
                />
              )}

            </button>
          );

        })}

      </nav>


      <div className="sidebar-bottom">

        <div className="model-status">

          <div className="model-status-icon">
            <Cpu size={17} />
          </div>

          <div>

            <div className="model-status-label">
              GNN GOVERNOR
            </div>

            <div className="model-status-value">
              CONNECTED
            </div>

          </div>

        </div>


        <div className="sidebar-version">
          RISK INTELLIGENCE v1.0
        </div>

      </div>

    </aside>
  );
}


// ============================================================================
// TOP BAR
// ============================================================================

function TopBar({
  page,
  dashboard,
}) {
  const labels = {
    command: "COMMAND CENTER",
    investigation: "CASE INVESTIGATION",
    evidence: "EVIDENCE INTELLIGENCE",
    governor: "RISK GOVERNOR",
    evaluation: "CLOSED-LOOP EVALUATION",
  };


  return (
    <header className="topbar">

      <div>

        <div className="breadcrumb">
          RISK INTELLIGENCE
          <ChevronRight size={13} />
          {labels[page]}
        </div>

        <div className="page-heading">
          {labels[page]}
        </div>

      </div>


      <div className="topbar-right">

        <div className="live-indicator">
          <span />
          LIVE SYSTEM
        </div>

        <div className="topbar-time">
          {new Date().toLocaleTimeString(
            "en-IN",
            {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            }
          )}
        </div>

        <div className="avatar">
          <UserRound size={16} />
        </div>

      </div>

    </header>
  );
}


// ============================================================================
// COMMAND CENTER
// ============================================================================

function CommandCenter({
  dashboard,
  evaluation,
  onInvestigate,
  onEvaluate,
  evaluationLoading,
}) {
  const metrics =
    evaluation?.population_metrics || [];


  return (
    <div className="page-content">

      <section className="hero">

        <div className="hero-copy">

          <div className="eyebrow">
            <Sparkles size={14} />
            ADAPTIVE ABUSE DEFENSE
          </div>

          <h1>
            AI that learns when
            <span> behavior changes.</span>
          </h1>

          <p>
            The Adaptive Risk Governor sits between
            an AI support agent and the business outcome,
            detecting strategic behavior before it becomes
            economic loss.
          </p>


          <div className="hero-actions">

            <button
              className="primary-button"
              onClick={onInvestigate}
            >
              <FileSearch size={17} />
              Investigate a Case
              <ArrowRight size={16} />
            </button>


            <button
              className="secondary-button"
              onClick={onEvaluate}
              disabled={evaluationLoading}
            >
              {evaluationLoading ? (
                <RefreshCw
                  size={16}
                  className="spin"
                />
              ) : (
                <Play size={16} />
              )}

              Run Closed-Loop Evaluation
            </button>

          </div>

        </div>


        <RiskNetwork />

      </section>


      <section className="metric-grid">

        <MetricCard
          icon={ShieldAlert}
          label="Adaptive Abuse"
          value={
            metrics.find(
              x =>
                x.population_group ===
                "ADAPTIVE_ABUSIVE"
            )?.governed_approval_rate
          }
          formatter={percent}
          caption="Governed approval rate"
          accent="red"
        />

        <MetricCard
          icon={Zap}
          label="Prevented Loss"
          value={
            metrics.reduce(
              (sum, x) =>
                sum +
                Number(
                  x.prevented_approval_value || 0
                ),
              0
            )
          }
          formatter={money}
          caption="Across evaluated populations"
          accent="cyan"
        />

        <MetricCard
          icon={Network}
          label="GNN Risk"
          value={
            metrics.length
              ? Math.max(
                  ...metrics.map(
                    x =>
                      Number(
                        x.maximum_gnn_risk || 0
                      )
                  )
                )
              : null
          }
          formatter={v =>
            v === null || v === undefined
              ? "—"
              : Number(v).toFixed(2)
          }
          caption="Maximum observed"
          accent="purple"
        />

        <MetricCard
          icon={Users}
          label="Episodes"
          value={
            evaluation?.total_episodes
          }
          formatter={number}
          caption="Closed-loop interactions"
          accent="green"
        />

      </section>


      <section className="architecture-section">

        <SectionHeading
          eyebrow="SYSTEM PIPELINE"
          title="How the Governor works"
          description="Every layer contributes evidence before the business decision is allowed to proceed."
        />


        <div className="pipeline">

          {dashboard?.architecture?.map(
            (item, index) => (
              <React.Fragment key={item.id}>

                <PipelineNode
                  item={item}
                  index={index}
                />

                {index <
                  dashboard.architecture.length -
                    1 && (
                  <div className="pipeline-arrow">
                    <ArrowRight size={18} />
                  </div>
                )}

              </React.Fragment>
            )
          )}

        </div>

      </section>


      <section className="lower-grid">

        <div className="panel">

          <PanelHeader
            icon={Shield}
            title="Governor Action Space"
            subtitle="The governor controls the agent's path"
          />


          <div className="action-list">

            <GovernorAction
              type="ALLOW"
              title="Allow Agent Decision"
              description="Risk remains within acceptable bounds."
              icon={CheckCircle2}
            />

            <GovernorAction
              type="EVIDENCE"
              title="Request Additional Evidence"
              description="Insufficient confidence requires another evidence pass."
              icon={FileSearch}
            />

            <GovernorAction
              type="HUMAN"
              title="Escalate to Human Review"
              description="High-risk behavior requires human intervention."
              icon={UserRound}
            />

          </div>

        </div>


        <div className="panel">

          <PanelHeader
            icon={BrainCircuit}
            title="Risk Intelligence"
            subtitle="Signals feeding the governor"
          />


          <div className="signal-cloud">

            {(
              dashboard?.signals || []
            ).map(signal => (
              <div
                className="signal-chip"
                key={signal}
              >
                <span />
                {signal}
              </div>
            ))}

          </div>


          <div className="intelligence-note">

            <Network size={18} />

            <span>
              Network and behavioral signals remain
              internal to the Governor and are not exposed
              directly to the support agent.
            </span>

          </div>

        </div>

      </section>

    </div>
  );
}


// ============================================================================
// RISK NETWORK
// ============================================================================

function RiskNetwork() {
  const nodes = [
    [18, 25],
    [34, 45],
    [49, 20],
    [63, 38],
    [78, 22],
    [86, 55],
    [65, 70],
    [44, 72],
    [27, 66],
    [52, 50],
  ];


  const edges = [
    [0, 1],
    [0, 2],
    [1, 3],
    [1, 8],
    [2, 3],
    [2, 4],
    [3, 6],
    [3, 9],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [7, 9],
    [8, 9],
    [9, 3],
  ];


  return (
    <div className="network-visual">

      <div className="network-header">

        <span>
          BEHAVIORAL NETWORK
        </span>

        <span className="network-live">
          <i />
          LIVE
        </span>

      </div>


      <svg
        className="network-svg"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >

        {edges.map(
          ([a, b], index) => {

            const start = nodes[a];
            const end = nodes[b];

            return (
              <line
                key={index}
                x1={start[0]}
                y1={start[1]}
                x2={end[0]}
                y2={end[1]}
                className="network-edge"
              />
            );
          }
        )}


        {nodes.map(
          ([x, y], index) => (
            <circle
              key={index}
              cx={x}
              cy={y}
              r={
                index === 9
                  ? 2.3
                  : 1.5
              }
              className={
                index === 9
                  ? "network-node core"
                  : "network-node"
              }
            />
          )
        )}

      </svg>


      <div className="network-footer">

        <div>
          <strong>10</strong>
          linked entities
        </div>

        <div>
          <strong>GNN</strong>
          relationship model
        </div>

        <div>
          <strong>0.20</strong>
          GNN fusion weight
        </div>

      </div>

    </div>
  );
}


// ============================================================================
// METRIC CARD
// ============================================================================

function MetricCard({
  icon: Icon,
  label,
  value,
  formatter,
  caption,
  accent,
}) {
  return (
    <div className={`metric-card ${accent}`}>

      <div className="metric-top">

        <div className="metric-icon">
          <Icon size={18} />
        </div>

        <div className="metric-label">
          {label}
        </div>

      </div>


      <div className="metric-value">
        {formatter(value)}
      </div>


      <div className="metric-caption">
        {caption}
      </div>

    </div>
  );
}


// ============================================================================
// PIPELINE NODE
// ============================================================================

function PipelineNode({
  item,
  index,
}) {
  const icons = [
    FileSearch,
    BrainCircuit,
    Network,
    Shield,
    Activity,
    GitBranch,
  ];

  const Icon = icons[index] || CircleDot;


  return (
    <div className="pipeline-node">

      <div className="pipeline-icon">
        <Icon size={20} />
      </div>

      <div className="pipeline-index">
        0{index + 1}
      </div>

      <div className="pipeline-title">
        {item.label}
      </div>

      <div className="pipeline-description">
        {item.description}
      </div>

      <div className="pipeline-status">
        <span />
        {item.status}
      </div>

    </div>
  );
}


// ============================================================================
// PANEL
// ============================================================================

function PanelHeader({
  icon: Icon,
  title,
  subtitle,
}) {
  return (
    <div className="panel-header">

      <div className="panel-header-icon">
        <Icon size={17} />
      </div>

      <div>

        <div className="panel-title">
          {title}
        </div>

        <div className="panel-subtitle">
          {subtitle}
        </div>

      </div>

    </div>
  );
}


// ============================================================================
// GOVERNOR ACTION
// ============================================================================

function GovernorAction({
  type,
  title,
  description,
  icon: Icon,
}) {
  return (
    <div className="governor-action">

      <div className={`action-icon ${type.toLowerCase()}`}>
        <Icon size={17} />
      </div>

      <div className="action-copy">

        <div className="action-title">
          {title}
        </div>

        <div className="action-description">
          {description}
        </div>

      </div>

      <div className="action-code">
        {type}
      </div>

    </div>
  );
}


// ============================================================================
// SECTION HEADING
// ============================================================================

function SectionHeading({
  eyebrow,
  title,
  description,
}) {
  return (
    <div className="section-heading">

      <div className="eyebrow">
        {eyebrow}
      </div>

      <h2>
        {title}
      </h2>

      <p>
        {description}
      </p>

    </div>
  );
}


// ============================================================================
// INVESTIGATION
// ============================================================================

function InvestigationPage({
  onGovernor,
}) {
  const [step, setStep] = useState(0);

  const steps = [
    {
      title: "Claim received",
      description:
        "Customer submits a refund request for an order.",
      icon: FileSearch,
    },
    {
      title: "Evidence retrieved",
      description:
        "Order, delivery and historical interaction evidence are collected.",
      icon: Database,
    },
    {
      title: "Behavior analyzed",
      description:
        "Temporal, semantic and strategic signals are evaluated.",
      icon: BrainCircuit,
    },
    {
      title: "Network inspected",
      description:
        "Graph relationships reveal connected behavioral patterns.",
      icon: Network,
    },
    {
      title: "Governor decision",
      description:
        "Risk is fused and the next safe action is selected.",
      icon: Shield,
    },
  ];


  return (
    <div className="page-content">

      <div className="case-header">

        <div>

          <div className="eyebrow">
            LIVE INVESTIGATION
          </div>

          <h1>
            Refund Case
            <span className="mono">
              {" "}#RG-2026-00421
            </span>
          </h1>

          <p>
            A complete trace from customer claim to
            Governor action.
          </p>

        </div>


        <div className="case-risk-badge">
          <div className="risk-pulse" />

          <div>

            <span>
              CURRENT RISK
            </span>

            <strong>
              HIGH
            </strong>

          </div>

        </div>

      </div>


      <div className="investigation-grid">

        <div className="panel investigation-main">

          <PanelHeader
            icon={Activity}
            title="Investigation Trace"
            subtitle="Evidence and intelligence pipeline"
          />


          <div className="trace">

            {steps.map(
              (item, index) => {

                const Icon = item.icon;

                const active =
                  index <= step;

                return (
                  <React.Fragment
                    key={item.title}
                  >

                    <button
                      className={
                        active
                          ? "trace-step active"
                          : "trace-step"
                      }
                      onClick={() =>
                        setStep(index)
                      }
                    >

                      <div className="trace-number">
                        {active ? (
                          <CheckCircle2 size={18} />
                        ) : (
                          index + 1
                        )}
                      </div>

                      <div className="trace-icon">
                        <Icon size={18} />
                      </div>

                      <div className="trace-copy">

                        <div className="trace-title">
                          {item.title}
                        </div>

                        <div className="trace-description">
                          {item.description}
                        </div>

                      </div>

                      <ChevronRight size={17} />

                    </button>


                    {index <
                      steps.length - 1 && (
                      <div
                        className={
                          active
                            ? "trace-connector active"
                            : "trace-connector"
                        }
                      />
                    )}

                  </React.Fragment>
                );
              }
            )}

          </div>

        </div>


        <div className="investigation-side">

          <CaseDetails />

          <RiskSignals />

        </div>

      </div>


      <div className="investigation-bottom">

        <div className="evidence-preview">

          <div className="preview-heading">
            <FileSearch size={17} />
            EVIDENCE PACKAGE
          </div>

          <div className="evidence-items">

            <EvidenceItem
              label="Order history"
              status="VERIFIED"
            />

            <EvidenceItem
              label="Delivery record"
              status="VERIFIED"
            />

            <EvidenceItem
              label="Claim consistency"
              status="FLAGGED"
            />

            <EvidenceItem
              label="Network relationships"
              status="FLAGGED"
            />

          </div>

        </div>


        <button
          className="primary-button large"
          onClick={onGovernor}
        >
          Inspect Risk Governor
          <ArrowRight size={17} />
        </button>

      </div>

    </div>
  );
}


// ============================================================================
// CASE DETAILS
// ============================================================================

function CaseDetails() {
  const details = [
    ["Customer", "CUS-084291"],
    ["Order", "ORD-729410"],
    ["Claim", "PRODUCT_NOT_RECEIVED"],
    ["Amount", "₹750"],
    ["Support decision", "APPROVE"],
  ];


  return (
    <div className="panel">

      <PanelHeader
        icon={Eye}
        title="Case Context"
        subtitle="Current interaction"
      />


      <div className="detail-list">

        {details.map(
          ([label, value]) => (
            <div
              className="detail-row"
              key={label}
            >

              <span>
                {label}
              </span>

              <strong>
                {value}
              </strong>

            </div>
          )
        )}

      </div>

    </div>
  );
}


// ============================================================================
// RISK SIGNALS
// ============================================================================

function RiskSignals() {
  const signals = [
    ["Behavioral risk", 0.81],
    ["Network risk", 0.73],
    ["Temporal abnormality", 0.69],
    ["Semantic inconsistency", 0.77],
    ["Strategic behavior", 0.84],
  ];


  return (
    <div className="panel">

      <PanelHeader
        icon={Zap}
        title="Risk Signals"
        subtitle="Internal Governor intelligence"
      />


      <div className="risk-signal-list">

        {signals.map(
          ([label, value]) => (
            <div
              className="risk-signal"
              key={label}
            >

              <div className="risk-signal-top">

                <span>
                  {label}
                </span>

                <strong>
                  {value.toFixed(2)}
                </strong>

              </div>

              <div className="risk-bar">
                <div
                  className={riskClass(value)}
                  style={{
                    width: `${value * 100}%`,
                  }}
                />
              </div>

            </div>
          )
        )}

      </div>

    </div>
  );
}


// ============================================================================
// EVIDENCE
// ============================================================================

function EvidenceItem({
  label,
  status,
}) {
  const flagged =
    status === "FLAGGED";


  return (
    <div className="evidence-item">

      <div
        className={
          flagged
            ? "evidence-status flagged"
            : "evidence-status"
        }
      >
        {flagged ? (
          <AlertTriangle size={14} />
        ) : (
          <CheckCircle2 size={14} />
        )}
      </div>

      <span>
        {label}
      </span>

      <small>
        {status}
      </small>

    </div>
  );
}


function EvidencePage() {
  const events = [
    {
      time: "00:01",
      title: "Claim received",
      description:
        "PRODUCT_NOT_RECEIVED · ₹750",
      icon: FileSearch,
      state: "complete",
    },
    {
      time: "00:03",
      title: "Transaction evidence",
      description:
        "Order and payment records retrieved.",
      icon: Database,
      state: "complete",
    },
    {
      time: "00:05",
      title: "Fulfillment evidence",
      description:
        "Delivery record and order history checked.",
      icon: CheckCircle2,
      state: "complete",
    },
    {
      time: "00:07",
      title: "Behavioral cross-check",
      description:
        "Claim behavior compared against historical pattern.",
      icon: BrainCircuit,
      state: "warning",
    },
    {
      time: "00:09",
      title: "Evidence package",
      description:
        "Contradictory signals prepared for Governor.",
      icon: Shield,
      state: "current",
    },
  ];


  return (
    <div className="page-content">

      <SectionHeading
        eyebrow="AGENT B · EVIDENCE INTELLIGENCE"
        title="Evidence before decision."
        description="The evidence layer creates a structured view of the interaction before the Governor decides how the support agent may proceed."
      />


      <div className="evidence-layout">

        <div className="panel">

          <PanelHeader
            icon={BrainCircuit}
            title="Evidence Agent Trace"
            subtitle="Observable reasoning sequence"
          />


          <div className="agent-trace">

            {events.map(
              (event, index) => {

                const Icon = event.icon;

                return (
                  <div
                    className={`agent-event ${event.state}`}
                    key={event.title}
                  >

                    <div className="event-time">
                      {event.time}
                    </div>

                    <div className="event-node">
                      <Icon size={16} />
                    </div>

                    <div className="event-copy">

                      <strong>
                        {event.title}
                      </strong>

                      <span>
                        {event.description}
                      </span>

                    </div>

                    {index <
                      events.length - 1 && (
                      <div className="event-line" />
                    )}

                  </div>
                );
              }
            )}

          </div>

        </div>


        <div className="panel evidence-package">

          <PanelHeader
            icon={Layers3}
            title="Evidence Package"
            subtitle="Structured output"
          />


          <div className="package-score">

            <div>
              <span>
                EVIDENCE CONFIDENCE
              </span>

              <strong>
                82%
              </strong>
            </div>

            <div className="confidence-ring">
              82
            </div>

          </div>


          <div className="package-section">

            <label>
              VERIFIED
            </label>

            <div className="package-row">
              Order record
              <CheckCircle2 size={15} />
            </div>

            <div className="package-row">
              Delivery record
              <CheckCircle2 size={15} />
            </div>

            <div className="package-row">
              Payment history
              <CheckCircle2 size={15} />
            </div>

          </div>


          <div className="package-section">

            <label>
              CONFLICTS
            </label>

            <div className="package-row warning">
              Claim history mismatch
              <AlertTriangle size={15} />
            </div>

            <div className="package-row warning">
              Network behavior anomaly
              <AlertTriangle size={15} />
            </div>

          </div>

        </div>

      </div>

    </div>
  );
}


// ============================================================================
// GOVERNOR
// ============================================================================

function GovernorPage() {
  const signals = [
    {
      name: "GNN risk",
      value: 0.79,
      source: "GraphSAGE",
    },
    {
      name: "Behavioral risk",
      value: 0.83,
      source: "Adaptive model",
    },
    {
      name: "Network risk",
      value: 0.73,
      source: "Interaction graph",
    },
    {
      name: "Strategic behavior",
      value: 0.84,
      source: "Adaptation detector",
    },
    {
      name: "Evidence consistency",
      value: 0.31,
      source: "Evidence engine",
    },
  ];


  const finalRisk = 0.81;


  return (
    <div className="page-content">

      <SectionHeading
        eyebrow="GOVERNOR CONTROL LAYER"
        title="The decision is not just a score."
        description="The Governor fuses risk intelligence and determines how the support agent is allowed to proceed."
      />


      <div className="governor-flow">

        <div className="governor-column">

          <div className="flow-label">
            INPUT INTELLIGENCE
          </div>


          {signals.map(
            signal => (
              <div
                className="intelligence-card"
                key={signal.name}
              >

                <div className="intelligence-card-icon">
                  <Network size={16} />
                </div>

                <div className="intelligence-card-copy">

                  <strong>
                    {signal.name}
                  </strong>

                  <span>
                    {signal.source}
                  </span>

                </div>

                <div
                  className={`intelligence-score ${riskClass(
                    signal.value
                  )}`}
                >
                  {signal.value.toFixed(2)}
                </div>

              </div>
            )
          )}

        </div>


        <div className="flow-connector">

          <div className="connector-line" />

          <ArrowRight size={21} />

          <span>
            FUSION
          </span>

        </div>


        <div className="governor-core">

          <div className="core-orbit orbit-one" />
          <div className="core-orbit orbit-two" />

          <div className="core-center">

            <Shield size={34} />

            <span>
              GNN
            </span>

            <strong>
              GOVERNOR
            </strong>

          </div>

        </div>


        <div className="flow-connector">

          <div className="connector-line" />

          <ArrowRight size={21} />

          <span>
            ACTION
          </span>

        </div>


        <div className="decision-card">

          <div className="decision-label">
            GOVERNOR DECISION
          </div>

          <div className="decision-risk">
            {finalRisk.toFixed(2)}
          </div>

          <div className="decision-level">
            HIGH RISK
          </div>

          <div className="decision-action">
            ESCALATE TO
            <br />
            HUMAN REVIEW
          </div>

          <div className="decision-code">
            ESCALATE_TO_HUMAN_REVIEW
          </div>

        </div>

      </div>


      <div className="governor-detail-grid">

        <div className="panel">

          <PanelHeader
            icon={GitBranch}
            title="Fusion Logic"
            subtitle="GNN-aware risk composition"
          />


          <div className="formula">

            <span>
              Base fused risk
            </span>

            <strong>
              0.82
            </strong>

            <span>
              × 80%
            </span>

            <span>
              +
            </span>

            <span>
              GNN risk
            </span>

            <strong>
              0.79
            </strong>

            <span>
              × 20%
            </span>

            <ArrowRight size={17} />

            <strong className="formula-result">
              0.81
            </strong>

          </div>


          <div className="metadata-row">

            <span>
              GNN fusion weight
            </span>

            <strong>
              0.20
            </strong>

          </div>

          <div className="metadata-row">

            <span>
              GNN enabled
            </span>

            <strong className="positive">
              TRUE
            </strong>

          </div>

          <div className="metadata-row">

            <span>
              Agent-A isolation
            </span>

            <strong className="positive">
              VERIFIED
            </strong>

          </div>

        </div>


        <div className="panel">

          <PanelHeader
            icon={ShieldAlert}
            title="Reason Codes"
            subtitle="Why the Governor intervened"
          />


          <div className="reason-list">

            <ReasonCode>
              HIGH_GNN_RISK
            </ReasonCode>

            <ReasonCode>
              STRATEGIC_BEHAVIOR_DETECTED
            </ReasonCode>

            <ReasonCode>
              NETWORK_ABNORMALITY
            </ReasonCode>

            <ReasonCode>
              EVIDENCE_CONFLICT
            </ReasonCode>

            <ReasonCode>
              GNN_HIGH_RISK_OVERRIDE
            </ReasonCode>

          </div>

        </div>

      </div>

    </div>
  );
}


function ReasonCode({
  children,
}) {
  return (
    <div className="reason-code">

      <span />

      <code>
        {children}
      </code>

      <ChevronRight size={15} />

    </div>
  );
}


// ============================================================================
// EVALUATION
// ============================================================================

function EvaluationPage({
  evaluation,
  onEvaluate,
  loading,
}) {
  return (
    <div className="page-content">

      <SectionHeading
        eyebrow="PHASE 18 · EVALUATION"
        title="Does the Governor actually work?"
        description="Closed-loop evaluation against adaptive and human populations. These metrics come from the canonical Phase 18 implementation."
      />


      <div className="evaluation-control">

        <div>

          <div className="control-title">
            Closed-loop GNN Governor Evaluation
          </div>

          <div className="control-description">
            Runs the real Phase 18 pipeline using seed 42.
          </div>

        </div>


        <button
          className="primary-button"
          onClick={onEvaluate}
          disabled={loading}
        >

          {loading ? (
            <>
              <RefreshCw
                size={16}
                className="spin"
              />
              Running...
            </>
          ) : (
            <>
              <Play size={16} />
              Run Evaluation
            </>
          )}

        </button>

      </div>


      {!evaluation && !loading && (
        <div className="evaluation-empty">

          <div className="empty-icon">
            <Activity size={25} />
          </div>

          <h3>
            No evaluation loaded
          </h3>

          <p>
            Run Phase 18 to populate this page
            with real closed-loop results.
          </p>

        </div>
      )}


      {evaluation && (
        <EvaluationResults
          evaluation={evaluation}
        />
      )}

    </div>
  );
}


// ============================================================================
// EVALUATION RESULTS
// ============================================================================

function EvaluationResults({
  evaluation,
}) {
  const metrics =
    evaluation.population_metrics || [];


  return (
    <div className="evaluation-results">

      <div className="metric-grid">

        <MetricCard
          icon={Activity}
          label="Episodes"
          value={evaluation.total_episodes}
          formatter={number}
          caption="Total closed-loop episodes"
          accent="cyan"
        />

        <MetricCard
          icon={ShieldAlert}
          label="Adaptive Advantage"
          value={
            evaluation.adaptive_governor_advantage
          }
          formatter={v =>
            v == null
              ? "—"
              : Number(v).toFixed(3)
          }
          caption="Governor advantage"
          accent="purple"
        />

        <MetricCard
          icon={Zap}
          label="Policy Change"
          value={
            evaluation.adaptive_policy_change
          }
          formatter={percent}
          caption="Adaptive policy change"
          accent="green"
        />

        <MetricCard
          icon={Network}
          label="Intervention Gap"
          value={
            evaluation.adaptive_vs_human_intervention_gap
          }
          formatter={percent}
          caption="Adaptive vs human gap"
          accent="red"
        />

      </div>


      <div className="panel">

        <PanelHeader
          icon={Users}
          title="Population Results"
          subtitle="Observed closed-loop behavior"
        />


        <div className="table-wrapper">

          <table>

            <thead>
              <tr>
                <th>Population</th>
                <th>Episodes</th>
                <th>Baseline Approval</th>
                <th>Governed Approval</th>
                <th>Reduction</th>
                <th>Intervention</th>
                <th>Prevented Value</th>
                <th>Mean GNN Risk</th>
              </tr>
            </thead>


            <tbody>

              {metrics.map(
                metric => (
                  <tr
                    key={
                      metric.population_group
                    }
                  >

                    <td>
                      <span className="population-name">
                        {metric.population_group}
                      </span>
                    </td>

                    <td>
                      {number(metric.episodes)}
                    </td>

                    <td>
                      {percent(
                        metric.baseline_approval_rate
                      )}
                    </td>

                    <td>
                      <strong>
                        {percent(
                          metric.governed_approval_rate
                        )}
                      </strong>
                    </td>

                    <td className="positive">
                      {percent(
                        metric.approval_reduction
                      )}
                    </td>

                    <td>
                      {percent(
                        metric.intervention_rate
                      )}
                    </td>

                    <td>
                      {money(
                        metric.prevented_approval_value
                      )}
                    </td>

                    <td>
                      {Number(
                        metric.mean_gnn_risk || 0
                      ).toFixed(3)}
                    </td>

                  </tr>
                )
              )}

            </tbody>

          </table>

        </div>

      </div>


      <div className="evaluation-insight">

        <div className="insight-icon">
          <BrainCircuit size={21} />
        </div>

        <div>

          <strong>
            What this evaluation proves
          </strong>

          <p>
            The Governor is evaluated inside the interaction
            loop rather than as a static classifier. Adaptive
            customers can change their behavior after observing
            previous outcomes, allowing the system to test
            whether the defense continues to detect abuse.
          </p>

        </div>

      </div>

    </div>
  );
}