import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowRight,
  Bot,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleDot,
  Cpu,
  Database,
  Eye,
  Fingerprint,
  Gauge,
  GitBranch,
  Globe2,
  Lock,
  Network,
  Radar,
  RefreshCw,
  Search,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Target,
  Terminal,
  Timer,
  TrendingDown,
  UserRound,
  Users,
  WalletCards,
  Zap,
} from "lucide-react";

import "./styles.css";


/* ============================================================================
   API
============================================================================ */

const API_BASE =
  import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function getDashboard() {
  const response = await fetch(`${API_BASE}/api/dashboard`);

  if (!response.ok) {
    throw new Error(
      `Dashboard request failed: ${response.status}`
    );
  }

  return response.json();
}

async function runPhase18(seed = 42) {
  const response = await fetch(
    `${API_BASE}/api/evaluation/phase18`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ seed }),
    }
  );

  if (!response.ok) {
    const body = await response.text();

    throw new Error(
      body || `Evaluation failed: ${response.status}`
    );
  }

  return response.json();
}


/* ============================================================================
   HELPERS
============================================================================ */

function pct(value, digits = 1) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(Number(value))
  ) {
    return "—";
  }

  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function pp(value, digits = 1) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(Number(value))
  ) {
    return "—";
  }

  return `${(Number(value) * 100).toFixed(digits)} pp`;
}

function money(value) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(Number(value))
  ) {
    return "—";
  }

  return `₹${Number(value).toLocaleString("en-IN", {
    maximumFractionDigits: 0,
  })}`;
}

function decimal(value, digits = 2) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(Number(value))
  ) {
    return "—";
  }

  return Number(value).toFixed(digits);
}

function clamp(value, min = 0, max = 1) {
  return Math.max(
    min,
    Math.min(max, Number(value) || 0)
  );
}

function riskLevel(value) {
  const n = Number(value);

  if (n >= 0.7) return "HIGH";
  if (n >= 0.3) return "MEDIUM";

  return "LOW";
}


/* ============================================================================
   POPULATION HELPERS
============================================================================ */

/*
 * Phase 18 identifies populations using population_group.
 *
 * Keep this resolver global because several pages consume the same
 * Phase 18 population records.
 */

function findPopulation(metrics, type) {
  if (!metrics) {
    return null;
  }

  const target = String(type)
    .trim()
    .toUpperCase();

  if (Array.isArray(metrics)) {
    return (
      metrics.find((item) => {
        const group = String(
          item?.population_group ??
          item?.population ??
          item?.population_type ??
          item?.type ??
          ""
        )
          .trim()
          .toUpperCase();

        return group === target;
      }) || null
    );
  }

  /*
   * Object keyed by population group:
   *
   * {
   *   ADAPTIVE_ABUSIVE: {...}
   * }
   */

  if (metrics[target]) {
    return metrics[target];
  }

  if (metrics[type]) {
    return metrics[type];
  }

  if (metrics[String(type).toLowerCase()]) {
    return metrics[String(type).toLowerCase()];
  }

  /*
   * Object containing population records rather than being directly
   * keyed by the population name.
   */

  for (const [key, value] of Object.entries(metrics)) {
    const group = String(
      value?.population_group ??
      value?.population ??
      value?.population_type ??
      value?.type ??
      key ??
      ""
    )
      .trim()
      .toUpperCase();

    if (group === target) {
      return value;
    }
  }

  return null;
}


function getMetric(obj, ...keys) {
  for (const key of keys) {
    if (
      obj &&
      obj[key] !== undefined &&
      obj[key] !== null
    ) {
      return obj[key];
    }
  }

  return null;
}


/*
 * Global prevented-value resolver.
 *
 * This is intentionally outside EvaluationPage so CommandCenter,
 * EvaluationPage and any future page can safely use it.
 */

function getPreventedValue(
  population,
  result = null,
  prefix = null
) {
  const direct = getMetric(
    population,
    "prevented_value",
    "value_prevented",
    "prevented_amount",
    "prevented_approval_value"
  );

  if (direct !== null) {
    return Number(direct);
  }

  if (result && prefix) {
    const topLevel = getMetric(
      result,
      `${prefix}_prevented_value`,
      `${prefix}_value_prevented`,
      `${prefix}_prevented_amount`
    );

    if (topLevel !== null) {
      return Number(topLevel);
    }
  }

  return null;
}


/* ============================================================================
   INTERSECTION HOOK
============================================================================ */

function useInView(options = {}) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!ref.current) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
        }
      },
      {
        threshold: 0.15,
        ...options,
      }
    );

    observer.observe(ref.current);

    return () => observer.disconnect();
  }, []);

  return [ref, visible];
}


/* ============================================================================
   SIGNAL THREADS
   Dependency-free replacement for WebThreads.
============================================================================ */

function SignalThreads({ dense = false }) {
  const threads = useMemo(
    () =>
      Array.from(
        { length: dense ? 13 : 9 },
        (_, index) => ({
          id: index,
          left: `${8 + index * (dense ? 7 : 10)}%`,
          delay: `${index * -1.8}s`,
          duration: `${9 + (index % 4) * 2}s`,
          drift: `${
            (index % 2 ? 1 : -1) *
            (10 + index * 2)
          }px`,
        })
      ),
    [dense]
  );

  return (
    <div
      className={`signal-threads ${
        dense ? "dense" : ""
      }`}
    >
      <div className="thread-grid" />

      {threads.map((thread) => (
        <div
          key={thread.id}
          className="signal-thread"
          style={{
            left: thread.left,
            animationDelay: thread.delay,
            animationDuration: thread.duration,
            "--thread-drift": thread.drift,
          }}
        />
      ))}

      <div className="thread-vignette" />
    </div>
  );
}


/* ============================================================================
   TELEMETRY
============================================================================ */

function TelemetryText({ children }) {
  const [text, setText] = useState(children);

  useEffect(() => {
    const original = String(children);
    const chars = "01X#<>[]{}";

    let timer;
    let iteration = 0;

    const run = () => {
      iteration += 1;

      const output = original
        .split("")
        .map((char, index) => {
          if (char === " ") {
            return " ";
          }

          if (index < iteration) {
            return char;
          }

          return chars[
            Math.floor(
              Math.random() * chars.length
            )
          ];
        })
        .join("");

      setText(output);

      if (iteration < original.length) {
        timer = window.setTimeout(run, 28);
      }
    };

    run();

    return () =>
      window.clearTimeout(timer);
  }, [children]);

  return (
    <span className="telemetry-text">
      {text}
    </span>
  );
}


/* ============================================================================
   GHOST CURSOR
============================================================================ */

function GhostCursor() {
  const [position, setPosition] = useState({
    x: 50,
    y: 50,
  });

  useEffect(() => {
    const handleMove = (event) => {
      const x =
        (event.clientX /
          window.innerWidth) *
        100;

      const y =
        (event.clientY /
          window.innerHeight) *
        100;

      setPosition({ x, y });
    };

    window.addEventListener(
      "pointermove",
      handleMove
    );

    return () => {
      window.removeEventListener(
        "pointermove",
        handleMove
      );
    };
  }, []);

  return (
    <div
      className="ghost-cursor"
      style={{
        "--cursor-x": `${position.x}%`,
        "--cursor-y": `${position.y}%`,
      }}
    />
  );
}


/* ============================================================================
   APP
============================================================================ */

export default function App() {
  const [page, setPage] = useState("why");
  const [dashboard, setDashboard] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [evaluationLoading, setEvaluationLoading] =
    useState(false);
  const [error, setError] = useState("");
  const [evaluationError, setEvaluationError] =
    useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data = await getDashboard();

        if (!cancelled) {
          setDashboard(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err.message ||
              "Unable to connect to backend."
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  async function executeEvaluation() {
    setEvaluationLoading(true);
    setEvaluationError("");

    try {
      const data = await runPhase18(42);

      setEvaluation(data);
    } catch (err) {
      setEvaluationError(
        err.message || "Evaluation failed."
      );
    } finally {
      setEvaluationLoading(false);
    }
  }

  useEffect(() => {
    executeEvaluation();
  }, []);

  const navigate = (nextPage) => {
    setPage(nextPage);

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  return (
    <div className="app-shell">
      <Sidebar
        page={page}
        navigate={navigate}
      />

      <div className="main-content">
        <TopBar
          page={page}
          evaluation={evaluation}
          loading={loading}
        />

        {error && (
          <div className="connection-banner">
            <AlertTriangle size={15} />

            <span>
              Backend unavailable — interface is
              running in presentation mode. Start
              FastAPI to load live evaluation data.
            </span>
          </div>
        )}

        <main className="page-content">
          {page === "why" && (
            <WhyMattersPage
              navigate={navigate}
              evaluation={evaluation}
            />
          )}

          {page === "command" && (
            <CommandCenter
              navigate={navigate}
              dashboard={dashboard}
              evaluation={evaluation}
            />
          )}

          {page === "investigate" && (
            <InvestigationPage />
          )}

          {page === "evidence" && (
            <EvidencePage />
          )}

          {page === "governor" && (
            <GovernorPage
              evaluation={evaluation}
            />
          )}

          {page === "evaluation" && (
            <EvaluationPage
              evaluation={evaluation}
              loading={evaluationLoading}
              error={evaluationError}
              onRun={executeEvaluation}
            />
          )}
        </main>
      </div>
    </div>
  );
}


/* ============================================================================
   SIDEBAR
============================================================================ */

function Sidebar({ page, navigate }) {
  const items = [
    {
      id: "why",
      label: "Why This Matters",
      icon: Radar,
    },
    {
      id: "command",
      label: "Command Center",
      icon: Activity,
    },
    {
      id: "investigate",
      label: "Case Investigation",
      icon: Search,
    },
    {
      id: "evidence",
      label: "Evidence Agent",
      icon: Fingerprint,
    },
    {
      id: "governor",
      label: "Risk Governor",
      icon: Shield,
    },
    {
      id: "evaluation",
      label: "Evaluation",
      icon: Gauge,
    },
  ];

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <Shield size={19} />
        </div>

        <div>
          <div className="brand-name">
            ARG
          </div>

          <div className="brand-subtitle">
            ADAPTIVE RISK GOVERNOR
          </div>
        </div>
      </div>

      <div className="sidebar-status">
        <span className="status-dot" />
        <span>SYSTEM OPERATIONAL</span>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-label">
          CONTROL PLANE
        </div>

        {items.map((item) => {
          const Icon = item.icon;

          return (
            <button
              key={item.id}
              className={`nav-item ${
                page === item.id
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                navigate(item.id)
              }
            >
              <Icon size={16} />

              <span>{item.label}</span>

              {page === item.id && (
                <ChevronRight
                  className="nav-chevron"
                  size={14}
                />
              )}
            </button>
          );
        })}
      </nav>

      <div className="sidebar-bottom">
        <div className="sidebar-mini-card">
          <div className="mini-label">
            <CircleDot size={11} />
            DEFENSE STATUS
          </div>

          <div className="mini-value">
            ACTIVE
          </div>

          <div className="mini-bar">
            <span />
          </div>

          <div className="mini-foot">
            Independent intervention layer
          </div>
        </div>

        <div className="version">
          ARG / PHASE 18
        </div>
      </div>
    </aside>
  );
}


/* ============================================================================
   TOP BAR
============================================================================ */

function TopBar({
  page,
  evaluation,
  loading,
}) {
  const labels = {
    why: "THREAT MODEL",
    command: "COMMAND CENTER",
    investigate: "CASE INVESTIGATION",
    evidence: "EVIDENCE INTELLIGENCE",
    governor: "RISK GOVERNOR",
    evaluation: "EVALUATION LAB",
  };

  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="topbar-path">
          ARG / {labels[page]}
        </span>

        <span className="topbar-divider" />

        <span className="topbar-live">
          <span className="status-dot" />
          LIVE SYSTEM
        </span>
      </div>

      <div className="topbar-right">
        <div className="topbar-chip">
          <Database size={12} />
          CANONICAL SRC
        </div>

        <div className="topbar-chip">
          <Lock size={12} />
          INFORMATION ISOLATED
        </div>

        <div className="topbar-seed">
          {loading
            ? "CONNECTING"
            : evaluation
            ? "SEED 42"
            : "AWAITING RUN"}
        </div>
      </div>
    </header>
  );
}


/* ============================================================================
   WHY THIS MATTERS
============================================================================ */

function WhyMattersPage({
  navigate,
  evaluation,
}) {
  const [heroRef, heroVisible] =
    useInView();

  return (
    <div className="page why-page">
      <section
        ref={heroRef}
        className={`hero-section ${
          heroVisible
            ? "is-visible"
            : ""
        }`}
      >
        <SignalThreads />
        <GhostCursor />

        <div className="hero-grid-overlay" />

        <div className="hero-content">
          <div className="eyebrow">
            <span className="eyebrow-line" />

            THE PROBLEM WASN'T HYPOTHETICAL

            <span className="eyebrow-live">
              ALREADY HAPPENING
            </span>
          </div>

          <h1>
            The agent you trust
            <br />
            with money is also
            <br />
            <span>
              the agent someone else
              is studying.
            </span>
          </h1>

          <p className="hero-copy">
            Autonomous systems do not need
            to be explicitly instructed to
            abuse a process. If an agent can
            observe outcomes, adapt its
            strategy and repeatedly probe a
            decision-maker, the decision
            boundary itself becomes an attack
            surface.
          </p>

          <div className="hero-actions">
            <button
              className="primary-button"
              onClick={() =>
                navigate("command")
              }
            >
              ENTER COMMAND CENTER
              <ArrowRight size={16} />
            </button>

            <button
              className="secondary-button"
              onClick={() =>
                navigate("evaluation")
              }
            >
              VIEW PROOF
              <TrendingDown size={15} />
            </button>
          </div>

          <div className="hero-telemetry">
            <div>
              <span>THREAT SURFACE</span>
              <strong>
                ADAPTIVE ABUSE
              </strong>
            </div>

            <div>
              <span>DEFENSE POSITION</span>
              <strong>
                PRE-ACTION
              </strong>
            </div>

            <div>
              <span>INTELLIGENCE</span>
              <strong>
                <TelemetryText>
                  BEHAVIOR + NETWORK + SEMANTIC
                </TelemetryText>
              </strong>
            </div>
          </div>
        </div>

        <div className="hero-visual">
          <ThreatSurface />
        </div>
      </section>

      <WhySection
        navigate={navigate}
        evaluation={evaluation}
      />
    </div>
  );
}


/* ============================================================================
   THREAT SURFACE
============================================================================ */

function ThreatSurface() {
  return (
    <div className="threat-surface">
      <div className="surface-header">
        <div>
          <span className="surface-kicker">
            LIVE RISK INTERCEPTION
          </span>

          <strong>
            DECISION SURFACE
          </strong>
        </div>

        <div className="surface-status">
          <span />
          MONITORING
        </div>
      </div>

      <div className="surface-body">
        <div className="surface-axis axis-x" />
        <div className="surface-axis axis-y" />

        <div className="surface-ring ring-1" />
        <div className="surface-ring ring-2" />
        <div className="surface-ring ring-3" />

        <div className="surface-core">
          <ShieldCheck size={25} />
          <span>GOVERNOR</span>
          <strong>INTERCEPT</strong>
        </div>

        <div className="surface-node node-a">
          <Bot size={14} />
          <span>AGENT A</span>
        </div>

        <div className="surface-node node-b">
          <BrainCircuit size={14} />
          <span>AGENT B</span>
        </div>

        <div className="surface-node node-network">
          <Network size={14} />
          <span>NETWORK</span>
        </div>

        <div className="surface-node node-semantic">
          <GitBranch size={14} />
          <span>SEMANTIC</span>
        </div>

        <div className="surface-packet packet-1" />
        <div className="surface-packet packet-2" />
        <div className="surface-packet packet-3" />
      </div>

      <div className="surface-footer">
        <span>PRIVATE SIGNALS</span>
        <span>FUSED RISK</span>
        <span>ACTION CONTROL</span>
      </div>
    </div>
  );
}


/* ============================================================================
   WHY SECTION
============================================================================ */

function WhySection({
  navigate,
  evaluation,
}) {
  return (
    <section className="why-section">
      <div className="section-intro">
        <div className="eyebrow">
          <span className="eyebrow-line" />
          HOW THE ATTACK SURFACE CHANGES
        </div>

        <h2>
          A static fraud model watches
          the claim.
          <br />
          <span>
            ARG watches the interaction.
          </span>
        </h2>

        <p>
          The problem is not simply whether
          one refund request looks
          suspicious. The problem is what
          happens when an adaptive actor
          learns how an autonomous support
          system responds.
        </p>
      </div>

      <div className="architecture-story">
        <StoryCard
          number="01"
          icon={Bot}
          title="Agent A"
          subtitle="SUPPORT DECISION AGENT"
          text="Evaluates the individual case using its permitted case-level information."
        />

        <div className="story-arrow">
          <ArrowRight size={20} />
        </div>

        <StoryCard
          number="02"
          icon={BrainCircuit}
          title="Agent B"
          subtitle="ADAPTIVE ADVERSARY"
          text="Observes outcomes across repeated interactions and changes its strategy."
          highlighted
        />

        <div className="story-arrow">
          <ArrowRight size={20} />
        </div>

        <StoryCard
          number="03"
          icon={Shield}
          title="Governor"
          subtitle="INDEPENDENT CONTROL LAYER"
          text="Uses signals Agent A cannot see to decide whether the autonomous action can proceed."
          governor
        />
      </div>

      <InformationBoundary />

      <div className="why-proof-strip">
        <div>
          <span>MEASURED IN PHASE 18</span>
          <strong>80</strong>
          <small>
            deterministic episodes
          </small>
        </div>

        <div>
          <span>ADAPTIVE ABUSE</span>

          <strong>
            {evaluation
              ? getAdaptiveReduction(
                  evaluation
                )
              : "21.46 pp"}
          </strong>

          <small>
            approval reduction
          </small>
        </div>

        <div>
          <span>
            LEGITIMATE POPULATIONS
          </span>

          <strong>0%</strong>

          <small>
            intervention
          </small>
        </div>

        <button
          className="proof-link"
          onClick={() =>
            navigate("evaluation")
          }
        >
          OPEN EVALUATION
          <ArrowRight size={15} />
        </button>
      </div>
    </section>
  );
}


/* ============================================================================
   STORY CARD
============================================================================ */

function StoryCard({
  number,
  icon: Icon,
  title,
  subtitle,
  text,
  highlighted,
  governor,
}) {
  return (
    <div
      className={`story-card ${
        highlighted
          ? "highlighted"
          : ""
      } ${
        governor
          ? "governor"
          : ""
      }`}
    >
      <div className="story-top">
        <span className="story-number">
          {number}
        </span>

        <div className="story-icon">
          <Icon size={18} />
        </div>
      </div>

      <div className="story-title">
        {title}
      </div>

      <div className="story-subtitle">
        {subtitle}
      </div>

      <p>{text}</p>

      <div className="story-signal">
        <span />

        {governor
          ? "SEES PRIVATE RISK SIGNALS"
          : highlighted
          ? "LEARNS FROM OUTCOMES"
          : "CASE-LEVEL VIEW"}
      </div>
    </div>
  );
}


/* ============================================================================
   INFORMATION BOUNDARY
============================================================================ */

function InformationBoundary() {
  return (
    <div className="information-boundary">
      <div className="boundary-side agent-view">
        <div className="boundary-title">
          <Eye size={15} />
          AGENT A CAN SEE
        </div>

        <div className="boundary-items">
          <span>Claim</span>
          <span>Evidence</span>
          <span>Customer history</span>
          <span>Amount</span>
          <span>Case context</span>
        </div>
      </div>

      <div className="boundary-core">
        <div className="boundary-lock">
          <Lock size={16} />
        </div>

        <span>
          INFORMATION BOUNDARY
        </span>

        <strong>
          Agent A does not receive
          <br />
          Governor-private signals.
        </strong>
      </div>

      <div className="boundary-side governor-view">
        <div className="boundary-title">
          <Shield size={15} />
          GOVERNOR CAN SEE
        </div>

        <div className="boundary-items">
          <span>Network behavior</span>
          <span>Strategic behavior</span>
          <span>Temporal patterns</span>
          <span>Semantic shifts</span>
          <span>GNN risk</span>
        </div>
      </div>
    </div>
  );
}


/* ============================================================================
   COMMAND CENTER
============================================================================ */

function CommandCenter({
  navigate,
  dashboard,
  evaluation,
}) {
  const result =
    evaluation?.result ||
    evaluation ||
    null;

  const populations =
    result?.population_metrics ||
    result?.population_results ||
    result?.populations ||
    {};

  const adaptive = findPopulation(
    populations,
    "ADAPTIVE_ABUSIVE"
  );

  const gnn = getMetric(
    adaptive,
    "mean_gnn_risk",
    "gnn_mean",
    "mean_gnn"
  );

  /*
   * Use the global resolver.
   * The old code called prevented(), which was scoped
   * inside EvaluationPage and therefore crashed this page.
   */

  const preventedValue =
    getPreventedValue(
      adaptive,
      result,
      "adaptive_abusive"
    );

  return (
    <div className="page command-page">
      <PageIntro
        eyebrow="COMMAND CENTER"
        title="Observe the attack surface."
        description="A single control plane for adaptive behavior, risk intelligence and economic outcomes."
      />

      <div className="command-status-row">
        <div className="status-large">
          <span className="status-dot" />
          GOVERNOR ONLINE
        </div>

        <div className="status-meta">
          <span>MODEL</span>
          <strong>
            GRAPH RISK GOVERNOR
          </strong>
        </div>

        <div className="status-meta">
          <span>MODE</span>
          <strong>
            CLOSED LOOP
          </strong>
        </div>

        <div className="status-meta">
          <span>AGENT ISOLATION</span>
          <strong>
            ENFORCED
          </strong>
        </div>
      </div>

      <div className="command-hero-grid">
        <div className="loop-panel">
          <PanelHeading
            kicker="ADAPTIVE LOOP"
            title="The adversary changes. The defense does not expose its logic."
          />

          <AdaptiveLoop />

          <div className="panel-footnote">
            <span>
              <span className="signal-dot" />
              OBSERVABLE OUTCOME
            </span>

            <span>
              AGENT B UPDATES POLICY
            </span>

            <span>
              GOVERNOR REMAINS INDEPENDENT
            </span>
          </div>
        </div>

        <div className="command-side">
          <CommandMetric
            icon={ShieldAlert}
            label="ABUSIVE VALUE PREVENTED"
            value={
              preventedValue === null
                ? "—"
                : money(preventedValue)
            }
            detail="Adaptive abusive population"
          />

          <CommandMetric
            icon={Network}
            label="MEAN GNN RISK"
            value={
              gnn === null
                ? "—"
                : decimal(gnn)
            }
            detail="Adaptive abusive population"
          />

          <CommandMetric
            icon={Target}
            label="GOVERNOR ACTIONS"
            value="3"
            detail="Allow · Evidence · Human review"
          />
        </div>
      </div>

      <div className="dashboard-section">
        <div className="section-header-row">
          <div>
            <div className="section-kicker">
              SYSTEM MAP
            </div>

            <h3>
              From claim to economic consequence
            </h3>
          </div>

          <button
            className="text-button"
            onClick={() =>
              navigate("governor")
            }
          >
            INSPECT GOVERNOR
            <ArrowRight size={14} />
          </button>
        </div>

        <SystemPipeline />
      </div>

      <div className="dashboard-grid">
        <RiskSignalPanel
          evaluation={evaluation}
        />

        <LiveInterventionPanel />
      </div>
    </div>
  );
}


/* ============================================================================
   ADAPTIVE LOOP
============================================================================ */

function AdaptiveLoop() {
  const [ref, visible] =
    useInView();

  const [cycle, setCycle] =
    useState(0);

  useEffect(() => {
    if (!visible) return;

    const interval =
      window.setInterval(() => {
        setCycle(
          (value) =>
            (value + 1) % 4
        );
      }, 2300);

    return () =>
      window.clearInterval(
        interval
      );
  }, [visible]);

  return (
    <div
      ref={ref}
      className={`adaptive-loop ${
        visible
          ? "is-visible"
          : ""
      }`}
    >
      <div className="loop-grid" />

      <div className="loop-agent agent-a">
        <div className="loop-agent-icon">
          <Bot size={19} />
        </div>

        <span>AGENT A</span>
        <small>
          DECISION POLICY
        </small>
      </div>

      <div className="loop-agent agent-b">
        <div className="loop-agent-icon">
          <BrainCircuit size={19} />
        </div>

        <span>AGENT B</span>
        <small>
          ADAPTIVE POLICY
        </small>
      </div>

      <div className="loop-governor">
        <div className="governor-core-icon">
          <Shield size={21} />
        </div>

        <span>GOVERNOR</span>
        <small>
          PRIVATE RISK LAYER
        </small>
      </div>

      <div className="loop-path path-a" />
      <div className="loop-path path-b" />
      <div className="loop-path path-c" />

      <div
        className={`loop-packet packet-a p${cycle}`}
      />

      <div
        className={`loop-packet packet-b p${cycle}`}
      />

      <div className="loop-readout">
        <div>
          <span>INTERACTION</span>

          <strong>
            {String(cycle + 1).padStart(
              2,
              "0"
            )}
          </strong>
        </div>

        <div>
          <span>
            POLICY RESPONSE
          </span>

          <strong>
            {cycle === 0
              ? "OBSERVE"
              : cycle === 1
              ? "PROBE"
              : cycle === 2
              ? "ADAPT"
              : "REPEAT"}
          </strong>
        </div>

        <div>
          <span>GOVERNOR</span>

          <strong className="positive">
            INDEPENDENT
          </strong>
        </div>
      </div>
    </div>
  );
}


/* ============================================================================
   COMMAND METRIC
============================================================================ */

function CommandMetric({
  icon: Icon,
  label,
  value,
  detail,
}) {
  return (
    <div className="command-metric">
      <div className="command-metric-icon">
        <Icon size={17} />
      </div>

      <div className="command-metric-label">
        {label}
      </div>

      <div className="command-metric-value">
        {value}
      </div>

      <div className="command-metric-detail">
        {detail}
      </div>
    </div>
  );
}


/* ============================================================================
   SYSTEM PIPELINE
============================================================================ */

function SystemPipeline() {
  const stages = [
    {
      icon: WalletCards,
      title: "CLAIM",
      value: "REFUND REQUEST",
    },
    {
      icon: Fingerprint,
      title: "EVIDENCE",
      value: "CONSISTENCY",
    },
    {
      icon: BrainCircuit,
      title: "AGENT A",
      value: "DECISION",
    },
    {
      icon: Network,
      title: "INTELLIGENCE",
      value: "SIGNAL FUSION",
    },
    {
      icon: Shield,
      title: "GOVERNOR",
      value: "RISK ACTION",
      active: true,
    },
    {
      icon: WalletCards,
      title: "OUTCOME",
      value: "ECONOMIC",
    },
    {
      icon: RefreshCw,
      title: "FEEDBACK",
      value: "ADAPTATION",
    },
  ];

  return (
    <div className="system-pipeline">
      {stages.map(
        (stage, index) => {
          const Icon = stage.icon;

          return (
            <React.Fragment
              key={stage.title}
            >
              <div
                className={`pipeline-stage ${
                  stage.active
                    ? "active"
                    : ""
                }`}
              >
                <div className="pipeline-icon">
                  <Icon size={16} />
                </div>

                <span>
                  {stage.title}
                </span>

                <strong>
                  {stage.value}
                </strong>
              </div>

              {index !==
                stages.length - 1 && (
                <div className="pipeline-connector">
                  <ArrowRight
                    size={13}
                  />
                </div>
              )}
            </React.Fragment>
          );
        }
      )}
    </div>
  );
}


/* ============================================================================
   RISK SIGNAL PANEL
============================================================================ */

function RiskSignalPanel({
  evaluation,
}) {
  const adaptive =
    findPopulation(
      evaluation?.result
        ?.population_metrics ||
        evaluation
          ?.population_metrics,
      "ADAPTIVE_ABUSIVE"
    );

  const values = [
    [
      "GNN RISK",
      getMetric(
        adaptive,
        "mean_gnn_risk",
        "gnn_mean",
        "mean_gnn"
      ),
      "gnn",
    ],
    [
      "FUSED RISK",
      getMetric(
        adaptive,
        "mean_fused_risk",
        "fused_mean",
        "mean_fused"
      ),
      "fused",
    ],
    [
      "INTERVENTION",
      getMetric(
        adaptive,
        "intervention_rate"
      ),
      "intervention",
    ],
    [
      "POLICY CHANGE",
      getMetric(
        adaptive,
        "mean_policy_change",
        "policy_change"
      ),
      "policy",
    ],
  ];

  return (
    <div className="dark-panel signal-panel">
      <PanelHeading
        kicker="RISK SIGNALS"
        title="Independent signals converge before action."
      />

      <div className="signal-bars">
        {values.map(
          ([label, value, key]) => (
            <div
              className="signal-bar-row"
              key={key}
            >
              <div className="signal-bar-meta">
                <span>{label}</span>

                <strong>
                  {value === null
                    ? "—"
                    : key ===
                        "intervention" ||
                      key === "policy"
                    ? pct(value)
                    : decimal(
                        value,
                        3
                      )}
                </strong>
              </div>

              <div className="signal-bar-track">
                <span
                  style={{
                    width: `${
                      clamp(value) *
                      100
                    }%`,
                  }}
                />
              </div>
            </div>
          )
        )}
      </div>

      <div className="signal-panel-note">
        <ShieldCheck size={14} />

        Signal values shown here come
        from the Phase 18 evaluation
        when available.
      </div>
    </div>
  );
}


/* ============================================================================
   LIVE INTERVENTION
============================================================================ */

function LiveInterventionPanel() {
  return (
    <div className="dark-panel intervention-panel">
      <PanelHeading
        kicker="INTERVENTION LOGIC"
        title="The Governor controls what happens next."
      />

      <div className="intervention-list">
        <InterventionRow
          level="LOW"
          action="ALLOW AGENT A DECISION"
          text="Autonomous workflow proceeds."
          icon={Check}
        />

        <InterventionRow
          level="MEDIUM"
          action="REQUEST ADDITIONAL EVIDENCE"
          text="Require stronger evidence before proceeding."
          icon={Search}
        />

        <InterventionRow
          level="HIGH"
          action="ESCALATE TO HUMAN REVIEW"
          text="Autonomous authority stops at the control boundary."
          icon={ShieldAlert}
          active
        />
      </div>

      <div className="intervention-rule">
        <Lock size={13} />

        <span>
          AGENT A NEVER RECEIVES GOVERNOR-INTERNAL RISK FEATURES
        </span>
      </div>
    </div>
  );
}


/* ============================================================================
   INTERVENTION ROW
============================================================================ */

function InterventionRow({
  level,
  action,
  text,
  icon: Icon,
  active,
}) {
  return (
    <div
      className={`intervention-row ${
        active
          ? "active"
          : ""
      }`}
    >
      <div className="intervention-level">
        {level}
      </div>

      <div className="intervention-icon">
        <Icon size={14} />
      </div>

      <div className="intervention-copy">
        <strong>{action}</strong>
        <span>{text}</span>
      </div>

      <ArrowRight size={13} />
    </div>
  );
}


/* ============================================================================
   INVESTIGATION
============================================================================ */

function InvestigationPage() {
  return (
    <div className="page">
      <PageIntro
        eyebrow="CASE INVESTIGATION"
        title="Inspect the decision before it becomes loss."
        description="A representative investigation view showing how a single refund request is decomposed into observable and private risk signals."
        badge="REPRESENTATIVE CASE VISUALIZATION"
      />

      <div className="investigation-layout">
        <div className="case-main dark-panel">
          <div className="case-header">
            <div>
              <div className="section-kicker">
                CASE RG-2026-00421
              </div>

              <h3>
                PRODUCT_NOT_RECEIVED
              </h3>
            </div>

            <div className="case-risk">
              <span>FUSED RISK</span>
              <strong>0.69</strong>
              <em>HIGH</em>
            </div>
          </div>

          <div className="case-meta-grid">
            <CaseField
              label="CUSTOMER"
              value="CUS-084291"
            />

            <CaseField
              label="ORDER"
              value="ORD-729410"
            />

            <CaseField
              label="AMOUNT"
              value="₹750"
            />

            <CaseField
              label="SUPPORT DECISION"
              value="APPROVE"
            />
          </div>

          <div className="case-section">
            <div className="case-section-title">
              <Activity size={14} />
              OBSERVABLE CASE SIGNALS
            </div>

            <RiskSignalCards />
          </div>
        </div>

        <div className="investigation-side">
          <DecisionTrace />
          <InformationIsolationCard />
        </div>
      </div>
    </div>
  );
}


/* ============================================================================
   CASE FIELD
============================================================================ */

function CaseField({
  label,
  value,
}) {
  return (
    <div className="case-field">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}


/* ============================================================================
   RISK SIGNAL CARDS
============================================================================ */

function RiskSignalCards() {
  const signals = [
    [
      "Governor risk",
      "0.81",
      "HIGH",
    ],
    [
      "Temporal abnormality",
      "0.73",
      "HIGH",
    ],
    [
      "Evidence consistency",
      "0.69",
      "MEDIUM",
    ],
    [
      "Request velocity",
      "0.77",
      "HIGH",
    ],
    [
      "Semantic switching",
      "0.84",
      "HIGH",
    ],
    [
      "Amount acceleration",
      "0.62",
      "MEDIUM",
    ],
  ];

  return (
    <div className="case-signal-grid">
      {signals.map(
        ([label, value, level]) => (
          <div
            className="case-signal"
            key={label}
          >
            <div className="case-signal-top">
              <span>{label}</span>

              <span
                className={`risk-pill ${level.toLowerCase()}`}
              >
                {level}
              </span>
            </div>

            <strong>{value}</strong>

            <div className="case-signal-track">
              <span
                style={{
                  width: `${
                    Number(value) *
                    100
                  }%`,
                }}
              />
            </div>
          </div>
        )
      )}
    </div>
  );
}


/* ============================================================================
   DECISION TRACE
============================================================================ */

function DecisionTrace() {
  const rows = [
    [
      "01",
      "Claim received",
      "OBSERVED",
    ],
    [
      "02",
      "Evidence evaluated",
      "OBSERVED",
    ],
    [
      "03",
      "Agent A proposes approval",
      "OBSERVED",
    ],
    [
      "04",
      "Governor evaluates private signals",
      "INTERCEPT",
    ],
    [
      "05",
      "Risk action selected",
      "ESCALATE",
    ],
  ];

  return (
    <div className="dark-panel trace-panel">
      <PanelHeading
        kicker="DECISION TRACE"
        title="Intervention occurs before the economic outcome."
      />

      <div className="trace-list">
        {rows.map(
          ([number, text, status]) => (
            <div
              className="trace-row"
              key={number}
            >
              <span className="trace-number">
                {number}
              </span>

              <div>
                <strong>{text}</strong>
                <span>{status}</span>
              </div>

              {status ===
              "INTERCEPT" ? (
                <ShieldAlert
                  size={14}
                />
              ) : (
                <Check size={14} />
              )}
            </div>
          )
        )}
      </div>
    </div>
  );
}


/* ============================================================================
   INFORMATION ISOLATION
============================================================================ */

function InformationIsolationCard() {
  return (
    <div className="isolation-card">
      <div className="isolation-icon">
        <Lock size={17} />
      </div>

      <div>
        <span>
          FORMAL INFORMATION BOUNDARY
        </span>

        <strong>
          Agent A cannot access GNN,
          network, strategic or fused
          Governor signals.
        </strong>
      </div>
    </div>
  );
}


/* ============================================================================
   EVIDENCE
============================================================================ */

function EvidencePage() {
  return (
    <div className="page">
      <PageIntro
        eyebrow="EVIDENCE INTELLIGENCE"
        title="Turn fragmented evidence into a decision package."
        description="The evidence layer organizes claim history, consistency and semantic changes before the Governor evaluates the case."
        badge="REPRESENTATIVE CASE VISUALIZATION"
      />

      <div className="evidence-layout">
        <div className="dark-panel evidence-package">
          <div className="package-header">
            <div>
              <div className="section-kicker">
                EVIDENCE PACKAGE
              </div>

              <h3>
                CASE RG-2026-00421
              </h3>
            </div>

            <div className="package-confidence">
              <span>
                PACKAGE CONFIDENCE
              </span>

              <strong>82%</strong>
            </div>
          </div>

          <div className="confidence-line">
            <span />
          </div>

          <div className="evidence-events">
            <EvidenceEvent
              icon={WalletCards}
              title="Current claim"
              text="Product reported as not received."
              status="CONSISTENT"
            />

            <EvidenceEvent
              icon={Timer}
              title="Temporal pattern"
              text="Request timing differs from expected customer behavior."
              status="FLAGGED"
            />

            <EvidenceEvent
              icon={GitBranch}
              title="Semantic comparison"
              text="Claim wording shifts across related requests."
              status="FLAGGED"
            />

            <EvidenceEvent
              icon={Database}
              title="Historical evidence"
              text="Previous interaction history available for comparison."
              status="AVAILABLE"
            />
          </div>
        </div>

        <div className="evidence-side">
          <EvidenceSourceMap />
          <EvidenceDisclaimer />
        </div>
      </div>
    </div>
  );
}


/* ============================================================================
   EVIDENCE EVENT
============================================================================ */

function EvidenceEvent({
  icon: Icon,
  title,
  text,
  status,
}) {
  return (
    <div className="evidence-event">
      <div className="evidence-event-icon">
        <Icon size={15} />
      </div>

      <div className="evidence-event-copy">
        <strong>{title}</strong>
        <span>{text}</span>
      </div>

      <div className="evidence-event-status">
        {status}
      </div>
    </div>
  );
}


/* ============================================================================
   EVIDENCE SOURCE MAP
============================================================================ */

function EvidenceSourceMap() {
  return (
    <div className="dark-panel source-map">
      <PanelHeading
        kicker="SOURCE MAP"
        title="Evidence provenance"
      />

      <div className="source-map-visual">
        <div className="source-line source-line-1" />
        <div className="source-line source-line-2" />
        <div className="source-line source-line-3" />

        <div className="source-node center">
          <Fingerprint size={17} />
          <span>CASE</span>
        </div>

        <div className="source-node top">
          <Database size={14} />
          <span>HISTORY</span>
        </div>

        <div className="source-node left">
          <Globe2 size={14} />
          <span>CONTEXT</span>
        </div>

        <div className="source-node right">
          <GitBranch size={14} />
          <span>SEMANTIC</span>
        </div>
      </div>
    </div>
  );
}


/* ============================================================================
   EVIDENCE DISCLAIMER
============================================================================ */

function EvidenceDisclaimer() {
  return (
    <div className="representative-note">
      <Sparkles size={14} />

      <div>
        <strong>UI STATUS</strong>

        <span>
          Evidence values in this view are
          representative visualization data,
          not additional Phase 18 measurements.
        </span>
      </div>
    </div>
  );
}


/* ============================================================================
   GOVERNOR
============================================================================ */

function GovernorPage({
  evaluation,
}) {
  const adaptive =
    findPopulation(
      evaluation?.result
        ?.population_metrics ||
        evaluation
          ?.population_metrics,
      "ADAPTIVE_ABUSIVE"
    );

  const gnn = getMetric(
    adaptive,
    "mean_gnn_risk",
    "gnn_mean",
    "mean_gnn"
  );

  const fused = getMetric(
    adaptive,
    "mean_fused_risk",
    "fused_mean",
    "mean_fused"
  );

  return (
    <div className="page governor-page">
      <PageIntro
        eyebrow="RISK GOVERNOR"
        title="Private intelligence. Controlled action."
        description="The Governor sits between the support agent and the economic outcome. It does not replace Agent A — it constrains what Agent A is allowed to do."
      />

      <div className="governor-hero">
        <div className="governor-engine dark-panel">
          <SignalThreads dense />

          <div className="governor-engine-content">
            <div className="engine-header">
              <div>
                <div className="section-kicker">
                  GRAPH RISK ENGINE
                </div>

                <h3>
                  Signal convergence
                </h3>
              </div>

              <div className="engine-status">
                <span />
                ACTIVE
              </div>
            </div>

            <GovernorMesh
              gnn={gnn}
              fused={fused}
            />
          </div>
        </div>

        <div className="governor-readout">
          <GovernorReadout
            label="GNN RISK"
            value={
              gnn === null
                ? "—"
                : decimal(gnn, 3)
            }
            detail="Adaptive abusive population"
          />

          <GovernorReadout
            label="FUSED RISK"
            value={
              fused === null
                ? "—"
                : decimal(fused, 3)
            }
            detail="Base + GNN fusion"
          />

          <GovernorReadout
            label="RISK LEVEL"
            value={
              fused === null
                ? "—"
                : riskLevel(fused)
            }
            detail="Governor action threshold"
          />

          <GovernorReadout
            label="ACTION SPACE"
            value="03"
            detail="Allow / Evidence / Human"
          />
        </div>
      </div>

      <div className="governor-bottom-grid">
        <GovernorArchitecture />
        <GovernorIntegrity />
      </div>
    </div>
  );
}


/* ============================================================================
   GOVERNOR MESH
============================================================================ */

function GovernorMesh({
  gnn,
  fused,
}) {
  const signals = [
    {
      label: "BEHAVIORAL",
      value: 0.78,
      x: 16,
      y: 25,
    },
    {
      label: "NETWORK",
      value: 0.86,
      x: 16,
      y: 50,
    },
    {
      label: "TEMPORAL",
      value: 0.71,
      x: 16,
      y: 75,
    },
    {
      label: "SEMANTIC",
      value: 0.82,
      x: 84,
      y: 25,
    },
    {
      label: "STRATEGIC",
      value: 0.88,
      x: 84,
      y: 50,
    },
    {
      label: "EVIDENCE",
      value: 0.64,
      x: 84,
      y: 75,
    },
  ];

  return (
    <div className="governor-mesh">
      <div className="mesh-lines">
        {signals.map(
          (signal) => (
            <div
              key={signal.label}
              className="mesh-line"
              style={{
                left: `${signal.x}%`,
                top: `${signal.y}%`,
                transform:
                  signal.x < 50
                    ? "rotate(0deg)"
                    : "rotate(180deg)",
              }}
            />
          )
        )}
      </div>

      {signals.map(
        (signal) => (
          <div
            key={signal.label}
            className="mesh-signal"
            style={{
              left: `${signal.x}%`,
              top: `${signal.y}%`,
            }}
          >
            <div
              className="mesh-signal-core"
              style={{
                "--signal-strength":
                  signal.value,
              }}
            />

            <span>
              {signal.label}
            </span>

            <strong>
              {signal.value.toFixed(
                2
              )}
            </strong>
          </div>
        )
      )}

      <div className="mesh-center">
        <div className="mesh-center-ring ring-a" />
        <div className="mesh-center-ring ring-b" />

        <div className="mesh-center-core">
          <Shield size={25} />

          <span>FUSED</span>

          <strong>
            {fused === null
              ? "—"
              : Number(fused).toFixed(
                  3
                )}
          </strong>
        </div>
      </div>

      <div className="mesh-bottom">
        <span>
          GNN{" "}
          <strong>
            {gnn === null
              ? "—"
              : Number(gnn).toFixed(
                  3
                )}
          </strong>
        </span>

        <span>
          ACTION{" "}
          <strong>
            {fused !== null &&
            Number(fused) >= 0.7
              ? "HUMAN REVIEW"
              : "CONTROLLED"}
          </strong>
        </span>
      </div>
    </div>
  );
}


/* ============================================================================
   GOVERNOR READOUT
============================================================================ */

function GovernorReadout({
  label,
  value,
  detail,
}) {
  return (
    <div className="governor-readout-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}


/* ============================================================================
   GOVERNOR ARCHITECTURE
============================================================================ */

function GovernorArchitecture() {
  return (
    <div className="dark-panel">
      <PanelHeading
        kicker="CONTROL ARCHITECTURE"
        title="Why the Governor remains defensive as Agent B adapts."
      />

      <div className="architecture-list">
        <ArchitectureRow
          icon={Users}
          title="Agent B observes outcomes"
          text="It can learn the behavior of the support decision agent."
        />

        <ArchitectureRow
          icon={Lock}
          title="Governor signals remain private"
          text="Agent A and the adaptive actor do not receive Governor-internal risk features."
        />

        <ArchitectureRow
          icon={ShieldCheck}
          title="Action is constrained"
          text="The Governor can allow, require evidence or escalate to human review."
        />

        <ArchitectureRow
          icon={RefreshCw}
          title="Closed loop is evaluated"
          text="The experiment measures what happens after the adversary changes its policy."
        />
      </div>
    </div>
  );
}


/* ============================================================================
   ARCHITECTURE ROW
============================================================================ */

function ArchitectureRow({
  icon: Icon,
  title,
  text,
}) {
  return (
    <div className="architecture-row">
      <div className="architecture-icon">
        <Icon size={15} />
      </div>

      <div>
        <strong>{title}</strong>
        <span>{text}</span>
      </div>
    </div>
  );
}


/* ============================================================================
   GOVERNOR INTEGRITY
============================================================================ */

function GovernorIntegrity() {
  return (
    <div className="dark-panel integrity-panel">
      <PanelHeading
        kicker="MODEL INTEGRITY"
        title="The control boundary is part of the design."
      />

      <div className="integrity-stat">
        <Lock size={17} />

        <div>
          <strong>
            INFORMATION ISOLATED
          </strong>

          <span>
            Agent A information boundary
            enforced
          </span>
        </div>
      </div>

      <div className="integrity-stat">
        <Cpu size={17} />

        <div>
          <strong>
            GRAPHSAGE GNN
          </strong>

          <span>
            Independent graph-based risk
            signal
          </span>
        </div>
      </div>

      <div className="integrity-stat">
        <ShieldCheck size={17} />

        <div>
          <strong>
            NO DENY ACTION
          </strong>

          <span>
            Denial remains in Agent A's
            decision space
          </span>
        </div>
      </div>
    </div>
  );
}


/* ============================================================================
   EVALUATION
============================================================================ */

function EvaluationPage({
  evaluation,
  loading,
  error,
  onRun,
}) {
  const result =
    evaluation?.result ??
    evaluation ??
    null;

  const rawPopulations =
    result?.population_metrics ??
    result?.population_results ??
    result?.populations ??
    {};

  const populationList =
    Array.isArray(rawPopulations)
      ? rawPopulations
      : Object.entries(
          rawPopulations
        ).map(
          ([key, value]) => ({
            ...(value || {}),
            population_group:
              value?.population_group ??
              value?.population ??
              value?.population_type ??
              value?.type ??
              key,
          })
        );

  function evaluationPopulation(
    name
  ) {
    const target = String(name)
      .trim()
      .toUpperCase();

    return (
      populationList.find(
        (item) => {
          const group = String(
            item?.population_group ??
            item?.population ??
            item?.population_type ??
            item?.type ??
            ""
          )
            .trim()
            .toUpperCase();

          return group === target;
        }
      ) ?? null
    );
  }

  const adaptive =
    evaluationPopulation(
      "ADAPTIVE_ABUSIVE"
    );

  const human =
    evaluationPopulation(
      "HUMAN_ABUSIVE"
    );

  const adaptiveLegitimate =
    evaluationPopulation(
      "ADAPTIVE_LEGITIMATE"
    );

  const humanLegitimate =
    evaluationPopulation(
      "HUMAN_LEGITIMATE"
    );

  function metric(
    population,
    ...keys
  ) {
    return getMetric(
      population,
      ...keys
    );
  }

  function baselineApproval(
    population,
    prefix
  ) {
    return (
      metric(
        population,
        "baseline_approval",
        "baseline_approval_rate",
        "baseline_approval_fraction"
      ) ??
      getMetric(
        result,
        `${prefix}_baseline_approval`,
        `${prefix}_baseline_approval_rate`
      )
    );
  }

  function governedApproval(
    population,
    prefix
  ) {
    return (
      metric(
        population,
        "governed_approval",
        "governed_approval_rate",
        "governed_approval_fraction"
      ) ??
      getMetric(
        result,
        `${prefix}_governed_approval`,
        `${prefix}_governed_approval_rate`
      )
    );
  }

  function reduction(
    population,
    prefix
  ) {
    const direct = metric(
      population,
      "approval_reduction",
      "reduction",
      "approval_reduction_rate"
    );

    if (direct !== null) {
      return Number(direct);
    }

    const baseline =
      baselineApproval(
        population,
        prefix
      );

    const governed =
      governedApproval(
        population,
        prefix
      );

    if (
      baseline !== null &&
      governed !== null
    ) {
      return (
        Number(baseline) -
        Number(governed)
      );
    }

    return null;
  }

  function intervention(
    population,
    prefix
  ) {
    return (
      metric(
        population,
        "intervention_rate",
        "governor_intervention_rate",
        "intervention_fraction"
      ) ??
      getMetric(
        result,
        `${prefix}_intervention_rate`,
        `${prefix}_governor_intervention_rate`
      )
    );
  }

  function prevented(
    population,
    prefix
  ) {
    return getPreventedValue(
      population,
      result,
      prefix
    );
  }

  function episodeCount(
    population
  ) {
    return metric(
      population,
      "episodes",
      "episode_count",
      "n_episodes"
    );
  }

  const adaptiveReduction =
    reduction(
      adaptive,
      "adaptive_abusive"
    );

  const humanReduction =
    reduction(
      human,
      "human_abusive"
    );

  const adaptiveLegitimateReduction =
    reduction(
      adaptiveLegitimate,
      "adaptive_legitimate"
    );

  const humanLegitimateReduction =
    reduction(
      humanLegitimate,
      "human_legitimate"
    );

  const adaptivePrevented =
    prevented(
      adaptive,
      "adaptive_abusive"
    );

  const humanPrevented =
    prevented(
      human,
      "human_abusive"
    );

  const combinedPrevented =
    adaptivePrevented !== null ||
    humanPrevented !== null
      ? Number(
          adaptivePrevented ?? 0
        ) +
        Number(
          humanPrevented ?? 0
        )
      : null;

  const globalEpisodes =
    getMetric(
      result?.global_metrics,
      "total_episodes",
      "episodes",
      "episode_count"
    ) ??
    getMetric(
      result,
      "total_episodes",
      "episodes",
      "episode_count"
    );

  const calculatedEpisodes =
    populationList.reduce(
      (total, population) =>
        total +
        Number(
          episodeCount(
            population
          ) ?? 0
        ),
      0
    );

  const episodes =
    globalEpisodes ??
    (calculatedEpisodes > 0
      ? calculatedEpisodes
      : null);

  const adaptiveLegitimateIntervention =
    intervention(
      adaptiveLegitimate,
      "adaptive_legitimate"
    );

  const humanLegitimateIntervention =
    intervention(
      humanLegitimate,
      "human_legitimate"
    );

  const legitimateInterventionValues =
    [
      adaptiveLegitimateIntervention,
      humanLegitimateIntervention,
    ].filter(
      (value) =>
        value !== null
    );

  const legitimateIntervention =
    legitimateInterventionValues.length >
    0
      ? Math.max(
          ...legitimateInterventionValues
        )
      : null;

  if (loading && !result) {
    return (
      <div className="page evaluation-page">
        <PageIntro
          eyebrow="EVALUATION LAB"
          title="Does the defense survive adaptation?"
          description="Running the canonical Phase 18 closed-loop evaluation."
        />

        <div className="evaluation-control">
          <div>
            <span>
              CANONICAL EVALUATION
            </span>

            <strong>
              PHASE 18 · CLOSED LOOP · SEED 42
            </strong>
          </div>

          <button
            className="primary-button"
            disabled
          >
            <RefreshCw
              size={15}
              className="spin"
            />

            RUNNING
          </button>
        </div>

        <div className="proof-hero">
          <div className="proof-primary">
            <span>
              COMBINED ABUSIVE VALUE PREVENTED
            </span>

            <strong>
              CALCULATING
            </strong>

            <small>
              Waiting for Phase 18
              measurements
            </small>
          </div>

          <div className="proof-secondary">
            <span>EPISODES</span>
            <strong>—</strong>
          </div>

          <div className="proof-secondary">
            <span>
              LEGITIMATE INTERVENTION
            </span>

            <strong>—</strong>
          </div>
        </div>
      </div>
    );
  }

  if (error && !result) {
    return (
      <div className="page evaluation-page">
        <PageIntro
          eyebrow="EVALUATION LAB"
          title="Does the defense survive adaptation?"
          description="The Phase 18 evaluation could not be completed."
        />

        <div className="evaluation-error">
          <AlertTriangle size={15} />
          <span>{error}</span>
        </div>

        <div className="evaluation-control">
          <div>
            <span>
              CANONICAL EVALUATION
            </span>

            <strong>
              PHASE 18 · CLOSED LOOP · SEED 42
            </strong>
          </div>

          <button
            className="primary-button"
            onClick={onRun}
            disabled={loading}
          >
            <RefreshCw
              size={15}
              className={
                loading
                  ? "spin"
                  : ""
              }
            />

            {loading
              ? "RUNNING"
              : "RUN EVALUATION"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page evaluation-page">
      <PageIntro
        eyebrow="EVALUATION LAB"
        title="Does the defense survive adaptation?"
        description="The evaluation compares the adaptive adversary against human abusive behavior and measures whether the Governor reduces abuse without intervening on legitimate traffic."
      />

      <div className="evaluation-control">
        <div>
          <span>
            CANONICAL EVALUATION
          </span>

          <strong>
            PHASE 18 · CLOSED LOOP · SEED 42
          </strong>
        </div>

        <button
          className="primary-button"
          onClick={onRun}
          disabled={loading}
        >
          {loading ? (
            <>
              <RefreshCw
                size={15}
                className="spin"
              />
              RUNNING
            </>
          ) : (
            <>
              <Zap size={15} />
              RUN EVALUATION
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="evaluation-error">
          <AlertTriangle size={15} />
          <span>{error}</span>
        </div>
      )}

      <div className="proof-hero">
        <div className="proof-primary">
          <span>
            COMBINED ABUSIVE VALUE PREVENTED
          </span>

          <strong>
            {combinedPrevented === null
              ? "—"
              : money(
                  combinedPrevented
                )}
          </strong>

          <small>
            Adaptive abusive + human abusive
            populations
          </small>
        </div>

        <div className="proof-secondary">
          <span>EPISODES</span>

          <strong>
            {episodes === null
              ? "—"
              : episodes}
          </strong>
        </div>

        <div className="proof-secondary">
          <span>
            LEGITIMATE INTERVENTION
          </span>

          <strong>
            {legitimateIntervention ===
            null
              ? "—"
              : pct(
                  legitimateIntervention,
                  1
                )}
          </strong>
        </div>
      </div>

      <div className="comparison-title">
        <div>
          <div className="section-kicker">
            POPULATION COMPARISON
          </div>

          <h3>
            Adaptive abuse vs human abuse
          </h3>
        </div>

        <span>
          Same Governor · Different adversary
        </span>
      </div>

      <div className="comparison-grid">
        <PopulationCard
          title="ADAPTIVE ABUSIVE"
          subtitle="Agent B learns from outcomes"
          icon={BrainCircuit}
          reduction={adaptiveReduction}
          prevented={adaptivePrevented}
          population={adaptive}
          highlighted
        />

        <PopulationCard
          title="HUMAN ABUSIVE"
          subtitle="Non-adaptive abusive behavior"
          icon={UserRound}
          reduction={humanReduction}
          prevented={humanPrevented}
          population={human}
        />

        <PopulationCard
          title="ADAPTIVE LEGITIMATE"
          subtitle="Legitimate behavior under adaptation"
          icon={Users}
          reduction={
            adaptiveLegitimateReduction
          }
          prevented={prevented(
            adaptiveLegitimate,
            "adaptive_legitimate"
          )}
          population={
            adaptiveLegitimate
          }
          legitimate
        />

        <PopulationCard
          title="HUMAN LEGITIMATE"
          subtitle="Legitimate human behavior"
          icon={UserRound}
          reduction={
            humanLegitimateReduction
          }
          prevented={prevented(
            humanLegitimate,
            "human_legitimate"
          )}
          population={
            humanLegitimate
          }
          legitimate
        />
      </div>

      <div className="evaluation-bottom">
        <EvaluationBars
          adaptive={adaptive}
          human={human}
        />

        <EvaluationIntegrity />
      </div>
    </div>
  );
}


/* ============================================================================
   POPULATION CARD
============================================================================ */

function PopulationCard({
  title,
  subtitle,
  icon: Icon,
  reduction,
  prevented,
  population,
  highlighted,
  legitimate,
}) {
  const baseline = getMetric(
    population,
    "baseline_approval",
    "baseline_approval_rate",
    "baseline_approval_fraction"
  );

  const governed = getMetric(
    population,
    "governed_approval",
    "governed_approval_rate",
    "governed_approval_fraction"
  );

  const intervention =
    getMetric(
      population,
      "intervention_rate",
      "governor_intervention_rate",
      "intervention_fraction"
    );

  const episodes = getMetric(
    population,
    "episodes",
    "episode_count",
    "n_episodes"
  );

  const meanGnn = getMetric(
    population,
    "mean_gnn_risk",
    "average_gnn_risk",
    "gnn_mean",
    "mean_gnn"
  );

  const meanFused = getMetric(
    population,
    "mean_fused_risk",
    "average_fused_risk",
    "fused_mean",
    "mean_fused"
  );

  return (
    <div
      className={`population-card ${
        highlighted
          ? "highlighted"
          : ""
      } ${
        legitimate
          ? "legitimate"
          : ""
      }`}
    >
      <div className="population-card-top">
        <div className="population-icon">
          <Icon size={17} />
        </div>

        <span>
          {legitimate
            ? "CONTROL CHECK"
            : highlighted
            ? "ADAPTIVE"
            : "BASELINE"}
        </span>
      </div>

      <h4>{title}</h4>

      <p>{subtitle}</p>

      <div className="population-main">
        <span>
          APPROVAL REDUCTION
        </span>

        <strong>
          {reduction === null
            ? "—"
            : pp(
                reduction,
                2
              )}
        </strong>
      </div>

      <div className="population-row">
        <span>
          Baseline approval
        </span>

        <strong>
          {baseline === null
            ? "—"
            : pct(
                baseline,
                2
              )}
        </strong>
      </div>

      <div className="population-row">
        <span>
          Governed approval
        </span>

        <strong>
          {governed === null
            ? "—"
            : pct(
                governed,
                2
              )}
        </strong>
      </div>

      <div className="population-row">
        <span>
          Intervention
        </span>

        <strong>
          {intervention === null
            ? "—"
            : pct(
                intervention,
                1
              )}
        </strong>
      </div>

      <div className="population-row">
        <span>Episodes</span>

        <strong>
          {episodes === null
            ? "—"
            : episodes}
        </strong>
      </div>

      {!legitimate && (
        <>
          <div className="population-row">
            <span>
              Mean GNN risk
            </span>

            <strong>
              {meanGnn === null
                ? "—"
                : decimal(
                    meanGnn,
                    3
                  )}
            </strong>
          </div>

          <div className="population-row">
            <span>
              Mean fused risk
            </span>

            <strong>
              {meanFused === null
                ? "—"
                : decimal(
                    meanFused,
                    3
                  )}
            </strong>
          </div>
        </>
      )}

      <div className="population-prevented">
        <span>
          VALUE PREVENTED
        </span>

        <strong>
          {prevented === null
            ? "—"
            : money(prevented)}
        </strong>
      </div>
    </div>
  );
}


/* ============================================================================
   EVALUATION BARS
============================================================================ */

function EvaluationBars({
  adaptive,
  human,
}) {
  function calculateReduction(
    population
  ) {
    if (!population) {
      return null;
    }

    const direct = getMetric(
      population,
      "approval_reduction",
      "reduction",
      "approval_reduction_rate"
    );

    if (direct !== null) {
      return Number(direct);
    }

    const baseline = getMetric(
      population,
      "baseline_approval",
      "baseline_approval_rate",
      "baseline_approval_fraction"
    );

    const governed = getMetric(
      population,
      "governed_approval",
      "governed_approval_rate",
      "governed_approval_fraction"
    );

    if (
      baseline !== null &&
      governed !== null
    ) {
      return (
        Number(baseline) -
        Number(governed)
      );
    }

    return null;
  }

  const adaptiveReduction =
    calculateReduction(
      adaptive
    );

  const humanReduction =
    calculateReduction(
      human
    );

  return (
    <div className="dark-panel evaluation-bars">
      <PanelHeading
        kicker="APPROVAL CONTAINMENT"
        title="Measured reduction in abusive approvals"
      />

      <ComparisonBar
        label="Adaptive abusive"
        value={adaptiveReduction}
        emphasized
      />

      <ComparisonBar
        label="Human abusive"
        value={humanReduction}
      />

      <div className="comparison-footnote">
        The adaptive population is the
        harder test: the adversary changes
        policy in response to observed
        outcomes.
      </div>
    </div>
  );
}


/* ============================================================================
   COMPARISON BAR
============================================================================ */

function ComparisonBar({
  label,
  value,
  emphasized,
}) {
  return (
    <div className="comparison-bar">
      <div>
        <span>{label}</span>

        <strong>
          {value === null
            ? "—"
            : pp(
                value,
                2
              )}
        </strong>
      </div>

      <div className="comparison-track">
        <span
          className={
            emphasized
              ? "emphasized"
              : ""
          }
          style={{
            width: `${
              clamp(value) *
              100
            }%`,
          }}
        />
      </div>
    </div>
  );
}


/* ============================================================================
   EVALUATION INTEGRITY
============================================================================ */

function EvaluationIntegrity() {
  return (
    <div className="dark-panel integrity-proof">
      <PanelHeading
        kicker="EVALUATION INTEGRITY"
        title="What the number means."
      />

      <div className="integrity-check">
        <Check size={14} />

        <span>
          Agent A information isolation is
          enforced.
        </span>
      </div>

      <div className="integrity-check">
        <Check size={14} />

        <span>
          Governor action space contains
          allow, evidence and human review.
        </span>
      </div>

      <div className="integrity-check">
        <Check size={14} />

        <span>
          Adaptive and human abusive
          populations are reported
          separately.
        </span>
      </div>

      <div className="integrity-check">
        <Check size={14} />

        <span>
          Legitimate intervention is
          tracked as a control metric.
        </span>
      </div>
    </div>
  );
}


/* ============================================================================
   COMMON
============================================================================ */

function PageIntro({
  eyebrow,
  title,
  description,
  badge,
}) {
  return (
    <div className="page-intro">
      <div className="page-intro-top">
        <div className="eyebrow">
          <span className="eyebrow-line" />
          {eyebrow}
        </div>

        {badge && (
          <div className="representative-badge">
            <CircleDot size={11} />
            {badge}
          </div>
        )}
      </div>

      <h1>{title}</h1>
      <p>{description}</p>
    </div>
  );
}


function PanelHeading({
  kicker,
  title,
}) {
  return (
    <div className="panel-heading">
      <div className="section-kicker">
        {kicker}
      </div>

      <h3>{title}</h3>
    </div>
  );
}


function getAdaptiveReduction(
  evaluation
) {
  const adaptive =
    findPopulation(
      evaluation?.result
        ?.population_metrics ||
        evaluation
          ?.population_metrics,
      "ADAPTIVE_ABUSIVE"
    );

  const value = getMetric(
    adaptive,
    "approval_reduction",
    "reduction",
    "approval_reduction_rate"
  );

  return value === null
    ? "21.46 pp"
    : pp(
        value,
        2
      );
}