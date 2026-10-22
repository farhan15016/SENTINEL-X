
import { useEffect, useState } from 'react'
import './App.css'

type ApiStatus = {
  project?: string
  status?: string
  rule_engine?: string
  machine_learning?: string
  database?: string
  error?: string
}

type Alert = {
  rule: string
  severity: string
  reason: string
}

type AnalysisResult = {
  project?: string
  engine?: string
  flagged?: boolean
  alert_count?: number
  alerts?: Alert[]
  error?: string
  details?: string
}

type HistoryItem = {
  id: number
  time: string
  failedLogins: number
  requestsPerMinute: number
  bytesSent: number
  result: AnalysisResult
}

type Page = 'overview' | 'analysis' | 'engines' | 'ml-analysis'

const API_BASE = 'http://127.0.0.1:8000'

type MlPrediction = {
  engine: string
  prediction: string
  label: number
  probabilities: Record<string, number> | null
  note?: string
  timestamp?: string
}

type AnomalyPrediction = {
  project?: string
  engine: string
  prediction: string
  label: number
  anomaly_score?: number
  contamination?: number
  note?: string
  timestamp?: string
  error?: string
  details?: string
}

// A fixed demonstration flow record. This is not live network traffic.
const DEMO_ML_RECORD: Record<string, string | number> = {
  dur: 0.000011,
  proto: 'udp',
  service: '-',
  state: 'INT',
  spkts: 2,
  dpkts: 0,
  sbytes: 496,
  dbytes: 0,
  rate: 90909.09,
  sttl: 254,
  dttl: 0,
  sload: 180363632,
  dload: 0,
  sloss: 0,
  dloss: 0,
  sinpkt: 0.011,
  dinpkt: 0,
  sjit: 0,
  djit: 0,
  swin: 0,
  stcpb: 0,
  dtcpb: 0,
  dwin: 0,
  tcprtt: 0,
  synack: 0,
  ackdat: 0,
  smean: 248,
  dmean: 0,
  trans_depth: 0,
  response_body_len: 0,
  ct_srv_src: 2,
  ct_state_ttl: 2,
  ct_dst_ltm: 1,
  ct_src_dport_ltm: 1,
  ct_dst_sport_ltm: 1,
  ct_dst_src_ltm: 2,
  is_ftp_login: 0,
  ct_ftp_cmd: 0,
  ct_flw_http_mthd: 0,
  ct_src_ltm: 1,
  ct_srv_dst: 2,
  is_sm_ips_ports: 0,
}

const MODEL_METRICS = [
  {
    name: 'Random Forest',
    type: 'Supervised learning',
    accuracy: 90.73,
    precision: 87.63,
    recall: 96.83,
    f1: 92.0,
    matrix: [
      [30806, 6194],
      [1436, 43896],
    ],
  },
  {
    name: 'Isolation Forest (baseline)',
    type: 'Unsupervised learning · contamination 0.01',
    accuracy: 55.30,
    precision: 96.43,
    recall: 19.53,
    f1: 32.48,
    matrix: [
      [36672, 328],
      [36478, 8854],
    ],
  },
  {
    name: 'Isolation Forest (tuned)',
    type: 'Unsupervised learning · contamination 0.20',
    accuracy: 70.66,
    precision: 76.06,
    recall: 68.17,
    f1: 71.90,
    matrix: [
      [27273, 9727],
      [14428, 30904],
    ],
  },
]

const TOP_FEATURES = [
  { name: 'ct_dst_src_ltm', value: 0.021425 },
  { name: 'service', value: 0.019997 },
  { name: 'dmean', value: 0.017422 },
  { name: 'sbytes', value: 0.010816 },
  { name: 'smean', value: 0.009726 },
  { name: 'dbytes', value: 0.008201 },
  { name: 'ct_srv_dst', value: 0.007515 },
  { name: 'ct_srv_src', value: 0.00705 },
  { name: 'synack', value: 0.006041 },
  { name: 'tcprtt', value: 0.005155 },
]

function App() {
  const [page, setPage] = useState<Page>('overview')
  const [status, setStatus] = useState<ApiStatus | null>(null)
  const [statusError, setStatusError] = useState('')
  const [failedLogins, setFailedLogins] = useState('15')
  const [requestsPerMinute, setRequestsPerMinute] = useState('800')
  const [bytesSent, setBytesSent] = useState('150000000')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [apiError, setApiError] = useState('')
  const [history, setHistory] = useState<HistoryItem[]>([])
const [mlResult, setMlResult] = useState<MlPrediction | null>(null)

const [mlLoading, setMlLoading] = useState(false)
const [mlError, setMlError] = useState('')
  const [anomalyResult, setAnomalyResult] = useState<AnomalyPrediction | null>(null)
  const [anomalyLoading, setAnomalyLoading] = useState(false)
  const [anomalyError, setAnomalyError] = useState('')
  async function checkStatus() {
    try {
      const response = await fetch(`${API_BASE}/api/status`)
      const data: ApiStatus = await response.json()

      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`)
      }

      setStatus(data)
      setStatusError('')
    } catch (error) {
      setStatus(null)
      setStatusError(
        error instanceof Error ? error.message : 'Unable to reach the API',
      )
    }
  }

  useEffect(() => {
    void checkStatus()

    const timer = window.setInterval(() => {
      void checkStatus()
    }, 10000)

    return () => window.clearInterval(timer)
  }, [])

  async function analyseEvent(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setLoading(true)
    setApiError('')
    setResult(null)

    const payload = {
      failed_logins: Number(failedLogins),
      requests_per_minute: Number(requestsPerMinute),
      bytes_sent: Number(bytesSent),
    }

    if (
      !Number.isSafeInteger(payload.failed_logins) ||
      !Number.isSafeInteger(payload.requests_per_minute) ||
      !Number.isSafeInteger(payload.bytes_sent) ||
      payload.failed_logins < 0 ||
      payload.requests_per_minute < 0 ||
      payload.bytes_sent < 0
    ) {
      setApiError('Enter valid non-negative whole numbers for all fields.')
      setLoading(false)
      return
    }

    try {
      const response = await fetch(`${API_BASE}/api/analyse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      const data: AnalysisResult = await response.json()

      if (!response.ok) {
        throw new Error(
          data.details || data.error || `Analysis failed (HTTP ${response.status})`,
        )
      }

      setResult(data)

      const item: HistoryItem = {
        id: Date.now(),
        time: new Date().toLocaleTimeString(),
        failedLogins: payload.failed_logins,
        requestsPerMinute: payload.requests_per_minute,
        bytesSent: payload.bytes_sent,
        result: data,
      }

      setHistory((previous) => [item, ...previous].slice(0, 20))
    } catch (error) {
      setApiError(
        error instanceof Error
          ? `${error.message}. Check the backend and its CORS configuration.`
          : 'Could not contact the SENTINEL-X backend.',
      )
    } finally {
      setLoading(false)
    }
  }

  async function runMlPrediction() {
    setMlLoading(true)
    setMlError('')
    setMlResult(null)

    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(DEMO_ML_RECORD),
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(
          data.details || data.error || `Prediction failed (HTTP ${response.status})`,
        )
      }

      setMlResult(data as MlPrediction)
    } catch (error) {
      setMlError(
        error instanceof Error
          ? `${error.message}. Confirm the backend is running and CORS is enabled.`
          : 'Unable to contact the machine-learning API.',
      )
    } finally {
      setMlLoading(false)
    }
  }

  async function runAnomalyPrediction() {
    setAnomalyLoading(true)
    setAnomalyError('')
    setAnomalyResult(null)

    try {
      const response = await fetch(`${API_BASE}/api/anomaly`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(DEMO_ML_RECORD),
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(
          data.details || data.error || `Anomaly detection failed (HTTP ${response.status})`,
        )
      }
      setAnomalyResult(data as AnomalyPrediction)
    } catch (error) {
      setAnomalyError(
        error instanceof Error
          ? `${error.message}. Confirm the backend and /api/anomaly endpoint are available.`
          : 'Unable to contact the Isolation Forest API.',
      )
    } finally {
      setAnomalyLoading(false)
    }
  }

  const isOnline = status !== null
  const modelReady =
    status?.machine_learning === 'loaded' ||
    status?.machine_learning === 'model_file_available'

  function navigate(nextPage: Page) {
    setPage(nextPage)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  function renderResult(analysis: AnalysisResult) {
    return (
      <div className={`result-box ${analysis.flagged ? 'danger' : 'safe'}`}>
        <div className="result-header">
          <strong>
            {analysis.flagged
              ? '⚠ Suspicious activity detected'
              : '✓ No rule alerts detected'}
          </strong>
          <span>{analysis.alert_count ?? 0} alert(s)</span>
        </div>

        <p>Engine: {analysis.engine || 'Rule-based'}</p>

        {analysis.alerts && analysis.alerts.length > 0 ? (
          <ul className="alert-list">
            {analysis.alerts.map((alert, index) => (
              <li key={`${alert.rule}-${index}`}>
                <span className={`severity ${alert.severity.toLowerCase()}`}>
                  {alert.severity}
                </span>
                <div>
                  <strong>{alert.rule}</strong>
                  <p>{alert.reason}</p>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p>No configured rules were triggered by these inputs.</p>
        )}
      </div>
    )
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">S</div>
          <div>
            <h1>SENTINEL<span>-X</span></h1>
            <p>CYBER INTELLIGENCE</p>
          </div>
        </div>

        <div className="nav-label">PLATFORM</div>

        <button
          type="button"
          className={`nav-item ${page === 'overview' ? 'active' : ''}`}
          onClick={() => navigate('overview')}
        >
          ◈ <span>Overview</span>
        </button>

        <button
          type="button"
          className={`nav-item ${page === 'analysis' ? 'active' : ''}`}
          onClick={() => navigate('analysis')}
        >
          ⌁ <span>Threat analysis</span>
        </button>

        <button
          type="button"
          className={`nav-item ${page === 'engines' ? 'active' : ''}`}
          onClick={() => navigate('engines')}
        >
          ⬡ <span>Detection engines</span>
        </button>

        <button
          type="button"
          className={`nav-item ${page === 'ml-analysis' ? 'active' : ''}`}
          onClick={() => navigate('ml-analysis')}
        >
          ◉ <span>ML analysis</span>
        </button>

        <div className="sidebar-bottom">
          <div className="status-dot" />
          <div>
            <strong>{isOnline ? 'API connected' : 'API unavailable'}</strong>
            <p>127.0.0.1:8000</p>
            {!isOnline && statusError && (
              <p className="footer-muted">{statusError}</p>
            )}
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">SECURITY OPERATIONS CENTER</p>
            <h2>
              {page === 'overview'
                ? 'Threat overview'
                : page === 'analysis'
                  ? 'Threat analysis'
                  : page === 'ml-analysis'
                    ? 'Machine-learning analysis'
                    : 'Detection engines'}
            </h2>
          </div>

          <div className={`connection-pill ${isOnline ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            {isOnline ? 'SYSTEM ONLINE' : 'API DISCONNECTED'}
          </div>
        </header>

        {page === 'overview' && (
          <>
            <section className="hero-panel" id="overview">
              <div className="hero-copy">
                <div className="live-label">
                  <span /> NETWORK DEFENSE SYSTEM
                </div>
                <h2>See threats.<br /><span>Stop surprises.</span></h2>
                <p>
                  An experimental network intrusion detection platform
                  combining machine learning with transparent rule-based alerts.
                </p>
                <button
                  type="button"
                  className="primary-link"
                  onClick={() => navigate('analysis')}
                >
                  Run a security analysis ↗
                </button>
              </div>

              <div className="radar" aria-label="Decorative radar graphic">
                <div className="radar-ring ring-one" />
                <div className="radar-ring ring-two" />
                <div className="radar-ring ring-three" />
                <div className="radar-cross cross-horizontal" />
                <div className="radar-cross cross-vertical" />
                <div className="radar-sweep" />
                <div className="radar-point point-one" />
                <div className="radar-point point-two" />
                <div className="radar-point point-three" />
                <div className="radar-center" />
              </div>
              <div className="hero-watermark">SX</div>
            </section>

            <section className="stats-grid">
              <article className="stat-card">
                <div className="stat-top"><span>PLATFORM</span><span className="stat-icon">◈</span></div>
                <h3>{status?.project || 'SENTINEL-X'}</h3>
                <p>Cybersecurity intelligence</p>
              </article>

              <article className="stat-card">
                <div className="stat-top"><span>RULE ENGINE</span><span className="stat-icon">⌁</span></div>
                <h3>{status?.rule_engine === 'ready' ? 'Ready' : isOnline ? 'Check status' : 'Offline'}</h3>
                <p>Deterministic security rules</p>
              </article>

              <article className="stat-card">
                <div className="stat-top"><span>ML MODEL</span><span className="stat-icon">⬡</span></div>
                <h3>{modelReady ? 'Available' : isOnline ? 'Unavailable' : 'Unknown'}</h3>
                <p>Random Forest pipeline</p>
              </article>

              <article className="stat-card">
                <div className="stat-top"><span>DATABASE</span><span className="stat-icon">▤</span></div>
                <h3>{status?.database === 'connected' ? 'Connected' : 'Not configured'}</h3>
                <p>Persistent event storage</p>
              </article>
            </section>

            <section className="analysis-layout">
              <article className="panel analysis-panel">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">RULE-BASED DETECTION</p>
                    <h2>Analyse a network event</h2>
                  </div>
                  <span className="panel-symbol">⌁</span>
                </div>

                <p className="panel-description">
                  Enter sample traffic metrics to test the current rule engine.
                </p>

                <AnalysisForm
                  failedLogins={failedLogins}
                  requestsPerMinute={requestsPerMinute}
                  bytesSent={bytesSent}
                  setFailedLogins={setFailedLogins}
                  setRequestsPerMinute={setRequestsPerMinute}
                  setBytesSent={setBytesSent}
                  analyseEvent={analyseEvent}
                  loading={loading}
                />

                {apiError && <div className="error-box">{apiError}</div>}
                {result && renderResult(result)}
              </article>

              <EngineSummary
                modelReady={modelReady}
                ruleReady={status?.rule_engine === 'ready'}
              />
            </section>
          </>
        )}

        {page === 'analysis' && (
          <section className="page-section">
            <div className="panel">
              <p className="eyebrow">INVESTIGATION WORKSPACE</p>
              <h2>Threat analysis</h2>
              <p className="panel-description">
                Submit traffic metrics to the live rule engine and review the
                returned alerts. This prototype analyses the values you enter;
                it does not monitor live network traffic.
              </p>

              <AnalysisForm
                failedLogins={failedLogins}
                requestsPerMinute={requestsPerMinute}
                bytesSent={bytesSent}
                setFailedLogins={setFailedLogins}
                setRequestsPerMinute={setRequestsPerMinute}
                setBytesSent={setBytesSent}
                analyseEvent={analyseEvent}
                loading={loading}
              />

              {apiError && <div className="error-box">{apiError}</div>}

              {result && (
                <div className="analysis-result">
                  <h3>Latest analysis result</h3>
                  {renderResult(result)}
                </div>
              )}
            </div>

            <div className="panel history-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">SESSION ACTIVITY</p>
                  <h2>Recent analyses</h2>
                </div>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => setHistory([])}
                  disabled={history.length === 0}
                >
                  Clear history
                </button>
              </div>

              <p className="panel-description">
                History is held in browser memory and clears when this page
                reloads. It is not stored in a database.
              </p>

              {history.length === 0 ? (
                <div className="empty-state">
                  <span>◷</span>
                  <strong>No analyses yet</strong>
                  <p>Run an analysis to see its results here.</p>
                </div>
              ) : (
                <div className="history-list">
                  {history.map((item) => (
                    <article className="history-item" key={item.id}>
                      <div>
                        <strong>
                          {item.result.flagged
                            ? 'Suspicious activity'
                            : 'No rule alerts'}
                        </strong>
                        <p>{item.time} · {item.result.alert_count ?? 0} alert(s)</p>
                        <small>
                          Logins: {item.failedLogins} · Requests/min: {item.requestsPerMinute}
                        </small>
                      </div>
                      <span className={item.result.flagged ? 'history-danger' : 'history-safe'}>
                        {item.result.flagged ? 'FLAGGED' : 'CLEAR'}
                      </span>
                    </article>
                  ))}
                </div>
              )}
            </div>
          </section>
        )}

        {page === 'ml-analysis' && (
          <section className="page-section">
            <div className="panel ml-analysis-panel">
              <p className="eyebrow">SUPERVISED CLASSIFICATION</p>
              <h2>Random Forest prediction</h2>
              <p className="panel-description">
                Send a complete 42-feature network-flow record to the trained
                UNSW-NB15 Random Forest pipeline and inspect its prediction.
              </p>

              <div className="note-box">
                <strong>Demonstration record</strong>
                <p>
                  Both detection buttons submit the same fixed example record with
                  all 42 required features. This prototype does not capture live traffic.
                </p>
              </div>

              <div className="ml-record-summary">
                <div><span>Input features</span><strong>42 / 42</strong></div>
                <div><span>Model</span><strong>Random Forest</strong></div>
                <div><span>Dataset</span><strong>UNSW-NB15</strong></div>
              </div>

              <button
                type="button"
                className="analyse-button"
                onClick={() => void runMlPrediction()}
                disabled={mlLoading || !isOnline}
              >
                {mlLoading ? 'Running prediction…' : 'Run Random Forest prediction →'}
              </button>

              {mlError && <div className="error-box">{mlError}</div>}
              {mlResult && (
                <div className={`ml-prediction-result ${mlResult.prediction === 'attack' ? 'ml-attack' : 'ml-normal'}`}>
                  <p className="eyebrow">RANDOM FOREST RESULT</p>
                  <h3>{mlResult.prediction === 'attack' ? '⚠ Attack predicted' : '✓ Normal traffic predicted'}</h3>
                  <p>Engine: {mlResult.engine} · Label: {mlResult.label}</p>
                  {mlResult.probabilities && (
                    <div className="ml-probability-list">
                      {Object.entries(mlResult.probabilities).map(([classLabel, probability]) => (
                        <div className="ml-probability-row" key={classLabel}>
                          <div className="ml-probability-heading">
                            <span>{classLabel === '1' ? 'Attack' : classLabel === '0' ? 'Normal' : `Class ${classLabel}`}</span>
                            <strong>{(probability * 100).toFixed(2)}%</strong>
                          </div>
                          <div className="feature-track">
                            <div className="feature-bar" style={{ width: `${Math.max(0, Math.min(100, probability * 100))}%` }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {mlResult.note && <p className="matrix-caption">{mlResult.note}</p>}
                  {mlResult.timestamp && <p className="matrix-caption">Evaluated: {new Date(mlResult.timestamp).toLocaleString()}</p>}
                </div>
              )}

              <div className="evaluation-heading anomaly-section-heading">
                <p className="eyebrow">UNSUPERVISED ANOMALY DETECTION</p>
                <h2>Isolation Forest</h2>
                <p className="panel-description">
                  Uses the separately trained tuned model (contamination 0.20) to
                  identify unusual records. An anomaly is not definitive proof of an attack.
                </p>
              </div>

              <button
                type="button"
                className="analyse-button"
                onClick={() => void runAnomalyPrediction()}
                disabled={anomalyLoading || !isOnline}
              >
                {anomalyLoading ? 'Checking for anomalies…' : 'Run Isolation Forest anomaly detection →'}
              </button>

              {anomalyError && <div className="error-box">{anomalyError}</div>}
              {anomalyResult && (
                <div className={`ml-prediction-result ${anomalyResult.prediction === 'attack' ? 'ml-attack' : 'ml-normal'}`}>
                  <p className="eyebrow">ISOLATION FOREST RESULT</p>
                  <h3>{anomalyResult.prediction === 'attack' ? '⚠ Anomalous traffic predicted' : '✓ No anomaly predicted'}</h3>
                  <p>Engine: {anomalyResult.engine} · Label: {anomalyResult.label}</p>
                  <div className="ml-record-summary anomaly-metrics">
                    <div>
                      <span>Anomaly score</span>
                      <strong>{typeof anomalyResult.anomaly_score === 'number' ? anomalyResult.anomaly_score.toFixed(5) : 'N/A'}</strong>
                    </div>
                    <div>
                      <span>Contamination</span>
                      <strong>{typeof anomalyResult.contamination === 'number' ? `${(anomalyResult.contamination * 100).toFixed(0)}%` : 'N/A'}</strong>
                    </div>
                    <div><span>Model</span><strong>Tuned Isolation Forest</strong></div>
                  </div>
                  {anomalyResult.note && <p className="matrix-caption">{anomalyResult.note}</p>}
                  {anomalyResult.timestamp && <p className="matrix-caption">Evaluated: {new Date(anomalyResult.timestamp).toLocaleString()}</p>}
                </div>
              )}

              {!isOnline && <p className="ml-helper">Backend is offline. Start the API before running either model.</p>}
            </div>
          </section>
        )}

        {page === 'engines' && (
          <section className="page-section">
            <div className="panel">
              <p className="eyebrow">MODEL STATUS</p>
              <h2>Detection engines</h2>
              <p className="panel-description">
                Current availability reported by the SENTINEL-X backend.
                Experimental components are labelled separately from the
                operational rule engine.
              </p>

              <div className="engine-detail">
                <div className="engine-icon">01</div>
                <div className="engine-detail-copy">
                  <h3>Rule-based detection</h3>
                  <p>Deterministic thresholds with explainable alert reasons.</p>
                  <small>Checks failed logins, request rate, and outbound bytes.</small>
                </div>
                <span className={`engine-state ${status?.rule_engine === 'ready' ? '' : 'muted'}`}>
                  {status?.rule_engine === 'ready' ? 'READY' : 'CHECK'}
                </span>
              </div>

              <div className="engine-detail">
                <div className="engine-icon">02</div>
                <div className="engine-detail-copy">
                  <h3>Random Forest</h3>
                  <p>Supervised classification using the trained UNSW-NB15 pipeline.</p>
                  <small>
                    Model file: {modelReady ? 'available' : 'unavailable'}
                    {status?.machine_learning === 'loaded' ? ' · loaded in memory' : ''}
                  </small>
                </div>
                <span className={`engine-state ${modelReady ? '' : 'muted'}`}>
                  {modelReady ? 'READY' : 'CHECK'}
                </span>
              </div>

              <div className="engine-detail">
                <div className="engine-icon">03</div>
                <div className="engine-detail-copy">
                  <h3>Isolation Forest</h3>
                  <p>Experimental unsupervised anomaly detection.</p>
                  <small>Integrated through the backend POST /api/anomaly endpoint.</small>
                </div>
                <span className={`engine-state ${isOnline ? '' : 'muted'}`}>{isOnline ? 'INTEGRATED' : 'CHECK'}</span>
              </div>

              <div className="note-box">
                <strong>Research prototype</strong>
                <p>
                  Model readiness does not mean every model is invoked by the
                  dashboard. This page reports backend status and documents
                  current integration boundaries.
                </p>
              </div>

              <button
                type="button"
                className="secondary-button"
                onClick={() => void checkStatus()}
              >
                Refresh engine status
              </button>
              {statusError && <div className="error-box">{statusError}</div>}

<ModelEvaluation />

            </div>
          </section>
        )}

        <footer>
          <span>SENTINEL-X <span className="footer-muted">/ SECURITY INTELLIGENCE</span></span>
          <span>LOCAL DEVELOPMENT BUILD · 2026</span>
        </footer>
      </main>
    </div>
  )
}

type AnalysisFormProps = {
  failedLogins: string
  requestsPerMinute: string
  bytesSent: string
  setFailedLogins: (value: string) => void
  setRequestsPerMinute: (value: string) => void
  setBytesSent: (value: string) => void
  analyseEvent: (event: React.FormEvent<HTMLFormElement>) => Promise<void>
  loading: boolean
}

function AnalysisForm({
  failedLogins,
  requestsPerMinute,
  bytesSent,
  setFailedLogins,
  setRequestsPerMinute,
  setBytesSent,
  analyseEvent,
  loading,
}: AnalysisFormProps) {
  return (
    <form onSubmit={analyseEvent} className="analysis-form">
      <label>
        Failed login attempts
        <input
          type="number"
          min="0"
          step="1"
          value={failedLogins}
          onChange={(event) => setFailedLogins(event.target.value)}
          required
        />
        <small>Alert threshold: 10 or more</small>
      </label>

      <label>
        Requests per minute
        <input
          type="number"
          min="0"
          step="1"
          value={requestsPerMinute}
          onChange={(event) => setRequestsPerMinute(event.target.value)}
          required
        />
        <small>Alert threshold: 500 or more</small>
      </label>

      <label>
        Outbound bytes sent
        <input
          type="number"
          min="0"
          step="1"
          value={bytesSent}
          onChange={(event) => setBytesSent(event.target.value)}
          required
        />
        <small>Alert threshold: 100,000,000 bytes</small>
      </label>

      <button className="analyse-button" type="submit" disabled={loading}>
        {loading ? 'Analysing…' : 'Run analysis →'}
      </button>
    </form>
  )
}

function EngineSummary({
  modelReady,
  ruleReady,
}: {
  modelReady: boolean
  ruleReady: boolean
}) {
  return (
    <article className="panel engine-panel">
      <p className="eyebrow">DETECTION ARCHITECTURE</p>
      <h2>Detection architecture</h2>
      <p className="panel-description">
        SENTINEL-X combines transparent rules with trained machine-learning
        research models.
      </p>

      <div className="engine-row">
        <div className="engine-icon">01</div>
        <div>
          <strong>Rule-based engine</strong>
          <p>Explicit thresholds and explainable alerts</p>
        </div>
        <span className={`engine-state ${ruleReady ? '' : 'muted'}`}>
          {ruleReady ? 'READY' : 'CHECK'}
        </span>
      </div>

      <div className="engine-row">
        <div className="engine-icon">02</div>
        <div>
          <strong>Random Forest</strong>
          <p>Supervised traffic classification</p>
        </div>
        <span className={`engine-state ${modelReady ? '' : 'muted'}`}>
          {modelReady ? 'READY' : 'CHECK'}
        </span>
      </div>

      <div className="engine-row">
        <div className="engine-icon">03</div>
        <div>
          <strong>Isolation Forest</strong>
          <p>Experimental anomaly detection</p>
        </div>
        <span className="engine-state muted">EXPERIMENTAL</span>
      </div>

      <div className="note-box">
        <strong>Research prototype</strong>
        <p>
          This dashboard displays local model and rule-engine results. It is
          not a production network security system.
        </p>
      </div>
    </article>
  )
}


function MetricCard({
  label,
  value,
}: {
  label: string
  value: number
}) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value.toFixed(2)}%</strong>
    </div>
  )
}

function ConfusionMatrix({
  matrix,
}: {
  matrix: number[][]
}) {
  return (
    <div className="matrix-grid">
      <div className="matrix-heading">Actual ↓ / Predicted →</div>
      <div className="matrix-cell matrix-correct">
        <small>True Negative</small>
        <strong>{matrix[0][0].toLocaleString('en-IN')}</strong>
      </div>
      <div className="matrix-cell matrix-error">
        <small>False Positive</small>
        <strong>{matrix[0][1].toLocaleString('en-IN')}</strong>
      </div>
      <div className="matrix-cell matrix-error">
        <small>False Negative</small>
        <strong>{matrix[1][0].toLocaleString('en-IN')}</strong>
      </div>
      <div className="matrix-cell matrix-correct">
        <small>True Positive</small>
        <strong>{matrix[1][1].toLocaleString('en-IN')}</strong>
      </div>
    </div>
  )
}

function ModelEvaluation() {
  return (
    <>
      <div className="evaluation-heading">
        <p className="eyebrow">HELD-OUT TEST SET</p>
        <h2>Model performance</h2>
        <p className="panel-description">
          Random Forest and Isolation Forest baseline/tuned results on the
          UNSW-NB15 test dataset. The tuned contamination was selected using
          a separate validation split before evaluation on the official test set.
        </p>
      </div>

      <div className="model-comparison-grid">
        {MODEL_METRICS.map((model) => (
          <article className="panel model-card" key={model.name}>
            <div className="model-card-heading">
              <div>
                <h3>{model.name}</h3>
                <p>{model.type}</p>
              </div>
              <span className="model-tag">
                {model.name === 'Random Forest' ? 'PRIMARY' : model.name.includes('(tuned)') ? 'TUNED EXPERIMENT' : 'BASELINE'}
              </span>
            </div>

            <div className="metric-grid">
              <MetricCard label="Accuracy" value={model.accuracy} />
              <MetricCard label="Precision" value={model.precision} />
              <MetricCard label="Attack recall" value={model.recall} />
              <MetricCard label="F1 score" value={model.f1} />
            </div>

            <h4 className="subsection-title">Confusion matrix</h4>
            <ConfusionMatrix matrix={model.matrix} />

            <p className="matrix-caption">
              Rows are actual classes; columns are predicted classes.
              Normal is the negative class and attack is the positive class.
            </p>
          </article>
        ))}
      </div>

      <article className="panel feature-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">MODEL INTERPRETABILITY</p>
            <h2>Top 10 influential features</h2>
          </div>
          <span className="panel-symbol">↗</span>
        </div>

        <p className="panel-description">
          Permutation importance measured using F1 score. Larger positive
          values indicate that shuffling a feature reduced model performance
          more strongly in this evaluation.
        </p>

        <div className="feature-list">
          {TOP_FEATURES.map((feature) => {
            const percentage =
              (feature.value / TOP_FEATURES[0].value) * 100

            return (
              <div className="feature-row" key={feature.name}>
                <span className="feature-name">{feature.name}</span>
                <div className="feature-track">
                  <div
                    className="feature-bar"
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <span className="feature-value">
                  {feature.value.toFixed(6)}
                </span>
              </div>
            )
          })}
        </div>

        <div className="note-box">
          <strong>Isolation Forest tuning result</strong>
          <p>
            The tuned experiment uses contamination=0.20, selected on a separate
            validation split. On the official test set it achieved 71.90% attack
            F1 and 68.17% attack recall, compared with 32.48% F1 and 19.53% recall
            for the baseline contamination=0.01 model. This is an experimental
            evaluation; it does not overwrite the saved baseline model used by the API.
          </p>
        </div>

        <div className="note-box">
          <strong>How to interpret this chart</strong>
          <p>
            These are relative permutation-importance scores, not percentages
            of attacks detected or causal proof that a feature creates an
            attack. Small differences may reflect evaluation variability.
          </p>
        </div>
      </article>
    </>
  )
}


export default App
