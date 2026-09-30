const $ = (id) => document.getElementById(id);

const state = {
  tab: "classify",
  schema: null,
  values: {},
};

const TAB_LABEL = {
  classify: "Run classify",
  forecast: "Run forecast",
  explain: "Run explain",
};

function apiHeaders(json) {
  const headers = {};
  if (json) headers["Content-Type"] = "application/json";
  const key = $("api-key").value.trim();
  if (key) headers["X-API-Key"] = key;
  return headers;
}

function setBanner(text) {
  const el = $("banner");
  if (!text) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.textContent = text;
}

function columnsForTab() {
  if (!state.schema) return [];
  if (state.tab === "forecast") {
    return state.schema.forecaster_features || [];
  }
  return state.schema.classifier_features || [];
}

function readGridIntoState() {
  document.querySelectorAll("#feature-grid input[data-col]").forEach((input) => {
    state.values[input.dataset.col] = input.value;
  });
}

function renderGrid() {
  const filter = $("filter").value.trim().toLowerCase();
  const cols = columnsForTab();
  const visible = filter
    ? cols.filter((c) => c.toLowerCase().includes(filter))
    : cols;
  const grid = $("feature-grid");
  grid.innerHTML = "";
  visible.forEach((col) => {
    const cell = document.createElement("div");
    cell.className = "cell";
    const label = document.createElement("label");
    label.textContent = col;
    label.title = col;
    label.setAttribute("for", `f-${col}`);
    const input = document.createElement("input");
    input.id = `f-${col}`;
    input.dataset.col = col;
    input.inputMode = "decimal";
    input.value = state.values[col] ?? "";
    input.addEventListener("input", () => {
      state.values[col] = input.value;
    });
    cell.append(label, input);
    grid.append(cell);
  });
  $("vector-meta").textContent = state.schema
    ? `${cols.length} required · showing ${visible.length}`
    : "Schema not loaded.";
}

function payloadFor(columns) {
  readGridIntoState();
  const features = {};
  const missing = [];
  const invalid = [];
  columns.forEach((col) => {
    const raw = (state.values[col] ?? "").trim();
    if (raw === "") {
      missing.push(col);
      return;
    }
    const n = Number(raw);
    if (!Number.isFinite(n)) invalid.push(col);
    else features[col] = n;
  });
  return { features, missing, invalid };
}

function renderStatus(health, schemaErr) {
  const ready = Boolean(health && health.ready);
  const pill = $("status-pill");
  pill.textContent = ready ? "ready" : "not ready";
  pill.className = `pill ${ready ? "ready" : "down"}`;
  $("fact-api").textContent = health ? health.status : "unreachable";
  $("fact-clf").textContent = health && health.classifier_loaded ? "loaded" : "missing";
  $("fact-fc").textContent = health && health.forecaster_loaded ? "loaded" : "missing";
  $("fact-ver").textContent = (state.schema && state.schema.model_version) || "—";
  $("version-label").textContent = state.schema?.model_version
    ? `model ${state.schema.model_version}`
    : "";
  if (schemaErr) setBanner(schemaErr);
  else if (!ready) {
    setBanner("Classifier is not loaded. Train with make pipeline, then reload.");
  } else setBanner("");
}

async function refresh() {
  let health = null;
  try {
    const h = await fetch("/health");
    health = await h.json();
  } catch {
    renderStatus(null, "Cannot reach the API.");
    return;
  }
  try {
    const s = await fetch("/schema");
    if (s.ok) {
      state.schema = await s.json();
      renderStatus(health, "");
      renderGrid();
    } else {
      state.schema = null;
      const body = await s.json().catch(() => ({}));
      renderStatus(health, body.detail || `Schema ${s.status}`);
      renderGrid();
    }
  } catch {
    renderStatus(health, "Schema request failed.");
  }
}

function showResult(html) {
  const el = $("result");
  el.hidden = false;
  el.innerHTML = html;
}

function classifyView(body) {
  const anomalous = Boolean(body.is_anomalous);
  const p = body.anomaly_probability;
  const pct = p == null ? null : Math.max(0, Math.min(100, p * 100));
  return `
    <p class="verdict ${anomalous ? "alert" : "ok"}">${anomalous ? "ANOMALY" : "BENIGN"}</p>
    ${
      pct == null
        ? ""
        : `<div class="meter" title="anomaly probability"><span style="width:${pct}%"></span></div>
           <p class="quiet">anomaly probability ${pct.toFixed(1)}%</p>`
    }
    <div class="meta-row">
      <span>${escapeHtml(body.model || "")}</span>
      <span>${escapeHtml(body.model_version || "")}</span>
      <span>${Number(body.latency_ms).toFixed(1)} ms</span>
    </div>`;
}

function forecastView(body) {
  return `
    <p class="verdict">${Number(body.predicted_traffic_volume).toLocaleString()}</p>
    <p class="quiet">predicted Total_Length_of_Fwd_Packets · horizon ${body.horizon_steps} step(s)</p>
    <div class="meta-row">
      <span>${escapeHtml(body.model || "")}</span>
      <span>${escapeHtml(body.model_version || "")}</span>
      <span>${Number(body.latency_ms).toFixed(1)} ms</span>
    </div>`;
}

function explainView(body) {
  const rows = (body.contributions || [])
    .slice()
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 16);
  const max = Math.max(...rows.map((r) => Math.abs(r.contribution)), 1e-9);
  const bars = rows
    .map((r) => {
      const mag = (Math.abs(r.contribution) / max) * 100;
      const cls = r.contribution >= 0 ? "pos" : "neg";
      return `<div class="bar-row">
        <span class="bar-name" title="${escapeHtml(r.feature)}">${escapeHtml(r.feature)}</span>
        <span class="bar-val">${r.contribution.toFixed(3)}</span>
        <div class="bar-track ${cls}"><span style="width:${mag}%"></span></div>
      </div>`;
    })
    .join("");
  return `
    <p class="quiet">method ${escapeHtml(body.method || "")} · ${escapeHtml(body.model || "")} · ${escapeHtml(body.model_version || "")}</p>
    <div class="bars">${bars || "<p class='quiet'>No contributions returned.</p>"}</div>`;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function run(event) {
  event.preventDefault();
  const cols = columnsForTab();
  if (!cols.length) {
    setBanner("No feature list for this tab. Load a trained model first.");
    return;
  }
  const { features, missing, invalid } = payloadFor(cols);
  if (missing.length || invalid.length) {
    const bits = [];
    if (missing.length) bits.push(`${missing.length} empty`);
    if (invalid.length) bits.push(`${invalid.length} non-numeric`);
    setBanner(`Vector incomplete: ${bits.join(", ")}. Use Fill ones or paste a complete JSON body.`);
    return;
  }
  setBanner("");
  $("busy").hidden = false;
  $("btn-run").disabled = true;
  const path = state.tab === "forecast" ? "/forecast" : state.tab === "explain" ? "/explain" : "/classify";
  try {
    const resp = await fetch(path, {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify({ features }),
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const detail = body.detail;
      const msg =
        typeof detail === "string"
          ? detail
          : detail && detail.message
            ? `${detail.message}${detail.missing ? ` (${detail.missing.length} missing)` : ""}`
            : `HTTP ${resp.status}`;
      setBanner(msg);
      $("result").hidden = true;
      return;
    }
    if (state.tab === "forecast") showResult(forecastView(body));
    else if (state.tab === "explain") showResult(explainView(body));
    else showResult(classifyView(body));
  } catch (err) {
    setBanner(String(err));
  } finally {
    $("busy").hidden = true;
    $("btn-run").disabled = false;
  }
}

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.classList.toggle("on", btn.dataset.tab === tab);
  });
  $("btn-run").textContent = TAB_LABEL[tab];
  renderGrid();
}

$("flow-form").addEventListener("submit", run);
$("filter").addEventListener("input", renderGrid);
$("btn-fill").addEventListener("click", () => {
  columnsForTab().forEach((col) => {
    state.values[col] = "1";
  });
  renderGrid();
});
$("btn-clear").addEventListener("click", () => {
  columnsForTab().forEach((col) => {
    state.values[col] = "";
  });
  renderGrid();
});
$("btn-apply-json").addEventListener("click", () => {
  try {
    const parsed = JSON.parse($("json-in").value);
    const features = parsed.features || parsed;
    if (!features || typeof features !== "object") throw new Error("expected {features: {...}}");
    Object.entries(features).forEach(([k, v]) => {
      state.values[k] = String(v);
    });
    renderGrid();
    setBanner("");
  } catch (err) {
    setBanner(`JSON: ${err.message || err}`);
  }
});
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => setTab(btn.dataset.tab));
});
$("api-key").value = sessionStorage.getItem("netanomaly.apiKey") || "";
$("api-key").addEventListener("change", () => {
  sessionStorage.setItem("netanomaly.apiKey", $("api-key").value);
});

refresh();
setInterval(refresh, 20000);
