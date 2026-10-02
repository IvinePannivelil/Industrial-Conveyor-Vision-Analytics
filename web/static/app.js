/**
 * Industrial Conveyor Vision Analytics — Frontend Controller
 * Handles real-time telemetry polling, canvas rendering, tab switching, and parameter updates.
 *
 * Telemetry separation:
 *   - pollTelemetry()            reads /api/telemetry          (per-frame live stats)
 *   - loadProductionHistory()    reads /api/production_history (aggregated CSV analytics)
 *
 * Production History source of truth: results/count_events.csv
 * Session OBJECTS COUNTED (top KPI) is independent from Production History totals.
 */

document.addEventListener("DOMContentLoaded", () => {

  // ── KPI Elements ────────────────────────────────────────────────────────────
  const kpiDetections     = document.getElementById("kpi-detections");
  const kpiDetectionsBadge = document.getElementById("kpi-detections-badge");
  const kpiCounted        = document.getElementById("kpi-counted");
  const kpiCountedBadge   = document.getElementById("kpi-counted-badge");
  const kpiTracks         = document.getElementById("kpi-tracks");
  const kpiTracksBadge    = document.getElementById("kpi-tracks-badge");
  const kpiTracksFooter   = document.getElementById("kpi-tracks-footer");
  const kpiLatency        = document.getElementById("kpi-latency");
  const kpiLatencyLabel   = document.getElementById("kpi-latency-label");
  const kpiLatencyBadge   = document.getElementById("kpi-latency-badge");
  const kpiFps            = document.getElementById("kpi-fps");
  const systemStatus      = document.getElementById("system-status");

  const hudTopLeft        = document.getElementById("hud-top-left");
  const hudTopRight       = document.getElementById("hud-top-right");
  const liveEventTicker   = document.getElementById("live-event-ticker");
  const eventsList        = document.getElementById("events-list");
  const streamInfoText    = document.getElementById("stream-info-text");

  const sliderConf  = document.getElementById("slider-conf");
  const valConf     = document.getElementById("val-conf");
  const sliderLine  = document.getElementById("slider-line");
  const valLine     = document.getElementById("val-line");

  const dirButtons  = document.querySelectorAll("#control-direction .seg-btn");
  const feedButtons = document.querySelectorAll("#control-feed .seg-btn");
  const toggleTracking = document.getElementById("toggle-tracking");
  const toggleCounting = document.getElementById("toggle-counting");
  const btnReset    = document.getElementById("btn-reset");

  const navButtons  = document.querySelectorAll(".nav-btn");
  const tabPanes    = document.querySelectorAll(".tab-pane");
  const docButtons  = document.querySelectorAll(".doc-link-btn");
  const docsContent = document.getElementById("docs-content");

  // Throughput sparkline chart (sidebar)
  const chartCanvas = document.getElementById("throughput-chart");
  const chartCtx    = chartCanvas ? chartCanvas.getContext("2d") : null;
  const throughputHistory = [];
  const maxHistoryPoints = 40;

  // ── Production History state ─────────────────────────────────────────────
  // Active query parameters — updated by preset buttons / custom range / filters
  let phState = {
    preset: "today",       // "today" | "30m" | "1h" | "custom"
    fromDt: null,          // ISO string YYYY-MM-DDTHH:MM:SS
    toDt:   null,          // ISO string YYYY-MM-DDTHH:MM:SS
    direction: "",          // "" | "DOWN" | "UP"
    source: "",             // "" | "simulator" | "live_camera"
    lastResult: null,       // last API response
  };

  // Production History DOM refs
  const phQfBtns       = document.querySelectorAll(".ph-qf-btn");
  const phCustomRange  = document.getElementById("ph-custom-range");
  const phFromDate     = document.getElementById("ph-from-date");
  const phFromTime     = document.getElementById("ph-from-time");
  const phToDate       = document.getElementById("ph-to-date");
  const phToTime       = document.getElementById("ph-to-time");
  const phApplyBtn     = document.getElementById("ph-apply-btn");
  const phDirBtns      = document.querySelectorAll("#ph-dir-ctrl .ph-seg-btn");
  const phSrcBtns      = document.querySelectorAll("#ph-src-ctrl .ph-seg-btn");
  const phTotal        = document.getElementById("ph-total");
  const phDown         = document.getElementById("ph-down");
  const phUp           = document.getElementById("ph-up");
  const phRate         = document.getElementById("ph-rate");
  const phMixedWarning = document.getElementById("ph-mixed-warning");
  const phSourceBadge  = document.getElementById("ph-source-badge");
  const phChartMeta    = document.getElementById("ph-chart-meta");
  const phBarCanvas    = document.getElementById("ph-bar-chart");
  const phEmptyState   = document.getElementById("ph-empty-state");
  const phEventsDetails = document.getElementById("ph-events-details");
  const phEventsTbody  = document.getElementById("ph-events-tbody");
  const phEventsCount  = document.getElementById("ph-events-count");
  const phNoteEl       = document.getElementById("ph-note");

  // Production History bar chart context
  const phChartCtx = phBarCanvas ? phBarCanvas.getContext("2d") : null;

  // ── Local Time Utility Helpers ───────────────────────────────────────────
  function localPad(n) {
    return String(n).padStart(2, "0");
  }
  function localDateStr(d = new Date()) {
    return `${d.getFullYear()}-${localPad(d.getMonth() + 1)}-${localPad(d.getDate())}`;
  }
  function localTimeStr(d = new Date()) {
    return `${localPad(d.getHours())}:${localPad(d.getMinutes())}`;
  }
  function localIsoString(d = new Date()) {
    return `${localDateStr(d)}T${localPad(d.getHours())}:${localPad(d.getMinutes())}:${localPad(d.getSeconds())}`;
  }
  function localMinutesAgo(n) {
    return localIsoString(new Date(Date.now() - n * 60 * 1000));
  }
  function localTodayStart() {
    return `${localDateStr()}T00:00:00`;
  }

  // ── Tab Switching ─────────────────────────────────────────────────────────
  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      navButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const activePane = document.getElementById(`tab-${targetTab}`);
      if (activePane) activePane.classList.add("active");
      if (targetTab === "documentation" && !docsContent.dataset.loaded) {
        loadDocumentation("readme");
      }
    });
  });

  // ── Documentation Loader ──────────────────────────────────────────────────
  function loadDocumentation(docName) {
    docsContent.innerHTML = `<div class="loading-spinner">Loading documentation (${docName})...</div>`;
    fetch(`/api/docs/${docName}`)
      .then(r => r.json())
      .then(data => {
        if (data.content && window.marked) {
          docsContent.innerHTML = marked.parse(data.content);
          docsContent.dataset.loaded = "true";
        } else {
          docsContent.innerHTML = `<p style="color:#F87171">Failed to parse: ${data.error || "Unknown"}</p>`;
        }
      })
      .catch(err => {
        docsContent.innerHTML = `<p style="color:#F87171">Network error: ${err}</p>`;
      });
  }
  docButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      docButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      loadDocumentation(btn.getAttribute("data-doc"));
    });
  });

  // ── Throughput Sparkline Chart (sidebar) ──────────────────────────────────
  function drawThroughputChart() {
    if (!chartCtx || !chartCanvas) return;
    const w = chartCanvas.width, h = chartCanvas.height;
    chartCtx.clearRect(0, 0, w, h);
    chartCtx.strokeStyle = "rgba(255,255,255,0.05)";
    chartCtx.lineWidth = 1;
    for (let y = 20; y < h; y += 30) {
      chartCtx.beginPath(); chartCtx.moveTo(0, y); chartCtx.lineTo(w, y); chartCtx.stroke();
    }
    if (throughputHistory.length < 2) return;
    const maxVal = Math.max(120, ...throughputHistory.map(p => p.throughput)) * 1.15;
    const stepX = w / (maxHistoryPoints - 1);
    const gradient = chartCtx.createLinearGradient(0, 0, 0, h);
    gradient.addColorStop(0, "rgba(0,240,255,0.25)");
    gradient.addColorStop(1, "rgba(0,240,255,0.0)");
    chartCtx.beginPath();
    chartCtx.moveTo(0, h);
    const startIndex = Math.max(0, throughputHistory.length - maxHistoryPoints);
    const slice = throughputHistory.slice(startIndex);
    slice.forEach((pt, idx) => {
      const x = idx * stepX;
      const y = h - (pt.throughput / maxVal) * (h - 15);
      idx === 0 ? chartCtx.lineTo(x, y) : chartCtx.lineTo(x, y);
    });
    chartCtx.lineTo((slice.length - 1) * stepX, h);
    chartCtx.closePath();
    chartCtx.fillStyle = gradient;
    chartCtx.fill();
    chartCtx.beginPath();
    chartCtx.strokeStyle = "#00F0FF";
    chartCtx.lineWidth = 2;
    slice.forEach((pt, idx) => {
      const x = idx * stepX;
      const y = h - (pt.throughput / maxVal) * (h - 15);
      idx === 0 ? chartCtx.moveTo(x, y) : chartCtx.lineTo(x, y);
    });
    chartCtx.stroke();
    if (slice.length > 0) {
      const lastX = (slice.length - 1) * stepX;
      const lastY = h - (slice[slice.length - 1].throughput / maxVal) * (h - 15);
      chartCtx.beginPath();
      chartCtx.arc(lastX, lastY, 4, 0, Math.PI * 2);
      chartCtx.fillStyle = "#FFFFFF";
      chartCtx.shadowColor = "#00F0FF";
      chartCtx.shadowBlur = 8;
      chartCtx.fill();
      chartCtx.shadowBlur = 0;
    }
  }

  // ── Telemetry Polling (4Hz) ───────────────────────────────────────────────
  let lastEventMsg = "";

  function pollTelemetry() {
    fetch("/api/telemetry")
      .then(r => r.json())
      .then(data => {
        if (systemStatus && data.system_status) systemStatus.textContent = data.system_status;

        if (kpiDetections) kpiDetections.textContent = data.current_detections;
        if (kpiCounted)    kpiCounted.textContent    = data.total_counted;
        if (kpiTracks)     kpiTracks.textContent     = data.active_tracks !== undefined ? data.active_tracks : 0;

        if (data.is_simulated) {
          if (kpiDetectionsBadge) { kpiDetectionsBadge.textContent = "SIMULATED FOV"; kpiDetectionsBadge.className = "kpi-badge badge-sim"; }
          if (kpiCountedBadge)    { kpiCountedBadge.textContent = "SIMULATED"; kpiCountedBadge.className = "kpi-badge badge-sim"; }
          if (kpiTracksBadge)     { kpiTracksBadge.textContent = "SIMULATED TRACKS"; kpiTracksBadge.className = "kpi-badge badge-sim"; }
          if (kpiTracksFooter)      kpiTracksFooter.textContent = "Simulator track IDs in frame";
          if (kpiLatencyLabel)      kpiLatencyLabel.textContent = "INFERENCE BUDGET";
          if (kpiLatencyBadge)    { kpiLatencyBadge.textContent = "TARGET BUDGET"; kpiLatencyBadge.className = "kpi-badge badge-target"; }
          if (kpiLatency) kpiLatency.innerHTML = `11.2 <span class="unit">ms target</span>`;
          if (kpiFps) kpiFps.textContent = `Simulation Rate: ${data.fps} FPS`;
        } else {
          if (kpiDetectionsBadge) { kpiDetectionsBadge.textContent = "LIVE CAMERA"; kpiDetectionsBadge.className = "kpi-badge badge-measured"; }
          if (kpiCountedBadge)    { kpiCountedBadge.textContent = "MEASURED"; kpiCountedBadge.className = "kpi-badge badge-measured"; }
          if (kpiTracksBadge)     { kpiTracksBadge.textContent = "BYTETRACK IDs"; kpiTracksBadge.className = "kpi-badge badge-measured"; }
          if (kpiTracksFooter)      kpiTracksFooter.textContent = "Active ByteTrack IDs this frame";
          if (data.model_loaded) {
            if (kpiLatencyLabel)  kpiLatencyLabel.textContent = "MODEL INFERENCE";
            if (kpiLatencyBadge){ kpiLatencyBadge.textContent = "MEASURED LOCALLY"; kpiLatencyBadge.className = "kpi-badge badge-measured"; }
            if (kpiLatency) kpiLatency.innerHTML = `${data.inference_latency_ms} <span class="unit">ms</span>`;
            if (kpiFps) kpiFps.textContent = `Measured runtime: ${data.fps} FPS`;
          } else {
            if (kpiLatencyLabel)  kpiLatencyLabel.textContent = "INFERENCE BUDGET";
            if (kpiLatencyBadge){ kpiLatencyBadge.textContent = "MODEL UNAVAILABLE"; kpiLatencyBadge.className = "kpi-badge badge-amber"; }
            if (kpiLatency) kpiLatency.innerHTML = `N/A <span class="unit">weights required</span>`;
            if (kpiFps) kpiFps.textContent = `Weights unconfirmed in public repo`;
          }
        }

        if (hudTopLeft)  hudTopLeft.textContent  = `YOLOv8n [${data.feed_mode.toUpperCase()}] • Latency: ${data.inference_latency_ms}ms`;
        if (hudTopRight) hudTopRight.textContent = `Conf: ${Math.round(data.conf_thresh * 100)}% | Line Y: ${Math.round(data.line_position * 100)}%`;

        if (data.events && data.events.length > 0) {
          const topEvent = data.events[0];
          if (topEvent.event !== lastEventMsg) {
            lastEventMsg = topEvent.event;
            if (liveEventTicker) liveEventTicker.textContent = `[${topEvent.timestamp}] ${topEvent.event}`;
            renderEventList(data.events);
          }
        }

        throughputHistory.push({ time: Date.now(), throughput: data.throughput_bpm });
        if (throughputHistory.length > maxHistoryPoints * 2) throughputHistory.shift();
        drawThroughputChart();
      })
      .catch(err => console.warn("Telemetry polling warning:", err));
  }

  function renderEventList(events) {
    if (!eventsList) return;
    eventsList.innerHTML = events.slice(0, 12).map(ev => `
      <div class="event-row ${ev.type || 'info'}">
        <span class="event-time">${ev.timestamp}</span>
        <span class="event-msg">${ev.event}</span>
        ${ev.total ? `<span class="event-badge">Total: ${ev.total}</span>` : ""}
      </div>`).join("");
  }

  // ── PRODUCTION HISTORY ────────────────────────────────────────────────────

  // Compute from/to ISO strings based on current preset (using local timezone)
  function computePresetRange(preset) {
    const now = localIsoString();
    switch (preset) {
      case "today": return { from: localTodayStart(), to: now };
      case "30m":   return { from: localMinutesAgo(30), to: now };
      case "1h":    return { from: localMinutesAgo(60), to: now };
      case "custom": {
        const fd = (phFromDate && phFromDate.value) ? phFromDate.value : localDateStr();
        const ft = (phFromTime && phFromTime.value) ? phFromTime.value : "00:00";
        const td = (phToDate   && phToDate.value)   ? phToDate.value   : localDateStr();
        const tt = (phToTime   && phToTime.value)   ? phToTime.value   : localTimeStr();
        return {
          from: `${fd}T${ft}:00`,
          to:   `${td}T${tt}:00`,
        };
      }
      default: return { from: localTodayStart(), to: now };
    }
  }

  // Fetch production history from API
  function fetchProductionHistory(preset, from, to, direction, source, includeEvents) {
    const params = new URLSearchParams();
    if (preset && preset !== "custom") {
      params.set("preset", preset);
    }
    if (from) params.set("from", from);
    if (to)   params.set("to", to);
    if (direction) params.set("direction", direction);
    if (source)    params.set("source", source);
    if (includeEvents) params.set("include_events", "1");
    params.set("event_limit", "200");
    return fetch(`/api/production_history?${params}`).then(r => r.json());
  }

  // Main render function
  function renderProductionHistory(data) {
    if (!data || data.error) {
      if (phTotal) phTotal.textContent = "Error";
      return;
    }
    phState.lastResult = data;

    // Summary cards
    if (phTotal) phTotal.textContent = data.total_objects.toLocaleString();
    if (phDown)  phDown.textContent  = data.down_count.toLocaleString();
    if (phUp)    phUp.textContent    = data.up_count.toLocaleString();
    if (phRate)  phRate.textContent  = data.average_rate.toFixed(1);

    // Source badge
    if (phSourceBadge) {
      if (data.source_filter === "simulator") {
        phSourceBadge.textContent = "SIMULATOR DATA";
      } else if (data.source_filter === "live_camera") {
        phSourceBadge.textContent = "LIVE CAMERA DATA";
      } else {
        phSourceBadge.textContent = "ALL SOURCES";
      }
    }

    // Mixed source warning
    if (phMixedWarning) {
      phMixedWarning.style.display = data.has_mixed_source ? "flex" : "none";
    }

    // Chart meta label
    if (phChartMeta) {
      phChartMeta.textContent = `${data.bucket_size_minutes}-min buckets · ${data.range_minutes.toFixed(0)} min range`;
    }

    // Bar chart
    drawProductionBarChart(data.buckets, data.bucket_size_minutes);

    // Events count (for drill-down summary line)
    if (phEventsCount) {
      phEventsCount.textContent = `${data.total_objects} event${data.total_objects !== 1 ? "s" : ""}`;
    }

    // Detailed events table (if included in response)
    if (data.events && phEventsTbody) {
      renderDetailedEvents(data.events);
    } else if (phEventsTbody && !phEventsDetails?.open) {
      phEventsTbody.innerHTML = `<tr><td colspan="5" class="ch-empty">Click ▸ DETAILED EVENTS to expand</td></tr>`;
    }
  }

  // Draw production bar chart on canvas
  function drawProductionBarChart(buckets, bucketMinutes) {
    if (!phChartCtx || !phBarCanvas) return;

    const container = document.getElementById("ph-chart-container");
    if (container) {
      phBarCanvas.width  = container.clientWidth || 700;
      phBarCanvas.height = 180;
    }

    const w = phBarCanvas.width;
    const h = phBarCanvas.height;
    const ctx = phChartCtx;
    ctx.clearRect(0, 0, w, h);

    // Empty state
    const nonEmpty = buckets.filter(b => b.count > 0);
    if (!buckets.length || nonEmpty.length === 0) {
      if (phEmptyState) phEmptyState.style.display = "flex";
      phBarCanvas.style.display = "none";
      return;
    }
    if (phEmptyState) phEmptyState.style.display = "none";
    phBarCanvas.style.display = "block";

    const maxCount = Math.max(...buckets.map(b => b.count), 1);
    const padL = 38, padR = 10, padT = 16, padB = 38;
    const chartW = w - padL - padR;
    const chartH = h - padT - padB;
    const barW = Math.max(2, (chartW / buckets.length) - 2);
    const gap  = Math.max(1, (chartW / buckets.length) - barW);

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    const gridSteps = 4;
    for (let i = 0; i <= gridSteps; i++) {
      const y = padT + chartH - (i / gridSteps) * chartH;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      // Y axis labels
      const val = Math.round((i / gridSteps) * maxCount);
      ctx.fillStyle = "rgba(148,163,184,0.7)";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(val, padL - 4, y + 4);
    }

    // Bars
    const barGradient = ctx.createLinearGradient(0, padT, 0, padT + chartH);
    barGradient.addColorStop(0, "rgba(16, 185, 129, 0.9)");
    barGradient.addColorStop(1, "rgba(16, 185, 129, 0.35)");

    buckets.forEach((b, i) => {
      const x = padL + i * (barW + gap);
      const barH = b.count === 0 ? 0 : Math.max(2, (b.count / maxCount) * chartH);
      const y = padT + chartH - barH;

      // Bar fill
      ctx.fillStyle = barGradient;
      ctx.beginPath();
      ctx.roundRect(x, y, barW, barH, [2, 2, 0, 0]);
      ctx.fill();

      // X label (show every Nth bucket to avoid crowding)
      const labelEvery = Math.max(1, Math.ceil(buckets.length / 10));
      if (i % labelEvery === 0) {
        ctx.fillStyle = "rgba(148,163,184,0.75)";
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.save();
        ctx.translate(x + barW / 2, padT + chartH + 10);
        // Rotate label if tight
        if (buckets.length > 12) {
          ctx.rotate(-Math.PI / 4);
          ctx.textAlign = "right";
        }
        ctx.fillText(b.label, 0, 0);
        ctx.restore();
      }
    });

    // Axis lines
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + chartH); ctx.lineTo(w - padR, padT + chartH);
    ctx.stroke();
  }

  // Render detailed events in drill-down table
  function renderDetailedEvents(events) {
    if (!phEventsTbody) return;
    if (!events.length) {
      phEventsTbody.innerHTML = `<tr><td colspan="5" class="ch-empty">No events in period.</td></tr>`;
      return;
    }
    const MAX_ROWS = 200;
    phEventsTbody.innerHTML = events.slice(0, MAX_ROWS).map(ev => {
      const srcClass = ev.source_mode === "simulator" ? "source-sim" : "source-live";
      const srcLabel = ev.source_mode === "simulator" ? "SIM" : "LIVE";
      const confPct  = ev.confidence ? (parseFloat(ev.confidence) * 100).toFixed(0) + "%" : "—";
      const dirIcon  = ev.direction === "DOWN" ? "↓" : (ev.direction === "UP" ? "↑" : ev.direction || "—");
      return `<tr>
        <td class="ch-time">${ev.timestamp || "—"}</td>
        <td class="ch-tid">#${ev.track_id}</td>
        <td class="ch-dir ${ev.direction === 'DOWN' ? 'dir-down' : 'dir-up'}">${dirIcon}</td>
        <td class="ch-conf">${confPct}</td>
        <td><span class="source-tag ${srcClass}">${srcLabel}</span></td>
      </tr>`;
    }).join("");
  }

  // Run a production history query with current phState
  function runProductionHistoryQuery(includeEvents) {
    const { from, to } = computePresetRange(phState.preset);
    phState.fromDt = from;
    phState.toDt = to;

    fetchProductionHistory(phState.preset, from, to, phState.direction, phState.source, includeEvents)
      .then(data => renderProductionHistory(data))
      .catch(err => {
        console.error("Production history error:", err);
        if (phTotal) phTotal.textContent = "—";
      });
  }

  // Wire preset quick-filter buttons
  phQfBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      phQfBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      phState.preset = btn.getAttribute("data-preset");

      if (phState.preset === "custom") {
        if (phCustomRange) phCustomRange.style.display = "block";
        // Pre-fill local date and time in custom inputs
        if (phFromDate && !phFromDate.value) phFromDate.value = localDateStr();
        if (phToDate   && !phToDate.value)   phToDate.value   = localDateStr();
        if (phFromTime && !phFromTime.value) phFromTime.value = "00:00";
        if (phToTime   && !phToTime.value)   phToTime.value   = localTimeStr();
      } else {
        if (phCustomRange) phCustomRange.style.display = "none";
        runProductionHistoryQuery(phEventsDetails?.open);
      }
    });
  });

  // Custom range Apply button
  if (phApplyBtn) {
    phApplyBtn.addEventListener("click", () => {
      phState.preset = "custom";
      runProductionHistoryQuery(phEventsDetails?.open);
    });
  }

  // Direction segmented control
  phDirBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      phDirBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      phState.direction = btn.getAttribute("data-dir");
      runProductionHistoryQuery(phEventsDetails?.open);
    });
  });

  // Source segmented control
  phSrcBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      phSrcBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      phState.source = btn.getAttribute("data-src");
      runProductionHistoryQuery(phEventsDetails?.open);
    });
  });

  // Detailed Events: load events when drill-down is opened
  if (phEventsDetails) {
    phEventsDetails.addEventListener("toggle", () => {
      if (phEventsDetails.open) {
        // Re-run with include_events=1
        runProductionHistoryQuery(true);
      }
    });
  }

  // ── Configuration Controls ────────────────────────────────────────────────
  function sendConfig(key, value) {
    fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ [key]: value }),
    }).catch(err => console.error("Config update error:", err));
  }

  if (sliderConf) {
    sliderConf.addEventListener("input", e => {
      const val = parseInt(e.target.value, 10);
      valConf.textContent = `${val}%`;
      sendConfig("conf_thresh", val / 100.0);
    });
  }
  if (sliderLine) {
    sliderLine.addEventListener("input", e => {
      const val = parseInt(e.target.value, 10);
      valLine.textContent = `${val}%`;
      sendConfig("line_position", val / 100.0);
    });
  }

  dirButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      dirButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      sendConfig("direction", btn.getAttribute("data-dir"));
    });
  });

  feedButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      feedButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const mode = btn.getAttribute("data-feed");
      sendConfig("feed_mode", mode);
      if (streamInfoText) {
        streamInfoText.textContent = mode === "camera"
          ? "Stream Source: Live USB / Built-in Camera"
          : "Stream Source: Industrial Conveyor Simulator";
      }
    });
  });

  if (toggleTracking) toggleTracking.addEventListener("change", e => sendConfig("tracking_enabled", e.target.checked));
  if (toggleCounting) toggleCounting.addEventListener("change", e => sendConfig("counting_enabled", e.target.checked));

  if (btnReset) {
    btnReset.addEventListener("click", () => {
      fetch("/api/reset", { method: "POST" })
        .then(() => {
          throughputHistory.length = 0;
          if (liveEventTicker) liveEventTicker.textContent = "Session counters reset — Production History CSV data preserved.";
          // Refresh production history to reflect current state
          setTimeout(() => runProductionHistoryQuery(phEventsDetails?.open), 300);
        })
        .catch(err => console.error("Reset error:", err));
    });
  }

  // ── Startup & Periodic Refresh ────────────────────────────────────────────
  // Pre-populate custom inputs with local date and time
  if (phFromDate) phFromDate.value = localDateStr();
  if (phToDate)   phToDate.value   = localDateStr();
  if (phFromTime) phFromTime.value = "00:00";
  if (phToTime)   phToTime.value   = localTimeStr();

  // Initial production history load
  runProductionHistoryQuery(false);

  // Telemetry polling: 4Hz
  setInterval(pollTelemetry, 250);

  // Production history auto-refresh: every 8s
  setInterval(() => {
    if (phState.preset !== "custom") {
      runProductionHistoryQuery(phEventsDetails?.open);
    }
  }, 8000);
});
