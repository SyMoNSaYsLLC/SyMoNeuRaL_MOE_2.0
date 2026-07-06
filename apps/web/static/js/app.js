const messages = document.getElementById("messages");
const promptEl = document.getElementById("prompt");
const busy = document.getElementById("busy");
const sendBtn = document.getElementById("send");

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function addMessage(role, content, meta = "") {
  const div = document.createElement("div");
  div.className = "msg " + role;

  const imageMatch = String(content).match(/!\[[^\]]*\]\(([^)]+)\)/);
  const fileMatch = String(content).match(/File:\s*`([^`]+)`/);

  let html = `<div class="meta">${escapeHtml(meta || role)}</div>`;

  if (imageMatch) {
    const url = imageMatch[1];
    const cleaned = String(content).replace(/!\[[^\]]*\]\([^\)]+\)/, "").trim();
    html += `<div>${escapeHtml(cleaned)}</div>`;
    html += `<img class="generated" src="${url}" alt="Generated image">`;
    html += `<div class="status" style="margin-top:8px"><a href="${url}" target="_blank">open image</a></div>`;
    if (fileMatch) {
      html += `<div class="status">Saved: ${escapeHtml(fileMatch[1])}</div>`;
    }
  } else {
    html += escapeHtml(content);
  }

  div.innerHTML = html;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

function clearMessages() {
  messages.innerHTML = "";
}

async function checkHealth() {
  const h = document.getElementById("health");
  try {
    const res = await fetch("/health", { cache: "no-store" });
    const j = await res.json();
    if (j.ok) {
      h.innerHTML = `<span class="ok">online</span>`;
    } else {
      const bad = Object.entries(j.workers || {}).filter(([_, ok]) => !ok).map(([k]) => k).join(", ");
      const backend = j?.runtime?.text_backend || "backend";
      const gpu = j?.gpu?.reason || "gpu";
      h.innerHTML = `<span class="err">degraded</span> · ${backend} · missing: ${bad || gpu}`;
    }
  } catch (e) {
    h.innerHTML = `<span class="err">offline</span>`;
  }
}

/* ---------- GPU telemetry panel ---------- */

const gpuDot = document.getElementById("gpu-dot");
const gpuName = document.getElementById("gpu-name");
const gpuMemText = document.getElementById("gpu-mem-text");
const gpuBarFill = document.getElementById("gpu-bar-fill");
const gpuUtil = document.getElementById("gpu-util");
const gpuTemp = document.getElementById("gpu-temp");
const gpuPower = document.getElementById("gpu-power");

let isBusy = false;

function setGpuDot(state) {
  gpuDot.classList.remove("online", "busy", "offline");
  gpuDot.classList.add(state);
}

async function pollGpuStats() {
  try {
    const res = await fetch("/gpu/stats", { cache: "no-store" });
    const g = await res.json();
    const dev = (g.devices && g.devices[0]) || null;

    if (!dev) {
      gpuName.textContent = "no GPU detected";
      gpuMemText.textContent = "–";
      gpuUtil.textContent = "–";
      gpuTemp.textContent = "–";
      gpuPower.textContent = "–";
      gpuBarFill.style.width = "0%";
      setGpuDot("offline");
      return;
    }

    gpuName.textContent = dev.name || "GPU";
    gpuMemText.textContent = `${dev.used_gb.toFixed(1)} / ${dev.total_gb.toFixed(1)} GB`;

    const pct = dev.total_gb > 0 ? Math.min(100, (dev.used_gb / dev.total_gb) * 100) : 0;
    gpuBarFill.style.width = pct.toFixed(0) + "%";
    gpuBarFill.style.background =
      pct >= 92 ? "var(--bad)" : pct >= 78 ? "var(--warn)" : "var(--accent)";

    gpuUtil.textContent = dev.utilization_pct != null ? `${dev.utilization_pct}%` : "–";
    gpuTemp.textContent = dev.temperature_c != null ? `${dev.temperature_c}°C` : "–";
    gpuPower.textContent = dev.power_w != null
      ? `${Math.round(dev.power_w)}${dev.power_limit_w ? " / " + Math.round(dev.power_limit_w) : ""}W`
      : "–";

    setGpuDot(isBusy ? "busy" : "online");
  } catch (e) {
    setGpuDot("offline");
  }
}

function normalizePromptForRouter(text) {
  const t = text.trim();

  // Do not force modes in UI. Just help obvious shorthand route correctly.
  if (/^(dragon|cat|dog|castle|car|logo|portrait|landscape)$/i.test(t)) {
    return "generate an image of " + t;
  }

  return t;
}

async function sendPrompt() {
  let prompt = normalizePromptForRouter(promptEl.value);
  if (!prompt) return;

  addMessage("user", prompt, "you");
  promptEl.value = "";
  busy.textContent = "SyMoNeuRaL is working...";
  sendBtn.disabled = true;
  isBusy = true;
  setGpuDot("busy");

  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify({
        model: "symoneural_moe",
        stream: false,
        messages: [{ role: "user", content: prompt }],
        max_tokens: 512
      })
    });

    const raw = await res.text();

    if (!res.ok) {
      throw new Error("HTTP " + res.status + "\n" + raw);
    }

    const data = JSON.parse(raw);
    const content =
      data?.choices?.[0]?.message?.content ||
      data?.choices?.[0]?.text ||
      JSON.stringify(data, null, 2);

    const model = data?.model || "SyMoNeuRaL MoE";
    addMessage("assistant", content, model);
  } catch (e) {
    addMessage("assistant", e.message || String(e), "error");
  } finally {
    busy.textContent = "";
    sendBtn.disabled = false;
    isBusy = false;
    setGpuDot("online");
    promptEl.focus();
  }
}

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendPrompt();
  }
});

/* ---------- GPU panel collapse/expand ---------- */

const gpuPanel = document.getElementById("gpu-panel");
const gpuToggle = document.getElementById("gpu-toggle");
const GPU_COLLAPSE_KEY = "symon_gpu_panel_collapsed";

function setGpuCollapsed(collapsed) {
  gpuPanel.classList.toggle("collapsed", collapsed);
  gpuToggle.setAttribute("aria-expanded", String(!collapsed));
  gpuToggle.setAttribute("aria-label", collapsed ? "Expand GPU panel" : "Collapse GPU panel");
  try {
    localStorage.setItem(GPU_COLLAPSE_KEY, collapsed ? "1" : "0");
  } catch (e) {
    // localStorage unavailable (private browsing etc.) -- collapse state just won't persist.
  }
}

gpuToggle.addEventListener("click", () => {
  setGpuCollapsed(!gpuPanel.classList.contains("collapsed"));
});

(function restoreGpuCollapsedState() {
  let saved = null;
  try {
    saved = localStorage.getItem(GPU_COLLAPSE_KEY);
  } catch (e) {
    saved = null;
  }
  if (saved === "1") {
    setGpuCollapsed(true);
  }
})();

checkHealth();
pollGpuStats();
setInterval(checkHealth, 15000);
setInterval(pollGpuStats, 5000);

addMessage(
  "assistant",
  "Ask for code. Ask for images. Ask for the build. SyMoNeuRaL routes it and gets to work.\n\nThis ain’t AI. This is SyMoNeuRaL.",
  "SyMoNeuRaL MoE"
);
