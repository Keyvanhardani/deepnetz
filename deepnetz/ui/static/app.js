// DeepNetz Dashboard — real-time monitoring via WebSocket

let ws = null;
let charts = {};

function initDashboard() {
  connectMonitor();
  fetchBackends();
  setInterval(fetchBackends, 30000); // refresh backends every 30s
}

function connectMonitor() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws/monitor`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateStats(data);
    updateCharts(data);
  };

  ws.onclose = () => setTimeout(connectMonitor, 3000);
  ws.onerror = () => ws.close();
}

function updateStats(data) {
  // CPU
  setText('cpu-percent', `${data.cpu.percent.toFixed(1)}%`);
  setBar('cpu-bar', data.cpu.percent);
  setText('cpu-cores', `${data.cpu.cores} cores`);

  // RAM
  const ramGB = (data.ram.used_mb / 1024).toFixed(1);
  const ramTotal = (data.ram.total_mb / 1024).toFixed(0);
  setText('ram-value', `${ramGB} / ${ramTotal} GB`);
  setBar('ram-bar', data.ram.percent);
  setText('ram-percent', `${data.ram.percent.toFixed(1)}%`);

  // GPU
  if (data.gpu.name) {
    setText('gpu-name', data.gpu.name);
    setText('gpu-util', `${data.gpu.util_percent.toFixed(0)}%`);
    setBar('gpu-bar', data.gpu.util_percent);
    const vramGB = (data.gpu.vram_used_mb / 1024).toFixed(1);
    const vramTotal = (data.gpu.vram_total_mb / 1024).toFixed(0);
    setText('vram-value', `${vramGB} / ${vramTotal} GB`);
    setBar('vram-bar', data.gpu.vram_percent);
    setText('gpu-temp', `${data.gpu.temp_c.toFixed(0)}°C`);
    setText('gpu-power', `${data.gpu.power_w.toFixed(0)}W`);
    document.getElementById('gpu-section').style.display = 'block';
  }

  // Inference
  if (data.inference.tps > 0) {
    setText('inf-tps', `${data.inference.tps.toFixed(1)} tok/s`);
    setText('inf-tokens', data.inference.tokens);
  }
}

function updateCharts(data) {
  if (!window.Chart) return;

  const ts = new Date().toLocaleTimeString();

  addChartPoint('cpuChart', ts, data.cpu.percent, 'CPU %', '#58a6ff');
  addChartPoint('ramChart', ts, data.ram.percent, 'RAM %', '#3fb950');
  if (data.gpu.name) {
    addChartPoint('gpuChart', ts, data.gpu.util_percent, 'GPU %', '#d29922');
    addChartPoint('vramChart', ts, data.gpu.vram_percent, 'VRAM %', '#f85149');
  }
}

function addChartPoint(id, label, value, dsLabel, color) {
  if (!charts[id]) {
    const ctx = document.getElementById(id);
    if (!ctx) return;
    charts[id] = new Chart(ctx.getContext('2d'), {
      type: 'line',
      data: { labels: [], datasets: [{ label: dsLabel, data: [], borderColor: color, borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { min: 0, max: 100, ticks: { color: '#8b949e' }, grid: { color: '#30363d' } }, x: { ticks: { color: '#8b949e', maxTicksLimit: 10 }, grid: { color: '#30363d' } } }, plugins: { legend: { display: false } } }
    });
  }
  const chart = charts[id];
  chart.data.labels.push(label);
  chart.data.datasets[0].data.push(value);
  if (chart.data.labels.length > 60) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update();
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setBar(id, percent) {
  const el = document.getElementById(id);
  if (el) el.style.width = `${Math.min(percent, 100)}%`;
}

async function fetchBackends() {
  try {
    const resp = await fetch('/v1/backends');
    const backends = await resp.json();
    const el = document.getElementById('backends-list');
    if (el) {
      el.innerHTML = backends.map(b =>
        `<tr><td>${b.name}</td><td>${b.version || '-'}</td><td><span class="badge ${b.available ? 'badge-active' : 'badge-inactive'}">${b.available ? 'active' : 'offline'}</span></td></tr>`
      ).join('');
    }
  } catch (e) {}
}

// Init on load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initDashboard);
} else {
  initDashboard();
}
