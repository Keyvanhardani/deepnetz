/**
 * DeepNetz SPA — Chat, Models, Monitor, Settings
 * All-in-one application logic. No frameworks, no build tools.
 */
var DN = DN || {};

// ── Router ──────────────────────────────────────────────────────────
DN.router = {
  init: function() {
    window.addEventListener('hashchange', function() { DN.router.route(); });
    document.querySelectorAll('.nav-item').forEach(function(el) {
      el.addEventListener('click', function(e) {
        var page = el.getAttribute('data-page');
        if (page) { e.preventDefault(); location.hash = '#/' + page; }
      });
    });
    DN.router.route();
  },
  route: function() {
    var hash = location.hash.replace('#/', '') || 'chat';
    document.querySelectorAll('.page').forEach(function(p) { p.classList.remove('active'); });
    document.querySelectorAll('.nav-item').forEach(function(n) { n.classList.remove('active'); });
    var page = document.getElementById('page-' + hash);
    if (page) page.classList.add('active');
    var nav = document.querySelector('[data-page="' + hash + '"]');
    if (nav) nav.classList.add('active');
    if (hash === 'models') DN.models.load();
    if (hash === 'monitor') DN.monitor.start();
  }
};

// ── API ─────────────────────────────────────────────────────────────
DN.api = {
  get: function(url) { return fetch(url).then(function(r) { return r.json(); }); },
  post: function(url, data) {
    return fetch(url, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(data) }).then(function(r) { return r.json(); });
  }
};

// ── Health Check ────────────────────────────────────────────────────
DN.health = {
  failCount: 0,
  check: function() {
    DN.api.get('/health').then(function(d) {
      DN.health.failCount = 0;
      var dot = document.getElementById('status-dot');
      var txt = document.getElementById('status-text');
      if (d.loading) {
        dot.className = 'dot loading';
        txt.textContent = 'Loading...';
      } else if (d.model) {
        dot.className = 'dot ok';
        txt.textContent = d.model.split('/').pop().split('.')[0];
        document.getElementById('chat-input').disabled = false;
        document.getElementById('chat-send').disabled = false;
      } else {
        dot.className = 'dot';
        txt.textContent = 'No model';
      }
      document.getElementById('sb-model').textContent = d.model || 'No model';
      document.getElementById('sb-backend').textContent = d.backend || '—';
    }).catch(function() {
      DN.health.failCount++;
    });
  }
};

// ── Chat ────────────────────────────────────────────────────────────
DN.chat = {
  sessions: [],
  activeId: null,

  init: function() {
    var input = document.getElementById('chat-input');
    input.addEventListener('input', function() { this.style.height = 'auto'; this.style.height = Math.min(this.scrollHeight, 160) + 'px'; });
    input.addEventListener('keydown', function(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); DN.chat.send(); } });
    DN.chat.loadSessions();
  },

  loadSessions: function() {
    DN.api.get('/v1/sessions').then(function(d) {
      DN.chat.sessions = d.sessions || [];
      DN.chat.renderList();
    }).catch(function() {});
  },

  renderList: function() {
    var el = document.getElementById('chat-list');
    if (!DN.chat.sessions.length) { el.innerHTML = '<div style="padding:12px;color:var(--text3);font-size:12px;text-align:center">No chats</div>'; return; }
    el.innerHTML = DN.chat.sessions.map(function(s) {
      return '<div class="chat-item' + (s.id === DN.chat.activeId ? ' active' : '') + '" onclick="DN.chat.switchTo(\'' + s.id + '\')">' + DN.util.esc(s.title || 'New Chat') + '</div>';
    }).join('');
  },

  newChat: function() {
    DN.api.post('/v1/sessions', {title: ''}).then(function(d) {
      DN.chat.activeId = d.id;
      DN.chat.sessions.unshift({id: d.id, title: 'New Chat'});
      DN.chat.renderList();
      document.getElementById('chat-messages').innerHTML = '<div class="chat-welcome" id="chat-welcome"><div class="welcome-icon">DN</div><h3>DeepNetz</h3><p>Load a model and start chatting.</p></div>';
      document.getElementById('chat-title').textContent = 'New Chat';
    });
  },

  switchTo: function(id) {
    DN.chat.activeId = id;
    DN.chat.renderList();
    DN.api.get('/v1/sessions/' + id).then(function(d) {
      document.getElementById('chat-title').textContent = d.title || 'New Chat';
      var el = document.getElementById('chat-messages');
      el.innerHTML = '';
      (d.messages || []).forEach(function(m) { DN.chat.renderMsg(m.role, m.content); });
    });
  },

  renderMsg: function(role, content) {
    var el = document.getElementById('chat-messages');
    var w = document.getElementById('chat-welcome');
    if (w) w.style.display = 'none';
    var div = document.createElement('div');
    div.className = 'msg msg-' + (role === 'user' ? 'user' : 'bot');
    var body = role === 'user' ? DN.util.esc(content) : DN.util.md(content);
    div.innerHTML = '<div class="msg-avatar">' + (role === 'user' ? 'U' : 'DN') + '</div><div class="msg-body">' + body + '</div>';
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
    return div;
  },

  send: function() {
    var input = document.getElementById('chat-input');
    var text = input.value.trim();
    if (!text) return;
    input.value = '';
    input.style.height = 'auto';
    document.getElementById('chat-send').disabled = true;

    if (!DN.chat.activeId) {
      DN.api.post('/v1/sessions', {title: ''}).then(function(d) {
        DN.chat.activeId = d.id;
        DN.chat.sessions.unshift({id: d.id, title: text.slice(0, 40)});
        DN.chat.renderList();
        DN.chat._doSend(text);
      });
    } else {
      // Update title on first message
      var s = DN.chat.sessions.find(function(s) { return s.id === DN.chat.activeId; });
      if (s && (s.title === 'New Chat' || !s.title)) {
        s.title = text.slice(0, 40) + (text.length > 40 ? '...' : '');
        document.getElementById('chat-title').textContent = s.title;
        DN.chat.renderList();
      }
      DN.chat._doSend(text);
    }
  },

  _doSend: function(text) {
    DN.chat.renderMsg('user', text);
    var streamDiv = DN.chat.renderMsg('assistant', '');
    var bodyEl = streamDiv.querySelector('.msg-body');
    bodyEl.innerHTML = '<span style="color:var(--text3)">...</span>';

    // Collect messages
    var msgs = [];
    document.querySelectorAll('#chat-messages .msg').forEach(function(m) {
      var role = m.classList.contains('msg-user') ? 'user' : 'assistant';
      var body = m.querySelector('.msg-body');
      if (body) msgs.push({role: role, content: body.textContent});
    });

    var temp = parseFloat(document.getElementById('cfg-temp').value) || 0.7;
    var tokens = parseInt(document.getElementById('cfg-tokens').value) || 1024;
    var reasoning = document.getElementById('cfg-reasoning') ? document.getElementById('cfg-reasoning').checked : false;

    fetch('/v1/chat/completions', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model:'deepnetz', messages:msgs, stream:true, max_tokens:tokens, temperature:temp, session_id:DN.chat.activeId, reasoning:reasoning})
    }).then(function(resp) {
      var reader = resp.body.getReader();
      var dec = new TextDecoder();
      var buf = '', full = '';
      function read() {
        reader.read().then(function(r) {
          if (r.done) {
            if (!full) bodyEl.innerHTML = '<span style="color:var(--text3)">No response</span>';
            document.getElementById('chat-send').disabled = false;
            document.getElementById('chat-input').focus();
            return;
          }
          buf += dec.decode(r.value, {stream:true});
          var lines = buf.split('\n'); buf = lines.pop() || '';
          for (var i = 0; i < lines.length; i++) {
            if (lines[i].startsWith('data: ') && lines[i] !== 'data: [DONE]') {
              try {
                var c = JSON.parse(lines[i].slice(6));
                var t = (c.choices && c.choices[0] && c.choices[0].delta) ? c.choices[0].delta.content || '' : '';
                if (t) { full += t; bodyEl.innerHTML = DN.util.md(full); document.getElementById('chat-messages').scrollTop = 999999; }
              } catch(e) {}
            }
          }
          read();
        });
      }
      read();
    }).catch(function(e) {
      bodyEl.innerHTML = '<span style="color:var(--red)">Error: ' + DN.util.esc(e.message) + '</span>';
      document.getElementById('chat-send').disabled = false;
    });
  }
};

// ── Models ──────────────────────────────────────────────────────────
DN.models = {
  load: function() {
    // Local models
    DN.api.get('/v1/models').then(function(d) {
      var el = document.getElementById('local-models');
      var models = d.data || [];
      if (!models.length) { el.innerHTML = '<div style="color:var(--text3)">No local models. Use CLI: deepnetz pull &lt;model&gt;</div>'; return; }
      el.innerHTML = models.map(function(m) {
        var size = m.size_mb ? (m.size_mb / 1024).toFixed(1) + ' GB' : '';
        return '<div class="model-card"><h4>' + DN.util.esc(m.id) + '</h4><div class="meta">' + DN.util.esc(m.backend || '') + ' ' + size + '</div><div class="actions"><button class="btn btn-sm btn-primary" onclick="DN.models.loadModel(\'' + DN.util.esc(m.path || m.id) + '\',\'' + DN.util.esc(m.backend || '') + '\')">Load</button></div></div>';
      }).join('');
    });
    // Registry cards
    DN.api.get('/v1/cards/search?limit=24').then(function(d) {
      var el = document.getElementById('registry-models');
      var cards = d.cards || [];
      if (!cards.length) { el.innerHTML = '<div style="color:var(--text3)">No cards cached. Run: deepnetz cards generate</div>'; return; }
      el.innerHTML = cards.map(function(c) {
        var params = c.params_b ? c.params_b.toFixed(0) + 'B' : '';
        var tags = (c.tags || []).map(function(t) { return '<span class="tag tag-' + t + '">' + t + '</span>'; }).join('');
        return '<div class="model-card"><h4>' + DN.util.esc(c.name) + '</h4><div class="meta">' + params + ' ' + c.architecture + '</div><div class="tags">' + tags + '</div></div>';
      }).join('');
    }).catch(function() {});
  },

  search: function(q) {
    if (!q) { DN.models.load(); return; }
    DN.api.get('/v1/cards/search?q=' + encodeURIComponent(q) + '&limit=20').then(function(d) {
      var el = document.getElementById('registry-models');
      var cards = d.cards || [];
      el.innerHTML = cards.map(function(c) {
        var params = c.params_b ? c.params_b.toFixed(0) + 'B' : '';
        var tags = (c.tags || []).map(function(t) { return '<span class="tag tag-' + t + '">' + t + '</span>'; }).join('');
        var dl = c.downloads > 1000 ? (c.downloads/1000).toFixed(0) + 'k' : c.downloads;
        return '<div class="model-card"><h4>' + DN.util.esc(c.name) + '</h4><div class="meta">' + params + ' &middot; ' + dl + ' downloads</div><div class="tags">' + tags + '</div></div>';
      }).join('') || '<div style="color:var(--text3)">No results</div>';
    });
  },

  loadModel: function(path, backend) {
    DN.api.post('/v1/models/load', {model: path, backend: backend}).then(function() {
      DN.health.check();
    });
  }
};

// ── Monitor ─────────────────────────────────────────────────────────
DN.monitor = {
  ws: null,
  tpsHistory: [],

  start: function() {
    if (DN.monitor.ws) return;
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    DN.monitor.ws = new WebSocket(proto + '//' + location.host + '/ws/monitor');
    DN.monitor.ws.onmessage = function(e) {
      var d = JSON.parse(e.data);
      DN.monitor.update(d);
    };
    DN.monitor.ws.onclose = function() { DN.monitor.ws = null; };
  },

  update: function(d) {
    // GPU
    var gpu = d.gpu || {};
    document.getElementById('mon-gpu-name').textContent = gpu.name || 'No GPU';
    var gpuPct = gpu.util_percent || 0;
    var bar = document.getElementById('mon-gpu-bar');
    bar.style.width = gpuPct + '%';
    bar.className = 'mon-bar-fill' + (gpuPct > 90 ? ' crit' : gpuPct > 70 ? ' warn' : '');
    document.getElementById('mon-gpu-vram').textContent = (gpu.vram_used_mb || 0) + '/' + (gpu.vram_total_mb || 0) + ' MB';
    document.getElementById('mon-gpu-temp').textContent = (gpu.temp_c || 0) + '°C';

    // CPU
    var cpuPct = d.cpu ? d.cpu.percent : 0;
    document.getElementById('mon-cpu-val').textContent = cpuPct.toFixed(0) + '%';
    var cpuBar = document.getElementById('mon-cpu-bar');
    cpuBar.style.width = cpuPct + '%';
    cpuBar.className = 'mon-bar-fill' + (cpuPct > 90 ? ' crit' : cpuPct > 70 ? ' warn' : '');
    document.getElementById('mon-cpu-cores').textContent = d.cpu ? d.cpu.cores : '?';

    // RAM
    var ram = d.ram || {};
    var ramPct = ram.percent || 0;
    document.getElementById('mon-ram-val').textContent = ramPct.toFixed(0) + '%';
    var ramBar = document.getElementById('mon-ram-bar');
    ramBar.style.width = ramPct + '%';
    ramBar.className = 'mon-bar-fill' + (ramPct > 90 ? ' crit' : ramPct > 70 ? ' warn' : '');
    document.getElementById('mon-ram-used').textContent = ((ram.used_mb || 0) / 1024).toFixed(1) + ' GB';
    document.getElementById('mon-ram-total').textContent = ((ram.total_mb || 0) / 1024).toFixed(1) + ' GB';

    // TPS
    var tps = d.inference ? d.inference.tps : 0;
    document.getElementById('mon-tps').textContent = tps.toFixed(1) + ' tok/s';
    document.getElementById('sb-tps').textContent = tps > 0 ? tps.toFixed(1) + ' tok/s' : '—';
    document.getElementById('sb-gpu').textContent = gpu.name ? gpu.name + ' (' + (gpu.vram_used_mb||0) + 'MB)' : 'CPU';

    // TPS Chart
    DN.monitor.tpsHistory.push(tps);
    if (DN.monitor.tpsHistory.length > 60) DN.monitor.tpsHistory.shift();
    DN.monitor.drawChart();
  },

  drawChart: function() {
    var canvas = document.getElementById('mon-tps-chart');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var w = canvas.width, h = canvas.height;
    var data = DN.monitor.tpsHistory;
    var max = Math.max.apply(null, data) || 1;

    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (var i = 0; i < data.length; i++) {
      var x = (i / 59) * w;
      var y = h - (data[i] / max) * (h - 4) - 2;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
};

// ── Settings ────────────────────────────────────────────────────────
DN.settings = {
  setDevice: function(val) {
    document.querySelectorAll('.toggle-group .toggle').forEach(function(b) {
      b.classList.toggle('active', b.getAttribute('data-val') === val);
    });
  },
  updateLabel: function(input, labelId) {
    document.getElementById(labelId).textContent = input.value;
  }
};

// ── Auth ────────────────────────────────────────────────────────────
DN.auth = {
  check: function() {
    DN.api.get('/v1/auth/status').then(function(d) {
      if (d.logged_in) {
        document.getElementById('auth-login-btn').style.display = 'none';
        document.getElementById('auth-profile').style.display = 'flex';
        document.getElementById('auth-name').textContent = d.api_key_prefix || 'Logged in';
      }
    }).catch(function() {});
  }
};

// ── Utilities ───────────────────────────────────────────────────────
DN.util = {
  esc: function(s) { var d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; },
  md: function(text) {
    if (!text) return '';
    var h = DN.util.esc(text);
    h = h.replace(/```(\w*)\n([\s\S]*?)```/g, function(m,l,c) { return '<pre><code>' + c.trim() + '</code></pre>'; });
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
    h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
    h = h.replace(/^### (.+)$/gm, '<strong>$1</strong>');
    h = h.replace(/^## (.+)$/gm, '<strong style="font-size:16px">$1</strong>');
    h = h.replace(/\n\n/g, '</p><p>');
    h = h.replace(/\n/g, '<br>');
    return '<p>' + h + '</p>';
  }
};

// ── Init ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  DN.router.init();
  DN.chat.init();
  DN.auth.check();
  DN.health.check();
  setInterval(DN.health.check, 5000);
});
