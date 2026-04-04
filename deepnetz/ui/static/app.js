/**
 * DeepNetz SPA — Chat, Models, Monitor, Settings
 * Production-quality UI. No frameworks, no build tools.
 */
var DN = DN || {};

/* ================================================================
   TOAST NOTIFICATIONS
   ================================================================ */
DN.toast = {
  show: function(msg, type, duration) {
    type = type || 'info';
    duration = duration || 3500;
    var container = document.getElementById('toast-container');
    var t = document.createElement('div');
    t.className = 'toast toast-' + type;
    var icons = {
      success: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>',
      error: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
      info: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
      warning: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
    };
    t.innerHTML = '<span class="toast-icon">' + (icons[type] || icons.info) + '</span><span class="toast-msg">' + DN.util.esc(msg) + '</span>';
    container.appendChild(t);
    requestAnimationFrame(function() { t.classList.add('toast-visible'); });
    setTimeout(function() {
      t.classList.remove('toast-visible');
      setTimeout(function() { if (t.parentNode) t.parentNode.removeChild(t); }, 300);
    }, duration);
  }
};

/* ================================================================
   UTILITIES
   ================================================================ */
DN.util = {
  esc: function(s) {
    var d = document.createElement('div');
    d.textContent = s || '';
    return d.innerHTML;
  },

  md: function(text) {
    if (!text) return '';
    // Process code blocks first (preserve content)
    var codeBlocks = [];
    var processed = text.replace(/```(\w*)\n([\s\S]*?)```/g, function(m, lang, code) {
      var idx = codeBlocks.length;
      codeBlocks.push({ lang: lang, code: code.trimEnd() });
      return '\x00CODEBLOCK' + idx + '\x00';
    });

    // Escape HTML
    var h = DN.util.esc(processed);

    // Inline code
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold, italic
    h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Headers
    h = h.replace(/^### (.+)$/gm, '<h4>$1</h4>');
    h = h.replace(/^## (.+)$/gm, '<h3>$1</h3>');
    h = h.replace(/^# (.+)$/gm, '<h2>$1</h2>');

    // Links
    h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // Unordered lists
    h = h.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
    h = h.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Ordered lists
    h = h.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Blockquotes
    h = h.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

    // Paragraphs
    h = h.replace(/\n\n/g, '</p><p>');
    h = h.replace(/\n/g, '<br>');
    h = '<p>' + h + '</p>';

    // Restore code blocks with copy button and language label
    h = h.replace(/\x00CODEBLOCK(\d+)\x00/g, function(m, idx) {
      var block = codeBlocks[parseInt(idx)];
      var langLabel = block.lang ? '<span class="code-lang">' + DN.util.esc(block.lang) + '</span>' : '';
      return '</p><div class="code-block-wrap">' +
        '<div class="code-block-header">' + langLabel +
        '<button class="code-copy-btn" onclick="DN.util.copyCode(this)" title="Copy code"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy</button>' +
        '</div><pre><code>' + DN.util.esc(block.code) + '</code></pre></div><p>';
    });

    // Clean empty paragraphs
    h = h.replace(/<p>\s*<\/p>/g, '');

    return h;
  },

  copyCode: function(btn) {
    var pre = btn.closest('.code-block-wrap').querySelector('pre code');
    if (pre) {
      navigator.clipboard.writeText(pre.textContent).then(function() {
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
        setTimeout(function() {
          btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
        }, 2000);
      });
    }
  },

  timeAgo: function(dateStr) {
    if (!dateStr) return '';
    var now = Date.now();
    var then = new Date(dateStr).getTime();
    var diff = Math.floor((now - then) / 1000);
    if (diff < 60) return 'just now';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
    if (diff < 2592000) return Math.floor(diff / 86400) + 'd ago';
    return new Date(dateStr).toLocaleDateString();
  },

  formatSize: function(mb) {
    if (!mb) return '';
    if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
    return Math.round(mb) + ' MB';
  },

  debounce: function(fn, delay) {
    var timer;
    return function() {
      var args = arguments;
      var ctx = this;
      clearTimeout(timer);
      timer = setTimeout(function() { fn.apply(ctx, args); }, delay);
    };
  }
};

/* ================================================================
   API HELPER
   ================================================================ */
DN.api = {
  get: function(url) {
    return fetch(url).then(function(r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    });
  },
  post: function(url, data) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    }).then(function(r) { return r.json(); });
  },
  put: function(url, data) {
    return fetch(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    }).then(function(r) { return r.json(); });
  },
  del: function(url) {
    return fetch(url, { method: 'DELETE' }).then(function(r) { return r.json(); });
  }
};

/* ================================================================
   UI HELPERS
   ================================================================ */
DN.ui = {
  toggleSidebar: function() {
    var sb = document.getElementById('sidebar');
    var ov = document.getElementById('sidebar-overlay');
    sb.classList.toggle('open');
    ov.classList.toggle('visible');
  },

  closeSidebar: function() {
    document.getElementById('sidebar').classList.remove('open');
    document.getElementById('sidebar-overlay').classList.remove('visible');
  }
};

/* ================================================================
   ROUTER
   ================================================================ */
DN.router = {
  current: 'chat',

  init: function() {
    // Handle browser back/forward
    window.addEventListener('popstate', function() { DN.router.route(); });

    // Sidebar overlay click
    document.getElementById('sidebar-overlay').addEventListener('click', DN.ui.closeSidebar);

    // Route from current URL
    DN.router.route();
  },

  go: function(page) {
    history.pushState(null, '', '/' + page);
    DN.router.route();
    DN.ui.closeSidebar();
  },

  route: function() {
    var path = location.pathname.replace(/^\//, '') || 'chat';
    if (path === '') path = 'chat';
    var validPages = ['chat', 'models', 'monitor', 'settings'];
    if (validPages.indexOf(path) === -1) path = 'chat';

    DN.router.current = path;

    // Toggle pages
    document.querySelectorAll('.page').forEach(function(p) {
      p.classList.toggle('hidden', p.id !== 'page-' + path);
    });

    // Toggle nav links
    document.querySelectorAll('.nav-link').forEach(function(n) {
      n.classList.toggle('active', n.getAttribute('data-page') === path);
    });

    // Page title
    var titles = { chat: 'Chat', models: 'Models', monitor: 'Monitor', settings: 'Settings' };
    document.getElementById('page-title').textContent = titles[path] || 'DeepNetz';

    // Show/hide chat sidebar section
    var chatSection = document.getElementById('sidebar-chat-section');
    if (chatSection) {
      chatSection.classList.toggle('hidden', path !== 'chat');
    }

    // Show/hide footer (hide on chat page)
    var footer = document.getElementById('app-footer');
    if (footer) {
      footer.classList.toggle('hidden', path === 'chat');
    }

    // Page-specific init
    if (path === 'models') DN.models.init();
    if (path === 'monitor') DN.monitor.start();
    if (path === 'settings') DN.settings.init();
    if (path === 'chat') DN.chat.focus();
  }
};

/* ================================================================
   HEALTH CHECK
   ================================================================ */
DN.health = {
  model: '',
  backend: '',
  loading: false,

  check: function() {
    DN.api.get('/health').then(function(d) {
      var prevModel = DN.health.model;
      DN.health.model = d.model || '';
      DN.health.backend = d.backend || 'none';
      DN.health.loading = d.loading || false;
      DN.health.updateUI();
      // Update toolbar when model changes
      if (DN.health.model && DN.health.model !== prevModel && DN.chat.updateToolbar) {
        DN.chat.updateToolbar();
      }
    }).catch(function() {});
  },

  updateUI: function() {
    var dot = document.getElementById('header-status-dot');
    var txt = document.getElementById('header-status-text');

    if (DN.health.loading) {
      dot.className = 'dot loading';
      txt.textContent = 'Loading model...';
    } else if (DN.health.model) {
      dot.className = 'dot online';
      // Show short model name
      var name = DN.health.model.split('/').pop().replace(/\.gguf$/i, '');
      txt.textContent = name;
    } else {
      dot.className = 'dot offline';
      txt.textContent = 'No model loaded';
    }

    // Update chat welcome
    DN.chat.updateWelcome();

    // Enable/disable chat input
    var inp = document.getElementById('chat-input');
    var btn = document.getElementById('chat-send');
    if (inp && btn) {
      var hasModel = !!DN.health.model && !DN.health.loading;
      inp.disabled = !hasModel;
      inp.placeholder = hasModel ? 'Send a message...' : 'Load a model first...';
      if (!DN.chat.sending) btn.disabled = !hasModel;
    }
  }
};

/* ================================================================
   CHAT
   ================================================================ */
DN.chat = {
  sessions: [],
  activeId: null,
  sending: false,
  allSessions: [], // unfiltered
  features: { think: false, reasoning: false, web_search: false, tool_call: false },
  imageData: null, // base64 image for vision

  init: function() {
    var input = document.getElementById('chat-input');
    // Auto-resize textarea
    input.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 180) + 'px';
      // Enable send button if there's text and model is loaded
      var btn = document.getElementById('chat-send');
      if (!DN.chat.sending) {
        btn.disabled = !this.value.trim() || !DN.health.model;
      }
    });

    // Enter to send, Shift+Enter for newline
    input.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        DN.chat.send();
      }
    });

    // Scroll to bottom button
    var chatMessages = document.getElementById('chat-messages');
    chatMessages.addEventListener('scroll', function() {
      var el = this;
      var nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
      document.getElementById('scroll-to-bottom').classList.toggle('hidden', nearBottom);
    });

    DN.chat.loadSessions();
  },

  focus: function() {
    var inp = document.getElementById('chat-input');
    if (inp && !inp.disabled) {
      setTimeout(function() { inp.focus(); }, 100);
    }
  },

  updateWelcome: function() {
    var title = document.getElementById('welcome-title');
    var sub = document.getElementById('welcome-subtitle');
    if (!title || !sub) return;
    if (DN.health.model) {
      var name = DN.health.model.split('/').pop().replace(/\.gguf$/i, '');
      title.textContent = name;
      sub.textContent = 'How can I help you today?';
    } else {
      title.textContent = 'DeepNetz';
      sub.textContent = 'Load a model under Models to start chatting.';
    }
  },

  loadSessions: function() {
    DN.api.get('/v1/sessions').then(function(d) {
      DN.chat.allSessions = (d.sessions || []).sort(function(a, b) {
        return (b.updated_at || b.created_at || '').localeCompare(a.updated_at || a.created_at || '');
      });
      DN.chat.sessions = DN.chat.allSessions.slice();
      DN.chat.renderList();
    }).catch(function() {});
  },

  renderList: function() {
    var el = document.getElementById('chat-list');
    if (!DN.chat.sessions.length) {
      el.innerHTML = '<div class="sidebar-empty">No conversations yet</div>';
      return;
    }
    el.innerHTML = DN.chat.sessions.map(function(s) {
      var isActive = s.id === DN.chat.activeId;
      var title = s.title || 'New Chat';
      var time = DN.util.timeAgo(s.updated_at || s.created_at);
      var msgCount = s.message_count || 0;
      return '<div class="conversation-item' + (isActive ? ' active' : '') + '" data-id="' + s.id + '" onclick="DN.chat.switchTo(\'' + s.id + '\')">' +
        '<div class="conversation-item-content">' +
          '<div class="conversation-item-title">' + DN.util.esc(title) + '</div>' +
          '<div class="conversation-item-meta">' + (msgCount > 0 ? msgCount + ' messages' : '') + (time ? ' \u00B7 ' + time : '') + '</div>' +
        '</div>' +
        '<div class="conversation-item-actions">' +
          '<button class="conv-action-btn" onclick="event.stopPropagation();DN.chat.renameSession(\'' + s.id + '\')" title="Rename">' +
            '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>' +
          '</button>' +
          '<button class="conv-action-btn conv-delete-btn" onclick="event.stopPropagation();DN.chat.deleteSession(\'' + s.id + '\')" title="Delete">' +
            '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>' +
          '</button>' +
        '</div>' +
      '</div>';
    }).join('');
  },

  filterSessions: function(query) {
    query = (query || '').toLowerCase().trim();
    if (!query) {
      DN.chat.sessions = DN.chat.allSessions.slice();
    } else {
      DN.chat.sessions = DN.chat.allSessions.filter(function(s) {
        return (s.title || '').toLowerCase().indexOf(query) !== -1;
      });
    }
    DN.chat.renderList();
  },

  newChat: function() {
    DN.api.post('/v1/sessions', { title: '' }).then(function(d) {
      DN.chat.activeId = d.id;
      DN.chat.allSessions.unshift({ id: d.id, title: 'New Chat', created_at: d.created_at, updated_at: d.created_at, message_count: 0 });
      DN.chat.sessions = DN.chat.allSessions.slice();
      DN.chat.renderList();
      DN.chat.clearMessages();
      DN.chat.showWelcome();
      DN.chat.focus();
    }).catch(function(e) {
      DN.toast.show('Failed to create chat: ' + e.message, 'error');
    });
  },

  switchTo: function(id) {
    DN.chat.activeId = id;
    DN.chat.renderList();
    DN.api.get('/v1/sessions/' + id).then(function(d) {
      DN.chat.clearMessages();
      if (!d.messages || d.messages.length === 0) {
        DN.chat.showWelcome();
      } else {
        DN.chat.hideWelcome();
        d.messages.forEach(function(m) {
          DN.chat.appendMessage(m.role, m.content, false);
        });
        DN.chat.scrollToBottom();
      }
    }).catch(function(e) {
      DN.toast.show('Failed to load conversation', 'error');
    });
  },

  renameSession: function(id) {
    var s = DN.chat.allSessions.find(function(x) { return x.id === id; });
    var current = s ? (s.title || 'New Chat') : '';
    // Create inline rename input
    var item = document.querySelector('.conversation-item[data-id="' + id + '"]');
    if (!item) return;
    var titleEl = item.querySelector('.conversation-item-title');
    var oldHTML = titleEl.innerHTML;
    var input = document.createElement('input');
    input.type = 'text';
    input.className = 'conv-rename-input';
    input.value = current;
    titleEl.innerHTML = '';
    titleEl.appendChild(input);
    input.focus();
    input.select();

    function doRename() {
      var newTitle = input.value.trim();
      if (newTitle && newTitle !== current) {
        DN.api.put('/v1/sessions/' + id, { title: newTitle }).then(function() {
          if (s) s.title = newTitle;
          DN.chat.renderList();
          DN.toast.show('Renamed', 'success');
        }).catch(function() {
          DN.toast.show('Rename failed', 'error');
          DN.chat.renderList();
        });
      } else {
        DN.chat.renderList();
      }
    }

    input.addEventListener('blur', doRename);
    input.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
      if (e.key === 'Escape') { titleEl.innerHTML = oldHTML; }
    });
  },

  deleteSession: function(id) {
    // Use a custom confirm via a temporary overlay
    if (!confirm('Delete this conversation?')) return;
    DN.api.del('/v1/sessions/' + id).then(function() {
      DN.chat.allSessions = DN.chat.allSessions.filter(function(s) { return s.id !== id; });
      DN.chat.sessions = DN.chat.sessions.filter(function(s) { return s.id !== id; });
      if (DN.chat.activeId === id) {
        DN.chat.activeId = null;
        DN.chat.clearMessages();
        DN.chat.showWelcome();
      }
      DN.chat.renderList();
      DN.toast.show('Conversation deleted', 'success');
    }).catch(function() {
      DN.toast.show('Delete failed', 'error');
    });
  },

  clearMessages: function() {
    var inner = document.getElementById('chat-messages-inner');
    // Remove all messages but keep welcome
    var msgs = inner.querySelectorAll('.message');
    msgs.forEach(function(m) { m.remove(); });
  },

  showWelcome: function() {
    var w = document.getElementById('chat-welcome');
    if (w) w.style.display = '';
    DN.chat.updateWelcome();
  },

  hideWelcome: function() {
    var w = document.getElementById('chat-welcome');
    if (w) w.style.display = 'none';
  },

  appendMessage: function(role, content, animate) {
    DN.chat.hideWelcome();
    var inner = document.getElementById('chat-messages-inner');
    var div = document.createElement('div');
    div.className = 'message message-' + (role === 'user' ? 'user' : 'assistant');
    if (animate !== false) div.style.animation = 'messageIn 200ms ease';

    var avatarContent = role === 'user' ? 'U' : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>';

    div.innerHTML = '<div class="message-avatar">' + avatarContent + '</div>' +
      '<div class="message-body"><div class="message-content">' +
      (role === 'user' ? DN.util.esc(content) : DN.util.md(content)) +
      '</div></div>';

    inner.appendChild(div);
    return div;
  },

  scrollToBottom: function() {
    var el = document.getElementById('chat-messages');
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  },

  toggleFeature: function(name) {
    DN.chat.features[name] = !DN.chat.features[name];
    var btn = document.getElementById('tb-' + (name === 'web_search' ? 'search' : name === 'tool_call' ? 'tools' : name));
    if (btn) btn.classList.toggle('active', DN.chat.features[name]);
  },

  uploadImage: function() {
    document.getElementById('image-upload').click();
  },

  handleImageUpload: function(input) {
    if (!input.files || !input.files[0]) return;
    var file = input.files[0];
    var reader = new FileReader();
    reader.onload = function(e) {
      DN.chat.imageData = e.target.result; // data:image/...;base64,...
      var preview = document.getElementById('chat-image-preview');
      document.getElementById('chat-image-thumb').src = DN.chat.imageData;
      preview.classList.remove('hidden');
      // Auto-enable vision toolbar button
      var btn = document.getElementById('tb-vision');
      if (btn) btn.classList.add('active');
    };
    reader.readAsDataURL(file);
    input.value = '';
  },

  clearImage: function() {
    DN.chat.imageData = null;
    document.getElementById('chat-image-preview').classList.add('hidden');
    document.getElementById('chat-image-thumb').src = '';
    var btn = document.getElementById('tb-vision');
    if (btn) btn.classList.remove('active');
  },

  updateToolbar: function() {
    // Check /v1/features to enable/disable toolbar buttons based on model capabilities
    DN.api.get('/v1/features').then(function(f) {
      var vis = document.getElementById('tb-vision');
      var reas = document.getElementById('tb-reasoning');
      var tools = document.getElementById('tb-tools');
      if (vis) vis.style.opacity = f.vision ? '1' : '0.4';
      if (reas) reas.style.opacity = (f.reasoning || f.thinking) ? '1' : '0.4';
      if (tools) tools.style.opacity = f.tool_calling ? '1' : '0.4';
    }).catch(function() {});
  },

  send: function() {
    var input = document.getElementById('chat-input');
    var text = input.value.trim();
    if (!text || DN.chat.sending || !DN.health.model) return;

    input.value = '';
    input.style.height = 'auto';
    DN.chat.sending = true;
    document.getElementById('chat-send').disabled = true;

    function doSend() {
      DN.chat.appendMessage('user', text);
      DN.chat.scrollToBottom();

      // Create typing indicator
      var typingDiv = DN.chat.appendMessage('assistant', '');
      var contentEl = typingDiv.querySelector('.message-content');
      contentEl.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
      DN.chat.scrollToBottom();

      // Get settings
      var temp = parseFloat(document.getElementById('chat-temp').value) || 0.7;
      var maxTokens = parseInt(document.getElementById('chat-max-tokens').value) || 1024;

      // Collect messages from DOM
      var msgs = [];
      document.querySelectorAll('#chat-messages-inner .message').forEach(function(m) {
        if (m === typingDiv) return;
        var role = m.classList.contains('message-user') ? 'user' : 'assistant';
        var body = m.querySelector('.message-content');
        if (body) {
          // If this is the last user message and has an image, include it
          var content = body.textContent;
          msgs.push({ role: role, content: content });
        }
      });

      // Build request body with feature flags
      var reqBody = {
        model: 'deepnetz',
        messages: msgs,
        stream: true,
        max_tokens: maxTokens,
        temperature: temp,
        session_id: DN.chat.activeId || ''
      };

      // Add feature flags
      if (DN.chat.features.think) reqBody.think_mode = true;
      if (DN.chat.features.reasoning) reqBody.reasoning = true;
      if (DN.chat.features.web_search) reqBody.web_search = true;
      if (DN.chat.features.tool_call) reqBody.tool_call = true;

      // Add image data for vision
      if (DN.chat.imageData) {
        reqBody.images = [DN.chat.imageData];
        DN.chat.clearImage();
      }

      fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqBody)
      }).then(function(resp) {
        if (!resp.ok) {
          return resp.json().then(function(e) {
            throw new Error(e.error ? e.error.message : 'Server error');
          });
        }

        var reader = resp.body.getReader();
        var dec = new TextDecoder();
        var buf = '';
        var fullText = '';

        function read() {
          reader.read().then(function(r) {
            if (r.done) {
              DN.chat.sending = false;
              document.getElementById('chat-send').disabled = !DN.health.model;
              DN.chat.focus();
              if (!fullText) contentEl.innerHTML = '<span class="text-dim">No response received</span>';
              // Refresh sessions to get updated title
              DN.chat.loadSessions();
              return;
            }
            buf += dec.decode(r.value, { stream: true });
            var lines = buf.split('\n');
            buf = lines.pop() || '';
            for (var i = 0; i < lines.length; i++) {
              var line = lines[i].trim();
              if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                try {
                  var chunk = JSON.parse(line.slice(6));
                  var token = (chunk.choices && chunk.choices[0] && chunk.choices[0].delta)
                    ? chunk.choices[0].delta.content || '' : '';
                  if (token) {
                    fullText += token;
                    contentEl.innerHTML = DN.util.md(fullText);
                    DN.chat.scrollToBottom();
                  }
                } catch (e) {}
              }
            }
            read();
          });
        }
        read();
      }).catch(function(e) {
        contentEl.innerHTML = '<span class="text-red">' + DN.util.esc(e.message) + '</span>';
        DN.chat.sending = false;
        document.getElementById('chat-send').disabled = !DN.health.model;
      });
    }

    // Auto-create session if none active
    if (!DN.chat.activeId) {
      DN.api.post('/v1/sessions', { title: '' }).then(function(d) {
        DN.chat.activeId = d.id;
        DN.chat.allSessions.unshift({ id: d.id, title: text.slice(0, 40) + (text.length > 40 ? '...' : ''), created_at: d.created_at, message_count: 0 });
        DN.chat.sessions = DN.chat.allSessions.slice();
        DN.chat.renderList();
        doSend();
      }).catch(function(e) {
        DN.toast.show('Failed to create session', 'error');
        DN.chat.sending = false;
        document.getElementById('chat-send').disabled = false;
      });
    } else {
      doSend();
    }
  }
};

/* ================================================================
   MODELS
   ================================================================ */
DN.models = {
  initialized: false,
  hubDebounce: null,

  init: function() {
    DN.models.loadLocalModels();
    DN.models.loadActiveModel();
    if (!DN.models.initialized) {
      DN.models.loadHub('');
      DN.models.initialized = true;
    }
  },

  loadActiveModel: function() {
    var section = document.getElementById('loaded-model-section');
    var card = document.getElementById('loaded-model-card');
    if (!DN.health.model) {
      section.classList.add('hidden');
      return;
    }
    section.classList.remove('hidden');
    var name = DN.health.model.split('/').pop().replace(/\.gguf$/i, '');
    card.innerHTML =
      '<div class="model-card-header">' +
        '<div><div class="model-card-name">' + DN.util.esc(name) + '</div>' +
        '<div class="model-card-path">' + DN.util.esc(DN.health.model) + '</div></div>' +
        '<span class="badge badge-active">Active</span>' +
      '</div>' +
      '<div class="model-card-meta">' +
        '<span class="badge badge-' + DN.health.backend + '">' + DN.util.esc(DN.health.backend) + '</span>' +
      '</div>' +
      '<div class="model-card-actions">' +
        '<button class="btn btn-sm" onclick="DN.models.unload()" style="border-color:var(--red);color:var(--red);">' +
          '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg> Unload' +
        '</button>' +
      '</div>';
  },

  loadLocalModels: function() {
    DN.api.get('/v1/models').then(function(d) {
      var grid = document.getElementById('local-models-grid');
      var empty = document.getElementById('local-models-empty');
      var models = d.data || [];

      if (!models.length) {
        grid.innerHTML = '';
        grid.appendChild(empty);
        empty.classList.remove('hidden');
        return;
      }

      empty.classList.add('hidden');
      var html = models.map(function(m) {
        var isActive = DN.health.model && (m.path === DN.health.model || m.id === DN.health.model);
        var name = m.id.split('/').pop().replace(/\.gguf$/i, '');
        var size = DN.util.formatSize(m.size_mb);
        return '<div class="model-card' + (isActive ? ' model-active' : '') + '">' +
          '<div class="model-card-header">' +
            '<div class="model-card-name">' + DN.util.esc(name) + '</div>' +
            (isActive ? '<span class="badge badge-active">Active</span>' : '') +
          '</div>' +
          '<div class="model-card-meta">' +
            '<span class="badge badge-' + (m.backend || 'native') + '">' + DN.util.esc(m.backend || 'native') + '</span>' +
            (size ? '<span class="model-card-size">' + size + '</span>' : '') +
          '</div>' +
          '<div class="model-card-path">' + DN.util.esc(m.path || m.id) + '</div>' +
          '<div class="model-card-actions">' +
            (isActive
              ? '<button class="btn btn-sm" onclick="DN.models.unload()" style="border-color:var(--red);color:var(--red);">Unload</button>'
              : '<button class="btn btn-sm btn-primary" onclick="DN.models.loadModel(\'' + DN.util.esc(m.path || m.id).replace(/'/g, "\\'") + '\',\'' + DN.util.esc(m.backend || '') + '\')">Load</button>'
            ) +
          '</div>' +
        '</div>';
      }).join('');
      grid.innerHTML = html;
    }).catch(function() {});
  },

  searchHub: DN.util.debounce(function(query) {
    DN.models.loadHub(query);
  }, 350),

  loadHub: function(query) {
    var grid = document.getElementById('hub-models-grid');
    var loading = document.getElementById('hub-loading');
    loading.classList.remove('hidden');
    grid.innerHTML = '';

    var url = '/v1/cards/search?limit=24';
    if (query) url += '&q=' + encodeURIComponent(query);

    DN.api.get(url).then(function(d) {
      loading.classList.add('hidden');
      var cards = d.cards || [];
      if (!cards.length) {
        grid.innerHTML = '<div class="empty-state"><p>No models found</p></div>';
        return;
      }
      grid.innerHTML = cards.map(function(c) {
        var params = c.params_b ? c.params_b.toFixed(1) + 'B' : '';
        var dl = c.downloads > 1000 ? Math.round(c.downloads / 1000) + 'k' : (c.downloads || 0);
        var tags = (c.tags || []).slice(0, 4).map(function(t) {
          return '<span class="model-tag">' + DN.util.esc(t) + '</span>';
        }).join('');
        var desc = c.architecture || '';
        var quants = (c.quants || []).length;

        return '<div class="model-card model-card-hub">' +
          '<div class="model-card-header">' +
            '<div class="model-card-name">' + DN.util.esc(c.name) + '</div>' +
          '</div>' +
          '<div class="model-card-meta">' +
            (params ? '<span class="model-card-size">' + params + ' params</span>' : '') +
            '<span class="model-card-size">' + desc + '</span>' +
            '<span class="model-card-size">' + dl + ' downloads</span>' +
          '</div>' +
          '<div class="model-card-tags">' + tags + '</div>' +
          (quants > 0 ? '<div class="model-card-quants">' + quants + ' quantizations available</div>' : '') +
          '<div class="model-card-actions">' +
            (c.hf_repos && c.hf_repos.length > 0
              ? '<button class="btn btn-sm btn-primary" onclick="DN.models.pullModel(\'' + DN.util.esc(c.hf_repos[0]).replace(/'/g, "\\'") + '\', this)"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Pull</button>'
              : '<span class="model-card-size">Use CLI: deepnetz pull ' + DN.util.esc(c.name) + '</span>'
            ) +
          '</div>' +
        '</div>';
      }).join('');
    }).catch(function() {
      loading.classList.add('hidden');
      grid.innerHTML = '<div class="empty-state"><p>Failed to load model cards</p></div>';
    });
  },

  loadModel: function(path, backend) {
    DN.toast.show('Loading model...', 'info', 10000);
    DN.health.loading = true;
    DN.health.updateUI();

    DN.api.post('/v1/models/load', { model: path, backend: backend || '' }).then(function(d) {
      if (d.status === 'error') {
        DN.toast.show('Load failed: ' + (d.error || 'Unknown error'), 'error', 5000);
      } else {
        DN.toast.show('Model loaded successfully', 'success');
      }
      DN.health.check();
      setTimeout(function() { DN.models.init(); }, 500);
    }).catch(function(e) {
      DN.toast.show('Load failed: ' + e.message, 'error', 5000);
      DN.health.loading = false;
      DN.health.updateUI();
    });
  },

  unload: function() {
    DN.api.post('/v1/models/unload', {}).then(function() {
      DN.toast.show('Model unloaded', 'success');
      DN.health.model = '';
      DN.health.backend = 'none';
      DN.health.updateUI();
      DN.models.init();
    }).catch(function(e) {
      DN.toast.show('Unload failed: ' + e.message, 'error');
    });
  },

  pullModel: function(repo, btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Pulling...';
    DN.toast.show('Downloading model from HuggingFace...', 'info', 15000);

    DN.api.post('/v1/models/download', { model: repo }).then(function(d) {
      if (d.status === 'error') {
        DN.toast.show('Download failed: ' + (d.error || 'Unknown'), 'error', 5000);
        btn.disabled = false;
        btn.textContent = 'Pull';
      } else {
        DN.toast.show('Model downloaded', 'success');
        btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Done';
        btn.className = 'btn btn-sm btn-success';
        DN.models.loadLocalModels();
      }
    }).catch(function(e) {
      DN.toast.show('Download failed: ' + e.message, 'error', 5000);
      btn.disabled = false;
      btn.textContent = 'Pull';
    });
  }
};

/* ================================================================
   MONITOR
   ================================================================ */
DN.monitor = {
  ws: null,
  tpsHistory: [],
  active: false,

  start: function() {
    DN.monitor.active = true;
    if (DN.monitor.ws && DN.monitor.ws.readyState === WebSocket.OPEN) return;

    // Load hardware info
    DN.api.get('/v1/hardware').then(function(d) {
      var gpuName = (d.gpus && d.gpus.length > 0) ? d.gpus[0].name : 'No GPU detected';
      document.getElementById('mon-gpu-name').textContent = gpuName;
      document.getElementById('mon-cpu-cores').textContent = (d.cpu_cores || '?') + ' cores';
    }).catch(function() {});

    // Load model info
    DN.api.get('/health').then(function(d) {
      document.getElementById('mon-model-name').textContent = d.model ? d.model.split('/').pop().replace(/\.gguf$/i, '') : 'No model loaded';
      document.getElementById('mon-model-meta').textContent = d.model ? 'Backend: ' + d.backend : '';
    }).catch(function() {});

    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    try {
      DN.monitor.ws = new WebSocket(proto + '//' + location.host + '/ws/monitor');
    } catch (e) { return; }

    DN.monitor.ws.onmessage = function(e) {
      try {
        var d = JSON.parse(e.data);
        DN.monitor.update(d);
      } catch (err) {}
    };

    DN.monitor.ws.onclose = function() {
      DN.monitor.ws = null;
      // Reconnect if still on monitor page
      if (DN.monitor.active && DN.router.current === 'monitor') {
        setTimeout(function() { DN.monitor.start(); }, 3000);
      }
    };

    DN.monitor.ws.onerror = function() {
      DN.monitor.ws = null;
    };
  },

  stop: function() {
    DN.monitor.active = false;
    if (DN.monitor.ws) {
      DN.monitor.ws.close();
      DN.monitor.ws = null;
    }
  },

  update: function(d) {
    // GPU
    var gpu = d.gpu || {};
    var gpuPct = gpu.util_percent || 0;
    document.getElementById('mon-gpu-pct').textContent = Math.round(gpuPct) + '%';
    var gpuBar = document.getElementById('mon-gpu-bar');
    gpuBar.style.width = gpuPct + '%';

    // GPU Temp
    var temp = gpu.temp_c || 0;
    var tempText = temp > 0 ? temp + '\u00B0C' : '-- \u00B0C';
    document.getElementById('mon-gpu-temp').textContent = tempText;

    // VRAM
    var vramUsed = (gpu.vram_used_mb || 0) / 1024;
    var vramTotal = (gpu.vram_total_mb || 0) / 1024;
    document.getElementById('mon-vram-val').textContent = vramUsed.toFixed(1) + ' / ' + vramTotal.toFixed(1) + ' GB';
    var vramPct = vramTotal > 0 ? (vramUsed / vramTotal * 100) : 0;
    document.getElementById('mon-vram-bar').style.width = vramPct + '%';

    // CPU
    var cpu = d.cpu || {};
    var cpuPct = cpu.percent || 0;
    document.getElementById('mon-cpu-pct').textContent = Math.round(cpuPct) + '%';
    document.getElementById('mon-cpu-bar').style.width = cpuPct + '%';

    // RAM
    var ram = d.ram || {};
    var ramUsed = (ram.used_mb || 0) / 1024;
    var ramTotal = (ram.total_mb || 0) / 1024;
    document.getElementById('mon-ram-val').textContent = ramUsed.toFixed(1) + ' / ' + ramTotal.toFixed(1) + ' GB';
    var ramPct = ram.percent || 0;
    document.getElementById('mon-ram-pct').textContent = Math.round(ramPct) + '%';
    document.getElementById('mon-ram-bar').style.width = ramPct + '%';

    // TPS (Inference speed)
    var inf = d.inference || {};
    var tps = inf.tps || 0;
    document.getElementById('mon-tps-value').textContent = tps.toFixed(1);

    // TPS History + Chart
    DN.monitor.tpsHistory.push(tps);
    if (DN.monitor.tpsHistory.length > 60) DN.monitor.tpsHistory.shift();
    DN.monitor.drawChart();
  },

  drawChart: function() {
    var canvas = document.getElementById('tps-canvas');
    if (!canvas) return;

    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    var ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    var w = rect.width;
    var h = rect.height;
    var data = DN.monitor.tpsHistory;
    var max = Math.max.apply(null, data.concat([1]));
    var padding = { top: 10, bottom: 25, left: 45, right: 10 };
    var chartW = w - padding.left - padding.right;
    var chartH = h - padding.top - padding.bottom;

    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(42, 42, 58, 0.6)';
    ctx.lineWidth = 1;
    for (var g = 0; g <= 4; g++) {
      var gy = padding.top + (chartH / 4) * g;
      ctx.beginPath();
      ctx.moveTo(padding.left, gy);
      ctx.lineTo(w - padding.right, gy);
      ctx.stroke();

      // Labels
      var val = max - (max / 4) * g;
      ctx.fillStyle = '#6a6a80';
      ctx.font = '11px system-ui';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(1), padding.left - 6, gy + 4);
    }

    // X-axis labels
    ctx.fillStyle = '#6a6a80';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('-60s', padding.left, h - 4);
    ctx.fillText('-30s', padding.left + chartW / 2, h - 4);
    ctx.fillText('now', w - padding.right, h - 4);

    if (data.length < 2) return;

    // Gradient fill
    var gradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartH);
    gradient.addColorStop(0, 'rgba(99, 102, 241, 0.3)');
    gradient.addColorStop(1, 'rgba(99, 102, 241, 0.02)');

    // Draw filled area
    ctx.beginPath();
    for (var i = 0; i < data.length; i++) {
      var x = padding.left + (i / 59) * chartW;
      var y = padding.top + chartH - (data[i] / max) * chartH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.lineTo(padding.left + ((data.length - 1) / 59) * chartW, padding.top + chartH);
    ctx.lineTo(padding.left, padding.top + chartH);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw line
    ctx.beginPath();
    for (var j = 0; j < data.length; j++) {
      var lx = padding.left + (j / 59) * chartW;
      var ly = padding.top + chartH - (data[j] / max) * chartH;
      if (j === 0) ctx.moveTo(lx, ly); else ctx.lineTo(lx, ly);
    }
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw current value dot
    if (data.length > 0) {
      var lastX = padding.left + ((data.length - 1) / 59) * chartW;
      var lastY = padding.top + chartH - (data[data.length - 1] / max) * chartH;
      ctx.beginPath();
      ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#6366f1';
      ctx.fill();
      ctx.strokeStyle = '#0a0a0f';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
};

/* ================================================================
   SETTINGS
   ================================================================ */
DN.settings = {
  init: function() {
    // Load backends
    DN.api.get('/v1/backends').then(function(backends) {
      var sel = document.getElementById('set-backend');
      sel.innerHTML = '<option value="auto">Auto</option>';
      backends.forEach(function(b) {
        if (b.available) {
          var opt = document.createElement('option');
          opt.value = b.name;
          opt.textContent = b.name;
          sel.appendChild(opt);
        }
      });
    }).catch(function() {});

    // Load current config
    DN.api.get('/v1/config').then(function(d) {
      if (d.default_backend && d.default_backend !== 'auto') {
        document.getElementById('set-backend').value = d.default_backend;
        document.getElementById('set-backend-val').textContent = d.default_backend;
      }
      if (d.target_context) {
        document.getElementById('set-ctx').value = d.target_context;
        document.getElementById('set-ctx-val').textContent = d.target_context;
      }
    }).catch(function() {});

    // Load hardware + server info
    DN.api.get('/health').then(function(d) {
      document.getElementById('info-status').textContent = d.loading ? 'Loading...' : (d.model ? 'Running' : 'Idle');
      document.getElementById('info-status').className = 'settings-info-value' + (d.model ? ' text-green' : '');
      document.getElementById('info-model').textContent = d.model || 'None';
      document.getElementById('info-backend').textContent = d.backend || '--';
    }).catch(function() {});

    DN.api.get('/v1/hardware').then(function(d) {
      document.getElementById('info-gpu').textContent = (d.gpus && d.gpus.length > 0)
        ? d.gpus[0].name + ' (' + DN.util.formatSize(d.gpus[0].vram_mb) + ')'
        : 'No GPU';
      document.getElementById('info-ram').textContent = DN.util.formatSize(d.ram_mb);
      document.getElementById('info-os').textContent = d.os || '--';
    }).catch(function() {});
  },

  save: function() {
    // Currently config is read-only via API, but we store settings locally
    var settings = {
      backend: document.getElementById('set-backend').value,
      gpu_layers: parseInt(document.getElementById('set-gpu-layers').value),
      context_length: parseInt(document.getElementById('set-ctx').value),
      kv_cache: document.getElementById('set-kv').value,
      temperature: parseFloat(document.getElementById('set-temp').value),
      top_p: parseFloat(document.getElementById('set-topp').value),
      top_k: parseInt(document.getElementById('set-topk').value),
      repeat_penalty: parseFloat(document.getElementById('set-repeat').value),
      max_tokens: parseInt(document.getElementById('set-maxtok').value)
    };
    localStorage.setItem('deepnetz_settings', JSON.stringify(settings));

    // Also update chat settings
    document.getElementById('chat-temp').value = settings.temperature;
    document.getElementById('chat-temp-val').textContent = settings.temperature;
    document.getElementById('chat-max-tokens').value = settings.max_tokens;

    DN.toast.show('Settings saved', 'success');
  },

  load: function() {
    try {
      var saved = JSON.parse(localStorage.getItem('deepnetz_settings') || '{}');
      if (saved.temperature !== undefined) {
        document.getElementById('set-temp').value = saved.temperature;
        document.getElementById('set-temp-val').textContent = saved.temperature;
        document.getElementById('chat-temp').value = saved.temperature;
        document.getElementById('chat-temp-val').textContent = saved.temperature;
      }
      if (saved.max_tokens !== undefined) {
        document.getElementById('set-maxtok').value = saved.max_tokens;
        document.getElementById('chat-max-tokens').value = saved.max_tokens;
      }
      if (saved.top_p !== undefined) {
        document.getElementById('set-topp').value = saved.top_p;
        document.getElementById('set-topp-val').textContent = saved.top_p;
      }
      if (saved.top_k !== undefined) {
        document.getElementById('set-topk').value = saved.top_k;
        document.getElementById('set-topk-val').textContent = saved.top_k;
      }
      if (saved.repeat_penalty !== undefined) {
        document.getElementById('set-repeat').value = saved.repeat_penalty;
        document.getElementById('set-repeat-val').textContent = saved.repeat_penalty;
      }
      if (saved.gpu_layers !== undefined) {
        document.getElementById('set-gpu-layers').value = saved.gpu_layers;
        document.getElementById('set-gpu-layers-val').textContent = saved.gpu_layers;
      }
      if (saved.context_length !== undefined) {
        document.getElementById('set-ctx').value = saved.context_length;
        document.getElementById('set-ctx-val').textContent = saved.context_length;
      }
    } catch (e) {}
  }
};

/* ================================================================
   AUTH
   ================================================================ */
DN.auth = {
  check: function() {
    // Check localStorage first
    var cfg = null;
    try { cfg = JSON.parse(localStorage.getItem('deepnetz_cfg')); } catch (e) {}

    if (cfg && cfg.username) {
      DN.auth.showLoggedIn(cfg.username);
      return;
    }

    // Check server
    DN.api.get('/v1/auth/status').then(function(d) {
      if (d.logged_in) {
        DN.auth.showLoggedIn(d.api_key_prefix || 'Logged in');
      }
    }).catch(function() {});
  },

  showLoggedIn: function(name) {
    document.getElementById('auth-login-btn').classList.add('hidden');
    var profile = document.getElementById('auth-profile');
    profile.classList.remove('hidden');
    document.getElementById('auth-name').textContent = name;
  },

  logout: function() {
    localStorage.removeItem('deepnetz_cfg');
    document.getElementById('auth-login-btn').classList.remove('hidden');
    document.getElementById('auth-profile').classList.add('hidden');
    DN.toast.show('Logged out', 'info');
  }
};

/* ================================================================
   INIT
   ================================================================ */
document.addEventListener('DOMContentLoaded', function() {
  DN.settings.load();
  DN.router.init();
  DN.chat.init();
  DN.auth.check();
  DN.health.check();

  // Periodic health check
  setInterval(DN.health.check, 5000);

  // Handle window resize for chart
  window.addEventListener('resize', DN.util.debounce(function() {
    if (DN.router.current === 'monitor') DN.monitor.drawChart();
  }, 200));
});
