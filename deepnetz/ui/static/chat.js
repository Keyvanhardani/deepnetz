// DeepNetz Chat — streaming chat with the loaded model

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
let conversationHistory = [];

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';

  // Add user message
  conversationHistory.push({ role: 'user', content: text });
  appendMessage('user', text);

  // Create assistant message placeholder
  const assistantEl = appendMessage('assistant', '');
  let fullResponse = '';

  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepnetz',
        messages: conversationHistory,
        stream: true,
        max_tokens: 1024,
        temperature: 0.7,
      }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
          try {
            const chunk = JSON.parse(line.slice(6));
            const content = chunk.choices?.[0]?.delta?.content || '';
            if (content) {
              fullResponse += content;
              assistantEl.querySelector('.msg-content').textContent = fullResponse;
              messagesEl.scrollTop = messagesEl.scrollHeight;
            }
          } catch (e) {}
        }
      }
    }

    conversationHistory.push({ role: 'assistant', content: fullResponse });
  } catch (error) {
    assistantEl.querySelector('.msg-content').textContent = `Error: ${error.message}`;
  }
}

function appendMessage(role, content) {
  const div = document.createElement('div');
  div.className = `msg msg-${role}`;
  div.innerHTML = `<div class="msg-content">${escapeHtml(content)}</div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
