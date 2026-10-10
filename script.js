"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const messagesEl = $("messages");
  const form = $("chat-form");
  const input = $("input");
  const sendBtn = $("send-btn");
  const errorEl = $("form-error");
  const categorySel = $("category");
  const suggestionsEl = $("suggestions");

  const STORAGE_KEY = "faq-chat-history";
  const THEME_KEY = "faq-theme";
  const WELCOME =
    "Hi! I answer common questions about technology, programming, AI basics, education and student life. " +
    "Ask me something or pick a suggestion below.";

  const WELCOME_TIME = new Date().toISOString();
  let history = []; // [{role, text, time, faqId?, matched?, confidence?, related?}]
  let busy = false;

  // ---------- storage (sessionStorage = browser session only) ----------
  function save() {
    try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(history)); } catch (_) { /* ignore */ }
  }
  function load() {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (_) { return []; }
  }

  // ---------- helpers ----------
  const timeLabel = (iso) =>
    new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  function showError(text) {
    errorEl.textContent = text;
    errorEl.hidden = !text;
  }
  function scrollDown() { messagesEl.scrollTop = messagesEl.scrollHeight; }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text; // textContent: never interprets HTML
    return node;
  }

  function chip(question) {
    const b = el("button", "chip", question);
    b.type = "button";
    b.addEventListener("click", () => send(question));
    return b;
  }

  // ---------- rendering ----------
  function renderMessage(m, isLatest) {
    const wrap = el("div", `msg ${m.role}${m.matched === false ? " unmatched" : ""}${m.error ? " error" : ""}`);
    wrap.appendChild(el("div", "bubble", m.text));

    const meta = el("div", "meta");
    meta.appendChild(el("span", "", timeLabel(m.time)));
    if (m.role === "bot" && m.matched && typeof m.confidence === "number") {
      meta.appendChild(el("span", "", `${m.category} · ${Math.round(m.confidence * 100)}% match`));
    }
    wrap.appendChild(meta);

    if (m.role === "bot" && m.matched && m.faqId) {
      const fb = el("div", "feedback");
      fb.appendChild(el("span", "meta", "Helpful?"));
      ["👍", "👎"].forEach((icon, i) => {
        const b = el("button", "", icon);
        b.type = "button";
        b.setAttribute("aria-label", i === 0 ? "Helpful" : "Not helpful");
        b.disabled = !!m.rated;
        b.addEventListener("click", () => rate(m, i === 0, fb));
        fb.appendChild(b);
      });
      wrap.appendChild(fb);
    }

    if (isLatest && m.role === "bot" && m.related && m.related.length) {
      const chips = el("div", "chips");
      m.related.forEach((r) => chips.appendChild(chip(r.question)));
      wrap.appendChild(chips);
    }
    return wrap;
  }

  function render() {
    messagesEl.replaceChildren();
    const welcome = { role: "bot", text: WELCOME, time: WELCOME_TIME };
    messagesEl.appendChild(renderMessage(welcome, false));
    const lastBotIndex = history.map((m) => m.role).lastIndexOf("bot");
    history.forEach((m, i) => messagesEl.appendChild(renderMessage(m, i === lastBotIndex)));
    suggestionsEl.hidden = history.length > 0;
    scrollDown();
  }

  function showTyping() {
    const wrap = el("div", "msg bot typing");
    wrap.id = "typing";
    const bubble = el("div", "bubble");
    for (let i = 0; i < 3; i++) bubble.appendChild(el("span", "dot"));
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollDown();
  }
  const hideTyping = () => { const t = $("typing"); if (t) t.remove(); };

  // ---------- API ----------
  async function api(path, options) {
    const res = await fetch(path, options);
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON response */ }
    if (!res.ok) throw new Error((data && data.error) || "Something went wrong. Please try again.");
    return data;
  }

  async function send(question) {
    question = (question || "").trim();
    if (busy) return;
    if (!question) { showError("Please type a question."); return; }
    showError("");
    busy = true;
    sendBtn.disabled = true;
    input.value = "";

    history.push({ role: "user", text: question, time: new Date().toISOString() });
    render();
    showTyping();

    try {
      const data = await api("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, category: categorySel.value || null }),
      });
      hideTyping();
      history.push({
        role: "bot", text: data.answer, time: new Date().toISOString(),
        faqId: data.faq_id, matched: data.matched, confidence: data.confidence,
        category: data.category, related: data.related,
      });
    } catch (err) {
      hideTyping();
      const msg = err instanceof TypeError ? "Can't reach the server. Check your connection and try again." : err.message;
      history.push({ role: "bot", text: msg, time: new Date().toISOString(), error: true });
    } finally {
      busy = false;
      sendBtn.disabled = false;
      save();
      render();
      input.focus();
    }
  }

  async function rate(message, helpful, container) {
    try {
      await api("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ faq_id: message.faqId, helpful }),
      });
      message.rated = true;
      save();
      container.querySelectorAll("button").forEach((b) => (b.disabled = true));
    } catch (err) { showError(err.message); }
  }

  // ---------- controls ----------
  form.addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });

  $("clear-btn").addEventListener("click", () => {
    history = [];
    save();
    showError("");
    render();
    input.focus();
  });

  $("export-btn").addEventListener("click", () => {
    if (!history.length) { showError("Nothing to export yet."); return; }
    showError("");
    const lines = history.map((m) => `[${new Date(m.time).toLocaleString()}] ${m.role === "user" ? "You" : "Bot"}: ${m.text}`);
    const blob = new Blob([lines.join("\n\n") + "\n"], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "faq-conversation.txt";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem(THEME_KEY, theme); } catch (_) { /* ignore */ }
  }
  $("theme-btn").addEventListener("click", () =>
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

  const dialog = $("stats-dialog");
  $("stats-btn").addEventListener("click", async () => {
    const body = $("stats-body");
    body.replaceChildren(el("p", "muted", "Loading…"));
    dialog.showModal();
    try {
      const s = await api("/api/stats");
      const dl = el("dl");
      const rows = [
        ["Questions asked", s.total_questions],
        ["Answered", s.answered],
        ["Not answered", s.unanswered],
        ["Answer rate", s.answer_rate === null ? "–" : `${Math.round(s.answer_rate * 100)}%`],
        ["👍 / 👎", `${s.feedback_helpful} / ${s.feedback_not_helpful}`],
      ];
      rows.forEach(([k, v]) => { dl.appendChild(el("dt", "", k)); dl.appendChild(el("dd", "", String(v))); });
      body.replaceChildren(dl);
      if (s.top_questions.length) {
        body.appendChild(el("strong", "", "Most asked"));
        const ol = el("ol");
        s.top_questions.forEach((q) => ol.appendChild(el("li", "", `${q.question} (${q.count})`)));
        body.appendChild(ol);
      }
    } catch (err) { body.replaceChildren(el("p", "form-error", err.message)); }
  });
  $("stats-close").addEventListener("click", () => dialog.close());

  // ---------- startup ----------
  async function init() {
    let theme = "light";
    try {
      theme = localStorage.getItem(THEME_KEY) ||
        (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    } catch (_) { /* ignore */ }
    applyTheme(theme);

    history = load();
    render();

    try {
      const cfg = await api("/api/config");
      cfg.categories.forEach((c) => {
        const o = el("option", "", c);
        o.value = c;
        categorySel.appendChild(o);
      });
      suggestionsEl.replaceChildren(el("p", "label", "Try asking:"));
      const chips = el("div", "chips");
      cfg.suggestions.forEach((s) => chips.appendChild(chip(s.question)));
      suggestionsEl.appendChild(chips);
    } catch (err) {
      showError(err.message);
      sendBtn.disabled = true;
    }
  }
  init();
})();
