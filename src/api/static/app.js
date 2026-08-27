// Chat UI. Talks to the same JSON API documented in src/api/README.md; the only
// endpoint unique to the browser is POST /chat/stream.

import { renderMarkdown } from "./markdown.js";

const $ = (id) => document.getElementById(id);

const els = {
  userId: $("user-id"),
  newThread: $("new-thread"),
  threadList: $("thread-list"),
  threadEmpty: $("thread-empty"),
  factList: $("fact-list"),
  factEmpty: $("fact-empty"),
  threadLabel: $("thread-label"),
  messages: $("messages"),
  emptyState: $("empty-state"),
  composer: $("composer"),
  input: $("input"),
  send: $("send"),
  forget: $("forget"),
  health: $("health"),
  themeToggle: $("theme-toggle"),
  menu: $("menu"),
  sidebar: $("sidebar"),
  scrim: $("scrim"),
};

const store = {
  get(key, fallback) {
    try {
      return localStorage.getItem(key) ?? fallback;
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      value === null ? localStorage.removeItem(key) : localStorage.setItem(key, value);
    } catch {
      /* private mode: the UI still works, it just forgets on reload */
    }
  },
};

const state = {
  userId: store.get("cairn.user", `u_${Math.random().toString(36).slice(2, 10)}`),
  threadId: store.get("cairn.thread", null),
  busy: false,
};

// --- api ---------------------------------------------------------------------

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`${options.method || "GET"} ${path} → ${response.status}`);
  }
  return response.json();
}

// --- rendering ---------------------------------------------------------------

const clone = (id) => $(id).content.firstElementChild.cloneNode(true);

function citationChip(number) {
  const sup = clone("tpl-marker");
  sup.textContent = number;
  return sup;
}

// The markdown an answer was rendered from. Copying that rather than the
// rendered text keeps list markers, fences and [S1] citations intact.
const answerSource = new WeakMap();

/** Render answer markdown, turning [S1] markers into superscript chips. */
function renderAnswer(target, text) {
  answerSource.set(target, text);
  renderMarkdown(target, text, citationChip);
  addCodeToolbars(target);
}

/** Wrap each rendered code block in its copy/download toolbar. */
function addCodeToolbars(root) {
  for (const pre of root.querySelectorAll("pre")) {
    const figure = clone("tpl-code");
    pre.replaceWith(figure);
    figure.append(pre);
    figure.querySelector("[data-lang]").textContent =
      pre.querySelector("code")?.dataset.lang || "text";
  }
}

function renderSources(target, citations) {
  target.replaceChildren();
  if (!citations) return; // restored from history: citations are derived per response
  if (citations.length === 0) {
    target.append(clone("tpl-ungrounded"));
    return;
  }
  for (const citation of citations) {
    const pill = clone("tpl-source");
    pill.querySelector("[data-source]").textContent = citation.source;
    pill.querySelector("[data-score]").textContent = citation.score.toFixed(2);
    target.append(pill);
  }
}

/** Emptying the transcript puts the prompts back, rather than a blank pane. */
function clearTranscript() {
  els.messages.replaceChildren(els.emptyState);
}

function addUserMessage(text) {
  els.emptyState.remove();
  const node = clone("tpl-user");
  node.firstElementChild.textContent = text;
  els.messages.append(node);
  scrollToBottom();
}

/** Returns handles so a streaming turn can keep writing into the same bubble. */
function addAssistantMessage() {
  els.emptyState.remove();
  const node = clone("tpl-assistant");
  const answer = node.querySelector("[data-answer]");
  const sources = node.querySelector("[data-sources]");
  els.messages.append(node);
  scrollToBottom();
  return { answer, sources };
}

function scrollToBottom(behavior = "smooth") {
  els.messages.scrollTo({ top: els.messages.scrollHeight, behavior });
}

// --- copy and download -------------------------------------------------------

async function copyText(value) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch {
    // Permission refused, or not focused. Fall through to the old path.
  }
  // navigator.clipboard needs a secure context, which this is not when the app
  // is reached on a LAN address rather than localhost.
  const area = document.createElement("textarea");
  area.value = value;
  area.setAttribute("readonly", "");
  area.style.cssText = "position:fixed;top:0;opacity:0";
  document.body.append(area);
  area.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch {
    copied = false;
  }
  area.remove();
  return copied;
}

const EXTENSIONS = {
  bash: "sh",
  c: "c",
  cpp: "cpp",
  css: "css",
  go: "go",
  html: "html",
  java: "java",
  javascript: "js",
  js: "js",
  json: "json",
  jsx: "jsx",
  markdown: "md",
  md: "md",
  python: "py",
  rust: "rs",
  sh: "sh",
  shell: "sh",
  sql: "sql",
  toml: "toml",
  ts: "ts",
  tsx: "tsx",
  typescript: "ts",
  yaml: "yml",
  yml: "yml",
};

function downloadText(value, lang) {
  const url = URL.createObjectURL(new Blob([value], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `cairn-snippet.${EXTENSIONS[lang?.toLowerCase()] || "txt"}`;
  document.body.append(link);
  link.click();
  link.remove();
  // Revoking in the same tick can cancel the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** Confirm in the button itself; a toast for this is more noise than signal. */
function flash(button, message) {
  const label = button.querySelector("[data-label]");
  if (!label || button.dataset.flash !== undefined) return;
  const original = label.textContent;
  label.textContent = message;
  button.dataset.flash = "";
  setTimeout(() => {
    label.textContent = original;
    delete button.dataset.flash;
  }, 1400);
}

function toast(message) {
  const node = clone("tpl-toast");
  node.textContent = message;
  document.body.append(node);
  setTimeout(() => node.remove(), 2600);
}

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  els.input.disabled = busy;
}

// --- the turn ----------------------------------------------------------------

/**
 * One turn, streamed. A provider that does not stream degrades to a single token
 * event carrying the whole answer, which needs no separate path here.
 */
async function send(message) {
  if (state.busy) return;
  if (!state.threadId) {
    try {
      await newThread({ silent: true });
    } catch {
      toast("could not start a conversation");
      return;
    }
  }

  setBusy(true);
  addUserMessage(message);

  const typing = clone("tpl-typing");
  els.messages.append(typing);
  scrollToBottom();

  let bubble = null;
  let streamed = "";
  let frame = 0;

  const open = () => {
    if (!bubble) {
      typing.remove();
      bubble = addAssistantMessage();
    }
    return bubble;
  };

  // Tokens arrive faster than the browser paints, and each one reparses the
  // whole answer. Coalescing to a frame keeps that off the critical path; the
  // frame reads `streamed` when it runs, so it always paints the latest text.
  const write = () => {
    open();
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      renderAnswer(bubble.answer, streamed);
      scrollToBottom("auto");
    });
  };

  const stopWriting = () => {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
  };

  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        user_id: state.userId,
        thread_id: state.threadId,
        message,
      }),
    });
    if (!response.ok) {
      const detail = await response
        .json()
        .then((b) => b.detail)
        .catch(() => null);
      throw new Error(detail || `The server returned ${response.status}.`);
    }

    for await (const event of readEvents(response)) {
      if (event.type === "token") {
        streamed += event.text;
        write();
      } else if (event.type === "restart") {
        // generate → clarify: the draft could not be cited, so it is not
        // shippable as grounded. Drop what was drawn rather than leave it up.
        streamed = "";
        write();
      } else if (event.type === "final") {
        // Wins over any frame still queued, which would repaint stale text.
        stopWriting();
        renderAnswer(open().answer, event.answer);
        renderSources(bubble.sources, event.citations);
        scrollToBottom();
      } else if (event.type === "error") {
        throw new Error(event.detail);
      }
    }
  } catch (error) {
    stopWriting();
    typing.remove();
    // Drop the partial answer: half a turn is not an answer. The reason stays on
    // screen, because it is usually something the reader has to go and fix.
    if (bubble) bubble.answer.closest("article").remove();
    const node = clone("tpl-error");
    node.querySelector("[data-detail]").textContent = error.message || "The request failed.";
    els.messages.append(node);
    scrollToBottom();
  } finally {
    typing.remove();
    setBusy(false);
    els.input.focus();
    refreshSidebar();
  }
}

/** Parse an SSE body into decoded event objects. */
async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // A chunk can split mid-event; only complete \n\n-terminated blocks parse.
    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const payload = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("");
      if (payload) yield JSON.parse(payload);
    }
  }
}

// --- threads -----------------------------------------------------------------

async function newThread({ silent = false } = {}) {
  const { thread_id } = await api("/threads", { method: "POST" });
  state.threadId = thread_id;
  store.set("cairn.thread", thread_id);
  els.threadLabel.textContent = thread_id;
  clearTranscript();
  if (!silent) toast("new conversation");
  markCurrentThread();
}

async function openThread(threadId) {
  state.threadId = threadId;
  store.set("cairn.thread", threadId);
  els.threadLabel.textContent = threadId;
  clearTranscript();
  markCurrentThread();

  try {
    const { messages } = await api(`/threads/${encodeURIComponent(threadId)}/history`);
    for (const message of messages) {
      if (message.role === "human") {
        addUserMessage(message.content);
      } else {
        const bubble = addAssistantMessage();
        renderAnswer(bubble.answer, message.content);
        renderSources(bubble.sources, null);
      }
    }
  } catch {
    // 404: the thread was minted but never took a turn, so it has no checkpoint.
  }
}

function markCurrentThread() {
  for (const button of els.threadList.querySelectorAll("[data-open-thread]")) {
    button.setAttribute("aria-current", String(button.dataset.thread === state.threadId));
  }
}

/** Delete one conversation. Durable facts survive it -- they are user-scoped. */
async function deleteThread(threadId) {
  const confirmed = confirm(
    `Delete conversation ${threadId}?\n\n` +
      "Its messages are erased. What cairn remembers about you is kept.",
  );
  if (!confirmed) return;

  try {
    await api(
      `/users/${encodeURIComponent(state.userId)}/threads/${encodeURIComponent(threadId)}`,
      { method: "DELETE" },
    );
  } catch (error) {
    toast(error.message);
    return;
  }

  if (state.threadId === threadId) {
    state.threadId = null;
    store.set("cairn.thread", null);
    clearTranscript();
    els.threadLabel.textContent = "no conversation yet";
  }
  toast("conversation deleted");
  refreshSidebar();
}

// --- sidebar -----------------------------------------------------------------

async function refreshSidebar() {
  const user = encodeURIComponent(state.userId);
  const [threads, facts] = await Promise.all([
    api(`/users/${user}/threads`).catch(() => ({ threads: [] })),
    api(`/users/${user}/facts`).catch(() => ({ facts: [] })),
  ]);

  els.threadList.replaceChildren();
  for (const threadId of threads.threads.slice().reverse()) {
    const item = clone("tpl-thread");
    const open = item.querySelector("[data-open-thread]");
    open.textContent = threadId;
    open.dataset.thread = threadId;
    open.addEventListener("click", () => {
      openThread(threadId);
      closeSidebar();
    });
    item.querySelector("[data-delete-thread]").addEventListener("click", () => {
      deleteThread(threadId);
    });
    els.threadList.append(item);
  }
  els.threadEmpty.hidden = threads.threads.length > 0;
  markCurrentThread();

  els.factList.replaceChildren();
  for (const fact of facts.facts) {
    const item = clone("tpl-fact");
    item.textContent = fact;
    els.factList.append(item);
  }
  els.factEmpty.hidden = facts.facts.length > 0;
}

async function refreshHealth() {
  try {
    const { env, llm_provider, llm_reachable } = await api("/health");
    // The API being up says nothing about the model being up. Green here used to
    // mean "cairn is running" while every turn was failing.
    const down = llm_reachable === false;
    els.health.replaceChildren();
    const dot = document.createElement("span");
    dot.className = `size-1.5 rounded-full ${down ? "bg-amber-500" : "bg-emerald-500"}`;
    const label = `${env} · ${llm_provider}${down ? " unreachable" : ""}`;
    els.health.append(dot, document.createTextNode(label));
    els.health.title = down ? `${llm_provider} is not answering` : "";
  } catch {
    els.health.textContent = "offline";
  }
}

// --- sidebar drawer ----------------------------------------------------------

const openSidebar = () => {
  els.sidebar.classList.remove("-translate-x-full");
  els.scrim.classList.remove("hidden");
};
const closeSidebar = () => {
  els.sidebar.classList.add("-translate-x-full");
  els.scrim.classList.add("hidden");
};

// --- wiring ------------------------------------------------------------------

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = els.input.value.trim();
  if (!message) return;
  els.input.value = "";
  autosize();
  send(message);
});

els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.composer.requestSubmit();
  }
});

function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = `${els.input.scrollHeight}px`;
}
els.input.addEventListener("input", autosize);

for (const button of document.querySelectorAll(".suggestion")) {
  button.addEventListener("click", () => send(button.textContent.trim()));
}

els.newThread.addEventListener("click", () => {
  newThread();
  closeSidebar();
});

els.userId.addEventListener("change", () => {
  const value = els.userId.value.trim();
  if (!value) {
    els.userId.value = state.userId;
    return;
  }
  state.userId = value;
  store.set("cairn.user", value);
  // A different person has a different thread list and different durable facts.
  state.threadId = null;
  store.set("cairn.thread", null);
  clearTranscript();
  els.threadLabel.textContent = "no conversation yet";
  refreshSidebar();
});

els.forget.addEventListener("click", async () => {
  const confirmed = confirm(
    `Delete every conversation and every remembered fact for ${state.userId}?\n\nThis cannot be undone.`,
  );
  if (!confirmed) return;
  try {
    const report = await api(`/users/${encodeURIComponent(state.userId)}`, { method: "DELETE" });
    state.threadId = null;
    store.set("cairn.thread", null);
    clearTranscript();
    els.threadLabel.textContent = "no conversation yet";
    toast(`deleted ${report.threads_deleted} conversations, ${report.facts_deleted} facts`);
    refreshSidebar();
  } catch (error) {
    toast(error.message);
  }
});

els.themeToggle.addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  store.set("cairn.theme", dark ? "dark" : "light");
});

els.menu.addEventListener("click", openSidebar);
els.scrim.addEventListener("click", closeSidebar);

// Delegated: answers are re-rendered on every streamed frame, so per-button
// listeners would be attached and thrown away dozens of times per turn.
els.messages.addEventListener("click", async (event) => {
  const button = event.target.closest("button");
  if (!button) return;

  if (button.hasAttribute("data-copy-code")) {
    const code = button.closest("[data-code]").querySelector("code");
    flash(button, (await copyText(code.textContent)) ? "Copied" : "Failed");
  } else if (button.hasAttribute("data-download-code")) {
    const code = button.closest("[data-code]").querySelector("code");
    downloadText(code.textContent, code.dataset.lang);
    flash(button, "Saved");
  } else if (button.hasAttribute("data-copy-answer")) {
    const answer = button.closest("article").querySelector("[data-answer]");
    flash(button, (await copyText(answerSource.get(answer) ?? answer.innerText)) ? "Copied" : "Failed");
  }
});

// --- boot --------------------------------------------------------------------

els.userId.value = state.userId;
store.set("cairn.user", state.userId);
els.threadLabel.textContent = state.threadId ?? "no conversation yet";

refreshHealth();
refreshSidebar();
if (state.threadId) openThread(state.threadId);
