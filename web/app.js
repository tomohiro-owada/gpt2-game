"use strict";

// Each mode walks ONE passage token-by-token. Choice modes show k options
// (the real next token + GPT-2's top decoys, shuffled); "type" mode asks you
// to type the next token yourself.
const MODES = {
  easy:   { label: "Easy",   k: 3, kind: "choices" },
  medium: { label: "Medium", k: 6, kind: "choices" },
  hard:   { label: "Hard",   k: 9, kind: "choices" },
  type:   { label: "Insane", kind: "type" },
};

const el = (id) => document.getElementById(id);
const screens = {
  start: el("start"),
  game: el("game"),
  end: el("end"),
};

let DATA = null;        // the selected family's dataset (rounds.<family>.json)
let FAMILIES = [];      // families from families.json
let family = null;      // the selected family { key, label, file, ... }
const DATASETS = {};    // family key -> loaded dataset (cache)
let game = null;        // see startGame()
let roughMode = "mixed"; // "clean" | "mixed" | "rough" — which passages to play

// the largest size in the family (the headline opponent + decoy source)
function largest() { return DATA.sizes[DATA.decoy_from]; }
function largestIdx() { return DATA.decoy_from; }

// the passages eligible to play, honoring the rough filter (never empty)
function pool() {
  const ps = DATA.passages;
  let sel;
  if (roughMode === "rough") sel = ps.filter((p) => p.rough);
  else if (roughMode === "mixed") sel = ps;
  else sel = ps.filter((p) => !p.rough);
  return sel.length ? sel : ps;
}

// a random passage from the pool, avoiding `avoid` when there's a choice
function pickPassage(avoid) {
  const ps = pool();
  if (ps.length === 1) return ps[0];
  let p;
  do { p = ps[Math.floor(Math.random() * ps.length)]; } while (p === avoid);
  return p;
}

// ---------- personal leaderboard (localStorage only) ----------
const SCORES_KEY = "gpt2game.scores";

function loadScores() {
  try { return JSON.parse(localStorage.getItem(SCORES_KEY)) || []; } catch (_) { return []; }
}
function saveScores(arr) {
  try { localStorage.setItem(SCORES_KEY, JSON.stringify(arr.slice(-200))); } catch (_) {}
}
function recordGame(entry) {
  const all = loadScores();
  all.push(entry);
  saveScores(all);
}

// Render the top games (by accuracy) into the element `id`, highlighting the
// row whose timestamp is `highlightTs`. Renders nothing if there's no history.
function renderLeaderboard(id, highlightTs) {
  const box = el(id);
  if (!box) return;
  const all = loadScores();
  if (!all.length) { box.innerHTML = ""; return; }

  const ranked = all.slice().sort(
    (a, b) => b.pct - a.pct || b.score - a.score || b.ts - a.ts);
  const fmtDate = (ts) => { const d = new Date(ts); return `${d.getMonth() + 1}/${d.getDate()}`; };

  const rows = ranked.slice(0, 8).map((g, i) => {
    const me = g.ts === highlightTs ? ' class="me"' : "";
    return `<tr${me}><td class="rk">${i + 1}</td><td class="pc">${g.pct}%</td>` +
      `<td>${g.score}/${g.total}</td><td>${escapeHtml(g.modeLabel)}</td>` +
      `<td class="op">vs ${escapeHtml(g.oppShort)}</td><td class="dt">${fmtDate(g.ts)}</td></tr>`;
  }).join("");

  const myRank = ranked.findIndex((g) => g.ts === highlightTs) + 1;
  const note = (highlightTs && myRank)
    ? `<p class="board-note">This game ranked <b>#${myRank}</b> of ${all.length}.</p>` : "";

  box.innerHTML =
    `<div class="board-head"><p class="kicker">your leaderboard · top ${Math.min(8, all.length)} of ${all.length}</p>` +
    `<button class="skip-btn board-clear">clear</button></div>` +
    `<table class="board-table"><tbody>${rows}</tbody></table>${note}`;
  box.querySelector(".board-clear").addEventListener("click", () => {
    saveScores([]); refreshBoards();
  });
}

function refreshBoards(highlightTs) {
  renderLeaderboard("board", highlightTs);
  renderLeaderboard("board-start");
}

// ---------- family + data loading ----------
let DEFAULT_FAMILY = "qwen3";

async function loadFamilies() {
  try {
    const r = await fetch("families.json", { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      if (j && Array.isArray(j.families) && j.families.length) {
        if (j.default) DEFAULT_FAMILY = j.default;
        return j.families;
      }
    }
  } catch (_) { /* fall through */ }
  return [{ key: "gpt2", label: "GPT-2", file: "rounds.gpt2.json", blurb: "" }];
}

async function loadDataset(fam) {
  const r = await fetch(fam.file, { cache: "no-store" });
  const j = await r.json();
  if (j && Array.isArray(j.passages) && j.passages.length) return j;
  throw new Error(`no passages in ${fam.file}`);
}

async function selectFamily(fam) {
  family = fam;
  document.querySelectorAll(".modelbtn").forEach((b) =>
    b.classList.toggle("active", b.dataset.key === fam.key));
  const note = el("data-note");
  note.className = "datanote";
  note.textContent = `loading ${fam.label}…`;
  DATA = DATASETS[fam.key] || (DATASETS[fam.key] = await loadDataset(fam));
  setDataNote();
}

function setDataNote() {
  const note = el("data-note");
  const kind = roughMode === "rough" ? "raw OWT " : roughMode === "mixed" ? "OWT " : "English OWT ";
  const big = largest();
  note.textContent =
    `✓ ${DATA.label}: ${DATA.sizes.length} sizes (${DATA.sizes[0].params}–${big.params}) ` +
    `grade every token · ${pool().length} ${kind}passages.`;
  note.className = "datanote real";
}

// ---------- helpers ----------
function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// Render a raw BPE token's visible content. A token that owns a leading space
// shows a faint "·" before it; a token with no leading space is rendered bare,
// so it simply glues onto the previous token with no gap — the absence of a "·"
// is itself the signal that it continues the previous word. Boundaries between
// glued tokens still show through their per-token fill/underline colours.
function tokenHtml(raw) {
  const lead = raw.startsWith(" ")
    ? `<span class="sp" aria-hidden="true">·</span>` : "";
  let body = escapeHtml(lead ? raw.slice(1) : raw)
    .replace(/\n/g, `<span class="ctrl">⏎</span>`)
    .replace(/\t/g, `<span class="ctrl">⇥</span>`);
  if (body === "") body = `<span class="ctrl">∅</span>`;
  return lead + body;
}

// Plain visible text of a token (used in result-banner prose, not the readout).
function tokenText(raw) {
  return raw.replace(/\n/g, " ").replace(/\t/g, " ");
}

// Lenient match for "type" mode: trim, lowercase, drop surrounding punctuation.
function normalize(s) {
  const t = s.trim().toLowerCase();
  const stripped = t.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");
  return stripped || t;
}

// Classify a typed guess against the real token:
//   right  — same letters AND same word-boundary (leading space matches)
//   near   — same letters but wrong space sense (e.g. "ing" for the token " ing")
//   wrong  — different letters
function classifyTyped(guess, trueToken) {
  if (normalize(guess) !== normalize(trueToken)) return "wrong";
  const typedLeadsSpace = /^\s/.test(guess);
  const trueLeadsSpace = trueToken.startsWith(" ");
  return typedLeadsSpace === trueLeadsSpace ? "right" : "near";
}

// Token with its leading space shown as "·", for result-banner prose.
function tokWithDot(raw) {
  return raw.startsWith(" ") ? `·${raw.slice(1)}` : raw;
}

function show(screen) {
  for (const k of Object.keys(screens)) screens[k].classList.add("hidden");
  screens[screen].classList.remove("hidden");
}

// ---------- game flow ----------
function startGame(modeKey, avoid) {
  if (!DATA || !DATA.passages) return;   // opponent dataset still loading
  if (game && game.timer) clearTimeout(game.timer);
  const mode = MODES[modeKey];
  const passage = pickPassage(avoid);
  game = {
    mode,
    modeKey,
    passage,
    stepIdx: 0,
    score: 0,                          // your correct guesses
    gptScore: 0,                       // tokens the largest size got (#1)
    sizeHits: DATA.sizes.map(() => 0), // per-size running hits, for the live ladder
    history: [],                       // { token, youResult, gptRight } per step
    answered: false,
    timer: null,                       // pending auto-advance timeout
  };
  el("mode-label").textContent = mode.kind === "type"
    ? `${mode.label} · ${DATA.label}`
    : `${mode.label} · ${mode.k} · ${DATA.label}`;
  el("result").classList.add("hidden");   // no "previous answer" yet
  show("game");
  renderStep();
}

// Render the passage so far as a stream of tokens:
//   • the opening prefix tokens get a neutral grey "given" underline,
//   • each revealed token is tinted by whether YOU got it (fill) and underlined
//     by whether GPT-2 got it (it "got it" when the real token was its #1 pick).
// animateLast pops the freshly-revealed token.
function renderWalk(elId, animateLast) {
  const prefixToks = game.passage.prefix_tokens || [game.passage.prefix];
  const parts = prefixToks.map(
    (t) => `<span class="tok given">${tokenHtml(t)}</span>`
  );

  game.history.forEach((h, i) => {
    const cls = [
      "tok",
      `you-${h.youResult}`,                       // right | near | wrong
      h.gptRight ? "gpt-right" : "gpt-wrong",
    ];
    if (animateLast && i === game.history.length - 1) cls.push("just");
    parts.push(`<span class="${cls.join(" ")}">${tokenHtml(h.token)}</span>`);
  });
  // <wbr> between tokens: the spans are glued with no whitespace, so without an
  // explicit break opportunity the line can't wrap and overflows the panel.
  el(elId).innerHTML = parts.join("<wbr>");
}

function updateScore() {
  el("score").innerHTML =
    `You <b>${game.score}</b> · <span class="bot">${largest().name}</span> <b>${game.gptScore}</b>`;
}

function setBar() {
  const total = game.passage.steps.length;
  el("bar").style.width = `${Math.round((game.history.length / total) * 100)}%`;
}

// the live in-game ladder: your running accuracy among the family's sizes
function updateLiveLadder() {
  const n = game.history.length;
  renderLadder(n ? Math.round((game.score / n) * 100) : null, "ladder-live", true);
}

function renderStep() {
  game.answered = false;
  const steps = game.passage.steps;
  const step = steps[game.stepIdx];

  el("progress").textContent = `Word ${game.stepIdx + 1} / ${steps.length}`;
  updateScore();
  setBar();
  updateLiveLadder();
  renderWalk("context", false);
  // NB: #result is left as-is — it keeps showing the PREVIOUS word's outcome
  // below the new question (we auto-advance, so there's no Next button).

  if (game.mode.kind === "choices") {
    el("choices").classList.remove("hidden");
    el("type-area").classList.add("hidden");
    renderChoices(step);
  } else {
    el("choices").classList.add("hidden");
    el("type-area").classList.remove("hidden");
    const input = el("type-input");
    input.value = "";
    input.disabled = false;
    el("type-submit").disabled = false;
    input.focus();
  }
}

function renderChoices(step) {
  // the real token + top decoys, sliced to k, shuffled
  const truth = {
    token: step.true_token,
    prob: step.true_prob,
    rank: step.true_rank,
    isTrue: true,
  };
  const decoys = step.decoys.slice(0, game.mode.k - 1).map((d) => ({
    token: d.token, prob: d.prob, rank: d.rank, isTrue: false,
  }));
  const options = shuffle([truth, ...decoys]);

  const choicesEl = el("choices");
  choicesEl.innerHTML = "";
  options.forEach((opt, i) => {
    const btn = document.createElement("button");
    btn.className = "choice";
    btn.innerHTML = `<span class="num">${i + 1}</span>${tokenHtml(opt.token)}`;
    btn._opt = opt;
    btn.addEventListener("click", () => onChoice(btn, opt));
    choicesEl.appendChild(btn);
  });
}

function onChoice(btn, opt) {
  if (game.answered) return;
  const step = game.passage.steps[game.stepIdx];

  // reveal prob + rank on every shown choice, mark right/wrong
  Array.from(el("choices").children).forEach((b) => {
    b.disabled = true;
    const o = b._opt;
    const pct = (o.prob * 100).toFixed(o.prob < 0.01 ? 2 : 1);
    b.querySelector(".prob")?.remove();
    const tag = document.createElement("span");
    tag.className = "prob";
    tag.textContent = `${pct}% · #${o.rank}`;
    b.appendChild(tag);
    if (o.isTrue) b.classList.add("correct");
    else if (b === btn) b.classList.add("wrong");
  });

  finishStep(opt.isTrue ? "right" : "wrong", step, null, opt.token);
}

function onType() {
  if (game.answered) return;
  const step = game.passage.steps[game.stepIdx];
  const guess = el("type-input").value;
  if (!guess.trim()) return;
  el("type-input").disabled = true;
  el("type-submit").disabled = true;
  finishStep(classifyTyped(guess, step.true_token), step, guess, null);
}

const ADVANCE_MS = 650;   // brief pause before auto-advancing (the result stays
                          // pinned below, so this can be short; Enter/Space skips it)

// classify a model size's prediction for this step into right/near/wrong
function sizeResult(m, trueToken) {
  if (m.rank === 1) return "right";
  return classifyTyped(m.top, trueToken) === "near" ? "near" : "wrong";
}

// Shared end-of-step: score, grow the passage, show the result, auto-advance.
// result is "right" | "near" | "wrong" ("near" only happens in type mode).
// userTok = the token the player clicked (choice mode); typed = their raw text.
function finishStep(result, step, typed, userTok) {
  game.answered = true;
  const big = step.models[largestIdx()];          // the headline (largest) size
  const gptRight = big.rank === 1;
  // a "near" miss still counts — you knew the word, just not the leading space
  if (result !== "wrong") game.score += 1;
  if (gptRight) game.gptScore += 1;
  // tally every size's hits this game, for the live ladder
  step.models.forEach((m, i) => { if (m.rank === 1) game.sizeHits[i] += 1; });
  game.history.push({ token: step.true_token, youResult: result, gptRight });
  updateScore();
  setBar();
  updateLiveLadder();

  // reveal the real next token by growing the (now color-coded) passage
  renderWalk("context", true);

  const icon = result === "right" ? "✓" : result === "near" ? "≈" : "✗";
  const youInner = userTok != null
    ? tokenHtml(userTok)
    : escapeHtml((typed || "").trim()) || `<span class="ctrl">∅</span>`;
  const gptResult = sizeResult(big, step.true_token);

  // row 1: real · you · the largest model's pick
  const row1 =
    `<span class="res-status">${icon}</span>` +
    `<span class="res-item"><span class="res-lab">real</span>` +
      `<span class="rtok real">${tokenHtml(step.true_token)}</span></span>` +
    `<span class="res-item"><span class="res-lab">you</span>` +
      `<span class="rtok r-${result}">${youInner}</span></span>` +
    `<span class="res-item"><span class="res-lab">${escapeHtml(largest().name)}</span>` +
      `<span class="rtok r-${gptResult}">${tokenHtml(big.top)}</span></span>`;

  // row 2: how EVERY size graded this token (✓ if its #1 was the real token)
  const strip = DATA.sizes.map((sz, i) => {
    const m = step.models[i];
    const cls = m.rank === 1 ? "ok" : "no";
    return `<span class="sz sz-${cls}" title="${escapeHtml(sz.name)}: ${m.rank === 1 ? "got it" : "missed (real was #" + m.rank + ")"}">${escapeHtml(sz.params)}</span>`;
  }).join("");

  const resEl = el("result");
  resEl.className = `result ${result === "right" ? "correct" : result}`;
  resEl.innerHTML =
    `<div class="res-row">${row1}</div>` +
    `<div class="res-sizes"><span class="res-lab">all sizes</span>${strip}</div>`;

  // auto-advance after a beat (press Enter/Space to skip the wait — see onKey)
  game.timer = setTimeout(advance, ADVANCE_MS);
}

function advance() {
  if (!game || !game.answered) return;
  if (game.timer) { clearTimeout(game.timer); game.timer = null; }
  // the revealed token is already in game.history; just move on
  game.stepIdx += 1;
  if (game.stepIdx >= game.passage.steps.length) return endGame();
  renderStep();
}

// Render the family's model-size ladder into `targetId` (rungs = each size's
// corpus-wide hit rate, ordered small→large), marking the player's accuracy
// `pct` (null = no YOU marker yet). compact = no verdict line (live in-game).
function renderLadder(pct, targetId, compact) {
  const box = el(targetId || "ladder");
  if (!box) return;
  const sizes = DATA.sizes;
  if (!sizes || !sizes.length) { box.innerHTML = ""; return; }

  const topPct = Math.max(...sizes.map((s) => s.pct));
  const max = Math.max(pct || 0, topPct) * 1.12;
  const bar = (p, cls) =>
    `<div class="rung-bar"><div class="rung-fill ${cls}" style="width:${(p / max) * 100}%"></div></div>`;

  const rows = sizes.map((r) =>
    `<div class="rung"><span class="rung-name">${escapeHtml(r.name)}` +
    `<span class="rung-params">${escapeHtml(r.params)}</span></span>` +
    `${bar(r.pct, "model")}<span class="rung-pct">${r.pct}%</span></div>`).join("");

  const youRow = pct == null ? "" :
    `<div class="rung you"><span class="rung-name">YOU<span class="rung-params">so far</span></span>` +
    `${bar(pct, "me")}<span class="rung-pct">${pct}%</span></div>`;

  let head = `<p class="kicker">${escapeHtml(DATA.label)} sizes — graded over the whole corpus</p>`;
  if (!compact && pct != null) {
    let near = sizes[0];
    for (const r of sizes) if (Math.abs(r.pct - pct) < Math.abs(near.pct - pct)) near = r;
    const tip = game.mode.kind === "type" ? ""
      : ` <span class="ladder-caveat">(choice modes flatter you — try Insane for a fair fight)</span>`;
    head = `<p class="kicker">${escapeHtml(DATA.label)} family — where you land</p>` +
      `<p class="ladder-verdict">You predicted like <b>${escapeHtml(near.name)}</b> (${near.pct}%).${tip}</p>`;
  }
  box.innerHTML = head + youRow + rows;
}

function endGame() {
  const total = game.passage.steps.length;
  const s = game.score;
  const big = largest().name;
  el("final-score").innerHTML = `You ${s} / ${total} &nbsp;·&nbsp; ${big} ${game.gptScore} / ${total}`;
  const pct = total ? Math.round((s / total) * 100) : 0;
  let verdict;
  if (s > game.gptScore) verdict = `${pct}% — you out-predicted ${big}. Showing off.`;
  else if (s === game.gptScore) verdict = `${pct}% — dead even with ${big}.`;
  else if (pct >= 60) verdict = `${pct}% — strong, though ${big} edged you out.`;
  else if (pct >= 30) verdict = `${pct}% — natural language is slippery.`;
  else verdict = `${pct}% — predicting real text is harder than it looks.`;
  el("verdict").textContent = verdict;
  renderWalk("end-passage", false);   // the full color-coded passage
  renderLadder(pct, "ladder", false);

  const entry = {
    ts: Date.now(),
    mode: game.modeKey, modeLabel: game.mode.label,
    opp: family.key, oppShort: DATA.label,
    score: s, gpt: game.gptScore, total, pct, roughMode,
  };
  recordGame(entry);
  refreshBoards(entry.ts);
  show("end");
}

// ---------- keyboard ----------
function onKey(e) {
  if (screens.game.classList.contains("hidden") || !game) return;

  if (game.answered) {
    // already auto-advancing; Enter/Space just skips the wait
    if (e.key === "Enter" || (e.key === " " && game.mode.kind !== "type")) {
      e.preventDefault();
      advance();
    }
    return;
  }
  if (game.mode.kind === "choices" && /^[1-9]$/.test(e.key)) {
    const btns = el("choices").children;
    const idx = parseInt(e.key, 10) - 1;
    if (idx < btns.length) { e.preventDefault(); btns[idx].click(); }
  }
}

// ---------- wire up ----------
function renderFamilyPicker() {
  const wrap = el("models");
  wrap.innerHTML = "";
  FAMILIES.forEach((f) => {
    const btn = document.createElement("button");
    btn.className = "modelbtn";
    btn.dataset.key = f.key;
    btn.innerHTML =
      `<span class="modelbtn-name">${escapeHtml(f.label)}</span>` +
      `<span class="modelbtn-blurb">${escapeHtml(f.blurb || "")}</span>`;
    btn.addEventListener("click", () => selectFamily(f));
    wrap.appendChild(btn);
  });
}

function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  el("theme-toggle").textContent = t === "light" ? "☾" : "☀";
}

function init() {
  // theme toggle (the inline <head> script already set the initial theme)
  applyTheme(document.documentElement.getAttribute("data-theme") || "dark");
  el("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
    try { localStorage.setItem("gpt2game.theme", next); } catch (_) {}
    applyTheme(next);
  });

  renderFamilyPicker();
  document.querySelectorAll(".mode").forEach((b) =>
    b.addEventListener("click", () => startGame(b.dataset.mode))
  );
  el("again").addEventListener("click", () => show("start"));
  el("skip").addEventListener("click", () => startGame(game.modeKey, game.passage));
  el("to-menu").addEventListener("click", () => {
    if (game && game.timer) clearTimeout(game.timer);
    show("start");
  });
  el("type-submit").addEventListener("click", onType);
  el("type-input").addEventListener("keydown", (e) => {
    // stopPropagation so this submit Enter doesn't also hit onKey and instantly
    // skip the result pause
    if (e.key === "Enter" && !game.answered) { e.preventDefault(); e.stopPropagation(); onType(); }
  });
  document.addEventListener("keydown", onKey);

  // text-style filter: clean (default) / mixed / rough, remembered across visits
  try {
    const saved = localStorage.getItem("gpt2game.roughmode");
    if (saved) roughMode = saved;
    else if (localStorage.getItem("gpt2game.rough") === "1") roughMode = "mixed"; // migrate
  } catch (_) {}
  const seg = el("rough-seg");
  const syncSeg = () => seg.querySelectorAll(".seg-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.rough === roughMode));
  seg.querySelectorAll(".seg-btn").forEach((b) =>
    b.addEventListener("click", () => {
      roughMode = b.dataset.rough;
      try { localStorage.setItem("gpt2game.roughmode", roughMode); } catch (_) {}
      syncSeg();
      setDataNote();
    }));
  syncSeg();

  renderLeaderboard("board-start");   // show history (if any) on the start screen
}

(async () => {
  try {
    FAMILIES = await loadFamilies();
    init();
    const def = FAMILIES.find((f) => f.key === DEFAULT_FAMILY) || FAMILIES[0];
    await selectFamily(def);
  } catch (e) {
    el("start").innerHTML =
      `<h2>Failed to load</h2><p class="muted">${escapeHtml(String(e))}</p>`;
  }
})();
