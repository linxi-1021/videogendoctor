/* VideoGenDoctor Annotation UI (local, static)
 * This file wires up the controls defined in index.html.
 */

(() => {
  "use strict";

  /** @typedef {{code:string, definition?:string, group_id?:string, group_label?:string}} FailureCodeDef */

  const els = {
    loadDefaultBtn: document.getElementById("loadDefaultBtn"),
    pickDirectoryBtn: document.getElementById("pickDirectoryBtn"),
    directoryInput: document.getElementById("directoryInput"),
    saveBtn: document.getElementById("saveBtn"),

    manifestPathInput: document.getElementById("manifestPathInput"),
    annotatorInput: document.getElementById("annotatorInput"),

    datasetStatus: document.getElementById("datasetStatus"),
    storageStatus: document.getElementById("storageStatus"),
    validationBanner: document.getElementById("validationBanner"),

    sampleCounter: document.getElementById("sampleCounter"),
    searchInput: document.getElementById("searchInput"),
    statusFilter: document.getElementById("statusFilter"),
    sampleList: document.getElementById("sampleList"),

    currentTitle: document.getElementById("currentTitle"),
    metaPerturb: document.getElementById("metaPerturb"),
    metaShotir: document.getElementById("metaShotir"),

    videoPlayerOriginal: document.getElementById("videoPlayerOriginal"),
    videoPlayer: document.getElementById("videoPlayer"),
    originalVideoHint: document.getElementById("originalVideoHint"),
    videoOverlay: document.getElementById("videoOverlay"),

    currentTimeLabel: document.getElementById("currentTimeLabel"),
    timelineInput: document.getElementById("timelineInput"),
    durationLabel: document.getElementById("durationLabel"),
    jumpTimeInput: document.getElementById("jumpTimeInput"),
    jumpTimeBtn: document.getElementById("jumpTimeBtn"),

    prevSampleBtn: document.getElementById("prevSampleBtn"),
    nextSampleBtn: document.getElementById("nextSampleBtn"),
    playPauseBtn: document.getElementById("playPauseBtn"),
    prevFrameBtn: document.getElementById("prevFrameBtn"),
    nextFrameBtn: document.getElementById("nextFrameBtn"),
    fpsInput: document.getElementById("fpsInput"),

    metaId: document.getElementById("metaId"),
    metaVideoPath: document.getElementById("metaVideoPath"),
    metaFailureCodes: document.getElementById("metaFailureCodes"),
    metaTimeState: document.getElementById("metaTimeState"),

    addFailureBtn: document.getElementById("addFailureBtn"),
    recordIdInput: document.getElementById("recordIdInput"),
    recordAnnotatorInput: document.getElementById("recordAnnotatorInput"),
    failureList: document.getElementById("failureList"),
    notesInput: document.getElementById("notesInput"),

    suggestedCodes: document.getElementById("suggestedCodes"),

    failureCardTemplate: document.getElementById("failureCardTemplate"),
  };

  const STORAGE_KEY = "vgd_annotation_ui_v0.1";

  /** @type {{rootName?: string, fileCount: number, filesByRelPath: Map<string, File>}} */
  const boundDir = {
    rootName: undefined,
    fileCount: 0,
    filesByRelPath: new Map(),
  };

  /** @type {Array<any>} */
  let dataset = [];

  /** @type {string[]} */
  let visibleSampleIds = [];

  /** @type {string|null} */
  let activeSampleId = null;

  /** @type {string|null} */
  let currentVideoObjectUrl = null;

  /** @type {File|null} */
  let currentVideoFile = null;

  /** @type {string|null} */
  let currentOriginalVideoObjectUrl = null;

  /** @type {File|null} */
  let currentOriginalVideoFile = null;

  /** @type {boolean} */
  let isSyncingPlayback = false;

  /** @type {Map<string, any>} */
  const annotationsById = new Map();

  /** @type {Map<string, string>} */
  const lastAutoNotesById = new Map();

  /** @type {Map<string, FailureCodeDef>} */
  const taxonomyByCode = new Map();

  /** @type {WeakSet<object>} */
  const autoFailureRefs = new WeakSet();

  /** @type {ReturnType<typeof setTimeout> | null} */
  let autosaveTimer = null;

  /** @type {Set<string>} */
  const autosavePendingIds = new Set();


  /** @type {Record<string, string>} */
  const CODE_LABELS = {
    // Identity
    ID_FACE_DRIFT: "Face appearance drift or discoloration (local tint shift, face-swap-like)",
    ID_BODY_DRIFT: "Body or clothing appearance drift (local tint shift, outfit-swap-like)",
    ID_WRONG_CHARACTER: "Unexpected character or subject appears",
    ID_CLOTHING_CHANGE: "Unexplained clothing change (sudden color/style switch)",
    ID_MULTIPLE_FACES: "Multiple faces appear when only one person is expected",
    ID_NO_FACE_VISIBLE: "No clear face visible in a near/mid shot (missing, occluded, or blurred)",

    // Scene
    SC_BG_INCONSISTENCY: "Unexpected background or scene switch (jump-cut-like)",
    SC_ENVIRONMENT_MISMATCH: "Environment does not match the spec (indoor/outdoor/location type mismatch)",
    SC_LIGHTING_SHIFT: "Abrupt lighting or brightness change (like lights toggling or shadow direction flipping)",
    SC_OBJECT_TELEPORT: "Object suddenly appears, disappears, or teleports",
    SC_BG_FLICKER: "Background flicker or exposure jumping (not caused by subject motion)",
    SC_WRONG_TIME_OF_DAY: "Wrong time of day (day/night or sunlight mood mismatch)",

    // Motion
    MO_JITTER: "Temporal jitter (small back-and-forth jumps or twitching)",
    MO_FRAME_DROP: "Dropped or repeated frames causing stutter or sudden jumps",
    MO_UNNATURAL_MOTION: "Physically implausible motion (interpenetration, floating, distortion)",
    MO_EVENT_MISSING: "Expected event never happens (missing after repetition or delay)",
    MO_ACTION_ORDER_WRONG: "Action order is wrong (reversed sequence)",
    MO_MOTION_BLUR_EXCESS: "Excessive motion blur (details become smeared)",
    MO_FROZEN_FRAME: "Frozen frame or prolonged stillness",
    MO_SEGMENT_BREAK: "Segment break (inserted repetition, abrupt pause, or discontinuity)",

    // Camera
    CA_MOVE_WRONG: "Camera movement mismatch (digital pan/zoom/rotation feels wrong)",
    CA_SHOT_TYPE_WRONG: "Incorrect shot type (close/mid/wide shot mismatch)",
    CA_UNINTENDED_ZOOM: "Unintended zoom (sudden zoom-in or zoom-out)",
    CA_SHAKE: "Camera shake (handheld-like wobble)",
    CA_WRONG_ANGLE: "Incorrect camera angle (abnormal top/down/side angle)",
    CA_ROLL: "Frame tilt or roll rotation",

    // Alignment
    AL_PROP_MISSING: "Prop or key object missing (partially erased or blurred out)",
    AL_PROP_WRONG_PLACEMENT: "Prop placed incorrectly (wrong location or drifting)",
    AL_TEXT_PROMPT_MISMATCH: "Content does not match the prompt (missing or extra key elements)",
    AL_CHARACTER_COUNT_WRONG: "Incorrect number of characters (too many or too few)",
    AL_WRONG_PROP: "Unexpected prop or object appears",
    AL_PROP_OCCLUDED: "Prop is occluded or unclear when it should be visible",

    // Style
    ST_COLOR_SHIFT: "Global color cast shift across the whole frame",
    ST_ART_STYLE_INCONSISTENCY: "Abrupt art style change (material, stroke, or rendering style shifts)",
    ST_COMPRESSION_ARTIFACT: "Heavy compression or resampling artifacts (blocking, aliasing, blurred detail)",
    ST_TEXTURE_FLICKER: "Texture flicker (fine details flash or jump)",
    ST_OVEREXPOSURE: "Overexposure (washed out, highlight details lost)",
    ST_UNDEREXPOSURE: "Underexposure (too dark, shadow details unclear)",
  };

  function getCodeLabel(code) {
    return CODE_LABELS[String(code || "").trim()] || "";
  }

  function getCodeGroupLabel(code) {
    const c = String(code || "").trim();
    if (c.startsWith("ID_")) return "ID";
    if (c.startsWith("SC_")) return "SC";
    if (c.startsWith("MO_")) return "MO";
    if (c.startsWith("CA_")) return "CA";
    if (c.startsWith("AL_")) return "AL";
    if (c.startsWith("ST_")) return "ST";
    return "";
  }

  function getCodeGroupId(code) {
    const c = String(code || "").trim();
    if (c.startsWith("ID_")) return "Identity";
    if (c.startsWith("SC_")) return "Scene";
    if (c.startsWith("MO_")) return "Motion";
    if (c.startsWith("CA_")) return "Camera";
    if (c.startsWith("AL_")) return "Alignment";
    if (c.startsWith("ST_")) return "Style";
    return "";
  }

  function setPill(el, text, kind = "muted") {
    el.textContent = text;
    el.classList.remove("muted", "success", "warning", "error");
    el.classList.add(kind);
  }

  function showBanner(kind, message) {
    if (!message) {
      els.validationBanner.classList.add("hidden");
      els.validationBanner.textContent = "";
      return;
    }
    els.validationBanner.classList.remove("hidden");
    els.validationBanner.textContent = message;
    els.validationBanner.style.color = "";
    els.validationBanner.style.borderColor = "";
    els.validationBanner.style.background = "";

    // Keep it simple: default is warning style from CSS, only override for error.
    if (kind === "error") {
      els.validationBanner.style.color = "var(--danger)";
      els.validationBanner.style.borderColor = "rgba(255, 107, 136, 0.3)";
      els.validationBanner.style.background = "rgba(255, 107, 136, 0.08)";
    }
  }

  function normalizePath(p) {
    return String(p || "")
      .replace(/\\/g, "/")
      .replace(/^\.(\/)+/, "")
      .replace(/^\//, "")
      .trim();
  }

  function basename(p) {
    const n = normalizePath(p);
    const parts = n.split("/").filter(Boolean);
    return parts.length ? parts[parts.length - 1] : n;
  }

  function dirname(p) {
    const n = normalizePath(p);
    const i = n.lastIndexOf("/");
    return i >= 0 ? n.slice(0, i + 1) : "";
  }

  function withoutExt(name) {
    return String(name || "").replace(/\.[a-z0-9]+$/i, "");
  }

  function extractSourceStem(sample) {
    const directCandidates = [
      sample?.source_id,
      sample?.original_id,
      sample?.meta?.source_id,
      sample?.meta?.original_id,
    ];
    for (const v of directCandidates) {
      const stem = String(v || "").trim();
      if (stem) return stem;
    }

    const sampleIdStem = String(sample?.id || "").split("__")[0].trim();
    if (sampleIdStem) return sampleIdStem;

    const videoStem = withoutExt(basename(sample?.video_path || "")).split("__")[0].trim();
    if (videoStem) return videoStem;

    const shotirStem = withoutExt(basename(sample?.shotir_path || ""))
      .replace(/_shotir$/i, "")
      .trim();
    if (shotirStem) return shotirStem;

    return "";
  }

  function getPlayers() {
    return [els.videoPlayerOriginal, els.videoPlayer].filter(Boolean);
  }

  function hasVideoSource(player) {
    return Boolean(player && player.getAttribute("src"));
  }

  function formatTime(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "00:00.000";
    const ms = Math.floor((seconds % 1) * 1000);
    const total = Math.floor(seconds);
    const s = total % 60;
    const m = Math.floor(total / 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
  }

  function safeJsonParse(line) {
    try {
      return JSON.parse(line);
    } catch {
      return null;
    }
  }

  function parseJsonl(text) {
    const out = [];
    const lines = String(text || "").split(/\r?\n/);
    for (const raw of lines) {
      const line = raw.trim();
      if (!line) continue;
      const obj = safeJsonParse(line);
      if (obj) out.push(obj);
    }
    return out;
  }

  async function fetchText(url) {
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
    return await resp.text();
  }

  async function readFileText(file) {
    return await file.text();
  }

  function updateStorageStatusPill() {
    const dirText = boundDir.fileCount
      ? `Bound directory: ${boundDir.rootName || "(unknown)"} · ${boundDir.fileCount} files`
      : "No local directory bound";
    const autosaveText = " · Autosave: localStorage";
    const kind = boundDir.fileCount ? "success" : "muted";
    setPill(els.storageStatus, dirText + autosaveText, kind);
  }

  function bindDirectoryFiles(fileList) {
    boundDir.filesByRelPath.clear();
    boundDir.rootName = undefined;
    boundDir.fileCount = 0;

    const files = Array.from(fileList || []);
    if (!files.length) {
      updateStorageStatusPill();
      return;
    }

    // webkitRelativePath is like "folder/sub/a.mp4".
    const firstPath = files[0].webkitRelativePath || files[0].name;
    const normFirst = normalizePath(firstPath);
    const rootSeg = normFirst.split("/")[0];
    boundDir.rootName = rootSeg;
    boundDir.fileCount = files.length;

    for (const f of files) {
      const rel = normalizePath(f.webkitRelativePath || f.name);
      let within = rel;
      if (boundDir.rootName && within.startsWith(boundDir.rootName + "/")) {
        within = within.slice(boundDir.rootName.length + 1);
      }
      boundDir.filesByRelPath.set(within, f);
      boundDir.filesByRelPath.set(rel, f);
      // Also index by basename for fallback.
      boundDir.filesByRelPath.set(basename(rel), f);
    }

    updateStorageStatusPill();
  }

  function findFileByPathHint(pathHint) {
    const hint = normalizePath(pathHint);
    if (!hint) return null;

    const candidates = new Set();
    candidates.add(hint);
    candidates.add(hint.replace(/^out\/dataset_v0\//, ""));
    candidates.add(hint.replace(/^out\/dataset_v\d+\//, ""));
    candidates.add(hint.replace(/^data\//, ""));

    if (boundDir.rootName) {
      candidates.add(hint.replace(new RegExp(`^${boundDir.rootName}/`), ""));
    }

    // direct map lookup
    for (const c of candidates) {
      const direct = boundDir.filesByRelPath.get(c);
      if (direct) return direct;
    }

    // suffix match
    const values = Array.from(boundDir.filesByRelPath.entries());
    for (const c of candidates) {
      const cn = normalizePath(c);
      for (const [k, f] of values) {
        const kn = normalizePath(k);
        if (kn.endsWith(cn)) return f;
      }
    }

    // basename fallback
    const base = basename(hint);
    return boundDir.filesByRelPath.get(base) || null;
  }

  async function loadTaxonomy() {
    taxonomyByCode.clear();

    const candidates = [
      "../../packages/videoeval/videoeval/taxonomy/failure_taxonomy_v0.1.json",
      "../../packages/videoeval/videoeval/taxonomy/failure_taxonomy.json",
      "../videoeval/taxonomy/failure_taxonomy_v0.1.json",
    ];

    let taxObj = null;

    // Try fetch from workspace-served paths first.
    for (const url of candidates) {
      try {
        const txt = await fetchText(url);
        taxObj = JSON.parse(txt);
        break;
      } catch {
        // ignore
      }
    }

    // Try from bound directory if present.
    if (!taxObj && boundDir.filesByRelPath.size) {
      const taxFile =
        findFileByPathHint("failure_taxonomy_v0.1.json") ||
        findFileByPathHint("failure_taxonomy.json");
      if (taxFile) {
        try {
          taxObj = JSON.parse(await readFileText(taxFile));
        } catch {
          taxObj = null;
        }
      }
    }

    if (taxObj && Array.isArray(taxObj.groups)) {
      for (const g of taxObj.groups) {
        const group_id = g.group_id;
        const group_label = g.group_label;
        const codes = Array.isArray(g.codes) ? g.codes : [];
        for (const c of codes) {
          if (!c || !c.code) continue;
          taxonomyByCode.set(c.code, {
            code: c.code,
            definition: c.definition,
            group_id,
            group_label,
          });
        }
      }
    }

    // Fallback: when taxonomy JSON cannot be loaded (common in file:// mode),
    // still expose the full code list and built-in labels.
    for (const code of Object.keys(CODE_LABELS)) {
      if (taxonomyByCode.has(code)) continue;
      taxonomyByCode.set(code, {
        code,
        group_id: getCodeGroupId(code),
        group_label: getCodeGroupLabel(code),
      });
    }

    // Fallback: include codes seen in dataset (once loaded).
  }

  async function loadManifestFromPath(path) {
    const p = normalizePath(path);
    if (!p) throw new Error("manifest path is empty");

    // Prefer bound directory reads when possible.
    const file = boundDir.filesByRelPath.size ? findFileByPathHint(p) : null;
    if (file) {
      return parseJsonl(await readFileText(file));
    }

    // Then try fetch.
    const txt = await fetchText(p);
    return parseJsonl(txt);
  }

  function getGlobalAnnotatorId() {
    const a = String(els.annotatorInput.value || "").trim();
    return a || "unknown";
  }

  function containsCjk(text) {
    return /[\u4e00-\u9fff]/.test(String(text || ""));
  }

  function migrateLegacyChineseNotes(rec) {
    if (!rec || !rec.id) return;
    const current = String(rec.notes || "");
    if (!current.trim() || !containsCjk(current)) return;

    const next = buildAutoNotes(rec);
    if (!next) return;

    rec.notes = next;
    lastAutoNotesById.set(rec.id, next);
  }

  function hydrateFromLocalStorage() {
    annotationsById.clear();
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const obj = JSON.parse(raw);
      if (!obj || typeof obj !== "object") return;
      const items = Array.isArray(obj.items) ? obj.items : [];
      for (const rec of items) {
        if (rec && rec.id) {
          migrateLegacyChineseNotes(rec);
          annotationsById.set(rec.id, rec);
        }
      }
    } catch {
      // ignore
    }
  }

  function persistToLocalStorage() {
    const items = Array.from(annotationsById.values());
    const payload = {
      version: "0.1",
      saved_at: new Date().toISOString(),
      items,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  }

  // Note: Browser-only mode cannot write into arbitrary local folders.
  // We keep autosave in localStorage, and provide a manual Save button to download JSONL.

  function syncActiveRecordFromInputs() {
    if (!activeSampleId) return;
    const rec = annotationsById.get(activeSampleId) || getOrInitRecord(activeSampleId);
    rec.annotator_id = String(els.recordAnnotatorInput.value || getGlobalAnnotatorId()).trim() || getGlobalAnnotatorId();
    rec.notes = String(els.notesInput.value || "");
  }

  async function flushAutosaveNow({ ids = null } = {}) {
    const flushIds = Array.isArray(ids) ? ids.filter(Boolean) : Array.from(autosavePendingIds);
    if (Array.isArray(ids)) {
      for (const id of flushIds) autosavePendingIds.delete(id);
    } else {
      autosavePendingIds.clear();
    }
    if (!flushIds.length) return;

    // Ensure the active panel edits are reflected into the record before saving.
    syncActiveRecordFromInputs();

    persistToLocalStorage();
  }

  function saveAllAnnotationsToDownload() {
    // Make sure the most recent edits are captured.
    syncActiveRecordFromInputs();
    persistToLocalStorage();
    exportAllRecordsAsJsonl();
    setPill(els.datasetStatus, `Saved: ${new Date().toLocaleTimeString()}`, "success");
  }

  function scheduleAutosave(sampleId) {
    const id = String(sampleId || "").trim();
    if (!id) return;
    autosavePendingIds.add(id);

    if (autosaveTimer) clearTimeout(autosaveTimer);
    autosaveTimer = setTimeout(() => {
      autosaveTimer = null;
      flushAutosaveNow().catch(() => {
        // ignore
      });
    }, 500);
  }

  function getSampleById(id) {
    return dataset.find((s) => s && s.id === id) || null;
  }

  function getOrInitRecord(sampleId) {
    const existing = annotationsById.get(sampleId);
    if (existing) return existing;

    const rec = {
      id: sampleId,
      annotator_id: getGlobalAnnotatorId(),
      top_failures: [],
      notes: "",
    };
    annotationsById.set(sampleId, rec);
    return rec;
  }

  function isFailureComplete(f) {
    if (!f || !f.code) return false;
    if (!("verified" in f) || f.verified === "") return false;
    if (!Number.isFinite(f.confidence)) return false;
    if (f.confidence < 0 || f.confidence > 1) return false;

    const ev = f.evidence || {};
    if (!Number.isFinite(ev.t0) || !Number.isFinite(ev.t1)) return false;
    if (ev.t0 < 0 || ev.t1 < 0 || ev.t0 > ev.t1) return false;

    return true;
  }

  function isRecordComplete(rec) {
    const failures = Array.isArray(rec?.top_failures) ? rec.top_failures : [];
    if (!failures.length) return false;
    return failures.every(isFailureComplete);
  }

  function formatNoteTimeRange(failure) {
    const ev = failure?.evidence || {};
    const t0 = Number.isFinite(ev.t0) ? Number(ev.t0).toFixed(3) : "?";
    const t1 = Number.isFinite(ev.t1) ? Number(ev.t1).toFixed(3) : "?";
    return `${t0}s-${t1}s`;
  }

  function describeConfidence(confidence) {
    if (!Number.isFinite(confidence)) return "confidence not filled";
    if (confidence >= 0.9) return `high confidence (confidence=${confidence.toFixed(2)})`;
    if (confidence >= 0.75) return `fairly confident (confidence=${confidence.toFixed(2)})`;
    if (confidence >= 0.5) return `moderately confident (confidence=${confidence.toFixed(2)})`;
    return `low confidence (confidence=${confidence.toFixed(2)})`;
  }

  function buildFailureNoteSentence(failure, idx) {
    const labelText = getCodeLabel(failure.code);
    const label = labelText ? `${failure.code} (${labelText})` : failure.code;
    const timeRange = formatNoteTimeRange(failure);
    const confidence = Number.isFinite(failure.confidence) ? Number(failure.confidence) : 1;
    const confText = describeConfidence(confidence);

    if (failure.verified === true) {
      return `${idx + 1}. For ${label}, ${confText}; confirmed present within ${timeRange}.`;
    }

    if (failure.verified === false) {
      return `${idx + 1}. For ${label}, ${confText}; judged not valid after reviewing ${timeRange}.`;
    }

    return `${idx + 1}. For ${label}, ${confText}; review not finished yet, pending inspection for ${timeRange}.`;
  }

  function buildAutoNotes(rec) {
    const failures = Array.isArray(rec?.top_failures) ? rec.top_failures.filter((f) => f && f.code) : [];
    if (!failures.length) return "";

    const lines = failures.map((failure, idx) => buildFailureNoteSentence(failure, idx));
    return lines.join("\n");
  }

  function syncAutoNotes(rec, { force = false } = {}) {
    if (!rec?.id) return;
    const next = buildAutoNotes(rec);
    const prevAuto = lastAutoNotesById.get(rec.id) || "";
    const current = String(rec.notes || "");

    if (force || !current.trim() || current === prevAuto) {
      rec.notes = next;
      if (activeSampleId === rec.id && els.notesInput) {
        els.notesInput.value = next;
      }
    }

    lastAutoNotesById.set(rec.id, next);
  }

  function rebuildVisibleSampleIds() {
    const q = String(els.searchInput.value || "").trim().toLowerCase();
    const status = String(els.statusFilter.value || "all");

    const ids = [];
    for (const s of dataset) {
      if (!s || !s.id) continue;

      const rec = annotationsById.get(s.id);
      const completed = rec ? isRecordComplete(rec) : false;

      if (status === "completed" && !completed) continue;
      if (status === "incomplete" && completed) continue;

      if (q) {
        const hay = [
          String(s.id || ""),
          String(s.video_path || ""),
          ...(Array.isArray(s.failure_codes) ? s.failure_codes : []),
          ...(rec?.top_failures || []).map((f) => f.code || ""),
        ]
          .join(" ")
          .toLowerCase();
        if (!hay.includes(q)) continue;
      }

      ids.push(s.id);
    }

    visibleSampleIds = ids;
  }

  function renderSampleList() {
    rebuildVisibleSampleIds();

    els.sampleList.innerHTML = "";

    const total = visibleSampleIds.length;
    const activeIndex = activeSampleId ? visibleSampleIds.indexOf(activeSampleId) : -1;
    els.sampleCounter.textContent = `${activeIndex >= 0 ? activeIndex + 1 : 0} / ${total}`;

    for (const id of visibleSampleIds) {
      const s = getSampleById(id);
      const rec = annotationsById.get(id);
      const completed = rec ? isRecordComplete(rec) : false;

      const item = document.createElement("div");
      item.className = "sample-item" + (id === activeSampleId ? " active" : "");
      item.tabIndex = 0;

      const idEl = document.createElement("div");
      idEl.className = "sample-id";
      idEl.textContent = id;

      const tags = document.createElement("div");
      tags.className = "sample-tags";

      const statusChip = document.createElement("span");
      statusChip.className = "chip " + (completed ? "success" : "muted");
      statusChip.textContent = completed ? "Completed" : "Incomplete";
      tags.appendChild(statusChip);

      const suggested = Array.isArray(s?.failure_codes) ? s.failure_codes : [];
      if (suggested.length) {
        const chip = document.createElement("span");
        chip.className = "chip muted";
        chip.textContent = suggested.slice(0, 3).join(" / ") + (suggested.length > 3 ? " ..." : "");
        tags.appendChild(chip);
      }

      item.appendChild(idEl);
      item.appendChild(tags);

      item.addEventListener("click", () => selectSample(id));
      item.addEventListener("keydown", (e) => {
        if (e.key === "Enter") selectSample(id);
      });

      els.sampleList.appendChild(item);
    }
  }

  function buildFailureCodeOptions(selectEl, suggestedCodes = []) {
    selectEl.innerHTML = "";

    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "Please select a code";
    selectEl.appendChild(blank);

    const allCodes = Array.from(taxonomyByCode.keys()).sort();
    const manifestCodes = Array.from(
      new Set(
        dataset
          .flatMap((s) => (Array.isArray(s?.failure_codes) ? s.failure_codes : []))
          .filter(Boolean)
      )
    ).sort();

    const merged = Array.from(new Set([...suggestedCodes, ...manifestCodes, ...allCodes])).filter(Boolean);

    for (const code of merged) {
      const opt = document.createElement("option");
      opt.value = code;
      const label = getCodeLabel(code);
      opt.textContent = label ? `${code} - ${label}` : code;
      selectEl.appendChild(opt);
    }
  }

  function markAutoFailures(sample, rec) {
    const queue = Array.isArray(sample?.failure_codes) ? sample.failure_codes.filter(Boolean) : [];
    if (!queue.length || !Array.isArray(rec?.top_failures)) return;

    const remaining = new Map();
    for (const code of queue) {
      remaining.set(code, (remaining.get(code) || 0) + 1);
    }

    for (const failure of rec.top_failures) {
      if (!failure || !failure.code) continue;
      const left = remaining.get(failure.code) || 0;
      if (left <= 0) continue;
      autoFailureRefs.add(failure);
      remaining.set(failure.code, left - 1);
    }
  }

  function createFailureCard(sample, rec, failure) {
    const frag = els.failureCardTemplate.content.cloneNode(true);
    const card = frag.querySelector(".failure-card");

    const codeSelect = frag.querySelector(".code-select");
    const codeReadonly = frag.querySelector(".code-readonly");
    const groupChip = frag.querySelector(".failure-group");
    const definitionEl = frag.querySelector(".failure-definition");
    const removeBtn = frag.querySelector(".remove-failure-btn");

    const verifiedSelect = frag.querySelector(".verified-select");
    const confidenceInput = frag.querySelector(".confidence-input");
    const t0Input = frag.querySelector(".t0-input");
    const t1Input = frag.querySelector(".t1-input");
    const setT0Btn = frag.querySelector(".set-t0-btn");
    const setT1Btn = frag.querySelector(".set-t1-btn");
    const statusEl = frag.querySelector(".card-status");

    const suggested = Array.isArray(sample?.failure_codes) ? sample.failure_codes : [];
    buildFailureCodeOptions(codeSelect, suggested);
    const isAuto = autoFailureRefs.has(failure);

    if (isAuto) {
      codeSelect.classList.add("hidden");
      codeReadonly.classList.remove("hidden");
    } else {
      codeSelect.classList.remove("hidden");
      codeReadonly.classList.add("hidden");
    }

    function clampEvidenceTime(value) {
      if (!Number.isFinite(value)) return NaN;
      const dur = els.videoPlayer.duration;
      const max = Number.isFinite(dur) && dur >= 0 ? dur : Number.POSITIVE_INFINITY;
      return Math.max(0, Math.min(value, max));
    }

    function syncUIFromFailure() {
      const t0 = clampEvidenceTime(failure?.evidence?.t0);
      const t1 = clampEvidenceTime(failure?.evidence?.t1);
      if (failure?.evidence) {
        failure.evidence.t0 = t0;
        failure.evidence.t1 = t1;
      }

      codeSelect.value = failure.code || "";
      codeReadonly.textContent = failure.code || "(not selected)";
      verifiedSelect.value = failure.verified === true ? "true" : failure.verified === false ? "false" : "";
      confidenceInput.value = Number.isFinite(failure.confidence) ? String(failure.confidence) : "1";
      t0Input.value = Number.isFinite(t0) ? String(t0) : "";
      t1Input.value = Number.isFinite(t1) ? String(t1) : "";

      const dur = els.videoPlayer.duration;
      const maxAttr = Number.isFinite(dur) && dur >= 0 ? String(dur) : "";
      t0Input.max = maxAttr;
      t1Input.max = maxAttr;

      const def = failure.code ? taxonomyByCode.get(failure.code) : null;
      if (def) {
        groupChip.textContent = `${def.group_label || def.group_id || ""}`.trim();
        groupChip.classList.remove("muted");
        const zh = getCodeLabel(failure.code);
        const en = def.definition || "";
        definitionEl.textContent = zh ? (en ? `${zh} / ${en}` : zh) : en;
      } else {
        groupChip.textContent = "";
        groupChip.classList.add("muted");
        definitionEl.textContent = failure.code ? "(No taxonomy definition available; showing code only)" : "Please select a failure code.";
      }

      const complete = isFailureComplete(failure);
      statusEl.textContent = complete ? "Completed" : "Incomplete";
      statusEl.classList.toggle("success", complete);
      statusEl.classList.toggle("muted", !complete);
      card.classList.toggle("completed", complete);

      const valid = validateFailure(failure).ok;
      card.classList.toggle("invalid", !valid && (failure.code || failure.verified !== ""));
    }

    function updateFailureFromUI() {
      if (!isAuto) {
        failure.code = String(codeSelect.value || "").trim() || "";
      }

      const v = String(verifiedSelect.value || "");
      if (v === "true") failure.verified = true;
      else if (v === "false") failure.verified = false;
      else failure.verified = "";

      const conf = parseFloat(confidenceInput.value);
      failure.confidence = Number.isFinite(conf) ? conf : 1;

      failure.evidence = failure.evidence || { t0: NaN, t1: NaN, keyframes: [] };
      failure.evidence.keyframes = Array.isArray(failure.evidence.keyframes) ? failure.evidence.keyframes : [];

      const t0 = parseFloat(t0Input.value);
      const t1 = parseFloat(t1Input.value);
      failure.evidence.t0 = clampEvidenceTime(t0);
      failure.evidence.t1 = clampEvidenceTime(t1);

      syncUIFromFailure();
      syncAutoNotes(rec);
      renderSampleList();
      updateValidationBanner();
      scheduleAutosave(rec.id);
    }

    function setTime(which) {
      const t = els.videoPlayer.currentTime;
      const v = Number.isFinite(t) ? clampEvidenceTime(Number(t.toFixed(3))) : 0;
      if (!failure.evidence) failure.evidence = { t0: NaN, t1: NaN, keyframes: [] };
      if (which === "t0") failure.evidence.t0 = v;
      if (which === "t1") failure.evidence.t1 = v;
      syncUIFromFailure();
      syncAutoNotes(rec);
      renderSampleList();
      updateValidationBanner();
      scheduleAutosave(rec.id);
    }

    codeSelect.addEventListener("change", updateFailureFromUI);
    verifiedSelect.addEventListener("change", updateFailureFromUI);
    confidenceInput.addEventListener("input", updateFailureFromUI);
    t0Input.addEventListener("input", updateFailureFromUI);
    t1Input.addEventListener("input", updateFailureFromUI);

    setT0Btn.addEventListener("click", () => setTime("t0"));
    setT1Btn.addEventListener("click", () => setTime("t1"));

    removeBtn.addEventListener("click", () => {
      const idx = rec.top_failures.indexOf(failure);
      if (idx >= 0) rec.top_failures.splice(idx, 1);
      card.remove();
      syncAutoNotes(rec, { force: true });
      renderSampleList();
      updateValidationBanner();
      scheduleAutosave(rec.id);
    });

    // Keyboard helpers within the card.
    card.addEventListener("keydown", (e) => {
      if (e.key === "i" || e.key === "I") {
        e.preventDefault();
        setTime("t0");
      }
      if (e.key === "o" || e.key === "O") {
        e.preventDefault();
        setTime("t1");
      }
    });

    syncUIFromFailure();
    return { card, frag };
  }

  function validateFailure(f) {
    if (!f) return { ok: false, message: "failure entry is empty" };
    if (!f.code) return { ok: false, message: "please select a code" };

    if (f.verified === "") return { ok: false, message: `${f.code}: please set verified` };

    if (!Number.isFinite(f.confidence) || f.confidence < 0 || f.confidence > 1) {
      return { ok: false, message: `${f.code}: confidence must be within [0,1]` };
    }

    const ev = f.evidence || {};
    if (!Number.isFinite(ev.t0) || !Number.isFinite(ev.t1)) {
      return { ok: false, message: `${f.code}: please fill in t0/t1` };
    }
    if (ev.t0 < 0 || ev.t1 < 0 || ev.t0 > ev.t1) {
      return { ok: false, message: `${f.code}: must satisfy 0 <= t0 <= t1` };
    }

    const dur = els.videoPlayer.duration;
    if (Number.isFinite(dur) && dur > 0) {
      if (ev.t0 > dur || ev.t1 > dur) {
        return { ok: false, message: `${f.code}: t0/t1 exceeds video duration` };
      }
    }

    return { ok: true, message: "" };
  }

  function updateValidationBanner() {
    if (!activeSampleId) {
      showBanner("", "");
      return;
    }

    const rec = annotationsById.get(activeSampleId);
    if (!rec) {
      showBanner("", "");
      return;
    }

    const fails = Array.isArray(rec.top_failures) ? rec.top_failures : [];
    if (!fails.length) {
      showBanner("", "");
      return;
    }

    for (const f of fails) {
      const v = validateFailure(f);
      if (!v.ok) {
        showBanner("error", v.message);
        return;
      }
    }

    // Soft quality hint: verified=false should explain why in notes.
    const hasVerifiedFalse = fails.some((f) => f && f.code && f.verified === false);
    const notes = String(rec.notes || "").trim();
    if (hasVerifiedFalse && !notes) {
      showBanner("warning", "Detected `verified=false`. Recommend explaining the reason in Notes.");
      return;
    }

    showBanner("", "");
  }

  function renderSuggestedCodes(sample) {
    if (!els.suggestedCodes) return;
    els.suggestedCodes.innerHTML = "";
    const codes = Array.isArray(sample?.failure_codes) ? sample.failure_codes : [];
    if (!codes.length) {
      const chip = document.createElement("span");
      chip.className = "chip muted";
      chip.textContent = "(none)";
      els.suggestedCodes.appendChild(chip);
      return;
    }
    for (const code of codes) {
      const chip = document.createElement("span");
      chip.className = "chip muted";
      const label = getCodeLabel(code);
      chip.textContent = label ? `${code} - ${label}` : code;
      els.suggestedCodes.appendChild(chip);
    }
  }

  function renderAnnotationPanel(sample) {
    if (!sample) return;

    const rec = getOrInitRecord(sample.id);

    // MVP requirement: auto-list manifest failure_codes as annotation cards.
    rec.top_failures = Array.isArray(rec.top_failures) ? rec.top_failures : [];
    if (rec.top_failures.length === 0) {
      const autoCodes = Array.isArray(sample.failure_codes) ? sample.failure_codes : [];
      for (const code of autoCodes) {
        if (!code) continue;
        const failure = {
          code,
          confidence: 1.0,
          verified: "",
          evidence: { t0: NaN, t1: NaN, keyframes: [] },
        };
        rec.top_failures.push(failure);
        autoFailureRefs.add(failure);
      }
    }

    markAutoFailures(sample, rec);
    syncAutoNotes(rec);

    els.recordIdInput.value = sample.id;
    els.recordAnnotatorInput.value = rec.annotator_id || getGlobalAnnotatorId();
    els.notesInput.value = rec.notes || "";

    renderSuggestedCodes(sample);
    els.failureList.innerHTML = "";
    
    // Create cards in batch to reduce DOM reflow.
    const fragment = document.createDocumentFragment();
    for (const f of rec.top_failures) {
      const { frag } = createFailureCard(sample, rec, f);
      fragment.appendChild(frag);
    }
    els.failureList.appendChild(fragment);

    updateValidationBanner();
  }

  function resolveVideoSource(pathHint) {
    const hint = normalizePath(pathHint);
    if (!hint) return null;

    const file = findFileByPathHint(hint);
    if (file) return { file, hint };

    // Fallback: when page is served by local static server, direct URL can still work.
    return { url: hint, hint };
  }

  function collectOriginalVideoPathCandidates(sample) {
    const out = [];
    const push = (v) => {
      const n = normalizePath(v);
      if (!n) return;
      if (!out.includes(n)) out.push(n);
    };

    const directKeys = [
      "original_video_path",
      "source_video_path",
      "raw_video_path",
      "reference_video_path",
      "clean_video_path",
      "source_path",
    ];
    for (const k of directKeys) {
      push(sample?.[k]);
      push(sample?.meta?.[k]);
    }

    const shotirPath = normalizePath(sample?.shotir_path || "");
    if (shotirPath) {
      push(shotirPath.replace(/_shotir(\.[a-z0-9]+)?$/i, ".mp4"));
    }

    const stem = extractSourceStem(sample);
    if (stem) {
      push(`data/${stem}.mp4`);
      push(`data/${stem}.mov`);
      push(`data/${stem}.webm`);
      push(`data/${stem}.mkv`);
    }

    const videoPathDir = dirname(sample?.video_path || "");
    if (stem && videoPathDir) {
      push(`${videoPathDir}${stem}.mp4`);
    }

    return out;
  }

  function getTargetVideoSourceForSample(sample) {
    if (!sample) return null;
    return resolveVideoSource(sample.video_path);
  }

  function getOriginalVideoSourceForSample(sample) {
    const candidates = collectOriginalVideoPathCandidates(sample);
    for (const c of candidates) {
      const source = resolveVideoSource(c);
      if (!source) continue;
      if (source.file) return source;
    }

    // No locally bound file found, still try first URL candidate.
    if (candidates.length) return resolveVideoSource(candidates[0]);
    return null;
  }

  function setVideoElementSource(player, source) {
    if (!player) return { objectUrl: null, file: null };

    player.pause();
    player.removeAttribute("src");
    player.querySelectorAll("source").forEach((s) => s.remove());

    if (!source) {
      player.load();
      return { objectUrl: null, file: null };
    }

    player.preload = "auto";
    if (source.file) {
      const objectUrl = URL.createObjectURL(source.file);
      player.src = objectUrl;
      player.load();
      return { objectUrl, file: source.file };
    }

    player.src = source.url;
    player.load();
    return { objectUrl: null, file: null };
  }

  function setOriginalVideoHint(message) {
    if (!els.originalVideoHint) return;
    if (!message) {
      els.originalVideoHint.classList.add("hidden");
      els.originalVideoHint.textContent = "";
      return;
    }
    els.originalVideoHint.textContent = message;
    els.originalVideoHint.classList.remove("hidden");
  }

  function showVideoNotFoundDetails(sample, hintOverride) {
    const hint = normalizePath(hintOverride || sample?.video_path || "");
    const size = boundDir.filesByRelPath.size;
    els.videoOverlay.classList.remove("hidden");
    if (size === 0) {
      els.videoOverlay.textContent =
        `No local directory is bound.\n` +
        `Expected file: ${hint || "(empty)"}\n\n` +
        `Click "Select Data Directory" at the top,\n` +
        `then choose the folder containing the video files (usually out/dataset_v0/ or data/).`;
    } else {
      els.videoOverlay.textContent =
        `A directory is bound, but this file was not found.\n` +
        `Expected file: ${hint || "(empty)"}\n` +
        `Indexed ${size} files. Please check whether the path is correct.`;
    }
  }

  function setVideoSourcesForSample(sample) {
    const targetSource = getTargetVideoSourceForSample(sample);
    const originalSource = getOriginalVideoSourceForSample(sample);

    // Always stop both before switching source.
    getPlayers().forEach((player) => player.pause());

    if (currentVideoObjectUrl) {
      URL.revokeObjectURL(currentVideoObjectUrl);
      currentVideoObjectUrl = null;
    }
    if (currentOriginalVideoObjectUrl) {
      URL.revokeObjectURL(currentOriginalVideoObjectUrl);
      currentOriginalVideoObjectUrl = null;
    }
    currentVideoFile = null;
    currentOriginalVideoFile = null;

    const targetBind = setVideoElementSource(els.videoPlayer, targetSource);
    currentVideoObjectUrl = targetBind.objectUrl;
    currentVideoFile = targetBind.file;

    const originalBind = setVideoElementSource(els.videoPlayerOriginal, originalSource);
    currentOriginalVideoObjectUrl = originalBind.objectUrl;
    currentOriginalVideoFile = originalBind.file;

    if (targetSource) {
      els.videoOverlay.classList.add("hidden");
    } else {
      showVideoNotFoundDetails(sample);
    }

    if (originalSource) {
      setOriginalVideoHint("");
    } else {
      const candidates = collectOriginalVideoPathCandidates(sample);
      const bestHint = candidates.length ? candidates[0] : `${extractSourceStem(sample) || "videoX"}.mp4`;
      setOriginalVideoHint(`Original video not found: ${bestHint}`);
    }
  }

  function updateMeta(sample) {
    els.currentTitle.textContent = sample ? sample.id : "No Sample Selected";
    els.metaId.textContent = sample?.id || "-";
    els.metaVideoPath.textContent = sample?.video_path || "-";
    els.metaFailureCodes.textContent =
      Array.isArray(sample?.failure_codes) && sample.failure_codes.length
        ? sample.failure_codes.join(", ")
        : "-";

    els.metaPerturb.textContent = sample?.perturbation_type || "-";

    if (sample?.shotir_path) {
      els.metaShotir.textContent = `ShotIR: ${basename(sample.shotir_path)}`;
      els.metaShotir.classList.remove("muted");
    } else {
      els.metaShotir.textContent = "ShotIR: -";
      els.metaShotir.classList.add("muted");
    }
  }

  function selectSample(id) {
    if (!id) return;
    const sample = getSampleById(id);
    if (!sample) return;
    activeSampleId = id;

    updateMeta(sample);

    // Render the side panel and list first to avoid conflicts between large DOM writes and media loading.
    // Keep annotator id in sync.
    const rec = getOrInitRecord(sample.id);
    rec.annotator_id =
      String(els.recordAnnotatorInput.value || getGlobalAnnotatorId()).trim() || getGlobalAnnotatorId();

    renderSampleList();
    renderAnnotationPanel(sample);

    // Switch video sources last and then call load().
    requestAnimationFrame(() => {
      if (activeSampleId !== sample.id) return;
      setVideoSourcesForSample(sample);
    });
  }

  function moveSample(delta) {
    if (!visibleSampleIds.length) return;
    const idx = activeSampleId ? visibleSampleIds.indexOf(activeSampleId) : -1;
    const nextIdx = Math.max(0, Math.min(visibleSampleIds.length - 1, (idx >= 0 ? idx : 0) + delta));
    selectSample(visibleSampleIds[nextIdx]);
  }

  function downloadBlob(filename, blob) {
    const a = document.createElement("a");
    const url = URL.createObjectURL(blob);
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  }

  function exportCurrentRecord() {
    if (!activeSampleId) return;
    const rec = annotationsById.get(activeSampleId) || getOrInitRecord(activeSampleId);
    const pretty = JSON.stringify(rec, null, 2);
    downloadBlob(`${activeSampleId}__annotation.json`, new Blob([pretty], { type: "application/json" }));
  }

  function exportAllRecordsAsJsonl() {
    const records = [];
    for (const s of dataset) {
      if (!s?.id) continue;
      const rec = annotationsById.get(s.id);
      if (!rec) continue;

      const hasFailure = Array.isArray(rec.top_failures) && rec.top_failures.some((f) => f && f.code);
      const hasNotes = String(rec.notes || "").trim().length > 0;
      if (!hasFailure && !hasNotes) continue;

      records.push(rec);
    }

    const lines = records.map((r) => JSON.stringify(r)).join("\n") + (records.length ? "\n" : "");
    downloadBlob("annotations.jsonl", new Blob([lines], { type: "application/jsonl" }));
  }

  function saveCurrentToLocalStorage() {
    if (!activeSampleId) return;
    const rec = annotationsById.get(activeSampleId) || getOrInitRecord(activeSampleId);
    rec.annotator_id = String(els.recordAnnotatorInput.value || getGlobalAnnotatorId()).trim() || getGlobalAnnotatorId();
    rec.notes = String(els.notesInput.value || "");

    flushAutosaveNow({ ids: [activeSampleId], interactiveFile: true })
      .catch(() => {
        // ignore
      })
      .finally(() => {
        setPill(els.datasetStatus, `Saved: ${new Date().toLocaleTimeString()}`, "success");
        renderSampleList();
      });
  }

  function addEmptyFailure() {
    if (!activeSampleId) return;
    const sample = getSampleById(activeSampleId);
    if (!sample) return;

    const rec = annotationsById.get(activeSampleId) || getOrInitRecord(activeSampleId);

    const f = {
      code: "",
      confidence: 1.0,
      verified: "",
      evidence: { t0: NaN, t1: NaN, keyframes: [] },
    };

    rec.top_failures = Array.isArray(rec.top_failures) ? rec.top_failures : [];
    rec.top_failures.push(f);

    const { frag } = createFailureCard(sample, rec, f);
    els.failureList.appendChild(frag);

    syncAutoNotes(rec);
    renderSampleList();
    updateValidationBanner();
    scheduleAutosave(activeSampleId);
  }

  function wireVideoControls() {
    const players = getPlayers();

    function getLeaderPlayer() {
      if (hasVideoSource(els.videoPlayer)) return els.videoPlayer;
      if (hasVideoSource(els.videoPlayerOriginal)) return els.videoPlayerOriginal;
      return els.videoPlayer;
    }

    function getSharedDuration() {
      const durations = players
        .filter((p) => hasVideoSource(p))
        .map((p) => p.duration)
        .filter((d) => Number.isFinite(d) && d > 0);
      if (!durations.length) return NaN;
      return Math.min(...durations);
    }

    function clampVideoTime(value) {
      const max = getSharedDuration();
      const upper = Number.isFinite(max) ? max : Number.POSITIVE_INFINITY;
      return Math.max(0, Math.min(value, upper));
    }

    function syncTimelineRange() {
      const d = getSharedDuration();
      els.timelineInput.max = Number.isFinite(d) ? String(d) : "0";
      els.durationLabel.textContent = formatTime(d);
    }

    function updateTimeUI(timeValue) {
      const t = Number.isFinite(timeValue) ? timeValue : 0;
      els.currentTimeLabel.textContent = formatTime(t);
      els.timelineInput.value = String(t);
      els.jumpTimeInput.value = t.toFixed(3);

      const fps = parseFloat(els.fpsInput.value);
      const step = Number.isFinite(fps) && fps > 0 ? 1 / fps : 1 / 30;
      els.metaTimeState.textContent = `${t.toFixed(3)}s / ${step.toFixed(4)}s`;
    }

    function setSyncedCurrentTime(timeValue, source = null) {
      const t = clampVideoTime(timeValue);
      for (const player of players) {
        if (!hasVideoSource(player)) continue;
        if (source && player === source) continue;
        if (Math.abs((player.currentTime || 0) - t) <= 0.02) continue;
        try {
          player.currentTime = t;
        } catch {
          // ignore
        }
      }
      updateTimeUI(t);
    }

    function syncTimeFrom(player) {
      if (!hasVideoSource(player)) return;
      setSyncedCurrentTime(player.currentTime, player);
    }

    function setSyncedPlayback(shouldPlay, source = null) {
      if (isSyncingPlayback) return;
      isSyncingPlayback = true;

      for (const player of players) {
        if (!hasVideoSource(player)) continue;
        if (source && player === source) continue;
        if (shouldPlay === !player.paused) continue;

        if (shouldPlay) {
          const p = player.play();
          if (p && typeof p.catch === "function") p.catch(() => {});
        } else {
          player.pause();
        }
      }

      queueMicrotask(() => {
        isSyncingPlayback = false;
      });
    }

    function jumpToManualTime() {
      const v = parseFloat(els.jumpTimeInput.value);
      if (!Number.isFinite(v)) return;
      setSyncedCurrentTime(v);
    }

    els.playPauseBtn.addEventListener("click", () => {
      const leader = getLeaderPlayer();
      if (!hasVideoSource(leader)) return;
      if (leader.paused) {
        const p = leader.play();
        if (p && typeof p.catch === "function") p.catch(() => {});
      } else {
        leader.pause();
      }
    });

    document.querySelectorAll("button[data-seek]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const delta = parseFloat(btn.getAttribute("data-seek"));
        if (!Number.isFinite(delta)) return;
        const leader = getLeaderPlayer();
        const t = (Number.isFinite(leader.currentTime) ? leader.currentTime : 0) + delta;
        setSyncedCurrentTime(t);
      });
    });

    function frameStepSeconds() {
      const fps = parseFloat(els.fpsInput.value);
      if (!Number.isFinite(fps) || fps <= 0) return 1 / 30;
      return 1 / fps;
    }

    els.prevFrameBtn.addEventListener("click", () => {
      const leader = getLeaderPlayer();
      setSyncedCurrentTime((Number.isFinite(leader.currentTime) ? leader.currentTime : 0) - frameStepSeconds());
    });

    els.nextFrameBtn.addEventListener("click", () => {
      const leader = getLeaderPlayer();
      setSyncedCurrentTime((Number.isFinite(leader.currentTime) ? leader.currentTime : 0) + frameStepSeconds());
    });

    els.timelineInput.addEventListener("input", () => {
      const v = parseFloat(els.timelineInput.value);
      if (Number.isFinite(v)) setSyncedCurrentTime(v);
    });

    els.jumpTimeBtn.addEventListener("click", jumpToManualTime);
    els.jumpTimeInput.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      e.preventDefault();
      jumpToManualTime();
    });

    for (const player of players) {
      player.addEventListener("loadedmetadata", () => {
        syncTimelineRange();
        const leader = getLeaderPlayer();
        updateTimeUI(Number.isFinite(leader.currentTime) ? leader.currentTime : 0);
      });

      player.addEventListener("timeupdate", () => {
        syncTimeFrom(player);
      });

      player.addEventListener("seeking", () => {
        syncTimeFrom(player);
      });

      player.addEventListener("play", () => {
        setSyncedPlayback(true, player);
      });

      player.addEventListener("pause", () => {
        setSyncedPlayback(false, player);
      });
    }

    // Global keyboard shortcuts (minimal, aligned with labels)
    document.addEventListener("keydown", (e) => {
      if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT")) {
        // Let form inputs handle typing; card-level handlers still work.
        return;
      }

      if (e.key === " ") {
        e.preventDefault();
        els.playPauseBtn.click();
        return;
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        const leader = getLeaderPlayer();
        setSyncedCurrentTime((Number.isFinite(leader.currentTime) ? leader.currentTime : 0) - 0.1);
        return;
      }
      if (e.key === "ArrowRight") {
        e.preventDefault();
        const leader = getLeaderPlayer();
        setSyncedCurrentTime((Number.isFinite(leader.currentTime) ? leader.currentTime : 0) + 0.1);
        return;
      }

      if (e.key === "[") {
        e.preventDefault();
        els.prevFrameBtn.click();
      } else if (e.key === "]") {
        e.preventDefault();
        els.nextFrameBtn.click();
      }
    });

    els.videoPlayer.addEventListener("error", () => {
      els.videoOverlay.classList.remove("hidden");
      const src = String(els.videoPlayer.currentSrc || els.videoPlayer.src || "");
      if (src.startsWith("blob:")) {
        els.videoOverlay.textContent =
          "Failed to load the video-to-annotate (blob). Common causes: unsupported codec or corrupted file. Try Edge/Chrome or rebind the directory.";
      } else {
        const errCode = els.videoPlayer.error ? els.videoPlayer.error.code : "?";
        els.videoOverlay.textContent =
          `Failed to load the video-to-annotate (URL, error=${errCode}). Current src: ${src || "(empty)"}\n` +
          'Make sure the page is opened from a local static server and the video path is accessible, or use "Select Data Directory".';
      }
    });

    if (els.videoPlayerOriginal) {
      els.videoPlayerOriginal.addEventListener("error", () => {
        const src = String(els.videoPlayerOriginal.currentSrc || els.videoPlayerOriginal.src || "");
        if (!src) {
          setOriginalVideoHint("Original video not found");
          return;
        }
        setOriginalVideoHint("Failed to load original video. Please check the path or codec.");
      });
    }

    syncTimelineRange();
    updateTimeUI(0);
  }

  function wireDatasetControls() {
    els.prevSampleBtn.addEventListener("click", () => moveSample(-1));
    els.nextSampleBtn.addEventListener("click", () => moveSample(1));

    els.searchInput.addEventListener("input", () => renderSampleList());
    els.statusFilter.addEventListener("change", () => {
      renderSampleList();
      // If active sample is filtered out, jump to first visible.
      if (activeSampleId && !visibleSampleIds.includes(activeSampleId)) {
        if (visibleSampleIds.length) selectSample(visibleSampleIds[0]);
      }
    });

    els.addFailureBtn.addEventListener("click", addEmptyFailure);

    els.notesInput.addEventListener("input", () => {
      if (!activeSampleId) return;
      const rec = annotationsById.get(activeSampleId) || getOrInitRecord(activeSampleId);
      rec.notes = String(els.notesInput.value || "");
      renderSampleList();
      scheduleAutosave(activeSampleId);
    });

    els.recordAnnotatorInput.addEventListener("input", () => {
      if (!activeSampleId) return;
      const rec = annotationsById.get(activeSampleId) || getOrInitRecord(activeSampleId);
      rec.annotator_id = String(els.recordAnnotatorInput.value || "").trim() || getGlobalAnnotatorId();
      scheduleAutosave(activeSampleId);
    });
  }

  async function tryLoadDefault() {
    setPill(els.datasetStatus, "Loading manifest...", "warning");
    showBanner("", "");

    try {
      await loadTaxonomy();
      const path = String(els.manifestPathInput.value || "").trim();
      const items = await loadManifestFromPath(path);
      if (!items.length) throw new Error("manifest is empty or failed to parse");

      dataset = items;
      setPill(els.datasetStatus, `Loaded dataset: ${dataset.length} items`, "success");

      // Ensure taxonomy includes any codes seen in manifest.
      for (const s of dataset) {
        for (const c of Array.isArray(s?.failure_codes) ? s.failure_codes : []) {
          if (!taxonomyByCode.has(c)) taxonomyByCode.set(c, { code: c });
        }
      }

      renderSampleList();

      if (visibleSampleIds.length) {
        selectSample(visibleSampleIds[0]);
      } else {
        showBanner("error", "No visible samples to display. Check the filters.");
      }

    } catch (err) {
      setPill(els.datasetStatus, "Load failed", "error");
      els.videoOverlay.classList.remove("hidden");
      els.videoOverlay.textContent = 'Default manifest load failed: click "Select Data Directory"';
      showBanner("error", String(err?.message || err));
      dataset = [];
      renderSampleList();
    }
  }

  async function tryLoadFromBoundDirectory() {
    if (!boundDir.filesByRelPath.size) return;

    // Prefer user-specified manifest path.
    const hint = String(els.manifestPathInput.value || "").trim();
    let manifestFile = hint ? findFileByPathHint(hint) : null;

    if (!manifestFile) {
      manifestFile =
        findFileByPathHint("manifest.jsonl") ||
        findFileByPathHint("out/dataset_v0/manifest.jsonl") ||
        findFileByPathHint("dataset_v0/manifest.jsonl");
    }

    if (!manifestFile) {
      showBanner("error", "manifest.jsonl was not found in the selected directory");
      return;
    }

    setPill(els.datasetStatus, "Loading manifest from directory...", "warning");

    try {
      await loadTaxonomy();
      const items = parseJsonl(await readFileText(manifestFile));
      if (!items.length) throw new Error("manifest is empty or failed to parse");
      dataset = items;

      for (const s of dataset) {
        for (const c of Array.isArray(s?.failure_codes) ? s.failure_codes : []) {
          if (!taxonomyByCode.has(c)) taxonomyByCode.set(c, { code: c });
        }
      }

      setPill(els.datasetStatus, `Loaded dataset: ${dataset.length} items`, "success");
      renderSampleList();
      if (visibleSampleIds.length) selectSample(visibleSampleIds[0]);
    } catch (err) {
      setPill(els.datasetStatus, "Load failed", "error");
      showBanner("error", String(err?.message || err));
      dataset = [];
      renderSampleList();
    }
  }

  function wireTopButtons() {
    els.loadDefaultBtn.addEventListener("click", () => {
      tryLoadDefault();
    });

    els.pickDirectoryBtn.addEventListener("click", () => {
      els.directoryInput.click();
    });

    els.directoryInput.addEventListener("change", async () => {
      bindDirectoryFiles(els.directoryInput.files);
      await tryLoadFromBoundDirectory();
    });

    els.saveBtn.addEventListener("click", () => saveAllAnnotationsToDownload());

    // Keep annotator in sync with the record panel.
    els.annotatorInput.addEventListener("input", () => {
      const a = getGlobalAnnotatorId();
      els.recordAnnotatorInput.value = a;
      if (activeSampleId) {
        const rec = annotationsById.get(activeSampleId) || getOrInitRecord(activeSampleId);
        rec.annotator_id = a;
        scheduleAutosave(activeSampleId);
      }
    });
  }

  function init() {
    hydrateFromLocalStorage();

    setPill(els.datasetStatus, "Dataset not loaded", "muted");
    updateStorageStatusPill();

    els.videoOverlay.classList.remove("hidden");

    wireTopButtons();
    wireVideoControls();
    wireDatasetControls();

    // Best-effort flush when the page is backgrounded/closed.
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") {
        flushAutosaveNow().catch(() => {
          // ignore
        });
      }
    });
    window.addEventListener("pagehide", () => {
      flushAutosaveNow().catch(() => {
        // ignore
      });
    });

    // Auto-load default manifest (best effort).
    tryLoadDefault();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();


