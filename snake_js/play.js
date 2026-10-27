/**
 * Observe tab: watch a snake agent play.
 */
(function (global) {
  "use strict";

  var E = global.SNAKE_ENV;
  var C = global.SNAKE_CONSTANTS;

  var CELL = 14;
  /*
   * Labeled speeds (UI). Real step rate is SPEED_RATE_SCALE times the label
   * (shifted 2 notches vs an older ladder): labeled 1 = former speed 4.
   * Ends at labeled 1024 via two extra fast notches (actual 2048 / 4096).
   */
  var SPEEDS = [
    0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
  ];
  var SPEED_RATE_SCALE = 4;
  var DRAW_MS = 33;
  /* First-paint Observe defaults (entertaining demo). */
  var OBSERVE_DEFAULT_SIZE = 10;
  var OBSERVE_DEFAULT_WIN_PCT = 100;
  var OBSERVE_DEFAULT_SPEED_IDX = 4; /* labeled speed 1 */

  function stepMsForSpeed(labelSpeed) {
    return 500 / (labelSpeed * SPEED_RATE_SCALE);
  }

  function formatSpeedLabel(speed) {
    if (speed >= 1) {
      return "speed " + speed;
    }
    if (speed === 0.5) {
      return "speed 1/2";
    }
    if (speed === 0.25) {
      return "speed 1/4";
    }
    if (speed === 0.125) {
      return "speed 1/8";
    }
    if (speed === 0.0625) {
      return "speed 1/16";
    }
    return "speed " + speed;
  }

  /** Map internal TSH phase ids to Observe log labels. */
  function formatTsPhaseForLog(phase) {
    if (phase === "tail") {
      return "tail_safe";
    }
    if (phase === "heal") {
      return "transfer_to_hamilton";
    }
    return phase || "tail_safe";
  }

  /** Phase label for multiphase Observe bots; null if single-phase. */
  function observePhaseLabel(family, env) {
    if (!env || !family) {
      return null;
    }
    if (family === "plan_ahead") {
      return env._planPhase || "plan";
    }
    if (family === "tail_safe_hybrid") {
      return formatTsPhaseForLog(env._tsStallPhase || "tail");
    }
    if (family === "master") {
      /* Odd Master = TSH (tail/heal/hamilton); even = pure Cell Tree. */
      if (env.size & 1) {
        return formatTsPhaseForLog(env._tsStallPhase || "tail");
      }
      return "cell_tree";
    }
    return null;
  }

  /* Head -> tip: green, blue, purple, red, orange, yellow. */
  var RAINBOW_STOPS = [
    [67, 160, 71],
    [30, 136, 229],
    [142, 36, 170],
    [229, 57, 53],
    [251, 140, 0],
    [253, 216, 53]
  ];
  var RAINBOW_LAST = RAINBOW_STOPS.length - 1;
  var _rainbowCacheLen = -1;
  var _rainbowCache = null;

  /** CSS colors for snake indices 0..len-1 (head..tip). Cached per length. */
  function rainbowPalette(len) {
    if (len === _rainbowCacheLen && _rainbowCache) {
      return _rainbowCache;
    }
    var out = new Array(len);
    var denom = len > 1 ? len - 1 : 1;
    var i;
    for (i = 0; i < len; i += 1) {
      var seg = (i / denom) * RAINBOW_LAST;
      var si = seg | 0;
      if (si >= RAINBOW_LAST) {
        si = RAINBOW_LAST - 1;
        seg = RAINBOW_LAST;
      }
      var f = seg - si;
      var a = RAINBOW_STOPS[si];
      var b = RAINBOW_STOPS[si + 1];
      out[i] = "rgb(" +
        ((a[0] + (b[0] - a[0]) * f + 0.5) | 0) + "," +
        ((a[1] + (b[1] - a[1]) * f + 0.5) | 0) + "," +
        ((a[2] + (b[2] - a[2]) * f + 0.5) | 0) + ")";
    }
    _rainbowCacheLen = len;
    _rainbowCache = out;
    return out;
  }

  function cellPxForSize(size) {
    if (size > 70) {
      return 5;
    }
    if (size > 40) {
      return 8;
    }
    return CELL;
  }

  function PlayMode(app, rootEl) {
    this.app = app;
    this.rootEl = rootEl;
    this.env = new E.SnakeEnv({ size: OBSERVE_DEFAULT_SIZE });
    this.algoId = "master";
    this.running = false;
    this.speedIdx = OBSERVE_DEFAULT_SPEED_IDX;
    this.winPct = OBSERVE_DEFAULT_WIN_PCT;
    this.fixedSeed = null;
    this.rainbowSnake = false;
    this._timer = null;
    this._nextDrawAt = 0;
    this._stepAccum = 0;
    this._playAgent = null;
    this._playFamily = null;
    this._playAlgoId = null;
    this._canvas = null;
    this.buildChrome();
  }

  PlayMode.prototype.buildChrome = function () {
    var root = this.rootEl;
    root.innerHTML = "";
    root.classList.add("snake-observe");

    var toolbar = document.createElement("div");
    toolbar.className = "toolbar";

    var botLabel = document.createElement("label");
    botLabel.className = "play-label";
    botLabel.htmlFor = "observeBot";
    botLabel.textContent = "Bot";
    toolbar.appendChild(botLabel);

    this.botSelect = document.createElement("select");
    this.botSelect.id = "observeBot";
    this.botSelect.className = "play-select";
    toolbar.appendChild(this.botSelect);

    var sizeLabel = document.createElement("label");
    sizeLabel.className = "play-label";
    sizeLabel.htmlFor = "observeSize";
    sizeLabel.textContent = "Size";
    toolbar.appendChild(sizeLabel);

    this.sizeInput = document.createElement("input");
    this.sizeInput.type = "number";
    this.sizeInput.id = "observeSize";
    this.sizeInput.className = "play-select";
    this.sizeInput.min = String(C.SIZE_MIN);
    this.sizeInput.max = String(C.SIZE_MAX);
    this.sizeInput.value = String(OBSERVE_DEFAULT_SIZE);
    this.sizeInput.style.width = "4.5rem";
    toolbar.appendChild(this.sizeInput);

    var winLabel = document.createElement("label");
    winLabel.className = "play-label";
    winLabel.htmlFor = "observeWinPct";
    winLabel.textContent = "Win at";
    toolbar.appendChild(winLabel);

    this.winSelect = document.createElement("select");
    this.winSelect.id = "observeWinPct";
    this.winSelect.className = "play-select";
    var pct;
    for (pct = 10; pct <= 100; pct += 10) {
      var opt = document.createElement("option");
      opt.value = String(pct);
      opt.textContent = pct + "% of board";
      if (pct === this.winPct) {
        opt.selected = true;
      }
      this.winSelect.appendChild(opt);
    }
    toolbar.appendChild(this.winSelect);

    this.btnStart = document.createElement("button");
    this.btnStart.type = "button";
    this.btnStart.className = "btn btn-primary";
    this.btnStart.textContent = "Start";
    toolbar.appendChild(this.btnStart);

    this.btnNew = document.createElement("button");
    this.btnNew.type = "button";
    this.btnNew.className = "btn";
    this.btnNew.textContent = "New board";
    toolbar.appendChild(this.btnNew);

    this.btnSlower = document.createElement("button");
    this.btnSlower.type = "button";
    this.btnSlower.className = "btn";
    this.btnSlower.textContent = "<<";
    toolbar.appendChild(this.btnSlower);

    this.speedLabel = document.createElement("span");
    this.speedLabel.className = "play-label";
    toolbar.appendChild(this.speedLabel);

    this.btnFaster = document.createElement("button");
    this.btnFaster.type = "button";
    this.btnFaster.className = "btn";
    this.btnFaster.textContent = ">>";
    toolbar.appendChild(this.btnFaster);

    root.appendChild(toolbar);

    var advToggle = document.createElement("button");
    advToggle.type = "button";
    advToggle.className = "observe-advanced-link";
    advToggle.textContent = "Advanced";
    advToggle.setAttribute("aria-expanded", "false");
    root.appendChild(advToggle);

    this.advancedPanel = document.createElement("div");
    this.advancedPanel.className = "observe-advanced";
    this.advancedPanel.hidden = true;

    var seedLabel = document.createElement("label");
    seedLabel.className = "play-label";
    seedLabel.htmlFor = "observeSeed";
    seedLabel.textContent = "Seed";
    this.advancedPanel.appendChild(seedLabel);

    this.seedInput = document.createElement("input");
    this.seedInput.type = "number";
    this.seedInput.id = "observeSeed";
    this.seedInput.className = "play-select";
    this.seedInput.min = "1";
    this.seedInput.step = "1";
    this.seedInput.placeholder = "random";
    this.seedInput.title = "Blank = random. New board applies.";
    this.seedInput.style.width = "8rem";
    this.advancedPanel.appendChild(this.seedInput);

    var rainbowLabel = document.createElement("label");
    rainbowLabel.className = "play-label hard-mode-toggle";
    rainbowLabel.htmlFor = "chkRainbowSnake";
    rainbowLabel.title = "Head green through blue, purple, red, orange to yellow tip.";
    this.chkRainbow = document.createElement("input");
    this.chkRainbow.type = "checkbox";
    this.chkRainbow.id = "chkRainbowSnake";
    this.chkRainbow.checked = this.rainbowSnake;
    rainbowLabel.appendChild(this.chkRainbow);
    rainbowLabel.appendChild(document.createTextNode(" Rainbow snake"));
    this.advancedPanel.appendChild(rainbowLabel);

    root.appendChild(this.advancedPanel);

    this.outcomeEl = document.createElement("div");
    this.outcomeEl.className = "observe-outcome";
    this.outcomeEl.hidden = true;
    this.outcomeEl.setAttribute("aria-live", "polite");
    root.appendChild(this.outcomeEl);

    this.statusEl = document.createElement("div");
    this.statusEl.className = "status-bar";
    this.statusEl.textContent = "Pick a bot and press Start to watch a snake run.";
    root.appendChild(this.statusEl);

    this.boardHost = document.createElement("div");
    this.boardHost.className = "snake-board-host";
    root.appendChild(this.boardHost);

    this.logEl = document.createElement("pre");
    this.logEl.className = "mono play-move-log";
    this.logEl.textContent = "(move log)";
    root.appendChild(this.logEl);

    var self = this;
    this.btnStart.addEventListener("click", function () {
      self.startRun();
    });
    this.btnNew.addEventListener("click", function () {
      self.stopRun();
      self.resetBoard();
      self.renderBoard();
      self.clearOutcome();
      self.statusEl.textContent = self.fixedSeed != null
        ? ("New board ready (seed " + self.fixedSeed + ").")
        : "New board ready.";
    });
    advToggle.addEventListener("click", function () {
      var open = self.advancedPanel.hidden;
      self.advancedPanel.hidden = !open;
      advToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    this.seedInput.addEventListener("change", function () {
      self.readFixedSeed();
    });
    this.btnSlower.addEventListener("click", function () {
      if (self.speedIdx > 0) {
        self.speedIdx -= 1;
        self.updateSpeedLabel();
      }
    });
    this.btnFaster.addEventListener("click", function () {
      if (self.speedIdx < SPEEDS.length - 1) {
        self.speedIdx += 1;
        self.updateSpeedLabel();
      }
    });
    this.sizeInput.addEventListener("change", function () {
      self.stopRun();
      self.resetBoard();
      self.renderBoard();
    });
    this.winSelect.addEventListener("change", function () {
      self.winPct = parseInt(self.winSelect.value, 10) || OBSERVE_DEFAULT_WIN_PCT;
      self.stopRun();
      self.resetBoard();
      self.renderBoard();
    });
    this.botSelect.addEventListener("change", function () {
      self.algoId = self.botSelect.value;
    });
    this.chkRainbow.addEventListener("change", function () {
      self.rainbowSnake = !!self.chkRainbow.checked;
      self.renderBoard();
    });

    this.updateSpeedLabel();
    this.resetBoard();
    this.renderBoard();
  };

  PlayMode.prototype.updateSpeedLabel = function () {
    this.speedLabel.textContent = formatSpeedLabel(SPEEDS[this.speedIdx]);
  };

  PlayMode.prototype.refreshCatalog = function () {
    var catalog = this.app.getPlayCatalog ? this.app.getPlayCatalog() : [];
    var prev = this.botSelect.value || this.algoId || "master";
    this.botSelect.innerHTML = "";
    var i;
    for (i = 0; i < catalog.length; i += 1) {
      var opt = document.createElement("option");
      opt.value = catalog[i].id;
      opt.textContent = catalog[i].name || catalog[i].label || catalog[i].id;
      this.botSelect.appendChild(opt);
    }
    this.botSelect.value = prev;
    if (!this.botSelect.value) {
      this.botSelect.value = "master";
    }
    this.algoId = this.botSelect.value || "master";
  };

  PlayMode.prototype.readSize = function () {
    var size = parseInt(this.sizeInput.value, 10);
    if (!isFinite(size)) {
      size = C.SIZE_MIN;
    }
    if (size < C.SIZE_MIN) {
      size = C.SIZE_MIN;
    }
    if (size > C.SIZE_MAX) {
      size = C.SIZE_MAX;
    }
    this.sizeInput.value = String(size);
    return size;
  };

  /** Parse Advanced seed input; blank clears fixedSeed (random boards). */
  PlayMode.prototype.readFixedSeed = function () {
    if (!this.seedInput) {
      this.fixedSeed = null;
      return null;
    }
    var raw = String(this.seedInput.value || "").trim();
    if (!raw) {
      this.fixedSeed = null;
      return null;
    }
    var seed = parseInt(raw, 10);
    if (!isFinite(seed) || seed < 1) {
      this.fixedSeed = null;
      this.seedInput.value = "";
      return null;
    }
    seed = seed >>> 0;
    if (!seed) {
      seed = 1;
    }
    this.fixedSeed = seed;
    this.seedInput.value = String(seed);
    return seed;
  };

  PlayMode.prototype.clearOutcome = function () {
    if (!this.outcomeEl) {
      return;
    }
    this.outcomeEl.hidden = true;
    this.outcomeEl.textContent = "";
    this.outcomeEl.classList.remove("observe-outcome-win", "observe-outcome-loss");
  };

  PlayMode.prototype.setEpisodeOutcome = function (info) {
    if (!this.outcomeEl || !this.env) {
      return;
    }
    var won = !!(info && info.success);
    var text;
    if (won) {
      text = "WIN - length " + this.env.snake.length + "/" + this.env.targetLength +
        " (" + this.env.apples + " apples) in " + this.env.steps + " steps.";
    } else {
      var why;
      if (info && info.death === "wall") {
        why = "hit a wall";
      } else if (info && info.death === "body") {
        why = "hit its own body";
      } else if (info && info.death === "timeout") {
        why = "ran out of steps (max " + this.env.maxSteps + ")";
      } else {
        why = (info && info.death) ? String(info.death) : "failed";
      }
      text = "LOSS - " + why + " after " + this.env.steps + " steps" +
        " (length " + this.env.snake.length + "/" + this.env.targetLength +
        ", apples " + this.env.apples + ").";
    }
    this.outcomeEl.hidden = false;
    this.outcomeEl.textContent = text;
    this.outcomeEl.classList.toggle("observe-outcome-win", won);
    this.outcomeEl.classList.toggle("observe-outcome-loss", !won);
  };

  PlayMode.prototype.applyWinTarget = function () {
    var pct = this.winPct || OBSERVE_DEFAULT_WIN_PCT;
    if (this.winSelect) {
      pct = parseInt(this.winSelect.value, 10) || pct;
      this.winPct = pct;
    }
    if (this.env && this.env.setWinFraction) {
      this.env.setWinFraction(pct / 100);
    }
    /* Observe: no step-timeout losses (Hamilton weaves at high Win-at %). */
    if (this.env) {
      this.env.disableStepTimeout = true;
    }
  };

  PlayMode.prototype.resetBoard = function () {
    var size = this.readSize();
    this.env.setSize(size);
    this.applyWinTarget();
    var APP = global.SNAKE_APP;
    var compass = !(APP && APP.isCompassOn) || APP.isCompassOn();
    var vr = C.DEFAULT_VISION;
    var seed = this.readFixedSeed();
    if (seed == null) {
      seed = (Math.random() * 0x7fffffff) | 0;
    }
    this.env.reset({
      seed: seed,
      size: size,
      visionRadius: vr,
      compassOn: compass
    });
    /* reset() restores default 25% target; re-apply Observe win %. */
    this.applyWinTarget();
    this.logEl.textContent = "(move log)";
    this._playAgent = null;
    this._playFamily = null;
    this._playAlgoId = null;
    this.clearOutcome();
    this.btnStart.textContent = "Start";
  };

  PlayMode.prototype.bindPlayAgent = function () {
    var APP = global.SNAKE_APP;
    var resolved = null;
    if (this.app && typeof this.app.resolvePlayAgent === "function") {
      resolved = this.app.resolvePlayAgent(this.algoId);
    }
    if (!resolved || !resolved.agent) {
      resolved = {
        agent: APP && APP.makeRandomAgent ? APP.makeRandomAgent() : null,
        family: "random"
      };
    }
    this._playAgent = resolved.agent;
    this._playFamily = resolved.family;
    this._playAlgoId = this.algoId;
    if (this._playAgent && typeof this._playAgent.visionRadius === "number") {
      this.env.setVision(this._playAgent.visionRadius, APP.isCompassOn ? APP.isCompassOn() : true);
    }
    if (APP && typeof APP.beginEpisodePrevDir === "function") {
      APP.beginEpisodePrevDir(this._playAgent);
    } else if (this._playAgent) {
      this._playAgent._prevDir = null;
    }
  };

  PlayMode.prototype.renderBoard = function () {
    var size = this.env.size;
    var cell = cellPxForSize(size);
    var w = size * cell;
    var canvas = this._canvas;
    if (!canvas) {
      canvas = document.createElement("canvas");
      this.boardHost.appendChild(canvas);
      this._canvas = canvas;
    }
    var dpr = (typeof window !== "undefined" && window.devicePixelRatio) ?
      window.devicePixelRatio : 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(w * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = w + "px";
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;

    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, w, w);

    var r;
    var c;
    var gap = cell - 1;
    ctx.fillStyle = "#2d4a22";
    for (r = 0; r < size; r += 1) {
      for (c = 0; c < size; c += 1) {
        ctx.fillRect(c * cell, r * cell, gap, gap);
      }
    }

    if (this.env.food) {
      ctx.fillStyle = "#e53935";
      ctx.fillRect(
        this.env.food[1] * cell,
        this.env.food[0] * cell,
        gap,
        gap
      );
    }

    var snake = this.env.snake;
    var len = snake.length;
    var i;
    var p;
    if (this.rainbowSnake) {
      var palette = rainbowPalette(len);
      for (i = len - 1; i >= 0; i -= 1) {
        p = snake[i];
        ctx.fillStyle = palette[i];
        ctx.fillRect(p[1] * cell, p[0] * cell, gap, gap);
      }
    } else {
      ctx.fillStyle = "#66bb6a";
      for (i = len - 1; i >= 1; i -= 1) {
        p = snake[i];
        ctx.fillRect(p[1] * cell, p[0] * cell, gap, gap);
      }
      if (len > 0) {
        p = snake[0];
        ctx.fillStyle = "#a5d6a7";
        ctx.fillRect(p[1] * cell, p[0] * cell, gap, gap);
      }
    }
  };

  var MOVE_LOG_MAX_LINES = 1000;

  PlayMode.prototype.appendLog = function (line) {
    var cur = this.logEl.textContent;
    var next;
    if (cur === "(move log)" || !cur) {
      next = line;
    } else {
      next = cur + "\n" + line;
    }
    var lines = next.split("\n");
    if (lines.length > MOVE_LOG_MAX_LINES) {
      lines = lines.slice(lines.length - MOVE_LOG_MAX_LINES);
      next = lines.join("\n");
    }
    this.logEl.textContent = next;
    this.logEl.scrollTop = this.logEl.scrollHeight;
  };

  PlayMode.prototype.stopRun = function () {
    this.running = false;
    if (this._timer) {
      clearTimeout(this._timer);
      this._timer = null;
    }
    /* Mid-episode pause: Resume. Fresh / finished board: Start. */
    var midEpisode = !!(this.env && !this.env.done && this.env.steps > 0);
    this.btnStart.textContent = midEpisode ? "Resume" : "Start";
  };

  PlayMode.prototype.episodeFinished = function () {
    return !!(this.env && this.env.done);
  };

  PlayMode.prototype.startRun = function () {
    if (this.running) {
      this.stopRun();
      this.statusEl.textContent = "Paused.";
      return;
    }
    /* Re-read Advanced seed on every Start. If it differs from the live board
       (typed seed then Resume, or forgot New board), reset so handoff matches. */
    var wantSeed = this.readFixedSeed();
    var seedMismatch = wantSeed != null && this.env && wantSeed !== this.env.seed;
    if (this.episodeFinished() || seedMismatch) {
      this.resetBoard();
      this.renderBoard();
    }
    this.clearOutcome();
    this.algoId = this.botSelect.value || "master";
    if (!this._playAgent || this.env.steps === 0 || this._playAlgoId !== this.algoId) {
      this.bindPlayAgent();
    }
    this.running = true;
    this.btnStart.textContent = "Pause";
    this._stepAccum = 0;
    this._nextDrawAt = performance.now();
    this.statusEl.textContent = "Running " + this.algoId +
      " [" + (this._playFamily || "?") + "] on " +
      this.env.size + "x" + this.env.size +
      " seed=" + this.env.seed;
    var startPhase = observePhaseLabel(this._playFamily, this.env);
    var food0 = this.env.food
      ? (this.env.food[0] + "," + this.env.food[1])
      : "none";
    /* One pasteable line for probes (scripts/probe-observe-seed.js). */
    this.appendLog(
      "handoff bot=" + this.algoId +
      " size=" + this.env.size +
      " seed=" + this.env.seed +
      " food0=" + food0 +
      " winPct=" + (this.winPct || OBSERVE_DEFAULT_WIN_PCT) +
      (startPhase ? " phase=" + startPhase : "")
    );
    this.drawLoop();
  };

  /** One env step; returns step result. */
  PlayMode.prototype.runOneStep = function () {
    var APP = global.SNAKE_APP;
    if (!this._playAgent) {
      this.bindPlayAgent();
    }
    if (this._playAgent && typeof this._playAgent.visionRadius === "number") {
      this.env.setVision(
        this._playAgent.visionRadius,
        APP && APP.isCompassOn ? APP.isCompassOn() : true
      );
    }
    var decision;
    if (APP && typeof APP.chooseDecision === "function") {
      decision = APP.chooseDecision(this._playAgent, this._playFamily, this.env, false);
    } else {
      decision = this.app.pickPlayMove(this.algoId, this.env);
    }
    var action = 0;
    if (decision && typeof decision === "object" && typeof decision.action === "number") {
      action = decision.action;
    } else if (typeof decision === "number") {
      action = decision;
    }
    var result = this.env.step(action);
    if (APP && typeof APP.advancePrevDir === "function") {
      APP.advancePrevDir(this._playAgent, action);
    } else if (this._playAgent) {
      this._playAgent._prevDir = action;
    }
    var name = E.ACTION_NAMES[action] || String(action);
    var note = name + " len=" + this.env.snake.length;
    var phase = observePhaseLabel(this._playFamily, this.env);
    if (phase) {
      note += " phase=" + phase;
    }
    if (result.info.success) {
      note += " WIN";
    }
    if (result.info.death) {
      note += " " + result.info.death;
      if (this.env._tsHealStall) {
        note += " healStall";
      }
      note += " seed=" + this.env.seed;
      note += " size=" + this.env.size;
    }
    /* Food note last so eats stand out at the end of the line. */
    var foodReward = typeof E.REWARD_FOOD === "number" ? E.REWARD_FOOD : 1;
    if (result.reward >= foodReward) {
      note += " apples=" + (this.env.apples || 0);
    }
    return { result: result, note: note, action: action };
  };

  /**
   * Fixed ~33ms draw cadence. Advance sim by DRAW_MS of game-time per tick;
   * may run many steps between draws at high speed.
   */
  PlayMode.prototype.drawLoop = function () {
    if (!this.running) {
      return;
    }
    var stepMs = stepMsForSpeed(SPEEDS[this.speedIdx]);
    this._stepAccum += DRAW_MS;
    var last = null;
    while (this._stepAccum >= stepMs && !this.env.done) {
      this._stepAccum -= stepMs;
      last = this.runOneStep();
      if (last.result.done) {
        break;
      }
    }
    this.renderBoard();
    if (last) {
      this.appendLog(last.note);
    }
    if (this.env.done) {
      this.stopRun();
      this.setEpisodeOutcome((last && last.result.info) || {});
      this.statusEl.textContent = "Board over. Press Start for a new board.";
      return;
    }
    this._nextDrawAt += DRAW_MS;
    var delay = Math.max(0, this._nextDrawAt - performance.now());
    var self = this;
    this._timer = setTimeout(function () {
      self.drawLoop();
    }, delay);
  };

  function initTabs(app, els) {
    var play = new PlayMode(app, els.panelPlay);

    function show(tab) {
      var isSearch = tab === "search";
      var isTrain = tab === "train";
      var isPlay = tab === "observe" || tab === "play";
      var isAbout = tab === "about";
      els.panelSearch.hidden = !isSearch;
      els.panelTrain.hidden = !isTrain;
      els.panelPlay.hidden = !isPlay;
      if (els.panelAbout) {
        els.panelAbout.hidden = !isAbout;
      }
      els.tabSearch.classList.toggle("tab-active", isSearch);
      els.tabTrain.classList.toggle("tab-active", isTrain);
      els.tabPlay.classList.toggle("tab-active", isPlay);
      if (els.tabAbout) {
        els.tabAbout.classList.toggle("tab-active", isAbout);
        els.tabAbout.setAttribute("aria-selected", isAbout ? "true" : "false");
      }
      els.tabSearch.setAttribute("aria-selected", isSearch ? "true" : "false");
      els.tabTrain.setAttribute("aria-selected", isTrain ? "true" : "false");
      els.tabPlay.setAttribute("aria-selected", isPlay ? "true" : "false");
      if (els.lab) {
        els.lab.setActiveTab(
          isSearch ? "search" : (isTrain ? "train" : (isAbout ? "about" : "observe"))
        );
      }
      /* Hide save countdown on Observe/About; autosave is also paused there. */
      var saveEl = document.getElementById("labSaveStatus");
      if (saveEl) {
        saveEl.hidden = !!(isPlay || isAbout);
      }
      if (isPlay) {
        play.refreshCatalog();
        play.renderBoard();
      }
    }

    els.tabSearch.addEventListener("click", function () {
      show("search");
    });
    els.tabTrain.addEventListener("click", function () {
      show("train");
    });
    els.tabPlay.addEventListener("click", function () {
      show("play");
    });
    if (els.tabAbout) {
      els.tabAbout.addEventListener("click", function () {
        show("about");
      });
    }

    /* Default: Observe demo - Master on 10x10 to 100%, already running. */
    show("play");
    play.startRun();
    return play;
  }

  global.SNAKE_PLAY = {
    PlayMode: PlayMode,
    initTabs: initTabs
  };
})(typeof window !== "undefined" ? window : global);
