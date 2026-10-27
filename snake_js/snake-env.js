/**
 * Snake environment for Shape Search.
 * Obs: (2r+1)^2 - 1 cells (skip head) coded -2 wall/OOB, -1 body, 0 empty, +1 food;
 * optional 4-bit food compass. Actions N/E/S/W. Death on wall/body. Food seeded.
 */
(function (global) {
  "use strict";

  var C = global.SNAKE_CONSTANTS;

  var DIRS = [
    [-1, 0],
    [0, 1],
    [1, 0],
    [0, -1]
  ];
  var ACTION_NAMES = ["N", "E", "S", "W"];

  var CELL_WALL = -2;
  var CELL_BODY = -1;
  var CELL_EMPTY = 0;
  var CELL_FOOD = 1;

  var REWARD_FOOD = 1;
  var REWARD_DEATH = -20;
  /* Per Manhattan step closer (+) / farther (-) to current food. Equal magnitude. */
  var REWARD_APPROACH = 0.1;

  /* Shared scratch for grid BFS / openings (avoids string-key body maps). */
  var _occScratch = null;
  var _occScratchN = 0;
  var _seenStamp = null;
  var _seenStampN = 0;
  var _seenGen = 1;
  var _bfsQr = null;
  var _bfsQc = null;
  var _bfsDist = null;
  var _bfsCap = 0;

  function occEnsureScratch(n) {
    var need = n * n;
    if (!_occScratch || _occScratchN < need) {
      _occScratch = new Uint8Array(need);
      _occScratchN = need;
    }
    if (!_seenStamp || _seenStampN < need) {
      _seenStamp = new Uint32Array(need);
      _seenStampN = need;
      _seenGen = 1;
    }
    if (!_bfsQr || _bfsCap < need) {
      _bfsQr = new Int16Array(need);
      _bfsQc = new Int16Array(need);
      _bfsDist = new Int16Array(need);
      _bfsCap = need;
    }
    return need;
  }

  function occNextStamp() {
    _seenGen += 1;
    if (_seenGen >= 0xfffffffe) {
      _seenStamp.fill(0);
      _seenGen = 1;
    }
    return _seenGen;
  }

  function occFillFromSnake(occ, n, snake) {
    var need = n * n;
    var i;
    for (i = 0; i < need; i += 1) {
      occ[i] = 0;
    }
    if (!snake) {
      return;
    }
    for (i = 0; i < snake.length; i += 1) {
      occ[snake[i][0] * n + snake[i][1]] = 1;
    }
  }

  /**
   * Leave-time (expiration): tip=1, neck=2, ..., head=L. Empty cells stay 0.
   * A walk of depth d may enter a body cell when leave[idx] <= d.
   */
  function leaveFillFromSnake(leave, n, snake) {
    var need = n * n;
    var i;
    var L;
    for (i = 0; i < need; i += 1) {
      leave[i] = 0;
    }
    if (!snake) {
      return;
    }
    L = snake.length;
    for (i = 0; i < L; i += 1) {
      leave[snake[i][0] * n + snake[i][1]] = L - i;
    }
  }

  /** O(L) leave write after a real step; clear vacated tip when non-eat. */
  function leaveSyncAfterStep(leave, n, snake, freedTip) {
    var i;
    var L;
    if (freedTip) {
      leave[freedTip[0] * n + freedTip[1]] = 0;
    }
    if (!snake) {
      return;
    }
    L = snake.length;
    for (i = 0; i < L; i += 1) {
      leave[snake[i][0] * n + snake[i][1]] = L - i;
    }
  }

  function occCopy(dst, src, n) {
    var need = n * n;
    var i;
    for (i = 0; i < need; i += 1) {
      dst[i] = src[i];
    }
  }

  /**
   * Apply candidate move onto occupancy (tip clear before head). Returns an
   * undo record so the caller can restore without a full-grid copy.
   */
  function occApplyMoveDelta(occ, n, snake, nr, nc, eating) {
    var tip = snake[snake.length - 1];
    var headIdx = nr * n + nc;
    var headWas = occ[headIdx];
    var tipIdx = -1;
    var tipWas = 0;
    if (!eating && tip) {
      tipIdx = tip[0] * n + tip[1];
      tipWas = occ[tipIdx];
      if (tipIdx !== headIdx) {
        occ[tipIdx] = 0;
      }
    }
    occ[headIdx] = 1;
    return {
      headIdx: headIdx,
      headWas: headWas,
      tipIdx: tipIdx,
      tipWas: tipWas
    };
  }

  function occUndoMoveDelta(occ, undo) {
    if (!undo) {
      return;
    }
    occ[undo.headIdx] = undo.headWas;
    if (undo.tipIdx >= 0 && undo.tipIdx !== undo.headIdx) {
      occ[undo.tipIdx] = undo.tipWas;
    }
  }

  function occFingerprint(env) {
    var s = env.snake;
    if (!s || !s.length) {
      return "0";
    }
    var tip = s[s.length - 1];
    return env.size + ":" + s.length + ":" + s[0][0] + "," + s[0][1] + ":" +
      tip[0] + "," + tip[1];
  }

  function occCountBody(occ, n) {
    var need = n * n;
    var i;
    var c = 0;
    for (i = 0; i < need; i += 1) {
      if (occ[i]) {
        c += 1;
      }
    }
    return c;
  }

  /**
   * Ensure env._occ matches snake. Rebuilds when dirty, resized, or fingerprint
   * / body-count mismatch (tests assign env.snake directly).
   */
  function ensureEnvOccupancy(env) {
    if (!env) {
      return;
    }
    var n = env.size;
    var need = n * n;
    var fp;
    var rebuild = false;
    if (!env._occ || env._occ.length !== need) {
      env._occ = new Uint8Array(need);
      rebuild = true;
    }
    if (!env._leave || env._leave.length !== need) {
      env._leave = new Int16Array(need);
      rebuild = true;
    }
    fp = occFingerprint(env);
    if (rebuild || env._occDirty || env._occFp !== fp) {
      rebuild = true;
    } else if (env.snake && env.snake.length &&
        occCountBody(env._occ, n) !== env.snake.length) {
      rebuild = true;
    } else if (env.snake && env.snake.length &&
        !env._occ[env.snake[0][0] * n + env.snake[0][1]]) {
      rebuild = true;
    }
    if (rebuild) {
      occFillFromSnake(env._occ, n, env.snake);
      leaveFillFromSnake(env._leave, n, env.snake);
      env._occFp = fp;
      env._occDirty = false;
      env._openDirty = true;
      env._openIncDirty = true;
    }
  }

  function markEnvOccSynced(env) {
    env._occDirty = false;
    env._occFp = occFingerprint(env);
  }

  function invalidateEnvOpenings(env) {
    env._openDirty = true;
    env._openIncDirty = true;
    env._openVac = null;
    env._openSolid = null;
  }

  /**
   * Flood from head over occ (head cell counted empty for traversal).
   * tipVacateIdx >= 0 also treated empty.
   * Returns stamp so callers can O(1)-test tip adjacency after a solid flood.
   */
  function occFloodFromHead(n, occ, hr, hc, tipVacateIdx) {
    occEnsureScratch(n);
    var stamp = occNextStamp();
    var qi = 0;
    var qe = 0;
    var count = 0;
    var d;
    var rr;
    var cc;
    var idx;
    var hIdx = hr * n + hc;
    _bfsQr[qe] = hr;
    _bfsQc[qe] = hc;
    qe += 1;
    _seenStamp[hIdx] = stamp;
    while (qi < qe) {
      hr = _bfsQr[qi];
      hc = _bfsQc[qi];
      qi += 1;
      idx = hr * n + hc;
      if (idx !== hIdx) {
        count += 1;
      }
      for (d = 0; d < 4; d += 1) {
        rr = hr + DIRS[d][0];
        cc = hc + DIRS[d][1];
        if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
          continue;
        }
        idx = rr * n + cc;
        if (_seenStamp[idx] === stamp) {
          continue;
        }
        if (occ[idx] && idx !== tipVacateIdx && idx !== hIdx) {
          continue;
        }
        _seenStamp[idx] = stamp;
        _bfsQr[qe] = rr;
        _bfsQc[qe] = cc;
        qe += 1;
      }
    }
    return {
      count: count,
      reachesTip: tipVacateIdx >= 0 && _seenStamp[tipVacateIdx] === stamp,
      stamp: stamp,
      hIdx: hIdx
    };
  }

  /**
   * Tip-reach with tip vacated == tip touches head or a solid-flood empty.
   * Equivalent to a second tip-vacate BFS after a solid flood (tipVacateIdx=-1).
   */
  function occTipReachableFromSolidFlood(n, tipIdx, hIdx, stamp) {
    if (tipIdx < 0) {
      return true;
    }
    if (tipIdx === hIdx) {
      return true;
    }
    var tr = (tipIdx / n) | 0;
    var tc = tipIdx - tr * n;
    var d;
    var rr;
    var cc;
    var nidx;
    for (d = 0; d < 4; d += 1) {
      rr = tr + DIRS[d][0];
      cc = tc + DIRS[d][1];
      if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
        continue;
      }
      nidx = rr * n + cc;
      if (nidx === hIdx || _seenStamp[nidx] === stamp) {
        return true;
      }
    }
    return false;
  }

  function makeRng(seed) {
    var s = (seed >>> 0) || 1;
    function rng() {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    }
    rng.getState = function () {
      return s;
    };
    rng.setState = function (v) {
      s = (v >>> 0) || 1;
    };
    return rng;
  }

  function targetLengthForSize(n) {
    return Math.ceil(0.25 * n * n);
  }

  /**
   * Episode step budget for the default 25% win length.
   * Old rule target*(2n)*2 is fine on small boards but under-budgets Hamilton
   * weave on larger N (e.g. 30x30 timed out). Fit from bench-maxsteps-fit.js:
   * measure Hamilton-skip steps to 25% until a size took >=10s, then
   * propMax(=2*p95) ≈ 0.157 * n^3.75; use ceil(0.16*n^3.75), never below old.
   */
  function maxEpisodeStepsForSize(n) {
    var oldBudget = targetLengthForSize(n) * (2 * n) * 2;
    var fitted = Math.ceil(0.16 * Math.pow(n, 3.75));
    return fitted > oldBudget ? fitted : oldBudget;
  }

  /** Scale the 25% budget when target length differs (e.g. Observe Win-at %). */
  function maxEpisodeStepsForTarget(n, targetLen) {
    var t25 = targetLengthForSize(n);
    var base = maxEpisodeStepsForSize(n);
    if (!t25) {
      return base;
    }
    var scaled = Math.ceil(base * (targetLen / t25));
    return scaled > base ? scaled : base;
  }

  function obsSizeForRadius(radius, compassOn) {
    var r = typeof radius === "number" ? radius : 2;
    if (r < 1) {
      r = 1;
    }
    if (r > 10) {
      r = 10;
    }
    var cells = (2 * r + 1) * (2 * r + 1) - 1;
    return cells + (compassOn ? 4 : 0);
  }

  function clampSize(n) {
    var s = typeof n === "number" ? n : C.SIZE_MIN;
    if (s < C.SIZE_MIN) {
      s = C.SIZE_MIN;
    }
    if (s > C.SIZE_MAX) {
      s = C.SIZE_MAX;
    }
    return s | 0;
  }

  function SnakeEnv(opts) {
    opts = opts || {};
    this.size = clampSize(opts.size);
    this.visionRadius = typeof opts.visionRadius === "number" ? opts.visionRadius : 2;
    this.compassOn = opts.compassOn !== false;
    this.snake = [];
    this.facing = 0;
    this.food = null;
    this.foodRng = makeRng(1);
    this.steps = 0;
    this.apples = 0;
    this.maxSteps = maxEpisodeStepsForSize(this.size);
    this.targetLength = targetLengthForSize(this.size);
    this.disableStepTimeout = false;
    this._masterPhase = "early";
    this.done = false;
    this.success = false;
    this._occ = null;
    this._leave = null;
    this._occDirty = true;
    this._occFp = "";
    this._openDirty = true;
    this._openIncDirty = true;
    this._compLab = null;
    this._compMeta = null;
    this._compNextId = 1;
  }

  SnakeEnv.prototype.setSize = function (n) {
    this.size = clampSize(n);
    this.targetLength = targetLengthForSize(this.size);
    this.maxSteps = maxEpisodeStepsForSize(this.size);
    this._occ = null;
    this._leave = null;
    this._occDirty = true;
    this._occFp = "";
    invalidateEnvOpenings(this);
    this._compLab = null;
    this._compMeta = null;
  };

  /** Win length as a fraction of board cells (e.g. 0.25). Clamped to 1..N*N. */
  SnakeEnv.prototype.setWinFraction = function (frac) {
    var f = typeof frac === "number" ? frac : 0.25;
    if (f < 0.01) {
      f = 0.01;
    }
    if (f > 1) {
      f = 1;
    }
    var cells = this.size * this.size;
    var len = Math.ceil(f * cells);
    if (len < 1) {
      len = 1;
    }
    if (len > cells) {
      len = cells;
    }
    this.targetLength = len;
    this.maxSteps = maxEpisodeStepsForTarget(this.size, len);
  };

  SnakeEnv.prototype.setVision = function (radius, compassOn) {
    if (typeof radius === "number") {
      this.visionRadius = radius;
    }
    if (typeof compassOn === "boolean") {
      this.compassOn = compassOn;
    }
  };

  SnakeEnv.prototype.inBounds = function (r, c) {
    return r >= 0 && c >= 0 && r < this.size && c < this.size;
  };

  SnakeEnv.prototype.bodySet = function () {
    var set = {};
    var i;
    for (i = 0; i < this.snake.length; i += 1) {
      set[this.snake[i][0] + "," + this.snake[i][1]] = true;
    }
    return set;
  };

  SnakeEnv.prototype.isBody = function (r, c) {
    if (this._occ && !this._occDirty && this._occ.length === this.size * this.size &&
        this._occFp === occFingerprint(this)) {
      return !!this._occ[r * this.size + c];
    }
    var i;
    for (i = 0; i < this.snake.length; i += 1) {
      if (this.snake[i][0] === r && this.snake[i][1] === c) {
        return true;
      }
    }
    return false;
  };

  SnakeEnv.prototype.nextFoodFromStream = function () {
    var n = this.size;
    var cells = n * n;
    /* When the snake covers most of the board, sample only empty cells. */
    if (this.snake.length > cells * 0.75) {
      var empties = [];
      var r;
      var c;
      var i;
      var filtered;
      for (r = 0; r < n; r += 1) {
        for (c = 0; c < n; c += 1) {
          if (!this.isBody(r, c)) {
            empties.push([r, c]);
          }
        }
      }
      /*
       * Odd boards omit (n-1,n-1) from the Hamilton cycle. Food there at
       * length n*n-2 phase-locks (one on-cycle gap stays on succ; eat has
       * freeNeighborsAfter=0 and does not win yet). Prefer any other empty;
       * at length n*n-1 the sole remaining cell (often omit) is a winning eat.
       */
      if (n % 2 === 1 && this.snake.length === cells - 2 && empties.length > 1) {
        filtered = [];
        for (i = 0; i < empties.length; i += 1) {
          if (empties[i][0] !== n - 1 || empties[i][1] !== n - 1) {
            filtered.push(empties[i]);
          }
        }
        if (filtered.length) {
          empties = filtered;
        }
      }
      if (!empties.length) {
        this.food = null;
        return null;
      }
      var pick = empties[Math.floor(this.foodRng() * empties.length)];
      this.food = pick;
      return this.food;
    }
    var attempts = 0;
    var maxAttempts = cells * 4 + 8;
    while (attempts < maxAttempts) {
      attempts += 1;
      r = Math.floor(this.foodRng() * n);
      c = Math.floor(this.foodRng() * n);
      if (this.isBody(r, c)) {
        /* Skip occupied cells; keep drawing from the seed stream. */
        continue;
      }
      this.food = [r, c];
      return this.food;
    }
    this.food = null;
    return null;
  };

  SnakeEnv.prototype.reset = function (opts) {
    opts = opts || {};
    if (typeof opts.size === "number") {
      this.setSize(opts.size);
    }
    if (typeof opts.visionRadius === "number") {
      this.visionRadius = opts.visionRadius;
    }
    if (typeof opts.compassOn === "boolean") {
      this.compassOn = opts.compassOn;
    }
    var seed = typeof opts.seed === "number" ? (opts.seed >>> 0) : ((Math.random() * 0x7fffffff) | 0);
    if (!seed) {
      seed = 1;
    }
    this.seed = seed;
    this.foodRng = makeRng(seed);
    this.steps = 0;
    this.apples = 0;
    this.done = false;
    this.success = false;
    this._masterPhase = "early";
    this._planAhead = null;
    this._planPhase = "plan";
    this._planHandoffLen = null;
    this._tsStallPhase = "tail";
    this._tsStallSeen = null;
    this._tsStallApples = null;
    this._tsHealStall = false;
    /* Cell Tree / Fringe episode markers (must not leak across seeds). */
    this._ctOddHamLock = false;
    this._ctOddLastSteps = null;
    this._ctBumpActions = null;
    this._ctBumpApproach = null;
    this._ctBumpAvoid = null;
    this._ctBumpHold = null;
    this._ctBumpCtSettle = false;
    this._ctFringeWhy = null;
    this._ctOddOpenCell = null;
    this._ctOddOpenFoodKey = null;
    this._ctOddLocks = null;
    this._ctOddLockPending = null;
    this._ctOddLockVariant = false;
    this._ctOddLockExp = false;
    this._ctOddHoleCell = null;
    this._ctOddPocket = null;
    this._ctOddTipStreak = 0;
    this._ctOddTipLen = null;
    this._ctOddPhase = null;
    this._ctOddHealStyle = null;
    this._ctOddHamFillT = null;
    this._ctOddHealSub = null;
    this._ctOddHealSkipSeen = null;
    this._ctOddHealStall = 0;
    this._ctOddStretchLen = null;
    this._ctOddHealBurstCap = null;
    this._ctOddHamChaseFrom = null;
    this._ctOddHamChaseGoal = null;
    this._ctOddHamChaseStall = 0;
    this._ctOddPostChaseHeal = false;
    this.facing = 0;
    /* Head near center, body south (facing North). Keep length-3 on-board. */
    var midC = Math.floor(this.size / 2);
    var headR = Math.min(Math.floor(this.size / 2), this.size - 3);
    if (headR < 0) {
      headR = 0;
    }
    this.snake = [
      [headR, midC],
      [headR + 1, midC],
      [headR + 2, midC]
    ];
    this._occDirty = true;
    invalidateEnvOpenings(this);
    ensureEnvOccupancy(this);
    this.nextFoodFromStream();
    if (this.done) {
      return this.observe();
    }
    return this.observe();
  };

  SnakeEnv.prototype.head = function () {
    return this.snake[0];
  };

  SnakeEnv.prototype.cellCode = function (r, c) {
    if (!this.inBounds(r, c)) {
      return CELL_WALL;
    }
    if (this.food && this.food[0] === r && this.food[1] === c) {
      return CELL_FOOD;
    }
    if (this.isBody(r, c)) {
      return CELL_BODY;
    }
    return CELL_EMPTY;
  };

  SnakeEnv.prototype.compassBits = function () {
    var bits = [0, 0, 0, 0];
    if (!this.food || !this.snake.length) {
      return bits;
    }
    var h = this.snake[0];
    var dr = this.food[0] - h[0];
    var dc = this.food[1] - h[1];
    if (dr < 0) {
      bits[0] = 1;
    }
    if (dr > 0) {
      bits[2] = 1;
    }
    if (dc > 0) {
      bits[1] = 1;
    }
    if (dc < 0) {
      bits[3] = 1;
    }
    return bits;
  };

  SnakeEnv.prototype.observe = function () {
    var r = this.visionRadius;
    if (r < 1) {
      r = 1;
    }
    if (r > 10) {
      r = 10;
    }
    var h = this.snake[0];
    var hr = h[0];
    var hc = h[1];
    var out = [];
    var dr;
    var dc;
    for (dr = -r; dr <= r; dr += 1) {
      for (dc = -r; dc <= r; dc += 1) {
        if (dr === 0 && dc === 0) {
          continue;
        }
        out.push(this.cellCode(hr + dr, hc + dc));
      }
    }
    if (this.compassOn) {
      var bits = this.compassBits();
      out.push(bits[0], bits[1], bits[2], bits[3]);
    }
    return out;
  };

  /** All four actions are always choosable (illegal moves die). */
  SnakeEnv.prototype.legalActions = function () {
    return [0, 1, 2, 3];
  };

  SnakeEnv.prototype.isImmediateDeath = function (action) {
    var a = action | 0;
    if (a < 0 || a > 3) {
      return true;
    }
    var h = this.snake[0];
    var nr = h[0] + DIRS[a][0];
    var nc = h[1] + DIRS[a][1];
    if (!this.inBounds(nr, nc)) {
      return true;
    }
    /* Moving into any body cell is death (tail does not vacate until after move;
       if not eating, tail cell frees - classic snake: tail moves so entering old
       tail is safe when not growing). */
    var grow = this.food && this.food[0] === nr && this.food[1] === nc;
    var i;
    var last = this.snake.length - 1;
    for (i = 0; i < this.snake.length; i += 1) {
      if (this.snake[i][0] === nr && this.snake[i][1] === nc) {
        if (!grow && i === last) {
          return false;
        }
        return true;
      }
    }
    return false;
  };

  SnakeEnv.prototype.safeActions = function () {
    var out = [];
    var a;
    for (a = 0; a < 4; a += 1) {
      if (!this.isImmediateDeath(a)) {
        out.push(a);
      }
    }
    return out;
  };

  SnakeEnv.prototype.manhattanToFood = function (r, c) {
    if (!this.food) {
      return 0;
    }
    return Math.abs(this.food[0] - r) + Math.abs(this.food[1] - c);
  };

  SnakeEnv.prototype.foodKnown = function () {
    if (!this.food || !this.snake.length) {
      return false;
    }
    if (this.compassOn) {
      return true;
    }
    var r = this.visionRadius;
    var h = this.snake[0];
    var fr = this.food[0];
    var fc = this.food[1];
    return Math.abs(fr - h[0]) <= r && Math.abs(fc - h[1]) <= r;
  };

  SnakeEnv.prototype.applyFoodConsumeAt = function (r, c) {
    this.apples += 1;
    this.snake.push(this.snake[this.snake.length - 1].slice());
    this._occDirty = true;
    invalidateEnvOpenings(this);
    if (this.snake.length >= this.targetLength) {
      this.food = null;
      this.done = true;
      this.success = true;
      return REWARD_FOOD;
    }
    this.nextFoodFromStream();
    return REWARD_FOOD;
  };

  SnakeEnv.prototype.step = function (action) {
    if (this.done) {
      return {
        obs: this.observe(),
        reward: 0,
        done: true,
        info: { success: this.success, apples: this.apples }
      };
    }
    var a = action | 0;
    if (a < 0 || a > 3) {
      a = this.facing;
    }
    this.steps += 1;
    var h = this.snake[0];
    var nr = h[0] + DIRS[a][0];
    var nc = h[1] + DIRS[a][1];
    this.facing = a;

    if (!this.inBounds(nr, nc)) {
      this.done = true;
      this.success = false;
      return {
        obs: this.observe(),
        reward: REWARD_DEATH,
        done: true,
        info: { success: false, apples: this.apples, death: "wall" }
      };
    }

    var eating = this.food && this.food[0] === nr && this.food[1] === nc;
    var lastIdx = this.snake.length - 1;
    var tipR = this.snake[lastIdx][0];
    var tipC = this.snake[lastIdx][1];
    var ontoVacatingTail = !eating && tipR === nr && tipC === nc;
    if (this.isBody(nr, nc) && !ontoVacatingTail) {
      this.done = true;
      this.success = false;
      return {
        obs: this.observe(),
        reward: REWARD_DEATH,
        done: true,
        info: { success: false, apples: this.apples, death: "body" }
      };
    }

    /* Sync occ to pre-move body, then apply 1-2 cell delta after snake update. */
    ensureEnvOccupancy(this);

    var newSnake = [[nr, nc]];
    var j;
    var keepLen = eating ? this.snake.length : this.snake.length - 1;
    for (j = 0; j < keepLen; j += 1) {
      newSnake.push(this.snake[j].slice());
    }
    this.snake = newSnake;

    if (!eating) {
      this._occ[tipR * this.size + tipC] = 0;
    }
    this._occ[nr * this.size + nc] = 1;
    if (!this._leave || this._leave.length !== this.size * this.size) {
      this._leave = new Int16Array(this.size * this.size);
      leaveFillFromSnake(this._leave, this.size, this.snake);
    } else {
      leaveSyncAfterStep(this._leave, this.size, this.snake,
        eating ? null : [tipR, tipC]);
    }
    markEnvOccSynced(this);
    openIncApplyStep(this, nr, nc, eating ? null : [tipR, tipC], this.food);

    var reward = 0;
    /* Shape vs the food chased this step (before respawn). Death paths skip this. */
    if (this.food) {
      var d0 = this.manhattanToFood(h[0], h[1]);
      var d1 = this.manhattanToFood(nr, nc);
      reward += (d0 - d1) * REWARD_APPROACH;
    }
    if (eating) {
      this.apples += 1;
      reward += REWARD_FOOD;
      if (this.snake.length >= this.targetLength) {
        this.food = null;
        this.done = true;
        this.success = true;
        invalidateEnvOpenings(this);
        return {
          obs: this.observe(),
          reward: reward,
          done: true,
          info: { success: true, apples: this.apples }
        };
      }
      this.nextFoodFromStream();
      openIncRefreshFood(this, this.food);
      if (this.done) {
        return {
          obs: this.observe(),
          reward: reward,
          done: true,
          info: { success: this.success, apples: this.apples }
        };
      }
    }

    if (!this.disableStepTimeout &&
        isFinite(this.maxSteps) &&
        this.steps >= this.maxSteps) {
      this.done = true;
      this.success = false;
      return {
        obs: this.observe(),
        reward: reward,
        done: true,
        info: { success: false, apples: this.apples, death: "timeout" }
      };
    }

    return {
      obs: this.observe(),
      reward: reward,
      done: false,
      info: { success: false, apples: this.apples }
    };
  };

  function legalFromObs(obs) {
    return [0, 1, 2, 3];
  }

  function obsToStateKey(obs, prevDir) {
    var parts = obs ? obs.slice() : [];
    parts.push(prevDir === null || prevDir === undefined ? -1 : (prevDir | 0));
    return parts.join(",");
  }

  function countVisited(env) {
    return env && env.apples ? env.apples : 0;
  }

  /**
   * BFS toward food when known (compass or in vision). Else systematic explore.
   * Fog-of-war: cannot path through unknown cells beyond vision except when compass
   * points toward food - then BFS on full empty+unknown as walkable if compass on
   * (food location known via compass, board geometry known for walls/body in vision
   * and body always known? Contract: food via compass/vision only. Body/walls:
   * agent can know full own body; walls are board edge. For pathfinding use full
   * body + edges; food target only if foodKnown.
   */
  function bfsNextAction(env) {
    if (!env || !env.snake || !env.snake.length) {
      return null;
    }
    var safe = env.safeActions();
    if (!safe.length) {
      return 0;
    }
    if (env.foodKnown && env.foodKnown()) {
      var pathAct = bfsToward(env, env.food[0], env.food[1], safe);
      if (pathAct !== null) {
        return pathAct;
      }
    }
    /* Systematic: prefer continue, else wall-side, else first safe. */
    var facing = env.facing;
    if (safe.indexOf(facing) >= 0) {
      return facing;
    }
    var i;
    for (i = 0; i < safe.length; i += 1) {
      return safe[i];
    }
    return 0;
  }

  function bfsToward(env, tr, tc, safe) {
    var n = env.size;
    var body = env.bodySet();
    var head = env.snake[0];
    var hr = head[0];
    var hc = head[1];
    var key = function (r, c) {
      return r + "," + c;
    };
    var q = [];
    var parent = {};
    var firstDir = {};
    var seen = {};
    seen[key(hr, hc)] = true;
    var a;
    for (a = 0; a < 4; a += 1) {
      if (safe.indexOf(a) < 0) {
        continue;
      }
      var nr = hr + DIRS[a][0];
      var nc = hc + DIRS[a][1];
      if (nr < 0 || nc < 0 || nr >= n || nc >= n) {
        continue;
      }
      var k = key(nr, nc);
      if (body[k] && !(env.snake[env.snake.length - 1][0] === nr &&
          env.snake[env.snake.length - 1][1] === nc)) {
        continue;
      }
      seen[k] = true;
      parent[k] = key(hr, hc);
      firstDir[k] = a;
      q.push([nr, nc]);
    }
    var qi = 0;
    while (qi < q.length) {
      var cur = q[qi];
      qi += 1;
      if (cur[0] === tr && cur[1] === tc) {
        return firstDir[key(cur[0], cur[1])];
      }
      for (a = 0; a < 4; a += 1) {
        nr = cur[0] + DIRS[a][0];
        nc = cur[1] + DIRS[a][1];
        if (nr < 0 || nc < 0 || nr >= n || nc >= n) {
          continue;
        }
        k = key(nr, nc);
        if (seen[k]) {
          continue;
        }
        if (body[k] && !(nr === tr && nc === tc)) {
          continue;
        }
        seen[k] = true;
        parent[k] = key(cur[0], cur[1]);
        firstDir[k] = firstDir[key(cur[0], cur[1])];
        q.push([nr, nc]);
      }
    }
    return null;
  }

  /**
   * Greedy mid: shorten Manhattan if safe; else same dir if safe; else random safe.
   */
  function greedyNextAction(env) {
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var h = env.snake[0];
    if (env.foodKnown && env.foodKnown()) {
      var best = null;
      var bestD = Infinity;
      var i;
      for (i = 0; i < safe.length; i += 1) {
        var a = safe[i];
        var nr = h[0] + DIRS[a][0];
        var nc = h[1] + DIRS[a][1];
        var d = env.manhattanToFood(nr, nc);
        if (d < bestD) {
          bestD = d;
          best = a;
        }
      }
      if (best !== null) {
        var curD = env.manhattanToFood(h[0], h[1]);
        if (bestD < curD) {
          return best;
        }
      }
    }
    if (safe.indexOf(env.facing) >= 0) {
      return env.facing;
    }
    return safe[Math.floor(Math.random() * safe.length)];
  }

  /**
   * Left-hand wall follower; body counts as wall. Break toward food if known.
   */
  function wallFollowerNextAction(env) {
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    if (env.foodKnown && env.foodKnown()) {
      var g = greedyNextAction(env);
      if (safe.indexOf(g) >= 0) {
        var h = env.snake[0];
        var nr = h[0] + DIRS[g][0];
        var nc = h[1] + DIRS[g][1];
        if (env.manhattanToFood(nr, nc) < env.manhattanToFood(h[0], h[1])) {
          return g;
        }
      }
    }
    var facing = (env.facing === 0 || env.facing === 1 || env.facing === 2 || env.facing === 3) ?
      env.facing : 0;
    var left = (facing + 3) % 4;
    var straight = facing;
    var right = (facing + 1) % 4;
    var back = (facing + 2) % 4;
    var order = [left, straight, right, back];
    var i;
    for (i = 0; i < order.length; i += 1) {
      if (safe.indexOf(order[i]) >= 0) {
        return order[i];
      }
    }
    return safe[0];
  }


  /** Cache: size -> { index, succ, len, path }. */
  var hamiltonCache = {};

  function cellsAdjacent(a, b) {
    return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) === 1;
  }

  function actionFromTo(r0, c0, r1, c1) {
    var dr = r1 - r0;
    var dc = c1 - c0;
    var a;
    for (a = 0; a < 4; a += 1) {
      if (DIRS[a][0] === dr && DIRS[a][1] === dc) {
        return a;
      }
    }
    return null;
  }

  /**
   * Even-width rectangle Hamilton cycle (column zigzag + row-0 return).
   * Covers all height×width cells. Requires width even and width>=2, height>=2.
   */
  function buildEvenWidthCycle(height, width) {
    var path = [];
    var r;
    var c;
    for (r = 0; r < height; r += 1) {
      path.push([r, 0]);
    }
    for (c = 1; c < width; c += 1) {
      if (c % 2 === 1) {
        for (r = height - 1; r >= 1; r -= 1) {
          path.push([r, c]);
        }
      } else {
        for (r = 1; r < height; r += 1) {
          path.push([r, c]);
        }
      }
    }
    for (c = width - 1; c >= 1; c -= 1) {
      path.push([0, c]);
    }
    return path;
  }

  /** Even n×n full Hamilton cycle (column zigzag). */
  function buildEvenHamiltonPath(n) {
    return buildEvenWidthCycle(n, n);
  }

  /** Row-major zigzag (transpose of column zigzag). */
  function buildRowZigzagPath(n) {
    return buildEvenWidthCycle(n, n).map(function (p) {
      return [p[1], p[0]];
    });
  }

  /**
   * 2x2 block cycle for even n: each 2x2 is a 4-cycle; merge along a spanning
   * tree of the meta-grid (works when m is odd too - no meta Hamilton needed).
   */
  function buildBlock2Path(n) {
    if (n < 2 || n % 2 !== 0) {
      return null;
    }
    var m = n / 2;
    function cellCycle(br, bc) {
      var r0 = br * 2;
      var c0 = bc * 2;
      return [[r0, c0], [r0, c0 + 1], [r0 + 1, c0 + 1], [r0 + 1, c0]];
    }
    var cur = cellCycle(0, 0);
    var used = { "0,0": true };
    var usedCount = 1;
    var guard = 0;
    while (usedCount < m * m && guard < m * m * m) {
      guard += 1;
      var found = null;
      var br;
      var bc;
      var d;
      for (br = 0; br < m && !found; br += 1) {
        for (bc = 0; bc < m && !found; bc += 1) {
          if (!used[br + "," + bc]) {
            continue;
          }
          for (d = 0; d < 4; d += 1) {
            var nr = br + DIRS[d][0];
            var nc = bc + DIRS[d][1];
            if (nr < 0 || nc < 0 || nr >= m || nc >= m) {
              continue;
            }
            if (used[nr + "," + nc]) {
              continue;
            }
            found = [nr, nc];
            break;
          }
        }
      }
      if (!found) {
        return null;
      }
      var nxt = cellCycle(found[0], found[1]);
      var merged = mergeTwoCycles(cur, nxt);
      if (!merged) {
        merged = mergeTwoCycles(nxt, cur);
      }
      if (!merged) {
        return null;
      }
      cur = merged;
      used[found[0] + "," + found[1]] = true;
      usedCount += 1;
    }
    if (!validateHamiltonPath(cur, n, false)) {
      return null;
    }
    return cur;
  }

  /**
   * Seeded Warnsdorff Hamilton cycle on full even board. Many short turns tend
   * to create d2/d3 grid chords (unlike long column zigzags).
   */
  function buildWarnsdorffEvenPath(n, seed) {
    if (n % 2 !== 0 || n < 2) {
      return null;
    }
    var total = n * n;
    var key = function (r, c) {
      return r + "," + c;
    };
    var rng = makeRng(seed || 1);
    var used = {};
    var path = [];
    var r = (seed * 3) % n;
    var c = (seed * 7) % n;
    var depth;
    for (depth = 1; depth <= total; depth += 1) {
      path.push([r, c]);
      used[key(r, c)] = true;
      if (depth === total) {
        break;
      }
      var nbrs = [];
      var a;
      for (a = 0; a < 4; a += 1) {
        var nr = r + DIRS[a][0];
        var nc = c + DIRS[a][1];
        var nk = key(nr, nc);
        if (nr < 0 || nc < 0 || nr >= n || nc >= n || used[nk]) {
          continue;
        }
        var deg = 0;
        var b;
        for (b = 0; b < 4; b += 1) {
          var tr = nr + DIRS[b][0];
          var tc = nc + DIRS[b][1];
          var tk = key(tr, tc);
          if (tr < 0 || tc < 0 || tr >= n || tc >= n || used[tk]) {
            continue;
          }
          deg += 1;
        }
        nbrs.push([nr, nc, deg, rng()]);
      }
      if (!nbrs.length) {
        return null;
      }
      nbrs.sort(function (x, y) {
        if (x[2] !== y[2]) {
          return x[2] - y[2];
        }
        return x[3] - y[3];
      });
      r = nbrs[0][0];
      c = nbrs[0][1];
    }
    if (path.length === total && cellsAdjacent(path[total - 1], path[0])) {
      return path;
    }
    return null;
  }

  function collectWarnsdorffEvenCandidates(n, trials) {
    var out = [];
    var seen = {};
    var t;
    var maxTrials = trials || (n <= 12 ? 800 : 200);
    for (t = 1; t <= maxTrials; t += 1) {
      var path = buildWarnsdorffEvenPath(n, t * 7919 + n * 17);
      if (!path || !validateHamiltonPath(path, n, false)) {
        continue;
      }
      var k = path.map(function (p) {
        return p[0] + ":" + p[1];
      }).join("|");
      if (seen[k]) {
        continue;
      }
      seen[k] = true;
      out.push({ name: "warns_" + t, path: path });
      if (out.length >= 12) {
        break;
      }
    }
    return out;
  }

  function reverseCyclePath(path) {
    return path.slice().reverse();
  }

  function rotateCyclePath(path, offset) {
    var k = ((offset % path.length) + path.length) % path.length;
    if (k === 0) {
      return path.slice();
    }
    return path.slice(k).concat(path.slice(0, k));
  }

  /**
   * Locality score: prefer cycles where many grid-adjacent cells are close on
   * the cycle (cheap shortcuts => less skip debt per jump).
   */
  function scoreCycleLocality(path, n) {
    var cy = pathToCycleData(path);
    var d2 = 0;
    var d3 = 0;
    var d4 = 0;
    var sumFar = 0;
    var farCount = 0;
    var r;
    var c;
    var a;
    var nr;
    var nc;
    var d;
    var dMin;
    for (r = 0; r < n; r += 1) {
      for (c = 0; c < n; c += 1) {
        if (cy.index[r + "," + c] === undefined) {
          continue;
        }
        for (a = 0; a < 4; a += 1) {
          nr = r + DIRS[a][0];
          nc = c + DIRS[a][1];
          if (nr < 0 || nc < 0 || nr >= n || nc >= n) {
            continue;
          }
          if (cy.index[nr + "," + nc] === undefined) {
            continue;
          }
          /* Count each undirected edge once. */
          if (nr < r || (nr === r && nc < c)) {
            continue;
          }
          d = cycleDist(cy, r, c, nr, nc);
          dMin = Math.min(d, cy.len - d);
          if (dMin <= 1) {
            continue;
          }
          sumFar += dMin;
          farCount += 1;
          if (dMin === 2) {
            d2 += 1;
          } else if (dMin === 3) {
            d3 += 1;
          } else if (dMin === 4) {
            d4 += 1;
          }
        }
      }
    }
    /* Higher is better: reward short chords, penalize long average. */
    var avgFar = farCount ? sumFar / farCount : 99;
    var score = d2 * 4 + d3 * 2 + d4 - avgFar;
    return { score: score, d2: d2, d3: d3, d4: d4, avgFar: avgFar };
  }

  /**
   * Improve locality via cycle segment reversals (2-opt style): when path[i]
   * is grid-adjacent to path[j] and path[i+1] to path[j+1], reversing the
   * middle keeps a Hamilton cycle and can create short chords.
   */
  function improveCycleLocality(path, n, rounds) {
    var cur = path.map(function (p) {
      return p.slice();
    });
    var bestScore = scoreCycleLocality(cur, n).score;
    var round;
    var improved;
    var len = cur.length;
    rounds = rounds || 40;
    for (round = 0; round < rounds; round += 1) {
      improved = false;
      var i;
      var j;
      for (i = 0; i < len; i += 1) {
        for (j = i + 2; j < len; j += 1) {
          if (i === 0 && j === len - 1) {
            continue;
          }
          var a = cur[i];
          var b = cur[(i + 1) % len];
          var c = cur[j];
          var d = cur[(j + 1) % len];
          if (!cellsAdjacent(a, c) || !cellsAdjacent(b, d)) {
            continue;
          }
          var next = cur.slice(0, i + 1);
          var mid = cur.slice(i + 1, j + 1).reverse();
          next = next.concat(mid);
          next = next.concat(cur.slice(j + 1));
          if (!validateHamiltonPath(next, n, false)) {
            continue;
          }
          var sc = scoreCycleLocality(next, n).score;
          if (sc > bestScore) {
            cur = next;
            bestScore = sc;
            improved = true;
            len = cur.length;
            break;
          }
        }
        if (improved) {
          break;
        }
      }
      if (!improved) {
        break;
      }
    }
    return cur;
  }

  /** Rectangle Hamilton cycle when at least one side is even. */
  function buildRectHamiltonCycle(h, w) {
    if (h < 2 || w < 2) {
      return null;
    }
    if (w % 2 === 0) {
      return buildEvenWidthCycle(h, w);
    }
    if (h % 2 === 0) {
      return buildEvenWidthCycle(w, h).map(function (p) {
        return [p[1], p[0]];
      });
    }
    return null;
  }

  /** Column-oriented zigzag on h×w (prefers vertical strips). */
  function buildRectColZigzag(h, w) {
    return buildRectHamiltonCycle(h, w);
  }

  /** Row-oriented zigzag on h×w (prefers horizontal strips). */
  function buildRectRowZigzag(h, w) {
    if (h < 2 || w < 2) {
      return null;
    }
    if (h % 2 === 0) {
      return buildEvenWidthCycle(w, h).map(function (p) {
        return [p[1], p[0]];
      });
    }
    if (w % 2 === 0) {
      return buildEvenWidthCycle(h, w);
    }
    return null;
  }

  function offsetPath(path, r0, c0) {
    return path.map(function (p) {
      return [p[0] + r0, p[1] + c0];
    });
  }

  /**
   * Merge two vertex-disjoint Hamilton cycles that share a parallel adjacent
   * edge pair (a adj b and succA adj one neighbor of b on B). Returns one cycle.
   */
  function mergeTwoCycles(pathA, pathB) {
    if (!pathA || !pathB || !pathA.length || !pathB.length) {
      return null;
    }
    var idxB = {};
    var j0;
    for (j0 = 0; j0 < pathB.length; j0 += 1) {
      idxB[pathB[j0][0] + "," + pathB[j0][1]] = j0;
    }
    var i;
    var d;
    for (i = 0; i < pathA.length; i += 1) {
      var a = pathA[i];
      var sa = pathA[(i + 1) % pathA.length];
      for (d = 0; d < 4; d += 1) {
        var br = a[0] + DIRS[d][0];
        var bc = a[1] + DIRS[d][1];
        var j = idxB[br + "," + bc];
        if (j === undefined) {
          continue;
        }
        var sbFwd = pathB[(j + 1) % pathB.length];
        var sbBack = pathB[(j - 1 + pathB.length) % pathB.length];
        var targets = [];
        if (cellsAdjacent(sa, sbFwd)) {
          targets.push({ end: sbFwd, dir: -1 });
        }
        if (cellsAdjacent(sa, sbBack)) {
          targets.push({ end: sbBack, dir: 1 });
        }
        var ti;
        for (ti = 0; ti < targets.length; ti += 1) {
          var out = pathA.slice(0, i + 1);
          var k = j;
          var guard = 0;
          for (;;) {
            out.push(pathB[k]);
            if (pathB[k][0] === targets[ti].end[0] &&
                pathB[k][1] === targets[ti].end[1]) {
              break;
            }
            k = (k + targets[ti].dir + pathB.length) % pathB.length;
            guard += 1;
            if (guard > pathB.length + 2) {
              out = null;
              break;
            }
          }
          if (!out) {
            continue;
          }
          out = out.concat(pathA.slice(i + 1));
          if (out.length === pathA.length + pathB.length) {
            return out;
          }
        }
      }
    }
    return null;
  }

  /**
   * half-n-half: left = column zigzag, right = row zigzag; merge at the seam.
   * Also try top/bottom split (row zig on top, col zig on bottom).
   */
  function buildHalfNHalfPath(n, style) {
    if (n < 4 || n % 2 !== 0) {
      return null;
    }
    var mid = n / 2;
    var a;
    var b;
    var merged;
    style = style || "lr";
    if (style === "lr" || style === "lr_swap") {
      a = style === "lr" ? buildRectColZigzag(n, mid) : buildRectRowZigzag(n, mid);
      b = style === "lr" ? buildRectRowZigzag(n, n - mid) : buildRectColZigzag(n, n - mid);
      if (!a || !b) {
        return null;
      }
      b = offsetPath(b, 0, mid);
      merged = mergeTwoCycles(a, b);
      if (merged && validateHamiltonPath(merged, n, false)) {
        return merged;
      }
      return null;
    }
    if (style === "tb" || style === "tb_swap") {
      a = style === "tb" ? buildRectRowZigzag(mid, n) : buildRectColZigzag(mid, n);
      b = style === "tb" ? buildRectColZigzag(n - mid, n) : buildRectRowZigzag(n - mid, n);
      if (!a || !b) {
        return null;
      }
      b = offsetPath(b, mid, 0);
      merged = mergeTwoCycles(a, b);
      if (merged && validateHamiltonPath(merged, n, false)) {
        return merged;
      }
      return null;
    }
    return null;
  }

  /**
   * quarters: four blocks with alternating col/row zigzag, pairwise merge.
   * When n%4==0, equal n/2 quadrants. When n%4==2, even-sized uneven split
   * (a×a / a×b / b×a / b×b with a=n/2-1, b=n/2+1) so each block has a cycle.
   */
  function buildQuartersPath(n, style) {
    if (n < 4 || n % 2 !== 0) {
      return null;
    }
    var ah;
    var aw;
    var bh;
    var bw;
    if (n % 4 === 0) {
      ah = aw = bh = bw = n / 2;
    } else {
      aw = n / 2 - 1;
      bw = n / 2 + 1;
      ah = aw;
      bh = bw;
      if (aw < 2 || bw < 2) {
        return null;
      }
    }
    var tl;
    var tr;
    var bl;
    var br;
    var top;
    var bot;
    var all;
    style = style || "alt";
    if (style === "alt") {
      tl = buildRectColZigzag(ah, aw);
      tr = buildRectRowZigzag(ah, bw);
      bl = buildRectRowZigzag(bh, aw);
      br = buildRectColZigzag(bh, bw);
    } else {
      tl = buildRectRowZigzag(ah, aw);
      tr = buildRectColZigzag(ah, bw);
      bl = buildRectColZigzag(bh, aw);
      br = buildRectRowZigzag(bh, bw);
    }
    if (!tl || !tr || !bl || !br) {
      return null;
    }
    tr = offsetPath(tr, 0, aw);
    bl = offsetPath(bl, ah, 0);
    br = offsetPath(br, ah, aw);
    top = mergeTwoCycles(tl, tr);
    if (!top) {
      top = mergeTwoCycles(tr, tl);
    }
    bot = mergeTwoCycles(bl, br);
    if (!bot) {
      bot = mergeTwoCycles(br, bl);
    }
    if (!top || !bot) {
      return null;
    }
    all = mergeTwoCycles(top, bot);
    if (!all) {
      all = mergeTwoCycles(bot, top);
    }
    if (all && validateHamiltonPath(all, n, false)) {
      return all;
    }
    return null;
  }

  /** Alias: uneven quarters when equal quadrants are impossible. */
  function buildQuartersLoosePath(n) {
    if (n < 6 || n % 2 !== 0) {
      return null;
    }
    if (n % 4 === 0) {
      return buildQuartersPath(n, "alt");
    }
    return buildQuartersPath(n, "alt");
  }

  /** Concentric ring cycle (spiral family): merge perimeter rings inward. */
  function buildSpiralRingPath(n) {
    if (n < 2 || n % 2 !== 0) {
      return null;
    }
    function ringCycle(o, s) {
      var path = [];
      var r0 = o;
      var c0 = o;
      var r1 = o + s - 1;
      var c1 = o + s - 1;
      var r;
      var c;
      if (s === 2) {
        return [[r0, c0], [r0, c1], [r1, c1], [r1, c0]];
      }
      for (c = c0; c <= c1; c += 1) {
        path.push([r0, c]);
      }
      for (r = r0 + 1; r <= r1; r += 1) {
        path.push([r, c1]);
      }
      for (c = c1 - 1; c >= c0; c -= 1) {
        path.push([r1, c]);
      }
      for (r = r1 - 1; r > r0; r -= 1) {
        path.push([r, c0]);
      }
      return path;
    }
    var cur = ringCycle((n / 2) - 1, 2);
    var s;
    for (s = 4; s <= n; s += 2) {
      var o = (n - s) / 2;
      var outer = ringCycle(o, s);
      var merged = mergeTwoCycles(outer, cur);
      if (!merged) {
        merged = mergeTwoCycles(cur, outer);
      }
      if (!merged) {
        return null;
      }
      cur = merged;
    }
    if (!validateHamiltonPath(cur, n, false)) {
      return null;
    }
    return cur;
  }

  /**
   * Coil spiral path closed into a cycle when possible: serpentine inward
   * layers with U-turns (path), then try reverse as orientation.
   * For even n we prefer ring merge; this is a secondary spiral shape built
   * by layer U-turns that form a cycle via a seam at the start corner.
   */
  function buildCoilSpiralPath(n) {
    if (n < 4 || n % 2 !== 0) {
      return null;
    }
    /* Use ring spiral first; alias kept for naming in benches. */
    return buildSpiralRingPath(n);
  }

  /** Hilbert curve as a path; close to a cycle only if ends are adjacent. */
  function buildHilbertPath(n) {
    if (n < 2 || (n & (n - 1)) !== 0) {
      return null;
    }
    /* n power of 2. */
    function rot(nSz, x, y, rx, ry) {
      if (ry === 0) {
        if (rx === 1) {
          x = nSz - 1 - x;
          y = nSz - 1 - y;
        }
        var t = x;
        x = y;
        y = t;
      }
      return [x, y];
    }
    function d2xy(nSz, d) {
      var x = 0;
      var y = 0;
      var s;
      var t = d;
      for (s = 1; s < nSz; s *= 2) {
        var rx = 1 & (t / 2);
        var ry = 1 & (t ^ rx);
        var xy = rot(s, x, y, rx, ry);
        x = xy[0];
        y = xy[1];
        x += s * rx;
        y += s * ry;
        t = (t / 4) | 0;
      }
      return [y, x];
    }
    var path = [];
    var i;
    var total = n * n;
    for (i = 0; i < total; i += 1) {
      path.push(d2xy(n, i));
    }
    if (!cellsAdjacent(path[0], path[total - 1])) {
      /* Hilbert path is not a cycle; stitch with a 2-opt style end fix via
         reverse of second half when n>=4 - usually still not adjacent.
         Fall back: return null (bench skips). */
      return null;
    }
    if (!validateHamiltonPath(path, n, false)) {
      return null;
    }
    return path;
  }

  /**
   * Hilbert-order meta cells (when m power of 2) merged as 2x2 blocks.
   * Falls back to block2 meta zigzag when Hilbert meta is not a cycle.
   */
  function buildHilbertBlockPath(n) {
    if (n < 4 || n % 2 !== 0) {
      return null;
    }
    var m = n / 2;
    var meta = null;
    if ((m & (m - 1)) === 0) {
      meta = [];
      var i;
      var total = m * m;
      function rot(nSz, x, y, rx, ry) {
        if (ry === 0) {
          if (rx === 1) {
            x = nSz - 1 - x;
            y = nSz - 1 - y;
          }
          var t = x;
          x = y;
          y = t;
        }
        return [x, y];
      }
      function d2xy(nSz, d) {
        var x = 0;
        var y = 0;
        var s;
        var t = d;
        for (s = 1; s < nSz; s *= 2) {
          var rx = 1 & (t / 2);
          var ry = 1 & (t ^ rx);
          var xy = rot(s, x, y, rx, ry);
          x = xy[0];
          y = xy[1];
          x += s * rx;
          y += s * ry;
          t = (t / 4) | 0;
        }
        return [y, x];
      }
      for (i = 0; i < total; i += 1) {
        meta.push(d2xy(m, i));
      }
      if (!cellsAdjacent(meta[0], meta[total - 1]) ||
          !validateHamiltonPath(meta, m, false)) {
        meta = null;
      }
    }
    if (!meta) {
      return buildBlock2Path(n);
    }
    function cellCycle(br, bc) {
      var r0 = br * 2;
      var c0 = bc * 2;
      return [[r0, c0], [r0, c0 + 1], [r0 + 1, c0 + 1], [r0 + 1, c0]];
    }
    var cur = cellCycle(meta[0][0], meta[0][1]);
    var mi;
    for (mi = 1; mi < meta.length; mi += 1) {
      var nxt = cellCycle(meta[mi][0], meta[mi][1]);
      var merged = mergeTwoCycles(cur, nxt);
      if (!merged) {
        merged = mergeTwoCycles(nxt, cur);
      }
      if (!merged) {
        return buildBlock2Path(n);
      }
      cur = merged;
    }
    if (!validateHamiltonPath(cur, n, false)) {
      return buildBlock2Path(n);
    }
    return cur;
  }

  /**
   * Static Cell-Tree-like tour: merge 2x2 RHD cycles in row-major spanning
   * order (distinct from block2's column-neighbor preference).
   */
  function buildCellTreeStaticPath(n) {
    if (n < 2 || n % 2 !== 0) {
      return null;
    }
    var m = n / 2;
    function cellCycle(br, bc) {
      var r0 = br * 2;
      var c0 = bc * 2;
      /* Opposite tile orientation vs block2. */
      return [[r0, c0], [r0 + 1, c0], [r0 + 1, c0 + 1], [r0, c0 + 1]];
    }
    var cur = cellCycle(0, 0);
    var used = { "0,0": true };
    var usedCount = 1;
    var guard = 0;
    while (usedCount < m * m && guard < m * m * m) {
      guard += 1;
      var found = null;
      var br;
      var bc;
      /* Prefer east then south (row-major growth). */
      var pref = [[0, 1], [1, 0], [0, -1], [-1, 0]];
      for (br = 0; br < m && !found; br += 1) {
        for (bc = 0; bc < m && !found; bc += 1) {
          if (!used[br + "," + bc]) {
            continue;
          }
          var pi;
          for (pi = 0; pi < 4; pi += 1) {
            var nr = br + pref[pi][0];
            var nc = bc + pref[pi][1];
            if (nr < 0 || nc < 0 || nr >= m || nc >= m) {
              continue;
            }
            if (used[nr + "," + nc]) {
              continue;
            }
            found = [nr, nc];
            break;
          }
        }
      }
      if (!found) {
        return null;
      }
      var nxt = cellCycle(found[0], found[1]);
      var merged = mergeTwoCycles(cur, nxt);
      if (!merged) {
        merged = mergeTwoCycles(nxt, cur);
      }
      if (!merged) {
        return null;
      }
      cur = merged;
      used[found[0] + "," + found[1]] = true;
      usedCount += 1;
    }
    if (!validateHamiltonPath(cur, n, false)) {
      return null;
    }
    return cur;
  }

  /** Merge vertical bands of width `bandW` (even) across the board. */
  function buildStripBandPath(n, bandW) {
    if (n < 2 || n % 2 !== 0 || bandW < 2 || bandW % 2 !== 0 || n % bandW !== 0) {
      return null;
    }
    var bands = n / bandW;
    var cur = null;
    var b;
    for (b = 0; b < bands; b += 1) {
      var band = buildRectHamiltonCycle(n, bandW);
      if (!band) {
        return null;
      }
      if (b % 2 === 1) {
        band = reverseCyclePath(band);
      }
      band = offsetPath(band, 0, b * bandW);
      if (!cur) {
        cur = band;
      } else {
        var merged = mergeTwoCycles(cur, band);
        if (!merged) {
          merged = mergeTwoCycles(band, cur);
        }
        if (!merged) {
          return null;
        }
        cur = merged;
      }
    }
    if (!validateHamiltonPath(cur, n, false)) {
      return null;
    }
    return cur;
  }

  /** 4-column ladder: merge n×4 band cycles. */
  function buildStrip4Path(n) {
    return buildStripBandPath(n, 4);
  }

  /** 2-column ladder (often matches col zigzag family). */
  function buildStrip2Path(n) {
    return buildStripBandPath(n, 2);
  }

  /**
   * Score: lower mean cycle distance from every cell to the nearest
   * near-middle cell (central 2x2). Higher return value is better.
   */
  function scoreNearMiddle(path, n) {
    var cy = pathToCycleData(path);
    var centers = [];
    var r0 = Math.floor((n - 1) / 2);
    var c0 = Math.floor((n - 1) / 2);
    var r;
    var c;
    for (r = r0; r <= r0 + (n % 2 === 0 ? 1 : 0) && r < n; r += 1) {
      for (c = c0; c <= c0 + (n % 2 === 0 ? 1 : 0) && c < n; c += 1) {
        if (cy.index[r + "," + c] !== undefined) {
          centers.push([r, c]);
        }
      }
    }
    if (!centers.length) {
      return -1e9;
    }
    var sum = 0;
    var count = 0;
    var i;
    for (i = 0; i < path.length; i += 1) {
      var p = path[i];
      var best = cy.len;
      var ci;
      for (ci = 0; ci < centers.length; ci += 1) {
        var d = cycleDist(cy, p[0], p[1], centers[ci][0], centers[ci][1]);
        var dMin = Math.min(d, cy.len - d);
        if (dMin < best) {
          best = dMin;
        }
      }
      sum += best;
      count += 1;
    }
    return count ? -(sum / count) : -1e9;
  }

  /**
   * Pick among constructive candidates the cycle that maximizes near-middle
   * access (min mean cycle distance to the central 2x2).
   */
  function buildNearMiddlePath(n, omitCorner) {
    var cands = [];
    var omit = !!omitCorner;
    function push(name, path) {
      if (path && validateHamiltonPath(path, n, omit)) {
        cands.push({ name: name, path: path, mid: scoreNearMiddle(path, n) });
      }
    }
    if (!omit) {
      push("col", buildEvenHamiltonPath(n));
      push("col_rev", reverseCyclePath(buildEvenHamiltonPath(n)));
      push("row", buildRowZigzagPath(n));
      push("row_rev", reverseCyclePath(buildRowZigzagPath(n)));
      push("half_lr", buildHalfNHalfPath(n, "lr"));
      push("half_tb", buildHalfNHalfPath(n, "tb"));
      push("spiral", buildSpiralRingPath(n));
      push("block2", buildBlock2Path(n));
      push("ct_static", buildCellTreeStaticPath(n));
      push("hilbert_block", buildHilbertBlockPath(n));
      var warns = collectWarnsdorffEvenCandidates(n, n <= 10 ? 400 : 100);
      var wi;
      for (wi = 0; wi < Math.min(warns.length, 6); wi += 1) {
        push(warns[wi].name, warns[wi].path);
      }
    } else {
      /* Agent omit protocol is BR-only; only BR-omit shapes are usable. */
      var oi;
      for (oi = 0; oi < 8; oi += 1) {
        push("oddWarn_" + oi, buildOddHamiltonPathSeeded(n, oi * 104729 + 13));
      }
    }
    if (!cands.length) {
      return null;
    }
    cands.sort(function (a, b) {
      return b.mid - a.mid;
    });
    return cands[0].path;
  }

  /**
   * Odd n: Warnsdorff Hamilton cycle omitting BR, searching from seedBase.
   * (Other omit corners need agent protocol changes; not used in bakeoffs.)
   */
  function buildOddHamiltonPathSeeded(n, seedBase) {
    var total = n * n - 1;
    var key = function (r, c) {
      return r + "," + c;
    };
    var blocked = key(n - 1, n - 1);
    var attempt;
    var maxAttempts = n <= 15 ? 2000 : 400;

    function tryGreedy(seed) {
      var rng = makeRng(seed);
      var used = {};
      var path = [];
      var r = Math.abs(seed * 3) % n;
      var c = Math.abs(seed * 7) % n;
      if (r === n - 1 && c === n - 1) {
        r = 0;
        c = 0;
      }
      var depth;
      for (depth = 1; depth <= total; depth += 1) {
        path.push([r, c]);
        used[key(r, c)] = true;
        if (depth === total) {
          break;
        }
        var nbrs = [];
        var a;
        for (a = 0; a < 4; a += 1) {
          var nr = r + DIRS[a][0];
          var nc = c + DIRS[a][1];
          var nk = key(nr, nc);
          if (nr < 0 || nc < 0 || nr >= n || nc >= n || used[nk] || nk === blocked) {
            continue;
          }
          var deg = 0;
          var b;
          for (b = 0; b < 4; b += 1) {
            var tr = nr + DIRS[b][0];
            var tc = nc + DIRS[b][1];
            var tk = key(tr, tc);
            if (tr < 0 || tc < 0 || tr >= n || tc >= n || used[tk] || tk === blocked) {
              continue;
            }
            deg += 1;
          }
          nbrs.push([nr, nc, deg, rng()]);
        }
        if (!nbrs.length) {
          return null;
        }
        nbrs.sort(function (x, y) {
          if (x[2] !== y[2]) {
            return x[2] - y[2];
          }
          return x[3] - y[3];
        });
        r = nbrs[0][0];
        c = nbrs[0][1];
      }
      if (path.length === total && cellsAdjacent(path[total - 1], path[0])) {
        return path;
      }
      return null;
    }

    for (attempt = 1; attempt <= maxAttempts; attempt += 1) {
      var found = tryGreedy(seedBase + attempt * 9973 + n * 13);
      if (found) {
        return found;
      }
    }
    return null;
  }

  /**
   * Curated Hamilton cycle variants for bakeoffs (even + odd).
   * Invalid / unsupported shapes are omitted.
   */
  function listHamiltonCycleVariants(n) {
    var omit = n % 2 === 1;
    var out = [];
    var seen = {};
    function add(name, path) {
      if (!path || !validateHamiltonPath(path, n, omit)) {
        return;
      }
      var k = path.map(function (p) {
        return p[0] + ":" + p[1];
      }).join("|");
      if (seen[k]) {
        return;
      }
      seen[k] = true;
      out.push({ name: name, path: path });
    }
    function addRev(name, path) {
      add(name, path);
      if (path) {
        add(name + "_rev", reverseCyclePath(path));
      }
    }

    if (!omit) {
      addRev("colZigzag", buildEvenHamiltonPath(n));
      addRev("rowZigzag", buildRowZigzagPath(n));
      addRev("half_lr", buildHalfNHalfPath(n, "lr"));
      addRev("half_lr_swap", buildHalfNHalfPath(n, "lr_swap"));
      addRev("half_tb", buildHalfNHalfPath(n, "tb"));
      addRev("half_tb_swap", buildHalfNHalfPath(n, "tb_swap"));
      addRev("quarters", buildQuartersPath(n, "alt"));
      addRev("quarters_swap", buildQuartersPath(n, "swap"));
      addRev("quarters_loose", buildQuartersLoosePath(n));
      addRev("spiral", buildSpiralRingPath(n));
      addRev("coil", buildCoilSpiralPath(n));
      addRev("block2", buildBlock2Path(n));
      addRev("hilbert", buildHilbertPath(n));
      addRev("hilbert_block", buildHilbertBlockPath(n));
      addRev("ct_static", buildCellTreeStaticPath(n));
      addRev("hilbert_block", buildHilbertBlockPath(n));
      addRev("strip2", buildStrip2Path(n));
      addRev("strip4", buildStrip4Path(n));
      var mid = buildNearMiddlePath(n, false);
      add("near_middle", mid);
      var warns = collectWarnsdorffEvenCandidates(n, n <= 10 ? 300 : 80);
      var wi;
      for (wi = 0; wi < Math.min(warns.length, 3); wi += 1) {
        add("warns_" + wi, warns[wi].path);
      }
      var opt = improveCycleLocality(buildEvenHamiltonPath(n), n, n <= 12 ? 40 : 10);
      addRev("col_opt", opt);
    } else {
      addRev("oddWarn", buildOddHamiltonPath(n));
      var oi;
      for (oi = 0; oi < 6; oi += 1) {
        add("oddWarn_s" + oi, buildOddHamiltonPathSeeded(n, oi * 104729 + 97));
      }
      add("near_middle", buildNearMiddlePath(n, true));
    }
    return out;
  }

  /** Candidate even-n Hamilton paths for locality search. */
  function listEvenHamiltonCandidates(n) {
    var raw = listHamiltonCycleVariants(n).filter(function (c) {
      return n % 2 === 0;
    });
    var bases = raw.slice();
    var bi;
    for (bi = 0; bi < Math.min(bases.length, 8); bi += 1) {
      var b = bases[bi];
      var optRounds = n <= 12 ? 60 : (n <= 24 ? 20 : 8);
      var improved = improveCycleLocality(b.path, n, optRounds);
      if (validateHamiltonPath(improved, n, false)) {
        raw.push({ name: b.name + "_opt", path: improved });
        raw.push({ name: b.name + "_opt_rev", path: reverseCyclePath(improved) });
      }
    }
    var extras = [];
    var ei;
    var baseForRot = raw.slice(0, Math.min(raw.length, 8));
    for (ei = 0; ei < baseForRot.length; ei += 1) {
      var step = Math.max(1, n);
      var off;
      for (off = step; off < n * n; off += step * Math.max(2, Math.floor(n / 2))) {
        extras.push({
          name: baseForRot[ei].name + "_rot" + off,
          path: rotateCyclePath(baseForRot[ei].path, off)
        });
      }
    }
    return raw.concat(extras);
  }

  /**
   * Pick production even-n cycle. Bakeoffs (10x10 Master/skip): Warnsdorff and
   * locality-max cycles stayed durable but used MORE steps; column zigzag
   * reverse was fastest. Keep the zigzag family only for auto-pick.
   */
  function pickBestEvenHamiltonPath(n) {
    var cands = [
      { name: "colZigzag_rev", path: reverseCyclePath(buildEvenHamiltonPath(n)) },
      { name: "colZigzag", path: buildEvenHamiltonPath(n) },
      { name: "rowZigzag_rev", path: reverseCyclePath(buildRowZigzagPath(n)) },
      { name: "rowZigzag", path: buildRowZigzagPath(n) }
    ];
    var i;
    for (i = 0; i < cands.length; i += 1) {
      if (!validateHamiltonPath(cands[i].path, n, false)) {
        continue;
      }
      cands[i].locality = scoreCycleLocality(cands[i].path, n);
      return cands[i];
    }
    return { name: "colZigzag", path: buildEvenHamiltonPath(n), locality: null };
  }

  /**
   * Splice two new cells into a cycle path by replacing edge (r1,c1)->(r2,c2)
   * with (r1,c1)->via1->via2->(r2,c2). Returns new path or null.
   */
  function splicePairIntoCycle(path, r1, c1, r2, c2, via1, via2) {
    var i;
    var n = path.length;
    for (i = 0; i < n; i += 1) {
      var a = path[i];
      var b = path[(i + 1) % n];
      if (a[0] === r1 && a[1] === c1 && b[0] === r2 && b[1] === c2) {
        return path.slice(0, i + 1).concat([via1, via2], path.slice(i + 1));
      }
      if (a[0] === r2 && a[1] === c2 && b[0] === r1 && b[1] === c1) {
        return path.slice(0, i + 1).concat([via2, via1], path.slice(i + 1));
      }
    }
    return null;
  }

  /**
   * Guaranteed odd n×n Hamilton cycle omitting BR (n-1,n-1).
   * Even Hamilton on the top-left (n-1)×(n-1), then pairwise-splice the
   * right column and bottom row (each has even length when n is odd).
   * Replaces the old even-(n-1)-only fallback that left the fringe off-cycle.
   */
  function buildOddOmitCornerConstructive(n) {
    if (n < 3 || n % 2 === 0) {
      return null;
    }
    var m = n - 1;
    var path = buildEvenHamiltonPath(m);
    if (!path || !validateHamiltonPath(path, m, false)) {
      return null;
    }
    var k;
    /* Right column (0..n-2, n-1): splice pairs (0,1), (2,3), ... using
       vertical edges on col n-2 of the inner cycle. */
    for (k = 0; k < m; k += 2) {
      var rA = k;
      var rB = k + 1;
      var next = splicePairIntoCycle(
        path,
        rA, m - 1,
        rB, m - 1,
        [rA, n - 1],
        [rB, n - 1]
      );
      if (!next) {
        return null;
      }
      path = next;
    }
    /* Bottom row (n-1, 0..n-2): splice pairs using horizontal edges on row n-2. */
    for (k = 0; k < m; k += 2) {
      var cA = k;
      var cB = k + 1;
      next = splicePairIntoCycle(
        path,
        m - 1, cA,
        m - 1, cB,
        [n - 1, cA],
        [n - 1, cB]
      );
      if (!next) {
        return null;
      }
      path = next;
    }
    if (!validateHamiltonPath(path, n, true)) {
      return null;
    }
    return path;
  }

  /**
   * Odd n×n Hamilton cycle omitting (n-1,n-1).
   * Prefer geometric splice (always O(n^2) when it works); Warnsdorff only as fallback.
   */
  function buildOddHamiltonPath(n) {
    var constructed = buildOddOmitCornerConstructive(n);
    if (constructed && validateHamiltonPath(constructed, n, true)) {
      return constructed;
    }
    var total = n * n - 1;
    var key = function (r, c) {
      return r + "," + c;
    };
    var blocked = key(n - 1, n - 1);
    var attempt;
    /* Fallback if splice construction ever fails for some n. */
    var maxAttempts = n <= 15 ? 4000 : (n <= 25 ? 25000 : 8000);

    function tryGreedy(seed) {
      var rng = makeRng(seed);
      var used = {};
      var path = [];
      var r = (seed * 3) % n;
      var c = (seed * 7) % n;
      if (r === n - 1 && c === n - 1) {
        r = 0;
        c = 0;
      }
      var depth;
      for (depth = 1; depth <= total; depth += 1) {
        path.push([r, c]);
        used[key(r, c)] = true;
        if (depth === total) {
          break;
        }
        var nbrs = [];
        var a;
        for (a = 0; a < 4; a += 1) {
          var nr = r + DIRS[a][0];
          var nc = c + DIRS[a][1];
          var nk = key(nr, nc);
          if (nr < 0 || nc < 0 || nr >= n || nc >= n || used[nk] || nk === blocked) {
            continue;
          }
          var deg = 0;
          var b;
          for (b = 0; b < 4; b += 1) {
            var tr = nr + DIRS[b][0];
            var tc = nc + DIRS[b][1];
            var tk = key(tr, tc);
            if (tr < 0 || tc < 0 || tr >= n || tc >= n || used[tk] || tk === blocked) {
              continue;
            }
            deg += 1;
          }
          nbrs.push([nr, nc, deg, rng()]);
        }
        if (!nbrs.length) {
          return null;
        }
        nbrs.sort(function (x, y) {
          if (x[2] !== y[2]) {
            return x[2] - y[2];
          }
          return x[3] - y[3];
        });
        r = nbrs[0][0];
        c = nbrs[0][1];
      }
      if (path.length === total && cellsAdjacent(path[total - 1], path[0])) {
        return path;
      }
      return null;
    }

    for (attempt = 1; attempt <= maxAttempts; attempt += 1) {
      var found = tryGreedy(attempt * 9973 + n * 13);
      if (found) {
        return found;
      }
    }
    return null;
  }

  function validateHamiltonPath(path, n, omitCorner) {
    if (!path || !path.length) {
      return false;
    }
    var expect = omitCorner ? n * n - 1 : n * n;
    if (path.length !== expect) {
      return false;
    }
    var seen = {};
    var i;
    for (i = 0; i < path.length; i += 1) {
      var p = path[i];
      if (p[0] < 0 || p[1] < 0 || p[0] >= n || p[1] >= n) {
        return false;
      }
      if (omitCorner && p[0] === n - 1 && p[1] === n - 1) {
        return false;
      }
      var k = p[0] + "," + p[1];
      if (seen[k]) {
        return false;
      }
      seen[k] = true;
      if (!cellsAdjacent(p, path[(i + 1) % path.length])) {
        return false;
      }
    }
    return true;
  }

  function pathToCycleData(path) {
    var index = {};
    var succ = {};
    var len = path.length;
    var i;
    for (i = 0; i < len; i += 1) {
      var a = path[i];
      var b = path[(i + 1) % len];
      var ka = a[0] + "," + a[1];
      index[ka] = i;
      succ[ka] = b;
    }
    return { index: index, succ: succ, len: len, path: path };
  }

  function getHamiltonCycle(n) {
    if (hamiltonCache[n]) {
      return hamiltonCache[n];
    }
    var omit = n % 2 === 1;
    var path;
    var picked = null;
    if (omit) {
      path = buildOddHamiltonPath(n);
      if (!validateHamiltonPath(path, n, true)) {
        throw new Error(
          "odd Hamilton cycle failed for n=" + n +
            " (need omit-corner cycle of len " + (n * n - 1) + ")"
        );
      }
      picked = { name: "oddOmitCorner_splice" };
      omit = true;
    } else {
      picked = pickBestEvenHamiltonPath(n);
      path = picked.path;
      if (!validateHamiltonPath(path, n, false)) {
        path = buildEvenHamiltonPath(n);
        picked = { name: "colZigzag_fallback" };
      }
    }
    var data = pathToCycleData(path);
    data.omitCorner = omit;
    data.builderName = picked && picked.name ? picked.name : (omit ? "oddWarnsdorff" : "colZigzag");
    if (picked && picked.locality) {
      data.locality = picked.locality;
    }
    hamiltonCache[n] = data;
    return data;
  }

  /** Test/bench helper: force a cycle path for size n (clears that cache entry). */
  function setHamiltonCyclePath(n, path, name) {
    var omit = n % 2 === 1;
    if (!validateHamiltonPath(path, n, omit)) {
      return false;
    }
    var data = pathToCycleData(path);
    data.omitCorner = omit;
    data.builderName = name || "custom";
    data.locality = scoreCycleLocality(path, n);
    hamiltonCache[n] = data;
    return true;
  }

  function clearHamiltonCache() {
    var k;
    for (k in hamiltonCache) {
      if (Object.prototype.hasOwnProperty.call(hamiltonCache, k)) {
        delete hamiltonCache[k];
      }
    }
  }

  function cycleDist(cy, r0, c0, r1, c1) {
    var i0 = cy.index[r0 + "," + c0];
    var i1 = cy.index[r1 + "," + c1];
    if (i0 === undefined || i1 === undefined) {
      return cy.len + 1;
    }
    return (i1 - i0 + cy.len) % cy.len;
  }

  /**
   * True if stepping to (nr,nc) is not a one-cell pocket: flood >= 2, reaches
   * tip (vacated when not eating), and leaves at least one free neighbor.
   */
  function pocketOk(env, nr, nc, eating) {
    if (!env || !env.snake || !env.snake.length) {
      return false;
    }
    var n = env.size;
    var body = {};
    var i;
    for (i = 0; i < env.snake.length; i += 1) {
      body[env.snake[i][0] + "," + env.snake[i][1]] = true;
    }
    var tip = env.snake[env.snake.length - 1];
    var tipKey = tip[0] + "," + tip[1];
    if (!eating) {
      delete body[tipKey];
    }
    delete body[nr + "," + nc];
    var q = [[nr, nc]];
    var seen = {};
    seen[nr + "," + nc] = true;
    var count = 0;
    var qi = 0;
    var d;
    var cur;
    var rr;
    var cc;
    var k;
    while (qi < q.length) {
      cur = q[qi];
      qi += 1;
      count += 1;
      for (d = 0; d < 4; d += 1) {
        rr = cur[0] + DIRS[d][0];
        cc = cur[1] + DIRS[d][1];
        if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
          continue;
        }
        k = rr + "," + cc;
        if (seen[k] || body[k]) {
          continue;
        }
        seen[k] = true;
        q.push([rr, cc]);
      }
    }
    if (count < 2 || !seen[tipKey]) {
      return false;
    }
    var keepLen = eating ? env.snake.length : env.snake.length - 1;
    var afterBody = {};
    afterBody[nr + "," + nc] = true;
    for (i = 0; i < keepLen; i += 1) {
      afterBody[env.snake[i][0] + "," + env.snake[i][1]] = true;
    }
    for (d = 0; d < 4; d += 1) {
      rr = nr + DIRS[d][0];
      cc = nc + DIRS[d][1];
      if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
        continue;
      }
      if (!afterBody[rr + "," + cc]) {
        return true;
      }
    }
    return false;
  }

  /** BFS to food when the step is pocket-safe; else greedy; else first safe. */
  function avoidPocketNextAction(env) {
    if (!env || !env.snake || !env.snake.length) {
      return 0;
    }
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var bfs = bfsNextAction(env);
    if (bfs !== null && safe.indexOf(bfs) >= 0) {
      var h = env.snake[0];
      var nr = h[0] + DIRS[bfs][0];
      var nc = h[1] + DIRS[bfs][1];
      var eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
      if (pocketOk(env, nr, nc, eating)) {
        return bfs;
      }
    }
    var g = greedyNextAction(env);
    if (g !== null && safe.indexOf(g) >= 0) {
      return g;
    }
    return safe[0];
  }

  function masterEarlyThreshold(n) {
    var fivePct = Math.ceil(0.05 * n * n);
    return fivePct > n ? fivePct : n;
  }

  /**
   * Early Master driver: Avoid Pocket (BFS if pocket-safe) but reject moves that
   * explode Hamilton skip debt beyond +N so heal can still clear.
   */
  function masterEarlyAction(env) {
    if (!env || !env.snake || !env.snake.length) {
      return 0;
    }
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var n = env.size;
    var cy = getHamiltonCycle(n);
    var skipNow = masterCycleSkip(cy, env.snake);
    var h = env.snake[0];
    var bfs = bfsNextAction(env);
    if (bfs !== null && safe.indexOf(bfs) >= 0) {
      var nr = h[0] + DIRS[bfs][0];
      var nc = h[1] + DIRS[bfs][1];
      var eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
      if (pocketOk(env, nr, nc, eating)) {
        var skipAfter = masterSkipAfterMove(cy, env.snake, nr, nc, eating);
        if (skipAfter <= skipNow + n) {
          return bfs;
        }
      }
    }
    var g = greedyNextAction(env);
    if (g !== null && safe.indexOf(g) >= 0) {
      var gr = h[0] + DIRS[g][0];
      var gc = h[1] + DIRS[g][1];
      var ge = !!(env.food && env.food[0] === gr && env.food[1] === gc);
      var gSkip = masterSkipAfterMove(cy, env.snake, gr, gc, ge);
      if (gSkip <= skipNow + n) {
        return g;
      }
    }
    return masterHamiltonAction(env, { healOnly: true });
  }

  /**
   * Skip debt: holes between consecutive body cells along the cycle.
   * Contiguous reverse-cycle body => every neck is one step behind its parent
   * => skip 0. A forward jump of size J leaves skip += J-1 between head and neck.
   */
  function masterCycleSkip(cy, snake) {
    if (!snake || snake.length < 2) {
      return 0;
    }
    var skip = 0;
    var i;
    var a;
    var b;
    var d;
    for (i = 0; i < snake.length - 1; i += 1) {
      a = snake[i];
      b = snake[i + 1];
      if (cy.index[a[0] + "," + a[1]] === undefined ||
          cy.index[b[0] + "," + b[1]] === undefined) {
        continue;
      }
      /* Neck b should be the cycle predecessor of a: forward b -> a is 1. */
      d = cycleDist(cy, b[0], b[1], a[0], a[1]);
      if (d > 1) {
        skip += d - 1;
      }
    }
    return skip;
  }

  /**
   * After moving head to (nr,nc), how much tip->head skip would remain.
   * Tip vacates when not eating (use second-last as new tip).
   */
  function masterSkipAfterMove(cy, snake, nr, nc, eating) {
    if (!snake || !snake.length) {
      return 0;
    }
    var newSnake = [[nr, nc]];
    var keepLen = eating ? snake.length : snake.length - 1;
    var j;
    for (j = 0; j < keepLen; j += 1) {
      newSnake.push(snake[j].slice());
    }
    return masterCycleSkip(cy, newSnake);
  }

  /**
   * Hamilton cycle step with fill-scaled skip debt on shortcuts.
   * opts.healOnly: prefer cycle successor (no food shortcuts) to clear skip debt.
   * opts.riskScale: override fill-derived scale (0..1); 0 = free-arc >= L only.
   * opts.vacateTipDist: measure free arc to the tip after a non-eat (neck).
   * opts.skipCap: max skipCost added for non-succ moves (soft pad).
   */
  function masterHamiltonAction(env, opts) {
    opts = opts || {};
    var healOnly = !!opts.healOnly;
    var vacateTipDist = !!opts.vacateTipDist;
    var skipCap = typeof opts.skipCap === "number" ? opts.skipCap : null;
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var n = env.size;
    var cy = getHamiltonCycle(n);
    var h = env.snake[0];
    var tail = env.snake[env.snake.length - 1];
    var hk = h[0] + "," + h[1];
    var cycleNext = cy.succ[hk];
    if (healOnly && cycleNext) {
      var healAct = actionFromTo(h[0], h[1], cycleNext[0], cycleNext[1]);
      if (healAct !== null && safe.indexOf(healAct) >= 0) {
        return healAct;
      }
    }
    /*
     * Odd boards omit far corner (n-1,n-1). Protocol:
     * 1) Follow the cycle until head borders that corner (no shortcuts).
     * 2) Step into the corner when food is there (if escape or win).
     * 3) From the corner, step back onto a cycle cell.
     */
    if (cy.omitCorner && !healOnly) {
      var omitR = n - 1;
      var omitC = n - 1;
      var onOmit = h[0] === omitR && h[1] === omitC;
      var foodOnOmit = !!(env.food && env.food[0] === omitR &&
        env.food[1] === omitC);
      var bordersOmit =
        Math.abs(h[0] - omitR) + Math.abs(h[1] - omitC) === 1;
      if (onOmit) {
        var rejoin = null;
        var ri;
        for (ri = 0; ri < safe.length; ri += 1) {
          var ra = safe[ri];
          var rr = h[0] + DIRS[ra][0];
          var rc = h[1] + DIRS[ra][1];
          if (cy.index[rr + "," + rc] !== undefined) {
            rejoin = ra;
            break;
          }
        }
        if (rejoin !== null) {
          return rejoin;
        }
      } else if (foodOnOmit) {
        if (bordersOmit) {
          var omitAct = actionFromTo(h[0], h[1], omitR, omitC);
          if (omitAct !== null && safe.indexOf(omitAct) >= 0) {
            var omitAfter = env.snake.length + 1;
            var omitWins = omitAfter >= n * n ||
              (typeof env.targetLength === "number" &&
                omitAfter >= env.targetLength);
            var omitFree = freeNeighborsAfter(env, omitR, omitC, true);
            var omitTipAdj =
              Math.abs(tail[0] - omitR) + Math.abs(tail[1] - omitC) === 1;
            if (omitWins || omitFree >= 1 ||
                (omitAfter === n * n - 1 && omitTipAdj)) {
              return omitAct;
            }
          }
        } else if (cycleNext) {
          var approachAct = actionFromTo(
            h[0], h[1], cycleNext[0], cycleNext[1]);
          if (approachAct !== null && safe.indexOf(approachAct) >= 0) {
            return approachAct;
          }
        }
      }
    }
    var needBase = env.snake.length;
    var fill = needBase / (n * n);
    var skipNow = masterCycleSkip(cy, env.snake);
    /* Explicit opts.riskScale (e.g. aggressive early) bypasses the floor-1 default.
       Default path still floors to 1 after full-episode bakeoff regressions. */
    var explicitRisk = typeof opts.riskScale === "number";
    var riskScale = explicitRisk
      ? opts.riskScale
      : (fill - 0.1) / 0.75;
    if (!explicitRisk && riskScale < 1) {
      riskScale = 1;
    }
    if (riskScale < 0) {
      riskScale = 0;
    }
    if (riskScale > 1) {
      riskScale = 1;
    }
    if (healOnly) {
      riskScale = 1;
    }
    var i;
    var a;
    var nr;
    var nc;
    var eating;
    var need;
    var distTail;
    var isCycleSucc;
    var skipAfter;
    var skipCost;
    var growBuf;
    var afterLen;
    var winsFill;
    var freeN;
    var flood;
    var tipR;
    var tipC;
    var candidates = [];
    var fallback = [];
    var trappedEats = [];

    for (i = 0; i < safe.length; i += 1) {
      a = safe[i];
      nr = h[0] + DIRS[a][0];
      nc = h[1] + DIRS[a][1];
      eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
      isCycleSucc = !!(cycleNext && cycleNext[0] === nr && cycleNext[1] === nc);
      if (healOnly && !isCycleSucc && !eating) {
        continue;
      }
      afterLen = needBase + (eating ? 1 : 0);
      winsFill = afterLen >= n * n ||
        (typeof env.targetLength === "number" && afterLen >= env.targetLength);
      freeN = freeNeighborsAfter(env, nr, nc, eating);
      flood = floodFromMove(env, nr, nc, eating).count;
      /* Defer zero-escape eats when a safer move may exist (5x5/15855). */
      if (eating && !winsFill && freeN < 1) {
        trappedEats.push({
          action: a,
          flood: flood,
          afterLen: afterLen,
          tipAdj: Math.abs(tail[0] - nr) + Math.abs(tail[1] - nc) === 1
        });
        continue;
      }
      skipAfter = masterSkipAfterMove(cy, env.snake, nr, nc, eating);
      growBuf = eating ? 1 : 0;
      skipCost = isCycleSucc ? 0 : Math.floor(skipAfter * riskScale);
      if (skipCap !== null && skipCost > skipCap) {
        skipCost = skipCap;
      }
      need = needBase + growBuf + skipCost;

      if (cy.index[nr + "," + nc] === undefined) {
        /* Odd boards omit one corner. Food there: eat when escape/win ok.
           Empty omit is only a last-resort garage (added below if needed). */
        if (!eating) {
          continue;
        }
        candidates.push({
          action: a,
          foodDist: 0,
          isCycleSucc: false,
          skipAfter: skipAfter,
          flood: flood,
          eating: true
        });
        continue;
      }

      /* Free arc head->tip; optional tip = cell that remains tip after non-eat. */
      tipR = tail[0];
      tipC = tail[1];
      if (vacateTipDist && !eating && env.snake.length >= 2) {
        tipR = env.snake[env.snake.length - 2][0];
        tipC = env.snake[env.snake.length - 2][1];
      }
      distTail = cycleDist(cy, nr, nc, tipR, tipC);
      if (distTail < need) {
        fallback.push({
          action: a,
          isCycleSucc: isCycleSucc,
          skipAfter: skipAfter,
          freeN: freeN,
          flood: flood,
          eating: eating,
          winsFill: winsFill
        });
        continue;
      }
      /* Large skip debt breaks the contiguous-arc free-arc theorem: do not
         trust cycleDist alone into a pocket (7x7 seed 205911 @skip=120). */
      if (skipNow > n && !winsFill && flood < needBase - 1) {
        fallback.push({
          action: a,
          isCycleSucc: isCycleSucc,
          skipAfter: skipAfter,
          freeN: freeN,
          flood: flood,
          eating: eating,
          winsFill: winsFill
        });
        continue;
      }

      var foodDist = cy.len;
      if (healOnly) {
        /* During heal, stick to the cycle; ignore food distance. */
        foodDist = isCycleSucc ? 0 : cy.len;
      } else if (env.food && cy.index[env.food[0] + "," + env.food[1]] !== undefined) {
        foodDist = cycleDist(cy, nr, nc, env.food[0], env.food[1]);
      } else if (eating) {
        foodDist = 0;
      } else if (env.food) {
        /* Food on omitted corner: rank by Manhattan so we approach it. */
        foodDist = Math.abs(env.food[0] - nr) + Math.abs(env.food[1] - nc);
      }
      candidates.push({
        action: a,
        foodDist: foodDist,
        isCycleSucc: isCycleSucc,
        skipAfter: skipAfter,
        flood: flood,
        eating: eating
      });
    }

    if (!candidates.length) {
      /*
       * Penultimate apple with tip adjacent: freeN counts tip as body (0) but
       * the next non-eat can step onto tip. Take that eat before cycleNext /
       * garage / fallback, or Observe phase-locks (7x7 seed 130748369).
       * When tip is not adjacent, keep deferring (5x5/15855 omit garage).
       */
      for (i = 0; i < trappedEats.length; i += 1) {
        if (trappedEats[i].afterLen === n * n - 1 && trappedEats[i].tipAdj) {
          return trappedEats[i].action;
        }
      }
      /* Prefer continuing the cycle / fallback before empty-omit garage. */
      if (cycleNext) {
        a = actionFromTo(h[0], h[1], cycleNext[0], cycleNext[1]);
        nr = cycleNext[0];
        nc = cycleNext[1];
        eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
        afterLen = needBase + (eating ? 1 : 0);
        winsFill = afterLen >= n * n ||
          (typeof env.targetLength === "number" && afterLen >= env.targetLength);
        flood = floodFromMove(env, nr, nc, eating).count;
        if (a !== null && safe.indexOf(a) >= 0 &&
            (winsFill || !eating ||
              freeNeighborsAfter(env, nr, nc, true) >= 1) &&
            !(skipNow > n && !winsFill && flood < needBase - 1)) {
          return a;
        }
      }
      if (fallback.length) {
        fallback.sort(function (x, y) {
          if ((x.flood || 0) !== (y.flood || 0)) {
            return (y.flood || 0) - (x.flood || 0);
          }
          if (x.isCycleSucc !== y.isCycleSucc) {
            return x.isCycleSucc ? -1 : 1;
          }
          return x.action - y.action;
        });
        return fallback[0].action;
      }
      /* Garage: empty omit with escape when freeN=0 eat was deferred. */
      for (i = 0; i < safe.length; i += 1) {
        a = safe[i];
        nr = h[0] + DIRS[a][0];
        nc = h[1] + DIRS[a][1];
        eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
        if (eating || cy.index[nr + "," + nc] !== undefined) {
          continue;
        }
        if (freeNeighborsAfter(env, nr, nc, false) < 1) {
          continue;
        }
        return a;
      }
      /* No safer alt: take the deferred freeN=0 eat rather than oscillate. */
      if (trappedEats.length) {
        return trappedEats[0].action;
      }
      return safe[0];
    }

    candidates.sort(function (x, y) {
      if (healOnly) {
        if (x.isCycleSucc !== y.isCycleSucc) {
          return x.isCycleSucc ? -1 : 1;
        }
        if ((x.skipAfter || 0) !== (y.skipAfter || 0)) {
          return (x.skipAfter || 0) - (y.skipAfter || 0);
        }
        return x.action - y.action;
      }
      /* Reject skip explosions before food chase (7x7 seed 205911: foodDist
         chose skipAfter 36 over cycleSucc skipAfter 2). Small skip deltas
         still lose to foodDist so normal chords stay fast. */
      var sx = x.skipAfter || 0;
      var sy = y.skipAfter || 0;
      if (Math.abs(sx - sy) > n) {
        return sx - sy;
      }
      /*
       * Prefer cycleSucc over an off-cycle step that raises skipAfter:
       * - off-cycle food (5x5 seed 197992 @len20 ate E skip4 vs N succ0)
       * - late chords once fill >= 40% (5x5/340534, 9x9/783998, ...)
       */
      if (x.isCycleSucc !== y.isCycleSucc) {
        var on = x.isCycleSucc ? x : y;
        var off = x.isCycleSucc ? y : x;
        if ((off.skipAfter || 0) > (on.skipAfter || 0)) {
          if ((off.eating && !on.eating) || fill >= 0.4) {
            return x.isCycleSucc ? -1 : 1;
          }
        }
      }
      if (x.foodDist !== y.foodDist) {
        return x.foodDist - y.foodDist;
      }
      if (sx !== sy) {
        return sx - sy;
      }
      if (x.isCycleSucc !== y.isCycleSucc) {
        return x.isCycleSucc ? -1 : 1;
      }
      return x.action - y.action;
    });
    return candidates[0].action;
  }

  /** Fill fraction where Hamilton Aggressive hands back to production Hamilton. */
  var HAMILTON_AGG_FILL_CAP = 0.2;

  /**
   * Aggressive Hamilton only while fill < cap; then production.
   * style: "soft" (skipCap=N + vacateTip) or "arc" (riskScale 0 + vacateTip).
   */
  function hamiltonFillCappedNextAction(env, style, cap) {
    if (!env || !env.snake || !env.snake.length) {
      return 0;
    }
    var fillCap = typeof cap === "number" ? cap : HAMILTON_AGG_FILL_CAP;
    var fill = env.snake.length / (env.size * env.size);
    if (fill >= fillCap) {
      return masterHamiltonAction(env);
    }
    if (style === "soft") {
      return masterHamiltonAction(env, {
        riskScale: 1,
        skipCap: env.size,
        vacateTipDist: true
      });
    }
    /* arc: free-arc chords (bench/ablation only) */
    return masterHamiltonAction(env, { riskScale: 0, vacateTipDist: true });
  }

  /**
   * Observe Hamilton Aggressive: skipCost capped at N while fill < 20%, then prod.
   */
  function hamiltonAggrNextAction(env) {
    return hamiltonFillCappedNextAction(env, "soft", HAMILTON_AGG_FILL_CAP);
  }

  /** Bench alias for free-arc fill-cap (not in Observe). */
  function hamiltonEarlyNextAction(env) {
    return hamiltonFillCappedNextAction(env, "arc", HAMILTON_AGG_FILL_CAP);
  }

  /** Bench alias for free-arc fill-cap (not in Observe). */
  function hamiltonArcNextAction(env) {
    return hamiltonFillCappedNextAction(env, "arc", HAMILTON_AGG_FILL_CAP);
  }

  /** Bench alias for Hamilton Aggressive (not in Observe). */
  function hamiltonSoftNextAction(env) {
    return hamiltonAggrNextAction(env);
  }

  /** Flood empty space from (nr,nc) after the move; tip vacated when not eating. */
  function floodFromMove(env, nr, nc, eating) {
    var empty = { count: 0, reachesTip: false };
    if (!env || !env.snake || !env.snake.length) {
      return empty;
    }
    var n = env.size;
    var body = {};
    var i;
    for (i = 0; i < env.snake.length; i += 1) {
      body[env.snake[i][0] + "," + env.snake[i][1]] = true;
    }
    var tip = env.snake[env.snake.length - 1];
    var tipKey = tip[0] + "," + tip[1];
    if (!eating) {
      delete body[tipKey];
    }
    delete body[nr + "," + nc];
    var q = [[nr, nc]];
    var seen = {};
    seen[nr + "," + nc] = true;
    var count = 0;
    var qi = 0;
    var d;
    var cur;
    var rr;
    var cc;
    var k;
    while (qi < q.length) {
      cur = q[qi];
      qi += 1;
      count += 1;
      for (d = 0; d < 4; d += 1) {
        rr = cur[0] + DIRS[d][0];
        cc = cur[1] + DIRS[d][1];
        if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
          continue;
        }
        k = rr + "," + cc;
        if (seen[k] || body[k]) {
          continue;
        }
        seen[k] = true;
        q.push([rr, cc]);
      }
    }
    return { count: count, reachesTip: !!seen[tipKey] };
  }

  function isWallAdjacentCell(n, r, c) {
    return r === 0 || c === 0 || r === n - 1 || c === n - 1;
  }

  /** Free orthogonal neighbors of (nr,nc) after the move (tip vacated if !eating). */
  function freeNeighborsAfter(env, nr, nc, eating) {
    if (!env || !env.snake || !env.snake.length) {
      return 0;
    }
    var n = env.size;
    var body = {};
    var i;
    var keepLen = eating ? env.snake.length : env.snake.length - 1;
    body[nr + "," + nc] = true;
    for (i = 0; i < keepLen; i += 1) {
      body[env.snake[i][0] + "," + env.snake[i][1]] = true;
    }
    var free = 0;
    var d;
    var rr;
    var cc;
    for (d = 0; d < 4; d += 1) {
      rr = nr + DIRS[d][0];
      cc = nc + DIRS[d][1];
      if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
        continue;
      }
      if (!body[rr + "," + cc]) {
        free += 1;
      }
    }
    return free;
  }

  /**
   * Shared early-game safety (from death-snap review: corridor/pocket traps).
   * opts.skipSlack: if set with cy+skipNow, reject skipAfter > skipNow + N.
   * opts.minFree: default 2 (1 when eating).
   * opts.floodMul: default 2 (empty flood >= floodMul*N and reaches tip).
   */
  function passesEarlySafety(env, nr, nc, eating, opts) {
    opts = opts || {};
    if (!pocketOk(env, nr, nc, eating)) {
      return { ok: false, reason: "pocket" };
    }
    var n = env.size;
    var floodMul = typeof opts.floodMul === "number" ? opts.floodMul : 2;
    var flood = floodFromMove(env, nr, nc, eating);
    if (!flood.reachesTip || flood.count < Math.max(2, Math.floor(n * floodMul))) {
      return { ok: false, reason: "flood" };
    }
    var minFree = typeof opts.minFree === "number"
      ? opts.minFree
      : (eating ? 1 : 2);
    if (freeNeighborsAfter(env, nr, nc, eating) < minFree) {
      return { ok: false, reason: "freeNeigh" };
    }
    if (opts.skipSlack && opts.cy) {
      var skipNow = typeof opts.skipNow === "number" ? opts.skipNow : 0;
      var skipAfter = masterSkipAfterMove(opts.cy, env.snake, nr, nc, eating);
      var slack = Math.max(1, n);
      if (skipAfter > skipNow + slack) {
        return { ok: false, reason: "skipSlack" };
      }
    }
    return { ok: true, reason: null };
  }

  /**
   * Safer BFS: chase food only when the step passes early safety (+ skip slack).
   * Fallback: greedy under same gate, then any gated safe, then healOnly Hamilton.
   */
  function safeBfsNextAction(env) {
    if (!env || !env.snake || !env.snake.length) {
      return 0;
    }
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var n = env.size;
    var cy = getHamiltonCycle(n);
    var skipNow = masterCycleSkip(cy, env.snake);
    var gateOpts = { skipSlack: true, cy: cy, skipNow: skipNow, floodMul: 1 };
    var h = env.snake[0];
    env._aggLastGateReject = null;

    function tryAction(a) {
      if (a === null || safe.indexOf(a) < 0) {
        return false;
      }
      var nr = h[0] + DIRS[a][0];
      var nc = h[1] + DIRS[a][1];
      var eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
      /* Always take food when adjacent; gates apply to non-eating steps. */
      if (eating) {
        return true;
      }
      var gate = passesEarlySafety(env, nr, nc, eating, gateOpts);
      if (!gate.ok) {
        env._aggLastGateReject = gate.reason;
        return false;
      }
      return true;
    }

    var bfs = bfsNextAction(env);
    if (tryAction(bfs)) {
      return bfs;
    }
    var g = greedyNextAction(env);
    if (tryAction(g)) {
      return g;
    }
    var i;
    var scored = [];
    for (i = 0; i < safe.length; i += 1) {
      var a = safe[i];
      var nr = h[0] + DIRS[a][0];
      var nc = h[1] + DIRS[a][1];
      var eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
      if (eating) {
        scored.push({ a: a, key: 0 });
        continue;
      }
      var gate = passesEarlySafety(env, nr, nc, eating, gateOpts);
      if (!gate.ok) {
        continue;
      }
      var man = env.food
        ? Math.abs(nr - env.food[0]) + Math.abs(nc - env.food[1])
        : 0;
      scored.push({ a: a, key: 1 + man });
    }
    if (scored.length) {
      scored.sort(function (x, y) {
        if (x.key !== y.key) {
          return x.key - y.key;
        }
        return x.a - y.a;
      });
      return scored[0].a;
    }
    env._aggLastGateReject = env._aggLastGateReject || "healFallback";
    return masterHamiltonAction(env, { healOnly: true });
  }

  var AGG_HANDOFF_FILL = 0.5;
  var AGG_VARIANTS = {
    aggRisk025: true,
    aggSkipSlack: true,
    aggPocket: true,
    aggFlood: true,
    aggWallFood: true,
    aggNetProgress: true,
    aggSafe: true,
    aggPocketSafe: true,
    aggNetSafe: true
  };

  function cycleFoodDist(cy, env, nr, nc, eating) {
    if (eating) {
      return 0;
    }
    if (env.food && cy.index[env.food[0] + "," + env.food[1]] !== undefined) {
      return cycleDist(cy, nr, nc, env.food[0], env.food[1]);
    }
    return cy.len;
  }

  /**
   * Aggressive early Hamilton (fill < 0.5): cycle-aware food chase with variant
   * safety gates. At/after 0.5: healOnly until skip0, then conservative Hamilton.
   * Safer variants (aggSafe / aggPocketSafe / aggNetSafe) use passesEarlySafety
   * on shortcuts (pocket + flood2N + freeNeigh>=2 + skipSlack).
   */
  function aggressiveHamiltonAction(env, variant) {
    if (!env || !env.snake || !env.snake.length) {
      return 0;
    }
    if (!AGG_VARIANTS[variant]) {
      variant = "aggPocket";
    }
    var n = env.size;
    var cells = n * n;
    var fill = env.snake.length / cells;
    var cy = getHamiltonCycle(n);
    var skipNow = masterCycleSkip(cy, env.snake);
    env._aggVariant = variant;
    env._aggLastGateReject = null;
    var useEarlySafety = variant === "aggSafe" ||
      variant === "aggPocketSafe" ||
      variant === "aggNetSafe";

    if (fill >= AGG_HANDOFF_FILL) {
      env._aggPhase = skipNow > 0 ? "heal" : "hamilton";
      if (skipNow > 0) {
        return masterHamiltonAction(env, { healOnly: true });
      }
      return masterHamiltonAction(env);
    }

    env._aggPhase = "agg";
    if (variant === "aggRisk025") {
      return masterHamiltonAction(env, { riskScale: 0.25 });
    }

    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var h = env.snake[0];
    var hk = h[0] + "," + h[1];
    var cycleNext = cy.succ[hk];
    var foodDistNow = cycleFoodDist(cy, env, h[0], h[1], false);
    /* floodMul 1 (=N empties) — 2N stalled growth past ~12% fill in bakeoff. */
    var gateOpts = {
      skipSlack: true,
      cy: cy,
      skipNow: skipNow,
      floodMul: 1
    };
    var candidates = [];
    var rejectReasons = [];
    var i;
    var a;
    var nr;
    var nc;
    var eating;
    var isCycleSucc;
    var skipAfter;
    var foodDist;
    var flood;
    var wallAdj;
    var skipInc;
    var foodDrop;
    var pass;
    var reason;
    var gate;

    for (i = 0; i < safe.length; i += 1) {
      a = safe[i];
      nr = h[0] + DIRS[a][0];
      nc = h[1] + DIRS[a][1];
      eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
      isCycleSucc = !!(cycleNext && cycleNext[0] === nr && cycleNext[1] === nc);
      skipAfter = masterSkipAfterMove(cy, env.snake, nr, nc, eating);
      foodDist = cycleFoodDist(cy, env, nr, nc, eating);
      flood = floodFromMove(env, nr, nc, eating);
      wallAdj = isWallAdjacentCell(n, nr, nc) ? 1 : 0;
      skipInc = skipAfter - skipNow;
      foodDrop = foodDistNow - foodDist;
      pass = true;
      reason = null;

      if (useEarlySafety) {
        /* Always allow eating (else pocketOk-on-eat blocks growth). Cycle
           successor is escape hatch. Shortcuts need early safety. */
        if (eating || isCycleSucc) {
          pass = true;
        } else {
          gate = passesEarlySafety(env, nr, nc, eating, gateOpts);
          if (!gate.ok) {
            pass = false;
            reason = gate.reason;
          } else if (variant === "aggNetSafe" &&
              skipInc > 0 && foodDrop < skipInc) {
            pass = false;
            reason = "netProgress";
          }
        }
      } else if (!eating && !isCycleSucc) {
        /* Legacy variants: gates on shortcuts only. */
        if (variant === "aggSkipSlack") {
          if (skipAfter > skipNow + n) {
            pass = false;
            reason = "skipSlack";
          }
        } else if (variant === "aggPocket") {
          if (!pocketOk(env, nr, nc, eating)) {
            pass = false;
            reason = "pocket";
          }
        } else if (variant === "aggFlood") {
          if (!pocketOk(env, nr, nc, eating) ||
              !flood.reachesTip ||
              flood.count < 2 * n) {
            pass = false;
            reason = "flood";
          }
        } else if (variant === "aggWallFood") {
          if (!pocketOk(env, nr, nc, eating) || skipAfter > skipNow + n) {
            pass = false;
            reason = "wallGate";
          }
        } else if (variant === "aggNetProgress") {
          if (!pocketOk(env, nr, nc, eating)) {
            pass = false;
            reason = "pocket";
          } else if (skipInc > 0 && foodDrop < skipInc) {
            pass = false;
            reason = "netProgress";
          }
        }
      }

      if (!pass) {
        rejectReasons.push(reason);
        continue;
      }
      candidates.push({
        action: a,
        foodDist: foodDist,
        skipAfter: skipAfter,
        isCycleSucc: isCycleSucc,
        wallAdj: wallAdj
      });
    }

    /* Soft fallback: if only cycle/heal would run, allow pocketOk+freeNeigh
       shortcuts so we do not stall before 20% when flood/skip gates are tight. */
    if (useEarlySafety) {
      var hasShortcut = false;
      for (i = 0; i < candidates.length; i += 1) {
        if (!candidates[i].isCycleSucc) {
          hasShortcut = true;
          break;
        }
      }
      if (!hasShortcut) {
        for (i = 0; i < safe.length; i += 1) {
          a = safe[i];
          nr = h[0] + DIRS[a][0];
          nc = h[1] + DIRS[a][1];
          eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
          isCycleSucc = !!(cycleNext && cycleNext[0] === nr && cycleNext[1] === nc);
          if (isCycleSucc) {
            continue;
          }
          if (!eating && (!pocketOk(env, nr, nc, eating) ||
              freeNeighborsAfter(env, nr, nc, eating) < 2)) {
            continue;
          }
          skipAfter = masterSkipAfterMove(cy, env.snake, nr, nc, eating);
          foodDist = cycleFoodDist(cy, env, nr, nc, eating);
          wallAdj = isWallAdjacentCell(n, nr, nc) ? 1 : 0;
          candidates.push({
            action: a,
            foodDist: foodDist,
            skipAfter: skipAfter,
            isCycleSucc: false,
            wallAdj: wallAdj
          });
        }
        if (candidates.length && !hasShortcut) {
          env._aggLastGateReject = "softPocket";
        }
      }
    }

    if (!candidates.length) {
      env._aggLastGateReject = rejectReasons.length
        ? rejectReasons[0]
        : "noCandidate";
      return masterHamiltonAction(env, { healOnly: true });
    }

    candidates.sort(function (x, y) {
      if (x.foodDist !== y.foodDist) {
        return x.foodDist - y.foodDist;
      }
      if ((variant === "aggWallFood" || variant === "aggSafe") &&
          x.wallAdj !== y.wallAdj) {
        return y.wallAdj - x.wallAdj;
      }
      if (x.skipAfter !== y.skipAfter) {
        return x.skipAfter - y.skipAfter;
      }
      if (x.isCycleSucc !== y.isCycleSucc) {
        return x.isCycleSucc ? -1 : 1;
      }
      return x.action - y.action;
    });
    return candidates[0].action;
  }

  /** Fixed-capacity ring buffer (death snapshot store). */
  function createRingBuffer(capacity) {
    var cap = capacity | 0;
    if (cap < 1) {
      cap = 1;
    }
    var items = [];
    return {
      push: function (item) {
        items.push(item);
        while (items.length > cap) {
          items.shift();
        }
      },
      toArray: function () {
        return items.slice();
      },
      clear: function () {
        items = [];
      },
      size: function () {
        return items.length;
      }
    };
  }

  /** Compact board state for aggressive-Hamilton death dumps. */
  function captureAggBoardSnap(env, meta) {
    meta = meta || {};
    var cy = env && env.size ? getHamiltonCycle(env.size) : null;
    return {
      steps: env ? env.steps : 0,
      fill: env && env.size
        ? env.snake.length / (env.size * env.size)
        : 0,
      skip: cy ? masterCycleSkip(cy, env.snake) : 0,
      facing: env ? env.facing : 0,
      food: env && env.food ? env.food.slice() : null,
      snake: env && env.snake
        ? env.snake.map(function (p) { return p.slice(); })
        : [],
      size: env ? env.size : 0,
      lastAction: meta.lastAction != null ? meta.lastAction : null,
      variant: meta.variant || (env && env._aggVariant) || null,
      lastGateReject: meta.lastGateReject != null
        ? meta.lastGateReject
        : (env && env._aggLastGateReject) || null
    };
  }

  function formatAggSnapAscii(snap) {
    if (!snap || !snap.size) {
      return "(empty snap)";
    }
    var n = snap.size;
    var grid = [];
    var r;
    var c;
    var i;
    for (r = 0; r < n; r += 1) {
      grid[r] = [];
      for (c = 0; c < n; c += 1) {
        grid[r][c] = ".";
      }
    }
    for (i = 0; i < snap.snake.length; i += 1) {
      r = snap.snake[i][0];
      c = snap.snake[i][1];
      if (r < 0 || c < 0 || r >= n || c >= n) {
        continue;
      }
      grid[r][c] = i === 0 ? "H" : "o";
    }
    if (snap.food) {
      r = snap.food[0];
      c = snap.food[1];
      if (r >= 0 && c >= 0 && r < n && c < n) {
        grid[r][c] = "F";
      }
    }
    var lines = [];
    lines.push(
      "steps=" + snap.steps +
      " fill=" + (snap.fill * 100).toFixed(1) + "%" +
      " skip=" + snap.skip +
      " facing=" + snap.facing +
      " lastA=" + snap.lastAction +
      " reject=" + snap.lastGateReject
    );
    for (r = 0; r < n; r += 1) {
      lines.push(grid[r].join(""));
    }
    return lines.join("\n");
  }

  /**
   * Short hint for how a hard death might have been avoided, from ring snaps.
   */
  function avoidanceHintFromSnaps(snaps, deathKind) {
    var last = snaps && snaps.length ? snaps[snaps.length - 1] : null;
    if (!last) {
      return "No snapshots; cannot diagnose " + (deathKind || "death") + ".";
    }
    var parts = [];
    parts.push("Death=" + (deathKind || "?") + " after steps=" + last.steps);
    if (last.lastGateReject) {
      parts.push(
        "Last gate reject was '" + last.lastGateReject +
        "'; a stricter gate or heal step may have avoided the trap."
      );
    }
    if (last.skip > last.size) {
      parts.push(
        "Skip debt " + last.skip + " > N=" + last.size +
        "; healOnly / skipSlack earlier may have cleared the gap."
      );
    }
    if (last.snake && last.snake.length) {
      var head = last.snake[0];
      var free = 0;
      var d;
      var body = {};
      var i;
      for (i = 0; i < last.snake.length; i += 1) {
        body[last.snake[i][0] + "," + last.snake[i][1]] = true;
      }
      for (d = 0; d < 4; d += 1) {
        var rr = head[0] + DIRS[d][0];
        var cc = head[1] + DIRS[d][1];
        if (rr < 0 || cc < 0 || rr >= last.size || cc >= last.size) {
          continue;
        }
        if (!body[rr + "," + cc]) {
          free += 1;
        }
      }
      if (free <= 1) {
        parts.push(
          "Head had <=1 free neighbor in last snap; pocketOk/flood gate " +
          "or earlier wall-corridor bias may have left an escape."
        );
      }
    }
    parts.push(
      "Compare BFS neighbor choice at this snap; if BFS differs and stays " +
      "pocket-safe, prefer that safety gate over pure foodDist."
    );
    return parts.join(" ");
  }

  /**
   * Even Master handoff fill % for Tail Safe -> HealTS -> Cell Tree.
   * null = use pure Cell Tree (no TS early). Sweep:
   * bench-notes/ts-heal-ct-handoff-sweep.md
   */
  var MASTER_EVEN_TS_CT_HANDOFF_PCT = null;

  /**
   * Master (Observe): odd = Tail Safe Hybrid; even = Cell Tree, or
   * Tail Safe -> HealTS(skip0) -> Cell Tree when MASTER_EVEN_TS_CT_HANDOFF_PCT
   * is a number. Pure Cell Tree remains Observe "Cell Tree".
   */
  function masterNextAction(env) {
    if (!env || typeof env.size !== "number") {
      return tailSafeHybridNextAction(env);
    }
    if (env.size & 1) {
      return tailSafeHybridNextAction(env);
    }
    if (typeof MASTER_EVEN_TS_CT_HANDOFF_PCT === "number") {
      return tailSafeCellTreeHybridNextAction(env, {
        handoffFillPct: MASTER_EVEN_TS_CT_HANDOFF_PCT
      });
    }
    var ct = global.SNAKE_ENV && global.SNAKE_ENV.cellTreeNextAction;
    if (typeof ct === "function") {
      return ct(env);
    }
    return tailSafeHybridNextAction(env);
  }

  /**
   * Legacy Master: Avoid Pocket until max(5% board, one width), heal until
   * skip0 (grace +N), then Hamilton. Kept for ablation benches.
   */
  function masterLegacyNextAction(env) {
    if (!env || !env.snake || !env.snake.length) {
      return 0;
    }
    var n = env.size;
    var threshold = masterEarlyThreshold(n);
    var phase = env._masterPhase || "early";
    var cy = getHamiltonCycle(n);
    var skip = masterCycleSkip(cy, env.snake);
    var len = env.snake.length;

    if (phase === "early") {
      if (len >= threshold) {
        phase = skip === 0 ? "hamilton" : "heal";
        env._masterPhase = phase;
      } else {
        env._masterPhase = "early";
        return masterEarlyAction(env);
      }
    }

    if (phase === "heal") {
      if (skip === 0 || len >= threshold + n) {
        env._masterPhase = "hamilton";
        return masterHamiltonAction(env);
      }
      env._masterPhase = "heal";
      return masterHamiltonAction(env, { healOnly: true });
    }

    env._masterPhase = "hamilton";
    return masterHamiltonAction(env);
  }

  /* ---------- planAhead: L-step path with tip vacation; reuse/extend ---------- */

  function planCopySnake(snake) {
    var out = [];
    var i;
    for (i = 0; i < snake.length; i += 1) {
      out.push(snake[i].slice());
    }
    return out;
  }

  function planApplyMove(snake, nr, nc, eating) {
    var next = [[nr, nc]];
    var keep = eating ? snake.length : snake.length - 1;
    var j;
    for (j = 0; j < keep; j += 1) {
      next.push(snake[j].slice());
    }
    return next;
  }

  /*
   * Idea G: ring-buffer snake with O(1) apply/undo for sequential plan sims.
   * Layout: cell i is at (_ringHead + i) % cap; tip at i = len-1.
   */
  var _ringR = null;
  var _ringC = null;
  var _ringCap = 0;
  var _ringHead = 0;
  var _ringLen = 0;
  var _ringUndoHead = null;
  var _ringUndoLen = null;
  var _ringUndoEat = null;
  var _ringUndoN = 0;
  var _ringUndoCap = 0;

  function ringEnsure(cap) {
    var need = cap < 4 ? 4 : cap;
    var i;
    var idx;
    var tmpR;
    var tmpC;
    if (_ringCap < need) {
      /* Grow: linearize live body to index 0 so wrap indices stay valid. */
      tmpR = new Int16Array(need);
      tmpC = new Int16Array(need);
      if (_ringR && _ringLen > 0) {
        for (i = 0; i < _ringLen; i += 1) {
          idx = (_ringHead + i) % _ringCap;
          tmpR[i] = _ringR[idx];
          tmpC[i] = _ringC[idx];
        }
      }
      _ringR = tmpR;
      _ringC = tmpC;
      _ringHead = 0;
      _ringCap = need;
      _ringUndoN = 0;
    }
    if (_ringUndoCap < need) {
      _ringUndoHead = new Int32Array(need);
      _ringUndoLen = new Int32Array(need);
      _ringUndoEat = new Uint8Array(need);
      _ringUndoCap = need;
    }
  }

  /** Load snake; pass boardN so cap fits a full-board body (no mid-sim grow). */
  function ringLoad(snake, boardN) {
    var i;
    var L = snake.length;
    var need = L + 8;
    if (boardN && boardN * boardN + 2 > need) {
      need = boardN * boardN + 2;
    }
    ringEnsure(need);
    _ringHead = 0;
    _ringLen = L;
    _ringUndoN = 0;
    for (i = 0; i < L; i += 1) {
      _ringR[i] = snake[i][0];
      _ringC[i] = snake[i][1];
    }
  }

  function ringHeadR() {
    return _ringR[_ringHead];
  }

  function ringHeadC() {
    return _ringC[_ringHead];
  }

  function ringLen() {
    return _ringLen;
  }

  function ringCellBlocked(n, r, c, eating) {
    var i;
    var idx;
    if (r < 0 || c < 0 || r >= n || c >= n) {
      return true;
    }
    for (i = 0; i < _ringLen; i += 1) {
      idx = (_ringHead + i) % _ringCap;
      if (_ringR[idx] === r && _ringC[idx] === c) {
        if (!eating && i === _ringLen - 1) {
          return false;
        }
        return true;
      }
    }
    return false;
  }

  function ringApply(nr, nc, eating) {
    var newHead;
    if (eating && _ringLen + 1 > _ringCap) {
      ringEnsure(_ringLen + 8);
    }
    _ringUndoHead[_ringUndoN] = _ringHead;
    _ringUndoLen[_ringUndoN] = _ringLen;
    _ringUndoEat[_ringUndoN] = eating ? 1 : 0;
    _ringUndoN += 1;
    newHead = (_ringHead - 1 + _ringCap) % _ringCap;
    _ringR[newHead] = nr;
    _ringC[newHead] = nc;
    _ringHead = newHead;
    if (eating) {
      _ringLen += 1;
    }
  }

  function ringUndo() {
    if (_ringUndoN <= 0) {
      return;
    }
    _ringUndoN -= 1;
    _ringHead = _ringUndoHead[_ringUndoN];
    _ringLen = _ringUndoLen[_ringUndoN];
  }

  function ringOccFill(occ, n) {
    var need = n * n;
    var i;
    var idx;
    for (i = 0; i < need; i += 1) {
      occ[i] = 0;
    }
    for (i = 0; i < _ringLen; i += 1) {
      idx = (_ringHead + i) % _ringCap;
      occ[_ringR[idx] * n + _ringC[idx]] = 1;
    }
  }

  function ringFloodCount(n) {
    occEnsureScratch(n);
    ringOccFill(_occScratch, n);
    return occFloodFromHead(n, _occScratch, ringHeadR(), ringHeadC(), -1).count;
  }

  function ringHeadReachesTip(n) {
    var tipI;
    var tipIdx;
    if (_ringLen < 2) {
      return true;
    }
    occEnsureScratch(n);
    ringOccFill(_occScratch, n);
    tipI = (_ringHead + _ringLen - 1) % _ringCap;
    tipIdx = _ringR[tipI] * n + _ringC[tipI];
    return occFloodFromHead(n, _occScratch, ringHeadR(), ringHeadC(), tipIdx)
      .reachesTip;
  }

  function ringFreeNeigh(n) {
    var d;
    var rr;
    var cc;
    var freeN = 0;
    var hr = ringHeadR();
    var hc = ringHeadC();
    for (d = 0; d < 4; d += 1) {
      rr = hr + DIRS[d][0];
      cc = hc + DIRS[d][1];
      if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
        continue;
      }
      if (!ringCellBlocked(n, rr, cc, true)) {
        freeN += 1;
      }
    }
    return freeN;
  }

  /** Materialize ring to a plain snake (for tip-reach helpers that need arrays). */
  function ringToSnake() {
    var out = [];
    var i;
    var idx;
    for (i = 0; i < _ringLen; i += 1) {
      idx = (_ringHead + i) % _ringCap;
      out.push([_ringR[idx], _ringC[idx]]);
    }
    return out;
  }

  function planCellBlocked(n, snake, r, c, eating) {
    if (r < 0 || c < 0 || r >= n || c >= n) {
      return true;
    }
    var i;
    var tip = snake.length - 1;
    for (i = 0; i < snake.length; i += 1) {
      if (snake[i][0] === r && snake[i][1] === c) {
        if (!eating && i === tip) {
          return false;
        }
        return true;
      }
    }
    return false;
  }

  /** Empty cells reachable from head after snake is in place (head cell excluded). */
  function planFloodFromSnake(n, snake) {
    if (!snake || !snake.length) {
      return 0;
    }
    occEnsureScratch(n);
    occFillFromSnake(_occScratch, n, snake);
    return occFloodFromHead(n, _occScratch, snake[0][0], snake[0][1], -1).count;
  }

  /**
   * Flood + tip-reach after a candidate move. Optional baseOcc (e.g. env._occ)
   * is mutated with make/undo (no full-grid copy); otherwise fills scratch from
   * snake then undoes the delta.
   */
  function planCandidateFloodTip(n, snake, nr, nc, eating, baseOcc) {
    if (!snake || !snake.length) {
      return { flood: 0, reachesTip: true, need: 0 };
    }
    var occ;
    var undo;
    var afterLen = eating ? snake.length + 1 : snake.length;
    var tipIdx = -1;
    var tipR;
    var tipC;
    var flood;
    var tipStats;
    if (baseOcc) {
      occ = baseOcc;
    } else {
      occEnsureScratch(n);
      occ = _occScratch;
      occFillFromSnake(occ, n, snake);
    }
    undo = occApplyMoveDelta(occ, n, snake, nr, nc, eating);
    if (eating) {
      tipR = snake[snake.length - 1][0];
      tipC = snake[snake.length - 1][1];
    } else if (snake.length >= 2) {
      tipR = snake[snake.length - 2][0];
      tipC = snake[snake.length - 2][1];
    } else {
      tipR = nr;
      tipC = nc;
    }
    tipIdx = tipR * n + tipC;
    /* One solid flood; tip-reach via O(1) adjacency (idea B). */
    tipStats = occFloodFromHead(n, occ, nr, nc, -1);
    flood = tipStats.count;
    occUndoMoveDelta(occ, undo);
    return {
      flood: flood,
      reachesTip: occTipReachableFromSolidFlood(n, tipIdx, tipStats.hIdx, tipStats.stamp),
      need: afterLen - 1
    };
  }

  function planCandidateFloodTipEnv(env, nr, nc, eating) {
    ensureEnvOccupancy(env);
    return planCandidateFloodTip(env.size, env.snake, nr, nc, eating, env._occ);
  }

  /** Tip-vacate BFS from head; stops when tip reached (idea D helper). */
  function occReachesTipEarly(n, occ, hr, hc, tipIdx) {
    if (tipIdx < 0) {
      return true;
    }
    var hIdx = hr * n + hc;
    if (tipIdx === hIdx) {
      return true;
    }
    occEnsureScratch(n);
    var stamp = occNextStamp();
    var qi = 0;
    var qe = 0;
    var d;
    var rr;
    var cc;
    var idx;
    _bfsQr[qe] = hr;
    _bfsQc[qe] = hc;
    qe += 1;
    _seenStamp[hIdx] = stamp;
    while (qi < qe) {
      hr = _bfsQr[qi];
      hc = _bfsQc[qi];
      qi += 1;
      idx = hr * n + hc;
      if (idx === tipIdx) {
        return true;
      }
      for (d = 0; d < 4; d += 1) {
        rr = hr + DIRS[d][0];
        cc = hc + DIRS[d][1];
        if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
          continue;
        }
        idx = rr * n + cc;
        if (_seenStamp[idx] === stamp) {
          continue;
        }
        if (occ[idx] && idx !== tipIdx && idx !== hIdx) {
          continue;
        }
        _seenStamp[idx] = stamp;
        _bfsQr[qe] = rr;
        _bfsQc[qe] = cc;
        qe += 1;
      }
    }
    return false;
  }

  /**
   * Tip-reach only after candidate delta (early-exit BFS). flood left null for
   * lazy fill (idea D).
   */
  function planCandidateTipReachEnv(env, nr, nc, eating) {
    ensureEnvOccupancy(env);
    var n = env.size;
    var snake = env.snake;
    var afterLen = eating ? snake.length + 1 : snake.length;
    var tipR;
    var tipC;
    var tipIdx;
    var undo;
    var reaches;
    undo = occApplyMoveDelta(env._occ, n, snake, nr, nc, eating);
    if (eating) {
      tipR = snake[snake.length - 1][0];
      tipC = snake[snake.length - 1][1];
    } else if (snake.length >= 2) {
      tipR = snake[snake.length - 2][0];
      tipC = snake[snake.length - 2][1];
    } else {
      tipR = nr;
      tipC = nc;
    }
    tipIdx = tipR * n + tipC;
    reaches = occReachesTipEarly(n, env._occ, nr, nc, tipIdx);
    occUndoMoveDelta(env._occ, undo);
    return {
      flood: null,
      reachesTip: reaches,
      need: afterLen - 1
    };
  }

  function planCandidateEnsureFlood(env, cand) {
    if (cand.flood !== null && cand.flood !== undefined) {
      return cand;
    }
    var stats = planCandidateFloodTipEnv(env, cand.nr, cand.nc, cand.eating);
    cand.flood = stats.flood;
    cand.need = stats.need;
    cand.reachesTip = stats.reachesTip;
    return cand;
  }

  /**
   * Largest empty component with tip vacated (openings that border the body).
   * Used by HealTS so tip-follow continues until the head can enter that opening.
   */
  function planMaxEmptyComponent(n, snake) {
    if (!snake || !snake.length) {
      return n * n;
    }
    var opens = planEmptyOpenings(n, snake, null, { vacateTip: true });
    return opens.maxSize;
  }

  /**
   * Score a candidate cell for planAhead. Lower is better.
   * style.wallAvoid (default): prefer interior so wall ring stays free as an
   * escape hatch - wall-adjacent cells pay +wallAvoidPenalty Manhattan.
   * style.wallPack + fill>=wallPackFill: opposite bias (wall-stick, man/slack).
   * style.strandAvoid (default): huge penalty when flood < empty after the move
   * (partition / last-bridge cut) so connected alternatives win.
   * Once on food's row/col, no wall bias - go for the fruit.
   * snakeLen = length AFTER the candidate move.
   */
  function planAheadScoreCell(n, nr, nc, eating, food, flood, snakeLen, style) {
    style = style || {};
    var man = food
      ? Math.abs(nr - food[0]) + Math.abs(nc - food[1])
      : 0;
    var empty = n * n - snakeLen;
    var strandedCells = empty - flood;
    if (strandedCells < 0) {
      strandedCells = 0;
    }
    var strandPen = typeof style.strandPenalty === "number"
      ? style.strandPenalty
      : 1e6;
    /* Per stranded cell so reconnecting beats deepening a cut when already split. */
    var strandCost = style.strandAvoid ? strandedCells * strandPen : 0;
    /* Eat wins hard only when free space stays one component. */
    if (eating && strandCost === 0) {
      return -1e9 - flood * 0.0001;
    }
    var fill = snakeLen / (n * n);
    var packAt = typeof style.wallPackFill === "number" ? style.wallPackFill : 0.1;
    var wallSlack = typeof style.wallSlack === "number" ? style.wallSlack : 1.2;
    var avoidPen = typeof style.wallAvoidPenalty === "number"
      ? style.wallAvoidPenalty
      : 1;
    var aligned = !!(food && (nr === food[0] || nc === food[1]));
    var score = man;
    var onWall = isWallAdjacentCell(n, nr, nc);
    if (style.wallPack && fill >= packAt && !aligned) {
      if (onWall) {
        /* Wall path allowed ~wallSlack times longer than straight Manhattan. */
        score = man / wallSlack;
      }
    } else if (style.wallAvoid && !aligned && onWall) {
      /* Keep walls free; pay a soft detour to stay interior when possible. */
      score = man + avoidPen;
    }
    score += strandCost;
    score -= flood * 0.0001;
    return score;
  }

  /**
   * Build a path of `need` cells from current snake. Prefer food; require
   * flood after each step >= remaining steps (body-scale safety).
   * style: { wallPack, wallPackFill } optional.
   */
  function planAheadBuild(env, need, style) {
    style = style || {};
    var n = env.size;
    var food = env.food ? env.food.slice() : null;
    var path = [];
    var step;
    var a;
    var nr;
    var nc;
    var eating;
    var best;
    var score;
    var flood;
    var remain;
    var afterLen;

    ringLoad(env.snake, n);
    for (step = 0; step < need; step += 1) {
      remain = need - step;
      best = null;
      for (a = 0; a < 4; a += 1) {
        nr = ringHeadR() + DIRS[a][0];
        nc = ringHeadC() + DIRS[a][1];
        eating = !!(food && food[0] === nr && food[1] === nc);
        if (ringCellBlocked(n, nr, nc, eating)) {
          continue;
        }
        ringApply(nr, nc, eating);
        flood = ringFloodCount(n);
        afterLen = ringLen();
        /* After this step, need (remain-1) more free steps. */
        if (flood < remain - 1) {
          ringUndo();
          continue;
        }
        if (style.tipReach && !ringHeadReachesTip(n)) {
          ringUndo();
          continue;
        }
        score = planAheadScoreCell(
          n, nr, nc, eating, food, flood, afterLen, style
        );
        ringUndo();
        if (!best || score < best.score ||
            (score === best.score && a < best.a)) {
          best = { nr: nr, nc: nc, eating: eating, score: score, a: a };
        }
      }
      if (!best) {
        return null;
      }
      path.push([best.nr, best.nc]);
      ringApply(best.nr, best.nc, best.eating);
      if (best.eating) {
        food = null;
      }
    }
    return path;
  }

  function planAheadValidate(env, path, style) {
    style = style || {};
    if (!path || !path.length) {
      return false;
    }
    var n = env.size;
    var food = env.food ? env.food.slice() : null;
    var i;
    var cell;
    var eating;
    var remain;
    ringLoad(env.snake, n);
    for (i = 0; i < path.length; i += 1) {
      cell = path[i];
      eating = !!(food && food[0] === cell[0] && food[1] === cell[1]);
      if (ringCellBlocked(n, cell[0], cell[1], eating)) {
        return false;
      }
      ringApply(cell[0], cell[1], eating);
      remain = path.length - i;
      if (ringFloodCount(n) < remain - 1) {
        return false;
      }
      if (style.tipReach && !ringHeadReachesTip(n)) {
        return false;
      }
      if (eating) {
        food = null;
      }
    }
    return true;
  }

  function planAheadExtendOne(env, path, style) {
    style = style || {};
    var n = env.size;
    var food = env.food ? env.food.slice() : null;
    var i;
    var cell;
    var eating;
    var needTail = 1;
    var a;
    var nr;
    var nc;
    var best = null;
    var flood;
    var score;
    var freeN;
    var afterLen;
    ringLoad(env.snake, n);
    for (i = 0; i < path.length; i += 1) {
      cell = path[i];
      eating = !!(food && food[0] === cell[0] && food[1] === cell[1]);
      if (ringCellBlocked(n, cell[0], cell[1], eating)) {
        return null;
      }
      ringApply(cell[0], cell[1], eating);
      if (eating) {
        food = null;
      }
    }
    for (a = 0; a < 4; a += 1) {
      nr = ringHeadR() + DIRS[a][0];
      nc = ringHeadC() + DIRS[a][1];
      eating = !!(food && food[0] === nr && food[1] === nc);
      if (ringCellBlocked(n, nr, nc, eating)) {
        continue;
      }
      ringApply(nr, nc, eating);
      flood = ringFloodCount(n);
      afterLen = ringLen();
      if (flood < needTail - 1) {
        ringUndo();
        continue;
      }
      if (style.tipReach && !ringHeadReachesTip(n)) {
        ringUndo();
        continue;
      }
      /* Soft: prefer freeNeigh >= 2 (large score penalty if narrower). */
      freeN = ringFreeNeigh(n);
      score = planAheadScoreCell(
        n, nr, nc, eating, food, flood, afterLen, style
      );
      if (!eating && freeN < 2) {
        score += 1e5;
      }
      ringUndo();
      if (!best || score < best.score || (score === best.score && a < best.a)) {
        best = { nr: nr, nc: nc, score: score, a: a };
      }
    }
    if (!best) {
      return null;
    }
    var out = [];
    for (i = 0; i < path.length; i += 1) {
      out.push(path[i].slice());
    }
    out.push([best.nr, best.nc]);
    return out;
  }

  function planAheadDesperateMove(env, style) {
    style = style || {};
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var n = env.size;
    var h = env.snake[0];
    var bi;
    var bestA = safe[0];
    var bestSc = 1e18;
    var ba;
    var bnr;
    var bnc;
    var beat;
    var bf;
    var bsc;
    var afterLen;
    ringLoad(env.snake, n);
    for (bi = 0; bi < safe.length; bi += 1) {
      ba = safe[bi];
      bnr = h[0] + DIRS[ba][0];
      bnc = h[1] + DIRS[ba][1];
      beat = !!(env.food && env.food[0] === bnr && env.food[1] === bnc);
      ringApply(bnr, bnc, beat);
      bf = ringFloodCount(n);
      afterLen = ringLen();
      bsc = planAheadScoreCell(
        n, bnr, bnc, beat, env.food, bf, afterLen, style
      );
      ringUndo();
      if (bsc < bestSc) {
        bestSc = bsc;
        bestA = ba;
      }
    }
    return bestA;
  }

  /**
   * Plan-ahead bot: keep a path of length ~= snake.length. Reuse/extend each
   * step; full replan after eat or if the stored plan is no longer safe.
   * opts.handoff: if no L-path, enter heal then Hamilton (via _planPhase).
   * opts.wallAvoid (default true unless wallPack): prefer interior cells.
   * opts.wallPack / wallPackFill: after fill threshold, prefer wall cells.
   */
  function planAheadNextAction(env, opts) {
    opts = opts || {};
    if (!env || !env.snake || !env.snake.length) {
      return 0;
    }
    var wallPack = !!opts.wallPack;
    var style = {
      wallPack: wallPack,
      wallPackFill: typeof opts.wallPackFill === "number" ? opts.wallPackFill : 0.1,
      wallSlack: typeof opts.wallSlack === "number" ? opts.wallSlack : 1.2,
      /* Default: always avoid packing walls so they remain an escape hatch. */
      wallAvoid: wallPack ? false : (opts.wallAvoid !== false),
      wallAvoidPenalty: typeof opts.wallAvoidPenalty === "number"
        ? opts.wallAvoidPenalty
        : 1,
      /* Opt-in: bakeoff @40% showed strandAvoid alone hurts vs wallAvoid-only. */
      strandAvoid: !!opts.strandAvoid,
      strandPenalty: typeof opts.strandPenalty === "number"
        ? opts.strandPenalty
        : 1e6
    };

    var needMul = typeof opts.planNeedMul === "number" && opts.planNeedMul > 0
      ? opts.planNeedMul
      : 1;
    var need = Math.max(1, Math.ceil(env.snake.length * needMul));
    style.tipReach = !!opts.tipReach;

    if (opts.handoff) {
      var phase = env._planPhase || "plan";
      var handoffAt = typeof opts.handoffAtLen === "number" ? opts.handoffAtLen : null;
      var cyH = getHamiltonCycle(env.size);
      var skipH = masterCycleSkip(cyH, env.snake);
      /* Length cap handoff: leave plan before typical early deaths. */
      if (phase === "plan" && handoffAt !== null && env.snake.length >= handoffAt) {
        env._planPhase = "heal";
        env._planHandoffLen = env.snake.length;
        phase = "heal";
      }
      /* Plan livelock: huge skip after early growth (14x14 seed 47531 stalled
         at len handoffAt-1). Require len>=2N so start-of-episode skip does not
         abort plan immediately. */
      if (phase === "plan" && skipH > env.size * env.size &&
          env.snake.length >= 2 * env.size) {
        env._planPhase = "heal";
        env._planHandoffLen = env.snake.length;
        phase = "heal";
      }
      if (phase === "heal") {
        /* Optional: return to planAhead once an L-path exists again. */
        if (opts.reenterPlan) {
          var rePath = planAheadBuild(env, need, style);
          if (rePath && rePath.length) {
            env._planPhase = "plan";
            env._planAhead = null;
            phase = "plan";
            /* Fall through into planAhead body below. */
          } else {
            return healPhaseNextAction(env, opts, style);
          }
        } else if (skipH === 0) {
          env._planPhase = "hamilton";
          return masterHamiltonAction(env);
        } else {
          return healPhaseNextAction(env, opts, style);
        }
      }
      if (phase === "hamilton") {
        return masterHamiltonAction(env);
      }
      env._planPhase = "plan";
    }
    var state = env._planAhead;
    var replan = false;
    if (!state || !state.path || !state.path.length) {
      replan = true;
    } else if (typeof state.apples === "number" && state.apples !== env.apples) {
      replan = true;
    } else if (!planAheadValidate(env, state.path, style)) {
      replan = true;
    }

    var path = null;
    if (!replan) {
      path = state.path;
      while (path.length < need) {
        var grown = planAheadExtendOne(env, path, style);
        if (!grown) {
          break;
        }
        path = grown;
      }
    } else {
      path = planAheadBuild(env, need, style);
    }

    if (!path || !path.length) {
      env._planAhead = { path: null, apples: env.apples, failed: true };
      if (opts.handoff) {
        env._planPhase = "heal";
        env._planHandoffLen = env.snake.length;
        return healPhaseNextAction(env, opts, style);
      }
      return planAheadDesperateMove(env, style);
    }

    var nextCell = path[0];
    var act = actionFromTo(
      env.snake[0][0],
      env.snake[0][1],
      nextCell[0],
      nextCell[1]
    );
    if (act === null) {
      env._planAhead = { path: null, apples: env.apples, failed: true };
      if (opts.handoff) {
        env._planPhase = "heal";
        env._planHandoffLen = env.snake.length;
        return healPhaseNextAction(env, opts, style);
      }
      var safe2 = env.safeActions();
      return safe2.length ? safe2[0] : (env.facing | 0);
    }

    /*
     * When skip debt is large, prefer the cycle successor over a narrower
     * planned step if succ keeps L-flood and has strictly more free neighbors
     * (7x7 seed 607678822: plan chose north freeN=2 over east succ freeN=3,
     * then body-died in heal). Clear plan so the next step replans.
     */
    var nPlan = env.size;
    var cyPlan = getHamiltonCycle(nPlan);
    var skipPlan = masterCycleSkip(cyPlan, env.snake);
    if (skipPlan > nPlan * nPlan && env.snake.length >= nPlan * 2) {
      var hPlan = env.snake[0];
      var succPlan = cyPlan.succ[hPlan[0] + "," + hPlan[1]];
      if (succPlan) {
        var succAct = actionFromTo(
          hPlan[0], hPlan[1], succPlan[0], succPlan[1]
        );
        var safePlan = env.safeActions();
        if (succAct !== null && succAct !== act &&
            safePlan.indexOf(succAct) >= 0) {
          var planEat = !!(env.food && env.food[0] === nextCell[0] &&
            env.food[1] === nextCell[1]);
          var planFree = freeNeighborsAfter(
            env, nextCell[0], nextCell[1], planEat
          );
          var succEat = !!(env.food && env.food[0] === succPlan[0] &&
            env.food[1] === succPlan[1]);
          var succFlood = floodFromMove(
            env, succPlan[0], succPlan[1], succEat
          ).count;
          var succFree = freeNeighborsAfter(
            env, succPlan[0], succPlan[1], succEat
          );
          if ((succEat || succFlood >= env.snake.length - 1) &&
              succFree > planFree) {
            env._planAhead = null;
            return succAct;
          }
        }
      }
    }

    var rest = [];
    var ri;
    for (ri = 1; ri < path.length; ri += 1) {
      rest.push(path[ri].slice());
    }
    var extended = planAheadExtendOne(env, path, style);
    if (extended && extended.length > 1) {
      rest = [];
      for (ri = 1; ri < extended.length; ri += 1) {
        rest.push(extended[ri].slice());
      }
    }
    env._planAhead = {
      path: rest,
      apples: env.apples,
      failed: false
    };
    /* After eating this step, next call will see apples bump and replan. */
    return act;
  }

  /**
   * Heal with body-scale lookahead: among safe moves, require flood-after >= L-1
   * (same idea as planAhead). Prefer eat, cycle succ, low skip, then higher
   * freeNeigh (escape-safe over 1-exit corridors when skip ties), then high flood.
   * Hard-drop narrow non-eat when a wide option exists and cycle skip > N*N
   * (stuck-heal corridor packing). Soft freeNeigh rank always. If nothing
   * passes L-flood, max flood among safe.
   */
  function healLookaheadNextAction(env, opts) {
    opts = opts || {};
    if (!env || !env.snake || !env.snake.length) {
      return 0;
    }
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var n = env.size;
    var L = env.snake.length;
    var cy = getHamiltonCycle(n);
    var h = env.snake[0];
    var hk = h[0] + "," + h[1];
    var cycleNext = cy.succ[hk];
    var candidates = [];
    var fallback = [];
    var i;
    var a;
    var nr;
    var nc;
    var eating;
    var flood;
    var skipAfter;
    var isSucc;
    var freeN;
    var bestWideSkip;
    var afterLen;
    var winsFill;
    var skipNow = masterCycleSkip(cy, env.snake);

    /*
     * Food on cycleNext must be taken even when flood < L-1 (7x7 seed
     * 827481772 @len 21: flood 9 < 20, HealLA walked south and never ate).
     * Off-cycle food still goes through the normal flood gate / ranking.
     */
    for (i = 0; i < safe.length; i += 1) {
      a = safe[i];
      nr = h[0] + DIRS[a][0];
      nc = h[1] + DIRS[a][1];
      eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
      isSucc = !!(cycleNext && cycleNext[0] === nr && cycleNext[1] === nc);
      if (!(eating && isSucc)) {
        continue;
      }
      afterLen = L + 1;
      winsFill = afterLen >= n * n ||
        (typeof env.targetLength === "number" && afterLen >= env.targetLength);
      if (winsFill || freeNeighborsAfter(env, nr, nc, true) >= 1) {
        return a;
      }
    }

    ringLoad(env.snake, n);
    for (i = 0; i < safe.length; i += 1) {
      a = safe[i];
      nr = h[0] + DIRS[a][0];
      nc = h[1] + DIRS[a][1];
      eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
      ringApply(nr, nc, eating);
      flood = ringFloodCount(n);
      ringUndo();
      freeN = freeNeighborsAfter(env, nr, nc, eating);
      skipAfter = masterSkipAfterMove(cy, env.snake, nr, nc, eating);
      isSucc = !!(cycleNext && cycleNext[0] === nr && cycleNext[1] === nc);
      afterLen = L + (eating ? 1 : 0);
      winsFill = afterLen >= n * n ||
        (typeof env.targetLength === "number" && afterLen >= env.targetLength);
      fallback.push({
        a: a,
        skipAfter: skipAfter,
        isSucc: isSucc,
        flood: flood,
        eating: eating,
        freeN: freeN
      });
      if (!winsFill && freeN < 1) {
        continue;
      }
      if (flood < L - 1) {
        continue;
      }
      candidates.push({
        a: a,
        skipAfter: skipAfter,
        isSucc: isSucc,
        flood: flood,
        eating: eating,
        freeN: freeN
      });
    }

    /*
     * Hard-drop narrow non-eat when a wide option exists and skip debt is still
     * large (Plan Ahead stuck-heal / corridor packing). Once skip is small
     * (Master finishing heal), only soft freeNeigh rank applies so we do not
     * stall clearing the cycle.
     */
    bestWideSkip = 1e18;
    for (i = 0; i < candidates.length; i += 1) {
      if (!candidates[i].eating && candidates[i].freeN >= 2) {
        bestWideSkip = 0;
        break;
      }
    }
    if (bestWideSkip === 0 && masterCycleSkip(cy, env.snake) > n * n) {
      /* Keep cycleSucc even when freeN=1 (seed 847806 @handoff20: dropping
         narrow succ for wide west left the cycle and body-died in a corridor). */
      candidates = candidates.filter(function (c) {
        return c.eating || c.isSucc || c.freeN >= 2;
      });
    }

    function rankHeal(x, y) {
      /* Heavy skip: prefer cycleSucc over off-cycle food so heal can clear
         (seed 827481772 ate east off-cycle @skip 82 and later body-died). */
      if (skipNow > n * n) {
        if (x.isSucc !== y.isSucc) {
          return x.isSucc ? -1 : 1;
        }
        if (x.eating !== y.eating) {
          return x.eating ? -1 : 1;
        }
      } else {
        if (x.eating !== y.eating) {
          return x.eating ? -1 : 1;
        }
        if (x.isSucc !== y.isSucc) {
          return x.isSucc ? -1 : 1;
        }
      }
      if (x.skipAfter !== y.skipAfter) {
        return x.skipAfter - y.skipAfter;
      }
      if (y.freeN !== x.freeN) {
        return y.freeN - x.freeN;
      }
      if (y.flood !== x.flood) {
        return y.flood - x.flood;
      }
      return x.a - y.a;
    }

    if (!candidates.length) {
      /* No L-flood move: take max flood among safe (reject tiny dead-ends). */
      fallback.sort(function (x, y) {
        if (y.flood !== x.flood) {
          return y.flood - x.flood;
        }
        return rankHeal(x, y);
      });
      return fallback[0].a;
    }

    candidates.sort(rankHeal);
    return candidates[0].a;
  }

  /**
   * Short tip-sim BFS during heal: search up to maxDepth moves with tip vacation
   * for a path that eats food while keeping flood >= L-1 after each step.
   * Returns first action, or null if none found.
   */
  function healTipSimNextAction(env, opts) {
    opts = opts || {};
    if (!env || !env.snake || !env.snake.length || !env.food) {
      return null;
    }
    var maxDepth = typeof opts.maxDepth === "number" ? opts.maxDepth : 20;
    var maxNodes = typeof opts.maxNodes === "number" ? opts.maxNodes : 5000;
    var n = env.size;
    var queue = [{
      snake: planCopySnake(env.snake),
      food: env.food.slice(),
      firstA: null,
      depth: 0
    }];
    var qi = 0;
    var nodes = 0;
    var cur;
    var a;
    var h;
    var nr;
    var nc;
    var eating;
    var after;
    var flood;
    var firstA;
    var L;

    while (qi < queue.length && nodes < maxNodes) {
      cur = queue[qi];
      qi += 1;
      if (cur.depth >= maxDepth || !cur.food) {
        continue;
      }
      h = cur.snake[0];
      L = cur.snake.length;
      for (a = 0; a < 4; a += 1) {
        nr = h[0] + DIRS[a][0];
        nc = h[1] + DIRS[a][1];
        eating = cur.food[0] === nr && cur.food[1] === nc;
        if (planCellBlocked(n, cur.snake, nr, nc, eating)) {
          continue;
        }
        after = planApplyMove(cur.snake, nr, nc, eating);
        flood = planFloodFromSnake(n, after);
        if (flood < after.length - 1) {
          continue;
        }
        firstA = cur.firstA === null ? a : cur.firstA;
        nodes += 1;
        if (eating) {
          return firstA;
        }
        queue.push({
          snake: after,
          food: cur.food,
          firstA: firstA,
          depth: cur.depth + 1
        });
      }
    }
    return null;
  }

  /** Heal-phase chooser: optional tip-sim to food, else HealLA, else healOnly. */
  function healPhaseNextAction(env, opts, style) {
    opts = opts || {};
    style = style || {};
    var tipA;
    var safe;
    if (opts.healTipSim) {
      tipA = healTipSimNextAction(env, {
        maxDepth: typeof opts.tipSimDepth === "number" ? opts.tipSimDepth : 20
      });
      if (tipA !== null) {
        safe = env.safeActions();
        if (safe.indexOf(tipA) >= 0) {
          return tipA;
        }
      }
    }
    if (opts.healLookahead) {
      return healLookaheadNextAction(env, { strandAvoid: style.strandAvoid });
    }
    return masterHamiltonAction(env, { healOnly: true });
  }

  /**
   * planAhead until length >= 15% of board (or no L-path); HealLA until skip0,
   * then Hamilton. Handoff is a fill fraction so it scales with n (was absolute 22).
   */
  function planAheadHybridNextAction(env) {
    var n = env && env.size ? env.size : 10;
    var handoffAtLen = Math.max(4, Math.ceil(n * n * 0.15));
    return planAheadNextAction(env, {
      handoff: true,
      handoffAtLen: handoffAtLen,
      healLookahead: true,
      wallAvoid: true
    });
  }

  /** Same hybrid without length cap (handoff only when L-path fails). */
  function planAheadHybridNoCapNextAction(env) {
    return planAheadNextAction(env, { handoff: true });
  }

  /**
   * Flood from head with tip cell treated as vacant. The tip cell is the space
   * the tip leaves on the next non-eat step (where a head following the body
   * would go) - not a still-occupied body segment.
   */
  function planVacatedTipStats(n, snake) {
    if (!snake || snake.length < 2) {
      return { reachesTip: true, count: 0 };
    }
    occEnsureScratch(n);
    occFillFromSnake(_occScratch, n, snake);
    var tip = snake[snake.length - 1];
    var tipIdx = tip[0] * n + tip[1];
    return occFloodFromHead(n, _occScratch, snake[0][0], snake[0][1], tipIdx);
  }

  /** True if head can reach the vacated tip cell. */
  function planHeadReachesTip(n, snake) {
    return planVacatedTipStats(n, snake).reachesTip;
  }

  /**
   * Tail-safe pose: vacated tip reachable and flood (tip vacant) >= L-1 so the
   * snake can follow its body without trapping in a tip-touching pocket.
   */
  function planTailSafePoseOk(n, snake) {
    if (!snake || snake.length < 2) {
      return true;
    }
    var stats = planVacatedTipStats(n, snake);
    return stats.reachesTip && stats.count >= snake.length - 1;
  }

  /**
   * Reconstruct parent chain from food back to head. Returns path excluding
   * head, or null.
   */
  function tailSafePathFromParents(h, fr, fc, n, parentR, parentC) {
    var path = [];
    var rr = fr;
    var cc = fc;
    var idx;
    while (!(rr === h[0] && cc === h[1])) {
      path.push([rr, cc]);
      idx = rr * n + cc;
      rr = parentR[idx];
      cc = parentC[idx];
      if (rr < 0) {
        return null;
      }
    }
    path.reverse();
    return path;
  }

  /**
   * Idea H: food-rooted dist + parent (tip vacated). Rebuild when food or
   * occupancy fingerprint changes. Path = walk parents from head to food.
   */
  function foodDistKey(env) {
    var food = env.food;
    if (!food) {
      return "";
    }
    return food[0] + "," + food[1] + ":" + (env._occFp || "");
  }

  function ensureFoodDistTipOnly(env) {
    var n;
    var need;
    var occ;
    var tip;
    var h;
    var tipIdx;
    var hIdx;
    var stamp;
    var qi;
    var qe;
    var d;
    var rr;
    var cc;
    var idx;
    var r0;
    var c0;
    var fr;
    var fc;
    var dist;
    var key;
    ensureEnvOccupancy(env);
    if (!env.food || !env.snake || !env.snake.length) {
      return false;
    }
    key = foodDistKey(env);
    n = env.size;
    need = n * n;
    if (env._foodDist && env._foodDist.length === need &&
        env._foodDistKey === key) {
      return true;
    }
    if (!env._foodDist || env._foodDist.length !== need) {
      env._foodDist = new Int16Array(need);
      env._foodParR = new Int16Array(need);
      env._foodParC = new Int16Array(need);
    }
    occ = env._occ;
    tip = env.snake[env.snake.length - 1];
    h = env.snake[0];
    tipIdx = tip[0] * n + tip[1];
    hIdx = h[0] * n + h[1];
    fr = env.food[0];
    fc = env.food[1];
    for (idx = 0; idx < need; idx += 1) {
      env._foodDist[idx] = -1;
      env._foodParR[idx] = -1;
      env._foodParC[idx] = -1;
    }
    occEnsureScratch(n);
    stamp = occNextStamp();
    qi = 0;
    qe = 0;
    _bfsQr[qe] = fr;
    _bfsQc[qe] = fc;
    _bfsDist[qe] = 0;
    qe += 1;
    idx = fr * n + fc;
    _seenStamp[idx] = stamp;
    env._foodDist[idx] = 0;
    while (qi < qe) {
      rr = _bfsQr[qi];
      cc = _bfsQc[qi];
      dist = _bfsDist[qi];
      qi += 1;
      r0 = rr;
      c0 = cc;
      for (d = 0; d < 4; d += 1) {
        rr = r0 + DIRS[d][0];
        cc = c0 + DIRS[d][1];
        if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
          continue;
        }
        idx = rr * n + cc;
        if (_seenStamp[idx] === stamp) {
          continue;
        }
        if (occ[idx] && idx !== tipIdx && idx !== hIdx) {
          continue;
        }
        _seenStamp[idx] = stamp;
        env._foodDist[idx] = dist + 1;
        env._foodParR[idx] = r0;
        env._foodParC[idx] = c0;
        _bfsQr[qe] = rr;
        _bfsQc[qe] = cc;
        _bfsDist[qe] = dist + 1;
        qe += 1;
      }
    }
    env._foodDistKey = key;
    return true;
  }

  /** Geodesic dist to food (tip vacated), or -1 if unreachable / no field. */
  function foodDistAt(env, r, c) {
    if (!ensureFoodDistTipOnly(env)) {
      return -1;
    }
    return env._foodDist[r * env.size + c];
  }

  /**
   * Tip-only food path: head-rooted BFS (stable when many shortest paths).
   * Food-rooted field (idea H) is for foodDistAt lookups - extracting the path
   * from food parents can pick a different equal-length first step (tip-pocket).
   */
  function tailSafeGridPathToFoodTipOnly(env) {
    var n = env.size;
    var occ = env._occ;
    var tip = env.snake[env.snake.length - 1];
    var h = env.snake[0];
    var food = env.food;
    var tipIdx = tip[0] * n + tip[1];
    var hIdx = h[0] * n + h[1];
    occEnsureScratch(n);
    var stamp = occNextStamp();
    var qi = 0;
    var qe = 0;
    var d;
    var rr;
    var cc;
    var idx;
    var parentR = new Int16Array(n * n);
    var parentC = new Int16Array(n * n);
    var found = false;
    var fr = food[0];
    var fc = food[1];
    _bfsQr[qe] = h[0];
    _bfsQc[qe] = h[1];
    qe += 1;
    _seenStamp[hIdx] = stamp;
    parentR[hIdx] = -1;
    parentC[hIdx] = -1;
    while (qi < qe) {
      rr = _bfsQr[qi];
      cc = _bfsQc[qi];
      qi += 1;
      if (rr === fr && cc === fc) {
        found = true;
        break;
      }
      var r0 = rr;
      var c0 = cc;
      for (d = 0; d < 4; d += 1) {
        rr = r0 + DIRS[d][0];
        cc = c0 + DIRS[d][1];
        if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
          continue;
        }
        idx = rr * n + cc;
        if (_seenStamp[idx] === stamp) {
          continue;
        }
        if (occ[idx] && idx !== tipIdx && idx !== hIdx) {
          continue;
        }
        _seenStamp[idx] = stamp;
        parentR[idx] = r0;
        parentC[idx] = c0;
        _bfsQr[qe] = rr;
        _bfsQc[qe] = cc;
        qe += 1;
      }
    }
    if (!found) {
      return null;
    }
    return tailSafePathFromParents(h, fr, fc, n, parentR, parentC);
  }

  /**
   * Leave-time food BFS: body cell at depth d walkable when leave[idx] <= d.
   * Path to food has no mid-path eats, so static leave matches tip advance.
   */
  function tailSafeGridPathToFoodLeaveAware(env) {
    var n = env.size;
    var occ = env._occ;
    var leave = env._leave;
    var h = env.snake[0];
    var food = env.food;
    var hIdx = h[0] * n + h[1];
    occEnsureScratch(n);
    var stamp = occNextStamp();
    var qi = 0;
    var qe = 0;
    var d;
    var rr;
    var cc;
    var idx;
    var dist;
    var distNext;
    var parentR = new Int16Array(n * n);
    var parentC = new Int16Array(n * n);
    var found = false;
    var fr = food[0];
    var fc = food[1];
    _bfsQr[qe] = h[0];
    _bfsQc[qe] = h[1];
    _bfsDist[qe] = 0;
    qe += 1;
    _seenStamp[hIdx] = stamp;
    parentR[hIdx] = -1;
    parentC[hIdx] = -1;
    while (qi < qe) {
      rr = _bfsQr[qi];
      cc = _bfsQc[qi];
      dist = _bfsDist[qi];
      qi += 1;
      if (rr === fr && cc === fc) {
        found = true;
        break;
      }
      var r0 = rr;
      var c0 = cc;
      distNext = dist + 1;
      for (d = 0; d < 4; d += 1) {
        rr = r0 + DIRS[d][0];
        cc = c0 + DIRS[d][1];
        if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
          continue;
        }
        idx = rr * n + cc;
        if (_seenStamp[idx] === stamp) {
          continue;
        }
        if (occ[idx] && leave[idx] > distNext && idx !== hIdx) {
          continue;
        }
        _seenStamp[idx] = stamp;
        parentR[idx] = r0;
        parentC[idx] = c0;
        _bfsQr[qe] = rr;
        _bfsQc[qe] = cc;
        _bfsDist[qe] = distNext;
        qe += 1;
      }
    }
    if (!found) {
      return null;
    }
    return tailSafePathFromParents(h, fr, fc, n, parentR, parentC);
  }

  /**
   * Shortest grid path head->food. Tip-only first (cheap on open boards);
   * leave-aware fallback when tip-vacate cannot reach food (idea C).
   */
  function tailSafeGridPathToFood(env) {
    if (!env.food || !env.snake || !env.snake.length) {
      return null;
    }
    ensureEnvOccupancy(env);
    var path = tailSafeGridPathToFoodTipOnly(env);
    if (path) {
      return path;
    }
    return tailSafeGridPathToFoodLeaveAware(env);
  }

  /**
   * Path walkable under static leave-times (O(1)/step), then tip-reach on the
   * final pose via ring sim (idea G - no per-step snake alloc).
   */
  function tailSafePathOk(env, path) {
    if (!path || !path.length) {
      return false;
    }
    ensureEnvOccupancy(env);
    var n = env.size;
    var leave = env._leave;
    var occ = env._occ;
    var h = env.snake[0];
    var hIdx = h[0] * n + h[1];
    var food = env.food ? env.food.slice() : null;
    var i;
    var cell;
    var eating;
    var idx;
    var advances;
    var eats = 0;
    ringLoad(env.snake, n);
    for (i = 0; i < path.length; i += 1) {
      cell = path[i];
      if (cell[0] < 0 || cell[1] < 0 || cell[0] >= n || cell[1] >= n) {
        return false;
      }
      eating = !!(food && food[0] === cell[0] && food[1] === cell[1]);
      /* Tip advances on non-eat moves; leave[idx] is from episode start. */
      advances = (i + 1) - eats - (eating ? 1 : 0);
      idx = cell[0] * n + cell[1];
      if (occ[idx] && leave[idx] > advances && idx !== hIdx) {
        return false;
      }
      ringApply(cell[0], cell[1], eating);
      if (eating) {
        food = null;
        eats += 1;
      }
    }
    return ringHeadReachesTip(n);
  }

  /** Leave-time at (r,c) after ensure (tests / debugging). Tip=1 .. head=L. */
  function leaveTimeAt(env, r, c) {
    if (!env || !env.inBounds(r, c)) {
      return 0;
    }
    ensureEnvOccupancy(env);
    return env._leave[r * env.size + c] | 0;
  }

  /** Orthogonal body neighbors of (r,c) on the current snake (incl. head/tip). */
  function countBodyAdj(env, r, c) {
    if (!env || !env.snake) {
      return 0;
    }
    var n = env.size;
    var d;
    var rr;
    var cc;
    var adj = 0;
    ensureEnvOccupancy(env);
    for (d = 0; d < 4; d += 1) {
      rr = r + DIRS[d][0];
      cc = c + DIRS[d][1];
      if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
        continue;
      }
      if (env._occ[rr * n + cc]) {
        adj += 1;
      }
    }
    return adj;
  }

  function tailSafeRankCand(best, cand, opts) {
    opts = opts || {};
    if (!best) {
      return cand;
    }
    if (cand.reachesTip && !best.reachesTip) {
      return cand;
    }
    if (cand.reachesTip === best.reachesTip && cand.eating && !best.eating) {
      return cand;
    }
    /* Prefer largest opening / roomier step before food Manhattan (601861). */
    if (cand.reachesTip === best.reachesTip && cand.eating === best.eating) {
      if (!!cand.inLargest !== !!best.inLargest) {
        return cand.inLargest ? cand : best;
      }
      if (cand.flood !== best.flood) {
        return cand.flood > best.flood ? cand : best;
      }
      if (opts.bodyHug && opts.cruise) {
        /* Cruise: pack along body inside one roomy tied-flood opening. */
        if ((cand.bodyAdj || 0) !== (best.bodyAdj || 0)) {
          return (cand.bodyAdj || 0) > (best.bodyAdj || 0) ? cand : best;
        }
        if ((cand.freeN || 0) !== (best.freeN || 0)) {
          return (cand.freeN || 0) > (best.freeN || 0) ? cand : best;
        }
      } else if ((cand.freeN || 0) !== (best.freeN || 0)) {
        return (cand.freeN || 0) > (best.freeN || 0) ? cand : best;
      }
      /* Equal openings: food side, else top-left entry. */
      if (!!cand.foodInOpen !== !!best.foodInOpen) {
        return cand.foodInOpen ? cand : best;
      }
      if ((cand.openTopR || 0) !== (best.openTopR || 0)) {
        return (cand.openTopR || 0) < (best.openTopR || 0) ? cand : best;
      }
      if ((cand.openTopC || 0) !== (best.openTopC || 0)) {
        return (cand.openTopC || 0) < (best.openTopC || 0) ? cand : best;
      }
      if (cand.man !== best.man) {
        return cand.man < best.man ? cand : best;
      }
      if (cand.a < best.a) {
        return cand;
      }
    }
    return best;
  }

  /**
   * Tail Safe (Observe): food path only if vacated tip still reachable after it.
   * Else tip-reaching one-steps. Ranking prefers largest-opening cell, then
   * flood, then freeN, then food Manhattan - so rim tip-chase does not beat an
   * equally-flooded open-room step (seed 601861 @len56). Tip-pocket escape when
   * tip flood < n, or tip leaves the largest opening while an L-flood escape
   * enters it. Else HealLA. No Hamilton.
   * opts.bodyHug: after a failed tip-safe food path, if tip-reaching options
   * share one roomy flood (cruise), rank bodyAdj; else freeN-max. Not in Observe.
   */
  function tailSafeNextAction(env, opts) {
    opts = opts || {};
    var bodyHug = !!opts.bodyHug;
    var rankOpts = {};
    if (!env || !env.snake || !env.snake.length) {
      return 0;
    }
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var n = env.size;
    var h = env.snake[0];
    var foodPath = tailSafeGridPathToFood(env);
    var firstA;
    var hasFoodPath = false;
    if (foodPath && foodPath.length && tailSafePathOk(env, foodPath)) {
      firstA = actionFromTo(h[0], h[1], foodPath[0][0], foodPath[0][1]);
      if (firstA !== null && safe.indexOf(firstA) >= 0) {
        hasFoodPath = true;
        return firstA;
      }
    }
    var i;
    var a;
    var nr;
    var nc;
    var eating;
    var bestTip = null;
    var bestEscape = null;
    var man;
    var flood;
    var freeN;
    var inLargest;
    var food = env.food;
    var openings = planEmptyOpeningsEnv(env, food, { vacateTip: true });
    var meta;
    var reachesTip;
    var need;
    var cand;
    var stats;
    var bodyAdj;
    var tipCands = [];
    var escCands = [];
    var cruise = false;
    var tipFlood0;
    var tipSameFlood;
    var tipRoomy;
    var tipRankPool = [];
    var anyLarge = false;
    var anyNonLarge = false;
    var useLazyFlood;
    ensureEnvOccupancy(env);
    for (i = 0; i < safe.length; i += 1) {
      nr = h[0] + DIRS[safe[i]][0];
      nc = h[1] + DIRS[safe[i]][1];
      if (openings.preferredCells[nr + "," + nc]) {
        anyLarge = true;
      } else {
        anyNonLarge = true;
      }
    }
    /* Lazy flood only helps when some safe cells are outside preferred opening. */
    useLazyFlood = anyLarge && anyNonLarge;
    for (i = 0; i < safe.length; i += 1) {
      a = safe[i];
      nr = h[0] + DIRS[a][0];
      nc = h[1] + DIRS[a][1];
      eating = !!(food && food[0] === nr && food[1] === nc);
      inLargest = !!openings.preferredCells[nr + "," + nc];
      freeN = freeNeighborsAfter(env, nr, nc, eating);
      bodyAdj = bodyHug ? countBodyAdj(env, nr, nc) : 0;
      meta = openings.cellMeta[nr + "," + nc] || null;
      /* Keep Manhattan for rank (geodesic man changed tip-pocket picks). */
      man = food
        ? Math.abs(food[0] - nr) + Math.abs(food[1] - nc)
        : 0;
      if (useLazyFlood && !inLargest && !eating) {
        /* Idea D: tip-reach only; skip full flood if tip-safe (cannot beat inLargest). */
        stats = planCandidateTipReachEnv(env, nr, nc, eating);
        cand = {
          a: a,
          nr: nr,
          nc: nc,
          eating: eating,
          man: man,
          flood: null,
          freeN: freeN,
          bodyAdj: bodyAdj,
          inLargest: inLargest,
          foodInOpen: !!(meta && meta.hasFood),
          openTopR: meta ? meta.topR : 1e9,
          openTopC: meta ? meta.topC : 1e9,
          reachesTip: stats.reachesTip,
          need: stats.need
        };
        if (stats.reachesTip) {
          /* Tip-safe but not inLargest / not eating - never ranks above inLargest. */
          continue;
        }
        planCandidateEnsureFlood(env, cand);
        if (cand.flood >= cand.need) {
          escCands.push(cand);
        }
        continue;
      }
      stats = planCandidateFloodTipEnv(env, nr, nc, eating);
      cand = {
        a: a,
        nr: nr,
        nc: nc,
        eating: eating,
        man: man,
        flood: stats.flood,
        freeN: freeN,
        bodyAdj: bodyAdj,
        inLargest: inLargest,
        foodInOpen: !!(meta && meta.hasFood),
        openTopR: meta ? meta.topR : 1e9,
        openTopC: meta ? meta.topC : 1e9,
        reachesTip: stats.reachesTip,
        need: stats.need
      };
      if (stats.reachesTip) {
        tipCands.push(cand);
      } else if (stats.flood >= stats.need) {
        escCands.push(cand);
      }
    }
    tipRankPool = tipCands;
    /*
     * Cruise: no tip-safe food path, >=2 tip-reaching steps, same flood, and
     * that flood is roomy (>= need and >= n) with freeN>=2 on each.
     */
    if (bodyHug && !hasFoodPath && tipCands.length >= 2) {
      tipFlood0 = tipCands[0].flood;
      tipSameFlood = true;
      tipRoomy = true;
      for (i = 0; i < tipCands.length; i += 1) {
        cand = tipCands[i];
        if (cand.flood !== tipFlood0) {
          tipSameFlood = false;
        }
        if ((cand.freeN || 0) < 2 ||
            cand.flood < (cand.need || 0) ||
            cand.flood < n) {
          tipRoomy = false;
        }
      }
      cruise = tipSameFlood && tipRoomy;
    }
    rankOpts = bodyHug
      ? { bodyHug: true, cruise: cruise, n: n }
      : {};
    for (i = 0; i < tipRankPool.length; i += 1) {
      bestTip = tailSafeRankCand(bestTip, tipRankPool[i], rankOpts);
    }
    for (i = 0; i < escCands.length; i += 1) {
      bestEscape = tailSafeRankCand(bestEscape, escCands[i], rankOpts);
    }
    if (bestTip && bestTip.flood >= n) {
      /* Prefer L-flood escape into largest opening over tip rim/coil. */
      if (bestEscape && bestEscape.inLargest && !bestTip.inLargest &&
          bestEscape.flood >= bestEscape.need) {
        return bestEscape.a;
      }
      return bestTip.a;
    }
    if (bestTip && bestTip.flood < n && bestEscape) {
      return bestEscape.a;
    }
    if (bestTip && bestEscape && !bestTip.inLargest && bestEscape.inLargest) {
      return bestEscape.a;
    }
    if (bestTip) {
      return bestTip.a;
    }
    return healLookaheadNextAction(env);
  }

  /**
   * Tail Safe body-hug variant (bench / ablation only - not in Observe).
   * Same tip-safe gates and food path. Prefer bodyAdj only in cruise mode:
   * no tip-safe food path, >=2 tip-reaching options sharing one roomy flood
   * (freeN>=2, flood>=L-1, flood>=n). Otherwise identical to Tail Safe ranking.
   */
  function tailSafeBodyHugNextAction(env) {
    return tailSafeNextAction(env, { bodyHug: true });
  }

  /**
   * Body-hug cruise until fill fraction (default 0.2), then plain Tail Safe.
   * Bench / ablation only - not in Observe.
   */
  function tailSafeBodyHugUntilFillNextAction(env, frac) {
    var f = typeof frac === "number" ? frac : 0.2;
    if (f < 0) {
      f = 0;
    }
    if (f > 1) {
      f = 1;
    }
    var n = env && env.size ? env.size : 10;
    var until = Math.max(3, Math.ceil(n * n * f));
    if (env && env.snake && env.snake.length >= until) {
      return tailSafeNextAction(env);
    }
    return tailSafeNextAction(env, { bodyHug: true });
  }

  /**
   * Plan Ahead (Observe): L-path with tip vacation; HealLA when no path;
   * return to plan when an L-path exists again. No Hamilton.
   */
  function planAheadReenterNextAction(env) {
    return planAheadNextAction(env, {
      handoff: true,
      healLookahead: true,
      reenterPlan: true,
      wallAvoid: true
      /* Dead-ends: flood-after < L-1 rejects the step; no strandAvoid needed. */
    });
  }

  /**
   * Plan+Tail hybrid (Observe): Plan Ahead L-path (flood >= remain) AND vacated
   * tip reachable after each planned step; HealLA reenter when no such path.
   */
  function planTailHybridNextAction(env) {
    return planAheadNextAction(env, {
      handoff: true,
      healLookahead: true,
      reenterPlan: true,
      wallAvoid: true,
      tipReach: true
    });
  }

  function tailSafePoseKey(env) {
    var h = env.snake[0];
    var f = env.food;
    var body = "";
    var i;
    for (i = 0; i < env.snake.length; i += 1) {
      body += env.snake[i][0] + "," + env.snake[i][1] + ";";
    }
    return h[0] + "," + h[1] + "|" +
      (f ? f[0] + "," + f[1] : "-") + "|" + env.facing + "|" + body;
  }

  /**
   * Enter Hamilton phase. By default keep following the cycle until
   * length >= targetLength (Observe Win-at %). Pass opts.skip0EarlyWin for
   * benches that treat skip0 as a guaranteed win without filling to target.
   */
  function tailSafeEnterHamilton(env, opts) {
    env._tsStallPhase = "hamilton";
    var earlyWin = !!(opts && opts.skip0EarlyWin);
    if (earlyWin || env.snake.length >= env.targetLength) {
      env.done = true;
      env.success = true;
      return env.facing | 0;
    }
    return masterHamiltonAction(env);
  }

  /** After HealTS reaches skip0: run Cell Tree for the rest of the episode. */
  function tailSafeEnterCellTree(env) {
    env._tsStallPhase = "cell_tree";
    var ct = global.SNAKE_ENV && global.SNAKE_ENV.cellTreeNextAction;
    if (typeof ct === "function") {
      return ct(env);
    }
    return masterHamiltonAction(env);
  }

  /* ---- Incremental empty-component labels (solid barrier / tip occupied) ---- */

  function openIncEnsure(env) {
    var n = env.size;
    var need = n * n;
    if (!env._compLab || env._compLab.length !== need) {
      env._compLab = new Int32Array(need);
      env._openIncDirty = true;
    }
    if (!env._compMeta) {
      env._compMeta = Object.create(null);
      env._openIncDirty = true;
    }
  }

  function openIncFind(env, id) {
    var meta = env._compMeta;
    var p = id;
    var root;
    var cur;
    if (!id || !meta[id]) {
      return 0;
    }
    while (meta[p] && meta[p].parent) {
      p = meta[p].parent;
    }
    root = p;
    cur = id;
    while (cur !== root && meta[cur] && meta[cur].parent) {
      p = meta[cur].parent;
      meta[cur].parent = root;
      cur = p;
    }
    return root;
  }

  function openIncUnion(env, a, b) {
    var meta = env._compMeta;
    a = openIncFind(env, a);
    b = openIncFind(env, b);
    if (!a) {
      return b;
    }
    if (!b || a === b) {
      return a;
    }
    if (meta[a].size < meta[b].size) {
      var t = a;
      a = b;
      b = t;
    }
    meta[b].parent = a;
    meta[a].size += meta[b].size;
    meta[a].hasFood = meta[a].hasFood || meta[b].hasFood;
    if (meta[b].topR < meta[a].topR ||
        (meta[b].topR === meta[a].topR && meta[b].topC < meta[a].topC)) {
      meta[a].topR = meta[b].topR;
      meta[a].topC = meta[b].topC;
    }
    meta[b].alive = false;
    return a;
  }

  function openIncNewId(env) {
    var id = env._compNextId | 0;
    if (id < 1) {
      id = 1;
    }
    env._compNextId = id + 1;
    return id;
  }

  function openIncRefreshCompMeta(env, root, seedR, seedC, food) {
    var n = env.size;
    var lab = env._compLab;
    var meta = env._compMeta;
    var fr = food ? food[0] : -1;
    var fc = food ? food[1] : -1;
    var stamp = occNextStamp();
    var qi = 0;
    var qe = 0;
    var count = 0;
    var hasFood = false;
    var topR = seedR;
    var topC = seedC;
    var d;
    var rr;
    var cc;
    var idx;
    var cr;
    var cc0;
    var nidx;
    var rid;
    occEnsureScratch(n);
    idx = seedR * n + seedC;
    _bfsQr[qe] = seedR;
    _bfsQc[qe] = seedC;
    qe += 1;
    _seenStamp[idx] = stamp;
    while (qi < qe) {
      cr = _bfsQr[qi];
      cc0 = _bfsQc[qi];
      qi += 1;
      count += 1;
      if (cr < topR || (cr === topR && cc0 < topC)) {
        topR = cr;
        topC = cc0;
      }
      if (fr === cr && fc === cc0) {
        hasFood = true;
      }
      for (d = 0; d < 4; d += 1) {
        rr = cr + DIRS[d][0];
        cc = cc0 + DIRS[d][1];
        if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
          continue;
        }
        nidx = rr * n + cc;
        if (_seenStamp[nidx] === stamp) {
          continue;
        }
        rid = lab[nidx];
        if (!rid || openIncFind(env, rid) !== root) {
          continue;
        }
        _seenStamp[nidx] = stamp;
        _bfsQr[qe] = rr;
        _bfsQc[qe] = cc;
        qe += 1;
      }
    }
    if (!meta[root]) {
      meta[root] = {};
    }
    meta[root].size = count;
    meta[root].hasFood = hasFood;
    meta[root].topR = topR;
    meta[root].topC = topC;
    meta[root].alive = count > 0;
    meta[root].parent = 0;
    return count;
  }

  function openIncRebuild(env, food) {
    ensureEnvOccupancy(env);
    openIncEnsure(env);
    var n = env.size;
    var occ = env._occ;
    var lab = env._compLab;
    var need = n * n;
    var r;
    var c;
    var idx;
    var qi;
    var qe;
    var d;
    var rr;
    var cc;
    var nidx;
    var cr;
    var cc0;
    var count;
    var hasFood;
    var topR;
    var topC;
    var id;
    var fr = food ? food[0] : -1;
    var fc = food ? food[1] : -1;
    var i;
    for (i = 0; i < need; i += 1) {
      lab[i] = 0;
    }
    env._compMeta = Object.create(null);
    env._compNextId = 1;
    occEnsureScratch(n);
    for (r = 0; r < n; r += 1) {
      for (c = 0; c < n; c += 1) {
        idx = r * n + c;
        if (occ[idx] || lab[idx]) {
          continue;
        }
        id = openIncNewId(env);
        qi = 0;
        qe = 0;
        _bfsQr[qe] = r;
        _bfsQc[qe] = c;
        qe += 1;
        lab[idx] = id;
        count = 0;
        hasFood = fr === r && fc === c;
        topR = r;
        topC = c;
        while (qi < qe) {
          cr = _bfsQr[qi];
          cc0 = _bfsQc[qi];
          qi += 1;
          count += 1;
          if (cr < topR || (cr === topR && cc0 < topC)) {
            topR = cr;
            topC = cc0;
          }
          if (fr === cr && fc === cc0) {
            hasFood = true;
          }
          for (d = 0; d < 4; d += 1) {
            rr = cr + DIRS[d][0];
            cc = cc0 + DIRS[d][1];
            if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
              continue;
            }
            nidx = rr * n + cc;
            if (occ[nidx] || lab[nidx]) {
              continue;
            }
            lab[nidx] = id;
            _bfsQr[qe] = rr;
            _bfsQc[qe] = cc;
            qe += 1;
          }
        }
        env._compMeta[id] = {
          size: count,
          hasFood: hasFood,
          topR: topR,
          topC: topC,
          alive: true,
          parent: 0
        };
      }
    }
    env._openIncDirty = false;
    env._openFoodKey = food ? food[0] + "," + food[1] : "-";
    env._openSolid = null;
    env._openVac = null;
    env._openDirty = false;
  }

  function openIncFreeCell(env, r, c, food) {
    openIncEnsure(env);
    var n = env.size;
    var lab = env._compLab;
    var idx = r * n + c;
    var d;
    var rr;
    var cc;
    var nidx;
    var nid;
    var root = 0;
    var fr = food ? food[0] : -1;
    var fc = food ? food[1] : -1;
    var hasFood = fr === r && fc === c;
    if (lab[idx]) {
      return;
    }
    for (d = 0; d < 4; d += 1) {
      rr = r + DIRS[d][0];
      cc = c + DIRS[d][1];
      if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
        continue;
      }
      nidx = rr * n + cc;
      nid = lab[nidx];
      if (!nid) {
        continue;
      }
      nid = openIncFind(env, nid);
      if (!nid) {
        continue;
      }
      root = root ? openIncUnion(env, root, nid) : nid;
    }
    if (!root) {
      root = openIncNewId(env);
      env._compMeta[root] = {
        size: 1,
        hasFood: hasFood,
        topR: r,
        topC: c,
        alive: true,
        parent: 0
      };
    } else {
      env._compMeta[root].size += 1;
      env._compMeta[root].hasFood = env._compMeta[root].hasFood || hasFood;
      if (r < env._compMeta[root].topR ||
          (r === env._compMeta[root].topR && c < env._compMeta[root].topC)) {
        env._compMeta[root].topR = r;
        env._compMeta[root].topC = c;
      }
    }
    lab[idx] = root;
    env._openSolid = null;
    env._openVac = null;
  }

  function openIncOccupyCell(env, r, c, food) {
    openIncEnsure(env);
    var n = env.size;
    var lab = env._compLab;
    var idx = r * n + c;
    var old = lab[idx];
    var root;
    var d;
    var rr;
    var cc;
    var nidx;
    var nid;
    var seeds = [];
    var seedSeen = Object.create(null);
    var i;
    var stamp;
    var qi;
    var qe;
    var cr;
    var cc0;
    var newId;
    if (!old) {
      return;
    }
    root = openIncFind(env, old);
    lab[idx] = 0;
    if (!root || !env._compMeta[root]) {
      return;
    }
    env._compMeta[root].size -= 1;
    if (env._compMeta[root].size <= 0) {
      env._compMeta[root].alive = false;
      env._compMeta[root].size = 0;
      env._openSolid = null;
      env._openVac = null;
      return;
    }
    for (d = 0; d < 4; d += 1) {
      rr = r + DIRS[d][0];
      cc = c + DIRS[d][1];
      if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
        continue;
      }
      nidx = rr * n + cc;
      nid = lab[nidx];
      if (!nid) {
        continue;
      }
      if (openIncFind(env, nid) !== root) {
        continue;
      }
      if (!seedSeen[nidx]) {
        seedSeen[nidx] = true;
        seeds.push([rr, cc]);
      }
    }
    if (seeds.length <= 1) {
      if (seeds.length === 1) {
        openIncRefreshCompMeta(env, root, seeds[0][0], seeds[0][1], food);
      }
      env._openSolid = null;
      env._openVac = null;
      return;
    }
    occEnsureScratch(n);
    stamp = occNextStamp();
    qi = 0;
    qe = 0;
    _bfsQr[qe] = seeds[0][0];
    _bfsQc[qe] = seeds[0][1];
    qe += 1;
    _seenStamp[seeds[0][0] * n + seeds[0][1]] = stamp;
    while (qi < qe) {
      cr = _bfsQr[qi];
      cc0 = _bfsQc[qi];
      qi += 1;
      for (d = 0; d < 4; d += 1) {
        rr = cr + DIRS[d][0];
        cc = cc0 + DIRS[d][1];
        if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
          continue;
        }
        nidx = rr * n + cc;
        if (_seenStamp[nidx] === stamp) {
          continue;
        }
        nid = lab[nidx];
        if (!nid || openIncFind(env, nid) !== root) {
          continue;
        }
        _seenStamp[nidx] = stamp;
        _bfsQr[qe] = rr;
        _bfsQc[qe] = cc;
        qe += 1;
      }
    }
    /* Relabel other islands first; only then refresh root (else root counts all). */
    for (i = 1; i < seeds.length; i += 1) {
      nidx = seeds[i][0] * n + seeds[i][1];
      if (_seenStamp[nidx] === stamp) {
        continue;
      }
      newId = openIncNewId(env);
      env._compMeta[newId] = {
        size: 0,
        hasFood: false,
        topR: seeds[i][0],
        topC: seeds[i][1],
        alive: true,
        parent: 0
      };
      qi = 0;
      qe = 0;
      _bfsQr[qe] = seeds[i][0];
      _bfsQc[qe] = seeds[i][1];
      qe += 1;
      _seenStamp[nidx] = stamp;
      lab[nidx] = newId;
      while (qi < qe) {
        cr = _bfsQr[qi];
        cc0 = _bfsQc[qi];
        qi += 1;
        for (d = 0; d < 4; d += 1) {
          rr = cr + DIRS[d][0];
          cc = cc0 + DIRS[d][1];
          if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
            continue;
          }
          nidx = rr * n + cc;
          if (_seenStamp[nidx] === stamp) {
            continue;
          }
          nid = lab[nidx];
          if (!nid || openIncFind(env, nid) !== root) {
            continue;
          }
          _seenStamp[nidx] = stamp;
          lab[nidx] = newId;
          _bfsQr[qe] = rr;
          _bfsQc[qe] = cc;
          qe += 1;
        }
      }
      openIncRefreshCompMeta(env, newId, seeds[i][0], seeds[i][1], food);
    }
    openIncRefreshCompMeta(env, root, seeds[0][0], seeds[0][1], food);
    env._openSolid = null;
    env._openVac = null;
  }

  function openIncApplyStep(env, hr, hc, freedTip, food) {
    if (!env) {
      return;
    }
    if (env._openIncDirty || !env._compLab) {
      env._openIncDirty = true;
      env._openSolid = null;
      env._openVac = null;
      return;
    }
    if (freedTip) {
      openIncFreeCell(env, freedTip[0], freedTip[1], food);
    }
    openIncOccupyCell(env, hr, hc, food);
    env._openFoodKey = food ? food[0] + "," + food[1] : "-";
  }

  function openIncRefreshFood(env, food) {
    if (!env || env._openIncDirty || !env._compMeta || !env._compLab) {
      invalidateEnvOpenings(env);
      return;
    }
    var key;
    var meta = env._compMeta;
    var fr = food ? food[0] : -1;
    var fc = food ? food[1] : -1;
    var n = env.size;
    var lab = env._compLab;
    var foodId = 0;
    var root;
    if (fr >= 0 && fc >= 0 && fr < n && fc < n) {
      foodId = lab[fr * n + fc];
      if (foodId) {
        foodId = openIncFind(env, foodId);
      }
    }
    for (key in meta) {
      if (!Object.prototype.hasOwnProperty.call(meta, key)) {
        continue;
      }
      if (!meta[key] || meta[key].parent || meta[key].alive === false) {
        continue;
      }
      root = openIncFind(env, +key);
      meta[key].hasFood = !!(foodId && root === foodId);
    }
    env._openFoodKey = food ? food[0] + "," + food[1] : "-";
    env._openSolid = null;
    env._openVac = null;
  }

  function openIncPickPreferred(env) {
    var meta = env._compMeta;
    var key;
    var m;
    var id;
    var maxSize = 0;
    var maxCount = 0;
    var preferred = null;
    var preferredId = 0;
    for (key in meta) {
      if (!Object.prototype.hasOwnProperty.call(meta, key)) {
        continue;
      }
      m = meta[key];
      if (!m || m.parent || m.alive === false || m.size <= 0) {
        continue;
      }
      id = +key;
      if (openIncFind(env, id) !== id) {
        continue;
      }
      if (m.size > maxSize) {
        maxSize = m.size;
        maxCount = 1;
        preferred = m;
        preferredId = id;
      } else if (m.size === maxSize) {
        maxCount += 1;
        if (!preferred) {
          preferred = m;
          preferredId = id;
        } else if (m.hasFood !== preferred.hasFood) {
          if (m.hasFood) {
            preferred = m;
            preferredId = id;
          }
        } else if (m.topR < preferred.topR ||
            (m.topR === preferred.topR && m.topC < preferred.topC)) {
          preferred = m;
          preferredId = id;
        }
      }
    }
    return {
      maxSize: maxSize,
      tiedMax: maxCount > 1,
      preferredId: preferredId,
      preferred: preferred
    };
  }

  function openIncPackSolid(env) {
    var n = env.size;
    var lab = env._compLab;
    var meta = env._compMeta;
    var pick = openIncPickPreferred(env);
    var out = {
      maxSize: pick.maxSize,
      tiedMax: pick.tiedMax,
      preferredCells: {},
      cellMeta: {}
    };
    var r;
    var c;
    var idx;
    var id;
    var root;
    var m;
    var ck;
    for (r = 0; r < n; r += 1) {
      for (c = 0; c < n; c += 1) {
        idx = r * n + c;
        id = lab[idx];
        if (!id) {
          continue;
        }
        root = openIncFind(env, id);
        m = meta[root];
        if (!m) {
          continue;
        }
        ck = r + "," + c;
        out.cellMeta[ck] = {
          size: m.size,
          hasFood: m.hasFood,
          topR: m.topR,
          topC: m.topC
        };
        if (root === pick.preferredId) {
          out.preferredCells[ck] = true;
        }
      }
    }
    return out;
  }

  /**
   * Tip-vacate view without mutating solid labels: tip cell joins adjacent
   * empty components.
   */
  function openIncPackVacateTip(env, food) {
    var n = env.size;
    var snake = env.snake;
    if (!snake || !snake.length) {
      return openIncPackSolid(env);
    }
    var tip = snake[snake.length - 1];
    var tr = tip[0];
    var tc = tip[1];
    var tipIdx = tr * n + tc;
    var lab = env._compLab;
    var meta = env._compMeta;
    var d;
    var rr;
    var cc;
    var nidx;
    var nid;
    var roots = [];
    var seen = Object.create(null);
    var i;
    var mergedSize = 1;
    var mergedFood = !!(food && food[0] === tr && food[1] === tc);
    var mergedTopR = tr;
    var mergedTopC = tc;
    var root;
    var m;
    var key;
    var id;
    var maxSize = 0;
    var maxCount = 0;
    var preferredId = -1;
    var preferredIsMerged = false;
    var preferred = null;
    var out = {
      maxSize: 0,
      tiedMax: false,
      preferredCells: {},
      cellMeta: {}
    };
    var fr = food ? food[0] : -1;
    var fc = food ? food[1] : -1;
    var ck;
    var inMerged;

    for (d = 0; d < 4; d += 1) {
      rr = tr + DIRS[d][0];
      cc = tc + DIRS[d][1];
      if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
        continue;
      }
      nidx = rr * n + cc;
      nid = lab[nidx];
      if (!nid) {
        continue;
      }
      nid = openIncFind(env, nid);
      if (!nid || seen[nid]) {
        continue;
      }
      seen[nid] = true;
      roots.push(nid);
      m = meta[nid];
      if (!m) {
        continue;
      }
      mergedSize += m.size;
      mergedFood = mergedFood || m.hasFood;
      if (m.topR < mergedTopR ||
          (m.topR === mergedTopR && m.topC < mergedTopC)) {
        mergedTopR = m.topR;
        mergedTopC = m.topC;
      }
    }

    function consider(size, hasFood, topR, topC, idOrMerged, isMerged) {
      if (size > maxSize) {
        maxSize = size;
        maxCount = 1;
        preferredId = idOrMerged;
        preferredIsMerged = isMerged;
        preferred = {
          size: size,
          hasFood: hasFood,
          topR: topR,
          topC: topC
        };
      } else if (size === maxSize) {
        maxCount += 1;
        if (!preferred) {
          preferredId = idOrMerged;
          preferredIsMerged = isMerged;
          preferred = {
            size: size,
            hasFood: hasFood,
            topR: topR,
            topC: topC
          };
        } else if (hasFood !== preferred.hasFood) {
          if (hasFood) {
            preferredId = idOrMerged;
            preferredIsMerged = isMerged;
            preferred = {
              size: size,
              hasFood: hasFood,
              topR: topR,
              topC: topC
            };
          }
        } else if (topR < preferred.topR ||
            (topR === preferred.topR && topC < preferred.topC)) {
          preferredId = idOrMerged;
          preferredIsMerged = isMerged;
          preferred = {
            size: size,
            hasFood: hasFood,
            topR: topR,
            topC: topC
          };
        }
      }
    }

    consider(mergedSize, mergedFood, mergedTopR, mergedTopC, -1, true);
    for (key in meta) {
      if (!Object.prototype.hasOwnProperty.call(meta, key)) {
        continue;
      }
      m = meta[key];
      id = +key;
      if (!m || m.parent || m.alive === false || m.size <= 0) {
        continue;
      }
      if (openIncFind(env, id) !== id) {
        continue;
      }
      if (seen[id]) {
        continue;
      }
      consider(m.size, m.hasFood, m.topR, m.topC, id, false);
    }

    out.maxSize = maxSize;
    out.tiedMax = maxCount > 1;
    for (rr = 0; rr < n; rr += 1) {
      for (cc = 0; cc < n; cc += 1) {
        nidx = rr * n + cc;
        if (nidx === tipIdx) {
          ck = rr + "," + cc;
          out.cellMeta[ck] = {
            size: mergedSize,
            hasFood: mergedFood,
            topR: mergedTopR,
            topC: mergedTopC
          };
          if (preferredIsMerged) {
            out.preferredCells[ck] = true;
          }
          continue;
        }
        nid = lab[nidx];
        if (!nid) {
          continue;
        }
        root = openIncFind(env, nid);
        m = meta[root];
        if (!m) {
          continue;
        }
        inMerged = !!seen[root];
        ck = rr + "," + cc;
        if (inMerged) {
          out.cellMeta[ck] = {
            size: mergedSize,
            hasFood: mergedFood,
            topR: mergedTopR,
            topC: mergedTopC
          };
          if (preferredIsMerged) {
            out.preferredCells[ck] = true;
          }
        } else {
          out.cellMeta[ck] = {
            size: m.size,
            hasFood: m.hasFood,
            topR: m.topR,
            topC: m.topC
          };
          if (!preferredIsMerged && root === preferredId) {
            out.preferredCells[ck] = true;
          }
        }
      }
    }
    return out;
  }

  function openIncEnsureFresh(env, food) {
    ensureEnvOccupancy(env);
    openIncEnsure(env);
    var foodKey = food ? food[0] + "," + food[1] : "-";
    if (env._openIncDirty || !env._compLab || env._openFoodKey !== foodKey) {
      openIncRebuild(env, food);
    }
  }

  /**
   * Empty components. Default vacates tip (next non-eat space). Pass
   * opts.vacateTip=false to treat the snake as a solid barrier - needed when a
   * chase-tail coil bipartitions the board into equal openings (8x8 seed
   * 427643): tip-vacate makes 25 vs 24, tip-occupied keeps 24 vs 24.
   * When several share max size, preferred opening is the one with food; if
   * still tied, the top-left-most (min r, then min c of the component's
   * top-left cell).
   */
  function planEmptyOpeningsFromOcc(n, occ, food, tipVacateIdx) {
    var out = {
      maxSize: 0,
      tiedMax: false,
      preferredCells: {},
      cellMeta: {}
    };
    occEnsureScratch(n);
    var stamp = occNextStamp();
    var comps = [];
    var r;
    var c;
    var cr;
    var cc0;
    var idx;
    var qi;
    var qe;
    var d;
    var rr;
    var cc;
    var nidx;
    var count;
    var cells;
    var hasFood;
    var topR;
    var topC;
    var maxSize = 0;
    var maxCount = 0;
    var fr = food ? food[0] : -1;
    var fc = food ? food[1] : -1;
    var i;
    var preferred;
    var comp;
    var ck;
    for (r = 0; r < n; r += 1) {
      for (c = 0; c < n; c += 1) {
        idx = r * n + c;
        if ((occ[idx] && idx !== tipVacateIdx) || _seenStamp[idx] === stamp) {
          continue;
        }
        qi = 0;
        qe = 0;
        cells = {};
        _bfsQr[qe] = r;
        _bfsQc[qe] = c;
        qe += 1;
        _seenStamp[idx] = stamp;
        cells[r + "," + c] = true;
        count = 0;
        hasFood = fr === r && fc === c;
        topR = r;
        topC = c;
        while (qi < qe) {
          cr = _bfsQr[qi];
          cc0 = _bfsQc[qi];
          qi += 1;
          count += 1;
          if (cr < topR || (cr === topR && cc0 < topC)) {
            topR = cr;
            topC = cc0;
          }
          if (fr === cr && fc === cc0) {
            hasFood = true;
          }
          for (d = 0; d < 4; d += 1) {
            rr = cr + DIRS[d][0];
            cc = cc0 + DIRS[d][1];
            if (rr < 0 || cc < 0 || rr >= n || cc >= n) {
              continue;
            }
            nidx = rr * n + cc;
            if (_seenStamp[nidx] === stamp) {
              continue;
            }
            if (occ[nidx] && nidx !== tipVacateIdx) {
              continue;
            }
            _seenStamp[nidx] = stamp;
            cells[rr + "," + cc] = true;
            _bfsQr[qe] = rr;
            _bfsQc[qe] = cc;
            qe += 1;
          }
        }
        comps.push({
          size: count,
          cells: cells,
          hasFood: hasFood,
          topR: topR,
          topC: topC
        });
        if (count > maxSize) {
          maxSize = count;
          maxCount = 1;
        } else if (count === maxSize) {
          maxCount += 1;
        }
      }
    }
    out.maxSize = maxSize;
    out.tiedMax = maxCount > 1;
    preferred = null;
    for (i = 0; i < comps.length; i += 1) {
      comp = comps[i];
      for (ck in comp.cells) {
        if (Object.prototype.hasOwnProperty.call(comp.cells, ck)) {
          out.cellMeta[ck] = {
            size: comp.size,
            hasFood: comp.hasFood,
            topR: comp.topR,
            topC: comp.topC
          };
        }
      }
      if (comp.size !== maxSize) {
        continue;
      }
      if (!preferred) {
        preferred = comp;
        continue;
      }
      if (comp.hasFood !== preferred.hasFood) {
        if (comp.hasFood) {
          preferred = comp;
        }
        continue;
      }
      if (comp.topR < preferred.topR ||
          (comp.topR === preferred.topR && comp.topC < preferred.topC)) {
        preferred = comp;
      }
    }
    if (preferred) {
      out.preferredCells = preferred.cells;
    }
    return out;
  }

  function planEmptyOpenings(n, snake, food, opts) {
    opts = opts || {};
    var vacateTip = opts.vacateTip !== false;
    var out = {
      maxSize: 0,
      tiedMax: false,
      preferredCells: {},
      cellMeta: {}
    };
    if (!snake || !snake.length) {
      return out;
    }
    occEnsureScratch(n);
    occFillFromSnake(_occScratch, n, snake);
    var tipIdx = -1;
    if (vacateTip) {
      tipIdx = snake[snake.length - 1][0] * n + snake[snake.length - 1][1];
    }
    return planEmptyOpeningsFromOcc(n, _occScratch, food, tipIdx);
  }

  /** Openings from env occupancy + incremental component labels. */
  function planEmptyOpeningsEnv(env, food, opts) {
    opts = opts || {};
    var vacateTip = opts.vacateTip !== false;
    var cacheKey = vacateTip ? "_openVac" : "_openSolid";
    openIncEnsureFresh(env, food);
    if (env[cacheKey]) {
      return env[cacheKey];
    }
    var packed = vacateTip
      ? openIncPackVacateTip(env, food)
      : openIncPackSolid(env);
    env[cacheKey] = packed;
    return packed;
  }

  /** True if cell is in the preferred largest tip-vacated empty component. */
  function planCellInLargestEmpty(n, snake, cell, food) {
    if (!snake || !snake.length || !cell) {
      return false;
    }
    var openings = planEmptyOpenings(n, snake, food || null);
    var ck = cell[0] + "," + cell[1];
    return !!openings.preferredCells[ck];
  }

  /**
   * Heal variant for Tail Safe -> Hamilton: tip-chase until a neighbor lies in
   * the largest empty opening (openings border body), then enter that opening.
   * Allow freeN=0 when stepping onto the vacated tip. Fallback HealLA.
   */
  function healTailSafeNextAction(env) {
    if (!env || !env.snake || !env.snake.length) {
      return 0;
    }
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var n = env.size;
    var h = env.snake[0];
    var food = env.food;
    var tip = env.snake[env.snake.length - 1];
    var cy = getHamiltonCycle(n);
    var cycleNext = cy.succ[h[0] + "," + h[1]];
    /* Solid barrier for equal-cut preference; tip-vacate for room size. */
    var openPref = planEmptyOpeningsEnv(env, food, { vacateTip: false });
    var openVac = planEmptyOpeningsEnv(env, food, { vacateTip: true });
    var globalMax = openVac.maxSize || planMaxEmptyComponent(n, env.snake);
    var largeNeed = globalMax > 0 ? globalMax - 1 : 0;
    var i;
    var a;
    var nr;
    var nc;
    var eating;
    var reachesTip;
    var flood;
    var freeN;
    var skipAfter;
    var isSucc;
    var manFood;
    var tipMan;
    var need;
    var ontoTip;
    var inLargest;
    var meta;
    var stats;
    var cands = [];
    var cand;
    var bestLarge = null;
    var bestChase = null;
    var bestEscape = null;
    var bestSuccLarge = null;
    var bestSuccAny = null;
    var bestPrefOpen = null;
    var hasChase = false;
    ensureEnvOccupancy(env);

    function rankOpenPref(best, c) {
      if (c.foodInOpen !== best.foodInOpen) {
        return c.foodInOpen ? c : best;
      }
      if (c.openTopR !== best.openTopR) {
        return c.openTopR < best.openTopR ? c : best;
      }
      if (c.openTopC !== best.openTopC) {
        return c.openTopC < best.openTopC ? c : best;
      }
      if (c.nr !== best.nr) {
        return c.nr < best.nr ? c : best;
      }
      if (c.nc !== best.nc) {
        return c.nc < best.nc ? c : best;
      }
      return null;
    }

    function rankHealTs(best, c) {
      if (!best) {
        return c;
      }
      if (c.isSucc !== best.isSucc) {
        return c.isSucc ? c : best;
      }
      if (c.skipAfter !== best.skipAfter) {
        return c.skipAfter < best.skipAfter ? c : best;
      }
      if (c.eating !== best.eating) {
        return c.eating ? c : best;
      }
      if (c.freeN !== best.freeN) {
        return c.freeN > best.freeN ? c : best;
      }
      if (c.flood !== best.flood) {
        return c.flood > best.flood ? c : best;
      }
      /* Equal openings: prefer food side, else top-left (427643). */
      var openPick = rankOpenPref(best, c);
      if (openPick) {
        return openPick;
      }
      if (c.manFood !== best.manFood) {
        return c.manFood < best.manFood ? c : best;
      }
      return c.a < best.a ? c : best;
    }

    /* Tip-chase: prefer onto tip, then closer to tip. */
    function rankChase(best, c) {
      if (!best) {
        return c;
      }
      if (c.ontoTip !== best.ontoTip) {
        return c.ontoTip ? c : best;
      }
      if (c.tipMan !== best.tipMan) {
        return c.tipMan < best.tipMan ? c : best;
      }
      return rankHealTs(best, c);
    }

    for (i = 0; i < safe.length; i += 1) {
      a = safe[i];
      nr = h[0] + DIRS[a][0];
      nc = h[1] + DIRS[a][1];
      eating = !!(food && food[0] === nr && food[1] === nc);
      stats = planCandidateFloodTipEnv(env, nr, nc, eating);
      reachesTip = stats.reachesTip;
      flood = stats.flood;
      need = stats.need;
      freeN = freeNeighborsAfter(env, nr, nc, eating);
      skipAfter = masterSkipAfterMove(cy, env.snake, nr, nc, eating);
      isSucc = !!(cycleNext && cycleNext[0] === nr && cycleNext[1] === nc);
      ontoTip = !!(tip && tip[0] === nr && tip[1] === nc);
      inLargest = !!openVac.preferredCells[nr + "," + nc];
      meta = openPref.cellMeta[nr + "," + nc] || null;
      manFood = food
        ? Math.abs(food[0] - nr) + Math.abs(food[1] - nc)
        : 0;
      tipMan = tip
        ? Math.abs(tip[0] - nr) + Math.abs(tip[1] - nc)
        : 0;
      /* Allow freeN=0 when stepping onto vacated tip (true tip-chase). */
      if (!eating && freeN < 1 && !ontoTip) {
        continue;
      }
      cand = {
        a: a,
        nr: nr,
        nc: nc,
        eating: eating,
        manFood: manFood,
        tipMan: tipMan,
        flood: flood,
        freeN: freeN,
        skipAfter: skipAfter,
        isSucc: isSucc,
        reachesTip: reachesTip,
        need: need,
        ontoTip: ontoTip,
        inLargest: inLargest,
        inPrefOpen: !!openPref.preferredCells[nr + "," + nc],
        foodInOpen: !!(meta && meta.hasFood),
        openTopR: meta ? meta.topR : 1e9,
        openTopC: meta ? meta.topC : 1e9
      };
      cands.push(cand);
      if (reachesTip || ontoTip) {
        hasChase = true;
      }
    }

    for (i = 0; i < cands.length; i += 1) {
      cand = cands[i];
      var entersLarge = cand.inLargest || cand.flood >= largeNeed;
      if (cand.isSucc && (cand.reachesTip || cand.flood >= cand.need || cand.ontoTip)) {
        bestSuccAny = rankHealTs(bestSuccAny, cand);
        if (entersLarge) {
          bestSuccLarge = rankHealTs(bestSuccLarge, cand);
        }
      }
      if (entersLarge && (cand.reachesTip || cand.flood >= cand.need || cand.ontoTip || cand.inLargest)) {
        bestLarge = rankHealTs(bestLarge, cand);
      }
      if (cand.inPrefOpen && (cand.flood >= cand.need || cand.inLargest || cand.ontoTip)) {
        bestPrefOpen = rankHealTs(bestPrefOpen, cand);
      }
      if (cand.reachesTip || cand.ontoTip) {
        bestChase = rankChase(bestChase, cand);
      } else if (cand.flood >= cand.need) {
        bestEscape = rankHealTs(bestEscape, cand);
      }
    }

    /*
     * Equal solid-barrier openings: enter food (else top-left) side before
     * succ/tip-chase livelock (427643).
     */
    if (openPref.tiedMax && bestPrefOpen) {
      return bestPrefOpen.a;
    }
    if (bestSuccLarge) {
      return bestSuccLarge.a;
    }
    if (!hasChase && bestSuccAny) {
      return bestSuccAny.a;
    }
    /* Enter largest opening when a neighbor cell lies in it. */
    if (bestLarge) {
      return bestLarge.a;
    }
    /* Else tip-chase; ignore smaller openings while chase remains. */
    if (bestChase) {
      return bestChase.a;
    }
    if (bestEscape) {
      return bestEscape.a;
    }
    return healLookaheadNextAction(env);
  }

  /**
   * Enter heal: clear pose markers so the first handoff is not a terminal stall.
   * Heal re-tracks poses; a repeat during heal sets _tsHealStall (true livelock).
   */
  function tailSafeEnterHeal(env) {
    env._tsStallPhase = "heal";
    env._tsStallHandoffLen = env.snake.length;
    env._tsStallSeen = Object.create(null);
    env._tsHealStall = false;
  }

  /**
   * Tail Safe with heal->Hamilton (or Cell Tree) handoff. Handoff when pose
   * repeats since last growth, or length >= opts.handoffAtLen / fill %. Pose
   * markers reset on heal entry; a pose repeat during heal sets _tsHealStall.
   * Skip0 leaves heal. opts.afterHeal: "hamilton" (default) or "cell_tree".
   * Heal = HealTS unless opts.healLookahead.
   */
  function tailSafeStallHandoffNextAction(env, opts) {
    opts = opts || {};
    if (!env || !env.snake || !env.snake.length) {
      return 0;
    }
    function healStep() {
      if (opts.healLookahead) {
        return healLookaheadNextAction(env);
      }
      return healTailSafeNextAction(env);
    }
    function noteHealPose() {
      if (env._tsStallApples !== env.apples) {
        env._tsStallApples = env.apples;
        env._tsStallSeen = Object.create(null);
      }
      if (!env._tsStallSeen) {
        env._tsStallSeen = Object.create(null);
      }
      var healKey = tailSafePoseKey(env);
      if (env._tsStallSeen[healKey]) {
        env._tsHealStall = true;
      } else {
        env._tsStallSeen[healKey] = 1;
      }
    }
    function leaveHeal() {
      if (opts.afterHeal === "cell_tree") {
        return tailSafeEnterCellTree(env);
      }
      return tailSafeEnterHamilton(env, opts);
    }
    var phase = env._tsStallPhase || "tail";
    if (phase === "hamilton") {
      return tailSafeEnterHamilton(env, opts);
    }
    if (phase === "cell_tree") {
      return tailSafeEnterCellTree(env);
    }
    if (phase === "heal") {
      var cy = getHamiltonCycle(env.size);
      var skip = masterCycleSkip(cy, env.snake);
      if (skip === 0) {
        return leaveHeal();
      }
      noteHealPose();
      return healStep();
    }
    var handoffAt = null;
    if (typeof opts.handoffAtLen === "number") {
      handoffAt = opts.handoffAtLen;
    } else if (typeof opts.handoffFill === "number") {
      /* Prefer handoffFillPct (0-100) to avoid 0.28*100 float ceil bugs. */
      handoffAt = Math.max(4, Math.ceil(
        (env.size * env.size * opts.handoffFill) - 1e-9
      ));
    } else if (typeof opts.handoffFillPct === "number") {
      handoffAt = Math.max(4, Math.ceil(
        (env.size * env.size * opts.handoffFillPct) / 100
      ));
    }
    if (handoffAt !== null && env.snake.length >= handoffAt) {
      tailSafeEnterHeal(env);
      return healStep();
    }
    if (env._tsStallApples !== env.apples) {
      env._tsStallApples = env.apples;
      env._tsStallSeen = Object.create(null);
    }
    if (!env._tsStallSeen) {
      env._tsStallSeen = Object.create(null);
    }
    var key = tailSafePoseKey(env);
    if (env._tsStallSeen[key]) {
      tailSafeEnterHeal(env);
      return healStep();
    }
    env._tsStallSeen[key] = 1;
    return tailSafeNextAction(env);
  }

  /**
   * Observe Tail Safe Hybrid (prior Master): Tail Safe + stall/handoffFillPct
   * (default 20) -> HealTS -> Hamilton until env.targetLength (Win-at %).
   * Pass opts.noLengthHandoff to remove fill handoff (pose-stall only).
   * Never early-exits at skip0.
   */
  function tailSafeHybridNextAction(env, opts) {
    opts = opts || {};
    var handOpts = { skip0EarlyWin: !!opts.skip0EarlyWin };
    if (opts.noLengthHandoff) {
      /* pose-stall handoff only (= handoff at 100% / remove fill trigger) */
    } else if (typeof opts.handoffFillPct === "number") {
      handOpts.handoffFillPct = opts.handoffFillPct;
    } else {
      handOpts.handoffFillPct = 20;
    }
    return tailSafeStallHandoffNextAction(env, handOpts);
  }

  /**
   * Even-board experiment: Tail Safe + stall/handoffFillPct (default 20) ->
   * HealTS until Hamilton skip0 -> Cell Tree. Odd boards should use TSH instead.
   */
  function tailSafeCellTreeHybridNextAction(env, opts) {
    opts = opts || {};
    var handOpts = {
      skip0EarlyWin: !!opts.skip0EarlyWin,
      afterHeal: "cell_tree"
    };
    if (opts.noLengthHandoff) {
      /* pose-stall only */
    } else if (typeof opts.handoffFillPct === "number") {
      handOpts.handoffFillPct = opts.handoffFillPct;
    } else {
      handOpts.handoffFillPct = 20;
    }
    return tailSafeStallHandoffNextAction(env, handOpts);
  }

  /**
   * Same hybrid for benches/probes: stop at Hamilton skip0. Do not use in
   * Observe. opts.noLengthHandoff / opts.handoffFillPct override fill handoff.
   */
  function tailSafeHybridBenchNextAction(env, opts) {
    opts = opts || {};
    return tailSafeHybridNextAction(env, {
      skip0EarlyWin: true,
      handoffFillPct: opts.handoffFillPct,
      noLengthHandoff: !!opts.noLengthHandoff
    });
  }

  /** Same hybrid; from 10% fill soft wall-stick (20% Manhattan slack). */
  function planAheadHybridWallNextAction(env) {
    return planAheadNextAction(env, {
      handoff: true,
      handoffAtLen: 13,
      wallPack: true,
      wallPackFill: 0.1,
      wallSlack: 1.2
    });
  }

  function snapshot(env) {
    return {
      size: env.size,
      snake: env.snake.map(function (p) { return p.slice(); }),
      facing: env.facing,
      food: env.food ? env.food.slice() : null,
      steps: env.steps,
      apples: env.apples,
      seed: env.seed,
      visionRadius: env.visionRadius,
      compassOn: env.compassOn,
      done: env.done,
      success: env.success,
      targetLength: env.targetLength,
      foodRngState: env.foodRng && env.foodRng.getState ? env.foodRng.getState() : null
    };
  }

  function restore(env, snap) {
    env.setSize(snap.size);
    env.snake = snap.snake.map(function (p) { return p.slice(); });
    env.facing = snap.facing;
    env.food = snap.food ? snap.food.slice() : null;
    env.steps = snap.steps;
    env.apples = snap.apples;
    env.seed = snap.seed;
    env.visionRadius = snap.visionRadius;
    env.compassOn = snap.compassOn;
    env.done = snap.done;
    env.success = snap.success;
    if (snap.foodRngState != null && env.foodRng && env.foodRng.setState) {
      env.foodRng.setState(snap.foodRngState);
    }
    if (typeof snap.targetLength === "number") {
      env.targetLength = snap.targetLength;
      env.maxSteps = maxEpisodeStepsForTarget(env.size, env.targetLength);
    } else {
      env.targetLength = targetLengthForSize(env.size);
      env.maxSteps = maxEpisodeStepsForSize(env.size);
    }
    env._occDirty = true;
    invalidateEnvOpenings(env);
    ensureEnvOccupancy(env);
  }

  global.SNAKE_ENV = {
    DIRS: DIRS,
    ACTION_NAMES: ACTION_NAMES,
    CELL_WALL: CELL_WALL,
    CELL_BODY: CELL_BODY,
    CELL_EMPTY: CELL_EMPTY,
    CELL_FOOD: CELL_FOOD,
    REWARD_FOOD: REWARD_FOOD,
    REWARD_DEATH: REWARD_DEATH,
    REWARD_APPROACH: REWARD_APPROACH,
    makeRng: makeRng,
    targetLengthForSize: targetLengthForSize,
    maxEpisodeStepsForSize: maxEpisodeStepsForSize,
    maxEpisodeStepsForTarget: maxEpisodeStepsForTarget,
    obsSizeForRadius: obsSizeForRadius,
    SnakeEnv: SnakeEnv,
    legalFromObs: legalFromObs,
    obsToStateKey: obsToStateKey,
    countVisited: countVisited,
    bfsNextAction: bfsNextAction,
    greedyNextAction: greedyNextAction,
    wallFollowerNextAction: wallFollowerNextAction,
    pocketOk: pocketOk,
    avoidPocketNextAction: avoidPocketNextAction,
    masterEarlyThreshold: masterEarlyThreshold,
    masterCycleSkip: masterCycleSkip,
    masterSkipAfterMove: masterSkipAfterMove,
    masterHamiltonAction: masterHamiltonAction,
    HAMILTON_AGG_FILL_CAP: HAMILTON_AGG_FILL_CAP,
    hamiltonFillCappedNextAction: hamiltonFillCappedNextAction,
    hamiltonAggrNextAction: hamiltonAggrNextAction,
    hamiltonEarlyNextAction: hamiltonEarlyNextAction,
    hamiltonArcNextAction: hamiltonArcNextAction,
    hamiltonSoftNextAction: hamiltonSoftNextAction,
    masterNextAction: masterNextAction,
    masterLegacyNextAction: masterLegacyNextAction,
    floodFromMove: floodFromMove,
    freeNeighborsAfter: freeNeighborsAfter,
    passesEarlySafety: passesEarlySafety,
    isWallAdjacentCell: isWallAdjacentCell,
    safeBfsNextAction: safeBfsNextAction,
    aggressiveHamiltonAction: aggressiveHamiltonAction,
    planAheadNextAction: planAheadNextAction,
    planAheadHybridNextAction: planAheadHybridNextAction,
    planAheadHybridNoCapNextAction: planAheadHybridNoCapNextAction,
    planAheadReenterNextAction: planAheadReenterNextAction,
    planTailHybridNextAction: planTailHybridNextAction,
    planAheadHybridWallNextAction: planAheadHybridWallNextAction,
    tailSafeNextAction: tailSafeNextAction,
    tailSafeBodyHugNextAction: tailSafeBodyHugNextAction,
    tailSafeBodyHugUntilFillNextAction: tailSafeBodyHugUntilFillNextAction,
    tailSafeStallHandoffNextAction: tailSafeStallHandoffNextAction,
    tailSafeHybridNextAction: tailSafeHybridNextAction,
    tailSafeHybridBenchNextAction: tailSafeHybridBenchNextAction,
    tailSafeCellTreeHybridNextAction: tailSafeCellTreeHybridNextAction,
    MASTER_EVEN_TS_CT_HANDOFF_PCT: MASTER_EVEN_TS_CT_HANDOFF_PCT,
    planHeadReachesTip: planHeadReachesTip,
    planCandidateFloodTipEnv: planCandidateFloodTipEnv,
    leaveTimeAt: leaveTimeAt,
    foodDistAt: foodDistAt,
    planEmptyOpenings: planEmptyOpenings,
    planCellInLargestEmpty: planCellInLargestEmpty,
    planTailSafePoseOk: planTailSafePoseOk,
    healLookaheadNextAction: healLookaheadNextAction,
    healTailSafeNextAction: healTailSafeNextAction,
    healTipSimNextAction: healTipSimNextAction,
    planAheadBuild: planAheadBuild,
    AGG_HANDOFF_FILL: AGG_HANDOFF_FILL,
    createRingBuffer: createRingBuffer,
    captureAggBoardSnap: captureAggBoardSnap,
    formatAggSnapAscii: formatAggSnapAscii,
    avoidanceHintFromSnaps: avoidanceHintFromSnaps,
    getHamiltonCycle: getHamiltonCycle,
    setHamiltonCyclePath: setHamiltonCyclePath,
    clearHamiltonCache: clearHamiltonCache,
    listEvenHamiltonCandidates: listEvenHamiltonCandidates,
    listHamiltonCycleVariants: listHamiltonCycleVariants,
    scoreCycleLocality: scoreCycleLocality,
    scoreNearMiddle: scoreNearMiddle,
    buildEvenHamiltonPath: buildEvenHamiltonPath,
    buildRowZigzagPath: buildRowZigzagPath,
    buildHalfNHalfPath: buildHalfNHalfPath,
    buildQuartersPath: buildQuartersPath,
    buildQuartersLoosePath: buildQuartersLoosePath,
    buildSpiralRingPath: buildSpiralRingPath,
    buildNearMiddlePath: buildNearMiddlePath,
    buildHilbertPath: buildHilbertPath,
    buildHilbertBlockPath: buildHilbertBlockPath,
    buildCellTreeStaticPath: buildCellTreeStaticPath,
    buildStrip2Path: buildStrip2Path,
    buildStrip4Path: buildStrip4Path,
    buildBlock2Path: buildBlock2Path,
    improveCycleLocality: improveCycleLocality,
    snapshot: snapshot,
    restore: restore,
    /* Compat aliases used during port */
    MazeEnv: SnakeEnv,
    generateMazeWithStats: function (opts) {
      opts = opts || {};
      var seed = 1;
      if (opts.rng) {
        seed = (opts.rng() * 0x7fffffff) | 0;
        if (!seed) {
          seed = 1;
        }
      }
      return {
        grid: null,
        stats: { solutionLen: 1 },
        seed: seed
      };
    }
  };
})(typeof window !== "undefined" ? window : global);
