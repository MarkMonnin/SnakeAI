/**
 * Snake AI Shape Search + Train One lab.
 * Ladder: board size SIZE_MIN..SIZE_MAX. Beat = success rate > 90% for 10 consecutive evals.
 */
(function (global) {
  "use strict";

  var APP = global.SNAKE_APP;
  var C = global.SNAKE_CONSTANTS;
  var E = global.SNAKE_ENV;

  var STORAGE_KEY = "learnSnake_shape_v1";
  var IDB_NAME = "SnakeAIShapeSearch";
  var IDB_STORE = "checkpoints";
  var AUTO_SAVE_MS = 30000;
  var EVAL_EPISODES = 20;
  var EVAL_EPISODES_LONG = 100;
  var BEAT_WR = 0.9;
  var DEMOTE_WR = 0.5;
  var BEAT_STREAK_NEED = 10;
  var BEST_WR_DIP = 0.10;
  var NEMESIS_MAX = 24;
  var WARMUP_MS = 100;
  var TRAIN_BETWEEN_EVAL_MS = 50;
  var TRAIN_TO_EVAL_RATIO = 4;
  var MAX_SURVIVORS = 10;
  var MAX_MUTANT_PARENTS = 3;
  var HISTORY_LEARNED_LIMIT = 10;
  var HISTORY_BASELINE_TOP_N = 5;
  var MAX_NEAT_COPIES = 3;
  var CHECKPOINT_FORMAT = "snake-shape-lab-v1";
  var EXPORT_FORMAT = "snake-shape-bot-v1";
  var TOURN_MAZES = 10;
  var NEAR_WIDTH_DUP = 3;
  var MIN_HIDDEN_LAYERS = 0;
  var MAX_HIDDEN_LAYERS = 6;
  var LR_MIN_NN = 0.003;
  var LR_MAX_NN = 0.1;
  var LR_MIN_TF = 0.0001;
  var LR_MAX_TF = 0.01;

  var nextId = 1;
  var neatSerialByFamily = { neat: 0, neat_value: 0 };

  function uid(prefix) {
    var id = prefix + "_" + nextId;
    nextId += 1;
    return id;
  }

  /** Baselines (not survivors, not population). */
  function isBaselineEntry(r) {
    if (!r) {
      return false;
    }
    var f = r.family;
    var id = r.id;
    return f === "random" || id === "random" ||
      f === "bfs" || id === "bfs" ||
      f === "greedy" || id === "greedy" ||
      f === "wall_follower" || id === "wall_follower" ||
      f === "avoid_pocket" || id === "avoid_pocket" ||
      f === "master" || id === "master" ||
      f === "hamilton" || id === "hamilton" ||
      f === "hamilton_aggr" || id === "hamilton_aggr" ||
      f === "cell_tree" || id === "cell_tree" ||
      f === "dhcr" || id === "dhcr" ||
      f === "dhcr_improved" || id === "dhcr_improved" ||
      f === "plan_ahead" || id === "plan_ahead" ||
      f === "tail_safe" || id === "tail_safe" ||
      f === "tail_safe_hybrid" || id === "tail_safe_hybrid";
  }

  function clampVision(r) {
    var v = typeof r === "number" ? r : C.DEFAULT_VISION;
    if (v < C.VISION_MIN) {
      v = C.VISION_MIN;
    }
    if (v > C.VISION_MAX) {
      v = C.VISION_MAX;
    }
    return v | 0;
  }

  function withRadiusLabel(base, radius) {
    return base + " r" + clampVision(radius);
  }

  function wrBeats(wr) {
    return typeof wr === "number" && wr > BEAT_WR;
  }

  function formatSizeTag(size) {
    var s = typeof size === "number" ? size : C.SIZE_MIN;
    if (s < C.SIZE_MIN) {
      s = C.SIZE_MIN;
    }
    if (s > C.SIZE_MAX) {
      s = C.SIZE_MAX;
    }
    return s + "x" + s;
  }

  function isNeatFamily(family) {
    return family === "neat" || family === "neat_value";
  }

  /** Diversity buckets for post-round floors. */
  var CAT_NEURAL = "neural";
  var CAT_TABULAR = "tabular";
  var CAT_EVO = "evo";
  var CATEGORY_FLOOR = {};
  CATEGORY_FLOOR[CAT_NEURAL] = 3;
  CATEGORY_FLOOR[CAT_TABULAR] = 3;
  CATEGORY_FLOOR[CAT_EVO] = 1;
  var CATEGORY_SEED_FAMILIES = {};
  CATEGORY_SEED_FAMILIES[CAT_NEURAL] = [
    "nn_policy", "nn_value", "reinforce_policy", "reinforce_value", "tfjs_policy", "tfjs_value"
  ];
  CATEGORY_SEED_FAMILIES[CAT_TABULAR] = ["qtable", "sarsa", "menace"];
  CATEGORY_SEED_FAMILIES[CAT_EVO] = ["neat", "neat_value", "genetic_menace"];

  function categoryOfFamily(family) {
    if (family === "qtable" || family === "sarsa" || family === "menace") {
      return CAT_TABULAR;
    }
    if (family === "neat" || family === "neat_value" || family === "genetic_menace") {
      return CAT_EVO;
    }
    return CAT_NEURAL;
  }

  function countCategory(pop, cat) {
    var n = 0;
    var i;
    for (i = 0; i < pop.length; i += 1) {
      if (pop[i] && categoryOfFamily(pop[i].family) === cat) {
        n += 1;
      }
    }
    return n;
  }

  function popHasId(pop, id) {
    var i;
    for (i = 0; i < pop.length; i += 1) {
      if (pop[i] && pop[i].id === id) {
        return true;
      }
    }
    return false;
  }

  function findInPop(pop, id) {
    var i;
    for (i = 0; i < pop.length; i += 1) {
      if (pop[i] && pop[i].id === id) {
        return pop[i];
      }
    }
    return null;
  }

  /** Same keep path as tournament survivors (shape + r; hard mode restores weights). */
  function keepIndividualInto(src, nextPop) {
    if (!src || !nextPop) {
      return null;
    }
    var kept = createIndividual({
      id: src.id,
      family: src.family,
      layerSizes: cloneSizes(src.layerSizes),
      learningRate: src.learningRate,
      visionRadius: src.visionRadius,
      label: makeUniqueLabel(withRadiusLabel(shapeLabelFor(src), src.visionRadius), nextPop),
      neatSerial: src.neatSerial
    });
    if (isHardModeLab()) {
      restoreAgent(kept, agentSnapshot(src));
      kept.bestWinRate = src.bestWinRate;
      kept.bestAgentDict = src.bestAgentDict;
    }
    nextPop.push(kept);
    return kept;
  }

  function seedFreshCategoryBot(cat, visionMode, nextPop, seedSlot) {
    var seeds = CATEGORY_SEED_FAMILIES[cat] || [];
    if (!seeds.length) {
      return null;
    }
    var vr = (visionMode !== "adaptive" && visionMode !== "Adaptive") ?
      clampVision(visionMode) : C.DEFAULT_VISION;
    var fam = seeds[(seedSlot || 0) % seeds.length];
    var sizes = null;
    if (APP.defaultLayerSizes) {
      sizes = APP.defaultLayerSizes(fam, vr, APP.isCompassOn ? APP.isCompassOn() : true);
    }
    var ind = createIndividual({
      family: fam,
      layerSizes: sizes,
      learningRate: APP.defaultLearningRate(fam),
      visionRadius: vr,
      label: makeUniqueLabel(withRadiusLabel(familyLabel(fam, sizes), vr), nextPop)
    });
    nextPop.push(ind);
    return ind;
  }

  /**
   * Top up under-filled categories by carrying the best tournament bots
   * from that category (same r / identity). Fresh seed only when still short.
   */
  function ensureCategoryFloors(nextPop, visionMode, learnedRanked, oldPop) {
    if (!nextPop) {
      return;
    }
    var ranked = learnedRanked || [];
    var prev = oldPop || [];
    var cats = [CAT_NEURAL, CAT_TABULAR, CAT_EVO];
    var c;
    for (c = 0; c < cats.length; c += 1) {
      var cat = cats[c];
      var need = CATEGORY_FLOOR[cat] || 0;
      var ri;
      for (ri = 0; ri < ranked.length && countCategory(nextPop, cat) < need; ri += 1) {
        var row = ranked[ri];
        if (!row || isBaselineEntry(row)) {
          continue;
        }
        if (categoryOfFamily(row.family) !== cat) {
          continue;
        }
        if (popHasId(nextPop, row.id)) {
          continue;
        }
        var src = findInPop(prev, row.id);
        if (!src) {
          continue;
        }
        keepIndividualInto(src, nextPop);
      }
      var seedSlot = 0;
      while (countCategory(nextPop, cat) < need && seedSlot < need + 4) {
        seedFreshCategoryBot(cat, visionMode, nextPop, seedSlot);
        seedSlot += 1;
      }
    }
  }

  function familyLabel(family, sizes) {
    var base = {
      nn_policy: "Backprop policy",
      nn_value: "Backprop value",
      reinforce_policy: "REINFORCE policy",
      reinforce_value: "REINFORCE value",
      tfjs_policy: "TF.js DQN",
      tfjs_value: "TF.js value",
      qtable: "Q-Learning",
      sarsa: "SARSA",
      menace: "MENACE",
      neat: "NEAT",
      neat_value: "NEAT value",
      genetic_menace: "Genetic MENACE"
    }[family] || family;
    if (!sizes) {
      return base;
    }
    var hidden = sizes.slice(1, -1).join("/");
    return base + " [" + hidden + "]";
  }

  /** If base is taken, append " 2", " 3", ... (e.g. "SARSA 2"). */
  function makeUniqueLabel(base, existing) {
    var used = {};
    var i;
    for (i = 0; i < existing.length; i += 1) {
      if (existing[i] && existing[i].label) {
        used[existing[i].label] = true;
      }
    }
    if (!used[base]) {
      return base;
    }
    var n = 2;
    while (used[base + " " + n]) {
      n += 1;
    }
    return base + " " + n;
  }

  function shapeLabelFor(ind) {
    if (!ind) {
      return "bot";
    }
    if (isNeatFamily(ind.family) && typeof ind.neatSerial === "number") {
      return (ind.family === "neat_value" ? "NEAT value " : "NEAT ") + ind.neatSerial;
    }
    return familyLabel(ind.family, ind.layerSizes);
  }

  function assignNeatIdentity(ind) {
    if (!ind || !isNeatFamily(ind.family)) {
      return ind;
    }
    if (typeof ind.neatSerial !== "number") {
      neatSerialByFamily[ind.family] = (neatSerialByFamily[ind.family] || 0) + 1;
      ind.neatSerial = neatSerialByFamily[ind.family];
    }
    var generated = (ind.family === "neat_value" ? "NEAT value " : "NEAT ") + ind.neatSerial;
    if (!ind.label || /^NEAT( value)? \d+$/.test(ind.label) ||
        ind.label === familyLabel(ind.family, ind.layerSizes)) {
      ind.label = generated;
    }
    return ind;
  }

  function cloneSizes(sizes) {
    return sizes ? sizes.slice() : null;
  }

  function layersNearDuplicate(a, b) {
    if (!a && !b) {
      return true;
    }
    if (!a || !b || a.length !== b.length) {
      return false;
    }
    if (a[0] !== b[0] || a[a.length - 1] !== b[b.length - 1]) {
      return false;
    }
    var i;
    for (i = 1; i < a.length - 1; i += 1) {
      if (Math.abs(a[i] - b[i]) > NEAR_WIDTH_DUP) {
        return false;
      }
    }
    return true;
  }

  /** True if nextPop already has same family with near-duplicate layer sizes. */
  function shapeTakenInPop(family, sizes, nextPop) {
    if (!sizes || !nextPop) {
      return false;
    }
    var i;
    for (i = 0; i < nextPop.length; i += 1) {
      var ind = nextPop[i];
      if (!ind || ind.family !== family) {
        continue;
      }
      if (layersNearDuplicate(sizes, ind.layerSizes)) {
        return true;
      }
    }
    return false;
  }

  /** Morph until not near-duplicate, or null if no distinct shape found. */
  function morphSizesDistinct(parentSizes, family, nextPop) {
    var attempt;
    for (attempt = 0; attempt < 12; attempt += 1) {
      var cand = morphSizes(parentSizes);
      if (!shapeTakenInPop(family, cand, nextPop)) {
        return cand;
      }
    }
    return null;
  }

  /**
   * Change hidden-layer count by +1 or -1.
   * Remove last hidden, or insert a new hidden before the output
   * (width = avg of neighbors, min 8).
   */
  function morphDepth(sizes, delta) {
    if (!sizes || sizes.length < 2) {
      return null;
    }
    var hidden = sizes.length - 2;
    var out;
    var w;
    if (delta < 0) {
      if (hidden <= MIN_HIDDEN_LAYERS) {
        return null;
      }
      out = sizes.slice();
      out.splice(out.length - 2, 1);
      return out;
    }
    if (delta > 0) {
      if (hidden >= MAX_HIDDEN_LAYERS) {
        return null;
      }
      out = sizes.slice();
      w = Math.max(8, Math.round((out[out.length - 2] + out[out.length - 1]) / 2));
      out.splice(out.length - 1, 0, w);
      return out;
    }
    return null;
  }

  function pushShapeChild(parent, sizes, nextPop, visionRadius) {
    if (!parent || !sizes || !nextPop) {
      return false;
    }
    var vr = clampVision(typeof visionRadius === "number" ? visionRadius : parent.visionRadius);
    var adj = sizes.slice();
    adj[0] = C.obsSizeForRadius(vr, APP.isCompassOn ? APP.isCompassOn() : true);
    if (shapeTakenInPop(parent.family, adj, nextPop)) {
      return false;
    }
    nextPop.push(createIndividual({
      family: parent.family,
      layerSizes: adj,
      learningRate: morphLr(parent.family, parent.learningRate),
      visionRadius: vr,
      label: makeUniqueLabel(withRadiusLabel(familyLabel(parent.family, adj), vr), nextPop)
    }));
    return true;
  }

  function ensureEvalMeta(ind) {
    if (!ind.peerPlayCounts) {
      ind.peerPlayCounts = {};
    }
    if (typeof ind.beatStreak !== "number") {
      ind.beatStreak = 0;
    }
    if (typeof ind.evalGamesTarget !== "number") {
      ind.evalGamesTarget = EVAL_EPISODES;
    }
    if (typeof ind.trainSinceEvalMs !== "number") {
      ind.trainSinceEvalMs = 0;
    }
    if (typeof ind.nextTrainTargetMs !== "number") {
      ind.nextTrainTargetMs = TRAIN_BETWEEN_EVAL_MS;
    }
    if (typeof ind.lastEvalMs !== "number") {
      ind.lastEvalMs = 0;
    }
    if (!ind.nemesisSeeds) {
      ind.nemesisSeeds = [];
    }
    if (typeof ind.mazePoolAfterId !== "number") {
      ind.mazePoolAfterId = 0;
    }
    if (typeof ind.warmupDone === "undefined") {
      ind.warmupDone = false;
    }
  }

  function createIndividual(spec) {
    var family = spec.family;
    var vr = clampVision(spec.visionRadius);
    var sizes = cloneSizes(spec.layerSizes);
    if (sizes) {
      if (APP.applyHardOutputSize) {
        sizes = APP.applyHardOutputSize(sizes, family);
      }
      sizes[0] = C.obsSizeForRadius(vr, APP.isCompassOn ? APP.isCompassOn() : true);
    } else if (APP.defaultLayerSizes) {
      sizes = APP.defaultLayerSizes(family, vr, APP.isCompassOn ? APP.isCompassOn() : true);
    }
    var lr = spec.learningRate;
    var agent = APP.createShapeAgent(family, sizes, lr, vr);
    var baseLabel = spec.label || familyLabel(family, sizes);
    if (baseLabel.indexOf(" r") < 0) {
      baseLabel = withRadiusLabel(baseLabel, vr);
    }
    var ind = {
      id: spec.id || uid("bot"),
      family: family,
      layerSizes: sizes,
      visionRadius: vr,
      learningRate: typeof lr === "number" ? lr : APP.defaultLearningRate(family),
      label: baseLabel,
      agent: agent,
      secondsUsed: 0,
      evalSeconds: 0,
      beaten: false,
      beatStreak: 0,
      secondsToBeat: null,
      lastEvalWinRate: null,
      lastEvalWins: null,
      lastEvalDraws: null,
      lastEvalGames: null,
      bestWinRate: null,
      bestAgentDict: null,
      peerPlayCounts: {},
      nemesisSeeds: [],
      mazePoolAfterId: 0,
      evalGamesTarget: EVAL_EPISODES,
      trainSinceEvalMs: 0,
      nextTrainTargetMs: TRAIN_BETWEEN_EVAL_MS,
      lastEvalMs: 0,
      warmupDone: false,
      neatSerial: spec.neatSerial
    };
    return assignNeatIdentity(ind);
  }

  function resetRoundMarks(ind) {
    ensureEvalMeta(ind);
    ind.beaten = false;
    ind.beatStreak = 0;
    ind.secondsToBeat = null;
    ind.lastEvalWinRate = null;
    ind.lastEvalWins = null;
    ind.lastEvalDraws = null;
    ind.lastEvalGames = null;
    ind.evalGamesTarget = EVAL_EPISODES;
    ind.trainSinceEvalMs = 0;
    ind.nextTrainTargetMs = TRAIN_BETWEEN_EVAL_MS;
    ind.lastEvalMs = 0;
    ind.warmupDone = false;
    ind.secondsUsed = 0;
    ind.evalSeconds = 0;
    ind.nemesisSeeds = [];
    ind.mazePoolAfterId = 0;
    ind.peerPlayCounts = {};
  }

  function agentSnapshot(ind) {
    return APP.shapeAgentToDict(ind.agent, ind.family);
  }

  function restoreAgent(ind, dict) {
    ind.agent = APP.shapeAgentFromDict(
      ind.family, dict, ind.layerSizes, ind.learningRate, ind.visionRadius
    );
  }

  function gen0Specs(visionRadius) {
    var vr = clampVision(visionRadius);
    var specs = [];
    var widths = [32, 64, 96];
    var morphFamilies = [
      "nn_policy", "nn_value", "reinforce_policy", "reinforce_value",
      "tfjs_policy", "tfjs_value"
    ];
    var f;
    var w;
    for (f = 0; f < morphFamilies.length; f += 1) {
      for (w = 0; w < widths.length; w += 1) {
        var fam = morphFamilies[f];
        var sizes;
        if (fam === "tfjs_policy" && C.TFJS_DQN_LAYER_SIZES) {
          sizes = C.TFJS_DQN_LAYER_SIZES[widths[w]].slice();
        } else if (fam === "tfjs_value" && C.TFJS_VALUE_LAYER_SIZES) {
          sizes = C.TFJS_VALUE_LAYER_SIZES[widths[w]].slice();
        } else {
          var base = APP.defaultLayerSizes(fam, vr, APP.isCompassOn ? APP.isCompassOn() : true);
          sizes = base ? base.slice() : null;
          if (sizes && sizes.length >= 3) {
            sizes[1] = widths[w];
            if (sizes.length > 3) {
              sizes[sizes.length - 2] = Math.max(16, widths[w] - 8);
            }
          }
        }
        if (sizes) {
          sizes[0] = C.obsSizeForRadius(vr, APP.isCompassOn ? APP.isCompassOn() : true);
        }
        specs.push({
          family: fam,
          layerSizes: sizes,
          learningRate: APP.defaultLearningRate(fam),
          visionRadius: vr
        });
      }
    }
    specs.push({ family: "qtable", visionRadius: vr });
    specs.push({ family: "sarsa", visionRadius: vr });
    specs.push({ family: "menace", visionRadius: vr });
    specs.push({ family: "genetic_menace", visionRadius: vr });
    specs.push({ family: "neat", visionRadius: vr });
    specs.push({ family: "neat", visionRadius: vr });
    specs.push({ family: "neat", visionRadius: vr });
    specs.push({ family: "neat_value", visionRadius: vr });
    specs.push({ family: "neat_value", visionRadius: vr });
    specs.push({ family: "neat_value", visionRadius: vr });
    return specs;
  }

  function morphSizes(sizes) {
    if (!sizes) {
      return null;
    }
    var out = sizes.slice();
    var i;
    for (i = 1; i < out.length - 1; i += 1) {
      out[i] = Math.max(8, out[i] + Math.floor(Math.random() * 17) - 8);
    }
    return out;
  }

  function morphLr(family, lr) {
    var base = typeof lr === "number" ? lr : APP.defaultLearningRate(family) || 0.03;
    var factor = 0.7 + Math.random() * 0.8;
    var next = base * factor;
    if (family === "tfjs_policy" || family === "tfjs_value") {
      return Math.min(LR_MAX_TF, Math.max(LR_MIN_TF, next));
    }
    return Math.min(LR_MAX_NN, Math.max(LR_MIN_NN, next));
  }

  /** Mutate copied weights / beads so tabular and NN parents both leave offspring. */
  function mutateCopiedAgent(agent, family) {
    if (!agent) {
      return;
    }
    if (agent.net && typeof agent.net.mutate === "function") {
      agent.net.mutate(0.08);
      return;
    }
    if (agent.fallback && agent.fallback.net && agent.fallback.net.mutate) {
      agent.fallback.net.mutate(0.08);
      return;
    }
    if (agent.inner && agent.inner.net && agent.inner.net.mutate) {
      agent.inner.net.mutate(0.08);
      return;
    }
    if (agent.best && agent.best.boxes) {
      agent = agent.best;
    }
    var map = agent.q || agent.boxes;
    if (map && typeof map.forEach === "function") {
      map.forEach(function (row) {
        var i;
        for (i = 0; i < row.length; i += 1) {
          row[i] += (Math.random() * 2 - 1) * (family === "menace" || family === "genetic_menace" ? 0.5 : 0.05);
          if (family === "menace" || family === "genetic_menace") {
            row[i] = Math.max(1, row[i]);
          }
        }
      });
    }
  }

  function isHardModeLab() {
    return !!(APP.isHardMode ? APP.isHardMode() : APP.hardMode);
  }

  function spawnOffspringFromParent(parent, nextPop, visionMode) {
    if (!parent) {
      return;
    }
    var keepWeights = isHardModeLab();
    var parentVr = clampVision(parent.visionRadius);
    var childVisions = [parentVr];
    if (visionMode === "adaptive") {
      if (parentVr > C.VISION_MIN) {
        childVisions.push(parentVr - 1);
      }
      if (parentVr < C.VISION_MAX) {
        childVisions.push(parentVr + 1);
      }
    }
    if (isNeatFamily(parent.family)) {
      var c;
      for (c = 0; c < MAX_NEAT_COPIES - 1; c += 1) {
        var vrN = childVisions[c % childVisions.length];
        var clone = createIndividual({
          family: parent.family,
          layerSizes: null,
          learningRate: parent.learningRate,
          visionRadius: vrN
        });
        if (keepWeights) {
          restoreAgent(clone, agentSnapshot(parent));
          mutateCopiedAgent(clone.agent, parent.family);
        }
        nextPop.push(clone);
      }
      return;
    }
    if (parent.layerSizes) {
      var vi;
      for (vi = 0; vi < childVisions.length; vi += 1) {
        var vr = childVisions[vi];
        var morphSz = morphSizesDistinct(parent.layerSizes, parent.family, nextPop);
        if (morphSz) {
          pushShapeChild(parent, morphSz, nextPop, vr);
        }
        pushShapeChild(parent, morphDepth(parent.layerSizes, -1), nextPop, vr);
        pushShapeChild(parent, morphDepth(parent.layerSizes, 1), nextPop, vr);
      }
      return;
    }
    /* Tabular / genetic: survivor keeps parent r; Adaptive adds fresh r+/-1 copies. */
    if (visionMode === "adaptive") {
      var tv;
      for (tv = 0; tv < childVisions.length; tv += 1) {
        var tvr = childVisions[tv];
        if (tvr === parentVr) {
          continue;
        }
        nextPop.push(createIndividual({
          family: parent.family,
          layerSizes: null,
          learningRate: parent.learningRate,
          visionRadius: tvr
        }));
      }
    }
  }

  function noteBeatStreak(ind, wr) {
    ensureEvalMeta(ind);
    if (wrBeats(wr)) {
      ind.beatStreak = Math.min(BEAT_STREAK_NEED, ind.beatStreak + 1);
    } else {
      ind.beatStreak = 0;
    }
    return ind.beatStreak;
  }

  /**
   * Save peak weights on new best WR. On a long-eval dip vs bestWinRate,
   * re-run the saved best on the same seeds and restore only if it wins more.
   */
  function applyKeepBest(ind, result, mazeSize) {
    ensureEvalMeta(ind);
    if (!result) {
      return;
    }
    var wr = result.winRate;
    var games = result.games;
    if (ind.bestWinRate === null || wr > ind.bestWinRate + 1e-9) {
      ind.bestWinRate = wr;
      ind.bestAgentDict = agentSnapshot(ind);
      return;
    }
    if (games < EVAL_EPISODES_LONG || !ind.bestAgentDict) {
      return;
    }
    if (!(wr + 1e-9 < ind.bestWinRate - BEST_WR_DIP)) {
      return;
    }
    if (!result.seeds || result.seeds.length < games) {
      return;
    }
    var bestAgent = APP.shapeAgentFromDict(
      ind.family,
      ind.bestAgentDict,
      ind.layerSizes,
      ind.learningRate,
      ind.visionRadius
    );
    var bestEval = APP.shapeEvalSuccess(bestAgent, ind.family, mazeSize, games, {
      seeds: result.seeds
    });
    if (bestEval.wins > result.wins) {
      restoreAgent(ind, ind.bestAgentDict);
    }
  }

  /* --- Shape Search --- */
  function ShapeSearch() {
    this.round = 1;
    this.mazeSize = C.SIZE_MIN;
    this.population = [];
    this.paused = false;
    this.userPaused = false;
    this.teachEachOther = false;
    this.learnFromMaster = false;
    /* "adaptive" or integer 1..10 */
    this.visionMode = "adaptive";
    this._teachPoolAfterId = 0;
    this._teachAlternate = false;
    this.bestBeatSec = null;
    this.lastTournament = null;
    this.tournamentSnapshots = [];
    this.history = [];
    this.statusLine = "Shape Search ready";
    this._currentId = null;
    this._sliceTarget = null;
    this._leaderAdvanceTarget = null;
    this.liveTourn = null;
    this.initPopulation();
  }

  ShapeSearch.prototype.effectiveVision = function () {
    if (this.visionMode === "adaptive" || this.visionMode === "Adaptive") {
      return C.DEFAULT_VISION;
    }
    return clampVision(this.visionMode);
  };

  ShapeSearch.prototype.setVisionMode = function (mode) {
    if (mode === "adaptive" || mode === "Adaptive") {
      this.visionMode = "adaptive";
    } else {
      this.visionMode = clampVision(parseInt(mode, 10));
    }
  };

  ShapeSearch.prototype.initPopulation = function () {
    var vr = this.visionMode === "adaptive" ? C.DEFAULT_VISION : this.effectiveVision();
    var specs = gen0Specs(vr);
    var i;
    this.population = [];
    for (i = 0; i < specs.length; i += 1) {
      this.population.push(createIndividual(specs[i]));
    }
  };

  ShapeSearch.prototype.setPaused = function (paused, fromUser) {
    this.paused = !!paused;
    if (fromUser) {
      this.userPaused = !!paused;
    }
  };

  ShapeSearch.prototype.resumeIfAllowed = function () {
    if (!this.userPaused) {
      this.paused = false;
    }
  };

  ShapeSearch.prototype.minTrainSec = function () {
    return 0;
  };

  ShapeSearch.prototype.maxTrainSec = function () {
    return null;
  };

  ShapeSearch.prototype.hasBeater = function () {
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      if (this.population[i].beaten) {
        return true;
      }
    }
    return false;
  };

  /** Once any bot has a 10-beat streak, stop training the rest and advance. */
  ShapeSearch.prototype.botTrainComplete = function (ind) {
    return !ind || this.hasBeater();
  };

  ShapeSearch.prototype.recomputeBestBeatSec = function () {
    var best = null;
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      var ind = this.population[i];
      if (typeof ind.secondsToBeat === "number") {
        if (best === null || ind.secondsToBeat < best) {
          best = ind.secondsToBeat;
        }
      }
    }
    this.bestBeatSec = best;
    return best;
  };

  ShapeSearch.prototype.noteToBeatSec = function (ind) {
    if (!ind || ind.secondsToBeat !== null) {
      return;
    }
    ind.secondsToBeat = ind.secondsUsed;
    this.recomputeBestBeatSec();
  };

  ShapeSearch.prototype.pickPeerFor = function (ind) {
    ensureEvalMeta(ind);
    if (!wrBeats(ind.lastEvalWinRate)) {
      return null;
    }
    var peers = [];
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      var p = this.population[i];
      if (p.id === ind.id) {
        continue;
      }
      ensureEvalMeta(p);
      if (wrBeats(p.lastEvalWinRate)) {
        peers.push(p);
      }
    }
    if (!peers.length) {
      return null;
    }
    peers.sort(function (a, b) {
      var ca = ind.peerPlayCounts[a.id] || 0;
      var cb = ind.peerPlayCounts[b.id] || 0;
      if (ca !== cb) {
        return ca - cb;
      }
      return a.secondsUsed - b.secondsUsed;
    });
    return peers[0];
  };

  ShapeSearch.prototype.notePeerPlay = function (a, b) {
    ensureEvalMeta(a);
    ensureEvalMeta(b);
    a.peerPlayCounts[b.id] = (a.peerPlayCounts[b.id] || 0) + 1;
    b.peerPlayCounts[a.id] = (b.peerPlayCounts[a.id] || 0) + 1;
  };

  ShapeSearch.prototype.botDueForEval = function (ind) {
    ensureEvalMeta(ind);
    if (!ind.warmupDone) {
      return false;
    }
    return ind.trainSinceEvalMs >= ind.nextTrainTargetMs;
  };

  ShapeSearch.prototype.roundFillComplete = function () {
    return this.hasBeater();
  };

  ShapeSearch.prototype.trainIndividual = function (ind, budgetMs) {
    ensureEvalMeta(ind);
    var size = this.mazeSize;
    if (!ind.warmupDone) {
      /* Slices are ~40ms; require accumulated warmup, not one giant slice. */
      var warm = Math.min(budgetMs, WARMUP_MS);
      var warmRes = APP.shapeTrainEpisodes(ind.agent, ind.family, size, warm, {
        explore: true,
        nemesisSeeds: [],
        poolAfterId: ind.mazePoolAfterId
      });
      ind.mazePoolAfterId = warmRes.poolAfterId;
      ind.secondsUsed += warm / 1000;
      ind.trainSinceEvalMs += warm;
      if (ind.trainSinceEvalMs >= WARMUP_MS) {
        ind.warmupDone = true;
      }
      return;
    }
    if (ind.family === "neat" || ind.family === "neat_value" || ind.family === "genetic_menace") {
      var evoRes = APP.shapeEvolveOne(ind.agent, ind.family, size, null, {
        poolAfterId: ind.mazePoolAfterId
      });
      ind.mazePoolAfterId = evoRes.poolAfterId;
      ind.secondsUsed += budgetMs / 1000;
      ind.trainSinceEvalMs += budgetMs;
      return;
    }
    /* No peer/self-play: maze bots only train on mazes when selected. */
    var trainRes = APP.shapeTrainEpisodes(ind.agent, ind.family, size, budgetMs, {
      explore: true,
      nemesisSeeds: ind.nemesisSeeds,
      poolAfterId: ind.mazePoolAfterId
    });
    ind.mazePoolAfterId = trainRes.poolAfterId;
    ind.secondsUsed += budgetMs / 1000;
    ind.trainSinceEvalMs += budgetMs;
  };

  ShapeSearch.prototype.runBotEval = function (ind) {
    ensureEvalMeta(ind);
    var n = ind.evalGamesTarget || EVAL_EPISODES;
    var result = APP.shapeEvalSuccess(ind.agent, ind.family, this.mazeSize, n);
    ind.lastEvalWins = result.wins;
    ind.lastEvalDraws = 0;
    ind.lastEvalGames = result.games;
    ind.lastEvalWinRate = result.winRate;
    ind.lastEvalMs = result.ms;
    ind.evalSeconds += result.ms / 1000;
    ind.trainSinceEvalMs = 0;
    var want = Math.max(TRAIN_BETWEEN_EVAL_MS, result.ms * TRAIN_TO_EVAL_RATIO);
    var doubled = Math.max(TRAIN_BETWEEN_EVAL_MS, (ind.nextTrainTargetMs || TRAIN_BETWEEN_EVAL_MS) * 2);
    ind.nextTrainTargetMs = Math.min(want, doubled);
    if (ind.nextTrainTargetMs < want && doubled >= want) {
      ind.nextTrainTargetMs = want;
    }
    if (result.failedSeeds && result.failedSeeds.length) {
      var i;
      for (i = 0; i < result.failedSeeds.length && ind.nemesisSeeds.length < NEMESIS_MAX; i += 1) {
        if (ind.nemesisSeeds.indexOf(result.failedSeeds[i]) < 0) {
          ind.nemesisSeeds.push(result.failedSeeds[i]);
        }
      }
    }
    if (n === EVAL_EPISODES && wrBeats(result.winRate)) {
      /* Extend grade to 100 on first clear. */
      var extra = APP.shapeEvalSuccess(ind.agent, ind.family, this.mazeSize, EVAL_EPISODES_LONG - EVAL_EPISODES, {
        seedBase: 424242
      });
      result.wins += extra.wins;
      result.games = EVAL_EPISODES_LONG;
      result.winRate = result.wins / result.games;
      result.seeds = (result.seeds || []).concat(extra.seeds || []);
      if (extra.failedSeeds && extra.failedSeeds.length) {
        result.failedSeeds = (result.failedSeeds || []).concat(extra.failedSeeds);
      }
      ind.lastEvalWins = result.wins;
      ind.lastEvalGames = result.games;
      ind.lastEvalWinRate = result.winRate;
      ind.evalGamesTarget = EVAL_EPISODES_LONG;
      ind.evalSeconds += extra.ms / 1000;
    }
    applyKeepBest(ind, result, this.mazeSize);
    noteBeatStreak(ind, result.winRate);
    if (ind.beatStreak >= BEAT_STREAK_NEED) {
      ind.beaten = true;
    }
    if (!wrBeats(result.winRate)) {
      ind.beaten = false;
    }
    if (result.games >= EVAL_EPISODES_LONG && wrBeats(result.winRate)) {
      this.noteToBeatSec(ind);
    }
    return result;
  };

  ShapeSearch.prototype.pickTrainTarget = function () {
    var i;
    var needing = [];
    for (i = 0; i < this.population.length; i += 1) {
      var ind = this.population[i];
      if (this.botDueForEval(ind)) {
        return { kind: "eval", ind: ind };
      }
      if (!this.botTrainComplete(ind)) {
        needing.push(ind);
      }
    }
    if (!needing.length) {
      return null;
    }
    needing.sort(function (a, b) {
      return a.secondsUsed - b.secondsUsed;
    });
    if (this.teachEachOther) {
      this._teachAlternate = !this._teachAlternate;
      if (this._teachAlternate) {
        return { kind: "teach", ind: needing[0] };
      }
    }
    return { kind: "train", ind: needing[0] };
  };

  /** Tournament baselines; optional visionRadius for peer-teach races. */
  function makeBaselineMasters(visionRadius) {
    var vr = typeof visionRadius === "number" ? visionRadius : undefined;
    var list = [
      {
        id: "master",
        family: "master",
        label: "Master",
        agent: APP.makeMasterAgent()
      },
      {
        id: "hamilton",
        family: "hamilton",
        label: "Hamilton",
        agent: APP.makeHamiltonAgent()
      },
      {
        id: "hamilton_aggr",
        family: "hamilton_aggr",
        label: "Hamilton Aggressive",
        agent: APP.makeHamiltonAggrAgent()
      },
      {
        id: "cell_tree",
        family: "cell_tree",
        label: "Cell Tree",
        agent: APP.makeCellTreeAgent()
      },
      {
        id: "dhcr",
        family: "dhcr",
        label: "DHCR",
        agent: APP.makeDhcrAgent()
      },
      {
        id: "dhcr_improved",
        family: "dhcr_improved",
        label: "DHCR Improved",
        agent: APP.makeDhcrImprovedAgent()
      },
      {
        id: "tail_safe",
        family: "tail_safe",
        label: "Tail Safe",
        agent: APP.makeTailSafeAgent()
      },
      {
        id: "tail_safe_hybrid",
        family: "tail_safe_hybrid",
        label: "Tail Safe Hybrid",
        agent: APP.makeTailSafeHybridAgent()
      },
      {
        id: "plan_ahead",
        family: "plan_ahead",
        label: "Plan Ahead",
        agent: APP.makePlanAheadAgent()
      },
      {
        id: "random",
        family: "random",
        label: "Random",
        agent: APP.makeRandomAgent()
      },
      {
        id: "greedy",
        family: "greedy",
        label: "Greedy",
        agent: APP.makeGreedyAgent()
      },
      {
        id: "wall_follower",
        family: "wall_follower",
        label: "Wall follower",
        agent: APP.makeWallFollowerAgent()
      },
      {
        id: "bfs",
        family: "bfs",
        label: "BFS",
        agent: APP.makeBfsAgent()
      },
      {
        id: "avoid_pocket",
        family: "avoid_pocket",
        label: "Avoid Pocket",
        agent: APP.makeAvoidPocketAgent()
      }
    ];
    var i;
    if (typeof vr === "number") {
      for (i = 0; i < list.length; i += 1) {
        list[i].visionRadius = vr;
      }
    }
    return list;
  }

  ShapeSearch.prototype.runPeerTeach = function (budgetMs) {
    if (!APP.shapePeerTeachPass || !this.population.length) {
      return;
    }
    var slice = Math.min(budgetMs || 120, 200);
    var result = APP.shapePeerTeachPass(this.population, this.mazeSize, {
      poolAfterId: this._teachPoolAfterId,
      learnFromMaster: !!this.learnFromMaster,
      makeMasters: function (teachVr) {
        return makeBaselineMasters(teachVr);
      }
    });
    this._teachPoolAfterId = result.poolAfterId;
    var n = this.population.length;
    var shareSec = (slice / 1000) / n;
    var shareMs = slice / n;
    var i;
    for (i = 0; i < n; i += 1) {
      var ind = this.population[i];
      ensureEvalMeta(ind);
      ind.secondsUsed += shareSec;
      ind.trainSinceEvalMs += shareMs;
    }
    if (result.teacherId) {
      this._currentId = result.teacherId;
      this.statusLine = "Teach r" + result.visionRadius + ": " + result.teacherId +
        " taught " + result.students + " on " + result.steps + " steps";
    }
  };

  ShapeSearch.prototype.buildTournamentEntries = function () {
    var entries = this.population.map(function (ind) {
      return {
        id: ind.id,
        family: ind.family,
        label: ind.label,
        agent: ind.agent,
        visionRadius: ind.visionRadius
      };
    });
    return entries.concat(makeBaselineMasters());
  };

  function sameTournamentPlace(a, b) {
    if (!a || !b) {
      return false;
    }
    if (a.successes !== b.successes) {
      return false;
    }
    var ae = a.avgEffSteps;
    var be = b.avgEffSteps;
    var aHas = typeof ae === "number";
    var bHas = typeof be === "number";
    if (aHas !== bHas) {
      return false;
    }
    if (aHas && Math.abs(ae - be) >= 1e-9) {
      return false;
    }
    return (a.apples || 0) === (b.apples || 0);
  }

  /**
   * Keep every bot tied for first. If fewer than MAX_SURVIVORS, fill next ranks.
   */
  function selectTournamentSurvivors(learned) {
    if (!learned || !learned.length) {
      return [];
    }
    var first = learned[0];
    var tied = [];
    var i;
    for (i = 0; i < learned.length; i += 1) {
      if (sameTournamentPlace(learned[i], first)) {
        tied.push(learned[i]);
      } else {
        break;
      }
    }
    if (tied.length >= MAX_SURVIVORS) {
      return tied;
    }
    return learned.slice(0, MAX_SURVIVORS);
  }

  ShapeSearch.prototype.finishRound = function () {
    var tourn = APP.shapeTournament(this.buildTournamentEntries(), this.mazeSize, TOURN_MAZES);
    this.lastTournament = tourn;
    this.liveTourn = null;
    var learned = tourn.ranked.filter(function (r) {
      return !isBaselineEntry(r);
    });
    var survivors = selectTournamentSurvivors(learned);
    var pop = this.population;
    this.tournamentSnapshots = survivors.map(function (r) {
      var ind = null;
      var si;
      for (si = 0; si < pop.length; si += 1) {
        if (pop[si].id === r.id) {
          ind = pop[si];
          break;
        }
      }
      return ind ? {
        id: ind.id,
        family: ind.family,
        label: ind.label,
        layerSizes: cloneSizes(ind.layerSizes),
        learningRate: ind.learningRate,
        visionRadius: ind.visionRadius,
        agentDict: agentSnapshot(ind),
        neatSerial: ind.neatSerial
      } : null;
    }).filter(Boolean);

    var histTop = learned.slice(0, HISTORY_LEARNED_LIMIT);
    var baselineTop = tourn.ranked.filter(function (r) {
      return isBaselineEntry(r);
    }).slice(0, HISTORY_BASELINE_TOP_N);
    this.history.unshift({
      round: this.round,
      mazeSize: this.mazeSize,
      mmLevel: this.mazeSize,
      top: histTop.concat(baselineTop),
      tournament: tourn.ranked
    });
    if (this.history.length > 20) {
      this.history.length = 20;
    }

    var oldPop = this.population;
    var nextPop = [];
    var i;
    for (i = 0; i < survivors.length; i += 1) {
      var src = findInPop(oldPop, survivors[i].id);
      if (!src) {
        continue;
      }
      keepIndividualInto(src, nextPop);
    }
    var parents = survivors.slice(0, MAX_MUTANT_PARENTS);
    var visionMode = this.visionMode;
    for (i = 0; i < parents.length; i += 1) {
      var p = findInPop(oldPop, parents[i].id);
      spawnOffspringFromParent(p, nextPop, visionMode);
    }
    ensureCategoryFloors(nextPop, visionMode, learned, oldPop);
    this.population = nextPop.length ? nextPop : this.population.map(function (ind) {
      resetRoundMarks(ind);
      return ind;
    });
    for (i = 0; i < this.population.length; i += 1) {
      resetRoundMarks(this.population[i]);
    }
    if (APP.mazePoolClear) {
      APP.mazePoolClear();
    }
    this._teachPoolAfterId = 0;
    this.round += 1;
    if (this.mazeSize < C.SIZE_MAX) {
      this.mazeSize += 1;
    }
    this.bestBeatSec = null;
    this.statusLine = "Advanced to " + formatSizeTag(this.mazeSize);
  };

  ShapeSearch.prototype.manualTournament = function () {
    this.lastTournament = APP.shapeTournament(this.buildTournamentEntries(), this.mazeSize, TOURN_MAZES);
  };

  ShapeSearch.prototype.tick = function (budgetMs) {
    if (this.liveTourn) {
      return;
    }
    if (this.roundFillComplete()) {
      this.finishRound();
      return;
    }
    if (this.paused) {
      return;
    }
    var pick = this.pickTrainTarget();
    if (!pick) {
      if (this.hasBeater()) {
        this.finishRound();
      }
      return;
    }
    if (pick.kind === "teach") {
      this.runPeerTeach(budgetMs);
      return;
    }
    this._currentId = pick.ind.id;
    if (pick.kind === "eval") {
      this.runBotEval(pick.ind);
      return;
    }
    var slice = Math.min(budgetMs || 120, 200);
    this.trainIndividual(pick.ind, slice);
  };

  ShapeSearch.prototype.getRows = function () {
    return this.population.map(function (ind) {
      ensureEvalMeta(ind);
      return {
        id: ind.id,
        label: ind.label,
        family: ind.family,
        secondsUsed: ind.secondsUsed,
        lastEvalWinRate: ind.lastEvalWinRate,
        lastEvalWins: ind.lastEvalWins,
        lastEvalDraws: ind.lastEvalDraws,
        lastEvalGames: ind.lastEvalGames,
        beatStreak: ind.beatStreak,
        beatStreakNeed: BEAT_STREAK_NEED,
        beaten: ind.beaten,
        secondsToBeat: ind.secondsToBeat
      };
    });
  };

  /* --- Train One --- */
  function TrainOne() {
    this.trainee = null;
    this.sourceId = null;
    this.paused = true;
    this.userPaused = true;
    this.mazeSize = C.SIZE_MIN;
    this.ladder = [];
    this.lastTournament = null;
    this.totalTrainSec = 0;
    this._trainSinceTourn = 0;
    this._nextTournAt = 1;
    this._lastTournMs = 1000;
  }

  TrainOne.prototype.startFromSnapshotOrLive = function (entry, liveInd) {
    var spec;
    if (liveInd) {
      spec = {
        family: liveInd.family,
        layerSizes: cloneSizes(liveInd.layerSizes),
        learningRate: liveInd.learningRate,
        visionRadius: liveInd.visionRadius,
        label: liveInd.label + " (trainee)",
        neatSerial: liveInd.neatSerial
      };
      this.trainee = createIndividual(spec);
      restoreAgent(this.trainee, agentSnapshot(liveInd));
      this.sourceId = liveInd.id;
    } else if (entry) {
      spec = {
        family: entry.family,
        layerSizes: cloneSizes(entry.layerSizes),
        learningRate: entry.learningRate,
        visionRadius: entry.visionRadius,
        label: (entry.label || entry.family) + " (trainee)",
        neatSerial: entry.neatSerial
      };
      this.trainee = createIndividual(spec);
      if (entry.agentDict) {
        restoreAgent(this.trainee, entry.agentDict);
      }
      this.sourceId = entry.id || null;
    }
    this.mazeSize = C.SIZE_MIN;
    this.ladder = [];
    this.totalTrainSec = 0;
    this._trainSinceTourn = 0;
    this._nextTournAt = 1;
    this.paused = false;
    this.userPaused = false;
  };

  TrainOne.prototype.setPaused = function (paused, fromUser) {
    this.paused = !!paused;
    if (fromUser) {
      this.userPaused = !!paused;
    }
  };

  TrainOne.prototype.restartLadder = function () {
    if (!this.trainee) {
      return;
    }
    this.mazeSize = C.SIZE_MIN;
    this.ladder = [];
    this._trainSinceTourn = 0;
    this._nextTournAt = 1;
    resetRoundMarks(this.trainee);
    this.trainee.bestWinRate = this.trainee.bestWinRate;
  };

  TrainOne.prototype.resetTrainee = function () {
    if (!this.trainee) {
      return;
    }
    var fam = this.trainee.family;
    var sizes = cloneSizes(this.trainee.layerSizes);
    var lr = this.trainee.learningRate;
    var label = this.trainee.label;
    this.trainee = createIndividual({
      family: fam,
      layerSizes: sizes,
      learningRate: lr,
      label: label
    });
    this.mazeSize = C.SIZE_MIN;
    this.ladder = [];
    this.totalTrainSec = 0;
    this._trainSinceTourn = 0;
    this._nextTournAt = 1;
  };

  TrainOne.prototype.tick = function (budgetMs, searchPop) {
    if (!this.trainee || this.paused) {
      return;
    }
    ensureEvalMeta(this.trainee);
    var size = this.mazeSize;
    var slice = Math.min(budgetMs || 120, 200);
    if (!this.trainee.warmupDone) {
      var warmOut = APP.shapeTrainEpisodes(
        this.trainee.agent, this.trainee.family, size, Math.min(slice, WARMUP_MS), {
          explore: true,
          poolAfterId: this.trainee.mazePoolAfterId
        }
      );
      this.trainee.mazePoolAfterId = warmOut.poolAfterId;
      this.trainee.warmupDone = true;
      this.trainee.secondsUsed += slice / 1000;
      this.totalTrainSec += slice / 1000;
      this._trainSinceTourn += slice / 1000;
      return;
    }
    if (this.trainee.family === "neat" || this.trainee.family === "neat_value" ||
        this.trainee.family === "genetic_menace") {
      var evoOut = APP.shapeEvolveOne(this.trainee.agent, this.trainee.family, size, null, {
        poolAfterId: this.trainee.mazePoolAfterId
      });
      this.trainee.mazePoolAfterId = evoOut.poolAfterId;
    } else {
      var trainOut = APP.shapeTrainEpisodes(
        this.trainee.agent, this.trainee.family, size, slice, {
          explore: true,
          nemesisSeeds: this.trainee.nemesisSeeds,
          poolAfterId: this.trainee.mazePoolAfterId
        }
      );
      this.trainee.mazePoolAfterId = trainOut.poolAfterId;
    }
    this.trainee.secondsUsed += slice / 1000;
    this.totalTrainSec += slice / 1000;
    this._trainSinceTourn += slice / 1000;
    this.trainee.trainSinceEvalMs += slice;

    if (this.botDueForEval(this.trainee)) {
      var ev = APP.shapeEvalSuccess(this.trainee.agent, this.trainee.family, size, EVAL_EPISODES);
      this.trainee.lastEvalWinRate = ev.winRate;
      this.trainee.lastEvalWins = ev.wins;
      this.trainee.lastEvalGames = ev.games;
      this.trainee.trainSinceEvalMs = 0;
      noteBeatStreak(this.trainee, ev.winRate);
      if (this.trainee.beatStreak >= BEAT_STREAK_NEED) {
        this.trainee.beaten = true;
      }
      if (!wrBeats(ev.winRate)) {
        this.trainee.beaten = false;
      }
      applyKeepBest(this.trainee, ev, size);
      if (ev.failedSeeds) {
        this.trainee.nemesisSeeds = ev.failedSeeds.slice(0, NEMESIS_MAX);
      }
      /* Collapse: WR below 50% drops one maze size (keep weights). */
      if (typeof ev.winRate === "number" && ev.winRate < DEMOTE_WR &&
          this.mazeSize > C.SIZE_MIN) {
        this.mazeSize -= 1;
        resetRoundMarks(this.trainee);
      }
    }

    if (this._trainSinceTourn >= this._nextTournAt) {
      this.runContenderTournament(searchPop || []);
    }
  };

  TrainOne.prototype.botDueForEval = function (ind) {
    ensureEvalMeta(ind);
    return ind.warmupDone && ind.trainSinceEvalMs >= (ind.nextTrainTargetMs || TRAIN_BETWEEN_EVAL_MS);
  };

  TrainOne.prototype.runContenderTournament = function (searchPop) {
    if (!this.trainee) {
      return;
    }
    var entries = [{
      id: "trainee",
      family: this.trainee.family,
      label: this.trainee.label,
      agent: this.trainee.agent
    }];
    var i;
    for (i = 0; i < Math.min(5, searchPop.length); i += 1) {
      entries.push({
        id: searchPop[i].id,
        family: searchPop[i].family,
        label: searchPop[i].label,
        agent: searchPop[i].agent
      });
    }
    entries.push({
      id: "master",
      family: "master",
      label: "Master",
      agent: APP.makeMasterAgent()
    });
    entries.push({
      id: "hamilton",
      family: "hamilton",
      label: "Hamilton",
      agent: APP.makeHamiltonAgent()
    });
    entries.push({
      id: "hamilton_aggr",
      family: "hamilton_aggr",
      label: "Hamilton Aggressive",
      agent: APP.makeHamiltonAggrAgent()
    });
    entries.push({
      id: "cell_tree",
      family: "cell_tree",
      label: "Cell Tree",
      agent: APP.makeCellTreeAgent()
    });
    entries.push({
      id: "dhcr",
      family: "dhcr",
      label: "DHCR",
      agent: APP.makeDhcrAgent()
    });
    entries.push({
      id: "dhcr_improved",
      family: "dhcr_improved",
      label: "DHCR Improved",
      agent: APP.makeDhcrImprovedAgent()
    });
    entries.push({
      id: "tail_safe",
      family: "tail_safe",
      label: "Tail Safe",
      agent: APP.makeTailSafeAgent()
    });
    entries.push({
      id: "tail_safe_hybrid",
      family: "tail_safe_hybrid",
      label: "Tail Safe Hybrid",
      agent: APP.makeTailSafeHybridAgent()
    });
    entries.push({
      id: "plan_ahead",
      family: "plan_ahead",
      label: "Plan Ahead",
      agent: APP.makePlanAheadAgent()
    });
    entries.push({
      id: "random",
      family: "random",
      label: "Random",
      agent: APP.makeRandomAgent()
    });
    entries.push({
      id: "greedy",
      family: "greedy",
      label: "Greedy",
      agent: APP.makeGreedyAgent()
    });
    entries.push({
      id: "wall_follower",
      family: "wall_follower",
      label: "Wall follower",
      agent: APP.makeWallFollowerAgent()
    });
    entries.push({
      id: "bfs",
      family: "bfs",
      label: "BFS",
      agent: APP.makeBfsAgent()
    });
    entries.push({
      id: "avoid_pocket",
      family: "avoid_pocket",
      label: "Avoid Pocket",
      agent: APP.makeAvoidPocketAgent()
    });
    var t0 = performance.now();
    this.lastTournament = APP.shapeTournament(entries, this.mazeSize, TOURN_MAZES);
    this._lastTournMs = performance.now() - t0;
    this._trainSinceTourn = 0;
    this._nextTournAt = Math.max(1, (this._lastTournMs / 1000) * 5);

    /* Advance when beat streak is filled (same gate as Shape Search). */
    ensureEvalMeta(this.trainee);
    if (this.trainee.beatStreak >= BEAT_STREAK_NEED || this.trainee.beaten) {
      this.trainee.beaten = true;
      this.ladder.push({
        mmLevel: this.mazeSize,
        mazeSize: this.mazeSize,
        seconds: this.totalTrainSec,
        winRate: this.trainee.lastEvalWinRate || 0
      });
      if (this.mazeSize < C.SIZE_MAX) {
        this.mazeSize += 1;
        resetRoundMarks(this.trainee);
      }
    }
  };

  /* --- ShapeLab shell --- */
  function ShapeLab(opts) {
    this.onSnapshot = opts && opts.onSnapshot;
    this.search = new ShapeSearch();
    this.trainOne = new TrainOne();
    this.activeTab = "search";
    this._raf = null;
    this._lastSave = 0;
    this._lastSaveAt = 0;
    this._savePending = false;
    this._saveFailed = false;
    this._saveNote = "not saved yet";
    this._lastPlayDepth = 0;
    this._running = false;
  }

  ShapeLab.prototype.saveStatusLine = function () {
    if (this._saveNote === "tick error") {
      return "tick error";
    }
    if (this._savePending) {
      return "saving...";
    }
    if (this._saveFailed) {
      return "save FAILED";
    }
    if (!this._lastSaveAt) {
      return this._saveNote || "not saved yet";
    }
    var agoSec = Math.max(0, Math.round((Date.now() - this._lastSaveAt) / 1000));
    var when;
    if (agoSec < 60) {
      when = "saved " + agoSec + "s ago";
    } else if (agoSec < 3600) {
      when = "saved " + Math.floor(agoSec / 60) + "m ago";
    } else {
      when = "last saved at " + new Date(this._lastSaveAt).toLocaleTimeString();
    }
    if (this._saveNote === "restored") {
      return "restored · " + when;
    }
    return when;
  };

  ShapeLab.prototype.setActiveTab = function (tab) {
    var prev = this.activeTab;
    this.activeTab = tab;
    if (prev === "search" && tab !== "search") {
      this.search.setPaused(true, false);
    }
    if (tab === "search") {
      this.search.resumeIfAllowed();
    }
  };

  ShapeLab.prototype.setBotSearchOffset = function () {
    /* No Minimax ply offset in maze lab. */
  };

  ShapeLab.prototype.tick = function () {
    /* Short slices so the browser can paint; long slices looked "stuck". */
    var budget = 40;
    if (this.activeTab === "search") {
      this.search.tick(budget);
    } else if (this.activeTab === "train") {
      this.trainOne.tick(budget, this.search.population);
    }
    var now = Date.now();
    if (this.onSnapshot && (!this._lastUiAt || now - this._lastUiAt >= 100)) {
      this._lastUiAt = now;
      this.onSnapshot(this.getSnapshot());
    }
    /* No autosave on Observe/About - nothing training-related is changing. */
    if (this.activeTab !== "observe" && this.activeTab !== "about" &&
        now - this._lastSave > AUTO_SAVE_MS) {
      this.saveLocal();
    }
  };

  ShapeLab.prototype.start = function () {
    if (this._running) {
      return;
    }
    this._running = true;
    var self = this;
    var lastFrame = performance.now();
    function loop() {
      if (!self._running) {
        return;
      }
      var now = performance.now();
      self._slowUi = (now - lastFrame) >= 1000;
      lastFrame = now;
      try {
        self.tick();
      } catch (err) {
        console.error(err);
        self._saveNote = "tick error";
      }
      self._raf = requestAnimationFrame(loop);
    }
    this._raf = requestAnimationFrame(loop);
  };

  ShapeLab.prototype.stop = function () {
    this._running = false;
    if (this._raf) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
  };

  ShapeLab.prototype.unionCatalog = function () {
    var seen = {};
    var out = [];
    var i;
    for (i = 0; i < this.search.population.length; i += 1) {
      var ind = this.search.population[i];
      if (!seen[ind.id]) {
        seen[ind.id] = true;
        out.push({ id: ind.id, label: ind.label, family: ind.family });
      }
    }
    for (i = 0; i < this.search.tournamentSnapshots.length; i += 1) {
      var snap = this.search.tournamentSnapshots[i];
      if (!seen[snap.id]) {
        seen[snap.id] = true;
        out.push({ id: snap.id, label: snap.label + " (tourn)", family: snap.family });
      }
    }
    return out;
  };

  ShapeLab.prototype.getPlayCatalog = function () {
    var cat = [
      { id: "master", name: "Master", family: "master" },
      { id: "hamilton", name: "Hamilton", family: "hamilton" },
      { id: "hamilton_aggr", name: "Hamilton Aggressive", family: "hamilton_aggr" },
      { id: "cell_tree", name: "Cell Tree", family: "cell_tree" },
      { id: "cell_tree_odd", name: "Cell Tree Odd", family: "cell_tree_odd" },
      { id: "dhcr", name: "DHCR", family: "dhcr" },
      { id: "dhcr_improved", name: "DHCR Improved", family: "dhcr_improved" },
      { id: "tail_safe", name: "Tail Safe", family: "tail_safe" },
      { id: "tail_safe_hybrid", name: "Tail Safe Hybrid", family: "tail_safe_hybrid" },
      { id: "plan_ahead", name: "Plan Ahead", family: "plan_ahead" },
      { id: "random", name: "Random", family: "random" },
      { id: "greedy", name: "Greedy", family: "greedy" },
      { id: "wall_follower", name: "Wall follower", family: "wall_follower" },
      { id: "bfs", name: "BFS", family: "bfs" },
      { id: "avoid_pocket", name: "Avoid Pocket", family: "avoid_pocket" }
    ];
    if (this.trainOne.trainee) {
      cat.push({
        id: "trainee",
        name: this.trainOne.trainee.label || "Trainee",
        family: this.trainOne.trainee.family
      });
    }
    var seen = {
      random: true,
      greedy: true,
      wall_follower: true,
      bfs: true,
      avoid_pocket: true,
      master: true,
      hamilton: true,
      hamilton_aggr: true,
      cell_tree: true,
      cell_tree_odd: true,
      cell_tree_fringe: true,
      dhcr: true,
      dhcr_improved: true,
      plan_ahead: true,
      tail_safe: true,
      tail_safe_hybrid: true,
      trainee: true
    };
    var i;
    for (i = 0; i < this.search.population.length; i += 1) {
      var ind = this.search.population[i];
      if (!seen[ind.id]) {
        seen[ind.id] = true;
        cat.push({ id: ind.id, name: ind.label, family: ind.family });
      }
    }
    return cat;
  };

  ShapeLab.prototype.resolvePlayAgent = function (algoId) {
    if (algoId === "random") {
      return { agent: APP.makeRandomAgent(), family: "random" };
    }
    if (algoId === "greedy") {
      return { agent: APP.makeGreedyAgent(), family: "greedy" };
    }
    if (algoId === "wall_follower") {
      return { agent: APP.makeWallFollowerAgent(), family: "wall_follower" };
    }
    if (algoId === "bfs") {
      return { agent: APP.makeBfsAgent(), family: "bfs" };
    }
    if (algoId === "avoid_pocket") {
      return { agent: APP.makeAvoidPocketAgent(), family: "avoid_pocket" };
    }
    if (algoId === "master") {
      return { agent: APP.makeMasterAgent(), family: "master" };
    }
    if (algoId === "hamilton") {
      return { agent: APP.makeHamiltonAgent(), family: "hamilton" };
    }
    if (algoId === "hamilton_aggr" ||
        algoId === "hamilton_soft" ||
        algoId === "hamilton_early" ||
        algoId === "hamilton_arc") {
      /* Old Early/Arc/Soft ids map to Aggr so saved Observe picks still run. */
      return { agent: APP.makeHamiltonAggrAgent(), family: "hamilton_aggr" };
    }
    if (algoId === "cell_tree") {
      return { agent: APP.makeCellTreeAgent(), family: "cell_tree" };
    }
    if (algoId === "cell_tree_odd" || algoId === "cell_tree_fringe" ||
        algoId === "cell_tree_odd_lock") {
      /* Odd Lock removed from Observe; old saves map to Cell Tree Odd. */
      return { agent: APP.makeCellTreeOddAgent(), family: "cell_tree_odd" };
    }
    if (algoId === "dhcr") {
      return { agent: APP.makeDhcrAgent(), family: "dhcr" };
    }
    if (algoId === "dhcr_improved") {
      return { agent: APP.makeDhcrImprovedAgent(), family: "dhcr_improved" };
    }
    if (algoId === "plan_ahead") {
      return { agent: APP.makePlanAheadAgent(), family: "plan_ahead" };
    }
    if (algoId === "tail_safe") {
      return { agent: APP.makeTailSafeAgent(), family: "tail_safe" };
    }
    if (algoId === "tail_safe_hybrid") {
      return { agent: APP.makeTailSafeHybridAgent(), family: "tail_safe_hybrid" };
    }
    if (algoId === "trainee" && this.trainOne.trainee) {
      return { agent: this.trainOne.trainee.agent, family: this.trainOne.trainee.family };
    }
    var i;
    for (i = 0; i < this.search.population.length; i += 1) {
      if (this.search.population[i].id === algoId) {
        return {
          agent: this.search.population[i].agent,
          family: this.search.population[i].family
        };
      }
    }
    return { agent: APP.makeRandomAgent(), family: "random" };
  };

  ShapeLab.prototype.pickPlayMove = function (algoId, env) {
    var resolved = this.resolvePlayAgent(algoId);
    this._lastPlayDepth = 0;
    if (APP.chooseDecision) {
      return APP.chooseDecision(resolved.agent, resolved.family, env, false);
    }
    return { action: APP.chooseAction(resolved.agent, resolved.family, env, false) };
  };

  /** Toggle hard-mode and rebuild all bots with fresh weights for the new mode. */
  ShapeLab.prototype.setHardMode = function (on) {
    var next = !!on;
    if (APP.setHardMode) {
      APP.setHardMode(next);
    } else {
      APP.hardMode = next;
    }
    this.resetBotsForMode();
    this._saveNote = next ? "hard-mode on" : "hard-mode off";
    this.saveLocal();
    return next;
  };

  ShapeLab.prototype.resetBotsForMode = function () {
    nextId = 1;
    neatSerialByFamily = { neat: 0, neat_value: 0 };
    this.search = new ShapeSearch();
    this.trainOne = new TrainOne();
    if (APP.mazePoolClear) {
      APP.mazePoolClear();
    }
  };

  ShapeLab.prototype.startTrainFromCatalog = function (id) {
    var i;
    for (i = 0; i < this.search.population.length; i += 1) {
      if (this.search.population[i].id === id) {
        this.trainOne.startFromSnapshotOrLive(null, this.search.population[i]);
        return true;
      }
    }
    for (i = 0; i < this.search.tournamentSnapshots.length; i += 1) {
      if (this.search.tournamentSnapshots[i].id === id) {
        this.trainOne.startFromSnapshotOrLive(this.search.tournamentSnapshots[i], null);
        return true;
      }
    }
    return false;
  };

  ShapeLab.prototype.promoteTraineeToSearch = function () {
    var t = this.trainOne.trainee;
    if (!t) {
      return false;
    }
    var i;
    if (this.trainOne.sourceId) {
      for (i = 0; i < this.search.population.length; i += 1) {
        if (this.search.population[i].id === this.trainOne.sourceId) {
          restoreAgent(this.search.population[i], agentSnapshot(t));
          resetRoundMarks(this.search.population[i]);
          this.search.population[i].bestWinRate = t.bestWinRate;
          this.search.population[i].bestAgentDict = t.bestAgentDict;
          return true;
        }
      }
    }
    var added = createIndividual({
      family: t.family,
      layerSizes: cloneSizes(t.layerSizes),
      learningRate: t.learningRate,
      label: (t.label || "imported").replace(" (trainee)", "")
    });
    restoreAgent(added, agentSnapshot(t));
    this.search.population.push(added);
    this.trainOne.sourceId = added.id;
    return true;
  };

  ShapeLab.prototype.exportTrainee = function () {
    var t = this.trainOne.trainee;
    if (!t) {
      return null;
    }
    return {
      format: EXPORT_FORMAT,
      family: t.family,
      layerSizes: cloneSizes(t.layerSizes),
      learningRate: t.learningRate,
      visionRadius: t.visionRadius,
      label: t.label,
      agent: agentSnapshot(t),
      mazeSize: this.trainOne.mazeSize,
      ladder: this.trainOne.ladder.slice()
    };
  };

  ShapeLab.prototype.importTrainee = function (data, name) {
    if (!data || data.format !== EXPORT_FORMAT) {
      throw new Error("Unsupported bot file");
    }
    var ind = createIndividual({
      family: data.family,
      layerSizes: cloneSizes(data.layerSizes),
      learningRate: data.learningRate,
      visionRadius: data.visionRadius,
      label: (name || data.label || data.family) + " (trainee)"
    });
    restoreAgent(ind, data.agent);
    this.trainOne.trainee = ind;
    this.trainOne.sourceId = null;
    this.trainOne.mazeSize = data.mazeSize || C.SIZE_MIN;
    this.trainOne.ladder = data.ladder || [];
    this.trainOne.paused = true;
    this.trainOne.userPaused = true;
    this.trainOne.totalTrainSec = 0;
  };

  ShapeLab.prototype.getSnapshot = function () {
    var s = this.search;
    var cur = null;
    var i;
    for (i = 0; i < s.population.length; i += 1) {
      if (s.population[i].id === s._currentId) {
        cur = s.population[i];
        break;
      }
    }
    var t = this.trainOne;
    return {
      saveStatusLine: this.saveStatusLine(),
      slowUi: !!this._slowUi,
      catalog: this.unionCatalog(),
      search: {
        round: s.round,
        mazeSize: s.mazeSize,
        mmLevel: s.mazeSize,
        botSearchOffset: 0,
        botMaxDepth: 0,
        paused: s.paused,
        userPaused: s.userPaused,
        currentLabel: cur ? cur.label : "-",
        currentUsed: cur ? cur.secondsUsed : 0,
        bestBeatSec: s.bestBeatSec,
        maxTrainSec: s.maxTrainSec(),
        hardMode: !!(APP.isHardMode ? APP.isHardMode() : APP.hardMode),
        compassOn: !!(APP.isCompassOn ? APP.isCompassOn() : APP.compassOn),
        visionMode: s.visionMode,
        teachEachOther: !!s.teachEachOther,
        learnFromMaster: !!s.learnFromMaster,
        rows: s.getRows(),
        lastTournament: s.lastTournament,
        tournamentHeading: s.liveTourn ? "Current tournament" : "Latest tournament",
        history: s.history,
        statusLine: s.statusLine
      },
      trainOne: {
        hasTrainee: !!t.trainee,
        label: t.trainee ? t.trainee.label : null,
        paused: t.paused,
        userPaused: t.userPaused,
        mazeSize: t.mazeSize,
        mmLevel: t.mazeSize,
        totalTrainSec: t.totalTrainSec,
        beatStreak: t.trainee ? (t.trainee.beatStreak || 0) : 0,
        beatStreakNeed: BEAT_STREAK_NEED,
        ladder: t.ladder,
        lastTournament: t.lastTournament
      }
    };
  };

  ShapeLab.prototype.toCheckpoint = function () {
    var self = this;
    return {
      format: CHECKPOINT_FORMAT,
      savedAt: new Date().toISOString(),
      hardMode: !!(APP.isHardMode ? APP.isHardMode() : APP.hardMode),
      compassOn: !!(APP.isCompassOn ? APP.isCompassOn() : APP.compassOn !== false),
      nextId: nextId,
      neatSerialByFamily: neatSerialByFamily,
      search: {
        round: self.search.round,
        mazeSize: self.search.mazeSize,
        visionMode: self.search.visionMode,
        paused: self.search.paused,
        userPaused: self.search.userPaused,
        teachEachOther: !!self.search.teachEachOther,
        learnFromMaster: !!self.search.learnFromMaster,
        bestBeatSec: self.search.bestBeatSec,
        history: self.search.history,
        tournamentSnapshots: self.search.tournamentSnapshots,
        population: self.search.population.map(function (ind) {
          return {
            id: ind.id,
            family: ind.family,
            layerSizes: cloneSizes(ind.layerSizes),
            learningRate: ind.learningRate,
            visionRadius: ind.visionRadius,
            label: ind.label,
            neatSerial: ind.neatSerial,
            secondsUsed: ind.secondsUsed,
            evalSeconds: ind.evalSeconds,
            beaten: ind.beaten,
            beatStreak: ind.beatStreak,
            secondsToBeat: ind.secondsToBeat,
            lastEvalWinRate: ind.lastEvalWinRate,
            lastEvalWins: ind.lastEvalWins,
            lastEvalGames: ind.lastEvalGames,
            bestWinRate: ind.bestWinRate,
            bestAgentDict: ind.bestAgentDict,
            evalGamesTarget: ind.evalGamesTarget,
            warmupDone: ind.warmupDone,
            agent: agentSnapshot(ind)
          };
        })
      },
      trainOne: self.trainOne.trainee ? {
        sourceId: self.trainOne.sourceId,
        mazeSize: self.trainOne.mazeSize,
        ladder: self.trainOne.ladder,
        totalTrainSec: self.trainOne.totalTrainSec,
        paused: self.trainOne.paused,
        userPaused: self.trainOne.userPaused,
        trainee: {
          family: self.trainOne.trainee.family,
          layerSizes: cloneSizes(self.trainOne.trainee.layerSizes),
          learningRate: self.trainOne.trainee.learningRate,
          visionRadius: self.trainOne.trainee.visionRadius,
          label: self.trainOne.trainee.label,
          agent: agentSnapshot(self.trainOne.trainee),
          bestWinRate: self.trainOne.trainee.bestWinRate,
          bestAgentDict: self.trainOne.trainee.bestAgentDict
        }
      } : null
    };
  };

  ShapeLab.prototype.fromCheckpoint = function (data) {
    if (!data || data.format !== CHECKPOINT_FORMAT) {
      return false;
    }
    if (APP.setHardMode) {
      APP.setHardMode(!!data.hardMode);
    } else {
      APP.hardMode = !!data.hardMode;
    }
    if (typeof data.compassOn === "boolean") {
      if (APP.setCompassOn) {
        APP.setCompassOn(data.compassOn);
      } else {
        APP.compassOn = data.compassOn;
      }
    }
    if (typeof data.nextId === "number") {
      nextId = data.nextId;
    }
    if (data.neatSerialByFamily) {
      neatSerialByFamily = data.neatSerialByFamily;
    }
    var s = data.search || {};
    this.search.round = s.round || 1;
    this.search.mazeSize = s.mazeSize || C.SIZE_MIN;
    if (s.visionMode !== undefined && s.visionMode !== null) {
      this.search.setVisionMode(s.visionMode);
    }
    /* Always resume Shape Search after restore (tab auto-pause must not stick). */
    this.search.paused = false;
    this.search.userPaused = false;
    this.search.bestBeatSec = s.bestBeatSec;
    this.search.teachEachOther = !!s.teachEachOther;
    this.search.learnFromMaster = !!s.learnFromMaster;
    this.search._teachPoolAfterId = 0;
    this.search._teachAlternate = false;
    this.search.history = s.history || [];
    this.search.tournamentSnapshots = s.tournamentSnapshots || [];
    this.search.population = [];
    var i;
    var pop = s.population || [];
    for (i = 0; i < pop.length; i += 1) {
      var p = pop[i];
      var ind = createIndividual({
        id: p.id,
        family: p.family,
        layerSizes: cloneSizes(p.layerSizes),
        learningRate: p.learningRate,
        visionRadius: p.visionRadius,
        label: p.label,
        neatSerial: p.neatSerial
      });
      restoreAgent(ind, p.agent);
      ind.secondsUsed = p.secondsUsed || 0;
      ind.evalSeconds = p.evalSeconds || 0;
      ind.beaten = !!p.beaten;
      ind.beatStreak = p.beatStreak || 0;
      ind.secondsToBeat = p.secondsToBeat;
      ind.lastEvalWinRate = p.lastEvalWinRate;
      ind.lastEvalWins = p.lastEvalWins;
      ind.lastEvalGames = p.lastEvalGames;
      ind.bestWinRate = p.bestWinRate;
      ind.bestAgentDict = p.bestAgentDict;
      ind.evalGamesTarget = p.evalGamesTarget || EVAL_EPISODES;
      ind.warmupDone = !!p.warmupDone;
      this.search.population.push(ind);
    }
    if (!this.search.population.length) {
      this.search.initPopulation();
    }
    if (data.trainOne && data.trainOne.trainee) {
      var td = data.trainOne.trainee;
      this.trainOne.trainee = createIndividual({
        family: td.family,
        layerSizes: cloneSizes(td.layerSizes),
        learningRate: td.learningRate,
        visionRadius: td.visionRadius,
        label: td.label
      });
      restoreAgent(this.trainOne.trainee, td.agent);
      this.trainOne.trainee.bestWinRate = td.bestWinRate;
      this.trainOne.trainee.bestAgentDict = td.bestAgentDict;
      this.trainOne.sourceId = data.trainOne.sourceId;
      this.trainOne.mazeSize = data.trainOne.mazeSize || C.SIZE_MIN;
      this.trainOne.ladder = data.trainOne.ladder || [];
      this.trainOne.totalTrainSec = data.trainOne.totalTrainSec || 0;
      this.trainOne.paused = data.trainOne.paused !== false;
      this.trainOne.userPaused = data.trainOne.userPaused !== false;
    }
    return true;
  };

  ShapeLab.prototype._markSaved = function (atMs) {
    var when = typeof atMs === "number" && atMs > 0 ? atMs : Date.now();
    this._lastSave = when;
    this._lastSaveAt = when;
    this._savePending = false;
    this._saveFailed = false;
    this._saveNote = "";
  };

  ShapeLab.prototype.saveLocal = function (cb) {
    var self = this;
    var data;
    var json;
    self._savePending = true;
    self._saveFailed = false;
    try {
      data = this.toCheckpoint();
      json = JSON.stringify(data);
    } catch (err) {
      self._savePending = false;
      self._saveFailed = true;
      self._saveNote = "save FAILED";
      if (typeof console !== "undefined" && console.warn) {
        console.warn("Snake lab checkpoint serialize failed", err);
      }
      if (cb) {
        cb({ ok: false, error: String(err) });
      }
      return;
    }
    /* localStorage is optional (often hits quota with NN weights). IndexedDB is primary. */
    var lsOk = false;
    try {
      localStorage.setItem(STORAGE_KEY, json);
      lsOk = true;
    } catch (lsErr) {
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch (ignore) {}
      if (typeof console !== "undefined" && console.warn) {
        console.warn("Snake lab localStorage save skipped", lsErr);
      }
    }
    function failBoth(err) {
      self._savePending = false;
      self._saveFailed = true;
      self._saveNote = "save FAILED";
      if (typeof console !== "undefined" && console.warn) {
        console.warn("Snake lab IndexedDB save failed", err);
      }
      if (cb) {
        cb({ ok: false, error: String(err || "idb") });
      }
    }
    function okSaved() {
      self._markSaved(Date.now());
      if (!lsOk) {
        self._saveNote = "saved (IndexedDB only)";
      }
      if (cb) {
        cb({ ok: true, localStorage: lsOk });
      }
    }
    try {
      var req = indexedDB.open(IDB_NAME, 1);
      req.onupgradeneeded = function () {
        var db = req.result;
        if (!db.objectStoreNames.contains(IDB_STORE)) {
          db.createObjectStore(IDB_STORE);
        }
      };
      req.onsuccess = function () {
        try {
          var db = req.result;
          var tx = db.transaction(IDB_STORE, "readwrite");
          /* Store JSON string so file:// / structured-clone quirks with TypedArrays are avoided. */
          tx.objectStore(IDB_STORE).put(json, STORAGE_KEY);
          tx.oncomplete = function () {
            okSaved();
          };
          tx.onerror = function () {
            if (lsOk) {
              okSaved();
            } else {
              failBoth(tx.error);
            }
          };
        } catch (e2) {
          if (lsOk) {
            okSaved();
          } else {
            failBoth(e2);
          }
        }
      };
      req.onerror = function () {
        if (lsOk) {
          okSaved();
        } else {
          failBoth(req.error);
        }
      };
    } catch (idbOpenErr) {
      if (lsOk) {
        okSaved();
      } else {
        failBoth(idbOpenErr);
      }
    }
  };

  ShapeLab.prototype.loadLocal = function (cb) {
    var self = this;
    function fromJson(raw) {
      try {
        var data = typeof raw === "string" ? JSON.parse(raw) : raw;
        return self.fromCheckpoint(data);
      } catch (err) {
        return false;
      }
    }
    var req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = function () {
      var db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE);
      }
    };
    req.onsuccess = function () {
      try {
        var db = req.result;
        var tx = db.transaction(IDB_STORE, "readonly");
        var g = tx.objectStore(IDB_STORE).get(STORAGE_KEY);
        g.onsuccess = function () {
          if (g.result && fromJson(g.result)) {
            if (cb) {
              cb(true, g.result);
            }
            return;
          }
          var ls = localStorage.getItem(STORAGE_KEY);
          var okLs = !!(ls && fromJson(ls));
          if (cb) {
            cb(okLs, okLs ? ls : null);
          }
        };
        g.onerror = function () {
          var ls = localStorage.getItem(STORAGE_KEY);
          var okLs = !!(ls && fromJson(ls));
          if (cb) {
            cb(okLs, okLs ? ls : null);
          }
        };
      } catch (err) {
        var ls2 = localStorage.getItem(STORAGE_KEY);
        var ok2 = !!(ls2 && fromJson(ls2));
        if (cb) {
          cb(ok2, ok2 ? ls2 : null);
        }
      }
    };
    req.onerror = function () {
      var ls = localStorage.getItem(STORAGE_KEY);
      var okLs = !!(ls && fromJson(ls));
      if (cb) {
        cb(okLs, okLs ? ls : null);
      }
    };
  };

  ShapeLab.prototype.resetAll = function () {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (err) {}
    try {
      var req = indexedDB.open(IDB_NAME, 1);
      req.onsuccess = function () {
        try {
          var db = req.result;
          if (!db.objectStoreNames.contains(IDB_STORE)) {
            return;
          }
          var tx = db.transaction(IDB_STORE, "readwrite");
          tx.objectStore(IDB_STORE).delete(STORAGE_KEY);
        } catch (e2) {}
      };
    } catch (e3) {}
    var keepHard = !!(APP.isHardMode ? APP.isHardMode() : APP.hardMode);
    if (APP.setHardMode) {
      APP.setHardMode(keepHard);
    } else {
      APP.hardMode = keepHard;
    }
    this.resetBotsForMode();
    this._saveNote = "reset";
    this._lastSaveAt = 0;
    this._saveFailed = false;
    this.saveLocal();
  };

  function init(opts) {
    var lab = new ShapeLab(opts || {});
    var started = false;
    function begin() {
      if (started) {
        return;
      }
      started = true;
      lab.start();
      if (lab.onSnapshot) {
        lab.onSnapshot(lab.getSnapshot());
      }
    }
    /* Paint immediately so the page never sits on "Loading...". */
    if (lab.onSnapshot) {
      lab.onSnapshot(lab.getSnapshot());
    }
    lab.loadLocal(function (ok, raw) {
      if (ok) {
        var savedAt = 0;
        try {
          var data = typeof raw === "string" ? JSON.parse(raw) : raw;
          if (data && data.savedAt) {
            savedAt = Date.parse(data.savedAt) || 0;
          }
        } catch (eRest) {
          savedAt = 0;
        }
        lab._markSaved(savedAt || Date.now());
        lab._saveNote = "restored";
        lab.search.paused = false;
        lab.search.userPaused = false;
      } else if (!lab._lastSaveAt) {
        lab._saveNote = "no prior save";
      }
      begin();
    });
    setTimeout(begin, 400);
    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "hidden") {
        lab.saveLocal();
      }
    });
    window.addEventListener("pagehide", function () {
      lab.saveLocal();
    });
    return lab;
  }

  global.SNAKE_SHAPE_LAB = {
    init: init,
    ShapeLab: ShapeLab,
    ShapeSearch: ShapeSearch,
    TrainOne: TrainOne,
    formatMmTag: formatSizeTag,
    formatSizeTag: formatSizeTag,
    BEAT_WR: BEAT_WR,
    DEMOTE_WR: DEMOTE_WR,
    BEAT_STREAK_NEED: BEAT_STREAK_NEED,
    HISTORY_LEARNED_LIMIT: HISTORY_LEARNED_LIMIT,
    historyTopRanks: function (tournament, limit) {
      if (!tournament) {
        return [];
      }
      var rows = tournament.ranked || tournament;
      return (rows || []).filter(function (r) {
        return !isBaselineEntry(r);
      }).slice(0, limit || HISTORY_LEARNED_LIMIT);
    },
    formatBeatMinimaxSuffix: function () {
      return "";
    },
    createIndividual: createIndividual,
    gen0Specs: gen0Specs,
    selectTournamentSurvivors: selectTournamentSurvivors,
    familyLabel: familyLabel,
    makeUniqueLabel: makeUniqueLabel,
    layersNearDuplicate: layersNearDuplicate,
    shapeTakenInPop: shapeTakenInPop,
    morphDepth: morphDepth,
    spawnOffspringFromParent: spawnOffspringFromParent,
    categoryOfFamily: categoryOfFamily,
    ensureCategoryFloors: ensureCategoryFloors,
    CATEGORY_FLOOR: CATEGORY_FLOOR,
    applyKeepBest: applyKeepBest,
    BEST_WR_DIP: BEST_WR_DIP,
    EVAL_EPISODES_LONG: EVAL_EPISODES_LONG,
    NEAR_WIDTH_DUP: NEAR_WIDTH_DUP,
    CHECKPOINT_FORMAT: CHECKPOINT_FORMAT,
    EXPORT_FORMAT: EXPORT_FORMAT,
    setHardMode: function (lab, on) {
      return lab.setHardMode(on);
    }
  };
})(typeof window !== "undefined" ? window : global);
