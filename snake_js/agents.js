/**
 * Maze shape agents: tabular, NN Q/policy, REINFORCE, NEAT-lite, TF.js DQN.
 */
(function (global) {
  "use strict";

  var C = global.SNAKE_CONSTANTS;
  var NN = global.SNAKE_NN;
  var E = global.SNAKE_ENV;

  var Q_MAX = C.Q_MAX_STATES || 10000;
  var MENACE_MAX = C.MENACE_MAX_BOXES || 10000;
  var GAMMA = 0.99;

  function decisionFromOutputs(out, legal, explore, sampleFn) {
    if (typeof sampleFn === "function") {
      return sampleFn(out, legal, explore);
    }
    if (explore && Math.random() < 0.15) {
      return legal[Math.floor(Math.random() * legal.length)];
    }
    return NN.argmaxLegal(out, legal, false);
  }

  function packDecision(action) {
    return action;
  }

  function lruTouch(map, key, max) {
    if (map.has(key)) {
      var v = map.get(key);
      map.delete(key);
      map.set(key, v);
      return v;
    }
    if (map.size >= max) {
      var first = map.keys().next().value;
      map.delete(first);
    }
    return null;
  }

  function ensureQRow(map, key, max) {
    var row = lruTouch(map, key, max);
    if (row) {
      return row;
    }
    row = [0, 0, 0, 0];
    map.set(key, row);
    return row;
  }

  function maxQLegal(row, nextObs) {
    var legal = E.legalFromObs(nextObs);
    var best = -Infinity;
    var i;
    for (i = 0; i < legal.length; i += 1) {
      if (row[legal[i]] > best) {
        best = row[legal[i]];
      }
    }
    return best === -Infinity ? 0 : best;
  }

  function compareEvolveFit(a, b) {
    var fa = a.fit || {};
    var fb = b.fit || {};
    var wa = typeof fa.wins === "number" ? fa.wins : 0;
    var wb = typeof fb.wins === "number" ? fb.wins : 0;
    if (wb !== wa) {
      return wb - wa;
    }
    var sa = typeof fa.avgStepPct === "number" ? fa.avgStepPct : Number.POSITIVE_INFINITY;
    var sb = typeof fb.avgStepPct === "number" ? fb.avgStepPct : Number.POSITIVE_INFINITY;
    if (sa !== sb) {
      return sa - sb;
    }
    var ea = typeof fa.explored === "number" ? fa.explored : 0;
    var eb = typeof fb.explored === "number" ? fb.explored : 0;
    return eb - ea;
  }

  /* --- Tabular Q --- */
  function QBrain() {
    this.q = new Map();
    this.alpha = 0.1;
    this.epsilon = 0.15;
    this.visits = new Map();
    this._prevDir = null;
  }

  QBrain.prototype.chooseAction = function (obs, legal, explore) {
    if (!legal.length) {
      return packDecision(0);
    }
    var key = E.obsToStateKey(obs, this._prevDir);
    var row = ensureQRow(this.q, key, Q_MAX);
    var action;
    if (explore && Math.random() < this.epsilon) {
      action = legal[Math.floor(Math.random() * legal.length)];
    } else {
      var best = -Infinity;
      var picks = [];
      var i;
      for (i = 0; i < legal.length; i += 1) {
        var a = legal[i];
        if (row[a] > best) {
          best = row[a];
          picks = [a];
        } else if (row[a] === best) {
          picks.push(a);
        }
      }
      action = picks[Math.floor(Math.random() * picks.length)];
    }
    return packDecision(action);
  };

  QBrain.prototype.learnTransition = function (obs, action, reward, nextObs, done) {
    var key = E.obsToStateKey(obs, this._prevDir);
    var row = ensureQRow(this.q, key, Q_MAX);
    var maxNext = 0;
    if (!done) {
      var nk = E.obsToStateKey(nextObs, action);
      var nrow = ensureQRow(this.q, nk, Q_MAX);
      maxNext = maxQLegal(nrow, nextObs);
    }
    var target = reward + (done ? 0 : GAMMA * maxNext);
    row[action] += this.alpha * (target - row[action]);
  };

  QBrain.prototype.toDict = function () {
    var entries = [];
    this.q.forEach(function (v, k) {
      entries.push([k, v.slice()]);
    });
    return { type: "qtable", entries: entries, alpha: this.alpha, epsilon: this.epsilon };
  };

  QBrain.prototype.fromDict = function (d) {
    this.q = new Map();
    var i;
    if (d && d.entries) {
      for (i = 0; i < d.entries.length; i += 1) {
        this.q.set(d.entries[i][0], d.entries[i][1].slice());
      }
    }
    if (d && typeof d.alpha === "number") {
      this.alpha = d.alpha;
    }
    if (d && typeof d.epsilon === "number") {
      this.epsilon = d.epsilon;
    }
  };

  /* --- SARSA --- */
  function SarsaBrain() {
    QBrain.call(this);
  }
  SarsaBrain.prototype = Object.create(QBrain.prototype);
  SarsaBrain.prototype.constructor = SarsaBrain;

  SarsaBrain.prototype.learnTransition = function (obs, action, reward, nextObs, done, nextAction) {
    var key = E.obsToStateKey(obs, this._prevDir);
    var row = ensureQRow(this.q, key, Q_MAX);
    var nextQ = 0;
    if (!done && typeof nextAction === "number") {
      var nk = E.obsToStateKey(nextObs, action);
      var nrow = ensureQRow(this.q, nk, Q_MAX);
      nextQ = nrow[nextAction];
    }
    var target = reward + (done ? 0 : GAMMA * nextQ);
    row[action] += this.alpha * (target - row[action]);
  };

  SarsaBrain.prototype.toDict = function () {
    var d = QBrain.prototype.toDict.call(this);
    d.type = "sarsa";
    return d;
  };

  /* --- MENACE beads --- */
  function MenaceBrain() {
    this.boxes = new Map();
    this._prevDir = null;
  }

  MenaceBrain.prototype._box = function (key) {
    var b = lruTouch(this.boxes, key, MENACE_MAX);
    if (b) {
      return b;
    }
    b = [3, 3, 3, 3];
    this.boxes.set(key, b);
    return b;
  };

  MenaceBrain.prototype.chooseAction = function (obs, legal, explore) {
    if (!legal.length) {
      return packDecision(0);
    }
    var key = E.obsToStateKey(obs, this._prevDir);
    var box = this._box(key);
    var i;
    var action;
    if (!explore) {
      var best = -1;
      var picks = [];
      for (i = 0; i < legal.length; i += 1) {
        var a = legal[i];
        if (box[a] > best) {
          best = box[a];
          picks = [a];
        } else if (box[a] === best) {
          picks.push(a);
        }
      }
      action = picks[Math.floor(Math.random() * picks.length)];
    } else {
      var total = 0;
      for (i = 0; i < legal.length; i += 1) {
        total += Math.max(1, box[legal[i]]);
      }
      var r = Math.random() * total;
      var acc = 0;
      action = legal[0];
      for (i = 0; i < legal.length; i += 1) {
        acc += Math.max(1, box[legal[i]]);
        if (r <= acc) {
          action = legal[i];
          break;
        }
      }
    }
    return packDecision(action);
  };

  MenaceBrain.prototype.learnEpisode = function (traj, success) {
    var delta = success ? 3 : -1;
    var i;
    for (i = 0; i < traj.length; i += 1) {
      var step = traj[i];
      var box = this._box(step.key);
      box[step.action] = Math.max(1, box[step.action] + delta);
    }
  };

  /* Per-step bead nudge from immediate reward; episode update still applies. */
  MenaceBrain.prototype.learnTransition = function (obs, action, reward) {
    var key = E.obsToStateKey(obs, this._prevDir);
    var box = this._box(key);
    var delta = 0;
    if (reward <= -15) {
      delta = -2;
    } else if (reward < 0) {
      delta = -1;
    } else if (reward >= 40) {
      delta = 2;
    } else if (reward > 0) {
      delta = 1;
    }
    if (delta !== 0) {
      box[action] = Math.max(1, box[action] + delta);
    }
  };

  MenaceBrain.prototype.toDict = function () {
    var entries = [];
    this.boxes.forEach(function (v, k) {
      entries.push([k, v.slice()]);
    });
    return { type: "menace", entries: entries };
  };

  MenaceBrain.prototype.fromDict = function (d) {
    this.boxes = new Map();
    var i;
    if (d && d.entries) {
      for (i = 0; i < d.entries.length; i += 1) {
        this.boxes.set(d.entries[i][0], d.entries[i][1].slice());
      }
    }
  };

  function GeneticMenaceAgent() {
    MenaceBrain.call(this);
    this.population = [];
    var i;
    for (i = 0; i < 8; i += 1) {
      this.population.push(new MenaceBrain());
    }
    this.best = this.population[0];
  }
  GeneticMenaceAgent.prototype = Object.create(MenaceBrain.prototype);
  GeneticMenaceAgent.prototype.constructor = GeneticMenaceAgent;

  GeneticMenaceAgent.prototype.chooseAction = function (obs, legal, explore) {
    if (this.best) {
      this.best._prevDir = this._prevDir;
    }
    return this.best.chooseAction(obs, legal, explore);
  };

  GeneticMenaceAgent.prototype.learnEpisode = function (traj, success) {
    if (this.best && this.best.learnEpisode) {
      this.best.learnEpisode(traj, success);
    }
  };

  GeneticMenaceAgent.prototype.learnTransition = function (obs, action, reward, nextObs, done) {
    if (this.best) {
      this.best._prevDir = this._prevDir;
      this.best.learnTransition(obs, action, reward, nextObs, done);
    }
  };

  GeneticMenaceAgent.prototype._ensurePopulation = function () {
    var target = 8;
    if (!this.population) {
      this.population = [];
    }
    if (!this.best) {
      this.best = new MenaceBrain();
    }
    if (!this.population.length) {
      this.population.push(this.best);
    } else if (this.population.indexOf(this.best) < 0) {
      this.population[0] = this.best;
    }
    while (this.population.length < target) {
      var child = new MenaceBrain();
      child.fromDict(this.best.toDict());
      this.population.push(child);
    }
  };

  GeneticMenaceAgent.prototype.evolve = function (fitnessFn) {
    this._ensurePopulation();
    var scored = [];
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      scored.push({ brain: this.population[i], fit: fitnessFn(this.population[i]) });
    }
    if (!scored.length) {
      return;
    }
    scored.sort(compareEvolveFit);
    this.best = scored[0].brain;
    var next = [scored[0].brain];
    if (scored.length > 1) {
      next.push(scored[1].brain);
    } else {
      var twin = new MenaceBrain();
      twin.fromDict(scored[0].brain.toDict());
      next.push(twin);
    }
    var parentN = Math.min(2, scored.length);
    while (next.length < 8) {
      var child = new MenaceBrain();
      child.fromDict(scored[Math.floor(Math.random() * parentN)].brain.toDict());
      next.push(child);
    }
    this.population = next;
  };

  GeneticMenaceAgent.prototype.toDict = function () {
    return { type: "genetic_menace", best: this.best.toDict() };
  };

  GeneticMenaceAgent.prototype.fromDict = function (d) {
    this.best = new MenaceBrain();
    if (d && d.best) {
      this.best.fromDict(d.best);
    }
    this.population = [this.best];
    this._ensurePopulation();
  };

  /* --- NN Q / policy --- */
  function NnAgent(layerSizes, learningRate) {
    this.net = new NN.NeuralNet(layerSizes || C.NN_POLICY_LAYER_SIZES[2], learningRate || 0.03);
    this.learningRate = this.net.learningRate;
    this.gamesPlayed = 0;
  }

  NnAgent.prototype.chooseAction = function (obs, legal, explore) {
    var out = this.net.getOutputs(obs);
    var self = this;
    return decisionFromOutputs(out, legal, explore, function (o, leg, ex) {
      if (ex && Math.random() < Math.max(0.05, 0.25 - self.gamesPlayed / 3000 * 0.2)) {
        return NN.sampleSoftmax(o, leg, 1);
      }
      return NN.argmaxLegal(o, leg, false);
    });
  };

  NnAgent.prototype.learnTransition = function (obs, action, reward, nextObs, done) {
    var maxNext = 0;
    if (!done) {
      var nout = this.net.getOutputs(nextObs);
      maxNext = maxQLegal(nout, nextObs);
    }
    var target = reward + (done ? 0 : GAMMA * maxNext);
    this.net.trainOnAction(obs, action, target);
  };

  NnAgent.prototype.toDict = function () {
    return { type: "nn", net: this.net.toJSON(), gamesPlayed: this.gamesPlayed };
  };

  NnAgent.prototype.fromDict = function (d) {
    if (d && d.net) {
      this.net.load(d.net);
    }
    this.gamesPlayed = (d && d.gamesPlayed) || 0;
  };

  /* Single-output value nets: range decode to N/E/S/W (bin centers). */
  var VALUE_DIR_CENTERS = [-0.75, -0.25, 0.25, 0.75];

  function dirFromScalar(u) {
    var x = typeof u === "number" && isFinite(u) ? u : 0;
    if (x < -0.5) {
      return 0;
    }
    if (x < 0) {
      return 1;
    }
    if (x < 0.5) {
      return 2;
    }
    return 3;
  }

  function scalarCenterForDir(action) {
    var a = action | 0;
    if (a < 0 || a > 3) {
      return 0;
    }
    return VALUE_DIR_CENTERS[a];
  }

  /** Prefer range decode; if illegal, nearest legal bin by |u - center|. */
  function legalDirFromScalar(u, legal) {
    if (!legal || !legal.length) {
      return 0;
    }
    var preferred = dirFromScalar(u);
    var i;
    for (i = 0; i < legal.length; i += 1) {
      if (legal[i] === preferred) {
        return preferred;
      }
    }
    var best = legal[0];
    var bestDist = Math.abs(u - scalarCenterForDir(best));
    for (i = 1; i < legal.length; i += 1) {
      var a = legal[i];
      var d = Math.abs(u - scalarCenterForDir(a));
      if (d < bestDist) {
        bestDist = d;
        best = a;
      }
    }
    return best;
  }

  function clampUnit(v) {
    if (typeof v !== "number" || !isFinite(v)) {
      return 0;
    }
    if (v < -1) {
      return -1;
    }
    if (v > 1) {
      return 1;
    }
    return v;
  }

  /**
   * Value nets are normally meant for minimax-style search (score positions).
   * This lab instead maps the single scalar to a direction via fixed ranges so
   * decisions do not peek at env rewards before committing a move.
   */
  function NnValueAgent(layerSizes, learningRate) {
    this.net = new NN.NeuralNet(layerSizes || C.NN_VALUE_LAYER_SIZES[2], learningRate || 0.03);
    this.learningRate = this.net.learningRate;
    this.gamesPlayed = 0;
  }

  NnValueAgent.prototype.chooseAction = function (obs, legal, explore) {
    if (!legal || !legal.length) {
      return packDecision(0);
    }
    if (explore && Math.random() < 0.15) {
      return packDecision(legal[Math.floor(Math.random() * legal.length)]);
    }
    var out = this.net.getOutputs(obs);
    var u = out && typeof out[0] === "number" ? out[0] : 0;
    return packDecision(legalDirFromScalar(u, legal));
  };

  /* After the real step: nudge scalar toward (good) or away from (bad) action bin. */
  NnValueAgent.prototype.learnTransition = function (obs, action, reward) {
    if (typeof reward !== "number" || !isFinite(reward) || reward === 0) {
      return;
    }
    var a = action | 0;
    if (a < 0 || a > 3) {
      return;
    }
    var center = scalarCenterForDir(a);
    var cur = this.net.getOutputs(obs)[0];
    if (typeof cur !== "number" || !isFinite(cur)) {
      cur = 0;
    }
    var alpha = Math.min(0.35, 0.03 * Math.abs(reward));
    var target;
    if (reward > 0) {
      target = cur + alpha * (center - cur);
    } else {
      var delta = center - cur;
      if (Math.abs(delta) < 1e-6) {
        target = cur + (center >= 0 ? -alpha : alpha);
      } else {
        target = cur - alpha * delta;
      }
    }
    this.net.trainOnOutputs(obs, [clampUnit(target)]);
  };

  NnValueAgent.prototype.toDict = function () {
    return { type: "nn_value", net: this.net.toJSON(), gamesPlayed: this.gamesPlayed };
  };

  NnValueAgent.prototype.fromDict = function (d) {
    if (d && d.net) {
      this.net.load(d.net);
    }
    this.gamesPlayed = (d && d.gamesPlayed) || 0;
  };

  function ReinforceAgent(learningRate, layerSizes) {
    NnAgent.call(this, layerSizes || C.NN_POLICY_LAYER_SIZES[2], learningRate || 0.01);
  }
  ReinforceAgent.prototype = Object.create(NnAgent.prototype);
  ReinforceAgent.prototype.constructor = ReinforceAgent;

  ReinforceAgent.prototype.learnEpisode = function (traj) {
    var i;
    var G = 0;
    for (i = traj.length - 1; i >= 0; i -= 1) {
      var step = traj[i];
      G = (typeof step.reward === "number" ? step.reward : 0) + GAMMA * G;
      this.net.trainOnAction(step.obs, step.action, G);
    }
  };

  ReinforceAgent.prototype.learnTransition = function () {};

  function ReinforceValueAgent(layerSizes, learningRate) {
    NnValueAgent.call(this, layerSizes, learningRate || 0.01);
  }
  ReinforceValueAgent.prototype = Object.create(NnValueAgent.prototype);
  ReinforceValueAgent.prototype.constructor = ReinforceValueAgent;

  /* --- NEAT-lite: mutate NN weights --- */
  function SimpleGaNeatAgent(layerSizes) {
    var sizes = layerSizes && layerSizes.length ?
      layerSizes.slice() : [C.OBS_SIZE, 16, C.ACTIONS];
    this.net = new NN.NeuralNet(sizes, 0.03);
    this.learningRate = 0.03;
    this.pop = [];
    var i;
    for (i = 0; i < 6; i += 1) {
      var n = this.net.clone();
      if (i) {
        n.mutate(0.1);
      }
      this.pop.push(n);
    }
  }

  SimpleGaNeatAgent.prototype.chooseAction = function (obs, legal, explore) {
    var out = this.net.getOutputs(obs);
    return decisionFromOutputs(out, legal, explore, function (o, leg, ex) {
      if (ex && Math.random() < 0.15) {
        return leg[Math.floor(Math.random() * leg.length)];
      }
      return NN.argmaxLegal(o, leg, false);
    });
  };

  SimpleGaNeatAgent.prototype.learnTransition = function () {};

  SimpleGaNeatAgent.prototype.evolve = function (fitnessFn) {
    var scored = [];
    var i;
    for (i = 0; i < this.pop.length; i += 1) {
      var saved = this.net;
      this.net = this.pop[i];
      scored.push({ net: this.pop[i], fit: fitnessFn(this) });
      this.net = saved;
    }
    if (!scored.length) {
      return;
    }
    scored.sort(compareEvolveFit);
    this.net = scored[0].net.clone();
    var next = [scored[0].net.clone()];
    if (scored.length > 1) {
      next.push(scored[1].net.clone());
    } else {
      next.push(scored[0].net.clone());
    }
    var targetPop = Math.max(6, this.pop.length || 0);
    while (next.length < targetPop) {
      var child = scored[0].net.clone();
      child.mutate(0.08);
      next.push(child);
    }
    this.pop = next;
  };

  SimpleGaNeatAgent.prototype.toDict = function () {
    return { type: "neat", net: this.net.toJSON() };
  };

  SimpleGaNeatAgent.prototype.fromDict = function (d) {
    if (d && d.net) {
      this.net.load(d.net);
    }
    if (!this.pop || !this.pop.length) {
      this.pop = [];
      var i;
      for (i = 0; i < 6; i += 1) {
        var n = this.net.clone();
        if (i) {
          n.mutate(0.1);
        }
        this.pop.push(n);
      }
    } else {
      this.pop[0] = this.net.clone();
    }
  };

  function SimpleGaNeatValueAgent(layerSizes) {
    var sizes = layerSizes && layerSizes.length ?
      layerSizes.slice() : [C.OBS_SIZE, 16, 1];
    this.net = new NN.NeuralNet(sizes, 0.03);
    this.learningRate = 0.03;
    this.pop = [];
    var i;
    for (i = 0; i < 6; i += 1) {
      var n = this.net.clone();
      if (i) {
        n.mutate(0.1);
      }
      this.pop.push(n);
    }
  }

  SimpleGaNeatValueAgent.prototype.chooseAction = NnValueAgent.prototype.chooseAction;
  SimpleGaNeatValueAgent.prototype.learnTransition = function () {};
  SimpleGaNeatValueAgent.prototype.evolve = SimpleGaNeatAgent.prototype.evolve;
  SimpleGaNeatValueAgent.prototype.toDict = function () {
    return { type: "neat_value", net: this.net.toJSON() };
  };
  SimpleGaNeatValueAgent.prototype.fromDict = function (d) {
    if (d && d.net) {
      this.net.load(d.net);
    }
  };

  /* --- TF.js DQN --- */
  function TfjsDqnAgent(layerSizes, learningRate) {
    this.layerSizes = (layerSizes || C.TFJS_DQN_SIZES).slice();
    this.learningRate = typeof learningRate === "number" ? learningRate : 0.001;
    this.gamma = GAMMA;
    this.memory = [];
    this.memoryCap = 50000;
    this.learnSteps = 0;
    this.gamesPlayed = 0;
    this.model = null;
    this.targetModel = null;
    this.optimizer = null;
    this._build();
  }

  TfjsDqnAgent.prototype._build = function () {
    var tf = global.tf;
    if (!tf) {
      this.fallback = new NnAgent(this.layerSizes, this.learningRate);
      return;
    }
    var sizes = this.layerSizes;
    function makeModel() {
      var model = tf.sequential();
      var i;
      for (i = 1; i < sizes.length - 1; i += 1) {
        model.add(tf.layers.dense({
          units: sizes[i],
          activation: "relu",
          inputShape: i === 1 ? [sizes[0]] : undefined
        }));
      }
      model.add(tf.layers.dense({ units: sizes[sizes.length - 1] }));
      return model;
    }
    this.model = makeModel();
    this.targetModel = makeModel();
    this._copyWeights();
    this.optimizer = tf.train.adam(this.learningRate);
  };

  TfjsDqnAgent.prototype._copyWeights = function () {
    if (!this.model || !this.targetModel) {
      return;
    }
    var w = this.model.getWeights();
    var copies = w.map(function (t) {
      return t.clone();
    });
    this.targetModel.setWeights(copies);
    copies.forEach(function (t) {
      t.dispose();
    });
  };

  TfjsDqnAgent.prototype.chooseAction = function (obs, legal, explore) {
    if (this.fallback) {
      return this.fallback.chooseAction(obs, legal, explore);
    }
    if (!legal || !legal.length) {
      return packDecision(0);
    }
    var eps = Math.max(0.05, 1.0 - this.gamesPlayed / 500);
    var tf = global.tf;
    var self = this;
    var q = null;
    if (this.model && tf) {
      try {
        tf.tidy(function () {
          var t = tf.tensor2d([obs]);
          q = Array.prototype.slice.call(self.model.predict(t).dataSync());
        });
      } catch (err) {
        q = null;
      }
    }
    if (explore && Math.random() < eps) {
      return packDecision(legal[Math.floor(Math.random() * legal.length)]);
    }
    if (!q) {
      return packDecision(legal[Math.floor(Math.random() * legal.length)]);
    }
    return decisionFromOutputs(q, legal, false);
  };

  TfjsDqnAgent.prototype.remember = function (obs, action, reward, nextObs, done) {
    this.memory.push({
      obs: obs.slice(),
      action: action,
      reward: reward,
      nextObs: nextObs.slice(),
      done: done
    });
    if (this.memory.length > this.memoryCap) {
      this.memory.shift();
    }
  };

  TfjsDqnAgent.prototype.learnTransition = function (obs, action, reward, nextObs, done) {
    if (this.fallback) {
      this.fallback.learnTransition(obs, action, reward, nextObs, done);
      return;
    }
    this.remember(obs, action, reward, nextObs, done);
    /* Learn sparsely - full Adam every step freezes Shape Search rAF. */
    if (this.memory.length >= 32 && (this.memory.length % 4 === 0)) {
      this.learnBatch(32);
    }
  };

  TfjsDqnAgent.prototype.learnBatch = function (batchSize) {
    if (this.fallback || this.memory.length < batchSize) {
      return;
    }
    var tf = global.tf;
    var batch = [];
    var i;
    for (i = 0; i < batchSize; i += 1) {
      batch.push(this.memory[Math.floor(Math.random() * this.memory.length)]);
    }
    var states = batch.map(function (b) {
      return b.obs;
    });
    var nextStates = batch.map(function (b) {
      return b.nextObs;
    });
    var self = this;
    var qNext = tf.tidy(function () {
      return self.targetModel.predict(tf.tensor2d(nextStates)).arraySync();
    });
    var targets = tf.tidy(function () {
      return self.model.predict(tf.tensor2d(states)).arraySync();
    });
    for (i = 0; i < batchSize; i += 1) {
      var b = batch[i];
      var maxN = maxQLegal(qNext[i], b.nextObs);
      targets[i][b.action] = b.reward + (b.done ? 0 : self.gamma * maxN);
    }
    var xs = tf.tensor2d(states);
    var ys = tf.tensor2d(targets);
    this.optimizer.minimize(function () {
      var pred = self.model.predict(xs);
      var loss = tf.losses.absoluteDifference(ys, pred);
      pred.dispose();
      return loss.mean();
    });
    xs.dispose();
    ys.dispose();
    this.learnSteps += 1;
    if (this.learnSteps % 200 === 0) {
      this._copyWeights();
    }
  };

  TfjsDqnAgent.prototype.toDict = function () {
    if (this.fallback) {
      return { type: "tfjs", fallback: this.fallback.toDict(), layerSizes: this.layerSizes };
    }
    return {
      type: "tfjs",
      layerSizes: this.layerSizes,
      learningRate: this.learningRate,
      gamesPlayed: this.gamesPlayed,
      weights: this.model.getWeights().map(function (w) {
        return { shape: w.shape, data: Array.from(w.dataSync()) };
      })
    };
  };

  TfjsDqnAgent.prototype.fromDict = function (d) {
    if (!d) {
      return;
    }
    if (d.fallback && this.fallback) {
      this.fallback.fromDict(d.fallback);
      return;
    }
    if (d.weights && this.model && global.tf) {
      var tf = global.tf;
      var tensors = d.weights.map(function (w) {
        return tf.tensor(w.data, w.shape);
      });
      this.model.setWeights(tensors);
      this._copyWeights();
      tensors.forEach(function (t) {
        t.dispose();
      });
    }
    this.gamesPlayed = d.gamesPlayed || 0;
  };

  function TfjsValueAgent(layerSizes, learningRate) {
    this.inner = new NnValueAgent(layerSizes || C.TFJS_VALUE_SIZES, learningRate || 0.001);
    this.learningRate = this.inner.learningRate;
  }
  TfjsValueAgent.prototype.chooseAction = function (obs, legal, explore, env) {
    return this.inner.chooseAction(obs, legal, explore, env);
  };
  TfjsValueAgent.prototype.learnTransition = function (obs, a, r, n, d) {
    this.inner.learnTransition(obs, a, r, n, d);
  };
  TfjsValueAgent.prototype.toDict = function () {
    return { type: "tfjs_value", inner: this.inner.toDict() };
  };
  TfjsValueAgent.prototype.fromDict = function (d) {
    if (d && d.inner) {
      this.inner.fromDict(d.inner);
    }
  };

  global.SNAKE_AGENTS = {
    QBrain: QBrain,
    SarsaBrain: SarsaBrain,
    MenaceBrain: MenaceBrain,
    GeneticMenaceAgent: GeneticMenaceAgent,
    NnAgent: NnAgent,
    NnValueAgent: NnValueAgent,
    ReinforceAgent: ReinforceAgent,
    ReinforceValueAgent: ReinforceValueAgent,
    SimpleGaNeatAgent: SimpleGaNeatAgent,
    SimpleGaNeatValueAgent: SimpleGaNeatValueAgent,
    TfjsDqnAgent: TfjsDqnAgent,
    TfjsValueAgent: TfjsValueAgent,
    dirFromScalar: dirFromScalar,
    scalarCenterForDir: scalarCenterForDir,
    legalDirFromScalar: legalDirFromScalar,
    GAMMA: GAMMA
  };
})(typeof window !== "undefined" ? window : global);
