/**
 * Snake AI Shape Search engine: create agents, train/eval episodes, tournaments.
 */
(function (global) {
  "use strict";

  var C = global.SNAKE_CONSTANTS;
  var E = global.SNAKE_ENV;
  var A = global.SNAKE_AGENTS;

  var SHAPE_FAMILIES = {
    nn_policy: {
      kind: "nn",
      builder: function (sizes, lr) {
        return new A.NnAgent(sizes, lr);
      }
    },
    nn_value: {
      kind: "nn",
      builder: function (sizes, lr) {
        return new A.NnValueAgent(sizes, lr);
      }
    },
    reinforce_policy: {
      kind: "reinforce",
      builder: function (sizes, lr) {
        return new A.ReinforceAgent(lr, sizes);
      }
    },
    reinforce_value: {
      kind: "nn",
      builder: function (sizes, lr) {
        return new A.ReinforceValueAgent(sizes, lr);
      }
    },
    tfjs_policy: {
      kind: "tfjs",
      builder: function (sizes, lr) {
        return new A.TfjsDqnAgent(sizes || C.TFJS_DQN_SIZES, lr);
      }
    },
    tfjs_value: {
      kind: "nn",
      builder: function (sizes, lr) {
        return new A.TfjsValueAgent(sizes, lr);
      }
    },
    qtable: {
      kind: "tabular",
      builder: function () {
        return new A.QBrain();
      }
    },
    sarsa: {
      kind: "tabular",
      builder: function () {
        return new A.SarsaBrain();
      }
    },
    menace: {
      kind: "menace",
      builder: function () {
        return new A.MenaceBrain();
      }
    },
    neat: {
      kind: "neat",
      builder: function (sizes) {
        return new A.SimpleGaNeatAgent(sizes);
      }
    },
    neat_value: {
      kind: "neat",
      builder: function (sizes) {
        return new A.SimpleGaNeatValueAgent(sizes);
      }
    },
    genetic_menace: {
      kind: "genetic",
      builder: function () {
        return new A.GeneticMenaceAgent();
      }
    }
  };

  var hardMode = false;
  var compassOn = true;

  function isHardMode() {
    return !!hardMode;
  }

  function setHardMode(on) {
    hardMode = !!on;
  }

  function isCompassOn() {
    return !!compassOn;
  }

  function setCompassOn(on) {
    compassOn = on !== false;
  }

  function applyHardOutputSize(sizes) {
    return sizes ? sizes.slice() : sizes;
  }

  function withInputDim(sizes, inputDim) {
    if (!sizes) {
      return null;
    }
    var out = sizes.slice();
    if (typeof inputDim === "number" && inputDim > 0) {
      out[0] = inputDim;
    }
    return out;
  }

  function defaultLayerSizes(family, visionRadius, compass) {
    var inp = C.obsSizeForRadius(
      typeof visionRadius === "number" ? visionRadius : C.DEFAULT_VISION,
      typeof compass === "boolean" ? compass : compassOn
    );
    var sizes = null;
    if (family === "nn_policy" || family === "reinforce_policy") {
      sizes = C.NN_POLICY_LAYER_SIZES[2].slice();
    } else if (family === "nn_value" || family === "reinforce_value") {
      sizes = C.NN_VALUE_LAYER_SIZES[2].slice();
    } else if (family === "tfjs_policy") {
      sizes = (C.TFJS_DQN_LAYER_SIZES && C.TFJS_DQN_LAYER_SIZES[64]) ?
        C.TFJS_DQN_LAYER_SIZES[64].slice() : C.TFJS_DQN_SIZES.slice();
    } else     if (family === "tfjs_value") {
      sizes = (C.TFJS_VALUE_LAYER_SIZES && C.TFJS_VALUE_LAYER_SIZES[64]) ?
        C.TFJS_VALUE_LAYER_SIZES[64].slice() : C.TFJS_VALUE_SIZES.slice();
    } else if (family === "neat") {
      sizes = [inp, 16, C.ACTIONS];
    } else if (family === "neat_value") {
      sizes = [inp, 16, 1];
    }
    if (sizes) {
      return withInputDim(applyHardOutputSize(sizes), inp);
    }
    return null;
  }

  function defaultLearningRate(family) {
    if (family === "tfjs_policy" || family === "tfjs_value") {
      return 0.001;
    }
    if (family === "reinforce_policy" || family === "reinforce_value") {
      return 0.01;
    }
    if (family === "nn_policy" || family === "nn_value") {
      return 0.03;
    }
    return null;
  }

  function createShapeAgent(family, layerSizes, learningRate, visionRadius) {
    var meta = SHAPE_FAMILIES[family];
    if (!meta) {
      throw new Error("Unknown shape family: " + family);
    }
    var vr = typeof visionRadius === "number" ? visionRadius : C.DEFAULT_VISION;
    var sizes = layerSizes ?
      withInputDim(applyHardOutputSize(layerSizes), C.obsSizeForRadius(vr, compassOn)) :
      defaultLayerSizes(family, vr, compassOn);
    var lr = typeof learningRate === "number" ? learningRate : defaultLearningRate(family);
    var agent = meta.builder(sizes, lr);
    if (agent) {
      agent.visionRadius = vr;
      if (typeof lr === "number") {
        agent.learningRate = lr;
        if (agent.net) {
          agent.net.learningRate = lr;
        }
      }
    }
    return agent;
  }

  function shapeAgentToDict(agent, family) {
    if (!agent || !agent.toDict) {
      return null;
    }
    var d = agent.toDict();
    d.family = family;
    return d;
  }

  function shapeAgentFromDict(family, dict, layerSizes, learningRate, visionRadius) {
    var vr = typeof visionRadius === "number" ? visionRadius :
      (dict && typeof dict.visionRadius === "number" ? dict.visionRadius : C.DEFAULT_VISION);
    var agent = createShapeAgent(family, layerSizes, learningRate, vr);
    if (dict && agent.fromDict) {
      agent.fromDict(dict);
    }
    agent.visionRadius = vr;
    return agent;
  }

  function unwrapDecision(decision) {
    if (decision && typeof decision === "object" && typeof decision.action === "number") {
      return { action: decision.action | 0 };
    }
    return { action: typeof decision === "number" ? (decision | 0) : -1 };
  }

  function normalizeAction(action, legal) {
    var a = typeof action === "number" ? (action | 0) : -1;
    if (a >= 0 && a <= 3) {
      return a;
    }
    if (legal && legal.length) {
      return legal[Math.floor(Math.random() * legal.length)];
    }
    return 0;
  }

  function resolveVision(agent, opts) {
    opts = opts || {};
    if (typeof opts.visionRadius === "number") {
      return opts.visionRadius;
    }
    if (agent && typeof agent.visionRadius === "number") {
      return agent.visionRadius;
    }
    return C.DEFAULT_VISION;
  }

  function chooseDecision(agent, family, env, explore) {
    var obs = env.observe();
    var legal = env.legalActions();
    if (!legal.length) {
      legal = [0, 1, 2, 3];
    }
    var raw;
    if (family === "random") {
      raw = legal[Math.floor(Math.random() * legal.length)];
    } else if (family === "bfs") {
      var ba = E.bfsNextAction(env);
      raw = ba === null ? legal[0] : ba;
    } else if (family === "greedy") {
      raw = E.greedyNextAction(env);
    } else if (family === "wall_follower") {
      raw = E.wallFollowerNextAction(env);
    } else if (family === "avoid_pocket") {
      raw = E.avoidPocketNextAction(env);
    } else if (family === "master") {
      raw = E.masterNextAction(env);
    } else if (family === "hamilton") {
      raw = E.masterHamiltonAction(env);
    } else if (family === "hamilton_aggr") {
      raw = E.hamiltonAggrNextAction(env);
    } else if (family === "cell_tree") {
      raw = E.cellTreeNextAction(env);
    } else if (family === "cell_tree_odd" || family === "cell_tree_fringe") {
      raw = E.cellTreeOddNextAction(env);
    } else if (family === "cell_tree_odd_lock") {
      /* Removed from Observe; treat as Cell Tree Odd. */
      raw = E.cellTreeOddNextAction(env);
    } else if (family === "dhcr") {
      raw = E.dhcrNextAction(env);
    } else if (family === "dhcr_improved") {
      raw = E.dhcrImprovedNextAction(env);
    } else if (family === "plan_ahead") {
      raw = E.planAheadReenterNextAction(env);
    } else if (family === "tail_safe") {
      raw = E.tailSafeNextAction(env);
    } else if (family === "tail_safe_hybrid") {
      raw = E.tailSafeHybridNextAction(env);
    } else if (!agent || typeof agent.chooseAction !== "function") {
      raw = legal[Math.floor(Math.random() * legal.length)];
    } else {
      raw = agent.chooseAction(obs, legal, !!explore, env);
    }
    var dec = unwrapDecision(raw);
    dec.action = normalizeAction(dec.action, legal);
    return dec;
  }

  function chooseAction(agent, family, env, explore) {
    return chooseDecision(agent, family, env, explore).action;
  }

  function beginEpisodePrevDir(agent) {
    if (agent) {
      agent._prevDir = null;
    }
  }

  function advancePrevDir(agent, action) {
    if (agent) {
      agent._prevDir = action;
    }
  }

  /**
   * Shared train seed pool (FIFO, max 1000). Same seed => same food stream.
   */
  var mazePoolMax = 1000;
  var mazePoolItems = [];
  var mazePoolNextId = 1;

  function mazePoolClear() {
    mazePoolItems = [];
    mazePoolNextId = 1;
  }

  function mazePoolCount() {
    return mazePoolItems.length;
  }

  function mazePoolGetMax() {
    return mazePoolMax;
  }

  function mazePoolSetMax(n) {
    if (typeof n === "number" && n >= 1) {
      mazePoolMax = n | 0;
      while (mazePoolItems.length > mazePoolMax) {
        mazePoolItems.shift();
      }
    }
  }

  function mazePoolPush(entry) {
    mazePoolItems.push(entry);
    while (mazePoolItems.length > mazePoolMax) {
      mazePoolItems.shift();
    }
  }

  function mazePoolGenerate(size, seed) {
    var s = typeof seed === "number" ? (seed >>> 0) : ((Math.random() * 0x7fffffff) | 0);
    if (!s) {
      s = 1;
    }
    var entry = {
      id: mazePoolNextId,
      size: size,
      seed: s,
      grid: null,
      stats: { solutionLen: 1 }
    };
    mazePoolNextId += 1;
    mazePoolPush(entry);
    return entry;
  }

  function mazePoolNext(size, afterId) {
    var minId = typeof afterId === "number" ? afterId : 0;
    var i;
    for (i = 0; i < mazePoolItems.length; i += 1) {
      var m = mazePoolItems[i];
      if (m.size === size && m.id > minId) {
        return m;
      }
    }
    return mazePoolGenerate(size);
  }

  function mazePoolGetOrCreateSeed(size, seed) {
    var i;
    for (i = 0; i < mazePoolItems.length; i += 1) {
      if (mazePoolItems[i].size === size && mazePoolItems[i].seed === seed) {
        return mazePoolItems[i];
      }
    }
    return mazePoolGenerate(size, seed);
  }

  /**
   * Run one episode. Returns { success, steps, return, traj, apples }.
   */
  function runEpisode(agent, family, env, opts) {
    opts = opts || {};
    var explore = !!opts.explore;
    var train = !!opts.train;
    var seed = opts.seed;
    var size = opts.size;
    var vr = resolveVision(agent, opts);
    var comp = typeof opts.compassOn === "boolean" ? opts.compassOn : compassOn;
    if (env.setVision) {
      env.setVision(vr, comp);
    } else {
      env.visionRadius = vr;
      env.compassOn = comp;
    }
    env.reset({ seed: seed, size: size, visionRadius: vr, compassOn: comp });
    beginEpisodePrevDir(agent);
    var traj = [];
    var totalReturn = 0;
    var success = false;
    var meta = SHAPE_FAMILIES[family];
    var isSarsa = family === "sarsa";
    var isMenace = meta && (meta.kind === "menace" || meta.kind === "genetic");
    var learnTd = train && meta && meta.kind !== "neat" && meta.kind !== "reinforce";
    var obs = env.observe();
    var decision = chooseDecision(agent, family, env, explore);
    var action = decision.action;

    while (true) {
      var prevDirAtChoice = agent ? agent._prevDir : null;
      var step = env.step(action);
      totalReturn += step.reward;
      traj.push({
        obs: obs.slice(),
        action: action,
        reward: step.reward,
        nextObs: step.obs.slice(),
        done: !!step.done,
        prevDir: prevDirAtChoice,
        key: E.obsToStateKey(obs, prevDirAtChoice)
      });

      if (learnTd && agent && agent.learnTransition) {
        if (isSarsa) {
          var nextAct = 0;
          var nextDec = null;
          if (!step.done) {
            advancePrevDir(agent, action);
            nextDec = chooseDecision(agent, family, env, explore);
            nextAct = nextDec.action;
            agent._prevDir = prevDirAtChoice;
          }
          agent.learnTransition(obs, action, step.reward, step.obs, step.done, nextAct);
          advancePrevDir(agent, action);
          if (!step.done && nextDec) {
            action = nextDec.action;
          }
        } else {
          agent.learnTransition(obs, action, step.reward, step.obs, step.done);
          advancePrevDir(agent, action);
        }
      } else {
        advancePrevDir(agent, action);
      }

      if (step.done) {
        success = !!step.info.success;
        break;
      }
      obs = step.obs;
      if (!isSarsa || !learnTd) {
        decision = chooseDecision(agent, family, env, explore);
        action = decision.action;
      }
    }

    if (train) {
      if (meta && meta.kind === "reinforce" && agent.learnEpisode) {
        agent.learnEpisode(traj, totalReturn);
      }
      if (isMenace && agent.learnEpisode) {
        agent.learnEpisode(traj, success);
      }
      if (agent && agent.gamesPlayed !== undefined) {
        agent.gamesPlayed += 1;
      }
    }
    return {
      success: success,
      steps: env.steps,
      return: totalReturn,
      traj: traj,
      apples: env.apples || 0,
      explored: E.countVisited(env)
    };
  }

  /** Train for up to budgetMs wall clock at given board size. */
  function shapeTrainEpisodes(agent, family, size, budgetMs, opts) {
    opts = opts || {};
    var env = opts.env || new E.SnakeEnv({ size: size });
    var explore = opts.explore !== false;
    var t0 = performance.now();
    var episodes = 0;
    var successes = 0;
    var nemesisSeeds = opts.nemesisSeeds || [];
    var afterId = typeof opts.poolAfterId === "number" ? opts.poolAfterId : 0;
    var usePool = opts.usePool !== false;
    var vr = resolveVision(agent, opts);
    while (performance.now() - t0 < budgetMs) {
      var maze = null;
      var seed;
      if (usePool) {
        if (nemesisSeeds.length && Math.random() < 0.35) {
          seed = nemesisSeeds[Math.floor(Math.random() * nemesisSeeds.length)];
          maze = mazePoolGetOrCreateSeed(size, seed);
        } else {
          maze = mazePoolNext(size, afterId);
          afterId = maze.id;
        }
      } else {
        seed = (Math.random() * 0x7fffffff) | 0;
      }
      var result = runEpisode(agent, family, env, {
        explore: explore,
        train: true,
        seed: maze ? maze.seed : seed,
        size: size,
        visionRadius: vr
      });
      episodes += 1;
      if (result.success) {
        successes += 1;
      }
    }
    return {
      episodes: episodes,
      successes: successes,
      ms: performance.now() - t0,
      poolAfterId: afterId
    };
  }

  var TEACH_REWARD = 0.5;
  var TEACH_PUNISH = -0.5;

  function isTeachableFamily(family) {
    var meta = SHAPE_FAMILIES[family];
    if (!meta) {
      return false;
    }
    return meta.kind !== "neat";
  }

  function visionRadiusOf(ind) {
    if (!ind) {
      return C.DEFAULT_VISION;
    }
    if (typeof ind.visionRadius === "number") {
      return ind.visionRadius;
    }
    if (ind.agent && typeof ind.agent.visionRadius === "number") {
      return ind.agent.visionRadius;
    }
    return C.DEFAULT_VISION;
  }

  function countTeachableInGroup(peers) {
    var n = 0;
    var i;
    for (i = 0; i < peers.length; i += 1) {
      if (peers[i] && isTeachableFamily(peers[i].family)) {
        n += 1;
      }
    }
    return n;
  }

  /**
   * Race one vision-radius peer group on a shared seed; winner teaches other
   * online learners in that same r-family only (obs dims must match).
   * When opts.learnFromMaster, baselines from opts.masters / opts.makeMasters
   * also race and may teach (they never receive teach updates).
   */
  function shapePeerTeachPass(population, size, opts) {
    opts = opts || {};
    var afterId = typeof opts.poolAfterId === "number" ? opts.poolAfterId : 0;
    var learnFromMaster = !!opts.learnFromMaster;
    if (!population || !population.length) {
      return { poolAfterId: afterId, teacherId: null, students: 0, steps: 0, visionRadius: null };
    }
    var byR = {};
    var i;
    for (i = 0; i < population.length; i += 1) {
      var vrKey = String(visionRadiusOf(population[i]));
      if (!byR[vrKey]) {
        byR[vrKey] = [];
      }
      byR[vrKey].push(population[i]);
    }
    var eligible = [];
    Object.keys(byR).forEach(function (k) {
      var group = byR[k];
      var teachable = countTeachableInGroup(group);
      if (learnFromMaster) {
        if (teachable >= 1) {
          eligible.push(k);
        }
      } else if (group.length >= 2) {
        eligible.push(k);
      }
    });
    if (!eligible.length) {
      return { poolAfterId: afterId, teacherId: null, students: 0, steps: 0, visionRadius: null };
    }
    var pick = eligible[Math.floor(Math.random() * eligible.length)];
    var peers = byR[pick];
    var teachVr = parseInt(pick, 10);

    var masters = [];
    if (learnFromMaster) {
      if (typeof opts.makeMasters === "function") {
        masters = opts.makeMasters(teachVr) || [];
      } else if (opts.masters && opts.masters.length) {
        masters = opts.masters;
      }
    }
    var racers = peers.slice();
    for (i = 0; i < masters.length; i += 1) {
      var m = masters[i];
      if (!m) {
        continue;
      }
      racers.push({
        id: m.id,
        family: m.family,
        label: m.label,
        agent: m.agent,
        visionRadius: teachVr
      });
    }

    var maze = mazePoolNext(size, afterId);
    afterId = maze.id;
    var env = opts.env || new E.SnakeEnv({ size: size });
    var raceOpts = {
      explore: false,
      train: false,
      seed: maze.seed,
      size: size,
      visionRadius: teachVr
    };
    var best = null;
    var bestResult = null;
    for (i = 0; i < racers.length; i += 1) {
      var cand = racers[i];
      var r = runEpisode(cand.agent, cand.family, env, raceOpts);
      if (!bestResult) {
        best = cand;
        bestResult = r;
        continue;
      }
      var better = false;
      if (r.success && !bestResult.success) {
        better = true;
      } else if (r.success === bestResult.success) {
        if (r.steps < bestResult.steps) {
          better = true;
        } else if (r.steps === bestResult.steps && r.return > bestResult.return) {
          better = true;
        } else if (r.steps === bestResult.steps && r.return === bestResult.return &&
            String(cand.id) < String(best.id)) {
          better = true;
        }
      }
      if (better) {
        best = cand;
        bestResult = r;
      }
    }
    var traj = bestResult.traj || [];
    var students = 0;
    for (i = 0; i < peers.length; i += 1) {
      var student = peers[i];
      if (student.id === best.id || !isTeachableFamily(student.family)) {
        continue;
      }
      if (visionRadiusOf(student) !== teachVr) {
        continue;
      }
      var agent = student.agent;
      if (!agent) {
        continue;
      }
      var meta = SHAPE_FAMILIES[student.family];
      var isReinforce = meta && meta.kind === "reinforce";
      if (!isReinforce && typeof agent.learnTransition !== "function") {
        continue;
      }
      if (isReinforce && typeof agent.learnEpisode !== "function") {
        continue;
      }
      students += 1;
      beginEpisodePrevDir(agent);
      var synth = [];
      var j;
      for (j = 0; j < traj.length; j += 1) {
        var step = traj[j];
        agent._prevDir = step.prevDir;
        var legal = E.legalFromObs ? E.legalFromObs(step.obs) : [0, 1, 2, 3];
        if (!legal || !legal.length) {
          legal = [0, 1, 2, 3];
        }
        var stubEnv = {
          observe: function () { return step.obs; },
          legalActions: function () { return legal; }
        };
        var dec = chooseDecision(agent, student.family, stubEnv, false);
        var studentAct = dec.action;
        var match = studentAct === step.action;
        var teachR = match ? TEACH_REWARD : TEACH_PUNISH;
        if (isReinforce) {
          synth.push({
            obs: step.obs.slice(),
            action: studentAct,
            reward: teachR
          });
        } else {
          agent.learnTransition(step.obs, studentAct, teachR, step.nextObs, true);
        }
        advancePrevDir(agent, studentAct);
      }
      if (isReinforce && synth.length) {
        agent.learnEpisode(synth);
      }
    }
    return {
      poolAfterId: afterId,
      teacherId: best.id,
      students: students,
      steps: traj.length,
      success: !!bestResult.success,
      visionRadius: teachVr
    };
  }

  function shapeTrainPeer(agentA, familyA, agentB, familyB, size, budgetMs, opts) {
    opts = opts || {};
    var half = budgetMs / 2;
    var a = shapeTrainEpisodes(agentA, familyA, size, half, {
      explore: true,
      poolAfterId: opts.poolAfterIdA,
      nemesisSeeds: opts.nemesisSeedsA
    });
    var b = shapeTrainEpisodes(agentB, familyB, size, half, {
      explore: true,
      poolAfterId: opts.poolAfterIdB,
      nemesisSeeds: opts.nemesisSeedsB
    });
    return { a: a, b: b };
  }

  function shapeEvolveOne(agent, family, size, populationPeers, opts) {
    if (!agent.evolve) {
      return { poolAfterId: opts && opts.poolAfterId || 0 };
    }
    opts = opts || {};
    var env = new E.SnakeEnv({ size: size });
    var afterId = typeof opts.poolAfterId === "number" ? opts.poolAfterId : 0;
    var vr = resolveVision(agent, opts);
    agent.evolve(function (cand) {
      if (cand && typeof agent.visionRadius === "number") {
        cand.visionRadius = agent.visionRadius;
      }
      var wins = 0;
      var stepPctSum = 0;
      var exploredSum = 0;
      var i;
      for (i = 0; i < 10; i += 1) {
        var maze = mazePoolNext(size, afterId);
        afterId = maze.id;
        var r = runEpisode(cand, family, env, {
          explore: true,
          train: family === "genetic_menace" || family === "menace",
          seed: maze.seed,
          size: size,
          visionRadius: vr
        });
        exploredSum += typeof r.explored === "number" ? r.explored : 0;
        if (r.success) {
          wins += 1;
          stepPctSum += r.steps / Math.max(1, env.maxSteps || 1);
        }
      }
      return {
        wins: wins,
        avgStepPct: wins > 0 ? stepPctSum / wins : Number.POSITIVE_INFINITY,
        explored: exploredSum / 10
      };
    });
    return { poolAfterId: afterId };
  }

  /** Greedy eval: success rate over N episodes. */
  function shapeEvalSuccess(agent, family, size, episodes, opts) {
    opts = opts || {};
    var env = opts.env || new E.SnakeEnv({ size: size });
    var wins = 0;
    var totalSteps = 0;
    var totalApples = 0;
    var failedSeeds = [];
    var seeds = [];
    var fixedSeeds = opts.seeds && opts.seeds.length ? opts.seeds : null;
    var vr = resolveVision(agent, opts);
    var i;
    var t0 = performance.now();
    for (i = 0; i < episodes; i += 1) {
      var seed;
      if (fixedSeeds && typeof fixedSeeds[i] === "number") {
        seed = fixedSeeds[i] >>> 0;
      } else if (typeof opts.seedBase === "number") {
        seed = (opts.seedBase + i * 104729) >>> 0;
      } else {
        seed = (Math.random() * 0x7fffffff) | 0;
      }
      seeds.push(seed);
      var r = runEpisode(agent, family, env, {
        explore: false,
        train: false,
        seed: seed,
        size: size,
        visionRadius: vr
      });
      if (r.success) {
        wins += 1;
      } else {
        failedSeeds.push(seed);
      }
      totalSteps += r.steps;
      totalApples += r.apples || 0;
    }
    return {
      wins: wins,
      losses: episodes - wins,
      draws: 0,
      games: episodes,
      winRate: wins / episodes,
      avgSteps: totalSteps / episodes,
      apples: totalApples,
      failedSeeds: failedSeeds,
      seeds: seeds,
      ms: performance.now() - t0
    };
  }

  /** Sort key: successes desc, avgEffSteps asc (nulls last), apples desc, id. */
  function compareTournamentRows(a, b) {
    if (b.successes !== a.successes) {
      return b.successes - a.successes;
    }
    var ae = a.avgEffSteps;
    var be = b.avgEffSteps;
    var aHas = typeof ae === "number";
    var bHas = typeof be === "number";
    if (aHas && bHas && ae !== be) {
      return ae - be;
    }
    if (aHas !== bHas) {
      return aHas ? -1 : 1;
    }
    if ((b.apples || 0) !== (a.apples || 0)) {
      return (b.apples || 0) - (a.apples || 0);
    }
    return String(a.id).localeCompare(String(b.id));
  }

  /**
   * Shared-seed tournament: every bot plays the same 10 board seeds.
   * Rank: boards beaten, then avg steps on wins (lower better), then apples.
   */
  function shapeTournament(entries, size, mazeCount, opts) {
    opts = opts || {};
    var mazes = [];
    var i;
    var seedBase = typeof opts.seedBase === "number" ? opts.seedBase : ((Math.random() * 0x7fffffff) | 0);
    for (i = 0; i < mazeCount; i += 1) {
      var seed = (seedBase + i * 7919) >>> 0;
      mazes.push({ seed: seed, grid: null, stats: { solutionLen: 1 } });
    }
    var env = new E.SnakeEnv({ size: size });
    var stats = {};
    var e;
    for (e = 0; e < entries.length; e += 1) {
      stats[entries[e].id] = {
        id: entries[e].id,
        family: entries[e].family,
        label: entries[e].label,
        wins: 0,
        losses: 0,
        draws: 0,
        points: 0,
        totalSteps: 0,
        successSteps: 0,
        successes: 0,
        apples: 0,
        avgEffSteps: null
      };
    }
    var t0 = performance.now();
    for (i = 0; i < mazes.length; i += 1) {
      var maze = mazes[i];
      for (e = 0; e < entries.length; e += 1) {
        var ent = entries[e];
        var r = runEpisode(ent.agent, ent.family, env, {
          explore: false,
          train: false,
          seed: maze.seed,
          size: size,
          visionRadius: resolveVision(ent.agent, ent)
        });
        var s = stats[ent.id];
        s.totalSteps += r.steps;
        s.apples += r.apples || 0;
        if (r.success) {
          s.wins += 1;
          s.successes += 1;
          s.points += 1;
          s.successSteps += r.steps;
        } else {
          s.losses += 1;
        }
      }
    }
    var ranked = entries.map(function (ent) {
      var row = stats[ent.id];
      row.avgEffSteps = row.successes > 0 ? row.successSteps / row.successes : null;
      return row;
    });
    ranked.sort(compareTournamentRows);
    return {
      ranked: ranked,
      durationMs: performance.now() - t0,
      moveBudgetMs: 0,
      mazeCount: mazeCount,
      size: size,
      seeds: mazes.map(function (m) { return m.seed; }),
      sharedMazes: true,
      inProgress: false
    };
  }

  function makeRandomAgent() {
    return {
      chooseAction: function (obs, legal) {
        return legal[Math.floor(Math.random() * legal.length)];
      },
      toDict: function () {
        return { type: "random" };
      },
      fromDict: function () {}
    };
  }

  function makeBfsAgent() {
    return {
      visionRadius: C.DEFAULT_VISION,
      chooseAction: function (obs, legal, explore, env) {
        var a = E.bfsNextAction(env || this._env);
        if (a !== null && a !== undefined) {
          return a;
        }
        return legal.length ? legal[0] : 0;
      },
      toDict: function () {
        return { type: "bfs" };
      },
      fromDict: function () {}
    };
  }

  function makeAvoidPocketAgent() {
    return {
      visionRadius: C.DEFAULT_VISION,
      chooseAction: function (obs, legal, explore, env) {
        return E.avoidPocketNextAction(env || this._env);
      },
      toDict: function () {
        return { type: "avoid_pocket" };
      },
      fromDict: function () {}
    };
  }

  function makeGreedyAgent() {
    return {
      visionRadius: C.DEFAULT_VISION,
      chooseAction: function (obs, legal, explore, env) {
        return E.greedyNextAction(env || this._env);
      },
      toDict: function () {
        return { type: "greedy" };
      },
      fromDict: function () {}
    };
  }

  function makeWallFollowerAgent() {
    return {
      visionRadius: C.DEFAULT_VISION,
      chooseAction: function (obs, legal, explore, env) {
        return E.wallFollowerNextAction(env || this._env);
      },
      toDict: function () {
        return { type: "wall_follower" };
      },
      fromDict: function () {}
    };
  }

  /** Master: Cell Tree on even n; Tail Safe Hybrid on odd n. */
  function makeMasterAgent() {
    return {
      visionRadius: C.VISION_MAX,
      chooseAction: function (obs, legal, explore, env) {
        return E.masterNextAction(env || this._env);
      },
      toDict: function () {
        return { type: "master" };
      },
      fromDict: function () {}
    };
  }

  /** Pure Hamilton cycle follower (Observe baseline). */
  function makeHamiltonAgent() {
    return {
      visionRadius: C.VISION_MAX,
      chooseAction: function (obs, legal, explore, env) {
        return E.masterHamiltonAction(env || this._env);
      },
      toDict: function () {
        return { type: "hamilton" };
      },
      fromDict: function () {}
    };
  }

  /** Hamilton Aggressive: skipCap=N under 20% fill, then production Hamilton. */
  function makeHamiltonAggrAgent() {
    return {
      visionRadius: C.VISION_MAX,
      chooseAction: function (obs, legal, explore, env) {
        return E.hamiltonAggrNextAction(env || this._env);
      },
      toDict: function () {
        return { type: "hamilton_aggr" };
      },
      fromDict: function () {}
    };
  }

  /** Cell Tree (twanvl): 2x2 right-hand + spanning tree; odd = Odd until ~36% then healTS->Ham. */
  function makeCellTreeAgent() {
    return {
      visionRadius: C.VISION_MAX,
      chooseAction: function (obs, legal, explore, env) {
        return E.cellTreeNextAction(env || this._env);
      },
      toDict: function () {
        return { type: "cell_tree" };
      },
      fromDict: function () {}
    };
  }

  /** Cell Tree Odd: fringe splices + locks on odd; no Ham. */
  function makeCellTreeOddAgent() {
    return {
      visionRadius: C.VISION_MAX,
      chooseAction: function (obs, legal, explore, env) {
        return E.cellTreeOddNextAction(env || this._env);
      },
      toDict: function () {
        return { type: "cell_tree_odd" };
      },
      fromDict: function () {}
    };
  }

  /** @deprecated alias - use makeCellTreeOddAgent */
  function makeCellTreeFringeAgent() {
    return makeCellTreeOddAgent();
  }

  /** DHCR: repairable Ham cycle (no SafeHam lock). */
  function makeDhcrAgent() {
    return {
      visionRadius: C.VISION_MAX,
      chooseAction: function (obs, legal, explore, env) {
        return E.dhcrNextAction(env || this._env);
      },
      toDict: function () {
        return { type: "dhcr" };
      },
      fromDict: function () {}
    };
  }

  /** DHCR Improved: SafeHam on livelock until next food. */
  function makeDhcrImprovedAgent() {
    return {
      visionRadius: C.VISION_MAX,
      chooseAction: function (obs, legal, explore, env) {
        return E.dhcrImprovedNextAction(env || this._env);
      },
      toDict: function () {
        return { type: "dhcr_improved" };
      },
      fromDict: function () {}
    };
  }

  /** Plan Ahead: L-path, heal when stuck, reenter plan when possible. */
  function makePlanAheadAgent() {
    return {
      visionRadius: C.VISION_MAX,
      chooseAction: function (obs, legal, explore, env) {
        return E.planAheadReenterNextAction(env || this._env);
      },
      toDict: function () {
        return { type: "plan_ahead" };
      },
      fromDict: function () {}
    };
  }

  /** Observe Tail Safe: tip-safe only (no heal/Ham handoff). */
  function makeTailSafeAgent() {
    return {
      visionRadius: C.VISION_MAX,
      chooseAction: function (obs, legal, explore, env) {
        return E.tailSafeNextAction(env || this._env);
      },
      toDict: function () {
        return { type: "tail_safe" };
      },
      fromDict: function () {}
    };
  }

  /** Prior Master: Tail Safe + stall/20% -> HealTS -> Hamilton. */
  function makeTailSafeHybridAgent() {
    return {
      visionRadius: C.VISION_MAX,
      chooseAction: function (obs, legal, explore, env) {
        return E.tailSafeHybridNextAction(env || this._env);
      },
      toDict: function () {
        return { type: "tail_safe_hybrid" };
      },
      fromDict: function () {}
    };
  }

  function displayName(id) {
    return (C.ALGO_NAMES && C.ALGO_NAMES[id]) || id;
  }

  global.SNAKE_APP = {
    SHAPE_FAMILIES: SHAPE_FAMILIES,
    PLAY_HUMAN: C.PLAY_HUMAN,
    MAZE_POOL_MAX: 1000,
    get hardMode() {
      return hardMode;
    },
    set hardMode(v) {
      hardMode = !!v;
    },
    isHardMode: isHardMode,
    setHardMode: setHardMode,
    get compassOn() {
      return compassOn;
    },
    set compassOn(v) {
      compassOn = v !== false;
    },
    isCompassOn: isCompassOn,
    setCompassOn: setCompassOn,
    applyHardOutputSize: applyHardOutputSize,
    withInputDim: withInputDim,
    createShapeAgent: createShapeAgent,
    defaultLayerSizes: defaultLayerSizes,
    defaultLearningRate: defaultLearningRate,
    shapeAgentToDict: shapeAgentToDict,
    shapeAgentFromDict: shapeAgentFromDict,
    chooseAction: chooseAction,
    chooseDecision: chooseDecision,
    beginEpisodePrevDir: beginEpisodePrevDir,
    advancePrevDir: advancePrevDir,
    runEpisode: runEpisode,
    shapeTrainEpisodes: shapeTrainEpisodes,
    shapePeerTeachPass: shapePeerTeachPass,
    isTeachableFamily: isTeachableFamily,
    TEACH_REWARD: TEACH_REWARD,
    TEACH_PUNISH: TEACH_PUNISH,
    shapeTrainPeer: shapeTrainPeer,
    shapeEvolveOne: shapeEvolveOne,
    shapeEvalSuccess: shapeEvalSuccess,
    shapeTournament: shapeTournament,
    compareTournamentRows: compareTournamentRows,
    makeRandomAgent: makeRandomAgent,
    makeBfsAgent: makeBfsAgent,
    makeAvoidPocketAgent: makeAvoidPocketAgent,
    makeGreedyAgent: makeGreedyAgent,
    makeWallFollowerAgent: makeWallFollowerAgent,
    makeMasterAgent: makeMasterAgent,
    makeHamiltonAgent: makeHamiltonAgent,
    makeHamiltonAggrAgent: makeHamiltonAggrAgent,
    makeCellTreeAgent: makeCellTreeAgent,
    makeCellTreeOddAgent: makeCellTreeOddAgent,
    makeCellTreeFringeAgent: makeCellTreeFringeAgent,
    makeDhcrAgent: makeDhcrAgent,
    makeDhcrImprovedAgent: makeDhcrImprovedAgent,
    makePlanAheadAgent: makePlanAheadAgent,
    makeTailSafeAgent: makeTailSafeAgent,
    makeTailSafeHybridAgent: makeTailSafeHybridAgent,
    displayName: displayName,
    mazePoolClear: mazePoolClear,
    mazePoolCount: mazePoolCount,
    mazePoolGetMax: mazePoolGetMax,
    mazePoolSetMax: mazePoolSetMax,
    mazePoolNext: mazePoolNext,
    mazePoolGenerate: mazePoolGenerate
  };
})(typeof window !== "undefined" ? window : global);
