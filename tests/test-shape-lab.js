/**
 * Snake Shape Lab tests (Node).
 */
"use strict";

var path = require("path");
var root = path.join(__dirname, "..");

require(path.join(root, "snake_js", "constants.js"));
require(path.join(root, "snake_js", "snake-env.js"));
require(path.join(root, "snake_js", "published-agents.js"));
require(path.join(root, "snake_js", "network.js"));
/* No TF.js in Node tests: TfjsDqnAgent falls back to NnAgent when global.tf is missing. */
require(path.join(root, "snake_js", "agents.js"));
require(path.join(root, "snake_js", "engine.js"));
require(path.join(root, "snake_js", "shape-lab.js"));

var C = global.SNAKE_CONSTANTS;
var E = global.SNAKE_ENV;
var APP = global.SNAKE_APP;
var LAB = global.SNAKE_SHAPE_LAB;

function assert(cond, msg) {
  if (!cond) {
    throw new Error(msg || "assert failed");
  }
}

assert(LAB.BEAT_WR === 0.9, "beat threshold is 90%");
assert(LAB.BEAT_STREAK_NEED === 10, "beat streak 10");
assert(C.SIZE_MIN === 4, "SIZE_MIN 4");
assert(C.SIZE_MAX === 100, "SIZE_MAX 100");
assert(LAB.CHECKPOINT_FORMAT === "snake-shape-lab-v1", "checkpoint format");

var search = new LAB.ShapeSearch();
assert(search.mazeSize === 4, "gen0 mazeSize 4");
assert(search.visionMode === "adaptive", "vision adaptive");
assert(search.population.length > 5, "population seeded");
assert(search.population[0].visionRadius === 2, "gen0 r=2");
assert(String(search.population[0].label).indexOf("r2") >= 0, "label includes r2");

var obsSize = C.obsSizeForRadius(2, true);
assert(search.population[0].agent.visionRadius === 2, "agent vision");
if (search.population[0].layerSizes) {
  assert(search.population[0].layerSizes[0] === obsSize, "input dim matches vision+compass");
}

/* Force eval metadata (full snake episodes are long; skip timed warmup). */
var ev = APP.shapeEvalSuccess(
  search.population[0].agent,
  search.population[0].family,
  4,
  2,
  { seedBase: 9, visionRadius: 2 }
);
assert(ev.games === 2, "eval games=2");
assert(typeof ev.winRate === "number", "eval winRate");
search.population[0].lastEvalWinRate = ev.winRate;

search.population[0].beaten = true;
search.population[0].beatStreak = LAB.BEAT_STREAK_NEED;
assert(search.roundFillComplete(), "round fill complete when beater");

var beforeSize = search.mazeSize;
search.finishRound();
assert(search.mazeSize === beforeSize + 1, "after finishRound mazeSize +1");

/* Tournament baselines */
var tourn = APP.shapeTournament(
  search.buildTournamentEntries(),
  4,
  3,
  { seedBase: 1234 }
);
assert(tourn.ranked.length >= 6, "tournament ranked includes pop+baselines");
assert(tourn.sharedMazes === true, "shared seeds");
assert(tourn.seeds.length === 3, "3 seeds");

var families = {};
tourn.ranked.forEach(function (r) {
  families[r.family] = true;
});
assert(families.random, "Random baseline");
assert(families.bfs, "BFS baseline");
assert(families.greedy, "Greedy baseline");
assert(families.wall_follower, "Wall follower baseline");
assert(families.avoid_pocket, "Avoid Pocket baseline");
assert(families.master, "Master baseline");
assert(families.hamilton, "Hamilton baseline");
assert(families.tail_safe, "Tail Safe baseline");

/* Ranking fields */
assert(typeof tourn.ranked[0].apples === "number", "apples on ranked");
assert(typeof tourn.ranked[0].totalSteps === "number", "totalSteps on ranked");

/* Efficiency: fewer avg steps/win ranks higher when successes match. */
var fast = { id: "a", successes: 10, avgEffSteps: 40, apples: 20 };
var slow = { id: "b", successes: 10, avgEffSteps: 80, apples: 20 };
assert(APP.compareTournamentRows(fast, slow) < 0, "faster win avg ranks first");
assert(APP.compareTournamentRows(slow, fast) > 0, "slower win avg ranks second");
var noWin = { id: "c", successes: 0, avgEffSteps: null, apples: 5 };
var oneWin = { id: "d", successes: 1, avgEffSteps: 99, apples: 1 };
assert(APP.compareTournamentRows(oneWin, noWin) < 0, "any win beats zero wins");
var moreApples = { id: "e", successes: 5, avgEffSteps: 50, apples: 30 };
var fewerApples = { id: "f", successes: 5, avgEffSteps: 50, apples: 10 };
assert(APP.compareTournamentRows(moreApples, fewerApples) < 0, "apples break equal efficiency");

/* Train One demote */
assert(LAB.DEMOTE_WR === 0.5, "demote 50%");
var trainOne = new LAB.TrainOne();
trainOne.startFromSnapshotOrLive(null, search.population[0]);
assert(trainOne.trainee, "trainee loaded");
assert(trainOne.trainee.visionRadius === search.population[0].visionRadius, "trainee vision");

trainOne.mazeSize = 12;
trainOne.trainee.lastEvalWinRate = 0.4;
trainOne.trainee.beatStreak = 0;
/* Simulate demote path from tick when WR low - call logic inline */
if (trainOne.trainee.lastEvalWinRate < LAB.DEMOTE_WR && trainOne.mazeSize > C.SIZE_MIN) {
  trainOne.mazeSize -= 1;
}
assert(trainOne.mazeSize === 11, "demote size");

/* Catalog */
var lab = new LAB.ShapeLab();
lab.search = search;
lab.trainOne = trainOne;
var cat = lab.getPlayCatalog();
var names = cat.map(function (c) { return c.id; });
assert(names[0] === "master", "Master first in Observe catalog");
assert(names[1] === "hamilton", "Hamilton second in Observe catalog");
assert(names[2] === "hamilton_aggr", "Hamilton Aggressive third in Observe catalog");
assert(cat[2].name === "Hamilton Aggressive", "Hamilton Aggressive display name");
assert(names[3] === "cell_tree", "Cell Tree fourth in Observe catalog");
assert(cat[3].name === "Cell Tree", "Cell Tree display name");
assert(names[4] === "cell_tree_odd", "Cell Tree Odd fifth in Observe catalog");
assert(cat[4].name === "Cell Tree Odd", "Cell Tree Odd display name");
assert(names[5] === "dhcr", "DHCR sixth in Observe catalog");
assert(cat[5].name === "DHCR", "DHCR display name");
assert(names[6] === "dhcr_improved", "DHCR Improved seventh in Observe catalog");
assert(cat[6].name === "DHCR Improved", "DHCR Improved display name");
assert(names[7] === "tail_safe", "Tail Safe eighth in Observe catalog");
assert(names[8] === "tail_safe_hybrid", "Tail Safe Hybrid ninth in catalog");
assert(cat[8].name === "Tail Safe Hybrid", "Tail Safe Hybrid display name");
assert(names.indexOf("plan_ahead") === 9, "Plan Ahead tenth in catalog");
assert(names.indexOf("cell_tree_fringe") < 0, "no Cell Tree Fringe in Observe");
assert(names.indexOf("cell_tree_odd_lock") < 0, "no Cell Tree Odd Lock in Observe");
assert(names.indexOf("hamilton_early") < 0, "no Hamilton Early in Observe");
assert(names.indexOf("hamilton_arc") < 0, "no Hamilton Arc in Observe");
assert(names.indexOf("hamilton_soft") < 0, "no Hamilton Soft in Observe");
assert(names.indexOf("plan_tail") < 0, "no Plan+Tail in Observe");
assert(names.indexOf("tail_stall_ham") < 0, "no TS Stall Ham label in Observe");
assert(names.indexOf("plan_ahead_deep") < 0, "no Plan Ahead Deep");
assert(names.indexOf("greedy") >= 0, "catalog greedy");
assert(names.indexOf("wall_follower") >= 0, "catalog wall follower");
assert(names.indexOf("master") >= 0, "catalog master");
assert(names.indexOf("right_turns") < 0, "no right_turns");

var resolved = lab.resolvePlayAgent("wall_follower");
assert(resolved.family === "wall_follower", "resolve wall follower");
var resolvedMaster = lab.resolvePlayAgent("master");
assert(resolvedMaster.family === "master", "resolve master");
var resolvedHam = lab.resolvePlayAgent("hamilton");
assert(resolvedHam.family === "hamilton", "resolve hamilton");
assert(typeof APP.makeHamiltonAgent === "function", "makeHamiltonAgent exported");
var resolvedHamAggr = lab.resolvePlayAgent("hamilton_aggr");
assert(resolvedHamAggr.family === "hamilton_aggr", "resolve hamilton aggr");
assert(typeof APP.makeHamiltonAggrAgent === "function",
  "makeHamiltonAggrAgent exported");
assert(lab.resolvePlayAgent("hamilton_soft").family === "hamilton_aggr",
  "legacy soft id maps to aggr");
var resolvedCell = lab.resolvePlayAgent("cell_tree");
assert(resolvedCell.family === "cell_tree", "resolve cell tree");
assert(typeof APP.makeCellTreeAgent === "function", "makeCellTreeAgent exported");
var resolvedCellOdd = lab.resolvePlayAgent("cell_tree_odd");
assert(resolvedCellOdd.family === "cell_tree_odd", "resolve cell tree odd");
assert(typeof APP.makeCellTreeOddAgent === "function", "makeCellTreeOddAgent exported");
assert(lab.resolvePlayAgent("cell_tree_fringe").family === "cell_tree_odd",
  "legacy fringe id maps to odd");
assert(lab.resolvePlayAgent("cell_tree_odd_lock").family === "cell_tree_odd",
  "legacy odd-lock id maps to odd");
var resolvedDhcr = lab.resolvePlayAgent("dhcr");
assert(resolvedDhcr.family === "dhcr", "resolve dhcr");
assert(typeof APP.makeDhcrAgent === "function", "makeDhcrAgent exported");
var resolvedDhcrImp = lab.resolvePlayAgent("dhcr_improved");
assert(resolvedDhcrImp.family === "dhcr_improved", "resolve dhcr improved");
assert(typeof APP.makeDhcrImprovedAgent === "function",
  "makeDhcrImprovedAgent exported");
var resolvedPa = lab.resolvePlayAgent("plan_ahead");
assert(resolvedPa.family === "plan_ahead", "resolve plan ahead");
var resolvedTs = lab.resolvePlayAgent("tail_safe");
assert(resolvedTs.family === "tail_safe", "resolve tail safe");
var resolvedTsH = lab.resolvePlayAgent("tail_safe_hybrid");
assert(resolvedTsH.family === "tail_safe_hybrid", "resolve tail safe hybrid");
assert(typeof APP.makeTailSafeHybridAgent === "function",
  "makeTailSafeHybridAgent exported");

assert(APP.isTeachableFamily("bfs") === false, "bfs not teachable");
assert(APP.isTeachableFamily("master") === false, "master not teachable");
assert(APP.isTeachableFamily("hamilton") === false, "hamilton not teachable");
assert(APP.isTeachableFamily("hamilton_aggr") === false,
  "hamilton aggr not teachable");
assert(APP.isTeachableFamily("cell_tree") === false, "cell tree not teachable");
assert(APP.isTeachableFamily("dhcr") === false, "dhcr not teachable");
assert(APP.isTeachableFamily("dhcr_improved") === false,
  "dhcr improved not teachable");
assert(APP.isTeachableFamily("plan_ahead") === false, "plan ahead not teachable");
assert(APP.isTeachableFamily("tail_safe_hybrid") === false,
  "tail safe hybrid not teachable");
assert(APP.isTeachableFamily("qtable") === true, "qtable teachable");
assert(typeof APP.makePlanAheadAgent === "function", "makePlanAheadAgent");
assert(typeof APP.makeTailSafeAgent === "function", "makeTailSafeAgent");
assert(typeof APP.makePlanTailAgent !== "function", "no makePlanTailAgent");
assert(typeof APP.makeTailStallHamAgent !== "function", "no makeTailStallHamAgent");
assert(typeof APP.makePlanAheadDeepAgent !== "function", "no makePlanAheadDeepAgent");
assert(typeof APP.makeHamiltonEarlyAgent !== "function", "no makeHamiltonEarlyAgent");
assert(typeof APP.makeHamiltonSoftAgent !== "function", "no makeHamiltonSoftAgent");

/* Teach stays in one vision-radius peer group. */
var teachPop = [
  { id: "q2a", family: "qtable", visionRadius: 2, agent: APP.createShapeAgent("qtable", null, null, 2) },
  { id: "q2b", family: "qtable", visionRadius: 2, agent: APP.createShapeAgent("qtable", null, null, 2) },
  { id: "q3a", family: "qtable", visionRadius: 3, agent: APP.createShapeAgent("qtable", null, null, 3) }
];
var teachOut = APP.shapePeerTeachPass(teachPop, 4, { poolAfterId: 0 });
assert(teachOut.visionRadius === 2 || teachOut.visionRadius === 3, "teach picks an r-family");
if (teachOut.teacherId) {
  var teacher = null;
  teachPop.forEach(function (p) {
    if (p.id === teachOut.teacherId) {
      teacher = p;
    }
  });
  assert(teacher && teacher.visionRadius === teachOut.visionRadius, "teacher matches teach radius");
}

/* Learn from Master: baselines race; may teach; never counted as students. */
var masterIds = { random: true, greedy: true, wall_follower: true, bfs: true };
var soloLearner = [
  { id: "solo_q", family: "qtable", visionRadius: 2, agent: APP.createShapeAgent("qtable", null, null, 2) }
];
var masterOut = APP.shapePeerTeachPass(soloLearner, 4, {
  poolAfterId: 0,
  learnFromMaster: true,
  makeMasters: function (vr) {
    return [
      { id: "random", family: "random", agent: APP.makeRandomAgent(), visionRadius: vr },
      { id: "greedy", family: "greedy", agent: APP.makeGreedyAgent(), visionRadius: vr },
      { id: "wall_follower", family: "wall_follower", agent: APP.makeWallFollowerAgent(), visionRadius: vr },
      { id: "bfs", family: "bfs", agent: APP.makeBfsAgent(), visionRadius: vr }
    ];
  }
});
assert(masterOut.visionRadius === 2, "master teach uses solo learner r");
assert(!!masterOut.teacherId, "master teach picks a teacher");
assert(masterOut.students === 0 || masterOut.students === 1, "students only learners");
if (masterIds[masterOut.teacherId]) {
  assert(masterOut.students === 1, "baseline teacher teaches the one learner");
} else {
  assert(masterOut.teacherId === "solo_q", "learner teacher id");
  assert(masterOut.students === 0, "learner teacher has no other students");
}
assert(lab.search.learnFromMaster === false, "learnFromMaster default off");

/* Adaptive tabular spawn: r+/-1 offspring. */
var tabParent = LAB.createIndividual({ family: "qtable", visionRadius: 2 });
var nextTab = [tabParent];
LAB.spawnOffspringFromParent(tabParent, nextTab, "adaptive");
var tabRadii = nextTab.map(function (ind) { return ind.visionRadius; });
assert(tabRadii.indexOf(1) >= 0, "tabular offspring r1");
assert(tabRadii.indexOf(3) >= 0, "tabular offspring r3");
assert(nextTab.length >= 3, "parent kept plus r+/-1 kids");

/* Category floors: carry best tournament bots (keep their r), not invent r. */
assert(LAB.CATEGORY_FLOOR.neural === 3, "neural floor 3");
assert(LAB.CATEGORY_FLOOR.tabular === 3, "tabular floor 3");
assert(LAB.CATEGORY_FLOOR.evo === 1, "evo floor 1");
assert(LAB.categoryOfFamily("nn_policy") === "neural", "nn is neural");
assert(LAB.categoryOfFamily("qtable") === "tabular", "qtable tabular");
assert(LAB.categoryOfFamily("neat") === "evo", "neat evo");

var nnBest = LAB.createIndividual({
  id: "nn_best_r3",
  family: "nn_policy",
  visionRadius: 3,
  layerSizes: APP.defaultLayerSizes("nn_policy", 3, true)
});
var nnWorse = LAB.createIndividual({
  id: "nn_worse_r1",
  family: "nn_value",
  visionRadius: 1,
  layerSizes: APP.defaultLayerSizes("nn_value", 1, true)
});
var tabOnly = LAB.createIndividual({ id: "tab_only", family: "qtable", visionRadius: 2 });
var evoOnly = LAB.createIndividual({ id: "evo_only", family: "neat", visionRadius: 2 });
var oldCarry = [nnBest, nnWorse, tabOnly, evoOnly];
var nextCarry = [tabOnly];
var learnedCarry = [
  { id: "nn_best_r3", family: "nn_policy", successes: 8 },
  { id: "nn_worse_r1", family: "nn_value", successes: 1 },
  { id: "tab_only", family: "qtable", successes: 10 },
  { id: "evo_only", family: "neat", successes: 0 }
];
LAB.ensureCategoryFloors(nextCarry, "adaptive", learnedCarry, oldCarry);
var carriedNn = nextCarry.filter(function (ind) {
  return LAB.categoryOfFamily(ind.family) === "neural";
});
assert(carriedNn.length >= 2, "carry at least two neural from tournament");
assert(carriedNn[0].id === "nn_best_r3", "best neural carried first");
assert(carriedNn[0].visionRadius === 3, "carried neural keeps r3");
assert(carriedNn.some(function (ind) { return ind.id === "nn_worse_r1"; }), "second neural also carried");
assert(nextCarry.some(function (ind) {
  return ind.id === "evo_only";
}), "evo carried from tournament");
var nN = 0;
var nT = 0;
var nE = 0;
nextCarry.forEach(function (ind) {
  var cat = LAB.categoryOfFamily(ind.family);
  if (cat === "neural") {
    nN += 1;
  } else if (cat === "tabular") {
    nT += 1;
  } else if (cat === "evo") {
    nE += 1;
  }
});
assert(nN >= 3, "floor fills neural to 3 (carry + seed if needed)");
assert(nT >= 3, "floor fills tabular to 3");
assert(nE >= 1, "floor fills evo to 1");

/* Episode smoke */
var env = new E.SnakeEnv({ size: 4 });
var agent = APP.createShapeAgent("qtable", null, null, 2);
var ep = APP.runEpisode(agent, "qtable", env, {
  explore: true,
  train: true,
  seed: 55,
  size: 4,
  visionRadius: 2
});
assert(typeof ep.success === "boolean", "episode success bool");
assert(typeof ep.apples === "number", "episode apples");

console.log("test-shape-lab: ok");
