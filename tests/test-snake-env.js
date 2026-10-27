/**
 * Snake env unit tests (Node).
 */
"use strict";

var path = require("path");
var fs = require("fs");
var root = path.join(__dirname, "..");
global.window = undefined;

require(path.join(root, "snake_js", "constants.js"));
require(path.join(root, "snake_js", "snake-env.js"));
require(path.join(root, "snake_js", "published-agents.js"));

var C = global.SNAKE_CONSTANTS;
var E = global.SNAKE_ENV;

function assert(cond, msg) {
  if (!cond) {
    throw new Error(msg || "assert failed");
  }
}

assert(C.SIZE_MIN === 4, "SIZE_MIN 4");
assert(C.SIZE_MAX === 100, "SIZE_MAX 100");
assert(E.targetLengthForSize(4) === 4, "4x4 target length 4");
assert(E.maxEpisodeStepsForSize(4) === 64, "4x4 max steps 64");
assert(E.targetLengthForSize(10) === 25, "10x10 target length 25");
assert(E.maxEpisodeStepsForSize(10) === 1000, "10x10 max steps 1000");
/* Large boards use fitted 2*p95 budget (ceil(0.16*n^3.75)), not old n^3. */
assert(E.maxEpisodeStepsForSize(30) === Math.ceil(0.16 * Math.pow(30, 3.75)),
  "30x30 uses fitted max steps");
assert(E.maxEpisodeStepsForSize(30) > 27000, "30x30 budget exceeds old 27000");
assert(typeof E.maxEpisodeStepsForTarget === "function", "target-scaled budget");
assert(E.maxEpisodeStepsForTarget(30, Math.ceil(0.3 * 30 * 30)) >
  E.maxEpisodeStepsForSize(30), "30% target scales above 25% budget");
assert(E.obsSizeForRadius(2, false) === 24, "r2 no compass = 24");
assert(E.obsSizeForRadius(2, true) === 28, "r2 compass = 28");

var env = new E.SnakeEnv({ size: 4, visionRadius: 2, compassOn: true });
var obs = env.reset({ seed: 42, size: 4, visionRadius: 2, compassOn: true });
assert(obs.length === 28, "observe length 28 with compass");
assert(env.snake.length === 3, "start length 3");
assert(env.facing === 0, "start facing North");
assert(env.snake[0][0] === 1 && env.snake[0][1] === 2, "4x4 start near center");
assert(env.snake[2][0] === 3, "4x4 tail on board");

var hasCodes = obs.slice(0, 24).every(function (v) {
  return v === -2 || v === -1 || v === 0 || v === 1;
});
assert(hasCodes, "vision cells use -2/-1/0/1");

assert(E.REWARD_APPROACH === 0.1, "approach shaping 0.1");

/* Death into wall: no approach shaping */
env.reset({ seed: 1, size: 4 });
env.snake = [[0, 2], [1, 2], [2, 2]];
env.food = [2, 0];
env.facing = 0;
var wallStep = env.step(0);
assert(wallStep.done && wallStep.reward === E.REWARD_DEATH, "wall death -20");

/* Closer to food => +REWARD_APPROACH */
env.reset({ seed: 2, size: 4 });
env.snake = [[2, 2], [3, 2], [3, 1]];
env.food = [0, 2];
env.facing = 0;
var closer = env.step(0);
assert(!closer.done, "approach step not done");
assert(Math.abs(closer.reward - E.REWARD_APPROACH) < 1e-9, "closer +0.1");

/* Farther from food => -REWARD_APPROACH */
env.reset({ seed: 3, size: 4 });
env.snake = [[1, 1], [2, 1], [3, 1]];
env.food = [1, 0];
env.facing = 1;
var farther = env.step(1);
assert(!farther.done, "away step not done");
assert(Math.abs(farther.reward + E.REWARD_APPROACH) < 1e-9, "farther -0.1");

/* Eat food: approach + food = 1.1 */
env.reset({ seed: 99, size: 4 });
env.snake = [[1, 2], [2, 2], [3, 2]];
env.food = [0, 2];
env.facing = 0;
var eat = env.step(0);
assert(Math.abs(eat.reward - (E.REWARD_FOOD + E.REWARD_APPROACH)) < 1e-9, "eat 1.1");
assert(env.snake.length === 4, "grew to 4");
assert(env.apples === 1, "apples 1");
assert(env.done && env.success, "4x4 length 4 is success");

/* Food never spawns on body; stream skips occupied cells. */
env.reset({ seed: 7, size: 4 });
env.snake = [[0, 0], [0, 1], [0, 2]];
env.foodRng = E.makeRng(12345);
var beforeLen = env.snake.length;
env.nextFoodFromStream();
assert(env.food, "food stream places on empty cell");
assert(!env.isBody(env.food[0], env.food[1]), "food not on body");
assert(env.snake.length === beforeLen, "skipping body does not grow snake");

env.setWinFraction(0.5);
assert(env.targetLength === 8, "50% of 4x4 is length 8");

/* Same seed => same food sequence */
var a = new E.SnakeEnv({ size: 4 });
var b = new E.SnakeEnv({ size: 4 });
a.reset({ seed: 777, size: 4 });
b.reset({ seed: 777, size: 4 });
assert(a.food[0] === b.food[0] && a.food[1] === b.food[1], "shared seed food");

assert(typeof E.bfsNextAction === "function", "bfs helper");
assert(typeof E.greedyNextAction === "function", "greedy helper");
assert(typeof E.wallFollowerNextAction === "function", "wall follower helper");
assert(typeof E.avoidPocketNextAction === "function", "avoid pocket helper");
assert(typeof E.pocketOk === "function", "pocketOk helper");
assert(typeof E.masterEarlyThreshold === "function", "master early threshold");
assert(typeof E.masterCycleSkip === "function", "master cycle skip");
assert(typeof E.masterNextAction === "function", "master helper");
assert(typeof E.getHamiltonCycle === "function", "hamilton cycle helper");
assert(E.masterEarlyThreshold(20) === 20, "early threshold 20x20 is one width");
assert(E.masterEarlyThreshold(10) === 10, "early threshold 10x10 is one width");
assert(E.masterEarlyThreshold(40) === Math.ceil(0.05 * 40 * 40), "early threshold large board 5%");

function assertCycle(n) {
  var cy = E.getHamiltonCycle(n);
  var omit = n % 2 === 1;
  var expect = omit ? n * n - 1 : n * n;
  assert(cy.len === expect, "cycle len n=" + n);
  var seen = {};
  var i;
  for (i = 0; i < cy.path.length; i += 1) {
    var p = cy.path[i];
    var q = cy.path[(i + 1) % cy.path.length];
    var k = p[0] + "," + p[1];
    assert(!seen[k], "cycle unique n=" + n);
    seen[k] = true;
    assert(Math.abs(p[0] - q[0]) + Math.abs(p[1] - q[1]) === 1, "cycle adj n=" + n);
  }
}
assertCycle(4);
assertCycle(5);
assertCycle(6);

/* Master returns a safe action on a coiled board with food in a pocket. */
var trap = new E.SnakeEnv({ size: 6, visionRadius: 10, compassOn: true });
trap.reset({ seed: 1, size: 6, visionRadius: 10, compassOn: true });
trap.snake = [
  [2, 2],
  [2, 1],
  [1, 1],
  [0, 1],
  [0, 2],
  [0, 3],
  [1, 3],
  [2, 3]
];
trap.facing = 1;
trap.food = [1, 2];
var masterAct = E.masterNextAction(trap);
var trapSafe = trap.safeActions();
assert(trapSafe.indexOf(masterAct) >= 0, "master pick is immediate-safe");
assert(typeof masterAct === "number", "master returns action");

/* Master prefers a flood-safe cycle step over a one-cell pocket when both are
   immediate-safe. Head at (1,1) facing East; body fills a U so North is a
   dead-end pocket and East follows open space toward tip. */
var pocket = new E.SnakeEnv({ size: 6, visionRadius: 10, compassOn: true });
pocket.reset({ seed: 2, size: 6, visionRadius: 10, compassOn: true });
pocket.snake = [
  [2, 2],
  [2, 1],
  [2, 0],
  [1, 0],
  [0, 0],
  [0, 1],
  [0, 2],
  [0, 3],
  [1, 3],
  [2, 3],
  [3, 3],
  [3, 2]
];
pocket.facing = 0;
pocket.food = [5, 5];
var pocketSafe = pocket.safeActions();
assert(pocketSafe.length >= 1, "pocket fixture has a safe move");
var pocketAct = E.masterNextAction(pocket);
assert(pocketSafe.indexOf(pocketAct) >= 0, "master pocket pick is safe");

/* Avoid Pocket returns an immediate-safe action. */
var apEnv = new E.SnakeEnv({ size: 8, visionRadius: 10, compassOn: true });
apEnv.reset({ seed: 9, size: 8, visionRadius: 10, compassOn: true });
var apAct = E.avoidPocketNextAction(apEnv);
assert(apEnv.safeActions().indexOf(apAct) >= 0, "avoid pocket pick is safe");

/* pocketOk rejects a one-cell dead-end (flood < 2 or no escape). */
var oneCell = new E.SnakeEnv({ size: 6, visionRadius: 10, compassOn: true });
oneCell.reset({ seed: 11, size: 6, visionRadius: 10, compassOn: true });
oneCell.snake = [
  [2, 1],
  [2, 2],
  [2, 3],
  [1, 3],
  [0, 3],
  [0, 2],
  [0, 1],
  [0, 0],
  [1, 0],
  [2, 0],
  [3, 0],
  [3, 1]
];
oneCell.facing = 0;
oneCell.food = [5, 5];
/* North from (2,1) into (1,1) is a single empty cell boxed by body. */
assert(!E.pocketOk(oneCell, 1, 1, false), "pocketOk rejects one-cell pocket");

/* Observe-style: disableStepTimeout never ends as timeout. */
var longEnv = new E.SnakeEnv({ size: 4, visionRadius: 4, compassOn: true });
longEnv.reset({ seed: 3, size: 4, visionRadius: 4, compassOn: true });
longEnv.setWinFraction(1);
longEnv.disableStepTimeout = true;
longEnv.maxSteps = 1;
longEnv.steps = 0;
var stepInfo = longEnv.step(longEnv.facing | 0);
var deathKind = stepInfo && stepInfo.info && stepInfo.info.death;
assert(deathKind !== "timeout", "disableStepTimeout skips timeout");

/* Tail Safe Hybrid (prior Master): starts in tail phase while short. */
var phaseEnv = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
phaseEnv.reset({ seed: 5, size: 10, visionRadius: 10, compassOn: true });
assert(phaseEnv._tsStallPhase === "tail", "TS hybrid tail phase after reset");
E.tailSafeHybridNextAction(phaseEnv);
assert(phaseEnv._tsStallPhase === "tail", "TS hybrid stays in tail while short");
/* Master: even = Cell Tree; odd = Tail Safe Hybrid. */
var masterEven = new E.SnakeEnv({ size: 8, visionRadius: 10, compassOn: true });
masterEven.reset({ seed: 5, size: 8, visionRadius: 10, compassOn: true });
var masterCt = E.masterNextAction(masterEven);
assert(masterCt === E.cellTreeNextAction(masterEven), "master even === cell tree");
assert(masterEven.safeActions().indexOf(masterCt) >= 0, "master even first step safe");
var masterOddA = new E.SnakeEnv({ size: 7, visionRadius: 10, compassOn: true });
masterOddA.reset({ seed: 5, size: 7, visionRadius: 10, compassOn: true });
var masterOddB = new E.SnakeEnv({ size: 7, visionRadius: 10, compassOn: true });
masterOddB.reset({ seed: 5, size: 7, visionRadius: 10, compassOn: true });
assert(E.masterNextAction(masterOddA) === E.tailSafeHybridNextAction(masterOddB),
  "master odd === tail safe hybrid");
/* PlanAhead hybrid still uses plan phase (ablation / Plan Ahead bot). */
assert(phaseEnv._planPhase === "plan", "planAhead phase field after reset");
E.planAheadHybridNextAction(phaseEnv);
assert(phaseEnv._planPhase === "plan", "planAhead stays in plan while short");
assert(typeof E.masterHamiltonAction === "function", "hamilton-only helper exported");
assert(typeof E.hamiltonAggrNextAction === "function", "hamilton aggr exported");
assert(typeof E.cellTreeNextAction === "function", "cell tree exported");
assert(typeof E.dhcrNextAction === "function", "dhcr exported");
assert(typeof E.dhcrImprovedNextAction === "function", "dhcr improved exported");
assert(E._publishedIsHamiltonianCycle(E._publishedCycleFromHamilton(6), 6),
  "even Ham cycle is Hamiltonian for DHCR");
assert(E._publishedIsHamiltonianCycle(E._publishedCycleFromHamilton(8), 8),
  "8x8 Ham cycle is Hamiltonian");

/* Cell Tree: even safe step; odd = Odd until fill>=T then heal->Ham.
 * Cell Tree Odd: no Ham. */
(function () {
  var odd = new E.SnakeEnv({ size: 5, visionRadius: 10, compassOn: true });
  odd.reset({ seed: 42, size: 5, visionRadius: 10, compassOn: true });
  var hamOdd = E.masterHamiltonAction(odd);
  var ctOdd = E.cellTreeNextAction(odd);
  assert(odd.safeActions().indexOf(ctOdd) >= 0, "cell tree odd first step safe");
  assert(E.dhcrNextAction(odd) === hamOdd, "dhcr odd -> Hamilton");
  /* T=0 stickyHam: enter ham immediately regardless of fringe. */
  odd.food = [4, 4];
  odd._ctOddHamLock = false;
  odd._ctOddPhase = null;
  odd._ctOddLastSteps = odd.steps;
  E.cellTreeOddFillHandoffNextAction(odd, 0, "stickyHam");
  assert(odd._ctOddPhase === "ham" || odd._ctOddHamLock === true,
    "fill-handoff T=0 stickyHam enters ham");
  /* T=1: stay Odd even with fringe food. */
  var oddHi = new E.SnakeEnv({ size: 5, visionRadius: 10, compassOn: true });
  oddHi.reset({ seed: 42, size: 5, visionRadius: 10, compassOn: true });
  oddHi.food = [4, 4];
  oddHi._ctOddHamLock = false;
  oddHi._ctOddPhase = null;
  oddHi._ctOddLastSteps = oddHi.steps;
  var hiAct = E.cellTreeOddFillHandoffNextAction(oddHi, 1, "healOnly");
  assert(oddHi.safeActions().indexOf(hiAct) >= 0, "fill-handoff T=1 fringe step safe");
  assert(oddHi._ctOddPhase === "odd", "fill-handoff T=1 stays odd");
  assert(oddHi._ctOddHamLock !== true, "fill-handoff T=1 does not lock");
  /* Below shipped T: fringe food stays Odd. */
  var oddEarly = new E.SnakeEnv({ size: 5, visionRadius: 10, compassOn: true });
  oddEarly.reset({ seed: 42, size: 5, visionRadius: 10, compassOn: true });
  oddEarly.food = [4, 4];
  oddEarly._ctOddHamLock = false;
  oddEarly._ctOddPhase = null;
  oddEarly._ctOddLastSteps = oddEarly.steps;
  assert(oddEarly.snake.length / 25 < E.CELL_TREE_ODD_HAM_FILL_T,
    "start fill below shipped T");
  var earlyAct = E.cellTreeNextAction(oddEarly);
  assert(oddEarly.safeActions().indexOf(earlyAct) >= 0, "cell tree early fringe safe");
  assert(oddEarly._ctOddPhase === "odd", "cell tree below T stays odd");
  assert(typeof E.cellTreeOddNextAction === "function", "cell tree odd exported");
  assert(typeof E.masterSkipAfterMove === "function", "skipAfter exported");
  var oddCto = new E.SnakeEnv({ size: 5, visionRadius: 10, compassOn: true });
  oddCto.reset({ seed: 42, size: 5, visionRadius: 10, compassOn: true });
  oddCto.food = [4, 4];
  oddCto._ctOddHamLock = false;
  oddCto._ctOddLastSteps = oddCto.steps;
  var ctoFringe = E.cellTreeOddNextAction(oddCto);
  assert(oddCto.safeActions().indexOf(ctoFringe) >= 0, "cell tree odd fringe step safe");
  assert(oddCto._ctOddHamLock !== true, "cell tree odd fringe food does not sticky-Ham");
  var even = new E.SnakeEnv({ size: 6, visionRadius: 10, compassOn: true });
  even.reset({ seed: 17, size: 6, visionRadius: 10, compassOn: true });
  even.setWinFraction(0.5);
  var ct = E.cellTreeNextAction(even);
  var dh = E.dhcrNextAction(even);
  assert(even.safeActions().indexOf(ct) >= 0, "cell tree first step safe");
  assert(even.safeActions().indexOf(dh) >= 0, "dhcr first step safe");
  var steps = 0;
  var hardDeath = false;
  while (!even.done && steps < 400) {
    var a = E.cellTreeNextAction(even);
    var st = even.step(a);
    steps += 1;
    if (st.done && st.info && (st.info.death === "wall" || st.info.death === "body")) {
      hardDeath = true;
    }
  }
  assert(!hardDeath, "cell tree 6x6 seed17: no wall/body in 400 steps");
})();

/* DHCR livelock pose (6x6/372210): second look at same pose locks to Ham. */
(function () {
  var posePath = path.join(root, "tests", "fixtures", "dhcr-stall-pose.json");
  var pose = JSON.parse(fs.readFileSync(posePath, "utf8"));
  var env = new E.SnakeEnv({ size: pose.size, visionRadius: 10, compassOn: true });
  env.reset({ seed: pose.seed, size: pose.size, visionRadius: 10, compassOn: true });
  env.setWinFraction(1);
  env.snake = pose.snake.map(function (p) { return p.slice(); });
  env.food = pose.food.slice();
  env.facing = pose.facing;
  var ham = E.masterHamiltonAction(env);
  assert(ham === pose.expectHamAction,
    "dhcr-stall pose: Ham wants " + pose.expectHamName);
  E.dhcrImprovedNextAction(env);
  var again = E.dhcrImprovedNextAction(env);
  assert(again === pose.expectHamAction,
    "dhcr-stall pose: Improved pose-repeat SafeHam -> " + pose.expectHamName +
    "; got " + again);
  assert(env._dhcrSafeHam === true, "dhcr-stall pose: SafeHam lock set");
})();

/* Former DHCR stall seed must full-fill under DHCR Improved. */
(function () {
  var env = new E.SnakeEnv({ size: 6, visionRadius: 10, compassOn: true });
  env.reset({ seed: 372210, size: 6, visionRadius: 10, compassOn: true });
  env.setWinFraction(1);
  env.disableStepTimeout = true;
  var g = 0;
  var info = null;
  var prev = env.snake.length;
  var since = 0;
  while (!env.done && g < 20000) {
    g += 1;
    if (env.snake.length !== prev) {
      prev = env.snake.length;
      since = 0;
    } else {
      since += 1;
      if (since > 5 * 36) {
        break;
      }
    }
    info = env.step(E.dhcrImprovedNextAction(env));
    if (env.success) {
      break;
    }
  }
  var death = info && info.info && info.info.death;
  assert(env.success || env.snake.length >= 36,
    "DHCR Improved 6x6/372210 full-fills; death=" + death +
    " len=" + env.snake.length);
})();

assert(typeof E.hamiltonFillCappedNextAction === "function",
  "hamilton fill-capped helper exported");
assert(E.HAMILTON_AGG_FILL_CAP === 0.2, "agg fill cap is 20%");
assert(typeof E.masterLegacyNextAction === "function", "legacy master exported");
assert(typeof E.tailSafeNextAction === "function", "pure Tail Safe exported");
assert(typeof E.tailSafeHybridNextAction === "function", "Tail Safe hybrid exported");
/* Hamilton Aggressive returns a safe action while short. */
(function () {
  var env = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
  env.reset({ seed: 11, size: 10, visionRadius: 10, compassOn: true });
  var aggr = E.hamiltonAggrNextAction(env);
  assert(env.safeActions().indexOf(aggr) >= 0, "hamilton aggr returns safe");
  assert(env.snake.length / 100 < 0.2, "fixture still in early fill");
})();

/* 7x7 seed 130748369 @len 47: head north of food, tip also adj, freeN=0.
   Must take south into food (old defer + cycleNext phase-locked forever). */
var penultN = new E.SnakeEnv({ size: 7, visionRadius: 10, compassOn: true });
penultN.reset({ seed: 130748369, size: 7, visionRadius: 10, compassOn: true });
penultN.setWinFraction(1);
penultN.snake = [
  [4, 5], [3, 5], [3, 4], [3, 3], [3, 2], [3, 1], [3, 0], [2, 0], [2, 1],
  [2, 2], [2, 3], [2, 4], [2, 5], [1, 5], [1, 4], [1, 3], [1, 2], [1, 1],
  [1, 0], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 6],
  [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [6, 5], [6, 4], [6, 3], [6, 2],
  [6, 1], [6, 0], [5, 0], [4, 0], [4, 1], [5, 1], [5, 2], [4, 2], [4, 3],
  [5, 3], [5, 4]
];
penultN.facing = 2;
penultN.food = [5, 5];
assert(penultN.snake.length === 47, "penultN fixture len 47");
assert(penultN.snake[0][0] === 4 && penultN.food[0] === 5,
  "penultN: head is north of food");
assert(E.freeNeighborsAfter(penultN, 5, 5, true) < 1,
  "penultN: eat reports freeN=0 (tip still body)");
assert(Math.abs(penultN.snake[46][0] - 5) + Math.abs(penultN.snake[46][1] - 5) === 1,
  "penultN: tip adjacent to food");
assert(penultN.safeActions().indexOf(2) >= 0, "penultN: south into food is safe");
assert(E.masterHamiltonAction(penultN) === 2,
  "penultN: Hamilton goes south into tip-adj food");
assert(E.hamiltonAggrNextAction(penultN) === 2,
  "penultN: Hamilton Aggressive goes south into tip-adj food");

/* Same seed/food, head one cell south of food: must go north into food. */
var penultS = new E.SnakeEnv({ size: 7, visionRadius: 10, compassOn: true });
penultS.reset({ seed: 130748369, size: 7, visionRadius: 10, compassOn: true });
penultS.setWinFraction(1);
penultS.snake = [
  [6, 5], [6, 4], [6, 3], [6, 2], [6, 1], [6, 0], [5, 0], [4, 0], [4, 1],
  [5, 1], [5, 2], [4, 2], [4, 3], [5, 3], [5, 4], [4, 4], [4, 5], [3, 5],
  [3, 4], [3, 3], [3, 2], [3, 1], [3, 0], [2, 0], [2, 1], [2, 2], [2, 3],
  [2, 4], [2, 5], [1, 5], [1, 4], [1, 3], [1, 2], [1, 1], [1, 0], [0, 0],
  [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 6], [2, 6], [3, 6],
  [4, 6], [5, 6]
];
penultS.facing = 1;
penultS.food = [5, 5];
assert(penultS.snake.length === 47, "penultS fixture len 47");
assert(penultS.snake[0][0] === 6 && penultS.food[0] === 5,
  "penultS: head is south of food");
assert(E.freeNeighborsAfter(penultS, 5, 5, true) < 1,
  "penultS: eat reports freeN=0 (tip still body)");
assert(Math.abs(penultS.snake[46][0] - 5) + Math.abs(penultS.snake[46][1] - 5) === 1,
  "penultS: tip adjacent to food");
assert(penultS.safeActions().indexOf(0) >= 0, "penultS: north into food is safe");
assert(E.masterHamiltonAction(penultS) === 0,
  "penultS: Hamilton goes north into tip-adj food");
assert(E.hamiltonAggrNextAction(penultS) === 0,
  "penultS: Hamilton Aggressive goes north into tip-adj food");

/* Explicit riskScale override is not floored to 1. */
var riskEnv = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
riskEnv.reset({ seed: 9, size: 10, visionRadius: 10, compassOn: true });
var riskAct = E.masterHamiltonAction(riskEnv, { riskScale: 0.25 });
assert(typeof riskAct === "number" && riskAct >= 0 && riskAct <= 3,
  "explicit riskScale returns a legal action");

/* Odd Hamilton omits (n-1,n-1). When food is there and head is adjacent,
   take it even at length >> 2 (old need>2 gate skipped the corner forever). */
var cornerEnv = new E.SnakeEnv({ size: 5, visionRadius: 10, compassOn: true });
cornerEnv.reset({ seed: 7, size: 5, visionRadius: 10, compassOn: true });
var cy5 = E.getHamiltonCycle(5);
assert(cy5.omitCorner === true, "5x5 omits a corner");
assert(cy5.index["4,4"] === undefined, "5x5 bottom-right off cycle");
cornerEnv.snake = [
  [3, 4],
  [3, 3],
  [3, 2],
  [3, 1],
  [3, 0],
  [2, 0],
  [1, 0],
  [0, 0]
];
cornerEnv.facing = 2;
cornerEnv.food = [4, 4];
assert(cornerEnv.safeActions().indexOf(2) >= 0, "south into corner food is safe");
assert(E.masterHamiltonAction(cornerEnv) === 2,
  "Hamilton eats food on omitted odd-board corner");

/* Same situation as Observe seed 139525417 @15x15 len 133: head north of
   omit corner with food there; must step south, not continue west on cycle. */
var odd15 = new E.SnakeEnv({ size: 15, visionRadius: 10, compassOn: true });
odd15.reset({ seed: 139525417, size: 15, visionRadius: 10, compassOn: true });
assert(E.getHamiltonCycle(15).index["14,14"] === undefined, "15x15 omits BR");
odd15.snake = [[13, 14], [13, 13], [13, 12], [13, 11], [13, 10]];
odd15.facing = 3;
odd15.food = [14, 14];
assert(odd15.safeActions().indexOf(2) >= 0, "15x15 south into corner food safe");
assert(E.masterHamiltonAction(odd15) === 2,
  "15x15 Hamilton eats omit-corner food (seed 139525417 pose)");
assert(E.freeNeighborsAfter(odd15, 14, 14, true) >= 1,
  "15x15 corner eat leaves an escape");

/* Odd omit-BR cycle must cover every cell except BR (old even-(n-1) fallback
   left the right/bottom fringe off-cycle -> Ham body/stall on n>=21). */
[9, 11, 21, 31, 51].forEach(function (nOdd) {
  E.clearHamiltonCache();
  var cyOdd = E.getHamiltonCycle(nOdd);
  assert(cyOdd.omitCorner === true, nOdd + "x" + nOdd + " omitCorner");
  assert(cyOdd.len === nOdd * nOdd - 1, nOdd + "x" + nOdd + " cycle len");
  assert(cyOdd.index[(nOdd - 1) + "," + (nOdd - 1)] === undefined,
    nOdd + "x" + nOdd + " omits BR");
  var rr;
  var cc;
  for (rr = 0; rr < nOdd; rr += 1) {
    for (cc = 0; cc < nOdd; cc += 1) {
      if (rr === nOdd - 1 && cc === nOdd - 1) {
        continue;
      }
      assert(cyOdd.index[rr + "," + cc] !== undefined,
        nOdd + " covers " + rr + "," + cc);
    }
  }
});

/* Refuse omit-corner food when eating would trap the head (0 free neighbors)
   and the eat does not win. Full-board target so mid-length is not a win. */
var trapCorner = new E.SnakeEnv({ size: 5, visionRadius: 10, compassOn: true });
trapCorner.reset({ seed: 8, size: 5, visionRadius: 10, compassOn: true });
trapCorner.setWinFraction(1);
trapCorner.snake = [
  [3, 4],
  [3, 3],
  [4, 3],
  [4, 2],
  [4, 1],
  [4, 0],
  [3, 0],
  [2, 0],
  [1, 0]
];
trapCorner.facing = 2;
trapCorner.food = [4, 4];
assert(trapCorner.safeActions().indexOf(2) >= 0, "trap: south into food still immediate-safe");
assert(E.freeNeighborsAfter(trapCorner, 4, 4, true) < 1, "trap: corner eat has 0 escapes");
assert(trapCorner.snake.length + 1 < 5 * 5, "trap: eat is not a board fill");
assert(E.masterHamiltonAction(trapCorner) !== 2,
  "Hamilton refuses omit-corner food that traps the head");

/* Last open cell is omit-corner food: eat even with 0 escapes (board fill win). */
var lastCorner = new E.SnakeEnv({ size: 5, visionRadius: 10, compassOn: true });
lastCorner.reset({ seed: 9, size: 5, visionRadius: 10, compassOn: true });
lastCorner.setWinFraction(1);
var lastSnake = [];
var lr;
var lc;
for (lr = 0; lr < 5; lr += 1) {
  for (lc = 0; lc < 5; lc += 1) {
    if (lr === 4 && lc === 4) {
      continue;
    }
    lastSnake.push([lr, lc]);
  }
}
/* Head must be adjacent to corner: put (3,4) first, then a contiguous fill.
   Rebuild as row-major with head at (3,4). */
lastSnake = [[3, 4]];
for (lr = 0; lr < 5; lr += 1) {
  for (lc = 0; lc < 5; lc += 1) {
    if ((lr === 3 && lc === 4) || (lr === 4 && lc === 4)) {
      continue;
    }
    lastSnake.push([lr, lc]);
  }
}
assert(lastSnake.length === 24, "last-corner body covers all but food");
lastCorner.snake = lastSnake;
lastCorner.facing = 2;
lastCorner.food = [4, 4];
assert(E.freeNeighborsAfter(lastCorner, 4, 4, true) < 1,
  "last cell: no escape after eat");
assert(E.masterHamiltonAction(lastCorner) === 2,
  "Hamilton eats omit-corner food when it fills the board");

/* 11x11 omit-corner food with 0 escapes after eat - Hamilton must refuse. */
var odd11pose = new E.SnakeEnv({ size: 11, visionRadius: 10, compassOn: true });
odd11pose.reset({ seed: 1652672884, size: 11, visionRadius: 10, compassOn: true });
odd11pose.setWinFraction(1);
odd11pose.snake = [
  [9, 10], [9, 9], [10, 9], [10, 8], [10, 7], [10, 6], [10, 5], [10, 4],
  [10, 3], [10, 2], [10, 1], [10, 0], [9, 0], [8, 0], [7, 0]
];
odd11pose.facing = 2;
odd11pose.food = [10, 10];
assert(E.getHamiltonCycle(11).index["10,10"] === undefined, "11x11 omits BR");
assert(E.freeNeighborsAfter(odd11pose, 10, 10, true) < 1,
  "11x11 corner eat would trap");
assert(E.masterHamiltonAction(odd11pose) !== 2,
  "11x11 Hamilton refuses trapped omit-corner food");

/* 7x7 seed 205911: foodDist used to pull off-cycle (skip 4->36) then body-die
   @len 11. Prefer low skipAfter; must reach 50% without hard death. */
var ham205911 = new E.SnakeEnv({ size: 7, visionRadius: 10, compassOn: true });
ham205911.reset({ seed: 205911, size: 7, visionRadius: 10, compassOn: true });
ham205911.setWinFraction(0.5);
ham205911.disableStepTimeout = true;
var ham205911Guard = 0;
var ham205911Info = null;
while (!ham205911.done && ham205911Guard < 20000 &&
    ham205911.snake.length < ham205911.targetLength) {
  ham205911Guard += 1;
  ham205911Info = ham205911.step(E.masterHamiltonAction(ham205911));
}
var ham205911Death = ham205911Info && ham205911Info.info &&
  ham205911Info.info.death;
assert(!(ham205911Death === "wall" || ham205911Death === "body"),
  "7x7 seed 205911 Hamilton has no hard death before 50%");
assert(ham205911.snake.length >= ham205911.targetLength || ham205911.success,
  "7x7 seed 205911 Hamilton reaches 50% fill");

/* HealLA must take food on cycleNext even when flood < L-1 (7x7 seed
   827481772 @len 21: flood 9 < 20, previously walked south forever). */
var healFood = new E.SnakeEnv({ size: 7, visionRadius: 10, compassOn: true });
healFood.reset({ seed: 827481772, size: 7, visionRadius: 10, compassOn: true });
healFood.setWinFraction(1);
healFood.snake = [
  [1, 6], [1, 5], [1, 4], [1, 3], [1, 2], [1, 1], [2, 1], [2, 2], [2, 3],
  [2, 4], [2, 5], [3, 5], [3, 4], [3, 3], [3, 2], [3, 1], [3, 0], [4, 0],
  [4, 1], [5, 1], [5, 2]
];
healFood.facing = 1;
healFood.food = [0, 6];
healFood._planPhase = "heal";
assert(healFood.snake.length === 21, "heal-food fixture len 21");
assert(E.floodFromMove(healFood, 0, 6, true).count < 20,
  "heal-food: eat flood below L-1");
assert(E.healLookaheadNextAction(healFood) === 0,
  "HealLA eats food on cycleNext despite low flood");
assert(E.planAheadHybridNextAction(healFood) === 0,
  "planAhead hybrid eats cycleNext food in heal at len 21");

/* 7x7 seed 607678822 @step 114 pose: planned north (freeN=2) vs cycleSucc
   east (freeN=3) with large skip - must take east. Restored board, no replay.
   Pin oddWarn_s3: production oddOmitCorner_splice has succ north here, so the
   freeN-vs-succ gate would not fire on the default cycle. */
var planSucc = new E.SnakeEnv({ size: 7, visionRadius: 10, compassOn: true });
planSucc.reset({ seed: 607678822, size: 7, visionRadius: 10, compassOn: true });
planSucc.setWinFraction(1);
planSucc.snake = [
  [3, 3], [4, 3], [5, 3], [6, 3], [6, 2], [5, 2], [5, 1], [5, 0],
  [4, 0], [3, 0], [2, 0], [1, 0], [0, 0], [0, 1], [0, 2], [0, 3],
  [1, 3], [1, 2]
];
planSucc.facing = 0;
planSucc.food = [0, 4];
planSucc.apples = 15;
planSucc._planPhase = "plan";
planSucc._planAhead = {
  path: [[2, 3], [1, 3], [0, 3]],
  apples: 15,
  failed: false
};
(function pinPlanSuccCycle() {
  var vars = E.listHamiltonCycleVariants(7);
  var vi;
  for (vi = 0; vi < vars.length; vi += 1) {
    if (vars[vi].name === "oddWarn_s3" && vars[vi].path) {
      assert(E.setHamiltonCyclePath(7, vars[vi].path, "oddWarn_s3"),
        "pin oddWarn_s3 for plan-succ");
      return;
    }
  }
  assert(false, "oddWarn_s3 variant missing");
}());
assert(E.getHamiltonCycle(7).succ["3,3"][0] === 3 &&
  E.getHamiltonCycle(7).succ["3,3"][1] === 4,
  "plan-succ: pinned cycleSucc is east");
assert(E.masterCycleSkip(E.getHamiltonCycle(7), planSucc.snake) > 49,
  "plan-succ fixture has large skip");
assert(E.freeNeighborsAfter(planSucc, 2, 3, false) <
  E.freeNeighborsAfter(planSucc, 3, 4, false),
  "plan-succ: east wider than planned north");
assert(E.planAheadHybridNextAction(planSucc) === 1,
  "plan prefers wider east cycleSucc over north (seed 607678822)");
E.clearHamiltonCache();

/* Odd endgame: at len n*n-2, food must not spawn on the omit corner when
   another empty exists (seed 607678822 previously phase-locked there). */
var omitSpawn = new E.SnakeEnv({ size: 7, visionRadius: 10, compassOn: true });
omitSpawn.reset({ seed: 1, size: 7, visionRadius: 10, compassOn: true });
omitSpawn.snake = [
  [5, 5], [6, 5], [6, 4], [6, 3], [6, 2], [6, 1], [6, 0], [5, 0],
  [4, 0], [4, 1], [5, 1], [5, 2], [4, 2], [4, 3], [5, 3], [5, 4],
  [4, 4], [4, 5], [3, 5], [3, 4], [3, 3], [3, 2], [3, 1], [3, 0],
  [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [1, 5], [1, 4],
  [1, 3], [1, 2], [1, 1], [1, 0], [0, 0], [0, 1], [0, 2], [0, 3],
  [0, 4], [0, 5], [0, 6], [1, 6], [2, 6], [3, 6], [4, 6]
];
assert(omitSpawn.snake.length === 47, "omit-spawn fixture len 47");
var omitSpawnTrial;
for (omitSpawnTrial = 0; omitSpawnTrial < 40; omitSpawnTrial += 1) {
  omitSpawn.nextFoodFromStream();
  assert(!(omitSpawn.food[0] === 6 && omitSpawn.food[1] === 6),
    "len 47 food spawn skips omit corner when another empty exists");
}
assert(omitSpawn.food[0] === 5 && omitSpawn.food[1] === 6,
  "len 47 food lands on the only on-cycle empty");

/* Omit spawn fix is covered above; full-fill on 607678822 is planAhead durability
   (not Master) and is tracked in research / not asserted here. */

/* Odd omit protocol: food on far corner, head not adjacent -> follow cycle. */
(function () {
  var env = new E.SnakeEnv({ size: 5, visionRadius: 10, compassOn: true });
  env.reset({ seed: 1, size: 5, visionRadius: 10, compassOn: true });
  env.setWinFraction(1);
  env.snake = [[2, 2], [2, 1], [2, 0]];
  env.facing = 1;
  env.food = [4, 4];
  var cy = E.getHamiltonCycle(5);
  var succ = cy.succ["2,2"];
  assert(succ, "omit-approach has cycleSucc");
  var expect = null;
  var di;
  for (di = 0; di < 4; di += 1) {
    if (env.snake[0][0] + E.DIRS[di][0] === succ[0] &&
        env.snake[0][1] + E.DIRS[di][1] === succ[1]) {
      expect = di;
    }
  }
  assert(expect !== null, "omit-approach succ is a dir");
  assert(Math.abs(env.snake[0][0] - 4) + Math.abs(env.snake[0][1] - 4) > 1,
    "omit-approach head not bordering corner");
  assert(E.masterHamiltonAction(env) === expect,
    "omit food: follow cycle until head borders corner");
})();

/* Pin an odd cycle variant whose succ(head) matches pose.cycleNext (policy
   fixtures predate oddOmitCorner_splice geometry). */
function pinOddCycleForPose(pose) {
  var head = pose.snake[0];
  var want = pose.cycleNext;
  var vars = E.listHamiltonCycleVariants(pose.size);
  var vi;
  for (vi = 0; vi < vars.length; vi += 1) {
    if (!vars[vi].path) {
      continue;
    }
    if (!E.setHamiltonCyclePath(pose.size, vars[vi].path, vars[vi].name)) {
      continue;
    }
    var succ = E.getHamiltonCycle(pose.size).succ[head[0] + "," + head[1]];
    if (succ && succ[0] === want[0] && succ[1] === want[1]) {
      return vars[vi].name;
    }
  }
  return null;
}

/* 5x5/197992 strand pose: prefer cycleSucc N over off-cycle food E. */
(function () {
  var posePath = path.join(root, "tests", "fixtures",
    "ham-offcycle-eat-pose.json");
  var pose = JSON.parse(fs.readFileSync(posePath, "utf8"));
  var pinned = pinOddCycleForPose(pose);
  assert(pinned, "offcycle-eat pose: pinned cycle with cycleNext");
  var env = new E.SnakeEnv({ size: pose.size, visionRadius: 10, compassOn: true });
  env.reset({ seed: pose.seed, size: pose.size, visionRadius: 10, compassOn: true });
  env.setWinFraction(1);
  env.snake = pose.snake.map(function (p) { return p.slice(); });
  env.food = pose.food.slice();
  env.facing = pose.facing;
  assert(env.safeActions().indexOf(pose.expectAction) >= 0,
    "offcycle-eat pose: expect action safe");
  assert(env.safeActions().indexOf(pose.rejectAction) >= 0,
    "offcycle-eat pose: reject action also immediate-safe");
  var act = E.masterHamiltonAction(env);
  assert(act === pose.expectAction,
    "offcycle-eat pose: Hamilton takes cycle " + pose.expectName +
    " not food " + pose.rejectName + "; got " + act);
  E.clearHamiltonCache();
})();

/* 5x5/340534 late chord: prefer cycleSucc W over skip-raising S at fill>40%. */
(function () {
  var posePath = path.join(root, "tests", "fixtures",
    "ham-late-chord-pose.json");
  var pose = JSON.parse(fs.readFileSync(posePath, "utf8"));
  var pinned = pinOddCycleForPose(pose);
  assert(pinned, "late-chord pose: pinned cycle with cycleNext");
  var env = new E.SnakeEnv({ size: pose.size, visionRadius: 10, compassOn: true });
  env.reset({ seed: pose.seed, size: pose.size, visionRadius: 10, compassOn: true });
  env.setWinFraction(1);
  env.snake = pose.snake.map(function (p) { return p.slice(); });
  env.food = pose.food.slice();
  env.facing = pose.facing;
  assert(env.snake.length / (pose.size * pose.size) >= 0.4,
    "late-chord pose is past 40% fill");
  assert(env.safeActions().indexOf(pose.expectAction) >= 0,
    "late-chord pose: expect action safe");
  assert(env.safeActions().indexOf(pose.rejectAction) >= 0,
    "late-chord pose: reject action immediate-safe");
  var act = E.masterHamiltonAction(env);
  assert(act === pose.expectAction,
    "late-chord pose: Hamilton takes cycle " + pose.expectName +
    " not " + pose.rejectName + "; got " + act);
  E.clearHamiltonCache();
})();

/* 5x5 seed 15855: refuse freeN=0 on-cycle eat; take empty omit garage (south). */
var odd5pose = new E.SnakeEnv({ size: 5, visionRadius: 10, compassOn: true });
odd5pose.reset({ seed: 15855, size: 5, visionRadius: 10, compassOn: true });
odd5pose.setWinFraction(1);
odd5pose.snake = [
  [3, 4], [3, 3], [2, 3], [2, 2], [3, 2], [3, 1], [2, 1], [1, 1], [1, 2],
  [1, 3], [1, 4], [0, 4], [0, 3], [0, 2], [0, 1], [0, 0], [1, 0], [2, 0],
  [3, 0], [4, 0], [4, 1], [4, 2], [4, 3]
];
odd5pose.facing = 1;
odd5pose.food = [2, 4];
odd5pose._planPhase = "hamilton";
assert(odd5pose.snake.length === 23, "odd5 pose len 23");
assert(E.freeNeighborsAfter(odd5pose, 2, 4, true) < 1, "odd5 north eat traps");
assert(E.freeNeighborsAfter(odd5pose, 4, 4, false) >= 1, "odd5 omit garage escapes");
assert(E.masterHamiltonAction(odd5pose) === 2,
  "Hamilton takes omit garage over freeN=0 eat (5x5/15855)");
/* Force Hamilton phase so Master hybrid is past Tail Safe. */
odd5pose._tsStallPhase = "hamilton";
assert(E.masterHamiltonAction(odd5pose) === 2,
  "Hamilton omit garage still holds for Master stack");

/* Master hybrid: length handoff clears markers; skip0 in heal marks win. */
var stallEnv = new E.SnakeEnv({ size: 6, visionRadius: 10, compassOn: true });
stallEnv.reset({ seed: 3, size: 6, visionRadius: 10, compassOn: true });
stallEnv._tsStallSeen = Object.create(null);
stallEnv._tsStallSeen["pre-handoff"] = 1;
E.tailSafeStallHandoffNextAction(stallEnv, { handoffAtLen: stallEnv.snake.length });
assert(stallEnv._tsStallPhase === "heal", "length handoff enters heal");
assert(stallEnv._tsStallSeen &&
  Object.keys(stallEnv._tsStallSeen).length === 0,
  "heal entry clears pose markers");

/* Default Master handoff is 20% of board (10x10 => len 20). */
(function () {
  var env = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
  env.reset({ seed: 3, size: 10, visionRadius: 10, compassOn: true });
  var need = Math.ceil(10 * 10 * 0.2);
  assert(need === 20, "20% of 10x10 is 20");
  var path = [];
  var r;
  var c;
  for (r = 0; r < 10 && path.length < need; r += 1) {
    for (c = 0; c < 10 && path.length < need; c += 1) {
      path.push([r, c]);
    }
  }
  env.snake = path;
  env.facing = 1;
  env.food = [9, 9];
  env._occDirty = true;
  env._tsStallPhase = "tail";
  E.tailSafeHybridNextAction(env);
  assert(env._tsStallPhase === "heal",
    "default hybrid hands off at 20% fill; phase=" + env._tsStallPhase);
})();

/* Skip0 while in heal: enter Hamilton and follow cycle; no early win if
   length < targetLength (Observe Win-at). Bench may pass skip0EarlyWin. */
var skip0Env = new E.SnakeEnv({ size: 6, visionRadius: 10, compassOn: true });
skip0Env.reset({ seed: 3, size: 6, visionRadius: 10, compassOn: true });
skip0Env.setWinFraction(1);
skip0Env.disableStepTimeout = true;
skip0Env._tsStallPhase = "heal";
(function () {
  var cy = E.getHamiltonCycle(6);
  var path = cy.path;
  assert(path && path.length === 36, "6x6 hamilton path");
  skip0Env.snake = [path[0].slice(), path[path.length - 1].slice(), path[path.length - 2].slice()];
  skip0Env.facing = 0;
  skip0Env.food = [3, 3];
  assert(E.masterCycleSkip(cy, skip0Env.snake) === 0, "skip0 fixture");
  var act = E.tailSafeHybridNextAction(skip0Env);
  assert(skip0Env._tsStallPhase === "hamilton", "skip0 enters hamilton phase");
  assert(!skip0Env.success,
    "skip0 does not early-win when length < targetLength");
  assert(typeof act === "number", "hamilton returns a cycle action");
  /* Bench shortcut: skip0EarlyWin still allowed. */
  var earlyEnv = new E.SnakeEnv({ size: 6, visionRadius: 10, compassOn: true });
  earlyEnv.reset({ seed: 3, size: 6, visionRadius: 10, compassOn: true });
  earlyEnv.setWinFraction(1);
  earlyEnv._tsStallPhase = "heal";
  earlyEnv.snake = [path[0].slice(), path[path.length - 1].slice(), path[path.length - 2].slice()];
  earlyEnv.facing = 0;
  earlyEnv.food = [3, 3];
  E.tailSafeHybridNextAction(earlyEnv, { skip0EarlyWin: true });
  assert(earlyEnv.success && earlyEnv._tsStallPhase === "hamilton",
    "skip0EarlyWin marks success for benches");
})();

/* Observe Win-at sets targetLength; benches stop at Hamilton (skip0EarlyWin). */
(function () {
  assert(Math.ceil(0.6 * 8 * 8) === 39, "60% of 8x8 is length 39");
  var env = new E.SnakeEnv({ size: 8, visionRadius: 10, compassOn: true });
  env.reset({ seed: 427643, size: 8, visionRadius: 10, compassOn: true });
  env.setWinFraction(0.6);
  env.disableStepTimeout = true;
  assert(env.targetLength === 39, "Observe win-at 60% => targetLength 39");
  var g = 0;
  while (!env.done && g < 200000) {
    g += 1;
    env.step(E.tailSafeHybridBenchNextAction(env));
    if (env._tsHealStall) {
      break;
    }
  }
  assert(env.success && env._tsStallPhase === "hamilton",
    "bench path: 427643 reaches Hamilton skip0 win");
  assert(typeof E.tailSafeHybridBenchNextAction === "function",
    "tailSafeHybridBenchNextAction exported");
})();
/* Aggressive variants return safe actions. */
var aggNames = [
  "aggRisk025", "aggSkipSlack", "aggPocket",
  "aggFlood", "aggWallFood", "aggNetProgress",
  "aggSafe", "aggPocketSafe", "aggNetSafe"
];
var ag;
for (ag = 0; ag < aggNames.length; ag += 1) {
  var aggEnv = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
  aggEnv.reset({ seed: 11 + ag, size: 10, visionRadius: 10, compassOn: true });
  var aa = E.aggressiveHamiltonAction(aggEnv, aggNames[ag]);
  var aggSafeActs = aggEnv.safeActions();
  assert(aggSafeActs.indexOf(aa) >= 0 || !aggSafeActs.length,
    "aggressiveHamiltonAction " + aggNames[ag] + " is safe");
}
assert(typeof E.floodFromMove === "function", "floodFromMove exported");
assert(typeof E.passesEarlySafety === "function", "passesEarlySafety exported");
assert(typeof E.safeBfsNextAction === "function", "safeBfsNextAction exported");
var safeBfsEnv = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
safeBfsEnv.reset({ seed: 21, size: 10, visionRadius: 10, compassOn: true });
var sba = E.safeBfsNextAction(safeBfsEnv);
assert(safeBfsEnv.safeActions().indexOf(sba) >= 0, "safeBfs returns safe action");

assert(typeof E.planAheadNextAction === "function", "planAheadNextAction exported");
assert(typeof E.planAheadHybridNextAction === "function", "planAheadHybrid exported");
var paEnv = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
paEnv.reset({ seed: 42, size: 10, visionRadius: 10, compassOn: true });
var pa = E.planAheadNextAction(paEnv);
assert(paEnv.safeActions().indexOf(pa) >= 0, "planAhead returns safe action");
var paPath = E.planAheadBuild(paEnv, paEnv.snake.length);
assert(paPath && paPath.length === paEnv.snake.length, "planAheadBuild length L");
var hyEnv = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
hyEnv.reset({ seed: 43, size: 10, visionRadius: 10, compassOn: true });
assert(hyEnv._planPhase === "plan", "hybrid starts in plan phase");
var hy = E.planAheadHybridNextAction(hyEnv);
assert(hyEnv.safeActions().indexOf(hy) >= 0, "hybrid returns safe action");
assert(typeof E.planAheadHybridWallNextAction === "function", "hybridWall exported");
var hwEnv = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
hwEnv.reset({ seed: 44, size: 10, visionRadius: 10, compassOn: true });
var hw = E.planAheadHybridWallNextAction(hwEnv);
assert(hwEnv.safeActions().indexOf(hw) >= 0, "hybridWall returns safe action");
assert(typeof E.healLookaheadNextAction === "function", "healLookahead exported");
var hlEnv = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
hlEnv.reset({ seed: 45, size: 10, visionRadius: 10, compassOn: true });
var hl = E.healLookaheadNextAction(hlEnv);
assert(hlEnv.safeActions().indexOf(hl) >= 0, "healLookahead returns safe action");
assert(typeof E.healTipSimNextAction === "function", "healTipSim exported");
var ts = E.healTipSimNextAction(hlEnv, { maxDepth: 8 });
assert(ts === null || hlEnv.safeActions().indexOf(ts) >= 0, "healTipSim null or safe");
assert(typeof E.planAheadReenterNextAction === "function", "reenter exported");
var reEnv = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
reEnv.reset({ seed: 46, size: 10, visionRadius: 10, compassOn: true });
var re = E.planAheadReenterNextAction(reEnv);
assert(reEnv.safeActions().indexOf(re) >= 0, "reenter returns safe action");
assert(typeof E.tailSafeNextAction === "function", "tailSafe exported");
assert(typeof E.tailSafeBodyHugNextAction === "function", "tailSafeBodyHug exported");
assert(typeof E.planTailSafePoseOk === "function", "planTailSafePoseOk exported");
assert(typeof E.leaveTimeAt === "function", "leaveTimeAt exported");
assert(typeof E.foodDistAt === "function", "foodDistAt exported");
/* Idea H: food-rooted dist matches tip-vacate geodesic on open start. */
(function () {
  var env = new E.SnakeEnv({ size: 8, visionRadius: 10, compassOn: true });
  env.reset({ seed: 48, size: 8, visionRadius: 10, compassOn: true });
  var h = env.snake[0];
  var d = E.foodDistAt(env, h[0], h[1]);
  assert(d >= 0, "foodDistAt reaches head on open board");
  assert(typeof d === "number" && d === (d | 0), "foodDistAt is int");
  var nr = h[0] - 1;
  var nc = h[1];
  if (env.inBounds(nr, nc) && !env.isBody(nr, nc)) {
    var dStep = E.foodDistAt(env, nr, nc);
    assert(dStep >= 0 && Math.abs(dStep - d) <= 1,
      "neighbor foodDist within 1 of head");
  }
})();
/* Leave-time grid: tip=1 .. head=L; leave-aware food when tip-only is sealed. */
(function () {
  var env = new E.SnakeEnv({ size: 4, visionRadius: 10, compassOn: true });
  env.reset({ seed: 1, size: 4, visionRadius: 10, compassOn: true });
  assert(E.leaveTimeAt(env, env.snake[0][0], env.snake[0][1]) === 3,
    "reset head leave = L");
  assert(E.leaveTimeAt(env, env.snake[2][0], env.snake[2][1]) === 1,
    "reset tip leave = 1");
  assert(E.leaveTimeAt(env, env.snake[1][0], env.snake[1][1]) === 2,
    "reset neck leave = 2");
  /* Seal: only bridge to food is through body; tip-only BFS cannot reach.
     Leave-aware walks tip then leave=2 cell to food. */
  env.snake = [
    [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [1, 1], [0, 1], [0, 0]
  ];
  env.facing = 0;
  env.food = [0, 2];
  env._occDirty = true;
  assert(E.leaveTimeAt(env, 0, 0) === 1, "sealed tip leave=1");
  assert(E.leaveTimeAt(env, 0, 1) === 2, "sealed leave=2 cell");
  var act = E.tailSafeNextAction(env);
  assert(act === 0,
    "leave-aware food path steps onto tip first; got " +
      (E.ACTION_NAMES[act] || act));
})();
var tsEnv = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
tsEnv.reset({ seed: 48, size: 10, visionRadius: 10, compassOn: true });
var ts = E.tailSafeNextAction(tsEnv);
assert(tsEnv.safeActions().indexOf(ts) >= 0, "tailSafe returns safe action");
/* Body-hug prefers the cell with more body neighbors when both tip-safe. */
(function () {
  var env = new E.SnakeEnv({ size: 6, visionRadius: 10, compassOn: true });
  env.reset({ seed: 1, size: 6, visionRadius: 10, compassOn: true });
  /* H@(2,2) facing E; body west then north then east. Safe: E (bodyAdj 2 via
     head+(1,3)) vs S (bodyAdj 1 via head only). */
  env.snake = [[2, 2], [2, 1], [1, 1], [1, 2], [1, 3]];
  env.facing = 1;
  /* No food path shortcut - equal man; bodyAdj must pick E over S. */
  env.food = null;
  env._occDirty = true;
  var hug = E.tailSafeBodyHugNextAction(env);
  assert(hug === 1,
    "bodyHug prefers east along body; got " + (E.ACTION_NAMES[hug] || hug));
  var hugSafe = env.safeActions();
  assert(hugSafe.indexOf(hug) >= 0, "bodyHug returns safe action");
})();
/* After a tip-gated step, vacated tip stays reachable with flood >= L-1. */
var tsNr = tsEnv.snake[0][0] + [[-1, 0], [0, 1], [1, 0], [0, -1]][ts][0];
var tsNc = tsEnv.snake[0][1] + [[-1, 0], [0, 1], [1, 0], [0, -1]][ts][1];
var tsEat = tsEnv.food && tsEnv.food[0] === tsNr && tsEnv.food[1] === tsNc;
var tsAfter = [ [tsNr, tsNc] ];
var tsKeep = tsEat ? tsEnv.snake.length : tsEnv.snake.length - 1;
var tsj;
for (tsj = 0; tsj < tsKeep; tsj += 1) {
  tsAfter.push(tsEnv.snake[tsj].slice());
}
assert(E.planHeadReachesTip(10, tsAfter),
  "tailSafe first step keeps vacated tip reachable");
/* Vacated tip = tip cell treated empty (space tip leaves on next non-eat). */
assert(E.planHeadReachesTip(10, [[0, 1], [0, 0], [1, 0]]),
  "head reaches vacated tip along body");
assert(typeof E.planTailSafePoseOk === "function", "planTailSafePoseOk exported");

/* Tip-pocket escape pose fixture (captured once from 484644173). Tip-reach
   flood < n must yield to flood >= L-1. No live seed replay in this test. */
(function () {
  var posePath = path.join(root, "tests", "fixtures", "tip-pocket-pose.json");
  assert(fs.existsSync(posePath),
    "tip-pocket pose missing under tests/fixtures");
  var pose = JSON.parse(fs.readFileSync(posePath, "utf8"));
  var env = new E.SnakeEnv({ size: 10, visionRadius: 10, compassOn: true });
  env.reset({ seed: 1, size: 10, visionRadius: 10, compassOn: true });
  env.snake = pose.snake.map(function (c) { return c.slice(); });
  env.facing = pose.facing;
  env.food = pose.food ? pose.food.slice() : null;
  env.apples = Math.max(0, env.snake.length - 3);
  var i;
  for (i = 1; i < env.snake.length; i += 1) {
    var p = env.snake[i - 1];
    var q = env.snake[i];
    assert(Math.abs(p[0] - q[0]) + Math.abs(p[1] - q[1]) === 1,
      "tip-pocket snake contiguous at " + i);
  }
  assert(pose.tip && pose.tip.length, "tip-pocket pose has tip cand");
  assert(pose.esc && pose.esc.length, "tip-pocket pose has escape cand");
  var tipA = pose.tip[0].a;
  var escA = pose.esc[0].a;
  var tnr = env.snake[0][0] + (tipA === 0 ? -1 : tipA === 2 ? 1 : 0);
  var tnc = env.snake[0][1] + (tipA === 1 ? 1 : tipA === 3 ? -1 : 0);
  var enr = env.snake[0][0] + (escA === 0 ? -1 : escA === 2 ? 1 : 0);
  var enc = env.snake[0][1] + (escA === 1 ? 1 : escA === 3 ? -1 : 0);
  var floodTip = E.floodFromMove(env, tnr, tnc, false).count;
  var floodEsc = E.floodFromMove(env, enr, enc, false).count;
  var need = env.snake.length - 1;
  assert(floodTip < env.size, "tip-pocket tip flood < n got " + floodTip);
  assert(floodEsc >= need,
    "tip-pocket escape flood >= L-1 got " + floodEsc + " need=" + need);
  var act = E.tailSafeNextAction(env);
  assert(act === escA,
    "tailSafe tip-pocket takes escape; got " +
      (E.ACTION_NAMES[act] || act) + " want " + (E.ACTION_NAMES[escA] || escA));
})();

/* Equal max openings: food side preferred, else top-left (427643 rule). */
(function () {
  var posePath = path.join(root, "tests", "fixtures", "equal-open-pose.json");
  assert(fs.existsSync(posePath),
    "equal-open pose missing under tests/fixtures");
  var pose = JSON.parse(fs.readFileSync(posePath, "utf8"));
  var n = 6;
  var snake = pose.snake;
  var food = pose.food;
  var opens = E.planEmptyOpenings(n, snake, food, { vacateTip: false });
  assert(opens.tiedMax, "equal-open fixture tiedMax");
  assert(!!opens.preferredCells[food[0] + "," + food[1]],
    "food cell preferred");
  var opensNoFood = E.planEmptyOpenings(n, snake, null, { vacateTip: false });
  assert(opensNoFood.tiedMax, "no-food still tied");
  assert(!!opensNoFood.preferredCells[
    opensNoFood.cellMeta[
      Object.keys(opensNoFood.preferredCells)[0]
    ].topR + "," + opensNoFood.cellMeta[
      Object.keys(opensNoFood.preferredCells)[0]
    ].topC
  ], "no-food tie has a preferred opening");
  var prefTop = null;
  var k;
  for (k in opensNoFood.preferredCells) {
    if (Object.prototype.hasOwnProperty.call(opensNoFood.preferredCells, k)) {
      prefTop = opensNoFood.cellMeta[k];
      break;
    }
  }
  assert(prefTop, "no-food preferred meta");
  var otherTop = null;
  for (k in opensNoFood.cellMeta) {
    if (!Object.prototype.hasOwnProperty.call(opensNoFood.cellMeta, k)) {
      continue;
    }
    if (opensNoFood.cellMeta[k].size !== opensNoFood.maxSize) {
      continue;
    }
    if (opensNoFood.preferredCells[k]) {
      continue;
    }
    otherTop = opensNoFood.cellMeta[k];
    break;
  }
  if (otherTop) {
    assert(prefTop.topR < otherTop.topR ||
      (prefTop.topR === otherTop.topR && prefTop.topC <= otherTop.topC),
      "no-food tie prefers top-left opening");
  }

  var env = new E.SnakeEnv({ size: n, visionRadius: 10, compassOn: true });
  env.reset({ seed: 1, size: n, visionRadius: 10, compassOn: true });
  env.snake = snake.map(function (c) { return c.slice(); });
  env.facing = pose.facing;
  env.food = food.slice();
  env.apples = Math.max(0, env.snake.length - 3);
  var act = E.healTailSafeNextAction(env);
  assert(act === pose.foodSide,
    "HealTS picks food-side opening; got " +
      (E.ACTION_NAMES[act] || act) + " want " +
      (E.ACTION_NAMES[pose.foodSide] || pose.foodSide));
})();

/*
 * True dead-end pocket (sealed): flood(W) < L-1 so Plan and Heal must reject W
 * via the body-scale flood gate alone (no strandAvoid). Food SW so Manhattan
 * would prefer W if the gate were ignored.
 */
(function () {
  var n = 20;
  var topEnv = new E.SnakeEnv({ size: n, visionRadius: 10, compassOn: true });
  topEnv.reset({ seed: 99, size: n, visionRadius: 10, compassOn: true });
  var snake = [];
  var r;
  var c;
  function add(rr, cc) {
    if (snake.length) {
      var p = snake[snake.length - 1];
      if (Math.abs(p[0] - rr) + Math.abs(p[1] - cc) !== 1) {
        throw new Error("snake gap " + p + " -> " + [rr, cc]);
      }
    }
    snake.push([rr, cc]);
  }
  add(0, 18);
  add(1, 18);
  for (c = 17; c >= 10; c -= 1) add(1, c);
  for (r = 2; r <= 7; r += 1) {
    if (r % 2 === 0) {
      for (c = 10; c <= 18; c += 1) add(r, c);
    } else {
      for (c = 18; c >= 10; c -= 1) add(r, c);
    }
  }
  for (r = 7; r >= 1; r -= 1) add(r, 9);
  add(0, 9);
  add(0, 8);
  add(1, 8);
  add(2, 8);
  topEnv.snake = snake;
  topEnv.facing = 0;
  topEnv.food = [10, 10];
  topEnv.apples = Math.max(0, snake.length - 3);
  topEnv._planAhead = null;
  topEnv._planPhase = "plan";
  topEnv._planHandoffLen = null;
  var L = topEnv.snake.length;
  var floodW = E.floodFromMove(topEnv, 0, 17, false).count;
  var floodE = E.floodFromMove(topEnv, 0, 19, false).count;
  assert(floodW < L - 1, "sealed W is flood-dead (flood=" + floodW + " L-1=" + (L - 1) + ")");
  assert(floodE >= L - 1, "sealed E has body-scale space (flood=" + floodE + ")");
  var planAct = E.planAheadNextAction(topEnv, {
    handoff: true,
    healLookahead: true,
    reenterPlan: true,
    wallAvoid: true,
    strandAvoid: false
  });
  assert(planAct === 1, "Plan mode rejects flood-dead W; got " + (E.ACTION_NAMES[planAct] || planAct));
  topEnv._planPhase = "heal";
  topEnv._planAhead = null;
  var healAct = E.healLookaheadNextAction(topEnv, { strandAvoid: false });
  assert(healAct === 1, "Heal mode rejects flood-dead W; got " + (E.ACTION_NAMES[healAct] || healAct));
})();

/*
 * User preturn board (Observe death debug): row0 ..........oo......H.
 * Pocket (0,12..17) sealed by oo at (0,10)(0,11) + body wall rows1-7 cols10-18.
 * Food SW so Manhattan prefers W; Plan/Heal must pick E via flood < L-1 alone.
 * Snake lengthened past the ASCII via hang on col18 + col17 return.
 */
(function () {
  var n = 20;
  var topEnv = new E.SnakeEnv({ size: n, visionRadius: 10, compassOn: true });
  topEnv.reset({ seed: 99, size: n, visionRadius: 10, compassOn: true });
  var snake = [];
  var r;
  var c;
  function add(rr, cc) {
    if (snake.length) {
      var p = snake[snake.length - 1];
      if (Math.abs(p[0] - rr) + Math.abs(p[1] - cc) !== 1) {
        throw new Error("preturn snake gap " + p + " -> " + [rr, cc]);
      }
    }
    snake.push([rr, cc]);
  }
  add(0, 18);
  add(1, 18);
  for (c = 17; c >= 12; c -= 1) add(1, c);
  add(2, 12);
  for (c = 13; c <= 18; c += 1) add(2, c);
  add(3, 18);
  for (c = 17; c >= 12; c -= 1) add(3, c);
  add(4, 12);
  for (c = 13; c <= 18; c += 1) add(4, c);
  add(5, 18);
  for (c = 17; c >= 12; c -= 1) add(5, c);
  add(6, 12);
  for (c = 13; c <= 18; c += 1) add(6, c);
  add(7, 18);
  for (r = 8; r <= 13; r += 1) add(r, 18);
  add(13, 17);
  for (r = 12; r >= 7; r -= 1) add(r, 17);
  for (c = 16; c >= 10; c -= 1) add(7, c);
  for (r = 6; r >= 1; r -= 1) add(r, 10);
  add(0, 10);
  add(0, 11);
  add(1, 11);
  for (r = 2; r <= 6; r += 1) add(r, 11);
  topEnv.snake = snake;
  topEnv.facing = 0;
  topEnv.food = [10, 10];
  topEnv.apples = Math.max(0, snake.length - 3);
  topEnv._planAhead = null;
  topEnv._planPhase = "plan";
  topEnv._planHandoffLen = null;
  var L = topEnv.snake.length;
  assert(snake[0][0] === 0 && snake[0][1] === 18, "preturn head at (0,18)");
  assert(L > 66, "preturn snake longer than minimal block (L=" + L + ")");
  var floodW = E.floodFromMove(topEnv, 0, 17, false).count;
  var floodE = E.floodFromMove(topEnv, 0, 19, false).count;
  assert(floodW < L - 1, "preturn W flood-dead (flood=" + floodW + " L-1=" + (L - 1) + ")");
  assert(floodE >= L - 1, "preturn E body-scale (flood=" + floodE + ")");
  var planAct = E.planAheadNextAction(topEnv, {
    handoff: true,
    healLookahead: true,
    reenterPlan: true,
    wallAvoid: true,
    strandAvoid: false
  });
  assert(planAct === 1, "preturn Plan picks E; got " + (E.ACTION_NAMES[planAct] || planAct));
  topEnv._planPhase = "heal";
  topEnv._planAhead = null;
  var healAct = E.healLookaheadNextAction(topEnv, { strandAvoid: false });
  assert(healAct === 1, "preturn Heal picks E; got " + (E.ACTION_NAMES[healAct] || healAct));
})();

/*
 * HealLA freeNeigh gate: W is a 1-exit mouth into a large left pocket; E is
 * open (freeNeigh>=2). Both pass flood >= L-1 on 12x12; HealLA must pick E.
 */
(function () {
  var env = new E.SnakeEnv({ size: 12, visionRadius: 10, compassOn: true });
  env.reset({ seed: 1, size: 12, visionRadius: 10, compassOn: true });
  env.snake = [
    [5, 5],
    [6, 5],
    [6, 4],
    [7, 4],
    [8, 4],
    [8, 5],
    [8, 6],
    [8, 7],
    [8, 8],
    [8, 9],
    [8, 10],
    [8, 11],
    [7, 11],
    [6, 11],
    [5, 11],
    [4, 11],
    [3, 11],
    [2, 11],
    [1, 11],
    [0, 11],
    [0, 10],
    [0, 9],
    [0, 8],
    [0, 7],
    [0, 6],
    [0, 5],
    [0, 4],
    [1, 4],
    [2, 4],
    [3, 4],
    [4, 4],
    [4, 5]
  ];
  env.facing = 1;
  env.food = [11, 0];
  env.apples = Math.max(0, env.snake.length - 3);
  env._planPhase = "heal";
  env._planAhead = null;
  var L = env.snake.length;
  var need = L - 1;
  var safe = env.safeActions();
  assert(safe.indexOf(1) >= 0 && safe.indexOf(3) >= 0, "fixture E and W safe");
  var freeE = E.freeNeighborsAfter(env, 5, 6, false);
  var freeW = E.freeNeighborsAfter(env, 5, 4, false);
  var fE = E.floodFromMove(env, 5, 6, false).count;
  var fW = E.floodFromMove(env, 5, 4, false).count;
  assert(freeE >= 2, "fixture E wide freeNeigh=" + freeE);
  assert(freeW < 2, "fixture W narrow freeNeigh=" + freeW);
  assert(fE >= need && fW >= need,
    "fixture both floods pass (E=" + fE + " W=" + fW + " need=" + need + ")");
  var healAct = E.healLookaheadNextAction(env, { strandAvoid: false });
  assert(healAct === 1, "HealLA prefers wide E over narrow W; got " +
    (E.ACTION_NAMES[healAct] || healAct));
})();

/* Prior wall@5709 regression is the sealed / freeNeigh pose fixtures above
   (no multi-thousand-step seed replay). */

assert(E.isWallAdjacentCell(10, 0, 5) === true, "wall-adjacent top edge");
assert(E.isWallAdjacentCell(10, 5, 5) === false, "center not wall-adjacent");

/* Death snap ring keeps at most 4. */
var ring = E.createRingBuffer(4);
var ri;
for (ri = 0; ri < 6; ri += 1) {
  ring.push(ri);
}
assert(ring.size() === 4, "ring buffer caps at 4");
assert(ring.toArray().join(",") === "2,3,4,5", "ring drops oldest");
var snapEnv = new E.SnakeEnv({ size: 8, visionRadius: 10, compassOn: true });
snapEnv.reset({ seed: 3, size: 8, visionRadius: 10, compassOn: true });
var snap = E.captureAggBoardSnap(snapEnv, { lastAction: 0, variant: "aggPocket" });
assert(snap.snake.length >= 3 && snap.variant === "aggPocket", "captureAggBoardSnap");
assert(E.formatAggSnapAscii(snap).indexOf("H") >= 0, "ascii snap has head");
assert(typeof E.avoidanceHintFromSnaps([snap], "body") === "string",
  "avoidance hint string");

var legal = E.legalFromObs(obs);
assert(legal.length === 4, "all 4 actions legal");

/* Observe handoff fingerprint + fringe markers clear on reset. */
(function () {
  var env = new E.SnakeEnv({ size: 11, visionRadius: 2, compassOn: true });
  env.reset({ seed: 1800958793, size: 11, visionRadius: 2, compassOn: true });
  assert(env.food && env.food[0] === 2 && env.food[1] === 7,
    "handoff food0 for 1800958793@11 is (2,7), got " +
    (env.food ? "(" + env.food[0] + "," + env.food[1] + ")" : "null"));
  env._ctOddHamLock = true;
  env._ctBumpActions = [1, 2];
  env.reset({ seed: 1800958793, size: 11, visionRadius: 2, compassOn: true });
  assert(!env._ctOddHamLock, "reset clears _ctOddHamLock");
  assert(!env._ctBumpActions, "reset clears _ctBumpActions");
})();

console.log("test-snake-env: ok");
