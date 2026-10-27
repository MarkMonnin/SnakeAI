(function (global) {
  "use strict";

  var ACTIONS = 4;
  var SIZE_MIN = 4;
  var SIZE_MAX = 100;
  var VISION_MIN = 1;
  var VISION_MAX = 10;
  var DEFAULT_VISION = 2;

  function obsSizeForRadius(radius, compassOn) {
    var r = typeof radius === "number" ? radius : DEFAULT_VISION;
    if (r < VISION_MIN) {
      r = VISION_MIN;
    }
    if (r > VISION_MAX) {
      r = VISION_MAX;
    }
    var cells = (2 * r + 1) * (2 * r + 1) - 1;
    return cells + (compassOn ? 4 : 0);
  }

  /* Gen0 default: r=2, compass on => 24 + 4 = 28. */
  var OBS_SIZE = obsSizeForRadius(DEFAULT_VISION, true);

  function layerWithInput(hidden, outDim, inputDim) {
    var inp = typeof inputDim === "number" ? inputDim : OBS_SIZE;
    var sizes = [inp];
    var i;
    for (i = 0; i < hidden.length; i += 1) {
      sizes.push(hidden[i]);
    }
    sizes.push(outDim);
    return sizes;
  }

  var NN_POLICY_LAYER_SIZES = {
    2: layerWithInput([64], ACTIONS),
    3: layerWithInput([64, 56, 48], ACTIONS)
  };
  var NN_VALUE_LAYER_SIZES = {
    2: layerWithInput([64], 1),
    3: layerWithInput([64, 56, 48], 1)
  };
  var TFJS_DQN_LAYER_SIZES = {
    32: layerWithInput([32, 32], ACTIONS),
    64: layerWithInput([64, 64], ACTIONS),
    96: layerWithInput([96, 96], ACTIONS),
    256: layerWithInput([256, 256], ACTIONS)
  };
  var TFJS_DQN_SIZES = TFJS_DQN_LAYER_SIZES[64].slice();
  var TFJS_VALUE_LAYER_SIZES = {
    32: layerWithInput([32, 32], 1),
    64: layerWithInput([64, 64], 1),
    96: layerWithInput([96, 96], 1)
  };
  var TFJS_VALUE_SIZES = TFJS_VALUE_LAYER_SIZES[64].slice();

  global.SNAKE_CONSTANTS = {
    OBS_SIZE: OBS_SIZE,
    ACTIONS: ACTIONS,
    SIZE_MIN: SIZE_MIN,
    SIZE_MAX: SIZE_MAX,
    VISION_MIN: VISION_MIN,
    VISION_MAX: VISION_MAX,
    DEFAULT_VISION: DEFAULT_VISION,
    obsSizeForRadius: obsSizeForRadius,
    POLICY_INPUT_DIM: OBS_SIZE,
    VALUE_INPUT_DIM: OBS_SIZE,
    NN_POLICY_LAYER_SIZES: NN_POLICY_LAYER_SIZES,
    NN_VALUE_LAYER_SIZES: NN_VALUE_LAYER_SIZES,
    TFJS_DQN_LAYER_SIZES: TFJS_DQN_LAYER_SIZES,
    TFJS_DQN_SIZES: TFJS_DQN_SIZES,
    TFJS_VALUE_LAYER_SIZES: TFJS_VALUE_LAYER_SIZES,
    TFJS_VALUE_SIZES: TFJS_VALUE_SIZES,
    Q_MAX_STATES: 10000,
    MENACE_MAX_BOXES: 10000,
    TRAINING_ALGO_IDS: [
      "nn2", "nn3",
      "nn_value2", "nn_value3",
      "neat", "neat_value",
      "reinforce2", "reinforce3",
      "reinforce_value2", "reinforce_value3",
      "tfjs", "tfjs_value",
      "qtable", "sarsa", "menace", "genetic_menace"
    ],
    ALGO_NAMES: {
      nn2: "Backprop NN (2L)",
      nn3: "Backprop NN (3L)",
      nn_value2: "Backprop NN value (2L)",
      nn_value3: "Backprop NN value (3L)",
      neat: "NEAT",
      neat_value: "NEAT (value)",
      reinforce2: "REINFORCE (2L)",
      reinforce3: "REINFORCE (3L)",
      reinforce_value2: "REINFORCE value (2L)",
      reinforce_value3: "REINFORCE value (3L)",
      tfjs: "TF.js DQN",
      tfjs_value: "TF.js value",
      qtable: "Q-Learning",
      sarsa: "SARSA",
      menace: "MENACE",
      genetic_menace: "Genetic MENACE",
      random: "Random",
      bfs: "BFS",
      greedy: "Greedy",
      wall_follower: "Wall follower",
      avoid_pocket: "Avoid Pocket",
      master: "Master",
      hamilton: "Hamilton",
      hamilton_aggr: "Hamilton Aggressive",
      cell_tree: "Cell Tree",
      cell_tree_odd: "Cell Tree Odd",
      cell_tree_fringe: "Cell Tree Odd",
      cell_tree_odd_lock: "Cell Tree Odd",
      dhcr: "DHCR",
      dhcr_improved: "DHCR Improved",
      plan_ahead: "Plan Ahead",
      tail_safe: "Tail Safe",
      tail_safe_hybrid: "Tail Safe Hybrid"
    },
    PLAY_HUMAN: "human"
  };
})(typeof window !== "undefined" ? window : global);
