(function (global) {
  "use strict";

  function relu(x) {
    return x > 0 ? x : 0;
  }

  function reluDerivFromPre(pre) {
    return pre > 0 ? 1 : 0;
  }

  function randScale(fanIn) {
    return Math.sqrt(2 / fanIn);
  }

  function randomMatrix(rows, cols, scale) {
    var m = [];
    var r;
    var c;
    for (r = 0; r < rows; r += 1) {
      var row = [];
      for (c = 0; c < cols; c += 1) {
        row.push((Math.random() * 2 - 1) * scale);
      }
      m.push(row);
    }
    return m;
  }

  function zeroVector(size) {
    var v = [];
    var i;
    for (i = 0; i < size; i += 1) {
      v.push(0);
    }
    return v;
  }

  function matVecMul(matrix, vector) {
    var out = [];
    var r;
    var c;
    for (r = 0; r < matrix.length; r += 1) {
      var sum = 0;
      for (c = 0; c < vector.length; c += 1) {
        sum += matrix[r][c] * vector[c];
      }
      out.push(sum);
    }
    return out;
  }

  function vecAdd(a, b) {
    var out = [];
    var i;
    for (i = 0; i < a.length; i += 1) {
      out.push(a[i] + b[i]);
    }
    return out;
  }

  function NeuralNet(layerSizes, learningRate) {
    this.layerSizes = layerSizes.slice();
    this.learningRate = typeof learningRate === "number" ? learningRate : 0.03;
    this.weights = [];
    this.biases = [];
    var i;
    for (i = 0; i < layerSizes.length - 1; i += 1) {
      var fanIn = layerSizes[i];
      var fanOut = layerSizes[i + 1];
      this.weights.push(randomMatrix(fanOut, fanIn, randScale(fanIn)));
      this.biases.push(zeroVector(fanOut));
    }
    this.activations = [];
    this.preActivations = [];
  }

  NeuralNet.prototype.forward = function (input) {
    this.activations = [input.slice()];
    this.preActivations = [];
    var activation = input;
    var layer;
    for (layer = 0; layer < this.weights.length; layer += 1) {
      var pre = vecAdd(matVecMul(this.weights[layer], activation), this.biases[layer]);
      this.preActivations.push(pre);
      if (layer < this.weights.length - 1) {
        activation = pre.map(relu);
      } else {
        activation = pre.slice();
      }
      this.activations.push(activation);
    }
    return activation;
  };

  NeuralNet.prototype.trainOnAction = function (input, actionIndex, target) {
    var output = this.forward(input);
    var gradOutput = zeroVector(output.length);
    var error = output[actionIndex] - target;
    gradOutput[actionIndex] = error;
    var grad = gradOutput;
    var layer;
    for (layer = this.weights.length - 1; layer >= 0; layer -= 1) {
      var prevActivation = this.activations[layer];
      var weightGrad = [];
      var r;
      var c;
      for (r = 0; r < this.weights[layer].length; r += 1) {
        var rowGrad = [];
        for (c = 0; c < prevActivation.length; c += 1) {
          rowGrad.push(grad[r] * prevActivation[c]);
        }
        weightGrad.push(rowGrad);
      }
      for (r = 0; r < this.weights[layer].length; r += 1) {
        for (c = 0; c < this.weights[layer][r].length; c += 1) {
          this.weights[layer][r][c] -= this.learningRate * weightGrad[r][c];
        }
        this.biases[layer][r] -= this.learningRate * grad[r];
      }
      if (layer === 0) {
        break;
      }
      var nextGrad = zeroVector(prevActivation.length);
      var j;
      var k;
      for (j = 0; j < prevActivation.length; j += 1) {
        var sum = 0;
        for (k = 0; k < grad.length; k += 1) {
          sum += this.weights[layer][k][j] * grad[k];
        }
        nextGrad[j] = sum * reluDerivFromPre(this.preActivations[layer - 1][j]);
      }
      grad = nextGrad;
    }
    return error * error;
  };

  NeuralNet.prototype.trainOnOutputs = function (input, targets) {
    var output = this.forward(input);
    var gradOutput = [];
    var i;
    for (i = 0; i < output.length; i += 1) {
      gradOutput.push(output[i] - targets[i]);
    }
    var grad = gradOutput;
    var layer;
    for (layer = this.weights.length - 1; layer >= 0; layer -= 1) {
      var prevActivation = this.activations[layer];
      var r;
      var c;
      for (r = 0; r < this.weights[layer].length; r += 1) {
        for (c = 0; c < this.weights[layer][r].length; c += 1) {
          this.weights[layer][r][c] -= this.learningRate * grad[r] * prevActivation[c];
        }
        this.biases[layer][r] -= this.learningRate * grad[r];
      }
      if (layer === 0) {
        break;
      }
      var nextGrad = zeroVector(prevActivation.length);
      var j;
      var k;
      for (j = 0; j < prevActivation.length; j += 1) {
        var sum = 0;
        for (k = 0; k < grad.length; k += 1) {
          sum += this.weights[layer][k][j] * grad[k];
        }
        nextGrad[j] = sum * reluDerivFromPre(this.preActivations[layer - 1][j]);
      }
      grad = nextGrad;
    }
  };

  NeuralNet.prototype.getOutputs = function (input) {
    return this.forward(input).slice();
  };

  NeuralNet.prototype.clone = function () {
    var n = new NeuralNet(this.layerSizes, this.learningRate);
    n.load(this.toJSON());
    return n;
  };

  NeuralNet.prototype.mutate = function (scale) {
    var s = typeof scale === "number" ? scale : 0.05;
    var l;
    var r;
    var c;
    for (l = 0; l < this.weights.length; l += 1) {
      for (r = 0; r < this.weights[l].length; r += 1) {
        for (c = 0; c < this.weights[l][r].length; c += 1) {
          this.weights[l][r][c] += (Math.random() * 2 - 1) * s;
        }
        this.biases[l][r] += (Math.random() * 2 - 1) * s;
      }
    }
  };

  NeuralNet.prototype.toJSON = function () {
    return {
      layerSizes: this.layerSizes.slice(),
      weights: JSON.parse(JSON.stringify(this.weights)),
      biases: JSON.parse(JSON.stringify(this.biases)),
      learningRate: this.learningRate
    };
  };

  NeuralNet.prototype.load = function (data) {
    if (!data) {
      return;
    }
    this.layerSizes = data.layerSizes ? data.layerSizes.slice() : this.layerSizes;
    this.weights = data.weights || this.weights;
    this.biases = data.biases || this.biases;
    if (typeof data.learningRate === "number") {
      this.learningRate = data.learningRate;
    }
  };

  function argmaxLegal(outputs, legal, explore) {
    if (!legal || !legal.length) {
      return 0;
    }
    if (explore && Math.random() < 0.1) {
      return legal[Math.floor(Math.random() * legal.length)];
    }
    var best = -Infinity;
    var picks = [];
    var i;
    for (i = 0; i < legal.length; i += 1) {
      var a = legal[i];
      var v = outputs && typeof outputs[a] === "number" && isFinite(outputs[a]) ?
        outputs[a] : -1e9;
      if (v > best) {
        best = v;
        picks = [a];
      } else if (v === best) {
        picks.push(a);
      }
    }
    if (!picks.length) {
      return legal[0];
    }
    return picks[Math.floor(Math.random() * picks.length)];
  }

  function sampleSoftmax(logits, legal, temperature) {
    var T = temperature > 0 ? temperature : 1;
    var maxL = -Infinity;
    var i;
    for (i = 0; i < legal.length; i += 1) {
      if (logits[legal[i]] > maxL) {
        maxL = logits[legal[i]];
      }
    }
    var exps = [];
    var sum = 0;
    for (i = 0; i < legal.length; i += 1) {
      var e = Math.exp((logits[legal[i]] - maxL) / T);
      exps.push(e);
      sum += e;
    }
    var r = Math.random() * sum;
    var acc = 0;
    for (i = 0; i < legal.length; i += 1) {
      acc += exps[i];
      if (r <= acc) {
        return legal[i];
      }
    }
    return legal[legal.length - 1];
  }

  global.SNAKE_NN = {
    NeuralNet: NeuralNet,
    argmaxLegal: argmaxLegal,
    sampleSoftmax: sampleSoftmax,
    zeroVector: zeroVector
  };
})(typeof window !== "undefined" ? window : global);
