/**
 * Published Snake AI ports (twanvl / AlphaPhoenix), MIT-inspired.
 * Cell Tree: 2x2 right-hand drive + spanning-tree constraints + detour.
 * DHCR: Dynamic Hamiltonian Cycle Repair with safer Ham fallback.
 * Even boards: full cell tree. Odd: (n-1)^2 top-left subspace + sticky Ham.
 */
(function (global) {
  "use strict";

  var E = global.SNAKE_ENV;
  if (!E) {
    throw new Error("published-agents.js needs snake-env.js first");
  }

  var DIRS = E.DIRS;
  var ACTION_NAMES = E.ACTION_NAMES;
  var INF = 1e15;
  var NOT_VISITED = -1;
  var ROOT = -2;
  /* Reusable dijkstra / parents buffers (two slots so nested calls stay safe). */
  var _dijSlot = [
    { n: 0, dist: null, prev: null, seen: null },
    { n: 0, dist: null, prev: null, seen: null }
  ];
  var _dijWhich = 0;
  var _parSlot = [null, null];
  var _parLen = [0, 0];
  var _parWhich = 0;

  function cellMoveInside(r, c) {
    if ((r & 1) === 0) {
      return (c & 1) === 0 ? 2 : 3;
    }
    return (c & 1) === 0 ? 1 : 0;
  }

  function cellMoveOutside(r, c) {
    if ((r & 1) === 0) {
      return (c & 1) === 0 ? 3 : 0;
    }
    return (c & 1) === 0 ? 2 : 1;
  }

  function isCellMove(r, c, dir) {
    return cellMoveInside(r, c) === dir || cellMoveOutside(r, c) === dir;
  }

  function cellIdx(r, c, cw) {
    return ((r / 2) | 0) * cw + ((c / 2) | 0);
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

  /** Body set: tip free when not growing (matches isImmediateDeath). */
  function buildBodyOcc(env) {
    var n = env.size;
    var occ = new Uint8Array(n * n);
    var snake = env.snake;
    var i;
    var last = snake.length - 1;
    for (i = 0; i < last; i += 1) {
      occ[snake[i][0] * n + snake[i][1]] = 1;
    }
    if (last >= 0) {
      /* Tip blocked only if we treat eat-growth; pathfinding leaves tip free. */
      occ[snake[last][0] * n + snake[last][1]] = 0;
    }
    return occ;
  }

  function allocParents(need) {
    _parWhich ^= 1;
    var w = _parWhich;
    var parents = _parSlot[w];
    var i;
    if (!parents || _parLen[w] !== need) {
      parents = new Int32Array(need);
      _parSlot[w] = parents;
      _parLen[w] = need;
    }
    for (i = 0; i < need; i += 1) {
      parents[i] = NOT_VISITED;
    }
    return parents;
  }

  function cellTreeParents(n, snake) {
    var cw = (n / 2) | 0;
    var ch = (n / 2) | 0;
    var parents = allocParents(cw * ch);
    var parent = ROOT;
    var i;
    for (i = snake.length - 1; i >= 0; i -= 1) {
      var ci = cellIdx(snake[i][0], snake[i][1], cw);
      if (parents[ci] === NOT_VISITED) {
        parents[ci] = parent;
      }
      parent = ci;
    }
    return { parents: parents, cw: cw, ch: ch };
  }

  /**
   * Spanning tree for a subspace: ignore fringe body so tip-to-head parent
   * walk does not feed out-of-range cellIdx into the subM grid (corrupts CT).
   */
  function cellTreeParentsSub(subM, snake) {
    var cw = (subM / 2) | 0;
    var ch = (subM / 2) | 0;
    var parents = allocParents(cw * ch);
    var parent = ROOT;
    var any = false;
    var i;
    for (i = snake.length - 1; i >= 0; i -= 1) {
      var r = snake[i][0];
      var c = snake[i][1];
      if (r >= subM || c >= subM) {
        continue;
      }
      any = true;
      var ci = cellIdx(r, c, cw);
      if (parents[ci] === NOT_VISITED) {
        parents[ci] = parent;
      }
      parent = ci;
    }
    if (!any) {
      return cellTreeParents(subM, snake);
    }
    return { parents: parents, cw: cw, ch: ch };
  }

  /**
   * Same as cellTreeParentsSub on virtual snake [[nr,nc], ...snake[0..lim)],
   * without allocating the new snake array (fringeCtEscapeAfter hot path).
   */
  function cellTreeParentsSubAfterStep(subM, snake, nr, nc, eating) {
    var cw = (subM / 2) | 0;
    var ch = (subM / 2) | 0;
    var parents = allocParents(cw * ch);
    var parent = ROOT;
    var lim = eating ? snake.length : snake.length - 1;
    var i;
    for (i = lim - 1; i >= 0; i -= 1) {
      var r = snake[i][0];
      var c = snake[i][1];
      if (r >= subM || c >= subM) {
        continue;
      }
      var ci = cellIdx(r, c, cw);
      if (parents[ci] === NOT_VISITED) {
        parents[ci] = parent;
      }
      parent = ci;
    }
    if (nr < subM && nc < subM) {
      var hi = cellIdx(nr, nc, cw);
      if (parents[hi] === NOT_VISITED) {
        parents[hi] = parent;
      }
    }
    return { parents: parents, cw: cw, ch: ch };
  }

  /**
   * In-sub cells that can take a CT cell-move onto portal (entry pred).
   * Direct adjacency is not enough - e.g. (9,2) cannot W to (9,1).
   */
  function fringeCtPortalPreds(portal, m) {
    var preds = [];
    if (!portal) {
      return preds;
    }
    var deltas = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    var d;
    for (d = 0; d < deltas.length; d += 1) {
      var r = portal[0] + deltas[d][0];
      var c = portal[1] + deltas[d][1];
      if (!inTopLeftSub(r, c, m)) {
        continue;
      }
      var a = actionFromTo(r, c, portal[0], portal[1]);
      if (a !== null && isCellMove(r, c, a)) {
        preds.push([r, c]);
      }
    }
    return preds;
  }

  /** CT goal while approaching a portal: portal if on a pred, else nearest pred. */
  function fringeCtApproachGoal(portal, m, head) {
    if (!portal) {
      return null;
    }
    if (head[0] === portal[0] && head[1] === portal[1]) {
      return portal;
    }
    var preds = fringeCtPortalPreds(portal, m);
    var i;
    for (i = 0; i < preds.length; i += 1) {
      if (head[0] === preds[i][0] && head[1] === preds[i][1]) {
        return portal;
      }
    }
    if (!preds.length) {
      return portal;
    }
    var best = preds[0];
    var bestMan = Math.abs(head[0] - best[0]) + Math.abs(head[1] - best[1]);
    for (i = 1; i < preds.length; i += 1) {
      var man = Math.abs(head[0] - preds[i][0]) + Math.abs(head[1] - preds[i][1]);
      if (man < bestMan) {
        bestMan = man;
        best = preds[i];
      }
    }
    return best;
  }

  function fringeCtBroken(env, msg) {
    throw new Error(
      "Cell Tree Odd broken (must stay Cell-Tree-compatible): " + msg +
      " head=" + (env.snake[0] && env.snake[0].join(",")) +
      " food=" + (env.food && env.food.join(",")) +
      " len=" + env.snake.length +
      " steps=" + env.steps
    );
  }

  /** Odd Observe bots: never Hamilton - last-resort safe step. */
  function cellTreeOddNoHamFallback(env) {
    var safe = env.safeActions();
    if (safe.indexOf(env.facing) >= 0) {
      return env.facing | 0;
    }
    if (safe.length) {
      return safe[0];
    }
    return env.facing | 0;
  }

  /**
   * Last resorts that stay Cell-Tree-compatible or have an explicit return
   * plan: tip-vacate, win eat, tip-adj H-F-T eat, in-sub eat with CT escape.
   * No free-cruise / non-CT scramble (those break CT with no return plan).
   */
  function fringeLockExpOrThrow(env, n, safe, h, msg) {
    var tipAct = fringeTipVacateFresh(env, safe);
    if (tipAct !== null) {
      fringeTipWhy(env, "late-tip");
      return tipAct;
    }
    if (!env._ctOddLockVariant || !env.food) {
      fringeCtBroken(env, msg);
    }
    var food = env.food;
    var m = n - 1;
    var i;
    for (i = 0; i < safe.length; i += 1) {
      var act = safe[i];
      var nr = h[0] + DIRS[act][0];
      var nc = h[1] + DIRS[act][1];
      if (!(food[0] === nr && food[1] === nc)) {
        continue;
      }
      var winEat = env.snake.length + 1 >= n * n;
      if (winEat) {
        env._ctFringeWhy = "late-eat";
        return act;
      }
      if (fringeTipAdjFood(env)) {
        env._ctFringeWhy = "late-tip-eat";
        return act;
      }
      if (nr < m && nc < m && fringeCtEscapeAfter(env, m, act)) {
        env._ctFringeWhy = "late-eat-ct";
        return act;
      }
    }
    fringeCtBroken(env, msg);
  }

  /** Only safe escape: step onto tip (vacates). Tip may be on fringe after a splice. */
  function fringeTipVacate(env, safe) {
    var tip = env.snake[env.snake.length - 1];
    var h = env.snake[0];
    var i;
    for (i = 0; i < safe.length; i += 1) {
      var nr = h[0] + DIRS[safe[i]][0];
      var nc = h[1] + DIRS[safe[i]][1];
      if (nr === tip[0] && nc === tip[1]) {
        return safe[i];
      }
    }
    return null;
  }

  /**
   * Tip-vacate with livelock brake: after 2n tip steps without length growth,
   * refuse until the snake grows (728565 tip orbit). Streak clears only on
   * growth - a brief CT blip must not re-arm tip-chase.
   */
  function fringeTipVacateFresh(env, safe) {
    var n = env.size;
    if (env._ctOddTipLen !== env.snake.length) {
      env._ctOddTipLen = env.snake.length;
      env._ctOddTipStreak = 0;
    }
    if ((env._ctOddTipStreak || 0) > 2 * n) {
      return null;
    }
    return fringeTipVacate(env, safe);
  }

  function fringeTipWhy(env, why) {
    if (env._ctOddTipLen !== env.snake.length) {
      env._ctOddTipLen = env.snake.length;
      env._ctOddTipStreak = 0;
    }
    env._ctOddTipStreak = (env._ctOddTipStreak || 0) + 1;
    env._ctFringeWhy = why;
  }

  /**
   * When every safe in-sub step is sealed (or none), take a safe fringe leave
   * so headOut recovery can walk the rim until Cell Tree reentry (return plan).
   * Never fall back to a sealed in-sub eat - that breaks CT with no return.
   */
  function fringeForcedLeave(env, m, safe) {
    var h = env.snake[0];
    var i;
    for (i = 0; i < safe.length; i += 1) {
      var a = safe[i];
      if (!fringeStepStaysInSub(env, m, a)) {
        continue;
      }
      var nr = h[0] + DIRS[a][0];
      var nc = h[1] + DIRS[a][1];
      var eats = !!(env.food && env.food[0] === nr && env.food[1] === nc);
      if (eats && !fringeCtEscapeAfter(env, m, a)) {
        continue;
      }
      return null;
    }
    for (i = 0; i < safe.length; i += 1) {
      if (!fringeStepStaysInSub(env, m, safe[i])) {
        return safe[i];
      }
    }
    return null;
  }

  function canMoveInCellTree(tree, r, c, nr, nc, dir) {
    if (!isCellMove(r, c, dir)) {
      return false;
    }
    var ca = cellIdx(r, c, tree.cw);
    var cb = cellIdx(nr, nc, tree.cw);
    if (ca === cb) {
      return true;
    }
    if (tree.parents[cb] === NOT_VISITED) {
      return true;
    }
    return tree.parents[ca] === cb;
  }

  /**
   * Dijkstra with parent pointers. edgeFn(r,c,nr,nc,dir) -> cost or INF.
   * Returns { dist, prev, found }.
   */
  function dijkstraTo(n, sr, sc, gr, gc, edgeFn) {
    var N = n * n;
    _dijWhich ^= 1;
    var slot = _dijSlot[_dijWhich];
    var dist;
    var prev;
    var seen;
    var i;
    if (slot.n !== N || !slot.dist) {
      slot.n = N;
      slot.dist = new Float64Array(N);
      slot.prev = new Int32Array(N);
      slot.seen = new Uint8Array(N);
    }
    dist = slot.dist;
    prev = slot.prev;
    seen = slot.seen;
    for (i = 0; i < N; i += 1) {
      dist[i] = INF;
      prev[i] = -1;
      seen[i] = 0;
    }
    var start = sr * n + sc;
    var goal = gr * n + gc;
    dist[start] = 0;
    var heap = [];
    function heapPush(d, idx) {
      heap.push({ d: d, idx: idx });
      var j = heap.length - 1;
      while (j > 0) {
        var p = (j - 1) >> 1;
        if (heap[p].d <= heap[j].d) {
          break;
        }
        var t = heap[p];
        heap[p] = heap[j];
        heap[j] = t;
        j = p;
      }
    }
    function heapPop() {
      if (!heap.length) {
        return null;
      }
      var top = heap[0];
      var last = heap.pop();
      if (heap.length) {
        heap[0] = last;
        var j = 0;
        for (;;) {
          var l = j * 2 + 1;
          var rgt = l + 1;
          if (l >= heap.length) {
            break;
          }
          var best = l;
          if (rgt < heap.length && heap[rgt].d < heap[l].d) {
            best = rgt;
          }
          if (heap[j].d <= heap[best].d) {
            break;
          }
          var t = heap[j];
          heap[j] = heap[best];
          heap[best] = t;
          j = best;
        }
      }
      return top;
    }
    heapPush(0, start);
    while (heap.length) {
      var cur = heapPop();
      if (!cur || cur.d !== dist[cur.idx]) {
        continue;
      }
      if (seen[cur.idx]) {
        continue;
      }
      seen[cur.idx] = 1;
      if (cur.idx === goal) {
        break;
      }
      var r = (cur.idx / n) | 0;
      var c = cur.idx - r * n;
      var a;
      for (a = 0; a < 4; a += 1) {
        var nr = r + DIRS[a][0];
        var nc = c + DIRS[a][1];
        if (nr < 0 || nc < 0 || nr >= n || nc >= n) {
          continue;
        }
        var cost = edgeFn(r, c, nr, nc, a);
        if (!(cost < INF)) {
          continue;
        }
        var ni = nr * n + nc;
        var nd = cur.d + cost;
        if (nd < dist[ni]) {
          dist[ni] = nd;
          prev[ni] = cur.idx;
          heapPush(nd, ni);
        }
      }
    }
    return { dist: dist, prev: prev, found: dist[goal] < INF };
  }

  /** Path as [goal, ..., firstStep] (twanvl style; firstStep = path[path.length-1]). */
  function readPath(n, prev, sr, sc, gr, gc) {
    var goal = gr * n + gc;
    var start = sr * n + sc;
    if (prev[goal] < 0 && goal !== start) {
      return [];
    }
    var out = [];
    var cur = goal;
    var guard = 0;
    var N = n * n;
    while (cur !== start && cur >= 0 && guard < N + 2) {
      out.push([(cur / n) | 0, cur - ((cur / n) | 0) * n]);
      cur = prev[cur];
      guard += 1;
    }
    return out;
  }

  function firstStepAction(path, hr, hc) {
    if (!path.length) {
      return null;
    }
    var step = path[path.length - 1];
    return actionFromTo(hr, hc, step[0], step[1]);
  }

  /** Simulate path with tip vacating (many_move_tail). Path = [goal,...,first]. */
  function afterMoves(env, path) {
    var n = env.size;
    var snake = env.snake.slice();
    var food = env.food;
    var i;
    for (i = path.length - 1; i >= 0; i -= 1) {
      var p = path[i];
      var eating = !!(food && food[0] === p[0] && food[1] === p[1]);
      snake.unshift([p[0], p[1]]);
      if (!eating) {
        snake.pop();
      }
    }
    return { size: n, snake: snake, food: food };
  }

  function cellTreeUnreachables(after, tree, distArr, n) {
    var occ = new Uint8Array(n * n);
    var snake = after.snake;
    var i;
    for (i = 0; i < snake.length; i += 1) {
      occ[snake[i][0] * n + snake[i][1]] = 1;
    }
    /* Tip free for flood neighbor steps from head side. */
    if (snake.length) {
      var tip = snake[snake.length - 1];
      occ[tip[0] * n + tip[1]] = 0;
    }
    var h = snake[0];
    var seen = new Uint8Array(n * n);
    var q = [h[0] * n + h[1]];
    seen[q[0]] = 1;
    var qi = 0;
    while (qi < q.length) {
      var idx = q[qi];
      qi += 1;
      var r = (idx / n) | 0;
      var c = idx - r * n;
      var a;
      for (a = 0; a < 4; a += 1) {
        var nr = r + DIRS[a][0];
        var nc = c + DIRS[a][1];
        if (nr < 0 || nc < 0 || nr >= n || nc >= n) {
          continue;
        }
        var ni = nr * n + nc;
        if (seen[ni] || occ[ni]) {
          continue;
        }
        if (!canMoveInCellTree(tree, r, c, nr, nc, a)) {
          continue;
        }
        seen[ni] = 1;
        q.push(ni);
      }
    }
    var any = false;
    var nearest = -1;
    var bestD = INF;
    for (i = 0; i < n * n; i += 1) {
      if (occ[i]) {
        continue;
      }
      if (!seen[i]) {
        any = true;
        if (distArr[i] < bestD) {
          bestD = distArr[i];
          nearest = i;
        }
      }
    }
    return { any: any, nearest: nearest, dist: bestD };
  }

  /**
   * Cell-tree core on an even board size n (full grid).
   * opts.subM: if set, restrict play to top-left subM x subM (even).
   * opts.goal: optional [r,c] target inside subM (default: food).
   */
  function cellTreeEvenCore(env, n, opts) {
    opts = opts || {};
    var subM = typeof opts.subM === "number" ? opts.subM : n;
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var h = env.snake[0];
    var fr = opts.goal ? opts.goal[0] : env.food[0];
    var fc = opts.goal ? opts.goal[1] : env.food[1];
    if (h[0] >= subM || h[1] >= subM || fr >= subM || fc >= subM) {
      return null;
    }
    var occ = fringeCachedBodyOcc(env, n);
    var r;
    var c;
    /* Paint outside subM: O(n*(n-subM)) - for odd Lock subM=n-1 this is O(n). */
    if (subM < n) {
      for (r = 0; r < subM; r += 1) {
        for (c = subM; c < n; c += 1) {
          occ[r * n + c] = 1;
        }
      }
      for (r = subM; r < n; r += 1) {
        for (c = 0; c < n; c += 1) {
          occ[r * n + c] = 1;
        }
      }
    }
    /* Soft-avoid cells (e.g. fringe exit door) stay CT-legal without BFS. */
    if (opts.block) {
      var blocks = opts.block;
      if (!Array.isArray(blocks[0])) {
        blocks = [blocks];
      }
      var bi;
      for (bi = 0; bi < blocks.length; bi += 1) {
        var br = blocks[bi][0];
        var bc = blocks[bi][1];
        if (br >= 0 && bc >= 0 && br < n && bc < n) {
          occ[br * n + bc] = 1;
        }
      }
    }
    var tree = subM < n
      ? fringeCachedTreeCur(env, subM)
      : cellTreeParents(subM, env.snake);
    var edgeFn = function (r0, c0, nr, nc, dir) {
      if (nr >= subM || nc >= subM) {
        return INF;
      }
      if (occ[nr * n + nc]) {
        return INF;
      }
      if (!canMoveInCellTree(tree, r0, c0, nr, nc, dir)) {
        return INF;
      }
      var ca = cellIdx(r0, c0, tree.cw);
      var cb = cellIdx(nr, nc, tree.cw);
      var toParent = tree.parents[ca] === cb;
      var toSame = ca === cb;
      var pen = toParent ? 2 : (toSame ? 0 : 1);
      return 1000 + pen;
    };
    var sp = dijkstraTo(n, h[0], h[1], fr, fc, edgeFn);
    var path = sp.found ? readPath(n, sp.prev, h[0], h[1], fr, fc) : [];
    var act = firstStepAction(path, h[0], h[1]);

    /* Unreachable detour only on full even boards (fringe empties confuse subM). */
    if (subM === n && path.length && act !== null) {
      var after = afterMoves(env, path);
      var afterTree = cellTreeParents(subM, after.snake);
      var unr = cellTreeUnreachables(after, afterTree, sp.dist, n);
      if (unr.any && unr.nearest >= 0 && unr.dist < INF) {
        var ur = (unr.nearest / n) | 0;
        var uc = unr.nearest - ur * n;
        var det = dijkstraTo(n, h[0], h[1], ur, uc, edgeFn);
        if (det.found) {
          var dpath = readPath(n, det.prev, h[0], h[1], ur, uc);
          var dAct = firstStepAction(dpath, h[0], h[1]);
          if (dAct !== null && safe.indexOf(dAct) >= 0) {
            return dAct;
          }
        }
        var a;
        for (a = 0; a < 4; a += 1) {
          if (a === act) {
            continue;
          }
          var nr = h[0] + DIRS[a][0];
          var nc = h[1] + DIRS[a][1];
          if (nr < 0 || nc < 0 || nr >= n || nc >= n) {
            continue;
          }
          if (edgeFn(h[0], h[1], nr, nc, a) < INF && safe.indexOf(a) >= 0) {
            return a;
          }
        }
      }
    }

    if (act !== null && safe.indexOf(act) >= 0) {
      return act;
    }

    var b;
    for (b = 0; b < safe.length; b += 1) {
      var aa = safe[b];
      var rr = h[0] + DIRS[aa][0];
      var cc = h[1] + DIRS[aa][1];
      if (rr >= subM || cc >= subM) {
        continue;
      }
      if (isCellMove(h[0], h[1], aa) &&
          canMoveInCellTree(tree, h[0], h[1], rr, cc, aa)) {
        return aa;
      }
    }
    return null;
  }

  function snakeOutsideSub(env, subM) {
    var i;
    for (i = 0; i < env.snake.length; i += 1) {
      if (env.snake[i][0] >= subM || env.snake[i][1] >= subM) {
        return true;
      }
    }
    return false;
  }

  function inTopLeftSub(r, c, m) {
    return r >= 0 && c >= 0 && r < m && c < m;
  }

  function onOddFringe(r, c, m, n) {
    return r >= 0 && c >= 0 && r < n && c < n && !inTopLeftSub(r, c, m);
  }

  /**
   * Exit door free, tip, or within nonEats vacates of becoming tip.
   * From fringe alt (F) before F->G->exit, use nonEats=2; from G use 1.
   */
  function fringeExitReady(env, m, exitDoor, nonEats) {
    if (!exitDoor || !inTopLeftSub(exitDoor[0], exitDoor[1], m)) {
      return false;
    }
    var budget = typeof nonEats === "number" ? nonEats : 1;
    var head = env.snake[0];
    var tip = env.snake[env.snake.length - 1];
    var k = exitDoor[0] + "," + exitDoor[1];
    /* Tip on exit is ready. Head on exit is not (neck would seal the exit). */
    if (k === tip[0] + "," + tip[1]) {
      return true;
    }
    if (k === head[0] + "," + head[1]) {
      return false;
    }
    var len = env.snake.length;
    var i;
    for (i = 0; i < len - 1; i += 1) {
      if (env.snake[i][0] === exitDoor[0] && env.snake[i][1] === exitDoor[1]) {
        return (len - 1 - i) <= budget;
      }
    }
    return true;
  }

  /**
   * Fringe food as a CT doorA->doorB segment rewrite.
   * Edge: one CT step between doors; splice doorA->F->G->doorB.
   * Corner: longer CT path doorA->via->doorB; splice via fringe through F.
   * Only returns pairs where doorA->...->doorB is CT-legal.
   */
  function findFringeCtSegment(env, m) {
    var n = env.size;
    var food = env.food;
    if (!food || !onOddFringe(food[0], food[1], m, n)) {
      return null;
    }
    var fr = food[0];
    var fc = food[1];
    var tip = env.snake[env.snake.length - 1];
    var tipKey = tip[0] + "," + tip[1];
    var body = {};
    var i;
    for (i = 0; i < env.snake.length - 1; i += 1) {
      body[env.snake[i][0] + "," + env.snake[i][1]] = true;
    }
    function freeish(r, c) {
      if (r < 0 || c < 0 || r >= n || c >= n) {
        return false;
      }
      if (r === fr && c === fc) {
        return true;
      }
      var k = r + "," + c;
      return k === tipKey || !body[k];
    }
    function ctEdge(a, b) {
      var act = actionFromTo(a[0], a[1], b[0], b[1]);
      return act !== null && isCellMove(a[0], a[1], act) ? act : null;
    }
    function buildActions(cells) {
      var acts = [];
      var j;
      for (j = 0; j < cells.length - 1; j += 1) {
        var a = actionFromTo(cells[j][0], cells[j][1],
          cells[j + 1][0], cells[j + 1][1]);
        if (a === null) {
          return null;
        }
        acts.push(a);
      }
      return acts;
    }
    var cands = [];

    /* Bottom fringe: CT along row m-1 is always eastbound. */
    if (fr === m && fc < m) {
      if (fc + 1 < m && freeish(m, fc + 1)) {
        var dA1 = [m - 1, fc];
        var dB1 = [m - 1, fc + 1];
        if (ctEdge(dA1, dB1) !== null) {
          var spl1 = buildActions([dA1, [m, fc], [m, fc + 1], dB1]);
          if (spl1) {
            cands.push({
              kind: "edge-bottom-east",
              doorA: dA1,
              doorB: dB1,
              gCell: [m, fc + 1],
              splice: spl1
            });
          }
        }
      }
      if (fc - 1 >= 0 && freeish(m, fc - 1)) {
        var dA2 = [m - 1, fc - 1];
        var dB2 = [m - 1, fc];
        if (ctEdge(dA2, dB2) !== null) {
          var spl2 = buildActions([dA2, [m, fc - 1], [m, fc], dB2]);
          if (spl2) {
            cands.push({
              kind: "edge-bottom-west",
              doorA: dA2,
              doorB: dB2,
              gCell: [m, fc - 1],
              splice: spl2
            });
          }
        }
      }
      /*
       * Near-SE on bottom: F at (m,m-1). East G is the board corner (not an
       * in-sub door). When west G is blocked, same CT start/finish as
       * via->north: via->F->corner->right fringe->doorB.
       */
      if (fc === m - 1 && m >= 2 && !freeish(m, m - 2) &&
          freeish(m, m) && freeish(m - 1, m) && freeish(m - 2, m)) {
        var dAn = [m - 1, m - 1];
        var dBn = [m - 2, m - 1];
        if (ctEdge(dAn, dBn) !== null) {
          var splN = buildActions([
            dAn, [m, m - 1], [m, m], [m - 1, m], [m - 2, m], dBn
          ]);
          if (splN) {
            cands.push({
              kind: "near-corner-bottom",
              doorA: dAn,
              doorB: dBn,
              via: dAn,
              splice: splN,
              openCell: [m - 2, m - 2]
            });
          }
        }
      }
    }

    /* Right fringe: CT along col m-1 is always northbound. */
    if (fc === m && fr < m) {
      if (fr - 1 >= 0 && freeish(fr - 1, m)) {
        var dA3 = [fr, m - 1];
        var dB3 = [fr - 1, m - 1];
        if (ctEdge(dA3, dB3) !== null) {
          var spl3 = buildActions([dA3, [fr, m], [fr - 1, m], dB3]);
          if (spl3) {
            cands.push({
              kind: "edge-right-north",
              doorA: dA3,
              doorB: dB3,
              gCell: [fr - 1, m],
              splice: spl3
            });
          }
        }
      }
      if (fr + 1 < m && freeish(fr + 1, m)) {
        var dA4 = [fr + 1, m - 1];
        var dB4 = [fr, m - 1];
        if (ctEdge(dA4, dB4) !== null) {
          var spl4 = buildActions([dA4, [fr + 1, m], [fr, m], dB4]);
          if (spl4) {
            cands.push({
              kind: "edge-right-south",
              doorA: dA4,
              doorB: dB4,
              gCell: [fr + 1, m],
              splice: spl4
            });
          }
        }
      }
      /*
       * Near-SE on right: F at (m-1,m). South G is the board corner (not an
       * in-sub door). Same CT start/finish as via->north when north G is
       * blocked: via->F->corner->bottom->(m-1,m-2)->via is via-twice; use
       * the long return onto doorB instead.
       */
      if (fr === m - 1 && m >= 2 && !freeish(m - 2, m) &&
          freeish(m, m) && freeish(m, m - 1) && freeish(m, m - 2) &&
          freeish(m - 1, m - 2) && freeish(m - 2, m - 2)) {
        var dAr = [m - 1, m - 1];
        var dBr = [m - 2, m - 1];
        if (ctEdge(dAr, dBr) !== null) {
          var splR = buildActions([
            dAr, [m - 1, m], [m, m], [m, m - 1], [m, m - 2],
            [m - 1, m - 2], [m - 2, m - 2], dBr
          ]);
          if (splR) {
            cands.push({
              kind: "near-corner-right",
              doorA: dAr,
              doorB: dBr,
              via: dAr,
              splice: splR,
              openCell: null
            });
          }
        }
      }
    }

    /*
     * Corner food (m,m): doorA->fringe->F->fringe->via->doorB (via once only).
     * If already on the fringe rim beside F, fringeAdjFoodAction eats F.
     */
    if (fr === m && fc === m && m >= 2 &&
        freeish(m, m - 2) && freeish(m, m - 1) && freeish(m - 1, m)) {
      var dAc = [m - 1, m - 2];
      var via = [m - 1, m - 1];
      var dBc = [m - 2, m - 1];
      if (ctEdge(dAc, via) !== null && ctEdge(via, dBc) !== null) {
        var splC = buildActions([
          dAc, [m, m - 2], [m, m - 1], [m, m], [m - 1, m], via, dBc
        ]);
        if (splC) {
          cands.push({
            kind: "corner",
            doorA: dAc,
            doorB: dBc,
            via: via,
            splice: splC,
            openCell: [m - 2, m - 2]
          });
        }
      }
    }
    /*
     * Corner-via: when the long corner approach (m,m-2)/(m-1,m-2) is sealed
     * but via is free, splice from via: S->E(F)->N->N->W onto doorB.
     * Same CT reconnect via->doorB as the long corner (via once only).
     */
    if (fr === m && fc === m && m >= 2 &&
        freeish(m, m - 1) && freeish(m - 1, m) && freeish(m - 2, m) &&
        freeish(m - 2, m - 1)) {
      var dAv = [m - 1, m - 1];
      var dBv = [m - 2, m - 1];
      if (ctEdge(dAv, dBv) !== null) {
        var splV = buildActions([
          dAv, [m, m - 1], [m, m], [m - 1, m], [m - 2, m], dBv
        ]);
        if (splV) {
          cands.push({
            kind: "corner-via",
            doorA: dAv,
            doorB: dBv,
            via: dAv,
            splice: splV,
            openCell: null
          });
        }
      }
    }

    if (!cands.length) {
      return null;
    }
    var head = env.snake[0];
    /*
     * Prefer tip-ready; among those prefer corner splices (hit F, not the
     * 3-cell rim around it); else closest doorA.
     */
    var best = null;
    var bestScore = 1e15;
    var readyBest = null;
    var readyScore = 1e15;
    var ci;
    for (ci = 0; ci < cands.length; ci += 1) {
      var seg = cands[ci];
      var budget = seg.splice.length - 1;
      seg.ready = fringeExitReady(env, m, seg.doorB, budget);
      /*
       * Tip on G (fringe partner): enter now. Splice steps onto tip then
       * through F; doorB clears as the tip advances during the bump.
       */
      if (!seg.ready && seg.gCell &&
          tip[0] === seg.gCell[0] && tip[1] === seg.gCell[1]) {
        seg.ready = true;
      }
      if (!seg.openCell) {
        seg.openCell = fringeSegOpenCell(seg, m);
      }
      var manA = Math.abs(head[0] - seg.doorA[0]) +
        Math.abs(head[1] - seg.doorA[1]);
      var cornerBoost = (seg.kind === "corner" || seg.kind === "corner-long" ||
        seg.kind === "corner-via" ||
        seg.kind === "near-corner-bottom" || seg.kind === "near-corner-right")
        ? 0
        : 1000;
      var score = cornerBoost + manA;
      if (seg.ready && score < readyScore) {
        readyScore = score;
        readyBest = seg;
      }
      if (score < bestScore) {
        bestScore = score;
        best = seg;
      }
    }
    return readyBest || best;
  }

  function fringeInSubCount(env, m) {
    var i;
    var n = 0;
    for (i = 0; i < env.snake.length; i += 1) {
      if (env.snake[i][0] < m && env.snake[i][1] < m) {
        n += 1;
      }
    }
    return n;
  }

  function fringeTipAdjFood(env) {
    var food = env.food;
    if (!food) {
      return false;
    }
    var tip = env.snake[env.snake.length - 1];
    return Math.abs(tip[0] - food[0]) + Math.abs(tip[1] - food[1]) === 1;
  }

  /**
   * Max growth-pair depth toward SE. Depth 0 = SW + NE end pairs; depth d
   * uses fringe index d. Cap at m-4 so SE corner corridor
   * (m,m-2)/(m,m-1)/(m-2,m)/(m-1,m) stays free for corner splices.
   * Scales with board size (e.g. 99x99 => m=98, many depths).
   */
  function fringeGrowthMaxDepth(m) {
    return Math.max(0, m - 4);
  }

  /** Head immediately behind tip (Manhattan 1) - room to tip-vacate next. */
  function fringeHeadBehindTip(env) {
    var h = env.snake[0];
    var tip = env.snake[env.snake.length - 1];
    return Math.abs(h[0] - tip[0]) + Math.abs(h[1] - tip[1]) === 1;
  }

  /**
   * Subspace too full: keep claiming the next free fringe growth pair
   * (iterative depths toward SE). Food-out growth only in the high-fill
   * band (emptyLeft <= 8) when the food splice is not tip-ready - early
   * foodOut growth at ~101 broke CT; the len-114 tip-livelock is where
   * continuing inward pairs is needed.
   * Also: near end with H right behind T, claim a pair before food (slack
   * for unexpected growth) - stall survey 47531 / 87126.
   */
  function fringeNeedGrowth(env, m) {
    var insub = fringeInSubCount(env, m);
    var food = env.food;
    var foodOut = !!(food && (food[0] >= m || food[1] >= m));
    var emptyLeft = env.size * env.size - env.snake.length;
    if (insub >= m * m - 1) {
      if (foodOut) {
        if (emptyLeft > 8) {
          return false;
        }
        var foodSeg = findFringeCtSegment(env, m);
        if (foodSeg && foodSeg.ready) {
          return false;
        }
      }
      return true;
    }
    /*
     * Near-end H-behind-T grow-before-food (emptyLeft<=8): CT door splice.
     */
    if (fringeHeadBehindTip(env) && emptyLeft >= 2 && emptyLeft <= 8) {
      return true;
    }
    /*
     * Mid-fill H-behind-T with fringe food (fill>=0.75, emptyLeft 9..28):
     * only when a growth pair is already exit-ready. Bare needGrow caused
     * long ct-grow-wait detours and regressed 316777.
     */
    if (foodOut && fringeHeadBehindTip(env) &&
        emptyLeft > 8 && emptyLeft <= 28) {
      var fillMid = env.snake.length / (env.size * env.size);
      if (fillMid >= 0.75) {
        var midFoodSeg = findFringeCtSegment(env, m);
        if (!(midFoodSeg && midFoodSeg.ready)) {
          var midGrow = findFringeGrowthSegment(env, m);
          if (midGrow && midGrow.ready) {
            return true;
          }
        }
      }
    }
    if (insub < m * m - 2) {
      return false;
    }
    if (!food || foodOut) {
      return false;
    }
    var h = env.snake[0];
    var safe = env.safeActions();
    var a;
    for (a = 0; a < 4; a += 1) {
      if (h[0] + DIRS[a][0] === food[0] && h[1] + DIRS[a][1] === food[1] &&
          safe.indexOf(a) >= 0 && !fringeCtEscapeAfter(env, m, a)) {
        return true;
      }
    }
    return false;
  }

  /**
   * When fringe food sits on one rim, only grow pairs on the other rim so
   * the food-edge splice/G cells stay intact (same-edge foodOut growth was
   * breaking CT at ~113). Corner food: late emptyLeft<=5, or H-behind-T
   * emptyLeft<=8, allows both rims (maxDepth skips SE corridor).
   * In-sub food: both edges. Returns "bottom" | "right" | null (both) |
   * "none" (no growth).
   */
  function fringeGrowthOnlyEdge(env, m) {
    var food = env.food;
    if (!food) {
      return null;
    }
    var onBottom = food[0] === m;
    var onRight = food[1] === m;
    if (onBottom && onRight) {
      var emptyCorner = env.size * env.size - env.snake.length;
      if (emptyCorner <= 5 ||
          (fringeHeadBehindTip(env) && emptyCorner <= 8)) {
        return null;
      }
      return "none";
    }
    if (onBottom) {
      return "right";
    }
    if (onRight) {
      return "bottom";
    }
    return null;
  }

  /**
   * Next empty fringe-pair splice (CT doorA->doorB rewrite), iterative by
   * depth: for d = 0..maxDepth, try bottom index d then right index d
   * (SW, NE, then each step toward SE). optional onlyEdge limits which rim.
   * Scales with m (99x99 => many depths).
   */
  function findFringeGrowthSegment(env, m) {
    var n = env.size;
    var tip = env.snake[env.snake.length - 1];
    var tipKey = tip[0] + "," + tip[1];
    var onlyEdge = fringeGrowthOnlyEdge(env, m);
    if (onlyEdge === "none") {
      return null;
    }
    var body = {};
    var i;
    for (i = 0; i < env.snake.length - 1; i += 1) {
      body[env.snake[i][0] + "," + env.snake[i][1]] = true;
    }
    function freeish(r, c) {
      if (r < 0 || c < 0 || r >= n || c >= n) {
        return false;
      }
      var k = r + "," + c;
      return k === tipKey || !body[k];
    }
    function ctEdge(a, b) {
      var act = actionFromTo(a[0], a[1], b[0], b[1]);
      return act !== null && isCellMove(a[0], a[1], act) ? act : null;
    }
    function buildActions(cells) {
      var acts = [];
      var j;
      for (j = 0; j < cells.length - 1; j += 1) {
        var a = actionFromTo(cells[j][0], cells[j][1],
          cells[j + 1][0], cells[j + 1][1]);
        if (a === null) {
          return null;
        }
        acts.push(a);
      }
      return acts;
    }
    function tryBottom(depth) {
      var c = depth;
      if (c + 1 >= m || !freeish(m, c) || !freeish(m, c + 1)) {
        return null;
      }
      var dAb = [m - 1, c];
      var dBb = [m - 1, c + 1];
      if (ctEdge(dAb, dBb) === null) {
        return null;
      }
      var splB = buildActions([dAb, [m, c], [m, c + 1], dBb]);
      if (!splB) {
        return null;
      }
      return {
        kind: "grow-bottom",
        depth: depth,
        doorA: dAb,
        doorB: dBb,
        splice: splB,
        priority: depth * 2
      };
    }
    function tryRight(depth) {
      var r = depth;
      if (r + 1 >= m || !freeish(r, m) || !freeish(r + 1, m)) {
        return null;
      }
      /* CT northbound on col m-1: doorA south of doorB. */
      var dAr = [r + 1, m - 1];
      var dBr = [r, m - 1];
      if (ctEdge(dAr, dBr) === null) {
        return null;
      }
      var splR = buildActions([dAr, [r + 1, m], [r, m], dBr]);
      if (!splR) {
        return null;
      }
      return {
        kind: "grow-right",
        depth: depth,
        doorA: dAr,
        doorB: dBr,
        splice: splR,
        priority: depth * 2 + 1
      };
    }
    var cands = [];
    var maxD = fringeGrowthMaxDepth(m);
    var d;
    for (d = 0; d <= maxD; d += 1) {
      if (onlyEdge !== "right") {
        var bottom = tryBottom(d);
        if (bottom) {
          cands.push(bottom);
        }
      }
      if (onlyEdge !== "bottom") {
        var right = tryRight(d);
        if (right) {
          cands.push(right);
        }
      }
    }
    /*
     * Near-end H-behind-T: food-rim ban left no cands, but a free pair on the
     * food rim away from F (87126 right:2 vs F@8,10) - allow those CT splices.
     */
    if (!cands.length && onlyEdge !== "none" && onlyEdge !== null &&
        fringeHeadBehindTip(env) && (n * n - env.snake.length) <= 8) {
      var food = env.food;
      function farFromFood(seg) {
        if (!food) {
          return true;
        }
        var cells = seg.kind === "grow-bottom"
          ? [[m, seg.depth], [m, seg.depth + 1]]
          : [[seg.depth + 1, m], [seg.depth, m]];
        var ci;
        for (ci = 0; ci < cells.length; ci += 1) {
          var cr = cells[ci][0];
          var cc = cells[ci][1];
          if (cr === food[0] && cc === food[1]) {
            return false;
          }
          if (Math.abs(cr - food[0]) + Math.abs(cc - food[1]) <= 1) {
            return false;
          }
        }
        return true;
      }
      /* onlyEdge is the allowed (non-food) rim; food rim is the other. */
      for (d = 0; d <= maxD; d += 1) {
        if (onlyEdge === "bottom") {
          var rFood = tryRight(d);
          if (rFood && farFromFood(rFood)) {
            cands.push(rFood);
          }
        } else if (onlyEdge === "right") {
          var bFood = tryBottom(d);
          if (bFood && farFromFood(bFood)) {
            cands.push(bFood);
          }
        }
      }
    }
    if (!cands.length) {
      return null;
    }
    var head = env.snake[0];
    var best = null;
    var bestScore = 1e15;
    var readyBest = null;
    var readyScore = 1e15;
    var ci;
    for (ci = 0; ci < cands.length; ci += 1) {
      var seg = cands[ci];
      var budget = seg.splice.length - 1;
      seg.ready = fringeExitReady(env, m, seg.doorB, budget);
      if (!seg.openCell) {
        seg.openCell = fringeSegOpenCell(seg, m);
      }
      var manA = Math.abs(head[0] - seg.doorA[0]) +
        Math.abs(head[1] - seg.doorA[1]);
      var score = seg.priority * 100 + manA;
      if (seg.ready && score < readyScore) {
        readyScore = score;
        readyBest = seg;
      }
      if (score < bestScore) {
        bestScore = score;
        best = seg;
      }
    }
    return readyBest || best;
  }

  /* ---- Cell Tree Odd Lock (experimental variant) ----
   * At high fill, permanently rewrite CT doorA->doorB through fringe pairs
   * once the path has used them (SW, NE, then next depths toward SE).
   */
  var CT_ODD_LOCK_FILL = 0.5;

  function fringeLockEnsure(env) {
    if (!env._ctOddLocks) {
      env._ctOddLocks = [];
    }
    return env._ctOddLocks;
  }

  function fringeLockKey(edge, depth) {
    return edge + ":" + depth;
  }

  function fringeLockHas(env, edge, depth) {
    return fringeLockEnsure(env).indexOf(fringeLockKey(edge, depth)) >= 0;
  }

  function fringeLockAdd(env, edge, depth) {
    var k = fringeLockKey(edge, depth);
    var locks = fringeLockEnsure(env);
    if (locks.indexOf(k) < 0) {
      locks.push(k);
    }
  }

  /** Next unlocked pair in SW/NE-then-inward order. */
  function fringeLockNextTarget(env, m) {
    var maxD = fringeGrowthMaxDepth(m);
    var d;
    for (d = 0; d <= maxD; d += 1) {
      if (!fringeLockHas(env, "bottom", d)) {
        return { edge: "bottom", depth: d };
      }
      if (!fringeLockHas(env, "right", d)) {
        return { edge: "right", depth: d };
      }
    }
    return null;
  }

  /**
   * Build a doorA->fringe->fringe->doorB splice for one pair, or null if the
   * fringe cells are blocked / not a CT edge.
   */
  function fringeBuildPairSegment(env, m, edge, depth) {
    var n = env.size;
    var tip = env.snake[env.snake.length - 1];
    var tipKey = tip[0] + "," + tip[1];
    var body = {};
    var i;
    for (i = 0; i < env.snake.length - 1; i += 1) {
      body[env.snake[i][0] + "," + env.snake[i][1]] = true;
    }
    function freeish(r, c) {
      if (r < 0 || c < 0 || r >= n || c >= n) {
        return false;
      }
      var k = r + "," + c;
      return k === tipKey || !body[k];
    }
    function ctEdge(a, b) {
      var act = actionFromTo(a[0], a[1], b[0], b[1]);
      return act !== null && isCellMove(a[0], a[1], act) ? act : null;
    }
    function buildActions(cells) {
      var acts = [];
      var j;
      for (j = 0; j < cells.length - 1; j += 1) {
        var a = actionFromTo(cells[j][0], cells[j][1],
          cells[j + 1][0], cells[j + 1][1]);
        if (a === null) {
          return null;
        }
        acts.push(a);
      }
      return acts;
    }
    var doorA;
    var doorB;
    var splice;
    var kind;
    if (edge === "bottom") {
      if (depth + 1 >= m || !freeish(m, depth) || !freeish(m, depth + 1)) {
        return null;
      }
      doorA = [m - 1, depth];
      doorB = [m - 1, depth + 1];
      if (ctEdge(doorA, doorB) === null) {
        return null;
      }
      splice = buildActions([doorA, [m, depth], [m, depth + 1], doorB]);
      kind = "lock-bottom";
    } else {
      if (depth + 1 >= m || !freeish(depth, m) || !freeish(depth + 1, m)) {
        return null;
      }
      doorA = [depth + 1, m - 1];
      doorB = [depth, m - 1];
      if (ctEdge(doorA, doorB) === null) {
        return null;
      }
      splice = buildActions([doorA, [depth + 1, m], [depth, m], doorB]);
      kind = "lock-right";
    }
    if (!splice) {
      return null;
    }
    var seg = {
      kind: kind,
      edge: edge,
      depth: depth,
      doorA: doorA,
      doorB: doorB,
      splice: splice
    };
    var budget = splice.length - 1;
    seg.ready = fringeExitReady(env, m, doorB, budget);
    return seg;
  }

  /** If the snake already occupies the next target pair and head is at doorB, lock it. */
  function fringeLockOpportunistic(env, m) {
    var t = fringeLockNextTarget(env, m);
    if (!t) {
      return;
    }
    var h = env.snake[0];
    var doorB = t.edge === "bottom"
      ? [m - 1, t.depth + 1]
      : [t.depth, m - 1];
    if (h[0] !== doorB[0] || h[1] !== doorB[1]) {
      return;
    }
    var f0 = t.edge === "bottom" ? [m, t.depth] : [t.depth + 1, m];
    var f1 = t.edge === "bottom" ? [m, t.depth + 1] : [t.depth, m];
    var has0 = false;
    var has1 = false;
    var i;
    for (i = 0; i < env.snake.length; i += 1) {
      if (env.snake[i][0] === f0[0] && env.snake[i][1] === f0[1]) {
        has0 = true;
      }
      if (env.snake[i][0] === f1[0] && env.snake[i][1] === f1[1]) {
        has1 = true;
      }
    }
    if (has0 && has1) {
      fringeLockAdd(env, t.edge, t.depth);
    }
  }

  /**
   * Lock-variant policy: at fill >= CT_ODD_LOCK_FILL, keep routing locked
   * pairs through the fringe, and acquire the next free pair in order.
   */
  function fringeLockPairAction(env, n, m, safe, h) {
    var fill = env.snake.length / (n * n);
    if (fill < CT_ODD_LOCK_FILL) {
      return null;
    }
    fringeLockOpportunistic(env, m);

    function tryEnter(seg, acquiring) {
      if (!seg || !seg.ready) {
        return null;
      }
      if (h[0] !== seg.doorA[0] || h[1] !== seg.doorA[1]) {
        return null;
      }
      var first = seg.splice[0];
      if (safe.indexOf(first) < 0) {
        return null;
      }
      env._ctBumpActions = seg.splice.slice(1);
      if (!env._ctBumpActions.length) {
        env._ctBumpActions = null;
      }
      env._ctOddOpenCell = null;
      env._ctOddOpenFoodKey = null;
      if (acquiring) {
        env._ctOddLockPending = { edge: seg.edge, depth: seg.depth };
        env._ctFringeWhy = "enter-lock-new";
      } else {
        env._ctOddLockPending = null;
        env._ctFringeWhy = "enter-lock";
      }
      return first;
    }

    function tryApproach(seg, why) {
      if (!seg) {
        return null;
      }
      var avoid = fringeAvoidList(env, seg.ready ? null : seg.doorB);
      var goal = fringeCtApproachGoal(seg.doorA, m, h);
      var act = fringeCtToward(env, n, m, goal, avoid);
      if (act === null) {
        return null;
      }
      env._ctFringeWhy = why;
      return act;
    }

    /*
     * Yield when growth is needed (lock runs before growth in the policy).
     * Under emptyLeft<=20 pressure: re-walk locked pairs and seek next.
     * Mid-fill: enter next lock only if already at doorA (lets CT chase
     * food; opportunistic + growth-pending still commit locks).
     * Probe 11x11: wins 198 (Odd stalls@118) + 728; still stalls 132@116.
     */
    if (fringeNeedGrowth(env, m)) {
      return null;
    }
    var emptyLeft = n * n - env.snake.length;
    var pressure = emptyLeft <= 20;
    var maxD = fringeGrowthMaxDepth(m);
    var d;
    if (pressure) {
      for (d = 0; d <= maxD; d += 1) {
        if (fringeLockHas(env, "bottom", d)) {
          var entB = tryEnter(fringeBuildPairSegment(env, m, "bottom", d),
            false);
          if (entB !== null) {
            return entB;
          }
        }
        if (fringeLockHas(env, "right", d)) {
          var entR = tryEnter(fringeBuildPairSegment(env, m, "right", d),
            false);
          if (entR !== null) {
            return entR;
          }
        }
      }
    }

    var next = fringeLockNextTarget(env, m);
    if (next) {
      var segN = fringeBuildPairSegment(env, m, next.edge, next.depth);
      var entN = tryEnter(segN, true);
      if (entN !== null) {
        return entN;
      }
      if (pressure) {
        var apN = tryApproach(segN,
          segN && segN.ready ? "ct-to-lock-new" : "ct-lock-wait");
        if (apN !== null) {
          return apN;
        }
      }
    }
    return null;
  }

  /**
   * Lock endgame: empty CT-aligned 2x2 holes (even,even origin) still inside
   * the subspace. Goal is a near-spanning path with one omit cell (usually
   * food) - weave these holes into the tour before tip-vacate livelocks.
   */
  /**
   * Per-decision memo for Odd Lock fringe path. Same results; avoids repeat
   * O(m^2) hole/empty scans and repeat current-snake CT trees within one step.
   * Disable with env._ctOddFringeCache === false for A/B timing.
   */
  function fringeDecCacheOn(env) {
    return env._ctOddFringeCache !== false;
  }

  function fringeDecCacheBegin(env, m) {
    if (!fringeDecCacheOn(env)) {
      env._ctOddDecCache = null;
      return;
    }
    env._ctOddDecCache = {
      steps: env.steps,
      len: env.snake.length,
      m: m,
      treeCur: null,
      bodyOcc: null,
      holes: null,
      emptiesInSub: null,
      emptiesAll: null,
      esc: Object.create(null),
      noSeal: Object.create(null)
    };
  }

  /** Mutable copy of tip-free body occ for CT pathfinding (same as buildBodyOcc). */
  function fringeCachedBodyOcc(env, n) {
    var c = fringeDecCacheGet(env, n - 1);
    if (c && c.bodyOcc) {
      return new Uint8Array(c.bodyOcc);
    }
    var occ = buildBodyOcc(env);
    if (c) {
      c.bodyOcc = new Uint8Array(occ);
    }
    return occ;
  }

  function fringeDecCacheGet(env, m) {
    var c = env._ctOddDecCache;
    if (!c || !fringeDecCacheOn(env)) {
      return null;
    }
    if (c.steps !== env.steps || c.len !== env.snake.length || c.m !== m) {
      return null;
    }
    return c;
  }

  function fringeCachedTreeCur(env, m) {
    var c = fringeDecCacheGet(env, m);
    if (!c) {
      return cellTreeParentsSub(m, env.snake);
    }
    if (!c.treeCur) {
      var t = cellTreeParentsSub(m, env.snake);
      /* Snapshot: parent scratch buffers are reused by later escape trees. */
      c.treeCur = {
        parents: new Int32Array(t.parents),
        cw: t.cw,
        ch: t.ch
      };
    }
    return c.treeCur;
  }

  function fringeEmptyCtHoles(env, m) {
    var c = fringeDecCacheGet(env, m);
    if (c && c.holes) {
      return c.holes;
    }
    var occ = new Uint8Array(m * m);
    var i;
    for (i = 0; i < env.snake.length; i += 1) {
      var sr = env.snake[i][0];
      var sc = env.snake[i][1];
      if (sr < m && sc < m) {
        occ[sr * m + sc] = 1;
      }
    }
    var holes = [];
    var r0;
    var c0;
    for (r0 = 0; r0 + 1 < m; r0 += 2) {
      for (c0 = 0; c0 + 1 < m; c0 += 2) {
        if (!occ[r0 * m + c0] && !occ[r0 * m + (c0 + 1)] &&
            !occ[(r0 + 1) * m + c0] && !occ[(r0 + 1) * m + (c0 + 1)]) {
          holes.push([r0, c0]);
        }
      }
    }
    if (c) {
      c.holes = holes;
    }
    return holes;
  }

  function fringeInEmptyCtHole(holes, r, c) {
    var i;
    for (i = 0; i < holes.length; i += 1) {
      var r0 = holes[i][0];
      var c0 = holes[i][1];
      if (r >= r0 && r <= r0 + 1 && c >= c0 && c <= c0 + 1) {
        return holes[i];
      }
    }
    return null;
  }

  function fringeCtCellEmptyCount(env, r0, c0) {
    var er;
    var ec;
    var nEmpty = 0;
    for (er = r0; er <= r0 + 1; er += 1) {
      for (ec = c0; ec <= c0 + 1; ec += 1) {
        var occ = false;
        var si;
        for (si = 0; si < env.snake.length; si += 1) {
          if (env.snake[si][0] === er && env.snake[si][1] === ec) {
            occ = true;
            break;
          }
        }
        if (!occ) {
          nEmpty += 1;
        }
      }
    }
    return nEmpty;
  }

  /** In-sub empties excluding food (omit cell stays out of the weave). */
  function fringeNonFoodEmptiesInSub(env, m) {
    var c = fringeDecCacheGet(env, m);
    if (c && c.emptiesInSub) {
      return c.emptiesInSub;
    }
    var food = env.food;
    var occ = new Uint8Array(m * m);
    var i;
    for (i = 0; i < env.snake.length; i += 1) {
      var sr = env.snake[i][0];
      var sc = env.snake[i][1];
      if (sr < m && sc < m) {
        occ[sr * m + sc] = 1;
      }
    }
    var out = [];
    var r;
    var cc;
    for (r = 0; r < m; r += 1) {
      for (cc = 0; cc < m; cc += 1) {
        if (occ[r * m + cc]) {
          continue;
        }
        if (food && food[0] === r && food[1] === cc) {
          continue;
        }
        out.push([r, cc]);
      }
    }
    if (c) {
      c.emptiesInSub = out;
    }
    return out;
  }

  /** 4-connected components among cells (each cell [r,c]). */
  function fringeCellComponents(cells) {
    var key = {};
    var i;
    for (i = 0; i < cells.length; i += 1) {
      key[cells[i][0] + "," + cells[i][1]] = i;
    }
    var seen = {};
    var comps = [];
    for (i = 0; i < cells.length; i += 1) {
      var sk = cells[i][0] + "," + cells[i][1];
      if (seen[sk]) {
        continue;
      }
      var stack = [cells[i]];
      var comp = [];
      seen[sk] = true;
      while (stack.length) {
        var cur = stack.pop();
        comp.push(cur);
        var d;
        for (d = 0; d < 4; d += 1) {
          var nr = cur[0] + DIRS[d][0];
          var nc = cur[1] + DIRS[d][1];
          var nk = nr + "," + nc;
          if (key[nk] === undefined || seen[nk]) {
            continue;
          }
          seen[nk] = true;
          stack.push([nr, nc]);
        }
      }
      comps.push(comp);
    }
    return comps;
  }

  function fringeCellInList(list, r, c) {
    var i;
    for (i = 0; i < list.length; i += 1) {
      if (list[i][0] === r && list[i][1] === c) {
        return true;
      }
    }
    return false;
  }

  function fringeOnOddFringeCell(r, c, m, n) {
    return r >= 0 && c >= 0 && r < n && c < n && (r >= m || c >= m);
  }

  /** Board empties excluding food. */
  function fringeNonFoodEmptiesAll(env, n) {
    var mSub = n - 1;
    var c = fringeDecCacheGet(env, mSub);
    if (c && c.emptiesAll) {
      return c.emptiesAll;
    }
    var food = env.food;
    var occ = new Uint8Array(n * n);
    var i;
    for (i = 0; i < env.snake.length; i += 1) {
      occ[env.snake[i][0] * n + env.snake[i][1]] = 1;
    }
    var out = [];
    var r;
    var cc;
    for (r = 0; r < n; r += 1) {
      for (cc = 0; cc < n; cc += 1) {
        if (occ[r * n + cc]) {
          continue;
        }
        if (food && food[0] === r && food[1] === cc) {
          continue;
        }
        out.push([r, cc]);
      }
    }
    if (c) {
      c.emptiesAll = out;
    }
    return out;
  }

  /**
   * After act, does the new head still have an empty or tip neighbor?
   * Stronger than CT-escape alone (blocks delayed hole-fill seals).
   */
  function fringeLockNoSeal(env, act) {
    var n = env.size;
    var cache = fringeDecCacheGet(env, n - 1);
    if (cache && cache.noSeal[act] !== undefined) {
      return cache.noSeal[act];
    }
    var h = env.snake[0];
    var tip = env.snake[env.snake.length - 1];
    var nr = h[0] + DIRS[act][0];
    var nc = h[1] + DIRS[act][1];
    var eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
    var occ = new Uint8Array(n * n);
    var lim = eating ? env.snake.length : env.snake.length - 1;
    var i;
    for (i = 0; i < lim; i += 1) {
      occ[env.snake[i][0] * n + env.snake[i][1]] = 1;
    }
    occ[nr * n + nc] = 1;
    var a;
    var ok = false;
    for (a = 0; a < 4; a += 1) {
      var r2 = nr + DIRS[a][0];
      var c2 = nc + DIRS[a][1];
      if (r2 < 0 || c2 < 0 || r2 >= n || c2 >= n) {
        continue;
      }
      if (!eating && r2 === tip[0] && c2 === tip[1]) {
        ok = true;
        break;
      }
      if (!occ[r2 * n + c2]) {
        ok = true;
        break;
      }
    }
    if (cache) {
      cache.noSeal[act] = ok;
    }
    return ok;
  }

  /**
   * After fringe pairs are all locked: weave leftover empty CT 2x2s and small
   * in-sub pockets into the path, then omit-divert into the last food cell.
   * Fills must leave a Cell Tree escape (return plan). No non-CT via scramble.
   */
  function fringeLockHoleAction(env, n, m, safe, h) {
    var emptyLeft = n * n - env.snake.length;
    var tipEarly = env.snake[env.snake.length - 1];
    /*
     * emptyLeft 6..8 + H-behind-T: pocket-fill only (no new hole-enter).
     * 55450/728565 tip-vacated while an in-sub 2x2 pocket was still open;
     * full weave at emptyLeft 8 regressed wins hard (35->18).
     */
    function escP(act) {
      return fringeLockNoSeal(env, act) && fringeCtEscapeAfter(env, m, act);
    }
    function escRim(act) {
      var er = h[0] + DIRS[act][0];
      var ec = h[1] + DIRS[act][1];
      if (er >= m || ec >= m) {
        return fringeLockNoSeal(env, act);
      }
      return escP(act);
    }
    /*
     * emptyLeft 6..8 + HBT: adj pocket-fill only (no ct-to-pocket wander).
     * Wider weave at emptyLeft 8 regressed 35->18.
     */
    var tipBraked = (env._ctOddTipStreak || 0) > 2 * n;
    var foodOutEarly = !!(env.food && (env.food[0] >= m || env.food[1] >= m));
    /*
     * Skip pocket weave while tip-braked with fringe food - prefer door/rim
     * (55450 pocket-wove away from NE F).
     */
    if (emptyLeft > 5 && emptyLeft <= 8 && env._ctOddLockVariant &&
        fringeHeadBehindTip(env) && env.snake.length / (n * n) >= 0.8 &&
        !(tipBraked && foodOutEarly)) {
      var pcs = fringeNonFoodEmptiesInSub(env, m);
      if (pcs.length >= 2 && pcs.length <= 4) {
        var pcomps = fringeCellComponents(pcs);
        var pci;
        var pk = null;
        for (pci = 0; pci < pcomps.length; pci += 1) {
          if (pcomps[pci].length >= 2 && pcomps[pci].length <= 4) {
            pk = pcomps[pci];
            break;
          }
        }
        if (pk) {
          var pai;
          for (pai = 0; pai < safe.length; pai += 1) {
            var pA = safe[pai];
            var pr = h[0] + DIRS[pA][0];
            var pc = h[1] + DIRS[pA][1];
            if (!fringeCellInList(pk, pr, pc)) {
              continue;
            }
            if (pr === tipEarly[0] && pc === tipEarly[1]) {
              continue;
            }
            if (!isCellMove(h[0], h[1], pA) || !escP(pA)) {
              continue;
            }
            env._ctOddPocket = pk;
            env._ctFringeWhy = "pocket-fill";
            return pA;
          }
        }
      }
    }
    /*
     * Tip-orbit brake active: CT toward rim door / adj rim fill (728565).
     * No new hole-enter. Rim landings use seal-only gate.
     */
    if (emptyLeft > 5 && emptyLeft <= 8 && env._ctOddLockVariant && tipBraked) {
      var pcsR = fringeNonFoodEmptiesInSub(env, m);
      var rimR = fringeNonFoodEmptiesAll(env, n);
      var foodOutR = !!(env.food && (env.food[0] >= m || env.food[1] >= m));
      if (pcsR.length === 0 && rimR.length > 0 && foodOutR &&
          !findFringeGrowthSegment(env, m)) {
        var rea;
        for (rea = 0; rea < safe.length; rea += 1) {
          var rE = safe[rea];
          var rEr = h[0] + DIRS[rE][0];
          var rEc = h[1] + DIRS[rE][1];
          if (!fringeCellInList(rimR, rEr, rEc) || !escRim(rE)) {
            continue;
          }
          env._ctFringeWhy = "fringe-fill";
          return rE;
        }
        var rGoal = rimR[0];
        var rej;
        var rBest = Math.abs(h[0] - rGoal[0]) + Math.abs(h[1] - rGoal[1]);
        for (rej = 1; rej < rimR.length; rej += 1) {
          var rM = Math.abs(h[0] - rimR[rej][0]) +
            Math.abs(h[1] - rimR[rej][1]);
          if (rM < rBest) {
            rBest = rM;
            rGoal = rimR[rej];
          }
        }
        var rDoor = fringeCtApproachGoal(
          rGoal[0] >= m ? [m - 1, rGoal[1]] : [rGoal[0], m - 1], m, h);
        var rToward = fringeCtToward(env, n, m, rDoor, null);
        if (rToward !== null && safe.indexOf(rToward) >= 0 && escP(rToward)) {
          env._ctFringeWhy = "ct-to-rim";
          return rToward;
        }
      }
    }
    /* Very late only: earlier hole-enter at emptyLeft=8 trapped 728@113. */
    if (emptyLeft > 5) {
      env._ctOddHoleCell = null;
      env._ctOddPocket = null;
      return null;
    }
    var allLocked = !fringeLockNextTarget(env, m);
    var fill = env.snake.length / (n * n);
    /* Lock-variant: weave without every pair locked once mostly full. */
    if (!allLocked &&
        !(env._ctOddLockVariant && fill >= 0.8 && emptyLeft <= 5)) {
      return null;
    }
    var tip = env.snake[env.snake.length - 1];
    var food = env.food;
    var holes = fringeEmptyCtHoles(env, m);

    function escapeOk(act) {
      return fringeLockNoSeal(env, act) && fringeCtEscapeAfter(env, m, act);
    }
    /* Rim landings are outside the sub - CT-escape-after is always false. */
    function escapeOkRim(act) {
      var nr = h[0] + DIRS[act][0];
      var nc = h[1] + DIRS[act][1];
      if (nr >= m || nc >= m) {
        return fringeLockNoSeal(env, act);
      }
      return escapeOk(act);
    }

    /* Omit divert: win / tip-adj / CT escape (in-sub) / fringe escape (rim). */
    if (food) {
      var fa;
      for (fa = 0; fa < 4; fa += 1) {
        if (h[0] + DIRS[fa][0] === food[0] && h[1] + DIRS[fa][1] === food[1] &&
            safe.indexOf(fa) >= 0) {
          var winEat = env.snake.length + 1 >= n * n;
          var foodInSub = food[0] < m && food[1] < m;
          var omitOk = winEat || fringeTipAdjFood(env);
          if (!omitOk && foodInSub) {
            omitOk = fringeCtEscapeAfter(env, m, fa);
          } else if (!omitOk) {
            omitOk = fringeEscapeAfterFringeEat(env, fa) ||
              fringeLockNoSeal(env, fa);
          }
          if (omitOk) {
            env._ctFringeWhy = "omit-divert";
            return fa;
          }
        }
      }
    }

    /*
     * emptyLeft==2: one non-food empty + food. Adj via only if CT-legal with
     * escape (return plan). Else CT-seek if via is in-sub.
     */
    if (emptyLeft === 2 && food) {
      var vias = fringeNonFoodEmptiesAll(env, n);
      if (vias.length === 1) {
        var via = vias[0];
        var va;
        for (va = 0; va < safe.length; va += 1) {
          var vAct = safe[va];
          var vnr = h[0] + DIRS[vAct][0];
          var vnc = h[1] + DIRS[vAct][1];
          if (vnr !== via[0] || vnc !== via[1]) {
            continue;
          }
          if (!fringeLockNoSeal(env, vAct)) {
            continue;
          }
          if (via[0] < m && via[1] < m) {
            var viaTree = fringeCachedTreeCur(env, m);
            if (!(isCellMove(h[0], h[1], vAct) &&
                canMoveInCellTree(viaTree, h[0], h[1], vnr, vnc, vAct) &&
                fringeCtEscapeAfter(env, m, vAct))) {
              continue;
            }
          }
          env._ctFringeWhy = "via-fill";
          return vAct;
        }
        if (via[0] < m && via[1] < m) {
          var viaGoal = fringeCtApproachGoal(via, m, h);
          var viaToward = fringeCtToward(env, n, m, viaGoal, null);
          if (viaToward !== null && safe.indexOf(viaToward) >= 0 &&
              fringeLockNoSeal(env, viaToward) &&
              fringeCtEscapeAfter(env, m, viaToward)) {
            env._ctFringeWhy = "ct-to-via";
            return viaToward;
          }
        }
      }
    }

    if (env._ctOddHoleCell) {
      var wr0 = env._ctOddHoleCell[0];
      var wc0 = env._ctOddHoleCell[1];
      if (fringeCtCellEmptyCount(env, wr0, wc0) === 0) {
        env._ctOddHoleCell = null;
      }
    }

    /*
     * If tip-wander left us inside a CT cell with 2+ empties (partial hole),
     * arm weave even without a formal hole-enter (132 livelock pose).
     */
    if (!env._ctOddHoleCell && h[0] < m && h[1] < m) {
      var ar0 = h[0] & ~1;
      var ac0 = h[1] & ~1;
      if (fringeCtCellEmptyCount(env, ar0, ac0) >= 2) {
        env._ctOddHoleCell = [ar0, ac0];
      }
    }

    /* Fill empties inside the hole we are weaving (never tip; escape-gated). */
    if (env._ctOddHoleCell && h[0] < m && h[1] < m) {
      var hr0 = env._ctOddHoleCell[0];
      var hc0 = env._ctOddHoleCell[1];
      if ((h[0] & ~1) === hr0 && (h[1] & ~1) === hc0) {
        var aFill;
        for (aFill = 0; aFill < safe.length; aFill += 1) {
          var af = safe[aFill];
          var fr = h[0] + DIRS[af][0];
          var fc = h[1] + DIRS[af][1];
          if ((fr & ~1) !== hr0 || (fc & ~1) !== hc0) {
            continue;
          }
          if (food && food[0] === fr && food[1] === fc) {
            continue;
          }
          if (fr === tip[0] && fc === tip[1]) {
            continue;
          }
          var hit = false;
          var sj;
          for (sj = 0; sj < env.snake.length - 1; sj += 1) {
            if (env.snake[sj][0] === fr && env.snake[sj][1] === fc) {
              hit = true;
              break;
            }
          }
          if (!hit && escapeOk(af)) {
            env._ctFringeWhy = af === cellMoveInside(h[0], h[1])
              ? "hole-tour"
              : "hole-fill";
            return af;
          }
        }
      }
    }

    /* Enter a fully empty CT 2x2 via a CT-legal escape-safe step. */
    /* Lock-exp: do not start a new hole weave above emptyLeft 3 (seals). */
    if (holes.length && !(env._ctOddLockExp && emptyLeft > 3)) {
      var best = null;
      var bestHole = null;
      var bestDist = 1e9;
      var a;
      for (a = 0; a < safe.length; a += 1) {
        var act = safe[a];
        var nr = h[0] + DIRS[act][0];
        var nc = h[1] + DIRS[act][1];
        if (nr === tip[0] && nc === tip[1]) {
          continue;
        }
        if (food && food[0] === nr && food[1] === nc) {
          continue;
        }
        var hole = fringeInEmptyCtHole(holes, nr, nc);
        if (!hole) {
          continue;
        }
        if (!isCellMove(h[0], h[1], act)) {
          continue;
        }
        if (!escapeOk(act)) {
          continue;
        }
        var dist = Math.abs(h[0] - hole[0]) + Math.abs(h[1] - hole[1]);
        if (dist < bestDist) {
          bestDist = dist;
          best = act;
          bestHole = hole;
        }
      }
      if (best !== null) {
        env._ctOddHoleCell = bestHole.slice();
        env._ctFringeWhy = "hole-enter";
        return best;
      }

      /* Seek a CT portal next to the hole so tip-vacate cannot walk away. */
      var hole0 = holes[0];
      var portals = [
        [hole0[0] + 2, hole0[1]],
        [hole0[0] + 2, hole0[1] + 1],
        [hole0[0] - 1, hole0[1]],
        [hole0[0] - 1, hole0[1] + 1],
        [hole0[0], hole0[1] + 2],
        [hole0[0] + 1, hole0[1] + 2],
        [hole0[0], hole0[1] - 1],
        [hole0[0] + 1, hole0[1] - 1]
      ];
      var pi;
      var goal = null;
      var goalMan = 1e9;
      for (pi = 0; pi < portals.length; pi += 1) {
        var pr = portals[pi][0];
        var pc = portals[pi][1];
        if (pr < 0 || pc < 0 || pr >= m || pc >= m) {
          continue;
        }
        var man = Math.abs(h[0] - pr) + Math.abs(h[1] - pc);
        if (man < goalMan) {
          goalMan = man;
          goal = [pr, pc];
        }
      }
      if (goal) {
        var approach = fringeCtApproachGoal(goal, m, h);
        var toward = fringeCtToward(env, n, m, approach, null);
        if (toward !== null && safe.indexOf(toward) >= 0) {
          env._ctFringeWhy = "ct-to-hole";
          return toward;
        }
      }
    }

    /*
     * Non-2x2 in-sub pockets (95045: 1x2 at row 8). Size 2..4 connected
     * non-food empties; CT-legal enter/fill with seal veto.
     */
    var pocketCells = fringeNonFoodEmptiesInSub(env, m);
    if (pocketCells.length >= 2 && pocketCells.length <= 4) {
      var comps = fringeCellComponents(pocketCells);
      var ci;
      var pocket = null;
      for (ci = 0; ci < comps.length; ci += 1) {
        if (comps[ci].length >= 2 && comps[ci].length <= 4) {
          pocket = comps[ci];
          break;
        }
      }
      if (pocket) {
        env._ctOddPocket = pocket;
        var pa;
        for (pa = 0; pa < safe.length; pa += 1) {
          var pAct = safe[pa];
          var pnr = h[0] + DIRS[pAct][0];
          var pnc = h[1] + DIRS[pAct][1];
          if (!fringeCellInList(pocket, pnr, pnc)) {
            continue;
          }
          if (pnr === tip[0] && pnc === tip[1]) {
            continue;
          }
          if (!isCellMove(h[0], h[1], pAct)) {
            continue;
          }
          if (!escapeOk(pAct)) {
            continue;
          }
          env._ctFringeWhy = "pocket-fill";
          return pAct;
        }
        var pGoal = pocket[0];
        var pBestMan = Math.abs(h[0] - pGoal[0]) + Math.abs(h[1] - pGoal[1]);
        var pj;
        for (pj = 1; pj < pocket.length; pj += 1) {
          var pm = Math.abs(h[0] - pocket[pj][0]) +
            Math.abs(h[1] - pocket[pj][1]);
          if (pm < pBestMan) {
            pBestMan = pm;
            pGoal = pocket[pj];
          }
        }
        var pApproach = fringeCtApproachGoal(pGoal, m, h);
        var pToward = fringeCtToward(env, n, m, pApproach, null);
        if (pToward !== null && safe.indexOf(pToward) >= 0 &&
            escapeOk(pToward)) {
          env._ctFringeWhy = "ct-to-pocket";
          return pToward;
        }
      }
    } else {
      env._ctOddPocket = null;
    }

    /*
     * Fringe residual (87126): no in-sub non-food empties left, only rim
     * cells + omit food. Narrower than "all empties on rim" so mid-endgame
     * fringe-tip cannot yank a working CT clear (regressed 132).
     * If a growth pair is still free, yield so needGrow can claim it
     * (1597639765 corner tip-loop past open bottom:2 doors).
     */
    var inSubLeft = fringeNonFoodEmptiesInSub(env, m);
    var rimEmpties = fringeNonFoodEmptiesAll(env, n);
    var foodOutLate = !!(food && (food[0] >= m || food[1] >= m));
    if (inSubLeft.length === 0 && rimEmpties.length > 0 && foodOutLate &&
        emptyLeft <= 4) {
      if (findFringeGrowthSegment(env, m)) {
        return null;
      }
      /*
       * Fringe-end F (NE/SW): tip-vacate before fringe-fill so tip leads the
       * rim climb (zigzag leaves an escape cell). Rim-fill then tip-chase
       * reaches NE with no escape (783998).
       */
      var fringeEndFood = food &&
        ((food[0] === 0 && food[1] === m) ||
          (food[0] === m && food[1] === 0));
      if (fringeEndFood) {
        var tipBeforeFill = fringeTipVacateFresh(env, safe);
        if (tipBeforeFill !== null) {
          fringeTipWhy(env, "tip-before-fill");
          return tipBeforeFill;
        }
      }
      var ra;
      for (ra = 0; ra < safe.length; ra += 1) {
        var rAct = safe[ra];
        var rnr = h[0] + DIRS[rAct][0];
        var rnc = h[1] + DIRS[rAct][1];
        if (!fringeCellInList(rimEmpties, rnr, rnc)) {
          continue;
        }
        if (!escapeOkRim(rAct)) {
          continue;
        }
        env._ctFringeWhy = "fringe-fill";
        return rAct;
      }
      /* CT toward door of nearest rim empty (316777 fragmented rim). */
      var rimGoal = rimEmpties[0];
      var rj;
      var rimBest = Math.abs(h[0] - rimGoal[0]) + Math.abs(h[1] - rimGoal[1]);
      for (rj = 1; rj < rimEmpties.length; rj += 1) {
        var rm = Math.abs(h[0] - rimEmpties[rj][0]) +
          Math.abs(h[1] - rimEmpties[rj][1]);
        if (rm < rimBest) {
          rimBest = rm;
          rimGoal = rimEmpties[rj];
        }
      }
      var rimDoor = fringeCtApproachGoal(
        rimGoal[0] >= m ? [m - 1, rimGoal[1]] : [rimGoal[0], m - 1], m, h);
      var rimToward = fringeCtToward(env, n, m, rimDoor, null);
      if (rimToward !== null && safe.indexOf(rimToward) >= 0 &&
          escapeOk(rimToward)) {
        env._ctFringeWhy = "ct-to-rim";
        return rimToward;
      }
      var tipRim = fringeTipVacateFresh(env, safe);
      if (tipRim !== null) {
        fringeTipWhy(env, "fringe-tip");
        return tipRim;
      }
      var bestRim = null;
      var bestRimMan = 1e9;
      for (ra = 0; ra < safe.length; ra += 1) {
        var sAct = safe[ra];
        if (!escapeOk(sAct)) {
          continue;
        }
        var snr = h[0] + DIRS[sAct][0];
        var snc = h[1] + DIRS[sAct][1];
        var sMan = Math.abs(snr - food[0]) + Math.abs(snc - food[1]);
        if (sMan < bestRimMan) {
          bestRimMan = sMan;
          bestRim = sAct;
        }
      }
      if (bestRim !== null) {
        env._ctFringeWhy = "fringe-seek";
        return bestRim;
      }
    }

    /*
     * Tip-to-G / tip-to-via: splice not ready, or emptyLeft==2 via not
     * enterable yet - tip-vacate instead of ct-wait.
     */
    if (emptyLeft <= 4) {
      var foodSeg = findFringeCtSegment(env, m);
      if ((foodSeg && !foodSeg.ready) || emptyLeft === 2) {
        var tipG = fringeTipVacateFresh(env, safe);
        if (tipG !== null) {
          fringeTipWhy(env, foodSeg && !foodSeg.ready ? "tip-to-g" : "tip-to-via");
          return tipG;
        }
      }
    }
    return null;
  }

  /**
   * After an on-fringe eat (tip stays put), can the next non-eat leave?
   * Empty neighbors count; tip also counts (next step vacates onto tip).
   */
  function fringeEscapeAfterFringeEat(env, act) {
    var h = env.snake[0];
    var tip = env.snake[env.snake.length - 1];
    var tipKey = tip[0] + "," + tip[1];
    var nr = h[0] + DIRS[act][0];
    var nc = h[1] + DIRS[act][1];
    var n = env.size;
    var body = {};
    var i;
    for (i = 0; i < env.snake.length; i += 1) {
      body[env.snake[i][0] + "," + env.snake[i][1]] = true;
    }
    body[nr + "," + nc] = true;
    var a;
    for (a = 0; a < 4; a += 1) {
      var r2 = nr + DIRS[a][0];
      var c2 = nc + DIRS[a][1];
      if (r2 < 0 || c2 < 0 || r2 >= n || c2 >= n) {
        continue;
      }
      var k = r2 + "," + c2;
      if (k === tipKey || !body[k]) {
        return true;
      }
    }
    return false;
  }

  /**
   * Adjacent fringe food eat: SE/near-SE, fringe-strip ends (NE/SW), or tip
   * already adj to F. Fringe-end / near-corner eats must leave an exit
   * (empty or tip); tip-adj mid-edge may eat without that gate.
   */
  function fringeAdjFoodAction(env, safe) {
    var food = env.food;
    if (!food) {
      return null;
    }
    var m = env.size - 1;
    var emptyLeft = env.size * env.size - env.snake.length;
    var nearCorner = (food[0] === m && food[1] === m) ||
      (food[0] === m && food[1] === m - 1) ||
      (food[0] === m - 1 && food[1] === m);
    var fringeEnd = (food[0] === 0 && food[1] === m) ||
      (food[0] === m && food[1] === 0);
    var tipAdj = fringeTipAdjFood(env);
    /* Mid-rim: tip-adj any time; else late (emptyLeft<=5) with escape. */
    if (!nearCorner && !fringeEnd && !tipAdj && emptyLeft > 5) {
      return null;
    }
    /* Need escape after eat when near-corner / fringe-end / tip not adj. */
    var needExit = nearCorner || fringeEnd || !tipAdj;
    var winEat = env.snake.length + 1 >= env.size * env.size;
    var h = env.snake[0];
    var a;
    for (a = 0; a < 4; a += 1) {
      if (h[0] + DIRS[a][0] === food[0] && h[1] + DIRS[a][1] === food[1] &&
          safe.indexOf(a) >= 0) {
        if (needExit && !winEat && !fringeEscapeAfterFringeEat(env, a)) {
          continue;
        }
        return a;
      }
    }
    return null;
  }

  /**
   * Disabled: stepping onto tip from the fringe while tip sits on G steals the
   * tip-on-G ready window for the CT doorA splice (integrity livelock on NE/SW
   * ends). Tip-on-G is handled by findFringeCtSegment ready + enter-bump.
   */
  function fringeChaseTipFood(/* env, m, safe */) {
    return null;
  }

  function fringeStepLandsOn(env, act, cell) {
    if (!cell) {
      return false;
    }
    var h = env.snake[0];
    var nr = h[0] + DIRS[act][0];
    var nc = h[1] + DIRS[act][1];
    if (Array.isArray(cell[0])) {
      var i;
      for (i = 0; i < cell.length; i += 1) {
        if (cell[i] && nr === cell[i][0] && nc === cell[i][1]) {
          return true;
        }
      }
      return false;
    }
    return nr === cell[0] && nc === cell[1];
  }

  /**
   * In-sub corner left empty when splicing through fringe food in the edge 2x2
   * (doorA/doorB on the sub rim). Soft-avoid after so the return pass does not
   * fill that garage cell.
   */
  function fringeSegOpenCell(seg, m) {
    if (!seg || !seg.doorA || !seg.doorB) {
      return null;
    }
    if (seg.kind === "corner" || seg.kind === "corner-long" ||
        seg.kind === "near-corner-bottom") {
      return m >= 2 ? [m - 2, m - 2] : null;
    }
    if (seg.kind === "near-corner-right") {
      return null;
    }
    var a = seg.doorA;
    var b = seg.doorB;
    if (a[0] === m - 1 && b[0] === m - 1 && m >= 2) {
      /* Bottom rim: leave open the cell above doorB (exit side). */
      return [m - 2, b[1]];
    }
    if (a[1] === m - 1 && b[1] === m - 1 && m >= 2) {
      /* Right rim: leave open the cell west of doorB. */
      return [b[0], m - 2];
    }
    return null;
  }

  function fringeAvoidList(env, extra) {
    var list = [];
    function add(cell) {
      if (!cell) {
        return;
      }
      var i;
      for (i = 0; i < list.length; i += 1) {
        if (list[i][0] === cell[0] && list[i][1] === cell[1]) {
          return;
        }
      }
      list.push(cell);
    }
    add(env._ctOddOpenCell);
    if (extra) {
      if (Array.isArray(extra[0])) {
        var j;
        for (j = 0; j < extra.length; j += 1) {
          add(extra[j]);
        }
      } else {
        add(extra);
      }
    }
    return list.length ? list : null;
  }

  /**
   * Odd-board Cell Tree experiments.
   * style:
   *  ham - always Ham (baseline)
   *  sub - top-left (n-1)^2; sticky Ham lock on leave
   *  subSoft - re-enter CT when snake+food back in subspace (unsafe)
   *  subFood - Ham while food on fringe; sticky lock only if body leaves
   *  subBump - CT to portal + scripted fringe out/eat/back (test variant)
   *  subBR - bottom-right (n-1)^2 (origin at (1,1)); sticky lock
   *  sub20 / sub50 - subspace then hard handoff at fill% of full board
   *  subMax - push subspace until m*m-1 then Ham
   *  subPick - pick TL/BR by food (unsafe vs sticky TL)
   */
  function cellTreeOddVariantNextAction(env, style) {
    var n = env.size;
    if (!(n & 1) || n < 3 || !env.food) {
      return E.masterHamiltonAction(env);
    }
    if (style === "ham") {
      return E.masterHamiltonAction(env);
    }
    if (style === "subBump") {
      return cellTreeOddFringeBumpNextAction(env);
    }
    /* Episode reset: clear sticky Ham lock. */
    if (typeof env._ctOddLastSteps === "number" && env.steps < env._ctOddLastSteps) {
      env._ctOddHamLock = false;
    }
    env._ctOddLastSteps = env.steps;
    var soft = style === "subSoft";
    var foodSoft = style === "subFood";
    var br = style === "subBR";
    var pick = style === "subPick";
    var m = n - 1;
    var sticky = !soft && !foodSoft;
    if ((sticky || foodSoft) && env._ctOddHamLock) {
      return E.masterHamiltonAction(env);
    }
    var fill = env.snake.length / (n * n);
    if (style === "sub20" && fill >= 0.2) {
      env._ctOddHamLock = true;
      return E.masterHamiltonAction(env);
    }
    if (style === "sub50" && fill >= 0.5) {
      env._ctOddHamLock = true;
      return E.masterHamiltonAction(env);
    }
    var maxSub = style === "subMax" ? (m * m - 1) : (m * m - 2);
    if (env.snake.length >= maxSub) {
      if (sticky) {
        env._ctOddHamLock = true;
      }
      return E.masterHamiltonAction(env);
    }

    if (pick) {
      var tlOk = !snakeOutsideSub(env, m) &&
        env.food[0] < m && env.food[1] < m;
      var brOk = !snakeOutsideSubBR(env, n) &&
        env.food[0] >= 1 && env.food[1] >= 1;
      var useBr = false;
      if (tlOk && brOk) {
        /* Prefer the subspace that keeps more board margin from food. */
        var dTl = Math.min(m - 1 - env.food[0], m - 1 - env.food[1]);
        var dBr = Math.min(env.food[0] - 1, env.food[1] - 1);
        useBr = dBr > dTl;
      } else if (brOk && !tlOk) {
        useBr = true;
      } else if (!tlOk && !brOk) {
        if (sticky) {
          env._ctOddHamLock = true;
        }
        return E.masterHamiltonAction(env);
      }
      var actP = useBr ? cellTreeEvenCoreBR(env, n, m) :
        cellTreeEvenCore(env, n, { subM: m });
      if (actP === null) {
        if (sticky) {
          env._ctOddHamLock = true;
        }
        return E.masterHamiltonAction(env);
      }
      return actP;
    }

    var foodOut;
    var bodyOut;
    if (br) {
      foodOut = env.food[0] < 1 || env.food[1] < 1;
      bodyOut = snakeOutsideSubBR(env, n);
    } else {
      foodOut = env.food[0] >= m || env.food[1] >= m;
      bodyOut = snakeOutsideSub(env, m);
    }
    if (bodyOut) {
      env._ctOddHamLock = true;
      return E.masterHamiltonAction(env);
    }
    if (foodOut) {
      if (sticky) {
        env._ctOddHamLock = true;
      }
      /* foodSoft: borrow Ham for fringe food; resume CT when food returns. */
      return E.masterHamiltonAction(env);
    }
    var act;
    if (br) {
      act = cellTreeEvenCoreBR(env, n, m);
    } else {
      act = cellTreeEvenCore(env, n, { subM: m });
    }
    if (act === null) {
      env._ctOddHamLock = true;
      return E.masterHamiltonAction(env);
    }
    return act;
  }

  function snakeOutsideSubBR(env, n) {
    var i;
    for (i = 0; i < env.snake.length; i += 1) {
      if (env.snake[i][0] < 1 || env.snake[i][1] < 1) {
        return true;
      }
    }
    return false;
  }

  /** Bottom-right (n-1)^2 subspace: map coords by -1, reuse even core logic. */
  function cellTreeEvenCoreBR(env, n, m) {
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var h = env.snake[0];
    var fr = env.food[0];
    var fc = env.food[1];
    if (h[0] < 1 || h[1] < 1 || fr < 1 || fc < 1) {
      return null;
    }
    var occ = buildBodyOcc(env);
    var r;
    var c;
    for (r = 0; r < n; r += 1) {
      for (c = 0; c < n; c += 1) {
        if (r < 1 || c < 1) {
          occ[r * n + c] = 1;
        }
      }
    }
    var mappedSnake = [];
    var i;
    for (i = 0; i < env.snake.length; i += 1) {
      mappedSnake.push([env.snake[i][0] - 1, env.snake[i][1] - 1]);
    }
    var tree = cellTreeParents(m, mappedSnake);
    var edgeFn = function (r0, c0, nr, nc, dir) {
      if (nr < 1 || nc < 1 || nr >= n || nc >= n) {
        return INF;
      }
      if (occ[nr * n + nc]) {
        return INF;
      }
      var ar0 = r0 - 1;
      var ac0 = c0 - 1;
      var anr = nr - 1;
      var anc = nc - 1;
      if (!canMoveInCellTree(tree, ar0, ac0, anr, anc, dir)) {
        return INF;
      }
      var ca = cellIdx(ar0, ac0, tree.cw);
      var cb = cellIdx(anr, anc, tree.cw);
      var toParent = tree.parents[ca] === cb;
      var toSame = ca === cb;
      var pen = toParent ? 2 : (toSame ? 0 : 1);
      return 1000 + pen;
    };
    var sp = dijkstraTo(n, h[0], h[1], fr, fc, edgeFn);
    var path = sp.found ? readPath(n, sp.prev, h[0], h[1], fr, fc) : [];
    var act = firstStepAction(path, h[0], h[1]);
    if (act !== null && safe.indexOf(act) >= 0) {
      return act;
    }
    var b;
    for (b = 0; b < safe.length; b += 1) {
      var aa = safe[b];
      var rr = h[0] + DIRS[aa][0];
      var cc = h[1] + DIRS[aa][1];
      if (rr < 1 || cc < 1 || rr >= n || cc >= n) {
        continue;
      }
      if (isCellMove(h[0] - 1, h[1] - 1, aa) &&
          canMoveInCellTree(tree, h[0] - 1, h[1] - 1, rr - 1, cc - 1, aa)) {
        return aa;
      }
    }
    return null;
  }

  function fringeStepStaysInSub(env, m, act) {
    var h = env.snake[0];
    var nr = h[0] + DIRS[act][0];
    var nc = h[1] + DIRS[act][1];
    return inTopLeftSub(nr, nc, m);
  }

  /**
   * Head stranded on fringe with no bump script: step back into the sub only.
   * Never walk along the fringe (that paints body outside the CT subspace).
   */
  function fringeStepIntoSub(env, m, safe) {
    var h = env.snake[0];
    var tip = env.snake[env.snake.length - 1];
    var tipKey = tip[0] + "," + tip[1];
    var body = {};
    var i;
    for (i = 0; i < env.snake.length - 1; i += 1) {
      body[env.snake[i][0] + "," + env.snake[i][1]] = true;
    }
    /*
     * Prefer a safe re-entry that does not eat. Eating from headOut seals when
     * tip would have been the only escape (tip does not vacate on grow).
     */
    for (i = 0; i < safe.length; i += 1) {
      var a = safe[i];
      var nr = h[0] + DIRS[a][0];
      var nc = h[1] + DIRS[a][1];
      if (!inTopLeftSub(nr, nc, m)) {
        continue;
      }
      if (env.food && env.food[0] === nr && env.food[1] === nc) {
        continue;
      }
      return a;
    }
    /*
     * No safe into-sub yet (neck still sealing): walk fringe toward a cell
     * that borders a free in-sub door (or toward tip). Never step onto food
     * unless tip-adj / post-eat escape (sealed mid-rim eat trapped 1597639765).
     */
    var best = null;
    var bestScore = -1e15;
    for (i = 0; i < safe.length; i += 1) {
      var act = safe[i];
      var r = h[0] + DIRS[act][0];
      var c = h[1] + DIRS[act][1];
      if (env.food && env.food[0] === r && env.food[1] === c) {
        if (!fringeTipAdjFood(env) && !fringeEscapeAfterFringeEat(env, act)) {
          continue;
        }
      }
      var score = -(Math.abs(r - tip[0]) + Math.abs(c - tip[1]));
      var d;
      for (d = 0; d < 4; d += 1) {
        var r2 = r + DIRS[d][0];
        var c2 = c + DIRS[d][1];
        if (!inTopLeftSub(r2, c2, m)) {
          continue;
        }
        var k2 = r2 + "," + c2;
        if (!body[k2] || k2 === tipKey) {
          score += 100;
        }
      }
      if (score > bestScore) {
        bestScore = score;
        best = act;
      }
    }
    if (best !== null) {
      return best;
    }
    /* Tip-vacate last: tip may be the only non-sealed safe neighbor. */
    var tipAct = fringeTipVacate(env, safe);
    if (tipAct !== null) {
      return tipAct;
    }
    /*
     * emptyLeft<=2 sole safe onto rim F: take rather than throw (269263).
     * Wider sole-safe (<=8) regressed wins hard - keep penultimate only.
     */
    var emptyLeftSO = env.size * env.size - env.snake.length;
    if (safe.length === 1 && env.food && emptyLeftSO <= 2) {
      var onlySO = safe[0];
      var sor = h[0] + DIRS[onlySO][0];
      var soc = h[1] + DIRS[onlySO][1];
      if (env.food[0] === sor && env.food[1] === soc) {
        return onlySO;
      }
    }
    fringeCtBroken(env, "stranded on fringe with no safe step");
  }

  /**
   * After a candidate in-sub step, is there still a CT-legal in-sub escape?
   * Rejects walking into boundary cul-de-sacs (only fringe / non-cell exits).
   */
  function fringeCtEscapeAfter(env, m, act) {
    var cache = fringeDecCacheGet(env, m);
    if (cache && cache.esc[act] !== undefined) {
      return cache.esc[act];
    }
    var h = env.snake[0];
    var tip = env.snake[env.snake.length - 1];
    var nr = h[0] + DIRS[act][0];
    var nc = h[1] + DIRS[act][1];
    if (!inTopLeftSub(nr, nc, m)) {
      if (cache) {
        cache.esc[act] = false;
      }
      return false;
    }
    var eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
    /* Eat with tip already adj to F: next non-eat steps onto tip (H-F-T). */
    if (eating && Math.abs(tip[0] - nr) + Math.abs(tip[1] - nc) === 1) {
      if (cache) {
        cache.esc[act] = true;
      }
      return true;
    }
    var i;
    /* Non-eat drops tip; eat keeps full body (tip does not vacate). */
    var lim = eating ? env.snake.length : env.snake.length - 1;
    var tree = cellTreeParentsSubAfterStep(m, env.snake, nr, nc, eating);
    /*
     * Body after move (tip free for neighbor checks): mark new head + kept
     * segments, omit new tip. Matches prior newSnake bodyLim marking.
     */
    var occ = new Uint8Array(m * m);
    occ[nr * m + nc] = 1;
    var markLim = eating ? lim : lim - 1;
    for (i = 0; i < markLim; i += 1) {
      var br = env.snake[i][0];
      var bc = env.snake[i][1];
      if (br < m && bc < m) {
        occ[br * m + bc] = 1;
      }
    }
    var a;
    var ok = false;
    for (a = 0; a < 4; a += 1) {
      var r2 = nr + DIRS[a][0];
      var c2 = nc + DIRS[a][1];
      if (!inTopLeftSub(r2, c2, m)) {
        continue;
      }
      if (occ[r2 * m + c2] && !(eating && r2 === tip[0] && c2 === tip[1])) {
        continue;
      }
      if (canMoveInCellTree(tree, nr, nc, r2, c2, a)) {
        ok = true;
        break;
      }
    }
    if (cache) {
      cache.esc[act] = ok;
    }
    return ok;
  }

  /**
   * Approach goal with CT-compatible steps only: close Manhattan among
   * tree-legal steps (dominant-axis tie-break), else cellTreeEvenCore toward
   * goal. Never flood-recover / raw non-CT Manhattan.
   */
  function fringeCtToward(env, n, m, goal, avoid) {
    var safe = env.safeActions();
    var h = env.snake[0];
    var tree = fringeCachedTreeCur(env, m);
    function okStep(act) {
      if (act === null || safe.indexOf(act) < 0) {
        return false;
      }
      if (!fringeStepStaysInSub(env, m, act)) {
        return false;
      }
      if (fringeStepLandsOn(env, act, avoid)) {
        return false;
      }
      var nr = h[0] + DIRS[act][0];
      var nc = h[1] + DIRS[act][1];
      if (!(isCellMove(h[0], h[1], act) &&
          canMoveInCellTree(tree, h[0], h[1], nr, nc, act))) {
        return false;
      }
      return fringeCtEscapeAfter(env, m, act);
    }
    var block = avoid || null;
    var hit = null;
    function tryGoal(g) {
      if (!g || !inTopLeftSub(g[0], g[1], m)) {
        return null;
      }
      var opts = { subM: m, goal: g };
      if (block) {
        opts.block = block;
      }
      var act = cellTreeEvenCore(env, n, opts);
      return okStep(act) ? act : null;
    }
    function manTo(r, c) {
      return Math.abs(r - goal[0]) + Math.abs(c - goal[1]);
    }
    function axisScore(r, c) {
      /* Prefer closing the larger of |dr|,|dc| so we do not slide into a
       * CT dead-end on the door's column while still far in row (or vice versa). */
      var dr = Math.abs(r - goal[0]);
      var dc = Math.abs(c - goal[1]);
      return (dr > dc ? dr * 1000 + dc : dc * 1000 + dr);
    }
    function pickCloser(maxMan) {
      var bestClose = null;
      var bestMan = maxMan + 1;
      var bestAxis = 1e15;
      var ci;
      for (ci = 0; ci < safe.length; ci += 1) {
        var ca = safe[ci];
        if (!okStep(ca)) {
          continue;
        }
        var cr = h[0] + DIRS[ca][0];
        var cc = h[1] + DIRS[ca][1];
        var man = manTo(cr, cc);
        if (man > maxMan) {
          continue;
        }
        var ax = axisScore(cr, cc);
        if (man < bestMan || (man === bestMan && ax < bestAxis)) {
          bestMan = man;
          bestAxis = ax;
          bestClose = ca;
        }
      }
      return bestClose;
    }
    if (goal) {
      var curMan = manTo(h[0], h[1]);
      /* Among CT steps that close man, dominant-axis tie-break (before core). */
      var closeStrict = pickCloser(curMan - 1);
      if (closeStrict !== null) {
        return closeStrict;
      }
      var closeEq = pickCloser(curMan);
      if (closeEq !== null) {
        return closeEq;
      }
      hit = tryGoal(goal);
      if (hit !== null) {
        return hit;
      }
      var deltas = [[-1, 0], [1, 0], [0, -1], [0, 1]];
      var d;
      for (d = 0; d < deltas.length; d += 1) {
        var alt = [goal[0] + deltas[d][0], goal[1] + deltas[d][1]];
        if (avoid && alt[0] === avoid[0] && alt[1] === avoid[1]) {
          continue;
        }
        hit = tryGoal(alt);
        if (hit !== null) {
          return hit;
        }
      }
    }
    var orbits = [[0, 0], [m - 1, 0], [0, m - 1], [m - 1, m - 1]];
    var oi;
    for (oi = 0; oi < orbits.length; oi += 1) {
      hit = tryGoal(orbits[oi]);
      if (hit !== null) {
        return hit;
      }
    }
    /* Any tree-legal in-sub step (keep circulating). */
    var si;
    for (si = 0; si < safe.length; si += 1) {
      if (okStep(safe[si])) {
        return safe[si];
      }
    }
    return null;
  }

  /**
   * Cell Tree Odd: in-sub always CT-compatible. Fringe food rewrites a
   * CT-consecutive doorA->doorB segment with an F+G (or longer corner) splice.
   * No Hamilton handoff - plays CT (+ fringe splices) until death.
   */
  function cellTreeOddFringeBumpNextAction(env) {
    var n = env.size;
    var m = n - 1;
    fringeDecCacheBegin(env, m);
    if (typeof env._ctOddLastSteps === "number" && env.steps < env._ctOddLastSteps) {
      env._ctBumpActions = null;
      env._ctOddOpenCell = null;
      env._ctOddOpenFoodKey = null;
      env._ctOddLocks = null;
      env._ctOddLockPending = null;
      env._ctOddTipStreak = 0;
      env._ctOddTipLen = null;
    }
    env._ctOddLastSteps = env.steps;
    if (env._ctOddTipLen !== env.snake.length) {
      env._ctOddTipLen = env.snake.length;
      env._ctOddTipStreak = 0;
    }

    var h = env.snake[0];
    var tip = env.snake[env.snake.length - 1];
    var safe = env.safeActions();
    if (!safe.length) {
      fringeCtBroken(env, "no safe actions (trapped)");
    }

    var foodOut = env.food[0] >= m || env.food[1] >= m;
    var headOut = h[0] >= m || h[1] >= m;
    var foodKey = env.food[0] + "," + env.food[1];
    if (env._ctOddOpenFoodKey && env._ctOddOpenFoodKey !== foodKey) {
      env._ctOddOpenCell = null;
      env._ctOddOpenFoodKey = null;
    }
    /* Tip reached the garage: stop avoiding so it can vacate, then clear. */
    if (env._ctOddOpenCell &&
        tip[0] === env._ctOddOpenCell[0] && tip[1] === env._ctOddOpenCell[1]) {
      env._ctOddOpenCell = null;
      env._ctOddOpenFoodKey = null;
    }

    /* Finish splice script (only time head may be on fringe by design). */
    if (env._ctBumpActions && env._ctBumpActions.length) {
      /*
       * Prefer NE/SW fringe-end or near-SE eat over a bump that walks past F
       * (55450). Mid-rim interrupt regressed other wins - keep narrow.
       */
      if (n * n - env.snake.length <= 8 && env.food) {
        var fB = env.food;
        var endB = (fB[0] === 0 && fB[1] === m) || (fB[0] === m && fB[1] === 0);
        var cornB = (fB[0] === m && fB[1] === m) ||
          (fB[0] === m && fB[1] === m - 1) ||
          (fB[0] === m - 1 && fB[1] === m);
        if (endB || cornB) {
          var eatBump = fringeAdjFoodAction(env, safe);
          if (eatBump !== null) {
            env._ctBumpActions = null;
            env._ctOddLockPending = null;
            env._ctFringeWhy = "fringe-eat";
            return eatBump;
          }
        }
      }
      var next = env._ctBumpActions[0];
      if (safe.indexOf(next) >= 0) {
        env._ctBumpActions = env._ctBumpActions.slice(1);
        if (!env._ctBumpActions.length) {
          env._ctBumpActions = null;
          /* Lock variant: committing a pair splice locks it into the path. */
          if (env._ctOddLockVariant && env._ctOddLockPending) {
            fringeLockAdd(env, env._ctOddLockPending.edge,
              env._ctOddLockPending.depth);
            env._ctOddLockPending = null;
            env._ctFringeWhy = "lock-commit";
            return next;
          }
        }
        env._ctFringeWhy = "bump";
        return next;
      }
      env._ctBumpActions = null;
      env._ctOddLockPending = null;
    }

    /*
     * On fringe: prefer eating adjacent food (esp. SE corner F) over walking
     * the 3 rim cells and skipping it; else step back into the sub.
     */
    if (headOut) {
      var emptyHeadOut = n * n - env.snake.length;
      /*
       * Before a late corner/mid-rim eat: if tip is adj, vacate first so tip
       * can cover the next rim F (728565 SE-eat then mid-rim seal).
       * Also at emptyLeft<=3 when F is adj but mid-rim gate refused (316777).
       * Raw tip on rim (return plan).
       */
      var tipPre = fringeTipVacate(env, safe);
      if (tipPre !== null) {
        var eatPre = fringeAdjFoodAction(env, safe);
        var fPre = env.food;
        var foodAdjPre = false;
        var fi;
        if (fPre) {
          for (fi = 0; fi < 4; fi += 1) {
            if (h[0] + DIRS[fi][0] === fPre[0] &&
                h[1] + DIRS[fi][1] === fPre[1] &&
                safe.indexOf(fi) >= 0) {
              foodAdjPre = true;
              break;
            }
          }
        }
        var cornerPre = fPre &&
          ((fPre[0] === m && fPre[1] === m) ||
            (fPre[0] === m && fPre[1] === m - 1) ||
            (fPre[0] === m - 1 && fPre[1] === m));
        var fringeEndPre = fPre &&
          ((fPre[0] === 0 && fPre[1] === m) ||
            (fPre[0] === m && fPre[1] === 0));
        var tipAdjFPre = fringeTipAdjFood(env);
        var tipBefore =
          (emptyHeadOut >= 4 && emptyHeadOut <= 8 && eatPre !== null &&
            (cornerPre || !tipAdjFPre)) ||
          (emptyHeadOut <= 3 && foodAdjPre && eatPre === null) ||
          /* Mid-rim eat with tip adj to head but not to F: tip first so
           * post-eat is not a sealed rim cul-de-sac (269263). */
          (emptyHeadOut <= 3 && foodAdjPre && eatPre !== null &&
            !tipAdjFPre && !cornerPre && !fringeEndPre);
        if (tipBefore) {
          env._ctOddTipStreak = 0;
          env._ctFringeWhy = "tip-before-eat";
          return tipPre;
        }
      }
      var eatOut = fringeAdjFoodAction(env, safe);
      if (eatOut !== null) {
        env._ctFringeWhy = "fringe-eat";
        return eatOut;
      }
      var chaseOut = fringeChaseTipFood(env, m, safe);
      if (chaseOut !== null) {
        env._ctFringeWhy = "fringe-chase";
        return chaseOut;
      }
      /*
       * Late fringe: tip-vacate before reentry / sealed-food refuse.
       * emptyLeft<=5 starts earlier so tip can reach mid-rim F (1597639765).
       * Fresh: tip-orbit brake still applies (clears only on growth).
       */
      if (emptyHeadOut <= 5) {
        var tipOut = fringeTipVacateFresh(env, safe);
        if (tipOut !== null) {
          fringeTipWhy(env, "tip-vacate");
          return tipOut;
        }
      }
      env._ctFringeWhy = "headOut-insub";
      return fringeStepIntoSub(env, m, safe);
    }

    /*
     * In-sub next to fringe food: eat if allowed; else at emptyLeft<=3 with
     * tip+F both adj (mid-rim refused), tip first before hole fringe-tip
     * livelock (316777). Do not tip-before when eat is allowed (len-120 stall).
     */
    if (foodOut) {
      var emptyFoodOut = n * n - env.snake.length;
      var eatIn = fringeAdjFoodAction(env, safe);
      if (eatIn !== null) {
        env._ctFringeWhy = "fringe-eat";
        return eatIn;
      }
      if (emptyFoodOut <= 3) {
        var tipInPre = fringeTipVacate(env, safe);
        if (tipInPre !== null) {
          var fAdjIn = false;
          var fii;
          for (fii = 0; fii < 4; fii += 1) {
            if (h[0] + DIRS[fii][0] === env.food[0] &&
                h[1] + DIRS[fii][1] === env.food[1] &&
                safe.indexOf(fii) >= 0) {
              fAdjIn = true;
              break;
            }
          }
          if (fAdjIn) {
            env._ctOddTipStreak = 0;
            env._ctFringeWhy = "tip-before-eat";
            return tipInPre;
          }
        }
      }
    }

    /* Lock variant: keep locked pairs on the fringe path; acquire next. */
    if (env._ctOddLockVariant) {
      var lockAct = fringeLockPairAction(env, n, m, safe, h);
      if (lockAct !== null) {
        return lockAct;
      }
      /* Weave empty CT 2x2 holes into the path; omit-divert last food. */
      var holeAct = fringeLockHoleAction(env, n, m, safe, h);
      if (holeAct !== null) {
        return holeAct;
      }
    }

    /*
     * Last empty is in-sub food: eat if adj (always wins at n*n-1); else
     * tip-follow until adj. Wider pocket endgames stay on CT + growth.
     */
    var emptyLeftEarly = n * n - env.snake.length;
    if (!foodOut && emptyLeftEarly <= 1) {
      var eatLast = null;
      var la;
      for (la = 0; la < 4; la += 1) {
        if (h[0] + DIRS[la][0] === env.food[0] &&
            h[1] + DIRS[la][1] === env.food[1] &&
            safe.indexOf(la) >= 0) {
          eatLast = la;
          break;
        }
      }
      if (eatLast !== null) {
        env._ctFringeWhy = "fringe-eat";
        return eatLast;
      }
      var tipLast = fringeTipVacateFresh(env, safe);
      if (tipLast !== null) {
        fringeTipWhy(env, "tip-vacate");
        return tipLast;
      }
      if (safe.length === 1) {
        env._ctFringeWhy = "endgame-step";
        return safe[0];
      }
    }

    /*
     * Subspace nearly full: park length on next free fringe pair by depth
     * (SW/NE then toward SE). Allowed with fringe food when that splice is
     * not tip-ready - keeps claiming the next pair instead of tip-livelock.
     */
    var needGrow = fringeNeedGrowth(env, m);
    if (needGrow) {
      var growSeg = findFringeGrowthSegment(env, m);
      if (growSeg) {
        var growAvoid = fringeAvoidList(env,
          growSeg.ready ? null : growSeg.doorB);
        var atGrowA = h[0] === growSeg.doorA[0] && h[1] === growSeg.doorA[1];
        if (growSeg.ready && atGrowA) {
          var growFirst = growSeg.splice[0];
          if (safe.indexOf(growFirst) >= 0) {
            env._ctBumpActions = growSeg.splice.slice(1);
            if (!env._ctBumpActions.length) {
              env._ctBumpActions = null;
            }
            /* Do not arm openCell on growth: that garage avoid sits next to
             * doorB and seals the post-splice exit (trapped at doorB). */
            env._ctOddOpenCell = null;
            env._ctOddOpenFoodKey = null;
            /* Lock variant: only lock if this growth pair is the next target. */
            if (env._ctOddLockVariant && typeof growSeg.depth === "number") {
              var gEdge = growSeg.kind === "grow-bottom" ? "bottom" : "right";
              var gNext = fringeLockNextTarget(env, m);
              if (gNext && gNext.edge === gEdge && gNext.depth === growSeg.depth) {
                env._ctOddLockPending = { edge: gEdge, depth: growSeg.depth };
              } else {
                env._ctOddLockPending = null;
              }
            }
            env._ctFringeWhy = "enter-grow";
            return growFirst;
          }
        }
        var growGoal = fringeCtApproachGoal(growSeg.doorA, m, h);
        var growAct = fringeCtToward(env, n, m, growGoal, growAvoid);
        if (growAct !== null) {
          env._ctFringeWhy = growSeg.ready ? "ct-to-grow" : "ct-grow-wait";
          return growAct;
        }
      }
    }

    /* Food in sub: pure CT; still soft-avoid garage from a prior fringe eat. */
    if (!foodOut) {
      var avoidIn = fringeAvoidList(env, null);
      var actIn = cellTreeEvenCore(env, n, {
        subM: m,
        block: avoidIn || undefined
      });
      if (actIn !== null && safe.indexOf(actIn) >= 0 &&
          fringeStepStaysInSub(env, m, actIn) &&
          !fringeStepLandsOn(env, actIn, avoidIn)) {
        var treeIn = fringeCachedTreeCur(env, m);
        var ir = h[0] + DIRS[actIn][0];
        var ic = h[1] + DIRS[actIn][1];
        if (isCellMove(h[0], h[1], actIn) &&
            canMoveInCellTree(treeIn, h[0], h[1], ir, ic, actIn) &&
            fringeCtEscapeAfter(env, m, actIn)) {
          env._ctFringeWhy = "ct";
          return actIn;
        }
      }
      var ctIn = fringeCtToward(env, n, m, env.food, avoidIn);
      if (ctIn !== null) {
        env._ctFringeWhy = "ct";
        return ctIn;
      }
      var holdIn = fringeCtHold(env, m, safe, avoidIn);
      if (holdIn !== null) {
        var holdR = h[0] + DIRS[holdIn][0];
        var holdC = h[1] + DIRS[holdIn][1];
        var holdEats = !!(env.food && env.food[0] === holdR &&
          env.food[1] === holdC);
        if (!holdEats || fringeCtEscapeAfter(env, m, holdIn)) {
          env._ctFringeWhy = "ct-hold";
          return holdIn;
        }
      }
      var tipIn = fringeTipVacateFresh(env, safe);
      if (tipIn !== null) {
        fringeTipWhy(env, "tip-vacate");
        return tipIn;
      }
      var leaveIn = fringeForcedLeave(env, m, safe);
      if (leaveIn !== null) {
        env._ctFringeWhy = "forced-leave";
        return leaveIn;
      }
      /*
       * Last resort H-F-T: CT refused the step onto F but tip is adj - eat
       * and tip-vacate next. Only at high fill (early tip-eat seals mid-game).
       */
      var emptyIn = n * n - env.snake.length;
      if (fringeTipAdjFood(env) && emptyIn <= 8) {
        var hft;
        for (hft = 0; hft < 4; hft += 1) {
          if (h[0] + DIRS[hft][0] === env.food[0] &&
              h[1] + DIRS[hft][1] === env.food[1] &&
              safe.indexOf(hft) >= 0) {
            env._ctFringeWhy = "tip-eat";
            return hft;
          }
        }
      }
      /*
       * emptyLeft<=3 sole safe onto in-sub F: take it (728565 CT refused
       * with tip one diagonal off). Nearly full - return plan is finish.
       */
      if (safe.length === 1 && emptyIn <= 3 && env.food) {
        var onlyF = safe[0];
        var ofr = h[0] + DIRS[onlyF][0];
        var ofc = h[1] + DIRS[onlyF][1];
        if (env.food[0] === ofr && env.food[1] === ofc) {
          env._ctFringeWhy = "late-eat";
          return onlyF;
        }
      }
      if (safe.length === 1) {
        var only = safe[0];
        var or = h[0] + DIRS[only][0];
        var oc = h[1] + DIRS[only][1];
        var onlyEat = !!(env.food && env.food[0] === or && env.food[1] === oc);
        if (!onlyEat || fringeTipAdjFood(env) || fringeCtEscapeAfter(env, m, only) ||
            env.snake.length + 1 >= n * n) {
          env._ctFringeWhy = "endgame-step";
          return only;
        }
      }
      return fringeLockExpOrThrow(env, n, safe, h,
        "no Cell-Tree-legal in-sub step (food in-sub)");
    }

    var seg = findFringeCtSegment(env, m);
    /*
     * Endgame / high-fill fringe wait: CT door approach cannot free a
     * body-blocked G or run a corner splice with no rim left. Follow tip
     * until adj to F - but not while a growth pair is still free (claim
     * those first via needGrow above).
     */
    var emptyLeft = n * n - env.snake.length;
    var growStill = emptyLeft > 1 ? findFringeGrowthSegment(env, m) : null;
    /*
     * Lock near-full with fringe food: do not tip-vacate yet - fall through to
     * CT door approach (95045 tip-livelock). Still tip-vacate at emptyLeft<=1
     * or when not Lock / not foodOut / not in the late band.
     */
    var lockDeferTip = env._ctOddLockVariant && foodOut &&
      emptyLeft > 1 && emptyLeft <= 5;
    if (emptyLeft <= 1 ||
        (emptyLeft <= 8 && (!seg || !seg.ready) && !growStill && !lockDeferTip)) {
      var eatEnd = fringeAdjFoodAction(env, safe);
      if (eatEnd !== null) {
        env._ctFringeWhy = "fringe-eat";
        return eatEnd;
      }
      var tipEnd = fringeTipVacateFresh(env, safe);
      if (tipEnd !== null) {
        fringeTipWhy(env, "tip-vacate");
        return tipEnd;
      }
      /* Sole safe step that CT refuses: take it (forced). Multi-safe
       * non-CT picks were trapping 728/132 - do not wander. */
      if (safe.length === 1) {
        env._ctFringeWhy = "endgame-step";
        return safe[0];
      }
    }
    /*
     * CT target = doorA (food-side entry). Soft-avoid doorB while waiting.
     * openCell (edge-2x2 garage) is only avoided after splice starts so the
     * return pass does not fill it - never during the approach.
     * If G is blocked (no seg yet), still approach the would-be doorA so we
     * are ready when tip frees G - do not null-goal orbit.
     */
    var doorA = seg ? seg.doorA : null;
    if (!doorA && env.food) {
      var fr0 = env.food[0];
      var fc0 = env.food[1];
      if (fc0 === m && fr0 < m) {
        if (fr0 + 1 < m) {
          doorA = [fr0 + 1, m - 1];
        } else if (fr0 - 1 >= 0) {
          doorA = [fr0, m - 1];
        }
      } else if (fr0 === m && fc0 < m) {
        if (fc0 - 1 >= 0) {
          doorA = [m - 1, fc0 - 1];
        } else if (fc0 + 1 < m) {
          doorA = [m - 1, fc0];
        }
      }
    }
    var extraAvoid = seg && !seg.ready ? seg.doorB : null;
    var avoid = fringeAvoidList(env, extraAvoid);
    var atDoorA = !!(doorA && h[0] === doorA[0] && h[1] === doorA[1]);
    var approachGoal = fringeCtApproachGoal(doorA, m, h);

    if (seg && seg.ready && atDoorA) {
      var first = seg.splice[0];
      if (safe.indexOf(first) >= 0) {
        env._ctBumpActions = seg.splice.slice(1);
        if (!env._ctBumpActions.length) {
          env._ctBumpActions = null;
        }
        /* Arm garage avoid only once we commit to the food-side splice. */
        if (seg.openCell) {
          env._ctOddOpenCell = seg.openCell.slice();
          env._ctOddOpenFoodKey = foodKey;
        }
        env._ctFringeWhy = "enter-bump";
        return first;
      }
    }

    /* Resume CT toward CT-reachable approach goal (pred or doorA). */
    var ctAct = fringeCtToward(env, n, m, approachGoal, avoid);
    if (ctAct !== null) {
      env._ctFringeWhy = seg && seg.ready ? "ct-to-door" : "ct-wait";
      return ctAct;
    }
    var holdOut = fringeCtHold(env, m, safe, avoid);
    if (holdOut !== null) {
      env._ctFringeWhy = "ct-hold";
      return holdOut;
    }
    var tipOut = fringeTipVacateFresh(env, safe);
    if (tipOut !== null) {
      fringeTipWhy(env, "tip-vacate");
      return tipOut;
    }
    var leaveOut = fringeForcedLeave(env, m, safe);
    if (leaveOut !== null) {
      env._ctFringeWhy = "forced-leave";
      return leaveOut;
    }
    return fringeLockExpOrThrow(env, n, safe, h,
      "no Cell-Tree-legal in-sub step (fringe food)");
  }

  /**
   * In-sub hold: CT-legal safe in-sub only. Never fringe, never non-CT, never
   * unsafe facing. null => caller throws (CT broken - do not early-Ham).
   */
  function fringeCtHold(env, m, safe, avoid) {
    var h = env.snake[0];
    var tree = fringeCachedTreeCur(env, m);
    function ok(act) {
      if (safe.indexOf(act) < 0 || !fringeStepStaysInSub(env, m, act)) {
        return false;
      }
      if (fringeStepLandsOn(env, act, avoid)) {
        return false;
      }
      var nr = h[0] + DIRS[act][0];
      var nc = h[1] + DIRS[act][1];
      return isCellMove(h[0], h[1], act) &&
        canMoveInCellTree(tree, h[0], h[1], nr, nc, act);
    }
    if (ok(env.facing)) {
      return env.facing | 0;
    }
    var i;
    for (i = 0; i < safe.length; i += 1) {
      if (ok(safe[i])) {
        return safe[i];
      }
    }
    return null;
  }

  /**
   * Fill fraction where Cell Tree Odd hands off to Heal (then Hamilton).
   * Only parameter: fill >= T enters heal regardless of fringe food.
   */
  var CELL_TREE_ODD_HAM_FILL_T = 0.36;

  /** Default heal style for Observe Cell Tree (healTS; ctHamThenTS did not beat it). */
  var CELL_TREE_ODD_HEAL_STYLE = "healTS";

  function ctOddSkipNow(env) {
    var cy = E.getHamiltonCycle(env.size);
    return E.masterCycleSkip(cy, env.snake);
  }

  /**
   * Pick safe step minimizing skipAfter. preferSucc / allowEat / maxSkipRaise
   * tune aggression. Returns action or null.
   */
  function ctOddSkipMinStep(env, opts) {
    opts = opts || {};
    var preferSucc = opts.preferSucc !== false;
    var allowEat = opts.allowEat !== false;
    var maxRaise = typeof opts.maxRaise === "number" ? opts.maxRaise : 1e9;
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var n = env.size;
    var cy = E.getHamiltonCycle(n);
    var h = env.snake[0];
    var skipNow = E.masterCycleSkip(cy, env.snake);
    var cycleNext = cy.succ[h[0] + "," + h[1]];
    var best = null;
    var i;
    for (i = 0; i < safe.length; i += 1) {
      var a = safe[i];
      var nr = h[0] + DIRS[a][0];
      var nc = h[1] + DIRS[a][1];
      var eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
      if (eating && !allowEat) {
        continue;
      }
      var isSucc = !!(cycleNext && cycleNext[0] === nr && cycleNext[1] === nc);
      var skipAfter = E.masterSkipAfterMove(cy, env.snake, nr, nc, eating);
      if (skipAfter > skipNow + maxRaise) {
        continue;
      }
      var flood = E.floodFromMove(env, nr, nc, eating).count;
      var cand = {
        action: a,
        skipAfter: skipAfter,
        isSucc: isSucc,
        eating: eating,
        flood: flood
      };
      if (!best) {
        best = cand;
        continue;
      }
      if (cand.skipAfter !== best.skipAfter) {
        if (cand.skipAfter < best.skipAfter) {
          best = cand;
        }
        continue;
      }
      if (preferSucc && cand.isSucc !== best.isSucc) {
        if (cand.isSucc) {
          best = cand;
        }
        continue;
      }
      if (cand.flood !== best.flood) {
        if (cand.flood > best.flood) {
          best = cand;
        }
        continue;
      }
      if (!cand.eating && best.eating) {
        best = cand;
      }
    }
    return best ? best.action : null;
  }

  function ctOddCycleSuccAct(env) {
    var cy = E.getHamiltonCycle(env.size);
    var h = env.snake[0];
    var cycleNext = cy.succ[h[0] + "," + h[1]];
    var safe = env.safeActions();
    if (!cycleNext || !safe.length) {
      return null;
    }
    var di;
    for (di = 0; di < 4; di += 1) {
      if (h[0] + DIRS[di][0] === cycleNext[0] &&
          h[1] + DIRS[di][1] === cycleNext[1] &&
          safe.indexOf(di) >= 0) {
        return di;
      }
    }
    return null;
  }

  function ctOddStepStats(env, act) {
    var h = env.snake[0];
    var tip = env.snake[env.snake.length - 1];
    var nr = h[0] + DIRS[act][0];
    var nc = h[1] + DIRS[act][1];
    var eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
    var freeN = E.freeNeighborsAfter(env, nr, nc, eating);
    var flood = E.floodFromMove(env, nr, nc, eating).count;
    var ontoTip = !!(tip && tip[0] === nr && tip[1] === nc);
    var n = env.size;
    var afterLen = env.snake.length + (eating ? 1 : 0);
    var wins = afterLen >= n * n ||
      (typeof env.targetLength === "number" && afterLen >= env.targetLength);
    return {
      nr: nr,
      nc: nc,
      eating: eating,
      freeN: freeN,
      flood: flood,
      ontoTip: ontoTip,
      wins: wins,
      need: env.snake.length + (eating ? 0 : 0)
    };
  }

  function ctOddCycleDist(cy, r0, c0, r1, c1) {
    var i0 = cy.index[r0 + "," + c0];
    var i1 = cy.index[r1 + "," + c1];
    if (i0 === undefined || i1 === undefined) {
      return cy.len + 1;
    }
    return (i1 - i0 + cy.len) % cy.len;
  }

  /**
   * Largest skip-debt gap on the Ham cycle: empty cells between a body neck
   * and its parent along the forward cycle. Returns nearest gap cell to head
   * (Manhattan), or null.
   */
  function ctOddLargestGapGoal(env) {
    var n = env.size;
    var cy = E.getHamiltonCycle(n);
    var snake = env.snake;
    if (!snake || snake.length < 2) {
      return null;
    }
    var body = {};
    var i;
    for (i = 0; i < snake.length; i += 1) {
      body[snake[i][0] + "," + snake[i][1]] = true;
    }
    var bestSize = 0;
    var bestCells = [];
    for (i = 0; i < snake.length - 1; i += 1) {
      var a = snake[i];
      var b = snake[i + 1];
      if (cy.index[a[0] + "," + a[1]] === undefined ||
          cy.index[b[0] + "," + b[1]] === undefined) {
        continue;
      }
      var d = ctOddCycleDist(cy, b[0], b[1], a[0], a[1]);
      if (d <= 1) {
        continue;
      }
      var gapSize = d - 1;
      if (gapSize < bestSize) {
        continue;
      }
      var cells = [];
      var cur = cy.succ[b[0] + "," + b[1]];
      var guard = 0;
      while (cur && !(cur[0] === a[0] && cur[1] === a[1]) && guard < cy.len) {
        guard += 1;
        var ck = cur[0] + "," + cur[1];
        if (!body[ck]) {
          cells.push([cur[0], cur[1]]);
        }
        cur = cy.succ[ck];
      }
      if (gapSize > bestSize) {
        bestSize = gapSize;
        bestCells = cells;
      } else if (gapSize === bestSize && cells.length > bestCells.length) {
        bestCells = cells;
      }
    }
    if (!bestCells.length) {
      return null;
    }
    var h = snake[0];
    var best = bestCells[0];
    var bestMan = Math.abs(best[0] - h[0]) + Math.abs(best[1] - h[1]);
    for (i = 1; i < bestCells.length; i += 1) {
      var man = Math.abs(bestCells[i][0] - h[0]) + Math.abs(bestCells[i][1] - h[1]);
      if (man < bestMan) {
        bestMan = man;
        best = bestCells[i];
      }
    }
    return { cell: best, gapSize: bestSize };
  }

  /** Step toward gap goal: max flood among moves that reduce Manhattan to goal. */
  function ctOddGapFillStep(env) {
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var goalInfo = ctOddLargestGapGoal(env);
    var succAct = ctOddCycleSuccAct(env);
    if (!goalInfo) {
      return succAct !== null ? succAct :
        E.masterHamiltonAction(env, { healOnly: true });
    }
    var goal = goalInfo.cell;
    var h = env.snake[0];
    var need = Math.max(1, env.snake.length - 1);
    var man0 = Math.abs(goal[0] - h[0]) + Math.abs(goal[1] - h[1]);
    var best = null;
    var i;
    for (i = 0; i < safe.length; i += 1) {
      var a = safe[i];
      var st = ctOddStepStats(env, a);
      if (st.eating && st.freeN < 1 && !st.wins && !st.ontoTip) {
        continue;
      }
      var man1 = Math.abs(goal[0] - st.nr) + Math.abs(goal[1] - st.nc);
      var closes = man1 < man0;
      var cand = {
        action: a,
        closes: closes,
        man: man1,
        flood: st.flood,
        freeN: st.freeN,
        isSucc: a === succAct
      };
      if (!best) {
        best = cand;
        continue;
      }
      if (cand.closes !== best.closes) {
        if (cand.closes) {
          best = cand;
        }
        continue;
      }
      if (cand.flood !== best.flood) {
        if (cand.flood > best.flood) {
          best = cand;
        }
        continue;
      }
      if (cand.freeN !== best.freeN) {
        if (cand.freeN > best.freeN) {
          best = cand;
        }
        continue;
      }
      if (cand.man !== best.man) {
        if (cand.man < best.man) {
          best = cand;
        }
        continue;
      }
      if (cand.isSucc && !best.isSucc) {
        best = cand;
      }
    }
    if (best && (best.closes || best.flood >= need || best.isSucc)) {
      return best.action;
    }
    return succAct !== null ? succAct :
      E.masterHamiltonAction(env, { healOnly: true });
  }

  /**
   * HealTS for CT handoff. Guards:
   * - Refuse zero-escape eats (269263 @T=0.40).
   * - After CT-Ham chase (`_ctOddPostChaseHeal`): if HealTS/succ would take
   *   cycleSucc with solid flood < L-1 (not onto tip/win), prefer a non-succ
   *   alt with better flood, flood >= L-1, or equal flood + higher freeN.
   *   467238 died riding Ham succ into a shrinking rim after a safe handoff.
   *   Keep succ when no safer alt - blind max-flood off-cycle regressed @0.38.
   *   Gate is post-chase only so plain healTS tip-chase at T=0.36 is unchanged.
   */
  function ctOddHealTSSafeStep(env) {
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var succAct = ctOddCycleSuccAct(env);
    var need = Math.max(0, env.snake.length - 1);
    var postChase = !!env._ctOddPostChaseHeal;

    function succUnderflooded(st) {
      if (!st || st.ontoTip || st.wins) {
        return false;
      }
      return st.flood < need;
    }

    function pickMaxFlood() {
      var best = null;
      var bi;
      for (bi = 0; bi < safe.length; bi += 1) {
        var s2 = ctOddStepStats(env, safe[bi]);
        if (s2.eating && s2.freeN < 1 && !s2.ontoTip && !s2.wins) {
          continue;
        }
        var cand = {
          action: safe[bi],
          flood: s2.flood,
          freeN: s2.freeN,
          isSucc: safe[bi] === succAct
        };
        if (!best) {
          best = cand;
          continue;
        }
        if (cand.flood !== best.flood) {
          if (cand.flood > best.flood) {
            best = cand;
          }
          continue;
        }
        if (cand.freeN !== best.freeN) {
          if (cand.freeN > best.freeN) {
            best = cand;
          }
          continue;
        }
        if (cand.isSucc && !best.isSucc) {
          best = cand;
        }
      }
      return best ? best.action : safe[0];
    }

    /**
     * Safer non-succ than underflooded succ; null if none (keep succ).
     * Equal flood + higher freeN counts (467238 rim ride: succ n2 vs west n3).
     * Worse flood does not - that off-cycle grab regressed ctHam@0.38.
     */
    function saferThanUnderfloodSucc(succSt) {
      var best = null;
      var bi;
      var baseFlood = succSt ? succSt.flood : -1;
      var baseFree = succSt ? succSt.freeN : -1;
      for (bi = 0; bi < safe.length; bi += 1) {
        if (safe[bi] === succAct) {
          continue;
        }
        var s2 = ctOddStepStats(env, safe[bi]);
        if (s2.eating && s2.freeN < 1 && !s2.ontoTip && !s2.wins) {
          continue;
        }
        var betterFlood = s2.flood > baseFlood || s2.flood >= need;
        var equalFloodMoreFree =
          s2.flood === baseFlood && s2.freeN > baseFree;
        if (!(betterFlood || equalFloodMoreFree || s2.ontoTip || s2.wins)) {
          continue;
        }
        var cand = {
          action: safe[bi],
          flood: s2.flood,
          freeN: s2.freeN
        };
        if (!best) {
          best = cand;
          continue;
        }
        if (cand.flood !== best.flood) {
          if (cand.flood > best.flood) {
            best = cand;
          }
          continue;
        }
        if (cand.freeN > best.freeN) {
          best = cand;
        }
      }
      return best ? best.action : null;
    }

    function maybeEscapeUnderfloodSucc(candidateAct) {
      if (!postChase || candidateAct !== succAct) {
        return candidateAct;
      }
      var st0 = ctOddStepStats(env, candidateAct);
      if (!succUnderflooded(st0)) {
        return candidateAct;
      }
      var alt = saferThanUnderfloodSucc(st0);
      return alt !== null ? alt : candidateAct;
    }

    var act = E.healTailSafeNextAction(env);
    if (safe.indexOf(act) < 0) {
      if (succAct !== null) {
        var sFall = ctOddStepStats(env, succAct);
        if (!(sFall.eating && sFall.freeN < 1 && !sFall.wins && !sFall.ontoTip)) {
          return maybeEscapeUnderfloodSucc(succAct);
        }
      }
      return pickMaxFlood();
    }
    var st = ctOddStepStats(env, act);
    if (st.eating && st.freeN < 1 && !st.wins && !st.ontoTip) {
      if (succAct !== null) {
        var ss = ctOddStepStats(env, succAct);
        if (!(ss.eating && ss.freeN < 1 && !ss.wins && !ss.ontoTip)) {
          return maybeEscapeUnderfloodSucc(succAct);
        }
      }
      return pickMaxFlood();
    }
    return maybeEscapeUnderfloodSucc(act);
  }

  /** Prefer interior (off-edge) when HealTS would stay in a freeN<=1 rim cell. */
  function ctOddHealTSInteriorStep(env) {
    var act = ctOddHealTSSafeStep(env);
    var safe = env.safeActions();
    if (!safe.length) {
      return act;
    }
    var n = env.size;
    var st = ctOddStepStats(env, act);
    var nextEdge = st.nr === 0 || st.nc === 0 ||
      st.nr === n - 1 || st.nc === n - 1;
    if (!(nextEdge && st.freeN <= 1 && !st.ontoTip)) {
      return act;
    }
    var best = null;
    var i;
    for (i = 0; i < safe.length; i += 1) {
      var s2 = ctOddStepStats(env, safe[i]);
      if (s2.eating && s2.freeN < 1 && !s2.wins && !s2.ontoTip) {
        continue;
      }
      var edge2 = s2.nr === 0 || s2.nc === 0 ||
        s2.nr === n - 1 || s2.nc === n - 1;
      var cand = {
        action: safe[i],
        edge: edge2,
        flood: s2.flood,
        freeN: s2.freeN
      };
      if (!best) {
        best = cand;
        continue;
      }
      if (cand.edge !== best.edge) {
        if (!cand.edge) {
          best = cand;
        }
        continue;
      }
      if (cand.flood !== best.flood) {
        if (cand.flood > best.flood) {
          best = cand;
        }
        continue;
      }
      if (cand.freeN > best.freeN) {
        best = cand;
      }
    }
    return best ? best.action : act;
  }

  /**
   * Empty cycle cells in tip-forward order (earliest first). If tip is off
   * cycle, absolute path-index order. Seam/late cells appear only after
   * earlier empties are listed - fill walks this list and skips unreachable.
   */
  function ctOddTipForwardEmpties(env) {
    var cy = E.getHamiltonCycle(env.size);
    var snake = env.snake;
    var out = [];
    if (!snake || !snake.length || !cy.path || !cy.path.length) {
      return out;
    }
    var body = {};
    var i;
    for (i = 0; i < snake.length; i += 1) {
      body[snake[i][0] + "," + snake[i][1]] = true;
    }
    var tip = snake[snake.length - 1];
    var tipKey = tip[0] + "," + tip[1];
    if (cy.index[tipKey] === undefined) {
      for (i = 0; i < cy.path.length; i += 1) {
        var p = cy.path[i];
        var pk = p[0] + "," + p[1];
        if (!body[pk]) {
          out.push({ cell: [p[0], p[1]], index: i, ord: out.length });
        }
      }
      return out;
    }
    var guard = 0;
    var cur = cy.succ[tipKey];
    while (cur && guard < cy.len) {
      guard += 1;
      var ck = cur[0] + "," + cur[1];
      if (cur[0] === tip[0] && cur[1] === tip[1]) {
        break;
      }
      if (!body[ck]) {
        out.push({
          cell: [cur[0], cur[1]],
          index: cy.index[ck],
          ord: out.length
        });
      }
      cur = cy.succ[ck];
    }
    return out;
  }

  function ctOddEarliestEmptyGoal(env) {
    var list = ctOddTipForwardEmpties(env);
    if (!list.length) {
      return null;
    }
    return {
      cell: list[0].cell,
      index: list[0].index,
      fromTip: true
    };
  }

  /**
   * Hard-safe move filter: never chase into a pocket. flood >= L-1 (or tip/win),
   * freeN>=1 (or tip/win), refuse zero-escape eats.
   */
  function ctOddHardSafeMoveCands(env) {
    var safe = env.safeActions();
    var need = Math.max(1, env.snake.length - 1);
    var succAct = ctOddCycleSuccAct(env);
    var cy = E.getHamiltonCycle(env.size);
    var cands = [];
    var i;
    for (i = 0; i < safe.length; i += 1) {
      var a = safe[i];
      var st = ctOddStepStats(env, a);
      if (st.eating && st.freeN < 1 && !st.wins && !st.ontoTip) {
        continue;
      }
      if (!st.eating && st.freeN < 1 && !st.ontoTip) {
        continue;
      }
      if (st.flood < need && !st.ontoTip && !st.wins) {
        continue;
      }
      var skipAfter = E.masterSkipAfterMove(cy, env.snake, st.nr, st.nc, st.eating);
      cands.push({
        action: a,
        nr: st.nr,
        nc: st.nc,
        flood: st.flood,
        freeN: st.freeN,
        skipAfter: skipAfter,
        isSucc: a === succAct ? 1 : 0,
        ontoTip: st.ontoTip ? 1 : 0,
        eating: st.eating ? 1 : 0
      });
    }
    return cands;
  }

  function ctOddPickBestSafe(cands) {
    if (!cands || !cands.length) {
      return null;
    }
    cands.sort(function (x, y) {
      if (x.flood !== y.flood) {
        return y.flood - x.flood;
      }
      if (x.freeN !== y.freeN) {
        return y.freeN - x.freeN;
      }
      if (x.skipAfter !== y.skipAfter) {
        return x.skipAfter - y.skipAfter;
      }
      if (x.isSucc !== y.isSucc) {
        return y.isSucc - x.isSucc;
      }
      return x.action - y.action;
    });
    return cands[0].action;
  }

  /**
   * Safety first, then earliest fillable cycle cell:
   * 1) Only hard-safe moves (flood/freeN).
   * 2) If a hard-safe move lands on an empty cycle cell, take the earliest such.
   * 3) Else walk tip-forward empties; for the earliest that has a hard-safe
   *    closing step, take the best closer.
   * 4) Else idle among hard-safe (flood/succ). Never drop safety to chase.
   */
  function ctOddEarliestFillStep(env) {
    var succAct = ctOddCycleSuccAct(env);
    var hard = ctOddHardSafeMoveCands(env);
    if (!hard.length) {
      return ctOddHealTSSafeStep(env);
    }
    var empties = ctOddTipForwardEmpties(env);
    var emptyOrd = {};
    var ei;
    for (ei = 0; ei < empties.length; ei += 1) {
      emptyOrd[empties[ei].cell[0] + "," + empties[ei].cell[1]] = empties[ei].ord;
    }
    var h = env.snake[0];
    var landBest = null;
    var hi;
    for (hi = 0; hi < hard.length; hi += 1) {
      var m = hard[hi];
      var lk = m.nr + "," + m.nc;
      if (emptyOrd[lk] === undefined) {
        continue;
      }
      var land = {
        action: m.action,
        ord: emptyOrd[lk],
        flood: m.flood,
        freeN: m.freeN,
        skipAfter: m.skipAfter,
        isSucc: m.isSucc
      };
      if (!landBest || land.ord < landBest.ord ||
          (land.ord === landBest.ord && land.flood > landBest.flood) ||
          (land.ord === landBest.ord && land.flood === landBest.flood &&
            land.freeN > landBest.freeN)) {
        landBest = land;
      }
    }
    if (landBest) {
      return landBest.action;
    }
    for (ei = 0; ei < empties.length; ei += 1) {
      var goal = empties[ei].cell;
      var man0 = Math.abs(goal[0] - h[0]) + Math.abs(goal[1] - h[1]);
      var closers = [];
      for (hi = 0; hi < hard.length; hi += 1) {
        var mv = hard[hi];
        var man1 = Math.abs(goal[0] - mv.nr) + Math.abs(goal[1] - mv.nc);
        if (man1 >= man0) {
          continue;
        }
        closers.push({
          action: mv.action,
          man: man1,
          flood: mv.flood,
          freeN: mv.freeN,
          skipAfter: mv.skipAfter,
          isSucc: mv.isSucc
        });
      }
      if (!closers.length) {
        continue;
      }
      closers.sort(function (x, y) {
        if (x.man !== y.man) {
          return x.man - y.man;
        }
        if (x.flood !== y.flood) {
          return y.flood - x.flood;
        }
        if (x.freeN !== y.freeN) {
          return y.freeN - x.freeN;
        }
        if (x.skipAfter !== y.skipAfter) {
          return x.skipAfter - y.skipAfter;
        }
        return y.isSucc - x.isSucc;
      });
      return closers[0].action;
    }
    if (succAct !== null) {
      var si;
      for (si = 0; si < hard.length; si += 1) {
        if (hard[si].action === succAct) {
          return succAct;
        }
      }
    }
    return ctOddPickBestSafe(hard);
  }

  /**
   * Earliest-fill while skip is large; HealTS once skip is moderate.
   */
  function ctOddEarliestThenTSStep(env) {
    var skip = ctOddSkipNow(env);
    var n = env.size;
    if (skip > n * n) {
      return ctOddEarliestFillStep(env);
    }
    return ctOddHealTSSafeStep(env);
  }

  /**
   * Run earliestFill for the first `cap` heal steps (cap frozen at heal entry),
   * then HealTS. Caps: L, 2L, 3L, or n at entry.
   */
  function ctOddEarliestBurstThenTSStep(env, subKey, capFn) {
    if (env._ctOddHealSub !== subKey) {
      env._ctOddHealSub = subKey;
      env._ctOddHealStall = 0;
      env._ctOddHealBurstCap = capFn(env);
    }
    env._ctOddHealStall = (env._ctOddHealStall || 0) + 1;
    if (env._ctOddHealStall <= (env._ctOddHealBurstCap || 0)) {
      return ctOddEarliestFillStep(env);
    }
    return ctOddHealTSSafeStep(env);
  }

  /**
   * HealTS by default; if a hard-safe move lands on any tip-forward empty,
   * take the earliest such landing instead (pack opportunistically).
   */
  function ctOddTSEarliestLandStep(env) {
    var hard = ctOddHardSafeMoveCands(env);
    var empties = ctOddTipForwardEmpties(env);
    if (hard.length && empties.length) {
      var emptyOrd = {};
      var ei;
      for (ei = 0; ei < empties.length; ei += 1) {
        emptyOrd[empties[ei].cell[0] + "," + empties[ei].cell[1]] =
          empties[ei].ord;
      }
      var landBest = null;
      var hi;
      for (hi = 0; hi < hard.length; hi += 1) {
        var m = hard[hi];
        var lk = m.nr + "," + m.nc;
        if (emptyOrd[lk] === undefined) {
          continue;
        }
        var land = {
          action: m.action,
          ord: emptyOrd[lk],
          flood: m.flood,
          freeN: m.freeN
        };
        if (!landBest || land.ord < landBest.ord ||
            (land.ord === landBest.ord && land.flood > landBest.flood)) {
          landBest = land;
        }
      }
      if (landBest) {
        return landBest.action;
      }
    }
    return ctOddHealTSSafeStep(env);
  }

  /**
   * When head is on-cycle, pack with earliestFill; otherwise HealTS approach.
   */
  function ctOddOnCycleEarliestElseTSStep(env) {
    var cy = E.getHamiltonCycle(env.size);
    var h = env.snake[0];
    if (cy.index[h[0] + "," + h[1]] !== undefined) {
      return ctOddEarliestFillStep(env);
    }
    return ctOddHealTSSafeStep(env);
  }

  /**
   * BFS through empties (tip vacated) toward earliest tip-forward empty that
   * lies in the reachable set; take first step if hard-safe. Else HealTS.
   */
  function ctOddEarliestBfsStep(env) {
    var empties = ctOddTipForwardEmpties(env);
    if (!empties.length) {
      return ctOddHealTSSafeStep(env);
    }
    var n = env.size;
    var snake = env.snake;
    var tip = snake[snake.length - 1];
    var blocked = {};
    var i;
    for (i = 0; i < snake.length - 1; i += 1) {
      blocked[snake[i][0] + "," + snake[i][1]] = true;
    }
    /* Tip vacates for non-eat roam. */
    var h = snake[0];
    var goalSet = {};
    for (i = 0; i < empties.length; i += 1) {
      goalSet[empties[i].cell[0] + "," + empties[i].cell[1]] = empties[i].ord;
    }
    var q = [[h[0], h[1]]];
    var prev = {};
    prev[h[0] + "," + h[1]] = null;
    var qi = 0;
    var foundKey = null;
    var foundOrd = 1e9;
    while (qi < q.length) {
      var cur = q[qi];
      qi += 1;
      var ck = cur[0] + "," + cur[1];
      if (goalSet[ck] !== undefined && !(cur[0] === h[0] && cur[1] === h[1])) {
        if (goalSet[ck] < foundOrd) {
          foundOrd = goalSet[ck];
          foundKey = ck;
        }
        /* Keep scanning same BFS layer? For earliest, continue BFS fully among
           reached goals - actually BFS order is by dist not ord. Collect all
           reached goals then pick min ord. */
      }
      var d;
      for (d = 0; d < 4; d += 1) {
        var nr = cur[0] + DIRS[d][0];
        var nc = cur[1] + DIRS[d][1];
        if (nr < 0 || nc < 0 || nr >= n || nc >= n) {
          continue;
        }
        var nk = nr + "," + nc;
        if (blocked[nk] || prev[nk] !== undefined) {
          continue;
        }
        if (tip && nr === tip[0] && nc === tip[1]) {
          /* tip ok (vacating) */
        } else if (blocked[nk]) {
          continue;
        }
        prev[nk] = ck;
        q.push([nr, nc]);
      }
    }
    /* Pick earliest goal among all reached. */
    var gk;
    foundKey = null;
    foundOrd = 1e9;
    for (gk in goalSet) {
      if (!Object.prototype.hasOwnProperty.call(goalSet, gk)) {
        continue;
      }
      if (prev[gk] === undefined) {
        continue;
      }
      if (goalSet[gk] < foundOrd) {
        foundOrd = goalSet[gk];
        foundKey = gk;
      }
    }
    if (!foundKey) {
      return ctOddHealTSSafeStep(env);
    }
    /* Walk back to step after head. */
    var stepKey = foundKey;
    var guard = 0;
    while (prev[stepKey] !== null && prev[stepKey] !== undefined && guard < n * n) {
      guard += 1;
      if (prev[stepKey] === h[0] + "," + h[1]) {
        break;
      }
      stepKey = prev[stepKey];
    }
    if (prev[stepKey] !== h[0] + "," + h[1]) {
      return ctOddHealTSSafeStep(env);
    }
    var parts = stepKey.split(",");
    var tr = parseInt(parts[0], 10);
    var tc = parseInt(parts[1], 10);
    var act = null;
    var di;
    for (di = 0; di < 4; di += 1) {
      if (h[0] + DIRS[di][0] === tr && h[1] + DIRS[di][1] === tc) {
        act = di;
        break;
      }
    }
    if (act === null) {
      return ctOddHealTSSafeStep(env);
    }
    var hard = ctOddHardSafeMoveCands(env);
    for (i = 0; i < hard.length; i += 1) {
      if (hard[i].action === act) {
        return act;
      }
    }
    return ctOddHealTSSafeStep(env);
  }

  /**
   * HealTS until skip stalls, then gapFill/succ until skip drops again.
   */
  function ctOddHealStallSwitchStep(env, fillMode) {
    var skip = ctOddSkipNow(env);
    var n = env.size;
    if (typeof env._ctOddHealSkipSeen !== "number" ||
        env._ctOddHealSub !== "stall") {
      env._ctOddHealSkipSeen = skip;
      env._ctOddHealStall = 0;
      env._ctOddHealSub = "stall";
    }
    if (skip < env._ctOddHealSkipSeen) {
      env._ctOddHealSkipSeen = skip;
      env._ctOddHealStall = 0;
    } else {
      env._ctOddHealStall = (env._ctOddHealStall || 0) + 1;
    }
    var stalled = env._ctOddHealStall >= Math.max(4, n);
    if (stalled) {
      if (fillMode === "gap") {
        return ctOddGapFillStep(env);
      }
      if (fillMode === "debt") {
        return ctOddDebtFillStep(env);
      }
      return E.masterHamiltonAction(env, { healOnly: true });
    }
    return ctOddHealTSSafeStep(env);
  }

  /**
   * Direct skip-debt fill: minimize skipAfter among safe moves.
   * Refuse zero-escape eats and freeN=0 non-tip. Prefer succ, then flood.
   */
  function ctOddDebtFillStep(env) {
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var cy = E.getHamiltonCycle(env.size);
    var skipNow = E.masterCycleSkip(cy, env.snake);
    var succAct = ctOddCycleSuccAct(env);
    var best = null;
    var i;
    for (i = 0; i < safe.length; i += 1) {
      var a = safe[i];
      var st = ctOddStepStats(env, a);
      if (st.eating && st.freeN < 1 && !st.wins && !st.ontoTip) {
        continue;
      }
      if (!st.eating && st.freeN < 1 && !st.ontoTip) {
        continue;
      }
      var skipAfter = E.masterSkipAfterMove(cy, env.snake, st.nr, st.nc, st.eating);
      var cand = {
        action: a,
        skipAfter: skipAfter,
        flood: st.flood,
        freeN: st.freeN,
        isSucc: a === succAct,
        room: st.freeN >= 2 ? 1 : 0
      };
      if (!best) {
        best = cand;
        continue;
      }
      if (cand.skipAfter !== best.skipAfter) {
        if (cand.skipAfter < best.skipAfter) {
          best = cand;
        }
        continue;
      }
      if (cand.room !== best.room) {
        if (cand.room > best.room) {
          best = cand;
        }
        continue;
      }
      if (cand.flood !== best.flood) {
        if (cand.flood > best.flood) {
          best = cand;
        }
        continue;
      }
      if (cand.isSucc && !best.isSucc) {
        best = cand;
      }
    }
    if (best) {
      return best.action;
    }
    return succAct !== null ? succAct :
      E.masterHamiltonAction(env, { healOnly: true });
  }

  /**
   * HealTS, but if the pick is a narrow edge cell with small flood, prefer the
   * max-flood safe alternative (corridor escape without interior bias).
   */
  function ctOddHealTSFloodEscStep(env) {
    var act = ctOddHealTSSafeStep(env);
    var safe = env.safeActions();
    if (!safe.length) {
      return act;
    }
    var n = env.size;
    var st = ctOddStepStats(env, act);
    var nextEdge = st.nr === 0 || st.nc === 0 ||
      st.nr === n - 1 || st.nc === n - 1;
    if (!(nextEdge && st.freeN <= 1 && st.flood < n && !st.ontoTip && !st.wins)) {
      return act;
    }
    var best = null;
    var i;
    for (i = 0; i < safe.length; i += 1) {
      var s2 = ctOddStepStats(env, safe[i]);
      if (s2.eating && s2.freeN < 1 && !s2.wins && !s2.ontoTip) {
        continue;
      }
      if (!s2.eating && s2.freeN < 1 && !s2.ontoTip) {
        continue;
      }
      var cand = {
        action: safe[i],
        flood: s2.flood,
        freeN: s2.freeN
      };
      if (!best || cand.flood > best.flood ||
          (cand.flood === best.flood && cand.freeN > best.freeN)) {
        best = cand;
      }
    }
    return best ? best.action : act;
  }

  /**
   * HealTS unless a freeN>=2 alternative exists with flood >= L-1; then prefer
   * roomy moves (soft corridor avoid).
   */
  function ctOddHealTSRoomStep(env) {
    var act = ctOddHealTSSafeStep(env);
    var safe = env.safeActions();
    if (!safe.length) {
      return act;
    }
    var st0 = ctOddStepStats(env, act);
    if (st0.freeN >= 2 || st0.ontoTip || st0.wins) {
      return act;
    }
    var need = Math.max(1, env.snake.length - 1);
    var best = null;
    var i;
    for (i = 0; i < safe.length; i += 1) {
      var s2 = ctOddStepStats(env, safe[i]);
      if (s2.eating && s2.freeN < 1 && !s2.wins && !s2.ontoTip) {
        continue;
      }
      if (s2.freeN < 2 && !s2.ontoTip) {
        continue;
      }
      if (s2.flood < need && !s2.ontoTip && !s2.wins) {
        continue;
      }
      var cand = {
        action: safe[i],
        flood: s2.flood,
        freeN: s2.freeN
      };
      if (!best || cand.flood > best.flood ||
          (cand.flood === best.flood && cand.freeN > best.freeN)) {
        best = cand;
      }
    }
    return best ? best.action : act;
  }

  /**
   * HealTS, but while skip debt is huge prefer higher freeN at equal flood.
   * Corridor forks pick freeN=2 over freeN=3 before freeN collapses; ungated
   * freeN greed livelocks packing, so only apply when skip > n*n.
   */
  function ctOddHealTSFreeNStep(env) {
    var act = ctOddHealTSSafeStep(env);
    var skip = ctOddSkipNow(env);
    var n = env.size;
    if (skip <= n * n) {
      return act;
    }
    var safe = env.safeActions();
    if (!safe.length) {
      return act;
    }
    var st0 = ctOddStepStats(env, act);
    if (st0.ontoTip || st0.wins) {
      return act;
    }
    var best = null;
    var i;
    for (i = 0; i < safe.length; i += 1) {
      var s2 = ctOddStepStats(env, safe[i]);
      if (s2.eating && s2.freeN < 1 && !s2.wins && !s2.ontoTip) {
        continue;
      }
      if (s2.freeN <= st0.freeN) {
        continue;
      }
      if (s2.flood < st0.flood) {
        continue;
      }
      var edge2 = s2.nr === 0 || s2.nc === 0 ||
        s2.nr === n - 1 || s2.nc === n - 1;
      var cand = {
        action: safe[i],
        freeN: s2.freeN,
        flood: s2.flood,
        edge: edge2 ? 1 : 0
      };
      if (!best) {
        best = cand;
        continue;
      }
      if (cand.freeN !== best.freeN) {
        if (cand.freeN > best.freeN) {
          best = cand;
        }
        continue;
      }
      if (cand.flood !== best.flood) {
        if (cand.flood > best.flood) {
          best = cand;
        }
        continue;
      }
      if (cand.edge < best.edge) {
        best = cand;
      }
    }
    return best ? best.action : act;
  }

  /**
   * Short debt-fill burst after heal entry, then HealTS. Aims to rejoin the
   * cycle before tip-chase walks an edge tube.
   */
  function ctOddDebtThenTSStep(env) {
    var n = env.size;
    if (env._ctOddHealSub !== "debtFirst") {
      env._ctOddHealSub = "debtFirst";
      env._ctOddHealStall = 0;
    }
    env._ctOddHealStall = (env._ctOddHealStall || 0) + 1;
    if (env._ctOddHealStall <= n * 3) {
      return ctOddDebtFillStep(env);
    }
    return ctOddHealTSFreeNStep(env);
  }

  /**
   * Tail-Safe Ham stretch heal (experiment):
   * 1) If cycleSucc is Tail-Safe (tip-reach + tip-vacate flood >= need), take it.
   * 2) Else among Tail-Safe moves, step toward the next open Ham stretch
   *    (lowest empty cycle index / consecutive empty run). No food seeking.
   * 3) If no length growth for 2*n*n steps, switch to healTS and reset the
   *    stall clock (one-way; stretch stalls were soft deaths in bakeoff).
   * 4) Caller exits to sticky Ham at skip === 0.
   */
  function ctOddTsHamStretchStep(env) {
    var n = env.size;
    var board = n * n;
    var len = env.snake.length;
    if (env._ctOddHealSub !== "stretch" && env._ctOddHealSub !== "healTS") {
      env._ctOddHealSub = "stretch";
      env._ctOddStretchLen = len;
      env._ctOddHealStall = 0;
    }
    if (env._ctOddStretchLen !== len) {
      env._ctOddStretchLen = len;
      env._ctOddHealStall = 0;
    } else {
      env._ctOddHealStall = (env._ctOddHealStall || 0) + 1;
    }
    if (env._ctOddHealSub === "healTS" ||
        env._ctOddHealStall >= 2 * board) {
      if (env._ctOddHealSub !== "healTS") {
        env._ctOddHealSub = "healTS";
        env._ctOddHealStall = 0;
      }
      return ctOddHealTSSafeStep(env);
    }

    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var cy = E.getHamiltonCycle(n);
    var h = env.snake[0];
    var tip = env.snake[env.snake.length - 1];
    var succAct = ctOddCycleSuccAct(env);
    var body = {};
    var bi;
    for (bi = 0; bi < env.snake.length; bi += 1) {
      body[env.snake[bi][0] + "," + env.snake[bi][1]] = true;
    }

    function moveTailSafe(act) {
      var nr = h[0] + DIRS[act][0];
      var nc = h[1] + DIRS[act][1];
      var eating = !!(env.food && env.food[0] === nr && env.food[1] === nc);
      var ontoTip = !!(tip && tip[0] === nr && tip[1] === nc);
      var freeN = E.freeNeighborsAfter(env, nr, nc, eating);
      var afterLen = env.snake.length + (eating ? 1 : 0);
      var wins = afterLen >= n * n ||
        (typeof env.targetLength === "number" && afterLen >= env.targetLength);
      if (eating && freeN < 1 && !ontoTip && !wins) {
        return null;
      }
      if (!eating && freeN < 1 && !ontoTip) {
        return null;
      }
      var stats = E.planCandidateFloodTipEnv(env, nr, nc, eating);
      if (!(stats.reachesTip || ontoTip)) {
        return null;
      }
      if (!wins && !ontoTip && stats.flood < stats.need) {
        return null;
      }
      return {
        action: act,
        nr: nr,
        nc: nc,
        flood: stats.flood,
        freeN: freeN,
        need: stats.need,
        isSucc: act === succAct
      };
    }

    if (succAct !== null) {
      var succOk = moveTailSafe(succAct);
      if (succOk) {
        return succAct;
      }
    }

    /* Next open stretch: first empty cycle cell at/after head+1 (wrap). */
    var goal = null;
    var hk = h[0] + "," + h[1];
    var startIdx = cy.index[hk];
    var from = typeof startIdx === "number" ? (startIdx + 1) % cy.len : 0;
    var scanned = 0;
    var si = from;
    while (scanned < cy.len) {
      var cell = cy.path[si];
      if (cell && !body[cell[0] + "," + cell[1]]) {
        goal = cell;
        break;
      }
      si = (si + 1) % cy.len;
      scanned += 1;
    }

    var best = null;
    var ai;
    for (ai = 0; ai < safe.length; ai += 1) {
      var cand = moveTailSafe(safe[ai]);
      if (!cand) {
        continue;
      }
      var man = goal
        ? Math.abs(goal[0] - cand.nr) + Math.abs(goal[1] - cand.nc)
        : 0;
      var onGoal = !!(goal && cand.nr === goal[0] && cand.nc === goal[1]);
      var row = {
        action: cand.action,
        man: man,
        onGoal: onGoal,
        flood: cand.flood,
        freeN: cand.freeN,
        isSucc: cand.isSucc
      };
      if (!best) {
        best = row;
        continue;
      }
      if (row.onGoal !== best.onGoal) {
        if (row.onGoal) {
          best = row;
        }
        continue;
      }
      if (row.man !== best.man) {
        if (row.man < best.man) {
          best = row;
        }
        continue;
      }
      if (row.flood !== best.flood) {
        if (row.flood > best.flood) {
          best = row;
        }
        continue;
      }
      if (row.freeN !== best.freeN) {
        if (row.freeN > best.freeN) {
          best = row;
        }
        continue;
      }
      if (row.isSucc && !best.isSucc) {
        best = row;
      }
    }
    if (best) {
      return best.action;
    }

    /* No Tip-Safe move: prefer tip-reaching max flood, else max flood. */
    var soft = null;
    for (ai = 0; ai < safe.length; ai += 1) {
      var nr2 = h[0] + DIRS[safe[ai]][0];
      var nc2 = h[1] + DIRS[safe[ai]][1];
      var eat2 = !!(env.food && env.food[0] === nr2 && env.food[1] === nc2);
      var tip2 = !!(tip && tip[0] === nr2 && tip[1] === nc2);
      var st2 = E.planCandidateFloodTipEnv(env, nr2, nc2, eat2);
      var fn2 = E.freeNeighborsAfter(env, nr2, nc2, eat2);
      if (eat2 && fn2 < 1 && !tip2) {
        continue;
      }
      var row2 = {
        action: safe[ai],
        reaches: !!(st2.reachesTip || tip2),
        flood: st2.flood,
        freeN: fn2
      };
      if (!soft) {
        soft = row2;
        continue;
      }
      if (row2.reaches !== soft.reaches) {
        if (row2.reaches) {
          soft = row2;
        }
        continue;
      }
      if (row2.flood !== soft.flood) {
        if (row2.flood > soft.flood) {
          soft = row2;
        }
        continue;
      }
      if (row2.freeN > soft.freeN) {
        soft = row2;
      }
    }
    return soft ? soft.action : safe[0];
  }

  /**
   * Cycle-arc fill: take succ when it drops skip; else step into largest debt
   * gap along the cycle; else debtFill; else HealTS.
   */
  function ctOddCycleArcFillStep(env) {
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    var cy = E.getHamiltonCycle(env.size);
    var skipNow = E.masterCycleSkip(cy, env.snake);
    var succAct = ctOddCycleSuccAct(env);
    if (succAct !== null) {
      var ss = ctOddStepStats(env, succAct);
      var skipSucc = E.masterSkipAfterMove(cy, env.snake, ss.nr, ss.nc, ss.eating);
      if (skipSucc < skipNow &&
          !(ss.eating && ss.freeN < 1 && !ss.wins && !ss.ontoTip)) {
        return succAct;
      }
    }
    var goalInfo = ctOddLargestGapGoal(env);
    if (goalInfo) {
      var goal = goalInfo.cell;
      var h = env.snake[0];
      var man0 = Math.abs(goal[0] - h[0]) + Math.abs(goal[1] - h[1]);
      var best = null;
      var i;
      for (i = 0; i < safe.length; i += 1) {
        var a = safe[i];
        var st = ctOddStepStats(env, a);
        if (st.eating && st.freeN < 1 && !st.wins && !st.ontoTip) {
          continue;
        }
        if (!st.eating && st.freeN < 1 && !st.ontoTip) {
          continue;
        }
        var skipAfter = E.masterSkipAfterMove(cy, env.snake, st.nr, st.nc, st.eating);
        var man1 = Math.abs(goal[0] - st.nr) + Math.abs(goal[1] - st.nc);
        var onGoal = st.nr === goal[0] && st.nc === goal[1];
        var cand = {
          action: a,
          onGoal: onGoal,
          closes: man1 < man0,
          skipAfter: skipAfter,
          flood: st.flood,
          freeN: st.freeN,
          isSucc: a === succAct
        };
        if (!best) {
          best = cand;
          continue;
        }
        if (cand.onGoal !== best.onGoal) {
          if (cand.onGoal) {
            best = cand;
          }
          continue;
        }
        if (cand.closes !== best.closes) {
          if (cand.closes) {
            best = cand;
          }
          continue;
        }
        if (cand.skipAfter !== best.skipAfter) {
          if (cand.skipAfter < best.skipAfter) {
            best = cand;
          }
          continue;
        }
        if (cand.flood !== best.flood) {
          if (cand.flood > best.flood) {
            best = cand;
          }
          continue;
        }
        if (cand.isSucc && !best.isSucc) {
          best = cand;
        }
      }
      if (best && (best.onGoal || best.closes || best.skipAfter < skipNow)) {
        return best.action;
      }
    }
    var debt = ctOddDebtFillStep(env);
    if (debt !== null && debt !== undefined) {
      return debt;
    }
    return ctOddHealTSSafeStep(env);
  }

  /**
   * After fill-T: CT-chase Hamilton indices in order (lowest open in-sub, then
   * next higher open in-sub, ...). Fringe Ham cells are OOB for targeting.
   * Stay CT in-sub via fringeCtToward / cellTreeEvenCore(opts.goal) - no food
   * swap, no forced non-CT steps. If mid-fringe-eat / bump script / head on
   * fringe: finish door-to-door via Odd fringe CT, then resume in-sub chase.
   * Exit HealTS when skip==0, no higher in-sub open index, CT cannot step, or
   * one goal stalls too long.
   */
  function ctOddCtHamChaseStep(env) {
    var n = env.size;
    var m = n - 1;
    var cy = E.getHamiltonCycle(n);
    var skip = E.masterCycleSkip(cy, env.snake);
    var h = env.snake[0];

    function handoffHealTS() {
      env._ctOddHealSub = "healTS";
      env._ctOddPostChaseHeal = true;
      env._ctOddHamChaseGoal = null;
      env._ctOddHamChaseStall = 0;
      return ctOddHealTSSafeStep(env);
    }

    if (skip === 0) {
      return handoffHealTS();
    }
    /* Mid-fringe-eat / pending bump: finish door-to-door, then resume chase. */
    var bumpPending = !!(env._ctBumpActions && env._ctBumpActions.length);
    if (bumpPending || !inTopLeftSub(h[0], h[1], m)) {
      try {
        env._ctOddLockVariant = true;
        env._ctOddLockExp = false;
        return cellTreeOddFringeBumpNextAction(env);
      } catch (e) {
        return handoffHealTS();
      }
    }

    var body = {};
    var i;
    for (i = 0; i < env.snake.length; i += 1) {
      body[env.snake[i][0] + "," + env.snake[i][1]] = true;
    }
    if (env._ctOddHamChaseGoal) {
      var cur = env._ctOddHamChaseGoal;
      if (h[0] === cur[0] && h[1] === cur[1]) {
        var curIdx = cy.index[cur[0] + "," + cur[1]];
        env._ctOddHamChaseFrom =
          typeof curIdx === "number" ? curIdx + 1 : (env._ctOddHamChaseFrom || 0) + 1;
        env._ctOddHamChaseGoal = null;
        env._ctOddHamChaseStall = 0;
      }
    }
    if (typeof env._ctOddHamChaseFrom !== "number") {
      env._ctOddHamChaseFrom = 0;
    }
    var goal = null;
    var from = env._ctOddHamChaseFrom;
    for (i = from; i < cy.len; i += 1) {
      var cell = cy.path[i];
      if (!cell) {
        continue;
      }
      /* Fringe = OOB for this chase - skip, keep scanning higher indices. */
      if (!inTopLeftSub(cell[0], cell[1], m)) {
        continue;
      }
      if (body[cell[0] + "," + cell[1]]) {
        continue;
      }
      goal = cell;
      env._ctOddHamChaseFrom = i;
      break;
    }
    if (!goal) {
      return handoffHealTS();
    }
    var prevGoal = env._ctOddHamChaseGoal;
    if (!prevGoal || prevGoal[0] !== goal[0] || prevGoal[1] !== goal[1]) {
      env._ctOddHamChaseStall = 0;
    }
    env._ctOddHamChaseGoal = goal;
    env._ctOddHamChaseStall = (env._ctOddHamChaseStall || 0) + 1;
    if (env._ctOddHamChaseStall > 4 * n) {
      return handoffHealTS();
    }

    var act = null;
    try {
      act = fringeCtToward(env, n, m, goal, null);
      if (act === null || act === undefined) {
        act = cellTreeEvenCore(env, n, { subM: m, goal: goal });
      }
    } catch (e) {
      act = null;
    }
    var safe = env.safeActions();
    if (act === null || act === undefined || safe.indexOf(act) < 0) {
      return handoffHealTS();
    }
    /* Refuse any step that leaves the sub (stay CT in-sub until heal). */
    var nr = h[0] + DIRS[act][0];
    var nc = h[1] + DIRS[act][1];
    if (!inTopLeftSub(nr, nc, m)) {
      return handoffHealTS();
    }
    return act;
  }

  /**
   * One heal step by style. Caller enters Ham when skip==0 (or style exit).
   */
  function ctOddHealStep(env, style) {
    var cy = E.getHamiltonCycle(env.size);
    var h = env.snake[0];
    var skip = E.masterCycleSkip(cy, env.snake);
    var cycleNext = cy.succ[h[0] + "," + h[1]];
    var act;
    if (style === "ctHamThenTS") {
      if (env._ctOddHealSub === "healTS") {
        return ctOddHealTSSafeStep(env);
      }
      return ctOddCtHamChaseStep(env);
    }
    if (style === "healTS") {
      return ctOddHealTSSafeStep(env);
    }
    if (style === "healTSRaw") {
      return E.healTailSafeNextAction(env);
    }
    if (style === "healTSInterior") {
      return ctOddHealTSInteriorStep(env);
    }
    if (style === "gapFill") {
      return ctOddGapFillStep(env);
    }
    if (style === "tsThenGap") {
      /* Tip-chase while skip is huge; fill largest gap once closer. */
      if (skip > env.size * env.size) {
        return ctOddHealTSSafeStep(env);
      }
      return ctOddGapFillStep(env);
    }
    if (style === "gapThenTS") {
      if (skip > env.size * env.size) {
        return ctOddGapFillStep(env);
      }
      return ctOddHealTSSafeStep(env);
    }
    if (style === "stallToGap") {
      return ctOddHealStallSwitchStep(env, "gap");
    }
    if (style === "stallToSucc") {
      return ctOddHealStallSwitchStep(env, "succ");
    }
    if (style === "stallToDebt") {
      return ctOddHealStallSwitchStep(env, "debt");
    }
    if (style === "debtFill") {
      return ctOddDebtFillStep(env);
    }
    if (style === "earliestFill") {
      return ctOddEarliestFillStep(env);
    }
    if (style === "earliestThenTS") {
      return ctOddEarliestThenTSStep(env);
    }
    if (style === "earliestBurst1L") {
      return ctOddEarliestBurstThenTSStep(env, "earliestBurst1L", function (e) {
        return e.snake.length;
      });
    }
    if (style === "earliestBurst2L") {
      return ctOddEarliestBurstThenTSStep(env, "earliestBurst2L", function (e) {
        return 2 * e.snake.length;
      });
    }
    if (style === "earliestBurst3L") {
      return ctOddEarliestBurstThenTSStep(env, "earliestBurst3L", function (e) {
        return 3 * e.snake.length;
      });
    }
    if (style === "earliestBurstN") {
      return ctOddEarliestBurstThenTSStep(env, "earliestBurstN", function (e) {
        return e.size;
      });
    }
    if (style === "earliestBurstHalfN") {
      return ctOddEarliestBurstThenTSStep(env, "earliestBurstHalfN", function (e) {
        return Math.max(1, Math.floor(e.size / 2));
      });
    }
    if (style === "earliestBurst4") {
      return ctOddEarliestBurstThenTSStep(env, "earliestBurst4", function () {
        return 4;
      });
    }
    if (style === "earliestBurst8") {
      return ctOddEarliestBurstThenTSStep(env, "earliestBurst8", function () {
        return 8;
      });
    }
    if (style === "tsEarliestLand") {
      return ctOddTSEarliestLandStep(env);
    }
    if (style === "onCycleEarliest") {
      return ctOddOnCycleEarliestElseTSStep(env);
    }
    if (style === "earliestBfs") {
      return ctOddEarliestBfsStep(env);
    }
    if (style === "tsHamStretch") {
      return ctOddTsHamStretchStep(env);
    }
    if (style === "cycleArcFill") {
      return ctOddCycleArcFillStep(env);
    }
    if (style === "healTSFloodEsc") {
      return ctOddHealTSFloodEscStep(env);
    }
    if (style === "healTSRoom") {
      return ctOddHealTSRoomStep(env);
    }
    if (style === "healTSFreeN") {
      return ctOddHealTSFreeNStep(env);
    }
    if (style === "debtThenTS") {
      return ctOddDebtThenTSStep(env);
    }
    if (style === "tsThenDebt") {
      if (skip > env.size * env.size) {
        return ctOddHealTSFreeNStep(env);
      }
      return ctOddDebtFillStep(env);
    }
    if (style === "healLA") {
      return E.healLookaheadNextAction(env);
    }
    if (style === "skipMin") {
      act = ctOddSkipMinStep(env, { allowEat: true, maxRaise: env.size });
      return act !== null ? act : E.masterHamiltonAction(env, { healOnly: true });
    }
    if (style === "skipMinNoEat") {
      act = ctOddSkipMinStep(env, { allowEat: false, maxRaise: env.size });
      return act !== null ? act : E.masterHamiltonAction(env, { healOnly: true });
    }
    if (style === "skipMinStrict") {
      act = ctOddSkipMinStep(env, { allowEat: true, maxRaise: 0 });
      return act !== null ? act : E.masterHamiltonAction(env, { healOnly: true });
    }
    if (style === "skipThenSucc") {
      if (skip > env.size) {
        act = ctOddSkipMinStep(env, { allowEat: false, maxRaise: env.size });
        if (act !== null) {
          return act;
        }
      }
      return E.masterHamiltonAction(env, { healOnly: true });
    }
    if (style === "healOnlyNoEat") {
      var safe = env.safeActions();
      if (cycleNext) {
        var sa = null;
        var di;
        for (di = 0; di < 4; di += 1) {
          if (h[0] + DIRS[di][0] === cycleNext[0] &&
              h[1] + DIRS[di][1] === cycleNext[1]) {
            sa = di;
            break;
          }
        }
        if (sa !== null && safe.indexOf(sa) >= 0) {
          return sa;
        }
      }
      act = ctOddSkipMinStep(env, { allowEat: false, maxRaise: 1e9 });
      return act !== null ? act : E.masterHamiltonAction(env, { healOnly: true });
    }
    /* healOnly / healSoft / stickyHam (caller skips heal). */
    return E.masterHamiltonAction(env, { healOnly: true });
  }

  var CELL_TREE_ODD_HEAL_STYLES = [
    "stickyHam",
    "healOnly",
    "healSoft",
    "healOnlyNoEat",
    "skipMin",
    "skipMinNoEat",
    "skipMinStrict",
    "skipThenSucc",
    "healTS",
    "ctHamThenTS",
    "healTSRaw",
    "healTSInterior",
    "healLA",
    "gapFill",
    "tsThenGap",
    "gapThenTS",
    "stallToGap",
    "stallToSucc",
    "stallToDebt",
    "debtFill",
    "earliestFill",
    "earliestThenTS",
    "earliestBurst1L",
    "earliestBurst2L",
    "earliestBurst3L",
    "earliestBurstN",
    "earliestBurstHalfN",
    "earliestBurst4",
    "earliestBurst8",
    "tsEarliestLand",
    "onCycleEarliest",
    "earliestBfs",
    "cycleArcFill",
    "tsHamStretch",
    "healTSFloodEsc",
    "healTSRoom",
    "healTSFreeN",
    "debtThenTS",
    "tsThenDebt"
  ];

  function ctOddHealExitReady(env, style) {
    var skip = ctOddSkipNow(env);
    if (style === "healSoft") {
      return skip <= env.size;
    }
    return skip === 0;
  }

  /**
   * Odd: Cell Tree Odd until fill >= T, then Heal (style) until packed, then
   * sticky Hamilton. Trigger is fill only (not fringe food).
   * opts.fillT / opts.healStyle or env._ctOddHamFillT / env._ctOddHealStyle.
   */
  function cellTreeOddFillHandoffNextAction(env, fillT, healStyle) {
    var n = env.size;
    if (n < 2 || !env.food) {
      if (n & 1) {
        return cellTreeOddNoHamFallback(env);
      }
      return E.masterHamiltonAction(env);
    }
    if (!(n & 1)) {
      var actEven = cellTreeEvenCore(env, n, {});
      if (actEven === null) {
        return E.masterHamiltonAction(env);
      }
      return actEven;
    }
    if (typeof env._ctOddLastSteps === "number" && env.steps < env._ctOddLastSteps) {
      env._ctOddHamLock = false;
      env._ctOddPhase = null;
    }
    env._ctOddLastSteps = env.steps;

    var T = typeof fillT === "number" ? fillT :
      (typeof env._ctOddHamFillT === "number" ? env._ctOddHamFillT :
        CELL_TREE_ODD_HAM_FILL_T);
    var style = healStyle || env._ctOddHealStyle || CELL_TREE_ODD_HEAL_STYLE;
    var fill = env.snake.length / (n * n);
    var phase = env._ctOddPhase;

    if (!phase) {
      phase = fill >= T ? (style === "stickyHam" ? "ham" : "heal") : "odd";
      env._ctOddPhase = phase;
    }
    if (phase === "odd" && fill >= T) {
      phase = style === "stickyHam" ? "ham" : "heal";
      env._ctOddPhase = phase;
      if (phase === "ham") {
        env._ctOddHamLock = true;
      }
    }
    if (phase === "heal") {
      if (style === "stickyHam" || ctOddHealExitReady(env, style)) {
        env._ctOddPhase = "ham";
        env._ctOddHamLock = true;
        return E.masterHamiltonAction(env);
      }
      return ctOddHealStep(env, style);
    }
    if (phase === "ham" || env._ctOddHamLock) {
      env._ctOddHamLock = true;
      env._ctOddPhase = "ham";
      return E.masterHamiltonAction(env);
    }
    env._ctOddLockVariant = true;
    env._ctOddLockExp = false;
    return cellTreeOddFringeBumpNextAction(env);
  }

  /**
   * Cell Tree (twanvl). Even: full board. Odd: Cell Tree Odd until fill T,
   * then heal -> sticky Hamilton.
   */
  function cellTreeNextAction(env) {
    var n = env.size;
    if (n < 2 || !env.food) {
      return E.masterHamiltonAction(env);
    }
    if (n & 1) {
      return cellTreeOddFillHandoffNextAction(env);
    }
    var act = cellTreeEvenCore(env, n, {});
    if (act === null) {
      return E.masterHamiltonAction(env);
    }
    return act;
  }

  /**
   * Cell Tree Odd (Observe): even = full-board Cell Tree; odd = Cell Tree on
   * (n-1)^2 + fringe splices, fringe-pair locks, late hole weave (CT escape).
   * H-behind-T grow-before-food; late corner-food rim growth. Temp CT breaks
   * only with a return plan. No Hamilton on odd. No free-cruise late finish.
   * Local/cheap on large boards (e.g. 99x99).
   */
  function cellTreeOddNextAction(env) {
    var n = env.size;
    if (n < 2 || !env.food) {
      if (n & 1) {
        return cellTreeOddNoHamFallback(env);
      }
      return E.masterHamiltonAction(env);
    }
    if (n & 1) {
      env._ctOddLockVariant = true;
      env._ctOddLockExp = false;
      return cellTreeOddFringeBumpNextAction(env);
    }
    var act = cellTreeEvenCore(env, n, {});
    if (act === null) {
      return E.masterHamiltonAction(env);
    }
    return act;
  }

  /** @deprecated alias - use cellTreeOddNextAction */
  function cellTreeFringeBumpNextAction(env) {
    return cellTreeOddNextAction(env);
  }

  /* ---------- DHCR ---------- */

  function cycleFromHamilton(n) {
    var cy = E.getHamiltonCycle(n);
    var next = new Int32Array(n * n);
    var i;
    for (i = 0; i < next.length; i += 1) {
      next[i] = -1;
    }
    var k;
    for (k in cy.succ) {
      if (!Object.prototype.hasOwnProperty.call(cy.succ, k)) {
        continue;
      }
      var parts = k.split(",");
      var r = parts[0] | 0;
      var c = parts[1] | 0;
      var t = cy.succ[k];
      next[r * n + c] = t[0] * n + t[1];
    }
    return next;
  }

  function isNeighborIdx(n, a, b) {
    var ar = (a / n) | 0;
    var ac = a - ar * n;
    var br = (b / n) | 0;
    var bc = b - br * n;
    return Math.abs(ar - br) + Math.abs(ac - bc) === 1;
  }

  function pathFrom(next, d) {
    var i;
    for (i = 0; i < next.length; i += 1) {
      if (next[i] === d) {
        return i;
      }
    }
    return -1;
  }

  function isHamiltonianCycle(next, n) {
    var N = n * n;
    if (next.length !== N) {
      return false;
    }
    var pos = 0;
    var seen = new Uint8Array(N);
    var i;
    for (i = 0; i < N; i += 1) {
      if (seen[pos]) {
        return false;
      }
      var nx = next[pos];
      if (nx < 0 || nx >= N || !isNeighborIdx(n, pos, nx)) {
        return false;
      }
      seen[pos] = 1;
      pos = nx;
    }
    return pos === 0;
  }

  function rotateClockwise(dir) {
    return (dir + 1) & 3;
  }

  function rotateCounterClockwise(dir) {
    return (dir + 3) & 3;
  }

  function dirBetween(n, a, b) {
    var ar = (a / n) | 0;
    var ac = a - ar * n;
    var br = (b / n) | 0;
    var bc = b - br * n;
    return actionFromTo(ar, ac, br, bc);
  }

  /**
   * Port of twanvl repair_cycle. Mutates next. Returns success.
   * Occupied: body except tip (u,v must be empty).
   */
  function repairCycle(n, occ, next, a, d) {
    if (!isNeighborIdx(n, a, d)) {
      return false;
    }
    if (next[a] === d) {
      return true;
    }
    var b = next[a];
    var c = pathFrom(next, d);
    if (b < 0 || c < 0) {
      return false;
    }
    if (!isNeighborIdx(n, b, c)) {
      return false;
    }
    var N = n * n;
    var cycle1 = new Uint8Array(N);
    var x;
    for (x = d; x !== a; x = next[x]) {
      if (cycle1[x]) {
        return false;
      }
      cycle1[x] = 1;
      if (x < 0) {
        return false;
      }
    }
    cycle1[a] = 1;
    for (x = b; x !== c; x = next[x]) {
      var y = next[x];
      var xy = dirBetween(n, x, y);
      if (xy === null) {
        continue;
      }
      var turns = [rotateClockwise(xy), rotateCounterClockwise(xy)];
      var ti;
      for (ti = 0; ti < 2; ti += 1) {
        var dir = turns[ti];
        var yr = (y / n) | 0;
        var yc = y - yr * n;
        var xr = (x / n) | 0;
        var xc = x - xr * n;
        var ur = yr + DIRS[dir][0];
        var uc = yc + DIRS[dir][1];
        var vr = xr + DIRS[dir][0];
        var vc = xc + DIRS[dir][1];
        if (ur < 0 || uc < 0 || ur >= n || uc >= n) {
          continue;
        }
        if (vr < 0 || vc < 0 || vr >= n || vc >= n) {
          continue;
        }
        var u = ur * n + uc;
        var v = vr * n + vc;
        if (cycle1[u] && cycle1[v] && next[u] === v && !occ[u] && !occ[v]) {
          next[a] = d;
          next[c] = b;
          next[x] = v;
          next[u] = y;
          return isHamiltonianCycle(next, n);
        }
      }
    }
    return false;
  }

  function cycleDistancesToGoal(next, n, goal) {
    var N = n * n;
    var dists = new Int32Array(N);
    var c = goal;
    var dist = N - 1;
    var guard = 0;
    do {
      c = next[c];
      dists[c] = dist;
      dist -= 1;
      guard += 1;
    } while (c !== goal && guard <= N + 2);
    return dists;
  }

  function ensureDhcr(env) {
    var n = env.size;
    if (env._dhcrN !== n || !env._dhcrNext) {
      env._dhcrNext = cycleFromHamilton(n);
      env._dhcrN = n;
      env._dhcrLastSteps = env.steps;
      return;
    }
    /* Episode reset: steps went backwards. */
    if (typeof env._dhcrLastSteps === "number" && env.steps < env._dhcrLastSteps) {
      env._dhcrNext = cycleFromHamilton(n);
    }
    env._dhcrLastSteps = env.steps;
  }

  function dhcrPoseKey(env) {
    var i;
    var parts = [];
    for (i = 0; i < env.snake.length; i += 1) {
      parts.push(env.snake[i][0] + "," + env.snake[i][1]);
    }
    var f = env.food;
    return parts.join(";") + "|f=" + (f ? f[0] + "," + f[1] : "-");
  }

  /** Lock to production Ham until length grows (escape repaired-cycle livelock). */
  function dhcrEnterSafeHam(env, n) {
    env._dhcrSafeHam = true;
    env._dhcrSafeHamLen = env.snake.length;
    env._dhcrNext = cycleFromHamilton(n);
    env._dhcrPoses = Object.create(null);
    env._dhcrStarve = 0;
    return E.masterHamiltonAction(env);
  }

  /**
   * Core DHCR step (twanvl-style repair + follow). No SafeHam lock.
   * opts.improved: pose/starve livelock -> lock prod Ham until next food.
   */
  function dhcrCoreNextAction(env, opts) {
    opts = opts || {};
    var improved = !!opts.improved;
    var n = env.size;
    if ((n & 1) || n < 2 || !env.food) {
      return E.masterHamiltonAction(env);
    }
    var safe = env.safeActions();
    if (!safe.length) {
      return env.facing | 0;
    }
    if (improved) {
      if (env._dhcrSafeHam) {
        if (env.snake.length > (env._dhcrSafeHamLen || 0)) {
          env._dhcrSafeHam = false;
          env._dhcrNext = cycleFromHamilton(n);
          env._dhcrPoses = Object.create(null);
          env._dhcrStarve = 0;
        } else {
          return E.masterHamiltonAction(env);
        }
      }
      if (env._dhcrTrackLen !== env.snake.length) {
        env._dhcrTrackLen = env.snake.length;
        env._dhcrStarve = 0;
        env._dhcrPoses = Object.create(null);
      } else {
        env._dhcrStarve = (env._dhcrStarve || 0) + 1;
      }
      if (!env._dhcrPoses) {
        env._dhcrPoses = Object.create(null);
      }
      var pk = dhcrPoseKey(env);
      if (env._dhcrPoses[pk] || env._dhcrStarve > n * n) {
        return dhcrEnterSafeHam(env, n);
      }
      env._dhcrPoses[pk] = true;
    }

    ensureDhcr(env);
    var next = env._dhcrNext;
    if (!isHamiltonianCycle(next, n)) {
      env._dhcrNext = cycleFromHamilton(n);
      next = env._dhcrNext;
    }
    var h = env.snake[0];
    var food = env.food;
    var pos = h[0] * n + h[1];
    var goal = food[0] * n + food[1];
    var occ = buildBodyOcc(env);
    /* Mark all body including tip for repair emptiness (twanvl grid). */
    var occRepair = new Uint8Array(n * n);
    var si;
    for (si = 0; si < env.snake.length; si += 1) {
      occRepair[env.snake[si][0] * n + env.snake[si][1]] = 1;
    }

    var cycDist = cycleDistancesToGoal(next, n, goal);
    var edgeFn = function (r, c, nr, nc) {
      var ni = nr * n + nc;
      if (occ[ni]) {
        return INF;
      }
      return 1000000 + cycDist[ni];
    };
    var sp = dijkstraTo(n, h[0], h[1], food[0], food[1], edgeFn);
    var path = sp.found ? readPath(n, sp.prev, h[0], h[1], food[0], food[1]) : [];
    var target = next[pos];
    if (path.length) {
      var step = path[path.length - 1];
      target = step[0] * n + step[1];
    }

    if (next[pos] !== target) {
      var backup = new Int32Array(next);
      if (!repairCycle(n, occRepair, next, pos, target)) {
        env._dhcrNext = backup;
        next = backup;
      } else if (!isHamiltonianCycle(next, n)) {
        env._dhcrNext = backup;
        next = backup;
      } else {
        env._dhcrNext = next;
      }
    }

    var follow = next[pos];
    if (follow < 0) {
      if (improved) {
        return dhcrEnterSafeHam(env, n);
      }
      env._dhcrNext = cycleFromHamilton(n);
      return E.masterHamiltonAction(env);
    }
    var fr = (follow / n) | 0;
    var fc = follow - fr * n;
    var act = actionFromTo(h[0], h[1], fr, fc);
    if (act === null || env.isImmediateDeath(act) || safe.indexOf(act) < 0) {
      if (improved) {
        return dhcrEnterSafeHam(env, n);
      }
      env._dhcrNext = cycleFromHamilton(n);
      return E.masterHamiltonAction(env);
    }
    return act;
  }

  /** Original-style DHCR (no SafeHam livelock lock). */
  function dhcrNextAction(env) {
    return dhcrCoreNextAction(env, { improved: false });
  }

  /** DHCR Improved: SafeHam on pose-repeat / starve until next food. */
  function dhcrImprovedNextAction(env) {
    return dhcrCoreNextAction(env, { improved: true });
  }

  E.cellTreeNextAction = cellTreeNextAction;
  E.cellTreeOddNextAction = cellTreeOddNextAction;
  E.cellTreeFringeBumpNextAction = cellTreeFringeBumpNextAction;
  E.cellTreeOddVariantNextAction = cellTreeOddVariantNextAction;
  E.cellTreeOddFillHandoffNextAction = cellTreeOddFillHandoffNextAction;
  E.CELL_TREE_ODD_HAM_FILL_T = CELL_TREE_ODD_HAM_FILL_T;
  E.CELL_TREE_ODD_HEAL_STYLE = CELL_TREE_ODD_HEAL_STYLE;
  E.CELL_TREE_ODD_HEAL_STYLES = CELL_TREE_ODD_HEAL_STYLES;
  E.dhcrNextAction = dhcrNextAction;
  E.dhcrImprovedNextAction = dhcrImprovedNextAction;
  E._publishedRepairCycle = repairCycle;
  E._publishedIsHamiltonianCycle = isHamiltonianCycle;
  E._publishedCycleFromHamilton = cycleFromHamilton;

  /* Silence unused in some bundlers */
  void ACTION_NAMES;
})(typeof window !== "undefined" ? window : global);
