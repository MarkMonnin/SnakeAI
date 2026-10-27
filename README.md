# Snake AI Shape Search

Browser lab for Snake-playing agents. The project lets you **train** neural and tabular AIs to climb a board-size ladder, but Snake turned out hard enough that **clever algorithms are the practical way to go**. Observe them race (including AIs in training), or keep searching for architectures that learn.

Open [`index.html`](index.html) in a modern browser (File > Open). No build step, no server, no install. Progress auto-saves in the browser (IndexedDB) while you use Shape Search or Train One.

## Tabs

| Tab | Purpose |
| --- | --- |
| **Observe** | Watch baselines and trainees play (Master, Hamilton, Cell Tree, DHCR, Tail Safe, Search/Train bots, and more). Most entertaining place to start. |
| **Shape Search** | Population of learners climbs 4x4 ... 100x100 with shared-seed tournaments. |
| **Train One** | Pull one bot out of Search and climb the ladder alone. |
| **About** | Credits and short algorithm notes. |

## Credits (algorithm origins)

These Observe solvers build on published ideas. Implementations here are ports and variants; please credit the originals.

| Algorithm | Origin |
| --- | --- |
| **Hamilton** | Classic "follow a Hamiltonian cycle on the grid," with controlled shortcuts. Widely popularized for Snake by [John Flux](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/) (Nokia 6110 series) and later demos (e.g. Code Bullet). The cycle + free-arc / skip gating here is in that tradition. |
| **Cell Tree** | [twanvl/snake](https://github.com/twanvl/snake/) - force right-hand drive in every 2x2 cell; the spanning tree of cells implies a Hamiltonian tour. Ported in `snake_js/published-agents.js`. |
| **DHCR** | Dynamic Hamiltonian Cycle Repair - [Brian Haidet / AlphaPhoenix](https://github.com/BrianHaidet/AlphaPhoenix/tree/master/Snake_AI_(2020a)_DHCR_with_strategy) (original MATLAB). Also discussed and reimplemented in [twanvl/snake](https://github.com/twanvl/snake/). Includes original-style DHCR and **DHCR Improved** (SafeHam loop escape). |
| **Tail Safe** | Tip-vacate floods + largest-opening ranking. Closely related published theme: ensure a path to the tip after eating ([chuyangliu/snake](https://github.com/chuyangliu/snake/blob/master/docs/algorithms.md) Path Solver). |

Other Observe bots (BFS, Greedy, Wall follower, Avoid Pocket, Plan Ahead, Master hybrids) are classical search heuristics or compositions in the lab.

## How the algorithms work (in short)

### Master
Picks the fastest full-safe algorithm by board parity: **Cell Tree** on even sizes, **Tail Safe Hybrid** on odd sizes. It is the default Observe bot and topped the 4..11 x100 tournament (0 deaths, fewest steps among perfect finishers).

### Cell Tree
Keeps the snake driving clockwise inside every 2x2 cell and only crosses between cells along a spanning tree, which implies a tour of the board. Food is chased under those constraints. On **even** boards that is the full grid. On **odd** boards it stays on Cell Tree Odd until fill reaches about 36%, then transitions to Hamilton.

### Cell Tree Odd
Dedicated no-Hamilton odd algorithm (Cell Tree hybrid listed above falls back to Hamilton for odd-size boards). Even boards use full-board Cell Tree. About a 54% win rate on 11x11 boards.

### DHCR / DHCR Improved
Starts from a Hamiltonian cycle and takes shortcuts only when the cycle can be **rewritten** so it stays Hamiltonian (repair, not a fixed tour). Improved handles when the repaired cycle loops (state repeat / starve), then returns to repair after the next apple.

### Hamilton / Hamilton Aggressive
Builds a fixed Hamiltonian cycle (column zigzag on even boards; odd omit-corner splice on odd) and usually follows the next cycle cell. Safe shortcuts (chords) are allowed when free-arc / skip cost says the snake will not strand itself; Aggressive takes more shortcuts early (under ~20% fill) then matches normal Hamilton.

### Tail Safe / Tail Safe Hybrid
Tail Safe ranks moves by tail-reachability and open space (treating the tip cell as vacated on the next non-eat), chasing food only when the tip stays reachable afterward. Tail Safe Hybrid runs Tail Safe until a loop or ~20% fill, then transitions to Hamilton.

### Plan Ahead
Builds a short forward path that stays in the largest empty opening, then enters Hamilton transition with a lookahead policy when stuck. It re-enters planning and does not hand off to Hamilton, though late-game can look similar to Hamilton because the transition method matches that.

### Avoid Pocket / BFS / Greedy / Wall follower / Random
Avoid Pocket prefers BFS-to-food steps that leave enough flood space (no one-cell traps). BFS takes a shortest path to food in the visible/open grid; Greedy steps toward food by Manhattan; Wall follower hugs walls with a greedy fallback; Random picks a legal move.

### Learners (Shape Search / Train One)
Tabular (Q, SARSA, MENACE), backprop policy/value, NEAT, and TensorFlow.js DQN train from local vision plus an optional food compass. They are the original point of the lab; finishing large boards still favors the algorithmic baselines above.

## Lab rules (short)

- Walls are solid; start length 3 near center facing North.
- Observe can fill to a chosen **Win at** percent (up to 100%).
- Shape Search beat: length `ceil(0.25 * N * N)` with success rate **> 90%** for **10** consecutive evals.
- Deaths: wall or body. Rewards: food `+1`, death `-20`, light approach/away shaping.

## Run / test

```bash
# Open in a browser
#   index.html

# Node unit tests (optional)
node tests/test-snake-env.js
node tests/test-shape-lab.js
```

TF.js is vendored at `vendor/tf.min.js` (no CDN required).

## Layout

```text
index.html
README.md
snake-shapesearch.css
snake_js/          env, agents, Shape Search, Observe, published ports
vendor/tf.min.js
tests/             unit tests + pose fixtures
```

## License / attribution

Code in this repository is for the Snake AI Shape Search lab. Published-algorithm ports credit their authors above; see their repositories for upstream licenses (twanvl/snake, AlphaPhoenix DHCR, and classic Ham writeups).
