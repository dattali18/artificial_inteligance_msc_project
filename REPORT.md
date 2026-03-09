## Project Report: AI Routing on OpenStreetMap

- Daniel Attali
- Sapir Bashan

### 1. Introduction and Problem Definition

The objective of this project is to calculate the optimal route between a source address and a destination address using real-world geographic data from OpenStreetMap (OSM). The problem is modeled as a shortest-path search on a directed graph representing the street network.

### 2. Algorithms Implemented

To solve the routing problem, three distinct algorithmic approaches were implemented and compared:

* **Search-Based Algorithm (A\*):** A deterministic, informed search algorithm utilizing the great-circle distance as an admissible heuristic.
* **Biology-Inspired Algorithm (Ant Colony Optimization - ACO):** A probabilistic algorithm simulating the pheromone-laying behavior of ants. The implementation utilizes a custom heuristic (inverse distance to goal) to guide the ants.
* **Reinforcement Learning (Q-Learning):** An episodic learning agent utilizing an epsilon-greedy policy. To mitigate sparse rewards on the massive OSM graph, reward shaping was applied to penalize loops and reward geographic progression toward the target.

### 3. Experimental Setup and Evaluation

The algorithms were evaluated across 100 randomly generated source-destination pairs on the local OSM map. Each algorithm was restricted to a practical computational budget (e.g., limited ants/iterations for ACO, 100 episodes for Q-Learning) to assess efficiency.

**Table 1: Performance Comparison (Averaged over 100 runs)**

| Algorithm | Avg. Path Cost (Distance) | Avg. Execution Time (s) | Avg. Nodes Expanded | Success Rate |
| --- | --- | --- | --- | --- |
| **A*** | 1283.98 | 0.0009 | 142.76 | 100% |
| **ACO** | $\infty$ (Did not converge) | 0.0034 | 793.32 | 0% |
| **Q-Learning** | $\infty$ (Did not converge) | 0.0798 | 43223.35 | 0% |

![visaluzation](images/a_star.png)

### 4. Verbal Analysis of Results

The evaluation highlights a stark contrast between deterministic search and probabilistic learning models on static, real-world spatial graphs:

* **A\* Performance:** A\* drastically outperformed the other methods in every metric. By utilizing an admissible heuristic, it efficiently expanded an average of only ~143 nodes, finding the optimal path in less than a millisecond.
* **ACO and Q-Learning Limitations:** Both ACO and Q-Learning struggled to find valid paths within the allocated computational budget, frequently returning infinite costs. The high branching factor and the presence of dead-ends (cul-de-sacs) in real urban environments trap the exploration phases of both algorithms.

You can look at the full report at `algorithm_comparision_results.csv` with all the details

### 5. Critique, Challenges, and Suggestions for Improvement

**The Challenge of State Space:** The primary challenge encountered was applying learning-based algorithms to a massive deterministic graph. While Q-learning and ACO excel in stochastic environments (e.g., dynamic traffic, changing edge weights) or constraint-satisfaction problems, they are highly inefficient for standard point-to-point routing compared to Dijkstra or A*.

**Sparse Rewards in RL:** In the initial Q-learning implementation, the agent suffered from severe sparse rewards, unable to randomly stumble upon a specific target node among thousands. While dense reward shaping (rewarding geographic proximity) improved exploration, it often led the agent into local minima where the shortest geographic distance was blocked by a physical barrier (like a highway or missing bridge), trapping the agent.

**Suggestions for Improvement:**

1. **State-Space Abstraction:** Instead of navigating node-by-node, the graph could be simplified into major intersections or hierarchical road networks to reduce the state space for the RL agent.
2. **Hybrid Approach:** Using A\* to generate an initial baseline route, and then using RL or ACO to optimize that specific corridor for dynamic variables (like simulated traffic or safety constraints).

### 6. AI Tools Used

We used AI tools like GitHub Copilot and Google Gmini To help us understand how to use the OSM tool, and also we use them to write simple and readable solution, and also interpret and write the values from the solutions 
