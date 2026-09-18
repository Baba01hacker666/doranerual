"""Zexo Conversational Dataset: Logic, Algorithms, Systems & Advanced Reasoning (Dialogues 59-68)."""

DIALOGUES_REASONING = [
    # Dialogue 59: Deductive Logic & Fallacies
    [
        ("Can you explain the difference between Modus Ponens, Modus Tollens, and the fallacy of Affirming the Consequent?",
         "These are core deductive argument forms in propositional logic, operating on a conditional premise 'If P, then Q' (P -> Q):\n\n"
         "1. Modus Ponens (Affirming the Antecedent - Valid):\n"
         "- Premise 1: P -> Q\n"
         "- Premise 2: P\n"
         "- Conclusion: Therefore, Q.\n"
         "Example: 'If it rains, the ground is wet. It is raining. Therefore, the ground is wet.'\n\n"
         "2. Modus Tollens (Denying the Consequent - Valid):\n"
         "- Premise 1: P -> Q\n"
         "- Premise 2: not Q\n"
         "- Conclusion: Therefore, not P.\n"
         "Example: 'If it rains, the ground is wet. The ground is not wet. Therefore, it is not raining.'\n\n"
         "3. Fallacy of Affirming the Consequent (Invalid):\n"
         "- Premise 1: P -> Q\n"
         "- Premise 2: Q\n"
         "- Fallacious Conclusion: Therefore, P.\n"
         "Example: 'If it rains, the ground is wet. The ground is wet. Therefore, it rained.' (Invalid because a sprinkler or burst pipe could have caused Q)."),
        ("How does understanding these logical structures help in debugging complex software bugs?",
         "Software bugs often arise when engineers commit affirming the consequent during troubleshooting. For example:\n"
         "'If the database crashes (P), the API returns HTTP 500 (Q). The API returned HTTP 500 (Q), so the database must have crashed (P).'\n"
         "A rigorous engineer applies Modus Tollens instead:\n"
         "'If the database crashed, we would see connection timeout logs. We do not see connection timeouts. Therefore, the database did not crash.'\n"
         "This isolates root causes methodically, ruling out hypotheses via contrapositive evidence rather than jumping to invalid conclusions.")
    ],

    # Dialogue 60: B+ Trees vs LSM-Trees
    [
        ("What are the key trade-offs between B+ Trees and Log-Structured Merge (LSM) Trees in modern storage engines?",
         "B+ Trees and LSM-Trees represent two distinct philosophies for managing persistent data on disk, navigating trade-offs between read amplification, write amplification, and space amplification:\n\n"
         "1. B+ Trees (e.g., PostgreSQL, MySQL InnoDB, SQLite):\n"
         "- Architecture: Balanced search tree where all data resides in leaf pages linked sequentially. Internal nodes store keys for routing.\n"
         "- Update Strategy: In-place updates on fixed-size disk blocks (e.g., 8KB or 16KB pages).\n"
         "- Read Performance: High. Point lookups and range scans take O(log_B N) with minimal random reads.\n"
         "- Write Performance: Slower. In-place updates require random disk writes and Write-Ahead Logging (WAL) to ensure crash recovery, causing high write amplification.\n\n"
         "2. LSM-Trees (e.g., RocksDB, Cassandra, ClickHouse, ScyllaDB):\n"
         "- Architecture: Multi-tiered. Writes append sequentially to an in-memory sorted buffer (MemTable) and a sequential commit log. When full, the MemTable flushes to disk as an immutable Sorted String Table (SSTable). Background threads periodically merge and deduplicate SSTables (Compaction).\n"
         "- Write Performance: Blazing fast. All disk writes are purely sequential, maximizing SSD throughput.\n"
         "- Read Performance: Slower. Point lookups may check the MemTable and multiple SSTable levels, mitigated using Bloom filters."),
        ("Under what workloads should an architect definitively choose an LSM-Tree over a B+ Tree?",
         "Choose an LSM-Tree when write throughput dominates read volume—such as time-series metrics, IoT sensor telemetry, financial audit logging, and messaging platforms. The sequential I/O allows LSM-trees to sustain 10x to 100x higher write throughput than B+ trees on identical hardware, while Bloom filters keep single-key point lookups efficient.")
    ],

    # Dialogue 61: Distributed Systems - The Raft Consensus Protocol
    [
        ("How does the Raft consensus algorithm guarantee state machine replication across a distributed cluster?",
         "Raft decomposes distributed consensus into three well-defined subproblems:\n\n"
         "1. Leader Election: Nodes start as Followers. If a follower misses heartbeats from a Leader within a randomized election timeout (e.g., 150-300ms), it transitions to Candidate, increments the Term number, votes for itself, and broadcasts RequestVote RPCs. A candidate receiving votes from a majority of nodes becomes the Leader.\n\n"
         "2. Log Replication: Clients send state transitions to the Leader. The Leader appends the command to its local log and sends AppendEntries RPCs to all followers. Once a majority of followers acknowledge writing the entry, the Leader commits it, applies it to its state machine, and replies to the client.\n\n"
         "3. Safety Invariants: Raft guarantees that if a leader commits a log entry in a given term, that entry will be present in the logs of all leaders for all subsequent terms. A candidate can only be elected if its log is at least as up-to-date as the majority's logs."),
        ("What happens during a network partition where the 5-node cluster splits into 2 nodes and 3 nodes?",
         "The cluster safely handles the split-brain scenario:\n"
         "- The minority partition (2 nodes) cannot achieve a quorum (3 votes required). Any leader in this partition cannot commit new client writes and will reject or queue them.\n"
         "- The majority partition (3 nodes) can still form a quorum. If the old leader was in the minority partition, the majority partition will elect a new leader with a higher term number and continue committing writes normally.\n"
         "- When the partition heals, the minority nodes observe the higher term number from the majority leader, step down to followers, overwrite any uncommitted conflicting entries, and replicate the canonical log history.")
    ],

    # Dialogue 62: Concurrency - Deadlocks & The Dining Philosophers
    [
        ("What are Coffman's four conditions for a deadlock to occur, and how do we break them in practice?",
         "For a deadlock to occur in a concurrent system, all four of Coffman's conditions must hold simultaneously:\n\n"
         "1. Mutual Exclusion: At least one resource must be held in a non-shareable mode (only one thread can use it at a time).\n"
         "2. Hold and Wait: A thread holds at least one resource and is waiting to acquire additional resources held by other threads.\n"
         "3. No Preemption: Resources cannot be forcibly confiscated from a thread; they can only be released voluntarily.\n"
         "4. Circular Wait: A closed chain of threads exists such that thread T_1 waits for T_2, T_2 waits for T_3, and T_k waits for T_1.\n\n"
         "To eliminate deadlocks, you must break at least ONE condition:\n"
         "- Break Circular Wait (Most Common): Impose a strict global linear order on all locks. If lock A always precedes lock B in acquisition order across every thread, circular wait is mathematically impossible.\n"
         "- Break Hold and Wait: Acquire all required locks simultaneously in a single atomic operation (e.g., `std::lock(m1, m2)` in C++), or release existing locks before requesting new ones.\n"
         "- Break No Preemption: Use lock timeouts (`try_lock_for` or `try_lock`). If acquiring the second lock fails, release the first lock, back off with random jitter, and retry."),
        ("How does lock ordering prevent deadlock in the classic Dining Philosophers problem?",
         "In the Dining Philosophers problem with N philosophers and N forks arranged in a circle, if every philosopher grabs their left fork first, all forks are held and everyone starves waiting for their right fork (circular wait).\n\n"
         "Imposing a strict resource ordering breaks the symmetry: label forks 0 to N-1. Every philosopher must acquire the lower-numbered fork before the higher-numbered fork. Philosopher N-1 (who sits between fork N-1 and fork 0) must now pick up fork 0 first instead of fork N-1! This eliminates the circular dependency, guaranteeing that at least one philosopher can always acquire both forks and eat.")
    ],

    # Dialogue 63: Cache Eviction Policies - LRU, LFU & ARC
    [
        ("How do LRU and LFU cache eviction policies compare, and why was Adaptive Replacement Cache (ARC) invented?",
         "Cache eviction policies govern which entry to discard when a fixed-size cache reaches capacity:\n\n"
         "1. Least Recently Used (LRU):\n"
         "- Mechanism: Evicts the item that hasn't been accessed for the longest time, implemented via a Doubly Linked List + Hash Map in O(1) time.\n"
         "- Weakness: Vulnerable to 'cache pollution'. A single sequential scan over a large dataset (e.g., a batch database backup) will evict all frequently accessed hot data.\n\n"
         "2. Least Frequently Used (LFU):\n"
         "- Mechanism: Evicts the item with the lowest access frequency counter.\n"
         "- Weakness: Accumulated stale frequency bias. An item accessed heavily in the past retains a high counter indefinitely and remains in cache even after it is never touched again.\n\n"
         "3. Adaptive Replacement Cache (ARC):\n"
         "- Mechanism: Dynamically balances recency and frequency using two dual-list pairs ($T_1/B_1$ for recency, $T_2/B_2$ for frequency). It maintains 'ghost caches' ($B_1, B_2$) that track evicted metadata.\n"
         "- Self-Tuning: If a cache hit occurs in ghost list $B_1$, ARC increases the recency cache capacity $p$; if a hit occurs in $B_2$, it expands frequency capacity. It automatically adapts to changing workloads in real-time without manual parameter tuning.")
    ],

    # Dialogue 64: Amortized Analysis - Dynamic Arrays
    [
        ("Why is appending to a dynamic array (like Python's list or C++'s std::vector) considered O(1) amortized time even though occasional resizes take O(N)?",
         "A static array cannot grow. A dynamic array allocates a contiguous block of capacity $C$. When appending item $N+1 > C$, the array must:\n"
         "1. Allocate a new contiguous array of size $2C$ (geometric doubling).\n"
         "2. Copy all $N$ existing elements to the new array.\n"
         "3. Insert the new element and free the old buffer.\n\n"
         "This resize step takes $O(N)$ work. However, resizing happens with exponentially decreasing frequency! We can prove this using the Accounting (Token) Method:\n\n"
         "- Charge 3 computational credits for every simple append:\n"
         "  * 1 credit pays for the immediate insertion into the empty slot.\n"
         "  * 1 credit is saved for moving this element during the next resize.\n"
         "  * 1 credit is saved for moving an older element that was copied earlier.\n"
         "- When the array doubles from size $K$ to $2K$, exactly $K$ new elements were inserted since the last resize, accumulating $2K$ saved credits. This precisely pays for copying all $K$ elements without overdrafting.\n\n"
         "Because each operation is charged a constant 3 credits, the amortized cost per append is strictly $O(1)$.")
    ],

    # Dialogue 65: Memory Hierarchy - Page Tables, TLB & Cache Lines
    [
        ("How does CPU Virtual Memory translation work, and why are Translation Lookaside Buffer (TLB) misses so costly?",
         "Virtual memory provides each process with an isolated, contiguous address space while physical RAM is fragmented into 4KB pages:\n\n"
         "1. Translation via Multi-Level Page Tables:\n"
         "- In 64-bit x86-64 architecture (4-level paging: PML4 -> PDPT -> PD -> PT), translating a virtual address to a physical address requires walking 4 hierarchical pointer tables in RAM.\n"
         "- If a CPU had to perform 4 memory lookups for every single memory access, memory latency would quadruple!\n\n"
         "2. Translation Lookaside Buffer (TLB):\n"
         "- The TLB is a specialized, ultra-fast hardware content-addressable memory (CAM) cache on the CPU that stores recent virtual-to-physical page mappings. A TLB hit takes ~1 clock cycle.\n\n"
         "3. The Cost of a TLB Miss:\n"
         "- On a TLB miss, hardware page table walkers must query main memory (DDR RAM) 4 times sequentially, costing 50-200 nanoseconds (~200-800 CPU cycles) of stall time.\n"
         "- In memory-intensive applications (databases, deep learning tensors), using Huge Pages (2MB or 1GB pages) covers 512x to 262,144x more address space per TLB entry, drastically slashing TLB misses.")
    ],

    # Dialogue 66: Graph Algorithms - Topological Sort & Cycle Detection
    [
        ("How do we find a valid topological ordering of tasks in a Directed Acyclic Graph (DAG), and how do we detect cycles?",
         "A topological sort linearly orders vertices such that for every directed edge $u \\to v$, task $u$ comes before task $v$. Two standard algorithms solve this in $O(V + E)$ time:\n\n"
         "1. Kahn's Algorithm (BFS with In-degrees):\n"
         "- Compute the in-degree (number of incoming edges) for every vertex.\n"
         "- Enqueue all vertices with in-degree 0 into a queue.\n"
         "- While queue is not empty: pop vertex $u$, append $u$ to result list, and decrement the in-degree of all its neighbors $v$. If neighbor $v$'s in-degree drops to 0, enqueue it.\n"
         "- Cycle Detection: If the final result list contains fewer than $|V|$ vertices, the graph contains at least one directed cycle (deadlock / circular dependency).\n\n"
         "2. Tarjan's / Kosaraju's DFS with Three-Color State:\n"
         "- Color vertices: White (unvisited), Gray (currently visiting on active call stack), Black (completely processed).\n"
         "- If a DFS from vertex $u$ encounters a neighbor $v$ that is currently Gray, a back-edge exists—proving the existence of a cycle!")
    ],

    # Dialogue 67: Dynamic Programming - Optimal Substructure & Overlapping Subproblems
    [
        ("What two properties must a computational problem possess to be solvable via Dynamic Programming?",
         "A problem can only be solved using Dynamic Programming if it exhibits both of the following fundamental properties:\n\n"
         "1. Optimal Substructure:\n"
         "- An optimal solution to the overall problem can be constructed from optimal solutions to its subproblems.\n"
         "- Example: In finding the shortest path from $A$ to $C$ via intermediate node $B$, the subpath from $A$ to $B$ must itself be the shortest path from $A$ to $B$.\n"
         "- Counterexample: Longest simple path does NOT have optimal substructure, because concatenation of two longest simple subpaths may create a loop.\n\n"
         "2. Overlapping Subproblems:\n"
         "- A naive recursive solution solves the exact same subproblems repeatedly rather than generating unique subproblems.\n"
         "- Example: In computing the $n$-th Fibonacci number $F(n) = F(n-1) + F(n-2)$, both branches recursively calculate $F(n-2)$ independently, exploding into $O(2^n)$ work.\n"
         "- DP caches each subproblem result (via memoization or tabulation) so each is computed exactly once, collapsing complexity to $O(n)$.")
    ],

    # Dialogue 68: Low-Level Optimization - Branch Prediction & SIMD Vectorization
    [
        ("Why does sorting an array before processing its elements sometimes speed up a loop by a factor of 5x?",
         "This dramatic speedup is caused by CPU Branch Prediction!\n\n"
         "Consider this loop:\n"
         "```c\nfor (int i = 0; i < N; i++) {\n    if (data[i] >= 128) sum += data[i];\n}\n```\n"
         "1. Modern Superscalar CPUs use pipelined execution (15-20 stages deep). To keep the pipeline full, the branch predictor guesses whether `if (data[i] >= 128)` will be true or false before evaluating the condition.\n"
         "2. When data is unsorted: Values fluctuate randomly, causing the predictor to mispredict ~50% of the time. Every misprediction requires flushing the entire pipeline, discarding in-flight instructions, and reloading from memory, wasting 15-20 CPU cycles per branch.\n"
         "3. When data is sorted: All values below 128 come first, followed by all values >= 128. The predictor learns the pattern instantly and achieves 99.9% accuracy. Zero pipeline flushes occur, running at peak IPC (instructions per cycle).\n\n"
         "In production C++ and engine code, we replace unpredictable branches with branchless arithmetic (such as bitwise masking or conditional moves `CMOV`) to guarantee deterministic high throughput.")
    ]
]
