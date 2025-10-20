(** Probabilistic Context-Free Grammars with inside-outside training.

    Implements PCFGs for syntactic parsing and hierarchical sequence modeling.
    Supports CKY parsing with dynamic programming, Viterbi decoding for finding
    the most likely parse tree, and inside-outside (EM) training for parameter
    estimation.

    {1 Overview}

    A PCFG defines probability distributions over parse trees for sequences of
    terminal symbols. The grammar consists of:
    - Nonterminal symbols (internal tree nodes)
    - Terminal symbols (leaf nodes, observations)
    - Production rules mapping nonterminals to either pairs of nonterminals
      (binary rules) or single terminals (unary rules)

    This implementation uses Chomsky Normal Form (CNF) where all rules are
    either binary or unary.

    {1 Usage}

    Define a simple grammar:
    {[
      (* Nonterminals: 0=S (start), 1=NP, 2=VP *)
      (* Terminals: 0="cat", 1="dog", 2="runs", 3="sleeps" *)

      let rules = [
        { lhs = 0; rhs = Binary (1, 2); prob = 1.0 };  (* S -> NP VP *)
        { lhs = 1; rhs = Unary 0; prob = 0.5 };        (* NP -> "cat" *)
        { lhs = 1; rhs = Unary 1; prob = 0.5 };        (* NP -> "dog" *)
        { lhs = 2; rhs = Unary 2; prob = 0.6 };        (* VP -> "runs" *)
        { lhs = 2; rhs = Unary 3; prob = 0.4 }         (* VP -> "sleeps" *)
      ] in

      let grammar = Pcfg.create
        ~start:0
        ~num_nonterminals:3
        ~num_terminals:4
        rules
    ]}

    Parse a sentence:
    {[
      let sentence = [| 0; 2 |] in
      (* "cat runs" *)
      let log_prob = Pcfg.log_probability grammar sentence in
      Printf.printf "Log probability: %f\n" log_prob
    ]}

    Find the best parse tree:
    {[
      let backpointers = Pcfg.viterbi grammar sentence in
      (* backpointers.(i).(j).(nonterminal) contains the best split *)
    ]}

    Train using inside-outside:
    {[
      let training_sentences = [
        [|0; 2|];  (* "cat runs" *)
        [|1; 3|]   (* "dog sleeps" *)
      ] in
      let trained = Pcfg.inside_outside grammar training_sentences
    ]}

    {1 Algorithms}

    - {b CKY parsing}: Bottom-up dynamic programming over sentence spans. Time
      complexity O(n^3 * G) where n is sentence length and G is grammar size.
    - {b Inside-Outside}: EM algorithm for unsupervised learning. Iteratively
      refines rule probabilities to maximize likelihood of training sentences.
    - {b Viterbi}: Finds maximum probability parse tree using dynamic
      programming.

    {1 Performance}

    All algorithms use dynamic programming over spans [i, j] in the input
    sentence. Space complexity is O(n^2 * N) where N is the number of
    nonterminals. For large grammars, consider pruning low-probability rules. *)

type nonterminal = int
(** Nonterminal symbol identifier.

    Nonterminals represent syntactic categories (e.g., NP, VP, S). Must be
    non-negative integers in the range [0, num_nonterminals - 1]. *)

type terminal = int
(** Terminal symbol identifier.

    Terminals represent observed tokens in the input sequence. Must be
    non-negative integers in the range [0, num_terminals - 1]. *)

type production =
  | Binary of nonterminal * nonterminal
  | Unary of terminal
      (** Production rule right-hand side.

          - [Binary (b, c)]: Expands to two nonterminals, representing a
            branching tree node.
          - [Unary term]: Expands to a single terminal symbol, representing a
            leaf node. *)

type rule = { lhs : nonterminal; rhs : production; prob : float }
(** Grammar production rule.

    Represents [lhs -> rhs] with probability [prob]. Probabilities for all rules
    with the same [lhs] are normalized to sum to 1.0 during grammar creation. *)

type t
(** Probabilistic Context-Free Grammar.

    Stores rules grouped by right-hand side for efficient CKY parsing.
    Probabilities are normalized per left-hand side nonterminal. *)

val create :
  start:nonterminal ->
  num_nonterminals:int ->
  num_terminals:int ->
  rule list ->
  t
(** [create ~start ~num_nonterminals ~num_terminals rules] builds a grammar.

    Rules are automatically normalized so that for each nonterminal, the
    probabilities of all productions sum to 1.0. Invalid probabilities (negative
    or all zeros) are replaced with uniform distributions.

    @param start
      The start symbol nonterminal. Parse trees must have this nonterminal as
      the root.
    @param num_nonterminals Number of nonterminal symbols in the grammar.
    @param num_terminals Number of terminal symbols in the grammar.

    @raise Invalid_argument if [num_nonterminals <= 0] or [num_terminals <= 0].
*)

val start_symbol : t -> nonterminal
(** [start_symbol grammar] returns the start symbol nonterminal. *)

val num_nonterminals : t -> int
(** [num_nonterminals grammar] returns the number of nonterminals. *)

val num_terminals : t -> int
(** [num_terminals grammar] returns the number of terminals. *)

val inside : t -> terminal array -> float array array array
(** [inside grammar sentence] computes inside probabilities using CKY.

    Returns a 3D array where [alpha.(i).(j).(A)] represents P(span \[i,j)
    derives from nonterminal A), the probability that the substring from
    position i (inclusive) to j (exclusive) can be generated by nonterminal A.

    The array has dimensions (n+1) x (n+1) x num_nonterminals where n is the
    sentence length. Only entries where j > i are meaningful.

    Time complexity: O(n^3 * |rules|). *)

val outside :
  t -> terminal array -> float array array array -> float array array array
(** [outside grammar sentence inside_chart] computes outside probabilities.

    Returns a 3D array where [beta.(i).(j).(A)] represents P(generating
    everything outside span \[i,j) | A generates span \[i,j)), the probability
    of the context outside the span given that nonterminal A generates the span.

    The [inside_chart] must be precomputed using {!inside} on the same sentence.
    Outside probabilities are used in the inside-outside algorithm for parameter
    estimation. *)

val log_probability : t -> terminal array -> float
(** [log_probability grammar sentence] computes sentence log probability.

    Returns the natural logarithm of P(sentence | grammar), computed using the
    inside algorithm. The probability sums over all possible parse trees.

    Sentences that cannot be generated by the grammar have very negative
    log-probabilities (approaching negative infinity). *)

val viterbi :
  t -> terminal array -> (int * nonterminal * nonterminal) array array array
(** [viterbi grammar sentence] finds the maximum probability parse tree.

    Returns a backpointer chart where [chart.(i).(j).(A) = (k, B, C)] indicates
    that the best way to parse span \[i,j) with nonterminal A is to split at
    position k with left child B and right child C.

    For unary rules (terminals), the backpointer is [(-1, -1, -1)]. Use the
    backpointers to reconstruct the parse tree by recursively following splits
    from the root span \[0, n) with the start symbol.

    {4 Example}

    {[
      let chart = Pcfg.viterbi grammar sentence in
      let n = Array.length sentence in
      let start = Pcfg.start_symbol grammar in
      let (k, left, right) = chart.(0).(n).(start) in
      (* Best split of full sentence is at position k *)
    ]} *)

val sample : t -> terminal array -> nonterminal list option
(** [sample grammar sentence] samples a random parse tree.

    Returns a parse tree as a pre-order traversal of nonterminals, or [None] if
    the sentence has zero probability under the grammar.

    The sampling is conditioned on the sentence: it draws from P(tree |
    sentence, grammar) by stochastically choosing splits proportional to their
    inside probabilities.

    Uses a random seed from [Random.bits ()], so results vary between calls.

    {4 Example}

    {[
      match Pcfg.sample grammar [| 0; 2 |] with
      | Some nonterminals ->
          List.iter (fun nt -> Printf.printf "%d " nt) nonterminals
      | None -> Printf.printf "Sentence is impossible under grammar\n"
    ]} *)

val inside_outside :
  ?tol:float -> ?max_iter:int -> t -> terminal array list -> t
(** [inside_outside grammar sentences] trains the grammar using EM.

    Performs inside-outside algorithm to find maximum likelihood rule
    probabilities given the training sentences. The initial [grammar] provides
    the structure (nonterminals, terminals, rule templates) and initial
    probabilities.

    @param tol Convergence tolerance on log-likelihood change. Default: 1e-4.
    @param max_iter Maximum number of EM iterations. Default: 50.

    Returns a new grammar with updated rule probabilities. Convergence occurs
    when the change in total log-likelihood across all sentences falls below
    [tol], or when [max_iter] iterations complete.

    {4 Example}

    {[
      let sentences = [[|0; 2|]; [|1; 3|]; [|0; 3|]] in
      let trained = Pcfg.inside_outside ~max_iter:100 initial_grammar sentences
    ]} *)
