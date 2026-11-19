# Designing in raven

Use this for any design or semantic decision: a new concept, a change to a
public `.mli`, a promise about results (bits, bounds, ordering, failure),
or a structure justified by performance. Fixes, kernels' internals and
renames don't need it. Work through it in order: the constraint, the
values, the hard constraints, the shape, then certainty and review.

## The constraint

Every part of raven must stay understandable by one person reading it: few
concepts, each fact in one place, code that reads as prose, as few lines
as the job allows. A design that only a team could hold in its head is
wrong for raven, however correct or fast. When a part outgrows this,
redesign it from what its fixes taught.

## Deciding between good things

Ranked. When two conflict, the higher one wins.

1. **Correct results.** A wrong number does more harm than a slow or
   missing one, because the user can't see it. Never keep a biased result
   for compatibility, and never compute where the user didn't place the
   work.
2. **Understandable code.** Few concepts, each fact in one place, prose.
   Familiar conventions count here, as knowledge the reader already has:
   departing from one needs a clear win.
3. **Performance at the hardware's limit.** Match or beat the best vendor
   on each machine, or the roofline where no vendor has the operation.
   Never by binding a vendor library.

Trading rules:

- **Correctness is never traded.**
- **Speed pays for its lines.** A performance change states its gain and
  its line cost; a gain that doesn't pay for its code is rejected (500
  lines for 2 ns isn't worth it). This holds below vendor parity too: a
  kernel may stop short of the vendor when the last gap costs more code
  than it is worth, and the gap is stated.
- **Simpler means less to hold:** concepts, files, names, cross-references
  a reader must keep in mind for one piece. Saving lines by intertwining
  independent parts makes the code worse.
- **Scope is not traded for simplicity.** A capability raven needs gets a
  simpler design; it is never cut to save lines.
- **The cost of changing today's code weighs nothing.** Diff size, the
  number of libraries touched or downstream rewrites never decide between
  two designs. Before 1.0 this includes compatibility: change APIs, bits
  and seeds whenever the design or a result improves, and state each break
  in its commit.

## Hard constraints

Hold these in every design. Each rules something out.

- **No dependencies, except for composition.** No system libraries, and
  no OCaml packages beyond the distribution; reimplement the subset you
  need, and measure against vendor libraries without linking them. A
  library may depend on an OCaml library only when that library's type
  appears in its public interface, so raven composes with the OCaml
  ecosystem (brot's `Jsont.json`, talon's `Bytesrw` readers and writers).
  Rules out a dependency used only inside an implementation, however small
  or well designed.

- **Nothing happens behind the caller's back.** Placement, cost and
  failure belong to the caller; a value holds no hidden state or I/O; a
  mistake raises at the call that made it. Rules out implicit moves
  between devices, conversions that silently allocate, and a device
  without kernels computing on the host.
- **What the types can state, they state.** A static fact is a type; a
  fact known only at run time is checked at run time. Rules out run-time
  checks of static facts, and typing no call site gains from.
- **One job per library; libraries compose.** Libraries work together
  through composable APIs, never through shared abstractions: they
  exchange nx values, structures and functions, and copy small helpers.
  Rules out a sibling dependency (hugin on talon), code shared between
  rig's drivers, autodiff vocabulary in nx, and a second compiler beside
  tolk.
- **The `.mli` is the source of truth.** Each rule lives in one `.mli`,
  which specifies and teaches it, from first principles. An `.mli` may cite
  the external work it follows (a standard, a paper) and other `.mli`
  files, never raven's internal documents. Rules out a separate book
  re-explaining interfaces, and references to RFCs, plans, ledgers or
  guides from code or an `.mli`.
- **Debt is paid by redesign.** When fixes pile up in a part, redesign it
  as you would knowing what they taught, keeping every fix and gain, and
  change every consumer in the same sweep, even if the build is broken in
  between. Rules out compatibility shims, bridge modules, dead fields, a
  bug filed for later, and a workaround for a gap in a library below.

## Shaping a design

The bar is the design a careful library author would still defend in ten
years: few concepts, each with something a caller can point at; total
functions over plain data; illegal states unrepresentable; no knobs and no
modes; call sites that read as prose. Judge a design by what a production
1.0 needs, never by "nothing uses it yet".

- **Make the right change.** Move the code toward the one true design,
  however much it touches. Diff size is never the measure.
- **Ask whether the layer is right before the second fix.** A mechanism
  that needs a second patch for the same class of bug is in the wrong
  place.
- **Lines are a signal.** A large diff or module usually means duplicated
  machinery or broken design. Look for it before adding more.
- **Layers.** Libraries form a hierarchy: a library uses any library below
  it, never one above it. Low-level mechanics (drivers, raw I/O, byte
  layouts) stay behind the layer that abstracts them; a caller that has the
  abstraction uses it, never what it hides. A layer too slow or too weak
  for its caller is fixed, never bypassed.
- **Composition over dependencies and abstractions.** Each library has one
  job and a minimal set of dependencies. Copy a small helper rather than
  create a shared library, unless the copies would state one fact that
  must stay the same (a format, a contract, a result that must match bit
  for bit); that fact lives in one place.
- **Vocabulary.** A familiar name is an entry point onto a general
  operation, never a mechanism of its own. A composite operation enters an
  interface only when composing it correctly takes knowledge the caller
  lacks, such as a numerically stable form; otherwise the caller composes
  it.
- **Visibility.** Everything stays out of the `.mli` unless the design
  needs it exported. Adding to a public `.mli` is a design change. A
  value another library truly needs is public under an expert or unsafe
  warning that states its contract, or behind `(**/**)`; never a
  `Private` module or a library split to hide it.
- **No abstraction before its second consumer.** A module type, functor
  or split module lands with the second user that needs it. A step whose
  consumer the plan already schedules is not early.
- **No arbitrary limits.** A bound that no hardware, operating system or
  specification imposes is a bug, however large.
- **Fix upstream.** A capability missing in nx, rune or tolk is built
  there, never worked around downstream.
- **Argue for the most elegant shape.** Each new concept (a module type,
  functor, scope, cache, mode or optional argument) weighs its
  alternatives (plain data, an existing noun, composition, not exporting
  it) and shows the chosen one captures the problem's structure with the
  fewest concepts and each fact in one place. A design that can't make
  the case is not ready.
- **Briefs state problems.** A brief to an agent gives the problem, its
  constraints and the decisions already made, never a mechanism nobody
  weighed.

## Becoming certain

Many people have solved most of our problems before, carefully, and
written down why. Decide from that work, read and checked, so the decision
is certain without the maintainer.

### Steps

1. **Blind designs.** One or two agents design the answer from the
   constraints above and references, without seeing anyone's proposal. A
   proposal handed to an auditor anchors it.
2. **Audit.** One agent compares the designs and today's code against the
   references, by the method below.
3. **Skeptic.** Another attacks the audit.
4. **Adjudication.** The lead writes what the audit concluded, what the
   skeptic changed, which evidence decided, and what remains the
   maintainer's call, stated as one question.
5. **Ledger.** The decision enters the campaign's ledger: decision,
   references, confidence, date, links. Check the ledger before reopening
   a question.
6. **Brief.** Only then does implementation start, from the decided
   design.

### What counts as a reference

All of these hold:

1. **Its semantics are specified:** a written spec or design paper, by
   people who designed for this exact problem.
2. **Its reasoning is published:** why, and what was rejected or
   regretted.
3. **It has held up,** or it is the documented lesson of a failure
   (OpenCL's `clSetKernelArg`, "an inherent race condition in the design
   of OpenCL").
4. **Preferably, independent designs converged on it.** One lineage
   copied several times is one design.

Say what makes it excellent: a property it guarantees, a failure it
prevents, a result someone measured. Age, size and popularity are not
reasons. An implementation without a written design (PyTorch, tinygrad,
JAX's code, today's nx) is a datapoint.

**Every citation is read, never recalled:** fetch it, and give title,
section, URL and a quote of at most two sentences. If it couldn't be
fetched, say so and weigh it as unread.

### Semantics from references, structure from raven's constraint

- **Domain references decide semantics:** what a result must be, which
  promises hold, which failures to avoid.
- **Design references decide structure:** how few concepts it takes.
  They qualify by the same bar plus a stated goal of simplicity the
  result met.
- **Take a reference's principle; weigh its mechanism.** The principle
  (what must hold, and why) transfers when raven's users need it. The
  mechanism that implements it is one candidate design among others, held
  to the constraint above like any design of ours: it wins only when no
  simpler shape keeps the principle.
- **Existence proofs shift the burden.** A measured result showing a
  simpler system reaches the same outcome means the complex mechanism
  isn't needed; it never shows the simple design correct (tinygrad is an
  existence proof, not a design reference). Complexity is justified by a
  user's need, never by a reference having it.

Example: MLIR and tinygrad share the principle of lowering progressively,
keeping high-level information until the step that needs it, and differ
on the mechanism: a dialect per level against one node type for every
level. The principle transfers; the mechanism is decided by the constraint,
here by which keeps fewer concepts for raven's compiler.

### Method, for each decision

1. **State it** as the code or proposal states it, with `file:line`.
2. **First principles:** who uses this, what they need, what breaks
   without it.
3. **References:** cite them and say what each gives, separating its
   principle from its mechanism; where they disagree, say why and which
   fits.
4. **Compare:** justify any deviation from every reference by a reason
   specific to raven (OCaml's GC and domains, zero dependencies, several
   vendors behind one contract), or adopt the reference.
5. **Measure:** every cost claim is a number from the real code path, or
   a model whose match to the code is shown.
6. **Verdict:** adopt (with signatures), keep the deviation (with the
   reason), or no reference (flagged for a second look), with the
   confidence that the reference can be followed blindly.

End the audit with a table: decision, reference, match, verdict,
confidence.

### The skeptic

- Fetch every citation again; check each quote says what the audit
  claims, in context.
- Check every code claim against the tree.
- Re-measure, on the real path where possible.
- Look for the shape the audit didn't consider.
- End with a table: claim, holds / weakened / refuted, why, what to
  decide.

Where audit and skeptic disagree, the evidence decides and the
adjudication says which way. A confirmed reference enters "References"
with the claim it supports; a misused one is recorded with its actual
scope.

## Reviewing a design

Read what the code does, case by case, before judging what it should be.
Then check for the three ways reviews have let broken designs through:

- **Judging only the shapes handed to you.** A choice between a callback
  and an option hides the shape where neither is needed. Propose at least
  one shape nobody has argued.
- **Accepting a cost as an argument.** "Zero allocation" justified mutable
  GPU frames, and "it is slower" then argued for replacing them; measured,
  both were wrong, and correctness decided the choice.
- **Not asking which layer a decision belongs to.** A function passed down
  so a lower layer can call back up means each layer is answering the
  other's question.

A change to a public `.mli` lands only after a review of its shape against
this document; passing tests and a correctness review don't make it ready.

## References

References that have passed the bar above: specified, their reasoning
published, held up or a documented failure. Each entry says what it is
authoritative on, why it is excellent, and where it has been misused. An
entry enters when an audit and its skeptic confirm the claim it supports.
Datapoints (implementations without a written design) are not listed.

### Programs and intermediate forms

- **SPIR-V specification** (Khronos, §2.5.1),
  <https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html>.
  Authoritative on a flat, numbered SSA form at the boundary where one side
  hands a program to another. Excellent because one specified form serves
  many independent producers and consumers. Scope: §2.8 is about type
  declarations, not values; a result type is written by the producer, not
  derived by validation.
- **LLVM** (Lattner & Adve, CGO 2004),
  <https://llvm.org/pubs/2004-01-30-CGO-LLVM.pdf>. Authoritative on a
  numbered SSA form as the interchange between tools.
- **MLIR** (Lattner et al., CGO 2021, <https://arxiv.org/pdf/2002.11054>;
  Rationale, <https://mlir.llvm.org/docs/Rationale/Rationale/>).
  Authoritative on SSA with regions and builders. Scope: its regret about
  global uniquing concerns uniqued *mutable* objects; MLIR interns immutable
  data itself. It says nothing against immutable expression values.
- **Halide** (Ragan-Kelley et al., PLDI 2013),
  <https://people.csail.mit.edu/jrk/halide-pldi13.pdf>. Authoritative on
  expressions as values in the host language, lowered to a backend form
  later.
- **Type-safe modular hash-consing** (Filliâtre & Conchon, ML Workshop
  2006), <https://usr.lmf.cnrs.fr/~jcf/publis/hash-consing2.pdf>.
  Authoritative on hash-consing in OCaml. Scope: about keys that stay valid
  indefinitely; it does not apply to a walk that ends before the GC can run.
- **XLA operation semantics**, <https://openxla.org/xla/operation_semantics>.
  Its builder handles carry their builder, which makes operators work and
  rejects a handle from another builder; errors are deferred to `Build()`.

### Distribution over devices

- **GSPMD** (Xu et al., 2021, <https://arxiv.org/abs/2105.04663>), with
  **GShard** (<https://arxiv.org/abs/2006.16668>), **Mesh-TensorFlow**
  (<https://arxiv.org/abs/1811.02084>) and **Alpa**
  (<https://arxiv.org/abs/2201.12023>): one lineage. Authoritative on
  deriving each shard's local operation from one rule per operation. Scope:
  Alpa §4.2's avoidance of replicated matmul is a cost-model choice, not a
  semantic rule.
- **Legion** (Bauer et al., SC 2012,
  <https://legion.stanford.edu/pdfs/sc2012.pdf>) and **Legate**
  (<https://legion.stanford.edu/pdfs/legate-preprint.pdf>): a variant chosen
  per point task, and Legate §3.3's projection functions per argument.
  Converges independently with the GSPMD lineage on one rule per operation.
  Scope: Legate's fallback to NumPy is a prototype limitation it lists, not
  a design.
- **High Performance Fortran 2.0** (<https://www.netlib.org/hpf/hpf-v20-final.ps.gz>)
  and **Chapel** (<https://chapel-lang.org/publications/hotpar10-final.pdf>,
  <https://chapel-lang.org/docs/primers/parIters.html>): decades of data
  distribution semantics; leader-follower iterators. HPF names reductions
  as where distribution changes results.
- **Efficient Reproducible Floating Point Summation and BLAS** (Demmel,
  Ahrens & Nguyen, UCB/EECS-2016-121), with ReproBLAS. Authoritative on
  results that are bitwise identical across any number of processors and
  any partition, at about 7n additions per sum.
- **Shardy**, sharding representation, "Axis splitting and sub-axes"
  (<https://openxla.org/shardy/sharding_representation>). Authoritative on
  a reshape whose result is sharded so that each device keeps its data.
  Excellent because it states the goal, to "avoid communication (i.e. keep
  the data where it is)". Scope: sub-axes split mesh axes and should divide
  their size, and a dimension its axes do not divide is padded, which leaves
  a balanced cut at block granularity without a reference.

### Scans and summation

- **Ladner & Fischer** (JACM 1980, <https://www.cse.chalmers.se/~ms/PP/ladner.pdf>)
  and **Brent & Kung** (1982,
  <https://maths-people.anu.edu.au/~brent/pd/rpb060.pdf>): work-efficient
  prefix networks, under 2n combines in about 2 log₂ n rounds, needing no
  identity element.
- **Hillis & Steele** (CACM 1986): the doubling network, n log n work in
  log₂ n rounds. Scope: it fits when processors are not virtualized.
- **Merrill & Garland**, "Single-pass Parallel Prefix Scan with Decoupled
  Look-back" (NVIDIA 2016,
  <https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf>).
  §2: when processors are virtualized, time follows total work. Also the
  design for a native one-pass GPU scan, and the lesson that a variable
  look-back makes float scans differ between runs.
- **Blelloch**, "Prefix Sums and Their Applications" (1993,
  <https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf>). Scope: its exclusive
  scan needs an identity element, which `Arg` lacks.
- **Higham**, *Accuracy and Stability of Numerical Algorithms* (2002):
  error bounds of summation by order of association.
- **Chan, Golub & LeVeque**, "Algorithms for Computing the Sample Variance"
  (1983,
  <https://www.math.pku.edu.cn/teachers/litj/notes/numer_anal/AmerStat_37_242_Chan_Variance.pdf>):
  updating against pairwise variance, and the combine of partial moments.
  Scope: it conjectures, and does not prove, a bound for the pairwise
  form.

### Launching kernels

- **CUDA driver API**, execution control
  (<https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html>):
  `cuLaunchKernel` takes each launch's parameters as values, and CUDA
  deprecated its mutable per-function setters (`cuParamSet*`,
  `cuFuncSetBlockShape`). Authoritative on a launch as an immutable value.
- **OpenCL 3.0**, footnote 39 on `clSetKernelArg`: "an inherent race
  condition in the design of OpenCL". Scope: its own fix is one kernel
  object per thread, not values per launch.
- **Vulkan**, threading behaviour and command pools
  (<https://docs.vulkan.org/spec/latest/chapters/fundamentals.html>):
  externally synchronized objects owned by one thread at a time, the model
  for a per-domain run.
- **SYCL 2020**: a handler created per submission. **Level Zero**
  (<https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/INTRO.html>)
  keeps mutable kernel arguments on purpose and says why; scope: it is no
  evidence that every API moved to values.

### Windows, frames and ranges

- **Leis, Kundhikanjana, Kemper & Neumann**, "Efficient Processing of
  Window Functions in Analytical SQL Queries" (PVLDB 8(10) 2015,
  <https://www.vldb.org/pvldb/vol8/p1058-leis.pdf>). Authoritative on SQL
  frames. A frame applies to navigation functions and non-distinct
  aggregates (§2.3), bounds may depend on the row, range mode takes exactly
  one numeric order key (fn. 1), and with an order the default frame runs
  through the row's peers. Excellent because §4.3.4 evaluates any frame in
  O(log n) with a segment tree. Scope: §2.1 calls partitioning, ordering and
  framing "three simple and orthogonal concepts", which does not support a
  two-construct split. §4.3.5's "safe choice" rests on partitioning and sorting
  dominating the query, so it transfers only to windows that sort.
- **Substrait**, window functions
  (<https://substrait.io/expressions/window_functions/>). Authoritative on
  one written definition of frames with their edge cases. A null ordering
  value makes an offset bound the current row, a bound defaults to
  unbounded, and a negative offset is invalid. Excellent because these
  cases are rules of the format. Scope: its rule against window functions
  in offsets covers bounds and says nothing about a reduction's operand.
- **Eckner**, "Algorithms for Unevenly Spaced Time Series: Moving Averages
  and Other Rolling Operators" (rev. 2019,
  <http://www.eckner.com/papers/Algorithms%20for%20Unevenly%20Spaced%20Time%20Series.pdf>).
  Authoritative on rolling operators over irregular times and the
  half-open window (t − τ, t] (§1.1, §5). Excellent because §4.1 shows the
  usual EMA implicitly applies next-point interpolation. Scope: §1.1 assumes
  strictly increasing times, so it gives no rule for ties. §7's warning
  concerns add-and-subtract updates and §6.2 uses the moment formula, so
  refusing that formula rests on Chan alone.
- **Tangwongsan, Hirzel, Schneider & Wu**, "General Incremental
  Sliding-Window Aggregation" (PVLDB 8(7) 2015,
  <https://www.vldb.org/pvldb/vol8/p702-tangwongsan.pdf>). Authoritative on
  sliding aggregation that requires only associativity (§2.3, §3.2).
  Excellent because it covers non-invertible functions such as Min and Max.
  Scope: a stream's sliding window.
- **DABA** (Tangwongsan, Hirzel & Schneider, extended 2020,
  <https://arxiv.org/pdf/2009.13768>) and **FiBA** (PVLDB 12(10) 2019,
  <https://www.vldb.org/pvldb/vol12/p1167-tangwongsan.pdf>). Authoritative
  on streaming sliding aggregation. Excellent because FiBA states its bound,
  amortized O(log d). Scope: both are streaming, and DABA's family works "only
  ... for FIFO windows".
- **Wesley & Xu**, "Incremental Computation of Common Windowed Holistic
  Aggregates" (PVLDB 9(12) 2016,
  <https://www.vldb.org/pvldb/vol9/p1221-wesley.pdf>). Authoritative on
  incremental holistic aggregates. Excellent because it measures the gain,
  "about 10×" over naive recomputation. Scope: speed only.
- **Alon & Schieber**, "Optimal Preprocessing for Answering On-Line Product
  Queries" (TR 71/87, 1987, <https://arxiv.org/pdf/2406.06321>).
  Authoritative on answering a range with one combine of two precomputed
  products (§2.1) after O(n log n) preprocessing. Excellent because §2.2
  proves that preprocessing optimal. Scope: the bound is for two steps per
  range, one combine, over arbitrary online queries. Four steps need O(n log*
  n) preprocessing (§1), and the lower bound assumes no inverse, since a group
  answers from prefix computations (§2.2).
- **Pébay, Terriberry, Kolla & Bennett**, "Numerically stable, scalable
  formulas for parallel and online computation of higher-order multivariate
  central moments with arbitrary weights" (Comput. Stat. 31, 2016,
  <https://people.xiph.org/~tterribe/pubs/P%C3%A9bayTerriberryKolla+16-_Numerically_Stable,_Scalable_Formulas_for_parallel_and_Online_Computation_of_Higher-Order_Multivariate_Central_Moments_with_Arbitrary_Weights.pdf>).
  Authoritative on combining central moments of two parts. Excellent
  because (2.10) and (3.21) state the combine in closed form. Scope: it carries
  no shift of the data, and §3 (3.4) of Chan gives only the principle of
  shifting by a data point. A combine that shifts at each step has a measured
  bound.

### Joins and intervals

- **Kulkarni & Michels**, "Temporal features in SQL:2011" (SIGMOD Record
  41(3) 2012,
  <https://sigmodrecord.org/publications/sigmodRecord/1209/pdfs/07.industry.kulkarni.pdf>).
  Authoritative on SQL:2011 periods. A period is closed-open, a declared
  constraint keeps its start before its end (§2.1), and OVERLAPS holds when
  two periods share a time point (§2.2.3). Excellent because the
  definition stands beside the constraint that makes the bare conjunction
  correct. Scope: it says nothing about a reversed pair or about predicates
  over undeclared pairs.
- **Allen**, "Maintaining Knowledge about Temporal Intervals" (CACM 26(11)
  1983, <https://cse.unl.edu/~choueiry/Documents/Allen-CACM1983.pdf>).
  Authoritative on half-open intervals as an artificial device for
  reasoning about time (§3). Excellent because it says what the convention
  buys, that intervals "could then meet as required". Scope: §3's ordered
  pair of points is the model he describes and sets aside for intervals as
  primitive, so it supports no precondition that a start precede its end.
- **Dignös, Böhlen, Gamper, Jensen & Moser**, "Leveraging range joins for
  the computation of overlap joins" (VLDB J. 31, 2022,
  <https://web.archive.org/web/20220528041618id_/https://link.springer.com/content/pdf/10.1007/s00778-021-00692-3.pdf>).
  Authoritative on overlap as two disjoint range joins. Excellent because
  the two ranges produce no duplicates, and §7.2.2 measures runtimes
  comparable to a forward scan. Scope: it assumes B < E, so it gives no meaning
  to an empty half-open period.
- **Bouros & Mamoulis**, "A Forward Scan based Plane Sweep Algorithm for
  Parallel Interval Joins" (PVLDB 10(11) 2017,
  <https://www.vldb.org/pvldb/vol10/p1346-bouros.pdf>). Authoritative on
  forward-scan interval joins. Excellent because the cost excluding
  sorting is O(|R| + |S| + K). Scope: its intervals
  are closed with start ≤ end, so a zero-length interval is a point.
- **Khayyat et al.**, "Lightning Fast and Space Efficient Inequality Joins"
  (PVLDB 8(13) 2015, <https://www.vldb.org/pvldb/vol8/p2074-khayyat.pdf>).
  Authoritative on inequality joins over two columns. Excellent because it
  publishes its bound, O(m log m + n log n + m·n) (§3.1), and says its
  runtime follows the output (§9). Scope: the quadratic bound is IEJoin's own
  (§3.1), and one-column ranges are output-sensitive. DuckDB's 9× and 500×
  against IEJoin
  (<https://duckdb.org/2023/09/15/asof-joins-fuzzy-temporal-lookups.html>)
  time a rewritten as-of.
- **Böhlen, Snodgrass & Soo**, "Coalescing in Temporal Databases" (VLDB
  1996, <https://www.vldb.org/conf/1996/P180.PDF>). Authoritative on
  merging tuples of equal value over adjacent or overlapping periods.
  Excellent because §1 states that "prior coalescing is necessary to ensure
  the semantics of some operators". Scope: only the abstract and §1 were
  checked.

### Lists and nesting

- **Dremel** (Melnik et al., PVLDB 3(1) 2010,
  <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36632.pdf>).
  Authoritative on naming nested fields by path, with an expression's
  nesting level following from the most repeated field it reads (§5).
  Excellent because `WITHIN` scopes a reduction by a path and a field
  reference means the same field everywhere. Scope: "Defining the language
  formally is out of scope of this paper". Protocol-buffer repeated fields
  cannot be null, so it gives no evidence on null against empty lists.
- **Arrow columnar format**
  (<https://arrow.apache.org/docs/format/Columnar.html>). Authoritative on
  list and struct layouts. Offsets reference the child array and stay
  monotonic even for null slots, a null entry may cover a non-empty
  segment, and a struct's validity is independent of its children's.
  Excellent because these corner cases are rules of the layout. Scope: a
  layout, so the result of a reduction over a null list is decided
  elsewhere.

### Host memory and registration

- **CUDA host memory**: driver API memory management
  (<https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html>),
  API synchronization behaviour
  (<https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html>),
  programming guide §2.6.3
  (<https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html>)
  and best practices §10.1.1
  (<https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>).
  Authoritative on page-locked memory and on copies of pageable memory.
  Excellent because it states the cost (`cuMemHostRegister` is "best used
  sparingly") and the rule that asynchronous copies need page-locked host
  memory. Scope: registration's tracking speeds later copies of the plain
  pointer, which is CUDA's mechanism. Where pageable access uses the host
  page tables, `cuMemHostRegister` might not page-lock. The sentence
  "returns only once the copy has completed" covers pinned memory too, and
  the asynchronous section is the evidence for staging. Unified Addressing
  §6.45.6
  (<https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html>)
  is about pointer values.
- **Vulkan `VK_EXT_external_memory_host`**, issues
  (<https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/main/appendices/VK_EXT_external_memory_host.adoc>)
  and host-pointer import
  (<https://docs.vulkan.org/refpages/latest/refpages/source/VkImportMemoryHostPointerInfoEXT.html>),
  **ROCr's `hsa_amd_memory_lock`**
  (<https://raw.githubusercontent.com/ROCm/ROCR-Runtime/amd-staging/runtime/hsa-runtime/inc/hsa_ext_amd.h>)
  and **`ibv_reg_mr(3)`**
  (<https://man7.org/linux/man-pages/man3/ibv_reg_mr.3.html>). Authoritative
  on registration as an explicit call that names its consumers or all of
  them and lasts until it is undone. Excellent because, with CUDA, three
  lineages (NVIDIA, AMD and its Vulkan extension, OFA) converged on it and none
  pins as a side effect of a copy. Scope: HSA returns one code,
  OUT_OF_RESOURCES, when a valid pin fails, so it does not support a split
  between never and no room. OpenCL 3.0 §5.2
  (<https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html>)
  lets an implementation use the pointer in place, which leaves it neutral.
- **Linux `pin_user_pages`**
  (<https://raw.githubusercontent.com/torvalds/linux/master/Documentation/core-api/pin_user_pages.rst>)
  and **`unevictable-lru`**, "Migrating MLOCKED Pages"
  (<https://raw.githubusercontent.com/torvalds/linux/master/Documentation/mm/unevictable-lru.rst>).
  Authoritative on the kinds of page pin. Excellent because it separates
  long-lived DMA pins (CASE 2) from pins with notifiers (CASE 3), and says
  mlocked pages can migrate. Scope: it describes the mechanism under CUDA's
  kernel driver. CASE 1 covers short pins for direct I/O,
  and it says nothing about device copies of pageable memory.

### Processor levels

- **x86-64 psABI**, "Micro-Architecture Levels"
  (<https://gitlab.com/x86-psABIs/x86-64-ABI>, low-level-sys-info.tex).
  Authoritative on x86-64-v2, v3 and v4 as cumulative sets of CPUID
  features, with v3 and v4 available only when fully enabled, XCR0 included.
  Excellent because GCC, glibc's `ld.so` and Go's `GOAMD64` select code by
  these levels. Scope: it requires run-time detection of features outside
  the levels, even those every current CPU has.
- **Go**, `GOAMD64` (<https://go.dev/wiki/MinimumRequirements>, proposal
  <https://github.com/golang/go/issues/45453>). Authoritative on a binary's required level. Excellent because the binary checks its level at startup.
  Scope: "More granular means more complication" answers a request for
  levels between v1 and v2. That reply supports fewer configurations to
  test and cache and says nothing about a machine stating its features to a
  compiler.
- **Linux kernel**, x86 feature flags
  (<https://docs.kernel.org/arch/x86/cpuinfo.html>), arm64 hwcaps
  (<https://docs.kernel.org/arch/arm64/elf_hwcaps.html>) and CPU feature
  registers (<https://docs.kernel.org/arch/arm64/cpu-feature-registers.html>).
  Authoritative on what a kernel reports as usable. Excellent because a
  flag reflects what the kernel enabled, XSAVE included, and an exported
  value is safe across every CPU. Scope: hwcaps' "cannot be probed
  reliably through other means" concerns features without architected
  EL0 discovery. The x86 page sends userspace to probing a feature, which
  backs an XCR0 check.
- **LLVM AMDGPU**, processors and target features
  (<https://llvm.org/docs/AMDGPUUsage.html>). Authoritative on a processor
  name as a code target. Excellent because "Generic processors allow
  execution of a single code object on any of the processors that it supports".
  Scope: AMD GPU targets, and a mismatch of features "may result in incorrect
  execution, or a reduction in performance".

### Precision in training

- **Micikevicius et al.**, "Mixed Precision Training" (ICLR 2018,
  <https://arxiv.org/abs/1710.03740>). Authoritative on a float32 master
  copy of weights updated in float32. Excellent because §3.1 names the
  failure it prevents, updates that vanish against a large weight. Scope: it
  says nothing on optimizer-state precision, and §3.1 finds activations
  dominate training memory. "Using a single-precision copy" follows the
  sentence on small gradients.
- **Zamirai, Zhang, Aberger & De Sa**, "Revisiting BFloat16 Training"
  (<https://arxiv.org/abs/2010.06192>). Authoritative on stochastic rounding
  of the weight update in 16-bit-FPU training (§3.2). Excellent because
  Table 4 measures it within 0.1% of 32-bit on five of seven tasks. Scope:
  rounding applies to the subtraction output only, "not a silver bullet" in
  its own words, and it does not cover float16 or seeding.
- **Croci, Fasi, Higham, Mary & Mikaitis**, "Stochastic rounding:
  implementation, error analysis and applications" (R. Soc. Open Sci.
  9:211631, 2022, <https://pmc.ncbi.nlm.nih.gov/articles/PMC8905452/>).
  Authoritative on stochastic rounding's zero-mean errors and freedom from
  stagnation (§6.1). Excellent because it states the cost, worst-case bounds a
  factor 2 larger than round-to-nearest. Scope: §8.5's PDE bounds rest on
  errors "independent in space and mean-independent in time", so reusing one
  draw at every step breaks the assumption.
- **OCP Microscaling Formats (MX) v1.0**, §5.4.1 and §6
  (<https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>).
  Authoritative on block-scaled formats and the dot over them. Excellent
  because the format's designers specify it, and §6.1 leaves internal
  precision and order implementation-defined. Scope: §6 covers MX formats
  only, so NVFP4 and FP8 with float32 blocks fall outside it.

### Attention and block-sparse products

- **FlashAttention-2** (Dao 2023, <https://arxiv.org/abs/2307.08691>).
  Authoritative on tiled attention that keeps only the logsumexp between
  the passes (§3.1.1, §3.1.2). Excellent because the backward pass recomputes
  from that one statistic per row. Scope: its 1.7-1.8× causal gain compares
  the kernel against itself without a mask.
- **Rabe & Staats**, "Self-attention Does Not Need O(n²) Memory" (2021,
  <https://arxiv.org/abs/2112.05682>). Authoritative on attention by chunks
  with recomputation, in memory below O(n²). Excellent because §5.1 and
  §5.2 measure the price, 8-13% forward and 30-35% forward and backward. The
  second figure is "expected due to the use of checkpointing". Scope: JAX on
  TPUv3, so the slowdown does not transfer to a host.
- **vLLM** (Kwon et al., SOSP 2023, <https://arxiv.org/abs/2309.06180>).
  Authoritative on a block table with reads that stop at each sequence's
  length. Excellent because a query's read is bounded by its own blocks
  (§4.1, Eq. 4) and a batch mixes lengths (§5.1). Scope: a read that runs
  to the table's width keeps vLLM's table and loses that per-sequence
  bound.
- **MegaBlocks** (Gale, Narayanan, Young & Zaharia,
  <https://arxiv.org/abs/2211.15841>). Authoritative on mixture-of-experts
  computation as block-sparse operations. Excellent because its abstract
  names the failure it prevents, dropping tokens or padding to capacity.
  Scope: §5.2 pads each group of tokens to a multiple of 128, and the
  block-sparse products index the weights by block, so it does not support
  grouping without padding.

### Iterative solvers

- **Kelley**, *Solving Nonlinear Equations with Iterative Methods* (SIAM
  2022), Ch. 1 §1.5, notebook text
  (<https://github.com/ctkelley/NotebookSIAMFANL/blob/master/SIAMFANLCh1.ipynb>).
  Authoritative on which stopping rules have theory. Excellent because it
  separates a step test, which has theory only for superlinear methods, from
  practice such as a safety factor, and ends at x_{n+1}. Scope: the residual
  as a rate indicator holds "in most cases", and the stagnation passage is
  §1.2.1.

### Designing for simplicity

These are authoritative on structure: how few concepts a system takes.

- **Project Oberon** (Wirth & Gutknecht,
  <https://people.inf.ethz.ch/wirth/ProjectOberon/PO.System.pdf>) and
  Wirth's **"A Plea for Lean Software"** (IEEE Computer 1995,
  <https://cr.yp.to/bib/1995/wirth.pdf>): a whole system one person can
  understand, published in full with its reasoning; the closest match to
  raven's constraint. Scope: its rule against hidden state is about
  interactive state the user can't see.
- **Plan 9** (<https://9p.io/sys/doc/9.html>,
  <https://9p.io/sys/doc/names.html>): a few uniform concepts (files, name
  spaces, one protocol) in place of many special-purpose ones, with its
  regrets published.
- **Lua** ("The Evolution of Lua", HOPL III,
  <https://www.lua.org/doc/hopl.pdf>; CACM 2011,
  <https://www.lua.org/doc/cacm2011.pdf>): mechanisms over policies, and a
  record of what was refused and removed. Scope: it shipped compatibility
  aids, so it does not support a rule against shims.
- **Lampson**, "Hints and Principles for Computer System Design" (2021,
  <https://arxiv.org/pdf/2011.02455>), the revision of his 1983 "Hints":
  "a lower layer is not allowed to call a routine in a higher layer"; a
  layer can be a client of several hosts; bypassing layers complicates the
  system.
- **Parnas**, "On the Criteria To Be Used in Decomposing Systems into
  Modules" (1972,
  <https://www.win.tue.nl/~wstomv/edu/2ip30/references/criteria_for_modularization.pdf>),
  with **Dijkstra's THE** system (EWD196,
  <https://www.cs.utexas.edu/users/EWD/ewd01xx/EWD196.PDF>): a module hides
  a design decision; a level may use any lower level.
- **Saltzer, Reed & Clark**, "End-to-End Arguments in System Design"
  (1984,
  <https://web.mit.edu/Saltzer/www/publications/endtoend/endtoend.pdf>):
  put a function where it can be done completely. Scope: placement for
  performance and reliability; rig's ordering is a correctness promise,
  not an end-to-end trade.
- **BLIS** (Van Zee & van de Geijn, ACM TOMS,
  <https://www.cs.utexas.edu/users/flame/pubs/blis1_toms_rev3.pdf>): dense
  linear algebra from a small set of kernels. Scope: a small kernel set
  that grows only by measured gaps; BLIS's own triangular solve ran at
  60-90% without a fused kernel, and it later added a small-matrix path.
- Secondary: **Hoare**, "The Emperor's Old Clothes" (Turing lecture), on
  simplicity as the price of reliability; **Iverson**, "Notation as a Tool
  of Thought" (<https://www.jsoftware.com/papers/tot1.htm>), on a small
  vocabulary. Scope: Iverson prefers phrases over new primitives, which
  argues against adding familiar names as mechanisms.
