(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels of {!Nx_kernel.S} for Apple GPUs of the Apple7 family.

    [computes_on d] is [true] iff [d]'s processor ({!Rig.arch}) is ["Apple7"],
    the GPU of an M1, M1 Pro, M1 Max or M1 Ultra, and this library's metallib
    loads on [d]. Its kernels are checked on that family alone, so it computes
    on no other. The first call for [d], of [computes_on] or of a kernel, loads
    the metallib, and a load that fails answers [false] from [computes_on] and
    [Declined] from a kernel, as every later call does. [computes_on d] raises
    {!Rig.Lost} if [d] is lost. Kernels claim through {!Nx_array.door} until
    their work is submitted, and answer [Done] once it is on [d]'s timeline; on
    a lost [d] they raise {!Rig.Lost}, as the door does. A call of many launches
    may return only once most of them have run, and its arrays stay claimed
    until then. Past its first submission, a call of one submission or several
    raises only {!Rig.Lost}, which the commit after a submission ({b Time}) may
    raise too, or the host's [Out_of_memory]; a destination it writes, donated
    or given, then holds unspecified values. Several domains may call its
    kernels concurrently. An operand another domain writes while a call runs
    gives unspecified results.

    A call's first launch of a GPU function on [d] waits while Metal compiles
    the function into a pipeline, and starts compiling the other functions of
    its family, those of calls like it, without waiting for them. While they
    compile, Metal starts and completes the process's GPU work later: on an M1
    Max, where Metal's cache holds none of a family's pipelines, a process's
    first call of the family took up to about 0.3 s, and each call after it up
    to about 0.13 s until the family was built. A call that weighs its launch
    against another function's, such as a transposed cast against a map's tiles,
    waits for that function's pipeline too. A call of several submissions waits
    so for the pipelines of all of them before its first. A contraction's family
    is its operands' dtype and [b]'s layout ({!section-contractions}). The first
    time a machine runs this version of them, a family's functions compiled
    together took about 2.6 s for maps, 0.5 s for casts ({b Time}), 3.7 s for
    reductions, 0.6 s for direct reductions along a row, 6 s for scans, 6.5 s
    for loops of any mode whose indices, or an operand's span of elements, reach
    [2{^32}], and 22.7 s for attention's ({!section-attention}), whose first
    call waits up to 1.5 s for its own function. Metal keeps pipelines across
    processes. On an M1 Max, a process's first call, a [float32] contraction of
    64 rows, columns and terms, takes about 220 ms where Metal's cache holds
    none of its pipelines and 150 ms where it holds them; the first call of
    another family, about 45 ms and 5 ms; and a call whose pipelines are made,
    0.3 ms. A map's tiles, its padded loads and its direct maps of 32 bits, of a
    narrow float or of bytes have functions of their own, so that a map that
    takes none of them runs none of their code: they hold 0.66 MB of the
    metallib's 9.5 MB, and maps compile about as fast for them. The matrix
    routines' own [float32] tiles hold 0.24 MB of it, in two sets of 0.12 MB:
    the negated tiles, which a routine's negated product alone takes, and the
    tiles that read a routine's options, which only a product takes that fills
    one triangle of its output or starts its first range of sums from that
    output.

    A kernel's work on [d] is one submission ({!Rig.Submission}) of
    {e launches}, each of one GPU function of the metallib, which run in order,
    or several such submissions in order ({b Time}). A kernel allocates
    {e words} on the OCaml heap, a number its operands' extents do not change,
    and keeps none once it returns but the submissions it keeps for later calls
    on [d]: for each GPU function, those of at most 64 calls or parts of calls
    that launch it first; of the Fourier transforms' specialised steps
    ({!section-fft}), the calls or parts holding at most 256 steps among them;
    and the plans of maps, at most 2 MiB of them, none of over 8 KiB. A map of
    the same program value as an earlier map, over operands of the same dtypes,
    layouts, paddings, fills and placement in memory, and of the same operands
    sharing buffers, takes the earlier map's plan, and allocates a few words for
    each operand in place of its planning's thousands. A call's values, its
    launches' grids and words, go into blocks that rig keeps for the life of the
    process ({!Rig.submit}): [32 + 16 w] bytes for each launch of a GPU function
    of [w] words, in as many sets as submits were ever under way at once, each
    as large as the largest submission it served. The exceptions are the calls
    that split into more launches than their section keeps ({b Time}), whose
    launches, and so whose words, grow with their extents: each takes a
    submission of its own, which no later call reuses. A call's plan holds host
    memory for each launch: about 200 bytes, 8 more for each of its words and 40
    for each buffer it addresses, so about 47 MB for 10{^ 5} launches of 16
    words and 4 buffers. On a Mac that memory is the GPU's too, so where a
    section states the largest array a call takes, it states one that leaves
    that room. Beside its arrays, a kernel takes {e workspace}, memory on [d]
    for its temporaries, {e tickets}, zero 32-bit words its groups count on, and
    {e constants}, tables of bytes that no launch writes, of the sizes its
    section states. It keeps at most 64 MiB of workspace, 4 MiB of tickets and
    16 constants of at most 64 MiB together for later calls on [d], and a call
    that needs more takes buffers of its own, which return to [d] once its work
    is done. Kept memory that a later call replaces returns once collected
    ({!Rig.section-reclaim}), since another domain's call may still use it. A
    kernel raises {!Rig.Out_of_memory} where [d] lacks that memory.

    {b Time.} Metal ends a command buffer that runs too long and the device is
    lost: every later call on [d] raises {!Rig.Lost}, and after repeated losses
    macOS may revoke the process's access to the GPU. Each launch's estimated
    time on the slowest GPU of the family, an M1 of 7 cores, is at most 50 ms,
    and longer work runs as several launches. The estimate bounds a launch
    within its submission, and is meant to err high there; the submission's own
    start, about 21 us of a lone call's 25 us on an M1 Max, is not estimated. It
    adds the time of the launch's dispatch, 1.37 us at most for an empty kernel
    in a chain of dependent dispatches on an M1 Max, the time of its memory
    traffic at 58 GB/s, 0.85 of the M1's bandwidth, counting every 128-byte line
    its operands' elements could touch wherever they lie, once for each run of
    touches a line or less apart, so that a line read again after the launch has
    moved others counts again, the time of its work over the lanes the launch
    fills at 0.91 of a [float32] operation a lane and a cycle, the rate
    independent [fma]s reach on an M1 Max, at the GPU's sustained clock, and the
    time of its longest chain of dependent steps for each wave of the groups its
    cores hold at once, as many as Metal lets a group of the kernel have
    threads. A dependent step's time, a dependent read's and a padded load's
    decode are measured, on an M1 Max; the operations each other section weighs
    are estimates. A call whose launches are estimated to take longer than 10 s
    together, or of more than 10{^ 5} launches, runs as several submissions in
    order, each of at most 10 s and 10{^ 5} launches. Once a call has submitted
    work estimated past 39 ms, it commits the GPU's work ({!Rig.Point.reached}),
    so that a command buffer holds at most 20 s of one domain's calls.

    {b Float64 and complex128.} The family has no float64 arithmetic. A kernel
    declines every call that computes on a [float64] or [complex128] value:
    arithmetic, casts between them and another dtype, comparisons and extremes.
    Kernels that move bits alone compute at those dtypes.

    {b Subnormal numbers.} At [float32] and [bfloat16], every kind of
    {!Nx_kernel.Prog} that computes a float reads a subnormal operand as a zero
    of its sign, and gives a zero of the exact result's sign where that result
    is nonzero and below [2{^-126}], the least normal number, in magnitude. A
    kind whose result is an integer value, a boolean or an operand's bits is
    exact, subnormal operands included: [Neg], [Abs], [Sign], [Floor], [Ceil],
    [Round], [Trunc], [Maximum], [Minimum] and the comparisons. A reduction's
    combines follow the same rule: a sum or a product combines as [Add] or [Mul]
    does, an extreme or an [Arg] as [Maximum] or [Minimum]. [complex64] keeps
    the bounds of {!Nx_kernel.Prog.section-complex_results} where no part of its
    computation underflows. [float16], the 8-bit floats and [float4_e2m1fn] keep
    subnormal operands and results, but their reductions and scans combine in
    [float32], and those combines follow [float32]'s rule: a product of tiny
    terms, reduced or scanned, whose running value falls below [2{^-126}] is a
    zero. So a scan of [float8_e5m2] terms near [2{^-15}] whose ninth product
    lies below it, then an infinity, gives NaN where a backend that keeps the
    product gives the infinity.

    Each section below states, for a family of kernels, what it computes, how
    its float results associate and what bounds them, and the memory it takes. A
    kernel of no section declines every call. *)

(** {1:moves Moves}

    [map] copies a program each of whose outputs is an operand's bits, the
    operand itself or a [Copy], a [Bitcast], or a [Cast] of it into its own
    dtype or between integers of one width, of a [Plain] load, at every dtype of
    a byte or more, into C-contiguous destinations; it runs every other program,
    and one into a destination that is not C-contiguous, as a loop ({!loops}).
    It copies an output in pieces, one launch each, within the time each launch
    allows. An operand whose elements lie at consecutive positions is copied as
    bytes, a thread an aligned block of 16 bytes of the destination, from one
    aligned load of 16 bytes where the source's bytes lie at the destination's
    address modulo 16, and from two otherwise; a block whose loads would leave
    the run or its launch's part copies its bytes one at a time. 16 bytes weigh
    128 operations. An operand whose innermost axis, once adjacent axes that
    step alike are merged, is a row of consecutive elements of 16 bytes or more
    beside other axes is copied as rows of bytes: a thread an aligned block of
    16 bytes of the destination, from at most two rows, each read by one or two
    aligned loads of 16 bytes where they lie within the bytes the operand's
    layout reaches, its bytes one at a time otherwise. A block weighs 128
    operations, and for each axis before its row 64 where the copy's byte
    indices fit 32 bits and 4096 past them. An operand of rows of more than 7
    axes once merged takes a constant of 24 bytes an axis. Any other operand is
    copied an element a thread: an element weighs 128 operations, and for each
    axis of its operand past the first once adjacent axes that step alike are
    merged, 64 more where the copy's indices fit 32 bits, a division by a
    multiplier the host gives, and 4096 past them, a 64-bit division the GPU
    emulates. Where they fit, its launches run a kernel of the element's unit of
    copy, 1, 2, 4 or 8 bytes, which holds neither the 64-bit path nor the other
    units'. A call of more than two launches an output takes a submission of its
    own. It takes no ticket, and takes a constant of 24 bytes an axis for each
    operand of more than 8 axes once merged. An output whose operand shares
    memory with another output's destination first copies the operand into
    workspace: the elements between the first and the last its layout reaches,
    or its elements in order where they are fewer, so that a broadcast operand
    takes only the elements it reads. *)

(** {1:loops Loops}

    [map], [reduce] and [scan] run a program's kinds through the functions of
    nx_kinds.h that nx.cpu's kinds call, so each gives nx.cpu's bits at every
    dtype, but where an operand or a result is subnormal
    ({b Subnormal numbers}). Every kind computes in its own dtype's precision or
    in [float32] for a float narrower than it, rounding once into the narrow
    float.

    {b Association.} A reduction folds its terms in nx.cpu's lane order: each
    output's terms in blocks of 1024, term [16 j + l] of a block folded in order
    into lane [l] of its 16, then the lanes and then the blocks combined by the
    pairwise tree. A scan chains each slice's terms in order, in chunks of 4096,
    each chunk from its slice's carry and the totals of the chunks before it,
    folded in order. [Logsumexp] and [Moments] reduce as nx.cpu's: the maximum
    or the mean first, then the terms' exponentials or squared deviations from
    it; and they scan by {!Nx_kernel}'s odd/even network, each result the joins
    of the same states nx.cpu's are. So reductions and scans have nx.cpu's bits,
    but where a combine underflows.

    {b Memory.} A reduction whose output has more than 16 blocks takes 32 bytes
    of workspace for every 16 blocks of each output, and a ticket for each
    output; a [Logsumexp] or [Moments] that takes its passes in launches of
    their own takes 4 bytes for each output more. A reduction whose group's
    terms run past a launch takes 32 bytes for each thread of a launch's groups
    more. A scan of more than one chunk a slice takes 32 bytes of workspace for
    each chunk and 32 for each slice, and one of a chunk a slice whose chunk
    runs past a launch 32 bytes for each slice. A [Logsumexp] or [Moments] scan
    of more than one block a slice takes 12 bytes for each block and 24 for each
    level of its blocks' network, one more than the bits of a slice's blocks,
    for each slice; a [Moments] scan takes 4 bytes for each slice more. A tiled
    map's group takes 4,224 bytes of threadgroup memory a staged load past the
    values it holds at once; a direct map's, which holds none, takes 16 bytes
    where it stages none. A loop's launch holds its parameters, the walk of its
    axes, its operands' strides and the program's tape, in 672 bytes; past them,
    it takes a constant of their bytes. Those bytes hold the operands' offsets
    and extents, so a call over other views of the same arrays takes a constant
    of its own, which the host writes and uploads before the call runs, and
    which replaces one a device keeps.

    A program too wide for a launch, past 12 operands and destinations, 10 for a
    reduction or a scan, whose workspace and tickets take two, or a group's 32
    KiB of threadgroup memory for the values it holds at once, runs in parts:
    each maps the values that later nodes read into temporaries in workspace,
    each of its node's dtype over the axes along which the node varies, and the
    parts give the bits one launch would. A map past a launch's buffers runs its
    outputs in passes, halved until each fits; a pass each of whose outputs
    moves an operand's bits, as {!moves} states, copies them as moves do.

    {b Time.} A loop's launch weighs each kind by its work on an element at its
    slowest, estimates all: 8 operations for a kind of a few, 512 for one of a
    polynomial, 4096 for a [complex64] transcendental and 1024 for a 64-bit
    integer division, each with 16 more for the tape's dispatch but in a direct
    map or reduction or a cast, and as many links of its chain, at most 512; and
    its chain by its loads' dependent reads. A map's padded load is charged the
    bytes its steps along the loop's axes move, as though each element were
    read, and its decode an element: 493 operations and 243 an axis of the loop
    where each element finds its coordinates, 257 an axis where it does at 4
    elements a thread, 81 an axis under quads, and, over 2{^ 32} indices or more
    or where a padded coordinate reaches 2{^ 31} or more, 2,179 and 677 for each
    64-bit division its coordinates take. A reduction's or a scan's padded load
    is charged a line an element. A reduction's group is charged the lines its
    outputs' terms touch together, so that outputs side by side share theirs as
    a row's terms do. A map whose loads, read first, take too much threadgroup
    memory for 4 elements a thread is also estimated with each load read at its
    first reader, and takes the plan of the lower estimate. A map is {e direct}
    where its program, as the tape runs it, reads at most three operands of 8,
    16 or 32 bits, not padded, each as it is, as a truth or decoded from
    [bfloat16] or [float16]; computes one kind, a [Where] or neither; and stores
    the result, as it is, wrapped to a narrow integer or encoded into [bfloat16]
    or [float16], into one destination of 8, 16 or 32 bits, over fewer than
    2{^ 32} indices of a loop of at most 8 axes, axes of one element dropped and
    two axes merged into one where every operand steps through them as one. It
    runs the tape's steps in registers, 4 elements a thread, and has the tape's
    bits. A direct map whose operands and destination are each of 32 bits, each
    taken and its result given as it is, whose threads take 4 consecutive
    indices, runs in a kernel of its own, which holds no path of the other
    widths and conversions: one for its long kinds, at [float32] [Recip],
    [Sqrt], [Exp], [Exp2], [Log], [Log2], [Log1p], [Expm1], [Sin], [Cos], [Tan],
    [Asin], [Acos], [Atan], [Sinh], [Cosh], [Tanh], [Erf], [Fdiv], [Mod], [Pow]
    and [Atan2], at [int32] and [uint32] [Recip], [Idiv], [Mod] and [Pow]; one
    for every other kind, a [Where] or neither, so that neither holds the
    other's code. Such a map whose destination's elements, in the order of their
    indices, lie one after another from its first, and each of whose loads'
    elements lie so or all at its first, runs, for each of the two, a kernel of
    its own that reads no axis, stride or position of its operands and holds no
    path that does; it reads or writes four consecutive elements at once where
    they lie on their size. A direct map whose operands and destination are each
    a [bfloat16], or each a [float16], each decoded and its result encoded,
    whose threads take 4 consecutive indices, runs likewise in kernels of its
    float's codec, one for its long kinds and one for the others. One whose
    operands and destination are each of 8 bits ([int8], [uint8] or [bool]),
    each element taken as it is or as a truth and its result as it is or
    wrapped, whose threads take 4 consecutive indices, runs likewise in a kernel
    of bytes but for its long kinds, which keep the general direct kernel. A map
    is a {e cast} where its program loads one operand, not padded, casts it into
    a dtype of 8, 16, 32 or 64 bits but [complex64], other than a cast that
    moves bits ({!moves}), and stores it into one destination over fewer than
    2{^ 32} indices. It runs in a kernel of the source's class and the target's
    width, 4 elements a thread, each operand's 4 as one vector where they lie on
    its size, takes no tiles, and has the tape's bits. Where its load is one a
    map would stage, as a transposed one, and the tape's tiles have the lower
    estimate, it runs as a map on them instead. A source of 8 or 4 bits, but
    [bit], takes its results from a table of its codes, the sources of one width
    sharing a kernel; any other source is a class of its own. A cast is never
    run as a direct map, though a lone decode from or encode into [bfloat16] or
    [float16] is one too. A map of 4 elements a thread over fewer than 2{^ 32}
    indices takes {e tiles} where a load would otherwise read a line for each
    element. Let [a] and [b] be the last two axes of extent past 1 of its loop,
    two axes merged into one where every operand steps through them as one. A
    load is {e staged} where it is plain, of 8, 16 or 32 bits, of stride 1 along
    [b] and of at least 32 elements along [a], either way, and [a] and [b] have
    32 indices or more. A group then takes 32 indices along [b] by 32
    consecutive along [a], and reads each staged load into threadgroup memory
    along [b] before the program runs; the estimate charges a tile's operands
    their lines along [a], and a staged load's along [b], and the staging 2
    operations an element. A map takes tiles where it reads no padded load, its
    group's values and stages fit in 32 KiB and its estimate with them is below
    its estimate without. No bit depends on the tiles. Loads of 64 bits, of
    [complex64] or of dtypes narrower than a byte, padded loads and the operands
    of reductions and scans are never staged. A reduction by monoids alone whose
    reduced axes are each of extent 1 runs as a map of its outputs' terms: each
    term at its carrier, a float or complex [Sum]'s added to [+0], which comes
    first, read from a constant at stride 0, then cast into its destination. So
    it has the reduction's bits, and it is a direct map where its term is its
    load and its dtypes are a direct map's. A [Monoid Sum], [Monoid Max],
    [Monoid Min] or [Arg] reduction, and each pass of a [Logsumexp] or
    [Moments], is {e direct} where its program reads one such operand and takes
    its element as the term, as it is, as a truth or decoded from [bfloat16] or
    [float16]; its value goes into a destination of 8, 16 or 32 bits of its
    carrier, or of [bfloat16] or [float16] from [float32], and an [Arg]'s
    position into its [int64] destination; and its loop has fewer than 2{^ 32}
    indices. Each lane reads 8 or 16 of its terms at once and folds them in
    registers in the tape's order, so it has the tape's bits. A direct reduction
    runs {e along a row}, in a kernel of its reduction and its element's dtype,
    where its carrier is [float32], its term a [float32] element as it is or,
    but for a [Logsumexp]'s or [Moments]' second pass, a [bfloat16] or [float16]
    one decoded, and its pass's reduced axes of extent past 1, at least one,
    step through every operand of the pass as one axis: in the order of [axes],
    each one's stride is the next one's times the next one's extent, no operand
    is padded and no node is a [Coord]. A pass holds a call's reductions, in
    halves while they need more planes or buffers than a launch holds, and the
    operands they read: the call's, or, in a program that runs in parts, the
    temporaries its parts wrote. Its lanes fold by plain additions and
    comparisons, and again by the tape's where a NaN or a zero could make them
    differ, so it too has the tape's bits. Its launch is weighed by its kernel's
    work on a term, not the tape's: a read every 8 terms, a decode and a plain
    fold, then the term read again and folded by the tape's, so its operand's
    bytes count twice; a [float16] term's count once, folded by the tape's
    alone. A direct [Logsumexp] or [Moments] whose outputs have at most 16
    blocks runs in one launch: each output's group folds its maximum or its
    mean, then reads its terms again and folds their exponentials or squared
    deviations from it; else its second pass runs in launches after its first's,
    reading the centers from workspace. The second pass's lanes fold by plain
    additions, and again by the tape's where a sum ends a NaN, so it has the
    tape's bits; its launch is weighed as a row's, an exponential at 32
    operations, the refold a term at a time as a chain of dependent steps, after
    its first pass's where they share it. A thread of any loop that would take 4
    elements a run takes 1 where a run of 4, with the most work any loop's mode
    adds to a run, would take a launch past 50 ms. A reduction's output whose
    groups take longer than a launch allows leaves its groups' states in
    workspace, and launches of a group each output merge them. A group whose
    terms take longer than a launch allows runs in launches that each take a run
    of each thread's terms, leaving its threads' states in workspace for the
    next. On an M1 Max, a [Logsumexp] of [float32] terms is declined from about
    2{^ 30} elements down 8192 columns, its outputs side by side and its terms a
    row apart, 2{^ 30.8} along rows of 4 terms and 2{^ 32.8} along rows of 8192;
    a [Moments] or a sum later.

    A scan by a monoid along an axis of extent 1 runs as a map too: each result
    is its term at its carrier, folded into the carry, which comes first, where
    the scan has one, else as a reduction's output of one term; a boolean term
    or carry is read as true or false. A scan cuts each slice into chunks of up
    to 4096 terms. [S] threads compute a chunk's terms, each every [S]th, and
    its first thread folds them in order, 16 links a term; one thread then folds
    a slice's chunk totals in order from its carry. [S] is the least power of
    two, from 16 where the scan's results step by one along the slice and a
    slice has 16 terms or more, else from 1, at which a thread's share of the
    chain of a term's program, its store aside, is at most the fold's 16 links;
    it stops at 256, and at the least power of two at or past the chunk's terms.
    Where either fold would take a launch past 50 ms by the estimate, it runs in
    launches that each take a run of its terms, 4 or 1 a thread, [S] halved down
    to its least where one run would take longer, leaving its state in workspace
    for the next. No bit depends on [S] or the launches. The chunk totals' fold
    is a chain: a 1-D scan of 2{^ 28} terms folds 65,536 totals one after
    another.

    A [Logsumexp] or [Moments] scan cuts each slice into blocks of [G] tiles of
    [256 v / W] terms, [v] the elements a thread takes, 4 or 1. Its group takes
    a block of each of [W] slices side by side, a power of two: as many as its
    tiles hold a slice's terms in, at least 32 where the results do not step by
    one along the slice, at most the slices. [G] is the most tiles, a power of
    two, that keep a block within 4096 terms and its group within a launch, down
    to one tile. A group scans its block's tiles in order in threadgroup memory,
    each tile's end carried by the network of the block's tiles before. A slice
    of more than one block takes three passes: every block's total, in parallel;
    the network over each slice's block totals, a thread a slice, about two
    joins a block, across launches where they take longer than one; then every
    block's results, in parallel, each from the network's result at the end of
    the block before. A [Moments] scan first maps each slice's first term, as a
    [float32], to the center its states deviate from. Where Metal lets a group
    have 256 threads, the fewest, a 1-D slice of 2{^ 24} terms takes 0.43 s by
    the estimate. A scan declines where one element's program takes a launch
    past 50 ms by these weights: past about 2,300 nodes of a polynomial kind, or
    1,800 of a [complex64] transcendental. *)

(** {1:contractions Contraction}

    [contract] computes a contraction, unsegmented or of [Rows] or [Contracting]
    segments: into a [float32] accumulator of operands, scales, [init] and [dst]
    of any dtype but [float64] and [complex128], and into a [complex64] one of
    unscaled [complex64] operands, [init] and [dst]; and into an integer
    accumulator, of unscaled operands, [init] and [dst] of any dtype but
    [float64] and [complex128]. Each element and [init] enter the accumulator,
    and the sum enters [dst], by their casts. An unsegmented or [Rows] call's
    float sums run on the matrix units. It declines every other call, and a
    [Contracting] one whose [dst] does not merge its groups with its batch axes,
    its row axes and its column axes each into one axis. Its [dst] may be of any
    layout, strided, offset, with negative strides or its axes stored apart,
    where every other kernel but the loops' takes only C-contiguous destinations
    ({!Nx_kernel.check_dst}). A [float16] or [bfloat16] product is exact in
    [float32]. A sum reads a subnormal element of [a], [b] or [init] of
    [float32] or [complex64] as a zero of its sign; an output of no terms is
    [init]'s cast, unread by a sum.

    The tiles of a [float32] call hold its operands' elements as [float16] where
    both operands are of [float16], [float8_e4m3fn], [float8_e5m2],
    [float4_e2m1fn], [int8], [uint8], [int4], [uint4], [bool] or [bit], dtypes
    all of whose values [float16] holds; else as [bfloat16] where both are of
    [bfloat16] or of those but [float16]; else as [float32]. That dtype is the
    call's {e stage}, [float32] for a scaled call. A call that neither the
    vector kernels (below) nor a loader takes on its own dtypes, of an unscaled
    operand not of its stage's dtype, is the call of that operand's cast into it
    ({!Nx_kernel.Prog.Cast}): the cast is made at each call, so a weight that
    many calls read is cheaper cast once. The call of the casts follows the
    rules below at its own dtypes, the vector kernels' and their association
    included.

    A scale ({!Nx_kernel.type-scale}) multiplies each element, rounded once into
    [float32]. A [Value] multiplies the element's cast by its own, a [float32]
    multiplication under {b Subnormal numbers}. An [Exponent] code [e]
    multiplies by [2{^e - 127}] through the element's bits, never through a
    float scale: code 255 gives NaN; NaN, the infinities and the zeros keep
    their value; an element subnormal in [float32] reads as a zero of its sign;
    the product is an infinity past the largest [float32], a zero of its sign
    below [2{^-126}], and exact otherwise.

    A scaled [float32] call whose tiles' rows ([a], or [b] where [n <= 16 < m])
    are unscaled and whose columns are a weight as a model stores it reads the
    weight and its scale where they lie, through tiles made for their dtypes,
    its {e loaders}: [float4_e2m1fn] by [Exponent] codes, and [int8] by [Value]s
    of [float16] or [float32], each with rows of [float32], [float16] or
    [bfloat16]. Each element times its scale is as above, so no bit depends on
    the loaders. A call of fewer than [2{^32}] terms takes them at every tile
    but 64 x 64 where each of the weight's runs fits its scale, and a weight
    narrower than a byte has no negative stride and its columns or its terms
    contiguous. Its batch axes and its groups' axis fit where the scale holds
    one block along them, or blocks of one element; its terms and its columns
    where each block is whole, the blocks along every axis of more than one
    element but the run's innermost are of one element, and the scale's strides
    merge over its blocks. A weight whose terms make two runs, the inner
    contiguous, as Q8_0's blocks of 32 quants between their scales, is read
    where it lies too, its terms along its columns, where its batch and free
    axes each merge.

    A [complex64] call computes its {e real form}, a [float32] call of [m] rows,
    [2n] columns and [2K] terms. Row [i]'s terms [2p] and [2p + 1] are [a]'s
    element [p]'s parts [ar] and [ai]; [b]'s element at term [p] and column [j]
    is the real block of terms [2p], [2p + 1] and columns [2j], [2j + 1] that
    holds [br], [bi] above [-bi], [br]. Column [2j] is the real part's sum, of
    [ar·br] then [ai·(-bi)] a product, and column [2j + 1] the imaginary part's,
    of [ar·bi] then [ai·br]: the products' real terms in the order
    {!Nx_kernel.section-contraction} numbers them.

    An integer call is exact modulo [2{^w}], [w] its accumulator's bits: each
    element and [init] enter the accumulator by their casts, the products and
    their sum wrap, and the sum enters [dst] by its cast. Where [a] and [b] are
    both integers of 8 bits or fewer or booleans, the matrix units compute it as
    a [float16] call of its extents: each product of two such values is below
    [2{^16}] in magnitude, so a sum of 256 of them is an integer below [2{^24}],
    which [float32] holds, and each block of 256 terms of a range sums at
    [float32], then adds into a 64-bit integer sum. Other integer calls, of
    wider integers or of floats, take a thread an output for each range of k,
    which reads each element at its own dtype, [S] as tiles' with [T] the groups
    of 256 outputs and [G] 384. Neither changes a bit by its association.

    A [Rows] call's id is read as an [int64] and tested [0 <= id < g] whole: an
    id below 0 or past [g], [2{^32} + 1] included, puts its row in no group. A
    prepass puts each row in a bucket, its batch element's group or the bucket
    of no group, and lists the rows bucket after bucket. Tiles of 32 x 64 where
    [m >= 16 g], [m] a batch element's rows, else of 16 x 64, each hold rows of
    one bucket in any order and multiply its group's [b]; the tiles of the
    bucket of no group compute no product, so their outputs are [init] alone, or
    [+0]. All batch elements' rows take at most
    [⌈batch m / R⌉ + min (batch g + 1) (batch m) - 1] rows of tiles of [R] rows,
    [T] being those times the tiles of 64 the columns take, [G] 288 for 32 x 64
    at [float32] and [float16] and 544 at [bfloat16], and as 16 x 64's
    otherwise. A row's sum is the association above whichever tile holds it, so
    no bit depends on the ids or on the other rows' groups. An integer [Rows]
    call takes the thread an output, which reads its row's group's [b].

    A [Contracting] call takes a thread an output, as the other integer calls,
    at every accumulator. A prepass lists each batch element's contracted
    indices by group, each group's in increasing order, after testing each id as
    a [Rows] call's. An output's terms are its group's products in that order,
    numbered from 0, and its [K] the group's count: [r] and [S] below follow
    from that count, each range of the group's own terms, with [T] the groups of
    256 outputs of the call and [G] 384. A [float32] call's sums are fused
    multiply-adds of the elements' casts, or of their products with their
    scales; a [complex64] call's thread computes a column of its real form,
    whose outputs [T] counts. An output of a group without terms has no
    products. So no bit depends on the ids but through the output's own group's
    count.

    {b Association.} A call computes tiles of outputs: of 16 x 64 where the
    rows, or the columns where [n <= 16 < m], are at most 16, [m] and [n] the
    rows' and columns' extents multiplied; else of 64 x 64 where the call has at
    least 96 of them, the batch elements times the tiles of 64 its rows and
    columns take; else of 32 x 32. An output's [r = ⌈K / 32⌉] spans of 32 terms
    go in [S] ranges, range [s] taking the spans from [⌊s r / S⌋] to below
    [⌊(s + 1) r / S⌋]. With [T] the call's tiles, [S = 1] where [2 T >= G], else
    [S = max 1 (min (min 64 ⌊r / 8⌋) ⌈G / T⌉)]. [G] is 192 for tiles of 64 x 64
    at [float32] and [float16], 384 at [bfloat16]; 416, 384 and 736 for 32 x 32
    at [float32], [float16] and [bfloat16]; 160, 352 and 352 for 16 x 64, at the
    call's stage. Each range sums from [+0] in increasing k, one fused
    multiply-add a term. The ranges' sums add in increasing k, rounding each
    addition, then [init], and the result rounds once into [dst]. So a call's
    bits depend on its dtypes, whether it is scaled and its extents alone. A
    [complex64] call follows these rules as its real form, and its tiles' rows
    are always [a]'s. Each range is one running sum, of up to [K] products, so
    [β(n)] is [2n + 1] ({!Nx_kernel.section-bounds}). A term passes at most [K]
    roundings, and one more for each operand of a [Value] scale, so the error
    factor is [γ(K + σ)], [σ] those operands, within [γ(β(K))]. A [complex64]
    part is within [γ(2K)] of the sum of its [2K] terms' magnitudes, so its
    error factor is [√2 γ(2K)], within [√2 γ(β(2K)) = √2 γ(4K + 1)]. Each
    multiply-add and addition follows {b Subnormal numbers} at [float32].

    A [float32] call takes the {e vector kernels} instead where its wider
    operand is of a key and its other operand of [float32], [float16] or
    [bfloat16], or scaled. The keys and their bounds on a call's rows are: 11
    for [float4_e2m1fn] by [Exponent] codes and 3 for [int8] by [Value]s of
    [float16] or [float32], 16 for both in a [Rows] call; 2 for unscaled
    [float16] and [bfloat16]; 1 for unscaled [float32]. Their bounds on [K] are
    [2{^23}] for unscaled [float32], [2{^22}] for unscaled [float16] and
    [bfloat16], [2{^19}] for the scaled keys, [2{^18}] in a [Rows] call. The
    wider operand is [b] where the rows number at most [b]'s bound a batch
    element, the columns at least 256 and [K] at most [b]'s bound; else, but in
    a [Rows] call, [a] where the columns number at most [a]'s bound, the rows at
    least 256 and [K] at most [a]'s bound. A call without terms takes the tiles'
    path. There an output's [r = ⌈K / 32⌉] spans of 32 terms are dealt to
    [S = min 32 (2{^⌊log₂ r⌋})] lanes, span [j] to lane [j mod S]; each lane
    sums its terms from [+0] in increasing k, one fused multiply-add a term; the
    lanes' sums combine by halving, lane [l] adding lane [l + h]'s for
    [h = S/2, ..., 1]; then [init], and one rounding into [dst]. A [Rows] row of
    no group is [init] alone, or [+0]. So a vector call's bits depend on its
    dtypes, whether it is scaled and [K] alone, and change where its rows or [K]
    cross a key's bounds or its columns 256: these constants are part of the
    contract. The error factor is the tiles' [γ(K + σ)].

    {b Memory.} An operand is first copied in C order into workspace, but where
    loaders read it in place, unless its batch, free and contracting axes each
    merge into one axis, its terms contiguous where its free axes give the
    tiles' rows ([a], or [b] where [n <= 16 < m]), its terms or its columns
    contiguous otherwise. So is an [init] whose axes do not merge or that shares
    memory with [dst] through another layout. A [complex64] call's [a] must have
    its terms contiguous to be read in place. Where [S > 1], the launches cut k,
    there is an [init], [dst] is of neither [float32] nor the operands' dtype or
    shares memory with an operand, or a [complex64] [dst]'s columns are not
    contiguous, each range's sums take 4 bytes of workspace an output, 8 a
    [complex64] one, and a last pass adds them into [dst]. An integer call's
    sums take 8 bytes an output, and the last pass is skipped only where the
    matrix units store into an integer [dst] of the accumulator's 32 or 64 bits.
    An operand narrower than a byte is widened into workspace in C order, a byte
    an element, where the matrix units read it or its axes do not merge, but
    where loaders read it, and so is such an [init] whose axes do not merge. A
    scaled operand's products with its scales are written into workspace in C
    order, 4 bytes an element, but where loaders read it. The casts of a call
    take one buffer of their distinct elements at the stage's dtype, an axis of
    stride 0 being one element there, of at most 64 MiB: past it the call is cut
    into pieces, each a call of a box of the outputs (ranges of batch elements,
    of [a]'s rows and of [b]'s columns, every term) that keeps the call's
    exchange, tiles, [S] and vector kernels, so no bit changes. Pieces take
    whole batch elements first; else, in each batch element, the operand whose
    cast takes at most 32 MiB is cast once and the other is cut, or both are cut
    within 32 MiB each, the smaller cast again for each piece of the larger:
    [p - 1] more casts of it for [p] pieces of the larger. A piece holds at
    least a row of [a] or a column of [b] with its terms. Where such a line's
    cast passes its share (what the other's cast leaves, or 32 MiB), the piece's
    call casts it in chunks of terms into a band of at most its share in
    workspace, before the tiles' launches of each chunk, which read the band in
    its place: each term of the line is read once more, and each chunk reads and
    writes its sums, 4 bytes an output a range. A call the vector kernels take
    casts each such line whole, and any call a line whose terms make several
    runs of strides: its buffer holds [k] elements of each such line at the
    stage's dtype, and the call raises {!Rig.Out_of_memory} where [d] lacks
    them; beside such a line, a line of the other operand takes a band of what
    it leaves of 64 MiB, and a band of a span of terms a range at least. Each
    piece casts and computes in submissions of its own, after the piece before,
    reading the operand not cast again: a 12 GB [int8] weight cut beside an [a]
    of 32 MiB reads about 40% more bytes than the call uncut, and a [Rows] call
    runs its prepass each piece. A piece's casts, its call and its copy each run
    as several submissions where their launches pass 10 s, as a call does.
    Without terms, a last pass alone writes [dst]. A [Rows] call's [b] is copied
    with its groups after its batch where its batch, contracting and free axes
    do not each merge, and its ids where their batch axes do not. Its prepass
    takes 4 bytes a row of a bucket, and for the tiles 4 more of a list, 12 a
    tile row of the bound and 8 a bucket. A [Contracting] call's prepass takes 4
    bytes a contracted index of a batch element, 8 a group of a batch element
    and 8 more a batch element, and 4 a group for each part of a batch element's
    indices, in parts of at least 4096 indices, of at most [2{^22}] such counts
    together; its float sums take 4 bytes an output a range, its real form's for
    [complex64], the most ranges a group of [K] terms takes. A call whose [dst]
    does not merge its batch axes, its row axes and its column axes each into
    one axis computes in pieces as above, each into a temporary of [dst]'s dtype
    in C order, then copies it into [dst]: the temporary takes the pieces'
    buffer, or half of it beside casts, which take the other half, and a piece
    is cut further, along its batch, then its rows, then its columns, down to
    one output, until its temporary fits. Each piece's copy reads and writes its
    outputs once more and is a launch of its own. A call but a [Contracting] one
    of [2{^31}] outputs or more computes in pieces as above, each of fewer, cut
    the same way. A contraction takes no ticket. A call of more than 8 launches
    takes a submission of its own.

    A vector call reads its wider operand where it lies where its terms are
    contiguous or form Q8_0's blocks of a multiple of 32 contiguous terms, or
    where its columns are contiguous and its terms merge. A [float4_e2m1fn] one
    must also start at a whole byte, with even strides and none negative, and
    where its terms are contiguous, a scaled one's scale blocks along k must be
    a multiple of 8 terms or all of them. Otherwise it copies an unscaled one in
    C order, and writes a scaled one's products with its scales, 4 bytes an
    element: Q8_0's blocks stored with their columns contiguous take that pass,
    about 0.7 ns an element of the weight on the M1 Max. Its other operand is
    copied where its terms are not contiguous, and written as its products where
    scaled. Its sums take 4 bytes of workspace an output, and a last pass adds
    them into [dst], where there is an [init], [dst] is neither [float32] nor of
    the dtype of an unscaled [float16] or [bfloat16] other operand, or it shares
    memory with an operand; a [Rows] call's with an [init] also takes 4 bytes a
    row for its rows' buckets.

    {b Time.} The tiles run in blocks of [c] tiles: of 4 rows of tiles by all
    the columns of tiles where both number 8 or more, else of one row. A core
    holds as many tiles' groups at once as its 64 KiB of threadgroup memory and
    its 3072 threads allow. A launch weighs its tiles' reads of [a] and [b], the
    tiles a wave holds sharing the rows and columns they read, 4 bytes an output
    of a range's sum, twice where it resumes one, 2 operations a product, 4 for
    tiles of 16 x 64, and a chain of a step a term. A loader weighs its weight's
    bytes an element, and 512 operations an element it holds. A cast is a map of
    its own, before the call of the casts. A loader over a weight of terms
    contiguous reads 4 to 16 terms at a time, with one scale, at least where the
    weight's blocks and its scale's blocks along k are of a multiple of 16 terms
    and at least 32, as Q8_0's and MXFP4's are; it reads other weights an
    element at a time, slower. A [float4_e2m1fn] weight's runs of 8 elements are
    read as words where its first element starts on 4 bytes and its steps are
    whole words: 16 rows of MXFP4's 5760 x 2880 stored k x n take about 0.42 ms
    on the M1 Max, and 0.66 ms where the weight starts 2 bytes into a word.
    Launches take whole ranges in windows of the most tiles within a launch's
    time, a multiple of [lcm (G, c)] tiles. Where one such unit takes longer,
    each launch takes one unit over a chunk of k, a multiple of 32 terms, in
    increasing k, resuming the sums before it; where those take longer, units of
    [c] tiles, then of one. A scaled operand's products take a thread an
    element, in launches of the most elements within a launch's time. An integer
    call off the matrix units takes its outputs in windows of the most groups of
    256 within a launch's time, or one group a launch over chunks of k. A [Rows]
    call's prepass is one launch of one group, which reads the ids and then the
    buckets twice for each chunk of 2048 buckets, and the call declines where it
    takes longer than a launch's time; a tile of its weighs its own reads of
    [b]. A [Contracting] call's prepass counts and lists its indices in launches
    of a group a part of them, and scans each batch element's counts in one
    group, 2048 counts at a time; the call declines where one launch takes
    longer than a launch's time. A vector call's threadgroups of 128 threads
    each take a column group, [4 (32 / S) c] columns of [c] a lane, 1 for
    [float32], 4 for [float4_e2m1fn] and 2 for the other keys, and a pass of
    rows: 2 for [float16] and [bfloat16] past one row outside [Rows] calls, else
    1. Over a weight of columns contiguous, its threadgroups are of 256 threads
    and 4 bytes of threadgroup memory a sum of each, at most 8, a lane holding
    16 bytes of an unscaled weight's adjacent columns and a pass one row, or 8
    adjacent columns of a scaled one and a pass 4 rows but for [Rows] calls and
    one row. A column group's passes run together, and its launches take the
    most column groups within a launch's time, each pass weighing a read of its
    columns' weight. No bit depends on the launches. A [float32] call of 4096
    rows, columns and terms takes 6 launches. *)

(** {1:fft Fourier transforms}

    [c2c] computes a [complex64] transform, [r2c] one of [float32] points into
    [complex64] bins and [c2r] one of [complex64] bins into [float32] points,
    whose every axis has [n] points of no prime factor past 7, or whose line
    runs Rader's or Bluestein's convolution: Rader's where [n] is a prime whose
    [n - 1] has no prime factor past 7 and is at most 4096, Bluestein's
    otherwise, of the 7-smooth length [m] from [2n - 1] to [2 (2n - 1)] that
    costs least, where [m] is at most [2{^20}], as it is for every [n] up to
    524,288; along an [r2c]'s or a [c2r]'s last axis of even [n], those of the
    line of [n / 2] points, which runs Bluestein's where not 7-smooth. A line of
    at most 4096 points runs in a group of threads, its points in threadgroup
    memory, in passes of DFTs, each pass past the first multiplying its points
    by roots of [n] first: its factors 2 first, three at a time as 8-point DFTs,
    then a 2- or 4-point DFT for those left, where the line then takes at most
    256 threads, each taking [⌊8 / R⌋] of a pass's [n / R] [R]-point DFTs, as
    every power of two up to 2048 does; else four at a time as 16-point DFTs,
    then a 2-, 4- or 8-point DFT; then its 7s, its 5s and its 3s. A longer line
    of [n = s M] points runs [M] lines of [s] points, multiplies each point by a
    root of [n], then runs [s] lines of [M] points, each the same way while [M]
    passes 4096, between transposes through workspace: at most [⌈log₂ n / 8⌉]
    lengths of lines, each of at most 4096 points. A line its layout does not
    keep in rows goes through workspace and back, its bits unchanged. At
    [n = 4096], where 16 lines or more lie side by side, one element apart, and
    a line's consecutive points lie at most 8 KiB apart, the three passes run
    instead as two launches of 16 lines a group, the first pass, then the other
    two, through workspace: a choice of speed, made from the layout, that
    computes the same bits. Lines of at most 2048 points that lie one element
    apart may run side by side, a group's lines' same points together, where
    that costs less than moving them into rows: the same bits too.

    A convolution runs its FFT of [m] points twice, each point of the first
    FFT's result multiplied by a kernel's between, then conjugated: in the
    line's group where [m] is at most 4096, else as longer lines, the chirps and
    the kernel's product in the transposes. Rader's, of [m = n - 1] points,
    loads point [g{^s}] as its point [s], [g] the least primitive root of [n],
    and gives bin 0 the sum of the first FFT's bin 0 and point 0, and bin
    [g{^-s}] the sum of point 0 and the conjugate of the second FFT's point [s].
    Bluestein's multiplies point [j] by the chirp [e{^-πi j² / n}] and pads to
    [m] points with zeros, and gives bin [k < n] the conjugate of the second
    FFT's point [k] times the chirp's [k]. The kernels and the chirps are
    rounded as the roots are.

    Along an [r2c]'s or a [c2r]'s last axis, a line of odd [n] runs as a complex
    line of real points: an [r2c] stores its bins up to [n / 2], and a [c2r]
    completes its bins by [X{_ n-k} = conj X{_ k}] as it loads them and stores
    the real parts. A line of even [n] runs as a line of [h = n / 2] complex
    points [x{_ 2j} + i x{_ 2j+1}]: an [r2c] splits its FFT's points [k] and
    [h - k] into bins [k] and [h - k], bin [h] from bin 0's pair, and a [c2r]
    makes its points [k] and [h - k] from bins [k] and [h - k] before its FFT,
    each pair with the root [w{_ n}{^ k}]: within the line's group, the root
    from one table, and past 4096 points [h], in launches of their own, a
    product of two. A [c2r] along several axes runs the others first, into
    workspace. The transforms decline those of other dtypes of an element or
    more, and those along an axis of more than [2{^32}] points, or whose line,
    the halved one along an [r2c]'s or a [c2r]'s last axis of even [n], is a
    Bluestein line of more than 524,288 points; one of no element writes
    nothing, and an [r2c] of no point stores bins of [+0]. Each part of a root
    is the [float32] nearest its exact value, unless that value lies within
    [2{^-98}] of a halfway point, from tables made in double-double arithmetic
    on the host.

    A line of [l] points other than 4096, a line of [n] points or one of a
    longer line's, or a convolution's FFT, runs a {e step}, a GPU function
    specialised to [l] and the line's convolution. A call's first use of new
    steps waits while Metal makes them, together: on an M1 Max, about 0.1 s for
    the first and 15 ms for each other, a convolution's about 0.25 s and 35 ms,
    and a process's first call about 0.2 s more; where Metal's cache holds a
    step, which it does not promise, about 1 ms. A call along [a] axes, each of
    at most [2{^32}] points, uses at most [4 a] steps. [d] keeps the functions
    of the 256 steps made last, whatever their use since; one that drops out is
    made again at its next use.

    A line of [n] points [x] whose parts lie below its threshold in magnitude,
    the least of those that apply to it: [(1 - 2{^-17}) (π/2) 2{^115}] for a
    complex 7-smooth line of at most 4096 points, [(1 - 2{^-17}) 2{^124} / n]
    for a real one or a longer one, [2{^108}] for a Rader line and
    [(1 - 2{^-17}) 2{^124} / m{^2}] for a Bluestein line, gives a result within
    [e 2{^-24} √n ‖x‖₂ + c n 2{^-126}] of its exact transform in norm, [‖x‖₂]
    the norm of its points, a [c2r]'s those of the line its bins complete, bin
    0's and bin [n / 2]'s imaginary parts dropped; [e = 5.35 log₂ n] for a
    7-smooth [n] and [91 log₂ n + 236] for another, but along an [r2c]'s last
    axis of even [n], [e(n / 2) + 11], and a [c2r]'s, [e(n / 2) + 20]; and
    [c = 13.29], [42] for a Rader line and [76 √n + 72] for a Bluestein line,
    along an [r2c]'s or a [c2r]'s last axis of even [n] the [c] of the line of
    [n / 2] points, which covers the flushing of its split or its pre-pass too,
    and which is never a Rader line: Bluestein's convolution computes it where
    [n / 2] is a prime past 7. The second term bounds the flushing of subnormals
    ({b Subnormal numbers}): each operation on a part errs by less than
    [2{^-126}] where it flushes, twice that for a product by a root, and the
    arithmetic that remains carries the error to the result multiplied by at
    most its norm; over the line's loads and each pass's products and levels of
    sums, its steps' in turn, that is [c n 2{^-126}], [c] a bound over every
    sequence of passes, finite since a pass's share falls with the points before
    it (7.52 at 4096, 12.98 at 2187). A convolution's [c] sums the same over its
    loads, its products, both FFTs and its sums, with the norm of the kernel's
    product and the FFT after it, which grows as [√n] for Bluestein's. No value
    a complex 7-smooth line of at most 4096 points computes, intermediate sums
    included, exceeds [4n/π] times its largest part in magnitude, so the exact
    values stay below [2{^128}] under [(π/2) 2{^115}], and the margin [2{^-17}]
    keeps their roundings finite. A real 7-smooth line's or a longer one's stay
    within [(2 + 2√2) n] times it: each value a sum over at most [n] points of
    at most twice a point's part each; an [r2c]'s split's two operands, from the
    sum and the rotated difference of its FFT's points [k] and [h - k], each
    within [√2 n] times it, as those points lie within [√2 n / 2] times it, and
    their sum twice a bin's real part; and a [c2r]'s pre-pass's points' parts
    within [2 + 2√2] times the bins'. A convolution's values stay below
    [2{^128}] under its own threshold. Rader's: its FFTs within [4m/π] times
    their points' parts, the kernel's within [√n / m] in magnitude, so within
    [(16√2/π²) m √n] times the line's largest part. Bluestein's: the chirped
    points within [√2] times it in magnitude, so the first FFT's values within
    [√2 n] times; the kernel's within [(2n - 1) / m]; the second FFT's values,
    partial sums included, and the chirps' products after it within
    [√2 n (2n - 1)], at most [(√2/2) m (m + 1)] times, below [2{^123.6}] under
    the threshold. A halved line's split at most doubles that and its pre-pass
    multiplies it by [2 + 2√2], below [2{^126}]. The line of parts [±m] whose
    bin 1 is greatest reaches [2{^128}] near [(π/2) 2{^115}], and past it a part
    of the result may be infinite or NaN. An [Inverse] transform is the
    conjugate of the [Forward] transform of the conjugated operand, bit for bit,
    signs of zeros included. A line's bits depend on its points alone, whatever
    its array's layout, batch or offset. *)

(** {1:sorting Sorts and numbering}

    [sort] and [unique] compute every dtype, every layout of their operand,
    direction and number of kept elements, and keys of any number of columns,
    into C-contiguous destinations ({!Nx_kernel.check_dst}). They use only
    integer arithmetic on the elements' bits, so their results are the
    contract's exact ones, subnormal numbers included.

    An element's key has [k] bytes: the element's, or one where it has fewer
    than 8 bits. A sort's launches depend on its slices' length [n]:
    - A slice of at most 4096 elements, 2048 of [complex128], sorts in a group's
      threadgroup memory: one launch.
    - A longer slice that keeps at most a quarter of that is selected: a launch
      zeroes the selects, a round of radix select takes a launch for each of the
      [k] key bytes and of the [p] bytes of [n - 1], a launch gathers the
      elements kept and one sorts them: [k + p + 3] launches.
    - Any other slice sorts by radix: a launch zeroes the digit counts, one
      counts the digits of every byte of the keys, then each byte takes a launch
      that moves the elements, after one that counts where a slice has several
      parts: [2 + k] or [2 + 2k] launches. A byte that every key of a slice
      shares is skipped within its launches.

    Where x's elements, of a byte or more, lie so far apart along each of its
    axes that no two share a 128-byte line, as along the first axis of every
    32nd column of a [float32] matrix, a sort copies its slices contiguous into
    workspace first, in a copy's launches, where its estimate is less that way.
    A launch whose estimate passes 50 ms runs as several, each over a range of
    its groups, and the call then takes a submission of its own. A sort is
    declined where it sorts in a block or by radix and its values or positions
    share a byte with [x], as where the values are [x].

    A numbering of [r > 0] rows of [w > 0] columns zeroes its bit words in one
    launch, sorts the rows a column at a time from the last, each column as a
    sort of 8-byte keys, then numbers them in four launches. A numbering of no
    row takes one launch. A numbering of more than 8 columns takes a submission
    of its own.

    {2:sorts_memory Memory}

    Workspace a call, [e] an element's bytes (one below 8 bits), [c] the block's
    4096 elements (2048 of [complex128]):
    - block: none.
    - select: [count * 1088] bytes of state, [count * 2112] from [2{^32}]
      elements a slice, and [count * c * (e + 8)] of candidates.
    - radix: [count * k * 256 * 8] bytes of digit counts, 16 from [2{^32}]
      elements a slice; [count * 4] of masks; [count * P * 256 * 4] where a
      slice has [P > 1] parts, 8 from [2{^32}]; and about [count * n * (e + 4)],
      [e + 8] from [2{^32}], for a buffer where [k >= 2] and a second where
      [k >= 3]. The second lies in the positions where a sort of 32-bit
      elements, below [2{^32}] a slice, keeps every element of slices whose
      positions follow one another.
    - numbering: about 64 bytes a row: 40 of order, numbers and first rows, 24
      of a column's sort buffers.

    A sort that copies its slices contiguous first takes [count * n * e] bytes
    more. A call whose workspace passes 64 MiB takes a buffer of its own. *)

(** {1:index Gathers, assemblies and folds}

    [gather], [gather_ranges] and [assemble] move elements bit for bit, at every
    dtype, from operands of any layout into C-contiguous destinations. Each
    moves its elements in {e units}, one a thread. Its axes are first merged,
    each into the one outside it, where the destination, the source and the
    positions all step alike, an element's bytes counting as an innermost axis:
    a C-contiguous [float32] [[4; 2]] is one axis of 32 bytes. For elements of a
    byte or more, the unit is then the widest of 16, 8, 4, 2 and 1 bytes that
    divides the addresses of both ends, each end's stride along every merged
    axis in bytes, and the innermost axis's bytes where both ends step one byte
    along it, else a byte; a take of whole rows moves 16 bytes a thread. An
    element of 1 or 4 bits is a unit of its own, stored by atomic operations on
    its 32-bit word, so that the elements beside it in that word, another
    array's included, keep their bits. A unit weighs 128 operations, 64 more for
    each axis past the first once adjacent axes that step alike are merged, 4096
    where a launch's units reach [2{^32}], and 512 more for an atomic store.
    Units go in launches each within the time a launch allows. A move of more
    than 4 axes once merged takes a constant of 40 bytes an axis. A call
    declines where one unit's launch would pass the time a launch allows.

    [gather_ranges] first writes, for each lead index, the running maxima of
    [ends] into workspace: 8 bytes a run and 16 a row, each group of 256 threads
    taking 4096 runs. A row of more than 4096 runs takes levels of the groups'
    maxima, 8 bytes each, and a launch a level that carries them into the level
    below. A thread a position of [dst] along the runs' axis then finds its run
    by halving the row's maxima, [⌈log₂ k⌉] reads for [k] runs, and writes its
    source position, which a gather then reads: 4 bytes a position where [x]'s
    runs' axis has fewer than [2{^31}] elements, else 8. The positions take at
    most 64 MiB of workspace: past it, the call writes and gathers them a chunk
    at a time, through the same 64 MiB.

    [assemble] writes the fill where its pieces do not cover [dst], as moves of
    a pattern in units: around the largest box of a piece whose ranges step by
    one, at most two boxes of [dst] an axis (along each axis, the indices before
    and after the box's, within the box's ranges along the axes before it), else
    over the whole of [dst]. It then moves each piece in the order given, a
    launch or more each, so that the last piece over an element gives it. A
    first piece that is [dst]'s own elements in place moves nothing. A fill
    around a box of [r] axes is so up to [2 r] launches, and a call of more than
    8 launches takes a submission of its own.

    [fold] computes at every dtype {!Nx_kernel.Prog.Add} takes but [float64] and
    [complex128], which it declines. A thread an element of [dst] finds the taps
    that read it and adds them from left to right in C order of their numbers,
    each addition {!Nx_kernel.Prog.Add}'s, so that a narrow float or a 4-bit
    integer rounds or wraps at each one: nx.cpu's association, and nx.cpu's bits
    where no [float32] or [bfloat16] operand or partial sum is subnormal. It
    finds a window's first tap that reads an element by a division by
    [step / dilation] and one by [step] where the dilation divides the step,
    else by at most [step / gcd (step, dilation)] divisions by [step], and each
    later one by an addition. A division is 32-bit, 64-bit where [dst] has
    [2{^32}] elements or more or a padded position or a step reaches [2{^31}].
    Its element indices, positions in [x], taps and tap numbers are 32-bit where
    each lies below [2{^30}], else 64-bit. An element of 1 or 4 bits is stored
    as a gather's is. Where an element's taps would take one launch past its
    time, the fold runs as launches of ranges of tap numbers, each adding its
    taps onto the sums the last left in [dst], so that the ranges leave every
    bit as one launch would. A fold takes a constant of 136 bytes an axis and a
    window, and 8 bytes a window for each range past the first.

    A fold without interior padding whose windows' taps each reach no further
    than their step reads each element of [dst] from at most one element of [x]:
    it moves those elements as [assemble] moves a piece, after a fill of [+0]
    around the box they cover where their indices follow each other along every
    axis, else over the whole of [dst]. Along an axis, the taps that land in
    [dst] are a first window's from some tap on, whole windows, and a last
    window's up to some tap: a move each product of these, at most [3{^w}] for
    [w] windows. Where their dispatches would take more than a launch's time,
    the fold kernel computes it instead. *)

(** {1:matrices Matrices}

    [cholesky], [lu], [solve_triangular] and [qr] compute at [float32] and
    [complex64] every call whose operands each have batch axes that merge into
    one strided axis. Every other call of them declines. {!section-spectral}
    states which calls of [eigh], [svd] and [eig] compute.

    Each routine computes on [a′], its operands with each subnormal element read
    as a zero of its sign, and its results are [a′]'s. Each element of
    [cholesky]'s, [lu]'s and [solve_triangular]'s results takes the operations
    of the unblocked right-looking algorithm: its updates in step order (from
    the last row for an upper [op a]), each one fused multiply-add into its own
    value, a complex one two for each of its real and imaginary parts, its terms
    in the order {!Nx_kernel.section-contraction} numbers them, and its division
    last. A division by a complex number whose imaginary part is [0], as
    Cholesky's pivots are, divides each part of the dividend by the divisor's
    real part, and any other is [Fdiv]'s ({!Nx_kernel.Prog}). A matrix's results
    are a function of its values alone, whatever its batch.

    With [γ] as {!Nx_kernel.section-bounds} defines it, [ν(k) = k 2{^-126}], [k]
    doubled at [complex64], and [|·|] taken elementwise, the bounds below hold
    for [a′]. Their [ν] terms bound what flushing below the least normal number
    adds.
    - [cholesky]: [L Lᴴ = a′ + Δ] with
      [|Δ_ij| <= γ(n + 1) (|L| |Lᴴ|)_ij + ν(n + 1) (1 + L_jj)], and NaN for a
      matrix that is not definite. The [Upper] factor is the [Lower] one's
      conjugate transpose, bit for bit.
    - [lu]: [L U = P a′ + Δ] with
      [|Δ_ij| <= γ(k + 2) (|L| |U|)_ij + ν(k + 2) (1 + |U_pp|)], [p = min(i, j)]
      and [P] the rows' permutation.
    - [qr]: with [g = γ(k (2m + 10))] and [h = ν(k (2m + 10))], each column
      [a′_j] has [‖a′_j - q r_j‖₂ <= g ‖a′_j‖₂ + √m h], and each element of
      [qᴴq - I] is at most [g] in magnitude.
    - [solve_triangular]: [(op a′ + Δ) x = b′ + δ] with
      [|Δ| <= γ(n + 2) |op a′|] and [|δ_i| <= ν(n + 2) (1 + |(op a′)_ii|)] for
      each column of [b′], and NaN for a matrix whose [op a′] has a zero on its
      diagonal. *)

(** {2:spectral Spectral routines}

    [eigh], [svd] and [eig] compute, at [float32] and [complex64], these calls,
    each of an [a] whose batch axes merge into one strided axis:
    - [eigh] of order at most 32, by cyclic Jacobi in a group's threadgroup
      memory; past 32, by reduction to a real tridiagonal [T] in panels of 32
      columns, then for [w] alone bisection on [T], and with [v] divide and
      conquer on [T], its leaves of 32 by Jacobi, and the panels' reflectors
      applied to its vectors;
    - [svd] of both extents at most 32, by one-sided cyclic Jacobi in a group's
      threadgroup memory; of [k = 0], its factors the identity's columns; and
      [s] alone past 32, by reduction to a bidiagonal matrix and bisection on
      its Golub-Kahan matrix. The reduction sums a column of more than 65,536
      rows in chunks of 65,536, combined in chunk order, and past 65,536 rows a
      reflector whose [β] it rescales takes the norm of its rescaled column as
      its first norm times [2{^103}];
    - [eig]'s [w] alone: each matrix balanced, reduced to Hessenberg form and to
      Schur form by shifted QR, a group a matrix.

    A call declines where one step of one matrix is estimated past one launch's
    50 ms ({b Time}, above), as [eig]'s QR sweep is, a group a matrix, past an
    order of about 3,400 at [float32] and 1,700 at [complex64]. Every other call
    of them declines. Each computes on [a′], as {!section-matrices} states, and
    a matrix's results are a function of its values alone, whatever its batch.

    Jacobi iterates to 60 sweeps, a root of divide and conquer's secular
    equation to 30 steps and [eig]'s QR to [30 max(10, h)] sweeps of the matrix
    in all, [h] the order balancing leaves to iterate on. A matrix on which an
    iteration does not converge within its limit, or that holds a NaN or an
    infinity where it is read, is NaN in every result. Otherwise, with [γ] and
    [ν] as {!section-matrices} defines them, [‖·‖_F] the Frobenius norm and
    [‖·‖₂] the spectral one:
    - [eigh]: with [g = γ(16n)] and [h = ν(16n)],
      [‖a′ - v diag w vᴴ‖_F <= g ‖a′‖_F + h] and [‖vᴴv - I‖_F <= g]; each
      eigenvalue within [g ‖a′‖₂ + h] of [a′]'s.
    - [svd]: with [g = γ(16 max(m, n) + 100)] and [h = ν(16 max(m, n) + 100)],
      [‖a′ - u diag s vh‖_F <= g ‖a′‖_F + h] over [u]'s first [k] columns and
      [vh]'s first [k] rows, [‖uᴴu - I‖_F <= g] and [‖vh vhᴴ - I‖_F <= g],
      reduced or complete; each singular value within [g ‖a′‖₂ + h] of [a′]'s.
    - [eig]: it balances [a′] into [b = D⁻¹ Pᵀ a′ P D], [P] a permutation and
      [D] diagonal, of powers of 2 that bring each row's norm near its column's,
      in at most 256 passes over the rows; a matrix that needs more keeps the
      [D] its 256 passes reach. With [g = γ(16n)] and [h = ν(16n)], [w] is the
      spectrum of [a′ + E] with [‖D⁻¹ Pᵀ E P D‖_F <= g ‖b‖_F + h].

    [eigh]'s [w] ascends. [svd]'s [s] descends. [eig]'s [w] is in the order the
    Schur form's diagonal leaves it. On real [a], [w]'s values that are not real
    come in adjacent conjugate pairs, the one of positive imaginary part first,
    a pair's imaginary part below [2{^-126}] in magnitude [±2{^-126}], its sign
    kept; a real value has a [+0] imaginary part.

    [eigh] past 32 takes workspace of at most [n² + 161 n + 2048] elements of
    [a]'s dtype, [3 n] floats, [6 n] ints and 32 bytes a matrix for [w] alone,
    and [5 n² + 129 n + 2048] elements, [3 n² + 82 n] floats, [8 n] ints and 24
    bytes with [v], its products' copies included; [svd]'s [s] alone past 32
    rows or columns, [M (k + 1) + 2 k] elements of [a]'s dtype, [7 k] floats,
    [6 k] ints and 88 bytes a matrix, [M = max(m, n)], and past [M] of 65,536,
    [max(2, k - 1) ⌈M / 65,536⌉] elements more; [eig], [3 n² + n] elements,
    [2 n² + 5 n] floats and 60 bytes a matrix. An [a] that is not C-contiguous
    is copied first, flushed, into [m n] elements more a matrix. Alignment adds
    at most 512 bytes a matrix and 2 KiB a call. *)

(** {1:attention Attention}

    [attention] computes every case {!Nx_kernel.S} offers at a [float32]
    accumulator, of [float16], [bfloat16] or [float32] queries, keys and values,
    over dense or paged keys, with or without lengths, and any pair program
    whose output 2 is the constant [true]. It declines a pair program of any
    other output 2, which keeps some visible pairs alone, and a call of
    [float64] queries or accumulator ({b Float64 and complex128}).

    A group runs up to 32 query rows of one key head: of [c] of its [g] query
    heads, [c = min g (max 1 (32 / min tq 32))], [min tq (32 / c)] rows each.
    Each row reads its keys in blocks of 64 from key 0, up to its lead index's
    length, and keeps [m], [l] and [a] of [dv] elements from [-∞], [+0] and
    [+0], as nx.cpu's: [q·k] fused in increasing dimension from [+0] times the
    scale; [m'] the greatest of [m] and the block's scores, a NaN where one is;
    where [m'] is [-∞] the weights are [0], [α = 1], and [m] and [l] stay as
    they were, else [α = exp (m - m')], the weights [exp (x - m')] and
    [l = α l + s] in one fused multiply-add, with [s] the weights' sum through
    nx.cpu's eight lanes. Then, whatever [m'], the block's part [p] sums the
    weights times the values from [+0], fused in increasing key, over the
    visible pairs alone where a value is a NaN or an infinity, and [a = α a + p]
    in one fused multiply-add. A block in which the row has no visible pair
    leaves its state as it was. Then [o = a / l + 0] rounded once, and
    [lse = m + log l]; a row of no weight takes [o = +0] and [lse = -∞]. [exp]
    and [log] are {!Nx_kernel.Prog.Exp}'s and {!Nx_kernel.Prog.Log}'s. So the
    result has nx.cpu's bits, but where a value is subnormal
    ({b Subnormal numbers}), and a value at a pair hidden from a row never
    reaches its [o]. A key past its length, or of a table entry outside \[[0],
    [pages]), is never read.

    {b The pair program.} A launch computes output 0 where it is the constant
    [true] or a [bool] load, and output 1 where it is the score, or the score
    added to, subtracted from, subtracted from by or multiplied by a [float32]
    load or a node that does not read the score. Other nodes of either output
    that do not read the score are {!map}'s launches into temporaries before it,
    at [lead… × hq × tq × tk] elements each. An output 1 of any other form takes
    a launch that stores each score, [4] bytes a pair, then {!map}'s launch of
    output 1 over them, before the launch that reads it. Each temporary holds
    its node's values, so the bits are the program's.

    {b Time.} A launch takes the most groups whose estimate fits a launch's time
    ({b Time budget}); a call whose group alone passes a launch's is declined.

    {b Memory.} A group takes threadgroup memory by its dtype and its [dk]: at
    [float16] and [bfloat16], 11376 bytes where [dk] is 64 or less and 20592
    where it passes 64; at [float32], 15472 and 28784. *)

include Nx_kernel.S

(**/**)

val bound_sorts : parts:int -> wide:int -> unit
(** [bound_sorts ~parts ~wide] makes a radix pass cut a slice into at most
    [parts] parts in place of 256, and a slice of at least [2{^wide}] elements
    take 64-bit positions and counts whose low words carry at bit [wide], in
    place of [2{^32}] and 32, so that tests reach them with small arrays. It
    applies to calls planned after it returns.

    Raises [Invalid_argument] if [parts] is not in \[1; 256\] or [wide] not in
    \[1; 32\]. *)

val bound_fresh : workspace:int -> tickets:int -> unit
(** [bound_fresh ~workspace ~tickets] makes a call take buffers of its own past
    [workspace] bytes of workspace and [tickets] bytes of tickets, in place of
    64 MiB and 4 MiB, so that tests reach them with small arrays. It applies to
    calls planned after it returns.

    Raises [Invalid_argument] if a bound is negative or above its kept size. *)

val bound_kept : launches:int -> unit
(** [bound_kept ~launches] makes a call of more than [launches] launches take a
    submission of its own, so that tests reach one with small arrays; [max_int]
    restores the library's rule. It applies to calls planned after it returns.

    Raises [Invalid_argument] if [launches] is negative. *)

val bound_iterations :
  sweeps:int -> steps:int -> qr:int -> passes:int -> secular:int -> unit
(** [bound_iterations ~sweeps ~steps ~qr ~passes ~secular] sets the spectral
    routines' limits ({!section-spectral}) to [sweeps] Jacobi sweeps,
    [qr max(10, h)] QR sweeps of [eig] in all, [passes] of [eig]'s balancing and
    [secular] steps of a root of divide and conquer's secular equation, so that
    tests reach each limit with small matrices; makes a launch of a step that
    resumes where the last stopped, [eig]'s QR sweeps among them, run at most
    [steps] of its steps, so that they reach every place a launch can end.
    [~sweeps:60 ~steps:max_int ~qr:30 ~passes:256 ~secular:30], the initial
    values, restores them. It applies to calls planned after it returns.

    Raises [Invalid_argument] if [sweeps], [steps], [passes] or [secular] is
    below [1] or [qr] is negative. *)

val bound_direct : bool -> unit
(** [bound_direct b] makes a direct map or reduction, or a cast ({b Time}), run
    its tape where [b] is [false], so that tests check the two against each
    other; [true], the initial value, restores the library's rule. It applies to
    calls planned after it returns. *)

val bound_copy : wide:int -> unit
(** [bound_copy ~wide] makes a copy of elements one at a time take 64-bit
    indices in a launch that reaches its element [2{^wide} - 256], in place of
    [2{^32} - 256], and a copy of rows take 64-bit byte indices in a launch that
    reaches its byte [2{^wide} - 256] and in every launch where an extent of its
    axes, its row's in bytes, passes [2{^wide} - 256], so that tests reach them
    with small arrays. It applies to calls planned after it returns.

    Raises [Invalid_argument] if [wide] is not in \[9; 32\]. *)

val bound_fft_wide : int -> unit
(** [bound_fft_wide n] makes a Fourier transform's side take 64-bit offsets
    where a line's points span more than [n] elements, in place of [2{^31} - 1],
    so that tests reach the transposes and pairs of such sides, and the plans
    that route their lines past the steps, with small arrays. It applies to
    calls planned after it returns.

    Raises [Invalid_argument] if [n] is past [2{^31} - 1] or below [8191], the
    span of a step's rows, a halved line's [8192] floats at most. *)

val bound_index :
  ?runs:int -> ?region:int -> ?narrow:int -> ?taps:int -> unit -> unit
(** [bound_index ~runs ~region ~narrow ~taps ()] makes a range gather's runs
    take levels of [runs] runs (defaults to 4096), a power of two at most 4096,
    so that tests reach two and three levels with few runs; its positions take
    at most [region] bytes at a time (defaults to 64 MiB), at least 8, so that
    tests reach several chunks with few positions; and they take 64 bits where
    the source axis has [2{^narrow}] elements or more (defaults to 31), from 1
    to 31, so that tests reach 64-bit positions with small arrays; and a fold
    divides in 64 bits where a padded position or a step reaches [2{^taps}]
    (defaults to 31), from 1 to 31, and holds its indices, positions and taps in
    64 bits where one reaches [2{^(taps - 1)}], so that tests reach both with
    small arrays. A bound left out takes its default. It applies to calls
    planned after it returns.

    Raises [Invalid_argument] if a bound is outside its range. *)

val bound_offsets : int -> unit
(** [bound_offsets n] makes an attention index its keys and values in 32 bits
    only where each reach is below [n], in place of [2{^31}], so that tests
    reach the 64-bit form with small arrays; [0] takes it always. The reaches
    are those of a row of [k] and of [v] over its elements and, where the keys
    are dense, of a block of at most 64 keys over its keys: the count less one
    times the step's magnitude. Both forms give the same bits. It applies to
    calls planned after it returns.

    Raises [Invalid_argument] if [n] is not in \[[0]; [2{^31}]\]. *)

val kept_calls : Rig.t -> int
(** [kept_calls d] is the number of submissions kept for later calls on [d], [0]
    before [d]'s first call. *)

val kept_keys_on : Rig.t -> int
(** [kept_keys_on d] is the number of specialised steps' keys whose functions
    [d] keeps for later calls, at most 256 or {!bound_keys}'s bound. *)

val kept_call_keys_on : Rig.t -> int
(** [kept_call_keys_on d] is the number of keys that the submissions kept for
    later calls on [d] launch, each counted once for each submission, at most
    256 or {!bound_keys}'s bound. *)

val bound_keys : int -> unit
(** [bound_keys n] makes a device keep the functions of the [n] keys made last,
    and submissions that launch at most [n] keys together, in place of 256, so
    that tests drop keys with few calls; [bound_keys 256] restores it. It
    applies to keys and submissions kept after it returns.

    Raises [Invalid_argument] if [n] is not in \[1; 256\]. *)

val forget_keys : Rig.t -> unit
(** [forget_keys d] drops the functions of the keys that [d] keeps and the
    submissions it keeps that launch one, so that a test starts from none.

    Raises [Invalid_argument] if nx.metal does not compute on [d]. *)

val prepare : Rig.t -> Fft.call list -> unit
(** [prepare d calls] makes on [d] the pipelines of every key [calls] launch
    that [d] does not keep, together, so that a test's first calls wait for
    none; [d] keeps them as a call's. Of more than 256 keys, it keeps the last
    made.

    Raises [Invalid_argument] if nx.metal does not compute on [d]. *)

val bound_constants : int -> unit
(** [bound_constants n] makes a device keep at most [n] bytes of constants, in
    place of 64 MiB, so that tests reach a call's own constants with small ones.
    It applies to calls submitted after it returns.

    Raises [Invalid_argument] if [n] is negative or above 64 MiB. *)

val resolved : Rig.t -> Kernels.t -> bool
(** [resolved d k] is [true] iff a call on [d] waited for [k]'s pipeline.

    Raises [Invalid_argument] if nx.metal does not compute on [d]. *)

val holds : Rig.t -> Kernels.t -> int
(** [holds d k] is the threads Metal lets a group of [k]'s pipeline on [d] have,
    which its registers bound, and by which nx.metal counts the groups of [k] a
    core holds at once.

    Raises [Invalid_argument] if nx.metal does not compute on [d]. *)

val kept_constants : Rig.t -> int
(** [kept_constants d] is the number of constants kept for later calls on [d],
    [0] before [d]'s first call. *)

val bound_plans : int -> unit
(** [bound_plans n] makes a device keep at most [n] bytes of maps' plans, in
    place of 2 MiB, so that tests compare calls that plan with calls that find
    their plan; [0] keeps none. It applies to calls planned after it returns.

    Raises [Invalid_argument] if [n] is negative or above 2 MiB. *)

val bound_pieces : int -> unit
(** [bound_pieces n] makes a call's casts take a buffer of at most [n] bytes, in
    place of 64 MiB, so that tests cut small calls into pieces. It applies to
    calls planned after it returns.

    Raises [Invalid_argument] if [n] is not positive or above 64 MiB. *)

val bound_outputs : int -> unit
(** [bound_outputs n] makes a contraction compute at most [n] outputs a piece,
    in place of [2{^31} - 1], so that tests cut small calls into pieces. It
    applies to calls planned after it returns.

    Raises [Invalid_argument] if [n] is not positive or above [2{^31} - 1]. *)

val kept_plans : Rig.t -> int
(** [kept_plans d] is the number of maps' plans kept for later calls on [d], of
    the current bounds or earlier ones, [0] before [d]'s first map. *)

val call :
  Rig.t ->
  written:Nx_array.any iarray ->
  read:Nx_array.any iarray ->
  Plan.t ->
  Nx_array.answer
(** [call d ~written ~read c] is the answer of a kernel on [d] that planned [c],
    claiming [written] and [read] through {!Nx_array.door}. *)

val product :
  ?init_first:bool ->
  ?neg:bool ->
  ?conj_a:bool ->
  ?conj_b:bool ->
  ?tri:[ `Lower | `Upper ] ->
  ?ranges:int ->
  ?workspace:bool ->
  acc:Nx_array.Dtype.any ->
  init:bool ->
  y:Nx_array.any ->
  Nx_array.any ->
  Nx_array.any ->
  Nx_array.answer
(** [product ~acc ~init ~y a m] is the matrix routines' product
    {!Contract.product} [y <- [y +] op(a) op(m)] of [a] [batch; rows; k] and [m]
    [batch; k; cols] into [y] [batch; rows; cols], as one call, with its
    options. With [workspace], each matrix is copied by {!Contract.copy} into a
    piece of the workspace laid out as it is in its buffer, the product computed
    there and [y] copied back. [y] shares no byte with [a] or [m]. It is
    [Declined] where a step finds no launches ({!Contract.Decline}). *)

module Kernels = Kernels
(** The metallib's kernels and their words. For tests. *)

module Plan = Plan
(** Calls of the kernels, as a family plans them. For tests. *)

module Operands = Operands
(** What a plan reads of a call's arrays. For tests. *)

module Moves = Moves
(** The plans of maps that move bits. For tests. *)

module Loop = Loop
(** The plans of maps, reductions and scans of programs. For tests. *)

module Contract = Contract
(** The plans of contractions. For tests. *)

module Pieces = Pieces
(** The pieces of a contraction whose casts pass a buffer. For tests. *)

module Factor = Factor
(** The plans of the matrix routines. For tests. *)

module Fft_tables = Fft_tables
(** The Fourier transforms' tables. For tests. *)

module Fft = Fft
(** The Fourier transforms' plans. For tests. *)

module Sort = Sort
(** The plans of sorts and numberings. For tests. *)

module Index = Index
(** The plans of gathers, scatters, assemblies and folds. For tests. *)

module Spectral = Spectral
(** The spectral routines' plans. For tests. *)

(**/**)
