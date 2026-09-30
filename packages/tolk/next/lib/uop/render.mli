(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Nodes as text: expressions and program listings.

    {!render} writes the value a node computes as a one-line expression, which
    diagnostics and kernel names print for indices and sizes. {!pp_uops} lists
    the nodes of a linear program, one per line. {!Ops.pp} prints a node as the
    calls that build it. *)

val render : ?simplify:bool -> Ops.t -> string
(** [render ~simplify u] is [u] written as an expression, after {!Ops.simplify}
    if [simplify] (default [true]). Each node is written from its sources:
    - a {!Op.Param}, {!Op.Buffer} or {!Op.Alloc} is its name, or [p], [b] or [a]
      followed by its slot if it has none;
    - an {!Op.After} is its first source, and an {!Op.Special} its name;
    - a {!Op.Range} is [r] followed by {!Ops.range_str}, and an unbounded loop,
      of type {!Dtype.Void}, is [loop] followed by the first part of its
      identity;
    - a constant, or a cast of one, is its value as {!Dtype.pp_const} writes it:
      [3], [1.5], [True], [Invalid];
    - any other cast is the type's name in parentheses before its operand:
      [(float)(x)];
    - {!Op.Neg}, {!Op.Reciprocal}, {!Op.Max}, {!Op.Mulacc}, {!Op.Where},
      {!Op.Cdiv} and {!Op.Cmod} are [(-x)], [(1/x)], [max(x, y)], [(x*y+z)],
      [(x if c else y)], [cdiv(x, y)] and [cmod(x, y)];
    - a movement is its source followed by [.reshape], [.expand], [.pad],
      [.shrink], [.permute] or [.flip] and its argument as a tuple: [(4,8)],
      [((0, 2),(1, 3))], [(1, 0)], and for a flip the axes reversed;
    - {!Op.Add}, {!Op.Sub}, {!Op.Mul}, {!Op.Floordiv}, {!Op.Floormod},
      {!Op.Shl}, {!Op.Shr}, {!Op.And}, {!Op.Or}, {!Op.Xor}, {!Op.Cmplt} and
      {!Op.Cmpne} are [+], [-], [*], [//], [%], [<<], [>>], [&], [|], [^], [<]
      and [!=] between their operands, in parentheses. An operand loses its own
      parentheses where precedence makes them redundant: from the tightest,
      [* // %], then [+ -], [<< >>], [&], [^] and [|]. Operations of equal
      precedence associate to the left, so [(a+b)+c] is [(a+b+c)] and [a-(b+c)]
      keeps its parentheses. A comparison keeps its operands' parentheses, and
      its own within any operation;
    - an {!Op.Index} or an {!Op.Stage} is each source after the first between
      brackets, without their outer parentheses: [[i][j]];
    - a load through an index is the indexed storage followed by the index:
      [buf[i]], and [(buf[i] if g else alt)] when guarded by [g];
    - a stack is its sources between braces, separated by commas: [{a,b}];
    - any other node is written as {!Ops.pp} prints it.

    Raises [Invalid_argument] if [simplify] and the symbolic rules are not
    installed, as {!Ops.simplify}; so does writing a movement whose argument
    sizes are nodes. *)

val srender : Ops.sint -> string
(** [srender s] is an integer [s] in decimal, and a node [s] as {!render} writes
    it. *)

val pp_uops : Format.formatter -> Ops.t list -> unit
(** [pp_uops] formats a list of nodes, a linear program, one line per node, in
    columns: its position, from [0], right-aligned on 4 columns; its operation
    as {!Op.pp} prints it, on 20; the ranges it runs inside ({!Ops.ranges}) as
    {!Ops.multirange_str} writes them in colour ({!Helpers.colored}), on 10; its
    type as {!Dtype.pp} prints it, on 40; its sources as a list, on 32; and its
    argument. A source is its position in the list, the value of a constant as a
    quoted literal ([['1.5']]), or ['--'] if it is not in the list. The argument
    prints as {!Ops.pp_arg} does, except a string, such as a name, source text
    or a single device, which prints without quotes, and a float constant, which
    prints as {!Dtype.pp_const} does. A column is padded with spaces and never
    cut. Lines are separated by newlines; the last is not ended. *)
