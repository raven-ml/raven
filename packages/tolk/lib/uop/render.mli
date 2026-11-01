(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Nodes as text: expressions and program listings.

    {!render} writes the value a node computes as a one-line expression, which
    diagnostics and kernel names print for indices and sizes. {!render_uir}
    writes a graph one node a line, as diagnostics print it. {!Ops.pp} prints a
    node as the calls that build it. *)

val render : ?simplify:bool -> Ops.t -> string
(** [render ~simplify u] is [u] written as an expression, after {!Shape.simplify}
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
    - {!Op.Neg}, {!Op.Recip}, {!Op.Max}, {!Op.Mulacc}, {!Op.Where},
      {!Op.Truncdiv} and {!Op.Truncmod} are [(-x)], [(1/x)], [max(x, y)],
      [(x*y+z)], [(x if c else y)], [truncdiv(x, y)] and [truncmod(x, y)];
    - a movement is its source followed by [.reshape], [.expand], [.pad],
      [.shrink] or [.permute] and its argument as a tuple: [(4,8)],
      [((0, 2),(1, 3))], [(1, 0)];
    - {!Op.Add}, {!Op.Sub}, {!Op.Mul}, {!Op.Div}, {!Op.Mod},
      {!Op.Shl}, {!Op.Shr}, {!Op.And}, {!Op.Or}, {!Op.Xor}, {!Op.Cmplt},
      {!Op.Cmpne} and {!Op.Cmpeq} are [+], [-], [*], [//], [%], [<<], [>>],
      [&], [|], [^], [<], [!=] and [==] between their operands, in
      parentheses. An operand loses its own parentheses where precedence makes
      them redundant: from the tightest, [* // %], then [+ -], [<< >>], [&],
      [^] and [|]. Operations of equal precedence associate to the left, so
      [(a+b)+c] is [(a+b+c)] and [a-(b+c)] keeps its parentheses. A
      comparison keeps its operands' parentheses, and its own within any
      operation;
    - an {!Op.Index} or an {!Op.Stage} is each source after the first between
      brackets, without their outer parentheses: [[i][j]];
    - a load through an index is the indexed storage followed by the index:
      [buf[i]], and [(buf[i] if g else alt)] when guarded by [g];
    - a stack is its sources between braces, separated by commas: [{a,b}];
    - any other node is written as {!Ops.pp} prints it.

    Raises [Invalid_argument] if [simplify] and the symbolic rules are not
    installed, as {!Shape.simplify}; so does writing a movement whose argument
    sizes are nodes. *)

val srender : Ops.sint -> string
(** [srender s] is an integer [s] in decimal, and a node [s] as {!render} writes
    it. *)

val render_uir : Ops.t list -> string
(** [render_uir uops] writes the nodes [uops], in order, one line each, as
    [%i = op srcs : arg]: its position among the lines, from [0]; its operation
    in lower case, in its colour ({!Helpers.colored}); its sources, separated
    by commas; and its argument after [" : "], if it writes one. A constant has
    no line and is written in its users, as is a stack of constants, as a tuple
    of them; any other source is written [%i]. Lines are separated by newlines;
    the last is not ended.

    The argument is written as:
    - for a constant, [invalid], [true] or [false], or its value as
      {!Dtype.pp_const} writes it, a constant being of a weak type or boolean
      ({!Ops.const});
    - for storage and variables, its name in double quotes if it has one,
      then [dtype=] and [slot=], and then [bounds=[lo,hi]], [multiple_of=],
      [addrspace=] (but global memory), [device=] (bare if a word of letters,
      digits, [_] and [:], else as {!Ops.pp_device} writes it) and
      [volatile=true], where they apply;
    - for a range, its axis type and the parts of its identity, separated by
      spaces: [LOOP 1 2];
    - for a kernel's sink, its name, and nothing for any other sink;
    - for a call, [name='f'] and [precompile=true], where they apply;
    - for a reduction, [op=add] and, unless [0], [pop=n];
    - for a cast, the type's {!Dtype.name}; for a copy to one device, its name;
      for a hardware index, its name;
    - for any other argument, the literal {!Ops.pp_arg} writes.

    Raises [Not_found] if a source that has a line is not among [uops]. *)
