(** UOp graphs as text: the format of the programs the disk cache keeps
    ({!Codegen.to_program}) and of the goldens, which tolk's [test/README.md]
    specifies.

    A graph is the nodes under a sink, one line each, in topological order:
    {v <index> <op> <dtype> [<sources>] <arg> tag=<tag> v}
    [test/gen/graph.py] writes the same text from tinygrad's graphs, so a graph
    golden reads into the UOps tinygrad built, and a graph built here writes as
    the golden tinygrad recorded. *)

val to_string : Ops.t -> string
(** [to_string sink] is the graph under [sink]. A node comes after its sources,
    visited in order, and after the nodes its argument holds. *)

val of_string : string -> Ops.t
(** [of_string text] is the sink of the graph [text], the node of its last line.
    Reading takes the graph's storage slots: every slot {!Ops.unique_num}
    returns afterwards is greater than each slot of the graph.

    Raises [Failure] naming the node, by its line's position, and what is wrong
    with it if [text] is empty, a line does not read, an index is not its line's
    position, a source or a node in an argument is not an earlier line, an
    argument does not fit its operation, or a node's written data type is not
    the one its operation, sources and argument derive. *)

(** {1:disk Disk} *)

val cached :
  table:string ->
  key:string ->
  valid:(Ops.t -> bool) ->
  (unit -> Ops.t) ->
  Ops.t * bool
(** [cached ~table ~key ~valid make] is the graph that the {!Helpers.Diskcache}
    holds for [key] in [table], and [true], if its entry reads as a graph that
    [valid] accepts; otherwise it is [make ()], which replaces the entry, and
    [false]. An entry that does not read is made anew, as one that is missing.

    Raises [Sys_error] if the entry cannot be written. *)
