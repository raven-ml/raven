(** UOp graphs as text, in the graph format that [test/README.md] specifies.

    A graph is the nodes under a sink, one line each, in topological order:
    {v <index> <op> <dtype> [<sources>] <arg> tag=<tag> v}
    [test/gen/graph.py] writes the same text from tinygrad's graphs, so a graph
    golden reads into the UOps tinygrad built, and a graph built here writes as
    the golden tinygrad recorded. *)

open Tolk_next

val to_string : Ops.t -> string
(** [to_string sink] is the graph under [sink]. A node comes after its sources,
    visited in order, and after the nodes its argument holds. *)

val of_string : string -> Ops.t
(** [of_string text] is the sink of the graph [text], the node of its last line.

    Raises [Failure] naming the node, by its line's position, and what is wrong
    with it if [text] is empty, a line does not read, an index is not its line's
    position, a source or a node in an argument is not an earlier line, an
    argument does not fit its operation, or a node's written data type is not
    the one its operation, sources and argument derive. *)
