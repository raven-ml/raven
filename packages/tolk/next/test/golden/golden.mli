(** Goldens: outputs of tinygrad, recorded by [test/gen/generate.py].

    A golden is a file [<name>.golden] beside the suite that reads it. Its first
    line names the tinygrad commit it was recorded from, as
    [# tinygrad <commit>], and the rest is its body. A suite lists its goldens
    in its stanza's [deps] and reads them by file name. *)

val text : string -> string
(** [text file] is the body of the golden [file].

    Raises [Sys_error] if [file] cannot be read, and [Failure] if it is not a
    golden. *)

(** {1:tables Tables}

    The body of a table golden is lines of cells separated by tabs: the column
    names, then one line per row. A cell is the text tinygrad prints for its
    value, such as [dtypes.float], [True] or [inf]. *)

type row
(** The type for a row of a table golden. *)

val table : string -> row list
(** [table file] is the rows of the table golden [file], in order.

    Raises [Sys_error] if [file] cannot be read, and [Failure] if it is not a
    golden, has no row, or has a row whose cells do not match its columns. *)

val cell : row -> string -> string
(** [cell row column] is the cell of [row] in [column].

    Raises [Invalid_argument] if the table has no column [column]. *)

val key : string list -> row -> string
(** [key columns row] names [row] by its cells in [columns], as
    [a=dtypes.half b=dtypes.char], for the [~name] of [Windtrap.cases].

    Raises [Invalid_argument] if the table lacks one of [columns]. *)
