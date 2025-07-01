type execution_result = {
  output : string;
  error : string option;
  status : [ `Error | `Success ];
}

val initialize_toplevel : unit -> unit
val ensure_terminator : string -> string
val execute : bool -> Format.formatter -> Format.formatter -> string -> bool

val format_output : string -> string
(** Format execution output appropriately based on content type. Returns the
    output wrapped in markdown code blocks for regular code output, or as-is for
    markdown content like images. *)
