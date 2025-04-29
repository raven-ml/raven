type execution_result = {
  output : string;
  error : string option;
  status : [ `Error | `Success ];
}
val refill_lexbuf :
  string -> int ref -> Format.formatter option -> bytes -> int -> int
val initialize_toplevel : unit -> unit
val ensure_terminator : string -> string
val execute : bool -> Format.formatter -> Format.formatter -> string -> bool
