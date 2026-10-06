(* An interpreter that covers what it forgets with a wildcard. *)

open Rune_internals

let call : type r. r Construct.t -> r Construct.answer option =
 fun c -> match[@warning "@4@8"] c with Detach _ -> None | _ -> None
