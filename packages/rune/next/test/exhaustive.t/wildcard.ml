(* An interpreter that covers what it forgets with a wildcard. *)

open Rune_next

let call : type r. r Construct.t -> (unit -> r) option =
 fun c -> match[@warning "@4@8"] c with Detach _ -> None | _ -> None
