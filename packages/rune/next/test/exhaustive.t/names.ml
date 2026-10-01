(* An interpreter that names every construct. *)

open Rune_next

let call : type r. r Construct.t -> (unit -> r) option =
 fun c ->
  match[@warning "@4@8"] c with
  | Scan _ | Remat _ | Barrier _ | Custom _ | Lanes _ | Lane_index _
  | Lane_count _ | Add _ | Detach _ ->
      None
