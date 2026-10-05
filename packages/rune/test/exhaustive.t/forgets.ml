(* An interpreter that forgets [Detach]. *)

open Rune_internals

let call : type r. r Construct.t -> (unit -> r) option =
 fun c ->
  match[@warning "@4@8"] c with
  | Loop _ | Remat _ | Barrier _ | Custom _ | Lanes _ | Lane_index _
  | Lane_count _ | Add _ ->
      None
