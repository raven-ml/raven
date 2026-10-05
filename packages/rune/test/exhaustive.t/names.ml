(* An interpreter that names every construct. *)

open Rune_internals

let call : type r. r Construct.t -> (unit -> r) option =
 fun c ->
  match[@warning "@4@8"] c with
  | Loop _ | Compiled _ | Remat _ | Barrier _ | Custom _ | Root _ | At_map _
  | Lanes _ | Lane_index _ | Lane_count _ | Add _ | Detach _ ->
      None
