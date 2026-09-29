(* An interpreter that covers what it forgets with a wildcard. *)

open Nx_effect

let run : type r. r Op.t -> r =
 fun op -> match[@warning "@4@8"] op with Read _ -> eval op | _ -> eval op
