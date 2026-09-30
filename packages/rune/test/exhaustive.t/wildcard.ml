(* An interpreter that covers what it forgets with a wildcard. *)

open Nx.Op

let run : type r. r t -> r =
 fun op -> match[@warning "@4@8"] op with Read _ -> eval op | _ -> eval op
