(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type status = Converged | Budget_spent | Not_bracketed | Not_finite | Stalled
type fact = Fact : string * (float, 'b) Nx.t -> fact

type 'a t = {
  fn : string;
  settings : string;
  value : 'a;
  error : 'a;
  status : (int32, Nx.int32_elt) Nx.t;
  evaluations : (int32, Nx.int32_elt) Nx.t;
  facts : fact list;
}

let code = function
  | Converged -> 0l
  | Budget_spent -> 1l
  | Not_bracketed -> 2l
  | Not_finite -> 3l
  | Stalled -> 4l

let statuses = [ Converged; Budget_spent; Not_bracketed; Not_finite; Stalled ]

let v ~fn ~settings ~value ~error ~status ~evaluations ~facts =
  { fn; settings; value; error; status; evaluations; facts }

let best t = t.value
let error t = t.error
let evaluations t = t.evaluations
let is st t = Nx.equal_s t.status (code st)
let ok t = is Converged t

(* The facts as a structure of tensors of any dtype. *)
let facts =
  let module M = struct
    type _ t = fact list

    let walk c l =
      let open Nx.Ptree.Walk in
      list
        (fun c (Fact (name, x)) ->
          case c name;
          Fact (name, tensor c x))
        c l
  end in
  Nx.Ptree.instantiate (module M)

let reason = function
  | Converged -> "converged"
  | Budget_spent -> "the budget is spent"
  | Not_bracketed -> "the ends do not bracket a zero: f has one sign at both"
  | Not_finite -> "a value of the function is not finite"
  | Stalled -> "the search stalled before meeting the tolerance"

let of_code c = List.find (fun s -> Int32.equal (code s) c) statuses

let message t i (status, (evaluations, (converged, facts))) =
  let lane =
    if Array.length i = 0 then ""
    else
      " lane ["
      ^ String.concat ", " (Array.to_list (Array.map string_of_int i))
      ^ "]"
  in
  let facts =
    List.map
      (fun (Fact (name, x)) -> Printf.sprintf "%s %g" name (Nx.item [] x))
      facts
  in
  let others = Int32.to_int (Nx.item [] converged) in
  Printf.sprintf "%s:%s: %s.\n  %s%s\n  %ld evaluations.%s" t.fn lane
    (reason (of_code (Nx.item [] status)))
    t.settings
    (if facts = [] then "" else "\n  " ^ String.concat ", " facts)
    (Nx.item [] evaluations)
    (match others with
    | 0 -> ""
    | 1 -> "\n  1 other lane converged."
    | n -> Printf.sprintf "\n  %d other lanes converged." n)

let get t =
  let ok = ok t in
  let converged = Nx.sum (Nx.cast Nx.int32 ok) in
  Nx.check
    Nx.Ptree.(pair tensor (pair tensor (pair tensor facts)))
    ok
    (t.status, (t.evaluations, (converged, t.facts)))
    (fun i d -> Failure (message t i d));
  t.value

let ptree (type a) (s : a Nx.Ptree.t) : a t Nx.Ptree.t =
  let module M = struct
    type nonrec _ t = a t

    let walk c t =
      let open Nx.Ptree.Walk in
      let value = field c "value" (structure s) t.value in
      let error = field c "error" (structure s) t.error in
      let status = field c "status" tensor t.status in
      let evaluations = field c "evaluations" tensor t.evaluations in
      let facts = field c "report" (structure facts) t.facts in
      { t with value; error; status; evaluations; facts }
  end in
  Nx.Ptree.instantiate (module M)

let pp ppf t =
  let codes = Nx.to_array (Nx.reshape [| -1 |] t.status) in
  let count st =
    Array.fold_left
      (fun n c -> if Int32.equal c (code st) then n + 1 else n)
      0 codes
  in
  let parts =
    List.filter_map
      (fun st ->
        let n = count st in
        if n = 0 then None
        else
          Some
            (Printf.sprintf "%d %s" n
               (match st with
               | Converged -> "converged"
               | Budget_spent -> "budget spent"
               | Not_bracketed -> "not bracketed"
               | Not_finite -> "not finite"
               | Stalled -> "stalled")))
      statuses
  in
  Format.fprintf ppf "%s: %s" t.fn (String.concat ", " parts)
