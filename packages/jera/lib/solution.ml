(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type status = Converged | Budget_spent | Not_bracketed | Not_finite | Stalled
type fact = Fact : string * (float, 'b) Nx.t -> fact
type spent = { used : (int32, Nx.int32_elt) Nx.t; unit : string; budget : int }

type 'a t = {
  fn : string;
  settings : string;
  fix : status -> (string * float) list -> string;
  spent : (string * int) option;
  value : 'a;
  error : 'a;
  status : (int32, Nx.int32_elt) Nx.t;
  evaluations : (int32, Nx.int32_elt) Nx.t;
  used : (int32, Nx.int32_elt) Nx.t;
  facts : fact list;
}

let code = function
  | Converged -> 0l
  | Budget_spent -> 1l
  | Not_bracketed -> 2l
  | Not_finite -> 3l
  | Stalled -> 4l

let statuses = [ Converged; Budget_spent; Not_bracketed; Not_finite; Stalled ]
let of_code c = List.find (fun s -> Int32.equal (code s) c) statuses

let v ~fn ~settings ?spent ~fix ~value ~error ~status ~evaluations ~facts () =
  let used, spent =
    match spent with
    | Some (s : spent) -> (s.used, Some (s.unit, s.budget))
    | None -> (Nx.zeros_like status, None)
  in
  { fn; settings; fix; spent; value; error; status; evaluations; used; facts }

let map ~fn f t = { t with fn; value = f t.value; error = f t.error }
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
  | Stalled -> "the search stopped without meeting the tolerance"

(* The report of the lane at [i], from its elements: its status, evaluations,
   budget used and facts, and the count of the other elements of its problem
   that converged. *)
let message t i ~status ~evaluations ~used ~converged ~facts =
  let lane =
    if Array.length i = 0 then ""
    else
      "lane ["
      ^ String.concat ", " (Array.to_list (Array.map string_of_int i))
      ^ "]: "
  in
  let status = of_code status in
  let values =
    List.map (fun (name, x) -> Printf.sprintf "%s %g" name x) facts
  in
  let spent =
    match t.spent with
    | Some (unit, budget) -> Printf.sprintf "%ld of %d %s, " used budget unit
    | None -> ""
  in
  String.concat "\n  "
    (List.filter
       (fun l -> l <> "")
       [
         Printf.sprintf "%s: %s%s." t.fn lane (reason status);
         t.settings;
         String.concat ", " values;
         Printf.sprintf "%s%ld evaluations." spent evaluations;
         t.fix status facts;
         (match converged with
         | 0 -> ""
         | 1 -> "1 other element of this problem converged."
         | n -> Printf.sprintf "%d other elements of this problem converged." n);
       ])

let get t =
  let ok = ok t in
  let converged = Nx.sum (Nx.cast Nx.int32 ok) in
  Nx.check
    Nx.Ptree.(pair tensor (pair tensor (pair tensor (pair tensor facts))))
    ok
    (t.status, (t.evaluations, (t.used, (converged, t.facts))))
    (fun i (status, (evaluations, (used, (converged, facts)))) ->
      Failure
        (message t i ~status:(Nx.item [] status)
           ~evaluations:(Nx.item [] evaluations) ~used:(Nx.item [] used)
           ~converged:(Int32.to_int (Nx.item [] converged))
           ~facts:
             (List.map (fun (Fact (name, x)) -> (name, Nx.item [] x)) facts)));
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
      let used = field c "used" tensor t.used in
      let facts = field c "report" (structure facts) t.facts in
      { t with value; error; status; evaluations; used; facts }
  end in
  Nx.Ptree.instantiate (module M)

let name = function
  | Converged -> "converged"
  | Budget_spent -> "budget spent"
  | Not_bracketed -> "not bracketed"
  | Not_finite -> "not finite"
  | Stalled -> "stalled"

(* Read on the host: the counts, then the first failing lane's report. *)
let pp ppf t =
  let host x = Nx.to_array (Nx.reshape [| -1 |] x) in
  let shape = Nx.shape t.status in
  let codes = host t.status in
  let count st =
    Array.fold_left
      (fun n c -> if Int32.equal c (code st) then n + 1 else n)
      0 codes
  in
  let parts =
    List.filter_map
      (fun st ->
        let n = count st in
        if n = 0 then None else Some (Printf.sprintf "%d %s" n (name st)))
      statuses
  in
  Format.fprintf ppf "%s: %s" t.fn (String.concat ", " parts);
  let failing = ref None in
  Array.iteri
    (fun k c ->
      if !failing = None && not (Int32.equal c (code Converged)) then
        failing := Some k)
    codes;
  match !failing with
  | None -> ()
  | Some k ->
      let index =
        let i = Array.make (Array.length shape) 0 and r = ref k in
        for a = Array.length shape - 1 downto 0 do
          i.(a) <- !r mod shape.(a);
          r := !r / shape.(a)
        done;
        i
      in
      let at x = (host x).(k) in
      let facts =
        List.map
          (fun (Fact (name, x)) ->
            let x = Nx.cast Nx.float64 (Nx.broadcast_to shape x) in
            (name, (host x).(k)))
          t.facts
      in
      Format.fprintf ppf "@\n%s"
        (message t index ~status:codes.(k) ~evaluations:(at t.evaluations)
           ~used:(at t.used) ~converged:(count Converged) ~facts)
