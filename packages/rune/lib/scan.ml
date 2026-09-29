(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Staged scan.

   [Rune.scan] is an eager fold. Under a jit trace, staging it as a loop in the
   compiled program — instead of an unrolled trace — requires the fold step to
   be captured once as a compiled sub-program and the scan itself to become a
   loop construct at the schedule level. [scan] therefore performs the [E_scan]
   effect. jit stages it; reverse, forward and vmap pass it on as the scan of
   their transformation when a stager lies beyond them, and otherwise fold
   eagerly under a nested instance of their handler, as does plain execution
   when no handler is present ([Effect.Unhandled] is catchable since OCaml 5.2).

   jit traces the fold step under its own handler only, so a transformation
   handler passes on a step that installs a nested instance of itself around the
   step it received. The transpose of a scan is a scan too: reverse mode
   records one that runs backward over the step's pullback, which every handler
   around it transforms or stages like any other.

   The carry, the rows and the outputs are structures whose types the effect
   cannot carry, so they travel as their tensors in walk order
   ([Nx.Ptree.flatten]): the typed [scan] rebuilds values from tensors inside
   the fold step and for the result, and every handler works on lists of
   tensors. *)

(* A structure's tensors, in walk order. *)
type leaves = Nx.packed list

(* One fold step, type-erased: [run] applies the scan body to the tensors of a
   carry and of a row, returning those of the next carry and of the outputs. *)
type step = { run : leaves -> leaves -> leaves * leaves }

(* [req_reverse] runs the steps from the last row to the first, as the
   transpose of a scan does. *)
type scan_req = {
  req_carry : leaves;
  req_xs : leaves;
  req_step : step;
  req_reverse : bool;
}

type scan_res = { r_carry : leaves; r_ys : leaves }

(* [E_scan_probe] asks whether an [E_scan] performed here would be staged as a
   compiled loop. Every handler with an [E_scan] case answers it as its case
   behaves: a staging jit answers [true], and a transformation passes it on,
   since it passes the scan on when it stages. Unhandled means [false].

   An exception raised while a handler runs the scan reaches its performer, as
   every handler's does ([Gate.deliver]): one the fold step raises belongs at
   the scan, where the eager fold raises it, and a transformation's fold step
   signals through its own exceptions. *)
type _ Effect.t +=
  | E_scan : scan_req -> scan_res Effect.t
  | E_scan_probe : bool Effect.t

let probe () =
  match Effect.perform E_scan_probe with
  | stages -> stages
  | exception Effect.Unhandled _ -> false

(* A stager may decline an [E_scan] it claimed when tracing the body reveals a
   loop it cannot compile (a carry whose shape changes across steps): it
   discontinues the scan with [Not_staged]. Every performer of [E_scan] must
   treat [Not_staged] like [Effect.Unhandled] and fold eagerly — the probe is an
   optimistic answer, not a promise. *)
exception Not_staged

(* [pass_on ~fold stage] answers an [E_scan] a transformation handler claimed:
   with [stage ()], which passes the scan on as the scan of the transformation,
   when a stager lies beyond, and with [fold ()], the eager fold under a nested
   instance of the handler, when none does or the stager declines. *)
let pass_on ~fold stage =
  if not (probe ()) then fold ()
  else match stage () with res -> res | exception Not_staged -> fold ()

(* [split n l] is the first [n] elements of [l] and the others: a transformed
   scan's leaves are the scan's followed by the ones the transformation adds. *)
let rec split n l =
  if n = 0 then ([], l)
  else
    match l with
    | x :: l ->
        let a, b = split (n - 1) l in
        (x :: a, b)
    | [] -> assert false

(* The number of steps: the common leading length of the rows. *)
let length xs =
  match xs with
  | [] -> invalid_arg "Rune.scan: xs has no leaf"
  | x :: _ ->
      let lead (Nx.P l) =
        match Nx.shape l with
        | [||] -> invalid_arg "Rune.scan: an xs leaf is a scalar"
        | shape -> shape.(0)
      in
      let n = lead x in
      if List.exists (fun l -> lead l <> n) xs then
        invalid_arg "Rune.scan: the xs leaves differ in their leading length";
      if n = 0 then invalid_arg "Rune.scan: xs is empty along the scan axis";
      n

(* The eager fold. Runs the body with ordinary Nx operations, so an enclosing
   handler (or a nested one installed by a handler's own [E_scan] case) observes
   every step. *)
let eager (req : scan_req) : scan_res =
  let n = length req.req_xs in
  let carry = ref req.req_carry in
  let ys = Array.make n [] in
  for k = 0 to n - 1 do
    let i = if req.req_reverse then n - 1 - k else k in
    let row =
      List.map (fun (Nx.P l) -> Nx.P (Nx.slice [ Nx.I i ] l)) req.req_xs
    in
    let c', y = req.req_step.run !carry row in
    carry := c';
    ys.(i) <- y
  done;
  (* The body checks that every step's outputs have the first step's skeleton,
     so the steps' tensors at one position share a dtype. *)
  let rec stack = function
    | [] :: _ | [] -> []
    | steps ->
        let (Nx.P y0) = List.hd (List.hd steps) in
        let dtype = Nx.dtype y0 in
        let column = List.map (fun y -> Nx.unpack dtype (List.hd y)) steps in
        Nx.P (Nx.stack ~axis:0 column) :: stack (List.map List.tl steps)
  in
  { r_carry = !carry; r_ys = stack (Array.to_list ys) }

(* [run req] is [req]'s scan: staged by the stager that claims it, or the eager
   fold, observed by whatever transformation handlers are installed, when none
   claims it or the claimer declines. *)
let run req =
  match Effect.perform (E_scan req) with
  | res -> res
  | exception (Effect.Unhandled (E_scan _) | Not_staged) -> eager req

(* [scan] itself. The body is wrapped in a step over tensors: it rebuilds the
   carry and the row from the tensors it receives, checks the skeletons of what
   the body returns, and flattens them. *)
let scan (type c x y) (cs : c Nx.Ptree.t) (xs_s : x Nx.Ptree.t)
    (ys_s : y Nx.Ptree.t) ~(f : c -> x -> c * y) ~(init : c) (xs : x) : c * y =
  let req_carry, _ = Nx.Ptree.flatten cs init in
  let req_xs, _ = Nx.Ptree.flatten xs_s xs in
  ignore (length req_xs : int);
  let first = ref None in
  let same ~this ~that s x y =
    ignore (Structure.map2 "Rune.scan" s ~this ~that (fun _ t _ -> t) x y)
  in
  let step c_leaves x_leaves =
    let c = Nx.Ptree.rebuild cs ~like:init c_leaves in
    let c', y = f c (Nx.Ptree.rebuild xs_s ~like:xs x_leaves) in
    same cs ~this:"the carry the body returned" ~that:"the carry it received" c'
      c;
    (match !first with
    | None -> first := Some y
    | Some y0 ->
        same ys_s ~this:"a step's outputs" ~that:"the first step's outputs" y y0);
    (fst (Nx.Ptree.flatten cs c'), fst (Nx.Ptree.flatten ys_s y))
  in
  let res =
    run { req_carry; req_xs; req_step = { run = step }; req_reverse = false }
  in
  let y0 =
    match !first with
    | Some y0 -> y0
    | None -> assert false (* Every performer runs the body at least once. *)
  in
  ( Nx.Ptree.rebuild cs ~like:init res.r_carry,
    Nx.Ptree.rebuild ys_s ~like:y0 res.r_ys )
