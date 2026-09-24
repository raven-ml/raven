(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Staged scan.

   [Rune.scan] is an eager fold. Under a jit trace, staging it as a loop in the
   compiled program — instead of an unrolled trace — requires the fold step to
   be captured once as a compiled sub-program and the scan itself to become a
   loop construct at the schedule level. [scan] therefore performs the [E_scan]
   effect; transformation handlers that cannot stage it (everything but jit and
   reverse) fall back to the eager fold, as does plain execution when no handler
   is present ([Effect.Unhandled] is catchable since OCaml 5.2).

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

(* [req_record] asks the stager to keep the carry entering every step, which a
   staged transpose reads: reverse-mode sets it when it records one. *)
type scan_req = {
  req_carry : leaves;
  req_xs : leaves;
  req_step : step;
  req_record : bool;
}

type scan_res = { r_carry : leaves; r_ys : leaves }

(* A backward scan, performed by the tape entry reverse-mode records for a
   staged [E_scan]. Carries everything jit needs to capture the body's pullback
   as a sub-program and emit the reversed loop: the body re-runner, the forward
   pass's inputs (to accumulate into), and the cotangents of the scan's
   outputs. *)
type scan_bwd = {
  bwd_step : step;
  bwd_carry : leaves; (* the scan's init carry, accumulated into *)
  bwd_xs : leaves; (* the scan's rows, accumulated into *)
  bwd_dc : leaves; (* cotangent of the final carry *)
  bwd_dys : leaves; (* cotangents of the stacked outputs *)
}

(* A tensor the scan body closes over (an external input of the loop) and the
   cotangent accumulated for it across the backward loop's iterations. The pair
   binds one existential instance, so the two tensors share a dtype by
   construction. *)
type closed_ctan =
  | Closed_ctan : ('a, 'b) Nx_effect.t * ('a, 'b) Nx_effect.t -> closed_ctan

(* Result of a staged backward scan: the cotangents of the init carry, of the
   rows (stacked like them), and of the tensors the body closes over. *)
type scan_bwd_res = {
  br_carry : leaves;
  br_xs : leaves;
  br_closed : closed_ctan list;
}

(* [E_scan_probe] asks: will the nearest [E_scan] claimer stage the scan as a
   compiled loop? Every handler with an [E_scan] case must also answer the
   probe: a staging jit answers [true]; a transformation handler answers
   [false], because its own [E_scan] case intercepts the scan before any stager
   above it could. Reverse-mode asks before re-performing [E_scan] — it records
   a staged-transpose tape entry (an [E_scan_bwd] only a staging jit can answer)
   exactly when the probe says [true], and otherwise folds eagerly so every step
   is taped. Unhandled means [false]. *)
type _ Effect.t +=
  | E_scan : scan_req -> scan_res Effect.t
  | E_scan_bwd : scan_bwd -> scan_bwd_res Effect.t
  | E_scan_probe : bool Effect.t

(* A stager may decline an [E_scan] it claimed when tracing the body reveals a
   loop it cannot compile (a carry whose shape changes across steps): it
   discontinues the scan with [Not_staged]. Every performer of [E_scan] must
   treat [Not_staged] like [Effect.Unhandled] and fold eagerly — the probe is an
   optimistic answer, not a promise. *)
exception Not_staged

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
  let ys = ref [] in
  for i = 0 to n - 1 do
    let row =
      List.map (fun (Nx.P l) -> Nx.P (Nx.slice [ Nx.I i ] l)) req.req_xs
    in
    let c', y = req.req_step.run !carry row in
    carry := c';
    ys := y :: !ys
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
  { r_carry = !carry; r_ys = stack (List.rev !ys) }

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
  let run c_leaves x_leaves =
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
  let req = { req_carry; req_xs; req_step = { run }; req_record = false } in
  let res =
    match Effect.perform (E_scan req) with
    | res -> res
    | exception (Effect.Unhandled _ | Not_staged) ->
        (* No staging handler (none claims the effect, or the claimer declined):
           the eager fold, observed by whatever transformation handlers are
           installed. *)
        eager req
  in
  let y0 =
    match !first with
    | Some y0 -> y0
    | None -> assert false (* Every performer runs the body at least once. *)
  in
  ( Nx.Ptree.rebuild cs ~like:init res.r_carry,
    Nx.Ptree.rebuild ys_s ~like:y0 res.r_ys )
