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

   The carry, the rows and the outputs are structures whose types are
   existential through the effect, so they travel packed with their traversal;
   packing and unpacking sites are the only places that erase and recover the
   types, and both are construction sites of the same module, so the coercion is
   sound by construction. *)

type packed_t = Packed_t : ('a, 'b) Nx_effect.t -> packed_t
type tree = Tree : (module Nx.Ptree.S with type t = 'a) * 'a -> tree

(* The leaves of a structure, in its traversal order. *)
let leaves (Tree ((module T), t)) =
  let acc = ref [] in
  T.iter (fun leaf -> acc := Packed_t leaf :: !acc) t;
  List.rev !acc

(* [unflatten tree ls] is [tree] with its leaves replaced, position for position
   in traversal order, by [ls], each of its position's dtype. A [T.map]
   callback's order is instance-defined, so positions are recovered through
   fresh markers: [T.map] replaces every leaf with its own marker, [T.iter]
   numbers the markers, and a second [T.map] looks each one up. *)
let unflatten (type a) (module T : Nx.Ptree.S with type t = a) (t : a)
    (ls : packed_t list) : a =
  let marked =
    T.map
      (fun leaf ->
        Nx_effect.symbolic (Nx_effect.context leaf) (Nx_effect.dtype leaf)
          (Nx.shape leaf))
      t
  in
  let positions = ref [] and i = ref 0 in
  T.iter
    (fun m ->
      positions := (Obj.repr m, !i) :: !positions;
      incr i)
    marked;
  let ls = Array.of_list ls in
  if Array.length ls <> !i then
    invalid_arg "Scan.unflatten: leaf count mismatch";
  T.map
    (fun (type b c) (m : (b, c) Nx_effect.t) : (b, c) Nx_effect.t ->
      let (Packed_t l) = ls.(List.assq (Obj.repr m) !positions) in
      (Obj.magic l : (b, c) Nx_effect.t))
    marked

(* One fold step, type-erased: [run] applies the scan body to a packed carry and
   a packed row, returning the packed next carry and outputs. *)
type step = { run : tree -> tree -> tree * tree }

(* [req_record] asks the stager to keep the carry entering every step, which a
   staged transpose reads: reverse-mode sets it when it records one. *)
type scan_req = {
  req_carry : tree;
  req_xs : tree;
  req_step : step;
  req_record : bool;
}

type scan_res = { r_carry : tree; r_ys : tree }

(* A backward scan, performed by the tape entry reverse-mode records for a
   staged [E_scan]. Carries everything jit needs to capture the body's pullback
   as a sub-program and emit the reversed loop: the body re-runner, the forward
   pass's inputs (to accumulate into), and the cotangents of the scan's
   outputs. *)
type scan_bwd = {
  bwd_step : step;
  bwd_carry : tree; (* the scan's init carry, accumulated into *)
  bwd_xs : tree; (* the scan's rows, accumulated into *)
  bwd_dc : tree; (* cotangent of the final carry *)
  bwd_dys : tree; (* cotangents of the stacked outputs *)
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
  br_carry : tree;
  br_xs : tree;
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
  match leaves xs with
  | [] -> invalid_arg "Rune.scan: xs has no leaf"
  | ls ->
      let lead (Packed_t l) =
        match Nx.shape l with
        | [||] -> invalid_arg "Rune.scan: an xs leaf is a scalar"
        | shape -> shape.(0)
      in
      let n = lead (List.hd ls) in
      if List.exists (fun l -> lead l <> n) ls then
        invalid_arg "Rune.scan: the xs leaves differ in their leading length";
      if n = 0 then invalid_arg "Rune.scan: xs is empty along the scan axis";
      n

(* The eager fold, over the packed representation. Runs the body with ordinary
   Nx operations, so an enclosing handler (or a nested one installed by a
   handler's own [E_scan] case) observes every step. *)
let eager (req : scan_req) : scan_res =
  let (Tree (xmod, xs)) = req.req_xs in
  let module X = (val xmod) in
  let n = length req.req_xs in
  let carry = ref req.req_carry in
  let ys = ref [] in
  for i = 0 to n - 1 do
    let row = Tree (xmod, X.map (fun l -> Nx.slice [ Nx.I i ] l) xs) in
    let c', y = req.req_step.run !carry row in
    carry := c';
    ys := y :: !ys
  done;
  let ys = List.rev !ys in
  (* Every step returns outputs of the same structure and element types, so the
     first step's leaves type the stacks the others are coerced into. *)
  let steps = List.map leaves ys in
  let stack_at k (Packed_t y0) =
    let rest =
      List.map
        (fun ls ->
          let (Packed_t y) = List.nth ls k in
          Obj.magic y)
        (List.tl steps)
    in
    Packed_t (Nx.stack ~axis:0 (y0 :: rest))
  in
  match ys with
  | Tree (ymod, y0) :: _ ->
      let stacked = List.mapi stack_at (List.hd steps) in
      { r_carry = !carry; r_ys = Tree (ymod, unflatten ymod y0 stacked) }
  | [] -> assert false

(* [scan] itself. The typed body is packed into [step] with locally abstract
   type witnesses; the effect result is unpacked back. *)
let scan (type c x y) (module C : Nx.Ptree.S with type t = c)
    (module X : Nx.Ptree.S with type t = x)
    (module Y : Nx.Ptree.S with type t = y) ~(f : c -> x -> c * y) ~(init : c)
    (xs : x) : c * y =
  let req_xs = Tree ((module X), xs) in
  ignore (length req_xs : int);
  let step : step =
    {
      run =
        (fun (Tree (_, c)) (Tree (_, x)) ->
          let c', y = f (Obj.magic c : c) (Obj.magic x : x) in
          (Tree ((module C), c'), Tree ((module Y), y)));
    }
  in
  let req =
    {
      req_carry = Tree ((module C), init);
      req_xs;
      req_step = step;
      req_record = false;
    }
  in
  let unpack { r_carry = Tree (_, c'); r_ys = Tree (_, ys) } =
    ((Obj.magic c' : c), (Obj.magic ys : y))
  in
  match Effect.perform (E_scan req) with
  | res -> unpack res
  | exception (Effect.Unhandled _ | Not_staged) ->
      (* No staging handler (none claims the effect, or the claimer declined):
         the eager fold, observed by whatever transformation handlers are
         installed. *)
      unpack (eager req)
