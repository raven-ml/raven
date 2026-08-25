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

   Because the carry and the scanned values are existentially typed through the
   effect, they travel packed; packing and unpacking sites are the only places
   that erase and recover the types, and both are construction sites of the same
   module, so the coercion is sound by construction. *)

type packed_t = Packed_t : ('a, 'b) Nx_effect.t -> packed_t

type c_packed =
  | Packed_c : (module Nx.Ptree.S with type t = 'c) * 'c -> c_packed

(* One fold step, type-erased: [run] applies the scan body to a packed carry and
   element, returning the packed next carry and output. *)
type step = { run : c_packed -> packed_t -> c_packed * packed_t }
type scan_res = { r_carry : c_packed; r_y : packed_t }
type scan_req = { req_carry : c_packed; req_x : packed_t; req_step : step }

(* A backward scan, performed by the tape entry reverse-mode records for a
   staged [E_scan]. Carries everything jit needs to capture the body's pullback
   as a sub-program and emit the reversed loop: the body re-runner, the forward
   pass's inputs (to accumulate into), the cotangents of the scan's outputs, and
   the static shapes. *)
type scan_bwd = {
  bwd_step : step;
  bwd_carry : c_packed; (* the scan's init carry, accumulated into *)
  bwd_x : packed_t; (* the scan's input sequence, accumulated into *)
  bwd_n : int;
  bwd_dc : c_packed; (* cotangent of the final carry, a structure *)
  bwd_dy : packed_t; (* cotangent of the stacked outputs *)
  bwd_y_shape : int array; (* shape of one element of the output stack *)
}

(* A tensor the scan body closes over (an external input of the loop) and the
   cotangent accumulated for it across the backward loop's iterations. The pair
   binds one existential instance, so the two tensors share a dtype by
   construction. *)
type closed_ctan =
  | Closed_ctan : ('a, 'b) Nx_effect.t * ('a, 'b) Nx_effect.t -> closed_ctan

(* Result of a staged backward scan: the cotangent of the init carry, the
   cotangent of the input sequence, and the cotangents of the tensors the body
   closes over. *)
type scan_bwd_res = {
  br_carry : c_packed;
  br_y : packed_t;
  br_closed : closed_ctan list;
}

(* [E_scan_probe] asks: will the nearest [E_scan] claimer stage the scan as a
   compiled loop? Every handler with an [E_scan] case must also answer the
   probe: a staging jit answers [true]; a transformation handler answers
   [false], because its own [E_scan] case intercepts the scan before any stager
   above it could. Reverse-mode asks before re-performing [E_scan] — it records
   a staged-transpose tape entry (an [E_scan_bwd] only a staging jit can
   answer) exactly when the probe says [true], and otherwise folds eagerly so
   every step is taped. Unhandled means [false]. *)
type _ Effect.t +=
  | E_scan : scan_req -> scan_res Effect.t
  | E_scan_bwd : scan_bwd -> scan_bwd_res Effect.t
  | E_scan_probe : bool Effect.t

(* A stager may decline an [E_scan] it claimed when tracing the body reveals a
   loop it cannot compile (a carry whose shape changes across steps): it
   discontinues the scan with [Not_staged]. Every performer of [E_scan] must
   treat [Not_staged] like [Effect.Unhandled] and fold eagerly — the probe is
   an optimistic answer, not a promise. *)
exception Not_staged

(* The eager fold, over the packed representation. Runs the body with ordinary
   Nx operations, so an enclosing handler (or a nested one installed by a
   handler's own [E_scan] case) observes every step, exactly as the historical
   unrolled scan did. *)
let eager (req : scan_req) : scan_res =
  let (Packed_t xs) = req.req_x in
  let shape = Nx.shape xs in
  let n = shape.(0) in
  let carry = ref req.req_carry in
  let ys = ref [] in
  for i = 0 to n - 1 do
    let xi = Nx.slice [ Nx.I i ] xs in
    let c', y = req.req_step.run !carry (Packed_t xi) in
    carry := c';
    ys := y :: !ys
  done;
  (* All steps return the same element type, so the first element unpacks the
     witness the stack is built under; the rest are coerced against it. *)
  let rec stack_list : type a b. (a, b) Nx.t -> packed_t list -> packed_t =
   fun y0 rest ->
    let rest =
      List.map (fun (Packed_t y) -> (Obj.magic y : (a, b) Nx.t)) rest
    in
    Packed_t (Nx.stack ~axis:0 (y0 :: rest))
  in
  match List.rev !ys with
  | Packed_t y0 :: rest -> { r_carry = !carry; r_y = stack_list y0 rest }
  | [] -> assert false

(* [scan] itself. The typed body is packed into [step] with a locally abstract
   type witness; the effect result is unpacked back. *)

let scan (type c) (module C : Nx.Ptree.S with type t = c)
    ~(f : c -> ('a, 'b) Nx.t -> c * ('d, 'e) Nx.t) ~(init : c)
    (xs : ('a, 'b) Nx.t) : c * ('d, 'e) Nx.t =
  let shape = Nx.shape xs in
  if Array.length shape = 0 then
    invalid_arg "Rune.scan: xs must have a leading scan axis";
  let n = shape.(0) in
  if n = 0 then invalid_arg "Rune.scan: xs is empty along the scan axis";
  let step : step =
    {
      run =
        (fun (Packed_c (_, c)) (Packed_t x) ->
          let c = (Obj.magic c : c) in
          let x = (Obj.magic x : ('a, 'b) Nx.t) in
          let c', y = f c x in
          (Packed_c ((module C), c'), Packed_t y));
    }
  in
  let req =
    {
      req_carry = Packed_c ((module C), init);
      req_x = Packed_t xs;
      req_step = step;
    }
  in
  match Effect.perform (E_scan req) with
  | res -> (
      match res with
      | { r_carry = Packed_c (_, c'); r_y = Packed_t y } ->
          ((Obj.magic c' : c), (Obj.magic y : ('d, 'e) Nx.t)))
  | exception (Effect.Unhandled _ | Not_staged) -> (
      (* No staging handler (none claims the effect, or the claimer declined):
         the eager fold, observed by whatever transformation handlers are
         installed. *)
      let res = eager req in
      match res with
      | { r_carry = Packed_c (_, c'); r_y = Packed_t y } ->
          ((Obj.magic c' : c), (Obj.magic y : ('d, 'e) Nx.t)))
