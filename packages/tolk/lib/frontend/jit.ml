(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Tensor-level JIT capture and replay.

   Thin adapter between the tensor surface and the engine JIT: it prepares
   the call's inputs (realize, resolve to buffer nodes, unbind variables) and
   delegates the warmup/capture/replay phasing to {!Tolk.Jit}. No tracing
   happens here — capture works because {!Run.realize} schedules through
   {!Tolk.Schedule.create_linear_with_vars}, which hands its linears to the
   installed capturer instead of executing them. *)

open Tolk_uop
module U = Uop
module T = Tensor

exception Jit_error = Tolk.Jit.Jit_error

type 'a t = {
  fxn : T.t array -> vars:U.t array -> 'a;
  mutable inner : 'a Tolk.Jit.tiny_jit option;
  mutable current : (T.t array * U.t array) option;
      (* Arguments of the in-flight call, read by the engine-facing function
         during warmup and capture. *)
}

let create fxn = { fxn; inner = None; current = None }

(* The engine JIT is created on first call so that constructing a JIT does
   not open the execution device. *)
let inner t =
  match t.inner with
  | Some jit -> jit
  | None ->
      let device = Run.device () in
      let to_program =
        Tolk.Codegen.to_program device (Tolk.Device.renderer device)
      in
      let jit =
        Tolk.Jit.create ~device ~to_program
          ~fxn:(fun _input_uops _var_vals ->
            match t.current with
            | Some (tensors, vars) -> t.fxn tensors ~vars
            | None -> invalid_arg "Jit: function called outside call")
          ()
      in
      t.inner <- Some jit;
      jit

let captured t =
  match t.inner with
  | Some jit -> Option.is_some (Tolk.Jit.captured jit)
  | None -> false

let reset t =
  (match t.inner with Some jit -> Tolk.Jit.reset jit | None -> ());
  t.current <- None

let is_realized tensor =
  match U.runtime_realization_state (T.uop tensor) with
  | U.Never_realized -> false
  | U.Runtime_dependent bufs ->
      List.for_all (fun b -> Run.buffer_of_node b <> None) bufs

(* Prepare the call: realize any unrealized input, resolve each input to its
   backing buffer node (rejecting duplicates and non-buffer inputs), and
   unbind each variable into a named value. *)
let prepare_inputs tensors vars =
  (match List.filter (fun x -> not (is_realized x)) (Array.to_list tensors) with
  | [] -> ()
  | unrealized -> Run.realize_many unrealized);
  let input_uops =
    Array.map
      (fun x ->
        let node = U.buf_uop (T.uop x) in
        if
          (not (Ops.equal (U.op node) Ops.Buffer))
          || Run.buffer_of_node node = None
        then raise (Jit_error "JIT inputs must be realized buffers");
        node)
      tensors
  in
  let seen = Hashtbl.create (Array.length input_uops) in
  Array.iter
    (fun node ->
      if Hashtbl.mem seen (U.tag node) then
        raise (Jit_error "duplicate inputs to JIT");
      Hashtbl.add seen (U.tag node) ())
    input_uops;
  let var_vals =
    Array.fold_left
      (fun acc bind ->
        let var, value =
          match U.as_bind bind with
          | Some _ -> U.unbind bind
          | None -> raise (Jit_error "JIT vars must be bound variables")
        in
        let name =
          match U.as_param var with
          | Some { param = { name = Some name; _ }; _ } -> name
          | _ -> raise (Jit_error "JIT vars must bind named variables")
        in
        (match List.assoc_opt name acc with
        | Some prev when prev <> value ->
            raise
              (Jit_error
                 (Printf.sprintf "conflicting values for JIT var %s: %d and %d"
                    name prev value))
        | _ -> ());
        (name, value) :: acc)
      [] vars
  in
  (input_uops, List.rev var_vals)

(* Buffers that must survive replay with their own storage: every buffer node
   with concrete device storage, plus every buffer node still reachable from
   a live tensor. Anything else in the captured schedule is an intermediate
   the memory planner folds into reused arena memory. *)
let held_buffers () =
  let held = Hashtbl.create 64 in
  let add node = Hashtbl.replace held (U.tag node) node in
  List.iter add (Run.buffer_nodes ());
  List.iter
    (fun tensor ->
      List.iter
        (fun node -> if Ops.equal (U.op node) Ops.Buffer then add node)
        (U.toposort (T.uop tensor)))
    (T.live_tensors ());
  Hashtbl.fold (fun _ node acc -> node :: acc) held []

let call ?(vars = [||]) t tensors =
  let input_uops, var_vals = prepare_inputs tensors vars in
  let jit = inner t in
  t.current <- Some (tensors, vars);
  Fun.protect
    ~finally:(fun () -> t.current <- None)
    (fun () ->
      Tolk.Jit.call jit input_uops var_vals ~buffers:Run.buffer_of_node
        ~held_buffers)
