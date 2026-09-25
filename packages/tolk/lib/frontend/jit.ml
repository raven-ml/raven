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
  outputs : 'a -> T.t list;
  mutable symbolic_outputs : (T.t * U.t * (U.t * int64) list) list option;
  mutable expected_inputs : (U.t * U.t list * Dtype.t) array option;
  mutable inner : 'a Tolk.Jit.tiny_jit option;
  mutable current : (T.t array * U.t array) option;
      (* Arguments of the in-flight call, read by the engine-facing function
         during warmup and capture. *)
}

let create ~outputs fxn =
  { fxn; outputs; symbolic_outputs = None; expected_inputs = None;
    inner = None; current = None }

(* The engine JIT is created on first call so that constructing a JIT does
   not open the execution device. *)
let inner t =
  match t.inner with
  | Some jit -> jit
  | None ->
      let device = Run.device () in
      let to_program device =
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
  t.current <- None;
  t.symbolic_outputs <- None;
  t.expected_inputs <- None

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
  let var_name var =
    match U.Arg.as_param_arg (U.arg var) with
    | Some { name = Some name; _ } -> name
    | _ -> raise (Jit_error "JIT vars must bind named variables")
  in
  let var_vals = ref [] in
  let add_binding (var, value) =
    let name = var_name var in
    match List.assoc_opt name !var_vals with
    | Some prev when prev <> value ->
        raise
          (Jit_error
             (Printf.sprintf "conflicting values for JIT var %s: %Ld and %Ld"
                name prev value))
    | Some _ -> ()
    | None -> var_vals := (name, value) :: !var_vals
  in
  let input_info = Array.map (fun tensor ->
      let node = T.uop tensor in
      let view = U.substitute [U.base node, U.noop ()] node
        |> U.graph_rewrite
             (Upat.Pattern_matcher.rewrite Tolk_uop.Movement.mop_cleanup) in
      let bindings = List.filter_map (fun bound ->
          if U.is_bound_var bound then Some (bound, U.unbind bound)
          else None) (U.toposort view) in
      List.iter (fun (_, binding) -> add_binding binding) bindings;
      let variables = List.map (fun (_, (var, _)) -> var) bindings
        |> List.sort_uniq U.compare
        |> List.sort (fun a b -> String.compare (var_name a) (var_name b)) in
      let view = U.substitute ~walk:true
          (List.map (fun (bound, (var, _)) -> bound, var) bindings) view in
      view, variables, U.dtype node) tensors in
  Array.iter (fun bind ->
      if not (U.is_bound_var bind) then
        raise (Jit_error "JIT vars must be bound variables");
      add_binding (U.unbind bind)) vars;
  (input_uops, List.rev !var_vals, input_info)

(* Buffers that must survive replay with their own storage: every buffer node
   with concrete device storage, plus every buffer node still reachable from
   a live tensor. Anything else in the captured schedule is an intermediate
   the memory planner folds into reused arena memory. *)
let held_buffers () =
  let held = Hashtbl.create 64 in
  let add node = Hashtbl.replace held (U.tag node) node in
  List.iter
    (fun tensor ->
      List.iter
        (fun node -> if Ops.equal (U.op node) Ops.Buffer then add node)
        (U.toposort (T.uop tensor)))
    (T.live_tensors ());
  Hashtbl.fold (fun _ node acc -> node :: acc) held []

(* Keep the captured graph as the source for every rebind. Rewriting the
   previous result would freeze variables simplified away by an earlier call. *)
let refresh_outputs t ret var_vals =
  let symbolic = match t.symbolic_outputs with
    | Some outputs -> outputs
    | None ->
        let outputs = List.filter_map (fun tensor ->
            let node = T.uop tensor in
            let bindings = List.filter_map (fun bound ->
                if U.is_bound_var bound then Some (bound, U.unbind bound)
                else None) (U.toposort node) in
            if bindings = [] then None
            else Some (tensor,
                U.substitute ~walk:true
                  (List.map (fun (bound, (var, _)) -> bound, var) bindings) node,
                List.map snd bindings)) (t.outputs ret) in
        t.symbolic_outputs <- Some outputs;
        outputs in
  List.iter (fun (tensor, node, bindings) ->
      let replacements = List.map (fun (var, captured_value) ->
          let value = match U.Arg.as_param_arg (U.arg var) with
            | Some {name = Some name; _} ->
                Option.value (List.assoc_opt name var_vals) ~default:captured_value
            | _ -> captured_value in
          var, U.bind ~var ~value:(U.const (Const.int64 Dtype.weakint value))) bindings in
      T.set_uop tensor (U.substitute ~walk:true replacements node)) symbolic

let call ?(vars = [||]) t tensors =
  let input_uops, var_vals, input_info = prepare_inputs tensors vars in
  (match t.expected_inputs with
  | Some expected when not (Array.equal
      (fun (view, variables, dtype) (other_view, other_variables, other_dtype) ->
        U.equal view other_view && List.equal U.equal variables other_variables
        && Dtype.equal dtype other_dtype)
      expected input_info) ->
      raise (Jit_error "input view or symbolic variable mismatch with JIT capture")
  | _ -> ());
  let jit = inner t in
  t.current <- Some (tensors, vars);
  Fun.protect
    ~finally:(fun () -> t.current <- None)
    (fun () ->
      let ret = Tolk.Jit.call jit input_uops var_vals ~held_buffers in
      if captured t then begin
        if Option.is_none t.expected_inputs then
          t.expected_inputs <- Some input_info;
        refresh_outputs t ret var_vals
      end;
      ret)
