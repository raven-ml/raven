(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* JIT compilation.

   Three-phase execution: warmup (cnt=0) runs eagerly, capture (cnt=1)
   records the computation as a LINEAR, exec (cnt>=2) replays that LINEAR
   with fresh input buffers.

   Capture installs itself in {!Realize.capturing}, so every schedule the
   function creates is recorded instead of executed. The recorded schedules
   are combined and lowered for replay: each input buffer node is substituted
   with a slotted PARAM, intermediate buffer memory is planned once over the
   combined LINEAR (buffers the caller holds keep their identity), and every
   kernel is compiled. Replay passes the current input buffer nodes to
   {!Realize.run_linear} as [input_uops], so PARAM slots resolve to the
   buffers backing the current inputs, and threads the per-call [var_vals]
   through to kernel launches. *)

open Tolk_uop
module U = Uop
module B = Device.Buffer

let debug = Helpers.getenv "DEBUG" 0

let command_output cmd =
  let ic = Unix.open_process_in cmd in
  Fun.protect
    ~finally:(fun () -> ignore (Unix.close_process_in ic))
    (fun () ->
      match input_line ic with
      | line -> String.trim line
      | exception End_of_file -> "")

let default_jit_level =
  let system = command_output "uname -s" in
  let machine = command_output "uname -m" in
  if
    String.equal system "Darwin"
    && List.mem machine [ "Intel"; "i386"; "x86_64" ]
  then 2
  else 1

let jit_level = Helpers.getenv "JIT" default_jit_level
let jit_batch_size = Helpers.getenv "JIT_BATCH_SIZE" 32
let graph_one_kernel = Helpers.getenv "GRAPH_ONE_KERNEL" 0 <> 0
let is_op op n = Ops.equal (U.op n) op

exception Jit_error of string

(* Buffer arguments of a scheduled call, dropping symbolic binds. *)
let call_arg_uops args = List.filter (fun s -> not (is_op Ops.Bind s)) args

let call_args call =
  match U.as_call call with Some { args; _ } -> args | None -> []

let call_body call =
  match U.as_call call with Some { body; _ } -> Some body | None -> None

(* Validation token: inputs must keep their size, dtype, and device across
   replays. *)
type input_info = {
  ii_size : int;
  ii_dtype : Dtype.t;
  ii_device : U.device option;
}

let input_info_of_uop u =
  {
    ii_size = List.fold_left ( * ) 1 (U.max_shape u);
    ii_dtype = U.dtype u;
    ii_device = U.device_of u;
  }

(* Graph batching

   Groups consecutive graph-compatible calls of a compiled LINEAR into
   CUSTOM_FUNCTION "graph" calls so replay dispatches each group as one
   batched launch through the device's {!Device.Graph} capability. A call is
   compatible when its body is a PROGRAM (or a COPY, if the capability
   supports copies) and every buffer argument lives on the batch's device;
   any other call breaks the batch. SLICE calls are dropped: they only bind
   offset views, which argument resolution derives structurally. The batch
   size limit doubles after each emitted graph. *)

let dedup xs =
  List.rev
    (List.fold_left (fun acc x -> if List.memq x acc then acc else x :: acc)
       [] xs)

(* All external inputs of the batch become the graph call's arguments. *)
let create_graph_call batch =
  let input_list =
    dedup
      (List.concat_map
         (fun si ->
           List.concat_map
             (fun arg ->
               List.filter
                 (fun u ->
                   match U.as_param u with
                   | Some { param = { slot; addrspace; _ }; _ } ->
                       slot >= 0 && addrspace <> Dtype.Alu
                   | None -> false)
                 (U.toposort arg))
             (call_args si))
         batch)
  in
  let cf = U.custom_function ~name:"graph" ~srcs:[ U.linear batch ] in
  let info : U.call_info =
    {
      grad_fxn = None;
      name = None;
      precompile = false;
      precompile_backward = false;
      aux = None;
    }
  in
  (* Compiled PROGRAM bodies keep their internal ranges (launch axes have no
     END), so the batched call is assembled with [U.replace], like
     {!Realize.pm_compile}, instead of re-running [U.call]'s range check. *)
  let call =
    U.call ~body:(U.custom_function ~name:"graph" ~srcs:[]) ~args:input_list
      ~info
  in
  U.replace call ~src:(Array.of_list (cf :: input_list)) ()

let device_prefix name =
  match String.index_opt name ':' with
  | Some i -> String.sub name 0 i
  | None -> name

(* [Some prefixes] of the devices a call's buffer arguments live on, or
   [None] when an argument is multi-device (never graphed). *)
let call_device_prefixes si =
  let rec loop acc = function
    | [] -> Some (dedup (List.rev acc))
    | b :: rest ->
        if is_op Ops.Bind b then loop acc rest
        else (
          match U.device_of b with
          | Some (U.Single d) -> loop (device_prefix d :: acc) rest
          | Some (U.Multi _ | U.Index _) -> None
          | None -> loop acc rest)
  in
  loop [] (call_args si)

let graph_split_rewrite ~device linear ~max_batch_size =
  let graph = Device.graph device in
  let graph_prefix = device_prefix (Device.name device) in
  let new_src = ref [] and batch = ref [] and batch_len = ref 0 in
  let max_batch_size = ref max_batch_size in
  let flush_batch () =
    (match List.rev !batch with
    | ([] | [ _ ]) as b when not graph_one_kernel ->
        new_src := List.rev_append b !new_src
    | b ->
        new_src := create_graph_call b :: !new_src;
        max_batch_size := !max_batch_size * 2;
        if debug >= 2 then
          Printf.eprintf "JIT GRAPHing batch with %d kernels\n%!"
            (List.length b));
    batch := [];
    batch_len := 0
  in
  List.iter
    (fun si ->
      match call_body si with
      | Some body when is_op Ops.Slice body -> ()
      | Some body ->
          let can_graph =
            (match graph with
            | Some g ->
                is_op Ops.Program body
                || (is_op Ops.Copy body && g.Device.Graph.supports_copy)
            | None -> false)
            &&
            match call_device_prefixes si with
            | Some prefixes ->
                List.for_all (String.equal graph_prefix) prefixes
            | None -> false
          in
          let can_extend =
            can_graph && (!max_batch_size = 0 || !batch_len < !max_batch_size)
          in
          if (not can_extend) && !batch <> [] then flush_batch ();
          if can_graph then begin
            batch := si :: !batch;
            incr batch_len
          end
          else new_src := si :: !new_src
      | None -> new_src := si :: !new_src)
    (U.children linear);
  if !batch <> [] then flush_batch ();
  U.linear (List.rev !new_src)

(* Graph batching for a compiled LINEAR, honoring the JIT level: the identity
   when batching is disabled (JIT >= 2). Also the entry point for callers that
   drive Realize directly (rune's jit) instead of going through [call]. *)
let batch_graphs ~device linear =
  if jit_level < 2 then
    graph_split_rewrite ~device linear ~max_batch_size:jit_batch_size
  else linear

(* Lower a captured LINEAR for replay: substitute each input buffer node with
   a PARAM carrying its slot index, plan intermediate buffer memory once over
   the combined schedule with [held_bufs] kept intact, compile every kernel,
   and batch graph-compatible calls. *)
let jit_lower ~device ~to_program linear held_bufs (input_uops : U.t array) =
  let mappings =
    List.mapi
      (fun i u ->
        let shape =
          match U.as_buffer u with
          | Some { shape; _ } -> Some shape
          | None -> None
        in
        ( u,
          U.param ~slot:i ~dtype:(U.dtype u) ?shape ?device:(U.device_of u) ()
        ))
      (Array.to_list input_uops)
  in
  let linear = U.substitute ~walk:true mappings linear in
  let linear = Schedule.memory_plan_rewrite linear held_bufs in
  let linear = Realize.pm_compile ~device ~to_program linear in
  batch_graphs ~device linear

(* Captured schedule *)

type 'a captured_jit = {
  ret : 'a;
  linear : U.t;
  device : Device.t;
  to_program : U.t -> U.t;
  binding : Realize.Buffers.t;
  expected_input_info : input_info array;
}

(* The direct calls of a lowered LINEAR, looking through graph batches: a
   CUSTOM_FUNCTION "graph" call stands for the calls of its LINEAR body. *)
let flatten_graph_calls linear =
  List.concat_map
    (fun call ->
      match call_body call with
      | Some body
        when is_op Ops.Custom_function body
             && U.Arg.as_string (U.arg body) = Some "graph" -> (
          match U.children body with
          | [ inner ] -> U.children inner
          | _ -> [ call ])
      | _ -> [ call ])
    (U.children linear)

(* Bind every non-input buffer argument the resolver knows to its concrete
   buffer, once at capture: weights, outputs, and held buffers keep their
   storage across replays. Planned intermediates are slices of arena buffers
   the binding allocates lazily; input PARAMs resolve per call. *)
let seed_known_buffers binding ~buffers linear =
  List.iter
    (fun call ->
      List.iter
        (fun arg ->
          let node = U.buf_uop arg in
          if not (Realize.Buffers.mem binding node) then
            match buffers node with
            | Some buf -> Realize.Buffers.seed binding node buf
            | None -> ())
        (call_arg_uops (call_args call)))
    (flatten_graph_calls linear)

let validate_inputs t (input_uops : U.t array) =
  let n = Array.length t.expected_input_info in
  if Array.length input_uops <> n then
    raise
      (Jit_error
         (Printf.sprintf "input count mismatch: expected %d, got %d" n
            (Array.length input_uops)));
  Array.iteri
    (fun i info ->
      let got = input_info_of_uop input_uops.(i) in
      if
        got.ii_size <> info.ii_size
        || not (Dtype.equal got.ii_dtype info.ii_dtype)
        || got.ii_device <> info.ii_device
      then
        raise
          (Jit_error
             (Printf.sprintf "input %d mismatch: expected (%d, %s)" i
                info.ii_size
                (Dtype.to_string info.ii_dtype))))
    t.expected_input_info

(* Replay the captured LINEAR: bind the current inputs, run with PARAM slots
   resolving through them, then release the input bindings so stale input
   buffers do not stay reachable. *)
let exec_captured ?(wait = false) t (input_uops : U.t array) var_vals ~buffers
    =
  validate_inputs t input_uops;
  Array.iter
    (fun u ->
      match buffers u with
      | Some buf -> Realize.Buffers.seed t.binding u buf
      | None ->
          raise
            (Jit_error
               (Format.asprintf "input %a has no backing buffer" U.pp u)))
    input_uops;
  Fun.protect
    ~finally:(fun () ->
      Array.iter (fun u -> Realize.Buffers.remove t.binding u) input_uops)
    (fun () ->
      Realize.run_linear ~device:t.device ~to_program:t.to_program t.binding
        ~var_vals ~input_uops ~jit:true ~wait t.linear);
  t.ret

(* TinyJit *)

type 'a tiny_jit = {
  fxn : (U.t array -> (string * int) list -> 'a) option;
  device : Device.t;
  to_program : U.t -> U.t;
  mutable captured : 'a captured_jit option;
  mutable cnt : int;
}

let captured t = t.captured

let create ~device ~to_program ?fxn ?captured ?prune:_ () =
  if Option.is_none fxn && Option.is_none captured then
    invalid_arg "need either a function or a CapturedJit";
  let cnt = if fxn = None then 2 else 0 in
  { fxn; device; to_program; captured; cnt }

let reset t =
  if t.fxn = None then invalid_arg "can't reset without function";
  t.cnt <- 0;
  t.captured <- None

(* Flatten the captured linears into one, inlining nested LINEAR nodes. *)
let combine_linears linears =
  U.linear
    (List.concat_map
       (fun l -> if is_op Ops.Linear l then U.children l else [ l ])
       linears)

let call ?wait ?held_buffers t (input_uops : U.t array)
    (var_vals : (string * int) list) ~(buffers : U.t -> B.t option) =
  let ret =
    if jit_level = 0 || t.cnt = 0 then
      (* Warmup: execute eagerly. *)
      (Option.get t.fxn) input_uops var_vals
    else if t.cnt = 1 then begin
      (* Capture: record the linears the function schedules. *)
      let fxn = Option.get t.fxn in
      if !Realize.capturing <> [] then
        raise (Jit_error "nested TinyJit is not supported");
      let linears = ref [] in
      Realize.capturing :=
        [ (fun linear _var_vals -> linears := linear :: !linears) ];
      let ret =
        Fun.protect
          ~finally:(fun () -> Realize.capturing := [])
          (fun () -> fxn input_uops var_vals)
      in
      let linears = List.rev !linears in
      if linears = [] then raise (Jit_error "didn't JIT anything!");
      if debug >= 1 then
        Printf.eprintf "JIT captured %d linears with %d inputs\n%!"
          (List.length linears) (Array.length input_uops);
      let held_bufs = match held_buffers with Some f -> f () | None -> [] in
      let linear =
        jit_lower ~device:t.device ~to_program:t.to_program
          (combine_linears linears) held_bufs input_uops
      in
      let binding = Realize.Buffers.create ~device:t.device in
      seed_known_buffers binding ~buffers linear;
      let captured =
        {
          ret;
          linear;
          device = t.device;
          to_program = t.to_program;
          binding;
          expected_input_info = Array.map input_info_of_uop input_uops;
        }
      in
      t.captured <- Some captured;
      exec_captured ?wait captured input_uops var_vals ~buffers
    end
    else
      (* Exec: replay the captured schedule. *)
      exec_captured ?wait (Option.get t.captured) input_uops var_vals ~buffers
  in
  t.cnt <- t.cnt + 1;
  ret
