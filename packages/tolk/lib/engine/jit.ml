(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* JIT compilation.

   Three-phase execution: warmup (cnt=0) runs eagerly, capture (cnt=1)
   records the computation as a LINEAR, exec (cnt>=2) replays that LINEAR
   with fresh input buffers.

   Replay executes the captured LINEAR directly through {!Realize.run_linear}:
   buffer arguments are resolved through a persistent binding that is seeded
   once from the capture-time buffer resolver, then re-seeded with the current
   input buffers on every call. *)

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
let is_op op n = Ops.equal (U.op n) op

exception Jit_error of string

(* Capture state *)

(* Non-empty during JIT capture. The schedule machinery calls [add_linear] to
   record each linear produced while the capture handler is active. *)
let capturing : U.t list ref option ref = ref None
let is_capturing () = Option.is_some !capturing

let add_linear linear =
  match !capturing with
  | None -> failwith "add_linear: not inside a JIT capture"
  | Some linears -> linears := linear :: !linears

(* Buffer arguments of a scheduled call, dropping symbolic binds. *)
let call_arg_uops args = List.filter (fun s -> not (is_op Ops.Bind s)) args

let call_args call =
  match U.as_call call with Some { args; _ } -> args | None -> []

(* Follow src[0] chains through movement ops to the underlying buffer state,
   then take its buffer node, matching how the schedule keys buffer arguments. *)
let rec unwrap_src node =
  match U.op node with
  | Ops.After | Ops.Buffer | Ops.Param | Ops.Mselect | Ops.Mstack | Ops.Bind ->
      node
  | _ -> ( match U.children node with s :: _ -> unwrap_src s | [] -> node)

let call_arg_buffer_node node = U.buf_uop (unwrap_src node)

(* Validation token: inputs must keep their shape, dtype, and device across
   replays. *)
type input_info = { ii_size : int; ii_dtype : Dtype.t; ii_device : string }

let input_info_of_buffer b =
  { ii_size = B.size b; ii_dtype = B.dtype b; ii_device = B.device b }

(* Captured schedule *)

type 'a captured_jit = {
  ret : 'a;
  linear : U.t;
  device : Device.t;
  to_program : U.t -> U.t;
  buffers : U.t -> B.t option;
  binding : Realize.Buffers.t;
  expected_input_info : input_info array;
  mutable input_nodes : (U.t * int) list;
  mutable first_run : bool;
}

(* Seed the binding from the capture-time buffer resolver, recording which
   argument nodes are external inputs so they can be re-seeded per call. Each
   call argument is bound to the buffer its underlying node resolves to; view
   and copy calls re-derive their outputs at run time. Runs once, on the first
   replay. *)
let seed_from_resolver t (input_bufs : B.t array) =
  let id_of_input = Hashtbl.create 16 in
  Array.iteri (fun i b -> Hashtbl.replace id_of_input (B.id b) i) input_bufs;
  let inputs = ref [] in
  List.iter
    (fun call ->
      List.iter
        (fun arg ->
          match t.buffers (call_arg_buffer_node arg) with
          | None -> ()
          | Some buf -> (
              match Hashtbl.find_opt id_of_input (B.id buf) with
              | Some slot -> inputs := (arg, slot) :: !inputs
              | None -> Realize.Buffers.seed t.binding arg buf))
        (call_arg_uops (call_args call)))
    (U.children t.linear);
  t.input_nodes <- !inputs

let seed_inputs t (input_bufs : B.t array) =
  List.iter
    (fun (node, slot) -> Realize.Buffers.seed t.binding node input_bufs.(slot))
    t.input_nodes

let validate_inputs t (input_bufs : B.t array) =
  let n = Array.length t.expected_input_info in
  if Array.length input_bufs <> n then
    raise
      (Jit_error
         (Printf.sprintf "input count mismatch: expected %d, got %d" n
            (Array.length input_bufs)));
  Array.iteri
    (fun i info ->
      let b = input_bufs.(i) in
      if
        B.size b <> info.ii_size
        || not (Dtype.equal (B.dtype b) info.ii_dtype)
        || B.device b <> info.ii_device
      then
        raise
          (Jit_error
             (Printf.sprintf "input %d mismatch: expected (%d, %s, %s)" i
                info.ii_size
                (Dtype.to_string info.ii_dtype)
                info.ii_device)))
    t.expected_input_info

(* Replay the captured LINEAR with fresh input buffers. *)
let exec_captured ?wait t (input_bufs : B.t array) var_vals =
  validate_inputs t input_bufs;
  if t.first_run then begin
    seed_from_resolver t input_bufs;
    t.first_run <- false
  end;
  seed_inputs t input_bufs;
  let wait = match wait with Some w -> w | None -> false in
  Realize.run_linear ~device:t.device ~to_program:t.to_program t.binding
    ~var_vals ~jit:true ~wait t.linear;
  t.ret

(* TinyJit *)

type 'a tiny_jit = {
  fxn : (B.t array -> (string * int) list -> 'a) option;
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

let call ?wait ?held_buffers:_ t (input_bufs : B.t array)
    (var_vals : (string * int) list) ~(buffers : U.t -> B.t option) =
  let ret =
    if jit_level = 0 || t.cnt = 0 then
      (* Warmup: execute eagerly. *)
      (Option.get t.fxn) input_bufs var_vals
    else if t.cnt = 1 then begin
      (* Capture: record the linears the function schedules. *)
      let fxn = Option.get t.fxn in
      if is_capturing () then raise (Jit_error "nested TinyJit is not supported");
      let linears = ref [] in
      capturing := Some linears;
      let ret =
        Fun.protect
          ~finally:(fun () -> capturing := None)
          (fun () -> fxn input_bufs var_vals)
      in
      let linears = List.rev !linears in
      if linears = [] then raise (Jit_error "didn't JIT anything!");
      if debug >= 1 then
        Printf.eprintf "JIT captured %d linears with %d inputs\n%!"
          (List.length linears) (Array.length input_bufs);
      let captured =
        {
          ret;
          linear =
            Realize.pm_compile ~device:t.device ~to_program:t.to_program
              (combine_linears linears);
          device = t.device;
          to_program = t.to_program;
          buffers;
          binding = Realize.Buffers.create ~device:t.device;
          expected_input_info = Array.map input_info_of_buffer input_bufs;
          input_nodes = [];
          first_run = true;
        }
      in
      t.captured <- Some captured;
      exec_captured ?wait captured input_bufs var_vals
    end
    else
      (* Exec: replay the captured schedule. *)
      exec_captured ?wait (Option.get t.captured) input_bufs var_vals
  in
  t.cnt <- t.cnt + 1;
  ret
