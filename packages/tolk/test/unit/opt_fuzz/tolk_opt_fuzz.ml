(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Differential correctness of the beam-search action set.

   Beam search ranks candidate kernels by measured runtime alone; it never
   inspects their results. An action that changes what a kernel computes is
   therefore invisible to the search and silently corrupts every program the
   search picks it for. This module closes that hole: for each workload kernel
   it compiles the unoptimised kernel and every applicable sequence of actions
   from [Search.actions], runs them all on identical inputs, and requires the
   optimised outputs to match the baseline.

   The action set that survives [apply_opt] depends on the renderer — a target
   without local memory drops every LOCAL, GROUP and tensor-core action — so
   the device is a parameter and each backend gets its own executable.

   Reductions may reassociate under GROUP/UNROLL, so float comparison is
   relative with a tolerance far below the magnitude of a real miscompile. *)
open Tolk
open Tolk_uop
module U = Uop
module D = Dtype
module C = Const
module P = Postrange

(* Graph construction *)

let mk_shape dims =
  match List.map (fun s -> U.const (C.int D.index s)) dims with
  | [ d ] -> d
  | ds -> U.stack ds

let mk_param ~slot ?(dtype = D.float32) shape =
  let shape = if shape = [] then None else Some (mk_shape shape) in
  U.param ~slot ~dtype ?shape ~device:(Single "CPU") ()

let add a b = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b
let mul a b = U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b
let reshape src dims = U.reshape ~src ~shape:(mk_shape dims)
let broadcast src dims = U.broadcast_to ~src ~shape:(mk_shape dims)

(* [a @ b] as a broadcast-multiply reduced over the contraction axis, the
   form the frontend lowers matmul to. *)
let matmul ~m ~k ~n a b =
  let ae = broadcast (reshape a [ m; 1; k ]) [ m; n; k ] in
  let bt = U.permute ~src:b ~order:[ 1; 0 ] in
  let be = broadcast (reshape bt [ 1; n; k ]) [ m; n; k ] in
  U.reduce_axis ~src:(mul ae be) ~op:Ops.Add ~axes:[ 2 ]

let wrap_sink srcs = U.sink (List.map (fun src -> U.contiguous ~src ()) srcs)

type workload = { name : string; sink : U.t }

let elementwise =
  let a = mk_param ~slot:0 [ 64; 64 ] in
  let b = mk_param ~slot:1 [ 64; 64 ] in
  let c = mk_param ~slot:2 [ 64; 64 ] in
  { name = "elementwise"; sink = wrap_sink [ add a (mul b c) ] }

let row_reduce =
  let x = mk_param ~slot:0 [ 64; 64 ] in
  {
    name = "row_reduce";
    sink = wrap_sink [ U.reduce_axis ~src:x ~op:Ops.Add ~axes:[ 1 ] ];
  }

let matmul_kernel =
  let a = mk_param ~slot:0 [ 32; 32 ] in
  let b = mk_param ~slot:1 [ 32; 32 ] in
  { name = "matmul"; sink = wrap_sink [ matmul ~m:32 ~k:32 ~n:32 a b ] }

(* Full reduce of a matmul to a scalar: the shape of the per-step loss in a
   recurrent net, where the miscompile was first observed. *)
let matmul_full_reduce =
  let a = mk_param ~slot:0 [ 32; 32 ] in
  let b = mk_param ~slot:1 [ 32; 32 ] in
  let h = matmul ~m:32 ~k:32 ~n:32 a b in
  {
    name = "matmul_full_reduce";
    sink = wrap_sink [ U.reduce_axis ~src:(mul h h) ~op:Ops.Add ~axes:[ 0; 1 ] ];
  }

(* Two chained matmuls summed, then reduced: one recurrent step. *)
let rnn_step =
  let d = 32 in
  let w_in = mk_param ~slot:0 [ d; d ] in
  let w_rec = mk_param ~slot:1 [ d; d ] in
  let h0 = mk_param ~slot:2 [ d; d ] in
  let x = mk_param ~slot:3 [ d; d ] in
  let h =
    add (matmul ~m:d ~k:d ~n:d x w_in) (matmul ~m:d ~k:d ~n:d h0 w_rec)
  in
  {
    name = "rnn_step";
    sink = wrap_sink [ U.reduce_axis ~src:(mul h h) ~op:Ops.Add ~axes:[ 0; 1 ] ];
  }

(* Two stores sharing one iteration space: the column axis is an output axis
   for the scaled copy and a reduce axis for the row sum. Guards that classify
   an axis by scanning every store at once — [globalizable_rngs] intersects
   over all of them — can call such an axis parallelisable on the strength of
   one store while the other reduces along it. *)
let multi_store =
  let x = mk_param ~slot:0 [ 64; 64 ] in
  {
    name = "multi_store";
    sink =
      wrap_sink
        [
          mul x (U.const (C.float D.float32 2.0));
          U.reduce_axis ~src:x ~op:Ops.Add ~axes:[ 1 ];
        ];
  }

let workloads =
  [
    elementwise;
    row_reduce;
    matmul_kernel;
    matmul_full_reduce;
    rnn_step;
    multi_store;
  ]

(* Kernel extraction *)

let kernels_of w =
  List.filter_map
    (fun node ->
      match (U.op node, U.as_call node) with
      | Ops.Call, Some { body; _ } when U.as_kernel_info body <> None ->
          Some body
      | _ -> None)
    (U.toposort (Rangeify.get_kernel_graph w.sink))

(* [with_opts ki_opts ast] is [ast] with its kernel info pinned to apply
   exactly [ki_opts], bypassing both beam search and the hand-coded
   heuristics. *)
let with_opts opts ast =
  match U.as_kernel_info ast with
  | None -> invalid_arg "with_opts: kernel body carries no kernel info"
  | Some ki ->
      U.sink
        ~kernel_info:{ ki with opts_to_apply = Some opts; applied_opts = [] }
        (Array.to_list (U.src ast))

(* Execution *)

(* Deterministic, well-conditioned inputs: values in [-1, 1] with no repeated
   pattern short enough to hide an indexing mistake behind a coincidence. *)
let fill_bytes ~size ~seed =
  let bytes = Bytes.create (size * 4) in
  for i = 0 to size - 1 do
    let v = sin (float_of_int ((i * 7) + (seed * 131)) *. 0.37) in
    Bytes.set_int32_le bytes (i * 4) (Int32.bits_of_float v)
  done;
  bytes

let read_f32 buf =
  let bytes = Device.Buffer.as_bytes buf in
  Array.init
    (Bytes.length bytes / 4)
    (fun i -> Int32.float_of_bits (Bytes.get_int32_le bytes (i * 4)))

(* Allocate one buffer per PARAM slot of [ast], seeded so that every run of
   every candidate sees byte-identical inputs. *)
let make_buffers dev ast =
  List.mapi
    (fun seed p ->
      let slot =
        match U.as_param p with
        | Some { param; _ } -> param.slot
        | None -> invalid_arg "make_buffers: not a PARAM"
      in
      let size = List.fold_left ( * ) 1 (U.max_shape p) in
      let buf = Device.create_buffer ~size ~dtype:(U.dtype p) dev in
      Device.Buffer.ensure_allocated buf;
      (slot, buf, size, seed))
    (P.bufs_from_ast ast)

let seed_buffers bufs =
  List.iter
    (fun (_, buf, size, seed) ->
      Device.Buffer.copyin buf (fill_bytes ~size ~seed))
    bufs

(* Compile [ast] and run it once against freshly seeded [bufs]; returns every
   buffer's contents afterwards, so an action that scribbles outside its
   output is caught alongside one that computes the wrong value. *)
let run_kernel dev ast bufs =
  let lowered =
    Codegen.full_rewrite_to_sink (Device.renderer dev) ast
  in
  let program = Device.compile_program dev (Linearizer.linearize lowered) in
  seed_buffers bufs;
  let args =
    List.map
      (fun slot ->
        match List.find_opt (fun (s, _, _, _) -> s = slot) bufs with
        | Some (_, buf, _, _) -> buf
        | None ->
            invalid_arg (Printf.sprintf "run_kernel: no buffer for slot %d" slot))
      (Program_spec.globals program)
  in
  let runner = Realize.Compiled_runner.create ~device:dev program in
  ignore (Realize.Compiled_runner.call runner args [] ~wait:true ~timeout:None);
  Device.synchronize dev;
  List.map (fun (slot, buf, _, _) -> (slot, read_f32 buf)) bufs

(* Comparison *)

let rel_tol = 1e-4

let first_mismatch baseline actual =
  let rec scan = function
    | [] -> None
    | (slot, expect) :: rest -> (
        match List.assoc_opt slot actual with
        | None -> Some (Printf.sprintf "slot %d missing from result" slot)
        | Some got ->
            if Array.length got <> Array.length expect then
              Some
                (Printf.sprintf "slot %d: length %d, expected %d" slot
                   (Array.length got) (Array.length expect))
            else
              let bad = ref None in
              Array.iteri
                (fun i e ->
                  if !bad = None then
                    let a = got.(i) in
                    let scale = Float.max 1.0 (Float.abs e) in
                    let off =
                      (Float.is_nan e <> Float.is_nan a)
                      || Float.abs (e -. a) > rel_tol *. scale
                    in
                    if off then
                      bad :=
                        Some
                          (Printf.sprintf "slot %d[%d]: got %.8g, expected %.8g"
                             slot i a e))
                expect;
              match !bad with Some m -> Some m | None -> scan rest)
  in
  scan baseline

let show_opts opts = String.concat "," (List.map U.Opt.to_string opts)

(* [Inapplicable] is an opt whose precondition fails. [Rejected] is one that
   applied but then failed to compile or run: [Search.try_compile] and the
   timing loop both swallow those, so beam drops the candidate rather than
   selecting it. Only [Diverged] — an optimised kernel that compiles, runs,
   and returns different values — is something beam can pick silently. *)
type outcome =
  | Inapplicable
  | Rejected of string
  | Matched
  | Diverged of string

let check_opts dev ~baseline ~bufs ~ast opts =
  match run_kernel dev (with_opts opts ast) bufs with
  | result -> (
      match first_mismatch baseline result with
      | None -> Matched
      | Some detail -> Diverged detail)
  | exception P.Opt_error _ -> Inapplicable
  | exception ((Out_of_memory | Stack_overflow) as exn) -> raise exn
  | exception exn -> Rejected (Printexc.to_string exn)

(* Beam search explores sequences, not single actions: round [n] applies one
   action to each scheduler kept from round [n-1]. Sweeping the applicable
   actions to [depth] therefore covers everything beam can reach in its first
   [depth] rounds, up to the beam width's pruning. Cost is exponential in
   [depth], so 2 is the default and deeper sweeps are opt-in. *)
let depth () =
  match int_of_string_opt (Option.value ~default:"" (Sys.getenv_opt "OPT_FUZZ_DEPTH")) with
  | Some d when d >= 1 -> d
  | _ -> 2

type result = {
  sequences : int;
  miscompiled : (string * string) list;
  rejected : (string * string) list;
}

let check_kernel dev ~index ast acc =
  let bufs = make_buffers dev ast in
  let baseline = run_kernel dev (with_opts [] ast) bufs in
  let acc = ref acc in
  let note field opts detail =
    let entry = (Printf.sprintf "kernel %d, %s" index (show_opts opts), detail) in
    acc := field !acc entry
  in
  let rec explore prefix remaining =
    List.iter
      (fun opt ->
        let opts = prefix @ [ opt ] in
        match check_opts dev ~baseline ~bufs ~ast opts with
        | Inapplicable -> ()
        | Rejected detail ->
            note (fun a e -> { a with rejected = e :: a.rejected }) opts detail
        | Matched ->
            acc := { !acc with sequences = !acc.sequences + 1 };
            if remaining > 1 then explore opts (remaining - 1)
        | Diverged detail ->
            acc := { !acc with sequences = !acc.sequences + 1 };
            note
              (fun a e -> { a with miscompiled = e :: a.miscompiled })
              opts detail)
      Search.actions
  in
  explore [] (depth ());
  !acc

let name w = w.name

let check dev w =
  let empty = { sequences = 0; miscompiled = []; rejected = [] } in
  let r =
    List.fold_left
      (fun acc (index, ast) -> check_kernel dev ~index ast acc)
      empty
      (List.mapi (fun i ast -> (i, ast)) (kernels_of w))
  in
  {
    r with
    miscompiled = List.rev r.miscompiled;
    rejected = List.rev r.rejected;
  }
