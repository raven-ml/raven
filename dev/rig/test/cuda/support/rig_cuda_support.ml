(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* CUDA, as the capability finds it *)

external bind_symbols : nativeint array -> unit = "rig_cuda_test_bind"
external current : unit -> nativeint = "rig_cuda_test_current"
external locked : int -> bool = "rig_cuda_test_locked"
external attribute : int -> int = "rig_cuda_test_attribute"
external register : int -> int -> unit = "rig_cuda_test_register"
external unregister : int -> unit = "rig_cuda_test_unregister"
external free_memory : unit -> int = "rig_cuda_test_free_memory"
external loaded_c : nativeint -> int * int = "rig_cuda_test_loaded"

let functions_loaded f = loaded_c (Nativeint.of_int f)

(* The GPU *)

let bind g =
  let symbol = (Rig_cuda.capability g).symbol in
  bind_symbols
    (Array.map
       (fun n -> Option.get (symbol n))
       [|
         "cuLaunchKernel";
         "cuCtxGetCurrent";
         "cuDevicePrimaryCtxRetain";
         "cuCtxPushCurrent_v2";
         "cuCtxPopCurrent_v2";
         "cuMemHostGetDevicePointer_v2";
         "cuDeviceGetAttribute";
         "cuMemcpyAsync";
         "cuMemcpyDtoH_v2";
         "cuMemcpyHtoD_v2";
         "cuMemHostRegister_v2";
         "cuMemHostUnregister";
         "cuMemGetInfo_v2";
         "cuGraphExecKernelNodeSetParams_v2";
         "cuGraphLaunch";
         "cuFuncIsLoaded";
         "cuFuncGetModule";
         "cuModuleGetFunctionCount";
         "cuModuleEnumerateFunctions";
         "cuCtxSynchronize";
       |])

module D = Rig_cuda

include Rig_gpu_support.Make (struct
  module D = Rig_cuda

  let class_ = "CUDA"
  let present () = Sys.file_exists "/dev/nvidiactl"
  let open_ () =
    let g = Rig_cuda.open_ 0 in
    Result.iter bind g;
    g
end)

(* Memory *)

external read_gpu : nativeint -> int -> string = "rig_cuda_test_read_gpu"
external write_gpu : nativeint -> string -> unit = "rig_cuda_test_write_gpu"
external stall_c : int -> int -> int -> unit = "rig_cuda_test_stall"

let stall spin ~flag ~ns = stall_c spin flag ns

(* Fills *)

type arg =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external failing_arg : int -> arg = "rig_cuda_test_failing"
external failing_fill : unit -> nativeint = "rig_cuda_test_failing_fill"

external launch_arg : int -> int -> int -> int -> int -> int -> int -> arg
  = "rig_cuda_test_launch_byte" "rig_cuda_test_launch"

external launch_fill : unit -> nativeint = "rig_cuda_test_launch_fill"
external seen_arg : arg -> nativeint = "rig_cuda_test_seen"

type fill = { fn : nativeint; arg : arg }

let part ~queue ?(after = [||]) f =
  let arg = Rig.Buffer.of_bigarray f.arg in
  {
    Rig.Submission.queue;
    after;
    work = Fill { fill = f.fn; arg; ring_units = 0; segment_bytes = 0 };
  }

let copy ~queue ?(after = [||]) ~dst src =
  { Rig.Submission.queue; after; work = Copy { src; dst } }

let failing code = { fn = failing_fill (); arg = failing_arg code }

let launch ?(count = 1) ?(shared = 0) f ~grid ~block a b =
  { fn = launch_fill (); arg = launch_arg f grid block shared count a b }

let seen f = seen_arg f.arg

external delayed_arg : int -> int -> int -> int -> int -> int -> arg
  = "rig_cuda_test_delayed_byte" "rig_cuda_test_delayed"

external delayed_fill : unit -> nativeint = "rig_cuda_test_delayed_fill"

let delayed ~spin ~flag ~ns ~dst ~src n =
  { fn = delayed_fill (); arg = delayed_arg spin flag ns dst src n }

let kernel ?(grid = 1) ?(block = 1) func a b =
  let args = Bytes.create 16 in
  Bytes.set_int64_le args 0 (Int64.of_int a);
  Bytes.set_int64_le args 8 (Int64.of_int b);
  {
    Rig_cuda_abi.func;
    grid = (grid, 1, 1);
    block = (block, 1, 1);
    shared = 0;
    args = Bytes.to_string args;
  }

external graph_arg :
  nativeint -> nativeint array -> int array -> int array -> string array -> arg
  = "rig_cuda_test_graph"

external graph_fill : unit -> nativeint = "rig_cuda_test_graph_fill"

let graph_launch (g : Rig_cuda_abi.graph) updates =
  let node (i, _) = g.nodes.(i) in
  let func (_, (k : Rig_cuda_abi.kernel)) = k.func in
  let sizes (_, (k : Rig_cuda_abi.kernel)) =
    let gx, gy, gz = k.grid and bx, by, bz = k.block in
    [| gx; gy; gz; bx; by; bz; k.shared |]
  in
  let args (_, (k : Rig_cuda_abi.kernel)) = k.args in
  let sizes = Array.concat (Array.to_list (Array.map sizes updates)) in
  let arg =
    graph_arg g.handle (Array.map node updates) (Array.map func updates) sizes
      (Array.map args updates)
  in
  { fn = graph_fill (); arg }

(* Kernels *)

let fixture ?(dir = "fixtures") f =
  In_channel.with_open_bin (Filename.concat dir f) In_channel.input_all

let loaded : (Rig_cuda.region, Rig_cuda.image) Rig_edge.code -> _ = function
  | Loaded m -> m
  | Place _ -> failwith "a CUDA device asked to place its code"

let kernels ?dir g =
  let m =
    loaded (Result.get_ok (Rig_cuda.image g (fixture ?dir "kernels.ptx")))
  in
  (m, fun name -> Option.get (Rig_cuda.entry m name))

(* Conformance *)

let binary () =
  ( fixture ~dir:"../cuda/fixtures" "kernels.ptx",
    [ "empty"; "double_index"; "spin"; "step"; "fault" ] )

let second () = if Rig_cuda.count () < 2 then None else Some (Rig_cuda.open_ 1)
let modules = Rig_gpu_support.loader (fun () -> fst (binary ()))
let func t f = Option.get (Rig.Image.entry (modules t.d) f)

(* The kernel [spin] holds until its flag, a zero word here, is not 0, or for
   its nanoseconds. *)
let zero t = Rig_gpu_support.arguments t.d (String.make 8 '\000')

let copy_words t ~dst ~src =
  let flag = zero t in
  let f =
    delayed ~spin:(func t "spin") ~flag:(Rig.Buffer.address flag) ~ns:0
      ~dst:(Rig.Buffer.address dst) ~src:(Rig.Buffer.address src)
      (Rig.Buffer.length src)
  in
  (part ~queue:"COMPUTE:0" f, flag)

let spin t ~ns =
  let flag = zero t in
  let f =
    launch (func t "spin") ~grid:1 ~block:1 (Rig.Buffer.address flag) ns
  in
  (part ~queue:"COMPUTE:0" f, flag)
