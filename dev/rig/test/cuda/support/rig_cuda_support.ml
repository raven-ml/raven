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

external context_arg : unit -> arg = "rig_cuda_test_context"
external context_fill : unit -> nativeint = "rig_cuda_test_context_fill"
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

let context () = { fn = context_fill (); arg = context_arg () }
let seen f = seen_arg f.arg

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
  (m, fun name -> (Option.get (Rig_cuda.entry m name)).code)

(* Launches. A launch's parameters hold its buffers' addresses: the caller
   names the buffers in its submission, whose slots the launch does not
   know. *)

let binary () =
  ( fixture ~dir:"../cuda/fixtures" "kernels.ptx",
    [ "empty"; "double_index"; "spin"; "step"; "fault" ] )

let launch_binary () = Some (fixture ~dir:"../cuda/fixtures" "launch.ptx")
let modules = Rig_gpu_support.loader (fun () -> fst (binary ()))
let launches = Rig_gpu_support.loader (fun () -> Option.get (launch_binary ()))

(* [kernel] of [image] on COMPUTE:0 over [groups] of [threads], with [params]
   bytes of parameters [store] stores. *)
let launching image kernel ~params ?(groups = 1) ?(threads = 1) store =
  let module Run = Rig.Submission.Run in
  let part =
    {
      Rig.Submission.queue = "COMPUTE:0";
      after = [||];
      work = Launch { image; kernel; params; refs = [||] };
    }
  in
  let block run b =
    Run.groups run b groups 1 1;
    Run.threads run b threads 1 1;
    store run b
  in
  { Rig_gpu_support.part; block }

let launch t kernel a b =
  launching (modules t.d) kernel ~params:16 (fun run k ->
      Rig.Submission.Run.int64 run k 0 a;
      Rig.Submission.Run.int64 run k 8 b)

(* Conformance *)

let second () = if Rig_cuda.count () < 2 then None else Some (Rig_cuda.open_ 1)

(* fixtures/launch.ptx's [copy dst src n] copies [n] 32-bit words, one a
   thread. *)
let copy_words t ~dst ~src =
  let module Run = Rig.Submission.Run in
  let n = Rig.Buffer.length src / 4 in
  let w =
    launching (launches t.d) "copy" ~params:20 ~groups:((n + 255) / 256)
      ~threads:256 (fun run k ->
        Run.int64 run k 0 (Rig.Buffer.address dst);
        Run.int64 run k 8 (Rig.Buffer.address src);
        Run.int32 run k 16 n)
  in
  (w, src)

(* The kernel [spin] holds until its flag, a zero word here, is not 0, or for
   its nanoseconds. *)
let spin t ~ns =
  let flag = Rig_gpu_support.arguments t.d (String.make 8 '\000') in
  (launch t "spin" (Rig.Buffer.address flag) ns, flag)
