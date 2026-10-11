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
    let g = (Rig_cuda.gpu 0).make () in
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
external spinning_arg : int -> int -> int -> arg = "rig_cuda_test_spinning"
external spinning_fill : unit -> nativeint = "rig_cuda_test_spinning_fill"
external data : arg -> int = "rig_cuda_test_data" [@@noalloc]

type fill = {
  fn : nativeint;
  arg : arg;
  state : Rig.access option;
  keep : Rig_gpu_support.kept;
}

let fn f = f.fn

let parameters f =
  match f.state with
  | None -> String.init (Bigarray.Array1.dim f.arg) (fun i -> f.arg.{i})
  | Some _ ->
      let bytes = Bytes.create 8 in
      Bytes.set_int64_ne bytes 0 (Int64.of_int (data f.arg));
      Bytes.to_string bytes

let part d f =
  let fixed =
    match f.state with
    | None -> []
    | Some access ->
        let arg = Rig.Buffer.of_bigarray f.arg in
        [ (Option.get (Rig.Buffer.borrow d arg), access) ]
  in
  let bytes = parameters f in
  let part = Rig.Submission.fill f.fn ~params:(String.length bytes) in
  let init s run i =
    Rig_gpu_support.params run (Rig.Submission.block s i) bytes
  in
  {
    Rig_gpu_support.part = (fun _ -> part);
    buffers = [||];
    fixed;
    keep = f.keep;
    init;
  }

let failing code =
  { fn = failing_fill (); arg = failing_arg code; state = None; keep = Kept () }

let context () =
  {
    fn = context_fill ();
    arg = context_arg ();
    state = Some Read_write;
    keep = Kept ();
  }

let seen f = seen_arg f.arg

let spinning spin ~flag ~ns =
  {
    fn = spinning_fill ();
    arg = spinning_arg spin flag ns;
    state = None;
    keep = Kept ();
  }

(* The C entry *)

external submit_c :
  nativeint -> int -> (int * nativeint * string) array -> string option
  = "rig_cuda_test_submit"

let submit_edge g ~v parts =
  let part (queue, f) = (queue, f.fn, parameters f) in
  submit_c (Rig_cuda.facts g).edge v (Array.of_list (List.map part parts))

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
  { fn = graph_fill (); arg; state = Some Read; keep = Kept g }

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
  let code name =
    match Rig_cuda.entry m name with
    | Entry e -> e.code
    | Absent | No_memory _ -> invalid_arg ("no kernel " ^ name)
  in
  (m, code)

(* Launches *)

let binary () =
  ( fixture ~dir:"../cuda/fixtures" "kernels.ptx",
    [ "empty"; "double_index"; "spin"; "step"; "fault" ] )

let launch_binary () = Some (fixture ~dir:"../cuda/fixtures" "launch.ptx")
let modules = Rig_gpu_support.loader (fun () -> fst (binary ()))
let launches = Rig_gpu_support.loader (fun () -> Option.get (launch_binary ()))

let launching = Rig_gpu_support.launching

let launch t kernel a b =
  launching (modules t.d) kernel ~params:16 [||] (fun run k ->
      Rig.Submission.Run.int64 run k 0 a;
      Rig.Submission.Run.int64 run k 8 b)

(* Conformance *)

let second () =
  if Rig_cuda.count () < 2 then None else Some ((Rig_cuda.gpu 1).make ())

(* fixtures/launch.ptx's [copy dst src n] copies [n] 32-bit words, one a
   thread. *)
let copy_words t ~dst ~src =
  let module Run = Rig.Submission.Run in
  let n = Rig.Buffer.length src / 4 in
  let w =
    launching (launches t.d) "copy" ~params:20
      ~groups:((n + 255) / 256)
      ~threads:256
      [| (dst, Rig.Read_write, 0); (src, Rig.Read, 8) |]
      (fun run k -> Run.int32 run k 16 n)
  in
  w

(* The kernel [spin] holds until its flag, a zero word here, is not 0, or for
   its nanoseconds. *)
let spin t ~ns =
  let flag = Rig_gpu_support.arguments t.d (String.make 8 '\000') in
  launching (modules t.d) "spin" ~params:16
    [| (flag, Rig.Read, 0) |]
    (fun run k -> Rig.Submission.Run.int64 run k 8 ns)
