(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* CUDA, as the capability finds it *)

external bind_symbols : nativeint array -> unit = "device_cuda_test_bind"
external lock : string -> bool = "device_cuda_test_lock"
external current : unit -> nativeint = "device_cuda_test_current"
external locked : nativeint -> bool = "device_cuda_test_locked"
external attribute : int -> int = "device_cuda_test_attribute"

(* The GPU *)

let gpu_lock = "DEVICE_CUDA_TEST_GPU_LOCK"

(* The lock is taken once and kept: [Some true] once taken. *)
let held = ref None

let take_lock () =
  match !held with
  | Some taken -> taken
  | None ->
      let taken =
        match Sys.getenv_opt gpu_lock with
        | None | Some "" -> skip ~reason:(gpu_lock ^ " names no lock file") ()
        | Some file -> lock file
      in
      held := Some taken;
      taken

let bind g =
  let { Device_cuda_abi.symbol } = Device_cuda.capability g in
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
       |])

let gpu () =
  if Device_cuda.count () = 0 then skip ~reason:"CUDA sees no GPU" ();
  if not (take_lock ()) then
    skip ~reason:"another process holds the GPU lock" ();
  let g = Result.get_ok (Device_cuda.open_ 0) in
  bind g;
  g

let with_gpu f =
  let g = gpu () in
  Fun.protect ~finally:(fun () -> ignore (Device_cuda.stop g)) (fun () -> f g)

(* Host memory *)

external page_size : unit -> int = "device_cuda_test_page_size"
external pages : int -> bool -> nativeint = "device_cuda_test_pages"
external free_pages : nativeint -> int -> unit = "device_cuda_test_free_pages"
external get64 : nativeint -> int = "device_cuda_test_get64"
external set64 : nativeint -> int -> unit = "device_cuda_test_set64"
external read : nativeint -> int -> string = "device_cuda_test_read"
external write : nativeint -> string -> unit = "device_cuda_test_write"
external read_gpu : nativeint -> int -> string = "device_cuda_test_read_gpu"
external write_gpu : nativeint -> string -> unit = "device_cuda_test_write_gpu"

let page = page_size ()

let get32 a i =
  Int32.to_int
    (String.get_int32_le
       (read (Nativeint.add a (Nativeint.of_int (4 * i))) 4)
       0)
  land 0xffff_ffff

let pages ?(read_only = false) n = pages n read_only

let wait g v =
  let word = Option.get (Device_cuda.host (Device_cuda.word g)) in
  let t0 = Sys.time () in
  while get64 word < v do
    if Sys.time () -. t0 > 10. then
      failf "the word stayed at %d below %d for 10 s" (get64 word) v;
    Domain.cpu_relax ()
  done

(* Fills *)

type arg

external arg_address : arg -> nativeint = "device_cuda_test_arg_address"
external failing_arg : int -> arg = "device_cuda_test_failing"
external failing_fill : unit -> nativeint = "device_cuda_test_failing_fill"

external launch_arg : int -> int -> int -> int -> int -> int -> arg
  = "device_cuda_test_launch_byte" "device_cuda_test_launch"

external launch_fill : unit -> nativeint = "device_cuda_test_launch_fill"
external seen_arg : arg -> nativeint = "device_cuda_test_seen"

type fill = { fn : nativeint; arg : arg }

let part g ~queue ?after f =
  Device_cuda.part g ~queue ?after (`Fill (f.fn, arg_address f.arg, 0, 0))

let failing code = { fn = failing_fill (); arg = failing_arg code }

let launch ?(count = 1) f ~grid ~block a b =
  { fn = launch_fill (); arg = launch_arg f grid block count a b }

let seen f = seen_arg f.arg

external delayed_arg : int -> int -> int -> int -> int -> int -> arg
  = "device_cuda_test_delayed_byte" "device_cuda_test_delayed"

external delayed_fill : unit -> nativeint = "device_cuda_test_delayed_fill"

let delayed ~spin ~flag ~ns ~dst ~src n =
  { fn = delayed_fill (); arg = delayed_arg spin flag ns dst src n }

external room : nativeint -> int -> bool -> int -> int -> int array -> int
  = "device_cuda_test_room_byte" "device_cuda_test_room"

let room g ~queue ~words ~units ~bytes ~after =
  room (Device_cuda.self g) queue words units bytes after

(* Kernels *)

let fixture ?(dir = "fixtures") f =
  In_channel.with_open_bin (Filename.concat dir f) In_channel.input_all

let kernels ?dir g =
  let m, _ = Result.get_ok (Device_cuda.image g (fixture ?dir "kernels.ptx")) in
  (m, fun name -> Option.get (Device_cuda.entry m name))
