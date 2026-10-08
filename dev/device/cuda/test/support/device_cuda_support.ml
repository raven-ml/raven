(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let strf = Printf.sprintf

(* CUDA, as the capability finds it *)

external bind_symbols : nativeint array -> unit = "device_cuda_test_bind"
external current : unit -> nativeint = "device_cuda_test_current"
external locked : int -> bool = "device_cuda_test_locked"
external attribute : int -> int = "device_cuda_test_attribute"
external register : int -> int -> unit = "device_cuda_test_register"
external unregister : int -> unit = "device_cuda_test_unregister"

(* The machine's GPU lock *)

external lock : string -> string -> int = "device_cuda_test_lock"

let gpu_lock = "/tmp/raven-device-gpu.lock"

(* The longest wait for the lock, in seconds: the machine's suites, from every
   checkout and user, take it in turn. *)
let gpu_wait = 300

let holder () =
  match In_channel.with_open_bin gpu_lock In_channel.input_all with
  | note -> String.trim note
  | exception Sys_error _ -> "a process that left no note"

(* [lock] naps 100 ms each time it is refused. *)
let rec take refused =
  match lock gpu_lock Sys.executable_name with
  | 0 -> ()
  | -1 when refused < gpu_wait * 10 -> take (refused + 1)
  | -1 ->
      failwith
        (strf "%s: still held after %d s, by %s" gpu_lock gpu_wait (holder ()))
  | errno -> failwith (strf "%s: errno %d" gpu_lock errno)

let hold_gpu () = if Device_cuda.count () > 0 then take 0

(* The GPU *)

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
         "cuMemHostRegister_v2";
         "cuMemHostUnregister";
       |])

(* The device gpu opened, until a test stops it: one a failed test left open is
   stopped by the next gpu. *)
let opened = ref None

let stop g =
  (match !opened with Some o when o == g -> opened := None | _ -> ());
  Device_cuda.stop g

let gpu () =
  if Device_cuda.count () = 0 then skip ~reason:"CUDA sees no GPU" ();
  hold_gpu ();
  Option.iter stop !opened;
  let g = Result.get_ok (Device_cuda.open_ 0) in
  opened := Some g;
  bind g;
  g

let with_gpu f =
  let g = gpu () in
  let stop_left () =
    match !opened with Some o when o == g -> stop g | _ -> ()
  in
  Fun.protect ~finally:stop_left (fun () -> f g)

(* Checks *)

let answer =
  Testable.make
    ~pp:(fun ppf -> function
      | `Ok -> Format.pp_print_string ppf "`Ok"
      | `Failed why -> Format.fprintf ppf "`Failed %S" why)
    ~equal:( = )

let still ?msg w x f ~ms =
  let t0 = Sys.time () in
  while Sys.time () -. t0 < Float.of_int ms /. 1000. do
    equal ?msg w x (f ())
  done

(* Host memory *)

external page_size : unit -> int = "device_cuda_test_page_size"
external pages : int -> bool -> int = "device_cuda_test_pages"
external free_pages : int -> int -> unit = "device_cuda_test_free_pages"
external get64 : int -> int = "device_cuda_test_get64"
external set64 : int -> int -> unit = "device_cuda_test_set64"
external read : int -> int -> string = "device_cuda_test_read"
external write : int -> string -> unit = "device_cuda_test_write"
external read_gpu : nativeint -> int -> string = "device_cuda_test_read_gpu"
external write_gpu : nativeint -> string -> unit = "device_cuda_test_write_gpu"

let page = page_size ()

let get32 a i =
  Int32.to_int (String.get_int32_le (read (a + (4 * i)) 4) 0) land 0xffff_ffff

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

let loaded = function
  | `Loaded m -> m
  | `Place _ -> failwith "a CUDA device asked to place its code"

let kernels ?dir g =
  let m =
    loaded (Result.get_ok (Device_cuda.image g (fixture ?dir "kernels.ptx")))
  in
  (m, fun name -> Option.get (Device_cuda.entry m name))
