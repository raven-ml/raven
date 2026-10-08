(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs

let strf = Printf.sprintf
let ( let* ) = Result.bind

external open_raw : string -> int = "caml_rig_nv_nvidia_open"
external close : int -> unit = "caml_rig_nv_nvidia_close"

external ioctl : int -> int -> Rig_nv.params -> int
  = "caml_rig_nv_nvidia_ioctl"

external map_raw : int -> int -> int -> int = "caml_rig_nv_nvidia_map"
external reserve_raw : int -> int -> int = "caml_rig_nv_nvidia_reserve"
external release_raw : int -> int -> int = "caml_rig_nv_nvidia_release"
external strerror : int -> string = "caml_rig_nv_nvidia_strerror"
external address : Rig_nv.params -> int = "caml_rig_nv_nvidia_address"

(* A stub's result: non-negative, or errno negated. *)
let result what r =
  if r < 0 then Error (strf "%s: %s" what (strerror (-r))) else Ok r

(* Parameters *)

external get16 : Rig_nv.params -> int -> int = "%caml_bigstring_get16"
external get32 : Rig_nv.params -> int -> int32 = "%caml_bigstring_get32"
external get64 : Rig_nv.params -> int -> int64 = "%caml_bigstring_get64"

external set16 : Rig_nv.params -> int -> int -> unit
  = "%caml_bigstring_set16"

external set32 : Rig_nv.params -> int -> int32 -> unit
  = "%caml_bigstring_set32"

external set64 : Rig_nv.params -> int -> int64 -> unit
  = "%caml_bigstring_set64"

let params n =
  let p = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill p '\000';
  p

let get p (at, n) =
  match n with
  | 1 -> Char.code (Bigarray.Array1.get p at)
  | 2 -> get16 p at
  | 4 -> Int32.to_int (get32 p at) land 0xffff_ffff
  | _ -> Int64.to_int (get64 p at)

let set p (at, n) v =
  match n with
  | 1 -> Bigarray.Array1.set p at (Char.unsafe_chr (v land 0xff))
  | 2 -> set16 p at (v land 0xffff)
  | 4 -> set32 p at (Int32.of_int v)
  | _ -> set64 p at (Int64.of_int v)

(* Files and mappings *)

let open_file path = result path (open_raw path)

let map fd at n =
  Result.map ignore (result "mapping memory for the GPU" (map_raw fd at n))

let reserve at n =
  Result.map ignore (result "reserving the GPU's addresses" (reserve_raw at n))

(* Returning addresses to the reservation cannot fail but for a bad argument,
   which is this library's: its error is dropped. *)
let release at n = ignore (release_raw at n : int)

(* The RM *)

(* An escape ioctl: read and write, of the parameters' size, in the driver's
   magic. *)
let escape fd nr p what =
  let request =
    (3 lsl 30)
    lor ((Bigarray.Array1.dim p land 0x1fff) lsl 16)
    lor (D.nv_ioctl_magic lsl 8) lor nr
  in
  Result.map ignore (result what (ioctl fd request p))

let register fd ~ctl =
  let p = params D.Register_fd.sizeof in
  set p D.Register_fd.ctl_fd ctl;
  escape fd D.nv_esc_register_fd p "registering a GPU file"

type t = {
  ctl : int;
  uvm : int;
  root : int;
  release : (module D.RELEASE);
  number : int;
  low : Va.t;
  main : Va.t;
}

let status_name (module R : D.RELEASE) s =
  match List.assoc_opt s R.statuses with Some n -> n | None -> strf "0x%x" s

let check c what s =
  if s = D.nv_ok then Ok ()
  else Error (strf "%s: %s" what (status_name c.release s))

(* NV_ESC_RM_ALLOC under [root]: the status and the new handle. *)
let alloc_raw ctl ~root ~parent cls p =
  let module A = D.Nvos21 in
  let a = params A.sizeof in
  set a A.h_root root;
  set a A.h_object_parent parent;
  set a A.h_class cls;
  Option.iter (fun p -> set a A.p_alloc_parms (address p)) p;
  let* () = escape ctl D.nv_esc_rm_alloc a "allocating a GPU object" in
  ignore (Sys.opaque_identity p);
  Ok (get a A.status, get a A.h_object_new)

let alloc c ~parent cls p = alloc_raw c.ctl ~root:c.root ~parent cls p

let control_raw ctl ~root obj cmd p =
  let module C = D.Nvos54 in
  let a = params C.sizeof in
  set a C.h_client root;
  set a C.h_object obj;
  set a C.cmd cmd;
  Option.iter
    (fun p ->
      set a C.params_size (Bigarray.Array1.dim p);
      set a C.params (address p))
    p;
  let* () = escape ctl D.nv_esc_rm_control a "controlling a GPU object" in
  ignore (Sys.opaque_identity p);
  Ok (get a C.status)

let rm c =
  let alloc ~parent cls p =
    let* s, h = alloc c ~parent cls p in
    let* () = check c (strf "allocating class 0x%x" cls) s in
    Ok h
  in
  let control obj cmd p =
    let* s = control_raw c.ctl ~root:c.root obj cmd p in
    check c (strf "command 0x%x" cmd) s
  in
  let free ~parent obj =
    let module F = D.Nvos00 in
    let a = params F.sizeof in
    set a F.h_root c.root;
    set a F.h_object_parent parent;
    set a F.h_object_old obj;
    let* () = escape c.ctl D.nv_esc_rm_free a "freeing a GPU object" in
    check c "freeing a GPU object" (get a F.status)
  in
  { Rig_nv.release = c.number; client = c.root; alloc; control; free }

(* Unified memory *)

let uvm c cmd p status what =
  let* _ = result what (ioctl c.uvm cmd p) in
  Ok (get p status)

let uvm_call c cmd p status what =
  let* s = uvm c cmd p status what in
  check c what s

(* The client *)

(* The GPU addresses the process allocates, reserved in the process so that
   nothing else maps there: memory the host maps too from 384 GiB, the rest from
   448 GiB, all below 2^40, the widest address of a channel's segments. They
   start above 258 GiB, where AddressSanitizer on x86_64 maps the shadow of its
   shadow gap when the gap is unprotected ([protect_shadow_gap=0]). *)
let low_base = 0x60_0000_0000
let main_base = 0x70_0000_0000
let top = 1 lsl 40
let ctl_path = "/dev/nvidiactl"
let uvm_path = "/dev/nvidia-uvm"

(* The release branch of a driver version, such as 615 of "615.71.09". *)
let branch version =
  match String.index_opt version '.' with
  | Some i -> int_of_string_opt (String.sub version 0 i)
  | None -> int_of_string_opt version

let driver_version ctl root =
  let module B = D.Build_version in
  let v = params B.sizeof in
  let* s =
    control_raw ctl ~root root D.nv0000_ctrl_cmd_system_get_build_version_v2
      (Some v)
  in
  if s <> D.nv_ok then
    Error (strf "reading the driver's version: status 0x%x" s)
  else
    let at, _, n = B.driver_version_buffer in
    let b = String.init n (fun i -> Bigarray.Array1.get v (at + i)) in
    Ok
      (match String.index_opt b '\000' with
      | Some i -> String.sub b 0 i
      | None -> b)

let client_lock = Mutex.create ()
let opened = ref None

let make_client () =
  let undo = ref [] in
  let taken f = undo := f :: !undo in
  let opened_client =
    let* ctl = open_file ctl_path in
    taken (fun () -> close ctl);
    let* s, root = alloc_raw ctl ~root:0 ~parent:0 D.nv01_root_client None in
    let* () =
      if s = D.nv_ok then Ok ()
      else Error (strf "creating a client of NVIDIA's driver: status 0x%x" s)
    in
    let* version = driver_version ctl root in
    let* number, layouts =
      match
        Option.bind (branch version) (fun n ->
            Option.map (fun r -> (n, r)) (D.release n))
      with
      | Some x -> Ok x
      | None ->
          Error
            (strf
               "NVIDIA's kernel driver %s is not supported; the supported \
                releases are %s"
               version
               (String.concat ", " (List.map string_of_int D.releases)))
    in
    let* uvm_fd = open_file uvm_path in
    taken (fun () -> close uvm_fd);
    let* mm = open_file uvm_path in
    taken (fun () -> close mm);
    let* () = reserve low_base (main_base - low_base) in
    taken (fun () -> release low_base (main_base - low_base));
    let* () = reserve main_base (top - main_base) in
    taken (fun () -> release main_base (top - main_base));
    let c =
      {
        ctl;
        uvm = uvm_fd;
        root;
        release = layouts;
        number;
        low = Va.make ~base:low_base (main_base - low_base);
        main = Va.make ~base:main_base (top - main_base);
      }
    in
    let module I = D.Uvm_initialize in
    let* () =
      uvm_call c D.uvm_initialize (params I.sizeof) I.rm_status
        "initializing NVIDIA's unified memory"
    in
    (* The memory manager ties unified memory to the process's address space,
       through a second file kept open for the process. Its registration is made
       once per process; a second one, such as CUDA's in the same process, is
       refused, and the first serves: its answer is dropped. *)
    let module M = D.Uvm_mm_initialize in
    let m = params M.sizeof in
    set m M.uvm_fd uvm_fd;
    ignore (ioctl mm D.uvm_mm_initialize m : int);
    Ok c
  in
  (match opened_client with
  | Ok _ -> ()
  | Error _ -> List.iter (fun f -> f ()) !undo);
  opened_client

let client () =
  Mutex.protect client_lock @@ fun () ->
  match !opened with
  | Some c -> Ok c
  | None ->
      let* c = make_client () in
      opened := Some c;
      Ok c

(* The handles the process names objects by, past those the RM chooses. *)
let next = Atomic.make 0x1000
let handle () = Atomic.fetch_and_add next 1 + 1
