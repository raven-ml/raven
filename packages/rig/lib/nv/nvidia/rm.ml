(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs

let strf = Printf.sprintf
let ( let* ) = Result.bind

(* The stubs raise [Unix.Unix_error] where a system call fails. *)

external open_raw : string -> int = "caml_rig_nv_nvidia_open"
external close : int -> unit = "caml_rig_nv_nvidia_close"

external ioctl : int -> int -> Rig_nv.params -> unit
  = "caml_rig_nv_nvidia_ioctl"

external map_raw : int -> int -> int -> unit = "caml_rig_nv_nvidia_map"
external reserve_raw : int -> int -> unit = "caml_rig_nv_nvidia_reserve"
external unmap_raw : int -> int -> unit = "caml_rig_nv_nvidia_unmap"
external unreserve_raw : int -> int -> unit = "caml_rig_nv_nvidia_unreserve"
external reserve_any_raw : int -> int -> int = "caml_rig_nv_nvidia_reserve_any"
external address : Rig_nv.params -> int = "caml_rig_nv_nvidia_address"
external host_page : unit -> int = "caml_rig_nv_nvidia_page_size" [@@noalloc]

external reraise : exn -> 'a = "%reraise"

let failed what e = Error (strf "%s: %s" what (Unix.error_message e))

(* [guard what f] is [f ()], or [Error] saying that [what] failed and why. *)
let guard what f =
  match f () with
  | v -> Ok v
  | exception Unix.Unix_error (e, _, _) -> failed what e

let params = Rig_nv.params
let get = Rig_nv.get
let set = Rig_nv.set
let bits (lo, n) v = (v land ((1 lsl n) - 1)) lsl lo

(* Files and mappings *)

let open_file path = guard path (fun () -> open_raw path)

let open_spare path =
  match open_raw path with
  | fd -> Ok (Some fd)
  | exception Unix.Unix_error ((EMFILE | ENFILE), _, _) -> Ok None
  | exception Unix.Unix_error (e, _, _) -> failed path e

let map fd at n = guard "mapping memory for the GPU" (fun () -> map_raw fd at n)

(* Returning addresses to the reservation cannot fail but for a bad argument,
   which is this library's: its error is dropped. *)
let unmap at n = try unmap_raw at n with Unix.Unix_error _ -> ()

(* Gives back a reservation, so that a later [reserve] of it succeeds; its error
   is dropped for the same reason. *)
let unreserve at n = try unreserve_raw at n with Unix.Unix_error _ -> ()

let reserve_any ~align n =
  guard "reserving the GPU's addresses" @@ fun () ->
  match reserve_any_raw n align with 0 -> None | at -> Some at

(* The RM *)

(* Linux's _IOWR(type, nr, size) (asm-generic/ioctl.h): read and write in bits
   31:30, the size in 29:16, the type in 15:8, the number in 7:0. *)
let ioc_read_write = 3
let ioc_size_bits = 14

let iowr ~type_ nr size =
  if size >= 1 lsl ioc_size_bits then
    invalid_arg (strf "Rm.escape: %d bytes of parameters, past 16 KiB" size);
  (ioc_read_write lsl 30) lor (size lsl 16) lor (type_ lsl 8) lor nr

let control_request =
  iowr ~type_:D.nv_ioctl_magic D.nv_esc_rm_control D.Nvos54.sizeof

let why = function Rig_nv.Lacking why | Refused why -> why

(* An escape ioctl of the driver's magic, with the parameters [p]: its
   failure as the driver's refusal, [Lacking] where the kernel lacked the
   memory for it (ENOMEM). *)
let escape_rm fd nr p what =
  let request = iowr ~type_:D.nv_ioctl_magic nr (Bigarray.Array1.dim p) in
  match ioctl fd request p with
  | () -> Ok ()
  | exception Unix.Unix_error (e, _, _) ->
      let message = strf "%s: %s" what (Unix.error_message e) in
      if e = Unix.ENOMEM then Error (Rig_nv.Lacking message)
      else Error (Rig_nv.Refused message)

let escape fd nr p what = Result.map_error why (escape_rm fd nr p what)

let register fd ~ctl =
  let p = params D.Register_fd.sizeof in
  set p D.Register_fd.ctl_fd ctl;
  escape fd D.nv_esc_register_fd p "registering a GPU file"

type t = {
  ctl : int;
  uvm : int;
  root : int;
  layouts : (module D.RELEASE);
  release : int;
  low : Va.t;
}

let status_name (module R : D.RELEASE) s =
  match List.assoc_opt s R.statuses with Some n -> n | None -> strf "0x%x" s

let check c what s =
  if s = D.nv_ok then Ok ()
  else Error (strf "%s: %s" what (status_name c.layouts s))

(* NV_ESC_RM_ALLOC under [root]: the status and the new handle. *)
let alloc_raw ctl ~root ~parent cls p =
  let module A = D.Nvos21 in
  let a = params A.sizeof in
  set a A.h_root root;
  set a A.h_object_parent parent;
  set a A.h_class cls;
  Option.iter (fun p -> set a A.p_alloc_parms (address p)) p;
  let* () = escape_rm ctl D.nv_esc_rm_alloc a "allocating a GPU object" in
  ignore (Sys.opaque_identity p);
  Ok (get a A.status, get a A.h_object_new)

let alloc c ~parent cls p =
  Result.map_error why (alloc_raw c.ctl ~root:c.root ~parent cls p)

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
  let* () = escape_rm ctl D.nv_esc_rm_control a "controlling a GPU object" in
  ignore (Sys.opaque_identity p);
  Ok (get a C.status)

let control c ?(refs = []) obj cmd p =
  List.iter (fun (field, q) -> set p field (address q)) refs;
  let* s =
    Result.map_error why (control_raw c.ctl ~root:c.root obj cmd (Some p))
  in
  ignore (Sys.opaque_identity refs);
  check c (strf "command 0x%x" cmd) s

(* NV_ESC_RM_FREE of [obj] under [parent]: the status. *)
let free_raw c ~parent obj =
  let module F = D.Nvos00 in
  let a = params F.sizeof in
  set a F.h_root c.root;
  set a F.h_object_parent parent;
  set a F.h_object_old obj;
  let* () = escape_rm c.ctl D.nv_esc_rm_free a "freeing a GPU object" in
  Ok (get a F.status)

let free c ~parent obj =
  let* s = Result.map_error why (free_raw c ~parent obj) in
  check c "freeing a GPU object" s

(* [x] if the RM answered [s] = NV_OK to the call [what], its refusal
   otherwise, or the escape's that failed before the RM answered. *)
let answer c what x = function
  | Error _ as e -> e
  | Ok s when s = D.nv_ok -> Ok x
  | Ok s ->
      let why = strf "%s: %s" what (status_name c.layouts s) in
      if s = D.nv_err_no_memory || s = D.nv_err_insufficient_resources then
        Error (Rig_nv.Lacking why)
      else Error (Refused why)

(* The driver's RM: each of its commands is the RM's control of that name,
   which the kernel driver runs as it is. *)
let rm c =
  let alloc ~parent cls p =
    match alloc_raw c.ctl ~root:c.root ~parent cls p with
    | Ok (s, h) -> answer c (strf "allocating class 0x%x" cls) h (Ok s)
    | Error _ as e -> e
  in
  let control cmd obj p =
    answer c (strf "command 0x%x" cmd) ()
      (control_raw c.ctl ~root:c.root obj cmd (Some p))
  in
  let token ch =
    let module W = D.Work_submit_token in
    let p = params W.sizeof in
    set p W.work_submit_token 0xffff_ffff;
    let* () = control D.nvc36f_ctrl_cmd_gpfifo_get_work_submit_token ch p in
    Ok (get p W.work_submit_token)
  in
  let boost subdevice =
    let module B = D.Perf_boost in
    let p = params B.sizeof in
    set p B.flags
      (bits D.nv2080_ctrl_perf_boost_flags_cmd
         D.nv2080_ctrl_perf_boost_flags_cmd_boost_to_max
      lor bits D.nv2080_ctrl_perf_boost_flags_cuda
            D.nv2080_ctrl_perf_boost_flags_cuda_yes
      lor bits D.nv2080_ctrl_perf_boost_flags_cuda_priority
            D.nv2080_ctrl_perf_boost_flags_cuda_priority_high);
    set p B.duration D.nv2080_ctrl_perf_boost_duration_infinite;
    control D.nv2080_ctrl_cmd_perf_boost subdevice p
  in
  let free ~parent obj =
    answer c "freeing a GPU object" () (free_raw c ~parent obj)
  in
  {
    Rig_nv.release = c.release;
    client = c.root;
    alloc;
    token;
    schedule = control D.nva06c_ctrl_cmd_gpfifo_schedule;
    boost;
    sm_errors = control D.nv83de_ctrl_cmd_debug_read_all_sm_error_states;
    mmu_faults = control D.nv83de_ctrl_cmd_debug_read_mmu_fault_info;
    free = Some free;
  }

(* Unified memory *)

let uvm c cmd p status what =
  let* () = guard what (fun () -> ioctl c.uvm cmd p) in
  Ok (get p status)

let uvm_call c cmd p status what =
  let* s = uvm c cmd p status what in
  check c what s

(* The client *)

(* The GPU addresses of memory the host maps too, reserved in the process so
   that nothing else maps there: from 384 GiB to the windows
   ({!Rig_nv.windows}), and the windows up to 2^40 too, so that no address the
   process gives another allocation lies in them. They start above 258 GiB,
   where AddressSanitizer on x86_64 maps the shadow of its shadow gap when the
   gap is unprotected ([protect_shadow_gap=0]). *)
let low_base = 0x60_0000_0000
let top = 1 lsl 40

(* The release branch of a driver version, such as 615 of "615.71.09". *)
let branch version =
  match String.index_opt version '.' with
  | Some i -> int_of_string_opt (String.sub version 0 i)
  | None -> int_of_string_opt version

let driver_version ctl root =
  let module B = D.Build_version in
  let v = params B.sizeof in
  let* s =
    Result.map_error why
      (control_raw ctl ~root root D.nv0000_ctrl_cmd_system_get_build_version_v2
         (Some v))
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

let client_lock = Rig_nv.Lock.create ()
let opened = ref None

(* The process's client, or [Error]: a failed make, and one an exception ends,
   gives back what it took, last first. *)
let make_client ~dev =
  let undo = ref [] in
  let taken f = undo := f :: !undo in
  let uvm_path = Filename.concat dev "nvidia-uvm" in
  let opened_client () =
    let* ctl = open_file (Filename.concat dev "nvidiactl") in
    taken (fun () -> close ctl);
    let* s, root =
      Result.map_error why
        (alloc_raw ctl ~root:0 ~parent:0 D.nv01_root_client None)
    in
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
    (* Recorded before it is taken, and marked taken with nothing allocated
       after the call: a later reserve of a range this process still held
       would fail for good. *)
    let reserved = ref false in
    taken (fun () -> if !reserved then unreserve low_base (top - low_base));
    let* () =
      match reserve_raw low_base (top - low_base) with
      | () ->
          reserved := true;
          Ok ()
      | exception Unix.Unix_error (e, _, _) ->
          failed "reserving the GPU's addresses" e
    in
    let c =
      {
        ctl;
        uvm = uvm_fd;
        root;
        layouts;
        release = number;
        low = Va.make ~base:low_base (Rig_nv.windows - low_base);
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
    (try ioctl mm D.uvm_mm_initialize m with Unix.Unix_error _ -> ());
    Ok (c, Some c)
  in
  match opened_client () with
  | Ok _ as c -> c
  | Error _ as e ->
      List.iter (fun f -> f ()) !undo;
      e
  | exception e ->
      List.iter (fun f -> f ()) !undo;
      reraise e

let client ~dev =
  Rig_nv.Lock.protect client_lock @@ fun () ->
  match !opened with
  | Some c -> Ok c
  | None ->
      (* [Some c] is made inside the make, whose handler gives the client
         back, and nothing allocates from its return to the store. *)
      match make_client ~dev with
      | Error _ as e -> e
      | Ok (c, kept) ->
          opened := kept;
          Ok c

(* The handles the process names objects by. A freed one is taken again
   before a new one, so the highest stays [first_handle] plus the most objects
   named at once, far below 0xcaf00000, RS_UNIQUE_HANDLE_BASE, from which the
   RM chooses handles (open-gpu-kernel-modules' g_rs_client_nvoc.h). No lock,
   as [Va]'s ranges take none. *)
let first_handle = 0x1000
let next = Atomic.make first_handle
let freed : int list Atomic.t = Atomic.make []

let rec handle () =
  match Atomic.get freed with
  | [] -> Atomic.fetch_and_add next 1 + 1
  | h :: rest as l ->
      if Atomic.compare_and_set freed l rest then h else handle ()

let rec free_handle h =
  let l = Atomic.get freed in
  if not (Atomic.compare_and_set freed l (h :: l)) then free_handle h
