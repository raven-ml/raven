(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
module P = Nx_nv_packet
module M = P.Methods
module Qmd = P.Qmd

let u32 = Hcq2.Queue.dword
let u64 n = int ~dtype:Dtype.Uint64 n

(* Devices *)

type channel = { entries : int; token : int }

type props = {
  compute_class : int;
  sass_version : int;
  shared_window : int;
  local_window : int;
  compute : channel;
  copy : channel;
}

(* Programs *)

type program = {
  program : P.Program.t;
  vars : Dtype.t list;
  kernargs_size : int;
  qmd : Ops.t Qmd.t; (* the template of the program's launches *)
}

let program_data props (obj : Device.Tiny_elf.t) =
  let cubin =
    match Nx_nv_cubin.of_string obj.lib with
    | Ok c -> c
    | Error why -> invalid_arg ("an NV program is no cubin: " ^ why)
  in
  let kernel =
    match Nx_nv_cubin.kernel cubin obj.name with
    | Some k -> k
    | None ->
        invalid_arg
          (Printf.sprintf "an NV program's cubin has no kernel %s, only %s"
             obj.name
             (String.concat ", " (Nx_nv_cubin.kernels cubin)))
  in
  let program =
    match
      P.Program.make ~compute_class:props.compute_class
        ~sass_version:props.sass_version ~shared_window:props.shared_window
        ~local_window:props.local_window kernel
    with
    | Ok p -> p
    | Error why -> invalid_arg (obj.name ^ ": " ^ why)
  in
  let sig_ = obj.signature in
  let nbufs =
    List.length
      (List.filter (fun (p : Device.Tiny_elf.param) -> p.name = None) sig_)
  in
  let vars =
    List.filteri (fun i _ -> i >= nbufs) sig_
    |> List.map (fun (p : Device.Tiny_elf.param) -> p.dtype)
  in
  (* Bank 0 holds the driver's parameters, then each argument in 8 bytes. *)
  let bank0 =
    List.find
      (fun (b : Nx_nv_cubin.bank) -> b.index = 0)
      (P.Program.banks program)
  in
  let kernargs_size =
    Helpers.round_up
      (max bank0.bytes
         (String.length (P.Program.driver_parameters program)
         + (List.length sig_ * 8)))
      256
  in
  ( String.length (Nx_nv_cubin.image cubin),
    { program; vars; kernargs_size; qmd = Qmd.make program } )

(* The local memory word of kernels that need [bytes] per thread on [devs]: one
   placeholder for every launch, which the engine binds to one word of the
   device. Distinct placeholders of one tag would become views of one buffer
   (Hcq2), which the word does not hold. Built by [build_program], under its
   lock. *)
let local_words = Hashtbl.create 4

let local_word devs bytes =
  let word =
    match Hashtbl.find_opt local_words (devs, bytes) with
    | Some w -> w
    | None ->
        let w =
          placeholder ~device:(Multi devs)
            ~tag:(Tag.Tuple [ String "nv_local"; Int bytes ])
            [ 1 ] Dtype.Uint32
        in
        Hashtbl.replace local_words (devs, bytes) w;
        w
  in
  load (index word [ int 0 ]) []

(* Each program's launch template is built once for its kernel of its cubin and
   its devices. Its cubin is the engine's to load, relocated by the device that
   runs it: the placeholder names the cubin and its kernel, and is as long as
   the image the device lays out, with room for the GPU's prefetch after it. *)
let programs = Hashtbl.create 16
let programs_lock = Mutex.create ()

let build_program props devs prg =
  let obj = Device.Tiny_elf.of_program prg in
  Mutex.protect programs_lock @@ fun () ->
  match Hashtbl.find_opt programs (obj.lib, obj.name, devs, props) with
  | Some p -> p
  | None ->
      let image_size, data = program_data props obj in
      let local = local_word devs (P.Program.local_bytes data.program) in
      let data = { data with qmd = Qmd.set_local_memory data.qmd local } in
      let p =
        ( data,
          placeholder ~slot:0 ~device:(Multi devs)
            ~tag:
              (Tag.Tuple [ String "program"; Bytes obj.lib; String obj.name ])
            [ image_size ] Dtype.Uint8 )
      in
      Hashtbl.replace programs (obj.lib, obj.name, devs, props) p;
      p

(* Queues *)

let queue props q : Hcq2.commands =
  let devs = Hcq2.Queue.devices q and name = Hcq2.Queue.name q in
  let dev = Multi devs and on = List.hd devs in
  let emit words = ignore (Hcq2.Queue.q q (Nv_packet.words words)) in
  let addr u = getaddr ~device:on u in
  let copy_queue = String.starts_with ~prefix:"COPY" name in
  (* NVQueue *)
  let wait signal value = emit (M.acquire (addr signal) value) in
  let release signal value ~timestamp =
    if timestamp then emit (M.release_stamp (addr signal) value)
    else emit (M.release (addr signal) value)
  in
  let submit_cmdbuf cmdbuf =
    let fifo = if copy_queue then props.copy else props.compute in
    let ib, off = Hcq2.unwrap_view cmdbuf in
    let word nm dt sz =
      placeholder ~device:dev ~volatile:true
        ~tag:(Tag.String (Hcq2.to_name [ nm; name ]))
        [ sz ] dt
    in
    let ring = word "ring" Dtype.Uint64 fifo.entries
    and gpput = word "gpput" Dtype.Uint32 1
    and doorbell = word "doorbell" Dtype.Uint32 1
    and put = word "put_value" Dtype.Uint64 1 in
    let dwords = max_numel cmdbuf * Dtype.itemsize (dtype cmdbuf) / 4 in
    if dwords > P.Gpfifo.max_words then
      raise
        (Hcq2.Over_capacity
           (Printf.sprintf
              "an NV command buffer of %d words exceeds a GPFIFO entry's %d"
              dwords P.Gpfifo.max_words));
    let gpentry =
      Hcq2.patch
        (word "gpentry" Dtype.Uint64 1)
        [
          ( int 0,
            Nv_packet.term (P.Gpfifo.entry (addr ib) ~offset:off ~words:dwords)
          );
        ]
    in
    let p = load (index put [ int 0 ]) [] in
    let written =
      barrier
        (store
           (index (after ring [ cmdbuf ])
              [ cast (mod_ p (int fifo.entries)) Dtype.Int32 ])
           (load (index gpentry [ int 0 ]) []))
        [ store (index put [ int 0 ]) (add p (int 1)) ]
    in
    let queued =
      barrier
        (store
           (index (after gpput [ written ]) [ int 0 ])
           (cast (mod_ (add p (int 1)) (int fifo.entries)) Dtype.Uint32))
        []
    in
    store (index (after doorbell [ queued ]) [ int 0 ]) (u32 fifo.token)
  in
  (* NVComputeQueue *)
  let chain = ref [] in
  (* The launches of a chain end with it: each descriptor, built back to front,
     points to the next, and the channel schedules the first. *)
  let end_chain () =
    let rec build = function
      | [] -> None
      | qmd :: rest ->
          let qmd =
            match build rest with
            | Some next -> Qmd.chain qmd (addr next)
            | None -> qmd
          in
          Some (Nv_packet.structure "qmd" (Qmd.structure qmd))
    in
    Option.iter (fun head -> emit (M.schedule (addr head))) (build !chain);
    chain := []
  in
  let exec call prg =
    if copy_queue then invalid_arg "an NV copy queue runs no program";
    let data, lib = build_program props devs prg in
    let info =
      match arg prg with
      | Program p -> p
      | _ -> invalid_arg "an NV command runs a compiled program"
    in
    let known = List.map (function Int n -> n | Sym _ -> 1) in
    let threads = Helpers.prod (known info.local_size) in
    let max_threads = P.Program.max_threads data.program in
    if threads > 1024 || max_threads < threads then
      invalid_arg
        (Printf.sprintf
           "Too many resources requested for launch, prod(local_size)=%d, \
            data.max_threads=%d"
           threads max_threads);
    let exceeds sizes limits =
      List.exists2
        (fun g m -> match g with Int g -> g > m | Sym _ -> false)
        sizes limits
    in
    if
      exceeds info.global_size [ 2147483647; 65535; 65535 ]
      || exceeds info.local_size [ 1024; 1024; 64 ]
    then invalid_arg "Invalid global/local dims";
    let qmd =
      List.fold_left2
        (fun q d s ->
          match s with
          | Int n -> Qmd.set_dim q d n
          | Sym u -> Qmd.patch_dim q d u)
        data.qmd
        P.[ Grid X; Grid Y; Grid Z; Block X; Block Y; Block Z ]
        (info.global_size @ info.local_size)
    in
    let qmd =
      Qmd.set_program qmd (add (addr lib) (int (P.Program.code data.program)))
    in
    (* constant buffer 0: the driver params, then the arguments *)
    let bufs = Realize.get_call_arg_uops call in
    let vals = Realize.get_call_var_uops call prg in
    let params = P.Program.driver_parameters data.program in
    let at = String.length params in
    let args =
      Hcq2.layout_args ~offset:at
        (List.map (fun g -> addr (List.nth bufs g)) info.globals
        @ List.map2 ccast vals data.vars)
    in
    let driver = Bytes.make data.kernargs_size '\000' in
    Bytes.blit_string params 0 driver 0 at;
    let cbuf = Nv_packet.region "cbuf" (Bytes.to_string driver) args in
    let qmd =
      List.fold_left
        (fun q (b : Nx_nv_cubin.bank) ->
          Qmd.set_bank q b.index
            (if b.index = 0 then addr cbuf else add (addr lib) (int b.offset)))
        qmd
        (P.Program.banks data.program)
    in
    chain := !chain @ [ qmd ]
  in
  let compute_release signal value ~timestamp =
    let released =
      match List.rev !chain with
      | prev :: before ->
          (if timestamp then Qmd.release_stamp else Qmd.release)
            prev (addr signal) value
          |> Option.map (fun q -> List.rev (q :: before))
      | [] -> None
    in
    match released with
    | Some c -> chain := c
    | None ->
        end_chain ();
        release signal value ~timestamp
  in
  (* NVCopyQueue *)
  let copy dst src n =
    if not copy_queue then invalid_arg "an NV compute queue copies nothing";
    emit (M.copy ~dst:(addr dst) ~src:(addr src) n)
  in
  (* The copy engine writes a 64-bit value in 32-bit words: its low word, then
     its high word, where the value lives when the low word wrapped to 0 and
     into a word of its own otherwise. *)
  let copy_signal signal value =
    let a = addr signal in
    let sink =
      placeholder ~device:dev ~volatile:true ~tag:(Tag.String "nv_sink") [ 1 ]
        Dtype.Uint32
    in
    emit (M.copy_release a value);
    emit
      (M.copy_release
         (where
            (eq (cast value Dtype.Uint32) (u32 0))
            (add a (u64 4))
            (addr sink))
         (shr value (int 32)))
  in
  let timestamp slot =
    if copy_queue then emit (M.copy_stamp (addr slot))
    else compute_release slot (u64 0) ~timestamp:true
  in
  {
    exec;
    copy;
    wait =
      (fun signal value ->
        end_chain ();
        wait signal value);
    signal =
      (fun signal value ->
        if copy_queue then copy_signal signal value
        else compute_release signal value ~timestamp:false);
    timestamp;
    memory_barrier =
      (fun () ->
        if not copy_queue then (
          end_chain ();
          emit M.invalidate_caches));
    (* A chain ends at a loop's edges and at the end of each trip: a trip's
       launches chain onto its own descriptors. The channel runs the chains it
       schedules at once: the batch orders them, each waiting for the launch
       before it (Hcq2's [make_ctx]). *)
    loop =
      (fun r body ->
        end_chain ();
        Hcq2.Queue.loop q r (fun () ->
            body ();
            end_chain ()));
    submit =
      (fun () ->
        end_chain ();
        submit_cmdbuf (Hcq2.bufferize_cmdbuf q "cmdbuf"));
  }

let queues ~host ~reaches props =
  {
    Hcq2.commands = queue props;
    copy_queue = true;
    submission = Buffered;
    host;
    reaches;
  }

(* Linking *)

type storage =
  | Program of { binary : string; name : string }
  | Ring of string
  | Gp_put of string
  | Put of string
  | Doorbell of string
  | Local of int

let storage u =
  match tag u with
  | Some (Tag.Tuple [ String "program"; Bytes binary; String name ]) ->
      Some (Program { binary; name })
  | Some (Tag.Tuple [ String "nv_local"; Int bytes ]) -> Some (Local bytes)
  | Some (Tag.String t) -> (
      (* [name_queue_index], as submit tags them. *)
      match List.rev (String.split_on_char '_' t) with
      | i :: queue :: name -> (
          let queue = String.uppercase_ascii queue ^ ":" ^ i in
          match String.concat "_" (List.rev name) with
          | "ring" -> Some (Ring queue)
          | "gpput" -> Some (Gp_put queue)
          | "put_value" -> Some (Put queue)
          | "doorbell" -> Some (Doorbell queue)
          | _ -> None)
      | _ -> None)
  | _ -> None
