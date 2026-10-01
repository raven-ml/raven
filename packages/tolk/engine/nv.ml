(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk
module B = Nx_device.Buffer
module N = Nx_nv_device

(* The GPU as the compiler encodes its commands. *)
let gpu n =
  let p = N.props n in
  let channel (c : N.channel) =
    { Ops_nv.entries = B.nbytes c.ring / 8; token = c.token }
  in
  {
    Ops_nv.compute_class = p.compute_class;
    sass_version = p.sass_version;
    shared_window = Nativeint.to_int (N.shared_window n);
    local_window = Nativeint.to_int (N.local_window n);
    compute = channel (N.compute n);
    copy = channel (N.copy n);
  }

let channel n = function
  | "COMPUTE:0" -> N.compute n
  | "COPY:0" -> N.copy n
  | q -> invalid_arg (q ^ " is no NV queue")

(* The word of each device holding the bytes per thread its local memory
   provides, kept for the life of the process: every batch's descriptors read it
   when their host program runs, so a batch linked before the memory grew
   launches with the grown memory. A device's lock orders its growths with the
   writes of the word, which only grows. *)
let lock = Mutex.create ()
let local_words = ref []

let local d n bytes =
  let w, device_lock =
    Mutex.protect lock @@ fun () ->
    match List.assq_opt d !local_words with
    | Some w -> w
    | None ->
        let w =
          (B.create Nx_device.host Nx_dtype.Scalar.Int32 1, Mutex.create ())
        in
        local_words := (d, w) :: !local_words;
        w
  in
  Mutex.protect device_lock @@ fun () ->
  let m = N.local_memory n bytes in
  (B.bigarray Bigarray.int32 w).{0} <- Int32.of_int m.per_thread;
  w

(* A program's cubin, loaded on the device, relocated: the image its launches
   address. *)
let program d ~binary ~name =
  match Nx_device.Program.load d ~binary ~name with
  | Error why -> failwith why
  | Ok p -> (Option.get (N.kernel p)).image

(* The storage of the placeholders NV's commands name: those of the device
   [name]. *)
let placeholder name d n u =
  let on_device =
    match Ops.device u with
    | Some (Single n) | Some (Multi [ n ]) -> n = name
    | Some (Multi ds) when List.mem name ds && Ops_nv.storage u <> None ->
        (* A batch's calls on several devices run as one call per device, so its
           queues are on one device each. *)
        invalid_arg
          (Format.asprintf "an NV placeholder on several devices: %a" Ops.pp u)
    | _ -> false
  in
  if not on_device then None
  else
    Option.map
      (function
        | Ops_nv.Program { binary; name } -> program d ~binary ~name
        | Ring q -> (channel n q).ring
        | Gp_put q -> (channel n q).gp_put
        | Put q -> (channel n q).put
        | Doorbell q -> (channel n q).doorbell
        | Local bytes -> local d n bytes)
      (Ops_nv.storage u)

(* The queues address the memory nx.device says the device reaches: other memory
   is copied through the host's. *)
let reaches d devices name =
  match List.assoc_opt name devices with
  | Some d' -> Nx_device.reaches d d'
  | None -> false

let queues ~host devices name d =
  Option.map
    (fun n ->
      ( Ops_nv.queues ~host:(Lazy.force host) ~reaches:(reaches d devices)
          (gpu n),
        placeholder name d n,
        ignore ))
    (N.of_device d)
