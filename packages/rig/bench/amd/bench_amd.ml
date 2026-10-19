(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPU 0 through the driver, its work submitted through rig, as a program
   submits it. Memory and image rows call the driver. A row waits by spinning on
   the word, except the wake rows.

   Each case opens its device in its own worker: KFD gives a process one address
   space per GPU, kept by the first render node that acquires it, so one process
   cannot hold both. Without an AMD GPU the suite has no rows.

   With RIG_AMD_PCI_FIRMWARE set, the GPU detached, the same rows run with no
   kernel driver ({!Rig_amd_support.open_gpu}), beside the opens of a full and a
   partial boot; the host mappings, which only KFD has, are left out.
   Driver-less is another configuration of the machine, so its rows are
   recorded under a machine key of their own, the host's with [-pci] appended,
   which the run sets (THUMPER_MACHINE): they compare by name with the rows of
   the section KFD's run records. *)

module A = Rig_amd
module B = Rig.Buffer
module Abi = Rig_amd_abi
module Packet = Abi.Packet
module Pm4 = Abi.Pm4

external pages : int -> int = "rig_amd_bench_pages"
external read : int -> int -> int = "rig_amd_bench_read"
external write : int -> string -> unit = "rig_amd_bench_write"

let kib = 1024
let mib = 1024 * kib
let get = function Ok x -> x | Error why -> failwith why
let host r = Option.get (A.locate r).host

(* Whether the rows open the GPU with no kernel driver. *)
let pci = Rig_amd_support.driverless ()

(* A worker ends without the exit's handlers, so each row's teardown closes
   rig's devices: a driver-less GPU left running would go on mastering the bus
   for the next worker. *)
let row name setup f =
  Thumper.bench_with_setup ~setup
    ~teardown:(fun _ -> Rig_amd_support.release ())
    name f

(* The driver *)

(* A device and the value its row last submitted. *)
type dev = { d : Rig.t; g : A.t; mutable v : int }

(* A device opened through rig. *)
let dev () =
  let { Rig_amd_support.d; g } = Rig_amd_support.open_ () in
  { d; g; v = 0 }

(* A run of the prepared submission of [parts] on [t]. *)
let prepare ?keep ?fixed t parts =
  Rig.Submission.(Run.make (make ?keep ?fixed t.d parts))

let submit t r =
  let p = Rig.submit r ~buffers:[||] in
  t.v <- Rig.Point.value p

let wait t =
  while A.signaled t.g < t.v do
    Domain.cpu_relax ()
  done

let run t s =
  submit t s;
  wait t

(* A submission whose one part, a copy of no bytes on the copy queue, places
   nothing: its value is released by the copy queue. *)
let on_copy t =
  prepare t
    Rig.Submission.
      [| copy ~src:(Buffer (B.create t.d 0)) ~dst:(Buffer (B.create t.d 0)) |]

(* Packets *)

let encode p = Packet.encode Int64.of_int p

(* Kernels *)

(* Read by the rows that run a kernel, so that the others, and a run that
   selects none of them, start without the fixture. *)
let binary =
  lazy
    (In_channel.with_open_bin "../../test/amd/fixtures/kernels_gfx1201.hsaco"
       In_channel.input_all)

let code_object = lazy (get (Abi.Code_object.of_string (Lazy.force binary)))

let kernel name =
  Option.get (Abi.Code_object.kernel (Lazy.force code_object) name)

(* One run of [k] over one workgroup of [threads] work-items, its image at
   [base], its arguments at [args]. *)
let dispatch gpu (k : Abi.Code_object.kernel) ~base ~args ~threads =
  Pm4.run gpu
    (Pm4.dispatch gpu k ~program:(base + k.entry) ~scratch:0 ~args ~packet:0
       ~threads:(threads, 1, 1) ~groups:(1, 1, 1) ())

(* The code object loaded on [t], and the address of its image. *)
let load t =
  let p = get (Rig.Image.load t.d (Lazy.force binary)) in
  let empty = Option.get (Rig.Image.resolve p "empty") in
  (p, Rig.Image.entry empty - (kernel "empty").descriptor)

(* Rows *)

let release_rows =
  let empty () =
    let t = dev () in
    (t, prepare t [||])
  in
  (* Values released by the compute queue and the copy queue in turn. *)
  let switching () =
    let t = dev () in
    (t, on_copy t, prepare t [||])
  in
  let copying () =
    let t = dev () in
    (t, on_copy t)
  in
  let mapped () =
    let t = dev () in
    (t, B.create ~memory:Mapped t.d 4096, prepare t [||])
  in
  Thumper.group "release"
    [
      row "driver" empty (fun (t, s) -> run t s);
      row "switch" switching (fun (t, copy, none) ->
          run t (if t.v land 1 = 0 then copy else none));
      row "copy" copying (fun (t, s) -> run t s);
      row "no-wait-100" empty (fun (t, s) ->
          for _ = 1 to 100 do
            submit t s
          done;
          wait t);
      row "mapped" mapped (fun (t, _, s) -> run t s);
    ]

let launch_rows =
  (* [n] launches of [empty] over one work-item, a part each. *)
  let launching n () =
    let t = dev () in
    let p, _ = load t in
    let fn = Option.get (Rig.Image.resolve p "empty") in
    let empty = Rig.Submission.launch fn ~params:0 in
    let r = prepare t (Array.make n empty) in
    for i = 0 to n - 1 do
      let b = Rig.Submission.(block (Run.submission r) i) in
      Rig.Submission.Run.groups r b 1 1 1;
      Rig.Submission.Run.threads r b 1 1 1
    done;
    (t, p, r)
  in
  (* Fixed dispatch packets, one words part per launch. *)
  let parts n () =
    let t = dev () in
    let p, base = load t in
    let gpu = (A.capability t.g).gpu in
    let one =
      B.of_string
        (encode (dispatch gpu (kernel "empty") ~base ~args:0 ~threads:1))
    in
    let r = prepare ~keep:p t (Array.make n (Rig.Submission.words one)) in
    (t, p, r)
  in
  (* [double_index] over 64 work-items, its argument the address of its run's
     buffer, which the hand-over adds to the offset the run stores. *)
  let params () =
    let t = dev () in
    let p, _ = load t in
    let out = B.create t.d 256 in
    let fn = Option.get (Rig.Image.resolve p "double_index") in
    let refs = Rig.Submission.[| write 0 ~at:0 |] in
    let r = prepare t [| Rig.Submission.launch ~refs fn ~params:8 |] in
    let b = Rig.Submission.(block (Run.submission r) 0) in
    Rig.Submission.Run.groups r b 1 1 1;
    Rig.Submission.Run.threads r b 64 1 1;
    Rig.Submission.Run.int64 r b 0 0L;
    (t, p, r, [| out |])
  in
  let bar () =
    let t = dev () in
    let p, base = load t in
    let gpu = (A.capability t.g).gpu in
    let out = B.create t.d 256 in
    let args =
      try B.create ~memory:Mapped t.d 8
      with Rig.Out_of_memory _ -> B.create ~memory:Pinned t.d 8
    in
    let encode args =
      encode (dispatch gpu (kernel "double_index") ~base ~args ~threads:64)
    in
    let ws = B.create Rig.host (String.length (encode 0)) in
    let s =
      prepare ~keep:p
        ~fixed:[ (args, Read); (out, Read_write) ]
        t
        [| Rig.Submission.words ws |]
    in
    let address = Rig.Submission.(address (Run.submission s)) in
    let bytes = encode (address args) in
    B.blit_from_string bytes 0 ws 0 (String.length bytes);
    (* The rows rewrite the arguments in place, as compiled code would. *)
    (t, p, B.write args Bigarray.int64 Fun.id, address out, s)
  in
  Thumper.group "launch"
    [
      row "1" (launching 1) (fun (t, _, s) -> run t s);
      row "64" (launching 64) (fun (t, _, s) -> run t s);
      row "64-parts" (parts 64) (fun (t, _, s) -> run t s);
      row "submits-64" (launching 1) (fun (t, _, s) ->
          for _ = 1 to 64 do
            submit t s
          done;
          wait t);
      row "1-params" params (fun (t, _, r, buffers) ->
          let p = Rig.submit r ~buffers in
          t.v <- Rig.Point.value p;
          wait t);
      row "1-bar-args" bar (fun (t, _, args, out, s) ->
          args.{0} <- Int64.of_int out;
          run t s);
    ]

let copy_rows =
  let n = 256 * mib in
  let copy t ~dst ~src n =
    let buffer memory = B.create ~memory t.d n in
    let src = buffer src and dst = buffer dst in
    Rig.Submission.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  let copying n (dst, src) () =
    let t = dev () in
    (t, prepare t [| copy t ~dst ~src n |])
  in
  let copy_row name n kinds =
    row name (copying n kinds) (fun (t, s) -> run t s)
  in
  let both () =
    let t = dev () in
    let h2d = copy t ~dst:Device ~src:Pinned n in
    let d2h = copy t ~dst:Pinned ~src:Device n in
    (t, prepare t [| h2d; d2h |])
  in
  let host_n = 64 * mib in
  let pinned () =
    let t = dev () in
    let r = Option.get (A.alloc t.g Rig_edge.Pinned host_n) in
    write (host r) (String.make host_n '\001');
    (t, host r)
  in
  Thumper.group "copy"
    [
      copy_row "h2d-256MiB" n (B.Device, B.Pinned);
      copy_row "d2h-256MiB" n (B.Pinned, B.Device);
      copy_row "d2d-256MiB" n (B.Device, B.Device);
      copy_row "16B" 16 (B.Device, B.Device);
      row "bidir-256MiB" both (fun (t, s) -> run t s);
      row "host-read-pinned" pinned (fun (_, p) -> read p host_n);
    ]

let alloc_rows =
  let alloc name n =
    row name dev (fun t ->
        A.free t.g (Option.get (A.alloc t.g Rig_edge.Device n)))
  in
  Thumper.group "alloc" [ alloc "64KiB" (64 * kib); alloc "64MiB" (64 * mib) ]

let map_host_rows =
  let n = 256 * mib in
  Thumper.group "map-host"
    [
      row "256MiB"
        (fun () -> (dev (), pages n))
        (fun (t, p) -> A.free t.g (Option.get (A.map_host t.g p n)));
    ]

let image_rows =
  Thumper.group "image"
    [
      (* The code object read, laid over device memory, ended and its memory
         freed. *)
      row "one-object" dev (fun t ->
          match A.image t.g (Lazy.force binary) with
          | Ok (Rig_edge.Place (n, lay)) ->
              let code = Option.get (A.alloc t.g Rig_edge.Device n) in
              A.unload t.g (fst (lay code));
              A.free t.g code
          | Ok (Rig_edge.Loaded _) ->
              failwith "code the device's library placed"
          | Error why -> failwith why);
    ]

(* A release waited by blocking in [sleep], which returns on the interrupt the
   release raises. *)
let wake_rows =
  let rec sleep t =
    let seen = A.signaled t.g in
    if seen < t.v then begin
      A.sleep t.g ~seen ~still_ms:1000;
      sleep t
    end
  in
  let empty () =
    let t = dev () in
    (t, prepare t [||])
  in
  let copy () =
    let t = dev () in
    (t, on_copy t)
  in
  Thumper.group "wake"
    [
      row "driver" empty (fun (t, s) ->
          submit t s;
          sleep t);
      row "copy" copy (fun (t, s) ->
          submit t s;
          sleep t);
    ]

(* Opens of the GPU with no kernel driver: a partial boot over the clean mark
   the last stop left, and a full one, which only a GPU reset to its bootloader
   takes, so that row times the reset too. Neither has a KFD twin. *)
let open_rows =
  let open_stop () =
    A.stop (get (Rig_gpu_support.opened (Rig_amd_support.gpu ()))) ~fault:None
  in
  Thumper.group "open"
    [
      row "partial" open_stop (fun () -> open_stop ());
      row "full" open_stop (fun () ->
          get (Rig_amd_pci.reset 0);
          open_stop ());
    ]

(* A full open, a reset included, takes over a second: the trials of the run
   with no kernel driver may take a minute. *)
let pci_deadline = 60.

let () =
  if pci then
    Unix.putenv "THUMPER_MACHINE" (Thumper.Baseline.machine_key () ^ "-pci");
  if Rig_amd_support.gpus () > 0 then
    let config =
      if pci then Thumper.Config.(deadline pci_deadline default)
      else Thumper.Config.default
    in
    exit
    @@ Thumper.run ~config "rig_amd"
         ((if pci then [ open_rows ] else [])
         @ [ release_rows; launch_rows; copy_rows; alloc_rows ]
         @ (if pci then [] else [ map_host_rows ])
         @ [ image_rows; wake_rows ])
