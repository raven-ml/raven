(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPU 0 through the driver, its work submitted through rig, each row beside the
   packets that bound it, placed from C on a compute and a copy queue the bench
   makes through KFD: a release by a release packet that raises the interrupt a
   sleeping host wakes on, as the driver's does, a ring of the doorbell and a
   spin on a word; a queue switch by the same after a wait for the value before;
   foreign waits by 64-bit waits; launches by the same dispatch words; copies by
   the same SDMA copies; allocations and host mappings by the KFD calls behind
   them. The floors' packets are the ABI's, encoded once as the driver encodes
   its own, without the driver's cache acquire, slot words and host data path
   flush. A row waits by spinning on the word, except the wake row.

   Each case opens its device, or the floors' queues, in its own worker: KFD
   gives a process one address space per GPU, kept by the first render node that
   acquires it, so one process cannot hold both. Without an AMD GPU the suite
   has no rows.

   With RIG_AMD_PCI_FIRMWARE set, the GPU detached, the same rows run with no
   kernel driver ({!Rig_amd_support.open_gpu}), named with a [pci-] marker,
   beside the opens of a full and a partial boot. The floors and the host
   mappings, which only KFD has, are left out. *)

module A = Rig_amd
module P = Rig_amd_amdgpu
module B = Rig.Buffer
module Abi = Rig_amd_abi
module Packet = Abi.Packet
module Pm4 = Abi.Pm4
module Sdma = Abi.Sdma

external start : int array -> unit = "rig_amd_bench_start"
external interrupt : unit -> int = "rig_amd_bench_interrupt"

external template : int -> string -> int64 array -> unit
  = "rig_amd_bench_template"

external floor_release : int -> unit = "rig_amd_bench_release"
external floor_release_agent : int -> unit = "rig_amd_bench_release_agent"
external floor_switch : unit -> unit = "rig_amd_bench_switch"
external floor_waits : int -> int -> unit = "rig_amd_bench_waits"
external floor_launch : string -> int -> unit = "rig_amd_bench_launch"
external floor_copy : int -> int -> int -> unit = "rig_amd_bench_copy"
external buffer : int -> int -> int = "rig_amd_bench_buffer"
external floor_alloc : int -> int -> unit = "rig_amd_bench_alloc"
external floor_map_host : int -> int -> unit = "rig_amd_bench_map_host"
external pages : int -> int = "rig_amd_bench_pages"
external read : int -> int -> int = "rig_amd_bench_read"
external set64 : int -> int -> unit = "rig_amd_bench_set64" [@@noalloc]
external fill_entry : unit -> nativeint = "rig_amd_bench_fill_entry"

external fill_arg :
  nativeint ->
  nativeint ->
  int ->
  string ->
  int64 array ->
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "rig_amd_bench_fill_arg"

external write : int -> string -> unit = "rig_amd_bench_write"

external raw_submit : nativeint array -> int -> int -> int -> int -> unit
  = "rig_amd_bench_submit"

let strf = Printf.sprintf
let kib = 1024
let mib = 1024 * kib
let get = function Ok x -> x | Error why -> failwith why
let host r = Option.get (A.host r)
let address r = Option.get (A.address r)
(* Whether the rows open the GPU with no kernel driver. *)
let pci = Rig_amd_support.driverless ()

let row name setup f =
  Thumper.bench_with_setup ~setup ((if pci then "pci-" else "") ^ name) f

(* A row of KFD's alone: the floors, and the host mappings. *)
let kfd_row name setup f = if pci then [] else [ row name setup f ]

(* The driver *)

type dev = { d : Rig.t; g : A.t; mutable v : int }

let opens = ref 0

(* A device opened through rig, under a name of its own. *)
let dev () =
  incr opens;
  let g = ref None in
  let make () =
    Result.map
      (fun x ->
        g := Some x;
        x)
      (Rig_amd_support.open_gpu ())
  in
  let d = get (Rig.open_ (module A) ~name:(strf "AMD:bench-%d" !opens) make) in
  { d; g = Option.get !g; v = 0 }

(* The prepared submission of [parts] on [t]. *)
let prepare t parts = Rig.Submission.make ~reads:0 ~writes:0 t.d parts

let submit t s =
  t.v <- Rig.Point.value (Rig.submit s ~reads:[||] ~writes:[||] ~waits:[||])

let wait t =
  while A.signaled t.g < t.v do
    Domain.cpu_relax ()
  done

let run t s =
  submit t s;
  wait t

let part queue work = { Rig.Submission.queue; after = [||]; work }

(* A submission whose one part, on the copy queue, places nothing: its value is
   released by the copy queue. *)
let on_copy t =
  let none = B.of_bigarray Bigarray.(Array1.create int32 c_layout 0) in
  prepare t [| part "COPY:0" (Words none) |]

(* Packets *)

let encode p = Packet.encode Int64.of_int p

(* A template's holes as the C reads them: each word's index, its width, the
   argument [arg v] of its value, then the operations on it in the order they
   apply, each with its constant's 64 bits. *)
let holes arg hs =
  let hole (at, w) =
    let rec flatten ops : _ Packet.term -> _ = function
      | Value v -> (arg v, ops)
      | Add (t, k) -> flatten ((0L, k) :: ops) t
      | Shift (t, n) -> flatten ((1L, Int64.of_int n) :: ops) t
      | Or (t, k) -> flatten ((2L, k) :: ops) t
    in
    let wide, t =
      match (w : _ Packet.word) with
      | W32 t -> (1, t)
      | W64 t -> (2, t)
      | Dword _ -> assert false
    in
    let a, ops = flatten [] t in
    List.map Int64.of_int [ at; wide; a; List.length ops ]
    @ List.concat_map (fun (op, k) -> [ op; k ]) ops
  in
  Array.of_list (List.concat_map hole hs)

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
  (p, Option.get (Rig.Image.entry p "empty") - (kernel "empty").descriptor)

(* The floors *)

let nodes = "/sys/devices/virtual/kfd/kfd/topology/nodes"

let ints file =
  In_channel.with_open_text file In_channel.input_lines
  |> List.filter_map (fun l ->
      match String.split_on_char ' ' l with
      | [ k; v ] -> Option.map (fun v -> (k, v)) (int_of_string_opt v)
      | _ -> None)

(* The KFD topology node of the GPU at bus address [bus], its properties and its
   GPU id. *)
let node bus =
  let at p =
    match (List.assoc_opt "domain" p, List.assoc_opt "location_id" p) with
    | Some d, Some l ->
        Printf.sprintf "%04x:%02x:%02x.%d" d (l lsr 8)
          ((l lsr 3) land 0x1f)
          (l land 7)
        = bus
    | _ -> false
  in
  let read n =
    let dir = Filename.concat nodes n in
    let p = ints (Filename.concat dir "properties") in
    if not (at p) then None
    else
      let id =
        In_channel.with_open_text
          (Filename.concat dir "gpu_id")
          In_channel.input_all
      in
      Some (("gpu_id", int_of_string (String.trim id)) :: p)
  in
  Option.get (List.find_map read (Array.to_list (Sys.readdir nodes)))

(* The floors' packets, in rig_amd_bench_stubs.c's order, their values the
   arguments 0, 1 and 2 of a use. *)
let templates gpu ~waits ~interrupt =
  [
    Pm4.release_mem gpu System ~interrupt 0 (Data_64 1);
    Pm4.release_mem gpu Agent ~interrupt 0 (Data_64 1);
    Pm4.wait gpu (Memory 0) Equal 1 ();
    (if waits then Pm4.wait_64 gpu 0 Greater_equal 1 () else []);
    Sdma.poll 0 Equal 1 ();
    Sdma.fence gpu 0 1;
    Sdma.copy_linear ~dst:0 ~src:1 ~bytes:2;
    Sdma.trap;
  ]

(* The floors' queues for GPU 0, once per worker; [waits] encodes the 64-bit
   wait, which only GPUs whose driver waits on other devices run. *)
let floor_with ~waits () =
  let bus = List.hd (P.gpus_at "/") in
  let gpu = get (P.gpu_at "/" bus) in
  let p = node bus in
  let prop k = List.assoc k p in
  start
    [|
      prop "gpu_id";
      prop "drm_render_minor";
      prop "cwsr_size";
      prop "ctl_stack_size";
      get (P.save_area_at "/" bus);
      Sdma.max_copy gpu;
    |];
  let set i p =
    let ws, hs = Packet.template (fun _ -> None) p in
    template i ws (holes Fun.id hs)
  in
  List.iteri set (templates gpu ~waits ~interrupt:(interrupt ()));
  gpu

let floor () = floor_with ~waits:false ()
let gpu_memory = 0
let system_memory = 1

(* The image of the code object as it lies in memory, at the GPU address of the
   result, copied in by the floor's copy queue. *)
let floor_load () =
  let b = Abi.Code_object.image (Lazy.force code_object) in
  let n = String.length b in
  let code = buffer gpu_memory n and staging = buffer system_memory n in
  write staging b;
  floor_copy code staging n;
  code

(* Rows *)

let release_rows =
  let empty () =
    let t = dev () in
    (t, prepare t [||])
  in
  (* Values released by the compute queue and the copy queue in turn, as the
     floor's switch releases them. *)
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
  @@ List.concat
    [
      [ row "driver" empty (fun (t, s) -> run t s) ];
      kfd_row "floor" floor (fun _ -> floor_release 1);
      kfd_row "floor-agent" floor (fun _ -> floor_release_agent 1);
      [
        row "switch" switching (fun (t, copy, none) ->
            run t (if t.v land 1 = 0 then copy else none));
      ];
      kfd_row "floor-switch" floor (fun _ -> floor_switch ());
      [
        row "copy" copying (fun (t, s) -> run t s);
        row "no-wait-100" empty (fun (t, s) ->
            for _ = 1 to 100 do
              submit t s
            done;
            wait t);
      ];
      kfd_row "floor-no-wait-100" floor (fun _ -> floor_release 100);
      [ row "mapped" mapped (fun (t, _, s) -> run t s) ];
    ]

(* Waits on a word the host set, and a wait across 2^32: the host sets the word
   to 2^32 k - 1, the queue waits for 2^32 k, and the host stores it after the
   submission. The row times the submission, the store and the release; that
   the wait holds its work back until the store is test_amd's to show. rig
   waits on other devices' values only, so these rows submit through the
   driver's C entries, on a device opened without it. *)
let wait_rows =
  (* The driver's entries, the device, and the word's host and GPU addresses,
     read once so that a submission allocates nothing. *)
  let raw () =
    let g = get (Rig_amd_support.open_gpu ()) in
    let w = Option.get (A.alloc g `Pinned 8) in
    ([| A.room_entry; A.submit_entry; A.self g |], g, host w, address w, ref 0)
  in
  let floor_waiting () =
    ignore (floor_with ~waits:true ());
    let w = buffer system_memory 4096 in
    set64 w 1;
    w
  in
  let spin g v =
    while A.signaled g < v do
      Domain.cpu_relax ()
    done
  in
  Thumper.group "waits"
  @@ List.concat
       [
         [
           row "4"
             (fun () ->
               let ((_, _, w, _, _) as r) = raw () in
               set64 w 1;
               r)
             (fun (f, g, _, at, v) ->
               incr v;
               raw_submit f !v at 1 4;
               spin g !v);
         ];
         kfd_row "floor-4" floor_waiting (fun at -> floor_waits at 4);
         [
           row "wait64" raw (fun (f, g, w, at, v) ->
               incr v;
               let target = !v lsl 32 in
               set64 w (target - 1);
               raw_submit f !v at target 1;
               set64 w target;
               spin g !v);
         ];
       ]

let launch_rows =
  (* A part placing the template [ws] with holes [hs] by the bench's fill, the
     fill's argument [arg] where the template has holes. *)
  let filled t ?(arg = 0) ?(bytes = 0) (ws, hs) =
    let c = A.capability t.g in
    let arg = B.of_bigarray (fill_arg c.place c.segment arg ws hs) in
    let ring_units = String.length ws / 4 in
    part "COMPUTE:0"
      (Fill { fill = fill_entry (); arg; ring_units; segment_bytes = bytes })
  in
  let launching count () =
    let t = dev () in
    let p, base = load t in
    let gpu = (A.capability t.g).gpu in
    let one = encode (dispatch gpu (kernel "empty") ~base ~args:0 ~threads:1) in
    let ws = String.concat "" (List.init count (fun _ -> one)) in
    (t, p, prepare t [| filled t (ws, [||]) |])
  in
  let floor_launching () =
    let gpu = floor () in
    let base = floor_load () in
    encode (dispatch gpu (kernel "empty") ~base ~args:0 ~threads:1)
  in
  (* [double_index] over 64 work-items, its argument, the output's address,
     written for each launch: into the argument segment by the fill, or into
     [`Mapped] memory by the host, which the driver flushes into the GPU's
     memory before the doorbell. *)
  let segment () =
    let t = dev () in
    let p, base = load t in
    let gpu = (A.capability t.g).gpu in
    let out = Option.get (A.alloc t.g `Device 256) in
    let k = kernel "double_index" in
    let known = function `Known n -> Some (Int64.of_int n) | `Args -> None in
    let run =
      Pm4.run gpu
        (Pm4.dispatch gpu k
           ~program:(`Known (base + k.entry))
           ~scratch:(`Known 0) ~args:`Args ~packet:(`Known 0)
           ~threads:(`Known 64, `Known 1, `Known 1)
           ~groups:(`Known 1, `Known 1, `Known 1)
           ())
    in
    let ws, hs = Packet.template known run in
    let f = filled t ~arg:(address out) ~bytes:8 (ws, holes (fun _ -> 0) hs) in
    (t, p, prepare t [| f |])
  in
  let bar () =
    let t = dev () in
    let p, base = load t in
    let gpu = (A.capability t.g).gpu in
    let out = Option.get (A.alloc t.g `Device 256) in
    let args = Option.get (A.alloc t.g `Mapped 8) in
    let ws =
      encode
        (dispatch gpu (kernel "double_index") ~base ~args:(address args)
           ~threads:64)
    in
    (t, p, host args, address out, prepare t [| filled t (ws, [||]) |])
  in
  Thumper.group "launch"
  @@ List.concat
       [
         [
           row "1" (launching 1) (fun (t, _, s) -> run t s);
           row "64" (launching 64) (fun (t, _, s) -> run t s);
         ];
         kfd_row "floor-1" floor_launching (fun w -> floor_launch w 1);
         kfd_row "floor-64" floor_launching (fun w -> floor_launch w 64);
         [
           row "1-segment" segment (fun (t, _, s) -> run t s);
           row "1-bar-args" bar (fun (t, _, args, out, s) ->
               set64 args out;
               run t s);
         ];
       ]

let copy_rows =
  let n = 256 * mib in
  let copy t ~dst ~src n =
    let buffer memory = B.create ~memory t.d n in
    part "COPY:0" (Copy { src = buffer src; dst = buffer dst })
  in
  let copying n (dst, src) () =
    let t = dev () in
    (t, prepare t [| copy t ~dst ~src n |])
  in
  let floor_copying n (dst, src) () =
    ignore (floor ());
    (buffer dst n, buffer src n)
  in
  let row_pair name n kinds floor_kinds =
    row name (copying n kinds) (fun (t, s) -> run t s)
    :: kfd_row ("floor-" ^ name) (floor_copying n floor_kinds)
         (fun (dst, src) -> floor_copy dst src n)
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
    let r = Option.get (A.alloc t.g `Pinned host_n) in
    write (host r) (String.make host_n '\001');
    (t, host r)
  in
  Thumper.group "copy"
    (row_pair "h2d-256MiB" n (B.Device, B.Pinned) (gpu_memory, system_memory)
    @ row_pair "d2h-256MiB" n (B.Pinned, B.Device) (system_memory, gpu_memory)
    @ row_pair "d2d-256MiB" n (B.Device, B.Device) (gpu_memory, gpu_memory)
    @ row_pair "16B" 16 (B.Device, B.Device) (gpu_memory, gpu_memory)
    @ [
        row "bidir-256MiB" both (fun (t, s) -> run t s);
        row "host-read-pinned" pinned (fun (_, p) -> read p host_n);
      ]
    @ kfd_row "floor-host-read-pinned"
        (fun () -> pages host_n)
        (fun p -> read p host_n))

let alloc_rows =
  let alloc name n =
    row name dev (fun t -> A.free t.g (Option.get (A.alloc t.g `Device n)))
    :: kfd_row ("floor-" ^ name) floor (fun _ -> floor_alloc gpu_memory n)
  in
  Thumper.group "alloc" (alloc "64KiB" (64 * kib) @ alloc "64MiB" (64 * mib))

let map_host_rows =
  let n = 256 * mib in
  Thumper.group "map-host"
    [
      row "256MiB"
        (fun () -> (dev (), pages n))
        (fun (t, p) -> A.free t.g (Option.get (A.map_host t.g p n)));
      row "floor-256MiB"
        (fun () ->
          ignore (floor ());
          pages n)
        (fun p -> floor_map_host p n);
    ]

let image_rows =
  Thumper.group "image"
    [
      (* The code object read, laid over device memory, ended and its memory
         freed. *)
      row "one-object" dev (fun t ->
          match A.image t.g (Lazy.force binary) with
          | Ok (`Place (n, lay)) ->
              let code = Option.get (A.alloc t.g `Device n) in
              A.unload t.g (fst (lay code));
              A.free t.g code
          | Ok (`Loaded _) -> failwith "code the device's library placed"
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
  let open_stop () = A.stop (get (Rig_amd_support.open_gpu ())) in
  Thumper.group "open"
    [
      row "partial" open_stop (fun () -> open_stop ());
      row "full" open_stop (fun () ->
          get (Rig_amd_pci.reset 0);
          open_stop ());
    ]

(* Whether GPU 0's device waits on other devices, asked in a child: this process
   opens no GPU, so that the workers it forks open theirs alone. The child stops
   the device before it ends, as it skips the exit's handlers. *)
let waits_on () =
  match Unix.fork () with
  | 0 ->
      let on =
        match Rig_amd_support.open_gpu () with
        | Ok g ->
            let on = A.waits_on g `Store in
            A.stop g;
            on
        | Error _ -> false
      in
      Unix._exit (if on then 0 else 1)
  | pid -> (
      match Unix.waitpid [] pid with _, Unix.WEXITED 0 -> true | _ -> false)

(* A full open, a reset included, takes over a second: the trials of the run
   with no kernel driver may take a minute. *)
let pci_deadline = 60.

let () =
  Rig_amd_support.hold_gpu ();
  if Rig_amd_support.gpus () > 0 then
    let config =
      if pci then Thumper.Config.(deadline pci_deadline default)
      else Thumper.Config.default
    in
    exit
    @@ Thumper.run ~config "rig_amd"
         ((if pci then [ open_rows ] else [])
         @ [ release_rows ]
         @ (if waits_on () then [ wait_rows ] else [])
         @ [ launch_rows; copy_rows; alloc_rows ]
         @ (if pci then [] else [ map_host_rows ])
         @ [ image_rows; wake_rows ])
