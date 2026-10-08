(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPU 0 through the driver, each row beside the packets that bound it, placed
   from C on a compute and a copy queue the bench makes through KFD: a release
   by a release packet that raises the interrupt a sleeping host wakes on, as
   the driver's does, a ring of the doorbell and a spin on a word; a queue
   switch by the same after a wait for the value before; foreign waits by 64-bit
   waits; launches by the same dispatch words; copies by the same SDMA copies;
   allocations and host mappings by the KFD calls behind them. The floors'
   packets are the ABI's, encoded once as the driver encodes its own, without
   the driver's cache acquire, slot words and host data path flush. A row waits
   by spinning on the word, except the wake row.

   Each case opens its device, or the floors' queues, in its own worker: KFD
   gives a process one address space per GPU, kept by the first render node that
   acquires it, so one process cannot hold both. Without an AMD GPU the suite
   has no rows. *)

module A = Device_amd
module P = Device_amd_amdgpu
module S = Device_amd_support
module Abi = Device_amd_abi
module Packet = Abi.Packet
module Pm4 = Abi.Pm4
module Sdma = Abi.Sdma

external start : int array -> unit = "device_amd_bench_start"
external interrupt : unit -> int = "device_amd_bench_interrupt"

external template : int -> string -> int array -> unit
  = "device_amd_bench_template"

external floor_release : int -> unit = "device_amd_bench_release"
external floor_release_agent : int -> unit = "device_amd_bench_release_agent"
external floor_switch : unit -> unit = "device_amd_bench_switch"
external floor_waits : nativeint -> int -> unit = "device_amd_bench_waits"
external floor_launch : string -> int -> unit = "device_amd_bench_launch"

external floor_copy : nativeint -> nativeint -> int -> unit
  = "device_amd_bench_copy"

external buffer : int -> int -> nativeint = "device_amd_bench_buffer"
external floor_alloc : int -> int -> unit = "device_amd_bench_alloc"
external floor_map_host : nativeint -> int -> unit = "device_amd_bench_map_host"
external pages : int -> nativeint = "device_amd_bench_pages"
external read : nativeint -> int -> int = "device_amd_bench_read"
external set64 : nativeint -> int -> unit = "device_amd_bench_set64" [@@noalloc]
external fill_entry : unit -> nativeint = "device_amd_bench_fill_entry"

external fill_arg :
  nativeint -> nativeint -> int -> string -> int array -> nativeint
  = "device_amd_bench_fill_arg"

let kib = 1024
let mib = 1024 * kib
let get = function Ok x -> x | Error why -> failwith why
let host r = Option.get (A.host r)
let address r = Option.get (A.address r)
let row name setup f = Thumper.bench_with_setup ~setup name f

(* The driver *)

type dev = { g : A.t; mutable v : int }

let dev () = { g = get (P.open_ 0); v = 0 }

let submit ?(waits = [||]) t ps =
  t.v <- t.v + 1;
  (match A.room t.g ps with
  | `Fits -> ()
  | `Later | `Never -> failwith "the parts do not fit");
  match A.submit t.g ~v:t.v ~waits ~handles:[||] ps with
  | `Ok -> ()
  | `Failed why -> failwith why

let wait t =
  while A.signaled t.g < t.v do
    Domain.cpu_relax ()
  done

let run ?waits t ps =
  submit ?waits t ps;
  wait t

(* Packets *)

let encode p = Packet.encode Int64.of_int p

let words p =
  let s = encode p in
  Array.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

(* A template's holes as the C reads them: each word's index, its width, the
   argument [arg v] of its value, then the operations on it in the order they
   apply. *)
let holes arg hs =
  let hole (at, w) =
    let rec flatten ops : _ Packet.term -> _ = function
      | Value v -> (arg v, ops)
      | Add (t, k) -> flatten ((0, Int64.to_int k) :: ops) t
      | Shift (t, n) -> flatten ((1, n) :: ops) t
      | Or (t, k) -> flatten ((2, Int64.to_int k) :: ops) t
    in
    let wide, t =
      match (w : _ Packet.word) with
      | W32 t -> (1, t)
      | W64 t -> (2, t)
      | Dword _ -> assert false
    in
    let a, ops = flatten [] t in
    [ at; wide; a; List.length ops ]
    @ List.concat_map (fun (op, k) -> [ op; k ]) ops
  in
  Array.of_list (List.concat_map hole hs)

(* Kernels *)

let binary =
  In_channel.with_open_bin "../test/fixtures/kernels_gfx1201.hsaco"
    In_channel.input_all

let code_object = get (Abi.Code_object.of_string binary)
let kernel name = Option.get (Abi.Code_object.kernel code_object name)

(* One run of [k] over one workgroup of [threads] work-items, its image at
   [base], its arguments at [args]. *)
let dispatch gpu (k : Abi.Code_object.kernel) ~base ~args ~threads =
  Pm4.run gpu
    (Pm4.dispatch gpu k ~program:(base + k.entry) ~scratch:0 ~args ~packet:0
       ~threads:(threads, 1, 1) ~groups:(1, 1, 1) ())

(* The code object loaded on the driver's device, its image copied in by the
   copy queue, and the address of its image. *)
let load t =
  match A.image t.g binary with
  | Ok (m, Some (code, bytes)) ->
      let n = String.length bytes in
      let staging = Option.get (A.alloc t.g `Pinned n) in
      S.write (host staging) bytes;
      run t
        [| A.part t.g ~queue:"COPY:0" (`Copy ((code, 0), (staging, 0), n)) |];
      let k = kernel "empty" in
      Option.get (A.entry m "empty") - k.descriptor
  | Ok (_, None) -> failwith "an image without code"
  | Error why -> failwith why

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

(* A compute queue's context save area as KFD sizes it (kfd_queue.c): each die's
   area and 32 bytes per wave for the debugger, the waves 32 per compute unit
   from GFX 10.1, before it 40 per compute unit up to 512 per shader engine. *)
let save_bytes (g : Abi.Gpu.t) ~cwsr =
  let round n a = (n + a - 1) / a * a in
  let waves =
    if compare g.target (10, 1, 0) < 0 then
      Int.min (g.compute_units * 40) (g.shader_engines * g.xccs * 512)
    else g.compute_units * 32
  in
  round ((cwsr + round (waves * 32) 64) * g.xccs) 4096

(* The floors' packets, in device_amd_bench_stubs.c's order, their values the
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
  let cwsr = prop "cwsr_size" in
  start
    [|
      prop "gpu_id";
      prop "drm_render_minor";
      cwsr;
      prop "ctl_stack_size";
      save_bytes gpu ~cwsr;
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
  let o = Abi.Code_object.elf code_object in
  let b = Bytes.make (Abi.Code_object.size code_object) '\000' in
  let put (s : Device_elf.section) =
    match s.offset with
    | Some off -> Bytes.blit_string o.file s.at b off s.length
    | None -> ()
  in
  Iarray.iter put o.sections;
  let patch (off, p) = Bytes.blit_string p 0 b off (String.length p) in
  List.iter patch (Abi.Code_object.patches code_object);
  let n = Bytes.length b in
  let code = buffer gpu_memory n and staging = buffer system_memory n in
  S.write staging (Bytes.unsafe_to_string b);
  floor_copy code staging n;
  Nativeint.to_int code

(* Rows *)

let release_rows =
  let switching () =
    let t = dev () in
    (t, [| A.part t.g ~queue:"COPY:0" (`Words [||]) |])
  in
  let mapped () =
    let t = dev () in
    (t, Option.get (A.alloc t.g `Mapped 4096))
  in
  Thumper.group "release"
    [
      row "driver" dev (fun t -> run t [||]);
      row "floor" floor (fun _ -> floor_release 1);
      row "floor-agent" floor (fun _ -> floor_release_agent 1);
      row "switch" switching (fun (t, copy) ->
          run t (if t.v land 1 = 0 then copy else [||]));
      row "floor-switch" floor (fun _ -> floor_switch ());
      row "no-wait-100" dev (fun t ->
          for _ = 1 to 100 do
            submit t [||]
          done;
          wait t);
      row "floor-no-wait-100" floor (fun _ -> floor_release 100);
      row "mapped" mapped (fun (t, _) -> run t [||]);
    ]

(* Waits on a word the host set, and a wait across 2^32: the host sets the word
   to 2^32 k - 1, the queue waits for 2^32 k, and the host stores it once a
   short spin saw the release held back. *)
let wait_rows =
  let waiting () =
    let t = dev () in
    let w = Option.get (A.alloc t.g `Pinned 8) in
    set64 (host w) 1;
    (t, Array.make 4 (`Word, address w, 1))
  in
  let floor_waiting () =
    ignore (floor_with ~waits:true ());
    let w = buffer system_memory 4096 in
    set64 w 1;
    w
  in
  let across () =
    let t = dev () in
    (t, Option.get (A.alloc t.g `Pinned 8))
  in
  Thumper.group "waits"
    [
      row "4" waiting (fun (t, waits) -> run ~waits t [||]);
      row "floor-4" floor_waiting (fun at -> floor_waits at 4);
      row "wait64" across (fun (t, w) ->
          let target = (t.v + 1) lsl 32 in
          set64 (host w) (target - 1);
          submit ~waits:[| (`Word, address w, target) |] t [||];
          for _ = 1 to 10_000 do
            if A.signaled t.g >= t.v then
              failwith "the wait passed before the host's store"
          done;
          set64 (host w) target;
          wait t);
    ]

let launch_rows =
  let empty t base =
    words
      (dispatch (A.capability t.g).gpu (kernel "empty") ~base ~args:0 ~threads:1)
  in
  let launching count () =
    let t = dev () in
    let base = load t in
    let ws = Array.concat (List.init count (fun _ -> empty t base)) in
    let f, arg = S.fill (A.capability t.g) ws ~bytes:0 in
    (t, [| A.part t.g ~queue:"COMPUTE:0" (`Fill (f, arg, Array.length ws, 0)) |])
  in
  let floor_launching () =
    let gpu = floor () in
    let base = floor_load () in
    encode (dispatch gpu (kernel "empty") ~base ~args:0 ~threads:1)
  in
  (* [double_index] over 64 work-items, its argument, the output's address,
     written for each launch: into the argument segment by a fill, or into
     [`Mapped] memory by the host, which the driver flushes into the GPU's
     memory before the doorbell. *)
  let segment () =
    let t = dev () in
    let base = load t in
    let c = A.capability t.g in
    let out = Option.get (A.alloc t.g `Device 256) in
    let k = kernel "double_index" in
    let known = function `Known n -> Some (Int64.of_int n) | `Args -> None in
    let p =
      Pm4.run c.gpu
        (Pm4.dispatch c.gpu k
           ~program:(`Known (base + k.entry))
           ~scratch:(`Known 0) ~args:`Args ~packet:(`Known 0)
           ~threads:(`Known 64, `Known 1, `Known 1)
           ~groups:(`Known 1, `Known 1, `Known 1)
           ())
    in
    let ws, hs = Packet.template known p in
    let arg =
      fill_arg c.place c.segment (address out) ws (holes (fun _ -> 0) hs)
    in
    let units = String.length ws / 4 in
    ( t,
      [| A.part t.g ~queue:"COMPUTE:0" (`Fill (fill_entry (), arg, units, 8)) |]
    )
  in
  let bar () =
    let t = dev () in
    let base = load t in
    let c = A.capability t.g in
    let out = Option.get (A.alloc t.g `Device 256) in
    let args = Option.get (A.alloc t.g `Mapped 8) in
    let ws =
      words
        (dispatch c.gpu (kernel "double_index") ~base ~args:(address args)
           ~threads:64)
    in
    let f, arg = S.fill c ws ~bytes:0 in
    let p =
      A.part t.g ~queue:"COMPUTE:0" (`Fill (f, arg, Array.length ws, 0))
    in
    (t, host args, address out, [| p |])
  in
  Thumper.group "launch"
    [
      row "1" (launching 1) (fun (t, ps) -> run t ps);
      row "64" (launching 64) (fun (t, ps) -> run t ps);
      row "floor-1" floor_launching (fun w -> floor_launch w 1);
      row "floor-64" floor_launching (fun w -> floor_launch w 64);
      row "1-segment" segment (fun (t, ps) -> run t ps);
      row "1-bar-args" bar (fun (t, args, out, ps) ->
          set64 args out;
          run t ps);
    ]

let copy_rows =
  let n = 256 * mib in
  let copying n (dst, src) () =
    let t = dev () in
    let alloc k = Option.get (A.alloc t.g k n) in
    let dst = alloc dst and src = alloc src in
    (t, [| A.part t.g ~queue:"COPY:0" (`Copy ((dst, 0), (src, 0), n)) |])
  in
  let floor_copying n (dst, src) () =
    ignore (floor ());
    (buffer dst n, buffer src n)
  in
  let copy name n kinds floor_kinds =
    [
      row name (copying n kinds) (fun (t, ps) -> run t ps);
      row ("floor-" ^ name) (floor_copying n floor_kinds) (fun (dst, src) ->
          floor_copy dst src n);
    ]
  in
  let both () =
    let t = dev () in
    let alloc k = Option.get (A.alloc t.g k n) in
    let part dst src =
      A.part t.g ~queue:"COPY:0" (`Copy ((alloc dst, 0), (alloc src, 0), n))
    in
    (t, [| part `Device `Pinned; part `Pinned `Device |])
  in
  let host_n = 64 * mib in
  let pinned () =
    let t = dev () in
    let r = Option.get (A.alloc t.g `Pinned host_n) in
    S.write (host r) (String.make host_n '\001');
    (t, host r)
  in
  Thumper.group "copy"
    (copy "h2d-256MiB" n (`Device, `Pinned) (gpu_memory, system_memory)
    @ copy "d2h-256MiB" n (`Pinned, `Device) (system_memory, gpu_memory)
    @ copy "d2d-256MiB" n (`Device, `Device) (gpu_memory, gpu_memory)
    @ copy "16B" 16 (`Device, `Device) (gpu_memory, gpu_memory)
    @ [
        row "bidir-256MiB" both (fun (t, ps) -> run t ps);
        row "host-read-pinned" pinned (fun (_, p) -> read p host_n);
        row "floor-host-read-pinned"
          (fun () -> pages host_n)
          (fun p -> read p host_n);
      ])

let alloc_rows =
  let alloc name n =
    [
      row name dev (fun t -> A.free t.g (Option.get (A.alloc t.g `Device n)));
      row ("floor-" ^ name) floor (fun _ -> floor_alloc gpu_memory n);
    ]
  in
  Thumper.group "alloc" (alloc "64KiB" (64 * kib) @ alloc "64MiB" (64 * mib))

let map_host_rows =
  let n = 256 * mib in
  Thumper.group "map-host"
    [
      row "256MiB"
        (fun () -> (dev (), pages n))
        (fun (t, p) -> A.unmap t.g (Option.get (A.map_host t.g p n)));
      row "floor-256MiB"
        (fun () ->
          ignore (floor ());
          pages n)
        (fun p -> floor_map_host p n);
    ]

let image_rows =
  Thumper.group "image"
    [
      row "one-object" dev (fun t ->
          match A.image t.g binary with
          | Ok (m, _) -> A.unload t.g m
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
  Thumper.group "wake"
    [
      row "driver" dev (fun t ->
          submit t [||];
          sleep t);
    ]

(* Whether GPU 0's device waits on other devices, asked in a child: this process
   opens no GPU, so that the workers it forks open theirs alone. *)
let waits_on () =
  match Unix.fork () with
  | 0 ->
      let on =
        match P.open_ 0 with Ok g -> A.waits_on g `Store | Error _ -> false
      in
      Unix._exit (if on then 0 else 1)
  | pid -> (
      match Unix.waitpid [] pid with _, Unix.WEXITED 0 -> true | _ -> false)

let () =
  if P.count () > 0 then
    exit
    @@ Thumper.run "device_amd"
         ([ release_rows ]
         @ (if waits_on () then [ wait_rows ] else [])
         @ [
             launch_rows;
             copy_rows;
             alloc_rows;
             map_host_rows;
             image_rows;
             wake_rows;
           ])
