(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type arg =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external fill_arg :
  nativeint -> nativeint -> int array -> int -> int -> int -> arg
  = "rig_amd_test_fill_arg_byte" "rig_amd_test_fill_arg"

external fill_entry : unit -> nativeint = "rig_amd_test_fill_entry"
external fill_address : arg -> int = "rig_amd_test_fill_address"
external data : arg -> int = "rig_amd_test_data"

(* The path *)

let pci_firmware =
  Option.map (String.split_on_char ':') (Sys.getenv_opt "RIG_AMD_PCI_FIRMWARE")

let driverless () = Option.is_some pci_firmware

let gpus () =
  match pci_firmware with
  | Some _ -> Rig_amd_pci.count ()
  | None -> Rig_amd_amdgpu.count ()

let open_gpu () =
  match pci_firmware with
  | Some firmware -> Rig_amd_pci.open_ ~firmware 0
  | None -> Rig_amd_amdgpu.open_ 0

module D = Rig_amd

include Rig_gpu_support.Make (struct
  module D = Rig_amd

  let class_ = "AMD"
  let present () = gpus () > 0
  let open_ = open_gpu
end)

let reached g v =
  let rec loop () =
    let seen = Rig_amd.signaled g in
    if seen < v then begin
      Rig_amd.sleep g ~seen ~still_ms:200;
      loop ()
    end
  in
  loop ()

(* Fills *)

type fill = { entry : nativeint; arg : arg }

let fill ?(code = 0) ?(split = 0) (c : Rig_amd_abi.Capability.t) ws ~bytes =
  {
    entry = fill_entry ();
    arg = fill_arg c.place c.segment ws split bytes code;
  }

let fill_address f = fill_address f.arg

let fill_part ~queue ?(after = [||]) f ~units ~bytes =
  {
    Rig.Submission.queue;
    after;
    work =
      Fill
        {
          fill = f.entry;
          arg = Rig.Buffer.of_bigarray f.arg;
          ring_units = units;
          segment_bytes = bytes;
        };
  }

let words_part ~queue ?(after = [||]) ws =
  let b = Bigarray.(Array1.create int32 c_layout (Array.length ws)) in
  Array.iteri (fun i w -> b.{i} <- Int32.of_int w) ws;
  { Rig.Submission.queue; after; work = Words (Rig.Buffer.of_bigarray b) }

(* Conformance *)

let fixture f = In_channel.with_open_bin ("../amd/fixtures/" ^ f) In_channel.input_all
let kernels_bin () = fixture "kernels_gfx1201.hsaco"
let work_bin () = fixture "work_gfx1201.hsaco"
let binary () = (kernels_bin (), [ "empty"; "double_index"; "spin"; "wild" ])
let second () = if driverless () then None else Some (open_gpu ())

(* Each fixture, loaded once on each device. *)
let kernels = Rig_gpu_support.loader kernels_bin
let work = Rig_gpu_support.loader work_bin

(* [name] of the fixture [load] loads, on COMPUTE:0 over one group of 64
   work-items, with [params] bytes of parameters [store] stores. *)
let launching t load name ~params store =
  let module Run = Rig.Submission.Run in
  let part =
    {
      Rig.Submission.queue = "COMPUTE:0";
      after = [||];
      work = Launch { image = load t.d; kernel = name; params; refs = [||] };
    }
  in
  let block run b =
    Run.groups run b 1 1 1;
    Run.threads run b 64 1 1;
    store run b
  in
  { Rig_gpu_support.part; block }

(* work.cl's [copy dst src n delay] copies [n] words. *)
let copy_words t ~dst ~src =
  let module Run = Rig.Submission.Run in
  let w =
    launching t work "copy" ~params:24 (fun run b ->
        Run.int64 run b 0 (Rig.Buffer.address dst);
        Run.int64 run b 8 (Rig.Buffer.address src);
        Run.int32 run b 16 (Rig.Buffer.length src / 4);
        Run.int32 run b 20 0)
  in
  (w, src)

(* kernels.cl's [spin flag n] sleeps [n] times 127 x 64 cycles, at most
   2.1 us at 4 GHz, then sets its flag, a word of its own here. *)
let spin t ~ns =
  let module Run = Rig.Submission.Run in
  let flag = Rig_gpu_support.arguments t.d (String.make 8 '\000') in
  let w =
    launching t kernels "spin" ~params:12 (fun run b ->
        Run.int64 run b 0 (Rig.Buffer.address flag);
        Run.int32 run b 8 ((ns / 2000) + 1))
  in
  (w, flag)

let launch_binary () = Some (fixture "launch_gfx1201.hsaco")

(* The C entries *)

module Edge = struct
  (* The ints the C side reads: queue, kind (rig_edge.h's, 0 for none), fill,
     argument, ring units, segment bytes, copy destination, source and bytes,
     the counts of [after] indices and of words, the indices, the words. A
     launch's block, its grid, groups, shared memory and parameters
     (rig_edge.h's struct rig_block), goes in the args, at the offset its ring
     units take when the part is handed over. *)
  type part = { ints : int array; keep : arg option; block : string }

  let index = function
    | "COMPUTE:0" -> 0
    | "COPY:0" -> 1
    | q -> invalid_arg ("Rig_amd_support.Edge: queue " ^ q)

  let rig_words = 1
  let rig_fill = 2
  let rig_copy = 3
  let rig_launch = 4

  (* The ints' indices of a launch's block offset, and of its words. *)
  let units_at = 4
  let block_align = 16

  let make ?keep ~queue ~kind ~after ?(fill = 0n) ?(arg = 0) ?(units = 0)
      ?(bytes = 0) ?(dst = 0) ?(src = 0) ?(copy = 0) words =
    let head =
      [|
        queue;
        kind;
        Nativeint.to_int fill;
        arg;
        units;
        bytes;
        dst;
        src;
        copy;
        Array.length after;
        Array.length words;
      |]
    in
    {
      ints =
        Array.concat
          [ head; after; Array.map (fun w -> w land 0xffff_ffff) words ];
      keep;
      block = "";
    }

  let words ~queue ?(after = [||]) ws =
    make ~queue:(index queue) ~kind:rig_words ~after ws

  let fill ~queue ?(after = [||]) f ~units ~bytes =
    make ~keep:f.arg ~queue:(index queue) ~kind:rig_fill ~after ~fill:f.entry
      ~arg:(data f.arg) ~units ~bytes [||]

  let copy ?(after = [||]) ~dst ~src n =
    make ~queue:1 ~kind:rig_copy ~after ~dst ~src ~copy:n [||]

  let launch ?(after = [||]) (e : Rig_edge.entry) ~groups:(gx, gy, gz)
      ~threads:(tx, ty, tz) ?(shared = 0) params refs =
    let header = Bytes.create 32 in
    List.iteri
      (fun i v -> Bytes.set_int32_le header (4 * i) (Int32.of_int v))
      [ gx; gy; gz; tx; ty; tz; shared; 0 ];
    let words =
      Array.concat (List.map (fun (at, slot) -> [| at; slot |]) refs)
    in
    let p =
      make ~queue:0 ~kind:rig_launch ~after ~fill:e.launch ~arg:e.code
        ~bytes:(String.length params) words
    in
    { p with block = Bytes.to_string header ^ params }

  (* The parts' ints with each launch's block offset, and the args. *)
  let args ps =
    let b = Buffer.create 256 in
    let place p =
      if p.block = "" then p.ints
      else begin
        while Buffer.length b mod block_align <> 0 do
          Buffer.add_char b '\000'
        done;
        let ints = Array.copy p.ints in
        ints.(units_at) <- Buffer.length b;
        Buffer.add_string b p.block;
        ints
      end
    in
    let ints = Array.map place ps in
    (ints, Buffer.contents b)

  let raw ~queue ?(work = `None) ?(after = [||]) () =
    match work with
    | `None -> make ~queue ~kind:0 ~after [||]
    | `Words n -> make ~queue ~kind:rig_words ~after (Array.make n 0)
    | `Fill -> make ~queue ~kind:rig_fill ~after ~fill:(fill_entry ()) [||]
    | `Copy n -> make ~queue ~kind:rig_copy ~after ~copy:n [||]

  external room_c : nativeint -> int array array -> string -> int
    = "rig_amd_test_room"

  external submit_c :
    nativeint ->
    int ->
    int array ->
    int array array ->
    string ->
    int array ->
    string option = "rig_amd_test_submit_byte" "rig_amd_test_submit"

  let room g ps =
    let ints, args = args ps in
    match room_c (Rig_amd.facts g).edge ints args with
    | 0 -> `Fits
    | 1 -> `Later
    | _ -> `Never

  let submit g ~v ?(waits = [||]) ?(slots = [||]) ps =
    let w =
      Array.concat (Array.to_list (Array.map (fun (a, x) -> [| a; x |]) waits))
    in
    let ints, args = args ps in
    let r = submit_c (Rig_amd.facts g).edge v w ints args slots in
    (* The fills' arguments lived through the call. *)
    Array.iter (fun p -> ignore (Sys.opaque_identity p.keep)) ps;
    match r with None -> `Ok | Some why -> `Failed why
end
