(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The path *)

let pci_firmware =
  Option.map (String.split_on_char ':') (Sys.getenv_opt "RIG_AMD_PCI_FIRMWARE")

let driverless () = Option.is_some pci_firmware

let gpus () =
  match pci_firmware with
  | Some _ -> Rig_amd_pci.count ()
  | None -> Rig_amd_amdgpu.count ()

let gpu () =
  match pci_firmware with
  | Some firmware -> Rig_amd_pci.gpu ~firmware 0
  | None -> Rig_amd_amdgpu.gpu 0

let open_gpu () = (gpu ()).make ()

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

let words_part ws =
  let b = Bigarray.(Array1.create int32 c_layout (Array.length ws)) in
  Array.iteri (fun i w -> b.{i} <- Int32.of_int w) ws;
  Rig.Submission.words (Rig.Buffer.of_bigarray b)

(* Conformance *)

let fixture f =
  In_channel.with_open_bin ("../amd/fixtures/" ^ f) In_channel.input_all

let kernels_bin () = fixture "kernels_gfx1201.hsaco"
let work_bin () = fixture "work_gfx1201.hsaco"
let binary () = (kernels_bin (), [ "empty"; "double_index"; "spin"; "wild" ])
let second () = if driverless () then None else Some (open_gpu ())

(* Each fixture, loaded once on each device. *)
let kernels = Rig_gpu_support.loader kernels_bin
let work = Rig_gpu_support.loader work_bin

(* [name] of the fixture [load] loads over one group of 64 work-items, with
   [params] bytes of parameters [store] stores. *)
let launching t load name ~params uses store =
  Rig_gpu_support.launching (load t.d) name ~params ~threads:64 uses store

(* work.cl's [copy dst src n delay] copies [n] words. *)
let copy_words t ~dst ~src =
  let module Run = Rig.Submission.Run in
  let w =
    launching t work "copy" ~params:24
      [| (dst, Rig.Read_write, 0); (src, Rig.Read, 8) |]
      (fun run b ->
        Run.int32 run b 16 (Rig.Buffer.length src / 4);
        Run.int32 run b 20 0)
  in
  w

(* rig_amd.mli, through KFD: "5 us once a measure within the bound is taken,
   plus the line's rate error over the time since the latest measure, at
   most 62.5 ms. That rate error is at most 10 us over the measures'
   distance, 10 ppm, and the clock's own until they are a second apart."
   The clock's own is taken within the same 10 ppm. With no kernel driver
   the device measures at open only, by a round trip through its compute
   queue, taken as within 10 us as CUDA's is, plus the clock's rate error
   since open. *)
let clock_error ~since =
  if driverless () then 5_000 + (since / 100_000) else 5_000 + 625

(* kernels.cl's [spin flag n] sleeps [n] times 127 x 64 cycles, at most 2.1 us
   at 4 GHz, then sets its flag, a word of its own here. *)
let spin t ~ns =
  let module Run = Rig.Submission.Run in
  let flag = Rig_gpu_support.arguments t.d (String.make 8 '\000') in
  let w =
    launching t kernels "spin" ~params:12
      [| (flag, Rig.Read_write, 0) |]
      (fun run b -> Run.int32 run b 8 ((ns / 2000) + 1))
  in
  w

let launch_binary () = Some (fixture "launch_gfx1201.hsaco")

(* The C entries *)

module Edge = struct
  (* The C representation: queue, kind, launch entry, code address, block
     offset, parameter count, copy destination, source and bytes, counts of
     after indices and words, then those indices and words. *)
  type part = { ints : int array; block : string }

  let index = function
    | "COMPUTE:0" -> 0
    | "COPY:0" -> 1
    | q -> invalid_arg ("Rig_amd_support.Edge: queue " ^ q)

  let rig_words = 1
  let rig_fill = 2
  let rig_copy = 3
  let rig_launch = 4

  (* The ints' indices used to place a part's parameter block. *)
  let block_at = 4
  let block_align = 16

  let make ~queue ~kind ~after ?(launch = 0n) ?(code = 0) ?(params = 0)
      ?(dst = 0) ?(src = 0) ?(copy = 0) words =
    let head =
      [|
        queue;
        kind;
        Nativeint.to_int launch;
        code;
        0;
        params;
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
      block = "";
    }

  let words ~queue ?(after = [||]) ws =
    make ~queue:(index queue) ~kind:rig_words ~after ws

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
      make ~queue:0 ~kind:rig_launch ~after ~launch:e.launch ~code:e.code
        ~params:(String.length params) words
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
        ints.(block_at) <- Buffer.length b;
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
    | `Fill -> make ~queue ~kind:rig_fill ~after [||]
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
    match r with None -> `Ok | Some why -> `Failed why
end
