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

let fill ?(code = 0) ?(split = 0) (c : Rig_amd.capability) ws ~bytes =
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

(* The C entries *)

module Edge = struct
  (* The ints the C side reads: queue, fill, argument, ring units, segment
     bytes, copy destination, source and bytes, the counts of [after] indices
     and of words, the indices, the words. *)
  type part = { ints : int array; keep : arg option }

  let index = function
    | "COMPUTE:0" -> 0
    | "COPY:0" -> 1
    | q -> invalid_arg ("Rig_amd_support.Edge: queue " ^ q)

  let make ?keep ~queue ~after ?(fill = 0n) ?(arg = 0) ?(units = 0) ?(bytes = 0)
      ?(dst = 0) ?(src = 0) ?(copy = 0) words =
    let head =
      [|
        queue;
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
    }

  let words ~queue ?(after = [||]) ws = make ~queue:(index queue) ~after ws

  let fill ~queue ?(after = [||]) f ~units ~bytes =
    make ~keep:f.arg ~queue:(index queue) ~after ~fill:f.entry ~arg:(data f.arg)
      ~units ~bytes [||]

  let copy ?(after = [||]) ~dst ~src n =
    make ~queue:1 ~after ~dst ~src ~copy:n [||]

  let raw ~queue ?(words = 0) ?(fill = false) ?(copy = 0) ?(after = [||]) () =
    let fill = if fill then fill_entry () else 0n in
    make ~queue ~after ~fill ~copy (Array.make words 0)

  external room_c : nativeint -> nativeint -> int array array -> int
    = "rig_amd_test_room"

  external submit_c :
    nativeint ->
    nativeint ->
    int ->
    int array ->
    int array array ->
    string option = "rig_amd_test_submit"

  let room g ps =
    match
      room_c Rig_amd.room_entry (Rig_amd.self g)
        (Array.map (fun p -> p.ints) ps)
    with
    | 0 -> `Fits
    | 1 -> `Later
    | _ -> `Never

  let submit g ~v ?(waits = [||]) ps =
    let w =
      Array.concat (Array.to_list (Array.map (fun (a, x) -> [| a; x |]) waits))
    in
    let r =
      submit_c Rig_amd.submit_entry (Rig_amd.self g) v w
        (Array.map (fun p -> p.ints) ps)
    in
    (* The fills' arguments lived through the call. *)
    Array.iter (fun p -> ignore (Sys.opaque_identity p.keep)) ps;
    match r with None -> `Ok | Some why -> `Failed why
end
