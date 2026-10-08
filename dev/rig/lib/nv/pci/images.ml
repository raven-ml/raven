(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let ( let* ) = Result.bind

type range = { contents : string; at : int; length : int }
type bootloader = { image : range; code : int; data : int; manifest : int }

type booter = {
  image : string;
  code : int * int;
  data : int * int;
  pkc : int;
  engines : int;
  ucode : int;
}

type fmc = { fmc : range; hash : range; signature : range; public_key : range }

type t = {
  gsp : range;
  signature : range;
  bootloader : bootloader;
  start : [ `Booter of booter | `Fmc of fmc ];
}

let names : Chip.family -> string list = function
  | Ampere -> Defs.firmware_ampere
  | Ada -> Defs.firmware_ada
  | Blackwell -> Defs.firmware_blackwell

let pinned = Defs.pinned
let origin = Defs.origin

(* Reading containers *)

(* A container whose fields point outside it is refused with [Short], one that
   holds what no firmware does with [Bad]. *)
exception Short
exception Bad of string

let u32 s off =
  if off < 0 || off + 4 > String.length s then raise Short
  else Int32.to_int (String.get_int32_le s off) land 0xffff_ffff

let field s base (off, _) = u32 s (base + off)

let sub s at length =
  if at < 0 || length < 0 || at + length > String.length s then raise Short
  else { contents = s; at; length }

let read what f s =
  match f s with
  | v -> Ok v
  | exception Short -> Error (strf "the %s points outside its file" what)
  | exception Bad why -> Error (strf "the %s %s" what why)

(* The data of a firmware container (nvfw_bin_hdr) and its header's offset. *)
let container s =
  let module B = Defs.Bin_header in
  ( sub s (field s 0 B.data_offset) (field s 0 B.data_size),
    field s 0 B.header_offset )

let bootloader =
  read "bootloader's container" @@ fun s ->
  let module U = Defs.Riscv_ucode_desc in
  let image, desc = container s in
  {
    image;
    code = field s desc U.monitor_code_offset;
    data = field s desc U.monitor_data_offset;
    manifest = field s desc U.manifest_offset;
  }

(* The booter: its container's data with the production signature written at the
   place its heavy-secure header names, its code the first application of its
   load header (nouveau's nvfw/hs.h, v2 headers). *)
let booter =
  read "booter's container" @@ fun s ->
  let module H = Defs.Hs_header in
  let module L = Defs.Hs_load_header in
  let data, hs = container s in
  let load = field s hs H.header_offset in
  (* These fields hold the offset of the word that holds the value. *)
  let indirect f = u32 s (field s hs f) in
  let patch = indirect H.patch_loc in
  let signatures = indirect H.num_sig in
  if signatures = 0 then raise (Bad "holds no signature");
  let length = field s hs H.sig_prod_size / signatures in
  let signature =
    sub s (field s hs H.sig_prod_offset + indirect H.patch_sig) length
  in
  if patch < 0 || patch + length > data.length then raise Short;
  let image = Bytes.of_string (String.sub s data.at data.length) in
  Bytes.blit_string s signature.at image patch length;
  let app, _, _ = L.app in
  let data_offset = field s load L.os_data_offset in
  (* The patch metadata: the fuse version, engines and ucode ID, words. *)
  let meta = field s hs H.meta_data_offset in
  {
    image = Bytes.unsafe_to_string image;
    code = (u32 s (load + app), u32 s (load + app + 4));
    data = (data_offset, field s load L.os_data_size);
    pkc = patch - data_offset;
    engines = u32 s (meta + 4);
    ucode = u32 s (meta + 8);
  }

(* ELF objects *)

let sections what s =
  match Rig_elf.of_string ~held:(fun _ -> false) s with
  | Error why -> Error (strf "the %s is no ELF object: %s" what why)
  | Ok o ->
      let section name =
        let is (x : Rig_elf.section) = x.name = name in
        match Iarray.find_opt is o.sections with
        | Some x -> Ok { contents = s; at = x.at; length = x.length }
        | None -> Error (strf "the %s has no section %s" what name)
      in
      Ok section

let family_suffix : Chip.family -> string = function
  | Ampere -> "ga10x"
  | Ada -> "ad10x"
  | Blackwell -> "gb20x"

let gsp family s =
  let* section = sections "GSP's firmware" s in
  let* image = section ".fwimage" in
  let* signature = section (".fwsignature_" ^ family_suffix family) in
  Ok (image, signature)

let fmc s =
  let* section = sections "FMC" s in
  let* fmc = section "image" in
  let* hash = section "hash" in
  let* signature = section "signature" in
  let* public_key = section "publickey" in
  Ok { fmc; hash; signature; public_key }

(* Reading the files *)

let find dirs name =
  Rig_pci.Firmware.find dirs name ~digest:(List.assoc name pinned)

let read family dirs =
  match names family with
  | [ gsp_name; bootloader_name; start_name ] ->
      let* g = find dirs gsp_name in
      let* b = find dirs bootloader_name in
      let* st = find dirs start_name in
      let* gsp, signature = gsp family g in
      let* bootloader = bootloader b in
      let* start =
        match family with
        | Blackwell -> Result.map (fun f -> `Fmc f) (fmc st)
        | Ampere | Ada -> Result.map (fun b -> `Booter b) (booter st)
      in
      Ok { gsp; signature; bootloader; start }
  | _ -> assert false (* Defs lists three images per family *)
