(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs

let strf = Printf.sprintf

type version = Device_amd_abi.Gpu.version

type gc = {
  engines : int;
  arrays : int;
  units : int;
  scratch_slots : int;
  waves : int;
  lds : int;
}

type t = {
  versions : (int * version) list;
  bases : (int * (int * int array) list) list;
  harvested : (int * int list) list;
  gc : gc;
}

let offset = D.discovery_tmr_offset
let bytes = D.discovery_tmr_size

(* A defect of the table, with the field and its byte offset. *)
exception Bad of string

let bad fmt = Printf.ksprintf (fun s -> raise (Bad s)) fmt

(* Reading *)

(* [get s what at (off, n)] is the little-endian unsigned field of [n] bytes at
   [at + off] of [s]. *)
let get s what at (off, n) =
  let i = at + off in
  if i < 0 || i + n > String.length s then
    bad "%s at byte %d lies outside the table's %d bytes" what i
      (String.length s);
  match n with
  | 1 -> String.get_uint8 s i
  | 2 -> String.get_uint16_le s i
  | 4 -> Int32.to_int (String.get_int32_le s i) land 0xffff_ffff
  | 8 -> Int64.to_int (String.get_int64_le s i)
  | n -> invalid_arg (strf "Discovery: a field of %d bytes" n)

(* [bit s what at (bit, 1)] is the bit field at bit [bit] from [at]. *)
let bit s what at (b, width) =
  (get s what (at + (b / 8)) (0, 1) lsr (b mod 8)) land ((1 lsl width) - 1)

(* The byte sum of the [n] bytes at [at], as the kernel's 16-bit checksum. *)
let checksum s what at n =
  if at < 0 || n < 0 || at + n > String.length s then
    bad "%s's %d bytes at byte %d lie outside the table's %d bytes" what n at
      (String.length s);
  let sum = ref 0 in
  for i = at to at + n - 1 do
    sum := !sum + Char.code (String.unsafe_get s i)
  done;
  !sum land 0xffff

let expect s what at field want =
  let got = get s what at field in
  if got <> want then
    bad "%s at byte %d is 0x%x, expected 0x%x" what (at + fst field) got want

let check_sum s what at n ~expected =
  let got = checksum s what at n in
  if got <> expected then
    bad "%s's checksum is 0x%04x, expected 0x%04x" what got expected

(* Tables *)

(* The offset of table [i] of the binary header, 0 if absent. *)
let table s i =
  let off, stride = D.Binary_header.table_list in
  let at = off + (i * stride) in
  ( get s "a table's offset" at D.Table_info.offset,
    get s "a table's checksum" at D.Table_info.checksum )

let binary s =
  expect s "the binary signature" 0 D.Binary_header.binary_signature
    D.binary_signature;
  let off, n = D.Binary_header.binary_checksum in
  let from = off + n in
  let size = get s "the binary size" 0 D.Binary_header.binary_size in
  let expected =
    get s "the binary checksum" 0 D.Binary_header.binary_checksum
  in
  check_sum s "the binary" from (size - from) ~expected

(* The width in bytes of the IPs' base addresses. *)
let base_width s ih =
  let version =
    get s "the IP table's version" ih D.Ip_discovery_header.version
  in
  if
    version >= 4
    && bit s "the IP table's flags" ih D.Ip_discovery_header.base_addr_64_bit
       = 1
  then 8
  else 4

(* A base address of 64 bits keeps its low 30 bits, a word address, as the
   kernel's [amdgpu_discovery_reg_base_init] does. *)
let word_base = 0x3fff_ffff

type ip = {
  hw_id : int;
  instance : int;
  version : version;
  segments : int array;
}

let ips s ih =
  let dies = get s "the number of dies" ih D.Ip_discovery_header.num_dies in
  let die_off, die_stride = D.Ip_discovery_header.die_info in
  let max_dies = (D.Ip_discovery_header.sizeof - die_off) / die_stride in
  if dies > max_dies then
    bad "the table lists %d dies, at most %d" dies max_dies;
  let width = base_width s ih in
  let out = ref [] in
  for die = 0 to dies - 1 do
    let info = ih + die_off + (die * die_stride) in
    let at = get s "a die's offset" info D.Die_info.die_offset in
    let id = get s "a die's ID" at D.Die_header.die_id in
    if id <> die then bad "die %d at byte %d has ID %d" die at id;
    let n = get s "a die's number of IPs" at D.Die_header.num_ips in
    let ip = ref (at + D.Die_header.sizeof) in
    for _ = 1 to n do
      let f what field = get s what !ip field in
      let count = f "an IP's number of bases" D.Ip_v4.num_base_address in
      let base i =
        get s "an IP's base" (!ip + D.Ip_v4.sizeof + (i * width)) (0, width)
      in
      let segments =
        Array.init count (fun i ->
            if width = 8 then base i land word_base else base i)
      in
      out :=
        {
          hw_id = f "an IP's hardware ID" D.Ip_v4.hw_id;
          instance = f "an IP's instance" D.Ip_v4.instance_number;
          version =
            ( f "an IP's major version" D.Ip_v4.major,
              f "an IP's minor version" D.Ip_v4.minor,
              f "an IP's revision" D.Ip_v4.revision );
          segments;
        }
        :: !out;
      ip := !ip + D.Ip_v4.sizeof + (width * count)
    done
  done;
  List.rev !out

(* Version 1 describes a GC by work-group processors, two compute units each;
   version 2 by compute units. *)
let gc s at =
  let f what field = get s what at field in
  match f "the GC table's version" D.Gpu_info_header.version_major with
  | 1 ->
      let open D.Gc_info_v1_0 in
      {
        engines = f "the GC's shader engines" gc_num_se;
        arrays = f "the GC's shader arrays" gc_num_sa_per_se;
        units =
          2
          * (f "the GC's WGPs" gc_num_wgp0_per_sa
            + f "the GC's WGPs" gc_num_wgp1_per_sa);
        scratch_slots = f "the GC's scratch slots" gc_max_scratch_slots_per_cu;
        waves = f "the GC's waves" gc_max_waves_per_simd;
        lds = 1024 * f "the GC's local data share" gc_lds_size;
      }
  | 2 ->
      let open D.Gc_info_v2_0 in
      {
        engines = f "the GC's shader engines" gc_num_se;
        arrays = f "the GC's shader arrays" gc_num_sh_per_se;
        units = f "the GC's compute units" gc_num_cu_per_sh;
        scratch_slots = f "the GC's scratch slots" gc_max_scratch_slots_per_cu;
        waves = f "the GC's waves" gc_max_waves_per_simd;
        lds = 1024 * f "the GC's local data share" gc_lds_size;
      }
  | v -> bad "the GC table at byte %d is of version %d, expected 1 or 2" at v

(* The harvest table's entries, up to the first of hardware ID 0. *)
let harvest s at =
  let off, stride = D.Harvest_table.list in
  let n = (D.Harvest_table.sizeof - off) / stride in
  let rec go i acc =
    if i = n then List.rev acc
    else
      let e = at + off + (i * stride) in
      let id = get s "a harvested block" e D.Harvest_info.hw_id in
      if id = 0 then List.rev acc
      else
        let inst =
          get s "a harvested instance" e D.Harvest_info.number_instance
        in
        go (i + 1) ((id, inst) :: acc)
  in
  go 0 []

(* [update k f l] sets [k]'s value in the association list [l] to [f] of its
   current one. *)
let update k f l = (k, f (List.assoc_opt k l)) :: List.remove_assoc k l
let by_key (a, _) (b, _) = compare a b
let sorted l = List.sort by_key l

let parse s =
  binary s;
  let ih, ih_sum = table s D.ip_discovery_table in
  if ih = 0 then bad "the binary lists no IP table";
  expect s "the IP table's signature" ih D.Ip_discovery_header.signature
    D.discovery_table_signature;
  check_sum s "the IP table" ih
    (get s "the IP table's size" ih D.Ip_discovery_header.size)
    ~expected:ih_sum;
  let at, gc_sum = table s D.gc_table in
  if at = 0 then bad "the binary lists no GC table";
  expect s "the GC table's ID" at D.Gpu_info_header.table_id D.gc_table_id;
  check_sum s "the GC table" at
    (get s "the GC table's size" at D.Gpu_info_header.size)
    ~expected:gc_sum;
  let gc = gc s at in
  let harvested =
    match table s D.harvest_info_table with
    | 0, _ -> []
    | hv, sum ->
        expect s "the harvest table's signature" hv
          D.Harvest_info_header.signature D.harvest_table_signature;
        check_sum s "the harvest table" hv D.Harvest_table.sizeof ~expected:sum;
        List.fold_left
          (fun acc (id, i) ->
            update id (fun l -> i :: Option.value ~default:[] l) acc)
          [] (harvest s hv)
        |> List.map (fun (id, l) -> (id, List.sort compare l))
        |> sorted
  in
  let ips = ips s ih in
  (* A later instance of a block replaces an earlier one, as in the kernel's
     register bases. *)
  let bases =
    List.fold_left
      (fun acc ip ->
        update ip.hw_id
          (fun insts ->
            let insts = Option.value ~default:[] insts in
            (ip.instance, ip.segments) :: List.remove_assoc ip.instance insts)
          acc)
      [] ips
    |> List.map (fun (id, insts) -> (id, sorted insts))
    |> sorted
  in
  let versions =
    List.fold_left
      (fun acc ip ->
        update ip.hw_id
          (function
            | Some (i, v) when i <= ip.instance -> (i, v)
            | _ -> (ip.instance, ip.version))
          acc)
      [] ips
    |> List.map (fun (id, (_, v)) -> (id, v))
    |> sorted
  in
  { versions; bases; harvested; gc }

let of_string s =
  match parse s with d -> Ok d | exception Bad why -> Error why

let version d b = List.assoc_opt b d.versions

let live d b =
  let fused = Option.value ~default:[] (List.assoc_opt b d.harvested) in
  List.filter
    (fun (i, _) -> not (List.mem i fused))
    (Option.value ~default:[] (List.assoc_opt b d.bases))

let name = D.hwid_name
