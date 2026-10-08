(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reading the objects a program load reads: AMD code objects of 16 and 128
   kernels (3 symbols each, their symbol tables stripped, as a library ships its
   kernels), a cubin as the NV loader reads it, and a host object of six
   kernels. Then finding a kernel's descriptor in the code object of 128, as the
   AMD loader does for each kernel it loads: the last one, behind every other
   symbol.

   Two floors bound [amd-128-kernels]. [floor-bytes] reads every field the
   reader decodes and builds nothing: each section header, each symbol with its
   name up to its NUL, each relocation. [floor-symbols] builds only the symbols
   the result holds, from lengths known beforehand. An array of more than 256
   words is allocated in the major heap, so the next minor collection promotes
   every symbol a table of more than 256 holds: from there on, a read costs that
   promotion too. *)

module Elf = Rig_elf

let strf = Printf.sprintf

(* The suite's fixtures. ../test/fixtures/README.md says how each is made. *)

let read path =
  In_channel.with_open_bin
    (Filename.concat "../test/fixtures" path)
    In_channel.input_all

let code_object k = read ("amd_" ^ string_of_int k ^ "_gfx1100.hsaco")
let cubin = read "simple_add_sm89.cubin"
let host = read "host_aarch64.o"

let of_string ?align obj () =
  match Elf.of_string ?align obj with Ok o -> o | Error e -> failwith e

let last_kernel (o : Elf.t) =
  let last = ref "" in
  Iarray.iter
    (fun (s : Elf.symbol) ->
      match s.place with
      | Image _ when String.ends_with ~suffix:".kd" s.name -> last := s.name
      | Image _ | Undefined | Absolute _ | Outside _ -> ())
    o.symbols;
  !last

(* Floors *)

let sht_symtab = 2
let sht_rela = 4
let sht_dynsym = 11
let u16 = String.get_uint16_le
let u32 s off = Int32.to_int (String.get_int32_le s off) land 0xffff_ffff
let u64 s off = Int64.to_int (String.get_int64_le s off)
let rec nul s i = if String.unsafe_get s i = '\000' then i else nul s (i + 1)

(* The fields of a valid 64-bit object, summed so that no read is dropped. *)
let floor_bytes obj () =
  let shoff = u64 obj 40 and entsize = u16 obj 58 and count = u16 obj 60 in
  let header i = shoff + (i * entsize) in
  let names = u64 obj (header (u16 obj 62) + 24) in
  let sum = ref 0 in
  for i = 0 to count - 1 do
    let h = header i in
    let kind = u32 obj (h + 4) and at = u64 obj (h + 24) in
    let size = u64 obj (h + 32) and link = u32 obj (h + 40) in
    sum :=
      !sum
      + nul obj (names + u32 obj h)
      + u64 obj (h + 8)
      + u64 obj (h + 16)
      + u32 obj (h + 44)
      + u64 obj (h + 48)
      + u64 obj (h + 56);
    if kind = sht_symtab || kind = sht_dynsym then begin
      let strings = u64 obj (header link + 24) in
      for k = 0 to (size / 24) - 1 do
        let e = at + (24 * k) in
        sum :=
          !sum
          + nul obj (strings + u32 obj e)
          + u16 obj (e + 6)
          + u64 obj (e + 8)
      done
    end
    else if kind = sht_rela then
      for k = 0 to (size / 24) - 1 do
        let e = at + (24 * k) in
        sum := !sum + u64 obj e + u64 obj (e + 8) + u64 obj (e + 16)
      done
  done;
  !sum

let floor_symbols (o : Elf.t) =
  let lengths =
    Iarray.map (fun (s : Elf.symbol) -> String.length s.name) o.symbols
  in
  fun () ->
    Iarray.mapi
      (fun k n : Elf.symbol ->
        {
          name = String.sub o.file 0 n;
          place = Image { section = 0; offset = k };
        })
      lengths

let () =
  let amd = code_object 128 in
  let o = of_string amd () in
  let kernel = last_kernel o in
  let amd_row k =
    Thumper.bench (strf "amd-%d-kernels" k) (of_string (code_object k))
  in
  exit
    (Thumper.run "rig_elf"
       [
         Thumper.group "of-string"
           [
             amd_row 16;
             amd_row 128;
             Thumper.bench "floor-bytes-amd-128-kernels" (floor_bytes amd);
             Thumper.bench "floor-symbols-amd-128-kernels" (floor_symbols o);
             Thumper.bench "cubin-sm89" (of_string ~align:128 cubin);
             Thumper.bench "host-aarch64" (of_string host);
           ];
         Thumper.group "symbol"
           [
             Thumper.bench "amd-128-kernels-last-kernel" (fun () ->
                 Elf.symbol o kernel);
           ];
       ])
