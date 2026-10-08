(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module P = Rig_amd_amdgpu
module Gpu = Rig_amd_abi.Gpu

let strf = Printf.sprintf

(* Machines as their files show them, written under this suite's directory: an
   R9700 (gfx1201) behind two bridges of its own, with its audio function, as
   nonnormal's /sys lists them. *)

let rec mkdirs d =
  if not (Sys.file_exists d) then begin
    mkdirs (Filename.dirname d);
    Sys.mkdir d 0o755
  end

let rec remove p =
  if Sys.file_exists p then
    if Sys.is_directory p then begin
      Array.iter (fun e -> remove (Filename.concat p e)) (Sys.readdir p);
      Sys.rmdir p
    end
    else Sys.remove p

let tree name files =
  let root = Filename.concat (Sys.getcwd ()) ("trees/" ^ name) in
  remove root;
  List.iter
    (fun (path, contents) ->
      let file = Filename.concat root path in
      mkdirs (Filename.dirname file);
      Out_channel.with_open_text file (fun oc -> output_string oc contents))
    files;
  root

let functions fns =
  List.concat_map
    (fun (bus, vendor, cls) ->
      [
        ("sys/bus/pci/devices/" ^ bus ^ "/vendor", vendor ^ "\n");
        ("sys/bus/pci/devices/" ^ bus ^ "/class", cls ^ "\n");
      ])
    fns

let r9700_functions =
  [
    ("0000:00:00.0", "0x8086", "0x060000");
    ("0000:03:00.0", "0x1002", "0x060400");
    ("0000:04:00.0", "0x1002", "0x060400");
    ("0000:05:00.0", "0x1002", "0x030000");
    ("0000:05:00.1", "0x1002", "0x040300");
    ("0000:0a:00.0", "0x8086", "0x020000");
  ]

let nodes = "sys/devices/virtual/kfd/kfd/topology/nodes/"
let discovery = "sys/class/drm/renderD128/device/ip_discovery/die/0/"

let r9700_node =
  [
    (nodes ^ "0/gpu_id", "0\n");
    (nodes ^ "0/properties", "cpu_cores_count 20\nsimd_count 0\n");
    (nodes ^ "1/gpu_id", "56387\n");
    ( nodes ^ "1/properties",
      "simd_count 128\n\
       max_waves_per_simd 16\n\
       lds_size_in_kb 64\n\
       array_count 8\n\
       simd_arrays_per_engine 2\n\
       cu_per_simd_array 8\n\
       simd_per_cu 2\n\
       max_slots_scratch_cu 32\n\
       gfx_target_version 120001\n\
       vendor_id 4098\n\
       location_id 1280\n\
       domain 0\n\
       drm_render_minor 128\n\
       cwsr_size 30699520\n\
       ctl_stack_size 28672\n\
       fw_version 3010\n\
       num_xcc 1\n" );
    ( nodes ^ "1/mem_banks/0/properties",
      "heap_type 2\nsize_in_bytes 34208743424\nflags 0\n" );
    (discovery ^ "11/0/major", "12\n");
    (discovery ^ "11/0/minor", "0\n");
    (discovery ^ "11/0/revision", "1\n");
    (discovery ^ "42/0/major", "7\n");
    (discovery ^ "42/0/minor", "0\n");
    (discovery ^ "42/0/revision", "1\n");
  ]

let r9700 () = tree "r9700" (functions r9700_functions @ r9700_node)

(* A GPU's node [1] and its blocks' versions, at bus [0000:c1:00.0] behind
   render node 129. *)
let node props ~gc:(a, b, c) ~sdma:(a', b', c') =
  let discovery = "sys/class/drm/renderD129/device/ip_discovery/die/0/" in
  let v hwid (x, y, z) =
    [
      (discovery ^ hwid ^ "/0/major", strf "%d\n" x);
      (discovery ^ hwid ^ "/0/minor", strf "%d\n" y);
      (discovery ^ hwid ^ "/0/revision", strf "%d\n" z);
    ]
  in
  functions [ ("0000:c1:00.0", "0x1002", "0x120000") ]
  @ [
      (nodes ^ "0/gpu_id", "0\n");
      (nodes ^ "0/properties", "cpu_cores_count 64\nsimd_count 0\n");
      (nodes ^ "1/gpu_id", "41216\n");
      ( nodes ^ "1/properties",
        String.concat ""
          (List.map
             (fun (k, n) -> strf "%s %d\n" k n)
             (props
             @ [
                 ("vendor_id", 4098);
                 ("location_id", 0xc100);
                 ("domain", 0);
                 ("drm_render_minor", 129);
                 ("lds_size_in_kb", 64);
                 ("max_slots_scratch_cu", 32);
                 ("fw_version", 150);
               ])) );
      ( nodes ^ "1/mem_banks/0/properties",
        "heap_type 1\nsize_in_bytes 206141652992\nflags 0\n" );
    ]
  @ v "11" (a, b, c)
  @ v "42" (a', b', c')

(* An MI300X of eight dies, 38 compute units and 4 shader engines each, which
   the kernel driver reports as 9.4.3. *)
let mi300x () =
  tree "mi300x"
    (node ~gc:(9, 4, 3) ~sdma:(4, 4, 2)
       [
         ("simd_count", 1216);
         ("simd_per_cu", 4);
         ("max_waves_per_simd", 8);
         ("array_count", 32);
         ("simd_arrays_per_engine", 1);
         ("gfx_target_version", 90403);
         ("num_xcc", 8);
         ("cwsr_size", 17965000);
         ("ctl_stack_size", 20480);
       ])

(* An MI210, of GFX 9.0.10: 104 compute units in 8 shader engines. *)
let mi210 () =
  tree "mi210"
    (node ~gc:(9, 4, 2) ~sdma:(4, 4, 0)
       [
         ("simd_count", 416);
         ("simd_per_cu", 4);
         ("max_waves_per_simd", 8);
         ("array_count", 8);
         ("simd_arrays_per_engine", 1);
         ("gfx_target_version", 90010);
         ("cwsr_size", 9043968);
         ("ctl_stack_size", 20480);
       ])

let gpu =
  Testable.make
    ~pp:(fun ppf (g : Gpu.t) ->
      let v ppf (a, b, c) = Format.fprintf ppf "%d.%d.%d" a b c in
      Format.fprintf ppf
        "{target %a; gc %a; sdma %a; xccs %d; engines %d; units %d; slots %d}" v
        g.target v g.gc v g.sdma g.xccs g.shader_engines g.compute_units
        g.scratch_slots)
    ~equal:( = )

let result_gpu =
  Testable.make
    ~pp:(fun ppf -> function
      | Ok g -> Format.fprintf ppf "Ok %a" (Testable.pp gpu) g
      | Error e -> Format.fprintf ppf "Error %S" e)
    ~equal:( = )

let numbering =
  group ~timeout:10. "numbering"
    [
      test "a machine's AMD GPUs are its display functions, in bus order"
        (fun () ->
          equal (list string) [ "0000:05:00.0" ] (P.gpus_at (r9700 ())));
      test "an accelerator is a GPU, and GPUs come in bus order" (fun () ->
          let root =
            tree "two"
              (functions
                 [
                   ("0000:c1:00.0", "0x1002", "0x120000");
                   ("0000:05:00.0", "0x1002", "0x030000");
                   ("0000:41:00.0", "0x10de", "0x030000");
                 ])
          in
          equal (list string)
            [ "0000:05:00.0"; "0000:c1:00.0" ]
            (P.gpus_at root));
      test "a machine without PCI files has no GPU" (fun () ->
          equal (list string) [] (P.gpus_at (tree "empty" [])));
      test "this machine's count is its files' GPUs" (fun () ->
          equal int (List.length (P.gpus_at "/")) (P.count ()));
    ]

let facts =
  group ~timeout:10. "facts"
    [
      test "the R9700 is a gfx1201 of 64 compute units" (fun () ->
          let expected =
            {
              Gpu.target = (12, 0, 1);
              gc = (12, 0, 1);
              sdma = (7, 0, 1);
              xccs = 1;
              shader_engines = 4;
              compute_units = 64;
              scratch_slots = 32;
            }
          in
          equal result_gpu (Ok expected) (P.gpu_at (r9700 ()) "0000:05:00.0"));
      test "an MI300X's partition, reported as 9.4.3, is a gfx942 of 8 dies"
        (fun () ->
          let expected =
            {
              Gpu.target = (9, 4, 2);
              gc = (9, 4, 3);
              sdma = (4, 4, 2);
              xccs = 8;
              shader_engines = 4;
              compute_units = 38;
              scratch_slots = 32;
            }
          in
          equal result_gpu (Ok expected) (P.gpu_at (mi300x ()) "0000:c1:00.0"));
      test "a GPU the driver holds no node of is refused" (fun () ->
          let root = tree "unheld" (functions r9700_functions) in
          equal result_gpu
            (Error "0000:05:00.0 is not held by the amdgpu driver")
            (P.gpu_at root "0000:05:00.0"));
    ]

(* The context save area KFD requires (kfd_queue.c): each die's [cwsr_size],
   then a debugger area of 32 bytes per wave rounded up to 64, the whole
   rounded up to a page. Its waves are 32 per compute unit of a die from GFX
   10.1; before, 40 per compute unit up to 512 per shader engine of the
   GPU. *)
let save_area =
  let result_int =
    Testable.make
      ~pp:(fun ppf -> function
        | Ok n -> Format.fprintf ppf "Ok %d" n
        | Error e -> Format.fprintf ppf "Error %S" e)
      ~equal:( = )
  in
  let at root = P.save_area_at root "0000:c1:00.0" in
  group ~timeout:10. "save area"
    [
      test "an R9700's: 2048 waves of 64 compute units" (fun () ->
          equal result_int
            (Ok (30699520 + (2048 * 32)))
            (P.save_area_at (r9700 ()) "0000:05:00.0"));
      test "an MI300X's: 1520 waves on each of its 8 dies, rounded to a page"
        (fun () ->
          let die = 17965000 + (1520 * 32) in
          equal result_int
            (Ok ((((8 * die) + 4095) / 4096) * 4096))
            (at (mi300x ())));
      test "an MI210's: 4096 waves, 512 for each of 8 engines" (fun () ->
          equal result_int (Ok (9043968 + (4096 * 32))) (at (mi210 ())));
      test "a GPU the driver holds no node of has none" (fun () ->
          equal result_int
            (Error "0000:05:00.0 is not held by the amdgpu driver")
            (P.save_area_at (tree "unheld" (functions r9700_functions)) "0000:05:00.0"));
    ]

let names =
  group ~timeout:10. "names"
    [
      test "GPUs are named AMD, AMD:1, ..." (fun () ->
          equal (list string)
            [ "AMD"; "AMD:1"; "AMD:12" ]
            (List.map P.device_name [ 0; 1; 12 ]));
      test "a negative GPU is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"device_name") (fun () ->
              P.device_name (-1));
          raises_match (Exn.invalid_arg ~substring:"open_") (fun () ->
              P.open_ (-1)));
      test "a GPU past the machine's is refused, naming their number" (fun () ->
          let n = P.count () in
          match P.open_ n with
          | Ok _ -> fail "opened a GPU past the machine's"
          | Error why ->
              equal string
                (Printf.sprintf "no GPU %d; the machine has %d AMD GPUs" n n)
                why);
    ]

let () = exit (run "rig_amd_amdgpu" [ numbering; facts; save_area; names ])
