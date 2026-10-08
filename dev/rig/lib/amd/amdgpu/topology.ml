(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let ( let* ) = Result.bind
let ( // ) = Filename.concat

type node = {
  index : int;
  gpu_id : int;
  render : int;
  gpu : Rig_amd_abi.Gpu.t;
  lds : int;
  mec : int;
  budget : int;
  visible : int;
  waves_per_cu : int;
  arrays : int;
  cu_per_array : int;
  cwsr : int;
  ctl_stack : int;
}

(* The hardware identifiers of the blocks whose versions name a GPU's formats,
   as the kernel driver lists them under ip_discovery (amdgpu's soc15_hw_ip.h:
   GC_HWID, SDMA0_HWID). *)
let gc_hwid = 11
let sdma_hwid = 42

(* The heap types of a node's memory banks that are the GPU's own memory: the
   part the host reaches through the BAR, and the rest (kfd_crat.h). *)
let heap_public = 1
let heap_private = 2

let read file =
  match In_channel.with_open_text file In_channel.input_all with
  | s -> Some (String.trim s)
  | exception Sys_error _ -> None

let entries dir = try Array.to_list (Sys.readdir dir) with Sys_error _ -> []
let int file = Option.bind (read file) int_of_string_opt

(* The "key value" lines of a properties file whose value is an integer. *)
let properties file =
  let line l =
    match String.split_on_char ' ' (String.trim l) with
    | [ k; v ] -> Option.map (fun v -> (k, v)) (int_of_string_opt v)
    | _ -> None
  in
  match read file with
  | None -> []
  | Some s -> List.filter_map line (String.split_on_char '\n' s)

(* PCI functions *)

let gpus root =
  let devices = root // "sys/bus/pci/devices" in
  let is_gpu bus =
    match
      (int (devices // bus // "vendor"), int (devices // bus // "class"))
    with
    | Some vendor, Some class_ -> Rig_amd.is_gpu ~vendor ~class_
    | _ -> false
  in
  List.sort String.compare (List.filter is_gpu (entries devices))

(* Nodes *)

let nodes root = root // "sys/devices/virtual/kfd/kfd/topology/nodes"

(* A node's location_id holds its function's bus number, then its device and
   function numbers (kfd_topology.c). *)
let bus_of ~domain loc =
  strf "%04x:%02x:%02x.%d" domain (loc lsr 8)
    ((loc lsr 3) land 0x1f)
    (loc land 7)

(* The GPU node at [bus]: a GPU split into partitions has a node per partition
   at its address, the first of which is the GPU. *)
let find root bus =
  let gpu n =
    let dir = nodes root // string_of_int n in
    let p = properties (dir // "properties") in
    match
      ( int (dir // "gpu_id"),
        List.assoc_opt "domain" p,
        List.assoc_opt "location_id" p )
    with
    | Some gpu_id, Some domain, Some loc
      when gpu_id <> 0 && bus_of ~domain loc = bus ->
        Some (n, gpu_id, p)
    | _ -> None
  in
  entries (nodes root)
  |> List.filter_map int_of_string_opt
  |> List.sort Int.compare |> List.find_map gpu

(* A target as gfx_target_version writes it, major * 10000 + minor * 100 +
   stepping. The kernel driver reports gfx942's partitions as 9.4.3, which
   compilers do not target. *)
let target v =
  let v = if v = 90403 then 90402 else v in
  (v / 10000, v / 100 mod 100, v mod 100)

let version root ~render ~bus hwid name =
  let dir =
    root
    // strf "sys/class/drm/renderD%d/device/ip_discovery/die/0/%d/0" render hwid
  in
  match
    (int (dir // "major"), int (dir // "minor"), int (dir // "revision"))
  with
  | Some a, Some b, Some c -> Ok (a, b, c)
  | _ ->
      Error (strf "%s: the amdgpu driver lists no %s version (%s)" bus name dir)

(* The bytes of the node's memory banks of the heap types [heaps]. *)
let banks dir heaps =
  let bank b =
    let p = properties (dir // "mem_banks" // b // "properties") in
    match (List.assoc_opt "heap_type" p, List.assoc_opt "size_in_bytes" p) with
    | Some h, Some n when List.mem h heaps -> n
    | _ -> 0
  in
  List.fold_left (fun n b -> n + bank b) 0 (entries (dir // "mem_banks"))

let node root bus =
  match find root bus with
  | None -> Error (strf "%s is not held by the amdgpu driver" bus)
  | Some (index, gpu_id, p) ->
      let dir = nodes root // string_of_int index in
      let prop k =
        match List.assoc_opt k p with
        | Some v -> Ok v
        | None -> Error (strf "%s: the amdgpu driver reports no %s" bus k)
      in
      let* target_version = prop "gfx_target_version" in
      let* render = prop "drm_render_minor" in
      let* simds = prop "simd_count" in
      let* simd_per_cu = prop "simd_per_cu" in
      let* arrays = prop "array_count" in
      let* arrays_per_engine = prop "simd_arrays_per_engine" in
      let* cu_per_array = prop "cu_per_simd_array" in
      let* waves_per_simd = prop "max_waves_per_simd" in
      let* lds_kib = prop "lds_size_in_kb" in
      let* scratch_slots = prop "max_slots_scratch_cu" in
      let* mec = prop "fw_version" in
      let* cwsr = prop "cwsr_size" in
      let* ctl_stack = prop "ctl_stack_size" in
      let* gc = version root ~render ~bus gc_hwid "GC" in
      let* sdma = version root ~render ~bus sdma_hwid "SDMA" in
      let xccs = Option.value ~default:1 (List.assoc_opt "num_xcc" p) in
      let gpu =
        {
          Rig_amd_abi.Gpu.target = target target_version;
          gc;
          sdma;
          xccs;
          shader_engines = arrays / arrays_per_engine / xccs;
          compute_units = simds / simd_per_cu / xccs;
          scratch_slots;
        }
      in
      Ok
        {
          index;
          gpu_id;
          render;
          gpu;
          lds = lds_kib * 1024;
          mec;
          budget = banks dir [ heap_public; heap_private ];
          visible = banks dir [ heap_public ];
          waves_per_cu = waves_per_simd * simd_per_cu;
          arrays = arrays_per_engine;
          cu_per_array;
          cwsr;
          ctl_stack;
        }

let linked root n n' =
  let dir = nodes root // string_of_int n in
  let to_ links l =
    List.assoc_opt "node_to" (properties (dir // links // l // "properties"))
  in
  List.exists
    (fun links ->
      List.exists (fun l -> to_ links l = Some n') (entries (dir // links)))
    [ "io_links"; "p2p_links" ]
