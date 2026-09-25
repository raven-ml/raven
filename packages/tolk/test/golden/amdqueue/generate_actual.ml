(* Inspect shared AMD encoder operation payloads at fixed link addresses. *)
open Tolk_amd
open Tolk_uop
module U = Uop
module D = Dtype
module F = Queue_fixture
module Hcq = Tolk_hcq.Hcq
let out_dir = Sys.argv.(1)
let u32 n = U.const (Const.int D.uint32 n)
let u64 = F.uint

let run chip target xccs gc_version nbio_version sdma_version =
  let name = "AMD:" ^ chip in
  let props = ["lds_size_in_kb",64; "max_slots_scratch_cu",4; "simd_count",32*xccs;
    "simd_per_cu",4; "array_count",2*xccs; "simd_arrays_per_engine",1] in
  let dev = device ~target ~xccs ~gc_version ~nbio_version ~sdma_version ~is_aql:false
      ~tmpring_size:0 ~scratch:(Hcq.Buffer.make ~va:0x200000n ~size:0x80000 ~meta:() ())
      ~is_am:false () in
  let ptr tag dtype size address = F.pointer ~device:name ~tag ~dtype ~size ~address in
  let signal = ptr "signal" D.uint64 2 0x400000 in
  let write = ptr "write" D.uint32 2 0x600000 in
  let copy_size = 2 * dev.max_copy_size + 0x400 in
  let src = ptr "source" D.uint8 copy_size 0x10000000 in
  let dst = ptr "destination" D.uint8 copy_size 0x20000000 in
  let call = F.program ~device:name ~dtype:D.int32
      ~binary:(F.read "../../fixtures/amd/simple_add_gfx1100.hsaco") in
  let build label compute nodes =
    let kind = if compute then "compute_0" else "copy_0" in
    let submit = U.custom_function ~name:("submit_amd_" ^ kind) ~srcs:[U.linear nodes; U.group []] in
    let encoded = Option.get (Encoded_queue.encode dev ~props ~name
      ~compute_ring_size:4096 ~copy_ring_size:(fun i -> if i=0 then Some 4096 else None) submit) in
    F.dump out_dir label chip (F.blob (if compute then "cmdbuf_compute" else "cmdbuf_copy_0") encoded) in
  let operation label compute mnemonic operands = build label compute [U.ins ~mnemonic ~operands ()] in
  (* The multi-XCC PM4 scratch partition is a documented deliberate divergence.
     Its dedicated runtime regression checks resident-wave ownership; no word
     masking makes it an exact target packet golden. *)
  if xccs = 1 then build "exec" true [call];
  operation "signal" true "store" [signal; u32 0x42];
  operation "wait" true "wait" [signal; u32 0x42];
  operation "timestamp" true "timestamp" [signal];
  operation "memory_barrier" true "barrier" [];
  List.iter (fun (label,size) ->
      let shrink buf = U.shrink ~src:buf ~offset:(F.int 0) ~size:(F.int size) in
      build label false [U.store_call ~dst:(shrink dst) ~src:(shrink src)])
    ["sdma_copy_small",0x1000; "sdma_copy_large",copy_size;
     "sdma_copy_exact",dev.max_copy_size; "sdma_copy_over_cap",dev.max_copy_size + 1];
  operation "sdma_signal" false "store" [signal; u32 0x42];
  operation "sdma_wait" false "wait" [signal; u32 0x42];
  operation "sdma_timestamp" false "timestamp" [signal];
  operation "sdma_write32" false "write" [write; u32 0x12345678];
  operation "sdma_write64" false "write" [write; u64 0x1122334455667788]

let () =
  run "gfx1100" (11,0,0) 1 (11,0,0) (4,3,0) (6,0,0);
  run "gfx942" (9,4,2) 8 (9,4,3) (7,9,0) (4,4,2)
