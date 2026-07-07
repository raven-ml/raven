(* Uop.export / Uop.import serialization tests. *)

open Windtrap
open Tolk_uop
module U = Uop

let call_info : U.call_info =
  {
    grad_fxn = None;
    metadata = [];
    name = None;
    precompile = false;
    precompile_backward = false;
    aux = None;
  }

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:16

let shape dims =
  dims
  |> List.map (fun n -> U.const (Const.int Dtype.Val.weakint n))
  |> U.stack ~dtype:Dtype.Val.weakint

let internal_buffer slot =
  U.buffer ~slot
    ~dtype:(Dtype.Ptr (global_ptr Dtype.Val.float32))
    ~shape:(shape [ 4 ]) ~addrspace:Global ()

(* A compiled-program-shaped graph exercising every argument embedding:
   kernel_info estimates ([Sym]), program_info [vars] and [Launch_sym]
   launch dimensions, a tagged node, and SLICE/COPY call bodies. *)
let compiled_program ~name () =
  let dt = Dtype.Val.int32 in
  let ptr = Dtype.Ptr (global_ptr dt) in
  let var = U.variable ~name:"n" ~min_val:1 ~max_val:128 ~dtype:dt () in
  let p0 = U.param ~slot:0 ~dtype:ptr () in
  let idx = U.index ~ptr:p0 ~idxs:[ U.const_int 0 ] ~as_ptr:true () in
  let sum =
    U.with_tag "serialize-test"
      (U.alu_binary ~op:Ops.Add ~lhs:(U.load ~src:idx ()) ~rhs:var)
  in
  let st = U.store ~dst:idx ~value:sum () in
  let kernel_info : U.kernel_info =
    {
      name;
      axis_types = [];
      dont_use_locals = false;
      applied_opts = [ U.Opt.Upcast { axis = 0; amount = 4 } ];
      opts_to_apply = None;
      estimates = Some { ops = U.Sym var; lds = U.Int 0; mem = U.Int 42 };
      beam = 0;
    }
  in
  let sink = U.sink ~kernel_info [ st ] in
  let kern_call = U.call ~body:sink ~args:[ p0 ] ~info:call_info in
  let sl = U.slice ~src:p0 ~offset:(U.const_int 0) ~size:8 ~dtype:ptr in
  let copy_call =
    U.call
      ~body:(U.copy ~src:sl ~device:(U.Single "CPU") ())
      ~args:[] ~info:call_info
  in
  let linear = U.linear [ kern_call; copy_call ] in
  let info : U.program_info =
    {
      name;
      global_size = [ U.Launch_sym var; U.Launch_int 1; U.Launch_int 1 ];
      local_size = None;
      vars = [ var ];
      globals = [ 0 ];
      outs = [ 0 ];
      ins = [ 0 ];
      aux = [ "aux-payload" ];
    }
  in
  U.program ~sink ~linear
    ~source:(U.source ("void " ^ name ^ "(int* a);"))
    ~binary:(U.binary "\x7fELF\x00\x01") ~info ()

let program_roundtrips_physically () =
  let prog = compiled_program ~name:"kern" () in
  is_true ~msg:"import (export u) == u" (U.import (U.export prog) == prog)

let import_reuses_live_nodes () =
  let blob =
    U.export (U.alu_binary ~op:Ops.Mul ~lhs:(U.const_int 6) ~rhs:(U.const_int 7))
  in
  let fresh =
    U.alu_binary ~op:Ops.Mul ~lhs:(U.const_int 6) ~rhs:(U.const_int 7)
  in
  is_true ~msg:"imported root is the live structurally-equal node"
    (U.import blob == fresh)

let node_tags_roundtrip () =
  let plain =
    U.alu_binary ~op:Ops.Add ~lhs:(U.const_int 1) ~rhs:(U.const_int 2)
  in
  let tagged = U.with_tag "t" plain in
  is_true ~msg:"tag participates in identity" (not (plain == tagged));
  let imported = U.import (U.export tagged) in
  is_true ~msg:"tagged round-trip" (imported == tagged);
  is_true ~msg:"tag preserved"
    (match U.node_tag imported with Some "t" -> true | _ -> false);
  is_true ~msg:"plain round-trip" (U.import (U.export plain) == plain)

let semantic_key_preserved () =
  let prog = compiled_program ~name:"kern" () in
  equal string (U.semantic_key prog)
    (U.semantic_key (U.import (U.export prog)))

let deep_chain_roundtrips () =
  let rec build i acc =
    if i = 0 then acc
    else
      build (i - 1)
        (U.alu_binary ~op:Ops.Add ~lhs:acc
           ~rhs:(U.const (Const.int Dtype.Val.int32 (i land 7))))
  in
  let root = build 50_000 (U.const (Const.int Dtype.Val.int32 0)) in
  is_true ~msg:"deep chain round-trip" (U.import (U.export root) == root)

let export_rejects_grad_fxn () =
  let info =
    { call_info with U.grad_fxn = Some (fun ~grad_output:_ ~call:_ -> []) }
  in
  let call = U.call ~body:(U.const_int 1) ~args:[] ~info in
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> U.export call)

let find_sub haystack needle =
  let n = String.length haystack and m = String.length needle in
  let rec loop i =
    if i + m > n then fail "substring not found"
    else if String.equal (String.sub haystack i m) needle then i
    else loop (i + 1)
  in
  loop 0

let import_rejects_malformed () =
  let failure f = raises_match (function Failure _ -> true | _ -> false) f in
  failure (fun () -> U.import "");
  failure (fun () -> U.import "definitely not an exported graph");
  let blob = U.export (U.const_int 42) in
  (* Truncated inside the magic, the version block, and the graph block. *)
  failure (fun () -> U.import (String.sub blob 0 4));
  failure (fun () -> U.import (String.sub blob 0 12));
  failure (fun () -> U.import (String.sub blob 0 (String.length blob - 4)));
  (* A future format version is rejected without touching the graph. *)
  let v1 = Marshal.to_string 1 [] in
  let v2 = Marshal.to_string 2 [] in
  let p = find_sub blob v1 in
  let bumped =
    String.sub blob 0 p ^ v2
    ^ String.sub blob
        (p + String.length v1)
        (String.length blob - p - String.length v1)
  in
  failure (fun () -> U.import bumped)

(* Buffer nodes hash-cons on their slot: an imported graph that carries a
   process-local internal slot collides with a local buffer minted with the
   same slot. The cache layer must renumber imported internal slots (see
   [Schedule.fresh_internal_buffer_slot]). *)
let buffer_slot_collision () =
  let exported = U.export (internal_buffer (-1)) in
  let local = internal_buffer (-1) in
  is_true ~msg:"imported internal slot aliases the local buffer"
    (U.import exported == local);
  is_true ~msg:"renumbered slot restores distinctness"
    (not (internal_buffer (-2) == local))

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

(* Export in a forked child (whose nodes the parent never sees), import in
   the parent: the graph must land on this process's hash-cons universe. The
   child uses a kernel name no other test builds, so its nodes are genuinely
   foreign to the parent until imported. *)
let cross_process_import () =
  let blob_file = Filename.temp_file "tolk-uop-export" ".blob" in
  let key_file = Filename.temp_file "tolk-uop-export" ".key" in
  Fun.protect
    ~finally:(fun () ->
      Sys.remove blob_file;
      Sys.remove key_file)
    (fun () ->
      (match Unix.fork () with
       | 0 ->
           let code =
             try
               let prog = compiled_program ~name:"kern_xp" () in
               let oc = open_out_bin blob_file in
               output_string oc (U.export prog);
               close_out oc;
               let oc = open_out_bin key_file in
               output_string oc (U.semantic_key prog);
               close_out oc;
               0
             with _ -> 1
           in
           Unix._exit code
       | pid ->
           let _, status = Unix.waitpid [] pid in
           (match status with
            | Unix.WEXITED 0 -> ()
            | _ -> fail "exporting child failed"));
      let imported = U.import (read_file blob_file) in
      (* Building the same graph now reuses the imported nodes. *)
      let fresh = compiled_program ~name:"kern_xp" () in
      is_true ~msg:"imported graph is this process's graph" (imported == fresh);
      equal string ~msg:"semantic key preserved across processes"
        (read_file key_file)
        (U.semantic_key imported);
      (* Argument-embedded uops are remapped onto the same physical nodes
         as the src edges. *)
      match U.as_program_info imported with
      | None -> fail "expected PROGRAM info"
      | Some info -> (
          let var = List.hd info.vars in
          (match info.global_size with
           | U.Launch_sym s :: _ ->
               is_true ~msg:"Launch_sym remapped consistently" (s == var)
           | _ -> fail "expected symbolic launch dimension");
          let sink = (U.src imported).(0) in
          match U.as_kernel_info sink with
          | Some { estimates = Some { ops = U.Sym s; _ }; _ } ->
              is_true ~msg:"estimates Sym remapped consistently" (s == var)
          | _ -> fail "expected symbolic estimate"))

let () =
  run "tolk.uop.serialize"
    [
      group "Serialization"
        [
          test "compiled program round-trips physically"
            program_roundtrips_physically;
          test "import reuses live structurally-equal nodes"
            import_reuses_live_nodes;
          test "node tags round-trip" node_tags_roundtrip;
          test "semantic_key is preserved" semantic_key_preserved;
          test "deep chains round-trip without stack overflow"
            deep_chain_roundtrips;
          test "export rejects gradient functions" export_rejects_grad_fxn;
          test "import rejects malformed input" import_rejects_malformed;
          test "imported internal buffer slots can collide"
            buffer_slot_collision;
          test "cross-process export/import lands on this universe"
            cross_process_import;
        ];
    ]
