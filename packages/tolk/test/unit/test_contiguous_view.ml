(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk_uop
module U = Uop

let contiguous_view = Tolk.Prepare.contiguous_view
let shape dims = U.stack (List.map U.const_int dims)
let byte_offset u = Option.map snd (contiguous_view u)
let buffer () = U.buffer ~slot:93820 ~dtype:Dtype.int32 ~shape:(shape [2; 3]) ()

let cancelling_movements () =
  let base = buffer () in
  let twice = U.permute ~src:(U.permute ~src:base ~order:[1; 0]) ~order:[1; 0] in
  equal (option int) (Some 0) (byte_offset twice);
  let twice = U.flip ~src:(U.flip ~src:base ~dims:[true; false]) ~dims:[true; false] in
  equal (option int) (Some 0) (byte_offset twice)

let effects_remain_in_anchor () =
  let base = U.buffer ~slot:93821 ~dtype:Dtype.int32 ~shape:(U.const_int 8) () in
  let store = U.store ~dst:base ~value:(U.broadcast_to
      ~src:(U.const_of_dtype Dtype.int32 (Const_scalar (`Int 1L)))
      ~shape:(U.const_int 8)) () in
  let after = U.after ~src:base ~deps:[store] in
  let view = U.shrink ~src:after ~offset:(U.const_int 2) ~size:(U.const_int 3) in
  match contiguous_view view with
  | None -> fail "effect-backed contiguous view was not proved"
  | Some (anchor, offset) ->
      equal int 8 offset;
      is_true ~msg:"the anchor keeps the pending store" (U.equal after anchor)

let typed_bitcast_anchor () =
  let base = U.buffer ~slot:93822 ~dtype:Dtype.float32 ~shape:(U.const_int 4) () in
  let bytes = U.bitcast ~src:base ~dtype:Dtype.uint8 in
  let view = U.shrink ~src:bytes ~offset:(U.const_int 1) ~size:(U.const_int 4) in
  match contiguous_view view with
  | None -> fail "byte slice was not proved"
  | Some (anchor, offset) ->
      equal int 1 offset;
      is_true ~msg:"subword offset keeps the byte-typed anchor" (U.equal bytes anchor)

let empty_and_storage_anchors () =
  let dims = U.const_int 8 in
  let sources =
    [U.buffer ~slot:93824 ~dtype:Dtype.float32 ~shape:dims ();
     U.alloc ~slot:93825 ~dtype:Dtype.float32 ~shape:dims ();
     U.param ~slot:93826 ~dtype:Dtype.float32 ~shape:dims ()] in
  List.iter (fun source ->
      List.iter (fun (offset, size) ->
          let view = U.shrink ~src:source ~offset:(U.const_int offset)
              ~size:(U.const_int size) in
          match contiguous_view view with
          | None -> fail "a storage-backed slice was not proved"
          | Some (anchor, bytes) ->
              is_true ~msg:"the query retains the storage identity" (U.equal source anchor);
              equal int (offset * 4) bytes) [0, 0; 3, 0; 2, 3]) sources

let tags_are_not_proofs () =
  let source = U.with_tag "caller_owned" (U.buffer ~slot:93827 ~dtype:Dtype.int32
      ~shape:(U.const_int 8) ()) in
  let tail = U.shrink ~src:source ~offset:(U.const_int 2) ~size:(U.const_int 3) in
  (match contiguous_view tail with
   | None -> fail "tagged contiguous source was rejected"
   | Some (anchor, _) ->
       is_true ~msg:"the caller's tag and anchor are preserved" (U.equal source anchor));
  let matrix = U.reshape ~src:source ~shape:(shape [2; 4]) in
  let columns = U.shrink ~src:matrix ~offset:(shape [0; 1]) ~size:(shape [2; 2]) in
  equal (option int) None (byte_offset columns)

let exact_offsets () =
  let large = 1 lsl 32 in
  let base = U.buffer ~slot:93828 ~dtype:Dtype.uint8 ~shape:(U.const_int 1) () in
  let repeated = U.expand ~src:base ~dims:(shape [large; large]) in
  let last = U.shrink ~src:repeated ~offset:(shape [large - 1; large - 1; 0])
      ~size:(shape [1; 1; 1]) in
  equal (option int) (Some 0) (byte_offset last);
  let source = U.param ~slot:93829 ~dtype:Dtype.float64 ~shape:(U.const_int max_int) () in
  let view = U.shrink ~src:source ~offset:(U.const_int (max_int / 8 + 1))
      ~size:(U.const_int 1) in
  raises_match (function Invalid_argument message ->
      message = "Indexing.contiguous_view: byte offset does not fit a host integer" | _ -> false)
    (fun () -> ignore (contiguous_view view))

let storage_view ~src ~offset ~size ~dtype =
  let offset = U.O.(offset * U.const_int (Dtype.itemsize (U.dtype src))) in
  let bytes = U.bitcast ~src ~dtype:Dtype.int8 in
  U.bitcast ~dtype ~src:(U.shrink ~src:bytes ~offset
      ~size:(U.const_int (size * Dtype.itemsize dtype)))

let existing_view_cases () =
  let buffer = U.buffer ~slot:93823 ~dtype:Dtype.int32 ~shape:(U.const_int 8) () in
  is_true ~msg:"Contiguous view offset for base buffer is zero"
    (byte_offset buffer = Some 0);
  let offset_slice =
    storage_view ~src:buffer ~offset:(Uop.const_int 3) ~size:2
      ~dtype:Dtype.int32
  in
  is_true ~msg:"Contiguous view offset accumulates slice offset"
    (byte_offset offset_slice = Some 12);
  let matrix_shape = Uop.stack [ Uop.const_int 4; Uop.const_int 5 ] in
  let matrix = Uop.buffer ~slot:2 ~dtype:Dtype.int32 ~shape:matrix_shape () in
  let row_slice =
    Uop.shrink ~src:matrix
      ~offset:(Uop.stack [ Uop.const_int 1; Uop.const_int 0 ])
      ~size:(Uop.stack [ Uop.const_int 2; Uop.const_int 5 ])
  in
  is_true ~msg:"Contiguous view offset handles full-row shrink"
    (byte_offset row_slice = Some 20);
  let col_slice =
    Uop.shrink ~src:matrix
      ~offset:(Uop.stack [ Uop.const_int 0; Uop.const_int 1 ])
      ~size:(Uop.stack [ Uop.const_int 4; Uop.const_int 2 ])
  in
  is_true ~msg:"Contiguous view offset rejects strided shrink"
    (byte_offset col_slice = None);
  let single_row_cols =
    Uop.shrink ~src:matrix
      ~offset:(Uop.stack [ Uop.const_int 1; Uop.const_int 2 ])
      ~size:(Uop.stack [ Uop.const_int 1; Uop.const_int 2 ])
  in
  is_true ~msg:"Contiguous view offset handles one-row column shrink"
    (byte_offset single_row_cols = Some 28);
  let reshaped_matrix =
    Uop.reshape ~src:matrix
      ~shape:(Uop.stack [ Uop.const_int 2; Uop.const_int 10 ])
  in
  let reshaped_rows =
    Uop.shrink ~src:reshaped_matrix
      ~offset:(Uop.stack [ Uop.const_int 1; Uop.const_int 0 ])
      ~size:(Uop.stack [ Uop.const_int 1; Uop.const_int 10 ])
  in
  is_true ~msg:"Contiguous view offset composes through reshape"
    (byte_offset reshaped_rows = Some 40);
  let zero_pad =
    Uop.pad ~src:matrix
      ~offset:(Uop.stack [ Uop.const_int 0; Uop.const_int 0 ])
      ~size:(Uop.stack [ Uop.const_int 4; Uop.const_int 5 ])
  in
  is_true ~msg:"Contiguous view offset accepts zero pad"
    (byte_offset zero_pad = Some 0);
  let positive_pad =
    Uop.pad ~src:matrix
      ~offset:(Uop.stack [ Uop.const_int 1; Uop.const_int 0 ])
      ~size:(Uop.stack [ Uop.const_int 5; Uop.const_int 5 ])
  in
  is_true ~msg:"Contiguous view offset rejects positive pad"
    (byte_offset positive_pad = None);
  let singleton_shape =
    Uop.stack [ Uop.const_int 1; Uop.const_int 3; Uop.const_int 4 ]
  in
  let singleton_matrix =
    Uop.buffer ~slot:4 ~dtype:Dtype.int32 ~shape:singleton_shape ()
  in
  let singleton_permute =
    Uop.permute ~src:singleton_matrix ~order:[ 1; 2; 0 ]
  in
  is_true ~msg:"Contiguous view offset accepts singleton-only permute"
    (byte_offset singleton_permute = Some 0);
  let flipped_singleton =
    Uop.flip ~src:singleton_matrix ~dims:[ true; false; false ]
  in
  is_true ~msg:"Contiguous view offset accepts singleton flip"
    (byte_offset flipped_singleton = Some 0);
  let flipped_nonsingleton =
    Uop.flip ~src:singleton_matrix ~dims:[ false; true; false ]
  in
  is_true ~msg:"Contiguous view offset rejects non-singleton flip"
    (byte_offset flipped_nonsingleton = None);
  let sym_one =
    Uop.param ~slot:(-1) ~dtype:Dtype.weakint ~vmin_vmax:(Bound.int (1), Bound.int (1))
      ~name:"one" ~addrspace:Dtype.Alu ()
  in
  let symbolic_singleton =
    Uop.buffer ~slot:5 ~dtype:Dtype.int32
      ~shape:(Uop.stack [ sym_one; Uop.const_int 5 ])
      ()
  in
  let symbolic_permute = Uop.permute ~src:symbolic_singleton ~order:[ 1; 0 ] in
  let symbolic_flip =
    Uop.flip ~src:symbolic_singleton ~dims:[ true; false ]
  in
  is_true ~msg:"Contiguous view offset accepts bounded symbolic singleton permute"
    (byte_offset symbolic_permute = Some 0);
  is_true ~msg:"Contiguous view offset accepts bounded symbolic singleton flip"
    (byte_offset symbolic_flip = Some 0)

let contiguous_prepend_expand () =
  let base = Uop.buffer ~slot:0 ~dtype:Dtype.int32
      ~shape:(Uop.const_int 2) () in
  let repeated = Uop.expand ~src:base ~dims:(Uop.const_int 2) in
  is_true ~msg:"a matching leading dimension still broadcasts the storage"
    (byte_offset repeated = None);
  let singleton = Uop.expand ~src:base ~dims:(Uop.const_int 1) in
  is_true ~msg:"a leading singleton preserves contiguous storage"
    (byte_offset singleton = Some 0);
  let tail = Uop.shrink ~src:singleton
      ~offset:(Uop.stack [Uop.const_int 0; Uop.const_int 1])
      ~size:(Uop.stack [Uop.const_int 1; Uop.const_int 1]) in
  is_true ~msg:"a shrink after a leading singleton retains its byte offset"
    (byte_offset tail = Some 4)

let bitcast_singleton_extent () =
  let base = U.buffer ~slot:93830 ~dtype:Dtype.int32 ~shape:(U.const_int 1) () in
  let bytes = U.bitcast ~src:base ~dtype:Dtype.uint8 in
  equal (option int) (Some 0) (byte_offset bytes);
  equal (option int) (Some 0)
    (byte_offset (U.shrink ~src:bytes ~offset:(U.const_int 0) ~size:(U.const_int 4)))

let symbolic_leading_extent () =
  let base = U.buffer ~slot:93831 ~dtype:Dtype.float32 ~shape:(U.const_int 6) () in
  let matrix = U.reshape ~src:base ~shape:(shape [3; 2]) in
  let rows = U.variable ~name:"rows" ~min_val:1 ~max_val:2 () in
  let view = U.shrink ~src:matrix ~offset:(shape [1; 0])
      ~size:(U.stack [rows; U.const_int 2]) in
  equal (option int) (Some 8) (byte_offset view)

let unsupported_devices () =
  List.iter (fun device ->
      let base = U.buffer ~slot:93832 ~dtype:Dtype.int32 ~shape:(U.const_int 1)
          ~device:(U.Single device) () in
      let bytes = U.bitcast ~src:base ~dtype:Dtype.uint8 in
      equal (option int) None (byte_offset bytes);
      is_true ~msg:"typed anchor resolution preserves backend view restrictions"
        (Option.is_none (Tolk.Callify.contiguous_view bytes))) ["WEBGPU"; "CL"]

let storage_windows_keep_allocation_boundaries () =
  let base = buffer () in
  let arithmetic = U.alu_binary ~op:Ops.Add ~lhs:base ~rhs:base in
  is_true ~msg:"a contiguous value still needs storage"
    (Option.is_none (Tolk.Indexing.storage_window arithmetic));
  let staged = U.contiguous ~src:arithmetic ~force:true () in
  let flat = U.reshape ~src:staged ~shape:(U.const_int 6) in
  let view = U.shrink ~src:flat ~offset:(U.const_int 2) ~size:(U.const_int 3) in
  match Tolk.Indexing.storage_window view with
  | Some (anchor, offset) ->
      is_true ~msg:"a stage owns the window, not its arithmetic input" (U.equal anchor staged);
      equal int 8 offset
  | None -> fail "staged storage window was not proved"

let storage_windows_keep_effects_and_typed_anchors () =
  let base = U.buffer ~slot:93833 ~dtype:Dtype.int32 ~shape:(U.const_int 8) () in
  let store = U.store ~dst:base ~value:base () in
  let after = U.after ~src:base ~deps:[store] in
  let bytes = U.bitcast ~src:after ~dtype:Dtype.uint8 in
  let view = U.shrink ~src:bytes ~offset:(U.const_int 1) ~size:(U.const_int 4) in
  match Tolk.Indexing.storage_window view with
  | Some (anchor, offset) ->
      is_true ~msg:"subword storage keeps its typed effect-bearing anchor" (U.equal anchor bytes);
      equal int 1 offset
  | None -> fail "typed effect-bearing storage window was not proved"

let shaped_constant_proof () =
  let base = U.buffer ~slot:93833 ~dtype:Dtype.uint32 ~shape:(shape [2; 3]) () in
  let always_false = U.alu_binary ~op:Ops.Cmplt ~lhs:base
      ~rhs:(U.const (Const.int Dtype.uint32 0)) in
  let self_compare = U.alu_binary ~op:Ops.Cmpne ~lhs:base ~rhs:base in
  let zero = U.alu_binary ~op:Ops.Xor ~lhs:base ~rhs:base in
  let flags = U.buffer ~slot:93834 ~dtype:Dtype.bool ~shape:(shape [2; 3]) () in
  let true_ = U.alu_binary ~op:Ops.Or ~lhs:flags ~rhs:(U.O.not_ flags) in
  let nested_flags = U.permute
      ~src:(U.permute ~src:flags ~order:[1; 0]) ~order:[1; 0] in
  let closure = U.O.where flags nested_flags (U.const_bool false) in
  equal (list int) [2; 3] (U.max_shape (Symbolic.simplify closure));
  List.iter (fun value ->
      let permuted = U.permute ~src:(U.cast ~src:value ~dtype:Dtype.int32)
          ~order:[1; 0] in
      equal (list int) [3; 2] (U.max_shape (Symbolic.simplify permuted));
      equal (option int) None (byte_offset permuted))
    [always_false; self_compare; zero; true_]

let () = run "Contiguous view"
    [test "storage windows retain allocation boundaries" storage_windows_keep_allocation_boundaries;
     test "storage windows retain typed effects" storage_windows_keep_effects_and_typed_anchors;
test "constant folding preserves tensor shape during view proofs" shaped_constant_proof;
     test "unsupported backends reject typed views" unsupported_devices;
     test "one-element bitcast views preserve byte extent" bitcast_singleton_extent;
     test "symbolic leading views compose their flattened index" symbolic_leading_extent;
     test "existing movement views preserve byte offsets" existing_view_cases;
     test "prepend EXPAND respects contiguous storage" contiguous_prepend_expand;
     test "cancelling movement chains prove a contiguous view" cancelling_movements;
     test "view anchors retain pending effects" effects_remain_in_anchor;
     test "subword byte offsets retain typed anchors" typed_bitcast_anchor;
     test "empty views preserve offsets and storage anchors" empty_and_storage_anchors;
     test "caller tags are preserved without certifying strided views" tags_are_not_proofs;
     test "view offsets use exact arithmetic before host narrowing" exact_offsets]
