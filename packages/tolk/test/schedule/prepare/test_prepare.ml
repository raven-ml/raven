(* Tests of Tolk.Prepare, in the order of prepare.mli: pm_mops and the index
   moves it makes, the expansion of bitcasts, what prepare_rangeify makes of
   recorded and hand-built graphs, contiguous_view and sequence. *)

open Windtrap
open Tolk

let uop = Uops.uop
let sint = Testable.make ~pp:Shape.Sint.pp ~equal:Shape.Sint.equal
let prepare = Prepare.prepare_rangeify
let size shape = List.fold_left ( * ) 1 shape
let ints = List.map (fun n -> Ops.Int n)
let cpu = "CPU"
let flat n = Shape.param ~device:cpu ~shape:[ Int n ] 1 Float32
let stored shape = Shape.reshape (flat (size shape)) (ints shape)
let rejects f = raises_match (Exn.invalid_arg ?sub:None) f

(* The nodes of [u] of the operation [op], call bodies included. *)
let nodes op u =
  List.filter (fun n -> Ops.op n = op) (Ops.toposort ~calls:Enter u)

(* The split's settings at their defaults, which the recorded programs and the
   split tests assume: splits on, a first reduction of at most 2^22 elements,
   and an input of at least 32768 times as many elements as the output. *)
let split_defaults =
  Setting.
    [
      B (split_reduceop, true);
      B (reduceop_split_size, 22);
      B (reduceop_split_threshold, 32768);
    ]

(* pm_mops

   Movements move towards the storage they view. The law: an index of a movement
   chain, rewritten, reads through the storage the element that the movements
   place there (Tensors). Pads are left out of the law: the validity a pad gives
   an index lasts only while the index carries it (Indexing.apply_movement_op),
   which the padded rows state. Storage element [j] holds [j + 1], so that a
   padded [0] is told from a read. *)

let mops u =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
    (After_sources Prepare.pm_mops)

let range ?(axis_type = Ops.Axis_type.Loop) n axis =
  Ops.range ~axis_type (Int n) [ axis ]

let ranges shape = List.mapi (fun axis n -> range n axis) shape

let copy_in =
  Ops.store (flat 4) (Shape.param ~device:cpu ~shape:[ Int 4 ] 2 Float32)

let pm_mops_rules =
  let x = stored [ 4; 8 ] in
  let r0 = range 8 0 and r1 = range 4 1 in
  [
    test "an index of a movement indexes its source at the mapped indices"
      (fun () ->
        let mapped =
          Indexing.apply_movement_op
            (ints [ 4; 8 ])
            (Permute [ 1; 0 ])
            [ r0; r1 ]
        in
        let flat_index =
          Indexing.apply_movement_op [ Int 32 ] (Reshape (ints [ 4; 8 ])) mapped
        in
        equal uop
          (Ops.index (flat 32) flat_index)
          (mops (Ops.index (Shape.permute x [ 1; 0 ]) [ r0; r1 ])));
    test
      "an index of a reshape's leading axes indexes its source when the \
       trailing axes are kept" (fun () ->
        let a = range 2 0 and b = range 2 1 in
        let lead =
          Indexing.apply_movement_op [ Int 4 ]
            (Reshape (ints [ 2; 2 ]))
            [ a; b ]
        in
        equal uop (Ops.index x lead)
          (mops (Ops.index (Shape.reshape x (ints [ 2; 2; 8 ])) [ a; b ])));
    test "an index of a reshape whose trailing axes change is left as it is"
      (fun () ->
        let u = Ops.index (Shape.reshape x (ints [ 2; 16 ])) [ range 2 0 ] in
        equal uop u (mops u));
    test "an index of a reshape's added leading axis is its source" (fun () ->
        let row = stored [ 8 ] in
        equal uop row
          (mops (Ops.index (Shape.reshape row (ints [ 1; 8 ])) [ range 1 0 ])));
    test "a store through a selection of an index gates the index" (fun () ->
        let c = Ops.lt r0 (Ops.int 6) in
        let value = Ops.float ~dtype:Float32 1. in
        equal uop
          (Ops.store (Ops.index (flat 32) [ Shape.valid r0 c ]) value)
          (mops
             (Ops.store
                (Ops.where c (Ops.index (flat 32) [ r0 ]) (Ops.int 0))
                value)));
    test "a store through a padded view of a gated index gates it twice"
      (fun () ->
        let a = Ops.lt r0 (Ops.int 6) and b = Ops.lt r1 (Ops.int 3) in
        let dst = Ops.index (flat 32) [ r0 ] in
        let value = Ops.float ~dtype:Float32 1. in
        let padded =
          Ops.where a
            (Ops.where b dst (Ops.int 0))
            (Shape.const_like dst `Invalid)
        in
        equal uop
          (Ops.store
             (Ops.index (flat 32) [ Shape.valid r0 (Ops.bitwise_and a b) ])
             value)
          (mops (Ops.store padded value)));
    test "a store through a selection of something but Invalid stays" (fun () ->
        let a = Ops.lt r0 (Ops.int 6) and b = Ops.lt r1 (Ops.int 3) in
        let dst = Ops.load (Ops.index (flat 32) [ r0 ]) [] in
        let st =
          Ops.store
            (Ops.where a
               (Ops.where b dst (Ops.int 0))
               (Ops.float ~dtype:Float32 1.))
            (Ops.float ~dtype:Float32 2.)
        in
        equal uop st (mops st));
    test "a movement after effects is the movement of its source after them"
      (fun () ->
        let u = flat 32 in
        equal uop
          (Shape.reshape (Ops.after u [ copy_in ]) (ints [ 4; 8 ]))
          (mops (Ops.after (Shape.reshape u (ints [ 4; 8 ])) [ copy_in ])));
    test "an index after effects is the index of its source after them"
      (fun () ->
        let u = flat 32 in
        equal uop
          (Ops.index (Ops.after u [ copy_in ]) [ r0 ])
          (mops (Ops.after (Ops.index u [ r0 ]) [ copy_in ])));
    (* A value in registers, the ALU's address space, as an index's operand
       is. *)
    test
      "an index of a value after effects is the index of the value after them"
      (fun () ->
        let v = Shape.param ~addrspace:Alu ~shape:[ Int 8 ] 3 Float32 in
        equal uop
          (Ops.after (Ops.index v [ r0 ]) [ copy_in ])
          (mops (Ops.index (Ops.after v [ copy_in ]) [ r0 ])));
    test "an end of a movement is the end of its source" (fun () ->
        let u = flat 32 in
        equal uop (Ops.end_ u [ r0 ])
          (mops (Ops.end_ (Shape.reshape u (ints [ 4; 8 ])) [ r0 ])));
    test
      "an index that loads from a storage state moves with the state as it is"
      (fun () ->
        let buf = Ops.new_buffer ~slot:2 cpu 4 Int32 in
        let r = range 4 2 in
        let unsimplified =
          Ops.O.(Ops.cast r Int32 * Ops.const ~dtype:Int32 (`Int Bigint.one))
        in
        let state =
          Ops.after buf
            [ Ops.end_ (Ops.store (Ops.index buf [ r ]) unsimplified) [ r ] ]
        in
        let row = Ops.cast (Ops.index state [ r1 ]) Weak_int in
        let moved = mops (Ops.index (Shape.permute x [ 1; 0 ]) [ r0; row ]) in
        equal (list uop) [ state ] (nodes After moved));
  ]

let concrete u =
  List.map
    (function Ops.Int n -> n | Sym _ -> fail "a generated shape is concrete")
    (Shape.shape u)

let rec coords = function
  | [] -> [ [] ]
  | n :: rest ->
      List.concat_map
        (fun i -> List.map (fun c -> i :: c) (coords rest))
        (List.init n Fun.id)

let pp_list pp ppf l =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") pp)
    l

let pp_movement ppf (m : Ops.movement) =
  let pp_sint = Shape.Sint.pp in
  let pair ppf (a, b) = Format.fprintf ppf "(%a, %a)" pp_sint a pp_sint b in
  match m with
  | Reshape s -> Format.fprintf ppf "reshape %a" (pp_list pp_sint) s
  | Expand s -> Format.fprintf ppf "expand %a" (pp_list pp_sint) s
  | Pad p -> Format.fprintf ppf "pad %a" (pp_list pair) p
  | Shrink b -> Format.fprintf ppf "shrink %a" (pp_list pair) b
  | Permute o -> Format.fprintf ppf "permute %a" (pp_list Format.pp_print_int) o

let rec each = function
  | [] -> Gen.constant []
  | g :: gs -> Gen.map (fun (x, xs) -> x :: xs) (Gen.pair g (each gs))

let divisors n = List.filter (fun d -> n mod d = 0) (List.init n succ)

let rec factors k n =
  let open Gen in
  if k = 1 then constant [ n ]
  else
    let* d = of_list (divisors n) in
    let+ rest = factors (k - 1) (n / d) in
    d :: rest

(* A movement of one of [kinds] that fits [shape]. An expand adds leading axes;
   a shrink and a pad take each axis' offset and size. *)
let movement kinds shape : Ops.movement Gen.t =
  let open Gen in
  let per_axis f = each (List.map f shape) in
  let* kind = of_list kinds in
  match kind with
  | `Reshape ->
      let* k = int_range 1 3 in
      let+ s = factors k (size shape) in
      Ops.Reshape (ints s)
  | `Expand ->
      let+ added = list ~size:(int_range 1 2) (int_range 1 3) in
      Ops.Expand (ints added)
  | `Shrink ->
      let+ b =
        per_axis (fun n ->
            let* start = int_range 0 (n - 1) in
            let+ length = int_range 1 (n - start) in
            (Ops.Int start, Ops.Int length))
      in
      Ops.Shrink b
  | `Permute ->
      let+ o = permutation (List.init (List.length shape) Fun.id) in
      Ops.Permute o
  | `Pad ->
      let+ p =
        per_axis (fun n ->
            let+ before = int_range 0 2 and+ after = int_range 1 2 in
            (Ops.Int before, Ops.Int (before + n + after)))
      in
      Ops.Pad p

let pp_shape = pp_list Format.pp_print_int

(* A shape of up to three axes of up to four elements, and up to three movements
   of one of [kinds], in turn. *)
let chains kinds =
  let open Gen in
  let rec moves shape k =
    if k = 0 then constant []
    else
      let* m = movement kinds shape in
      let next = concrete (Shape.mop (stored shape) m) in
      let+ rest = moves next (k - 1) in
      m :: rest
  in
  (let* shape = list ~size:(int_range 1 3) (int_range 1 4) in
   let* k = int_range 1 3 in
   let+ ms = moves shape k in
   (shape, ms))
  |> with_pp (fun ppf (shape, ms) ->
      Format.fprintf ppf "%a of %a" (pp_list pp_movement) ms pp_shape shape)

let chain = chains [ `Reshape; `Expand; `Shrink; `Permute ]
let moved (shape, ms) = List.fold_left Shape.mop (stored shape) ms
let iota n = Array.init n (fun j -> `Float (float_of_int (j + 1)))
let tensor_values memory u = Array.to_list (Tensors.eval ~buffers:memory u)

(* [reads memory rs u cs] is [u] at each binding [cs] of the ranges [rs]. *)
let reads memory rs u cs =
  let names = List.map (fun r -> Option.get (Interpreter.name r)) rs in
  List.map
    (fun c ->
      let vars =
        List.combine names (List.map (fun i -> `Int (Bigint.of_int i)) c)
      in
      Interpreter.eval ~vars ~buffers:memory u)
    cs

let is_reshape = function Ops.Reshape _ -> true | _ -> false
let is_shrink = function Ops.Shrink _ -> true | _ -> false
let is_permute = function Ops.Permute _ -> true | _ -> false
let is_pad = function Ops.Pad _ -> true | _ -> false
let is_expand = function Ops.Expand _ -> true | _ -> false

let stretches = function
  | Ops.Expand added -> List.exists (fun n -> n <> Ops.Int 1) added
  | _ -> false

let reads_what_it_moves (shape, ms) =
  cover "a reshape" (List.exists is_reshape ms);
  cover "an expand that adds an axis of more than one element"
    (List.exists stretches ms);
  cover "a shrink" (List.exists is_shrink ms);
  cover "a permute" (List.exists is_permute ms);
  let u = moved (shape, ms) in
  let out_shape = concrete u in
  let rs = ranges out_shape in
  let read = mops (Ops.index u rs) in
  let memory = [ (1, iota (size shape)) ] in
  equal (list Dtypes.const) (tensor_values memory u)
    (reads memory rs read (coords out_shape))

(* A pad, then a movement whose simplification makes the index constant on an
   axis: the index drops the pad's gate, as tinygrad HEAD's does, and a caller
   masks the padded values with the pad's condition. Each row is a chain, the
   index tinygrad gives into its storage, [0] or the range of an axis of the
   view, and the pad's condition, an axis of the view below or from a bound. *)
let padded =
  [
    ( "a reshape to an axis of one element",
      ([ 1 ], [ Ops.Reshape (ints [ 1 ]); Pad [ (Int 0, Int 2) ] ]),
      `Zero,
      (0, `Below 1) );
    ( "an expand that drops the leading index",
      ([ 2 ], [ Expand (ints [ 2 ]); Pad [ (Int 1, Int 3); (Int 0, Int 2) ] ]),
      `Axis 1,
      (0, `From 1) );
    ( "a shrink onto an axis of one element",
      ( [ 3; 2 ],
        [
          Pad [ (Int 0, Int 3); (Int 2, Int 6) ];
          Shrink [ (Int 0, Int 1); (Int 0, Int 3) ];
        ] ),
      `Zero,
      (1, `From 2) );
  ]

let lost_index rs = function `Zero -> Ops.int 0 | `Axis k -> List.nth rs k

let pad_condition rs (axis, bound) =
  let r = List.nth rs axis in
  match bound with
  | `Below n -> Ops.O.(r < int n)
  | `From n -> Ops.O.(r >= int n)

let row_name (name, _, _, _) = name

let pads =
  [
    cases ~id:row_name
      "an index through a pad and a movement that makes it constant loses the \
       pad's gate"
      padded (fun (_, (shape, ms), lost, _) ->
        let u = moved (shape, ms) in
        let rs = ranges (concrete u) in
        equal uop
          (Ops.index (flat (size shape)) [ lost_index rs lost ])
          (mops (Ops.index u rs)));
    cases ~id:row_name
      "an index through a pad, masked by the pad's condition, reads what the \
       movements place"
      padded (fun (_, (shape, ms), _, mask) ->
        let u = moved (shape, ms) in
        let rs = ranges (concrete u) in
        let read = mops (Ops.index u rs) in
        let masked =
          Ops.where (pad_condition rs mask) read (Ops.float ~dtype:Float32 0.)
        in
        let memory = [ (1, iota (size shape)) ] in
        equal (list Dtypes.const) (tensor_values memory u)
          (reads memory rs masked (coords (concrete u))));
  ]

let pm_mops =
  group "pm_mops"
    (pm_mops_rules
    @ [
        prop "an index of movements reads the element they move" chain
          reads_what_it_moves;
      ]
    @ pads)

(* move_index and pm_mops_in *)

let one_movement =
  let open Gen in
  (let* shape = list ~size:(int_range 1 3) (int_range 1 4) in
   let+ m = movement [ `Reshape; `Expand; `Shrink; `Permute; `Pad ] shape in
   (shape, m))
  |> with_pp (fun ppf (shape, m) ->
      Format.fprintf ppf "%a of %a" pp_movement m pp_shape shape)

let moves_as_apply_movement_op (shape, m) =
  cover "a reshape" (is_reshape m);
  cover "an expand" (is_expand m);
  cover "a shrink" (is_shrink m);
  cover "a permute" (is_permute m);
  cover "a pad" (is_pad m);
  let rs = ranges (concrete (Shape.mop (stored shape) m)) in
  equal (list uop)
    (Indexing.apply_movement_op (ints shape) m rs)
    (Prepare.move_index (ints shape) m rs)

let move_index =
  group "move_index"
    [
      (* Each kind is drawn at 1/5: at 300 cases a run misses one at 0.8^300,
         1e-29. *)
      prop ~count:300
        "is Indexing.apply_movement_op on indices that load nothing"
        one_movement moves_as_apply_movement_op;
    ]

(* Rewritten twice with one memo, fresh, then filled by the first rewrite. *)
let moves_as_pm_mops chain =
  let u = moved chain in
  let idx = Ops.index u (ranges (concrete u)) in
  let memo = Indexing.memo () in
  let with_memo () =
    Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:memo idx
      (After_sources (Prepare.pm_mops_in Fun.id))
  in
  equal uop (mops idx) (with_memo ());
  equal ~msg:"on the memo it filled" uop (mops idx) (with_memo ())

let pm_mops_in =
  group "pm_mops_in"
    [
      prop "rewrites as pm_mops does, on a fresh memo and on the one it filled"
        chain moves_as_pm_mops;
    ]

(* expand_bitcast

   The expansion of a bitcast computes the bitcast's bits: Tensors reads a
   bitcast to another size as the bytes of each row of the last axis,
   little-endian, and computes the expansion's shifts and casts. Operands are
   drawn as their bits, and results compared as bits, NaNs' payloads
   included. *)

(* The pairs of types the expansion is drawn over: to wider types, then to
   narrower ones. *)
let resizes =
  Dtype.
    [
      (Uint8, Float32);
      (Float16, Float64);
      (Int8, Int16);
      (Float32, Uint8);
      (Float32, Float16);
      (Int64, Int32);
    ]

let word = function
  | 1 -> Dtype.Uint8
  | 2 -> Dtype.Uint16
  | 4 -> Dtype.Uint32
  | _ -> Dtype.Uint64

let bits dtype (v : Dtype.value) =
  Dtype.bitcast dtype (word (Dtype.itemsize dtype)) v

(* A value of [dtype] drawn as its bytes, half of them 0x00, 0xFF, 0x7F or 0x80:
   a float is then often a NaN with a payload, quiet or signalling, an infinity
   or a zero of either sign. *)
let drawn_bits dtype : Dtype.value Gen.t =
  let open Gen in
  let byte =
    frequency [ (1, of_list [ 0x00; 0xFF; 0x7F; 0x80 ]); (1, int_range 0 255) ]
  in
  let+ bytes = list ~size:(constant (Dtype.itemsize dtype)) byte in
  let n =
    List.fold_left
      (fun acc b -> Bigint.(shift_left acc 8 + of_int b))
      Bigint.zero bytes
  in
  Dtype.bitcast (word (Dtype.itemsize dtype)) dtype (`Int n)

(* A pair of [resizes], a shape whose last axis holds whole elements of the
   result, empty ones included, and the operand's elements. *)
let bitcasts =
  let open Gen in
  (let* from, into = of_list resizes in
   let rate = max 1 (Dtype.itemsize into / Dtype.itemsize from) in
   let* lead = list ~size:(int_range 0 1) (int_range 0 2) in
   let* k = int_range 1 2 in
   let shape = lead @ [ k * rate ] in
   let+ xs = list ~size:(constant (size shape)) (drawn_bits from) in
   (from, into, shape, xs))
  |> with_pp (fun ppf (from, into, shape, xs) ->
      Format.fprintf ppf "%a of %a to %a, the operand's bits %a" pp_shape shape
        Dtype.pp from Dtype.pp into
        (pp_list (Testable.pp Dtypes.value))
        (List.map (bits from) xs))

let is_nan = function `Float x -> Float.is_nan x | _ -> false

let expands_to_its_bits (from, into, shape, xs) =
  cover "to a wider type" (Dtype.itemsize into > Dtype.itemsize from);
  cover "to a narrower type" (Dtype.itemsize into < Dtype.itemsize from);
  cover "a NaN operand" (List.exists is_nan xs);
  cover "an empty shape" (size shape = 0);
  let bc =
    Ops.bitcast (Shape.param ~device:cpu ~shape:(ints shape) 1 from) into
  in
  let as_bits u =
    Array.to_list
      (Array.map
         (function
           | #Dtype.value as v -> bits into v
           | `Invalid -> fail "an Invalid element")
         (Tensors.eval ~buffers:[ (1, Array.of_list xs) ] u))
  in
  let expanded = require_some (Prepare.expand_bitcast bc) in
  equal (list Dtypes.value) (as_bits bc) (as_bits expanded)

let expand_bitcast =
  let x = Shape.param ~device:cpu ~shape:[ Int 2 ] 1 Float32 in
  group "expand_bitcast"
    [
      (* A case has a NaN operand at about 0.15, and an empty shape at 1/6: at
         300 cases a run misses one at 0.85^300, 6e-22, or (5/6)^300, 2e-24. *)
      prop ~count:300 "computes the bitcast's bits" bitcasts expands_to_its_bits;
      test "is None where the sizes are equal" (fun () ->
          is_none ~pp:(Testable.pp uop)
            (Prepare.expand_bitcast (Ops.bitcast x Int32)));
      test "is None for a value on a disk" (fun () ->
          let disk =
            Shape.param ~device:"DISK:/tmp/tolk" ~shape:[ Int 8 ] 1 Uint8
          in
          is_none ~pp:(Testable.pp uop)
            (Prepare.expand_bitcast (Ops.bitcast disk Float32)));
    ]

(* prepare_rangeify: recorded programs

   Each program of [programs] is a graph tinygrad's scheduling hands
   prepare_rangeify; its golden is what tinygrad's prepare_rangeify makes of it,
   or, where tolk differs, the [.tolk] beside it, both written by
   test/gen/schedule/prepare.py. *)

let program name = Golden.sink (name ^ ".golden")

let programs =
  [
    "add";
    "add_at_first_twice";
    "add_nothing";
    "alias_of_placed";
    "arange";
    "assign";
    "assign_bitcast";
    "assign_bitcast_wider";
    "assign_broadcast";
    "assign_cross_device";
    "assign_deviceless_const";
    "assign_disjoint_self";
    "assign_double_bitcast";
    "assign_flipped_self";
    "assign_overlapping_shrink_pending";
    "assign_own_contents";
    "assign_permuted_self";
    "assign_reshaped";
    "assign_reshaped_self";
    "assign_shifted_self";
    "assign_shrink_then_bitcast";
    "assign_shrunk_in_place";
    "assign_shrunk_self";
    "assign_to_disk";
    "assign_to_disk_1";
    "assign_to_function_output";
    "assign_twice";
    "attention";
    "below_threshold";
    "bitcast_half";
    "bitcast_long";
    "bitcast_narrow";
    "bitcast_on_disk";
    "bitcast_wide";
    "cat";
    "clone";
    "contiguous";
    "contiguous_backward";
    "contiguous_of_storage";
    "contiguous_permuted";
    "conv";
    "copy";
    "copy_computed";
    "copy_from_disk";
    "copy_into_storage";
    "copy_permuted_from_disk";
    "copy_same_device";
    "copy_staged_view_from_disk";
    "copy_view";
    "cumsum";
    "custom_kernel";
    "detach";
    "disk_staged_view";
    "embedding";
    "flip_of_other";
    "gather_by_self";
    "gather_of_reshape";
    "gather_of_self";
    "gpt_oss_decode";
    "gpt_oss_decode_1";
    "gpt_oss_prefill";
    "gpt_oss_prefill_1";
    "hazard_behind_other_after";
    "inline_function";
    "inline_symbolic";
    "matmul";
    "max_of_nothing";
    "no_split";
    "pad";
    "placed_after_read";
    "placed_through_view";
    "precompiled_function";
    "precompiled_function_1";
    "setitem";
    "setitem_tensor";
    "softmax";
    "sort";
    "split_at_threshold";
    "split_expanded";
    "split_max";
    "split_prime";
    "split_rows";
    "split_sum";
    "split_unit_axis";
    "stack";
    "store_ordered_before";
    "sum";
    "sum_of_nothing";
    "variable_shrink";
    "where";
  ]

(* The settings a program was recorded under: the defaults, and splits off for
   [no_split]. *)
let recorded_under name =
  if name = "no_split" then
    split_defaults @ [ B (Setting.split_reduceop, false) ]
  else split_defaults

let prepared name =
  Setting.context (recorded_under name) (fun () -> prepare (program name))

let goldens =
  group "prepare_rangeify › prepares each recorded program as its golden states"
    (List.map
       (fun name ->
         let file = name ^ "_prepared.golden" in
         Golden.graph file (fun () ->
             Uops.numbered_like (Golden.sink file) (prepared name)))
       programs)

(* What a program writes into the storage it is given, its parameters and
   buffers (Tensors); call-local storage (Op.Alloc) holds scratch values, which
   a pass may add or remove. Memory holds small integers, so a sum a split
   reorders is exact either way while its partial sums stay within the integers
   its type holds exactly, as the recorded programs' do, and writes compare bit
   for bit. *)

(* [element dtype k] is [k], from [-3] to [7], as a [dtype] value; shifted to be
   non-negative in an unsigned type. *)
let element dtype k : Dtype.value =
  if Dtype.is_float dtype then `Float (float_of_int k)
  else if Dtype.equal dtype Bool then `Bool (k > 0)
  else if Dtype.is_unsigned dtype then `Int (Bigint.of_int (k + 3))
  else `Int (Bigint.of_int k)

let storage u =
  List.filter_map
    (fun n ->
      match (Ops.op n, Ops.arg n) with
      | (Param | Buffer | Alloc), Param p when p.addrspace <> Alu -> Some (n, p)
      | _ -> None)
    (Ops.toposort ~calls:Enter u)

let filled u =
  List.map
    (fun (n, (p : Ops.param_arg)) ->
      let at j = element p.dtype ((((j * 7) + (p.slot * 3)) mod 11) - 3) in
      (p.slot, Array.init (Shape.max_numel n) at))
    (storage u)

let write = triple int int Dtypes.value

(* [writes_kept name pass] is the test that the program [name] after [pass]
   writes into its given storage what it wrote before. *)
let writes_kept name pass =
  test name (fun () ->
      let sink = program name in
      let buffers = filled sink in
      let given =
        List.filter_map
          (fun (n, (p : Ops.param_arg)) ->
            if Ops.op n = Alloc then None else Some p.slot)
          (storage sink)
      in
      let into_given = List.filter (fun (s, _, _) -> List.mem s given) in
      equal (list write)
        (into_given (Tensors.writes ~buffers sink))
        (into_given (Tensors.writes ~buffers (pass name))))

(* Programs Tensors does not run: a call of a kernel or of a compiled schedule,
   and a symbolic shape. *)
let unevaluated =
  [
    "custom_kernel";
    "precompiled_function";
    "precompiled_function_1";
    "gpt_oss_prefill_1";
    "gpt_oss_decode_1";
    "variable_shrink";
    "inline_symbolic";
    "gpt_oss_decode";
  ]

let evaluated =
  List.filter (fun name -> not (List.mem name unevaluated)) programs

let values =
  group "prepare_rangeify › keeps what each recorded program writes"
    (List.map (fun name -> writes_kept name prepared) evaluated)

(* prepare_rangeify: hand-built functions

   Each states one rule, on the CPU: an output of sixteen elements in slot [0],
   inputs in the next slots. *)

let out = Shape.param ~device:cpu ~shape:[ Int 16 ] 0 Float32
let input slot = Shape.param ~device:cpu ~shape:[ Int 16 ] slot Float32
let stores value = Ops.sink [ Ops.after out [ Ops.store out value ] ]
let writes = list (triple int int Dtypes.value)

let copy_body =
  Ops.sink
    [
      Ops.store (input 0) (Shape.param ~device:cpu ~shape:[ Int 16 ] 1 Float32);
    ]

(* The sizes of the storage [u] makes, call-local storage (Op.Alloc). *)
let made_sizes u = List.map Shape.max_numel (nodes Alloc u)

(* The values [u] stores into [dest]. *)
let stored_into dest u =
  List.filter_map
    (fun st -> if Ops.nth st 0 == dest then Some (Ops.nth st 1) else None)
    (nodes Store u)

let outputs =
  let value = Ops.exp2 (input 1) in
  group "prepare_rangeify › outputs"
    [
      test
        "a value computed into new storage of the output's size is computed \
         into the output" (fun () ->
          let scratch = Shape.alloc ~device:cpu [ Int 16 ] Float32 in
          let computed = Ops.after scratch [ Ops.store scratch value ] in
          equal uop (stores value) (prepare (stores computed)));
      test "a materialised value is computed into the output" (fun () ->
          equal uop (stores value)
            (prepare (Ops.sink [ Ops.store out (Ops.v Stage ~src:[ value ]) ])));
      cases ~id:fst "a value that is neither is stored as it is"
        [
          ("a value of other storage", fun () -> stores value);
          (* Ranges decide what to materialise of a value that reads its
             destination. *)
          ( "a value that permutes other storage and reads its destination",
            fun () ->
              let square u = Shape.reshape u (ints [ 4; 4 ]) in
              let value =
                Ops.O.(Shape.permute (square (input 1)) [ 1; 0 ] + square out)
              in
              Ops.sink [ Ops.after out [ Ops.store (square out) value ] ] );
        ]
        (fun (_, sink) ->
          let sink = sink () in
          equal uop sink (prepare sink));
      (* It is stored into new storage of the output's size. *)
      test "a materialised value that reads the output is not placed there"
        (fun () ->
          let reads_out = Ops.O.(out + float 1.) in
          let u =
            prepare
              (Ops.sink [ Ops.store out (Ops.v Stage ~src:[ reads_out ]) ])
          in
          equal (list int) [ 16 ] (made_sizes u));
      test "a value materialised into two outputs is placed in the first"
        (fun () ->
          let other = Shape.param ~device:cpu ~shape:[ Int 16 ] 2 Float32 in
          let staged = Ops.v Stage ~src:[ value ] in
          let u =
            prepare (Ops.sink [ Ops.store out staged; Ops.store other staged ])
          in
          equal (list uop) [ value ] (stored_into out u));
      test
        "a value materialised into two outputs is stored into the second from \
         the first" (fun () ->
          let other = Shape.param ~device:cpu ~shape:[ Int 16 ] 2 Float32 in
          let staged = Ops.v Stage ~src:[ value ] in
          let u =
            prepare (Ops.sink [ Ops.store out staged; Ops.store other staged ])
          in
          equal (list uop)
            [ Ops.after out [ Ops.store out value ] ]
            (stored_into other u));
      test "every read of storage placed in an output reads the output"
        (fun () ->
          let scratch = Shape.alloc ~device:cpu [ Int 16 ] Float32 in
          let other = Shape.param ~device:cpu ~shape:[ Int 16 ] 2 Float32 in
          let computed = Ops.after scratch [ Ops.store scratch value ] in
          let read_by u =
            Ops.after other [ Ops.store other Ops.O.(u + float 1.) ]
          in
          equal uop
            (Ops.sink [ Ops.after out [ Ops.store out value ]; read_by out ])
            (prepare (Ops.sink [ Ops.store out computed; read_by scratch ])));
      test "a value materialised into a view of an output is computed into it"
        (fun () ->
          let half = Shape.shrink out [ Some (Int 0, Int 8) ] in
          let value = Ops.exp2 (flat 8) in
          equal uop
            (Ops.sink [ Ops.after out [ Ops.store half value ] ])
            (prepare (Ops.sink [ Ops.store half (Ops.v Stage ~src:[ value ]) ])));
      (* The store of the output into itself that is left is dropped. *)
      test "an output that stores new storage takes the storage's place"
        (fun () ->
          let fresh = Ops.new_buffer ~slot:5 cpu 16 Float32 in
          equal uop (Ops.sink [ out ])
            (prepare (Ops.sink [ Ops.store out fresh ])));
    ]

let copies_of_input_1 = List.init 16 (fun j -> (0, j, `Float (float_of_int j)))

let inline_calls =
  let r = Ops.range ~axis_type:Loop (Int 4) [ 0 ] in
  let call = Ops.call copy_body [ out; input 1 ] in
  group "prepare_rangeify › inline calls"
    [
      test "an inline call is replaced by its body on its arguments" (fun () ->
          equal uop (stores (input 1)) (prepare (Ops.sink [ call ])));
      test "an argument of another size than its parameter is refused"
        (fun () ->
          let short = Shape.param ~device:cpu ~shape:[ Int 8 ] 1 Float32 in
          rejects (fun () ->
              prepare (Ops.sink [ Ops.call copy_body [ out; short ] ])));
      test "an argument of another type than its parameter is refused"
        (fun () ->
          let ints = Shape.param ~device:cpu ~shape:[ Int 16 ] 1 Int32 in
          rejects (fun () ->
              prepare (Ops.sink [ Ops.call copy_body [ out; ints ] ])));
      test "a shaped argument of a scalar parameter is refused" (fun () ->
          let scalar = Shape.param 1 Float32 in
          let body =
            Ops.sink [ Ops.store (input 0) Ops.O.(input 0 + scalar) ]
          in
          rejects (fun () ->
              prepare (Ops.sink [ Ops.call body [ out; input 1 ] ])));
      (* Two calls of one body inlined: the body's storage, slot 7, is storage
         of its own in each. *)
      test "an inline body's own storage is renamed" (fun () ->
          let body_storage slot =
            Shape.alloc ~slot ~device:cpu [ Int 16 ] Float32
          in
          let body =
            Ops.sink
              [
                Ops.store (input 0)
                  (Ops.after (body_storage 7)
                     [ Ops.store (body_storage 7) (input 1) ]);
              ]
          in
          let inlined dst src =
            let scratch = Shape.alloc ~device:cpu [ Int 16 ] Float32 in
            let held = Ops.after scratch [ Ops.store scratch src ] in
            [ held; Ops.after dst [ Ops.store dst held ] ]
          in
          let expected =
            Ops.sink (inlined out (input 1) @ inlined (input 2) (input 3))
          in
          let u =
            prepare
              (Ops.sink
                 [
                   Ops.call body [ out; input 1 ];
                   Ops.call body [ input 2; input 3 ];
                 ])
          in
          equal uop expected (Uops.numbered_like expected u));
      test
        "an argument that is the first part of larger storage is passed as it \
         is" (fun () ->
          let wide = Shape.param ~device:cpu ~shape:[ Int 32 ] 1 Float32 in
          let first = Shape.shrink wide [ Some (Int 0, Int 16) ] in
          let u = prepare (Ops.sink [ Ops.call copy_body [ out; first ] ]) in
          equal writes copies_of_input_1
            (Tensors.writes
               ~buffers:
                 [ (1, Array.init 32 (fun j -> `Float (float_of_int j))) ]
               u));
      test "an argument of another shape than its parameter is passed flat"
        (fun () ->
          let square = Shape.reshape (input 1) (ints [ 4; 4 ]) in
          let u = prepare (Ops.sink [ Ops.call copy_body [ out; square ] ]) in
          equal writes copies_of_input_1
            (Tensors.writes
               ~buffers:
                 [ (1, Array.init 16 (fun j -> `Float (float_of_int j))) ]
               u));
      test "a call a loop runs is compiled on its own" (fun () ->
          let loop c = Ops.sink [ Ops.after out [ Ops.end_ c [ r ] ] ] in
          equal uop
            (loop (Ops.call ~precompile:true copy_body [ out; input 1 ]))
            (prepare (loop call)));
      test
        "an end of an end of a call is one end over the outer end's ranges, \
         then the inner's" (fun () ->
          let outer = Ops.range ~axis_type:Loop (Int 3) [ 1 ] in
          let compiled = Ops.call ~precompile:true copy_body [ out; input 1 ] in
          equal uop
            (Ops.sink [ Ops.after out [ Ops.end_ compiled [ outer; r ] ] ])
            (prepare
               (Ops.sink
                  [ Ops.after out [ Ops.end_ (Ops.end_ call [ r ]) [ outer ] ] ])));
    ]

(* Calls compiled on their own

   A kernel given as code, or a precompiled call, takes flat storage at each
   argument. Each call copies four floats from its second argument into its
   first. *)

let copy_kernel ps =
  let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
  let at k = Ops.index (List.nth ps k) [ i ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"copy4" ())
    [ Ops.end_ (Ops.store (at 0) (at 1)) [ i ] ]

let kernel_call args = Ops.sink (Shape.custom_kernel args copy_kernel)

let precompiled_call args =
  let p slot = Shape.param ~shape:[ Int 4 ] ~device:cpu slot Float32 in
  let c = Ops.call ~precompile:true (Ops.sink [ Ops.store (p 0) (p 1) ]) args in
  Ops.sink (List.map (fun a -> Ops.after a [ c ]) args)

let call_kinds = [ `Kernel; `Precompiled ]

let call_name = function
  | `Kernel -> "a kernel"
  | `Precompiled -> "a precompiled call"

let call kind args =
  match kind with
  | `Kernel -> kernel_call args
  | `Precompiled -> precompiled_call args

(* The arguments of the one call of [u]. *)
let call_args u =
  match List.filter (fun n -> Ops.op n = Call) (Ops.toposort ~calls:Skip u) with
  | [ c ] -> List.tl (Ops.src c)
  | cs -> failf "%d calls" (List.length cs)

let calls_on_their_own =
  let a = Shape.param ~device:cpu ~shape:[ Int 8 ] 0 Float32 in
  let y = Shape.param ~device:cpu ~shape:[ Int 4 ] 1 Float32 in
  let first_four = Shape.shrink a [ Some (Ops.Int 0, Ops.Int 4) ] in
  let last_four = Shape.shrink a [ Some (Ops.Int 4, Ops.Int 8) ] in
  let row k =
    Shape.shrink
      (Shape.reshape a (ints [ 2; 4 ]))
      [ Some (Int k, Int (k + 1)); None ]
  in
  let runs =
    [
      ("the first four, as they are", first_four, first_four);
      ("the first row, as it is", row 0, row 0);
      ("the last four, as they are", last_four, last_four);
      ( "the second row viewed flat, as the last four",
        Shape.reshape (row 1) [ Int 4 ],
        last_four );
    ]
  in
  let rows =
    List.concat_map (fun k -> List.map (fun r -> (k, r)) runs) call_kinds
  in
  group "prepare_rangeify › calls compiled on their own"
    [
      cases
        ~id:(fun (kind, (run, _, _)) -> call_name kind ^ ", " ^ run)
        "a run of storage, wherever it starts, is taken as one shrink of the \
         storage"
        rows
        (fun (kind, (_, view, run)) ->
          equal (list uop) [ run; y ]
            (call_args (prepare (call kind [ view; y ]))));
      cases ~id:call_name
        "a state of such a run is passed as the run of the state" call_kinds
        (fun kind ->
          let k = Ops.store first_four y in
          let arg = Ops.after first_four [ k ] in
          equal (list uop)
            [
              Shape.shrink (Ops.after a [ k ]) [ Some (Ops.Int 0, Ops.Int 4) ];
              y;
            ]
            (call_args (prepare (call kind [ arg; y ]))));
      (* The call's parameters start on 16 bytes, its argument 4 bytes past. *)
      cases ~id:call_name
        "storage the call only reads, off its parameter's start, is stored \
         into new storage"
        call_kinds (fun kind ->
          let off =
            Shape.param ~phase:4 ~device:cpu ~shape:[ Int 4 ] 2 Float32
          in
          let buf = Shape.alloc ~device:cpu [ Int 4 ] Float32 in
          let copied = Ops.after buf [ Ops.store buf off ] in
          let u = prepare (call kind [ y; off ]) in
          equal (list uop) [ y; copied ]
            (call_args (Uops.numbered_like copied u)));
    ]

(* Stores into broadcasts

   Through an expand that stretches an axis of one element, several elements of
   a view are one cell of its storage: in [row[c] = y[4r + c]] each of [y]'s
   four rows would land on [row]. *)

let broadcast_stores =
  let row = Shape.param ~device:cpu ~shape:[ Int 4 ] 0 Float32 in
  let y = Shape.param ~device:cpu ~shape:(ints [ 4; 4 ]) 1 Float32 in
  let wide u = Shape.expand (Shape.reshape u (ints [ 1; 4 ])) (ints [ 4; 4 ]) in
  let first u = Shape.shrink u [ Some (Ops.Int 0, Ops.Int 1); None ] in
  let into ?gate dest v =
    Ops.sink [ Ops.after row [ Ops.store ?gate dest v ] ]
  in
  let other = Shape.param ~device:cpu ~shape:[ Int 4 ] 2 Float32 in
  let state = Ops.after row [ Ops.store row other ] in
  let deep =
    Shape.expand (Shape.reshape row (ints [ 1; 1; 4 ])) (ints [ 4; 4; 4 ])
  in
  let n = Ops.variable "n" (`Int (Bigint.of_int 2)) (`Int (Bigint.of_int 4)) in
  let tall =
    Shape.expand (Shape.reshape row (ints [ 1; 4 ])) [ Ops.Sym n; Int 4 ]
  in
  let refusals =
    [
      ("the broadcast", into (wide row) y, "of shape (4, 4)");
      ( "its transpose",
        into (Shape.permute (wide row) [ 1; 0 ]) y,
        "of shape (4, 4)" );
      ( "its bitcast",
        into (Ops.bitcast (wide row) Int32) (Ops.bitcast y Int32),
        "of shape (4, 4)" );
      ("its detach", into (Ops.v Detach ~src:[ wide row ]) y, "of shape (4, 4)");
      ( "a gated store",
        into ~gate:(Ops.bool true) (wide row) y,
        "of shape (4, 4)" );
      ("a broadcast of a state", into (wide state) y, "of shape (4, 4)");
      ( "a state of a broadcast",
        into (Ops.after (wide row) [ Ops.store row other ]) y,
        "of shape (4, 4)" );
      ( "its gradient marker",
        into (Ops.v Contiguous_backward ~src:[ wide row ]) y,
        "of shape (4, 4)" );
      ( "an index of the first axis of a broadcast",
        into (Ops.index deep [ Ops.int 1 ]) y,
        "of shape (4, 4)" );
      ("one row of it", into (first (wide row)) (first y), "of shape (1, 4)");
      (* The message prints the size's variable, [n]. *)
      ( "a broadcast to a symbolic size",
        into tall (Shape.const_like tall (`Float 1.)),
        "name='n'" );
    ]
  in
  let iota n = Array.init n (fun j -> `Float (float_of_int j)) in
  group "prepare_rangeify › broadcast stores"
    [
      cases
        ~id:(fun (view, _, _) -> view)
        "a store through a broadcast, or a view of one, is refused" refusals
        (fun (_, sink, shape) ->
          raises_match
            (fun e ->
              Exn.invalid_arg ~sub:"a store into a broadcast of shape" e
              && Exn.invalid_arg ~sub:shape e)
            (fun () -> prepare sink));
      test "an inline call that writes a broadcast argument is refused"
        (fun () ->
          let arg = Shape.reshape (wide row) (ints [ 16 ]) in
          raises_match
            (Exn.invalid_arg ~sub:"a store into a broadcast of shape (16,)")
            (fun () ->
              prepare (Ops.sink [ Ops.call copy_body [ arg; input 1 ] ])));
      test "an expand that only adds axes of one is stored through" (fun () ->
          let y = Shape.param ~device:cpu ~shape:(ints [ 1; 4 ]) 1 Float32 in
          let u = prepare (into (Shape.expand row (ints [ 1; 4 ])) y) in
          equal writes
            (List.init 4 (fun j -> (0, j, `Float (float_of_int j))))
            (Tensors.writes ~buffers:[ (1, iota 4) ] u));
      (* It writes nothing: preparing drops it, as any store of no element. *)
      test "a store through an expand that empties an axis is dropped"
        (fun () ->
          let none = Shape.param ~device:cpu ~shape:(ints [ 0; 4 ]) 1 Float32 in
          let empty =
            Shape.expand (Shape.reshape row (ints [ 1; 4 ])) (ints [ 0; 4 ])
          in
          equal (list uop) [] (nodes Store (prepare (into empty none))));
      test "a transpose of storage is stored through" (fun () ->
          let a = Shape.param ~device:cpu ~shape:(ints [ 4; 4 ]) 0 Float32 in
          let sink =
            Ops.sink [ Ops.after a [ Ops.store (Shape.permute a [ 1; 0 ]) y ] ]
          in
          let transposed k = (4 * (k mod 4)) + (k / 4) in
          equal writes
            (List.init 16 (fun k ->
                 (0, k, `Float (float_of_int (transposed k)))))
            (Tensors.writes ~buffers:[ (1, iota 16) ] (prepare sink)));
    ]

(* prepare_rangeify, from the leaves up *)

let reduced_into total r = Ops.sink [ Ops.after total [ Ops.store total r ] ]

let from_the_leaves =
  let scalar_out = Shape.param ~device:cpu ~shape:[] 0 Float32 in
  group "prepare_rangeify › from the leaves up"
    [
      test
        "a split reduction keeps its first reduction's output within 2^22 \
         elements" (fun () ->
          let wide =
            Shape.param ~device:cpu ~shape:(ints [ 65536; 32768 ]) 1 Float32
          in
          let total = Shape.param ~device:cpu ~shape:[ Int 65536 ] 0 Float32 in
          let u =
            Setting.context split_defaults (fun () ->
                prepare (reduced_into total (Shape.rop wide Add [ 1 ])))
          in
          equal (list int) [ 1 lsl 22 ] (made_sizes u));
      test "a split takes the largest divisor from 256 down" (fun () ->
          let u =
            Setting.context split_defaults (fun () ->
                prepare
                  (reduced_into scalar_out
                     (Shape.rop (flat (128 * 257)) Add [ 0 ])))
          in
          equal (list int) [ 128 ] (made_sizes u));
      (* 65537 is prime: no divisor from 256 to 8 divides it, and its padding to
         65792 splits by 256. *)
      test
        "a split pads with its identity an axis no divisor from 256 to 8 \
         divides" (fun () ->
          let u =
            Setting.context split_defaults (fun () ->
                prepare
                  (reduced_into scalar_out (Shape.rop (flat 65537) Max [ 0 ])))
          in
          equal ~msg:"the first reduction's output" (list int) [ 256 ]
            (made_sizes u);
          mem ~msg:"the maximum's identity" Dtypes.const
            (`Float Float.neg_infinity)
            (List.map Ops.value (nodes Const u)));
      (* 2039, 2053 and 3329 are prime. The longest reduced axis of at least 8 *
         256 elements is padded to a multiple of 256, which the first
         reduction's output holds; a sum's identity, 0, pads with a plain pad.
         3329 pads to 14 * 256, and 2053 * 14 elements stay below the threshold,
         so the first reduction is not split again. *)
      cases
        ~id:(fun (name, _, _) -> name)
        "a split pads only the longest reduced axis of at least 8 times the \
         largest split"
        [
          ("one axis of 2039, too short", [ 2039 ], ([], []));
          ( "axes of 2053 and 3329, the second the longer",
            [ 2053; 3329 ],
            ([ 256 ], [ [ 2053; 3584 ] ]) );
        ]
        (fun (_, shape, (sizes, padded)) ->
          let wide = Shape.param ~device:cpu ~shape:(ints shape) 1 Float32 in
          let all = List.mapi (fun axis _ -> axis) shape in
          let u =
            Setting.context split_defaults (fun () ->
                prepare (reduced_into scalar_out (Shape.rop wide Add all)))
          in
          equal
            (pair (list int) (list (list sint)))
            (sizes, List.map ints padded)
            (made_sizes u, List.map Shape.shape (nodes Pad u)));
      test "a reduction of a symbolic shape is not split" (fun () ->
          let v =
            Ops.variable "v"
              (`Int (Bigint.of_int 1))
              (`Int (Bigint.of_int 65536))
          in
          let part = Shape.shrink (flat 65536) [ Some (Int 0, Sym v) ] in
          let u =
            Setting.context split_defaults (fun () ->
                prepare (reduced_into scalar_out (Shape.rop part Add [ 0 ])))
          in
          equal (list uop) [] (nodes Alloc u));
      test "a detach and a gradient marker are their source" (fun () ->
          let value = Ops.exp2 (input 1) in
          let marked =
            Ops.v Contiguous_backward ~src:[ Ops.v Detach ~src:[ value ] ]
          in
          equal uop (stores value) (prepare (stores marked)));
      cases ~id:fst
        "the items a sink lists lose the movements, indices and bitcasts \
         around them"
        [
          ( "a bitcast of a reshape",
            fun stored ->
              Ops.bitcast (Shape.reshape stored (ints [ 4; 4 ])) Int32 );
          ("an index", fun stored -> Ops.index stored [ Ops.int 3 ]);
        ]
        (fun (_, around) ->
          let stored = Ops.after out [ Ops.store out (input 1) ] in
          equal (list uop) [ stored ]
            (Ops.src (prepare (Ops.sink [ around stored ]))));
      test "a copy to the device its value is on is the value" (fun () ->
          equal uop
            (stores (input 1))
            (prepare (stores (Ops.copy_to_device (input 1) cpu))));
      test "a copy to another device than its destination's is stored first"
        (fun () ->
          let elsewhere =
            Shape.param ~device:"CPU:1" ~shape:[ Int 16 ] 0 Float32
          in
          let copied = Ops.copy_to_device (input 1) "CPU:2" in
          let u =
            prepare
              (Ops.sink [ Ops.after elsewhere [ Ops.store elsewhere copied ] ])
          in
          equal
            (list
               (option (Testable.make ~pp:Ops.pp_device ~equal:String.equal)))
            [ Some "CPU:2" ]
            (List.map Ops.device (nodes Alloc u)));
      test "a materialisation of storage is the storage" (fun () ->
          equal uop
            (stores (input 1))
            (prepare (stores (Ops.v Stage ~src:[ input 1 ]))));
      test "two equal stores into one storage are two states" (fun () ->
          let v = input 1 in
          let first = Ops.after out [ Ops.store out v ] in
          let second = Ops.sink [ Ops.after first [ Ops.store first v ] ] in
          equal uop second (prepare second));
      test "a store into a bitcast of storage stores the value bitcast"
        (fun () ->
          let v = Shape.param ~device:cpu ~shape:[ Int 16 ] 1 Int32 in
          equal uop
            (stores (Ops.bitcast v Float32))
            (prepare (Ops.sink [ Ops.store (Ops.bitcast out Int32) v ])));
      (* Bytes 2 to 5 of two words: the second half of the first and the first
         half of the second. Words are little-endian, 0x01020304 is bytes 04 03
         02 01. *)
      test "a store into a run of bytes inside elements keeps their other bytes"
        (fun () ->
          let words = Shape.param ~device:cpu ~shape:[ Int 4 ] 0 Uint32 in
          let bytes = Shape.param ~device:cpu ~shape:[ Int 4 ] 1 Uint8 in
          let into_bytes =
            Shape.shrink (Ops.bitcast words Uint8) [ Some (Int 2, Int 6) ]
          in
          let word n = `Int (Bigint.of_int n) in
          let buffers =
            [
              (0, [| word 0x01020304; word 0x02040608; word 0; word 0 |]);
              (1, [| word 0xA0; word 0xA1; word 0xA2; word 0xA3 |]);
            ]
          in
          let u =
            prepare
              (Ops.sink [ Ops.after words [ Ops.store into_bytes bytes ] ])
          in
          equal writes
            [ (0, 0, word 0xA1A00304); (0, 1, word 0x0204A3A2) ]
            (Tensors.writes ~buffers u));
      test "a store through bytes that are no run of storage is refused"
        (fun () ->
          let words = Shape.param ~device:cpu ~shape:[ Int 4 ] 0 Uint32 in
          let bytes = Shape.param ~device:cpu ~shape:(ints [ 4; 4 ]) 1 Uint8 in
          let columns =
            Shape.permute
              (Shape.reshape (Ops.bitcast words Uint8) (ints [ 4; 4 ]))
              [ 1; 0 ]
          in
          raises_match (Exn.invalid_arg ~sub:"these bytes are no such run")
            (fun () ->
              prepare (Ops.sink [ Ops.after words [ Ops.store columns bytes ] ])));
      test "a bitcast copied from a disk is computed after the copy" (fun () ->
          let disk =
            Shape.param ~device:"DISK:/tmp/tolk" ~shape:[ Int 64 ] 1 Uint8
          in
          let u =
            prepare (stores (Ops.copy_to_device (Ops.bitcast disk Float32) cpu))
          in
          let on_disk n = Ops.device n = Some "DISK:/tmp/tolk" in
          equal
            (list (pair Dtypes.dtype bool))
            [ (Uint8, false) ]
            (List.filter_map
               (fun n ->
                 if Ops.nth n 1 == disk then
                   Some (Ops.dtype (Ops.nth n 0), on_disk (Ops.nth n 0))
                 else None)
               (nodes Store u)));
      cases ~id:fst
        "a reduction of an empty axis to a non-empty result is the operation's \
         identity"
        [ ("a sum", Op.Add); ("a maximum", Op.Max) ]
        (fun (_, op) ->
          let row = Shape.param ~device:cpu ~shape:[ Int 4 ] 0 Float32 in
          let nothing =
            Shape.param ~device:cpu ~shape:(ints [ 0; 4 ]) 1 Float32
          in
          equal uop
            (reduced_into row
               (Shape.const_like row (Ops.identity_element op Float32)))
            (prepare (reduced_into row (Shape.rop nothing op [ 0 ]))));
      (* The store into no element is a value with an empty axis too. *)
      test "a value with an empty axis is zero" (fun () ->
          let none = Shape.param ~device:cpu ~shape:[ Int 0 ] 0 Float32 in
          let nothing = Shape.param ~device:cpu ~shape:[ Int 0 ] 1 Float32 in
          equal uop
            (Ops.sink [ Ops.const ~dtype:Float32 (`Float 0.) ])
            (prepare
               (Ops.sink
                  [ Ops.after none [ Ops.store none (Ops.exp2 nothing) ] ])));
      cases ~id:fst
        "a no-op among the effects a sink or an After lists is dropped"
        [
          ( "a sink",
            Ops.sink [ Ops.v Noop; Ops.after out [ Ops.store out (input 1) ] ]
          );
          ( "an After",
            Ops.sink [ Ops.after out [ Ops.v Noop; Ops.store out (input 1) ] ]
          );
        ]
        (fun (_, sink) -> equal uop (stores (input 1)) (prepare sink));
    ]

(* [written n] is [n] stores in sequence, each into storage of its own made
   after the value it stores, which reads the storage the previous store wrote.
   Chains of other lengths store into other slots, and share no node. *)
let written n =
  let rec link prev k =
    if k > n then prev
    else
      let value = Ops.exp2 prev in
      let dest = input (n + k) in
      link (Ops.after dest [ Ops.store dest value ]) (k + 1)
  in
  Ops.sink [ link (input n) 1 ]

(* [words f] is the words [f ()] allocates. *)
let words f =
  let before = Gc.minor_words () in
  ignore (Sys.opaque_identity (f ()));
  Gc.minor_words () -. before

(* Work linear in the length is [a * n + b] with [b >= 0], so twice the length
   costs at most twice the work; a walk per store that reaches every store
   before it would cost four times. The tenth of slack covers tables that double
   their capacity at different lengths in the two runs. *)
let work =
  group "prepare_rangeify"
    [
      test
        "twice the stores of a chain into later storage allocate at most 2.2 \
         times the words to prepare" (fun () ->
          let work n =
            let sink = written n in
            words (fun () -> prepare sink)
          in
          let short = work 250 and long = work 500 in
          less float_exact ~than:(2.2 *. short) long);
    ]

(* contiguous_view

   The laws: a view found, [Some (b, offset)], has its elements, in row-major
   order, [b]'s from [offset] on (Tensors), [b] holding distinct elements; and
   shrinks and permutes of storage's elements are found exactly when their
   elements are such a run. Other runs are not always found, as in tinygrad,
   whose simplification of the index does not reach them: flips that cancel
   across a reshape, a read within one copy of an expanded value, and a shrink
   of a reshape of a shrink, whose index is the range plus a constant through
   divisions the simplifier does not recombine. *)

let view = option (pair uop sint)

let contiguous_views =
  let b = flat 32 in
  let x = Shape.reshape b (ints [ 4; 8 ]) in
  let bytes = Ops.bitcast b Uint8 in
  let case name expected u =
    test name (fun () -> equal view expected (Prepare.contiguous_view u))
  in
  let s = Ops.variable "s" (`Int (Bigint.of_int 0)) (`Int (Bigint.of_int 2)) in
  let start = Shape.bind s (`Int (Bigint.of_int 1)) in
  let from_start ~per u =
    Shape.shrink u
      [
        Some
          ( Ops.Sym Ops.O.(start * int per),
            Ops.Sym Ops.O.((start * int per) + int 8) );
      ]
  in
  [
    case "storage is its own view from 0" (Some (b, Int 0)) b;
    case "a reshape keeps the order" (Some (b, Int 0)) x;
    case "a shrink of leading rows starts at their first element"
      (Some (b, Int 8))
      (Shape.shrink x [ Some (Int 1, Int 3); None ]);
    case "a shrink of inner columns skips elements" None
      (Shape.shrink x [ None; Some (Int 2, Int 4) ]);
    case "a permute reorders" None (Shape.permute x [ 1; 0 ]);
    case "a flip reorders" None (Shape.flip x [ 1 ]);
    (let x = Shape.reshape b (ints [ 2; 1; 16 ]) in
     case "flips that cancel across a reshape are not found" None
       (Shape.flip (Shape.reshape (Shape.flip x [ 0 ]) (ints [ 2; 16 ])) [ 0 ]));
    (let x = stored [ 1; 2; 4; 2 ] in
     case "a read within one copy of an expanded value is not found" None
       (Shape.shrink
          (Shape.reshape (Shape.expand x (ints [ 2; 2; 4; 2 ])) (ints [ 32 ]))
          [ Some (Int 6, Int 9) ]));
    (let x = stored [ 3; 3; 3 ] in
     case "a shrink of a reshape of a shrink is not found" None
       (Shape.shrink
          (Shape.reshape
             (Shape.shrink x [ Some (Int 0, Int 2); None; None ])
             (ints [ 18 ]))
          [ Some (Int 6, Int 10) ]));
    case "an expand repeats" None
      (Shape.expand (Shape.reshape b (ints [ 1; 32 ])) (ints [ 2; 32 ]));
    case "a pad adds elements" None (Shape.pad b [ Some (Int 1, Int 0) ]);
    case "a bitcast to bytes is a view of the storage" (Some (b, Int 0)) bytes;
    case "an element is a view at its offset"
      (Some (b, Int 11))
      (Shape.shrink x [ Some (Int 1, Int 2); Some (Int 3, Int 4) ]);
    case "an empty view is no view" None
      (Shape.shrink x [ Some (Int 2, Int 2); None ]);
    case "rows from a bound start start at its rows"
      (Some (b, Shape.ssimplify Ops.O.(start * int 8)))
      (Shape.shrink x
         [ Some (Ops.Sym start, Ops.Sym Ops.O.(start + int 2)); None ]);
    case "bytes from a bound start on whole elements start at their element"
      (Some (b, Sym start))
      (from_start ~per:4 bytes);
    case "bytes from a bound start inside an element are no view" None
      (from_start ~per:1 bytes);
    (let r = Ops.range ~axis_type:Weak (Int 2) [ 0 ] in
     case "rows from a range's start are no view" None
       (Shape.shrink x [ Some (Ops.Sym r, Ops.Sym Ops.O.(r + int 2)); None ]));
    case "elements from an unbound variable's start are no view" None
      (Shape.shrink b [ Some (Ops.Sym s, Ops.Sym Ops.O.(s + int 4)) ]);
    (* In a call's body, a parameter of the call's scalar argument. *)
    (let p =
       Shape.param ~addrspace:Alu
         ~vmin_vmax:(`Int (Bigint.of_int 1), `Int (Bigint.of_int 3))
         5 Int32
     in
     let start = Ops.cast p Weak_int in
     case "elements from a scalar parameter's start start at it"
       (Some (b, Sym start))
       (Shape.shrink b [ Some (Ops.Sym start, Ops.Sym Ops.O.(start + int 4)) ]));
    (let ints = Shape.param ~device:cpu ~shape:[ Int 8 ] 1 Int32 in
     let ordered =
       Ops.after ints
         [ Ops.store ints (Shape.param ~device:cpu ~shape:[ Int 8 ] 2 Int32) ]
     in
     case "a view of storage after its stores is a view of the ordered storage"
       (Some (ordered, Int 2))
       (Shape.shrink ordered [ Some (Int 2, Int 5) ]));
    (let out = Shape.param ~device:cpu ~shape:[ Int 8 ] 1 Int32 in
     let at = Shape.param ~device:cpu ~shape:[ Int 16 ] 2 Float32 in
     let rows =
       Shape.reshape (Shape.arange ~dtype:Weak_int 4) (ints [ 4; 1 ])
     in
     let gather =
       Ops.index (Shape.reshape (stored [ 4; 4 ]) (ints [ 16 ])) [ rows ]
     in
     let halves = Ops.O.((Shape.arange ~dtype:Weak_int 16 + int 1) // int 2) in
     let ordered =
       Ops.after out
         [
           Ops.store
             (Shape.shrink at [ Some (Int 0, Int 4) ])
             (Shape.reshape gather (ints [ 4 ]));
           Ops.store at (Ops.cast halves Float32);
         ]
     in
     case
       "a view of storage after effects is a view of that storage, whatever \
        the effects compute"
       (Some (ordered, Int 2))
       (Shape.shrink ordered [ Some (Int 2, Int 5) ]));
    (let rows =
       Ops.variable "rows" (`Int (Bigint.of_int 1)) (`Int (Bigint.of_int 2))
     in
     let m = Shape.reshape (flat 6) (ints [ 3; 2 ]) in
     case "a view of a symbolic size is no view" None
       (Shape.shrink m
          [ Some (Int 1, Sym Ops.O.(int 1 + rows)); Some (Int 0, Int 2) ]));
    case "two permutes that cancel are a view"
      (Some (b, Int 0))
      (Shape.permute (Shape.permute x [ 1; 0 ]) [ 1; 0 ]);
    case "an expand of an axis of one element is a view"
      (Some (b, Int 0))
      (Shape.expand (Shape.reshape x (ints [ 1; 4; 8 ])) (ints [ 1; 4; 8 ]));
    case "bytes on whole elements start at their element"
      (Some (b, Int 1))
      (Shape.shrink bytes [ Some (Int 4, Int 12) ]);
    case "a bitcast of a view with an axis of one element is a view"
      (Some (b, Int 0))
      (Ops.bitcast (Shape.reshape b (ints [ 1; 32 ])) Uint8);
    case "bytes within an element are a view of the bytes"
      (Some (bytes, Int 2))
      (Shape.shrink bytes [ Some (Int 2, Int 6) ]);
    (let words = Shape.param ~device:cpu ~shape:[ Int 2 ] 1 Uint32 in
     case "bytes of one whole element are a view of the storage"
       (Some (words, Int 1))
       (Shape.shrink (Ops.bitcast words Uint8) [ Some (Int 4, Int 8) ]));
    case "an expanded constant is no view" None
      (Shape.expand
         (Shape.reshape (Ops.const ~dtype:Float32 (`Float 5.0)) (ints [ 1 ]))
         (ints [ 4 ]));
    (let one = Ops.const ~dtype:Int64 (`Int Bigint.one) in
     case "a constant is its own view from 0" (Some (one, Int 0)) one);
    (let one = Ops.const ~dtype:Int64 (`Int Bigint.one) in
     case "a constant of one element is a view of the constant"
       (Some (one, Int 0))
       (Shape.reshape one (ints [ 1 ])));
    (let sum =
       Ops.add b (Shape.param ~device:cpu ~shape:[ Int 32 ] 2 Float32)
     in
     case "a computed value is its own view from 0" (Some (sum, Int 0)) sum);
  ]

let is_run memory elements =
  let n = Array.length elements and m = Array.length memory in
  let at off = Array.for_all2 ( = ) elements (Array.sub memory off n) in
  List.find_opt at (List.init (max 0 (m - n + 1)) Fun.id)

(* [viewed chain] is the view found for the movement of [chain], and the run of
   its storage that its elements are, if they are one. *)
let viewed (shape, ms) =
  let u = moved (shape, ms) in
  let memory = iota (size shape) in
  let elements =
    Array.map
      (function #Dtype.value as v -> v | `Invalid -> `Float nan)
      (Tensors.eval ~buffers:[ (1, memory) ] u)
  in
  let run =
    Option.map
      (fun off -> (flat (size shape), Ops.Int off))
      (is_run memory elements)
  in
  (Prepare.contiguous_view u, run)

let found_is_a_run chain =
  let found, run = viewed chain in
  cover "a view" (Option.is_some found);
  if Option.is_some found then equal view run found

let found_exactly_when_a_run chain =
  let found, run = viewed chain in
  cover "a run" (Option.is_some run);
  cover "no run" (Option.is_none run);
  equal view run found

let contiguous_view =
  group "contiguous_view"
    (contiguous_views
    @ [
        prop "a view found is exactly a run of its storage" chain found_is_a_run;
        (* No run at about 0.18: at 200 cases a run misses one at 0.82^200,
           6e-18. *)
        prop ~count:200
          "shrinks and permutes are found exactly when they are a run"
          (chains [ `Shrink; `Permute ])
          found_exactly_when_a_run;
      ])

(* sequence *)

let sequence =
  let r = Ops.range ~axis_type:Loop (Int 4) [ 0 ] in
  let copy_into dst = Ops.call copy_body [ dst; input 1 ] in
  let cond = Shape.param ~device:cpu 9 Bool in
  let st = Ops.store out (input 1) and st2 = Ops.store (input 2) (input 1) in
  let half lo = Shape.shrink out [ Some (Int lo, Int (lo + 8)) ] in
  group "sequence"
    [
      test "two writes over one state are refused, naming both" (fun () ->
          let v = Shape.param ~device:cpu ~shape:[ Int 8 ] 1 Float32 in
          let both e =
            Exn.invalid_arg ~sub:"the store into elements 0 to 7" e
            && Exn.invalid_arg ~sub:"the store into elements 8 to 15" e
          in
          raises_match both (fun () ->
              Prepare.sequence
                (Ops.sink
                   [
                     Ops.after out [ Ops.store (half 0) v ];
                     Ops.after out [ Ops.store (half 8) v ];
                   ])));
      (* Views of one parameter, each the parameter of a run of its storage, are
         that storage: two Afters over them are two writes over its first
         state. *)
      test
        "Afters over two views of one parameter are two writes over one state"
        (fun () ->
          let w = Shape.param ~shape:[ Int 8 ] ~device:cpu 0 Float32 in
          let view slot =
            let p = Shape.param ~shape:[ Int 4 ] ~device:cpu slot Float32 in
            Ops.replace p ~src:(Ops.src p @ [ w ])
          in
          let v1 = view 1 and v2 = view 2 in
          let const c =
            Shape.expand
              (Shape.reshape (Ops.cast (Ops.float c) Float32) [ Int 1 ])
              [ Int 4 ]
          in
          raises_match (Exn.invalid_arg ~sub:"two writes over one state")
            (fun () ->
              Prepare.sequence
                (Ops.sink
                   [
                     Ops.after v1 [ Ops.store v1 (const 1.) ];
                     Ops.after v2 [ Ops.store v2 (const 2.) ];
                   ])));
      (* The first store reads the state the second makes. *)
      test "an effect listed over a state that needs it is refused" (fun () ->
          let reads_later = Ops.store out (Ops.exp2 (Ops.after out [ st ])) in
          raises_match
            (Exn.invalid_arg
               ~sub:"an effect is listed over a state that needs it") (fun () ->
              Prepare.sequence (Ops.sink [ Ops.after out [ reads_later; st ] ])));
      cases ~id:fst "a loop that runs anything but one call is refused"
        [
          ("an end of two stores", Ops.end_ (Ops.group [ st; st2 ]) [ r ]);
          ( "an end of two calls",
            Ops.end_ (Ops.group [ copy_into out; copy_into (input 2) ]) [ r ] );
          ( "a back edge of a store",
            Ops.after out [ Ops.backedge st ~loop:r ~cond ] );
        ]
        (fun (_, loop) ->
          raises_match
            (Exn.invalid_arg
               ~sub:"a loop outside a kernel runs one call each trip")
            (fun () -> Prepare.sequence (Ops.sink [ loop ])));
      test "a call over a range that is no Loop is refused" (fun () ->
          let weak = Ops.range ~axis_type:Weak (Int 4) [ 1 ] in
          raises_match
            (Exn.invalid_arg ~sub:"make the range with ~axis_type:Loop")
            (fun () ->
              Prepare.sequence
                (Ops.sink
                   [ Ops.after out [ Ops.end_ (copy_into out) [ weak ] ] ])));
      group "keeps what each recorded program writes"
        (List.map
           (fun name ->
             writes_kept name (fun name -> Prepare.sequence (program name)))
           evaluated);
    ]

let () =
  exit
    (run "Tolk.Prepare"
       [
         pm_mops;
         move_index;
         pm_mops_in;
         expand_bitcast;
         goldens;
         values;
         outputs;
         inline_calls;
         calls_on_their_own;
         from_the_leaves;
         broadcast_stores;
         work;
         contiguous_view;
         sequence;
       ])
