open Fehu
open Windtrap

let rng = Rune.Rng.key 42
let value = testable ~pp:Value.pp ~equal:Value.equal ()

(* Helpers *)

let int32_scalar v = Rune.scalar Rune.int32 (Int32.of_int v)

let int32_vec arr =
  Rune.create Rune.int32 [| Array.length arr |] (Array.map Int32.of_int arr)

let float32_vec arr = Rune.create Rune.float32 [| Array.length arr |] arr

let read_float32_vec t =
  let n = (Rune.shape t).(0) in
  let arr : float array = Rune.to_array (Rune.reshape [| n |] t) in
  arr

(* Discrete *)

let test_discrete_default () =
  let s = Space.Discrete.create 3 in
  equal ~msg:"n is 3" int 3 (Space.Discrete.n s);
  equal ~msg:"start is 0" int 0 (Space.Discrete.start s)

let test_discrete_custom_start () =
  let s = Space.Discrete.create ~start:5 3 in
  equal ~msg:"start is 5" int 5 (Space.Discrete.start s);
  equal ~msg:"n is 3" int 3 (Space.Discrete.n s)

let test_discrete_contains () =
  let s = Space.Discrete.create 3 in
  is_true ~msg:"contains 0" (Space.contains s (int32_scalar 0));
  is_true ~msg:"contains 1" (Space.contains s (int32_scalar 1));
  is_true ~msg:"contains 2" (Space.contains s (int32_scalar 2));
  is_false ~msg:"not contains 3" (Space.contains s (int32_scalar 3));
  is_false ~msg:"not contains -1" (Space.contains s (int32_scalar (-1)))

let test_discrete_contains_with_start () =
  let s = Space.Discrete.create ~start:5 3 in
  is_true ~msg:"contains 5" (Space.contains s (int32_scalar 5));
  is_true ~msg:"contains 7" (Space.contains s (int32_scalar 7));
  is_false ~msg:"not contains 4" (Space.contains s (int32_scalar 4));
  is_false ~msg:"not contains 8" (Space.contains s (int32_scalar 8))

let test_discrete_sample () =
  let s = Space.Discrete.create 3 in
  let v, _ = Space.sample s ~rng in
  is_true ~msg:"sample is valid" (Space.contains s v)

let test_discrete_pack_unpack () =
  let s = Space.Discrete.create 3 in
  let v = int32_scalar 2 in
  let packed = Space.pack s v in
  equal ~msg:"pack produces Int 2" value (Value.Int 2) packed;
  let unpacked = Space.unpack s packed in
  is_ok ~msg:"unpack succeeds" unpacked

let test_discrete_unpack_invalid () =
  let s = Space.Discrete.create 3 in
  is_error ~msg:"unpack out of range" (Space.unpack s (Value.Int 5));
  is_error ~msg:"unpack wrong type" (Space.unpack s (Value.String "x"))

let test_discrete_boundary_values () =
  let s = Space.Discrete.create 3 in
  let bvs = Space.boundary_values s in
  equal ~msg:"2 boundary values" int 2 (List.length bvs);
  equal ~msg:"first boundary" value (Value.Int 0) (List.hd bvs);
  equal ~msg:"last boundary" value (Value.Int 2) (List.nth bvs 1)

let test_discrete_boundary_single () =
  let s = Space.Discrete.create 1 in
  let bvs = Space.boundary_values s in
  equal ~msg:"1 boundary for n=1" int 1 (List.length bvs)

let test_discrete_shape () =
  let s = Space.Discrete.create 3 in
  is_none ~msg:"discrete shape is None" (Space.shape s)

let test_discrete_error_zero () =
  raises_invalid_arg "Space.Discrete.create: n must be strictly positive"
    (fun () -> Space.Discrete.create 0)

let test_discrete_error_negative () =
  raises_invalid_arg "Space.Discrete.create: n must be strictly positive"
    (fun () -> Space.Discrete.create (-1))

(* Box *)

let test_box_1d () =
  let s = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let low, high = Space.Box.bounds s in
  equal ~msg:"low" (array (float 0.)) [| 0.0 |] low;
  equal ~msg:"high" (array (float 0.)) [| 10.0 |] high

let test_box_contains () =
  let s = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  is_true ~msg:"mid value" (Space.contains s (float32_vec [| 5.0 |]));
  is_true ~msg:"low bound" (Space.contains s (float32_vec [| 0.0 |]));
  is_true ~msg:"high bound" (Space.contains s (float32_vec [| 10.0 |]));
  is_false ~msg:"below low" (Space.contains s (float32_vec [| -0.1 |]));
  is_false ~msg:"above high" (Space.contains s (float32_vec [| 10.1 |]))

let test_box_sample () =
  let s = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let v, _ = Space.sample s ~rng in
  is_true ~msg:"sample in bounds" (Space.contains s v)

let test_box_pack_unpack () =
  let s = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let v = float32_vec [| 5.0 |] in
  let packed = Space.pack s v in
  let unpacked = Space.unpack s packed in
  is_ok ~msg:"round-trip succeeds" unpacked;
  match unpacked with
  | Ok t ->
      let arr = read_float32_vec t in
      equal ~msg:"value preserved" (float 0.01) 5.0 arr.(0)
  | Error _ -> fail "unreachable"

let test_box_boundary_values () =
  let s = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let bvs = Space.boundary_values s in
  equal ~msg:"2 boundaries" int 2 (List.length bvs)

let test_box_boundary_identical () =
  let s = Space.Box.create ~low:[| 5.0 |] ~high:[| 5.0 |] in
  let bvs = Space.boundary_values s in
  equal ~msg:"1 boundary when identical" int 1 (List.length bvs)

let test_box_shape_1d () =
  let s = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  is_some ~msg:"shape is Some" (Space.shape s);
  equal ~msg:"shape [|1|]" (array int) [| 1 |] (Option.get (Space.shape s))

let test_box_2d () =
  let s = Space.Box.create ~low:[| 0.0; -1.0 |] ~high:[| 1.0; 1.0 |] in
  equal ~msg:"shape [|2|]" (array int) [| 2 |] (Option.get (Space.shape s));
  is_true ~msg:"2d in bounds" (Space.contains s (float32_vec [| 0.5; 0.0 |]));
  is_false ~msg:"2d out of bounds"
    (Space.contains s (float32_vec [| 0.5; 2.0 |]))

let test_box_error_empty () =
  raises_invalid_arg "Space.Box.create: low cannot be empty" (fun () ->
      Space.Box.create ~low:[||] ~high:[||])

let test_box_error_mismatch () =
  raises_invalid_arg
    "Space.Box.create: low and high must have identical lengths" (fun () ->
      Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0; 2.0 |])

let test_box_error_low_gt_high () =
  raises_match ~msg:"low > high raises"
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> Space.Box.create ~low:[| 5.0 |] ~high:[| 1.0 |])

(* Multi_binary *)

let test_mb_contains () =
  let s = Space.Multi_binary.create 3 in
  is_true ~msg:"all zeros" (Space.contains s (int32_vec [| 0; 0; 0 |]));
  is_true ~msg:"all ones" (Space.contains s (int32_vec [| 1; 1; 1 |]));
  is_true ~msg:"mixed" (Space.contains s (int32_vec [| 0; 1; 0 |]));
  is_false ~msg:"value 2 invalid" (Space.contains s (int32_vec [| 0; 2; 0 |]));
  is_false ~msg:"wrong length" (Space.contains s (int32_vec [| 0; 1 |]))

let test_mb_sample () =
  let s = Space.Multi_binary.create 3 in
  let v, _ = Space.sample s ~rng in
  is_true ~msg:"sample valid" (Space.contains s v)

let test_mb_boundary_values () =
  let s = Space.Multi_binary.create 3 in
  let bvs = Space.boundary_values s in
  equal ~msg:"2 boundaries" int 2 (List.length bvs)

let test_mb_shape () =
  let s = Space.Multi_binary.create 3 in
  equal ~msg:"shape [|3|]" (option (array int)) (Some [| 3 |]) (Space.shape s)

let test_mb_error () =
  raises_invalid_arg "Space.Multi_binary.create: n must be strictly positive"
    (fun () -> Space.Multi_binary.create 0)

(* Multi_discrete *)

let test_md_contains () =
  let s = Space.Multi_discrete.create [| 3; 4 |] in
  is_true ~msg:"valid" (Space.contains s (int32_vec [| 0; 0 |]));
  is_true ~msg:"upper valid" (Space.contains s (int32_vec [| 2; 3 |]));
  is_false ~msg:"first oob" (Space.contains s (int32_vec [| 3; 0 |]));
  is_false ~msg:"second oob" (Space.contains s (int32_vec [| 0; 4 |]));
  is_false ~msg:"negative" (Space.contains s (int32_vec [| -1; 0 |]))

let test_md_sample () =
  let s = Space.Multi_discrete.create [| 3; 4 |] in
  let v, _ = Space.sample s ~rng in
  is_true ~msg:"sample valid" (Space.contains s v)

let test_md_shape () =
  let s = Space.Multi_discrete.create [| 3; 4 |] in
  equal ~msg:"shape [|2|]" (option (array int)) (Some [| 2 |]) (Space.shape s)

let test_md_error_empty () =
  raises_invalid_arg "Space.Multi_discrete.create: nvec must not be empty"
    (fun () -> Space.Multi_discrete.create [||])

let test_md_error_zero_element () =
  raises_match ~msg:"nvec element <= 0 raises"
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> Space.Multi_discrete.create [| 3; 0 |])

(* Tuple *)

let test_tuple_contains () =
  let ds = Space.Discrete.create 3 in
  let bs = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
  let s = Space.Tuple.create [ Pack ds; Pack bs ] in
  let valid = [ Value.Int 1; Value.Float_array [| 0.5 |] ] in
  is_true ~msg:"valid tuple" (Space.contains s valid);
  let bad_length = [ Value.Int 1 ] in
  is_false ~msg:"wrong length" (Space.contains s bad_length);
  let bad_value = [ Value.Int 5; Value.Float_array [| 0.5 |] ] in
  is_false ~msg:"invalid element" (Space.contains s bad_value)

let test_tuple_sample () =
  let ds = Space.Discrete.create 3 in
  let bs = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
  let s = Space.Tuple.create [ Pack ds; Pack bs ] in
  let v, _ = Space.sample s ~rng in
  is_true ~msg:"sample valid" (Space.contains s v)

let test_tuple_pack_unpack () =
  let ds = Space.Discrete.create 3 in
  let bs = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
  let s = Space.Tuple.create [ Pack ds; Pack bs ] in
  let v = [ Value.Int 1; Value.Float_array [| 0.5 |] ] in
  let packed = Space.pack s v in
  let unpacked = Space.unpack s packed in
  is_ok ~msg:"round-trip succeeds" unpacked

let test_tuple_empty () =
  let s = Space.Tuple.create [] in
  is_true ~msg:"empty tuple valid" (Space.contains s []);
  is_false ~msg:"non-empty invalid" (Space.contains s [ Value.Int 0 ])

(* Dict *)

let test_dict_contains () =
  let ds = Space.Discrete.create 3 in
  let bs = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
  let s = Space.Dict.create [ ("action", Pack ds); ("obs", Pack bs) ] in
  let valid =
    [ ("action", Value.Int 1); ("obs", Value.Float_array [| 0.5 |]) ]
  in
  is_true ~msg:"valid dict" (Space.contains s valid);
  let missing_key = [ ("action", Value.Int 1) ] in
  is_false ~msg:"missing key" (Space.contains s missing_key);
  let extra_key =
    [
      ("action", Value.Int 1);
      ("obs", Value.Float_array [| 0.5 |]);
      ("extra", Value.Int 0);
    ]
  in
  is_false ~msg:"extra key" (Space.contains s extra_key)

let test_dict_sample () =
  let ds = Space.Discrete.create 3 in
  let s = Space.Dict.create [ ("a", Pack ds) ] in
  let v, _ = Space.sample s ~rng in
  is_true ~msg:"sample valid" (Space.contains s v)

let test_dict_error_duplicate () =
  let ds = Space.Discrete.create 3 in
  raises_match ~msg:"duplicate key raises"
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> Space.Dict.create [ ("a", Pack ds); ("a", Pack ds) ])

(* Text *)

let test_text_contains () =
  let s = Space.Text.create () in
  is_true ~msg:"alpha string" (Space.contains s "hello");
  is_true ~msg:"empty string" (Space.contains s "");
  is_true ~msg:"with digits" (Space.contains s "abc123");
  is_true ~msg:"with space" (Space.contains s "hello world")

let test_text_contains_invalid () =
  let s = Space.Text.create ~charset:"abc" () in
  is_false ~msg:"char outside charset" (Space.contains s "abcd")

let test_text_contains_too_long () =
  let s = Space.Text.create ~max_length:3 () in
  is_false ~msg:"exceeds max_length" (Space.contains s "abcd");
  is_true ~msg:"at max_length" (Space.contains s "abc")

let test_text_sample () =
  let s = Space.Text.create () in
  let v, _ = Space.sample s ~rng in
  is_true ~msg:"sample valid" (Space.contains s v);
  is_true ~msg:"sample non-empty" (String.length v > 0)

let test_text_boundary_values () =
  let s = Space.Text.create () in
  let bvs = Space.boundary_values s in
  equal ~msg:"2 boundaries" int 2 (List.length bvs)

let test_text_error_max_length () =
  raises_invalid_arg "Space.Text.create: max_length must be positive" (fun () ->
      Space.Text.create ~max_length:0 ())

let test_text_error_charset () =
  raises_invalid_arg "Space.Text.create: charset must not be empty" (fun () ->
      Space.Text.create ~charset:"" ())

(* Sequence *)

let test_seq_contains () =
  let ds = Space.Discrete.create 3 in
  let s = Space.Sequence.create ~min_length:1 ~max_length:3 ds in
  let v1 = int32_scalar 0 in
  let v2 = int32_scalar 2 in
  is_true ~msg:"length 1" (Space.contains s [ v1 ]);
  is_true ~msg:"length 3" (Space.contains s [ v1; v2; v1 ]);
  is_false ~msg:"empty" (Space.contains s []);
  is_false ~msg:"too long" (Space.contains s [ v1; v2; v1; v2 ])

let test_seq_contains_unbounded () =
  let ds = Space.Discrete.create 3 in
  let s = Space.Sequence.create ~min_length:0 ds in
  is_true ~msg:"empty is valid" (Space.contains s []);
  is_true ~msg:"long is valid"
    (Space.contains s (List.init 100 (fun _ -> int32_scalar 0)))

let test_seq_sample () =
  let ds = Space.Discrete.create 3 in
  let s = Space.Sequence.create ~min_length:1 ~max_length:5 ds in
  let v, _ = Space.sample s ~rng in
  is_true ~msg:"sample valid" (Space.contains s v)

let test_seq_sample_fixed () =
  let ds = Space.Discrete.create 3 in
  let s = Space.Sequence.create ~min_length:2 ds in
  let v, _ = Space.sample s ~rng in
  equal ~msg:"fixed length 2" int 2 (List.length v)

let test_seq_pack_unpack () =
  let ds = Space.Discrete.create 3 in
  let s = Space.Sequence.create ~min_length:1 ~max_length:3 ds in
  let v = [ int32_scalar 0; int32_scalar 1 ] in
  let packed = Space.pack s v in
  let unpacked = Space.unpack s packed in
  is_ok ~msg:"round-trip succeeds" unpacked

let test_seq_error_min_negative () =
  let ds = Space.Discrete.create 3 in
  raises_invalid_arg "Space.Sequence.create: min_length must be non-negative"
    (fun () -> Space.Sequence.create ~min_length:(-1) ds)

let test_seq_error_max_lt_min () =
  let ds = Space.Discrete.create 3 in
  raises_invalid_arg "Space.Sequence.create: max_length must be >= min_length"
    (fun () -> Space.Sequence.create ~min_length:5 ~max_length:2 ds)

(* Discrete helpers *)

let test_discrete_to_int () =
  let v = Space.Discrete.of_int 5 in
  equal ~msg:"to_int round-trip" int 5 (Space.Discrete.to_int v)

let test_discrete_of_int () =
  let v = Space.Discrete.of_int 3 in
  let s = Space.Discrete.create 5 in
  is_true ~msg:"of_int creates valid element" (Space.contains s v)

(* Spec *)

let test_spec_discrete () =
  let s = Space.Discrete.create ~start:2 4 in
  let sp = Space.spec s in
  equal ~msg:"discrete spec" bool true
    (Space.equal_spec sp (Space.Discrete { start = 2; n = 4 }))

let test_spec_box () =
  let s = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
  let sp = Space.spec s in
  equal ~msg:"box spec" bool true
    (Space.equal_spec sp (Space.Box { low = [| 0.0 |]; high = [| 1.0 |] }))

let test_spec_equal_same () =
  let s1 = Space.Discrete.create 3 in
  let s2 = Space.Discrete.create 3 in
  is_true ~msg:"same spaces equal spec"
    (Space.equal_spec (Space.spec s1) (Space.spec s2))

let test_spec_not_equal_different () =
  let s1 = Space.Discrete.create 3 in
  let s2 = Space.Discrete.create 4 in
  is_false ~msg:"different spaces not equal spec"
    (Space.equal_spec (Space.spec s1) (Space.spec s2))

let test_spec_not_equal_kinds () =
  let s1 = Space.Discrete.create 3 in
  let s2 = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
  is_false ~msg:"different kinds not equal spec"
    (Space.equal_spec (Space.spec s1) (Space.spec s2))

let test_spec_tuple () =
  let ds = Space.Discrete.create 3 in
  let bs = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
  let s = Space.Tuple.create [ Pack ds; Pack bs ] in
  let sp = Space.spec s in
  let expected =
    Space.Tuple
      [
        Space.Discrete { start = 0; n = 3 };
        Space.Box { low = [| 0.0 |]; high = [| 1.0 |] };
      ]
  in
  is_true ~msg:"tuple spec" (Space.equal_spec sp expected)

let test_spec_dict () =
  let ds = Space.Discrete.create 3 in
  let s = Space.Dict.create [ ("a", Pack ds) ] in
  let sp = Space.spec s in
  let expected = Space.Dict [ ("a", Space.Discrete { start = 0; n = 3 }) ] in
  is_true ~msg:"dict spec" (Space.equal_spec sp expected)

(* Tuple.unpack validation *)

let test_tuple_unpack_validates_elements () =
  let ds = Space.Discrete.create 3 in
  let s = Space.Tuple.create [ Pack ds ] in
  (* Value.Int 5 is out of range for Discrete(n=3, start=0) *)
  let bad = Value.List [ Value.Int 5 ] in
  is_error ~msg:"unpack rejects invalid element" (Space.unpack s bad)

let test_tuple_unpack_valid () =
  let ds = Space.Discrete.create 3 in
  let s = Space.Tuple.create [ Pack ds ] in
  let good = Value.List [ Value.Int 1 ] in
  is_ok ~msg:"unpack accepts valid element" (Space.unpack s good)

(* Entry point *)

let () =
  run "Fehu.Space"
    [
      group "Discrete"
        [
          test "default start" test_discrete_default;
          test "custom start" test_discrete_custom_start;
          test "contains valid/invalid" test_discrete_contains;
          test "contains with start" test_discrete_contains_with_start;
          test "sample" test_discrete_sample;
          test "pack/unpack" test_discrete_pack_unpack;
          test "unpack invalid" test_discrete_unpack_invalid;
          test "boundary values" test_discrete_boundary_values;
          test "boundary single" test_discrete_boundary_single;
          test "shape" test_discrete_shape;
          test "error n=0" test_discrete_error_zero;
          test "error n<0" test_discrete_error_negative;
          test "to_int round-trip" test_discrete_to_int;
          test "of_int valid" test_discrete_of_int;
        ];
      group "Box"
        [
          test "1d create and bounds" test_box_1d;
          test "contains" test_box_contains;
          test "sample" test_box_sample;
          test "pack/unpack" test_box_pack_unpack;
          test "boundary values" test_box_boundary_values;
          test "boundary identical" test_box_boundary_identical;
          test "shape 1d" test_box_shape_1d;
          test "2d" test_box_2d;
          test "error empty" test_box_error_empty;
          test "error mismatched lengths" test_box_error_mismatch;
          test "error low > high" test_box_error_low_gt_high;
        ];
      group "Multi_binary"
        [
          test "contains" test_mb_contains;
          test "sample" test_mb_sample;
          test "boundary values" test_mb_boundary_values;
          test "shape" test_mb_shape;
          test "error n=0" test_mb_error;
        ];
      group "Multi_discrete"
        [
          test "contains" test_md_contains;
          test "sample" test_md_sample;
          test "shape" test_md_shape;
          test "error empty" test_md_error_empty;
          test "error element <= 0" test_md_error_zero_element;
        ];
      group "Tuple"
        [
          test "contains" test_tuple_contains;
          test "sample" test_tuple_sample;
          test "pack/unpack" test_tuple_pack_unpack;
          test "empty tuple" test_tuple_empty;
          test "unpack validates elements" test_tuple_unpack_validates_elements;
          test "unpack valid" test_tuple_unpack_valid;
        ];
      group "Dict"
        [
          test "contains" test_dict_contains;
          test "sample" test_dict_sample;
          test "error duplicate keys" test_dict_error_duplicate;
        ];
      group "Text"
        [
          test "contains" test_text_contains;
          test "contains invalid charset" test_text_contains_invalid;
          test "contains too long" test_text_contains_too_long;
          test "sample" test_text_sample;
          test "boundary values" test_text_boundary_values;
          test "error max_length=0" test_text_error_max_length;
          test "error empty charset" test_text_error_charset;
        ];
      group "Sequence"
        [
          test "contains bounded" test_seq_contains;
          test "contains unbounded" test_seq_contains_unbounded;
          test "sample" test_seq_sample;
          test "sample fixed length" test_seq_sample_fixed;
          test "pack/unpack" test_seq_pack_unpack;
          test "error min < 0" test_seq_error_min_negative;
          test "error max < min" test_seq_error_max_lt_min;
        ];
      group "spec"
        [
          test "discrete" test_spec_discrete;
          test "box" test_spec_box;
          test "equal same" test_spec_equal_same;
          test "not equal different" test_spec_not_equal_different;
          test "not equal kinds" test_spec_not_equal_kinds;
          test "tuple" test_spec_tuple;
          test "dict" test_spec_dict;
        ];
    ]
