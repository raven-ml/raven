(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Descriptors: the contraction encoder against what C reads and what its
   accessors give, its shape relation against the rule it states, and the
   grouped view against every element's position by brute force. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module S = Nx_kernel.Spec
module V = Nx_kernel.Spec.Contract_view

let pairs = array (pair int int)

(* Contractions *)

(* A contraction drawn by its groups' extents. [a] is laid out in memory as
   batch, rows, contracted axes and [b] as batch, contracted, columns, each
   C-contiguous or over drawn strides, then permuted: [pa.(i)] is the memory
   axis of [a]'s axis [i]. *)
type case = {
  batch : int array;
  rows : int array;
  contracted : int array;
  columns : int array;
  pa : int array;
  pb : int array;
  la : L.t;
  lb : L.t;
  ly : L.t;
  li : L.t option;
  acc : D.any;
  out : D.any;
}

let pp_case ppf c =
  Format.fprintf ppf
    "batch %a rows %a contracted %a columns %a pa %a pb %a a %a b %a y %a%s"
    pp_ints c.batch pp_ints c.rows pp_ints c.contracted pp_ints c.columns
    pp_ints c.pa pp_ints c.pb L.pp c.la L.pp c.lb L.pp c.ly
    (match c.li with None -> "" | Some l -> Format.asprintf " init %a" L.pp l)

let inverse p =
  let q = Array.make (Array.length p) 0 in
  Array.iteri (fun i m -> q.(m) <- i) p;
  q

(* The pairs of [c]: each group's memory axis, as an axis of [a] and [b]. *)
let spec_pairs c =
  let nb = Array.length c.batch and nr = Array.length c.rows in
  let ia = inverse c.pa and ib = inverse c.pb in
  let batch = Array.init nb (fun k -> (ia.(k), ib.(k))) in
  let contracting =
    Array.init (Array.length c.contracted) (fun k ->
        (ia.(nb + nr + k), ib.(nb + k)))
  in
  (batch, contracting)

let spec c =
  let batch, contracting = spec_pairs c in
  S.contract ~batch ~contracting ~acc:c.acc ~out:c.out ~init:(c.li <> None)

(* The axes no pair names, in axis order. *)
let free r named =
  List.filter (fun ax -> not (List.mem ax named)) (List.init r Fun.id)

(* The result's shape by the rule: the batch extents in pair order, then [a]'s
   free extents, then [b]'s, each in axis order. *)
let result_shape a b batch contracting =
  let named side ps = Array.to_list (Array.map side ps) in
  let fa = free (Array.length a) (named fst batch @ named fst contracting) in
  let fb = free (Array.length b) (named snd batch @ named snd contracting) in
  Array.concat
    [
      Array.map (fun (i, _) -> a.(i)) batch;
      Array.of_list (List.map (fun ax -> a.(ax)) fa);
      Array.of_list (List.map (fun ax -> b.(ax)) fb);
    ]

let extents = Gen.array ~size:(Gen.int_range 0 2) extent
let axis_order n = Gen.map Array.of_list (Gen.permutation (List.init n Fun.id))

(* A layout of shape [s]: C-contiguous, which a view groups, or, unless
   [canonical], over drawn strides. *)
let layout_of ~canonical s =
  if canonical then Gen.constant (L.contiguous s)
  else Gen.one_of [ Gen.constant (L.contiguous s); strided_of s ]

let permuted l p = Option.get (L.move (M.Permute p) l)

(* [p] with the memory axes [lo] to [lo + n - 1] in increasing order of the axes
   that hold them: a free group's axes keep memory order, so they lie as one run
   in a C-contiguous layout. *)
let in_order p lo n =
  let next = ref lo in
  Array.map
    (fun m ->
      if m >= lo && m < lo + n then begin
        let m' = !next in
        incr next;
        m'
      end
      else m)
    p

let accs =
  [ D.Any D.Float32; D.Any D.Float64; D.Any D.Int32; D.Any D.Complex64 ]

let pp_dtype ppf (D.Any dt) = D.pp ppf dt

let case ~canonical =
  let open Gen in
  let layout_of = layout_of ~canonical in
  let* batch = extents in
  let* rows = extents in
  let* contracted = extents in
  let* columns = extents in
  let ma = Array.concat [ batch; rows; contracted ] in
  let mb = Array.concat [ batch; contracted; columns ] in
  let nb = Array.length batch and nr = Array.length rows in
  let nk = Array.length contracted in
  let order p lo n = if canonical then in_order p lo n else p in
  let* pa = map (fun p -> order p nb nr) (axis_order (Array.length ma)) in
  let* pb =
    map
      (fun p -> order p (nb + nk) (Array.length columns))
      (axis_order (Array.length mb))
  in
  let* la = layout_of ma in
  let* lb = layout_of mb in
  let la = permuted la pa and lb = permuted lb pb in
  let c =
    {
      batch;
      rows;
      contracted;
      columns;
      pa;
      pb;
      la;
      lb;
      ly = la;
      li = None;
      acc = D.Any D.Float32;
      out = D.Any D.Float32;
    }
  in
  let b, k = spec_pairs c in
  let y = result_shape (L.shape la) (L.shape lb) b k in
  let* ly = layout_of y in
  let* li = option (layout_of y) in
  let+ acc = of_list ~pp:pp_dtype accs and+ out = of_list ~pp:pp_dtype D.all in
  { c with ly; li; acc; out }

let any_case = Gen.with_pp pp_case (case ~canonical:false)
let canonical_case = Gen.with_pp pp_case (case ~canonical:true)

(* The encoder *)

let code (D.Any dt) = D.code dt

let law_encoding c =
  let s = spec c in
  let batch, contracting = spec_pairs c in
  let flat ps = List.concat_map (fun (i, j) -> [ i; j ]) (Array.to_list ps) in
  let fields =
    [
      1;
      code c.acc;
      code c.out;
      Bool.to_int (c.li <> None);
      Array.length batch;
      Array.length contracting;
    ]
    @ flat batch @ flat contracting
  in
  equal ~msg:"C reads" (list int) fields
    (Array.to_list (Nx_kernel_support.contract_fields s));
  equal ~msg:"batch" pairs batch (S.batch s);
  equal ~msg:"contracting" pairs contracting (S.contracting s);
  equal ~msg:"acc" int (code c.acc) (code (S.acc s));
  equal ~msg:"out" int (code c.out) (code (S.out s));
  equal ~msg:"init" bool (c.li <> None) (S.init s)

let test_encoder_refuses () =
  let refuses ~msg ?(acc = D.Any D.Float32) batch contracting =
    raises_match ~msg Exn.invalid_arg (fun () ->
        S.contract ~batch ~contracting ~acc ~out:(D.Any D.Float32) ~init:false)
  in
  refuses ~msg:"negative" [| (-1, 0) |] [||];
  refuses ~msg:"past max_rank" [||] [| (0, L.max_rank) |];
  refuses ~msg:"a twice" [| (0, 0) |] [| (0, 1) |];
  refuses ~msg:"b twice" [| (0, 1); (1, 1) |] [||];
  List.iter
    (fun acc -> refuses ~msg:"narrow acc" ~acc [||] [| (0, 0) |])
    [
      D.Any D.Bool;
      D.Any D.Bit;
      D.Any D.Float16;
      D.Any D.Bfloat16;
      D.Any D.Float8_e4m3fn;
      D.Any D.Float8_e5m2;
      D.Any D.Float4_e2m1fn;
    ];
  ignore
    (S.contract
       ~batch:[| (31, 31) |]
       ~contracting:[||] ~acc:(D.Any D.Int8) ~out:(D.Any D.Bool) ~init:true)

(* Shapes *)

let shapes_of c =
  let y = L.shape c.ly in
  let ins = [| L.shape c.la; L.shape c.lb |] in
  ((if c.li = None then ins else Array.append ins [| y |]), y)

let result_of = function
  | Ok ys -> Ok (Array.to_list (Array.map Array.to_list ys))
  | Error _ -> Error ()

let law_shapes c =
  let ins, y = shapes_of c in
  equal ~msg:"fits"
    (result (list (list int)) unit)
    (Ok [ Array.to_list y ])
    (result_of (S.shapes (spec c) ins))

let test_shapes_refuse () =
  let s ~init =
    S.contract
      ~batch:[| (0, 0) |]
      ~contracting:[| (2, 1) |]
      ~acc:(D.Any D.Float32) ~out:(D.Any D.Float32) ~init
  in
  let a = [| 2; 3; 4 |] and b = [| 2; 4; 5 |] in
  let fits ~msg spec ins =
    equal ~msg
      (result (list (list int)) unit)
      (Error ())
      (result_of (S.shapes spec ins))
  in
  equal
    (result (list (list int)) unit)
    (Ok [ [ 2; 3; 5 ] ])
    (result_of (S.shapes (s ~init:false) [| a; b |]));
  fits ~msg:"a contracting extent" (s ~init:false) [| a; [| 2; 3; 5 |] |];
  fits ~msg:"a batch extent" (s ~init:false) [| a; [| 3; 4; 5 |] |];
  fits ~msg:"b's rank" (s ~init:false) [| a; [| 2 |] |];
  fits ~msg:"an operand short" (s ~init:false) [| a |];
  fits ~msg:"no init" (s ~init:true) [| a; b |];
  fits ~msg:"init's shape" (s ~init:true) [| a; b; [| 2; 5; 3 |] |]

(* Views *)

(* An array of float32 over [l], on a host buffer that holds its positions. *)
let over l =
  let hi = snd (L.span l) in
  A.Any (A.v D.Float32 l (Rig.Buffer.create Rig.host (4 * (hi + 1))))

let operands c =
  let ops = [| over c.la; over c.lb |] in
  ( (match c.li with None -> ops | Some l -> Array.append ops [| over l |]),
    over c.ly )

(* Each operand's axes of each group, by the rule. *)
let groups c =
  let batch, contracting = spec_pairs c in
  let ra = L.rank c.la and rb = L.rank c.lb in
  let side f ps = Array.to_list (Array.map f ps) in
  let fa = free ra (side fst batch @ side fst contracting) in
  let fb = free rb (side snd batch @ side snd contracting) in
  let nb = Array.length batch and nfa = List.length fa in
  let y =
    [
      (V.Batch, List.init nb Fun.id);
      (V.Row, List.init nfa (fun k -> nb + k));
      (V.Column, List.init (List.length fb) (fun k -> nb + nfa + k));
    ]
  in
  [
    ( V.A,
      c.la,
      [
        (V.Batch, side fst batch);
        (V.Row, fa);
        (V.Contracted, side fst contracting);
      ] );
    ( V.B,
      c.lb,
      [
        (V.Batch, side snd batch);
        (V.Column, fb);
        (V.Contracted, side snd contracting);
      ] );
    (V.Dst, c.ly, y);
  ]
  @ match c.li with None -> [] | Some l -> [ (V.Init, l, y) ]

let pp_operand ppf o =
  Format.pp_print_string ppf
    (match o with V.A -> "a" | V.B -> "b" | V.Init -> "init" | V.Dst -> "dst")

(* Every element of every operand lies where the view places it: its group's
   index, flat in C order over the group's axes, times the group's stride. *)
let check_view c v =
  List.iter
    (fun (o, l, gs) ->
      let s = L.shape l in
      List.iter
        (fun (g, axes) ->
          equal ~msg:"extent" int
            (List.fold_left (fun n ax -> n * s.(ax)) 1 axes)
            (V.extent v g))
        gs;
      List.iter
        (fun idx ->
          let at =
            List.fold_left
              (fun p (g, axes) ->
                let flat =
                  List.fold_left (fun f ax -> (f * s.(ax)) + idx.(ax)) 0 axes
                in
                p + (flat * V.stride v o g))
              (V.offset v o) gs
          in
          equal
            ~msg:(Format.asprintf "%a at %a" pp_operand o pp_ints idx)
            int (position l idx) at)
        (indices s))
    (groups c)

let law_view c =
  let v = V.make () in
  let ops, dst = operands c in
  let grouped = V.fill v (spec c) ~dst ops in
  cover "groups" grouped;
  cover "declines" (not grouped);
  if grouped then check_view c v

let law_canonical c =
  let v = V.make () in
  let ops, dst = operands c in
  equal ~msg:"groups" bool true (V.fill v (spec c) ~dst ops);
  check_view c v

(* A view that has grouped one call declines another that does not fit. *)
let test_view_declines () =
  let s =
    S.contract ~batch:[||]
      ~contracting:[| (1, 0) |]
      ~acc:(D.Any D.Float32) ~out:(D.Any D.Float32) ~init:false
  in
  let arr s = over (L.contiguous s) in
  let v = V.make () in
  equal ~msg:"fits" bool true
    (V.fill v s ~dst:(arr [| 2; 3 |]) [| arr [| 2; 4 |]; arr [| 4; 3 |] |]);
  let declines ~msg dst ops = equal ~msg bool false (V.fill v s ~dst ops) in
  declines ~msg:"contracted extents"
    (arr [| 2; 3 |])
    [| arr [| 2; 4 |]; arr [| 5; 3 |] |];
  declines ~msg:"dst's shape"
    (arr [| 3; 2 |])
    [| arr [| 2; 4 |]; arr [| 4; 3 |] |];
  declines ~msg:"a's rank" (arr [| 2; 3 |]) [| arr [| 4 |]; arr [| 4; 3 |] |];
  declines ~msg:"an init too many"
    (arr [| 2; 3 |])
    [| arr [| 2; 4 |]; arr [| 4; 3 |]; arr [| 2; 3 |] |];
  let a =
    A.Any
      (A.v D.Float32
         (L.v ~strides:[| 1; 8 |] [| 2; 4 |])
         (Rig.Buffer.create Rig.host 128))
  in
  equal ~msg:"strided contraction" bool true
    (V.fill v s ~dst:(arr [| 2; 3 |]) [| a; arr [| 4; 3 |] |]);
  equal int 8 (V.stride v V.A V.Contracted);
  raises_match Exn.invalid_arg (fun () -> V.stride v V.A V.Column);
  raises_match Exn.invalid_arg (fun () -> V.stride v V.Dst V.Contracted);
  raises_match Exn.invalid_arg (fun () -> V.offset v V.Init)

let minor_words f =
  let before = Gc.minor_words () in
  f ();
  Gc.minor_words () -. before

let test_view_allocates_nothing () =
  let s =
    S.contract
      ~batch:[| (0, 0) |]
      ~contracting:[| (2, 1) |]
      ~acc:(D.Any D.Float32) ~out:(D.Any D.Float32) ~init:true
  in
  let arr s = over (L.contiguous s) in
  let ops = [| arr [| 2; 3; 4 |]; arr [| 2; 4; 5 |]; arr [| 2; 3; 5 |] |] in
  let dst = arr [| 2; 3; 5 |] and v = V.make () in
  let calls () =
    for _ = 1 to 1000 do
      ignore (V.fill v s ~dst ops)
    done
  in
  let nothing () = () in
  calls ();
  equal float_exact (minor_words nothing) (minor_words calls)

let tests =
  [
    group "encoder"
      [
        prop "C reads what the encoder writes, and so do the accessors" any_case
          law_encoding;
        test
          "refuses negative, repeated and out-of-range axes and narrow \
           accumulators"
          test_encoder_refuses;
      ];
    group "shapes"
      [
        prop "the result is batch, then a's free axes, then b's" any_case
          law_shapes;
        test "refuses operands that do not fit" test_shapes_refuse;
      ];
    group "contract view"
      [
        prop "places every element where its layout does" any_case law_view;
        prop "groups C-contiguous operands of any axis order" canonical_case
          law_canonical;
        test "declines what does not fit, and names only an operand's axes"
          test_view_declines;
        test "fill allocates nothing" test_view_allocates_nothing;
      ];
  ]

let () = exit (run "nx_kernel.spec" tests)
