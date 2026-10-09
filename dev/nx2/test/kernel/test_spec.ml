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

(* C reads, through nx_contract_view, what the accessors give, and 0 for an
   axis an operand lacks and for an absent init, though the view held a call
   with init and every axis before. *)
let law_view_in_c c =
  let v = V.make () in
  let arr s = over (L.contiguous s) in
  let full =
    S.contract
      ~batch:[| (0, 0) |]
      ~contracting:[| (2, 1) |]
      ~acc:(D.Any D.Float32) ~out:(D.Any D.Float32) ~init:true
  in
  let ops = [| arr [| 2; 3; 4 |]; arr [| 2; 4; 5 |]; arr [| 2; 3; 5 |] |] in
  equal ~msg:"the first call groups" bool true
    (V.fill v full ~dst:(arr [| 2; 3; 5 |]) ops);
  let ops, dst = operands c in
  cover "without init" (Option.is_none c.li);
  if V.fill v (spec c) ~dst ops then begin
    let f = Nx_kernel_support.view_fields v in
    let present o = o <> V.Init || Option.is_some c.li in
    let has o x =
      present o
      &&
      match (o, x) with
      | V.A, V.Column | V.B, V.Row | (V.Init | V.Dst), V.Contracted -> false
      | _ -> true
    in
    let axes = V.[| Batch; Row; Column; Contracted |] in
    let operands = V.[| A; B; Init; Dst |] in
    let want =
      Array.concat
        [
          Array.map (V.extent v) axes;
          Array.map
            (fun o -> if present o then V.offset v o else 0)
            operands;
          Array.concat
            (Array.to_list
               (Array.map
                  (fun o ->
                    Array.map
                      (fun x -> if has o x then V.stride v o x else 0)
                      axes)
                  operands));
        ]
    in
    equal (array int) want f
  end

(* A view refuses a call that does not fit its descriptor, and declines one
   whose group does not merge. *)
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
  let misfit ~msg dst ops =
    raises_match ~msg (Exn.invalid_arg ~substring:"fill") (fun () ->
        V.fill v s ~dst ops)
  in
  misfit ~msg:"contracted extents"
    (arr [| 2; 3 |])
    [| arr [| 2; 4 |]; arr [| 5; 3 |] |];
  misfit ~msg:"dst's shape"
    (arr [| 3; 2 |])
    [| arr [| 2; 4 |]; arr [| 4; 3 |] |];
  misfit ~msg:"a's rank" (arr [| 2; 3 |]) [| arr [| 4 |]; arr [| 4; 3 |] |];
  misfit ~msg:"an init too many"
    (arr [| 2; 3 |])
    [| arr [| 2; 4 |]; arr [| 4; 3 |]; arr [| 2; 3 |] |];
  (* A misfit raises even behind a group that does not merge: [a]'s batch
     axes transposed, its contracted extent 4 against [b]'s 5. *)
  let sb =
    S.contract
      ~batch:[| (0, 0); (1, 1) |]
      ~contracting:[| (3, 2) |]
      ~acc:(D.Any D.Float32) ~out:(D.Any D.Float32) ~init:false
  in
  let at =
    Option.get
      (L.move (M.Permute [| 1; 0; 2; 3 |]) (L.contiguous [| 3; 2; 2; 4 |]))
  in
  let a = A.Any (A.v D.Float32 at (Rig.Buffer.create Rig.host 192)) in
  raises_match ~msg:"behind an unmerged group"
    (Exn.invalid_arg ~substring:"fill")
    (fun () ->
      V.fill v sb ~dst:(arr [| 2; 3; 2; 2 |]) [| a; arr [| 2; 3; 5; 2 |] |]);
  (* [b]'s two contracted axes transposed lie as no run. *)
  let s2 =
    S.contract ~batch:[||]
      ~contracting:[| (1, 0); (2, 1) |]
      ~acc:(D.Any D.Float32) ~out:(D.Any D.Float32) ~init:false
  in
  let bt =
    Option.get (L.move (M.Permute [| 1; 0; 2 |]) (L.contiguous [| 3; 4; 5 |]))
  in
  let b = A.Any (A.v D.Float32 bt (Rig.Buffer.create Rig.host 240)) in
  equal ~msg:"no run" bool false
    (V.fill v s2 ~dst:(arr [| 2; 5 |]) [| arr [| 2; 4; 3 |]; b |]);
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

(* Maps *)

module P = Nx_kernel.Prog

let hex s =
  String.concat ""
    (List.init (String.length s) (fun i -> Printf.sprintf "%02x" (Char.code s.[i])))

let map_dtypes = D.[ Any Float32; Any Int8; Any Bool; Any Int4; Any Complex64 ]

type map_case = { ins : D.any array; loads : S.load array; shape : int array }

let pp_load ppf = function
  | S.Plain -> Format.pp_print_string ppf "plain"
  | Padded { pad; _ } ->
      Format.fprintf ppf "padded lo %a hi %a interior %a, %d windows" pp_ints
        pad.lo pp_ints pad.hi pp_ints pad.interior (Array.length pad.windows)

let pp_map_case ppf c =
  Format.fprintf ppf "shape %a, loads %a" pp_ints c.shape
    (Format.pp_print_list pp_load)
    (Array.to_list c.loads)

let load_of (D.Any dt) r =
  let open Gen in
  let small = int_range (-2) 3 in
  let* padded = bool in
  if not padded then constant S.Plain
  else
    let* lo = array ~size:(constant r) small in
    let* hi = array ~size:(constant r) small in
    let* interior = array ~size:(constant r) (int_range 0 2) in
    let* window = option (pair (int_range 0 (max 0 (r - 1))) (int_range 1 3)) in
    let windows =
      match window with
      | Some (axis, size) when r > 0 ->
          [| { Nx_array.Move.axis; size; step = 1 + (size mod 2); dilation = 1 } |]
      | _ -> [||]
    in
    constant
      (S.Padded
         { fill = P.bits dt (D.one dt); pad = { lo; hi; interior; windows } })

let map_case =
  Gen.with_pp pp_map_case
    (let open Gen in
     let* shape = Nx_array_gen.shape in
     let* ins =
       array ~size:(int_range 1 (P.max_operands / 2)) (of_list ~pp:(fun ppf (D.Any d) -> D.pp ppf d) map_dtypes)
     in
     let+ loads =
       let rec go k acc =
         if k = Array.length ins then constant (Array.of_list (List.rev acc))
         else
           let* l = load_of ins.(k) (Array.length shape) in
           go (k + 1) (l :: acc)
       in
       go 0 []
     in
     { ins; loads; shape })

let prog_of c =
  P.v ~ins:c.ins
    (Array.mapi (fun k _ -> P.In k) c.ins)
    ~outs:(Array.mapi (fun k _ -> k) c.ins)

(* What C reads, rendered as the support reader renders it: [family]'s code,
   the program, the axes, each reduction's codes, the loads. *)
let render ?(family = 2) ?(axes = [||]) ?(reductions = [||]) (p : P.t) loads =
  let line = function
    | S.Plain -> "\nplain"
    | Padded { fill; pad } ->
        let nums =
          Array.concat
            [
              pad.lo;
              pad.hi;
              pad.interior;
              Array.concat
                (Array.to_list
                   (Array.map
                      (fun (w : Nx_array.Move.window) ->
                        [| w.axis; w.size; w.step; w.dilation |])
                      pad.windows));
            ]
        in
        Printf.sprintf "\npadded %d %d %s%s" (Array.length pad.lo)
          (Array.length pad.windows)
          (hex (fill ^ String.make (16 - String.length fill) '\000'))
          (String.concat "" (Array.to_list (Array.map (Printf.sprintf " %d") nums)))
  in
  let reduction (code, k, D.Any dt) =
    Printf.sprintf "\nreduction %d %d %d" code k (D.code dt)
  in
  Printf.sprintf "family %d prog %s\naxes" family (hex (p :> string))
  ^ String.concat "" (Array.to_list (Array.map (Printf.sprintf " %d") axes))
  ^ String.concat "" (Array.to_list (Array.map reduction reductions))
  ^ String.concat "" (Array.to_list (Array.map line loads))

let law_map_encoding c =
  let p = prog_of c in
  let s = S.map p ~loads:c.loads in
  equal ~msg:"C reads" string (render p c.loads) (Nx_kernel_support.loop s);
  equal ~msg:"prog" string (p :> string) (S.prog s : P.t :> string);
  equal ~msg:"loads" bool true (S.loads s = c.loads)

(* The shape an operand of shape [x] has once loaded, by the rule. *)
let loaded x = function
  | S.Plain -> Some x
  | Padded { pad; _ } ->
      let padded =
        Array.mapi
          (fun i d ->
            pad.lo.(i) + pad.hi.(i) + d + (pad.interior.(i) * max 0 (d - 1)))
          x
      in
      if Array.exists (fun d -> d < 0) padded then None
      else if pad.windows = [||] then Some padded
      else
        match Nx_array.Move.shape (Window pad.windows) padded with
        | y -> Some y
        | exception Invalid_argument _ -> None

let law_map_shapes c =
  let s = S.map (prog_of c) ~loads:c.loads in
  let ins = Array.map (fun _ -> c.shape) c.ins in
  let shapes = Array.map (loaded c.shape) c.loads in
  let want =
    match shapes.(0) with
    | Some y when Array.for_all (fun z -> z = Some y) shapes ->
        Ok (List.init (Array.length c.ins) (fun _ -> Array.to_list y))
    | _ -> Error ()
  in
  cover "fits" (Result.is_ok want);
  cover "padded" (Array.exists (fun l -> l <> S.Plain) c.loads);
  equal (result (list (list int)) unit) want (result_of (S.shapes s ins))

let test_map_refuses () =
  let f32 = D.Any D.Float32 in
  let p = P.v ~ins:[| f32 |] [| P.In 0 |] ~outs:[| 0 |] in
  let pad = { S.lo = [| 0 |]; hi = [| 0 |]; interior = [| 0 |]; windows = [||] } in
  let fill = P.bits D.Float32 0. in
  let refuses ~msg loads =
    raises_match ~msg Exn.invalid_arg (fun () -> S.map p ~loads)
  in
  refuses ~msg:"a load too many" [| S.Plain; Plain |];
  refuses ~msg:"no load" [||];
  refuses ~msg:"a fill of the wrong width"
    [| Padded { fill = "\000"; pad } |];
  refuses ~msg:"lengths differ"
    [| Padded { fill; pad = { pad with hi = [| 0; 0 |] } } |];
  refuses ~msg:"negative interior"
    [| Padded { fill; pad = { pad with interior = [| -1 |] } } |];
  refuses ~msg:"a window past the rank"
    [|
      Padded
        {
          fill;
          pad =
            {
              pad with
              windows = [| { Nx_array.Move.axis = 1; size = 1; step = 1; dilation = 1 } |];
            };
        };
    |];
  refuses ~msg:"an empty window"
    [|
      Padded
        {
          fill;
          pad =
            {
              pad with
              windows = [| { Nx_array.Move.axis = 0; size = 0; step = 1; dilation = 1 } |];
            };
        };
    |];
  let none = P.v ~ins:[||] [| P.Coord 0 |] ~outs:[| 0 |] in
  equal ~msg:"no operand, no shape" bool true
    (Result.is_error (S.shapes (S.map none ~loads:[||]) [||]))

(* Reductions and scans *)

let reductions_all =
  S.
    [
      Monoid Sum;
      Monoid Prod;
      Monoid Max;
      Monoid Min;
      Monoid Logsumexp;
      Moments;
      Arg Max;
      Arg Min;
    ]

(* nx_spec.h's code of each reduction: its place among the type's cases. *)
let code r = Option.get (List.find_index (( = ) r) reductions_all)

let reduction_name = function
  | S.Monoid Sum -> "Sum"
  | Monoid Prod -> "Prod"
  | Monoid Max -> "Max"
  | Monoid Min -> "Min"
  | Monoid Logsumexp -> "Logsumexp"
  | Moments -> "Moments"
  | Arg Max -> "Arg Max"
  | Arg Min -> "Arg Min"

let pp_reduction ppf (r, k, D.Any dt) =
  Format.fprintf ppf "%s of %d into %a" (reduction_name r) k D.pp dt

(* RFC 0034's domains: Sum and Prod take every dtype but booleans,
   Logsumexp and Moments floats, the extremes every dtype. *)
let accepted r (D.Any dt) =
  match (r, D.kind dt) with
  | S.Monoid (Sum | Prod), D.Boolean -> false
  | (Monoid Logsumexp | Moments), D.Float -> true
  | (Monoid Logsumexp | Moments), _ -> false
  | _ -> true

let results = function S.Monoid _ -> 1 | Moments | Arg _ -> 2

type reduce_case = {
  loop : map_case;
  axes : int array;
  rs : (S.reduction * int * D.any) array;
}

let pp_reduce_case ppf c =
  Format.fprintf ppf "%a, axes %a, %a" pp_map_case c.loop pp_ints c.axes
    (Format.pp_print_list pp_reduction)
    (Array.to_list c.rs)

(* A reduction the encoder takes: axes a subset of the shape's, and one or
   two reductions, each of an output whose dtype it accepts, as many as fit
   beside the loads. *)
let reduce_case =
  Gen.with_pp pp_reduce_case
    (let open Gen in
     let* loop = map_case in
     let rank = Array.length loop.shape in
     let* keep = array ~size:(constant rank) bool in
     let axes =
       Array.of_list (List.filter (fun i -> keep.(i)) (List.init rank Fun.id))
     in
     let one =
       let* k = int_range 0 (Array.length loop.ins - 1) in
       let* r = of_list (List.filter (fun r -> accepted r loop.ins.(k)) reductions_all) in
       let+ dt = of_list ~pp:(fun ppf (D.Any d) -> D.pp ppf d) D.all in
       (r, k, dt)
     in
     let+ drawn = list ~size:(int_range 1 2) one in
     let room = P.max_operands - Array.length loop.ins in
     let rec fit used = function
       | [] -> []
       | ((r, _, _) as x) :: xs ->
           if used + results r > room then [] else x :: fit (used + results r) xs
     in
     { loop; axes; rs = Array.of_list (fit 0 drawn) })

let reduce_of c = S.reduce (prog_of c.loop) ~loads:c.loop.loads ~axes:c.axes c.rs

let law_reduce_encoding c =
  let p = prog_of c.loop in
  let s = reduce_of c in
  let codes = Array.map (fun (r, k, dt) -> (code r, k, dt)) c.rs in
  equal ~msg:"C reads" string
    (render ~family:3 ~axes:c.axes ~reductions:codes p c.loop.loads)
    (Nx_kernel_support.loop s);
  equal ~msg:"prog" string (p :> string) (S.prog s : P.t :> string);
  equal ~msg:"loads" bool true (S.loads s = c.loop.loads);
  equal ~msg:"axes" (array int) c.axes (S.axes s);
  equal ~msg:"reductions" bool true (S.reductions s = c.rs)

let law_scan_encoding c =
  let p = prog_of c.loop in
  let rank = Array.length c.loop.shape in
  let axis = if rank = 0 then 0 else rank - 1 in
  let r, k, dt = c.rs.(0) in
  let r = if r = S.Moments then S.Monoid Max else r in
  let s = S.scan p ~loads:c.loop.loads ~axis (r, k, dt) in
  equal ~msg:"C reads" string
    (render ~family:4 ~axes:[| axis |] ~reductions:[| (code r, k, dt) |] p
       c.loop.loads)
    (Nx_kernel_support.loop s);
  equal ~msg:"axes" (array int) [| axis |] (S.axes s);
  equal ~msg:"reductions" bool true (S.reductions s = [| (r, k, dt) |])

(* The results' shapes by the rule: the loaded shape, without the axes for
   a reduction, one per result; an extreme with an output and no term has
   none. *)
let law_reduce_shapes c =
  let s = reduce_of c in
  let ins = Array.map (fun _ -> c.loop.shape) c.loop.ins in
  let shapes = Array.map (loaded c.loop.shape) c.loop.loads in
  let want =
    match shapes.(0) with
    | Some y when Array.for_all (fun z -> z = Some y) shapes ->
        let kept =
          List.filteri (fun i _ -> not (Array.mem i c.axes)) (Array.to_list y)
        in
        let terms = Array.fold_left (fun n a -> n * y.(a)) 1 c.axes in
        let extreme = function
          | S.Monoid (Max | Min) | Arg _ -> true
          | _ -> false
        in
        if
          Array.exists (fun a -> a >= Array.length y) c.axes
          || terms = 0
             && List.fold_left ( * ) 1 kept > 0
             && Array.exists (fun (r, _, _) -> extreme r) c.rs
        then Error ()
        else
          Ok
            (List.concat_map
               (fun (r, _, _) -> List.init (results r) (fun _ -> kept))
               (Array.to_list c.rs))
    | _ -> Error ()
  in
  cover "fits" (Result.is_ok want);
  cover "no term" (Array.exists (fun a -> c.loop.shape.(a) = 0) c.axes);
  cover "two results" (Array.exists (fun (r, _, _) -> results r = 2) c.rs);
  equal (result (list (list int)) unit) want (result_of (S.shapes s ins))

let test_reduce_refuses () =
  let f32 = D.Any D.Float32 and b8 = D.Any D.Bool and i32 = D.Any D.Int32 in
  let p dt = P.v ~ins:[| dt |] [| P.In 0 |] ~outs:[| 0 |] in
  let refuses ~msg ?(dt = f32) ?(loads = [| S.Plain |]) axes rs =
    raises_match ~msg Exn.invalid_arg (fun () ->
        S.reduce (p dt) ~loads ~axes rs)
  in
  let sum = (S.Monoid Sum, 0, f32) in
  refuses ~msg:"axes out of order" [| 1; 0 |] [| sum |];
  refuses ~msg:"a repeated axis" [| 0; 0 |] [| sum |];
  refuses ~msg:"a negative axis" [| -1 |] [| sum |];
  refuses ~msg:"an axis past the most rank" [| L.max_rank |] [| sum |];
  refuses ~msg:"no reduction" [| 0 |] [||];
  refuses ~msg:"an output the program lacks" [| 0 |] [| (S.Monoid Sum, 1, f32) |];
  refuses ~msg:"a sum of booleans" ~dt:b8 [| 0 |] [| (S.Monoid Sum, 0, b8) |];
  refuses ~msg:"moments of integers" ~dt:i32 [| 0 |] [| (S.Moments, 0, i32) |];
  refuses ~msg:"a load too many" ~loads:[| S.Plain; Plain |] [| 0 |] [| sum |];
  refuses ~msg:"results past the most operands" [| 0 |]
    [| (S.Moments, 0, f32); (S.Arg Max, 0, f32) |];
  raises_match ~msg:"a scan of moments" Exn.invalid_arg (fun () ->
      S.scan (p f32) ~loads:[| S.Plain |] ~axis:0 (S.Moments, 0, f32));
  let y = S.shapes (S.scan (p f32) ~loads:[| S.Plain |] ~axis:1 (S.Arg Max, 0, f32)) in
  equal ~msg:"a scan keeps the shape, twice for Arg"
    (result (list (list int)) unit)
    (Ok [ [ 2; 3 ]; [ 2; 3 ] ])
    (result_of (y [| [| 2; 3 |] |]));
  equal ~msg:"a scan along an axis past the rank" bool true
    (Result.is_error (y [| [| 2 |] |]));
  let max0 = S.reduce (p f32) ~loads:[| S.Plain |] ~axes:[| 1 |] [| (S.Monoid Max, 0, f32) |] in
  equal ~msg:"a maximum of no term" bool true
    (Result.is_error (S.shapes max0 [| [| 2; 0 |] |]));
  equal ~msg:"a maximum with no output" (result (list (list int)) unit)
    (Ok [ [ 0 ] ])
    (result_of (S.shapes max0 [| [| 0; 0 |] |]))

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
    group "maps"
      [
        prop "C reads what map was given, and so do the readers" map_case
          law_map_encoding;
        prop "results have the operands' one shape once loaded" map_case
          law_map_shapes;
        test "refuses loads that do not fit the program" test_map_refuses;
      ];
    group "reductions and scans"
      [
        prop "C reads what reduce was given, and so do the readers"
          reduce_case law_reduce_encoding;
        prop "C reads what scan was given, and so do the readers" reduce_case
          law_scan_encoding;
        prop "results drop the axes, one per result" reduce_case
          law_reduce_shapes;
        test "refuses axes and reductions that do not fit, and states shapes"
          test_reduce_refuses;
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
        test
          "refuses what does not fit, declines an unmerged group and names \
           only an operand's axes"
          test_view_declines;
        prop "C reads what the accessors give, and 0 where they raise" any_case
          law_view_in_c;
        test "fill allocates nothing" test_view_allocates_nothing;
      ];
  ]

let () = exit (run "nx_kernel.spec" tests)
