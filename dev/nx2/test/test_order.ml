(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's order and counted lengths against models of the sort order: sort,
   argsort and top_k against a stable sort of each slice at every dtype;
   order_key's order against the sort order; lexsort, searchsorted and unique
   against list models of their keys; positions against repeated indices;
   quantile and histogram against their definitions. *)

open Windtrap
module D = Nx_array.Dtype

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

(* The sort order *)

let float_cmp a b =
  match (Float.is_nan a, Float.is_nan b) with
  | true, true -> 0
  | true, false -> 1
  | false, true -> -1
  | false, false ->
      if a = b then compare (Float.sign_bit b) (Float.sign_bit a)
      else compare a b

let complex_cmp (a : Complex.t) (b : Complex.t) =
  let nan (z : Complex.t) = Float.is_nan z.re || Float.is_nan z.im in
  match (nan a, nan b) with
  | true, true -> 0
  | true, false -> 1
  | false, true -> -1
  | false, false ->
      let c = float_cmp a.re b.re in
      if c <> 0 then c else float_cmp a.im b.im

let cmp (type v s) (dt : (v, s) D.t) : v -> v -> int =
  match dt with
  | D.Float64 -> float_cmp
  | D.Float32 -> float_cmp
  | D.Float16 -> float_cmp
  | D.Bfloat16 -> float_cmp
  | D.Float8_e4m3fn -> float_cmp
  | D.Float8_e5m2 -> float_cmp
  | D.Float4_e2m1fn -> float_cmp
  | D.Int64 -> Int64.compare
  | D.Uint64 -> Int64.unsigned_compare
  | D.Int32 -> Int32.compare
  | D.Uint32 -> Int32.unsigned_compare
  | D.Int16 -> Int.compare
  | D.Uint16 -> Int.compare
  | D.Int8 -> Int.compare
  | D.Uint8 -> Int.compare
  | D.Int4 -> Int.compare
  | D.Uint4 -> Int.compare
  | D.Complex128 -> complex_cmp
  | D.Complex64 -> complex_cmp
  | D.Bool -> Bool.compare
  | D.Bit -> Bool.compare

(* [cmp] with [-0.] equal to [+0.], as searchsorted compares. *)
let numeric (type v s) (dt : (v, s) D.t) : v -> v -> int =
  match D.kind dt with
  | D.Float -> fun a b -> cmp dt (a +. 0.) (b +. 0.)
  | _ -> cmp dt

(* Values *)

let specials =
  [ 0.; -0.; nan; -.nan; infinity; neg_infinity; 1.; -1.; 0.5; -2.5; 3.; 448. ]

let value (type v s) (dt : (v, s) D.t) : v Gen.t =
  let open Gen in
  let f = frequency [ (2, of_list specials); (1, float) ] in
  match dt with
  | D.Float64 -> f
  | D.Float32 -> f
  | D.Float16 -> f
  | D.Bfloat16 -> f
  | D.Float8_e4m3fn -> f
  | D.Float8_e5m2 -> f
  | D.Float4_e2m1fn -> f
  | D.Complex128 ->
      let+ re, im = pair f f in
      { Complex.re; im }
  | D.Complex64 ->
      let+ re, im = pair f f in
      { Complex.re; im }
  | D.Int64 -> int64
  | D.Uint64 -> int64
  | D.Int32 -> int32
  | D.Uint32 -> int32
  | D.Int16 -> int_range (-32768) 32767
  | D.Uint16 -> int_range 0 65535
  | D.Int8 -> int_range (-128) 127
  | D.Uint8 -> int_range 0 255
  | D.Int4 -> int_range (-8) 7
  | D.Uint4 -> int_range 0 15
  | D.Bool -> bool
  | D.Bit -> bool

(* [n] values of [dt], half from a small pool so that ties are common. *)
let values dt n =
  let open Gen in
  let* pool = list ~size:(int_range 1 4) (value dt) in
  array ~size:(constant n) (frequency [ (1, value dt); (1, of_list pool) ])

let all = D.all
let keyed = List.filter (fun (D.Any dt) -> not (D.is D.Complex dt)) D.all
let shape r = Gen.array ~size:(Gen.constant r) (Gen.int_range 0 5)

(* An array of [dt] and [s], and its elements as Nx reads them back. *)
let made dt s vs =
  let x = Nx.create dt s vs in
  (x, Nx.to_array x)

(* The flat positions of each slice along [a] of shape [s], in C order of the
   slices. *)
let slices s a =
  let r = Array.length s in
  let stride = Array.make r 1 in
  for i = r - 2 downto 0 do
    stride.(i) <- stride.(i + 1) * s.(i + 1)
  done;
  (* The slices' starts, one per index of the other axes, in C order. *)
  let rec starts i =
    if i = r then [ 0 ]
    else
      let rest = starts (i + 1) in
      if i = a then rest
      else
        List.concat
          (List.init s.(i) (fun c ->
               List.map (fun p -> p + (c * stride.(i))) rest))
  in
  List.map (fun p -> List.init s.(a) (fun k -> p + (k * stride.(a)))) (starts 0)

(* The stable order of [xs] by [c]. *)
let stable c xs =
  List.stable_sort
    (fun i j -> c xs.(i) xs.(j))
    (List.init (Array.length xs) Fun.id)

(* sort, argsort, top_k *)

type sort_case =
  | Sort_case :
      ('v, 's) D.t * int array * int * bool * int * 'v array
      -> sort_case

let sort_case =
  let open Gen in
  with_pp
    (fun ppf (Sort_case (dt, s, a, d, k, _)) ->
      Format.fprintf ppf "%s %a axis %d%s k %d" (D.name dt) pp_ints s a
        (if d then " descending" else "")
        k)
    (let* (D.Any dt) = of_list all in
     let* r = int_range 1 3 in
     let* s = shape r in
     let* a = int_range (-r) (r - 1) in
     let* d = bool in
     let n = s.(if a < 0 then a + r else a) in
     let* k = int_range 0 n in
     let+ vs = values dt (Array.fold_left ( * ) 1 s) in
     Sort_case (dt, s, a, d, k, vs))

let law_sort (Sort_case (dt, s, a, descending, k, vs)) =
  let x, xs = made dt s vs in
  let r = Array.length s in
  let a' = if a < 0 then a + r else a in
  let c = cmp dt in
  let order = if descending then fun u v -> c v u else c in
  let values, positions = Nx.sort ~descending ~axis:a x in
  let tv, tp = Nx.top_k ~k ~axis:a x in
  let vs' = Nx.to_array values and ps = Nx.to_array positions in
  let tvs = Nx.to_array tv and tps = Nx.to_array tp in
  let kept = Array.copy s in
  kept.(a') <- k;
  equal ~msg:"top_k shape" (array int) kept (Nx.shape tv);
  cover "ties"
    (Array.length xs > List.length (List.sort_uniq c (Array.to_list xs)));
  List.iter2
    (fun slice top ->
      let here = Array.of_list (List.map (fun p -> xs.(p)) slice) in
      let want = stable order here in
      let got = List.map (fun p -> Int64.to_int ps.(p)) slice in
      equal ~msg:"positions" (list int) want got;
      List.iter2
        (fun p i -> equal ~msg:"values" int 0 (c vs'.(p) here.(i)))
        slice want;
      let dwant =
        List.filteri (fun j _ -> j < k) (stable (fun u v -> c v u) here)
      in
      equal ~msg:"top_k positions" (list int) dwant
        (List.map (fun p -> Int64.to_int tps.(p)) top);
      List.iter2
        (fun p i -> equal ~msg:"top_k values" int 0 (c tvs.(p) here.(i)))
        top dwant)
    (slices s a') (slices kept a');
  equal ~msg:"argsort" (array int64) ps
    (Nx.to_array (Nx.argsort ~descending ~axis:a x))

let test_sort_cases () =
  let f32 xs = Nx.create Nx.float32 [| Array.length xs |] xs in
  let v, p = Nx.sort (Nx.create Nx.int32 [| 5 |] [| 3l; 1l; 4l; 1l; 5l |]) in
  equal (array int32) [| 1l; 1l; 3l; 4l; 5l |] (Nx.to_array v);
  equal (array int64) [| 1L; 3L; 0L; 2L; 4L |] (Nx.to_array p);
  let v, _ = Nx.sort ~descending:true (f32 [| 1.; nan; -0.; 0. |]) in
  equal int 0
    (compare
       (List.map Int64.bits_of_float [ 1.; 0.; -0. ])
       (List.map Int64.bits_of_float (List.tl (Array.to_list (Nx.to_array v)))));
  equal bool true (Float.is_nan (Nx.to_array v).(0));
  let v, p = Nx.top_k ~k:2 (f32 [| 3.; 1.; 4.; 1.; 5. |]) in
  equal (array float_exact) [| 5.; 4. |] (Nx.to_array v);
  equal (array int64) [| 4L; 2L |] (Nx.to_array p);
  equal (array int) [| 0 |] (Nx.shape (fst (Nx.top_k ~k:0 (f32 [| 1. |]))));
  let z =
    Nx.create Nx.complex128 [| 3 |]
      Complex.[| one; { re = 1.; im = -1. }; zero |]
  in
  equal (array int64) [| 0L; 1L |] (Nx.to_array (snd (Nx.top_k ~k:2 z)));
  invalid ~by:"Nx.top_k" (fun () -> Nx.top_k ~k:2 (f32 [| 1. |]));
  invalid ~by:"Nx.top_k" (fun () -> Nx.top_k ~k:(-1) (f32 [| 1. |]));
  invalid ~by:"Nx.sort" (fun () -> Nx.sort (Nx.scalar Nx.int8 1));
  invalid ~by:"Nx.argsort" (fun () -> Nx.argsort ~axis:1 (f32 [| 1. |]))

(* order_key *)

type key_case = Key_case : ('v, 's) D.t * 'v array -> key_case

let key_case =
  let open Gen in
  with_pp
    (fun ppf (Key_case (dt, vs)) ->
      Format.fprintf ppf "%s [%d]" (D.name dt) (Array.length vs))
    (let* (D.Any dt) = of_list keyed in
     let+ vs = values dt 12 in
     Key_case (dt, vs))

let widths = [ D.Any D.Uint8; D.Any D.Uint16; D.Any D.Uint32; D.Any D.Uint64 ]

let law_key (Key_case (dt, vs)) =
  let x, xs = made dt [| Array.length vs |] vs in
  let c = cmp dt in
  List.iter
    (fun (D.Any kd) ->
      if D.bits kd >= D.bits dt then begin
        let k = Nx.to_array (Nx.cast Nx.uint64 (Nx.order_key kd x)) in
        Array.iteri
          (fun i u ->
            Array.iteri
              (fun j v ->
                equal ~msg:(D.name kd) int
                  (compare (c u v) 0)
                  (compare (Int64.unsigned_compare k.(i) k.(j)) 0))
              xs)
          xs
      end)
    widths

let test_key_cases () =
  let i8 = Nx.create Nx.int8 [| 3 |] [| -128; 0; 127 |] in
  equal (array int) [| 0; 128; 255 |] (Nx.to_array (Nx.order_key Nx.uint8 i8));
  let f = Nx.create Nx.float64 [| 4 |] [| -0.; 0.; nan; -.nan |] in
  equal (array int64)
    [| Int64.max_int; Int64.min_int; -1L; -1L |]
    (Nx.to_array (Nx.order_key Nx.uint64 f));
  let i = Nx.create Nx.int32 [| 3 |] [| -5l; 0l; 7l |] in
  equal (array int64)
    (Nx.to_array (Nx.order_key Nx.uint64 i))
    (Nx.to_array (Nx.order_key Nx.uint64 (Nx.cast Nx.int64 i)));
  let h = Nx.create Nx.float8_e4m3fn [| 4 |] [| -2.; -0.; 0.5; nan |] in
  equal (array int)
    (Nx.to_array (Nx.order_key Nx.uint16 h))
    (Nx.to_array (Nx.order_key Nx.uint16 (Nx.cast Nx.float16 h)));
  invalid ~by:"Nx.order_key" (fun () -> Nx.order_key Nx.uint8 i);
  invalid ~by:"Nx.order_key" (fun () -> Nx.order_key Nx.int64 i);
  invalid ~by:"Nx.order_key" (fun () ->
      Nx.order_key Nx.uint64 (Nx.create Nx.complex64 [| 1 |] [| Complex.one |]))

(* lexsort, searchsorted, unique *)

type rows_case = Rows_case : ('v, 's) D.t * int array * 'v array -> rows_case

(* 1-d keys, or [n; w] rows, of a dtype with keys. *)
let rows_case =
  let open Gen in
  with_pp
    (fun ppf (Rows_case (dt, s, _)) ->
      Format.fprintf ppf "%s %a" (D.name dt) pp_ints s)
    (let* (D.Any dt) = of_list keyed in
     let* n = int_range 0 7 in
     let* s =
       one_of [ constant [| n |]; map (fun w -> [| n; w |]) (int_range 0 3) ]
     in
     let+ vs = values dt (Array.fold_left ( * ) 1 s) in
     Rows_case (dt, s, vs))

(* The keys of [xs] of shape [s], as rows. *)
let rows s xs =
  let w = if Array.length s = 1 then 1 else s.(1) in
  Array.init s.(0) (fun i -> Array.sub xs (i * w) w)

let lex c a b =
  let rec go j =
    if j = Array.length a then 0
    else
      let d = c a.(j) b.(j) in
      if d <> 0 then d else go (j + 1)
  in
  go 0

let law_lexsort (Rows_case (dt, s, vs)) =
  let x, xs = made dt s vs in
  let rs = rows s xs in
  equal (list int)
    (stable (lex (cmp dt)) rs)
    (List.map Int64.to_int (Array.to_list (Nx.to_array (Nx.lexsort x))))

type search_case =
  | Search_case :
      ('v, 's) D.t * int array * 'v array * int array * 'v array * bool
      -> search_case

let search_case =
  let open Gen in
  with_pp
    (fun ppf (Search_case (dt, s, _, t, _, sorted)) ->
      Format.fprintf ppf "%s %a in %a%s" (D.name dt) pp_ints t pp_ints s
        (if sorted then "" else " unsorted"))
    (let* (D.Any dt) = of_list keyed in
     let* m = int_range 0 9 in
     let* rowsy = bool in
     let* w = int_range 0 3 in
     let* q = int_range 0 6 in
     let s = if rowsy then [| m; w |] else [| m |] in
     let t = if rowsy then [| q; w |] else [| 2; q |] in
     let* vs = values dt (Array.fold_left ( * ) 1 s) in
     let* ws = values dt (Array.fold_left ( * ) 1 t) in
     let+ sorted = frequency [ (4, constant true); (1, constant false) ] in
     Search_case (dt, s, vs, t, ws, sorted))

let law_search (Search_case (dt, s, vs, t, ws, sorted)) =
  let c = numeric dt in
  let _, xs = made dt s vs in
  let rs = rows s xs in
  let rs =
    if sorted then
      Array.of_list (List.map (fun i -> rs.(i)) (stable (lex c) rs))
    else rs
  in
  let s_ = Nx.create dt s (Array.concat (Array.to_list rs)) in
  let v, wv = made dt t ws in
  let keys =
    if Array.length s = 2 then rows t wv else Array.map (fun x -> [| x |]) wv
  in
  let m = Array.length rs in
  List.iter
    (fun (side, before) ->
      let got = Nx.to_array (Nx.searchsorted ~side s_ v) in
      equal ~msg:"shape" int (Array.length keys) (Array.length got);
      if sorted then
        equal (array int64)
          (Array.map
             (fun k ->
               Int64.of_int
                 (Array.fold_left
                    (fun n r -> if before (lex c r k) then n + 1 else n)
                    0 rs))
             keys)
          got
      else
        Array.iter
          (fun p -> equal bool true (p >= 0L && p <= Int64.of_int m))
          got)
    [ (`Left, fun d -> d < 0); (`Right, fun d -> d <= 0) ]

let test_search_cases () =
  let f64 xs = Nx.create Nx.float64 [| Array.length xs |] xs in
  let knots = f64 [| 0.; 1.; 1.; 2. |] and x = f64 [| -0.; 1.; 1.5; nan |] in
  equal (array int64) [| 0L; 1L; 3L; 4L |]
    (Nx.to_array (Nx.searchsorted ~side:`Left knots x));
  equal (array int64) [| 1L; 3L; 3L; 4L |]
    (Nx.to_array (Nx.searchsorted ~side:`Right knots x));
  let m = Nx.create Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  invalid ~by:"Nx.searchsorted" (fun () ->
      Nx.searchsorted ~side:`Left m (Nx.create Nx.int32 [| 1 |] [| 1l |]));
  invalid ~by:"Nx.searchsorted" (fun () ->
      Nx.searchsorted ~side:`Left (Nx.scalar Nx.int32 1l) m);
  invalid ~by:"Nx.lexsort" (fun () ->
      Nx.lexsort (Nx.zeros Nx.int8 [| 1; 1; 1 |]));
  equal (array int64) [| 0L; 1L; 2L |]
    (Nx.to_array (Nx.lexsort (Nx.zeros Nx.int8 [| 3; 0 |])))

type unique_case = Unique_case : rows_case * int option -> unique_case

let unique_case =
  let open Gen in
  with_pp
    (fun ppf (Unique_case (Rows_case (dt, s, _), size)) ->
      Format.fprintf ppf "%s %a size %s" (D.name dt) pp_ints s
        (match size with None -> "-" | Some l -> string_of_int l))
    (let* (Rows_case (_, s, _) as r) = rows_case in
     let+ size = option (int_range 0 (s.(0) + 2)) in
     Unique_case (r, size))

let law_unique (Unique_case (Rows_case (dt, s, vs), size)) =
  let x, xs = made dt s vs in
  let rs = rows s xs in
  let c = lex (cmp dt) in
  let groups = ref [] in
  let ids =
    Array.mapi
      (fun i r ->
        let rec find j = function
          | [] ->
              groups := !groups @ [ (i, ref 1) ];
              j
          | (f, n) :: rest ->
              if c rs.(f) r = 0 then (
                incr n;
                j)
              else find (j + 1) rest
        in
        Int64.of_int (find 0 !groups))
      rs
  in
  let g = Nx.unique ?size x in
  let k = List.length !groups in
  let l = Option.value ~default:k size in
  let pad f d =
    Array.init l (fun j -> if j < k then f (List.nth !groups j) else d)
  in
  cover "padded" (l > k);
  cover "cut" (l < k);
  equal ~msg:"ids" (array int64) ids (Nx.to_array g.ids);
  equal ~msg:"count" int64 (Int64.of_int k) (Nx.item [] g.count);
  equal ~msg:"first" (array int64)
    (pad (fun (f, _) -> Int64.of_int f) (Int64.of_int s.(0)))
    (Nx.to_array g.first);
  equal ~msg:"counts" (array int64)
    (pad (fun (_, n) -> Int64.of_int !n) 0L)
    (Nx.to_array g.counts)

type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

let mark i ~by op = Nx.Prim.results ~by (fun _ f -> Nx.Prim.traced i f Mark) op

let test_unique_cases () =
  let g = Nx.unique (Nx.create Nx.int32 [| 6 |] [| 7l; 3l; 7l; 7l; 5l; 3l |]) in
  equal (array int64) [| 0L; 1L; 0L; 0L; 2L; 1L |] (Nx.to_array g.ids);
  equal (array int64) [| 0L; 1L; 4L |] (Nx.to_array g.first);
  equal (array int64) [| 3L; 2L; 1L |] (Nx.to_array g.counts);
  let f = Nx.create Nx.float32 [| 5 |] [| nan; -0.; 0.; -.nan; 0. |] in
  equal (array int64) [| 0L; 1L; 2L; 0L; 2L |] (Nx.to_array (Nx.unique f).ids);
  let e = Nx.unique (Nx.zeros Nx.int8 [| 0 |]) in
  equal int64 0L (Nx.item [] e.count);
  equal (array int) [| 0 |] (Nx.shape e.first);
  invalid ~by:"Nx.unique" (fun () -> Nx.unique ~size:(-1) f);
  invalid ~by:"Nx.unique" (fun () ->
      Nx.unique (Nx.create Nx.complex64 [| 1 |] [| Complex.one |]));
  let x = Nx.create Nx.int32 [| 3 |] [| 1l; 1l; 2l |] in
  Nx.Prim.interpret ~name:"test.order" Values mark (fun i ->
      let t = Nx.Prim.traced i (Nx.Prim.form x) Mark in
      raises_match
        (Exn.invalid_arg
           ~substring:"Nx.unique: the value is traced by test.order") (fun () ->
          Nx.unique t);
      raises_match
        (Exn.invalid_arg
           ~substring:"Nx.positions: the value is traced by test.order")
        (fun () -> Nx.positions t);
      ignore (Nx.unique ~size:2 t);
      ignore (Nx.positions ~size:4 t))

(* Groups over many keys: 70,000 int32 keys of three values, the later
   two first appearing past 25,000. *)
let test_unique_long () =
  let n = 70_000 and cut = 25_000 in
  let key i = if i < cut then 5l else if i mod 3 = 0 then -3l else 9l in
  let g = Nx.unique (Nx.create Nx.int32 [| n |] (Array.init n key)) in
  let id i = if i < cut then 0L else if i mod 3 = 0 then 2L else 1L in
  equal ~msg:"ids" (array int64) (Array.init n id) (Nx.to_array g.ids);
  equal ~msg:"first" (array int64) [| 0L; 25_000L; 25_002L |] (Nx.to_array g.first);
  equal ~msg:"counts" (array int64) [| 25_000L; 30_000L; 15_000L |]
    (Nx.to_array g.counts);
  equal ~msg:"count" int64 3L (Nx.item [] g.count)

(* positions *)

type count_case =
  | Count_case : ('v, 's) D.t * 'v array * int option -> count_case

let count_case =
  let open Gen in
  with_pp
    (fun ppf (Count_case (dt, vs, size)) ->
      Format.fprintf ppf "%s [%d] size %s" (D.name dt) (Array.length vs)
        (match size with None -> "-" | Some l -> string_of_int l))
    (let* size = option (int_range 0 12) in
     let* n = int_range 0 8 in
     one_of
       [
         (let+ vs = array ~size:(constant n) bool in
          Count_case (D.Bool, vs, size));
         (let+ vs = array ~size:(constant n) (int_range (-2) 3) in
          Count_case (D.Int8, vs, size));
         (let+ vs = array ~size:(constant n) (int_range 0 3) in
          Count_case (D.Uint4, vs, size));
         (let+ vs = array ~size:(constant n) (int32_range (-2l) 3l) in
          Count_case (D.Int32, vs, size));
         (let+ vs =
            array ~size:(constant n)
              (one_of [ int64_range 0L 3L; constant Int64.min_int ])
          in
          Count_case (D.Uint64, vs, size));
       ])

let count_of (type v s) (dt : (v, s) D.t) (v : v) =
  match dt with
  | D.Bool -> if v then 1 else 0
  | D.Int8 -> Stdlib.max v 0
  | D.Uint4 -> v
  | D.Int32 -> Stdlib.max (Int32.to_int v) 0
  | D.Uint64 -> if Int64.compare v 0L < 0 then 0 else Int64.to_int v
  | _ -> assert false

let law_positions (Count_case (dt, vs, size)) =
  let c = Nx.create dt [| Array.length vs |] vs in
  let n = Array.length vs in
  let want =
    List.concat
      (List.mapi
         (fun i v -> List.init (count_of dt v) (fun _ -> Int64.of_int i))
         (Array.to_list vs))
  in
  let want =
    match size with
    | None -> want
    | Some l ->
        List.init l (fun j ->
            if j < List.length want then List.nth want j else Int64.of_int n)
  in
  equal (list int64) want (Array.to_list (Nx.to_array (Nx.positions ?size c)))

let test_positions_cases () =
  equal (array int64) [| 1L; 3L |]
    (Nx.to_array
       (Nx.positions (Nx.create Nx.bool [| 4 |] [| false; true; false; true |])));
  equal (array int64) [| 0L; 0L; 2L |]
    (Nx.to_array (Nx.positions (Nx.create Nx.int32 [| 3 |] [| 2l; 0l; 1l |])));
  equal (array int64) [| 1L; 2L; 2L |]
    (Nx.to_array
       (Nx.positions ~size:3 (Nx.create Nx.bool [| 2 |] [| false; true |])));
  invalid ~by:"Nx.positions" (fun () ->
      Nx.positions (Nx.zeros Nx.float32 [| 2 |]));
  invalid ~by:"Nx.positions" (fun () ->
      Nx.positions (Nx.zeros Nx.bool [| 2; 2 |]));
  invalid ~by:"Nx.positions" (fun () ->
      Nx.positions ~size:(-1) (Nx.zeros Nx.bool [| 2 |]))

(* Positions of many: 9000 booleans all true, and counts summing past
   9000, with [size] and without. *)
let test_positions_long () =
  let n = 9000 in
  equal ~msg:"booleans" (array int64) (Array.init n Int64.of_int)
    (Nx.to_array (Nx.positions (Nx.create Nx.bool [| n |] (Array.make n true))));
  let c = Nx.create Nx.int32 [| 3 |] [| 5000l; 4000l; 1l |] in
  let want = Array.init 9001 (fun j -> if j < 5000 then 0L else if j < 9000 then 1L else 2L) in
  equal ~msg:"counts" (array int64) want (Nx.to_array (Nx.positions c));
  equal ~msg:"counts, sized" (array int64) (Array.append want [| 3L; 3L |])
    (Nx.to_array (Nx.positions ~size:9003 c))

(* argwhere *)

(* The coordinates of [s]'s index [i] in C order. *)
let coordinates s i =
  let r = Array.length s in
  let c = Array.make r 0 and i = ref i in
  for j = r - 1 downto 0 do
    c.(j) <- !i mod s.(j);
    i := !i / s.(j)
  done;
  c

let argwhere_case =
  let open Gen in
  let shape =
    array ~size:(int_range 0 3)
      (frequency [ (1, constant 0); (1, constant 1); (3, int_range 2 4) ])
  in
  let element = of_list [ 0.; -0.; Float.nan; 1.; -2.5 ] in
  let* s = shape in
  let n = Array.fold_left ( * ) 1 s in
  let* vs = array ~size:(constant n) element in
  let+ size = option (int_range 0 (n + 2)) in
  (s, vs, size)

(* Each row is the coordinates of an element other than zero, NaN included,
   in C order; with [size], pad rows of the extents. *)
let law_argwhere (s, vs, size) =
  let r = Array.length s in
  cover "0-d" (r = 0);
  cover "an empty value" (Array.length vs = 0 && r > 0);
  cover "padded" (match size with Some l -> l > 0 | None -> false);
  let rows =
    List.concat
      (List.mapi
         (fun i v -> if v = 0. then [] else [ coordinates s i ])
         (Array.to_list vs))
  in
  let rows =
    match size with
    | None -> rows
    | Some l ->
        List.init l (fun j -> if j < List.length rows then List.nth rows j else s)
  in
  let y = Nx.argwhere ?size (Nx.create Nx.float64 s vs) in
  equal ~msg:"shape" (array int) [| List.length rows; r |] (Nx.shape y);
  let want = List.concat_map (fun c -> List.map Int64.of_int (Array.to_list c)) rows in
  equal ~msg:"rows" (list int64) want (Array.to_list (Nx.to_array y))

let test_argwhere_cases () =
  let t = Nx.create Nx.int32 [| 2; 3 |] [| 0l; 1l; 0l; 2l; 0l; 3l |] in
  equal (array int64) [| 0L; 1L; 1L; 0L; 1L; 2L |] (Nx.to_array (Nx.argwhere t));
  equal (array int) [| 3; 2 |] (Nx.shape (Nx.argwhere t));
  equal (array int) [| 1; 0 |] (Nx.shape (Nx.argwhere (Nx.scalar Nx.bool true)));
  equal (array int) [| 0; 0 |] (Nx.shape (Nx.argwhere (Nx.scalar Nx.bool false)));
  invalid ~by:"Nx.argwhere" (fun () -> Nx.argwhere ~size:(-1) t);
  Nx.Prim.interpret ~name:"test.order" Values mark (fun i ->
      let x = Nx.Prim.traced i (Nx.Prim.form t) Mark in
      raises_match
        (Exn.invalid_arg
           ~substring:"Nx.argwhere: the value is traced by test.order")
        (fun () -> Nx.argwhere x);
      ignore (Nx.argwhere ~size:2 x))

(* nonzero *)

(* Column [j] of argwhere's rows, a pad entry the axis' extent; a 0-d value
   has no axis to give. *)
let law_nonzero (s, vs, size) =
  let r = Array.length s in
  let t = Nx.create Nx.float64 s vs in
  if r = 0 then invalid ~by:"Nx.nonzero" (fun () -> Nx.nonzero ?size t)
  else
    let rows = Nx.to_array (Nx.argwhere ?size t) in
    let k = Array.length rows / r in
    let cs = Nx.nonzero ?size t in
    equal ~msg:"axes" int r (List.length cs);
    List.iteri
      (fun j c ->
        equal ~msg:"column" (array int64)
          (Array.init k (fun i -> rows.((i * r) + j)))
          (Nx.to_array c))
      cs

let test_nonzero_cases () =
  let t = Nx.create Nx.int32 [| 2; 3 |] [| 0l; 1l; 0l; 2l; 0l; 3l |] in
  (match Nx.nonzero t with
  | [ r; c ] ->
      equal (array int64) [| 0L; 1L; 1L |] (Nx.to_array r);
      equal (array int64) [| 1L; 0L; 2L |] (Nx.to_array c)
  | _ -> fail "two axes");
  (match Nx.nonzero ~size:4 t with
  | [ r; c ] ->
      equal (array int64) [| 0L; 1L; 1L; 2L |] (Nx.to_array r);
      equal (array int64) [| 1L; 0L; 2L; 3L |] (Nx.to_array c)
  | _ -> fail "two axes");
  invalid ~by:"Nx.nonzero" (fun () -> Nx.nonzero (Nx.scalar Nx.bool true));
  invalid ~by:"Nx.nonzero" (fun () -> Nx.nonzero ~size:(-1) t);
  Nx.Prim.interpret ~name:"test.order" Values mark (fun i ->
      let x = Nx.Prim.traced i (Nx.Prim.form t) Mark in
      raises_match
        (Exn.invalid_arg
           ~substring:"Nx.nonzero: the value is traced by test.order")
        (fun () -> Nx.nonzero x);
      ignore (Nx.nonzero ~size:2 x))

(* extract *)

type extract_case =
  | Extract_case : {
      dt : ('v, 's) D.t;
      cond : float array;
      boolean : bool;
      xs : int array;
      size : int option;
      fill : bool;
    }
      -> extract_case

(* A condition of floats or booleans and an operand of every dtype, of one
   number of elements in two shapes. *)
let extract_case =
  let open Gen in
  with_pp
    (fun ppf (Extract_case c) ->
      Format.fprintf ppf "%s [%d] by %s, size %s%s" (D.name c.dt)
        (Array.length c.cond)
        (if c.boolean then "bool" else "float64")
        (match c.size with None -> "-" | Some l -> string_of_int l)
        (if c.fill then ", filled" else ""))
    (let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all in
     let* n = int_range 0 6 in
     let* boolean = bool in
     let* cond = array ~size:(constant n) (of_list [ 0.; -0.; Float.nan; 1.; -2.5 ]) in
     let* xs = array ~size:(constant n) (int_range 1 9) in
     let* size = option (int_range 0 (n + 2)) in
     let+ fill = bool in
     Extract_case { dt; cond; boolean; xs; size; fill })

let law_extract (Extract_case c) =
  let n = Array.length c.cond in
  let held i = if c.boolean then c.cond.(i) <> 0. else not (c.cond.(i) = 0.) in
  let x =
    Nx.cast c.dt (Nx.create Nx.int64 [| 1; n |] (Array.map Int64.of_int c.xs))
  in
  let xv = Nx.to_array x in
  let fill = if c.fill then Some (D.one c.dt) else None in
  let pad = Option.value fill ~default:(D.zero c.dt) in
  let picked = List.filter held (List.init n Fun.id) in
  cover "padded" (match c.size with Some l -> l > List.length picked | None -> false);
  cover "cut" (match c.size with Some l -> l < List.length picked | None -> false);
  let want =
    match c.size with
    | None -> List.map (fun i -> xv.(i)) picked
    | Some l ->
        List.init l (fun j ->
            if j < List.length picked then xv.(List.nth picked j) else pad)
  in
  let cond = Nx.create Nx.float64 [| n |] c.cond in
  let y =
    if c.boolean then Nx.extract ?size:c.size ?fill (Nx.not_equal_s cond 0.) x
    else Nx.extract ?size:c.size ?fill cond x
  in
  let w = Testable.make ~pp:(D.pp_value c.dt) ~equal:( = ) in
  equal (list w) want (Array.to_list (Nx.to_array y))

let test_extract_cases () =
  let x = Nx.create Nx.int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  equal (array int32) [| 3l; 4l; 5l; 6l |]
    (Nx.to_array (Nx.extract (Nx.greater_s x 2l) x));
  equal (array int32) [| 1l; 3l; -1l |]
    (Nx.to_array
       (Nx.extract ~size:3 ~fill:(-1l)
          (Nx.create Nx.int8 [| 6 |] [| 1; 0; 2; 0; 0; 0 |])
          x));
  invalid ~by:"Nx.extract" (fun () -> Nx.extract (Nx.zeros Nx.bool [| 5 |]) x);
  invalid ~by:"Nx.extract" (fun () ->
      Nx.extract ~size:(-1) (Nx.zeros Nx.bool [| 6 |]) x);
  invalid ~by:"Nx.extract" (fun () ->
      Nx.extract ~fill:300 (Nx.zeros Nx.bool [| 2 |]) (Nx.zeros Nx.uint8 [| 2 |]));
  Nx.Prim.interpret ~name:"test.order" Values mark (fun i ->
      let c = Nx.Prim.traced i (Nx.Prim.form (Nx.zeros Nx.bool [| 6 |])) Mark in
      raises_match
        (Exn.invalid_arg
           ~substring:"Nx.extract: the value is traced by test.order")
        (fun () -> Nx.extract c x);
      ignore (Nx.extract ~size:2 c x))

(* quantile *)

type quantile_case =
  | Quantile_case :
      (float, 's) D.t
      * float
      * int array
      * int option
      * float array
      * float array
      -> quantile_case

let quantile_case =
  let open Gen in
  with_pp
    (fun ppf (Quantile_case (dt, _, s, a, qs, _)) ->
      Format.fprintf ppf "%s %a axis %s, %d probabilities" (D.name dt) pp_ints s
        (match a with None -> "-" | Some a -> string_of_int a)
        (Array.length qs))
    (let case (type s) (dt : (float, s) D.t) eps =
       let* r = int_range 1 3 in
       let* s = array ~size:(constant r) (int_range 1 4) in
       let* a = option (int_range 0 (r - 1)) in
       let* qs =
         array ~size:(int_range 0 4)
           (one_of [ float_range 0. 1.; of_list [ 0.; 0.5; 1. ] ])
       in
       let+ vs =
         array
           ~size:(constant (Array.fold_left ( * ) 1 s))
           (frequency [ (6, float_range (-100.) 100.); (1, of_list specials) ])
       in
       Quantile_case (dt, eps, s, a, qs, vs)
     in
     one_of
       [
         case D.Float64 0x1p-52; case D.Float32 0x1p-22; case D.Float16 0x1p-10;
       ])

let law_quantile (Quantile_case (dt, eps, s, a, qs, vs)) =
  let x, xs = made dt s vs in
  (* The least subnormal of the dtype, a subnormal result's error. *)
  let least = (D.float_format dt).min_normal *. (D.float_format dt).epsilon in
  let s', a' =
    match a with None -> ([| Array.length xs |], 0) | Some a -> (s, a)
  in
  let got = Nx.quantile ?axis:a qs x in
  let rest =
    Array.of_list (List.filteri (fun i _ -> i <> a') (Array.to_list s'))
  in
  equal ~msg:"shape" (array int)
    (Array.append [| Array.length qs |] rest)
    (Nx.shape got);
  let g = Nx.to_array got in
  let sl = slices s' a' in
  let per = List.length sl in
  Array.iteri
    (fun j q ->
      List.iteri
        (fun i slice ->
          let sorted =
            List.sort float_cmp (List.map (fun p -> xs.(p)) slice)
            |> Array.of_list
          in
          let n = Array.length sorted in
          let h = q *. Float.of_int (n - 1) in
          let lo = Float.to_int (Float.floor h) in
          let u = sorted.(lo) and v = sorted.(Stdlib.min (lo + 1) (n - 1)) in
          (* The fraction as the interpolation's format holds it. *)
          let f = h -. Float.floor h in
          let f =
            if D.bits dt = 64 then f
            else Int32.float_of_bits (Int32.bits_of_float f)
          in
          let want = if f = 0. || u = v then u else u +. (f *. (v -. u)) in
          let r = g.((j * per) + i) in
          if Float.is_nan want then equal ~msg:"NaN" bool true (Float.is_nan r)
          else if f = 0. || u = v || not (Float.is_finite want) then
            equal ~msg:"exact" int 0 (float_cmp want r)
          else
            at_most ~msg:"error" float_exact
              ~than:((eps *. (Float.abs u +. Float.abs v)) +. least)
              (Float.abs (r -. want)))
        sl)
    qs

let test_quantile_cases () =
  let x = Nx.create Nx.float64 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
  equal (array float_exact) [| 1.; 3.; 4.; 5. |]
    (Nx.to_array (Nx.quantile [| 0.; 0.5; 0.75; 1. |] x));
  let m = Nx.create Nx.float64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  equal (array float_exact) [| 1.5; 3.5 |]
    (Nx.to_array (Nx.quantile ~axis:1 [| 0.5 |] m));
  equal (array int) [| 1; 2 |] (Nx.shape (Nx.quantile ~axis:1 [| 0.5 |] m));
  let i = Nx.create Nx.float32 [| 2 |] [| neg_infinity; infinity |] in
  equal (array float_exact) [| neg_infinity |]
    (Nx.to_array (Nx.quantile [| 0. |] i));
  invalid ~by:"Nx.quantile" (fun () -> Nx.quantile [| 1.5 |] x);
  invalid ~by:"Nx.quantile" (fun () -> Nx.quantile [| nan |] x);
  invalid ~by:"Nx.quantile" (fun () ->
      Nx.quantile [| 0.5 |] (Nx.zeros Nx.float32 [| 0 |]));
  invalid ~by:"Nx.quantile" (fun () -> Nx.quantile ~axis:2 [| 0.5 |] m)

(* histogram *)

let edges_gen =
  let open Gen in
  let+ es =
    list ~size:(int_range 2 6)
      (map (fun i -> Float.of_int i /. 2.) (int_range (-6) 6))
  in
  let es = List.sort_uniq compare es in
  Array.of_list (if List.length es < 2 then [ -1.; 1. ] else es)

let coord =
  Gen.(
    frequency
      [
        (6, map (fun i -> Float.of_int i /. 4.) (int_range (-16) 16));
        (1, of_list specials);
      ])

let law_histogram (es, (xs, ys), ws) =
  let f64 a = Nx.create Nx.float64 [| Array.length a |] a in
  let e1, e2 = es in
  let bin e x =
    let m = Array.length e in
    if not (x >= e.(0) && x <= e.(m - 1)) then None
    else if x = e.(m - 1) then Some (m - 2)
    else
      let j = ref 0 in
      while x >= e.(!j + 1) do
        incr j
      done;
      Some !j
  in
  let n1 = Array.length e1 - 1 and n2 = Array.length e2 - 1 in
  let want = Array.make (n1 * n2) 0. in
  Array.iteri
    (fun i x ->
      match (bin e1 x, bin e2 ys.(i)) with
      | Some a, Some b -> want.((a * n2) + b) <- want.((a * n2) + b) +. ws.(i)
      | _ -> ())
    xs;
  let dims = [ (f64 e1, f64 xs); (f64 e2, f64 ys) ] in
  equal ~msg:"weighted" (array float_exact) want
    (Nx.to_array (Nx.histogram ~weights:(f64 ws) dims));
  let counts = Array.make (n1 * n2) 0. in
  Array.iteri
    (fun i x ->
      match (bin e1 x, bin e2 ys.(i)) with
      | Some a, Some b -> counts.((a * n2) + b) <- counts.((a * n2) + b) +. 1.
      | _ -> ())
    xs;
  let h = Nx.histogram dims in
  equal ~msg:"shape" (array int) [| n1; n2 |] (Nx.shape h);
  equal ~msg:"counts" (array float_exact) counts (Nx.to_array h)

let histogram_case =
  let open Gen in
  let* e1 = edges_gen in
  let* e2 = edges_gen in
  let* n = int_range 0 12 in
  let* xs = array ~size:(constant n) coord in
  let* ys = array ~size:(constant n) coord in
  let+ ws = array ~size:(constant n) (float_range (-2.) 2.) in
  ((e1, e2), (xs, ys), ws)

let test_histogram_cases () =
  let e = Nx.create Nx.float64 [| 4 |] [| 0.; 1.; 2.; 3. |] in
  let x = Nx.create Nx.float64 [| 7 |] [| 0.; 0.5; 1.; 3.; 3.5; nan; -0. |] in
  equal (array float_exact) [| 3.; 1.; 1. |]
    (Nx.to_array (Nx.histogram [ (e, x) ]));
  let e16 = Nx.cast Nx.float16 e and x16 = Nx.cast Nx.float16 x in
  equal (array float_exact) [| 3.; 1.; 1. |]
    (Nx.to_array (Nx.histogram [ (e16, x16) ]));
  invalid ~by:"Nx.histogram" (fun () ->
      Nx.histogram ([] : ((float, D.float64_elt, Nx.host) Nx.t * _) list));
  invalid ~by:"Nx.histogram" (fun () ->
      Nx.histogram [ (Nx.zeros Nx.float64 [| 1 |], x) ]);
  invalid ~by:"Nx.histogram" (fun () ->
      Nx.histogram [ (e, x); (e, Nx.zeros Nx.float64 [| 2 |]) ]);
  invalid ~by:"Nx.histogram" (fun () -> Nx.histogram ~weights:e [ (e, x) ])

let () =
  exit
    (run "nx order"
       [
         group "sort argsort top_k"
           [
             prop "a stable sort of each slice" sort_case law_sort;
             test "cases" test_sort_cases;
           ];
         group "order_key"
           [
             prop "keys compare as the sort order" key_case law_key;
             test "cases" test_key_cases;
           ];
         group "lexsort searchsorted"
           [
             prop "lexsort is a stable sort of the rows" rows_case law_lexsort;
             prop "searchsorted counts the keys before" search_case law_search;
             test "cases" test_search_cases;
           ];
         group "unique"
           [
             prop "groups in order of first appearance" unique_case law_unique;
             test "cases" test_unique_cases;
             test "70,000 keys" test_unique_long;
           ];
         group "positions"
           [
             prop "each index repeated its count" count_case law_positions;
             test "cases" test_positions_cases;
             test "9000 positions" test_positions_long;
           ];
         group "argwhere"
           [
             prop "the coordinates of each element other than zero"
               argwhere_case law_argwhere;
             test "cases" test_argwhere_cases;
           ];
         group "nonzero extract"
           [
             prop "nonzero is argwhere's columns" argwhere_case law_nonzero;
             test "nonzero cases" test_nonzero_cases;
             prop "extract reads where the condition is not zero" extract_case
               law_extract;
             test "extract cases" test_extract_cases;
           ];
         group "quantile"
           [
             prop "interpolated order statistics" quantile_case law_quantile;
             test "cases" test_quantile_cases;
           ];
         group "histogram"
           [
             prop "points counted into their cells" histogram_case law_histogram;
             test "cases" test_histogram_cases;
           ];
       ])
