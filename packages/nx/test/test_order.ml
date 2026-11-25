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
open Nx_support
module D = Nx_array.Dtype

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

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

(* Views *)

(* How a value lies: contiguous; every axis flipped; every axis read at a step
   of 2; from an offset of one slice along axis 0; or one slice along axis 0
   repeated, a stride of 0. *)
type lanes_view = Plain | Flipped | Strided | Offset | Repeated

let lanes_views = [ Plain; Flipped; Strided; Offset; Repeated ]

let lanes_view_name = function
  | Plain -> "plain"
  | Flipped -> "flipped"
  | Strided -> "strided"
  | Offset -> "offset"
  | Repeated -> "repeated"

let numel s = Array.fold_left ( * ) 1 s

(* The elements [xs] of shape [s] broadcast to shape [b], in C order. *)
let spread s xs b =
  Array.init (numel b) (fun k ->
      let i = index b k in
      let p = ref 0 in
      Array.iteri (fun a e -> p := (!p * e) + if e = 1 then 0 else i.(a)) s;
      xs.(!p))

(* [xs] of shape [s] with each slice along axis 0 its first, as a [Repeated]
   view reads them. *)
let repeated s xs =
  if Array.length xs = 0 then xs
  else
    let one = Array.copy s in
    one.(0) <- 1;
    spread one xs s

(* [xs] of shape [s] as [view] lays them out. A [Repeated] view reads the first
   slice along axis 0 alone. *)
let laid dt s xs view =
  let n = Array.length xs and zero = D.zero dt in
  match view with
  | Plain -> Nx.create dt s xs
  | Flipped -> Nx.flip (Nx.create dt s (Array.init n (fun i -> xs.(n - 1 - i))))
  | Strided ->
      (* [xs] at the even indices of a value twice as long on every axis. *)
      let w = Array.map (fun d -> 2 * d) s in
      let at i =
        if Array.exists (fun c -> c mod 2 = 1) i then zero
        else
          let p = ref 0 in
          Array.iteri (fun a c -> p := (!p * s.(a)) + (c / 2)) i;
          xs.(!p)
      in
      let x = Nx.create dt w (Array.init (numel w) (fun k -> at (index w k))) in
      Nx.slice (Array.to_list (Array.map (fun d -> Nx.Rs (0, 2 * d, 2)) s)) x
  | Offset ->
      let w = Array.copy s in
      w.(0) <- s.(0) + 1;
      let x = Array.append (Array.make (numel w - n) zero) xs in
      Nx.slice [ Nx.R (1, s.(0) + 1) ] (Nx.create dt w x)
  | Repeated ->
      let one = Array.copy s in
      one.(0) <- 1;
      let first =
        Array.init (numel one) (fun i -> if i < n then xs.(i) else zero)
      in
      Nx.broadcast_to s (Nx.create dt one first)

(* sort, argsort, top_k *)

type sort_case =
  | Sort_case :
      ('v, 's) D.t * int array * int * bool * int * 'v array * lanes_view
      -> sort_case

let sort_case =
  let open Gen in
  with_pp
    (fun ppf (Sort_case (dt, s, a, d, k, _, view)) ->
      Format.fprintf ppf "%s %a axis %d%s k %d %s" (D.name dt) pp_ints s a
        (if d then " descending" else "")
        k (lanes_view_name view))
    (let* (D.Any dt) = of_list all in
     let* r = int_range 1 3 in
     let* s = shape r in
     let* a = int_range (-r) (r - 1) in
     let* d = bool in
     let n = s.(if a < 0 then a + r else a) in
     let* k = int_range 0 n in
     let* vs = values dt (Array.fold_left ( * ) 1 s) in
     let+ view = of_list lanes_views in
     Sort_case (dt, s, a, d, k, vs, view))

(* An array's elements as bits, in C order: its bytes, or for a dtype packed
   below a byte each element's value, which is its bits there. *)
let bits (type v s) (x : (v, s, Nx.host) Nx.t) : int array =
  match Nx.dtype x with
  | D.Int4 -> Nx.to_array x
  | D.Uint4 -> Nx.to_array x
  | D.Bit -> Array.map Bool.to_int (Nx.to_array x)
  | D.Float4_e2m1fn ->
      Array.map (fun v -> Int64.to_int (Int64.bits_of_float v)) (Nx.to_array x)
  | _ -> Nx_support.bytes (Option.get (Nx.Repr.array ~by:"test" x))

(* Each slice's positions are a stable sort of its elements in the sort order,
   or its reverse; the values are the elements at those positions, bit for bit;
   top_k is the descending sort's first k. *)
let law_sort (Sort_case (dt, s, a, descending, k, vs, view)) =
  let x = laid dt s vs view in
  let xs = Nx.to_array x in
  let r = Array.length s in
  let a' = if a < 0 then a + r else a in
  let c = cmp dt in
  let order = if descending then fun u v -> c v u else c in
  cover "ties"
    (Array.length xs > List.length (List.sort_uniq c (Array.to_list xs)));
  List.iter
    (fun w -> cover (lanes_view_name w) (view = w && Array.length xs > 0))
    lanes_views;
  let values, positions = Nx.sort ~descending ~axis:a x in
  let ps = Nx.to_array positions in
  equal ~msg:"values, bit for bit" (array int)
    (bits (Nx.take_along_axis ~axis:a positions x))
    (bits values);
  let tv, tp = Nx.top_k ~k ~axis:a x in
  let kept = Array.copy s in
  kept.(a') <- k;
  equal ~msg:"top_k's shape" (array int) kept (Nx.shape tv);
  let tps = Nx.to_array tp in
  List.iter2
    (fun slice top ->
      let here = Array.of_list (List.map (fun p -> xs.(p)) slice) in
      equal ~msg:"positions" (list int) (stable order here)
        (List.map (fun p -> Int64.to_int ps.(p)) slice);
      equal ~msg:"top_k's positions" (list int)
        (List.filteri (fun j _ -> j < k) (stable (fun u v -> c v u) here))
        (List.map (fun p -> Int64.to_int tps.(p)) top))
    (slices s a') (slices kept a');
  equal ~msg:"top_k's values, bit for bit" (array int)
    (bits (Nx.take_along_axis ~axis:a tp x))
    (bits tv);
  equal ~msg:"argsort" (array int64) ps
    (Nx.to_array (Nx.argsort ~descending ~axis:a x))

(* The doc's orders: stable ascending; descending as its exact reverse, NaN
   first and [+0.] before [-0.]; top_k's greatest first, a NaN before every
   number, complex by real then imaginary part. *)
let test_sort_rows () =
  let f32 xs = Nx.create Nx.float32 [| Array.length xs |] xs in
  let bits x = Array.map Int64.bits_of_float (Nx.to_array x) in
  let v, p = Nx.sort (Nx.create Nx.int32 [| 5 |] [| 3l; 1l; 4l; 1l; 5l |]) in
  equal ~msg:"ascending" (array int32) [| 1l; 1l; 3l; 4l; 5l |] (Nx.to_array v);
  equal ~msg:"ascending positions" (array int64) [| 1L; 3L; 0L; 2L; 4L |]
    (Nx.to_array p);
  let v, _ = Nx.sort ~descending:true (f32 [| 1.; nan; -0.; 0. |]) in
  equal ~msg:"descending" (array float_exact) [| nan; 1.; 0.; -0. |]
    (Nx.to_array v);
  equal ~msg:"descending, +0. before -0." (array int64)
    (Array.map Int64.bits_of_float [| 1.; 0.; -0. |])
    (Array.sub (bits v) 1 3);
  let v, p = Nx.top_k ~k:2 (f32 [| 3.; 1.; 4.; 1.; 5. |]) in
  equal ~msg:"top_k" (array float_exact) [| 5.; 4. |] (Nx.to_array v);
  equal ~msg:"top_k positions" (array int64) [| 4L; 2L |] (Nx.to_array p);
  equal ~msg:"top_k, a NaN first" (array float_exact) [| nan; 2. |]
    (Nx.to_array (fst (Nx.top_k ~k:2 (f32 [| 1.; 2.; nan |]))));
  equal ~msg:"top_k of none" (array int) [| 0 |]
    (Nx.shape (fst (Nx.top_k ~k:0 (f32 [| 1. |]))));
  let z =
    Nx.create Nx.complex128 [| 3 |]
      Complex.[| one; { re = 1.; im = -1. }; zero |]
  in
  equal ~msg:"complex, by real then imaginary part" (array int64) [| 0L; 1L |]
    (Nx.to_array (snd (Nx.top_k ~k:2 z)))

let test_sort_refusals () =
  let f32 xs = Nx.create Nx.float32 [| Array.length xs |] xs in
  invalid ~by:"Nx.top_k" (fun () -> Nx.top_k ~k:2 (f32 [| 1. |]));
  invalid ~by:"Nx.top_k" (fun () -> Nx.top_k ~k:(-1) (f32 [| 1. |]));
  invalid ~by:"Nx.top_k" (fun () -> Nx.top_k ~k:0 ~axis:1 (f32 [| 1. |]));
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

(* The float a key of [w] bits reads [dt]'s values in, with the unsigned integer
   of its width: [dt] itself at its own width, bfloat16 at 16 bits,
   float8_e4m3fn for float4_e2m1fn at 8, else float16, float32, float64. *)
type width_float = Wf : (float, 'r) D.t * ('k, 'q) D.t -> width_float

let float_of_width (type s) (dt : (float, s) D.t) w =
  match (w, dt) with
  | 16, D.Bfloat16 -> Wf (D.Bfloat16, D.Uint16)
  | 8, D.Float8_e5m2 -> Wf (D.Float8_e5m2, D.Uint8)
  | 8, _ -> Wf (D.Float8_e4m3fn, D.Uint8)
  | 16, _ -> Wf (D.Float16, D.Uint16)
  | 32, _ -> Wf (D.Float32, D.Uint32)
  | _ -> Wf (D.Float64, D.Uint64)

(* The keys nx.mli's formulas give [x]'s elements at [w] bits, as unsigned
   words: an integer's value with the sign bit flipped where signed, a boolean's
   [0] or [1], a float's bits in the float of the width, all flipped where its
   sign bit is set and only the sign bit elsewhere, every NaN all ones. *)
let formula (type v s) (x : (v, s, Nx.host) Nx.t) w : int64 array =
  let mask = if w = 64 then -1L else Int64.(sub (shift_left 1L w) 1L) in
  let sign = Int64.shift_left 1L (w - 1) in
  let flip b =
    if Int64.logand b sign <> 0L then Int64.logand (Int64.lognot b) mask
    else Int64.logxor b sign
  in
  match Nx.dtype x with
  | D.Uint64 -> Nx.to_array (Nx.bitcast Nx.int64 x)
  | dt -> (
      match D.kind dt with
      | D.Float ->
          let (Wf (fw, uk)) = float_of_width dt w in
          let f = Nx.cast fw x in
          let b = Nx.to_array (Nx.cast Nx.int64 (Nx.bitcast uk f)) in
          Array.mapi
            (fun i v -> if Float.is_nan v then mask else flip b.(i))
            (Nx.to_array f)
      | D.Signed ->
          Array.map
            (fun v -> Int64.logand (Int64.add v sign) mask)
            (Nx.to_array (Nx.cast Nx.int64 x))
      | _ -> Nx.to_array (Nx.cast Nx.int64 x))

(* Keys compare, unsigned, as their elements sort and are equal iff they are;
   each is nx.mli's formula; [bitwise_not] of a key reverses its order. *)
let law_key (Key_case (dt, vs)) =
  let x, xs = made dt [| Array.length vs |] vs in
  let c = cmp dt in
  List.iter
    (fun (D.Any kd) ->
      if D.bits kd >= D.bits dt then begin
        let key = Nx.order_key kd x in
        let k = Nx.to_array (Nx.cast Nx.int64 (Nx.cast Nx.uint64 key)) in
        let nk =
          Nx.to_array
            (Nx.cast Nx.int64 (Nx.cast Nx.uint64 (Nx.bitwise_not key)))
        in
        equal
          ~msg:(D.name kd ^ ", the formula")
          (array int64)
          (formula x (D.bits kd))
          k;
        Array.iteri
          (fun i u ->
            Array.iteri
              (fun j v ->
                let order = compare (c u v) 0 in
                equal ~msg:(D.name kd) int order
                  (compare (Int64.unsigned_compare k.(i) k.(j)) 0);
                equal
                  ~msg:(D.name kd ^ ", bitwise_not")
                  int (-order)
                  (compare (Int64.unsigned_compare nk.(i) nk.(j)) 0))
              xs)
          xs
      end)
    widths

(* The doc's keys: [-0.] just below [+0.], every NaN all ones, an int64 cast
   keeping a signed integer's key, a float of a narrower format keyed through
   float16 at 16 bits. *)
let test_key_rows () =
  let i8 = Nx.create Nx.int8 [| 3 |] [| -128; 0; 127 |] in
  equal ~msg:"int8 at 8 bits" (array int) [| 0; 128; 255 |]
    (Nx.to_array (Nx.order_key Nx.uint8 i8));
  let f = Nx.create Nx.float64 [| 4 |] [| -0.; 0.; nan; -.nan |] in
  equal ~msg:"float64's zeros and NaNs" (array int64)
    [| Int64.max_int; Int64.min_int; -1L; -1L |]
    (Nx.to_array (Nx.order_key Nx.uint64 f));
  let i = Nx.create Nx.int32 [| 3 |] [| -5l; 0l; 7l |] in
  equal ~msg:"an int64 cast" (array int64)
    (Nx.to_array (Nx.order_key Nx.uint64 i))
    (Nx.to_array (Nx.order_key Nx.uint64 (Nx.cast Nx.int64 i)));
  let h = Nx.create Nx.float8_e4m3fn [| 4 |] [| -2.; -0.; 0.5; nan |] in
  equal ~msg:"float8_e4m3fn at 16 bits" (array int)
    (Nx.to_array (Nx.order_key Nx.uint16 h))
    (Nx.to_array (Nx.order_key Nx.uint16 (Nx.cast Nx.float16 h)))

let test_key_refusals () =
  let i = Nx.create Nx.int32 [| 3 |] [| -5l; 0l; 7l |] in
  invalid ~by:"Nx.order_key" (fun () -> Nx.order_key Nx.uint8 i);
  invalid ~by:"Nx.order_key" (fun () -> Nx.order_key Nx.int64 i);
  invalid ~by:"Nx.order_key" (fun () ->
      Nx.order_key Nx.uint64 (Nx.create Nx.complex64 [| 1 |] [| Complex.one |]))

(* lexsort, searchsorted, unique *)

type rows_case = Rows_case : ('v, 's) D.t * int array * 'v array -> rows_case

(* 1-d keys, or [n; w] rows, of a dtype with keys, [n] drawn by [ns]. *)
let rows_of ns =
  let open Gen in
  with_pp
    (fun ppf (Rows_case (dt, s, _)) ->
      Format.fprintf ppf "%s %a" (D.name dt) pp_ints s)
    (let* (D.Any dt) = of_list keyed in
     let* n = ns in
     let* s =
       one_of [ constant [| n |]; map (fun w -> [| n; w |]) (int_range 0 3) ]
     in
     let+ vs = values dt (Array.fold_left ( * ) 1 s) in
     Rows_case (dt, s, vs))

let rows_case = rows_of (Gen.int_range 0 7)

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
          (fun p ->
            at_least ~msg:"a count" int64 ~than:0L p;
            at_most ~msg:"a count" int64 ~than:(Int64.of_int m) p)
          got)
    [ (`Left, fun d -> d < 0); (`Right, fun d -> d <= 0) ]

(* The doc's knots: [-0.] counts as [+0.], NaN past every number. *)
let test_search_rows () =
  let f64 xs = Nx.create Nx.float64 [| Array.length xs |] xs in
  let knots = f64 [| 0.; 1.; 1.; 2. |] and x = f64 [| -0.; 1.; 1.5; nan |] in
  equal ~msg:"`Left" (array int64) [| 0L; 1L; 3L; 4L |]
    (Nx.to_array (Nx.searchsorted ~side:`Left knots x));
  equal ~msg:"`Right" (array int64) [| 1L; 3L; 3L; 4L |]
    (Nx.to_array (Nx.searchsorted ~side:`Right knots x))

let test_lexsort_rows () =
  equal ~msg:"no column" (array int64) [| 0L; 1L; 2L |]
    (Nx.to_array (Nx.lexsort (Nx.zeros Nx.int8 [| 3; 0 |])));
  invalid ~by:"Nx.lexsort" (fun () ->
      Nx.lexsort (Nx.zeros Nx.int8 [| 1; 1; 1 |]));
  invalid ~by:"Nx.lexsort" (fun () ->
      Nx.lexsort (Nx.create Nx.complex64 [| 1 |] [| Complex.one |]))

let test_search_refusals () =
  let m = Nx.create Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  invalid ~by:"Nx.searchsorted" (fun () ->
      Nx.searchsorted ~side:`Left m (Nx.create Nx.int32 [| 1 |] [| 1l |]));
  invalid ~by:"Nx.searchsorted" (fun () ->
      Nx.searchsorted ~side:`Left m (Nx.zeros Nx.int32 [| 1; 3 |]));
  invalid ~by:"Nx.searchsorted" (fun () ->
      Nx.searchsorted ~side:`Left (Nx.scalar Nx.int32 1l) m);
  let z = Nx.create Nx.complex64 [| 1 |] [| Complex.one |] in
  invalid ~by:"Nx.searchsorted" (fun () -> Nx.searchsorted ~side:`Left z z)

(* The operations [f] applies, by name, each computed eagerly. *)
let operations f =
  let log = ref [] in
  Nx.Prim.interpret ~name:"test.order" Extent
    (fun _ ~by op ->
      log := Nx.Prim.name op :: !log;
      Nx.Prim.eval ~by op)
    (fun _ -> ignore (f ()));
  List.rev !log

let count name f = List.length (List.filter (String.equal name) (operations f))

(* [⌈log₂ (m + 1)⌉]. *)
let rounds m =
  let rec go r = if m lsr r = 0 then r else go (r + 1) in
  go 0

(* What nx.mli counts: lexsort sorts once per column, top_k sorts once, and a
   search of [m] keys takes [⌈log₂ (m + 1)⌉] rounds, each a gather. *)
let test_counts () =
  List.iter
    (fun w ->
      equal
        ~msg:(Printf.sprintf "lexsort of %d columns" w)
        int w
        (count "Sort" (fun () -> Nx.lexsort (Nx.zeros Nx.int32 [| 5; w |]))))
    [ 0; 1; 2; 3; 4 ];
  equal ~msg:"top_k" int 1
    (count "Sort" (fun () -> Nx.top_k ~k:2 (Nx.zeros Nx.int32 [| 3; 6 |])));
  List.iter
    (fun m ->
      let s = Nx.create Nx.int32 [| m |] (Array.init m Int32.of_int) in
      let v = Nx.create Nx.int32 [| 3 |] [| 0l; 4l; 9l |] in
      equal
        ~msg:(Printf.sprintf "searchsorted of %d keys" m)
        int (rounds m)
        (count "Gather" (fun () -> Nx.searchsorted ~side:`Left s v));
      let s = Nx.reshape [| 1; m |] s and v = Nx.reshape [| 1; 3 |] v in
      equal
        ~msg:(Printf.sprintf "searchsorted_along_axis of %d keys" m)
        int (rounds m)
        (count "Gather" (fun () ->
             Nx.searchsorted_along_axis ~axis:1 ~side:`Right s v)))
    [ 1; 2; 3; 7; 8; 9 ]

(* searchsorted_along_axis *)

type along_case =
  | Along_case : {
      dt : ('v, 's) D.t;
      axis : int;
      s : int array;
      vs : 'v array;
      t : int array;
      ws : 'v array;
      sorted : bool;
      view : lanes_view;
      cut : bool;  (** Whether [s] is also searched split along [axis]. *)
    }
      -> along_case

(* Lanes of [m] keys along [axis] of a value of rank 1 to 3, sought by lanes of
   [k] keys. Each other axis has one extent in both values, or extent 1 in one
   of them. Half the cases are floats, so that one in six or more holds a NaN
   key. *)
let along_case =
  let open Gen in
  let floats = List.filter (fun (D.Any dt) -> D.is D.Float dt) keyed in
  with_pp
    (fun ppf (Along_case { dt; axis; s; t; sorted; view; _ }) ->
      Format.fprintf ppf "%s %a in %a axis %d%s %s" (D.name dt) pp_ints t
        pp_ints s axis
        (if sorted then "" else " unsorted")
        (lanes_view_name view))
    (let* (D.Any dt) = one_of [ of_list floats; of_list keyed ] in
     let* r = int_range 1 3 in
     let* axis = int_range (-r) (r - 1) in
     let a = if axis < 0 then axis + r else axis in
     let* m = int_range 0 6 in
     let* k = int_range 0 4 in
     let* off =
       array ~size:(constant r) (pair (int_range 0 3) (int_range 0 2))
     in
     let extent lane one i =
       let e, b = off.(i) in
       if i = a then lane else if b = one then 1 else e
     in
     let s = Array.init r (extent m 1) and t = Array.init r (extent k 2) in
     let* vs = values dt (Array.fold_left ( * ) 1 s) in
     let* ws = values dt (Array.fold_left ( * ) 1 t) in
     let* sorted = frequency [ (4, constant true); (1, constant false) ] in
     let* view = of_list lanes_views in
     let+ cut = bool in
     Along_case { dt; axis; s; vs; t; ws; sorted; view; cut })

let has_nan (type v s) (dt : (v, s) D.t) (xs : v array) =
  match D.kind dt with D.Float -> Array.exists Float.is_nan xs | _ -> false

module S2 = (val Nx.devices [ Nx_support.memory 0; Nx_support.memory 1 ])

let law_along (Along_case { dt; axis; s; vs; t; ws; sorted; view; cut }) =
  let r = Array.length s in
  let a = if axis < 0 then axis + r else axis in
  let c = numeric dt in
  let read s xs =
    let xs = Nx.to_array (Nx.create dt s xs) in
    if view = Repeated then repeated s xs else xs
  in
  let xs = read s vs and wv = read t ws in
  if sorted then
    List.iter
      (fun lane ->
        let keys = Array.of_list (List.map (fun p -> xs.(p)) lane) in
        List.iter2 (fun p i -> xs.(p) <- keys.(i)) lane (stable (cmp dt) keys))
      (slices s a);
  (* The result's shape [b], and [b] with [s]'s lanes, [bs]. *)
  let b =
    Array.mapi
      (fun i e -> if i = a then t.(a) else if e = 1 then t.(i) else e)
      s
  in
  let bs = Array.copy b in
  bs.(a) <- s.(a);
  let lanes shape ys =
    Array.of_list
      (List.map
         (fun l -> Array.of_list (List.map (fun p -> ys.(p)) l))
         (slices shape a))
  in
  let ls = lanes bs (spread s xs bs) and lv = lanes b (spread t wv b) in
  (* Whether some key is sought in a lane, and in one that holds keys. *)
  let sought = numel b > 0 in
  let m = s.(a) in
  cover "a lane of no key" (sought && m = 0);
  cover "a NaN key"
    (has_nan dt (spread s xs bs) || has_nan dt (spread t wv b));
  cover "an unsorted lane" (sought && (not sorted) && m > 1);
  cover "a broadcast lane"
    (sought && m > 0
    && List.exists (fun i -> i <> a && s.(i) <> t.(i)) (List.init r Fun.id));
  List.iter
    (fun w -> cover (lanes_view_name w) (sought && view = w))
    lanes_views;
  let x = laid dt s xs view and v = laid dt t wv view in
  let search side = Nx.searchsorted_along_axis ~axis ~side x v in
  let left = search `Left and right = search `Right in
  cover "lanes split along the axis" (cut && m > 1 && sought);
  if cut then begin
    (* Over a cut axis, as take_along_axis gathers. *)
    let split = Nx.place (S2.split ~axis:a) x in
    List.iter
      (fun (side, here) ->
        equal ~msg:"lanes split along the axis" (array int64) (Nx.to_array here)
          (Nx.to_array
             (Nx.searchsorted_along_axis ~axis ~side split (Nx.place S2.on v))))
      [ (`Left, left); (`Right, right) ]
  end;
  equal ~msg:"`Left's shape" (array int) b (Nx.shape left);
  equal ~msg:"`Right's shape" (array int) b (Nx.shape right);
  let lo = lanes b (Nx.to_array left) and hi = lanes b (Nx.to_array right) in
  let alone side l w =
    let n = Array.length in
    Nx.to_array
      (Nx.searchsorted ~side
         (Nx.create dt [| n l |] l)
         (Nx.create dt [| n w |] w))
  in
  let m64 = Int64.of_int m in
  Array.iteri
    (fun j l ->
      let w = lv.(j) in
      equal ~msg:"`Left, the lane alone" (array int64) (alone `Left l w) lo.(j);
      equal ~msg:"`Right, the lane alone" (array int64) (alone `Right l w)
        hi.(j);
      Array.iteri
        (fun i key ->
          let lo = lo.(j).(i) and hi = hi.(j).(i) in
          if not sorted then
            List.iter
              (fun p ->
                at_most ~msg:"a count" int64 ~than:m64 p;
                at_most ~msg:"a count" int64 ~than:p 0L)
              [ lo; hi ]
          else begin
            at_most ~msg:"`Left" int64 ~than:hi lo;
            let equals =
              Array.fold_left (fun n k -> if c k key = 0 then n + 1 else n) 0 l
            in
            equal ~msg:"`Right - `Left, the keys equal" int equals
              (Int64.to_int (Int64.sub hi lo))
          end)
        w)
    ls

let test_along_cases () =
  let along ~axis ~side s v =
    Nx.to_array (Nx.searchsorted_along_axis ~axis ~side s v)
  in
  let f64 s xs = Nx.create Nx.float64 s xs in
  let s = f64 [| 2; 4 |] [| 0.; 1.; 1.; 2.; neg_infinity; -0.; 3.; nan |]
  and v = f64 [| 2; 2 |] [| 1.; 1.5; 0.; nan |] in
  equal (array int64) [| 1L; 3L; 1L; 3L |] (along ~axis:1 ~side:`Left s v);
  equal (array int64) [| 3L; 3L; 2L; 4L |] (along ~axis:(-1) ~side:`Right s v);
  (* Lanes along axis 0: [s]'s one lane is each of [v]'s two. *)
  let s = Nx.create Nx.int32 [| 3; 1 |] [| 1l; 2l; 3l |]
  and v = Nx.create Nx.int32 [| 1; 2 |] [| 2l; 0l |] in
  equal (array int64) [| 1L; 0L |] (along ~axis:0 ~side:`Left s v);
  equal (array int64) [| 0L; 0L; 0L; 0L; 0L; 0L |]
    (along ~axis:0 ~side:`Right
       (Nx.zeros Nx.int8 [| 0; 2 |])
       (Nx.zeros Nx.int8 [| 3; 2 |]));
  let m = Nx.zeros Nx.int32 [| 2; 3 |] in
  let by = "Nx.searchsorted_along_axis" in
  invalid ~by (fun () -> along ~axis:2 ~side:`Left m m);
  invalid ~by (fun () -> along ~axis:(-3) ~side:`Left m m);
  invalid ~by (fun () ->
      along ~axis:0 ~side:`Left m (Nx.zeros Nx.int32 [| 3 |]));
  invalid ~by (fun () ->
      along ~axis:1 ~side:`Left m (Nx.zeros Nx.int32 [| 3; 1 |]));
  let z = Nx.create Nx.complex64 [| 1 |] [| Complex.one |] in
  invalid ~by (fun () -> along ~axis:0 ~side:`Left z z)

type unique_case = Unique_case : rows_case * int option -> unique_case

(* Rows of int32 keys taking [2n + 1023], [2n + 1024] and [2n + 1025] values for
   [n = 100], and int16 rows of two columns taking [2n + 1024], either side of
   where nx.cpu's numbering moves from a direct table to a hash. *)
let near_rows =
  let n = 100 in
  let column d =
    let r = (2 * n) + 1024 + d in
    Array.init n (fun i ->
        if i = 0 then 0 else if i = 1 then r - 1 else i * 37 mod r)
  in
  List.map
    (fun d ->
      Unique_case
        (Rows_case (D.Int32, [| n |], Array.map Int32.of_int (column d)), None))
    [ -1; 0; 1 ]
  @ [
      Unique_case
        ( Rows_case
            ( D.Int16,
              [| n; 2 |],
              Array.init (2 * n) (fun j ->
                  if j mod 2 = 0 then (column 0).(j / 2) - 600 else 7) ),
          None );
    ]

(* A case's rows, each made its first. *)
let one_key (Rows_case (dt, s, vs)) =
  let w = if Array.length s = 1 then 1 else s.(1) in
  Rows_case (dt, s, Array.init (Array.length vs) (fun i -> vs.(i mod w)))

(* A case's rows without those equal to an earlier one, as Nx reads them. *)
let distinct (Rows_case (dt, s, vs)) =
  let _, xs = made dt s vs in
  let c = lex (cmp dt) in
  let kept =
    Array.fold_left
      (fun kept r ->
        if List.exists (fun k -> c k r = 0) kept then kept else r :: kept)
      [] (rows s xs)
  in
  let n = List.length kept in
  let s = if Array.length s = 1 then [| n |] else [| n; s.(1) |] in
  Rows_case (dt, s, Array.concat (List.rev kept))

let pp_unique ppf (Unique_case (Rows_case (dt, s, _), size)) =
  Format.fprintf ppf "%s %a size %s" (D.name dt) pp_ints s
    (match size with None -> "-" | Some l -> string_of_int l)

(* Rows: half drawn, a quarter one key over 2 to 7 rows, a quarter drawn
   without their repeats. Of the [k] keys, a third have no size, a third a
   size that cuts them, below [k] (or 0 where [k] is 0), and a third one of
   [k] to [k + 2]. "one key of many rows" is at least 1/4 of the cases,
   "cut" 0.30, "padded" 2/9 and "sized to its keys" 1/9, so the law draws
   200 cases, which miss the last with probability 6 10^-11. *)
let unique_case =
  let open Gen in
  with_pp pp_unique
    (let* r =
       frequency
         [
           (2, rows_case);
           (1, map one_key (rows_of (int_range 2 7)));
           (1, map distinct rows_case);
         ]
     in
     let (Rows_case (_, keys, _)) = distinct r in
     let k = keys.(0) in
     let+ size =
       frequency
         [
           (1, constant None);
           (1, map Option.some (int_range 0 (Int.max 0 (k - 1))));
           (1, map Option.some (int_range k (k + 2)));
         ]
     in
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
  let n = s.(0) in
  cover "padded" (l > k);
  cover "cut" (l < k);
  cover "sized to its keys" (size = Some k);
  cover "no size" (size = None);
  cover "one key of many rows" (k = 1 && n > 1);
  cover "every row its own key" (k = n && n > 1);
  equal ~msg:"ids" (array int64) ids (Nx.to_array g.ids);
  equal ~msg:"count" int64 (Int64.of_int k) (Nx.item [] g.count);
  equal ~msg:"first" (array int64)
    (pad (fun (f, _) -> Int64.of_int f) (Int64.of_int s.(0)))
    (Nx.to_array g.first);
  equal ~msg:"counts" (array int64)
    (pad (fun (_, n) -> Int64.of_int !n) 0L)
    (Nx.to_array g.counts)

(* The doc's groups: by first appearance, every NaN one group and the zeros two,
   none of no key. *)
let test_unique_rows () =
  let g = Nx.unique (Nx.create Nx.int32 [| 6 |] [| 7l; 3l; 7l; 7l; 5l; 3l |]) in
  equal ~msg:"ids" (array int64)
    [| 0L; 1L; 0L; 0L; 2L; 1L |]
    (Nx.to_array g.ids);
  equal ~msg:"first" (array int64) [| 0L; 1L; 4L |] (Nx.to_array g.first);
  equal ~msg:"counts" (array int64) [| 3L; 2L; 1L |] (Nx.to_array g.counts);
  let f = Nx.create Nx.float32 [| 5 |] [| nan; -0.; 0.; -.nan; 0. |] in
  equal ~msg:"NaNs and zeros" (array int64) [| 0L; 1L; 2L; 0L; 2L |]
    (Nx.to_array (Nx.unique f).ids);
  let e = Nx.unique (Nx.zeros Nx.int8 [| 0 |]) in
  equal ~msg:"no key, its count" int64 0L (Nx.item [] e.count);
  equal ~msg:"no key, its groups" (array int) [| 0 |] (Nx.shape e.first)

let test_unique_refusals () =
  let f = Nx.create Nx.float32 [| 2 |] [| nan; 0. |] in
  invalid ~by:"Nx.unique" (fun () -> Nx.unique ~size:(-1) f);
  invalid ~by:"Nx.unique" (fun () -> Nx.unique (Nx.zeros Nx.int8 [| 1; 1; 1 |]));
  invalid ~by:"Nx.unique" (fun () ->
      Nx.unique (Nx.create Nx.complex64 [| 1 |] [| Complex.one |]))

(* An unsized unique numbers its keys in one Unique asking for [min n (max 4096
   (n / 16))] entries. *)
let test_unique_one () =
  List.iter
    (fun n ->
      let x =
        Nx.create Nx.int32 [| n |]
          (Array.init n (fun i -> Int32.of_int (i mod 5)))
      in
      let sizes = ref [] in
      let rule : type r. Nx.Prim.interpretation -> by:string -> r Nx.Prim.t -> r
          =
       fun _ ~by op ->
        (match op with
        | Nx.Prim.Unique { size; _ } -> sizes := size :: !sizes
        | _ -> ());
        Nx.Prim.eval ~by op
      in
      Nx.Prim.interpret ~name:"test.order" Extent rule (fun _ ->
          ignore (Nx.unique x));
      equal
        ~msg:(Printf.sprintf "%d keys" n)
        (list int)
        [ Int.min n (Int.max 4096 (n / 16)) ]
        !sizes)
    [ 0; 3; 5000; 70_000 ]

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

(* Groups past the size an unsized unique first asks for: 80,000 int64 keys
   of 50,000 values, [i * 7919 mod 50,000], against a table of keys numbered
   as they are first met. *)
let test_unique_many () =
  let n = 80_000 in
  let key i = Int64.of_int (i * 7919 mod 50_000) in
  let g = Nx.unique (Nx.create Nx.int64 [| n |] (Array.init n key)) in
  let seen = Hashtbl.create n and first = ref [] in
  let ids =
    Array.init n (fun i ->
        match Hashtbl.find_opt seen (key i) with
        | Some (id, c) ->
            incr c;
            id
        | None ->
            let c = ref 1 in
            Hashtbl.add seen (key i) (Hashtbl.length seen, c);
            first := (i, c) :: !first;
            Hashtbl.length seen - 1)
  in
  let first = Array.of_list (List.rev !first) in
  equal ~msg:"ids" (array int64) (Array.map Int64.of_int ids) (Nx.to_array g.ids);
  equal ~msg:"first" (array int64)
    (Array.map (fun (f, _) -> Int64.of_int f) first)
    (Nx.to_array g.first);
  equal ~msg:"counts" (array int64)
    (Array.map (fun (_, c) -> Int64.of_int !c) first)
    (Nx.to_array g.counts);
  equal ~msg:"count" int64 (Int64.of_int (Array.length first)) (Nx.item [] g.count)

(* Lengths that are data *)

type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

let mark i ~by op = Nx.Prim.results ~by (fun _ f -> Nx.Prim.traced i f Mark) op

(* A traced value's length cannot be read: each function raises naming itself,
   and with [size] reads nothing and returns. *)
let test_traced () =
  let x = Nx.create Nx.int32 [| 6 |] [| 1l; 1l; 2l; 0l; 0l; 3l |] in
  let named by =
    Exn.invalid_arg ~sub:(by ^ ": the value is traced by test.order")
  in
  Nx.Prim.interpret ~name:"test.order" Values mark (fun i ->
      let t = Nx.Prim.traced i (Nx.Prim.form x) Mark in
      raises_match (named "Nx.unique") (fun () -> Nx.unique t);
      raises_match (named "Nx.positions") (fun () -> Nx.positions t);
      raises_match (named "Nx.argwhere") (fun () -> Nx.argwhere t);
      raises_match (named "Nx.nonzero") (fun () -> Nx.nonzero t);
      raises_match (named "Nx.extract") (fun () -> Nx.extract t x);
      ignore (Nx.unique ~size:2 t);
      ignore (Nx.positions ~size:4 t);
      ignore (Nx.argwhere ~size:2 t);
      ignore (Nx.nonzero ~size:2 t);
      ignore (Nx.extract ~size:2 t x))

(* positions *)

(* How the counts lie: in order, reversed, every second element of a longer
   array, or one element broadcast. *)
type view = Contiguous | Reversed | Stepped | Broadcast

type count_case =
  | Count_case : ('v, 's) D.t * 'v array * int option * view -> count_case

let view_name = function
  | Contiguous -> "contiguous"
  | Reversed -> "reversed"
  | Stepped -> "stepped"
  | Broadcast -> "broadcast"

(* [vs] as [view] lays them out. *)
let counts_of (type v s) (dt : (v, s) D.t) (vs : v array) view =
  let n = Array.length vs in
  match view with
  | Contiguous -> Nx.create dt [| n |] vs
  | Reversed ->
      let r = Array.init n (fun i -> vs.(n - 1 - i)) in
      Nx.flip (Nx.create dt [| n |] r)
  | Stepped ->
      (* Each count beside another, read past by the step. *)
      let at i = if i mod 2 = 0 then i / 2 else n - 1 - (i / 2) in
      let w = Array.init (2 * n) (fun i -> vs.(at i)) in
      Nx.slice Nx.[ Rs (0, 2 * n, 2) ] (Nx.create dt [| 2 * n |] w)
  | Broadcast -> Nx.broadcast_to [| n |] (Nx.create dt [| 1 |] [| vs.(0) |])

let count_of (type v s) (dt : (v, s) D.t) (v : v) =
  match dt with
  | D.Bool -> if v then 1 else 0
  | D.Int8 -> Stdlib.max v 0
  | D.Uint4 -> v
  | D.Int32 -> Stdlib.max (Int32.to_int v) 0
  | D.Uint64 -> if Int64.compare v 0L < 0 then 0 else Int64.to_int v
  | _ -> assert false

let count_case =
  let open Gen in
  let case (type v s) (dt : (v, s) D.t) (elt : v Gen.t) n =
    let* view =
      if n = 0 then of_list [ Contiguous; Reversed; Stepped ]
      else of_list [ Contiguous; Reversed; Stepped; Broadcast ]
    in
    let* vs =
      match view with
      | Broadcast -> map (Array.make n) elt
      | Contiguous | Reversed | Stepped -> array ~size:(constant n) elt
    in
    (* A size near the total, where the last end lands at or past it. *)
    let total = Array.fold_left (fun t v -> t + count_of dt v) 0 vs in
    let+ size =
      frequency
        [
          (1, constant None);
          (1, map Option.some (int_range 0 12));
          (2, map (fun d -> Some (Stdlib.max 0 (total + d))) (int_range (-1) 1));
        ]
    in
    Count_case (dt, vs, size, view)
  in
  with_pp
    (fun ppf (Count_case (dt, vs, size, view)) ->
      Format.fprintf ppf "%s [%d] %s size %s" (D.name dt) (Array.length vs)
        (view_name view)
        (match size with None -> "-" | Some l -> string_of_int l))
    (let* n = int_range 0 8 in
     one_of
       [
         case D.Bool bool n;
         case D.Int8 (int_range (-2) 3) n;
         case D.Uint4 (int_range 0 3) n;
         case D.Int32 (int32_range (-2l) 3l) n;
         case D.Uint64
           (one_of [ int64_range 0L 3L; constant Int64.min_int ])
           n;
       ])

let law_positions (Count_case (dt, vs, size, view)) =
  let n = Array.length vs in
  let counts = Array.map (count_of dt) vs in
  let total = Array.fold_left ( + ) 0 counts in
  cover "empty" (n = 0);
  cover "no index counted" (n > 0 && total = 0);
  cover "every index counted"
    (n > 0 && Array.for_all (fun k -> k > 0) counts);
  cover "padded" (match size with Some l -> l > total | None -> false);
  cover "cut short" (match size with Some l -> l < total | None -> false);
  cover "sized to the total" (size = Some total && total > 0);
  List.iter
    (fun v -> cover (view_name v) (view = v))
    [ Contiguous; Reversed; Stepped; Broadcast ];
  let want =
    List.concat
      (List.mapi (fun i k -> List.init k (fun _ -> Int64.of_int i))
         (Array.to_list counts))
  in
  let want =
    match size with
    | None -> want
    | Some l ->
        List.init l (fun j ->
            if j < List.length want then List.nth want j else Int64.of_int n)
  in
  let c = counts_of dt vs view in
  equal (list int64) want (Array.to_list (Nx.to_array (Nx.positions ?size c)))

let test_positions_cases () =
  let bools vs = Nx.create Nx.bool [| Array.length vs |] vs in
  equal ~msg:"empty" (array int64) [||]
    (Nx.to_array (Nx.positions (bools [||])));
  equal ~msg:"empty, sized" (array int64) [| 0L; 0L |]
    (Nx.to_array (Nx.positions ~size:2 (bools [||])));
  equal ~msg:"none holds" (array int64) [||]
    (Nx.to_array (Nx.positions (bools [| false; false; false |])));
  equal ~msg:"none holds, sized" (array int64) [| 3L; 3L |]
    (Nx.to_array (Nx.positions ~size:2 (bools [| false; false; false |])));
  equal ~msg:"all hold" (array int64) [| 0L; 1L; 2L |]
    (Nx.to_array (Nx.positions (bools [| true; true; true |])));
  equal ~msg:"all hold, sized 0" (array int64) [||]
    (Nx.to_array (Nx.positions ~size:0 (bools [| true; true; true |])));
  (* Counts whose sum passes int64's range, sized: the first positions. *)
  let big = Nx.create Nx.int64 [| 3 |] [| Int64.max_int; Int64.max_int; 3L |] in
  equal ~msg:"past int64, sized" (array int64) [| 0L; 0L; 0L; 0L; 0L |]
    (Nx.to_array (Nx.positions ~size:5 big));
  equal ~msg:"to int64's end, sized" (array int64) [| 0L; 0L; 0L |]
    (Nx.to_array
       (Nx.positions ~size:3 (Nx.create Nx.int64 [| 2 |] [| Int64.max_int; 1L |])));
  equal (array int64) [| 1L; 3L |]
    (Nx.to_array
       (Nx.positions (Nx.create Nx.bool [| 4 |] [| false; true; false; true |])));
  equal (array int64) [| 0L; 0L; 2L |]
    (Nx.to_array (Nx.positions (Nx.create Nx.int32 [| 3 |] [| 2l; 0l; 1l |])));
  equal (array int64) [| 1L; 2L; 2L |]
    (Nx.to_array
       (Nx.positions ~size:3 (Nx.create Nx.bool [| 2 |] [| false; true |])));
  (* Along an axis of c's length a pad is out of range: take reads zero. *)
  let c = Nx.create Nx.int32 [| 3 |] [| 1l; 0l; 1l |] in
  let v = Nx.create Nx.float64 [| 3 |] [| 5.; 6.; 7. |] in
  equal ~msg:"take at the pads" (array float_exact) [| 5.; 7.; 0.; 0. |]
    (Nx.to_array (Nx.take ~axis:0 (Nx.positions ~size:4 c) v));
  invalid ~by:"Nx.positions" (fun () ->
      Nx.positions (Nx.zeros Nx.float32 [| 2 |]));
  invalid ~by:"Nx.positions" (fun () ->
      Nx.positions (Nx.create Nx.complex64 [| 1 |] [| Complex.one |]));
  invalid ~by:"Nx.positions" (fun () ->
      Nx.positions (Nx.zeros Nx.bool [| 2; 2 |]));
  invalid ~by:"Nx.positions" (fun () ->
      Nx.positions ~size:(-1) (Nx.zeros Nx.bool [| 2 |]));
  (* Without [size], a sum past [max_int] is no length. *)
  let past = "Nx.positions: the counts' sum passes max_int" in
  raises_match (Exn.invalid_arg ~sub:past) (fun () -> Nx.positions big);
  let to_max = Int64.of_int max_int in
  raises_match (Exn.invalid_arg ~sub:past) (fun () ->
      Nx.positions (Nx.create Nx.int64 [| 2 |] [| to_max; 1L |]));
  raises_match (Exn.invalid_arg ~sub:past) (fun () ->
      Nx.positions (Nx.create Nx.int64 [| 4 |] [| 1L; Int64.max_int; 0L; 2L |]))

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
  invalid ~by:"Nx.argwhere" (fun () -> Nx.argwhere ~size:(-1) t)

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
  invalid ~by:"Nx.nonzero" (fun () -> Nx.nonzero ~size:(-1) t)


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
      Nx.extract ~fill:300 (Nx.zeros Nx.bool [| 2 |])
        (Nx.zeros Nx.uint8 [| 2 |]))

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
     (* Each float dtype, with the unit roundoff of the precision it
        interpolates in. *)
     one_of
       [
         case D.Float64 0x1p-53;
         case D.Float32 0x1p-24;
         case D.Float16 0x1p-24;
         case D.Bfloat16 0x1p-24;
         case D.Float8_e4m3fn 0x1p-24;
         case D.Float8_e5m2 0x1p-24;
         case D.Float4_e2m1fn 0x1p-24;
       ])

(* [(1 - f) a + f b] at float64, to far below an ulp: each product and the
   difference [1 - f] split into a rounded part and its exact error. *)
let lerp_exact a b f =
  let two_product x y =
    let p = x *. y in
    (p, Float.fma x y (-.p))
  in
  let g = 1. -. f in
  let g_err = 1. -. g -. f in
  let p, pe = two_product g a and q, qe = two_product f b in
  let sum = p +. q in
  let err =
    if Float.abs p >= Float.abs q then p -. sum +. q else q -. sum +. p
  in
  sum +. (err +. pe +. qe +. (g_err *. a))

(* Each quantile is [(1 - f) a + f b] between the order statistics around [h = q
   (n - 1)] rounded at float64, [a] exactly where the fraction, rounded to the
   interpolation's precision, is zero or [a = b], and an infinity or a NaN as
   that sum of infinities gives it. Elsewhere it is within nx.mli's [3u (|a| +
   |b|)] at float64 and [4u (|a| + |b|)] at float32 of the sum at the float64
   fraction, and a narrow format's rounding once into its dtype, half its
   spacing at the result, or its least subnormal below its normal range. The
   float64 reference is double-word (lerp_exact), the float32 one plain float64,
   both far below the bound. *)
(* Ends below the normal range, whose midpoint, half the least subnormal,
   rounds. *)
let subnormal_ends =
  Quantile_case
    (D.Float64, 0x1p-53, [| 2 |], None, [| 0.5 |], [| 0.; 0x1p-1074 |])

let law_quantile (Quantile_case (dt, u, s, a, qs, vs)) =
  let x, xs = made dt s vs in
  let fmt = D.float_format dt in
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
  cover "a narrow float" (D.bits dt < 32);
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
          let a = sorted.(lo) and b = sorted.(Stdlib.min (lo + 1) (n - 1)) in
          let f = h -. Float.floor h in
          let rounded =
            if D.bits dt = 64 then f
            else Int32.float_of_bits (Int32.bits_of_float f)
          in
          let r = g.((j * per) + i) in
          let msg = Printf.sprintf "q %h of %d" q n in
          if rounded = 0. || a = b then equal ~msg float_exact a r
          else if not (Float.is_finite a && Float.is_finite b) then
            (* Beside an infinity or a NaN, [a + b]. *)
            equal ~msg float_exact (a +. b) r
          else begin
            cover "an interpolated value" true;
            let want =
              if D.bits dt = 64 then lerp_exact a b f
              else ((1. -. f) *. a) +. (f *. b)
            in
            let k = if D.bits dt = 64 then 3. else 4. in
            let spacing =
              Float.max fmt.min_normal (Float.abs want) *. fmt.epsilon
            in
            let into = if D.bits dt < 32 then spacing /. 2. else 0. in
            (* Below the normal range each of the [k] roundings errs by up to
               the precision's least subnormal: nx.mli's bound is barring
               underflow. *)
            let least = if D.bits dt = 64 then 0x1p-1074 else 0x1p-149 in
            at_most ~msg float_exact
              ~than:
                ((k *. ((u *. (Float.abs a +. Float.abs b)) +. least)) +. into)
              (Nx_support.gap r want)
          end)
        sl)
    qs

module Counted2 =
  (val Nx.devices ~kernels:(module Nx_support.Counting)
         [ Nx_support.memory 0; Nx_support.memory 1 ])

(* Each device's copy of [y], whole on each, in C order. *)
let copies y =
  Nx.sync y;
  List.map Nx_array.to_array (Iarray.to_list (Nx.Repr.shards ~by:"test" y))

(* On two devices, each device's quantiles of each probability are those it
   gives alone, by as many kernel calls as one probability takes. *)
let law_quantile_each (Quantile_case (dt, _, s, a, qs, vs)) =
  cover "no probability" (qs = [||]);
  cover "one probability" (Array.length qs = 1);
  cover "several probabilities" (Array.length qs > 1);
  let x = Nx.place Counted2.on (Nx.create dt s vs) in
  let quantile qs =
    Nx_support.Counting.reset ();
    let y = Nx.quantile ?axis:a qs x in
    let calls = Nx_support.Counting.calls () in
    (y, calls)
  in
  let got, calls = quantile qs in
  equal ~msg:"probabilities" int (Array.length qs) (Nx.shape got).(0);
  let per = Array.fold_left ( * ) 1 (Nx.shape got) / max 1 (Array.length qs) in
  Array.iteri
    (fun j q ->
      let one, single = quantile [| q |] in
      equal ~msg:"kernel calls of one probability" int single calls;
      let want = Nx.to_array one in
      List.iter
        (fun c -> equal_elements dt want (Array.sub c (j * per) per))
        (copies got))
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

(* Between finite order statistics the interpolation stays finite, however far
   apart they lie: the 0.5 quantile of [-max_float; max_float] is their
   midpoint, 0. An infinite statistic beside a finite one is that infinity. *)
let test_quantile_finite () =
  let mid (type s) (dt : (float, s) Nx.dtype) big =
    Nx.to_array (Nx.quantile [| 0.5 |] (Nx.create dt [| 2 |] [| -.big; big |]))
  in
  equal ~msg:"float64" (array float_exact) [| 0. |]
    (mid Nx.float64 Float.max_float);
  equal ~msg:"float32" (array float_exact) [| 0. |]
    (mid Nx.float32 (Int32.float_of_bits 0x7f7fffffl));
  let x = Nx.create Nx.float64 [| 2 |] [| Float.neg_infinity; 1. |] in
  equal ~msg:"-inf beside 1" (array float_exact) [| Float.neg_infinity |]
    (Nx.to_array (Nx.quantile [| 0.5 |] x));
  (* At float32 the fraction [1 - 2^-30] rounds to 1, where [(1 - f) a] would be
     [0 × -inf]. *)
  let x = Nx.create Nx.float32 [| 2 |] [| Float.neg_infinity; 1. |] in
  equal ~msg:"-inf beside 1, a fraction that rounds to 1" (array float_exact)
    [| Float.neg_infinity |]
    (Nx.to_array (Nx.quantile [| 1. -. 0x1p-30 |] x))

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

type coordinates = Coordinates : (float, 's) D.t -> coordinates

(* Points counted into their cells and weights summed there, coordinates and
   edges of one float dtype, weights of float64. *)
(* The edges, halves in [-3, 3], and the coordinates, quarters, are exact
   in each dtype drawn. *)
let law_histogram (Coordinates dt, (es, (xs, ys), ws)) =
  let f64 a = Nx.create Nx.float64 [| Array.length a |] a in
  let made a =
    let x = Nx.create dt [| Array.length a |] a in
    (x, Nx.to_array x)
  in
  let (e1t, e1), (e2t, e2) = (made (fst es), made (snd es)) in
  let (xt, xs), (yt, ys) = (made xs, made ys) in
  let on_last e x = x = e.(Array.length e - 1) in
  cover "a point on the last edge"
    (Array.exists (on_last e1) xs || Array.exists (on_last e2) ys);
  cover "a NaN coordinate" (Array.exists Float.is_nan xs);
  cover "-0. on an edge"
    (Array.exists (fun x -> x = 0. && Float.sign_bit x && Array.mem 0. e1) xs);
  cover "a narrow coordinate" (D.bits dt < 32);
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
  let dims = [ (e1t, xt); (e2t, yt) ] in
  equal ~msg:"weighted" (array float_exact) want
    (Nx.to_array (Nx.weighted_histogram ~weights:(f64 ws) dims));
  let counts = Array.make (n1 * n2) 0L in
  Array.iteri
    (fun i x ->
      match (bin e1 x, bin e2 ys.(i)) with
      | Some a, Some b ->
          counts.((a * n2) + b) <- Int64.succ counts.((a * n2) + b)
      | _ -> ())
    xs;
  let h = Nx.histogram dims in
  equal ~msg:"shape" (array int) [| n1; n2 |] (Nx.shape h);
  equal ~msg:"counts" (array int64) counts (Nx.to_array h)

let histogram_case =
  let open Gen in
  let* dt =
    of_list
      [
        Coordinates D.Float64;
        Coordinates D.Float32;
        Coordinates D.Float16;
        Coordinates D.Bfloat16;
        Coordinates D.Float8_e4m3fn;
      ]
  in
  let* e1 = edges_gen in
  let* e2 = edges_gen in
  let* n = int_range 0 12 in
  let* xs = array ~size:(constant n) coord in
  let* ys = array ~size:(constant n) coord in
  let+ ws = array ~size:(constant n) (float_range (-2.) 2.) in
  (dt, ((e1, e2), (xs, ys), ws))

(* [-0.] on the edge [0.], points on each last edge and a NaN. *)
let histogram_edges =
  ( Coordinates D.Float64,
    ( ([| -1.; 0.; 1. |], [| 0.; 2. |]),
      ([| -0.; 1.; nan |], [| 0.5; 2.; 1. |]),
      [| 1.; 2.; 3. |] ) )

let test_histogram_cases () =
  let e = Nx.create Nx.float64 [| 4 |] [| 0.; 1.; 2.; 3. |] in
  let x = Nx.create Nx.float64 [| 7 |] [| 0.; 0.5; 1.; 3.; 3.5; nan; -0. |] in
  equal (array int64) [| 3L; 1L; 1L |] (Nx.to_array (Nx.histogram [ (e, x) ]));
  let e16 = Nx.cast Nx.float16 e and x16 = Nx.cast Nx.float16 x in
  equal (array int64) [| 3L; 1L; 1L |]
    (Nx.to_array (Nx.histogram [ (e16, x16) ]));
  (* Weights in their own dtype: integers, and float64 over float16
     coordinates, whose sum float16 would round. *)
  let w = Nx.create Nx.int32 [| 7 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l |] in
  equal (array int32) [| 10l; 3l; 4l |]
    (Nx.to_array (Nx.weighted_histogram ~weights:w [ (e16, x16) ]));
  let w = Nx.create Nx.float64 [| 7 |] [| 1e-4; 0.; 1.; 0.; 1.; 0.; 1. |] in
  equal (array float_exact) [| 1. +. 1e-4; 1.; 0. |]
    (Nx.to_array (Nx.weighted_histogram ~weights:w [ (e16, x16) ]));
  (* float16 weights sum in float32 and round once: 1 and 2^-11 twice make
     1 + 2^-10, where each float16 addition would round to 1. *)
  let h = Float.ldexp 1. (-11) in
  let w = Nx.create Nx.float16 [| 7 |] [| 1.; h; 0.; 0.; 0.; 0.; h |] in
  equal (array float_exact)
    [| 1. +. Float.ldexp 1. (-10); 0.; 0. |]
    (Nx.to_array (Nx.weighted_histogram ~weights:w [ (e16, x16) ]));
  let c re im = { Complex.re; im } in
  let w =
    Nx.create Nx.complex64 [| 7 |]
      [| c 1. 2.; c 3. (-1.); c 0. 5.; c 0. 0.; c 9. 9.; c 9. 9.; c 0. 0. |]
  in
  let z = Nx.to_array (Nx.weighted_histogram ~weights:w [ (e, x) ]) in
  equal ~msg:"real parts" (array float_exact) [| 4.; 0.; 0. |]
    (Array.map (fun z -> z.Complex.re) z);
  equal ~msg:"imaginary parts" (array float_exact) [| 1.; 5.; 0. |]
    (Array.map (fun z -> z.Complex.im) z);
  (* int4 weights wrap: 7 + 7 is -2. *)
  let w = Nx.create Nx.int4 [| 7 |] [| 7; 7; 1; -8; 3; 3; 0 |] in
  equal (array int) [| -2; 1; -8 |]
    (Nx.to_array (Nx.weighted_histogram ~weights:w [ (e, x) ]));
  let empty = Nx.zeros Nx.float64 [| 0 |] in
  equal (array int64) [| 0L; 0L; 0L |]
    (Nx.to_array (Nx.histogram [ (e, empty) ]));
  equal (array float_exact) [| 0.; 0.; 0. |]
    (Nx.to_array (Nx.weighted_histogram ~weights:empty [ (e, empty) ]));
  invalid ~by:"Nx.histogram" (fun () ->
      Nx.histogram ([] : ((float, D.float64_elt, Nx.host) Nx.t * _) list));
  invalid ~by:"Nx.histogram" (fun () ->
      Nx.histogram [ (Nx.zeros Nx.float64 [| 1 |], x) ]);
  invalid ~by:"Nx.histogram" (fun () ->
      Nx.histogram [ (e, x); (e, Nx.zeros Nx.float64 [| 2 |]) ]);
  invalid ~by:"Nx.weighted_histogram" (fun () ->
      Nx.weighted_histogram ~weights:e [ (e, x) ]);
  invalid ~by:"Nx.weighted_histogram" (fun () ->
      Nx.weighted_histogram ~weights:(Nx.zeros Nx.bool [| 7 |]) [ (e, x) ]);
  invalid ~by:"Nx.weighted_histogram" (fun () ->
      Nx.weighted_histogram ~weights:x
        ([] : ((float, D.float64_elt, Nx.host) Nx.t * _) list))

(* A count is exact whatever the coordinates' dtype: float16 holds every
   integer only up to 2048, bfloat16 up to 256 and float8_e4m3fn up to 16,
   so one more point in one bin tells a count rounded into the coordinates'
   dtype. *)
let test_histogram_narrow_counts () =
  let count (type s) (dt : (float, s) Nx.dtype) n =
    let e = Nx.cast dt (Nx.create Nx.float64 [| 2 |] [| 0.; 1. |]) in
    let x = Nx.zeros dt [| n |] in
    equal ~msg:(D.name dt) (array int64) [| Int64.of_int n |]
      (Nx.to_array (Nx.cast Nx.int64 (Nx.histogram [ (e, x) ])))
  in
  count Nx.float16 2049;
  count Nx.bfloat16 257;
  count Nx.float8_e4m3fn 17

let () =
  exit
    (run "nx order"
       [
         group "sort argsort top_k"
           [
             prop ~count:200
               "a stable sort of each slice, its values its positions' \
                elements bit for bit, top_k its descending first k"
               sort_case law_sort;
             test "the doc's orders" test_sort_rows;
             test "raises on an axis outside the rank or k outside [0, n]"
               test_sort_refusals;
           ];
         group "order_key"
           [
             prop "keys compare as the sort order and are its formulas" key_case
               law_key;
             test "the doc's keys" test_key_rows;
             test "raises on complex, a dtype not of the four, a narrower one"
               test_key_refusals;
           ];
         group "lexsort searchsorted"
           [
             prop "lexsort is a stable sort of the rows" rows_case law_lexsort;
             test
               "lexsort of no column leaves every row; raises on 3-d or complex"
               test_lexsort_rows;
             prop "searchsorted counts the keys before, or before or equal"
               search_case law_search;
             test "the doc's knots" test_search_rows;
             test "raises on 0-d or complex keys, rows of another width"
               test_search_refusals;
             test "the doc's counts of sorts and gathers" test_counts;
           ];
         group "searchsorted_along_axis"
           [
             (* Each view, and a lane of no key, is about one case in ten. *)
             prop ~count:400 "each lane is searchsorted of that lane alone"
               along_case law_along;
             test "the doc's lanes and refusals" test_along_cases;
           ];
         group "lengths that are data"
           [
             test
               "a traced length raises naming the function, unless size fixes \
                it"
               test_traced;
           ];
         group "positions"
           [
             (* A size of the total and an empty count are each about one
                case in eight. *)
             prop ~count:300 "each index repeated its count" count_case
               law_positions;
             test "the doc's positions, pads past the counts and refusals"
               test_positions_cases;
             test "9000 positions, with and without a size" test_positions_long;
           ];
         group "argwhere"
           [
             prop "the coordinates of each element other than zero"
               argwhere_case law_argwhere;
             test "the doc's coordinates and refusals" test_argwhere_cases;
           ];
         group "nonzero extract"
           [
             prop "nonzero is argwhere's columns" argwhere_case law_nonzero;
             test "nonzero's columns, pads and refusals" test_nonzero_cases;
             prop "extract reads where the condition is not zero" extract_case
               law_extract;
             test "extract's elements, fill and refusals" test_extract_cases;
           ];
         group "unique"
           [
             prop ~count:200 ~examples:near_rows
               "groups in order of first appearance" unique_case law_unique;
             test "the doc's groups" test_unique_rows;
             test "raises on a negative size, 3-d or complex keys"
               test_unique_refusals;
             test "one Unique numbers an unsized unique's keys" test_unique_one;
             test "groups 70,000 keys past the first numbering's 4096 entries"
               test_unique_long;
             test "50,000 groups of 80,000 keys" test_unique_many;
           ];
         group "quantile"
           [
             prop ~examples:[ subnormal_ends ]
               "interpolates between order statistics within its bound"
               quantile_case law_quantile;
             prop "each probability as alone, by one probability's kernels"
               quantile_case law_quantile_each;
             test "the doc's quantiles and refusals" test_quantile_cases;
             test "between finite statistics the interpolation is finite"
               test_quantile_finite;
           ];
         group "histogram"
           [
             prop ~examples:[ histogram_edges ]
               "points counted into their cells" histogram_case law_histogram;
             test "the doc's cells and refusals" test_histogram_cases;
             test "counts are exact in narrow coordinates"
               test_histogram_narrow_counts;
           ];
       ])
