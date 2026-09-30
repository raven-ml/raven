(* Tests of Tolk_next.Decomp_dtype: the pass rewrites kernels as tinygrad's
   does, its conversions of the narrow floats are IEEE conversions that give the
   codes Dtype folds (DIVERGENCES D9), and its 64-bit integers compute what
   64-bit integers compute. *)

open Windtrap
open Tolk_next
open Dtypes
module PM = Ops.Pattern_matcher

let target = Result.get_ok (Helpers.Target.parse "")
let rejects f = raises_match (Exn.invalid_arg ?substring:None) f

(* Targets and the pass *)

(* [lacking dts] is a target that has every data type but [dts]; [told names f]
   is [f ()] with the setting that names the data types to emulate. *)
let lacking dts = Renderer.v ~native:(fun dt -> not (List.mem dt dts)) target
let everything = lacking []
let told names f = Helpers.context [ B (Helpers.emulated_dtypes, names) ] f

(* [emulate on kernel] is [kernel] rewritten for the target [on] by the pass as
   code generation runs it, with the weak constants of its rules committed. *)
let emulate on kernel =
  Ops.graph_rewrite ~ctx:(Decomp_dtype.ctx on) kernel
    (PM.append Decomp_dtype.pm_dtype_decomps
       (PM.with_ctx Uop_weak.pm_commit_weak))

let narrows =
  Dtype.[ Float16; Bfloat16; Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz ]

let longs = Dtype.[ Int64; Uint64 ]
let storage dt = if Dtype.bitsize dt = 8 then Dtype.Uint8 else Dtype.Uint16
let size dt = 1 lsl Dtype.bitsize dt
let is_16_bit dt = Dtype.bitsize dt = 16

(* Codes and values *)

let as_int v =
  match (v :> Dtype.const) with
  | `Int z -> Z.to_int z
  | v -> failf "%a is not an integer" (Testable.pp const) v

let as_float v =
  match (v :> Dtype.const) with
  | `Float x -> x
  | v -> failf "%a is not a float" (Testable.pp const) v

(* [encode dt v] is the code of [v] as a [dt] value, rounded as Dtype folds it;
   [decode dt c] is the value of the code [c]. *)
let encode dt v = as_int (Dtype.bitcast dt (storage dt) (Dtype.truncate dt v))
let decode dt c = as_float (Dtype.bitcast (storage dt) dt (`Int (Z.of_int c)))

(* [bits x] is the code of the float32 [x]; [f32 b] is the float32 of the code
   [b]. *)
let bits x = as_int (Dtype.bitcast Float32 Uint32 (`Float x))
let f32 b = as_float (Dtype.bitcast Uint32 Float32 (`Int (Z.of_int b)))
let is_nan_code dt c = Float.is_nan (decode dt c)

(* [greatest dt] is the code of [dt]'s greatest finite value, [top dt] that
   value, and [edge dt] that value plus half an ulp. *)
let greatest dt =
  let rec down c = if Float.is_finite (decode dt c) then c else down (c - 1) in
  down ((size dt / 2) - 1)

let top dt = decode dt (greatest dt)
let edge dt = top dt +. ((top dt -. decode dt (greatest dt - 1)) /. 2.)

(* [below x] is the float32 before the positive float32 [x]. *)
let below x = f32 (bits x - 1)

(* [codes ~all dt] is the codes of [dt] that are not NaNs: all of them if [all],
   and otherwise a sample of both signs, the codes nearest zero, each side of
   the least normal and of the greatest finite value, the infinities, and every
   5th code of an 8-bit float or 331st of a 16-bit one. *)
let codes ~all dt =
  let n = size dt in
  let every =
    if all then List.init n Fun.id
    else
      let half = n / 2 and least_normal = 1 lsl snd (Dtype.finfo dt) in
      let stride = if n = 256 then 5 else 331 in
      let near c = List.init 9 (fun d -> c + d - 4) in
      List.concat
        [
          List.init 16 Fun.id;
          near least_normal;
          near (greatest dt);
          [ greatest dt + 1 ];
          List.init (half / stride) (fun k -> k * stride);
        ]
      |> List.filter (fun c -> c >= 0 && c < half)
      |> List.concat_map (fun c -> [ c; c + half ])
      |> List.sort_uniq compare
  in
  List.filter (fun c -> not (is_nan_code dt c)) every

let nan_codes dt = List.filter (is_nan_code dt) (List.init (size dt) Fun.id)

(* [neighbours ~all dt] is the pairs of the finite values of neighbouring codes
   of [codes ~all dt], the second of greater magnitude. *)
let neighbours ~all dt =
  let finite c = c < size dt && Float.is_finite (decode dt c) in
  List.filter_map
    (fun c ->
      if
        finite c
        && finite (c + 1)
        && Float.abs (decode dt (c + 1)) > Float.abs (decode dt c)
      then Some (decode dt c, decode dt (c + 1))
      else None)
    (codes ~all dt)

(* [narrowing_inputs ~all dt] is the float32 codes a narrowing to [dt] is
   checked at, of both signs: the value of each finite code of [codes ~all dt]
   and the float32s one and two ulps each side of it; each midpoint between
   neighbouring codes and one ulp each side of it; the greatest finite value
   plus half an ulp and one ulp each side of it; and values beyond every narrow
   float. *)
let narrowing_inputs ~all dt =
  let around x ds =
    List.filter
      (fun b -> b >= 0)
      (List.map (fun d -> bits (Float.abs x) + d) ds)
  in
  let of_code c =
    let v = decode dt c in
    if Float.is_finite v then around v [ -2; -1; 0; 1; 2 ] else []
  in
  List.concat
    [
      List.concat_map of_code (codes ~all dt);
      List.concat_map
        (fun (v, w) -> around ((v +. w) /. 2.) [ -1; 0; 1 ])
        (neighbours ~all dt);
      around (edge dt) [ -1; 0; 1 ];
      List.map bits
        [ Float.infinity; 1e10; 3.4028234663852886e38; 1e-30; 1e-45 ];
    ]
  |> List.concat_map (fun b -> [ b; b lor 0x80000000 ])
  |> List.sort_uniq compare

(* [no_misses what misses] fails, naming the first of [misses], unless it is
   empty. *)
let no_misses what = function
  | [] -> ()
  | misses ->
      failf "%d %s differ; the first:@\n%s" (List.length misses) what
        (String.concat "\n" (List.filteri (fun i _ -> i < 12) misses))

let hex =
  Testable.make ~pp:(fun ppf c -> Format.fprintf ppf "0x%x" c) ~equal:Int.equal

(* [float32] compares floats as float32 codes. *)
let float32 =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%h" x)
    ~equal:(fun a b -> bits a = bits b)

(* [same_nan dt want got] is [true] iff the codes [want] and [got] of [dt] are
   equal, or are NaNs of the same sign and [dt] is {!Dtype.Fp8e5m2}: Dtype keeps
   the payload of the 16-bit NaNs only, and the other 8-bit floats have one NaN
   code of each sign. *)
let same_nan dt want got =
  want = got
  || dt = Dtype.Fp8e5m2 && is_nan_code dt want && is_nan_code dt got
     && want lsr 7 = got lsr 7

(* [code_of dt] compares codes of [dt] with {!same_nan}. *)
let code_of dt =
  Testable.make
    ~pp:(fun ppf c -> Format.fprintf ppf "0x%x (%h)" c (decode dt c))
    ~equal:(same_nan dt)

let code c = `Int (Z.of_int c)

(* f2f *)

let at_code u c = Interpreter.eval ~params:[ (0, code c) ] u

(* [widening dt] is the float32 code that f2f widens a [dt] code to; [narrowing
   ~sat dt] is the [dt] code that f2f narrows a float32 code to. *)
let widening dt =
  Ops.bitcast (Decomp_dtype.f2f (Ops.param 0 (storage dt)) dt Float32) Uint32

let narrowing ?sat dt =
  Decomp_dtype.f2f ?sat (Ops.param 0 Dtype.Uint32) Float32 dt

let widens_exactly ~all dt () =
  let u = widening dt in
  List.filter_map
    (fun c ->
      let want = decode dt c and got = f32 (as_int (at_code u c)) in
      if bits want = bits got then None
      else Some (Printf.sprintf "0x%x: want %h, got %h" c want got))
    (codes ~all dt)
  |> no_misses "codes"

let narrows_as_dtype_folds ~all dt () =
  let u = narrowing dt in
  List.filter_map
    (fun b ->
      let want = encode dt (`Float (f32 b)) and got = as_int (at_code u b) in
      if want = got then None
      else
        Some
          (Printf.sprintf "%h (0x%08x): want 0x%x (%h), got 0x%x (%h)" (f32 b) b
             want (decode dt want) got (decode dt got)))
    (narrowing_inputs ~all dt)
  |> no_misses "float32s"

(* [exhaustive name law] is the slow tests of [law ~all:true] for each narrow
   float, named [name] of its name. *)
let exhaustive name law =
  List.map (fun dt -> slow (name (alias dt)) (law ~all:true dt)) narrows

let f2f =
  group "f2f"
    (List.concat_map
       (fun dt ->
         [
           test
             ("widening a " ^ alias dt ^ " to a float32 is exact")
             (widens_exactly ~all:false dt);
           test
             ("narrowing a float32 to a " ^ alias dt ^ " rounds as Dtype folds")
             (narrows_as_dtype_folds ~all:false dt);
         ])
       narrows
    @ exhaustive
        (Printf.sprintf "widening every %s code is exact")
        widens_exactly
    @ exhaustive
        (Printf.sprintf
           "narrowing to every %s code and its neighbours rounds as Dtype folds")
        narrows_as_dtype_folds
    @ [
        test
          "without saturation, an 8-bit float overflows to its infinity or NaN"
          (fun () ->
            List.iter
              (fun dt ->
                equal ~msg:(alias dt) hex
                  (encode dt (`Float Float.infinity))
                  (as_int (at_code (narrowing ~sat:false dt) (bits 1e10))))
              Dtype.fp8s);
        test "f2f converts only between a narrow float and a float32" (fun () ->
            let v = Ops.param 0 Dtype.Uint16 in
            rejects (fun () -> Decomp_dtype.f2f v Float16 Bfloat16);
            rejects (fun () -> Decomp_dtype.f2f v Float16 Float64);
            rejects (fun () ->
                Decomp_dtype.f2f (Ops.param 0 Uint32) Float32 Float32);
            rejects (fun () ->
                Decomp_dtype.f2f (Ops.param 0 Uint64) Float64 Float16);
            rejects (fun () -> Decomp_dtype.f2f v Int16 Float32);
            rejects (fun () ->
                Decomp_dtype.f2f (Ops.param 0 Uint32) Float32 Float64);
            rejects (fun () ->
                Decomp_dtype.f2f (Ops.param 0 Uint64) Float64 Float32));
      ])

(* f2f_clamp *)

let clamped ?sat dt x =
  as_float
    (Interpreter.eval
       ~params:[ (0, `Float x) ]
       (Decomp_dtype.f2f_clamp ?sat (Ops.param 0 Dtype.Float32) dt))

let infinity_of s = Float.copy_sign Float.infinity s

(* [overflows_at_the_edge ?sat dt] checks that the clamp keeps [dt]'s greatest
   value and the float32 below the edge, and is an infinity from the edge. *)
let overflows_at_the_edge ?sat dt =
  List.iter
    (fun s ->
      let x = clamped ?sat dt in
      equal ~msg:"the greatest value" float32 (s *. top dt) (x (s *. top dt));
      equal ~msg:"below the edge" float32
        (s *. below (edge dt))
        (x (s *. below (edge dt)));
      equal ~msg:"the edge" float32 (infinity_of s) (x (s *. edge dt));
      equal ~msg:"an infinity" float32 (infinity_of s) (x (infinity_of s)))
    [ 1.; -1. ]

let f2f_clamp =
  group "f2f_clamp"
    [
      cases ~name:alias
        "a 16-bit float's clamp is an infinity from its greatest value plus \
         half an ulp"
        Dtype.[ Float16; Bfloat16 ]
        (fun dt ->
          overflows_at_the_edge dt;
          overflows_at_the_edge ~sat:true dt);
      cases ~name:alias
        "without saturation, an 8-bit float's clamp is an infinity from its \
         greatest value plus half an ulp"
        Dtype.fp8s
        (overflows_at_the_edge ~sat:false);
      cases ~name:alias
        "with saturation, an 8-bit float's clamp is its greatest value and \
         keeps infinities"
        Dtype.fp8s (fun dt ->
          List.iter
            (fun s ->
              let m = s *. top dt in
              equal ~msg:"far above" float32 m (clamped dt (s *. 1e10));
              equal ~msg:"the edge" float32 m (clamped dt (s *. edge dt));
              equal ~msg:"just above" float32 m
                (clamped dt (s *. f32 (bits (top dt) + 1)));
              equal ~msg:"just below" float32
                (s *. below (top dt))
                (clamped dt (s *. below (top dt)));
              equal ~msg:"an infinity" float32 (infinity_of s)
                (clamped dt (infinity_of s)))
            [ 1.; -1. ]);
      test "a clamped NaN is a NaN" (fun () ->
          List.iter
            (fun dt ->
              is_true ~msg:(alias dt) (Float.is_nan (clamped dt Float.nan)))
            narrows);
    ]

(* Kernels *)

(* [kernel ins out n f] is the kernel that stores, for each [r] below [n], [f]
   of the elements [r] of the storage of [ins], in slots 0, 1, and so on, at
   element [r] of the [out] storage in the next slot. *)
let kernel ins out n f =
  let r = Ops.range (Int n) [ 0 ] in
  let at slot dt = Ops.index (Ops.param ~shape:[ Int n ] slot dt) [ r ] in
  let xs = List.mapi (fun slot dt -> Ops.load (at slot dt) []) ins in
  Ops.sink [ Ops.end_ (Ops.store (at (List.length ins) out) (f xs)) [ r ] ]

let cast_to dt xs = Ops.cast (List.hd xs) dt

(* [outputs k inputs] is what the kernel [k] writes to its output given the
   storage [inputs], in element order; [written ~on k inputs] is that of [k]
   emulated on the target [on]. *)
let outputs k inputs =
  let out = List.length inputs in
  Interpreter.writes
    ~buffers:(List.mapi (fun slot a -> (slot, Array.of_list a)) inputs)
    k
  |> List.filter_map (fun (slot, _, v) -> if slot = out then Some v else None)

let written ~on k inputs = outputs (emulate on k) inputs
let on_narrows = lacking narrows

(* [converts ~from ~to_ inputs want] checks that the emulated cast of each of
   [inputs], values of [from], to the narrow float [to_] writes the code [want
   x]. *)
let converts ?(on = on_narrows) ~from ~to_ inputs want =
  let k = kernel [ from ] to_ (List.length inputs) (cast_to to_) in
  List.combine inputs (written ~on k [ inputs ])
  |> List.filter_map (fun (x, got) ->
      let want = want x and got = as_int got in
      if same_nan to_ want got then None
      else
        Some
          (Format.asprintf "%a: want 0x%x (%h), got 0x%x (%h)"
             (Testable.pp value) x want (decode to_ want) got (decode to_ got)))
  |> no_misses "casts"

let copies_every_code ~all dt () =
  let cs = codes ~all dt in
  let k = kernel [ dt ] dt (List.length cs) List.hd in
  List.combine cs (written ~on:on_narrows k [ List.map code cs ])
  |> List.filter_map (fun (c, got) ->
      if c = as_int got then None
      else Some (Printf.sprintf "0x%x: got 0x%x" c (as_int got)))
  |> no_misses "codes"

let casts_float32s ~all dt () =
  converts ~from:Float32 ~to_:dt
    (List.map (fun b -> `Float (f32 b)) (narrowing_inputs ~all dt))
    (encode dt)

let widens_to_float32 ~all dt () =
  let cs = codes ~all dt in
  let k = kernel [ dt ] Float32 (List.length cs) (cast_to Float32) in
  List.combine cs (written ~on:on_narrows k [ List.map code cs ])
  |> List.filter_map (fun (c, got) ->
      let want = decode dt c and got = as_float got in
      if bits want = bits got then None
      else Some (Printf.sprintf "0x%x: want %h, got %h" c want got))
  |> no_misses "codes"

(* [near_ties ~all dt] is the doubles of both signs near the ties of [dt]: each
   midpoint between neighbouring codes, and the midpoint scaled by [1 ± 2^-40],
   which a float32 rounds onto the midpoint; and the same about the greatest
   finite value plus half an ulp. *)
let near_ties ~all dt =
  let near m = [ m; m *. (1. +. 0x1p-40); m *. (1. -. 0x1p-40) ] in
  List.concat_map (fun (v, w) -> near ((v +. w) /. 2.)) (neighbours ~all dt)
  @ near (edge dt)
  |> List.concat_map (fun x -> [ `Float x; `Float (-.x) ])

(* [near_integer_ties ~all dt (lo, hi)] is the integers in [lo, hi] each side of
   the ties of [dt] that are integers, and the edges of the range. *)
let near_integer_ties ~all dt (lo, hi) =
  let near_mid (v, w) =
    let m = (v +. w) /. 2. in
    if Float.is_integer m && Float.abs m >= 1. then
      List.map (fun d -> Z.(of_float m + of_int d)) [ -1; 0; 1 ]
    else []
  in
  (List.concat_map
     (fun (v, w) -> near_mid (v, w) @ near_mid (-.v, -.w))
     (neighbours ~all dt)
  @ Z.[ lo; succ lo; pred hi; hi; zero; one ])
  |> List.filter (fun z -> Z.leq lo z && Z.leq z hi)
  |> List.sort_uniq Z.compare
  |> List.map (fun z -> `Int z)

let casts_doubles ~all dt () =
  converts ~from:Float64 ~to_:dt (near_ties ~all dt) (encode dt)

(* The integer types more precise than a float32. *)
let wide_integers = Dtype.[ Int32; Uint32; Int64; Uint64 ]

let casts_integers ~all dt () =
  List.iter
    (fun from ->
      converts ~from ~to_:dt
        (near_integer_ties ~all dt (int_bounds from))
        (encode dt))
    wide_integers

let casts_narrow_integers ~all dt () =
  List.iter
    (fun from ->
      let lo, hi = int_bounds from in
      let values =
        if Dtype.bitsize from = 8 then
          List.init (Z.to_int Z.(hi - lo) + 1) (fun k -> `Int Z.(lo + of_int k))
        else near_integer_ties ~all dt (lo, hi)
      in
      converts ~from ~to_:dt values (encode dt))
    Dtype.[ Int8; Uint8; Int16; Uint16 ]

(* [undefined_casts ~on k inputs] is the elements at which the kernel [k],
   emulated on [on], converts a float to an integer type that cannot hold it,
   which C leaves undefined; given the storage [inputs]. *)
let undefined_casts ~on k inputs =
  let nodes = Ops.toposort (emulate on k) in
  let r = List.find (fun u -> Ops.op u = Range) nodes in
  let casts =
    List.filter
      (fun u ->
        Ops.op u = Cast
        && Dtype.is_int (Ops.dtype u)
        && Dtype.is_float (Ops.dtype (Ops.nth u 0)))
      nodes
  in
  let buffers = List.mapi (fun slot a -> (slot, Array.of_list a)) inputs in
  let undefined vars c =
    match Interpreter.eval ~vars ~buffers (Ops.nth c 0) with
    | `Float x ->
        let lo, hi = int_bounds (Ops.dtype c) in
        Float.is_nan x
        || Float.trunc x < Z.to_float lo
        || Float.trunc x > Z.to_float hi
    | _ -> false
  in
  List.filter
    (fun i ->
      List.exists
        (undefined [ (Option.get (Interpreter.name r), code i) ])
        casts)
    (List.init (List.length (List.hd inputs)) Fun.id)

let casts_integers_in_range dt () =
  List.iter
    (fun from ->
      let values = near_integer_ties ~all:false dt (int_bounds from) in
      let k = kernel [ from ] dt (List.length values) (cast_to dt) in
      equal ~msg:(alias from) (list value) []
        (List.map (List.nth values)
           (undefined_casts ~on:on_narrows k [ values ])))
    wide_integers

(* [casts_between ~all fr to_] checks that an emulated cast between two emulated
   narrow floats rounds each code of [fr] once to [to_]. *)
let casts_between ~all fr to_ () =
  converts ~from:fr ~to_
    (List.map code (codes ~all fr))
    (fun c -> encode to_ (`Float (decode fr (as_int c))))

let pairs =
  List.concat_map
    (fun fr ->
      List.filter_map
        (fun to_ -> if fr = to_ then None else Some (fr, to_))
        narrows)
    narrows

let pair_name (fr, to_) = alias fr ^ " to " ^ alias to_

let arithmetic =
  [
    ("add", Op.Add, Ops.O.( + ));
    ("sub", Op.Sub, Ops.O.( - ));
    ("mul", Op.Mul, Ops.O.( * ));
    ("max", Op.Max, Ops.maximum);
  ]

(* [in_float32 op dt a b] is the code of [op] on the values of the [dt] codes
   [a] and [b], computed in float32 and rounded to [dt]. *)
let in_float32 op dt a b =
  let x =
    Ops.exec_alu op Float32 [ `Float (decode dt a); `Float (decode dt b) ]
  in
  encode dt (`Float (as_float x))

let computes_in_float32 dt =
  let kernels =
    List.map
      (fun (name, op, f) ->
        let k =
          kernel [ dt; dt ] dt 1 (fun xs -> f (List.nth xs 0) (List.nth xs 1))
        in
        (name, (op, lazy (emulate on_narrows k))))
      arithmetic
  in
  let finite = Array.of_list (codes ~all:true dt) in
  let operand =
    Gen.map (Array.get finite) (Gen.int_range 0 (Array.length finite - 1))
  in
  let pp ppf (name, a, b) = Format.fprintf ppf "%s 0x%x 0x%x" name a b in
  prop ~count:300
    ("emulated " ^ alias dt
   ^ " arithmetic computes in float32 and rounds once, at the store")
    (Gen.with_pp pp
       (Gen.triple
          (Gen.of_list ~pp:Format.pp_print_string (List.map fst kernels))
          operand operand))
    (fun (name, a, b) ->
      let op, k = List.assoc name kernels in
      let got = outputs (Lazy.force k) [ [ code a ]; [ code b ] ] in
      equal (list (code_of dt)) [ in_float32 op dt a b ] (List.map as_int got))

let comparisons_read_values dt () =
  let cs = codes ~all:false dt in
  let a = cs @ cs and b = List.rev (cs @ cs) in
  let k =
    kernel [ dt; dt ] Bool (List.length a) (fun xs ->
        Ops.O.(List.nth xs 0 < List.nth xs 1))
  in
  let want = List.map2 (fun a b -> `Bool (decode dt a < decode dt b)) a b in
  equal (list value) want
    (written ~on:on_narrows k [ List.map code a; List.map code b ])

(* [quieted dt c] is the code [c] of [dt] through a conversion to float32 and
   back: itself, but a signalling NaN, which a conversion quiets by setting the
   top bit of its mantissa. The fnuz formats' one NaN and e4m3's are quiet. *)
let quieted dt c =
  let quiet = 1 lsl (snd (Dtype.finfo dt) - 1) in
  if is_nan_code dt c && not (List.mem dt Dtype.fp8_fnuz) then c lor quiet
  else c

let selects_codes cs dt () =
  let others = List.rev cs in
  let conds = List.mapi (fun i _ -> i mod 2 = 0) cs in
  let k =
    kernel [ Bool; dt; dt ] dt (List.length cs) (fun xs ->
        Ops.where (List.nth xs 0) (List.nth xs 1) (List.nth xs 2))
  in
  let want =
    List.map2
      (fun c (x, y) -> quieted dt (if c then x else y))
      conds (List.combine cs others)
  in
  equal (list hex) want
    (List.map as_int
       (written ~on:on_narrows k
          [
            List.map (fun c -> `Bool c) conds;
            List.map code cs;
            List.map code others;
          ]))

(* A bit reinterpretation of a narrow float reads or writes its storage. *)
let reinterprets ~all dt () =
  let st = storage dt in
  let cs =
    if all || size dt = 256 then List.init (size dt) Fun.id
    else codes ~all dt @ nan_codes dt |> List.filteri (fun i _ -> i < 4096)
  in
  let read =
    kernel [ dt ] st (List.length cs) (fun xs -> Ops.bitcast (List.hd xs) st)
  in
  equal ~msg:"read" (list hex) cs
    (List.map as_int (written ~on:on_narrows read [ List.map code cs ]));
  let finite = codes ~all dt in
  let write =
    kernel [ st ] dt (List.length finite) (fun xs ->
        Ops.bitcast (List.hd xs) dt)
  in
  equal ~msg:"written" (list hex) finite
    (List.map as_int (written ~on:on_narrows write [ List.map code finite ]))

let computed_then_reinterpreted dt () =
  let st = storage dt and cs = codes ~all:false dt in
  let k =
    kernel [ dt ] st (List.length cs) (fun xs ->
        Ops.bitcast Ops.O.(List.hd xs + float 1.) st)
  in
  let one = encode dt (`Float 1.) in
  equal
    (list (code_of dt))
    (List.map (fun c -> in_float32 Add dt c one) cs)
    (List.map as_int (written ~on:on_narrows k [ List.map code cs ]))

let emulated_floats =
  group "emulated narrow floats"
    (List.concat_map
       (fun dt ->
         let name = alias dt in
         [
           test
             ("an emulated copy of " ^ name ^ "s keeps every code")
             (copies_every_code ~all:false dt);
           test
             ("an emulated cast of float32s to " ^ name
            ^ " rounds as Dtype folds")
             (casts_float32s ~all:false dt);
           test
             ("an emulated cast of a " ^ name ^ " to a float32 is exact")
             (widens_to_float32 ~all:false dt);
           test
             ("an emulated cast of doubles to " ^ name ^ " rounds once")
             (casts_doubles ~all:false dt);
           test
             ("an emulated cast of integers more precise than a float32 to "
            ^ name ^ " rounds once")
             (casts_integers ~all:false dt);
           test
             ("an emulated cast of 8 and 16-bit integers to " ^ name
            ^ " rounds as Dtype folds")
             (casts_narrow_integers ~all:false dt);
           test
             ("an emulated cast of an integer to " ^ name
            ^ " converts no float to an integer that cannot hold it")
             (casts_integers_in_range dt);
           computes_in_float32 dt;
           test
             ("an emulated comparison of " ^ name ^ "s compares their values")
             (comparisons_read_values dt);
           test
             ("an emulated selection of " ^ name ^ "s keeps their codes")
             (selects_codes (codes ~all:false dt) dt);
           test
             ("a bit reinterpretation of a " ^ name
            ^ " reads and writes its storage")
             (reinterprets ~all:false dt);
           test
             ("a " ^ name
            ^ " computed, then reinterpreted, is the code of the result")
             (computed_then_reinterpreted dt);
         ])
       narrows
    @ [
        cases ~name:pair_name
          "an emulated cast between two emulated narrow floats rounds once"
          Dtype.
            [
              (Float16, Fp8e4m3);
              (Bfloat16, Float16);
              (Float16, Bfloat16);
              (Fp8e5m2, Fp8e4m3fnuz);
            ]
          (fun (fr, to_) -> casts_between ~all:false fr to_ ());
      ]
    @ exhaustive
        (Printf.sprintf "an emulated copy of every %s code keeps it")
        copies_every_code
    @ exhaustive
        (Printf.sprintf
           "an emulated cast of float32s to every %s code and its neighbours \
            rounds as Dtype folds")
        casts_float32s
    @ exhaustive
        (Printf.sprintf "an emulated cast of every %s to a float32 is exact")
        widens_to_float32
    @ exhaustive
        (Printf.sprintf
           "an emulated cast of doubles near every %s tie rounds once")
        casts_doubles
    @ exhaustive
        (Printf.sprintf
           "an emulated cast of integers near every %s tie rounds once")
        casts_integers
    @ exhaustive
        (Printf.sprintf
           "an emulated cast of 16-bit integers near every %s tie rounds once")
        casts_narrow_integers
    @ exhaustive
        (Printf.sprintf
           "a bit reinterpretation of every %s reads and writes its storage")
        reinterprets
    @ List.map
        (fun p ->
          slow
            ("an emulated cast of every " ^ pair_name p ^ " rounds once")
            (casts_between ~all:true (fst p) (snd p)))
        pairs)

(* NaNs *)

(* The quiet float32 NaNs, and the signalling ones, of both signs. *)
let quiet_nans = [ 0x7fc00000; 0x7fc12345; 0x7fe00001; 0x7fffffff ]
let signalling_nans = [ 0x7f800001; 0x7fa00000; 0x7fbfffff ]
let both_signs = List.concat_map (fun b -> [ b; b lor 0x80000000 ])

(* [is_nan_of dt ~negative c] is [true] iff [c] is a NaN code of [dt] with the
   sign [negative], or the one NaN of an fnuz format. *)
let is_nan_of dt ~negative c =
  is_nan_code dt c
  && (List.mem dt Dtype.fp8_fnuz || c lsr (Dtype.bitsize dt - 1) = 1 = negative)

let copies cs dt =
  let k = kernel [ dt ] dt (List.length cs) List.hd in
  equal (list hex)
    (List.map (quieted dt) cs)
    (List.map as_int (written ~on:on_narrows k [ List.map code cs ]))

let nans =
  group "NaNs"
    [
      test "f2f widens a NaN code to a float32 NaN of its sign and payload"
        (fun () ->
          List.iter
            (fun dt ->
              let u = widening dt in
              List.iter
                (fun c ->
                  let got = f32 (as_int (at_code u c)) in
                  is_true
                    ~msg:(Printf.sprintf "%s 0x%x is %h" (alias dt) c got)
                    (Float.is_nan got
                    && Float.sign_bit got = Float.sign_bit (decode dt c));
                  if is_16_bit dt then
                    equal ~msg:(alias dt) hex (bits (decode dt c)) (bits got))
                (nan_codes dt))
            narrows);
      test "f2f narrows a quiet NaN to the NaN code Dtype folds" (fun () ->
          List.iter
            (fun dt ->
              let u = narrowing dt in
              List.iter
                (fun b ->
                  equal
                    ~msg:(Printf.sprintf "%s of 0x%08x" (alias dt) b)
                    (code_of dt)
                    (encode dt (`Float (f32 b)))
                    (as_int (at_code u b)))
                (both_signs quiet_nans))
            narrows);
      test "f2f narrows a signalling NaN to a NaN of its sign" (fun () ->
          List.iter
            (fun dt ->
              let u = narrowing dt in
              List.iter
                (fun b ->
                  let c = as_int (at_code u b) in
                  is_true
                    ~msg:(Printf.sprintf "%s of 0x%08x is 0x%x" (alias dt) b c)
                    (is_nan_of dt ~negative:(b lsr 31 = 1) c))
                (both_signs signalling_nans))
            narrows);
      cases ~name:alias
        "an emulated copy keeps every NaN code, quieting a signalling one"
        narrows (fun dt -> copies (nan_codes dt) dt);
      cases ~name:alias
        "an emulated selection keeps NaN codes, quieting a signalling one"
        narrows (fun dt -> selects_codes (nan_codes dt) dt ());
      cases ~name:alias
        "an emulated cast of a float32 or a double NaN is the NaN code Dtype \
         folds"
        narrows (fun dt ->
          let values =
            List.map (fun b -> `Float (f32 b)) (both_signs quiet_nans)
          in
          converts ~from:Float32 ~to_:dt values (encode dt);
          converts ~from:Float64 ~to_:dt values (encode dt));
      cases ~name:alias
        "an emulated cast of a NaN code to a float32 is a NaN of its sign"
        narrows (fun dt ->
          let cs = nan_codes dt in
          let k = kernel [ dt ] Float32 (List.length cs) (cast_to Float32) in
          List.iter2
            (fun c got ->
              let got = as_float got in
              is_true
                ~msg:(Printf.sprintf "0x%x is %h" c got)
                (Float.is_nan got
                && Float.sign_bit got = Float.sign_bit (decode dt c)))
            cs
            (written ~on:on_narrows k [ List.map code cs ]));
    ]

(* 64-bit integers *)

let on_32_bits = lacking longs
let word_of = function Dtype.Int64 -> Dtype.Int32 | _ -> Dtype.Uint32

let z_of v =
  match (v :> Dtype.const) with
  | `Int z -> z
  | v -> failf "%a is not an integer" (Testable.pp const) v

(* [stored dt vs] is the storage of the values [vs] of [dt] on a target without
   64-bit integers: two 32-bit words for each 64-bit integer, low first. *)
let stored dt vs =
  if List.mem dt longs then
    List.concat_map
      (fun v ->
        List.map
          (fun k ->
            Dtype.truncate (word_of dt) (`Int (Z.extract (z_of v) (32 * k) 32)))
          [ 0; 1 ])
      vs
  else vs

let rec of_words dt = function
  | lo :: hi :: rest ->
      Dtype.truncate dt
        (`Int Z.(extract (z_of lo) 0 32 + shift_left (z_of hi) 32))
      :: of_words dt rest
  | [] -> []
  | [ _ ] -> failf "a 64-bit integer lacks its high word"

(* [computes ins out f] is the function that the emulated kernel of [f] from
   [ins] to [out] computes, on one element. *)
let computes ins out f =
  let k = lazy (emulate on_32_bits (kernel ins out 1 f)) in
  fun args ->
    let got =
      outputs (Lazy.force k) (List.map2 (fun dt v -> stored dt [ v ]) ins args)
    in
    match if List.mem out longs then of_words out got else got with
    | [ v ] -> v
    | vs -> failf "%d values written" (List.length vs)

let wrap dt z = Dtype.truncate dt (`Int z)

let nonzero dt =
  Gen.such_that (fun v -> not (Z.equal (z_of v) Z.zero)) (value_of dt)

let shift_count dt = Gen.map (fun n -> `Int (Z.of_int n)) (Gen.int_range 0 63)

(* [law name ins out f args oracle] is the property that the emulated [f]
   computes [oracle] on the arguments that [args] draws. *)
let law ?tags ?(count = 100) ?examples name ins out f args oracle =
  let run = computes ins out f in
  let pp =
    Format.pp_print_list ~pp_sep:Format.pp_print_space (Testable.pp value)
  in
  prop ?tags ~count ?examples name (Gen.with_pp pp args) (fun args ->
      equal value (oracle args) (run args))

let binary ?tags ?count dt ?(b = value_of dt) ?examples name op oracle =
  let examples =
    Option.map
      (List.map (fun (a, b) -> [ wrap dt a; `Int (Z.of_int b) ]))
      examples
  in
  law ?tags ?count ?examples
    (alias dt ^ " " ^ name)
    [ dt; dt ] dt
    (fun xs -> Ops.alu (List.nth xs 0) op [ List.nth xs 1 ])
    (Gen.map (fun (a, b) -> [ a; b ]) (Gen.pair (value_of dt) b))
    (function
      | [ a; b ] -> wrap dt (oracle (z_of a) (z_of b)) | _ -> assert false)

let compares dt name op oracle =
  law
    (alias dt ^ " " ^ name)
    [ dt; dt ] Bool
    (fun xs -> Ops.alu (List.nth xs 0) op [ List.nth xs 1 ])
    (Gen.map (fun (a, b) -> [ a; b ]) (Gen.pair (value_of dt) (value_of dt)))
    (function
      | [ a; b ] -> `Bool (oracle (Z.compare (z_of a) (z_of b)))
      | _ -> assert false)

(* [converts_from from dt] is the law that the emulated cast from [from] to [dt]
   wraps the value it casts. *)
let converts_from from dt =
  law
    (Format.asprintf "%s from %s" (alias dt) (alias from))
    [ from ] dt (cast_to dt)
    (Gen.map (fun v -> [ v ]) (value_of from))
    (function
      | [ `Bool b ] -> wrap dt (if b then Z.one else Z.zero)
      | [ v ] -> wrap dt (z_of v)
      | _ -> assert false)

(* The floats beyond a 32-bit word's range, which the low word of an emulated
   cast to a 64-bit integer must not convert to 32 bits directly. *)
let wide_floats dt =
  let xs = [ 0x1p31; 0x1p32 +. 512.; 0x1p40; 0x1p62 ] in
  List.map
    (fun x -> `Float x)
    (if dt = Dtype.Int64 then xs @ List.map Float.neg xs else xs)

(* [float32_in dt] draws float32s whose truncation [dt] holds. *)
let float32_in dt =
  let lo, hi = int_bounds dt in
  Gen.such_that
    (fun x -> Z.to_float lo <= x && x < Z.to_float hi)
    (Gen.map
       (fun x -> as_float (Dtype.truncate Float32 (`Float x)))
       finite_float)

(* tinygrad's shifts of 64-bit integers, as values and distances. *)
let shl_examples dt =
  let values =
    if dt = Dtype.Int64 then [ -0x1234; 0x80000001; -1; 0x1234; 1 ]
    else [ 0x80000001; 0x80000001; 1; 0xFEDC; 1 ]
  in
  List.combine (List.map Z.of_int values) [ 0; 5; 31; 32; 62 ]

let shr_examples dt =
  let values =
    if dt = Dtype.Int64 then
      List.map Z.of_int
        [ -(1 lsl 40); -1; -(1 lsl 50); -(1 lsl 40); 0x123456789ABCDEF ]
    else List.init 5 (fun _ -> Z.of_string "0xFEDCBA9876543210")
  in
  List.combine values [ 0; 5; 31; 32; 63 ]

(* 64-bit integers whose leading word rounds up to a power of two as a float32:
   a high word of [2^25 - 1], and a low word of [2^31 - 1] under a zero high
   word. *)
let leading_word_examples dt =
  let zs = Z.[ (of_int 0x1ffffff lsl 32) + of_int 12345; of_int 0x7fffffff ] in
  let zs = if dt = Dtype.Int64 then zs @ List.map Z.neg zs else zs in
  List.map (fun z -> [ `Int z ]) zs

let long_laws dt =
  let other = if dt = Dtype.Int64 then Dtype.Uint64 else Dtype.Int64 in
  [
    binary dt "add" Add Z.add;
    binary dt "sub" Sub Z.sub;
    binary dt "mul" Mul Z.mul;
    binary dt "and" And Z.logand;
    binary dt "or" Or Z.logor;
    binary dt "xor" Xor Z.logxor;
    binary dt "max" Max Z.max;
    binary ~count:20 dt ~b:(nonzero dt) "truncating division" Cdiv Z.div;
    binary ~count:20 dt ~b:(nonzero dt) "truncating remainder" Cmod Z.rem;
    binary ~tags:[ "slow" ] ~count:2000 dt ~b:(nonzero dt)
      "truncating division, over 2000 cases" Cdiv Z.div;
    binary ~tags:[ "slow" ] ~count:2000 dt ~b:(nonzero dt)
      "truncating remainder, over 2000 cases" Cmod Z.rem;
    binary dt ~b:(shift_count dt) ~examples:(shl_examples dt) "shl" Shl
      (fun a n -> Z.shift_left a (Z.to_int n));
    binary dt ~b:(shift_count dt) ~examples:(shr_examples dt) "shr" Shr
      (fun a n -> Z.shift_right a (Z.to_int n));
    compares dt "cmplt" Cmplt (fun c -> c < 0);
    compares dt "cmpeq" Cmpeq (fun c -> c = 0);
    compares dt "cmpne" Cmpne (fun c -> c <> 0);
    law
      (alias dt ^ " negation")
      [ dt ] dt
      (fun xs -> Ops.alu (List.hd xs) Neg [])
      (Gen.map (fun v -> [ v ]) (value_of dt))
      (function [ a ] -> wrap dt (Z.neg (z_of a)) | _ -> assert false);
    law
      (alias dt ^ " selection")
      [ Bool; dt; dt ] dt
      (fun xs -> Ops.where (List.nth xs 0) (List.nth xs 1) (List.nth xs 2))
      (Gen.map
         (fun (c, (a, b)) -> [ `Bool c; a; b ])
         (Gen.pair Gen.bool (Gen.pair (value_of dt) (value_of dt))))
      (function [ `Bool c; a; b ] -> if c then a else b | _ -> assert false);
    converts_from Int32 dt;
    converts_from Uint32 dt;
    converts_from Int8 dt;
    converts_from Bool dt;
    converts_from dt Int32;
    converts_from dt Int8;
    converts_from dt Uint16;
    converts_from dt other;
    law
      (alias dt ^ " bitcast to " ^ alias other)
      [ dt ] other
      (fun xs -> Ops.bitcast (List.hd xs) other)
      (Gen.map (fun v -> [ v ]) (value_of dt))
      (function [ v ] -> wrap other (z_of v) | _ -> assert false);
    law
      ~examples:(List.map (fun v -> [ v ]) (wide_floats dt))
      (alias dt ^ " from float32, rounded towards zero")
      [ Float32 ] dt (cast_to dt)
      (Gen.map (fun x -> [ `Float x ]) (float32_in dt))
      (function
        | [ `Float x ] -> wrap dt (Z.of_float (Float.trunc x))
        | _ -> assert false);
    law ~examples:(leading_word_examples dt)
      (alias dt ^ " to float32, the nearest")
      [ dt ] Float32 (cast_to Float32)
      (Gen.map (fun v -> [ v ]) (value_of dt))
      (function [ v ] -> Dtype.truncate Float32 v | _ -> assert false);
    law
      (alias dt ^ " to double, the nearest")
      [ dt ] Float64 (cast_to Float64)
      (Gen.map (fun v -> [ v ]) (value_of dt))
      (function [ v ] -> Dtype.truncate Float64 v | _ -> assert false);
  ]

let long_arithmetic =
  group "emulated 64-bit integers"
    (List.concat_map long_laws longs
    @ [
        cases ~name:alias
          "an emulated cast of a float32 to a 64-bit integer converts no float \
           to a word that cannot hold it"
          longs (fun dt ->
            let values = wide_floats dt in
            let k = kernel [ Float32 ] dt (List.length values) (cast_to dt) in
            equal (list value) []
              (List.map (List.nth values)
                 (undefined_casts ~on:on_32_bits k [ values ])));
      ])

(* D9: narrow float conversions are IEEE conversions *)

(* [casts ~from dt table] checks that the emulated cast of each value of
   [table], of type [from], to [dt] writes the code beside it. *)
let casts ?(from = Dtype.Float32) dt table =
  let values = List.map fst table in
  let k = kernel [ from ] dt (List.length values) (cast_to dt) in
  equal
    (list (pair value hex))
    table
    (List.combine values
       (List.map as_int (written ~on:on_narrows k [ values ])))

let integer s = `Int (Z.of_string s)

let least_subnormals =
  Dtype.
    [
      (Float16, 0x1p-24);
      (Bfloat16, 0x1p-133);
      (Fp8e4m3, 0x1p-9);
      (Fp8e5m2, 0x1p-16);
      (Fp8e4m3fnuz, 0x1p-10);
      (Fp8e5m2fnuz, 0x1p-17);
    ]

let d9 =
  group "D9"
    [
      cases
        ~name:(fun (dt, _) -> alias dt)
        "an emulated narrow float keeps its subnormals, both ways"
        least_subnormals
        (fun (dt, least) ->
          casts dt
            [
              (`Float least, 0x01);
              (`Float (least *. 0.75), 0x01);
              (`Float (least *. 0.5), 0x00);
              (`Float (least *. 1.5), 0x02);
              (`Float (-.least), (size dt / 2) + 0x01);
            ];
          let k = kernel [ dt ] Float32 1 (cast_to Float32) in
          equal ~msg:"widened" value (`Float least)
            (List.hd (written ~on:on_narrows k [ [ code 1 ] ])));
      test
        "a 16-bit float overflows from its greatest value plus half an ulp, \
         the tie to infinity" (fun () ->
          casts Float16
            [
              (`Float 65504., 0x7bff);
              (`Float (below 65520.), 0x7bff);
              (`Float 65520., 0x7c00);
              (`Float (-65520.), 0xfc00);
            ];
          casts Bfloat16
            [
              (`Float (below (0x1p128 -. 0x1p119)), 0x7f7f);
              (`Float (0x1p128 -. 0x1p119), 0x7f80);
              (`Float (-.(0x1p128 -. 0x1p119)), 0xff80);
            ]);
      cases
        ~name:(fun (dt, _) -> alias dt)
        "an infinity stays one in e5m2 and is the NaN of e4m3 and the fnuz \
         formats, and a finite overflow saturates"
        Dtype.
          [
            (Fp8e4m3, [ 0x7f; 0xff; 0x7e; 0xfe ]);
            (Fp8e5m2, [ 0x7c; 0xfc; 0x7b; 0xfb ]);
            (Fp8e4m3fnuz, [ 0x80; 0x80; 0x7f; 0xff ]);
            (Fp8e5m2fnuz, [ 0x80; 0x80; 0x7f; 0xff ]);
          ]
        (fun (dt, codes) ->
          casts dt
            (List.combine
               [
                 `Float Float.infinity;
                 `Float Float.neg_infinity;
                 `Float 1e10;
                 `Float (-1e10);
               ]
               codes));
      test "a double, or an integer more precise than a float32, rounds once"
        (fun () ->
          casts ~from:Float64 Bfloat16
            [ (`Float (1. +. 0x1p-8 +. 0x1p-40), 0x3f81) ];
          casts ~from:Float64 Float16 [ (`Float (65520. -. 0x1p-20), 0x7bff) ];
          casts ~from:Int32 Bfloat16
            [
              (integer "16842753", 0x4b81);
              (integer "-16842753", 0xcb81);
              (integer "2147483647", 0x4f00);
              (integer "-2147483648", 0xcf00);
            ];
          casts ~from:Uint32 Bfloat16 [ (integer "4294967295", 0x4f80) ];
          casts ~from:Int64 Bfloat16
            [
              (integer "1103806595073", 0x5381);
              (integer "9223372036854775807", 0x5f00);
              (integer "-9223372036854775808", 0xdf00);
            ];
          casts ~from:Uint64 Bfloat16
            [
              (integer "9259400833873739777", 0x5f01);
              (integer "18446744073709551615", 0x5f80);
            ];
          casts ~from:Uint32 Float16 [ (integer "4294967295", 0x7c00) ];
          casts ~from:Int32 Fp8e4m3 [ (integer "2147483647", 0x7e) ]);
      cases ~name:alias
        "an fnuz format stores an underflow to negative zero as positive zero"
        Dtype.fp8_fnuz (fun dt ->
          let least = List.assoc dt least_subnormals in
          casts dt
            [
              (`Float (-0.), 0x00);
              (`Float (-1e-30), 0x00);
              (`Float (-.least *. 0.5), 0x00);
            ]);
      test "an emulated 64-bit integer converts to a float32 once" (fun () ->
          let converts from to_ z =
            let k = kernel [ from ] to_ 1 (cast_to to_) in
            let v = integer z in
            outputs
              (emulate (lacking (longs @ narrows)) k)
              [ stored from [ v ] ]
          in
          equal (list value)
            [ `Float 4505803482464256. ]
            (converts Uint64 Float32 "4505803214028801");
          equal (list value)
            [ `Float (-9007200328482816.) ]
            (converts Int64 Float32 "-9007199791611905");
          equal (list hex) [ 0xdebb ]
            (List.map as_int (converts Int64 Bfloat16 "-6719370644036780033")));
      cases
        ~name:(fun (dt, _) -> alias dt)
        "an emulated narrow-float copy quiets a signalling NaN; a native copy \
         keeps its bits"
        Dtype.
          [
            (Float16, [ (0x7c01, 0x7e01); (0xfd55, 0xff55) ]);
            (Bfloat16, [ (0x7f81, 0x7fc1); (0xffbf, 0xffff) ]);
            (Fp8e5m2, [ (0x7d, 0x7f); (0xfd, 0xff) ]);
          ]
        (fun (dt, table) ->
          let k = kernel [ dt ] dt (List.length table) List.hd in
          let codes = List.map fst table in
          equal
            (list (pair hex hex))
            table
            (List.combine codes
               (List.map as_int
                  (written ~on:on_narrows k [ List.map code codes ]))));
      test "an OCP 8-bit float keeps its negative zero" (fun () ->
          casts Fp8e4m3 [ (`Float (-0.), 0x80); (`Float (-1e-30), 0x80) ];
          casts Fp8e5m2 [ (`Float (-0.), 0x80); (`Float (-1e-30), 0x80) ]);
    ]

(* Goldens *)

(* A golden holds a kernel and what tinygrad's pass makes of it, with each call
   of f2f and f2f_clamp held as a placeholder: a custom node over its operand,
   whose code names the call and its arguments
   (gen/codegen/decomp/decomp_dtype.py). The conversions are D9's. *)

let dtype_named s = Result.get_ok (Dtype.of_string s)

let sat_named = function
  | "True" -> true
  | "False" -> false
  | s -> failf "%S is not a boolean" s

let conversion u =
  match (Ops.arg u, Ops.src u) with
  | Code { code; _ }, [ v ] -> (
      match String.split_on_char ' ' code with
      | [ "f2f"; fr; to_; sat ] ->
          Some
            (Decomp_dtype.f2f ~sat:(sat_named sat) v (dtype_named fr)
               (dtype_named to_))
      | [ "f2f_clamp"; dt; sat ] ->
          Some (Decomp_dtype.f2f_clamp ~sat:(sat_named sat) v (dtype_named dt))
      | _ -> None)
  | _ -> None

(* [with_conversions u] is [u] with its placeholders replaced by the conversions
   they stand for, the weak constants of the conversions committed as the pass
   commits them. *)
let with_conversions u =
  let restore =
    PM.v
      [ PM.rule (Ops.Upat.op Custom ~name:"c") (fun m -> conversion (m "c")) ]
  in
  Ops.graph_rewrite ~ctx:()
    (Ops.graph_rewrite ~ctx:() u restore)
    Uop_weak.pm_commit_weak

let narrow_kernels =
  [
    "add";
    "mulsub";
    "maximum";
    "where";
    "exp2";
    "sqrt";
    "sum";
    "lt";
    "from_float";
    "to_float";
    "from_char";
    "to_char";
    "flip";
    "gather";
    "pad";
    "bitcast_to_storage";
    "bitcast_from_storage";
    "bitcast_sum";
  ]

let long_kernels dt =
  [
    "add";
    "sub";
    "neg";
    "mul";
    (if dt = Dtype.Int64 then "div" else "mod");
    "shl";
    "shr";
    "shl_by";
    "xor";
    "and_or";
    "lt";
    "eq";
    "maximum";
    "where";
    "sum";
    "pad";
    "add_const";
    "from_int";
    "from_uint";
    "from_bool";
    "to_char";
    "bitcast";
  ]

(* Each golden, with the data types the setting names and those the other target
   lacks. *)
let goldens =
  List.concat_map
    (fun dt ->
      List.map (fun k -> (alias dt ^ "_" ^ k, [ dt ], [ dt ])) narrow_kernels)
    narrows
  @ Dtype.
      [
        ("half_fp8e4m3_cast", [ Float16; Fp8e4m3 ], [ Float16; Fp8e4m3 ]);
        ("bfloat16_half_cast", [ Float16; Bfloat16 ], [ Float16; Bfloat16 ]);
      ]
  @ List.concat_map
      (fun dt ->
        List.map
          (fun k -> (alias dt ^ "_" ^ k, [ Dtype.Int64 ], longs))
          (long_kernels dt))
      longs
  @ [ ("ulong_named_alone", [ Dtype.Uint64 ], [ Dtype.Uint64 ]) ]

(* [golden name] is the sink of the golden [name], read once. *)
let golden =
  let read = Hashtbl.create 160 in
  fun name ->
    match Hashtbl.find_opt read name with
    | Some g -> g
    | None ->
        let g = Golden.sink (name ^ ".golden") in
        Hashtbl.add read name g;
        g

(* [same_graph want got] checks that [got] is [want], which hash-consing makes
   one node, and fails with the diff of their graphs. *)
let same_graph want got =
  if not (Ops.equal want got) then
    equal text (Graph.to_string want) (Graph.to_string got)

(* [emulates_as_tinygrad (name, named, lacks)] checks the golden [name] on the
   target that lacks [lacks]; [told_as_lacking] checks that the target told to
   emulate [named] rewrites it the same way. *)
let emulates_as_tinygrad (name, _, lacks) =
  test (name ^ ".golden") (fun () ->
      let g = golden name in
      same_graph
        (with_conversions (Ops.nth g 1))
        (emulate (lacking lacks) (Ops.nth g 0)))

let told_as_lacking ?tags (name, named, lacks) =
  test ?tags (name ^ ".golden, told to emulate") (fun () ->
      let k = Ops.nth (golden name) 0 in
      same_graph
        (emulate (lacking lacks) k)
        (told (List.map alias named) (fun () -> emulate everything k)))

(* [by_default kernel (name, _, _)] is [true] iff [name] is the golden of
   [kernel] of one type: the goldens a check runs on by default, the others
   running under the slow tag. *)
let by_default kernel (name, _, _) =
  List.exists (fun dt -> name = alias dt ^ "_" ^ kernel) (narrows @ longs)

(* The kernels of the goldens compute, emulated, what they compute natively.
   Their buffers are filled from a fixed seed, each within what its kernel
   reads: an index within the gathered storage, a shift count below 64, a
   nonzero divisor, and a float that a 64-bit integer holds. The kernels that
   compute in a narrow float are left out: emulated, they compute in float32 and
   round once, at the store. *)

let computes_natively =
  [
    "flip";
    "gather";
    "pad";
    "where";
    "maximum";
    "lt";
    "sum";
    "from_float";
    "to_float";
    "from_char";
    "to_char";
    "bitcast_to_storage";
    "bitcast_from_storage";
  ]

let params k =
  List.filter_map
    (fun u ->
      match (Ops.op u, Ops.arg u) with Param, Param a -> Some a | _ -> None)
    (Ops.toposort k)
  |> List.sort_uniq (fun (a : Ops.param_arg) b -> Int.compare a.slot b.slot)

let random_z rng bits =
  let word () = Z.of_int (Random.State.bits rng land 0xffff) in
  Z.extract
    Z.(word () + (word () lsl 16) + (word () lsl 32) + (word () lsl 48))
    0 bits

let contains affix s =
  let n = String.length affix in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = affix || at (i + 1))
  in
  at 0

let finite_codes =
  lazy
    (List.map
       (fun dt ->
         let finite c = Float.is_finite (decode dt c) in
         (dt, Array.of_list (List.filter finite (codes ~all:true dt))))
       narrows)

(* [draw rng name ~last dt] draws an element of a storage of type [dt] of the
   golden [name], the last of its kernel's storage if [last]: the value its
   native kernel reads, and the values its emulated kernel reads in its place. A
   kernel's output is its first storage, and a shift count or a divisor its
   last. *)
let draw rng name ~last dt =
  let same v = (v, [ v ]) in
  match dt with
  | _ when List.mem dt narrows ->
      let finite = List.assoc dt (Lazy.force finite_codes) in
      let c = finite.(Random.State.int rng (Array.length finite)) in
      (`Float (decode dt c), [ code c ])
  | Dtype.Int64 | Uint64 ->
      let z =
        if contains "shl_by" name && last then
          Z.of_int (Random.State.int rng 64)
        else if (contains "div" name || contains "mod" name) && last then
          Z.succ (random_z rng 31)
        else random_z rng 64
      in
      let v = Dtype.truncate dt (`Int z) in
      (v, stored dt [ v ])
  | Float32 when contains "long" name ->
      let x = Random.State.float rng 0x1p62 in
      let x = if contains "ulong" name then x else x -. 0x1p61 in
      same (Dtype.truncate Float32 (`Float x))
  | Float32 ->
      let x = f32 (Z.to_int (random_z rng 31) mod 0x7f800000) in
      same (`Float (if Random.State.bool rng then x else -.x))
  | Int32 when contains "gather" name ->
      same (`Int (Z.of_int (Random.State.int rng 16)))
  | Bool -> same (`Bool (Random.State.bool rng))
  | dt -> same (Dtype.truncate dt (`Int (random_z rng (Dtype.bitsize dt))))

let emulated_writes_natively ?tags (name, named, lacks) =
  test ?tags (name ^ ".golden computes what it computes natively") (fun () ->
      let k = Ops.nth (golden name) 0 in
      let rng = Random.State.make [| Hashtbl.hash name |] in
      let slots = params k in
      let buffers =
        List.map
          (fun (a : Ops.param_arg) ->
            let n = Option.value ~default:1 a.size in
            let last = a.slot = List.length slots - 1 in
            (a.slot, List.init n (fun _ -> draw rng name ~last a.dtype)))
          slots
      in
      let native =
        Interpreter.writes
          ~buffers:
            (List.map
               (fun (s, xs) -> (s, Array.of_list (List.map fst xs)))
               buffers)
          k
      in
      let emulated =
        Interpreter.writes
          ~buffers:
            (List.map
               (fun (s, xs) -> (s, Array.of_list (List.concat_map snd xs)))
               buffers)
          (told (List.map alias named) (fun () -> emulate (lacking lacks) k))
      in
      let dtype_of slot =
        (List.find (fun (a : Ops.param_arg) -> a.slot = slot) slots).dtype
      in
      let rec decoded = function
        | (s, i, lo) :: (_, _, hi) :: rest when List.mem (dtype_of s) longs ->
            (s, i / 2, List.hd (of_words (dtype_of s) [ lo; hi ]))
            :: decoded rest
        | (s, i, `Int c) :: rest when List.mem (dtype_of s) narrows ->
            (s, i, `Float (decode (dtype_of s) (Z.to_int c))) :: decoded rest
        | w :: rest -> w :: decoded rest
        | [] -> []
      in
      equal (list (triple int int value)) native (decoded emulated))

let natively_comparable =
  List.concat_map
    (fun dt -> List.map (fun k -> alias dt ^ "_" ^ k) computes_natively)
    narrows
  @ List.concat_map
      (fun dt -> List.map (fun k -> alias dt ^ "_" ^ k) (long_kernels dt))
      longs

let graphs =
  group "goldens"
    (List.map emulates_as_tinygrad goldens
    @ List.map
        (fun g ->
          if by_default "add" g then told_as_lacking g
          else told_as_lacking ~tags:[ "slow" ] g)
        goldens
    @ List.filter_map
        (fun ((name, _, _) as g) ->
          if not (List.mem name natively_comparable) then None
          else if by_default "flip" g || by_default "add" g then
            Some (emulated_writes_natively g)
          else Some (emulated_writes_natively ~tags:[ "slow" ] g))
        goldens
    @ [
        test "every golden is checked" (fun () ->
            let files =
              Sys.readdir (Filename.dirname Sys.executable_name)
              |> Array.to_list
              |> List.filter (fun f -> Filename.check_suffix f ".golden")
            in
            equal
              (slist string String.compare)
              (List.map (fun (n, _, _) -> n ^ ".golden") goldens)
              files);
      ])

(* The pass *)

let decomps on k =
  Ops.graph_rewrite ~ctx:(Decomp_dtype.ctx on) k Decomp_dtype.pm_dtype_decomps

let add dt =
  kernel [ dt; dt ] dt 4 (fun xs -> Ops.O.(List.nth xs 0 + List.nth xs 1))

let pass =
  group "pm_dtype_decomps"
    [
      test "a target with every data type keeps its kernels" (fun () ->
          List.iter
            (fun dt ->
              equal ~msg:(alias dt) Uops.uop (add dt)
                (decomps everything (add dt)))
            (narrows @ longs));
      test "a kernel without narrow floats or 64-bit integers is kept"
        (fun () ->
          List.iter
            (fun dt ->
              equal ~msg:(alias dt) Uops.uop (add dt)
                (decomps (lacking (narrows @ longs)) (add dt)))
            Dtype.[ Float32; Float64; Int32; Uint32; Int8; Bool ]);
      test "unsigned 64-bit integers are emulated when the signed ones are"
        (fun () ->
          equal Uops.uop
            (decomps (lacking longs) (add Uint64))
            (decomps (lacking [ Int64 ]) (add Uint64)));
      test "64-bit storage holds two 32-bit words per element" (fun () ->
          List.iter
            (fun dt ->
              let k = decomps on_32_bits (kernel [ dt ] dt 7 List.hd) in
              equal ~msg:(alias dt)
                (list (pair dtype (option int)))
                [ (word_of dt, Some 14); (word_of dt, Some 14) ]
                (List.map
                   (fun (a : Ops.param_arg) -> (a.dtype, a.size))
                   (params k)))
            longs);
      test "64-bit storage of no known size keeps none" (fun () ->
          let p = Ops.param ~addrspace:(Some Global) 0 Int64 in
          let k =
            Ops.sink
              [ Ops.store (Ops.index p [ Ops.int 0 ]) (Ops.int ~dtype:Int64 3) ]
          in
          equal
            (list (pair dtype (option int)))
            [ (Int32, None) ]
            (List.map
               (fun (a : Ops.param_arg) -> (a.dtype, a.size))
               (params (decomps on_32_bits k))));
      test "a 64-bit integer variable cannot be emulated" (fun () ->
          let c =
            Ops.variable ~dtype:Int64 "c" (`Int Z.zero)
              (`Int (Z.shift_left Z.one 40))
          in
          let k = kernel [] Int64 1 (fun _ -> c) in
          rejects (fun () -> decomps on_32_bits k));
      test "a setting that names no data type is refused" (fun () ->
          told [ "nonsense" ] (fun () ->
              rejects (fun () -> decomps everything (add Float16))));
    ]

let () =
  exit
    (run "Tolk_next.Decomp_dtype"
       [
         f2f;
         f2f_clamp;
         emulated_floats;
         nans;
         d9;
         long_arithmetic;
         graphs;
         pass;
       ])
