(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Folds over cut axes, through Nx: a reduction, a contraction or a scan over
   an axis cut across devices gives an exact one's bits on one device, and a
   float one the fold of its tiles' stacked partials, within partial.mli's
   bounds; each moves its partials, not its operands. An operation over a
   value at any placement, folds and scans along cut axes among them, gives
   its bits on one device. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module C = Nx_support.Counting

let m = Nx_support.memory

module Count = (val Nx.devices ~kernels:(module C) [ m 0 ])
module Count2 = (val Nx.devices ~kernels:(module C) [ m 0; m 1 ])

let elements x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))

(* The bytes of [x]'s elements, read on the host. *)
let bytes (type v s d) (x : (v, s, d) Nx.t) =
  let a = Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)) in
  let flat =
    A.move (A.Move.Reshape [| A.Layout.numel (A.layout a) |]) (A.copy a)
  in
  A.to_array (Option.get (A.bitcast D.Uint8 (Option.get flat)))

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Placements *)

module Set3 = (val Nx.devices [ m 0; m 1; m 2 ])
module Set4 = (val Nx.devices [ m 0; m 1; m 2; m 3 ])

let mesh4 = Nx.Mesh.v Set4.v [ ("a", 2); ("b", 2) ]

(* Where a value lies: split over three or four devices; a 2 x 2 mesh cutting
   one axis over both its axes or two axes over one each; or one axis over its
   minor axis, the major holding copies. *)
type where = Split3 of int | Split4 of int | Cuts of int * int | Minor of int

let pp_where ppf = function
  | Split3 a -> Format.fprintf ppf "split %d over 3" a
  | Split4 a -> Format.fprintf ppf "split %d over 4" a
  | Cuts (x, y) when x = y -> Format.fprintf ppf "cut %d over 2x2" x
  | Cuts (x, y) -> Format.fprintf ppf "cuts %d/a %d/b" x y
  | Minor x -> Format.fprintf ppf "cut %d/b" x

(* The axes [w] cuts, with their tiles. *)
let tiles = function
  | Split3 a -> [ (a, 3) ]
  | Split4 a -> [ (a, 4) ]
  | Cuts (x, y) when x = y -> [ (x, 4) ]
  | Cuts (x, y) -> [ (x, 2); (y, 2) ]
  | Minor x -> [ (x, 2) ]

let where_of rank =
  let axis = Gen.int_range 0 (rank - 1) in
  Gen.one_of
    [
      Gen.map (fun a -> Split3 a) axis;
      Gen.map (fun a -> Split4 a) axis;
      Gen.map (fun (x, y) -> Cuts (x, y)) (Gen.pair axis axis);
      Gen.map (fun x -> Minor x) axis;
    ]

(* A dtype, the elements drawn for it, how its elements' bits print, and a
   map and a fold it takes: a float's or a complex number's the maximum,
   whose bits no cut changes. *)
type ('v, 's) kind = {
  dtype : ('v, 's) D.t;
  values : 'v list;
  show : 'v -> string;
  combine : 'd. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t;
  fold : 'd. int -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t;
}

type any_kind = Kind : ('v, 's) kind -> any_kind

let float_bits v = Printf.sprintf "%Lx" (Int64.bits_of_float v)

(* [v] of [dt] as [show] prints it, every NaN alike: a NaN result promises a
   NaN, never which one. Under [~parts] a complex number's parts are each
   so, as a sum adds them apart; else a complex number with a NaN part is a
   NaN. *)
let nan_blind (type v s) ~parts (dt : (v, s) D.t) (show : v -> string) (v : v)
    =
  match D.kind dt with
  | D.Float -> if Float.is_nan v then "nan" else show v
  | D.Complex ->
      let part x = if Float.is_nan x then Float.nan else x in
      if not (Float.is_nan v.Complex.re || Float.is_nan v.im) then show v
      else if parts then show { Complex.re = part v.re; im = part v.im }
      else "nan"
  | D.Signed | D.Unsigned | D.Boolean -> show v

(* Floats whose sums depend on their association, signed zeros, infinities and
   NaNs of both signs and two payloads. *)
let floats =
  [
    1e7; -1e7; 1.; 0.1; 3.3; 0.; -0.; infinity; neg_infinity; nan; -.nan;
    Int32.float_of_bits 0x7fc00123l; Int32.float_of_bits 0xffa00001l;
  ]

(* The maximum along [a], or over no element the sum, [+0]: both keep their
   bits on one device wherever [x] lies. *)
let extreme a x =
  if (Nx.shape x).(a) = 0 then Nx.sum ~axes:[ a ] x else Nx.max ~axes:[ a ] x

let float_kind dtype =
  Kind
    {
      dtype;
      values = floats;
      show = float_bits;
      combine = Nx.add;
      fold = extreme;
    }

let kinds =
  [
    float_kind D.Float32;
    float_kind D.Float16;
    float_kind D.Bfloat16;
    Kind
      {
        dtype = D.Complex64;
        values =
          List.map2
            (fun re im -> { Complex.re; im })
            floats (List.rev floats);
        show = (fun c -> float_bits c.Complex.re ^ "," ^ float_bits c.im);
        combine = Nx.add;
        fold = extreme;
      };
    Kind
      {
        dtype = D.Int32;
        values = [ 0l; 1l; -1l; 7l; Int32.max_int; Int32.min_int ];
        show = Int32.to_string;
        combine = Nx.add;
        fold = (fun a x -> Nx.sum ~axes:[ a ] x);
      };
    Kind
      {
        dtype = D.Int8;
        values = [ 0; 1; -1; 5; 127; -128 ];
        show = string_of_int;
        combine = Nx.add;
        fold = (fun a x -> Nx.sum ~axes:[ a ] x);
      };
    Kind
      {
        dtype = D.Bool;
        values = [ true; false ];
        show = string_of_bool;
        combine = Nx.logical_xor;
        fold = (fun a x -> Nx.any ~axes:[ a ] x);
      };
  ]

(* A movement that keeps every axis of a value in order. *)
type movement = Unit_axis | Reversed | Leading | Sliced

(* An operation over [x]: [combine x x]; a fold along an axis; a movement,
   then the moved value combined with itself; and [x] combined with a ramp of
   its shape where it lies: an arange's cumulative sum along an axis, cast. *)
type op = Combine | Fold of int | Moved of movement | Beside of int

let pp_op ppf = function
  | Combine -> Format.pp_print_string ppf "combined"
  | Fold a -> Format.fprintf ppf "folded along %d" a
  | Moved Unit_axis -> Format.pp_print_string ppf "given a unit axis"
  | Moved Reversed -> Format.pp_print_string ppf "transposed"
  | Moved Leading -> Format.pp_print_string ppf "broadcast along a new axis"
  | Moved Sliced -> Format.pp_print_string ppf "sliced along its last axis"
  | Beside a -> Format.fprintf ppf "beside a ramp scanned along %d" a

let apply (type v s d) (k : (v, s) kind) op (x : (v, s, d) Nx.t) :
    (v, s, d) Nx.t =
  let s = Nx.shape x in
  let r = Array.length s in
  match op with
  | Combine -> k.combine x x
  | Fold a -> k.fold a x
  | Moved mv ->
      let y =
        match mv with
        | Unit_axis -> Nx.reshape (Array.append s [| 1 |]) x
        | Reversed -> Nx.transpose x
        | Leading -> Nx.broadcast_to (Array.append [| 2 |] s) x
        | Sliced ->
            let last i = if i = r - 1 then Nx.R (1, s.(i)) else Nx.A in
            Nx.slice (List.init r last) x
      in
      k.combine y y
  | Beside a ->
      (* The ramp made whole on each of [x]'s devices, then cut as [x] is. *)
      let n = Array.fold_left ( * ) 1 s in
      let p = Nx.placement x in
      let ramp = Nx.arange_at (Nx.beside x) D.Int32 0 n 1 in
      let c = Nx.cumsum ~axis:a (Nx.place p (Nx.reshape s ramp)) in
      k.combine x (Nx.cast k.dtype c)

type case =
  | Case : ('v, 's) kind * int array * 'v array * where * op -> case

let pp_case ppf (Case (k, s, data, w, op)) =
  Format.fprintf ppf "%s %a [%s] at %a, %a" (D.name k.dtype) pp_ints s
    (String.concat "; " (Array.to_list (Array.map k.show data)))
    pp_where w pp_op op

let placed_case =
  let open Gen in
  with_pp pp_case
    (let* rank = int_range 1 3 in
     let* s = array ~size:(constant rank) (of_list [ 0; 1; 2; 3; 5; 7 ]) in
     let* w = where_of rank in
     let axis = int_range 0 (rank - 1) in
     let* op =
       one_of
         [
           constant Combine;
           map (fun a -> Fold a) axis;
           map (fun mv -> Moved mv)
             (of_list [ Unit_axis; Reversed; Leading; Sliced ]);
           map (fun a -> Beside a) axis;
         ]
     in
     let* (Kind k) = of_list kinds in
     let n = Array.fold_left ( * ) 1 s in
     let+ data = array ~size:(constant n) (of_list k.values) in
     Case (k, s, data, w, op))

(* [y]'s elements' bits, read on the host, every NaN alike. *)
let shown (type v s d) ~parts (k : (v, s) kind) (y : (v, s, d) Nx.t) =
  Array.map (nan_blind ~parts k.dtype k.show) (elements y)

let law_placed (Case (k, s, data, w, op)) =
  let ts = tiles w in
  let cut = List.map fst ts in
  let extent a = s.(a) in
  cover "an uneven cut" (List.exists (fun (a, t) -> extent a mod t <> 0) ts);
  cover "an empty window"
    (List.exists (fun (a, t) -> extent a > 0 && extent a < t) ts);
  cover "a cut of an empty axis" (List.exists (fun a -> extent a = 0) cut);
  cover "a cut over several mesh axes"
    (match w with Cuts (x, y) -> x = y | _ -> false);
  (match op with
  | Fold a -> cover "a fold along an axis no cut names" (not (List.mem a cut))
  | Moved Sliced ->
      cover "a slice that keeps a cut axis"
        (not (List.mem (Array.length s - 1) cut))
  | Moved _ -> cover "a movement of a cut value" true
  | Beside a ->
      cover "a ramp scanned along a cut axis" (List.mem a cut);
      cover "a ramp scanned along a cut axis, beside another cut"
        (List.mem a cut && List.length cut > 1)
  | Combine -> ());
  cover "a signalling NaN" (signals k.dtype data);
  let x = signalling k.dtype s data in
  (* A maximum's NaN is one of the elements with a NaN part, which one
     unspecified; other operations keep a complex number's parts apart. *)
  let parts = match op with Fold _ -> false | _ -> true in
  let expected = shown ~parts k (apply k op x) in
  (* A slice that drops the first element of a cut axis reads it whole,
     which raises where the value has elements; read at [q], that cut
     undone, as the refusal writes, it computes. *)
  let last = Array.length s - 1 in
  let refused =
    op = Moved Sliced && List.mem last cut && not (Array.mem 0 s)
  in
  cover "a slice that reads a cut axis whole" refused;
  let run p q =
    let y = Nx.place p x in
    if not refused then shown ~parts k (apply k op y)
    else begin
      let prefix =
        Format.asprintf
          "Nx.slice: operand 0 lies at %a and is read at %a; Nx.place ("
          Nx.Placement.pp p Nx.Placement.pp q
      in
      raises_match (Exn.invalid_arg ~substring:prefix) (fun () -> apply k op y);
      shown ~parts k (apply k op (Nx.place q y))
    end
  in
  let actual =
    match w with
    | Split3 a -> run (Set3.split ~axis:a) Set3.on
    | Split4 a -> run (Set4.split ~axis:a) Set4.on
    | Cuts (a, b) ->
        let cuts =
          if a = b then [ (a, [ "a"; "b" ]) ]
          else [ (a, [ "a" ]); (b, [ "b" ]) ]
        in
        let kept = List.filter (fun (c, _) -> c <> last) cuts in
        let q = if kept = [] then Set4.on else Nx.Placement.mesh mesh4 kept in
        run (Nx.Placement.mesh mesh4 cuts) q
    | Minor a -> run (Nx.Placement.mesh mesh4 [ (a, [ "b" ]) ]) Set4.on
  in
  equal (array string) expected actual

(* Assemblies of cut values *)

(* A concatenation with a value of the given extent along an axis, a stack
   of two values along a new axis, or a pad by widths per axis. *)
type assembly = Concat of int * int | Stack of int | Pad of (int * int) array

let pp_assembly ppf = function
  | Concat (a, e) ->
      Format.fprintf ppf "concatenated along %d with %d more" a e
  | Stack a -> Format.fprintf ppf "stacked along %d" a
  | Pad ws ->
      Format.fprintf ppf "padded by [%s]"
        (String.concat "; "
           (Array.to_list
              (Array.map (fun (b, e) -> Printf.sprintf "(%d, %d)" b e) ws)))

(* The second operand's shape: [s], or [s] of [e] elements along [a]. *)
let second_shape s = function
  | Concat (a, e) ->
      let s = Array.copy s in
      s.(a) <- e;
      s
  | Stack _ | Pad _ -> s

(* A value of shape [s] at [w], the assembly, its second operand's elements
   and whether that operand lies whole on each device of the first's set. *)
type acase =
  | Acase :
      ('v, 's) kind * int array * 'v array * where * assembly * 'v array * bool
      -> acase

let pp_acase ppf (Acase (k, s, data, w, asm, data', beside)) =
  let show d = String.concat "; " (Array.to_list (Array.map k.show d)) in
  Format.fprintf ppf "%s %a [%s] at %a, %a [%s]%s" (D.name k.dtype) pp_ints s
    (show data) pp_where w pp_assembly asm (show data')
    (if beside then " whole on each device" else "")

let assembled_case =
  let open Gen in
  with_pp pp_acase
    (let* rank = int_range 1 3 in
     let* s = array ~size:(constant rank) (of_list [ 0; 1; 2; 3; 5; 7 ]) in
     let* w = where_of rank in
     let cut = List.map fst (tiles w) in
     let free =
       List.filter (fun a -> not (List.mem a cut)) (List.init rank Fun.id)
     in
     let width = pair (int_range 0 2) (int_range 0 2) in
     let pad ws =
       Pad (Array.mapi (fun a w -> if List.mem a cut then (0, 0) else w) ws)
     in
     let concat =
       if free = [] then []
       else
         [
           map
             (fun (a, e) -> Concat (a, e))
             (pair (of_list free) (of_list [ 0; 1; 3 ]));
         ]
     in
     let* asm =
       one_of
         ([
            map (fun a -> Stack a) (int_range 0 rank);
            map pad (array ~size:(constant rank) width);
          ]
         @ concat)
     in
     let* beside = bool in
     let* (Kind k) = of_list kinds in
     let n = Array.fold_left ( * ) 1 s in
     let n' = Array.fold_left ( * ) 1 (second_shape s asm) in
     let* data = array ~size:(constant n) (of_list k.values) in
     let+ data' = array ~size:(constant n') (of_list k.values) in
     Acase (k, s, data, w, asm, data', beside))

(* An assembly that leaves a cut axis whole gives its result cut along that
   axis over the same devices, each device assembling its window. *)
let law_assembled (Acase (k, s, data, w, asm, data', beside)) =
  let ts = tiles w in
  let cut = List.map fst ts in
  cover "an uneven cut" (List.exists (fun (a, t) -> s.(a) mod t <> 0) ts);
  cover "an empty window"
    (List.exists (fun (a, t) -> s.(a) > 0 && s.(a) < t) ts);
  cover "a cut of an empty axis" (List.exists (fun a -> s.(a) = 0) cut);
  cover "a cut of a one-element axis" (List.exists (fun a -> s.(a) = 1) cut);
  cover "a cut over several mesh axes"
    (match w with Cuts (x, y) -> x = y | _ -> false);
  (match asm with
  | Concat (_, e) ->
      cover "a concatenation beside a cut" true;
      cover "a concatenation of nothing more" (e = 0);
      cover "a concatenation with a value whole on each device" beside
  | Stack a ->
      cover "a stack that moves a cut axis" (List.exists (fun c -> c >= a) cut);
      cover "a stack after every cut axis" (List.for_all (fun c -> c < a) cut);
      cover "a stack at or before a cut one-element axis"
        (List.exists (fun c -> c >= a && s.(c) = 1) cut);
      cover "a stack with a value whole on each device" beside
  | Pad ws ->
      cover "a pad beside a cut" (Array.exists (fun (b, e) -> b + e > 0) ws));
  let fill = List.hd k.values in
  let assemble (type d) (x : (_, _, d) Nx.t) (y : (_, _, d) Nx.t) =
    match asm with
    | Concat (a, _) -> Nx.concatenate ~axis:a [ x; y ]
    | Stack a -> Nx.stack ~axis:a [ x; y ]
    | Pad ws -> Nx.pad ws fill x
  in
  let x = signalling k.dtype s data in
  let y = signalling k.dtype (second_shape s asm) data' in
  let expected = bytes (assemble x y) in
  (* The result's axis that axis [c] of the operands becomes. *)
  let lift c = match asm with Stack a when c >= a -> c + 1 | _ -> c in
  let at (type d) (p : d Nx.Placement.t) (q : d Nx.Placement.t) =
    let placed = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal in
    let y = Nx.place (if beside then Nx.Placement.uncut p else p) y in
    let r = assemble (Nx.place p x) y in
    equal placed q (Nx.placement r);
    bytes r
  in
  let mesh cuts = Nx.Placement.mesh mesh4 cuts in
  let cuts a b =
    if a = b then mesh [ (a, [ "a"; "b" ]) ]
    else mesh [ (a, [ "a" ]); (b, [ "b" ]) ]
  in
  let actual =
    match w with
    | Split3 a -> at (Set3.split ~axis:a) (Set3.split ~axis:(lift a))
    | Split4 a -> at (Set4.split ~axis:a) (Set4.split ~axis:(lift a))
    | Cuts (a, b) -> at (cuts a b) (cuts (lift a) (lift b))
    | Minor a -> at (mesh [ (a, [ "b" ]) ]) (mesh [ (lift a, [ "b" ]) ])
  in
  equal (array int) expected actual

(* A concatenation reads the axis it joins along whole. *)
let test_concat_cut () =
  let x = Nx.place (Set4.split ~axis:0) (Nx.zeros D.Float32 [| 4; 2 |]) in
  raises
    (Invalid_argument
       "Nx.concatenate: operand 0 lies at split ~axis:0 [m0; m1; m2; m3] and \
        is read at on [m0; m1; m2; m3]; Nx.place (Placement.on s) x")
    (fun () -> Nx.concatenate ~axis:0 [ x; x ])

(* Reductions over cut axes *)

(* A dtype, the elements drawn for it, how their bits print, and which are
   NaNs. *)
type ('v, 's) reduced = {
  rdtype : ('v, 's) D.t;
  rvalues : 'v list;
  rshow : 'v -> string;
  is_nan : 'v -> bool;
}

type any_reduced = Reduced : ('v, 's) reduced -> any_reduced

let complexes =
  List.map2 (fun re im -> { Complex.re; im }) floats (List.rev floats)

(* A float's bits, any NaN shown alike: which NaN a result is is
   unspecified. A complex number with a NaN part is a NaN as a whole. *)
let nan_bits v = if Float.is_nan v then "nan" else float_bits v

let complex_bits c =
  if Float.is_nan c.Complex.re || Float.is_nan c.im then "nan"
  else float_bits c.re ^ "," ^ float_bits c.im

let reduced_kinds =
  let never _ = false in
  let float dt =
    Reduced
      {
        rdtype = dt;
        rvalues = floats;
        rshow = nan_bits;
        is_nan = Float.is_nan;
      }
  in
  let small dt values =
    Reduced
      {
        rdtype = dt;
        rvalues = values;
        rshow = string_of_int;
        is_nan = never;
      }
  in
  let truth dt =
    Reduced
      {
        rdtype = dt;
        rvalues = [ true; false ];
        rshow = string_of_bool;
        is_nan = never;
      }
  in
  [
    float D.Float32; float D.Float64; float D.Float16; float D.Bfloat16;
    Reduced
      {
        rdtype = D.Complex64;
        rvalues = complexes;
        rshow = complex_bits;
        is_nan = (fun c -> Float.is_nan c.Complex.re || Float.is_nan c.im);
      };
    Reduced
      {
        rdtype = D.Complex128;
        rvalues = complexes;
        rshow = complex_bits;
        is_nan = (fun c -> Float.is_nan c.Complex.re || Float.is_nan c.im);
      };
    Reduced
      {
        rdtype = D.Int32;
        rvalues = [ 0l; 1l; -1l; 7l; Int32.max_int; Int32.min_int ];
        rshow = Int32.to_string;
        is_nan = never;
      };
    small D.Int8 [ 0; 1; -1; 5; 127; -128 ];
    small D.Uint8 [ 0; 1; 3; 200; 255 ];
    small D.Int4 [ 0; 1; -1; 3; 7; -8 ];
    small D.Uint4 [ 0; 1; 5; 15 ];
    truth D.Bool; truth D.Bit;
  ]

type reduction =
  | Sum
  | Prod
  | Max
  | Min
  | Argmax
  | Argmin
  | All
  | Any
  | Count
  | Mean
  | Var
  | Std
  | Logsumexp
  | Logmeanexp

let reduction_name = function
  | Sum -> "sum"
  | Prod -> "prod"
  | Max -> "max"
  | Min -> "min"
  | Argmax -> "argmax"
  | Argmin -> "argmin"
  | All -> "all"
  | Any -> "any"
  | Count -> "count"
  | Mean -> "mean"
  | Var -> "var"
  | Std -> "std"
  | Logsumexp -> "logsumexp"
  | Logmeanexp -> "logmeanexp"

(* The reductions of floats alone. *)
let of_floats = [ Var; Std; Logsumexp; Logmeanexp ]

(* The reductions [dt] takes beside the extremes, positions and truths. *)
let folds (type v s) (dt : (v, s) D.t) =
  match D.kind dt with
  | D.Float -> [ Sum; Prod; Mean ] @ of_floats
  | D.Complex -> [ Sum; Prod; Mean ]
  | D.Signed | D.Unsigned -> [ Sum; Prod ]
  | D.Boolean -> []

type reduced_case =
  | Reduced_case :
      ('v, 's) reduced
      * int array
      * 'v array
      * where
      * int list option
      * reduction
      -> reduced_case

let pp_reduced_case ppf (Reduced_case (k, s, data, w, axes, r)) =
  Format.fprintf ppf "%s along %s of %s %a [%s] at %a" (reduction_name r)
    (match axes with
    | None -> "every axis"
    | Some l -> String.concat "," (List.map string_of_int l))
    (D.name k.rdtype) pp_ints s
    (String.concat "; " (Array.to_list (Array.map k.rshow data)))
    pp_where w

(* [r] of the floats [x] along [axes], for [r] a reduction of floats alone. *)
let of_float (type s d) r axes (x : (float, s, d) Nx.t) =
  match r with
  | Var -> Nx.var ?axes x
  | Std -> Nx.std ?axes x
  | Logsumexp -> Nx.logsumexp ?axes x
  | Logmeanexp -> Nx.logmeanexp ?axes x
  | _ -> invalid_arg "of_float: not a reduction of floats alone"

(* Each result's bits as strings, [rshow] printing an element's, every NaN
   alike, or the message of the refusal. *)
let reduce (type v s d) (rshow : v -> string) r axes (x : (v, s, d) Nx.t) =
  let blind parts y =
    Array.map (nan_blind ~parts (Nx.dtype x) rshow) (elements y)
  in
  let show = blind false and parts = blind true in
  let int64s y = Array.map Int64.to_string (elements y) in
  let bools y = Array.map string_of_bool (elements y) in
  let axis = Option.map List.hd axes in
  match
    match r with
    | Sum -> parts (Nx.sum ?axes x)
    | Prod -> show (Nx.prod ?axes x)
    | Max -> show (Nx.max ?axes x)
    | Min -> show (Nx.min ?axes x)
    | Argmax -> int64s (Nx.argmax ?axis x)
    | Argmin -> int64s (Nx.argmin ?axis x)
    | All -> bools (Nx.all ?axes x)
    | Any -> bools (Nx.any ?axes x)
    | Count -> int64s (Nx.count ?axes (Nx.cast D.Bool x))
    | Mean -> parts (Nx.mean ?axes x)
    | Var | Std | Logsumexp | Logmeanexp -> (
        match D.kind (Nx.dtype x) with
        | D.Float -> show (of_float r axes x)
        | D.Complex | D.Signed | D.Unsigned | D.Boolean ->
            invalid_arg "reduce: a reduction of floats alone")
  with
  | v -> Ok v
  | exception Invalid_argument m -> Error m

let printed = function
  | Ok v -> String.concat " " (Array.to_list v)
  | Error m -> "raises " ^ m

(* {!reduce} of [x] placed as [w]. *)
let reduce_at (type v s) w (rshow : v -> string) r axes
    (x : (v, s, Nx.Host.d) Nx.t) =
  match w with
  | Split3 a -> reduce rshow r axes (Nx.place (Set3.split ~axis:a) x)
  | Split4 a -> reduce rshow r axes (Nx.place (Set4.split ~axis:a) x)
  | Cuts (a, b) ->
      let cuts =
        if a = b then [ (a, [ "a"; "b" ]) ] else [ (a, [ "a" ]); (b, [ "b" ]) ]
      in
      reduce rshow r axes (Nx.place (Nx.Placement.mesh mesh4 cuts) x)
  | Minor a ->
      reduce rshow r axes
        (Nx.place (Nx.Placement.mesh mesh4 [ (a, [ "b" ]) ]) x)

(* [y]'s bits, as [k] shows them for a sub-byte dtype, whose element's bits
   are its value, which no byte holds alone. *)
let shown_bits (type v s d) (k : (v, s) reduced) (y : (v, s, d) Nx.t) =
  if D.bits k.rdtype < 8 then Array.map k.rshow (elements y)
  else Array.map string_of_int (bytes y)

(* Gathers along cut axes *)

(* A gather along an axis at positions of one axis, or of the value's rank. *)
type gather = Take of int | Along of int

let pp_gather ppf = function
  | Take a -> Format.fprintf ppf "take along %d" a
  | Along a -> Format.fprintf ppf "take_along_axis %d" a

(* Where the positions lie: whole on each of the value's devices; cut alike
   along the value's cut axes but the gathered one; beside a value cut over
   the mesh's minor axis, along an axis over its major one; or split along
   an axis over every device of the value's set, where no order reads them
   beside the value. *)
type positions = Whole | Alike | Major of int | Spread of int

let pp_positions ppf = function
  | Whole -> Format.pp_print_string ppf "whole"
  | Alike -> Format.pp_print_string ppf "cut alike"
  | Major a -> Format.fprintf ppf "cut %d/a" a
  | Spread a -> Format.fprintf ppf "split %d" a

(* A value's dtype, shape and elements, where it lies, the gather, where the
   positions lie, and their shape and elements. *)
type gcase =
  | Gcase :
      ('v, 's) reduced
      * int array
      * 'v array
      * where
      * gather
      * positions
      * int array
      * int64 array
      -> gcase

let pp_gcase ppf (Gcase (k, s, data, w, g, at, ps, p)) =
  Format.fprintf ppf "%s %a [%s] at %a, %a %a [%s] %a" (D.name k.rdtype) pp_ints
    s
    (String.concat "; " (Array.to_list (Array.map k.rshow data)))
    pp_where w pp_gather g pp_ints ps
    (String.concat "; " (Array.to_list (Array.map Int64.to_string p)))
    pp_positions at

(* The dtypes a gather draws: each of its own bits' dtype, packed 4-bit and
   8-bit floats and 64-bit integers among them. *)
let gathered_kinds =
  let never _ = false in
  let float dt values =
    Reduced
      { rdtype = dt; rvalues = values; rshow = nan_bits; is_nan = Float.is_nan }
  in
  let int64 dt =
    Reduced
      {
        rdtype = dt;
        rvalues = [ 0L; 1L; -1L; Int64.max_int; Int64.min_int ];
        rshow = Int64.to_string;
        is_nan = never;
      }
  in
  reduced_kinds
  @ [
      float D.Float8_e4m3fn floats;
      float D.Float8_e5m2 floats;
      float D.Float4_e2m1fn [ 0.; -0.; 0.5; 1.5; 6.; -6. ];
      int64 D.Int64;
      int64 D.Uint64;
      Reduced
        {
          rdtype = D.Int16;
          rvalues = [ 0; 1; -1; 32767; -32768 ];
          rshow = string_of_int;
          is_nan = never;
        };
    ]

let gathered_case =
  let open Gen in
  with_pp pp_gcase
    (let* rank = int_range 1 3 in
     let* s = array ~size:(constant rank) (of_list [ 0; 1; 2; 3; 5; 7 ]) in
     let* w = where_of rank in
     let cut = List.map fst (tiles w) in
     let* a = one_of [ of_list cut; int_range 0 (rank - 1) ] in
     let* g = of_list [ Take a; Along a ] in
     let* m = int_range 0 4 in
     let ps =
       match g with
       | Take _ -> [| m |]
       | Along a -> Array.mapi (fun i e -> if i = a then m else e) s
     in
     (* Cut along an axis the value's cut also becomes, positions would
        need both cuts on one axis. *)
     let alike =
       match g with
       | Take _ -> []
       | Along a -> if List.exists (( <> ) a) cut then [ Alike ] else []
     in
     let major =
       match (w, g) with
       | Minor c, Along a ->
           List.filter_map
             (fun i -> if i = c && c <> a then None else Some (Major i))
             (List.init rank Fun.id)
       | Minor _, Take _ -> [ Major 0 ]
       | (Split3 _ | Split4 _ | Cuts _), _ -> []
     in
     (* Split along an axis the value's cut also becomes, or beside a value
        of no element, which every device holds, positions are read where
        they lie. *)
     let spread =
       if Array.mem 0 s then []
       else
       match g with
       | Take _ -> [ Spread 0 ]
       | Along a ->
           List.filter_map
             (fun i ->
               if List.mem i cut && i <> a then None else Some (Spread i))
             (List.init rank Fun.id)
     in
     let* at = of_list ((Whole :: alike) @ major @ spread) in
     let position =
       one_of
         [
           map Int64.of_int (int_range (-2) (s.(a) + 1));
           of_list [ Int64.min_int; Int64.max_int ];
         ]
     in
     let* p = array ~size:(constant (Array.fold_left ( * ) 1 ps)) position in
     let* (Reduced k) = of_list gathered_kinds in
     let+ data =
       array ~size:(constant (Array.fold_left ( * ) 1 s)) (of_list k.rvalues)
     in
     Gcase (k, s, data, w, g, at, ps, p))

(* A gather of a value at any placement, at positions whole on each of its
   devices, cut along axes the value's gathered cut leaves, or split where
   no order reads them, gives its bits on one device. Positions split so
   move whole to every device, and the result lies as beside positions
   whole there. *)
let law_gathered (Gcase (k, s, data, w, g, at, ps, p)) =
  let ts = tiles w in
  let cut = List.map fst ts in
  let a = match g with Take a | Along a -> a in
  let e = s.(a) in
  cover "an uneven cut" (List.exists (fun (c, t) -> s.(c) mod t <> 0) ts);
  cover "an empty window"
    (List.exists (fun (c, t) -> s.(c) > 0 && s.(c) < t) ts);
  cover "a cut over several mesh axes"
    (match w with Cuts (x, y) -> x = y | _ -> false);
  cover "a gather along a cut axis" (List.mem a cut);
  cover "a gather along a cut empty axis" (List.mem a cut && e = 0);
  cover "a gather along a cut axis with an empty tile"
    (List.exists (fun (c, t) -> c = a && e > 0 && e < t) ts);
  cover "a gather along a cut axis, beside another cut"
    (List.mem a cut && List.length cut > 1);
  cover "a gather along an axis no cut names" (not (List.mem a cut));
  cover "positions cut alike, gathered along a cut axis"
    (at = Alike && List.mem a cut);
  cover "positions cut over the major axis, gathered along a cut axis"
    (match at with
    | Major _ -> List.mem a cut
    | Whole | Alike | Spread _ -> false);
  cover "positions cut along the gathered axis"
    (at = Major a && match g with Along _ -> true | Take _ -> false);
  let spread =
    match at with Spread _ -> true | Whole | Alike | Major _ -> false
  in
  cover "positions moved, gathered along a cut axis" (spread && List.mem a cut);
  cover "positions moved, beside a cut of another axis"
    (spread && List.exists (( <> ) a) cut);
  cover "positions moved, split along the gathered axis"
    (at = Spread a && match g with Along _ -> true | Take _ -> false);
  cover "one position moved" (spread && Array.length p = 1);
  (* Split positions move only where no device holds what it would read
     beside the value: where every cut axis has a nonempty tile on each
     device. Where one holds no element, the route may read them where
     they lie. *)
  let placed =
    match at with
    | Whole | Alike | Major _ -> true
    | Spread i ->
        let n = match w with Split3 _ -> 3 | Split4 _ | Cuts _ | Minor _ -> 4 in
        List.for_all (fun (c, t) -> s.(c) >= t) ts && ps.(i) >= n
  in
  cover "positions moved where no device holds its reads" (spread && placed);
  cover "a negative position" (Array.exists (fun q -> q < 0L) p);
  cover "a position past the axis"
    (Array.exists (fun q -> q >= Int64.of_int e) p);
  (* Tile [t - 1] of [t] starts at [(t - 1) (e / t) + min (t - 1) (e mod t)]. *)
  let last t = ((t - 1) * (e / t)) + Stdlib.min (t - 1) (e mod t) in
  cover "a position in the last tile of a cut axis"
    (List.exists
       (fun (c, t) ->
         c = a && e >= t
         && Array.exists
              (fun q -> q >= Int64.of_int (last t) && q < Int64.of_int e)
              p)
       ts);
  cover "no position" (p = [||]);
  cover "a signalling NaN" (signals k.rdtype data);
  cover "a sub-byte dtype" (D.bits k.rdtype < 8);
  cover "a complex128" (D.equal k.rdtype D.Complex128);
  cover "a float8"
    (D.equal k.rdtype D.Float8_e4m3fn || D.equal k.rdtype D.Float8_e5m2);
  cover "a float4" (D.equal k.rdtype D.Float4_e2m1fn);
  cover "a 64-bit integer"
    (D.equal k.rdtype D.Int64 || D.equal k.rdtype D.Uint64);
  let gather (type d) (p : (int64, _, d) Nx.t) (x : (_, _, d) Nx.t) =
    match g with
    | Take a -> Nx.take ~axis:a p x
    | Along a -> Nx.take_along_axis ~axis:a p x
  in
  let x = signalling k.rdtype s data in
  let p =
    Nx.Repr.of_array Nx.Host.v (A.of_array D.Int64 (Iarray.of_array ps) p)
  in
  let bits y = shown_bits k y in
  let expected = bits (gather p x) in
  (* The result lies at the union of the positions' cuts and [x]'s, but
     the gathered axis's ({!Nx.section-routing}). [major i] is where the
     positions cut along [i] lie, and the result. *)
  let run (type d) (q : d Nx.Placement.t) ~(alike : d Nx.Placement.t)
      ~(want : d Nx.Placement.t) major =
    let r, want =
      match at with
      | Whole -> (Nx.Placement.uncut q, want)
      | Alike -> (alike, want)
      | Major i -> major i
      | Spread i -> (Nx.Placement.split ~axis:i (Nx.Placement.devices q), want)
    in
    let y = gather (Nx.place r p) (Nx.place q x) in
    let placement =
      Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal
    in
    (* Read where they lie, split positions put their cut on the result's
       axis they become. *)
    let kept i =
      let o = match g with Along _ -> i | Take _ -> a in
      Nx.Placement.split ~axis:o (Nx.Placement.devices q)
    in
    (match at with
    | Spread i when not placed ->
        let y = Nx.placement y in
        let want = if Nx.Placement.equal want y then want else kept i in
        equal ~msg:"where it lies" placement want y
    | Whole | Alike | Major _ | Spread _ ->
        equal ~msg:"where it lies" placement want (Nx.placement y));
    bits y
  in
  let mesh cuts = Nx.Placement.mesh mesh4 cuts in
  let off cuts = List.filter (fun (c, _) -> c <> a) cuts in
  let none _ = invalid_arg "law_gathered: a major axis beside no minor cut" in
  let split (type d) (q : d Nx.Placement.t) c =
    run q ~alike:q ~want:(if c = a then Nx.Placement.uncut q else q) none
  in
  let actual =
    match w with
    | Split3 c -> split (Set3.split ~axis:c) c
    | Split4 c -> split (Set4.split ~axis:c) c
    | Cuts (c, d) ->
        let cuts =
          if c = d then [ (c, [ "a"; "b" ]) ]
          else [ (c, [ "a" ]); (d, [ "b" ]) ]
        in
        let rest = mesh (off cuts) in
        run (mesh cuts) ~alike:rest ~want:rest none
    | Minor c ->
        let cuts = [ (c, [ "b" ]) ] in
        let rest = mesh (off cuts) in
        (* The positions' axis [i] becomes the result's [i] in
           [take_along_axis], and [take]'s [a]. *)
        let major i =
          let o = match g with Along _ -> i | Take _ -> a in
          let cuts = List.sort compare ((o, [ "a" ]) :: off cuts) in
          (mesh [ (i, [ "a" ]) ], mesh cuts)
        in
        run (mesh cuts) ~alike:rest ~want:rest major
  in
  equal (array string) expected actual

(* Positions cut over the devices that cut the gathered axis move whole to
   every device, which then gathers from its tile: the result lies whole on
   every device, as beside positions whole there. *)
let test_gather_both_cut () =
  let v = Nx.arange D.Float32 0 8 1 in
  let x = Nx.place (Set4.split ~axis:0) v in
  let host =
    Nx.Repr.of_array Nx.Host.v (A.of_array D.Int64 [| 4 |] [| 7L; 0L; 3L; 4L |])
  in
  let p = Nx.place (Set4.split ~axis:0) host in
  let y = Nx.take_along_axis ~axis:0 p x in
  equal (placement ()) Set4.on (Nx.placement y);
  equal (array int) (bytes (Nx.take_along_axis ~axis:0 host v)) (bytes y)

(* Positions that no order reads where they lie move whole to every device
   of the set, whatever the operation reshaped them into: positions on one
   device beside a value cut across the set, and a [take]'s positions cut
   along their axis, of one element or several, beside a value cut along
   another or along the gathered axis. The result lies as beside positions
   whole on every device: where the value lies, or whole on every device
   where the value is cut along the gathered axis. *)
let test_positions_moved () =
  let v = Nx.reshape [| 4; 6 |] (Nx.arange D.Float32 0 24 1) in
  let positions l =
    Nx.Repr.of_array Nx.Host.v
      (A.of_array D.Int64 [| List.length l |] (Array.of_list l))
  in
  let check ?want name axis host p x =
    let y = Nx.take ~axis p x in
    let want = Option.value want ~default:(Nx.placement x) in
    equal ~msg:name (placement ()) want (Nx.placement y);
    equal ~msg:name (array int) (bytes (Nx.take ~axis host v)) (bytes y)
  in
  let host = positions [ 3L; 0L; 5L; 1L ] in
  check "one device" 1 host
    (Nx.place (Nx.Placement.device Set4.v (m 0)) host)
    (Nx.place (Set4.split ~axis:0) v);
  check "a take's positions cut" 0 host
    (Nx.place (Set4.split ~axis:0) host)
    (Nx.place (Set4.split ~axis:1) v);
  let one = positions [ 2L ] in
  check "one position cut" 1 one
    (Nx.place (Set4.split ~axis:0) one)
    (Nx.place (Set4.split ~axis:0) v);
  check ~want:Set4.on "one position cut, along a cut axis" 1 one
    (Nx.place (Set4.split ~axis:0) one)
    (Nx.place (Set4.split ~axis:1) v)

(* Positions cut finer than the value they read, along an axis it holds in
   halves, are read where they lie: each quarter lies within a half. *)
let test_positions_finer () =
  let v = Nx.reshape [| 8; 4 |] (Nx.arange D.Float32 0 32 1) in
  let host =
    Nx.Repr.of_array Nx.Host.v
      (A.of_array D.Int64 [| 8; 2 |]
         (Array.init 16 (fun i -> Int64.of_int (i mod 5))))
  in
  let x = Nx.place (Nx.Placement.mesh mesh4 [ (0, [ "a" ]) ]) v in
  let p = Nx.place (Set4.split ~axis:0) host in
  let y = Nx.take_along_axis ~axis:1 p x in
  let placement =
    Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal
  in
  equal placement (Nx.placement p) (Nx.placement y);
  equal (array int) (bytes (Nx.take_along_axis ~axis:1 host v)) (bytes y)

(* An [x] whose tiles' results would stack past the greatest rank, one more
   axis than [x]'s and one more for a [complex128]'s bits, is read whole
   along the gathered axis, which raises where it is cut there; one of an
   axis fewer gathers where it lies. *)
type ranked = Ranked : ('v, 's) D.t * int * bool -> ranked

let ranked =
  [
    Ranked (D.Float32, A.Layout.max_rank, true);
    Ranked (D.Float32, A.Layout.max_rank - 1, false);
    Ranked (D.Complex128, A.Layout.max_rank - 1, true);
    Ranked (D.Complex128, A.Layout.max_rank - 2, false);
  ]

let ranked_name (Ranked (dt, r, _)) =
  Printf.sprintf "%s of %d axes" (D.name dt) r

let test_gather_rank (Ranked (dt, r, whole)) =
  let s = Array.init r (fun i -> if i = 0 then 4 else 1) in
  let x = Nx.cast dt (Nx.reshape s (Nx.arange D.Float32 0 4 1)) in
  let p =
    Nx.Repr.of_array Nx.Host.v
      (A.of_array D.Int64 (Iarray.of_array s) [| 3L; 1L; 0L; 2L |])
  in
  let y = Nx.place (Set4.split ~axis:0) x in
  let gather () = Nx.take_along_axis ~axis:0 (Nx.place (Nx.beside y) p) y in
  if whole then
    raises
      (Invalid_argument
         "Nx.take_along_axis: operand 1 lies at split ~axis:0 [m0; m1; m2; m3] \
          and is read at on [m0; m1; m2; m3]; Nx.place (Placement.on s) y")
      gather
  else
    equal (array int)
      (bytes (Nx.take_along_axis ~axis:0 p x))
      (bytes (gather ()))

(* [take] without an axis reads the flattening, which keeps the cut of the
   leading axis in tiles that divide both extents. *)
let test_take_flat () =
  let x = Nx.reshape [| 4; 6 |] (Nx.arange D.Float32 0 24 1) in
  let p =
    Nx.Repr.of_array Nx.Host.v
      (A.of_array D.Int64 [| 7 |] [| 6L; 0L; 23L; 24L; -1L; 11L; 12L |])
  in
  let y = Nx.place (Set4.split ~axis:0) x in
  equal (array int)
    (bytes (Nx.take p x))
    (bytes (Nx.take (Nx.place (Nx.beside y) p) y))

(* Scatters into cut targets *)

(* Where positions and updates lie: whole on each of the target's devices,
   or cut alike along its cut axes but the scattered one. *)
type beside = Uncut | Cut_alike

let pp_beside ppf = function
  | Uncut -> Format.pp_print_string ppf "whole"
  | Cut_alike -> Format.pp_print_string ppf "cut alike"

(* A target's dtype, shape and elements, where it lies, the axis, how it
   combines, where the positions and updates lie, their shape, the
   positions and the updates. *)
type scase =
  | Scase :
      ('v, 's) reduced
      * int array
      * 'v array
      * where
      * int
      * Nx.combine
      * beside
      * int array
      * int64 array
      * 'v array
      -> scase

let combine_name = function
  | Nx.Set -> "set"
  | Add -> "add"
  | Max -> "max"
  | Min -> "min"

let pp_scase ppf (Scase (k, s, data, w, a, c, at, us, p, u)) =
  let show f l = String.concat "; " (Array.to_list (Array.map f l)) in
  Format.fprintf ppf "%s %a [%s] at %a, %s along %d at %a [%s] of %a [%s]"
    (D.name k.rdtype) pp_ints s (show k.rshow data) pp_where w
    (combine_name c) a pp_ints us (show Int64.to_string p) pp_beside at
    (show k.rshow u)

let scattered_case =
  let open Gen in
  with_pp pp_scase
    (let* rank = int_range 1 3 in
     let* s = array ~size:(constant rank) (of_list [ 0; 1; 2; 3; 5; 7 ]) in
     let* w = where_of rank in
     let cut = List.map fst (tiles w) in
     let* a = one_of [ of_list cut; int_range 0 (rank - 1) ] in
     let* m = int_range 0 6 in
     let us = Array.mapi (fun i e -> if i = a then m else e) s in
     let alike = if List.exists (( <> ) a) cut then [ Cut_alike ] else [] in
     let* at = of_list (Uncut :: alike) in
     let n = Array.fold_left ( * ) 1 us in
     let position =
       one_of
         [
           map Int64.of_int (int_range (-2) (s.(a) + 1));
           of_list [ Int64.min_int; Int64.max_int ];
         ]
     in
     let* p = array ~size:(constant n) position in
     let* (Reduced k) = of_list reduced_kinds in
     let combines =
       match D.kind k.rdtype with
       | D.Boolean -> [ Nx.Set; Max; Min ]
       | D.Float | D.Complex | D.Signed | D.Unsigned ->
           [ Nx.Set; Add; Max; Min ]
     in
     let* c = of_list combines in
     let value = of_list k.rvalues in
     let* data = array ~size:(constant (Array.fold_left ( * ) 1 s)) value in
     let+ u = array ~size:(constant n) value in
     Scase (k, s, data, w, a, c, at, us, p, u))

(* A scatter into a target at any placement, of positions and updates whole
   on each of its devices or cut alike along axes off the scattered one,
   gives its bits on one device. *)
let law_scattered (Scase (k, s, data, w, a, c, at, us, p, u)) =
  let ts = tiles w in
  let cut = List.map fst ts in
  let e = s.(a) in
  cover "an uneven cut" (List.exists (fun (c, t) -> s.(c) mod t <> 0) ts);
  cover "an empty window"
    (List.exists (fun (c, t) -> s.(c) > 0 && s.(c) < t) ts);
  cover "a cut over several mesh axes"
    (match w with Cuts (x, y) -> x = y | _ -> false);
  cover "a scatter along a cut axis" (List.mem a cut);
  cover "a scatter along a cut empty axis" (List.mem a cut && e = 0);
  cover "a scatter along a cut axis, beside another cut"
    (List.mem a cut && List.length cut > 1);
  cover "a scatter along an axis no cut names" (not (List.mem a cut));
  cover "an uneven cut of the scattered axis"
    (List.exists (fun (c, t) -> c = a && e mod t <> 0) ts);
  cover "an int64 extreme position, scattered along a cut axis"
    (List.mem a cut
    && Array.exists (fun q -> q = Int64.min_int || q = Int64.max_int) p);
  cover "updates cut alike, scattered along a cut axis"
    (at = Cut_alike && List.mem a cut);
  List.iter
    (fun c' ->
      cover
        ("a scatter by " ^ combine_name c' ^ " along a cut axis")
        (c = c' && List.mem a cut))
    [ Nx.Set; Add; Max; Min ];
  let inside q = q >= 0L && q < Int64.of_int e in
  cover "a repeated position"
    (let q = List.filter inside (Array.to_list p) in
     List.length (List.sort_uniq compare q) < List.length q);
  cover "a negative position" (Array.exists (fun q -> q < 0L) p);
  cover "a position past the axis"
    (Array.exists (fun q -> q >= Int64.of_int e) p);
  let last t = ((t - 1) * (e / t)) + Stdlib.min (t - 1) (e mod t) in
  cover "a position in the last tile of a cut axis"
    (List.exists
       (fun (c, t) ->
         c = a && e >= t
         && Array.exists (fun q -> q >= Int64.of_int (last t) && inside q) p)
       ts);
  cover "no update" (p = [||]);
  cover "a signalling NaN" (signals k.rdtype data || signals k.rdtype u);
  cover "a sub-byte dtype" (D.bits k.rdtype < 8);
  let x = signalling k.rdtype s data and u = signalling k.rdtype us u in
  let p =
    Nx.Repr.of_array Nx.Host.v (A.of_array D.Int64 (Iarray.of_array us) p)
  in
  let scatter p u x = Nx.scatter ~combine:c ~axis:a p u x in
  (* A maximum's or a minimum's NaN is a NaN, its bits unpromised. *)
  let bits y =
    match c with
    | Nx.Set | Add -> shown_bits k y
    | Max | Min -> Array.map k.rshow (elements y)
  in
  let expected = bits (scatter p u x) in
  let run (type d) (q : d Nx.Placement.t) (alike : d Nx.Placement.t) =
    let r = match at with Uncut -> Nx.Placement.uncut q | Cut_alike -> alike in
    let y = scatter (Nx.place r p) (Nx.place r u) (Nx.place q x) in
    let placement =
      Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal
    in
    equal ~msg:"where it lies" placement q (Nx.placement y);
    let donated =
      scatter (Nx.place r p) (Nx.place r u) (Nx.donate (Nx.place q x))
    in
    equal ~msg:"into a donated target" (array string) (bits y) (bits donated);
    bits y
  in
  let mesh cuts = Nx.Placement.mesh mesh4 cuts in
  let off cuts = mesh (List.filter (fun (c, _) -> c <> a) cuts) in
  let split3 c = Set3.split ~axis:c and split4 c = Set4.split ~axis:c in
  let actual =
    match w with
    | Split3 c -> run (split3 c) (split3 c)
    | Split4 c -> run (split4 c) (split4 c)
    | Cuts (c, d) ->
        let cuts =
          if c = d then [ (c, [ "a"; "b" ]) ]
          else [ (c, [ "a" ]); (d, [ "b" ]) ]
        in
        run (mesh cuts) (off cuts)
    | Minor c ->
        let cuts = [ (c, [ "b" ]) ] in
        run (mesh cuts) (off cuts)
  in
  equal (array string) expected actual

(* Updates cut along the scattered axis raise: a device's window takes
   updates from every device. The refusal names the [Nx.place] that puts
   them whole on each device, which then computes. Positions cut so move
   whole to every device. *)
let test_scatter_cut_updates () =
  let x = Nx.zeros D.Float32 [| 8 |] in
  let u = Nx.arange D.Float32 1 5 1 in
  let host =
    Nx.Repr.of_array Nx.Host.v (A.of_array D.Int64 [| 4 |] [| 7L; 0L; 3L; 4L |])
  in
  let split v = Nx.place (Set4.split ~axis:0) v in
  let scatter p u = Nx.scatter ~axis:0 p u (split x) in
  let refusal =
    Invalid_argument
      "Nx.scatter: operand 1 lies at split ~axis:0 [m0; m1; m2; m3] and is \
       read at on [m0; m1; m2; m3]; Nx.place (Placement.on s) y"
  in
  raises refusal (fun () -> scatter (split host) (split u));
  raises refusal (fun () -> scatter (Nx.place Set4.on host) (split u));
  let expected = bytes (Nx.scatter ~axis:0 host u x) in
  equal (array int) expected
    (bytes (scatter (Nx.place Set4.on host) (Nx.place Set4.on u)));
  equal (array int) expected (bytes (scatter (split host) (Nx.place Set4.on u)))

(* Scatters of cut updates *)

(* Where an [Add]'s target lies beside positions and updates split along
   the scattered axis: whole on every device of three or four, the
   updates split over them, or cut along an axis over the mesh's minor
   axis, the updates over its major one. *)
type summed = On3 | On4 | Beside of int

let pp_summed ppf = function
  | On3 -> Format.pp_print_string ppf "whole over 3"
  | On4 -> Format.pp_print_string ppf "whole over 4"
  | Beside c -> Format.fprintf ppf "cut %d/b" c

(* Elements whose sums are exact in every order: integers, which wrap, and
   floats of few bits, signed zeros, infinities and NaNs among them, a
   NaN's bits shown alike. *)
let summed_kinds =
  let float dt =
    Reduced
      {
        rdtype = dt;
        rvalues = [ 0.; -0.; 1.; -1.; 2.; 0.5; infinity; neg_infinity; nan ];
        rshow = nan_bits;
        is_nan = Float.is_nan;
      }
  in
  let complex =
    Reduced
      {
        rdtype = D.Complex64;
        rvalues =
          List.map
            (fun (re, im) -> { Complex.re; im })
            [ (0., 0.); (1., -1.); (-0., 2.); (0.5, 0.); (infinity, 1.) ];
        rshow = complex_bits;
        is_nan = (fun c -> Float.is_nan c.Complex.re || Float.is_nan c.im);
      }
  in
  let ints =
    List.filter
      (fun (Reduced k) ->
        match D.kind k.rdtype with
        | D.Signed | D.Unsigned -> true
        | D.Float | D.Complex | D.Boolean -> false)
      reduced_kinds
  in
  [ float D.Float32; float D.Float64; float D.Float16; float D.Bfloat16 ]
  @ (complex :: ints)

(* A target's dtype, shape and elements, where it lies, the axis, the
   updates' shape, the positions and the updates. *)
type sumcase =
  | Sumcase :
      ('v, 's) reduced
      * int array
      * 'v array
      * summed
      * int
      * int array
      * int64 array
      * 'v array
      -> sumcase

let pp_sumcase ppf (Sumcase (k, s, data, w, a, us, p, u)) =
  let show f l = String.concat "; " (Array.to_list (Array.map f l)) in
  Format.fprintf ppf "%s %a [%s] at %a, add along %d %a [%s] of [%s]"
    (D.name k.rdtype) pp_ints s (show k.rshow data) pp_summed w a pp_ints us
    (show Int64.to_string p) (show k.rshow u)

let summed_case =
  let open Gen in
  with_pp pp_sumcase
    (let* rank = int_range 1 3 in
     let* s = array ~size:(constant rank) (of_list [ 0; 1; 2; 3; 5; 7 ]) in
     let* a = int_range 0 (rank - 1) in
     let* w =
       one_of
         [
           of_list [ On3; On4 ];
           map (fun c -> Beside c) (int_range 0 (rank - 1));
         ]
     in
     let* m = int_range 0 9 in
     let us = Array.mapi (fun i e -> if i = a then m else e) s in
     let n = Array.fold_left ( * ) 1 us in
     let position =
       one_of
         [
           map Int64.of_int (int_range (-2) (s.(a) + 1));
           of_list [ Int64.min_int; Int64.max_int ];
         ]
     in
     let* p = array ~size:(constant n) position in
     let* (Reduced k) = of_list summed_kinds in
     let value = of_list k.rvalues in
     let* data = array ~size:(constant (Array.fold_left ( * ) 1 s)) value in
     let+ u = array ~size:(constant n) value in
     Sumcase (k, s, data, w, a, us, p, u))

(* A scatter's [Add] of positions and updates split along the scattered axis
   sums each device's updates into the target where it lies, whole or cut,
   and gives the values it gives on one device. *)
let law_summed (Sumcase (k, s, data, w, a, us, p, u)) =
  let devices = match w with On3 -> 3 | On4 | Beside _ -> 4 in
  let tiles = match w with Beside _ -> 2 | On3 | On4 -> devices in
  let e = s.(a) and m = us.(a) in
  let inside q = q >= 0L && q < Int64.of_int e in
  cover "a target cut beside the updates"
    (match w with Beside _ -> true | On3 | On4 -> false);
  cover "a target cut along the scattered axis" (w = Beside a);
  cover "an empty tile of updates" (m > 0 && m < tiles);
  cover "no update" (Array.length p = 0);
  cover "updates in several tiles" (m >= tiles && Array.length p > 0);
  cover "a position outside the target"
    (Array.exists (fun q -> not (inside q)) p);
  cover "a repeated position"
    (let q = List.filter inside (Array.to_list p) in
     List.length (List.sort_uniq compare q) < List.length q);
  cover "a sub-byte dtype" (D.bits k.rdtype < 8);
  let kind =
    match D.kind k.rdtype with
    | D.Float -> "a float"
    | D.Complex -> "a complex number"
    | D.Signed | D.Unsigned | D.Boolean -> "an integer"
  in
  List.iter
    (fun c -> cover c (kind = c))
    [ "a float"; "a complex number"; "an integer" ];
  let x = A.of_array k.rdtype (Iarray.of_array s) data in
  let x = Nx.Repr.of_array Nx.Host.v x in
  let u =
    Nx.Repr.of_array Nx.Host.v (A.of_array k.rdtype (Iarray.of_array us) u)
  in
  let p =
    Nx.Repr.of_array Nx.Host.v (A.of_array D.Int64 (Iarray.of_array us) p)
  in
  let scatter p u x = Nx.scatter ~combine:Add ~axis:a p u x in
  let shown y = Array.map k.rshow (elements y) in
  let expected = shown (scatter p u x) in
  let run (type d) (q : d Nx.Placement.t) (r : d Nx.Placement.t) =
    let y = scatter (Nx.place r p) (Nx.place r u) (Nx.place q x) in
    equal ~msg:"where it lies" (placement ()) q (Nx.placement y);
    shown y
  in
  let actual =
    match w with
    | On3 -> run Set3.on (Set3.split ~axis:a)
    | On4 -> run Set4.on (Set4.split ~axis:a)
    | Beside c ->
        run
          (Nx.Placement.mesh mesh4 [ (c, [ "b" ]) ])
          (Nx.Placement.mesh mesh4 [ (a, [ "a" ]) ])
  in
  equal (array string) expected actual

(* Float reductions over cut axes *)

(* Over axes cut across devices, a float or complex sum, product, mean,
   logsumexp or moments combines partials that round ({!Partial}): each
   tile reduced into the carrier as one device reduces it, the partials
   stacked along a new leading axis in the tiles' order and reduced over it
   by the kernels, bit for bit. The devices are memory devices that nx.cpu
   computes on, so the host's reduction of the stack is theirs. *)

let gamma u k =
  let k = Float.of_int (Stdlib.max 0 k) in
  k *. u /. (1. -. (k *. u))

let u64 = epsilon_float /. 2.

(* Whether [r] of [dt] combines partials that round. *)
let rounds (type v s) (dt : (v, s) D.t) r =
  match D.kind dt with
  | D.Float | D.Complex -> (
      match r with
      | Sum | Prod | Mean | Var | Std | Logsumexp | Logmeanexp -> true
      | Max | Min | Argmax | Argmin | All | Any | Count -> false)
  | D.Signed | D.Unsigned | D.Boolean -> false

(* The elements of [r] of the floats or complex numbers [x]. *)
let fold_elements (type v s d) r axes (x : (v, s, d) Nx.t) : v array =
  match r with
  | Sum -> elements (Nx.sum ?axes x)
  | Prod -> elements (Nx.prod ?axes x)
  | Mean -> elements (Nx.mean ?axes x)
  | Var | Std | Logsumexp | Logmeanexp -> (
      match D.kind (Nx.dtype x) with
      | D.Float -> elements (of_float r axes x)
      | D.Complex | D.Signed | D.Unsigned | D.Boolean ->
          invalid_arg "fold_elements: a reduction of floats alone")
  | Max | Min | Argmax | Argmin | All | Any | Count ->
      invalid_arg "fold_elements: an exact reduction"

(* A polymorphic computation over a placed value. *)
type ('v, 's, 'a) over = { over : 'd. ('v, 's, 'd) Nx.t -> 'a }

let at_where (type v s a) w (f : (v, s, a) over) (x : (v, s, Nx.Host.d) Nx.t) : a =
  match w with
  | Split3 a -> f.over (Nx.place (Set3.split ~axis:a) x)
  | Split4 a -> f.over (Nx.place (Set4.split ~axis:a) x)
  | Cuts (a, b) ->
      let cuts =
        if a = b then [ (a, [ "a"; "b" ]) ] else [ (a, [ "a" ]); (b, [ "b" ]) ]
      in
      f.over (Nx.place (Nx.Placement.mesh mesh4 cuts) x)
  | Minor a -> f.over (Nx.place (Nx.Placement.mesh mesh4 [ (a, [ "b" ]) ]) x)

(* Balanced windows along an axis of [e] over [t], each its start and
   count: the first [e mod t] hold one more. *)
let windows e t =
  let q = e / t and r = e mod t in
  List.init t (fun j -> ((j * q) + Stdlib.min j r, q + if j < r then 1 else 0))

(* The tiles of [x] cut along [cuts], [(axis, windows)] by increasing
   axis: its windows that hold terms, in C order of their first
   indices. *)
let tiles_of (type v s d) cuts (x : (v, s, d) Nx.t) =
  let s = Nx.shape x in
  let pick acc (a, t) =
    let held = List.filter (fun (_, n) -> n > 0) (windows s.(a) t) in
    List.concat_map (fun l -> List.map (fun w -> (a, w) :: l) held) acc
  in
  let slice l =
    let index a =
      match List.assoc_opt a l with
      | Some (start, n) -> Nx.R (start, start + n)
      | None -> Nx.A
    in
    Nx.slice (List.init (Array.length s) index) x
  in
  List.map slice (List.fold_left pick [ [] ] cuts)

(* The partials [ps], each kept along the reduced axes, stacked along a new
   leading axis in order: a reduction over axis [0] closes them. *)
let stack ps = Nx.stack ~axis:0 ps

(* A reduction of floats alone [r] of [x] along [axes] from its [tiles], as
   {!combined}: the variance from the mean of the tiles' sums, then the
   tiles' sums of the deviations from it and of their squares; a logsumexp
   as the logsumexp of the tiles'. *)
let floats_combined (type s) r axes tiles (x : (float, s, Nx.Host.d) Nx.t) =
  let dt = Nx.dtype x in
  let (Nx_kernel.Any_float w) = Nx_kernel.float_carrier dt in
  let k v = Nx.scalar w v in
  let n = List.fold_left (fun n a -> n * Nx.dim a x) 1 axes in
  let n = k (Float.of_int n) in
  let xs = List.map (Nx.cast w) tiles in
  let sum t = Nx.sum ~axes ~keepdims:true t in
  let sums f = Nx.sum ~axes:[ 0 ] (stack (List.map (fun t -> sum (f t)) xs)) in
  let var () =
    let s = sums Fun.id in
    let m = Nx.where (Nx.isnan s) s (Nx.div s n) in
    let s1 = sums (fun t -> Nx.sub t m) in
    let s2 = sums (fun t -> Nx.square (Nx.sub t m)) in
    let q = Nx.div (Nx.sub s2 (Nx.mul s1 (Nx.div s1 n))) n in
    let v = Nx.where (Nx.less q (k 0.)) (k 0.) q in
    let v = Nx.where (Nx.equal s2 (k infinity)) s2 v in
    Nx.where (Nx.isnan m) m v
  in
  let lse () =
    let each = List.map (fun t -> Nx.logsumexp ~axes ~keepdims:true t) xs in
    Nx.logsumexp ~axes:[ 0 ] (stack each)
  in
  match r with
  | Var -> Nx.cast dt (var ())
  | Std -> Nx.cast dt (Nx.sqrt (var ()))
  | Logsumexp -> Nx.cast dt (lse ())
  | Logmeanexp -> Nx.cast dt (Nx.sub (lse ()) (Nx.log n))
  | _ -> invalid_arg "floats_combined: not a reduction of floats alone"

(* The float or complex fold [r] of [x] along [axes] from its [tiles], as
   partial.mli combines them: each tile reduced on the host into the
   carrier, the partials stacked and reduced, rounded once to [x]'s dtype;
   each result kept along [axes]. *)
let combined (type v s) r axes tiles (x : (v, s, Nx.Host.d) Nx.t) : v array =
  let dt = Nx.dtype x in
  let (D.Any c) = Nx_kernel.carrier (D.Any dt) in
  let n = List.fold_left (fun n a -> n * Nx.dim a x) 1 axes in
  let partials f = List.map (fun t -> f (Nx.cast c t)) tiles in
  let sum t = Nx.sum ~axes ~keepdims:true t in
  match r with
  | Sum -> elements (Nx.cast dt (Nx.sum ~axes:[ 0 ] (stack (partials sum))))
  | Prod ->
      let prod t = Nx.prod ~axes ~keepdims:true t in
      elements (Nx.cast dt (Nx.prod ~axes:[ 0 ] (stack (partials prod))))
  | Mean ->
      let n = Nx.scalar c (D.of_float c (Float.of_int n)) in
      let s = Nx.sum ~axes:[ 0 ] (stack (partials sum)) in
      elements (Nx.cast dt (Nx.div s n))
  | _ -> (
      match D.kind dt with
      | D.Float -> elements (floats_combined r axes tiles x)
      | D.Complex | D.Signed | D.Unsigned | D.Boolean ->
          invalid_arg "combined: not a float fold")

(* The float fold [r] of [x] placed as [w]: its tiles' partials stacked
   and reduced, or its bits on one device where [w] cuts no reduced axis or
   one has no element, which the route then reads whole. *)
let float_law (type v s) (show : v -> string) w r axes
    (x : (v, s, Nx.Host.d) Nx.t) =
  let s = Nx.shape x in
  let along =
    match axes with None -> List.init (Array.length s) Fun.id | Some l -> l
  in
  let cuts = List.filter (fun (a, _) -> List.mem a along) (tiles w) in
  let got = at_where w { over = (fun y -> fold_elements r axes y) } x in
  let want =
    if cuts = [] || List.exists (fun a -> s.(a) = 0) along then
      fold_elements r axes x
    else combined r along (tiles_of (List.sort compare cuts) x) x
  in
  equal (array string) (Array.map show want) (Array.map show got)

let reduced_case =
  let open Gen in
  with_pp pp_reduced_case
    (let* rank = int_range 1 3 in
     let* s = array ~size:(constant rank) (of_list [ 0; 1; 2; 3; 5; 7 ]) in
     let* w = where_of rank in
     let* (Reduced k) = of_list reduced_kinds in
     let* r =
       of_list
         (folds k.rdtype @ [ Max; Min; Argmax; Argmin; All; Any; Count ])
     in
     (* Axes along the cut ones most often; one for a position. *)
     let cut = List.map fst (tiles w) in
     let* axes =
       match r with
       | Argmax | Argmin ->
           frequency
             [
               (3, map (fun a -> Some [ a ]) (of_list cut));
               (1, map (fun a -> Some [ a ]) (int_range 0 (rank - 1)));
               (1, constant None);
             ]
       | _ ->
           frequency
             [
               ( 4,
                 map
                   (fun keep ->
                     let l =
                       List.filter (fun a -> keep.(a)) (List.init rank Fun.id)
                     in
                     Some (if l = [] then cut else l))
                   (array ~size:(constant rank) bool) );
               (1, constant None);
             ]
     in
     let n = Array.fold_left ( * ) 1 s in
     let+ data = array ~size:(constant n) (of_list k.rvalues) in
     Reduced_case (k, s, data, w, axes, r))

let law_reduced (Reduced_case (k, s, data, w, axes, r)) =
  let ts = tiles w and rank = Array.length s in
  let along = match axes with None -> List.init rank Fun.id | Some l -> l in
  let cut_along = List.filter (fun (a, _) -> List.mem a along) ts in
  let over = cut_along <> [] in
  cover "a reduction over a cut axis" over;
  cover ("a reduction over a cut axis: " ^ reduction_name r) over;
  cover "a cut over a reduced axis after the first reduced one"
    (List.exists (fun (a, _) -> a > List.hd along) cut_along);
  cover "an uneven cut over a reduced axis"
    (List.exists (fun (a, t) -> s.(a) mod t <> 0) cut_along);
  cover "an empty window over a reduced axis"
    (List.exists (fun (a, t) -> s.(a) > 0 && s.(a) < t) cut_along);
  cover "a cut reduced axis whose one tile holds its terms"
    (List.exists (fun (a, _) -> s.(a) = 1) cut_along);
  cover "a 0-d result over a cut axis" (over && List.length along = rank);
  cover "two cut reduced axes, one with an empty window"
    (List.length cut_along = 2
    && List.exists (fun (a, t) -> s.(a) > 0 && s.(a) < t) cut_along);
  cover "a cut reduced axis of no element"
    (List.exists (fun (a, _) -> s.(a) = 0) cut_along);
  cover "a reduced axis cut over two mesh axes"
    (over && match w with Cuts (x, y) -> x = y | _ -> false);
  cover "a reduced axis cut beside a kept one"
    (over && List.length cut_along < List.length ts);
  cover "a reduced axis cut, the other mesh axis holding copies"
    (over && match w with Minor _ -> true | _ -> false);
  cover "a position over a cut axis"
    (over && (r = Argmax || r = Argmin));
  cover "a NaN term over a cut axis" (over && Array.exists k.is_nan data);
  cover "a signalling NaN term over a cut axis" (over && signals k.rdtype data);
  cover "a sub-byte dtype over a cut axis" (over && D.bits k.rdtype < 8);
  cover "a float sum, product or moments over a cut axis"
    (over && rounds k.rdtype r);
  let x = signalling k.rdtype s data in
  let one_term t = List.for_all (fun a -> Nx.dim a t = 1) along in
  cover "a float sum, product or moments over a tile of one term"
    (over && rounds k.rdtype r
    && List.exists one_term (tiles_of (List.sort compare cut_along) x));
  (* Without an axis, a position is read over [x] reshaped to one axis: it
     raises where that reshape does, naming itself. *)
  let flat =
    match (r, axes) with
    | (Argmax | Argmin), None ->
        let n = Array.fold_left ( * ) 1 s in
        let refusal y =
          match Nx.reshape [| n |] y with
          | _ -> None
          | exception Invalid_argument m -> Some m
        in
        at_where w { over = refusal } x
    | _ -> None
  in
  cover "a position among all elements whose reshape raises" (flat <> None);
  if rounds k.rdtype r then float_law k.rshow w r axes x
  else
    let want =
      match flat with
      | None -> printed (reduce k.rshow r axes x)
      | Some m ->
          let by = "Nx.reshape: " in
          let n = String.length by in
          equal bool ~msg:"the reshape's refusal" true
            (String.starts_with ~prefix:by m);
          let rest = String.sub m n (String.length m - n) in
          "raises Nx." ^ reduction_name r ^ ": " ^ rest
    in
    equal string want (printed (reduce_at w k.rshow r axes x))

(* The minor words a float32 sum along an axis cut across devices allocates grow
   with the devices, each holding a partial that the closing stacks and moves,
   and not with the elements: three devices allocate more than one, and four
   more than three, and ten times the rows over three devices add less than a
   word per added element, which an allocation per element would exceed. *)
let test_closing_words () =
  let sum (type d) (p : d Nx.Placement.t) s =
    let n = Array.fold_left ( * ) 1 s in
    let x =
      Nx.place p
        (Nx.Repr.of_array Nx.Host.v
           (A.of_array D.Float32 (Iarray.of_array s) (Array.make n 1.)))
    in
    words (fun () -> Nx.sum ~axes:[ 0 ] x)
  in
  let module One = (val Nx.devices [ m 0 ]) in
  let three = sum (Set3.split ~axis:0) [| 6; 4 |] in
  less float_exact ~msg:"one device" ~than:three (sum One.on [| 6; 4 |]);
  less float_exact ~msg:"four devices"
    ~than:(sum (Set4.split ~axis:0) [| 6; 4 |])
    three;
  less float_exact ~msg:"ten times the rows"
    ~than:(three +. Float.of_int ((60 - 6) * 4))
    (sum (Set3.split ~axis:0) [| 60; 4 |])

(* An extreme's position over two axes of a value cut along the second
   over three devices: the first tile, columns 0 and 1, holds an extreme at
   position 4, the second at position 2. The position is the smaller,
   whichever tile comes first, as on one device. Positions are numbered in
   C order of the reduced indices. *)
let test_arg_ties () =
  let arg (type d) e (x : (float, D.float32_elt, d) Nx.t) =
    let open Nx.Prim in
    let (_, i), () =
      eval ~by:"test_arg_ties"
        (Reduce
           {
             layout = A.Layout.contiguous (Iarray.of_array (Nx.shape x));
             axes = [| 0; 1 |];
             prog = Nx_kernel.Prog.of_node ~ins:[| D.Any D.Float32 |] (In 0);
             reductions = [ Arg (e, 0, D.Float32) ];
             loads = [| Plain x |];
           })
    in
    (elements i).(0)
  in
  let ties msg (e : Nx_kernel.extreme) xs =
    let x = Nx.create Nx.float32 [| 2; 4 |] xs in
    equal ~msg int64 2L (arg e x);
    equal ~msg int64 2L (arg e (Nx.place (Set3.split ~axis:1) x))
  in
  ties "maxima" Max [| 0.; 0.; 9.; 0.; 9.; 0.; 0.; 0. |];
  ties "minima" Min [| 0.; 0.; -9.; 0.; -9.; 0.; 0.; 0. |];
  ties "NaNs" Max [| 0.; 0.; nan; 0.; -.nan; 9.; 0.; 0. |]

(* A float or complex maximum or minimum over a cut axis is a NaN where a
   term is, whichever tiles hold one: which NaN is unspecified. *)
let test_cut_extreme_nan () =
  let nan k = Int64.float_of_bits (Int64.logor 0x7ff8_0000_0000_0000L k) in
  let cut x = Nx.place (Set3.split ~axis:0) x in
  let extremes (type s) (x : (float, s, Nx.host) Nx.t) =
    let one f = Float.is_nan (elements (f (cut x))).(0) in
    [ one (fun x -> Nx.max x); one (fun x -> Nx.min x) ]
  in
  let complex_extremes x =
    let z = Nx.complex Nx.complex64 ~re:(Nx.ones_like x) ~im:x in
    let one f =
      let c = (elements (f (cut z))).(0) in
      Float.is_nan c.Complex.re || Float.is_nan c.im
    in
    [ one (fun x -> Nx.max x); one (fun x -> Nx.min x) ]
  in
  let nans msg xs =
    let x = Nx.create Nx.float64 [| 6 |] xs in
    let x32 = Nx.cast Nx.float32 x in
    equal ~msg (list bool) (List.init 6 (fun _ -> true))
      (extremes x @ extremes x32 @ complex_extremes x32)
  in
  nans "in the first tile" [| nan 7L; 1.; 2.; 3.; 4.; 5. |];
  nans "in the last tile" [| 1.; 2.; 3.; 4.; 5.; nan 7L |];
  nans "in two tiles" [| 1.; 2.; nan 7L; 4.; -.nan 9L; 0. |];
  nans "beside infinities"
    [| Float.infinity; 1.; Float.neg_infinity; nan 3L; 2.; 0. |]

(* Float reductions over cut axes of many terms *)

(* A float or complex dtype and how its elements' bits print, NaN signs and
   payloads included. *)
type lane_kind = Lane_kind : ('v, 's) D.t * ('v -> string) -> lane_kind

let complex_bits c = float_bits c.Complex.re ^ "," ^ float_bits c.im

let lane_kinds =
  [
    Lane_kind (D.Float32, float_bits);
    Lane_kind (D.Float64, float_bits);
    Lane_kind (D.Float16, float_bits);
    Lane_kind (D.Bfloat16, float_bits);
    Lane_kind (D.Complex64, complex_bits);
    Lane_kind (D.Complex128, complex_bits);
  ]

(* Terms that decide a result's bits: NaNs of both signs and with a payload,
   infinities and [-0.]. *)
type special = Nan of float | Inf | Neg_inf | Neg_zero

let specials =
  List.map (fun x -> Nan x) [ nan; -.nan; Int32.float_of_bits 0x7fc00123l ]
  @ [ Inf; Neg_inf; Neg_zero ]

let special_value = function
  | Nan x -> x
  | Inf -> infinity
  | Neg_inf -> neg_infinity
  | Neg_zero -> -0.

type lane_case = {
  lkind : lane_kind;
  lshape : int array;
  laxes : int list option;
  lwhere : where;
  lr : reduction;
  seed : int;
  marks : (float * special) list;
      (* Specials at fractions of the terms, in C order. *)
}

let pp_lane_case ppf c =
  let (Lane_kind (dt, _)) = c.lkind in
  Format.fprintf ppf "%s along %s of %s %a at %a, seed %d, specials %s"
    (reduction_name c.lr)
    (match c.laxes with
    | None -> "every axis"
    | Some l -> String.concat "," (List.map string_of_int l))
    (D.name dt) pp_ints c.lshape pp_where c.lwhere c.seed
    (String.concat "; "
       (List.map
          (fun (f, s) -> Printf.sprintf "%g:%h" f (special_value s))
          c.marks))

(* A finite term for [r] from [st]: spread over 2^±20 for a sum, near 1 for a
   product, below 10 for a logsumexp. *)
let finite r st =
  match r with
  | Prod ->
      (if Random.State.bool st then 1. else -1.)
      *. (0.999 +. Random.State.float st 0.002)
  | Logsumexp | Logmeanexp -> Random.State.float st 210. -. 200.
  | _ ->
      Float.ldexp (Random.State.float st 2. -. 1.) (Random.State.int st 41 - 20)

(* The case's terms, of [dt]: finite ones drawn from its seed, then its
   specials, one part of a complex number each. *)
let lane_data (type v s) (dt : (v, s) D.t) c : v array =
  let st = Random.State.make [| c.seed |] in
  let n = Array.fold_left ( * ) 1 c.lshape in
  let floats = Array.init n (fun _ -> finite c.lr st) in
  let parts = Array.init n (fun _ -> 0.01 *. finite c.lr st) in
  let at f = Stdlib.min (n - 1) (int_of_float (f *. Float.of_int n)) in
  let complex = D.is D.Complex dt in
  List.iter
    (fun (f, sp) ->
      if n > 0 then
        if complex && Random.State.bool st then parts.(at f) <- special_value sp
        else floats.(at f) <- special_value sp)
    c.marks;
  match D.kind dt with
  | D.Float -> floats
  | D.Complex -> Array.map2 (fun re im -> { Complex.re; im }) floats parts
  | D.Signed | D.Unsigned | D.Boolean -> invalid_arg "lane_data: not a float"

let lane_case =
  let open Gen in
  with_pp pp_lane_case
    (let* rank = int_range 1 3 in
     (* One axis of many terms, the others few. *)
     let* big = int_range 0 (rank - 1) in
     let* lshape = array ~size:(constant rank) (of_list [ 1; 2; 3; 5 ]) in
     let* many =
       if rank = 1 then
         of_list [ 0; 2; 3; 1023; 1024; 1025; 2049; 3000; 4096; 4097 ]
       else of_list [ 1000; 1024; 1025; 2048; 2049 ]
     in
     lshape.(big) <- many;
     let* lwhere = where_of rank in
     let* (Lane_kind (dt, _) as lkind) = of_list lane_kinds in
     let* lr =
       of_list
         (match D.kind dt with
         | D.Float -> [ Sum; Prod; Mean ] @ of_floats
         | _ -> [ Sum; Prod; Mean ])
     in
     let cut = List.map fst (tiles lwhere) in
     let* laxes =
       frequency
         [
           ( 4,
             map
               (fun keep ->
                 let l =
                   List.filter (fun a -> keep.(a)) (List.init rank Fun.id)
                 in
                 Some (List.sort_uniq compare (cut @ l)))
               (array ~size:(constant rank) bool) );
           (1, constant None);
         ]
     in
     let* seed = int in
     (* Infinities of both signs, whose sum is a NaN made from numbers. *)
     let+ marks =
       frequency
         [
           (1, constant []);
           ( 2,
             list ~size:(int_range 1 3)
               (pair (float_range 0. 1.) (of_list specials)) );
           ( 1,
             let+ a = float_range 0. 1. and+ b = float_range 0. 1. in
             [ (a, Inf); (b, Neg_inf) ] );
         ]
     in
     { lkind; lshape; laxes; lwhere; lr; seed; marks })

let law_lanes c =
  let (Lane_kind (dt, show)) = c.lkind in
  let rank = Array.length c.lshape in
  let along =
    match c.laxes with None -> List.init rank Fun.id | Some l -> l
  in
  let ts = tiles c.lwhere in
  let cut_along = List.filter (fun (a, _) -> List.mem a along) ts in
  let data = lane_data dt c in
  let n = List.fold_left (fun n a -> n * c.lshape.(a)) 1 along in
  cover ("a reduction of " ^ D.name dt) true;
  cover ("a reduction by " ^ reduction_name c.lr) true;
  cover "no term" (n = 0);
  cover "an empty window along a reduced axis"
    (List.exists
       (fun (a, t) -> c.lshape.(a) > 0 && c.lshape.(a) < t)
       cut_along);
  cover "an uneven cut along a reduced axis"
    (List.exists (fun (a, t) -> c.lshape.(a) mod t <> 0) cut_along);
  cover "a cut reduced axis after the first reduced one"
    (List.exists (fun (a, _) -> a > List.hd along) cut_along);
  cover "two reduced axes cut" (List.length cut_along = 2);
  cover "a reduced axis cut over two mesh axes"
    (match c.lwhere with Cuts (a, b) -> a = b | _ -> false);
  let has p = List.exists (fun (_, s) -> p s) c.marks in
  cover "infinities of both signs and no NaN term"
    (List.length along = rank && has (( = ) Inf) && has (( = ) Neg_inf)
    && not (has (function Nan _ -> true | _ -> false)));
  cover "a signalling NaN term" (signals dt data);
  let x = signalling dt c.lshape data in
  float_law (nan_blind ~parts:(c.lr <> Prod) dt show) c.lwhere c.lr c.laxes x

(* Contractions over cut axes *)

(* An operand dtype, the accumulator a contraction of it takes, how its
   elements' bits print, a draw of one, and for floats and complex numbers
   an element of a float's value, its parts, the accumulator's unit
   roundoff [u], the unit roundoff of the rounding into the operand's dtype
   where it is narrower, [out_u], the error of an underflow there, [tiny],
   and its largest finite value, [max]. *)
type ckind =
  | Ckind : {
      cdtype : ('v, 's) D.t;
      acc : D.any;
      cshow : 'v -> string;
      draw : Random.State.t -> 'v;
      floats : ('v floats) option;
    }
      -> ckind

and 'v floats = {
  of_float : float -> 'v;
  parts : 'v -> float * float;
  u : float;
  out_u : float;
  tiny : float;
  max : float;
}

(* A finite float spread over 2^±20, so that sums depend on their
   association, or [-0.]. *)
let spread st =
  if Random.State.int st 16 = 0 then -0.
  else
    Float.ldexp (Random.State.float st 2. -. 1.) (Random.State.int st 41 - 20)

(* A contraction's bits, every NaN alike: a NaN result promises a NaN, never
   which one. Where a float16 operand overflows to infinities, their sums
   are NaNs. *)
let product_bits v = if Float.is_nan v then "nan" else float_bits v
let complex_product_bits c = product_bits c.Complex.re ^ "," ^ product_bits c.im
let u32 = Float.ldexp 1. (-24)
let f32_max = Float.ldexp (Float.of_int 0xffffff) 104

let ckinds =
  let float dt acc ~u ~out_u ~tiny ~max =
    let floats =
      { of_float = Fun.id; parts = (fun x -> (x, 0.)); u; out_u; tiny; max }
    in
    Ckind
      { cdtype = dt; acc = D.Any acc; cshow = product_bits; draw = spread;
        floats = Some floats }
  in
  let complex dt ~u ~max =
    let floats =
      { of_float = (fun re -> { Complex.re; im = 0. });
        parts = (fun c -> (c.Complex.re, c.im)); u; out_u = 0.; tiny = 0.;
        max }
    in
    Ckind
      {
        cdtype = dt;
        acc = D.Any dt;
        cshow = complex_product_bits;
        draw = (fun st -> { Complex.re = spread st; im = spread st });
        floats = Some floats;
      }
  in
  let int32s = [| 0l; 1l; -1l; 7l; Int32.max_int; Int32.min_int |] in
  [
    float D.Float32 D.Float32 ~u:u32 ~out_u:0. ~tiny:0. ~max:f32_max;
    float D.Float64 D.Float64 ~u:u64 ~out_u:0. ~tiny:0. ~max:max_float;
    float D.Float16 D.Float32 ~u:u32 ~out_u:(Float.ldexp 1. (-11))
      ~tiny:(Float.ldexp 1. (-25)) ~max:65504.;
    float D.Bfloat16 D.Float32 ~u:u32 ~out_u:(Float.ldexp 1. (-8))
      ~tiny:(Float.ldexp 1. (-134)) ~max:(Float.ldexp 255. 120);
    complex D.Complex64 ~u:u32 ~max:f32_max;
    complex D.Complex128 ~u:u64 ~max:max_float;
    Ckind
      {
        cdtype = D.Int32;
        acc = D.Any D.Int32;
        cshow = Int32.to_string;
        draw = (fun st -> int32s.(Random.State.int st (Array.length int32s)));
        floats = None;
      };
    Ckind
      {
        cdtype = D.Int8;
        acc = D.Any D.Int32;
        cshow = string_of_int;
        draw = (fun st -> Random.State.int st 256 - 128);
        floats = None;
      };
  ]

(* Specials at fractions of [a]'s elements, or of [b]'s where the flag is
   set: a draw of a few, a quarter of the time. *)
let marks_of =
  let open Gen in
  frequency
    [
      (3, constant []);
      ( 1,
        list ~size:(int_range 1 2)
          (triple bool (float_range 0. 1.) (of_list specials)) );
    ]

(* [n] elements by [draw] from [st], with the specials of [marks] on
   [b]'s side or not as [side], for floats and complex numbers. *)
let drawn (type v) (draw : Random.State.t -> v) (floats : v floats option) st
    marks side n : v array =
  let data = Array.init n (fun _ -> draw st) in
  (match floats with
  | Some f when n > 0 ->
      List.iter
        (fun (on_b, at, sp) ->
          if on_b = side then
            let i = Stdlib.min (n - 1) (int_of_float (at *. Float.of_int n)) in
            data.(i) <- f.of_float (special_value sp))
        marks
  | Some _ | None -> ());
  data

let pp_marks ppf marks =
  Format.pp_print_string ppf
    (String.concat "; "
       (List.map
          (fun (b, f, s) ->
            Printf.sprintf "%s %g:%h" (if b then "b" else "a") f
              (special_value s))
          marks))

(* Covers of a case's specials, where it has products. *)
let cover_marks (type v) (floats : v floats option) marks products =
  let has p =
    products > 0 && Option.is_some floats
    && List.exists (fun (_, _, s) -> p s) marks
  in
  cover "a NaN operand element" (has (function Nan _ -> true | _ -> false));
  cover "an infinite operand element"
    (has (function Inf | Neg_inf -> true | _ -> false));
  cover "a -0 operand element" (has (( = ) Neg_zero))

(* Where the operands lie, the contracting pair [p] cut alike in both: over
   three or four devices; over both mesh axes; over the minor one, beside
   [a]'s rows over the major one; over the minor one, beside the batch over
   the major one in both; or pair 0 over the major axis and pair 1 over the
   minor one. Or [b]'s pair [p] alone cut, over the minor axis, and [a]'s
   batch over the major one and rows over the minor one. *)
type ccut =
  | Pair3 of int
  | Pair4 of int
  | Pair_mesh of int
  | Pair_rows of int
  | Pair_batch of int
  | Pairs
  | Batch_rows of int

(* Where [init] lies: at the result's placement, or whole on every device,
   which differs where the result is cut. *)
type iwhere = Init_as_result | Init_everywhere

type ccase = {
  ck : ckind;
  nb : int option;  (* The batch's extent, if there is one. *)
  rows : int;
  cols : int;
  ks : int array;  (* The contracting pairs' extents. *)
  ccut : ccut;
  init : bool;
  iwhere : iwhere;
  cseed : int;
  cmarks : (bool * float * special) list;
}

let pp_ccase ppf c =
  let (Ckind k) = c.ck in
  let cut =
    match c.ccut with
    | Pair3 p -> Printf.sprintf "pair %d over 3" p
    | Pair4 p -> Printf.sprintf "pair %d over 4" p
    | Pair_mesh p -> Printf.sprintf "pair %d over 2x2" p
    | Pair_rows p -> Printf.sprintf "pair %d/b rows/a" p
    | Pair_batch p -> Printf.sprintf "pair %d/b batch/a" p
    | Pairs -> "pair 0/a pair 1/b"
    | Batch_rows p -> Printf.sprintf "batch/a rows/b, b's pair %d/b" p
  in
  Format.fprintf ppf "%s %s%d x %d by %a, %s%s, seed %d, specials %a"
    (D.name k.cdtype)
    (match c.nb with Some b -> Printf.sprintf "%d of " b | None -> "")
    c.rows c.cols pp_ints c.ks cut
    (match (c.init, c.iwhere) with
    | false, _ -> ""
    | true, Init_as_result -> " from init"
    | true, Init_everywhere -> " from init on every device")
    c.cseed pp_marks c.cmarks

(* The axis of [a] and of [b] that the contracting pair [p] names: [a] is
   [[batch; rows; pairs]], [b] [[batch; pairs; columns]]. *)
let pair_axes c p =
  let ob = if Option.is_some c.nb then 1 else 0 in
  (ob + 1 + p, ob + p)

(* Extents of one contracting pair either side of nx.cpu's lane blocks of
   1024 terms, 512 complex products, and of its blocks of 4096 products,
   and of several of those. *)
let pair_extents =
  [
    0; 1; 2; 3; 5; 511; 512; 513; 1023; 1024; 1025; 1500; 2047; 2048; 2049;
    4095; 4096; 4097; 8191; 8192; 8193; 12289; 20481;
  ]

(* Cases over one contracting pair of [extents], or, where [pairs], over
   two a quarter of the time. *)
let ccase_of ~pairs ~extents =
  let open Gen in
  with_pp pp_ccase
    (let* ck = of_list ckinds in
     let* two =
       if pairs then frequency [ (3, constant false); (1, constant true) ]
       else constant false
     in
     let* ccut =
       if two then
         of_list
           [ Pair3 1; Pair4 1; Pair_mesh 1; Pair_rows 1; Pair_mesh 0; Pairs ]
       else
         of_list
           [
             Pair3 0; Pair4 0; Pair_mesh 0; Pair_rows 0; Pair_batch 0;
             Batch_rows 0;
           ]
     in
     let* nb =
       match ccut with
       | Pair_batch _ | Batch_rows _ -> map Option.some (of_list [ 1; 2; 3 ])
       | _ -> frequency [ (3, constant None); (1, constant (Some 2)) ]
     in
     let* rows, cols =
       frequency
         [ (3, pair (int_range 1 3) (int_range 1 3));
           (1, pair (int_range 8 9) (int_range 8 9)) ]
     in
     let* ks =
       if two then
         let+ k0 = of_list [ 1; 2; 3; 5 ]
         and+ k1 = of_list [ 0; 1; 3; 300; 511; 700; 1700 ] in
         [| k0; k1 |]
       else
         let+ k = of_list extents in
         [| k |]
     in
     let* init = bool in
     let* iwhere = of_list [ Init_as_result; Init_everywhere ] in
     let* cseed = int in
     let+ cmarks = marks_of in
     { ck; nb; rows; cols; ks; ccut; init; iwhere; cseed; cmarks })

let ccase = ccase_of ~pairs:true ~extents:pair_extents

(* Contractions over one pair either side of a library's blocks of 4096
   products ({!Nx_support.Blocked}), and over enough blocks that one window
   holds several. *)
let bcase =
  ccase_of ~pairs:false
    ~extents:
      [
        1; 2; 1023; 1024; 1025; 2047; 2048; 2049; 4095; 4096; 4097; 8191; 8192;
        8193; 12289; 16384; 20481; 36865;
      ]

(* The case's operands and its contraction's [batch] and [contracting]
   pairs. *)
let ccase_values (type v s) (k : (v, s) D.t) draw floats c =
  let st = Random.State.make [| c.cseed |] in
  let lead = match c.nb with Some b -> [| b |] | None -> [||] in
  let sa = Array.concat [ lead; [| c.rows |]; c.ks ] in
  let sb = Array.concat [ lead; c.ks; [| c.cols |] ] in
  let so = Array.concat [ lead; [| c.rows; c.cols |] ] in
  let host side s =
    let n = Array.fold_left ( * ) 1 s in
    let data = drawn draw floats st c.cmarks side n in
    Nx.Repr.of_array Nx.Host.v (A.of_array k (Iarray.of_array s) data)
  in
  let a = host false sa and b = host true sb and i = host false so in
  let batch = match c.nb with Some _ -> [| (0, 0) |] | None -> [||] in
  let contracting = Array.init (Array.length c.ks) (pair_axes c) in
  (a, b, i, batch, contracting)

(* The contraction of [a] and [b] by [batch] and [contracting] from [init]
   in [acc] into [out]. *)
let contracted (type v s w q d) acc (out : (w, q) D.t) batch contracting
    (a : (v, s, d) Nx.t) (b : (v, s, d) Nx.t) (init : (w, q, d) Nx.t option) :
    (w, q, d) Nx.t =
  Nx.Prim.eval ~by:"test"
    (Contract
       {
         batch = Iarray.of_array batch;
         contracting = Iarray.of_array contracting;
         acc;
         out;
         a;
         b;
         init;
         a_scale = None;
         b_scale = None;
         segments = None;
       })

(* One kernel library's sets of memory devices: one, three and four, and a
   2 x 2 mesh of the four. *)
module type Sets = sig
  module One : Nx.Devices
  module Three : Nx.Devices
  module Four : Nx.Devices

  val mesh : Four.d Nx.Mesh.t
end

module Cpu_sets = struct
  module One = Nx.Host
  module Three = Set3
  module Four = Set4

  let mesh = mesh4
end

module Blocked_sets = struct
  let on ds = Nx.devices ~kernels:(module Nx_support.Blocked) (List.map m ds)

  module One = (val on [ 0 ])
  module Three = (val on [ 0; 1; 2 ])
  module Four = (val on [ 0; 1; 2; 3 ])

  let mesh = Nx.Mesh.v Four.v [ ("a", 2); ("b", 2) ]
end

(* The sum of [xs.(i) ys.(i)] as if in twice float64's precision, then
   rounded (Ogita, Rump and Oishi, "Accurate Sum and Dot Product", SIAM J.
   Sci. Comput. 26(6), 2005, Algorithm 5.3, Dot2): within [u64 |s| +
   γ(n)² Σ|x y|] of the exact [s]. *)
let dot2 xs ys =
  let s = ref 0. and c = ref 0. in
  Array.iteri
    (fun i x ->
      let p = x *. ys.(i) in
      let t = !s +. p in
      let z = t -. !s in
      c := !c +. (!s -. (t -. z)) +. (p -. z) +. Float.fma x ys.(i) (-.p);
      s := t)
    xs;
  !s +. !c

(* [got], an output of [kc] products of parts [xs] and [ys], within
   {!Nx.contract}'s bound of their sum, [mag] their magnitudes' sum: NaN
   where a float sum of them is, that infinity where it is one. A complex
   init's NaN part makes [mag], which holds the init's modulus, a NaN
   beside this part's sum of numbers: the bound is then a NaN and promises
   nothing for this part. An infinite [mag] makes the bound infinite, which
   still rejects a NaN. *)
let within (type v) (f : v floats) kc ~mag xs ys got =
  let naive = ref 0. in
  Array.iteri (fun i x -> naive := !naive +. (x *. ys.(i))) xs;
  let show x = if Float.is_nan x then "nan" else Printf.sprintf "%h" x in
  if not (Float.is_finite !naive) then equal string (show !naive) (show got)
  else if Float.is_nan mag then ()
  else
    let s = dot2 xs ys in
    let g = gamma (2. *. f.u) (kc + 1) *. mag in
    let n = Float.of_int (2 * kc) in
    (* With no product this term is 0, though [mag] may be infinite. *)
    let terms = if n = 0. then 0. else (n *. u64) ** 2. *. mag in
    let reference = (u64 *. Float.abs s) +. terms in
    (* No rounding into the result adds nothing, though [g] is infinite. *)
    let rounding = if f.out_u = 0. then 0. else f.out_u *. (Float.abs s +. g) in
    let bound = g +. rounding +. f.tiny +. reference in
    if Float.abs got = infinity && Float.abs s +. bound >= f.max then
      (* An overflow of the rounding into the result's dtype. *)
      equal string (show (Float.copy_sign infinity s)) (show got)
    else
      at_most ~msg:(Printf.sprintf "%h against %h" got s) float_exact
        ~than:bound
        (Float.abs (got -. s))

(* Each output of [a] [[nb; rows; k]] by [b] [[nb; k; cols]] from [init]
   [[nb; rows; cols]], the elements [xa], [xb] and [xi] in C order, within
   {!Nx.contract}'s bound in [got]: a complex output's parts each over its
   [2k] products of parts. *)
let bounded (type v) (f : v floats) ~complex ~rows ~cols ~k xa xb
    (xi : v array option) (got : v array) =
  Array.iteri
    (fun o y ->
      let bi = o / (rows * cols) and i = o / cols mod rows and j = o mod cols in
      let pa l = f.parts xa.((((bi * rows) + i) * k) + l)
      and pb l = f.parts xb.((((bi * k) + l) * cols) + j) in
      let ir, ii = match xi with Some xi -> f.parts xi.(o) | None -> (0., 0.) in
      let side g e = Array.append (Array.init k g) [| e |] in
      let ar = side (fun l -> fst (pa l)) ir in
      let ai = side (fun l -> snd (pa l)) ii in
      let br = side (fun l -> fst (pb l)) 1. in
      let bi = side (fun l -> snd (pb l)) 0. in
      let mag = ref (Float.hypot ir ii) in
      for l = 0 to k - 1 do
        mag := !mag +. (Float.hypot ar.(l) ai.(l) *. Float.hypot br.(l) bi.(l))
      done;
      (* [x] without init's term, which only its own part takes. *)
      let no_init x = Array.mapi (fun l e -> if l = k then 0. else e) x in
      let yr, yi = f.parts y in
      within f k ~mag:!mag (Array.append ar (no_init ai))
        (Array.append br (Array.map Float.neg bi))
        yr;
      if complex then
        within f k ~mag:!mag (Array.append (no_init ar) ai)
          (Array.append bi br) yi)
    got



(* The case's tiles, its windows along the contracting pairs [ccut] cuts
   that hold products, in C order of their first indices: each a list of
   [(pair, (start, count))]. *)
let ctiles c =
  let cuts =
    match c.ccut with
    | Pair3 p -> [ (p, 3) ]
    | Pair4 p | Pair_mesh p -> [ (p, 4) ]
    | Pair_rows p | Pair_batch p | Batch_rows p -> [ (p, 2) ]
    | Pairs -> [ (0, 2); (1, 2) ]
  in
  let pick acc (p, t) =
    let held = List.filter (fun (_, n) -> n > 0) (windows c.ks.(p) t) in
    List.concat_map (fun l -> List.map (fun w -> l @ [ (p, w) ]) held) acc
  in
  List.fold_left pick [ [] ] cuts

(* [x]'s window [tile] along the pairs, [side] picking its axis of one. *)
let tiled c side tile x =
  let index i =
    match List.find_opt (fun (p, _) -> side (pair_axes c p) = i) tile with
    | Some (_, (s, n)) -> Nx.R (s, s + n)
    | None -> Nx.A
  in
  Nx.slice (List.init (Nx.ndim x) index) x

(* A contraction cut along its contracting pairs is, as partial.mli states,
   each tile contracted on one device into the accumulator from no init,
   the sums stacked in the tiles' order and summed over the stack by the
   kernels' reduction, then [init], rounded once: the same bits, whatever
   the library contracts by. Both libraries reduce as nx.cpu does, so the
   host sums the stack. Where the result is cut too, each device contracts
   its tiles of its own window of the result. [pairs] where cases may have
   no product or two pairs, and the library is nx.cpu. *)
let law_cut ~pairs (module S : Sets) c =
  let (Ckind k) = c.ck in
  let a, b, init, batch, contracting =
    ccase_values k.cdtype k.draw k.floats c
  in
  let cut_pair, tiles =
    match c.ccut with
    | Pair3 p -> (p, 3)
    | Pair4 p | Pair_mesh p -> (p, 4)
    | Pair_rows p | Pair_batch p | Batch_rows p -> (p, 2)
    | Pairs -> (1, 2)
  in
  let kc = c.ks.(cut_pair) in
  let terms = Array.fold_left ( * ) 1 c.ks in
  let ts = ctiles c in
  let products t =
    Array.fold_left ( * ) 1
      (Array.mapi
         (fun p k ->
           match List.assoc_opt p t with Some (_, n) -> n | None -> k)
         c.ks)
  in
  cover ("a contraction of " ^ D.name k.cdtype) true;
  cover "from init" c.init;
  cover "from init on every device beside a cut result"
    (c.init && c.iwhere = Init_everywhere
    &&
    match c.ccut with
    | Pair_rows _ | Pair_batch _ | Batch_rows _ -> true
    | _ -> false);
  if pairs then begin
    (* nx.cpu's chain order (nx_cpu.mli), for a tile of [m] rows. *)
    let m = match c.ccut with Pair_rows _ -> (c.rows + 1) / 2 | _ -> c.rows in
    let n = c.cols in
    let chained d = m > 4 && n > 4 && m * n >= 64 && d < 128 * m * n in
    cover "a tile in chain order"
      (terms > 0 && List.exists (fun t -> chained (products t)) ts);
    cover "no product" (terms = 0);
    cover "a cut on a pair after the first" (cut_pair > 0);
    cover "two pairs cut" (c.ccut = Pairs)
  end
  else
    cover "a tile of several blocks"
      (List.exists (fun t -> products t > 8192) ts);
  cover "an empty window along the cut pair" (kc > 0 && kc < tiles);
  cover "an uneven cut" (kc mod tiles <> 0);
  cover "a tile of one product"
    (terms > 0 && List.exists (fun t -> products t = 1) ts);
  cover "a pair cut over both mesh axes"
    (match c.ccut with Pair_mesh _ -> true | _ -> false);
  cover "a pair cut beside rows"
    (match c.ccut with Pair_rows _ -> true | _ -> false);
  cover "a pair cut beside the batch"
    (match c.ccut with Pair_batch _ -> true | _ -> false);
  cover "a pair cut in b alone beside a's batch and rows"
    (match c.ccut with Batch_rows _ -> true | _ -> false);
  cover_marks k.floats c.cmarks terms;
  let dt = Nx.dtype a in
  let (D.Any ad) = k.acc in
  let bits' = Array.map k.cshow in
  let bits y = bits' (elements y) in
  let one x = Nx.place S.One.on x in
  let expected =
    if terms = 0 then
      bits
        (contracted k.acc dt batch contracting (one a) (one b)
           (if c.init then Some (one init) else None))
    else
      let window a b init =
        let part t =
          contracted k.acc ad batch contracting (tiled c fst t a)
            (tiled c snd t b) None
        in
        let sum = Nx.sum ~axes:[ 0 ] (stack (List.map part ts)) in
        let sum = if c.init then Nx.add sum (Nx.cast ad init) else sum in
        Nx.cast dt sum
      in
      (* The result's windows along its axis [ax] over two devices, [b]
         cut along it too where [both]. *)
      let halves ax ~both =
        let at (s, n) x =
          let index i = if i = ax then Nx.R (s, s + n) else Nx.A in
          Nx.slice (List.init (Nx.ndim x) index) x
        in
        let held =
          List.filter (fun (_, n) -> n > 0) (windows (Nx.dim ax init) 2)
        in
        Nx.concatenate ~axis:ax
          (List.map
             (fun w ->
               window (at w (one a)) (if both then at w (one b) else one b)
                 (at w (one init)))
             held)
      in
      bits
        (match c.ccut with
        | Pair_rows _ ->
            halves (if Option.is_some c.nb then 1 else 0) ~both:false
        | Pair_batch _ -> halves 0 ~both:true
        | _ -> window (one a) (one b) (one init))
  in
  let at (type d) (pa : d Nx.Placement.t) (pb : d Nx.Placement.t)
      (pi : d Nx.Placement.t) =
    elements
      (contracted k.acc dt batch contracting (Nx.place pa a) (Nx.place pb b)
         (if c.init then Some (Nx.place pi init) else None))
  in
  let mesh cuts = Nx.Placement.mesh S.mesh cuts in
  let result cuts =
    match c.iwhere with
    | Init_as_result -> mesh cuts
    | Init_everywhere -> S.Four.on
  in
  let pa p = fst (pair_axes c p) and pb p = snd (pair_axes c p) in
  let actual () =
    match c.ccut with
    | Pair3 p ->
        at (S.Three.split ~axis:(pa p)) (S.Three.split ~axis:(pb p)) S.Three.on
    | Pair4 p ->
        at (S.Four.split ~axis:(pa p)) (S.Four.split ~axis:(pb p)) S.Four.on
    | Pair_mesh p ->
        let both axis = mesh [ (axis, [ "a"; "b" ]) ] in
        at (both (pa p)) (both (pb p)) S.Four.on
    | Pair_rows p ->
        let rows = if Option.is_some c.nb then 1 else 0 in
        at
          (mesh [ (rows, [ "a" ]); (pa p, [ "b" ]) ])
          (mesh [ (pb p, [ "b" ]) ])
          (result [ (rows, [ "a" ]) ])
    | Pair_batch p ->
        at
          (mesh [ (0, [ "a" ]); (pa p, [ "b" ]) ])
          (mesh [ (0, [ "a" ]); (pb p, [ "b" ]) ])
          (result [ (0, [ "a" ]) ])
    | Batch_rows p ->
        (* The minor mesh axis cuts [a]'s rows and [b]'s pair: [b] is read
           at the batch's cut alone, which it holds where its pair has no
           element; placed there, each device contracts its window. *)
        let batch_rows = [ (0, [ "a" ]); (1, [ "b" ]) ] in
        let lies = mesh [ (pb p, [ "b" ]) ] and read = mesh [ (0, [ "a" ]) ] in
        if kc > 0 then
          raises
            (Invalid_argument
               (Format.asprintf
                  "test: operand 1 lies at %a and is read at %a; Nx.place \
                   (Placement.mesh (Mesh.v s [ (\"g0\", 2); (\"g1\", 2) ]) \
                   [ (0, [ \"g0\" ]) ]) y"
                  Nx.Placement.pp lies Nx.Placement.pp read))
            (fun () -> at (mesh batch_rows) lies (result batch_rows));
        at (mesh batch_rows) read (result batch_rows)
    | Pairs ->
        at
          (mesh [ (pa 0, [ "a" ]); (pa 1, [ "b" ]) ])
          (mesh [ (pb 0, [ "a" ]); (pb 1, [ "b" ]) ])
          S.Four.on
  in
  let got = actual () in
  match (c.ccut, k.floats) with
  | Batch_rows _, Some f ->
      (* [b] is read whole along its pair: each device contracts its window
         of the result. *)
      equal (array string) ~msg:"a second run" (bits' got) (bits' (actual ()));
      bounded f ~complex:(D.is D.Complex k.cdtype) ~rows:c.rows ~cols:c.cols
        ~k:terms (elements a) (elements b)
        (if c.init then Some (elements init) else None)
        got
  | _ -> equal (array string) expected (bits' got)

(* An output of no products from a complex init with an infinite real
   part: the bound on its imaginary part, which [mag] makes infinite, is
   infinite, never a NaN. *)
let ccase_examples =
  let complex64 =
    List.find (fun (Ckind k) -> D.name k.cdtype = "complex64") ckinds
  in
  [
    { ck = complex64; nb = Some 1; rows = 1; cols = 1; ks = [| 0 |];
      ccut = Batch_rows 0; init = true; iwhere = Init_as_result; cseed = 0;
      cmarks = [ (false, 0., Inf) ] };
  ]

(* A matrix product through Nx over cut operands: [matmul], [dot] of a
   matrix and a vector, and [einsum]; over three or four devices along the
   contracting axis, over both mesh axes, beside rows on the other one, or
   along the rows alone. *)
type nx_product = Matmul | Dot_vector | Einsum
type nwhere = Over3 | Over4 | Over_mesh | Beside_rows | Rows_alone

type nx_case = {
  nk : ckind;
  product : nx_product;
  nrows : int;
  ncols : int;
  k : int;
  nwhere : nwhere;
  nseed : int;
  nmarks : (bool * float * special) list;
}

let pp_nx_case ppf c =
  let (Ckind k) = c.nk in
  Format.fprintf ppf "%s %s %d x %d by %d, %s, seed %d, specials %a"
    (match c.product with
    | Matmul -> "matmul"
    | Dot_vector -> "dot"
    | Einsum -> "einsum")
    (D.name k.cdtype) c.nrows c.ncols c.k
    (match c.nwhere with
    | Over3 -> "over 3"
    | Over4 -> "over 4"
    | Over_mesh -> "over 2x2"
    | Beside_rows -> "beside rows"
    | Rows_alone -> "along rows")
    c.nseed pp_marks c.nmarks

let nx_case =
  let open Gen in
  with_pp pp_nx_case
    (let* nk = of_list ckinds in
     (* nx.cpu's chain order takes many rows and columns, its lane order few:
        half the cases have many of both. *)
     let* many = bool in
     let* product =
       of_list
         (if many then [ Matmul; Einsum ] else [ Matmul; Dot_vector; Einsum ])
     in
     let* nrows = of_list (if many then [ 9; 12 ] else [ 1; 2; 5; 9 ]) in
     let* ncols = of_list (if many then [ 9; 12 ] else [ 1; 2; 5; 9 ]) in
     let* k = of_list [ 0; 1; 2; 3; 7; 1025; 1500; 2049 ] in
     let* nwhere =
       of_list [ Over3; Over4; Over_mesh; Beside_rows; Rows_alone ]
     in
     let* nseed = int in
     let+ nmarks = marks_of in
     { nk; product; nrows; ncols; k; nwhere; nseed; nmarks })

(* A matrix product over operands cut across devices is within
   {!Nx.contract}'s bound of the exact one and the same on a second run;
   an integer one has its bits on one device. *)
let law_nx_case c =
  let (Ckind k) = c.nk in
  let st = Random.State.make [| c.nseed |] in
  let host side s =
    let n = Array.fold_left ( * ) 1 s in
    Nx.Repr.of_array Nx.Host.v
      (A.of_array k.cdtype (Iarray.of_array s)
         (drawn k.draw k.floats st c.nmarks side n))
  in
  let a = host false [| c.nrows; c.k |] in
  let cols =
    match c.product with Dot_vector -> 1 | Matmul | Einsum -> c.ncols
  in
  let b =
    match c.product with
    | Dot_vector -> host true [| c.k |]
    | Matmul | Einsum -> host true [| c.k; c.ncols |]
  in
  cover ("a product of " ^ D.name k.cdtype) true;
  cover "an empty window" (c.k > 0 && c.k < 3 && c.nwhere <> Rows_alone);
  cover "a cut along the rows alone" (c.nwhere = Rows_alone);
  cover_marks k.floats c.nmarks c.k;
  let product (type d) (a : (_, _, d) Nx.t) (b : (_, _, d) Nx.t) =
    match c.product with
    | Matmul -> Nx.matmul a b
    | Dot_vector -> Nx.dot a b
    | Einsum -> Nx.einsum (Nx.Pattern.v "i j, j k -> i k | j") [ a; b ]
  in
  let mesh cuts = Nx.Placement.mesh mesh4 cuts in
  let at (type d) (pa : d Nx.Placement.t) (pb : d Nx.Placement.t) =
    elements (product (Nx.place pa a) (Nx.place pb b))
  in
  let placed () =
    match c.nwhere with
    | Over3 -> at (Set3.split ~axis:1) (Set3.split ~axis:0)
    | Over4 -> at (Set4.split ~axis:1) (Set4.split ~axis:0)
    | Over_mesh -> at (mesh [ (1, [ "a"; "b" ]) ]) (mesh [ (0, [ "a"; "b" ]) ])
    | Beside_rows ->
        at (mesh [ (0, [ "a" ]); (1, [ "b" ]) ]) (mesh [ (0, [ "b" ]) ])
    | Rows_alone -> at (Set4.split ~axis:0) Set4.on
  in
  let got = placed () in
  let bits = Array.map k.cshow in
  equal (array string) ~msg:"a second run" (bits got) (bits (placed ()));
  match k.floats with
  | None -> equal (array string) (bits (elements (product a b))) (bits got)
  | Some f ->
      bounded f ~complex:(D.is D.Complex k.cdtype) ~rows:c.nrows ~cols ~k:c.k
        (elements a) (elements b) None got

(* A float32 contraction over two pairs, each cut over one axis of a 2 x 2
   mesh, so that each device holds one product. Its four sums, stacked in
   C order of the tiles' first indices, are [1e7; -1e7; 0.1; 0]: nx.cpu
   adds lanes 0 and 2, 1 and 3, then the two, which gives [0.]. Stacked
   with the second pair's tiles major, they would give [0.1]. The pairs'
   order leaves the stack's alone. *)
let test_contract_pair_order () =
  let a =
    Nx.place
      (Nx.Placement.mesh mesh4 [ (0, [ "a" ]); (1, [ "b" ]) ])
      (Nx.create Nx.float32 [| 2; 2 |] [| 1e7; -1e7; 0.1; 0. |])
  in
  let b = Nx.ones_like a in
  let sum axes = (elements (Nx.tensordot ~axes a b)).(0) in
  equal ~msg:"pairs in order" float_exact 0. (sum ([ 0; 1 ], [ 0; 1 ]));
  equal ~msg:"pairs reversed" float_exact 0. (sum ([ 1; 0 ], [ 1; 0 ]))

(* Scans along cut axes *)

type scan = Cumsum | Cumprod | Cummax | Cummin

let scan_name = function
  | Cumsum -> "cumsum"
  | Cumprod -> "cumprod"
  | Cummax -> "cummax"
  | Cummin -> "cummin"

(* Each result's bits as strings, [show] printing an element's, every NaN
   alike, or the message of the refusal. *)
let scan_bits (type v s d) (show : v -> string) r axis (x : (v, s, d) Nx.t) =
  let parts = r = Cumsum in
  match
    match r with
    | Cumsum -> Nx.cumsum ~axis x
    | Cumprod -> Nx.cumprod ~axis x
    | Cummax -> Nx.cummax ~axis x
    | Cummin -> Nx.cummin ~axis x
  with
  | y -> Ok (Array.map (nan_blind ~parts (Nx.dtype x) show) (elements y))
  | exception Invalid_argument m -> Error m

(* {!scan_bits} of [x] placed as [w]. *)
let scan_at (type v s) w (show : v -> string) r axis
    (x : (v, s, Nx.Host.d) Nx.t) =
  match w with
  | Split3 a -> scan_bits show r axis (Nx.place (Set3.split ~axis:a) x)
  | Split4 a -> scan_bits show r axis (Nx.place (Set4.split ~axis:a) x)
  | Cuts (a, b) ->
      let cuts =
        if a = b then [ (a, [ "a"; "b" ]) ] else [ (a, [ "a" ]); (b, [ "b" ]) ]
      in
      scan_bits show r axis (Nx.place (Nx.Placement.mesh mesh4 cuts) x)
  | Minor a ->
      scan_bits show r axis
        (Nx.place (Nx.Placement.mesh mesh4 [ (a, [ "b" ]) ]) x)

type scan_case = {
  skind : any_reduced;
  sshape : int array;
  saxis : int;
  swhere : where;
  sr : scan;
  sseed : int;
  smarks : (float * special) list;
}

let pp_scan_case ppf c =
  let (Reduced k) = c.skind in
  Format.fprintf ppf "%s along %d of %s %a at %a, seed %d, specials %s"
    (scan_name c.sr) c.saxis (D.name k.rdtype) pp_ints c.sshape pp_where
    c.swhere c.sseed
    (String.concat "; "
       (List.map
          (fun (f, s) -> Printf.sprintf "%g:%h" f (special_value s))
          c.smarks))

(* The case's terms: a float's or a complex's as {!lane_data} draws a sum's or
   a product's, another dtype's drawn from its values. *)
let scan_data (type v s) (k : (v, s) reduced) c : v array =
  match D.kind k.rdtype with
  | D.Float | D.Complex ->
      lane_data k.rdtype
        {
          lkind = Lane_kind (k.rdtype, k.rshow);
          lshape = c.sshape;
          laxes = None;
          lwhere = c.swhere;
          lr = (if c.sr = Cumprod then Prod else Sum);
          seed = c.sseed;
          marks = c.smarks;
        }
  | D.Signed | D.Unsigned | D.Boolean ->
      let st = Random.State.make [| c.sseed |] in
      let values = Array.of_list k.rvalues in
      Array.init
        (Array.fold_left ( * ) 1 c.sshape)
        (fun _ -> values.(Random.State.int st (Array.length values)))

let scan_case =
  let open Gen in
  with_pp pp_scan_case
    (let* rank = int_range 1 3 in
     let* swhere = where_of rank in
     let cut = List.map fst (tiles swhere) in
     (* Along a cut axis most often. *)
     let* saxis =
       frequency [ (5, of_list cut); (1, int_range 0 (rank - 1)) ]
     in
     let* many =
       frequency
         [
           (1, of_list [ 0; 1; 2; 3; 5; 7 ]);
           ( 3,
             of_list
               [ 4095; 4096; 4097; 6000; 8192; 9001; 12289; 16384; 20000 ] );
         ]
     in
     let* sshape = array ~size:(constant rank) (of_list [ 1; 2; 3 ]) in
     sshape.(saxis) <- many;
     let* (Reduced k as skind) = of_list reduced_kinds in
     let* sr =
       of_list
         (match D.kind k.rdtype with
         | D.Boolean -> [ Cummax; Cummin ]
         | _ -> [ Cumsum; Cumprod; Cummax; Cummin ])
     in
     let* sseed = int in
     let early = float_range 0. 0.3 and late = float_range 0.6 1. in
     let made a b = [ (a, Inf); (b, Neg_inf) ] in
     let nan_term = map (fun x -> Nan x) (of_list [ nan; -.nan ]) in
     let+ smarks =
       frequency
         [
           (1, constant []);
           ( 2,
             list ~size:(int_range 1 3)
               (pair (float_range 0. 1.) (of_list specials)) );
           ( 1,
             let+ a = early and+ b = early and+ c = late and+ x = nan_term in
             made a b @ [ (c, x) ] );
           ( 1,
             let+ a = late and+ b = late and+ c = early and+ x = nan_term in
             (c, x) :: made a b );
         ]
     in
     { skind; sshape; saxis; swhere; sr; sseed; smarks })

(* Balanced windows' starts along an axis of [e] over [t] tiles. *)
let starts e t =
  let q = e / t and r = e mod t in
  List.init t (fun j -> (j * q) + Stdlib.min j r)

(* The float cumsum or cumprod [r] of [x] along [axis], cut into [t]
   windows there, as partial.mli computes it: each window that holds terms
   scanned alone on the host in the carrier; their totals, each window's
   last result, stacked along a new leading axis and scanned along it; and
   each window but the first combined with the totals' scan at the window
   before it, rounded once to [x]'s dtype. *)
let scan_combined (type v s) r axis t (x : (v, s, Nx.Host.d) Nx.t) : v array =
  let dt = Nx.dtype x in
  let (D.Any c) = Nx_kernel.carrier (D.Any dt) in
  let cumsum = r = Cumsum in
  let scan ~axis p = if cumsum then Nx.cumsum ~axis p else Nx.cumprod ~axis p in
  let combine p q = if cumsum then Nx.add p q else Nx.mul p q in
  let scans =
    List.map (fun p -> scan ~axis (Nx.cast c p)) (tiles_of [ (axis, t) ] x)
  in
  let last p =
    let n = Nx.dim axis p in
    let index a = if a = axis then Nx.R (n - 1, n) else Nx.A in
    Nx.slice (List.init (Nx.ndim p) index) p
  in
  let prefix = scan ~axis:0 (stack (List.map last scans)) in
  let before i =
    let index a = if a = 0 then Nx.I (i - 1) else Nx.A in
    Nx.slice (List.init (Nx.ndim prefix) index) prefix
  in
  let window i p = Nx.cast dt (if i = 0 then p else combine (before i) p) in
  elements (Nx.concatenate ~axis (List.mapi window scans))

(* The float scan [r] of [x] along [axis] placed as [w]: its windows
   scanned from the totals before them, or its bits on one device where
   [w] does not cut [axis] or it has no element. *)
let scan_law (type v s) (show : v -> string) w r axis
    (x : (v, s, Nx.Host.d) Nx.t) =
  let scanned y = elements (if r = Cumsum then Nx.cumsum ~axis y else Nx.cumprod ~axis y) in
  let got = at_where w { over = scanned } x in
  let want =
    match List.assoc_opt axis (tiles w) with
    | Some t when Nx.dim axis x > 0 -> scan_combined r axis t x
    | _ -> scanned x
  in
  equal (array string) (Array.map show want) (Array.map show got)

let law_scans c =
  let (Reduced k) = c.skind in
  let n = c.sshape.(c.saxis) in
  let cut =
    List.filter_map
      (fun (a, t) -> if a = c.saxis then Some t else None)
      (tiles c.swhere)
  in
  let along = cut <> [] in
  let ss = List.concat_map (fun t -> starts n t) cut in
  let chunk = 4096 in
  cover "a scan along a cut axis" along;
  cover ("a scan along a cut axis: " ^ scan_name c.sr) along;
  cover ("a scan along a cut axis of " ^ D.name k.rdtype) along;
  cover "chunks across windows" (along && n > chunk);
  cover "a window starting inside a chunk past the first"
    (List.exists (fun s -> s > chunk && s mod chunk <> 0) ss);
  cover "a window starting at a chunk's start"
    (List.exists (fun s -> s > 0 && s mod chunk = 0) ss);
  cover "windows that share a chunk"
    (List.exists
       (fun s ->
         s mod chunk <> 0
         && List.exists
              (fun s' -> s' < s && s' / chunk = s / chunk && s' mod chunk <> 0)
              ss)
       ss);
  cover "an empty window along the scanned axis"
    (List.exists (fun t -> n > 0 && n < t) cut);
  cover "an uneven cut along the scanned axis"
    (List.exists (fun t -> n mod t <> 0) cut);
  cover "a scanned axis of no term" (along && n = 0);
  cover "a scanned axis cut over two mesh axes"
    (along && match c.swhere with Cuts (a, b) -> a = b | _ -> false);
  cover "a scanned axis cut, the other mesh axis holding copies"
    (along && match c.swhere with Minor _ -> true | _ -> false);
  cover "a scanned axis cut beside another"
    (along && List.length (tiles c.swhere) > 1);
  let floats =
    match D.kind k.rdtype with
    | D.Float | D.Complex -> true
    | D.Signed | D.Unsigned | D.Boolean -> false
  in
  let nans = List.exists (function _, Nan _ -> true | _ -> false) c.smarks in
  let infs =
    List.mem Inf (List.map snd c.smarks)
    && List.mem Neg_inf (List.map snd c.smarks)
  in
  cover "a NaN term along a cut axis" (along && floats && nans);
  cover "infinities of both signs along a cut axis" (along && floats && infs);
  let data = scan_data k c in
  cover "a signalling NaN term along a cut axis"
    (along && signals k.rdtype data);
  let x = signalling k.rdtype c.sshape data in
  if floats && (c.sr = Cumsum || c.sr = Cumprod) then
    scan_law k.rshow c.swhere c.sr c.saxis x
  else
    equal string
      (printed (scan_bits k.rshow c.sr c.saxis x))
      (printed (scan_at c.swhere k.rshow c.sr c.saxis x))

(* A float32 cumsum along an axis cut across devices keeps the bound of a
   sum of each result's terms, as on one device: the result of terms [0]
   to [i] within [γ(i) Σ|x|] of their exact sum, then its rounding. The
   exact sum is the float64 sum compensated by Neumaier's TwoSum, within
   [2 u64] of it relatively, plus [n² u64²] of the magnitudes. *)
let pp_scan_bound ppf (n, w, seed) =
  Format.fprintf ppf "%d terms at %a, seed %d" n pp_where w seed

let scan_bound_case =
  let open Gen in
  with_pp pp_scan_bound
    (let* n = frequency [ (1, int_range 1 16); (2, int_range 4000 9000) ] in
     let* w = of_list [ Split3 0; Split4 0; Cuts (0, 0); Minor 0 ] in
     let+ seed = int in
     (n, w, seed))

let law_scan_bound (n, w, seed) =
  let st = Random.State.make [| seed |] in
  let term _ =
    let m = Random.State.float st 2. -. 1. in
    f32 (Float.ldexp m (Random.State.int st 21 - 10))
  in
  let xs = Array.init n term in
  let x = Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 [| n |] xs) in
  let got = at_where w { over = (fun y -> elements (Nx.cumsum ~axis:0 y)) } x in
  cover "a window past the first chunk" (n > 4096 * 2);
  cover "an empty window" (n < 4);
  let u32 = ldexp 1. (-24) in
  let s = ref 0. and c = ref 0. and mag = ref 0. in
  Array.iteri
    (fun i x ->
      let t = !s +. x in
      c :=
        !c
        +. if Float.abs !s >= Float.abs x then !s -. t +. x else x -. t +. !s;
      s := t;
      mag := !mag +. Float.abs x;
      let exact = !s +. !c in
      let k = Float.of_int (i + 1) in
      let slack =
        (2. *. u64 *. Float.abs exact) +. (k *. k *. u64 *. u64 *. !mag)
      in
      let bound = (gamma u32 i *. !mag) +. (u32 *. Float.abs exact) +. slack in
      let msg = Printf.sprintf "result %d: %h against %h" i got.(i) exact in
      at_most ~msg float_exact ~than:bound (Float.abs (got.(i) -. exact)))
    xs

(* The variance of float32 terms [c + y], [y] in [[-1, 1)] and [|c|] up
   to [10{^6}], over an axis cut across devices: within nx.mli's bound of
   the exact one, which a mean far from [0] keeps relative to the
   variance. *)
let pp_shifted ppf (c, n, w, seed) =
  Format.fprintf ppf "%d terms near %g at %a, seed %d" n c pp_where w seed

let shifted_case =
  let open Gen in
  with_pp pp_shifted
    (let* e = int_range 0 6 in
     let* sign = of_list [ 1.; -1. ] in
     let* n = int_range 2 64 in
     let* w = of_list [ Split3 0; Split4 0; Cuts (0, 0); Minor 0 ] in
     let+ seed = int in
     (sign *. (10. ** Float.of_int e), n, w, seed))

let law_shifted (c, n, w, seed) =
  let st = Random.State.make [| seed |] in
  let xs = Array.init n (fun _ -> f32 (c +. Random.State.float st 2. -. 1.)) in
  let x = Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 [| n |] xs) in
  let got = (at_where w { over = (fun y -> elements (Nx.var y)) } x).(0) in
  cover "a mean of 1e5 or more" (Float.abs c >= 1e5);
  cover "a tile of one term" (n < 8);
  let k = Float.of_int n in
  let mean f = Array.fold_left (fun s x -> s +. f x) 0. xs /. k in
  let mu = mean Fun.id in
  let v = mean (fun x -> (x -. mu) *. (x -. mu)) and m = mean Float.abs in
  let g j = gamma (ldexp 1. (-24)) j in
  let bound =
    ((g (n + 4) +. (2. *. g n *. g n)) *. v)
    +. (2. *. g (n + 6) *. g n *. m *. Float.sqrt v)
    +. (4. *. g (n + 4) *. g n *. g n *. m *. m)
  in
  (* The float64 reference's own error. *)
  let slack = ((k +. 4.) *. u64 *. v) +. ((k *. u64 *. m) ** 2.) in
  let msg = Printf.sprintf "%h against %h" got v in
  at_most ~msg float_exact ~than:(bound +. slack) (Float.abs (got -. v))

(* Traffic *)

(* nx.cpu, counting its loop positions and products, over three and four memory
   devices. *)
module Count3 = (val Nx.devices ~kernels:(module C) [ m 0; m 1; m 2 ])
module Count4 = (val Nx.devices ~kernels:(module C) [ m 0; m 1; m 2; m 3 ])

(* What a case counts. *)
type traffic = Positions | Products

(* A fold over an axis cut across devices, and the most it may compute, below
   what reading its operands whole on each device would: [setup ()] places the
   operands, the thunk it gives computes the fold, and the one that gives reads
   the result. *)
type moved = {
  name : string;
  traffic : traffic;
  under : int;
  setup : unit -> unit -> unit -> unit;
}

(* [f] over [x] at [p]. *)
let folded name ~under p x f =
  let setup () =
    let x = Nx.place p (Nx.Repr.of_array Nx.Host.v x) in
    fun () ->
      let y = f x in
      fun () -> ignore (elements y)
  in
  { name; traffic = Positions; under; setup }

let ramp dt n =
  A.of_array dt [| n |] (Array.init n (fun i -> Float.of_int (i mod 7) /. 7.))

let moved =
  let on3 axis = Count3.split ~axis and on4 = Count4.split ~axis:0 in
  (* 240 terms over three devices, 4 outputs: each device reduces its window of
     the terms, then the three partials per output it receives. Reading the
     operand whole, each would reduce all 240 terms. *)
  let ints s = A.of_array D.Int32 s (Array.init 240 Int32.of_int) in
  let floats s = A.of_array D.Float32 s (Array.init 240 Float.of_int) in
  let rows : int iarray = [| 60; 4 |] and columns : int iarray = [| 4; 60 |] in
  let small =
    [
      folded "an integer sum over three devices" ~under:480 (on3 0) (ints rows)
        (fun x -> Nx.sum ~axes:[ 0 ] x);
      folded "a maximum over three devices" ~under:480 (on3 0) (floats rows)
        (fun x -> Nx.max ~axes:[ 0 ] x);
      folded "positions over three devices" ~under:480 (on3 1) (floats columns)
        (fun x -> Nx.argmax ~axis:1 x);
      folded "a reduction of every axis over three devices" ~under:480 (on3 1)
        (ints columns) (fun x -> Nx.min x);
    ]
  in
  (* Read whole, each of four devices would compute every term: [4 n] positions.
     Cut, each reduces its window once, two passes for a logsumexp or a
     variance, then the four partials per output it receives. A complex sum
     reduces each part as a float. *)
  let complexes n =
    A.of_array D.Complex64 [| n |]
      (Array.init n (fun i -> { Complex.re = Float.of_int (i mod 5); im = 1. }))
  in
  let reductions n =
    let of_n what = Printf.sprintf "%s of %d terms" what n in
    [
      folded (of_n "a sum") ~under:(2 * n) on4 (ramp D.Float32 n) Nx.sum;
      folded (of_n "a product") ~under:(2 * n) on4 (ramp D.Float64 n) Nx.prod;
      folded (of_n "a float16 mean") ~under:(2 * n) on4 (ramp D.Float16 n)
        Nx.mean;
      folded (of_n "a complex sum") ~under:(3 * n) on4 (complexes n) Nx.sum;
      folded (of_n "a logsumexp") ~under:(4 * n) on4 (ramp D.Float32 n)
        Nx.logsumexp;
    ]
  in
  let variance =
    folded "a variance of 8192 terms" ~under:(4 * 8192) on4
      (ramp D.Float32 8192) Nx.var
  in
  (* Read whole, each of four devices would contract all of the [k] terms of
     each output: [4 m n k] products. Cut, each contracts its own window of
     them, and only sums move. *)
  let contraction (m, n, k) =
    let floats s =
      A.of_array D.Float32 (Iarray.of_array s)
        (Array.init (Array.fold_left ( * ) 1 s) (fun i ->
             Float.of_int (i mod 7) /. 7.))
    in
    let setup () =
      let place axis x =
        Nx.place (Count4.split ~axis) (Nx.Repr.of_array Nx.Host.v x)
      in
      let a = place 1 (floats [| m; k |]) and b = place 0 (floats [| k; n |]) in
      fun () ->
        let y = Nx.matmul a b in
        fun () -> ignore (elements y)
    in
    let name = Printf.sprintf "a matrix product of %d x %d by %d" m n k in
    { name; traffic = Products; under = 2 * m * n * k; setup }
  in
  (* Read whole, each of four devices would scan every term: [4 n] positions.
     Cut, each device scans its window, then each but the first combines its
     scan with the totals' scan before it, a map of its window. *)
  let scans n =
    let of_n what = Printf.sprintf "%s of %d terms" what n in
    let ints = A.of_array D.Int32 [| n |] (Array.init n Int32.of_int) in
    [
      folded (of_n "an integer cumsum") ~under:(2 * n) on4 ints
        (Nx.cumsum ~axis:0);
      folded (of_n "a float cummax") ~under:(2 * n) on4 (ramp D.Float32 n)
        (Nx.cummax ~axis:0);
      folded (of_n "a float cumsum") ~under:(3 * n) on4 (ramp D.Float32 n)
        (Nx.cumsum ~axis:0);
      folded (of_n "a float16 cumprod") ~under:(4 * n) on4 (ramp D.Float16 n)
        (Nx.cumprod ~axis:0);
    ]
  in
  List.concat
    [
      small;
      reductions 8192;
      reductions 6000;
      [ variance ];
      List.map contraction
        [ (8, 8, 4096); (2, 2, 4096); (2, 2, 6000); (8, 8, 6000) ];
      scans 16384;
      scans 9001;
    ]

let law_moved c =
  let compute = c.setup () in
  C.reset ();
  let read = compute () in
  let got =
    match c.traffic with
    | Positions -> C.positions ()
    | Products -> C.products ()
  in
  read ();
  less int ~than:c.under got

(* The bytes [d] holds after four rounds of collection, each of which
   returns to the device what it collected, and its cache freed. *)
let settled d =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create d 0);
    Rig.free_cache d
  done;
  Rig.allocated d

(* The bytes each of the four memory devices holds more after [f ()], read
   settled before it. A value read by a device that lacks its window is
   gathered there into memory of the device's own. *)
let grown f =
  let devices = List.init 4 m in
  let before = List.map settled devices in
  let y = f () in
  (y, List.map2 (fun d b -> Rig.allocated d - b) devices before)

(* A scatter into a target split over four devices, of positions and
   updates whole on each, writes each tile where it lies: each device holds
   its tile of the results and the positions it shifts, under two tiles.
   Read whole, each device would gather the four tiles, as placing the
   target whole on each device does. *)
let test_scatter_in_place () =
  let n = 1 lsl 16 in
  let tile = 4 * (n / 4) in
  let split = Set4.split ~axis:0 in
  let target () = Nx.place split (Nx.zeros D.Float32 [| n |]) in
  let positions =
    Nx.place Set4.on
      (Nx.Repr.of_array Nx.Host.v
         (A.of_array D.Int64 [| 4 |] [| 0L; 17000L; 40000L; 65535L |]))
  in
  let updates = Nx.place Set4.on (Nx.ones D.Float32 [| 4 |]) in
  let scatter x = Nx.scatter ~axis:0 positions updates x in
  ignore (elements (scatter (target ())));
  let x = target () in
  let y, grew = grown (fun () -> scatter x) in
  equal (placement ()) split (Nx.placement y);
  List.iteri
    (fun k g -> less ~msg:(Printf.sprintf "m%d" k) int ~than:(2 * tile) g)
    grew;
  let _, whole = grown (fun () -> Nx.place Set4.on x) in
  List.iteri
    (fun k g ->
      at_least ~msg:(Printf.sprintf "m%d, the target whole" k) int ~than:(4 * tile) g)
    whole

(* A take of positions split over four devices, beside a value split along
   another axis, moves the positions alone: each device holds a copy of all
   of them and its tile of the results. Placing the value whole would add
   three of its tiles on each device. *)
let test_take_moves_positions () =
  let n = 4096 and columns = 64 in
  let split axis = Set4.split ~axis in
  let value () = Nx.place (split 1) (Nx.zeros D.Float32 [| n; columns |]) in
  let p = Nx.place (split 0) (Nx.arange D.Int64 0 n 1) in
  let take x = Nx.take ~axis:0 p x in
  ignore (elements (take (value ())));
  let x = value () in
  let y, grew = grown (fun () -> take x) in
  equal (placement ()) (split 1) (Nx.placement y);
  let positions = 8 * n and tile = 4 * n * (columns / 4) in
  List.iteri
    (fun k g -> equal ~msg:(Printf.sprintf "m%d" k) int (positions + tile) g)
    grew

(* An [Add] of many updates split over four devices into a small target
   whole on each moves partial sums of the target's size, never the
   updates: gathering them would add three quarters of them on each
   device. *)
let test_summed_moves_sums () =
  let n = 64 and m = 1 lsl 16 in
  let split = Set4.split ~axis:0 in
  let positions =
    Nx.place split
      (Nx.Repr.of_array Nx.Host.v
         (A.of_array D.Int64 [| m |]
            (Array.init m (fun i -> Int64.of_int (i mod n)))))
  in
  let updates = Nx.place split (Nx.ones D.Float32 [| m |]) in
  let target () = Nx.place Set4.on (Nx.zeros D.Float32 [| n |]) in
  let add x = Nx.scatter ~combine:Add ~axis:0 positions updates x in
  ignore (elements (add (target ())));
  let x = target () in
  let y, grew = grown (fun () -> add x) in
  equal (placement ()) Set4.on (Nx.placement y);
  equal (array float_exact) (Array.make n (Float.of_int (m / n))) (elements y);
  List.iteri
    (fun k g ->
      less ~msg:(Printf.sprintf "m%d" k) int ~than:(3 * 4 * m / 4) g)
    grew

let traffic =
  group "traffic"
    [
      cases
        ~name:(fun c -> c.name)
        "a fold over a cut axis moves partials, not its operands" moved
        law_moved;
      test "a scatter into a cut target writes each tile where it lies"
        test_scatter_in_place;
      test "a take of positions no order reads moves the positions alone"
        test_take_moves_positions;
      test "an addition of updates split over devices moves sums alone"
        test_summed_moves_sums;
    ]

let placed =
  group "placements"
    [
      prop ~count:1000
        "an operation over a value at any placement gives its bits on one \
         device"
        placed_case law_placed;
      prop ~count:500
        "an assembly of values cut along axes it leaves whole gives their \
         bits, cut along those axes over the same devices"
        assembled_case law_assembled;
      test "a concatenation along a cut axis reads it whole" test_concat_cut;
      prop ~count:1000
        "a gather of a value at any placement, at positions whole on each of \
         its devices, cut along axes its gathered cut leaves, or split, \
         gives its bits on one device"
        gathered_case law_gathered;
      test "take of a flattening cut along its leading axis" test_take_flat;
      test "a gather of positions cut as the gathered axis moves them"
        test_gather_both_cut;
      test "positions no order reads move whole to every device"
        test_positions_moved;
      test "positions cut finer than their value are read where they lie"
        test_positions_finer;
      cases ~name:ranked_name
        "a gather past the greatest rank reads its axis whole" ranked
        test_gather_rank;
      prop ~count:1000
        "a scatter into a target at any placement, of positions and updates \
         whole on each of its devices or cut alike off the scattered axis, \
         gives its bits on one device"
        scattered_case law_scattered;
      prop ~count:1000
        "a scatter's addition of updates split along its axis sums them into \
         the target where it lies, as on one device"
        summed_case law_summed;
      test "a scatter of cut updates raises, naming the fix; cut positions \
            move"
        test_scatter_cut_updates;
    ]

let reductions =
  group "reductions"
    [
      prop ~count:300
        "a float32 variance over a cut axis is within its bound however far \
         its mean from 0"
        shifted_case law_shifted;
      prop ~count:2000
        "a reduction over cut axes gives an exact one's bits on one device, \
         a float one the reduction of its tiles' stacked partials"
        reduced_case law_reduced;
      test "a position over cut axes is the smallest of a tie across tiles"
        test_arg_ties;
      test
        "a sum over an axis cut across devices allocates with the devices, \
         not the elements"
        test_closing_words;
      test "a float maximum or minimum over a cut axis is a NaN beside one"
        test_cut_extreme_nan;
      prop ~count:300
        "a float reduction over cut axes of many terms reduces its tiles' \
         stacked partials"
        lane_case law_lanes;
    ]

let contractions =
  group "contractions"
    [
      prop ~count:400 ~examples:ccase_examples
        "a contraction over cut contracting axes sums its tiles' stacked \
         sums, then init"
        ccase
        (law_cut ~pairs:true (module Cpu_sets));
      prop ~count:300
        "a contraction cut over its contracted extent sums its tiles' \
         stacked sums, for a library that sums blocks its own way"
        bcase
        (law_cut ~pairs:false (module Blocked_sets));
      prop ~count:200
        "a matrix product over cut operands is within its bound and the \
         same on every run"
        nx_case law_nx_case;
      test "a contraction cut along two pairs stacks its tiles in C order"
        test_contract_pair_order;
    ]

let scans =
  group "scans"
    [
      prop ~count:500
        "a scan along cut axes gives an exact one's bits on one device, a \
         float sum's or product's each window's scan from its totals' scan"
        scan_case law_scans;
      prop ~count:200
        "a float32 cumsum along a cut axis is within the bound of a sum of \
         each result's terms"
        scan_bound_case law_scan_bound;
    ]

let () =
  exit (run "nx partial" [ placed; reductions; contractions; scans; traffic ])
