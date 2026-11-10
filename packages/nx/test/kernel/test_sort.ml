(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sorts through every kernel library the host runs: the values are the operand
   at the positions, in order, and keeping k gives the first k of the whole
   sort. nx.cpu's group checks the order Nx_kernel.S.sort states against a
   stable sort built here, at every dtype, through views, with NaN
   payloads, -0 and ties. *)

open Windtrap
open Elements
module Support = Nx_kernels_support

(* How an operand's elements are drawn: from a few values, so that ties are
   common; within a narrow span of keys, a base element's high bytes over its
   given number of random low ones; each at random; or from one value three
   times in four, else at random, so that a long slice's top digit leaves one
   bucket of most elements; a float's specials three times in four, both
   zeros, infinities, NaNs of both signs and several payloads, subnormals and
   extremes, else at random; or, of an element of 2 to 8 bytes, a base
   element with one byte at random, the byte changing every [stripe]
   elements, so that the slices of one call vary in different bytes. *)
type draw = Tied | Narrow of int | Spread | Skewed | Special | Stripes

type case = {
  axis : int;
  descending : bool;
  k : int option;
  x : A.any;
  draw : draw;
  views : string list;
}

let pp_case ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf "sort%s along %d%s of %a %a, %s (%s)"
    (if c.descending then " descending" else "")
    c.axis
    (match c.k with None -> "" | Some k -> Printf.sprintf " keeping %d" k)
    D.pp (A.dtype x) pp_ints (shape_of c.x)
    (match c.draw with
    | Tied -> "tied"
    | Narrow low -> Printf.sprintf "narrow over %d bytes" low
    | Spread -> "spread"
    | Skewed -> "skewed"
    | Special -> "specials"
    | Stripes -> "striped")
    (String.concat ", " c.views)

(* Shapes: small ones of any rank, slices at the edges of the insertion sort's,
   and long ones that sort by radix or keep a few. *)
let shape_gen =
  let open Gen in
  let small = array ~size:(int_range 1 4) (int_range 0 6) in
  let edge =
    map (fun n -> [| 2; n |]) (of_list ~pp:Format.pp_print_int [ 63; 64; 65 ])
  in
  let long =
    let* n = int_range 100 3000 in
    let+ before = array ~size:(int_range 0 1) (int_range 1 3) in
    Array.append before [| n |]
  in
  frequency [ (5, small); (1, edge); (2, long) ]

(* A special float of [bits] bits, [eb] of them its exponent's, as its bits:
   a zero, the top exponent alone, with its least or every fraction bit, or
   half of them, the least subnormal, the greatest finite or one below the top
   exponent, one and its successor, each of either sign. *)
let special_bits rs ~bits ~eb =
  let open Int64 in
  let mb = bits - 1 - eb in
  let top = shift_left (pred (shift_left 1L eb)) mb in
  let frac = pred (shift_left 1L mb) in
  let one = shift_left (pred (shift_left 1L (eb - 1))) mb in
  let pool =
    [| 0L; top; logor top 1L; logor top frac; 1L; pred top; one; succ one;
       logor top (shift_right_logical frac 1) |]
  in
  let x = pool.(Random.State.int rs (Array.length pool)) in
  if Random.State.bool rs then logor x (shift_left 1L (bits - 1)) else x

(* [x]'s low [n] bytes, a sub-byte code in one. *)
let le n x =
  String.init n (fun j ->
      Char.chr Int64.(to_int (logand (shift_right_logical x (8 * j)) 0xFFL)))

(* A special element of [d], or a random one where [d] has no float. *)
let special (D.Any dt as d) rs =
  let f bits eb = le (Int.max 1 (bits / 8)) (special_bits rs ~bits ~eb) in
  match dt with
  | D.Float64 -> f 64 11
  | D.Float32 -> f 32 8
  | D.Float16 -> f 16 5
  | D.Bfloat16 -> f 16 8
  | D.Float8_e4m3fn -> f 8 4
  | D.Float8_e5m2 -> f 8 5
  | D.Float4_e2m1fn -> f 4 2
  | D.Complex64 -> f 32 8 ^ f 32 8
  | D.Complex128 -> f 64 11 ^ f 64 11
  | _ -> element d rs

(* The elements a striped draw keeps one random byte for: the length of the
   slices that vary in one byte each. *)
let stripe = 4100

(* An operand of [d] drawn from [seed] as [draw] says. A narrow element keeps
   the base's high bytes, so that integers lie within a few low bytes of each
   other and floats within a few low bytes of their significands. *)
let drawn draw d s v seed =
  let rs = Random.State.make [| seed |] in
  let pool = Array.init 6 (fun _ -> element d rs) in
  let base = element d rs in
  operand d s v (fun k ->
      match draw with
      | Special ->
          if Random.State.int rs 4 = 0 then element d rs else special d rs
      | Stripes ->
          let w = String.length base in
          let at = k / stripe mod w in
          if w < 2 || w > 8 then element d rs
          else
            String.mapi
              (fun j c ->
                if j = at then Char.chr (Random.State.int rs 256) else c)
              base
      | Tied ->
          if Random.State.int rs 3 = 0 then element d rs
          else pool.(Random.State.int rs 6)
      | Skewed -> if Random.State.int rs 4 = 0 then element d rs else base
      | Narrow low ->
          String.mapi
            (fun j c ->
              if j < low then Char.chr (Random.State.int rs 256) else c)
            base
      | Spread ->
          String.init (String.length base) (fun _ ->
              Char.chr (Random.State.int rs 256)))

let case_gen ?(dtypes = D.all) ?(shapes = shape_gen) () =
  let open Gen in
  let* d = of_list ~pp:pp_dtype dtypes in
  let* s = shapes in
  let r = Array.length s in
  let* axis = int_range 0 (r - 1) in
  let* descending = bool in
  let* k =
    frequency
      [
        (2, constant None);
        (1, map Option.some (int_range 0 (min s.(axis) 4)));
        (1, map Option.some (int_range 0 s.(axis)));
      ]
  in
  let* v = view_of r in
  let (D.Any dt) = d in
  let* draw =
    frequency
      [
        (4, constant Tied);
        (1, map (fun low -> Narrow low) (int_range 1 (max 1 (D.bits dt / 8))));
        (1, constant Spread);
        (1, constant Special);
      ]
  in
  let+ seed = int in
  { axis; descending; k; x = drawn draw d s v seed; draw; views = view_names v }

(* A constructed case along the last axis, drawn from seed 1, so that every seed
   reaches its regime. *)
let example ?k ?(descending = false) ?(draw = Tied) ?view d s =
  let r = Array.length s in
  let v = Option.value view ~default:(plain r) in
  {
    axis = r - 1;
    descending;
    k;
    x = drawn draw d s v 1;
    draw;
    views = view_names v;
  }

let kept c = match c.k with None -> (shape_of c.x).(c.axis) | Some k -> k

let sort_on (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (A.Any x) = c.x in
  let y = Array.copy (shape_of c.x) in
  y.(c.axis) <- kept c;
  let y = Iarray.of_array y in
  let x = Support.on b x in
  let values = A.create b.device (A.dtype x) y in
  let positions = A.create b.device D.Int64 y in
  (* Destinations start as 0xA5 bytes, not zeros: an element the kernels do
     not store differs from the reference unless the reference holds those
     bytes there. *)
  Rig.Buffer.fill (A.buffer values) '\xA5';
  Rig.Buffer.fill (A.buffer positions) '\xA5';
  match K.sort ~axis:c.axis ~descending:c.descending ~values ~positions x with
  | A.Done -> Some (A.Any values, A.Any positions)
  | A.Declined -> failf "%s declined a sort" b.name
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Each slice's elements with their positions, stably sorted, the first [kept c]
   of them: the values' bits and the positions, in C order of the results. *)
let expected c =
  let d = dtype_of c.x and s = shape_of c.x in
  let es = elements c.x in
  let n = s.(c.axis) and keep = kept c in
  let y = Array.copy s in
  y.(c.axis) <- keep;
  let values = Array.make (total y) "" and ps = Array.make (total y) 0L in
  let cmp (a, _) (b, _) =
    let o = compare_elements d a b in
    if c.descending then -o else o
  in
  Array.iteri
    (fun k _ ->
      let i = index_of y k in
      if i.(c.axis) = 0 then begin
        let slice =
          List.init n (fun j ->
              let ij = Array.copy i in
              ij.(c.axis) <- j;
              (es.(flat s ij), j))
        in
        List.iteri
          (fun j (e, p) ->
            if j < keep then begin
              let ij = Array.copy i in
              ij.(c.axis) <- j;
              values.(flat y ij) <- e;
              ps.(flat y ij) <- Int64.of_int p
            end)
          (List.stable_sort cmp slice)
      end)
    values;
  (values, ps)

let agrees (b : Support.backend) c =
  match sort_on b c with
  | None -> ()
  | Some (v, p) ->
      let ev, ep = expected c in
      equal ~msg:"positions" (array int64) ep (positions p);
      equal ~msg:"values" (array string) ev (elements v)

let law_reference (b : Support.backend) c =
  let n = (shape_of c.x).(c.axis) in
  cover "descending" c.descending;
  cover "keeps some" (c.k <> None && kept c < n);
  cover "radix" (n >= 64 && c.k = None);
  cover "complex128's two keys" (dtype_of c.x = D.Any D.Complex128 && n >= 2);
  cover "sub-byte"
    (let (D.Any d) = dtype_of c.x in
     D.bits d < 8 && n >= 2);
  cover "narrow 8-byte keys"
    (let (D.Any d) = dtype_of c.x in
     (match c.draw with Narrow _ -> true | _ -> false)
     && n >= 64
     && D.bits d = 64);
  agrees b c

(* One slice sorts on the job's threads where it is long and keeps more than a
   few. *)
let law_large (b : Support.backend) c =
  let long =
    Array.length (shape_of c.x) = 1
    && kept c > 64
    && dtype_of c.x <> D.Any D.Complex128
  in
  let wide = long && (let (D.Any d) = dtype_of c.x in D.bits d = 64) in
  cover "one long slice" long;
  cover "long wide keys spread" (wide && c.draw = Spread);
  cover "long wide keys skewed" (wide && c.draw = Skewed);
  cover "long wide keys, keeping all" (wide && c.k = None);
  cover "sub-byte slices on the job's threads"
    (let (D.Any d) = dtype_of c.x in
     D.bits d < 8 && total (shape_of c.x) >= 100_000);
  agrees b c

(* Long slices: one or a few of 4097 to 40,000 elements, along the last axis
   or the first, keeping all, at most 1024 or more. *)
let long_gen =
  let open Gen in
  let* d = of_list ~pp:pp_dtype D.all in
  let (D.Any dt) = d in
  let* n =
    frequency [ (3, int_range 4097 12_000); (1, int_range 12_001 40_000) ]
  in
  let* outer = int_range 1 3 in
  let* last = bool in
  let s =
    if outer = 1 then [| n |]
    else if last then [| outer; n |]
    else [| n; outer |]
  in
  let axis = if outer > 1 && not last then 0 else Array.length s - 1 in
  let* k =
    frequency
      [
        (2, constant None);
        (2, map Option.some (int_range 1 1024));
        (1, map Option.some (int_range 1025 n));
      ]
  in
  let* descending = bool in
  let* v = view_of (Array.length s) in
  let* draw =
    frequency
      [
        (2, constant Tied);
        (1, map (fun low -> Narrow low) (int_range 1 (max 1 (D.bits dt / 8))));
        (1, constant Spread);
        (1, constant Skewed);
        (2, constant Special);
      ]
  in
  let+ seed = int in
  { axis; descending; k; x = drawn draw d s v seed; draw; views = view_names v }

let long = Gen.with_pp pp_case long_gen

(* Long slices of each width of key, those of 16 bytes past 2048 elements;
   floats' specials; sub-byte slices that share bytes; slices of one call
   that vary in different bytes, keeping all or some; 8-byte keys of a narrow
   span, keeping a few; through an outer axis, reversed and stepped. *)
let long_examples =
  [
    example (D.Any D.Float32) [| 20_000 |] ~draw:Special;
    example (D.Any D.Float64) [| 9000 |] ~draw:Special ~descending:true;
    example (D.Any D.Float16) [| 6000 |] ~draw:Special ~k:700;
    example (D.Any D.Bfloat16) [| 2; 5000 |] ~draw:Special;
    example (D.Any D.Float8_e4m3fn) [| 5000 |] ~draw:Special ~descending:true;
    example (D.Any D.Complex128) [| 2; 3000 |] ~draw:Special;
    example (D.Any D.Complex128) [| 9000 |] ~k:100 ~descending:true;
    example (D.Any D.Complex64) [| 9000 |] ~draw:Special ~k:5000;
    example (D.Any D.Int64) [| 2; 9000 |] ~draw:(Narrow 3) ~k:1000;
    example (D.Any D.Uint64) [| 30_000 |] ~draw:Spread ~descending:true;
    example (D.Any D.Int4) [| 3; 5001 |] ~k:4000;
    example (D.Any D.Bit) [| 3; 5001 |] ~descending:true;
    example (D.Any D.Int32) [| 9; stripe |] ~draw:Stripes;
    example (D.Any D.Float64) [| 9; stripe |] ~draw:Stripes ~descending:true
      ~k:2000;
    example (D.Any D.Uint16) [| 9; stripe |] ~draw:Stripes ~k:2000;
    {
      (example (D.Any D.Int16) [| 9000; 3 |] ~draw:Tied
         ~view:{ (plain 2) with reversed = Some 1; stepped = true })
      with
      axis = 0;
      k = Some 50;
    };
  ]

let law_long (b : Support.backend) c =
  let (D.Any d) = dtype_of c.x in
  let s = shape_of c.x in
  let n = s.(c.axis) in
  cover "keeps all" (c.k = None);
  cover "keeps at most 1024" (kept c <= 1024);
  cover "keeps more than 1024, not all" (kept c > 1024 && kept c < n);
  cover "16-byte keys past 2048" (D.bits d = 128 && n > 2048);
  cover "8-byte keys" (D.bits d = 64);
  cover "sub-byte" (D.bits d < 8);
  cover "floats' specials" (c.draw = Special);
  cover "slices that vary in different bytes" (c.draw = Stripes);
  cover "several slices" (total s / n > 1);
  cover "along an outer axis" (c.axis < Array.length s - 1);
  agrees b c

(* Arrays at their least alignment: a fresh array of [dt] and shape [s] on
   [b]'s device whose first byte lies [align] bytes past its buffer's, the
   least alignment nx_array.mli promises a complex number (one component's
   width), holding [x]'s elements where given. *)
let shifted (b : Support.backend) dt s ~align x =
  let bytes = D.bytes dt (total s) in
  let buffer = Rig.Buffer.create b.device (bytes + align) in
  let view = Rig.Buffer.view buffer ~first:align ~length:bytes in
  let y = A.v dt (L.contiguous (Iarray.of_array s)) view in
  Option.iter (fun x -> A.transfer ~src:x ~dst:y) x;
  y

(* Complex numbers of both widths at a component's alignment, by each path:
   short slices, long ones keeping all and a few. *)
let aligned_cases =
  List.concat_map
    (fun d ->
      [
        example d [| 2; 70 |] ~draw:Special;
        example d [| 9000 |] ~draw:Special ~descending:true;
        example d [| 2; 9000 |] ~k:10;
      ])
    [ D.Any D.Complex64; D.Any D.Complex128 ]

(* The operand and both destinations at their least alignment give the
   reference's values and positions. *)
let law_aligned (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (A.Any x) = c.x in
  let dt = A.dtype x and s = shape_of c.x in
  let align = D.bits dt / 16 in
  let y = Array.copy s in
  y.(c.axis) <- kept c;
  let x = shifted b dt s ~align (Some x) in
  let values = shifted b dt y ~align None in
  let positions = A.create b.device D.Int64 (Iarray.of_array y) in
  match K.sort ~axis:c.axis ~descending:c.descending ~values ~positions x with
  | A.Done ->
      let ev, ep = expected c in
      equal ~msg:"positions" (array int64) ep
        (Elements.positions (A.Any positions));
      equal ~msg:"values" (array string) ev (elements (A.Any values))
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Laws for every library *)

(* The values are the operand's elements at the positions along the axis. *)
let law_values_at_positions (b : Support.backend) c =
  match sort_on b c with
  | None -> ()
  | Some (v, p) ->
      let s = shape_of c.x and es = elements c.x and ps = positions p in
      let y = shape_of v in
      let at =
        Array.mapi
          (fun k q ->
            let i = index_of y k in
            i.(c.axis) <- Int64.to_int q;
            es.(flat s i))
          ps
      in
      equal (array string) at (elements v)

(* Keeping k gives the first k of the whole sort. *)
let law_prefix (b : Support.backend) c =
  match (sort_on b { c with k = None }, sort_on b c) with
  | Some (fv, fp), Some (v, p) ->
      let y = shape_of v and s = shape_of fv in
      let prefix a =
        Array.init (total y) (fun k ->
            let i = index_of y k in
            a.(flat s i))
      in
      equal ~msg:"positions" (array int64) (prefix (positions fp)) (positions p);
      equal ~msg:"values" (array string) (prefix (elements fv)) (elements v)
  | None, None -> ()
  | _ -> failf "%s declined one of the two" b.name

(* Refusals *)

(* A sort's shapes, [x], [values] and [positions], along [along], each near
   a fit: [values] keeps up to one past [x]'s extent. *)
type misfit = { along : int; shapes : int array array }

let misfits =
  Gen.with_pp
    (fun ppf c ->
      Format.fprintf ppf "sort along %d of %a into %a and %a" c.along pp_ints
        c.shapes.(0) pp_ints c.shapes.(1) pp_ints c.shapes.(2))
    (let open Gen in
     let* x = Nx_array_gen.shape in
     let* axis = axis_near x in
     let d = if axis >= 0 && axis < Array.length x then x.(axis) else 0 in
     let* k = int_range 0 (d + 1) in
     let* values = near (along x axis k) in
     let+ positions = near values in
     { along = axis; shapes = [| x; values; positions |] })

(* The kernel keeps [values]' extent along the axis: Nx_kernel.Shape.sort's
   [k]. *)
let law_refuses (b : Support.backend) c =
  let module K = (val b.kernels) in
  let x = c.shapes.(0) and values = c.shapes.(1) and positions = c.shapes.(2) in
  let k =
    if c.along >= 0 && c.along < Array.length values then
      Some values.(c.along)
    else None
  in
  let fit =
    Nx_kernel.Shape.sort ~axis:c.along ~k x = Ok values && positions = values
  in
  cover "fits" fit;
  cover "misfits" (not fit);
  let x =
    Support.on b
      (A.of_array D.Float32 (Iarray.of_array x) (Array.make (total x) 0.))
  in
  Misfit.law ~msg:"a sort" ~what:"a shape" b ~fit
    [| (D.Any D.Float32, values); (D.Any D.Int64, positions) |]
    (fun g ds ->
      K.sort ~axis:c.along ~descending:false
        ~values:(A.expect D.Float32 ds.(0))
        ~positions:(A.expect D.Int64 ds.(1))
        (g.Misfit.g x))

(* Guard pages *)

(* Slices of one element to a few vectors past the widest, at the edges of
   the insertion sort's, and long enough to sort by radix. *)
let guard_shapes =
  let open Gen in
  let* n =
    frequency
      [
        (6, int_range 1 72);
        (1, of_list ~pp:Format.pp_print_int [ 63; 64; 65; 127; 128; 129 ]);
        (2, int_range 100 3000);
      ]
  in
  let+ outer = array ~size:(int_range 0 2) (int_range 1 4) in
  Array.append outer [| n |]

(* The same values and positions with the operand, through its view, and
   both destinations against an inaccessible page. *)
let law_guards (b : Support.backend) (c, side) =
  let module K = (val b.kernels) in
  let after = side = Support.After in
  let s = shape_of c.x in
  cover "along the innermost axis" (c.axis = Array.length s - 1);
  cover "along an outer axis" (c.axis < Array.length s - 1);
  cover "by radix" (s.(c.axis) >= 100);
  cover "keeping some" (kept c < s.(c.axis));
  cover "sub-byte" (let (D.Any dt) = dtype_of c.x in D.bits dt < 8);
  cover "a stepped operand at the page before"
    ((not after) && List.mem "stepped" c.views);
  cover "a reversed operand at the page after"
    (after && List.mem "reversed" c.views);
  let y = Array.copy s in
  y.(c.axis) <- kept c;
  let call xs ds =
    let (A.Any x) = xs.(0) in
    let values = A.expect (A.dtype x) ds.(0) in
    let positions = A.expect D.Int64 ds.(1) in
    K.sort ~axis:c.axis ~descending:c.descending ~values ~positions x
  in
  equal answer A.Done
    (same_guarded side ~call [| c.x |]
       [| (dtype_of c.x, y); (D.Any D.Int64, y) |])

(* The suite *)

let cases = Gen.with_pp pp_case (case_gen ())

let examples =
  [
    example (D.Any D.Complex128) [| 2; 70 |];
    example (D.Any D.Complex128) [| 9 |] ~descending:true;
    example (D.Any D.Int64) [| 2; 300 |] ~draw:(Narrow 3);
    example (D.Any D.Complex64) [| 300 |] ~draw:(Narrow 5) ~descending:true;
    example (D.Any D.Int4) [| 3; 70 |];
    example (D.Any D.Float4_e2m1fn) [| 9 |] ~descending:true;
  ]

(* Slices the job's threads share, and one long slice. *)
let large =
  Gen.with_pp pp_case
    (case_gen
       ~shapes:(Gen.of_list [ [| 100_000 |]; [| 400; 500 |]; [| 3; 40_000 |] ])
       ())

(* One long slice of 4-byte keys packed with positions; of 8-byte keys beside
   them, keeping some or all, spread, so that they take their top digit first,
   or skewed, so that they fall back to LSD passes; and of 8-byte keys of a
   narrow span, packed. *)
let large_examples =
  [
    example (D.Any D.Float32) [| 100_000 |];
    example (D.Any D.Int64) [| 100_000 |] ~descending:true ~k:75_000;
    example (D.Any D.Int64) [| 100_000 |] ~draw:Spread;
    example (D.Any D.Float64) [| 100_000 |] ~draw:Spread ~descending:true;
    example (D.Any D.Uint64) [| 100_000 |] ~draw:Skewed;
    example (D.Any D.Float64) [| 100_000 |] ~draw:Skewed ~k:75_000;
    example (D.Any D.Int64) [| 100_000 |] ~draw:(Narrow 3);
    (* Sub-byte slices on the job's threads, of an odd length, so that
       neighbouring slices' values share a byte, read through views. *)
    example (D.Any D.Int4) [| 400; 501 |]
      ~view:{ (plain 2) with stepped = true; reversed = Some 1 };
    example (D.Any D.Uint4) [| 400; 501 |] ~descending:true ~k:301
      ~view:{ (plain 2) with perm = [| 1; 0 |] };
    example (D.Any D.Bit) [| 400; 501 |]
      ~view:{ (plain 2) with reversed = Some 0 };
  ]

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop "values are the operand at the positions" cases
        (run (law_values_at_positions b));
      prop "keeping k gives the first k of the whole sort" cases
        (run (law_prefix b));
      prop ~examples "sorts each slice stably in the order" cases
        (run (law_reference b));
      test "sorts complex numbers at a component's alignment" (fun () ->
          List.iter (run (law_aligned b)) aligned_cases);
      prop ~count:12 ~examples:long_examples
        "sorts long slices stably in the order, keeping any number" long
        (run (law_long b));
      prop
        "computes what Nx_kernel.Shape fits; another shape is not Done and \
         writes nothing outside its destinations"
        misfits (run (law_refuses b));
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop ~count:4 ~examples:large_examples "large sorts on the job's threads"
        large
        (run (law_large b));
      prop ~count:300
        "reads and writes no byte past the operand and destinations, \
         whatever pages lie beside them"
        (Gen.pair (case_gen ~shapes:guard_shapes ()) sides)
        (run (law_guards b));
    ]

(* nx_cuda.mli: a sort takes a workspace of at most 70 bytes an element
   and a few kilobytes. One of 2^24 elements takes one past the 64 MiB the
   device keeps. *)
let test_loop_workspace (b : Support.backend) () =
  let module K = (val b.kernels) in
  let module A = Nx_array in
  let n = 1 lsl 24 in
  let x = A.create b.device A.Dtype.Float32 [| n |] in
  let values = A.create b.device A.Dtype.Float32 [| n |] in
  let positions = A.create b.device A.Dtype.Int64 [| n |] in
  Kernels_cases.loop_held b ~calls:10 ~bound:((70 * n) + 65536) (fun () ->
      equal answer A.Done (K.sort ~axis:0 ~descending:false ~values ~positions x);
      Rig.Buffer.wait (A.buffer values) Read_write)

let cuda (b : Support.backend) =
  group ("nx.cuda " ^ b.name)
    [
      test "sorts in a loop hold one sort's workspace" (fun () ->
          b.around (test_loop_workspace b));
    ]

(* rig.nv *)

(* rig.nv loads the cubin CUDA loads, so a sort through it has CUDA's
   bits: what it can get wrong, its launches, memory and order, shows in
   them. Each case runs through CUDA, then through rig.nv. *)
let nv_laws (first : Support.backend) (nv : Support.backend) =
  let agree call = Elements.agree_bits first nv call in
  group nv.name
    [
      group "bits are cuda's"
        [
          prop "of sorts" cases (fun c ->
              agree (fun b ->
                  match sort_on b c with Some (v, p) -> [| v; p |] | None -> [||]));
          prop ~count:12 ~examples:long_examples "of long sorts" long (fun c ->
              let (D.Any d) = dtype_of c.x in
              let n = (shape_of c.x).(c.axis) in
              cover "keeps more than 1024, not all" (kept c > 1024 && kept c < n);
              cover "16-byte keys past 2048" (D.bits d = 128 && n > 2048);
              agree (fun b ->
                  match sort_on b c with Some (v, p) -> [| v; p |] | None -> [||]));
        ];
    ]

let () =
  let nv =
    match (Support.gpus (), Support.nv ()) with
    | first :: _, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.sort"
       (List.map laws (Support.backends ())
       @ [ cpu Support.cpu ]
       @ List.map cuda (Support.gpus ())
       @ nv))
