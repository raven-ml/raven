(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx_kernel.S.sort through every kernel library the host runs, against a stable
   sort built here in the order nx_kernel.mli states, at every dtype, through
   views, with NaN payloads, -0 and ties. nx.cpu's group sorts on the job's
   threads and against inaccessible pages. *)

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

(* A sort and the recipe of its operand: [x] of [dtype] and [shape] through
   [view], its elements drawn from [seed] as [draw] says. *)
type case = {
  dtype : D.any;
  shape : int array;
  view : view;
  draw : draw;
  seed : int;
  axis : int;
  descending : bool;
  k : int option;
}

let pp_case ppf c =
  Format.fprintf ppf "sort%s along %d%s of %a %a, %s (%s), seed %d"
    (if c.descending then " descending" else "")
    c.axis
    (match c.k with None -> "" | Some k -> Printf.sprintf " keeping %d" k)
    pp_dtype c.dtype pp_ints c.shape
    (match c.draw with
    | Tied -> "tied"
    | Narrow low -> Printf.sprintf "narrow over %d bytes" low
    | Spread -> "spread"
    | Skewed -> "skewed"
    | Special -> "specials"
    | Stripes -> "striped")
    (String.concat ", " (view_names c.view))
    c.seed

let views c = view_names c.view

let bits c =
  let (D.Any d) = c.dtype in
  D.bits d

let extent c = c.shape.(c.axis)
let kept c = match c.k with None -> extent c | Some k -> k

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

(* The case's operand. A narrow element keeps the base's high bytes, so that
   integers lie within a few low bytes of each other and floats within a few low
   bytes of their significands. *)
let operand_of c =
  let d = c.dtype in
  let rs = Random.State.make [| c.seed |] in
  let pool = Array.init 6 (fun _ -> element d rs) in
  let base = element d rs in
  operand d c.shape c.view (fun k ->
      match c.draw with
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

(* Shapes: small ones of any rank, slices of 63 to 65 elements, ones of 100 to
   3000, and one to three slices of 4097 to 40,000, along the last axis or the
   first. *)
let shape_gen =
  let open Gen in
  let small = array ~size:(int_range 1 4) (int_range 0 6) in
  let edge =
    map (fun n -> [| 2; n |]) (of_list ~pp:Format.pp_print_int [ 63; 64; 65 ])
  in
  let mid =
    let* n = int_range 100 3000 in
    let+ before = array ~size:(int_range 0 1) (int_range 1 3) in
    Array.append before [| n |]
  in
  let long =
    let* n =
      frequency [ (3, int_range 4097 12_000); (1, int_range 12_001 40_000) ]
    in
    let* outer = int_range 1 3 in
    let+ last = bool in
    if outer = 1 then [| n |]
    else if last then [| outer; n |]
    else [| n; outer |]
  in
  frequency [ (5, small); (1, edge); (2, mid); (1, long) ]

(* Sorts along any axis, keeping all, a few or any number, of operands through
   any view, broadcast along an axis one time in six. *)
let case_gen ?(shapes = shape_gen) () =
  let open Gen in
  let* dtype = of_list ~pp:pp_dtype D.all in
  let* shape = shapes in
  let r = Array.length shape in
  let* axis = int_range 0 (r - 1) in
  let* descending = bool in
  let* k =
    frequency
      [
        (2, constant None);
        (1, map Option.some (int_range 0 (min shape.(axis) 4)));
        (1, map Option.some (int_range 0 shape.(axis)));
      ]
  in
  let* broadcast =
    frequency
      [ (5, constant []); (1, map (fun a -> [ a ]) (int_range 0 (r - 1))) ]
  in
  let* view = view_of ~broadcast r in
  let (D.Any dt) = dtype in
  let* draw =
    frequency
      [
        (4, constant Tied);
        (1, map (fun low -> Narrow low) (int_range 1 (max 1 (D.bits dt / 8))));
        (1, constant Spread);
        (1, constant Skewed);
        (1, constant Special);
      ]
  in
  let+ seed = int in
  { dtype; shape; view; draw; seed; axis; descending; k }

let sorts = Gen.with_pp pp_case (case_gen ())

(* A constructed case along the last axis, drawn from seed 1, so that every seed
   reaches its regime. *)
let example ?k ?(descending = false) ?(draw = Tied) ?view dtype shape =
  let r = Array.length shape in
  let view = Option.value view ~default:(plain r) in
  { dtype; shape; view; draw; seed = 1; axis = r - 1; descending; k }

(* The case's sort of [x] by [b]'s kernels: the values and the positions. *)
let sort_on (b : Support.backend) c x =
  let module K = (val b.kernels) in
  let (A.Any x) = x in
  let y = Iarray.of_array (along c.shape c.axis (kept c)) in
  let x = Support.on b x in
  let values = A.create b.device (A.dtype x) y in
  let positions = A.create b.device D.Int64 y in
  (* Destinations start as 0xA5 bytes, not zeros: an element the kernels do
     not store differs from the reference unless the reference holds those
     bytes there. *)
  Rig.Buffer.fill (A.buffer values) '\xA5';
  Rig.Buffer.fill (A.buffer positions) '\xA5';
  match K.sort ~axis:c.axis ~descending:c.descending ~values ~positions x with
  | A.Done -> (A.Any values, A.Any positions)
  | A.Declined -> failf "%s declined a sort" b.name
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* The order of nx_kernel.mli over the elements [es] of [d], by their indices,
   each element decoded once: "-∞ < … < -0 < +0 < … < +∞ < NaN", every NaN, and
   every complex number with a NaN part, equal; complex numbers by real part,
   then imaginary part; integers by value; [false] first. *)
let order_of (D.Any dt as d) es =
  let nans na nb k =
    match (na, nb) with
    | true, true -> 0
    | true, false -> 1
    | false, true -> -1
    | false, false -> k ()
  in
  if D.is D.Complex dt then
    let h = D.bits dt / 16 in
    let get = if h = 4 then get32 else get64 in
    let re = Array.map (fun e -> get e 0) es in
    let im = Array.map (fun e -> get e h) es in
    let nan p = Float.is_nan re.(p) || Float.is_nan im.(p) in
    fun p q ->
      nans (nan p) (nan q) (fun () ->
          match compare_floats re.(p) re.(q) with
          | 0 -> compare_floats im.(p) im.(q)
          | c -> c)
  else if D.is D.Float dt then
    let get e =
      if is_narrow d then decode d e
      else if D.bits dt = 32 then get32 e 0
      else get64 e 0
    in
    let v = Array.map get es in
    fun p q ->
      nans
        (Float.is_nan v.(p))
        (Float.is_nan v.(q))
        (fun () -> compare_floats v.(p) v.(q))
  else if D.is D.Boolean dt then
    let v = Array.map (fun e -> e <> "\000") es in
    fun p q -> Bool.compare v.(p) v.(q)
  else
    let v = Array.map (int_value d) es in
    if D.is D.Signed dt then fun p q -> Int64.compare v.(p) v.(q)
    else fun p q -> Int64.unsigned_compare v.(p) v.(q)

(* The reference: each slice of [x]'s elements with their positions, stably
   sorted in [order_of]'s order, reversed where descending with equal elements
   kept in increasing position, the first [kept c] of them: the values' bits and
   the positions, in C order of the results. *)
let expected c x =
  let es = elements x in
  let order = order_of c.dtype es in
  let n = extent c and keep = kept c in
  let r = Array.length c.shape in
  let inner = total (Array.sub c.shape (c.axis + 1) (r - c.axis - 1)) in
  let outer = total (Array.sub c.shape 0 c.axis) in
  let values = Array.make (outer * keep * inner) "" in
  let ps = Array.make (outer * keep * inner) 0L in
  let slice = Array.make n 0 in
  for o = 0 to outer - 1 do
    for i = 0 to inner - 1 do
      let at j = (((o * n) + j) * inner) + i in
      let cmp p q =
        let r = order (at p) (at q) in
        if c.descending then -r else r
      in
      for j = 0 to n - 1 do
        slice.(j) <- j
      done;
      Array.stable_sort cmp slice;
      for j = 0 to keep - 1 do
        let y = (((o * keep) + j) * inner) + i in
        values.(y) <- es.(at slice.(j));
        ps.(y) <- Int64.of_int slice.(j)
      done
    done
  done;
  (values, ps)

let check c x (values, positions) =
  let ev, ep = expected c x in
  equal ~msg:"positions" (array int64) ep (Elements.positions positions);
  equal ~msg:"values" (array string) ev (elements values)

let agrees (b : Support.backend) c =
  let x = operand_of c in
  check c x (sort_on b c x)

(* Each slice's order *)

(* Ties, narrow spans, specials and stripes in slices of every length and of
   every width of key; a broadcast operand, whose slice along its broadcast axis
   ties whole; keeping none, a few, more than 1024 or all, of slices along the
   last axis or an outer one, reversed and stepped; operands of no element. *)
let examples =
  [
    example (D.Any D.Complex128) [| 2; 70 |];
    example (D.Any D.Complex128) [| 9 |] ~descending:true;
    example (D.Any D.Int64) [| 2; 300 |] ~draw:(Narrow 3);
    example (D.Any D.Complex64) [| 300 |] ~draw:(Narrow 5) ~descending:true;
    example (D.Any D.Int4) [| 3; 70 |];
    example (D.Any D.Float4_e2m1fn) [| 9 |] ~descending:true;
    example (D.Any D.Int64) [| 3; 9 |] ~k:0;
    example (D.Any D.Float32) [| 3; 70 |] ~descending:true
      ~view:{ (plain 2) with broadcast = [ 1 ] };
    example (D.Any D.Uint8) [| 0; 5 |];
    example (D.Any D.Float64) [| 5; 0 |];
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
      (example (D.Any D.Int16) [| 9000; 3 |]
         ~view:{ (plain 2) with reversed = Some 1; stepped = true })
      with
      axis = 0;
      k = Some 50;
    };
  ]

let law_order (b : Support.backend) c =
  let n = extent c in
  let long = n > 4096 in
  cover "descending" c.descending;
  cover "keeps some" (c.k <> None && kept c < n);
  cover "keeps none of a nonempty slice" (kept c = 0 && total c.shape > 0);
  cover "slices of 64 or more kept whole" (n >= 64 && c.k = None);
  cover "complex128's two keys" (c.dtype = D.Any D.Complex128 && n >= 2);
  cover "sub-byte" (bits c < 8 && n >= 2);
  cover "narrow 8-byte keys"
    ((match c.draw with Narrow _ -> true | _ -> false)
    && n >= 64
    && bits c = 64);
  cover "a broadcast operand" (List.mem "broadcast" (views c));
  cover "an operand of no element" (total c.shape = 0);
  cover "a slice past 4096 keeping more than 1024, not all"
    (long && kept c > 1024 && kept c < n);
  cover "complex128 slices past 2048" (bits c = 128 && n > 2048);
  cover "8-byte keys in a slice past 4096" (long && bits c = 64);
  cover "sub-byte slices past 4096" (long && bits c < 8);
  cover "floats' specials in a slice past 4096" (long && c.draw = Special);
  cover "slices that vary in different bytes" (c.draw = Stripes);
  cover "several slices past 4096" (long && total c.shape > n);
  cover "a slice past 4096 along an outer axis"
    (long && c.axis < Array.length c.shape - 1);
  agrees b c

(* Complex numbers at their alignment *)

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

(* Complex numbers of both widths, by each path: short slices, long ones keeping
   all and a few, and a complex64 slice of 64Ki, which nx.cpu sorts whole on the
   job's threads where it has more than one. *)
let aligned_cases =
  List.concat_map
    (fun d ->
      [
        example d [| 2; 70 |] ~draw:Special;
        example d [| 9000 |] ~draw:Special ~descending:true;
        example d [| 2; 9000 |] ~k:10;
      ])
    [ D.Any D.Complex64; D.Any D.Complex128 ]
  @ [ example (D.Any D.Complex64) [| 1 lsl 16 |] ~draw:Special ]

(* The operand and both destinations at their least alignment. *)
let test_aligned (b : Support.backend) c =
  let module K = (val b.kernels) in
  let x = operand_of c in
  let (A.Any a) = x in
  let dt = A.dtype a in
  let align = D.bits dt / 16 in
  let y = along c.shape c.axis (kept c) in
  let xs = shifted b dt c.shape ~align (Some a) in
  let values = shifted b dt y ~align None in
  let positions = A.create b.device D.Int64 (Iarray.of_array y) in
  match K.sort ~axis:c.axis ~descending:c.descending ~values ~positions xs with
  | A.Done -> check c x (A.Any values, A.Any positions)
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Misfits *)

(* A sort's shapes, [x], [values] and [positions], along [along], each near
   a fit: [values] keeps up to one past [x]'s extent. *)
type misfit = { along : int; shapes : int array array }

let pp_misfit ppf c =
  Format.fprintf ppf "sort along %d of %a into %a and %a" c.along pp_ints
    c.shapes.(0) pp_ints c.shapes.(1) pp_ints c.shapes.(2)

let misfits =
  Gen.with_pp pp_misfit
    (let open Gen in
     let* x = Nx_array_gen.shape in
     let* axis = axis_near x in
     let d = if axis >= 0 && axis < Array.length x then x.(axis) else 0 in
     let* k = int_range 0 (d + 1) in
     let* values = near (along x axis k) in
     let+ positions = near values in
     { along = axis; shapes = [| x; values; positions |] })

(* A call that fits, keeping 2 of 4, and one whose destinations keep one past
   the axis's extent. *)
let misfit_examples =
  [
    { along = 1; shapes = [| [| 3; 4 |]; [| 3; 2 |]; [| 3; 2 |] |] };
    { along = 1; shapes = [| [| 3; 4 |]; [| 3; 5 |]; [| 3; 5 |] |] };
  ]

(* The kernel keeps [values]' extent along the axis: Nx_kernel.Shape.sort's
   [k]. *)
let law_misfit (b : Support.backend) c =
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

(* nx.cpu *)

(* Slices the job's threads share, and one slice of 64Ki, the length from which
   nx.cpu sorts a single slice on several threads. *)
let threaded =
  Gen.with_pp pp_case
    (case_gen
       ~shapes:(Gen.of_list [ [| 1 lsl 16 |]; [| 400; 500 |]; [| 3; 40_000 |] ])
       ())

(* One slice of 4-byte keys; of 8-byte keys, keeping some or all, spread or
   skewed; and of 8-byte keys of a narrow span. *)
let threaded_examples =
  let long = 1 lsl 16 in
  [
    example (D.Any D.Float32) [| long |];
    example (D.Any D.Int64) [| long |] ~descending:true ~k:50_000;
    example (D.Any D.Int64) [| long |] ~draw:Spread;
    example (D.Any D.Float64) [| long |] ~draw:Spread ~descending:true;
    example (D.Any D.Uint64) [| long |] ~draw:Skewed;
    example (D.Any D.Float64) [| long |] ~draw:Skewed ~k:50_000;
    example (D.Any D.Int64) [| long |] ~draw:(Narrow 3);
    (* Sub-byte slices on the job's threads, of an odd length, so that
       neighbouring slices' values share a byte, read through views. *)
    example (D.Any D.Int4) [| 400; 501 |]
      ~view:{ (plain 2) with stepped = true; reversed = Some 1 };
    example (D.Any D.Uint4) [| 400; 501 |] ~descending:true ~k:301
      ~view:{ (plain 2) with perm = [| 1; 0 |] };
    example (D.Any D.Bit) [| 400; 501 |]
      ~view:{ (plain 2) with reversed = Some 0 };
  ]

let law_threaded (b : Support.backend) c =
  let one = Array.length c.shape = 1 && kept c > 64 in
  let wide = one && bits c = 64 in
  cover "one slice of 64Ki keeping more than 64" one;
  cover "one slice of 64Ki of 8-byte keys, spread" (wide && c.draw = Spread);
  cover "one slice of 64Ki of 8-byte keys, skewed" (wide && c.draw = Skewed);
  cover "one slice of 64Ki of 8-byte keys, kept whole" (wide && c.k = None);
  cover "sub-byte slices of 100,000 elements or more"
    (bits c < 8 && total c.shape >= 100_000);
  agrees b c

(* Slices of one element to a few vectors past the widest, of 63 to 65 and 127
   to 129 elements, and of 100 to 3000. *)
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

let guarded =
  Gen.pair (Gen.with_pp pp_case (case_gen ~shapes:guard_shapes ())) sides

(* The same values and positions with the operand, through its view, and
   both destinations against an inaccessible page. *)
let law_guards (b : Support.backend) (c, side) =
  let module K = (val b.kernels) in
  let after = side = Support.After in
  cover "along the innermost axis" (c.axis = Array.length c.shape - 1);
  cover "along an outer axis" (c.axis < Array.length c.shape - 1);
  cover "slices of 100 or more" (extent c >= 100);
  cover "keeping some" (kept c < extent c);
  cover "sub-byte" (bits c < 8);
  cover "a stepped operand at the page before"
    ((not after) && List.mem "stepped" (views c));
  cover "a reversed operand at the page after"
    (after && List.mem "reversed" (views c));
  let y = along c.shape c.axis (kept c) in
  let call xs ds =
    let (A.Any x) = xs.(0) in
    let values = A.expect (A.dtype x) ds.(0) in
    let positions = A.expect D.Int64 ds.(1) in
    K.sort ~axis:c.axis ~descending:c.descending ~values ~positions x
  in
  equal answer A.Done
    (same_guarded side ~call
       [| operand_of c |]
       [| (c.dtype, y); (D.Any D.Int64, y) |])

(* nx_cuda.mli: a sort takes a workspace of at most 70 bytes an element
   and a few kilobytes. One of 2^24 elements takes one past the 64 MiB the
   device keeps, a buffer of the call's own, which the bound counts in the
   driver's units ({!Rig.allocation}). *)
let test_loop_workspace (b : Support.backend) () =
  let module K = (val b.kernels) in
  let module A = Nx_array in
  let n = 1 lsl 24 in
  let x = A.create b.device A.Dtype.Float32 [| n |] in
  let values = A.create b.device A.Dtype.Float32 [| n |] in
  let positions = A.create b.device A.Dtype.Int64 [| n |] in
  let bound = Rig.allocation b.device ((70 * n) + 65536) in
  Kernels_cases.loop_held b ~calls:10 ~bound (fun () ->
      equal answer A.Done (K.sort ~axis:0 ~descending:false ~values ~positions x);
      Rig.Buffer.wait (A.buffer values) Read_write)

(* The suite *)

let laws (d : Support.declared) =
  let run f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  group d.label
    [
      prop ~examples
        "orders each slice stably, keeping the first elements of the order"
        sorts (run law_order);
      cases "orders complex numbers at a component's alignment"
        ~id:(Format.asprintf "%a" pp_case)
        aligned_cases (run test_aligned);
      prop ~examples:misfit_examples
        "a call of shapes Nx_kernel.Shape.sort refuses is not Done and writes \
         nothing outside its destinations"
        misfits (run law_misfit);
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f b x) in
  group ("nx.cpu " ^ b.name)
    [
      prop ~count:4 ~examples:threaded_examples
        "orders each slice stably on the job's threads" threaded
        (run law_threaded);
      prop ~count:300
        "reads and writes no byte past the operand and destinations, whatever \
         pages lie beside them"
        guarded (run law_guards);
    ]

let cuda (d : Support.declared) =
  group ("nx.cuda " ^ d.label)
    [
      test "sorts in a loop hold one sort's workspace" (fun () ->
          let b = Support.opened d in
          b.around (test_loop_workspace b));
    ]

(* rig.nv loads the cubin CUDA loads, so a sort through it has CUDA's
   bits: what it can get wrong, its launches, memory and order, shows in
   them. Each case runs through CUDA, then through rig.nv. *)
let nv_laws (first : Support.declared) (nv : Support.declared) =
  group nv.label
    [
      group "bits are cuda's"
        [
          prop ~examples "of sorts" sorts (fun c ->
              let n = extent c in
              cover "keeps more than 1024, not all" (kept c > 1024 && kept c < n);
              cover "complex128 slices past 2048" (bits c = 128 && n > 2048);
              let x = operand_of c in
              Elements.agree_bits (Support.opened first) (Support.opened nv)
                (fun b ->
                  let v, p = sort_on b c x in
                  [| v; p |]));
        ];
    ]

let () =
  let nv =
    match (Support.declared_cuda, Support.declared_nv) with
    | Some first, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.sort"
       (List.map laws
          (Support.declare Support.cpu
          :: List.filter_map Fun.id
               [
                 Support.declared_cuda;
                 Support.declared_amd;
                 Support.declared_metal;
               ])
       @ [ cpu Support.cpu ]
       @ List.map cuda (Option.to_list Support.declared_cuda)
       @ nv))
