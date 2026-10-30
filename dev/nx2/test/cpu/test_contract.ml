(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's contractions, as nx_cpu.mli states them: the cases it computes,
   how it scales, wraps and rounds, bit for bit, against a reference built
   here, and the same bits under every layout of the same values. *)

open Windtrap
open Nx_array_gen
open Contract_cases
module A = Nx_array
module D = Nx_array.Dtype
module Support = Nx_kernels_support

(* Cases *)

let integer_accs =
  D.
    [
      Any Int4; Any Uint4; Any Int8; Any Uint8; Any Int16; Any Uint16;
      Any Int32; Any Uint32; Any Int64; Any Uint64;
    ]

(* The floats narrower than float32, which a float32 sum is rounded to. *)
let narrow =
  D.
    [
      Any Float16; Any Bfloat16; Any Float8_e4m3fn; Any Float8_e5m2;
      Any Float4_e2m1fn;
    ]

(* Cases into a result of another dtype than acc: mostly a narrow float, at
   times any dtype. *)
let rounded =
  let pick = Gen.of_list ~pp:pp_dtype in
  Gen.with_pp pp_case
    (Gen.bind (pick (accs @ integer_accs)) (fun acc ->
         Gen.bind
           (Gen.frequency [ (3, pick narrow); (1, pick D.all) ])
           (fun out ->
             let dts = if is_integer acc then integers else held acc in
             case_of ~acc ~out ~dts ())))

(* The reference *)

(* Whether nx_cpu.mli says nx.cpu computes [c]: an acc it computes in, of
   operands of any dtype, into any result, at any layout. *)
let cpu_computes c = List.mem c.acc accs || is_integer c.acc

(* The order *)

(* [x] rounded to float32. *)
let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* float32's fused multiply-add: a · b, exact in a double for float32 [a] and
   [b], plus [c], rounded to odd in a double and then to float32, which
   rounds once since 53 >= 2·24 + 2. *)
let fma32 a b c =
  let p = a *. b in
  let s, e = two_sum c p in
  if e = 0. || not (Float.is_finite s) then round32 s
  else if Int64.logand (Int64.bits_of_float s) 1L = 1L then round32 s
  else round32 (if e > 0. then Float.succ s else Float.pred s)

(* float32's addition, rounded once through a double for the same reason. *)
let add32 x y = round32 (x +. y)

(* [o]'s result in [acc], in chain order where [chain], one fused chain
   from init or +0; or in lane order, blocks of 1024 terms in 16 lanes,
   summed by trees, then init, with no term init as it is. *)
let ordered ~chain acc o =
  let fma, add =
    if part_of acc = D.Any D.Float32 then (fma32, add32)
    else (Exact.fma, ( +. ))
  in
  let k = Array.length o.av in
  if chain then begin
    let acc = ref (Option.value ~default:0. o.init) in
    for t = 0 to k - 1 do
      acc := fma o.av.(t) o.bv.(t) !acc
    done;
    !acc
  end
  else begin
    let block x =
      let l = Array.make 16 0. in
      for t = x * 1024 to min k ((x + 1) * 1024) - 1 do
        l.(t mod 16) <- fma o.av.(t) o.bv.(t) l.(t mod 16)
      done;
      List.iter
        (fun w ->
          for i = 0 to w - 1 do
            l.(i) <- add l.(i) l.(i + w)
          done)
        [ 8; 4; 2; 1 ];
      l.(0)
    in
    let rec tree s n =
      if n = 1 then s.(0)
      else
        let h = ref 1 in
        while 2 * !h < n do
          h := 2 * !h
        done;
        add (tree s !h) (tree (Array.sub s !h (n - !h)) (n - !h))
    in
    let n = (k + 1023) / 1024 in
    let sum = if n = 0 then 0. else tree (Array.init n block) n in
    match o.init with None -> sum | Some i -> if n = 0 then i else add i sum
  end

(* Each output adds its products in the order nx_cpu.mli states for its
   shape ({!order}). *)
let law_order b c =
  covers c;
  match run b c with
  | None when cpu_computes c ->
      failf "nx.cpu declined a case nx_cpu.mli says it computes"
  | None -> cover "declined" true
  | Some y ->
      let want = ordered ~chain:(order c = Chain) c.acc in
      equal (list string) []
        (wrong c y (fun o x ->
             let w = want o in
             if bits x = bits w then None
             else Some (Printf.sprintf "%h, expected %h" x w)))

(* Contractions of few outputs over a depth of several blocks of lanes,
   4097 or 200003 products, which the threads share. *)
let deep_cases =
  let open Gen in
  with_pp pp_case
    (let* acc = of_list ~pp:pp_dtype D.[ Any Float32; Any Float64 ] in
     let* k = of_list [ 4097; 200003 ] in
     let* m = int_range 1 4 and+ n = int_range 1 6 in
     let* va = view 2 0 and+ vb = view 2 0 and+ vi = view 2 0 in
     let* with_init = bool in
     let+ seed = int in
     make ~acc ~out:acc ([||], [| m |], [| n |], [| k |]) (acc, va) (acc, vb)
       (if with_init then Some vi else None)
       seed)

let deep_examples =
  let f32 = D.Any D.Float32 and f64 = D.Any D.Float64 and v = Elements.plain 2 in
  List.map
    (fun (acc, k) ->
      make ~acc ~out:acc ([||], [| 3 |], [| 5 |], [| k |]) (acc, v) (acc, v)
        (Some v) 1)
    [ (f32, 4097); (f32, 200003); (f64, 4097); (f64, 200003) ]

(* A contraction on the job's threads has the bits of the same contraction
   on one thread. *)
let law_threads b c =
  let _, k, _ = sizes c in
  cover "float32" (c.acc = D.Any D.Float32);
  cover "float64" (c.acc = D.Any D.Float64);
  cover "a depth of 4097" (k = 4097);
  cover "a depth of 200003" (k = 200003);
  cover "init" (c.init <> None);
  match (run b c, Nx_cpu.one_thread (fun () -> run b c)) with
  | Some y, Some y' ->
      equal (list string) [] (differ_bits c.acc ~other:"one thread" y y')
  | _ -> failf "nx.cpu declined a float contraction"

(* A result of another dtype than acc is the result in acc, from init cast
   to acc, cast to it: the sum rounds once. *)
let law_rounds_once b c =
  cover "a narrow float result" (List.mem c.out narrow);
  cover "an integer acc" (is_integer c.acc);
  cover "a 4-bit acc"
    (let (D.Any acc) = c.acc in
     is_integer c.acc && D.bits acc = 4);
  cover "a complex acc" (is_complex c.acc);
  cover "init" (c.init <> None);
  match run b c with
  | None when cpu_computes c ->
      failf "nx.cpu declined a case nx_cpu.mli says it computes"
  | None -> cover "declined" true
  | Some y -> (
      let c' =
        { c with out = c.acc; init = Option.map (cast_to c.acc) c.init }
      in
      match run b c' with
      | None -> failf "nx.cpu declined the case into acc"
      | Some ya ->
          cover "computed" true;
          let want = cast_to c.out ya in
          equal (list string) []
            (differ_bits c.out ~other:"the cast of acc's" y want))

(* Integers *)

let integer_cases =
  Gen.with_pp pp_case
    (Gen.bind (Gen.of_list ~pp:pp_dtype integer_accs) (fun acc ->
         Gen.bind
           (Gen.frequency
              [
                (2, Gen.constant integers); (1, Gen.constant (others integers));
              ])
           (fun dts -> case_of ~acc ~out:acc ~dts ())))

(* [x]'s values in an array one element longer along its last axis, read
   from its second element: adjacent along that axis, at an odd offset. *)
let shifted (A.Any x as a) =
  let s = shape_of a in
  let last = Array.length s - 1 in
  let made = Array.mapi (fun i e -> if i = last then e + 1 else e) s in
  let at i e = { M.start = (if i = last then 1 else 0); count = e; step = 1 } in
  let y =
    Option.get
      (A.move
         (M.Slice (Iarray.of_array (Array.mapi at s)))
         (A.create Rig.host (A.dtype x) (Iarray.of_array made)))
  in
  A.transfer ~src:x ~dst:y;
  A.Any y

(* Int8 and uint8 operands into 32-bit accumulators, whose products pass 16
   bits: every element of a and b at the dtype's extreme and init at the
   accumulator's largest value, so that every sum wraps ([extreme]); b
   stored [n × k] ([stored]); operands adjacent along k at an odd offset
   ([shifted]); rows, columns and depths from one to a few thousand, an
   empty batch and a batch of two. *)
let byte_examples =
  let plain = Elements.plain in
  let by_rows r =
    { (plain r) with perm = Array.init r (fun i -> if i < r - 2 then i else (2 * r) - 3 - i) }
  in
  List.concat_map
    (fun (acc, d, low, top) ->
      let case e vb =
        let ra, _, ry = ranks e in
        make ~acc ~out:acc e (d, plain ra) (d, vb) (Some (plain ry)) 3
      in
      let extreme e vb =
        let c = case e vb in
        let fill v (A.Any x as a) =
          let s = shape_of a in
          let vs = Array.make (total s) v in
          let vs = A.Any (A.of_array D.Int64 (Iarray.of_array s) vs) in
          let src = A.expect (A.dtype x) (cast_to (D.Any (A.dtype x)) vs) in
          A.transfer ~src ~dst:x;
          a
        in
        { c with a = fill low c.a; b = fill low c.b; init = Option.map (fill top) c.init }
      in
      let stored e vb =
        let c = case e vb in
        let (A.Any b) = c.b in
        { c with b = A.Any (A.copy b) }
      in
      [
        stored ([||], [| 130 |], [| 40 |], [| 2052 |]) (by_rows 2);
        case ([||], [| 129 |], [| 50 |], [| 64 |]) (plain 2);
        case ([||], [| 9 |], [| 33 |], [| 7 |]) (by_rows 2);
        (let c = case ([||], [| 5 |], [| 9 |], [| 64 |]) (by_rows 2) in
         { c with a = shifted c.a; b = shifted c.b });
        stored ([| 0 |], [| 3 |], [| 5 |], [| 4 |]) (by_rows 3);
        stored ([| 2 |], [| 3 |], [| 7 |], [| 20 |]) (by_rows 3);
        case ([||], [| 2 |], [| 1 |], [| 7 |]) (plain 2);
        case ([||], [| 8 |], [| 1024 |], [| 12 |]) (plain 2);
        case ([||], [| 33 |], [| 1024 |], [| 4096 |]) (plain 2);
        extreme ([||], [| 16 |], [| 40 |], [| 2048 |]) (plain 2);
        extreme ([||], [| 7 |], [| 100 |], [| 70 |]) (plain 2);
        extreme ([||], [| 8 |], [| 100 |], [| 70 |]) (plain 2);
        (let c = stored ([||], [| 2 |], [| 3 |], [| 4096 |]) (by_rows 2) in
         { c with a = shifted c.a });
        extreme ([||], [| 2 |], [| 300 |], [| 37 |]) (plain 2);
        extreme ([||], [| 15 |], [| 20 |], [| 18 |]) (plain 2);
      ])
    D.[ (Any Int32, Any Int8, -128L, 0x7FFFFFFFL);
        (Any Uint32, Any Uint8, 255L, 0xFFFFFFFFL) ]

(* [x]'s values as int64s: an integer's own value, a bool's 0 or 1. *)
let ints x =
  let d = A.create Rig.host D.Int64 (Iarray.of_array (shape_of x)) in
  cast_into d x;
  A.to_array d

(* [v] modulo [acc]'s width, read as [acc] reads its bits. *)
let wrap (D.Any dt) v =
  let w = D.bits dt in
  let signed = match D.kind dt with D.Signed -> true | _ -> false in
  if w = 64 then v
  else
    let m = Int64.logand v (Int64.pred (Int64.shift_left 1L w)) in
    if signed && Int64.compare m (Int64.shift_left 1L (w - 1)) >= 0 then
      Int64.sub m (Int64.shift_left 1L w)
    else m

(* Each output is init, or 0, plus the products of its operands' values
   modulo the accumulator's width: int64 arithmetic wraps modulo 2^64, a
   multiple of every width. *)
let integer_covers (c : case) =
  let wider (A.Any x) = D.Any (A.dtype x) <> c.acc in
  cover "an operand of another dtype" (wider c.a || wider c.b);
  let (D.Any acc) = c.acc in
  cover "a 64-bit accumulator" (D.bits acc = 64);
  cover "a 4-bit accumulator" (D.bits acc = 4);
  cover "an operand acc does not hold" (converted c);
  cover "a float or complex operand into a 4-bit accumulator"
    (D.bits acc = 4
    && Array.exists
         (fun (A.Any x) ->
           match D.kind (A.dtype x) with
           | D.Float | D.Complex -> true
           | _ -> false)
         (operands c))

(* An integer output is its products' sum from init modulo the
   accumulator's width. *)
let law_wraps b (c : case) =
  integer_covers c;
  cover "init" (c.init <> None);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "stepped"; "broadcast"; "permuted" ];
  match run b c with
  | None when cpu_computes c ->
      failf "nx.cpu declined a case nx_cpu.mli says it computes"
  | None -> cover "declined" true
  | Some y ->
      let _, oa, ob, walk = index c in
      let ints x = ints (cast_to c.acc x) in
      let a = ints c.a and bv = ints c.b and init = Option.map ints c.init in
      let got = ints y and bad = ref [] in
      (* Each column of b gathered once, its values mapped by [f], as
         outputs gathers them: read across b's rows for every output, 33 ×
         1024 outputs of 4096 products took 1.3 s, 0.5 gathered. *)
      let gathered f =
        let columns = Hashtbl.create 256 in
        fun rb ->
          match Hashtbl.find_opt columns rb with
          | Some col -> col
          | None ->
              let col = Array.map (fun t -> f bv.(rb + t)) ob in
              Hashtbl.add columns rb col;
              col
      in
      (* An accumulator narrower than OCaml's 63-bit ints sums in them: their
         arithmetic wraps modulo 2^63, a multiple of its width. *)
      let (D.Any acc) = c.acc in
      let sum =
        if D.bits acc < 63 then (
          let a = Array.map Int64.to_int a and column = gathered Int64.to_int in
          fun s0 ra rb ->
            let col = column rb in
            let s = ref (Int64.to_int s0) in
            for t = 0 to Array.length oa - 1 do
              s := !s + (a.(ra + oa.(t)) * col.(t))
            done;
            Int64.of_int !s)
        else
          let column = gathered Fun.id in
          fun s0 ra rb ->
            let col = column rb in
            let s = ref s0 in
            for t = 0 to Array.length oa - 1 do
              s := Int64.add !s (Int64.mul a.(ra + oa.(t)) col.(t))
            done;
            !s
      in
      walk (fun yi k ra rb ->
          let s0 = match init with Some i -> i.(k) | None -> 0L in
          let want = wrap c.acc (sum s0 ra rb) in
          if want <> got.(k) && List.length !bad < 8 then
            bad := Format.asprintf "%a: %Ld, expected %Ld" pp_ints yi got.(k) want :: !bad);
      equal (list string) [] (List.rev !bad)

let law_computes b c =
  cover "an operand acc does not hold" (converted c);
  cover "a result of another dtype" (c.out <> c.acc);
  equal ~msg:"computed" bool (cpu_computes c) (Option.is_some (run b c))

(* With no products nx.cpu gives each output init's bits, -0 and a
   signalling NaN included, below 64 outputs and from 64, in both orders:
   [ud] is an unsigned dtype of [fd]'s width, [bits] init's values in it. *)
let empty (type u ue f fe) (b : Support.backend) (ud : (u, ue) D.t)
    (fd : (f, fe) D.t) ty (bits : u array) =
  let module K = (val b.kernels) in
  let at (m, n) =
    let want = Array.init (m * n) (fun i -> bits.(i mod Array.length bits)) in
    let init = Option.get (A.bitcast fd (A.of_array ud [| m; n |] want)) in
    let a = A.create Rig.host fd [| m; 0 |] and b = A.create Rig.host fd [| 0; n |] in
    let dst = A.create Rig.host fd [| m; n |] in
    (match
       K.contract ~batch:[||] ~contracting:[| (1, 0) |] ~segments:None
         ~acc:(D.Any fd) ~dst ~init:(Some init) (plain a) (plain b)
     with
    | A.Done -> ()
    | r -> failf "contract answered %a" Nx_array_support.pp_answer r);
    equal (array ty) want (A.to_array (Option.get (A.bitcast ud dst)))
  in
  List.iter at [ (1, 3); (8, 24) ]

let test_empty b () =
  empty b D.Uint32 D.Float32 int32 [| 0x80000000l; 0x7F800001l; 0x3FC00000l |];
  empty b D.Uint64 D.Float64 int64
    [| 0x8000000000000000L; 0x7FF0000000000001L; 0x3FF8000000000000L |]

(* Scales *)

(* How a scale's array lies: C-contiguous, stepped along its last axis, or
   broadcast along its first. *)
type lies = Contiguous | Stepped | Broadcast

(* A scale drawn for an operand: [Exponent] codes, or [Value]s of [kind];
   its extents; how it lies; its values from [seed], led by the specials
   where [specials]. *)
type spec = {
  kind : D.any option;
  extents : int array;
  lies : lies;
  specials : bool;
  seed : int;
}

let pp_spec ppf s =
  let kind ppf = function
    | None -> Format.pp_print_string ppf "Exponent"
    | Some d -> Format.fprintf ppf "Value %a" pp_dtype d
  in
  let lies = function
    | Contiguous -> "contiguous"
    | Stepped -> "stepped"
    | Broadcast -> "broadcast"
  in
  Format.fprintf ppf "%a %a %s%s (seed %d)" kind s.kind pp_ints s.extents
    (lies s.lies)
    (if s.specials then " with specials" else "")
    s.seed

let pp_scaled ppf (c, sa, sb) =
  let spec ppf = function
    | None -> Format.pp_print_string ppf "none"
    | Some s -> pp_spec ppf s
  in
  Format.fprintf ppf "%a@ a scale %a@ b scale %a" pp_case c spec sa spec sb

(* The codes at a format's edges: the smallest scale, 1, the largest two,
   NaN. *)
let codes = [| 0; 127; 253; 254; 255 |]

(* [s]'s scale. Codes are near 1, values as [value] draws them; with
   [specials], led by each code of [codes] or by NaN. *)
let scale_of s : Nx_kernel.scale =
  let r = Random.State.make [| s.seed |] in
  let e = s.extents and rank = Array.length s.extents in
  let stored =
    match s.lies with
    | Stepped when rank > 0 ->
        Array.mapi (fun i x -> if i = rank - 1 then 2 * x else x) e
    | Broadcast when rank > 0 -> Array.mapi (fun i x -> if i = 0 then 1 else x) e
    | _ -> e
  in
  let view (A.Any x) =
    let move m x = Option.get (A.move m x) in
    let step i x =
      if i = rank - 1 then { M.start = 1; count = x; step = 2 }
      else { M.start = 0; count = x; step = 1 }
    in
    match s.lies with
    | Stepped when rank > 0 ->
        A.Any (move (M.Slice (Iarray.of_array (Array.mapi step e))) x)
    | Broadcast when rank > 0 -> A.Any (move (M.Broadcast (Iarray.of_array e)) x)
    | _ -> A.Any x
  in
  let n = total stored in
  match s.kind with
  | None ->
      let code i =
        if s.specials && i < Array.length codes then codes.(i)
        else if Random.State.int r 4 = 0 then
          codes.(Random.State.int r (Array.length codes))
        else 117 + Random.State.int r 21
      in
      Exponent
        (A.expect D.Uint8
           (view
              (A.Any
                 (A.of_array D.Uint8 (Iarray.of_array stored)
                    (Array.init n code)))))
  | Some dt ->
      let v i = if s.specials && i = 0 then nan else value dt r in
      let (A.Any x) =
        let stored = Iarray.of_array stored in
        view (cast_to dt (A.Any (A.of_array D.Float64 stored (Array.init n v))))
      in
      Value x

let scale_extents : Nx_kernel.scale -> int array = function
  | Exponent x -> Iarray.to_array (L.shape (A.layout x))
  | Value x -> Iarray.to_array (L.shape (A.layout x))

(* A scale's values in C order of its indices, read into [acc]. *)
let decoded acc : Nx_kernel.scale -> float array = function
  | Exponent x ->
      Array.map
        (fun e -> if e = 255 then nan else Float.ldexp 1. (e - 127))
        (A.to_array x)
  | Value x -> A.to_array (to64 (cast_to acc (A.Any x)))

(* [x] read into [acc], each element times its block's scale of [s] rounded
   once in [acc], C-contiguous in [acc]: what a scaled contraction reads;
   and whether a product of finite factors overflowed. *)
let host_scaled acc x s =
  let shape = shape_of x and se = scale_extents s in
  let xs = A.to_array (to64 (cast_to acc x)) and ss = decoded acc s in
  let r = Array.length shape in
  let ratio = Array.init r (fun i -> if se.(i) = 0 then 1 else shape.(i) / se.(i)) in
  let stride = Array.make r 1 in
  for i = r - 2 downto 0 do
    stride.(i) <- stride.(i + 1) * se.(i + 1)
  done;
  let f32 = acc = D.Any D.Float32 and over = ref false in
  let scaled flat v =
    let q = ref flat and at = ref 0 in
    for i = r - 1 downto 0 do
      at := !at + (!q mod shape.(i) / ratio.(i) * stride.(i));
      q := !q / shape.(i)
    done;
    let y = v *. ss.(!at) in
    let y = if f32 then round32 y else y in
    if Float.is_finite v && Float.is_finite ss.(!at) && not (Float.is_finite y)
    then over := true;
    y
  in
  let ys = Array.mapi scaled xs in
  (cast_to acc (A.Any (A.of_array D.Float64 (Iarray.of_array shape) ys)), !over)

(* A scale for an operand of shape [s]: along each axis a ratio of 1, the
   whole axis, or a divisor of it. *)
let spec_for s =
  let open Gen in
  let axis e =
    let ds = List.filter (fun d -> d < e && e mod d = 0) [ 2; 3; 4; 32 ] in
    if e = 0 then of_list [ 0; 1 ]
    else
      frequency
        ([ (3, constant e); (2, constant 1) ]
        @ if ds = [] then [] else [ (3, map (fun d -> e / d) (of_list ds)) ])
  in
  let rec axes = function
    | [] -> constant []
    | e :: es ->
        let* x = axis e in
        let+ xs = axes es in
        x :: xs
  in
  let* extents = axes (Array.to_list s) in
  let* kind =
    frequency
      [
        (1, constant None);
        (1, map Option.some (of_list ~pp:pp_dtype D.[ Any Float32; Any Float64; Any Float16; Any Bfloat16 ]));
      ]
  in
  let* lies = of_list [ Contiguous; Contiguous; Stepped; Broadcast ] in
  let* specials = bool in
  let+ seed = int in
  { kind; extents = Array.of_list extents; lies; specials; seed }

(* [c] with a scale on [a], on [b] or on both. *)
let with_scales c =
  let open Gen in
  let* sa = spec_for (shape_of c.a) and+ sb = spec_for (shape_of c.b) in
  let+ which = of_list [ `A; `B; `Both; `Both ] in
  match which with
  | `A -> (c, Some sa, None)
  | `B -> (c, None, Some sb)
  | `Both -> (c, Some sa, Some sb)

(* Float cases of operands of any dtype, into acc or a narrow float. *)
let scaled_cases =
  let open Gen in
  with_pp pp_scaled
    (let* acc = of_list ~pp:pp_dtype D.[ Any Float32; Any Float64 ] in
     let* dts =
       frequency
         [ (2, constant (held acc)); (1, constant (others (held acc))) ]
     in
     let* out =
       frequency [ (3, constant ~pp:pp_dtype acc); (1, of_list ~pp:pp_dtype narrow) ]
     in
     let* c = case_of ~acc ~out ~dts () in
     with_scales c)

(* The ratio along a contraction of [k]: 32 where it divides [k], else
   [k]'s largest divisor up to 64. *)
let ratio_of k =
  if k mod 32 = 0 then 32
  else
    let rec down d = if k mod d = 0 then d else down (d - 1) in
    down (min k 64)

(* Products either side of lane order's blocks of 1024 and of 4096, in
   both accs: a float32 [2; k] times bfloat16 [k; 3], each scaled in
   blocks along k; and every regime of both accs, its operands scaled in
   turn in 2-D blocks, stepped, broadcast or contiguous: a and b, a alone,
   b alone. *)
let scaled_examples =
  let plain = Elements.plain in
  let spec ?(lies = Contiguous) kind extents seed =
    Some { kind; extents; lies; specials = true; seed }
  in
  let along acc k =
    let c =
      make ~acc ~out:acc ([||], [| 2 |], [| 3 |], [| k |])
        (D.Any D.Float32, plain 2) (D.Any D.Bfloat16, plain 2) None k
    in
    let q = k / ratio_of k in
    (c, spec (Some (D.Any D.Float16)) [| 2; q |] 1, spec None [| q; 3 |] 2)
  in
  let ks = [ 1; 1023; 1024; 1025; 4095; 4096; 4097 ] in
  let accs = D.[ Any Float32; Any Float64 ] in
  let blocks x =
    Array.map
      (fun e ->
        match List.find_opt (fun d -> d < e && e mod d = 0) [ 2; 3; 4 ] with
        | Some d -> e / d
        | None -> e)
      (shape_of x)
  in
  let regime i c =
    let lies = [| Stepped; Broadcast; Contiguous |].(i / 3 mod 3) in
    let sa = spec ~lies None (blocks c.a) i in
    let sb = spec ~lies (Some c.acc) (blocks c.b) i in
    match i mod 3 with
    | 0 -> (c, sa, sb)
    | 1 -> (c, sa, None)
    | _ -> (c, None, sb)
  in
  (* A scale over an empty axis: a [2; 2^28; 0] by [2^28; 0; 3], a scaled
     in whole axes, with and without init. A scale table once reached past
     its allocation there. *)
  let empty init =
    let f64 = D.Any D.Float64 in
    let c =
      make ~acc:f64 ~out:f64
        ([||], [| 2 |], [| 3 |], [| 1 lsl 28; 0 |])
        (f64, plain 3) (f64, plain 3)
        (if init then Some (plain 2) else None)
        3
    in
    (c, spec None [| 2; 1; 1 |] 3, None)
  in
  List.concat_map (fun acc -> List.map (along acc) ks) accs
  @ List.mapi regime (regimes (D.Any D.Float32) @ regimes (D.Any D.Float64))
  @ [ empty false; empty true ]

let scaled_covers (c, sa, sb) =
  cover "float32" (c.acc = D.Any D.Float32);
  cover "float64" (c.acc = D.Any D.Float64);
  cover "chain order" (order c = Chain);
  cover "lane order" (order c = Lane);
  let specs =
    List.filter_map
      (fun (x, s) -> Option.map (fun s -> (shape_of x, s)) s)
      [ (c.a, sa); (c.b, sb) ]
  in
  let any f = List.exists f specs in
  let axes f (x, s) = List.length (List.filter Fun.id (Array.to_list (Array.map2 f x s.extents))) in
  cover "an Exponent scale" (any (fun (_, s) -> s.kind = None));
  cover "a Value scale" (any (fun (_, s) -> s.kind <> None));
  cover "a ratio of 1" (any (fun xs -> axes (fun e d -> e > 1 && d = e) xs > 0));
  cover "a whole axis" (any (fun xs -> axes (fun e d -> e > 1 && d = 1) xs > 0));
  cover "2-D blocks" (any (fun xs -> axes (fun e d -> e > 1 && d < e) xs >= 2));
  cover "a stepped scale" (any (fun (_, s) -> s.lies = Stepped));
  cover "a broadcast scale" (any (fun (_, s) -> s.lies = Broadcast));
  cover "a transposed operand" (List.mem "permuted" c.views);
  cover "a broadcast operand" (List.mem "broadcast" c.views);
  let (A.Any w) = c.b in
  let st = Iarray.to_array (L.strides (A.layout w)) in
  cover "two contracting pairs" (Array.length c.contracting = 2);
  cover "b stepped along k"
    (c.contracting <> [||] && st.(snd c.contracting.(0)) = 2);
  cover "b broadcast across its columns" (Array.length st = 2 && st.(1) = 0);
  cover "a batch axis" (c.batch <> [||]);
  let _, k, _ = sizes c in
  List.iter
    (fun n -> cover (Printf.sprintf "%d products" n) (k = n))
    [ 1; 1023; 1024; 1025; 4095; 4096; 4097 ];
  let made = List.map (fun (_, s) -> (s, scale_of s)) specs in
  let has_code e =
    List.exists
      (fun (_, sc) ->
        match sc with
        | Nx_kernel.Exponent x -> Array.mem e (A.to_array x)
        | Value _ -> false)
      made
  in
  Array.iter (fun e -> cover (Printf.sprintf "code %d" e) (has_code e)) codes;
  cover "a NaN value"
    (List.exists
       (fun (_, sc) ->
         match sc with
         | Nx_kernel.Value _ -> Array.exists Float.is_nan (decoded c.acc sc)
         | Exponent _ -> false)
       made)

(* A scaled contraction has the bits of the contraction of its operands
   scaled on the host, each element rounded once into acc. *)
let scaled_bits b (c, sa, sb) =
  let sa = Option.map scale_of sa and sb = Option.map scale_of sb in
  let scaled x = function None -> (x, false) | Some s -> host_scaled c.acc x s in
  let a, oa = scaled c.a sa and b', ob = scaled c.b sb in
  cover "a product that overflows" (oa || ob);
  match (run ?sa ?sb b c, run b { c with a; b = b' }) with
  | Some y, Some y' ->
      equal (list string) [] (differ_bits c.acc ~other:"host-scaled" y y')
  | _ -> failf "nx.cpu declined a float contraction"

let law_scaled b x =
  scaled_covers x;
  scaled_bits b x

(* Block-scaled float4

   A [m × k] by float4 [k × n] product, b scaled in blocks of [mx_ratio]
   along k and of each column or all of them ([mx_whole]), as MXFP4's
   weights are: b stored [n; k] and read along k ([mx_along]), as a
   weight is, or stored [k; n]; its codes from the second of a byte where
   [mx_odd]; a float4 too where [mx_a], scaled likewise along k. *)
type mx = {
  mx_acc : D.any;
  mx_m : int;
  mx_n : int;
  mx_k : int;
  mx_ratio : int;
  mx_whole : bool;
  mx_along : bool;
  mx_odd : bool;
  mx_a : bool;
  mx_kind : D.any option;
  mx_seed : int;
}

(* float4's values by code, e2m1: sign, two exponent bits, one of
   significand. *)
let e2m1 =
  [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6.; -0.; -0.5; -1.; -1.5; -2.; -3.; -4.; -6. |]

(* A float4 array of shape [made], its codes drawn uniformly from
   [seed]. *)
let float4 made seed =
  let r = Random.State.make [| seed |] in
  let xs = Array.init (total made) (fun _ -> e2m1.(Random.State.int r 16)) in
  A.expect D.Float4_e2m1fn
    (cast_to (D.Any D.Float4_e2m1fn)
       (A.Any (A.of_array D.Float64 (Iarray.of_array made) xs)))

(* The distinct float4 codes of [x]. *)
let f4_codes x =
  let v = A.to_array (to64 x) in
  List.length
    (List.sort_uniq compare
       (Array.to_list (Array.map (fun f -> (f, Float.sign_bit f)) v)))

(* A float4 [k × n] array drawn from [seed]: made [n; k + odd] and
   permuted where [along], else [k; n + odd], then sliced from [odd] along
   the axis whose codes are adjacent. *)
let float4_b ~along ~odd k n seed =
  let made = if along then [| n; k + odd |] else [| k; n + odd |] in
  let x = float4 made seed in
  let move m x = Option.get (A.move m x) in
  let window = Iarray.of_list
      [ { M.start = 0; count = made.(0); step = 1 };
        { M.start = odd; count = made.(1) - odd; step = 1 } ]
  in
  let x = move (M.Slice window) x in
  A.Any (if along then move (M.Permute [| 1; 0 |]) x else x)

let mx_case x =
  let k = x.mx_k and n = x.mx_n and m = x.mx_m in
  let da = if x.mx_a then D.Any D.Float4_e2m1fn else D.Any D.Bfloat16 in
  let a = operand da [| m; k |] (Elements.plain 2) x.mx_seed in
  let odd = if x.mx_odd then 1 else 0 in
  let b = float4_b ~along:x.mx_along ~odd k n (x.mx_seed + 1) in
  let c =
    { batch = [||]; contracting = [| (1, 0) |];
      acc = x.mx_acc; a; b; init = None; out = x.mx_acc; views = [] }
  in
  let blocks = if k = 0 then 0 else k / x.mx_ratio in
  let spec extents i =
    { kind = x.mx_kind; extents; lies = Contiguous; specials = true;
      seed = x.mx_seed + i }
  in
  let cols = if x.mx_whole && n > 0 then 1 else n in
  let sa = if x.mx_a then Some (spec [| m; blocks |] 2) else None in
  (c, sa, Some (spec [| blocks; cols |] 3))

let pp_mx ppf x = pp_scaled ppf (mx_case x)

let mx_of ?(acc = D.Any D.Float32) ?(ratio = 32)
    ?(whole = false) ?(along = true) ?(odd = false) ?(f4_a = false) ?kind
    ?(seed = 1) m n k =
  { mx_acc = acc; mx_m = m; mx_n = n; mx_k = k;
    mx_ratio = ratio; mx_whole = whole; mx_along = along; mx_odd = odd;
    mx_a = f4_a; mx_kind = kind; mx_seed = seed }

let mx_draws =
  let open Gen in
  with_pp pp_mx
    (let* acc = of_list ~pp:pp_dtype D.[ Any Float32; Any Float64 ] in
     let* k = of_list [ 0; 32; 64; 96; 1024; 1056; 2048; 4096; 4128 ] in
     let* ratio =
       if k = 0 then constant 32
       else of_list (List.filter (fun r -> k mod r = 0) [ 32; 64; 96 ])
     in
     let* m = frequency [ (3, int_range 1 5); (2, int_range 6 40); (1, constant 0) ] in
     let* n = frequency [ (3, int_range 1 40); (1, of_list [ 0; 1; 300 ]) ] in
     let* whole = bool and+ along = frequency [ (3, constant true); (1, constant false) ] in
     let* odd = bool and+ f4_a = frequency [ (4, constant false); (1, constant true) ] in
     let* kind =
       frequency
         [ (3, constant None);
           (1, map Option.some (of_list ~pp:pp_dtype D.[ Any Float16; Any Float32 ])) ]
     in
     let+ seed = int in
     mx_of ~acc ~ratio ~whole ~along ~odd ~f4_a ?kind ~seed m n k)

(* The product of a token by a weight, either side of lane order's blocks
   and of 4096, in both accs; a weight read along its columns; codes from
   inside a byte; float4 a; Value scales; chain order over 13 columns and
   over hundreds of rows; and the empty and one-element products. *)
let mx_examples =
  List.concat_map
    (fun acc ->
      List.map (fun k -> mx_of ~acc 1 24 k) [ 32; 1024; 1056; 4096; 4128 ]
      @ [ mx_of ~acc ~along:false ~odd:true 3 17 1056;
          mx_of ~acc ~odd:true ~ratio:64 2 9 4096;
          mx_of ~acc ~f4_a:true ~whole:true 5 7 2048;
          mx_of ~acc ~kind:(D.Any D.Float16) 4 33 1024;
          mx_of ~acc 9 13 1056;
          mx_of ~acc 1 1 32; mx_of ~acc 0 5 64;
          mx_of ~acc 3 0 64; mx_of ~acc 2 3 0 ])
    D.[ Any Float32; Any Float64 ]
  @ [ mx_of ~acc:(D.Any D.Float32) 260 264 1056;
      mx_of ~acc:(D.Any D.Float64) ~along:false 260 264 1056 ]

let mx_covers x (c, _, _) =
  cover "every float4 code" (f4_codes c.b = 16);
  let lane = order c = Lane and products = x.mx_m * x.mx_n > 0 && x.mx_k > 0 in
  cover "float32" (x.mx_acc = D.Any D.Float32);
  cover "float64" (x.mx_acc = D.Any D.Float64);
  cover "lane order, b read along k" (lane && products && x.mx_along);
  cover "lane order, b read along its columns" (lane && products && not x.mx_along);
  cover "chain order, b read along k" ((not lane) && products && x.mx_along);
  cover "chain order, b read along its columns"
    ((not lane) && products && not x.mx_along);
  cover "k past a block of lanes" (x.mx_k > 1024);
  cover "k past nx's block" (x.mx_k > 4096);
  cover "blocks of 64 or more" (x.mx_k > 0 && x.mx_ratio > 32);
  cover "codes from inside a byte" (products && x.mx_odd);
  (* The float4 stage's paths: chain order's slivers stage 8 columns' codes
     a lane, then one column at a time; lane order dots several columns
     against each row. *)
  let from_byte = products && x.mx_along && not x.mx_odd in
  List.iter
    (fun (acc, name) ->
      let on = x.mx_acc = acc in
      let chain = on && (not lane) && from_byte in
      cover (name ^ ", chain order, b along k from a byte's first, 8 columns or more")
        (chain && x.mx_n >= 8);
      cover
        (name ^ ", chain order, b along k from a byte's first, columns not a \
                 multiple of 8")
        (chain && x.mx_n mod 8 <> 0);
      cover
        (name ^ ", lane order, b along k, columns not a multiple of 4, two \
                 rows or more")
        (on && lane && products && x.mx_along && x.mx_n mod 4 <> 0 && x.mx_m >= 2))
    D.[ (Any Float32, "float32"); (Any Float64, "float64") ];
  cover "a float4 a" (products && x.mx_a);
  cover "a Value scale" (products && x.mx_kind <> None);
  cover "a scale per column" (products && x.mx_n > 1 && not x.mx_whole);
  cover "no output" (x.mx_m * x.mx_n = 0);
  cover "one output" (x.mx_m * x.mx_n = 1);
  cover "no product" (x.mx_k = 0 && x.mx_m * x.mx_n > 0)

let law_mx b x =
  let c = mx_case x in
  mx_covers x c;
  scaled_bits b c

(* Block-scaled float4 weights through other layouts, in both accs: two
   contracting pairs that merge, one apart, and pairs in another order
   than b's axes; b stepped along k, broadcast across its columns; a batch
   axis. *)
let mx_layouts =
  let move m x = A.Any (Option.get (A.move m x)) in
  let bf16 s seed = operand (D.Any D.Bfloat16) s (Elements.plain (Array.length s)) seed in
  let codes extents = Some { kind = None; extents; lies = Contiguous; specials = true; seed = 5 } in
  let range start count step = { M.start; count; step } in
  let layouts =
    [
      ([||], [| (1, 0); (2, 1) |], bf16 [| 3; 2; 64 |] 1,
       move (M.Permute [| 1; 2; 0 |]) (float4 [| 5; 2; 64 |] 2), [| 2; 2; 5 |]);
      ([||], [| (1, 0); (2, 1) |], bf16 [| 3; 2; 64 |] 1,
       move (M.Permute [| 2; 1; 0 |]) (float4 [| 5; 64; 2 |] 2), [| 2; 2; 5 |]);
      ([||], [| (2, 1); (1, 0) |],
       bf16 [| 3; 2; 64 |] 1, move (M.Permute [| 1; 2; 0 |]) (float4 [| 5; 2; 64 |] 2),
       [| 2; 2; 5 |]);
      ([||], [| (1, 0) |], bf16 [| 2; 1056 |] 1,
       move (M.Permute [| 1; 0 |])
         (Option.get
            (A.move (M.Slice [| range 0 7 1; range 1 1056 2 |]) (float4 [| 7; 2112 |] 2))),
       [| 33; 7 |]);
      ([||], [| (1, 0) |], bf16 [| 3; 1056 |] 1,
       move (M.Broadcast [| 1056; 9 |]) (float4 [| 1056; 1 |] 2), [| 33; 9 |]);
      ([| (0, 0) |], [| (2, 1) |], bf16 [| 2; 3; 128 |] 1,
       move (M.Permute [| 0; 2; 1 |]) (float4 [| 2; 6; 128 |] 2), [| 2; 4; 6 |]);
    ]
  in
  List.concat_map
    (fun (batch, contracting, a, b, extents) ->
      List.map
        (fun acc ->
          ( { batch; contracting; acc; a; b; init = None; out = acc;
              views = [] },
            None, codes extents ))
        D.[ Any Float32; Any Float64 ])
    layouts

(* Integer and complex cases with scales. *)
let unscalable =
  let open Gen in
  with_pp pp_scaled
    (let* acc =
       of_list ~pp:pp_dtype
         D.[ Any Int4; Any Int8; Any Int32; Any Int64; Any Complex64; Any Complex128 ]
     in
     let dts = if is_integer acc then integers else held acc in
     let* c = case_of ~acc ~out:acc ~dts () in
     with_scales c)

(* nx.cpu declines a scale with an integer or complex acc, and writes
   nothing. *)
let law_unscalable b (c, sa, sb) =
  cover "integer" (is_integer c.acc);
  cover "complex" (is_complex c.acc);
  let y, _, _ = sizes c in
  let threes =
    A.Any (A.of_array D.Float64 (Iarray.of_array y) (Array.make (total y) 3.))
  in
  let dst = on b (cast_to c.out threes) in
  let before = planes c.out (host dst) in
  let init = Option.map (on b) c.init in
  let sa = Option.map (fun s -> scale_on b (scale_of s)) sa in
  let sb = Option.map (fun s -> scale_on b (scale_of s)) sb in
  match contract ?sa ?sb b.kernels c ~dst (on b c.a) (on b c.b) init with
  | A.Declined ->
      equal (array (array float_exact)) before (planes c.out (host dst))
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

(* nx.cpu answers Shape_mismatch for a scale that does not fit its
   operand, a [2 × 4] by [4 × 3] product's, and writes nothing: of
   another rank, with an extent that does not divide the operand's, or of
   0 along an axis with elements. *)
let test_scale_misfits (b : Support.backend) () =
  let module K = (val b.kernels) in
  let arr s =
    let ones = Array.make (total s) 1. in
    let x = A.of_array D.Float32 (Iarray.of_array s) ones in
    A.expect D.Float32 (on b (A.Any x))
  in
  let codes s =
    let x = A.of_array D.Uint8 (Iarray.of_array s) (Array.make (total s) 127) in
    Nx_kernel.Exponent (A.expect D.Uint8 (on b (A.Any x)))
  in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let refused ~msg ?sa ?sb () =
    let dst = arr [| 2; 3 |] in
    let before = A.to_array (to64 (host (A.Any dst))) in
    let op x scale : Nx_kernel.operand = { array = A.Any x; scale } in
    List.iter
      (fun acc ->
        equal ~msg answer A.Shape_mismatch
          (K.contract ~batch:[||] ~contracting:[| (1, 0) |]
             ~segments:None ~acc ~dst ~init:None
             (op (arr [| 2; 4 |]) sa)
             (op (arr [| 4; 3 |]) sb));
        equal ~msg:(msg ^ ": untouched") (array float_exact) before
          (A.to_array (to64 (host (A.Any dst)))))
      D.[ Any Float32; Any Float64 ]
  in
  refused ~msg:"a's scale of rank 1" ~sa:(codes [| 2 |]) ();
  refused ~msg:"b's scale of rank 3" ~sb:(codes [| 1; 1; 1 |]) ();
  refused ~msg:"a's 3 blocks along 4" ~sa:(codes [| 2; 3 |]) ();
  refused ~msg:"b's 2 blocks along 3" ~sb:(codes [| 4; 2 |]) ();
  refused ~msg:"a's 4 blocks along 2" ~sa:(codes [| 4; 4 |]) ();
  refused ~msg:"b's 0 blocks along 4" ~sb:(codes [| 0; 3 |]) ();
  refused ~msg:"a fitting scale beside a misfit" ~sa:(codes [| 2; 1 |])
    ~sb:(codes [| 3; 3 |]) ()

(* Segments

   A contraction cut into groups by offsets (Nx_kernel.segments): [Rows]
   cuts a free axis of a, each row taking b's group along a free axis of
   b; [Contracting] cuts a contracting pair, each group's window summed
   alone into its slice of a leading axis. Each is checked against
   unsegmented contractions of views, one a row or a group, whose bits
   nx.cpu states: a group sums as the call of its extents does. *)

(* What a segmented contraction cuts: a's free axis and b's, or a
   contracting pair. *)
type cut = Cut_rows of { a_axis : int; b_axis : int } | Cut_pair of int

(* A segmented contraction: [sg]'s order, pairs, dtypes and operands, its
   init of the segmented result's shape; [g] groups cut as [cut] says by
   the offsets [offs], [g + 1] for each batch element in C order, read
   through a stepped view where [stepped]; its operands' scales. [min_int]
   and [max_int] among [offs] stand for int64's (wide). *)
type seg = {
  sg : case;
  cut : cut;
  g : int;
  offs : int array array;
  stepped : bool;
  sa : spec option;
  sb : spec option;
}

(* The int64 offset [x] stands for. *)
let wide x =
  if x = min_int then Int64.min_int
  else if x = max_int then Int64.max_int
  else Int64.of_int x

let pp_seg ppf s =
  let cut ppf = function
    | Cut_rows { a_axis; b_axis } ->
        Format.fprintf ppf "rows of a's axis %d by b's axis %d" a_axis b_axis
    | Cut_pair p -> Format.fprintf ppf "contracting pair %d" p
  in
  let sep ppf () = Format.pp_print_string ppf " " in
  let offs ppf o =
    Format.fprintf ppf "[%s]"
      (String.concat "; " (Array.to_list (Array.map (fun x -> Int64.to_string (wide x)) o)))
  in
  Format.fprintf ppf "%a@ %a, %d groups, offsets %a%s" pp_scaled
    (s.sg, s.sa, s.sb) cut s.cut s.g
    (Format.pp_print_list ~pp_sep:sep offs)
    (Array.to_list s.offs)
    (if s.stepped then " stepped" else "")

(* The batch extents: a's along the batch pairs. *)
let batch_extents c = Array.map (fun (i, _) -> (shape_of c.a).(i)) c.batch

(* The extent of the axis [s] cuts. *)
let cut_extent s =
  let a = shape_of s.sg.a in
  match s.cut with
  | Cut_rows { a_axis; _ } -> a.(a_axis)
  | Cut_pair p -> a.(fst s.sg.contracting.(p))

let segments_of s offsets : _ Nx_kernel.segments =
  match s.cut with
  | Cut_rows { a_axis; b_axis } -> Rows { offsets; a_axis; b_axis }
  | Cut_pair pair -> Contracting { offsets; pair }

(* [s]'s offsets, int64 of the batch extents then [g + 1]: where
   [stepped], every other element of an array twice as long along its
   last axis. *)
let offsets_of s =
  let shape = Array.append (batch_extents s.sg) [| s.g + 1 |] in
  let xs = Array.concat (Array.to_list s.offs) in
  if not s.stepped then
    A.of_array D.Int64 (Iarray.of_array shape) (Array.map wide xs)
  else
    let r = Array.length shape and w = s.g + 1 in
    let made = Array.mapi (fun i e -> if i = r - 1 then 2 * e else e) shape in
    let at q = if q mod 2 = 0 then 999L else wide xs.((q / (2 * w) * w) + (q mod (2 * w) / 2)) in
    let x =
      A.of_array D.Int64 (Iarray.of_array made) (Array.init (total made) at)
    in
    let step i e =
      if i = r - 1 then { M.start = 1; count = e; step = 2 }
      else { M.start = 0; count = e; step = 1 }
    in
    Option.get (A.move (M.Slice (Iarray.of_array (Array.mapi step shape))) x)

let seg_shape s =
  let c = s.sg in
  let o = Array.append (batch_extents c) [| s.g + 1 |] in
  match
    Nx_kernel.Shape.contract ~batch:(Iarray.of_array c.batch)
      ~contracting:(Iarray.of_array c.contracting)
      ~segments:(Some (segments_of s o)) (shape_of c.a, None)
      (shape_of c.b, None)
  with
  | Ok y -> y
  | Error e -> failf "the segments do not fit: %s" e

(* The group of index [r] under the offsets [o] of [g] groups: the
   largest [e < g] with [o.(e) <= r] when [o.(0) <= r < o.(g)], the groups
   Nx_kernel.segments states by another rule. *)
let group_of o g r =
  if g = 0 || r < o.(0) || r >= o.(g) then None
  else
    let e = ref (g - 1) in
    while o.(!e) > r do
      decr e
    done;
    Some !e

(* The indices of the cut axis in group [e] under [o]. *)
let members s o e =
  List.filter (fun r -> group_of o s.g r = Some e) (List.init (cut_extent s) Fun.id)

(* [x] cut along each axis [i] of [cuts] to [count] indices from
   [start]. *)
let restrict (A.Any x as a) cuts =
  let r = Array.map (fun e -> { M.start = 0; count = e; step = 1 }) (shape_of a) in
  List.iter (fun (i, start, count) -> r.(i) <- { M.start; count; step = 1 }) cuts;
  A.Any (Option.get (A.move (M.Slice (Iarray.of_array r)) x))

let reshaped (A.Any x) s =
  A.Any (Option.get (A.move (M.Reshape (Iarray.of_array s)) x))

(* The axes of a side of rank [rank] that no pair of [c] names. *)
let free side c rank =
  let paired i =
    Array.exists (fun p -> side p = i) c.batch
    || Array.exists (fun p -> side p = i) c.contracting
  in
  List.filter (fun i -> not (paired i)) (List.init rank Fun.id)

let before l x = List.length (List.filter (fun i -> i < x) l)

(* nx.cpu's order for [s], chosen once for the call from its rows, columns
   and depth, [Rows]' columns those of one index along [b_axis]
   (nx_cpu.mli). *)
let seg_order s =
  let c = s.sg in
  let m, n = rows_cols c and _, k, _ = sizes c in
  match s.cut with
  | Cut_rows _ when s.g > 0 -> order_of c.acc m (n / s.g) k
  | Cut_rows _ | Cut_pair _ -> order_of c.acc m n k

(* [s]'s result by views of the operands scaled on the host: each row in a
   group alone with its group of b, or each group's window alone, summed
   in [s]'s order ({!ordered}), an integer one by an unsegmented
   contraction; init as acc holds it, or +0, at every other output. *)
let seg_reference (b : Support.backend) s sa sb =
  let module K = (val b.kernels) in
  let c = s.sg in
  let scaled x = function None -> x | Some sc -> fst (host_scaled c.acc x sc) in
  let a = scaled c.a sa and b' = scaled c.b sb in
  let y = seg_shape s in
  let ys = Iarray.of_array y in
  let start =
    match c.init with
    | Some i -> cast_to c.acc i
    | None -> A.Any (A.of_array D.Float64 ys (Array.make (total y) 0.))
  in
  let dst = cast_to c.out start in
  let nb = Array.length c.batch in
  let chain = seg_order s = Chain in
  let model a b init =
    let y, each = outputs { c with a; b; init; views = [] } in
    let parts = if is_complex c.acc then 2 else 1 in
    let v = Array.init parts (fun _ -> Array.make (total y) 0.) in
    each (fun _ k parts ->
        Array.iteri (fun p o -> v.(p).(k) <- ordered ~chain c.acc o) parts);
    let ys = Iarray.of_array y in
    if Array.length v = 1 then A.Any (A.of_array D.Float64 ys v.(0))
    else
      A.Any
        (A.of_array D.Complex128 ys
           (Array.map2 (fun re im -> { Complex.re; im }) v.(0) v.(1)))
  in
  let call (A.Any d) a b init =
    if not (is_integer c.acc) then cast_into d (model a b init)
    else
      let init = Option.map (A.expect (A.dtype d)) init in
      match
        K.contract ~batch:(Iarray.of_array c.batch)
          ~contracting:(Iarray.of_array c.contracting) ~segments:None
          ~acc:c.acc ~dst:d ~init
          { array = a; scale = None }
          { array = b; scale = None }
      with
      | A.Done -> ()
      | r ->
          failf "an unsegmented contraction answered %a"
            Nx_array_support.pp_answer r
  in
  let along side beta = List.init nb (fun k -> (side c.batch.(k), beta.(k), 1)) in
  let at shift beta = List.init nb (fun k -> (k + shift, beta.(k), 1)) in
  let row beta o =
    match s.cut with
    | Cut_rows { a_axis; b_axis } ->
        let fa = free fst c (Array.length (shape_of c.a)) in
        let fb = free snd c (Array.length (shape_of c.b)) in
        let ya = nb + before fa a_axis in
        let yb = nb + List.length fa + before fb b_axis in
        (* dst's row [r], with an axis of 1 where b's group was. *)
        let lift r x =
          let x = restrict x ((ya, r, 1) :: at 0 beta) in
          let sh = shape_of x in
          reshaped x
            (Array.concat
               [ Array.sub sh 0 yb; [| 1 |]; Array.sub sh yb (Array.length sh - yb) ])
        in
        for r = 0 to cut_extent s - 1 do
          match group_of o s.g r with
          | None -> ()
          | Some e ->
              call (lift r dst)
                (restrict a ((a_axis, r, 1) :: along fst beta))
                (restrict b' ((b_axis, e, 1) :: along snd beta))
                (Option.map (lift r) c.init)
        done
    | Cut_pair p ->
        let ia, ib = c.contracting.(p) in
        (* dst's slice [e], without the groups' axis. *)
        let lift e x =
          let x = restrict x ((0, e, 1) :: at 1 beta) in
          let sh = shape_of x in
          reshaped x (Array.sub sh 1 (Array.length sh - 1))
        in
        for e = 0 to s.g - 1 do
          let lo, count =
            match members s o e with [] -> (0, 0) | r :: _ as l -> (r, List.length l)
          in
          call (lift e dst)
            (restrict a ((ia, lo, count) :: along fst beta))
            (restrict b' ((ib, lo, count) :: along snd beta))
            (Option.map (lift e) c.init)
        done
  in
  List.iteri (fun q beta -> row beta s.offs.(q)) (indices (batch_extents c));
  dst

(* Offsets of [g] groups along [n] indices, from [r]: a partition of the
   axis; increasing within it; every index in one group; or any values
   from -2 to [n + 2] and int64's extremes, unordered. *)
let draw_offsets r g n =
  let sorted () =
    let o = Array.init (g + 1) (fun _ -> Random.State.int r (n + 1)) in
    Array.sort compare o;
    o
  in
  match Random.State.int r 4 with
  | 0 ->
      let o = sorted () in
      o.(0) <- 0;
      o.(g) <- n;
      o
  | 1 -> sorted ()
  | 2 ->
      let e = Random.State.int r (max g 1) in
      Array.init (g + 1) (fun i -> if i <= e then 0 else n)
  | _ ->
      let any _ =
        match Random.State.int r (n + 7) with
        | x when x = n + 5 -> min_int
        | x when x = n + 6 -> max_int
        | x -> x - 2
      in
      Array.init (g + 1) any

(* The segmented contraction of the extents [e] as [make] makes it, cut by
   [cut], positions among [e]'s rows and columns or its contracted axes,
   with an init of the segmented result's shape through [vi]. *)
let seg_make ~acc ~out ((batch, _, _, con) as e) (da, va) (db, vb) vi
    ~cut ~g ?(stepped = false) offs seed =
  let c = make ~acc ~out e (da, va) (db, vb) None seed in
  let nb = Array.length batch and nk = Array.length con in
  let ia = Elements.inverse va.perm and ib = Elements.inverse vb.perm in
  let cut =
    match cut with
    | `Rows (ri, ci) -> Cut_rows { a_axis = ia.(nb + ri); b_axis = ib.(nb + nk + ci) }
    | `Pair p -> Cut_pair p
  in
  let s = { sg = c; cut; g; offs; stepped; sa = None; sb = None } in
  match vi with
  | None -> s
  | Some vi ->
      let y = seg_shape s in
      let init =
        operand c.out (Array.map (fun i -> y.(i)) (Elements.inverse vi.perm)) vi (seed + 2)
      in
      { s with sg = { c with init = Some init; views = c.views @ Elements.view_names vi } }

(* Draws: rows or a pair cut into up to 6 groups along up to 40 indices,
   or a pair of 4090 to 4200 products, over a batch of up to two axes
   whose elements draw their own offsets, of float accs mostly, with
   scales where acc is a float. *)
let seg_draws =
  let open Gen in
  let axes most lo hi = array ~size:(int_range 0 most) (int_range lo hi) in
  let put x i v =
    Array.concat [ Array.sub x 0 i; [| v |]; Array.sub x i (Array.length x - i) ]
  in
  with_pp pp_seg
    (let* acc =
       frequency
         [ (4, of_list ~pp:pp_dtype D.[ Any Float32; Any Float64 ]);
           (1, of_list ~pp:pp_dtype D.[ Any Complex64; Any Int32 ]) ]
     in
     let float = not (is_integer acc || is_complex acc) in
     let dts = if is_integer acc then integers else held acc in
     let pick = frequency [ (3, constant ~pp:pp_dtype acc); (1, of_list ~pp:pp_dtype dts) ] in
     let* da = pick and+ db = pick in
     let* out =
       if float then frequency [ (3, constant ~pp:pp_dtype acc); (1, of_list ~pp:pp_dtype narrow) ]
       else constant ~pp:pp_dtype acc
     in
     let* g = frequency [ (1, constant 0); (2, constant 1); (6, int_range 2 6) ] in
     let* n = frequency [ (5, int_range 0 12); (2, int_range 13 40) ] in
     let* batch = axes 2 1 3 and+ rows_cut = bool in
     let* rows, cols, con, cut =
       if rows_cut then
         let* rows = axes 1 1 3 and+ cols = axes 1 0 4 in
         let* con =
           frequency [ (4, axes 2 0 5); (1, map (fun k -> [| k |]) (int_range 1000 1100)) ]
         in
         let+ ri = int_range 0 (Array.length rows)
         and+ ci = int_range 0 (Array.length cols) in
         (put rows ri n, put cols ci g, con, `Rows (ri, ci))
       else
         let* rows = axes 2 1 4 and+ cols = axes 2 1 4 and+ con = axes 1 1 3 in
         let* k = frequency [ (5, constant n); (1, int_range 4090 4200) ] in
         let+ p = int_range 0 (Array.length con) in
         (rows, cols, put con p k, `Pair p)
     in
     let e = (batch, rows, cols, con) in
     let nb = Array.length batch in
     let ra, rb, ry = ranks e in
     let ry = match cut with `Rows _ -> ry - 1 | `Pair _ -> ry + 1 in
     let* va = view ra nb and+ vb = view rb nb and+ vi = view ry 0 in
     let* with_init = bool and+ stepped = bool and+ seed = int and+ oseed = int in
     let r = Random.State.make [| oseed |] in
     let along = match cut with `Rows (ri, _) -> rows.(ri) | `Pair p -> con.(p) in
     let offs = Array.init (total batch) (fun _ -> draw_offsets r g along) in
     let s =
       seg_make ~acc ~out e (da, va) (db, vb)
         (if with_init then Some vi else None)
         ~cut ~g ~stepped offs seed
     in
     if not float then constant s
     else
       let* sa = spec_for (shape_of s.sg.a) and+ sb = spec_for (shape_of s.sg.b) in
       let+ which = of_list [ `None; `None; `A; `B; `Both ] in
       match which with
       | `None -> s
       | `A -> { s with sa = Some sa }
       | `B -> { s with sb = Some sb }
       | `Both -> { s with sa = Some sa; sb = Some sb })

(* A scale of codes of [extents]. *)
let codes_of extents seed =
  Some { kind = None; extents; lies = Contiguous; specials = true; seed }

(* Experts: [t] bfloat16 tokens of [k] by float4 weights [g; n; k] of [g]
   experts, stored along k, scaled in blocks of 32 along it as MXFP4's
   are, the tokens cut into the experts' groups by [o]. *)
let experts ?(acc = D.Any D.Float32) t g n k o =
  let a = operand (D.Any D.Bfloat16) [| t; k |] (Elements.plain 2) 7 in
  let b = operand (D.Any D.Float4_e2m1fn) [| g; n; k |] (Elements.plain 3) 8 in
  let c =
    { batch = [||]; contracting = [| (1, 2) |]; acc; a; b; init = None;
      out = acc; views = [] }
  in
  { sg = c; cut = Cut_rows { a_axis = 0; b_axis = 0 }; g; offs = [| o |];
    stepped = false; sa = None; sb = codes_of [| g; n; k / 32 |] 9 }

(* [m] bfloat16 rows of [k] by float4 [k; n] stored along k, scaled in
   blocks of 32 along it, k cut into windows by [o]. *)
let windows ?(acc = D.Any D.Float32) m n k o =
  let a = operand (D.Any D.Bfloat16) [| m; k |] (Elements.plain 2) 3 in
  let along = { (Elements.plain 2) with stored = [| 1; 0 |] } in
  let b = operand (D.Any D.Float4_e2m1fn) [| k; n |] along 4 in
  let c =
    { batch = [||]; contracting = [| (1, 0) |]; acc; a; b; init = None;
      out = acc; views = [] }
  in
  { sg = c; cut = Cut_pair 0; g = Array.length o - 1; offs = [| o |];
    stepped = false; sa = None; sb = codes_of [| k / 32; n |] 5 }

(* Cases each cover names, in both orders and accs, with the MXFP4 weights
   of experts and windows of k either side of nx's block, within and on
   the scales' blocks. *)
let seg_examples =
  let f32 = D.Any D.Float32 and f64 = D.Any D.Float64 in
  let ex ?(acc = f32) ?out ?(da = acc) ?(db = acc) ?(view = false)
      ?(init = false) ?stepped ?sa ?sb ~cut ~g e offs =
    let (batch, _, _, _) = e in
    let nb = Array.length batch in
    let ra, rb, ry = ranks e in
    let ry = match cut with `Rows _ -> ry - 1 | `Pair _ -> ry + 1 in
    (* Axes reversed, the last made axis stepped, broadcast along the
       [batch] leading made axes. *)
    let v ?(batch = nb) r =
      if not view then Elements.plain r
      else
        let p = Array.init r (fun i -> r - 1 - i) in
        let axes = List.init r Fun.id in
        { (Elements.plain r) with stepped = true; perm = p;
          broadcast = List.filter (fun i -> p.(i) < batch) axes }
    in
    let out = Option.value out ~default:acc in
    let s =
      seg_make ~acc ~out e (da, v ra) (db, v rb)
        (if init then Some (v ~batch:0 ry) else None)
        ~cut ~g ?stepped offs 11
    in
    let blocks x d = Array.map (fun e -> if e mod d = 0 then e / d else e) (shape_of x) in
    let spec x = function
      | None -> None
      | Some d -> Some { kind = Some (D.Any D.Float16); extents = blocks x d; lies = Stepped; specials = true; seed = d }
    in
    { s with sa = spec s.sg.a sa; sb = spec s.sg.b sb }
  in
  [
    ex ~init:true ~cut:(`Rows (0, 0)) ~g:3 ([||], [| 20 |], [| 3; 4 |], [| 7 |])
      [| [| 2; 5; 5; 18 |] |];
    (* Calls in chain order and in lane order by their depth whose groups
       alone would sum in the other. *)
    ex ~cut:(`Rows (0, 0)) ~g:3 ([||], [| 20 |], [| 3; 5 |], [| 7 |])
      [| [| 2; 5; 5; 18 |] |];
    ex ~cut:(`Pair 0) ~g:2 ([||], [| 8 |], [| 8 |], [| 8200 |])
      [| [| 0; 100; 8200 |] |];
    ex ~acc:f64 ~view:true ~stepped:true ~cut:(`Rows (0, 0)) ~g:2
      ([| 2 |], [| 40 |], [| 2 |], [| 5; 3 |])
      [| [| 0; 3; 40 |]; [| 0; 40; 40 |] |];
    ex ~out:(D.Any D.Float16) ~init:true ~sa:2 ~cut:(`Rows (1, 0)) ~g:4
      ([||], [| 2; 9 |], [| 4 |], [| 1030 |])
      [| [| 5; 2; 9; 7; 3 |] |];
    ex ~acc:f64 ~sb:2 ~cut:(`Rows (0, 1)) ~g:5 ([||], [| 6 |], [| 2; 5 |], [| 4 |])
      [| [| -2; 4; 11; 4; 12; 3 |] |];
    ex ~init:true ~cut:(`Rows (0, 0)) ~g:3 ([||], [| 6 |], [| 3; 2 |], [| 5 |])
      [| [| min_int; 2; max_int; 5 |] |];
    (* a's blocks of 4 rows and b's of 2 groups, cut at rows 2, 7 and 9. *)
    (let s =
       ex ~init:true ~cut:(`Rows (0, 0)) ~g:4 ([||], [| 12 |], [| 4 |], [| 64 |])
         [| [| 0; 2; 7; 9; 12 |] |]
     in
     let spec extents seed =
       Some { kind = Some (D.Any D.Float16); extents; lies = Stepped; specials = true; seed }
     in
     { s with sa = spec [| 3; 2 |] 4; sb = spec [| 2; 2 |] 2 });
    ex ~init:true ~cut:(`Rows (0, 0)) ~g:0 ([||], [| 3 |], [| 0 |], [| 2 |]) [| [| 1 |] |];
    ex ~acc:(D.Any D.Complex64) ~init:true ~cut:(`Rows (0, 0)) ~g:2
      ([| 2 |], [| 5 |], [| 2 |], [| 3 |])
      [| [| 1; 1; 4 |]; [| 0; 2; 5 |] |];
    ex ~cut:(`Pair 0) ~g:3 ([||], [| 2 |], [| 3 |], [| 4200 |])
      [| [| 10; 1000; 4097; 4200 |] |];
    ex ~acc:f64 ~init:true ~view:true ~sa:3 ~sb:3 ~cut:(`Pair 1) ~g:2
      ([| 3 |], [| 2 |], [| 2; 2 |], [| 3; 30 |])
      [| [| 0; 0; 30 |]; [| 0; 1; 30 |]; [| 31; -1; 2 |] |];
    ex ~out:(D.Any D.Bfloat16) ~init:true ~cut:(`Pair 0) ~g:0
      ([||], [| 2 |], [| 2 |], [| 5 |]) [| [| 5 |] |];
    ex ~cut:(`Pair 0) ~g:3 ([||], [| 2 |], [| 3 |], [| 6 |])
      [| [| 1; max_int; 3; max_int |] |];
    ex ~acc:(D.Any D.Int32) ~da:(D.Any D.Int8) ~init:true ~cut:(`Pair 0) ~g:3
      ([||], [| 3 |], [| 2 |], [| 9 |])
      [| [| 0; 4; 4; 9 |] |];
    experts 37 5 9 1056 [| 0; 7; 7; 20; 36; 37 |];
    experts ~acc:f64 60 4 6 64 [| 0; 3; 43; 43; 50 |];
    experts 1 4 300 4096 [| 0; 0; 1; 1; 1 |];
    windows 3 5 4160 [| 0; 16; 1040; 4100; 4160 |];
    windows ~acc:f64 3 5 4160 [| 0; 32; 1056; 4128; 4160 |];
    windows 2 4 4160 [| 0; 16; 1040; 4100; 4160 |];
    windows ~acc:f64 2 4 4160 [| 0; 32; 1056; 4128; 4160 |];
  ]

let seg_covers s =
  let c = s.sg and n = cut_extent s in
  let rows = match s.cut with Cut_rows _ -> true | Cut_pair _ -> false in
  cover "rows" rows;
  cover "a contracting pair" (not rows);
  cover "float32" (c.acc = D.Any D.Float32);
  cover "float64" (c.acc = D.Any D.Float64);
  cover "a complex accumulator" (is_complex c.acc);
  cover "an integer accumulator" (is_integer c.acc);
  let some f = Array.exists f s.offs in
  let groups o f = List.exists (fun e -> f e (members s o e)) (List.init s.g Fun.id) in
  cover "no group" (s.g = 0);
  cover "an empty group" (some (fun o -> groups o (fun _ m -> m = [])));
  cover "a group of one" (some (fun o -> groups o (fun _ m -> List.length m = 1)));
  cover "every index in one group"
    (n > 1 && some (fun o -> groups o (fun _ m -> List.length m = n)));
  cover "offsets at the axis's bounds" (s.g > 0 && some (fun o -> o.(0) = 0 && o.(s.g) = n));
  cover "indices before the first group" (n > 0 && some (fun o -> o.(0) > 0));
  cover "indices past the last group" (some (fun o -> o.(s.g) < n && o.(s.g) > 0));
  cover "offsets that decrease"
    (some (fun o -> Array.exists Fun.id (Array.init s.g (fun e -> o.(e + 1) < o.(e)))));
  cover "offsets outside the axis" (some (Array.exists (fun x -> x < 0 || x > n)));
  cover "offsets at int64's extremes"
    (some (Array.exists (fun x -> x = min_int || x = max_int)));
  cover "offsets differing across the batch"
    (Array.exists (fun o -> o <> s.offs.(0)) s.offs);
  cover "stepped offsets" s.stepped;
  cover "an init" (c.init <> None);
  cover "a result of another dtype" (c.out <> c.acc);
  cover "an operand through a view" (c.views <> []);
  cover "a scaled operand" (s.sa <> None || s.sb <> None);
  let float4 = function
    | A.Any x -> D.Any (A.dtype x) = D.Any D.Float4_e2m1fn
  in
  cover "a block-scaled float4 operand" (s.sb <> None && float4 c.b);
  let starts f = some (fun o -> groups o (fun _ m -> match m with [] -> false | lo :: _ -> f lo (List.length m))) in
  cover "a group past a tile of 16 rows" (rows && starts (fun lo k -> lo mod 16 <> 0 && k > 16));
  cover "a window across nx's block of 4096 products"
    ((not rows) && starts (fun lo k -> lo < 4096 && lo + k > 4096));
  cover "a window across lane order's block of 1024"
    ((not rows) && starts (fun lo k -> lo < 1024 && lo + k > 1024));
  let ratio = match s.sb with
    | Some sp when not rows -> (
        let ib = snd c.contracting.(match s.cut with Cut_pair p -> p | _ -> 0) in
        let e = (shape_of c.b).(ib) and d = sp.extents.(ib) in
        if d = 0 then 1 else e / d)
    | _ -> 1
  in
  cover "a window from within a scale's block"
    (ratio > 1 && starts (fun lo _ -> lo mod ratio <> 0));
  let row_ratio = match (s.sa, s.cut) with
    | Some sp, Cut_rows { a_axis; _ } ->
        let d = sp.extents.(a_axis) in
        if d = 0 then 1 else (shape_of c.a).(a_axis) / d
    | _ -> 1
  in
  cover "a group of rows from within a scale's block"
    (row_ratio > 1 && starts (fun lo _ -> lo mod row_ratio <> 0));
  cover "windows on a scale's blocks" (ratio >= 32 && starts (fun lo _ -> lo > 0 && lo mod ratio = 0));
  let float = not (is_integer c.acc) and chain = seg_order s = Chain in
  cover "a float call in chain order" (float && chain);
  cover "a float call in lane order" (float && not chain);
  let few _ m = m <> [] && List.length m <= 4 in
  cover "a group of 4 rows or fewer in a call in chain order"
    (rows && float && chain && some (fun o -> groups o few))

(* Each output of a segmented contraction adds its group's products in the
   order nx.cpu chooses for the whole call (seg_reference). *)
let law_segments (b : Support.backend) s =
  seg_covers s;
  let c = s.sg in
  let sa = Option.map scale_of s.sa and sb = Option.map scale_of s.sb in
  let (D.Any out) = c.out in
  let dst = A.create b.device out (Iarray.of_array (seg_shape s)) in
  let module K = (val b.kernels) in
  let init = Option.map (fun i -> A.expect out (on b i)) c.init in
  let offsets = A.expect D.Int64 (on b (A.Any (offsets_of s))) in
  let sa' = Option.map (scale_on b) sa and sb' = Option.map (scale_on b) sb in
  match
    K.contract ~batch:(Iarray.of_array c.batch)
      ~contracting:(Iarray.of_array c.contracting)
      ~segments:(Some (segments_of s offsets)) ~acc:c.acc ~dst ~init
      { array = on b c.a; scale = sa' }
      { array = on b c.b; scale = sb' }
  with
  | A.Done ->
      equal (list string) []
        (differ_bits c.acc ~other:"expected" (host (A.Any dst))
           (seg_reference b s sa sb))
  | r -> failf "a segmented contraction answered %a" Nx_array_support.pp_answer r

(* nx.cpu answers Shape_mismatch for segments that do not fit, and writes
   nothing: offsets of another shape than the batch extents then [g + 1],
   a cut axis in a pair or past its operand's rank, a pair past the pairs,
   a result of another shape than the segmented one. No offsets' values
   are refused: any values name their groups. *)
let test_segment_misfits (b : Support.backend) () =
  let module K = (val b.kernels) in
  let arr s =
    let ones = Array.make (total s) 1. in
    let x = A.of_array D.Float32 (Iarray.of_array s) ones in
    A.expect D.Float32 (on b (A.Any x))
  in
  let offs s =
    A.expect D.Int64
      (on b
         (A.Any
            (A.of_array D.Int64 (Iarray.of_array s)
               (Array.init (total s) Int64.of_int))))
  in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let refused ~msg ?(batch : _ iarray = [||])
      ?(contracting : _ iarray = [| (1, 0) |]) segments dst a b' =
    let before = A.to_array (to64 (host (A.Any dst))) in
    equal ~msg answer A.Shape_mismatch
      (K.contract ~batch ~contracting ~segments:(Some segments)
         ~acc:(D.Any D.Float32) ~dst ~init:None (plain a) (plain b'));
    equal ~msg:(msg ^ ": untouched") (array float_exact) before
      (A.to_array (to64 (host (A.Any dst))))
  in
  let rows ?(a_axis = 0) ?(b_axis = 1) offsets =
    Nx_kernel.Rows { offsets; a_axis; b_axis }
  in
  let pair ?(pair = 0) offsets = Nx_kernel.Contracting { offsets; pair } in
  (* [2; 4] by [4; 3; 3] cut along b's axis 1 into 3 groups: [2; 3]. *)
  let a = arr [| 2; 4 |] and b3 = arr [| 4; 3; 3 |] and y = arr [| 2; 3 |] in
  refused ~msg:"offsets of rank 2" (rows (offs [| 1; 4 |])) y a b3;
  refused ~msg:"3 offsets for 3 groups" (rows (offs [| 3 |])) y a b3;
  refused ~msg:"a's axis in a pair" (rows ~a_axis:1 (offs [| 4 |])) y a b3;
  refused ~msg:"a's axis past its rank" (rows ~a_axis:2 (offs [| 4 |])) y a b3;
  refused ~msg:"a's axis -1" (rows ~a_axis:(-1) (offs [| 4 |])) y a b3;
  refused ~msg:"b's axis in a pair" (rows ~b_axis:0 (offs [| 4 |])) y a b3;
  refused ~msg:"b's axis past its rank" (rows ~b_axis:3 (offs [| 4 |])) y a b3;
  refused ~msg:"a result keeping b's cut axis" (rows (offs [| 4 |]))
    (arr [| 2; 3; 3 |]) a b3;
  (* [2; 4] by [4; 3] cut along the pair into 2 groups: [2; 2; 3]. *)
  let b2 = arr [| 4; 3 |] in
  refused ~msg:"no offset" (pair (offs [| 0 |])) (arr [| 0; 2; 3 |]) a b2;
  refused ~msg:"a pair past the pairs" (pair ~pair:1 (offs [| 3 |]))
    (arr [| 2; 2; 3 |]) a b2;
  refused ~msg:"pair -1" (pair ~pair:(-1) (offs [| 3 |])) (arr [| 2; 2; 3 |]) a b2;
  refused ~msg:"a result without the groups' axis" (pair (offs [| 3 |])) y a b2;
  refused ~msg:"a groups' axis of 3 for 2 groups" (pair (offs [| 3 |]))
    (arr [| 3; 2; 3 |]) a b2;
  (* [2; 2; 4] by [2; 4; 3] over a batch of 2: [2; 2; 2; 3]. *)
  let batch : _ iarray = [| (0, 0) |] and contracting : _ iarray = [| (2, 1) |] in
  refused ~msg:"offsets without the batch's extents" ~batch ~contracting
    (pair (offs [| 3 |])) (arr [| 2; 2; 2; 3 |]) (arr [| 2; 2; 4 |]) (arr [| 2; 4; 3 |]);
  refused ~msg:"offsets of another batch extent" ~batch ~contracting
    (pair (offs [| 3; 3 |])) (arr [| 2; 2; 2; 3 |]) (arr [| 2; 2; 4 |]) (arr [| 2; 4; 3 |])

(* A row of [Rows] in no group has no products: [init] as [acc] holds it,
   rounded into [dst], or [+0]. A [4; 2] by [1; 2] of zeros, cut by
   [1; 3]: rows 0 and 3 in no group, rows 1 and 2 adding zeros to init. *)
let test_rows_in_none (b : Support.backend) () =
  let module K = (val b.kernels) in
  let run (type s) (dt : (float, s) D.t) acc init =
    let arr s xs = A.expect dt (on b (A.Any (A.of_array dt s xs))) in
    let dst = arr [| 4 |] (Array.make 4 7.) in
    let offsets =
      A.expect D.Int64 (on b (A.Any (A.of_array D.Int64 [| 2 |] [| 1L; 3L |])))
    in
    (match
       K.contract ~batch:[||] ~contracting:[| (1, 1) |]
         ~segments:(Some (Nx_kernel.Rows { offsets; a_axis = 0; b_axis = 0 }))
         ~acc ~dst ~init:(Option.map (arr [| 4 |]) init)
         (plain (arr [| 4; 2 |] (Array.make 8 0.)))
         (plain (arr [| 1; 2 |] [| 0.; 0. |]))
     with
    | A.Done -> ()
    | r -> failf "contract answered %a" Nx_array_support.pp_answer r);
    A.to_array (to64 (host (A.Any dst)))
  in
  let xs = [| 300.; -300.5; 7.9; 0.1 |] in
  let int8 x = Float.of_int (D.of_float D.Int8 (D.of_float D.Float32 x)) in
  equal ~msg:"int8 acc into float32" (array float_exact) (Array.map int8 xs)
    (run D.Float32 (D.Any D.Int8) (Some xs));
  let xs = [| 0.1; 1e40; 3.; -1e-50 |] in
  equal ~msg:"float32 acc into float64" (array float_exact)
    (Array.map (D.of_float D.Float32) xs)
    (run D.Float64 (D.Any D.Float32) (Some xs));
  equal ~msg:"no init" (array float_exact) (Array.make 4 0.)
    (run D.Float32 (D.Any D.Float32) None)

(* nx.cpu reads the offsets whole before any write, so they may be [dst]
   itself: a [3; 2] by [2; 2] cut by dst's [0; 1; 2], row 0 in group 0,
   row 1 in group 1, row 2 in none. *)
let test_offsets_in_dst (b : Support.backend) () =
  let module K = (val b.kernels) in
  let arr s xs = A.expect D.Int64 (on b (A.Any (A.of_array D.Int64 s xs))) in
  let dst = arr [| 3 |] [| 0L; 1L; 2L |] in
  (match
     K.contract ~batch:[||] ~contracting:[| (1, 1) |]
       ~segments:(Some (Nx_kernel.Rows { offsets = dst; a_axis = 0; b_axis = 0 }))
       ~acc:(D.Any D.Int64) ~dst ~init:None
       (plain (arr [| 3; 2 |] [| 1L; 2L; 3L; 4L; 5L; 6L |]))
       (plain (arr [| 2; 2 |] [| 10L; 20L; 100L; 200L |]))
   with
  | A.Done -> ()
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r);
  equal (array int64) [| 50L; 1100L; 0L |]
    (A.to_array (A.expect D.Int64 (host (A.Any dst))))

(* Guard pages *)

(* A [m × k] by [k × n] product whose operands and dst each lie against an
   inaccessible page on [side]: b a window of columns [c0, c0 + n) of a
   [k × cols] array, so that its rows lie off any alignment, and the
   window ends at the page where [c0 + n = cols]. Slivers of a matrix
   unit's half and whole tiles (16 and 32 float32 columns, 8 and 16
   float64) and rows past one tile are drawn often. With [heads] 2, a
   batch of 2 × 2 products whose operands are stored [2; m; 2; k] and
   [2; k; 2; cols] and read [2; 2; m; k] and [2; 2; k; cols], as
   attention's heads are, into a C-contiguous [2; 2; m; n]: the batch
   makes two runs, and the product loops over the outer one. *)
type guard_case = {
  f64 : bool;
  heads : int;
  side : Support.side;
  m : int;
  n : int;
  k : int;
  cols : int;
  c0 : int;
  seed : int;
}

let pp_guard ppf g =
  Format.fprintf ppf "%s %s heads %d m %d n %d k %d cols %d c0 %d seed %d"
    (if g.f64 then "float64" else "float32")
    (match g.side with Support.Before -> "before" | After -> "after")
    g.heads g.m g.n g.k g.cols g.c0 g.seed

let guard_cases =
  let open Gen in
  Gen.with_pp pp_guard
    (let* f64 = bool in
     let* heads = frequency [ (2, constant 1); (1, constant 2) ] in
     let half = if f64 then 8 else 16 in
     let* after = bool in
     let side = if after then Support.After else Support.Before in
     let* m = frequency [ (1, int_range 1 32); (2, int_range 33 72) ] in
     let* k = int_range 1 140 in
     let* tiles = int_range 0 2 in
     let* edge =
       frequency
         [ (2, constant half); (1, int_range 1 (2 * half - 1)); (1, constant 0) ]
     in
     let n = max 1 ((tiles * 2 * half) + edge) in
     let* extra = int_range 0 8 in
     let* c0 = frequency [ (1, constant extra); (2, int_range 0 extra) ] in
     let+ seed = int in
     { f64; heads; side; m; n; k; cols = n + extra; c0; seed })

let guarded_product (type s) (b : Support.backend) (dt : (float, s) D.t) g =
  let (module K : Nx_kernel.S) = b.kernels in
  let st = Random.State.make [| g.seed |] in
  let values s =
    A.of_array dt (Iarray.of_array s)
      (Array.init (total s) (fun _ -> Random.State.float st 2. -. 1.))
  in
  let move m x = Option.get (A.move m x) in
  (* An [r × c] matrix, or with heads [2; r; heads; c] read
     [2; heads; r; c]. *)
  let matrix make r c =
    if g.heads = 1 then make [| r; c |]
    else move (M.Permute [| 0; 2; 1; 3 |]) (make [| 2; r; g.heads; c |])
  in
  let (batch : _ iarray), (contracting : _ iarray) =
    if g.heads = 1 then ([||], [| (1, 0) |])
    else ([| (0, 0); (1, 1) |], [| (3, 2) |])
  in
  let window x =
    let all c = { M.start = 0; count = c; step = 1 } in
    let cols = { M.start = g.c0; count = g.n; step = 1 } in
    let s = Iarray.to_array (L.shape (A.layout x)) in
    let r = Array.length s in
    move
      (M.Slice (Iarray.init r (fun i -> if i = r - 1 then cols else all s.(i))))
      x
  in
  let product ~dst a x =
    match
      K.contract ~batch ~contracting ~segments:None
        ~acc:(D.Any dt) ~dst ~init:None (plain a)
        (plain (window x))
    with
    | A.Done -> ()
    | r -> failf "contract answered %a" Nx_array_support.pp_answer r
  in
  let guard x =
    let g', free = Support.guarded g.side dt (A.layout x) in
    A.transfer ~src:x ~dst:g';
    (g', free)
  in
  let a = matrix values g.m g.k and x = matrix values g.k g.cols in
  let want =
    if g.heads = 1 then A.create Rig.host dt [| g.m; g.n |]
    else A.create Rig.host dt [| 2; g.heads; g.m; g.n |]
  in
  product ~dst:want a x;
  let ga, free_a = guard a and gx, free_x = guard x in
  let gd, free_d = guard want in
  Fun.protect
    ~finally:(fun () ->
      free_a ();
      free_x ();
      free_d ())
    (fun () ->
      product ~dst:gd ga gx;
      let bits y = Array.map Int64.bits_of_float (A.to_array y) in
      equal (array int64) (bits want) (bits gd))

let law_guards (b : Support.backend) g =
  let half = if g.f64 then 8 else 16 in
  cover "a sliver of half a matrix unit's tile" (g.n mod (2 * half) = half);
  cover "rows past one tile" (g.m > 2 * half);
  cover "b's window ending at the page"
    (g.side = Support.After && g.c0 + g.n = g.cols);
  cover "b's window starting at the page"
    (g.side = Support.Before && g.c0 = 0);
  cover "a half-tile sliver of b ending at the page, rows past one tile"
    (g.n mod (2 * half) = half
    && g.side = Support.After
    && g.c0 + g.n = g.cols
    && g.m > 2 * half);
  cover "two heads stored apart" (g.heads = 2);
  cover "two heads stored apart, b's window ending at the page"
    (g.heads = 2 && g.side = Support.After && g.c0 + g.n = g.cols);
  if g.f64 then guarded_product b D.Float64 g
  else guarded_product b D.Float32 g

(* nx.cpu under each table the host runs. *)
(* An [acc] Nx_kernel.accumulates refuses, a narrow float's or a boolean's, is
   refused with Wrong_dtype: the caller's error, which nx.cpu never declines. *)
let test_acc_refused (b : Support.backend) () =
  let module K = (val b.kernels) in
  let a = A.create Rig.host D.Float32 [| 2; 3 |] in
  let x = A.create Rig.host D.Float32 [| 3; 2 |] in
  let dst = A.create Rig.host D.Float32 [| 2; 2 |] in
  List.iter
    (fun acc ->
      equal ~msg:(Format.asprintf "%a" pp_dtype acc) Elements.answer
        A.Wrong_dtype
        (K.contract ~batch:[||] ~contracting:[| (1, 0) |]
           ~segments:None ~acc ~dst ~init:None (plain a) (plain x)))
    D.[ Any Float16; Any Bfloat16; Any Bool ]

(* Float32 products either side of each bound of nx.cpu's order: 4 and 5
   rows or columns, 63 and 64 outputs, a depth of 128 per output and one
   less. *)
let order_edges =
  let f32 = D.Any D.Float32 and v = Elements.plain 2 in
  List.map
    (fun (m, n, k) ->
      make ~acc:f32 ~out:f32 ([||], [| m |], [| n |], [| k |]) (f32, v)
        (f32, v) None 1)
    [
      (4, 16, 8); (5, 13, 8); (16, 4, 8); (13, 5, 8); (5, 12, 8); (7, 9, 8);
      (8, 8, 8); (8, 8, 8191); (8, 8, 8192);
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      test "refuses an acc a contraction does not compute in" (fun () ->
          b.around (test_acc_refused b));
      prop ~examples:(examples @ large_examples @ order_edges)
        "each output adds its products in the order nx_cpu.mli states"
        computed (run (law_order b));
      prop ~count:16 ~examples:deep_examples
        "a contraction on the job's threads has one thread's bits" deep_cases
        (run (law_threads b));
      prop ~examples:(List.concat_map regimes integer_accs @ byte_examples)
        "an integer output is its products' sum modulo the accumulator"
        integer_cases (run (law_wraps b));
      prop "a result of another dtype rounds the sum in acc once" rounded
        (run (law_rounds_once b));
      prop "computes every contraction nx_cpu.mli lists"
        any_case (run (law_computes b));
      prop ~examples:(scaled_examples @ mx_layouts)
        "a scaled contraction has the bits of its operands scaled on the host"
        scaled_cases (run (law_scaled b));
      prop "declines scales in an integer or complex accumulator" unscalable
        (run (law_unscalable b));
      prop ~examples:mx_examples
        "a block-scaled float4 operand has the bits of its elements scaled \
         on the host"
        mx_draws (run (law_mx b));
      test "refuses a scale that does not fit its operand" (fun () ->
          b.around (test_scale_misfits b));
      prop ~examples:seg_examples
        "each output of a segmented contraction adds its products in the \
         whole call's order"
        seg_draws (run (law_segments b));
      test "refuses segments that do not fit" (fun () ->
          b.around (test_segment_misfits b));
      test "a row in no group is init as acc holds it, or +0" (fun () ->
          b.around (test_rows_in_none b));
      test "reads the offsets before writing dst, which may be them" (fun () ->
          b.around (test_offsets_in_dst b));
      prop ~count:500
        "reads and writes no byte past its operands and dst, whatever pages lie \
         beside them"
        guard_cases (run (law_guards b));
      test "with no products, an output is init's bits" (run (test_empty b));
    ]

let () =
  exit (Windtrap.run "nx_cpu.contract" (List.map cpu Support.cpus))
