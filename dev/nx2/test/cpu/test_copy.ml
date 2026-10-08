(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Copies and casts, each against a reference built element by element, over
   every dtype and pair of dtypes, drawn layouts and drawn bytes, under every
   target table the host runs. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module B = Rig.Buffer
module S = Nx_cpu_support

let strf = Printf.sprintf
let pp_dtype ppf (D.Any dt) = D.pp ppf dt
let dtypes = Gen.of_list ~pp:pp_dtype D.all

(* Bits *)

(* The bits of [a]'s elements in C order of indices: the bytes of a byte-wide
   dtype's, one code per element of a sub-byte one. *)
let bits_of (type v s) (a : (v, s) A.t) =
  match D.bits (A.dtype a) with
  | 1 -> Array.map Bool.to_int (A.to_array (A.expect D.Bit (A.Any a)))
  | 4 -> A.to_array (Option.get (A.bitcast D.Uint4 a))
  | _ -> A.to_array (Option.get (A.bitcast D.Uint8 a))

(* The first elements whose bits differ between [want] and [got], arrays of one
   dtype and shape, beside the element of [src] each came from. *)
let differ ?src (want : _ A.t) (got : _ A.t) =
  let per = max 1 (D.bits (A.dtype want) / 8) in
  let w = bits_of want and g = bits_of got in
  let hex bits per i =
    String.concat "" (List.init per (fun j -> strf "%02x" bits.((i * per) + j)))
  in
  let source i =
    match src with
    | None -> ""
    | Some (A.Any s) ->
        let sp = max 1 (D.bits (A.dtype s) / 8) in
        " from " ^ hex (bits_of s) sp i
  in
  let bad = ref [] in
  if w <> g then
    for i = (Array.length w / per) - 1 downto 0 do
      if Array.sub w (i * per) per <> Array.sub g (i * per) per then
        bad :=
          strf "element %d: %s, expected %s%s" i (hex g per i) (hex w per i)
            (source i)
          :: !bad
    done;
  List.filteri (fun i _ -> i < 8) !bad

let same ?src want got = equal (list string) [] (differ ?src want got)

(* The reference cast *)

(* [v] rounded to nearest, ties to even, at [p] significant bits, by integer
   arithmetic: [v] read unsigned unless [signed]. The result is exact in a
   double. *)
let round_int ~signed p v =
  let bit_length m =
    let rec go n m =
      if m = 0L then n else go (n + 1) (Int64.shift_right_logical m 1)
    in
    go 0 m
  in
  let negative = signed && Int64.compare v 0L < 0 in
  let m = if negative then Int64.neg v else v in
  let n = bit_length m in
  let r =
    if n <= p then
      (* [m] below 2^63 converts exactly; 2^63 itself is -2^63's magnitude. *)
      if Int64.compare m 0L < 0 then Float.ldexp 1. 63 else Int64.to_float m
    else
      let s = n - p in
      let q = Int64.shift_right_logical m s in
      let rest = Int64.logand m (Int64.pred (Int64.shift_left 1L s)) in
      let c = Int64.unsigned_compare rest (Int64.shift_left 1L (s - 1)) in
      let up = c > 0 || (c = 0 && Int64.logand q 1L = 1L) in
      Float.ldexp (Int64.to_float (if up then Int64.succ q else q)) s
  in
  if negative then -.r else r

(* The exact value of [x], an element of the integer dtype [dt], read unsigned
   where [dt] is. *)
let int64_of (type v s) (dt : (v, s) D.t) (x : v) : int64 =
  match dt with
  | D.Int64 -> x
  | D.Uint64 -> x
  | D.Int32 -> Int64.of_int32 x
  | D.Uint32 -> Int64.logand (Int64.of_int32 x) 0xFFFF_FFFFL
  | D.Int16 -> Int64.of_int x
  | D.Uint16 -> Int64.of_int x
  | D.Int8 -> Int64.of_int x
  | D.Uint8 -> Int64.of_int x
  | D.Int4 -> Int64.of_int x
  | D.Uint4 -> Int64.of_int x
  | _ -> invalid_arg "int64_of: not an integer dtype"

(* The low [bits] of [v] as a representative: signed or not. *)
let low ~signed bits v =
  let x =
    Int64.to_int (Int64.logand v (Int64.pred (Int64.shift_left 1L bits)))
  in
  if signed && x >= 1 lsl (bits - 1) then x - (1 lsl bits) else x

(* The integer [v] modulo [dt]'s width, as [dt]'s value. *)
let wrap (type w r) (dt : (w, r) D.t) (v : int64) : w =
  match dt with
  | D.Int64 -> v
  | D.Uint64 -> v
  | D.Int32 -> Int64.to_int32 v
  | D.Uint32 -> Int64.to_int32 v
  | D.Int16 -> low ~signed:true 16 v
  | D.Uint16 -> low ~signed:false 16 v
  | D.Int8 -> low ~signed:true 8 v
  | D.Uint8 -> low ~signed:false 8 v
  | D.Int4 -> low ~signed:true 4 v
  | D.Uint4 -> low ~signed:false 4 v
  | _ -> invalid_arg "wrap: not an integer dtype"

(* The integer [v], read unsigned unless [signed], stored into [d]: rounded once
   to a float format, modulo the width to an integer, [v <> 0] to a boolean. *)
let of_integer (type w r) (d : (w, r) D.t) ~signed v : w =
  match D.kind d with
  | D.Float ->
      D.of_float d (round_int ~signed ((D.float_format d).fraction_bits + 1) v)
  | D.Complex ->
      let p = if D.bits d = 64 then 24 else 53 in
      { Complex.re = round_int ~signed p v; im = 0. }
  | D.Boolean -> v <> 0L
  | D.Signed -> wrap d v
  | D.Unsigned -> wrap d v

(* [x], an element of [s], stored into [d] as a cast does. *)
let cast_value (type v s w r) (s : (v, s) D.t) (d : (w, r) D.t) (x : v) : w =
  match D.kind s with
  | D.Float -> D.of_float d x
  | D.Complex -> (
      match D.kind d with
      | D.Boolean -> x.re <> 0. || x.im <> 0.
      | D.Complex ->
          { Complex.re = (D.of_float d x.re).re; im = (D.of_float d x.im).re }
      | _ -> D.of_float d x.re)
  | D.Boolean -> of_integer d ~signed:true (if x then 1L else 0L)
  | D.Signed -> of_integer d ~signed:true (int64_of s x)
  | D.Unsigned -> of_integer d ~signed:false (int64_of s x)

(* The bits of [a]'s float32s: one per element of a float32 array, two of a
   complex64 one, on a last axis. *)
let words (type v s) (a : (v, s) A.t) =
  Option.get (A.bitcast D.Uint32 (Option.get (A.bitcast D.Float32 a)))

(* [a] cast to [d], element by element; a cast to [a]'s own dtype is a copy. A
   float32 that stays one, into or out of complex64's real part, keeps its bits:
   OCaml reads a float32 as a double, which quiets a signalling NaN, so those
   bits are copied here. *)
let reference (type v s w r) (a : (v, s) A.t) (d : (w, r) D.t) : (w, r) A.t =
  let s = A.dtype a in
  match D.equal_witness s d with
  | Some Type.Equal -> A.copy a
  | None ->
      let want =
        A.of_array d
          (L.shape (A.layout a))
          (Array.map (cast_value s d) (A.to_array a))
      in
      let each f = List.iter f (indices (L.shape (A.layout a))) in
      let re i = Array.append i [| 0 |] in
      (match (D.Any s, D.Any d) with
      | D.Any D.Float32, D.Any D.Complex64 ->
          let w = words want and x = words a in
          each (fun i -> A.set w (re i) (A.get x i))
      | D.Any D.Complex64, D.Any D.Float32 ->
          let w = words want and x = words a in
          each (fun i -> A.set w i (A.get x (re i)))
      | _ -> ());
      want

(* Arrays over drawn bytes *)

(* [x]'s low [w] bytes, least significant first. *)
let le w x =
  String.init w (fun i ->
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical x (8 * i)) 0xFFL)))

let f32_bits x = Int64.of_int32 (Int32.bits_of_float x)

(* Values at the edges of the conversion table: zeros, infinities, NaNs
   signalling and quiet, ties, the narrow formats' largest values and those past
   them, and the integer ranges' bounds. *)
let floats =
  [
    0.;
    -0.;
    infinity;
    neg_infinity;
    1.;
    -1.;
    0.5;
    1.5;
    2.5;
    -2.5;
    127.5;
    -128.5;
    255.5;
    6.;
    7.;
    448.;
    464.;
    57344.;
    65504.;
    65519.99;
    65520.;
    2147483647.;
    2147483648.;
    -2147483649.;
    4294967295.5;
    4294967296.;
    0x1p63;
    -0x1p63;
    0x1p64;
    1e-40;
    5e-324;
    1e300;
    3.4028235e38;
    1e39;
  ]

let specials (type v s) (dt : (v, s) D.t) =
  let f64 = List.map Int64.bits_of_float floats in
  let f32 = List.map f32_bits floats in
  let nan32 = [ 0x7FC00000L; 0xFFC12345L; 0x7F800001L; 0xFFA00002L ] in
  let nan64 =
    [ 0x7FF8000000000000L; 0x7FF0000000000001L; 0xFFF4000000000002L ]
  in
  let ints =
    [
      0L;
      1L;
      -1L;
      Int64.min_int;
      Int64.max_int;
      0x80L;
      0x7FL;
      0x8000L;
      0x7FFFL;
      0x8000_0000L;
      0x7FFF_FFFFL;
      0x100_0001L;
      0x20_0000_0000_0001L;
      0x7FFF_FFFF_FFFF_FC00L;
      -0x800L;
    ]
  in
  match dt with
  | D.Float64 -> List.map (le 8) (f64 @ nan64)
  | D.Float32 -> List.map (le 4) (f32 @ nan32)
  | D.Complex128 ->
      List.concat_map
        (fun x -> [ le 8 x ^ le 8 0L; le 8 0L ^ le 8 x ])
        (f64 @ nan64)
  | D.Complex64 ->
      List.concat_map
        (fun x -> [ le 4 x ^ le 4 0L; le 4 0L ^ le 4 x ])
        (f32 @ nan32)
  | D.Bool -> List.map (le 1) [ 0L; 1L; 2L; 255L ]
  | _ when D.is D.Signed dt || D.is D.Unsigned dt ->
      List.map (le (D.bits dt / 8)) ints
  | _ -> []

type case = Case : ('v, 's) A.t -> case

(* [n] bytes from the seed [seed], with a linear congruential generator. *)
let bytes_of_seed seed n =
  let x = ref (seed land 0xFFFFFFFFFFFF) in
  String.init n (fun _ ->
      x := ((!x * 0x5DEECE66D) + 11) land 0xFFFFFFFFFFFF;
      Char.chr ((!x lsr 24) land 0xFF))

(* A contiguous array of [dt] of shape [s] over the bytes of [seed], every
   seventh element one of [dt]'s specials in turn. *)
let seeded (type v s) (dt : (v, s) D.t) s seed : (v, s) A.t =
  let n = Array.fold_left ( * ) 1 s in
  let data = Bytes.of_string (bytes_of_seed seed (max 1 (D.bytes dt n))) in
  (match specials dt with
  | [] -> ()
  | sp ->
      let sp = Array.of_list sp and w = D.bits dt / 8 in
      for i = 0 to (n / 7) - 1 do
        Bytes.blit_string sp.(i mod Array.length sp) 0 data (7 * i * w) w
      done);
  A.v dt (L.contiguous s) (B.of_string (Bytes.to_string data))

(* How a generator fills an array of a dtype and shape. *)
type fill = { fill : 'v 's. ('v, 's) D.t -> int array -> ('v, 's) A.t Gen.t }

(* [seeded] from a drawn seed: data for arrays too large to draw. *)
let seeded_gen = { fill = (fun dt s -> Gen.map (seeded dt s) Gen.int) }

let pp_case ppf (Case a) =
  Format.fprintf ppf "%a %a" D.pp (A.dtype a) L.pp (A.layout a)

(* A contiguous array of [dt] of shape [s] over drawn bytes, each element's
   random or one of [dt]'s specials. *)
let drawn (type v s) (dt : (v, s) D.t) s : (v, s) A.t Gen.t =
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  let bytes w = string_of ~size:(const w) char in
  let+ data =
    if D.bits dt < 8 then bytes (D.bytes dt n)
    else
      let w = D.bits dt / 8 in
      let element =
        match specials dt with
        | [] -> bytes w
        | sp -> frequency [ (3, bytes w); (1, of_list sp) ]
      in
      map (String.concat "") (list ~size:(const n) element)
  in
  A.v dt (L.contiguous s) (B.of_string (if data = "" then "\000" else data))

(* An array of any dtype and shape over drawn bytes, viewed through a movement:
   strided, broadcast, windowed, transposed or reshaped. *)
let case_of dts =
  let open Gen in
  let* (D.Any dt) = dts in
  let* s = shape in
  let* a = drawn dt s in
  let+ m = option (movement ~apart:false s) in
  let a = Option.value ~default:a (Option.bind m (fun m -> A.move m a)) in
  Case a

let case = Gen.with_pp pp_case (case_of dtypes)

(* Arrays of a few hundred thousand elements at most, in shapes about the walk's
   block and tile sizes, transposed or stepped: a job of several threads and
   blocks. *)
let large_of dts =
  let open Gen in
  let* (D.Any dt) = dts in
  let side =
    of_list ~pp:Format.pp_print_int [ 1; 7; 63; 64; 65; 255; 256; 257; 300 ]
  in
  let* h = of_list ~pp:Format.pp_print_int [ 255; 256; 257; 300 ] in
  let* w = side in
  let* view = int_range 0 2 in
  let s =
    match view with 0 -> [| h; w |] | 1 -> [| w; h |] | _ -> [| h; 2 * w |]
  in
  let+ seed = int in
  let a = seeded dt s seed in
  match view with
  | 0 -> Case a
  | 1 -> Case (Option.get (A.move (M.Permute [| 1; 0 |]) a))
  | _ ->
      let all = { M.start = 0; count = h; step = 1 } in
      Case
        (Option.get
           (A.move (M.Slice [| all; { M.start = 1; count = w; step = 2 } |]) a))

let large = Gen.with_pp pp_case (large_of dtypes)

let covers (Case a) =
  let l = A.layout a in
  let n = L.numel l in
  cover "no element" (n = 0);
  cover "one element" (n = 1);
  cover "strided" (n > 1 && not (L.is_contiguous l));
  cover "broadcast" (Array.exists (fun s -> s = 0) (L.strides l) && n > 1);
  cover "sub-byte" (D.bits (A.dtype a) < 8)

let covers_large (Case a) =
  let l = A.layout a in
  cover "transposed" (L.rank l = 2 && abs (L.stride l 0) < abs (L.stride l 1));
  cover "stepped" (L.rank l = 2 && L.stride l 1 = 2);
  cover "several blocks" (L.numel l > 4096)

(* Copies *)

let law_copy (Case a) =
  let dst = A.create Rig.host (A.dtype a) (L.shape (A.layout a)) in
  equal int 0 (Nx_cpu.copy ~dst a);
  same a dst

(* A written view: the array transposed, or every other element of an array
   twice as long on its last axis, over drawn bytes the copy must keep. *)
let into ?(fill = { fill = drawn }) (type v s) (dt : (v, s) D.t) s =
  let open Gen in
  let r = Array.length s in
  let* stepped = bool in
  if r = 0 || not stepped then
    let rev a = Array.init (Array.length a) (fun i -> a.(r - 1 - i)) in
    let+ base = fill.fill dt (rev s) in
    (base, Option.get (A.move (M.Permute (rev (Array.init r Fun.id))) base))
  else
    let wide = Array.mapi (fun i d -> if i = r - 1 then 2 * d else d) s in
    let+ base = fill.fill dt wide in
    let range i d =
      if i = r - 1 then { M.start = 1; count = d; step = 2 }
      else { M.start = 0; count = d; step = 1 }
    in
    (base, Option.get (A.move (M.Slice (Array.mapi range s)) base))

(* Stores [a]'s elements' bits into [d], a view of its dtype and shape, one
   element at a time through nx.array's stores: bytes of a byte-wide dtype,
   codes of a sub-byte one. *)
let set_bits (type v s) (a : (v, s) A.t) (d : (v, s) A.t) =
  let each (type w r) (src : (w, r) A.t) (dst : (w, r) A.t) =
    List.iter
      (fun i -> A.set dst i (A.get src i))
      (indices (L.shape (A.layout src)))
  in
  match D.bits (A.dtype a) with
  | 1 -> each (A.expect D.Bit (A.Any a)) (A.expect D.Bit (A.Any d))
  | 4 ->
      each (Option.get (A.bitcast D.Uint4 a)) (Option.get (A.bitcast D.Uint4 d))
  | _ ->
      each (Option.get (A.bitcast D.Uint8 a)) (Option.get (A.bitcast D.Uint8 d))

let contents b =
  let s = Bytes.create (B.length b) in
  B.blit_to_bytes b 0 s 0 (Bytes.length s);
  Bytes.to_string s

(* A copy into a written view writes its elements and no other bit of its
   buffer: a sub-byte view shares bytes with the elements between its own. *)
let copy_into_keeps (Case a, Case base, Case dst) =
  let dst = A.expect (A.dtype a) (A.Any dst) in
  let base = A.expect (A.dtype a) (A.Any base) in
  let want =
    A.v (A.dtype a) (A.layout base) (B.of_string (contents (A.buffer base)))
  in
  set_bits a (A.v (A.dtype a) (A.layout dst) (A.buffer want));
  equal int 0 (Nx_cpu.copy ~dst a);
  equal string (contents (A.buffer want)) (contents (A.buffer base))

let law_copy_into ((Case a, _, Case dst) as c) =
  cover "a sub-byte view" (D.bits (A.dtype a) < 8);
  cover "a transposed view" (L.rank (A.layout dst) > 1);
  copy_into_keeps c

let copy_into =
  let open Gen in
  let* (Case a) = case in
  let+ base, dst = into (A.dtype a) (L.shape (A.layout a)) in
  (Case a, Case base, Case dst)

let pp_into ppf (c, _, Case dst) =
  Format.fprintf ppf "%a into %a" pp_case c L.pp (A.layout dst)

let copy_into = Gen.with_pp pp_into copy_into

let test_copy_nan_payloads () =
  let bits = [| 0x7fc00001l; 0xffa00002l; 0x80000000l; 0x7f800001l |] in
  let f = Option.get (A.bitcast D.Float32 (A.of_array D.Uint32 [| 4 |] bits)) in
  let dst = A.create Rig.host D.Float32 [| 4 |] in
  equal int 0 (Nx_cpu.copy ~dst f);
  equal (array int32) bits (A.to_array (Option.get (A.bitcast D.Uint32 dst)))

(* Casts *)

let law_cast (Case a, D.Any d) =
  let dst = A.create Rig.host d (L.shape (A.layout a)) in
  equal int 0 (Nx_cpu.cast ~dst a);
  same ~src:(A.Any a) (reference a d) dst

(* A case and a destination dtype, the case's own one time in five. *)
let pair c =
  let open Gen in
  let* (Case a as c) = c in
  let+ d =
    frequency [ (4, dtypes); (1, constant ~pp:pp_dtype (D.Any (A.dtype a))) ]
  in
  (c, d)

(* A cast into a written view writes each of its elements as the reference
   does, and no other bit of its buffer. *)
let cast_into_keeps (Case a, Case base, Case dst) =
  let d = A.dtype base in
  let dst = A.expect d (A.Any dst) in
  let want = A.v d (A.layout base) (B.of_string (contents (A.buffer base))) in
  set_bits (reference a d) (A.v d (A.layout dst) (A.buffer want));
  equal int 0 (Nx_cpu.cast ~dst a);
  equal string (contents (A.buffer want)) (contents (A.buffer base))

let law_cast_into ((Case a, Case base, Case dst) as c) =
  let d = A.dtype base in
  cover "a sub-byte view" (D.bits d < 8);
  cover "a transposed view" (L.rank (A.layout dst) > 1);
  cover "another dtype" (not (D.equal (A.dtype a) d));
  cast_into_keeps c

let cast_into ?fill c =
  let open Gen in
  let* (Case a) = c in
  let* (D.Any d) = dtypes in
  let+ base, dst = into ?fill d (L.shape (A.layout a)) in
  (Case a, Case base, Case dst)

(* Every code of a format of at most 16 bits, cast to every dtype. *)
let test_every_code (D.Any s) () =
  let w = D.bits s in
  let n = 1 lsl w in
  let codes = Array.init n Fun.id in
  let a =
    match w with
    | 1 -> A.Any (A.of_array D.Bit [| 2 |] [| false; true |])
    | 4 -> A.Any (A.of_array D.Uint4 [| n |] codes)
    | 8 -> A.Any (A.of_array D.Uint8 [| n |] codes)
    | _ -> A.Any (A.of_array D.Uint16 [| n |] codes)
  in
  let (A.Any p) = a in
  let a =
    match D.bits s with
    | 1 -> A.expect s (A.Any p)
    | _ -> Option.get (A.bitcast s p)
  in
  List.iter
    (fun (D.Any d) ->
      let dst = A.create Rig.host d [| L.numel (A.layout a) |] in
      equal int 0 (Nx_cpu.cast ~dst a);
      equal ~msg:(D.name d) (list string) []
        (differ ~src:(A.Any a) (reference a d) dst))
    D.all

let narrow = List.filter (fun (D.Any dt) -> D.bits dt <= 16) D.all

(* Floats about every rounding point of the narrow float formats, as float32 and
   as float64, cast to every dtype: the integer stores meet NaN, the
   infinities and the bounds of every range there too. *)

(* float32 whose low half is about a float16 or bfloat16 rounding bit, or all
   ones, the largest float32 below a power of two; every float8 and float4
   rounding bit lies in the high half. *)
let sweep32 =
  lazy
    (Array.of_list
       (List.concat_map
          (fun hi ->
            List.map
              (fun lo ->
                Int32.float_of_bits (Int32.of_int ((hi lsl 16) lor lo)))
              [ 0; 1; 0x0FFF; 0x1000; 0x1001; 0x7FFF; 0x8000; 0x8001; 0xFFFF ])
          (List.init 0x10000 Fun.id)))

(* The float64 neighbours of float32 values with their low 12 bits clear, which
   hold every tie of every narrow format: rounding through float32 to nearest
   first would land on a tie. *)
let sweep64 =
  lazy
    (Array.of_list
       (List.concat_map
          (fun hi ->
            let x =
              Int32.float_of_bits (Int32.of_int ((hi lsl 16) lor 0x1000))
            in
            if Float.is_finite x then [ Float.pred x; x; Float.succ x ]
            else [ x ])
          (List.init 0x10000 Fun.id)))

let test_sweep (type s) (s : (float, s) D.t) (xs : float array Lazy.t) () =
  let xs = Lazy.force xs in
  let a = A.of_array s [| Array.length xs |] xs in
  List.iter
    (fun (D.Any d) ->
      let dst = A.create Rig.host d [| Array.length xs |] in
      equal int 0 (Nx_cpu.cast ~dst a);
      equal ~msg:(D.name d) (list string) []
        (differ ~src:(A.Any a) (reference a d) dst))
    D.all

(* Integers about every rounding point of the float formats: m·2^e and the
   points half a unit about it, for m of 2 to 54 significant bits, as int64 and
   uint64, cast to every float dtype. *)
let ties =
  let open Int64 in
  List.concat_map
    (fun p ->
      List.concat_map
        (fun e ->
          let m = logor (shift_left 1L (p - 1)) 1L in
          let x = shift_left m e and half = shift_left 1L (e - 1) in
          List.concat_map
            (fun y -> [ y; pred y; succ y; neg y ])
            [ x; add x half; sub x half ])
        (List.init (64 - p) (fun e -> e + 1)))
    [ 2; 3; 4; 8; 11; 24; 25; 53; 54 ]

let test_integer_ties () =
  let n = List.length ties in
  let srcs =
    [
      A.Any (A.of_array D.Int64 [| n |] (Array.of_list ties));
      A.Any (A.of_array D.Uint64 [| n |] (Array.of_list ties));
    ]
  in
  List.iter
    (fun (A.Any a) ->
      List.iter
        (fun (D.Any d) ->
          let dst = A.create Rig.host d [| n |] in
          equal int 0 (Nx_cpu.cast ~dst a);
          equal
            ~msg:(strf "%s to %s" (D.name (A.dtype a)) (D.name d))
            (list string) []
            (differ ~src:(A.Any a) (reference a d) dst))
        D.all)
    srcs

(* A cast whose operands do not fit the caches reads and writes them through
   buffers. *)
let test_past_the_caches () =
  let n = (1 lsl 22) + 3 in
  let x i = float_of_int ((i * 7919 mod 65537) - 32768) *. 1.25 in
  let a = A.of_array D.Float32 [| n |] (Array.init n x) in
  List.iter
    (fun (D.Any d) ->
      let dst = A.create Rig.host d [| n |] in
      equal int 0 (Nx_cpu.cast ~dst a);
      equal ~msg:(D.name d) (list string) [] (differ (reference a d) dst))
    [ D.Any D.Float64; D.Any D.Int32; D.Any D.Float16 ]

(* A copy and a cast of several MiB into an int4 view that starts inside a
   byte, in rows of an odd length: blocks on different threads share their
   end bytes. *)
let test_sub_byte_threads () =
  let rows = 513 and cols = 4097 in
  let base = seeded D.Int4 [| rows; cols + 1 |] 5 in
  let view =
    M.Slice
      [|
        { M.start = 0; count = rows; step = 1 };
        { M.start = 1; count = cols; step = 1 };
      |]
  in
  let dst = Option.get (A.move view base) in
  cast_into_keeps (Case (seeded D.Int8 [| rows; cols |] 7), Case base, Case dst);
  copy_into_keeps (Case (seeded D.Int4 [| rows; cols |] 9), Case base, Case dst)

(* The door *)

(* A kernel refused by the door answers its code and writes nothing. *)
let refused ~substring code dst =
  let before = bits_of dst in
  raises_match (Exn.invalid_arg ~substring) (fun () ->
      A.refused "Nx.cast" code [ A.Any dst ]);
  equal (array int) before (bits_of dst)

let test_refusals () =
  let f32 = D.Float32 in
  let x = A.of_array f32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let i32 = A.of_array D.Int32 [| 2; 3 |] (Array.make 6 7l) in
  let t = Option.get (A.move (M.Permute [| 1; 0 |]) x) in
  let d = A.of_array f32 [| 2; 3 |] (Array.make 6 9.) in
  refused ~substring:"dtype" (S.copy d i32) d;
  refused ~substring:"shapes" (S.copy d t) d;
  refused ~substring:"shapes" (S.cast d t) d;
  let d = A.of_array D.Int32 [| 2; 3 |] (Array.make 6 9l) in
  refused ~substring:"dtype" (S.copy d x) d;
  (* Into bytes the source reads, and into a broadcast. *)
  let flat = A.of_array f32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let part start =
    Option.get (A.move (M.Slice [| { M.start; count = 3; step = 1 } |]) flat)
  in
  refused ~substring:"shares bytes" (S.copy (part 0) (part 1)) (part 0);
  refused ~substring:"shares bytes" (S.cast (part 0) (part 2)) (part 0);
  let b =
    Option.get
      (A.move
         (M.Broadcast [| 2; 3 |])
         (A.of_array f32 [| 3 |] [| 0.; 0.; 0. |]))
  in
  refused ~substring:"twice" (S.cast b i32) b

(* Operands outlive the call *)

(* While another domain collects and compacts, copies of arrays no other value
   holds run with the runtime released: their buffers stay alive until the call
   ends. *)
let test_collect_during_call () =
  let stop = Atomic.make false in
  let gc =
    Domain.spawn (fun () ->
        while not (Atomic.get stop) do
          Gc.compact ()
        done)
  in
  let n = 1 lsl 18 in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set stop true;
      Domain.join gc)
    (fun () ->
      for k = 1 to 4 do
        let dst = A.create Rig.host D.Float64 [| n |] in
        let e =
          Nx_cpu.cast ~dst (A.of_array D.Float32 [| n |] (Array.make n 1.5))
        in
        equal int 0 e;
        equal ~msg:(strf "round %d" k) float_exact 1.5 (A.get dst [| n - 1 |])
      done)

(* Sub-byte writes from two domains *)

let int4s = abstract "a"
let nibble = Gen.int_range (-8) 7

(* A copy or cast into elements 1 to 4 of six int4s writes their bytes, the
   first and last shared with elements 0 and 5: stores to those from another
   domain are kept. The source starts on a byte or inside one. Copies from two
   domains race on elements 1 to 4, so only elements 0 and 5 are read. *)
let into_middle cast phase xs a =
  let src =
    if cast then A.Any (A.of_array D.Int8 [| 4 |] xs)
    else
      let all =
        A.of_array D.Int4 [| phase + 4 |] (Array.append (Array.make phase 0) xs)
      in
      A.Any
        (Option.get
           (A.move (M.Slice [| { M.start = phase; count = 4; step = 1 } |]) all))
  in
  let dst =
    Option.get (A.move (M.Slice [| { M.start = 1; count = 4; step = 1 } |]) a)
  in
  let (A.Any src) = src in
  Nx_cpu.cast ~dst src

let commands =
  let nibbles = Gen.array ~size:(Gen.constant 4) nibble in
  let neighbour = Gen.map (fun last -> if last then 5 else 0) Gen.bool in
  [
    command "create"
      (Gen.unit @-> makes int4s)
      (fun () -> Array.make 6 0)
      (fun () -> A.of_array D.Int4 [| 6 |] (Array.make 6 0));
    command "write"
      (Gen.bool @-> Gen.int_range 0 1 @-> nibbles @-> int4s ^-> returns int)
      (fun _ _ xs m ->
        Array.blit xs 0 m 1 4;
        0)
      into_middle;
    command "set"
      (neighbour @-> nibble @-> int4s ^-> returns unit)
      (fun i x m -> m.(i) <- x)
      (fun i x a -> A.set a [| i |] x);
    command "get"
      (neighbour @-> int4s ^-> returns int)
      (fun i m -> m.(i))
      (fun i a -> A.get a [| i |]);
  ]

(* The suite *)

let laws target =
  let run f x = S.with_target target (fun () -> f x) in
  let unit f () = S.with_target target f in
  group target
    [
      prop "copy is bits for bits" case
        (run (fun c ->
             covers c;
             law_copy c));
      prop ~count:20 "copy of large views is bits for bits" large
        (run (fun c ->
             covers_large c;
             law_copy c));
      prop "a copy into a view keeps the rest of its buffer" copy_into
        (run law_copy_into);
      test "copy keeps NaN payloads" (unit test_copy_nan_payloads);
      prop "a cast into a view is the reference and keeps the rest of its buffer"
        (Gen.with_pp pp_into (cast_into case))
        (run law_cast_into);
      prop ~count:20
        "a cast into a large view is the reference and keeps the rest of its \
         buffer"
        (Gen.with_pp pp_into (cast_into ~fill:seeded_gen large))
        (run law_cast_into);
      prop "cast is the reference, element by element" (pair case)
        (run (fun ((Case a as c), D.Any d) ->
             covers c;
             cover "to the source's dtype" (D.equal (A.dtype a) d);
             law_cast (c, D.Any d)));
      prop ~count:20 "cast of large views is the reference" (pair large)
        (run (fun (c, d) ->
             covers_large c;
             law_cast (c, d)));
      group "every code of the narrow dtypes"
        (List.map
           (fun (D.Any s as d) -> test (D.name s) (unit (test_every_code d)))
           narrow);
      test "float32 about every rounding point, to every dtype"
        (unit (test_sweep D.Float32 sweep32));
      test "float64 about every tie, to every dtype"
        (unit (test_sweep D.Float64 sweep64));
      test "integers about every rounding point" (unit test_integer_ties);
      test "a cast past the caches is the reference" (unit test_past_the_caches);
      test "sub-byte views on several threads keep each other's bits"
        (unit test_sub_byte_threads);
    ]

let tests =
  [
    group "targets" (List.map laws (S.targets ()));
    test "the door refuses before any write" test_refusals;
    test "operands outlive a released call" test_collect_during_call;
    stateful ~domains:2 "writes into one byte from two domains keep each other"
      commands;
  ]

let () = exit (run "nx_cpu" tests)
