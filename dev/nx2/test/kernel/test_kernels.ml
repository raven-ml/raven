(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The laws of Nx_kernel.S, run through every kernel library the host runs:
   copies and casts into fresh C-contiguous destinations, each against a
   reference built element by element, over every dtype and pair of dtypes,
   drawn source layouts and drawn bytes; a kind the backend states it computes
   is never declined, and a decline writes nothing; a destination identical to
   an operand read at its own index gives a fresh one's bits; and operands
   kept alive across a call. nx.cpu's own refusals have a group of their own. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module B = Rig.Buffer
module Support = Nx_kernels_support

(* The kernels a law calls. *)
type kernels = (module Nx_kernel.S)

let copy (k : kernels) ~dst a =
  let module K = (val k) in
  K.apply1 Nx_kernel.Prog.Copy ~dst a

let cast (k : kernels) ~dst a =
  let module K = (val k) in
  K.apply1 Nx_kernel.Prog.Cast ~dst a

let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = )

(* [a] where the host reads it: [a] itself on the host, a copy elsewhere. *)
let host a =
  if Rig.equal (A.device a) Rig.host then a else A.to_device Rig.host a

(* [a] on [b]'s device, its layout kept. *)
let on (b : Support.backend) a =
  if Rig.equal (A.device a) b.device then a else A.to_device b.device a

(* Fails if [b] declines [k] at [dt], a case it states it computes. *)
let check_declined (b : Support.backend) k (D.Any dt as d) =
  if b.computes k d then
    failf "%s declined a kind it computes at %s" b.name (D.name dt)

(* [op] of [a] by [b]'s kernels, into a fresh C-contiguous array of [dt] on
   [b]'s device, read back where the host reads it; [None] if the kernels
   decline a kind they do not state they compute. *)
let run_op (b : Support.backend) op dt a =
  let module K = (val b.kernels) in
  let a = on b a in
  let dst = A.create b.device dt (L.shape (A.layout a)) in
  match K.apply1 op ~dst a with
  | A.Done -> Some (host dst)
  | A.Declined ->
      check_declined b (K1 op) (D.Any (A.dtype a));
      None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let declines = function
  | Some _ -> cover "computed" true
  | None -> cover "declined" true

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
   strided, broadcast, windowed, transposed or reshaped. A drawn movement
   repeats elements a few times in a hundred, so one case in five is
   broadcast along a new leading axis. *)
let case_of dts =
  let open Gen in
  let* (D.Any dt) = dts in
  let* s = shape in
  let* a = drawn dt s in
  let lead =
    let+ e = int_range 2 3 in
    Some (M.Broadcast (Array.append [| e |] s))
  in
  let+ m = frequency [ (4, option (movement ~apart:false s)); (1, lead) ] in
  let a = Option.value ~default:a (Option.bind m (fun m -> A.move m a)) in
  Case a

let case = Gen.with_pp pp_case (case_of dtypes)

(* An array of [dt], values from [seed], [h] by [w] and seen through [view]:
   as made, transposed, stepped, or cut into 2x2 windows whose last two axes
   are swapped, as a pooling layer reads them. *)
let large_at dt h w view seed =
  let s =
    match view with 0 -> [| h; w |] | 1 -> [| w; h |] | _ -> [| h; 2 * w |]
  in
  let a = seeded dt s seed in
  match view with
  | 0 -> Case a
  | 1 -> Case (Option.get (A.move (M.Permute [| 1; 0 |]) a))
  | 2 ->
      let all = { M.start = 0; count = h; step = 1 } in
      Case
        (Option.get
           (A.move (M.Slice [| all; { M.start = 1; count = w; step = 2 } |]) a))
  | _ ->
      let w axis = { M.axis; size = 2; step = 2; dilation = 1 } in
      let v = Option.get (A.move (M.Window [| w 0; w 1 |]) a) in
      Case (Option.get (A.move (M.Permute [| 0; 1; 3; 2 |]) v))

(* Arrays of a few hundred thousand elements at most, in shapes about the
   walk's block and tile sizes, through each view: a job of several threads
   and blocks. *)
let large_of dts =
  let open Gen in
  let* (D.Any dt) = dts in
  let side =
    of_list ~pp:Format.pp_print_int [ 1; 7; 63; 64; 65; 255; 256; 257; 300 ]
  in
  let* h = of_list ~pp:Format.pp_print_int [ 255; 256; 257; 300 ] in
  let* w = side in
  let* view = of_list ~pp:Format.pp_print_int [ 0; 1; 2; 3 ] in
  let+ seed = int in
  large_at dt h w view seed

let large = Gen.with_pp pp_case (large_of dtypes)

(* One large array through each view: the 32 drawn cases of a large law miss
   a view one time in ten thousand. *)
let large_views = List.map (fun v -> large_at D.Float32 257 65 v 1) [ 0; 1; 2; 3 ]

(* Transposed arrays whose rows lie a multiple of 4 KiB apart, as a
   4096-wide matrix's do: a tile's rows share their addresses' low bits. *)
let aliased =
  Gen.with_pp pp_case
    (let open Gen in
     let* (D.Any dt) = dtypes in
     let* h = of_list ~pp:Format.pp_print_int [ 1; 9; 70 ] in
     let+ seed = int in
     Case
       (Option.get
          (A.move (M.Permute [| 1; 0 |]) (seeded dt [| h; 4096 |] seed))))

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
  cover "2x2 windows" (L.rank l = 4);
  cover "several blocks" (L.numel l > 4096)

(* Copies *)

let law_copy b (Case a) =
  let r = run_op b Copy (A.dtype a) a in
  declines r;
  Option.iter (same a) r

let test_copy_nan_payloads b () =
  let bits = [| 0x7fc00001l; 0xffa00002l; 0x80000000l; 0x7f800001l |] in
  let f = Option.get (A.bitcast D.Float32 (A.of_array D.Uint32 [| 4 |] bits)) in
  Option.iter
    (fun dst ->
      equal (array int32) bits
        (A.to_array (Option.get (A.bitcast D.Uint32 dst))))
    (run_op b Copy D.Float32 f)

(* Casts *)

let law_cast b (Case a, D.Any d) =
  let r = run_op b Cast d a in
  declines r;
  Option.iter (same ~src:(A.Any a) (reference a d)) r

(* A case and a destination dtype, the case's own one time in five. *)
let pair c =
  let open Gen in
  let* (Case a as c) = c in
  let+ d =
    frequency [ (4, dtypes); (1, constant ~pp:pp_dtype (D.Any (A.dtype a))) ]
  in
  (c, d)

(* Every code of a format of at most 16 bits, cast to every dtype. *)
let test_every_code b (D.Any s) () =
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
      Option.iter
        (fun dst ->
          equal ~msg:(D.name d) (list string) []
            (differ ~src:(A.Any a) (reference a d) dst))
        (run_op b Cast d a))
    D.all

let narrow = List.filter (fun (D.Any dt) -> D.bits dt <= 16) D.all

(* Floats about every rounding point of the narrow float formats, as float32 and
   as float64, cast to every dtype: the integer stores meet NaN, the infinities
   and the bounds of every range there too. *)

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

let test_sweep b (type s) (s : (float, s) D.t) (xs : float array Lazy.t) () =
  let xs = Lazy.force xs in
  let a = A.of_array s [| Array.length xs |] xs in
  List.iter
    (fun (D.Any d) ->
      Option.iter
        (fun dst ->
          equal ~msg:(D.name d) (list string) []
            (differ ~src:(A.Any a) (reference a d) dst))
        (run_op b Cast d a))
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

let test_integer_ties b () =
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
          Option.iter
            (fun dst ->
              equal
                ~msg:(strf "%s to %s" (D.name (A.dtype a)) (D.name d))
                (list string) []
                (differ ~src:(A.Any a) (reference a d) dst))
            (run_op b Cast d a))
        D.all)
    srcs

(* A cast whose operands do not fit the caches reads and writes them through
   buffers. *)
let test_past_the_caches b () =
  let n = (1 lsl 22) + 3 in
  let x i = float_of_int ((i * 7919 mod 65537) - 32768) *. 1.25 in
  let a = A.of_array D.Float32 [| n |] (Array.init n x) in
  List.iter
    (fun (D.Any d) ->
      Option.iter
        (fun dst ->
          equal ~msg:(D.name d) (list string) [] (differ (reference a d) dst))
        (run_op b Cast d a))
    [ D.Any D.Float64; D.Any D.Int32; D.Any D.Float16 ]

(* Declines *)

(* Every unary kind, with its name in nx_kinds.h. *)
let unaries =
  Nx_kernel.Prog.
    [
      (Neg, "neg");
      (Recip, "recip");
      (Abs, "abs");
      (Sign, "sign");
      (Sqrt, "sqrt");
      (Exp, "exp");
      (Exp2, "exp2");
      (Log, "log");
      (Log2, "log2");
      (Log1p, "log1p");
      (Expm1, "expm1");
      (Sin, "sin");
      (Cos, "cos");
      (Tan, "tan");
      (Asin, "asin");
      (Acos, "acos");
      (Atan, "atan");
      (Sinh, "sinh");
      (Cosh, "cosh");
      (Tanh, "tanh");
      (Erf, "erf");
      (Floor, "floor");
      (Ceil, "ceil");
      (Round, "round");
      (Trunc, "trunc");
    ]

(* Every kind of one operand, to every kernel's [apply1]. *)
let op1s =
  Nx_kernel.Prog.[ Copy; Cast; Bitcast ]
  @ List.map (fun (u, _) -> Nx_kernel.Prog.Unary u) unaries

(* A kernel that declines writes nothing: [dst] keeps its drawn bytes. *)
let law_declined_apply1 (b : Support.backend) (Case a, D.Any d) =
  let module K = (val b.kernels) in
  let a = on b a and dst = on b (seeded d (L.shape (A.layout a)) 99) in
  List.iter
    (fun op ->
      let before = bits_of (host dst) in
      match K.apply1 op ~dst a with
      | A.Declined ->
          cover "declined" true;
          check_declined b (K1 op) (D.Any (A.dtype a));
          equal (array int) before (bits_of (host dst))
      | _ -> cover "computed" true)
    op1s

let test_declined_contract (b : Support.backend) () =
  let module K = (val b.kernels) in
  let spec =
    Nx_kernel.Spec.contract ~batch:[||]
      ~contracting:[| (1, 0) |]
      ~acc:(D.Any D.Float32) ~out:(D.Any D.Float32) ~init:false
  in
  let x = on b (A.of_array D.Float32 [| 2; 3 |] (Array.make 6 1.)) in
  let z = on b (A.of_array D.Float32 [| 3; 2 |] (Array.make 6 2.)) in
  let y = on b (A.of_array D.Float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |]) in
  let before = bits_of (host y) in
  match K.contract spec ~dst:(A.Any y) [| A.Any x; A.Any z |] with
  | A.Declined -> equal (array int) before (bits_of (host y))
  | A.Done -> ()
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

(* nx.cpu's refusals *)

(* Each answers the refusal nx_cpu.mli states, before any write. *)
let refuses r dst call =
  let before = bits_of dst in
  equal answer r (call ());
  equal (array int) before (bits_of dst)

let test_refusals k () =
  let f32 = D.Float32 in
  let x = A.of_array f32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let i32 = A.of_array D.Int32 [| 2; 3 |] (Array.make 6 7l) in
  let t = Option.get (A.move (M.Permute [| 1; 0 |]) x) in
  let d = A.of_array f32 [| 2; 3 |] (Array.make 6 9.) in
  refuses A.Wrong_dtype d (fun () -> copy k ~dst:d i32);
  refuses A.Shape_mismatch d (fun () -> copy k ~dst:d t);
  refuses A.Shape_mismatch d (fun () -> cast k ~dst:d t);
  let d = A.of_array D.Int32 [| 2; 3 |] (Array.make 6 9l) in
  refuses A.Wrong_dtype d (fun () -> copy k ~dst:d x);
  (* Into bytes the source reads, and into a broadcast. *)
  let flat = A.of_array f32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let part start =
    Option.get (A.move (M.Slice [| { M.start; count = 3; step = 1 } |]) flat)
  in
  refuses A.Overlapping (part 0) (fun () -> copy k ~dst:(part 0) (part 1));
  refuses A.Overlapping (part 0) (fun () -> cast k ~dst:(part 0) (part 2));
  let b =
    Option.get
      (A.move
         (M.Broadcast [| 2; 3 |])
         (A.of_array f32 [| 3 |] [| 0.; 0.; 0. |]))
  in
  refuses A.Repeated_elements b (fun () -> cast k ~dst:b i32)

(* Kinds of no, two and three operands *)

module P = Nx_kernel.Prog

let op2s =
  List.map
    (fun b -> P.Binary b)
    P.
      [
        Add; Sub; Mul; Fdiv; Idiv; Mod; Pow; Atan2; Maximum; Minimum; And; Or;
        Xor; Threefry;
      ]
  @ List.map (fun c -> P.Compare c) P.[ Equal; Not_equal; Less; Less_equal ]

let name2 = function
  | P.Binary b -> (
      match b with
      | Add -> "add"
      | Sub -> "sub"
      | Mul -> "mul"
      | Fdiv -> "fdiv"
      | Idiv -> "idiv"
      | Mod -> "mod"
      | Pow -> "pow"
      | Atan2 -> "atan2"
      | Maximum -> "maximum"
      | Minimum -> "minimum"
      | And -> "and"
      | Or -> "or"
      | Xor -> "xor"
      | Threefry -> "threefry")
  | Compare c -> (
      match c with
      | Equal -> "equal"
      | Not_equal -> "not_equal"
      | Less -> "less"
      | Less_equal -> "less_equal")

(* [v]'s low [w] bytes, least significant first. *)
let low_bytes w v =
  List.init w (fun i ->
      Int64.to_int (Int64.logand (Int64.shift_right_logical v (8 * i)) 0xFFL))

(* The bits of nx_kinds.h's kind [name] at each index of [ops], all of the
   dtype [dt], in C order, as [bits_of] gives them: as [dt] for a kind of
   [dt], as a boolean for a comparison. A narrow float computes in float32 and
   rounds once on the store; int4 and uint4 in the 32-bit type of their
   signedness, their result's low bits kept. *)
let rec expected : type v s.
    string -> (v, s) D.t -> compare:bool -> (v, s) A.t array -> int array =
 fun name dt ~compare ops ->
  let shape = L.shape (A.layout ops.(0)) in
  match D.kind dt with
  | D.Float when D.bits dt < 32 ->
      let r =
        expected name D.Float32 ~compare
          (Array.map (fun a -> reference a D.Float32) ops)
      in
      if compare then r
      else
        let b = String.init (Array.length r) (fun i -> Char.chr r.(i)) in
        let b = if b = "" then "\000" else b in
        bits_of
          (reference (A.v D.Float32 (L.contiguous shape) (B.of_string b)) dt)
  | _ ->
  let per idx =
    match D.kind dt with
    | D.Float when D.bits dt = 32 ->
        let args =
          Array.map
            (fun a -> Int32.to_int (A.get (words a) idx) land 0xFFFF_FFFF)
            ops
        in
        let r = Nx_kinds_support.f32 name args in
        if compare then [ Bool.to_int (r <> 0) ]
        else low_bytes 4 (Int64.of_int r)
    | D.Float ->
        let r = Nx_kinds_support.f64 name (Array.map (fun a -> A.get a idx) ops) in
        if compare then [ Bool.to_int (r <> 0.) ]
        else low_bytes 8 (Int64.bits_of_float r)
    | D.Boolean ->
        let args = Array.map (fun a -> if A.get a idx then 1L else 0L) ops in
        [ Bool.to_int (Nx_kinds_support.int "u32" name args <> 0L) ]
    | D.Signed | D.Unsigned ->
        let args = Array.map (fun a -> int64_of dt (A.get a idx)) ops in
        let ty =
          (if D.is D.Signed dt then "i" else "u")
          ^ if D.bits dt = 64 then "64" else "32"
        in
        let r =
          match name with
          | "threefry" -> Nx_kinds_support.threefry args.(0) args.(1)
          | "floor" | "ceil" | "round" | "trunc" -> args.(0)
          | _ -> Nx_kinds_support.int ty name args
        in
        if compare then [ Bool.to_int (r <> 0L) ]
        else if D.bits dt < 8 then [ Int64.to_int r land 0xF ]
        else low_bytes (D.bits dt / 8) r
    | D.Complex when D.bits dt = 64 ->
        let part a i =
          Int32.to_int (A.get (words a) (Array.append idx [| i |])) land 0xFFFF_FFFF
        in
        let args =
          Array.concat (Array.to_list (Array.map (fun a -> [| part a 0; part a 1 |]) ops))
        in
        let r = Nx_kinds_support.c64 name args in
        if compare then [ Bool.to_int (r.(0) <> 0) ]
        else low_bytes 4 (Int64.of_int r.(0)) @ low_bytes 4 (Int64.of_int r.(1))
    | D.Complex ->
        let args =
          Array.concat
            (Array.to_list
               (Array.map (fun a -> let z = A.get a idx in [| z.Complex.re; z.im |]) ops))
        in
        let r = Nx_kinds_support.c128 name args in
        if compare then [ Bool.to_int (r.(0) <> 0.) ]
        else low_bytes 8 (Int64.bits_of_float r.(0)) @ low_bytes 8 (Int64.bits_of_float r.(1))
  in
  Array.of_list (List.concat_map per (indices shape))

(* [x]'s shape over drawn bytes, its axes laid out in reverse or in order. *)
let alike (type v s) (x : (v, s) A.t) : (v, s) A.t Gen.t =
  let open Gen in
  let s = L.shape (A.layout x) in
  let r = Array.length s in
  let rev = Array.init r (fun i -> r - 1 - i) in
  let* flip = bool in
  if flip && r >= 2 then
    let+ y = drawn (A.dtype x) (Array.map (fun i -> s.(i)) rev) in
    Option.get (A.move (M.Permute rev) y)
  else drawn (A.dtype x) s

type pair = Pair : ('v, 's) A.t * ('v, 's) A.t -> pair

let pp_pair ppf (Pair (x, y)) =
  Format.fprintf ppf "%a %a, %a" D.pp (A.dtype x) L.pp (A.layout x) L.pp
    (A.layout y)

let pairs =
  Gen.with_pp pp_pair
    (let open Gen in
     let* (Case x) = case_of dtypes in
     let+ y = alike x in
     Pair (x, y))

(* Large views, as the copy laws draw them, against a contiguous operand:
   blocks staged through transposes and jobs of several threads. *)
let large_pairs =
  Gen.with_pp pp_pair
    (let open Gen in
     let* (Case x) =
       large_of
         (of_list ~pp:pp_dtype
            D.[ Any Float32; Any Int8; Any Float64; Any Bfloat16; Any Int4; Any Bit ])
     in
     let+ seed = int in
     Pair (x, seeded (A.dtype x) (L.shape (A.layout x)) seed))

(* The answer of [run] into a seeded destination of [dt] and [shape], checked:
   [Done] with the bytes [want ()], a decline only of a case [b] does not
   claim, and nothing written but on [Done]. *)
let answers (b : Support.backend) kind (D.Any at) (D.Any dt) shape ~accepted
    ~want (into : 'v 's. ('v, 's) A.t -> A.answer) =
  let dst = on b (seeded dt shape 99) in
  let before = bits_of (host dst) in
  match into dst with
  | A.Done ->
      cover "computed" true;
      equal ~msg:"accepted" bool true accepted;
      equal (array int) (want ()) (bits_of (host dst))
  | A.Declined ->
      cover "declined" true;
      check_declined b kind (D.Any at);
      equal ~msg:"declined writes nothing" (array int) before
        (bits_of (host dst))
  | A.Wrong_dtype when not accepted ->
      cover "outside its domain" true;
      equal ~msg:"refused writes nothing" (array int) before
        (bits_of (host dst))
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let law_apply1 ?(kinds = unaries) (b : Support.backend) (Case x) =
  let module K = (val b.kernels) in
  let dt = A.dtype x and shape = L.shape (A.layout x) in
  let x' = on b x in
  List.iter
    (fun (u, name) ->
      let k = P.Unary u in
      answers b (K1 k) (D.Any dt) (D.Any dt) shape
        ~accepted:(P.accepts1 k dt dt)
        ~want:(fun () -> expected name dt ~compare:false [| x |])
        (fun dst -> K.apply1 k ~dst x'))
    kinds

(* A bitcast keeps every element's bits, sub-byte codes and NaN payloads
   included. *)
let law_bitcast (b : Support.backend) (Case x, D.Any d) =
  let module K = (val b.kernels) in
  let dt = A.dtype x in
  let x = on b x in
  cover "of one width" (D.bits dt = D.bits d);
  answers b (K1 Bitcast) (D.Any dt) (D.Any d)
    (L.shape (A.layout x))
    ~accepted:(P.accepts1 Bitcast dt d)
    ~want:(fun () -> bits_of (host x))
    (fun dst -> K.apply1 Bitcast ~dst x)

(* A case and a dtype of its width, or any dtype one time in five. *)
let same_width c =
  let open Gen in
  let* (Case a as c) = c in
  let w = D.bits (A.dtype a) in
  let alike = List.filter (fun (D.Any d) -> D.bits d = w) D.all in
  let+ d =
    frequency [ (4, of_list ~pp:pp_dtype alike); (1, dtypes) ]
  in
  (c, d)

let law_apply2 ?(kinds = op2s) (b : Support.backend) (Pair (x, y)) =
  let module K = (val b.kernels) in
  let dt = A.dtype x and shape = L.shape (A.layout x) in
  let x' = on b x and y' = on b y in
  List.iter
    (fun k ->
      let compare = match k with P.Compare _ -> true | Binary _ -> false in
      let rd = if compare then D.Any D.Bool else D.Any dt in
      answers b (K2 k) (D.Any dt) rd shape ~accepted:(P.accepts2 k dt)
        ~want:(fun () -> expected (name2 k) dt ~compare [| x; y |])
        (fun dst -> K.apply2 k ~dst x' y'))
    kinds

(* Where picks each element's bytes; Fma is nx_kinds.h's. *)
let law_apply3 (b : Support.backend) (Pair (x, y), seed) =
  let module K = (val b.kernels) in
  let dt = A.dtype x and shape = L.shape (A.layout x) in
  let x = on b x and y = on b y in
  let w = max 1 (D.bits dt / 8) in
  let where (type s) (c : (bool, s) A.t) =
    let c = on b c in
    answers b (K3 Where) (D.Any dt) (D.Any dt) shape
      ~accepted:(P.accepts3 Where (A.dtype c) dt)
      ~want:(fun () ->
        let cs = A.to_array (host c) and xs = bits_of (host x)
        and ys = bits_of (host y) in
        Array.concat
          (List.mapi
             (fun i c -> Array.sub (if c then xs else ys) (i * w) w)
             (Array.to_list cs)))
      (fun dst -> K.apply3 Where ~dst:(A.expect dt (A.Any dst)) c x y)
  in
  if seed mod 2 = 0 then where (seeded D.Bool shape seed)
  else where (seeded D.Bit shape seed);
  let z = on b (seeded dt shape (seed + 1)) in
  answers b (K3 Fma) (D.Any dt) (D.Any dt) shape
    ~accepted:(P.accepts3 Fma dt dt)
    ~want:(fun () -> expected "fma" dt ~compare:false [| host x; host y; host z |])
    (fun dst -> K.apply3 Fma ~dst:(A.expect dt (A.Any dst)) x y z)

let law_apply0 (b : Support.backend) (D.Any dt, shape, seed) =
  let module K = (val b.kernels) in
  let w = max 1 (D.bits dt / 8) in
  let n = Array.fold_left ( * ) 1 shape in
  let e = bits_of (seeded dt [| 1 |] seed) in
  let fill = String.init (Array.length e) (fun i -> Char.chr e.(i)) in
  answers b (K0 (Fill fill)) (D.Any dt) (D.Any dt) shape
    ~accepted:(P.accepts0 (Fill fill) dt)
    ~want:(fun () -> Array.concat (List.init n (fun _ -> Array.sub e 0 w)))
    (fun dst -> K.apply0 (Fill fill) ~dst);
  if Array.length shape > 0 then begin
    let axis = seed mod Array.length shape in
    (* Each index along [axis] stored as a cast from int64 stores it. *)
    answers b (K0 (Iota axis)) (D.Any dt) (D.Any dt) shape
      ~accepted:(P.accepts0 (Iota axis) dt)
      ~want:(fun () ->
        let i = List.map (fun idx -> Int64.of_int idx.(axis)) (indices shape) in
        bits_of (reference (A.of_array D.Int64 shape (Array.of_list i)) dt))
      (fun dst -> K.apply0 (Iota axis) ~dst)
  end

(* Values the kinds' documentation states, through nx.cpu. *)
(* NaN bits *)

(* [n] elements of [dt] drawn from its specials alone, by [seed]: NaNs of
   both signs, signalling and quiet, with payloads, infinities, zeros, and
   numbers that some kinds make NaN from. *)
let specials_only (type v s) (dt : (v, s) D.t) n seed : (v, s) A.t =
  let sp = Array.of_list (specials dt) in
  let b = bytes_of_seed seed n in
  let pick i = sp.(Char.code b.[i] * 7 mod Array.length sp) in
  A.v dt (L.contiguous [| n |]) (B.of_string (String.concat "" (List.init n pick)))

(* A float kind's NaN results, those it passes and those it makes, are the
   same bits under every target table and into a destination that is its
   operand: the hardware's choice of NaN varies between targets and between
   vector and scalar code, and nx_kinds.h pins it. *)
let law_nan_bits (D.Any dt, seed) =
  let tables =
    List.filter (fun (b : Support.backend) -> Rig.equal b.device Rig.host)
      Support.backends
  in
  match D.kind dt with
  | D.Float when D.bits dt >= 32 ->
      let n = 300 in
      let x = specials_only dt n seed and y = specials_only dt n (seed + 1) in
      let z = specials_only dt n (seed + 2) in
      let kinds =
        List.map
          (fun (u, name) -> (name, fun (module K : Nx_kernel.S) dst x -> K.apply1 (Unary u) ~dst x))
          unaries
        @ List.filter_map
            (fun k ->
              match k with
              | P.Binary b when P.accepts2 k dt ->
                  Some (name2 k, fun (module K : Nx_kernel.S) dst x -> K.apply2 (Binary b) ~dst x y)
              | _ -> None)
            op2s
        @ [ ("fma", fun (module K : Nx_kernel.S) dst x -> K.apply3 Fma ~dst x y z) ]
      in
      List.iter
        (fun (name, run) ->
          let under (b : Support.backend) ~alias =
            b.around (fun () ->
                let x' = A.copy x in
                let dst = if alias then x' else A.create Rig.host dt [| n |] in
                equal ~msg:name answer A.Done (run b.kernels dst x');
                bits_of dst)
          in
          let want = under (List.hd tables) ~alias:false in
          List.iter
            (fun (b : Support.backend) ->
              equal ~msg:(strf "%s on %s" name b.name) (array int) want
                (under b ~alias:false);
              equal ~msg:(strf "%s in place on %s" name b.name) (array int) want
                (under b ~alias:true))
            tables)
        kinds
  | _ -> ()

let test_apply_values () =
  let k = (module Nx_cpu : Nx_kernel.S) in
  let module K = (val k) in
  let done_ ~msg a = equal ~msg answer A.Done a in
  let f32 xs = A.of_array D.Float32 [| Array.length xs |] xs in
  let d = A.create Rig.host D.Float32 [| 2 |] in
  done_ ~msg:"add" (K.apply2 (Binary Add) ~dst:d (f32 [| 1.; 2. |]) (f32 [| 3.; 0.5 |]));
  equal ~msg:"add" (array float_exact) [| 4.; 2.5 |] (A.to_array d);
  let i8 xs = A.of_array D.Int8 [| Array.length xs |] xs in
  let d = A.create Rig.host D.Int8 [| 3 |] in
  done_ ~msg:"idiv"
    (K.apply2 (Binary Idiv) ~dst:d (i8 [| -128; 7; 5 |]) (i8 [| -1; 0; -2 |]));
  equal ~msg:"idiv: the least value by -1, by zero, toward zero" (array int)
    [| -128; 0; -2 |] (A.to_array d);
  let u8 xs = A.of_array D.Uint8 [| Array.length xs |] xs in
  let b = A.create Rig.host D.Bool [| 2 |] in
  done_ ~msg:"less" (K.apply2 (Compare Less) ~dst:b (u8 [| 200; 1 |]) (u8 [| 100; 2 |]));
  equal ~msg:"less: unsigned order" (array bool) [| false; true |] (A.to_array b);
  let b2 = A.create Rig.host D.Bool [| 1 |] in
  done_ ~msg:"less"
    (K.apply2 (Compare Less) ~dst:b2 (f32 [| -0. |]) (f32 [| 0. |]));
  equal ~msg:"less: -0 equals +0" (array bool) [| false |] (A.to_array b2);
  let m = A.create Rig.host D.Float32 [| 1 |] in
  done_ ~msg:"maximum"
    (K.apply2 (Binary Maximum) ~dst:m (f32 [| -0. |]) (f32 [| 0. |]));
  equal ~msg:"maximum: -0 orders below +0" bool false
    (Float.sign_bit (A.get m [| 0 |]));
  let d = A.create Rig.host D.Float32 [| 2 |] in
  done_ ~msg:"where" (K.apply3 Where ~dst:d b (f32 [| 1.; 2. |]) (f32 [| 3.; 4. |]));
  equal ~msg:"where" (array float_exact) [| 3.; 2. |] (A.to_array d);
  let d = A.create Rig.host D.Int32 [| 2; 3 |] in
  done_ ~msg:"iota" (K.apply0 (Iota 1) ~dst:d);
  equal ~msg:"iota along axis 1" (array int32) [| 0l; 1l; 2l; 0l; 1l; 2l |]
    (A.to_array d);
  (* Large enough for a job of several units, each starting inside a run. *)
  let big = A.create Rig.host D.Int32 [| 3; 70001 |] in
  List.iter
    (fun axis ->
      done_ ~msg:"iota" (K.apply0 (Iota axis) ~dst:big);
      let want =
        Array.init (3 * 70001) (fun p ->
            Int32.of_int (if axis = 0 then p / 70001 else p mod 70001))
      in
      equal ~msg:(strf "iota along axis %d of 3x70001" axis) (array int32) want
        (A.to_array big))
    [ 0; 1 ];
  done_ ~msg:"fill" (K.apply0 (Fill (P.bits D.Int32 7l)) ~dst:d);
  equal ~msg:"fill" (array int32) (Array.make 6 7l) (A.to_array d);
  let i4 xs = A.of_array D.Int4 [| Array.length xs |] xs in
  let d = A.create Rig.host D.Int4 [| 2 |] in
  done_ ~msg:"add" (K.apply2 (Binary Add) ~dst:d (i4 [| 7; -8 |]) (i4 [| 1; -1 |]));
  equal ~msg:"add: int4 wraps" (array int) [| -8; 7 |] (A.to_array d);
  let bf16 xs = A.of_array D.Bfloat16 [| Array.length xs |] xs in
  let d = A.create Rig.host D.Bfloat16 [| 1 |] in
  done_ ~msg:"add"
    (K.apply2 (Binary Add) ~dst:d (bf16 [| 256. |]) (bf16 [| 1. |]));
  equal ~msg:"add: bfloat16 rounds once, ties to even" (array float_exact)
    [| 256. |] (A.to_array d);
  let unary ~msg u xs want =
    let d = A.create Rig.host D.Float32 [| Array.length xs |] in
    done_ ~msg (K.apply1 (Unary u) ~dst:d (f32 xs));
    equal ~msg (array float_exact) want (A.to_array d)
  in
  unary ~msg:"round: half away from zero" Round [| 2.5; -2.5; 0.49999997 |]
    [| 3.; -3.; 0. |];
  unary ~msg:"exp2: exact at integers" Exp2 [| 3.; -1.; 10.; -149. |]
    [| 8.; 0.5; 1024.; 0x1p-149 |];
  unary ~msg:"log2: exact at powers of two" Log2 [| 8.; 0.5; 0x1p-149 |]
    [| 3.; -1.; -149. |];
  let d = A.create Rig.host D.Float32 [| 1 |] in
  done_ ~msg:"sign" (K.apply1 (Unary Sign) ~dst:d (f32 [| Float.nan |]));
  equal ~msg:"sign: NaN for a NaN" bool true (Float.is_nan (A.get d [| 0 |]));
  let d = A.create Rig.host D.Int8 [| 4 |] in
  done_ ~msg:"recip" (K.apply1 (Unary Recip) ~dst:d (i8 [| 1; -1; 2; 0 |]));
  equal ~msg:"recip: x for 1 and -1, 0 otherwise" (array int) [| 1; -1; 0; 0 |]
    (A.to_array d);
  let d = A.create Rig.host D.Uint32 [| 1 |] in
  done_ ~msg:"bitcast" (K.apply1 Bitcast ~dst:d (f32 [| 1. |]));
  equal ~msg:"bitcast: 1.0's bits" (array int32) [| 0x3f800000l |]
    (A.to_array d);
  equal ~msg:"bitcast across widths is refused" answer A.Wrong_dtype
    (K.apply1 Bitcast ~dst:(A.create Rig.host D.Float64 [| 1 |]) (f32 [| 1. |]));
  equal ~msg:"exp on int32 is refused" answer A.Wrong_dtype
    (K.apply1 (Unary Exp) ~dst:(A.create Rig.host D.Int32 [| 1 |])
       (A.of_array D.Int32 [| 1 |] [| 1l |]));
  equal ~msg:"idiv on float32 is refused" answer A.Wrong_dtype
    (K.apply2 (Binary Idiv) ~dst:(A.create Rig.host D.Float32 [| 2 |])
       (f32 [| 1.; 2. |]) (f32 [| 1.; 2. |]))

(* A map the kernels decline writes nothing. *)
let test_declined_map (b : Support.backend) () =
  let module K = (val b.kernels) in
  let p =
    P.v ~ins:[| D.Any D.Float32 |] [| P.In 0; Op2 (Binary Add, 0, 0) |]
      ~outs:[| 1 |]
  in
  let s = Nx_kernel.Spec.map p ~loads:[| Plain |] in
  let x = on b (A.of_array D.Float32 [| 3 |] [| 1.; 2.; 3. |]) in
  let y = on b (A.of_array D.Float32 [| 3 |] [| 9.; 9.; 9. |]) in
  let before = bits_of (host y) in
  match K.map s ~dsts:[| A.Any y |] [| A.Any x |] with
  | A.Declined -> equal (array int) before (bits_of (host y))
  | A.Done -> equal (array float_exact) [| 2.; 4.; 6. |] (A.to_array (host y))
  | r -> failf "map answered %a" Nx_array_support.pp_answer r

(* In place *)

(* [k dst a] into a fresh destination, then into [a] itself as a C-contiguous
   copy, [a] being an operand [k] reads only at the result's own index: one
   answer, and on [Done] the same bits. *)
let in_place ~msg (b : Support.backend) a k =
  let fresh = A.create b.device (A.dtype a) (L.shape (A.layout a)) in
  let want = k fresh a in
  let alias = on b (A.copy (host a)) in
  equal ~msg answer want (k alias alias);
  if want = A.Done then begin
    cover "computed" true;
    equal ~msg (list string) [] (differ (host fresh) (host alias))
  end

let binaries = List.filter (function P.Binary _ -> true | _ -> false) op2s

(* An operand of [apply1] to [apply3], or a [Plain] load of [map], may be the
   destination itself: the kernels read each index before they write it. *)
let law_in_place ?(ones = op1s) ?(kinds = binaries) (b : Support.backend)
    (Pair (x, y), seed) =
  let module K = (val b.kernels) in
  let dt = A.dtype x and shape = L.shape (A.layout x) in
  let x = on b x and y = on b y in
  List.iter
    (fun op ->
      in_place ~msg:"apply1 into its operand" b x (fun dst x ->
          K.apply1 op ~dst x))
    ones;
  List.iter
    (fun k ->
      let msg into = strf "%s into %s" (name2 k) into in
      in_place ~msg:(msg "the first") b x (fun dst x -> K.apply2 k ~dst x y);
      in_place ~msg:(msg "the second") b y (fun dst y -> K.apply2 k ~dst x y);
      in_place ~msg:(msg "both") b x (fun dst x -> K.apply2 k ~dst x x))
    kinds;
  let c = on b (seeded D.Bool shape seed) in
  let z = on b (seeded dt shape (seed + 1)) in
  in_place ~msg:"where into the second" b x (fun dst x ->
      K.apply3 Where ~dst c x y);
  in_place ~msg:"where into the third" b y (fun dst y ->
      K.apply3 Where ~dst c x y);
  in_place ~msg:"fma into the addend" b z (fun dst z ->
      K.apply3 Fma ~dst x y z);
  if P.accepts2 (Binary Add) dt then begin
    let p =
      P.v ~ins:[| D.Any dt; D.Any dt |]
        [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
        ~outs:[| 2 |]
    in
    let s = Nx_kernel.Spec.map p ~loads:[| Plain; Plain |] in
    in_place ~msg:"map into a load" b x (fun dst x ->
        K.map s ~dsts:[| A.Any dst |] [| A.Any x; A.Any y |])
  end

(* Operands outlive the call *)

(* While another domain collects and compacts, copies of arrays no other value
   holds run with the runtime released: their buffers stay alive until the call
   ends. *)
let test_collect_during_call b () =
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
      for round = 1 to 4 do
        let src = A.of_array D.Float32 [| n |] (Array.make n 1.5) in
        Option.iter
          (fun dst ->
            equal ~msg:(strf "round %d" round) float_exact 1.5
              (A.get dst [| n - 1 |]))
          (run_op b Cast D.Float64 src)
      done)

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  let unit f () = b.around f in
  group b.name
    [
      prop "copy is bits for bits" case
        (run (fun c ->
             covers c;
             law_copy b c));
      prop ~count:32 ~examples:large_views "copy of large views is bits for bits"
        large
        (run (fun c ->
             covers_large c;
             law_copy b c));
      prop ~count:16 "copy of views with rows 4 KiB apart is bits for bits"
        aliased (run (law_copy b));
      test "copy keeps NaN payloads" (unit (test_copy_nan_payloads b));
      prop "cast is the reference, element by element" (pair case)
        (run (fun ((Case a as c), D.Any d) ->
             covers c;
             cover "to the source's dtype" (D.equal (A.dtype a) d);
             law_cast b (c, D.Any d)));
      prop ~count:32
        ~examples:(List.map (fun c -> (c, D.Any D.Float64)) large_views)
        "cast of large views is the reference" (pair large)
        (run (fun (c, d) ->
             covers_large c;
             law_cast b (c, d)));
      group "every code of the narrow dtypes"
        (List.map
           (fun (D.Any s as d) -> test (D.name s) (unit (test_every_code b d)))
           narrow);
      test "float32 about every rounding point, to every dtype"
        (unit (test_sweep b D.Float32 sweep32));
      test "float64 about every tie, to every dtype"
        (unit (test_sweep b D.Float64 sweep64));
      test "integers about every rounding point" (unit (test_integer_ties b));
      test "a cast past the caches is the reference"
        (unit (test_past_the_caches b));
      prop "a declined kind of one operand writes nothing" (pair case)
        (run (law_declined_apply1 b));
      test "a declined contraction writes nothing"
        (unit (test_declined_contract b));
      prop "kinds of one operand are nx_kinds.h's at each index" case
        (run (fun c ->
             covers c;
             law_apply1 b c));
      prop ~count:8 "kinds of one operand over large views" large
        (run
           (law_apply1
              ~kinds:P.[ (Exp, "exp"); (Sin, "sin"); (Neg, "neg") ]
              b));
      prop "bitcast keeps every element's bits" (same_width case)
        (run (fun ((c, _) as x) ->
             covers c;
             law_bitcast b x));
      prop "kinds of two operands are nx_kinds.h's at each index" pairs
        (run (law_apply2 b));
      prop ~count:8 "kinds of two operands over large views" large_pairs
        (run (law_apply2 ~kinds:P.[ Binary Add; Compare Less ] b));
      prop "where picks bytes and fma is nx_kinds.h's"
        (Gen.pair pairs Gen.nat)
        (run (law_apply3 b));
      prop "fill stores its element and iota each index"
        (Gen.triple dtypes shape Gen.nat)
        (run (law_apply0 b));
      test "a declined map writes nothing" (unit (test_declined_map b));
      prop "an operand read at the result's own index may be the destination"
        (Gen.pair pairs Gen.nat)
        (run (fun ((Pair (x, _), _) as c) ->
             covers (Case x);
             law_in_place b c));
      prop ~count:8 "in place over large views" (Gen.pair large_pairs Gen.nat)
        (run
           (law_in_place
              ~ones:P.[ Copy; Unary Neg; Unary Sin ]
              ~kinds:P.[ Binary Add; Binary Mul ]
              b));
      test "operands outlive a released call"
        (unit (test_collect_during_call b));
    ]

(* nx.cpu under the table the host runs best: what nx_cpu.mli promises beyond
   Nx_kernel.S. *)
let cpu =
  group "nx.cpu"
    [
      prop ~count:8 "NaN results are one set of bits on every table and in place"
        (Gen.pair (Gen.of_list ~pp:pp_dtype D.[ Any Float32; Any Float64 ]) Gen.int)
        law_nan_bits;
      test "refuses before any write"
        (test_refusals (module Nx_cpu : Nx_kernel.S));
      test "kinds give the values their documentation states" test_apply_values;
    ]

let () =
  exit (run "nx_kernel.kernels" (List.map laws Support.backends @ [ cpu ]))
