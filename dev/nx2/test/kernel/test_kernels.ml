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
    61436.;
    61440.;
    61444.;
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

(* A uint4 copy on several threads whose blocks start mid-byte: the walk
   gives each thread a share of an odd count of elements. *)
let test_sub_byte_shares b () =
  let n = (105 lsl 13) + 1 in
  let a =
    A.of_array D.Uint4 [| n |] (Array.init n (fun i -> i * 7919 land 15))
  in
  Option.iter (same a) (run_op b Copy D.Uint4 a)

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

(* [a] cast to every dtype is the reference. *)
let casts_are_reference b a =
  List.iter
    (fun (D.Any d) ->
      Option.iter
        (fun dst ->
          equal ~msg:(D.name d) (list string) []
            (differ ~src:(A.Any a) (reference a d) dst))
        (run_op b Cast d a))
    D.all

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
  casts_are_reference b a

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
  casts_are_reference b (A.of_array s [| Array.length xs |] xs)

(* Every length from 0 to 64, cast to every dtype, so that each run's
   vector loops and its scalar tail meet every cast: float32 about 1.0 from
   the sweep, and bfloat16 codes from 0x7F7C, which cross into NaN. *)
let test_lengths b () =
  let xs = Lazy.force sweep32 in
  for n = 0 to 64 do
    test_sweep b D.Float32 (lazy (Array.sub xs ((0x3F80 * 9) + n) n)) ();
    let codes =
      A.of_array D.Uint16 [| n |] (Array.init n (fun i -> 0x7F7C + i))
    in
    casts_are_reference b (Option.get (A.bitcast D.Bfloat16 codes))
  done

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

(* Every special of every dtype, cast to every dtype. *)
let test_specials b (D.Any s) () =
  let sp = specials s in
  let a =
    A.v s (L.contiguous [| List.length sp |]) (B.of_string (String.concat "" sp))
  in
  List.iter
    (fun (D.Any d) ->
      Option.iter
        (fun dst ->
          equal ~msg:(D.name d) (list string) []
            (differ ~src:(A.Any a) (reference a d) dst))
        (run_op b Cast d a))
    D.all

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
  let x = on b (A.of_array D.Float32 [| 2; 3 |] (Array.make 6 1.)) in
  let z = on b (A.of_array D.Float32 [| 3; 2 |] (Array.make 6 2.)) in
  let y = on b (A.of_array D.Float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |]) in
  let before = bits_of (host y) in
  match
    K.contract ~order:Lane ~batch:[||] ~contracting:[| (1, 0) |]
      ~acc:(D.Any D.Float32) ~dst:y ~init:None x z
  with
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

(* A contiguous bfloat16 pair: a kind that stages every operand, on several
   threads. *)
let large_bfloat16_pair =
  let (Case x) = large_at D.Bfloat16 257 65 0 1 in
  Pair (x, seeded (A.dtype x) (L.shape (A.layout x)) 2)

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

(* Idiv and Mod by a divisor repeated along the rows, each power of two the
   dtype holds, are nx_kinds.h's, over dividends with the dtype's extremes:
   the rows shift and mask for such a divisor. *)
let law_power_of_two_divisors (b : Support.backend) seed =
  let module K = (val b.kernels) in
  let n = 300 in
  let each (D.Any dt) =
    let x = seeded dt [| n |] seed in
    let signed = D.is D.Signed dt in
    for k = 0 to D.bits dt - if signed then 2 else 1 do
      let one = A.create Rig.host dt [| 1 |] in
      let p = A.of_array D.Float64 [| 1 |] [| Float.ldexp 1. k |] in
      ignore (Nx_cpu.apply1 Cast ~dst:one p);
      let y = A.v dt (L.v ~strides:[| 0 |] [| n |]) (A.buffer one) in
      let x' = on b x and y' = on b y in
      List.iter
        (fun kind ->
          answers b (K2 kind) (D.Any dt) (D.Any dt) [| n |] ~accepted:true
            ~want:(fun () -> expected (name2 kind) dt ~compare:false [| x; y |])
            (fun dst -> K.apply2 kind ~dst x' y'))
        P.[ Binary Idiv; Binary Mod ]
    done
  in
  List.iter each
    (List.filter
       (fun (D.Any d) -> D.is D.Signed d || D.is D.Unsigned d)
       D.all)

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

(* Fill, and Iota along each axis, into a C-contiguous destination at an
   offset into a larger buffer, mid-byte for a sub-byte dtype, store what
   they store into a fresh array and leave the elements around it; a
   library that declines one there declines it inside too, writing
   nothing. *)
let law_apply0_inside (b : Support.backend) (D.Any dt, s, (lead, trail), seed)
    =
  let module K = (val b.kernels) in
  let n = Array.fold_left ( * ) 1 s in
  let strides = L.strides (L.contiguous s) in
  let w = max 1 (D.bits dt / 8) in
  cover "sub-byte, mid-byte" (D.bits dt < 8 && lead mod (8 / D.bits dt) <> 0);
  if Rig.equal b.device Rig.host then
    cover "on the job's threads" (n >= 1_000_000);
  let inside k =
    let whole = on b (seeded dt [| lead + n + trail |] seed) in
    let dst = A.v dt (L.v ~offset:lead ~strides s) (A.buffer whole) in
    let fresh = A.create b.device dt s in
    let before = bits_of (host whole) in
    match K.apply0 k ~dst:fresh with
    | A.Declined ->
        check_declined b (K0 k) (D.Any dt);
        equal ~msg:"inside" answer A.Declined (K.apply0 k ~dst);
        equal ~msg:"declined writes nothing" (array int) before
          (bits_of (host whole))
    | r ->
        equal ~msg:"fresh" answer A.Done r;
        let e = bits_of (host fresh) in
        equal ~msg:"inside" answer A.Done (K.apply0 k ~dst);
        let want =
          Array.mapi
            (fun i x ->
              if i / w >= lead && i / w < lead + n then e.(i - (lead * w))
              else x)
            before
        in
        equal (array int) want (bits_of (host whole))
  in
  let e = bits_of (seeded dt [| 1 |] (seed + 1)) in
  inside (P.Fill (String.init (Array.length e) (fun i -> Char.chr e.(i))));
  if P.accepts0 (Iota 0) dt then
    for axis = 0 to Array.length s - 1 do
      inside (P.Iota axis)
    done

let fill_insides =
  Gen.with_pp
    (fun ppf (d, s, (lead, trail), _) ->
      Format.fprintf ppf "%a %a after %d, before %d" pp_dtype d
        Nx_array_gen.pp_ints s lead trail)
    (let open Gen in
     let* d = dtypes in
     let* s = shape in
     let* around = pair (int_range 0 9) (int_range 0 9) in
     let+ seed = nat in
     (d, s, around, seed))

(* Sub-byte destinations of a million elements, starting and ending
   mid-byte, on nx.cpu's job threads: its tables alone run them. *)
let large_fills =
  D.[ Any Int4; Any Bit; Any Float4_e2m1fn ]
  |> List.map (fun d -> (d, [| 1_000_001 |], (3, 5), 7))

(* NaN bits *)

(* nx.cpu under each target table the host runs. *)
let cpu_tables =
  List.filter (fun (b : Support.backend) -> Rig.equal b.device Rig.host)
    Support.backends

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
  let tables = cpu_tables in
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

(* Values the kinds' documentation states, through nx.cpu. *)
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
  let x = on b (A.of_array D.Float32 [| 3 |] [| 1.; 2.; 3. |]) in
  let y = on b (A.of_array D.Float32 [| 3 |] [| 9.; 9.; 9. |]) in
  let before = bits_of (host y) in
  match K.map p ~dsts:[| A.Any y |] [| Nx_kernel.Plain x |] with
  | A.Declined -> equal (array int) before (bits_of (host y))
  | A.Done -> equal (array float_exact) [| 2.; 4.; 6. |] (A.to_array (host y))
  | r -> failf "map answered %a" Nx_array_support.pp_answer r

(* A load identical to a destination through another view of its buffer is
   read before the destination is written: float32 offset 1 over a buffer
   and offset 0 over the view of the buffer from its fourth byte are one
   array. *)
let test_map_identical_view (b : Support.backend) () =
  let module K = (val b.kernels) in
  let n = 5 in
  let whole =
    on b (A.of_array D.Float32 [| n + 1 |] [| 0.; 1.; 2.; 3.; 4.; 5. |])
  in
  let buffer = A.buffer whole in
  let x = A.v D.Float32 (L.v ~offset:1 ~strides:[| 1 |] [| n |]) buffer in
  let neg =
    A.v D.Float32 (L.contiguous [| n |])
      (B.view buffer ~first:4 ~length:(4 * n))
  in
  let sum = A.create b.device D.Float32 [| n |] in
  let p =
    P.v ~ins:[| D.Any D.Float32 |]
      [| P.In 0; Op1 (Unary Neg, D.Any D.Float32, 0); Op2 (Binary Add, 0, 0) |]
      ~outs:[| 1; 2 |]
  in
  match K.map p ~dsts:[| A.Any neg; A.Any sum |] [| Nx_kernel.Plain x |] with
  | A.Declined -> ()
  | A.Done ->
      equal ~msg:"neg" (array float_exact) [| -1.; -2.; -3.; -4.; -5. |]
        (A.to_array (host neg));
      equal ~msg:"add" (array float_exact) [| 2.; 4.; 6.; 8.; 10. |]
        (A.to_array (host sum))
  | r -> failf "map answered %a" Nx_array_support.pp_answer r

(* A map of constants alone holds no value in a slot: it runs at its
   destination's shape all the same. *)
let test_constant_map () =
  let f32 = D.Any D.Float32 in
  let p = P.v ~ins:[||] [| P.Const (f32, P.bits D.Float32 2.5) |] ~outs:[| 0 |] in
  let y = A.create Rig.host D.Float32 [| 3; 5 |] in
  match Nx_cpu.map p ~dsts:[| A.Any y |] [||] with
  | A.Done -> equal (array float_exact) (Array.make 15 2.5) (A.to_array y)
  | r -> failf "map answered %a" Nx_array_support.pp_answer r

(* A map whose coordinate names an axis past its rank answers
   [Shape_mismatch] and writes nothing. *)
let test_coord_past_rank () =
  let p =
    P.v ~ins:[||] [| P.Coord 1; Op1 (Cast, D.Any D.Float32, 0) |] ~outs:[| 1 |]
  in
  let y = A.of_array D.Float32 [| 3 |] [| 9.; 9.; 9. |] in
  equal ~msg:"answer" answer A.Shape_mismatch
    (Nx_cpu.map p ~dsts:[| A.Any y |] [||]);
  equal ~msg:"untouched" (array float_exact) [| 9.; 9.; 9. |] (A.to_array y)

(* A map that holds many complex128 values gets blocks of few elements; over
   a transposed float32 operand the walk tiles rows, as many as fit the
   block. Each output is [k] times [2x], cast back to float32. *)
let test_tiled_short_blocks () =
  let n = 64 and k = 140 in
  let f32 = D.Any D.Float32 and c128 = D.Any D.Complex128 in
  let cast = P.Op1 (Cast, c128, 0) in
  let twice = Array.init k (fun _ -> P.Op2 (Binary Add, 1, 1)) in
  let sums =
    Array.init (k - 1) (fun j ->
        if j = 0 then P.Op2 (Binary Add, 2, 3)
        else P.Op2 (Binary Add, 2 + k + j - 1, 2 + j + 1))
  in
  let nodes = Array.concat [ [| P.In 0; cast |]; twice; sums ] in
  let last = Array.length nodes - 1 in
  let nodes = Array.append nodes [| P.Op1 (Cast, f32, last) |] in
  let p = P.v ~ins:[| f32 |] (Iarray.of_array nodes) ~outs:[| last + 1 |] in
  let data = Array.init (n * n) Float.of_int in
  let x =
    Option.get
      (A.move (M.Permute [| 1; 0 |]) (A.of_array D.Float32 [| n; n |] data))
  in
  let y = A.create Rig.host D.Float32 [| n; n |] in
  match
    Nx_cpu.map p ~dsts:[| A.Any y |] [| Nx_kernel.Plain x |]
  with
  | A.Done ->
      let want =
        Array.init (n * n) (fun q ->
            let i = q / n and j = q mod n in
            Float.of_int (2 * k) *. data.((j * n) + i))
      in
      equal (array float_exact) want (A.to_array y)
  | r -> failf "map answered %a" Nx_array_support.pp_answer r

(* Programs *)

exception Skip

(* A program over the dtypes [ins], drawn node by node from [st]: one to
   [length] nodes past its In nodes, each a kind whose operands' dtypes it
   takes, a few constants and coordinates among them, every In first; its
   outputs the last node and, sometimes, an earlier one. At most [room]
   operands, outputs and coordinate axes, each axis below [rank]. *)
let program ?(length = 6) st (ins : D.any array) ~room ~rank =
  let nins = Array.length ins in
  let nodes = ref (Array.to_list (Array.init nins (fun k -> P.In k))) in
  let tys = ref (Array.to_list ins) in
  let axes = ref [] in
  let pick l = List.nth l (Random.State.int st (List.length l)) in
  let all_dtypes = Array.of_list D.all in
  let draw () =
    let n = List.length !nodes and ty i = List.nth !tys i in
    let any () = Random.State.int st n in
    let alike i = List.filter (fun j -> ty j = ty i) (List.init n Fun.id) in
    match Random.State.int st 9 with
    | 0 ->
        let (D.Any dt) = all_dtypes.(Random.State.int st (Array.length all_dtypes)) in
        let e = bits_of (seeded dt [| 1 |] (Random.State.bits st)) in
        let e = if D.is D.Boolean dt then [| e.(0) land 1 |] else e in
        P.Const (D.Any dt, String.init (Array.length e) (fun i -> Char.chr e.(i)))
    | 1 ->
        let a = Random.State.int st (max rank 1) in
        if rank = 0 || List.mem a !axes || nins + List.length !axes + 2 > room
        then
          P.Op1 (Copy, ty 0, 0)
        else begin
          axes := a :: !axes;
          P.Coord a
        end
    | 2 ->
        let i = any () in
        P.Op1 (Cast, all_dtypes.(Random.State.int st (Array.length all_dtypes)), i)
    | 3 ->
        let i = any () in
        let (D.Any x) = ty i in
        let same = List.filter (fun (D.Any d) -> D.bits d = D.bits x) D.all in
        P.Op1 (Bitcast, pick same, i)
    | 4 | 5 ->
        let i = any () in
        P.Op1 (Unary (fst (pick unaries)), ty i, i)
    | 6 | 7 ->
        let i = any () in
        P.Op2 (pick op2s, i, pick (alike i))
    | _ ->
        let i = any () in
        let conds =
          List.filter
            (fun j -> let (D.Any d) = ty j in D.is D.Boolean d)
            (List.init n Fun.id)
        in
        if Random.State.bool st then P.Op3 (Fma, i, pick (alike i), pick (alike i))
        else if conds = [] then P.Op2 (Compare Less, i, pick (alike i))
        else P.Op3 (Where, pick conds, i, pick (alike i))
  in
  let dtype_of node =
    match node with
    | P.In k -> ins.(k)
    | Coord _ -> D.Any D.Int64
    | Const (d, _) -> d
    | Op1 (_, d, _) -> d
    | Op2 (Compare _, _, _) -> D.Any D.Bool
    | Op2 (_, i, _) | Op3 (_, _, i, _) -> List.nth !tys i
  in
  let operands = function
    | P.In _ | Coord _ | Const _ -> []
    | Op1 (_, _, i) -> [ i ]
    | Op2 (_, i, j) -> [ i; j ]
    | Op3 (_, i, j, k) -> [ i; j; k ]
  in
  (* Whether [node]'s kind takes its operands' dtypes [dts], as Prog.v
     states. *)
  let accepts (node : P.node) dts =
    match (node, dts) with
    | Op1 (k, D.Any y, _), [ D.Any x ] -> P.accepts1 k x y
    | Op2 (k, _, _), [ D.Any x; D.Any x' ] -> D.equal x x' && P.accepts2 k x
    | Op3 (k, _, _, _), [ D.Any c; D.Any x; D.Any x' ] ->
        D.equal x x' && P.accepts3 k c x
    | _ -> true
  in
  for _ = 1 to 1 + Random.State.int st length do
    let rec attempt tries =
      let node = draw () in
      if accepts node (List.map (List.nth !tys) (operands node)) then node
      else if tries = 0 then P.Op1 (Copy, List.nth !tys 0, 0)
      else attempt (tries - 1)
    in
    let node = attempt 50 in
    nodes := !nodes @ [ node ];
    tys := !tys @ [ dtype_of node ]
  done;
  let n = List.length !nodes in
  let outs =
    if n > nins + 1 && nins + List.length !axes + 2 <= room && Random.State.bool st
    then [| n - 1; nins + Random.State.int st (n - 1 - nins) |]
    else [| n - 1 |]
  in
  ( P.v ~ins:(Iarray.of_array ins) (Iarray.of_list !nodes)
      ~outs:(Iarray.of_array outs),
    Array.of_list !tys )

(* Each node of [p] run on its own through [b]'s kinds of no to three
   operands, over the operands [ops] of [shape]: the arrays of every node.
   Raises [Skip] where a kind is declined. *)
let split (b : Support.backend) (p : P.t) (types : D.any array) shape
    (ops : A.any array) =
  let module K = (val b.kernels) in
  let values = Array.make (Iarray.length p.nodes) ops.(0) in
  let fresh i =
    let (D.Any d) = types.(i) in
    A.Any (A.create b.device d shape)
  in
  let run answer = if answer <> A.Done then raise Skip in
  for i = 0 to Iarray.length p.nodes - 1 do
    values.(i) <-
      (match Iarray.get p.nodes i with
      | In k -> ops.(k)
      | Coord a ->
          let dst = A.create b.device D.Int64 shape in
          run (K.apply0 (Iota a) ~dst);
          A.Any dst
      | Const (_, bits) ->
          let (A.Any dst) = fresh i in
          run (K.apply0 (Fill bits) ~dst);
          A.Any dst
      | Op1 (k, _, j) ->
          let (A.Any dst) = fresh i in
          let (A.Any x) = values.(j) in
          run (K.apply1 k ~dst x);
          A.Any dst
      | Op2 (k, j, l) ->
          let (A.Any dst) = fresh i in
          let (A.Any x) = values.(j) in
          run (K.apply2 k ~dst x (A.expect (A.dtype x) values.(l)));
          A.Any dst
      | Op3 (k, c, j, l) ->
          let (A.Any c) = values.(c) in
          let (A.Any x) = values.(j) in
          let dst = A.expect (A.dtype x) (fresh i) in
          run (K.apply3 k ~dst c x (A.expect (A.dtype x) values.(l)));
          A.Any dst)
  done;
  values

(* [p]'s nodes, one per line. *)
let describe (p : P.t) =
  let dname (D.Any d) = D.name d in
  let node = function
    | P.In k -> strf "in %d" k
    | Coord a -> strf "coord %d" a
    | Const (d, b) ->
        strf "const %s 0x%s" (dname d)
          (String.concat "" (List.init (String.length b) (fun i -> strf "%02x" (Char.code b.[i]))))
    | Op1 (Copy, d, i) -> strf "copy %s %d" (dname d) i
    | Op1 (Cast, d, i) -> strf "cast %s %d" (dname d) i
    | Op1 (Bitcast, d, i) -> strf "bitcast %s %d" (dname d) i
    | Op1 (Unary u, d, i) -> strf "%s %s %d" (List.assoc u unaries) (dname d) i
    | Op2 (k, i, j) -> strf "%s %d %d" (name2 k) i j
    | Op3 (Where, c, i, j) -> strf "where %d %d %d" c i j
    | Op3 (Fma, a, b, c) -> strf "fma %d %d %d" a b c
  in
  String.concat "; "
    (List.mapi (fun i n -> strf "%d: %s" i (node n)) (Iarray.to_list p.nodes))

(* The most values [p] holds at once: each from the node that computes it
   through the last node that reads it, or to the end for an output. *)
let held (p : P.t) =
  let n = Iarray.length p.nodes in
  let last = Array.init n Fun.id in
  let read i j = if last.(j) < i then last.(j) <- i in
  for i = 0 to n - 1 do
    match Iarray.get p.nodes i with
    | P.In _ | Coord _ | Const _ -> ()
    | Op1 (_, _, j) -> read i j
    | Op2 (_, j, k) ->
        read i j;
        read i k
    | Op3 (_, j, k, l) ->
        read i j;
        read i k;
        read i l
  done;
  Array.iter (fun o -> last.(o) <- n) (Iarray.to_array p.outs);
  let most = ref 0 in
  for i = 0 to n - 1 do
    let live = ref 0 in
    for v = 0 to i do
      if last.(v) >= i then incr live
    done;
    most := max !most !live
  done;
  !most

(* [x] as a padded load of no padding, filled with zeros: it reads [x] as
   a plain load does. *)
let unpadded (type v s) (x : (v, s) A.t) =
  let none = Array.make (L.rank (A.layout x)) 0 in
  let pad =
    { Nx_kernel.lo = none; hi = none; interior = none; windows = [||] }
  in
  let zero = B.of_string (String.make (D.bytes (A.dtype x) 1) '\000') in
  let fill = A.get (A.v (A.dtype x) (L.contiguous [||]) zero) [||] in
  Nx_kernel.Padded { x; fill; pad }

(* A program's results are its nodes' results, each node run on its own,
   bit for bit: an intermediate holds its dtype's value exactly. With
   [declines], a map that declines a program [declines] does not accept
   fails. [length] bounds the program's nodes past its loads. *)
let law_map_split ?declines ?(labels = true) ?length ?(padded = false)
    (b : Support.backend) (Pair (x, y), seed) =
  let module K = (val b.kernels) in
  let st = Random.State.make [| seed |] in
  let dt = D.Any (A.dtype x) and shape = L.shape (A.layout x) in
  let x = on b x and y = on b y in
  let ins, ops =
    if Random.State.bool st then ([| dt; dt |], [| A.Any x; A.Any y |])
    else ([| dt |], [| A.Any x |])
  in
  (* nx.cpu's loop holds sixteen operands: a program within them is one
     map. *)
  let rank = Array.length shape in
  let p, types = program ?length st ins ~room:16 ~rank in
  let load (A.Any x) = if padded then unpadded x else Nx_kernel.Plain x in
  let loads = Array.map load ops in
  match split b p types shape ops with
  | exception Skip -> cover "a node declined" true
  | values -> (
      let dsts =
        Array.map
          (fun o ->
            let (D.Any d) = types.(o) in
            A.Any (on b (seeded d shape (seed + o))))
          (Iarray.to_array p.outs)
      in
      match K.map p ~dsts loads with
      | A.Declined -> (
          cover "map declined" true;
          match declines with
          | Some ok when not (ok p types) ->
              failf "map declined a program it computes: %d nodes"
                (Iarray.length p.nodes)
          | _ -> ())
      | A.Done ->
          cover "computed" true;
          if labels then begin
          cover "several nodes" (Iarray.length p.nodes > Array.length ins + 1);
          let has f = Iarray.exists f p.nodes in
          cover "a coordinate" (has (function P.Coord _ -> true | _ -> false));
          cover "a constant" (has (function P.Const _ -> true | _ -> false));
          cover "a cast" (has (function P.Op1 (Cast, _, _) -> true | _ -> false));
          cover "a where" (has (function P.Op3 (Where, _, _, _) -> true | _ -> false));
          cover "a sub-byte bitcast"
            (has (function
              | P.Op1 (Bitcast, D.Any d, j) ->
                  let (D.Any x) = types.(j) in
                  D.bits x < 8 && not (D.equal x d)
              | _ -> false));
          cover "two outputs" (Array.length (Iarray.to_array p.outs) = 2);
          let axes =
            List.sort_uniq compare
              (List.filter_map
                 (function P.Coord a -> Some a | _ -> None)
                 (List.init (Iarray.length p.nodes) (Iarray.get p.nodes)))
          in
          cover "more than four loop operands"
            (Array.length ins + Iarray.length p.outs + List.length axes > 4);
          let repeated (A.Any x) =
            let l = A.layout x in
            L.numel l > 1 && Array.for_all (( = ) 0) (L.strides l)
          in
          cover "a load of one repeated element" (Array.exists repeated ops)
          end;
          if length <> None then
            cover "more than eight values held" (held p > 8);
          Array.iteri
            (fun k o ->
              let (A.Any want) = values.(o) in
              let (A.Any got) = dsts.(k) in
              equal ~msg:(strf "output %d of %s" k (describe p)) (array int)
                (bits_of (host want))
                (bits_of (host got)))
            (Iarray.to_array p.outs)
      | r -> failf "map answered %a" Nx_array_support.pp_answer r)

(* A program of as many loads and outputs as a program has, reading the
   coordinate along every axis, as its nodes run one by one. *)
let law_map_full (b : Support.backend) (Case x) =
  let module K = (val b.kernels) in
  let shape = L.shape (A.layout x) in
  let rank = Array.length shape and nins = P.max_operands - 1 in
  let i64 = D.Any D.Int64 in
  let coord a = [ P.Coord a; P.Op2 (Binary Add, nins + (2 * a), nins + (2 * a) + 1) ] in
  let nodes =
    List.init nins (fun k -> P.In k)
    @ (P.Op1 (Cast, i64, 0) :: List.concat_map coord (List.init rank Fun.id))
  in
  let p =
    P.v
      ~ins:(Iarray.init nins (fun _ -> D.Any (A.dtype x)))
      (Iarray.of_list nodes)
      ~outs:[| List.length nodes - 1 |]
  in
  let types = Array.init (List.length nodes) (P.dtype p) in
  let x = on b x in
  let ops = Array.make nins (A.Any x) in
  cover "coordinates" (rank > 0);
  match split b p types shape ops with
  | exception Skip -> fail "a node declined"
  | values ->
      let dst = A.create b.device D.Int64 shape in
      equal ~msg:"map" answer A.Done
        (K.map p ~dsts:[| A.Any dst |] (Array.map (fun (A.Any x) -> Nx_kernel.Plain x) ops));
      let (A.Any want) = values.(List.length nodes - 1) in
      equal (array int) (bits_of (host want)) (bits_of (host dst))

(* A program that holds hundreds of values at once, over a padded load and
   the coordinates, as its nodes run one by one: [m] products of the operand,
   cast to int64, by a constant or a coordinate, all summed at the end. *)
let law_map_wide (b : Support.backend) ((Case x), m) =
  let module K = (val b.kernels) in
  let shape = L.shape (A.layout x) in
  let rank = Array.length shape and i64 = D.Any D.Int64 in
  let nodes = ref [ P.Op1 (Cast, i64, 0); P.In 0 ] in
  let add n =
    nodes := n :: !nodes;
    List.length !nodes - 1
  in
  let coords = List.init rank (fun a -> add (P.Coord a)) in
  let c = 1 in
  let factor i =
    match coords with
    | [] -> add (P.Const (i64, P.bits D.Int64 (Int64.of_int i)))
    | _ -> List.nth coords (i mod rank)
  in
  let values = List.init m (fun i -> add (P.Op2 (Binary Mul, c, factor i))) in
  let sum =
    List.fold_left (fun acc v -> add (P.Op2 (Binary Add, acc, v)))
      (List.hd values) (List.tl values)
  in
  let nodes = Array.of_list (List.rev !nodes) in
  let p =
    P.v ~ins:(Iarray.init 1 (fun _ -> D.Any (A.dtype x)))
      (Iarray.of_list (Array.to_list nodes)) ~outs:[| sum |]
  in
  let types = Array.init (Array.length nodes) (P.dtype p) in
  let x = on b x in
  cover "more than 256 values held" (held p > 256);
  cover "coordinates" (rank > 0);
  match split b p types shape [| A.Any x |] with
  | exception Skip -> fail "a node declined"
  | v ->
      let dst = A.create b.device D.Int64 shape in
      equal ~msg:"map" answer A.Done
        (K.map p ~dsts:[| A.Any dst |] [| unpadded x |]);
      let (A.Any want) = v.(sum) in
      equal (array int) (bits_of (host want)) (bits_of (host dst))

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
    in_place ~msg:"map into a load" b x (fun dst x ->
        K.map p ~dsts:[| A.Any dst |] Nx_kernel.[| Plain x; Plain y |])
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

(* Real transforms *)

type real = Real : (float, 'r) D.t -> real
type complex = Complex : (Complex.t, 'c) D.t -> complex

(* A real transform takes a real and a complex dtype of one precision,
   float32 with complex64 and float64 with complex128: it answers
   Wrong_dtype at every other pair, and computes or declines those two. *)
let test_real_transforms (b : Support.backend) () =
  let module K = (val b.kernels) in
  let reals =
    D.[ Real Float64; Real Float32; Real Float16; Real Bfloat16;
        Real Float8_e4m3fn; Real Float8_e5m2; Real Float4_e2m1fn ]
  in
  let pairs = [ ("float32", "complex64"); ("float64", "complex128") ] in
  List.iter
    (fun (Real r) ->
      List.iter
        (fun (Complex c) ->
          let takes = List.mem (D.name r, D.name c) pairs in
          let check name got =
            let msg = strf "%s of %s and %s" name (D.name r) (D.name c) in
            if not takes then equal ~msg answer A.Wrong_dtype got
            else if got = A.Wrong_dtype then failf "%s answered Wrong_dtype" msg
          in
          let real s = A.create b.device r s in
          let complex s = A.create b.device c s in
          check "r2c" (K.r2c ~axes:[| 0 |] ~dst:(complex [| 3 |]) (real [| 4 |]));
          check "c2r" (K.c2r ~axes:[| 0 |] ~dst:(real [| 4 |]) (complex [| 3 |])))
        D.[ Complex Complex64; Complex Complex128 ])
    reals

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
      test "a sub-byte copy shared mid-byte is bits for bits"
        (unit (test_sub_byte_shares b));
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
      test "every length to 64, to every dtype" (unit (test_lengths b));
      test "integers about every rounding point" (unit (test_integer_ties b));
      group "every special to every dtype"
        (List.filter_map
           (fun (D.Any s as d) ->
             if D.bits s < 8 || specials s = [] then None
             else Some (test (D.name s) (unit (test_specials b d))))
           D.all);
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
        (run (fun (Pair (x, y) as p) ->
             (* Operands whose loop is one run of steps of one or none, in
                a dtype the kinds compute in, as nx.cpu takes in one row:
                neither sub-byte nor a float narrower than float32. *)
             let dt = A.dtype x and lx = A.layout x in
             let c =
               L.coalesce [| L.contiguous (L.shape lx); lx; A.layout y |]
             in
             let one_run =
               L.rank c.(0) = 1
               && Array.for_all (fun l -> abs (L.stride l 0) <= 1) c
               && Array.for_all (fun l -> L.stride l 0 >= 0) c
               && D.bits dt >= 8
               && not (D.is D.Float dt && D.bits dt < 32)
             in
             cover "operands of one run in their own dtype" one_run;
             cover "operands of one run, of one element"
               (one_run && L.numel lx = 1);
             law_apply2 b p));
      prop ~count:8 ~examples:[ large_bfloat16_pair ]
        "kinds of two operands over large views" large_pairs
        (run (law_apply2 ~kinds:P.[ Binary Add; Compare Less ] b));
      prop ~count:4 "idiv and mod by a repeated power of two are nx_kinds.h's"
        Gen.nat
        (run (law_power_of_two_divisors b));
      prop "where picks bytes and fma is nx_kinds.h's"
        (Gen.pair pairs Gen.nat)
        (run (law_apply3 b));
      prop "fill stores its element and iota each index"
        (Gen.triple dtypes shape Gen.nat)
        (run (law_apply0 b));
      prop
        ~examples:(if Rig.equal b.device Rig.host then large_fills else [])
        "fill and iota store inside a buffer and nothing around it"
        fill_insides
        (run (law_apply0_inside b));
      test "a declined map writes nothing" (unit (test_declined_map b));
      test "a map reads a load identical to a destination before writing it"
        (unit (test_map_identical_view b));
      prop ~count:300 "a map gives the bits of its nodes run one by one"
        (Gen.pair pairs Gen.int)
        (run (fun ((Pair (x, _), _) as c) ->
             covers (Case x);
             law_map_split b c));
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
      test "real transforms take one precision"
        (unit (test_real_transforms b));
    ]

(* Erf at float64 is nx_kinds.h's at each index over rows whose pieces mix
   its two ranges, either one the most, into a fresh destination and into
   the operand itself: a row computes the range most of a piece's lanes are
   in and redoes the others. *)
let law_erf_ranges (near, seed) =
  let n = 1000 in
  let st = Random.State.make [| seed |] in
  let draw () =
    let a = Random.State.float st 1. in
    let v =
      if Random.State.float st 1. < near then a *. 2.5
      else 2.5 +. Float.succ 0. +. (a *. 5.)
    in
    if Random.State.bool st then -.v else v
  in
  let xs = Array.init n (fun _ -> draw ()) in
  let want = Array.map (fun v -> Int64.bits_of_float (Nx_kinds_support.f64 "erf" [| v |])) xs in
  let x = A.of_array D.Float64 [| n |] xs in
  let fresh = A.create Rig.host D.Float64 [| n |] in
  equal answer A.Done (Nx_cpu.apply1 (Unary Erf) ~dst:fresh x);
  equal ~msg:"fresh" (array int64) want (Array.map Int64.bits_of_float (A.to_array fresh));
  equal answer A.Done (Nx_cpu.apply1 (Unary Erf) ~dst:x x);
  equal ~msg:"in place" (array int64) want (Array.map Int64.bits_of_float (A.to_array x))

(* A job begun from a job's body runs to its end and adds what it is given,
   whether the outer job runs on the pool's threads or on the calling thread
   with the runtime released: 64 units of 64 MiB each, and 1 unit of 2 MiB.
   Each of the n outer units adds 0 to n - 1. *)
let test_nested_jobs () =
  List.iter
    (fun (n, cost) ->
      equal
        ~msg:(Printf.sprintf "%d units of %d bytes" n cost)
        int
        (n * (n * (n - 1) / 2))
        (Support.nested n cost))
    [ (64, 64 * 1024 * 1024); (1, 2 * 1024 * 1024); (3, 0) ]

(* Programs that hold hundreds to thousands of values at once *)

(* How a held program's values are made: each straight from the operand, so
   that no slot is reused before the end, or each after a few steps, some
   folded into the sum as they go, so that slots are reused. *)
type pattern = Never | Reuse

let pp_pattern ppf p =
  Format.pp_print_string ppf (match p with Never -> "never" | Reuse -> "reuse")

let pp_side ppf = function
  | None -> Format.pp_print_string ppf "unguarded"
  | Some Support.Before -> Format.pp_print_string ppf "before"
  | Some Support.After -> Format.pp_print_string ppf "after"

(* A program over one operand of [xt]: its cast to [dt], then [k] values of
   [dt] made from it by kinds of one operand and of two with a constant,
   held as [pattern] says and summed by kinds of two; its outputs the sum
   and, sometimes, the cast. Raises [Skip] where [dt] takes no cast from
   [xt] or no kind of two operands. *)
let held_program st (D.Any xt) (D.Any dt) ~k ~pattern =
  let t = D.Any dt in
  let rev = ref [] and tys = ref [] and n = ref 0 in
  let add node ty =
    rev := node :: !rev;
    tys := ty :: !tys;
    incr n;
    !n - 1
  in
  let x = add (P.In 0) (D.Any xt) in
  if not (P.accepts1 Cast xt dt) then raise Skip;
  let base = add (P.Op1 (Cast, t, x)) t in
  let bins =
    List.filter
      (fun k -> match k with P.Binary _ -> P.accepts2 k dt | _ -> false)
      op2s
  in
  let uns = List.filter (fun (u, _) -> P.accepts1 (Unary u) dt dt) unaries in
  if bins = [] then raise Skip;
  let pick l = List.nth l (Random.State.int st (List.length l)) in
  let const () =
    let e = bits_of (seeded dt [| 1 |] (Random.State.bits st)) in
    let e = if D.is D.Boolean dt then [| e.(0) land 1 |] else e in
    add (P.Const (t, String.init (Array.length e) (fun i -> Char.chr e.(i)))) t
  in
  let step prev =
    if uns <> [] && Random.State.bool st then
      add (P.Op1 (Unary (fst (pick uns)), t, prev)) t
    else
      let c = const () in
      add (P.Op2 (pick bins, prev, c)) t
  in
  let acc = ref None and pending = Queue.create () in
  let fold v =
    acc :=
      Some
        (match !acc with
        | None -> v
        | Some a -> add (P.Op2 (pick bins, a, v)) t)
  in
  for _ = 1 to k do
    let r = match pattern with Never -> 0 | Reuse -> Random.State.int st 5 in
    let prev = ref base in
    for _ = 1 to r do
      prev := step !prev
    done;
    Queue.push (step !prev) pending;
    if pattern = Reuse && Random.State.int st 4 = 0 then
      fold (Queue.pop pending)
  done;
  Queue.iter fold pending;
  let last = Option.get !acc in
  let nodes = Array.of_list (List.rev !rev) in
  let outs = if Random.State.bool st then [| last |] else [| last; base |] in
  ( P.v ~ins:(Iarray.of_list [ D.Any xt ]) (Iarray.of_array nodes)
      ~outs:(Iarray.of_array outs),
    Array.of_list (List.rev !tys) )

(* A map of a held program gives the bits of its nodes run one by one, at
   every dtype, its operand and destinations in memory of their own beside
   a guard page or not. *)
let law_map_held ?(small = false) (b : Support.backend)
    (Case x, D.Any dt, k, pattern, side, seed) =
  let module K = (val b.kernels) in
  let st = Random.State.make [| seed |] in
  let shape = L.shape (A.layout x) in
  match held_program st (D.Any (A.dtype x)) (D.Any dt) ~k ~pattern with
  | exception Skip -> cover "cast declined" true
  | p, types -> (
      let h = held p in
      cover "more than 256 held" (h > 256);
      cover "more than 1040 held" (h > 1040);
      cover "reuse" (pattern = Reuse);
      cover "never" (pattern = Never);
      cover "guarded" (side <> None);
      let numel = Array.fold_left ( * ) 1 shape in
      if small then cover "a few elements" (numel <= 4)
      else cover "hundreds of elements" (numel >= 200);
      match split b p types shape [| A.Any x |] with
      | exception Skip -> fail "a node declined"
      | values ->
          let frees = ref [] in
          let guard (type v s) (dt : (v, s) D.t) l : (v, s) A.t =
            match side with
            | None -> A.create b.device dt (L.shape l)
            | Some s ->
                let g, free = Support.guarded s dt l in
                frees := free :: !frees;
                g
          in
          Fun.protect
            ~finally:(fun () -> List.iter (fun f -> f ()) !frees)
            (fun () ->
              let x' =
                match side with
                | None -> x
                | Some _ ->
                    let g = guard (A.dtype x) (L.contiguous shape) in
                    equal ~msg:"copy" answer A.Done
                      (Nx_cpu.apply1 P.Copy ~dst:g x);
                    g
              in
              let dsts =
                Array.map
                  (fun o ->
                    let (D.Any d) = types.(o) in
                    A.Any (guard d (L.contiguous shape)))
                  (Iarray.to_array p.outs)
              in
              equal ~msg:"map" answer A.Done
                (K.map p ~dsts [| Nx_kernel.Plain x' |]);
              Array.iteri
                (fun i o ->
                  let (A.Any want) = values.(o) in
                  let (A.Any got) = dsts.(i) in
                  equal
                    ~msg:
                      (strf "output %d, %d nodes, %d held, %s from %s" i
                         (Iarray.length p.nodes) h (D.name dt)
                         (D.name (A.dtype x)))
                    (array int) (bits_of (host want)) (bits_of (host got)))
                (Iarray.to_array p.outs)))

let held_cases cases =
  let open Gen in
  let* c = cases in
  let* dt = dtypes in
  let* k = frequency [ (3, int_range 257 400); (1, int_range 1000 1600) ] in
  let* pattern = of_list ~pp:pp_pattern [ Never; Reuse ] in
  let* side =
    of_list ~pp:pp_side [ None; Some Support.Before; Some Support.After ]
  in
  let+ seed = int in
  (c, dt, k, pattern, side, seed)

(* nx.cpu under the table the host runs best: what nx_cpu.mli promises beyond
   Nx_kernel.S. *)
let cpu =
  let never _ _ = false in
  let maps (b : Support.backend) =
    [
      prop ~count:300
        (strf "%s computes every map its loop holds" b.name)
        (Gen.pair pairs Gen.int)
        (fun c -> b.around (fun () -> law_map_split ~declines:never b c));
      prop ~count:300
        (strf "%s computes held programs of a few elements" b.name)
        (held_cases case)
        (fun c -> b.around (fun () -> law_map_held ~small:true b c));
      prop ~count:12
        (strf "%s computes held programs over large views" b.name)
        (held_cases large)
        (fun c -> b.around (fun () -> law_map_held b c));
      prop ~count:20
        (strf "%s computes a map of hundreds of values held at once" b.name)
        (Gen.pair case (Gen.int_range 257 1000))
        (fun c -> b.around (fun () -> law_map_wide b c));
      prop ~count:100
        (strf "%s computes programs of hundreds of nodes over padded loads"
           b.name)
        (Gen.pair pairs Gen.int)
        (fun c ->
          b.around (fun () ->
              law_map_split ~length:400 ~labels:false
                ~padded:true ~declines:never
                b c));
      prop
        (strf "%s computes a map of every operand with coordinates" b.name)
        case
        (fun c -> b.around (fun () -> law_map_full b c));
      prop ~count:32
        (strf "%s computes maps over large views" b.name)
        (Gen.pair large_pairs Gen.int)
        (fun ((Pair (x, _), _) as c) ->
          covers_large (Case x);
          b.around (fun () ->
              law_map_split ~declines:never ~labels:false b c));
    ]
  in
  group "nx.cpu"
    (List.concat_map maps cpu_tables
    @ [
      prop ~count:8 "NaN results are one set of bits on every table and in place"
        (Gen.pair (Gen.of_list ~pp:pp_dtype D.[ Any Float32; Any Float64 ]) Gen.int)
        law_nan_bits;
      test "refuses before any write"
        (test_refusals (module Nx_cpu : Nx_kernel.S));
      test "kinds give the values their documentation states" test_apply_values;
      test "a job begun from a job's body runs to its end" test_nested_jobs;
      test "a map of constants alone fills its destination" test_constant_map;
      test "a map's coordinate past its rank is refused" test_coord_past_rank;
      test "a map of short blocks over a transposed operand tiles them"
        test_tiled_short_blocks;
      prop ~count:20 "erf at float64 is nx_kinds.h's over pieces of both ranges"
        (Gen.pair (Gen.of_list ~pp:Format.pp_print_float [ 0.02; 0.4; 0.6; 0.98 ]) Gen.int)
        law_erf_ranges;
    ])

let () =
  exit (run "nx_kernel.kernels" (List.map laws Support.backends @ [ cpu ]))
