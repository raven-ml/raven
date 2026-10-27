(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The laws of Nx_kernel.S, run through every kernel library the host runs:
   copies and casts into fresh C-contiguous destinations, each against a
   reference built element by element, over every dtype and pair of dtypes,
   drawn source layouts and drawn bytes; a kind the backend states it computes
   is never declined, and a decline writes nothing; a destination identical to
   an operand read at its own index gives a fresh one's bits; operands
   kept alive across a call; and each kind the contract fixes exact, the
   others within nx_kinds.h's bounds. *)

open Windtrap
open Nx_array_gen
open Kernels_cases
module A = Nx_array
module D = Nx_array.Dtype
module B = Rig.Buffer
module Support = Nx_kernels_support

(* [op] of [a] by [b]'s kernels, into a fresh C-contiguous array of [dt] on
   [b]'s device, read back where the host reads it; [None] if the kernels
   decline a kind they do not state they compute. *)
let run_op (b : Support.backend) op dt a =
  let module K = (val b.kernels) in
  let a = on b a in
  let dst = A.create b.device dt (L.shape (A.layout a)) in
  match Support.map1 (module K) op ~dst a with
  | A.Done -> Some (host dst)
  | A.Declined ->
      check_declined b (K1 op) (D.Any (A.dtype a));
      None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let declines = function
  | Some _ -> cover "computed" true
  | None -> cover "declined" true

(* Bits *)

(* The first elements whose bits differ between [want] and [got], arrays of one
   dtype and shape, beside the element of [src] each came from; with
   [any_nan], two NaNs do not differ. *)
let differ ?src ?(any_nan = false) (want : _ A.t) (got : _ A.t) =
  let d = D.Any (A.dtype want) in
  let per = max 1 (D.bits (A.dtype want) / 8) in
  let w = bits_of want and g = bits_of got in
  let same i =
    let e = elements_of d (Array.sub w (i * per) per)
    and f = elements_of d (Array.sub g (i * per) per) in
    if any_nan then Testable.equal (Elements.same d) e.(0) f.(0)
    else e.(0) = f.(0)
  in
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
      if not (same i) then
        bad :=
          strf "element %d: %s, expected %s%s" i (hex g per i) (hex w per i)
            (source i)
          :: !bad
    done;
  List.filteri (fun i _ -> i < 8) !bad

let same ?src want got = equal (list string) [] (differ ?src want got)

(* Arrays over drawn bytes *)

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

(* Transposed copies whose square blocks nx.cpu stores past the caches:
   rows 4 KiB apart, which x86-64 transposes through a buffer, and float64
   ones, of 32 to 34 MiB, past fudge's and nonnormal's largest caches, and
   rows of 4000 float32, which end in a tile half as wide as the others,
   of 64 MiB, past four times the M1 Max's. *)
let test_streamed_copies b () =
  List.iter
    (fun (Case a) -> Option.iter (same a) (run_op b Copy (A.dtype a) a))
    [
      large_at D.Float32 4096 2048 1 1;
      large_at D.Float32 4200 4000 1 2;
      large_at D.Float64 1500 3000 1 3;
    ]

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

(* Every kind of one operand. *)
let op1s =
  Nx_kernel.Prog.[ Copy; Cast; Bitcast ]
  @ List.map (fun (u, _) -> Nx_kernel.Prog.Unary u) unaries

(* A kernel that declines writes nothing: [dst] keeps its drawn bytes. *)
let law_declined_map1 (b : Support.backend) (Case a, D.Any d) =
  let module K = (val b.kernels) in
  let a = on b a and dst = on b (seeded d (L.shape (A.layout a)) 99) in
  List.iter
    (fun op ->
      let before = bits_of (host dst) in
      match Support.map1 (module K) op ~dst a with
      | A.Declined ->
          cover "declined" true;
          check_declined b (K1 op) (D.Any (A.dtype a));
          equal (array int) before (bits_of (host dst))
      | _ -> cover "computed" true)
    (List.filter (fun op -> Nx_kernel.Prog.accepts1 op (A.dtype a) d) op1s)

let test_declined_contract (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = on b (A.of_array D.Float32 [| 2; 3 |] (Array.make 6 1.)) in
  let z = on b (A.of_array D.Float32 [| 3; 2 |] (Array.make 6 2.)) in
  let y = on b (A.of_array D.Float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |]) in
  let before = bits_of (host y) in
  match
    K.contract ~batch:[||] ~contracting:[| (1, 0) |]
      ~segments:None ~acc:(D.Any D.Float32) ~dst:y ~init:None
      { Nx_kernel.array = A.Any x; scale = None }
      { Nx_kernel.array = A.Any z; scale = None }
  with
  | A.Declined -> equal (array int) before (bits_of (host y))
  | A.Done -> ()
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

(* Kinds of no, two and three operands *)

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
    (fun dst -> Support.map1 (module K) Bitcast ~dst x)

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
      ignore (Support.map1 (module Nx_cpu) Cast ~dst:one p);
      let y = A.v dt (L.v ~strides:[| 0 |] [| n |]) (A.buffer one) in
      let x' = on b x and y' = on b y in
      List.iter
        (fun kind ->
          answers b (K2 kind) (D.Any dt) (D.Any dt) [| n |] ~accepted:true
            ~want:(fun () -> expected (name2 kind) dt ~compare:false [| x; y |])
            (fun dst -> Support.map2 (module K) kind ~dst x' y'))
        P.[ Binary Idiv; Binary Mod ]
    done
  in
  List.iter each
    (List.filter
       (fun (D.Any d) -> D.is D.Signed d || D.is D.Unsigned d)
       D.all)

(* Threefry's contiguous rows run two blocks of vectors at a time, 16
   elements on x86-64 and 8 on arm64, then a tail of the elements
   left one at a time: a length that is no multiple of 8 leaves a tail on
   every table. Over lengths around the blocks, with the counters or the
   keys one element repeated, its bits are nx_kinds.h's. *)
let law_threefry_rows (b : Support.backend) (n, repeated, seed) =
  let module K = (val b.kernels) in
  let dt = D.Uint64 in
  let operand which s =
    if repeated <> which then seeded dt [| n |] s
    else A.v dt (L.v ~strides:[| 0 |] [| n |]) (A.buffer (seeded dt [| 1 |] s))
  in
  let x = operand 1 seed and y = operand 2 (seed + 1) in
  cover "both contiguous" (repeated = 0);
  cover "counters repeated" (repeated = 1);
  cover "keys repeated" (repeated = 2);
  cover "a tail on every table" (n mod 8 <> 0);
  cover "several blocks" (n >= 32);
  let x' = on b x and y' = on b y in
  answers b (K2 (Binary Threefry)) (D.Any dt) (D.Any dt) [| n |] ~accepted:true
    ~want:(fun () -> expected "threefry" dt ~compare:false [| x; y |])
    (fun dst -> Support.map2 (module K) (Binary Threefry) ~dst x' y')

(* Where picks each element's bytes; Fma is nx_kinds.h's. *)
let law_map3 (b : Support.backend) (Pair (x, y), seed) =
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
      (fun dst -> Support.map3 (module K) Where ~dst:(A.expect dt (A.Any dst)) c x y)
  in
  if seed mod 2 = 0 then where (seeded D.Bool shape seed)
  else where (seeded D.Bit shape seed);
  let z = on b (seeded dt shape (seed + 1)) in
  answers b (K3 Fma) (D.Any dt) (D.Any dt) shape
    ~accepted:(P.accepts3 Fma dt dt)
    ~want:(fun () -> expected "fma" dt ~compare:false [| host x; host y; host z |])
    (fun dst -> Support.map3 (module K) Fma ~dst:(A.expect dt (A.Any dst)) x y z)

let law_map0 (b : Support.backend) (D.Any dt, shape, seed) =
  let module K = (val b.kernels) in
  let w = max 1 (D.bits dt / 8) in
  let n = Array.fold_left ( * ) 1 shape in
  let s = Iarray.of_array shape in
  let e = bits_of (seeded dt [| 1 |] seed) in
  let fill = String.init (Array.length e) (fun i -> Char.chr e.(i)) in
  answers b (K0 (Fill fill)) (D.Any dt) (D.Any dt) s ~accepted:true
    ~want:(fun () -> Array.concat (List.init n (fun _ -> Array.sub e 0 w)))
    (fun dst -> Support.map0 (module K) (Fill fill) ~dst);
  if Array.length shape > 0 then begin
    let axis = seed mod Array.length shape in
    (* Each index along [axis], an int64; a destination of another dtype is
       not the program's. *)
    let iota (D.Any d) =
      answers b (K0 (Iota axis)) (D.Any d) (D.Any d) s
        ~accepted:(D.equal d D.Int64)
        ~want:(fun () ->
          let i = List.map (fun idx -> Int64.of_int idx.(axis)) (indices shape) in
          bits_of (A.of_array D.Int64 s (Array.of_list i)))
        (fun dst -> Support.map0 (module K) (Iota axis) ~dst)
    in
    iota (D.Any D.Int64);
    if not (D.equal dt D.Int64) then iota (D.Any dt)
  end

(* Fill, and Iota along each axis, into a C-contiguous destination at an
   offset into a larger buffer, mid-byte for a sub-byte dtype, store what
   they store into a fresh array and leave the elements around it; a
   library that declines one there declines it inside too, writing
   nothing. Iota's destination is an int64. *)
let law_map0_inside (b : Support.backend) (D.Any dt, s, (lead, trail), seed)
    =
  let module K = (val b.kernels) in
  let n = Array.fold_left ( * ) 1 s in
  let s = Iarray.of_array s in
  let strides = L.strides (L.contiguous s) in
  cover "sub-byte, mid-byte" (D.bits dt < 8 && lead mod (8 / D.bits dt) <> 0);
  if Rig.equal b.device Rig.host then
    cover "on the job's threads" (n >= 1_000_000);
  let inside (type v s) (dt : (v, s) D.t) k =
    let w = max 1 (D.bits dt / 8) in
    let whole = on b (seeded dt [| lead + n + trail |] seed) in
    let dst = A.v dt (L.v ~offset:lead ~strides s) (A.buffer whole) in
    let fresh = A.create b.device dt s in
    let before = bits_of (host whole) in
    match Support.map0 (module K) k ~dst:fresh with
    | A.Declined ->
        check_declined b (K0 k) (D.Any dt);
        equal ~msg:"inside" answer A.Declined (Support.map0 (module K) k ~dst);
        equal ~msg:"declined writes nothing" (array int) before
          (bits_of (host whole))
    | r ->
        equal ~msg:"fresh" answer A.Done r;
        let e = bits_of (host fresh) in
        equal ~msg:"inside" answer A.Done (Support.map0 (module K) k ~dst);
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
  inside dt (Support.Fill (String.init (Array.length e) (fun i -> Char.chr e.(i))));
  for axis = 0 to Iarray.length s - 1 do
    inside D.Int64 (Support.Iota axis)
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

(* A map given other destinations or loads than its program's outputs and
   operands is refused with [Shape_mismatch] and writes nothing. *)
let test_map_arity (b : Support.backend) () =
  let module K = (val b.kernels) in
  let p =
    P.v ~ins:[| D.Any D.Float32 |] [| P.In 0; Op2 (Binary Add, 0, 0) |]
      ~outs:[| 1 |]
  in
  let x = on b (A.of_array D.Float32 [| 3 |] [| 1.; 2.; 3. |]) in
  let dst () = on b (A.of_array D.Float32 [| 3 |] [| 9.; 9.; 9. |]) in
  let load = Nx_kernel.Plain x in
  List.iter
    (fun (msg, ys, loads) ->
      let dsts = Array.map (fun y -> A.Any y) ys in
      equal ~msg answer A.Shape_mismatch (K.map p ~dsts loads);
      Array.iter
        (fun y ->
          equal ~msg:(msg ^ ": untouched") (array float_exact) [| 9.; 9.; 9. |]
            (A.to_array (host y)))
        ys)
    [
      ("no destination", [||], [| load |]);
      ("two destinations", [| dst (); dst () |], [| load |]);
      ("no load", [| dst () |], [||]);
      ("two loads", [| dst () |], [| load; load |]);
    ]

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

(* Programs *)

(* A padded load and a coordinate: the map holds its loads' padded copies
   and, after them, the coordinate's index. On a domain of its own, whose
   thread has kept no scratch, each buffer the map takes is of its exact
   size. Each output is its load's element plus its column. *)
let test_padded_coordinate (b : Support.backend) () =
  let module K = (val b.kernels) in
  let rows = 4 and cols = 5 in
  let x =
    on b
      (A.of_array D.Int64 [| rows; cols |]
         (Array.init (rows * cols) Int64.of_int))
  in
  let p =
    P.v ~ins:[| D.Any D.Int64 |]
      [| P.In 0; P.Coord 1; Op2 (Binary Add, 0, 1) |]
      ~outs:[| 2 |]
  in
  let dst = A.create b.device D.Int64 [| rows; cols |] in
  let run () = K.map p ~dsts:[| A.Any dst |] [| unpadded x |] in
  match Domain.join (Domain.spawn run) with
  | A.Declined -> ()
  | A.Done ->
      equal (array int)
        (Array.init (rows * cols) (fun i -> i + (i mod cols)))
        (Array.map Int64.to_int (A.to_array (host dst)))
  | r -> failf "map answered %a" Nx_array_support.pp_answer r

(* In place *)

(* [k dst a] into a fresh destination, then into [a] itself as a C-contiguous
   copy, [a] being an operand [k] reads only at the result's own index: one
   answer, and on [Done] the same bits, any NaN for a NaN with [any_nan]. *)
let in_place ?any_nan ~msg (b : Support.backend) a k =
  let fresh = A.create b.device (A.dtype a) (L.shape (A.layout a)) in
  let want = k fresh a in
  let alias = on b (A.copy (host a)) in
  equal ~msg answer want (k alias alias);
  if want = A.Done then begin
    cover "computed" true;
    equal ~msg (list string) [] (differ ?any_nan (host fresh) (host alias))
  end

let binaries = List.filter (function P.Binary _ -> true | _ -> false) op2s

(* A [Plain] load of [map], of one node or more, may be the destination
   itself: the kernels read each index before they write it. *)
let law_in_place ?(ones = op1s) ?(kinds = binaries) (b : Support.backend)
    (Pair (x, y), seed) =
  let module K = (val b.kernels) in
  let dt = A.dtype x and shape = L.shape (A.layout x) in
  let x = on b x and y = on b y in
  List.iter
    (fun op ->
      in_place ~any_nan:(any_nan (K1 op)) ~msg:"one operand into itself" b x
        (fun dst x -> Support.map1 (module K) op ~dst x))
    (List.filter (fun op -> P.accepts1 op dt dt) ones);
  let map2 k ~dst x y = Support.map2 (module K) k ~dst x y in
  List.iter
    (fun k ->
      let msg into = strf "%s into %s" (name2 k) into in
      let blind = any_nan (K2 k) in
      in_place ~any_nan:blind ~msg:(msg "the first") b x (fun dst x ->
          map2 k ~dst x y);
      in_place ~any_nan:blind ~msg:(msg "the second") b y (fun dst y ->
          map2 k ~dst x y);
      in_place ~any_nan:blind ~msg:(msg "both") b x (fun dst x ->
          map2 k ~dst x x))
    (List.filter (fun k -> P.accepts2 k dt) kinds);
  let c = on b (seeded D.Bool shape seed) in
  let z = on b (seeded dt shape (seed + 1)) in
  in_place ~msg:"where into the second" b x (fun dst x ->
      Support.map3 (module K) Where ~dst c x y);
  in_place ~msg:"where into the third" b y (fun dst y ->
      Support.map3 (module K) Where ~dst c x y);
  if P.accepts3 Fma dt dt then
    in_place ~any_nan:true ~msg:"fma into the addend" b z (fun dst z ->
        Support.map3 (module K) Fma ~dst x y z);
  if P.accepts2 (Binary Add) dt then begin
    let p =
      P.v ~ins:[| D.Any dt; D.Any dt |]
        [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
        ~outs:[| 2 |]
    in
    in_place ~any_nan:true ~msg:"map into a load" b x (fun dst x ->
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

(* The contract's kinds

   Every kernel library gives a kind its doc, IEEE 754 or modular arithmetic
   defines exactly that result, and holds the others to nx_kinds.h's bounds
   (Nx_kernel.Prog's preamble). nx_kinds.h's functions give the exact
   results' bits, which test_kinds checks against their meaning. *)

(* The kinds of one operand the contract fixes at [dt]: every one of an
   integer and a boolean, a float's but the transcendentals, a complex
   number's Neg. *)
let exact1 (D.Any dt) =
  let float = D.is D.Float dt and complex = D.is D.Complex dt in
  List.filter
    (fun ((u : P.unary), _) ->
      match u with
      | Neg -> true
      | Recip | Abs | Sign | Sqrt | Floor | Ceil | Round | Trunc -> not complex
      | _ -> not (float || complex))
    unaries

(* The kinds of two operands it fixes: but Pow, Atan2 and a complex
   number's Mul and Fdiv. *)
let exact2 (D.Any dt) =
  let complex = D.is D.Complex dt in
  List.filter
    (function
      | P.Binary (Pow | Atan2) -> false
      | Binary (Mul | Fdiv) -> not complex
      | Binary _ | Compare _ -> true)
    op2s

(* nx_kinds.h's table: each float kind's bound in ulps at float32 and at
   float64, with its name there. *)
let float_bounds =
  let u k = Support.K1 (P.Unary k) and b k = Support.K2 (P.Binary k) in
  P.
    [
      (u Exp, "exp", 2, 2); (u Exp2, "exp2", 2, 2); (u Sin, "sin", 2, 2);
      (u Cos, "cos", 2, 2); (u Asin, "asin", 2, 2); (u Acos, "acos", 2, 2);
      (u Atan, "atan", 2, 2); (u Cosh, "cosh", 2, 2); (u Tanh, "tanh", 2, 2);
      (u Erf, "erf", 2, 2); (u Expm1, "expm1", 1, 2); (u Log, "log", 1, 2);
      (u Log2, "log2", 1, 2); (u Log1p, "log1p", 1, 2);
      (b Atan2, "atan2", 3, 2); (u Sinh, "sinh", 3, 2); (u Tan, "tan", 4, 2);
      (b Pow, "pow", 4, 2);
    ]

(* The C library's function of each kind at f64, the reference at f32. *)
let libm = function
  | "exp" -> Float.exp | "exp2" -> Float.exp2 | "sin" -> Float.sin
  | "cos" -> Float.cos | "asin" -> Float.asin | "acos" -> Float.acos
  | "atan" -> Float.atan | "cosh" -> Float.cosh | "tanh" -> Float.tanh
  | "erf" -> Float.erf | "expm1" -> Float.expm1 | "log" -> Float.log
  | "log2" -> Float.log2 | "log1p" -> Float.log1p | "sinh" -> Float.sinh
  | "tan" -> Float.tan
  | k -> invalid_arg ("libm: " ^ k)

let libm2 = function
  | "atan2" -> Float.atan2
  | "pow" -> Float.pow
  | k -> invalid_arg ("libm2: " ^ k)

(* [k] of [xs], and of [ys] for a kind of two operands, by [b]'s kernels
   into a fresh array of [dt]; [None] where they decline it. *)
let run_kind (type v s) (b : Support.backend) k (dt : (v, s) D.t) xs ys =
  let module K = (val b.kernels) in
  let n = Array.length xs in
  let x = on b (A.of_array dt [| n |] xs) in
  let dst = A.create b.device dt [| n |] in
  let r =
    match k with
    | Support.K1 op -> Support.map1 (module K) op ~dst x
    | K2 op -> Support.map2 (module K) op ~dst x (on b (A.of_array dt [| n |] ys))
    | K0 _ | K3 _ -> invalid_arg "run_kind"
  in
  match r with
  | A.Done -> Some (A.to_array (host dst))
  | A.Declined ->
      check_declined b k (D.Any dt);
      None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Ordered-bit ranks of binary32 values, -0 and +0 one apart. *)
let rank32 x =
  let b = Int32.to_int (Int32.bits_of_float x) in
  if b < 0 then -(b land 0x7FFF_FFFF) - 1 else b

let of_rank32 r =
  Int32.float_of_bits (Int32.of_int (if r < 0 then -(r + 1) lor 0x8000_0000 else r))

let ulps32 a c =
  if Float.is_nan a || Float.is_nan c then
    if Float.is_nan a && Float.is_nan c then 0 else max_int
  else abs (rank32 a - rank32 c)

(* The binary32 values [r], the C library's f64 value, may stand for: [r]
   rounded once, and its neighbour too where [r] is too close to a tie to
   tell. *)
let nearest32 r =
  let a = Int32.float_of_bits (Int32.bits_of_float r) in
  if Float.is_nan r || a = r || not (Float.is_finite a) then [ a ]
  else
    let n = of_rank32 (rank32 a + if r > a then 1 else -1) in
    let mid = (a +. n) /. 2. in
    if Float.abs (r -. mid) <= 0x1p-50 *. Float.abs r then [ a; n ] else [ a ]

(* Fails at the first of [got] past [bound] ulps from [want], as [dist]
   measures them, or a zero of the wrong sign where [want] is one. *)
let within_ulps ~name ~bound ~dist ~show got want =
  Array.iteri
    (fun i g ->
      let w = want.(i) in
      let zero = List.exists (fun w -> w = 0.) w in
      let wrong_zero =
        g = 0. && zero
        && List.for_all (fun w -> Float.sign_bit g <> Float.sign_bit w) w
      in
      let e = if wrong_zero then max_int else dist g w in
      if e > bound then
        failf "%s %s: %h, %d ulps from %s, past %d" name (show i) g
          (if e = max_int then -1 else e)
          (String.concat " or " (List.map (Printf.sprintf "%h") w))
          bound)
    got

(* Each float kind at float64 within its bound of the goldens' correctly
   rounded values. *)
let test_f64_kinds (b : Support.backend) () =
  List.iter
    (fun (k, name, _, bound) ->
      let rows = Array.of_list (Exact.golden name) in
      let arity = Array.length rows.(0) - 2 in
      let col j = Array.map (fun r -> r.(j)) rows in
      let xs = col 0 and ys = if arity = 2 then col 1 else col 0 in
      match run_kind b k D.Float64 xs ys with
      | None -> ()
      | Some got ->
          let show i =
            if arity = 2 then Printf.sprintf "%h %h" xs.(i) ys.(i)
            else Printf.sprintf "%h" xs.(i)
          in
          within_ulps ~name ~bound
            ~dist:(fun g w -> Exact.ulps64 g (List.hd w))
            ~show got
            (Array.map (fun r -> [ r.(arity) ]) rows))
    float_bounds

(* Operands where C99 Annex F gives a kind a special value. *)
let specials32 =
  [| 0.; -0.; infinity; neg_infinity; nan; 1.; -1.; 0.5; 2.; 0x1p-149 |]

let special_pairs =
  let s = [ 0.; -0.; infinity; neg_infinity; nan; 1.; -1.; 0.5; 2.; -3. ] in
  Array.of_list (List.concat_map (fun x -> List.map (fun y -> (x, y)) s) s)

(* The float32 values at which nx_kinds.h's f32 kinds switch formulas,
   saturate or clamp, each with its neighbours of either sign: every
   (k + 1/2) ln 2 and k + 1/2 below the exponentials' overflow and every
   sqrt 2 2^k, where a reduction switches, with eight; every multiple of
   pi/4 below the trigonometric switch at 2^15, with four; and the kinds'
   constants with eight. *)
let switch_points =
  let around k v =
    let b = Int32.to_int (Int32.bits_of_float (Float.abs v)) land 0x7FFF_FFFF in
    List.concat_map
      (fun d ->
        let p = b + d in
        if p < 0 || p >= 0x8000_0000 then []
        else [ Int32.float_of_bits (Int32.of_int p); Int32.float_of_bits (Int32.of_int (p lor 0x8000_0000)) ])
      (List.init ((2 * k) + 1) (fun i -> i - k))
  in
  let ln2 = 0.69314718055994531 and pio4 = 0.78539816339744831 in
  let halves =
    List.init 153 (fun k -> (Float.of_int k +. 0.5) *. ln2)
    @ List.init 151 (fun k -> Float.of_int k +. 0.5)
  in
  let rec multiples k =
    let v = Float.of_int k *. pio4 in
    if v >= 32768. then [] else v :: multiples (k + 1)
  in
  let sqrt2 = List.init 277 (fun k -> Float.ldexp 1.4142135623730951 (k - 149)) in
  let constants =
    [
      88.72283935546875; 89.; -104.; 103.97208; 129.; 151.; 128.; 149.; 126.;
      17.328680; 26.; 709.782712893384; 708.3964185322641; 745.1332191019411;
      710.; 746.; 1025.; 1076.; 1024.; 1074.; 40.; 37.42994775023705; 86.; 708.;
      0.70710678118654757; 1.4142135623730951; 0.29289321881345243;
      0.41421356237309515; 32768.; 268435456.; 0.5; 1.; 0.4375; 0.6875; 1.1875;
      2.4375; 0.875; 3.92; 1.75; 2.75; 4.; 5.95; 5.921587195794507; 90.;
      89.41598629223294; 712.; 710.4758600739439; 9.010913339;
      19.06154746539849; 13.; 22.18070977791825; 44.3614195558365;
      88.722839111672999; 0x1p-12; 0x1p-24; 0x1p-25; 0x1p-27; 0x1p-53; 0x1p-54;
      0x1p-100; 0x1p24; 0x1p23; 0x1p52; 0x1p53; 2.; 3.;
    ]
  in
  Array.of_list
    (List.concat_map (around 8) (halves @ sqrt2 @ constants)
    @ List.concat_map (around 4) (multiples 1))

(* Each float kind at float32 within its bound of the C library's f64
   function rounded once: at Annex F's operands, at the switch points and
   at every 2^14th bit pattern from [offset], every binade of both signs
   reached, and for a kind of two operands at Annex F's pairs and pairs
   from [seed], half of any patterns and half where [pow] and [atan2]
   neither overflow nor underflow. *)
let law_f32_kinds (b : Support.backend) (offset, seed) =
  let step = 1 lsl 14 in
  let xs =
    Array.concat
      [
        specials32;
        switch_points;
        Array.init ((1 lsl 32) / step) (fun i ->
            Int32.float_of_bits (Int32.of_int ((i * step) + offset)));
      ]
  in
  let r = Random.State.make [| seed |] in
  let pair i =
    if i land 1 = 0 then
      let any () = Int32.float_of_bits (Random.State.bits32 r) in
      (any (), any ())
    else
      let x = Float.ldexp (Random.State.float r 1.) (Random.State.int r 9 - 4) in
      (Exact.round_f32 x, Exact.round_f32 (Random.State.float r 40. -. 20.))
  in
  let ps = Array.append special_pairs (Array.init (1 lsl 14) pair) in
  List.iter
    (fun (k, name, bound, _) ->
      let two = match k with Support.K2 _ -> true | _ -> false in
      let xs, ys = if two then (Array.map fst ps, Array.map snd ps) else (xs, xs) in
      let want =
        if two then Array.map2 (fun x y -> nearest32 (libm2 name x y)) xs ys
        else Array.map (fun x -> nearest32 (libm name x)) xs
      in
      match run_kind b k D.Float32 xs ys with
      | None -> ()
      | Some got ->
          let show i =
            if two then Printf.sprintf "%h %h" xs.(i) ys.(i)
            else Printf.sprintf "%h" xs.(i)
          in
          within_ulps ~name ~bound
            ~dist:(fun g w ->
              List.fold_left (fun m w -> min m (ulps32 g w)) max_int w)
            ~show got want)
    float_bounds

(* nx_kinds.h's complex section: each transcendental kind's normwise bound
   in units of its parts' u, with its golden's name. pow's is
   [3 + 3 |w log z|]. *)
let complex_bounds =
  P.
    [
      (Sqrt, "sqrt", 3.); (Sign, "sign", 3.); (Exp, "exp", 4.);
      (Exp2, "exp2", 4.); (Log, "log", 4.); (Log2, "log2", 4.);
      (Asin, "asin", 5.); (Acos, "acos", 5.); (Atan, "atan", 5.);
      (Asinh, "asinh", 5.); (Acosh, "acosh", 5.); (Atanh, "atanh", 5.);
      (Sin, "sin", 6.); (Cos, "cos", 6.); (Tan, "tan", 6.); (Sinh, "sinh", 6.);
      (Cosh, "cosh", 6.); (Tanh, "tanh", 6.);
    ]

let complex_of (re, im) = { Complex.re; im }

(* Each complex transcendental kind, and pow, within its bound of the
   goldens' correctly rounded values, at complex128 and, where the operands
   are binary32's and the value's modulus a normal one, at complex64. *)
let test_complex_kinds (b : Support.backend) () =
  let check name k ~bound rows =
    let rows = Array.of_list rows in
    let n = if rows = [||] then 0 else Array.length rows.(0) - 4 in
    let op j r = { Complex.re = r.(2 * j); im = r.((2 * j) + 1) } in
    let at u (type s) (dt : (Complex.t, s) D.t) rows =
      let xs = Array.map (op 0) rows in
      let ys = if n = 4 then Array.map (op 1) rows else xs in
      match run_kind b k dt xs ys with
      | None -> ()
      | Some got ->
          Array.iteri
            (fun i (g : Complex.t) ->
              let row = rows.(i) in
              let e = Exact.normwise u (g.re, g.im) (Exact.golden_value row n) in
              let bound = bound row in
              if not (e <= bound) then
                failf "%s %s: %h %h, %gu past %gu" name
                  (String.concat " "
                     (List.map (Printf.sprintf "%h")
                        (Array.to_list (Array.sub row 0 n))))
                  g.re g.im e bound)
            got
    in
    at 0x1p-53 D.Complex128 rows;
    at 0x1p-24 D.Complex64
      (Array.of_list
         (List.filter
            (fun r -> Exact.single_row (Array.sub r 0 n) (Exact.golden_value r n))
            (Array.to_list rows)))
  in
  List.iter
    (fun (u, name, bound) ->
      check name (Support.K1 (P.Unary u)) ~bound:(fun _ -> bound)
        (Exact.complex_golden name))
    complex_bounds;
  let pow_bound r =
    let m (re, im) = Float.hypot re im in
    let z = (r.(0), r.(1)) and w = (r.(2), r.(3)) in
    3. +. (3. *. m w *. m (Float.log (m z), Float.atan2 r.(1) r.(0)))
  in
  check "pow" (Support.K2 (P.Binary Pow)) ~bound:pow_bound
    (Exact.complex_golden "pow")

(* Mul within 2u, Fdiv and Recip within 4u of the exact values at complex64
   and complex128, over parts in [2^-40, 2^40]; an operand with a NaN part
   makes both parts NaN. *)
let law_complex_arith (b : Support.backend) seed =
  let r = Random.State.make [| seed |] in
  let part () =
    let m = 1. +. Random.State.float r 1. in
    let m = if Random.State.bool r then -.m else m in
    Float.ldexp m (Random.State.int r 81 - 40)
  in
  let n = 4096 in
  let draw () =
    Array.init n (fun i ->
        let re = part () and im = part () in
        match i mod 32 with
        | 0 -> (nan, im)
        | 1 -> (re, nan)
        | _ -> (re, im))
  in
  let xs = draw () and ys = draw () in
  let has_nan (a, c) = Float.is_nan a || Float.is_nan c in
  let either x y = has_nan x || has_nan y in
  let at (type s) u round (dt : (Complex.t, s) D.t) =
    let xs = Array.map round xs and ys = Array.map round ys in
    let zs a = Array.map complex_of a in
    let check name bound exact nan_in = function
      | None -> ()
      | Some got ->
          Array.iteri
            (fun i (g : Complex.t) ->
              let x = xs.(i) and y = ys.(i) in
              let show () =
                Printf.sprintf "%s %h %h, %h %h: %h %h" name (fst x) (snd x)
                  (fst y) (snd y) g.re g.im
              in
              if nan_in x y then begin
                if not (Float.is_nan g.re && Float.is_nan g.im) then
                  failf "%s, both parts not NaN" (show ())
              end
              else
                let e = Exact.normwise u (g.re, g.im) (exact x y) in
                if not (e <= bound) then
                  failf "%s, %gu past %gu" (show ()) e bound)
            got
    in
    let two k = run_kind b (Support.K2 (P.Binary k)) dt (zs xs) (zs ys) in
    check "mul" 2. Exact.exact_mul either (two Mul);
    check "fdiv" 4. Exact.exact_div either (two Fdiv);
    check "recip" 4.
      (fun _ y -> Exact.exact_div (1., 0.) y)
      (fun _ y -> has_nan y)
      (run_kind b (Support.K1 (P.Unary Recip)) dt (zs ys) (zs ys))
  in
  let f32 (a, c) = (Exact.round_f32 a, Exact.round_f32 c) in
  at 0x1p-53 Fun.id D.Complex128;
  at 0x1p-24 f32 D.Complex64

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
      test "a transposed copy past the caches is bits for bits"
        (unit (test_streamed_copies b));
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
        (run (law_declined_map1 b));
      test "a declined contraction writes nothing"
        (unit (test_declined_contract b));
      prop "bitcast keeps every element's bits" (same_width case)
        (run (fun ((c, _) as x) ->
             covers c;
             law_bitcast b x));
      prop ~count:4 "idiv and mod by a repeated power of two are nx_kinds.h's"
        Gen.nat
        (run (law_power_of_two_divisors b));
      prop "Threefry's rows are nx_kinds.h's around their vector blocks"
        Gen.(triple (int_range 0 80) (int_range 0 2) nat)
        (run (law_threefry_rows b));
      prop "where picks bytes and fma is nx_kinds.h's"
        (Gen.pair pairs Gen.nat)
        (run (law_map3 b));
      prop "fill stores its element and iota each index"
        (Gen.triple dtypes shape Gen.nat)
        (run (law_map0 b));
      prop
        ~examples:(if Rig.equal b.device Rig.host then large_fills else [])
        "fill and iota store inside a buffer and nothing around it"
        fill_insides
        (run (law_map0_inside b));
      test "a declined map writes nothing" (unit (test_declined_map b));
      test "a map of other destinations or loads than its program's is refused"
        (unit (test_map_arity b));
      test "a map reads a load identical to a destination before writing it"
        (unit (test_map_identical_view b));
      test "a map of a padded load reads its coordinates"
        (unit (test_padded_coordinate b));
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
      prop "a kind of one operand the contract fixes gives its result" case
        (run (fun (Case x as c) ->
             covers c;
             law_map1 ~kinds:(exact1 (D.Any (A.dtype x))) b c));
      prop "a kind of two operands the contract fixes gives its result" pairs
        (run (fun (Pair (x, _) as p) ->
             covers (Case x);
             law_map2 ~kinds:(exact2 (D.Any (A.dtype x))) b p));
      test "float kinds at float64 are within nx_kinds.h's bounds of the goldens"
        (unit (test_f64_kinds b));
      prop ~count:1 "float kinds at float32 are within nx_kinds.h's bounds"
        Gen.(pair (int_range 0 ((1 lsl 14) - 1)) int)
        (run (law_f32_kinds b));
      test "complex kinds are within nx_kinds.h's bounds of the goldens"
        (unit (test_complex_kinds b));
      prop ~count:4
        "complex mul, fdiv and recip are within their bounds, NaN parts NaN"
        Gen.int
        (run (law_complex_arith b));
    ]

let () = exit (run "nx_kernel.kernels" (List.map laws Support.backends))
