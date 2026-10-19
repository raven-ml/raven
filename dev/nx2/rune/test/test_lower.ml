(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The dispatch and the loads of the lowering, through operations run on the
   host against nx's eager result. *)

open Windtrap
module D = Nx_array.Dtype
module P = Nx.Prim
module K = Nx_kernel.Prog

let copy x = Lowered.run (fun () -> P.eval ~by:"t" (P.Copy x))

let smoke =
  group "copies"
    [
      test "a transposed view is its elements" (fun () ->
          let x =
            Nx.transpose
              (Nx.create D.Int32 [| 2; 3 |] [| 0l; 1l; 2l; 3l; 4l; 5l |])
          in
          equal (array int32) (Nx.to_array x) (copy x));
      test "an int4 value is its elements" (fun () ->
          let x = Nx.create D.Int4 [| 4 |] [| -8; -1; 0; 7 |] in
          equal (array int) (Nx.to_array x) (copy x));
      test "a map of coordinates is their values" (fun () ->
          let prog = Nx_kernel.Prog.v ~ins:[||] [| Coord 1 |] ~outs:[| 0 |] in
          let map () =
            let y, () =
              P.eval ~by:"t"
                (P.Map
                   {
                     layout = Nx_array.Layout.contiguous [| 2; 3 |];
                     prog;
                     outs = [ D.Int64 ];
                     loads = Iarray.of_array [||];
                   })
            in
            y
          in
          equal (array int64) [| 0L; 1L; 2L; 0L; 1L; 2L |] (Lowered.run map));
    ]

(* Movements and padded loads move elements: each gives eager's elements, bit
   for bit, on every dtype. *)

module M = Nx_array.Move

let name (D.Any d) = D.name d

let hex b =
  String.concat ""
    (List.init (String.length b) (fun i ->
         Printf.sprintf "%02x" (Char.code b.[String.length b - 1 - i])))

(* [agrees dt f] compares the bits of [f]'s elements, eager and lowered. *)
let agrees (type v s) (dt : (v, s) D.t) (f : unit -> (v, s, _) Nx.t) =
  let bits xs = Array.to_list (Array.map (fun x -> hex (K.bits dt x)) xs) in
  equal (list string) (bits (Nx.to_array (f ()))) (bits (Lowered.run f))

(* A value of [dt] and shape [s] of random bits. *)
let value (type v s) (dt : (v, s) D.t) s st : (v, s, Nx.host) Nx.t =
  let n = Array.fold_left ( * ) 1 s in
  let byte () = Random.State.int st 256 in
  match dt with
  | D.Bool -> Nx.create dt s (Array.init n (fun _ -> Random.State.bool st))
  | D.Bit -> Nx.create dt s (Array.init n (fun _ -> Random.State.bool st))
  | D.Int4 | D.Uint4 | D.Float4_e2m1fn ->
      Nx.bitcast dt
        (Nx.create D.Uint4 s (Array.init n (fun _ -> Random.State.int st 16)))
  | _ ->
      let w = D.bytes dt 1 in
      if w = 1 then
        Nx.bitcast dt (Nx.create D.Uint8 s (Array.init n (fun _ -> byte ())))
      else
        Nx.bitcast dt
          (Nx.create D.Uint8 (Array.append s [| w |])
             (Array.init (n * w) (fun _ -> byte ())))

let int st lo hi = lo + Random.State.int st (hi - lo + 1)
let shape st = Array.init (int st 1 3) (fun _ -> int st 0 4)

(* Movements *)

type kind = Reshape | Broadcast | Permute | Slice | Window

let kinds = [ Reshape; Broadcast; Permute; Slice; Window ]

let kind_name = function
  | Reshape -> "reshape"
  | Broadcast -> "broadcast"
  | Permute -> "permute"
  | Slice -> "slice"
  | Window -> "window"

let divisor st n =
  let ds = List.filter (fun d -> n mod d = 0) (List.init n succ) in
  List.nth ds (Random.State.int st (List.length ds))

let reshape st n =
  if n = 0 then [| int st 0 3; 0 |]
  else
    let a = divisor st n in
    let b = divisor st (n / a) in
    [| a; b; n / a / b |]

let broadcast st s =
  let lead = Array.init (int st 0 2) (fun _ -> int st 0 3) in
  Array.append lead (Array.map (fun d -> if d = 1 then int st 0 3 else d) s)

let permutation st r =
  let p = Array.init r Fun.id in
  for i = r - 1 downto 1 do
    let j = Random.State.int st (i + 1) in
    let t = p.(i) in
    p.(i) <- p.(j);
    p.(j) <- t
  done;
  p

let range st d =
  let step = (if Random.State.bool st then 1 else -1) * int st 1 3 in
  if d = 0 then { M.start = 0; count = 0; step }
  else
    let start = Random.State.int st d in
    let most =
      if step > 0 then ((d - 1 - start) / step) + 1 else (start / -step) + 1
    in
    { M.start; count = int st 0 most; step }

(* Windows on some axes of a shape of rank [r], at least one. *)
let windows st r =
  let some = List.filter (fun _ -> Random.State.bool st) (List.init r Fun.id) in
  let axes = if some = [] then [ Random.State.int st r ] else some in
  Array.of_list
    (List.map
       (fun axis ->
         { M.axis; size = int st 1 3; step = int st 1 2; dilation = int st 1 2 })
       axes)

let movement st k s =
  let a = Iarray.of_array in
  match k with
  | Reshape -> M.Reshape (a (reshape st (Array.fold_left ( * ) 1 s)))
  | Broadcast -> M.Broadcast (a (broadcast st s))
  | Permute -> M.Permute (a (permutation st (Array.length s)))
  | Slice -> M.Slice (a (Array.map (range st) s))
  | Window -> M.Window (a (windows st (Array.length s)))

let moved m x = P.eval ~by:"t" (P.Move (m, x))

let cover_move m s =
  let ranges = match m with M.Slice rs -> Iarray.to_list rs | _ -> [] in
  cover "a reversing slice"
    (List.exists (fun (r : M.range) -> r.step < 0 && r.count > 1) ranges);
  cover "an empty slice" (List.exists (fun (r : M.range) -> r.count = 0) ranges);
  cover "a reversed empty axis"
    (List.exists2
       (fun (r : M.range) d -> r.step < 0 && d = 0)
       ranges
       (if ranges = [] then [] else Array.to_list s));
  cover "an empty value" (Array.mem 0 s)

let classes =
  [
    ("float", fun (D.Any d) -> D.is D.Float d);
    ("complex", fun (D.Any d) -> D.is D.Complex d);
    ("boolean", fun (D.Any d) -> D.is D.Boolean d);
    ( "integer",
      fun (D.Any d) ->
        not (D.is D.Float d || D.is D.Complex d || D.is D.Boolean d) );
    ("sub-byte", fun (D.Any d) -> D.bits d < 8);
  ]

let cover_dtype d =
  cover (name d) true;
  List.iter (fun (label, is) -> cover label (is d)) classes

let movements =
  let pp ppf (d, k, seed) =
    Format.fprintf ppf "(%s, %s, %d)" (name d) (kind_name k) seed
  in
  let gen =
    Gen.with_pp pp
      (Gen.triple (Gen.of_list D.all) (Gen.of_list kinds)
         (Gen.int_range 0 1_000_000))
  in
  prop "movements give eager's bits"
    ~count:3
      (* Every dtype, the kinds in turn, from seeds whose draws reach each
         cover. *)
    ~examples:
      (List.mapi (fun i d -> (d, List.nth kinds (i mod 5), i + 30)) D.all)
    gen
    (fun ((D.Any dt as d), k, seed) ->
      cover_dtype d;
      List.iter (fun k' -> cover (kind_name k') (k = k')) kinds;
      let st = Random.State.make [| seed |] in
      let rec draw () =
        let s = shape st in
        let m = movement st k s in
        match M.shape m s with
        | _ -> (s, m)
        | exception Invalid_argument _ -> draw ()
      in
      let s, m = draw () in
      cover_move m s;
      let x = value dt s st in
      agrees dt (fun () -> moved m x))

(* Padded loads *)

type fill = Bits | Negative_zero | Nan | Negative_nan | Signalling_nan

let fills = [ Bits; Negative_zero; Nan; Negative_nan; Signalling_nan ]

let fill_name = function
  | Bits -> "a fill of random bits"
  | Negative_zero -> "a -0. fill"
  | Nan -> "a NaN fill with a payload"
  | Negative_nan -> "a negative NaN fill with a payload"
  | Signalling_nan -> "a signalling NaN fill"

let special = function
  | Bits | Negative_zero -> -0.
  | Nan -> Int64.float_of_bits 0x7FF8_1234_5000_0000L
  | Negative_nan -> Int64.float_of_bits 0xFFFA_BCDE_0000_0000L
  | Signalling_nan -> Int64.float_of_bits 0x7FF0_0000_2000_0000L

(* The fill [f] of [dt]: special floats on float dtypes, a complex dtype's
   imaginary part the real part negated, random bits otherwise. *)
let fill (type v s) (dt : (v, s) D.t) f st : v =
  let bits () = (Nx.to_array (value dt [| 1 |] st)).(0) in
  if f = Bits then bits ()
  else
    match dt with
    | D.Float64 -> special f
    | D.Float32 -> special f
    | D.Float16 -> special f
    | D.Bfloat16 -> special f
    | D.Float8_e4m3fn -> special f
    | D.Float8_e5m2 -> special f
    | D.Float4_e2m1fn -> special f
    | D.Complex64 -> { re = special f; im = Float.neg (special f) }
    | D.Complex128 -> { re = special f; im = Float.neg (special f) }
    | _ -> bits ()

let identity dt =
  Nx_kernel.Prog.v
    ~ins:(Iarray.of_array [| D.Any dt |])
    (Iarray.of_array [| Nx_kernel.Prog.In 0 |])
    ~outs:(Iarray.of_array [| 0 |])

let padded (type v s) (x : (v, s, _) Nx.t) (fill : v) pad shape =
  let dt = Nx.dtype x in
  let y, () =
    P.eval ~by:"t"
      (P.Map
         {
           layout = Nx_array.Layout.contiguous shape;
           prog = identity dt;
           outs = [ dt ];
           loads = Iarray.of_array [| P.Padded { x; fill; pad } |];
         })
  in
  y

let random_pad st s =
  let r = Array.length s in
  let a = Iarray.of_array in
  let ends () = a (Array.init r (fun _ -> int st (-2) 2)) in
  let pad =
    {
      Nx_kernel.lo = ends ();
      hi = ends ();
      interior = a (Array.init r (fun _ -> int st 0 2));
      windows = a [||];
    }
  in
  if Random.State.bool st then pad else { pad with windows = a (windows st r) }

let cover_pad (pad : Nx_kernel.pad) out =
  let exists f a = Iarray.exists f a in
  cover "a cropping end"
    (exists (fun n -> n < 0) pad.lo || exists (fun n -> n < 0) pad.hi);
  cover "a padding end"
    (exists (fun n -> n > 0) pad.lo || exists (fun n -> n > 0) pad.hi);
  cover "interior padding" (exists (fun n -> n > 0) pad.interior);
  cover "windows" (Iarray.length pad.windows > 0);
  cover "an empty load" (Array.mem 0 out)

let is_float (D.Any d) = D.is D.Float d || D.is D.Complex d

(* Each float dtype with one special fill, each fill reaching several. *)
let pad_examples =
  let specials = [ Signalling_nan; Negative_zero; Nan; Negative_nan ] in
  let floats = List.filter is_float D.all in
  List.mapi (fun i d -> (d, Bits, i)) D.all
  @ List.mapi (fun i d -> (d, List.nth specials (i mod 4), 100 + i)) floats

let loads =
  let pp ppf (d, f, seed) =
    Format.fprintf ppf "(%s, %s, %d)" (name d) (fill_name f) seed
  in
  let gen =
    Gen.with_pp pp
      (Gen.triple (Gen.of_list D.all) (Gen.of_list fills)
         (Gen.int_range 0 1_000_000))
  in
  prop "padded loads give eager's bits" ~count:3 ~examples:pad_examples gen
    (fun ((D.Any dt as d), f, seed) ->
      let f = if is_float d then f else Bits in
      cover_dtype d;
      List.iter (fun f' -> cover (fill_name f') (f = f')) fills;
      let st = Random.State.make [| seed |] in
      let rec draw () =
        let s = shape st in
        let pad = random_pad st s in
        match Nx_kernel.Shape.padded pad s with
        | Ok out -> (s, pad, out)
        | Error _ -> draw ()
      in
      let s, pad, out = draw () in
      cover_pad pad out;
      let x = value dt s st in
      let fill = fill dt f st in
      agrees dt (fun () -> padded x fill pad out))

let () = exit (run "Lower" [ smoke; movements; loads ])
