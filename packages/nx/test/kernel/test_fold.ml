(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Folds through every kernel library the host runs: each result element against
   a reference that sums, with no +0 before the first, the elements the load
   reads from it in C order of their taps, +0 where it reads none, bit for bit
   or both NaN, through views of the operand, at every dtype but the booleans,
   on the job's threads as on one; each part of a float or complex fold within
   its bound of the exact sum; maps of padded loads through the same geometry, a
   loaded element the element the fold adds it into, or the fill; its declines
   and refusals; and, on nx.cpu, no byte read or written past the operand or the
   result. *)

open Windtrap
open Elements
module Support = Nx_kernels_support

(* A fold's case; [load] views the operand of the map whose padded load
   reads the fold's geometry. *)
type case = {
  shape : int array;
  pad : Nx_kernel.pad;
  x : A.any;
  views : string list;
  load : view;
}

let pp_case ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf
    "fold into %a of %a %a: lo %a hi %a interior %a, windows %s (%s), load \
     (%s)"
    pp_ints c.shape D.pp (A.dtype x) pp_ints (shape_of c.x) pp_ints
    (Iarray.to_array c.pad.lo) pp_ints (Iarray.to_array c.pad.hi) pp_ints
    (Iarray.to_array c.pad.interior)
    (String.concat "; "
       (Iarray.to_list
          (Iarray.map
             (fun (w : M.window) ->
               Printf.sprintf "%d:%dx%d+%d" w.axis w.size w.step w.dilation)
             c.pad.windows)))
    (String.concat ", " c.views)
    (String.concat ", " (view_names c.load))

(* The shape a load by [p] gives an array of shape [s], if any. *)
let loaded (p : Nx_kernel.pad) s = Result.to_option (Nx_kernel.Shape.padded p s)

let folded = List.filter (fun (D.Any d) -> not (D.is D.Boolean d)) D.all

(* How a case's first axis leads: as drawn; unwindowed and unpadded, so
   that the fold's units are slabs of the result; or unwindowed and padded,
   so that the fold starts the whole result first. The last two window the
   last axis, of a result of two or three axes. [Pad] draws no window and no
   interior padding on any axis: the fold is a pad, as a slice's transpose
   is. Unless [last] draws its last extent, its result has more than one
   element; its operand keeps an element on each axis, which may all be
   padding. [Drawn] draws the empty pads. *)
type lead = Drawn | Slabs | Padded | Pad

(* The complex and sub-byte dtypes, which the laws cover, drawn more often than
   the others, so that every run of a hundred cases reaches them. *)
let weight (D.Any d) =
  if D.is D.Complex d then 10 else if D.bits d < 8 then 6 else 2

let weighted dtypes =
  Gen.frequency (List.map (fun d -> (weight d, Gen.constant d)) dtypes)

(* A [Pad]'s dtype: one of whole bytes 7 times in 8, any 1 time in 8, so of
   whole bytes 0.966 of the time over [folded]. [Pad] leads one draw in
   five, so that 0.193 of the draws are "a pad of whole bytes", and 100
   cases cover it but with probability below 6 10^-10; discarded draws of
   the other leads and [Drawn]'s pads only raise it. *)
let pad_dtype dtypes =
  let bytes = List.filter (fun (D.Any d) -> D.bits d >= 8) dtypes in
  if bytes = [] then weighted dtypes
  else Gen.frequency [ (7, weighted bytes); (1, weighted dtypes) ]

(* A finite number of full precision within 2^±4 in magnitude: sums of three or
   more such have bits that depend on the order of their terms. *)
let mixed rs =
  Float.ldexp (Random.State.float rs 2. -. 1.) (Random.State.int rs 9 - 4)

(* An operand's element: {!Elements.element}'s half the time, else, at float32,
   float64 and the complex dtypes, a [mixed] number. *)
let term d rs =
  let mixed () = mixed rs in
  if Random.State.bool rs then element d rs
  else
    match d with
    | D.Any D.Float32 -> f32 (mixed ())
    | D.Any D.Float64 -> f64 (mixed ())
    | D.Any D.Complex64 -> f32 (mixed ()) ^ f32 (mixed ())
    | D.Any D.Complex128 -> f64 (mixed ()) ^ f64 (mixed ())
    | _ -> element d rs

(* [last], where given, draws the result's last extent. *)
let case_gen ?(dtypes = folded) ?last () =
  let open Gen in
  let* lead =
    frequency
      [
        (2, constant Drawn);
        (1, constant Slabs);
        (1, constant Padded);
        (1, constant Pad);
      ]
  in
  let* d = if lead = Pad then pad_dtype dtypes else weighted dtypes in
  let first_two = Array.mapi (fun a v -> if a = 0 then max v 2 else v) in
  let* shape =
    match lead with
    | Drawn -> array ~size:(int_range 0 3) (int_range 0 5)
    | Pad -> map first_two (array ~size:(int_range 1 3) (int_range 1 5))
    | Slabs | Padded ->
        map first_two (array ~size:(int_range 2 3) (int_range 1 5))
  in
  let* shape =
    match last with
    | Some n when Array.length shape > 0 ->
        let+ n = n in
        let last = Array.length shape - 1 in
        Array.mapi (fun a v -> if a = last then n else v) shape
    | _ -> constant shape
  in
  let r = Array.length shape in
  let small = int_range (-2) 3 in
  let* lo = array ~size:(constant r) small in
  let* hi = array ~size:(constant r) small in
  let* interior = array ~size:(constant r) (int_range 0 2) in
  let* on = array ~size:(constant r) bool in
  let* sizes = array ~size:(constant r) (int_range 1 3) in
  let* steps = array ~size:(constant r) (int_range 1 3) in
  let* dilations = array ~size:(constant r) (int_range 1 2) in
  let leads = lead = Slabs || lead = Padded in
  let first a v z = if leads && a = 0 then z else v in
  let lo0 = if lead = Padded then 1 else 0 in
  let lo = Array.mapi (fun a v -> first a v lo0) lo in
  let hi = Array.mapi (fun a v -> first a v (max v 0)) hi in
  (* A [Pad]'s loaded extent [shape.(a) + lo.(a) + hi.(a)] is at least 1. *)
  let hi =
    if lead <> Pad then hi
    else Array.mapi (fun a v -> max v (1 - shape.(a) - lo.(a))) hi
  in
  let interior =
    Array.mapi (fun a v -> if lead = Pad then 0 else first a v 0) interior
  in
  let on =
    Array.mapi
      (fun a v ->
        lead <> Pad
        && ((leads && a = r - 1) || first a v false))
      on
  in
  let windows =
    Array.of_list
      (List.filter_map
         (fun a ->
           if on.(a) then
             Some
               {
                 M.axis = a;
                 size = sizes.(a);
                 step = steps.(a);
                 dilation = dilations.(a);
               }
           else None)
         (List.init r Fun.id))
  in
  let pad =
    let lo = Iarray.of_array lo and hi = Iarray.of_array hi in
    let interior = Iarray.of_array interior in
    { Nx_kernel.lo; hi; interior; windows = Iarray.of_array windows }
  in
  let* seed = int in
  let xs = loaded pad shape in
  let rx = match xs with Some s -> Array.length s | None -> 0 in
  let* v = view_of rx in
  let+ load = view_of r in
  match xs with
  | None -> None
  | Some xs ->
      let rs = Random.State.make [| seed |] in
      let x = operand d xs v (fun _ -> term d rs) in
      Some { shape; pad; x; views = view_names v; load }

(* Folds into a destination of other elements: the fold writes each. *)
let fold_on (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (A.Any x) = c.x in
  let d = D.Any (A.dtype x) and rs = Random.State.make [| 5 |] in
  let dst =
    A.expect (A.dtype x)
      (operand d c.shape (plain (Array.length c.shape)) (fun _ -> element d rs))
  in
  let x = Support.on b x and dst = Support.on b dst in
  match K.fold c.pad ~dst x with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> failf "%s declined a fold" b.name
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* The result element each operand element lands on, or [None] for one the load
   reads from padding. *)
let target c xi =
  let r = Array.length c.shape in
  let q = Array.sub xi 0 r in
  Iarray.iteri
    (fun k (w : M.window) ->
      q.(w.axis) <- (xi.(w.axis) * w.step) + (xi.(r + k) * w.dilation))
    c.pad.windows;
  let i =
    Array.mapi
      (fun a q ->
        let t = q - Iarray.get c.pad.lo a
        and s = Iarray.get c.pad.interior a + 1 in
        if t < 0 || t mod s <> 0 || t / s >= c.shape.(a) then -1 else t / s)
      q
  in
  if Array.exists (fun v -> v < 0) i then None else Some (flat c.shape i)

(* The number of the operand's elements the load reads from each result
   index. *)
let reached c =
  let xs = shape_of c.x in
  let counts = Array.make (total c.shape) 0 in
  for k = 0 to total xs - 1 do
    match target c (index_of xs k) with
    | Some y -> counts.(y) <- counts.(y) + 1
    | None -> ()
  done;
  counts

(* Whether [e], an element of [d], is nonzero and of the least normal
   magnitude or below, at float32 or bfloat16 or in a part of a complex64:
   a sum where a library flushes may differ there. *)
let tiny (D.Any dt) e =
  let small exp man = (exp = 0 && man <> 0) || (exp = 1 && man = 0) in
  let f32 at =
    let b = Int32.to_int (String.get_int32_le e at) land 0xFFFFFFFF in
    small ((b lsr 23) land 0xFF) (b land 0x7FFFFF)
  in
  match dt with
  | D.Float32 -> f32 0
  | D.Complex64 -> f32 0 || f32 4
  | D.Bfloat16 ->
      let b = String.get_uint16_le e 0 in
      small ((b lsr 7) land 0xFF) (b land 0x7F)
  | _ -> false

(* [a] plus [b], element by element, two arrays of [d] and shape [s], by
   nx.cpu's map of Prog.Add, which test_kernels holds to nx_kinds.h's. *)
let added d s a b =
  let (A.Any x) = of_elements d s a in
  let y = A.expect (A.dtype x) (of_elements d s b) in
  let dst = A.create Rig.host (A.dtype x) (Iarray.of_array s) in
  match Support.map2 (module Nx_cpu) (Binary Add) ~dst x y with
  | A.Done -> elements (A.Any dst)
  | r -> failf "an add answered %a" Nx_array_support.pp_answer r

(* The fold's result by its reference, and for each element whether an addition
   into it took or gave a [tiny] element. It adds a tap at a time: at one tap
   the load reads each result index from one operand index at most, so a tap's
   additions are one elementwise Add over the result. An element's first term is
   taken as it is, with no +0 before it. Where [reversed], the taps come in the
   other order, an association no library states. *)
let reference ?(reversed = false) c =
  let d = dtype_of c.x in
  let xs = shape_of c.x and es = elements c.x in
  let r = Array.length c.shape and nw = Iarray.length c.pad.windows in
  let n = total c.shape in
  let acc = Array.make n (zero d) and held = Array.make n false in
  let under = Array.make n false in
  let taps = Array.sub xs r nw and outer = Array.sub xs 0 r in
  for t = 0 to total taps - 1 do
    let j = index_of taps (if reversed then total taps - 1 - t else t) in
    let term = Array.make n (zero d) and hit = Array.make n false in
    for k = 0 to total outer - 1 do
      let xi = Array.append (index_of outer k) j in
      match target c xi with
      | None -> ()
      | Some y ->
          term.(y) <- es.(flat xs xi);
          hit.(y) <- true
    done;
    let sums =
      if Array.exists2 ( && ) hit held then added d c.shape acc term else term
    in
    for y = 0 to n - 1 do
      if hit.(y) && held.(y) then begin
        let a = acc.(y) and e = term.(y) and s = sums.(y) in
        if tiny d a || tiny d e || tiny d s then under.(y) <- true;
        acc.(y) <- s
      end
      else if hit.(y) then begin
        acc.(y) <- term.(y);
        held.(y) <- true
      end
    done
  done;
  (acc, under)

(* [y] has the elements [want], bit for bit or both NaN, but where [b]
   flushes, at the elements [under] marks. *)
let agrees_with (b : Support.backend) c y (want, under) =
  let got = elements y in
  if b.underflow = Flush then
    Array.iteri (fun i u -> if u then got.(i) <- want.(i)) under;
  equal (array (same (dtype_of c.x))) want got

(* [y] has the reference's elements, bit for bit or both NaN, but where [b]
   flushes, at the elements whose additions took or gave a tiny element. *)
let agrees b c y = agrees_with b c y (reference c)

(* The axes before the first a window takes. *)
let leading c =
  let r = Array.length c.shape in
  let windowed a =
    Iarray.exists (fun (w : M.window) -> w.axis = a) c.pad.windows
  in
  let rec go a = if a < r && not (windowed a) then go (a + 1) else a in
  go 0

(* Whether every result index along the leading axes receives an operand
   index: the fold then starts and adds its result a slab at a time. *)
let slabs c =
  let p = c.pad in
  let covers a =
    Iarray.get p.lo a = 0
    && Iarray.get p.hi a >= 0
    && (Iarray.get p.interior a = 0 || c.shape.(a) <= 1)
  in
  List.for_all covers (List.init (leading c) Fun.id)

let law_reference (b : Support.backend) = function
  | None -> assume false
  | Some c -> (
      let n = leading c and r = Array.length c.shape in
      let work =
        Iarray.length c.pad.windows > 0 && total c.shape > 0 && total (shape_of c.x) > 0
      in
      let units = total (Array.sub c.shape 0 n) in
      cover "windows" (Iarray.length c.pad.windows > 0);
      cover "a pad of whole bytes"
        (Iarray.length c.pad.windows = 0
        && Iarray.for_all (( = ) 0) c.pad.interior
        && total c.shape > 1
        && total (shape_of c.x) > 0
        &&
        let (D.Any d) = dtype_of c.x in
        D.bits d >= 8);
      cover "interior" (Iarray.exists (fun i -> i > 0) c.pad.interior);
      cover "cropped" (Iarray.exists (fun l -> l < 0) c.pad.lo);
      cover "slabs started per unit" (work && n > 0 && units > 1 && slabs c);
      cover "whole started, a leading axis padded"
        (work && n > 0 && not (slabs c));
      (let (D.Any d) = dtype_of c.x in
       let counts = reached c in
       let float = D.is D.Float d || D.is D.Complex d in
       cover "a float element no tap reaches"
         (work && float && Array.exists (( = ) 0) counts);
       cover "a float element of one tap"
         (work && float && Array.exists (( = ) 1) counts));
      let taps = Array.sub (shape_of c.x) r (Iarray.length c.pad.windows) in
      cover "taps innermost" (work && n < r && total taps > 1);
      (let (D.Any d) = dtype_of c.x in
       cover "a narrow float" (work && is_narrow (D.Any d));
       cover "sub-byte" (work && D.bits d < 8);
       cover "complex" (work && D.is D.Complex d));
      let want = reference c in
      (* The values tell the taps' order apart: some element's sum differs from
         the reversed taps', neither a NaN. *)
      let d = dtype_of c.x in
      let other = fst (reference ~reversed:true c) in
      cover "a float element whose sum the taps' order changes"
        (let (D.Any dt) = d in
         (D.is D.Float dt || D.is D.Complex dt)
         && Array.exists2
              (fun a b -> a <> b && not (nan_of d a || nan_of d b))
              (fst want) other);
      match fold_on b c with
      | None -> failf "%s declined a fold" b.name
      | Some y -> agrees_with b c y want)

(* Each part of a float or complex fold's element is within γ(n - 1) Σ|x| of the
   exact sum of the [n] parts it adds, where they are finite and their
   magnitudes' sum is below half the dtype's largest float, so that no sum
   overflows. Where [b] flushes, each addition reads a subnormal operand as a
   zero and gives a zero for a subnormal result, so each of the n terms and each
   of the n - 1 sums may lose less than the least normal number, 2^-126, and the
   later additions round those losses: the bound adds (1 + γ(n - 1)) (2n - 1)
   2^-126. The exact sum is a TwoSum pair [hi + lo]: [hi]'s errors are exact,
   and [lo] adds two a term, each at most [u'] times the magnitudes' sum [mag],
   [u'] float64's unit roundoff, so its [k]th of [2n] roundings errs by at most
   [k u'² mag], and the pair is within [n (2n + 1) u'² mag], below [2n (n + 2)
   u'² mag], of the exact sum. *)
let law_bound (b : Support.backend) = function
  | None -> assume false
  | Some c -> (
      let (D.Any dt) = dtype_of c.x in
      let single = D.bits dt = 32 || (D.bits dt = 64 && D.is D.Complex dt) in
      let parts = if D.is D.Complex dt then 2 else 1 in
      let w = D.bits dt / 8 / parts in
      let get e p = if single then get32 e (p * w) else get64 e (p * w) in
      let u = if single then 0x1p-24 else 0x1p-53 in
      let top = if single then 0x1.fffffep127 else max_float in
      let xs = shape_of c.x and es = elements c.x in
      let terms = Array.make (total c.shape * parts) [] in
      for k = 0 to total xs - 1 do
        match target c (index_of xs k) with
        | None -> ()
        | Some y ->
            for p = 0 to parts - 1 do
              let at = (y * parts) + p in
              terms.(at) <- get es.(k) p :: terms.(at)
            done
      done;
      let held t =
        List.for_all Float.is_finite t
        && List.fold_left (fun a x -> a +. Float.abs x) 0. t <= top /. 2.
      in
      cover "an element of several finite terms"
        (Array.exists (fun t -> List.length t > 1 && held t) terms);
      cover "complex" (parts = 2);
      match fold_on b c with
      | None -> failf "%s declined a fold" b.name
      | Some y ->
          let ys = elements y in
          let got i = get ys.(i / parts) (i mod parts) in
          let bad = ref [] in
          Array.iteri
            (fun i t ->
              if held t then begin
                (* The exact sum: TwoSum's errors carried in a second part. *)
                let hi = ref 0. and lo = ref 0. in
                List.iter
                  (fun x ->
                    let s = !hi +. x in
                    let z = s -. !hi in
                    lo := !lo +. (!hi -. (s -. z)) +. (x -. z);
                    hi := s)
                  t;
                let exact = !hi +. !lo in
                let mag = List.fold_left (fun a x -> a +. Float.abs x) 0. t in
                let n = Float.of_int (max 0 (List.length t - 1)) in
                let g = n *. u /. (1. -. (n *. u)) in
                let flushed =
                  if single && b.underflow = Flush then
                    ((2. *. n) +. 1.) *. 0x1p-126
                  else 0.
                in
                let slack =
                  Float.of_int (2 * List.length t * (List.length t + 2))
                  *. 0x1p-106 *. mag
                in
                let bound = (g *. mag) +. slack +. ((1. +. g) *. flushed) in
                (* Each part subtracted, so that rounding [exact] adds no
                   error. *)
                let off = got i -. !hi -. !lo in
                if not (Float.abs off <= bound) then
                  bad :=
                    Printf.sprintf
                      "element %d, part %d: %h is %h from %h, past %h"
                      (i / parts) (i mod parts) (got i) off exact bound
                    :: !bad
              end)
            terms;
          equal (list string) []
            (List.filteri (fun i _ -> i < 8) (List.rev !bad)))

(* A map of the padded load of an array of the case's result shape, seen
   through the case's [load] view, loads at each index the element the fold
   would add that index into, or the fill where the load reads padding. *)
let law_load (b : Support.backend) = function
  | None -> assume false
  | Some c -> (
      let module K = (val b.kernels) in
      let (A.Any x) = c.x in
      let dt = A.dtype x and d = dtype_of c.x in
      let rs = Random.State.make [| 7 |] in
      let a =
        Support.on b
          (A.expect dt (operand d c.shape c.load (fun _ -> element d rs)))
      in
      let drawn = Rig.Buffer.of_string (element d rs) in
      let fill = A.get (A.v dt (L.contiguous [||]) drawn) [||] in
      let fill_bits = Nx_kernel.Prog.bits dt fill in
      let xs = shape_of c.x in
      let p = Nx_kernel.Prog.v ~ins:[| d |] [| In 0 |] ~outs:[| 0 |] in
      (* The load beside 100 unused constants: a program long enough that
         nx.cpu prices it with the domain lock released. *)
      let long =
        Nx_kernel.Prog.v ~ins:[| d |]
          (Iarray.init 101 (fun i ->
               if i = 0 then Nx_kernel.Prog.In 0
               else Const (D.Any D.Uint8, "\000")))
          ~outs:[| 0 |]
      in
      let some = total xs > 0 in
      cover "padding read" (some && total c.shape < total xs);
      cover "a windowed load" (some && Iarray.length c.pad.windows > 0);
      cover "sub-byte" (some && D.bits dt < 8);
      cover "a stepped operand"
        (some && c.load.stepped && Support.gapped (A.layout a));
      cover "a reversed operand" (some && c.load.reversed <> None);
      let load = Nx_kernel.Padded { x = a; fill; pad = c.pad } in
      let es = elements (A.Any a) in
      let want =
        Array.init (total xs) (fun k ->
            match target c (index_of xs k) with
            | Some y -> es.(y)
            | None -> fill_bits)
      in
      List.iter
        (fun p ->
          let dst = A.create b.device dt (Iarray.of_array xs) in
          match K.map p ~dsts:[| A.Any dst |] [| load |] with
          | A.Declined -> failf "%s declined a padded map" b.name
          | A.Done -> equal (array string) want (elements (A.Any dst))
          | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r)
        [ p; long ])

(* Folds a library may take past a few windows in registers: four windows
   over four axes, and a window whose step and padding reach 2^30. *)
let test_wide_windows (b : Support.backend) () =
  let fold pad d shape =
    let rs = Random.State.make [| 13 |] in
    let xs = Option.get (loaded pad shape) in
    let x = operand d xs (plain (Array.length xs)) (fun _ -> element d rs) in
    let c = { shape; pad; x; views = []; load = plain (Array.length shape) } in
    match fold_on b c with
    | None -> failf "%s declined a fold" b.name
    | Some y -> agrees b c y
  in
  let w axis ~size ~step = { M.axis; size; step; dilation = 1 } in
  let four =
    {
      Nx_kernel.lo = [| 0; 0; 0; 0 |];
      hi = [| 0; 0; 0; 0 |];
      interior = [| 0; 0; 0; 0 |];
      windows = Iarray.init 4 (fun a -> w a ~size:2 ~step:1);
    }
  in
  fold four (D.Any D.Float32) [| 2; 2; 2; 2 |];
  fold four (D.Any D.Int16) [| 3; 2; 2; 3 |];
  let far =
    {
      Nx_kernel.lo = [| 1 lsl 30 |];
      hi = [| 0 |];
      interior = [| 0 |];
      windows = [| w 0 ~size:1 ~step:(1 lsl 30) |];
    }
  in
  fold far (D.Any D.Float32) [| 1 |]

(* A convolution's input gradient on nx.cpu's threads, at float32, int4 and
   bfloat16, the last two through views of the operand. nx.cpu gives a job a
   thread for each 128 KiB of its cost (job.c), a fold's four bytes of work for
   each byte it adds or reads (assemble.c), so each fold by slabs, whose units
   add slabs of the result alone, int4's of an odd number of elements, which
   share a byte with their neighbours', is worth 2.8 to 4.5 threads. A result
   whose units miss some slab starts whole first, a walk of its bytes, a
   sub-byte dtype's elements, that takes a second thread from 256 KiB: each fold
   cropped to one batch row starts a result of 3 2^17 or more, from an operand
   of that row alone. A worker of one core runs each on one thread, where the
   test checks the sums alone. *)
let test_threads (b : Support.backend) () =
  let w axis = { M.axis; size = 3; step = 1; dilation = 1 } in
  let conv lo0 =
    {
      Nx_kernel.lo = [| lo0; 0; 1; 1 |];
      hi = [| 0; 0; 1; 1 |];
      interior = [| 0; 0; 0; 0 |];
      windows = [| w 2; w 3 |];
    }
  in
  let fold pad d shape v =
    let rs = Random.State.make [| 11 |] in
    let x =
      operand d (Option.get (loaded pad shape)) v (fun _ -> element d rs)
    in
    let c = { shape; pad; x; views = view_names v; load = plain 4 } in
    match fold_on b c with
    | None -> failf "%s declined a fold" b.name
    | Some y -> agrees b c y
  in
  let slabs = conv 0 and cropped shape = conv (1 - shape.(0)) in
  let stepped = { (plain 6) with stepped = true }
  and reversed = { (plain 6) with reversed = Some 3 } in
  fold slabs (D.Any D.Float32) [| 4; 8; 8; 8 |] (plain 6);
  fold slabs (D.Any D.Int4) [| 8; 8; 9; 9 |] stepped;
  fold slabs (D.Any D.Bfloat16) [| 4; 8; 9; 9 |] reversed;
  let whole d shape v = fold (cropped shape) d shape v in
  whole (D.Any D.Float32) [| 12; 16; 24; 24 |] (plain 6);
  whole (D.Any D.Int4) [| 16; 8; 59; 59 |] stepped;
  whole (D.Any D.Bfloat16) [| 13; 16; 32; 32 |] reversed

(* Rows longer than a slot of float32, of a narrow float and of int4: they add
   through their carrier a piece at a time. *)
let test_long_rows (b : Support.backend) () =
  let pad =
    {
      Nx_kernel.lo = [| 0; 1 |];
      hi = [| 0; 1 |];
      interior = [| 0; 0 |];
      windows = [| { M.axis = 1; size = 3; step = 1; dilation = 1 } |];
    }
  in
  let shape = [| 2; 5000 |] in
  List.iter
    (fun d ->
      let rs = Random.State.make [| 13 |] in
      let x =
        operand d
          (Option.get (loaded pad shape))
          (plain 3)
          (fun _ -> element d rs)
      in
      let c = { shape; pad; x; views = []; load = plain 2 } in
      match fold_on b c with
      | None -> failf "%s declined a fold" b.name
      | Some y -> agrees b c y)
    D.[ Any Bfloat16; Any Int4 ]

(* Folds and a map's padded load that do not fit, by a shape, a pad of another
   rank or booleans, which Prog.Add does not take: each with its misfit's class,
   its destination's dtype and shape, and its call by a library, the operand,
   made on the host, on the library's device. *)
let misfits =
  let p =
    { Nx_kernel.lo = [| 1 |]; hi = [| 1 |]; interior = [| 0 |]; windows = [||] }
  in
  let f32 = D.Any D.Float32 in
  let floats n =
    A.Any (A.of_array D.Float32 [| n |] (Array.init n Float.of_int))
  in
  let fold (A.Any x) (b : Support.backend) (g : Misfit.guard) ds =
    let module K = (val b.kernels) in
    let x = g.g (Support.on b x) in
    K.fold p ~dst:(A.expect (A.dtype x) ds.(0)) x
  in
  let map (b : Support.backend) (g : Misfit.guard) ds =
    let module K = (val b.kernels) in
    let id = Nx_kernel.Prog.v ~ins:[| f32 |] [| In 0 |] ~outs:[| 0 |] in
    let x = g.g (Support.on b (A.expect D.Float32 (floats 5))) in
    let pad = { p with lo = [| 1; 0 |] } in
    K.map id ~dsts:(Iarray.of_array ds) [| Padded { x; fill = 0.; pad } |]
  in
  let bools =
    A.Any (A.of_array D.Bool [| 5 |] [| true; false; true; false; true |])
  in
  [
    ( "an operand of another shape",
      "a shape",
      [| (f32, [| 3 |]) |],
      fold (floats 4) );
    ( "a destination of another shape",
      "a shape",
      [| (f32, [| 4 |]) |],
      fold (floats 5) );
    ("a map's pad of another rank", "an attribute", [| (f32, [| 7 |]) |], map);
    ("booleans", "a dtype", [| (D.Any D.Bool, [| 3 |]) |], fold bools);
  ]

(* Not Done, and nothing written outside the destination. *)
let check_misfit (b : Support.backend) (msg, what, dsts, call) =
  Misfit.check ~msg ~what b dsts (call b)

(* A destination over its operand's bytes, one element on: the door refuses the
   fold as Overlapping. *)
let test_overlapping (b : Support.backend) () =
  let module K = (val b.kernels) in
  let all = A.create b.device D.Float32 [| 7 |] in
  let from start count =
    Option.get (A.move (M.Slice [| { M.start; count; step = 1 } |]) all)
  in
  let p =
    { Nx_kernel.lo = [| 1 |]; hi = [| 1 |]; interior = [| 0 |]; windows = [||] }
  in
  equal answer A.Overlapping (K.fold p ~dst:(from 1 4) (from 0 6))

(* The suite *)

(* Cases of [dtypes], printed, or [None] where the pad fits no load. *)
let cases_of ?dtypes ?last () =
  Gen.with_pp
    (fun ppf -> function
      | None -> Format.pp_print_string ppf "no load" | Some c -> pp_case ppf c)
    (case_gen ?dtypes ?last ())

let cases = cases_of ()
let all_cases = cases_of ~dtypes:D.all ()

let float_cases =
  cases_of
    ~dtypes:D.[ Any Float32; Any Float64; Any Complex64; Any Complex128 ]
    ()

(* Guard pages *)

let guard_cases = Gen.pair (cases_of ~last:tails ()) sides

(* The same result with the operand, through its view, and the destination
   against an inaccessible page. *)
let law_guards (b : Support.backend) (c, side) =
  let module K = (val b.kernels) in
  match c with
  | None -> reject ()
  | Some c ->
      let after = side = Support.After in
      cover "windows" (Iarray.length c.pad.windows > 0);
      cover "a pad of whole bytes"
        (Iarray.length c.pad.windows = 0
        && Iarray.for_all (( = ) 0) c.pad.interior
        && total c.shape > 1
        && total (shape_of c.x) > 0
        &&
        let (D.Any d) = dtype_of c.x in
        D.bits d >= 8);
      cover "sub-byte" (let (D.Any dt) = dtype_of c.x in D.bits dt < 8);
      cover "a stepped operand at the page before"
        ((not after) && List.mem "stepped" c.views);
      cover "a reversed operand at the page after"
        (after && List.mem "reversed" c.views);
      let call xs ds =
        let (A.Any x) = xs.(0) in
        K.fold c.pad ~dst:(A.expect (A.dtype x) ds.(0)) x
      in
      equal answer A.Done
        (same_guarded ~any_nan:true side ~call [| c.x |]
           [| (dtype_of c.x, c.shape) |])

(* A fold at a dtype [b] states it computes nothing on declines before any
   write, its destination's bytes as they were; at another it is Done. *)
let test_declines (b : Support.backend) (D.Any dt as d) =
  let module K = (val b.kernels) in
  let p =
    { Nx_kernel.lo = [| 1 |]; hi = [| 0 |]; interior = [| 0 |]; windows = [||] }
  in
  let rs = Random.State.make [| 3 |] in
  let drawn s =
    Support.on b (A.expect dt (operand d s (plain 1) (fun _ -> element d rs)))
  in
  let x = drawn [| 4 |] and dst = drawn [| 3 |] in
  let before = elements (A.Any dst) in
  if b.arith d then equal answer A.Done (K.fold p ~dst x)
  else begin
    equal ~msg:"the answer" answer A.Declined (K.fold p ~dst x);
    equal ~msg:"the destination" (array string) before (elements (A.Any dst))
  end

(* Folds of 1s by windows of 3 taps, one tap the least subnormal, at float32 and
   bfloat16: a library that flushes adds it as a zero into the one element it
   reaches, and the other 15 have the reference's bits. *)
let subnormal_taps =
  let p =
    {
      Nx_kernel.lo = [| 1 |];
      hi = [| 1 |];
      interior = [| 0 |];
      windows = [| { M.axis = 0; size = 3; step = 1; dilation = 1 } |];
    }
  in
  let fold d ~one ~least =
    let bits v =
      String.init (String.length one) (fun j ->
          Char.chr ((v lsr (8 * j)) land 0xFF))
    in
    let at = (5 * 3) + 1 in
    let x =
      operand d [| 16; 3 |] (plain 2) (fun k ->
          if k = at then bits least else one)
    in
    { shape = [| 16 |]; pad = p; x; views = []; load = plain 1 }
  in
  [
    fold (D.Any D.Float32) ~one:(f32 1.) ~least:1;
    fold (D.Any D.Bfloat16) ~one:"\x80\x3f" ~least:1;
  ]

(* Two float32 terms whose exact sum, 2^-125 - 1.5 2^-126 = 2^-127, is
   subnormal: a backend that flushes gives +0. The two windows of 2 taps
   over one element padded on both sides read it at taps 1 and 0. *)
let subnormal_sum =
  let p =
    {
      Nx_kernel.lo = [| 1 |];
      hi = [| 1 |];
      interior = [| 0 |];
      windows = [| { M.axis = 0; size = 2; step = 1; dilation = 1 } |];
    }
  in
  let terms = [| 0.; 0x1p-125; -0x1.8p-126; 0. |] in
  let x =
    operand (D.Any D.Float32) [| 2; 2 |] (plain 2) (fun k -> f32 terms.(k))
  in
  { shape = [| 1 |]; pad = p; x; views = []; load = plain 1 }

(* Folds of 3 slabs, each 4 elements windowed by [w] along the last axis: at
   complex64, of windows of 2 taps; at bfloat16, of windows of 1 tap 2 apart,
   which leave elements no tap reaches; and at int4. The draws reach a narrow
   float, complex64 and slabs started per unit one case in 9 to 11 where float64
   and complex128 are declined. At float32, 3 rows of 8 elements under windows
   of 4 taps over [mixed] numbers, whose sums the taps' order changes, which the
   draws reach about one case in 110. *)
let fold_examples =
  let slabs d (w : M.window) =
    let zeros = Iarray.of_array [| 0; 0 |] in
    let pad =
      { Nx_kernel.lo = zeros; hi = zeros; interior = zeros; windows = [| w |] }
    in
    let shape = [| 3; 4 |] in
    let xs = Option.get (loaded pad shape) in
    let rs = Random.State.make [| 3 |] in
    let x = operand d xs (plain (Array.length xs)) (fun _ -> element d rs) in
    { shape; pad; x; views = []; load = plain 2 }
  in
  let w size step = { M.axis = 1; size; step; dilation = 1 } in
  let ordered =
    let zeros = Iarray.of_array [| 0; 0 |] in
    let pad =
      {
        Nx_kernel.lo = zeros;
        hi = zeros;
        interior = zeros;
        windows = [| w 4 1 |];
      }
    in
    let shape = [| 3; 8 |] in
    let xs = Option.get (loaded pad shape) in
    let rs = Random.State.make [| 5 |] in
    let x =
      operand (D.Any D.Float32) xs
        (plain (Array.length xs))
        (fun _ -> f32 (mixed rs))
    in
    { shape; pad; x; views = []; load = plain 2 }
  in
  [
    slabs (D.Any D.Complex64) (w 2 1);
    slabs (D.Any D.Bfloat16) (w 1 2);
    slabs (D.Any D.Int4) (w 2 1);
    ordered;
  ]

let suite (d : Support.declared) =
  let around f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  (* A case of a dtype the library states it computes nothing on is no case of
     the law. *)
  let computed law c =
    let b = Support.opened d in
    (match c with
    | Some c when not (b.arith (dtype_of c.x)) -> assume false
    | _ -> ());
    b.around (fun () -> law b c)
  in
  let some = List.map Option.some in
  group d.label
    [
      group "fold"
        [
          group "sums each element's taps in order, +0 where none"
            [
              prop
                ~examples:(some (fold_examples @ subnormal_taps))
                "at every dtype, through views" cases (computed law_reference);
              test "a convolution's input gradient on the job's threads"
                (fun () -> around test_threads ());
              test "four windows, and a step and padding past 2^30" (fun () ->
                  around test_wide_windows ());
              test "rows longer than a slot, at bfloat16 and int4" (fun () ->
                  around test_long_rows ());
            ];
          prop
            "each part of a float fold is within γ(n - 1) Σ|x| of the exact \
             sum, and the flushes' losses where it flushes"
            ~examples:[ Some subnormal_sum ] float_cases (computed law_bound);
          prop
            "a Padded load reads at each index the element the fold adds it \
             into, or the fill"
            all_cases (around law_load);
          Windtrap.cases
            ~id:(fun (D.Any dt) -> D.name dt)
            "declines exactly the dtypes it computes nothing on, before any \
             write"
            folded (around test_declines);
          Windtrap.cases
            ~id:(fun (msg, _, _, _) -> msg)
            "a call of shapes, pads or dtypes that do not fit is not Done and \
             writes nothing outside its destination"
            misfits (around check_misfit);
          test
            "a destination over its operand's bytes is refused as Overlapping"
            (fun () -> around test_overlapping ());
        ];
    ]

(* Guard pages lie on host memory. *)
let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop ~count:300
        "reads and writes no byte past the operand and dst, whatever pages lie \
         beside them"
        guard_cases (run (law_guards b));
    ]

(* rig.nv *)

(* rig.nv loads the cubin CUDA loads, so a fold through it has CUDA's
   bits: what it can get wrong, its launches, memory and order, shows in
   them. Each case runs through CUDA, then through rig.nv. A NaN agrees
   with any NaN: nx_cuda.mli leaves a NaN result's sign and payload
   unspecified. *)
let nv_laws (first : Support.declared) (nv : Support.declared) =
  let agree call =
    Elements.agree (Support.opened first) (Support.opened nv) call
  in
  let some = function Some y -> [| y |] | None -> [||] in
  group nv.label
    [
      group "bits are cuda's, any NaN for a NaN"
        [
          prop "of folds at every dtype they take" cases (function
            | None -> ()
            | Some c -> agree (fun b -> some (fold_on b c)));
        ];
    ]

let () =
  let nv =
    match (Support.declared_cuda, Support.declared_nv) with
    | Some first, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.fold"
       (List.map suite
          (Support.declare Support.cpu
          :: List.filter_map Fun.id
               [
                 Support.declared_cuda;
                 Support.declared_amd;
                 Support.declared_metal;
               ])
       @ [ cpu Support.cpu ]
       @ nv))
