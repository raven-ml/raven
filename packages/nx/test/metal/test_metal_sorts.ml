(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.metal's sorts on the machine's Metal GPU 0, skipping without one: each
   path nx_metal.mli names (Sorts and numbering) gives nx.cpu's bits and takes
   the launches the section states, under the hidden bounds that reach 64-bit
   positions and carries, many parts, split launches and buffers of a call's own
   with small arrays. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Nx_metal.Kernels
module Plan = Nx_metal.Plan
module S = Nx_metal_support
module Support = Nx_kernels_support

let pp_any ppf (D.Any dt) = D.pp ppf dt

let answer =
  Testable.make ~pp:Nx_array_support.pp_answer ~equal:(fun a b -> a = b)

let backend () =
  match Support.metal () with
  | Some b -> b
  | None -> skip ~reason:"no Metal GPU nx.metal computes on" ()

(* Inputs

   An operand's elements: each at random, every one the same, or each one of
   four, so that ties are common and a key byte may be shared; or at random but
   in the first half of the operand's bytes, whose elements keep only their low
   byte, so that slices there share their other key bytes and take fewer passes
   than the slices after them. A sub-byte element's byte holds equal codes. *)
type draw = Random | Equal | Few | Mixed

let pp_draw ppf d =
  Format.pp_print_string ppf
    (match d with
    | Random -> "random"
    | Equal -> "equal"
    | Few -> "four values"
    | Mixed -> "low bytes in the first half")

(* [n] bytes of elements of [eb] bytes drawn by [d]. *)
let draw_bytes st d ~eb ~sub n =
  let element () =
    if sub then String.make 1 (Char.chr (Random.State.int st 16 * 17))
    else String.init eb (fun _ -> Char.chr (Random.State.int st 256))
  in
  let b = Bytes.create n in
  (match d with
  | Random | Mixed ->
      for i = 0 to n - 1 do
        let low = d = Random || 2 * i >= n || i mod eb = 0 in
        Bytes.set b i
          (if low then Char.chr (Random.State.int st 256) else '\000')
      done
  | Equal | Few ->
      let pool =
        Array.init (if d = Equal then 1 else 4) (fun _ -> element ())
      in
      let w = String.length pool.(0) in
      let i = ref 0 in
      while !i < n do
        let e = pool.(Random.State.int st (Array.length pool)) in
        Bytes.blit_string e 0 b !i (Int.min w (n - !i));
        i := !i + w
      done);
  Bytes.unsafe_to_string b

(* A fresh array of [dt] over [layout]'s span, its bytes drawn by [d]. *)
let drawn st (type v s) (dt : (v, s) D.t) d layout =
  let _, hi = L.span layout in
  let bits = D.bits dt in
  let bytes = ((hi * bits) + 7) / 8 in
  let base = A.create (S.device ()) D.Uint8 (Iarray.of_array [| bytes |]) in
  let b = A.buffer base in
  let s = draw_bytes st d ~eb:(Int.max 1 (bits / 8)) ~sub:(bits < 8) bytes in
  Rig.Buffer.blit_from_string s 0 b 0 bytes;
  A.v dt layout b

(* Sorts *)

type path = Block | Select | Radix

type case = {
  dt : D.any;
  slices : int array;  (** The other axes' extents. *)
  n : int;
  outer : bool;  (** Along the first axis, the others after it. *)
  steps : int;  (** Of the slices' axes, which no merge shortens past 1. *)
  keep : int;
  descending : bool;
  draw : draw;
  parts : int;
  wide : int;
  budget : float option;
  fresh : bool;
  plan : Nx_metal.Sort.outer;  (** A strided slice's plan. *)
  positions : int;  (** The least bytes of a select's positions. *)
  seed : int;
}

let pp_case ppf c =
  Format.fprintf ppf
    "sort%s along %s of %a, %d slices of %d keeping %d%s, %a; parts %d, wide \
     %d, positions of %d bytes or more, %s%s%s"
    (if c.descending then " descending" else "")
    (if c.outer then "the first axis" else "the last axis")
    pp_any c.dt
    (Array.fold_left ( * ) 1 c.slices)
    c.n c.keep
    (if c.steps > 1 then Printf.sprintf ", slices %d apart" c.steps else "")
    pp_draw c.draw c.parts c.wide c.positions
    (match c.plan with
    | Nx_metal.Sort.Estimate -> "plan by estimate"
    | Strided -> "strided"
    | Transposed -> "transposed")
    (match c.budget with
    | None -> ""
    | Some s -> Printf.sprintf ", launches of %g s" s)
    (if c.fresh then ", buffers of its own" else "")

let shape c =
  if c.outer then Array.append [| c.n |] c.slices
  else Array.append c.slices [| c.n |]

let axis c = if c.outer then 0 else Array.length c.slices

(* x's layout: C order, but each slice axis outside another axis [steps - 1]
   elements further apart, so that no two merge. *)
let x_layout c =
  let s = shape c in
  let r = Array.length s in
  let strides = Array.make r 1 in
  for j = r - 2 downto 0 do
    let gap = if j = axis c then 0 else c.steps - 1 in
    strides.(j) <- (strides.(j + 1) * s.(j + 1)) + gap
  done;
  L.v ~strides:(Iarray.of_array strides) (Iarray.of_array s)

(* nx_metal.mli's block length: 4096 elements, 2048 of 16 bytes. *)
let cap (D.Any dt) = if D.bits dt = 128 then 2048 else 4096

let path c =
  if c.n <= cap c.dt then Block
  else if c.keep <= cap c.dt / 4 then Select
  else Radix

let key_bytes (D.Any dt) = Int.max 1 (D.bits dt / 8)

(* The bytes of [n - 1]. *)
let position_bytes n =
  let rec go b = if b < 8 && (n - 1) lsr (8 * b) <> 0 then go (b + 1) else b in
  go 1

let with_bounds c f =
  let restore () =
    Nx_metal.bound_sorts ~parts:256 ~wide:32;
    Nx_metal.Sort.bound_outer Estimate;
    Nx_metal.Sort.bound_positions 1;
    Plan.bound_budget Plan.budget;
    let mib = 1024 * 1024 in
    Nx_metal.bound_fresh ~workspace:(64 * mib) ~tickets:(4 * mib)
  in
  Fun.protect ~finally:restore (fun () ->
      Nx_metal.bound_sorts ~parts:c.parts ~wide:c.wide;
      Nx_metal.Sort.bound_outer c.plan;
      Nx_metal.Sort.bound_positions c.positions;
      Option.iter Plan.bound_budget c.budget;
      if c.fresh then Nx_metal.bound_fresh ~workspace:0 ~tickets:0;
      f ())

let kernel_count prefix launches =
  List.length
    (List.filter
       (fun (l : Plan.traced) -> String.starts_with ~prefix (K.name l.kernel))
       launches)

(* The parts of nx.metal's plan of the sort of [x] into [values] and
   [positions], under the bounds in force: those its door submits. *)
let planned_parts c ~values ~positions x =
  Plan.unseen (fun () ->
      match
        Nx_metal.Sort.sort
          ~holds:(Nx_metal.holds (S.device ()))
          ~axis:(axis c) ~descending:c.descending ~values ~positions x
      with
      | Ok (Some p) -> Array.length p.parts
      | Ok None | Error _ -> 0)

(* The sort of [c] on the GPU and on nx.cpu: values and positions of each, and
   the GPU's launches; [None] where nx.metal declines and [declinable]. [parts]
   gets the parts of the GPU call's plan. *)
let sorts ?(declinable = false) ?parts c =
  let (D.Any dt) = c.dt in
  let st = Random.State.make [| c.seed |] in
  let x = drawn st dt c.draw (x_layout c) in
  let y = shape c in
  y.(axis c) <- c.keep;
  let y = Iarray.of_array y in
  let on d k =
    let values = A.create d dt y and positions = A.create d D.Int64 y in
    let r = k ~values ~positions in
    (r, A.Any values, A.Any positions)
  in
  let (got, gv, gp), launches =
    Plan.traced (fun () ->
        with_bounds c (fun () ->
            on (S.device ()) (fun ~values ~positions ->
                Option.iter
                  (fun r -> r := planned_parts c ~values ~positions x)
                  parts;
                Nx_metal.sort ~axis:(axis c) ~descending:c.descending ~values
                  ~positions x)))
  in
  let xh = Support.host x in
  let module C = (val Support.cpu.kernels) in
  let want, wv, wp =
    on Rig.host (fun ~values ~positions ->
        C.sort ~axis:(axis c) ~descending:c.descending ~values ~positions xh)
  in
  equal ~msg:"nx.cpu's answer" answer A.Done want;
  match got with
  | A.Declined when declinable -> None
  | A.Done -> Some ((gv, gp), (wv, wp), launches)
  | r -> failf "nx.metal answered %a" Nx_array_support.pp_answer r

let agree ((gv, gp), (wv, wp)) =
  equal ~msg:"positions" (array string) (Support.elements wp)
    (Support.elements gp);
  equal ~msg:"values" (array string) (Support.elements wv) (Support.elements gv)

(* Whether x's slices step by more than one element, of a byte or more, which
   nx.metal may copy contiguous before it sorts them. *)
let strided c =
  let (D.Any dt) = c.dt in
  c.n > 1 && D.bits dt >= 8 && abs (L.stride (x_layout c) (axis c)) <> 1

(* nx_metal.mli's launches of each path, unsplit, after a copy's where x is
   copied: one block; [k + p + 3] to select, [p] the positions' bytes; [2 + k]
   or [2 + 2k] by radix, the [k] counts where a slice has several parts. *)
let check_launches c launches =
  let k = key_bytes c.dt in
  let copies = kernel_count "copy" launches in
  let total = List.length launches - copies in
  cover "a strided slice copied" (copies > 0);
  cover "a strided slice read where it lies" (strided c && copies = 0);
  (match c.plan with
  | Nx_metal.Sort.Estimate -> ()
  | Strided -> equal ~msg:"copies of strided slices" int 0 copies
  | Transposed -> equal ~msg:"slices copied" bool (strided c) (copies > 0));
  let copied = List.filteri (fun i _ -> i < copies) launches in
  equal ~msg:"copies first" int copies (kernel_count "copy" copied);
  match path c with
  | Block -> equal ~msg:"block launches" int 1 total
  | Select ->
      let p = Int.max c.positions (position_bytes c.n) in
      equal ~msg:"select launches" int (k + p + 3) total
  | Radix ->
      let counts = kernel_count "sort_count" launches in
      cover "several parts" (counts > 0);
      cover "one part" (counts = 0);
      equal ~msg:"radix counts" bool true (counts = 0 || counts = k);
      equal ~msg:"radix launches" int (2 + k + counts) total

let law_sorts c =
  let p = path c in
  cover "block" (p = Block);
  cover "select" (p = Select);
  cover "radix" (p = Radix);
  cover "slices of a cap" (c.n = cap c.dt);
  cover "slices of a cap and one" (c.n = cap c.dt + 1);
  cover "several slices a block group"
    (p = Block && c.n > 1
    && 2 * c.n <= cap c.dt
    && Array.fold_left ( * ) 1 c.slices > 1);
  cover "one element a slice" (c.n = 1);
  cover "keeps a quarter of a cap" (c.keep = cap c.dt / 4);
  cover "keeps a quarter of a cap and one" (c.keep = (cap c.dt / 4) + 1);
  cover "along the first axis" c.outer;
  cover "along the first axis, copied"
    (c.outer && c.plan = Transposed && strided c);
  cover "along the first axis, where it lies" (c.outer && c.plan = Strided);
  cover "8-byte positions" (p = Select && c.positions = 8);
  cover "slices apart" (c.steps > 1);
  cover "complex128" (c.dt = D.Any D.Complex128);
  cover "64-bit keys by radix"
    (p = Radix
    &&
    let (D.Any dt) = c.dt in
    D.bits dt = 64);
  cover "sub-byte by radix"
    (p = Radix
    &&
    let (D.Any dt) = c.dt in
    D.bits dt < 8);
  cover "64-bit positions and carries" (p <> Block && c.n >= 1 lsl c.wide);
  cover "every key equal" (c.draw = Equal && p <> Block);
  cover "four keys" (c.draw = Few && p <> Block);
  match sorts c with
  | None -> ()
  | Some (got, want, launches) ->
      agree (got, want);
      if c.budget = None && not c.fresh then check_launches c launches

(* Every dtype, at slice lengths about each path's edges. *)
let case_gen =
  let open Gen in
  (* Every dtype, and more often those whose covers are rarest among them:
     16-byte keys, sub-byte elements and 8-byte keys. *)
  let bits b = List.filter (fun (D.Any d) -> b (D.bits d)) D.all in
  let* dt =
    frequency
      [
        (4, of_list ~pp:pp_any D.all);
        (1, constant (D.Any D.Complex128));
        (1, of_list ~pp:pp_any (bits (fun b -> b < 8)));
        (1, of_list ~pp:pp_any (bits (fun b -> b = 64)));
      ]
  in
  let cap = cap dt in
  let* p = int_range 0 2 in
  let* n =
    match p with
    | 0 ->
        frequency
          [
            (1, constant 1);
            (3, int_range 2 64);
            (2, int_range 65 (cap - 2));
            (2, of_list ~pp:Format.pp_print_int [ cap - 1; cap ]);
          ]
    | _ ->
        frequency
          [ (1, constant (cap + 1)); (3, int_range (cap + 2) (3 * cap)) ]
  in
  let* keep =
    match p with
    | 1 -> frequency [ (2, constant (cap / 4)); (3, int_range 1 (cap / 4)) ]
    | 2 ->
        frequency
          [
            (2, constant ((cap / 4) + 1));
            (1, constant n);
            (2, int_range ((cap / 4) + 1) n);
          ]
    | _ -> frequency [ (3, constant n); (1, int_range 1 n) ]
  in
  let* slices =
    if n <= 64 then array ~size:(int_range 0 3) (int_range 1 5)
    else if n <= cap then array ~size:(int_range 0 2) (int_range 1 4)
    else
      frequency
        [ (2, constant [||]); (1, map (fun k -> [| k |]) (int_range 2 3)) ]
  in
  let* outer = frequency [ (3, constant false); (1, constant true) ] in
  let* steps = frequency [ (4, constant 1); (1, int_range 2 3) ] in
  let* descending = bool in
  let* draw =
    frequency [ (2, constant Random); (1, constant Equal); (2, constant Few) ]
  in
  let* parts = frequency [ (2, constant 256); (1, int_range 1 4) ] in
  let* wide =
    frequency
      [
        (2, constant 32); (1, int_range 10 (Int.max 10 (position_bytes n * 8)));
      ]
  in
  let* plan =
    frequency
      [
        (2, constant Nx_metal.Sort.Estimate);
        (1, constant Nx_metal.Sort.Strided);
        (2, constant Nx_metal.Sort.Transposed);
      ]
  in
  let* positions =
    frequency [ (3, constant 1); (2, constant 8); (1, int_range 2 7) ]
  in
  let+ seed = nat in
  {
    dt;
    slices;
    n;
    outer;
    steps;
    keep;
    descending;
    draw;
    parts;
    wide;
    budget = None;
    fresh = false;
    plan;
    positions;
    seed;
  }

let cases = Gen.with_pp pp_case case_gen

(* A case of [n] elements a slice, keeping all, ascending, of random float32
   keys unless stated. *)
let example ?(dt = D.Any D.Float32) ?(slices = [||]) ?(outer = false)
    ?(steps = 1) ?keep ?(descending = false) ?(draw = Random) ?(parts = 256)
    ?(wide = 32) ?(plan = Nx_metal.Sort.Estimate) ?(positions = 1) n =
  let keep = Option.value keep ~default:n in
  {
    dt;
    slices;
    n;
    outer;
    steps;
    keep;
    descending;
    draw;
    parts;
    wide;
    budget = None;
    fresh = false;
    plan;
    positions;
    seed = n + Array.length slices;
  }

(* Every cover of law_sorts, so that each run reaches each of them. *)
let examples =
  let f32 = D.Any D.Float32 in
  let cap = cap f32 in
  [
    example ~slices:[| 3; 2 |] 1;
    example ~slices:[| 4 |] ~steps:2 100;
    example ~slices:[| 2 |] cap;
    example ~dt:(D.Any D.Complex128) ~slices:[| 2 |] ~descending:true 2048;
    example ~slices:[| 32 |] ~outer:true ~plan:Transposed 300;
    example ~slices:[| 32 |] ~outer:true ~plan:Transposed (cap + 1);
    example ~slices:[| 32 |] ~outer:true ~plan:Strided 300;
    example ~keep:(cap / 4) ~positions:8 ~draw:Equal (cap + 1);
    example ~keep:((cap / 4) + 1) ~draw:Few (3 * cap);
    example ~parts:1 (cap + 1);
    example ~slices:[| 2 |] (3 * cap);
    example ~dt:(D.Any D.Int64) ~wide:12 (cap + 1);
    example ~dt:(D.Any D.Uint4) ~slices:[| 2 |] (2 * cap);
  ]

(* Splits: slices whose kernels' launches split under a lowered budget, each
   launch a range of their groups. Block: many slices, 1 ms. Select: slices of 2
   or 3 chunks, 350 to 700 us, which the candidates' block sort, about 300 us a
   slice at 2 position bytes, allows. Radix: 1 to 4 slices of 8 to 25 chunks and
   many parts, 150 to 400 us, which a wave of sort_hist's 7 chunks, about 130 us
   at 4-byte keys, allows, and whose chunks pass the 7 to 77 a launch then
   holds; or many slices of one chunk, 500 us, which a slice's sort_scatter of
   one part, about 380 us with 8-byte positions, allows. Once in three, buffers
   of the call's own. *)
let split_gen =
  let open Gen in
  let* dt =
    of_list ~pp:pp_any [ D.Any D.Float32; D.Any D.Int16; D.Any D.Uint8 ]
  in
  let* p = int_range 0 3 in
  let* n =
    match p with
    | 0 -> int_range 1000 4096
    | 1 -> int_range 16385 40000
    | 2 -> int_range 120000 400000
    | _ -> int_range 4097 9000
  in
  let* count =
    match p with
    | 0 -> int_range 200 400
    | 2 -> frequency [ (1, constant 1); (1, int_range 2 4) ]
    | _ -> int_range 32 64
  in
  let* keep =
    match p with
    | 0 -> frequency [ (2, constant n); (1, int_range 1 n) ]
    | 1 -> int_range 1 1024
    | _ -> frequency [ (2, constant n); (1, int_range 1025 n) ]
  in
  let* budget =
    let log lo hi =
      map (fun u -> lo *. ((hi /. lo) ** u)) (float_range 0. 1.)
    in
    match p with
    | 0 -> constant 1e-3
    | 1 -> log 3.5e-4 7e-4
    | 2 -> log 1.5e-4 4e-4
    | _ -> constant 5e-4
  in
  let* parts = frequency [ (2, constant 256); (1, int_range 1 3) ] in
  let* wide = frequency [ (3, constant 32); (1, constant 12) ] in
  let* fresh = frequency [ (2, constant false); (1, constant true) ] in
  let* draw = frequency [ (3, constant Random); (1, constant Few) ] in
  let+ seed = nat in
  {
    dt;
    slices = [| count |];
    n;
    outer = false;
    steps = 1;
    keep;
    descending = false;
    draw;
    parts;
    wide;
    budget = Some budget;
    fresh;
    plan = Nx_metal.Sort.Estimate;
    positions = 1;
    seed;
  }

let splits = Gen.with_pp pp_case split_gen
let word (l : Plan.traced) i = Int64.to_int (Bytes.get_int64_le l.words (8 * i))

(* Whether a launch of [prefix]'s kernels starts or ends within a slice's units:
   a slice's ticket then counts its units across launches. *)
let across prefix launches =
  List.exists
    (fun (l : Plan.traced) ->
      String.starts_with ~prefix (K.name l.kernel)
      &&
      let units = word l K.sort_units in
      let first = word l K.sort_first and stop = word l K.sort_end in
      units > 1 && (first mod units <> 0 || stop mod units <> 0))
    launches

(* A radix sort's budget of a few slices can fall below what one group's work
   takes; such a call is declined and says nothing. The others' budgets hold a
   group's work. [parted]: under lowered part bounds, the call runs in several
   parts with the same bits (plan.mli). *)
let law_splits ?(parted = false) c =
  let p = path c in
  cover "block" (p = Block);
  cover "select" (p = Select);
  cover "radix" (p = Radix);
  cover "buffers of its own" c.fresh;
  let declinable = p = Radix && c.slices.(0) <= 4 in
  let parts = ref 0 in
  match sorts ~declinable ~parts c with
  | None -> assume false
  | Some (got, want, launches) ->
      if parted then cover "a call of several parts" (!parts > 1);
      agree (got, want);
      let k = key_bytes c.dt in
      let split =
        match p with
        | Block -> kernel_count "sort_block" launches > 1
        | Select -> kernel_count "sort_round" launches > k + position_bytes c.n
        | Radix ->
            kernel_count "sort_scatter" launches > k
            || kernel_count "sort_hist" launches > 1
      in
      cover "a split block" (p = Block && split);
      cover "a split select" (p = Select && split);
      cover "a split radix" (p = Radix && split);
      cover "a split radix of one slice" (p = Radix && split && c.slices.(0) = 1);
      cover "sort_hist's units of a slice across launches"
        (across "sort_hist" launches);
      cover "sort_count's units of a slice across launches"
        (across "sort_count" launches);
      cover "sort_round's units of a slice across launches"
        (across "sort_round" launches)

(* Radix sorts of 2 to 4 slices, the first of which shares every key byte but
   the lowest and the last of which shares none, so that their passes differ,
   keeping all; at 32-bit keys the positions destination holds a pass's
   buffer. *)
let masks_gen =
  let open Gen in
  let* dt =
    of_list ~pp:pp_any
      [
        D.Any D.Int32;
        D.Any D.Uint32;
        D.Any D.Float32;
        D.Any D.Int16;
        D.Any D.Int64;
        D.Any D.Complex64;
      ]
  in
  let* count = int_range 2 4 in
  let* n = int_range (cap dt + 1) 20000 in
  let* parts = frequency [ (2, constant 256); (1, int_range 1 3) ] in
  let* wide = frequency [ (3, constant 32); (1, constant 12) ] in
  let* descending = bool in
  let+ seed = nat in
  {
    dt;
    slices = [| count |];
    n;
    outer = false;
    steps = 1;
    keep = n;
    descending;
    draw = Mixed;
    parts;
    wide;
    budget = None;
    fresh = false;
    plan = Nx_metal.Sort.Estimate;
    positions = 1;
    seed;
  }

let masks = Gen.with_pp pp_case masks_gen

let law_masks c =
  let (D.Any dt) = c.dt in
  cover "slices of different passes, 2-byte keys" (D.bits dt = 16);
  cover "slices of different passes, 4-byte keys" (D.bits dt = 32);
  cover "slices of different passes, 8-byte keys" (D.bits dt = 64);
  cover "4-byte keys with 32-bit positions"
    (D.bits dt = 32 && c.n < 1 lsl c.wide);
  match sorts c with
  | None -> fail "a radix sort declined"
  | Some (got, want, _) -> agree (got, want)

(* nx_metal.mli: more than 8 kept axes, which a launch's words don't hold, give
   the same bits. *)
let test_deep () =
  ignore (backend ());
  List.iter
    (fun n ->
      let c =
        {
          dt = D.Any D.Int16;
          slices = Array.make 9 2;
          n;
          outer = false;
          steps = 3;
          keep = n;
          descending = true;
          draw = Few;
          parts = 256;
          wide = 32;
          budget = None;
          fresh = false;
          plan = Nx_metal.Sort.Estimate;
          positions = 1;
          seed = n;
        }
      in
      match sorts c with
      | None -> fail "a sort of 16-bit elements declined"
      | Some (got, want, _) -> agree (got, want))
    [ 7; 300 ]

let test_bounds () =
  raises (Invalid_argument "Nx_metal.bound_sorts: parts") (fun () ->
      Nx_metal.bound_sorts ~parts:0 ~wide:32);
  raises (Invalid_argument "Nx_metal.bound_sorts: parts") (fun () ->
      Nx_metal.bound_sorts ~parts:257 ~wide:32);
  raises (Invalid_argument "Nx_metal.bound_sorts: wide") (fun () ->
      Nx_metal.bound_sorts ~parts:256 ~wide:0);
  raises (Invalid_argument "Nx_metal.bound_sorts: wide") (fun () ->
      Nx_metal.bound_sorts ~parts:256 ~wide:33);
  raises (Invalid_argument "Nx_metal.Sort.bound_positions") (fun () ->
      Nx_metal.Sort.bound_positions 0);
  raises (Invalid_argument "Nx_metal.Sort.bound_positions") (fun () ->
      Nx_metal.Sort.bound_positions 9)

(* Sub-byte values at any bit of their words *)

(* Where x lies: in a buffer of its own, or in the bytes of the values'
   allocation just before or just after the values'. *)
type beside = Apart | Before | After

type mid = {
  c : case;
  first : int;  (** The values' first byte in their word. *)
  lead : int;  (** The values' elements before their first. *)
  tail : int;  (** The values' bytes past the byte of their last. *)
  x_at : beside;
}

let pp_mid ppf m =
  Format.fprintf ppf "%a; values from byte %d, %d elements in, %d bytes over%s"
    pp_case m.c m.first m.lead m.tail
    (match m.x_at with
    | Apart -> ""
    | Before -> ", x before them"
    | After -> ", x after them")

(* Sub-byte sorts by each path into values that start and end anywhere in their
   32-bit words, with x apart or in the bytes beside the values. *)
let mid_gen =
  let open Gen in
  let* dt =
    of_list ~pp:pp_any (List.filter (fun (D.Any d) -> D.bits d < 8) D.all)
  in
  let cap = cap dt in
  let* p = int_range 0 2 in
  let* n =
    match p with
    | 0 -> frequency [ (1, constant 1); (3, int_range 2 cap) ]
    | _ -> int_range (cap + 1) (2 * cap)
  in
  let* keep =
    match p with
    | 0 -> int_range 1 n
    | 1 -> int_range 1 (cap / 4)
    | _ -> int_range ((cap / 4) + 1) n
  in
  let (D.Any d) = dt in
  let* slices = array ~size:(int_range 0 2) (int_range 1 3) in
  let* descending = bool in
  let* draw =
    frequency [ (2, constant Random); (1, constant Equal); (2, constant Few) ]
  in
  let* first = int_range 0 3 in
  let* lead = int_range 0 ((32 / D.bits d) - 1) in
  let* tail = int_range 0 3 in
  let* x_at =
    frequency [ (1, constant Apart); (1, constant Before); (1, constant After) ]
  in
  let+ seed = nat in
  let c =
    {
      dt;
      slices;
      n;
      outer = false;
      steps = 1;
      keep;
      descending;
      draw;
      parts = 256;
      wide = 32;
      budget = None;
      fresh = false;
      plan = Nx_metal.Sort.Estimate;
      positions = 1;
      seed;
    }
  in
  { c; first; lead; tail; x_at }

let mids = Gen.with_pp pp_mid mid_gen

(* nx_metal.mli: a sort into values of any offset gives nx.cpu's values and
   positions and leaves every other bit of their allocation, x beside them in a
   word they share included: x shares no byte with them. *)
let law_mid m =
  let c = m.c in
  let (D.Any dt) = c.dt in
  let bits = D.bits dt in
  let st = Random.State.make [| c.seed |] in
  let y = shape c in
  y.(axis c) <- c.keep;
  let kept = Array.fold_left ( * ) 1 y in
  let y = Iarray.of_array y in
  let vbytes = ((((m.lead + kept) * bits) + 7) / 8) + m.tail in
  let fresh = drawn st dt c.draw (x_layout c) in
  let _, hi = L.span (x_layout c) in
  let xbytes = ((hi * bits) + 7) / 8 in
  (* The values' first byte, after x's bytes where x lies before them. *)
  let vfirst =
    match m.x_at with
    | Before -> ((xbytes + 3) / 4 * 4) + m.first
    | Apart | After -> m.first
  in
  let size = vfirst + vbytes + if m.x_at = After then xbytes else 0 in
  let all = Rig.Buffer.create (S.device ()) (size + 4) in
  Rig.Buffer.fill all '\xA5';
  let beside first =
    let b = Bytes.create xbytes in
    Rig.Buffer.blit_to_bytes (A.buffer fresh) 0 b 0 xbytes;
    let view = Rig.Buffer.view all ~first ~length:xbytes in
    Rig.Buffer.blit_from_string (Bytes.unsafe_to_string b) 0 view 0 xbytes;
    A.v dt (x_layout c) view
  in
  let x, shares =
    match m.x_at with
    | Apart -> (fresh, false)
    | Before -> (beside (vfirst - xbytes), vfirst mod 4 <> 0)
    | After -> (beside (vfirst + vbytes), (vfirst + vbytes) mod 4 <> 0)
  in
  let read () =
    let b = Bytes.create (size + 4) in
    Rig.Buffer.blit_to_bytes all 0 b 0 (size + 4);
    b
  in
  let before = read () in
  let lv = L.v ~offset:m.lead ~strides:(L.strides (L.contiguous y)) y in
  let values = A.v dt lv (Rig.Buffer.view all ~first:vfirst ~length:vbytes) in
  let positions = A.create (S.device ()) D.Int64 y in
  let p = path c in
  cover "a block sort" (p = Block);
  cover "a select" (p = Select);
  cover "a radix sort" (p = Radix);
  let lo = (8 * vfirst) + (m.lead * bits) in
  let hi = lo + (kept * bits) in
  cover "values from a word's middle" (lo mod 32 <> 0);
  cover "values to a word's middle" (hi mod 32 <> 0);
  cover "values from a byte's middle" (m.lead * bits mod 8 <> 0);
  cover "x in the values' first word" (shares && m.x_at = Before);
  cover "x in the values' last word" (shares && m.x_at = After);
  cover "x beside the values, in no word of theirs"
    (m.x_at <> Apart && not shares);
  cover "x in a word of the values, a block sort" (shares && p = Block);
  cover "x in a word of the values, a select" (shares && p = Select);
  cover "x in a word of the values, a radix sort" (shares && p = Radix);
  let got =
    with_bounds c (fun () ->
        Nx_metal.sort ~axis:(axis c) ~descending:c.descending ~values ~positions
          x)
  in
  equal ~msg:"nx.metal's answer" answer A.Done got;
  let xh = Support.host x in
  let module C = (val Support.cpu.kernels) in
  let wv = A.create Rig.host dt y and wp = A.create Rig.host D.Int64 y in
  equal ~msg:"nx.cpu's answer" answer A.Done
    (C.sort ~axis:(axis c) ~descending:c.descending ~values:wv ~positions:wp xh);
  agree ((A.Any values, A.Any positions), (A.Any wv, A.Any wp));
  (* Every bit of the allocation but the values' own. *)
  let others b =
    Array.init
      (8 * Bytes.length b)
      (fun i ->
        if i >= lo && i < hi then 0
        else (Char.code (Bytes.get b (i / 8)) lsr (i mod 8)) land 1)
  in
  equal ~msg:"bits outside the values" (array int) (others before)
    (others (read ()))

(* nx_metal.mli: a sort in a block or by radix into values that are x itself is
   declined, at every width. *)
let test_identical () =
  ignore (backend ());
  List.iter
    (fun (D.Any dt, shape) ->
      let x = A.create (S.device ()) dt (Iarray.of_array shape) in
      let positions = A.create (S.device ()) D.Int64 (Iarray.of_array shape) in
      let axis = Array.length shape - 1 in
      equal
        ~msg:(Format.asprintf "%a %d" D.pp dt shape.(axis))
        answer A.Declined
        (Nx_metal.sort ~axis ~descending:false ~values:x ~positions x))
    [
      (D.Any D.Float32, [| 3; 100 |]);
      (D.Any D.Float32, [| 5000 |]);
      (D.Any D.Uint4, [| 3; 100 |]);
      (D.Any D.Int4, [| 5000 |]);
      (D.Any D.Int64, [| 2; 5000 |]);
    ]

(* Plans of any length

   nx_metal.mli's launches and workspace of each path, planned from the
   operands' facts with no buffer, at shapes up to 2^31 elements, a core holding
   one group of each kernel, the fewest. The estimates are baselines: a change
   to a weight or a cost shows in their diff. *)

let mib = 1024 * 1024

(* nx_metal.mli's workspace bytes of a sort of [count] slices of [n] keeping
   [keep], its pieces rounded up to 256 bytes: at most 256 parts a slice, and
   the copy where x is copied. *)
let stated (D.Any dt) ~n ~count ~keep ~copied =
  let e = Int.max 1 (D.bits dt / 8) and k = key_bytes (D.Any dt) in
  let big = n >= 1 lsl 32 in
  let c = cap (D.Any dt) in
  let piece b = (b + 255) / 256 * 256 in
  let copy = if copied then piece (count * n * e) else 0 in
  if n <= c then copy
  else if keep <= c / 4 then
    copy
    + piece (count * if big then 2112 else 1088)
    + piece (count * c * (e + 8))
  else
    let buffer = piece (count * n * (e + if big then 8 else 4)) in
    copy
    + piece (count * k * 256 * if big then 16 else 8)
    + piece (count * 4)
    + piece (count * 256 * 256 * if big then 8 else 4)
    + (if k >= 2 then buffer else 0)
    + if k >= 3 then buffer else 0

(* The kernels of [p]'s launches after its tickets' zeroing, but a sort kernel's
   launches past the first of a split, which start past its first unit. *)
let runs (p : Plan.t) =
  Array.to_list p.launches
  |> List.filteri (fun i _ -> i >= p.zeroing)
  |> List.filter_map (fun (l : Plan.launch) ->
      let name = K.name l.kernel in
      let past () = Bytes.get_int64_le l.words (8 * K.sort_first) > 0L in
      if String.starts_with ~prefix:"sort_" name && past () then None
      else Some name)

(* The workspace bytes of [p]: the device's, or a buffer of its own. *)
let workspace (p : Plan.t) =
  let own = ref 0 in
  Array.iteri
    (fun i -> function
      | Plan.Fresh b when i <> p.tickets_slot -> own := !own + b | _ -> ())
    p.slots;
  p.workspace + !own

let launch_seconds (l : Plan.launch) =
  Plan.seconds ~threads:(K.threads l.kernel) ~groups:l.groups
    ~resident:l.resident l.cost

let test_long_plans () =
  let operand buffer dt shape =
    {
      Nx_metal.Operands.dtype = dt;
      layout = L.contiguous (Iarray.of_array shape);
      buffer;
      skew = 0;
      residue = 0;
      pad = None;
    }
  in
  let one ?(axis = -1) ?keep (D.Any d as dt) shape =
    let r = Array.length shape in
    let axis = if axis < 0 then r - 1 else axis in
    let n = shape.(axis) in
    let keep = Option.value keep ~default:n in
    let count = Array.fold_left ( * ) 1 shape / n in
    let y = Array.copy shape in
    y.(axis) <- keep;
    let what =
      Printf.sprintf "%s %s%s%s" (D.name d)
        (String.concat "x" (Array.to_list (Array.map string_of_int shape)))
        (if axis <> r - 1 then Printf.sprintf " axis %d" axis else "")
        (if keep < n then Printf.sprintf " keep %d" keep else "")
    in
    let got =
      Nx_metal.Sort.of_facts
        ~holds:(fun k -> K.threads k)
        ~axis ~descending:false ~shares:false ~values:(operand 0 dt y)
        ~positions:(operand 1 (D.Any D.Int64) y)
        (operand 2 dt shape)
    in
    match got with
    | Error A.Declined -> what ^ ": declined"
    | Ok None -> what ^ ": no plan"
    | Error a -> failf "%s: %a" what Nx_array_support.pp_answer a
    | Ok (Some p) ->
        let seconds = Array.map launch_seconds p.launches in
        let longest = Array.fold_left Float.max 0. seconds in
        let total = Array.fold_left ( +. ) 0. seconds in
        at_most
          ~msg:(what ^ ", longest launch")
          float_exact ~than:Plan.budget longest;
        Array.iteri
          (fun i (q : Plan.part) ->
            at_most
              ~msg:(Printf.sprintf "%s, part %d" what i)
              float_exact ~than:Plan.part_budget
              (Array.fold_left (fun t l -> t +. launch_seconds l) 0. q.run))
          p.parts;
        let ks = runs p in
        let copy = String.starts_with ~prefix:"copy" in
        let copied = List.exists copy ks in
        let ks = List.filter (fun k -> not (copy k)) ks in
        let k = key_bytes dt and c = cap dt in
        let path =
          if n <= c then [ "block" ]
          else if keep <= c / 4 then
            let pb = position_bytes n in
            List.init (k + pb + 3) (fun _ -> "select")
          else
            let counts =
              List.length
                (List.filter (String.starts_with ~prefix:"sort_count") ks)
            in
            List.init (2 + k + counts) (fun _ -> "radix")
        in
        equal
          ~msg:(what ^ ", launches by nx_metal.mli")
          int (List.length path) (List.length ks);
        at_most ~msg:(what ^ ", workspace") int
          ~than:(stated dt ~n ~count ~keep ~copied)
          (workspace p);
        let parts =
          match Array.length p.parts with
          | 1 -> ""
          | n -> Printf.sprintf " in %d parts" n
        in
        Printf.sprintf "%s: %d launches%s, %.3f s, max %.2f ms, %d MiB" what
          (Array.length p.launches) parts total (1e3 *. longest)
          ((workspace p + mib - 1) / mib)
  in
  let f32 = D.Any D.Float32 in
  let lines =
    [
      one f32 [| 4096; 4096 |];
      one f32 [| 1 lsl 19; 4096 |];
      one (D.Any D.Complex128) [| 8192; 2048 |];
      one (D.Any D.Int4) [| 4096; 4096 |];
      one f32 ~keep:1024 [| 1 lsl 24 |];
      one f32 ~keep:8 [| 64; 1 lsl 18 |];
      one f32 ~keep:1024 [| 1 lsl 31 |];
      one f32 [| 1 lsl 24 |];
      one f32 ~keep:(1 lsl 23) [| 1 lsl 24 |];
      one f32 [| 1024; 16384 |];
      one f32 [| 1 lsl 15; 1 lsl 16 |];
      one f32 [| 1 lsl 28 |];
      one f32 [| 1 lsl 31 |];
      one (D.Any D.Int64) [| 1 lsl 24 |];
      one (D.Any D.Complex128) [| 1 lsl 23 |];
      one (D.Any D.Uint8) [| 1 lsl 24 |];
      one (D.Any D.Int4) [| 1 lsl 24 |];
      one f32 ~axis:0 [| 4096; 4096 |];
      one f32 ~axis:0 [| 16384; 1024 |];
      one f32 ~axis:0 [| 1 lsl 16; 1 lsl 15 |];
    ]
  in
  expect (String.concat "\n" lines)
  @@ __POS_OF__
       {|
    float32 4096x4096: 3 launches, 0.116 s, max 49.99 ms, 0 MiB
    float32 524288x4096: 299 launches in 2 parts, 14.908 s, max 49.99 ms, 0 MiB
    complex128 8192x2048: 14 launches, 0.670 s, max 49.78 ms, 0 MiB
    int4 4096x4096: 1 launches, 0.032 s, max 31.91 ms, 0 MiB
    float32 16777216 keep 1024: 10 launches, 0.077 s, max 9.70 ms, 1 MiB
    float32 64x262144 keep 8: 10 launches, 0.082 s, max 9.70 ms, 4 MiB
    float32 2147483648 keep 1024: 224 launches in 2 parts, 10.984 s, max 50.00 ms, 1 MiB
    float32 16777216: 14 launches, 0.408 s, max 49.99 ms, 129 MiB
    float32 16777216 keep 8388608: 14 launches, 0.408 s, max 49.99 ms, 257 MiB
    float32 1024x16384: 10 launches, 0.371 s, max 49.96 ms, 137 MiB
    float32 32768x65536: 953 launches in 5 parts, 47.455 s, max 50.00 ms, 16641 MiB
    float32 268435456: 148 launches, 7.021 s, max 50.00 ms, 2049 MiB
    float32 2147483648: declined
    int64 16777216: 34 launches, 1.012 s, max 49.89 ms, 385 MiB
    complex128 8388608: 50 launches, 1.656 s, max 49.95 ms, 321 MiB
    uint8 16777216: 5 launches, 0.101 s, max 49.90 ms, 1 MiB
    int4 16777216: 5 launches, 0.100 s, max 49.82 ms, 1 MiB
    float32 4096x4096 axis 0: 5 launches, 0.223 s, max 49.97 ms, 0 MiB
    float32 16384x1024 axis 0: 19 launches, 0.828 s, max 49.99 ms, 265 MiB
    float32 65536x32768 axis 0: 2133 launches in 11 parts, 106.274 s, max 49.99 ms, 33025 MiB
    |}

(* Numbering *)

(* [rows] rows of [w] int64 words, each one of [distinct] rows. *)
let unique_on st ~rows ~w ~distinct =
  let pool =
    Array.init distinct (fun _ ->
        Array.init w (fun _ -> Random.State.int64 st 4L))
  in
  let data =
    Array.concat (List.init rows (fun _ -> pool.(Random.State.int st distinct)))
  in
  A.of_array D.Int64 (Iarray.of_array [| rows; w |]) data

(* A numbering on the GPU and on nx.cpu gives the same bits, and is the GPU's
   launches; [None] where nx.metal declines and [declinable]. *)
let numbers ?(declinable = false) ~rows ~w ~distinct ~seed () =
  let st = Random.State.make [| seed |] in
  let keys = unique_on st ~rows ~w ~distinct in
  let run d (module M : Nx_kernel.S) keys =
    let ids = A.create d D.Int64 (Iarray.of_array [| rows |]) in
    let first = A.create d D.Int64 (Iarray.of_array [| rows |]) in
    let counts = A.create d D.Int64 (Iarray.of_array [| rows |]) in
    let count = A.create d D.Int64 (Iarray.of_array [||]) in
    let r = M.unique ~ids ~first ~counts ~count keys in
    ( r,
      List.map Support.elements
        [ A.Any ids; A.Any first; A.Any counts; A.Any count ] )
  in
  let (r, got), launches =
    Plan.traced (fun () ->
        run (S.device ()) (module Nx_metal) (Support.on (backend ()) keys))
  in
  let module C = (val Support.cpu.kernels) in
  let w_answer, want = run Rig.host (module C) keys in
  equal ~msg:"nx.cpu's answer" answer A.Done w_answer;
  match r with
  | A.Declined when declinable -> None
  | _ ->
      equal answer A.Done r;
      equal (list (array string)) want got;
      Some launches

let numbered ~rows ~w ~distinct ~seed =
  ignore (numbers ~rows ~w ~distinct ~seed ())

(* nx_metal.mli: a numbering of more than 8 columns takes a submission of its
   own; one of 8 is kept. *)
let test_unique_columns () =
  let d = (backend ()).device in
  numbered ~rows:5000 ~w:9 ~distinct:40 ~seed:1;
  let before = Nx_metal.kept_calls d in
  numbered ~rows:5000 ~w:9 ~distinct:40 ~seed:2;
  equal ~msg:"9 columns" int before (Nx_metal.kept_calls d);
  numbered ~rows:5000 ~w:8 ~distinct:40 ~seed:3;
  let kept = Nx_metal.kept_calls d in
  numbered ~rows:5000 ~w:8 ~distinct:40 ~seed:4;
  equal ~msg:"8 columns, again" int kept (Nx_metal.kept_calls d);
  equal ~msg:"8 columns" bool true (kept > before)

type numbering = {
  rows : int;
  w : int;
  distinct : int;
  nparts : int;
  nwide : int;
  nbudget : float option;
  nseed : int;
}

(* Under a lowered budget, a numbering's launches that take a ticket of the call
   split across its parts; a budget below what one group's work takes is
   declined, and such a case says nothing. *)
let law_numbers c =
  ignore (backend ());
  cover "sorted by radix" (c.rows > 4096);
  cover "one column" (c.w = 1);
  cover "several columns" (c.w > 1);
  cover "64-bit positions" (c.rows >= 1 lsl c.nwide);
  let restore () =
    Nx_metal.bound_sorts ~parts:256 ~wide:32;
    Plan.bound_budget Plan.budget
  in
  Fun.protect ~finally:restore (fun () ->
      Nx_metal.bound_sorts ~parts:c.nparts ~wide:c.nwide;
      Option.iter Plan.bound_budget c.nbudget;
      match
        numbers ~declinable:(c.nbudget <> None) ~rows:c.rows ~w:c.w
          ~distinct:c.distinct ~seed:c.nseed ()
      with
      | None -> assume false
      | Some launches ->
          let launched k =
            List.length
              (List.filter (fun (l : Plan.traced) -> l.kernel = k) launches)
          in
          cover "unique_count's parts across launches"
            (launched K.Unique_count > 1);
          cover "unique_heads' parts across launches"
            (launched K.Unique_heads > 1))

(* Numberings of up to 12,000 rows; once in three, of 40,000 to 80,000 rows of 2
   to 4 columns, under a budget of 400 to 700 us, which a column sort's
   sort_hist of a chunk read through the order, about 360 us, allows. *)
let numberings =
  let open Gen in
  let pp ppf c =
    Format.fprintf ppf "%d rows of %d words, %d distinct; parts %d, wide %d%s"
      c.rows c.w c.distinct c.nparts c.nwide
      (match c.nbudget with
      | None -> ""
      | Some s -> Printf.sprintf ", launches of %g s" s)
  in
  with_pp pp
    (let* split = frequency [ (2, constant false); (1, constant true) ] in
     let* rows =
       if split then int_range 40000 80000
       else frequency [ (1, int_range 1 4096); (1, int_range 4097 12000) ]
     in
     let* w =
       if split then int_range 2 4
       else frequency [ (1, constant 1); (2, int_range 2 4) ]
     in
     let* nbudget =
       if split then
         map (fun u -> Some (4e-4 *. ((7e-4 /. 4e-4) ** u))) (float_range 0. 1.)
       else constant None
     in
     let+ distinct = int_range 1 50
     and+ nparts = frequency [ (2, constant 256); (1, int_range 1 3) ]
     and+ nwide = frequency [ (1, constant 32); (1, constant 12) ]
     and+ nseed = nat in
     { rows; w; distinct; nparts; nwide; nbudget; nseed })

let gpu name f =
  test name (fun () ->
      ignore (backend ());
      f ())

(* Every cover of the split, mask, sub-byte and numbering laws, so that each run
   reaches each of them. *)
let split_examples =
  let split ?(fresh = false) ?keep ~count ~budget n =
    { (example ?keep ~slices:[| count |] n) with budget = Some budget; fresh }
  in
  [
    split ~count:400 ~budget:1e-3 ~fresh:true 4096;
    split ~count:61 ~keep:1024 ~budget:3.5e-4 40000;
    split ~count:47 ~keep:700 ~budget:5e-4 35000;
    split ~count:33 ~keep:1 ~budget:7e-4 20000;
    split ~count:1 ~budget:1.5e-4 400000;
    split ~count:4 ~budget:1.5e-4 300000;
    split ~count:64 ~budget:5e-4 6000;
  ]

let mask_examples =
  List.map
    (fun (dt, wide) ->
      { (example ~dt ~slices:[| 3 |] ~draw:Mixed ~wide 9000) with keep = 9000 })
    [ (D.Any D.Int16, 32); (D.Any D.Int32, 32); (D.Any D.Int64, 12) ]

let mid_examples =
  let mid ?keep dt n ~first ~lead ~tail x_at =
    { c = example ?keep ~dt ~slices:[| 3 |] n; first; lead; tail; x_at }
  in
  [
    mid (D.Any D.Uint4) 100 ~first:1 ~lead:1 ~tail:0 After;
    mid (D.Any D.Uint4) 100 ~first:0 ~lead:0 ~tail:1 After;
    mid (D.Any D.Bit) 100 ~first:2 ~lead:3 ~tail:0 Before;
    mid (D.Any D.Bit) ~keep:100 5000 ~first:2 ~lead:3 ~tail:0 After;
    mid (D.Any D.Int4) ~keep:10 5000 ~first:1 ~lead:0 ~tail:0 Before;
    mid (D.Any D.Int4) 5000 ~first:3 ~lead:0 ~tail:0 After;
    mid (D.Any D.Float4_e2m1fn) 5000 ~first:2 ~lead:5 ~tail:2 Before;
    mid (D.Any D.Float4_e2m1fn) 5000 ~first:1 ~lead:5 ~tail:2 Apart;
  ]

let numbering_examples =
  [
    {
      rows = 80000;
      w = 4;
      distinct = 50;
      nparts = 256;
      nwide = 32;
      nbudget = Some 4e-4;
      nseed = 1;
    };
    {
      rows = 5000;
      w = 1;
      distinct = 7;
      nparts = 256;
      nwide = 12;
      nbudget = None;
      nseed = 2;
    };
  ]

let () =
  exit
    (Windtrap.run "nx.metal sorts"
       [
         group "sorts"
           [
             prop "each path gives nx.cpu's bits in the launches stated"
               ~examples cases (fun c ->
                 ignore (backend ());
                 law_sorts c);
             prop ~count:60 ~examples:split_examples
               "split launches give nx.cpu's bits" splits (fun c ->
                 ignore (backend ());
                 law_splits c);
             prop ~count:30 ~examples:split_examples
               "split launches cut into parts at every launch give nx.cpu's \
                bits"
               splits (fun c ->
                 ignore (backend ());
                 S.parted ~launches:1 (fun () -> law_splits ~parted:true c));
             prop ~count:30 ~examples:split_examples
               "a sort past a part's seconds runs in parts with nx.cpu's bits"
               splits (fun c ->
                 ignore (backend ());
                 S.parted ~seconds:(2. *. Plan.dispatch) (fun () ->
                     law_splits ~parted:true c));
             prop ~count:40 ~examples:mask_examples
               "slices of different passes give nx.cpu's bits" masks (fun c ->
                 ignore (backend ());
                 law_masks c);
             prop ~count:100 ~examples:mid_examples
               "sub-byte values at any bit give nx.cpu's bits and keep the \
                others"
               mids (fun m ->
                 ignore (backend ());
                 law_mid m);
             gpu "values that are x itself decline in a block or by radix"
               test_identical;
             gpu "more than 8 kept axes give nx.cpu's bits" test_deep;
             test "the hidden bounds refuse what they don't hold" test_bounds;
             test
               "plans to 2^31 elements keep the launches and workspace stated"
               test_long_plans;
           ];
         group "numbering"
           [
             prop ~count:60 ~examples:numbering_examples
               "rows number as nx.cpu numbers them" numberings law_numbers;
             gpu "more than 8 columns take a submission of their own"
               test_unique_columns;
           ];
       ])
