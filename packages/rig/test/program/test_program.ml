(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module G = Rig_program
open Rig_program

let timeout = 60.
let strf = Printf.sprintf

(* Fresh Polled devices: a name opens once. *)
let opened = Atomic.make 0

let polled ?peers name =
  P.open_ ?peers (strf "program:%s-%d" name (Atomic.fetch_and_add opened 1))

let le64 v =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int v);
  Bytes.to_string b

let read b =
  let n = B.length b in
  let h = B.create Rig.host n in
  B.copy ~src:b ~dst:h;
  let s = Bytes.create n in
  B.blit_to_bytes h 0 s 0 n;
  Bytes.to_string s

let of_bytes d s =
  let b = B.create d (String.length s) in
  B.copy ~src:(B.of_string s) ~dst:b;
  b

let data bytes = { G.bytes; holes = [||] }
let functions = data "functions"

let hole ?(width = G.W64) ?(add = 0) ?(shift = 0) at x =
  { G.at; width; terms = [| (x, 1) |]; add; shift }

(* A launch of Polled's [fill]: for [groups] groups, the 64-bit word [i] of its
   slot [0] is [base + i]. *)
let fill ?(holes : _ iarray = [||]) ?(use = G.Write) ~image ~groups base =
  let bytes = String.make 8 '\000' ^ le64 base in
  (Launch
        {
          image;
          name = "fill";
          params = { bytes; holes };
          refs = [| { G.slot = 0; at = 0; use } |];
          groups = (Fixed groups, Fixed 1, Fixed 1);
          threads = (Fixed 1, Fixed 1, Fixed 1);
          shared = Fixed 0;
        })

(* A launch of Polled's [copy]: [bytes] bytes from its slot [src] (defaults
   to [0]) to its slot [dst] (defaults to [1]). *)
let kcopy ?(src = 0) ?(dst = 1) ~image bytes =
  (Launch
        {
          image;
          name = "copy";
          params = data (String.make 16 '\000' ^ le64 bytes);
          refs =
            [|
              { G.slot = src; at = 0; use = Read };
              { G.slot = dst; at = 8; use = Write };
            |];
          groups = (Fixed 1, Fixed 1, Fixed 1);
          threads = (Fixed 1, Fixed 1, Fixed 1);
          shared = Fixed 0;
        })

(* A submit whose slots are [buffers]'s, their accesses the refs'. *)
let submit ?(buffers : _ iarray = [||]) ?(fixed : _ iarray = [||]) device parts
    =
  G.Submit { device; parts; slots = Iarray.map fst buffers; fixed }

let alloc ?(copies = G.One) ?(init = data "") device bytes =
  G.Alloc { device; kind = B.Device; bytes; init; copies }

let all memory = G.Memory { memory; offset = 0; length = 64 }

(* A copy part of [length] bytes from slot [src] (defaults to [0]) at
   [src_offset] (defaults to [0]) to slot [dst] (defaults to [1]) at
   [dst_offset] (defaults to [0]). *)
let pcopy ?(src = 0) ?(src_offset : _ iarray = [||]) ?(dst = 1)
    ?(dst_offset : _ iarray = [||]) length =
  G.Copy { src; src_offset; dst; dst_offset; length }

let load_ok t ds =
  require_ok ~pp:Format.pp_print_string (G.load t (Iarray.of_list ds))

(* The host code of [f]'s object in [dir], an image on device [device]. *)
let host_image ?(dir = "../fixtures") device f =
  { G.device; binary = data (Rig_host_support.fixture ~dir f) }

(* Steps against a model *)

(* Three memories and two inputs of 64 bytes, each on one of two devices, and
   steps over them. *)

let size = 64

type loc = Mem of int | In of int

type op =
  | Fill of { dev : int; dst : loc; groups : int; base : int }
  | Kcopy of { dev : int; src : loc; dst : loc; bytes : int }
  | Pcopy of {
      dev : int;
      src : loc;
      dst : loc;
      length : int;
      src_at : int;
      dst_at : int;
      per_run : bool;  (* Whether the run reads the offsets from its ints. *)
    }
  | Move of { src : loc; dst : loc }

type case = {
  mem_dev : int array;
  two : bool array;
  init : string array;
  in_dev : int array;
  inputs : string array;
  ops : op list;
}

let pp_loc ppf = function
  | Mem i -> Format.fprintf ppf "m%d" i
  | In i -> Format.fprintf ppf "x%d" i

let pp_op ppf = function
  | Fill { dev; dst; groups; base } ->
      Format.fprintf ppf "fill@%d %a %d groups from %d" dev pp_loc dst groups
        base
  | Kcopy { dev; src; dst; bytes } ->
      Format.fprintf ppf "copy@%d %a->%a %d bytes" dev pp_loc src pp_loc dst
        bytes
  | Pcopy { dev; src; dst; length; src_at; dst_at; per_run } ->
      Format.fprintf ppf "copy part@%d %a+%d->%a+%d %d bytes%s" dev pp_loc src
        src_at pp_loc dst dst_at length
        (if per_run then " per run" else "")
  | Move { src; dst } -> Format.fprintf ppf "move %a->%a" pp_loc src pp_loc dst

let pp_case ppf c =
  Format.fprintf ppf "memory on [%s] two [%s]; inputs on [%s]; [%a]"
    (String.concat ";" (Array.to_list (Array.map string_of_int c.mem_dev)))
    (String.concat ";" (Array.to_list (Array.map string_of_bool c.two)))
    (String.concat ";" (Array.to_list (Array.map string_of_int c.in_dev)))
    (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") pp_op)
    c.ops

let gen_case =
  let open Gen in
  let dev = int_range 0 1 in
  let loc =
    one_of
      [
        map (fun i -> Mem i) (int_range 0 2);
        map (fun i -> In i) (int_range 0 1);
      ]
  in
  let two_locs =
    let* src = loc in
    let+ dst = such_that (fun d -> d <> src) loc in
    (src, dst)
  in
  let op =
    frequency
      [
        ( 3,
          let+ dev = dev
          and+ dst = loc
          and+ groups = int_range 1 (size / 8)
          and+ base = int_range 0 1_000_000 in
          Fill { dev; dst; groups; base } );
        ( 2,
          let+ dev = dev
          and+ src, dst = two_locs
          and+ words = int_range 0 (size / 8) in
          Kcopy { dev; src; dst; bytes = 8 * words } );
        ( 2,
          let+ dev = dev
          and+ src, dst = two_locs
          and+ length = one_of [ constant size; int_range 0 size ]
          and+ src_at = int_range 0 size
          and+ dst_at = int_range 0 size
          and+ per_run = bool in
          let fit at = at mod (size - length + 1) in
          Pcopy
            {
              dev;
              src;
              dst;
              length;
              src_at = fit src_at;
              dst_at = fit dst_at;
              per_run;
            } );
        (* A copy within one slot, between its two halves. *)
        ( 1,
          let+ dev = dev
          and+ at = loc
          and+ length = int_range 0 (size / 2)
          and+ a = int_range 0 size
          and+ b = int_range 0 size
          and+ down = bool
          and+ per_run = bool in
          let half = size / 2 in
          let low = a mod (half - length + 1)
          and high = half + (b mod (half - length + 1)) in
          let src_at, dst_at = if down then (high, low) else (low, high) in
          Pcopy { dev; src = at; dst = at; length; src_at; dst_at; per_run } );
        ( 1,
          let+ src, dst = two_locs in
          Move { src; dst } );
      ]
  in
  let bytes = string_of ~size:(constant size) char in
  let+ mem_dev = array ~size:(constant 3) dev
  and+ two = array ~size:(constant 3) bool
  and+ init = array ~size:(constant 3) bytes
  and+ in_dev = array ~size:(constant 2) dev
  and+ inputs = array ~size:(constant 2) bytes
  and+ ops = list ~size:(int_range 1 6) op in
  { mem_dev; two; init; in_dev; inputs; ops }

let gen_case = Gen.with_pp pp_case gen_case
let devices = lazy (fst (polled "model-a"), fst (polled "model-b"))

(* The offset [at] of a copy part's end: a constant, or [2 * w + at mod 2]
   where word [w] of the run's ints holds [at / 2]. *)
let copy_offset ~per_run w at : _ iarray =
  if per_run then [| (G.Int w, 2); (Fixed (at land 1), 1) |]
  else [| (Fixed at, 1) |]

(* The run's ints: words [2 * i] and [2 * i + 1] hold half the offsets of op
   [i], a copy part whose offsets the run reads. *)
let copy_ints c =
  let ops = Array.of_list c.ops in
  Array.init
    (2 * Array.length ops)
    (fun w ->
      match ops.(w / 2) with
      | Pcopy { src_at; dst_at; _ } ->
          (if w land 1 = 0 then src_at else dst_at) / 2
      | Fill _ | Kcopy _ | Move _ -> 0)

let describe c =
  let slot = function Mem m -> all m | In i -> G.Input i in
  let image d = d in
  let step i = function
    | Fill { dev; dst; groups; base } ->
        submit dev
          ~buffers:[| (slot dst, Rig.Read_write) |]
          [| fill ~image:(image dev) ~groups base |]
    | Kcopy { dev; src; dst; bytes } ->
        submit dev
          ~buffers:
            [|
              (slot src, Rig.Read);
              (slot dst, Rig.Read_write);
            |]
          [| kcopy ~image:(image dev) bytes |]
    | Pcopy { dev; src; dst; length; src_at; dst_at; per_run } ->
        submit dev
          ~buffers:[| (slot src, Rig.Read); (slot dst, Rig.Read_write) |]
          [|
            Copy
              {
                src = 0;
                src_offset = copy_offset ~per_run (2 * i) src_at;
                dst = 1;
                dst_offset = copy_offset ~per_run ((2 * i) + 1) dst_at;
                length;
              };
          |]
    | Move { src; dst } -> G.Move { src = slot src; dst = slot dst }
  in
  let a, b = Lazy.force devices in
  {
    G.devices = [| Rig.arch a; Rig.arch b |];
    ints = 2 * List.length c.ops;
    memory =
      Iarray.init 3 (fun m ->
          alloc
            ~copies:(if c.two.(m) then Two else One)
            ~init:(data c.init.(m))
            c.mem_dev.(m) size);
    images =
      [|
        { device = 0; binary = functions }; { device = 1; binary = functions };
      |];
    inputs =
      Iarray.map
        (fun device -> { G.device; bytes = size })
        (Iarray.of_array c.in_dev);
    steps = Iarray.of_list (List.mapi step c.ops);
  }

(* The model: memory of two copies is one per run parity. *)
let model c =
  let mem =
    Array.init 3 (fun m -> Array.init 2 (fun _ -> Bytes.of_string c.init.(m)))
  in
  fun n ->
    let inputs = Array.map Bytes.of_string c.inputs in
    let cell = function
      | Mem m -> mem.(m).(if c.two.(m) then n land 1 else 0)
      | In i -> inputs.(i)
    in
    List.iter
      (function
        | Fill { dst; groups; base; _ } ->
            for i = 0 to groups - 1 do
              Bytes.set_int64_le (cell dst) (8 * i) (Int64.of_int (base + i))
            done
        | Kcopy { src; dst; bytes; _ } ->
            Bytes.blit (cell src) 0 (cell dst) 0 bytes
        | Pcopy { src; dst; length; src_at; dst_at; _ } ->
            Bytes.blit (cell src) src_at (cell dst) dst_at length
        | Move { src; dst } -> Bytes.blit (cell src) 0 (cell dst) 0 size)
      c.ops;
    Array.map Bytes.to_string inputs

let model_law c =
  let a, b = Lazy.force devices in
  let ds = [| a; b |] in
  let p = load_ok (describe c) [ a; b ] in
  let expect = model c in
  for n = 0 to 2 do
    let xs = Array.map2 (fun d s -> of_bytes ds.(d) s) c.in_dev c.inputs in
    let points = G.run p { inputs = xs; ints = copy_ints c } in
    let ran =
      List.filter
        (fun d ->
          List.exists
            (function
              | Fill { dev; _ } | Kcopy { dev; _ } | Pcopy { dev; _ } ->
                  dev = d
              | Move _ -> false)
            c.ops)
        [ 0; 1 ]
    in
    equal ~msg:"one point per device that ran, in their order" (list string)
      (List.map
         (fun d -> strf "%s:%d" (Rig.name ds.(d)) (Rig.submitted ds.(d)))
         ran)
      (List.map
         (fun pt -> Format.asprintf "%a" Rig.Point.pp pt)
         (Array.to_list points));
    equal
      ~msg:(strf "the inputs after run %d" n)
      (array string) (expect n) (Array.map read xs)
  done;
  (* A step that names memory another device's step wrote before. *)
  let rec crossed last = function
    | [] -> false
    | op :: ops ->
        let dev, reads, writes =
          match op with
          | Fill { dev; dst; _ } -> (Some dev, [], [ dst ])
          | Kcopy { dev; src; dst; _ } | Pcopy { dev; src; dst; _ } ->
              (Some dev, [ src ], [ dst ])
          | Move { src; dst } -> (None, [ src ], [ dst ])
        in
        let seen l =
          match (List.assoc_opt l last, dev) with
          | Some (Some d'), Some d -> d' <> d
          | _ -> false
        in
        List.exists seen (reads @ writes)
        || crossed (List.map (fun l -> (l, dev)) writes @ last) ops
  in
  cover "a step follows another device's step" (crossed [] c.ops);
  cover "memory of two copies" (Array.exists Fun.id c.two);
  cover "an input another device borrows"
    (List.exists
       (function
         | Fill { dev; dst = In i; _ }
         | Kcopy { dev; dst = In i; _ }
         | Pcopy { dev; dst = In i; _ } ->
             dev <> c.in_dev.(i)
         | _ -> false)
       c.ops);
  cover "a move" (List.exists (function Move _ -> true | _ -> false) c.ops);
  cover "a copy part into an input"
    (List.exists (function Pcopy { dst = In _; _ } -> true | _ -> false) c.ops);
  cover "a copy part out of an input"
    (List.exists (function Pcopy { src = In _; _ } -> true | _ -> false) c.ops);
  cover "a copy part between memories"
    (List.exists
       (function Pcopy { src = Mem _; dst = Mem _; _ } -> true | _ -> false)
       c.ops);
  cover "a copy part of part of its slots"
    (List.exists
       (function Pcopy { length; _ } -> length < size | _ -> false)
       c.ops);
  cover "a copy part within one slot"
    (List.exists
       (function Pcopy { src; dst; _ } -> src = dst | _ -> false)
       c.ops);
  cover "a copy part at offsets the run reads"
    (List.exists
       (function
         | Pcopy { per_run; src_at; dst_at; _ } ->
             per_run && src_at + dst_at > 0
         | _ -> false)
       c.ops)

(* Holes *)

type hole_case = { width : G.width; add : int; shift : int }

let gen_hole =
  let open Gen in
  let+ width = of_list [ G.W32; G.W64 ]
  and+ add = int_range 0 (1 lsl 20)
  and+ shift = int_range 0 62 in
  { width; add; shift }

let gen_hole =
  Gen.with_pp
    (fun ppf h ->
      Format.fprintf ppf "%s, add %d, shift %d"
        (match h.width with W32 -> "W32" | W64 -> "W64")
        h.add h.shift)
    gen_hole

let shaped h at x =
  { G.at; width = h.width; terms = [| (x, 1) |]; add = h.add; shift = h.shift }

let word s i = Int64.to_int (String.get_int64_le s (8 * i))

(* A fill of one group whose base is a hole: it stores the base at word [i] of
   its write slot. *)
let fill_word i hole =
  (Launch
        {
          image = 0;
          name = "fill";
          params = { bytes = le64 (8 * i) ^ le64 0; holes = [| hole |] };
          refs = [| { G.slot = 0; at = 0; use = Write } |];
          groups = (Fixed 1, Fixed 1, Fixed 1);
          threads = (Fixed 1, Fixed 1, Fixed 1);
          shared = Fixed 0;
        })

(* Memory 0's address lands in word 0 of input 0, through a plain hole; the same
   through [h], in its word 1; and [h] over it in memory 1's init, then memory
   0's handle, both moved into input 1. *)
let hole_law h =
  let d = fst (Lazy.force devices) in
  let address = G.Address { memory = 0; on = 0 } in
  let init =
    {
      G.bytes = String.make 16 '\000';
      holes = [| shaped h 0 address; hole 8 (G.Handle 0) |];
    }
  in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 0;
      memory = [| alloc 0 8; alloc ~init 0 16 |];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 16 }; { device = 0; bytes = 16 } |];
      steps =
        [|
          submit 0
            ~buffers:[| (Input 0, Rig.Read_write) |]
            ~fixed:[| ({ memory = 0; offset = 0; length = 8 }, Rig.Read) |]
            [| fill_word 0 (hole 8 (G.Leaf address)) |];
          submit 0
            ~buffers:[| (Input 0, Rig.Read_write) |]
            ~fixed:[| ({ memory = 0; offset = 0; length = 8 }, Rig.Read) |]
            [| fill_word 1 (shaped h 8 (G.Leaf address)) |];
          Move
            {
              src = Memory { memory = 1; offset = 0; length = 16 };
              dst = Input 1;
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let x = B.create d 16 and y = B.create d 16 in
  ignore (G.run p { inputs = [| x; y |]; ints = [||] });
  let x = read x and y = read y in
  let a = word x 0 in
  let w = (a + h.add) lsr h.shift in
  let low32 = w land 0xffff_ffff in
  equal ~msg:"a launch's hole" int
    (match h.width with W32 -> low32 | W64 -> w)
    (word x 1);
  let memory_word =
    match h.width with
    | W32 -> low32 (* the init's other 4 bytes stay 0 *)
    | W64 -> w
  in
  equal ~msg:"a memory's hole" int memory_word (word y 0);
  equal ~msg:"Polled's handle is the address" int a (word y 1);
  cover "a shift past the address's bits" (h.shift > 40);
  cover "a W32 hole" (h.width = W32)

(* Affine holes *)

(* A term of an affine hole: word [i] of the frame's ints, a constant, memory
   0's address. *)
type term = Frame of int | Const of int | Addr

type affine = {
  width : G.width;
  add : int;
  shift : int;
  terms : (term * int) list;
  frames : (int * int) list;  (** The frame's two ints, per run. *)
}

let pp_affine ppf c =
  let term ppf = function
    | Frame i, f -> Format.fprintf ppf "int %d * %d" i f
    | Const n, f -> Format.fprintf ppf "%d * %d" n f
    | Addr, f -> Format.fprintf ppf "address * %d" f
  in
  Format.fprintf ppf "%s, add %d, shift %d, terms [%a], frames [%s]"
    (match c.width with W32 -> "W32" | W64 -> "W64")
    c.add c.shift
    (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") term)
    c.terms
    (String.concat "; "
       (List.map (fun (a, b) -> Printf.sprintf "%d,%d" a b) c.frames))

let gen_affine =
  let open Gen in
  let factor =
    frequency
      [
        (3, int_range (-4) 4);
        (1, of_list [ 1 lsl 32; -(1 lsl 40); max_int; min_int ]);
      ]
  in
  let term =
    one_of
      [
        map (fun i -> Frame i) (int_range 0 1);
        map (fun n -> Const n) (int_range (-(1 lsl 30)) (1 lsl 30));
        constant Addr;
      ]
  in
  let+ width = of_list [ G.W32; G.W64 ]
  and+ add = int_range (-(1 lsl 20)) (1 lsl 20)
  and+ shift =
    frequency [ (1, of_list [ 0; 1; 31; 32; 33; 61; 62 ]); (2, int_range 0 62) ]
  and+ terms = list ~size:(int_range 0 4) (pair term factor)
  and+ frames = list ~size:(int_range 1 3) (pair int int) in
  { width; add; shift; terms; frames }

let gen_affine = Gen.with_pp pp_affine gen_affine

(* The word an affine hole of [c] writes, [value] giving its terms' values:
   the [int] sum taken modulo [2{^64}], divided by [2{^shift}], and for [W32]
   its low 32 bits. *)
let affine_sum c value =
  List.fold_left (fun s (x, f) -> s + (value x * f)) c.add c.terms

let affine_word c value =
  let sum = Int64.of_int (affine_sum c value) in
  let w = Int64.unsigned_div sum (Int64.shift_left 1L c.shift) in
  match c.width with W32 -> Int64.logand w 0xffff_ffffL | W64 -> w

(* A launch's hole of the case's terms writes, on each run, the sum of its
   terms' values times their factors, plus [add], shifted, as the case's
   reference computes it; and the same hole over memory leaves in memory 1's
   init, filled at load, moved into input 1. *)
let affine_law c =
  let d = fst (Lazy.force devices) in
  let address = G.Address { memory = 0; on = 0 } in
  let value_of : term -> G.value = function
    | Frame i -> Int i
    | Const n -> Fixed n
    | Addr -> Leaf address
  in
  let leaf_terms = List.map (fun (_, f) -> (address, f)) c.terms in
  let affine at terms =
    let terms = Iarray.of_list terms in
    { G.at; width = c.width; terms; add = c.add; shift = c.shift }
  in
  let fixed : _ iarray =
    [| ({ G.memory = 0; offset = 0; length = 8 }, Rig.Read) |]
  in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 2;
      memory =
        [|
          alloc 0 8;
          alloc 0 8
            ~init:
              {
                G.bytes = String.make 8 '\000';
                holes = [| affine 0 leaf_terms |];
              };
        |];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 16 }; { device = 0; bytes = 8 } |];
      steps =
        [|
          submit 0
            ~buffers:[| (Input 0, Rig.Read_write) |]
            ~fixed
            [| fill_word 0 (hole 8 (G.Leaf address)) |];
          submit 0
            ~buffers:[| (Input 0, Rig.Read_write) |]
            ~fixed
            [|
              fill_word 1
                (affine 8 (List.map (fun (x, f) -> (value_of x, f)) c.terms));
            |];
          Move
            {
              src = Memory { memory = 1; offset = 0; length = 8 };
              dst = Input 1;
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let shifted_negative = ref false in
  List.iter
    (fun (i0, i1) ->
      let x = B.create d 16 and y = B.create d 8 in
      Array.iter Rig.Point.wait
        (G.run p { inputs = [| x; y |]; ints = [| i0; i1 |] });
      let x = read x and y = read y in
      let a = word x 0 in
      let value = function
        | Frame 0 -> i0
        | Frame _ -> i1
        | Const n -> n
        | Addr -> a
      in
      if c.shift > 0 && affine_sum c value < 0 then shifted_negative := true;
      let word64 s i = String.get_int64_le s (8 * i) in
      equal ~msg:"a launch's hole" int64 (affine_word c value) (word64 x 1);
      equal ~msg:"a memory's hole" int64
        (affine_word c (fun _ -> a))
        (word64 y 0))
    c.frames;
  let factors = List.map snd c.terms in
  cover "two terms or more" (List.length c.terms >= 2);
  cover "no term" (c.terms = []);
  cover "a negative factor" (List.exists (fun f -> f < 0) factors);
  cover "a term the frame gives"
    (List.exists (function Frame _, _ -> true | _ -> false) c.terms);
  cover "a W32 hole" (c.width = W32);
  cover "a shift at a 32-bit boundary" (c.shift = 31 || c.shift = 32);
  cover "a negative sum, shifted" !shifted_negative

(* Two copies *)

let one_device_fill ?(copies = G.One) d =
  {
    G.devices = [| Rig.arch d |];
    ints = 0;
    memory = [| alloc ~copies 0 64 |];
    images = [| { device = 0; binary = functions } |];
    inputs = [||];
    steps =
      [|
        submit 0
          ~buffers:[| (all 0, Rig.Read_write) |]
          [| fill ~image:0 ~groups:1 7 |];
      |];
  }

let test_two_waits () =
  let d, pd = polled "two" in
  let p = load_ok (one_device_fill ~copies:Two d) [ d ] in
  ignore (P.launches pd);
  let empty = { G.inputs = [||]; ints = [||] } in
  ignore (G.run p empty);
  ignore (G.run p empty);
  equal ~msg:"runs 0 and 1 handed over, neither run" int 0
    (List.length (P.launches pd));
  let last = G.run p empty in
  greater ~msg:"run 2 waited for run 0's work on its copy" int ~than:0
    (List.length (P.launches pd));
  Array.iter Rig.Point.wait last

let test_one_waits_not () =
  let d, pd = polled "one" in
  let p = load_ok (one_device_fill d) [ d ] in
  ignore (P.launches pd);
  let empty = { G.inputs = [||]; ints = [||] } in
  let last = List.init 3 (fun _ -> G.run p empty) in
  equal ~msg:"memory of one copy: runs wait only by its stamps, in the queue"
    int 0
    (List.length (P.launches pd));
  List.iter (Array.iter Rig.Point.wait) last

(* After *)

let test_after () =
  let a, _ = polled "after-a" and b, _ = polled "after-b" in
  let pa = load_ok (one_device_fill a) [ a ] in
  let pb = load_ok (one_device_fill b) [ b ] in
  let empty = { G.inputs = [||]; ints = [||] } in
  let pt = (G.run pa empty).(0) in
  equal ~msg:"a's run not reached" bool true
    (Rig.signaled a < Rig.Point.value pt);
  let last = G.run ~waits:[| pt |] pb empty in
  greater ~msg:"b's run followed it" int
    ~than:(Rig.Point.value pt - 1)
    (Rig.signaled a);
  Array.iter Rig.Point.wait last

(* Refusals *)

let base d =
  {
    G.devices = [| Rig.arch d |];
    ints = 0;
    memory = [| alloc 0 64 |];
    images = [| { device = 0; binary = functions } |];
    inputs = [| { device = 0; bytes = 64 } |];
    steps =
      [|
        submit 0
          ~buffers:[| (Input 0, Rig.Read_write) |]
          [| fill ~image:0 ~groups:1 0 |];
      |];
  }

let with_params bytes (t : G.t) =
  {
    t with
    steps =
      [|
        submit 0
          ~buffers:[| (Input 0, Rig.Read_write) |]
          [|
            (Launch
                  {
                    image = 0;
                    name = "fill";
                    params = data bytes;
                    refs = [||];
                    groups = (Fixed 1, Fixed 1, Fixed 1);
                    threads = (Fixed 1, Fixed 1, Fixed 1);
                    shared = Fixed 0;
                  });
          |];
      |];
  }

let addr memory = G.Address { memory; on = 0 }

let holed ?width ?shift n h =
  { G.bytes = String.make n '\000'; holes = [| hole ?width ?shift 0 h |] }

(* [base d] whose step copies [length] bytes of its slot [0] into its slot
   [1], of [slots], and fills its slot [0] with a ref of [use]. *)
let copying ?(ints = 0) ?use ?src_offset ?dst_offset ?(length = 64) slots d =
  let rw s = (s, Rig.Read_write) in
  {
    (base d) with
    ints;
    steps =
      [|
        submit 0
          ~buffers:(Iarray.map rw slots)
          [|
            fill ?use ~image:0 ~groups:1 0;
            pcopy ?src_offset ?dst_offset length;
          |];
      |];
  }

let refusals =
  let one d = [ d ] in
  let head = G.Memory { memory = 0; offset = 0; length = 8 } in
  [
    ("as many devices as the program's", (fun d -> base d), fun _ -> []);
    ( "a device's arch",
      (fun d -> { (base d) with devices = [| "no such arch" |] }),
      one );
    ( "a memory's device",
      (fun d -> { (base d) with memory = [| alloc 1 64 |] }),
      one );
    ( "a memory's size",
      (fun d -> { (base d) with memory = [| alloc 0 (-1) |] }),
      one );
    ( "an init longer than its memory",
      (fun d ->
        {
          (base d) with
          memory = [| alloc ~init:(data (String.make 65 'x')) 0 64 |];
        }),
      one );
    ( "a hole past its bytes",
      (fun d ->
        { (base d) with memory = [| alloc ~init:(holed 4 (addr 0)) 0 64 |] }),
      one );
    ( "a shift of 63",
      (fun d ->
        {
          (base d) with
          memory = [| alloc ~init:(holed ~shift:63 8 (addr 0)) 0 64 |];
        }),
      one );
    ( "memory of two copies named in memory of one",
      (fun d ->
        {
          (base d) with
          memory =
            [| alloc ~copies:Two 0 8; alloc ~init:(holed 8 (addr 0)) 0 8 |];
        }),
      one );
    ( "memory of two copies named in an image",
      (fun d ->
        {
          (base d) with
          memory = [| alloc ~copies:Two 0 8 |];
          images = [| { device = 0; binary = holed 16 (addr 0) } |];
        }),
      one );
    ( "an entry of a later image",
      (fun d ->
        {
          (base d) with
          images =
            [|
              {
                device = 0;
                binary = holed 16 (G.Entry { image = 1; name = "fill" });
              };
              { device = 0; binary = functions };
            |];
        }),
      one );
    ( "a view past its memory",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0
                ~buffers:
                  [|
                    ( Memory { memory = 0; offset = 60; length = 8 },
                      Rig.Read_write );
                  |]
                [| fill ~image:0 ~groups:1 0 |];
            |];
        }),
      one );
    ( "an input's index",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0
                ~buffers:[| (Input 3, Rig.Read_write) |]
                [| fill ~image:0 ~groups:1 0 |];
            |];
        }),
      one );
    ( "an image's index",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0
                ~buffers:[| (Input 0, Rig.Read_write) |]
                [| fill ~image:5 ~groups:1 0 |];
            |];
        }),
      one );
    ( "parameters of no whole word",
      (fun d -> with_params (String.make 6 '\000') (base d)),
      one );
    ( "parameters over 4096 bytes",
      (fun d -> with_params (String.make 4100 '\000') (base d)),
      one );
    ( "an address on no device",
      (fun d ->
        {
          (base d) with
          memory =
            [| alloc ~init:(holed 8 (G.Address { memory = 0; on = 2 })) 0 64 |];
        }),
      one );
    ( "holes that share a byte",
      (fun d ->
        {
          (base d) with
          memory =
            [|
              alloc
                ~init:
                  {
                    bytes = String.make 16 '\000';
                    holes = [| hole 0 (addr 0); hole ~width:W32 4 (addr 0) |];
                  }
                0 64;
            |];
        }),
      one );
    ( "a slot no ref names",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0
                ~buffers:[| (Input 0, Rig.Read_write); (Input 0, Rig.Read) |]
                [| fill ~image:0 ~groups:1 0 |];
            |];
        }),
      one );
    ( "memory named by an unordered ref and a copy",
      copying ~use:Unordered [| all 0; Input 0 |],
      one );
    ( "an input named by an unordered ref and a copy",
      copying ~use:Unordered [| Input 0; all 0 |],
      one );
    ( "memory named by an unordered and a read ref",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0
                ~buffers:[| (all 0, Rig.Read_write); (Input 0, Rig.Read) |]
                [|
                  fill ~use:Unordered ~image:0 ~groups:1 0; kcopy ~image:0 8;
                |];
            |];
        }),
      one );
    ( "a copy past the end of its source",
      copying ~ints:2 ~length:16 [| head; Ints |],
      one );
    ( "a copy past the end of its destination",
      copying ~ints:1 ~length:16 [| all 0; Ints |],
      one );
    ( "a copy at a negative offset",
      copying ~src_offset:[| (Fixed (-8), 1) |] ~length:8 [| all 0; Input 0 |],
      one );
    ( "a copy of a negative length",
      copying ~length:(-1) [| all 0; Input 0 |],
      one );
    ( "a copy between overlapping bytes of the ints",
      copying ~ints:2 ~dst_offset:[| (Fixed 4, 1) |] ~length:8 [| Ints; Ints |],
      one );
    ( "a copy between overlapping bytes of an input",
      copying ~dst_offset:[| (Fixed 4, 1) |] ~length:8 [| Input 0; Input 0 |],
      one );
    ( "a copy offset of a leaf",
      copying ~src_offset:[| (Leaf (Handle 0), 1) |] ~length:8
        [| all 0; Input 0 |],
      one );
    ( "a copy offset past an int",
      copying ~src_offset:[| (Fixed max_int, 2) |] ~length:8
        [| all 0; Input 0 |],
      one );
    ( "a copy offset whose sum passes an int",
      copying
        ~src_offset:[| (Fixed max_int, 1); (Fixed 1, 1) |]
        ~length:8 [| all 0; Input 0 |],
      one );
    ( "a copy offset of an int past the ints",
      copying ~ints:1
        ~src_offset:[| (Int 1, 1) |]
        ~length:8 [| all 0; Input 0 |],
      one );
    ( "a ref past its slots",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0
                ~buffers:[| (Input 0, Rig.Read_write) |]
                [| kcopy ~image:0 8 |];
            |];
        }),
      one );
    ( "a grid a run refuses",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0
                ~buffers:[| (Input 0, Rig.Read_write) |]
                [| fill ~image:0 ~groups:(1 lsl 33) 0 |];
            |];
        }),
      one );
    ( "an image its device refuses",
      (fun d ->
        {
          (base d) with
          images = [| { device = 0; binary = data "nonsense" } |];
        }),
      one );
    ( "a function its image lacks",
      (fun d ->
        {
          (base d) with
          memory =
            [|
              alloc ~init:(holed 8 (G.Entry { image = 0; name = "nope" })) 0 64;
            |];
        }),
      one );
  ]

let test_refusal (_, t, ds) =
  let d = fst (polled "refused") in
  is_error
    ~pp:(fun _ _ -> ())
    ~msg:"load answers Error"
    (G.load (t d) (Iarray.of_list (ds d)))

let test_unborrowable () =
  let a, _ = polled ~peers:false "lone-a"
  and b, _ = polled ~peers:false "lone-b" in
  let t =
    {
      G.devices = [| Rig.arch a; Rig.arch b |];
      ints = 0;
      memory = [| alloc 1 64 |];
      images = [| { device = 0; binary = functions } |];
      inputs = [||];
      steps =
        [|
          submit 0
            ~buffers:[| (all 0, Rig.Read_write) |]
            [| fill ~image:0 ~groups:1 0 |];
        |];
    }
  in
  let why = require_error (G.load t [| a; b |]) in
  contains ~msg:"names the device" ~sub:(Rig.name a) why

(* An Address is of memory with an address, which memory its driver names by
   handle only has not. *)
let test_handle_only () =
  let d, _ =
    P.open_ ~addresses:false
      (strf "program:handle-only-%d" (Atomic.fetch_and_add opened 1))
  in
  let t =
    {
      (base d) with
      steps =
        [|
          submit 0
            ~buffers:[| (Input 0, Rig.Read_write) |]
            ~fixed:[| (G.{ memory = 0; offset = 0; length = 64 }, Rig.Read) |]
            [| fill ~image:0 ~groups:1 0 ~holes:[| hole 8 (Leaf (addr 0)) |] |];
        |];
    }
  in
  let why = require_error (G.load t [| d |]) in
  contains ~msg:"says why" ~sub:"by its handle only" why

let test_machines () =
  let d = fst (polled "machines") in
  let far =
    Rig_support.machine (strf "program-%d" (Atomic.fetch_and_add opened 1))
  in
  let t = { (base d) with devices = [| Rig.arch d; Rig.arch far |] } in
  raises_match Exn.invalid_arg (fun () -> G.load t [| d; far |])

let test_frame_refusals () =
  let d = fst (polled "frame") and e = fst (polled "frame-other") in
  let p = load_ok (base d) [ d ] in
  let refused msg inputs =
    raises_match ~msg Exn.invalid_arg (fun () ->
        G.run p { inputs; ints = [||] })
  in
  refused "no input" [||];
  refused "two inputs" [| B.create d 64; B.create d 64 |];
  refused "an input of another device" [| B.create e 64 |];
  refused "an input of fewer bytes" [| B.create d 63 |];
  Array.iter Rig.Point.wait
    (G.run p { inputs = [| B.create d 65 |]; ints = [||] })

(* Host code and loops *)

(* A launch on device [host] of test/fixtures' [affine], image [image], which
   stores [a * in[i] + c] at [out[i]] for [i] below [n]: [x.(0) <- a * x.(0)
   + c]. *)
let step_affine ?(host = 1) ?(image = 1) x ~a ~c =
  submit host
    ~buffers:[| (x, Rig.Read_write) |]
    [|
      Launch
        {
          image;
          name = "affine";
          params =
            {
              bytes = String.make 16 '\000' ^ le64 1 ^ String.make 16 '\000';
              holes = [| hole 24 a; hole 32 c |];
            };
          refs =
            [|
              { G.slot = 0; at = 0; use = Write }; { G.slot = 0; at = 8; use = Read };
            |];
          groups = (Fixed 1, Fixed 1, Fixed 1);
          threads = (Fixed 1, Fixed 1, Fixed 1);
          shared = Fixed 0;
        };
    |]

(* A program on Polled's [d] (device 0) and the host (device 1), with Polled's
   functions as image 0 and [affine] as image 1. *)
let with_host d (t : G.t) =
  {
    t with
    G.devices = [| Rig.arch d; Rig.arch Rig.host |];
    images = [| { device = 0; binary = functions }; host_image 1 "affine" |];
  }

let one_word d = of_bytes d (le64 0)
let word_of b = word (read b) 0

(* A first-byte address or handle reaches all of its memory. Every step that
   reads a pointer-bearing container must name its pointees, transitively. *)
let memory_hole_refusals () =
  let d, _ = polled "memory-holes" in
  let whole m = { memory = m; offset = 0; length = 64 } in
  let short m = { memory = m; offset = 0; length = 8 } in
  let launch leaf = fill ~image:0 ~groups:1 0 ~holes:[| hole 8 (Leaf leaf) |] in
  let geometry leaf =
    match launch leaf with
    | Launch l ->
        G.Launch
          { l with params = data (String.make 16 '\000'); shared = Leaf leaf }
    | _ -> assert false
  in
  let main = G.Entry { image = 0; name = "main" } in
  let filling ?(target = main) leaf =
    G.Fill
      {
        fill = target;
        params = holed 16 (Leaf leaf);
        refs = [| { G.slot = 0; at = 8; use = Write } |];
      }
  in
  (* A fill that reads slots [0] to [n - 1]. *)
  let reading n =
    G.Fill
      {
        fill = main;
        params = data (String.make (8 * n) '\000');
        refs = Iarray.init n (fun slot -> { G.slot; at = 8 * slot; use = Read });
      }
  in
  let sub ?(fixed : _ iarray = [||])
      ?(buffers : _ iarray = [| (Input 0, Rig.Read_write) |]) parts =
    G.Submit { device = 0; fixed; slots = Iarray.map fst buffers; parts }
  in
  let t = base d in
  let cases =
    [
      ( "launch Address without fixed memory",
        { t with steps = [| sub [| launch (addr 0) |] |] } );
      ( "launch Address through a slot",
        {
          t with
          steps =
            [|
              sub
                ~buffers:[| (Memory (whole 0), Rig.Read) |]
                [| launch (addr 0) |];
            |];
        } );
      ( "launch Address through partial fixed memory",
        {
          t with
          steps =
            [| sub ~fixed:[| (short 0, Rig.Read) |] [| launch (addr 0) |] |];
        } );
      ( "launch Handle without whole memory",
        {
          t with
          steps =
            [|
              sub
                ~buffers:[| (Memory (short 0), Rig.Read) |]
                [| launch (Handle 0) |];
            |];
        } );
      ( "launch geometry Address",
        { t with steps = [| sub [| geometry (addr 0) |] |] } );
      ( "launch geometry Handle",
        { t with steps = [| sub [| geometry (Handle 0) |] |] } );
      ("fill Address", { t with steps = [| sub [| filling (addr 0) |] |] });
      ("fill Handle", { t with steps = [| sub [| filling (Handle 0) |] |] });
      ( "fill Address function",
        {
          t with
          steps =
            [|
              sub
                ~fixed:[| (whole 0, Rig.Read) |]
                [| filling ~target:(addr 0) main |];
            |];
        } );
      ( "fill Handle function",
        {
          t with
          steps =
            [|
              sub
                ~fixed:[| (whole 0, Rig.Read) |]
                [| filling ~target:(Handle 0) main |];
            |];
        } );
      ( "image Address",
        {
          t with
          images = [| { device = 0; binary = holed 8 (addr 0) } |];
          steps = [||];
        } );
      ( "image Handle",
        {
          t with
          images = [| { device = 0; binary = holed 8 (Handle 0) } |];
          steps = [||];
        } );
      ( "loop Address",
        {
          t with
          steps =
            [|
              Loop
                {
                  trips = Leaf (addr 0);
                  trip = None;
                  flag = Some (Memory (whole 0), 0);
                  body = [||];
                };
            |];
        } );
      ( "loop Handle without whole memory",
        {
          t with
          steps =
            [|
              Loop
                {
                  trips = Leaf (Handle 0);
                  trip = None;
                  flag = Some (Memory (short 0), 0);
                  body = [||];
                };
            |];
        } );
      ( "init Address without a fixing step",
        {
          t with
          memory = [| alloc ~init:(holed 8 (addr 0)) 0 64 |];
          steps = [||];
        } );
      ( "init Handle omitted by Submit",
        {
          t with
          memory = [| alloc ~init:(holed 8 (Handle 1)) 0 64; alloc 0 64 |];
          steps =
            [|
              sub
                ~buffers:[| (Memory (short 0), Rig.Read) |]
                [| reading 1 |];
            |];
        } );
      ( "transitive init Handle",
        {
          t with
          memory =
            [|
              alloc ~init:(holed 8 (Handle 1)) 0 64;
              alloc ~init:(holed 8 (Handle 2)) 0 64;
              alloc 0 64;
            |];
          steps =
            [|
              sub
                ~buffers:
                  [|
                    (Memory (short 0), Rig.Read); (Memory (whole 1), Rig.Read);
                  |]
                [| reading 2 |];
            |];
        } );
    ]
  in
  List.iter
    (fun (name, t) ->
      match G.load t [| d |] with
      | Error why -> contains ~msg:name ~sub:"memory" why
      | Ok _ -> failf "load accepted %s" name)
    cases

type ints_case = { n : int; a : int; c : int }

let gen_ints =
  let open Gen in
  let+ n = int_range 0 8 and+ a = int_range 0 3 and+ c = int_range 1 4 in
  { n; a; c }

let gen_ints =
  Gen.with_pp
    (fun ppf i -> Format.fprintf ppf "n %d, a %d, c %d" i.n i.a i.c)
    gen_ints

(* The frame's int [n] becomes [a * n + c] through host code; a launch over as
   many groups writes as many words. *)
let ints_law i =
  let d = fst (Lazy.force devices) in
  let t =
    with_host d
    {
      G.devices = [||];
      ints = 1;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 256 } |];
      steps =
        [|
          step_affine Ints ~a:(Fixed i.a) ~c:(Fixed i.c);
          G.Submit
            {
              device = 0;
              slots = [| Input 0 |];
              fixed = [||];
              parts =
                [|
                  (match fill ~image:0 ~groups:1 100 with
                  | Launch l -> G.Launch { l with groups = (Int 0, Fixed 1, Fixed 1) }
                  | f -> f);
                |];
            };
        |];
    }
  in
  let p = load_ok t [ d; Rig.host ] in
  let x = of_bytes d (String.make 256 '\000') in
  ignore (G.run p { inputs = [| x |]; ints = [| i.n |] });
  let m = (i.a * i.n) + i.c in
  equal ~msg:"a launch over the computed groups" string
    (String.concat "" (List.init m (fun j -> le64 (100 + j)))
    ^ String.make (256 - (8 * m)) '\000')
    (read x)

(* A loop of [k] trips: host code counts the trips in input 0, and a launch
   stores each trip's index in input 1. *)
let loop_law k =
  let d = fst (Lazy.force devices) in
  let t =
    with_host d
    {
      G.devices = [||];
      ints = 2;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = Iarray.init 2 (fun _ -> { G.device = 0; bytes = 8 });
      steps =
        [|
          Loop
            {
              trips = Int 0;
              trip = Some 1;
              flag = None;
              body =
                [|
                  step_affine (Input 0) ~a:(Fixed 1) ~c:(Fixed 1);
                  submit 0
                    ~buffers:[| (Input 1, Rig.Read_write) |]
                    [|
                      fill ~image:0 ~groups:1 0
                        ~holes:[| hole 8 (G.Int 1 : value) |];
                    |];
                |];
            };
        |];
    }
  in
  let p = load_ok t [ d; Rig.host ] in
  let counter = one_word d and last = one_word d in
  ignore (G.run p { inputs = [| counter; last |]; ints = [| k |] });
  equal ~msg:"trips" int (max 0 k) (word_of counter);
  equal ~msg:"the last trip's index" int (max 0 (k - 1)) (word_of last)

(* Host code that stores [0] into word [w] of the run's ints. *)
let clear_int w =
  submit 1
    ~buffers:[| (G.Ints, Rig.Read_write) |]
    [|
      Launch
        {
          image = 1;
          name = "affine";
          params =
            data (le64 (8 * w) ^ le64 (8 * w) ^ le64 1 ^ le64 0 ^ le64 0);
          refs =
            [|
              { G.slot = 0; at = 0; use = Write };
              { G.slot = 0; at = 8; use = Read };
            |];
          groups = (Fixed 1, Fixed 1, Fixed 1);
          threads = (Fixed 1, Fixed 1, Fixed 1);
          shared = Fixed 0;
        };
    |]

(* A loop whose flag a step clears in its first trip runs once, in memory the
   program makes, in an input and in the ints, the flag's byte inside its slot:
   the flag is read after that work. *)
let test_flag () =
  let d, _ = polled "flag" in
  let mem = G.Memory { memory = 0; offset = 0; length = 8 } in
  let clear slot =
    submit 0
      ~buffers:[| (slot, Rig.Read_write) |]
      [| fill ~image:0 ~groups:1 0 |]
  in
  let trips flag clear =
    let t =
      with_host d
        {
          G.devices = [||];
          ints = 3;
          memory = [| alloc ~init:(data (le64 1)) 0 8 |];
          images = [||];
          inputs = [| { device = 0; bytes = 8 }; { device = 0; bytes = 8 } |];
          steps =
            [|
              Loop
                {
                  trips = Fixed 10;
                  trip = None;
                  flag = Some flag;
                  body =
                    [|
                      clear; step_affine (Input 0) ~a:(Fixed 1) ~c:(Fixed 1);
                    |];
                };
            |];
        }
    in
    let p = load_ok t [ d; Rig.host ] in
    let counter = one_word d in
    let frame =
      { G.inputs = [| counter; of_bytes d (le64 1) |]; ints = [| 1; 1; 1 |] }
    in
    ignore (G.run p frame);
    word_of counter
  in
  equal ~msg:"memory" int 1 (trips (mem, 0) (clear mem));
  equal ~msg:"an input" int 1 (trips (Input 1, 0) (clear (Input 1)));
  equal ~msg:"the ints" int 1 (trips (Ints, 0) (clear_int 0));
  equal ~msg:"word 2 of the ints" int 1 (trips (Ints, 16) (clear_int 2))

(* A loop's flag past its input's bytes raises at the run, which load cannot
   know. *)
let test_flag_outside () =
  let d, _ = polled "flag-outside" in
  let t =
    {
      (base d) with
      steps =
        [|
          Loop
            {
              trips = Fixed 1;
              trip = None;
              flag = Some (Input 0, 8);
              body = [||];
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let ones = of_bytes d (String.make 64 '\001') in
  ignore (G.run p { inputs = [| ones |]; ints = [||] });
  raises_match ~msg:"byte 8 of 8" Exn.invalid_arg (fun () ->
      G.run p { inputs = [| of_bytes d (le64 1) |]; ints = [||] })

(* [calling], image 1, calls [affine], image 0, through its address, an
   [Entry] leaf in its parameters: three trips of a loop add 3. *)
let calling ~(callee : G.leaf) x =
  submit 0
    ~buffers:[| (x, Rig.Read_write) |]
    [|
      Launch
        {
          image = 1;
          name = "calling";
          params =
            {
              bytes =
                String.make 24 '\000' ^ le64 1 ^ le64 1 ^ le64 1;
              holes = [| hole 0 (Leaf callee) |];
            };
          refs =
            [|
              { G.slot = 0; at = 8; use = Write }; { G.slot = 0; at = 16; use = Read };
            |];
          groups = (Fixed 1, Fixed 1, Fixed 1);
          threads = (Fixed 1, Fixed 1, Fixed 1);
          shared = Fixed 0;
        };
    |]

let test_entry_leaf () =
  let t =
    {
      G.devices = [| Rig.arch Rig.host |];
      ints = 0;
      memory = [||];
      images = [| host_image 0 "affine"; host_image 0 "calling" |];
      inputs = [| { device = 0; bytes = 8 } |];
      steps =
        [|
          Loop
            {
              trips = Fixed 3;
              trip = None;
              flag = None;
              body =
                [|
                  calling
                    ~callee:(Entry { image = 0; name = "affine" })
                    (Input 0);
                |];
            };
        |];
    }
  in
  let p = load_ok t [ Rig.host ] in
  let counter = one_word Rig.host in
  ignore (G.run p { inputs = [| counter |]; ints = [||] });
  equal ~msg:"three calls of affine" int 3 (word_of counter)

let host_refusals =
  let two d = [ d; Rig.host ] in
  let with_steps ?(ints = 0) d steps =
    with_host d { (base d) with ints; steps }
  in
  [
      ( "host code that does not link",
        (fun d ->
          {
            (with_steps d [||]) with
            images = [| { device = 1; binary = data "nonsense" } |];
          }),
        two );
      ( "a host value of a rail this machine has not",
        (fun d ->
          with_steps d
            [| step_affine (Input 0) ~a:(Leaf (Ready 9)) ~c:(Fixed 0) |]),
        two );
      ( "a host value of a function its image lacks",
        (fun d ->
          with_steps d
            [|
              step_affine (Input 0)
                ~a:(Leaf (Entry { image = 0; name = "nope" }))
                ~c:(Fixed 0);
            |]),
        two );
      ( "a loop's trips of a rail this machine has not",
        (fun d ->
          with_steps d
            [|
              Loop
                {
                  trips = Leaf (Ready 9);
                  trip = None;
                  flag = None;
                  body = [||];
                };
            |]),
        two );
      ( "an int past the ints",
        (fun d ->
          with_steps ~ints:1 d [| step_affine Ints ~a:(Int 1) ~c:(Fixed 0) |]),
        two );
      ( "a trip past the ints",
        (fun d ->
          with_steps ~ints:1 d
            [|
              Loop { trips = Fixed 1; trip = Some 1; flag = None; body = [||] };
            |]),
        two );
      ( "a flag of no byte",
        (fun d ->
          with_steps d
            [|
              Loop
                {
                  trips = Fixed 1;
                  trip = None;
                  flag =
                    Some (Memory { memory = 0; offset = 0; length = 0 }, 0);
                  body = [||];
                };
            |]),
        two );
      ( "a flag past its view",
        (fun d ->
          with_steps d
            [|
              Loop
                {
                  trips = Fixed 1;
                  trip = None;
                  flag =
                    Some (Memory { memory = 0; offset = 0; length = 8 }, 8);
                  body = [||];
                };
            |]),
        two );
      ( "a flag at a negative byte",
        (fun d ->
          with_steps d
            [|
              Loop
                {
                  trips = Fixed 1;
                  trip = None;
                  flag = Some (Input 0, -1);
                  body = [||];
                };
            |]),
        two );
      ( "a flag past the ints",
        (fun d ->
          with_steps ~ints:1 d
            [|
              Loop
                {
                  trips = Fixed 1;
                  trip = None;
                  flag = Some (Ints, 8);
                  body = [||];
                };
            |]),
        two );
    ]

(* Memory a host launch names that the host does not address. *)
let test_host_unaddressed () =
  let d, _ =
    P.open_ ~host_visible:false
      (strf "program:hidden-%d" (Atomic.fetch_and_add opened 1))
  in
  let t =
    with_host d
      {
        (base d) with
        steps = [| step_affine (all 0) ~a:(Fixed 1) ~c:(Fixed 0) |];
      }
  in
  is_error ~pp:(fun _ _ -> ()) (G.load t [| d; Rig.host |])

(* A host launch writes its input 0: a read-only borrow of a file there is
   refused before the code runs, and the file keeps its bytes. *)
let test_read_only_input () =
  let path = Filename.temp_file "rig-program" ".bin" in
  Fun.protect ~finally:(fun () -> Sys.remove path) @@ fun () ->
  Out_channel.with_open_bin path (fun oc -> output_string oc (le64 5));
  let file = Result.get_ok (Rig_disk.of_file path) in
  let x = Option.get (B.borrow Rig.host file) in
  let t =
    {
      G.devices = [| Rig.arch Rig.host |];
      ints = 0;
      memory = [||];
      images = [| host_image 0 "affine" |];
      inputs = [| { device = 0; bytes = 8 } |];
      steps =
        [| step_affine ~host:0 ~image:0 (Input 0) ~a:(Fixed 2) ~c:(Fixed 1) |];
    }
  in
  let p = load_ok t [ Rig.host ] in
  raises_match ~msg:"a read-only input" Exn.invalid_arg (fun () ->
      G.run p { inputs = [| x |]; ints = [||] });
  equal ~msg:"the file's bytes" string (le64 5)
    (In_channel.with_open_bin path In_channel.input_all)

(* A copy part writes its destination: a read-only borrow of a file there is
   refused before the run's work, and the file keeps its bytes. *)
let test_read_only_copy () =
  let d, _ = polled "read-only-copy" in
  let path = Filename.temp_file "rig-program" ".bin" in
  Fun.protect ~finally:(fun () -> Sys.remove path) @@ fun () ->
  Out_channel.with_open_bin path (fun oc -> output_string oc (le64 5));
  let file = Result.get_ok (Rig_disk.of_file path) in
  let x = require_some (B.borrow d file) in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 0;
      memory = [| alloc ~init:(data (le64 7)) 0 8 |];
      images = [||];
      inputs = [| { device = 0; bytes = 8 } |];
      steps =
        [|
          submit 0
            ~buffers:
              [|
                (Memory { memory = 0; offset = 0; length = 8 }, Rig.Read);
                (Input 0, Rig.Read_write);
              |]
            [| pcopy 8 |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  raises_match ~msg:"a read-only input" Exn.invalid_arg (fun () ->
      G.run p { inputs = [| x |]; ints = [||] });
  equal ~msg:"the file's bytes" string (le64 5)
    (In_channel.with_open_bin path In_channel.input_all)

let test_ints_refusal () =
  let d = fst (polled "ints") in
  let p = load_ok { (base d) with ints = 1 } [ d ] in
  raises_match ~msg:"two ints for one" Exn.invalid_arg (fun () ->
      G.run p { inputs = [| B.create d 64 |]; ints = [| 1; 2 |] })

(* Bytes *)

(* Descriptions of any shape, well formed or not: the format holds every value
   of the type. *)
let gen_t =
  let open Gen in
  let n = int_range (-2) 5000 in
  let few g = map Iarray.of_array (array ~size:(int_range 0 3) g) in
  let str = string_of ~size:(int_range 0 12) char in
  let leaf =
    one_of
      [
        (let+ memory = n and+ on = n in
         Address { memory; on });
        map (fun m -> Handle m) n;
        (let+ image = n and+ name = str in
         Entry { image; name });
        map (fun r -> Ready r) n;
      ]
  in
  let value =
    one_of
      [
        map (fun x -> Fixed x) int;
        map (fun i -> Int i) n;
        map (fun l -> Leaf l) leaf;
      ]
  in
  let data f =
    let+ bytes = str
    and+ holes =
      few
        (let+ at = n
         and+ width = of_list [ W32; W64 ]
         and+ terms = few (pair f int)
         and+ add = int
         and+ shift = n in
         { at; width; terms; add; shift })
    in
    { bytes; holes }
  in
  let view =
    let+ memory = n and+ offset = n and+ length = n in
    { memory; offset; length }
  in
  let slot =
    one_of
      [
        map (fun v -> Memory v) view;
        map (fun i -> (Input i : slot)) n;
        constant Ints;
      ]
  in
  let access = of_list [ Rig.Read; Rig.Read_write ] in
  let ref =
    let+ at = n and+ slot = n and+ use = of_list [ G.Read; Write; Unordered ] in
    { G.slot; at; use }
  in
  let triple = triple value value value in
  let part =
    one_of
      [
        map (fun v -> Words v) view;
        (let+ fill = leaf
         and+ params = data value
         and+ refs = few ref in
         G.Fill { fill; params; refs });
        (let+ src = n
         and+ src_offset = few (pair value int)
         and+ dst = n
         and+ dst_offset = few (pair value int)
         and+ length = n in
         (Copy { src; src_offset; dst; dst_offset; length } : G.part));
        (let+ image = n
         and+ name = str
         and+ params = data value
         and+ refs = few ref
         and+ groups = triple
         and+ threads = triple
         and+ shared = value in
         Launch { image; name; params; refs; groups; threads; shared });
      ]
  in
  let rec step depth =
    let leaves =
      [
        (let+ device = n
         and+ parts = few part
         and+ slots = few slot
         and+ fixed = few (pair view access) in
         Submit { device; parts; slots; fixed });
        (let+ src = slot and+ dst = slot in
         G.Move { src; dst });
      ]
    in
    if depth = 0 then one_of leaves
    else
      one_of
        ((let+ trips = value
          and+ trip = option n
          and+ flag = option (pair slot n)
          and+ body = few (step (depth - 1)) in
          Loop { trips; trip; flag; body })
        :: leaves)
  in
  let+ devices = few str
  and+ memory =
    few
      (one_of
         [
           (let+ device = n
            and+ kind = of_list [ B.Device; B.Pinned; B.Mapped ]
            and+ bytes = n
            and+ init = data leaf
            and+ copies = of_list [ One; Two ] in
            Alloc { device; kind; bytes; init; copies });
           (let+ rail = n and+ area = of_list [ Outbound; Inbound; Counts ] in
            Rail { rail; area });
         ])
  and+ images =
    few
      (let+ device = n and+ binary = data leaf in
       { device; binary })
  and+ inputs =
    few
      (let+ device = n and+ bytes = n in
       { device; bytes })
  and+ ints = n
  and+ steps = few (step 2) in
  { devices; memory; images; inputs; ints; steps }

let pp_t ppf t =
  Format.fprintf ppf "a description of %d bytes, %d steps"
    (String.length (G.to_string t))
    (Iarray.length t.steps)

let gen_t = Gen.with_pp pp_t gen_t
let description = Testable.make ~pp:pp_t ~equal:( = )

let round_trip t =
  equal (result description string) (Ok t) (G.of_string (G.to_string t))

(* A cut or a changed byte of a description's bytes decodes or answers [Error]:
   [of_string] raises nothing. *)
let damaged (t, cut, at, byte) =
  let s = G.to_string t in
  let n = String.length s in
  let cut = String.sub s 0 (cut mod (n + 1)) in
  let changed =
    String.mapi (fun i c -> if i = at mod n then Char.chr byte else c) s
  in
  ignore (G.of_string cut : (G.t, string) result);
  ignore (G.of_string changed : (G.t, string) result);
  cover "a cut inside" (String.length cut < n)

let gen_damage =
  let open Gen in
  let+ t = gen_t and+ cut = nat and+ at = nat and+ byte = int_range 0 255 in
  (t, cut, at, byte)

let test_version () =
  let s = Bytes.of_string (G.to_string (base (fst (polled "version")))) in
  Bytes.set_int64_le s 12 7L;
  let why = require_error (G.of_string (Bytes.to_string s)) in
  contains ~msg:"names the versions" ~sub:"version 7, not 13" why

(* A description cut inside its version is malformed where the cut is. *)
let test_cut_version () =
  let s = String.sub (G.to_string (base (fst (polled "cut-version")))) 0 15 in
  let why = require_error (G.of_string s) in
  contains ~msg:"says where" ~sub:"at byte 12" why

(* Rails *)

let area n = Bigarray.(Array1.create char c_layout n)

(* An end of a rail of this process whose ready function is the support's
   [ready]: a call stores its count at the first word of [counts]. *)
let fake_end () =
  Rig_remote_abi.end_ ~outbound:(area 512) ~inbound:(area 512)
    ~counts:(area 384) ~ready:ignore ~ready_fn:Rig_support.ready

let rail_program =
  {
    G.devices = [| Rig.arch Rig.host |];
    memory =
      [| Rail { rail = 7; area = Outbound }; Rail { rail = 7; area = Counts } |];
    images = [| host_image 0 "ready" |];
    inputs = [||];
    ints = 1;
    steps =
      [|
        submit 0
          ~buffers:
            [|
              (Memory { memory = 0; offset = 0; length = 8 }, Rig.Read_write);
              (Memory { memory = 1; offset = 0; length = 384 }, Rig.Read_write);
            |]
          [|
            (* test/fixtures' [ready] stores [value] at [out], then calls
               [ready] with [counts] and [count]. *)
            Launch
              {
                image = 0;
                name = "ready";
                params =
                  {
                    bytes = String.make 32 '\000' ^ le64 1;
                    holes = [| hole 16 (Leaf (Ready 7)); hole 24 (Int 0) |];
                  };
                refs =
                  [|
                    { G.slot = 0; at = 0; use = Write };
                    { G.slot = 1; at = 8; use = Write };
                  |];
                groups = (Fixed 1, Fixed 1, Fixed 1);
                threads = (Fixed 1, Fixed 1, Fixed 1);
                shared = Fixed 0;
              };
          |];
      |];
  }

(* A host launch writes a rail's outbound area and calls its ready function,
   which a [Ready] leaf names, with its counts. *)
let test_rail () =
  let e = fake_end () in
  let rails id = if id = 7 then Some e else None in
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load ~rails rail_program [| Rig.host |])
  in
  ignore (G.run p { inputs = [||]; ints = [| 42 |] });
  let outbound = String.init 8 (Bigarray.Array1.get e.outbound) in
  equal ~msg:"the outbound area" int 42 (word outbound 0);
  equal ~msg:"the ready function's count" int 1
    (word (read (B.of_bigarray e.counts)) 0)

let test_no_rail () =
  let why = require_error (G.load rail_program [| Rig.host |]) in
  contains ~msg:"names the rail" ~sub:"rail 7" why

(* Constants beside holes *)

type beside = {
  bytes : int64;  (** The word the description holds. *)
  value : int;
  width : G.width;
  shift : int;
  per_run : bool;  (** The value is the run's int, else fixed. *)
}

let gen_beside =
  let open Gen in
  let+ width = of_list [ G.W32; G.W64 ]
  and+ shift = int_range 0 62
  and+ value = int_range 0 max_int
  and+ raw = int64
  and+ clash = bool
  and+ per_run = bool in
  let field =
    let w = Int64.of_int (value lsr shift) in
    match width with W32 -> Int64.logand w 0xffff_ffffL | W64 -> w
  in
  let raw =
    match width with W32 -> Int64.logand raw 0xffff_ffffL | W64 -> raw
  in
  let bytes = if clash then raw else Int64.logand raw (Int64.lognot field) in
  { bytes; value; width; shift; per_run }

let gen_beside =
  Gen.with_pp
    (fun ppf b ->
      Format.fprintf ppf "bytes %Lx, value %x >> %d, %s, %s" b.bytes b.value
        b.shift
        (match b.width with W32 -> "W32" | W64 -> "W64")
        (if b.per_run then "per run" else "fixed"))
    gen_beside

(* A launch of [main] whose 8 parameter bytes hold [b.bytes] with a hole over
   [b.value]: a value that meets their set bits is refused, at load where it is
   fixed and at run where it is the run's; otherwise the launch reads the bytes
   with the value's bits ORed in, every other bit kept. *)
let beside_law b =
  let d, pd = polled "beside" in
  let leaf : value = if b.per_run then Int 0 else Fixed b.value in
  let bytes = Bytes.create 8 in
  Bytes.set_int64_le bytes 0 b.bytes;
  let t =
    {
      G.devices = [| Rig.arch d |];
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = [||];
      ints = 1;
      steps =
        [|
          submit 0
            [|
              (Launch
                    {
                      image = 0;
                      name = "main";
                      params =
                        {
                          bytes = Bytes.to_string bytes;
                          holes =
                            [|
                              {
                                G.at = 0;
                                width = b.width;
                                terms = [| (leaf, 1) |];
                                add = 0;
                                shift = b.shift;
                              };
                            |];
                        };
                      refs = [||];
                      groups = (Fixed 1, Fixed 1, Fixed 1);
                      threads = (Fixed 1, Fixed 1, Fixed 1);
                      shared = Fixed 0;
                    });
            |];
        |];
    }
  in
  let field =
    let w = Int64.of_int (b.value lsr b.shift) in
    match b.width with W32 -> Int64.logand w 0xffff_ffffL | W64 -> w
  in
  let clashes = Int64.logand b.bytes field <> 0L in
  cover "a value that meets set bits" clashes;
  cover "a value beside set bits" ((not clashes) && b.bytes <> 0L);
  cover "a value read per run" b.per_run;
  match (G.load t [| d |], clashes && not b.per_run) with
  | Error _, true -> ()
  | Ok _, true -> fail "load took a fixed value that meets set bits"
  | Error why, false -> failf "load: %s" why
  | Ok p, false ->
      ignore (P.launches pd);
      let frame = { G.inputs = [||]; ints = [| b.value |] } in
      if clashes then
        raises_match ~msg:"run refuses it" Exn.invalid_arg (fun () ->
            G.run p frame)
      else begin
        let pt = (G.run p frame).(0) in
        Rig.Point.wait pt;
        let l = List.hd (P.launches pd) in
        equal ~msg:"the bytes with the value ORed in" int64
          (Int64.logor b.bytes field)
          (String.get_int64_le l.params 0)
      end

(* Integers at their extremes *)

(* A description's integers may come from another process's bytes: each field
   whose sum with another bounds an access is checked where a sum wraps. *)
type extreme =
  | Ints of int
  | Hole_at of int
  | View_at of int
  | View_length of int

let pp_extreme ppf = function
  | Ints n -> Format.fprintf ppf "ints %d" n
  | Hole_at n -> Format.fprintf ppf "a hole at %d" n
  | View_at n -> Format.fprintf ppf "a view at %d" n
  | View_length n -> Format.fprintf ppf "a view of %d bytes" n

let extremes =
  List.concat_map
    (fun n ->
      let fields = [ Hole_at n; View_at n; View_length n ] in
      if n < 0 || n > max_int / 8 then Ints n :: fields else fields)
    [
      max_int;
      max_int - 1;
      max_int - 3;
      max_int - 7;
      min_int;
      min_int + 1;
      -1;
      1 lsl 60;
      (max_int / 8) + 1;
      (max_int / 16) + 1;
    ]

(* Each answers [Error] at load: none loads, and none raises. *)
let extreme_law e =
  let d = fst (Lazy.force devices) in
  let t = base d in
  let view offset length =
    submit 0
      ~buffers:[| (Memory { memory = 0; offset; length }, Rig.Read_write) |]
      [| fill ~image:0 ~groups:1 0 |]
  in
  let t =
    match e with
    | Ints n -> { t with ints = n }
    | Hole_at n ->
        {
          t with
          memory =
            [|
              alloc
                ~init:
                  {
                    bytes = String.make 8 '\000';
                    holes = [| hole n (addr 0) |];
                  }
                0 64;
            |];
        }
    | View_at n -> { t with steps = [| view n 8 |] }
    | View_length n -> { t with steps = [| view 8 n |] }
  in
  is_error ~pp:(fun _ _ -> ()) ~msg:"load answers Error" (G.load t [| d |])

(* The ints and the run's work *)

(* A loop of two trips stores its trip in int 0, and each trip's launch copies
   that word's low byte, read through an Ints slot, to byte [trip] of the input:
   the launch's offset is a hole over the trip. The gate holds the device's
   work, so trip 1's store waits for trip 0's work. *)
let test_trip_follows_work () =
  let d, pd = polled "trip-order" in
  let copy_byte =
    (Launch
          {
            image = 0;
            name = "copy";
            params =
              {
                bytes = le64 0 ^ le64 0 ^ le64 1;
                holes = [| hole 8 (Int 0 : value) |];
              };
            refs =
              [|
                { G.slot = 0; at = 0; use = Read };
                { G.slot = 1; at = 8; use = Write };
              |];
            groups = (Fixed 1, Fixed 1, Fixed 1);
            threads = (Fixed 1, Fixed 1, Fixed 1);
            shared = Fixed 0;
          })
  in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 1;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 8 } |];
      steps =
        [|
          Loop
            {
              trips = Fixed 2;
              trip = Some 0;
              flag = None;
              body =
                [|
                  submit 0
                    ~buffers:[| (Ints, Rig.Read); (Input 0, Rig.Read_write) |]
                    [| copy_byte |];
                |];
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let out = of_bytes d (String.make 8 '\255') in
  P.gate pd;
  let opener =
    Domain.spawn (fun () ->
        Rig_support.await "a sleep at the gate" (fun () -> P.sleepers pd > 0);
        P.open_gate pd)
  in
  ignore (G.run p { inputs = [| out |]; ints = [||] });
  let bytes = read out in
  Domain.join opener;
  equal ~msg:"each trip's work read its own trip" string "\000\001"
    (String.sub bytes 0 2)

(* A launch writes int 0 through an Ints slot, and a later launch's hole over
   int 0 reads what it wrote. *)
let test_int_follows_writer () =
  let d, _ = polled "int-order" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 1;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 8 } |];
      steps =
        [|
          submit 0
            ~buffers:[| (Ints, Rig.Read_write) |]
            [| fill ~image:0 ~groups:1 42 |];
          submit 0
            ~buffers:[| (Input 0, Rig.Read_write) |]
            [| fill ~image:0 ~groups:1 0 ~holes:[| hole 8 (Int 0 : value) |] |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  let out = one_word d in
  ignore (G.run p { inputs = [| out |]; ints = [| 0 |] });
  equal ~msg:"the word the first launch wrote" int 42 (word_of out)

(* A launch's refs bind the input's address anew for every run. *)
let test_input_ref () =
  let d, _ = polled "input-ref" in
  let f = fill ~image:0 ~groups:1 0 in
  let f =
    match f with
    | Launch l ->
        G.Launch
          {
            l with
            refs =
              [|
                { G.slot = 0; at = 0; use = Write };
                { G.slot = 0; at = 8; use = Write };
              |];
          }
    | _ -> assert false
  in
  let t =
    {
      (base d) with
      steps = [| submit 0 ~buffers:[| (Input 0, Rig.Read_write) |] [| f |] |];
    }
  in
  let p = load_ok t [ d ] in
  List.iter
    (fun x ->
      let keeper = Sub.make ~fixed:[ (x, Rig.Read) ] d [||] in
      let expected = Sub.address keeper x in
      ignore (G.run p { inputs = [| x |]; ints = [||] });
      equal ~msg:"the stored address" int expected (word_of x))
    [ B.create d 64; B.create d 64 ]

(* Fill parameters share launch parameters' frame and ref semantics. *)
let test_fill_params () =
  let d = Rig.memory_device "program-fill-params" in
  let t =
    {
      devices = [| Rig.arch d; Rig.arch Rig.host |];
      memory = [||];
      images = [| host_image ~dir:"fixtures" 1 "fill" |];
      inputs = [| { device = 0; bytes = 8 } |];
      ints = 1;
      steps =
        [|
          submit 0
            ~buffers:[| (Input 0, Rig.Read_write) |]
            [|
              (G.Fill
                    {
                      fill = Entry { image = 0; name = "fill" };
                      params =
                        {
                          bytes = String.make 16 '\000';
                          holes = [| hole 8 (Int 0) |];
                        };
                      refs = [| { G.slot = 0; at = 0; use = Write } |];
                    });
            |];
        |];
    }
  in
  let p = load_ok t [ d; Rig.host ] in
  List.iter
    (fun n ->
      let out = B.create d 8 in
      Array.iter Rig.Point.wait (G.run p { inputs = [| out |]; ints = [| n |] });
      equal ~msg:"the run's parameter at its ref" int n (word_of out))
    [ 42; 99 ]

let test_fill_empty () =
  let d = Rig.memory_device "program-fill-empty" in
  let t =
    {
      devices = [| Rig.arch d; Rig.arch Rig.host |];
      memory = [||];
      images = [| host_image ~dir:"fixtures" 1 "fill" |];
      inputs = [||];
      ints = 0;
      steps =
        [|
          submit 0
            [|
              (G.Fill { fill = Entry { image = 0; name = "fill_empty" }; params = data ""; refs = [||] });
            |];
        |];
    }
  in
  let points = G.run (load_ok t [ d; Rig.host ]) { inputs = [||]; ints = [||] } in
  equal ~msg:"one submission" int 1 (Array.length points);
  Array.iter Rig.Point.wait points

(* Every run reaches its own copy through the address initialized after the
   fixing submission was made. The init points through another container. *)
let test_init_address () =
  let d = Rig.memory_device "program-init-address" in
  let v memory = { memory; offset = 0; length = 8 } in
  let t =
    {
      devices = [| Rig.arch d; Rig.arch Rig.host |];
      memory =
        [|
          alloc ~copies:Two 0 8; alloc ~copies:Two ~init:(holed 8 (addr 0)) 0 8;
        |];
      images = [| host_image ~dir:"fixtures" 1 "fill" |];
      inputs = [| { device = 0; bytes = 8 } |];
      ints = 1;
      steps =
        [|
          submit 0
            ~buffers:[| (Memory (v 1), Rig.Read) |]
            ~fixed:[| (v 0, Rig.Read_write) |]
            [|
              (G.Fill
                    {
                      fill = Entry { image = 0; name = "fill_indirect" };
                      params =
                        {
                          bytes = String.make 16 '\000';
                          holes = [| hole 8 (Int 0) |];
                        };
                      refs = [| { G.slot = 0; at = 0; use = Write } |];
                    });
            |];
          G.Move { src = Memory (v 0); dst = Input 0 };
        |];
    }
  in
  let p = load_ok t [ d; Rig.host ] in
  let out = B.create d 8 in
  List.iter
    (fun n ->
      ignore (G.run p { inputs = [| out |]; ints = [| n |] });
      equal ~msg:"the initialized address reaches this run's copy" int n
        (word_of out))
    [ 42; 99; 123 ]

(* The submission keeps both its fill and the host function whose address its
   initialized memory holds after the loaded program leaves scope. *)
let test_init_code_kept () =
  let d, pd = polled "init-code-kept" in
  let out = one_word d in
  let[@inline never] submit () =
    let t =
      {
        G.devices = [| Rig.arch d; Rig.arch Rig.host |];
        memory =
          [|
            alloc ~init:(holed 8 (Entry { image = 1; name = "affine" })) 0 8;
          |];
        images =
          [| host_image ~dir:"fixtures" 1 "fill"; host_image 1 "affine" |];
        inputs = [| { device = 0; bytes = 8 } |];
        ints = 0;
        steps =
          [|
            submit 0
              ~buffers:
                [|
                  (Memory { memory = 0; offset = 0; length = 8 }, Rig.Read);
                  (Input 0, Rig.Read_write);
                |]
              [|
                (G.Fill
                      {
                        fill = Entry { image = 0; name = "fill_call" };
                        params = data (String.make 16 '\000');
                        refs =
                          [|
                            { G.slot = 0; at = 0; use = Read }; { G.slot = 1; at = 8; use = Write };
                          |];
                      });
              |];
          |];
      }
    in
    G.run (load_ok t [ d; Rig.host ]) { inputs = [| out |]; ints = [||] }
  in
  let points = submit () in
  equal ~msg:"work is held before collection" int 1 (P.queued pd);
  for _ = 1 to 3 do
    Gc.full_major ()
  done;
  Array.iter Rig.Point.wait points;
  equal ~msg:"the initialized entry ran after collection" int 1 (word_of out)

(* What run reads of the caller's *)

(* run reads its frame once, before its first step: a launch writes input 0, a
   move copies it into input 1 and waits for the launch at the gate, while
   another domain puts a buffer of no bytes in the frame as input 2. The run
   goes on with the buffer the frame held, which its last launch writes. *)
let test_frame_read_once () =
  let d, pd = polled "read-once" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 0;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = Iarray.init 3 (fun _ -> { G.device = 0; bytes = 8 });
      steps =
        [|
          submit 0
            ~buffers:[| (Input 0, Rig.Read_write) |]
            [| fill ~image:0 ~groups:1 42 |];
          G.Move { src = Input 0; dst = Input 1 };
          submit 0
            ~buffers:[| (Input 2, Rig.Read_write) |]
            [| fill ~image:0 ~groups:1 7 |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  let frame =
    { G.inputs = [| one_word d; one_word d; one_word d |]; ints = [||] }
  in
  let z = frame.inputs.(2) in
  P.gate pd;
  let changer =
    Domain.spawn (fun () ->
        Rig_support.await "the move at the gate" (fun () -> P.sleepers pd > 0);
        frame.inputs.(2) <- B.create d 0;
        P.open_gate pd)
  in
  ignore (G.run p frame);
  Domain.join changer;
  equal ~msg:"the move ran" int 42 (word_of frame.inputs.(1));
  equal ~msg:"the frame's earlier input 2" int 7 (word_of z)

(* A Copy part copies its source slot into its destination slot on its step's
   device, whether each holds the program's memory or a run's input: memory
   into an input, that input into another, the other into memory, and memory
   into memory, with each run's inputs. A move takes the last out. *)
let test_copy_part () =
  let d, _ = polled "copy-part" in
  let input = { G.device = 0; bytes = 8 } in
  let view memory offset = G.Memory { memory; offset; length = 8 } in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 0;
      memory =
        [| alloc ~init:(data (le64 42 ^ le64 7)) 0 16; alloc 0 16 |];
      images = [||];
      inputs = [| input; input; input |];
      steps =
        [|
          submit 0
            ~buffers:
              [|
                (view 0 8, Rig.Read);
                (Input 0, Rig.Read_write);
                (Input 1, Rig.Read_write);
                (view 1 0, Rig.Read_write);
                (view 1 8, Rig.Read_write);
              |]
            [|
              pcopy ~src:0 ~dst:1 8;
              pcopy ~src:1 ~dst:2 8;
              pcopy ~src:2 ~dst:3 8;
              pcopy ~src:3 ~dst:4 8;
            |];
          G.Move
            {
              src = Memory { memory = 1; offset = 8; length = 8 };
              dst = Input 2;
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  for run = 1 to 2 do
    let a = one_word d and b = one_word d and c = one_word d in
    Array.iter Rig.Point.wait (G.run p { inputs = [| a; b; c |]; ints = [||] });
    List.iteri
      (fun i x ->
        equal ~msg:(Printf.sprintf "run %d's input %d" run i) int 7 (word_of x))
      [ a; b; c ]
  done

(* A copy part in a loop moves with its trips: trip [i] copies row [i] of
   memory into row [3 - i] of the input, so the input holds the rows
   reversed. *)
let test_copy_loop () =
  let d, _ = polled "copy-loop" in
  let rows = [ 10; 11; 12; 13 ] in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 1;
      memory =
        [| alloc ~init:(data (String.concat "" (List.map le64 rows))) 0 32 |];
      images = [||];
      inputs = [| { device = 0; bytes = 32 } |];
      steps =
        [|
          Loop
            {
              trips = Fixed 4;
              trip = Some 0;
              flag = None;
              body =
                [|
                  submit 0
                    ~buffers:
                      [|
                        ( Memory { memory = 0; offset = 0; length = 32 },
                          Rig.Read );
                        (Input 0, Rig.Read_write);
                      |]
                    [|
                      pcopy
                        ~src_offset:[| (Int 0, 8) |]
                        ~dst_offset:[| (Int 0, -8); (Fixed 24, 1) |]
                        8;
                    |];
                |];
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let x = of_bytes d (String.make 32 '\000') in
  ignore (G.run p { inputs = [| x |]; ints = [||] });
  equal ~msg:"the rows reversed" string
    (String.concat "" (List.map le64 (List.rev rows)))
    (read x)

(* A copy's end that a run's ints put outside its slot, or whose offset is past
   an int, is refused at that step: [load] cannot know where it lies. *)
let test_copy_outside () =
  let d, _ = polled "copy-outside" in
  let load copy =
    load_ok
      {
        (base d) with
        ints = 1;
        steps =
          [|
            submit 0
              ~buffers:[| (all 0, Rig.Read); (Input 0, Rig.Read_write) |]
              [| copy |];
          |];
      }
      [ d ]
  in
  let input = of_bytes d (String.make 64 '\000') in
  let run p v = G.run p { inputs = [| input |]; ints = [| v |] } in
  let src = load (pcopy ~src_offset:[| (Int 0, 1) |] 8)
  and dst = load (pcopy ~dst_offset:[| (Int 0, 1) |] 8)
  and twice = load (pcopy ~src_offset:[| (Int 0, 2) |] 8) in
  ignore (run src 56);
  ignore (run dst 56);
  raises_match ~msg:"a source from byte 60 of 64" Exn.invalid_arg (fun () ->
      run src 60);
  raises_match ~msg:"a destination from byte 60 of 64" Exn.invalid_arg
    (fun () -> run dst 60);
  raises_match ~msg:"a negative offset" Exn.invalid_arg (fun () ->
      run src (-8));
  raises_match ~msg:"an offset past an int" Exn.invalid_arg (fun () ->
      run twice max_int)

(* A Submit step of more parts than its device's most parts loads and runs
   where the device's queues have room for it: three fills of one word each,
   on a device that always runs two. *)
let test_past_most_parts () =
  let d, _ =
    P.open_ ~capacity:4 ~most_parts:2
      (strf "program:past-most-parts-%d" (Atomic.fetch_and_add opened 1))
  in
  let t =
    {
      (base d) with
      steps =
        [|
          submit 0
            ~buffers:[| (Input 0, Rig.Read_write) |]
            [|
              fill ~image:0 ~groups:1 1;
              fill ~image:0 ~groups:1 2;
              fill ~image:0 ~groups:1 3;
            |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  let x = of_bytes d (String.make 64 '\000') in
  Array.iter Rig.Point.wait (G.run p { inputs = [| x |]; ints = [||] });
  equal ~msg:"the last fill's word" int 3 (word_of x)

(* A copy within the ints, both ends at offsets the run reads from them: word
   1 into word 3. *)
let test_copy_ints () =
  let d, _ = polled "copy-ints" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 4;
      memory = [||];
      images = [||];
      inputs = [| { G.device = 0; bytes = 32 } |];
      steps =
        [|
          submit 0
            ~buffers:[| (Ints, Rig.Read); (Ints, Rig.Read_write) |]
            [|
              pcopy
                ~src_offset:[| (Int 0, 1) |]
                ~dst_offset:[| (Int 1, 1) |]
                8;
            |];
          G.Move { src = Ints; dst = Input 0 };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let x = of_bytes d (String.make 32 '\000') in
  ignore (G.run p { inputs = [| x |]; ints = [| 8; 24; 42 |] });
  equal ~msg:"the ints" string
    (le64 8 ^ le64 24 ^ le64 42 ^ le64 24)
    (read x)

(* A memory slot that only copies name takes no slot of the step's
   submission, and the slots after it are numbered without it: memory 0, named
   by the copy alone, reaches memory 1, which the copy writes and a launch
   reads into input 0, named by the launch's write ref between them. *)
let test_renumbered () =
  let d, _ = polled "renumbered" in
  let bytes = String.init 64 (fun i -> Char.chr (i + 1)) in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 0;
      memory = [| alloc ~init:(data bytes) 0 64; alloc 0 64 |];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 64 } |];
      steps =
        [|
          submit 0
            ~buffers:
              [|
                (all 0, Rig.Read);
                (Input 0, Rig.Read_write);
                (all 1, Rig.Read_write);
              |]
            [| pcopy ~dst:2 64; kcopy ~src:2 ~dst:1 ~image:0 64 |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  let x = B.create d 64 in
  Array.iter Rig.Point.wait (G.run p { inputs = [| x |]; ints = [||] });
  equal ~msg:"input 0" string bytes (read x)

(* Run 2 waits for run 0's work on its copy before it writes its ints: the
   frame's ints changed during that wait change nothing. *)
let test_frame_ints_read_once () =
  let d, pd = polled "ints-once" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 1;
      memory = [| alloc ~copies:Two 0 64 |];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 8 } |];
      steps =
        [|
          submit 0
            ~buffers:[| (all 0, Rig.Read_write) |]
            [| fill ~image:0 ~groups:1 7 |];
          submit 0
            ~buffers:[| (Input 0, Rig.Read_write) |]
            [| fill ~image:0 ~groups:1 0 ~holes:[| hole 8 (Int 0 : value) |] |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  for _ = 1 to 2 do
    ignore (G.run p { inputs = [| one_word d |]; ints = [| 1 |] })
  done;
  let frame = { G.inputs = [| one_word d |]; ints = [| 5 |] } in
  P.gate pd;
  let changer =
    Domain.spawn (fun () ->
        Rig_support.await "run 2 at the gate" (fun () -> P.sleepers pd > 0);
        frame.ints.(0) <- 99;
        P.open_gate pd)
  in
  let last = G.run p frame in
  Domain.join changer;
  Array.iter Rig.Point.wait last;
  equal ~msg:"the frame's int when run was called" int 5
    (word_of frame.inputs.(0))

(* A loaded program that is unreachable returns its memory and unloads its
   images once the work of its last run is done. *)
let test_collected () =
  let d, pd = polled "collected" in
  let count call = List.length (List.filter (( = ) call) (P.log pd)) in
  (* Collects, then drains [d] as an allocation does, for a fixed number of
     rounds: the program's collection frees its images a round later. *)
  let drain () =
    for _ = 1 to 3 do
      Gc.full_major ();
      ignore (B.create ~memory:Pinned d 8)
    done
  in
  let before = P.allocated pd Device in
  let[@inline never] run_dropped () =
    let p = load_ok (one_device_fill ~copies:Two d) [ d ] in
    ignore (G.run p { inputs = [||]; ints = [||] })
  in
  run_dropped ();
  drain ();
  Rig.free_cache d;
  equal ~msg:"images while the run's work is unrun" int 0 (count "unload");
  equal ~msg:"memory while the run's work is unrun" int (before + 128)
    (P.allocated pd Device);
  ignore (P.run pd);
  drain ();
  Rig.free_cache d;
  equal ~msg:"images once the work ran" int 1 (count "unload");
  equal ~msg:"memory once the work ran" int before (P.allocated pd Device)

(* Runs of one loaded program, from two domains, each read the word the run
   before it wrote: a run copies its memory to its input, then writes its int
   there. *)
let swap_program d =
  let word = { G.memory = 0; offset = 0; length = 8 } in
  {
    G.devices = [| Rig.arch d |];
    ints = 1;
    memory = [| alloc ~init:(data (le64 0)) 0 8 |];
    images = [| { device = 0; binary = functions } |];
    inputs = [| { device = 0; bytes = 8 } |];
    steps =
      [|
        submit 0
          ~buffers:
            [| (Memory word, Rig.Read_write); (Input 0, Rig.Read_write) |]
          [| kcopy ~image:0 8 |];
        submit 0
          ~buffers:[| (Memory word, Rig.Read_write) |]
          [| fill ~image:0 ~groups:1 0 ~holes:[| hole 8 (Int 0 : value) |] |];
      |];
  }

let swap_device = lazy (fst (polled "swap"))
let swapping = abstract "p"

let swap_commands =
  [
    command "load"
      (Gen.unit @-> makes swapping)
      (fun () -> ref 0)
      (fun () ->
        let d = Lazy.force swap_device in
        load_ok (swap_program d) [ d ]);
    command "run"
      (swapping ^-> Gen.int_range 1 1000 @-> returns int)
      (fun last v ->
        let w = !last in
        last := v;
        w)
      (fun p v ->
        let out = one_word (Lazy.force swap_device) in
        Array.iter Rig.Point.wait
          (G.run p { inputs = [| out |]; ints = [| v |] });
        word_of out);
  ]

(* Loading on another machine's devices goes through its agent, which takes no
   rails from this process. *)
let test_far_rails () =
  let far =
    Rig_support.machine
      (strf "program-rails-%d" (Atomic.fetch_and_add opened 1))
  in
  let t = { (base far) with devices = [| Rig.arch far |] } in
  raises_match Exn.invalid_arg (fun () ->
      G.load ~rails:(fun _ -> None) t [| far |]);
  is_error ~pp:(fun _ _ -> ()) ~msg:"without rails: no agent's device"
    (G.load t [| far |])

(* run checks its frame before its first step: with its last input too small, it
   raises having submitted nothing. *)
let test_frame_checked_first () =
  let d, _ = polled "checked-first" in
  let t =
    {
      (base d) with
      inputs = Iarray.init 2 (fun _ -> { G.device = 0; bytes = 8 });
      steps =
        [|
          submit 0
            ~buffers:[| (Input 0, Rig.Read_write) |]
            [| fill ~image:0 ~groups:1 1 |];
          submit 0
            ~buffers:[| (Input 1, Rig.Read_write) |]
            [| fill ~image:0 ~groups:1 2 |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  let inputs = [| one_word d; B.create d 7 |] in
  let submitted = Rig.submitted d in
  raises_match ~msg:"input 1 of 7 bytes" Exn.invalid_arg (fun () ->
      G.run p { inputs; ints = [||] });
  equal ~msg:"submissions" int submitted (Rig.submitted d)

(* test/fixtures' [waiting], image 0 on the host (device 1): it sets word 1 of
   its first parameter's memory, then spins until word 0 is set, at most as
   many rounds as its second parameter. Its later parameters are refs to the
   other slots, which it leaves alone. *)
let waiting_step ?(groups : G.value = Fixed 1) slots rounds =
  let n = Iarray.length slots in
  submit 1
    ~buffers:(Iarray.map (fun s -> (s, Rig.Read_write)) slots)
    [|
      Launch
        {
          image = 0;
          name = "waiting";
          params =
            { bytes = String.make (8 * (n + 1)) '\000'; holes = [| hole 8 rounds |] };
          refs =
            Iarray.init n (fun slot ->
                { G.slot; at = (if slot = 0 then 0 else 8 * (slot + 1)); use = Write });
          groups = (groups, Fixed 1, Fixed 1);
          threads = (Fixed 1, Fixed 1, Fixed 1);
          shared = Fixed 0;
        };
    |]

let host_program ?groups d slots rounds =
  {
    G.devices = [| Rig.arch d; Rig.arch Rig.host |];
    ints = 1;
    memory = [||];
    images = [| host_image 1 "waiting" |];
    inputs = Iarray.init 3 (fun _ -> { G.device = 0; bytes = 16 });
    steps = [| waiting_step ?groups slots rounds |];
  }

let host_frame d =
  {
    G.inputs = Array.init 3 (fun _ -> of_bytes d (String.make 16 '\000'));
    ints = [| 0 |];
  }

let check_donation expected b =
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun claims ->
      equal ~msg:"the buffer can be donated" bool expected
        (Rig.Claim.exclusive claims b))

(* The launch names the same buffer twice; both claims must end together, after
   the linked code returns. *)
let test_host_claim_call () =
  let d, _ = polled "host-claims-call" in
  let t = host_program d [| Input 0; Input 0 |] (Fixed 20_000_000_000) in
  let p = load_ok t [ d; Rig.host ] and frame = host_frame d in
  let x = frame.inputs.(0) in
  let at = Rig_support.at x in
  let runner = Domain.spawn (fun () -> ignore (G.run p frame)) in
  Fun.protect
    ~finally:(fun () ->
      Rig_support.store at 1;
      Domain.join runner)
    (fun () ->
      Rig_support.await "the code running" (fun () ->
          Rig_support.load (at + 8) = 1);
      check_donation false x);
  check_donation true x

(* The claim starts before the host waits for the preceding device write. *)
let test_host_claim_wait () =
  let d, pd = polled "host-claims-wait" in
  let t = host_program d [| Input 0 |] (Fixed 0) in
  let t =
    {
      t with
      images = Iarray.append t.images [| { device = 0; binary = functions } |];
      steps =
        Iarray.append
          [|
            submit 0
              ~buffers:[| (Input 0, Rig.Read_write) |]
              [| fill ~image:1 ~groups:1 1 |];
          |]
          t.steps;
    }
  in
  let p = load_ok t [ d; Rig.host ] and frame = host_frame d in
  P.gate pd;
  let runner = Domain.spawn (fun () -> ignore (G.run p frame)) in
  Fun.protect
    ~finally:(fun () ->
      P.open_gate pd;
      Domain.join runner)
    (fun () ->
      Rig_support.await "the host launch waiting" (fun () -> P.sleepers pd > 0);
      check_donation false frame.inputs.(0));
  check_donation true frame.inputs.(0)

(* A later buffer can refuse its claim after the earlier ones were taken. *)
let test_host_claim_refusal index =
  let d, _ = polled "host-claims-refusal" in
  let t = host_program d (Iarray.init 3 (fun i -> G.Input i)) (Fixed 0) in
  let p = load_ok t [ d; Rig.host ] and frame = host_frame d in
  Rig.Claim.with_ ~read:[]
    ~donate:[ frame.inputs.(index) ]
    (fun claims ->
      equal bool true (Rig.Claim.exclusive claims frame.inputs.(index));
      raises_match Exn.invalid_arg (fun () -> G.run p frame));
  Array.iter (check_donation true) frame.inputs;
  ignore (G.run p frame);
  Array.iter (check_donation true) frame.inputs

(* A launch the host never runs, of no group, releases its claims. *)
let test_host_claim_grid_refusal () =
  let d, _ = polled "host-claims-grid" in
  let t = host_program ~groups:(Int 0) d [| Input 0 |] (Fixed 0) in
  let p = load_ok t [ d; Rig.host ] and frame = host_frame d in
  frame.ints.(0) <- 0;
  raises_match Exn.invalid_arg (fun () -> G.run p frame);
  check_donation true frame.inputs.(0);
  frame.ints.(0) <- 1;
  ignore (G.run p frame);
  check_donation true frame.inputs.(0)

(* run keeps its inputs until it returns: while a host launch runs on input 0,
   the frame holds another buffer there and the input is collected. Its memory
   is not reused before the launch returns. *)
let test_host_input_kept () =
  let d, _ = polled "host-kept" in
  let t =
    {
      (host_program d [| Input 0 |] (Fixed 20_000_000_000)) with
      inputs = [| { device = 0; bytes = 16 } |];
      ints = 0;
    }
  in
  let p = load_ok t [ d; Rig.host ] in
  let at, frame =
    (fun () ->
      let x = B.create d 16 in
      let at = Rig_support.at x in
      Rig_support.store at 0;
      Rig_support.store (at + 8) 0;
      (at, { G.inputs = [| x |]; ints = [||] }))
      ()
  in
  let runner = Domain.spawn (fun () -> ignore (G.run p frame)) in
  Rig_support.await "the code running" (fun () -> Rig_support.load (at + 8) = 1);
  frame.inputs.(0) <- B.create d 16;
  Gc.full_major ();
  Gc.full_major ();
  let again = B.create d 16 in
  let reused = Rig_support.at again = at in
  Rig_support.store at 1;
  Domain.join runner;
  equal ~msg:"the input's memory reused while the code ran" bool false reused

(* An [Entry] leaf names a function of a host image that only the loaded
   program holds, which run keeps while [calling] calls it. An unmapped callee
   faults. *)
let test_entry_leaf_kept () =
  let d, _ = polled "entry-kept" in
  let t =
    {
      G.devices = [| Rig.arch d; Rig.arch Rig.host |];
      ints = 0;
      memory = [||];
      images = [| host_image 1 "waiting"; host_image 1 "calling" |];
      inputs = [| { device = 0; bytes = 16 } |];
      steps =
        [|
          submit 1
            ~buffers:[| (Input 0, Rig.Read_write) |]
            [|
              Launch
                {
                  image = 1;
                  name = "calling";
                  params =
                    {
                      bytes = String.make 24 '\000';
                      holes =
                        [|
                          hole 0 (Leaf (Entry { image = 0; name = "waiting" }));
                          hole 16 (Fixed 20_000_000_000);
                        |];
                    };
                  refs = [| { G.slot = 0; at = 8; use = Write } |];
                  groups = (Fixed 1, Fixed 1, Fixed 1);
                  threads = (Fixed 1, Fixed 1, Fixed 1);
                  shared = Fixed 0;
                };
            |];
        |];
    }
  in
  let x = B.create d 16 in
  let at = Rig_support.at x in
  Rig_support.store at 0;
  Rig_support.store (at + 8) 0;
  let runner =
    Domain.spawn (fun () ->
        ignore
          (G.run (load_ok t [ d; Rig.host ]) { inputs = [| x |]; ints = [||] }))
  in
  Rig_support.await "the callee running" (fun () ->
      Rig_support.load (at + 8) = 1);
  for _ = 1 to 3 do
    Gc.full_major ()
  done;
  Rig_support.store at 1;
  Domain.join runner

let tests =
  [
    group ~timeout "steps"
      [
        prop
          "a run leaves the bytes of its steps run in order, on any device, \
           copies at the offsets they state, memory of two copies alternating \
           by run, and answers one point per device that ran"
          gen_case model_law;
      ];
    group ~timeout "holes"
      [
        prop
          "a hole holds the low bits of its leaf plus add, shifted, in memory \
           and in a launch"
          gen_hole hole_law;
        prop
          "a hole holds its terms' values times their factors, plus add, \
           shifted, read on each run"
          gen_affine affine_law;
        prop
          "a value that meets set bits of its bytes is refused, and the bits \
           outside each value keep the description's bytes"
          gen_beside beside_law;
      ];
    group ~timeout "bytes"
      [
        prop "of_string reads back what to_string wrote" gen_t round_trip;
        prop "of_string answers a cut or a changed byte without raising"
          gen_damage damaged;
        test "of_string names another version" test_version;
        test "of_string says where a cut version lies" test_cut_version;
      ];
    group ~timeout "host code"
      [
        prop
          "host code computes ints from the frame's, which a launch's geometry \
           reads"
          gen_ints ints_law;
        prop
          "a loop runs its body as many times as its trips, none for a \
           negative count, each trip's index in the ints"
          (Gen.int_range (-2) 6) loop_law;
        test "a loop stops at a flag a step cleared" test_flag;
        test "run raises for a loop's flag past its input" test_flag_outside;
        test "a host launch calls a function an Entry leaf names"
          test_entry_leaf;
        test "run keeps the host function an Entry leaf names"
          test_entry_leaf_kept;
      ];
    group ~timeout "rails"
      [
        test "host code fills a rail's area and calls its ready function"
          test_rail;
        test "load answers Error for a rail this machine has not" test_no_rail;
      ];
    group ~timeout "order"
      [
        test "run n of memory of two copies follows run n - 2 on its copy"
          test_two_waits;
        test "memory of one copy waits for nothing on the host"
          test_one_waits_not;
        test "a run's first submission follows its after points" test_after;
        test "a loop's trip is stored once the work that reads the ints is done"
          test_trip_follows_work;
        test "an Int is read once the work that writes the ints is done"
          test_int_follows_writer;
        stateful ~domains:2 "runs of one program from two domains serialize"
          swap_commands;
      ];
    group ~timeout "life"
      [
        test
          "an unreachable program returns its memory and images after its \
           work"
          test_collected;
      ];
    group ~timeout "refusals"
      [
        test "every address and handle names memory kept by its step"
          memory_hole_refusals;
        cases
          ~name:(Format.asprintf "%a" pp_extreme)
          "load answers Error for integers whose sums would wrap" extremes
          extreme_law;
        cases
          ~name:(fun (n, _, _) -> n)
          "load answers Error for" (refusals @ host_refusals) test_refusal;
        test "load answers Error for memory a device cannot borrow"
          test_unborrowable;
        test
          "load answers Error for host code over memory the host does not \
           address"
          test_host_unaddressed;
        test "load answers Error for an Address of memory named by handle only"
          test_handle_only;
        test "load raises for devices of several machines" test_machines;
        test "load raises for rails given with another machine's devices"
          test_far_rails;
        test "run raises for a frame that does not fit" test_frame_refusals;
        test "run raises for more ints than the program's" test_ints_refusal;
        test "run raises for a read-only input a step writes"
          test_read_only_input;
        test "run raises for a read-only input a copy part writes"
          test_read_only_copy;
      ];
    group ~timeout "frame"
      [
        test "run reads its frame once, before its first step"
          test_frame_read_once;
        test "run reads its frame's ints once, before it waits for run n - 2"
          test_frame_ints_read_once;
        test "run raises for a frame it refuses before its first step"
          test_frame_checked_first;
        test "run keeps its inputs while a host launch runs"
          test_host_input_kept;
        test "a host launch claims its buffers while it runs"
          test_host_claim_call;
        test "a host launch claims its buffers before it waits"
          test_host_claim_wait;
        cases
          ~name:(Printf.sprintf "input %d")
          "a host launch releases earlier claims after a claim refusal"
          [ 0; 1; 2 ] test_host_claim_refusal;
        test "a host launch of no group releases its claims"
          test_host_claim_grid_refusal;
        test "a launch's ref is its input's address in each run" test_input_ref;
        test "a fill reads each run's parameters and refs" test_fill_params;
        test "a fill with no parameters submits" test_fill_empty;
        test "a Copy part copies between its step's slots" test_copy_part;
        test "a Copy part in a loop moves with its trips" test_copy_loop;
        test "run raises for a copy its ints put outside its slot"
          test_copy_outside;
        test "a Copy part copies within the ints at offsets they hold"
          test_copy_ints;
        test "a step of more parts than its device's most parts runs"
          test_past_most_parts;
        test "memory only copies name takes no slot of the submission"
          test_renumbered;
        test "an init Address comes from a step fixing its run's copy"
          test_init_address;
        test "an init Entry keeps its host function through queued work"
          test_init_code_kept;
      ];
  ]

let () = exit (Windtrap.run "rig.program" tests)
