(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Range gathers through every kernel library the host runs: each is the
   library's gather at the positions the contract's sentence gives, bit for
   bit, over every dtype and view; nx.cpu gives one thread's bits on many;
   and each library refuses what does not fit. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module Support = Nx_kernels_support
open Elements
open Index_cases

(* Cases *)

(* How a row's runs are drawn: ends that sum lengths, any ends, or ends
   near int64's least before starts near its greatest. *)
type mode = Packed | Wild | Wrap

type case = {
  size : int;
  starts : A.any;
  ends : A.any;
  x : A.any;
  mode : mode;
  views : string list;
}

let pp_case ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf "ranges of %d from %a %a by %a, %s: starts %a ends %a (%s)"
    c.size D.pp (A.dtype x) pp_ints (shape_of c.x) pp_ints (shape_of c.starts)
    (match c.mode with Packed -> "packed" | Wild -> "wild" | Wrap -> "wrap")
    (Format.pp_print_list ~pp_sep:Format.pp_print_space
       (fun ppf v -> Format.fprintf ppf "%Ld" v))
    (Array.to_list (positions c.starts))
    (Format.pp_print_list ~pp_sep:Format.pp_print_space
       (fun ppf v -> Format.fprintf ppf "%Ld" v))
    (Array.to_list (positions c.ends))
    (String.concat ", " c.views)

(* A start: inside the axis, at its edges and past them, a little before
   it so that a run straddles 0, and int64's extremes. *)
let start rs n =
  match Random.State.int rs 10 with
  | 0 -> -1L
  | 1 -> Int64.of_int n
  | 2 -> Int64.of_int (n - 1)
  | 3 -> Int64.of_int (-2)
  | 4 -> Int64.min_int
  | 5 -> Int64.max_int
  | _ -> Int64.of_int (Random.State.int rs (n + 1))

(* A packed run's length: -1, 0, 1, a few, the axis and more, and
   int64's greatest. *)
let length rs n =
  match Random.State.int rs 12 with
  | 0 -> -1L
  | 1 -> 0L
  | 2 -> 1L
  | 3 -> Int64.max_int
  | _ -> Int64.of_int (Random.State.int rs (n + 3))

(* Any end. *)
let wild_end rs =
  match Random.State.int rs 8 with
  | 0 -> Int64.min_int
  | 1 -> Int64.max_int
  | 2 -> -1L
  | 3 -> 0L
  | _ -> Int64.of_int (Random.State.int rs 14)

(* The greatest of [es] and 0. *)
let greatest es =
  Array.fold_left (fun m e -> if Int64.compare e m > 0 then e else m) 0L es

let case_gen =
  let open Gen in
  let sub = List.filter (fun (D.Any dt) -> D.bits dt < 8) all_dtypes in
  (* A quarter of the cases are rows of one byte, as text's are, over a
     plain source, their runs mostly packed: the copies of nx.cpu's line
     path. *)
  let* line = frequency [ (3, constant false); (1, constant true) ] in
  let* d =
    if line then constant (D.Any D.Uint8)
    else
      frequency
        [ (3, of_list ~pp:pp_dtype all_dtypes); (1, of_list ~pp:pp_dtype sub) ]
  in
  let* lead =
    frequency
      [
        (2, constant [||]);
        (2, map (fun e -> [| e |]) (int_range 1 3));
        (1, array ~size:(constant 2) (int_range 0 2));
      ]
  in
  let* k =
    frequency
      [
        (2, constant 0);
        (6, int_range 1 6);
        (1, int_range 7 40);
        (1, int_range 4097 9000);
      ]
  in
  let* n =
    frequency [ (1, constant 0); (6, int_range 1 9); (1, int_range 60 300) ]
  in
  let* cells =
    if line then constant [||]
    else
      of_list ~pp:pp_ints
        [ [||]; [||]; [| 3 |]; [| 2; 0 |]; [| 2; 3 |]; [| 5 |] ]
  in
  let* mode =
    let packed = if line then 12 else 4 in
    frequency
      [ (packed, constant Packed); (2, constant Wild); (3, constant Wrap) ]
  in
  let* pick = int_range 0 4 in
  let a = Array.length lead in
  let xs = Array.concat [ lead; [| n |]; cells ] in
  let rx = Array.length xs in
  let flags r odds =
    array ~size:(constant r)
      (frequency [ (odds, constant false); (1, constant true) ])
  in
  let marked b =
    List.filter (fun i -> b.(i)) (List.init (Array.length b) Fun.id)
  in
  let* bx = flags rx 5 in
  let* xv =
    if line then constant (plain rx) else view_of ~broadcast:(marked bx) rx
  in
  let* bs = flags a 3 in
  let* sv = view_of ~broadcast:(marked bs) (a + 1) in
  let+ seed = int in
  let rs = Random.State.make [| seed |] in
  let ss = Array.append lead [| k |] in
  let rows = total lead in
  (* Ends are C-contiguous, so that a row's draws follow its runs: running
     sums, any values, or, in [Wrap], ends near int64's least in even runs
     and small ones in odd runs. *)
  let es = Array.make (rows * k) 0L in
  for r = 0 to rows - 1 do
    let last = ref 0L in
    for j = 0 to k - 1 do
      let e =
        match mode with
        | Packed -> Int64.add !last (length rs n)
        | Wild -> wild_end rs
        | Wrap ->
            if j mod 2 = 0 then
              Int64.add Int64.min_int (Int64.of_int (Random.State.int rs 3))
            else Int64.of_int (3 + Random.State.int rs 10)
      in
      es.((r * k) + j) <- e;
      last := e
    done
  done;
  let at = ref 0 in
  let ends =
    int64s ss (plain (a + 1)) (fun () ->
        let e = es.(!at) in
        incr at;
        e)
  in
  (* In [Wrap], starts near int64's greatest pair with the ends before them,
     so that a run reads the axis from position [p - 1 - c] on modulo
     2^64. *)
  let starts =
    int64s ss sv (fun () ->
        if mode = Wrap then
          Int64.sub Int64.max_int (Int64.of_int (Random.State.int rs 3))
        else start rs n)
  in
  let cap = (2 * n) + 12 in
  let most = Int64.to_int (Int64.min (greatest es) (Int64.of_int cap)) in
  let size =
    match pick with
    | 0 -> 0
    | 1 -> max 0 (most - 1)
    | 2 -> most
    | 3 -> most + 3
    | _ -> Random.State.int rs (cap + 1)
  in
  {
    size;
    starts;
    ends;
    x = operand d xs xv (fun _ -> element d rs);
    mode;
    views =
      List.map (( ^ ) "x ") (view_names xv)
      @ List.map (( ^ ) "starts ") (view_names sv);
  }

let drawn = Gen.with_pp pp_case case_gen

(* The contract's sentence *)

(* What a run's span reads: its rows, then the first and the past-last of
   them inside [0, n). *)
type span = {
  m : int;
  lo : int;
  hi : int;
  w : int64;
  wrapped : bool;
  s : int64;
  d : int;  (** The destination's element its first row inside lands on. *)
}

(* Each lead row's positions, [-1] past every end, by the sentence: row [p]
   reads [starts.{j} + p - ends.{j - 1}] in int64 of the first run [j]
   whose end exceeds [p]; and the spans each run reads. *)
let reference c =
  let ss = shape_of c.starts in
  let a = Array.length ss - 1 in
  let k = ss.(a) and n = (shape_of c.x).(a) in
  let rows = total (Array.sub ss 0 a) in
  let xs = shape_of c.x in
  let per = total (Array.sub xs (a + 1) (Array.length xs - a - 1)) in
  let st = positions c.starts and en = positions c.ends in
  let pos = Array.make (rows * c.size) (-1L) in
  let spans = ref [] in
  for r = 0 to rows - 1 do
    let j = ref 0 and current = ref (-1) in
    for p = 0 to c.size - 1 do
      let p64 = Int64.of_int p in
      while !j < k && Int64.compare en.((r * k) + !j) p64 <= 0 do incr j done;
      if !j < k then begin
        let s = st.((r * k) + !j) in
        let before = if !j = 0 then 0L else en.((r * k) + !j - 1) in
        let w = Int64.add s (Int64.sub p64 before) in
        pos.((r * c.size) + p) <- w;
        if !current <> !j then begin
          current := !j;
          (* The run's rows here: up to its end or the size. *)
          let e = en.((r * k) + !j) in
          let stop =
            if Int64.compare e (Int64.of_int c.size) >= 0 then c.size
            else Int64.to_int e
          in
          let m = stop - p in
          let inside i =
            let q = Int64.add w (Int64.of_int i) in
            Int64.compare q 0L >= 0 && Int64.compare q (Int64.of_int n) < 0
          in
          let first = ref m and last = ref (-1) in
          for i = 0 to m - 1 do
            if inside i then begin
              if !first = m then first := i;
              last := i
            end
          done;
          let lo = if !last < 0 then m else !first in
          let hi = if !last < 0 then m else !last + 1 in
          (* A row inside the axis that the exact sum puts outside it: a
             start at least 0 whose sum with the row's offset wraps past
             2^64 to below the start. *)
          let wrapped =
            lo < hi
            && Int64.compare s 0L >= 0
            && Int64.compare (Int64.add w (Int64.of_int lo)) s < 0
          in
          let d = ((r * c.size) + p + lo) * per in
          spans := { m; lo; hi; w; wrapped; s; d } :: !spans
        end
      end
    done
  done;
  (pos, !spans)

(* The positions [pos] of rows [[lead…; size]] at each cell: a fresh array
   of the result's shape. *)
let broadcast_cells c pos =
  let xs = shape_of c.x and ss = shape_of c.starts in
  let a = Array.length ss - 1 in
  let ys = Array.copy xs in
  ys.(a) <- c.size;
  let per = total (Array.sub xs (a + 1) (Array.length xs - a - 1)) in
  let at = ref 0 in
  int64s ys (plain (Array.length ys)) (fun () ->
      let p = pos.(!at / per) in
      incr at;
      p)

let ranges_on (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (A.Any x) = c.x in
  let xs = shape_of c.x and a = Array.length (shape_of c.starts) - 1 in
  let ys = Array.copy xs in
  ys.(a) <- c.size;
  let dst = A.create b.device (A.dtype x) (Iarray.of_array ys) in
  let x = Support.on b x in
  let starts = Support.on b (A.expect D.Int64 c.starts) in
  let ends = Support.on b (A.expect D.Int64 c.ends) in
  match K.gather_ranges ~dst ~starts ~ends x with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> failf "%s declined a range gather" b.name
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Laws *)

let law_gather (b : Support.backend) c =
  let ss = shape_of c.starts in
  let a = Array.length ss - 1 in
  let k = ss.(a) and n = (shape_of c.x).(a) in
  let pos, spans = reference c in
  let en = positions c.ends in
  let bits = (fun (D.Any dt) -> D.bits dt) (dtype_of c.x) in
  let any f = List.exists f spans in
  let packed = c.mode = Packed && k > 0 in
  let most = greatest en and s64 = Int64.of_int c.size in
  (* Whether a row's runs step from one end to the next as [f] says. *)
  let steps f =
    Array.exists Fun.id
      (Array.mapi (fun i e -> i mod max 1 k > 0 && f en.(i - 1) e) en)
  in
  let views prefix = List.exists (String.starts_with ~prefix) c.views in
  cover "no run" (k = 0);
  cover "rows of more than 4096 runs" (k > 4096);
  cover "an empty axis" (n = 0);
  cover "a size of 0" (c.size = 0);
  cover "no lead axis" (a = 0);
  cover "two lead axes" (a = 2);
  cover "cells" (Array.length (shape_of c.x) > a + 1);
  cover "packed runs short of the size" (packed && Int64.compare most s64 < 0);
  cover "packed runs ending at the size" (packed && most = s64);
  cover "packed runs cut at the size"
    (packed && Int64.compare most s64 > 0 && c.size > 0);
  cover "a run straddling 0" (any (fun s -> s.lo > 0 && s.lo < s.hi));
  cover "a run straddling the axis's end"
    (any (fun s -> s.hi < s.m && s.lo < s.hi));
  cover "a run wholly outside the axis" (any (fun s -> s.m > 0 && s.lo = s.hi));
  cover "a run from int64's least" (any (fun s -> s.s = Int64.min_int));
  cover "a run from int64's greatest" (any (fun s -> s.s = Int64.max_int));
  cover "a negative length"
    (c.mode <> Wrap && steps (fun e0 e -> Int64.compare e e0 < 0));
  cover "a length of int64's greatest"
    (steps (fun e0 e -> Int64.sub e e0 = Int64.max_int));
  cover "a run inside the axis only modulo 2^64" (any (fun s -> s.wrapped));
  cover "a source element read twice"
    (let seen = Hashtbl.create 16 and twice = ref false in
     let n64 = Int64.of_int n in
     Array.iteri
       (fun i p ->
         if Int64.compare p 0L >= 0 && Int64.compare p n64 < 0 then begin
           let key = (i / max 1 c.size, p) in
           if Hashtbl.mem seen key then twice := true
           else Hashtbl.add seen key ()
         end)
       pos;
     !twice);
  let sub off =
    bits < 8 && any (fun s -> s.lo < s.hi && off s mod (8 / bits) <> 0)
  in
  cover "a sub-byte run into the middle of a byte" (sub (fun s -> s.d));
  cover "a sub-byte run from the middle of a byte"
    (sub (fun s -> Int64.to_int (Int64.add s.w (Int64.of_int s.lo))));
  let line =
    bits = 8 && Array.length (shape_of c.x) = a + 1 && not (views "x")
  in
  cover "one-byte rows on the line path" (line && k > 0 && c.size > 0);
  cover "one-byte packed runs on the line path"
    (line && packed && n > 0 && c.size > 0);
  cover "strided starts" (views "starts");
  cover "a strided or broadcast source" (views "x");
  match ranges_on b c with
  | None -> ()
  | Some y -> (
      let idx = broadcast_cells c pos in
      match gather_on b ~axis:a idx c.x with
      | None -> failf "%s computed a range gather, declined a gather" b.name
      | Some g -> equal (array string) (elements g) (elements y))

(* nx.cpu: one thread and many *)

(* Runs of 1 to 20 elements and one of [long], over [lead] rows, at [dt]:
   a size that splits among threads, a long run among short ones, and a
   sub-byte row that ends inside a byte. *)
let split_case dt ~lead ~long ~size =
  let rs = Random.State.make [| 7 |] in
  let n = 4099 and k = 2000 in
  let es = Array.make (lead * k) 0L in
  for r = 0 to lead - 1 do
    let last = ref 0L in
    for j = 0 to k - 1 do
      let l = if j = k / 2 then long else 1 + Random.State.int rs 20 in
      last := Int64.add !last (Int64.of_int l);
      es.((r * k) + j) <- !last
    done
  done;
  let at = ref 0 and ss = [| lead; k |] in
  {
    size;
    starts =
      int64s ss (plain 2) (fun () -> Int64.of_int (Random.State.int rs n - 3));
    ends =
      int64s ss (plain 2) (fun () ->
          let e = es.(!at) in
          incr at;
          e);
    x = operand dt [| lead; n |] (plain 2) (fun _ -> element dt rs);
    mode = Packed;
    views = [];
  }

let split_cases =
  let case name dt ~lead ~long ~size =
    (name, split_case (D.Any dt) ~lead ~long ~size)
  in
  [
    case "bytes, one long run" D.Uint8 ~lead:1 ~long:300_000 ~size:321_000;
    case "int4, rows ending inside a byte" D.Int4 ~lead:3 ~long:200_001
      ~size:221_001;
    case "float64, a tail of zeros" D.Float64 ~lead:2 ~long:40_000
      ~size:90_001;
  ]

let test_threads (_, c) =
  let b = Support.cpu in
  let run () = b.around (fun () -> ranges_on b c) in
  match (Nx_cpu.one_thread run, run ()) with
  | Some y, Some y' -> equal (array string) (elements y) (elements y')
  | _ -> fail "nx.cpu declined a range gather"

(* Misfits *)

(* Range gathers that do not fit: not Done, and nothing written outside
   the destination. A destination over its source fits, and the door
   refuses it as Overlapping. *)
let misfits (b : Support.backend) =
  let module K = (val b.kernels) in
  let make dt zero s =
    Support.on b (A.of_array dt (Iarray.of_array s) (Array.make (total s) zero))
  in
  let ints = make D.Int64 0L and floats = make D.Float32 0. in
  let call ~dst ~runs:(s, e) x =
    K.gather_ranges ~dst ~starts:(ints s) ~ends:(ints e) x
  in
  let row name dst ~runs x =
    ( name,
      fun () ->
        Misfit.check ~msg:name ~what:"a shape" b
          [| (D.Any D.Float32, dst) |]
          (fun g ds ->
            call ~dst:(A.expect D.Float32 ds.(0)) ~runs (g.Misfit.g x)) )
  in
  [
    row "starts and ends of other shapes" [| 4 |] ~runs:([| 2 |], [| 3 |])
      (floats [| 5 |]);
    row "0-d starts" [| 4 |] ~runs:([||], [||]) (floats [| 5 |]);
    row "another lead extent" [| 3; 4 |]
      ~runs:([| 2; 2 |], [| 2; 2 |])
      (floats [| 3; 5 |]);
    row "a destination of other cells" [| 4; 2 |] ~runs:([| 2 |], [| 2 |])
      (floats [| 5; 3 |]);
    row "a destination of another rank" [| 4 |] ~runs:([| 2 |], [| 2 |])
      (floats [| 5; 3 |]);
    row "a source without the ranged axis" [| 2 |]
      ~runs:([| 2; 2 |], [| 2; 2 |])
      (floats [| 2 |]);
    ( "a destination over its source",
      fun () ->
        let x = floats [| 8 |] in
        let range = { A.Move.start = 1; count = 4; step = 1 } in
        let dst = Option.get (A.move (Slice [| range |]) x) in
        equal answer A.Overlapping (call ~dst ~runs:([| 2 |], [| 2 |]) x) );
  ]

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop ~count:300
        "a range gather is the gather at the positions its sentence gives, bit \
         for bit"
        drawn (run (law_gather b));
      cases
        "a call that does not fit is not Done and writes nothing outside its \
         destination"
        ~id:fst (misfits b)
        (fun (_, f) -> b.around f);
    ]

(* rig.nv *)

(* rig.nv loads the cubin CUDA loads, so a range gather through it has CUDA's
   bits: what it can get wrong, its launches, memory and order, shows in
   them. Each case runs through CUDA, then through rig.nv. *)
let nv_laws (first : Support.backend) (nv : Support.backend) =
  let agree call = Elements.agree_bits first nv call in
  let some = function Some y -> [| y |] | None -> [||] in
  group nv.name
    [
      group "bits are cuda's"
        [
          prop ~count:300 "of range gathers" drawn (fun c ->
              agree (fun b -> some (ranges_on b c)));
        ];
    ]

let () =
  let nv =
    match (Support.gpus (), Support.nv ()) with
    | first :: _, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.ranges"
       (List.map laws (Support.backends ())
       @ [
           cases "nx.cpu gives one thread's bits on many" ~id:fst split_cases
             test_threads;
         ]
       @ nv))
