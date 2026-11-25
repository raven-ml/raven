(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx_kernel.S's gather_ranges on every kernel library the host runs: each is
   the library's gather at the positions its sentence gives, bit for bit, over
   every dtype and view; each library refuses what does not fit and a
   destination over its source; nx.cpu gives one thread's bits on many. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module Support = Nx_kernels_support
open Elements
open Index_cases

(* Cases *)

(* How a row's runs are drawn: ends that sum lengths, any ends, or ends
   near int64's least before starts near its greatest. *)
type mode = Packed | Wild | Wrap

(* How ends are laid out: C-contiguous, every other element of rows twice as
   long, rows stored backwards, or one row broadcast along the first lead
   axis. *)
type layout = Contiguous | Stepped | Reversed | Broadcast

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

(* The ends [es] of rows of lead axes [made] and [k] runs each, laid out by
   [el], in C order those of [es]; broadcast, they repeat along the first lead
   axis to [lead]. *)
let ends_of el ~lead made k es =
  let rows = total made and shape = Array.append made [| k |] in
  let ints s v = A.of_array D.Int64 (Iarray.of_array s) v in
  let whole e = { M.start = 0; count = e; step = 1 } in
  let slice last x =
    let ranges = Array.append (Array.map whole made) [| last |] in
    Option.get (A.move (M.Slice (Iarray.of_array ranges)) x)
  in
  A.Any
    (match el with
    | Contiguous -> ints shape es
    | Stepped ->
        (* -7 between ends, which a kernel reading the gaps would take for
           ends. *)
        ints
          (Array.append made [| 2 * k |])
          (Array.init
             (rows * 2 * k)
             (fun i -> if i mod 2 = 0 then es.(i / 2) else -7L))
        |> slice { M.start = 0; count = k; step = 2 }
    | Reversed ->
        ints shape
          (Array.init (rows * k) (fun i ->
               es.((i / k * k) + (k - 1 - (i mod k)))))
        |> slice { M.start = max 0 (k - 1); count = k; step = -1 }
    | Broadcast ->
        Option.get
          (A.move
             (M.Broadcast (Iarray.of_array (Array.append lead [| k |])))
             (ints shape es)))

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
  let* el =
    let broadcast = if Array.length lead > 0 then 1 else 0 in
    frequency
      [
        (3, constant Contiguous);
        (1, constant Stepped);
        (1, constant Reversed);
        (broadcast, constant Broadcast);
      ]
  in
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
  (* The lead axes of the rows made: one along the first where broadcast. *)
  let made =
    Array.mapi (fun i e -> if i = 0 && el = Broadcast then 1 else e) lead
  in
  let rows = total made in
  (* Each made row's ends in run order: running sums, any values, or, in [Wrap],
     ends near int64's least in even runs and small ones in odd runs. *)
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
  let ends = ends_of el ~lead made k es in
  let ends_views =
    match el with
    | Contiguous -> []
    | Stepped -> [ "ends stepped" ]
    | Reversed -> [ "ends reversed" ]
    | Broadcast -> [ "ends broadcast" ]
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
      @ List.map (( ^ ) "starts ") (view_names sv)
      @ ends_views;
  }

let drawn = Gen.with_pp pp_case case_gen

(* Two packed runs of int4, the first from position 1, the second into the
   destination's second element, each mid-byte, which the draws reach about
   one case in 16. *)
let mid_byte_runs =
  let d = D.Any D.Int4 and rs = Random.State.make [| 1 |] in
  let int64s_of ps =
    let at = ref 0 in
    int64s [| 2 |] (plain 1) (fun () ->
        let p = ps.(!at) in
        incr at;
        p)
  in
  {
    size = 4;
    starts = int64s_of [| 1L; 3L |];
    ends = int64s_of [| 1L; 4L |];
    x = operand d [| 8 |] (plain 1) (fun _ -> element d rs);
    mode = Packed;
    views = [];
  }

(* The contract's sentence *)

(* Each lead row's positions, [-1] past every end, by the sentence: position [p]
   reads [starts.{j} + p - ends.{j - 1}] in int64 of the first run [j] whose end
   exceeds [p]; and the run each reads, as its index into [starts], [-1] past
   every end. *)
let reference c =
  let ss = shape_of c.starts in
  let a = Array.length ss - 1 in
  let k = ss.(a) and rows = total (Array.sub ss 0 a) in
  let st = positions c.starts and en = positions c.ends in
  let pos = Array.make (rows * c.size) (-1L) in
  let run = Array.make (rows * c.size) (-1) in
  for r = 0 to rows - 1 do
    let j = ref 0 in
    for p = 0 to c.size - 1 do
      let p64 = Int64.of_int p in
      while !j < k && Int64.compare en.((r * k) + !j) p64 <= 0 do
        incr j
      done;
      if !j < k then begin
        let before = if !j = 0 then 0L else en.((r * k) + !j - 1) in
        pos.((r * c.size) + p) <-
          Int64.add st.((r * k) + !j) (Int64.sub p64 before);
        run.((r * c.size) + p) <- (r * k) + !j
      end
    done
  done;
  (pos, run)

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
  | A.Done -> A.Any dst
  | A.Declined -> failf "%s declined a range gather" b.name
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Laws *)

let law_gather (b : Support.backend) c =
  let ss = shape_of c.starts and xs = shape_of c.x in
  let a = Array.length ss - 1 in
  let k = ss.(a) and n = xs.(a) and size = c.size in
  let pos, run = reference c in
  let st = positions c.starts and en = positions c.ends in
  let bits = (fun (D.Any dt) -> D.bits dt) (dtype_of c.x) in
  let n64 = Int64.of_int n in
  let inside q = Int64.compare q 0L >= 0 && Int64.compare q n64 < 0 in
  let exists f = Array.exists Fun.id (Array.init (Array.length pos) f) in
  (* Position [i] and the one before it in its row read one run. *)
  let within i = i mod size > 0 && run.(i) >= 0 && run.(i - 1) = run.(i) in
  (* Position [i] is its run's first inside the axis. *)
  let enters i =
    run.(i) >= 0 && inside pos.(i) && not (within i && inside pos.(i - 1))
  in
  let crosses q q' =
    exists (fun i -> within i && pos.(i - 1) = q && pos.(i) = q')
  in
  let packed = c.mode = Packed && k > 0 in
  let most = greatest en and s64 = Int64.of_int size in
  (* Whether a row's runs step from one end to the next as [f] says. *)
  let steps f =
    Array.exists Fun.id
      (Array.mapi (fun i e -> i mod max 1 k > 0 && f en.(i - 1) e) en)
  in
  let views prefix = List.exists (String.starts_with ~prefix) c.views in
  cover "no run" (k = 0);
  cover "rows of more than 4096 runs" (k > 4096);
  cover "an empty axis" (n = 0);
  cover "a size of 0" (size = 0);
  cover "no lead axis" (a = 0);
  cover "two lead axes" (a = 2);
  cover "cells" (Array.length xs > a + 1);
  cover "packed runs short of the size" (packed && Int64.compare most s64 < 0);
  cover "packed runs ending at the size" (packed && most = s64);
  cover "packed runs cut at the size"
    (packed && Int64.compare most s64 > 0 && size > 0);
  cover "a run straddling 0" (crosses (-1L) 0L);
  cover "a run straddling the axis's end" (n > 0 && crosses (Int64.pred n64) n64);
  cover "a run wholly outside the axis"
    (let outside = Array.make (Array.length st) false in
     Array.iter (fun j -> if j >= 0 then outside.(j) <- true) run;
     Array.iteri
       (fun i j -> if j >= 0 && inside pos.(i) then outside.(j) <- false)
       run;
     Array.exists Fun.id outside);
  let opens v =
    exists (fun i -> run.(i) >= 0 && (not (within i)) && st.(run.(i)) = v)
  in
  cover "a run from int64's least" (opens Int64.min_int);
  cover "a run from int64's greatest" (opens Int64.max_int);
  cover "a negative length"
    (c.mode <> Wrap && steps (fun e0 e -> Int64.compare e e0 < 0));
  cover "a length of int64's greatest"
    (steps (fun e0 e -> Int64.sub e e0 = Int64.max_int));
  (* A position inside the axis below its run's start of at least 0: its exact
     sum passed 2^64. *)
  cover "a run inside the axis only modulo 2^64"
    (exists (fun i ->
         run.(i) >= 0
         && inside pos.(i)
         && Int64.compare st.(run.(i)) 0L >= 0
         && Int64.compare pos.(i) st.(run.(i)) < 0));
  cover "a source element read twice"
    (let seen = Hashtbl.create 16 and twice = ref false in
     Array.iteri
       (fun i p ->
         if inside p then begin
           let key = (i / max 1 size, p) in
           if Hashtbl.mem seen key then twice := true
           else Hashtbl.add seen key ()
         end)
       pos;
     !twice);
  let per = total (Array.sub xs (a + 1) (Array.length xs - a - 1)) in
  let mid at =
    bits < 8 && exists (fun i -> enters i && at i mod (8 / bits) <> 0)
  in
  cover "a sub-byte run into the middle of a byte" (mid (fun i -> i * per));
  cover "a sub-byte run from the middle of a byte"
    (mid (fun i -> Int64.to_int pos.(i)));
  let line = bits = 8 && Array.length xs = a + 1 && not (views "x") in
  cover "one-byte rows on the line path" (line && k > 0 && size > 0);
  cover "one-byte packed runs on the line path"
    (line && packed && n > 0 && size > 0);
  let placed =
    let on (A.Any x) = A.Any (Support.on b x) in
    { c with starts = on c.starts; ends = on c.ends; x = on c.x }
  in
  let strided (A.Any x) = not (L.is_contiguous (A.layout x)) in
  let gapped (A.Any x) = Support.gapped (A.layout x) in
  cover "strided starts" (strided placed.starts);
  cover "stepped ends" (views "ends stepped");
  cover "reversed ends" (views "ends reversed");
  cover "broadcast ends" (views "ends broadcast");
  cover "a strided or broadcast source" (strided placed.x);
  cover "a source with gaps" (gapped placed.x);
  let y = ranges_on b placed in
  let g = gathered b ~axis:a (broadcast_cells c pos) placed.x in
  equal (array string) (elements g) (elements y)

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
    (name, lazy (split_case (D.Any dt) ~lead ~long ~size))
  in
  [
    case "bytes, one long run" D.Uint8 ~lead:1 ~long:300_000 ~size:321_000;
    case "int4, rows ending inside a byte" D.Int4 ~lead:3 ~long:200_001
      ~size:221_001;
    case "float64, a tail of zeros" D.Float64 ~lead:2 ~long:40_000
      ~size:90_001;
  ]

let test_threads (_, c) =
  let b = Support.cpu and c = Lazy.force c in
  let run () = b.around (fun () -> ranges_on b c) in
  equal (array string) (elements (Nx_cpu.one_thread run)) (elements (run ()))

(* Misfits *)

(* Range gathers that do not fit, as the shapes of dst, starts, ends and x. *)
let misfits =
  [
    ("starts and ends of other shapes", ([| 4 |], [| 2 |], [| 3 |], [| 5 |]));
    ("0-d starts", ([| 4 |], [||], [||], [| 5 |]));
    ("another lead extent", ([| 3; 4 |], [| 2; 2 |], [| 2; 2 |], [| 3; 5 |]));
    ("a destination of other cells", ([| 4; 2 |], [| 2 |], [| 2 |], [| 5; 3 |]));
    ("a destination of another rank", ([| 4 |], [| 2 |], [| 2 |], [| 5; 3 |]));
    ( "a source without the ranged axis",
      ([| 2 |], [| 2; 2 |], [| 2; 2 |], [| 2 |]) );
  ]

let zeros d zero s =
  A.of_array d (Iarray.of_array s) (Array.make (total s) zero)

let test_misfit (b : Support.backend) (name, (dst, s, e, x)) =
  let module K = (val b.kernels) in
  let ints s = Support.on b (zeros D.Int64 0L s) in
  let x = Support.on b (zeros D.Float32 0. x) in
  Misfit.check ~msg:name ~what:"a shape" b
    [| (D.Any D.Float32, dst) |]
    (fun g ds ->
      K.gather_ranges
        ~dst:(A.expect D.Float32 ds.(0))
        ~starts:(g.Misfit.g (ints s))
        ~ends:(g.Misfit.g (ints e))
        (g.Misfit.g x))

(* A destination of four elements of an eight-element source from its second,
   which fits as a call. *)
let test_overlap (b : Support.backend) =
  let module K = (val b.kernels) in
  let x = Support.on b (zeros D.Float32 0. [| 8 |]) in
  let range = { M.start = 1; count = 4; step = 1 } in
  let dst = Option.get (A.move (M.Slice (Iarray.of_array [| range |])) x) in
  let runs = Support.on b (zeros D.Int64 0L [| 2 |]) in
  equal answer A.Overlapping (K.gather_ranges ~dst ~starts:runs ~ends:runs x)

(* The suite *)

let laws (d : Support.declared) =
  let backend () = Support.opened d in
  let run f x =
    let b = backend () in
    b.around (fun () -> f b x)
  in
  group d.label
    [
      group "gather_ranges"
        [
          prop ~count:300 ~examples:[ mid_byte_runs ]
            "is the gather at the positions its runs give, bit for bit" drawn
            (run law_gather);
          cases
            "a call Shape.gather_ranges does not fit is not Done and writes \
             nothing outside dst"
            ~id:fst misfits (run test_misfit);
          test "a destination over its source is refused as Overlapping"
            (fun () ->
              let b = backend () in
              b.around (fun () -> test_overlap b));
        ];
    ]

(* rig.nv *)

(* rig.nv loads the cubin CUDA loads, so a range gather through it has CUDA's
   bits: what it can get wrong, its launches, memory and order, shows in
   them. Each case runs through CUDA, then through rig.nv. *)
let nv_laws (first : Support.declared) (nv : Support.declared) =
  let agree call =
    Elements.agree_bits (Support.opened first) (Support.opened nv) call
  in
  group nv.label
    [
      group "bits are cuda's"
        [
          prop ~count:300 ~examples:[ mid_byte_runs ] "of range gathers" drawn
            (fun c -> agree (fun b -> [| ranges_on b c |]));
        ];
    ]

let () =
  let nv =
    match (Support.declared_cuda, Support.declared_nv) with
    | Some first, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.ranges"
       (List.map laws
          (Support.declare Support.cpu
          :: List.filter_map Fun.id
               [
                 Support.declared_cuda;
                 Support.declared_amd;
                 Support.declared_metal;
               ])
       @ [
           group "nx.cpu"
             [
               cases "a range gather gives one thread's bits on many" ~id:fst
                 split_cases test_threads;
             ];
         ]
       @ nv))
