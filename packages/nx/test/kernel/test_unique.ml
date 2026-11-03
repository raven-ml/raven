(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Numberings through every kernel library the host runs, against a table of
   rows numbered as they are first met: the ids, the first rows and counts at
   any length, and the count, through views of the keys. nx.cpu's group
   draws the keys its doc says find their numbers by a direct table and those
   that hash, and checks that one thread and the pool give the same bits. *)

open Windtrap
open Elements
module Support = Nx_kernels_support

(* How keys are drawn: from a few values, so that keys repeat; each word at
   random; equal in their low 40 bits, differing in a few high ones, so that
   their hashes' low bits agree; or within 8 of a base, a small span. *)
type draw = Tied | Spread | High | Narrow

type case = { keys : A.any; m : int; draw : draw; views : string list }

let pp_case ppf c =
  Format.fprintf ppf "unique of %a keeping %d, %s (%s)" pp_ints
    (shape_of c.keys) c.m
    (match c.draw with
    | Tied -> "tied"
    | Spread -> "spread"
    | High -> "high bits"
    | Narrow -> "narrow")
    (String.concat ", " c.views)

(* A word drawn from [rs] as [draw] says, of [pool]. *)
let word rs pool draw =
  let pick () = pool.(Random.State.int rs (Array.length pool)) in
  match draw with
  | Spread -> Random.State.bits64 rs
  | Tied ->
      if Random.State.int rs 4 = 0 then Random.State.bits64 rs else pick ()
  | High ->
      Int64.logor
        (Int64.logand (pick ()) 0xFF_FFFF_FFFFL)
        (Int64.shift_left (Int64.of_int (Random.State.int rs 6)) 40)
  | Narrow -> Int64.add pool.(0) (Int64.of_int (Random.State.int rs 8))

let drawn draw s v seed =
  let rs = Random.State.make [| seed |] in
  let pool = Array.init 3 (fun _ -> Random.State.bits64 rs) in
  int64s s v (fun () -> word rs pool draw)

let shape_gen =
  let open Gen in
  let* n =
    frequency
      [ (4, int_range 0 40); (1, int_range 100 3000); (1, constant 1) ]
  in
  let+ w =
    frequency [ (4, int_range 1 3); (1, constant 0); (1, int_range 4 6) ]
  in
  [| n; w |]

let case_gen ?(shapes = shape_gen) () =
  let open Gen in
  let* s = shapes in
  let* m =
    frequency
      [
        (2, int_range 0 (s.(0) + 2));
        (1, constant s.(0));
        (1, int_range 0 3);
      ]
  in
  let* bc = frequency [ (5, constant []); (1, constant [ 0 ]) ] in
  let* v = view_of ~broadcast:bc 2 in
  let* draw =
    frequency
      [
        (3, constant Tied);
        (1, constant Spread);
        (1, constant High);
        (1, constant Narrow);
      ]
  in
  let+ seed = int in
  { keys = drawn draw s v seed; m; draw; views = view_names v }

let cases = Gen.with_pp pp_case (case_gen ())

(* A constructed case through [view], drawn from seed 1, so that every seed
   reaches its regime. *)
let example ?(view = plain 2) ?m draw s =
  let m = Option.value m ~default:s.(0) in
  { keys = drawn draw s view 1; m; draw; views = view_names view }

let examples =
  [
    example Tied [| 0; 2 |];
    example Tied [| 0; 0 |] ~m:2;
    example Tied [| 5; 0 |] ~m:3;
    example Tied [| 7; 2 |] ~view:{ (plain 2) with broadcast = [ 0 ] };
    example Spread [| 9; 1 |] ~m:4;
    example Spread [| 9; 3 |] ~m:12 ~view:{ (plain 2) with perm = [| 1; 0 |] };
    example High [| 30; 1 |];
    example Narrow [| 30; 2 |] ~view:{ (plain 2) with reversed = Some 0 };
  ]

(* The case's numbering by the kernels, or [None] where they decline. *)
let unique_on (b : Support.backend) c =
  let module K = (val b.kernels) in
  let s = shape_of c.keys in
  let (A.Any keys) = c.keys in
  let keys = A.expect D.Int64 (A.Any keys) in
  let make n = A.create Rig.host D.Int64 (Iarray.of_array n) in
  let ids = make [| s.(0) |] and first = make [| c.m |] in
  let counts = make [| c.m |] and count = make [||] in
  match K.unique ~ids ~first ~counts ~count keys with
  | A.Done ->
      let get = A.to_array in
      Some (get ids, get first, get counts, A.get count [||])
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* The rows numbered as they are first met: each row's number, and each
   key's first row and count. *)
let expected c =
  let s = shape_of c.keys in
  let n = s.(0) and w = s.(1) in
  let words = positions c.keys in
  let seen = Hashtbl.create 16 and keys = ref [] in
  let ids =
    Array.init n (fun i ->
        let row = Array.sub words (i * w) w in
        match Hashtbl.find_opt seen row with
        | Some (g, rows) ->
            incr rows;
            g
        | None ->
            let g = Hashtbl.length seen in
            let rows = ref 1 in
            Hashtbl.add seen row (g, rows);
            keys := (i, rows) :: !keys;
            g)
  in
  (Array.map Int64.of_int ids, Array.of_list (List.rev !keys))

let agrees (b : Support.backend) c =
  match unique_on b c with
  | None -> failf "%s declined a numbering" b.name
  | Some (ids, first, counts, count) ->
      let n = (shape_of c.keys).(0) in
      let eids, keys = expected c in
      let k = Array.length keys in
      let at f d = Array.init c.m (fun g -> if g < k then f keys.(g) else d) in
      equal ~msg:"ids" (array int64) eids ids;
      equal ~msg:"count" int64 (Int64.of_int k) count;
      equal ~msg:"first" (array int64)
        (at (fun (f, _) -> Int64.of_int f) (Int64.of_int n))
        first;
      equal ~msg:"counts" (array int64)
        (at (fun (_, r) -> Int64.of_int !r) 0L)
        counts

let law_reference (b : Support.backend) c =
  let s = shape_of c.keys in
  let _, keys = expected c in
  let k = Array.length keys in
  cover "no row" (s.(0) = 0);
  cover "no word" (s.(1) = 0 && s.(0) > 0);
  cover "one key of many rows" (k = 1 && s.(0) > 1);
  cover "every row its own key" (k = s.(0) && s.(0) > 1);
  cover "first and counts cut" (c.m < k);
  cover "first and counts past the keys" (c.m > k);
  cover "keys equal in their low bits" (c.draw = High && k > 1);
  agrees b c

(* Refusals *)

let law_refuses (b : Support.backend) (keys, ids, first, counts, count) =
  let module K = (val b.kernels) in
  let make s = A.create Rig.host D.Int64 (Iarray.of_array s) in
  let fits =
    match Nx_kernel.Shape.unique ~size:first.(0) keys with
    | Ok (i, f, c) ->
        i = ids && f = first && f = counts && c = count
    | Error _ -> false
    | exception Invalid_argument _ -> false
  in
  let got =
    K.unique ~ids:(make ids) ~first:(make first) ~counts:(make counts)
      ~count:(make count) (make keys)
  in
  cover "fits" fits;
  cover "misfits" (not fits);
  equal answer (if fits then A.Done else A.Shape_mismatch) got

let misfits =
  let open Gen in
  let near d =
    frequency [ (3, constant d); (1, int_range (max 0 (d - 1)) (d + 1)) ]
  in
  let* n = int_range 0 4 in
  let* w = int_range 0 2 in
  let* m = int_range 0 3 in
  let* rank = frequency [ (5, constant 2); (1, constant 1); (1, constant 3) ] in
  let keys = Array.sub [| n; w; 1 |] 0 rank in
  let* i = near n in
  let* c = near m in
  let+ count = frequency [ (5, constant [||]); (1, constant [| 1 |]) ] in
  (keys, [| i |], [| m |], [| c |], count)

let misfits =
  Gen.with_pp
    (fun ppf (k, i, f, c, n) ->
      Format.fprintf ppf "keys %a ids %a first %a counts %a count %a" pp_ints k
        pp_ints i pp_ints f pp_ints c pp_ints n)
    misfits

(* nx.cpu *)

(* Rows enough for several threads, each word drawn from a few values or at
   random, of one to five words. *)
let threaded =
  let open Gen in
  let shapes =
    let* n = int_range 40_000 120_000 in
    let+ w = int_range 1 5 in
    [| n; w |]
  in
  Gen.with_pp pp_case (case_gen ~shapes ())

(* Many rows: a hash table of a few keys and of about a key a row, keys of
   five words, and a direct table, through a view. *)
let large =
  [
    example Tied [| 60_000; 2 |] ~m:5;
    example Spread [| 60_000; 1 |];
    example Spread [| 50_000; 5 |] ~m:100;
    example High [| 50_000; 1 |];
    example Narrow [| 70_000; 3 |] ~view:{ (plain 2) with perm = [| 1; 0 |] };
  ]

(* Keys of at most 4 words whose columns' spans multiply to at most twice the
   rows and 1024 more take a direct table. *)
let direct c =
  let s = shape_of c.keys in
  let n = s.(0) and w = s.(1) in
  let words = positions c.keys in
  let span j =
    let lo = ref (-1L) and hi = ref 0L in
    for i = 0 to n - 1 do
      let x = words.((i * w) + j) in
      if Int64.unsigned_compare x !lo < 0 then lo := x;
      if Int64.unsigned_compare x !hi > 0 then hi := x
    done;
    Int64.sub !hi !lo
  in
  let limit = Int64.of_int ((2 * n) + 1024) in
  let rec values j acc =
    if j = w then true
    else
      let d = span j in
      if Int64.unsigned_compare d limit >= 0 then false
      else
        let acc = Int64.mul acc (Int64.succ d) in
        Int64.unsigned_compare acc limit <= 0 && values (j + 1) acc
  in
  n > 0 && w > 0 && w <= 4 && values 0 1L

let law_paths (b : Support.backend) c =
  cover "direct table" (direct c);
  cover "hash table" (not (direct c));
  agrees b c

let law_large (b : Support.backend) c =
  let _, keys = expected c in
  let k = Array.length keys in
  cover "a few keys" (k <= 64);
  cover "about a key a row" (2 * k > (shape_of c.keys).(0));
  agrees b c

let law_threads (b : Support.backend) c =
  cover "direct table" (direct c);
  cover "hash table" (not (direct c));
  match (Nx_cpu.one_thread (fun () -> unique_on b c), unique_on b c) with
  | Some (i1, f1, c1, k1), Some (i, f, c', k) ->
      equal ~msg:"ids" (array int64) i1 i;
      equal ~msg:"first" (array int64) f1 f;
      equal ~msg:"counts" (array int64) c1 c';
      equal ~msg:"count" int64 k1 k
  | _ -> failf "%s declined a numbering" b.name

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop ~examples "numbers rows in order of first appearance" cases
        (run (law_reference b));
      prop "refuses what Nx_kernel.Shape refuses and other shapes" misfits
        (run (law_refuses b));
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop ~examples "numbers by a direct table or a hash table" cases
        (run (law_paths b));
      prop ~count:8 ~examples:large "one thread and the pool give the same bits"
        threaded
        (run (law_threads b));
      prop ~count:4 ~examples:large "numbers many rows on the pool" threaded
        (run (law_large b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.unique"
       (List.map laws Support.backends @ [ cpu Support.cpu ]))
