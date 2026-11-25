(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx_kernel.S.unique through every kernel library the host runs, against a
   table of rows numbered as they are first met: the ids, the first rows and
   counts at any length, and the count, through views of the keys. nx.cpu's
   group checks its hash's seed and states and that one thread and the pool give
   the same bits. *)

open Windtrap
open Elements
module Support = Nx_kernels_support

(* nx.cpu draws its hash's seed per process; a fixed one makes the draws of
   colliding rows, and the hash's probes, the same on every run, and is the seed
   tests restore. *)
let suite_seed = 0x5EED_0F_0A11L
let () = Nx_cpu.unique_seed suite_seed

(* [f ()] with nx.cpu's hash keyed by [s], then by the suite's seed again. *)
let under_seed s f =
  Fun.protect
    ~finally:(fun () -> Nx_cpu.unique_seed suite_seed)
    (fun () ->
      Nx_cpu.unique_seed s;
      f ())

(* How keys are drawn: from a few values, so that keys repeat; each word at
   random; rows of two words [[|a; c lxor state a|]] that all hash alike,
   their [a] among 64 values, so that their tags agree and a probe passes
   every key before it; within 8 of a base, a small span; or within [d] of a
   base, [d] the [w]-th root of the rows, so that keys repeat across the
   pool's blocks and their span is too wide for a block's direct table. *)
type draw = Tied | Spread | Collide | Narrow | Dense

(* A numbering keeping [m] keys' first rows and counts, and the recipe of its
   keys: [[n; w]] through [view], drawn from [seed] as [draw] says. *)
type case = { shape : int array; view : view; draw : draw; seed : int; m : int }

let pp_case ppf c =
  Format.fprintf ppf "unique of %a keeping %d, %s (%s), seed %d" pp_ints c.shape
    c.m
    (match c.draw with
    | Tied -> "tied"
    | Spread -> "spread"
    | Collide -> "one hash"
    | Narrow -> "narrow"
    | Dense -> "dense")
    (String.concat ", " (view_names c.view))
    c.seed

(* A word drawn from [rs] as [draw] says, of [pool], [d] a dense draw's
   span. *)
let word rs pool d draw =
  let pick () = pool.(Random.State.int rs (Array.length pool)) in
  match draw with
  | Spread -> Random.State.bits64 rs
  | Tied ->
      if Random.State.int rs 4 = 0 then Random.State.bits64 rs else pick ()
  | Collide -> Int64.add pool.(0) (Int64.of_int (Random.State.int rs 64))
  | Narrow -> Int64.add pool.(0) (Int64.of_int (Random.State.int rs 8))
  | Dense -> Int64.add pool.(0) (Int64.of_int (Random.State.int rs d))

(* The second word of a colliding row led by [a]. *)
let collide = 0x0123_4567_89AB_CDEFL
let partner a = Int64.logxor collide (Nx_cpu.unique_state [| a |])

(* The case's keys. *)
let keys_of c =
  let s = c.shape in
  let rs = Random.State.make [| c.seed |] in
  let pool = Array.init 3 (fun _ -> Random.State.bits64 rs) in
  let d =
    if s.(1) = 0 then 2
    else
      Int.max 2
        (truncate (Float.of_int s.(0) ** (1. /. Float.of_int s.(1))))
  in
  match c.draw with
  | Collide ->
      (* Plain rows of two words, drawn in order: each leader, then its
         partner. *)
      let last = ref 0L and lead = ref true in
      int64s s c.view (fun () ->
          let w = if !lead then word rs pool d c.draw else partner !last in
          last := w;
          lead := not !lead;
          w)
  | Tied | Spread | Narrow | Dense ->
      int64s s c.view (fun () -> word rs pool d c.draw)

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
  let* draw =
    frequency
      [
        (3, constant Tied);
        (1, constant Spread);
        (1, constant Collide);
        (1, constant Narrow);
        (1, constant Dense);
      ]
  in
  let* s = shapes in
  let shape = if draw = Collide then [| s.(0); 2 |] else s in
  let* m =
    frequency
      [
        (2, int_range 0 (shape.(0) + 2));
        (1, constant shape.(0));
        (1, int_range 0 3);
      ]
  in
  let* bc = frequency [ (5, constant []); (1, constant [ 0 ]) ] in
  let* view = view_of ~broadcast:bc 2 in
  let view = if draw = Collide then plain 2 else view in
  let+ seed = int in
  { shape; view; draw; seed; m }

let numberings = Gen.with_pp pp_case (case_gen ())

(* A constructed case through [view], drawn from seed 1, so that every seed
   reaches its regime. *)
let example ?(view = plain 2) ?m draw shape =
  let m = Option.value m ~default:shape.(0) in
  { shape; view; draw; seed = 1; m }

(* The case's numbering of [keys] by the kernels: ids, first, counts and
   count. *)
let unique_on (b : Support.backend) c keys =
  let module K = (val b.kernels) in
  let keys = Support.on b (A.expect D.Int64 keys) in
  let make n = A.create b.device D.Int64 (Iarray.of_array n) in
  let ids = make [| c.shape.(0) |] and first = make [| c.m |] in
  let counts = make [| c.m |] and count = make [||] in
  (* Destinations start as 0xA5 bytes, not zeros: an entry the kernels do
     not store differs from the reference, which holds no such entry. *)
  List.iter
    (fun a -> Rig.Buffer.fill (A.buffer a) '\xA5')
    [ ids; first; counts; count ];
  match K.unique ~ids ~first ~counts ~count keys with
  | A.Done ->
      let get a = A.to_array (Support.host a) in
      (get ids, get first, get counts, (get count).(0))
  | A.Declined -> failf "%s declined a numbering" b.name
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* The rows of [words], [[n; w]] in C order, numbered as they are first met:
   each row's number, and each key's first row and count. *)
let expected c words =
  let n = c.shape.(0) and w = c.shape.(1) in
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

let same_numbering (i, f, n, k) (i', f', n', k') =
  equal ~msg:"ids" (array int64) i i';
  equal ~msg:"first" (array int64) f f';
  equal ~msg:"counts" (array int64) n n';
  equal ~msg:"count" int64 k k'

(* The numbering of [keys], whose words are [words], agrees with {!expected}:
   entries of first and counts from the keys' count on get [n] and [0]. *)
let agrees (b : Support.backend) c keys words =
  let n = c.shape.(0) in
  let eids, ks = expected c words in
  let k = Array.length ks in
  let at f d = Array.init c.m (fun g -> if g < k then f ks.(g) else d) in
  same_numbering
    ( eids,
      at (fun (f, _) -> Int64.of_int f) (Int64.of_int n),
      at (fun (_, r) -> Int64.of_int !r) 0L,
      Int64.of_int k )
    (unique_on b c keys)

(* Whether some row of [words] and the row before it satisfy [f]. *)
let follows f c words =
  let n = c.shape.(0) and w = c.shape.(1) in
  let row i = Array.sub words (i * w) w in
  let rec from i = i < n && (f (row (i - 1)) (row i) || from (i + 1)) in
  from 1

(* A row of the previous row's key, and one of its first word alone. *)
let repeats = follows ( = )
let leads = follows (fun a b -> Array.length a > 1 && a.(0) = b.(0) && a <> b)

(* Whether nx.cpu numbers [words] by a direct table, as nx_cpu.mli states: keys
   of at most 4 words whose columns' spans multiply to at most twice [rows] and
   1024 more, and 2^20, [rows] the rows of a block, all of them on one
   thread. *)
let direct ?rows c words =
  let n = c.shape.(0) and w = c.shape.(1) in
  let span j =
    let lo = ref (-1L) and hi = ref 0L in
    for i = 0 to n - 1 do
      let x = words.((i * w) + j) in
      if Int64.unsigned_compare x !lo < 0 then lo := x;
      if Int64.unsigned_compare x !hi > 0 then hi := x
    done;
    Int64.sub !hi !lo
  in
  let rows = Option.value rows ~default:n in
  let limit = Int64.of_int (Int.min (1 lsl 20) ((2 * rows) + 1024)) in
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

(* Numbering *)

(* No row, no word, rows broadcast, permuted or reversed, first and counts of no
   entry, cut or past the keys, keys of one hash and of a narrow span. *)
let examples =
  [
    example Tied [| 0; 2 |];
    example Tied [| 0; 0 |] ~m:2;
    example Tied [| 5; 0 |] ~m:3;
    example Tied [| 5; 2 |] ~m:0;
    example Tied [| 7; 2 |] ~view:{ (plain 2) with broadcast = [ 0 ] };
    example Tied [| 7; 2 |]
      ~view:{ (plain 2) with perm = [| 1; 0 |]; broadcast = [ 0 ] };
    example Tied [| 40; 2 |] ~view:{ (plain 2) with perm = [| 1; 0 |] };
    example Spread [| 9; 1 |] ~m:4;
    example Spread [| 9; 3 |] ~m:12 ~view:{ (plain 2) with perm = [| 1; 0 |] };
    example Tied [| 200; 10 |] ~m:50;
    example Collide [| 30; 2 |];
    example Collide [| 2000; 2 |] ~m:5;
    example Narrow [| 30; 2 |] ~view:{ (plain 2) with reversed = Some 0 };
  ]

(* The direct and the hash table are nx_cpu.mli's paths, drawn for every
   library. *)
let law_order (b : Support.backend) c =
  let s = c.shape in
  let keys = keys_of c in
  let words = positions keys in
  let k = Array.length (snd (expected c words)) in
  cover "no row" (s.(0) = 0);
  cover "no word" (s.(1) = 0 && s.(0) > 0);
  cover "one key of many rows" (k = 1 && s.(0) > 1);
  cover "every row its own key" (k = s.(0) && s.(0) > 1);
  cover "first and counts cut" (c.m < k);
  cover "first and counts past the keys" (c.m > k);
  cover "no first or counts for some rows" (c.m = 0 && s.(0) > 0);
  cover "keys of one hash" (c.draw = Collide && k > 1);
  cover "a row of the previous row's key" (repeats c words);
  cover "a row of the previous row's key, its words copied"
    (s.(1) > 1 && List.mem "permuted" (view_names c.view) && repeats c words);
  cover "a row of the previous row's first word alone" (leads c words);
  cover "a direct table" (direct c words);
  cover "a hash table" (not (direct c words));
  agrees b c keys words

(* Rows enough for several of the pool's blocks of 16,384 rows, each word drawn
   as {!case_gen} draws it, of one to five words. *)
let threaded =
  let open Gen in
  let shapes =
    let* n = int_range 33_000 50_000 in
    let+ w = int_range 1 5 in
    [| n; w |]
  in
  Gen.with_pp pp_case (case_gen ~shapes ())

(* Many rows: of one word, sorted then numbered whole, and of several, through
   views, with first and counts cut or past the keys. *)
let many =
  [
    example Spread [| 40_000; 1 |] ~m:100;
    example Dense [| 40_000; 1 |] ~m:50_000;
    example Tied [| 40_000; 3 |] ~m:5 ~view:{ (plain 2) with perm = [| 1; 0 |] };
    example Narrow [| 40_000; 2 |] ~view:{ (plain 2) with reversed = Some 0 };
    example Spread [| 30_000; 0 |] ~m:2;
  ]

(* Many rows on nx.cpu's paths: a hash table of a few keys and of about a key a
   row, keys of five words, and a direct table, through a view, and keys that
   repeat across blocks, of one and two words. *)
let large =
  [
    example Tied [| 40_000; 2 |] ~m:5;
    example Spread [| 40_000; 5 |] ~m:100;
    example Collide [| 40_000; 2 |] ~m:2;
    example Narrow [| 40_000; 3 |] ~view:{ (plain 2) with perm = [| 1; 0 |] };
    example Dense [| 40_000; 1 |];
    example Dense [| 40_000; 2 |] ~m:7;
  ]

let law_many (b : Support.backend) c =
  let keys = keys_of c in
  let words = positions keys in
  let k = Array.length (snd (expected c words)) in
  cover "one word" (c.shape.(1) = 1);
  cover "several words" (c.shape.(1) > 1);
  cover "first and counts cut" (c.m < k);
  cover "first and counts past the keys" (c.m > k);
  cover "a few keys" (k <= 64);
  cover "about a key a row" (2 * k > c.shape.(0));
  agrees b c keys words

(* Misfits *)

let pp_misfit ppf (k, i, f, c, n) =
  Format.fprintf ppf "keys %a ids %a first %a counts %a count %a" pp_ints k
    pp_ints i pp_ints f pp_ints c pp_ints n

(* Shapes of keys, ids, first, counts and count, each near a fit. *)
let misfits =
  let open Gen in
  let near d =
    frequency [ (3, constant d); (1, int_range (max 0 (d - 1)) (d + 1)) ]
  in
  with_pp pp_misfit
    (let* n = int_range 0 4 in
     let* w = int_range 0 2 in
     let* m = int_range 0 3 in
     let* rank =
       frequency [ (5, constant 2); (1, constant 1); (1, constant 3) ]
     in
     let keys = Array.sub [| n; w; 1 |] 0 rank in
     let* i = near n in
     let* c = near m in
     let+ count = frequency [ (5, constant [||]); (1, constant [| 1 |]) ] in
     (keys, [| i |], [| m |], [| c |], count))

let law_misfit (b : Support.backend) (keys, ids, first, counts, count) =
  let module K = (val b.kernels) in
  let make s = A.create b.device D.Int64 (Iarray.of_array s) in
  let fits =
    match Nx_kernel.Shape.unique ~size:first.(0) keys with
    | Ok (i, f, c) -> i = ids && f = first && f = counts && c = count
    | Error _ -> false
  in
  cover "fits" fits;
  cover "misfits" (not fits);
  let call (g : Misfit.guard) ds =
    let at k = A.expect D.Int64 ds.(k) in
    K.unique ~ids:(at 0) ~first:(at 1) ~counts:(at 2) ~count:(at 3)
      (g.g (make keys))
  in
  let dsts =
    Array.map (fun s -> (D.Any D.Int64, s)) [| ids; first; counts; count |]
  in
  Misfit.law ~msg:"a numbering" ~what:"a shape" b ~fit:fits dsts call

(* nx.cpu *)

(* The numbering under another seed has the same bits. *)
let law_seeds (b : Support.backend) (c, seed) =
  let keys = keys_of c in
  same_numbering (unique_on b c keys)
    (under_seed seed (fun () -> unique_on b c keys))

(* A seed, a [c] and 1 to 64 leaders, often repeated. *)
let leaders =
  let open Gen in
  with_pp
    (fun ppf (seed, c, a) ->
      Format.fprintf ppf "seed %Ld, c %Ld, leaders %a" seed c
        (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf ->
             Format.fprintf ppf "%Ld"))
        (Array.to_list a))
    (let* seed = int64 in
     let* c = int64 in
     let* pool = array ~size:(constant 4) int64 in
     let+ a =
       array ~size:(int_range 1 64)
         (frequency [ (3, int64); (1, map (Array.get pool) (int_range 0 3)) ])
     in
     (seed, c, a))

(* nx_cpu.mli: "a row [[|a; c lxor unique_state [|a|]|]] has the hash of every
   other such row of the same [c]". *)
let law_one_state (seed, c, a) =
  let distinct = List.sort_uniq Int64.compare (Array.to_list a) in
  cover "two leaders or more" (List.length distinct > 1);
  let states =
    under_seed seed (fun () ->
        Array.map
          (fun a ->
            Nx_cpu.unique_state
              [| a; Int64.logxor c (Nx_cpu.unique_state [| a |]) |])
          a)
  in
  equal ~msg:"states" int 1
    (List.length (List.sort_uniq Int64.compare (Array.to_list states)))

(* Rows of [w] words that a multiply and xor-shift step would hash alike
   under every seed: flipping a word's top bit changed that step's output by
   [d = 2^63 lxor 2^31] whatever the input, and the next word cancels [d].
   Row [f] flips word [i]'s top bit where bit [i] of [f] is set. Under nx.cpu's
   hash they leave distinct states. *)
let chain_rows (w, base, rows) =
  let top = Int64.min_int in
  let d = Int64.logor top 0x8000_0000L in
  List.init rows (fun f ->
      Array.init w (fun i ->
          let flip j = j >= 0 && j < w - 1 && (f lsr j) land 1 = 1 in
          let prev = if flip (i - 1) then d else 0L in
          let own = if flip i then top else 0L in
          Int64.logxor base.(i) (Int64.logxor prev own)))

let chains =
  let open Gen in
  with_pp
    (fun ppf ((w, _, rows), seed) ->
      Format.fprintf ppf "%d rows of %d words, seed %Ld" rows w seed)
    (let* w = int_range 2 20 in
     let* base = array ~size:(constant w) int64 in
     let* seed = int64 in
     let rows = Int.min 64 (1 lsl (w - 1)) in
     constant ((w, base, rows), seed))

let law_chain (((_, _, rows) as c), seed) =
  let states =
    under_seed seed (fun () -> List.map Nx_cpu.unique_state (chain_rows c))
  in
  equal ~msg:"distinct states" int rows
    (List.length (List.sort_uniq Int64.compare states))

(* On the pool a block has from 16,384 rows to all of them: the cases a
   direct table numbers whatever the blocks, those it numbers on one thread
   alone, and those it numbers on none. *)
let law_threads (b : Support.backend) c =
  let keys = keys_of c in
  let words = positions keys in
  let direct_one = direct c words
  and direct_blocks = direct ~rows:16_384 c words in
  cover "direct table on any blocks" direct_blocks;
  cover "direct table on one thread, hash table on the pool"
    (direct_one && not direct_blocks);
  cover "hash table on any blocks" (not direct_one);
  cover "keys of one hash" (c.draw = Collide);
  same_numbering
    (Nx_cpu.one_thread (fun () -> unique_on b c keys))
    (unique_on b c keys)

(* nx.cuda: keys whose sorts span several submissions of two columns' sorts
   each, the cut nx_cuda.mli states past 8 columns. *)
let spanned =
  [
    example Tied [| 3000; 3 |];
    example Spread [| 5000; 5 |] ~m:7
      ~view:{ (plain 2) with perm = [| 1; 0 |] };
    example Narrow [| 40; 6 |] ~view:{ (plain 2) with reversed = Some 0 };
  ]

(* The columns of its keys a numbering's submission sorts without
   Nx_cuda.numbering_words: 8, the most it takes. *)
let run_words = 8

(* [f ()] with a numbering's submissions sorting at most [n] columns. *)
let under_words n f =
  Fun.protect
    ~finally:(fun () -> Nx_cuda.numbering_words run_words)
    (fun () ->
      Nx_cuda.numbering_words n;
      f ())

(* nx_cuda.mli: a numbering takes a workspace of at most 75 bytes a row
   and a few kilobytes. One whose sorts span several submissions takes one
   of its own. *)
let test_loop_workspace (b : Support.backend) () =
  let module K = (val b.kernels) in
  let module A = Nx_array in
  let n = 1024 and d = b.device in
  let index s = A.create d A.Dtype.Int64 s in
  let keys = index [| n; 4 |] in
  let ids = index [| n |] and first = index [| n |] and counts = index [| n |] in
  let count = index [||] in
  under_words 1 (fun () ->
      Kernels_cases.loop_held b ~bound:((75 * n) + 65536) (fun () ->
          equal answer A.Done (K.unique ~ids ~first ~counts ~count keys);
          Rig.Buffer.wait (A.buffer count) Read_write))

(* The suite *)

let laws (d : Support.declared) =
  let run f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  group d.label
    [
      prop ~examples "numbers rows in order of first appearance" numberings
        (run law_order);
      prop ~count:4 ~examples:(many @ large)
        "numbers many rows in order of first appearance" threaded (run law_many);
      prop
        "a call of shapes Nx_kernel.Shape.unique refuses is not Done and \
         writes nothing outside its destinations"
        misfits (run law_misfit);
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f b x) in
  group ("nx.cpu " ^ b.name)
    [
      prop "gives the same bits under any seed"
        (Gen.pair numberings Gen.int64)
        (run law_seeds);
      prop "rows [|a; c lxor unique_state [|a|]|] of one c leave one state"
        leaders law_one_state;
      prop "rows that cancel a word's difference leave distinct hash states"
        chains law_chain;
      prop ~count:8 ~examples:large "gives one thread's bits on the pool"
        threaded (run law_threads);
    ]

let cuda (d : Support.declared) =
  let run f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  group ("nx.cuda " ^ d.label)
    [
      cases "numbers keys whose sorts span several submissions"
        ~id:(Format.asprintf "%a" pp_case)
        spanned
        (run (fun b c ->
             let keys = keys_of c in
             under_words 2 (fun () -> agrees b c keys (positions keys))));
      test "numberings in a loop hold one numbering's workspace" (fun () ->
          let b = Support.opened d in
          b.around (test_loop_workspace b));
    ]

(* rig.nv *)

(* The numbering's destinations, as arrays. *)
let numbered b c keys =
  let ints a =
    A.Any (A.of_array D.Int64 (Iarray.of_array [| Array.length a |]) a)
  in
  let ids, first, counts, count = unique_on b c keys in
  [| ints ids; ints first; ints counts; ints [| count |] |]

(* rig.nv loads the cubin CUDA loads, so a numbering through it has CUDA's
   bits: what it can get wrong, its launches, memory and order, shows in
   them. Each case runs through CUDA, then through rig.nv. *)
let nv_laws (first : Support.declared) (nv : Support.declared) =
  let agree c =
    let keys = keys_of c in
    Elements.agree (Support.opened first) (Support.opened nv) (fun b ->
        numbered b c keys)
  in
  group nv.label
    [
      group "bits are cuda's"
        [
          prop ~examples "of numberings" numberings agree;
          cases "of numberings whose sorts span several submissions"
            ~id:(Format.asprintf "%a" pp_case) spanned (fun c ->
              under_words 2 (fun () -> agree c));
        ];
    ]

let () =
  let nv =
    match (Support.declared_cuda, Support.declared_nv) with
    | Some first, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.unique"
       (List.map laws
          (Support.declare Support.cpu
          :: List.filter_map Fun.id
               [
                 Support.declared_cuda;
                 Support.declared_amd;
                 Support.declared_metal;
               ])
       @ [ cpu Support.cpu ]
       @ List.map cuda (Option.to_list Support.declared_cuda)
       @ nv))
