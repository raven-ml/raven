(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Decoding with a key-value cache: the five laws of attention.mli, Cache and
   cached. *)

open Windtrap
open Kaun

(* A grouped layer over 8 features: 4 query heads and 2 key-value heads of 2
   features, biased, with rotary positions. The layers are float64, so the
   tolerance below is far from every error a wrong read makes. *)

let head_dim = 2
let embed = 8
let rope = fixture (fun () -> Rope.make ~head_dim ~context:64 ())

let make_layer key =
  Attention.make ~key:(Nx.Rng.key key) ~bias_init:(Init.uniform ~scale:1.)
    ~kv_dim:4 ~embed_dim:embed Nx.float64

let layer = fixture (fun () -> make_layer 1)
let attention = Nx.Ptree.instantiate (module Attention)
let cache_tree = Nx.Ptree.instantiate (module Attention.Cache)

let cache slots =
  Attention.Cache.make ~slots ~kv_heads:2 ~head_dim Nx.float64 Nx.Host.on

let call p c index x = Attention.cached ~head_dim ~rope:(rope ()) p c index x

(* Features in [-1, 1), the range the tolerance assumes. *)
let draw shape = Nx.sub_s (Nx.mul_s (Nx.Rng.uniform Nx.float64 shape) 2.) 1.
let flat t = Nx.to_array (Nx.reshape [| -1 |] t)
let int64s shape a = Nx.create Nx.int64 shape (Array.map Int64.of_int a)
let ints l = String.concat ";" (List.map string_of_int l)

let matrix rows =
  int64s
    [| Array.length rows; Array.length rows.(0) |]
    (Array.concat (Array.to_list rows))

(* The index of tokens at [pos] in sequences held at the slots [table]. *)
let index_at ~pos ~table =
  Cache_index.make ~pos:(matrix pos) ~table:(matrix table) ()

let windowed window index =
  Option.fold ~none:index ~some:(fun w -> Cache_index.window w index) window

let pp_window ppf w =
  Format.pp_print_string ppf
    (Option.fold ~none:"no window" ~some:(Printf.sprintf "window %d") w)

(* Two float64 evaluations of one output that group its sums differently differ
   by at most [reassociation]. With features in [-1, 1), Glorot weights below
   0.71, biases in [0, 1) and u = 2^-53, a sum of n terms is within n u of the
   sum of their magnitudes (Higham, Accuracy and Stability of Numerical
   Algorithms, 2002, §3.1). So an evaluation is within 60 u of the exact q, k
   and v (sums of 9 terms below 6.7), within 160 u of their rotations (pairs of
   norm below 9.5), and within 5e3 u of the scores (2 products scaled by 1/√2,
   below 128). A shift d of the scores moves the softmax weights by 2 |d| in
   sum, and the weights round within 25 u each over at most 18 keys: 1e4 u. Each
   head's output, an average of values below 6.7, is then within 6.8e4 u, and
   the output projection, 8 terms through weights below 0.62, within 3.4e5 u.
   Two evaluations differ by twice that, 6.8e5 u, rounded up here. *)
let reassociation = 1e6 *. Float.epsilon /. 2.

let close ?msg expected actual =
  equal ?msg (array (float reassociation)) (flat expected) (flat actual)

(* The output projection of zero, which a padded token's output is. *)
let padded_output p =
  flat (Linear.apply p.Attention.out (Nx.zeros Nx.float64 [| 1; embed |]))

(* A whole pass of [x], one lane, under [window]. *)
let whole_pass ?window p x =
  let index = Cache_index.whole ~batch:1 ~seq:(Nx.dim 1 x) () in
  fst (call p (cache 0) (windowed window index) x)

(* The pieces of [cached] as the .mli composes them, with sinks and a score
   scale. *)
let composed ?scale ?sinks p c index x =
  let pos = Cache_index.positions index in
  let rope = rope () in
  let q = Rope.apply rope ~pos (Attention.split ~head_dim p.Attention.q x) in
  let k = Rope.apply rope ~pos (Attention.split ~head_dim p.k x) in
  let v = Attention.split ~head_dim p.v x in
  let kv, c = Attention.Cache.extend index c k v in
  let mask = Cache_index.visible index in
  (Attention.merge p.out (Attention.attend ~mask ?scale ?sinks q kv), c)

(* A decode step as a compiled call takes it: the tokens, the index and the
   cache. *)
type step = {
  x : Nx.host Nx.float64_t;
  index : Nx.host Cache_index.t;
  c : (Nx.host Nx.float64_t, Nx.host) Attention.Cache.t;
}

let step_tree =
  Nx.Ptree.v (fun cur (s : step) ->
      let open Nx.Ptree.Walk in
      let x = field cur "x" tensor s.x in
      let index = field cur "index" Cache_index.walk s.index in
      let c = field cur "c" (structure cache_tree) s.c in
      { x; index; c })

let step_of p { x; index; c } =
  let y, c = call p c index x in
  { x = y; index; c }

(* Law 1: chunking is invariant, against a model of each sequence's tokens and
   slots. A decoder holds three sequences of up to [context] tokens in a pool of
   [pool] slots, over rows (sequence [s] owns slots [s * context] on) or a paged
   table (a position takes the next slot of a drawn order when first fed), under
   a window or none. A call feeds a chunk to each sequence, padded on the left,
   feeds one sequence as one-token lanes, or lets an empty sequence share
   another's tokens by naming its slots. Each output is the whole pass of its
   sequence so far, a padded token's the output projection of zero, and after
   every call each slot holds what the whole feed of its sequence stores there,
   and a slot that holds no position zero. *)

let sequences = 3
and context = 6
and pool = 24
and vocab = 16

let vocabulary =
  fixture (fun () ->
      Nx.Rng.with_key (Nx.Rng.key 50) (fun () -> draw [| vocab; embed |]))

(* The features of token ids, lanes by tokens. *)
let embed_ids ids = Nx.take ~axis:0 (matrix ids) (vocabulary ())

type layout = Rows | Paged of int array

(* Position [j] of sequence [s] is at slot [slot.(s).(j)], [-1] where none is; a
   paged table has handed out [taken] slots of its order. *)
type table = { layout : layout; slot : int array array; mutable taken : int }

let table layout =
  let slot s j =
    match layout with Rows -> (s * context) + j | Paged _ -> -1
  in
  {
    layout;
    slot = Array.init sequences (fun s -> Array.init context (slot s));
    taken = 0;
  }

let allocate t s j =
  match t.layout with
  | Paged order when t.slot.(s).(j) < 0 ->
      t.slot.(s).(j) <- order.(t.taken);
      t.taken <- t.taken + 1
  | Rows | Paged _ -> ()

(* [dst] names the slots of [src]'s first [n] positions. *)
let share t ~src ~dst n = Array.blit t.slot.(src) 0 t.slot.(dst) 0 n

(* The tokens of [ids] that fit after [len] of them. *)
let fit len ids = List.filteri (fun i _ -> len + i < context) ids

(* What feeding [ids] whole gives: the outputs of a whole pass, and the keys and
   values that one call over rows stores, position [j] at slot [j]. *)
let whole_feed window ids =
  let n = List.length ids in
  if n = 0 then ([||], [||], [||])
  else
    let p = layer () and x = embed_ids [| Array.of_list ids |] in
    let _, c = call p (cache n) (Cache_index.rows ~context:n [| n |]) x in
    (flat (whole_pass ?window p x), flat c.keys, flat c.values)

type model = {
  window : int option;
  ids : int list array;
  slots : table;
  shared : bool array;
  whole : (float array * float array * float array) array;
}

type decoder = {
  window : int option;
  table : table;
  lens : int array;
  mutable pools : (Nx.host Nx.float64_t, Nx.host) Attention.Cache.t;
}

let start_model (layout, window) : model =
  cover "rows" (layout = Rows);
  cover "a paged table" (layout <> Rows);
  cover "a window" (window <> None);
  cover "no window" (window = None);
  {
    window;
    ids = Array.make sequences [];
    slots = table layout;
    shared = Array.make sequences false;
    whole = Array.make sequences ([||], [||], [||]);
  }

let start_system (layout, window) : decoder =
  {
    window;
    table = table layout;
    lens = Array.make sequences 0;
    pools = cache pool;
  }

(* Sequence [s] takes [ids] in the model: its slots, tokens and whole feed. *)
let append (m : model) s ids =
  let len = List.length m.ids.(s) in
  List.iteri (fun i _ -> allocate m.slots s (len + i)) ids;
  m.ids.(s) <- m.ids.(s) @ ids;
  m.whole.(s) <- whole_feed m.window m.ids.(s)

(* The last [n] outputs of sequence [s]'s whole pass, after [pad] padded
   ones. *)
let expected (m : model) s ~pad n =
  let outputs, _, _ = m.whole.(s) in
  let padded = padded_output (layer ()) in
  Array.append
    (Array.concat (List.init pad (fun _ -> padded)))
    (Array.sub outputs (Array.length outputs - (n * embed)) (n * embed))

let feed_model (m : model) chunks =
  let lens = Array.map List.length m.ids in
  let chunks = Array.mapi (fun s c -> fit lens.(s) c) chunks in
  let n = Array.map List.length chunks in
  let width = max 1 (Array.fold_left max 0 n) in
  let any f = Array.exists Fun.id (Array.init sequences f) in
  let fed = any (fun s -> n.(s) > 0) in
  cover "a lane of padding beside a fed lane" (fed && Array.mem 0 n);
  cover "a call of padding alone" (not fed);
  cover "a chunk after earlier tokens"
    (any (fun s -> lens.(s) > 0 && n.(s) >= 2));
  cover "a sequence fed to its context"
    (any (fun s -> n.(s) > 0 && lens.(s) + n.(s) = context));
  cover "a sequence fed past a shared prefix"
    (any (fun s -> n.(s) > 0 && m.shared.(s)));
  Array.iteri (fun s c -> if c <> [] then append m s c) chunks;
  Array.concat
    (List.init sequences (fun s -> expected m s ~pad:(width - n.(s)) n.(s)))

(* One call over [d]'s pools under [d]'s window. *)
let decode (d : decoder) index ids =
  let y, pools =
    call (layer ()) d.pools (windowed d.window index) (embed_ids ids)
  in
  d.pools <- pools;
  flat y

(* Sequence [s] takes positions [len] to [len + n - 1]. *)
let advance (d : decoder) s n =
  for j = d.lens.(s) to d.lens.(s) + n - 1 do
    allocate d.table s j
  done;
  d.lens.(s) <- d.lens.(s) + n

let feed_system (d : decoder) chunks =
  let chunks =
    Array.mapi (fun s c -> Array.of_list (fit d.lens.(s) c)) chunks
  in
  let n = Array.map Array.length chunks in
  let width = max 1 (Array.fold_left max 0 n) in
  let pos =
    Array.init sequences (fun s ->
        let pad = width - n.(s) in
        Array.init width (fun i -> if i < pad then -1 else d.lens.(s) + i - pad))
  in
  let ids =
    Array.init sequences (fun s ->
        Array.append (Array.make (width - n.(s)) 0) chunks.(s))
  in
  Array.iteri (advance d) n;
  decode d (index_at ~pos ~table:d.table.slot) ids

let lanes_model (m : model) (s, ids) =
  let len = List.length m.ids.(s) in
  let ids = fit len ids in
  cover "lanes of a sequence that holds earlier tokens" (len > 0);
  cover "several lanes of one sequence" (List.length ids >= 2);
  append m s ids;
  expected m s ~pad:0 (List.length ids)

let lanes_system (d : decoder) (s, ids) =
  let ids = Array.of_list (fit d.lens.(s) ids) in
  let n = Array.length ids in
  let pos = Array.init n (fun i -> [| d.lens.(s) + i |]) in
  advance d s n;
  let index =
    Cache_index.make
      ~row:(int64s [| n |] (Array.make n s))
      ~pos:(matrix pos) ~table:(matrix d.table.slot) ()
  in
  decode d index (Array.map (fun i -> [| i |]) ids)

let share_model (m : model) (src, dst) k =
  let k = min k (List.length m.ids.(src)) in
  share m.slots ~src ~dst k;
  m.ids.(dst) <- List.filteri (fun i _ -> i < k) m.ids.(src);
  m.whole.(dst) <- whole_feed m.window m.ids.(dst);
  m.shared.(src) <- true;
  m.shared.(dst) <- true

let share_system (d : decoder) (src, dst) k =
  let k = min k d.lens.(src) in
  share d.table ~src ~dst k;
  d.lens.(dst) <- k

(* Each slot holds what the whole feed of its sequence stores at the position it
   holds, and a slot that holds no position is zero. *)
let stores_whole_feeds (m : model) (d : decoder) =
  let width = 2 * head_dim in
  let keys = Array.make (pool * width) 0. in
  let values = Array.copy keys in
  Array.iteri
    (fun s ids ->
      let _, k, v = m.whole.(s) in
      List.iteri
        (fun j _ ->
          let slot = m.slots.slot.(s).(j) in
          Array.blit k (j * width) keys (slot * width) width;
          Array.blit v (j * width) values (slot * width) width)
        ids)
    m.ids;
  let near = array (float reassociation) in
  equal ~msg:"keys" near keys (flat d.pools.keys);
  equal ~msg:"values" near values (flat d.pools.values)

let decoder = abstract ~invariant:stores_whole_feeds "d"
let id = Gen.int_range 0 (vocab - 1)
let sequence = Gen.int_range 0 (sequences - 1)

let window_gen =
  Gen.(frequency [ (1, constant None); (2, map Option.some (int_range 1 4)) ])

(* The key a property draws its values from. *)
let key_gen = Gen.int_range 0 1_000_000

let config =
  let layout =
    Gen.(
      one_of
        [
          constant Rows;
          map
            (fun o -> Paged (Array.of_list o))
            (permutation (List.init pool Fun.id));
        ])
  in
  Gen.with_pp
    (fun ppf (layout, window) ->
      Format.fprintf ppf "%s, %a"
        (match layout with
        | Rows -> "rows"
        | Paged o -> "paged " ^ ints (Array.to_list o))
        pp_window window)
    (Gen.pair layout window_gen)

let chunks =
  let size = Gen.(frequency [ (2, constant 0); (3, int_range 1 4) ]) in
  Gen.with_pp
    (fun ppf c ->
      Format.pp_print_string ppf
        (String.concat " | " (Array.to_list (Array.map ints c))))
    Gen.(array ~size:(constant sequences) (list ~size id))

let lane_tokens =
  Gen.with_pp
    (fun ppf (s, ids) -> Format.fprintf ppf "sequence %d: %s" s (ints ids))
    Gen.(pair sequence (list ~size:(int_range 1 3) id))

(* An empty sequence and one that holds tokens. *)
let empty_and_held =
  among (pair int int) decoder (fun (m : model) ->
      let held s = m.ids.(s) <> [] in
      List.concat_map
        (fun dst ->
          if held dst then []
          else
            List.filter_map
              (fun src -> if held src then Some (src, dst) else None)
              (List.init sequences Fun.id))
        (List.init sequences Fun.id))

let prefix = Gen.int_range 1 (context - 1)
let outputs = array (float reassociation)

let decoding =
  [
    command "start" (config @-> makes decoder) start_model start_system;
    command "feed"
      (decoder ^-> chunks @-> returns outputs)
      feed_model feed_system;
    command "lanes"
      ~pre:(fun (m : model) (s, _) -> List.length m.ids.(s) < context)
      (decoder ^-> lane_tokens @-> returns outputs)
      lanes_model lanes_system;
    command "share"
      (decoder ^-> empty_and_held ^-> prefix @-> returns unit)
      share_model share_system;
  ]

(* Law 2: a draft is its paths. A tree of draft tokens is verified in one call:
   token [i] continues token [parents.(i)], token 0 the cache. *)

let ancestors parents =
  let rec path i = if i < 0 then [] else i :: path parents.(i) in
  Array.init (Array.length parents) (fun i -> List.rev (path i))

let depths parents =
  Array.map (fun path -> List.length path - 1) (ancestors parents)

(* [sees.(i).(j)]: token [j] is token [i] or one of its ancestors. *)
let tree_sees parents =
  let n = Array.length parents in
  let paths = ancestors parents in
  Nx.create Nx.bool [| 1; n; n |]
    (Array.init (n * n) (fun t -> List.mem (t mod n) paths.(t / n)))

(* One sequence over a table of 12 columns whose slots are shuffled; draft
   tokens store at slots 12 and up, which no table names. *)
let draft_table = [| 7; 2; 11; 0; 5; 9; 1; 10; 3; 6; 4; 8 |]
let draft_slot i = 19 - i

(* The cache after a prefix of [p] tokens, with NaN in the slots of the columns
   past it: a draft must not read them, and sequential decoding writes them
   before it reads them. *)
let after_prefix ~window layer x p =
  let c =
    if p = 0 then cache 20
    else
      let index =
        index_at ~pos:[| Array.init p Fun.id |] ~table:[| draft_table |]
      in
      snd
        (call layer (cache 20) (windowed window index)
           (Nx.slice [ A; R (0, p) ] x))
  in
  let nan = Nx.full Nx.float64 [| 1; 2; head_dim |] Float.nan in
  let poison t s = Nx.set [ Nx.R (s, s + 1) ] nan t in
  let unwritten = Array.sub draft_table p (12 - p) in
  {
    Attention.Cache.keys = Array.fold_left poison c.keys unwritten;
    values = Array.fold_left poison c.values unwritten;
  }

(* The outputs of [tokens] fed one by one from position [p] through [table]. *)
let sequential ~window layer c table ~p tokens =
  let ys, _ =
    List.fold_left
      (fun (ys, (c, at)) x ->
        let index =
          windowed window (index_at ~pos:[| [| at |] |] ~table:[| table |])
        in
        let y, c = call layer c index x in
        (y :: ys, (c, at + 1)))
      ([], (c, p))
      tokens
  in
  List.rev ys

(* A draft of [parents] after a prefix of [p] tokens gives each token the output
   of decoding its path alone, read off each leaf's path; keeping the path to
   token [kept], the next token's output is the one after that path. *)
let law_draft (p, (parents, kept), window, key) =
  let n = Array.length parents in
  let children i =
    Array.fold_left (fun k q -> if q = i then k + 1 else k) 0 parents
  in
  let chain = Array.for_all (fun i -> children i <= 1) (Array.init n Fun.id) in
  cover "one token" (n = 1);
  cover "a chain of several tokens" (n > 1 && chain);
  cover "a branching tree" (not chain);
  cover "keeping a token with children" (children kept > 0);
  cover "no prefix" (p = 0);
  cover "a window" (window <> None);
  cover "no window" (window = None);
  Nx.Rng.with_key (Nx.Rng.key key) @@ fun () ->
  let layer = layer () in
  let x = draw [| 1; max p 1; embed |] in
  let tokens = Array.init n (fun _ -> draw [| 1; 1; embed |]) in
  let bonus = draw [| 1; 1; embed |] in
  let c = after_prefix ~window layer x p in
  let depth = depths parents and paths = ancestors parents in
  let index =
    windowed window
      (Cache_index.draft
         ~slots:(int64s [| 1; n |] (Array.init n draft_slot))
         ~sees:(tree_sees parents)
         (index_at
            ~pos:[| Array.map (fun d -> p + d) depth |]
            ~table:[| draft_table |]))
  in
  let y, c' =
    call layer c index (Nx.concatenate ~axis:1 (Array.to_list tokens))
  in
  let along path = List.map (fun j -> tokens.(j)) path in
  Array.iteri
    (fun leaf path ->
      if children leaf = 0 then
        List.iter2
          (fun j alone ->
            close
              ~msg:(Printf.sprintf "token %d is its path decoded alone" j)
              alone
              (Nx.slice [ A; R (j, j + 1) ] y))
          path
          (sequential ~window layer c draft_table ~p (along path)))
    paths;
  (* Keeping a path names its slots at the columns of its positions. *)
  let table = Array.copy draft_table in
  List.iteri (fun d j -> table.(p + d) <- draft_slot j) paths.(kept);
  let next = p + depth.(kept) + 1 in
  let y_next, _ =
    call layer c'
      (windowed window (index_at ~pos:[| [| next |] |] ~table:[| table |]))
      bonus
  in
  let expected =
    List.nth
      (sequential ~window layer c draft_table ~p
         (along paths.(kept) @ [ bonus ]))
      (depth.(kept) + 1)
  in
  close ~msg:"after keeping a path, the next token follows it" expected y_next

(* Token [i > 0] continues a token before it; one token a quarter of the
   time. *)
let tree_gen =
  Gen.(
    let* n = frequency [ (1, constant 1); (3, int_range 2 6) ] in
    let+ picks = list ~size:(constant n) (int_range 0 1000) in
    let picks = Array.of_list picks in
    ( Array.init n (fun i -> if i = 0 then -1 else picks.(i) mod i),
      picks.(0) mod n ))

let draft_gen =
  Gen.with_pp
    (fun ppf (p, (parents, kept), window, key) ->
      Format.fprintf ppf "prefix %d, parents [%s], kept %d, %a, key %d" p
        (ints (Array.to_list parents))
        kept pp_window window key)
    Gen.(quad (int_range 0 4) tree_gen window_gen key_gen)

(* Law 3: packing is invariant. Two lanes of 9 tokens, each holding up to three
   sequences of up to three tokens. *)
let law_packing (lanes, window, key) =
  let lens = Array.of_list (List.map Array.of_list lanes) in
  let sum = Array.fold_left ( + ) 0 in
  cover "several sequences in a lane"
    (Array.exists (fun l -> Array.length l > 1) lens);
  cover "an empty sequence" (Array.exists (Array.mem 0) lens);
  cover "a one-token sequence" (Array.exists (Array.mem 1) lens);
  cover "a lane of padding" (Array.exists (fun l -> sum l = 0) lens);
  cover "a window" (window <> None);
  cover "no window" (window = None);
  Nx.Rng.with_key (Nx.Rng.key key) @@ fun () ->
  let layer = layer () in
  let x = draw [| 2; 9; embed |] in
  let y, _ =
    call layer (cache 0) (windowed window (Cache_index.packed ~seq:9 lens)) x
  in
  Array.iteri
    (fun b lens ->
      let at = ref (9 - sum lens) in
      Array.iter
        (fun n ->
          if n > 0 then begin
            let part t = Nx.slice [ R (b, b + 1); R (!at, !at + n) ] t in
            close
              ~msg:(Printf.sprintf "lane %d, the sequence at %d" b !at)
              (whole_pass ?window layer (part x))
              (part y)
          end;
          at := !at + n)
        lens)
    lens

let packing_gen =
  let lens = Gen.(list ~size:(int_range 0 3) (int_range 0 3)) in
  Gen.with_pp
    (fun ppf (lanes, window, key) ->
      Format.fprintf ppf "%s, %a, key %d"
        (String.concat " | " (List.map ints lanes))
        pp_window window key)
    Gen.(triple (list ~size:(constant 2) lens) window_gen key_gen)

(* Law 4: the whole-sequence pass is the cached pass. Lanes of [lens] tokens
   padded on the left to [seq], and rows of [seq + extra] slots each. *)
let law_whole (seq, lens, extra, key) =
  let batch = Array.length lens in
  cover "a lane of padding" (Array.mem 0 lens);
  cover "a padded lane" (Array.exists (fun n -> n > 0 && n < seq) lens);
  cover "one token" (seq = 1);
  Nx.Rng.with_key (Nx.Rng.key key) @@ fun () ->
  let p = layer () and c = cache 0 in
  let x = draw [| batch; seq; embed |] in
  let y, c' = call p c (Cache_index.whole ~lens ~batch ~seq ()) x in
  Array.iteri
    (fun b n ->
      let lane t = Nx.slice [ R (b, b + 1); R (seq - n, seq) ] t in
      if n > 0 then
        close
          ~msg:(Printf.sprintf "lane %d is causal attention over itself" b)
          (Attention.apply ~head_dim ~mask:Nx.causal ~rope:(rope ()) p (lane x))
          (lane y))
    lens;
  satisfies ~msg:"the cache is returned as given" ~claim:"the given pools" pass
    (fun (c' : _ Attention.Cache.t) ->
      c'.keys == c.keys && c'.values == c.values)
    c';
  let context = seq + extra in
  let rows, _ =
    call p (cache (batch * context)) (Cache_index.rows ~context lens) x
  in
  close ~msg:"over rows and a fresh cache" y rows

let whole_gen =
  Gen.with_pp
    (fun ppf (seq, lens, extra, key) ->
      Format.fprintf ppf "seq %d, lens [%s], context seq + %d, key %d" seq
        (ints (Array.to_list lens))
        extra key)
    Gen.(
      let* longest = int_range 0 4 in
      let* others = list ~size:(int_range 0 2) (int_range 0 longest) in
      let* extra = int_range 0 2 in
      let+ key = key_gen in
      (max 1 longest, Array.of_list (longest :: others), extra, key))

(* Law 5: one tensor per cache leaf. Two blocks' caches hold one pool of a
   two-token prefix, donated to one compiled step: each block reads the prefix
   as it was and writes its own token at slot 2. *)
let test_one_tensor_per_leaf () =
  Compiled.here ();
  Nx.Rng.with_key (Nx.Rng.key 44) @@ fun () ->
  let blocks = [ layer (); make_layer 2 ] in
  let x = draw [| 1; 3; embed |] in
  let _, c =
    call (layer ()) (cache 4)
      (Cache_index.rows ~context:4 [| 2 |])
      (Nx.slice [ A; R (0, 2) ] x)
  in
  let index = index_at ~pos:[| [| 2 |] |] ~table:[| [| 0; 1; 2; 3 |] |] in
  let next = Nx.slice [ A; R (2, 3) ] x in
  let eager = List.map (fun p -> call p c index next) blocks in
  let caches = Nx.Ptree.list cache_tree in
  let step =
    Rune.jit
      Nx.Ptree.(caches @-> returns (list (pair tensor cache_tree)))
      (List.map2 (fun p c -> call p c index next) blocks)
  in
  let compiled = step (Nx.Ptree.donate caches [ c; c ]) in
  List.iteri
    (fun i ((y, (c : _ Attention.Cache.t)), (y', (c' : _ Attention.Cache.t))) ->
      let msg what = Printf.sprintf "block %d, %s" i what in
      close ~msg:(msg "outputs") y y';
      close ~msg:(msg "keys") c.keys c'.keys;
      close ~msg:(msg "values") c.values c'.values)
    (List.combine eager compiled)

(* Cache.make *)

let test_make_zeros () =
  let c = cache 3 in
  let zeros = Array.make 12 0. in
  let pool = pair (array int) (array float_exact) in
  equal ~msg:"keys" pool ([| 3; 2; 2 |], zeros) (Nx.shape c.keys, flat c.keys);
  equal ~msg:"values" pool
    ([| 3; 2; 2 |], zeros)
    (Nx.shape c.values, flat c.values)

let two_devices =
  fixture (fun () ->
      Nx.devices
        (List.init 2 (fun i ->
             Result.get_ok (Rig.software (Printf.sprintf "kaun-cache-%d" i)))))

let test_make_at () =
  let module Two = (val two_devices ()) in
  let at = Two.split ~axis:1 in
  let c = Attention.Cache.make ~slots:4 ~kv_heads:2 ~head_dim Nx.float64 at in
  let placement =
    Testable.make ~pp:Nx.Placement.pp ~equal:(fun a b -> Nx.Placement.equal a b)
  in
  equal ~msg:"keys" placement at (Nx.placement c.keys);
  equal ~msg:"values" placement at (Nx.placement c.values)

let test_make_no_slot () =
  equal (array int) [| 0; 2; 2 |] (Nx.shape (cache 0).keys)

let make_refusals =
  [ ("slots -1", -1, 2, 2); ("kv_heads 0", 4, 0, 2); ("head_dim 0", 4, 2, 0) ]

let test_make_raises (_, slots, kv_heads, head_dim) =
  raises_match (Exn.invalid_arg ~sub:"Attention.Cache.make") (fun () ->
      Attention.Cache.make ~slots ~kv_heads ~head_dim Nx.float64 Nx.Host.on)

(* Cache.walk *)

let paths s x =
  List.rev
    (Nx.Ptree.fold s
       (fun path _ acc -> Nx.Ptree.Path.to_string path :: acc)
       x [])

let test_walk_paths () =
  equal ~msg:"one cache" (list string) [ "keys"; "values" ]
    (paths cache_tree (cache 2));
  equal ~msg:"a decoder's caches" (list string)
    [ "0.keys"; "0.values"; "1.keys"; "1.values" ]
    (paths (Nx.Ptree.list cache_tree) [ cache 2; cache 2 ])

(* Cache.extend *)

(* Keys and values of two lanes of three tokens, and a cache of 8 slots with
   drawn pools. *)
let extend_inputs () =
  Nx.Rng.with_key (Nx.Rng.key 45) @@ fun () ->
  let kv () = draw [| 2; 2; 3; head_dim |] in
  let pool () = draw [| 8; 2; head_dim |] in
  (kv (), kv (), { Attention.Cache.keys = pool (); values = pool () })

let bits t = Array.map Int64.bits_of_float (flat t)
let rows () = Cache_index.rows ~context:4 [| 3; 2 |]

let test_extend_whole () =
  let k, v, c = extend_inputs () in
  let kv, c' =
    Attention.Cache.extend (Cache_index.whole ~batch:2 ~seq:3 ()) c k v
  in
  (match kv with
  | Nx.Dense d ->
      equal ~msg:"k" (array int64) (bits k) (bits d.k);
      equal ~msg:"v" (array int64) (bits v) (bits d.v)
  | Nx.Paged _ -> fail "a whole index gives Paged keys");
  satisfies ~msg:"the cache" ~claim:"the given pools" pass
    (fun (c' : _ Attention.Cache.t) ->
      c'.keys == c.keys && c'.values == c.values)
    c'

let test_extend_paged () =
  let k, v, c = extend_inputs () in
  let index = rows () in
  let kv, c' = Attention.Cache.extend index c k v in
  let store pool t = Cache_index.store index (Nx.swapaxes 1 2 t) pool in
  equal ~msg:"keys stored" (array int64) (bits (store c.keys k)) (bits c'.keys);
  equal ~msg:"values stored" (array int64)
    (bits (store c.values v))
    (bits c'.values);
  let view = pair (array int) (array int64) in
  match kv with
  | Nx.Paged p ->
      equal ~msg:"the table is the pages" (array int64)
        (Nx.to_array (Option.get (Cache_index.pages index)))
        (Nx.to_array p.table);
      equal ~msg:"k views the keys" view
        ([| 8; 2; 1; head_dim |], bits c'.keys)
        (Nx.shape p.k, bits p.k);
      equal ~msg:"v views the values" view
        ([| 8; 2; 1; head_dim |], bits c'.values)
        (Nx.shape p.v, bits p.v)
  | Nx.Dense _ -> fail "an index with pages gives Dense keys"

let test_extend_dense () =
  let k, v, c = extend_inputs () in
  let columns = int64s [| 2; 3; 2 |] [| 0; 1; 1; 2; 2; 0; 0; 0; 1; 1; 1; 0 |] in
  let index = Cache_index.select columns (rows ()) in
  let kv, c' = Attention.Cache.extend index c k v in
  let through pool t =
    let seen, pool = Cache_index.extend index (Nx.swapaxes 1 2 t) pool in
    (Nx.swapaxes 1 2 seen, pool)
  in
  let k_seen, keys = through c.keys k and v_seen, values = through c.values v in
  equal ~msg:"keys stored" (array int64) (bits keys) (bits c'.keys);
  equal ~msg:"values stored" (array int64) (bits values) (bits c'.values);
  match kv with
  | Nx.Dense d ->
      equal ~msg:"k" (array int64) (bits k_seen) (bits d.k);
      equal ~msg:"v" (array int64) (bits v_seen) (bits d.v)
  | Nx.Paged _ -> fail "a selection gives Paged keys"

let extend_refusals =
  let first_two t = Nx.slice [ A; A; R (0, 2) ] t in
  let two_axes t = Nx.reshape [| 8; 2 * head_dim |] t in
  [
    ( "k of another shape",
      fun k v c -> Attention.Cache.extend (rows ()) c (first_two k) v );
    ( "v of another shape",
      fun k v c -> Attention.Cache.extend (rows ()) c k (first_two v) );
    ( "pools of two axes",
      fun k v (c : _ Attention.Cache.t) ->
        Attention.Cache.extend (rows ())
          { Attention.Cache.keys = two_axes c.keys; values = two_axes c.values }
          k v );
  ]

let test_extend_raises (_, extend) =
  let k, v, c = extend_inputs () in
  raises_match Exn.invalid_arg (fun () -> extend k v c)

(* cached *)

let test_cached_functional () =
  Nx.Rng.with_key (Nx.Rng.key 46) @@ fun () ->
  let pool () = draw [| 4; 2; head_dim |] in
  let c = { Attention.Cache.keys = pool (); values = pool () } in
  let keys = flat c.keys and values = flat c.values in
  ignore
    (call (layer ()) c
       (Cache_index.rows ~context:4 [| 2 |])
       (draw [| 1; 2; embed |]));
  equal ~msg:"keys" (array float_exact) keys (flat c.keys);
  equal ~msg:"values" (array float_exact) values (flat c.values)

(* Under a window of [w] a token of 6 sees the last [w] positions at or before
   its own: the band of a captured mask. *)
let window_cases = [ 1; 3; 6; max_int ]

let test_cached_window w =
  Nx.Rng.with_key (Nx.Rng.key 47) @@ fun () ->
  let p = layer () in
  let x = draw [| 1; 6; embed |] in
  let band =
    Nx.create Nx.bool [| 6; 6 |]
      (Array.init 36 (fun t ->
           let i = t / 6 and j = t mod 6 in
           j <= i && j > i - w))
  in
  close
    (Attention.apply ~head_dim ~mask:(fun _ -> band) ~rope:(rope ()) p x)
    (whole_pass ~window:w p x)

(* Lane 0 holds one token after two padded ones. *)
let padded_cases =
  [
    ("over a whole index", Cache_index.whole ~lens:[| 1; 3 |] ~batch:2 ~seq:3);
    ("through a table", fun () -> Cache_index.rows ~context:4 [| 1; 3 |]);
  ]

let test_cached_padded (_, index) =
  Nx.Rng.with_key (Nx.Rng.key 48) @@ fun () ->
  let p = layer () in
  let y, _ = call p (cache 8) (index ()) (draw [| 2; 3; embed |]) in
  let padded = padded_output p in
  equal (array float_exact)
    (Array.append padded padded)
    (flat (Nx.slice [ R (0, 1); R (0, 2) ] y))

(* A hole, an unallocated column or one naming a slot outside the pool, that a
   token may see is no key: the outputs are attention over the columns the token
   sees, without the hole, from the keys Cache_index.extend reads. *)
let test_cached_hole () =
  Compiled.here ();
  Nx.Rng.with_key (Nx.Rng.key 61) @@ fun () ->
  let p = layer () in
  let x = draw [| 1; 2; embed |] in
  let pool () = draw [| 6; 2; head_dim |] in
  let c = { Attention.Cache.keys = pool (); values = pool () } in
  let table = [| 4; -1; 99; 5; 1 |] in
  let index = index_at ~pos:[| [| 0; 3 |] |] ~table:[| table |] in
  let pos = Cache_index.positions index in
  let q = Rope.apply (rope ()) ~pos (Attention.split ~head_dim p.q x) in
  let k = Rope.apply (rope ()) ~pos (Attention.split ~head_dim p.k x) in
  let v = Attention.split ~head_dim p.v x in
  let read t pool =
    Nx.swapaxes 1 2 (fst (Cache_index.extend index (Nx.swapaxes 1 2 t) pool))
  in
  let allocated =
    Nx.create Nx.bool [| 1; 1; 1; 5 |]
      (Array.map (fun t -> t >= 0 && t < 6) table)
  in
  let mask =
    Nx.logical_and allocated
      (Nx.reshape [| 1; 1; 2; 5 |] (Cache_index.mask index))
  in
  let expected =
    Attention.merge p.out
      (Attention.attend
         ~mask:(fun _ -> mask)
         q
         (Nx.Dense { k = read k c.keys; v = read v c.values }))
  in
  close ~msg:"eager" expected (fst (call p c index x));
  let compiled =
    Rune.jit Nx.Ptree.(step_tree @-> returns step_tree) (step_of p)
  in
  close ~msg:"compiled" expected (compiled { x; index; c }).x

(* A draft token that stores nothing, at [-1] or outside the pool, is seen by no
   token, itself included: the outputs are those of the draft whose [sees] hides
   it from every token. *)
let test_cached_draft_storing_nothing () =
  Compiled.here ();
  Nx.Rng.with_key (Nx.Rng.key 62) @@ fun () ->
  let p = layer () in
  let c = after_prefix ~window:None p (draw [| 1; 3; embed |]) 3 in
  let x = draw [| 1; 3; embed |] in
  let table = index_at ~pos:[| [| 3; 4; 5 |] |] ~table:[| draft_table |] in
  let chain = tree_sees [| -1; 0; 1 |] in
  let unseen =
    Nx.logical_and chain
      (Nx.create Nx.bool [| 1; 1; 3 |] [| true; false; true |])
  in
  let draft slot sees =
    Cache_index.draft ~slots:(int64s [| 1; 3 |] [| 12; slot; 14 |]) ~sees table
  in
  let compiled =
    Rune.jit Nx.Ptree.(step_tree @-> returns step_tree) (step_of p)
  in
  List.iter
    (fun slot ->
      let msg what = Printf.sprintf "slot %d, %s" slot what in
      let expected, _ = call p c (draft 13 unseen) x in
      close ~msg:(msg "eager") expected (fst (call p c (draft slot chain) x));
      close ~msg:(msg "compiled") expected
        (compiled { x; index = draft slot chain; c }).x)
    [ -1; 99 ]

(* Two lanes of three tokens over tables of six columns and a pool of 24 slots.
   Lane [b]'s table names slots [8b] to [8b + 7] in some order, with holes
   ([-1]) and slots outside the pool; under a draft, token [i] of lane [b]
   stores at slot [16 + 4b + i], or at [-1] or outside the pool. *)
type paged = {
  index : Nx.host Cache_index.t;
  table : int array array;
  pos : int array array;
  window : int option;
  draft : int array array option;
  sees : Nx.host Nx.bool_t option;
  key : int;
}

let paged_index ~table ~pos ~window ~draft ~sees =
  let index = index_at ~pos ~table in
  let index =
    match (draft, sees) with
    | Some slots, Some sees ->
        Cache_index.draft ~slots:(matrix slots) ~sees index
    | _ -> index
  in
  windowed window index

let paged_width = 6
and paged_slots = 24

let pp_paged ppf (s : paged) =
  let lanes a =
    String.concat " | "
      (Array.to_list (Array.map (fun a -> ints (Array.to_list a)) a))
  in
  Format.fprintf ppf "table %s pos %s %a draft %s key %d" (lanes s.table)
    (lanes s.pos) pp_window s.window
    (Option.fold ~none:"none" ~some:lanes s.draft)
    s.key

(* Sequence [b]'s table: slots [8b] to [8b + 7] in some order, with holes. *)
let lane_table b =
  let open Gen in
  let* order = permutation (List.init 8 (fun i -> (8 * b) + i)) in
  let+ entries =
    list ~size:(constant paged_width)
      (frequency [ (6, constant false); (2, constant true); (1, bool) ])
  in
  let order = Array.of_list order in
  Array.of_list
    (List.mapi
       (fun j bad ->
         if not bad then order.(j) else if j mod 2 = 0 then -1 else 99)
       entries)

let window_of has_window =
  Gen.(if has_window then map Option.some (int_range 1 4) else constant None)

let paged_of ~has_window ~drafted =
  let open Gen in
  let* t0 = lane_table 0 in
  let* t1 = lane_table 1 in
  let* window = window_of has_window in
  let* key = key_gen in
  let+ pos, draft, sees =
    if not drafted then
      (* [n] tokens from [p0], padded on the left, within the table. *)
      let lane =
        let* n = frequency [ (1, constant 0); (6, int_range 1 3) ] in
        let+ p0 = int_range 0 (paged_width - max n 1) in
        Array.init 3 (fun i -> if i < 3 - n then -1 else p0 + i - (3 - n))
      in
      let+ l0 = lane and+ l1 = lane in
      ([| l0; l1 |], None, None)
    else
      (* A tree of three tokens from the lane's least position [p0], or of two
         after a token of padding. *)
      let lane b =
        let* p0 = int_range 0 paged_width in
        let* p2 = int_range 0 1 in
        let* padded = frequency [ (3, constant false); (1, constant true) ] in
        let parents = if padded then [| -1; -1; 1 |] else [| -1; 0; p2 |] in
        let+ slots =
          list ~size:(constant 3)
            (frequency [ (4, constant true); (1, constant false); (1, bool) ])
        in
        let slots =
          Array.of_list
            (List.mapi
               (fun i ok ->
                 if ok then 16 + (4 * b) + i else if i = 1 then -1 else 99)
               slots)
        in
        let pos = Array.map (fun d -> p0 + d) (depths parents) in
        if padded then pos.(0) <- -1;
        (pos, slots, parents)
      in
      let+ p0, s0, a0 = lane 0 and+ p1, s1, a1 = lane 1 in
      let sees = Nx.concatenate ~axis:0 [ tree_sees a0; tree_sees a1 ] in
      ([| p0; p1 |], Some [| s0; s1 |], Some sees)
  in
  let table = [| t0; t1 |] in
  let index = paged_index ~table ~pos ~window ~draft ~sees in
  { index; table; pos; window; draft; sees; key }

let paged_gen =
  Gen.(
    let* has_window = bool in
    let* drafted = bool in
    paged_of ~has_window ~drafted)

(* The pages the .mli states for [s], a row per lane: the lane's table at each
   column that no rule hides, the draft's slots after it. *)
let expected_pages (s : paged) =
  Array.init 2 (fun b ->
      let real = List.filter (fun p -> p >= 0) (Array.to_list s.pos.(b)) in
      let last = List.fold_left max (-1) real in
      let least = List.fold_left min max_int real in
      let upto = if s.draft = None then last else min (least - 1) last in
      let below j =
        match s.window with Some w -> j <= least - w | None -> false
      in
      let table =
        Array.mapi
          (fun j t -> if j > upto || below j then -1 else t)
          s.table.(b)
      in
      match s.draft with
      | None -> table
      | Some slots ->
          Array.append table
            (Array.mapi
               (fun i slot -> if s.pos.(b).(i) >= 0 then slot else -1)
               slots.(b)))

(* Whether a token of [s] may see a hole: a column its lane's pages keep whose
   table entry is unallocated or outside the pool. *)
let sees_a_hole (s : paged) =
  let allocated = Array.map (Array.map (fun _ -> 0)) s.table in
  let live = expected_pages { s with table = allocated } in
  let hole t = t < 0 || t >= paged_slots in
  Array.exists Fun.id
    (Array.init (2 * paged_width) (fun c ->
         let b = c / paged_width and j = c mod paged_width in
         live.(b).(j) = 0 && hole s.table.(b).(j)))

let paged_step =
  lazy
    (let step = step_of (layer ()) in
     (step, Rune.jit Nx.Ptree.(step_tree @-> returns step_tree) step))

(* A slot that no entry of the pages names is never read: NaN there changes no
   output, eager or compiled, and the compiled step gives eager's outputs. *)
let law_paged_reads (s : paged) =
  Compiled.here ();
  let outside t = t < 0 || t >= paged_slots in
  cover "a draft" (s.draft <> None);
  cover "a window" (s.window <> None);
  cover "a hole a token may see" (sees_a_hole s);
  cover "a draft token that stores nothing"
    (Option.fold ~none:false
       ~some:(Array.exists (Array.exists outside))
       s.draft);
  Nx.Rng.with_key (Nx.Rng.key s.key) @@ fun () ->
  let step, compiled = Lazy.force paged_step in
  let read = Array.concat (Array.to_list (expected_pages s)) in
  let pool () = draw [| paged_slots; 2; head_dim |] in
  let c = { Attention.Cache.keys = pool (); values = pool () } in
  let unread = Nx.full Nx.float64 [| 1; 2; head_dim |] Float.nan in
  let poison t =
    List.fold_left
      (fun t slot ->
        if Array.mem slot read then t
        else Nx.set [ Nx.R (slot, slot + 1) ] unread t)
      t
      (List.init paged_slots Fun.id)
  in
  let poisoned =
    { Attention.Cache.keys = poison c.keys; values = poison c.values }
  in
  let x = draw [| 2; 3; embed |] in
  let run f c = (f { x; index = s.index; c }).x in
  let eager = run step c in
  equal ~msg:"eager" (array float_exact) (flat eager) (flat (run step poisoned));
  let jitted = run compiled c in
  equal ~msg:"compiled" (array float_exact) (flat jitted)
    (flat (run compiled poisoned));
  close ~msg:"compiled = eager" eager jitted

(* Cache_index.extend's reads under Cache_index.mask give the outputs of
   cached's read through the pages, holes and draft tokens that store nothing
   included. Slots outside the pool are drawn as [-1], since the mask reads no
   pool and cannot tell them. *)
let law_paths_agree (s : paged) =
  let unallocated =
    Array.map (Array.map (fun t -> if t >= paged_slots then -1 else t))
  in
  let table = unallocated s.table and draft = Option.map unallocated s.draft in
  let s = { s with table; draft } in
  let index =
    paged_index ~table ~pos:s.pos ~window:s.window ~draft ~sees:s.sees
  in
  cover "a draft" (draft <> None);
  cover "a hole a token may see" (sees_a_hole s);
  Nx.Rng.with_key (Nx.Rng.key s.key) @@ fun () ->
  let p = layer () in
  let pool () = draw [| paged_slots; 2; head_dim |] in
  let c = { Attention.Cache.keys = pool (); values = pool () } in
  let x = draw [| 2; 3; embed |] in
  let pos = Cache_index.positions index in
  let q = Rope.apply (rope ()) ~pos (Attention.split ~head_dim p.q x) in
  let k = Rope.apply (rope ()) ~pos (Attention.split ~head_dim p.k x) in
  let v = Attention.split ~head_dim p.v x in
  let read t pool =
    Nx.swapaxes 1 2 (fst (Cache_index.extend index (Nx.swapaxes 1 2 t) pool))
  in
  let m = Cache_index.mask index in
  let m = Nx.reshape [| Nx.dim 0 m; 1; Nx.dim 1 m; Nx.dim 2 m |] m in
  let dense =
    Attention.merge p.out
      (Attention.attend
         ~mask:(fun _ -> m)
         q
         (Nx.Dense { k = read k c.keys; v = read v c.values }))
  in
  close dense (fst (call p c index x))

(* Steps of one form under Rune.vmap. The steps agree on everything an index's
   walk reports, so that they stack, and draw each tensor apart: tokens, pools,
   positions, tables, window sizes and draft slots. *)
type form = Table | Draft | Lanes | Whole | Packed

let form_name = function
  | Table -> "a table"
  | Draft -> "a draft"
  | Lanes -> "one-token lanes"
  | Whole -> "a whole index"
  | Packed -> "a packed index"

type mapped = {
  form : form;
  has_window : bool;
  resolved : bool;
  steps : (string * Nx.host Cache_index.t * int option * int) list;
}

let pp_mapped ppf m =
  Format.fprintf ppf "%s%s%s: %s" (form_name m.form)
    (if m.has_window then ", windowed" else "")
    (if m.resolved then ", resolved" else "")
    (String.concat " / " (List.map (fun (d, _, _, _) -> d) m.steps))

let table_step ~has_window ~drafted =
  Gen.map
    (fun (s : paged) ->
      (Format.asprintf "%a" pp_paged s, s.index, s.window, s.key))
    (paged_of ~has_window ~drafted)

(* Three lanes of one token over two sequences' tables, a lane of row [-1]
   padding; lane [l] of sequence [r] is at position [p0.(r) + l]. *)
let lanes_step ~has_window =
  Gen.(
    let+ t0 = lane_table 0
    and+ t1 = lane_table 1
    and+ rows = list ~size:(constant 3) (int_range (-1) 1)
    and+ p0 = list ~size:(constant 2) (int_range 0 3)
    and+ window = window_of has_window
    and+ key = key_gen in
    let rows = Array.of_list rows and p0 = Array.of_list p0 in
    let pos =
      Array.mapi (fun l r -> [| (if r < 0 then -1 else p0.(r) + l) |]) rows
    in
    let index =
      Cache_index.make
        ~row:(int64s [| 3 |] rows)
        ~pos:(matrix pos) ~table:(matrix [| t0; t1 |]) ()
    in
    ( Format.asprintf "rows %s pos %s %a key %d"
        (ints (Array.to_list rows))
        (ints (Array.to_list (Array.map (fun p -> p.(0)) pos)))
        pp_window window key,
      windowed window index,
      window,
      key ))

(* Two lanes of three tokens, of [lens] each. *)
let whole_step ~has_window =
  Gen.(
    let+ lens = pair (int_range 0 3) (int_range 0 3)
    and+ window = window_of has_window
    and+ key = key_gen in
    let lens = [| fst lens; snd lens |] in
    ( Format.asprintf "lens %s %a key %d"
        (ints (Array.to_list lens))
        pp_window window key,
      windowed window (Cache_index.whole ~lens ~batch:2 ~seq:3 ()),
      window,
      key ))

(* Two lanes of three tokens, each holding up to two sequences. *)
let packed_step ~has_window =
  Gen.(
    let lane =
      let* first = int_range 0 3 in
      let+ second = int_range 0 (3 - first) in
      [| first; second |]
    in
    let+ l0 = lane and+ l1 = lane and+ window = window_of has_window
    and+ key = key_gen in
    ( Format.asprintf "lens %s | %s %a key %d"
        (ints (Array.to_list l0))
        (ints (Array.to_list l1))
        pp_window window key,
      windowed window (Cache_index.packed ~seq:3 [| l0; l1 |]),
      window,
      key ))

let mapped_gen =
  let open Gen in
  let* form = of_list [ Table; Draft; Lanes; Whole; Packed ] in
  let* has_window = bool in
  let* resolved = bool in
  let* n = int_range 1 3 in
  let step =
    match form with
    | Table -> table_step ~has_window ~drafted:false
    | Draft -> table_step ~has_window ~drafted:true
    | Lanes -> lanes_step ~has_window
    | Whole -> whole_step ~has_window
    | Packed -> packed_step ~has_window
  in
  let resolve (d, index, window, key) =
    (d, (if resolved then Cache_index.resolve index else index), window, key)
  in
  let+ steps = list ~size:(constant n) (map resolve step) in
  { form; has_window; resolved; steps }

(* [Rune.vmap] over steps of their own tokens, caches and indices, stacked along
   a new leading axis, is each step: its outputs and written pools. *)
let law_vmap (m : mapped) =
  let n = List.length m.steps in
  List.iter
    (fun f -> cover (form_name f) (m.form = f))
    [ Table; Draft; Lanes; Whole; Packed ];
  cover "a window" m.has_window;
  let windows = List.map (fun (_, _, w, _) -> w) m.steps in
  cover "steps under different windows"
    (List.length (List.sort_uniq compare windows) > 1);
  cover "a resolved index that addresses a pool"
    (m.resolved && m.form <> Whole && m.form <> Packed);
  cover "one step" (n = 1);
  cover "three steps" (n = 3);
  let p = layer () in
  let slots = match m.form with Whole | Packed -> 0 | _ -> paged_slots in
  let tokens =
    match m.form with Lanes -> [| 3; 1; embed |] | _ -> [| 2; 3; embed |]
  in
  let steps =
    List.map
      (fun (_, index, _, key) ->
        Nx.Rng.with_key (Nx.Rng.key key) @@ fun () ->
        let pool () = draw [| slots; 2; head_dim |] in
        let c = { Attention.Cache.keys = pool (); values = pool () } in
        { x = draw tokens; index; c })
      m.steps
  in
  let lead t = Nx.reshape (Array.append [| 1 |] (Nx.shape t)) t in
  let stacked =
    match List.map (Nx.Ptree.map step_tree (fun _ t -> lead t)) steps with
    | [] -> invalid_arg "law_vmap: no step"
    | s :: rest ->
        List.fold_left
          (Nx.Ptree.map2 step_tree (fun _ a b ->
               Nx.concatenate ~axis:0 [ a; b ]))
          s rest
  in
  let alone = List.map (step_of p) steps in
  let mapped =
    Rune.vmap Nx.Ptree.(step_tree @-> returns step_tree) (step_of p) stacked
  in
  List.iteri
    (fun i alone ->
      let at t = Nx.slice [ I i ] t in
      let msg what = Printf.sprintf "step %d, %s" i what in
      close ~msg:(msg "outputs") alone.x (at mapped.x);
      close ~msg:(msg "keys") alone.c.keys (at mapped.c.keys);
      close ~msg:(msg "values") alone.c.values (at mapped.c.values))
    alone

let test_cached_pieces () =
  Nx.Rng.with_key (Nx.Rng.key 49) @@ fun () ->
  let p = layer () in
  let x = draw [| 2; 3; embed |] in
  let exact ~msg (y, (c : _ Attention.Cache.t)) (y', (c' : _ Attention.Cache.t))
      =
    equal ~msg (array float_exact) (flat y) (flat y');
    equal ~msg:(msg ^ ", keys") (array float_exact) (flat c.keys) (flat c'.keys);
    equal ~msg:(msg ^ ", values") (array float_exact) (flat c.values)
      (flat c'.values)
  in
  let index = Cache_index.window 2 (rows ()) in
  exact ~msg:"through a cache"
    (call p (cache 8) index x)
    (composed p (cache 8) index x);
  let whole = Cache_index.whole ~batch:2 ~seq:3 () in
  exact ~msg:"over a whole index"
    (call p (cache 0) whole x)
    (composed p (cache 0) whole x)

(* Law 1 holds for the pieces composed with sinks and a score scale. *)
let test_cached_sinks_chunking () =
  Nx.Rng.with_key (Nx.Rng.key 53) @@ fun () ->
  let p = layer () in
  let sinks = Nx.create Nx.float64 [| 4 |] [| -1.5; 0.25; 2.0; 0.75 |] in
  let x = draw [| 1; 7; embed |] in
  let table = [| Array.init 8 Fun.id |] in
  let feed chunks =
    let _, ys, _ =
      List.fold_left
        (fun (at, ys, c) n ->
          let index =
            Cache_index.window 3
              (index_at ~pos:[| Array.init n (fun i -> at + i) |] ~table)
          in
          let y, c =
            composed ~scale:0.9 ~sinks p c index
              (Nx.slice [ A; R (at, at + n) ] x)
          in
          (at + n, y :: ys, c))
        (0, [], cache 8)
        chunks
    in
    Nx.concatenate ~axis:1 (List.rev ys)
  in
  let whole =
    fst
      (composed ~scale:0.9 ~sinks p (cache 0)
         (Cache_index.window 3 (Cache_index.whole ~batch:1 ~seq:7 ()))
         x)
  in
  close ~msg:"token by token" whole (feed [ 1; 1; 1; 1; 1; 1; 1 ]);
  close ~msg:"uneven chunks" whole (feed [ 3; 1; 2; 1 ])

(* A decode loop through one compiled step, its cache donated: token [i] at
   position [i], its slot taken from [order] as it comes, under a window of
   [windows.(i)]. Each step changes the position, the table and the window's
   size. *)
let order = [| 3; 0; 4; 2; 1 |]
and windows = [| 1; 3; max_int; 2; 4 |]

let test_cached_one_program () =
  Compiled.here ();
  Nx.Rng.with_key (Nx.Rng.key 24) @@ fun () ->
  let p = layer () in
  let x = draw [| 1; 5; embed |] in
  let traces = ref 0 in
  let step =
    Rune.jit
      Nx.Ptree.(step_tree @-> returns step_tree)
      (fun s ->
        incr traces;
        step_of p s)
  in
  (* A fresh pool is one zero broadcast over its shape, and each step returns
     storage. rune.mli compiles one variant per layout of the arguments, so the
     pools are copied to storage first, and every step takes one layout. *)
  let c = ref (Nx.Ptree.map cache_tree (fun _ t -> Nx.copy t) (cache 5)) in
  Array.iteri
    (fun i w ->
      let table = Array.mapi (fun j s -> if j <= i then s else -1) order in
      let index =
        Cache_index.window w (index_at ~pos:[| [| i |] |] ~table:[| table |])
      in
      let token t = Nx.slice [ A; R (i, i + 1) ] t in
      let s = step { x = token x; index; c = Nx.Ptree.donate cache_tree !c } in
      c := s.c;
      close
        ~msg:(Printf.sprintf "token %d, window %d" i w)
        (token (whole_pass ~window:w p x))
        s.x)
    windows;
  equal ~msg:"traces" int 1 !traces

(* Buffers compare as the byte of memory they start at. A call that wrote a
   value in its own storage returns it over the memory it was given. *)
let at =
  Testable.make
    ~pp:(fun ppf b ->
      Format.fprintf ppf "%s at %d"
        (Rig.name (Rig.Buffer.device b))
        (Rig.Buffer.offset b))
    ~equal:(fun b b' ->
      Rig.Buffer.overlaps b b' && Rig.Buffer.offset b = Rig.Buffer.offset b')

(* Each leaf's storage, read without sharing it: a value over its memory would
   keep the next call from taking it. *)
let storage c =
  Nx.Ptree.fold cache_tree
    (fun _ t acc -> acc @ [ Nx.Repr.read ~by:"test" t Nx_array.buffer ])
    c []

let test_cached_donated_in_place () =
  Compiled.here ();
  Nx.Rng.with_key (Nx.Rng.key 51) @@ fun () ->
  let p = layer () in
  let step =
    Rune.jit
      Nx.Ptree.(
        tensor @-> v Cache_index.walk @-> cache_tree
        @-> returns (pair tensor cache_tree))
      (fun x index c -> call p c index x)
  in
  let x = draw [| 1; 4; embed |] in
  let step i index c =
    snd
      (step
         (Nx.slice [ A; R (i, i + 1) ] x)
         index
         (Nx.Ptree.donate cache_tree c))
  in
  let index = ref (Cache_index.rows ~context:4 [| 1 |]) in
  let c = ref (step 0 !index (cache 4)) in
  for i = 1 to 3 do
    let before = storage !c in
    index := Cache_index.advance !index;
    c := step i !index !c;
    equal ~msg:(Printf.sprintf "step %d" i) (list at) before (storage !c)
  done

(* A compiled draft over two trees replays one trace and gives eager's outputs
   and stores. *)
let test_cached_draft_compiled () =
  Compiled.here ();
  Nx.Rng.with_key (Nx.Rng.key 41) @@ fun () ->
  let p = layer () in
  let x = draw [| 1; 4; embed |] in
  let c = after_prefix ~window:None p (draw [| 1; 3; embed |]) 3 in
  let traces = ref 0 in
  let compiled =
    Rune.jit
      Nx.Ptree.(step_tree @-> returns step_tree)
      (fun s ->
        incr traces;
        step_of p s)
  in
  List.iter
    (fun parents ->
      let index =
        Cache_index.draft
          ~slots:(int64s [| 1; 4 |] [| 12; 15; 13; 14 |])
          ~sees:(tree_sees parents)
          (index_at
             ~pos:[| Array.map (fun d -> 3 + d) (depths parents) |]
             ~table:[| draft_table |])
      in
      let eager = step_of p { x; index; c }
      and jitted = compiled { x; index; c } in
      close ~msg:"outputs" eager.x jitted.x;
      (* The other slots hold the prefix's NaN poison. *)
      let drafted s = Nx.slice [ R (12, 16) ] s.c.keys in
      close ~msg:"stored keys" (drafted eager) (drafted jitted))
    [ [| -1; 0; 0; 1 |]; [| -1; 0; 1; 1 |] ];
  equal ~msg:"traces" int 1 !traces

(* Compiled, one program serves every packing of two lanes of 6 tokens. *)
let test_cached_packed_compiled () =
  Compiled.here ();
  Nx.Rng.with_key (Nx.Rng.key 54) @@ fun () ->
  let p = layer () in
  let x = draw [| 2; 6; embed |] in
  let forward (x, index) = fst (call p (cache 0) index x) in
  let traces = ref 0 in
  let compiled =
    Rune.jit
      Nx.Ptree.(pair tensor (v Cache_index.walk) @-> returns tensor)
      (fun a ->
        incr traces;
        forward a)
  in
  List.iter
    (fun lens ->
      let index = Cache_index.packed ~seq:6 lens in
      close (forward (x, index)) (compiled (x, index)))
    [ [| [| 2; 4 |]; [| 6 |] |]; [| [| 1; 1; 3 |]; [| 3; 2 |] |] ];
  equal ~msg:"traces" int 1 !traces

(* A prompt of [n - 1] tokens then one more, through [table]. *)
let gradient_cases =
  [
    ("from a prompt into a step", None, [| 2; 0; 1; -1; -1 |], 3);
    ("under a window over a hole", Some 2, [| 2; -1; 0; 3; 1 |], 4);
  ]

let test_cached_gradients (_, window, table, n) =
  Nx.Rng.with_key (Nx.Rng.key 25) @@ fun () ->
  let x = draw [| 1; n; embed |] in
  let loss p =
    let index pos =
      windowed window (index_at ~pos:[| pos |] ~table:[| table |])
    in
    let y1, c =
      call p (cache 4)
        (index (Array.init (n - 1) Fun.id))
        (Nx.slice [ A; R (0, n - 1) ] x)
    in
    let y2, _ = call p c (index [| n - 1 |]) (Nx.slice [ A; R (n - 1, n) ] x) in
    Nx.add (Nx.sum (Nx.mul y1 y1)) (Nx.sum (Nx.mul y2 y2))
  in
  match Rune.check_grads attention loss (layer ()) with
  | Ok () -> ()
  | Error m -> fail m

(* The gradients of a loss over a real lane beside a lane of padding equal those
   of the real lane alone: the padded queries see no key, so their weights and
   gradients are zero. Both sum the same products in another order. The backward
   pass runs the forward's stages in reverse through the same softmax, so each
   cotangent is within the forward's relative error at the heads, 6.8e4 u over
   6.7 or 1e4 u (see [reassociation]), doubled for the backward pass's own sums.
   The cotangents are below 1 at the output (|y| below 34, the loss a mean of
   y^2 over 72 entries), 4.6 at the heads, 62 at the weights and 124 in sum at
   the scores, so the largest gradient, q's projection's, sums 9 tokens' terms
   through keys below 9.5/√2 and stays below 7.5e3. Two evaluations differ by at
   most 2 · 2e4 u · 7.5e3 = 3e8 u. *)
let padded_gradient_cases =
  [
    ("eager", fun loss -> Rune.grad attention loss);
    ( "compiled",
      fun loss ->
        Compiled.here ();
        Rune.jit
          Nx.Ptree.(attention @-> returns attention)
          (Rune.grad attention loss) );
  ]

let test_cached_padded_gradients (_, grad) =
  Nx.Rng.with_key (Nx.Rng.key 52) @@ fun () ->
  let n = 9 in
  let x = draw [| 2; n; embed |] in
  let loss index x p =
    let y = fst (call p (cache 0) index x) in
    let real = Nx.slice [ R (Nx.dim 0 y - 1, Nx.dim 0 y) ] y in
    Nx.mean (Nx.mul real real)
  in
  let leaves g =
    Nx.Ptree.fold attention
      (fun _ t acc -> flat (Nx.cast Nx.float64 t) :: acc)
      g []
  in
  let padded =
    grad
      (loss (Cache_index.whole ~lens:[| 0; n |] ~batch:2 ~seq:n ()) x)
      (layer ())
  in
  let alone =
    Rune.grad attention
      (loss (Cache_index.whole ~batch:1 ~seq:n ()) (Nx.slice [ R (1, 2) ] x))
      (layer ())
  in
  List.iter2
    (equal (array (float (3e8 *. Float.epsilon /. 2.))))
    (leaves alone) (leaves padded)

let cached_refusals =
  let x () = Nx.zeros Nx.float64 [| 1; 2; embed |] in
  let attend ?(head_dim = head_dim) ?(kv_heads = 2) p x =
    let c =
      Attention.Cache.make ~slots:4 ~kv_heads ~head_dim:2 Nx.float64 Nx.Host.on
    in
    ignore
      (Attention.cached ~head_dim p c (Cache_index.rows ~context:4 [| 2 |]) x)
  in
  let float8 = Nx.float8_e4m3fn in
  [
    ( "a float8 x",
      fun () ->
        ignore
          (Attention.cached ~head_dim
             (Attention.make ~key:(Nx.Rng.key 3) ~kv_dim:4 ~embed_dim:embed
                float8)
             (Attention.Cache.make ~slots:4 ~kv_heads:2 ~head_dim float8
                Nx.Host.on)
             (Cache_index.rows ~context:4 [| 2 |])
             (Nx.zeros float8 [| 1; 2; embed |])) );
    ( "x of another shape",
      fun () -> attend (layer ()) (Nx.zeros Nx.float64 [| 1; 3; embed |]) );
    ("head_dim 0", fun () -> attend ~head_dim:0 (layer ()) (x ()));
    ( "head_dim 3, which does not divide 8",
      fun () -> attend ~head_dim:3 (layer ()) (x ()) );
    ( "key and value projections of widths 4 and 2",
      fun () ->
        let v =
          Linear.make ~key:(Nx.Rng.key 4) ~inputs:embed ~outputs:2 Nx.float64
        in
        attend { (layer ()) with v } (x ()) );
    ( "3 key-value heads for 4 query heads",
      fun () ->
        attend ~kv_heads:3
          (Attention.make ~key:(Nx.Rng.key 5) ~kv_dim:6 ~embed_dim:embed
             Nx.float64)
          (x ()) );
    ( "a cache of 1 key-value head for 2",
      fun () -> attend ~kv_heads:1 (layer ()) (x ()) );
  ]

let test_cached_raises (_, f) =
  raises_match (Exn.invalid_arg ~sub:"Attention.cached") f

let () =
  exit
    (run "kaun attention cache"
       [
         group "laws"
           [
             stateful ~count:200 ~steps:20
               "chunking leaves the outputs and the written slots unchanged"
               decoding;
             prop ~count:100 "a draft is its paths" draft_gen law_draft;
             prop ~count:100 "packing is invariant" packing_gen law_packing;
             prop ~count:100 "the whole-sequence pass is the cached pass"
               whole_gen law_whole;
             test "each cache leaf reads and writes its own tensor"
               test_one_tensor_per_leaf;
           ];
         group "Cache.make"
           [
             test "is zeros of [slots; kv_heads; head_dim]" test_make_zeros;
             test "is made at its placement" test_make_at;
             test "takes 0 slots" test_make_no_slot;
             cases "raises on negative slots or a dimension below 1"
               ~id:(fun (id, _, _, _) -> id)
               make_refusals test_make_raises;
           ];
         group "Cache.walk"
           [ test "walks keys then values, by index in a list" test_walk_paths ];
         group "Cache.extend"
           [
             test "on a whole index is k, v and the cache as given"
               test_extend_whole;
             test "with pages is the stored pools read through them"
               test_extend_paged;
             test "otherwise is Cache_index.extend with the axes exchanged"
               test_extend_dense;
             cases "raises on k, v or pools of another shape" ~id:fst
               extend_refusals test_extend_raises;
           ];
         group "cached"
           [
             test "never mutates its argument" test_cached_functional;
             cases "under a window sees only the last positions"
               ~id:(fun w -> if w = max_int then "max_int" else string_of_int w)
               window_cases test_cached_window;
             cases "gives a padded token the output projection of zero" ~id:fst
               padded_cases test_cached_padded;
             test "gives a hole no weight, eager and compiled" test_cached_hole;
             test "gives a draft token that stores nothing no weight"
               test_cached_draft_storing_nothing;
             prop ~count:100
               "reads no slot the pages do not name, eager and compiled"
               (Gen.with_pp pp_paged paged_gen)
               law_paged_reads;
             prop ~count:100 "is Cache_index.extend's read under its mask"
               (Gen.with_pp pp_paged paged_gen)
               law_paths_agree;
             prop ~count:100
               "under Rune.vmap over steps of their own caches and indices is \
                each step"
               (Gen.with_pp pp_mapped mapped_gen)
               law_vmap;
             test "is the composition of its pieces" test_cached_pieces;
             test "composed with sinks is invariant under chunking"
               test_cached_sinks_chunking;
             test "traces once for every position, table and window size"
               test_cached_one_program;
             test "writes a donated cache in its own storage"
               test_cached_donated_in_place;
             test "traces once for every draft tree" test_cached_draft_compiled;
             test "traces once for every packing" test_cached_packed_compiled;
             cases "is differentiable"
               ~id:(fun (id, _, _, _) -> id)
               gradient_cases test_cached_gradients;
             cases "takes no gradient from a lane of padding" ~id:fst
               padded_gradient_cases test_cached_padded_gradients;
             cases "raises on an input or a geometry it does not take" ~id:fst
               cached_refusals test_cached_raises;
           ];
       ])
