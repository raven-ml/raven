(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The indexed store under [Rune.jit]: a k-row [Nx.scatter] into a donated pool,
   whose step time must not grow with the pool, and the gradient of [Nx.take]
   into a vocabulary-sized table, whose cost must follow the tokens. Each step
   ends with a scalar read that depends on the written tensor, so a timing
   covers the device work. Run with DEV set; on CPU, storage is reused only
   under RUNE_JIT_FORCE_COPY=1. *)

type state = { pool : Nx.float32_t; rows : Nx.int32_t; values : Nx.float32_t }

module State = struct
  type t = state

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) s =
    { pool = f s.pool; rows = f s.rows; values = f s.values }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) s t =
    {
      pool = f s.pool t.pool;
      rows = f s.rows t.rows;
      values = f s.values t.values;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) s =
    f s.pool;
    f s.rows;
    f s.values
end

type written = { next : Nx.float32_t; probe : Nx.float32_t }

module Written = struct
  type t = written

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) w =
    { next = f w.next; probe = f w.probe }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) v w =
    { next = f v.next w.next; probe = f v.probe w.probe }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) w =
    f w.next;
    f w.probe
end

let heads = 8
let width = 64

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a;
  a.(Array.length a / 2)

let timed ~warmup ~runs step =
  for _ = 1 to warmup do
    step ()
  done;
  Gc.full_major ();
  let words = Gc.minor_words () in
  let times =
    List.init runs (fun _ ->
        let t0 = Unix.gettimeofday () in
        step ();
        (Unix.gettimeofday () -. t0) *. 1e3)
  in
  let words = (Gc.minor_words () -. words) /. float_of_int runs in
  (median times, List.fold_left min infinity times, words)

let scatter_case ~runs ~n ~k =
  let write { pool; rows; values } =
    let indices =
      Nx.broadcast_to [| k; heads; width |] (Nx.reshape [| k; 1; 1 |] rows)
    in
    let next = Nx.scatter ~axis:0 ~indices ~values pool in
    { next; probe = Nx.sum (Nx.slice [ Nx.R (0, 1) ] next) }
  in
  let donate = Sys.getenv_opt "DONATE" <> Some "0" in
  let step = Rune.jit2 ~donate (module State) (module Written) write in
  let rows =
    Nx.create Nx.int32 [| k |]
      (Array.init k (fun i -> Int32.of_int (i * (n / k))))
  in
  let values = Nx.ones Nx.float32 [| k; heads; width |] in
  let pool = ref (Nx.zeros Nx.float32 [| n; heads; width |]) in
  let once () =
    let w = step { pool = !pool; rows; values } in
    pool := w.next;
    ignore (Nx.item [] w.probe : float)
  in
  Rune.reset_jit_stats ();
  let med, best, words = timed ~warmup:3 ~runs once in
  let reused = (Rune.jit_stats ()).reused_bytes / (runs + 3) in
  Printf.printf
    "scatter n=%-7d k=%-3d median %8.3f ms  min %8.3f ms  %9.0f words/call  \
     reused %d MB/call\n\
     %!"
    n k med best words (reused / 1_000_000)

type cache = { rows_of : Nx.float32_t; pos : Nx.int32_t }

module Cache = struct
  type t = cache

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) c =
    { rows_of = f c.rows_of; pos = f c.pos }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) c d =
    { rows_of = f c.rows_of d.rows_of; pos = f c.pos d.pos }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) c =
    f c.rows_of;
    f c.pos
end

(* One row written at a run-time position into a donated cache. *)
let window_case ~runs ~n =
  let row = Nx.ones Nx.float32 [| 1; width |] in
  let write { rows_of; pos } =
    let next = Nx.set [ Nx.D (pos, 1) ] row rows_of in
    { next; probe = Nx.sum (Nx.slice [ Nx.R (0, 1) ] next) }
  in
  let step = Rune.jit2 ~donate:true (module Cache) (module Written) write in
  let cache = ref (Nx.zeros Nx.float32 [| n; width |]) in
  let at = ref 0 in
  let once () =
    let pos = Nx.scalar Nx.int32 (Int32.of_int (!at mod n)) in
    incr at;
    let w = step { rows_of = !cache; pos } in
    cache := w.next;
    ignore (Nx.item [] w.probe : float)
  in
  let med, best, words = timed ~warmup:3 ~runs once in
  Printf.printf
    "window  n=%-7d      median %8.3f ms  min %8.3f ms  %9.0f words/call\n%!" n
    med best words

let take_grad_case ~runs ~vocab ~dim ~tokens =
  let ids =
    Nx.create Nx.int32 [| tokens |]
      (Array.init tokens (fun i -> Int32.of_int (i * 7919 mod vocab)))
  in
  let loss table =
    let e = Nx.take ~axis:0 ~indices:ids table in
    Nx.sum (Nx.mul e e)
  in
  let probe table = Nx.sum (Nx.slice [ Nx.R (0, 1) ] (Rune.grad' loss table)) in
  let step = Rune.jit' probe in
  let table = Nx.ones Nx.float32 [| vocab; dim |] in
  let once () = ignore (Nx.item [] (step table) : float) in
  let med, best, words = timed ~warmup:2 ~runs once in
  Printf.printf
    "take grad %dx%d, %d tokens: median %8.3f ms  min %8.3f ms  %9.0f words/call\n\
     %!"
    vocab dim tokens med best words

let ints name default =
  match Sys.getenv_opt name with
  | Some s -> List.map int_of_string (String.split_on_char ',' s)
  | None -> default

(* RUNS sets the timed steps per case, ONLY picks [scatter], [window] or [take],
   and N and K pick the pool sizes and row counts, as comma-separated lists. *)
let () =
  let runs = List.hd (ints "RUNS" [ 30 ]) in
  let wanted name =
    match Sys.getenv_opt "ONLY" with None -> true | Some o -> o = name
  in
  if wanted "scatter" then
    List.iter
      (fun n ->
        List.iter (fun k -> scatter_case ~runs ~n ~k) (ints "K" [ 1; 64 ]))
      (ints "N" [ 4096; 131072 ]);
  if wanted "window" then
    List.iter (fun n -> window_case ~runs ~n) [ 4096; 1048576 ];
  if wanted "take" then take_grad_case ~runs ~vocab:128256 ~dim:64 ~tokens:1024
