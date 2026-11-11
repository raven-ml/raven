(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jit compiled again, as rune.mli's Numerics states: a variant's
   kernels are the same text in a fresh process, in processes that compiled
   other programs first, in either order, and on a second domain. On the
   host the kernels are Rune.jit's; for an NVIDIA, an Apple and an AMD GPU,
   which a compiled call reaches only through a device, they are the
   schedule of rune's lowering of each operation, rendered without the
   device or its toolchain. Each process runs this executable again, with
   tolk's disk caches off, so that each compiles every kernel. A node's
   place in tolk's order follows when it was built, and for each target
   some history builds the variant's nodes in another order, which the test
   checks. *)

open Windtrap

let f32 = Nx.float32

(* Operands *)

let floats s seed =
  let n = Array.fold_left ( * ) 1 s in
  Nx.create f32 s
    (Array.init n (fun k -> Float.of_int (((k * 7919) + seed) mod 97) /. 31.))

let positions s d =
  let n = Array.fold_left ( * ) 1 s in
  Nx.create Nx.int64 s (Array.init n (fun k -> Int64.of_int (k * 37 mod d)))

let x = floats [| 64; 32 |] 1
let w = floats [| 32; 16 |] 2
let p = positions [| 200; 16 |] 64
let u = floats [| 200; 16 |] 3

(* Programs *)

(* The variant the law compiles again: a contraction, a softmax's
   reductions, a scatter's Add and a sum. The other programs share its
   operands and some of its operations, so that a process that compiled
   them first holds some of its nodes already. *)
let target a =
  let y = Nx.softmax ~axes:[ 1 ] (Nx.matmul a w) in
  let z = Nx.scatter ~combine:Add ~axis:0 p u y in
  Nx.add z (Nx.sum ~axes:[ 0 ] ~keepdims:true z)

let others =
  [
    (fun a -> Nx.softmax ~axes:[ 1 ] (Nx.matmul a w));
    (fun a -> Nx.scatter ~combine:Add ~axis:0 p u (Nx.matmul a w));
    (fun a -> Nx.sum ~axes:[ 0 ] (Nx.exp (Nx.matmul a w)));
  ]

(* The GPU targets, whose kernels are lowered and rendered without a
   device or its toolchain: an NVIDIA GPU of compute capability 8.9, an
   Apple GPU of family 9 and an AMD GPU of gfx1201. *)
let gpus =
  List.map
    (fun t -> Result.get_ok (Tolk.Helpers.Target.of_string t))
    [ "CUDA:CUDA:sm_89"; "METAL:METAL:Apple9"; "AMD:HIP:gfx1201" ]

let label (t : Tolk.Helpers.Target.t) = t.device ^ " " ^ t.arch

(* [f] compiled on the host by Rune.jit, and lowered and rendered for each
   GPU target. *)
let compile f =
  ignore (Rune.jit' f x);
  List.iter
    (fun target -> ignore (Lowered.sources ~target (fun () -> f x)))
    gpus

(* A kernel as a process reports it: its target, its name, the order of
   its nodes, as a digest of their keys sorted by tolk's order, and the
   source its target's renderer writes for it. *)
type kernel = {
  target : string;
  name : string;
  order : string;
  source : string;
}

let order ast =
  let nodes = Tolk.Ops.toposort ~calls:Enter ast in
  let keys = List.map Tolk.Ops.key (List.sort Tolk.Ops.compare nodes) in
  Digest.to_hex (Digest.string (String.concat "" keys))

let host_kernels f =
  let (), ks = Tolk.Realize.kernels f in
  List.map
    (fun (k : Tolk.Realize.kernel) ->
      let p = Tolk.Codegen.linearize k.ast k.renderer in
      {
        target = "host";
        name = k.name;
        order = order k.ast;
        source = k.renderer.render (Tolk.Ops.src (Tolk.Ops.nth p 1));
      })
    ks

let gpu_kernels t f =
  List.map
    (fun (name, ast, source) ->
      { target = label t; name; order = order ast; source })
    (Lowered.sources ~target:t f)

let target_kernels () =
  host_kernels (fun () -> ignore (Rune.jit' target x))
  @ List.concat_map (fun t -> gpu_kernels t (fun () -> target x)) gpus

(* The histories *)

let histories =
  [
    ("fresh", target_kernels);
    ( "others first",
      fun () ->
        List.iter compile others;
        target_kernels () );
    ( "others first, reversed",
      fun () ->
        List.iter compile (List.rev others);
        target_kernels () );
    ( "on a second domain, others first on the first",
      fun () ->
        List.iter compile others;
        Domain.join (Domain.spawn target_kernels) );
    ( "others first on a second domain",
      fun () ->
        Domain.join (Domain.spawn (fun () -> List.iter compile others));
        target_kernels () );
  ]

let history_variable = "RUNE_TEST_HISTORY"

(* A child process writes its kernels to its standard output. *)
let child h =
  let ks = (List.assoc h histories) () in
  set_binary_mode_out stdout true;
  output_value stdout (ks : kernel list);
  exit 0

(* The kernels of a new process that runs history [h]. *)
let run_history h =
  let out = Filename.temp_file "recompile" ".bin" in
  let fd = Unix.openfile out [ Unix.O_WRONLY; O_TRUNC ] 0o600 in
  let env =
    Array.append
      [| history_variable ^ "=" ^ h; "CACHELEVEL=0" |]
      (Unix.environment ())
  in
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process_env exe [| exe |] env Unix.stdin fd Unix.stderr
  in
  Unix.close fd;
  let status = snd (Unix.waitpid [] pid) in
  let ks =
    match status with
    | Unix.WEXITED 0 ->
        In_channel.with_open_bin out (fun ic -> (input_value ic : kernel list))
    | Unix.WEXITED n -> failf "history %S exited %d" h n
    | Unix.WSIGNALED n | Unix.WSTOPPED n ->
        failf "history %S ended by signal %d" h n
  in
  Sys.remove out;
  ks

(* The first line of [a] that differs from [b]'s, and [b]'s. *)
let first_difference a b =
  let rec go = function
    | x :: a, y :: b -> if x = y then go (a, b) else (x, y)
    | x :: _, [] -> (x, "")
    | [], y :: _ -> ("", y)
    | [], [] -> ("", "")
  in
  go (String.split_on_char '\n' a, String.split_on_char '\n' b)

let same_sources h ~fresh ks =
  let names ks = List.map (fun k -> k.target ^ " " ^ k.name) ks in
  equal ~msg:(h ^ ": kernel names") (list string) (names fresh) (names ks);
  List.iter2
    (fun f k ->
      if f.source <> k.source then
        let a, b = first_difference f.source k.source in
        failf "%s: kernel %s for %s has %S where a fresh process has %S" h
          k.name k.target b a)
    fresh ks

let test_same_sources () =
  let fresh = run_history "fresh" in
  let on t ks = List.filter (fun k -> k.target = t) ks in
  let targets = "host" :: List.map label gpus in
  List.iter
    (fun t ->
      let n = List.length (on t fresh) in
      if n < 3 then failf "the variant compiled %d kernels for %s" n t)
    targets;
  let others =
    List.filter_map
      (fun (h, _) ->
        if h = "fresh" then None
        else
          let ks = run_history h in
          same_sources h ~fresh ks;
          Some ks)
      histories
  in
  (* Each target's kernels are built in another order by some history. *)
  let orders t ks = List.map (fun k -> k.order) (on t ks) in
  List.iter
    (fun t ->
      if List.for_all (fun ks -> orders t ks = orders t fresh) others then
        failf "no history built the variant's nodes for %s in another order" t)
    targets

let () =
  match Sys.getenv_opt history_variable with
  | Some h -> child h
  | None ->
      exit
      @@ Windtrap.run "rune.recompile"
           [
             test
               "a variant compiled again renders the same kernels, whatever \
                the process compiled before"
               test_same_sources;
           ]
