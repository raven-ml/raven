(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Greedy decoding with gpt-oss through key-value caches, from fixed token ids:
   the tiny test checkpoints have random weights, so there is no text to read.
   It shows the decode loop and times it: one step function serves the prefill
   and every single-token step, and under [--jit DEVICE] compiles once for each
   of the two shapes.

   Usage: main.exe [--repo REPO] [--jit DEVICE] [--count N]. *)

open Kaun

let prompt =
  [|
    200006l;
    17360l;
    200008l;
    3575l;
    553l;
    17554l;
    162016l;
    11l;
    261l;
    4410l;
    6439l;
    2359l;
  |]

let generate ?device cfg (params : Gpt_oss.t) ~count =
  let module Step = struct
    type t = {
      token : Nx.int32_t;
      index : Cache_index.t;
      caches : Nx.float32_t Gpt_oss.Cache.t;
    }

    let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) s =
      let token = f s.token in
      let index = Cache_index.map f s.index in
      let caches = Gpt_oss.Cache.map f s.caches in
      { token; index; caches }

    let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) a b =
      let token = f a.token b.token in
      let index = Cache_index.map2 f a.index b.index in
      let caches = Gpt_oss.Cache.map2 f a.caches b.caches in
      { token; index; caches }

    let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) s =
      f s.token;
      Cache_index.iter f s.index;
      Gpt_oss.Cache.iter f s.caches
  end in
  let step (s : Step.t) =
    let seq = Nx.dim 1 s.token in
    let h, caches = Gpt_oss.cached cfg params s.caches s.index s.token in
    let logits = Gpt_oss.logits cfg params (Nx.slice [ A; I (seq - 1) ] h) in
    {
      Step.token = Nx.reshape [| 1; 1 |] (Nx.argmax ~axis:1 logits);
      index = Cache_index.advance s.index;
      caches;
    }
  in
  let step =
    match device with
    | None -> step
    | Some device ->
        Rune.jit2 ~device ~donate:true (module Step) (module Step) step
  in
  let n0 = Array.length prompt in
  let context = n0 + count in
  let timed s =
    let t0 = Unix.gettimeofday () in
    let s = step s in
    let token = Nx.item [ 0; 0 ] s.Step.token in
    (s, token, Unix.gettimeofday () -. t0)
  in
  let state, first, prefill =
    timed
      {
        Step.token = Nx.create Nx.int32 [| 1; n0 |] prompt;
        index = Cache_index.rows ~context [| n0 |];
        caches = Gpt_oss.cache cfg ~slots:context Nx.float32;
      }
  in
  Printf.printf "prefill of %d tokens: %.3f s\n%!" n0 prefill;
  let out = Array.make count first in
  let times = Array.make (count - 1) 0.0 in
  let state = ref state in
  for n = 1 to count - 1 do
    let s, token, t = timed !state in
    state := s;
    out.(n) <- token;
    times.(n - 1) <- t
  done;
  if count > 1 then begin
    Printf.printf "first step: %.3f s\n" times.(0);
    let later = Array.sub times 1 (count - 2) in
    Array.sort compare later;
    if later <> [||] then
      Printf.printf "later steps: median %.1f ms\n"
        (1e3 *. later.(Array.length later / 2))
  end;
  out

let () =
  let repo = ref "tiny-random/gpt-oss-mxfp4" in
  let jit = ref "" and count = ref 16 in
  Arg.parse
    [
      ("--repo", Arg.Set_string repo, "A single-file gpt-oss checkpoint");
      ("--jit", Arg.Set_string jit, "Compile the step for this device");
      ("--count", Arg.Set_int count, "Number of tokens to generate");
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "main.exe [--repo REPO] [--jit DEVICE] [--count N]";
  let cfg, params = Gpt_oss.from_pretrained !repo in
  let device = if !jit = "" then None else Some !jit in
  let out = generate ?device cfg params ~count:!count in
  print_endline
    (String.concat " " (Array.to_list (Array.map Int32.to_string out)))
