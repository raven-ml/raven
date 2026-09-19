(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Validation of the gpt-oss example against a reference recorded one block at a
   time, for checkpoints whose whole forward pass fits neither transformers nor
   an eager run here: gpt-oss-20b.

   A [fixtures/*-stream.json] file, written by [reference_stream.py], names its
   repository and holds, for three prompts of 12, 43 and 167 tokens, the
   router's experts of every token in every block, the residual stream after
   every block's attention and after the block at a few positions (its mean,
   root mean square and largest magnitude, its first features and fixed
   projections on rows of +1 and -1), whole streams at the last position for
   three blocks, the normalised last stream, and the largest logits and the
   argmax of every position.

   The checkpoint is loaded through [Gpt_oss.of_hf] at the dtype it is stored
   at, which copies nothing. The model then runs one block at a time through
   [Gpt_oss.hidden] on a one-block model whose token table is the residual
   stream, so that a block's float leaves are cast to the compute dtype for the
   time of its call only and nothing else is kept. The attention is also run
   alone to split a block's error between its attention and its experts and to
   give the router the input the block gives it.

   Errors of a stream are in units of the reference stream's root mean square
   at that position: the largest difference over the first features, the
   largest difference over the projections divided by the square root of the
   width, and the differences of the three statistics.

   Usage: validate_stream.exe FIXTURE [--blocks N] [--prompt NAME] [--dtype DT]
   [--tol X]. With [--blocks] only the first [N] blocks run and the head is
   skipped. Not part of the test suite: it needs the download, 13.8 GB for
   gpt-oss-20b. *)

open Kaun

let json_of_file path =
  let ic = open_in_bin path in
  let s =
    Fun.protect
      ~finally:(fun () -> close_in ic)
      (fun () -> In_channel.input_all ic)
  in
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok j -> j
  | Error e -> failwith (path ^ ": " ^ e)

let members = function
  | Jsont.Object (mems, _) -> List.map (fun ((name, _), v) -> (name, v)) mems
  | _ -> failwith "fixture: not an object"

let mem name j =
  match List.assoc_opt name (members j) with
  | Some v -> v
  | None -> failwith ("fixture: missing " ^ name)

let find name j = List.assoc_opt name (members j)

let list = function
  | Jsont.Array (l, _) -> l
  | _ -> failwith "fixture: array expected"

let number = function
  | Jsont.Number (f, _) -> f
  | _ -> failwith "fixture: number expected"

let string = function
  | Jsont.String (s, _) -> s
  | _ -> failwith "fixture: string expected"

let floats j = Array.of_list (List.map number (list j))
let ints j = Array.map int_of_float (floats j)
let failures = ref 0
let first_failure = ref None

let check name ok detail =
  Printf.printf "%-52s %s%s\n%!" name (if ok then "ok" else "FAIL") detail;
  if not ok then begin
    incr failures;
    if !first_failure = None then first_failure := Some name
  end

(* The projection rows of the recorder, from the same generator. *)
let signs ~rows ~dim =
  let state = ref 12345 in
  Array.init (rows * dim) (fun _ ->
      state := ((!state * 1103515245) + 12345) land 0x7FFFFFFF;
      if (!state lsr 16) land 1 = 1 then 1.0 else -1.0)

let host t =
  Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous (Nx.cast Nx.float32 t)))

let worst_of a = Array.fold_left (fun m d -> if d <= m then m else d) 0.0 a

(* The error of row [t] of the stream [x], [tokens * dim] values, against a
   recorded summary, in units of the reference's root mean square. A value that
   is not finite gives an infinite error. *)
let summary_error ~signs ~dim summary x t =
  let v = Array.sub x (t * dim) dim in
  let rms' = number (mem "rms" summary) in
  let n = float_of_int dim in
  let mean = Array.fold_left ( +. ) 0.0 v /. n in
  let rms = sqrt (Array.fold_left (fun s e -> s +. (e *. e)) 0.0 v /. n) in
  let absmax = Array.fold_left (fun m e -> Float.max m (Float.abs e)) 0.0 v in
  let absmax' = number (mem "absmax" summary) in
  let first = floats (mem "first" summary) in
  let projection = floats (mem "projection" summary) in
  let projected j =
    let s = ref 0.0 in
    for i = 0 to dim - 1 do
      s := !s +. (signs.((j * dim) + i) *. v.(i))
    done;
    !s
  in
  let errors =
    Array.concat
      [
        Array.mapi (fun i e -> Float.abs (v.(i) -. e) /. rms') first;
        Array.mapi
          (fun j e -> Float.abs (projected j -. e) /. (rms' *. sqrt n))
          projection;
        [|
          Float.abs (mean -. number (mem "mean" summary)) /. rms';
          Float.abs (rms -. rms') /. rms';
          Float.abs (absmax -. absmax') /. absmax';
        |];
      ]
  in
  if Array.for_all Float.is_finite v then worst_of errors else Float.infinity

let summaries_error ~signs ~dim ~positions recorded x =
  worst_of
    (Array.of_list
       (List.map2
          (fun summary t -> summary_error ~signs ~dim summary x t)
          (list recorded) positions))

(* A whole row against a recorded one, in the same unit. *)
let row_error ~dim recorded x t =
  let v = Array.sub x (t * dim) dim in
  let n = float_of_int dim in
  let rms' =
    sqrt (Array.fold_left (fun s e -> s +. (e *. e)) 0.0 recorded /. n)
  in
  if Array.for_all Float.is_finite v then
    worst_of (Array.mapi (fun i e -> Float.abs (v.(i) -. e) /. rms') recorded)
  else Float.infinity

(* The router's experts against the recorded ones. torch.topk and [Nx.top_k]
   both give the greatest logit first, so the rows are compared in order, and as
   sets where the order differs. [margin] is the least gap the reference saw
   between a selected logit and a rejected one: below the error of the router's
   logits a different set is no discrepancy. *)
let routing ~k ~margin recorded experts =
  let tokens = Array.length recorded / k in
  let sets = ref 0 and orders = ref 0 in
  for t = 0 to tokens - 1 do
    let row a = Array.sub a (t * k) k in
    let r = row recorded and e = row experts in
    if r <> e then begin
      incr orders;
      Array.sort compare r;
      Array.sort compare e;
      if r <> e then incr sets
    end
  done;
  ( !sets = 0,
    Printf.sprintf "experts: %d of %d tokens differ as sets, %d in order, margin %.1e"
      !sets tokens (!orders - !sets) margin )

let run (type c) ~tol ~blocks ~only fx (dt : (float, c) Nx.dtype) =
  let repo = string (mem "repo" fx) in
  List.iter
    (fun (file, sha) ->
      Printf.printf "%s/%s, reference recorded from sha256 %s\n%!" repo file
        (string sha))
    (members (mem "files_sha256" fx));
  let cfg = Gpt_oss.config_of_json (Kaun_hf.load_config repo) in
  let ckpt = Kaun_hf.load_checkpoint repo in
  let (Gpt_oss.Dtype stored) = Gpt_oss.stored_dtype ckpt in
  let p = Gpt_oss.of_hf cfg stored ckpt in
  let dim = cfg.dim and k = cfg.experts_per_token in
  let layers = List.length cfg.layers in
  let depth = match blocks with None -> layers | Some n -> min n layers in
  let signs = signs ~rows:(int_of_float (number (mem "projections" fx))) ~dim in
  let cast t = Nx.cast dt t in
  (* [Gpt_oss.map] needs a table at the stored dtype; the stream replaces it. *)
  let nothing = { Embedding.table = Nx.zeros stored [| 1; 1 |] } in
  let one_block b x =
    let m =
      Gpt_oss.map cast
        { Gpt_oss.tok = nothing; blocks = [ b ]; norm = p.norm; head = None }
    in
    { m with tok = { Embedding.table = x } }
  in
  let prompt (name, recorded) =
    let ids = ints (mem "ids" recorded) in
    let tokens = Array.length ids in
    let positions = Array.to_list (ints (mem "positions" recorded)) in
    let last = tokens - 1 in
    let label what = Printf.sprintf "%s (%d tokens): %s" name tokens what in
    let error e = Printf.sprintf "  (%.1e)" e in
    let ids_t =
      Nx.create Nx.int32 [| 1; tokens |] (Array.map Int32.of_int ids)
    in
    let x = ref (cast (Nx.reshape [| tokens; dim |] (Embedding.apply p.tok ids_t))) in
    let e =
      summaries_error ~signs ~dim ~positions (mem "embedded" recorded) (host !x)
    in
    check (label "embedding") (e < tol) (error e);
    let rows = Nx.reshape [| 1; tokens |] (Nx.arange Nx.int32 0 tokens 1) in
    let index = Cache_index.whole ~batch:1 ~seq:tokens () in
    List.iteri
      (fun i (layer, (b, rb)) ->
        if i < depth then begin
          let started = Unix.gettimeofday () in
          let one = { cfg with Gpt_oss.layers = [ layer ] } in
          let m = one_block b !x in
          let b' = List.hd m.blocks in
          let x3 = Nx.reshape [| 1; tokens; dim |] !x in
          let attended, _ =
            Gpt_oss.attention cfg layer b'
              (List.hd (Gpt_oss.cache one ~slots:0 dt))
              index
              (Rms_norm.apply ~eps:cfg.norm_eps b'.attn_norm x3)
          in
          let middle = Nx.add x3 attended in
          let experts, _ =
            Moe.route ~k b'.moe
              (Nx.reshape [| tokens; dim |]
                 (Rms_norm.apply ~eps:cfg.norm_eps b'.ffn_norm middle))
          in
          let y = Nx.reshape [| tokens; dim |] (Gpt_oss.hidden one m rows) in
          let block = Printf.sprintf "block %d" i in
          let e =
            summaries_error ~signs ~dim ~positions (mem "after_attention" rb)
              (host middle)
          in
          check (label (block ^ " after attention")) (e < tol) (error e);
          let routed, detail =
            routing ~k
              ~margin:(number (mem "margin" rb))
              (ints (mem "experts" rb))
              (Array.map Int32.to_int
                 (Nx.to_array (Nx.reshape [| -1 |] experts)))
          in
          check (label (block ^ " router")) routed ("  (" ^ detail ^ ")");
          let y_host = host y in
          let e =
            summaries_error ~signs ~dim ~positions (mem "after_block" rb) y_host
          in
          let e =
            match find "after_block_last" rb with
            | None -> e
            | Some row -> Float.max e (row_error ~dim (floats row) y_host last)
          in
          check
            (label (block ^ " after experts"))
            (e < tol)
            (Printf.sprintf "  (%.1e, %.0f s)" e (Unix.gettimeofday () -. started));
          x := y;
          Gc.full_major ()
        end)
      (List.combine cfg.layers
         (List.combine p.blocks (list (mem "blocks" recorded))));
    if depth = layers then begin
      let m =
        Gpt_oss.map cast
          {
            Gpt_oss.tok = (if cfg.tied then p.tok else nothing);
            blocks = [];
            norm = p.norm;
            head = p.head;
          }
      in
      let normed = Rms_norm.apply ~eps:cfg.norm_eps m.norm !x in
      let normed_host = host normed in
      let e =
        Float.max
          (summaries_error ~signs ~dim ~positions (mem "normed" recorded)
             normed_host)
          (row_error ~dim (floats (mem "normed_last" recorded)) normed_host last)
      in
      check (label "final norm") (e < tol) (error e);
      let logits = host (Gpt_oss.logits cfg m !x) in
      let vocab = Array.length logits / tokens in
      let top_ids = List.map ints (list (mem "top_ids" recorded)) in
      let top_values = List.map floats (list (mem "top_values" recorded)) in
      let worst = ref 0.0 in
      List.iteri
        (fun t (ids, values) ->
          let scale = worst_of (Array.map Float.abs values) in
          Array.iteri
            (fun j id ->
              let d = Float.abs (logits.((t * vocab) + id) -. values.(j)) /. scale in
              if not (d <= !worst) then worst := d)
            ids)
        (List.combine top_ids top_values);
      check (label "largest logits of every position") (!worst < tol) (error !worst);
      let argmax = ints (mem "argmax_per_position" recorded) in
      let differing = ref [] in
      Array.iteri
        (fun t expected ->
          let best = ref 0 in
          for v = 1 to vocab - 1 do
            if logits.((t * vocab) + v) > logits.((t * vocab) + !best) then
              best := v
          done;
          if !best <> expected then begin
            let values = List.nth top_values t in
            differing :=
              Printf.sprintf "position %d: %d for %d, reference margin %.1e" t
                !best expected
                (values.(0) -. values.(1))
              :: !differing
          end)
        argmax;
      check (label "argmax of every position") (!differing = [])
        (if !differing = [] then ""
         else "  (" ^ String.concat "; " (List.rev !differing) ^ ")")
    end
  in
  List.iter
    (fun (name, recorded) ->
      if only = [] || List.mem name only then prompt (name, recorded))
    (members (mem "prompts" fx))

let () =
  let fixture = ref "" and blocks = ref 0 and only = ref [] in
  let dtype = ref "float32" and tol = ref 0.0 in
  Arg.parse
    [
      ("--blocks", Arg.Set_int blocks, "Run the first N blocks only");
      ( "--prompt",
        Arg.String (fun p -> only := p :: !only),
        "Run this prompt only; may be repeated" );
      ("--dtype", Arg.Set_string dtype, "float32 (default) or bfloat16");
      ("--tol", Arg.Set_float tol, "Tolerance, in units of a stream's rms");
    ]
    (fun a -> fixture := a)
    "validate_stream.exe FIXTURE [--blocks N] [--prompt NAME] [--dtype DT] \
     [--tol X]";
  if !fixture = "" then failwith "validate_stream.exe: no fixture given";
  let fx = json_of_file !fixture in
  (* Float32 against float32 differs by the kernels and the order of
     accumulation, compounded over the blocks. Half precision keeps about three
     digits per block. *)
  let tol =
    if !tol > 0.0 then !tol else if !dtype = "float32" then 1e-3 else 1e-1
  in
  let (Gpt_oss.Dtype dt) = Gpt_oss.dtype_of_string !dtype in
  run ~tol
    ~blocks:(if !blocks > 0 then Some !blocks else None)
    ~only:!only fx dt;
  match !first_failure with
  | None -> Printf.printf "all checks passed (tolerance %.0e)\n" tol
  | Some name ->
      Printf.printf "%d checks failed; the first: %s\n" !failures name;
      exit 1
