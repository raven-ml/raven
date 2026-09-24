(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Character-level LSTM language model: the Raven side of the comparison.

   Written with the user-facing stack only — [Kaun.Embedding], [Kaun.Linear],
   [Kaun.Loss], [Vega.sgd_step], [Rune.value_and_grad], [Rune.jit] — and loading
   the weights the PyTorch side generated, so both programs train the same model
   from the same numbers on the same batches.

   Kaun has no recurrent layer, so the cell is written out: one linear map of
   the whole input sequence, then a loop over time applying the recurrent map
   and the four gates. That is the shape the model has anyway; what the loop
   costs relative to PyTorch's fused [nn.LSTM] kernel is one of the things this
   comparison measures.

   Implements the runner protocol in packages/kaun/bench/compare/README.md:

   model.exe variants model.exe run --spec S --fixture F --variant V --device D
   --steps N --cache {cold,warm} *)

(* Spec *)

type spec = {
  vocab : int;
  embed : int;
  hidden : int;
  seq_len : int;
  batch : int;
  batches : int;  (** the fixture's batch count, cycled over by the runner *)
  lr : float;
  momentum : float;
}

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let read_spec path =
  let members =
    match Jsont_bytesrw.decode_string Jsont.json (read_file path) with
    | Ok (Jsont.Object (members, _)) -> members
    | Ok _ -> failwith (path ^ ": expected an object")
    | Error e -> failwith (path ^ ": " ^ e)
  in
  let number name =
    match Jsont.Json.find_mem name members with
    | Some (_, Jsont.Number (v, _)) -> v
    | Some _ -> failwith (path ^ ": " ^ name ^ " is not a number")
    | None -> failwith (path ^ ": no \"" ^ name ^ "\" member")
  in
  {
    vocab = int_of_float (number "vocab");
    embed = int_of_float (number "embed");
    hidden = int_of_float (number "hidden");
    seq_len = int_of_float (number "seq_len");
    batch = int_of_float (number "batch");
    batches = int_of_float (number "steps");
    lr = number "lr";
    momentum = number "momentum";
  }

(* Model *)

module Lstm = struct
  type 'a t = {
    emb : 'a Kaun.Embedding.t;  (** [vocab; embed] *)
    ih : 'a Kaun.Linear.t;  (** [embed; 4 * hidden], the input-to-gate map *)
    hh : 'a Kaun.Linear.t;  (** [hidden; 4 * hidden], the state-to-gate map *)
    head : 'a Kaun.Linear.t;  (** [hidden; vocab] *)
  }

  let walk c p =
    let open Nx.Ptree.Walk in
    let emb = field c "emb" Kaun.Embedding.walk p.emb in
    let ih = field c "ih" Kaun.Linear.walk p.ih in
    let hh = field c "hh" Kaun.Linear.walk p.hh in
    let head = field c "head" Kaun.Linear.walk p.head in
    { emb; ih; hh; head }
end

(* The structure the transformations walk, and the momentum state over it. *)

let model = Nx.Ptree.instantiate (module Lstm)
let state = Nx.Ptree.pair model (Vega.sgd_ptree model)

(* Logits for a whole [batch; seq_len] id grid, as [batch; seq_len; vocab].

   The input-to-gate map does not depend on the recurrent state, so it applies
   to the sequence in one matmul; only the state-to-gate map runs per step. The
   four gate blocks sit side by side along the last axis in the order the
   fixture's weights store them: input, forget, cell, output. *)
let logits spec p ids =
  let x = Kaun.Embedding.apply p.Lstm.emb ids in
  let xg = Kaun.Linear.apply p.Lstm.ih x in
  let zero = Nx.zeros Nx.float32 [| spec.batch; spec.hidden |] in
  let h = ref zero and c = ref zero in
  let outputs = ref [] in
  for t = 0 to spec.seq_len - 1 do
    let gates =
      Nx.add (Nx.slice [ A; I t ] xg) (Kaun.Linear.apply p.Lstm.hh !h)
    in
    match Nx.split ~axis:1 4 gates with
    | [ gi; gf; gg; go ] ->
        let i = Nx.sigmoid gi and f = Nx.sigmoid gf in
        let g = Nx.tanh gg and o = Nx.sigmoid go in
        c := Nx.add (Nx.mul f !c) (Nx.mul i g);
        h := Nx.mul o (Nx.tanh !c);
        outputs := !h :: !outputs
    | _ -> assert false
  done;
  Kaun.Linear.apply p.Lstm.head (Nx.stack ~axis:1 (List.rev !outputs))

let loss spec p inputs targets =
  let y = logits spec p inputs in
  Kaun.Loss.softmax_cross_entropy_sparse
    (Nx.reshape [| spec.batch * spec.seq_len; spec.vocab |] y)
    (Nx.reshape [| spec.batch * spec.seq_len |] targets)

(* Training step

   The step's batch is read and the parameters and the optimizer state are
   consumed: a compiled step writes the updated state over the storage of the
   one it was given. The loss comes back beside the updated state. *)

let train_step spec inputs targets (params, opt) =
  let objective p = loss spec p inputs targets in
  let l, grads = Rune.value_and_grad model objective params in
  let state =
    Vega.sgd_step model ~lr:(Vega.lr spec.lr) ~momentum:spec.momentum opt
      ~params ~grads
  in
  (l, state)

(* Fixture

   The file is a PyTorch state dict, so its weights are in PyTorch's layout:
   [outputs; inputs] for every affine map, against Kaun's [inputs; outputs].
   Loading is therefore two moves — read the entries under their PyTorch names
   into a structure with the file's shapes, then transpose into the model. The
   weights are all float32, so they come out of the checkpoint as one structure;
   the token grid is int32 and comes out as the entry it is. *)

module Weights = struct
  type 'a t = {
    emb_weight : 'a;  (** [vocab; embed] *)
    head_bias : 'a;  (** [vocab] *)
    head_weight : 'a;  (** [vocab; hidden] *)
    bias_hh : 'a;  (** [4 * hidden] *)
    bias_ih : 'a;  (** [4 * hidden] *)
    weight_hh : 'a;  (** [4 * hidden; hidden] *)
    weight_ih : 'a;  (** [4 * hidden; embed] *)
  }

  (* Leaf paths are the state dict's own names. *)
  let walk c x =
    let open Nx.Ptree.Walk in
    let emb_weight = field c "emb.weight" leaf x.emb_weight in
    let head_bias = field c "head.bias" leaf x.head_bias in
    let head_weight = field c "head.weight" leaf x.head_weight in
    let bias_hh = field c "lstm.bias_hh_l0" leaf x.bias_hh in
    let bias_ih = field c "lstm.bias_ih_l0" leaf x.bias_ih in
    let weight_hh = field c "lstm.weight_hh_l0" leaf x.weight_hh in
    let weight_ih = field c "lstm.weight_ih_l0" leaf x.weight_ih in
    {
      emb_weight;
      head_bias;
      head_weight;
      bias_hh;
      bias_ih;
      weight_hh;
      weight_ih;
    }
end

let load_fixture spec path =
  let h4 = 4 * spec.hidden in
  let template =
    {
      Weights.emb_weight = Nx.zeros Nx.float32 [| spec.vocab; spec.embed |];
      head_bias = Nx.zeros Nx.float32 [| spec.vocab |];
      head_weight = Nx.zeros Nx.float32 [| spec.vocab; spec.hidden |];
      bias_hh = Nx.zeros Nx.float32 [| h4 |];
      bias_ih = Nx.zeros Nx.float32 [| h4 |];
      weight_hh = Nx.zeros Nx.float32 [| h4; spec.hidden |];
      weight_ih = Nx.zeros Nx.float32 [| h4; spec.embed |];
    }
  in
  let ckpt = Kaun.Checkpoint.load path in
  let f =
    Kaun.Checkpoint.to_value
      (Nx.Ptree.instantiate (module Weights))
      ~like:template ckpt
  in
  let tokens = Nx.unpack Nx.int32 (Kaun.Checkpoint.get "tokens" ckpt) in
  if Nx.shape tokens <> [| spec.batches; spec.batch; spec.seq_len + 1 |] then
    failwith (path ^ ": the token grid does not have the spec's shape");
  let linear w b =
    { Kaun.Linear.w = Nx.contiguous (Nx.transpose w); b = Some b }
  in
  let params =
    {
      Lstm.emb = { Kaun.Embedding.table = f.emb_weight };
      ih = linear f.weight_ih f.bias_ih;
      hh = linear f.weight_hh f.bias_hh;
      head = linear f.head_weight f.head_bias;
    }
  in
  (params, tokens)

(* Runner *)

let now_ms () = Unix.gettimeofday () *. 1e3

let run spec ~fixture ~variant ~device ~steps =
  let params, tokens = load_fixture spec fixture in
  let step =
    match variant with
    | "eager" -> train_step spec
    | "jit" ->
        let device = match device with "metal" -> "METAL" | _ -> "CPU" in
        Rune.jit
          ~devices:[ Rune.device device ]
          Nx.Ptree.(
            tensor @-> tensor @-> consumes state @@ returns (pair tensor state))
          (train_step spec)
    | v -> failwith ("unknown variant " ^ v)
  in
  let n_batches = (Nx.shape tokens).(0) in
  let state = ref (params, Vega.sgd_init model params) in
  let losses = Array.make steps 0. and step_ms = Array.make steps 0. in
  for i = 0 to steps - 1 do
    let batch = Nx.slice [ I (i mod n_batches) ] tokens in
    let inputs = Nx.contiguous (Nx.slice [ A; R (0, spec.seq_len) ] batch) in
    let targets =
      Nx.contiguous (Nx.slice [ A; R (1, spec.seq_len + 1) ] batch)
    in
    let t0 = now_ms () in
    let loss, next = step inputs targets !state in
    (* Reading the loss forces the step to complete: on a device the value is
       resident until read. The state stays unread and resident, which is what
       makes the next call transfer-free. *)
    let l = Nx.item [] loss in
    let t1 = now_ms () in
    losses.(i) <- l;
    step_ms.(i) <- t1 -. t0;
    state := next
  done;
  (losses, step_ms)

(* Output *)

let json_floats a =
  String.concat ", "
    (Array.to_list (Array.map (fun x -> Printf.sprintf "%.17g" x) a))

let emit ~variant ~device ~losses ~step_ms =
  Printf.printf
    {|{"side": "raven", "variant": "%s", "device": "%s", "losses": [%s], "step_ms": [%s], "version": "%s"}
|}
    variant device (json_floats losses) (json_floats step_ms) Sys.ocaml_version

(* Is a jit device usable in this build? Probed by compiling a trivial kernel on
   it: build-time device availability is not observable any other way, and a
   variant this process cannot run must not be reported as one it can. *)
let device_works name =
  match
    Rune.jit'
      ~devices:[ Rune.device name ]
      (fun x -> Nx.add x x)
      (Nx.ones Nx.float32 [| 4 |])
  with
  | (_ : Nx.float32_t) -> true
  | exception _ -> false

let emit_variants () =
  let variants =
    ("eager", "cpu")
    :: List.filter_map
         (fun (v, d, probe) -> if device_works probe then Some (v, d) else None)
         [ ("jit", "cpu", "CPU"); ("jit", "metal", "METAL") ]
  in
  Printf.printf {|{"side": "raven", "variants": [%s]}
|}
    (String.concat ", "
       (List.map
          (fun (v, d) ->
            Printf.sprintf {|{"variant": "%s", "device": "%s"}|} v d)
          variants))

(* Command line *)

let usage () =
  prerr_endline
    "usage: model.exe variants\n\
    \       model.exe run --spec S --fixture F --variant V --device D --steps \
     N [--cache cold|warm]";
  exit 2

let flags argv =
  let rec go acc = function
    | [] -> acc
    | flag :: value :: rest when String.length flag > 2 && flag.[0] = '-' ->
        go ((String.sub flag 2 (String.length flag - 2), value) :: acc) rest
    | _ -> usage ()
  in
  go [] argv

let required flags name =
  match List.assoc_opt name flags with Some v -> v | None -> usage ()

let () =
  match Array.to_list Sys.argv with
  | [ _; "variants" ] -> emit_variants ()
  | _ :: "run" :: rest ->
      let flags = flags rest in
      (* Cold means no compiled program is served from an earlier process. Read
         per call by the jit cache, so setting it here is enough. *)
      if List.assoc_opt "cache" flags = Some "cold" then
        Unix.putenv "JITCACHE" "0";
      let spec = read_spec (required flags "spec") in
      let variant = required flags "variant" in
      let device = required flags "device" in
      let steps = int_of_string (required flags "steps") in
      let losses, step_ms =
        run spec ~fixture:(required flags "fixture") ~variant ~device ~steps
      in
      emit ~variant ~device ~losses ~step_ms
  | _ -> usage ()
