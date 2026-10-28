(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* One configuration of the load benchmark: load a Llama checkpoint from the
   cache, import it at a dtype, and with [--device] place each leaf on the
   device as it is imported and run one compiled forward pass over eight tokens.
   It prints the wall time to the end of each phase; [run.sh] measures the
   process's peak memory around it, one process per configuration. *)

let () =
  let repo = ref Llama.default_repo in
  let dtype = ref "" and device = ref "" in
  Arg.parse
    [
      ("--repo", Arg.Set_string repo, "Repository (default: Llama 3.2 1B)");
      ( "--dtype",
        Arg.Set_string dtype,
        "float32, float16 or bfloat16 (default: the checkpoint's own)" );
      ( "--device",
        Arg.Set_string device,
        "Run one compiled forward pass on this device" );
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "bench_load.exe [--repo REPO] [--dtype DT] [--device DEVICE]";
  let t0 = Unix.gettimeofday () in
  let since () = Unix.gettimeofday () -. t0 in
  let cfg = Llama.config_of_json (Kaun_hf.load_config !repo) in
  let weights = Kaun_hf.load_safetensors !repo in
  let loaded = since () in
  let (Llama.Dtype dt) =
    if !dtype = "" then Llama.stored_dtype weights
    else Llama.dtype_of_string !dtype
  in
  (* [import at] imports each leaf at [at] as it is read. *)
  let import at =
    let params = Llama.of_hf ~placement:(fun _ ~axis:_ -> at) cfg dt weights in
    let imported = since () in
    Printf.printf "load %.3f s, import %.3f s" loaded imported;
    params
  in
  if !device = "" then ignore (import Nx.Host.on)
  else begin
    let module D = (val Nx.devices [ Devices.first !device ]) in
    let params = import D.on in
    let cfg = Llama.place_config D.on cfg in
    let ids =
      Nx.place D.on (Nx.create Nx.int64 [| 1; 8 |] (Array.init 8 Int64.of_int))
    in
    let forward ids =
      Llama.logits cfg params
        (Nx.slice [ A; I 7 ] (Llama.hidden cfg params ids))
    in
    let before = Rune.evaluations () in
    let logits = Rune.jit' forward ids in
    ignore (Nx.item [ 0; 0 ] logits : float);
    if Rune.evaluations () > before then
      failwith "bench_load: the first compiled call ran evaluated";
    Printf.printf ", first compiled call %.3f s" (since ())
  end;
  print_newline ()
