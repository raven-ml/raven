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
  let ckpt = Kaun_hf.load_checkpoint !repo in
  let loaded = since () in
  let (Llama.Dtype dt) =
    if !dtype = "" then Llama.stored_dtype ckpt
    else Llama.dtype_of_string !dtype
  in
  let device = if !device = "" then None else Some !device in
  let params = Llama.of_hf ?device cfg dt ckpt in
  let imported = since () in
  Printf.printf "load %.3f s, import %.3f s" loaded imported;
  Option.iter
    (fun device ->
      let ids = Nx.create Nx.int32 [| 1; 8 |] (Array.init 8 Int32.of_int) in
      let forward ids =
        Llama.logits cfg params
          (Nx.slice [ A; I 7 ] (Llama.hidden cfg params ids))
      in
      let logits = Rune.jit' ~device forward ids in
      ignore (Nx.item [ 0; 0 ] (Nx.cast Nx.float32 logits));
      Printf.printf ", first compiled call %.3f s" (since ()))
    device;
  print_newline ()
