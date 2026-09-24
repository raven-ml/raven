(* Parity case: a running sum along the leading axis of a (1000, 3) tensor.

   An axis longer than 512 scans in two stages. The axis is padded to 1024 and
   split into four chunks of 256, each scanned on its own; the four chunk
   totals are scanned in turn, shifted by one so each chunk sees the total
   before it, and added back to every element of their chunk. The padding is
   dropped from the front. Three kernels: the chunk scans, the scan of their
   totals, and the combine.

   Backends are limited to cpu and metal: kernel-name counters are shared
   across backends, so the reference must be generated with exactly the
   backends the OCaml side renders.

   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_frontend

let backends =
  List.filter
    (fun (name, _) -> name = "cpu" || name = "metal")
    Helpers.all_backends

let build () =
  let x = Tensor.of_uop (Helpers.mk_param ~idx:0 [ 1000; 3 ]) in
  Helpers.wrap_sink [ Tensor.uop (Op.cumsum ~axis:0 x) ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
