(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let repo = "polymathic-ai/aion-base"

(* The commit the PyTorch reference and upstream's goldens were checked
   against, 2026-11-18. *)
let revision = "40541618104bab0fa85c8af68daeb867a720bb8c"

let codec ?cache_dir ?offline name =
  let file = "codecs/" ^ name ^ "/model.safetensors" in
  Nx_io.load_safetensors
    (Kaun_hf.download_file ?cache_dir ?offline ~revision ~file repo)

type model = Base | Large

(* aion-large's commit, read on 2026-11-19; its codecs/ files equal
   aion-base's at revision, byte for byte. *)
let large = ("polymathic-ai/aion-large", "cfdb89e8a22ddbdbbc4fe65c30fa2eb6eb7f9bcd")

let weights ?cache_dir ?offline m =
  let repo, revision = match m with Base -> (repo, revision) | Large -> large in
  Nx_io.load_safetensors
    (Kaun_hf.download_file ?cache_dir ?offline ~revision ~file:"model.safetensors"
       repo)
