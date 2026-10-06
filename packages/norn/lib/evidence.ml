(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('u, 'f) t = {
  sample : ('u, 'f) Weighted.t;
  log_evidence : (float, 'f) Nx.t;
  error : (float, 'f) Nx.t;
  information : (float, 'f) Nx.t;
  stop : Nx.int32_t; (* [converged], [remaining] or [temperature] *)
  reached : (float, 'f) Nx.t; (* the payload of [stop] *)
}

type ('u, 'f) evidence = ('u, 'f) t

let converged = 0l
let remaining = 1l
let temperature = 2l

let v ~sample ~log_evidence ~error ~information ~stop ~reached =
  { sample; log_evidence; error; information; stop; reached }

let log_evidence z = z.log_evidence
let error z = z.error
let information z = z.information
let sample z = z.sample

let ptree (type u f) (u : u Nx.Ptree.t) : (u, f) t Nx.Ptree.t =
  let w = Weighted.ptree u in
  let module S = struct
    type _ t = (u, f) evidence

    let walk c z =
      let open Nx.Ptree.Walk in
      let sample = field c "sample" (structure w) z.sample in
      let log_evidence = field c "log_evidence" tensor z.log_evidence in
      let error = field c "error" tensor z.error in
      let information = field c "information" tensor z.information in
      let stop = field c "stop" tensor z.stop in
      let reached = field c "reached" tensor z.reached in
      { sample; log_evidence; error; information; stop; reached }
  end in
  Nx.Ptree.nest (module S) Nx.Ptree.unit

type stop = Converged | Remaining of float | Temperature of float

let float x = Nx.item [] (Nx.cast Nx.float64 x)

let stop z =
  let code = Nx.item [] z.stop in
  if code = remaining then Remaining (float z.reached)
  else if code = temperature then Temperature (float z.reached)
  else Converged

let pp ppf z =
  Format.fprintf ppf "ln Z = %.2f ± %.2f, H = %.2f nats" (float z.log_evidence)
    (float z.error) (float z.information);
  match stop z with
  | Converged -> ()
  | Remaining r -> Format.fprintf ppf ", budget spent with %.2f nats left" r
  | Temperature b -> Format.fprintf ppf ", budget spent at β = %.2f" b
