(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Functions nx composes from kinds and exact operations, compiled by Rune.jit
   on the host, give eager's bits over drawn operands (Kinds), and a function of
   one or two operands at the narrow floats' codes (Kinds.narrow). *)

open Windtrap
open Rune_support

let law k =
  prop ~count:12
    ~examples:[ Kinds.example k ]
    (Kinds.name k ^ " gives eager's bits")
    (Kinds.gen k)
    (Kinds.law (module Nx.Host) ~f64:true k)

let narrow k =
  test
    (Kinds.name k ^ " of every narrow float gives eager's bits")
    (fun () -> Kinds.narrow (module Nx.Host) k)

let unary =
  Kinds.
    [
      asinh;
      acosh;
      atanh;
      rsqrt;
      erfc;
      erfinv;
      ndtr;
      log_ndtr;
      ndtri;
      lgamma;
      digamma;
      i0e;
      i1e;
    ]

let binary =
  Kinds.
    [
      hypot;
      lbeta;
      gammainc;
      gammaincc;
      log_gammainc;
      log_gammaincc;
      gammaincinv;
      gammainccinv;
    ]

let ternary =
  Kinds.
    [
      betainc;
      betaincc;
      log_betainc;
      log_betaincc;
      betaincinv;
      betainccinv;
    ]

let () =
  exit
    (run "Rune.jit composed functions"
       [
         group "composed" (List.map law (unary @ binary @ ternary));
         group "narrow floats" (List.map narrow (unary @ binary));
       ])
