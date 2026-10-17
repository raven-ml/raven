(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The channel formats follow NVIDIA's class headers in
   open-gpu-kernel-modules 570.144: clc56f.h (a channel's host methods, ring
   entries and pushbuffer headers), clc7c0.h (compute) and clc7b5.h
   (copy). *)

module Template = Template
module Method = Method
module Gpfifo = Gpfifo
module Gpu = Gpu

let key : Gpu.t Type.Id.t = Type.Id.make ()
let declare c = Rig_edge.Capability (key, c)

module Local_memory = Local_memory
module Cubin = Cubin
module Launch = Launch
module Qmd = Qmd
