(* This file is selected when nx.metal and rune.jit.metal are available *)

let is_available = true

include Nx_metal
module Jit_backend = Rune_jit_metal
