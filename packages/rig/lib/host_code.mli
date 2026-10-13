(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The host's linker: relocatable ELF objects laid out in executable memory,
    as {!Rig.Image.load} on {!Rig.host} loads them ({!Rig.section-host_code}).
    Its C runs the launches of the code it links (the host's [submit] in
    [rig_host_code.c]). *)

val link : string -> (Def.linked, string) result
(** [link obj] is [obj] linked into new executable memory, or [Error why] for
    an object that breaks a rule of {!Rig.section-host_linking}, or that the
    process cannot map ([why] is the system's reason). The memory is unmapped
    once the result is collected. *)

val entry : Def.linked -> string -> int option
(** [entry c f] is the address of [c]'s function [f], a symbol of an
    executable section, or [None]. *)
