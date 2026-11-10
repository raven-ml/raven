(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The host's linker: relocatable ELF objects laid out in executable memory,
    as {!Rig.Image.load} on {!Rig.host} and on a software device loads them
    ({!Rig.section-host_linking}). Its C runs the launches of the code it
    links (the host's and the software device's [submit] in
    [rig_host_code.c]). *)

val link : string -> (Def.linked, string) result
(** [link obj] is the link of [obj]: the one an earlier call made, while it
    is reachable, or else [obj] linked into new executable memory. It is
    [Error why] for an object that breaks a rule of
    {!Rig.section-host_linking}, or that the process cannot map ([why] is the
    system's reason). Linked code holds no variables, so one link serves
    every image of its object. The memory is unmapped once the link is
    collected. *)

val entry : Def.linked -> string -> int option
(** [entry c f] is the address of [c]'s function [f], a symbol of an
    executable section, or [None]. *)

val on_exchange : (int -> unit) -> unit
(** [on_exchange f] makes the next new link call [f a] once, [a] being the
    link's first address, after {!link} found no link of its object and before
    it adds its own. [f] is unset before it runs. For rig's tests
    ({!Rig.on_host_exchange}). *)
