(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Compiled programs, each lowered through tolk and linked once per variant of
    its arguments.

    This module implements the compiled programs of {!Rune.jit}
    ({!Rune.section-compiled}). The rules a caller sees, on donation
    ({!Rune.section-donation}), checks ({!Rune.section-checks}) and evaluated
    programs ({!Rune.section-evaluated}), are stated there.

    To {e lower} a program is to translate it into a tolk graph, and to {e link}
    it is to compile that graph into a schedule tolk's engine runs. Lowering or
    linking {e declines} a program it cannot compile, and gives a reason.
    {!make} lists lowering's reasons, and an [Invalid_argument] or [Failure]
    that lowering or linking raises is a decline too. nx then {e evaluates} the
    program, in the sense that it applies each operation eagerly ({!evaluate}).
    A program whose values lie on one device has that {e device}: the one its
    first argument, or else its first result, lies whole on. A program over
    several devices is {e split} first ({!Pieces}): its program of one-device
    operations and copies lowers each operation on its own device, binds each
    piece of an argument as a parameter of its own, and gathers each result
    over several devices from its pieces.

    An argument's {e run} is the storage its layout reads, from the element at
    or below the lowest position the layout reaches whose bits start a multiple
    of 16 bytes from its buffer's start, through the highest position. Each
    parameter binds its argument's run ({!within}), each constant is bound once,
    and each result is stored into memory of its own or into a donated
    argument's. A run's {e phase} is the number of bytes by which its first byte
    lies past an address that is a multiple of 16.

    A {e variant} of a program is, for each argument, its layout over its run,
    its run's phase and whether it is donated, together with the value of tolk's
    settings that shape what it compiles at its search width
    ({!Tolk.Jit.shaping}) and whether
    {!Tolk.Setting.debug} is 2 or more. The life cycle of a compiled program
    goes like this:
    + {!make} lowers the program once, to find whether lowering declines it,
      and links nothing. It lowers it at the variant that the first call's
      arguments' forms give, their layouts and donation at phase [0], for a
      program on one device. Where lowering declines that variant, or for a
      split program or without a first call, it lowers it over C-contiguous
      arguments at phase [0] that none donates, and a decline there
      declines the program.
    + A call ({!run}) at a variant new to the program lowers and links that
      variant, or keeps its decline.
    + Later calls at that variant run its schedule, or evaluate the program
      where the variant was declined.

    A compiled program may be run from any domain. It links one variant at a
    time, and a call that needs a new variant while another call links one waits
    for it. When lowering or linking a variant raises other than to decline, the
    program keeps nothing for that variant, and the next call at it links again.
    The counters {!variants} and {!evaluations} count per domain. *)

type t
(** The type for compiled programs. *)

val within : ('v, 's) Nx.Dtype.t -> Nx_array.Layout.t -> Nx_array.Layout.t
(** [within dt l] is [l], a layout of a value of [dt], with its offset counted
    from the start of its run. It is [l] where [l] has no element. *)

val report : ('a, out_channel, unit) format -> 'a
(** [report fmt] prints [rune.jit: ] then [fmt]'s text as one line on standard
    error and flushes it, if the variable [RUNE_JIT_DEBUG] held a nonzero
    integer when the module was initialised. Otherwise it prints nothing. *)

val make :
  name:string ->
  ?beam:int ->
  ?parallel:int ->
  ?first:Program.value array ->
  Program.t ->
  Program.form array ->
  (t, string) result
(** [make ~name ~beam ~parallel ~first p forms] is [p] ready to run over
    arguments of [forms], its kernels searched at the width [beam], or else the
    one tolk's settings give, on [parallel] domains, or else as many as they
    give; or [Error why] where lowering declines it, where [why] names the
    value, construct or failure. [name] is the entry point that {!run}'s
    messages name. [first] are the arguments of the first call, of which [make]
    reads the forms and donation, and nothing else: a first call at the
    variant they give links the lowering [make] made.

    Lowering declines a map's construct ({!Program.Lanes},
    {!Program.Lane_index}), a scan that runs some of its rows, an iterate that
    stacks its steps or whose lanes stop apart, a root, a root's linear solve or
    operator, an addition, a linear call with no function to apply, an operation
    that {!Lower.op} does not lower, a check whose flags lie on another set than
    the program's first argument or else its first result, a program with no
    value, and a value with an element on a device other than the program's
    device. It declines what {!Pieces.split} declines. It also declines a
    program with a value on a device that computes nothing. A split program's
    values may lie on devices of several targets: each device's kernels compile
    for its own. *)

val run : t -> Program.value array -> Program.value array
(** [run t args] is the outputs of [t]'s program applied to [args], which must
    be untraced values of the forms [t] was made for. Each result is fresh
    memory in C order at its output's form's placement, or the memory of a
    donated argument it takes ({!Rune.section-donation}).

    The call is a consumer of [args] ({!Nx.Prim.Consumer}). It claims the
    donated handles among them, and reads each argument as it was at the claim.
    Where {!Nx.Prim.Consumer.take} does not give the memory of an argument a
    result takes, the result computes into a copy of it. A declined variant's
    results are all copies. Once the program is submitted, the call issues its
    checks as one check of nx on the program's set ({!Rune.section-checks}),
    which raises the first that failed, in the program's order: at once on the
    host, and at the next wait on another set.

    A split program reads each argument's arrays, one per device of its
    placement, and lends none of them.

    Raises [Invalid_argument] if [args] lie on several device sets
    ({!Nx.Placement.devices}) or if {!Nx.Prim.Consumer.with_} raises it, and
    what linking a new variant raises other than to decline, such as an
    interrupted search. *)

val variants : unit -> int
(** [variants ()] is the number of variants that calls on the calling domain
    lowered and linked, or whose decline they kept. *)

val evaluations : unit -> int
(** [evaluations ()] is the number of calls on the calling domain that evaluated
    their program: each {!run} of a declined variant, plus each call counted by
    {!evaluated_call}. *)

val evaluated_call : unit -> unit
(** [evaluated_call ()] counts one call in {!evaluations}, for a caller that
    evaluates a program {!make} declined. *)

val evaluate :
  fresh:bool ->
  name:string ->
  Program.t ->
  Program.value array ->
  Program.value array
(** [evaluate ~fresh ~name p args] is [p]'s outputs evaluated over [args], its
    operations by nx and its calls as their sites run them, none of them traced,
    each a copy in memory of its own. Where [fresh] holds, [p]'s outputs are
    each already in memory of its own, as those of {!Program.outline}'s programs
    are, and are not copied. Where [args] hold donated handles, it claims every
    argument first, as {!run} does, with [name] in its messages, and evaluates
    [p] over their arrays. *)
