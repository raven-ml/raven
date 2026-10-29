(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Programs, which record a traced function, and the constructs that carry
    them.

    A {e program} has parameters, equations and outputs. Its {e variables} are
    the values a tracer ({!Trace}) made while it traced, each numbered by its
    payload ({!Var}) below the program's [slots]. A traced program numbers them
    from [0] in the order the tracer met them, the function's arguments first. A
    program made from another, as by {!widen}, may number them in another order.
    An equation applies an [Nx.Prim] operation or a construct to variables and
    {e constants}, the values no [Nx.Prim] interpretation traces: an array, or a
    donated handle ([Nx.donate]) read as its arrays. A {e construct} is an
    operation that a transformation answers by a rule of its own, such as a
    loop, which carries its step's program. The docs below describe a program as
    {e over} its parameters and {e to} its outputs, each listed in order.

    {!eval} issues each equation again, an operation through [Nx.Prim.eval] and
    a construct through {!perform}, under the interpretations around the call. A
    total's {e running value} is its start plus the additions made to it so far
    within the innermost collect ({!Rune2.Total.collect}) or tracer that holds
    it, where a collect starts from its zero and a tracer from a parameter. A
    program {e threads} each total it adds to: its last parameters are the
    totals' running values in, its additions add to them in the order they were
    made, and its last outputs are the {e running values out}. A construct whose
    programs thread totals has one {e sum} per total and shape they thread, the
    chain of that total's additions in the order they are made ({!sums}). The
    construct takes each sum's start from the collect or tracer around its call
    and hands it back the sum's end ({!run}), so evaluating a program adds to no
    total by itself.

    A construct is performed as one effect ({!Construct}). The innermost handler
    that answers it gives its results: a tracer's handler, a map, forward mode,
    or for an [Add] a collect of its total. One that no handler answers, or that
    a handler discontinues with {!Evaluate}, runs its {e default} where it was
    performed, inside every interpretation around that point. Each constructor
    states its default.

    A program is never changed once made, and any domain may evaluate it. A
    call's {!site} keeps the programs transformations derive from it, one per
    derivation, which the calls of every domain share ({!derive}). *)

(** {1:values Values}

    A program's variables and constants are nx values of any dtype on any device
    set, held as {!value}s. A {!form} is the dtype, layout and placement a
    tracer records for one. *)

type value =
  | V : ('v, 's, 'd) Nx.t -> value  (** A value of any dtype and set. *)

type form =
  | F : ('v, 's, 'd) Nx.Prim.form -> form
      (** A value's form, of any dtype and set. *)

val form : value -> form
(** [form v] is [v]'s form. *)

val cast : ('v, 's) Nx.dtype -> 'd Nx.devices -> value -> ('v, 's, 'd) Nx.t
(** [cast dt s v] is [v] at [dt] and [s]'s brand, as {!Nx.Prim.coerce} checks
    it.

    Raises [Invalid_argument] if [v]'s dtype is not [dt] or [v] does not lie on
    [s]. *)

val coerce_at : ('v, 's, 'd) Nx.Prim.form -> ('w, 'r, 'e) Nx.t -> ('v, 's, 'd) Nx.t option
(** [coerce_at f x] is {!Nx.Prim.coerce} at [f]'s dtype and set. *)

val cast_at : ('v, 's, 'd) Nx.Prim.form -> value -> ('v, 's, 'd) Nx.t
(** [cast_at f v] is {!cast} at [f]'s dtype and set. *)

val like : ('v, 's, 'd) Nx.t -> value -> ('v, 's, 'd) Nx.t
(** [like x v] is {!cast} of [v] at [x]'s dtype and set. *)

val same : value -> value -> bool
(** [same u v] is [true] iff [u] and [v] are one value. *)

(** {1:names Names}

    A map's axis and a total are names. Each is fresh when made, on any domain,
    and is compared by identity. *)

type axis
(** The type for names of maps. *)

val axis : unit -> axis
(** [axis ()] is a fresh name. *)

val same_axis : axis -> axis -> bool
(** [same_axis a b] is [true] iff [a] and [b] are one name. *)

type ('v, 's, 'd) total
(** The type for totals of [('v, 's, 'd) Nx.t] values. A {e total} names a sum
    that additions ({!add_to}) accumulate within a collect
    ({!Rune2.Total.collect}). *)

val total : unit -> ('v, 's, 'd) total
(** [total ()] is a fresh total. *)

val same_total :
  ('v, 's, 'd) total ->
  ('w, 'r, 'e) total ->
  (('v, 's, 'd) Nx.t, ('w, 'r, 'e) Nx.t) Type.eq option
(** [same_total t u] is [Some Equal] iff [t] and [u] are one total. *)

val total_id : ('v, 's, 'd) total -> int
(** [total_id t] is a number no other total has. *)

type 'a probe =
  | Probe : { id : 'a Type.Id.t; tree : ('a, 'd) Nx.Ptree.t } -> 'a probe
      (** A probe of values of [tree]'s structure. *)

val probe : ('a, 'd) Nx.Ptree.t -> 'a probe
(** [probe s] is a fresh probe of values of [s]'s structure. *)

val probe_id : 'a probe -> int
(** [probe_id p] is a number no other probe has. *)

val same_probe : 'a probe -> 'b probe -> ('a, 'b) Type.eq option
(** [same_probe p q] is [Some Equal] iff [p] and [q] are one probe. *)

val fresh_site : unit -> int
(** [fresh_site ()] is a number no other site has. *)

type asked =
  | Asked : 'a probe * bool -> asked
      (** A probe a trace asked about, and whether a collect of it was open. *)

(** {1:programs Programs}

    A program's equations are in the order they were issued, and each reads only
    parameters, constants and earlier equations' results. *)

type ('v, 's, 'd) Nx.Prim.payload +=
  | Var : int -> ('v, 's, 'd) Nx.Prim.payload  (** A variable's number. *)

type addition =
  | Addition :
      ('v, 's, 'd) total * ('v, 's, 'd) Nx.Prim.form * ('v, 's, 'd) Nx.Prim.form
      -> addition
      (** A total a program threads, and the forms of its running value in and
          out. *)

type t = private {
  owner : Nx.Prim.interpretation;  (** The tracer whose variables it has. *)
  params : int array;
      (** The variables its arguments bind, in order, then each total's running
          value in. A traced program's arguments end with its captures. *)
  eqns : eqn array;  (** In the order they are issued. *)
  outputs : value array;
      (** Variables or constants, each total's running value out last. *)
  adds : addition array;  (** The totals it threads, in order. *)
  slots : int;  (** The number of variables. *)
  emits : bool;
      (** Whether an equation emits ({!Emit}), or a program its constructs carry
          does. *)
}

and eqn =
  | Op : { by : string; op : 'r Nx.Prim.t; base : int } -> eqn
      (** [op], applied by the function named [by] in messages. Its result [k]
          is the variable [base + k]. *)
  | Do : { c : construct; base : int } -> eqn
      (** [c]. Its result [k] is the variable [base + k]. *)

and construct =
  | Detach : ('v, 's, 'd) Nx.t -> construct
      (** The operand, with a zero derivative; one result. Its default is the
          operand. *)
  | Lanes : axis * ('v, 's, 'd) Nx.t -> construct
      (** Every lane's operand stacked along a new axis 0, in the map named by
          the axis; one result. The map answers it where no tracer inside the
          map receives it. Its default raises [Invalid_argument]. *)
  | Lane_index : axis * 'd Nx.Placement.t -> construct
      (** The calling lane's index in the map named by the axis, one 0-d [int32]
          result at the placement, which must cut no axis. It is answered as
          [Lanes] is, and its default raises [Invalid_argument]. *)
  | Add : ('v, 's, 'd) total * ('v, 's, 'd) Nx.t -> construct
      (** The operand added to the total; no result. Its default does nothing.
      *)
  | Emit : emission -> construct
      (** An emission offered to the innermost collect of its probe; no result.
          Its default does nothing. *)
  | Scan : {
      step : t;
          (** Over the carry, a row and the captures; to the carry, the outputs
              and the running values out. *)
      length : int;  (** The number of rows. *)
      trips : value option;
          (** A 0-d [int32] from [0] to [length], the number of leading rows
              that run. Every row runs where it is [None]. The outputs at the
              other rows are zeros. *)
      reverse : bool;  (** Whether the rows that run run from the last. *)
      init : value array;
      xs : value array;  (** Rows along axis 0. *)
      captures : value array;
      seeds : value array;
          (** Where each sum starts, one per sum in {!sums}' order. Each starts
              at the identity [-0.] ({!eval}) where [seeds] is empty. *)
    }
      -> construct
      (** [step] folded over the rows of [xs] from the carry [init]. Its results
          are the last carry, each output stacked along a new axis 0, then the
          sums. Its default evaluates [step] once per row that runs, and raises
          [Invalid_argument] where [trips] lies outside [0] to [length], which a
          compiled program clamps to them. *)
  | Iterate : {
      name : string;  (** The entry point, in messages. *)
      step : t;
          (** Over the carry and its captures; to the carry, the outputs where
              [rows] stacks them, and the running values out. *)
      until : t;  (** Over the carry and its captures; to one boolean. *)
      max : int;
      rows : rows option;
          (** Where given, the results after the carry are the number of steps
              taken, a 0-d [int32] at the uncut placement of [until]'s last
              result, then each output stacked along a new axis 0, one row per
              step as {!rows} counts them. *)
      stopped : int option;
          (** The index of the carry holding one stop flag per lane of the maps
              around the loop. When [until] is still false after [max] steps,
              the message names the first lane whose flag is false. *)
      init : value array;
      step_captures : value array;
      until_captures : value array;
      seeds : value array;  (** As a scan's. *)
    }
      -> construct
      (** [step] applied to the carry from [init] while [until] is false, at
          most [max] times. Its results are the last carry, [rows]' results,
          then the sums. Its default evaluates [until] and [step] in turn,
          [until] at most [max + 1] times, and raises [Invalid_argument] where
          [until] is still false after [max] steps. *)
  | Call : {
      body : t;
      args : value array;
      site : site;
          (** [body]'s compiled form where it has one, applied to arguments of
              [args]' forms then its totals' running values in, and the programs
              derived from it. *)
      seeds : value array;  (** As a scan's. *)
    }
      -> construct
      (** [body] applied to [args]. Its default runs the site's compiled form
          where the site has one, no operand and no running value in is traced,
          and no tracer stages the operations around the call ({!staging}). It
          evaluates [body] otherwise. *)
  | Custom : {
      name : string;  (** The entry point, in messages. *)
      primal : t;
          (** Over the arguments and the captures; to the [ny] results, the
              residuals and the running values out. *)
      derivative : derivative;
      ny : int;
      args : value array;
      captures : value array;
      residuals : bool;
          (** Whether the residuals among the results, the values the derivative
              reads beside the arguments, are used. Where they are, they carry
              the derivatives of every differentiation around the call. A
              differentiation applying the rule asks the call it performs
              outward for them, and one asked for its own residuals derives them
              again. *)
      applied : bool;
          (** Whether a reverse mode inside the call applies its {!Pullback}.
              Where it does, a forward mode around differentiates the primal's
              code. Where it does not, a forward mode raises for a result with a
              tangent ({!Jvp}). *)
      seeds : value array;  (** Where [primal]'s sums start, as a scan's. *)
    }
      -> construct
      (** A call of a function with a custom derivative: the results, the
          residuals (unused where [residuals] is [false]), [primal]'s sums, then
          a tangent map's running values out. The default evaluates [primal],
          without what only the residuals read where [residuals] is [false]. A
          tangent map's running values out are then zeros. *)
  | Linear : {
      name : string;
      apply : t option;
          (** Over the arguments and the captures; to the results and the
              running values out. [None] where only the transpose is known, as
              for a {!Pullback} rule's tangents, and always for {!Host}. *)
      transpose : transpose;
      args : value array;
      captures : value array;
      residuals : value array;
      forms : form array;  (** The results'. *)
    }
      -> construct
      (** A function linear in its arguments with the transpose [transpose]:
          [apply]'s results, then its running values out. Its default evaluates
          [apply]. Without [apply], it is the tangents of a {!Pullback} rule's
          results, which only a tape receives, and its default raises
          [Invalid_argument]. *)
  | Remat : {
      body : t;
      args : value array;
      seeds : value array;
      differentiated : bool;
    }
      -> construct
      (** [body] applied to [args], the function's arguments then its captures.
          Its reverse mode keeps [args] alone and computes [body] again. Its
          sums start from [seeds], as a scan's. Its default evaluates [body].
          [differentiated] is whether [body] is a part a differentiation
          derived, which computes the function's values again: compiled, it
          reads those of [args] the program computes through a barrier, so
          that what it computes from them shares no value with the program
          around it and runs once every one of them is computed. *)
  | Switch : switch -> construct
      (** The branch at [index] applied to [captures]; the others run nothing.
          Its sums are every branch's, as a loop's. A branch that does not run
          leaves its own sums where they start. Its default reads [index] and
          raises [Invalid_argument] where no branch has it. *)
  | Operator : {
      id : int;  (** The operator's number. *)
      transposed : bool;  (** Whether it applies the operator's transpose. *)
      lanes : (int * bool array) list;
          (** The maps it applies the operator over, outermost first, each with
              its lanes and which of [data] hold them. The arguments and those
              data hold every lane's value along leading axes of these lengths.
          *)
      args : value array;
      data : value array;
          (** The values the operator is made from, on which it does not act
              linearly. *)
      forms : form array;  (** The results' forms, which are the arguments'. *)
    }
      -> construct
      (** An abstract linear operator, as a linear solve is traced against,
          applied to [args], one value per array of the space it acts on. A
          program holding one must be filled ({!fill}) before it is evaluated.
          Its default raises [Invalid_argument]. *)
  | Root : {
      name : string;  (** The entry point, in messages. *)
      system : system;
          (** Its programs, at the level of the construct, so that a map of the
              construct maps them. *)
      solution : value array;  (** The solution's arrays. *)
      theta : value array;
          (** The values from outside the residual reads, which the solution
              depends on. *)
      captures : value array;  (** The linear solve's captures. *)
    }
      -> construct
      (** [solution], stated to be a zero of [system.residual] over it and
          [theta]. Its default is [solution]; its derivative solves the system's
          derivative ({!Solve}). *)
  | Solve : {
      name : string;
      system : system;
      transposed : bool;  (** Whether it solves the transposed system. *)
      b : value array;
          (** The right-hand side, one array per array of the solution. *)
      data : value array;  (** The solution, then [theta]. *)
      captures : value array;  (** The linear solve's captures. *)
    }
      -> construct
      (** The [v] with [J v = b], or [Jᵀ v = b] where [transposed], where [J] is
          the residual's derivative along the solution at [data]. It is the
          system's linear solve with its operator filled ({!fill}) by [J] or
          [Jᵀ], and is linear in [b]. Its default fills the operator and
          evaluates the linear solve. *)

and system = {
  id : int;
      (** The number of the operator ({!Operator}) the linear solve is traced
          against. *)
  operator :
    transposed:bool -> lanes:(int * bool array) list -> t * value array;
      (** The programs of [J] and [Jᵀ] at one lane, over the maps an operator
          lies under ({!fill}). *)
  residual : t;  (** Over the solution then [theta]. *)
  jacobian : t;  (** [J v], over [v], the solution then [theta]. *)
  transpose : t;  (** [Jᵀ w], over [w], the solution then [theta]. *)
  solve : t;
      (** The linear solve against the operator [id], over the right-hand side,
          the solution, [theta], then its captures. *)
}

and switch = {
  name : string;  (** The entry point, in messages. *)
  index : value;  (** A 0-d [int32]: the branch that runs. *)
  branches : t array;
      (** Each over [captures], then its totals' running values in; to the
          results, of one form in every branch, then its running values out. *)
  captures : value array;
  seeds : value array;  (** As a scan's. *)
}

(** An emission of one site: [values] are the arrays of a value of [like]'s
    structure, each with [lead] axes of lanes, then one of rows, one per
    execution of the site, then the value's shape. *)
and emission =
  | Emission : {
      probe : 'a probe;
      like : 'a;  (** Of the emitted structure, its arrays unread. *)
      site : int;  (** The site: one per [emit] call, kept by a trace. *)
      lead : int;
      values : value array;
      mask : value option;
          (** A boolean over the first [lead + 1] axes of [values]: the rows
              that ran, where some did not, as a held lane's. *)
    }
      -> emission

and rows =
  | Taken
      (** As many rows as steps taken. No tracer can stage such an iterate. *)
  | Max  (** [max] rows, zeros past the steps taken. *)

and derivative =
  | Tangent of t
      (** A tangent map: over a tangent per argument and the residuals; to the
          results' tangents and the additions. *)
  | Pullback of t
      (** A pullback: over a cotangent per result and the residuals; to a
          cotangent per argument and the additions. *)

and transpose =
  | Traced of t
      (** Over a cotangent per result and the residuals; to a cotangent per
          argument and the running values out. *)
  | Host of (value array -> value array)
      (** A custom rule's pullback run on values ({!Jvp}): from a cotangent per
          result to a cotangent per argument, its additions made where it runs.
          A tape holds one, and a trace of that tape's application; only its
          backward pass ({!Linear}) runs it, and a pass that needs the
          transpose's program raises [Invalid_argument]. *)

and site
(** The type for a call's {e site}: its program, compiled where the site is made
    by {!site}, and the programs transformations derive from it ({!derive}). *)

val make :
  owner:Nx.Prim.interpretation ->
  params:int array ->
  eqn list ->
  outputs:value array ->
  adds:addition array ->
  slots:int ->
  t
(** [make ~owner ~params eqns ~outputs ~adds ~slots] is the program of the
    equations [eqns], in order. *)

val sums : t list -> addition array * form array * int array list
(** [sums ps] is [(adds, forms, feeds)], the sums of a loop of the programs
    [ps]. [adds] holds each sum's first addition, in the order [ps] first thread
    them, and [forms] the sums' forms. [feeds] holds, per program, the sum each
    of its totals feeds, in order. *)

val feeds : t -> addition array * form array * int array
(** [feeds step] is {!sums} of [[step]], with [step]'s feeds alone. *)

val lane_names : int array -> int -> string
(** [lane_names shape i] is how an iterate whose lanes stop apart names, when it
    runs out of steps, the lane at index [i] in C order of lanes of [shape]:
    [", in lane k"] for each axis's position [k], outermost first. It is [""]
    where [i] is at least the number of lanes. *)

val eval : ?totals:value array -> t -> value array -> value array
(** [eval ~totals p args] is [p]'s outputs with its arguments bound to [args]
    and its totals' running values in to [totals], in order. Without [totals],
    each running value in is [-0.], the identity that keeps every sign ([0]
    where the dtype has no negative zero), so that the running values out are
    the sums of its additions. Raises [Invalid_argument] where [args] and
    [totals] do not match [p]'s parameters, and what an equation raises. *)

val eval_seeded : t -> seeds:value array -> value array -> value array
(** [eval_seeded p ~seeds args] is [p]'s outputs over [args], its sums started
    from [seeds] as a scan's are. *)

val prune : t -> outputs:int array -> t
(** [prune p ~outputs] is [p] with its outputs [outputs], in order, then its
    totals' running values out. It keeps the equations they read and every
    check. *)

(** {1:sites Sites}

    A transformation of a call derives a program from the call's ({!derive}):
    its forward-mode derivative, its map, its parts known and unknown to a tape,
    or its transpose. A site keeps one program per derivation and argument
    forms, compared by the forms' dtypes, shapes and placements and by the lanes
    of the maps the site's program names. Two calls share a derived program only
    when these are equal, and a derived program compiles once. *)

val site :
  name:string ->
  compile:(t -> form array -> value array -> value array) ->
  t ->
  form array ->
  site
(** [site ~name ~compile p forms] is the site of [p] over arguments of [forms],
    for the entry point [name]: its compiled form is [compile p forms], applied
    to arguments of [forms], and each program derived from [p] is compiled by
    [compile] too. *)

val axes : site -> axis list
(** [axes s] is the maps that [s]'s program or a program it carries names
    through [Lanes] or [Lane_index], in the order first named. *)

val customs : site -> bool
(** [customs s] is [true] iff [s]'s program or a program it carries holds a
    custom rule or a linear call, the constructs whose derivatives depend on
    whether a tape holds the tangents. *)

type derivation =
  | Jvp of { tangents : bool array; reverse : bool }
      (** Forward mode, [tangents.(k)] whether argument [k] has a tangent;
          [reverse] whether the tangents are a tape's, which only a program
          holding a custom rule or a linear call tells apart ({!customs}). Over
          the primals, then the tangents there are; to the results, the tangents
          there are, then the running values out. *)
  | Vmap of { lanes : bool array; n : int; named : bool }
      (** A map of [n] lanes, [lanes.(k)] whether argument [k] holds every
          lane's along a leading axis of [n]; [named] whether the program names
          the map. Over the arguments; to the results, each with that leading
          axis where it has lanes, then the running values out, every lane's. *)
  | Known of { unknown : bool array; donated : bool array }
      (** The part of the program that reads no argument where [unknown] holds,
          computed at the call. Over the other arguments; to its results, the
          residuals it computes, then the running values out. Its linear part
          is over the unknown arguments then the residuals, to the unknown
          results. A residual that is an argument of the known part is that
          argument, unless [donated] holds of one of the known part's
          arguments. The known part consumes a donated argument and then
          returns every residual. *)
  | Transpose of { linear : bool array; cts : bool array }
      (** The transpose of a linear program over the arguments where [linear]
          holds, given a cotangent of each result where [cts] holds: over the
          other arguments and the cotangents; to the cotangents of the linear
          arguments that receive one, then the running values out. *)

type derived = {
  program : t;
  at : site;  (** [program]'s site. *)
  flags : bool array;
      (** By derivation: which results have a tangent, which have lanes, which
          results are unknown, or which linear arguments receive a
          cotangent. *)
  linear : (t * site * int option array) option;
      (** A [Known] derivation's linear part, and for each residual the
          argument of the known part it is, which the known part does not
          return, or [None] for one it computes and returns. *)
}
(** The type for derived programs. *)

val derive :
  ?map:axis * int ->
  site ->
  derivation ->
  form array ->
  (unit ->
  (t * form array) * bool array * (t * form array * int option array) option) ->
  derived
(** [derive ~map s d forms make] is the program of [s] derived by [d] from
    arguments of [forms], made by [make ()] the first time [s] meets [d] and
    [forms]. [make ()] gives the program and its arguments' forms, its flags,
    and a linear part with its arguments' forms and its residuals' sources.
    [map] is a map that the derivation runs in, [(axis, n)], whose lanes no
    handler around answers. The site's key takes [n] as that map's lanes. *)

val derivations : unit -> int
(** [derivations ()] is the number of programs [derive] made on the calling
    domain. *)

(** {1:constructs Constructs}

    A construct's results are those of the innermost handler that answers
    {!Construct}, or its default's. Those of a loop, a switch, a call, a remat
    and a custom rule end with their programs' sums ({!sums}). A custom rule's
    then end with its tangent map's running values out, and a linear call's with
    its [apply]'s. {!run} delivers them to their totals. *)

type _ Effect.t +=
  | Construct : construct -> value array Effect.t
        (** The effect {!perform} performs: a handler that answers it continues
            with the construct's results, of the forms {!forms} gives. *)

exception Evaluate
(** Discontinuing {!Construct} with [Evaluate] runs the construct's default. No
    handler outside sees the construct. *)

val perform : construct -> value array
(** [perform c] performs {!Construct} [c] and returns its results. A default
    whose programs emit gathers each site's rows and, as it returns or raises,
    performs one {!Emit} per site, its rows in order concatenated along the
    axis after their lanes. *)

val run : construct -> value array
(** [run c] performs [c] and delivers its sums to their totals. A sum starts
    from its total's running value where a collect or tracer answers {!Running},
    and its end then replaces that value ({!Replace}), so that the total equals
    what the additions give made one by one. Otherwise the sum starts from the
    identity and its end is added to the total ({!add_to}), as are a custom
    rule's tangent map's running values out. Raises [Invalid_argument], before
    [c] runs, where a collect would refuse one of its additions ({!accepts}). *)

val forms : construct -> form array
(** [forms c] is the forms of [c]'s results, as its default gives them. Raises
    [Invalid_argument] as {!perform} would for [Lanes] outside its map. *)

(** {2:running Running values}

    The effects below let a construct start its sums from the running values
    around its call and hand back their ends. A construct asks for a total's
    running value ({!Running}) at the form it traced it at. Where a collect or
    tracer answers, the construct starts from that value and hands back the
    running value out ({!Replace}). Where none answers, the construct starts
    from the identity and adds the running value out to the total ({!add_to}),
    as {!threaded} also does where the value has another form. A map answers
    {!Running} with [None], so a construct inside it with no collect or tracer
    of the total between starts from the identity, and the map sums its lanes'
    ends. *)

type _ Effect.t +=
  | Running :
      ('v, 's, 'd) total * ('v, 's, 'd) Nx.Prim.form
      -> ('v, 's, 'd) Nx.t option Effect.t
        (** Answered by the innermost collect of the total around the call with
            its running value, where that has the form's shape. A tracer other
            than a tape between answers with a variable holding the total's
            running value where the call is, which the program it traces
            threads. A map between answers [None]. *)
  | Replace : ('v, 's, 'd) total * ('v, 's, 'd) Nx.t -> unit Effect.t
        (** Answered by the collect or tracer that answered {!Running}, whose
            running value becomes the value given. *)

val running :
  ('v, 's, 'd) total -> ('v, 's, 'd) Nx.Prim.form -> ('v, 's, 'd) Nx.t option
(** [running t f] is the answer to {!Running} of [t] at [f]'s shape, [None] with
    none around. *)

val threaded : ?lanes:int -> t -> (value array -> value array) -> value array
(** [threaded ~lanes p f] calls [f] with the running values in of [p]'s totals
    and returns its results before the running values out, handing those back.
    With [lanes], each has a leading axis of [lanes], one row per lane of a map.
*)

val threading :
  ?lanes:int -> addition array -> (value array -> value array * 'a) -> 'a
(** [threading ~lanes adds f] is [snd (f totals)], where [totals] are the
    running values in of the totals [adds] thread, as {!threaded} takes them. It
    hands [fst (f totals)] back as their running values out. *)

val add_to : ('v, 's, 'd) total -> ('v, 's, 'd) Nx.t -> unit
(** [add_to t x] adds [x] to [t]: the running value of the innermost collect or
    tracer that answers for [t] ({!Running}) becomes itself plus [x], and with
    none [Add] of [x] is performed. *)

(** Where a total's running value lies. *)
type 'd where =
  | Nowhere  (** Nothing around the call answers with a placement. *)
  | At of 'd Nx.Placement.t option
      (** At this placement; [None] where the holder, a jit's program, takes it
          where the first addition lies. *)

type _ Effect.t +=
  | Where : ('v, 's, 'd) total -> 'd where Effect.t
        (** Answered by the innermost collect of the total around the call with
            its running value's placement. A tracer between that threads the
            total answers with its own running value's placement where the
            holder around it answers [At]. Otherwise a tracer answers as the
            holder around it does, and the jit tracer [At None] where none is
            around. A map passes it on. *)

type _ Effect.t +=
  | Open : 'a probe -> bool Effect.t
        (** Answered [true] by a collect of the probe around the call, and
            [false] by an evaluation that offers nothing ({!silenced}); a tracer
            passes it on and records the answer. *)

val opened : 'a probe -> bool
(** [opened p] is the answer to {!Open}, [false] with none around. *)

type _ Effect.t +=
  | Within : bool Effect.t
        (** Answered [true] by every tracer, map and forward mode: an
            interpretation that may answer a construct outside a handler between
            it and the construct. *)

val within : unit -> bool
(** [within ()] is the answer to {!Within}, [false] with none around. *)

val emits_from : t -> bool array -> bool
(** [emits_from p from] is [true] iff an emission in [p] reads a value computed
    from an argument [k] with [from.(k)], a construct over one whose programs
    emit counting as one. *)

val observed : t -> bool
(** [observed p] is [true] iff [p] or a program it carries emits to a probe with
    a collect open ({!opened}). *)

val gathering : ?lanes:bool -> (unit -> 'a) -> 'a
(** [gathering ~lanes f] is [f ()], each site's emissions in it gathered and, as
    [f] returns or raises, performed as one {!Emit} per site in the order the
    sites first emitted, as {!perform} does. With [lanes], a map's: the first
    axis of each, the map's lanes, is merged with the next. *)

val silenced : (unit -> 'a) -> 'a
(** [silenced f] is [f ()] with every emission inside it dropped, every probe
    closed ({!Open}), and every construct that leaves it, answered by a handler
    around it, performed without the emissions of its programs: as a backward
    pass, a remat's replay, a root's derivative and a custom rule's primal
    derived again evaluate their programs, which offer nothing. *)

val where : ('v, 's, 'd) total -> 'd where
(** [where t] is the answer to {!Where}, [Nowhere] with none around. *)

type _ Effect.t +=
  | Accepts : ('v, 's, 'd) total * ('v, 's, 'd) Nx.Prim.form -> unit Effect.t
        (** Answered by the innermost collect of the total around the call,
            which raises [Invalid_argument] where it would refuse an addition of
            the form's shape; tracers and maps pass it on. *)

val accepts : ('v, 's, 'd) total -> ('v, 's, 'd) Nx.Prim.form -> unit
(** [accepts t f] performs {!Accepts}, nothing with none around. *)

val admits : ('v, 's, 'd) total -> ('v, 's, 'd) Nx.Prim.form -> unit
(** [admits t f] raises [Invalid_argument] where nothing answers {!Running} of
    [t] at [f] and {!accepts} of [t] at [f] raises, so where {!add_to} of a
    value of [f]'s shape would be refused. *)

val unset : value -> 'a
(** [unset v] raises [Invalid_argument] for a running value [v] computed on a
    device set its total's running value does not lie on. *)

(** {2:operands Operands and carried programs}

    The values below take a construct's operands and programs apart, and make
    programs and forms from them. *)

val operands : construct -> value list
(** [operands c] is the values [c] reads, captures included. *)

val carried : construct -> t list
(** [carried c] is the programs [c] carries. *)

type map = { f : 'v 's 'd. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t }
(** A function on values of every dtype and set. *)

val map_operands : map -> construct -> construct
(** [map_operands m c] is [c] with each operand [x] replaced by [m.f x]. *)

val map_programs : (t -> t) -> construct -> construct
(** [map_programs f c] is [c] with each program [p] it carries replaced by
    [f p]. A call gets a site of its new body, with no compiled form and no
    derived program yet. *)

val stacked : int -> form -> form
(** [stacked n f] is the form of a value of [f]'s with a leading axis of [n], in
    C order. *)

val switched : t array -> form array
(** [switched branches] is the forms of a switch's results over [branches],
    those of its first branch's results, which every branch shares. *)

val widen : t -> at:int array -> int -> t
(** [widen p ~at n] is [p] over [n] arguments, where [p]'s argument [k] is
    argument [at.(k)] and the others are unread, then its totals' running values
    in. *)

val unite : (t * value array) array -> params:int -> t array * value array
(** [unite traced ~params] is [(ps, captures)]. Each program of [traced] is over
    [params] parameters then its own captures. Its counterpart in [ps] is over
    those parameters then [captures], the captures of them all with each value
    once. *)

val fill :
  t ->
  id:int ->
  (transposed:bool -> lanes:(int * bool array) list -> t * value array) ->
  t
(** [fill p ~id body] is [p] with each application of the operator [id], in [p]
    and the programs its constructs carry, replaced by a call of [q] over the
    application's arguments, its data, then [captures], where
    [(q, captures) = body ~transposed ~lanes]. *)

val count : value -> int
(** [count v] is the value of [v], a 0-d [int32]. *)

(** {2:queries Maps and staging}

    A tracer {e stages} the operations applied in its extent when it records
    each one, those over constants included ({!Trace.kind}). *)

type _ Effect.t +=
  | Bound : axis -> int option Effect.t
        (** Answered by each map with its number of lanes, when the axis names
            it. *)

val bound : axis -> int option
(** [bound a] is [Some n] inside the map named [a] of [n] lanes, and [None] with
    none around. *)

val mentions : axis -> t -> bool
(** [mentions a p] is [true] iff [p] or a program its constructs carry reads the
    lanes or the lane index of the map named [a]. *)

val reads_lanes : ?index:bool -> t -> bool
(** [reads_lanes p] is [true] iff [p] or a program its constructs carry reads
    the lanes of a map, or its lane index unless [index] is [false]. *)

val reads_params : t -> bool array -> bool
(** [reads_params p mask] is [true] iff one of [p]'s arguments [k] with
    [mask.(k)] is read by what [p]'s results need. *)

type _ Effect.t +=
  | Staging : bool Effect.t
        (** Answered [true] by the jit and staged tracers ({!Trace.kind}), which
            record every operation applied while they trace. *)

val staging : unit -> bool
(** [staging ()] is [true] iff a tracer around the call stages the operations
    applied there. A call's default there evaluates its body and does not run
    its compiled form. *)

(** {2:rules Custom rules on values}

    A custom rule's call ([Custom.jvp_call]) that no tracer stages first asks
    the innermost interpretation around it how the rule runs. *)

type mode =
  | Forward  (** The rule's derivative is a tangent map. *)
  | Reverse  (** The rule's derivative is a pullback. *)

type rule = {
  name : string;  (** The entry point, in messages. *)
  mode : mode;
  args : value array;
  rule : value array -> value array * (value array -> value array);
      (** The user's rule: the results and the derivative, for arguments of
          [args]' forms. Each runs under [Nx.Rng.refusing], named [name]. *)
}

type _ Effect.t +=
  | Rule : rule -> (unit -> value array) option Effect.t
        (** Answered by each interpretation of rune2. A differentiation that
            tracks one of the arguments, whose primals no interpretation traces,
            answers with the function the call applies where it was performed:
            it runs the rule on those primals and returns the call's results.
            Every other interpretation of rune2 answers [None]. An
            interpretation outside rune2 ([Nx.Prim.interpret]) never answers
            the effect, so one between the call and the differentiation does
            not keep the rule from running in place; a rule that captures its
            values then raises, as a value used outside its owner's extent. *)

val rule : rule -> (unit -> value array) option
(** [rule r] is the answer to [Rule r], [None] with no interpretation around. *)

(** {1:printing Printing}

    A program prints as each constant it or a program it carries reads, as
    [cK : dtype shape at placement], numbered in the order they are first read;
    then [program (%0, …)] with its arguments and [totals (…)] with its running
    values in, each equation, [return] with its results and [totals] with its
    running values out. Each program's variables are numbered from where the
    previous one's end, so no two programs share a name. An operation prints as
    its results, each [%n : dtype shape at placement], [=], the operation as
    {!Nx.Prim.pp} formats it, and on the next line [with x0 = …] for each
    operand. A construct prints as its results, its operands, and the programs
    it carries under labels, indented (a root its residual, a solve its linear
    solve). Maps are numbered in the order the program first names them. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a program. *)
