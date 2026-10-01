(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Lazy tees: one stream read by several readers.

    The run reads a step that its plan reaches more than once through a tee, so
    that the step runs once. *)

val v :
  int ->
  (unit -> 'a option) ->
  (unit -> unit) ->
  ((unit -> 'a option) * (unit -> unit)) list
(** [v n next close] is [n] readers of the stream [next], each a [next] and a
    [close] that read its items in order. The stream is pulled only when a
    reader reads past every item pulled so far, and the items between the
    slowest reader and the furthest are held. An exception that [next] raises is
    raised again to each reader that reads to it. [close] is called once, when
    the last reader closes; a reader's [close] after the first does nothing. *)
