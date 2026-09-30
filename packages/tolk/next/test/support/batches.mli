(** Batches run on a model of their devices.

    A batch ({!Tolk_next.Hcq2.sched_batches}) hands each queue a list of
    commands: calls, and the instructions [wait], [store], [timestamp] and
    [barrier] on 64-bit words. The model runs them as the devices would, with
    each word a number: a queue runs its commands in order, a [wait] holds it
    until its word reaches its value, and the queues are taken in turn. Every
    device starts having submitted {!submitted} and signaled what [finished]
    says, so the batch's work signals [submitted + 1]. *)

open Tolk_next

val queues : Ops.t -> ((string * string) * Ops.t list) list
(** [queues batch] is the commands of each queue of [batch], by its first
    device and its name, in the order the batch submits them. *)

val calls : Ops.t -> Ops.t list
(** [calls batch] is the calls of [batch], queue by queue. *)

val submitted : int
(** [submitted] is [5], the value each device's work before the batch
    signals. *)

(** The type for what the model saw happen. *)
type event =
  | Call of int  (** The call of that position in {!calls} ran. *)
  | Signaled of string  (** The device's signal word took its value. *)

val pp_event : Format.formatter -> event -> unit
(** [pp_event] formats an event as [call 2] or [signaled CPU:1]. *)

type outcome = {
  events : event list;  (** What happened, in order. *)
  left : ((string * string) * int) list;
      (** Each queue with the number of its commands that never ran. *)
  signal_words : (string * int) list;
      (** Each device of the batch with the value of its signal word. *)
}
(** The type for runs of the model. *)

val run : ?finished:(string * int) list -> ?order:(string * string) list -> Ops.t -> outcome
(** [run ~finished ~order batch] runs [batch] until no queue can go on. A device
    has signaled the value [finished] gives it, and {!submitted} otherwise.
    Each step runs one command: the first that can run of the queues taken in
    [order] (default the batch's order).

    Raises [Invalid_argument] if a command is none of these. *)

val rotations : Ops.t -> (string * string) list list
(** [rotations batch] is each rotation of [batch]'s queue order. *)
