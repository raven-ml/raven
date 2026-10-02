(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Time zones, decoded from TZif files.

    A {e zone} gives, at each instant, the UTC offset of a region's wall clock.
    Instants are {e POSIX seconds}: [int64] seconds since 1970-01-01 00:00:00
    UTC, with no leap seconds. A {e database} holds zones by name.

    A zone answers two questions: its offset at an instant ({!offset_s}) and the
    offsets at which its clock reads a local time ({!val-local}).

    Take a database with {!system} or {!of_dir}, read its zones with {!find},
    and pass them where they are needed: there is no global database. A database
    names a directory, and zones are immutable values, so any domain may use
    either at any time.

    Zones come from TZif files of versions 1 to 4 (RFC 9636), the format of a
    system's zoneinfo directory. A file lists a zone's transitions up to some
    instant, and from version 2 on, its footer holds the rule that gives every
    offset after them. A file whose transition times count leap seconds, such as
    those under [right/], is converted to POSIX seconds, so every zone answers
    in POSIX seconds. *)

(** {1:zones Zones} *)

type zone
(** The type for time zones. *)

val utc : zone
(** [utc] is Coordinated Universal Time: the zone named ["UTC"], whose offset is
    [0] at every instant. *)

val name : zone -> string
(** [name z] is the name of [z]: the name its database holds it under ({!find}),
    such as ["Europe/Paris"], and ["UTC"] for {!utc}. *)

val offset_s : zone -> int64 -> int
(** [offset_s z t] is the UTC offset of [z] at the POSIX second [t], in seconds.
    At [t], the wall clock of [z] reads [t + offset_s z t], counted in seconds
    since 1970-01-01 00:00:00 of that clock.

    It is defined for every [t]. With the transitions of [z]'s file in time
    order:
    - before the first transition, the offset is that of the file's first time
      type;
    - from a transition up to, but excluding, the next, the offset is that of
      the transition's time type;
    - from the last transition on, the last transition's offset holds until the
      first instant after it at which the offset of the footer's rule changes,
      and the rule's from then on. A rule whose offset never changes, such as
      daylight saving time all year, leaves the last transition's offset in
      place, and so does a missing rule, in a version 1 file or under an empty
      footer.

    A file without transitions follows its rule at every [t], and has its first
    time type's offset when it has no rule.

    It costs O(log n) for a zone of n transitions. *)

(** The type for the answers of {!val-local}: the UTC offsets at which a zone's
    wall clock reads a local time [t]. An offset [o] stands for the instant
    [t - o]. Across a single transition, [before] and [after] are the offsets in
    effect before and after it. *)
type local =
  | Unique of int  (** [Unique o]: the clock reads [t] once, at [t - o]. *)
  | Ambiguous of { before : int; after : int }
      (** [Ambiguous { before; after }]: the clock was set back across [t] and
          reads [t] more than once, first at [t - before] and last at
          [t - after]. Here [after < before]. *)
  | Gap of { before : int; after : int }
      (** [Gap { before; after }]: the clock was set forward across [t] and
          never reads [t]. It reads less than [t] at every instant up to
          [t - after], and more than [t] at every instant from [t - before] on.
          Here [before < after]. When a single transition falls between those
          two instants, the clock reads [t] less the gap's length at [t - after]
          and [t] plus it at [t - before]. *)

val local : zone -> int64 -> local
(** [local z t] is the offsets [o] at which the wall clock of [z] reads [t],
    that is, those with [offset_s z (t - o) = o]. [t] counts seconds since
    1970-01-01 00:00:00 on that clock. Beyond [int64]'s range, [z]'s offset is
    the one at the nearest end.

    It is defined for every [t]. It costs O(k log n) for a zone of n
    transitions, where k counts the zone's transitions within its span of
    offsets around [t]: one or two for real zones. *)

(** {1:databases Databases} *)

type db
(** The type for zone databases: a directory of TZif files, read as zones are
    found. *)

val of_dir : string -> (db, Error.t) result
(** [of_dir dir] is the database of the TZif files under [dir], as a system's
    zoneinfo directory holds them. It only checks that [dir] is a directory,
    following it if it is a symbolic link: {!find} reads the zones.

    The result is [Error e] if [dir] is not a directory or cannot be examined.
*)

val system : unit -> (db, Error.t) result
(** [system ()] is [of_dir d] for the system's zoneinfo directory [d]: the value
    of the environment variable [TZDIR] when it is set and not empty, and
    ["/usr/share/zoneinfo"] otherwise. It reads the environment on each call.

    A [TZDIR] that names no directory is an error, with no fallback to
    ["/usr/share/zoneinfo"]. Windows has no zoneinfo directory: there, set
    [TZDIR], or [system] reads ["/usr/share/zoneinfo"] on the current drive. *)

val find : db -> string -> (zone, Error.t) result
(** [find db name] is the zone [name] of [db], read and decoded from the file
    [name] under [db]'s directory, and named [name] ({!name}).
    - {b Names.} [name] is made of segments separated by ['/'] on every system,
      as in ["America/Argentina/Buenos_Aires"]. A segment is not empty, is
      neither ["."] nor [".."], and holds only ASCII letters, digits, ['.'],
      ['-'], ['_'] and ['+'], the characters of tz's names. Its part before its
      first ['.'] does not name a Windows device, in upper or lower case:
      ["CON"], ["PRN"], ["AUX"], ["NUL"], or ["COM"] or ["LPT"] followed by one
      digit. So on every system, a name designates a path under the directory.
      Whether names are case-sensitive is up to the file system.
    - {b Files.} [find] reads the file that [name] designates, following
      symbolic links. A name that designates a FIFO or a device reads it, and
      may block.

    The result is [Error e] if [name] is not a name as above, if it designates a
    directory, if its file cannot be read, or if the file breaks RFC 9636,
    except that a footer that disagrees with the last transition is read as
    {!offset_s} says. A footer that names daylight saving time without its rule,
    and a transition before a leap-second table truncated at the start, whose
    POSIX time RFC 9636 leaves unspecified, are errors too. A file of a version
    above 4 is read as version 4, as RFC 9636 intends. [e] locates the failure:
    the path, and for a malformed TZif file, the byte range where decoding
    failed.

    Each call reads the file again, a few kilobytes, so find a zone once and
    keep it. *)
