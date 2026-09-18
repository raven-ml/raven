(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Regular expressions of tokenizer files.

    Tokenizer files carry patterns written for a Unicode-aware engine (Ruby
    syntax, Oniguruma). This translates them to {!Re}, which matches bytes, so
    that they match the same text: a class stands for its code points as UTF-8
    sequences, and every match starts and ends on a character boundary.

    Supported: literals, [.] (any character but newline), the classes [\s],
    [\d], [\w] and their negations, [\p{..}] and [\P{..}] over general
    categories ([L], [Lu], [Nd], [Letter], [Uppercase_Letter], ...) and [Any],
    bracket classes with ranges and negation, groups [(..)], [(?:..)] and
    [(?<name>..)], comments [(?#..)], alternation, greedy and lazy quantifiers
    ([*], [+], [?], [{n}], [{n,}], [{,m}], [{n,m}]), the anchors [^], [$], [\A],
    [\z], [\Z] and [\G], the escapes [\t \n \r \f \v \e \a \0 \xHH \x{H..}], the
    case-insensitive option [(?i)], [(?-i)], [(?i:..)], [(?-i:..)], and a
    lookahead [(?=..)] or [(?!..)] that ends the pattern.

    Under [(?i)] a character stands for every character with the same simple
    case folding, so ['s] matches ['S] and the long s of U+017F. A folding to
    several characters is not followed: [(?i:ss)] does not match U+00DF.

    A lookahead is matched as a trailing context: the text it inspects is
    consumed by the automaton and then given back. That is only sound where
    nothing of the match follows it, so it must be the last piece of the
    pattern, of one of its alternatives, or of an unquantified group in that
    position; a lookahead can end each alternative. The body of [(?=..)] is any
    supported pattern without a lookahead of its own, and the body of [(?!..)]
    is one character: a literal, an escape, [.] or a bracket class. At the end
    of the searched range [(?!..)] holds.

    Rejected with an error: the options other than [i], lookbehind, a lookahead
    anywhere else, atomic groups, backreferences, possessive quantifiers, word
    boundaries, POSIX brackets, class intersection and nesting, properties other
    than general categories, byte escapes above [\x7F], and a [^] that can end a
    match, whose meaning at the end of a text closing with a newline cannot be
    reproduced.

    One difference remains: a quantified group whose earlier alternative can
    match empty, as [(x?|a)+] on ["a"], matches empty here where the original
    engine goes on to match ["a"]. It comes from how alternatives are tried
    under repetition and cannot be corrected by translation.

    Text that is not valid UTF-8 is searched like any other, and a byte that
    belongs to no character matches no class, negated ones included. *)

type t
(** The type for compiled patterns. *)

val compile : ?anchors:bool -> string -> (t, string) result
(** [compile pattern] is [pattern] compiled, or an error saying which construct
    is not supported or where the syntax is invalid.

    [anchors] is whether the anchors are accepted; defaults to [true]. They look
    at the whole string even when {!find} is given a range of it, so a caller
    that searches ranges standing for whole texts compiles without them. *)

val find : t -> string -> pos:int -> stop:int -> (int * int) option
(** [find t s ~pos ~stop] is the start and the end of the leftmost match of [t]
    in the bytes of [s] between [pos] and [stop], if any. The end is where a
    trailing lookahead starts. A match can be empty. *)
