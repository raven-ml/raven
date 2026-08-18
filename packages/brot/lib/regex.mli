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
    [\z], [\Z] and [\G], and the escapes [\t \n \r \f \v \e \a \0 \xHH \x{H..}].

    Rejected with an error: options ([(?i)] and the like), lookaround, atomic
    groups, backreferences, possessive quantifiers, word boundaries, POSIX
    brackets, class intersection and nesting, properties other than general
    categories, byte escapes above [\x7F], and a [^] that can end a match, whose
    meaning at the end of a text closing with a newline cannot be reproduced.

    One difference remains: a quantified group whose earlier alternative can
    match empty, as [(x?|a)+] on ["a"], matches empty here where the original
    engine goes on to match ["a"]. It comes from how alternatives are tried
    under repetition and cannot be corrected by translation. *)

val compile : string -> (Re.re, string) result
(** [compile pattern] is [pattern] compiled, or an error saying which construct
    is not supported or where the syntax is invalid. *)
