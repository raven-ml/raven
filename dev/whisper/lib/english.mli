(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** English text normalised for scoring transcripts: Whisper's English text
    normaliser (Radford et al.,
    {e Robust Speech Recognition via Large-Scale Weak Supervision}, 2022,
    Appendix C).

    A word error rate ({!Wer}) compares a transcript with a reference word by
    word. Differences of form, such as case, punctuation, contractions, numbers
    spelled out and British spellings, are not errors of recognition, and
    {!normalize} removes them from both before they are compared. It applies, in
    order:
    + Lowercasing (Unicode's full mapping, with a final capital sigma as [ς]).
    + Removing text between [<] or [\[] and the next [>] or [\]], and between
      [(] and the next [)] when it holds a character.
    + Removing the words [hmm], [mm], [mhm], [mmm], [uh] and [um].
    + Removing white space before an apostrophe.
    + Expanding contractions: [won't], [can't], [let's], [ain't] ([aint]),
      [y'all], [wanna], [gotta], [gonna], [i'ma], [imma], [woulda], [coulda],
      [shoulda] and [ma'am] ([madam]); the titles [mr], [mrs], [st], [dr],
      [prof], [capt], [gov], [ald], [gen], [sen], [rep], [pres], [rev], [hon],
      [asst], [assoc], [lt], [col], [jr], [sr] and [esq]; [had] and [has] before
      [been], [gone], [done] and [got]; then [n't], ['re], ['s], ['d], ['ll],
      ['t], ['ve] and ['m].
    + Removing a comma between digits, and a period not followed by a digit [0]
      to [9].
    + Decomposing compatibility characters (NFKD), dropping nonspacing marks,
      writing [œ], [ø], [æ], [ß], [đ], [ð], [þ] and [ł] and their capitals in
      ASCII, and replacing every other mark, symbol and punctuation, but
      [. % $ ¢ € £], by a space.
    + Writing numbers in digits: ["twenty one"] is [21], ["one oh one"] [101],
      ["two point five"] [2.5], ["three and a half"] [3.5], ["five percent"]
      [5%], ["twenty dollars and seven cents"] [$20.07], ordinals and plurals
      keep their suffix (["twentieth"] is [20th], ["the sixties"] [the 60s]),
      and [1] and [1s] standing alone are written [one] and [ones].
    + Writing British spellings in American ones, word by word.
    + Removing [. $ ¢ € £] not followed by a digit and [%] not preceded by one.
    + Replacing each run of white space by one space.

    Words are maximal runs of letters, numbers and [_]; a digit is a decimal
    digit of any script. The result is not stripped: text that starts with a
    space keeps it. *)

val normalize : string -> string
(** [normalize s] is the UTF-8 text [s] normalised as above. Bytes that are not
    UTF-8 read as U+FFFD. *)
