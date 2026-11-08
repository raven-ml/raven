open Talon

let z = Tz.utc

let e =
  Expr.Temporal.localize z ~ambiguous:`Fail ~gap:`Fail (Col.instant "at")
