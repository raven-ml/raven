(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Binary = Binary
module Decimal = Decimal
module Time = Time
module Kind = Kind
module Record = Record
module Type = Type
module Schema = Schema
module Error = Error
module Tz = Tz
module Sel = Sel
module Order = Order
module Window = Window
module Expr = Expr

module Col = struct
  let v k name = Expr.make (Expr.Handle (k, name))
  let bool name = v Kind.bool name
  let int name = v Kind.int name
  let float name = v Kind.float name
  let string name = v Kind.string name
  let binary name = v Kind.binary name
  let decimal name = v Kind.decimal name
  let date name = v Kind.date name
  let instant name = v Kind.instant name
  let span name = v Kind.span name
end

module Ext = struct
  type ('e, 's) t = ('e, 's) Expr.ext

  let v ~name ?metadata ~ordered storage ~dec ~enc : _ t =
    { type_ = Type.ext ~name ?metadata storage; storage; ordered; dec; enc }

  let col e name = Expr.make (Expr.Ext_handle (e, name))
  let storage e x = Expr.make (Expr.Storage (e, x))
  let wrap e x = Expr.make (Expr.Wrap (e, x))
end
