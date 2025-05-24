include Buffer
include Descriptor
include Views
include Astype
module Parallel = Parallel

module type Backend_intf = Backend_intf.S

module Make (B : Backend_intf.S) = struct
  include B

  type ('a, 'b) t = ('a, 'b) B.b_t

  include Backend_creation.Make (B)
  include Backend_access.Make (B)
  include Backend_bitwise.Make (B)
  include Backend_hof.Make (B)
  include Backend_ops.Make (B)
  include Backend_logic.Make (B)
  include Backend_linalg.Make (B)
  include Backend_transform.Make (B)
  include Backend_interop.Make (B)
end
