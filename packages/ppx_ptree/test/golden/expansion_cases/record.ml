type 'a t = { w : 'a; b : 'a option; layers : 'a Linear.t list }
[@@deriving ptree]
