type shape = int array [@@deriving show]
type var = string [@@deriving show]

type expr =
  | Num of float
  | Literal of shape * expr list
  | Var of var
  | VarDecl of string * shape * expr
  | Return of expr option
  | BinOp of char * expr * expr
  | Call of string * expr list
  | Print of expr
[@@deriving show]

type proto = Prototype of string * var list [@@deriving show]
type func = Function of proto * expr list [@@deriving show]
type modul = func list [@@deriving show]
