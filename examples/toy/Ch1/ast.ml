type shape = int list

type expr =
  | Num of float
  | Literal of int list * expr list
  | Var of string
  | VarDecl of string * shape * expr
  | Return of expr option
  | BinOp of char * expr * expr
  | Call of string * expr list
  | Print of expr

type proto = Prototype of string * string list
type func = Function of proto * expr list
type modul = func list
