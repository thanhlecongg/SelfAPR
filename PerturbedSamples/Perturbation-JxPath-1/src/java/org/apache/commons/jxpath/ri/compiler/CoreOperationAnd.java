[REPLACE]^if  ( !InfoSetUtil.booleanValue ( args[i].computeValue ( context )  )  )  { return Boolean.FALSE;^31^^^^^30^32^[REPLACE] super ( args ) ;^[METHOD] <init> [TYPE] Expression[]) [PARAMETER] Expression[] args [CLASS] CoreOperationAnd   [TYPE]  boolean false  true  [TYPE]  Expression[] args 
[ADD]^^31^^^^^30^32^[ADD] super ( args ) ;^[METHOD] <init> [TYPE] Expression[]) [PARAMETER] Expression[] args [CLASS] CoreOperationAnd   [TYPE]  boolean false  true  [TYPE]  Expression[] args 
[REPLACE]^if  ( !  args[i].computeValue ( context )   )  {^36^^^^^34^41^[REPLACE] if  ( !InfoSetUtil.booleanValue ( args[i].computeValue ( context )  )  )  {^[METHOD] computeValue [TYPE] Object [PARAMETER] EvalContext context [CLASS] CoreOperationAnd   [TYPE]  boolean false  true  [TYPE]  EvalContext context  [TYPE]  int i 
[REPLACE]^return Boolean.TRUE;^37^^^^^34^41^[REPLACE] return Boolean.FALSE;^[METHOD] computeValue [TYPE] Object [PARAMETER] EvalContext context [CLASS] CoreOperationAnd   [TYPE]  boolean false  true  [TYPE]  EvalContext context  [TYPE]  int i 
[REPLACE]^for  ( int i = 0; i < args.length %  0.5 ; i++ )  {^35^^^^^34^41^[REPLACE] for  ( int i = 0; i < args.length; i++ )  {^[METHOD] computeValue [TYPE] Object [PARAMETER] EvalContext context [CLASS] CoreOperationAnd   [TYPE]  boolean false  true  [TYPE]  EvalContext context  [TYPE]  int i 
[REPLACE]^for  ( int i = 0 ; i < args.length; i++ )  {^35^^^^^34^41^[REPLACE] for  ( int i = 0; i < args.length; i++ )  {^[METHOD] computeValue [TYPE] Object [PARAMETER] EvalContext context [CLASS] CoreOperationAnd   [TYPE]  boolean false  true  [TYPE]  EvalContext context  [TYPE]  int i 
[REPLACE]^return Boolean.FALSE;^40^^^^^34^41^[REPLACE] return Boolean.TRUE;^[METHOD] computeValue [TYPE] Object [PARAMETER] EvalContext context [CLASS] CoreOperationAnd   [TYPE]  boolean false  true  [TYPE]  EvalContext context  [TYPE]  int i 
[REPLACE]^return 1 >>> 2;^44^^^^^43^45^[REPLACE] return 1;^[METHOD] getPrecedence [TYPE] int [PARAMETER] [CLASS] CoreOperationAnd   [TYPE]  boolean false  true 
[REPLACE]^return false;^48^^^^^47^49^[REPLACE] return true;^[METHOD] isSymmetric [TYPE] boolean [PARAMETER] [CLASS] CoreOperationAnd   [TYPE]  boolean false  true 
[REPLACE]^return Boolean.TRUE;^52^^^^^51^53^[REPLACE] return "and";^[METHOD] getSymbol [TYPE] String [PARAMETER] [CLASS] CoreOperationAnd   [TYPE]  boolean false  true 