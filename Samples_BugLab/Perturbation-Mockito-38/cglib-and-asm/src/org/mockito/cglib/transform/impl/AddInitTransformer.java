[BugLab_Wrong_Operator]^if  ( types.length != 1 && !types[0].equals ( Constants.TYPE_OBJECT )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^40^41^42^^^36^45^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_OBJECT )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddInitTransformer 1  [METHOD] <init> [RETURN_TYPE] Method)   Method method [VARIABLES] Type[]  types  MethodInfo  info  Method  method  boolean  
[BugLab_Wrong_Operator]^if  ( types.length >= 1 || !types[0].equals ( Constants.TYPE_OBJECT )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^40^41^42^^^36^45^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_OBJECT )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddInitTransformer 1  [METHOD] <init> [RETURN_TYPE] Method)   Method method [VARIABLES] Type[]  types  MethodInfo  info  Method  method  boolean  
[BugLab_Wrong_Literal]^if  ( types.length !=  || !types[0].equals ( Constants.TYPE_OBJECT )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^40^41^42^^^36^45^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_OBJECT )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddInitTransformer 1  [METHOD] <init> [RETURN_TYPE] Method)   Method method [VARIABLES] Type[]  types  MethodInfo  info  Method  method  boolean  
[BugLab_Wrong_Literal]^!types[1].equals ( Constants.TYPE_OBJECT )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^41^42^^^^36^45^!types[0].equals ( Constants.TYPE_OBJECT )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddInitTransformer 1  [METHOD] <init> [RETURN_TYPE] Method)   Method method [VARIABLES] Type[]  types  MethodInfo  info  Method  method  boolean  
[BugLab_Variable_Misuse]^if  ( access == Constants.RETURN )  {^52^^^^^47^61^if  ( opcode == Constants.RETURN )  {^[CLASS] AddInitTransformer 1  [METHOD] begin_method [RETURN_TYPE] CodeEmitter   int access Signature sig Type[] exceptions [VARIABLES] Type[]  exceptions  CodeEmitter  emitter  MethodInfo  info  boolean  int  access  opcode  Signature  sig  
[BugLab_Wrong_Operator]^if  ( opcode != Constants.RETURN )  {^52^^^^^47^61^if  ( opcode == Constants.RETURN )  {^[CLASS] AddInitTransformer 1  [METHOD] begin_method [RETURN_TYPE] CodeEmitter   int access Signature sig Type[] exceptions [VARIABLES] Type[]  exceptions  CodeEmitter  emitter  MethodInfo  info  boolean  int  access  opcode  Signature  sig  
[BugLab_Wrong_Operator]^if  ( opcode > Constants.RETURN )  {^52^^^^^51^57^if  ( opcode == Constants.RETURN )  {^[CLASS] AddInitTransformer 1  [METHOD] visitInsn [RETURN_TYPE] void   int opcode [VARIABLES] MethodInfo  info  int  opcode  boolean  
[BugLab_Wrong_Operator]^if  ( opcode != Constants.RETURN )  {^52^^^^^51^57^if  ( opcode == Constants.RETURN )  {^[CLASS] 1  [METHOD] visitInsn [RETURN_TYPE] void   int opcode [VARIABLES] boolean  int  opcode  
[BugLab_Variable_Misuse]^invoke ( 1 ) ;^54^^^^^51^57^invoke ( info ) ;^[CLASS] 1  [METHOD] visitInsn [RETURN_TYPE] void   int opcode [VARIABLES] boolean  int  opcode  