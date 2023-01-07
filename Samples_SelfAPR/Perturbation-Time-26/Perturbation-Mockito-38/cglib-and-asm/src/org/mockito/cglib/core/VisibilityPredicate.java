[P8_Replace_Mix]^private boolean privateOk;^23^^^^^18^28^private boolean protectedOk;^[CLASS] VisibilityPredicate   [VARIABLES] 
[P1_Replace_Type]^private char pkg;^24^^^^^19^29^private String pkg;^[CLASS] VisibilityPredicate   [VARIABLES] 
[P8_Replace_Mix]^this.protectedOk =  null;^27^^^^^26^29^this.protectedOk = protectedOk;^[CLASS] VisibilityPredicate  [METHOD] <init> [RETURN_TYPE] Class,boolean)   Class source boolean protectedOk [VARIABLES] Class  source  boolean  protectedOk  String  pkg  
[P8_Replace_Mix]^pkg =  TypeUtils.getPackageName ( Type.getType ( null )  ) ;^28^^^^^26^29^pkg = TypeUtils.getPackageName ( Type.getType ( source )  ) ;^[CLASS] VisibilityPredicate  [METHOD] <init> [RETURN_TYPE] Class,boolean)   Class source boolean protectedOk [VARIABLES] Class  source  boolean  protectedOk  String  pkg  
[P14_Delete_Statement]^^28^^^^^26^29^pkg = TypeUtils.getPackageName ( Type.getType ( source )  ) ;^[CLASS] VisibilityPredicate  [METHOD] <init> [RETURN_TYPE] Class,boolean)   Class source boolean protectedOk [VARIABLES] Class  source  boolean  protectedOk  String  pkg  
[P1_Replace_Type]^short  mod =  ( arg instanceof Member )  ?  (  ( Member ) arg ) .getModifiers (  )  :  (  ( Integer ) arg ) .intValue (  ) ;^32^^^^^31^42^int mod =  ( arg instanceof Member )  ?  (  ( Member ) arg ) .getModifiers (  )  :  (  ( Integer ) arg ) .intValue (  ) ;^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P2_Replace_Operator]^int mod =  ( arg  <=  Member )  ?  (  ( Member ) arg ) .getModifiers (  )  :  (  ( Integer ) arg ) .intValue (  ) ;^32^^^^^31^42^int mod =  ( arg instanceof Member )  ?  (  ( Member ) arg ) .getModifiers (  )  :  (  ( Integer ) arg ) .intValue (  ) ;^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P6_Replace_Expression]^int mod  =   (  ( Integer ) arg ) .intValue (  ) ;^32^^^^^31^42^int mod =  ( arg instanceof Member )  ?  (  ( Member ) arg ) .getModifiers (  )  :  (  ( Integer ) arg ) .intValue (  ) ;^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P6_Replace_Expression]^int mod  =   (  ( Member ) arg ) .getModifiers (  )  ;^32^^^^^31^42^int mod =  ( arg instanceof Member )  ?  (  ( Member ) arg ) .getModifiers (  )  :  (  ( Integer ) arg ) .intValue (  ) ;^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P8_Replace_Mix]^int mod =  ( arg instanceof Member )  ?  (  ( Member ) arg )  .getDeclaringClass (  )   :  (  ( Integer ) arg ) .intValue (  ) ;^32^^^^^31^42^int mod =  ( arg instanceof Member )  ?  (  ( Member ) arg ) .getModifiers (  )  :  (  ( Integer ) arg ) .intValue (  ) ;^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P7_Replace_Invocation]^if  ( Modifier.isProtected ( mod )  )  {^33^^^^^31^42^if  ( Modifier.isPrivate ( mod )  )  {^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P13_Insert_Block]^if  ( isPublic ( mod )  )  {     return true; }else     if  ( isProtected ( mod )  )  {         return protectedOk;     }else {         return pkg.equals ( getPackageName ( getType (  (  ( Member )   ( arg )  ) .getDeclaringClass (  )  )  )  ) ;     }^33^^^^^31^42^[Delete]^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P13_Insert_Block]^if  ( isProtected ( mod )  )  {     return protectedOk; }else {     return pkg.equals ( getPackageName ( getType (  (  ( Member )   ( arg )  ) .getDeclaringClass (  )  )  )  ) ; }^33^^^^^31^42^[Delete]^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P6_Replace_Expression]^} else {^35^^^^^31^42^} else if  ( Modifier.isPublic ( mod )  )  {^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P7_Replace_Invocation]^} else if  ( Modifier.isPrivate ( mod )  )  {^35^^^^^31^42^} else if  ( Modifier.isPublic ( mod )  )  {^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P8_Replace_Mix]^}  if  ( Modifier.isPrivate ( mod )  )  {^35^^^^^31^42^} else if  ( Modifier.isPublic ( mod )  )  {^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P9_Replace_Statement]^} else if  ( Modifier.isProtected ( mod )  )  {^35^^^^^31^42^} else if  ( Modifier.isPublic ( mod )  )  {^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P6_Replace_Expression]^} else {^37^^^^^31^42^} else if  ( Modifier.isProtected ( mod )  )  {^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P7_Replace_Invocation]^} else if  ( Modifier.isPrivate ( mod )  )  {^37^^^^^31^42^} else if  ( Modifier.isProtected ( mod )  )  {^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P15_Unwrap_Block]^return protectedOk;^37^38^39^40^41^31^42^} else if  ( Modifier.isProtected ( mod )  )  { return protectedOk; } else { return pkg.equals ( TypeUtils.getPackageName ( Type.getType (  (  ( Member ) arg ) .getDeclaringClass (  )  )  )  ) ; }^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P16_Remove_Block]^^37^38^39^40^41^31^42^} else if  ( Modifier.isProtected ( mod )  )  { return protectedOk; } else { return pkg.equals ( TypeUtils.getPackageName ( Type.getType (  (  ( Member ) arg ) .getDeclaringClass (  )  )  )  ) ; }^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P7_Replace_Invocation]^return pkg.equals ( TypeUtils.getPackageName ( Type.getType (  (  ( Member ) arg )  .getModifiers (  )   )  )  ) ;^40^^^^^31^42^return pkg.equals ( TypeUtils.getPackageName ( Type.getType (  (  ( Member ) arg ) .getDeclaringClass (  )  )  )  ) ;^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P13_Insert_Block]^if  ( isProtected ( mod )  )  {     return protectedOk; }else {     return pkg.equals ( getPackageName ( getType (  (  ( Member )   ( arg )  ) .getDeclaringClass (  )  )  )  ) ; }^40^^^^^31^42^[Delete]^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P14_Delete_Statement]^^40^^^^^31^42^return pkg.equals ( TypeUtils.getPackageName ( Type.getType (  (  ( Member ) arg ) .getDeclaringClass (  )  )  )  ) ;^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P3_Replace_Literal]^return false;^36^^^^^31^42^return true;^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P8_Replace_Mix]^}  if  ( Modifier.isPrivate ( mod )  )  {^37^^^^^31^42^} else if  ( Modifier.isProtected ( mod )  )  {^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  
[P3_Replace_Literal]^return true;^34^^^^^31^42^return false;^[CLASS] VisibilityPredicate  [METHOD] evaluate [RETURN_TYPE] boolean   Object arg [VARIABLES] Object  arg  boolean  protectedOk  String  pkg  int  mod  