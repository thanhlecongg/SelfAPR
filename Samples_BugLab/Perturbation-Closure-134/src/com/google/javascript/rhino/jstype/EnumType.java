[buglab_swap_variables]^super ( name, "enum{" + registry + "}", null ) ;^71^^^^^70^73^super ( registry, "enum{" + name + "}", null ) ;^[CLASS] EnumType  [METHOD] <init> [RETURN_TYPE] JSType)   JSTypeRegistry registry String name JSType elementsType [VARIABLES] JSTypeRegistry  registry  Set  elements  JSType  elementsType  boolean  String  name  long  serialVersionUID  EnumElementType  elementsType  
[buglab_swap_variables]^super (  "enum{" + name + "}", null ) ;^71^^^^^70^73^super ( registry, "enum{" + name + "}", null ) ;^[CLASS] EnumType  [METHOD] <init> [RETURN_TYPE] JSType)   JSTypeRegistry registry String name JSType elementsType [VARIABLES] JSTypeRegistry  registry  Set  elements  JSType  elementsType  boolean  String  name  long  serialVersionUID  EnumElementType  elementsType  
[buglab_swap_variables]^this.elementsType = new EnumElementType ( name, elementsType, registry ) ;^72^^^^^70^73^this.elementsType = new EnumElementType ( registry, elementsType, name ) ;^[CLASS] EnumType  [METHOD] <init> [RETURN_TYPE] JSType)   JSTypeRegistry registry String name JSType elementsType [VARIABLES] JSTypeRegistry  registry  Set  elements  JSType  elementsType  boolean  String  name  long  serialVersionUID  EnumElementType  elementsType  
[buglab_swap_variables]^this.elementsType = new EnumElementType (  elementsType, name ) ;^72^^^^^70^73^this.elementsType = new EnumElementType ( registry, elementsType, name ) ;^[CLASS] EnumType  [METHOD] <init> [RETURN_TYPE] JSType)   JSTypeRegistry registry String name JSType elementsType [VARIABLES] JSTypeRegistry  registry  Set  elements  JSType  elementsType  boolean  String  name  long  serialVersionUID  EnumElementType  elementsType  
[buglab_swap_variables]^this.elementsType = new EnumElementType ( elementsType, registry, name ) ;^72^^^^^70^73^this.elementsType = new EnumElementType ( registry, elementsType, name ) ;^[CLASS] EnumType  [METHOD] <init> [RETURN_TYPE] JSType)   JSTypeRegistry registry String name JSType elementsType [VARIABLES] JSTypeRegistry  registry  Set  elements  JSType  elementsType  boolean  String  name  long  serialVersionUID  EnumElementType  elementsType  
[buglab_swap_variables]^this.elementsType = new EnumElementType ( registry,  name ) ;^72^^^^^70^73^this.elementsType = new EnumElementType ( registry, elementsType, name ) ;^[CLASS] EnumType  [METHOD] <init> [RETURN_TYPE] JSType)   JSTypeRegistry registry String name JSType elementsType [VARIABLES] JSTypeRegistry  registry  Set  elements  JSType  elementsType  boolean  String  name  long  serialVersionUID  EnumElementType  elementsType  
[buglab_swap_variables]^this.elementsType = new EnumElementType ( registry, name, elementsType ) ;^72^^^^^70^73^this.elementsType = new EnumElementType ( registry, elementsType, name ) ;^[CLASS] EnumType  [METHOD] <init> [RETURN_TYPE] JSType)   JSTypeRegistry registry String name JSType elementsType [VARIABLES] JSTypeRegistry  registry  Set  elements  JSType  elementsType  boolean  String  name  long  serialVersionUID  EnumElementType  elementsType  
[buglab_swap_variables]^this.elementsType = new EnumElementType ( registry, elementsType ) ;^72^^^^^70^73^this.elementsType = new EnumElementType ( registry, elementsType, name ) ;^[CLASS] EnumType  [METHOD] <init> [RETURN_TYPE] JSType)   JSTypeRegistry registry String name JSType elementsType [VARIABLES] JSTypeRegistry  registry  Set  elements  JSType  elementsType  boolean  String  name  long  serialVersionUID  EnumElementType  elementsType  
[buglab_swap_variables]^return defineDeclaredProperty ( elementsType, name, false ) ;^96^^^^^94^97^return defineDeclaredProperty ( name, elementsType, false ) ;^[CLASS] EnumType  [METHOD] defineElement [RETURN_TYPE] boolean   String name [VARIABLES] Set  elements  String  name  boolean  long  serialVersionUID  EnumElementType  elementsType  
[buglab_swap_variables]^return defineDeclaredProperty (  elementsType, false ) ;^96^^^^^94^97^return defineDeclaredProperty ( name, elementsType, false ) ;^[CLASS] EnumType  [METHOD] defineElement [RETURN_TYPE] boolean   String name [VARIABLES] Set  elements  String  name  boolean  long  serialVersionUID  EnumElementType  elementsType  
[buglab_swap_variables]^return defineDeclaredProperty ( name,  false ) ;^96^^^^^94^97^return defineDeclaredProperty ( name, elementsType, false ) ;^[CLASS] EnumType  [METHOD] defineElement [RETURN_TYPE] boolean   String name [VARIABLES] Set  elements  String  name  boolean  long  serialVersionUID  EnumElementType  elementsType  
[buglab_swap_variables]^return this.equals ( FALSE )  ? TRUE : that;^112^^^^^107^113^return this.equals ( that )  ? TRUE : FALSE;^[CLASS] EnumType  [METHOD] testForEquality [RETURN_TYPE] TernaryValue   JSType that [VARIABLES] Set  elements  JSType  that  TernaryValue  result  boolean  long  serialVersionUID  EnumElementType  elementsType  
[buglab_swap_variables]^return this.equals ( that )  ? FALSE : TRUE;^112^^^^^107^113^return this.equals ( that )  ? TRUE : FALSE;^[CLASS] EnumType  [METHOD] testForEquality [RETURN_TYPE] TernaryValue   JSType that [VARIABLES] Set  elements  JSType  that  TernaryValue  result  boolean  long  serialVersionUID  EnumElementType  elementsType  