[BugLab_Variable_Misuse]^return name;^81^^^^^80^82^return this.name;^[CLASS] TableOrder  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] TableOrder  BY_COLUMN  BY_ROW  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( this != obj )  {^93^^^^^92^104^if  ( this == obj )  {^[CLASS] TableOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  TableOrder  BY_COLUMN  BY_ROW  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^94^^^^^92^104^return true;^[CLASS] TableOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  TableOrder  BY_COLUMN  BY_ROW  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( ! ( obj  !=  TableOrder )  )  {^96^^^^^92^104^if  ( ! ( obj instanceof TableOrder )  )  {^[CLASS] TableOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  TableOrder  BY_COLUMN  BY_ROW  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^97^^^^^92^104^return false;^[CLASS] TableOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  TableOrder  BY_COLUMN  BY_ROW  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !this.name.equals ( BY_ROW.name )  )  {^100^^^^^92^104^if  ( !this.name.equals ( that.name )  )  {^[CLASS] TableOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  TableOrder  BY_COLUMN  BY_ROW  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( !this.name.equals ( name )  )  {^100^^^^^92^104^if  ( !this.name.equals ( that.name )  )  {^[CLASS] TableOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  TableOrder  BY_COLUMN  BY_ROW  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^101^^^^^92^104^return false;^[CLASS] TableOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  TableOrder  BY_COLUMN  BY_ROW  that  String  name  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^if  ( !this.name.equals ( that.name.name )  )  {^100^^^^^92^104^if  ( !this.name.equals ( that.name )  )  {^[CLASS] TableOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  TableOrder  BY_COLUMN  BY_ROW  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^103^^^^^92^104^return true;^[CLASS] TableOrder  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  TableOrder  BY_COLUMN  BY_ROW  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return name.hashCode (  ) ;^112^^^^^111^113^return this.name.hashCode (  ) ;^[CLASS] TableOrder  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] TableOrder  BY_COLUMN  BY_ROW  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( this.equals ( TableOrder.that )  )  {^123^^^^^122^130^if  ( this.equals ( TableOrder.BY_ROW )  )  {^[CLASS] TableOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] TableOrder  BY_COLUMN  BY_ROW  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^else if  ( this.equals ( TableOrder.that )  )  {^126^^^^^122^130^else if  ( this.equals ( TableOrder.BY_COLUMN )  )  {^[CLASS] TableOrder  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] TableOrder  BY_COLUMN  BY_ROW  that  String  name  boolean  long  serialVersionUID  