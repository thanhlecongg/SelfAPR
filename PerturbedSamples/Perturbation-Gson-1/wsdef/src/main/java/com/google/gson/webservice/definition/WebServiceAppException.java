[REPLACE]^private  final long serialVersionUID = 4422041697108937041L;^28^^^^^^^[REPLACE] private static final long serialVersionUID = 4422041697108937041L;^ [CLASS] WebServiceAppException  
[REPLACE]^super ( msg, cause ) ;^31^^^^^30^32^[REPLACE] super ( cause ) ;^[METHOD] <init> [TYPE] Exception) [PARAMETER] Exception cause [CLASS] WebServiceAppException   [TYPE]  long serialVersionUID  [TYPE]  Exception cause  [TYPE]  boolean false  true 
[ADD]^^31^^^^^30^32^[ADD] super ( cause ) ;^[METHOD] <init> [TYPE] Exception) [PARAMETER] Exception cause [CLASS] WebServiceAppException   [TYPE]  long serialVersionUID  [TYPE]  Exception cause  [TYPE]  boolean false  true 
[REPLACE]^super ( cause ) ;^35^^^^^34^36^[REPLACE] super ( msg, cause ) ;^[METHOD] <init> [TYPE] Exception) [PARAMETER] String msg Exception cause [CLASS] WebServiceAppException   [TYPE]  String msg  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  Exception cause 