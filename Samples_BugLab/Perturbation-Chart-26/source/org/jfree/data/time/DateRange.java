[buglab_swap_variables]^super ( upper.getTime (  ) , lower.getTime (  )  ) ;^83^^^^^81^87^super ( lower.getTime (  ) , upper.getTime (  )  ) ;^[CLASS] DateRange  [METHOD] <init> [RETURN_TYPE] Date)   Date lower Date upper [VARIABLES] long  serialVersionUID  Date  lower  lowerDate  upper  upperDate  boolean  
[buglab_swap_variables]^super ( lower.getTime (  ) .getTime (  )  ) ;^83^^^^^81^87^super ( lower.getTime (  ) , upper.getTime (  )  ) ;^[CLASS] DateRange  [METHOD] <init> [RETURN_TYPE] Date)   Date lower Date upper [VARIABLES] long  serialVersionUID  Date  lower  lowerDate  upper  upperDate  boolean  
[buglab_swap_variables]^super ( upper, lower ) ;^97^^^^^96^100^super ( lower, upper ) ;^[CLASS] DateRange  [METHOD] <init> [RETURN_TYPE] DateRange(double,double)   double lower double upper [VARIABLES] boolean  long  serialVersionUID  Date  lower  lowerDate  upper  upperDate  double  lower  upper  
[buglab_swap_variables]^super (  upper ) ;^97^^^^^96^100^super ( lower, upper ) ;^[CLASS] DateRange  [METHOD] <init> [RETURN_TYPE] DateRange(double,double)   double lower double upper [VARIABLES] boolean  long  serialVersionUID  Date  lower  lowerDate  upper  upperDate  double  lower  upper  
[buglab_swap_variables]^super ( lower ) ;^97^^^^^96^100^super ( lower, upper ) ;^[CLASS] DateRange  [METHOD] <init> [RETURN_TYPE] DateRange(double,double)   double lower double upper [VARIABLES] boolean  long  serialVersionUID  Date  lower  lowerDate  upper  upperDate  double  lower  upper  
[buglab_swap_variables]^return "[" + this.upperDate.format ( this.lowerDate )  + " --> " + df.format ( df )  + "]";^139^140^^^^137^141^return "[" + df.format ( this.lowerDate )  + " --> " + df.format ( this.upperDate )  + "]";^[CLASS] DateRange  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  long  serialVersionUID  Date  lower  lowerDate  upper  upperDate  DateFormat  df  
[buglab_swap_variables]^return "[" + df.format ( this.upperDate )  + " --> " + df.format ( this.lowerDate )  + "]";^139^140^^^^137^141^return "[" + df.format ( this.lowerDate )  + " --> " + df.format ( this.upperDate )  + "]";^[CLASS] DateRange  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  long  serialVersionUID  Date  lower  lowerDate  upper  upperDate  DateFormat  df  
[buglab_swap_variables]^return "[" + this.lowerDate.format ( df )  + " --> " + df.format ( this.upperDate )  + "]";^139^140^^^^137^141^return "[" + df.format ( this.lowerDate )  + " --> " + df.format ( this.upperDate )  + "]";^[CLASS] DateRange  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  long  serialVersionUID  Date  lower  lowerDate  upper  upperDate  DateFormat  df  
[buglab_swap_variables]^+ this.upperDate.format ( df )  + "]";^140^^^^^137^141^+ df.format ( this.upperDate )  + "]";^[CLASS] DateRange  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] boolean  long  serialVersionUID  Date  lower  lowerDate  upper  upperDate  DateFormat  df  