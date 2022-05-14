[buglab_swap_variables]^this ( upperBound, lowerBound, true, true ) ;^88^^^^^87^89^this ( lowerBound, upperBound, true, true ) ;^[CLASS] SimpleHistogramBin  [METHOD] <init> [RETURN_TYPE] SimpleHistogramBin(double,double)   double lowerBound double upperBound [VARIABLES] boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  long  serialVersionUID  int  itemCount  
[buglab_swap_variables]^this (  upperBound, true, true ) ;^88^^^^^87^89^this ( lowerBound, upperBound, true, true ) ;^[CLASS] SimpleHistogramBin  [METHOD] <init> [RETURN_TYPE] SimpleHistogramBin(double,double)   double lowerBound double upperBound [VARIABLES] boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  long  serialVersionUID  int  itemCount  
[buglab_swap_variables]^this ( lowerBound,  true, true ) ;^88^^^^^87^89^this ( lowerBound, upperBound, true, true ) ;^[CLASS] SimpleHistogramBin  [METHOD] <init> [RETURN_TYPE] SimpleHistogramBin(double,double)   double lowerBound double upperBound [VARIABLES] boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  long  serialVersionUID  int  itemCount  
[buglab_swap_variables]^if  ( upperBound >= lowerBound )  {^102^^^^^99^110^if  ( lowerBound >= upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] <init> [RETURN_TYPE] SimpleHistogramBin(double,double,boolean,boolean)   double lowerBound double upperBound boolean includeLowerBound boolean includeUpperBound [VARIABLES] boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  long  serialVersionUID  int  itemCount  
[buglab_swap_variables]^if  ( this.lowerBound < value )  {^160^^^^^156^173^if  ( value < this.lowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] accepts [RETURN_TYPE] boolean   double value [VARIABLES] boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.upperBound > value )  {^163^^^^^156^173^if  ( value > this.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] accepts [RETURN_TYPE] boolean   double value [VARIABLES] boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.lowerBound == value )  {^166^^^^^156^173^if  ( value == this.lowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] accepts [RETURN_TYPE] boolean   double value [VARIABLES] boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.upperBound == value )  {^169^^^^^156^173^if  ( value == this.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] accepts [RETURN_TYPE] boolean   double value [VARIABLES] boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.upperBound < bin.lowerBound.lowerBound )  {^184^^^^^183^197^if  ( this.upperBound < bin.lowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( bin.lowerBound < this.upperBound )  {^184^^^^^183^197^if  ( this.upperBound < bin.lowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.lowerBound > bin.upperBound.upperBound )  {^187^^^^^183^197^if  ( this.lowerBound > bin.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( bin > this.lowerBound.upperBound )  {^187^^^^^183^197^if  ( this.lowerBound > bin.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.lowerBound > bin )  {^187^^^^^183^197^if  ( this.lowerBound > bin.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( bin == this.upperBound.lowerBound )  {^190^^^^^183^197^if  ( this.upperBound == bin.lowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^return bin && this.includeUpperBound.includeLowerBound;^191^^^^^183^197^return this.includeUpperBound && bin.includeLowerBound;^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^return bin.includeLowerBound && this.includeUpperBound;^191^^^^^183^197^return this.includeUpperBound && bin.includeLowerBound;^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^return this.includeUpperBound && bin.includeLowerBound.includeLowerBound;^191^^^^^183^197^return this.includeUpperBound && bin.includeLowerBound;^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( bin.upperBound == this.lowerBound )  {^193^^^^^183^197^if  ( this.lowerBound == bin.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.lowerBound == bin )  {^193^^^^^183^197^if  ( this.lowerBound == bin.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^return this.includeLowerBound && bin.includeUpperBound.includeUpperBound;^194^^^^^183^197^return this.includeLowerBound && bin.includeUpperBound;^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^return bin.includeUpperBound && this.includeLowerBound;^194^^^^^183^197^return this.includeLowerBound && bin.includeUpperBound;^[CLASS] SimpleHistogramBin  [METHOD] overlapsWith [RETURN_TYPE] boolean   SimpleHistogramBin bin [VARIABLES] SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.lowerBound < bin.lowerBound.lowerBound )  {^213^^^^^208^227^if  ( this.lowerBound < bin.lowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( bin.lowerBound < this.lowerBound )  {^213^^^^^208^227^if  ( this.lowerBound < bin.lowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.lowerBound < bin )  {^213^^^^^208^227^if  ( this.lowerBound < bin.lowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( bin > this.lowerBound.lowerBound )  {^216^^^^^208^227^if  ( this.lowerBound > bin.lowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( bin.lowerBound > this.lowerBound )  {^216^^^^^208^227^if  ( this.lowerBound > bin.lowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( bin < this.upperBound.upperBound )  {^220^^^^^208^227^if  ( this.upperBound < bin.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( bin.upperBound < this.upperBound )  {^220^^^^^208^227^if  ( this.upperBound < bin.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( bin > this.upperBound.upperBound )  {^223^^^^^208^227^if  ( this.upperBound > bin.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( bin.upperBound > this.upperBound )  {^223^^^^^208^227^if  ( this.upperBound > bin.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] compareTo [RETURN_TYPE] int   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  bin  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( that != this.lowerBound.lowerBound )  {^241^^^^^236^257^if  ( this.lowerBound != that.lowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  that  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( that.lowerBound != this.lowerBound )  {^241^^^^^236^257^if  ( this.lowerBound != that.lowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  that  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.upperBound != that.upperBound.upperBound )  {^244^^^^^236^257^if  ( this.upperBound != that.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  that  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( that.upperBound != this.upperBound )  {^244^^^^^236^257^if  ( this.upperBound != that.upperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  that  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.includeLowerBound != that.includeLowerBound.includeLowerBound )  {^247^^^^^236^257^if  ( this.includeLowerBound != that.includeLowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  that  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( that.includeLowerBound != this.includeLowerBound )  {^247^^^^^236^257^if  ( this.includeLowerBound != that.includeLowerBound )  {^[CLASS] SimpleHistogramBin  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  that  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( that != this.includeUpperBound.includeUpperBound )  {^250^^^^^236^257^if  ( this.includeUpperBound != that.includeUpperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  that  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( that.includeUpperBound != this.includeUpperBound )  {^250^^^^^236^257^if  ( this.includeUpperBound != that.includeUpperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  that  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.includeUpperBound != that )  {^250^^^^^236^257^if  ( this.includeUpperBound != that.includeUpperBound )  {^[CLASS] SimpleHistogramBin  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  that  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( this.itemCount != that.itemCount.itemCount )  {^253^^^^^236^257^if  ( this.itemCount != that.itemCount )  {^[CLASS] SimpleHistogramBin  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  that  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  
[buglab_swap_variables]^if  ( that.itemCount != this.itemCount )  {^253^^^^^236^257^if  ( this.itemCount != that.itemCount )  {^[CLASS] SimpleHistogramBin  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Object  obj  SimpleHistogramBin  that  boolean  includeLowerBound  includeUpperBound  double  lowerBound  upperBound  value  long  serialVersionUID  int  count  itemCount  