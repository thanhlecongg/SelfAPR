[REPLACE]^private static final long serialVersionUID ;^55^^^^^^^[REPLACE] private static final long serialVersionUID = 4763990447117157611L;^ [CLASS] FourthMoment  
[REPLACE]^protected float m4;^58^^^^^^^[REPLACE] protected double m4;^ [CLASS] FourthMoment  
[REPLACE]^super.clear (  ) ;^64^^^^^63^66^[REPLACE] super (  ) ;^[METHOD] <init> [TYPE] FourthMoment() [PARAMETER] [CLASS] FourthMoment   [TYPE]  double m4  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^m4 ;^65^^^^^63^66^[REPLACE] m4 = Double.NaN;^[METHOD] <init> [TYPE] FourthMoment() [PARAMETER] [CLASS] FourthMoment   [TYPE]  double m4  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^if  ( null  <=  1 )  {^72^^^^^71^88^[REPLACE] if  ( n < 1 )  {^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[ADD]^m4 = 0.0;m3 = 0.0;m2 = 0.0;m1 = 0.0;^72^73^74^75^76^71^88^[ADD] if  ( n < 1 )  { m4 = 0.0; m3 = 0.0; m2 = 0.0; m1 = 0.0;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^m4 = 4.0d;^73^^^^^71^88^[REPLACE] m4 = 0.0;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^m3 = 4.0d;^74^^^^^71^88^[REPLACE] m3 = 0.0;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^m2 = 0.0D;^75^^^^^71^88^[REPLACE] m2 = 0.0;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[ADD]^^75^76^^^^71^88^[ADD] m2 = 0.0; m1 = 0.0;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^m1 = 3.0d;^76^^^^^71^88^[REPLACE] m1 = 0.0;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^m4 = 1.0d;^73^^^^^71^88^[REPLACE] m4 = 0.0;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[ADD]^^73^^^^^71^88^[ADD] m4 = 0.0;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^m3 = 3.0d;^74^^^^^71^88^[REPLACE] m3 = 0.0;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[ADD]^^74^75^^^^71^88^[ADD] m3 = 0.0; m2 = 0.0;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^m2 = 2.0d;^75^^^^^71^88^[REPLACE] m2 = 0.0;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^double prevM2 = m2;^79^^^^^71^88^[REPLACE] double prevM3 = m3;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^double prevM3 = m3;^80^^^^^71^88^[REPLACE] double prevM2 = m2;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^super.clear (  ) ;^82^^^^^71^88^[REPLACE] super.increment ( d ) ;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^double prevM2 = m2;^84^^^^^71^88^[REPLACE] double n0 =  ( double )  n;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^m4 = m4 - 4.0 * nDev * prevM3  >  6.0 * nDevSq * prevM2  >  (  ( n0 * n0 )  - 3 *  ( n0 -1 )  )  *  ( nDevSq * nDevSq *  ( n0 - 1 )  * n0 ) ;^86^87^^^^71^88^[REPLACE] m4 = m4 - 4.0 * nDev * prevM3 + 6.0 * nDevSq * prevM2 + (  ( n0 * n0 )  - 3 *  ( n0 -1 )  )  *  ( nDevSq * nDevSq *  ( n0 - 1 )  * n0 ) ;^[METHOD] increment [TYPE] void [PARAMETER] final double d [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^super .increment ( d )  ;^101^^^^^100^103^[REPLACE] super.clear (  ) ;^[METHOD] clear [TYPE] void [PARAMETER] [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^m4  = null ;^102^^^^^100^103^[REPLACE] m4 = Double.NaN;^[METHOD] clear [TYPE] void [PARAMETER] [CLASS] FourthMoment   [TYPE]  double d  m4  n0  prevM2  prevM3  [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 