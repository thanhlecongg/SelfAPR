[buglab_swap_variables]^super ( param, base, minDaysInFirstWeek ) ;^148^^^^^147^149^super ( base, param, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] <init> [RETURN_TYPE] Object,int)   Chronology base Object param int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  Object  param  Chronology  base  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  
[buglab_swap_variables]^super (  param, minDaysInFirstWeek ) ;^148^^^^^147^149^super ( base, param, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] <init> [RETURN_TYPE] Object,int)   Chronology base Object param int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  Object  param  Chronology  base  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  
[buglab_swap_variables]^super ( base, minDaysInFirstWeek, param ) ;^148^^^^^147^149^super ( base, param, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] <init> [RETURN_TYPE] Object,int)   Chronology base Object param int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  Object  param  Chronology  base  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  
[buglab_swap_variables]^super ( base,  minDaysInFirstWeek ) ;^148^^^^^147^149^super ( base, param, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] <init> [RETURN_TYPE] Object,int)   Chronology base Object param int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  Object  param  Chronology  base  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  
[buglab_swap_variables]^super ( minDaysInFirstWeek, param, base ) ;^148^^^^^147^149^super ( base, param, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] <init> [RETURN_TYPE] Object,int)   Chronology base Object param int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  Object  param  Chronology  base  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  
[buglab_swap_variables]^super ( base, param ) ;^148^^^^^147^149^super ( base, param, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] <init> [RETURN_TYPE] Object,int)   Chronology base Object param int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  Object  param  Chronology  base  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  
[buglab_swap_variables]^cCache.put (  chronos ) ;^119^^^^^110^139^cCache.put ( zone, chronos ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^cCache.put ( zone ) ;^119^^^^^110^139^cCache.put ( zone, chronos ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^chrono = new GregorianChronology ( ZonedChronology.getInstance ( minDaysInFirstWeek, zone ) , null, chrono ) ;^132^133^^^^128^134^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^chrono = new GregorianChronology ( ZonedChronology.getInstance (  zone ) , null, minDaysInFirstWeek ) ;^132^133^^^^128^134^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, minDaysInFirstWeek ) , null, zone ) ;^132^133^^^^128^134^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono ) , null, minDaysInFirstWeek ) ;^132^133^^^^128^134^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null ) ;^132^133^^^^128^134^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^chrono = new GregorianChronology ( ZonedChronology.getInstance ( zone, chrono ) , null, minDaysInFirstWeek ) ;^132^133^^^^128^134^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^( ZonedChronology.getInstance ( zone, chrono ) , null, minDaysInFirstWeek ) ;^133^^^^^128^134^( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^( ZonedChronology.getInstance (  zone ) , null, minDaysInFirstWeek ) ;^133^^^^^128^134^( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^( ZonedChronology.getInstance ( chrono ) , null, minDaysInFirstWeek ) ;^133^^^^^128^134^( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^chrono = new GregorianChronology ( ZonedChronology.getInstance ( minDaysInFirstWeek, zone ) , null, chrono ) ;^132^133^^^^110^139^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^chrono = new GregorianChronology ( ZonedChronology.getInstance (  zone ) , null, minDaysInFirstWeek ) ;^132^133^^^^110^139^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^chrono = new GregorianChronology ( ZonedChronology.getInstance ( zone, chrono ) , null, minDaysInFirstWeek ) ;^132^133^^^^110^139^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono ) , null, minDaysInFirstWeek ) ;^132^133^^^^110^139^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null ) ;^132^133^^^^110^139^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, minDaysInFirstWeek ) , null, zone ) ;^132^133^^^^110^139^chrono = new GregorianChronology ( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^( ZonedChronology.getInstance ( zone, chrono ) , null, minDaysInFirstWeek ) ;^133^^^^^110^139^( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^( ZonedChronology.getInstance (  zone ) , null, minDaysInFirstWeek ) ;^133^^^^^110^139^( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^( ZonedChronology.getInstance ( chrono ) , null, minDaysInFirstWeek ) ;^133^^^^^110^139^( ZonedChronology.getInstance ( chrono, zone ) , null, minDaysInFirstWeek ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^GregorianChronology[] chronos = zone.get ( cCache ) ;^116^^^^^110^139^GregorianChronology[] chronos = cCache.get ( zone ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^cCache.put ( chronos, zone ) ;^119^^^^^110^139^cCache.put ( zone, chronos ) ;^[CLASS] GregorianChronology  [METHOD] getInstance [RETURN_TYPE] GregorianChronology   DateTimeZone zone int minDaysInFirstWeek [VARIABLES] boolean  GregorianChronology  INSTANCE_UTC  chrono  Map  cCache  GregorianChronology[]  chronos  ArrayIndexOutOfBoundsException  e  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDaysInFirstWeek  DateTimeZone  zone  
[buglab_swap_variables]^return minDays == null ? getInstance ( DateTimeZone.UTC, base )  : getInstance ( base.getZone (  ) , minDays ) ;^158^159^160^^^154^161^return base == null ? getInstance ( DateTimeZone.UTC, minDays )  : getInstance ( base.getZone (  ) , minDays ) ;^[CLASS] GregorianChronology  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Chronology  base  boolean  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDays  minDaysInFirstWeek  GregorianChronology  INSTANCE_UTC  chrono  
[buglab_swap_variables]^return base == null ? getInstance ( DateTimeZone.UTC )  : getInstance ( base.getZone (  ) , minDays ) ;^158^159^160^^^154^161^return base == null ? getInstance ( DateTimeZone.UTC, minDays )  : getInstance ( base.getZone (  ) , minDays ) ;^[CLASS] GregorianChronology  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Chronology  base  boolean  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDays  minDaysInFirstWeek  GregorianChronology  INSTANCE_UTC  chrono  
[buglab_swap_variables]^getInstance ( minDays.getZone (  ) , base ) ;^160^^^^^154^161^getInstance ( base.getZone (  ) , minDays ) ;^[CLASS] GregorianChronology  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Chronology  base  boolean  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDays  minDaysInFirstWeek  GregorianChronology  INSTANCE_UTC  chrono  
[buglab_swap_variables]^getInstance ( base.getZone (  )  ) ;^160^^^^^154^161^getInstance ( base.getZone (  ) , minDays ) ;^[CLASS] GregorianChronology  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Chronology  base  boolean  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  minDays  minDaysInFirstWeek  GregorianChronology  INSTANCE_UTC  chrono  
[buglab_swap_variables]^leapYears =  ( leapYears >> 2 )  - year +  ( leapYears >> 2 ) ;^211^^^^^200^218^leapYears =  ( year >> 2 )  - leapYears +  ( leapYears >> 2 ) ;^[CLASS] GregorianChronology  [METHOD] calculateFirstDayOfYearMillis [RETURN_TYPE] long   int year [VARIABLES] boolean  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  leapYears  minDays  minDaysInFirstWeek  year  GregorianChronology  INSTANCE_UTC  chrono  
[buglab_swap_variables]^leapYears =  (  ( leapYears + 3 )  >> 2 )  - year +  (  ( leapYears + 3 )  >> 2 )  - 1;^209^^^^^200^218^leapYears =  (  ( year + 3 )  >> 2 )  - leapYears +  (  ( leapYears + 3 )  >> 2 )  - 1;^[CLASS] GregorianChronology  [METHOD] calculateFirstDayOfYearMillis [RETURN_TYPE] long   int year [VARIABLES] boolean  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  leapYears  minDays  minDaysInFirstWeek  year  GregorianChronology  INSTANCE_UTC  chrono  
[buglab_swap_variables]^return  ( leapYears * 365L +  ( year - DAYS_0000_TO_1970 )  )  * DateTimeConstants.MILLIS_PER_DAY;^217^^^^^200^218^return  ( year * 365L +  ( leapYears - DAYS_0000_TO_1970 )  )  * DateTimeConstants.MILLIS_PER_DAY;^[CLASS] GregorianChronology  [METHOD] calculateFirstDayOfYearMillis [RETURN_TYPE] long   int year [VARIABLES] boolean  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  leapYears  minDays  minDaysInFirstWeek  year  GregorianChronology  INSTANCE_UTC  chrono  
[buglab_swap_variables]^return  ( year * 365L +  ( DAYS_0000_TO_1970 - leapYears )  )  * DateTimeConstants.MILLIS_PER_DAY;^217^^^^^200^218^return  ( year * 365L +  ( leapYears - DAYS_0000_TO_1970 )  )  * DateTimeConstants.MILLIS_PER_DAY;^[CLASS] GregorianChronology  [METHOD] calculateFirstDayOfYearMillis [RETURN_TYPE] long   int year [VARIABLES] boolean  Map  cCache  long  MILLIS_PER_MONTH  MILLIS_PER_YEAR  serialVersionUID  int  DAYS_0000_TO_1970  MAX_YEAR  MIN_YEAR  leapYears  minDays  minDaysInFirstWeek  year  GregorianChronology  INSTANCE_UTC  chrono  