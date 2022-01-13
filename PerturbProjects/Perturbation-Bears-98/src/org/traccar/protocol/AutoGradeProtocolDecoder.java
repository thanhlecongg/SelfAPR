[REPLACE]^private  final Pattern PATTERN =  new DateBuilder (  )  .text ( " ( " ) .number ( "d{12}" ) .number ( " ( d{15} ) " )^36^37^38^39^^36^59^[REPLACE] private static final Pattern PATTERN = new PatternBuilder (  ) .text ( " ( " ) .number ( "d{12}" ) .number ( " ( d{15} ) " )^ [CLASS] AutoGradeProtocolDecoder  
[REPLACE]^if  ( parser.matches (  )  )  {^66^^^^^63^105^[REPLACE] if  ( !parser.matches (  )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^return true;^67^^^^^63^105^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^return this;^67^^^^^63^105^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[ADD]^^70^^^^^63^105^[ADD] DeviceSession deviceSession = getDeviceSession ( channel, remoteAddress, parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^if  ( deviceSession != null )  {^71^^^^^63^105^[REPLACE] if  ( deviceSession == null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^return this;^72^^^^^63^105^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^return false;^72^^^^^63^105^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position .setTime (  )  ;^76^^^^^63^105^[REPLACE] position.setProtocol ( getProtocolName (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[ADD]^position.setDeviceId ( deviceSession.getDeviceId (  )  ) ;^76^77^^^^63^105^[ADD] position.setProtocol ( getProtocolName (  )  ) ; position.setDeviceId ( deviceSession.getDeviceId (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[ADD]^^76^^^^^63^105^[ADD] position.setProtocol ( getProtocolName (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position .setValid (  )  ;^77^^^^^63^105^[REPLACE] position.setDeviceId ( deviceSession.getDeviceId (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^DateBuilder dateBuilder = new DateBuilder (  ) .setDateReverse ( parser.next (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^79^80^^^^63^105^[REPLACE] DateBuilder dateBuilder = new DateBuilder (  ) .setDateReverse ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.setValid ( parser .nextInt (  )  .equals ( "A" )  ) ;^82^^^^^63^105^[REPLACE] position.setValid ( parser.next (  ) .equals ( "A" )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.setValid ( parser.next (  )  .expression ( null )   ) ;^82^^^^^63^105^[REPLACE] position.setValid ( parser.next (  ) .equals ( "A" )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[ADD]^^82^^^^^63^105^[ADD] position.setValid ( parser.next (  ) .equals ( "A" )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.setValid ( parser.nextInt (  ) .equals ( "A" )  ) ;^82^^^^^63^105^[REPLACE] position.setValid ( parser.next (  ) .equals ( "A" )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.set ( parser.nextCoordinate (  )  ) ;^83^^^^^63^105^[REPLACE] position.setLatitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^parser.nextInt (  )  ;^83^^^^^63^105^[REPLACE] position.setLatitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.set ( parser.nextCoordinate (  )  ) ;^84^^^^^63^105^[REPLACE] position.setLongitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[ADD]^^84^^^^^63^105^[ADD] position.setLongitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.setLongitude ( parser.nextInt (  )  ) ;^84^^^^^63^105^[REPLACE] position.setLongitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.setCourse ( parser.nextDouble (  )  )  ;^85^^^^^63^105^[REPLACE] position.setSpeed ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.setSpeed ( parser .nextCoordinate (  )   ) ;^85^^^^^63^105^[REPLACE] position.setSpeed ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[ADD]^^85^^^^^63^105^[ADD] position.setSpeed ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^dateBuilder.setTime ( parser.next (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^87^^^^^63^105^[REPLACE] dateBuilder.setTime ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[ADD]^^87^^^^^63^105^[ADD] dateBuilder.setTime ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REMOVE]^parser.next (  )  ;^87^^^^^63^105^[REMOVE] ^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^dateBuilder.setTime ( parser .next (  )  , parser^87^^^^^63^105^[REPLACE] dateBuilder.setTime ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^parser.next (  )  ;^87^^^^^63^105^[REPLACE] dateBuilder.setTime ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position .getDate (  )  ;^88^^^^^63^105^[REPLACE] position.setTime ( dateBuilder.getDate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.setTime ( dateBuilder.setTime (  )  ) ;^88^^^^^63^105^[REPLACE] position.setTime ( dateBuilder.getDate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.set ( parser.nextDouble (  )  ) ;^90^^^^^63^105^[REPLACE] position.setCourse ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.setCourse ( parser.next (  )  ) ;^90^^^^^63^105^[REPLACE] position.setCourse ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^int status = parser.next (  ) .charAt ( 0 >> 0 ) ;^92^^^^^63^105^[REPLACE] int status = parser.next (  ) .charAt ( 0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.KEY_IGNITION, BitUtil.check ( status, 0 - 0 )  ) ;^94^^^^^63^105^[REPLACE] position.set ( Position.KEY_IGNITION, BitUtil.check ( status, 0 )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.KEY_IGNITION, BitUtil.check ( status, 0 >> 4 )  ) ;^94^^^^^63^105^[REPLACE] position.set ( Position.KEY_IGNITION, BitUtil.check ( status, 0 )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[REPLACE]^for  ( int i = 1 << 0; i <= 5; i++ )  {^96^^^^^63^105^[REPLACE] for  ( int i = 1; i <= 5; i++ )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
[ADD]^^100^101^102^^^63^105^[ADD] for  ( int i = 1; i <= 5; i++ )  { position.set ( "can" + i, parser.next (  )  ) ; }^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] AutoGradeProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  SocketAddress remoteAddress  [TYPE]  int i  status  [TYPE]  Parser parser 
