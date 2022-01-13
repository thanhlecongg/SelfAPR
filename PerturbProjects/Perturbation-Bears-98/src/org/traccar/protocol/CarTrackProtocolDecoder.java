[REPLACE]^private static final Pattern PATTERN =  new DateBuilder (  )  .text ( "$$" ) .number ( " ( d+ ) " ) .text ( "?" ) .expression ( "*" )^36^37^38^39^^36^57^[REPLACE] private static final Pattern PATTERN = new PatternBuilder (  ) .text ( "$$" ) .number ( " ( d+ ) " ) .text ( "?" ) .expression ( "*" )^ [CLASS] CarTrackProtocolDecoder  
[REPLACE]^if  ( parser.matches (  )  )  {^64^^^^^61^106^[REPLACE] if  ( !parser.matches (  )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^return null;^64^65^66^^^61^106^[ADD] if  ( !parser.matches (  )  )  { return null; }^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^return true;^65^^^^^61^106^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position .setCourse (  )  ;^69^^^^^61^106^[REPLACE] position.setProtocol ( getProtocolName (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^69^^^^^61^106^[ADD] position.setProtocol ( getProtocolName (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^DeviceSession deviceSession = getDeviceSession ( channel, remoteAddress, parser .nextInt (  )   ) ;^71^^^^^61^106^[REPLACE] DeviceSession deviceSession = getDeviceSession ( channel, remoteAddress, parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^if  ( deviceSession != null )  {^72^^^^^61^106^[REPLACE] if  ( deviceSession == null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^return true;^73^^^^^61^106^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^return false;^73^^^^^61^106^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( deviceSession.getDeviceId (  )  ) ;^75^^^^^61^106^[REPLACE] position.setDeviceId ( deviceSession.getDeviceId (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( "command", parser.nextInt (  )  ) ;^77^^^^^61^106^[REPLACE] position.set ( "command", parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^parser.nextInt (  )  ;^77^^^^^61^106^[REPLACE] position.set ( "command", parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^DateBuilder dateBuilder = new DateBuilder (  ) .setTime ( parser.next (  ) , parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^79^80^^^^61^106^[REPLACE] DateBuilder dateBuilder = new DateBuilder (  ) .setTime ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setSpeed ( parser.nextDouble (  )  )  ;^82^^^^^61^106^[REPLACE] position.setValid ( parser.next (  ) .equals ( "A" )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^position.setLatitude ( parser.nextCoordinate (  )  ) ;^82^83^^^^61^106^[ADD] position.setValid ( parser.next (  ) .equals ( "A" )  ) ; position.setLatitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setValid ( parser.next (  ) .number ( "A" )  ) ;^82^^^^^61^106^[REPLACE] position.setValid ( parser.next (  ) .equals ( "A" )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setValid ( parser.nextInt (  ) .equals ( "A" )  ) ;^82^^^^^61^106^[REPLACE] position.setValid ( parser.next (  ) .equals ( "A" )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( parser.nextCoordinate (  )  ) ;^83^^^^^61^106^[REPLACE] position.setLatitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^83^84^^^^61^106^[ADD] position.setLatitude ( parser.nextCoordinate (  )  ) ; position.setLongitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^parser.nextInt (  )  ;^83^^^^^61^106^[REPLACE] position.setLatitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setLatitude ( parser.nextCoordinate (  )  )  ;^84^^^^^61^106^[REPLACE] position.setLongitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^position.setSpeed ( parser.nextDouble (  )  ) ;^84^85^^^^61^106^[ADD] position.setLongitude ( parser.nextCoordinate (  )  ) ; position.setSpeed ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setLongitude ( parser.nextInt (  )  ) ;^84^^^^^61^106^[REPLACE] position.setLongitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setSpeed ( parser.next (  )  ) ;^85^^^^^61^106^[REPLACE] position.setSpeed ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setCourse ( parser.next (  )  ) ;^86^^^^^61^106^[REPLACE] position.setCourse ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^parser.next (  )  ;^86^^^^^61^106^[REPLACE] position.setCourse ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^86^^^^^61^106^[ADD] position.setCourse ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^dateBuilder.setDateReverse ( parser.next (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^88^^^^^61^106^[REPLACE] dateBuilder.setDateReverse ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setTime ( dateBuilder.setTime (  )  ) ;^89^^^^^61^106^[REPLACE] position.setTime ( dateBuilder.getDate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.PREFIX_IO  1, parser.next (  )  ) ;^91^^^^^61^106^[REPLACE] position.set ( Position.PREFIX_IO + 1, parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.PREFIX_IO + 1, parser.nextInt (  )  ) ;^91^^^^^61^106^[REPLACE] position.set ( Position.PREFIX_IO + 1, parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^93^^^^^61^106^[ADD] String odometer = parser.next (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^odometer = odometer.replace ( ";", "B" )  ;^94^^^^^61^106^[REPLACE] odometer = odometer.replace ( ":", "A" ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^odometer = odometer.replace ( ":", "A" )  ;^95^^^^^61^106^[REPLACE] odometer = odometer.replace ( ";", "B" ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^odometer = odometer.replace ( ":", "A" )  ;^96^^^^^61^106^[REPLACE] odometer = odometer.replace ( "<", "C" ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^odometer = odometer.replace ( ":", "A" )  ;^97^^^^^61^106^[REPLACE] odometer = odometer.replace ( "=", "D" ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^odometer = odometer.replace ( ":", "A" )  ;^98^^^^^61^106^[REPLACE] odometer = odometer.replace ( ">", "E" ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^odometer = odometer.replace ( ":", "A" )  ;^99^^^^^61^106^[REPLACE] odometer = odometer.replace ( "?", "F" ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.KEY_ODOMETER, Integer.parseInt ( odometer, 7 )  ) ;^100^^^^^61^106^[REPLACE] position.set ( Position.KEY_ODOMETER, Integer.parseInt ( odometer, 16 )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.KEY_ODOMETER, Integer.parseInt ( odometer, 6 )  ) ;^100^^^^^61^106^[REPLACE] position.set ( Position.KEY_ODOMETER, Integer.parseInt ( odometer, 16 )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REMOVE]^position.set ( Position.KEY_ODOMETER, parseInt ( odometer, 16 )  )  ;^100^^^^^61^106^[REMOVE] ^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^parser.nextInt (  ) ;^102^^^^^61^106^[REPLACE] parser.next (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set (  (  ( PREFIX_IO )  + 1 ) , parser.next (  )  )  ;^103^^^^^61^106^[REPLACE] position.set ( Position.PREFIX_ADC + 1, parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^parser.nextInt (  )  ;^103^^^^^61^106^[REPLACE] position.set ( Position.PREFIX_ADC + 1, parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CarTrackProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String odometer  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
