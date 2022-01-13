[REPLACE]^private static final Pattern PATTERN =  new DateBuilder (  )  .number ( "d+:" ) .number ( " ( d+ ) :" ) .number ( "d+:" ) .text ( "*," )^38^39^40^41^^38^63^[REPLACE] private static final Pattern PATTERN = new PatternBuilder (  ) .number ( "d+:" ) .number ( " ( d+ ) :" ) .number ( "d+:" ) .text ( "*," )^ [CLASS] SupermateProtocolDecoder  
[ADD]^^69^^^^^67^124^[ADD] Parser parser = new Parser ( PATTERN,  ( String )  msg ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^if  ( !parser.nextInt (   )  ||  ( parser.nextInt ( 16 )  )  == 8  )  )  {^70^^^^^67^124^[REPLACE] if  ( !parser.matches (  )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^return false;^71^^^^^67^124^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^return true;^71^^^^^67^124^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^74^^^^^67^124^[ADD] Position position = new Position (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position .setTime (  )  ;^75^^^^^67^124^[REPLACE] position.setProtocol ( getProtocolName (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REMOVE]^position.set ( "signal", parser.next (  )  )  ;^75^^^^^67^124^[REMOVE] ^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^String imei = parser .nextInt ( this )  ;^77^^^^^67^124^[REPLACE] String imei = parser.next (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^78^^^^^67^124^[ADD] DeviceSession deviceSession = getDeviceSession ( channel, remoteAddress, imei ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^if  ( deviceSession != null )  {^79^^^^^67^124^[REPLACE] if  ( deviceSession == null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^return this;^80^^^^^67^124^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^return true;^80^^^^^67^124^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( deviceSession.getDeviceId (  )  ) ;^82^^^^^67^124^[REPLACE] position.setDeviceId ( deviceSession.getDeviceId (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^82^^^^^67^124^[ADD] position.setDeviceId ( deviceSession.getDeviceId (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( "command", parser.next (  )  )  ;^84^^^^^67^124^[REPLACE] position.set ( "commandId", parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^84^85^^^^67^124^[ADD] position.set ( "commandId", parser.next (  )  ) ; position.set ( "command", parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( "commandId", parser .nextInt ( null )   ) ;^84^^^^^67^124^[REPLACE] position.set ( "commandId", parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( "commandId", parser.next (  )  )  ;^85^^^^^67^124^[REPLACE] position.set ( "command", parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( "command", parser.nextDouble (  )  ) ;^85^^^^^67^124^[REPLACE] position.set ( "command", parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REMOVE]^parser.nextInt ( 16 )  ;^85^^^^^67^124^[REMOVE] ^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setValid ( parser.nextDouble (  ) .equals ( "A" )  ) ;^87^^^^^67^124^[REPLACE] position.setValid ( parser.next (  ) .equals ( "A" )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setValid ( parser.next (  ) .number ( "A" )  ) ;^87^^^^^67^124^[REPLACE] position.setValid ( parser.next (  ) .equals ( "A" )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^87^^^^^67^124^[ADD] position.setValid ( parser.next (  ) .equals ( "A" )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^DateBuilder dateBuilder = new DateBuilder (  ) .setDate ( parser.nextInt ( 16L ) , parser.nextInt ( 16L ) , parser.nextInt ( 16L )  ) .setTime ( parser.nextInt ( 16L ) , parser.nextInt ( 16L ) , parser.nextInt ( 16L )  ) ;^89^90^91^^^67^124^[REPLACE] DateBuilder dateBuilder = new DateBuilder (  ) .setDate ( parser.nextInt ( 16 ) , parser.nextInt ( 16 ) , parser.nextInt ( 16 )  ) .setTime ( parser.nextInt ( 16 ) , parser.nextInt ( 16 ) , parser.nextInt ( 16 )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setTime ( dateBuilder.setDate (  )  ) ;^92^^^^^67^124^[REPLACE] position.setTime ( dateBuilder.getDate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setTime ( dateBuilder .setDate (  )   ) ;^92^^^^^67^124^[REPLACE] position.setTime ( dateBuilder.getDate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^if  (   8 )  {^94^^^^^67^124^[REPLACE] if  ( parser.nextInt ( 16 )  == 8 )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setLongitude ( parser.nextInt ( 16 )   600000.0 ) ;^97^^^^^94^98^[REPLACE] position.setLatitude ( parser.nextInt ( 16 )  / 600000.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setLatitude ( parser.nextInt ( 16 / 2 )  / 600000.0 ) ;^97^^^^^94^98^[REPLACE] position.setLatitude ( parser.nextInt ( 16 )  / 600000.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^97^^^^^94^98^[ADD] position.setLatitude ( parser.nextInt ( 16 )  / 600000.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setLongitude ( -parser.nextInt ( 16 >>> 4 )   600000.0 ) ;^95^^^^^67^124^[REPLACE] position.setLatitude ( -parser.nextInt ( 16 )  / 600000.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setLatitude ( -parser.nextInt ( 16 + 4 )  / 600000.0 ) ;^95^^^^^67^124^[REPLACE] position.setLatitude ( -parser.nextInt ( 16 )  / 600000.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^if  (   8 )  {^100^^^^^67^124^[REPLACE] if  ( parser.nextInt ( 16 )  == 8 )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setLongitude (  (  ( - ( parser.nextInt ( 16 )  )  )  / 600000.0 )  )  ;^103^^^^^100^104^[REPLACE] position.setLongitude ( parser.nextInt ( 16 )  / 600000.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^103^^^^^100^104^[ADD] position.setLongitude ( parser.nextInt ( 16 )  / 600000.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setLongitude ( parser.nextInt ( 16 / 2 )  / 600000.0 ) ;^103^^^^^100^104^[REPLACE] position.setLongitude ( parser.nextInt ( 16 )  / 600000.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setLatitude (  (  ( - ( parser.nextInt ( 16 )  )  )  / 600000.0 )  )  ;^101^^^^^67^124^[REPLACE] position.setLongitude ( -parser.nextInt ( 16 )  / 600000.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setLongitude ( -parser.nextInt ( 12 )  / 600000.0 ) ;^101^^^^^67^124^[REPLACE] position.setLongitude ( -parser.nextInt ( 16 )  / 600000.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setSpeed ( parser.nextInt ( 0 )   100.0 ) ;^106^^^^^67^124^[REPLACE] position.setSpeed ( parser.nextInt ( 16 )  / 100.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^106^107^^^^67^124^[ADD] position.setSpeed ( parser.nextInt ( 16 )  / 100.0 ) ; position.setCourse ( parser.nextInt ( 16 )  / 100.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^parser.nextInt ( 16 )  ;^106^^^^^67^124^[REPLACE] position.setSpeed ( parser.nextInt ( 16 )  / 100.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setSpeed (  (  ( parser.nextInt ( 16 )  )  / 100.0 )  )  ;^107^^^^^67^124^[REPLACE] position.setCourse ( parser.nextInt ( 16 )  / 100.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.setCourse ( parser.nextInt ( 16 + 4 )  / 100.0 ) ;^107^^^^^67^124^[REPLACE] position.setCourse ( parser.nextInt ( 16 )  / 100.0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position .set ( imei )  ;^109^^^^^67^124^[REPLACE] position.set ( Position.KEY_STATUS, parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.KEY_STATUS, parser.nextDouble (  )  ) ;^109^^^^^67^124^[REPLACE] position.set ( Position.KEY_STATUS, parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position .set ( content )  ;^110^^^^^67^124^[REPLACE] position.set ( "signal", parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( "signal", parser .nextInt ( null )   ) ;^110^^^^^67^124^[REPLACE] position.set ( "signal", parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^110^^^^^67^124^[ADD] position.set ( "signal", parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.KEY_POWER, parser.next (  )  ) ;^111^^^^^67^124^[REPLACE] position.set ( Position.KEY_POWER, parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( "signal", parser.next (  )  )  ;^112^^^^^67^124^[REPLACE] position.set ( "oil", parser.nextInt ( 16 )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( "oil", parser.nextInt ( 2 )  ) ;^112^^^^^67^124^[REPLACE] position.set ( "oil", parser.nextInt ( 16 )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.KEY_ODOMETER, parser.nextInt ( 2 )  ) ;^113^^^^^67^124^[REPLACE] position.set ( Position.KEY_ODOMETER, parser.nextInt ( 16 )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.KEY_ODOMETER, parser.nextInt ( 16L )  ) ;^113^^^^^67^124^[REPLACE] position.set ( Position.KEY_ODOMETER, parser.nextInt ( 16 )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^if  ( channel == null )  {^115^^^^^67^124^[REPLACE] if  ( channel != null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[ADD]^^116^^^^^67^124^[ADD] Calendar calendar = Calendar.getInstance (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
[REPLACE]^String content = String.format ( "#1:%s:1:*,00000000,UP,%02x%02x%02x,%02x%02x%02x#", imei, calendar.get ( Calendar.YEAR ) , calendar.get ( Calendar.MONTH )   1, calendar.get ( Calendar.DAY_OF_MONTH ) , calendar.get ( Calendar.HOUR_OF_DAY ) , calendar.get ( Calendar.MINUTE ) , calendar.get ( Calendar.SECOND )  ) ;^117^118^119^^^67^124^[REPLACE] String content = String.format ( "#1:%s:1:*,00000000,UP,%02x%02x%02x,%02x%02x%02x#", imei, calendar.get ( Calendar.YEAR ) , calendar.get ( Calendar.MONTH )  + 1, calendar.get ( Calendar.DAY_OF_MONTH ) , calendar.get ( Calendar.HOUR_OF_DAY ) , calendar.get ( Calendar.MINUTE ) , calendar.get ( Calendar.SECOND )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] SupermateProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Calendar calendar  [TYPE]  Object msg  [TYPE]  String content  imei  [TYPE]  SocketAddress remoteAddress  [TYPE]  Parser parser 
