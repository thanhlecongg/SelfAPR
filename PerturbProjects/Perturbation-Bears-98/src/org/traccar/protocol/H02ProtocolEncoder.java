[REPLACE]^private static  String MARKER = "HQ";^27^^^^^^^[REPLACE] private static final String MARKER = "HQ";^ [CLASS] H02ProtocolEncoder  
[REPLACE]^String uniqueId = getUniqueId ( command .getType (  )   ) ;^44^^^^^43^65^[REPLACE] String uniqueId = getUniqueId ( command.getDeviceId (  )  ) ;^[METHOD] encodeCommand [TYPE] Object [PARAMETER] Command command DateTime time [CLASS] H02ProtocolEncoder   [TYPE]  String MARKER  param  type  uniqueId  [TYPE]  boolean false  true  [TYPE]  Command command  [TYPE]  DateTime time 
[REPLACE]^return formatCommand ( time, type, "SCF", "0", "0" ) ;^48^^^^^43^65^[REPLACE] return formatCommand ( time, uniqueId, "SCF", "0", "0" ) ;^[METHOD] encodeCommand [TYPE] Object [PARAMETER] Command command DateTime time [CLASS] H02ProtocolEncoder   [TYPE]  String MARKER  param  type  uniqueId  [TYPE]  boolean false  true  [TYPE]  Command command  [TYPE]  DateTime time 
[REPLACE]^return formatCommand ( time, uniqueId, "SCF", "0", "0" )  ;^50^^^^^43^65^[REPLACE] return formatCommand ( time, uniqueId, "SCF", "1", "1" ) ;^[METHOD] encodeCommand [TYPE] Object [PARAMETER] Command command DateTime time [CLASS] H02ProtocolEncoder   [TYPE]  String MARKER  param  type  uniqueId  [TYPE]  boolean false  true  [TYPE]  Command command  [TYPE]  DateTime time 
[REPLACE]^return formatCommand ( time, type, "S20", "1", "3", "10", "3", "5", "5", "3", "5", "3", "5", "3", "5" ) ;^52^53^^^^43^65^[REPLACE] return formatCommand ( time, uniqueId, "S20", "1", "3", "10", "3", "5", "5", "3", "5", "3", "5", "3", "5" ) ;^[METHOD] encodeCommand [TYPE] Object [PARAMETER] Command command DateTime time [CLASS] H02ProtocolEncoder   [TYPE]  String MARKER  param  type  uniqueId  [TYPE]  boolean false  true  [TYPE]  Command command  [TYPE]  DateTime time 
[REPLACE]^return formatCommand ( time, uniqueId, "SCF", "0", "0" )  ;^55^^^^^43^65^[REPLACE] return formatCommand ( time, uniqueId, "S20", "0", "0" ) ;^[METHOD] encodeCommand [TYPE] Object [PARAMETER] Command command DateTime time [CLASS] H02ProtocolEncoder   [TYPE]  String MARKER  param  type  uniqueId  [TYPE]  boolean false  true  [TYPE]  Command command  [TYPE]  DateTime time 
[REPLACE]^return formatCommand ( time, uniqueId, "S71", "22", command.getUniqueIdAttributes (  ) .get ( Command.KEY_FREQUENCY ) .toString (  )  ) ;^57^58^^^^43^65^[REPLACE] return formatCommand ( time, uniqueId, "S71", "22", command.getAttributes (  ) .get ( Command.KEY_FREQUENCY ) .toString (  )  ) ;^[METHOD] encodeCommand [TYPE] Object [PARAMETER] Command command DateTime time [CLASS] H02ProtocolEncoder   [TYPE]  String MARKER  param  type  uniqueId  [TYPE]  boolean false  true  [TYPE]  Command command  [TYPE]  DateTime time 
[REPLACE]^return this;^64^^^^^43^65^[REPLACE] return null;^[METHOD] encodeCommand [TYPE] Object [PARAMETER] Command command DateTime time [CLASS] H02ProtocolEncoder   [TYPE]  String MARKER  param  type  uniqueId  [TYPE]  boolean false  true  [TYPE]  Command command  [TYPE]  DateTime time 
