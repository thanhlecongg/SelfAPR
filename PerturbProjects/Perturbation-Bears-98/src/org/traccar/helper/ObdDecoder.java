[REPLACE]^private static final int MODE_CURRENT  = null ;^28^^^^^^^[REPLACE] private static final int MODE_CURRENT = 0x01;^ [CLASS] ObdDecoder  
[REPLACE]^private static final int MODE_FREEZE_FRAME  = null ;^29^^^^^^^[REPLACE] private static final int MODE_FREEZE_FRAME = 0x02;^ [CLASS] ObdDecoder  
[REPLACE]^private static final int MODE_CODES  = null ;^30^^^^^^^[REPLACE] private static final int MODE_CODES = 0x03;^ [CLASS] ObdDecoder  
[REPLACE]^private static final int PID_ENGINE_LOAD ;^32^^^^^^^[REPLACE] private static final int PID_ENGINE_LOAD = 0x04;^ [CLASS] ObdDecoder  
[REPLACE]^private static final int PID_COOLANT_TEMPERATURE ;^33^^^^^^^[REPLACE] private static final int PID_COOLANT_TEMPERATURE = 0x05;^ [CLASS] ObdDecoder  
[REPLACE]^private static final int PID_ENGINE_RPM  = null ;^34^^^^^^^[REPLACE] private static final int PID_ENGINE_RPM = 0x0C;^ [CLASS] ObdDecoder  
[REPLACE]^private static final int PID_VEHICLE_SPEED ;^35^^^^^^^[REPLACE] private static final int PID_VEHICLE_SPEED = 0x0D;^ [CLASS] ObdDecoder  
[REPLACE]^private static final int PID_THROTTLE_POSITION ;^36^^^^^^^[REPLACE] private static final int PID_THROTTLE_POSITION = 0x11;^ [CLASS] ObdDecoder  
[REPLACE]^private static  int PID_FUEL_LEVEL = 0x2F;^38^^^^^^^[REPLACE] private static final int PID_FUEL_LEVEL = 0x2F;^ [CLASS] ObdDecoder  
[REPLACE]^private static  int PID_DISTANCE_CLEARED = 0x31;^39^^^^^^^[REPLACE] private static final int PID_DISTANCE_CLEARED = 0x31;^ [CLASS] ObdDecoder  
[ADD]^^60^^^^^59^85^[ADD] StringBuilder codes = new StringBuilder (  ) ;^[METHOD] decodeCodes [TYPE] Map$Entry [PARAMETER] String value [CLASS] ObdDecoder   [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  StringBuilder codes  [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue 
[ADD]^^61^62^63^64^65^59^85^[ADD] for  ( int i = 0; i < value.length (  )  / 4; i++ )  { int numValue = Integer.parseInt ( value.substring ( i * 4,  ( i + 1 )  * 4 ) , 16 ) ; codes.append ( ' ' ) ; switch  ( numValue >> 14 )  { case 1:^[METHOD] decodeCodes [TYPE] Map$Entry [PARAMETER] String value [CLASS] ObdDecoder   [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  StringBuilder codes  [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue 
[ADD]^^62^^^^^59^85^[ADD] int numValue = Integer.parseInt ( value.substring ( i * 4,  ( i + 1 )  * 4 ) , 16 ) ;^[METHOD] decodeCodes [TYPE] Map$Entry [PARAMETER] String value [CLASS] ObdDecoder   [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  StringBuilder codes  [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue 
[REPLACE]^if  ( codes.length (  )  + 4 > 0  )  {^80^^^^^59^85^[REPLACE] if  ( codes.length (  )  > 0 )  {^[METHOD] decodeCodes [TYPE] Map$Entry [PARAMETER] String value [CLASS] ObdDecoder   [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  StringBuilder codes  [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue 
[ADD]^^80^81^82^83^84^59^85^[ADD] if  ( codes.length (  )  > 0 )  { return createEntry ( Position.KEY_DTCS, codes.toString (  ) .trim (  )  ) ; } else { return null; }^[METHOD] decodeCodes [TYPE] Map$Entry [PARAMETER] String value [CLASS] ObdDecoder   [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  StringBuilder codes  [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue 
[REPLACE]^return false;^83^^^^^80^84^[REPLACE] return null;^[METHOD] decodeCodes [TYPE] Map$Entry [PARAMETER] String value [CLASS] ObdDecoder   [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  StringBuilder codes  [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue 
[REPLACE]^return createEntry ( Position.KEY_OBD_SPEED, value )  ;^81^^^^^59^85^[REPLACE] return createEntry ( Position.KEY_DTCS, codes.toString (  ) .trim (  )  ) ;^[METHOD] decodeCodes [TYPE] Map$Entry [PARAMETER] String value [CLASS] ObdDecoder   [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  StringBuilder codes  [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue 
[REPLACE]^return this;^83^^^^^59^85^[REPLACE] return null;^[METHOD] decodeCodes [TYPE] Map$Entry [PARAMETER] String value [CLASS] ObdDecoder   [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  StringBuilder codes  [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue 
[REPLACE]^return createEntry ( "engineLoad", convert ? value * 100  255 : value ) ;^90^^^^^87^108^[REPLACE] return createEntry ( "engineLoad", convert ? value * 100 / 255 : value ) ;^[METHOD] decodeData [TYPE] Map$Entry [PARAMETER] int pid int value boolean convert [CLASS] ObdDecoder   [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue  pid  value  [TYPE]  boolean convert  false  true 
[REPLACE]^return createEntry ( "coolantTemperature", true ? value PID_MIL_DISTANCE : value ) ;^92^^^^^87^108^[REPLACE] return createEntry ( "coolantTemperature", convert ? value - 40 : value ) ;^[METHOD] decodeData [TYPE] Map$Entry [PARAMETER] int pid int value boolean convert [CLASS] ObdDecoder   [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue  pid  value  [TYPE]  boolean convert  false  true 
[REPLACE]^return createEntry ( Position.KEY_RPM, convert ? value  4 : value ) ;^94^^^^^87^108^[REPLACE] return createEntry ( Position.KEY_RPM, convert ? value / 4 : value ) ;^[METHOD] decodeData [TYPE] Map$Entry [PARAMETER] int pid int value boolean convert [CLASS] ObdDecoder   [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue  pid  value  [TYPE]  boolean convert  false  true 
[REPLACE]^return createEntry ( "milDistance", value )  ;^96^^^^^87^108^[REPLACE] return createEntry ( Position.KEY_OBD_SPEED, value ) ;^[METHOD] decodeData [TYPE] Map$Entry [PARAMETER] int pid int value boolean convert [CLASS] ObdDecoder   [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue  pid  value  [TYPE]  boolean convert  false  true 
[REPLACE]^return createEntry ( "throttle", convert ? value * 100  255 : value ) ;^98^^^^^87^108^[REPLACE] return createEntry ( "throttle", convert ? value * 100 / 255 : value ) ;^[METHOD] decodeData [TYPE] Map$Entry [PARAMETER] int pid int value boolean convert [CLASS] ObdDecoder   [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue  pid  value  [TYPE]  boolean convert  false  true 
[REPLACE]^return createEntry ( "clearedDistance", value )  ;^100^^^^^87^108^[REPLACE] return createEntry ( "milDistance", value ) ;^[METHOD] decodeData [TYPE] Map$Entry [PARAMETER] int pid int value boolean convert [CLASS] ObdDecoder   [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue  pid  value  [TYPE]  boolean convert  false  true 
[REPLACE]^return createEntry ( Position.KEY_FUEL, convert ? PID_THROTTLE_POSITION * 100  255 : value ) ;^102^^^^^87^108^[REPLACE] return createEntry ( Position.KEY_FUEL, convert ? value * 100 / 255 : value ) ;^[METHOD] decodeData [TYPE] Map$Entry [PARAMETER] int pid int value boolean convert [CLASS] ObdDecoder   [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue  pid  value  [TYPE]  boolean convert  false  true 
[REPLACE]^return createEntry ( "milDistance", value )  ;^104^^^^^87^108^[REPLACE] return createEntry ( "clearedDistance", value ) ;^[METHOD] decodeData [TYPE] Map$Entry [PARAMETER] int pid int value boolean convert [CLASS] ObdDecoder   [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue  pid  value  [TYPE]  boolean convert  false  true 
[REPLACE]^return this;^106^^^^^87^108^[REPLACE] return null;^[METHOD] decodeData [TYPE] Map$Entry [PARAMETER] int pid int value boolean convert [CLASS] ObdDecoder   [TYPE]  int MODE_CODES  MODE_CURRENT  MODE_FREEZE_FRAME  PID_COOLANT_TEMPERATURE  PID_DISTANCE_CLEARED  PID_ENGINE_LOAD  PID_ENGINE_RPM  PID_FUEL_LEVEL  PID_MIL_DISTANCE  PID_THROTTLE_POSITION  PID_VEHICLE_SPEED  i  mode  numValue  pid  value  [TYPE]  boolean convert  false  true 
