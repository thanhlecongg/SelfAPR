[REPLACE]^private final boolean success;^25^^^^^^^[REPLACE] public final boolean success;^ [CLASS] Result  
[REPLACE]^private final JSError[] warnings;^27^^^^^^^[REPLACE] public final JSError[] warnings;^ [CLASS] Result  
[REPLACE]^private final  boolean  debugLog;^28^^^^^^^[REPLACE] public final String debugLog;^ [CLASS] Result  
[REPLACE]^public  VariableMap variableMap;^29^^^^^^^[REPLACE] public final VariableMap variableMap;^ [CLASS] Result  
[REPLACE]^public  VariableMap propertyMap;^30^^^^^^^[REPLACE] public final VariableMap propertyMap;^ [CLASS] Result  
[REPLACE]^public  VariableMap namedAnonFunctionMap;^31^^^^^^^[REPLACE] public final VariableMap namedAnonFunctionMap;^ [CLASS] Result  
[REPLACE]^public  FunctionInformationMap functionInformationMap;^32^^^^^^^[REPLACE] public final FunctionInformationMap functionInformationMap;^ [CLASS] Result  
[REPLACE]^public  SourceMap sourceMap;^33^^^^^^^[REPLACE] public final SourceMap sourceMap;^ [CLASS] Result  
[REPLACE]^public  Map<String, Integer> cssNames;^34^^^^^^^[REPLACE] public final Map<String, Integer> cssNames;^ [CLASS] Result  
[REPLACE]^public final  int  externExport;^35^^^^^^^[REPLACE] public final String externExport;^ [CLASS] Result  
[REPLACE]^this.success = errors.length /  0.5  == 0;^43^^^^^37^54^[REPLACE] this.success = errors.length == 0;^[METHOD] <init> [TYPE] Map) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport Integer> cssNames [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[REPLACE]^this.errors =  warnings;^44^^^^^37^54^[REPLACE] this.errors = errors;^[METHOD] <init> [TYPE] Map) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport Integer> cssNames [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[REPLACE]^this.warnings =  errors;^45^^^^^37^54^[REPLACE] this.warnings = warnings;^[METHOD] <init> [TYPE] Map) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport Integer> cssNames [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[REPLACE]^this.debugLog =  externExport;^46^^^^^37^54^[REPLACE] this.debugLog = debugLog;^[METHOD] <init> [TYPE] Map) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport Integer> cssNames [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[REPLACE]^this.variableMap =  propertyMap;^47^^^^^37^54^[REPLACE] this.variableMap = variableMap;^[METHOD] <init> [TYPE] Map) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport Integer> cssNames [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[REPLACE]^this.propertyMap =  variableMap;^48^^^^^37^54^[REPLACE] this.propertyMap = propertyMap;^[METHOD] <init> [TYPE] Map) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport Integer> cssNames [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[REPLACE]^this.namedAnonFunctionMap =  null;^49^^^^^37^54^[REPLACE] this.namedAnonFunctionMap = namedAnonFunctionMap;^[METHOD] <init> [TYPE] Map) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport Integer> cssNames [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[REPLACE]^this.functionInformationMap =  null;^50^^^^^37^54^[REPLACE] this.functionInformationMap = functionInformationMap;^[METHOD] <init> [TYPE] Map) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport Integer> cssNames [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[REPLACE]^this.sourceMap =  null;^51^^^^^37^54^[REPLACE] this.sourceMap = sourceMap;^[METHOD] <init> [TYPE] Map) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport Integer> cssNames [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[REPLACE]^this.externExport =  debugLog;^52^^^^^37^54^[REPLACE] this.externExport = externExport;^[METHOD] <init> [TYPE] Map) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport Integer> cssNames [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[ADD]^^52^^^^^37^54^[ADD] this.externExport = externExport;^[METHOD] <init> [TYPE] Map) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport Integer> cssNames [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[REPLACE]^this.cssNames =  this;^53^^^^^37^54^[REPLACE] this.cssNames = cssNames;^[METHOD] <init> [TYPE] Map) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport Integer> cssNames [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[REPLACE]^this ( warnings, warnings, debugLog, variableMap, propertyMap, namedAnonFunctionMap, functionInformationMap, sourceMap, externExport, true ) ;^62^63^64^^^57^65^[REPLACE] this ( errors, warnings, debugLog, variableMap, propertyMap, namedAnonFunctionMap, functionInformationMap, sourceMap, externExport, null ) ;^[METHOD] <init> [TYPE] String) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 
[ADD]^^62^63^64^^^57^65^[ADD] this ( errors, warnings, debugLog, variableMap, propertyMap, namedAnonFunctionMap, functionInformationMap, sourceMap, externExport, null ) ;^[METHOD] <init> [TYPE] String) [PARAMETER] JSError[] errors JSError[] warnings String debugLog VariableMap variableMap VariableMap propertyMap VariableMap namedAnonFunctionMap FunctionInformationMap functionInformationMap SourceMap sourceMap String externExport [CLASS] Result   [TYPE]  FunctionInformationMap functionInformationMap  [TYPE]  boolean false  success  true  [TYPE]  VariableMap namedAnonFunctionMap  propertyMap  variableMap  [TYPE]  SourceMap sourceMap  [TYPE]  JSError[] errors  warnings  [TYPE]  String debugLog  externExport  [TYPE]  Map cssNames 