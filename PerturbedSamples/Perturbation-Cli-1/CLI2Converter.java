[REPLACE]^final GroupBuilder gbuilder = new GroupBuilder (  ) ;^50^^^^^48^94^[REPLACE] final DefaultOptionBuilder obuilder = new DefaultOptionBuilder (  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^obuilder.withRequired ( option1.getId (  )  ) ;^51^^^^^48^94^[REPLACE] obuilder.withRequired ( option1.isRequired (  )  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^final String shortName = option1.getType (  ) ;^53^^^^^48^94^[REPLACE] final String shortName = option1.getOpt (  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[ADD]^^53^^^^^48^94^[ADD] final String shortName = option1.getOpt (  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^if ( shortName==null && !" ".equals ( shortName )  ) {^54^^^^^48^94^[REPLACE] if ( shortName!=null && !" ".equals ( shortName )  ) {^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^obuilder.withShortName ( longName ) ;^55^^^^^48^94^[REPLACE] obuilder.withShortName ( shortName ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^final String longName = option1.getOpt (  ) ;^58^^^^^48^94^[REPLACE] final String longName = option1.getLongOpt (  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^if ( shortName==null ) {^59^^^^^48^94^[REPLACE] if ( longName!=null ) {^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[ADD]^obuilder.withLongName ( longName ) ;^59^60^61^^^48^94^[ADD] if ( longName!=null ) { obuilder.withLongName ( longName ) ; }^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^obuilder.withLongName ( shortName ) ;^60^^^^^48^94^[REPLACE] obuilder.withLongName ( longName ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^obuilder.withId ( option1 .getOpt (  )   ) ;^62^^^^^48^94^[REPLACE] obuilder.withId ( option1.getId (  )  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^obuilder.withId ( option1.getOpt (  )  ) ;^62^^^^^48^94^[REPLACE] obuilder.withId ( option1.getId (  )  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^final String description = option1.getOpt (  ) ;^64^^^^^48^94^[REPLACE] final String description = option1.getDescription (  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^if ( description==null ) {^65^^^^^48^94^[REPLACE] if ( description!=null ) {^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^obuilder.withShortName ( description ) ;^66^^^^^48^94^[REPLACE] obuilder.withDescription ( description ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[ADD]^^66^^^^^48^94^[ADD] obuilder.withDescription ( description ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^if ( ! option1.hasArg (  )  ) {^69^^^^^48^94^[REPLACE] if ( option1.hasArg (  )  ) {^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^if ( ! option1.hasValueSeparator (  )  ) {^74^^^^^48^94^[REPLACE] if ( option1.hasValueSeparator (  )  ) {^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[ADD]^^74^75^76^^^48^94^[ADD] if ( option1.hasValueSeparator (  )  ) { abuilder.withSubsequentSeparator ( option1.getValueSeparator (  )  ) ; }^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^abuilder .withValidator ( this )  ;^75^^^^^48^94^[REPLACE] abuilder.withSubsequentSeparator ( option1.getValueSeparator (  )  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^abuilder.withSubsequentSeparator ( option1 .hasValueSeparator (  )   ) ;^75^^^^^48^94^[REPLACE] abuilder.withSubsequentSeparator ( option1.getValueSeparator (  )  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^if ( ! option1.hasOptionalArg (  )  ) {^77^^^^^48^94^[REPLACE] if ( option1.hasOptionalArg (  )  ) {^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^abuilder.withMaximum ( option1.getArgs (  )  ) ;^82^^^^^77^83^[REPLACE] abuilder.withMinimum ( option1.getArgs (  )  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^abuilder.withMinimum ( option1 .getArgName (  )   ) ;^82^^^^^77^83^[REPLACE] abuilder.withMinimum ( option1.getArgs (  )  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^abuilder.withMaximum ( 0 - 3 ) ;^78^^^^^48^94^[REPLACE] abuilder.withMinimum ( 0 ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^if ( type==null ) {^86^^^^^48^94^[REPLACE] if ( type!=null ) {^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^abuilder.withName ( new TypeHandlerValidator ( type )  ) ;^87^^^^^48^94^[REPLACE] abuilder.withValidator ( new TypeHandlerValidator ( type )  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^final GroupBuilder gbuilder = new GroupBuilder (  ) ;^70^^^^^48^94^[REPLACE] final ArgumentBuilder abuilder = new ArgumentBuilder (  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[ADD]^^70^^^^^48^94^[ADD] final ArgumentBuilder abuilder = new ArgumentBuilder (  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^final String argName = option1.getArgs (  ) ;^71^^^^^48^94^[REPLACE] final String argName = option1.getArgName (  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[ADD]^^71^^^^^48^94^[ADD] final String argName = option1.getArgName (  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^final Object type = option1.getOpt (  ) ;^85^^^^^48^94^[REPLACE] final Object type = option1.getType (  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^abuilder.withSubsequentSeparator ( option1.hasValueSeparator (  )  ) ;^75^^^^^48^94^[REPLACE] abuilder.withSubsequentSeparator ( option1.getValueSeparator (  )  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[ADD]^abuilder.withMinimum ( 0 ) ;^77^78^79^^^48^94^[ADD] if ( option1.hasOptionalArg (  )  ) { abuilder.withMinimum ( 0 ) ; }^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^abuilder.withMinimum ( option1.getArgName (  )  ) ;^82^^^^^77^83^[REPLACE] abuilder.withMinimum ( option1.getArgs (  )  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REMOVE]^obuilder.withRequired ( option1.isRequired (  )  ) ;^82^^^^^77^83^[REMOVE] ^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^abuilder.withMinimum ( 3 ) ;^78^^^^^48^94^[REPLACE] abuilder.withMinimum ( 0 ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^if ( type==false ) {^86^^^^^48^94^[REPLACE] if ( type!=null ) {^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^return obuilder.withArgument (  ) ;^93^^^^^48^94^[REPLACE] return obuilder.create (  ) ;^[METHOD] option [TYPE] Option [PARAMETER] Option option1 [CLASS] CLI2Converter   [TYPE]  boolean false  true  [TYPE]  ArgumentBuilder abuilder  [TYPE]  Option option1  [TYPE]  DefaultOptionBuilder obuilder  [TYPE]  Object type  [TYPE]  String argName  description  longName  shortName 
[REPLACE]^final Set optionGroups = new HashSet (  ) ;^104^^^^^102^119^[REPLACE] final GroupBuilder gbuilder = new GroupBuilder (  ) ;^[METHOD] group [TYPE] Group [PARAMETER] OptionGroup optionGroup1 [CLASS] CLI2Converter   [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^for ( final Iterator i = options1.getOptions (  ) .iterator (  ) ;i.hasNext (  ) ; ) {^106^^^^^102^119^[REPLACE] for ( final Iterator i = optionGroup1.getOptions (  ) .iterator (  ) ;i.hasNext (  ) ; ) {^[METHOD] group [TYPE] Group [PARAMETER] OptionGroup optionGroup1 [CLASS] CLI2Converter   [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[ADD]^^106^107^108^109^110^102^119^[ADD] for ( final Iterator i = optionGroup1.getOptions (  ) .iterator (  ) ;i.hasNext (  ) ; ) { final org.apache.commons.cli.Option option1 =  ( org.apache.commons.cli.Option ) i.next (  ) ; final Option option2 = option ( option1 ) ; gbuilder.withOption ( option2 ) ; }^[METHOD] group [TYPE] Group [PARAMETER] OptionGroup optionGroup1 [CLASS] CLI2Converter   [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^final org.apache.commons.cli.Option option1 =  ( org.apache.commons.cli.Option ) i.hasNext (  ) ;^107^^^^^102^119^[REPLACE] final org.apache.commons.cli.Option option1 =  ( org.apache.commons.cli.Option ) i.next (  ) ;^[METHOD] group [TYPE] Group [PARAMETER] OptionGroup optionGroup1 [CLASS] CLI2Converter   [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^final Option option2 = option ( option2 ) ;^108^^^^^102^119^[REPLACE] final Option option2 = option ( option1 ) ;^[METHOD] group [TYPE] Group [PARAMETER] OptionGroup optionGroup1 [CLASS] CLI2Converter   [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^for ( final Iterator i = optionGroup1 .isRequired (  )  .iterator (  ) ;i.hasNext (  ) ; ) {^106^^^^^102^119^[REPLACE] for ( final Iterator i = optionGroup1.getOptions (  ) .iterator (  ) ;i.hasNext (  ) ; ) {^[METHOD] group [TYPE] Group [PARAMETER] OptionGroup optionGroup1 [CLASS] CLI2Converter   [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^gbuilder.withMaximum ( 2 ) ;^112^^^^^102^119^[REPLACE] gbuilder.withMaximum ( 1 ) ;^[METHOD] group [TYPE] Group [PARAMETER] OptionGroup optionGroup1 [CLASS] CLI2Converter   [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^if ( optionGroup1 % 4.isRequired (  )  ) {^114^^^^^102^119^[REPLACE] if ( optionGroup1.isRequired (  )  ) {^[METHOD] group [TYPE] Group [PARAMETER] OptionGroup optionGroup1 [CLASS] CLI2Converter   [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^gbuilder.withMinimum ( 2 ) ;^115^^^^^102^119^[REPLACE] gbuilder.withMinimum ( 1 ) ;^[METHOD] group [TYPE] Group [PARAMETER] OptionGroup optionGroup1 [CLASS] CLI2Converter   [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[ADD]^^115^^^^^102^119^[ADD] gbuilder.withMinimum ( 1 ) ;^[METHOD] group [TYPE] Group [PARAMETER] OptionGroup optionGroup1 [CLASS] CLI2Converter   [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^return gbuilder.GroupBuilder (  ) ;^118^^^^^102^119^[REPLACE] return gbuilder.create (  ) ;^[METHOD] group [TYPE] Group [PARAMETER] OptionGroup optionGroup1 [CLASS] CLI2Converter   [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^final Set optionGroups = new HashSet (  ) ;^129^^^^^127^149^[REPLACE] final GroupBuilder gbuilder = new GroupBuilder (  ) ;^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[ADD]^^129^^^^^127^149^[ADD] final GroupBuilder gbuilder = new GroupBuilder (  ) ;^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^final GroupBuilder gbuilder = new GroupBuilder (  ) ;^131^^^^^127^149^[REPLACE] final Set optionGroups = new HashSet (  ) ;^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^for ( final Iterator i = options1.getOptions (  ) .iterator (  ) ;i.hasNext (  ) ; ) {^133^^^^^127^149^[REPLACE] for ( final Iterator i = options1.getOptionGroups (  ) .iterator (  ) ;i.hasNext (  ) ; ) {^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^final OptionGroup optionGroup1 =  ( OptionGroup ) i .hasNext (  )  ;^134^^^^^127^149^[REPLACE] final OptionGroup optionGroup1 =  ( OptionGroup ) i.next (  ) ;^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^final String shortName = option1.getOpt (  ) ;^135^^^^^127^149^[REPLACE] Group group = group ( optionGroup1 ) ;^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^for ( final Iterator i = options1 .getOptions (  )  .iterator (  ) ;i.hasNext (  ) ; ) {^133^^^^^127^149^[REPLACE] for ( final Iterator i = options1.getOptionGroups (  ) .iterator (  ) ;i.hasNext (  ) ; ) {^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^if ( optionInAGroup ( option1,optionGroups )  ) {^142^^^^^127^149^[REPLACE] if ( !optionInAGroup ( option1,optionGroups )  ) {^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^final Option option2 = option ( option2 ) ;^143^^^^^127^149^[REPLACE] final Option option2 = option ( option1 ) ;^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[ADD]^^143^144^^^^127^149^[ADD] final Option option2 = option ( option1 ) ; gbuilder.withOption ( option2 ) ;^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^for ( final Iterator i = options1.getOptions (  ) .contains (  ) ;i.hasNext (  ) ; ) {^140^^^^^127^149^[REPLACE] for ( final Iterator i = options1.getOptions (  ) .iterator (  ) ;i.hasNext (  ) ; ) {^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[ADD]^final Option option2 = option ( option1 ) ;gbuilder.withOption ( option2 ) ;^142^143^144^145^^127^149^[ADD] if ( !optionInAGroup ( option1,optionGroups )  ) { final Option option2 = option ( option1 ) ; gbuilder.withOption ( option2 ) ; }^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^final org.apache.commons.cli.Option option1 =  ( org.apache.commons.cli.Option ) i.hasNext (  ) ;^141^^^^^127^149^[REPLACE] final org.apache.commons.cli.Option option1 =  ( org.apache.commons.cli.Option ) i.next (  ) ;^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^for ( final Iterator i = options1.getOptionGroups (  ) .iterator (  ) ;i.hasNext (  ) ; ) {^140^^^^^127^149^[REPLACE] for ( final Iterator i = options1.getOptions (  ) .iterator (  ) ;i.hasNext (  ) ; ) {^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^return gbuilder.GroupBuilder (  ) ;^148^^^^^127^149^[REPLACE] return gbuilder.create (  ) ;^[METHOD] group [TYPE] Group [PARAMETER] Options options1 [CLASS] CLI2Converter   [TYPE]  Options options1  [TYPE]  Group group  [TYPE]  Set optionGroups  [TYPE]  OptionGroup optionGroup1  [TYPE]  boolean false  true  [TYPE]  GroupBuilder gbuilder  [TYPE]  Iterator i  [TYPE]  Option option1  option2 
[REPLACE]^if ( group .isRequired (  )  .contains ( option1 )  ) {^154^^^^^151^159^[REPLACE] if ( group.getOptions (  ) .contains ( option1 )  ) {^[METHOD] optionInAGroup [TYPE] boolean [PARAMETER] Option option1 Set optionGroups [CLASS] CLI2Converter   [TYPE]  Set optionGroups  [TYPE]  OptionGroup group  [TYPE]  boolean false  true  [TYPE]  Iterator i  [TYPE]  Option option1 
[REPLACE]^return false;^155^^^^^151^159^[REPLACE] return true;^[METHOD] optionInAGroup [TYPE] boolean [PARAMETER] Option option1 Set optionGroups [CLASS] CLI2Converter   [TYPE]  Set optionGroups  [TYPE]  OptionGroup group  [TYPE]  boolean false  true  [TYPE]  Iterator i  [TYPE]  Option option1 
[REPLACE]^for  ( Iterator i = optionGroups .contains ( null )  ; i.hasNext (  ) ; )  {^152^^^^^151^159^[REPLACE] for  ( Iterator i = optionGroups.iterator (  ) ; i.hasNext (  ) ; )  {^[METHOD] optionInAGroup [TYPE] boolean [PARAMETER] Option option1 Set optionGroups [CLASS] CLI2Converter   [TYPE]  Set optionGroups  [TYPE]  OptionGroup group  [TYPE]  boolean false  true  [TYPE]  Iterator i  [TYPE]  Option option1 
[ADD]^^152^153^154^155^156^151^159^[ADD] for  ( Iterator i = optionGroups.iterator (  ) ; i.hasNext (  ) ; )  { OptionGroup group =  ( OptionGroup )  i.next (  ) ; if ( group.getOptions (  ) .contains ( option1 )  ) { return true; }^[METHOD] optionInAGroup [TYPE] boolean [PARAMETER] Option option1 Set optionGroups [CLASS] CLI2Converter   [TYPE]  Set optionGroups  [TYPE]  OptionGroup group  [TYPE]  boolean false  true  [TYPE]  Iterator i  [TYPE]  Option option1 
[REPLACE]^if ( ! group.getOptions (  ) .contains ( option1 )  ) {^154^^^^^151^159^[REPLACE] if ( group.getOptions (  ) .contains ( option1 )  ) {^[METHOD] optionInAGroup [TYPE] boolean [PARAMETER] Option option1 Set optionGroups [CLASS] CLI2Converter   [TYPE]  Set optionGroups  [TYPE]  OptionGroup group  [TYPE]  boolean false  true  [TYPE]  Iterator i  [TYPE]  Option option1 
[REPLACE]^OptionGroup group =  ( OptionGroup )  i.hasNext (  ) ;^153^^^^^151^159^[REPLACE] OptionGroup group =  ( OptionGroup )  i.next (  ) ;^[METHOD] optionInAGroup [TYPE] boolean [PARAMETER] Option option1 Set optionGroups [CLASS] CLI2Converter   [TYPE]  Set optionGroups  [TYPE]  OptionGroup group  [TYPE]  boolean false  true  [TYPE]  Iterator i  [TYPE]  Option option1 
[REPLACE]^for  ( Iterator i = optionGroups .contains ( true )  ; i.hasNext (  ) ; )  {^152^^^^^151^159^[REPLACE] for  ( Iterator i = optionGroups.iterator (  ) ; i.hasNext (  ) ; )  {^[METHOD] optionInAGroup [TYPE] boolean [PARAMETER] Option option1 Set optionGroups [CLASS] CLI2Converter   [TYPE]  Set optionGroups  [TYPE]  OptionGroup group  [TYPE]  boolean false  true  [TYPE]  Iterator i  [TYPE]  Option option1 
[REPLACE]^return true;^158^^^^^151^159^[REPLACE] return false;^[METHOD] optionInAGroup [TYPE] boolean [PARAMETER] Option option1 Set optionGroups [CLASS] CLI2Converter   [TYPE]  Set optionGroups  [TYPE]  OptionGroup group  [TYPE]  boolean false  true  [TYPE]  Iterator i  [TYPE]  Option option1 
[REPLACE]^private  Object type;^164^^^^^^^[REPLACE] private final Object type;^[METHOD] optionInAGroup [TYPE] boolean [PARAMETER] Option option1 Set optionGroups [CLASS] TypeHandlerValidator   [TYPE]  Set optionGroups  [TYPE]  OptionGroup group  [TYPE]  boolean false  true  [TYPE]  Iterator i  [TYPE]  Option option1 
[REPLACE]^this.type =  null;^173^^^^^172^174^[REPLACE] this.type = type;^[METHOD] <init> [TYPE] Object) [PARAMETER] Object type [CLASS] TypeHandlerValidator   [TYPE]  Object type  [TYPE]  boolean false  true 
[REPLACE]^final Object converted = TypeHandler.createValue ( value,type ) ;^180^^^^^179^189^[REPLACE] final ListIterator i = values.listIterator (  ) ;^[METHOD] validate [TYPE] void [PARAMETER] List values [CLASS] TypeHandlerValidator   [TYPE]  Object converted  type  [TYPE]  List values  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  ListIterator i 
[ADD]^^180^^^^^179^189^[ADD] final ListIterator i = values.listIterator (  ) ;^[METHOD] validate [TYPE] void [PARAMETER] List values [CLASS] TypeHandlerValidator   [TYPE]  Object converted  type  [TYPE]  List values  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  ListIterator i 
[REPLACE]^if ( converted!=null ) {^184^^^^^179^189^[REPLACE] if ( converted==null ) {^[METHOD] validate [TYPE] void [PARAMETER] List values [CLASS] TypeHandlerValidator   [TYPE]  Object converted  type  [TYPE]  List values  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  ListIterator i 
[ADD]^^184^185^186^^^179^189^[ADD] if ( converted==null ) { throw new InvalidArgumentException  (" ")  ; }^[METHOD] validate [TYPE] void [PARAMETER] List values [CLASS] TypeHandlerValidator   [TYPE]  Object converted  type  [TYPE]  List values  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  ListIterator i 
[REPLACE]^return ;^185^^^^^179^189^[REPLACE] throw new InvalidArgumentException  (" ")  ;^[METHOD] validate [TYPE] void [PARAMETER] List values [CLASS] TypeHandlerValidator   [TYPE]  Object converted  type  [TYPE]  List values  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  ListIterator i 
[REPLACE]^final String value =  ( String ) i .hasNext (  )  ;^182^^^^^179^189^[REPLACE] final String value =  ( String ) i.next (  ) ;^[METHOD] validate [TYPE] void [PARAMETER] List values [CLASS] TypeHandlerValidator   [TYPE]  Object converted  type  [TYPE]  List values  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  ListIterator i 
[REPLACE]^final ListIterator i = values.listIterator (  ) ;^183^^^^^179^189^[REPLACE] final Object converted = TypeHandler.createValue ( value,type ) ;^[METHOD] validate [TYPE] void [PARAMETER] List values [CLASS] TypeHandlerValidator   [TYPE]  Object converted  type  [TYPE]  List values  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  ListIterator i 
[REPLACE]^if ( converted!=this ) {^184^^^^^179^189^[REPLACE] if ( converted==null ) {^[METHOD] validate [TYPE] void [PARAMETER] List values [CLASS] TypeHandlerValidator   [TYPE]  Object converted  type  [TYPE]  List values  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  ListIterator i 
[REPLACE]^final String value =  ( String ) i.hasNext (  ) ;^182^^^^^179^189^[REPLACE] final String value =  ( String ) i.next (  ) ;^[METHOD] validate [TYPE] void [PARAMETER] List values [CLASS] TypeHandlerValidator   [TYPE]  Object converted  type  [TYPE]  List values  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  ListIterator i 
[ADD]^^185^^^^^179^189^[ADD] throw new InvalidArgumentException  (" ")  ;^[METHOD] validate [TYPE] void [PARAMETER] List values [CLASS] TypeHandlerValidator   [TYPE]  Object converted  type  [TYPE]  List values  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  ListIterator i 