[REPLACE]^addWindowListener ( this ) ;^65^^^^^64^67^[REPLACE] super ( title ) ;^[METHOD] <init> [TYPE] String) [PARAMETER] String title [CLASS] ApplicationFrame   [TYPE]  boolean false  true  [TYPE]  String title 
[REPLACE]^super ( title ) ;^66^^^^^64^67^[REPLACE] addWindowListener ( this ) ;^[METHOD] <init> [TYPE] String) [PARAMETER] String title [CLASS] ApplicationFrame   [TYPE]  boolean false  true  [TYPE]  String title 
[REPLACE]^if  ( event.getWindow (  )   ||  this )  {^75^^^^^74^79^[REPLACE] if  ( event.getWindow (  )  == this )  {^[METHOD] windowClosing [TYPE] void [PARAMETER] WindowEvent event [CLASS] ApplicationFrame   [TYPE]  boolean false  true  [TYPE]  WindowEvent event 
[ADD]^^75^76^77^78^^74^79^[ADD] if  ( event.getWindow (  )  == this )  { dispose (  ) ; System.exit ( 0 ) ; }^[METHOD] windowClosing [TYPE] void [PARAMETER] WindowEvent event [CLASS] ApplicationFrame   [TYPE]  boolean false  true  [TYPE]  WindowEvent event 
[REPLACE]^addWindowListener ( this ) ;^76^^^^^74^79^[REPLACE] dispose (  ) ;^[METHOD] windowClosing [TYPE] void [PARAMETER] WindowEvent event [CLASS] ApplicationFrame   [TYPE]  boolean false  true  [TYPE]  WindowEvent event 
[ADD]^^76^^^^^74^79^[ADD] dispose (  ) ;^[METHOD] windowClosing [TYPE] void [PARAMETER] WindowEvent event [CLASS] ApplicationFrame   [TYPE]  boolean false  true  [TYPE]  WindowEvent event 
[REPLACE]^3   ;^77^^^^^74^79^[REPLACE] System.exit ( 0 ) ;^[METHOD] windowClosing [TYPE] void [PARAMETER] WindowEvent event [CLASS] ApplicationFrame   [TYPE]  boolean false  true  [TYPE]  WindowEvent event 