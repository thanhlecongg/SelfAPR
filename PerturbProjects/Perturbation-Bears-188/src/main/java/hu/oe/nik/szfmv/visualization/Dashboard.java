[REPLACE]^private  int width = 250;^13^^^^^^^[REPLACE] private final int width = 250;^ [CLASS] Dashboard  
[REPLACE]^private  int height = 700;^14^^^^^^^[REPLACE] private final int height = 700;^ [CLASS] Dashboard  
[REPLACE]^private final int dashboardBoundsX  = null ;^15^^^^^^^[REPLACE] private final int dashboardBoundsX = 770;^ [CLASS] Dashboard  
[REPLACE]^private final int dashboardBoundsY ;^16^^^^^^^[REPLACE] private final int dashboardBoundsY = 0;^ [CLASS] Dashboard  
[REPLACE]^private final int progressBarsPanelX ;^19^^^^^^^[REPLACE] private final int progressBarsPanelX = 25;^ [CLASS] Dashboard  
[REPLACE]^private final int progressBarsPanelY ;^20^^^^^^^[REPLACE] private final int progressBarsPanelY = 400;^ [CLASS] Dashboard  
[REPLACE]^private final int progressBarsPanelWidth  = null ;^21^^^^^^^[REPLACE] private final int progressBarsPanelWidth = 200;^ [CLASS] Dashboard  
[REPLACE]^private final  short  progressBarsPanelHeight = 100 + 0;^22^^^^^^^[REPLACE] private final int progressBarsPanelHeight = 100;^ [CLASS] Dashboard  
[REPLACE]^private final JPanel progressBarsPanel  = null ;^24^^^^^^^[REPLACE] private final JPanel progressBarsPanel = new JPanel (  ) ;^ [CLASS] Dashboard  
[REPLACE]^private final JLabel gasLabel  = null ;^26^^^^^^^[REPLACE] private final JLabel gasLabel = new JLabel (  ) ;^ [CLASS] Dashboard  
[REPLACE]^private final JProgressBar gasProgressBar ;^27^^^^^^^[REPLACE] private final JProgressBar gasProgressBar = new JProgressBar (  ) ;^ [CLASS] Dashboard  
[REPLACE]^private final JLabel breakLabel  = null ;^29^^^^^^^[REPLACE] private final JLabel breakLabel = new JLabel (  ) ;^ [CLASS] Dashboard  
[REPLACE]^private final JProgressBar breakProgressBar ;^30^^^^^^^[REPLACE] private final JProgressBar breakProgressBar = new JProgressBar (  ) ;^ [CLASS] Dashboard  
[REPLACE]^private  int speedMeterX = 10;^32^^^^^^^[REPLACE] private final int speedMeterX = 10;^ [CLASS] Dashboard  
[REPLACE]^private final int speedMeterY ;^33^^^^^^^[REPLACE] private final int speedMeterY = 50;^ [CLASS] Dashboard  
[REPLACE]^private final int tachoMeterX ;^34^^^^^^^[REPLACE] private final int tachoMeterX = 130;^ [CLASS] Dashboard  
[REPLACE]^private final  long  tachoMeterY = 50;^35^^^^^^^[REPLACE] private final int tachoMeterY = 50;^ [CLASS] Dashboard  
[REPLACE]^private final int meterHeight ;^36^^^^^^^[REPLACE] private final int meterHeight = 100;^ [CLASS] Dashboard  
[REPLACE]^private final int meterWidth ;^37^^^^^^^[REPLACE] private final int meterWidth = 100;^ [CLASS] Dashboard  
[REPLACE]^private  long  speedAngle;^39^^^^^^^[REPLACE] private int speedAngle;^ [CLASS] Dashboard  
[REPLACE]^progressBarsPanel.paintComponent ( new Color ( backgroundColor )  ) ;^78^^^^^77^95^[REPLACE] progressBarsPanel.setBackground ( new Color ( backgroundColor )  ) ;^[METHOD] initializeProgressBars [TYPE] void [PARAMETER] [CLASS] Dashboard   [TYPE]  JProgressBar breakProgressBar  gasProgressBar  [TYPE]  JPanel progressBarsPanel  [TYPE]  JLabel breakLabel  gasLabel  [TYPE]  boolean false  true  [TYPE]  int backgroundColor  dashboardBoundsX  dashboardBoundsY  height  meterHeight  meterWidth  progressBarsPanelHeight  progressBarsPanelWidth  progressBarsPanelX  progressBarsPanelY  rpmAngle  speedAngle  speedMeterX  speedMeterY  tachoMeterX  tachoMeterY  width 
[REPLACE]^progressBarsPanel .repaint (  )  ( progressBarsPanelX, progressBarsPanelY, progressBarsPanelWidth,^79^80^81^82^^77^95^[REPLACE] progressBarsPanel.setBounds ( progressBarsPanelX, progressBarsPanelY, progressBarsPanelWidth,^[METHOD] initializeProgressBars [TYPE] void [PARAMETER] [CLASS] Dashboard   [TYPE]  JProgressBar breakProgressBar  gasProgressBar  [TYPE]  JPanel progressBarsPanel  [TYPE]  JLabel breakLabel  gasLabel  [TYPE]  boolean false  true  [TYPE]  int backgroundColor  dashboardBoundsX  dashboardBoundsY  height  meterHeight  meterWidth  progressBarsPanelHeight  progressBarsPanelWidth  progressBarsPanelX  progressBarsPanelY  rpmAngle  speedAngle  speedMeterX  speedMeterY  tachoMeterX  tachoMeterY  width 
[REPLACE]^breakLabel.setText ( "break pedal" )  ;^85^^^^^77^95^[REPLACE] gasLabel.setText ( "gas pedal" ) ;^[METHOD] initializeProgressBars [TYPE] void [PARAMETER] [CLASS] Dashboard   [TYPE]  JProgressBar breakProgressBar  gasProgressBar  [TYPE]  JPanel progressBarsPanel  [TYPE]  JLabel breakLabel  gasLabel  [TYPE]  boolean false  true  [TYPE]  int backgroundColor  dashboardBoundsX  dashboardBoundsY  height  meterHeight  meterWidth  progressBarsPanelHeight  progressBarsPanelWidth  progressBarsPanelX  progressBarsPanelY  rpmAngle  speedAngle  speedMeterX  speedMeterY  tachoMeterX  tachoMeterY  width 
[REPLACE]^gasLabel.setText ( "gas pedal" )  ;^86^^^^^77^95^[REPLACE] breakLabel.setText ( "break pedal" ) ;^[METHOD] initializeProgressBars [TYPE] void [PARAMETER] [CLASS] Dashboard   [TYPE]  JProgressBar breakProgressBar  gasProgressBar  [TYPE]  JPanel progressBarsPanel  [TYPE]  JLabel breakLabel  gasLabel  [TYPE]  boolean false  true  [TYPE]  int backgroundColor  dashboardBoundsX  dashboardBoundsY  height  meterHeight  meterWidth  progressBarsPanelHeight  progressBarsPanelWidth  progressBarsPanelX  progressBarsPanelY  rpmAngle  speedAngle  speedMeterX  speedMeterY  tachoMeterX  tachoMeterY  width 
[ADD]^gasProgressBar.setStringPainted ( true ) ;^86^87^^^^77^95^[ADD] breakLabel.setText ( "break pedal" ) ; gasProgressBar.setStringPainted ( true ) ;^[METHOD] initializeProgressBars [TYPE] void [PARAMETER] [CLASS] Dashboard   [TYPE]  JProgressBar breakProgressBar  gasProgressBar  [TYPE]  JPanel progressBarsPanel  [TYPE]  JLabel breakLabel  gasLabel  [TYPE]  boolean false  true  [TYPE]  int backgroundColor  dashboardBoundsX  dashboardBoundsY  height  meterHeight  meterWidth  progressBarsPanelHeight  progressBarsPanelWidth  progressBarsPanelX  progressBarsPanelY  rpmAngle  speedAngle  speedMeterX  speedMeterY  tachoMeterX  tachoMeterY  width 
[REPLACE]^gasProgressBar.setStringPainted ( false ) ;^87^^^^^77^95^[REPLACE] gasProgressBar.setStringPainted ( true ) ;^[METHOD] initializeProgressBars [TYPE] void [PARAMETER] [CLASS] Dashboard   [TYPE]  JProgressBar breakProgressBar  gasProgressBar  [TYPE]  JPanel progressBarsPanel  [TYPE]  JLabel breakLabel  gasLabel  [TYPE]  boolean false  true  [TYPE]  int backgroundColor  dashboardBoundsX  dashboardBoundsY  height  meterHeight  meterWidth  progressBarsPanelHeight  progressBarsPanelWidth  progressBarsPanelX  progressBarsPanelY  rpmAngle  speedAngle  speedMeterX  speedMeterY  tachoMeterX  tachoMeterY  width 
[REPLACE]^gasProgressBar.setStringPainted ( true )  ;^88^^^^^77^95^[REPLACE] breakProgressBar.setStringPainted ( true ) ;^[METHOD] initializeProgressBars [TYPE] void [PARAMETER] [CLASS] Dashboard   [TYPE]  JProgressBar breakProgressBar  gasProgressBar  [TYPE]  JPanel progressBarsPanel  [TYPE]  JLabel breakLabel  gasLabel  [TYPE]  boolean false  true  [TYPE]  int backgroundColor  dashboardBoundsX  dashboardBoundsY  height  meterHeight  meterWidth  progressBarsPanelHeight  progressBarsPanelWidth  progressBarsPanelX  progressBarsPanelY  rpmAngle  speedAngle  speedMeterX  speedMeterY  tachoMeterX  tachoMeterY  width 
[REPLACE]^progressBarsPanel.setLayout ( gasLabel ) ;^91^^^^^77^95^[REPLACE] progressBarsPanel.add ( gasLabel ) ;^[METHOD] initializeProgressBars [TYPE] void [PARAMETER] [CLASS] Dashboard   [TYPE]  JProgressBar breakProgressBar  gasProgressBar  [TYPE]  JPanel progressBarsPanel  [TYPE]  JLabel breakLabel  gasLabel  [TYPE]  boolean false  true  [TYPE]  int backgroundColor  dashboardBoundsX  dashboardBoundsY  height  meterHeight  meterWidth  progressBarsPanelHeight  progressBarsPanelWidth  progressBarsPanelX  progressBarsPanelY  rpmAngle  speedAngle  speedMeterX  speedMeterY  tachoMeterX  tachoMeterY  width 
[REPLACE]^progressBarsPanel.setLayout ( gasProgressBar ) ;^92^^^^^77^95^[REPLACE] progressBarsPanel.add ( gasProgressBar ) ;^[METHOD] initializeProgressBars [TYPE] void [PARAMETER] [CLASS] Dashboard   [TYPE]  JProgressBar breakProgressBar  gasProgressBar  [TYPE]  JPanel progressBarsPanel  [TYPE]  JLabel breakLabel  gasLabel  [TYPE]  boolean false  true  [TYPE]  int backgroundColor  dashboardBoundsX  dashboardBoundsY  height  meterHeight  meterWidth  progressBarsPanelHeight  progressBarsPanelWidth  progressBarsPanelX  progressBarsPanelY  rpmAngle  speedAngle  speedMeterX  speedMeterY  tachoMeterX  tachoMeterY  width 
[REPLACE]^progressBarsPanel.setLayout ( breakLabel ) ;^93^^^^^77^95^[REPLACE] progressBarsPanel.add ( breakLabel ) ;^[METHOD] initializeProgressBars [TYPE] void [PARAMETER] [CLASS] Dashboard   [TYPE]  JProgressBar breakProgressBar  gasProgressBar  [TYPE]  JPanel progressBarsPanel  [TYPE]  JLabel breakLabel  gasLabel  [TYPE]  boolean false  true  [TYPE]  int backgroundColor  dashboardBoundsX  dashboardBoundsY  height  meterHeight  meterWidth  progressBarsPanelHeight  progressBarsPanelWidth  progressBarsPanelX  progressBarsPanelY  rpmAngle  speedAngle  speedMeterX  speedMeterY  tachoMeterX  tachoMeterY  width 
[REPLACE]^progressBarsPanel.setLayout ( breakProgressBar ) ;^94^^^^^77^95^[REPLACE] progressBarsPanel.add ( breakProgressBar ) ;^[METHOD] initializeProgressBars [TYPE] void [PARAMETER] [CLASS] Dashboard   [TYPE]  JProgressBar breakProgressBar  gasProgressBar  [TYPE]  JPanel progressBarsPanel  [TYPE]  JLabel breakLabel  gasLabel  [TYPE]  boolean false  true  [TYPE]  int backgroundColor  dashboardBoundsX  dashboardBoundsY  height  meterHeight  meterWidth  progressBarsPanelHeight  progressBarsPanelWidth  progressBarsPanelX  progressBarsPanelY  rpmAngle  speedAngle  speedMeterX  speedMeterY  tachoMeterX  tachoMeterY  width 
