[REPLACE]^plot.setCircular ( false ) ;^72^^^^^71^74^[REPLACE] super ( title ) ;^[METHOD] <init> [TYPE] String) [PARAMETER] String title [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  String title 
[REMOVE]^plot.setCircular ( false ) ;^72^^^^^71^74^[REMOVE] ^[METHOD] <init> [TYPE] String) [PARAMETER] String title [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  String title 
[REPLACE]^JFreeChart chart = createChart ( createDataset (  )  ) ;^73^^^^^71^74^[REPLACE] setContentPane ( createDemoPanel (  )  ) ;^[METHOD] <init> [TYPE] String) [PARAMETER] String title [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  String title 
[ADD]^^73^^^^^71^74^[ADD] setContentPane ( createDemoPanel (  )  ) ;^[METHOD] <init> [TYPE] String) [PARAMETER] String title [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  String title 
[REPLACE]^PiePlot plot =  ( PiePlot )  chart.getPlot (  ) ;^82^^^^^81^90^[REPLACE] DefaultPieDataset dataset = new DefaultPieDataset (  ) ;^[METHOD] createDataset [TYPE] PieDataset [PARAMETER] [CLASS] PieChartDemo1   [TYPE]  DefaultPieDataset dataset  [TYPE]  boolean false  true 
[ADD]^^82^83^^^^81^90^[ADD] DefaultPieDataset dataset = new DefaultPieDataset (  ) ; dataset.setValue ( "One", new Double ( 43.2 )  ) ;^[METHOD] createDataset [TYPE] PieDataset [PARAMETER] [CLASS] PieChartDemo1   [TYPE]  DefaultPieDataset dataset  [TYPE]  boolean false  true 
[REPLACE]^plot.setSectionOutlinesVisible ( false ) ;^83^^^^^81^90^[REPLACE] dataset.setValue ( "One", new Double ( 43.2 )  ) ;^[METHOD] createDataset [TYPE] PieDataset [PARAMETER] [CLASS] PieChartDemo1   [TYPE]  DefaultPieDataset dataset  [TYPE]  boolean false  true 
[REPLACE]^dataset .DefaultPieDataset (  )  ;^84^^^^^81^90^[REPLACE] dataset.setValue ( "Two", new Double ( 10.0 )  ) ;^[METHOD] createDataset [TYPE] PieDataset [PARAMETER] [CLASS] PieChartDemo1   [TYPE]  DefaultPieDataset dataset  [TYPE]  boolean false  true 
[REPLACE]^setContentPane ( createDemoPanel (  )  ) ;^85^^^^^81^90^[REPLACE] dataset.setValue ( "Three", new Double ( 27.5 )  ) ;^[METHOD] createDataset [TYPE] PieDataset [PARAMETER] [CLASS] PieChartDemo1   [TYPE]  DefaultPieDataset dataset  [TYPE]  boolean false  true 
[REPLACE]^dataset .DefaultPieDataset (  )  ;^86^^^^^81^90^[REPLACE] dataset.setValue ( "Four", new Double ( 17.5 )  ) ;^[METHOD] createDataset [TYPE] PieDataset [PARAMETER] [CLASS] PieChartDemo1   [TYPE]  DefaultPieDataset dataset  [TYPE]  boolean false  true 
[REPLACE]^plot.setSectionOutlinesVisible ( false ) ;^87^^^^^81^90^[REPLACE] dataset.setValue ( "Five", new Double ( 11.0 )  ) ;^[METHOD] createDataset [TYPE] PieDataset [PARAMETER] [CLASS] PieChartDemo1   [TYPE]  DefaultPieDataset dataset  [TYPE]  boolean false  true 
[ADD]^^87^88^^^^81^90^[ADD] dataset.setValue ( "Five", new Double ( 11.0 )  ) ; dataset.setValue ( "Six", new Double ( 19.4 )  ) ;^[METHOD] createDataset [TYPE] PieDataset [PARAMETER] [CLASS] PieChartDemo1   [TYPE]  DefaultPieDataset dataset  [TYPE]  boolean false  true 
[REPLACE]^plot.setSectionOutlinesVisible ( false ) ;^88^^^^^81^90^[REPLACE] dataset.setValue ( "Six", new Double ( 19.4 )  ) ;^[METHOD] createDataset [TYPE] PieDataset [PARAMETER] [CLASS] PieChartDemo1   [TYPE]  DefaultPieDataset dataset  [TYPE]  boolean false  true 
[REPLACE]^return new ChartPanel ( chart ) ;^89^^^^^81^90^[REPLACE] return dataset;^[METHOD] createDataset [TYPE] PieDataset [PARAMETER] [CLASS] PieChartDemo1   [TYPE]  DefaultPieDataset dataset  [TYPE]  boolean false  true 
[REPLACE]^PieChartDemo1 demo = new PieChartDemo1 ( "Pie Chart Demo 1" ) ;^109^^^^^99^117^[REPLACE] PiePlot plot =  ( PiePlot )  chart.getPlot (  ) ;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] PieDataset dataset [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  PieDataset dataset  [TYPE]  PiePlot plot  [TYPE]  JFreeChart chart 
[ADD]^^109^^^^^99^117^[ADD] PiePlot plot =  ( PiePlot )  chart.getPlot (  ) ;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] PieDataset dataset [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  PieDataset dataset  [TYPE]  PiePlot plot  [TYPE]  JFreeChart chart 
[REPLACE]^plot.setCircular ( true ) ;^110^^^^^99^117^[REPLACE] plot.setSectionOutlinesVisible ( false ) ;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] PieDataset dataset [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  PieDataset dataset  [TYPE]  PiePlot plot  [TYPE]  JFreeChart chart 
[REPLACE]^plot.setNoDataMessage ( new Font ( "SansSerif", Font.PLAIN, 12 )  ) ;^111^^^^^99^117^[REPLACE] plot.setLabelFont ( new Font ( "SansSerif", Font.PLAIN, 12 )  ) ;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] PieDataset dataset [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  PieDataset dataset  [TYPE]  PiePlot plot  [TYPE]  JFreeChart chart 
[REPLACE]^plot.setLabelFont ( "No data available" ) ;^112^^^^^99^117^[REPLACE] plot.setNoDataMessage ( "No data available" ) ;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] PieDataset dataset [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  PieDataset dataset  [TYPE]  PiePlot plot  [TYPE]  JFreeChart chart 
[REMOVE]^plot.setLabelGap ( 0.02 ) ;^112^^^^^99^117^[REMOVE] ^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] PieDataset dataset [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  PieDataset dataset  [TYPE]  PiePlot plot  [TYPE]  JFreeChart chart 
[REPLACE]^plot.setCircular ( true ) ;^113^^^^^99^117^[REPLACE] plot.setCircular ( false ) ;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] PieDataset dataset [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  PieDataset dataset  [TYPE]  PiePlot plot  [TYPE]  JFreeChart chart 
[REPLACE]^plot.setLabelGap ( 0.0d ) ;^114^^^^^99^117^[REPLACE] plot.setLabelGap ( 0.02 ) ;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] PieDataset dataset [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  PieDataset dataset  [TYPE]  PiePlot plot  [TYPE]  JFreeChart chart 
[REPLACE]^return new ChartPanel ( chart ) ;^115^^^^^99^117^[REPLACE] return chart;^[METHOD] createChart [TYPE] JFreeChart [PARAMETER] PieDataset dataset [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  PieDataset dataset  [TYPE]  PiePlot plot  [TYPE]  JFreeChart chart 
[REPLACE]^PiePlot plot =  ( PiePlot )  chart.getPlot (  ) ;^125^^^^^124^127^[REPLACE] JFreeChart chart = createChart ( createDataset (  )  ) ;^[METHOD] createDemoPanel [TYPE] JPanel [PARAMETER] [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  JFreeChart chart 
[REPLACE]^return chart;^126^^^^^124^127^[REPLACE] return new ChartPanel ( chart ) ;^[METHOD] createDemoPanel [TYPE] JPanel [PARAMETER] [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  JFreeChart chart 
[REPLACE]^PiePlot plot =  ( PiePlot )  chart.getPlot (  ) ;^144^^^^^134^148^[REPLACE] PieChartDemo1 demo = new PieChartDemo1 ( "Pie Chart Demo 1" ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  String[] args  [TYPE]  PieChartDemo1 demo 
[ADD]^^144^^^^^134^148^[ADD] PieChartDemo1 demo = new PieChartDemo1 ( "Pie Chart Demo 1" ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  String[] args  [TYPE]  PieChartDemo1 demo 
[REPLACE]^demo .setContentPane ( null )  ;^145^^^^^134^148^[REPLACE] demo.pack (  ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  String[] args  [TYPE]  PieChartDemo1 demo 
[REPLACE]^demo   ;^146^^^^^134^148^[REPLACE] RefineryUtilities.centerFrameOnScreen ( demo ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  String[] args  [TYPE]  PieChartDemo1 demo 
[REMOVE]^setContentPane ( createDemoPanel (  )  ) ;^146^^^^^134^148^[REMOVE] ^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  String[] args  [TYPE]  PieChartDemo1 demo 
[REPLACE]^demo.setContentPane ( false ) ;^147^^^^^134^148^[REPLACE] demo.setVisible ( true ) ;^[METHOD] main [TYPE] void [PARAMETER] String[] args [CLASS] PieChartDemo1   [TYPE]  boolean false  true  [TYPE]  String[] args  [TYPE]  PieChartDemo1 demo 