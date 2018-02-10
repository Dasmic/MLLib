using Dasmic.MLLib.UnitTest.Core;

namespace UnitTests_DecisionTree
{
    public class Base:UnitTestBase
    {
     
    
        /// <summary>
        /// R:
        /// input.data<-read.csv("C:\\Dasmic\\Development\\NET\\MachineLearning\\UnitTests\\FileOperationsLibTest\\bin\\Debug\\CSVFileWriteTest.csv")
        /// </summary>
        protected void initData_Mitchell_Book()
        {
            _attributeHeaders = new string[] {
                                     "Outlook",
                                    "Temperature",
                                    "Humidity",
                                    "Wind",
                                    "PlayTennis"};

            _trainingData = new double[5][];
            //0-Sunny,1-OverCast,2-Rain
            _trainingData[0] = new double[] { 0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2 };
            //0-Hot, 1-Mild, 2-Cool
            _trainingData[1] = new double[] { 0, 0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 1, 0, 1 };
            //0-Normal, 1-High
            _trainingData[2] = new double[] { 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1 };
            //0-Weak,1-Strong
            _trainingData[3] = new double[] { 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1 };
            //0-No,1-Yes
            _trainingData[4] = new double[] { 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0 };

            _indexTargetAttribute = 4;
        }

        /// <summary>
        /// R:
        /// input.data<-read.csv("C:\\Dasmic\\Development\\NET\\MachineLearning\\UnitTests\\FileOperationsLibTest\\bin\\Debug\\CSVFileWriteTest.csv")
        /// </summary>
        protected void initData_Mitchell_Book_Missingdata()
        {
            _attributeHeaders = new string[] {
                                     "Outlook",
                                    "Temperature",
                                    "Humidity",
                                    "Wind",
                                    "PlayTennis"};

            _trainingData = new double[5][];
            //0-Sunny,1-OverCast,2-Rain - Outlook
            _trainingData[0] = new double[] { 0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2 };
            //0-Hot, 1-Mild, 2-Cool - Temperature
            _trainingData[1] = new double[] { 0, 0, 999, 1, 2, 2, 2, 1, 2, 1, 1, 1, 0, 1 };
            //0-Normal, 1-High - Humidity
            _trainingData[2] = new double[] { 1, 1, 1, 999, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1 };
            //0-Weak,1-Strong - Wind
            _trainingData[3] = new double[] { 0, 1, 0, 999, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1 };
            //0-No,1-Yes - PlayTennis
            _trainingData[4] = new double[] { 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0 };
            _indexTargetAttribute = 4;
        }

        /// <summary>
        /// R:
        /// input.data<-read.csv("C:\\Dasmic\\Development\\NET\\MachineLearning\\UnitTests\\FileOperationsLibTest\\bin\\Debug\\CSVFileWriteTest.csv")
        /// </summary>
        protected void initData_Jason()
        {
            _attributeHeaders = new string[] {
                                     "X1",
                                    "X2",
                                    "Y"};
            _trainingData = new double[3][];            
            _trainingData[0] = new double[] {2.771244718,1.728571309,3.678319846,
                                        3.961043357,2.999208922,7.497545867,9.00220326,
                                        7.444542326,10.12493903,6.642287351 };
            _trainingData[1] = new double[] {1.784783929,1.169761413,2.81281357,
                                        2.61995032,2.209014212, 3.162953546,
                                        3.339047188,0.476683375,3.234550982,3.319983761 };
            _trainingData[2] = new double[] {0,0,0,0,0,1,1,1,1,1 };
            _indexTargetAttribute = 2;

            _validationData = new double[3][];
            _validationData[0] = new double[] {2.343875381,3.536904049,2.801395588,
                3.656342926,2.853194386,8.907647835,
                9.752464513,8.016361622,6.58490395,7.142525173};
            _validationData[1] = new double[] {2.051757824,3.032932531,2.786327755,
                                    2.581460765,1.052331062,3.730540859,3.740754624,
                                    3.013408249,2.436333477,3.650120799};
            _validationData[2] = new double[] {
                                0,0,0,0,0,1,1,1,1,1};

        }

        protected void initData_four_samples()
        {
            _attributeHeaders = new string[] {
                                     "X1",
                                    "X2",
                                    "Y"};
            _trainingData = new double[3][];
            _trainingData[0] = new double[] {
                        1.500958319,3.107545266,
                        4.090032824,5.38660215};

            _trainingData[1] = new double[] {
                        2.535482186,2.162569456,
                        3.123409313,2.109488166};

            _trainingData[2] = new double[] {
                        0,0,0,0};

            _indexTargetAttribute = 2;
        }

        protected void initData_same_four_samples()
        {
            _attributeHeaders = new string[] {
                                     "X1",
                                    "X2",
                                    "Y"};
            _trainingData = new double[3][];
            _trainingData[0] = new double[] {
                        1.500958319,1.500958319,
                        1.500958319,1.500958319};

            _trainingData[1] = new double[] {
                        2.535482186,2.535482186,
                        2.535482186,2.535482186};

            _trainingData[2] = new double[] {
                        0,0,0,0};

            _indexTargetAttribute = 2;
        }


        /// <summary>
        /// This test was taken from Bagged decision tree and 
        /// the CART algorithm failed in this
        /// </summary>
        protected void initData_special_no_splitting_possible()
        {
            _attributeHeaders = new string[] {
                                     "X1",
                                    "X2",
                                    "Y"};
            _trainingData = new double[3][];
            _trainingData[0] = new double[] {
                        5.38660215, 6.633669528,8.749958452,
                        8.749958452,4.589131161};

            _trainingData[1] = new double[] {
                        2.109488166, 2.749508563,2.676022211,
                        2.676022211,0.925340325};

            _trainingData[2] = new double[] {
                        0,1,1,1,1};

            _indexTargetAttribute = 2;
        }

    }
}
