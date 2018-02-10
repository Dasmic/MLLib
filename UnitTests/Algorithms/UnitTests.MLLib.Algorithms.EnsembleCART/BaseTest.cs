using Dasmic.MLLib.UnitTest.Core;


namespace UnitTests.MLLib.Algorithms.EnsembleCART
{
    public class BaseTest:UnitTestBase
    {
        protected void initData_Jason_Bagging()
        {
            _attributeHeaders = new string[] {
                                     "X1",
                                    "X2",
                                    "Y"};
            _trainingData = new double[3][];
            _trainingData[0] = new double[] {
                        2.309572387,1.500958319,3.107545266,
                        4.090032824,5.38660215,6.451823468,
                        6.633669528,8.749958452,4.589131161,
                        6.619322828};

            _trainingData[1] = new double[] {
                        1.168959634,2.535482186,2.162569456,
                        3.123409313,2.109488166,0.242952387,
                        2.749508563,2.676022211,0.925340325,3.831050828
                        };

            _trainingData[2] = new double[] {
                        0,0,0,0,0,1,1,1,1,1};

            _indexTargetAttribute = 2;            
        }


        protected void initData_Jason_3_features()
        {
            _attributeHeaders = new string[] {
                                     "X1",
                                    "X2",
                                    "X3",
                                    "Y"};
            _trainingData = new double[4][];
            _trainingData[0] = new double[] {
                        2.309572387,1.500958319,3.107545266,
                        4.090032824,5.38660215,6.451823468,
                        6.633669528,8.749958452,4.589131161,
                        6.619322828};

            _trainingData[1] = new double[] {
                        1.168959634,2.535482186,2.162569456,
                        3.123409313,2.109488166,0.242952387,
                        2.749508563,2.676022211,0.925340325,3.831050828
                        };

            _trainingData[2] = new double[] {
                        1.168959634,2.535482186,2.162569456,
                        3.123409313,2.109488166,0.242952387,
                        2.749508563,2.676022211,0.925340325,3.831050828
                        };

            _trainingData[3] = new double[] {
                        0,0,0,0,0,1,1,1,1,1};

            _indexTargetAttribute = 3;
        }

        protected void initData_Jason_AdaBoost()
        {
            _attributeHeaders = new string[] {
                                     "X1",
                                    "X2",
                                    "Y"};
            _trainingData = new double[3][];
            _trainingData[0] = new double[] {
                        3.64754035,2.612663842,2.363359679,
                        4.932600453,3.776154753,8.673960793,
                        5.861599451,8.984677361,7.467380954,
                        4.436284412};

            _trainingData[1] = new double[] {
                    2.996793259,4.459457779,1.506982189,
                    1.299008795,3.157451378,2.122873405,
                    0.003512817,1.768161009,0.187045945,
                    0.862698005};

            _trainingData[2] = new double[] {
                        0,0,0,0,0,1,1,1,1,1};

            _indexTargetAttribute = 2;
        }
    }
}
