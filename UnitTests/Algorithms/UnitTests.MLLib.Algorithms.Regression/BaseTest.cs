using Dasmic.MLLib.UnitTest.Core;

namespace UnitTests_Regression
{
    public class BaseTest : UnitTestBase
    {
        /// <summary>
        /// Dataset taken from:
        /// Data taken from:
        /// 
        /// http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis
        /// </summary>
        protected void InitData_dataset_regression_web_example()
        {
            _attributeHeaders = new string[] {
                                     "Factor1",
                                     "Factor2",
                                    "Yield"};
            _indexTargetAttribute = 2;

            _trainingData = new double[3][];
            _trainingData[0] = new double[] { 41.9, 43.4,43.9,44.5,47.3,47.5,47.9,50.2,52.8,53.2,56.7,57.0,63.5,65.3,71.1,77.0,77.8};
            _trainingData[1] = new double[] { 29.1,29.3,29.5,29.7,29.9,30.3,30.5,30.7,30.8,30.9,31.5,31.7,31.9,32.0,32.1,32.5,32.9};
            _trainingData[2] = new double[] { 251.3,251.3,248.3,267.5,273.0,276.5,270.3,274.9,285.0,290.0,297.0,302.5,304.5,309.3,321.7,330.7,349.0 };
        }

        //Initialize data from Jason's book
        protected void Init_dataset_jason_linear_regression()
        {
            _attributeHeaders = new string[] {
                                     "X",
                                    "Y"};
            _indexTargetAttribute = 1;

            _trainingData = new double[2][];
            _trainingData[0] = new double[] { 1, 2, 4, 3, 5 };
            _trainingData[1] = new double[] { 1, 3, 3, 2, 5 };
        }

        protected void Init_dataset_power()
        {
            _attributeHeaders = new string[] {
                                     "X",
                                    "Y"};
            _indexTargetAttribute = 1;

            _trainingData = new double[2][];
            _trainingData[0] = new double[] { 1, 2, 3, 4, 5 };
            _trainingData[1] = new double[] { 1, 4, 9, 16, 25 };
        }

        /// <summary>
        /// Load the Pythagoras Data set
        /// </summary>
        protected void Init_dataset_pythagoras()
        {
            LoadFromDataSet(EnumDataSets.Pythagoras,-1);           
        }

    }
}
