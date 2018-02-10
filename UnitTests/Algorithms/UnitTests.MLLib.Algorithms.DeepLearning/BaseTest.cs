using Dasmic.MLLib.UnitTest.Core;

namespace UnitTests.MLLib.Algorithms.DeepLearning
{
    public class BaseTest : UnitTestBase
    {
        /// <summary>
        //Initialize data from Jason's book
        /// R:
        /// X = list(c(1, 2, 4, 3, 5))
        /// Y = list(c(1, 3, 3, 2, 5))
        /// </summary>
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

        /// <summary>
        /// Load the Pythagoras Data set
        /// </summary>
        protected void Init_dataset_pythagoras()
        {
            LoadFromDataSet(EnumDataSets.Pythagoras, -1);
        }
    }
}
